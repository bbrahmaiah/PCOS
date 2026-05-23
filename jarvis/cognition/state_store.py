from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from threading import RLock
from typing import Any

from jarvis.cognition.models import (
    CognitionFailure,
    CognitionRequest,
    CognitionResponse,
    CognitionSnapshot,
    CognitionToken,
)
from jarvis.runtime.observability.structured_logger import get_logger


class CognitionRunState(StrEnum):
    """
    Internal state for the cognition runtime.

    IDLE:
        No active cognition request.

    THINKING:
        A request is active and non-streaming reasoning is running.

    STREAMING:
        A request is active and streaming tokens are being produced.

    CANCELLING:
        Cancellation was requested and the active request should stop.

    COMPLETED:
        Last request completed successfully.

    FAILED:
        Last request failed.

    CANCELLED:
        Last request was cancelled.
    """

    IDLE = "idle"
    THINKING = "thinking"
    STREAMING = "streaming"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class CognitionTransitionResult:
    """
    Result of a state transition attempt.
    """

    accepted: bool
    previous_state: CognitionRunState
    current_state: CognitionRunState
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class CognitionStateStoreSnapshot:
    """
    Full state-store snapshot for tests and runtime diagnostics.
    """

    state: CognitionRunState
    active_request_id: str | None
    active_turn_id: str | None
    active_correlation_id: str | None
    active_text: str | None
    streaming: bool
    cancelling: bool
    token_count: int
    started_count: int
    completed_count: int
    failed_count: int
    cancelled_count: int
    last_response: CognitionResponse | None
    last_failure: CognitionFailure | None
    last_token: CognitionToken | None
    last_error: str | None
    last_started_at: datetime | None
    last_finished_at: datetime | None
    updated_at: datetime


def utc_now() -> datetime:
    return datetime.now(UTC)


class CognitionStateStore:
    """
    Thread-safe state store for the cognition runtime.

    Responsibilities:
    - track one active cognition request
    - guard state transitions
    - expose immutable snapshots
    - track cancellation
    - track response/failure/token counters

    Non-responsibilities:
    - no LLM calls
    - no event publishing
    - no memory retrieval
    - no tool execution
    """

    def __init__(self, *, name: str = "cognition_state_store") -> None:
        if not name.strip():
            raise ValueError("name cannot be empty.")

        self._name = name
        self._lock = RLock()
        self._logger = get_logger("cognition.state_store")

        self._state = CognitionRunState.IDLE
        self._active_request: CognitionRequest | None = None
        self._last_response: CognitionResponse | None = None
        self._last_failure: CognitionFailure | None = None
        self._last_token: CognitionToken | None = None
        self._last_error: str | None = None

        self._started_count = 0
        self._completed_count = 0
        self._failed_count = 0
        self._cancelled_count = 0
        self._token_count = 0

        self._last_started_at: datetime | None = None
        self._last_finished_at: datetime | None = None
        self._updated_at = utc_now()

    @property
    def name(self) -> str:
        return self._name

    def start_request(
        self,
        request: CognitionRequest,
    ) -> CognitionTransitionResult:
        """
        Start a cognition request.

        The store allows only one active request. Later orchestration can route
        multiple requests to separate workers, but this store remains strict and
        safe for one active brain turn.
        """

        with self._lock:
            previous = self._state

            if self._has_active_request():
                return self._reject(
                    previous_state=previous,
                    reason="another cognition request is already active",
                )

            self._state = (
                CognitionRunState.STREAMING
                if request.policy.streaming_enabled
                else CognitionRunState.THINKING
            )
            self._active_request = request
            self._last_response = None
            self._last_failure = None
            self._last_token = None
            self._last_error = None
            self._token_count = 0
            self._started_count += 1
            self._last_started_at = utc_now()
            self._last_finished_at = None
            self._touch()

            self._logger.info(
                "cognition_request_started",
                store=self._name,
                request_id=request.request_id,
                turn_id=request.turn_id,
                state=self._state.value,
                streaming=request.policy.streaming_enabled,
            )

            return self._accept(previous)

    def mark_streaming_started(self) -> CognitionTransitionResult:
        """
        Move an active request into streaming state.
        """

        with self._lock:
            previous = self._state

            if self._active_request is None:
                return self._reject(
                    previous_state=previous,
                    reason="no active cognition request",
                )

            if self._state == CognitionRunState.CANCELLING:
                return self._reject(
                    previous_state=previous,
                    reason="cannot start streaming while cancelling",
                )

            if self._state not in {
                CognitionRunState.THINKING,
                CognitionRunState.STREAMING,
            }:
                return self._reject(
                    previous_state=previous,
                    reason=f"cannot stream from state {self._state.value}",
                )

            self._state = CognitionRunState.STREAMING
            self._touch()

            return self._accept(previous)

    def record_token(
        self,
        token: CognitionToken,
    ) -> CognitionTransitionResult:
        """
        Record one streamed token for the active request.
        """

        with self._lock:
            previous = self._state

            if self._active_request is None:
                return self._reject(
                    previous_state=previous,
                    reason="no active cognition request",
                )

            if token.request_id != self._active_request.request_id:
                return self._reject(
                    previous_state=previous,
                    reason="token request_id does not match active request",
                )

            if self._state == CognitionRunState.CANCELLING:
                return self._reject(
                    previous_state=previous,
                    reason="cannot record token while cancelling",
                )

            if self._state not in {
                CognitionRunState.THINKING,
                CognitionRunState.STREAMING,
            }:
                return self._reject(
                    previous_state=previous,
                    reason=f"cannot record token from state {self._state.value}",
                )

            self._state = CognitionRunState.STREAMING
            self._last_token = token
            self._token_count += 1
            self._touch()

            return self._accept(previous)

    def request_cancel(
        self,
        *,
        request_id: str | None = None,
        reason: str | None = None,
    ) -> CognitionTransitionResult:
        """
        Mark the active request as cancelling.
        """

        with self._lock:
            previous = self._state

            if self._active_request is None:
                return self._reject(
                    previous_state=previous,
                    reason="no active cognition request",
                )

            if request_id is not None and request_id != self._active_request.request_id:
                return self._reject(
                    previous_state=previous,
                    reason="cancel request_id does not match active request",
                )

            if self._state in {
                CognitionRunState.COMPLETED,
                CognitionRunState.FAILED,
                CognitionRunState.CANCELLED,
                CognitionRunState.IDLE,
            }:
                return self._reject(
                    previous_state=previous,
                    reason=f"cannot cancel from state {self._state.value}",
                )

            self._state = CognitionRunState.CANCELLING
            self._last_error = reason
            self._touch()

            self._logger.info(
                "cognition_cancel_requested",
                store=self._name,
                request_id=self._active_request.request_id,
                reason=reason,
            )

            return self._accept(previous)

    def complete_request(
        self,
        response: CognitionResponse,
    ) -> CognitionTransitionResult:
        """
        Complete the active request with a final response.
        """

        with self._lock:
            previous = self._state

            if self._active_request is None:
                return self._reject(
                    previous_state=previous,
                    reason="no active cognition request",
                )

            if response.request_id != self._active_request.request_id:
                return self._reject(
                    previous_state=previous,
                    reason="response request_id does not match active request",
                )

            if self._state == CognitionRunState.CANCELLING:
                return self._reject(
                    previous_state=previous,
                    reason="cannot complete while cancelling",
                )

            self._state = CognitionRunState.COMPLETED
            self._last_response = response
            self._last_failure = None
            self._last_error = None
            self._completed_count += 1
            self._last_finished_at = utc_now()
            self._clear_active_request()
            self._touch()

            self._logger.info(
                "cognition_request_completed",
                store=self._name,
                request_id=response.request_id,
                response_id=response.response_id,
            )

            return self._accept(previous)

    def fail_request(
        self,
        failure: CognitionFailure,
    ) -> CognitionTransitionResult:
        """
        Fail the active request with a typed failure.
        """

        with self._lock:
            previous = self._state

            if self._active_request is None:
                return self._reject(
                    previous_state=previous,
                    reason="no active cognition request",
                )

            if failure.request_id != self._active_request.request_id:
                return self._reject(
                    previous_state=previous,
                    reason="failure request_id does not match active request",
                )

            self._state = CognitionRunState.FAILED
            self._last_failure = failure
            self._last_response = None
            self._last_error = failure.message
            self._failed_count += 1
            self._last_finished_at = utc_now()
            self._clear_active_request()
            self._touch()

            self._logger.error(
                "cognition_request_failed",
                store=self._name,
                request_id=failure.request_id,
                failure_id=failure.failure_id,
                failure_kind=failure.kind.value,
                failure_message=failure.message,
            )
            return self._accept(previous)

    def cancel_request(
        self,
        *,
        request_id: str | None = None,
        reason: str | None = None,
    ) -> CognitionTransitionResult:
        """
        Confirm cancellation of the active request.
        """

        with self._lock:
            previous = self._state

            if self._active_request is None:
                return self._reject(
                    previous_state=previous,
                    reason="no active cognition request",
                )

            if request_id is not None and request_id != self._active_request.request_id:
                return self._reject(
                    previous_state=previous,
                    reason="cancel request_id does not match active request",
                )

            self._state = CognitionRunState.CANCELLED
            self._last_error = reason
            self._cancelled_count += 1
            self._last_finished_at = utc_now()
            self._clear_active_request()
            self._touch()

            self._logger.info(
                "cognition_request_cancelled",
                store=self._name,
                request_id=request_id,
                reason=reason,
            )

            return self._accept(previous)

    def reset(self) -> None:
        """
        Reset runtime state and counters.
        """

        with self._lock:
            self._state = CognitionRunState.IDLE
            self._active_request = None
            self._last_response = None
            self._last_failure = None
            self._last_token = None
            self._last_error = None
            self._started_count = 0
            self._completed_count = 0
            self._failed_count = 0
            self._cancelled_count = 0
            self._token_count = 0
            self._last_started_at = None
            self._last_finished_at = None
            self._touch()

            self._logger.info("cognition_state_store_reset", store=self._name)

    def snapshot(self) -> CognitionStateStoreSnapshot:
        """
        Return a full immutable state-store snapshot.
        """

        with self._lock:
            active_request = self._active_request

            return CognitionStateStoreSnapshot(
                state=self._state,
                active_request_id=(
                    active_request.request_id if active_request else None
                ),
                active_turn_id=active_request.turn_id if active_request else None,
                active_correlation_id=(
                    active_request.correlation_id if active_request else None
                ),
                active_text=active_request.text if active_request else None,
                streaming=self._state == CognitionRunState.STREAMING,
                cancelling=self._state == CognitionRunState.CANCELLING,
                token_count=self._token_count,
                started_count=self._started_count,
                completed_count=self._completed_count,
                failed_count=self._failed_count,
                cancelled_count=self._cancelled_count,
                last_response=self._last_response,
                last_failure=self._last_failure,
                last_token=self._last_token,
                last_error=self._last_error,
                last_started_at=self._last_started_at,
                last_finished_at=self._last_finished_at,
                updated_at=self._updated_at,
            )

    def cognition_snapshot(self) -> CognitionSnapshot:
        """
        Return the public lightweight cognition snapshot model from Step 1.
        """

        full = self.snapshot()

        return CognitionSnapshot(
            active_request_id=full.active_request_id,
            active_turn_id=full.active_turn_id,
            running=full.state
            in {
                CognitionRunState.THINKING,
                CognitionRunState.STREAMING,
                CognitionRunState.CANCELLING,
            },
            streaming=full.streaming,
            cancelling=full.cancelling,
            completed_count=full.completed_count,
            failed_count=full.failed_count,
            cancelled_count=full.cancelled_count,
            last_response_id=(
                full.last_response.response_id if full.last_response else None
            ),
            last_error=full.last_error,
            metadata={
                "state": full.state.value,
                "token_count": full.token_count,
                "started_count": full.started_count,
            },
        )

    def _has_active_request(self) -> bool:
        return self._active_request is not None and self._state in {
            CognitionRunState.THINKING,
            CognitionRunState.STREAMING,
            CognitionRunState.CANCELLING,
        }

    def _clear_active_request(self) -> None:
        self._active_request = None

    def _touch(self) -> None:
        self._updated_at = utc_now()

    def _accept(
        self,
        previous_state: CognitionRunState,
    ) -> CognitionTransitionResult:
        return CognitionTransitionResult(
            accepted=True,
            previous_state=previous_state,
            current_state=self._state,
        )

    def _reject(
        self,
        *,
        previous_state: CognitionRunState,
        reason: str,
    ) -> CognitionTransitionResult:
        self._logger.info(
            "cognition_state_transition_rejected",
            store=self._name,
            state=self._state.value,
            reason=reason,
        )

        return CognitionTransitionResult(
            accepted=False,
            previous_state=previous_state,
            current_state=self._state,
            reason=reason,
        )


def metadata_copy(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Return a shallow metadata copy.

    Kept as a small utility for future state-store extensions where we need to
    avoid sharing caller-owned dictionaries.
    """

    return dict(metadata)