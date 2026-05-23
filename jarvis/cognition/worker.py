from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any

from jarvis.cognition.adapters import (
    CancellableCognitionAdapter,
    CognitionAdapter,
    CognitionAdapterResult,
    CognitionAdapterSnapshot,
)
from jarvis.cognition.models import (
    CognitionFailure,
    CognitionFailureKind,
    CognitionRequest,
    CognitionResponse,
)
from jarvis.cognition.state_store import (
    CognitionStateStore,
    CognitionStateStoreSnapshot,
)
from jarvis.runtime.observability.structured_logger import get_logger


@dataclass(frozen=True, slots=True)
class CognitionWorkerConfig:
    """
    Runtime configuration for CognitionWorker.
    """

    name: str = "cognition_worker"
    fail_fast_on_adapter_error: bool = False

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")


@dataclass(frozen=True, slots=True)
class CognitionWorkerResult:
    """
    Result of one direct cognition worker execution.
    """

    request_id: str
    accepted: bool
    response: CognitionResponse | None = None
    failure: CognitionFailure | None = None
    reason: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.accepted and self.response is not None

    @property
    def failed(self) -> bool:
        return self.accepted and self.failure is not None

    @property
    def rejected(self) -> bool:
        return not self.accepted


@dataclass(frozen=True, slots=True)
class CognitionWorkerSnapshot:
    """
    Observable worker snapshot.

    This combines worker status, state-store status, and adapter diagnostics.
    """

    name: str
    started: bool
    processed_count: int
    success_count: int
    failure_count: int
    rejected_count: int
    cancel_requested_count: int
    last_request_id: str | None
    last_error: str | None
    state: CognitionStateStoreSnapshot
    adapter: CognitionAdapterSnapshot


class CognitionWorker:
    """
    Brain runtime worker.

    Responsibilities:
    - accept typed CognitionRequest objects
    - guard execution through CognitionStateStore
    - call the configured CognitionAdapter
    - record completion/failure/cancellation state
    - expose observable diagnostics

    Non-responsibilities:
    - no microphone/STT/TTS knowledge
    - no direct LLM implementation knowledge
    - no memory retrieval yet
    - no tool execution
    - no dialogue bridge wiring yet

    Step 6/7 will connect this worker to EventBus bridges.
    """

    def __init__(
        self,
        *,
        adapter: CognitionAdapter,
        state_store: CognitionStateStore | None = None,
        config: CognitionWorkerConfig | None = None,
    ) -> None:
        self._config = config or CognitionWorkerConfig()
        self._config.validate()

        self._adapter = adapter
        self._state_store = state_store or CognitionStateStore()
        self._lock = RLock()
        self._logger = get_logger("cognition.worker")

        self._started = False
        self._processed_count = 0
        self._success_count = 0
        self._failure_count = 0
        self._rejected_count = 0
        self._cancel_requested_count = 0
        self._last_request_id: str | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def started(self) -> bool:
        with self._lock:
            return self._started

    @property
    def state_store(self) -> CognitionStateStore:
        return self._state_store

    @property
    def adapter(self) -> CognitionAdapter:
        return self._adapter

    def on_start(self) -> None:
        """
        Start the worker.

        This is intentionally idempotent. Runtime boot can safely call it once,
        tests can call it multiple times, and future kernel wiring remains safe.
        """

        with self._lock:
            if self._started:
                return

            self._started = True
            self._logger.info(
                "cognition_worker_started",
                worker=self.name,
                adapter=self._adapter.name,
            )

    def on_stop(self) -> None:
        """
        Stop the worker.

        Stopping does not reset counters. It only prevents new direct processing.
        """

        with self._lock:
            if not self._started:
                return

            self._started = False
            self._logger.info("cognition_worker_stopped", worker=self.name)

    def process_request(
        self,
        request: CognitionRequest,
    ) -> CognitionWorkerResult:
        """
        Process one cognition request through the adapter.

        This is the first true brain runtime loop:
        request → state start → adapter → state complete/fail.
        """

        with self._lock:
            if not self._started:
                self._rejected_count += 1
                self._last_request_id = request.request_id
                self._last_error = "worker is not started"

                return CognitionWorkerResult(
                    request_id=request.request_id,
                    accepted=False,
                    reason="worker is not started",
                )

            self._processed_count += 1
            self._last_request_id = request.request_id
            self._last_error = None

        start_transition = self._state_store.start_request(request)

        if not start_transition.accepted:
            with self._lock:
                self._rejected_count += 1
                self._last_error = start_transition.reason

            return CognitionWorkerResult(
                request_id=request.request_id,
                accepted=False,
                reason=start_transition.reason,
            )

        self._logger.info(
            "cognition_worker_processing",
            worker=self.name,
            request_id=request.request_id,
            adapter=self._adapter.name,
        )

        try:
            adapter_result = self._adapter.generate(request)

        except Exception as exc:
            return self._handle_adapter_exception(
                request=request,
                exc=exc,
            )

        return self._handle_adapter_result(
            request=request,
            adapter_result=adapter_result,
        )

    def request_cancel(
        self,
        *,
        request_id: str | None = None,
        reason: str | None = None,
    ) -> bool:
        """
        Request cancellation for the active cognition request.

        The store always records the cancellation intent. If the adapter supports
        backend cancellation, the worker forwards the request.
        """

        transition = self._state_store.request_cancel(
            request_id=request_id,
            reason=reason,
        )

        if not transition.accepted:
            with self._lock:
                self._last_error = transition.reason

            return False

        with self._lock:
            self._cancel_requested_count += 1

        if isinstance(self._adapter, CancellableCognitionAdapter):
            active_request_id = (
                request_id
                or self._state_store.snapshot().active_request_id
            )

            if active_request_id is not None:
                self._adapter.cancel(
                    request_id=active_request_id,
                    reason=reason,
                )

        self._logger.info(
            "cognition_worker_cancel_requested",
            worker=self.name,
            request_id=request_id,
            reason=reason,
        )

        return True

    def confirm_cancelled(
        self,
        *,
        request_id: str | None = None,
        reason: str | None = None,
    ) -> bool:
        """
        Confirm cancellation after adapter/runtime has stopped generation.
        """

        transition = self._state_store.cancel_request(
            request_id=request_id,
            reason=reason,
        )

        if not transition.accepted:
            with self._lock:
                self._last_error = transition.reason

            return False

        self._logger.info(
            "cognition_worker_cancelled",
            worker=self.name,
            request_id=request_id,
            reason=reason,
        )

        return True

    def reset(self) -> None:
        """
        Reset worker counters and state store.

        The adapter is not reset here because not every real adapter will support
        reset. Fake adapters can still be reset directly in tests.
        """

        with self._lock:
            self._processed_count = 0
            self._success_count = 0
            self._failure_count = 0
            self._rejected_count = 0
            self._cancel_requested_count = 0
            self._last_request_id = None
            self._last_error = None

        self._state_store.reset()

        self._logger.info("cognition_worker_reset", worker=self.name)

    def snapshot(self) -> CognitionWorkerSnapshot:
        """
        Return a full worker diagnostic snapshot.
        """

        with self._lock:
            return CognitionWorkerSnapshot(
                name=self.name,
                started=self._started,
                processed_count=self._processed_count,
                success_count=self._success_count,
                failure_count=self._failure_count,
                rejected_count=self._rejected_count,
                cancel_requested_count=self._cancel_requested_count,
                last_request_id=self._last_request_id,
                last_error=self._last_error,
                state=self._state_store.snapshot(),
                adapter=self._adapter.snapshot(),
            )

    def _handle_adapter_result(
        self,
        *,
        request: CognitionRequest,
        adapter_result: CognitionAdapterResult,
    ) -> CognitionWorkerResult:
        if adapter_result.request_id != request.request_id:
            failure = self._make_worker_failure(
                request=request,
                message="adapter result request_id does not match request",
                metadata={
                    "adapter_result_request_id": adapter_result.request_id,
                },
            )
            return self._fail_request(
                request=request,
                failure=failure,
            )

        if adapter_result.response is not None:
            complete_transition = self._state_store.complete_request(
                adapter_result.response
            )

            if not complete_transition.accepted:
                failure = self._make_worker_failure(
                    request=request,
                    message=(
                        complete_transition.reason
                        or "state store rejected completion"
                    ),
                )
                return self._fail_request(
                    request=request,
                    failure=failure,
                )

            with self._lock:
                self._success_count += 1
                self._last_error = None

            self._logger.info(
                "cognition_worker_completed",
                worker=self.name,
                request_id=request.request_id,
                response_id=adapter_result.response.response_id,
            )

            return CognitionWorkerResult(
                request_id=request.request_id,
                accepted=True,
                response=adapter_result.response,
            )

        if adapter_result.failure is not None:
            return self._fail_request(
                request=request,
                failure=adapter_result.failure,
            )

        failure = self._make_worker_failure(
            request=request,
            message="adapter result contained no response or failure",
        )

        return self._fail_request(
            request=request,
            failure=failure,
        )

    def _handle_adapter_exception(
        self,
        *,
        request: CognitionRequest,
        exc: Exception,
    ) -> CognitionWorkerResult:
        failure = self._make_worker_failure(
            request=request,
            message=f"{type(exc).__name__}: {exc}",
            metadata={
                "exception_type": type(exc).__name__,
            },
        )

        result = self._fail_request(
            request=request,
            failure=failure,
        )

        if self._config.fail_fast_on_adapter_error:
            raise exc

        return result

    def _fail_request(
        self,
        *,
        request: CognitionRequest,
        failure: CognitionFailure,
    ) -> CognitionWorkerResult:
        fail_transition = self._state_store.fail_request(failure)

        with self._lock:
            self._failure_count += 1
            self._last_error = failure.message

        if not fail_transition.accepted:
            self._logger.error(
                "cognition_worker_failure_state_rejected",
                worker=self.name,
                request_id=request.request_id,
                failure_id=failure.failure_id,
                failure_message=failure.message,
                reason=fail_transition.reason,
            )

        self._logger.error(
            "cognition_worker_failed",
            worker=self.name,
            request_id=request.request_id,
            failure_id=failure.failure_id,
            failure_kind=failure.kind.value,
            failure_message=failure.message,
        )

        return CognitionWorkerResult(
            request_id=request.request_id,
            accepted=True,
            failure=failure,
        )

    def _make_worker_failure(
        self,
        *,
        request: CognitionRequest,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> CognitionFailure:
        return CognitionFailure(
            request_id=request.request_id,
            kind=CognitionFailureKind.ADAPTER_ERROR,
            message=message,
            recoverable=True,
            metadata={
                "worker": self.name,
                "adapter": self._adapter.name,
                **(metadata or {}),
            },
        )