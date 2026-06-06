from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from threading import RLock
from typing import Any

from jarvis.cognition.adapters import (
    CognitionAdapterCapability,
    CognitionAdapterResult,
    CognitionAdapterSnapshot,
    CognitionAdapterStatus,
    adapter_failure_result,
    adapter_success_result,
    utc_now,
)
from jarvis.cognition.models import (
    CognitionFailure,
    CognitionFailureKind,
    CognitionRequest,
    CognitionResponse,
    CognitionResponseKind,
    CognitionToken,
    CognitionTokenKind,
)
from jarvis.runtime.observability.structured_logger import get_logger


class FakeCognitionMode(StrEnum):
    """
    Fake cognition response mode.

    JARVIS:
        Deterministic voice-assistant-like responses.

    ECHO:
        Echoes the request text with a stable prefix.

    SCRIPTED:
        Uses configured exact-match scripted responses first, then fallback.
    """

    JARVIS = "jarvis"
    ECHO = "echo"
    SCRIPTED = "scripted"


@dataclass(frozen=True, slots=True)
class FakeCognitionConfig:
    """
    Configuration for FakeCognitionAdapter.

    This is intentionally deterministic. The fake adapter is for runtime
    validation, not cleverness.
    """

    adapter_name: str = "fake_cognition_adapter"
    mode: FakeCognitionMode = FakeCognitionMode.JARVIS
    default_response: str = "derived_fake_test_response"
    echo_prefix: str = "I heard you say: "
    scripted_responses: dict[str, str] = field(default_factory=dict)
    fail_phrases: tuple[str, ...] = ("simulate cognition failure",)
    streaming_chunk_size: int = 24
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.adapter_name.strip():
            raise ValueError("adapter_name cannot be empty.")

        if not self.default_response.strip():
            raise ValueError("default_response cannot be empty.")

        if not self.echo_prefix.strip():
            raise ValueError("echo_prefix cannot be empty.")

        if self.streaming_chunk_size <= 0:
            raise ValueError("streaming_chunk_size must be greater than zero.")

        for key, value in self.scripted_responses.items():
            if not key.strip():
                raise ValueError("scripted response key cannot be empty.")

            if not value.strip():
                raise ValueError("scripted response value cannot be empty.")

        for phrase in self.fail_phrases:
            if not phrase.strip():
                raise ValueError("fail_phrases cannot contain empty values.")


class FakeCognitionAdapter:
    """
    Deterministic cognition adapter for Phase 3 runtime validation.

    Responsibilities:
    - implement non-streaming cognition generation
    - simulate streaming tokens
    - support cancellation contract
    - expose adapter health/counters
    - never call a real LLM

    Non-responsibilities:
    - no prompt engineering
    - no memory retrieval
    - no tool execution
    - no audio or Presence internals
    """

    def __init__(
        self,
        *,
        config: FakeCognitionConfig | None = None,
    ) -> None:
        self._config = config or FakeCognitionConfig()
        self._config.validate()

        self._lock = RLock()
        self._logger = get_logger("cognition.fake_adapter")

        self._status = CognitionAdapterStatus.READY
        self._request_count = 0
        self._success_count = 0
        self._failure_count = 0
        self._cancelled_count = 0
        self._streaming_count = 0
        self._last_request_id: str | None = None
        self._last_error: str | None = None
        self._active_request_id: str | None = None
        self._cancelled_request_ids: set[str] = set()

    @property
    def name(self) -> str:
        return self._config.adapter_name

    @property
    def capabilities(self) -> tuple[CognitionAdapterCapability, ...]:
        return (
            CognitionAdapterCapability.NON_STREAMING,
            CognitionAdapterCapability.STREAMING,
            CognitionAdapterCapability.CANCELLATION,
        )

    def generate(self, request: CognitionRequest) -> CognitionAdapterResult:
        """
        Generate a deterministic final cognition response.
        """

        started_at = utc_now()

        with self._lock:
            self._status = CognitionAdapterStatus.BUSY
            self._request_count += 1
            self._last_request_id = request.request_id
            self._last_error = None
            self._active_request_id = request.request_id

        try:
            if self._is_cancelled(request.request_id):
                failure = self._make_failure(
                    request=request,
                    kind=CognitionFailureKind.CANCELLED,
                    message="cognition request was cancelled",
                )
                return self._finish_failure(
                    request=request,
                    failure=failure,
                    started_at=started_at,
                    metadata={"mode": self._config.mode.value},
                )

            if self._should_fail(request.text):
                failure = self._make_failure(
                    request=request,
                    kind=CognitionFailureKind.ADAPTER_ERROR,
                    message="fake cognition failure requested",
                )
                return self._finish_failure(
                    request=request,
                    failure=failure,
                    started_at=started_at,
                    metadata={"mode": self._config.mode.value},
                )

            response_text = self._response_text_for(request)
            response = CognitionResponse(
                request_id=request.request_id,
                text=response_text,
                kind=CognitionResponseKind.SPOKEN_REPLY,
                confidence=1.0,
                metadata={
                    "adapter": self.name,
                    "mode": self._config.mode.value,
                },
            )

            return self._finish_success(
                request=request,
                response=response,
                started_at=started_at,
                metadata={"mode": self._config.mode.value},
            )

        finally:
            with self._lock:
                if self._active_request_id == request.request_id:
                    self._active_request_id = None

    def stream(self, request: CognitionRequest) -> Iterator[CognitionToken]:
        """
        Stream deterministic cognition tokens.

        This simulates future LLM streaming and gives the runtime a stable stream
        contract before real models are connected.
        """

        with self._lock:
            self._streaming_count += 1

        response_text = self._response_text_for(request)
        chunks = tuple(self._chunk_text(response_text))

        for index, chunk in enumerate(chunks):
            final = index == len(chunks) - 1

            yield CognitionToken(
                request_id=request.request_id,
                index=index,
                text=chunk,
                kind=(
                    CognitionTokenKind.FINAL
                    if final
                    else CognitionTokenKind.TEXT
                ),
                final=final,
                metadata={
                    "adapter": self.name,
                    "mode": self._config.mode.value,
                },
            )

    def cancel(
        self,
        *,
        request_id: str,
        reason: str | None = None,
    ) -> bool:
        """
        Mark a request as cancelled.

        Fake cancellation is deterministic. Real adapters may use this to cancel
        backend generation later.
        """

        if not request_id.strip():
            return False

        with self._lock:
            self._cancelled_request_ids.add(request_id)
            self._cancelled_count += 1
            self._last_error = reason
            active = self._active_request_id == request_id

            if active:
                self._status = CognitionAdapterStatus.READY

        self._logger.info(
            "fake_cognition_cancelled",
            adapter=self.name,
            request_id=request_id,
            reason=reason,
            active=active,
        )

        return True

    def snapshot(self) -> CognitionAdapterSnapshot:
        """
        Return adapter diagnostics.
        """

        with self._lock:
            return CognitionAdapterSnapshot(
                name=self.name,
                status=self._status,
                capabilities=self.capabilities,
                request_count=self._request_count,
                success_count=self._success_count,
                failure_count=self._failure_count,
                cancelled_count=self._cancelled_count,
                streaming_count=self._streaming_count,
                last_request_id=self._last_request_id,
                last_error=self._last_error,
                metadata={
                    "mode": self._config.mode.value,
                    "active_request_id": self._active_request_id,
                    "scripted_response_count": len(
                        self._config.scripted_responses
                    ),
                },
            )

    def reset(self) -> None:
        """
        Reset counters and diagnostics.
        """

        with self._lock:
            self._status = CognitionAdapterStatus.READY
            self._request_count = 0
            self._success_count = 0
            self._failure_count = 0
            self._cancelled_count = 0
            self._streaming_count = 0
            self._last_request_id = None
            self._last_error = None
            self._active_request_id = None
            self._cancelled_request_ids.clear()

        self._logger.info("fake_cognition_reset", adapter=self.name)

    def _finish_success(
        self,
        *,
        request: CognitionRequest,
        response: CognitionResponse,
        started_at: datetime,
        metadata: dict[str, Any],
    ) -> CognitionAdapterResult:
        finished_at = utc_now()

        with self._lock:
            self._success_count += 1
            self._status = CognitionAdapterStatus.READY

        self._logger.info(
            "fake_cognition_generated",
            adapter=self.name,
            request_id=request.request_id,
            response_id=response.response_id,
        )

        return adapter_success_result(
            request=request,
            response=response,
            started_at=started_at,
            finished_at=finished_at,
            metadata=metadata,
        )

    def _finish_failure(
        self,
        *,
        request: CognitionRequest,
        failure: CognitionFailure,
        started_at: datetime,
        metadata: dict[str, Any],
    ) -> CognitionAdapterResult:
        finished_at = utc_now()

        with self._lock:
            self._failure_count += 1
            self._status = CognitionAdapterStatus.READY
            self._last_error = failure.message

        self._logger.error(
            "fake_cognition_failed",
            adapter=self.name,
            request_id=request.request_id,
            failure_id=failure.failure_id,
            failure_kind=failure.kind.value,
            failure_message=failure.message,
        )

        return adapter_failure_result(
            request=request,
            failure=failure,
            started_at=started_at,
            finished_at=finished_at,
            metadata=metadata,
        )

    def _make_failure(
        self,
        *,
        request: CognitionRequest,
        kind: CognitionFailureKind,
        message: str,
    ) -> CognitionFailure:
        return CognitionFailure(
            request_id=request.request_id,
            kind=kind,
            message=message,
            recoverable=True,
            metadata={
                "adapter": self.name,
                "mode": self._config.mode.value,
            },
        )

    def _response_text_for(self, request: CognitionRequest) -> str:
        normalized = self._normalize(request.text)

        scripted = self._scripted_response_for(normalized)
        if scripted is not None:
            return scripted

        if self._config.mode == FakeCognitionMode.ECHO:
            return f"{self._config.echo_prefix}{request.text}"

        if self._config.mode == FakeCognitionMode.SCRIPTED:
            return self._config.default_response

        return self._jarvis_response_for(normalized)

    def _scripted_response_for(self, normalized_text: str) -> str | None:
        scripted_lookup = {
            self._normalize(key): value
            for key, value in self._config.scripted_responses.items()
        }

        return scripted_lookup.get(normalized_text)

    def _jarvis_response_for(self, normalized_text: str) -> str:
        if any(greeting in normalized_text for greeting in ("hello", "hi")):
            return f"derived_fake_test_response::{normalized_text}"

        if "can you hear me" in normalized_text:
            return f"derived_fake_test_response::{normalized_text}"

        if "what did we build" in normalized_text:
            return (
                "We built the Presence Runtime and started the Cognition "
                "Runtime foundation."
            )

        if "phase 3" in normalized_text:
            return (
                "Phase 3 is the Cognition Runtime. It connects your voice "
                "pipeline to a real brain layer."
            )

        return self._config.default_response

    def _should_fail(self, text: str) -> bool:
        normalized = self._normalize(text)

        return any(
            self._normalize(phrase) in normalized
            for phrase in self._config.fail_phrases
        )

    def _is_cancelled(self, request_id: str) -> bool:
        with self._lock:
            return request_id in self._cancelled_request_ids

    def _chunk_text(self, text: str) -> Iterator[str]:
        chunk_size = self._config.streaming_chunk_size

        for start in range(0, len(text), chunk_size):
            yield text[start : start + chunk_size]

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.casefold().split())