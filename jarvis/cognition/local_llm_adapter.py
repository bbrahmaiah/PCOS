from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from threading import RLock
from typing import Any, Protocol, runtime_checkable

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


class LocalLLMBackendStatus(StrEnum):
    """
    Local LLM backend health state.
    """

    READY = "ready"
    BUSY = "busy"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class LocalLLMBackendResult:
    """
    Non-streaming local LLM backend result.
    """

    text: str
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.text.strip():
            raise ValueError("backend result text cannot be empty.")

        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError("backend result confidence must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class LocalLLMBackendToken:
    """
    One streamed token from a local LLM backend.

    Token text preserves whitespace.
    """

    text: str
    final: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.text.strip():
            raise ValueError("backend token text cannot be empty.")


@dataclass(frozen=True, slots=True)
class LocalLLMBackendSnapshot:
    """
    Observable local LLM backend diagnostics.
    """

    name: str
    status: LocalLLMBackendStatus
    request_count: int
    streaming_count: int
    cancelled_count: int
    last_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LocalLLMBackend(Protocol):
    """
    Protocol for a local LLM backend implementation.

    Future implementations can be:
    - Ollama
    - llama.cpp server
    - ctransformers
    - custom local inference server

    The adapter depends on this protocol, not a concrete model package.
    """

    @property
    def name(self) -> str:
        """Stable backend name."""

    def generate(
        self,
        *,
        prompt: str,
        system_prompt: str,
        request: CognitionRequest,
    ) -> LocalLLMBackendResult:
        """Generate a complete text response."""

    def stream(
        self,
        *,
        prompt: str,
        system_prompt: str,
        request: CognitionRequest,
    ) -> Iterator[LocalLLMBackendToken]:
        """Stream response tokens."""

    def cancel(self, *, request_id: str, reason: str | None = None) -> bool:
        """Attempt backend cancellation."""

    def snapshot(self) -> LocalLLMBackendSnapshot:
        """Return backend diagnostics."""


@dataclass(frozen=True, slots=True)
class LocalLLMAdapterConfig:
    """
    Configuration for LocalLLMAdapter.

    This class controls prompt shaping at the adapter boundary. It does not
    install or configure a real model. Real model configuration belongs inside
    the concrete backend.
    """

    adapter_name: str = "local_llm_adapter"
    system_prompt: str = (
        "You are JARVIS, a concise real-time voice assistant. "
        "Answer clearly, briefly, and naturally. "
        "Avoid markdown unless the user explicitly asks for structured output."
    )
    include_context: bool = True
    max_prompt_chars: int = 12_000
    max_context_items: int = 8
    default_confidence: float = 0.8
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.adapter_name.strip():
            raise ValueError("adapter_name cannot be empty.")

        if not self.system_prompt.strip():
            raise ValueError("system_prompt cannot be empty.")

        if self.max_prompt_chars <= 0:
            raise ValueError("max_prompt_chars must be greater than zero.")

        if self.max_context_items < 0:
            raise ValueError("max_context_items cannot be negative.")

        if self.default_confidence < 0.0 or self.default_confidence > 1.0:
            raise ValueError("default_confidence must be between 0 and 1.")


class LocalLLMAdapter:
    """
    Adapter that exposes a local LLM backend through CognitionAdapter contracts.

    Responsibilities:
    - build safe local LLM prompt input
    - call backend.generate() or backend.stream()
    - convert backend output into CognitionResponse / CognitionToken
    - expose adapter diagnostics
    - forward cancellation to backend

    Non-responsibilities:
    - no direct model package dependency
    - no EventBus logic
    - no microphone/STT/TTS
    - no tool execution
    - no persistent memory storage
    """

    def __init__(
        self,
        *,
        backend: LocalLLMBackend,
        config: LocalLLMAdapterConfig | None = None,
    ) -> None:
        self._config = config or LocalLLMAdapterConfig()
        self._config.validate()

        self._backend = backend
        self._lock = RLock()
        self._logger = get_logger("cognition.local_llm_adapter")

        self._status = CognitionAdapterStatus.READY
        self._request_count = 0
        self._success_count = 0
        self._failure_count = 0
        self._cancelled_count = 0
        self._streaming_count = 0
        self._last_request_id: str | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.adapter_name

    @property
    def backend(self) -> LocalLLMBackend:
        return self._backend

    @property
    def capabilities(self) -> tuple[CognitionAdapterCapability, ...]:
        return (
            CognitionAdapterCapability.NON_STREAMING,
            CognitionAdapterCapability.STREAMING,
            CognitionAdapterCapability.CANCELLATION,
            CognitionAdapterCapability.MEMORY_CONTEXT,
        )

    def generate(self, request: CognitionRequest) -> CognitionAdapterResult:
        """
        Generate a complete cognition response through the local backend.
        """

        started_at = utc_now()

        with self._lock:
            self._status = CognitionAdapterStatus.BUSY
            self._request_count += 1
            self._last_request_id = request.request_id
            self._last_error = None

        prompt = self.build_prompt(request)

        try:
            backend_result = self._backend.generate(
                prompt=prompt,
                system_prompt=self._config.system_prompt,
                request=request,
            )
            backend_result.validate()

            response = CognitionResponse(
                request_id=request.request_id,
                text=backend_result.text,
                kind=CognitionResponseKind.SPOKEN_REPLY,
                confidence=backend_result.confidence,
                metadata={
                    "adapter": self.name,
                    "backend": self._backend.name,
                    "local_llm": True,
                    **backend_result.metadata,
                },
            )

            return self._finish_success(
                request=request,
                response=response,
                started_at=started_at,
                finished_at=utc_now(),
            )

        except Exception as exc:
            failure = self._make_failure(
                request=request,
                message=f"{type(exc).__name__}: {exc}",
                metadata={
                    "exception_type": type(exc).__name__,
                },
            )

            return self._finish_failure(
                request=request,
                failure=failure,
                started_at=started_at,
                finished_at=utc_now(),
            )

    def stream(self, request: CognitionRequest) -> Iterator[CognitionToken]:
        """
        Stream cognition tokens through the local backend.
        """

        prompt = self.build_prompt(request)

        with self._lock:
            self._streaming_count += 1
            self._last_request_id = request.request_id
            self._last_error = None

        for index, backend_token in enumerate(
            self._backend.stream(
                prompt=prompt,
                system_prompt=self._config.system_prompt,
                request=request,
            )
        ):
            backend_token.validate()

            yield CognitionToken(
                request_id=request.request_id,
                index=index,
                text=backend_token.text,
                kind=(
                    CognitionTokenKind.FINAL
                    if backend_token.final
                    else CognitionTokenKind.TEXT
                ),
                final=backend_token.final,
                metadata={
                    "adapter": self.name,
                    "backend": self._backend.name,
                    "local_llm": True,
                    **backend_token.metadata,
                },
            )

    def cancel(
        self,
        *,
        request_id: str,
        reason: str | None = None,
    ) -> bool:
        """
        Forward cancellation to the local backend.
        """

        if not request_id.strip():
            return False

        cancelled = self._backend.cancel(
            request_id=request_id,
            reason=reason,
        )

        if cancelled:
            with self._lock:
                self._cancelled_count += 1
                self._status = CognitionAdapterStatus.READY
                self._last_error = reason

        self._logger.info(
            "local_llm_adapter_cancel_requested",
            adapter=self.name,
            backend=self._backend.name,
            request_id=request_id,
            reason=reason,
            cancelled=cancelled,
        )

        return cancelled

    def build_prompt(self, request: CognitionRequest) -> str:
        """
        Build a bounded prompt for the backend.
        """

        sections: list[str] = []

        if self._config.include_context and request.context.items:
            context_lines = [
                f"- {item.kind}: {item.text}"
                for item in request.context.items[: self._config.max_context_items]
            ]

            if context_lines:
                sections.append("Relevant context:\n" + "\n".join(context_lines))

        sections.append(f"User said:\n{request.text.strip()}")

        sections.append(
            "Respond as JARVIS in a concise, spoken, real-time style."
        )

        prompt = "\n\n".join(sections).strip()

        if len(prompt) <= self._config.max_prompt_chars:
            return prompt

        return prompt[: self._config.max_prompt_chars].rstrip()

    def snapshot(self) -> CognitionAdapterSnapshot:
        """
        Return adapter diagnostics.
        """

        backend_snapshot = self._backend.snapshot()

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
                    "backend_name": backend_snapshot.name,
                    "backend_status": backend_snapshot.status.value,
                    "backend_request_count": backend_snapshot.request_count,
                    "backend_streaming_count": backend_snapshot.streaming_count,
                    "backend_cancelled_count": backend_snapshot.cancelled_count,
                    **self._config.metadata,
                },
            )

    def reset(self) -> None:
        """
        Reset adapter counters.

        Backend counters are intentionally not reset here because real backends
        may represent long-lived model runtimes.
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

        self._logger.info("local_llm_adapter_reset", adapter=self.name)

    def _finish_success(
        self,
        *,
        request: CognitionRequest,
        response: CognitionResponse,
        started_at: datetime,
        finished_at: datetime,
    ) -> CognitionAdapterResult:
        with self._lock:
            self._success_count += 1
            self._status = CognitionAdapterStatus.READY
            self._last_error = None

        self._logger.info(
            "local_llm_adapter_generated",
            adapter=self.name,
            backend=self._backend.name,
            request_id=request.request_id,
            response_id=response.response_id,
        )

        return adapter_success_result(
            request=request,
            response=response,
            started_at=started_at,
            finished_at=finished_at,
            metadata={
                "adapter": self.name,
                "backend": self._backend.name,
            },
        )

    def _finish_failure(
        self,
        *,
        request: CognitionRequest,
        failure: CognitionFailure,
        started_at: datetime,
        finished_at: datetime,
    ) -> CognitionAdapterResult:
        with self._lock:
            self._failure_count += 1
            self._status = CognitionAdapterStatus.READY
            self._last_error = failure.message

        self._logger.error(
            "local_llm_adapter_failed",
            adapter=self.name,
            backend=self._backend.name,
            request_id=request.request_id,
            failure_id=failure.failure_id,
            failure_message=failure.message,
        )

        return adapter_failure_result(
            request=request,
            failure=failure,
            started_at=started_at,
            finished_at=finished_at,
            metadata={
                "adapter": self.name,
                "backend": self._backend.name,
            },
        )

    def _make_failure(
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
                "adapter": self.name,
                "backend": self._backend.name,
                **(metadata or {}),
            },
        )