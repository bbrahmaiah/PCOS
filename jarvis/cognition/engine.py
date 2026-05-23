from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any

from jarvis.cognition.adapters import (
    CognitionAdapter,
    CognitionAdapterResult,
    CognitionAdapterSnapshot,
)
from jarvis.cognition.models import (
    CognitionFailure,
    CognitionFailureKind,
    CognitionPlan,
    CognitionPlanKind,
    CognitionRequest,
    CognitionResponse,
    CognitionResponseKind,
)
from jarvis.runtime.observability.structured_logger import get_logger


@dataclass(frozen=True, slots=True)
class CognitionEngineConfig:
    """
    Runtime configuration for CognitionEngine.
    """

    name: str = "cognition_engine"
    fallback_response_text: str = "I had trouble thinking that through, sir."
    max_input_chars: int = 12_000
    enforce_spoken_limit: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if not self.fallback_response_text.strip():
            raise ValueError("fallback_response_text cannot be empty.")

        if self.max_input_chars <= 0:
            raise ValueError("max_input_chars must be greater than zero.")


@dataclass(frozen=True, slots=True)
class CognitionEngineResult:
    """
    Result of one cognition engine execution.
    """

    request_id: str
    plan: CognitionPlan
    response: CognitionResponse | None = None
    failure: CognitionFailure | None = None
    adapter_result: CognitionAdapterResult | None = None

    @property
    def succeeded(self) -> bool:
        return self.response is not None

    @property
    def failed(self) -> bool:
        return self.failure is not None


@dataclass(frozen=True, slots=True)
class CognitionEngineSnapshot:
    """
    Observable engine diagnostics.
    """

    name: str
    request_count: int
    success_count: int
    failure_count: int
    fallback_count: int
    last_request_id: str | None
    last_response_id: str | None
    last_failure_id: str | None
    last_error: str | None
    adapter: CognitionAdapterSnapshot


class CognitionEngine:
    """
    Thinking orchestration layer.

    Responsibilities:
    - validate and normalize cognition requests
    - create a first-pass response plan
    - call the configured cognition adapter
    - normalize adapter success/failure into engine result
    - enforce basic spoken-response safety limits
    - expose diagnostics

    Non-responsibilities:
    - no EventBus subscriptions
    - no microphone/STT/TTS knowledge
    - no direct model implementation
    - no tool execution
    - no long-term memory system yet
    """

    def __init__(
        self,
        *,
        adapter: CognitionAdapter,
        config: CognitionEngineConfig | None = None,
    ) -> None:
        self._config = config or CognitionEngineConfig()
        self._config.validate()

        self._adapter = adapter
        self._lock = RLock()
        self._logger = get_logger("cognition.engine")

        self._request_count = 0
        self._success_count = 0
        self._failure_count = 0
        self._fallback_count = 0
        self._last_request_id: str | None = None
        self._last_response_id: str | None = None
        self._last_failure_id: str | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def adapter(self) -> CognitionAdapter:
        return self._adapter

    def run(self, request: CognitionRequest) -> CognitionEngineResult:
        """
        Run one cognition request through planning and adapter execution.
        """

        with self._lock:
            self._request_count += 1
            self._last_request_id = request.request_id
            self._last_response_id = None
            self._last_failure_id = None
            self._last_error = None

        plan = self.create_plan(request)

        validation_failure = self._validate_request(request=request, plan=plan)
        if validation_failure is not None:
            return self._finish_failure(
                request=request,
                plan=plan,
                failure=validation_failure,
                adapter_result=None,
            )

        self._logger.info(
            "cognition_engine_running",
            engine=self.name,
            adapter=self._adapter.name,
            request_id=request.request_id,
            plan_kind=plan.kind.value,
        )

        try:
            adapter_result = self._adapter.generate(request)

        except Exception as exc:
            failure = self._make_failure(
                request=request,
                kind=CognitionFailureKind.ADAPTER_ERROR,
                message=f"{type(exc).__name__}: {exc}",
                metadata={
                    "engine": self.name,
                    "adapter": self._adapter.name,
                    "exception_type": type(exc).__name__,
                },
            )
            return self._finish_failure(
                request=request,
                plan=plan,
                failure=failure,
                adapter_result=None,
            )

        if adapter_result.request_id != request.request_id:
            failure = self._make_failure(
                request=request,
                kind=CognitionFailureKind.ADAPTER_ERROR,
                message="adapter result request_id does not match request",
                metadata={
                    "engine": self.name,
                    "adapter": self._adapter.name,
                    "adapter_result_request_id": adapter_result.request_id,
                },
            )
            return self._finish_failure(
                request=request,
                plan=plan,
                failure=failure,
                adapter_result=adapter_result,
            )

        if adapter_result.failure is not None:
            return self._finish_failure(
                request=request,
                plan=plan,
                failure=adapter_result.failure,
                adapter_result=adapter_result,
            )

        if adapter_result.response is None:
            failure = self._make_failure(
                request=request,
                kind=CognitionFailureKind.ADAPTER_ERROR,
                message="adapter result contained no response",
                metadata={
                    "engine": self.name,
                    "adapter": self._adapter.name,
                },
            )
            return self._finish_failure(
                request=request,
                plan=plan,
                failure=failure,
                adapter_result=adapter_result,
            )

        response = self._normalize_response(
            request=request,
            plan=plan,
            response=adapter_result.response,
        )

        return self._finish_success(
            request=request,
            plan=plan,
            response=response,
            adapter_result=adapter_result,
        )

    def create_plan(self, request: CognitionRequest) -> CognitionPlan:
        """
        Create an initial response plan.

        This is deliberately simple in Step 8. Later steps will expand this into
        richer planning, safety, memory, and tool policy.
        """

        normalized_text = self._normalize_text(request.text)
        notes: tuple[str, ...]

        if not normalized_text:
            kind = CognitionPlanKind.ASK_CLARIFICATION
            confidence = 0.0
            needs_clarification = True
            notes = ("empty request text",)

        elif self._looks_like_clarification_needed(normalized_text):
            kind = CognitionPlanKind.ASK_CLARIFICATION
            confidence = 0.6
            needs_clarification = True
            notes = ("request appears underspecified",)

        else:
            kind = CognitionPlanKind.DIRECT_ANSWER
            confidence = 1.0
            needs_clarification = False
            notes = ()

        return CognitionPlan(
            request_id=request.request_id,
            kind=kind,
            confidence=confidence,
            needs_clarification=needs_clarification,
            notes=notes,
            metadata={
                "engine": self.name,
                "spoken_style": request.policy.spoken_style.value,
                "allow_tools": request.policy.allow_tools,
                "allow_memory_lookup": request.policy.allow_memory_lookup,
            },
        )

    def fallback_response_for(
        self,
        *,
        request: CognitionRequest,
        failure: CognitionFailure,
    ) -> CognitionResponse:
        """
        Build a safe fallback response for a cognition failure.

        This is not automatically returned from run(), because workers/bridges
        need to preserve typed failure paths. It is available for fallback
        bridges and future recovery policies.
        """

        with self._lock:
            self._fallback_count += 1

        return CognitionResponse(
            request_id=request.request_id,
            text=self._config.fallback_response_text,
            kind=CognitionResponseKind.ERROR_FALLBACK,
            confidence=0.0,
            metadata={
                "engine": self.name,
                "failure_id": failure.failure_id,
                "failure_kind": failure.kind.value,
            },
        )

    def snapshot(self) -> CognitionEngineSnapshot:
        """
        Return engine diagnostics.
        """

        with self._lock:
            return CognitionEngineSnapshot(
                name=self.name,
                request_count=self._request_count,
                success_count=self._success_count,
                failure_count=self._failure_count,
                fallback_count=self._fallback_count,
                last_request_id=self._last_request_id,
                last_response_id=self._last_response_id,
                last_failure_id=self._last_failure_id,
                last_error=self._last_error,
                adapter=self._adapter.snapshot(),
            )

    def reset(self) -> None:
        """
        Reset engine counters.

        The adapter is not reset because real adapters may hold model state.
        """

        with self._lock:
            self._request_count = 0
            self._success_count = 0
            self._failure_count = 0
            self._fallback_count = 0
            self._last_request_id = None
            self._last_response_id = None
            self._last_failure_id = None
            self._last_error = None

        self._logger.info("cognition_engine_reset", engine=self.name)

    def _validate_request(
        self,
        *,
        request: CognitionRequest,
        plan: CognitionPlan,
    ) -> CognitionFailure | None:
        if not request.text.strip():
            return self._make_failure(
                request=request,
                kind=CognitionFailureKind.VALIDATION_ERROR,
                message="request text cannot be empty",
                metadata={
                    "engine": self.name,
                    "plan_id": plan.plan_id,
                },
            )

        if len(request.text) > self._config.max_input_chars:
            return self._make_failure(
                request=request,
                kind=CognitionFailureKind.VALIDATION_ERROR,
                message="request text exceeds engine input limit",
                metadata={
                    "engine": self.name,
                    "plan_id": plan.plan_id,
                    "max_input_chars": self._config.max_input_chars,
                    "actual_chars": len(request.text),
                },
            )

        return None

    def _normalize_response(
        self,
        *,
        request: CognitionRequest,
        plan: CognitionPlan,
        response: CognitionResponse,
    ) -> CognitionResponse:
        text = response.text.strip()

        if self._config.enforce_spoken_limit:
            text = self._bounded_text(
                text=text,
                max_chars=request.policy.max_response_chars,
            )

        return response.model_copy(
            update={
                "text": text,
                "plan": response.plan or plan,
                "metadata": {
                    **response.metadata,
                    "engine": self.name,
                    "plan_id": plan.plan_id,
                },
            }
        )

    def _finish_success(
        self,
        *,
        request: CognitionRequest,
        plan: CognitionPlan,
        response: CognitionResponse,
        adapter_result: CognitionAdapterResult,
    ) -> CognitionEngineResult:
        with self._lock:
            self._success_count += 1
            self._last_response_id = response.response_id
            self._last_failure_id = None
            self._last_error = None

        self._logger.info(
            "cognition_engine_completed",
            engine=self.name,
            request_id=request.request_id,
            response_id=response.response_id,
            plan_id=plan.plan_id,
        )

        return CognitionEngineResult(
            request_id=request.request_id,
            plan=plan,
            response=response,
            adapter_result=adapter_result,
        )

    def _finish_failure(
        self,
        *,
        request: CognitionRequest,
        plan: CognitionPlan,
        failure: CognitionFailure,
        adapter_result: CognitionAdapterResult | None,
    ) -> CognitionEngineResult:
        with self._lock:
            self._failure_count += 1
            self._last_failure_id = failure.failure_id
            self._last_response_id = None
            self._last_error = failure.message

        self._logger.error(
            "cognition_engine_failed",
            engine=self.name,
            request_id=request.request_id,
            failure_id=failure.failure_id,
            failure_kind=failure.kind.value,
            failure_message=failure.message,
            plan_id=plan.plan_id,
        )

        return CognitionEngineResult(
            request_id=request.request_id,
            plan=plan,
            failure=failure,
            adapter_result=adapter_result,
        )

    def _make_failure(
        self,
        *,
        request: CognitionRequest,
        kind: CognitionFailureKind,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> CognitionFailure:
        return CognitionFailure(
            request_id=request.request_id,
            kind=kind,
            message=message,
            recoverable=True,
            metadata=metadata or {},
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.casefold().split())

    @staticmethod
    def _looks_like_clarification_needed(normalized_text: str) -> bool:
        return normalized_text in {
            "what",
            "why",
            "how",
            "do it",
            "this",
            "that",
            "explain",
        }

    @staticmethod
    def _bounded_text(
        *,
        text: str,
        max_chars: int,
    ) -> str:
        if len(text) <= max_chars:
            return text

        return text[:max_chars].rstrip()