from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from jarvis.cognition.action_planning import (
    ToolActionPermissionMode,
    ToolActionPlanner,
    ToolActionType,
)
from jarvis.cognition.local_llm_adapter import (
    LocalLLMAdapter,
    LocalLLMBackendResult,
    LocalLLMBackendSnapshot,
    LocalLLMBackendStatus,
    LocalLLMBackendToken,
)
from jarvis.cognition.memory import (
    InMemoryShortTermMemoryStore,
    ShortTermMemoryKind,
    ShortTermMemoryPriority,
)
from jarvis.cognition.models import (
    CognitionRequest,
    CognitionRuntimePolicy,
    SpokenResponseStyle,
)
from jarvis.cognition.planning import (
    ResponseAnswerMode,
    ResponsePlanner,
)
from jarvis.cognition.session_context import ConversationSessionStore
from jarvis.cognition.spoken_policy import SpokenDialoguePolicy
from jarvis.cognition.streaming import (
    CognitionStreamingState,
    StreamingTokenPipeline,
)
from jarvis.runtime.observability.structured_logger import get_logger


@dataclass(frozen=True, slots=True)
class CognitionValidationCheck:
    """
    One Phase 3 validation check.
    """

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class CognitionPhase3ValidationReport:
    """
    Phase 3 cognition integration validation report.
    """

    passed: bool
    checks: tuple[CognitionValidationCheck, ...]

    @property
    def passed_count(self) -> int:
        return sum(1 for check in self.checks if check.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for check in self.checks if not check.passed)

    @property
    def total_count(self) -> int:
        return len(self.checks)


class ValidationLocalLLMBackend:
    """
    Deterministic local LLM backend used only for integration validation.

    It proves the adapter boundary works without requiring a real model install.
    """

    def __init__(self) -> None:
        self.request_count = 0
        self.streaming_count = 0
        self.cancelled_count = 0
        self.last_prompt: str | None = None
        self.last_system_prompt: str | None = None
        self.last_request_id: str | None = None

    @property
    def name(self) -> str:
        return "validation_local_llm_backend"

    def generate(
        self,
        *,
        prompt: str,
        system_prompt: str,
        request: CognitionRequest,
    ) -> LocalLLMBackendResult:
        self.request_count += 1
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        self.last_request_id = request.request_id

        return LocalLLMBackendResult(
            text=(
                "Yes sir. The cognition runtime is connected and ready "
                "for the next integration step."
            ),
            confidence=0.91,
            metadata={
                "validation_backend": True,
            },
        )

    def stream(
        self,
        *,
        prompt: str,
        system_prompt: str,
        request: CognitionRequest,
    ) -> Iterator[LocalLLMBackendToken]:
        self.streaming_count += 1
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        self.last_request_id = request.request_id

        yield LocalLLMBackendToken(text="Yes sir.")
        yield LocalLLMBackendToken(text=" Streaming cognition")
        yield LocalLLMBackendToken(text=" is connected.", final=True)

    def cancel(
        self,
        *,
        request_id: str,
        reason: str | None = None,
    ) -> bool:
        self.cancelled_count += 1
        self.last_request_id = request_id
        return True

    def snapshot(self) -> LocalLLMBackendSnapshot:
        return LocalLLMBackendSnapshot(
            name=self.name,
            status=LocalLLMBackendStatus.READY,
            request_count=self.request_count,
            streaming_count=self.streaming_count,
            cancelled_count=self.cancelled_count,
        )


class CognitionPhase3Validator:
    """
    Validates that Phase 3 cognition components work together.

    Responsibilities:
    - run deterministic end-to-end cognition validation
    - prove session context, memory, planning, action planning, adapter,
      streaming, and spoken policy can connect
    - produce a clear pass/fail report

    Non-responsibilities:
    - no real LLM install
    - no laptop control
    - no tool execution
    - no microphone/STT/TTS execution
    """

    def __init__(self) -> None:
        self._logger = get_logger("cognition.phase3_validation")

    def validate(self) -> CognitionPhase3ValidationReport:
        checks: list[CognitionValidationCheck] = []

        checks.extend(self._validate_conversation_cognition_path())
        checks.extend(self._validate_streaming_path())
        checks.extend(self._validate_action_safety_path())

        report = CognitionPhase3ValidationReport(
            passed=all(check.passed for check in checks),
            checks=tuple(checks),
        )

        self._logger.info(
            "phase3_cognition_validation_completed",
            passed=report.passed,
            passed_count=report.passed_count,
            failed_count=report.failed_count,
            total_count=report.total_count,
        )

        return report

    def _validate_conversation_cognition_path(
        self,
    ) -> tuple[CognitionValidationCheck, ...]:
        checks: list[CognitionValidationCheck] = []

        session = ConversationSessionStore()
        memory = InMemoryShortTermMemoryStore()
        planner = ResponsePlanner()
        action_planner = ToolActionPlanner()
        backend = ValidationLocalLLMBackend()
        adapter = LocalLLMAdapter(backend=backend)
        spoken_policy = SpokenDialoguePolicy()

        memory.remember_text(
            "Phase 3 includes cognition runtime, planning, memory, "
            "streaming, and action contracts.",
            kind=ShortTermMemoryKind.PROJECT_CONTEXT,
            priority=ShortTermMemoryPriority.HIGH,
        )

        request = CognitionRequest(
            request_id="phase3-validation-request",
            text="What is the current cognition status?",
            turn_id="phase3-validation-turn",
            policy=CognitionRuntimePolicy(
                allow_memory_lookup=True,
                allow_tools=True,
                spoken_style=SpokenResponseStyle.CONCISE,
            ),
        )

        session.add_user_request(request)
        session_enriched = session.enrich_request(request)
        memory_enriched = memory.enrich_request(session_enriched)

        decision = planner.plan(memory_enriched)
        action_plan = action_planner.plan(
            CognitionRequest(
                request_id="phase3-validation-action",
                text="open diagnostics",
                policy=CognitionRuntimePolicy(allow_tools=True),
            )
        )
        adapter_result = adapter.generate(memory_enriched)

        if adapter_result.response is not None:
            shaped_response = spoken_policy.apply_to_response(
                adapter_result.response,
                style=SpokenResponseStyle.CONCISE,
            )
            session.add_assistant_response(shaped_response)

        session_snapshot = session.snapshot()
        memory_snapshot = memory.snapshot()
        adapter_snapshot = adapter.snapshot()
        spoken_snapshot = spoken_policy.snapshot()

        checks.append(
            self._check(
                name="session_context_connected",
                passed=session_snapshot.turn_count == 2,
                detail="session captured user and assistant turns",
            )
        )
        checks.append(
            self._check(
                name="memory_context_connected",
                passed=memory_enriched.context.item_count >= 1
                and memory_snapshot.item_count >= 1,
                detail="short-term memory enriched cognition request",
            )
        )
        checks.append(
            self._check(
                name="response_planning_connected",
                passed=decision.answer_mode == ResponseAnswerMode.DIRECT,
                detail="response planner created direct answer decision",
            )
        )
        checks.append(
            self._check(
                name="action_planning_connected",
                passed=action_plan.proposal_count == 1
                and action_plan.proposals[0].action_type
                == ToolActionType.OPEN_APPLICATION,
                detail="action planner created safe structured proposal",
            )
        )
        checks.append(
            self._check(
                name="local_llm_adapter_connected",
                passed=adapter_result.succeeded
                and adapter_result.response is not None
                and adapter_snapshot.success_count == 1,
                detail="local LLM adapter boundary generated response",
            )
        )
        checks.append(
            self._check(
                name="spoken_policy_connected",
                passed=spoken_snapshot.prepared_count == 1,
                detail="spoken policy shaped cognition response",
            )
        )

        return tuple(checks)

    def _validate_streaming_path(self) -> tuple[CognitionValidationCheck, ...]:
        backend = ValidationLocalLLMBackend()
        adapter = LocalLLMAdapter(backend=backend)
        pipeline = StreamingTokenPipeline(adapter=adapter)

        request = CognitionRequest(
            request_id="phase3-validation-streaming",
            text="Stream the cognition status.",
            policy=CognitionRuntimePolicy(streaming_enabled=True),
        )

        result = pipeline.stream_request(request)
        snapshot = pipeline.snapshot()

        return (
            self._check(
                name="streaming_pipeline_connected",
                passed=result.state == CognitionStreamingState.COMPLETED
                and result.response is not None
                and len(result.tokens) == 3
                and len(result.speech_chunks) >= 1,
                detail="streaming pipeline produced tokens and speech chunks",
            ),
            self._check(
                name="streaming_whitespace_preserved",
                passed=result.response is not None
                and "Streaming cognition" in result.response.text,
                detail="streamed token whitespace reconstructed correctly",
            ),
            self._check(
                name="streaming_diagnostics_connected",
                passed=snapshot.completed_count == 1
                and snapshot.token_count == 3,
                detail="streaming diagnostics recorded completion",
            ),
        )

    def _validate_action_safety_path(
        self,
    ) -> tuple[CognitionValidationCheck, ...]:
        planner = ToolActionPlanner()

        dangerous_plan = planner.plan(
            CognitionRequest(
                request_id="phase3-validation-dangerous-action",
                text="delete system32",
            )
        )
        terminal_plan = planner.plan(
            CognitionRequest(
                request_id="phase3-validation-terminal-action",
                text="run powershell Get-Process",
            )
        )
        safe_plan = planner.plan(
            CognitionRequest(
                request_id="phase3-validation-safe-action",
                text="open notepad",
            )
        )

        return (
            self._check(
                name="dangerous_action_blocked",
                passed=dangerous_plan.blocked
                and dangerous_plan.safety.permission_mode
                == ToolActionPermissionMode.BLOCKED,
                detail="dangerous laptop action was blocked",
            ),
            self._check(
                name="terminal_action_blocked_by_default",
                passed=terminal_plan.blocked
                and terminal_plan.safety.permission_mode
                == ToolActionPermissionMode.BLOCKED,
                detail="terminal execution is blocked by default",
            ),
            self._check(
                name="safe_action_requires_permission",
                passed=not safe_plan.blocked
                and safe_plan.safety.permission_mode
                == ToolActionPermissionMode.CONFIRMATION_REQUIRED,
                detail="safe action remains permission-controlled",
            ),
        )

    @staticmethod
    def _check(
        *,
        name: str,
        passed: bool,
        detail: str,
    ) -> CognitionValidationCheck:
        return CognitionValidationCheck(
            name=name,
            passed=passed,
            detail=detail,
        )


def validate_phase3_cognition() -> CognitionPhase3ValidationReport:
    """
    Run Phase 3 cognition integration validation.
    """

    return CognitionPhase3Validator().validate()