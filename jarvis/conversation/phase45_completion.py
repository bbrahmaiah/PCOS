from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import RLock

from pydantic import Field, field_validator

from jarvis.conversation.attention import (
    AttentionDisposition,
    AttentionPriority,
    AttentionRuntime,
    AttentionSignal,
    AttentionSignalKind,
)
from jarvis.conversation.endpointing import (
    AdaptiveEndpointingEngine,
    EndpointAction,
    EndpointingInput,
)
from jarvis.conversation.interrupt_controller import (
    InterruptReason,
)
from jarvis.conversation.models import (
    ConversationMode,
    ConversationModel,
    TurnDetectionInput,
    TurnUrgency,
)
from jarvis.conversation.runtime import (
    RealConversationInput,
    RealConversationRuntime,
    RealConversationRuntimeStatus,
)
from jarvis.conversation.session_runtime import (
    ConversationContinuityStatus,
    ConversationSessionRuntime,
)
from jarvis.conversation.state_machine import (
    ConversationState,
    ConversationStateEvent,
    ConversationStateEventKind,
    ConversationStateMachine,
)
from jarvis.conversation.streaming import (
    StreamingConversationCoordinator,
    StreamingConversationEvent,
    StreamingCoordinatorAction,
    StreamingEventKind,
    StreamingLifecycle,
)
from jarvis.conversation.turn_detection import AdaptiveTurnDetector
from jarvis.runtime.observability.structured_logger import get_logger


class Phase45CompletionStatus(StrEnum):
    """
    Final Phase 4.5 completion status.
    """

    PASSED = "passed"
    FAILED = "failed"


class Phase45CompletionCheckKind(StrEnum):
    """
    Type of Phase 4.5 completion check.
    """

    TURN_DETECTION = "turn_detection"
    STATE_MACHINE = "state_machine"
    ENDPOINTING = "endpointing"
    STREAMING = "streaming"
    INTERRUPTION = "interruption"
    SESSION_CONTINUITY = "session_continuity"
    ATTENTION = "attention"
    REAL_RUNTIME = "real_runtime"
    COMPLETION = "completion"


class Phase45CompletionCheck(ConversationModel):
    """
    One formal Phase 4.5 completion check.
    """

    name: str
    kind: Phase45CompletionCheckKind
    passed: bool
    detail: str
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("name", "detail")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class Phase45CompletionResult(ConversationModel):
    """
    Final Phase 4.5 completion result.
    """

    status: Phase45CompletionStatus
    checks: tuple[Phase45CompletionCheck, ...]
    metadata: dict[str, object] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == Phase45CompletionStatus.PASSED

    @property
    def check_count(self) -> int:
        return len(self.checks)

    @property
    def passed_count(self) -> int:
        return sum(1 for check in self.checks if check.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for check in self.checks if not check.passed)


@dataclass(frozen=True, slots=True)
class Phase45CompletionGateConfig:
    """
    Configuration for the Phase 4.5 completion gate.
    """

    name: str = "phase45_adaptive_conversation_completion_gate"

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")


@dataclass(frozen=True, slots=True)
class Phase45CompletionGateSnapshot:
    """
    Observable diagnostics for Phase45CompletionGate.
    """

    name: str
    run_count: int
    last_status: Phase45CompletionStatus | None
    last_passed_count: int
    last_failed_count: int
    last_error: str | None


class Phase45CompletionGate:
    """
    Final completion gate for Phase 4.5 Adaptive Conversational Runtime.

    Responsibilities:
    - validate turn detection behavior
    - validate state transitions
    - validate adaptive endpointing
    - validate streaming coordination
    - validate interruption propagation
    - validate session continuity
    - validate attention routing
    - validate assembled real conversation runtime

    Non-responsibilities:
    - no microphone access
    - no real LLM calls
    - no real TTS playback
    - no tool execution
    - no direct memory writes
    """

    def __init__(
        self,
        *,
        config: Phase45CompletionGateConfig | None = None,
    ) -> None:
        self._config = config or Phase45CompletionGateConfig()
        self._config.validate()

        self._lock = RLock()
        self._logger = get_logger("conversation.phase45_completion")

        self._run_count = 0
        self._last_status: Phase45CompletionStatus | None = None
        self._last_passed_count = 0
        self._last_failed_count = 0
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def run(self) -> Phase45CompletionResult:
        """
        Run the formal Phase 4.5 completion gate.
        """

        checks = (
            self._check_turn_detection(),
            self._check_state_machine(),
            self._check_endpointing(),
            self._check_streaming(),
            self._check_interrupt_controller(),
            self._check_session_continuity(),
            self._check_attention_runtime(),
            self._check_real_runtime(),
        )
        completion_check = self._check_completion_contract(checks=checks)
        all_checks = (*checks, completion_check)

        status = (
            Phase45CompletionStatus.PASSED
            if all(check.passed for check in all_checks)
            else Phase45CompletionStatus.FAILED
        )
        result = Phase45CompletionResult(
            status=status,
            checks=all_checks,
            metadata={
                "gate": self.name,
                "phase": "phase45_adaptive_conversational_runtime",
                "validated_capability": "continuous_conversation_runtime",
            },
        )
        self._record_result(result)

        self._logger.info(
            "phase45_completion_gate_completed",
            gate=self.name,
            status=result.status.value,
            passed_count=result.passed_count,
            failed_count=result.failed_count,
        )

        return result

    def snapshot(self) -> Phase45CompletionGateSnapshot:
        """
        Return gate diagnostics.
        """

        with self._lock:
            return Phase45CompletionGateSnapshot(
                name=self.name,
                run_count=self._run_count,
                last_status=self._last_status,
                last_passed_count=self._last_passed_count,
                last_failed_count=self._last_failed_count,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Reset gate diagnostics.
        """

        with self._lock:
            self._run_count = 0
            self._last_status = None
            self._last_passed_count = 0
            self._last_failed_count = 0
            self._last_error = None

        self._logger.info("phase45_completion_gate_reset", gate=self.name)

    def _check_turn_detection(self) -> Phase45CompletionCheck:
        detector = AdaptiveTurnDetector()

        command = detector.evaluate(
            TurnDetectionInput(
                turn_id="gate-turn-command",
                transcript="run tests",
                silence_ms=500,
                conversation_mode=ConversationMode.COMMAND,
                transcript_stability=0.9,
            )
        )
        thinking_pause = detector.evaluate(
            TurnDetectionInput(
                turn_id="gate-turn-thinking",
                transcript="Jarvis I want to",
                silence_ms=700,
                conversation_mode=ConversationMode.DISCUSSION,
                transcript_stability=0.9,
            )
        )
        interrupt = detector.evaluate(
            TurnDetectionInput(
                turn_id="gate-turn-interrupt",
                transcript="stop",
                is_assistant_speaking=True,
                is_speech_active=True,
                speech_ms=300,
                vad_confidence=0.9,
            )
        )

        passed = (
            command.should_start_cognition
            and not thinking_pause.should_start_cognition
            and interrupt.should_cancel_response
        )

        return Phase45CompletionCheck(
            name="phase45_turn_detection",
            kind=Phase45CompletionCheckKind.TURN_DETECTION,
            passed=passed,
            detail=(
                "adaptive turn detection handles command, thinking pause, "
                "and interruption"
            ),
            metadata={
                "command_decision": command.decision.value,
                "thinking_pause_decision": thinking_pause.decision.value,
                "interrupt_decision": interrupt.decision.value,
            },
        )

    def _check_state_machine(self) -> Phase45CompletionCheck:
        machine = ConversationStateMachine()

        first = machine.transition(
            ConversationStateEvent(
                kind=ConversationStateEventKind.START_LISTENING,
                reason="completion gate started listening",
            )
        )
        second = machine.transition(
            ConversationStateEvent(
                kind=ConversationStateEventKind.TURN_FINALIZED,
                reason="completion gate finalized turn",
            )
        )
        third = machine.transition(
            ConversationStateEvent(
                kind=ConversationStateEventKind.RESPONSE_STARTED,
                reason="completion gate response started",
            )
        )
        fourth = machine.transition(
            ConversationStateEvent(
                kind=ConversationStateEventKind.INTERRUPTED,
                reason="completion gate interrupted response",
            )
        )

        passed = (
            first.next_state == ConversationState.LISTENING
            and second.next_state == ConversationState.THINKING
            and third.next_state == ConversationState.SPEAKING
            and fourth.next_state == ConversationState.INTERRUPTED
        )

        return Phase45CompletionCheck(
            name="phase45_state_machine",
            kind=Phase45CompletionCheckKind.STATE_MACHINE,
            passed=passed,
            detail="conversation state machine preserves behavioral continuity",
            metadata={
                "final_state": machine.current_state.value,
                "transition_count": machine.snapshot().transition_count,
            },
        )

    def _check_endpointing(self) -> Phase45CompletionCheck:
        engine = AdaptiveEndpointingEngine()

        command = engine.evaluate(
            EndpointingInput(
                signal=TurnDetectionInput(
                    turn_id="gate-endpoint-command",
                    transcript="run tests",
                    silence_ms=500,
                    conversation_mode=ConversationMode.COMMAND,
                    transcript_stability=0.9,
                ),
                conversation_state=ConversationState.LISTENING,
            )
        )
        incomplete = engine.evaluate(
            EndpointingInput(
                signal=TurnDetectionInput(
                    turn_id="gate-endpoint-incomplete",
                    transcript="Jarvis I want to",
                    silence_ms=700,
                    conversation_mode=ConversationMode.DISCUSSION,
                ),
                conversation_state=ConversationState.USER_THINKING,
            )
        )
        interrupt = engine.evaluate(
            EndpointingInput(
                signal=TurnDetectionInput(
                    turn_id="gate-endpoint-interrupt",
                    transcript="stop",
                    is_assistant_speaking=True,
                    is_speech_active=True,
                    speech_ms=300,
                    vad_confidence=0.9,
                ),
                conversation_state=ConversationState.SPEAKING,
            )
        )

        passed = (
            command.action == EndpointAction.START_COGNITION
            and incomplete.action == EndpointAction.WAIT_FOR_USER
            and interrupt.action == EndpointAction.INTERRUPT_RESPONSE
        )

        return Phase45CompletionCheck(
            name="phase45_endpointing",
            kind=Phase45CompletionCheckKind.ENDPOINTING,
            passed=passed,
            detail=(
                "adaptive endpointing chooses cognition, patience, "
                "and interruption correctly"
            ),
            metadata={
                "command_action": command.action.value,
                "incomplete_action": incomplete.action.value,
                "interrupt_action": interrupt.action.value,
            },
        )

    def _check_streaming(self) -> Phase45CompletionCheck:
        coordinator = StreamingConversationCoordinator()
        start = coordinator.accept_event(
            StreamingConversationEvent(
                turn_id="gate-stream",
                kind=StreamingEventKind.USER_TURN_FINALIZED,
                text="run tests",
            )
        )
        token = coordinator.add_cognition_token(
            turn_id="gate-stream",
            token=(
                "Memory is active and ready for continuous conversation."
            ),
        )
        completed = coordinator.complete_cognition(turn_id="gate-stream")

        passed = (
            start.should_start_cognition
            and (
                StreamingCoordinatorAction.BUFFER_TOKEN in token.actions
                or StreamingCoordinatorAction.EMIT_SPEECH_CHUNK in token.actions
            )
            and completed.lifecycle == StreamingLifecycle.COMPLETED
        )

        return Phase45CompletionCheck(
            name="phase45_streaming",
            kind=Phase45CompletionCheckKind.STREAMING,
            passed=passed,
            detail=(
                "streaming coordinator starts cognition, streams tokens, "
                "and completes response"
            ),
            metadata={
                "start_actions": tuple(action.value for action in start.actions),
                "token_actions": tuple(action.value for action in token.actions),
                "completed_actions": tuple(
                    action.value for action in completed.actions
                ),
            },
        )

    def _check_interrupt_controller(self) -> Phase45CompletionCheck:
        runtime = RealConversationRuntime()

        output = runtime.accept_input(
            RealConversationInput(
                turn_id="gate-interrupt",
                transcript="stop",
                is_assistant_speaking=True,
                is_speech_active=True,
                speech_ms=300,
                vad_confidence=0.9,
                conversation_mode=ConversationMode.COMMAND,
            )
        )

        passed = (
            output.interrupt_decision is not None
            and output.interrupt_decision.reason
            in {
                InterruptReason.USER_STOP_INTENT,
                InterruptReason.USER_BARGE_IN,
            }
            and output.should_cancel_active_work
        )

        return Phase45CompletionCheck(
            name="phase45_interrupt_controller",
            kind=Phase45CompletionCheckKind.INTERRUPTION,
            passed=passed,
            detail=(
                "interrupt controller cancels active work and preserves "
                "listening recovery"
            ),
            metadata={
                "status": output.status.value,
                "actions": tuple(action.value for action in output.actions),
                "interrupt_reason": (
                    output.interrupt_decision.reason.value
                    if output.interrupt_decision is not None
                    else None
                ),
            },
        )

    def _check_session_continuity(self) -> Phase45CompletionCheck:
        session = ConversationSessionRuntime()

        user = session.add_user_turn(
            "Jarvis explain adaptive endpointing",
            topic="adaptive endpointing",
            objective="understand",
            conversation_mode=ConversationMode.QUESTION,
        )
        assistant = session.add_assistant_turn(
            "Adaptive endpointing waits intelligently before responding.",
            expects_follow_up=True,
        )

        block = assistant.as_context_block()
        passed = (
            user.status == ConversationContinuityStatus.ACTIVE
            and assistant.turn_count == 2
            and "Conversation session continuity:" in block
            and "adaptive endpointing" in block
        )

        return Phase45CompletionCheck(
            name="phase45_session_continuity",
            kind=Phase45CompletionCheckKind.SESSION_CONTINUITY,
            passed=passed,
            detail=(
                "session runtime preserves topic, objective, turns, "
                "and follow-up continuity"
            ),
            metadata={
                "turn_count": assistant.turn_count,
                "status": assistant.status.value,
                "active_topic": assistant.active_topic,
                "follow_up": assistant.follow_up_expectation.value,
            },
        )

    def _check_attention_runtime(self) -> Phase45CompletionCheck:
        attention = AttentionRuntime()

        focus = attention.submit_signal(
            AttentionSignal(
                kind=AttentionSignalKind.USER_TURN,
                text="adaptive conversation runtime",
                priority=AttentionPriority.HIGH,
                urgency=TurnUrgency.HIGH,
            )
        )
        interrupt = attention.submit_signal(
            AttentionSignal(
                kind=AttentionSignalKind.INTERRUPTION,
                text="stop",
                priority=AttentionPriority.CRITICAL,
                urgency=TurnUrgency.CRITICAL,
                state=ConversationState.INTERRUPTED,
            )
        )

        passed = (
            focus.disposition == AttentionDisposition.FOCUS
            and interrupt.disposition == AttentionDisposition.INTERRUPT
            and interrupt.should_interrupt
        )

        return Phase45CompletionCheck(
            name="phase45_attention_runtime",
            kind=Phase45CompletionCheckKind.ATTENTION,
            passed=passed,
            detail=(
                "attention runtime focuses active user target and interrupts "
                "critical signals"
            ),
            metadata={
                "focus_disposition": focus.disposition.value,
                "interrupt_disposition": interrupt.disposition.value,
                "current_focus_id": attention.snapshot().current_focus_id,
            },
        )

    def _check_real_runtime(self) -> Phase45CompletionCheck:
        runtime = RealConversationRuntime()

        user = runtime.accept_input(
            RealConversationInput(
                turn_id="gate-real-runtime",
                transcript="How does memory gateway work?",
                silence_ms=900,
                conversation_mode=ConversationMode.QUESTION,
                transcript_stability=0.9,
            )
        )
        token = runtime.add_cognition_token(
            turn_id="gate-real-runtime",
            token="Memory gateway is the safe boundary.",
        )
        assistant = runtime.add_assistant_response(
            "Memory gateway is the safe boundary between cognition and memory.",
            turn_id="gate-real-runtime",
            expects_follow_up=True,
        )
        context = runtime.context_block()

        passed = (
            user.should_start_cognition
            and token.status
            in {
                RealConversationRuntimeStatus.STREAMING_RESPONSE,
                RealConversationRuntimeStatus.SPEAKING,
            }
            and assistant.session_snapshot is not None
            and "Conversation session continuity:" in context
            and "Attention runtime:" in context
        )

        return Phase45CompletionCheck(
            name="phase45_real_conversation_runtime",
            kind=Phase45CompletionCheckKind.REAL_RUNTIME,
            passed=passed,
            detail=(
                "assembled runtime coordinates endpointing, streaming, "
                "session, and attention"
            ),
            metadata={
                "user_status": user.status.value,
                "token_status": token.status.value,
                "assistant_status": assistant.status.value,
                "snapshot_status": runtime.snapshot().status.value,
            },
        )

    def _check_completion_contract(
        self,
        *,
        checks: tuple[Phase45CompletionCheck, ...],
    ) -> Phase45CompletionCheck:
        failed = tuple(check.name for check in checks if not check.passed)
        passed = not failed

        return Phase45CompletionCheck(
            name="phase45_completion_contract",
            kind=Phase45CompletionCheckKind.COMPLETION,
            passed=passed,
            detail=(
                "Phase 4.5 Adaptive Conversational Runtime is complete"
                if passed
                else "Phase 4.5 has failed prerequisite checks"
            ),
            metadata={
                "failed_prerequisites": failed,
                "validated_capabilities": (
                    "turn_detection",
                    "conversation_state_machine",
                    "adaptive_endpointing",
                    "streaming_conversation",
                    "interrupt_controller",
                    "session_continuity",
                    "attention_runtime",
                    "real_conversation_runtime_assembly",
                ),
            },
        )

    def _record_result(self, result: Phase45CompletionResult) -> None:
        with self._lock:
            self._run_count += 1
            self._last_status = result.status
            self._last_passed_count = result.passed_count
            self._last_failed_count = result.failed_count
            self._last_error = (
                None
                if result.passed
                else f"phase45 completion failed: {result.failed_count} checks"
            )


def complete_phase45_conversation() -> Phase45CompletionResult:
    """
    Convenience function for scripts and tests.
    """

    gate = Phase45CompletionGate()

    return gate.run()