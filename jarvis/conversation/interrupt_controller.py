from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from threading import RLock

from pydantic import Field, field_validator

from jarvis.conversation.endpointing import EndpointAction, EndpointingDecision
from jarvis.conversation.models import (
    ConversationModel,
    InterruptionIntent,
    new_conversation_id,
    utc_now,
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
    StreamingEventKind,
)
from jarvis.runtime.observability.structured_logger import get_logger


class InterruptPriority(StrEnum):
    """
    Priority level for interruption handling.
    """

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class InterruptScope(StrEnum):
    """
    Scope of runtime work affected by an interruption.
    """

    SPEECH_ONLY = "speech_only"
    COGNITION_ONLY = "cognition_only"
    SPEECH_AND_COGNITION = "speech_and_cognition"
    ALL_ACTIVE_WORK = "all_active_work"


class InterruptAction(StrEnum):
    """
    Runtime action requested by InterruptController.
    """

    NONE = "none"
    PAUSE_SPEECH = "pause_speech"
    CANCEL_SPEECH = "cancel_speech"
    CANCEL_COGNITION = "cancel_cognition"
    CANCEL_STREAMING = "cancel_streaming"
    CANCEL_TOOLS = "cancel_tools"
    RETURN_TO_LISTENING = "return_to_listening"
    REQUEST_CLARIFICATION = "request_clarification"


class InterruptReason(StrEnum):
    """
    Reason for an interrupt decision.
    """

    NO_INTERRUPT = "no_interrupt"
    USER_BARGE_IN = "user_barge_in"
    USER_STOP_INTENT = "user_stop_intent"
    USER_CANCEL_INTENT = "user_cancel_intent"
    USER_PAUSE_INTENT = "user_pause_intent"
    USER_WAIT_INTENT = "user_wait_intent"
    USER_CORRECTION = "user_correction"
    ENDPOINTING_CANCEL = "endpointing_cancel"
    STREAMING_CANCEL = "streaming_cancel"


class InterruptRequest(ConversationModel):
    """
    Explicit interruption request.

    Future workers can create this from VAD/STT/barge-in signals, endpointing
    decisions, or tool cancellation requests.
    """

    interrupt_id: str = Field(default_factory=new_conversation_id)
    turn_id: str
    transcript: str = ""
    intent: InterruptionIntent = InterruptionIntent.NONE
    priority: InterruptPriority = InterruptPriority.NORMAL
    scope: InterruptScope = InterruptScope.SPEECH_AND_COGNITION
    assistant_was_speaking: bool = False
    cognition_was_active: bool = False
    tools_were_active: bool = False
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("interrupt_id", "turn_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("transcript")
    @classmethod
    def _clean_transcript(cls, value: str) -> str:
        return " ".join(value.strip().split())


class InterruptDecision(ConversationModel):
    """
    Output of InterruptController.

    This is orchestration-ready: future workers can consume actions to cancel
    TTS, cognition streams, tool work, and return to listening.
    """

    interrupt_id: str
    turn_id: str
    interrupted: bool
    reason: InterruptReason
    priority: InterruptPriority
    scope: InterruptScope
    actions: tuple[InterruptAction, ...]
    should_cancel_speech: bool
    should_cancel_cognition: bool
    should_cancel_streaming: bool
    should_cancel_tools: bool
    should_return_to_listening: bool
    should_request_clarification: bool = False
    decided_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("interrupt_id", "turn_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @property
    def cancelled_anything(self) -> bool:
        return (
            self.should_cancel_speech
            or self.should_cancel_cognition
            or self.should_cancel_streaming
            or self.should_cancel_tools
        )


@dataclass(frozen=True, slots=True)
class InterruptControllerConfig:
    """
    Configuration for InterruptController.
    """

    name: str = "interrupt_controller"
    cancel_tools_on_critical: bool = True
    return_to_listening_after_interrupt: bool = True
    correction_requests_clarification: bool = False

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")


@dataclass(frozen=True, slots=True)
class InterruptControllerSnapshot:
    """
    Observable diagnostics for InterruptController.
    """

    name: str
    interrupt_count: int
    ignored_count: int
    speech_cancel_count: int
    cognition_cancel_count: int
    streaming_cancel_count: int
    tool_cancel_count: int
    return_to_listening_count: int
    last_reason: InterruptReason | None
    last_priority: InterruptPriority | None
    last_turn_id: str | None
    last_error: str | None


class InterruptController:
    """
    Runtime interrupt controller for real-time conversation.

    Responsibilities:
    - convert endpointing interruption into cancellation decisions
    - propagate cancellation to speech, cognition, streaming, and future tools
    - coordinate state machine recovery
    - coordinate streaming coordinator cancellation
    - support barge-in and explicit stop/cancel/pause/wait intents
    - expose diagnostics

    Non-responsibilities:
    - no actual TTS implementation
    - no actual LLM cancellation primitive
    - no tool execution
    - no microphone/VAD/STT implementation
    """

    def __init__(
        self,
        *,
        config: InterruptControllerConfig | None = None,
        state_machine: ConversationStateMachine | None = None,
        streaming_coordinator: StreamingConversationCoordinator | None = None,
    ) -> None:
        self._config = config or InterruptControllerConfig()
        self._config.validate()

        self._state_machine = state_machine
        self._streaming_coordinator = streaming_coordinator

        self._lock = RLock()
        self._logger = get_logger("conversation.interrupt_controller")

        self._interrupt_count = 0
        self._ignored_count = 0
        self._speech_cancel_count = 0
        self._cognition_cancel_count = 0
        self._streaming_cancel_count = 0
        self._tool_cancel_count = 0
        self._return_to_listening_count = 0
        self._last_reason: InterruptReason | None = None
        self._last_priority: InterruptPriority | None = None
        self._last_turn_id: str | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def handle_request(self, request: InterruptRequest) -> InterruptDecision:
        """
        Handle one explicit interruption request.
        """

        with self._lock:
            self._last_turn_id = request.turn_id
            self._last_error = None

        try:
            decision = self._decide(request)
            self._record(decision)
            self._apply_side_effect_boundaries(decision)

            self._logger.info(
                "conversation_interrupt_decided",
                controller=self.name,
                turn_id=request.turn_id,
                interrupted=decision.interrupted,
                reason=decision.reason.value,
                priority=decision.priority.value,
                actions=tuple(action.value for action in decision.actions),
            )

            return decision

        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"

            raise

    def handle_endpointing_decision(
        self,
        decision: EndpointingDecision,
    ) -> InterruptDecision:
        """
        Convert an endpointing decision into an interrupt request.
        """

        turn_decision = decision.turn_decision
        request = InterruptRequest(
            turn_id=turn_decision.turn_id,
            transcript=turn_decision.transcript,
            intent=turn_decision.interruption_intent,
            priority=self._priority_from_endpointing(decision),
            scope=self._scope_from_endpointing(decision),
            assistant_was_speaking=(
                decision.conversation_state == ConversationState.SPEAKING
            ),
            cognition_was_active=(
                decision.conversation_state == ConversationState.THINKING
            ),
            tools_were_active=False,
            metadata={
                "endpoint_action": decision.action.value,
                "turn_decision": turn_decision.decision.value,
                "endpoint_reason": decision.reason,
            },
        )

        return self.handle_request(request)

    def interrupt_from_streaming_event(
        self,
        event: StreamingConversationEvent,
    ) -> InterruptDecision:
        """
        Convert a streaming cancellation/interruption event into a decision.
        """

        intent = (
            InterruptionIntent.CANCEL
            if event.kind == StreamingEventKind.CANCEL_REQUESTED
            else InterruptionIntent.BARGE_IN
        )

        return self.handle_request(
            InterruptRequest(
                turn_id=event.turn_id,
                transcript=event.text,
                intent=intent,
                priority=InterruptPriority.CRITICAL,
                scope=InterruptScope.SPEECH_AND_COGNITION,
                assistant_was_speaking=True,
                cognition_was_active=True,
                metadata={
                    "streaming_event_kind": event.kind.value,
                },
            )
        )

    def snapshot(self) -> InterruptControllerSnapshot:
        """
        Return interrupt controller diagnostics.
        """

        with self._lock:
            return InterruptControllerSnapshot(
                name=self.name,
                interrupt_count=self._interrupt_count,
                ignored_count=self._ignored_count,
                speech_cancel_count=self._speech_cancel_count,
                cognition_cancel_count=self._cognition_cancel_count,
                streaming_cancel_count=self._streaming_cancel_count,
                tool_cancel_count=self._tool_cancel_count,
                return_to_listening_count=self._return_to_listening_count,
                last_reason=self._last_reason,
                last_priority=self._last_priority,
                last_turn_id=self._last_turn_id,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Reset diagnostics. Does not reset injected collaborators.
        """

        with self._lock:
            self._interrupt_count = 0
            self._ignored_count = 0
            self._speech_cancel_count = 0
            self._cognition_cancel_count = 0
            self._streaming_cancel_count = 0
            self._tool_cancel_count = 0
            self._return_to_listening_count = 0
            self._last_reason = None
            self._last_priority = None
            self._last_turn_id = None
            self._last_error = None

        self._logger.info("interrupt_controller_reset", controller=self.name)

    def _decide(self, request: InterruptRequest) -> InterruptDecision:
        if request.intent == InterruptionIntent.NONE:
            return InterruptDecision(
                interrupt_id=request.interrupt_id,
                turn_id=request.turn_id,
                interrupted=False,
                reason=InterruptReason.NO_INTERRUPT,
                priority=InterruptPriority.LOW,
                scope=request.scope,
                actions=(InterruptAction.NONE,),
                should_cancel_speech=False,
                should_cancel_cognition=False,
                should_cancel_streaming=False,
                should_cancel_tools=False,
                should_return_to_listening=False,
                metadata={
                    "controller": self.name,
                },
            )

        reason = self._reason_from_intent(request.intent)
        actions = self._actions_for_request(request)
        should_cancel_speech = InterruptAction.CANCEL_SPEECH in actions
        should_cancel_cognition = InterruptAction.CANCEL_COGNITION in actions
        should_cancel_streaming = InterruptAction.CANCEL_STREAMING in actions
        should_cancel_tools = InterruptAction.CANCEL_TOOLS in actions
        should_return_to_listening = (
            InterruptAction.RETURN_TO_LISTENING in actions
        )
        should_request_clarification = (
            InterruptAction.REQUEST_CLARIFICATION in actions
        )

        return InterruptDecision(
            interrupt_id=request.interrupt_id,
            turn_id=request.turn_id,
            interrupted=True,
            reason=reason,
            priority=request.priority,
            scope=request.scope,
            actions=actions,
            should_cancel_speech=should_cancel_speech,
            should_cancel_cognition=should_cancel_cognition,
            should_cancel_streaming=should_cancel_streaming,
            should_cancel_tools=should_cancel_tools,
            should_return_to_listening=should_return_to_listening,
            should_request_clarification=should_request_clarification,
            metadata={
                "controller": self.name,
                "assistant_was_speaking": request.assistant_was_speaking,
                "cognition_was_active": request.cognition_was_active,
                "tools_were_active": request.tools_were_active,
                "transcript": request.transcript,
            },
        )

    def _actions_for_request(
        self,
        request: InterruptRequest,
    ) -> tuple[InterruptAction, ...]:
        actions: list[InterruptAction] = []

        if request.intent == InterruptionIntent.PAUSE:
            actions.append(InterruptAction.PAUSE_SPEECH)

        if request.scope in {
            InterruptScope.SPEECH_ONLY,
            InterruptScope.SPEECH_AND_COGNITION,
            InterruptScope.ALL_ACTIVE_WORK,
        }:
            actions.append(InterruptAction.CANCEL_SPEECH)

        if request.scope in {
            InterruptScope.COGNITION_ONLY,
            InterruptScope.SPEECH_AND_COGNITION,
            InterruptScope.ALL_ACTIVE_WORK,
        }:
            actions.append(InterruptAction.CANCEL_COGNITION)

        if request.scope in {
            InterruptScope.SPEECH_AND_COGNITION,
            InterruptScope.ALL_ACTIVE_WORK,
        }:
            actions.append(InterruptAction.CANCEL_STREAMING)

        if (
            request.scope == InterruptScope.ALL_ACTIVE_WORK
            or (
                request.priority == InterruptPriority.CRITICAL
                and request.tools_were_active
                and self._config.cancel_tools_on_critical
            )
        ):
            actions.append(InterruptAction.CANCEL_TOOLS)

        if (
            request.intent == InterruptionIntent.CORRECTION
            and self._config.correction_requests_clarification
        ):
            actions.append(InterruptAction.REQUEST_CLARIFICATION)

        if self._config.return_to_listening_after_interrupt:
            actions.append(InterruptAction.RETURN_TO_LISTENING)

        return tuple(dict.fromkeys(actions)) or (InterruptAction.NONE,)

    @staticmethod
    def _reason_from_intent(intent: InterruptionIntent) -> InterruptReason:
        if intent == InterruptionIntent.CANCEL:
            return InterruptReason.USER_CANCEL_INTENT

        if intent == InterruptionIntent.STOP:
            return InterruptReason.USER_STOP_INTENT

        if intent == InterruptionIntent.PAUSE:
            return InterruptReason.USER_PAUSE_INTENT

        if intent == InterruptionIntent.WAIT:
            return InterruptReason.USER_WAIT_INTENT

        if intent == InterruptionIntent.CORRECTION:
            return InterruptReason.USER_CORRECTION

        if intent == InterruptionIntent.BARGE_IN:
            return InterruptReason.USER_BARGE_IN

        return InterruptReason.NO_INTERRUPT

    @staticmethod
    def _priority_from_endpointing(
        decision: EndpointingDecision,
    ) -> InterruptPriority:
        if decision.action == EndpointAction.CANCEL_RESPONSE:
            return InterruptPriority.CRITICAL

        if decision.action == EndpointAction.INTERRUPT_RESPONSE:
            return InterruptPriority.HIGH

        return InterruptPriority.LOW

    @staticmethod
    def _scope_from_endpointing(
        decision: EndpointingDecision,
    ) -> InterruptScope:
        if decision.action == EndpointAction.CANCEL_RESPONSE:
            return InterruptScope.ALL_ACTIVE_WORK

        if decision.action == EndpointAction.INTERRUPT_RESPONSE:
            return InterruptScope.SPEECH_AND_COGNITION

        return InterruptScope.SPEECH_ONLY

    def _apply_side_effect_boundaries(
        self,
        decision: InterruptDecision,
    ) -> None:
        """
        Apply only boundary-level coordinator/state changes.

        Real worker cancellation will be implemented by future runtime workers.
        This method safely coordinates existing Step 1 and Step 3 components.
        """

        if not decision.interrupted:
            return

        if self._streaming_coordinator is not None:
            event_kind = (
                StreamingEventKind.CANCEL_REQUESTED
                if decision.reason == InterruptReason.USER_CANCEL_INTENT
                else StreamingEventKind.INTERRUPT_REQUESTED
            )
            self._streaming_coordinator.accept_event(
                StreamingConversationEvent(
                    turn_id=decision.turn_id,
                    kind=event_kind,
                    text=decision.reason.value,
                    metadata={
                        "interrupt_id": decision.interrupt_id,
                    },
                )
            )

        if self._state_machine is not None:
            self._state_machine.transition(
                ConversationStateEvent(
                    turn_id=decision.turn_id,
                    kind=ConversationStateEventKind.INTERRUPTED,
                    reason="interrupt controller moved runtime to interrupted",
                    metadata={
                        "interrupt_id": decision.interrupt_id,
                        "interrupt_reason": decision.reason.value,
                    },
                )
            )

            if decision.should_return_to_listening:
                self._state_machine.transition(
                    ConversationStateEvent(
                        turn_id=decision.turn_id,
                        kind=ConversationStateEventKind.START_LISTENING,
                        reason="interrupt controller returned to listening",
                        metadata={
                            "interrupt_id": decision.interrupt_id,
                        },
                    )
                )

    def _record(self, decision: InterruptDecision) -> None:
        with self._lock:
            self._last_reason = decision.reason
            self._last_priority = decision.priority
            self._last_turn_id = decision.turn_id

            if not decision.interrupted:
                self._ignored_count += 1
                return

            self._interrupt_count += 1

            if decision.should_cancel_speech:
                self._speech_cancel_count += 1

            if decision.should_cancel_cognition:
                self._cognition_cancel_count += 1

            if decision.should_cancel_streaming:
                self._streaming_cancel_count += 1

            if decision.should_cancel_tools:
                self._tool_cancel_count += 1

            if decision.should_return_to_listening:
                self._return_to_listening_count += 1