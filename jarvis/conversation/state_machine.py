from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import RLock

from pydantic import Field, field_validator

from jarvis.conversation.models import (
    ConversationModel,
    TurnDecisionKind,
    TurnDetectionDecision,
    new_conversation_id,
    utc_now,
)
from jarvis.runtime.observability.structured_logger import get_logger


class ConversationState(StrEnum):
    """
    Runtime state for continuous adaptive conversation.

    These states make the assistant behavior explicit instead of chaotic.
    """

    IDLE = "idle"
    LISTENING = "listening"
    USER_THINKING = "user_thinking"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    WAITING = "waiting"
    FOLLOW_UP = "follow_up"
    BACKGROUND_REASONING = "background_reasoning"


class ConversationStateEventKind(StrEnum):
    """
    Event that can move the conversation state machine.
    """

    START_LISTENING = "start_listening"
    SPEECH_STARTED = "speech_started"
    USER_PAUSED = "user_paused"
    TURN_MAYBE_COMPLETE = "turn_maybe_complete"
    TURN_FINALIZED = "turn_finalized"
    COGNITION_STARTED = "cognition_started"
    RESPONSE_STARTED = "response_started"
    RESPONSE_COMPLETED = "response_completed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"
    WAIT_FOR_USER = "wait_for_user"
    FOLLOW_UP_REQUESTED = "follow_up_requested"
    BACKGROUND_STARTED = "background_started"
    BACKGROUND_COMPLETED = "background_completed"
    TIMEOUT = "timeout"
    RESET = "reset"


class ConversationStateTransitionStatus(StrEnum):
    """
    Result status for one state transition.
    """

    APPLIED = "applied"
    IGNORED = "ignored"
    REJECTED = "rejected"


class ConversationStateEvent(ConversationModel):
    """
    One event submitted to the conversation state machine.
    """

    event_id: str = Field(default_factory=new_conversation_id)
    turn_id: str | None = None
    kind: ConversationStateEventKind
    reason: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("event_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("turn_id")
    @classmethod
    def _clean_optional_turn_id(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None


class ConversationStateTransition(ConversationModel):
    """
    One transition result from the conversation state machine.
    """

    transition_id: str = Field(default_factory=new_conversation_id)
    event: ConversationStateEvent
    previous_state: ConversationState
    next_state: ConversationState
    status: ConversationStateTransitionStatus
    reason: str
    changed: bool
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("transition_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class ConversationStateMachineConfig:
    """
    Configuration for ConversationStateMachine.
    """

    name: str = "conversation_state_machine"
    initial_state: ConversationState = ConversationState.IDLE
    reject_invalid_transitions: bool = False

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")


@dataclass(frozen=True, slots=True)
class ConversationStateMachineSnapshot:
    """
    Observable diagnostics for the conversation state machine.
    """

    name: str
    current_state: ConversationState
    transition_count: int
    applied_count: int
    ignored_count: int
    rejected_count: int
    interruption_count: int
    last_event_kind: ConversationStateEventKind | None
    last_transition_status: ConversationStateTransitionStatus | None
    last_error: str | None


class ConversationStateMachine:
    """
    State machine for continuous adaptive conversation.

    Responsibilities:
    - keep one explicit current conversation state
    - make transitions deterministic and observable
    - prevent chaotic behavior during listening/thinking/speaking
    - support interruption and recovery
    - convert turn-detection decisions into state events

    Non-responsibilities:
    - no microphone access
    - no STT
    - no LLM calls
    - no TTS playback
    - no tool execution
    """

    _TRANSITIONS: dict[
        ConversationState,
        dict[ConversationStateEventKind, ConversationState],
    ] = {
        ConversationState.IDLE: {
            ConversationStateEventKind.START_LISTENING: (
                ConversationState.LISTENING
            ),
            ConversationStateEventKind.SPEECH_STARTED: (
                ConversationState.LISTENING
            ),
            ConversationStateEventKind.BACKGROUND_STARTED: (
                ConversationState.BACKGROUND_REASONING
            ),
            ConversationStateEventKind.RESET: ConversationState.IDLE,
        },
        ConversationState.LISTENING: {
            ConversationStateEventKind.USER_PAUSED: (
                ConversationState.USER_THINKING
            ),
            ConversationStateEventKind.TURN_MAYBE_COMPLETE: (
                ConversationState.USER_THINKING
            ),
            ConversationStateEventKind.TURN_FINALIZED: (
                ConversationState.THINKING
            ),
            ConversationStateEventKind.INTERRUPTED: (
                ConversationState.INTERRUPTED
            ),
            ConversationStateEventKind.CANCELLED: (
                ConversationState.INTERRUPTED
            ),
            ConversationStateEventKind.TIMEOUT: ConversationState.WAITING,
            ConversationStateEventKind.RESET: ConversationState.IDLE,
        },
        ConversationState.USER_THINKING: {
            ConversationStateEventKind.SPEECH_STARTED: (
                ConversationState.LISTENING
            ),
            ConversationStateEventKind.TURN_FINALIZED: (
                ConversationState.THINKING
            ),
            ConversationStateEventKind.TURN_MAYBE_COMPLETE: (
                ConversationState.USER_THINKING
            ),
            ConversationStateEventKind.WAIT_FOR_USER: (
                ConversationState.WAITING
            ),
            ConversationStateEventKind.INTERRUPTED: (
                ConversationState.INTERRUPTED
            ),
            ConversationStateEventKind.CANCELLED: (
                ConversationState.INTERRUPTED
            ),
            ConversationStateEventKind.TIMEOUT: ConversationState.WAITING,
            ConversationStateEventKind.RESET: ConversationState.IDLE,
        },
        ConversationState.THINKING: {
            ConversationStateEventKind.COGNITION_STARTED: (
                ConversationState.THINKING
            ),
            ConversationStateEventKind.RESPONSE_STARTED: (
                ConversationState.SPEAKING
            ),
            ConversationStateEventKind.WAIT_FOR_USER: (
                ConversationState.WAITING
            ),
            ConversationStateEventKind.INTERRUPTED: (
                ConversationState.INTERRUPTED
            ),
            ConversationStateEventKind.CANCELLED: (
                ConversationState.INTERRUPTED
            ),
            ConversationStateEventKind.BACKGROUND_STARTED: (
                ConversationState.BACKGROUND_REASONING
            ),
            ConversationStateEventKind.RESET: ConversationState.IDLE,
        },
        ConversationState.SPEAKING: {
            ConversationStateEventKind.RESPONSE_COMPLETED: (
                ConversationState.FOLLOW_UP
            ),
            ConversationStateEventKind.INTERRUPTED: (
                ConversationState.INTERRUPTED
            ),
            ConversationStateEventKind.CANCELLED: (
                ConversationState.INTERRUPTED
            ),
            ConversationStateEventKind.SPEECH_STARTED: (
                ConversationState.INTERRUPTED
            ),
            ConversationStateEventKind.RESET: ConversationState.IDLE,
        },
        ConversationState.INTERRUPTED: {
            ConversationStateEventKind.SPEECH_STARTED: (
                ConversationState.LISTENING
            ),
            ConversationStateEventKind.START_LISTENING: (
                ConversationState.LISTENING
            ),
            ConversationStateEventKind.WAIT_FOR_USER: (
                ConversationState.WAITING
            ),
            ConversationStateEventKind.TURN_FINALIZED: (
                ConversationState.THINKING
            ),
            ConversationStateEventKind.RESET: ConversationState.IDLE,
        },
        ConversationState.WAITING: {
            ConversationStateEventKind.SPEECH_STARTED: (
                ConversationState.LISTENING
            ),
            ConversationStateEventKind.START_LISTENING: (
                ConversationState.LISTENING
            ),
            ConversationStateEventKind.FOLLOW_UP_REQUESTED: (
                ConversationState.FOLLOW_UP
            ),
            ConversationStateEventKind.TIMEOUT: ConversationState.IDLE,
            ConversationStateEventKind.RESET: ConversationState.IDLE,
        },
        ConversationState.FOLLOW_UP: {
            ConversationStateEventKind.SPEECH_STARTED: (
                ConversationState.LISTENING
            ),
            ConversationStateEventKind.START_LISTENING: (
                ConversationState.LISTENING
            ),
            ConversationStateEventKind.WAIT_FOR_USER: (
                ConversationState.WAITING
            ),
            ConversationStateEventKind.TIMEOUT: ConversationState.IDLE,
            ConversationStateEventKind.RESET: ConversationState.IDLE,
        },
        ConversationState.BACKGROUND_REASONING: {
            ConversationStateEventKind.BACKGROUND_COMPLETED: (
                ConversationState.WAITING
            ),
            ConversationStateEventKind.SPEECH_STARTED: (
                ConversationState.LISTENING
            ),
            ConversationStateEventKind.INTERRUPTED: (
                ConversationState.INTERRUPTED
            ),
            ConversationStateEventKind.CANCELLED: (
                ConversationState.INTERRUPTED
            ),
            ConversationStateEventKind.RESET: ConversationState.IDLE,
        },
    }

    def __init__(
        self,
        *,
        config: ConversationStateMachineConfig | None = None,
    ) -> None:
        self._config = config or ConversationStateMachineConfig()
        self._config.validate()

        self._lock = RLock()
        self._logger = get_logger("conversation.state_machine")

        self._state = self._config.initial_state
        self._transition_count = 0
        self._applied_count = 0
        self._ignored_count = 0
        self._rejected_count = 0
        self._interruption_count = 0
        self._last_event_kind: ConversationStateEventKind | None = None
        self._last_transition_status: ConversationStateTransitionStatus | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def current_state(self) -> ConversationState:
        with self._lock:
            return self._state

    def transition(
        self,
        event: ConversationStateEvent,
    ) -> ConversationStateTransition:
        """
        Apply one state event.
        """

        with self._lock:
            previous = self._state
            self._transition_count += 1
            self._last_event_kind = event.kind
            self._last_error = None

            next_state = self._next_state(
                state=previous,
                event_kind=event.kind,
            )

            if next_state is None:
                transition = self._invalid_transition(
                    event=event,
                    previous=previous,
                )
                self._record_transition(transition)

                if self._config.reject_invalid_transitions:
                    self._last_error = transition.reason
                    raise ValueError(transition.reason)

                return transition

            changed = next_state != previous
            status = (
                ConversationStateTransitionStatus.APPLIED
                if changed
                else ConversationStateTransitionStatus.IGNORED
            )
            reason = (
                f"conversation state changed {previous.value} -> "
                f"{next_state.value}"
                if changed
                else f"conversation state remained {previous.value}"
            )

            self._state = next_state
            transition = ConversationStateTransition(
                event=event,
                previous_state=previous,
                next_state=next_state,
                status=status,
                reason=reason,
                changed=changed,
                metadata={
                    "machine": self.name,
                },
            )
            self._record_transition(transition)

        self._logger.info(
            "conversation_state_transitioned",
            state_machine=self.name,
            event_kind=event.kind.value,
            previous_state=transition.previous_state.value,
            next_state=transition.next_state.value,
            status=transition.status.value,
            changed=transition.changed,
        )

        return transition

    def apply_turn_decision(
        self,
        decision: TurnDetectionDecision,
    ) -> ConversationStateTransition:
        """
        Convert a turn-detection decision into a state transition.
        """

        event = self._event_from_turn_decision(decision)

        return self.transition(event)

    def reset(self) -> None:
        """
        Reset state machine to the configured initial state.
        """

        with self._lock:
            self._state = self._config.initial_state
            self._transition_count = 0
            self._applied_count = 0
            self._ignored_count = 0
            self._rejected_count = 0
            self._interruption_count = 0
            self._last_event_kind = None
            self._last_transition_status = None
            self._last_error = None

        self._logger.info("conversation_state_machine_reset", machine=self.name)

    def snapshot(self) -> ConversationStateMachineSnapshot:
        """
        Return observable state machine diagnostics.
        """

        with self._lock:
            return ConversationStateMachineSnapshot(
                name=self.name,
                current_state=self._state,
                transition_count=self._transition_count,
                applied_count=self._applied_count,
                ignored_count=self._ignored_count,
                rejected_count=self._rejected_count,
                interruption_count=self._interruption_count,
                last_event_kind=self._last_event_kind,
                last_transition_status=self._last_transition_status,
                last_error=self._last_error,
            )

    def can_transition(
        self,
        *,
        state: ConversationState,
        event_kind: ConversationStateEventKind,
    ) -> bool:
        """
        Return whether a transition is allowed.
        """

        return self._next_state(state=state, event_kind=event_kind) is not None

    def _next_state(
        self,
        *,
        state: ConversationState,
        event_kind: ConversationStateEventKind,
    ) -> ConversationState | None:
        return self._TRANSITIONS.get(state, {}).get(event_kind)

    def _invalid_transition(
        self,
        *,
        event: ConversationStateEvent,
        previous: ConversationState,
    ) -> ConversationStateTransition:
        return ConversationStateTransition(
            event=event,
            previous_state=previous,
            next_state=previous,
            status=ConversationStateTransitionStatus.REJECTED,
            reason=(
                f"invalid conversation transition from {previous.value} "
                f"using event {event.kind.value}"
            ),
            changed=False,
            metadata={
                "machine": self.name,
                "invalid": True,
            },
        )

    def _record_transition(
        self,
        transition: ConversationStateTransition,
    ) -> None:
        self._last_transition_status = transition.status

        if transition.status == ConversationStateTransitionStatus.APPLIED:
            self._applied_count += 1

        elif transition.status == ConversationStateTransitionStatus.IGNORED:
            self._ignored_count += 1

        else:
            self._rejected_count += 1

        if transition.next_state == ConversationState.INTERRUPTED:
            self._interruption_count += 1

    @staticmethod
    def _event_from_turn_decision(
        decision: TurnDetectionDecision,
    ) -> ConversationStateEvent:
        if decision.decision == TurnDecisionKind.FINALIZE:
            kind = ConversationStateEventKind.TURN_FINALIZED
            reason = "turn detector finalized user turn"

        elif decision.decision == TurnDecisionKind.MAYBE_COMPLETE:
            kind = ConversationStateEventKind.TURN_MAYBE_COMPLETE
            reason = "turn detector marked turn maybe complete"

        elif decision.decision == TurnDecisionKind.INTERRUPT:
            kind = ConversationStateEventKind.INTERRUPTED
            reason = "turn detector detected interruption"

        elif decision.decision == TurnDecisionKind.CANCEL:
            kind = ConversationStateEventKind.CANCELLED
            reason = "turn detector detected cancellation"

        else:
            kind = ConversationStateEventKind.USER_PAUSED
            reason = "turn detector is waiting for more user input"

        return ConversationStateEvent(
            turn_id=decision.turn_id,
            kind=kind,
            reason=reason,
            metadata={
                "turn_decision": decision.decision.value,
                "turn_reason": decision.reason.value,
                "should_start_cognition": decision.should_start_cognition,
                "should_cancel_response": decision.should_cancel_response,
            },
        )