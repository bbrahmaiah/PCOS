from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.conversation import (
    AdaptiveTurnDetector,
    ConversationState,
    ConversationStateEvent,
    ConversationStateEventKind,
    ConversationStateMachine,
    ConversationStateMachineConfig,
    ConversationStateTransitionStatus,
    TurnDecisionKind,
    TurnDetectionInput,
)


def event(kind: ConversationStateEventKind) -> ConversationStateEvent:
    return ConversationStateEvent(
        kind=kind,
        reason=f"test event {kind.value}",
    )


def test_conversation_state_machine_config_rejects_invalid_name() -> None:
    with pytest.raises(ValueError):
        ConversationStateMachineConfig(name=" ").validate()


def test_conversation_state_event_requires_reason() -> None:
    with pytest.raises(ValidationError):
        ConversationStateEvent(
            kind=ConversationStateEventKind.START_LISTENING,
            reason=" ",
        )


def test_state_machine_starts_idle() -> None:
    machine = ConversationStateMachine()

    assert machine.current_state == ConversationState.IDLE


def test_state_machine_moves_idle_to_listening() -> None:
    machine = ConversationStateMachine()

    transition = machine.transition(
        event(ConversationStateEventKind.START_LISTENING)
    )

    assert transition.status == ConversationStateTransitionStatus.APPLIED
    assert transition.previous_state == ConversationState.IDLE
    assert transition.next_state == ConversationState.LISTENING
    assert machine.current_state == ConversationState.LISTENING


def test_state_machine_listening_to_user_thinking() -> None:
    machine = ConversationStateMachine()
    machine.transition(event(ConversationStateEventKind.START_LISTENING))

    transition = machine.transition(
        event(ConversationStateEventKind.USER_PAUSED)
    )

    assert transition.next_state == ConversationState.USER_THINKING
    assert machine.current_state == ConversationState.USER_THINKING


def test_state_machine_user_thinking_back_to_listening() -> None:
    machine = ConversationStateMachine()
    machine.transition(event(ConversationStateEventKind.START_LISTENING))
    machine.transition(event(ConversationStateEventKind.USER_PAUSED))

    transition = machine.transition(
        event(ConversationStateEventKind.SPEECH_STARTED)
    )

    assert transition.next_state == ConversationState.LISTENING


def test_state_machine_finalized_turn_enters_thinking() -> None:
    machine = ConversationStateMachine()
    machine.transition(event(ConversationStateEventKind.START_LISTENING))

    transition = machine.transition(
        event(ConversationStateEventKind.TURN_FINALIZED)
    )

    assert transition.next_state == ConversationState.THINKING


def test_state_machine_thinking_to_speaking_to_follow_up() -> None:
    machine = ConversationStateMachine()
    machine.transition(event(ConversationStateEventKind.START_LISTENING))
    machine.transition(event(ConversationStateEventKind.TURN_FINALIZED))

    speaking = machine.transition(
        event(ConversationStateEventKind.RESPONSE_STARTED)
    )
    follow_up = machine.transition(
        event(ConversationStateEventKind.RESPONSE_COMPLETED)
    )

    assert speaking.next_state == ConversationState.SPEAKING
    assert follow_up.next_state == ConversationState.FOLLOW_UP
    assert machine.current_state == ConversationState.FOLLOW_UP


def test_state_machine_interruption_from_speaking() -> None:
    machine = ConversationStateMachine()
    machine.transition(event(ConversationStateEventKind.START_LISTENING))
    machine.transition(event(ConversationStateEventKind.TURN_FINALIZED))
    machine.transition(event(ConversationStateEventKind.RESPONSE_STARTED))

    transition = machine.transition(
        event(ConversationStateEventKind.INTERRUPTED)
    )
    snapshot = machine.snapshot()

    assert transition.next_state == ConversationState.INTERRUPTED
    assert snapshot.interruption_count == 1


def test_state_machine_recovers_from_interruption_to_listening() -> None:
    machine = ConversationStateMachine()
    machine.transition(event(ConversationStateEventKind.START_LISTENING))
    machine.transition(event(ConversationStateEventKind.TURN_FINALIZED))
    machine.transition(event(ConversationStateEventKind.RESPONSE_STARTED))
    machine.transition(event(ConversationStateEventKind.INTERRUPTED))

    transition = machine.transition(
        event(ConversationStateEventKind.SPEECH_STARTED)
    )

    assert transition.next_state == ConversationState.LISTENING


def test_state_machine_waiting_timeout_returns_idle() -> None:
    machine = ConversationStateMachine(
        config=ConversationStateMachineConfig(
            initial_state=ConversationState.WAITING
        )
    )

    transition = machine.transition(event(ConversationStateEventKind.TIMEOUT))

    assert transition.next_state == ConversationState.IDLE


def test_state_machine_background_reasoning_flow() -> None:
    machine = ConversationStateMachine()

    background = machine.transition(
        event(ConversationStateEventKind.BACKGROUND_STARTED)
    )
    completed = machine.transition(
        event(ConversationStateEventKind.BACKGROUND_COMPLETED)
    )

    assert background.next_state == ConversationState.BACKGROUND_REASONING
    assert completed.next_state == ConversationState.WAITING


def test_invalid_transition_is_rejected_without_raise_by_default() -> None:
    machine = ConversationStateMachine()

    transition = machine.transition(
        event(ConversationStateEventKind.RESPONSE_STARTED)
    )

    assert transition.status == ConversationStateTransitionStatus.REJECTED
    assert transition.changed is False
    assert machine.current_state == ConversationState.IDLE


def test_invalid_transition_can_raise() -> None:
    machine = ConversationStateMachine(
        config=ConversationStateMachineConfig(
            reject_invalid_transitions=True
        )
    )

    with pytest.raises(ValueError):
        machine.transition(event(ConversationStateEventKind.RESPONSE_STARTED))


def test_state_machine_can_transition() -> None:
    machine = ConversationStateMachine()

    assert machine.can_transition(
        state=ConversationState.IDLE,
        event_kind=ConversationStateEventKind.START_LISTENING,
    )
    assert not machine.can_transition(
        state=ConversationState.IDLE,
        event_kind=ConversationStateEventKind.RESPONSE_STARTED,
    )


def test_state_machine_apply_turn_decision_finalize() -> None:
    detector = AdaptiveTurnDetector()
    machine = ConversationStateMachine()
    machine.transition(event(ConversationStateEventKind.START_LISTENING))

    decision = detector.evaluate(
        TurnDetectionInput(
            turn_id="turn-1",
            transcript="run tests",
            silence_ms=500,
        )
    )
    transition = machine.apply_turn_decision(decision)

    assert decision.decision == TurnDecisionKind.FINALIZE
    assert transition.next_state == ConversationState.THINKING


def test_state_machine_apply_turn_decision_maybe_complete() -> None:
    detector = AdaptiveTurnDetector()
    machine = ConversationStateMachine()
    machine.transition(event(ConversationStateEventKind.START_LISTENING))

    decision = detector.evaluate(
        TurnDetectionInput(
            turn_id="turn-1",
            transcript="Can you explain the memory gateway?",
            silence_ms=650,
        )
    )
    transition = machine.apply_turn_decision(decision)

    assert decision.decision == TurnDecisionKind.MAYBE_COMPLETE
    assert transition.next_state == ConversationState.USER_THINKING


def test_state_machine_apply_turn_decision_interrupt() -> None:
    detector = AdaptiveTurnDetector()
    machine = ConversationStateMachine(
        config=ConversationStateMachineConfig(
            initial_state=ConversationState.SPEAKING
        )
    )

    decision = detector.evaluate(
        TurnDetectionInput(
            turn_id="turn-1",
            transcript="stop",
        )
    )
    transition = machine.apply_turn_decision(decision)

    assert decision.decision == TurnDecisionKind.INTERRUPT
    assert transition.next_state == ConversationState.INTERRUPTED


def test_state_machine_snapshot_and_reset() -> None:
    machine = ConversationStateMachine()

    machine.transition(event(ConversationStateEventKind.START_LISTENING))
    snapshot = machine.snapshot()

    assert snapshot.transition_count == 1
    assert snapshot.applied_count == 1
    assert snapshot.current_state == ConversationState.LISTENING

    machine.reset()
    reset_snapshot = machine.snapshot()

    assert reset_snapshot.transition_count == 0
    assert reset_snapshot.current_state == ConversationState.IDLE


def test_conversation_state_enum_values_are_stable() -> None:
    assert ConversationState.IDLE.value == "idle"
    assert ConversationState.LISTENING.value == "listening"
    assert ConversationState.USER_THINKING.value == "user_thinking"
    assert ConversationState.THINKING.value == "thinking"
    assert ConversationState.SPEAKING.value == "speaking"
    assert ConversationState.INTERRUPTED.value == "interrupted"
    assert ConversationState.WAITING.value == "waiting"
    assert ConversationState.FOLLOW_UP.value == "follow_up"