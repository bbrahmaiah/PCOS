from __future__ import annotations

import pytest

from jarvis.presence.models import PresenceMode, PresenceState, TurnPhase
from jarvis.presence.state import TurnStateMachine, TurnTrigger


def test_state_machine_starts_listening_from_idle() -> None:
    machine = TurnStateMachine()
    state = PresenceState()

    result = machine.start_listening(state)

    assert result.changed is True
    assert result.allowed is True
    assert result.trigger == TurnTrigger.START_LISTENING
    assert result.next_state.mode == PresenceMode.LISTENING
    assert result.next_state.turn_phase == TurnPhase.LISTENING_FOR_USER
    assert result.next_state.awake is True
    assert result.next_state.listening is True


def test_wake_detected_creates_turn_id() -> None:
    machine = TurnStateMachine()
    state = PresenceState()

    result = machine.wake_detected(state)

    assert result.changed is True
    assert result.next_state.current_turn_id is not None
    assert result.next_state.last_wake_at is not None
    assert result.next_state.mode == PresenceMode.LISTENING


def test_user_speech_flow_to_waiting_for_response() -> None:
    machine = TurnStateMachine()

    state = machine.wake_detected(PresenceState(), turn_id="turn-1").next_state
    state = machine.user_speech_started(state).next_state

    assert state.mode == PresenceMode.USER_SPEAKING
    assert state.turn_phase == TurnPhase.CAPTURING_USER_SPEECH
    assert state.user_speaking is True
    assert state.last_user_speech_at is not None

    state = machine.user_speech_ended(state).next_state

    assert state.mode == PresenceMode.PROCESSING_SPEECH
    assert state.turn_phase == TurnPhase.TRANSCRIBING
    assert state.user_speaking is False

    state = machine.transcript_ready(state).next_state

    assert state.mode == PresenceMode.PROCESSING_SPEECH
    assert state.turn_phase == TurnPhase.WAITING_FOR_RESPONSE


def test_assistant_speaking_flow_returns_to_listening() -> None:
    machine = TurnStateMachine()

    state = machine.wake_detected(PresenceState(), turn_id="turn-1").next_state
    state = machine.user_speech_started(state).next_state
    state = machine.user_speech_ended(state).next_state
    state = machine.transcript_ready(state).next_state

    state = machine.assistant_response_started(
        state,
        speech_request_id="speech-1",
        cancellation_token_id="cancel-1",
    ).next_state

    assert state.mode == PresenceMode.ASSISTANT_SPEAKING
    assert state.turn_phase == TurnPhase.SPEAKING_RESPONSE
    assert state.assistant_speaking is True
    assert state.interruptible is True
    assert state.active_speech_request_id == "speech-1"
    assert state.active_cancellation_token_id == "cancel-1"
    assert state.last_assistant_speech_at is not None

    state = machine.assistant_response_finished(state).next_state

    assert state.mode == PresenceMode.LISTENING
    assert state.turn_phase == TurnPhase.LISTENING_FOR_USER
    assert state.assistant_speaking is False
    assert state.active_speech_request_id is None
    assert state.active_cancellation_token_id is None


def test_interruption_flow() -> None:
    machine = TurnStateMachine()

    state = machine.wake_detected(PresenceState(), turn_id="turn-1").next_state
    state = machine.user_speech_started(state).next_state
    state = machine.user_speech_ended(state).next_state
    state = machine.transcript_ready(state).next_state
    state = machine.assistant_response_started(
        state,
        speech_request_id="speech-1",
        cancellation_token_id="cancel-1",
    ).next_state

    result = machine.interruption_detected(
        state,
        cancellation_token_id="cancel-1",
    )

    assert result.changed is True
    assert result.next_state.mode == PresenceMode.INTERRUPTED
    assert result.next_state.turn_phase == TurnPhase.INTERRUPTED
    assert result.next_state.listening is True
    assert result.next_state.assistant_speaking is False
    assert result.next_state.active_cancellation_token_id == "cancel-1"

    resumed = machine.user_speech_started(result.next_state).next_state

    assert resumed.mode == PresenceMode.USER_SPEAKING
    assert resumed.turn_phase == TurnPhase.CAPTURING_USER_SPEECH


def test_sleep_requested_from_any_state() -> None:
    machine = TurnStateMachine()
    state = machine.wake_detected(PresenceState()).next_state

    result = machine.sleep_requested(state)

    assert result.changed is True
    assert result.next_state.mode == PresenceMode.SLEEPING
    assert result.next_state.turn_phase == TurnPhase.WAITING_FOR_WAKE
    assert result.next_state.awake is False
    assert result.next_state.active is False


def test_reset_returns_to_default_state() -> None:
    machine = TurnStateMachine()
    state = machine.wake_detected(PresenceState()).next_state

    result = machine.reset(state)
    default_state = PresenceState()

    assert result.changed is True
    assert result.next_state.model_dump(exclude={"updated_at"}) == (
        default_state.model_dump(exclude={"updated_at"})
    )


def test_error_and_recover_flow() -> None:
    machine = TurnStateMachine()
    state = machine.wake_detected(PresenceState(), turn_id="turn-1").next_state

    errored = machine.error_occurred(state, error="stt failed").next_state

    assert errored.mode == PresenceMode.ERROR
    assert errored.turn_phase == TurnPhase.FAILED
    assert errored.last_error == "stt failed"

    recovered = machine.recover(errored).next_state

    assert recovered.mode == PresenceMode.IDLE
    assert recovered.turn_phase == TurnPhase.NONE
    assert recovered.last_error is None


def test_invalid_transition_is_rejected_without_mutation() -> None:
    machine = TurnStateMachine()
    state = PresenceState()

    result = machine.user_speech_ended(state)

    assert result.changed is False
    assert result.allowed is False
    assert result.next_state == state
    assert result.reason is not None


def test_assistant_response_requires_speech_request_id() -> None:
    machine = TurnStateMachine()

    state = machine.wake_detected(PresenceState(), turn_id="turn-1").next_state
    state = machine.user_speech_started(state).next_state
    state = machine.user_speech_ended(state).next_state
    state = machine.transcript_ready(state).next_state

    with pytest.raises(ValueError):
        machine.assistant_response_started(
            state,
            speech_request_id="   ",
        )


def test_sleeping_can_wake_again() -> None:
    machine = TurnStateMachine()

    state = machine.sleep_requested(PresenceState()).next_state
    state = machine.wake_detected(state, turn_id="turn-2").next_state

    assert state.mode == PresenceMode.LISTENING
    assert state.turn_phase == TurnPhase.LISTENING_FOR_USER
    assert state.current_turn_id == "turn-2"