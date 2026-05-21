from __future__ import annotations

import pytest

from jarvis.presence.models import PresenceMode, PresenceState, TurnPhase
from jarvis.presence.state import PresenceStateStore, TurnTrigger
from jarvis.runtime.events import EventBus
from jarvis.runtime.shared.enums import EventType


def assert_default_presence_state(state: PresenceState) -> None:
    assert state.mode == PresenceMode.IDLE
    assert state.turn_phase == TurnPhase.NONE
    assert state.awake is False
    assert state.listening is False
    assert state.user_speaking is False
    assert state.assistant_speaking is False
    assert state.current_turn_id is None
    assert state.active_speech_request_id is None
    assert state.active_cancellation_token_id is None
    assert state.last_error is None
    assert state.active is False
    assert state.interruptible is False


def test_store_starts_with_default_state() -> None:
    store = PresenceStateStore()

    snapshot = store.snapshot()

    assert_default_presence_state(snapshot.state)
    assert snapshot.transition_count == 0
    assert snapshot.last_transition is None


def test_store_rejects_invalid_history_limit() -> None:
    with pytest.raises(ValueError):
        PresenceStateStore(history_limit=0)


def test_store_rejects_empty_event_source() -> None:
    with pytest.raises(ValueError):
        PresenceStateStore(event_source="   ")


def test_store_applies_valid_transition() -> None:
    store = PresenceStateStore()

    result = store.start_listening()
    snapshot = store.snapshot()
    state = store.current_state()

    assert result.changed is True
    assert state.mode == PresenceMode.LISTENING
    assert state.turn_phase == TurnPhase.LISTENING_FOR_USER
    assert state.awake is True
    assert state.listening is True
    assert snapshot.transition_count == 1


def test_store_records_rejected_transition_without_mutating_state() -> None:
    store = PresenceStateStore()

    result = store.user_speech_ended()
    snapshot = store.snapshot()
    last_transition = snapshot.last_transition
    state = store.current_state()

    assert result.changed is False
    assert_default_presence_state(state)
    assert snapshot.transition_count == 1
    assert last_transition is not None
    assert last_transition.changed is False
    assert last_transition.trigger == TurnTrigger.USER_SPEECH_ENDED


def test_store_emits_state_changed_event() -> None:
    bus = EventBus(name="presence_test_bus")
    store = PresenceStateStore(event_bus=bus)

    store.wake_detected(turn_id="turn-1")

    history = bus.history()

    assert len(history) == 1
    assert history[0].event_type == EventType.PRESENCE_STATE_CHANGED
    assert history[0].payload["trigger"] == TurnTrigger.WAKE_DETECTED.value
    assert history[0].payload["next_mode"] == PresenceMode.LISTENING.value
    assert history[0].payload["current_turn_id"] == "turn-1"


def test_store_does_not_emit_event_for_rejected_transition() -> None:
    bus = EventBus(name="presence_test_bus")
    store = PresenceStateStore(event_bus=bus)

    store.user_speech_ended()

    assert bus.history() == ()


def test_store_full_voice_turn_flow() -> None:
    store = PresenceStateStore()

    store.wake_detected(turn_id="turn-1")
    store.user_speech_started()
    store.user_speech_ended()
    store.transcript_ready()
    store.assistant_response_started(
        speech_request_id="speech-1",
        cancellation_token_id="cancel-1",
    )

    state = store.current_state()

    assert state.mode == PresenceMode.ASSISTANT_SPEAKING
    assert state.turn_phase == TurnPhase.SPEAKING_RESPONSE
    assert state.current_turn_id == "turn-1"
    assert state.active_speech_request_id == "speech-1"
    assert state.active_cancellation_token_id == "cancel-1"

    store.assistant_response_finished()

    state = store.current_state()

    assert state.mode == PresenceMode.LISTENING
    assert state.active_speech_request_id is None
    assert state.active_cancellation_token_id is None


def test_store_interruption_flow() -> None:
    store = PresenceStateStore()

    store.wake_detected(turn_id="turn-1")
    store.user_speech_started()
    store.user_speech_ended()
    store.transcript_ready()
    store.assistant_response_started(
        speech_request_id="speech-1",
        cancellation_token_id="cancel-1",
    )

    result = store.interruption_detected(cancellation_token_id="cancel-1")
    state = store.current_state()

    assert result.changed is True
    assert state.mode == PresenceMode.INTERRUPTED
    assert state.turn_phase == TurnPhase.INTERRUPTED
    assert state.active_cancellation_token_id == "cancel-1"


def test_store_sleep_reset_error_recover_flow() -> None:
    store = PresenceStateStore()

    store.wake_detected(turn_id="turn-1")
    store.sleep_requested()

    assert store.current_state().mode == PresenceMode.SLEEPING

    store.wake_detected(turn_id="turn-2")
    store.error_occurred(error="vad failed")

    assert store.current_state().mode == PresenceMode.ERROR
    assert store.current_state().last_error == "vad failed"

    store.recover()

    state = store.current_state()

    assert state.mode == PresenceMode.IDLE
    assert state.turn_phase == TurnPhase.NONE
    assert state.last_error is None

    store.wake_detected(turn_id="turn-3")
    store.reset()

    assert_default_presence_state(store.current_state())


def test_store_history_is_bounded() -> None:
    store = PresenceStateStore(history_limit=3)

    store.start_listening()
    store.sleep_requested()
    store.wake_detected(turn_id="turn-1")
    store.user_speech_started()

    history = store.history()

    assert len(history) == 3
    assert history[-1].trigger == TurnTrigger.USER_SPEECH_STARTED


def test_store_metadata_is_in_event_payload() -> None:
    bus = EventBus(name="presence_test_bus")
    store = PresenceStateStore(event_bus=bus)

    store.wake_detected(
        turn_id="turn-1",
        metadata={"wake_word": "jarvis", "confidence": 0.98},
    )

    event = bus.history()[0]

    assert event.payload["metadata"] == {
        "wake_word": "jarvis",
        "confidence": 0.98,
    }


def test_store_initial_state_can_be_injected() -> None:
    initial_state = PresenceState(
        mode=PresenceMode.SLEEPING,
        turn_phase=TurnPhase.WAITING_FOR_WAKE,
    )
    store = PresenceStateStore(initial_state=initial_state)

    state = store.current_state()

    assert state.mode == PresenceMode.SLEEPING
    assert state.turn_phase == TurnPhase.WAITING_FOR_WAKE