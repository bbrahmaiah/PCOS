from __future__ import annotations

import pytest

from jarvis.presence.adapters import FakeAudioPlaybackAdapter
from jarvis.presence.models import PresenceMode, SpeechChunk, TurnPhase
from jarvis.presence.state import PresenceStateStore
from jarvis.presence.workers import AudioPlaybackWorker, InterruptionWorker
from jarvis.runtime.events import EventBus, RuntimeEvent
from jarvis.runtime.shared.enums import EventCategory, EventPriority, EventType


def make_user_started_event(
    *,
    segment_id: str = "segment-1",
    frame_id: str = "frame-1",
) -> RuntimeEvent:
    return RuntimeEvent(
        event_type=EventType.PRESENCE_USER_STARTED_SPEAKING,
        category=EventCategory.PRESENCE,
        source="test_vad_worker",
        payload={
            "segment_id": segment_id,
            "frame_id": frame_id,
        },
    )


def make_chunk(
    *,
    request_id: str = "speech-request-1",
) -> SpeechChunk:
    return SpeechChunk(
        request_id=request_id,
        audio_data=b"\x00\x01",
        final=True,
    )


def prepare_store_for_assistant_speaking(
    *,
    bus: EventBus,
    request_id: str = "speech-request-1",
) -> PresenceStateStore:
    store = PresenceStateStore(event_bus=bus)

    store.wake_detected(turn_id="turn-1")
    store.user_speech_started()
    store.user_speech_ended()
    store.transcript_ready()
    store.assistant_response_started(speech_request_id=request_id)

    return store


def test_interruption_worker_requires_positive_tick_interval() -> None:
    bus = EventBus(name="presence_test_bus")
    store = PresenceStateStore(event_bus=bus)

    with pytest.raises(ValueError):
        InterruptionWorker(
            event_bus=bus,
            presence_store=store,
            tick_interval_seconds=0,
        )


def test_interruption_worker_rejects_empty_name() -> None:
    bus = EventBus(name="presence_test_bus")
    store = PresenceStateStore(event_bus=bus)

    with pytest.raises(ValueError):
        InterruptionWorker(
            event_bus=bus,
            presence_store=store,
            name="   ",
        )


def test_interruption_worker_subscribes_on_start() -> None:
    bus = EventBus(name="presence_test_bus")
    store = PresenceStateStore(event_bus=bus)
    worker = InterruptionWorker(event_bus=bus, presence_store=store)

    worker.on_start()

    assert worker.interruption_snapshot().subscribed is True

    worker.on_stop()


def test_interruption_worker_ignores_non_user_started_event() -> None:
    bus = EventBus(name="presence_test_bus")
    store = PresenceStateStore(event_bus=bus)
    worker = InterruptionWorker(event_bus=bus, presence_store=store)

    event = RuntimeEvent(
        event_type=EventType.RUNTIME_TICK,
        category=EventCategory.RUNTIME,
        source="test",
    )

    worker.handle_user_started_speaking_event(event)

    snapshot = worker.interruption_snapshot()

    assert snapshot.ignored_events == 1
    assert snapshot.interruptions_requested == 0


def test_interruption_worker_does_nothing_when_assistant_not_speaking() -> None:
    bus = EventBus(name="presence_test_bus")
    store = PresenceStateStore(event_bus=bus)
    worker = InterruptionWorker(event_bus=bus, presence_store=store)

    result = worker.process_user_started_speaking(
        segment_id="segment-1",
        frame_id="frame-1",
    )

    snapshot = worker.interruption_snapshot()

    assert result is None
    assert snapshot.processed_user_speech_events == 1
    assert snapshot.non_interrupting_speech_events == 1
    assert snapshot.interruptions_requested == 0
    assert not any(
        event.event_type == EventType.INTERRUPT_REQUESTED
        for event in bus.history()
    )


def test_interruption_worker_requests_interrupt_when_assistant_speaking() -> None:
    bus = EventBus(name="presence_test_bus")
    store = prepare_store_for_assistant_speaking(bus=bus)
    worker = InterruptionWorker(event_bus=bus, presence_store=store)

    interruption_id = worker.process_user_started_speaking(
        segment_id="segment-1",
        frame_id="frame-1",
    )

    snapshot = worker.interruption_snapshot()
    state = store.current_state()
    interrupt_events = [
        event for event in bus.history()
        if event.event_type == EventType.INTERRUPT_REQUESTED
    ]

    assert interruption_id is not None
    assert snapshot.interruptions_requested == 1
    assert snapshot.last_interruption_id == interruption_id
    assert snapshot.last_request_id == "speech-request-1"
    assert state.assistant_speaking is False
    assert state.mode == PresenceMode.INTERRUPTED
    assert state.turn_phase == TurnPhase.INTERRUPTED
    assert len(interrupt_events) == 1
    assert interrupt_events[0].priority == EventPriority.CRITICAL
    assert interrupt_events[0].payload["request_id"] == "speech-request-1"
    assert interrupt_events[0].payload["reason"] == "user_barge_in"


def test_interruption_worker_can_process_event_payload() -> None:
    bus = EventBus(name="presence_test_bus")
    store = prepare_store_for_assistant_speaking(bus=bus)
    worker = InterruptionWorker(event_bus=bus, presence_store=store)

    worker.handle_user_started_speaking_event(
        make_user_started_event(segment_id="segment-9", frame_id="frame-9")
    )

    snapshot = worker.interruption_snapshot()

    assert snapshot.interruptions_requested == 1
    assert snapshot.last_segment_id == "segment-9"


def test_interruption_worker_subscribed_flow_with_playback_stop() -> None:
    bus = EventBus(name="presence_test_bus")
    store = prepare_store_for_assistant_speaking(bus=bus)
    playback_adapter = FakeAudioPlaybackAdapter()
    playback_worker = AudioPlaybackWorker(
        event_bus=bus,
        playback_adapter=playback_adapter,
        presence_store=store,
    )
    interruption_worker = InterruptionWorker(
        event_bus=bus,
        presence_store=store,
    )
    chunk = make_chunk()

    playback_worker.on_start()
    interruption_worker.on_start()

    playback_worker.process_chunk(chunk=chunk)

    assert playback_adapter.is_playing is True

    bus.publish_sync(make_user_started_event())

    playback_snapshot = playback_worker.playback_snapshot()
    interruption_snapshot = interruption_worker.interruption_snapshot()

    assert interruption_snapshot.interruptions_requested == 1
    assert playback_snapshot.playback_stopped == 1
    assert playback_adapter.is_playing is False
    assert any(
        event.event_type == EventType.AUDIO_PLAYBACK_STOPPED
        for event in bus.history()
    )
    assert any(
        event.event_type == EventType.ASSISTANT_SPEAKING_STOPPED
        for event in bus.history()
    )

    playback_worker.on_stop()
    interruption_worker.on_stop()


def test_interruption_event_does_not_stop_playback_when_idle() -> None:
    bus = EventBus(name="presence_test_bus")
    store = PresenceStateStore(event_bus=bus)
    playback_adapter = FakeAudioPlaybackAdapter()
    playback_worker = AudioPlaybackWorker(
        event_bus=bus,
        playback_adapter=playback_adapter,
        presence_store=store,
    )

    playback_worker.on_start()

    event = RuntimeEvent(
        event_type=EventType.INTERRUPT_REQUESTED,
        category=EventCategory.PRESENCE,
        source="test",
        payload={"request_id": "speech-request-1"},
    )

    playback_worker.handle_interrupt_requested_event(event)

    snapshot = playback_worker.playback_snapshot()

    assert snapshot.stop_requests == 1
    assert snapshot.playback_stopped == 0
    assert playback_adapter.is_playing is False

    playback_worker.on_stop()