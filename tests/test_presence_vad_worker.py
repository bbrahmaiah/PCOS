from __future__ import annotations

import pytest

from jarvis.presence.adapters import FakeVoiceActivityAdapter, make_fake_audio_frame
from jarvis.presence.models import AudioFrame, VoiceActivity, VoiceActivityState
from jarvis.presence.state import PresenceStateStore
from jarvis.presence.workers import VADWorker
from jarvis.runtime.events import EventBus, RuntimeEvent
from jarvis.runtime.shared.enums import EventCategory, EventPriority, EventType


class FailingVoiceActivityAdapter(FakeVoiceActivityAdapter):
    def analyze(self, frame: AudioFrame) -> VoiceActivity:
        raise RuntimeError("vad adapter failed")


def make_audio_event(frame: AudioFrame) -> RuntimeEvent:
    return RuntimeEvent(
        event_type=EventType.AUDIO_FRAME_CAPTURED,
        category=EventCategory.PRESENCE,
        source="test_voice_input",
        payload={
            "frame": frame,
            "frame_id": frame.frame_id,
        },
    )


def test_vad_worker_requires_positive_tick_interval() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeVoiceActivityAdapter()

    with pytest.raises(ValueError):
        VADWorker(
            event_bus=bus,
            vad_adapter=adapter,
            tick_interval_seconds=0,
        )


def test_vad_worker_subscribes_on_start() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeVoiceActivityAdapter()
    worker = VADWorker(event_bus=bus, vad_adapter=adapter)

    worker.on_start()

    assert worker.vad_snapshot().subscribed is True

    worker.on_stop()

    assert adapter.reset_count == 1


def test_vad_worker_ignores_non_audio_event() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeVoiceActivityAdapter()
    worker = VADWorker(event_bus=bus, vad_adapter=adapter)

    event = RuntimeEvent(
        event_type=EventType.RUNTIME_TICK,
        category=EventCategory.RUNTIME,
        source="test",
    )

    worker.handle_audio_frame_event(event)

    snapshot = worker.vad_snapshot()

    assert snapshot.ignored_events == 1
    assert snapshot.processed_frames == 0


def test_vad_worker_ignores_audio_event_without_frame() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeVoiceActivityAdapter()
    worker = VADWorker(event_bus=bus, vad_adapter=adapter)

    event = RuntimeEvent(
        event_type=EventType.AUDIO_FRAME_CAPTURED,
        category=EventCategory.PRESENCE,
        source="test",
        payload={"frame_id": "missing-frame"},
    )

    worker.handle_audio_frame_event(event)

    snapshot = worker.vad_snapshot()

    assert snapshot.ignored_events == 1
    assert snapshot.processed_frames == 0


def test_vad_worker_records_silence() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeVoiceActivityAdapter(
        states=(VoiceActivityState.SILENCE,),
    )
    worker = VADWorker(event_bus=bus, vad_adapter=adapter)

    frame = make_fake_audio_frame()
    activity = worker.process_frame(frame)

    snapshot = worker.vad_snapshot()

    assert activity.state == VoiceActivityState.SILENCE
    assert bus.history() == ()
    assert snapshot.processed_frames == 1
    assert snapshot.silence_frames == 1
    assert snapshot.active_segment is False


def test_vad_worker_emits_speech_started_events() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeVoiceActivityAdapter(
        states=(VoiceActivityState.SPEECH_STARTED,),
    )
    worker = VADWorker(event_bus=bus, vad_adapter=adapter)

    frame = make_fake_audio_frame(frame_index=1)
    worker.process_frame(frame)

    history = bus.history()
    snapshot = worker.vad_snapshot()

    assert len(history) == 2
    assert history[0].event_type == EventType.PRESENCE_USER_STARTED_SPEAKING
    assert history[0].priority == EventPriority.HIGH
    assert history[1].event_type == EventType.AUDIO_SPEECH_SEGMENT_STARTED
    assert history[0].payload["frame_id"] == frame.frame_id
    assert history[0].payload["segment_id"] == history[1].payload["segment_id"]
    assert snapshot.speech_started_count == 1
    assert snapshot.active_segment is True
    assert snapshot.active_segment_frame_count == 1


def test_vad_worker_builds_completed_speech_segment() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeVoiceActivityAdapter(
        states=(
            VoiceActivityState.SPEECH_STARTED,
            VoiceActivityState.SPEECH_CONTINUING,
            VoiceActivityState.SPEECH_ENDED,
        ),
    )
    worker = VADWorker(event_bus=bus, vad_adapter=adapter)

    frames = (
        make_fake_audio_frame(frame_index=0),
        make_fake_audio_frame(frame_index=1),
        make_fake_audio_frame(frame_index=2),
    )

    for frame in frames:
        worker.process_frame(frame)

    history = bus.history()
    completed = history[-1]
    snapshot = worker.vad_snapshot()

    assert len(history) == 4
    assert history[0].event_type == EventType.PRESENCE_USER_STARTED_SPEAKING
    assert history[1].event_type == EventType.AUDIO_SPEECH_SEGMENT_STARTED
    assert history[2].event_type == EventType.PRESENCE_USER_STOPPED_SPEAKING
    assert history[3].event_type == EventType.AUDIO_SPEECH_SEGMENT_COMPLETED
    assert completed.payload["frames"] == frames
    assert completed.payload["frame_ids"] == tuple(frame.frame_id for frame in frames)
    assert completed.payload["frame_count"] == 3
    assert snapshot.active_segment is False
    assert snapshot.speech_segments_completed == 1


def test_vad_worker_updates_presence_store() -> None:
    bus = EventBus(name="presence_test_bus")
    store = PresenceStateStore(event_bus=bus)
    store.wake_detected(turn_id="turn-1")

    adapter = FakeVoiceActivityAdapter(
        states=(
            VoiceActivityState.SPEECH_STARTED,
            VoiceActivityState.SPEECH_ENDED,
        ),
    )
    worker = VADWorker(
        event_bus=bus,
        vad_adapter=adapter,
        presence_store=store,
    )

    worker.process_frame(make_fake_audio_frame(frame_index=0))

    assert store.current_state().user_speaking is True

    worker.process_frame(make_fake_audio_frame(frame_index=1))

    assert store.current_state().user_speaking is False
    assert store.current_state().turn_phase.value == "transcribing"
    assert any(
        event.event_type == EventType.PRESENCE_STATE_CHANGED
        for event in bus.history()
    )


def test_vad_worker_can_process_event_payload_frame() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeVoiceActivityAdapter(
        states=(VoiceActivityState.SPEECH_STARTED,),
    )
    worker = VADWorker(event_bus=bus, vad_adapter=adapter)
    frame = make_fake_audio_frame()

    worker.handle_audio_frame_event(make_audio_event(frame))

    assert worker.vad_snapshot().speech_started_count == 1
    assert len(bus.history()) == 2


def test_vad_worker_subscribed_event_flow_with_publish_sync() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeVoiceActivityAdapter(
        states=(VoiceActivityState.SPEECH_STARTED,),
    )
    worker = VADWorker(event_bus=bus, vad_adapter=adapter)
    frame = make_fake_audio_frame()

    worker.on_start()
    bus.publish_sync(make_audio_event(frame))

    assert worker.vad_snapshot().speech_started_count == 1
    assert len(bus.history()) == 3

    worker.on_stop()


def test_vad_worker_records_adapter_failure() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FailingVoiceActivityAdapter()
    worker = VADWorker(event_bus=bus, vad_adapter=adapter)
    frame = make_fake_audio_frame()

    with pytest.raises(RuntimeError):
        worker.process_frame(frame)

    snapshot = worker.vad_snapshot()

    assert snapshot.processed_frames == 1
    assert snapshot.vad_failures == 1
    assert snapshot.last_error is not None