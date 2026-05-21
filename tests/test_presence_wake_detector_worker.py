from __future__ import annotations

import pytest

from jarvis.presence.adapters import FakeWakeWordAdapter, make_fake_audio_frame
from jarvis.presence.models import AudioFrame
from jarvis.presence.state import PresenceStateStore
from jarvis.presence.workers import WakeDetectorWorker
from jarvis.runtime.events import EventBus, RuntimeEvent
from jarvis.runtime.shared.enums import EventCategory, EventPriority, EventType


class FailingWakeWordAdapter(FakeWakeWordAdapter):
    def detect(self, frame: AudioFrame):  # type: ignore[no-untyped-def]
        raise RuntimeError("wake adapter failed")


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


def test_wake_detector_requires_positive_tick_interval() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeWakeWordAdapter()

    with pytest.raises(ValueError):
        WakeDetectorWorker(
            event_bus=bus,
            wake_word_adapter=adapter,
            tick_interval_seconds=0,
        )


def test_wake_detector_subscribes_on_start() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeWakeWordAdapter()
    worker = WakeDetectorWorker(event_bus=bus, wake_word_adapter=adapter)

    worker.on_start()

    snapshot = worker.wake_snapshot()

    assert snapshot.subscribed is True

    worker.on_stop()

    assert adapter.reset_count == 1


def test_wake_detector_ignores_non_audio_event() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeWakeWordAdapter()
    worker = WakeDetectorWorker(event_bus=bus, wake_word_adapter=adapter)

    event = RuntimeEvent(
        event_type=EventType.RUNTIME_TICK,
        category=EventCategory.RUNTIME,
        source="test",
    )

    worker.handle_audio_frame_event(event)

    snapshot = worker.wake_snapshot()

    assert snapshot.ignored_events == 1
    assert snapshot.processed_frames == 0


def test_wake_detector_ignores_audio_event_without_frame() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeWakeWordAdapter()
    worker = WakeDetectorWorker(event_bus=bus, wake_word_adapter=adapter)

    event = RuntimeEvent(
        event_type=EventType.AUDIO_FRAME_CAPTURED,
        category=EventCategory.PRESENCE,
        source="test",
        payload={"frame_id": "missing-frame"},
    )

    worker.handle_audio_frame_event(event)

    snapshot = worker.wake_snapshot()

    assert snapshot.ignored_events == 1
    assert snapshot.processed_frames == 0


def test_wake_detector_records_missed_detection() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeWakeWordAdapter(default_detect=False)
    worker = WakeDetectorWorker(event_bus=bus, wake_word_adapter=adapter)
    frame = make_fake_audio_frame()

    detection = worker.process_frame(frame)

    snapshot = worker.wake_snapshot()

    assert detection is None
    assert bus.history() == ()
    assert snapshot.processed_frames == 1
    assert snapshot.missed_detections == 1
    assert snapshot.wake_detections == 0


def test_wake_detector_publishes_wake_event() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeWakeWordAdapter(default_detect=True)
    worker = WakeDetectorWorker(event_bus=bus, wake_word_adapter=adapter)
    frame = make_fake_audio_frame(frame_index=3)

    detection = worker.process_frame(frame)

    history = bus.history()
    snapshot = worker.wake_snapshot()

    assert detection is not None
    assert len(history) == 1
    assert history[0].event_type == EventType.PRESENCE_WAKE_DETECTED
    assert history[0].priority == EventPriority.HIGH
    assert history[0].source == "wake_detector_worker"
    assert history[0].payload["frame_id"] == frame.frame_id
    assert history[0].payload["wake_word"] == "jarvis"
    assert snapshot.processed_frames == 1
    assert snapshot.wake_detections == 1
    assert snapshot.last_wake_word == "jarvis"


def test_wake_detector_updates_presence_store() -> None:
    bus = EventBus(name="presence_test_bus")
    store = PresenceStateStore(event_bus=bus)
    adapter = FakeWakeWordAdapter(default_detect=True)
    worker = WakeDetectorWorker(
        event_bus=bus,
        wake_word_adapter=adapter,
        presence_store=store,
    )

    frame = make_fake_audio_frame()

    worker.process_frame(frame)

    state = store.current_state()
    history = bus.history()

    assert state.awake is True
    assert state.listening is True
    assert state.current_turn_id is not None
    assert any(
        event.event_type == EventType.PRESENCE_STATE_CHANGED
        for event in history
    )
    assert any(
        event.event_type == EventType.PRESENCE_WAKE_DETECTED
        for event in history
    )


def test_wake_detector_can_process_event_payload_frame() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeWakeWordAdapter(default_detect=True)
    worker = WakeDetectorWorker(event_bus=bus, wake_word_adapter=adapter)
    frame = make_fake_audio_frame()

    worker.handle_audio_frame_event(make_audio_event(frame))

    assert worker.wake_snapshot().wake_detections == 1
    assert len(bus.history()) == 1


def test_wake_detector_subscribed_event_flow_with_publish_sync() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeWakeWordAdapter(default_detect=True)
    worker = WakeDetectorWorker(event_bus=bus, wake_word_adapter=adapter)
    frame = make_fake_audio_frame()

    worker.on_start()
    bus.publish_sync(make_audio_event(frame))

    assert worker.wake_snapshot().wake_detections == 1
    assert len(bus.history()) == 2

    worker.on_stop()


def test_wake_detector_records_adapter_failure() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FailingWakeWordAdapter()
    worker = WakeDetectorWorker(event_bus=bus, wake_word_adapter=adapter)
    frame = make_fake_audio_frame()

    with pytest.raises(RuntimeError):
        worker.process_frame(frame)

    snapshot = worker.wake_snapshot()

    assert snapshot.processed_frames == 1
    assert snapshot.detection_failures == 1
    assert snapshot.last_error is not None