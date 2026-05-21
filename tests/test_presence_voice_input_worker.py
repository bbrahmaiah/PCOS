from __future__ import annotations

import pytest

from jarvis.presence.adapters import FakeMicrophoneAdapter, make_fake_audio_frame
from jarvis.presence.workers import VoiceInputWorker
from jarvis.runtime.events import EventBus
from jarvis.runtime.shared.enums import EventPriority, EventType


def test_voice_input_worker_requires_positive_poll_interval() -> None:
    bus = EventBus(name="presence_test_bus")
    microphone = FakeMicrophoneAdapter()

    with pytest.raises(ValueError):
        VoiceInputWorker(
            event_bus=bus,
            microphone=microphone,
            poll_interval_seconds=0,
        )


def test_voice_input_worker_auto_starts_microphone_on_start() -> None:
    bus = EventBus(name="presence_test_bus")
    microphone = FakeMicrophoneAdapter()
    worker = VoiceInputWorker(event_bus=bus, microphone=microphone)

    assert microphone.is_running is False

    worker.on_start()

    assert microphone.is_running is True

    worker.on_stop()

    assert microphone.is_running is False


def test_voice_input_worker_can_disable_auto_start() -> None:
    bus = EventBus(name="presence_test_bus")
    microphone = FakeMicrophoneAdapter()
    worker = VoiceInputWorker(
        event_bus=bus,
        microphone=microphone,
        auto_start_microphone=False,
    )

    worker.on_start()

    assert microphone.is_running is False


def test_voice_input_worker_publishes_audio_frame_event() -> None:
    bus = EventBus(name="presence_test_bus")
    frame = make_fake_audio_frame(frame_index=7)
    microphone = FakeMicrophoneAdapter(frames=(frame,))
    worker = VoiceInputWorker(event_bus=bus, microphone=microphone)

    worker.on_start()
    worker.run_once()

    history = bus.history()
    snapshot = worker.voice_snapshot()

    assert len(history) == 1
    assert history[0].event_type == EventType.AUDIO_FRAME_CAPTURED
    assert history[0].priority == EventPriority.NORMAL
    assert history[0].source == "voice_input_worker"
    assert history[0].payload["frame_id"] == frame.frame_id
    assert history[0].payload["frame_index"] == 7
    assert history[0].payload["sample_rate"] == 16_000
    assert history[0].payload["channels"] == 1
    assert snapshot.captured_frames == 1
    assert snapshot.empty_reads == 0
    assert snapshot.last_frame_id == frame.frame_id

    worker.on_stop()


def test_voice_input_worker_records_empty_reads() -> None:
    bus = EventBus(name="presence_test_bus")
    microphone = FakeMicrophoneAdapter()
    worker = VoiceInputWorker(event_bus=bus, microphone=microphone)

    worker.on_start()
    worker.run_once()

    snapshot = worker.voice_snapshot()

    assert bus.history() == ()
    assert snapshot.captured_frames == 0
    assert snapshot.empty_reads == 1

    worker.on_stop()


def test_voice_input_worker_can_publish_multiple_frames() -> None:
    bus = EventBus(name="presence_test_bus")
    frames = (
        make_fake_audio_frame(frame_index=0),
        make_fake_audio_frame(frame_index=1),
        make_fake_audio_frame(frame_index=2),
    )
    microphone = FakeMicrophoneAdapter(frames=frames)
    worker = VoiceInputWorker(event_bus=bus, microphone=microphone)

    worker.on_start()

    worker.run_once()
    worker.run_once()
    worker.run_once()
    worker.run_once()

    history = bus.history()
    snapshot = worker.voice_snapshot()

    assert len(history) == 3
    assert [event.payload["frame_index"] for event in history] == [0, 1, 2]
    assert snapshot.captured_frames == 3
    assert snapshot.empty_reads == 1
    assert snapshot.last_frame_id == frames[-1].frame_id

    worker.on_stop()


def test_voice_input_worker_run_once_autostarts_microphone() -> None:
    bus = EventBus(name="presence_test_bus")
    frame = make_fake_audio_frame()
    microphone = FakeMicrophoneAdapter(frames=(frame,))
    worker = VoiceInputWorker(event_bus=bus, microphone=microphone)

    assert microphone.is_running is False

    worker.run_once()

    assert microphone.is_running is True
    assert len(bus.history()) == 1


def test_voice_input_worker_does_not_autostart_when_disabled() -> None:
    bus = EventBus(name="presence_test_bus")
    frame = make_fake_audio_frame()
    microphone = FakeMicrophoneAdapter(frames=(frame,))
    worker = VoiceInputWorker(
        event_bus=bus,
        microphone=microphone,
        auto_start_microphone=False,
    )

    worker.run_once()

    snapshot = worker.voice_snapshot()

    assert microphone.is_running is False
    assert bus.history() == ()
    assert snapshot.empty_reads == 1