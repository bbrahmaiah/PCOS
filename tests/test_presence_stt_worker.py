from __future__ import annotations

import pytest

from jarvis.presence.adapters import FakeSpeechToTextAdapter, make_fake_audio_frame
from jarvis.presence.models import AudioFrame, Transcript, TranscriptKind
from jarvis.presence.state import PresenceStateStore
from jarvis.presence.workers import STTWorker
from jarvis.runtime.events import EventBus, RuntimeEvent
from jarvis.runtime.shared.enums import EventCategory, EventPriority, EventType


class FailingSpeechToTextAdapter(FakeSpeechToTextAdapter):
    def transcribe(self, frames: tuple[AudioFrame, ...]) -> Transcript:
        raise RuntimeError("stt adapter failed")


def make_segment_event(
    *,
    segment_id: str = "segment-1",
    frames: tuple[AudioFrame, ...] | None = None,
) -> RuntimeEvent:
    segment_frames = frames or (
        make_fake_audio_frame(frame_index=0),
        make_fake_audio_frame(frame_index=1),
    )

    return RuntimeEvent(
        event_type=EventType.AUDIO_SPEECH_SEGMENT_COMPLETED,
        category=EventCategory.PRESENCE,
        source="test_vad_worker",
        payload={
            "segment_id": segment_id,
            "frames": segment_frames,
            "frame_count": len(segment_frames),
        },
    )


def test_stt_worker_requires_positive_tick_interval() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeSpeechToTextAdapter()

    with pytest.raises(ValueError):
        STTWorker(
            event_bus=bus,
            stt_adapter=adapter,
            tick_interval_seconds=0,
        )


def test_stt_worker_subscribes_on_start() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeSpeechToTextAdapter()
    worker = STTWorker(event_bus=bus, stt_adapter=adapter)

    worker.on_start()

    assert worker.stt_snapshot().subscribed is True

    worker.on_stop()

    assert adapter.reset_count == 1


def test_stt_worker_ignores_non_segment_event() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeSpeechToTextAdapter()
    worker = STTWorker(event_bus=bus, stt_adapter=adapter)

    event = RuntimeEvent(
        event_type=EventType.RUNTIME_TICK,
        category=EventCategory.RUNTIME,
        source="test",
    )

    worker.handle_speech_segment_event(event)

    snapshot = worker.stt_snapshot()

    assert snapshot.ignored_events == 1
    assert snapshot.processed_segments == 0


def test_stt_worker_ignores_segment_event_without_frames() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeSpeechToTextAdapter()
    worker = STTWorker(event_bus=bus, stt_adapter=adapter)

    event = RuntimeEvent(
        event_type=EventType.AUDIO_SPEECH_SEGMENT_COMPLETED,
        category=EventCategory.PRESENCE,
        source="test",
        payload={"segment_id": "segment-1"},
    )

    worker.handle_speech_segment_event(event)

    snapshot = worker.stt_snapshot()

    assert snapshot.ignored_events == 1
    assert snapshot.processed_segments == 0


def test_stt_worker_publishes_final_transcript() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeSpeechToTextAdapter(
        text="open the browser",
        kind=TranscriptKind.FINAL,
        confidence=0.97,
    )
    worker = STTWorker(event_bus=bus, stt_adapter=adapter)

    event = make_segment_event(segment_id="segment-1")
    worker.handle_speech_segment_event(event)

    history = bus.history()
    snapshot = worker.stt_snapshot()

    assert len(history) == 1
    assert history[0].event_type == EventType.PRESENCE_TRANSCRIPT_FINAL
    assert history[0].priority == EventPriority.HIGH
    assert history[0].source == "stt_worker"
    assert history[0].payload["segment_id"] == "segment-1"
    assert history[0].payload["text"] == "open the browser"
    assert history[0].payload["confidence"] == 0.97
    assert history[0].payload["frame_count"] == 2
    assert snapshot.processed_segments == 1
    assert snapshot.final_transcripts == 1
    assert snapshot.last_transcript_kind == TranscriptKind.FINAL.value


def test_stt_worker_publishes_partial_transcript() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeSpeechToTextAdapter(
        text="open",
        kind=TranscriptKind.PARTIAL,
    )
    worker = STTWorker(event_bus=bus, stt_adapter=adapter)

    worker.handle_speech_segment_event(make_segment_event())

    event = bus.history()[0]
    snapshot = worker.stt_snapshot()

    assert event.event_type == EventType.PRESENCE_TRANSCRIPT_PARTIAL
    assert event.priority == EventPriority.NORMAL
    assert event.payload["kind"] == TranscriptKind.PARTIAL.value
    assert snapshot.partial_transcripts == 1


def test_stt_worker_updates_presence_store_on_final_transcript() -> None:
    bus = EventBus(name="presence_test_bus")
    store = PresenceStateStore(event_bus=bus)
    store.wake_detected(turn_id="turn-1")
    store.user_speech_started()
    store.user_speech_ended()

    adapter = FakeSpeechToTextAdapter(
        text="hello jarvis",
        kind=TranscriptKind.FINAL,
    )
    worker = STTWorker(
        event_bus=bus,
        stt_adapter=adapter,
        presence_store=store,
    )

    worker.handle_speech_segment_event(make_segment_event())

    state = store.current_state()

    assert state.turn_phase.value == "waiting_for_response"
    assert any(
        event.event_type == EventType.PRESENCE_TRANSCRIPT_FINAL
        for event in bus.history()
    )
    assert any(
        event.event_type == EventType.PRESENCE_STATE_CHANGED
        for event in bus.history()
    )


def test_stt_worker_does_not_update_store_on_partial_transcript() -> None:
    bus = EventBus(name="presence_test_bus")
    store = PresenceStateStore(event_bus=bus)
    store.wake_detected(turn_id="turn-1")
    store.user_speech_started()
    store.user_speech_ended()

    adapter = FakeSpeechToTextAdapter(
        text="hello",
        kind=TranscriptKind.PARTIAL,
    )
    worker = STTWorker(
        event_bus=bus,
        stt_adapter=adapter,
        presence_store=store,
    )

    worker.handle_speech_segment_event(make_segment_event())

    assert store.current_state().turn_phase.value == "transcribing"


def test_stt_worker_rejects_failed_transcription() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FailingSpeechToTextAdapter()
    worker = STTWorker(event_bus=bus, stt_adapter=adapter)

    result = worker.process_segment(
        segment_id="segment-1",
        frames=(make_fake_audio_frame(),),
    )

    history = bus.history()
    snapshot = worker.stt_snapshot()

    assert result is None
    assert len(history) == 1
    assert history[0].event_type == EventType.PRESENCE_TRANSCRIPT_REJECTED
    assert history[0].payload["kind"] == TranscriptKind.REJECTED.value
    assert history[0].payload["error"] == "RuntimeError: stt adapter failed"
    assert snapshot.transcription_failures == 1
    assert snapshot.rejected_transcripts == 1
    assert snapshot.last_error is not None


def test_stt_worker_rejects_empty_segment_id() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeSpeechToTextAdapter()
    worker = STTWorker(event_bus=bus, stt_adapter=adapter)

    with pytest.raises(ValueError):
        worker.process_segment(
            segment_id="   ",
            frames=(make_fake_audio_frame(),),
        )


def test_stt_worker_rejects_empty_frames() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeSpeechToTextAdapter()
    worker = STTWorker(event_bus=bus, stt_adapter=adapter)

    with pytest.raises(ValueError):
        worker.process_segment(
            segment_id="segment-1",
            frames=(),
        )


def test_stt_worker_subscribed_event_flow_with_publish_sync() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeSpeechToTextAdapter()
    worker = STTWorker(event_bus=bus, stt_adapter=adapter)

    worker.on_start()
    bus.publish_sync(make_segment_event())

    assert worker.stt_snapshot().final_transcripts == 1
    assert len(bus.history()) == 2

    worker.on_stop()