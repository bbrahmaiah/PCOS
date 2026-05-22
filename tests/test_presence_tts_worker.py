from __future__ import annotations

import pytest

from jarvis.presence.adapters import FakeTextToSpeechAdapter
from jarvis.presence.models import SpeechChunk, SpeechRequest
from jarvis.presence.workers import TTSWorker
from jarvis.runtime.events import EventBus, RuntimeEvent
from jarvis.runtime.shared.enums import EventCategory, EventPriority, EventType


class FailingTextToSpeechAdapter(FakeTextToSpeechAdapter):
    def synthesize(self, request: SpeechRequest) -> tuple[SpeechChunk, ...]:
        raise RuntimeError("tts adapter failed")


def make_response_ready_event(
    *,
    text: str = "Yes sir.",
    response_id: str = "response-1",
    request_id: str = "dialogue-request-1",
) -> RuntimeEvent:
    return RuntimeEvent(
        event_type=EventType.ASSISTANT_RESPONSE_READY,
        category=EventCategory.DIALOGUE,
        source="test_dialogue_worker",
        payload={
            "response_id": response_id,
            "request_id": request_id,
            "text": text,
            "voice_id": "jarvis-test",
            "interruptible": True,
        },
    )


def test_tts_worker_requires_positive_tick_interval() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeTextToSpeechAdapter()

    with pytest.raises(ValueError):
        TTSWorker(
            event_bus=bus,
            tts_adapter=adapter,
            tick_interval_seconds=0,
        )


def test_tts_worker_rejects_empty_name() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeTextToSpeechAdapter()

    with pytest.raises(ValueError):
        TTSWorker(
            event_bus=bus,
            tts_adapter=adapter,
            name="   ",
        )


def test_tts_worker_rejects_empty_default_voice_id() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeTextToSpeechAdapter()

    with pytest.raises(ValueError):
        TTSWorker(
            event_bus=bus,
            tts_adapter=adapter,
            default_voice_id="   ",
        )


def test_tts_worker_subscribes_on_start() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeTextToSpeechAdapter()
    worker = TTSWorker(event_bus=bus, tts_adapter=adapter)

    worker.on_start()

    assert worker.tts_snapshot().subscribed is True

    worker.on_stop()

    assert adapter.reset_count == 1


def test_tts_worker_ignores_non_response_event() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeTextToSpeechAdapter()
    worker = TTSWorker(event_bus=bus, tts_adapter=adapter)

    event = RuntimeEvent(
        event_type=EventType.RUNTIME_TICK,
        category=EventCategory.RUNTIME,
        source="test",
    )

    worker.handle_response_ready_event(event)

    snapshot = worker.tts_snapshot()

    assert snapshot.ignored_events == 1
    assert snapshot.processed_responses == 0


def test_tts_worker_ignores_response_without_text() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeTextToSpeechAdapter()
    worker = TTSWorker(event_bus=bus, tts_adapter=adapter)

    event = RuntimeEvent(
        event_type=EventType.ASSISTANT_RESPONSE_READY,
        category=EventCategory.DIALOGUE,
        source="test",
        payload={"response_id": "response-1"},
    )

    worker.handle_response_ready_event(event)

    snapshot = worker.tts_snapshot()

    assert snapshot.ignored_events == 1
    assert snapshot.processed_responses == 0


def test_tts_worker_synthesizes_response_into_chunks() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeTextToSpeechAdapter(
        chunk_audio=(b"\x00\x01", b"\x02\x03"),
    )
    worker = TTSWorker(event_bus=bus, tts_adapter=adapter)

    chunks = worker.process_response(
        text="Yes sir.",
        response_id="response-1",
        dialogue_request_id="dialogue-request-1",
        voice_id="jarvis-test",
    )

    history = bus.history()
    snapshot = worker.tts_snapshot()

    assert len(chunks) == 2
    assert len(history) == 4
    assert history[0].event_type == EventType.TTS_SYNTHESIS_STARTED
    assert history[1].event_type == EventType.AUDIO_SPEECH_CHUNK_READY
    assert history[2].event_type == EventType.AUDIO_SPEECH_CHUNK_READY
    assert history[3].event_type == EventType.TTS_SYNTHESIS_COMPLETED
    assert history[1].priority == EventPriority.HIGH
    assert history[1].payload["chunk"] == chunks[0]
    assert history[1].payload["response_id"] == "response-1"
    assert history[3].payload["chunk_count"] == 2
    assert snapshot.processed_responses == 1
    assert snapshot.synthesis_started == 1
    assert snapshot.synthesis_completed == 1
    assert snapshot.chunks_published == 2
    assert snapshot.last_text == "Yes sir."
    assert snapshot.last_chunk_id == chunks[-1].chunk_id


def test_tts_worker_can_process_response_ready_event() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeTextToSpeechAdapter()
    worker = TTSWorker(event_bus=bus, tts_adapter=adapter)

    worker.handle_response_ready_event(make_response_ready_event(text="Done."))

    history = bus.history()

    assert worker.tts_snapshot().synthesis_completed == 1
    assert len(history) == 3
    assert history[0].payload["voice_id"] == "jarvis-test"
    assert history[1].payload["response_id"] == "response-1"


def test_tts_worker_subscribed_event_flow_with_publish_sync() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeTextToSpeechAdapter()
    worker = TTSWorker(event_bus=bus, tts_adapter=adapter)

    worker.on_start()
    bus.publish_sync(make_response_ready_event(text="Ready."))

    assert worker.tts_snapshot().synthesis_completed == 1
    assert len(bus.history()) == 4

    worker.on_stop()


def test_tts_worker_rejects_empty_text_direct_call() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeTextToSpeechAdapter()
    worker = TTSWorker(event_bus=bus, tts_adapter=adapter)

    with pytest.raises(ValueError):
        worker.process_response(text="   ")


def test_tts_worker_publishes_failure_event() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FailingTextToSpeechAdapter()
    worker = TTSWorker(event_bus=bus, tts_adapter=adapter)

    chunks = worker.process_response(
        text="This will fail.",
        response_id="response-1",
    )

    history = bus.history()
    snapshot = worker.tts_snapshot()

    assert chunks == ()
    assert len(history) == 2
    assert history[0].event_type == EventType.TTS_SYNTHESIS_STARTED
    assert history[1].event_type == EventType.TTS_SYNTHESIS_FAILED
    assert history[1].payload["error"] == "RuntimeError: tts adapter failed"
    assert snapshot.synthesis_failures == 1
    assert snapshot.synthesis_completed == 0
    assert snapshot.last_error is not None