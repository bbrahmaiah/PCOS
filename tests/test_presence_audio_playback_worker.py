from __future__ import annotations

import pytest

from jarvis.presence.adapters import (
    FakeAudioPlaybackAdapter,
    PlaybackResult,
    PlaybackStatus,
)
from jarvis.presence.models import SpeechChunk
from jarvis.presence.state import PresenceStateStore
from jarvis.presence.workers import AudioPlaybackWorker
from jarvis.runtime.events import EventBus, RuntimeEvent
from jarvis.runtime.shared.enums import EventCategory, EventPriority, EventType


class RaisingAudioPlaybackAdapter(FakeAudioPlaybackAdapter):
    def play(self, chunk: SpeechChunk) -> PlaybackResult:
        raise RuntimeError("playback crashed")


def make_chunk(
    *,
    request_id: str = "request-1",
    chunk_index: int = 0,
    final: bool = True,
) -> SpeechChunk:
    return SpeechChunk(
        request_id=request_id,
        audio_data=b"\x00\x01",
        chunk_index=chunk_index,
        final=final,
    )


def make_chunk_event(chunk: SpeechChunk) -> RuntimeEvent:
    return RuntimeEvent(
        event_type=EventType.AUDIO_SPEECH_CHUNK_READY,
        category=EventCategory.PRESENCE,
        source="test_tts_worker",
        payload={
            "chunk": chunk,
            "chunk_id": chunk.chunk_id,
            "request_id": chunk.request_id,
        },
    )


def prepare_store_for_assistant_response() -> PresenceStateStore:
    bus = EventBus(name="presence_state_test_bus")
    store = PresenceStateStore(event_bus=bus)

    store.wake_detected(turn_id="turn-1")
    store.user_speech_started()
    store.user_speech_ended()
    store.transcript_ready()

    return store


def test_audio_playback_worker_requires_positive_tick_interval() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeAudioPlaybackAdapter()

    with pytest.raises(ValueError):
        AudioPlaybackWorker(
            event_bus=bus,
            playback_adapter=adapter,
            tick_interval_seconds=0,
        )


def test_audio_playback_worker_rejects_empty_name() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeAudioPlaybackAdapter()

    with pytest.raises(ValueError):
        AudioPlaybackWorker(
            event_bus=bus,
            playback_adapter=adapter,
            name="   ",
        )


def test_audio_playback_worker_subscribes_on_start() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeAudioPlaybackAdapter()
    worker = AudioPlaybackWorker(event_bus=bus, playback_adapter=adapter)

    worker.on_start()

    assert worker.playback_snapshot().subscribed is True

    worker.on_stop()


def test_audio_playback_worker_ignores_non_chunk_event() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeAudioPlaybackAdapter()
    worker = AudioPlaybackWorker(event_bus=bus, playback_adapter=adapter)

    event = RuntimeEvent(
        event_type=EventType.RUNTIME_TICK,
        category=EventCategory.RUNTIME,
        source="test",
    )

    worker.handle_speech_chunk_event(event)

    snapshot = worker.playback_snapshot()

    assert snapshot.ignored_events == 1
    assert snapshot.processed_chunks == 0


def test_audio_playback_worker_ignores_chunk_event_without_chunk_object() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeAudioPlaybackAdapter()
    worker = AudioPlaybackWorker(event_bus=bus, playback_adapter=adapter)

    event = RuntimeEvent(
        event_type=EventType.AUDIO_SPEECH_CHUNK_READY,
        category=EventCategory.PRESENCE,
        source="test",
        payload={"chunk_id": "missing-object"},
    )

    worker.handle_speech_chunk_event(event)

    snapshot = worker.playback_snapshot()

    assert snapshot.ignored_events == 1
    assert snapshot.processed_chunks == 0


def test_audio_playback_worker_plays_chunk() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeAudioPlaybackAdapter()
    worker = AudioPlaybackWorker(event_bus=bus, playback_adapter=adapter)
    chunk = make_chunk()

    result = worker.process_chunk(chunk=chunk)

    history = bus.history()
    snapshot = worker.playback_snapshot()

    assert result.status == PlaybackStatus.STARTED
    assert len(history) == 2
    assert history[0].event_type == EventType.ASSISTANT_SPEAKING_STARTED
    assert history[1].event_type == EventType.AUDIO_PLAYBACK_STARTED
    assert history[1].priority == EventPriority.HIGH
    assert history[1].payload["chunk_id"] == chunk.chunk_id
    assert history[1].payload["request_id"] == chunk.request_id
    assert snapshot.processed_chunks == 1
    assert snapshot.playback_started == 1
    assert snapshot.active_request_id == chunk.request_id
    assert snapshot.active_chunk_id == chunk.chunk_id
    assert snapshot.adapter_playing is True


def test_audio_playback_worker_can_process_chunk_event() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeAudioPlaybackAdapter()
    worker = AudioPlaybackWorker(event_bus=bus, playback_adapter=adapter)
    chunk = make_chunk()

    worker.handle_speech_chunk_event(make_chunk_event(chunk))

    assert worker.playback_snapshot().playback_started == 1
    assert len(bus.history()) == 2


def test_audio_playback_worker_subscribed_event_flow_with_publish_sync() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeAudioPlaybackAdapter()
    worker = AudioPlaybackWorker(event_bus=bus, playback_adapter=adapter)
    chunk = make_chunk()

    worker.on_start()
    bus.publish_sync(make_chunk_event(chunk))

    assert worker.playback_snapshot().playback_started == 1
    assert len(bus.history()) == 3

    worker.on_stop()


def test_audio_playback_worker_stop_playback() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeAudioPlaybackAdapter()
    worker = AudioPlaybackWorker(event_bus=bus, playback_adapter=adapter)
    chunk = make_chunk()

    worker.process_chunk(chunk=chunk)
    stopped = worker.stop_playback(reason="test_stop")

    snapshot = worker.playback_snapshot()

    assert stopped is not None
    assert stopped.status == PlaybackStatus.STOPPED
    assert snapshot.stop_requests == 1
    assert snapshot.playback_stopped == 1
    assert snapshot.active_request_id is None
    assert snapshot.adapter_playing is False
    assert any(
        event.event_type == EventType.AUDIO_PLAYBACK_STOPPED
        for event in bus.history()
    )
    assert any(
        event.event_type == EventType.ASSISTANT_SPEAKING_STOPPED
        for event in bus.history()
    )


def test_audio_playback_worker_stop_when_idle_returns_none() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeAudioPlaybackAdapter()
    worker = AudioPlaybackWorker(event_bus=bus, playback_adapter=adapter)

    result = worker.stop_playback()

    assert result is None
    assert worker.playback_snapshot().stop_requests == 1


def test_audio_playback_worker_updates_presence_store_on_start() -> None:
    bus = EventBus(name="presence_test_bus")
    store = prepare_store_for_assistant_response()
    adapter = FakeAudioPlaybackAdapter()
    worker = AudioPlaybackWorker(
        event_bus=bus,
        playback_adapter=adapter,
        presence_store=store,
    )
    chunk = make_chunk(request_id="speech-request-1")

    worker.process_chunk(chunk=chunk)

    state = store.current_state()

    assert state.assistant_speaking is True
    assert state.active_speech_request_id == "speech-request-1"


def test_audio_playback_worker_updates_presence_store_on_stop() -> None:
    bus = EventBus(name="presence_test_bus")
    store = prepare_store_for_assistant_response()
    adapter = FakeAudioPlaybackAdapter()
    worker = AudioPlaybackWorker(
        event_bus=bus,
        playback_adapter=adapter,
        presence_store=store,
    )
    chunk = make_chunk(request_id="speech-request-1")

    worker.process_chunk(chunk=chunk)
    worker.stop_playback(reason="finished")

    state = store.current_state()

    assert state.assistant_speaking is False
    assert state.active_speech_request_id is None


def test_audio_playback_worker_handles_playback_failure_result() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = FakeAudioPlaybackAdapter(fail_playback=True)
    worker = AudioPlaybackWorker(event_bus=bus, playback_adapter=adapter)
    chunk = make_chunk()

    result = worker.process_chunk(chunk=chunk)

    snapshot = worker.playback_snapshot()

    assert result.status == PlaybackStatus.FAILED
    assert snapshot.playback_failed == 1
    assert snapshot.last_error == "Fake playback failure."
    assert any(
        event.event_type == EventType.AUDIO_PLAYBACK_FAILED
        for event in bus.history()
    )


def test_audio_playback_worker_handles_adapter_exception() -> None:
    bus = EventBus(name="presence_test_bus")
    adapter = RaisingAudioPlaybackAdapter()
    worker = AudioPlaybackWorker(event_bus=bus, playback_adapter=adapter)
    chunk = make_chunk()

    result = worker.process_chunk(chunk=chunk)

    snapshot = worker.playback_snapshot()

    assert result.status == PlaybackStatus.FAILED
    assert snapshot.playback_failed == 1
    assert snapshot.last_error == "RuntimeError: playback crashed"
    assert any(
        event.event_type == EventType.AUDIO_PLAYBACK_FAILED
        for event in bus.history()
    )