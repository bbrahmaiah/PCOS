from __future__ import annotations

import time
from collections.abc import Callable

import pytest

from jarvis.presence import PresenceEngine, PresenceEngineAdapters
from jarvis.presence.adapters import (
    FakeAudioPlaybackAdapter,
    FakeMicrophoneAdapter,
    FakeSpeechToTextAdapter,
    FakeTextToSpeechAdapter,
    FakeVoiceActivityAdapter,
    FakeWakeWordAdapter,
    make_fake_audio_frame,
)
from jarvis.presence.models import VoiceActivityState
from jarvis.runtime.shared.enums import EventType


def wait_until(
    predicate: Callable[[], bool],
    *,
    timeout_seconds: float = 1.0,
    interval_seconds: float = 0.01,
) -> None:
    """
    Wait for async EventBus delivery in integration tests.

    PresenceEngine intentionally uses async delivery for normal pipeline
    events. Tests wait for the runtime to settle instead of assuming every
    subscriber finished immediately after publish().
    """

    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        if predicate():
            return

        time.sleep(interval_seconds)

    assert predicate()


def history_contains(engine: PresenceEngine, event_type: EventType) -> bool:
    return any(
        event.event_type == event_type
        for event in engine.event_bus.history()
    )


def prepare_presence_for_response(engine: PresenceEngine) -> None:
    """
    Move PresenceStateStore into the valid state for assistant speech.

    Valid transition chain:
        idle
        -> listening
        -> user_speaking
        -> processing_speech/transcribing
        -> processing_speech/waiting_for_response

    Then AudioPlaybackWorker can legally transition the state into
    assistant_speaking when playback starts.
    """

    engine.presence_store.wake_detected(turn_id="turn-1")
    engine.presence_store.user_speech_started()
    engine.presence_store.user_speech_ended()
    engine.presence_store.transcript_ready()


def make_engine_with_scripted_audio() -> PresenceEngine:
    frames = (
        make_fake_audio_frame(frame_index=0),
        make_fake_audio_frame(frame_index=1),
        make_fake_audio_frame(frame_index=2),
    )

    adapters = PresenceEngineAdapters(
        microphone=FakeMicrophoneAdapter(frames=frames),
        wake_word=FakeWakeWordAdapter(detection_pattern=(True, False, False)),
        vad=FakeVoiceActivityAdapter(
            states=(
                VoiceActivityState.SPEECH_STARTED,
                VoiceActivityState.SPEECH_CONTINUING,
                VoiceActivityState.SPEECH_ENDED,
            ),
        ),
        stt=FakeSpeechToTextAdapter(text="hello jarvis"),
        tts=FakeTextToSpeechAdapter(),
        playback=FakeAudioPlaybackAdapter(),
    )

    return PresenceEngine(adapters=adapters)


def test_presence_engine_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        PresenceEngine(name="   ")


def test_presence_engine_builds_default_fake_runtime() -> None:
    engine = PresenceEngine()

    snapshot = engine.snapshot()

    assert snapshot.name == "presence_engine"
    assert snapshot.running is False
    assert snapshot.worker_count == 8
    assert engine.workers.voice_input.name == "voice_input_worker"
    assert engine.workers.interruption.name == "interruption_worker"


def test_presence_engine_start_stop() -> None:
    engine = PresenceEngine()

    engine.start()

    try:
        started = engine.snapshot()

        assert started.running is True
        assert started.started is True
        assert engine.workers.wake_detector.wake_snapshot().subscribed is True
        assert engine.workers.vad.vad_snapshot().subscribed is True
        assert engine.workers.stt.stt_snapshot().subscribed is True
        assert (
            engine.workers.dialogue_bridge.dialogue_snapshot().subscribed
            is True
        )
        assert engine.workers.tts.tts_snapshot().subscribed is True
        assert engine.workers.audio_playback.playback_snapshot().subscribed is True
        assert (
            engine.workers.interruption.interruption_snapshot().subscribed
            is True
        )

    finally:
        engine.stop()

    stopped = engine.snapshot()

    assert stopped.running is False
    assert stopped.stopped is True


def test_presence_engine_voice_to_transcript_pipeline() -> None:
    engine = make_engine_with_scripted_audio()

    engine.start()

    try:
        engine.workers.voice_input.run_once()
        engine.workers.voice_input.run_once()
        engine.workers.voice_input.run_once()

        wait_until(
            lambda: history_contains(
                engine,
                EventType.PRESENCE_TRANSCRIPT_FINAL,
            )
        )
        wait_until(
            lambda: history_contains(
                engine,
                EventType.ASSISTANT_RESPONSE_REQUESTED,
            )
        )

        history = engine.event_bus.history()

        assert any(
            event.event_type == EventType.WAKE_WORD_DETECTED
            or event.event_type == EventType.PRESENCE_WAKE_DETECTED
            for event in history
        )
        assert any(
            event.event_type == EventType.AUDIO_SPEECH_SEGMENT_COMPLETED
            for event in history
        )
        assert any(
            event.event_type == EventType.PRESENCE_TRANSCRIPT_FINAL
            for event in history
        )
        assert any(
            event.event_type == EventType.ASSISTANT_RESPONSE_REQUESTED
            for event in history
        )

    finally:
        engine.stop()


def test_presence_engine_response_ready_to_playback_pipeline() -> None:
    engine = make_engine_with_scripted_audio()

    engine.start()

    try:
        prepare_presence_for_response(engine)

        engine.publish_response_ready(text="Yes sir.")

        wait_until(
            lambda: history_contains(
                engine,
                EventType.AUDIO_PLAYBACK_STARTED,
            )
        )

        history = engine.event_bus.history()
        playback_snapshot = engine.workers.audio_playback.playback_snapshot()

        assert any(
            event.event_type == EventType.TTS_SYNTHESIS_COMPLETED
            for event in history
        )
        assert any(
            event.event_type == EventType.AUDIO_SPEECH_CHUNK_READY
            for event in history
        )
        assert any(
            event.event_type == EventType.AUDIO_PLAYBACK_STARTED
            for event in history
        )
        assert playback_snapshot.playback_started == 1
        assert engine.presence_store.current_state().assistant_speaking is True

    finally:
        engine.stop()


def test_presence_engine_interruption_pipeline() -> None:
    engine = make_engine_with_scripted_audio()

    engine.start()

    try:
        prepare_presence_for_response(engine)

        engine.publish_response_ready(text="I am speaking now.")

        wait_until(lambda: engine.adapters.playback.is_playing is True)
        wait_until(
            lambda: engine.presence_store.current_state().assistant_speaking
            is True
        )

        engine.workers.interruption.process_user_started_speaking(
            segment_id="interrupt-segment",
            frame_id="interrupt-frame",
        )

        wait_until(
            lambda: history_contains(
                engine,
                EventType.AUDIO_PLAYBACK_STOPPED,
            )
        )

        history = engine.event_bus.history()
        playback_snapshot = engine.workers.audio_playback.playback_snapshot()
        state = engine.presence_store.current_state()

        assert any(
            event.event_type == EventType.INTERRUPT_REQUESTED
            for event in history
        )
        assert any(
            event.event_type == EventType.AUDIO_PLAYBACK_STOPPED
            for event in history
        )
        assert playback_snapshot.playback_stopped == 1
        assert engine.adapters.playback.is_playing is False
        assert state.assistant_speaking is False

    finally:
        engine.stop()


def test_presence_engine_rejects_empty_response_text() -> None:
    engine = PresenceEngine()

    with pytest.raises(ValueError):
        engine.publish_response_ready(text="   ")