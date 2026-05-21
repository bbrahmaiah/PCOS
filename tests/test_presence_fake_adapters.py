from __future__ import annotations

import pytest

from jarvis.presence.adapters import (
    FakeAudioPlaybackAdapter,
    FakeMicrophoneAdapter,
    FakeSpeechToTextAdapter,
    FakeTextToSpeechAdapter,
    FakeVoiceActivityAdapter,
    FakeWakeWordAdapter,
    PlaybackStatus,
    make_fake_audio_frame,
)
from jarvis.presence.models import (
    SpeechRequest,
    TranscriptKind,
    VoiceActivityState,
)


def test_make_fake_audio_frame_creates_valid_frame() -> None:
    frame = make_fake_audio_frame(frame_index=7)

    assert frame.source == "fake_microphone"
    assert frame.frame_index == 7
    assert frame.byte_count == 2
    assert frame.sample_rate == 16_000


def test_fake_microphone_requires_start_before_reading() -> None:
    frame = make_fake_audio_frame()
    adapter = FakeMicrophoneAdapter(frames=(frame,))

    assert adapter.is_running is False
    assert adapter.remaining_frames == 1
    assert adapter.read_frame() is None

    adapter.start()

    read = adapter.read_frame()

    assert read is not None
    assert read.frame_id == frame.frame_id
    assert adapter.remaining_frames == 0

    adapter.stop()

    assert adapter.is_running is False


def test_fake_microphone_can_push_frames() -> None:
    adapter = FakeMicrophoneAdapter()
    frame = make_fake_audio_frame(frame_index=3)

    adapter.push_frame(frame)
    adapter.start()

    assert adapter.remaining_frames == 1
    assert adapter.read_frame() == frame


def test_fake_wake_word_adapter_pattern() -> None:
    frame = make_fake_audio_frame()
    adapter = FakeWakeWordAdapter(detection_pattern=(False, True))

    assert adapter.detect(frame) is None

    detection = adapter.detect(frame)

    assert detection is not None
    assert detection.frame_id == frame.frame_id
    assert detection.wake_word == "jarvis"
    assert detection.confidence == 0.99
    assert adapter.detect_calls == 2


def test_fake_wake_word_adapter_rejects_empty_wake_word() -> None:
    with pytest.raises(ValueError):
        FakeWakeWordAdapter(wake_word="   ")


def test_fake_wake_word_default_detect() -> None:
    frame = make_fake_audio_frame()
    adapter = FakeWakeWordAdapter(default_detect=True)

    detection = adapter.detect(frame)

    assert detection is not None
    assert detection.wake_word == "jarvis"


def test_fake_vad_adapter_emits_configured_states() -> None:
    frame = make_fake_audio_frame()
    adapter = FakeVoiceActivityAdapter(
        states=(
            VoiceActivityState.SILENCE,
            VoiceActivityState.SPEECH_STARTED,
        )
    )

    silence = adapter.analyze(frame)
    speech = adapter.analyze(frame)

    assert silence.is_speech is False
    assert silence.energy == 0.0
    assert speech.is_speech is True
    assert speech.state == VoiceActivityState.SPEECH_STARTED
    assert adapter.analyze_calls == 2


def test_fake_stt_transcribes_non_empty_frames() -> None:
    frame = make_fake_audio_frame()
    adapter = FakeSpeechToTextAdapter(
        text="open the browser",
        kind=TranscriptKind.FINAL,
    )

    transcript = adapter.transcribe((frame,))

    assert transcript.text == "open the browser"
    assert transcript.is_final is True
    assert adapter.transcribe_calls == 1
    assert adapter.last_frame_count == 1


def test_fake_stt_rejects_empty_frames() -> None:
    adapter = FakeSpeechToTextAdapter()

    with pytest.raises(ValueError):
        adapter.transcribe(())


def test_fake_tts_creates_chunks() -> None:
    adapter = FakeTextToSpeechAdapter(
        chunk_audio=(b"\x00\x01", b"\x02\x03"),
    )
    request = SpeechRequest(text="Yes sir.")

    chunks = adapter.synthesize(request)

    assert len(chunks) == 2
    assert chunks[0].request_id == request.request_id
    assert chunks[0].final is False
    assert chunks[1].final is True
    assert adapter.synthesize_calls == 1
    assert adapter.last_request_id == request.request_id


def test_fake_playback_records_start_stop() -> None:
    tts = FakeTextToSpeechAdapter()
    playback = FakeAudioPlaybackAdapter()
    request = SpeechRequest(text="Yes sir.")
    chunk = tts.synthesize(request)[0]

    started = playback.play(chunk)

    assert started.status == PlaybackStatus.STARTED
    assert playback.is_playing is True
    assert len(playback.played_chunks) == 1

    stopped = playback.stop()

    assert stopped is not None
    assert stopped.status == PlaybackStatus.STOPPED
    assert playback.is_playing is False
    assert len(playback.stop_results) == 1


def test_fake_playback_can_complete_current_chunk() -> None:
    tts = FakeTextToSpeechAdapter()
    playback = FakeAudioPlaybackAdapter()
    request = SpeechRequest(text="Done.")
    chunk = tts.synthesize(request)[0]

    playback.play(chunk)
    completed = playback.complete_current()

    assert completed is not None
    assert completed.status == PlaybackStatus.COMPLETED
    assert playback.is_playing is False


def test_fake_playback_failure_result() -> None:
    tts = FakeTextToSpeechAdapter()
    playback = FakeAudioPlaybackAdapter(fail_playback=True)
    request = SpeechRequest(text="Failure test.")
    chunk = tts.synthesize(request)[0]

    result = playback.play(chunk)

    assert result.status == PlaybackStatus.FAILED
    assert result.error == "Fake playback failure."
    assert playback.is_playing is False