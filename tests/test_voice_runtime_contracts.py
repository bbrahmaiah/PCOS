from __future__ import annotations

import pytest

from jarvis.voice import (
    VoiceDeviceHealth,
    VoiceInputFrame,
    VoiceInputFrameKind,
    VoiceInterruptKind,
    VoiceInterruptSignal,
    VoiceLauncherResult,
    VoicePlaybackState,
    VoicePlaybackStatus,
    VoiceRuntimeConfig,
    VoiceRuntimeMode,
    VoiceRuntimeSnapshot,
    VoiceRuntimeStatus,
    VoiceSpeechSegment,
    VoiceSpeechSegmentStatus,
    VoiceTranscript,
    VoiceTranscriptKind,
    VoiceTTSChunk,
    VoiceTTSChunkStatus,
    VoiceTTSRequest,
    default_voice_runtime_state,
    make_voice_frame_id,
    make_voice_interrupt_id,
    make_voice_playback_id,
    make_voice_segment_id,
    make_voice_session_id,
    make_voice_transcript_id,
    make_voice_tts_chunk_id,
    utc_now,
)


def test_voice_runtime_config_defaults() -> None:
    config = VoiceRuntimeConfig()

    assert config.mode == VoiceRuntimeMode.REAL_VOICE
    assert config.assistant_name == "JARVIS"
    assert config.user_label == "Balu"
    assert config.wake_word == "jarvis"
    assert config.sample_rate_hz == 16_000


def test_voice_runtime_config_validation() -> None:
    with pytest.raises(ValueError):
        VoiceRuntimeConfig(assistant_name=" ")

    with pytest.raises(ValueError):
        VoiceRuntimeConfig(user_label=" ")

    with pytest.raises(ValueError):
        VoiceRuntimeConfig(wake_word=" ")

    with pytest.raises(ValueError):
        VoiceRuntimeConfig(sample_rate_hz=0)

    with pytest.raises(ValueError):
        VoiceRuntimeConfig(channels=2)


def test_default_voice_runtime_state() -> None:
    state = default_voice_runtime_state()

    assert state.status == VoiceRuntimeStatus.CREATED
    assert state.microphone_health == VoiceDeviceHealth.READY
    assert state.listening is False
    assert state.assistant_speaking is False


def test_voice_input_frame_validation() -> None:
    session_id = make_voice_session_id()
    frame = VoiceInputFrame(
        frame_id=make_voice_frame_id(),
        session_id=session_id,
        kind=VoiceInputFrameKind.PCM16_MONO,
        sample_rate_hz=16_000,
        channels=1,
        data=b"audio",
        captured_at=utc_now(),
        duration_ms=20,
    )

    assert frame.session_id == session_id

    with pytest.raises(ValueError):
        VoiceInputFrame(
            frame_id=make_voice_frame_id(),
            session_id=session_id,
            kind=VoiceInputFrameKind.PCM16_MONO,
            sample_rate_hz=16_000,
            channels=1,
            data=b"",
            captured_at=utc_now(),
            duration_ms=20,
        )


def test_voice_speech_segment_validation() -> None:
    session_id = make_voice_session_id()
    segment = VoiceSpeechSegment(
        segment_id=make_voice_segment_id(),
        session_id=session_id,
        status=VoiceSpeechSegmentStatus.STARTED,
        started_at=utc_now(),
        confidence=0.8,
        frame_count=3,
    )

    assert segment.confidence == 0.8

    with pytest.raises(ValueError):
        VoiceSpeechSegment(
            segment_id=make_voice_segment_id(),
            session_id=session_id,
            status=VoiceSpeechSegmentStatus.STARTED,
            started_at=utc_now(),
            confidence=2.0,
        )


def test_voice_transcript_validation() -> None:
    session_id = make_voice_session_id()
    segment_id = make_voice_segment_id()
    transcript = VoiceTranscript(
        transcript_id=make_voice_transcript_id(),
        session_id=session_id,
        segment_id=segment_id,
        kind=VoiceTranscriptKind.FINAL,
        text="Jarvis explain PID.",
        confidence=0.95,
        created_at=utc_now(),
    )

    assert transcript.text == "Jarvis explain PID."

    with pytest.raises(ValueError):
        VoiceTranscript(
            transcript_id=make_voice_transcript_id(),
            session_id=session_id,
            segment_id=segment_id,
            kind=VoiceTranscriptKind.FINAL,
            text=" ",
            confidence=0.95,
            created_at=utc_now(),
        )


def test_voice_tts_request_and_chunk_validation() -> None:
    session_id = make_voice_session_id()
    request = VoiceTTSRequest(
        session_id=session_id,
        text="Certainly.",
        voice="default",
        created_at=utc_now(),
    )

    assert request.text == "Certainly."

    chunk = VoiceTTSChunk(
        chunk_id=make_voice_tts_chunk_id(),
        session_id=session_id,
        status=VoiceTTSChunkStatus.SYNTHESIZED,
        audio=b"audio",
        sample_rate_hz=16_000,
        duration_ms=300,
        created_at=utc_now(),
    )

    assert chunk.duration_ms == 300

    with pytest.raises(ValueError):
        VoiceTTSChunk(
            chunk_id=make_voice_tts_chunk_id(),
            session_id=session_id,
            status=VoiceTTSChunkStatus.SYNTHESIZED,
            audio=b"",
            sample_rate_hz=16_000,
            duration_ms=300,
            created_at=utc_now(),
        )


def test_voice_playback_and_interrupt_signal() -> None:
    session_id = make_voice_session_id()
    playback = VoicePlaybackState(
        playback_id=make_voice_playback_id(),
        session_id=session_id,
        status=VoicePlaybackStatus.IDLE,
        chunk_id=None,
        started_at=None,
        stopped_at=None,
    )

    assert playback.status == VoicePlaybackStatus.IDLE

    interrupt = VoiceInterruptSignal(
        interrupt_id=make_voice_interrupt_id(),
        session_id=session_id,
        kind=VoiceInterruptKind.BARGE_IN,
        text="wait",
        confidence=0.95,
        created_at=utc_now(),
    )

    assert interrupt.kind == VoiceInterruptKind.BARGE_IN

    with pytest.raises(ValueError):
        VoiceInterruptSignal(
            interrupt_id=make_voice_interrupt_id(),
            session_id=session_id,
            kind=VoiceInterruptKind.BARGE_IN,
            text=" ",
            confidence=0.95,
            created_at=utc_now(),
        )


def test_voice_runtime_snapshot_and_launcher_result() -> None:
    state = default_voice_runtime_state()
    snapshot = VoiceRuntimeSnapshot(
        state=state,
        captured_frames=1,
        speech_segments=1,
        partial_transcripts=1,
        final_transcripts=1,
        tts_chunks=1,
        interruptions=0,
        created_at=utc_now(),
    )

    assert snapshot.captured_frames == 1

    result = VoiceLauncherResult(
        status=VoiceRuntimeStatus.LISTENING,
        state=state,
        message="voice runtime ready",
        created_at=utc_now(),
    )

    assert result.status == VoiceRuntimeStatus.LISTENING


def test_voice_enum_values_are_stable() -> None:
    assert VoiceRuntimeStatus.LISTENING.value == "listening"
    assert VoiceRuntimeMode.REAL_VOICE.value == "real_voice"
    assert VoiceTranscriptKind.INTERRUPTION.value == "interruption"