from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.presence.models import (
    AudioFrame,
    PresenceMode,
    PresenceState,
    SpeechChunk,
    SpeechPriority,
    SpeechRequest,
    Transcript,
    TranscriptKind,
    TurnPhase,
    VoiceActivity,
    VoiceActivityState,
)


def test_audio_frame_validates_and_computes_duration() -> None:
    frame = AudioFrame(
        source="fake_microphone",
        audio_data=b"\x00\x01" * 160,
        sample_rate=16_000,
        channels=1,
        sample_width_bytes=2,
    )

    assert frame.source == "fake_microphone"
    assert frame.byte_count == 320
    assert frame.sample_count == 160
    assert frame.duration_ms == 10.0


def test_audio_frame_rejects_empty_audio() -> None:
    with pytest.raises(ValidationError):
        AudioFrame(
            source="fake_microphone",
            audio_data=b"",
        )


def test_voice_activity_contract() -> None:
    activity = VoiceActivity(
        frame_id="frame-1",
        state=VoiceActivityState.SPEECH_STARTED,
        is_speech=True,
        confidence=0.95,
        energy=0.8,
    )

    assert activity.frame_id == "frame-1"
    assert activity.is_speech is True
    assert activity.confidence == 0.95


def test_voice_activity_rejects_invalid_confidence() -> None:
    with pytest.raises(ValidationError):
        VoiceActivity(
            frame_id="frame-1",
            state=VoiceActivityState.SPEECH_STARTED,
            is_speech=True,
            confidence=2.0,
        )


def test_transcript_contract() -> None:
    transcript = Transcript(
        segment_id="segment-1",
        text="hello jarvis",
        kind=TranscriptKind.FINAL,
        confidence=0.99,
        language="en",
        alternatives=("hello service",),
    )

    assert transcript.text == "hello jarvis"
    assert transcript.is_final is True
    assert transcript.is_partial is False
    assert transcript.alternatives == ("hello service",)


def test_transcript_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        Transcript(
            segment_id="segment-1",
            text="   ",
            kind=TranscriptKind.FINAL,
        )


def test_speech_request_contract() -> None:
    request = SpeechRequest(
        text="Yes sir.",
        voice_id="jarvis-default",
        priority=SpeechPriority.HIGH,
        speed=1.1,
        interruptible=True,
        correlation_id="corr-1",
    )

    assert request.text == "Yes sir."
    assert request.voice_id == "jarvis-default"
    assert request.priority == SpeechPriority.HIGH
    assert request.interruptible is True


def test_speech_request_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        SpeechRequest(text="   ")


def test_speech_chunk_contract() -> None:
    chunk = SpeechChunk(
        request_id="request-1",
        audio_data=b"\x00\x01" * 100,
        chunk_index=1,
        final=True,
    )

    assert chunk.request_id == "request-1"
    assert chunk.byte_count == 200
    assert chunk.final is True


def test_speech_chunk_rejects_empty_audio() -> None:
    with pytest.raises(ValidationError):
        SpeechChunk(
            request_id="request-1",
            audio_data=b"",
        )


def test_presence_state_defaults() -> None:
    state = PresenceState()

    assert state.mode == PresenceMode.IDLE
    assert state.turn_phase == TurnPhase.NONE
    assert state.awake is False
    assert state.active is False
    assert state.interruptible is False


def test_presence_state_active_and_interruptible() -> None:
    state = PresenceState(
        mode=PresenceMode.ASSISTANT_SPEAKING,
        turn_phase=TurnPhase.SPEAKING_RESPONSE,
        awake=True,
        assistant_speaking=True,
        active_speech_request_id="speech-1",
    )

    assert state.active is True
    assert state.interruptible is True


def test_presence_models_are_frozen() -> None:
    frame = AudioFrame(
        source="fake_microphone",
        audio_data=b"\x00\x01",
    )

    with pytest.raises(ValidationError):
        frame.source = "other"