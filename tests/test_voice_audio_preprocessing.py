from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from jarvis.voice import (
    PassthroughAudioPreprocessor,
    VoiceAudioPreprocessingPolicy,
    VoiceAudioPreprocessingRuntime,
    VoiceAudioPreprocessingStatus,
    VoiceInputFrame,
    VoiceInputFrameKind,
    WebRTCVADAudioPreprocessor,
    make_voice_frame_id,
    make_voice_session_id,
    utc_now,
)


class FakeWebRTCVad:
    def __init__(self, aggressiveness: int) -> None:
        self.aggressiveness = aggressiveness

    def is_speech(self, audio: bytes, sample_rate: int) -> bool:
        del sample_rate
        return any(byte != 0 for byte in audio)


def _frame(
    data: bytes = b"\x01\x00" * 320,
    *,
    duration_ms: int = 20,
) -> VoiceInputFrame:
    return VoiceInputFrame(
        frame_id=make_voice_frame_id(),
        session_id=make_voice_session_id(),
        kind=VoiceInputFrameKind.PCM16_MONO,
        sample_rate_hz=16_000,
        channels=1,
        data=data,
        captured_at=utc_now(),
        duration_ms=duration_ms,
    )


def test_passthrough_audio_preprocessor_preserves_frame() -> None:
    runtime = VoiceAudioPreprocessingRuntime(
        adapter=PassthroughAudioPreprocessor()
    )
    frame = _frame()

    result = runtime.process_frame(frame)

    assert result.status == VoiceAudioPreprocessingStatus.READY
    assert result.frame is frame
    assert result.metadata["audio_preprocessed"] is False
    assert runtime.snapshot()["processed_frames"] == 1


def test_passthrough_fails_when_aec_is_required() -> None:
    runtime = VoiceAudioPreprocessingRuntime(
        adapter=PassthroughAudioPreprocessor(),
        policy=VoiceAudioPreprocessingPolicy(require_echo_cancellation=True),
    )

    result = runtime.process_frame(_frame())

    assert result.status == VoiceAudioPreprocessingStatus.FAILED
    assert result.metadata["missing_features"] == ("echo_cancellation",)


def test_webrtc_vad_marks_speech_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "webrtcvad",
        SimpleNamespace(Vad=FakeWebRTCVad),
    )
    runtime = VoiceAudioPreprocessingRuntime(
        adapter=WebRTCVADAudioPreprocessor(),
        policy=VoiceAudioPreprocessingPolicy(vad_aggressiveness=3),
    )

    result = runtime.process_frame(_frame())

    assert result.status == VoiceAudioPreprocessingStatus.READY
    assert result.frame is not None
    assert result.frame.metadata["audio_preprocessor"] == "webrtcvad"
    assert result.frame.metadata["webrtc_vad_speech"] is True
    assert result.frame.metadata["vad_aggressiveness"] == 3


def test_webrtc_vad_can_drop_non_speech_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "webrtcvad",
        SimpleNamespace(Vad=FakeWebRTCVad),
    )
    runtime = VoiceAudioPreprocessingRuntime(
        adapter=WebRTCVADAudioPreprocessor(),
        policy=VoiceAudioPreprocessingPolicy(drop_non_speech=True),
    )

    result = runtime.process_frame(_frame(data=b"\x00\x00" * 320))

    assert result.status == VoiceAudioPreprocessingStatus.DROPPED
    assert result.frame is None
    assert result.metadata["webrtc_vad_speech"] is False
    assert runtime.snapshot()["dropped_frames"] == 1


def test_webrtc_vad_reports_aec_unavailable_when_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "webrtcvad",
        SimpleNamespace(Vad=FakeWebRTCVad),
    )
    runtime = VoiceAudioPreprocessingRuntime(
        adapter=WebRTCVADAudioPreprocessor(),
        policy=VoiceAudioPreprocessingPolicy(
            require_echo_cancellation=True,
            require_noise_suppression=True,
            require_auto_gain_control=True,
        ),
    )

    result = runtime.process_frame(_frame())

    assert result.status == VoiceAudioPreprocessingStatus.FAILED
    assert result.metadata["missing_features"] == (
        "echo_cancellation",
        "noise_suppression",
        "auto_gain_control",
    )
