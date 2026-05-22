from __future__ import annotations

import pytest

from jarvis.presence.adapters import PlaybackResult, PlaybackStatus
from jarvis.presence.full_voice_smoke import (
    FullVoiceSmokeReport,
    FullVoiceSmokeTurn,
    make_canned_transcript,
)
from jarvis.presence.models import SpeechChunk, Transcript
from jarvis.presence.voice_tuning import (
    VoiceRuntimePreset,
    VoiceRuntimeProfile,
    WarmupSpeechSegment,
    build_full_voice_smoke_config,
    build_vad_adapter,
    format_full_voice_report,
    get_voice_runtime_profile,
    warm_stt_model,
)


class StubWarmableStt:
    def __init__(self, *, fail: bool = False, returns_transcript: bool = False) -> None:
        self.fail = fail
        self.returns_transcript = returns_transcript
        self.calls = 0

    def transcribe(self, segment: WarmupSpeechSegment) -> Transcript | None:
        self.calls += 1

        if self.fail:
            raise RuntimeError("warmup failed")

        if not self.returns_transcript:
            return None

        return make_canned_transcript(
            segment_id=segment.segment_id,
            text="warmup",
        )


def test_get_voice_runtime_profile_accepts_enum_and_string() -> None:
    fast_from_enum = get_voice_runtime_profile(VoiceRuntimePreset.FAST)
    fast_from_string = get_voice_runtime_profile("fast")

    assert fast_from_enum == fast_from_string
    assert fast_from_enum.stt_model == "tiny"


def test_get_voice_runtime_profile_rejects_unknown_name() -> None:
    with pytest.raises(ValueError):
        get_voice_runtime_profile("unknown")


def test_voice_runtime_profile_rejects_invalid_values() -> None:
    profile = VoiceRuntimeProfile(
        preset=VoiceRuntimePreset.FAST,
        stt_model=" ",
        vad_threshold=120.0,
        vad_silence_threshold=60.0,
        speech_start_frames=2,
        speech_end_frames=10,
        adaptive_vad=False,
        duration_seconds=35.0,
        max_frames=4_000,
        response_text="Yes sir.",
    )

    with pytest.raises(ValueError):
        profile.validate()


def test_build_vad_adapter_from_profile() -> None:
    profile = get_voice_runtime_profile("fast")
    vad = build_vad_adapter(profile)

    assert vad.name == "energy_vad_adapter"


def test_build_full_voice_smoke_config_from_profile() -> None:
    profile = get_voice_runtime_profile("fast")
    config = build_full_voice_smoke_config(
        profile,
        require_wake=False,
        keep_listening=False,
    )

    assert config.duration_seconds == profile.duration_seconds
    assert config.max_frames == profile.max_frames
    assert config.require_wake is False
    assert config.stop_after_first_turn is True
    assert config.response_text == profile.response_text


def test_build_full_voice_smoke_config_supports_overrides() -> None:
    profile = get_voice_runtime_profile("fast")
    config = build_full_voice_smoke_config(
        profile,
        require_wake=True,
        keep_listening=True,
        duration_seconds=10.0,
        response_text="Online.",
    )

    assert config.duration_seconds == 10.0
    assert config.require_wake is True
    assert config.stop_after_first_turn is False
    assert config.response_text == "Online."


def test_warm_stt_model_reports_success_without_transcript() -> None:
    stt = StubWarmableStt(returns_transcript=False)

    result = warm_stt_model(stt)

    assert result.completed is True
    assert result.transcript_returned is False
    assert result.error is None
    assert stt.calls == 1


def test_warm_stt_model_reports_success_with_transcript() -> None:
    stt = StubWarmableStt(returns_transcript=True)

    result = warm_stt_model(stt)

    assert result.completed is True
    assert result.transcript_returned is True
    assert result.error is None
    assert stt.calls == 1


def test_warm_stt_model_reports_failure() -> None:
    stt = StubWarmableStt(fail=True)

    result = warm_stt_model(stt)

    assert result.completed is False
    assert result.transcript_returned is False
    assert result.error == "RuntimeError: warmup failed"


def test_format_full_voice_report() -> None:
    transcript = make_canned_transcript(text="hello jarvis")
    chunk = SpeechChunk(
        request_id="request-1",
        chunk_id="chunk-1",
        audio_data=b"\x00\x01",
        sample_rate=16_000,
        channels=1,
        final=True,
        metadata={},
    )
    playback_result = PlaybackResult(
        result_id="result-1",
        chunk_id="chunk-1",
        request_id="request-1",
        status=PlaybackStatus.STARTED,
        metadata={},
    )
    turn = FullVoiceSmokeTurn(
        transcript=transcript,
        response_text="Yes sir.",
        chunks=(chunk,),
        playback_results=(playback_result,),
    )
    report = FullVoiceSmokeReport(
        passed=True,
        started_at=transcript.created_at,
        finished_at=transcript.created_at,
        duration_ms=100.0,
        frames_read=10,
        wake_detected=True,
        speech_completed=True,
        turns=(turn,),
        errors=(),
    )

    output = format_full_voice_report(report)

    assert "Passed: True" in output
    assert "heard: hello jarvis" in output
    assert "response: Yes sir." in output
    assert "playback: started" in output