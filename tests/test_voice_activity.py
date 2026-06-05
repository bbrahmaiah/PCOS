from __future__ import annotations

import pytest

from jarvis.voice import (
    VoiceActivityDecision,
    VoiceActivityPolicy,
    VoiceActivityRuntime,
    VoiceActivityRuntimeStatus,
    VoiceInputFrame,
    VoiceInputFrameKind,
    VoiceSpeechSegmentStatus,
    make_voice_frame_id,
    make_voice_session_id,
    utc_now,
)


def _pcm16_frame(amplitude: int, *, duration_ms: int = 20) -> VoiceInputFrame:
    sample = int(amplitude).to_bytes(
        2,
        byteorder="little",
        signed=True,
    )
    samples = 320
    return VoiceInputFrame(
        frame_id=make_voice_frame_id(),
        session_id=make_voice_session_id(),
        kind=VoiceInputFrameKind.PCM16_MONO,
        sample_rate_hz=16_000,
        channels=1,
        data=sample * samples,
        captured_at=utc_now(),
        duration_ms=duration_ms,
    )


def test_voice_activity_policy_validation() -> None:
    with pytest.raises(ValueError):
        VoiceActivityPolicy(min_energy=0)

    with pytest.raises(ValueError):
        VoiceActivityPolicy(start_trigger_frames=0)

    with pytest.raises(ValueError):
        VoiceActivityPolicy(end_silence_ms=0)


def test_voice_activity_detects_silence() -> None:
    runtime = VoiceActivityRuntime(
        policy=VoiceActivityPolicy(min_energy=500.0)
    )

    result = runtime.analyze_frame(_pcm16_frame(20))

    assert result.decision == VoiceActivityDecision.SILENCE
    assert result.segment is None


def test_voice_activity_waits_for_start_confirmation() -> None:
    runtime = VoiceActivityRuntime(
        policy=VoiceActivityPolicy(
            min_energy=500.0,
            start_trigger_frames=2,
        )
    )

    first = runtime.analyze_frame(_pcm16_frame(2_000))
    second = runtime.analyze_frame(_pcm16_frame(2_000))

    assert first.decision == VoiceActivityDecision.NOISE
    assert second.decision == VoiceActivityDecision.SPEECH_STARTED
    assert second.segment is not None
    assert second.segment.status == VoiceSpeechSegmentStatus.STARTED


def test_voice_activity_continues_speech() -> None:
    runtime = VoiceActivityRuntime(
        policy=VoiceActivityPolicy(
            min_energy=500.0,
            start_trigger_frames=1,
        )
    )

    start = runtime.analyze_frame(_pcm16_frame(2_000))
    continued = runtime.analyze_frame(_pcm16_frame(2_000))

    assert start.speech_started is True
    assert continued.decision == VoiceActivityDecision.SPEECH_CONTINUED
    assert continued.segment is not None
    assert continued.segment.status == VoiceSpeechSegmentStatus.ACTIVE


def test_voice_activity_holds_through_short_pause() -> None:
    runtime = VoiceActivityRuntime(
        policy=VoiceActivityPolicy(
            min_energy=500.0,
            start_trigger_frames=1,
            min_speech_ms=100,
            end_silence_ms=300,
        )
    )

    runtime.analyze_frame(_pcm16_frame(2_000, duration_ms=100))
    pause = runtime.analyze_frame(_pcm16_frame(20, duration_ms=100))

    assert pause.decision == VoiceActivityDecision.HOLDING_FOR_COMPLETION
    assert pause.holding_for_completion is True


def test_voice_activity_ends_after_natural_silence() -> None:
    runtime = VoiceActivityRuntime(
        policy=VoiceActivityPolicy(
            min_energy=500.0,
            start_trigger_frames=1,
            min_speech_ms=100,
            end_silence_ms=300,
        )
    )

    runtime.analyze_frame(_pcm16_frame(2_000, duration_ms=100))
    runtime.analyze_frame(_pcm16_frame(20, duration_ms=100))
    runtime.analyze_frame(_pcm16_frame(20, duration_ms=100))
    ended = runtime.analyze_frame(_pcm16_frame(20, duration_ms=100))

    assert ended.decision == VoiceActivityDecision.SPEECH_ENDED
    assert ended.speech_ended is True
    assert ended.segment is not None
    assert ended.segment.status == VoiceSpeechSegmentStatus.ENDED


def test_voice_activity_reset_clears_segment() -> None:
    runtime = VoiceActivityRuntime(
        policy=VoiceActivityPolicy(
            min_energy=500.0,
            start_trigger_frames=1,
        )
    )

    runtime.analyze_frame(_pcm16_frame(2_000))
    result = runtime.reset()
    snapshot = runtime.snapshot()

    assert result.decision == VoiceActivityDecision.SILENCE
    assert snapshot.current_segment_id is None
    assert snapshot.status == VoiceActivityRuntimeStatus.READY


def test_voice_activity_snapshot_tracks_counts() -> None:
    runtime = VoiceActivityRuntime(
        policy=VoiceActivityPolicy(
            min_energy=500.0,
            start_trigger_frames=1,
        )
    )

    runtime.analyze_frame(_pcm16_frame(20))
    runtime.analyze_frame(_pcm16_frame(2_000))
    snapshot = runtime.snapshot()

    assert snapshot.analyzed_frames == 2
    assert snapshot.speech_segments == 1
    assert snapshot.last_energy > 0


def test_voice_activity_invalid_frame_degrades() -> None:
    runtime = VoiceActivityRuntime()
    frame = VoiceInputFrame(
        frame_id=make_voice_frame_id(),
        session_id=make_voice_session_id(),
        kind=VoiceInputFrameKind.PCM16_MONO,
        sample_rate_hz=16_000,
        channels=1,
        data=b"\x00",
        captured_at=utc_now(),
        duration_ms=20,
    )

    result = runtime.analyze_frame(frame)

    assert result.status == VoiceActivityRuntimeStatus.DEGRADED
    assert result.decision == VoiceActivityDecision.NOISE


def test_voice_activity_enum_values_are_stable() -> None:
    assert VoiceActivityDecision.SPEECH_STARTED.value == "speech_started"
    assert VoiceActivityDecision.SPEECH_ENDED.value == "speech_ended"
    assert VoiceActivityRuntimeStatus.SPEECH_ACTIVE.value == "speech_active"