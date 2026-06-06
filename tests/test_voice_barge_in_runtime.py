from __future__ import annotations

from dataclasses import dataclass

import pytest

from jarvis.voice import (
    VoiceBargeInDisposition,
    VoiceBargeInPolicy,
    VoiceBargeInRequest,
    VoiceBargeInRuntime,
    VoiceBargeInRuntimeStatus,
    VoiceDeviceHealth,
    VoiceInterruptKind,
    VoiceInterruptSignal,
    VoicePlaybackAdapterReport,
    VoicePlaybackPolicy,
    VoicePlaybackRuntime,
    VoicePlaybackRuntimeStatus,
    VoiceRuntimeConfig,
    VoiceSpeakerDeviceInfo,
    VoiceTranscript,
    VoiceTranscriptKind,
    VoiceTTSChunk,
    VoiceTTSChunkStatus,
    make_voice_interrupt_id,
    make_voice_segment_id,
    make_voice_session_id,
    make_voice_transcript_id,
    make_voice_tts_chunk_id,
    utc_now,
)


@dataclass
class FakeSpeakerAdapter:
    stopped: bool = False
    fail_stop: bool = False

    def prepare(
        self,
        config: VoiceRuntimeConfig,
        policy: VoicePlaybackPolicy,
    ) -> VoiceSpeakerDeviceInfo:
        return VoiceSpeakerDeviceInfo(
            provider="fake",
            name="Fake Speaker",
            sample_rate_hz=22_050,
            channels=1,
            health=VoiceDeviceHealth.READY,
        )

    def play(
        self,
        chunk: VoiceTTSChunk,
        policy: VoicePlaybackPolicy,
    ) -> VoicePlaybackAdapterReport:
        return VoicePlaybackAdapterReport(
            played=True,
            sample_rate_hz=chunk.sample_rate_hz,
            duration_ms=chunk.duration_ms,
            latency_ms=10.0,
            bytes_played=len(chunk.audio),
        )

    def stop(self) -> None:
        if self.fail_stop:
            raise RuntimeError("stop failed")
        self.stopped = True

    def close(self) -> None:
        return None


def _transcript(
    text: str,
    *,
    confidence: float = 0.95,
    kind: VoiceTranscriptKind = VoiceTranscriptKind.PARTIAL,
) -> VoiceTranscript:
    return VoiceTranscript(
        transcript_id=make_voice_transcript_id(),
        session_id=make_voice_session_id(),
        segment_id=make_voice_segment_id(),
        kind=kind,
        text=text,
        confidence=confidence,
        created_at=utc_now(),
    )


def _chunk() -> VoiceTTSChunk:
    return VoiceTTSChunk(
        chunk_id=make_voice_tts_chunk_id(),
        session_id=make_voice_session_id(),
        status=VoiceTTSChunkStatus.SYNTHESIZED,
        audio=b"RIFFfakewav",
        sample_rate_hz=22_050,
        duration_ms=300,
        created_at=utc_now(),
    )


def _playback(
    *,
    fail_stop: bool = False,
) -> VoicePlaybackRuntime:
    runtime = VoicePlaybackRuntime(adapter=FakeSpeakerAdapter(fail_stop=fail_stop))
    runtime.enqueue_chunk(_chunk())
    return runtime


def test_barge_in_policy_validation() -> None:
    with pytest.raises(ValueError):
        VoiceBargeInPolicy(min_confidence=2.0)

    with pytest.raises(ValueError):
        VoiceBargeInPolicy(max_stop_latency_ms=0)


def test_barge_in_prepare_requires_playback_controller() -> None:
    runtime = VoiceBargeInRuntime()

    result = runtime.prepare()

    assert result.status == VoiceBargeInRuntimeStatus.DEGRADED


def test_barge_in_ignores_when_assistant_not_speaking() -> None:
    runtime = VoiceBargeInRuntime(playback=_playback())

    result = runtime.evaluate_transcript(
        VoiceBargeInRequest(
            transcript=_transcript("wait"),
            assistant_speaking=False,
        )
    )

    assert result.status == VoiceBargeInRuntimeStatus.IGNORED
    assert result.playback_result is None


def test_barge_in_stops_playback_on_wait() -> None:
    playback = _playback()
    runtime = VoiceBargeInRuntime(playback=playback)

    result = runtime.evaluate_transcript(
        VoiceBargeInRequest(
            transcript=_transcript("wait"),
            assistant_speaking=True,
            active_response_text="PID control has three terms.",
        )
    )

    assert result.status == VoiceBargeInRuntimeStatus.INTERRUPTED
    assert result.disposition == VoiceBargeInDisposition.STOP_PLAYBACK
    assert result.playback_result is not None
    assert result.playback_result.status == VoicePlaybackRuntimeStatus.STOPPED
    assert result.interrupted_context is not None
    assert result.interrupted_context.response_text == (
        "PID control has three terms."
    )


def test_barge_in_cancel_response() -> None:
    runtime = VoiceBargeInRuntime(playback=_playback())

    result = runtime.evaluate_transcript(
        VoiceBargeInRequest(
            transcript=_transcript("cancel that"),
            assistant_speaking=True,
        )
    )

    assert result.status == VoiceBargeInRuntimeStatus.INTERRUPTED
    assert result.disposition == VoiceBargeInDisposition.CANCEL_RESPONSE
    assert result.signal is not None
    assert result.signal.kind == VoiceInterruptKind.CANCEL


def test_barge_in_user_correction() -> None:
    runtime = VoiceBargeInRuntime(playback=_playback())

    result = runtime.evaluate_transcript(
        VoiceBargeInRequest(
            transcript=_transcript("actually compare PID with LQR"),
            assistant_speaking=True,
        )
    )

    assert result.status == VoiceBargeInRuntimeStatus.INTERRUPTED
    assert result.disposition == VoiceBargeInDisposition.USER_CORRECTION
    assert result.signal is not None
    assert result.signal.kind == VoiceInterruptKind.USER_CORRECTION


def test_barge_in_new_question() -> None:
    runtime = VoiceBargeInRuntime(playback=_playback())

    result = runtime.evaluate_transcript(
        VoiceBargeInRequest(
            transcript=_transcript("what does integral mean"),
            assistant_speaking=True,
        )
    )

    assert result.status == VoiceBargeInRuntimeStatus.INTERRUPTED
    assert result.disposition == VoiceBargeInDisposition.NEW_QUESTION
    assert result.signal is not None
    assert result.signal.kind == VoiceInterruptKind.BARGE_IN


def test_barge_in_ignores_low_confidence_noise() -> None:
    runtime = VoiceBargeInRuntime(playback=_playback())

    result = runtime.evaluate_transcript(
        VoiceBargeInRequest(
            transcript=_transcript("maybe something", confidence=0.2),
            assistant_speaking=True,
        )
    )

    assert result.status == VoiceBargeInRuntimeStatus.IGNORED
    assert result.disposition == VoiceBargeInDisposition.IGNORE


def test_barge_in_allows_low_confidence_emergency_stop() -> None:
    runtime = VoiceBargeInRuntime(playback=_playback())

    result = runtime.evaluate_transcript(
        VoiceBargeInRequest(
            transcript=_transcript("stop", confidence=0.25),
            assistant_speaking=True,
        )
    )

    assert result.status == VoiceBargeInRuntimeStatus.INTERRUPTED
    assert result.disposition == VoiceBargeInDisposition.STOP_PLAYBACK


def test_barge_in_evaluate_signal_stops_playback() -> None:
    runtime = VoiceBargeInRuntime(playback=_playback())
    signal = VoiceInterruptSignal(
        interrupt_id=make_voice_interrupt_id(),
        session_id=make_voice_session_id(),
        kind=VoiceInterruptKind.STOP,
        text="stop",
        confidence=0.9,
        created_at=utc_now(),
    )

    result = runtime.evaluate_signal(signal)

    assert result.status == VoiceBargeInRuntimeStatus.INTERRUPTED
    assert result.disposition == VoiceBargeInDisposition.STOP_PLAYBACK


def test_barge_in_stop_failure_is_safe() -> None:
    runtime = VoiceBargeInRuntime(playback=_playback(fail_stop=True))

    result = runtime.evaluate_transcript(
        VoiceBargeInRequest(
            transcript=_transcript("wait"),
            assistant_speaking=True,
        )
    )

    assert result.status == VoiceBargeInRuntimeStatus.FAILED
    assert result.metadata["error"] == "stop failed"


def test_barge_in_snapshot_tracks_counts() -> None:
    runtime = VoiceBargeInRuntime(playback=_playback())

    runtime.evaluate_transcript(
        VoiceBargeInRequest(
            transcript=_transcript("wait"),
            assistant_speaking=True,
        )
    )
    snapshot = runtime.snapshot()

    assert snapshot.evaluated_transcripts == 1
    assert snapshot.interruptions == 1
    assert snapshot.last_signal_text == "wait"
    assert snapshot.last_stop_latency_ms is not None


def test_barge_in_reset() -> None:
    runtime = VoiceBargeInRuntime(playback=_playback())

    runtime.evaluate_transcript(
        VoiceBargeInRequest(
            transcript=_transcript("wait"),
            assistant_speaking=True,
        )
    )
    result = runtime.reset()
    snapshot = runtime.snapshot()

    assert result.status == VoiceBargeInRuntimeStatus.CREATED
    assert snapshot.last_signal_text is None


def test_barge_in_enum_values_are_stable() -> None:
    assert VoiceBargeInRuntimeStatus.INTERRUPTED.value == "interrupted"
    assert VoiceBargeInDisposition.USER_CORRECTION.value == "user_correction"