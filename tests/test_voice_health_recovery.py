from __future__ import annotations

from dataclasses import dataclass

from jarvis.voice import (
    VoiceActivityRuntimeStatus,
    VoiceActivitySnapshot,
    VoiceBargeInRuntimeStatus,
    VoiceBargeInSnapshot,
    VoiceCognitionSnapshot,
    VoiceCognitionStatus,
    VoiceHealthComponents,
    VoiceHealthPolicy,
    VoiceHealthRecoveryRuntime,
    VoiceHealthStatus,
    VoiceHealthSubsystem,
    VoiceMicrophoneCaptureSnapshot,
    VoiceMicrophoneCaptureStatus,
    VoicePlaybackRuntimeStatus,
    VoicePlaybackSnapshot,
    VoiceRecoveryAction,
    VoiceSTTRuntimeStatus,
    VoiceSTTSnapshot,
    VoiceTTSRuntimeStatus,
    VoiceTTSSnapshot,
    utc_now,
)


@dataclass
class FakeMicrophone:
    status: VoiceMicrophoneCaptureStatus = VoiceMicrophoneCaptureStatus.READY
    prepared: int = 0
    stopped: int = 0

    def snapshot(self) -> VoiceMicrophoneCaptureSnapshot:
        return VoiceMicrophoneCaptureSnapshot(
            status=self.status,
            device=None,
            captured_frames=1,
            captured_bytes=2,
            consecutive_failures=1 if self.status else 0,
            last_error="mic failed"
            if self.status == VoiceMicrophoneCaptureStatus.FAILED
            else None,
            created_at=utc_now(),
        )

    def prepare(self) -> object:
        self.prepared += 1
        self.status = VoiceMicrophoneCaptureStatus.READY
        return object()

    def stop(self) -> object:
        self.stopped += 1
        return object()


@dataclass
class FakeVad:
    status: VoiceActivityRuntimeStatus = VoiceActivityRuntimeStatus.READY
    resets: int = 0

    def snapshot(self) -> VoiceActivitySnapshot:
        return VoiceActivitySnapshot(
            status=self.status,
            current_segment_id=None,
            analyzed_frames=1,
            speech_segments=0,
            speech_ms=0,
            silence_ms=0,
            pending_start_frames=0,
            noise_floor=1.0,
            last_energy=1.0,
            created_at=utc_now(),
        )

    def reset(self) -> object:
        self.resets += 1
        self.status = VoiceActivityRuntimeStatus.READY
        return object()


@dataclass
class FakeSTT:
    status: VoiceSTTRuntimeStatus = VoiceSTTRuntimeStatus.READY
    prepared: int = 0
    resets: int = 0

    def snapshot(self) -> VoiceSTTSnapshot:
        return VoiceSTTSnapshot(
            status=self.status,
            partial_model=None,
            final_model=None,
            partial_transcripts=0,
            final_transcripts=0,
            empty_results=0,
            failed_results=1 if self.status == VoiceSTTRuntimeStatus.FAILED else 0,
            low_confidence_results=0,
            last_text=None,
            last_latency_ms=None,
            last_safety=None,
            last_error="stt failed"
            if self.status == VoiceSTTRuntimeStatus.FAILED
            else None,
            created_at=utc_now(),
        )

    def prepare(self) -> object:
        self.prepared += 1
        self.status = VoiceSTTRuntimeStatus.READY
        return object()

    def reset(self) -> object:
        self.resets += 1
        self.status = VoiceSTTRuntimeStatus.READY
        return object()


@dataclass
class FakeCognition:
    status: VoiceCognitionStatus = VoiceCognitionStatus.READY
    prepared: int = 0

    def snapshot(self) -> VoiceCognitionSnapshot:
        return VoiceCognitionSnapshot(
            status=self.status,
            prepared=True,
            responses=0,
            ignored=0,
            degraded=0,
            failed=1 if self.status == VoiceCognitionStatus.FAILED else 0,
            prefetches=0,
            last_text=None,
            last_response_text=None,
            last_latency_ms=None,
            last_context_latency_ms=None,
            last_response_latency_ms=None,
            last_safety=None,
            last_error="cognition failed"
            if self.status == VoiceCognitionStatus.FAILED
            else None,
            created_at=utc_now(),
        )

    def prepare(self) -> object:
        self.prepared += 1
        self.status = VoiceCognitionStatus.READY
        return object()


@dataclass
class FakeTTS:
    status: VoiceTTSRuntimeStatus = VoiceTTSRuntimeStatus.READY
    prepared: int = 0
    resets: int = 0

    def snapshot(self) -> VoiceTTSSnapshot:
        return VoiceTTSSnapshot(
            status=self.status,
            voice=None,
            synthesized_requests=0,
            synthesized_chunks=0,
            failed_requests=1 if self.status == VoiceTTSRuntimeStatus.FAILED else 0,
            degraded_requests=0,
            last_text=None,
            last_latency_ms=None,
            last_first_chunk_latency_ms=None,
            last_error="tts failed"
            if self.status == VoiceTTSRuntimeStatus.FAILED
            else None,
            created_at=utc_now(),
        )

    def prepare(self) -> object:
        self.prepared += 1
        self.status = VoiceTTSRuntimeStatus.READY
        return object()

    def reset(self) -> object:
        self.resets += 1
        self.status = VoiceTTSRuntimeStatus.READY
        return object()


@dataclass
class FakePlayback:
    status: VoicePlaybackRuntimeStatus = VoicePlaybackRuntimeStatus.READY
    prepared: int = 0
    resets: int = 0
    clears: int = 0
    stops: int = 0

    def snapshot(self) -> VoicePlaybackSnapshot:
        return VoicePlaybackSnapshot(
            status=self.status,
            speaker=None,
            queued_chunks=0,
            played_chunks=0,
            failed_chunks=1
            if self.status == VoicePlaybackRuntimeStatus.FAILED
            else 0,
            stopped_count=0,
            current_playback=None,
            last_latency_ms=None,
            last_first_audio_latency_ms=None,
            last_error="playback failed"
            if self.status == VoicePlaybackRuntimeStatus.FAILED
            else None,
            created_at=utc_now(),
        )

    def prepare(self) -> object:
        self.prepared += 1
        self.status = VoicePlaybackRuntimeStatus.READY
        return object()

    def reset(self) -> object:
        self.resets += 1
        self.status = VoicePlaybackRuntimeStatus.READY
        return object()

    def clear(self) -> object:
        self.clears += 1
        return object()

    def stop(self) -> object:
        self.stops += 1
        return object()


@dataclass
class FakeBargeIn:
    status: VoiceBargeInRuntimeStatus = VoiceBargeInRuntimeStatus.READY
    prepared: int = 0
    resets: int = 0

    def snapshot(self) -> VoiceBargeInSnapshot:
        return VoiceBargeInSnapshot(
            status=self.status,
            evaluated_transcripts=0,
            ignored=0,
            interruptions=0,
            failed_interruptions=1
            if self.status == VoiceBargeInRuntimeStatus.FAILED
            else 0,
            last_disposition=None,
            last_signal_text=None,
            last_latency_ms=None,
            last_stop_latency_ms=None,
            last_error="barge-in failed"
            if self.status == VoiceBargeInRuntimeStatus.FAILED
            else None,
            created_at=utc_now(),
        )

    def prepare(self) -> object:
        self.prepared += 1
        self.status = VoiceBargeInRuntimeStatus.READY
        return object()

    def reset(self) -> object:
        self.resets += 1
        self.status = VoiceBargeInRuntimeStatus.READY
        return object()


def _all_components() -> VoiceHealthComponents:
    return VoiceHealthComponents(
        microphone=FakeMicrophone(),
        vad=FakeVad(),
        stt=FakeSTT(),
        cognition=FakeCognition(),
        tts=FakeTTS(),
        playback=FakePlayback(),
        barge_in=FakeBargeIn(),
    )


def test_voice_health_policy_validation() -> None:
    try:
        VoiceHealthPolicy(max_degraded_subsystems=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected policy validation failure")


def test_voice_health_reports_healthy_when_all_connected() -> None:
    runtime = VoiceHealthRecoveryRuntime(components=_all_components())

    result = runtime.check()

    assert result.status == VoiceHealthStatus.HEALTHY
    assert result.healthy is True
    assert len(result.subsystem_health) == 7


def test_voice_health_reports_failed_when_required_subsystem_missing() -> None:
    runtime = VoiceHealthRecoveryRuntime(
        components=VoiceHealthComponents(),
    )

    result = runtime.check()

    assert result.status == VoiceHealthStatus.FAILED
    assert any(
        item.subsystem == VoiceHealthSubsystem.MICROPHONE
        and item.status == VoiceHealthStatus.FAILED
        for item in result.subsystem_health
    )


def test_voice_health_optional_vad_missing_is_degraded_only() -> None:
    components = _all_components()
    components.vad = None
    runtime = VoiceHealthRecoveryRuntime(components=components)

    result = runtime.check()

    assert result.status == VoiceHealthStatus.DEGRADED


def test_voice_health_detects_degraded_microphone() -> None:
    components = _all_components()
    mic = FakeMicrophone(status=VoiceMicrophoneCaptureStatus.DEGRADED)
    components.microphone = mic
    runtime = VoiceHealthRecoveryRuntime(components=components)

    result = runtime.check()

    assert result.status == VoiceHealthStatus.DEGRADED
    assert any(
        item.subsystem == VoiceHealthSubsystem.MICROPHONE
        and item.recommended_action == VoiceRecoveryAction.PREPARE
        for item in result.subsystem_health
    )


def test_voice_health_recovers_degraded_microphone() -> None:
    mic = FakeMicrophone(status=VoiceMicrophoneCaptureStatus.DEGRADED)
    components = _all_components()
    components.microphone = mic
    runtime = VoiceHealthRecoveryRuntime(components=components)

    result = runtime.recover()

    assert result.recovery_attempts
    assert mic.prepared == 1
    assert result.metadata["post_recovery_status"] == "healthy"


def test_voice_health_recovers_failed_stt_with_reset() -> None:
    stt = FakeSTT(status=VoiceSTTRuntimeStatus.FAILED)
    components = _all_components()
    components.stt = stt
    runtime = VoiceHealthRecoveryRuntime(components=components)

    result = runtime.recover()

    assert result.recovery_attempts
    assert stt.resets == 1
    assert result.metadata["post_recovery_status"] == "healthy"


def test_voice_health_recovers_failed_tts_with_reset() -> None:
    tts = FakeTTS(status=VoiceTTSRuntimeStatus.FAILED)
    components = _all_components()
    components.tts = tts
    runtime = VoiceHealthRecoveryRuntime(components=components)

    result = runtime.recover()

    assert result.recovery_attempts
    assert tts.resets == 1
    assert result.metadata["post_recovery_status"] == "healthy"


def test_voice_health_recovers_failed_playback_with_reset() -> None:
    playback = FakePlayback(status=VoicePlaybackRuntimeStatus.FAILED)
    components = _all_components()
    components.playback = playback
    runtime = VoiceHealthRecoveryRuntime(components=components)

    result = runtime.recover()

    assert result.recovery_attempts
    assert playback.resets == 1
    assert result.metadata["post_recovery_status"] == "healthy"


def test_voice_health_recovers_failed_barge_in_with_reset() -> None:
    barge_in = FakeBargeIn(status=VoiceBargeInRuntimeStatus.FAILED)
    components = _all_components()
    components.barge_in = barge_in
    runtime = VoiceHealthRecoveryRuntime(components=components)

    result = runtime.recover()

    assert result.recovery_attempts
    assert barge_in.resets == 1
    assert result.metadata["post_recovery_status"] == "healthy"


def test_voice_health_snapshot_tracks_counts() -> None:
    runtime = VoiceHealthRecoveryRuntime(components=_all_components())

    runtime.check()
    runtime.recover()
    snapshot = runtime.snapshot()

    assert snapshot.checks >= 2
    assert snapshot.recovery_runs == 1
    assert snapshot.last_latency_ms is not None


def test_voice_health_enum_values_are_stable() -> None:
    assert VoiceHealthStatus.HEALTHY.value == "healthy"
    assert VoiceHealthSubsystem.MICROPHONE.value == "microphone"
    assert VoiceRecoveryAction.RESET.value == "reset"