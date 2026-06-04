from __future__ import annotations

import pytest

from jarvis.live import (
    LiveAudioRuntimeSnapshot,
    LiveAudioRuntimeStatus,
    LiveDialogueRuntime,
    LiveEventBridgeRuntime,
    LiveHealthMonitorPolicy,
    LiveHealthMonitorRuntime,
    LiveHealthMonitorStatus,
    LiveHealthSignal,
    LiveHealthSignalKind,
    LiveInterruptionRuntime,
    LiveResponseBoundaryRuntime,
    LiveResponseDraft,
    LiveResponseGenerationRequest,
    LiveResponseGenerationSource,
    LiveSessionConfig,
    LiveSessionMode,
    LiveSessionStateRuntime,
    LiveSubsystem,
    LiveSubsystemState,
    LiveSubsystemStatus,
    utc_now,
)


class HealthyAudioRuntime:
    def snapshot(self) -> LiveAudioRuntimeSnapshot:
        from jarvis.live import LiveAudioRuntimeSnapshot

        return LiveAudioRuntimeSnapshot(
            status=LiveAudioRuntimeStatus.READY,
            prepared=True,
            captured_frames=1,
            transcripts=1,
            synthesized_responses=1,
            played_responses=1,
            blocked_count=0,
            created_at=utc_now(),
        )

class HealthFakeGenerator:
    def generate(
        self,
        request: LiveResponseGenerationRequest,
    ) -> LiveResponseDraft:
        return LiveResponseDraft(
            text="Generated health-monitor test response.",
            generation_source=LiveResponseGenerationSource.RESPONSE_GENERATOR,
            token_count=4,
        )


class UnpreparedAudioRuntime:
    def snapshot(self) -> LiveAudioRuntimeSnapshot:

        return LiveAudioRuntimeSnapshot(
            status=LiveAudioRuntimeStatus.READY,
            prepared=False,
            captured_frames=0,
            transcripts=0,
            synthesized_responses=0,
            played_responses=0,
            blocked_count=0,
            created_at=utc_now(),
        )


def _state() -> LiveSessionStateRuntime:
    state = LiveSessionStateRuntime(
        config=LiveSessionConfig(
            mode=LiveSessionMode.REAL_VOICE,
            real_microphone_enabled=True,
            real_stt_enabled=True,
            real_tts_enabled=True,
        )
    )
    state.start()
    state.mark_ready()
    return state


def test_health_monitor_policy_validation() -> None:
    with pytest.raises(ValueError):
        LiveHealthMonitorPolicy(max_blocked_before_degraded=0)

    with pytest.raises(ValueError):
        LiveHealthMonitorPolicy(
            max_blocked_before_degraded=5,
            max_blocked_before_critical=3,
        )


def test_health_monitor_reports_degraded_when_optional_runtimes_missing() -> None:
    state = _state()
    monitor = LiveHealthMonitorRuntime(live_state=state)

    result = monitor.check()

    assert result.status == LiveHealthMonitorStatus.DEGRADED
    assert result.needs_recovery is True
    assert result.signals
    assert result.bridge_result is not None


def test_health_monitor_reports_healthy_with_connected_runtimes() -> None:
    state = _state()
    bridge = LiveEventBridgeRuntime(live_state=state)
    response_boundary = LiveResponseBoundaryRuntime()
    dialogue = LiveDialogueRuntime(
        live_state=state,
        bridge=bridge,
        response_generator=HealthFakeGenerator(),
    )
    interruption = LiveInterruptionRuntime(
        live_state=state,
        bridge=bridge,
        dialogue=dialogue,
    )
    monitor = LiveHealthMonitorRuntime(
        live_state=state,
        bridge=bridge,
        audio=HealthyAudioRuntime(),  # type: ignore[arg-type]
        response_boundary=response_boundary,
        dialogue=dialogue,
        interruption=interruption,
    )

    result = monitor.check()

    assert result.status == LiveHealthMonitorStatus.HEALTHY
    assert result.healthy is True


def test_health_monitor_detects_unprepared_audio_in_voice_mode() -> None:
    state = _state()
    bridge = LiveEventBridgeRuntime(live_state=state)
    response_boundary = LiveResponseBoundaryRuntime()
    dialogue = LiveDialogueRuntime(
        live_state=state,
        bridge=bridge,
        response_generator=HealthFakeGenerator(),
    )
    interruption = LiveInterruptionRuntime(
        live_state=state,
        bridge=bridge,
        dialogue=dialogue,
    )
    monitor = LiveHealthMonitorRuntime(
        live_state=state,
        bridge=bridge,
        audio=UnpreparedAudioRuntime(),  # type: ignore[arg-type]
        response_boundary=response_boundary,
        dialogue=dialogue,
        interruption=interruption,
    )

    result = monitor.check()

    assert result.status == LiveHealthMonitorStatus.CRITICAL
    assert result.needs_recovery is True


def test_health_monitor_records_failed_subsystem_signal() -> None:
    state = _state()
    monitor = LiveHealthMonitorRuntime(live_state=state)
    signal = LiveHealthSignal(
        kind=LiveHealthSignalKind.SUBSYSTEM,
        subsystem=LiveSubsystem.STT,
        status=LiveHealthMonitorStatus.FAILED,
        message="STT failed.",
        created_at=utc_now(),
    )

    result = monitor.record_signal(signal)

    assert result.status == LiveHealthMonitorStatus.FAILED
    assert result.state_result is not None
    assert result.state_result.state.health_status.value == "failed"


def test_health_monitor_reads_failed_session_subsystem() -> None:
    state = _state()
    state.update_subsystem(
        LiveSubsystemState(
            subsystem=LiveSubsystem.STT,
            status=LiveSubsystemStatus.FAILED,
            message="STT failed.",
            updated_at=utc_now(),
        )
    )
    monitor = LiveHealthMonitorRuntime(
        live_state=state,
        audio=HealthyAudioRuntime(),  # type: ignore[arg-type]
        response_boundary=LiveResponseBoundaryRuntime(),
    )

    result = monitor.check()

    assert result.status == LiveHealthMonitorStatus.FAILED


def test_health_monitor_snapshot_tracks_signals() -> None:
    state = _state()
    monitor = LiveHealthMonitorRuntime(live_state=state)

    monitor.record_signal(
        LiveHealthSignal(
            kind=LiveHealthSignalKind.SUBSYSTEM,
            subsystem=LiveSubsystem.STT,
            status=LiveHealthMonitorStatus.DEGRADED,
            message="STT degraded.",
            created_at=utc_now(),
        )
    )
    snapshot = monitor.snapshot()

    assert snapshot.status == LiveHealthMonitorStatus.DEGRADED
    assert snapshot.signal_count == 1
    assert snapshot.degraded_count == 1


def test_health_monitor_enum_values_are_stable() -> None:
    assert LiveHealthMonitorStatus.HEALTHY.value == "healthy"
    assert LiveHealthSignalKind.AUDIO.value == "audio"