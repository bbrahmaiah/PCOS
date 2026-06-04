from __future__ import annotations

import pytest

from jarvis.live import (
    LiveEventBridgeRuntime,
    LiveHealthMonitorOperation,
    LiveHealthMonitorResult,
    LiveHealthMonitorStatus,
    LiveHealthSignal,
    LiveHealthSignalKind,
    LiveRecoveryAction,
    LiveRecoveryOperation,
    LiveRecoveryPolicy,
    LiveRecoveryRuntime,
    LiveRecoveryRuntimeStatus,
    LiveSessionConfig,
    LiveSessionMode,
    LiveSessionStateRuntime,
    LiveSubsystem,
    utc_now,
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


def _health_result(
    *,
    status: LiveHealthMonitorStatus,
    signal: LiveHealthSignal | None = None,
) -> LiveHealthMonitorResult:
    signals = (signal,) if signal is not None else ()
    return LiveHealthMonitorResult(
        status=status,
        operation=LiveHealthMonitorOperation.CHECK,
        signals=signals,
        bridge_result=None,
        state_result=None,
        reason=status.value,
        created_at=utc_now(),
    )


def _signal(
    *,
    kind: LiveHealthSignalKind,
    subsystem: LiveSubsystem,
    status: LiveHealthMonitorStatus,
    message: str,
) -> LiveHealthSignal:
    return LiveHealthSignal(
        kind=kind,
        subsystem=subsystem,
        status=status,
        message=message,
        created_at=utc_now(),
    )


def test_live_recovery_policy_validation() -> None:
    with pytest.raises(ValueError):
        LiveRecoveryPolicy(max_recovery_attempts=0)


def test_live_recovery_evaluates_no_recovery_for_healthy() -> None:
    runtime = LiveRecoveryRuntime(live_state=_state())
    result = runtime.evaluate(
        _health_result(status=LiveHealthMonitorStatus.HEALTHY)
    )

    assert result.status == LiveRecoveryRuntimeStatus.READY
    assert result.plan is not None
    assert result.plan.action == LiveRecoveryAction.NONE


def test_live_recovery_plans_audio_restart_for_degraded_audio() -> None:
    runtime = LiveRecoveryRuntime(live_state=_state())
    result = runtime.evaluate(
        _health_result(
            status=LiveHealthMonitorStatus.DEGRADED,
            signal=_signal(
                kind=LiveHealthSignalKind.AUDIO,
                subsystem=LiveSubsystem.MICROPHONE,
                status=LiveHealthMonitorStatus.DEGRADED,
                message="audio degraded",
            ),
        )
    )

    assert result.plan is not None
    assert result.plan.action == LiveRecoveryAction.RESTART_AUDIO_BOUNDARY


def test_live_recovery_starts_recovery_and_routes_event() -> None:
    state = _state()
    bridge = LiveEventBridgeRuntime(live_state=state)
    runtime = LiveRecoveryRuntime(live_state=state, bridge=bridge)

    result = runtime.recover(
        _health_result(
            status=LiveHealthMonitorStatus.DEGRADED,
            signal=_signal(
                kind=LiveHealthSignalKind.EVENT_BRIDGE,
                subsystem=LiveSubsystem.EVENT_BUS,
                status=LiveHealthMonitorStatus.DEGRADED,
                message="bridge degraded",
            ),
        )
    )

    assert result.status == LiveRecoveryRuntimeStatus.READY
    assert result.operation == LiveRecoveryOperation.RECOVER
    assert result.plan is not None
    assert result.state_result is not None
    assert result.bridge_result is not None


def test_live_recovery_marks_recovered() -> None:
    state = _state()
    bridge = LiveEventBridgeRuntime(live_state=state)
    runtime = LiveRecoveryRuntime(live_state=state, bridge=bridge)

    runtime.recover(
        _health_result(
            status=LiveHealthMonitorStatus.DEGRADED,
            signal=_signal(
                kind=LiveHealthSignalKind.AUDIO,
                subsystem=LiveSubsystem.STT,
                status=LiveHealthMonitorStatus.DEGRADED,
                message="STT degraded",
            ),
        )
    )
    result = runtime.mark_recovered(
        subsystem=LiveSubsystem.STT,
        reason="STT ready again",
    )

    assert result.status == LiveRecoveryRuntimeStatus.READY
    assert result.state_result is not None
    assert result.bridge_result is not None
    snapshot = runtime.snapshot()
    assert snapshot.recovered_count == 1


def test_live_recovery_fails_session_for_failed_health() -> None:
    state = _state()
    runtime = LiveRecoveryRuntime(live_state=state)

    result = runtime.recover(
        _health_result(
            status=LiveHealthMonitorStatus.FAILED,
            signal=_signal(
                kind=LiveHealthSignalKind.SUBSYSTEM,
                subsystem=LiveSubsystem.STT,
                status=LiveHealthMonitorStatus.FAILED,
                message="STT failed",
            ),
        )
    )

    assert result.status == LiveRecoveryRuntimeStatus.FAILED
    assert result.state_result is not None
    assert result.state_result.state.health_status.value == "failed"


def test_live_recovery_abandon_marks_failure() -> None:
    runtime = LiveRecoveryRuntime(live_state=_state())

    result = runtime.abandon(
        subsystem=LiveSubsystem.STT,
        reason="manual abandon",
    )

    assert result.status == LiveRecoveryRuntimeStatus.FAILED


def test_live_recovery_snapshot_tracks_attempts() -> None:
    runtime = LiveRecoveryRuntime(live_state=_state())
    runtime.recover(
        _health_result(
            status=LiveHealthMonitorStatus.DEGRADED,
            signal=_signal(
                kind=LiveHealthSignalKind.AUDIO,
                subsystem=LiveSubsystem.MICROPHONE,
                status=LiveHealthMonitorStatus.DEGRADED,
                message="audio degraded",
            ),
        )
    )
    snapshot = runtime.snapshot()

    assert snapshot.status == LiveRecoveryRuntimeStatus.READY
    assert snapshot.attempt_count == 1
    assert snapshot.last_plan is not None


def test_live_recovery_blocks_after_max_attempts() -> None:
    runtime = LiveRecoveryRuntime(
        live_state=_state(),
        policy=LiveRecoveryPolicy(max_recovery_attempts=1),
    )
    health = _health_result(
        status=LiveHealthMonitorStatus.CRITICAL,
        signal=_signal(
            kind=LiveHealthSignalKind.AUDIO,
            subsystem=LiveSubsystem.MICROPHONE,
            status=LiveHealthMonitorStatus.CRITICAL,
            message="audio critical",
        ),
    )

    first = runtime.recover(health)
    second = runtime.recover(health)

    assert first.status == LiveRecoveryRuntimeStatus.READY
    assert second.status == LiveRecoveryRuntimeStatus.FAILED


def test_live_recovery_reason_validation() -> None:
    runtime = LiveRecoveryRuntime(live_state=_state())

    with pytest.raises(ValueError):
        runtime.mark_recovered(subsystem=LiveSubsystem.STT, reason=" ")

    with pytest.raises(ValueError):
        runtime.abandon(subsystem=LiveSubsystem.STT, reason=" ")


def test_live_recovery_enum_values_are_stable() -> None:
    assert LiveRecoveryRuntimeStatus.READY.value == "ready"
    assert LiveRecoveryAction.RESTART_AUDIO_BOUNDARY.value == (
        "restart_audio_boundary"
    )