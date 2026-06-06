from __future__ import annotations

from dataclasses import dataclass

from jarvis.voice import (
    VoiceDailyDriverGateReport,
    VoiceDailyDriverGateStatus,
    VoiceDailyDriverProfile,
    VoiceRuntimeLauncher,
    VoiceRuntimeLauncherConfig,
    VoiceRuntimeLauncherEvent,
    VoiceRuntimeLauncherOperation,
    VoiceRuntimeLauncherStatus,
    VoiceSessionLoopEvent,
    VoiceSessionLoopOperation,
    VoiceSessionLoopResult,
    VoiceSessionLoopSnapshot,
    VoiceSessionLoopStatus,
    utc_now,
)


def _session_result(
    *,
    status: VoiceSessionLoopStatus,
    operation: VoiceSessionLoopOperation,
    event: VoiceSessionLoopEvent | None = None,
    message: str = "session_result",
) -> VoiceSessionLoopResult:
    return VoiceSessionLoopResult(
        status=status,
        operation=operation,
        event=event,
        transcript=None,
        cognition_result=None,
        tts_result=None,
        playback_result=None,
        barge_in_result=None,
        health_result=None,
        message=message,
        latency_ms=1.0,
        created_at=utc_now(),
    )


def _session_snapshot(
    *,
    running: bool = False,
) -> VoiceSessionLoopSnapshot:
    return VoiceSessionLoopSnapshot(
        status=VoiceSessionLoopStatus.LISTENING
        if running
        else VoiceSessionLoopStatus.STOPPED,
        running=running,
        assistant_speaking=False,
        cycles=1,
        captured_frames=1,
        speech_segments=0,
        partial_transcripts=0,
        final_transcripts=0,
        responses=0,
        tts_outputs=0,
        played_outputs=0,
        interruptions=0,
        recoveries=0,
        consecutive_failures=0,
        buffered_segment_frames=0,
        last_event=None,
        last_transcript_text=None,
        last_response_text=None,
        last_latency_ms=1.0,
        last_error=None,
        created_at=utc_now(),
    )


def _gate_report(
    status: VoiceDailyDriverGateStatus,
) -> VoiceDailyDriverGateReport:
    return VoiceDailyDriverGateReport(
        status=status,
        profile=VoiceDailyDriverProfile.BRONZE,
        checks=(),
        started_at=utc_now(),
        finished_at=utc_now(),
        latency_ms=1.0,
    )


@dataclass
class FakeGate:
    status: VoiceDailyDriverGateStatus = VoiceDailyDriverGateStatus.PASSED
    calls: int = 0

    def run(self) -> VoiceDailyDriverGateReport:
        self.calls += 1
        return _gate_report(self.status)


@dataclass
class FakeSessionLoop:
    started: bool = False
    stopped: bool = False
    fail_start: bool = False
    fail_run: bool = False
    run_calls: int = 0

    def start(self) -> VoiceSessionLoopResult:
        self.started = True
        if self.fail_start:
            return _session_result(
                status=VoiceSessionLoopStatus.FAILED,
                operation=VoiceSessionLoopOperation.START,
                event=VoiceSessionLoopEvent.ERROR,
                message="start_failed",
            )
        return _session_result(
            status=VoiceSessionLoopStatus.LISTENING,
            operation=VoiceSessionLoopOperation.START,
            event=VoiceSessionLoopEvent.STARTED,
            message="start_ok",
        )

    def run(
        self,
        *,
        max_cycles: int | None = None,
        max_seconds: float | None = None,
    ) -> VoiceSessionLoopResult:
        self.run_calls += 1
        if self.fail_run:
            return _session_result(
                status=VoiceSessionLoopStatus.FAILED,
                operation=VoiceSessionLoopOperation.RUN,
                event=VoiceSessionLoopEvent.ERROR,
                message="run_failed",
            )
        return _session_result(
            status=VoiceSessionLoopStatus.LISTENING,
            operation=VoiceSessionLoopOperation.RUN,
            event=VoiceSessionLoopEvent.PLAYBACK_FINISHED,
            message="run_ok",
        )

    def stop(self) -> VoiceSessionLoopResult:
        self.stopped = True
        return _session_result(
            status=VoiceSessionLoopStatus.STOPPED,
            operation=VoiceSessionLoopOperation.STOP,
            event=VoiceSessionLoopEvent.STOPPED,
            message="stop_ok",
        )

    def snapshot(self) -> VoiceSessionLoopSnapshot:
        return _session_snapshot(running=self.started and not self.stopped)


def test_voice_runtime_launcher_config_validation() -> None:
    try:
        VoiceRuntimeLauncherConfig(bounded_cycles=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected invalid bounded_cycles to fail")


def test_voice_runtime_launcher_boot_runs_daily_driver_gate() -> None:
    gate = FakeGate()
    launcher = VoiceRuntimeLauncher(
        session_loop=FakeSessionLoop(),
        daily_driver_gate=gate,
        config=VoiceRuntimeLauncherConfig(run_forever=False),
    )

    result = launcher.boot()

    assert result.status == VoiceRuntimeLauncherStatus.READY
    assert result.event == VoiceRuntimeLauncherEvent.DAILY_DRIVER_GATE_PASSED
    assert gate.calls == 1
    assert launcher.snapshot().booted is True


def test_voice_runtime_launcher_boot_fails_when_gate_fails() -> None:
    launcher = VoiceRuntimeLauncher(
        session_loop=FakeSessionLoop(),
        daily_driver_gate=FakeGate(status=VoiceDailyDriverGateStatus.FAILED),
        config=VoiceRuntimeLauncherConfig(run_forever=False),
    )

    result = launcher.boot()

    assert result.status == VoiceRuntimeLauncherStatus.FAILED
    assert result.event == VoiceRuntimeLauncherEvent.DAILY_DRIVER_GATE_FAILED


def test_voice_runtime_launcher_blocks_degraded_gate_by_default() -> None:
    launcher = VoiceRuntimeLauncher(
        session_loop=FakeSessionLoop(),
        daily_driver_gate=FakeGate(status=VoiceDailyDriverGateStatus.DEGRADED),
        config=VoiceRuntimeLauncherConfig(run_forever=False),
    )

    result = launcher.boot()

    assert result.status == VoiceRuntimeLauncherStatus.FAILED
    assert result.event == VoiceRuntimeLauncherEvent.DAILY_DRIVER_GATE_DEGRADED


def test_voice_runtime_launcher_can_allow_degraded_gate() -> None:
    launcher = VoiceRuntimeLauncher(
        session_loop=FakeSessionLoop(),
        daily_driver_gate=FakeGate(status=VoiceDailyDriverGateStatus.DEGRADED),
        config=VoiceRuntimeLauncherConfig(
            run_forever=False,
            allow_degraded_gate=True,
        ),
    )

    result = launcher.boot()

    assert result.status == VoiceRuntimeLauncherStatus.DEGRADED


def test_voice_runtime_launcher_runs_bounded_real_voice_path() -> None:
    loop = FakeSessionLoop()
    launcher = VoiceRuntimeLauncher(
        session_loop=loop,
        daily_driver_gate=FakeGate(),
        config=VoiceRuntimeLauncherConfig(
            run_forever=False,
            bounded_cycles=2,
        ),
    )

    result = launcher.run()

    assert result.status == VoiceRuntimeLauncherStatus.RUNNING
    assert result.operation == VoiceRuntimeLauncherOperation.RUN
    assert loop.started is True
    assert loop.run_calls == 1


def test_voice_runtime_launcher_fails_when_session_start_fails() -> None:
    launcher = VoiceRuntimeLauncher(
        session_loop=FakeSessionLoop(fail_start=True),
        daily_driver_gate=FakeGate(),
        config=VoiceRuntimeLauncherConfig(run_forever=False),
    )

    result = launcher.run()

    assert result.status == VoiceRuntimeLauncherStatus.FAILED
    assert result.reason == "voice session failed to start"


def test_voice_runtime_launcher_fails_when_session_run_fails() -> None:
    launcher = VoiceRuntimeLauncher(
        session_loop=FakeSessionLoop(fail_run=True),
        daily_driver_gate=FakeGate(),
        config=VoiceRuntimeLauncherConfig(run_forever=False),
    )

    result = launcher.run()

    assert result.status == VoiceRuntimeLauncherStatus.FAILED
    assert result.reason == "voice session failed during run"


def test_voice_runtime_launcher_stop_stops_session() -> None:
    loop = FakeSessionLoop()
    launcher = VoiceRuntimeLauncher(
        session_loop=loop,
        daily_driver_gate=FakeGate(),
        config=VoiceRuntimeLauncherConfig(run_forever=False),
    )

    launcher.boot()
    result = launcher.stop()

    assert result.status == VoiceRuntimeLauncherStatus.STOPPED
    assert result.event == VoiceRuntimeLauncherEvent.SESSION_STOPPED
    assert loop.stopped is True


def test_voice_runtime_launcher_snapshot_tracks_state() -> None:
    launcher = VoiceRuntimeLauncher(
        session_loop=FakeSessionLoop(),
        daily_driver_gate=FakeGate(),
        config=VoiceRuntimeLauncherConfig(run_forever=False),
    )

    launcher.boot()
    snapshot = launcher.snapshot()

    assert snapshot.booted is True
    assert snapshot.boot_count == 1
    assert snapshot.session_snapshot is not None


def test_voice_runtime_launcher_enum_values_are_stable() -> None:
    assert VoiceRuntimeLauncherStatus.RUNNING.value == "running"
    assert VoiceRuntimeLauncherOperation.RUN.value == "run"
    assert VoiceRuntimeLauncherEvent.SESSION_STARTED.value == "session_started"