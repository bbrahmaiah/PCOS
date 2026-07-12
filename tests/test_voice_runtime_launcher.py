from __future__ import annotations

from dataclasses import dataclass, field

from jarvis.voice import (
    VoiceDailyDriverGateReport,
    VoiceDailyDriverGateStatus,
    VoiceDailyDriverProfile,
    VoiceLiveSpineReport,
    VoiceLiveSpineStatus,
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
    metadata: dict[str, object] | None = None,
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
        metadata=metadata or {},
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
    stop_after_run_calls: int | None = None
    run_event: VoiceSessionLoopEvent | None = VoiceSessionLoopEvent.PLAYBACK_FINISHED
    run_calls: int = 0
    snapshot_calls: int = 0
    live_snapshot_calls: int = 0
    snapshot_metadata: dict[str, object] = field(default_factory=dict)

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
        if (
            self.stop_after_run_calls is not None
            and self.run_calls >= self.stop_after_run_calls
        ):
            return _session_result(
                status=VoiceSessionLoopStatus.STOPPED,
                operation=VoiceSessionLoopOperation.RUN,
                event=VoiceSessionLoopEvent.STOPPED,
                message="run_stopped",
            )
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
            event=self.run_event,
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
        self.snapshot_calls += 1
        return _session_snapshot(
            running=self.started and not self.stopped,
            metadata=self.snapshot_metadata,
        )

    def live_snapshot(self) -> VoiceSessionLoopSnapshot:
        self.live_snapshot_calls += 1
        return _session_snapshot(
            running=self.started and not self.stopped,
            metadata=self.snapshot_metadata,
        )


@dataclass
class FakeSpineMonitor:
    calls: int = 0

    def inspect(self, snapshot: VoiceSessionLoopSnapshot) -> VoiceLiveSpineReport:
        self.calls += 1
        return VoiceLiveSpineReport(
            status=VoiceLiveSpineStatus.HEALTHY,
            message="healthy",
            checks={"checked": True, "running": snapshot.running},
            created_at=utc_now(),
        )


def test_voice_runtime_launcher_config_validation() -> None:
    try:
        VoiceRuntimeLauncherConfig(bounded_cycles=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected invalid bounded_cycles to fail")

    try:
        VoiceRuntimeLauncherConfig(live_spine_monitor_every_cycles=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected invalid spine monitor interval to fail")


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


def test_voice_runtime_launcher_run_forever_preserves_session_failure() -> None:
    loop = FakeSessionLoop(fail_run=True)
    launcher = VoiceRuntimeLauncher(
        session_loop=loop,
        daily_driver_gate=FakeGate(),
        config=VoiceRuntimeLauncherConfig(
            run_forever=True,
            idle_sleep_seconds=0.0,
        ),
    )

    result = launcher.run()

    assert result.status == VoiceRuntimeLauncherStatus.FAILED
    assert result.reason == "voice session failed during run"
    assert result.session_result is not None
    assert result.session_result.status == VoiceSessionLoopStatus.FAILED
    assert loop.stopped is True


def test_voice_runtime_launcher_run_forever_exits_when_session_stops() -> None:
    loop = FakeSessionLoop(stop_after_run_calls=1)
    launcher = VoiceRuntimeLauncher(
        session_loop=loop,
        daily_driver_gate=FakeGate(),
        config=VoiceRuntimeLauncherConfig(
            run_forever=True,
            idle_sleep_seconds=0.0,
        ),
    )

    result = launcher.run()

    assert result.status == VoiceRuntimeLauncherStatus.STOPPED
    assert result.session_result is not None
    assert result.session_result.status == VoiceSessionLoopStatus.STOPPED
    assert loop.run_calls == 1
    assert loop.stopped is True


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


def test_voice_runtime_launcher_records_live_spine_report() -> None:
    loop = FakeSessionLoop(
        snapshot_metadata={
            "perception": {"packets": 1},
            "fsm_violations": 0,
            "playback_status": "ready",
        }
    )
    launcher = VoiceRuntimeLauncher(
        session_loop=loop,
        daily_driver_gate=FakeGate(),
        config=VoiceRuntimeLauncherConfig(run_forever=False),
    )

    result = launcher.run()
    snapshot = launcher.snapshot()

    assert result.status == VoiceRuntimeLauncherStatus.RUNNING
    assert "live_spine" in result.metadata
    assert snapshot.metadata["live_spine_monitor_enabled"] is True
    live_spine = snapshot.metadata["live_spine"]
    assert isinstance(live_spine, dict)
    assert live_spine["status"] == VoiceLiveSpineStatus.HEALTHY.value


def test_voice_runtime_launcher_throttles_spine_monitor_on_quiet_cycles() -> None:
    loop = FakeSessionLoop(
        stop_after_run_calls=5,
        run_event=None,
        snapshot_metadata={
            "perception": {"packets": 1},
            "fsm_violations": 0,
            "playback_status": "ready",
        },
    )
    monitor = FakeSpineMonitor()
    launcher = VoiceRuntimeLauncher(
        session_loop=loop,
        daily_driver_gate=FakeGate(),
        spine_monitor=monitor,
        config=VoiceRuntimeLauncherConfig(
            run_forever=True,
            idle_sleep_seconds=0.0,
            live_spine_monitor_every_cycles=50,
        ),
    )

    result = launcher.run()

    assert result.status == VoiceRuntimeLauncherStatus.STOPPED
    assert loop.run_calls == 5
    assert monitor.calls == 2
    assert loop.live_snapshot_calls == 2
    assert loop.snapshot_calls == 0


def test_voice_runtime_launcher_live_snapshot_is_lightweight() -> None:
    loop = FakeSessionLoop(
        snapshot_metadata={
            "perception": {"packets": 1},
            "fsm_violations": 0,
            "playback_status": "ready",
        }
    )
    launcher = VoiceRuntimeLauncher(
        session_loop=loop,
        daily_driver_gate=FakeGate(),
        config=VoiceRuntimeLauncherConfig(run_forever=False),
    )

    launcher.boot()
    snapshot = launcher.live_snapshot()

    assert snapshot.session_snapshot is not None
    assert loop.live_snapshot_calls == 1
    assert loop.snapshot_calls == 0


def test_voice_runtime_launcher_enum_values_are_stable() -> None:
    assert VoiceRuntimeLauncherStatus.RUNNING.value == "running"
    assert VoiceRuntimeLauncherOperation.RUN.value == "run"
    assert VoiceRuntimeLauncherEvent.SESSION_STARTED.value == "session_started"
