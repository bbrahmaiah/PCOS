from __future__ import annotations

from dataclasses import dataclass

from jarvis.voice import (
    VoiceDailyDriverCheckKind,
    VoiceDailyDriverContextProbeResult,
    VoiceDailyDriverContextSource,
    VoiceDailyDriverGate,
    VoiceDailyDriverGateConfig,
    VoiceDailyDriverGateStatus,
    VoiceDailyDriverProfile,
    VoiceSessionLoopEvent,
    VoiceSessionLoopOperation,
    VoiceSessionLoopResult,
    VoiceSessionLoopSnapshot,
    VoiceSessionLoopStatus,
    utc_now,
)


def _loop_result(
    *,
    status: VoiceSessionLoopStatus,
    operation: VoiceSessionLoopOperation,
    event: VoiceSessionLoopEvent | None = None,
    message: str = "loop_result",
    latency_ms: float = 1.0,
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
        latency_ms=latency_ms,
        created_at=utc_now(),
    )


def _snapshot(
    *,
    status: VoiceSessionLoopStatus = VoiceSessionLoopStatus.LISTENING,
    running: bool = True,
    cycles: int = 3,
    captured_frames: int = 3,
    consecutive_failures: int = 0,
) -> VoiceSessionLoopSnapshot:
    return VoiceSessionLoopSnapshot(
        status=status,
        running=running,
        assistant_speaking=False,
        cycles=cycles,
        captured_frames=captured_frames,
        speech_segments=1,
        partial_transcripts=1,
        final_transcripts=1,
        responses=1,
        tts_outputs=1,
        played_outputs=1,
        interruptions=0,
        recoveries=0,
        consecutive_failures=consecutive_failures,
        buffered_segment_frames=0,
        last_event=VoiceSessionLoopEvent.PLAYBACK_FINISHED,
        last_transcript_text="context_probe_transcript",
        last_response_text="generated_response_from_cognition",
        last_latency_ms=1.0,
        last_error=None,
        created_at=utc_now(),
    )


@dataclass
class FakeVoiceSessionLoop:
    started: bool = False
    stopped: bool = False
    fail_start: bool = False
    fail_run: bool = False
    degraded_failures: int = 0

    def start(self) -> VoiceSessionLoopResult:
        self.started = True
        if self.fail_start:
            return _loop_result(
                status=VoiceSessionLoopStatus.FAILED,
                operation=VoiceSessionLoopOperation.START,
                event=VoiceSessionLoopEvent.ERROR,
                message="start_failed",
            )
        return _loop_result(
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
        if self.fail_run:
            return _loop_result(
                status=VoiceSessionLoopStatus.FAILED,
                operation=VoiceSessionLoopOperation.RUN,
                event=VoiceSessionLoopEvent.ERROR,
                message="run_failed",
            )
        return _loop_result(
            status=VoiceSessionLoopStatus.LISTENING,
            operation=VoiceSessionLoopOperation.RUN,
            event=VoiceSessionLoopEvent.PLAYBACK_FINISHED,
            message="run_ok",
        )

    def stop(self) -> VoiceSessionLoopResult:
        self.stopped = True
        return _loop_result(
            status=VoiceSessionLoopStatus.STOPPED,
            operation=VoiceSessionLoopOperation.STOP,
            event=VoiceSessionLoopEvent.STOPPED,
            message="stop_ok",
        )

    def snapshot(self) -> VoiceSessionLoopSnapshot:
        return _snapshot(
            running=not self.stopped,
            consecutive_failures=self.degraded_failures,
        )


@dataclass
class FakeContextProbe:
    missing_environment: bool = False
    fixed_response: bool = False

    def probe(self) -> VoiceDailyDriverContextProbeResult:
        sources = {
            VoiceDailyDriverContextSource.TRANSCRIPT,
            VoiceDailyDriverContextSource.SESSION,
            VoiceDailyDriverContextSource.MEMORY,
            VoiceDailyDriverContextSource.ENVIRONMENT,
            VoiceDailyDriverContextSource.GOALS,
            VoiceDailyDriverContextSource.PERSONALITY,
            VoiceDailyDriverContextSource.TOOLS,
            VoiceDailyDriverContextSource.DEVELOPER,
            VoiceDailyDriverContextSource.HEALTH,
            VoiceDailyDriverContextSource.RESPONSE_BOUNDARY,
        }
        if self.missing_environment:
            sources.remove(VoiceDailyDriverContextSource.ENVIRONMENT)

        return VoiceDailyDriverContextProbeResult(
            available_sources=frozenset(sources),
            response_origin="cognition_response_boundary",
            uses_generated_response=not self.fixed_response,
            uses_fixed_response=self.fixed_response,
            context_signature="fake_context_probe",
            latency_ms=1.0,
        )


def test_voice_daily_driver_gate_config_validation() -> None:
    try:
        VoiceDailyDriverGateConfig(run_cycles=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected invalid run_cycles to fail")


def test_voice_daily_driver_gate_passes_with_healthy_loop() -> None:
    loop = FakeVoiceSessionLoop()
    gate = VoiceDailyDriverGate(
        session_loop=loop,
        context_probe=FakeContextProbe(),
        config=VoiceDailyDriverGateConfig(run_cycles=2),
    )

    report = gate.run()

    assert report.status == VoiceDailyDriverGateStatus.PASSED
    assert report.passed is True
    assert loop.started is True
    assert loop.stopped is True
    assert report.passed_count == len(report.checks)


def test_voice_daily_driver_gate_has_expected_checks() -> None:
    gate = VoiceDailyDriverGate(
        session_loop=FakeVoiceSessionLoop(),
        context_probe=FakeContextProbe(),
    )

    report = gate.run()
    kinds = {check.kind for check in report.checks}

    assert VoiceDailyDriverCheckKind.RESPONSE_ORIGIN_BOUNDARY in kinds
    assert VoiceDailyDriverCheckKind.SITUATION_CONTEXT_AVAILABLE in kinds
    assert VoiceDailyDriverCheckKind.SESSION_START in kinds
    assert VoiceDailyDriverCheckKind.SESSION_RUN in kinds
    assert VoiceDailyDriverCheckKind.SESSION_STOP in kinds
    assert VoiceDailyDriverCheckKind.LOOP_TELEMETRY in kinds
    assert VoiceDailyDriverCheckKind.VOICE_ORGAN_COVERAGE in kinds
    assert VoiceDailyDriverCheckKind.DAILY_DRIVER_READINESS in kinds


def test_voice_daily_driver_gate_fails_when_context_is_incomplete() -> None:
    gate = VoiceDailyDriverGate(
        session_loop=FakeVoiceSessionLoop(),
        context_probe=FakeContextProbe(missing_environment=True),
    )

    report = gate.run()

    assert report.status == VoiceDailyDriverGateStatus.FAILED
    assert any(
        check.kind == VoiceDailyDriverCheckKind.SITUATION_CONTEXT_AVAILABLE
        and check.status == VoiceDailyDriverGateStatus.FAILED
        for check in report.checks
    )


def test_voice_daily_driver_gate_fails_when_response_origin_is_fixed() -> None:
    gate = VoiceDailyDriverGate(
        session_loop=FakeVoiceSessionLoop(),
        context_probe=FakeContextProbe(fixed_response=True),
    )

    report = gate.run()

    assert report.status == VoiceDailyDriverGateStatus.FAILED
    assert any(
        check.kind == VoiceDailyDriverCheckKind.RESPONSE_ORIGIN_BOUNDARY
        and check.status == VoiceDailyDriverGateStatus.FAILED
        for check in report.checks
    )


def test_voice_daily_driver_gate_fails_when_start_fails() -> None:
    gate = VoiceDailyDriverGate(
        session_loop=FakeVoiceSessionLoop(fail_start=True),
        context_probe=FakeContextProbe(),
    )

    report = gate.run()

    assert report.status == VoiceDailyDriverGateStatus.FAILED
    assert any(
        check.kind == VoiceDailyDriverCheckKind.SESSION_START
        and check.status == VoiceDailyDriverGateStatus.FAILED
        for check in report.checks
    )


def test_voice_daily_driver_gate_fails_when_run_fails() -> None:
    gate = VoiceDailyDriverGate(
        session_loop=FakeVoiceSessionLoop(fail_run=True),
        context_probe=FakeContextProbe(),
    )

    report = gate.run()

    assert report.status == VoiceDailyDriverGateStatus.FAILED
    assert any(
        check.kind == VoiceDailyDriverCheckKind.SESSION_RUN
        and check.status == VoiceDailyDriverGateStatus.FAILED
        for check in report.checks
    )


def test_voice_daily_driver_gate_degrades_on_failure_boundary_warning() -> None:
    gate = VoiceDailyDriverGate(
        session_loop=FakeVoiceSessionLoop(degraded_failures=1),
        context_probe=FakeContextProbe(),
    )

    report = gate.run()

    assert report.status == VoiceDailyDriverGateStatus.DEGRADED
    assert report.degraded_count >= 1


def test_voice_daily_driver_gate_metadata_is_preserved() -> None:
    gate = VoiceDailyDriverGate(
        session_loop=FakeVoiceSessionLoop(),
        context_probe=FakeContextProbe(),
        config=VoiceDailyDriverGateConfig(
            metadata={"purpose": "daily_driver_validation"},
        ),
    )

    report = gate.run()

    assert report.metadata["purpose"] == "daily_driver_validation"


def test_voice_daily_driver_enum_values_are_stable() -> None:
    assert VoiceDailyDriverGateStatus.PASSED.value == "passed"
    assert VoiceDailyDriverProfile.BRONZE.value == "bronze"
    assert VoiceDailyDriverCheckKind.SESSION_RUN.value == "session_run"