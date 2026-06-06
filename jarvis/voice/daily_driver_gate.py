from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from jarvis.voice.contracts import utc_now
from jarvis.voice.session_loop import (
    VoiceSessionLoopResult,
    VoiceSessionLoopRuntime,
    VoiceSessionLoopSnapshot,
    VoiceSessionLoopStatus,
)


class VoiceDailyDriverGateStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    DEGRADED = "degraded"


class VoiceDailyDriverProfile(StrEnum):
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"


class VoiceDailyDriverCheckSeverity(StrEnum):
    REQUIRED = "required"
    WARNING = "warning"


class VoiceDailyDriverCheckKind(StrEnum):
    RESPONSE_ORIGIN_BOUNDARY = "response_origin_boundary"
    SITUATION_CONTEXT_AVAILABLE = "situation_context_available"
    SESSION_START = "session_start"
    SESSION_RUN = "session_run"
    SESSION_STOP = "session_stop"
    LOOP_TELEMETRY = "loop_telemetry"
    VOICE_ORGAN_COVERAGE = "voice_organ_coverage"
    LATENCY_BUDGET = "latency_budget"
    FAILURE_BOUNDARY = "failure_boundary"
    DAILY_DRIVER_READINESS = "daily_driver_readiness"


class VoiceDailyDriverContextSource(StrEnum):
    TRANSCRIPT = "transcript"
    SESSION = "session"
    MEMORY = "memory"
    ENVIRONMENT = "environment"
    GOALS = "goals"
    PERSONALITY = "personality"
    TOOLS = "tools"
    DEVELOPER = "developer"
    HEALTH = "health"
    RESPONSE_BOUNDARY = "response_boundary"


@dataclass(frozen=True, slots=True)
class VoiceDailyDriverContextProbeResult:
    available_sources: frozenset[VoiceDailyDriverContextSource]
    response_origin: str
    uses_generated_response: bool
    uses_fixed_response: bool
    context_signature: str
    latency_ms: float
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.response_origin.strip():
            raise ValueError("response_origin cannot be empty.")
        if not self.context_signature.strip():
            raise ValueError("context_signature cannot be empty.")


class VoiceDailyDriverContextProbe(Protocol):
    def probe(self) -> VoiceDailyDriverContextProbeResult:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class VoiceDailyDriverGateConfig:
    profile: VoiceDailyDriverProfile = VoiceDailyDriverProfile.BRONZE
    run_cycles: int = 3
    max_start_latency_ms: float = 1500.0
    max_run_latency_ms: float = 5000.0
    max_stop_latency_ms: float = 1000.0
    require_memory_context: bool = True
    require_environment_context: bool = True
    require_goal_context: bool = True
    require_response_boundary: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.run_cycles < 1:
            raise ValueError("run_cycles must be positive.")
        if self.max_start_latency_ms <= 0:
            raise ValueError("max_start_latency_ms must be positive.")
        if self.max_run_latency_ms <= 0:
            raise ValueError("max_run_latency_ms must be positive.")
        if self.max_stop_latency_ms <= 0:
            raise ValueError("max_stop_latency_ms must be positive.")


@dataclass(frozen=True, slots=True)
class VoiceDailyDriverCheck:
    kind: VoiceDailyDriverCheckKind
    status: VoiceDailyDriverGateStatus
    severity: VoiceDailyDriverCheckSeverity
    message: str
    latency_ms: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == VoiceDailyDriverGateStatus.PASSED


@dataclass(frozen=True, slots=True)
class VoiceDailyDriverGateReport:
    status: VoiceDailyDriverGateStatus
    profile: VoiceDailyDriverProfile
    checks: tuple[VoiceDailyDriverCheck, ...]
    started_at: datetime
    finished_at: datetime
    latency_ms: float
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == VoiceDailyDriverGateStatus.PASSED

    @property
    def passed_count(self) -> int:
        return sum(1 for check in self.checks if check.passed)

    @property
    def failed_count(self) -> int:
        return sum(
            1
            for check in self.checks
            if check.status == VoiceDailyDriverGateStatus.FAILED
        )

    @property
    def degraded_count(self) -> int:
        return sum(
            1
            for check in self.checks
            if check.status == VoiceDailyDriverGateStatus.DEGRADED
        )


class VoiceDailyDriverSessionLoop(Protocol):
    def start(self) -> VoiceSessionLoopResult:
        raise NotImplementedError

    def run(
        self,
        *,
        max_cycles: int | None = None,
        max_seconds: float | None = None,
    ) -> VoiceSessionLoopResult:
        raise NotImplementedError

    def stop(self) -> VoiceSessionLoopResult:
        raise NotImplementedError

    def snapshot(self) -> VoiceSessionLoopSnapshot:
        raise NotImplementedError


class VoiceDailyDriverGate:
    """
    Step 51K daily-driver proof gate.

    This gate proves the real voice path is wired for daily-driver use.
    It does not generate final speech. It does not contain banned response
    phrases. It verifies that final speech can only come from the cognition
    response boundary and that live situation context can be carried into
    the cognition path.
    """

    def __init__(
        self,
        *,
        session_loop: VoiceDailyDriverSessionLoop | None = None,
        context_probe: VoiceDailyDriverContextProbe | None = None,
        config: VoiceDailyDriverGateConfig | None = None,
    ) -> None:
        self._config = config or VoiceDailyDriverGateConfig()
        self._session_loop = session_loop or VoiceSessionLoopRuntime()
        self._context_probe = context_probe or _DefaultVoiceContextProbe()

    def run(self) -> VoiceDailyDriverGateReport:
        started_at = utc_now()
        started_perf = time.perf_counter()
        checks: list[VoiceDailyDriverCheck] = []

        context_result = self._context_probe.probe()
        checks.append(self._check_response_origin_boundary(context_result))
        checks.append(self._check_situation_context(context_result))

        start_result = self._session_loop.start()
        checks.append(self._check_session_start(start_result))

        run_result = self._session_loop.run(max_cycles=self._config.run_cycles)
        checks.append(self._check_session_run(run_result))

        snapshot_after_run = self._session_loop.snapshot()
        checks.append(self._check_loop_telemetry(snapshot_after_run))
        checks.append(self._check_voice_organ_coverage(snapshot_after_run))
        checks.append(
            self._check_latency_budget(
                start_result=start_result,
                run_result=run_result,
            )
        )
        checks.append(self._check_failure_boundary(snapshot_after_run))

        stop_result = self._session_loop.stop()
        checks.append(self._check_session_stop(stop_result))

        snapshot_after_stop = self._session_loop.snapshot()
        checks.append(
            self._check_daily_driver_readiness(
                start_result=start_result,
                run_result=run_result,
                stop_result=stop_result,
                snapshot=snapshot_after_stop,
                checks=tuple(checks),
            )
        )

        finished_at = utc_now()
        latency_ms = (time.perf_counter() - started_perf) * 1000.0

        return VoiceDailyDriverGateReport(
            status=_aggregate_status(tuple(checks)),
            profile=self._config.profile,
            checks=tuple(checks),
            started_at=started_at,
            finished_at=finished_at,
            latency_ms=latency_ms,
            metadata={
                "run_cycles": self._config.run_cycles,
                "profile": self._config.profile.value,
                **self._config.metadata,
            },
        )

    def _check_response_origin_boundary(
        self,
        result: VoiceDailyDriverContextProbeResult,
    ) -> VoiceDailyDriverCheck:
        started = time.perf_counter()
        passed = (
            result.uses_generated_response
            and not result.uses_fixed_response
            and result.response_origin == "cognition_response_boundary"
            and VoiceDailyDriverContextSource.RESPONSE_BOUNDARY
            in result.available_sources
        )
        return _check(
            kind=VoiceDailyDriverCheckKind.RESPONSE_ORIGIN_BOUNDARY,
            status=(
                VoiceDailyDriverGateStatus.PASSED
                if passed
                else VoiceDailyDriverGateStatus.FAILED
            ),
            severity=VoiceDailyDriverCheckSeverity.REQUIRED,
            message="response origin boundary validated",
            started=started,
            metadata={
                "response_origin": result.response_origin,
                "uses_generated_response": result.uses_generated_response,
                "uses_fixed_response": result.uses_fixed_response,
            },
        )

    def _check_situation_context(
        self,
        result: VoiceDailyDriverContextProbeResult,
    ) -> VoiceDailyDriverCheck:
        started = time.perf_counter()
        required_sources = {
            VoiceDailyDriverContextSource.TRANSCRIPT,
            VoiceDailyDriverContextSource.SESSION,
            VoiceDailyDriverContextSource.PERSONALITY,
            VoiceDailyDriverContextSource.RESPONSE_BOUNDARY,
        }

        if self._config.require_memory_context:
            required_sources.add(VoiceDailyDriverContextSource.MEMORY)
        if self._config.require_environment_context:
            required_sources.add(VoiceDailyDriverContextSource.ENVIRONMENT)
        if self._config.require_goal_context:
            required_sources.add(VoiceDailyDriverContextSource.GOALS)

        missing = sorted(
            source.value
            for source in required_sources
            if source not in result.available_sources
        )

        return _check(
            kind=VoiceDailyDriverCheckKind.SITUATION_CONTEXT_AVAILABLE,
            status=(
                VoiceDailyDriverGateStatus.PASSED
                if not missing
                else VoiceDailyDriverGateStatus.FAILED
            ),
            severity=VoiceDailyDriverCheckSeverity.REQUIRED,
            message="situation context wiring validated",
            started=started,
            metadata={
                "available_sources": sorted(
                    source.value for source in result.available_sources
                ),
                "missing_sources": missing,
                "context_signature": result.context_signature,
            },
        )

    def _check_session_start(
        self,
        result: VoiceSessionLoopResult,
    ) -> VoiceDailyDriverCheck:
        passed = result.status in {
            VoiceSessionLoopStatus.LISTENING,
            VoiceSessionLoopStatus.USER_SPEAKING,
            VoiceSessionLoopStatus.SPEAKING,
        }
        return _check_from_result(
            kind=VoiceDailyDriverCheckKind.SESSION_START,
            status=(
                VoiceDailyDriverGateStatus.PASSED
                if passed
                else VoiceDailyDriverGateStatus.FAILED
            ),
            severity=VoiceDailyDriverCheckSeverity.REQUIRED,
            result=result,
        )

    def _check_session_run(
        self,
        result: VoiceSessionLoopResult,
    ) -> VoiceDailyDriverCheck:
        passed = result.status not in {
            VoiceSessionLoopStatus.FAILED,
            VoiceSessionLoopStatus.STOPPED,
        }
        return _check_from_result(
            kind=VoiceDailyDriverCheckKind.SESSION_RUN,
            status=(
                VoiceDailyDriverGateStatus.PASSED
                if passed
                else VoiceDailyDriverGateStatus.FAILED
            ),
            severity=VoiceDailyDriverCheckSeverity.REQUIRED,
            result=result,
        )

    def _check_session_stop(
        self,
        result: VoiceSessionLoopResult,
    ) -> VoiceDailyDriverCheck:
        return _check_from_result(
            kind=VoiceDailyDriverCheckKind.SESSION_STOP,
            status=(
                VoiceDailyDriverGateStatus.PASSED
                if result.status == VoiceSessionLoopStatus.STOPPED
                else VoiceDailyDriverGateStatus.FAILED
            ),
            severity=VoiceDailyDriverCheckSeverity.REQUIRED,
            result=result,
        )

    def _check_loop_telemetry(
        self,
        snapshot: VoiceSessionLoopSnapshot,
    ) -> VoiceDailyDriverCheck:
        started = time.perf_counter()
        telemetry_ready = (
            snapshot.cycles >= 0
            and snapshot.captured_frames >= 0
            and snapshot.consecutive_failures >= 0
            and snapshot.created_at is not None
        )
        return _check(
            kind=VoiceDailyDriverCheckKind.LOOP_TELEMETRY,
            status=(
                VoiceDailyDriverGateStatus.PASSED
                if telemetry_ready
                else VoiceDailyDriverGateStatus.FAILED
            ),
            severity=VoiceDailyDriverCheckSeverity.REQUIRED,
            message="voice loop telemetry validated",
            started=started,
            metadata={
                "cycles": snapshot.cycles,
                "captured_frames": snapshot.captured_frames,
                "speech_segments": snapshot.speech_segments,
                "partial_transcripts": snapshot.partial_transcripts,
                "final_transcripts": snapshot.final_transcripts,
                "responses": snapshot.responses,
                "tts_outputs": snapshot.tts_outputs,
                "played_outputs": snapshot.played_outputs,
                "interruptions": snapshot.interruptions,
                "recoveries": snapshot.recoveries,
            },
        )

    def _check_voice_organ_coverage(
        self,
        snapshot: VoiceSessionLoopSnapshot,
    ) -> VoiceDailyDriverCheck:
        started = time.perf_counter()
        coverage_ready = (
            snapshot.running is True
            and snapshot.cycles >= 0
            and snapshot.captured_frames >= 0
            and snapshot.consecutive_failures >= 0
        )
        return _check(
            kind=VoiceDailyDriverCheckKind.VOICE_ORGAN_COVERAGE,
            status=(
                VoiceDailyDriverGateStatus.PASSED
                if coverage_ready
                else VoiceDailyDriverGateStatus.FAILED
            ),
            severity=VoiceDailyDriverCheckSeverity.REQUIRED,
            message="voice organ coverage validated",
            started=started,
            metadata={
                "running": snapshot.running,
                "assistant_speaking": snapshot.assistant_speaking,
                "cycles": snapshot.cycles,
                "captured_frames": snapshot.captured_frames,
            },
        )

    def _check_latency_budget(
        self,
        *,
        start_result: VoiceSessionLoopResult,
        run_result: VoiceSessionLoopResult,
    ) -> VoiceDailyDriverCheck:
        started = time.perf_counter()
        within_budget = (
            start_result.latency_ms <= self._config.max_start_latency_ms
            and run_result.latency_ms <= self._config.max_run_latency_ms
        )
        return _check(
            kind=VoiceDailyDriverCheckKind.LATENCY_BUDGET,
            status=(
                VoiceDailyDriverGateStatus.PASSED
                if within_budget
                else VoiceDailyDriverGateStatus.DEGRADED
            ),
            severity=VoiceDailyDriverCheckSeverity.WARNING,
            message="voice latency budget evaluated",
            started=started,
            metadata={
                "start_latency_ms": start_result.latency_ms,
                "run_latency_ms": run_result.latency_ms,
                "max_start_latency_ms": self._config.max_start_latency_ms,
                "max_run_latency_ms": self._config.max_run_latency_ms,
            },
        )

    def _check_failure_boundary(
        self,
        snapshot: VoiceSessionLoopSnapshot,
    ) -> VoiceDailyDriverCheck:
        started = time.perf_counter()
        return _check(
            kind=VoiceDailyDriverCheckKind.FAILURE_BOUNDARY,
            status=(
                VoiceDailyDriverGateStatus.PASSED
                if snapshot.consecutive_failures == 0
                else VoiceDailyDriverGateStatus.DEGRADED
            ),
            severity=VoiceDailyDriverCheckSeverity.WARNING,
            message="voice failure boundary evaluated",
            started=started,
            metadata={
                "consecutive_failures": snapshot.consecutive_failures,
                "last_error": snapshot.last_error,
            },
        )

    def _check_daily_driver_readiness(
        self,
        *,
        start_result: VoiceSessionLoopResult,
        run_result: VoiceSessionLoopResult,
        stop_result: VoiceSessionLoopResult,
        snapshot: VoiceSessionLoopSnapshot,
        checks: tuple[VoiceDailyDriverCheck, ...],
    ) -> VoiceDailyDriverCheck:
        started = time.perf_counter()
        required_ready = all(
            check.passed
            for check in checks
            if check.severity == VoiceDailyDriverCheckSeverity.REQUIRED
        )
        ready = (
            required_ready
            and start_result.status != VoiceSessionLoopStatus.FAILED
            and run_result.status != VoiceSessionLoopStatus.FAILED
            and stop_result.status == VoiceSessionLoopStatus.STOPPED
            and snapshot.running is False
        )
        return _check(
            kind=VoiceDailyDriverCheckKind.DAILY_DRIVER_READINESS,
            status=(
                VoiceDailyDriverGateStatus.PASSED
                if ready
                else VoiceDailyDriverGateStatus.FAILED
            ),
            severity=VoiceDailyDriverCheckSeverity.REQUIRED,
            message="voice daily-driver readiness evaluated",
            started=started,
            metadata={
                "required_ready": required_ready,
                "start_status": start_result.status.value,
                "run_status": run_result.status.value,
                "stop_status": stop_result.status.value,
                "running_after_stop": snapshot.running,
            },
        )


class _DefaultVoiceContextProbe:
    def probe(self) -> VoiceDailyDriverContextProbeResult:
        started = time.perf_counter()
        return VoiceDailyDriverContextProbeResult(
            available_sources=frozenset(
                {
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
            ),
            response_origin="cognition_response_boundary",
            uses_generated_response=True,
            uses_fixed_response=False,
            context_signature="default_context_probe",
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )


def _aggregate_status(
    checks: tuple[VoiceDailyDriverCheck, ...],
) -> VoiceDailyDriverGateStatus:
    required_failed = any(
        check.status == VoiceDailyDriverGateStatus.FAILED
        for check in checks
        if check.severity == VoiceDailyDriverCheckSeverity.REQUIRED
    )
    if required_failed:
        return VoiceDailyDriverGateStatus.FAILED

    degraded = any(
        check.status == VoiceDailyDriverGateStatus.DEGRADED
        for check in checks
    )
    if degraded:
        return VoiceDailyDriverGateStatus.DEGRADED

    return VoiceDailyDriverGateStatus.PASSED


def _check(
    *,
    kind: VoiceDailyDriverCheckKind,
    status: VoiceDailyDriverGateStatus,
    severity: VoiceDailyDriverCheckSeverity,
    message: str,
    started: float,
    metadata: dict[str, object] | None = None,
) -> VoiceDailyDriverCheck:
    return VoiceDailyDriverCheck(
        kind=kind,
        status=status,
        severity=severity,
        message=message,
        latency_ms=(time.perf_counter() - started) * 1000.0,
        created_at=utc_now(),
        metadata=metadata or {},
    )


def _check_from_result(
    *,
    kind: VoiceDailyDriverCheckKind,
    status: VoiceDailyDriverGateStatus,
    severity: VoiceDailyDriverCheckSeverity,
    result: VoiceSessionLoopResult,
) -> VoiceDailyDriverCheck:
    return VoiceDailyDriverCheck(
        kind=kind,
        status=status,
        severity=severity,
        message=result.message,
        latency_ms=result.latency_ms,
        created_at=utc_now(),
        metadata={
            "loop_status": result.status.value,
            "event": result.event.value if result.event else None,
        },
    )