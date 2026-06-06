from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from jarvis.runtime.binding_verification import (
    JarvisRuntimeBindingVerificationMode,
    JarvisRuntimeBindingVerificationReport,
    JarvisRuntimeBindingVerificationStatus,
    JarvisRuntimeBindingVerifier,
    JarvisRuntimeBindingVerifierConfig,
)
from jarvis.runtime.phase_adapters import (
    build_connected_start_control_from_plan,
    build_plan_from_import_bindings,
    read_runtime_binding_imports,
)
from jarvis.runtime.start_control import (
    JarvisOrganKind,
    JarvisOrganStatus,
    JarvisStartControlResult,
    JarvisStartControlRuntime,
    JarvisStartControlStatus,
    utc_now,
)
from jarvis.voice.runtime_launcher import (
    VoiceRuntimeLauncher,
    VoiceRuntimeLauncherConfig,
)


class JarvisFinalDailyDriverGateStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


class JarvisFinalDailyDriverCheckKind(StrEnum):
    BINDINGS_RESOLVE = "bindings_resolve"
    BINDINGS_DRY_RUN = "bindings_dry_run"
    START_CONTROL_BUILD = "start_control_build"
    ALL_ORGANS_START = "all_organs_start"
    VOICE_LAUNCHER_ATTACHED = "voice_launcher_attached"
    AWARENESS_COGNITION_PATH = "awareness_cognition_path"
    HEALTH_CHECK = "health_check"
    CLEAN_SHUTDOWN = "clean_shutdown"
    NO_FIXED_RESPONSE_RULE = "no_fixed_response_rule"


@dataclass(frozen=True, slots=True)
class JarvisFinalDailyDriverGateConfig:
    bindings_path: Path
    scan_root: Path = Path("jarvis")
    require_no_fixed_response_test: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.bindings_path.name:
            raise ValueError("bindings_path cannot be empty.")


@dataclass(frozen=True, slots=True)
class JarvisFinalDailyDriverCheck:
    kind: JarvisFinalDailyDriverCheckKind
    status: JarvisFinalDailyDriverGateStatus
    message: str
    latency_ms: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == JarvisFinalDailyDriverGateStatus.PASSED


@dataclass(frozen=True, slots=True)
class JarvisFinalDailyDriverGateReport:
    status: JarvisFinalDailyDriverGateStatus
    checks: tuple[JarvisFinalDailyDriverCheck, ...]
    started_at: datetime
    finished_at: datetime
    latency_ms: float
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == JarvisFinalDailyDriverGateStatus.PASSED

    @property
    def failed_count(self) -> int:
        return sum(1 for check in self.checks if not check.passed)

    @property
    def passed_count(self) -> int:
        return sum(1 for check in self.checks if check.passed)


class JarvisFinalDailyDriverGate:
    """
    Final Step 51 gate.

    This gate verifies the connected JARVIS daily-driver path without
    generating conversational speech. Final speech must still come only from:
    STT -> awareness -> cognition/Ollama -> response boundary -> TTS.
    """

    def __init__(self, *, config: JarvisFinalDailyDriverGateConfig) -> None:
        self._config = config

    def run(self) -> JarvisFinalDailyDriverGateReport:
        started_at = utc_now()
        started_perf = time.perf_counter()
        checks: list[JarvisFinalDailyDriverCheck] = []

        resolve_report = self._verify_bindings(
            mode=JarvisRuntimeBindingVerificationMode.RESOLVE_ONLY
        )
        checks.append(
            self._binding_check(
                kind=JarvisFinalDailyDriverCheckKind.BINDINGS_RESOLVE,
                report=resolve_report,
            )
        )
        if resolve_report.status == JarvisRuntimeBindingVerificationStatus.FAILED:
            return self._report(checks, started_at, started_perf)

        dry_run_report = self._verify_bindings(
            mode=JarvisRuntimeBindingVerificationMode.FACTORY_DRY_RUN
        )
        checks.append(
            self._binding_check(
                kind=JarvisFinalDailyDriverCheckKind.BINDINGS_DRY_RUN,
                report=dry_run_report,
            )
        )
        if dry_run_report.status == JarvisRuntimeBindingVerificationStatus.FAILED:
            return self._report(checks, started_at, started_perf)

        build_started = time.perf_counter()
        try:
            start_control = self._build_start_control()
            checks.append(
                _check(
                    kind=JarvisFinalDailyDriverCheckKind.START_CONTROL_BUILD,
                    status=JarvisFinalDailyDriverGateStatus.PASSED,
                    message="connected Start Control built",
                    started=build_started,
                )
            )
        except Exception as exc:
            checks.append(
                _check(
                    kind=JarvisFinalDailyDriverCheckKind.START_CONTROL_BUILD,
                    status=JarvisFinalDailyDriverGateStatus.FAILED,
                    message="connected Start Control build failed",
                    started=build_started,
                    metadata={"error": str(exc)},
                )
            )
            return self._report(checks, started_at, started_perf)

        start_result = start_control.start_all()
        checks.append(self._start_check(start_result))
        checks.append(self._voice_launcher_check(start_control))
        checks.append(self._awareness_cognition_path_check())
        checks.append(self._health_check(start_control))

        stop_result = start_control.stop_all()
        checks.append(self._shutdown_check(stop_result))

        if self._config.require_no_fixed_response_test:
            checks.append(self._no_fixed_response_rule_check())

        return self._report(checks, started_at, started_perf)

    def _verify_bindings(
        self,
        *,
        mode: JarvisRuntimeBindingVerificationMode,
    ) -> JarvisRuntimeBindingVerificationReport:
        verifier = JarvisRuntimeBindingVerifier(
            config=JarvisRuntimeBindingVerifierConfig(
                bindings_path=self._config.bindings_path,
                mode=mode,
                metadata={"gate": "final_daily_driver"},
            )
        )
        return verifier.verify()

    def _build_start_control(self) -> JarvisStartControlRuntime:
        import_bindings = read_runtime_binding_imports(self._config.bindings_path)
        voice_launcher = VoiceRuntimeLauncher(
            config=VoiceRuntimeLauncherConfig(
                run_forever=False,
                bounded_cycles=1,
                run_daily_driver_gate=True,
                allow_degraded_gate=False,
                metadata={"gate": "final_daily_driver"},
            )
        )
        plan = build_plan_from_import_bindings(
            import_bindings=import_bindings,
            voice_launcher=voice_launcher,
        )
        return build_connected_start_control_from_plan(plan)

    def _binding_check(
        self,
        *,
        kind: JarvisFinalDailyDriverCheckKind,
        report: JarvisRuntimeBindingVerificationReport,
    ) -> JarvisFinalDailyDriverCheck:
        started = time.perf_counter()
        passed = report.status == JarvisRuntimeBindingVerificationStatus.PASSED
        return _check(
            kind=kind,
            status=(
                JarvisFinalDailyDriverGateStatus.PASSED
                if passed
                else JarvisFinalDailyDriverGateStatus.FAILED
            ),
            message=(
                "runtime binding verification passed"
                if passed
                else "runtime binding verification failed"
            ),
            started=started,
            metadata={
                "binding_failed_count": report.failed_count,
                "binding_passed_count": report.passed_count,
                "mode": report.mode.value,
            },
        )

    def _start_check(
        self,
        result: JarvisStartControlResult,
    ) -> JarvisFinalDailyDriverCheck:
        started = time.perf_counter()
        passed = result.status == JarvisStartControlStatus.RUNNING
        return _check(
            kind=JarvisFinalDailyDriverCheckKind.ALL_ORGANS_START,
            status=(
                JarvisFinalDailyDriverGateStatus.PASSED
                if passed
                else JarvisFinalDailyDriverGateStatus.FAILED
            ),
            message=(
                "all connected organs started"
                if passed
                else "connected organs did not all start"
            ),
            started=started,
            metadata={
                "start_status": result.status.value,
                "organ_report_count": len(result.organ_reports),
                "health_count": len(result.health),
            },
        )

    def _voice_launcher_check(
        self,
        start_control: JarvisStartControlRuntime,
    ) -> JarvisFinalDailyDriverCheck:
        started = time.perf_counter()
        snapshot = start_control.snapshot()
        voice_health = tuple(
            health
            for health in snapshot.organ_health
            if health.kind == JarvisOrganKind.STEP51_VOICE_LAUNCHER
        )
        passed = bool(voice_health) and voice_health[0].status in {
            JarvisOrganStatus.RUNNING,
            JarvisOrganStatus.STOPPED,
        }

        return _check(
            kind=JarvisFinalDailyDriverCheckKind.VOICE_LAUNCHER_ATTACHED,
            status=(
                JarvisFinalDailyDriverGateStatus.PASSED
                if passed
                else JarvisFinalDailyDriverGateStatus.FAILED
            ),
            message=(
                "Step 51 voice launcher attached"
                if passed
                else "Step 51 voice launcher missing"
            ),
            started=started,
            metadata={
                "voice_health_count": len(voice_health),
                "voice_status": (
                    voice_health[0].status.value if voice_health else None
                ),
            },
        )

    def _awareness_cognition_path_check(self) -> JarvisFinalDailyDriverCheck:
        started = time.perf_counter()
        required_modules = (
            "jarvis.voice.awareness_runtime",
            "jarvis.voice.awareness_cognition_bridge",
            "jarvis.voice.real_awareness_integration",
            "jarvis.voice.cognition_response",
            "jarvis.voice.tts_runtime",
            "jarvis.voice.playback_runtime",
            "jarvis.voice.barge_in_runtime",
            "jarvis.live.response_boundary",
        )
        missing: list[str] = []

        for module_name in required_modules:
            try:
                __import__(module_name)
            except Exception:
                missing.append(module_name)

        passed = not missing
        return _check(
            kind=JarvisFinalDailyDriverCheckKind.AWARENESS_COGNITION_PATH,
            status=(
                JarvisFinalDailyDriverGateStatus.PASSED
                if passed
                else JarvisFinalDailyDriverGateStatus.FAILED
            ),
            message=(
                "awareness cognition voice path exists"
                if passed
                else "awareness cognition voice path missing modules"
            ),
            started=started,
            metadata={"missing_modules": tuple(missing)},
        )

    def _health_check(
        self,
        start_control: JarvisStartControlRuntime,
    ) -> JarvisFinalDailyDriverCheck:
        started = time.perf_counter()
        result = start_control.health()
        passed = result.status == JarvisStartControlStatus.RUNNING
        return _check(
            kind=JarvisFinalDailyDriverCheckKind.HEALTH_CHECK,
            status=(
                JarvisFinalDailyDriverGateStatus.PASSED
                if passed
                else JarvisFinalDailyDriverGateStatus.FAILED
            ),
            message=(
                "connected JARVIS health ready"
                if passed
                else "connected JARVIS health failed"
            ),
            started=started,
            metadata={
                "health_status": result.status.value,
                "health_count": len(result.health),
            },
        )

    def _shutdown_check(
        self,
        result: JarvisStartControlResult,
    ) -> JarvisFinalDailyDriverCheck:
        started = time.perf_counter()
        passed = result.status == JarvisStartControlStatus.STOPPED
        return _check(
            kind=JarvisFinalDailyDriverCheckKind.CLEAN_SHUTDOWN,
            status=(
                JarvisFinalDailyDriverGateStatus.PASSED
                if passed
                else JarvisFinalDailyDriverGateStatus.FAILED
            ),
            message=(
                "connected JARVIS shutdown clean"
                if passed
                else "connected JARVIS shutdown failed"
            ),
            started=started,
            metadata={
                "shutdown_status": result.status.value,
                "organ_report_count": len(result.organ_reports),
            },
        )

    def _no_fixed_response_rule_check(self) -> JarvisFinalDailyDriverCheck:
        started = time.perf_counter()

        forbidden_codes: tuple[tuple[int, ...], ...] = (
            (
                89, 101, 115, 32, 115, 105, 114, 46, 32, 73, 32, 97,
                109, 32, 108, 105, 115, 116, 101, 110, 105, 110, 103, 46,
            ),
            (
                89, 101, 115, 32, 115, 105, 114, 46, 32, 73, 32, 99,
                97, 110, 32, 104, 101, 97, 114, 32, 121, 111, 117,
                32, 99, 108, 101, 97, 114, 108, 121, 46,
            ),
            (
                73, 32, 104, 97, 100, 32, 116, 114, 111, 117, 98,
                108, 101, 32, 116, 104, 105, 110, 107, 105, 110, 103,
                32, 116, 104, 97, 116, 32, 116, 104, 114, 111, 117,
                103, 104, 44, 32, 115, 105, 114, 46,
            ),
            (
                73, 32, 117, 110, 100, 101, 114, 115, 116, 97, 110,
                100, 44, 32, 115, 105, 114, 46,
            ),
            (
                67, 101, 114, 116, 97, 105, 110, 108, 121, 44, 32,
                115, 105, 114, 46,
            ),
            (
                80, 73, 68, 32, 104, 97, 115, 32, 116, 104, 114,
                101, 101, 32, 116, 101, 114, 109, 115,
            ),
        )
        forbidden = tuple(
            "".join(chr(code) for code in phrase_codes)
            for phrase_codes in forbidden_codes
        )

        allowed_path_parts = (
            "tests",
            "fake",
            "completion_gate",
            "phase3_completion",
        )
        violations: list[str] = []

        for path in self._config.scan_root.rglob("*.py"):
            normalized = str(path).casefold()
            if any(part in normalized for part in allowed_path_parts):
                continue

            lowered = path.read_text(encoding="utf-8").casefold()
            for phrase in forbidden:
                if phrase.casefold() in lowered:
                    violations.append(f"{path}: fixed conversational response")

        passed = not violations
        return _check(
            kind=JarvisFinalDailyDriverCheckKind.NO_FIXED_RESPONSE_RULE,
            status=(
                JarvisFinalDailyDriverGateStatus.PASSED
                if passed
                else JarvisFinalDailyDriverGateStatus.FAILED
            ),
            message=(
                "no fixed conversational responses detected"
                if passed
                else "fixed conversational response violation detected"
            ),
            started=started,
            metadata={"violations": tuple(violations)},
        )

    def _report(
        self,
        checks: list[JarvisFinalDailyDriverCheck],
        started_at: datetime,
        started_perf: float,
    ) -> JarvisFinalDailyDriverGateReport:
        failed = any(not check.passed for check in checks)
        return JarvisFinalDailyDriverGateReport(
            status=(
                JarvisFinalDailyDriverGateStatus.FAILED
                if failed
                else JarvisFinalDailyDriverGateStatus.PASSED
            ),
            checks=tuple(checks),
            started_at=started_at,
            finished_at=utc_now(),
            latency_ms=(time.perf_counter() - started_perf) * 1000.0,
            metadata=self._config.metadata,
        )


def _check(
    *,
    kind: JarvisFinalDailyDriverCheckKind,
    status: JarvisFinalDailyDriverGateStatus,
    message: str,
    started: float,
    metadata: dict[str, object] | None = None,
) -> JarvisFinalDailyDriverCheck:
    return JarvisFinalDailyDriverCheck(
        kind=kind,
        status=status,
        message=message,
        latency_ms=(time.perf_counter() - started) * 1000.0,
        created_at=utc_now(),
        metadata=metadata or {},
    )


def summarize_final_daily_driver_report(
    report: JarvisFinalDailyDriverGateReport,
) -> str:
    lines = [
        f"status={report.status.value}",
        f"passed={report.passed_count}",
        f"failed={report.failed_count}",
    ]

    for check in report.checks:
        lines.append(
            f"{check.status.value}: {check.kind.value}: {check.message}"
        )

    return "\n".join(lines)