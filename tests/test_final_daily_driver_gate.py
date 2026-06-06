from __future__ import annotations

import sys
from pathlib import Path

from jarvis.runtime import (
    JarvisFinalDailyDriverCheckKind,
    JarvisFinalDailyDriverGate,
    JarvisFinalDailyDriverGateConfig,
    JarvisFinalDailyDriverGateStatus,
    JarvisOrganKind,
    summarize_final_daily_driver_report,
)


def _write_runtime_factory_package(tmp_path: Path) -> str:
    package = tmp_path / "final_gate_package"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "factories.py").write_text(
        "\n".join(
            (
                "class Health:",
                "    status = 'running'",
                "",
                "class Runtime:",
                "    def __init__(self):",
                "        self.started = False",
                "    def start(self):",
                "        self.started = True",
                "    def stop(self):",
                "        self.started = False",
                "    def recover(self):",
                "        self.started = True",
                "    def health(self):",
                "        return Health()",
                "",
                "def create_runtime():",
                "    return Runtime()",
            )
        ),
        encoding="utf-8",
    )
    sys.path.insert(0, str(tmp_path))
    return "final_gate_package.factories:create_runtime"


def _bindings_file(tmp_path: Path) -> Path:
    import_path = _write_runtime_factory_package(tmp_path)
    phases = (
        "phase1_kernel",
        "phase1_events",
        "phase1_observability",
        "phase2_presence",
        "phase2_voice",
        "phase3_cognition",
        "phase4_memory",
        "phase5_tools",
        "phase6_orchestration",
        "phase7_streaming_latency",
        "phase8_environment",
        "phase9_cognitive_session",
    )
    path = tmp_path / "runtime_bindings.env"
    path.write_text(
        "\n".join(f"{phase}={import_path}" for phase in phases),
        encoding="utf-8",
    )
    return path


def test_final_daily_driver_gate_passes(tmp_path: Path) -> None:
    gate = JarvisFinalDailyDriverGate(
        config=JarvisFinalDailyDriverGateConfig(
            bindings_path=_bindings_file(tmp_path),
            scan_root=Path("jarvis"),
        )
    )

    report = gate.run()

    assert report.status == JarvisFinalDailyDriverGateStatus.PASSED
    assert report.failed_count == 0


def test_final_daily_driver_gate_has_required_checks(tmp_path: Path) -> None:
    gate = JarvisFinalDailyDriverGate(
        config=JarvisFinalDailyDriverGateConfig(
            bindings_path=_bindings_file(tmp_path),
            scan_root=Path("jarvis"),
        )
    )

    report = gate.run()
    kinds = {check.kind for check in report.checks}

    assert JarvisFinalDailyDriverCheckKind.BINDINGS_RESOLVE in kinds
    assert JarvisFinalDailyDriverCheckKind.BINDINGS_DRY_RUN in kinds
    assert JarvisFinalDailyDriverCheckKind.START_CONTROL_BUILD in kinds
    assert JarvisFinalDailyDriverCheckKind.ALL_ORGANS_START in kinds
    assert JarvisFinalDailyDriverCheckKind.VOICE_LAUNCHER_ATTACHED in kinds
    assert JarvisFinalDailyDriverCheckKind.AWARENESS_COGNITION_PATH in kinds
    assert JarvisFinalDailyDriverCheckKind.HEALTH_CHECK in kinds
    assert JarvisFinalDailyDriverCheckKind.CLEAN_SHUTDOWN in kinds
    assert JarvisFinalDailyDriverCheckKind.NO_FIXED_RESPONSE_RULE in kinds


def test_final_daily_driver_gate_fails_missing_bindings(tmp_path: Path) -> None:
    gate = JarvisFinalDailyDriverGate(
        config=JarvisFinalDailyDriverGateConfig(
            bindings_path=tmp_path / "missing.env",
            scan_root=Path("jarvis"),
        )
    )

    report = gate.run()

    assert report.status == JarvisFinalDailyDriverGateStatus.FAILED
    assert report.checks[0].kind == JarvisFinalDailyDriverCheckKind.BINDINGS_RESOLVE


def test_final_daily_driver_summary(tmp_path: Path) -> None:
    gate = JarvisFinalDailyDriverGate(
        config=JarvisFinalDailyDriverGateConfig(
            bindings_path=_bindings_file(tmp_path),
            scan_root=Path("jarvis"),
        )
    )

    report = gate.run()
    summary = summarize_final_daily_driver_report(report)

    assert "status=passed" in summary
    assert "clean_shutdown" in summary


def test_final_daily_driver_enum_values_are_stable() -> None:
    assert JarvisFinalDailyDriverGateStatus.PASSED.value == "passed"
    assert JarvisOrganKind.STEP51_VOICE_LAUNCHER.value == "step51_voice_launcher"