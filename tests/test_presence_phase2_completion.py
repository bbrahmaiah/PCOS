from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from jarvis.presence.phase2_completion import (
    Phase2CommandResult,
    Phase2CompletionConfig,
    Phase2CompletionGate,
    Phase2CompletionReport,
    default_phase2_completion_config,
    format_phase2_completion_report,
)


def make_result(
    *,
    name: str = "example",
    passed: bool = True,
    return_code: int = 0,
) -> Phase2CommandResult:
    return Phase2CommandResult(
        name=name,
        command=("echo", "ok"),
        passed=passed,
        return_code=return_code,
        duration_ms=10.0,
        stdout="ok",
        stderr="",
    )


def make_report(
    *,
    passed: bool = True,
    results: tuple[Phase2CommandResult, ...] | None = None,
) -> Phase2CompletionReport:
    now = datetime.now(UTC)
    clean_results = results if results is not None else (make_result(),)

    return Phase2CompletionReport(
        passed=passed,
        started_at=now,
        finished_at=now,
        duration_ms=10.0,
        results=clean_results,
        known_good_voice_command=(
            "python .\\scripts\\run_jarvis_voice_smoke.py --preset fast"
        ),
    )


def test_phase2_command_result_command_text() -> None:
    result = make_result()

    assert result.command_text == "echo ok"


def test_phase2_completion_report_counts() -> None:
    report = make_report(
        passed=False,
        results=(
            make_result(name="pass", passed=True, return_code=0),
            make_result(name="fail", passed=False, return_code=1),
        ),
    )

    assert report.total_count == 2
    assert report.passed_count == 1
    assert report.failed_count == 1
    assert len(report.failed_results) == 1
    assert report.failed_results[0].name == "fail"


def test_phase2_completion_config_rejects_invalid_timeout() -> None:
    with pytest.raises(ValueError):
        Phase2CompletionConfig(
            project_root=Path.cwd(),
            timeout_seconds=0,
        ).validate()


def test_phase2_completion_config_rejects_missing_project_root() -> None:
    with pytest.raises(ValueError):
        Phase2CompletionConfig(
            project_root=Path("this-path-does-not-exist"),
        ).validate()


def test_phase2_completion_gate_builds_expected_base_commands() -> None:
    config = Phase2CompletionConfig(
        project_root=Path.cwd(),
        include_pytest=False,
        include_validation=False,
        include_latency_profile=False,
    )
    gate = Phase2CompletionGate(config)

    names = tuple(name for name, _command in gate._commands())

    assert names == ("ruff_fix_check", "ruff_check", "mypy")


def test_phase2_completion_gate_builds_full_commands() -> None:
    config = Phase2CompletionConfig(project_root=Path.cwd())
    gate = Phase2CompletionGate(config)

    names = tuple(name for name, _command in gate._commands())

    assert names == (
        "ruff_fix_check",
        "ruff_check",
        "mypy",
        "pytest",
        "presence_validation",
        "presence_latency_profile",
    )


def test_format_phase2_completion_report_for_success() -> None:
    report = make_report(passed=True)

    output = format_phase2_completion_report(report)

    assert "JARVIS Phase 2 Completion Gate" in output
    assert "Passed: True" in output
    assert "PHASE 2 STATUS: COMPLETE" in output
    assert "run_jarvis_voice_smoke.py --preset fast" in output


def test_format_phase2_completion_report_for_failure() -> None:
    report = make_report(
        passed=False,
        results=(
            make_result(name="fail", passed=False, return_code=1),
        ),
    )

    output = format_phase2_completion_report(report)

    assert "Passed: False" in output
    assert "Failures:" in output
    assert "## fail" in output


def test_default_phase2_completion_config_points_to_project_root() -> None:
    config = default_phase2_completion_config()

    assert config.project_root.exists()
    assert config.project_root.name == "JARVIS_OS"