from __future__ import annotations

from jarvis.runtime.validation import (
    RuntimeIntegrationValidator,
    ValidationCheck,
    ValidationReport,
    utc_now,
)


def test_runtime_integration_validator_passes() -> None:
    report = RuntimeIntegrationValidator().run()

    assert report.passed is True
    assert report.failed_count == 0
    assert report.passed_count >= 7

    check_names = {check.name for check in report.checks}

    assert "kernel_started" in check_names
    assert "state_engine_updates" in check_names
    assert "security_permission_flow" in check_names
    assert "cancellation_manager" in check_names
    assert "scheduler_executes_task" in check_names
    assert "runtime_health_check" in check_names
    assert "runtime_event_flow" in check_names
    assert "kernel_stopped" in check_names


def test_validation_report_failed_checks_property() -> None:
    report = ValidationReport(
        passed=False,
        checks=(
            ValidationCheck(name="ok", passed=True),
            ValidationCheck(name="bad", passed=False, error="boom"),
        ),
        started_at=utc_now(),
        finished_at=utc_now(),
        duration_ms=1.0,
    )

    assert report.passed_count == 1
    assert report.failed_count == 1
    assert len(report.failed_checks) == 1
    assert report.failed_checks[0].name == "bad"


def test_validation_check_stores_details() -> None:
    check = ValidationCheck(
        name="demo",
        passed=True,
        details={"system": "JARVIS"},
    )

    assert check.name == "demo"
    assert check.passed is True
    assert check.details["system"] == "JARVIS"