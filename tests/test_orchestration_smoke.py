from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.orchestration import (
    OrchestrationSmokeConfig,
    OrchestrationSmokeReport,
    OrchestrationSmokeRuntime,
    SmokeCheckKind,
    SmokeCheckResult,
    SmokeCheckStatus,
    SmokeReason,
)


def test_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        OrchestrationSmokeConfig(name=" ").validate()


def test_config_requires_at_least_three_parallel_tasks() -> None:
    config = OrchestrationSmokeConfig(synthetic_parallel_task_count=2)

    with pytest.raises(ValueError):
        config.validate()


def test_check_result_requires_message() -> None:
    with pytest.raises(ValidationError):
        SmokeCheckResult(
            kind=SmokeCheckKind.WORKERS_HEALTHY,
            status=SmokeCheckStatus.PASSED,
            reason=SmokeReason.CHECK_PASSED,
            message=" ",
        )


def test_check_result_passed_property() -> None:
    result = SmokeCheckResult(
        kind=SmokeCheckKind.WORKERS_HEALTHY,
        status=SmokeCheckStatus.PASSED,
        reason=SmokeReason.CHECK_PASSED,
        message="ok",
    )

    assert result.passed is True


def test_smoke_runtime_passes_all_checks() -> None:
    runtime = OrchestrationSmokeRuntime()

    report = runtime.run()

    assert report.success is True
    assert report.failed_count == 0
    assert report.passed_count == len(report.checks)


def test_smoke_runtime_runs_required_checks() -> None:
    runtime = OrchestrationSmokeRuntime()

    report = runtime.run()
    kinds = {check.kind for check in report.checks}

    assert SmokeCheckKind.WORKERS_HEALTHY in kinds
    assert SmokeCheckKind.SCHEDULER_RUNS in kinds
    assert SmokeCheckKind.PARALLEL_TASKS_COMPLETE in kinds
    assert SmokeCheckKind.INTERRUPT_PROPAGATES in kinds
    assert SmokeCheckKind.DEADLOCK_DETECTOR_WORKS in kinds
    assert SmokeCheckKind.CIRCUIT_BREAKER_TRIPS in kinds
    assert SmokeCheckKind.RECOVERY_RECONSTRUCTS in kinds
    assert SmokeCheckKind.LOAD_SHEDDING_PROTECTS_CONVERSATION in kinds
    assert SmokeCheckKind.PROACTIVE_WORK_CANCELLABLE in kinds
    assert SmokeCheckKind.INTEGRATION_BOUNDARIES_HOLD in kinds
    assert SmokeCheckKind.SECURITY_BOUNDARIES_HOLD in kinds


def test_smoke_report_raise_for_failure_passes_when_successful() -> None:
    runtime = OrchestrationSmokeRuntime()
    report = runtime.run()

    report.raise_for_failure()


def test_smoke_report_raise_for_failure_raises_when_failed() -> None:
    report = OrchestrationSmokeReport(
        success=False,
        reason=SmokeReason.CHECK_FAILED,
        summary="failed",
        checks=(
            SmokeCheckResult(
                kind=SmokeCheckKind.WORKERS_HEALTHY,
                status=SmokeCheckStatus.FAILED,
                reason=SmokeReason.WORKERS_NOT_HEALTHY,
                message="bad",
            ),
        ),
        passed_count=0,
        failed_count=1,
    )

    with pytest.raises(RuntimeError):
        report.raise_for_failure()


def test_smoke_latest_report_is_queryable() -> None:
    runtime = OrchestrationSmokeRuntime()

    report = runtime.run()

    assert runtime.latest_report() == report
    assert runtime.snapshot().report_count == 1


def test_smoke_reports_are_queryable() -> None:
    runtime = OrchestrationSmokeRuntime()

    runtime.run()
    runtime.run()

    assert len(runtime.reports()) == 2


def test_smoke_snapshot_tracks_success() -> None:
    runtime = OrchestrationSmokeRuntime()

    runtime.run()
    snapshot = runtime.snapshot()

    assert snapshot.report_count == 1
    assert snapshot.last_success is True
    assert snapshot.last_failed_count == 0


def test_smoke_reset_clears_reports() -> None:
    runtime = OrchestrationSmokeRuntime()

    runtime.run()
    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.report_count == 0
    assert snapshot.last_reason == SmokeReason.RUNTIME_RESET


def test_all_smoke_checks_have_stable_values() -> None:
    assert SmokeCheckKind.WORKERS_HEALTHY.value == "workers_healthy"
    assert SmokeCheckKind.SCHEDULER_RUNS.value == "scheduler_runs"
    assert SmokeCheckKind.PARALLEL_TASKS_COMPLETE.value == (
        "parallel_tasks_complete"
    )
    assert SmokeCheckKind.INTERRUPT_PROPAGATES.value == "interrupt_propagates"
    assert SmokeCheckKind.DEADLOCK_DETECTOR_WORKS.value == (
        "deadlock_detector_works"
    )
    assert SmokeCheckKind.CIRCUIT_BREAKER_TRIPS.value == "circuit_breaker_trips"
    assert SmokeCheckKind.RECOVERY_RECONSTRUCTS.value == "recovery_reconstructs"
    assert SmokeCheckKind.LOAD_SHEDDING_PROTECTS_CONVERSATION.value == (
        "load_shedding_protects_conversation"
    )
    assert SmokeCheckKind.PROACTIVE_WORK_CANCELLABLE.value == (
        "proactive_work_cancellable"
    )
    assert SmokeCheckKind.INTEGRATION_BOUNDARIES_HOLD.value == (
        "integration_boundaries_hold"
    )
    assert SmokeCheckKind.SECURITY_BOUNDARIES_HOLD.value == (
        "security_boundaries_hold"
    )


def test_enum_values_are_stable() -> None:
    assert SmokeCheckStatus.PASSED.value == "passed"
    assert SmokeReason.REPORT_CREATED.value == "report_created"