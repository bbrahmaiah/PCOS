from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.orchestration import (
    Phase6CompletionCertificate,
    Phase6CompletionCheckKind,
    Phase6CompletionCheckResult,
    Phase6CompletionGateConfig,
    Phase6CompletionGateReport,
    Phase6CompletionGateRuntime,
    Phase6GateReason,
    Phase6GateStatus,
    Phase6SealLevel,
)


def test_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        Phase6CompletionGateConfig(name=" ").validate()


def test_check_result_requires_message() -> None:
    with pytest.raises(ValidationError):
        Phase6CompletionCheckResult(
            kind=Phase6CompletionCheckKind.TYPED_CONTRACTS_COMPLETE,
            status=Phase6GateStatus.PASSED,
            reason=Phase6GateReason.CHECK_PASSED,
            message=" ",
        )


def test_check_result_passed_property() -> None:
    result = Phase6CompletionCheckResult(
        kind=Phase6CompletionCheckKind.TYPED_CONTRACTS_COMPLETE,
        status=Phase6GateStatus.PASSED,
        reason=Phase6GateReason.CHECK_PASSED,
        message="ok",
    )

    assert result.passed is True


def test_certificate_requires_summary() -> None:
    with pytest.raises(ValidationError):
        Phase6CompletionCertificate(
            seal_level=Phase6SealLevel.SEALED,
            summary=" ",
            passed_count=20,
            failed_count=0,
        )


def test_completion_gate_passes() -> None:
    runtime = Phase6CompletionGateRuntime()

    report = runtime.run()

    assert report.success is True
    assert report.seal_level == Phase6SealLevel.SEALED
    assert report.failed_count == 0
    assert report.certificate is not None


def test_completion_gate_runs_all_required_checks() -> None:
    runtime = Phase6CompletionGateRuntime()

    report = runtime.run()
    kinds = {check.kind for check in report.checks}

    assert Phase6CompletionCheckKind.TYPED_CONTRACTS_COMPLETE in kinds
    assert Phase6CompletionCheckKind.WORKER_REGISTRY_HEALTHY in kinds
    assert Phase6CompletionCheckKind.TASK_JOB_MODEL_STABLE in kinds
    assert Phase6CompletionCheckKind.ATTENTION_PROTECTS_CONVERSATION in kinds
    assert Phase6CompletionCheckKind.RESOURCE_BUDGET_ENFORCED in kinds
    assert Phase6CompletionCheckKind.STATE_MACHINE_CORRECT in kinds
    assert Phase6CompletionCheckKind.SCHEDULER_WORKS in kinds
    assert Phase6CompletionCheckKind.CONTEXT_SNAPSHOT_PREVENTS_DRIFT in kinds
    assert Phase6CompletionCheckKind.WORKER_COORDINATION_EVENT_BUS_ONLY in kinds
    assert Phase6CompletionCheckKind.BACKGROUND_YIELDS_TO_FOREGROUND in kinds
    assert Phase6CompletionCheckKind.INTERRUPTS_PROPAGATE_IN_ORDER in kinds
    assert Phase6CompletionCheckKind.DEADLOCK_DETECTION_PREVENTION in kinds
    assert Phase6CompletionCheckKind.CIRCUIT_BREAKERS_ISOLATE_FAILURES in kinds
    assert Phase6CompletionCheckKind.FULL_OBSERVABILITY in kinds
    assert Phase6CompletionCheckKind.LOAD_SHEDDING_PROTECTS_UX in kinds
    assert Phase6CompletionCheckKind.RECOVERY_RECONSTRUCTS_STATE in kinds
    assert Phase6CompletionCheckKind.PHASES_1_TO_5_INTEGRATED in kinds
    assert Phase6CompletionCheckKind.PROACTIVE_ENGINE_STABLE in kinds
    assert Phase6CompletionCheckKind.SMOKE_RUNTIME_PASSES in kinds
    assert Phase6CompletionCheckKind.SECURITY_AUDIT_PASSES in kinds


def test_completion_gate_report_raise_for_failure_passes_when_successful() -> None:
    runtime = Phase6CompletionGateRuntime()
    report = runtime.run()

    report.raise_for_failure()


def test_completion_gate_report_raise_for_failure_raises_when_failed() -> None:
    failed_check = Phase6CompletionCheckResult(
        kind=Phase6CompletionCheckKind.SECURITY_AUDIT_PASSES,
        status=Phase6GateStatus.FAILED,
        reason=Phase6GateReason.SECURITY_FAILED,
        message="security failed",
    )
    report = Phase6CompletionGateReport(
        success=False,
        reason=Phase6GateReason.GATE_FAILED,
        seal_level=Phase6SealLevel.UNSEALED,
        summary="failed",
        checks=(failed_check,),
        passed_count=0,
        failed_count=1,
    )

    with pytest.raises(RuntimeError):
        report.raise_for_failure()


def test_latest_report_is_queryable() -> None:
    runtime = Phase6CompletionGateRuntime()

    report = runtime.run()

    assert runtime.latest_report() == report
    assert runtime.snapshot().report_count == 1


def test_reports_are_queryable() -> None:
    runtime = Phase6CompletionGateRuntime()

    runtime.run()
    runtime.run()

    assert len(runtime.reports()) == 2


def test_snapshot_tracks_success() -> None:
    runtime = Phase6CompletionGateRuntime()

    runtime.run()
    snapshot = runtime.snapshot()

    assert snapshot.report_count == 1
    assert snapshot.last_success is True
    assert snapshot.last_seal_level == Phase6SealLevel.SEALED
    assert snapshot.last_failed_count == 0


def test_reset_clears_reports() -> None:
    runtime = Phase6CompletionGateRuntime()

    runtime.run()
    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.report_count == 0
    assert snapshot.last_reason == Phase6GateReason.RUNTIME_RESET


def test_certificate_metadata_marks_real_time_os() -> None:
    runtime = Phase6CompletionGateRuntime()
    report = runtime.run()

    assert report.certificate is not None
    assert report.certificate.metadata["real_time_personal_cognition_os"] is True


def test_enum_values_are_stable() -> None:
    assert Phase6GateStatus.PASSED.value == "passed"
    assert Phase6GateReason.GATE_PASSED.value == "gate_passed"
    assert Phase6SealLevel.SEALED.value == "sealed"
    assert Phase6CompletionCheckKind.SECURITY_AUDIT_PASSES.value == (
        "security_audit_passes"
    )