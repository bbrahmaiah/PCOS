from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.orchestration import (
    SecurityAttackKind,
    SecurityAttackVector,
    SecurityAuditFinding,
    SecurityAuditReason,
    SecurityAuditReport,
    SecurityAuditStatus,
    SecurityControl,
    SecurityHardeningAuditConfig,
    SecurityHardeningAuditRuntime,
)


def vector(
    *,
    kind: SecurityAttackKind = SecurityAttackKind.TASK_INJECTION,
) -> SecurityAttackVector:
    return SecurityAttackVector(
        kind=kind,
        description="test vector",
        expected_control=SecurityControl.TYPED_CONTRACTS,
        payload={"x": 1},
    )


def test_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        SecurityHardeningAuditConfig(name=" ").validate()


def test_config_rejects_invalid_interrupt_limit() -> None:
    config = SecurityHardeningAuditConfig(max_interrupts_per_second=0)

    with pytest.raises(ValueError):
        config.validate()


def test_config_rejects_empty_worker_allowlist() -> None:
    config = SecurityHardeningAuditConfig(allowed_worker_ids=())

    with pytest.raises(ValueError):
        config.validate()


def test_attack_vector_requires_description() -> None:
    with pytest.raises(ValidationError):
        SecurityAttackVector(
            kind=SecurityAttackKind.TASK_INJECTION,
            description=" ",
            expected_control=SecurityControl.TYPED_CONTRACTS,
        )


def test_finding_passed_property_when_blocked() -> None:
    finding = SecurityAuditFinding(
        vector=vector(),
        status=SecurityAuditStatus.BLOCKED,
        reason=SecurityAuditReason.ATTACK_BLOCKED,
        blocked=True,
        message="blocked",
    )

    assert finding.passed is True


def test_finding_passed_property_false_when_allowed() -> None:
    finding = SecurityAuditFinding(
        vector=vector(),
        status=SecurityAuditStatus.ALLOWED,
        reason=SecurityAuditReason.ATTACK_ALLOWED,
        blocked=False,
        message="allowed",
    )

    assert finding.passed is False


def test_security_audit_passes_all_vectors() -> None:
    runtime = SecurityHardeningAuditRuntime()

    report = runtime.run()

    assert report.success is True
    assert report.failed_count == 0
    assert report.passed_count == len(report.findings)
    assert report.blocked_count == len(report.findings)


def test_security_audit_contains_required_attack_vectors() -> None:
    runtime = SecurityHardeningAuditRuntime()

    report = runtime.run()
    kinds = {finding.vector.kind for finding in report.findings}

    assert SecurityAttackKind.PROMPT_DIRECT_SCHEDULE in kinds
    assert SecurityAttackKind.TASK_INJECTION in kinds
    assert SecurityAttackKind.WORKER_SPOOFING in kinds
    assert SecurityAttackKind.PRIORITY_MANIPULATION in kinds
    assert SecurityAttackKind.RESOURCE_EXHAUSTION in kinds
    assert SecurityAttackKind.INTERRUPT_FLOODING in kinds
    assert SecurityAttackKind.DEADLOCK_INJECTION in kinds
    assert SecurityAttackKind.CONTEXT_SNAPSHOT_POISONING in kinds
    assert SecurityAttackKind.BACKGROUND_TASK_HIJACKING in kinds
    assert SecurityAttackKind.DIRECT_TOOL_EXECUTION in kinds
    assert SecurityAttackKind.PROACTIVE_ACTION_ABUSE in kinds
    assert SecurityAttackKind.RECOVERY_BYPASS in kinds


def test_prompt_direct_schedule_is_blocked() -> None:
    runtime = SecurityHardeningAuditRuntime()

    report = runtime.run()
    finding = next(
        item
        for item in report.findings
        if item.vector.kind == SecurityAttackKind.PROMPT_DIRECT_SCHEDULE
    )

    assert finding.blocked is True
    assert finding.reason == SecurityAuditReason.DIRECT_SCHEDULING_BLOCKED


def test_task_injection_is_blocked() -> None:
    runtime = SecurityHardeningAuditRuntime()

    report = runtime.run()
    finding = next(
        item
        for item in report.findings
        if item.vector.kind == SecurityAttackKind.TASK_INJECTION
    )

    assert finding.blocked is True
    assert finding.reason == SecurityAuditReason.DIRECT_EXECUTION_BLOCKED


def test_worker_spoofing_is_blocked() -> None:
    runtime = SecurityHardeningAuditRuntime()

    report = runtime.run()
    finding = next(
        item
        for item in report.findings
        if item.vector.kind == SecurityAttackKind.WORKER_SPOOFING
    )

    assert finding.blocked is True
    assert finding.reason == SecurityAuditReason.WORKER_SPOOFING_BLOCKED


def test_resource_exhaustion_is_blocked() -> None:
    runtime = SecurityHardeningAuditRuntime()

    report = runtime.run()
    finding = next(
        item
        for item in report.findings
        if item.vector.kind == SecurityAttackKind.RESOURCE_EXHAUSTION
    )

    assert finding.blocked is True
    assert finding.reason == SecurityAuditReason.RESOURCE_EXHAUSTION_BLOCKED


def test_direct_tool_execution_is_blocked() -> None:
    runtime = SecurityHardeningAuditRuntime()

    report = runtime.run()
    finding = next(
        item
        for item in report.findings
        if item.vector.kind == SecurityAttackKind.DIRECT_TOOL_EXECUTION
    )

    assert finding.blocked is True
    assert finding.reason == SecurityAuditReason.DIRECT_EXECUTION_BLOCKED


def test_proactive_action_abuse_is_blocked() -> None:
    runtime = SecurityHardeningAuditRuntime()

    report = runtime.run()
    finding = next(
        item
        for item in report.findings
        if item.vector.kind == SecurityAttackKind.PROACTIVE_ACTION_ABUSE
    )

    assert finding.blocked is True
    assert finding.reason == SecurityAuditReason.PROACTIVE_ACTION_BLOCKED


def test_audit_report_raise_for_failure_passes_when_successful() -> None:
    runtime = SecurityHardeningAuditRuntime()
    report = runtime.run()

    report.raise_for_failure()


def test_audit_report_raise_for_failure_raises_when_failed() -> None:
    finding = SecurityAuditFinding(
        vector=vector(),
        status=SecurityAuditStatus.ALLOWED,
        reason=SecurityAuditReason.ATTACK_ALLOWED,
        blocked=False,
        message="allowed",
    )
    report = SecurityAuditReport(
        success=False,
        reason=SecurityAuditReason.ATTACK_ALLOWED,
        summary="failed",
        findings=(finding,),
        passed_count=0,
        failed_count=1,
        blocked_count=0,
    )

    with pytest.raises(RuntimeError):
        report.raise_for_failure()


def test_latest_report_is_queryable() -> None:
    runtime = SecurityHardeningAuditRuntime()

    report = runtime.run()

    assert runtime.latest_report() == report
    assert runtime.snapshot().report_count == 1


def test_reports_are_queryable() -> None:
    runtime = SecurityHardeningAuditRuntime()

    runtime.run()
    runtime.run()

    assert len(runtime.reports()) == 2


def test_snapshot_tracks_success() -> None:
    runtime = SecurityHardeningAuditRuntime()

    runtime.run()
    snapshot = runtime.snapshot()

    assert snapshot.report_count == 1
    assert snapshot.last_success is True
    assert snapshot.last_failed_count == 0
    assert snapshot.last_blocked_count is not None
    assert snapshot.last_blocked_count > 0


def test_reset_clears_reports() -> None:
    runtime = SecurityHardeningAuditRuntime()

    runtime.run()
    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.report_count == 0
    assert snapshot.last_reason == SecurityAuditReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert SecurityAttackKind.TASK_INJECTION.value == "task_injection"
    assert SecurityAuditStatus.BLOCKED.value == "blocked"
    assert SecurityControl.TYPED_CONTRACTS.value == "typed_contracts"
    assert SecurityAuditReason.ATTACK_BLOCKED.value == "attack_blocked"