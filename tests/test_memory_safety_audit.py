from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.memory import (
    MemorySafetyAuditCheck,
    MemorySafetyAuditor,
    MemorySafetyAuditorConfig,
    MemorySafetyAuditStatus,
    MemorySafetyRiskLevel,
    audit_phase4_memory_safety,
)
from scripts.audit_memory_safety import main


def test_memory_safety_auditor_config_rejects_invalid_name() -> None:
    with pytest.raises(ValueError):
        MemorySafetyAuditorConfig(name=" ").validate()


def test_memory_safety_audit_check_requires_text_fields() -> None:
    with pytest.raises(ValidationError):
        MemorySafetyAuditCheck(
            name=" ",
            passed=True,
            risk_level=MemorySafetyRiskLevel.LOW,
            detail="valid",
        )

    with pytest.raises(ValidationError):
        MemorySafetyAuditCheck(
            name="check",
            passed=True,
            risk_level=MemorySafetyRiskLevel.LOW,
            detail=" ",
        )


def test_memory_safety_auditor_passes() -> None:
    auditor = MemorySafetyAuditor()

    result = auditor.audit()
    snapshot = auditor.snapshot()

    assert result.passed is True
    assert result.status == MemorySafetyAuditStatus.PASSED
    assert result.failed_count == 0
    assert result.passed_count == result.check_count
    assert snapshot.audit_count == 1
    assert snapshot.last_status == MemorySafetyAuditStatus.PASSED


def test_memory_safety_auditor_can_disable_restricted_opt_in_check() -> None:
    auditor = MemorySafetyAuditor(
        config=MemorySafetyAuditorConfig(
            include_restricted_opt_in_check=False
        )
    )

    result = auditor.audit()
    names = {check.name for check in result.checks}

    assert result.passed is True
    assert "restricted_context_requires_explicit_opt_in" not in names


def test_memory_safety_auditor_reset() -> None:
    auditor = MemorySafetyAuditor()

    auditor.audit()
    auditor.reset()
    snapshot = auditor.snapshot()

    assert snapshot.audit_count == 0
    assert snapshot.last_status is None
    assert snapshot.last_error is None


def test_audit_phase4_memory_safety_function_passes() -> None:
    result = audit_phase4_memory_safety()

    assert result.passed is True
    assert result.failed_count == 0


def test_audit_memory_safety_script_main_passes() -> None:
    exit_code = main()

    assert exit_code == 0


def test_memory_safety_enum_values_are_stable() -> None:
    assert MemorySafetyAuditStatus.PASSED.value == "passed"
    assert MemorySafetyAuditStatus.FAILED.value == "failed"
    assert MemorySafetyRiskLevel.LOW.value == "low"
    assert MemorySafetyRiskLevel.CRITICAL.value == "critical"