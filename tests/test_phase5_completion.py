from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.tools import (
    Phase5CompletionCheck,
    Phase5CompletionCheckKind,
    Phase5CompletionGate,
    Phase5CompletionGateConfig,
    Phase5CompletionStatus,
)


def test_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        Phase5CompletionGateConfig(name=" ").validate()


def test_check_requires_detail() -> None:
    with pytest.raises(ValidationError):
        Phase5CompletionCheck(
            kind=Phase5CompletionCheckKind.CONTRACTS,
            passed=True,
            detail=" ",
        )


def test_phase5_completion_gate_passes() -> None:
    gate = Phase5CompletionGate()

    report = gate.run()

    assert report.passed is True
    assert report.status == Phase5CompletionStatus.PASSED
    assert report.failed_checks == ()
    assert len(report.checks) == 19


def test_phase5_completion_gate_covers_all_required_checks() -> None:
    gate = Phase5CompletionGate()

    report = gate.run()
    kinds = {check.kind for check in report.checks}

    assert kinds == {
        Phase5CompletionCheckKind.CONTRACTS,
        Phase5CompletionCheckKind.REGISTRY,
        Phase5CompletionCheckKind.EXECUTION_PROTOCOL,
        Phase5CompletionCheckKind.PERMISSION_POLICY,
        Phase5CompletionCheckKind.VALIDATION,
        Phase5CompletionCheckKind.SAFE_SHELL,
        Phase5CompletionCheckKind.FILE_SYSTEM,
        Phase5CompletionCheckKind.IDE_RUNTIME,
        Phase5CompletionCheckKind.INTERRUPTION_ROLLBACK,
        Phase5CompletionCheckKind.PLANNER,
        Phase5CompletionCheckKind.AUDIT_LOG,
        Phase5CompletionCheckKind.HUMAN_APPROVAL,
        Phase5CompletionCheckKind.TOOL_MEMORY,
        Phase5CompletionCheckKind.PARALLEL_SCHEDULER,
        Phase5CompletionCheckKind.SMOKE_RUNTIME,
        Phase5CompletionCheckKind.SAFE_AUTONOMY,
        Phase5CompletionCheckKind.COGNITION_BRIDGE,
        Phase5CompletionCheckKind.SECURITY_HARDENING,
        Phase5CompletionCheckKind.FULL_PIPELINE,
    }


def test_snapshot_and_reset() -> None:
    gate = Phase5CompletionGate()

    gate.run()
    snapshot = gate.snapshot()

    assert snapshot.run_count == 1
    assert snapshot.last_status == Phase5CompletionStatus.PASSED
    assert snapshot.last_passed is True

    gate.reset()
    reset_snapshot = gate.snapshot()

    assert reset_snapshot.run_count == 0
    assert reset_snapshot.last_status is None
    assert reset_snapshot.last_passed is None


def test_enum_values_are_stable() -> None:
    assert Phase5CompletionStatus.PASSED.value == "passed"
    assert Phase5CompletionCheckKind.FULL_PIPELINE.value == "full_pipeline"