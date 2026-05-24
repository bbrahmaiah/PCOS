from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from jarvis.memory import (
    Phase4CompletionCheck,
    Phase4CompletionCheckKind,
    Phase4CompletionGate,
    Phase4CompletionGateConfig,
    Phase4CompletionStatus,
    complete_phase4_memory,
)
from scripts.complete_phase4 import main


def test_phase4_completion_gate_config_rejects_invalid_name() -> None:
    with pytest.raises(ValueError):
        Phase4CompletionGateConfig(name=" ").validate()


def test_phase4_completion_check_requires_text_fields() -> None:
    with pytest.raises(ValidationError):
        Phase4CompletionCheck(
            name=" ",
            kind=Phase4CompletionCheckKind.COMPLETION,
            passed=True,
            detail="valid",
        )

    with pytest.raises(ValidationError):
        Phase4CompletionCheck(
            name="check",
            kind=Phase4CompletionCheckKind.COMPLETION,
            passed=True,
            detail=" ",
        )


def test_phase4_completion_gate_passes(tmp_path: Path) -> None:
    gate = Phase4CompletionGate(
        config=Phase4CompletionGateConfig(
            sqlite_path=tmp_path / "memory.db",
        )
    )

    result = gate.run()
    snapshot = gate.snapshot()

    assert result.passed is True
    assert result.status == Phase4CompletionStatus.PASSED
    assert result.failed_count == 0
    assert result.passed_count == result.check_count
    assert snapshot.run_count == 1
    assert snapshot.last_status == Phase4CompletionStatus.PASSED


def test_complete_phase4_memory_function_passes(tmp_path: Path) -> None:
    result = complete_phase4_memory(sqlite_path=tmp_path / "memory.db")

    assert result.passed is True
    assert result.failed_count == 0


def test_phase4_completion_can_disable_optional_checks(tmp_path: Path) -> None:
    result = complete_phase4_memory(
        sqlite_path=tmp_path / "memory.db",
        include_sqlite=False,
        include_vector=False,
        include_safety_audit=False,
    )
    names = {check.name for check in result.checks}

    assert result.passed is True
    assert "phase4_memory_safety_audit" not in names


def test_phase4_completion_gate_reset(tmp_path: Path) -> None:
    gate = Phase4CompletionGate(
        config=Phase4CompletionGateConfig(
            sqlite_path=tmp_path / "memory.db",
        )
    )

    gate.run()
    gate.reset()
    snapshot = gate.snapshot()

    assert snapshot.run_count == 0
    assert snapshot.last_status is None
    assert snapshot.last_error is None


def test_complete_phase4_script_main_passes(tmp_path: Path) -> None:
    exit_code = main(
        [
            "--sqlite-path",
            str(tmp_path / "memory.db"),
        ]
    )

    assert exit_code == 0


def test_phase4_completion_enum_values_are_stable() -> None:
    assert Phase4CompletionStatus.PASSED.value == "passed"
    assert Phase4CompletionStatus.FAILED.value == "failed"
    assert Phase4CompletionCheckKind.VALIDATION.value == "validation"
    assert Phase4CompletionCheckKind.SAFETY_AUDIT.value == "safety_audit"
    assert Phase4CompletionCheckKind.COMPLETION.value == "completion"