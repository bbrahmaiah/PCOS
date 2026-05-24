from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from jarvis.memory import (
    MemoryPhase4ValidationCheck,
    MemoryPhase4ValidationStatus,
    MemoryPhase4Validator,
    MemoryPhase4ValidatorConfig,
    validate_phase4_memory,
)
from scripts.validate_memory_phase4 import main


def test_memory_phase4_validator_config_rejects_invalid_name() -> None:
    with pytest.raises(ValueError):
        MemoryPhase4ValidatorConfig(name=" ").validate()


def test_memory_phase4_validation_check_requires_text_fields() -> None:
    with pytest.raises(ValidationError):
        MemoryPhase4ValidationCheck(
            name=" ",
            passed=True,
            detail="valid",
        )

    with pytest.raises(ValidationError):
        MemoryPhase4ValidationCheck(
            name="check",
            passed=True,
            detail=" ",
        )


def test_memory_phase4_validator_passes(tmp_path: Path) -> None:
    validator = MemoryPhase4Validator(
        config=MemoryPhase4ValidatorConfig(
            sqlite_path=tmp_path / "memory.db",
        )
    )

    result = validator.validate()
    snapshot = validator.snapshot()

    assert result.passed is True
    assert result.status == MemoryPhase4ValidationStatus.PASSED
    assert result.failed_count == 0
    assert result.passed_count == result.check_count
    assert snapshot.validation_count == 1
    assert snapshot.last_status == MemoryPhase4ValidationStatus.PASSED


def test_validate_phase4_memory_function_passes(tmp_path: Path) -> None:
    result = validate_phase4_memory(sqlite_path=tmp_path / "memory.db")

    assert result.passed is True
    assert result.failed_count == 0


def test_memory_phase4_validator_can_disable_optional_checks() -> None:
    result = validate_phase4_memory(
        include_sqlite=False,
        include_vector=False,
    )

    names = {check.name for check in result.checks}

    assert result.passed is True
    assert "sqlite_persistence" not in names
    assert "vector_boundary" not in names


def test_memory_phase4_validator_reset(tmp_path: Path) -> None:
    validator = MemoryPhase4Validator(
        config=MemoryPhase4ValidatorConfig(
            sqlite_path=tmp_path / "memory.db",
        )
    )

    validator.validate()
    validator.reset()

    snapshot = validator.snapshot()

    assert snapshot.validation_count == 0
    assert snapshot.last_status is None
    assert snapshot.last_error is None


def test_validate_memory_phase4_script_main_passes(tmp_path: Path) -> None:
    exit_code = main(
        [
            "--sqlite-path",
            str(tmp_path / "memory.db"),
        ]
    )

    assert exit_code == 0


def test_memory_phase4_validation_status_values_are_stable() -> None:
    assert MemoryPhase4ValidationStatus.PASSED.value == "passed"
    assert MemoryPhase4ValidationStatus.FAILED.value == "failed"