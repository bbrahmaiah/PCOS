from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.conversation import (
    Phase45CompletionCheck,
    Phase45CompletionCheckKind,
    Phase45CompletionGate,
    Phase45CompletionGateConfig,
    Phase45CompletionStatus,
    complete_phase45_conversation,
)
from scripts.complete_phase45 import main


def test_phase45_completion_gate_config_rejects_invalid_name() -> None:
    with pytest.raises(ValueError):
        Phase45CompletionGateConfig(name=" ").validate()


def test_phase45_completion_check_requires_text_fields() -> None:
    with pytest.raises(ValidationError):
        Phase45CompletionCheck(
            name=" ",
            kind=Phase45CompletionCheckKind.COMPLETION,
            passed=True,
            detail="valid",
        )

    with pytest.raises(ValidationError):
        Phase45CompletionCheck(
            name="check",
            kind=Phase45CompletionCheckKind.COMPLETION,
            passed=True,
            detail=" ",
        )


def test_phase45_completion_gate_passes() -> None:
    gate = Phase45CompletionGate()

    result = gate.run()
    snapshot = gate.snapshot()

    assert result.passed is True
    assert result.status == Phase45CompletionStatus.PASSED
    assert result.failed_count == 0
    assert result.passed_count == result.check_count
    assert snapshot.run_count == 1
    assert snapshot.last_status == Phase45CompletionStatus.PASSED


def test_complete_phase45_conversation_function_passes() -> None:
    result = complete_phase45_conversation()

    assert result.passed is True
    assert result.failed_count == 0


def test_phase45_completion_gate_reset() -> None:
    gate = Phase45CompletionGate()

    gate.run()
    gate.reset()
    snapshot = gate.snapshot()

    assert snapshot.run_count == 0
    assert snapshot.last_status is None
    assert snapshot.last_error is None


def test_complete_phase45_script_main_passes() -> None:
    assert main() == 0


def test_phase45_completion_enum_values_are_stable() -> None:
    assert Phase45CompletionStatus.PASSED.value == "passed"
    assert Phase45CompletionStatus.FAILED.value == "failed"
    assert Phase45CompletionCheckKind.TURN_DETECTION.value == "turn_detection"
    assert Phase45CompletionCheckKind.STATE_MACHINE.value == "state_machine"
    assert Phase45CompletionCheckKind.ENDPOINTING.value == "endpointing"
    assert Phase45CompletionCheckKind.REAL_RUNTIME.value == "real_runtime"
    assert Phase45CompletionCheckKind.COMPLETION.value == "completion"