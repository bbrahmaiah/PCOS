from __future__ import annotations

import pytest

from jarvis.cognitive import (
    Phase9CompletionCheckKind,
    Phase9CompletionGate,
    Phase9CompletionGateConfig,
    Phase9CompletionGateStatus,
)


def test_phase9_completion_config_rejects_empty_user_label() -> None:
    with pytest.raises(ValueError):
        Phase9CompletionGateConfig(user_label=" ")


def test_phase9_completion_gate_passes() -> None:
    report = Phase9CompletionGate(
        config=Phase9CompletionGateConfig(user_label="Balu")
    ).run()

    assert report.status == Phase9CompletionGateStatus.PASSED
    assert report.passed is True
    assert report.failed_count == 0
    assert report.error is None

    kinds = {check.kind for check in report.checks}

    assert Phase9CompletionCheckKind.DESIGN_GATE in kinds
    assert Phase9CompletionCheckKind.ATTENTION_RUNTIME in kinds
    assert Phase9CompletionCheckKind.WORKING_MEMORY_RUNTIME in kinds
    assert Phase9CompletionCheckKind.GOAL_RUNTIME in kinds
    assert Phase9CompletionCheckKind.PLANNING_RUNTIME in kinds
    assert Phase9CompletionCheckKind.PERSONALITY_RUNTIME in kinds
    assert Phase9CompletionCheckKind.SESSION_RUNTIME in kinds
    assert Phase9CompletionCheckKind.INTEGRATION_RUNTIME in kinds
    assert Phase9CompletionCheckKind.INTERRUPTION_BEHAVIOR in kinds
    assert Phase9CompletionCheckKind.SAFETY_BOUNDARY in kinds
    assert Phase9CompletionCheckKind.PRESENCE_CONTINUITY in kinds


def test_phase9_completion_gate_has_all_checks() -> None:
    report = Phase9CompletionGate().run()

    assert len(report.checks) == 11
    assert report.passed_count == 11


def test_phase9_completion_gate_metadata_is_preserved() -> None:
    report = Phase9CompletionGate(
        config=Phase9CompletionGateConfig(
            user_label="Balu",
            metadata={"phase": "9"},
        )
    ).run()

    assert report.metadata["phase"] == "9"


def test_phase9_completion_enum_values_are_stable() -> None:
    assert Phase9CompletionGateStatus.PASSED.value == "passed"
    assert Phase9CompletionCheckKind.SESSION_RUNTIME.value == "session_runtime"