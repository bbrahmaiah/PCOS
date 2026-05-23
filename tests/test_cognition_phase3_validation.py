from __future__ import annotations

from jarvis.cognition import (
    CognitionPhase3Validator,
    ValidationLocalLLMBackend,
    validate_phase3_cognition,
)


def test_phase3_cognition_validation_passes() -> None:
    report = validate_phase3_cognition()

    assert report.passed is True
    assert report.failed_count == 0
    assert report.passed_count == report.total_count
    assert report.total_count >= 12


def test_phase3_cognition_validation_check_names_are_stable() -> None:
    report = CognitionPhase3Validator().validate()

    check_names = {check.name for check in report.checks}

    assert "session_context_connected" in check_names
    assert "memory_context_connected" in check_names
    assert "response_planning_connected" in check_names
    assert "action_planning_connected" in check_names
    assert "local_llm_adapter_connected" in check_names
    assert "spoken_policy_connected" in check_names
    assert "streaming_pipeline_connected" in check_names
    assert "streaming_whitespace_preserved" in check_names
    assert "streaming_diagnostics_connected" in check_names
    assert "dangerous_action_blocked" in check_names
    assert "terminal_action_blocked_by_default" in check_names
    assert "safe_action_requires_permission" in check_names


def test_validation_local_llm_backend_generates_and_streams() -> None:
    backend = ValidationLocalLLMBackend()

    assert backend.name == "validation_local_llm_backend"

    snapshot = backend.snapshot()

    assert snapshot.request_count == 0
    assert snapshot.streaming_count == 0
    assert snapshot.cancelled_count == 0