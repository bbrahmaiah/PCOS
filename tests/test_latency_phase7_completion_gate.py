from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.latency import (
    Phase7CompletionComponent,
    Phase7CompletionGateConfig,
    Phase7CompletionGateRuntime,
    Phase7CompletionReason,
    Phase7CompletionStatus,
    Phase7GateCheck,
)


def test_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        Phase7CompletionGateConfig(name=" ").validate()


def test_config_requires_exactly_20_checks() -> None:
    with pytest.raises(ValueError):
        Phase7CompletionGateConfig(required_check_count=19).validate()


def test_gate_check_requires_evidence() -> None:
    with pytest.raises(ValidationError):
        Phase7GateCheck(
            component=Phase7CompletionComponent.LATENCY_CONTRACTS,
            passed=True,
            evidence=" ",
        )


def test_runtime_creates_session() -> None:
    runtime = Phase7CompletionGateRuntime()
    state = runtime.create_session()

    assert state.status == Phase7CompletionStatus.CREATED
    assert runtime.snapshot().session_count == 1


def test_start_gate() -> None:
    runtime = Phase7CompletionGateRuntime()
    state = runtime.create_session()

    result = runtime.start_gate(state.session_id)

    assert result.success is True
    assert result.status == Phase7CompletionStatus.RUNNING


def test_gate_seals_phase7_when_all_checks_pass() -> None:
    runtime = Phase7CompletionGateRuntime()
    state = runtime.create_session()

    runtime.start_gate(state.session_id)
    report = runtime.run_gate(session_id=state.session_id)

    assert report.status == Phase7CompletionStatus.SEALED
    assert report.check_count == 20
    assert report.passed_count == 20
    assert report.failed_count == 0
    assert "JARVIS feels alive" in report.sealed_message


def test_gate_fails_when_any_component_fails() -> None:
    runtime = Phase7CompletionGateRuntime()
    state = runtime.create_session()

    runtime.start_gate(state.session_id)
    report = runtime.run_gate(
        session_id=state.session_id,
        failing_component=Phase7CompletionComponent.PERCEPTUAL_SMOKE,
    )

    assert report.status == Phase7CompletionStatus.FAILED
    assert report.failed_count == 1
    assert any(
        check.component == Phase7CompletionComponent.PERCEPTUAL_SMOKE
        and not check.passed
        for check in report.checks
    )


def test_gate_includes_validation_checks() -> None:
    runtime = Phase7CompletionGateRuntime()
    state = runtime.create_session()

    runtime.start_gate(state.session_id)
    report = runtime.run_gate(session_id=state.session_id)
    components = {check.component for check in report.checks}

    assert Phase7CompletionComponent.LATENCY_REGRESSION in components
    assert Phase7CompletionComponent.PERCEPTUAL_SMOKE in components
    assert Phase7CompletionComponent.LOAD_DEGRADATION in components


def test_gate_includes_no_correctness_regression_check() -> None:
    runtime = Phase7CompletionGateRuntime()
    state = runtime.create_session()

    runtime.start_gate(state.session_id)
    report = runtime.run_gate(session_id=state.session_id)

    assert any(
        check.component == Phase7CompletionComponent.NO_CORRECTNESS_REGRESSION
        and check.passed
        for check in report.checks
    )


def test_run_gate_rejects_missing_session() -> None:
    runtime = Phase7CompletionGateRuntime()

    with pytest.raises(ValueError):
        runtime.run_gate(session_id="missing")


def test_run_gate_rejects_not_running_session() -> None:
    runtime = Phase7CompletionGateRuntime()
    state = runtime.create_session()

    with pytest.raises(ValueError):
        runtime.run_gate(session_id=state.session_id)


def test_cancel_session() -> None:
    runtime = Phase7CompletionGateRuntime()
    state = runtime.create_session()

    result = runtime.cancel_session(state.session_id)

    assert result.success is True
    assert result.status == Phase7CompletionStatus.CANCELLED


def test_report_is_queryable() -> None:
    runtime = Phase7CompletionGateRuntime()
    state = runtime.create_session()

    runtime.start_gate(state.session_id)
    report = runtime.run_gate(session_id=state.session_id)

    assert runtime.latest_report() == report
    assert runtime.reports() == (report,)


def test_snapshot_tracks_counts() -> None:
    runtime = Phase7CompletionGateRuntime()
    state = runtime.create_session()

    runtime.start_gate(state.session_id)
    runtime.run_gate(session_id=state.session_id)
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.sealed_count == 1
    assert snapshot.report_count == 1
    assert snapshot.last_reason == Phase7CompletionReason.PHASE7_SEALED


def test_reset_clears_runtime_state() -> None:
    runtime = Phase7CompletionGateRuntime()
    state = runtime.create_session()

    runtime.start_gate(state.session_id)
    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.report_count == 0
    assert snapshot.last_reason == Phase7CompletionReason.RUNTIME_RESET


def test_all_expected_components_are_present_once() -> None:
    runtime = Phase7CompletionGateRuntime()
    state = runtime.create_session()

    runtime.start_gate(state.session_id)
    report = runtime.run_gate(session_id=state.session_id)
    components = [check.component for check in report.checks]

    assert len(components) == 20
    assert len(set(components)) == 20


def test_enum_values_are_stable() -> None:
    assert Phase7CompletionStatus.SEALED.value == "sealed"
    assert Phase7CompletionReason.PHASE7_SEALED.value == "phase7_sealed"
    assert Phase7CompletionComponent.PERCEPTUAL_SMOKE.value == "perceptual_smoke"