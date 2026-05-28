from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.latency import (
    DegradationMode,
    LoadDegradationRuntimeConfig,
    LoadDegradationValidationRuntime,
    LoadSample,
    LoadScenarioEvaluation,
    LoadScenarioKind,
    LoadValidationReason,
    LoadValidationReport,
    LoadValidationStatus,
)


def test_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        LoadDegradationRuntimeConfig(name=" ").validate()


def test_config_rejects_invalid_targets() -> None:
    with pytest.raises(ValueError):
        LoadDegradationRuntimeConfig(conversation_target_ms=0).validate()

    with pytest.raises(ValueError):
        LoadDegradationRuntimeConfig(memory_retrieval_target_ms=0).validate()


def test_config_requires_realistic_scale() -> None:
    with pytest.raises(ValueError):
        LoadDegradationRuntimeConfig(long_conversation_turns=49).validate()

    with pytest.raises(ValueError):
        LoadDegradationRuntimeConfig(memory_pressure_entries=9999).validate()

    with pytest.raises(ValueError):
        LoadDegradationRuntimeConfig(interruption_storm_count=4).validate()


def test_sample_rejects_invalid_latency() -> None:
    with pytest.raises(ValidationError):
        LoadSample(
            scenario=LoadScenarioKind.LONG_CONVERSATION,
            latency_ms=-1,
        )


def test_runtime_creates_session_with_default_scenarios() -> None:
    runtime = LoadDegradationValidationRuntime()
    state = runtime.create_session()

    assert state.status == LoadValidationStatus.CREATED
    assert state.scenario_count == 4
    assert runtime.snapshot().session_count == 1


def test_runtime_starts_session() -> None:
    runtime = LoadDegradationValidationRuntime()
    state = runtime.create_session()

    result = runtime.start_session(state.session_id)

    assert result.success is True
    assert result.status == LoadValidationStatus.RUNNING


def test_record_sample_tracks_degradation_and_load_shedding() -> None:
    runtime = LoadDegradationValidationRuntime()
    state = runtime.create_session()

    runtime.start_session(state.session_id)
    result = runtime.record_sample(
        session_id=state.session_id,
        sample=LoadSample(
            scenario=LoadScenarioKind.BACKGROUND_TASK_PRESSURE,
            latency_ms=500.0,
            background_tasks=3,
            degradation_mode=DegradationMode.SHED_BACKGROUND,
            load_shedding_active=True,
        ),
    )

    updated = runtime.state_for(state.session_id)

    assert result.success is True
    assert updated is not None
    assert updated.sample_count == 1
    assert updated.degradation_count == 1
    assert updated.load_shedding_count == 1


def test_record_sample_rejects_non_running_session() -> None:
    runtime = LoadDegradationValidationRuntime()
    state = runtime.create_session()

    result = runtime.record_sample(
        session_id=state.session_id,
        sample=LoadSample(
            scenario=LoadScenarioKind.LONG_CONVERSATION,
            latency_ms=500.0,
        ),
    )

    assert result.success is False
    assert result.reason == LoadValidationReason.SESSION_NOT_RUNNING


def test_simulated_suite_passes() -> None:
    runtime = LoadDegradationValidationRuntime()
    state = runtime.create_session()

    runtime.start_session(state.session_id)
    report = runtime.run_simulated_suite(session_id=state.session_id)

    assert report.status == LoadValidationStatus.PASSED
    assert report.scenario_count == 4
    assert report.failed_count == 0
    assert report.degradation_count > 0
    assert report.load_shedding_count > 0


def test_simulated_suite_fails_when_degraded_badly() -> None:
    runtime = LoadDegradationValidationRuntime()
    state = runtime.create_session()

    runtime.start_session(state.session_id)
    report = runtime.run_simulated_suite(
        session_id=state.session_id,
        failing=True,
    )

    assert report.status == LoadValidationStatus.FAILED
    assert report.failed_count == 4


def test_long_conversation_latency_drift_is_detected() -> None:
    runtime = LoadDegradationValidationRuntime()
    state = runtime.create_session()

    runtime.start_session(state.session_id)

    for turn in range(1, 51):
        runtime.record_sample(
            session_id=state.session_id,
            sample=LoadSample(
                scenario=LoadScenarioKind.LONG_CONVERSATION,
                latency_ms=500.0 + (turn * 4.0),
                turn_index=turn,
                degradation_mode=DegradationMode.COMPRESS_CONTEXT,
            ),
        )

    report = runtime.build_report(state.session_id)
    long_eval = _evaluation_for(report, LoadScenarioKind.LONG_CONVERSATION)

    assert long_eval.status == LoadValidationStatus.FAILED
    assert long_eval.reason == LoadValidationReason.SCENARIO_FAILED_LATENCY_DRIFT


def test_background_pressure_requires_load_shedding() -> None:
    runtime = LoadDegradationValidationRuntime()
    state = runtime.create_session()

    runtime.start_session(state.session_id)

    for _ in range(8):
        runtime.record_sample(
            session_id=state.session_id,
            sample=LoadSample(
                scenario=LoadScenarioKind.BACKGROUND_TASK_PRESSURE,
                latency_ms=500.0,
                background_tasks=3,
                degradation_mode=DegradationMode.SHED_BACKGROUND,
                load_shedding_active=False,
            ),
        )

    report = runtime.build_report(state.session_id)
    evaluation = _evaluation_for(report, LoadScenarioKind.BACKGROUND_TASK_PRESSURE)

    assert evaluation.status == LoadValidationStatus.FAILED
    assert evaluation.reason == LoadValidationReason.SCENARIO_FAILED_DEGRADATION


def test_memory_pressure_budget_failure_is_detected() -> None:
    runtime = LoadDegradationValidationRuntime()
    state = runtime.create_session()

    runtime.start_session(state.session_id)

    for _ in range(8):
        runtime.record_sample(
            session_id=state.session_id,
            sample=LoadSample(
                scenario=LoadScenarioKind.MEMORY_PRESSURE,
                latency_ms=250.0,
                memory_entries=10_000,
                degradation_mode=DegradationMode.FAST_MEMORY,
            ),
        )

    report = runtime.build_report(state.session_id)
    evaluation = _evaluation_for(report, LoadScenarioKind.MEMORY_PRESSURE)

    assert evaluation.status == LoadValidationStatus.FAILED
    assert evaluation.reason == LoadValidationReason.SCENARIO_FAILED_BUDGET


def test_interruption_storm_recovery_failure_is_detected() -> None:
    runtime = LoadDegradationValidationRuntime()
    state = runtime.create_session()

    runtime.start_session(state.session_id)

    for index in range(5):
        runtime.record_sample(
            session_id=state.session_id,
            sample=LoadSample(
                scenario=LoadScenarioKind.INTERRUPTION_STORM,
                latency_ms=200.0,
                interruption_index=index + 1,
                degradation_mode=DegradationMode.RECOVERY_PRIORITY,
                recovery_clean=index != 3,
            ),
        )

    report = runtime.build_report(state.session_id)
    evaluation = _evaluation_for(report, LoadScenarioKind.INTERRUPTION_STORM)

    assert evaluation.status == LoadValidationStatus.FAILED
    assert evaluation.reason == (
        LoadValidationReason.SCENARIO_FAILED_INTERRUPTION_RECOVERY
    )


def test_cancel_session() -> None:
    runtime = LoadDegradationValidationRuntime()
    state = runtime.create_session()

    result = runtime.cancel_session(state.session_id)

    assert result.success is True
    assert result.status == LoadValidationStatus.CANCELLED


def test_report_is_queryable() -> None:
    runtime = LoadDegradationValidationRuntime()
    state = runtime.create_session()

    runtime.start_session(state.session_id)
    report = runtime.run_simulated_suite(session_id=state.session_id)

    assert runtime.latest_report() == report
    assert runtime.reports() == (report,)


def test_snapshot_tracks_counts() -> None:
    runtime = LoadDegradationValidationRuntime()
    state = runtime.create_session()

    runtime.start_session(state.session_id)
    runtime.run_simulated_suite(session_id=state.session_id)
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.passed_count == 1
    assert snapshot.sample_count > 0
    assert snapshot.report_count == 1


def test_reset_clears_runtime_state() -> None:
    runtime = LoadDegradationValidationRuntime()
    state = runtime.create_session()

    runtime.start_session(state.session_id)
    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.sample_count == 0
    assert snapshot.last_reason == LoadValidationReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert LoadScenarioKind.LONG_CONVERSATION.value == "long_conversation"
    assert DegradationMode.SHED_BACKGROUND.value == "shed_background"
    assert LoadValidationStatus.PASSED.value == "passed"


def _evaluation_for(
    report: LoadValidationReport,
    scenario: LoadScenarioKind,
) -> LoadScenarioEvaluation:
    return [
        evaluation
        for evaluation in report.evaluations
        if evaluation.scenario == scenario
    ][0]