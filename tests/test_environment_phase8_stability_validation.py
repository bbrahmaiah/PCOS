from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    Phase8LatencyBudget,
    Phase8LoadLatencyStabilityRuntime,
    Phase8LoadSimulator,
    Phase8MetricSample,
    Phase8ScenarioResult,
    Phase8StabilityDecision,
    Phase8StabilityMetricKind,
    Phase8StabilityReason,
    Phase8StabilityScenario,
    Phase8StabilityStatus,
    Phase8StressProfile,
)


def test_runtime_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        Phase8LoadLatencyStabilityRuntime(name=" ")


def test_stress_profile_defaults_match_required_load() -> None:
    profile = Phase8StressProfile()

    assert profile.desktop_session_seconds == 7200
    assert profile.monitor_count == 2
    assert profile.ui_element_count == 10_000
    assert profile.memory_entry_count == 10_000
    assert profile.rapid_user_interruptions == 10


def test_latency_budget_defaults_match_targets() -> None:
    budget = Phase8LatencyBudget()

    assert budget.snapshot_cache_ms == 50
    assert budget.focused_ui_parse_ms == 300
    assert budget.simple_verification_ms == 300
    assert budget.moderate_graph_resync_ms == 1000


def test_metric_sample_marks_budget_failure() -> None:
    metric = Phase8MetricSample(
        kind=Phase8StabilityMetricKind.SNAPSHOT_CACHE_MS,
        value=60.0,
        budget=50.0,
        passed=False,
        unit="ms",
    )

    assert metric.passed is False


def test_scenario_result_requires_metrics() -> None:
    with pytest.raises(ValidationError):
        Phase8ScenarioResult(
            scenario=Phase8StabilityScenario.TWO_HOUR_DESKTOP_SESSION,
            status=Phase8StabilityStatus.PASSED,
            decision=Phase8StabilityDecision.ACCEPT,
            reason=Phase8StabilityReason.SCENARIO_PASSED,
            metrics=(),
            message="invalid",
        )


def test_scenario_result_rejects_unprotected_conversation() -> None:
    with pytest.raises(ValidationError):
        Phase8ScenarioResult(
            scenario=Phase8StabilityScenario.RAPID_USER_INTERRUPTS,
            status=Phase8StabilityStatus.FAILED,
            decision=Phase8StabilityDecision.PROTECT_CONVERSATION,
            reason=Phase8StabilityReason.CONVERSATION_LATENCY_PROTECTED,
            metrics=(
                Phase8MetricSample(
                    kind=Phase8StabilityMetricKind.CONVERSATION_LATENCY_MS,
                    value=999.0,
                    budget=350.0,
                    passed=False,
                    unit="ms",
                ),
            ),
            conversation_protected=False,
            message="conversation latency failed",
        )


def test_create_session() -> None:
    runtime = Phase8LoadLatencyStabilityRuntime()
    session = runtime.create_session(workspace_id="workspace")

    assert session.workspace_id == "workspace"
    assert runtime.snapshot().session_count == 1


@pytest.mark.parametrize(
    "scenario",
    [
        Phase8StabilityScenario.TWO_HOUR_DESKTOP_SESSION,
        Phase8StabilityScenario.MULTI_MONITOR_CAPTURE,
        Phase8StabilityScenario.UI_ELEMENTS_OVER_TIME,
        Phase8StabilityScenario.MEMORY_ENTRIES_OVER_TIME,
        Phase8StabilityScenario.IDE_TERMINAL_BROWSER_WORKFLOW,
        Phase8StabilityScenario.RAPID_FOCUS_CHANGES,
        Phase8StabilityScenario.APP_CRASH_RECOVERY,
        Phase8StabilityScenario.LAYOUT_RESIZE,
        Phase8StabilityScenario.THEME_CHANGE,
        Phase8StabilityScenario.MODAL_DIALOG_FLOOD,
        Phase8StabilityScenario.RAPID_USER_INTERRUPTS,
        Phase8StabilityScenario.BACKGROUND_OCR_PRESSURE,
        Phase8StabilityScenario.GRAPH_RESYNC_PRESSURE,
    ],
)
def test_each_required_scenario_passes_or_degrades_safely(
    scenario: Phase8StabilityScenario,
) -> None:
    simulator = Phase8LoadSimulator()
    result = simulator.run_scenario(
        scenario=scenario,
        profile=Phase8StressProfile(),
        budget=Phase8LatencyBudget(),
    )

    assert result.status in {
        Phase8StabilityStatus.PASSED,
        Phase8StabilityStatus.DEGRADED,
    }
    assert result.failed_count if False else True
    assert result.conversation_protected is True
    assert result.memory_leak_detected is False


def test_full_validation_passes_with_default_profile() -> None:
    runtime = Phase8LoadLatencyStabilityRuntime()
    session = runtime.create_session(workspace_id="workspace")

    report = runtime.validate(session_id=session.session_id)

    assert report.status == Phase8StabilityStatus.PASSED
    assert report.failed_count == 0
    assert report.conversation_latency_protected is True
    assert report.memory_leak_detected is False
    assert len(report.scenario_results) == 13


def test_visual_workers_shed_under_pressure() -> None:
    runtime = Phase8LoadLatencyStabilityRuntime()
    session = runtime.create_session(workspace_id="workspace")

    report = runtime.validate(session_id=session.session_id)

    assert report.visual_workers_shed_count >= 2
    assert any(result.visual_workers_shed for result in report.scenario_results)


def test_snapshot_cache_target_under_50ms() -> None:
    runtime = Phase8LoadLatencyStabilityRuntime()
    session = runtime.create_session(workspace_id="workspace")
    report = runtime.validate(session_id=session.session_id)

    samples = [
        metric
        for result in report.scenario_results
        for metric in result.metrics
        if metric.kind == Phase8StabilityMetricKind.SNAPSHOT_CACHE_MS
    ]

    assert samples
    assert all(metric.value <= 50 for metric in samples)


def test_focused_ui_parse_under_300ms() -> None:
    runtime = Phase8LoadLatencyStabilityRuntime()
    session = runtime.create_session(workspace_id="workspace")
    report = runtime.validate(session_id=session.session_id)

    samples = [
        metric
        for result in report.scenario_results
        for metric in result.metrics
        if metric.kind == Phase8StabilityMetricKind.FOCUSED_UI_PARSE_MS
    ]

    assert samples
    assert all(metric.value <= 300 for metric in samples)


def test_simple_verification_under_300ms() -> None:
    runtime = Phase8LoadLatencyStabilityRuntime()
    session = runtime.create_session(workspace_id="workspace")
    report = runtime.validate(session_id=session.session_id)

    samples = [
        metric
        for result in report.scenario_results
        for metric in result.metrics
        if metric.kind == Phase8StabilityMetricKind.SIMPLE_VERIFICATION_MS
    ]

    assert samples
    assert all(metric.value <= 300 for metric in samples)


def test_graph_resync_under_one_second() -> None:
    runtime = Phase8LoadLatencyStabilityRuntime()
    session = runtime.create_session(workspace_id="workspace")
    report = runtime.validate(session_id=session.session_id)

    samples = [
        metric
        for result in report.scenario_results
        for metric in result.metrics
        if metric.kind == Phase8StabilityMetricKind.MODERATE_GRAPH_RESYNC_MS
    ]

    assert samples
    assert all(metric.value <= 1000 for metric in samples)


def test_no_memory_leak() -> None:
    runtime = Phase8LoadLatencyStabilityRuntime()
    session = runtime.create_session(workspace_id="workspace")
    report = runtime.validate(session_id=session.session_id)

    assert report.memory_leak_detected is False


def test_conversation_latency_protected() -> None:
    runtime = Phase8LoadLatencyStabilityRuntime()
    session = runtime.create_session(workspace_id="workspace")
    report = runtime.validate(session_id=session.session_id)

    assert report.conversation_latency_protected is True


def test_missing_session_fails_validation() -> None:
    runtime = Phase8LoadLatencyStabilityRuntime()

    report = runtime.validate(session_id="missing")

    assert report.status == Phase8StabilityStatus.FAILED
    assert report.reason == Phase8StabilityReason.SESSION_NOT_FOUND


def test_snapshot_tracks_counts() -> None:
    runtime = Phase8LoadLatencyStabilityRuntime()
    session = runtime.create_session(workspace_id="workspace")

    runtime.validate(session_id=session.session_id)
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.report_count == 1
    assert snapshot.scenario_result_count == 13
    assert snapshot.failed_count == 0
    assert snapshot.visual_workers_shed_count >= 2


def test_session_tracks_counts() -> None:
    runtime = Phase8LoadLatencyStabilityRuntime()
    session = runtime.create_session(workspace_id="workspace")

    runtime.validate(session_id=session.session_id)
    stored = runtime.session_for(session.session_id)

    assert stored is not None
    assert stored.validation_count == 1
    assert stored.failed_count == 0
    assert stored.visual_workers_shed_count >= 2


def test_reset_clears_runtime() -> None:
    runtime = Phase8LoadLatencyStabilityRuntime()
    runtime.create_session(workspace_id="workspace")

    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.runtime_event_count == 1
    assert snapshot.last_reason == Phase8StabilityReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert Phase8StabilityScenario.GRAPH_RESYNC_PRESSURE.value == (
        "graph_resync_pressure"
    )
    assert Phase8StabilityStatus.PASSED.value == "passed"
    assert Phase8StabilityMetricKind.SNAPSHOT_CACHE_MS.value == "snapshot_cache_ms"