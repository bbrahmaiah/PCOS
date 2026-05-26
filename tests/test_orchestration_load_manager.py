from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.orchestration import (
    BottleneckKind,
    BottleneckReport,
    BottleneckSeverity,
    CognitiveLoadLevel,
    CognitiveLoadManagerConfig,
    CognitiveLoadManagerRuntime,
    CognitiveLoadMonitor,
    LoadDecisionReason,
    LoadPolicy,
    LoadSheddingAction,
    LoadSheddingDecision,
    LoadSheddingStrategy,
    LoadSheddingTarget,
    LoadSignal,
    ObservabilityReason,
    OrchestrationDashboard,
    OrchestrationHealth,
    RuntimeMetricSample,
)


def signal(
    *,
    worker: int = 10,
    resource: int = 10,
    queue_depth: int = 0,
    interrupts: int = 0,
    bottlenecks: int = 0,
    critical_bottlenecks: int = 0,
    health: OrchestrationHealth = OrchestrationHealth.HEALTHY,
) -> LoadSignal:
    return LoadSignal(
        worker_utilization_percent=worker,
        resource_utilization_percent=resource,
        queue_depth=queue_depth,
        interrupt_frequency=interrupts,
        bottleneck_count=bottlenecks,
        critical_bottleneck_count=critical_bottlenecks,
        dashboard_health=health,
    )


def dashboard(
    *,
    worker: int = 10,
    resource: int = 10,
    queue_depth: int = 0,
    interrupts: int = 0,
    health: OrchestrationHealth = OrchestrationHealth.HEALTHY,
    bottlenecks: tuple[BottleneckReport, ...] = (),
) -> OrchestrationDashboard:
    return OrchestrationDashboard(
        health=health,
        summary="test dashboard",
        metrics=RuntimeMetricSample(
            worker_utilization_percent=worker,
            budget_consumption_percent=resource,
            queue_depth=queue_depth,
            interrupt_frequency=interrupts,
        ),
        bottlenecks=bottlenecks,
    )


def critical_bottleneck() -> BottleneckReport:
    return BottleneckReport(
        kind=BottleneckKind.DEADLOCK,
        severity=BottleneckSeverity.CRITICAL,
        reason=ObservabilityReason.DEADLOCK_BOTTLENECK_DETECTED,
        message="deadlock detected",
    )


def test_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        CognitiveLoadManagerConfig(name=" ").validate()


def test_config_rejects_non_monotonic_thresholds() -> None:
    config = CognitiveLoadManagerConfig(
        elevated_threshold_percent=70,
        high_threshold_percent=60,
    )

    with pytest.raises(ValueError):
        config.validate()


def test_signal_pressure_uses_maximum_pressure_source() -> None:
    item = signal(worker=20, resource=30, queue_depth=12)

    assert item.pressure_percent == 60


def test_policy_classifies_normal() -> None:
    policy = LoadPolicy()

    level, reason = policy.classify(signal(worker=10))

    assert level == CognitiveLoadLevel.NORMAL
    assert reason == LoadDecisionReason.LOAD_NORMAL


def test_policy_classifies_elevated() -> None:
    policy = LoadPolicy()

    level, reason = policy.classify(signal(worker=65))

    assert level == CognitiveLoadLevel.ELEVATED
    assert reason == LoadDecisionReason.LOAD_ELEVATED


def test_policy_classifies_high() -> None:
    policy = LoadPolicy()

    level, reason = policy.classify(signal(worker=78))

    assert level == CognitiveLoadLevel.HIGH
    assert reason == LoadDecisionReason.LOAD_HIGH


def test_policy_classifies_critical() -> None:
    policy = LoadPolicy()

    level, reason = policy.classify(signal(resource=90))

    assert level == CognitiveLoadLevel.CRITICAL
    assert reason == LoadDecisionReason.LOAD_CRITICAL


def test_policy_classifies_shedding() -> None:
    policy = LoadPolicy()

    level, reason = policy.classify(signal(worker=96))

    assert level == CognitiveLoadLevel.SHEDDING
    assert reason == LoadDecisionReason.LOAD_SHEDDING


def test_policy_critical_bottleneck_forces_critical() -> None:
    policy = LoadPolicy()

    level, reason = policy.classify(
        signal(worker=10, critical_bottlenecks=1)
    )

    assert level == CognitiveLoadLevel.CRITICAL
    assert reason == LoadDecisionReason.LOAD_CRITICAL


def test_policy_hysteresis_holds_previous_level() -> None:
    policy = LoadPolicy()

    level, reason = policy.classify(
        signal(worker=70),
        current_level=CognitiveLoadLevel.HIGH,
    )

    assert level == CognitiveLoadLevel.HIGH
    assert reason == LoadDecisionReason.HYSTERESIS_HOLD


def test_policy_hysteresis_allows_recovery_after_twenty_percent_drop() -> None:
    policy = LoadPolicy()

    level, reason = policy.classify(
        signal(worker=50),
        current_level=CognitiveLoadLevel.HIGH,
    )

    assert level == CognitiveLoadLevel.NORMAL
    assert reason == LoadDecisionReason.LOAD_NORMAL


def test_conversation_cannot_be_shed() -> None:
    with pytest.raises(ValidationError):
        LoadSheddingDecision(
            target=LoadSheddingTarget.CONVERSATION,
            action=LoadSheddingAction.SHED,
            reason=LoadDecisionReason.CONVERSATION_PROTECTED,
            allowed=True,
            message="bad decision",
        )


def test_strategy_normal_only_protects_conversation() -> None:
    strategy = LoadSheddingStrategy()

    decisions = strategy.decisions_for(CognitiveLoadLevel.NORMAL)

    assert len(decisions) == 1
    assert decisions[0].target == LoadSheddingTarget.CONVERSATION
    assert decisions[0].action == LoadSheddingAction.PROTECT


def test_strategy_high_sheds_background_first() -> None:
    strategy = LoadSheddingStrategy()

    decisions = strategy.decisions_for(CognitiveLoadLevel.HIGH)

    targets = [item.target for item in decisions]

    assert targets[0] == LoadSheddingTarget.CONVERSATION
    assert LoadSheddingTarget.BACKGROUND_MAINTENANCE in targets
    assert LoadSheddingTarget.BACKGROUND_PREFETCH in targets
    assert LoadSheddingTarget.TOOL_PLANNING not in targets


def test_strategy_critical_compresses_non_critical_memory() -> None:
    strategy = LoadSheddingStrategy()

    decisions = strategy.decisions_for(CognitiveLoadLevel.CRITICAL)

    assert any(
        item.target == LoadSheddingTarget.NON_CRITICAL_MEMORY
        and item.action == LoadSheddingAction.COMPRESS
        for item in decisions
    )


def test_strategy_shedding_defers_tool_planning() -> None:
    strategy = LoadSheddingStrategy()

    decisions = strategy.decisions_for(CognitiveLoadLevel.SHEDDING)

    assert any(
        item.target == LoadSheddingTarget.TOOL_PLANNING
        and item.action == LoadSheddingAction.DEFER
        for item in decisions
    )


def test_monitor_debug_event_for_high_load() -> None:
    monitor = CognitiveLoadMonitor(
        config=CognitiveLoadManagerConfig(debug_mode=True)
    )

    assessment = monitor.assess(signal(worker=80))

    assert assessment.level == CognitiveLoadLevel.HIGH
    assert assessment.user_visible_events == (
        "JARVIS is under load, background tasks paused",
    )


def test_monitor_no_debug_event_when_debug_disabled() -> None:
    monitor = CognitiveLoadMonitor(
        config=CognitiveLoadManagerConfig(debug_mode=False)
    )

    assessment = monitor.assess(signal(worker=80))

    assert assessment.user_visible_events == ()


def test_signal_from_dashboard_reads_metrics_and_bottlenecks() -> None:
    item = LoadSignal.from_dashboard(
        dashboard(
            worker=70,
            resource=80,
            queue_depth=3,
            interrupts=2,
            health=OrchestrationHealth.CRITICAL,
            bottlenecks=(critical_bottleneck(),),
        )
    )

    assert item.worker_utilization_percent == 70
    assert item.resource_utilization_percent == 80
    assert item.critical_bottleneck_count == 1
    assert item.dashboard_health == OrchestrationHealth.CRITICAL


def test_runtime_records_signal() -> None:
    runtime = CognitiveLoadManagerRuntime()

    result = runtime.record_signal(signal(worker=80))

    assert result.success is True
    assert result.assessment is not None
    assert result.assessment.level == CognitiveLoadLevel.HIGH
    assert runtime.latest_assessment() == result.assessment


def test_runtime_records_dashboard() -> None:
    runtime = CognitiveLoadManagerRuntime()

    result = runtime.record_dashboard(dashboard(worker=96))

    assert result.success is True
    assert result.reason == LoadDecisionReason.DASHBOARD_RECORDED
    assert result.assessment is not None
    assert result.assessment.level == CognitiveLoadLevel.SHEDDING


def test_runtime_snapshot_tracks_latest_state() -> None:
    runtime = CognitiveLoadManagerRuntime(
        config=CognitiveLoadManagerConfig(debug_mode=True)
    )

    runtime.record_signal(signal(worker=80))
    snapshot = runtime.snapshot()

    assert snapshot.assessment_count == 1
    assert snapshot.last_level == CognitiveLoadLevel.HIGH
    assert snapshot.last_pressure_percent == 80
    assert snapshot.debug_mode is True


def test_runtime_reset_clears_state() -> None:
    runtime = CognitiveLoadManagerRuntime()

    runtime.record_signal(signal(worker=80))
    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.assessment_count == 0
    assert snapshot.last_level is None
    assert snapshot.last_reason == LoadDecisionReason.RUNTIME_RESET


def test_runtime_hysteresis_uses_previous_runtime_level() -> None:
    runtime = CognitiveLoadManagerRuntime()

    runtime.record_signal(signal(worker=80))
    second = runtime.record_signal(signal(worker=70))

    assert second.assessment is not None
    assert second.assessment.level == CognitiveLoadLevel.HIGH
    assert second.assessment.reason == LoadDecisionReason.HYSTERESIS_HOLD


def test_assessment_should_shed_when_high() -> None:
    monitor = CognitiveLoadMonitor()

    assessment = monitor.assess(signal(worker=80))

    assert assessment.should_shed is True
    assert assessment.conversation_protected is True


def test_enum_values_are_stable() -> None:
    assert CognitiveLoadLevel.NORMAL.name == "NORMAL"
    assert LoadSheddingTarget.CONVERSATION.value == "conversation"
    assert LoadSheddingAction.PROTECT.value == "protect"
    assert LoadDecisionReason.HYSTERESIS_HOLD.value == "hysteresis_hold"