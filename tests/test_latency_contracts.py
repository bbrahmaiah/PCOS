from __future__ import annotations

import time

import pytest
from pydantic import ValidationError

from jarvis.latency import (
    LatencyBudget,
    LatencyBudgetRegistry,
    LatencyBudgetRegistryConfig,
    LatencyMeasurement,
    LatencyMeasurementRuntime,
    LatencyMeasurementRuntimeConfig,
    LatencyOperation,
    LatencyPercentile,
    LatencySeverity,
    LatencySpan,
    LatencySubsystem,
    LatencyTarget,
    LatencyTimer,
    PercentileTracker,
    PercentileTrackerConfig,
    default_latency_budgets,
)


def measurement(
    *,
    operation: LatencyOperation = LatencyOperation.STT_FIRST_TOKEN,
    subsystem: LatencySubsystem = LatencySubsystem.PRESENCE,
    duration_ms: float = 100.0,
) -> LatencyMeasurement:
    return LatencyMeasurement(
        operation=operation,
        subsystem=subsystem,
        duration_ms=duration_ms,
    )


def target(
    *,
    operation: LatencyOperation = LatencyOperation.STT_FIRST_TOKEN,
    subsystem: LatencySubsystem = LatencySubsystem.PRESENCE,
    target_ms: int = 100,
    warning_ms: int = 150,
    max_ms: int = 200,
) -> LatencyTarget:
    return LatencyTarget(
        operation=operation,
        subsystem=subsystem,
        percentile=LatencyPercentile.P95,
        target_ms=target_ms,
        warning_ms=warning_ms,
        max_ms=max_ms,
        description="test target",
    )


def budget() -> LatencyBudget:
    return LatencyBudget(
        operation=LatencyOperation.STT_FIRST_TOKEN,
        subsystem=LatencySubsystem.PRESENCE,
        owner="presence_runtime",
        targets=(target(),),
    )


def test_target_rejects_empty_description() -> None:
    with pytest.raises(ValidationError):
        LatencyTarget(
            operation=LatencyOperation.STT_FIRST_TOKEN,
            subsystem=LatencySubsystem.PRESENCE,
            percentile=LatencyPercentile.P95,
            target_ms=100,
            warning_ms=150,
            max_ms=200,
            description=" ",
        )


def test_target_requires_ordered_thresholds() -> None:
    with pytest.raises(ValidationError):
        target(target_ms=150, warning_ms=100, max_ms=200)

    with pytest.raises(ValidationError):
        target(target_ms=100, warning_ms=200, max_ms=150)


def test_budget_requires_targets() -> None:
    with pytest.raises(ValidationError):
        LatencyBudget(
            operation=LatencyOperation.STT_FIRST_TOKEN,
            subsystem=LatencySubsystem.PRESENCE,
            owner="presence_runtime",
            targets=(),
        )


def test_budget_rejects_mismatched_target_operation() -> None:
    wrong = target(operation=LatencyOperation.MEMORY_RETRIEVAL)

    with pytest.raises(ValidationError):
        LatencyBudget(
            operation=LatencyOperation.STT_FIRST_TOKEN,
            subsystem=LatencySubsystem.PRESENCE,
            owner="presence_runtime",
            targets=(wrong,),
        )


def test_budget_rejects_mismatched_target_subsystem() -> None:
    wrong = target(subsystem=LatencySubsystem.MEMORY)

    with pytest.raises(ValidationError):
        LatencyBudget(
            operation=LatencyOperation.STT_FIRST_TOKEN,
            subsystem=LatencySubsystem.PRESENCE,
            owner="presence_runtime",
            targets=(wrong,),
        )


def test_budget_target_lookup() -> None:
    item = budget()

    assert item.target_for(LatencyPercentile.P95) is not None
    assert item.target_for(LatencyPercentile.P99) is None


def test_default_budgets_cover_critical_operations() -> None:
    operations = {item.operation for item in default_latency_budgets()}

    assert LatencyOperation.STT_FIRST_TOKEN in operations
    assert LatencyOperation.STT_FINALIZATION in operations
    assert LatencyOperation.MEMORY_RETRIEVAL in operations
    assert LatencyOperation.CONTEXT_BUILD in operations
    assert LatencyOperation.LLM_FIRST_TOKEN in operations
    assert LatencyOperation.LLM_FULL_RESPONSE in operations
    assert LatencyOperation.TTS_FIRST_AUDIO in operations
    assert LatencyOperation.PLAYBACK_STARTUP in operations
    assert LatencyOperation.INTERRUPT_RESPONSE in operations
    assert LatencyOperation.RECOVERY_RECONSTRUCT in operations


def test_registry_registers_defaults() -> None:
    registry = LatencyBudgetRegistry()

    assert len(registry.budgets()) >= 10


def test_registry_can_disable_defaults() -> None:
    registry = LatencyBudgetRegistry(
        config=LatencyBudgetRegistryConfig(register_defaults=False)
    )

    assert registry.budgets() == ()


def test_registry_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        LatencyBudgetRegistryConfig(name=" ").validate()


def test_registry_evaluates_ok_latency() -> None:
    registry = LatencyBudgetRegistry(
        config=LatencyBudgetRegistryConfig(register_defaults=False),
        budgets=(budget(),),
    )

    result = registry.evaluate(measurement(duration_ms=90))

    assert result.severity == LatencySeverity.OK
    assert result.within_target is True
    assert result.within_max is True


def test_registry_evaluates_warning_latency() -> None:
    registry = LatencyBudgetRegistry(
        config=LatencyBudgetRegistryConfig(register_defaults=False),
        budgets=(budget(),),
    )

    result = registry.evaluate(measurement(duration_ms=125))

    assert result.severity == LatencySeverity.WARNING
    assert result.within_target is False
    assert result.within_max is True
    assert len(registry.violations()) == 1


def test_registry_evaluates_violation_latency() -> None:
    registry = LatencyBudgetRegistry(
        config=LatencyBudgetRegistryConfig(register_defaults=False),
        budgets=(budget(),),
    )

    result = registry.evaluate(measurement(duration_ms=175))

    assert result.severity == LatencySeverity.VIOLATION
    assert result.within_target is False
    assert result.within_max is True
    assert len(registry.violations()) == 1


def test_registry_evaluates_critical_latency() -> None:
    registry = LatencyBudgetRegistry(
        config=LatencyBudgetRegistryConfig(register_defaults=False),
        budgets=(budget(),),
    )

    result = registry.evaluate(measurement(duration_ms=250))

    assert result.severity == LatencySeverity.CRITICAL
    assert result.within_target is False
    assert result.within_max is False
    assert len(registry.violations()) == 1


def test_registry_rejects_missing_budget() -> None:
    registry = LatencyBudgetRegistry(
        config=LatencyBudgetRegistryConfig(register_defaults=False)
    )

    with pytest.raises(ValueError):
        registry.evaluate(measurement())


def test_measurement_rejects_negative_duration() -> None:
    with pytest.raises(ValidationError):
        LatencyMeasurement(
            operation=LatencyOperation.STT_FIRST_TOKEN,
            subsystem=LatencySubsystem.PRESENCE,
            duration_ms=-1,
        )


def test_span_duration_and_measurement_conversion() -> None:
    span = LatencySpan(
        operation=LatencyOperation.STT_FIRST_TOKEN,
        subsystem=LatencySubsystem.PRESENCE,
        start_ns=1_000_000,
        end_ns=3_000_000,
    )

    assert span.duration_ms() == 2.0
    assert span.to_measurement().duration_ms == 2.0


def test_span_rejects_invalid_time_order() -> None:
    with pytest.raises(ValidationError):
        LatencySpan(
            operation=LatencyOperation.STT_FIRST_TOKEN,
            subsystem=LatencySubsystem.PRESENCE,
            start_ns=10,
            end_ns=5,
        )


def test_percentile_tracker_rejects_invalid_config() -> None:
    with pytest.raises(ValueError):
        PercentileTrackerConfig(max_samples_per_operation=0).validate()


def test_percentile_tracker_empty_snapshot() -> None:
    tracker = PercentileTracker()

    snapshot = tracker.snapshot_for(
        operation=LatencyOperation.STT_FIRST_TOKEN,
        subsystem=LatencySubsystem.PRESENCE,
    )

    assert snapshot.sample_count == 0
    assert snapshot.p50_ms == 0.0
    assert snapshot.worst_ms == 0.0


def test_percentile_tracker_records_samples() -> None:
    tracker = PercentileTracker()

    for value in (10.0, 20.0, 30.0, 40.0, 50.0):
        tracker.record(measurement(duration_ms=value))

    snapshot = tracker.snapshot_for(
        operation=LatencyOperation.STT_FIRST_TOKEN,
        subsystem=LatencySubsystem.PRESENCE,
    )

    assert snapshot.sample_count == 5
    assert snapshot.p50_ms == 30.0
    assert snapshot.worst_ms == 50.0


def test_percentile_tracker_limits_samples() -> None:
    tracker = PercentileTracker(
        config=PercentileTrackerConfig(max_samples_per_operation=3)
    )

    for value in (10.0, 20.0, 30.0, 40.0):
        tracker.record(measurement(duration_ms=value))

    assert tracker.sample_count(
        operation=LatencyOperation.STT_FIRST_TOKEN,
        subsystem=LatencySubsystem.PRESENCE,
    ) == 3


def test_measurement_runtime_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        LatencyMeasurementRuntimeConfig(name=" ").validate()


def test_measurement_runtime_records_ok_measurement() -> None:
    runtime = LatencyMeasurementRuntime()

    result = runtime.record(
        measurement(
            operation=LatencyOperation.STT_FIRST_TOKEN,
            subsystem=LatencySubsystem.PRESENCE,
            duration_ms=100,
        )
    )

    assert result is not None
    assert result.severity == LatencySeverity.OK
    assert runtime.snapshot().measurement_count == 1


def test_measurement_runtime_records_violation_event() -> None:
    runtime = LatencyMeasurementRuntime()

    result = runtime.record(
        measurement(
            operation=LatencyOperation.STT_FIRST_TOKEN,
            subsystem=LatencySubsystem.PRESENCE,
            duration_ms=10_000,
        )
    )

    assert result is not None
    assert result.severity == LatencySeverity.CRITICAL
    assert runtime.snapshot().critical_count == 1
    assert len(runtime.events()) == 2


def test_measurement_runtime_can_disable_budget_evaluation() -> None:
    runtime = LatencyMeasurementRuntime(
        config=LatencyMeasurementRuntimeConfig(evaluate_budgets=False)
    )

    result = runtime.record(measurement(duration_ms=10_000))

    assert result is None
    assert runtime.snapshot().measurement_count == 1
    assert runtime.snapshot().critical_count == 0


def test_measurement_runtime_records_span() -> None:
    runtime = LatencyMeasurementRuntime()
    span = LatencySpan(
        operation=LatencyOperation.STT_FIRST_TOKEN,
        subsystem=LatencySubsystem.PRESENCE,
        start_ns=1_000_000,
        end_ns=51_000_000,
    )

    result = runtime.record_span(span)

    assert result is not None
    assert result.duration_ms == 50.0
    assert runtime.measurements()[0].span_id == span.span_id


def test_latency_timer_records_measurement() -> None:
    runtime = LatencyMeasurementRuntime()

    with LatencyTimer(
        runtime=runtime,
        operation=LatencyOperation.STT_FIRST_TOKEN,
        subsystem=LatencySubsystem.PRESENCE,
    ) as timer:
        time.sleep(0.001)

    assert timer.span is not None
    assert runtime.snapshot().measurement_count == 1


def test_measurement_runtime_reset_clears_state() -> None:
    runtime = LatencyMeasurementRuntime()

    runtime.record(measurement(duration_ms=10_000))
    runtime.reset()

    snapshot = runtime.snapshot()

    assert snapshot.measurement_count == 0
    assert snapshot.critical_count == 0
    assert snapshot.percentile_count == 0


def test_enum_values_are_stable() -> None:
    assert LatencySubsystem.PRESENCE.value == "presence"
    assert LatencyOperation.STT_FIRST_TOKEN.value == "stt_first_token"
    assert LatencyPercentile.P95.value == 95
    assert LatencySeverity.CRITICAL.value == "critical"