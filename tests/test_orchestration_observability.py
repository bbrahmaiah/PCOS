from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.orchestration import (
    BackgroundTaskReason,
    BackgroundTaskRuntimeSnapshot,
    BottleneckDetector,
    BottleneckKind,
    BottleneckSeverity,
    CircuitBreakerReason,
    CircuitBreakerRuntimeSnapshot,
    DeadlockDetectionReason,
    DeadlockDetectorSnapshot,
    InterruptPropagationReason,
    InterruptPropagatorSnapshot,
    ObservabilityReason,
    OrchestrationHealth,
    OrchestrationObservabilityConfig,
    OrchestrationObservabilityRuntime,
    ResourceBudgetRuntimeSnapshot,
    ResourceUtilizationView,
    RuntimeMetricSample,
    TaskGraphView,
    TaskScheduleDecision,
    TaskScheduleReason,
    TaskSchedulerSnapshot,
    WorkerHealthView,
)


def scheduler_snapshot(
    *,
    scheduled_count: int = 1,
    deferred_count: int = 0,
) -> TaskSchedulerSnapshot:
    return TaskSchedulerSnapshot(
        name="scheduler",
        scheduled_count=scheduled_count,
        deferred_count=deferred_count,
        denied_count=0,
        skipped_count=0,
        active_assignment_count=scheduled_count,
        last_decision=TaskScheduleDecision.SCHEDULED,
        last_reason=TaskScheduleReason.TASK_SCHEDULED,
    )


def budget_snapshot(
    *,
    total_capacity: int = 100,
    total_reserved: int = 10,
) -> ResourceBudgetRuntimeSnapshot:
    return ResourceBudgetRuntimeSnapshot(
        name="budget",
        pool_count=2,
        reservation_count=1,
        total_capacity=total_capacity,
        total_reserved=total_reserved,
        evaluation_count=1,
        allow_count=1,
        warn_count=0,
        deny_count=0,
        last_decision=None,
        last_reason=None,
    )


def background_snapshot(
    *,
    scheduled_count: int = 1,
    yielded_count: int = 0,
) -> BackgroundTaskRuntimeSnapshot:
    return BackgroundTaskRuntimeSnapshot(
        name="background",
        registered_count=1,
        scheduled_count=scheduled_count,
        yielded_count=yielded_count,
        shed_count=0,
        cancelled_count=0,
        rejected_count=0,
        last_reason=BackgroundTaskReason.TASK_SCHEDULED,
    )


def interrupt_snapshot(*, dispatch_count: int = 0) -> InterruptPropagatorSnapshot:
    return InterruptPropagatorSnapshot(
        name="interrupts",
        active_interrupt_count=0,
        completed_count=0,
        escalated_count=0,
        rejected_count=0,
        dispatch_count=dispatch_count,
        acknowledgement_count=0,
        last_reason=InterruptPropagationReason.INTERRUPT_COMPLETED,
    )


def deadlock_snapshot(*, detected_count: int = 0) -> DeadlockDetectorSnapshot:
    return DeadlockDetectorSnapshot(
        name="deadlocks",
        wait_edge_count=0,
        detected_count=detected_count,
        resolved_count=0,
        rejected_count=0,
        timeout_count=0,
        last_reason=DeadlockDetectionReason.NO_DEADLOCK,
    )


def circuit_snapshot(*, open_count: int = 0) -> CircuitBreakerRuntimeSnapshot:
    return CircuitBreakerRuntimeSnapshot(
        name="breakers",
        breaker_count=open_count,
        open_count=open_count,
        half_open_count=0,
        closed_count=0,
        failure_count=open_count,
        fallback_count=0,
        rejected_count=0,
        last_reason=CircuitBreakerReason.WORKER_ALLOWED,
    )


def worker_view(*, utilization_percent: int = 10) -> WorkerHealthView:
    return WorkerHealthView(
        total_workers=2,
        healthy_workers=2,
        degraded_workers=0,
        unhealthy_workers=0,
        active_tasks=1,
        queued_tasks=0,
        utilization_percent=utilization_percent,
    )


def test_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        OrchestrationObservabilityConfig(name=" ").validate()


def test_worker_view_rejects_invalid_counts() -> None:
    with pytest.raises(ValidationError):
        WorkerHealthView(
            total_workers=1,
            healthy_workers=1,
            degraded_workers=1,
            unhealthy_workers=0,
        )


def test_resource_view_from_budget_snapshot() -> None:
    view = ResourceUtilizationView.from_budget_snapshot(
        budget_snapshot(total_capacity=100, total_reserved=85)
    )

    assert view.utilization_percent == 85
    assert view.warning is True


def test_task_graph_completion_percent() -> None:
    view = TaskGraphView(
        job_id="job-1",
        total_tasks=4,
        completed_tasks=2,
    )

    assert view.completion_percent == 50


def test_metric_sample_validates_percentages() -> None:
    with pytest.raises(ValidationError):
        RuntimeMetricSample(worker_utilization_percent=101)


def test_bottleneck_detector_reports_no_bottleneck() -> None:
    detector = BottleneckDetector()

    reports = detector.detect(
        scheduler=scheduler_snapshot(),
        resources=ResourceUtilizationView.from_budget_snapshot(budget_snapshot()),
        workers=worker_view(),
    )

    assert len(reports) == 1
    assert reports[0].kind == BottleneckKind.NONE
    assert reports[0].severity == BottleneckSeverity.INFO


def test_detector_detects_resource_bottleneck() -> None:
    detector = BottleneckDetector()

    reports = detector.detect(
        resources=ResourceUtilizationView.from_budget_snapshot(
            budget_snapshot(total_capacity=100, total_reserved=90)
        )
    )

    assert any(report.kind == BottleneckKind.RESOURCE for report in reports)


def test_detector_detects_worker_bottleneck() -> None:
    detector = BottleneckDetector()

    reports = detector.detect(workers=worker_view(utilization_percent=90))

    assert any(report.kind == BottleneckKind.WORKER for report in reports)


def test_detector_detects_scheduler_bottleneck() -> None:
    detector = BottleneckDetector()

    reports = detector.detect(scheduler=scheduler_snapshot(deferred_count=20))

    assert any(report.kind == BottleneckKind.SCHEDULER for report in reports)


def test_detector_detects_interrupt_bottleneck() -> None:
    detector = BottleneckDetector()

    reports = detector.detect(interrupts=interrupt_snapshot(dispatch_count=10))

    assert any(report.kind == BottleneckKind.INTERRUPT for report in reports)


def test_detector_detects_deadlock_bottleneck() -> None:
    detector = BottleneckDetector()

    reports = detector.detect(deadlocks=deadlock_snapshot(detected_count=1))

    assert any(report.kind == BottleneckKind.DEADLOCK for report in reports)


def test_detector_detects_circuit_breaker_bottleneck() -> None:
    detector = BottleneckDetector()

    reports = detector.detect(circuit_breakers=circuit_snapshot(open_count=1))

    assert any(
        report.kind == BottleneckKind.CIRCUIT_BREAKER for report in reports
    )


def test_detector_detects_background_bottleneck() -> None:
    detector = BottleneckDetector()

    reports = detector.detect(
        background=background_snapshot(scheduled_count=1, yielded_count=3)
    )

    assert any(report.kind == BottleneckKind.BACKGROUND for report in reports)


def test_build_dashboard_healthy() -> None:
    runtime = OrchestrationObservabilityRuntime()

    result = runtime.build_dashboard(
        scheduler=scheduler_snapshot(),
        budget=budget_snapshot(),
        workers=worker_view(),
        background=background_snapshot(),
        interrupts=interrupt_snapshot(),
        deadlocks=deadlock_snapshot(),
        circuit_breakers=circuit_snapshot(),
    )

    assert result.success is True
    assert result.reason == ObservabilityReason.DASHBOARD_CREATED
    assert result.dashboard is not None
    assert result.dashboard.health == OrchestrationHealth.HEALTHY
    assert result.dashboard.metrics.worker_utilization_percent == 10


def test_build_dashboard_critical_from_deadlock() -> None:
    runtime = OrchestrationObservabilityRuntime()

    result = runtime.build_dashboard(deadlocks=deadlock_snapshot(detected_count=1))

    assert result.dashboard is not None
    assert result.dashboard.health == OrchestrationHealth.CRITICAL


def test_latest_dashboard_returns_most_recent() -> None:
    runtime = OrchestrationObservabilityRuntime()

    runtime.build_dashboard(workers=worker_view(utilization_percent=10))
    runtime.build_dashboard(workers=worker_view(utilization_percent=90))

    latest = runtime.latest_dashboard()

    assert latest is not None
    assert latest.health == OrchestrationHealth.DEGRADED


def test_snapshot_counts_dashboard_state() -> None:
    runtime = OrchestrationObservabilityRuntime()

    runtime.build_dashboard(deadlocks=deadlock_snapshot(detected_count=1))
    snapshot = runtime.snapshot()

    assert snapshot.dashboard_count == 1
    assert snapshot.bottleneck_count == 1
    assert snapshot.last_health == OrchestrationHealth.CRITICAL


def test_reset_clears_runtime_state() -> None:
    runtime = OrchestrationObservabilityRuntime()

    runtime.build_dashboard()
    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.dashboard_count == 0
    assert snapshot.last_reason == ObservabilityReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert OrchestrationHealth.HEALTHY.value == "healthy"
    assert BottleneckKind.RESOURCE.value == "resource"
    assert BottleneckSeverity.CRITICAL.value == "critical"
    assert ObservabilityReason.DASHBOARD_CREATED.value == "dashboard_created"