from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import RLock

from pydantic import Field, field_validator, model_validator

from jarvis.orchestration.background import BackgroundTaskRuntimeSnapshot
from jarvis.orchestration.budgets import ResourceBudgetRuntimeSnapshot
from jarvis.orchestration.circuit_breakers import CircuitBreakerRuntimeSnapshot
from jarvis.orchestration.deadlocks import DeadlockDetectorSnapshot
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.interrupts import InterruptPropagatorSnapshot
from jarvis.orchestration.models import OrchestrationModel
from jarvis.orchestration.scheduler import TaskSchedulerSnapshot
from jarvis.orchestration.state_machine import OrchestrationStateMachineSnapshot


class OrchestrationHealth(StrEnum):
    """
    High-level orchestration health classification.
    """

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class BottleneckKind(StrEnum):
    """
    Bottleneck classes detected by observability.
    """

    NONE = "none"
    WORKER = "worker"
    RESOURCE = "resource"
    SCHEDULER = "scheduler"
    INTERRUPT = "interrupt"
    DEADLOCK = "deadlock"
    CIRCUIT_BREAKER = "circuit_breaker"
    BACKGROUND = "background"


class BottleneckSeverity(StrEnum):
    """
    Bottleneck severity.
    """

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ObservabilityReason(StrEnum):
    """
    Machine-readable observability reason.
    """

    DASHBOARD_CREATED = "dashboard_created"
    METRICS_RECORDED = "metrics_recorded"
    BOTTLENECK_NOT_DETECTED = "bottleneck_not_detected"
    WORKER_BOTTLENECK_DETECTED = "worker_bottleneck_detected"
    RESOURCE_BOTTLENECK_DETECTED = "resource_bottleneck_detected"
    SCHEDULER_BOTTLENECK_DETECTED = "scheduler_bottleneck_detected"
    INTERRUPT_BOTTLENECK_DETECTED = "interrupt_bottleneck_detected"
    DEADLOCK_BOTTLENECK_DETECTED = "deadlock_bottleneck_detected"
    CIRCUIT_BREAKER_BOTTLENECK_DETECTED = "circuit_breaker_bottleneck_detected"
    BACKGROUND_BOTTLENECK_DETECTED = "background_bottleneck_detected"
    RUNTIME_RESET = "runtime_reset"


class WorkerHealthView(OrchestrationModel):
    """
    Queryable worker health view.

    This view is intentionally independent from worker execution.
    """

    total_workers: int = Field(default=0, ge=0)
    healthy_workers: int = Field(default=0, ge=0)
    degraded_workers: int = Field(default=0, ge=0)
    unhealthy_workers: int = Field(default=0, ge=0)
    active_tasks: int = Field(default=0, ge=0)
    queued_tasks: int = Field(default=0, ge=0)
    utilization_percent: int = Field(default=0, ge=0, le=100)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_worker_counts(self) -> WorkerHealthView:
        counted = (
            self.healthy_workers
            + self.degraded_workers
            + self.unhealthy_workers
        )

        if counted > self.total_workers:
            raise ValueError("worker health counts cannot exceed total workers.")

        return self


class ResourceUtilizationView(OrchestrationModel):
    """
    Queryable resource utilization view.
    """

    pool_count: int = Field(default=0, ge=0)
    total_capacity: int = Field(default=0, ge=0)
    total_reserved: int = Field(default=0, ge=0)
    utilization_percent: int = Field(default=0, ge=0, le=100)
    warning: bool = False
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_reserved(self) -> ResourceUtilizationView:
        if self.total_reserved > self.total_capacity and self.total_capacity > 0:
            raise ValueError("reserved resources cannot exceed total capacity.")

        return self

    @classmethod
    def from_budget_snapshot(
        cls,
        snapshot: ResourceBudgetRuntimeSnapshot,
    ) -> ResourceUtilizationView:
        percent = 0

        if snapshot.total_capacity > 0:
            percent = int(
                (snapshot.total_reserved / snapshot.total_capacity) * 100
            )

        return cls(
            pool_count=snapshot.pool_count,
            total_capacity=snapshot.total_capacity,
            total_reserved=snapshot.total_reserved,
            utilization_percent=min(100, percent),
            warning=percent >= 80,
        )


class TaskGraphView(OrchestrationModel):
    """
    Queryable task graph view.
    """

    job_id: str
    total_tasks: int = Field(ge=0)
    ready_tasks: int = Field(default=0, ge=0)
    scheduled_tasks: int = Field(default=0, ge=0)
    completed_tasks: int = Field(default=0, ge=0)
    failed_tasks: int = Field(default=0, ge=0)
    blocked_tasks: int = Field(default=0, ge=0)
    queue_depth: int = Field(default=0, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("job_id")
    @classmethod
    def _required_job_id(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("job_id cannot be empty.")

        return cleaned

    @property
    def completion_percent(self) -> int:
        if self.total_tasks <= 0:
            return 0

        return int((self.completed_tasks / self.total_tasks) * 100)


class CoordinationEventView(OrchestrationModel):
    """
    Queryable coordination event metrics.
    """

    message_count: int = Field(default=0, ge=0)
    assignment_count: int = Field(default=0, ge=0)
    result_count: int = Field(default=0, ge=0)
    progress_count: int = Field(default=0, ge=0)
    health_count: int = Field(default=0, ge=0)
    rejected_count: int = Field(default=0, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)


class RuntimeMetricSample(OrchestrationModel):
    """
    Runtime metric sample.

    These metrics are queryable at runtime and feed Step 14 later.
    """

    tasks_per_second: float = Field(default=0.0, ge=0.0)
    worker_utilization_percent: int = Field(default=0, ge=0, le=100)
    budget_consumption_percent: int = Field(default=0, ge=0, le=100)
    queue_depth: int = Field(default=0, ge=0)
    interrupt_frequency: int = Field(default=0, ge=0)
    background_completion_rate: float = Field(default=0.0, ge=0.0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)


class BottleneckReport(OrchestrationModel):
    """
    Bottleneck detector output.
    """

    kind: BottleneckKind
    severity: BottleneckSeverity
    reason: ObservabilityReason
    message: str
    constrained_component: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("message")
    @classmethod
    def _required_message(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("message cannot be empty.")

        return cleaned

    @property
    def detected(self) -> bool:
        return self.kind != BottleneckKind.NONE


class OrchestrationDashboard(OrchestrationModel):
    """
    Full queryable orchestration dashboard.

    This is runtime state visibility, not logs only.
    """

    health: OrchestrationHealth
    summary: str
    state_machine: OrchestrationStateMachineSnapshot | None = None
    scheduler: TaskSchedulerSnapshot | None = None
    resources: ResourceUtilizationView | None = None
    workers: WorkerHealthView | None = None
    task_graphs: tuple[TaskGraphView, ...] = ()
    coordination: CoordinationEventView | None = None
    background: BackgroundTaskRuntimeSnapshot | None = None
    interrupts: InterruptPropagatorSnapshot | None = None
    deadlocks: DeadlockDetectorSnapshot | None = None
    circuit_breakers: CircuitBreakerRuntimeSnapshot | None = None
    metrics: RuntimeMetricSample
    bottlenecks: tuple[BottleneckReport, ...] = ()
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("summary")
    @classmethod
    def _required_summary(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("summary cannot be empty.")

        return cleaned


class ObservabilityResult(OrchestrationModel):
    """
    Result of an observability operation.
    """

    reason: ObservabilityReason
    success: bool
    message: str
    dashboard: OrchestrationDashboard | None = None
    bottleneck: BottleneckReport | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("message")
    @classmethod
    def _required_message(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("message cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class OrchestrationObservabilityConfig:
    """
    Orchestration observability configuration.
    """

    name: str = "orchestration_observability"
    resource_warning_percent: int = 80
    worker_warning_percent: int = 80
    queue_warning_depth: int = 16
    interrupt_warning_count: int = 8

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if not 1 <= self.resource_warning_percent <= 100:
            raise ValueError("resource_warning_percent must be 1..100.")

        if not 1 <= self.worker_warning_percent <= 100:
            raise ValueError("worker_warning_percent must be 1..100.")

        if self.queue_warning_depth < 0:
            raise ValueError("queue_warning_depth cannot be negative.")

        if self.interrupt_warning_count < 0:
            raise ValueError("interrupt_warning_count cannot be negative.")


@dataclass(frozen=True, slots=True)
class OrchestrationObservabilitySnapshot:
    """
    Observability runtime diagnostics.
    """

    name: str
    dashboard_count: int
    bottleneck_count: int
    last_health: OrchestrationHealth | None
    last_reason: ObservabilityReason | None


class BottleneckDetector:
    """
    Runtime bottleneck detector.

    Identifies the most likely constraint in the orchestration system.
    """

    def __init__(
        self,
        *,
        config: OrchestrationObservabilityConfig | None = None,
    ) -> None:
        self._config = config or OrchestrationObservabilityConfig()
        self._config.validate()

    def detect(
        self,
        *,
        scheduler: TaskSchedulerSnapshot | None = None,
        resources: ResourceUtilizationView | None = None,
        workers: WorkerHealthView | None = None,
        background: BackgroundTaskRuntimeSnapshot | None = None,
        interrupts: InterruptPropagatorSnapshot | None = None,
        deadlocks: DeadlockDetectorSnapshot | None = None,
        circuit_breakers: CircuitBreakerRuntimeSnapshot | None = None,
    ) -> tuple[BottleneckReport, ...]:
        reports: list[BottleneckReport] = []

        if deadlocks is not None and deadlocks.detected_count > 0:
            reports.append(
                BottleneckReport(
                    kind=BottleneckKind.DEADLOCK,
                    severity=BottleneckSeverity.CRITICAL,
                    reason=ObservabilityReason.DEADLOCK_BOTTLENECK_DETECTED,
                    message="deadlock detector has active detected events",
                    constrained_component="deadlock_detector",
                    metadata={"detected_count": deadlocks.detected_count},
                )
            )

        if circuit_breakers is not None and circuit_breakers.open_count > 0:
            reports.append(
                BottleneckReport(
                    kind=BottleneckKind.CIRCUIT_BREAKER,
                    severity=BottleneckSeverity.CRITICAL,
                    reason=(
                        ObservabilityReason
                        .CIRCUIT_BREAKER_BOTTLENECK_DETECTED
                    ),
                    message="one or more worker circuit breakers are open",
                    constrained_component="circuit_breaker_runtime",
                    metadata={"open_count": circuit_breakers.open_count},
                )
            )

        if resources is not None:
            if resources.utilization_percent >= (
                self._config.resource_warning_percent
            ):
                reports.append(
                    BottleneckReport(
                        kind=BottleneckKind.RESOURCE,
                        severity=BottleneckSeverity.WARNING,
                        reason=(
                            ObservabilityReason
                            .RESOURCE_BOTTLENECK_DETECTED
                        ),
                        message="resource utilization is above warning threshold",
                        constrained_component="resource_budget_runtime",
                        metadata={
                            "utilization_percent": (
                                resources.utilization_percent
                            )
                        },
                    )
                )

        if workers is not None:
            if workers.utilization_percent >= self._config.worker_warning_percent:
                reports.append(
                    BottleneckReport(
                        kind=BottleneckKind.WORKER,
                        severity=BottleneckSeverity.WARNING,
                        reason=ObservabilityReason.WORKER_BOTTLENECK_DETECTED,
                        message="worker utilization is above warning threshold",
                        constrained_component="worker_pool",
                        metadata={
                            "utilization_percent": workers.utilization_percent
                        },
                    )
                )

        if scheduler is not None:
            if scheduler.deferred_count >= self._config.queue_warning_depth:
                reports.append(
                    BottleneckReport(
                        kind=BottleneckKind.SCHEDULER,
                        severity=BottleneckSeverity.WARNING,
                        reason=(
                            ObservabilityReason
                            .SCHEDULER_BOTTLENECK_DETECTED
                        ),
                        message="scheduler deferred queue is above threshold",
                        constrained_component="task_scheduler",
                        metadata={"deferred_count": scheduler.deferred_count},
                    )
                )

        if interrupts is not None:
            if interrupts.dispatch_count >= self._config.interrupt_warning_count:
                reports.append(
                    BottleneckReport(
                        kind=BottleneckKind.INTERRUPT,
                        severity=BottleneckSeverity.WARNING,
                        reason=(
                            ObservabilityReason
                            .INTERRUPT_BOTTLENECK_DETECTED
                        ),
                        message="interrupt frequency is above threshold",
                        constrained_component="interrupt_propagator",
                        metadata={"dispatch_count": interrupts.dispatch_count},
                    )
                )

        if background is not None:
            if background.yielded_count > background.scheduled_count:
                reports.append(
                    BottleneckReport(
                        kind=BottleneckKind.BACKGROUND,
                        severity=BottleneckSeverity.INFO,
                        reason=(
                            ObservabilityReason
                            .BACKGROUND_BOTTLENECK_DETECTED
                        ),
                        message="background tasks are yielding more than running",
                        constrained_component="background_task_runtime",
                        metadata={
                            "yielded_count": background.yielded_count,
                            "scheduled_count": background.scheduled_count,
                        },
                    )
                )

        if not reports:
            reports.append(
                BottleneckReport(
                    kind=BottleneckKind.NONE,
                    severity=BottleneckSeverity.INFO,
                    reason=ObservabilityReason.BOTTLENECK_NOT_DETECTED,
                    message="no bottleneck detected",
                )
            )

        return tuple(reports)


class OrchestrationObservabilityRuntime:
    """
    Phase 6 Orchestration Observability Runtime.

    Responsibilities:
    - build queryable dashboards
    - expose task, worker, resource, and control-layer state
    - detect bottlenecks
    - produce metrics for Cognitive Load Manager

    Non-responsibilities:
    - no task execution
    - no scheduling
    - no worker coordination
    - no mutation of other runtimes
    """

    def __init__(
        self,
        *,
        config: OrchestrationObservabilityConfig | None = None,
        bottleneck_detector: BottleneckDetector | None = None,
    ) -> None:
        self._config = config or OrchestrationObservabilityConfig()
        self._config.validate()

        self._bottleneck_detector = bottleneck_detector or BottleneckDetector(
            config=self._config
        )
        self._dashboards: list[OrchestrationDashboard] = []
        self._lock = RLock()

        self._last_health: OrchestrationHealth | None = None
        self._last_reason: ObservabilityReason | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def build_dashboard(
        self,
        *,
        state_machine: OrchestrationStateMachineSnapshot | None = None,
        scheduler: TaskSchedulerSnapshot | None = None,
        budget: ResourceBudgetRuntimeSnapshot | None = None,
        workers: WorkerHealthView | None = None,
        task_graphs: tuple[TaskGraphView, ...] = (),
        coordination: CoordinationEventView | None = None,
        background: BackgroundTaskRuntimeSnapshot | None = None,
        interrupts: InterruptPropagatorSnapshot | None = None,
        deadlocks: DeadlockDetectorSnapshot | None = None,
        circuit_breakers: CircuitBreakerRuntimeSnapshot | None = None,
    ) -> ObservabilityResult:
        resources = (
            ResourceUtilizationView.from_budget_snapshot(budget)
            if budget is not None
            else None
        )
        metrics = self._metrics(
            scheduler=scheduler,
            resources=resources,
            workers=workers,
            task_graphs=task_graphs,
            background=background,
            interrupts=interrupts,
        )
        bottlenecks = self._bottleneck_detector.detect(
            scheduler=scheduler,
            resources=resources,
            workers=workers,
            background=background,
            interrupts=interrupts,
            deadlocks=deadlocks,
            circuit_breakers=circuit_breakers,
        )
        health = self._health_from_bottlenecks(bottlenecks)
        dashboard = OrchestrationDashboard(
            health=health,
            summary=self._summary(health),
            state_machine=state_machine,
            scheduler=scheduler,
            resources=resources,
            workers=workers,
            task_graphs=task_graphs,
            coordination=coordination,
            background=background,
            interrupts=interrupts,
            deadlocks=deadlocks,
            circuit_breakers=circuit_breakers,
            metrics=metrics,
            bottlenecks=bottlenecks,
        )

        with self._lock:
            self._dashboards.append(dashboard)
            self._last_health = health
            self._last_reason = ObservabilityReason.DASHBOARD_CREATED

        return ObservabilityResult(
            reason=ObservabilityReason.DASHBOARD_CREATED,
            success=True,
            message="orchestration dashboard created",
            dashboard=dashboard,
        )

    def latest_dashboard(self) -> OrchestrationDashboard | None:
        with self._lock:
            if not self._dashboards:
                return None

            return self._dashboards[-1]

    def dashboards(self) -> tuple[OrchestrationDashboard, ...]:
        with self._lock:
            return tuple(self._dashboards)

    def snapshot(self) -> OrchestrationObservabilitySnapshot:
        with self._lock:
            bottleneck_count = 0

            if self._dashboards:
                latest = self._dashboards[-1]
                bottleneck_count = sum(
                    1 for item in latest.bottlenecks if item.detected
                )

            return OrchestrationObservabilitySnapshot(
                name=self.name,
                dashboard_count=len(self._dashboards),
                bottleneck_count=bottleneck_count,
                last_health=self._last_health,
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        with self._lock:
            self._dashboards.clear()
            self._last_health = None
            self._last_reason = ObservabilityReason.RUNTIME_RESET

    @staticmethod
    def _metrics(
        *,
        scheduler: TaskSchedulerSnapshot | None,
        resources: ResourceUtilizationView | None,
        workers: WorkerHealthView | None,
        task_graphs: tuple[TaskGraphView, ...],
        background: BackgroundTaskRuntimeSnapshot | None,
        interrupts: InterruptPropagatorSnapshot | None,
    ) -> RuntimeMetricSample:
        scheduled_count = scheduler.scheduled_count if scheduler else 0
        queue_depth = sum(graph.queue_depth for graph in task_graphs)

        if scheduler is not None:
            queue_depth += scheduler.deferred_count

        background_rate = 0.0

        if background is not None:
            total_background = background.scheduled_count + background.yielded_count

            if total_background > 0:
                background_rate = background.scheduled_count / total_background

        return RuntimeMetricSample(
            tasks_per_second=float(scheduled_count),
            worker_utilization_percent=(
                workers.utilization_percent if workers is not None else 0
            ),
            budget_consumption_percent=(
                resources.utilization_percent if resources is not None else 0
            ),
            queue_depth=queue_depth,
            interrupt_frequency=(
                interrupts.dispatch_count if interrupts is not None else 0
            ),
            background_completion_rate=background_rate,
        )

    @staticmethod
    def _health_from_bottlenecks(
        bottlenecks: tuple[BottleneckReport, ...],
    ) -> OrchestrationHealth:
        detected = tuple(item for item in bottlenecks if item.detected)

        if not detected:
            return OrchestrationHealth.HEALTHY

        if any(item.severity == BottleneckSeverity.CRITICAL for item in detected):
            return OrchestrationHealth.CRITICAL

        return OrchestrationHealth.DEGRADED

    @staticmethod
    def _summary(health: OrchestrationHealth) -> str:
        return {
            OrchestrationHealth.HEALTHY: "orchestration runtime is healthy",
            OrchestrationHealth.DEGRADED: "orchestration runtime is degraded",
            OrchestrationHealth.CRITICAL: "orchestration runtime is critical",
            OrchestrationHealth.UNKNOWN: "orchestration runtime is unknown",
        }[health]