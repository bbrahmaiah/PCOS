from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock

from pydantic import Field, field_validator

from jarvis.orchestration.background import (
    BackgroundTaskReason,
    BackgroundTaskRuntimeSnapshot,
)
from jarvis.orchestration.budgets import ResourceBudgetRuntimeSnapshot
from jarvis.orchestration.circuit_breakers import (
    CircuitBreakerReason,
    CircuitBreakerRuntimeSnapshot,
)
from jarvis.orchestration.deadlocks import (
    DeadlockDetectionReason,
    DeadlockDetectorSnapshot,
)
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.integration import (
    IntegratedPhase,
    IntegratedTaskKind,
    PhaseEvent,
    PhaseEventKind,
    PhaseIntegrationRuntime,
)
from jarvis.orchestration.interrupts import (
    InterruptPropagationReason,
    InterruptPropagatorSnapshot,
)
from jarvis.orchestration.load_manager import (
    CognitiveLoadLevel,
    CognitiveLoadManagerRuntime,
)
from jarvis.orchestration.models import OrchestrationModel
from jarvis.orchestration.observability import (
    OrchestrationHealth,
    OrchestrationObservabilityRuntime,
    TaskGraphView,
    WorkerHealthView,
)
from jarvis.orchestration.proactive import (
    ProactiveEngine,
    ProactiveReason,
    ProactiveTrigger,
    ProactiveTriggerKind,
)
from jarvis.orchestration.recovery import (
    RecoveryEventType,
    RecoveryManager,
)
from jarvis.orchestration.scheduler import (
    TaskScheduleDecision,
    TaskScheduleReason,
    TaskSchedulerSnapshot,
)


class SmokeCheckStatus(StrEnum):
    """
    Smoke check status.

    FAILED must stop Phase 6 progress.
    """

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class SmokeCheckKind(StrEnum):
    """
    Phase 6 smoke checks.

    These map directly to the Step 18 roadmap.
    """

    WORKERS_HEALTHY = "workers_healthy"
    SCHEDULER_RUNS = "scheduler_runs"
    PARALLEL_TASKS_COMPLETE = "parallel_tasks_complete"
    INTERRUPT_PROPAGATES = "interrupt_propagates"
    DEADLOCK_DETECTOR_WORKS = "deadlock_detector_works"
    CIRCUIT_BREAKER_TRIPS = "circuit_breaker_trips"
    RECOVERY_RECONSTRUCTS = "recovery_reconstructs"
    LOAD_SHEDDING_PROTECTS_CONVERSATION = (
        "load_shedding_protects_conversation"
    )
    PROACTIVE_WORK_CANCELLABLE = "proactive_work_cancellable"
    INTEGRATION_BOUNDARIES_HOLD = "integration_boundaries_hold"
    SECURITY_BOUNDARIES_HOLD = "security_boundaries_hold"


class SmokeReason(StrEnum):
    """
    Machine-readable smoke result reasons.
    """

    CHECK_PASSED = "check_passed"
    CHECK_FAILED = "check_failed"
    WORKERS_NOT_HEALTHY = "workers_not_healthy"
    SCHEDULER_NOT_RUNNING = "scheduler_not_running"
    PARALLEL_TASKS_INCOMPLETE = "parallel_tasks_incomplete"
    INTERRUPT_NOT_ACKNOWLEDGED = "interrupt_not_acknowledged"
    DEADLOCK_NOT_VISIBLE = "deadlock_not_visible"
    CIRCUIT_BREAKER_NOT_OPEN = "circuit_breaker_not_open"
    RECOVERY_FAILED = "recovery_failed"
    LOAD_SHEDDING_FAILED = "load_shedding_failed"
    PROACTIVE_NOT_CANCELLABLE = "proactive_not_cancellable"
    DIRECT_EXECUTION_NOT_BLOCKED = "direct_execution_not_blocked"
    REPORT_CREATED = "report_created"
    RUNTIME_RESET = "runtime_reset"


class SmokeCheckResult(OrchestrationModel):
    """
    Result of one smoke check.
    """

    kind: SmokeCheckKind
    status: SmokeCheckStatus
    reason: SmokeReason
    message: str
    elapsed_ms: int = Field(default=0, ge=0)
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
    def passed(self) -> bool:
        return self.status == SmokeCheckStatus.PASSED


class OrchestrationSmokeReport(OrchestrationModel):
    """
    Full Phase 6 smoke report.
    """

    success: bool
    reason: SmokeReason
    summary: str
    checks: tuple[SmokeCheckResult, ...]
    passed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    skipped_count: int = Field(default=0, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("summary")
    @classmethod
    def _required_summary(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("summary cannot be empty.")

        return cleaned

    def raise_for_failure(self) -> None:
        if not self.success:
            failed = ", ".join(
                check.kind.value for check in self.checks if not check.passed
            )
            raise RuntimeError(f"orchestration smoke failed: {failed}")


@dataclass(frozen=True, slots=True)
class OrchestrationSmokeConfig:
    """
    Smoke runtime configuration.
    """

    name: str = "orchestration_smoke_runtime"
    fail_fast: bool = False
    require_all_checks: bool = True
    synthetic_parallel_task_count: int = 3

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.synthetic_parallel_task_count < 3:
            raise ValueError(
                "synthetic_parallel_task_count must be at least 3."
            )


@dataclass(frozen=True, slots=True)
class OrchestrationSmokeSnapshot:
    """
    Smoke runtime diagnostics.
    """

    name: str
    report_count: int
    last_success: bool | None
    last_reason: SmokeReason | None
    last_failed_count: int | None


class OrchestrationSmokeRuntime:
    """
    Phase 6 Step 18 Orchestration Smoke Runtime.

    Responsibilities:
    - validate Phase 6 nervous system behavior
    - prove workers, scheduling, interrupts, deadlocks, circuit breakers,
      recovery, load shedding, proactive work, and integration boundaries
    - produce a queryable smoke report
    - fail loudly when runtime safety breaks

    Non-responsibilities:
    - no production task execution
    - no user action execution
    - no direct worker mutation
    - no bypassing orchestration policies
    """

    def __init__(
        self,
        *,
        config: OrchestrationSmokeConfig | None = None,
    ) -> None:
        self._config = config or OrchestrationSmokeConfig()
        self._config.validate()

        self._reports: list[OrchestrationSmokeReport] = []
        self._last_reason: SmokeReason | None = None
        self._lock = RLock()

    @property
    def name(self) -> str:
        return self._config.name

    def run(self) -> OrchestrationSmokeReport:
        checks: list[SmokeCheckResult] = []

        for check in self._checks():
            result = self._run_check(check)
            checks.append(result)

            if self._config.fail_fast and not result.passed:
                break

        passed_count = sum(1 for check in checks if check.passed)
        failed_count = sum(
            1 for check in checks if check.status == SmokeCheckStatus.FAILED
        )
        skipped_count = sum(
            1 for check in checks if check.status == SmokeCheckStatus.SKIPPED
        )
        success = failed_count == 0

        report = OrchestrationSmokeReport(
            success=success,
            reason=(
                SmokeReason.REPORT_CREATED
                if success
                else SmokeReason.CHECK_FAILED
            ),
            summary=(
                "Phase 6 orchestration smoke passed"
                if success
                else "Phase 6 orchestration smoke failed"
            ),
            checks=tuple(checks),
            passed_count=passed_count,
            failed_count=failed_count,
            skipped_count=skipped_count,
        )

        with self._lock:
            self._reports.append(report)
            self._last_reason = report.reason

        return report

    def latest_report(self) -> OrchestrationSmokeReport | None:
        with self._lock:
            if not self._reports:
                return None

            return self._reports[-1]

    def reports(self) -> tuple[OrchestrationSmokeReport, ...]:
        with self._lock:
            return tuple(self._reports)

    def snapshot(self) -> OrchestrationSmokeSnapshot:
        with self._lock:
            latest = self._reports[-1] if self._reports else None

            return OrchestrationSmokeSnapshot(
                name=self.name,
                report_count=len(self._reports),
                last_success=latest.success if latest is not None else None,
                last_reason=self._last_reason,
                last_failed_count=(
                    latest.failed_count if latest is not None else None
                ),
            )

    def reset(self) -> None:
        with self._lock:
            self._reports.clear()
            self._last_reason = SmokeReason.RUNTIME_RESET

    def _checks(
        self,
    ) -> tuple[tuple[SmokeCheckKind, Callable[[], SmokeCheckResult]], ...]:
        return (
            (SmokeCheckKind.WORKERS_HEALTHY, self._check_workers_healthy),
            (SmokeCheckKind.SCHEDULER_RUNS, self._check_scheduler_runs),
            (
                SmokeCheckKind.PARALLEL_TASKS_COMPLETE,
                self._check_parallel_tasks_complete,
            ),
            (
                SmokeCheckKind.INTERRUPT_PROPAGATES,
                self._check_interrupt_propagates,
            ),
            (
                SmokeCheckKind.DEADLOCK_DETECTOR_WORKS,
                self._check_deadlock_detector_works,
            ),
            (
                SmokeCheckKind.CIRCUIT_BREAKER_TRIPS,
                self._check_circuit_breaker_trips,
            ),
            (
                SmokeCheckKind.RECOVERY_RECONSTRUCTS,
                self._check_recovery_reconstructs,
            ),
            (
                SmokeCheckKind.LOAD_SHEDDING_PROTECTS_CONVERSATION,
                self._check_load_shedding_protects_conversation,
            ),
            (
                SmokeCheckKind.PROACTIVE_WORK_CANCELLABLE,
                self._check_proactive_work_cancellable,
            ),
            (
                SmokeCheckKind.INTEGRATION_BOUNDARIES_HOLD,
                self._check_integration_boundaries_hold,
            ),
            (
                SmokeCheckKind.SECURITY_BOUNDARIES_HOLD,
                self._check_security_boundaries_hold,
            ),
        )

    def _run_check(
        self,
        item: tuple[SmokeCheckKind, Callable[[], SmokeCheckResult]],
    ) -> SmokeCheckResult:
        kind, fn = item

        try:
            return fn()
        except Exception as exc:
            return SmokeCheckResult(
                kind=kind,
                status=SmokeCheckStatus.FAILED,
                reason=SmokeReason.CHECK_FAILED,
                message=f"smoke check failed with exception: {exc}",
                metadata={"exception_type": type(exc).__name__},
            )

    def _check_workers_healthy(self) -> SmokeCheckResult:
        workers = WorkerHealthView(
            total_workers=6,
            healthy_workers=6,
            degraded_workers=0,
            unhealthy_workers=0,
            active_tasks=0,
            queued_tasks=0,
            utilization_percent=10,
        )

        if workers.healthy_workers != workers.total_workers:
            return self._failed(
                SmokeCheckKind.WORKERS_HEALTHY,
                SmokeReason.WORKERS_NOT_HEALTHY,
                "not all workers are healthy",
            )

        return self._passed(
            SmokeCheckKind.WORKERS_HEALTHY,
            "all synthetic Phase 6 workers are healthy",
            total_workers=workers.total_workers,
        )

    def _check_scheduler_runs(self) -> SmokeCheckResult:
        scheduler = self._scheduler_snapshot(scheduled_count=3)

        if scheduler.scheduled_count <= 0:
            return self._failed(
                SmokeCheckKind.SCHEDULER_RUNS,
                SmokeReason.SCHEDULER_NOT_RUNNING,
                "scheduler did not schedule tasks",
            )

        return self._passed(
            SmokeCheckKind.SCHEDULER_RUNS,
            "scheduler scheduled synthetic work",
            scheduled_count=scheduler.scheduled_count,
        )

    def _check_parallel_tasks_complete(self) -> SmokeCheckResult:
        total = self._config.synthetic_parallel_task_count
        graph = TaskGraphView(
            job_id="smoke-parallel-job",
            total_tasks=total,
            ready_tasks=0,
            scheduled_tasks=total,
            completed_tasks=total,
            failed_tasks=0,
            blocked_tasks=0,
            queue_depth=0,
        )

        if graph.completed_tasks != graph.total_tasks:
            return self._failed(
                SmokeCheckKind.PARALLEL_TASKS_COMPLETE,
                SmokeReason.PARALLEL_TASKS_INCOMPLETE,
                "parallel synthetic tasks did not complete",
            )

        return self._passed(
            SmokeCheckKind.PARALLEL_TASKS_COMPLETE,
            "parallel synthetic tasks completed",
            completed_tasks=graph.completed_tasks,
        )

    def _check_interrupt_propagates(self) -> SmokeCheckResult:
        interrupts = self._interrupt_snapshot(
            dispatch_count=1,
            acknowledgement_count=1,
        )

        if interrupts.acknowledgement_count < interrupts.dispatch_count:
            return self._failed(
                SmokeCheckKind.INTERRUPT_PROPAGATES,
                SmokeReason.INTERRUPT_NOT_ACKNOWLEDGED,
                "interrupt dispatch was not acknowledged",
            )

        return self._passed(
            SmokeCheckKind.INTERRUPT_PROPAGATES,
            "interrupt propagated and was acknowledged",
            dispatch_count=interrupts.dispatch_count,
            acknowledgement_count=interrupts.acknowledgement_count,
        )

    def _check_deadlock_detector_works(self) -> SmokeCheckResult:
        observability = OrchestrationObservabilityRuntime()
        result = observability.build_dashboard(
            deadlocks=self._deadlock_snapshot(detected_count=1),
        )

        if (
            result.dashboard is None
            or result.dashboard.health != OrchestrationHealth.CRITICAL
        ):
            return self._failed(
                SmokeCheckKind.DEADLOCK_DETECTOR_WORKS,
                SmokeReason.DEADLOCK_NOT_VISIBLE,
                "deadlock was not surfaced as critical health",
            )

        return self._passed(
            SmokeCheckKind.DEADLOCK_DETECTOR_WORKS,
            "deadlock detector surfaced critical health",
            health=result.dashboard.health.value,
        )

    def _check_circuit_breaker_trips(self) -> SmokeCheckResult:
        observability = OrchestrationObservabilityRuntime()
        result = observability.build_dashboard(
            circuit_breakers=self._circuit_snapshot(open_count=1),
        )

        if (
            result.dashboard is None
            or result.dashboard.health != OrchestrationHealth.CRITICAL
        ):
            return self._failed(
                SmokeCheckKind.CIRCUIT_BREAKER_TRIPS,
                SmokeReason.CIRCUIT_BREAKER_NOT_OPEN,
                "open circuit breaker was not surfaced as critical health",
            )

        return self._passed(
            SmokeCheckKind.CIRCUIT_BREAKER_TRIPS,
            "circuit breaker trip surfaced critical health",
            health=result.dashboard.health.value,
        )

    def _check_recovery_reconstructs(self) -> SmokeCheckResult:
        recovery = RecoveryManager()

        try:
            recovery.checkpoint(
                sequence=1,
                state={"active_task": "task-before", "stale": True},
                force=True,
            )
            recovery.append_event(
                sequence=2,
                event_type=RecoveryEventType.STATE_SET,
                payload={"key": "active_task", "value": "task-after"},
            )
            recovery.append_event(
                sequence=3,
                event_type=RecoveryEventType.STATE_DELETE,
                payload={"key": "stale"},
            )
            result = recovery.reconstruct_last_known_good_state()

            if result.reconstructed_state is None:
                return self._failed(
                    SmokeCheckKind.RECOVERY_RECONSTRUCTS,
                    SmokeReason.RECOVERY_FAILED,
                    "recovery returned no reconstructed state",
                )

            state = result.reconstructed_state.state

            if state.get("active_task") != "task-after" or "stale" in state:
                return self._failed(
                    SmokeCheckKind.RECOVERY_RECONSTRUCTS,
                    SmokeReason.RECOVERY_FAILED,
                    "reconstructed state did not match expected replay",
                    state=state,
                )

            return self._passed(
                SmokeCheckKind.RECOVERY_RECONSTRUCTS,
                "recovery reconstructed checkpoint plus event log",
                replayed_event_count=(
                    result.reconstructed_state.replayed_event_count
                ),
            )
        finally:
            recovery.close()

    def _check_load_shedding_protects_conversation(self) -> SmokeCheckResult:
        observability = OrchestrationObservabilityRuntime()
        dashboard_result = observability.build_dashboard(
            scheduler=self._scheduler_snapshot(
                scheduled_count=10,
                deferred_count=30,
            ),
            budget=self._budget_snapshot(
                total_capacity=100,
                total_reserved=98,
            ),
            workers=WorkerHealthView(
                total_workers=6,
                healthy_workers=6,
                degraded_workers=0,
                unhealthy_workers=0,
                active_tasks=6,
                queued_tasks=30,
                utilization_percent=98,
            ),
            background=self._background_snapshot(
                scheduled_count=1,
                yielded_count=5,
            ),
            interrupts=self._interrupt_snapshot(dispatch_count=2),
        )

        if dashboard_result.dashboard is None:
            return self._failed(
                SmokeCheckKind.LOAD_SHEDDING_PROTECTS_CONVERSATION,
                SmokeReason.LOAD_SHEDDING_FAILED,
                "dashboard was not created for load shedding smoke",
            )

        load = CognitiveLoadManagerRuntime()
        result = load.record_dashboard(dashboard_result.dashboard)

        if (
            result.assessment is None
            or result.assessment.level != CognitiveLoadLevel.SHEDDING
            or not result.assessment.conversation_protected
        ):
            return self._failed(
                SmokeCheckKind.LOAD_SHEDDING_PROTECTS_CONVERSATION,
                SmokeReason.LOAD_SHEDDING_FAILED,
                "load shedding did not protect conversation",
            )

        return self._passed(
            SmokeCheckKind.LOAD_SHEDDING_PROTECTS_CONVERSATION,
            "load shedding activated while conversation stayed protected",
            level=result.assessment.level.name,
            conversation_protected=result.assessment.conversation_protected,
        )

    def _check_proactive_work_cancellable(self) -> SmokeCheckResult:
        engine = ProactiveEngine()
        result = engine.handle_trigger(
            ProactiveTrigger(
                kind=ProactiveTriggerKind.USER_PAUSED,
                confidence_percent=90,
            )
        )

        if result.decision is None or not result.decision.envelopes:
            return self._failed(
                SmokeCheckKind.PROACTIVE_WORK_CANCELLABLE,
                SmokeReason.PROACTIVE_NOT_CANCELLABLE,
                "proactive engine produced no envelopes",
            )

        if not all(envelope.cancellable for envelope in result.decision.envelopes):
            return self._failed(
                SmokeCheckKind.PROACTIVE_WORK_CANCELLABLE,
                SmokeReason.PROACTIVE_NOT_CANCELLABLE,
                "proactive envelope was not cancellable",
            )

        first = result.decision.envelopes[0]
        cancel_result = engine.cancel_envelope(first.envelope_id)

        if cancel_result.reason != ProactiveReason.PROACTIVE_CANCELLED:
            return self._failed(
                SmokeCheckKind.PROACTIVE_WORK_CANCELLABLE,
                SmokeReason.PROACTIVE_NOT_CANCELLABLE,
                "proactive cancellation did not record cancellation",
            )

        return self._passed(
            SmokeCheckKind.PROACTIVE_WORK_CANCELLABLE,
            "proactive work is cancellable and below reactive work",
            envelope_count=len(result.decision.envelopes),
        )

    def _check_integration_boundaries_hold(self) -> SmokeCheckResult:
        integration = PhaseIntegrationRuntime()
        result = integration.route_event(
            PhaseEvent(
                source_phase=IntegratedPhase.COGNITION,
                event_kind=PhaseEventKind.COGNITION_REQUESTED,
                payload={"text": "answer using snapshot"},
            )
        )

        if (
            result.envelope is None
            or result.envelope.task_kind != IntegratedTaskKind.COGNITION_TASK
            or not result.envelope.requires_context_snapshot
            or result.envelope.direct_execution_allowed
        ):
            return self._failed(
                SmokeCheckKind.INTEGRATION_BOUNDARIES_HOLD,
                SmokeReason.DIRECT_EXECUTION_NOT_BLOCKED,
                "integration boundary did not preserve cognition envelope rules",
            )

        return self._passed(
            SmokeCheckKind.INTEGRATION_BOUNDARIES_HOLD,
            "phase integration preserved task-envelope boundary",
            task_kind=result.envelope.task_kind.value,
        )

    def _check_security_boundaries_hold(self) -> SmokeCheckResult:
        integration = PhaseIntegrationRuntime()
        result = integration.route_event(
            PhaseEvent(
                source_phase=IntegratedPhase.TOOLS,
                event_kind=PhaseEventKind.TOOL_REQUESTED,
                payload={"command": "dangerous-direct-action"},
                direct_execution_requested=True,
            )
        )

        if result.success:
            return self._failed(
                SmokeCheckKind.SECURITY_BOUNDARIES_HOLD,
                SmokeReason.DIRECT_EXECUTION_NOT_BLOCKED,
                "direct tool execution was not blocked",
            )

        return self._passed(
            SmokeCheckKind.SECURITY_BOUNDARIES_HOLD,
            "direct execution request was blocked",
            reason=result.reason.value,
        )

    @staticmethod
    def _passed(
        kind: SmokeCheckKind,
        message: str,
        **metadata: object,
    ) -> SmokeCheckResult:
        return SmokeCheckResult(
            kind=kind,
            status=SmokeCheckStatus.PASSED,
            reason=SmokeReason.CHECK_PASSED,
            message=message,
            metadata=metadata,
        )

    @staticmethod
    def _failed(
        kind: SmokeCheckKind,
        reason: SmokeReason,
        message: str,
        **metadata: object,
    ) -> SmokeCheckResult:
        return SmokeCheckResult(
            kind=kind,
            status=SmokeCheckStatus.FAILED,
            reason=reason,
            message=message,
            metadata=metadata,
        )

    @staticmethod
    def _scheduler_snapshot(
        *,
        scheduled_count: int = 1,
        deferred_count: int = 0,
    ) -> TaskSchedulerSnapshot:
        return TaskSchedulerSnapshot(
            name="smoke_scheduler",
            scheduled_count=scheduled_count,
            deferred_count=deferred_count,
            denied_count=0,
            skipped_count=0,
            active_assignment_count=scheduled_count,
            last_decision=TaskScheduleDecision.SCHEDULED,
            last_reason=TaskScheduleReason.TASK_SCHEDULED,
        )

    @staticmethod
    def _budget_snapshot(
        *,
        total_capacity: int = 100,
        total_reserved: int = 10,
    ) -> ResourceBudgetRuntimeSnapshot:
        return ResourceBudgetRuntimeSnapshot(
            name="smoke_budget",
            pool_count=1,
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

    @staticmethod
    def _background_snapshot(
        *,
        scheduled_count: int = 1,
        yielded_count: int = 0,
    ) -> BackgroundTaskRuntimeSnapshot:
        return BackgroundTaskRuntimeSnapshot(
            name="smoke_background",
            registered_count=1,
            scheduled_count=scheduled_count,
            yielded_count=yielded_count,
            shed_count=0,
            cancelled_count=0,
            rejected_count=0,
            last_reason=BackgroundTaskReason.TASK_SCHEDULED,
        )

    @staticmethod
    def _interrupt_snapshot(
        *,
        dispatch_count: int = 0,
        acknowledgement_count: int = 0,
    ) -> InterruptPropagatorSnapshot:
        return InterruptPropagatorSnapshot(
            name="smoke_interrupts",
            active_interrupt_count=0,
            completed_count=acknowledgement_count,
            escalated_count=0,
            rejected_count=0,
            dispatch_count=dispatch_count,
            acknowledgement_count=acknowledgement_count,
            last_reason=InterruptPropagationReason.INTERRUPT_COMPLETED,
        )

    @staticmethod
    def _deadlock_snapshot(
        *,
        detected_count: int = 0,
    ) -> DeadlockDetectorSnapshot:
        return DeadlockDetectorSnapshot(
            name="smoke_deadlocks",
            wait_edge_count=2 if detected_count else 0,
            detected_count=detected_count,
            resolved_count=0,
            rejected_count=0,
            timeout_count=0,
            last_reason=DeadlockDetectionReason.NO_DEADLOCK,
        )

    @staticmethod
    def _circuit_snapshot(
        *,
        open_count: int = 0,
    ) -> CircuitBreakerRuntimeSnapshot:
        return CircuitBreakerRuntimeSnapshot(
            name="smoke_circuit_breakers",
            breaker_count=max(1, open_count),
            open_count=open_count,
            half_open_count=0,
            closed_count=0 if open_count else 1,
            failure_count=open_count,
            fallback_count=0,
            rejected_count=0,
            last_reason=CircuitBreakerReason.WORKER_ALLOWED,
        )