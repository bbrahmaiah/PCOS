from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import RLock

from pydantic import Field, field_validator

from jarvis.orchestration.attention import (
    AttentionDecision,
    AttentionRuntime,
)
from jarvis.orchestration.budgets import (
    ResourceBudgetRuntime,
)
from jarvis.orchestration.ids import (
    TaskId,
    WorkerId,
    utc_now,
    validate_task_id,
    validate_worker_id,
)
from jarvis.orchestration.models import (
    OrchestrationModel,
    OrchestratorState,
    TaskPriority,
    TaskRequest,
    TaskStatus,
)
from jarvis.orchestration.registry import WorkerDescriptor, WorkerRegistry
from jarvis.orchestration.state_machine import OrchestrationStateMachine
from jarvis.orchestration.task_graph import TaskGraph


class TaskScheduleDecision(StrEnum):
    """
    Scheduler decision for a task.
    """

    SCHEDULED = "scheduled"
    DEFERRED = "deferred"
    DENIED = "denied"
    SKIPPED = "skipped"


class TaskScheduleReason(StrEnum):
    """
    Machine-readable scheduler reason.
    """

    TASK_SCHEDULED = "task_scheduled"
    TASK_ALREADY_SCHEDULED = "task_already_scheduled"
    TASK_NOT_READY = "task_not_ready"
    NO_READY_TASKS = "no_ready_tasks"
    ORCHESTRATOR_NOT_READY = "orchestrator_not_ready"
    ATTENTION_DEFERRED = "attention_deferred"
    ATTENTION_SUPPRESSED = "attention_suppressed"
    BUDGET_DENIED = "budget_denied"
    NO_WORKER_AVAILABLE = "no_worker_available"
    RESERVATION_FAILED = "reservation_failed"
    INVALID_WORKER_STATE = "invalid_worker_state"
    TASK_COMPLETED = "task_completed"
    SCHEDULER_RESET = "scheduler_reset"


class ScheduledTask(OrchestrationModel):
    """
    A task accepted by the scheduler and assigned to a worker.

    This is not execution. It is a scheduled assignment contract.
    """

    task_id: TaskId
    worker_id: WorkerId
    status: TaskStatus = TaskStatus.SCHEDULED
    priority: TaskPriority
    scheduled_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("task_id")
    @classmethod
    def _validate_task_id(cls, value: str) -> str:
        return validate_task_id(value)

    @field_validator("worker_id")
    @classmethod
    def _validate_worker_id(cls, value: str) -> str:
        return validate_worker_id(value)


class TaskScheduleResult(OrchestrationModel):
    """
    Result of a single scheduling decision.
    """

    task_id: TaskId
    decision: TaskScheduleDecision
    reason: TaskScheduleReason
    success: bool
    message: str
    worker_id: WorkerId | None = None
    scheduled_task: ScheduledTask | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("task_id")
    @classmethod
    def _validate_task_id(cls, value: str) -> str:
        return validate_task_id(value)

    @field_validator("worker_id")
    @classmethod
    def _validate_worker_id(cls, value: str | None) -> str | None:
        if value is None:
            return None

        return validate_worker_id(value)

    @field_validator("message")
    @classmethod
    def _required_message(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("message cannot be empty.")

        return cleaned


class TaskScheduleBatchResult(OrchestrationModel):
    """
    Result of scheduling ready tasks from a graph.
    """

    job_id: str
    results: tuple[TaskScheduleResult, ...]
    scheduled_count: int
    deferred_count: int
    denied_count: int
    skipped_count: int
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @property
    def scheduled(self) -> tuple[TaskScheduleResult, ...]:
        return tuple(
            result
            for result in self.results
            if result.decision == TaskScheduleDecision.SCHEDULED
        )

    @property
    def blocked(self) -> bool:
        return self.scheduled_count == 0 and (
            self.deferred_count > 0 or self.denied_count > 0
        )


class DependencyResolver:
    """
    Narrow dependency interface over TaskGraph.

    Scheduler uses this instead of inspecting dependencies manually.
    """

    @staticmethod
    def ready_tasks(graph: TaskGraph) -> tuple[TaskRequest, ...]:
        return graph.ready_tasks()


class DeadlineTracker:
    """
    Deadline ordering helper.

    Earlier timeout means stronger scheduling urgency.
    """

    @staticmethod
    def deadline_rank(task: TaskRequest) -> int:
        if task.timeout_ms is None:
            return 2_147_483_647

        return task.timeout_ms


class PriorityQueue:
    """
    Deterministic priority ordering for ready tasks.

    Lower TaskPriority value means higher priority.
    """

    @staticmethod
    def order(tasks: tuple[TaskRequest, ...]) -> tuple[TaskRequest, ...]:
        return tuple(
            sorted(
                tasks,
                key=lambda task: (
                    int(task.priority),
                    DeadlineTracker.deadline_rank(task),
                    task.created_at,
                    task.task_id,
                ),
            )
        )


@dataclass(frozen=True, slots=True)
class TaskSchedulerConfig:
    """
    Task scheduler configuration.
    """

    name: str = "task_scheduler"
    max_tasks_per_cycle: int = 8
    allowed_orchestrator_states: tuple[OrchestratorState, ...] = (
        OrchestratorState.IDLE,
        OrchestratorState.COORDINATING,
        OrchestratorState.BUSY,
        OrchestratorState.LOAD_SHEDDING,
    )

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.max_tasks_per_cycle <= 0:
            raise ValueError("max_tasks_per_cycle must be positive.")


@dataclass(frozen=True, slots=True)
class TaskSchedulerSnapshot:
    """
    Scheduler diagnostics.
    """

    name: str
    scheduled_count: int
    deferred_count: int
    denied_count: int
    skipped_count: int
    active_assignment_count: int
    last_decision: TaskScheduleDecision | None
    last_reason: TaskScheduleReason | None


class TaskScheduler:
    """
    Phase 6 Task Scheduler.

    Responsibilities:
    - ask TaskGraph which tasks are ready
    - order ready tasks by priority and deadline
    - ask AttentionRuntime whether task may run now
    - ask WorkerRegistry for healthy capable workers
    - ask ResourceBudgetRuntime to reserve declared resources
    - produce ScheduledTask assignment contracts

    Non-responsibilities:
    - no task execution
    - no worker implementation
    - no direct resource mutation outside ResourceBudgetRuntime
    - no direct attention mutation
    - no dependency guessing
    """

    def __init__(
        self,
        *,
        registry: WorkerRegistry,
        attention: AttentionRuntime,
        budgets: ResourceBudgetRuntime,
        state_machine: OrchestrationStateMachine,
        config: TaskSchedulerConfig | None = None,
    ) -> None:
        self._config = config or TaskSchedulerConfig()
        self._config.validate()

        self._registry = registry
        self._attention = attention
        self._budgets = budgets
        self._state_machine = state_machine
        self._lock = RLock()

        self._assignments: dict[TaskId, ScheduledTask] = {}
        self._scheduled_count = 0
        self._deferred_count = 0
        self._denied_count = 0
        self._skipped_count = 0
        self._last_decision: TaskScheduleDecision | None = None
        self._last_reason: TaskScheduleReason | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def schedule_ready(self, graph: TaskGraph) -> TaskScheduleBatchResult:
        """
        Schedule ready tasks from a graph.

        This method proposes assignments only. Workers still execute later.
        """

        if not self._orchestrator_ready():
            deferred_results = tuple(
                self._result(
                    task=task,
                    decision=TaskScheduleDecision.DEFERRED,
                    reason=TaskScheduleReason.ORCHESTRATOR_NOT_READY,
                    success=False,
                    message="orchestrator state does not allow scheduling",
                )
                for task in PriorityQueue.order(
                    DependencyResolver.ready_tasks(graph)
                )
            )

            if not deferred_results:
                deferred_results = (
                    TaskScheduleResult(
                        task_id=self._synthetic_task_id_for_empty_graph(graph),
                        decision=TaskScheduleDecision.DEFERRED,
                        reason=TaskScheduleReason.ORCHESTRATOR_NOT_READY,
                        success=False,
                        message="orchestrator state does not allow scheduling",
                    ),
                )

            return self._batch(job_id=graph.job_id, results=deferred_results)

        ready = PriorityQueue.order(DependencyResolver.ready_tasks(graph))

        if not ready:
            return self._batch(job_id=graph.job_id, results=())

        limited = ready[: self._config.max_tasks_per_cycle]
        results: list[TaskScheduleResult] = []

        for task in limited:
            results.append(self._schedule_one(task))

        return self._batch(job_id=graph.job_id, results=tuple(results))

    def complete(self, task_id: TaskId) -> TaskScheduleResult:
        """
        Complete scheduler ownership of a task assignment.

        This releases reserved resources and reduces worker load.
        """

        validated_task_id = validate_task_id(task_id)

        with self._lock:
            assignment = self._assignments.pop(validated_task_id, None)

        if assignment is None:
            return TaskScheduleResult(
                task_id=validated_task_id,
                decision=TaskScheduleDecision.SKIPPED,
                reason=TaskScheduleReason.TASK_NOT_READY,
                success=False,
                message="task assignment not found",
            )

        self._budgets.release(validated_task_id)
        lookup = self._registry.get(assignment.worker_id)

        if lookup.descriptor is not None:
            current_load = lookup.descriptor.load
            next_active = max(0, current_load.active_tasks - 1)
            self._registry.update_load(
                assignment.worker_id,
                active_tasks=next_active,
                queued_tasks=current_load.queued_tasks,
            )

        result = TaskScheduleResult(
            task_id=validated_task_id,
            decision=TaskScheduleDecision.SKIPPED,
            reason=TaskScheduleReason.TASK_COMPLETED,
            success=True,
            message="scheduled task completed and released",
            worker_id=assignment.worker_id,
            scheduled_task=assignment,
        )
        self._record(result)

        return result

    def assignment_for(self, task_id: TaskId) -> ScheduledTask | None:
        """
        Return active assignment for task.
        """

        validated_task_id = validate_task_id(task_id)

        with self._lock:
            return self._assignments.get(validated_task_id)

    def active_assignments(self) -> tuple[ScheduledTask, ...]:
        """
        Return active scheduled assignments.
        """

        with self._lock:
            return tuple(self._assignments.values())

    def reset(self) -> None:
        """
        Clear scheduler assignments and metrics.

        This does not release budgets. Use only in test/bootstrap boundaries.
        """

        with self._lock:
            self._assignments.clear()
            self._scheduled_count = 0
            self._deferred_count = 0
            self._denied_count = 0
            self._skipped_count = 0
            self._last_decision = None
            self._last_reason = None

    def snapshot(self) -> TaskSchedulerSnapshot:
        """
        Return scheduler diagnostics.
        """

        with self._lock:
            return TaskSchedulerSnapshot(
                name=self.name,
                scheduled_count=self._scheduled_count,
                deferred_count=self._deferred_count,
                denied_count=self._denied_count,
                skipped_count=self._skipped_count,
                active_assignment_count=len(self._assignments),
                last_decision=self._last_decision,
                last_reason=self._last_reason,
            )

    def _schedule_one(self, task: TaskRequest) -> TaskScheduleResult:
        with self._lock:
            if task.task_id in self._assignments:
                result = self._result(
                    task=task,
                    decision=TaskScheduleDecision.SKIPPED,
                    reason=TaskScheduleReason.TASK_ALREADY_SCHEDULED,
                    success=True,
                    message="task already scheduled",
                    scheduled_task=self._assignments[task.task_id],
                    worker_id=self._assignments[task.task_id].worker_id,
                )
                self._record(result)

                return result

        attention = self._attention.evaluate(task)

        if attention.decision == AttentionDecision.DEFER:
            result = self._result(
                task=task,
                decision=TaskScheduleDecision.DEFERRED,
                reason=TaskScheduleReason.ATTENTION_DEFERRED,
                success=False,
                message="attention runtime deferred task",
                metadata={"attention_reason": attention.reason.value},
            )
            self._record(result)

            return result

        if attention.decision == AttentionDecision.SUPPRESS:
            result = self._result(
                task=task,
                decision=TaskScheduleDecision.DEFERRED,
                reason=TaskScheduleReason.ATTENTION_SUPPRESSED,
                success=False,
                message="attention runtime suppressed task",
                metadata={"attention_reason": attention.reason.value},
            )
            self._record(result)

            return result

        budget = self._budgets.evaluate(task)

        if not budget.allowed:
            result = self._result(
                task=task,
                decision=TaskScheduleDecision.DENIED,
                reason=TaskScheduleReason.BUDGET_DENIED,
                success=False,
                message="resource budget denied task",
                metadata={"budget_reason": budget.reason.value},
            )
            self._record(result)

            return result

        worker = self._select_worker(task)

        if worker is None:
            result = self._result(
                task=task,
                decision=TaskScheduleDecision.DEFERRED,
                reason=TaskScheduleReason.NO_WORKER_AVAILABLE,
                success=False,
                message="no healthy capable worker available",
            )
            self._record(result)

            return result

        reservation = self._budgets.reserve(task)

        if not reservation.success:
            result = self._result(
                task=task,
                decision=TaskScheduleDecision.DENIED,
                reason=TaskScheduleReason.RESERVATION_FAILED,
                success=False,
                message="resource reservation failed",
                metadata={"budget_reason": reservation.reason.value},
            )
            self._record(result)

            return result

        scheduled = ScheduledTask(
            task_id=task.task_id,
            worker_id=worker.worker_id,
            priority=task.priority,
            metadata={
                "attention_decision": attention.decision.value,
                "budget_decision": reservation.decision.value,
            },
        )

        with self._lock:
            self._assignments[task.task_id] = scheduled

        self._registry.update_load(
            worker.worker_id,
            active_tasks=worker.load.active_tasks + 1,
            queued_tasks=worker.load.queued_tasks,
        )

        result = self._result(
            task=task,
            decision=TaskScheduleDecision.SCHEDULED,
            reason=TaskScheduleReason.TASK_SCHEDULED,
            success=True,
            message="task scheduled",
            scheduled_task=scheduled,
            worker_id=worker.worker_id,
        )
        self._record(result)

        return result

    def _select_worker(self, task: TaskRequest) -> WorkerDescriptor | None:
        candidates = self._registry.find_for_task(task)

        if not candidates:
            return None

        return sorted(
            candidates,
            key=lambda worker: (
                worker.load.load_factor,
                worker.load.active_tasks,
                worker.worker_id,
            ),
        )[0]

    def _orchestrator_ready(self) -> bool:
        return (
            self._state_machine.state.state
            in self._config.allowed_orchestrator_states
        )

    def _batch(
        self,
        *,
        job_id: str,
        results: tuple[TaskScheduleResult, ...],
    ) -> TaskScheduleBatchResult:
        return TaskScheduleBatchResult(
            job_id=job_id,
            results=results,
            scheduled_count=sum(
                1
                for result in results
                if result.decision == TaskScheduleDecision.SCHEDULED
            ),
            deferred_count=sum(
                1
                for result in results
                if result.decision == TaskScheduleDecision.DEFERRED
            ),
            denied_count=sum(
                1
                for result in results
                if result.decision == TaskScheduleDecision.DENIED
            ),
            skipped_count=sum(
                1
                for result in results
                if result.decision == TaskScheduleDecision.SKIPPED
            ),
        )

    @staticmethod
    def _result(
        *,
        task: TaskRequest,
        decision: TaskScheduleDecision,
        reason: TaskScheduleReason,
        success: bool,
        message: str,
        worker_id: WorkerId | None = None,
        scheduled_task: ScheduledTask | None = None,
        metadata: dict[str, object] | None = None,
    ) -> TaskScheduleResult:
        return TaskScheduleResult(
            task_id=task.task_id,
            decision=decision,
            reason=reason,
            success=success,
            message=message,
            worker_id=worker_id,
            scheduled_task=scheduled_task,
            metadata=metadata or {},
        )

    def _record(self, result: TaskScheduleResult) -> None:
        with self._lock:
            self._last_decision = result.decision
            self._last_reason = result.reason

            if result.decision == TaskScheduleDecision.SCHEDULED:
                self._scheduled_count += 1

            elif result.decision == TaskScheduleDecision.DEFERRED:
                self._deferred_count += 1

            elif result.decision == TaskScheduleDecision.DENIED:
                self._denied_count += 1

            else:
                self._skipped_count += 1

    @staticmethod
    def _synthetic_task_id_for_empty_graph(graph: TaskGraph) -> TaskId:
        for task in graph.job.tasks:
            return task.task_id

        raise ValueError("task graph has no tasks.")