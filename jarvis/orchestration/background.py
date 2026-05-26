from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.orchestration.attention import (
    AttentionDecision,
    AttentionRuntime,
)
from jarvis.orchestration.budgets import worker_slots
from jarvis.orchestration.ids import TaskId, utc_now, validate_task_id
from jarvis.orchestration.models import (
    JobRequest,
    OrchestrationModel,
    OrchestratorState,
    ResourceBudget,
    TaskKind,
    TaskPriority,
    TaskRequest,
    WorkerCapability,
)
from jarvis.orchestration.scheduler import (
    TaskScheduler,
    TaskScheduleResult,
)
from jarvis.orchestration.state_machine import OrchestrationStateMachine
from jarvis.orchestration.task_graph import TaskGraph


def new_background_task_id() -> str:
    return f"bgtask_{uuid4().hex}"


def new_background_token_id() -> str:
    return f"bgtoken_{uuid4().hex}"


class BackgroundTaskKind(StrEnum):
    """
    Supported background task kinds.

    Background work is useful, but dangerous if not controlled.
    """

    MEMORY_CONSOLIDATION = "memory_consolidation"
    CONTEXT_PREFETCH = "context_prefetch"
    WORKSPACE_SCAN = "workspace_scan"
    HEALTH_CHECK = "health_check"
    LOG_ROTATION = "log_rotation"
    BACKGROUND_SUMMARY = "background_summary"


class BackgroundTaskStatus(StrEnum):
    """
    Runtime lifecycle for background tasks.
    """

    REGISTERED = "registered"
    READY = "ready"
    SCHEDULED = "scheduled"
    YIELDED = "yielded"
    SHED = "shed"
    CANCELLED = "cancelled"
    DISABLED = "disabled"
    REJECTED = "rejected"


class BackgroundTaskDecision(StrEnum):
    """
    Decision returned by the background task runtime.
    """

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SCHEDULED = "scheduled"
    YIELDED = "yielded"
    SHED = "shed"
    CANCELLED = "cancelled"


class BackgroundTaskReason(StrEnum):
    """
    Machine-readable background runtime reason.
    """

    TASK_REGISTERED = "task_registered"
    TASK_ALREADY_REGISTERED = "task_already_registered"
    TASK_DISABLED = "task_disabled"
    TASK_CANCELLED = "task_cancelled"
    TASK_NOT_FOUND = "task_not_found"
    TASK_SCHEDULED = "task_scheduled"
    TASK_YIELDED_TO_ATTENTION = "task_yielded_to_attention"
    TASK_SHED_DURING_LOAD_SHEDDING = "task_shed_during_load_shedding"
    TASK_REJECTED_NOT_CANCELLABLE = "task_rejected_not_cancellable"
    TASK_REJECTED_NOT_BACKGROUND = "task_rejected_not_background"
    SCHEDULER_DEFERRED = "scheduler_deferred"
    SCHEDULER_DENIED = "scheduler_denied"
    RUNTIME_RESET = "runtime_reset"


class BackgroundCancellationToken(OrchestrationModel):
    """
    Cancellation token for background work.

    Every background task must be cancellable.
    """

    token_id: str = Field(default_factory=new_background_token_id)
    task_id: TaskId
    cancel_requested: bool = False
    reason: str | None = None
    created_at: object = Field(default_factory=utc_now)
    cancelled_at: object | None = None
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("token_id")
    @classmethod
    def _validate_token_id(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("token_id cannot be empty.")

        if not cleaned.startswith("bgtoken_"):
            raise ValueError("token_id must start with 'bgtoken_'.")

        return cleaned

    @field_validator("task_id")
    @classmethod
    def _validate_task_id(cls, value: str) -> str:
        return validate_task_id(value)

    def request_cancel(self, reason: str) -> BackgroundCancellationToken:
        """
        Return a cancelled copy of this token.
        """

        cleaned = reason.strip()

        if not cleaned:
            raise ValueError("cancel reason cannot be empty.")

        return self.model_copy(
            update={
                "cancel_requested": True,
                "reason": cleaned,
                "cancelled_at": utc_now(),
            }
        )


class BackgroundTaskRequest(OrchestrationModel):
    """
    User/runtime request to register controlled background work.
    """

    background_task_id: str = Field(default_factory=new_background_task_id)
    kind: BackgroundTaskKind
    name: str
    description: str
    resource_budgets: tuple[ResourceBudget, ...] = Field(
        default_factory=lambda: (worker_slots(1),)
    )
    enabled: bool = True
    cancellable: bool = True
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("background_task_id", "name", "description")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("background_task_id")
    @classmethod
    def _validate_background_task_id(cls, value: str) -> str:
        if not value.startswith("bgtask_"):
            raise ValueError("background_task_id must start with 'bgtask_'.")

        return value

    @model_validator(mode="after")
    def _validate_background_request(self) -> BackgroundTaskRequest:
        if not self.cancellable:
            raise ValueError("background tasks must be cancellable.")

        if not self.resource_budgets:
            raise ValueError("background tasks require resource budgets.")

        return self


class BackgroundTaskDescriptor(OrchestrationModel):
    """
    Registered background task descriptor.

    Descriptor stores the controlled runtime object. It does not execute.
    """

    request: BackgroundTaskRequest
    task: TaskRequest
    cancellation_token: BackgroundCancellationToken
    status: BackgroundTaskStatus = BackgroundTaskStatus.REGISTERED
    registered_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_descriptor(self) -> BackgroundTaskDescriptor:
        if self.task.priority != TaskPriority.BACKGROUND:
            raise ValueError("background tasks must use BACKGROUND priority.")

        if not self.task.background:
            raise ValueError("background task request must be marked background.")

        if self.cancellation_token.task_id != self.task.task_id:
            raise ValueError("cancellation token task_id must match task id.")

        return self

    @property
    def task_id(self) -> TaskId:
        return self.task.task_id

    @property
    def kind(self) -> BackgroundTaskKind:
        return self.request.kind

    @property
    def enabled(self) -> bool:
        return self.request.enabled

    @property
    def cancelled(self) -> bool:
        return self.cancellation_token.cancel_requested


class BackgroundTaskRuntimeResult(OrchestrationModel):
    """
    Result of a background runtime operation.
    """

    decision: BackgroundTaskDecision
    reason: BackgroundTaskReason
    success: bool
    message: str
    task_id: TaskId | None = None
    descriptor: BackgroundTaskDescriptor | None = None
    schedule_result: TaskScheduleResult | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("task_id")
    @classmethod
    def _validate_task_id(cls, value: str | None) -> str | None:
        if value is None:
            return None

        return validate_task_id(value)

    @field_validator("message")
    @classmethod
    def _required_message(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("message cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class BackgroundTaskRuntimeConfig:
    """
    Background task runtime configuration.
    """

    name: str = "background_task_runtime"
    max_schedule_per_cycle: int = 4

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.max_schedule_per_cycle <= 0:
            raise ValueError("max_schedule_per_cycle must be positive.")


@dataclass(frozen=True, slots=True)
class BackgroundTaskRuntimeSnapshot:
    """
    Background task runtime diagnostics.
    """

    name: str
    registered_count: int
    scheduled_count: int
    yielded_count: int
    shed_count: int
    cancelled_count: int
    rejected_count: int
    last_reason: BackgroundTaskReason | None


class BackgroundTaskRuntime:
    """
    Phase 6 Background Task Runtime.

    Responsibilities:
    - register safe background tasks
    - force all background tasks to lowest priority
    - yield immediately when attention protects conversation or speech
    - shed first under LOAD_SHEDDING
    - route scheduling through TaskScheduler
    - keep cancellation explicit and observable

    Non-responsibilities:
    - no task execution
    - no direct worker calls
    - no hidden maintenance loops
    - no bypass of attention, budget, scheduler, or state machine
    """

    def __init__(
        self,
        *,
        scheduler: TaskScheduler,
        attention: AttentionRuntime,
        state_machine: OrchestrationStateMachine,
        config: BackgroundTaskRuntimeConfig | None = None,
    ) -> None:
        self._config = config or BackgroundTaskRuntimeConfig()
        self._config.validate()

        self._scheduler = scheduler
        self._attention = attention
        self._state_machine = state_machine
        self._tasks: dict[TaskId, BackgroundTaskDescriptor] = {}
        self._lock = RLock()

        self._scheduled_count = 0
        self._yielded_count = 0
        self._shed_count = 0
        self._cancelled_count = 0
        self._rejected_count = 0
        self._last_reason: BackgroundTaskReason | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def register(
        self,
        request: BackgroundTaskRequest,
    ) -> BackgroundTaskRuntimeResult:
        """
        Register a controlled background task.
        """

        task = self._task_from_request(request)
        token = BackgroundCancellationToken(task_id=task.task_id)
        descriptor = BackgroundTaskDescriptor(
            request=request,
            task=task,
            cancellation_token=token,
        )

        with self._lock:
            if descriptor.task_id in self._tasks:
                result = self._result(
                    decision=BackgroundTaskDecision.REJECTED,
                    reason=BackgroundTaskReason.TASK_ALREADY_REGISTERED,
                    success=False,
                    message="background task already registered",
                    task_id=descriptor.task_id,
                    descriptor=self._tasks[descriptor.task_id],
                )
                self._record(result)

                return result

            self._tasks[descriptor.task_id] = descriptor

        result = self._result(
            decision=BackgroundTaskDecision.ACCEPTED,
            reason=BackgroundTaskReason.TASK_REGISTERED,
            success=True,
            message="background task registered",
            task_id=descriptor.task_id,
            descriptor=descriptor,
        )
        self._record(result)

        return result

    def schedule_cycle(self) -> tuple[BackgroundTaskRuntimeResult, ...]:
        """
        Attempt to schedule background tasks for one controlled cycle.
        """

        with self._lock:
            descriptors = tuple(self._tasks.values())

        runnable = tuple(
            descriptor
            for descriptor in descriptors
            if descriptor.enabled
            and not descriptor.cancelled
            and descriptor.status
            not in {
                BackgroundTaskStatus.SCHEDULED,
                BackgroundTaskStatus.CANCELLED,
                BackgroundTaskStatus.SHED,
                BackgroundTaskStatus.DISABLED,
            }
        )
        limited = runnable[: self._config.max_schedule_per_cycle]

        return tuple(self.schedule_one(descriptor.task_id) for descriptor in limited)

    def schedule_one(self, task_id: TaskId) -> BackgroundTaskRuntimeResult:
        """
        Schedule one background task if safe.

        The scheduler still only proposes; workers execute later.
        """

        descriptor = self.descriptor_for(task_id)

        if descriptor is None:
            result = self._result(
                decision=BackgroundTaskDecision.REJECTED,
                reason=BackgroundTaskReason.TASK_NOT_FOUND,
                success=False,
                message="background task not found",
                task_id=task_id,
            )
            self._record(result)

            return result

        if not descriptor.enabled:
            return self._mark(
                descriptor=descriptor,
                status=BackgroundTaskStatus.DISABLED,
                decision=BackgroundTaskDecision.REJECTED,
                reason=BackgroundTaskReason.TASK_DISABLED,
                success=False,
                message="background task disabled",
            )

        if descriptor.cancelled:
            return self._mark(
                descriptor=descriptor,
                status=BackgroundTaskStatus.CANCELLED,
                decision=BackgroundTaskDecision.CANCELLED,
                reason=BackgroundTaskReason.TASK_CANCELLED,
                success=False,
                message="background task cancelled",
            )

        if self._state_machine.state.state == OrchestratorState.LOAD_SHEDDING:
            return self._mark(
                descriptor=descriptor,
                status=BackgroundTaskStatus.SHED,
                decision=BackgroundTaskDecision.SHED,
                reason=BackgroundTaskReason.TASK_SHED_DURING_LOAD_SHEDDING,
                success=False,
                message="background task shed during load shedding",
            )

        attention_result = self._attention.evaluate(descriptor.task)

        if attention_result.decision in {
            AttentionDecision.DEFER,
            AttentionDecision.SUPPRESS,
        }:
            return self._mark(
                descriptor=descriptor,
                status=BackgroundTaskStatus.YIELDED,
                decision=BackgroundTaskDecision.YIELDED,
                reason=BackgroundTaskReason.TASK_YIELDED_TO_ATTENTION,
                success=False,
                message="background task yielded to attention runtime",
                metadata={"attention_reason": attention_result.reason.value},
            )

        graph = TaskGraph(
            job=JobRequest(
                name=descriptor.request.name,
                description=descriptor.request.description,
                tasks=(descriptor.task,),
            )
        )
        batch = self._scheduler.schedule_ready(graph)

        if batch.scheduled_count > 0:
            schedule_result = batch.scheduled[0]

            return self._mark(
                descriptor=descriptor,
                status=BackgroundTaskStatus.SCHEDULED,
                decision=BackgroundTaskDecision.SCHEDULED,
                reason=BackgroundTaskReason.TASK_SCHEDULED,
                success=True,
                message="background task scheduled",
                schedule_result=schedule_result,
            )

        if batch.denied_count > 0:
            schedule_result = batch.results[0]

            return self._mark(
                descriptor=descriptor,
                status=BackgroundTaskStatus.YIELDED,
                decision=BackgroundTaskDecision.YIELDED,
                reason=BackgroundTaskReason.SCHEDULER_DENIED,
                success=False,
                message="scheduler denied background task",
                schedule_result=schedule_result,
            )

        if batch.deferred_count > 0:
            schedule_result = batch.results[0]

            return self._mark(
                descriptor=descriptor,
                status=BackgroundTaskStatus.YIELDED,
                decision=BackgroundTaskDecision.YIELDED,
                reason=BackgroundTaskReason.SCHEDULER_DEFERRED,
                success=False,
                message="scheduler deferred background task",
                schedule_result=schedule_result,
            )

        return self._mark(
            descriptor=descriptor,
            status=BackgroundTaskStatus.YIELDED,
            decision=BackgroundTaskDecision.YIELDED,
            reason=BackgroundTaskReason.SCHEDULER_DEFERRED,
            success=False,
            message="no background task was scheduled",
        )

    def cancel(
        self,
        task_id: TaskId,
        *,
        reason: str = "background task cancelled",
    ) -> BackgroundTaskRuntimeResult:
        """
        Cancel one background task.
        """

        descriptor = self.descriptor_for(task_id)

        if descriptor is None:
            result = self._result(
                decision=BackgroundTaskDecision.REJECTED,
                reason=BackgroundTaskReason.TASK_NOT_FOUND,
                success=False,
                message="background task not found",
                task_id=task_id,
            )
            self._record(result)

            return result

        updated = descriptor.model_copy(
            update={
                "status": BackgroundTaskStatus.CANCELLED,
                "cancellation_token": (
                    descriptor.cancellation_token.request_cancel(reason)
                ),
                "updated_at": utc_now(),
            }
        )

        with self._lock:
            self._tasks[descriptor.task_id] = updated

        result = self._result(
            decision=BackgroundTaskDecision.CANCELLED,
            reason=BackgroundTaskReason.TASK_CANCELLED,
            success=True,
            message="background task cancelled",
            task_id=descriptor.task_id,
            descriptor=updated,
        )
        self._record(result)

        return result

    def shed_all(self) -> tuple[BackgroundTaskRuntimeResult, ...]:
        """
        Shed all non-cancelled background tasks.

        Background is first to shed under pressure.
        """

        with self._lock:
            descriptors = tuple(self._tasks.values())

        return tuple(
            self._mark(
                descriptor=descriptor,
                status=BackgroundTaskStatus.SHED,
                decision=BackgroundTaskDecision.SHED,
                reason=BackgroundTaskReason.TASK_SHED_DURING_LOAD_SHEDDING,
                success=False,
                message="background task shed",
            )
            for descriptor in descriptors
            if not descriptor.cancelled
        )

    def descriptor_for(self, task_id: TaskId) -> BackgroundTaskDescriptor | None:
        validated_task_id = validate_task_id(task_id)

        with self._lock:
            return self._tasks.get(validated_task_id)

    def descriptors(self) -> tuple[BackgroundTaskDescriptor, ...]:
        with self._lock:
            return tuple(self._tasks.values())

    def snapshot(self) -> BackgroundTaskRuntimeSnapshot:
        with self._lock:
            return BackgroundTaskRuntimeSnapshot(
                name=self.name,
                registered_count=len(self._tasks),
                scheduled_count=self._scheduled_count,
                yielded_count=self._yielded_count,
                shed_count=self._shed_count,
                cancelled_count=self._cancelled_count,
                rejected_count=self._rejected_count,
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        with self._lock:
            self._tasks.clear()
            self._scheduled_count = 0
            self._yielded_count = 0
            self._shed_count = 0
            self._cancelled_count = 0
            self._rejected_count = 0
            self._last_reason = BackgroundTaskReason.RUNTIME_RESET

    @staticmethod
    def _task_from_request(request: BackgroundTaskRequest) -> TaskRequest:
        kind = {
            BackgroundTaskKind.MEMORY_CONSOLIDATION: (
                TaskKind.BACKGROUND_MAINTENANCE
            ),
            BackgroundTaskKind.CONTEXT_PREFETCH: TaskKind.BACKGROUND_MAINTENANCE,
            BackgroundTaskKind.WORKSPACE_SCAN: TaskKind.BACKGROUND_MAINTENANCE,
            BackgroundTaskKind.HEALTH_CHECK: TaskKind.HEALTH_CHECK,
            BackgroundTaskKind.LOG_ROTATION: TaskKind.BACKGROUND_MAINTENANCE,
            BackgroundTaskKind.BACKGROUND_SUMMARY: TaskKind.BACKGROUND_MAINTENANCE,
        }[request.kind]

        capability = (
            WorkerCapability.OBSERVABILITY
            if request.kind == BackgroundTaskKind.HEALTH_CHECK
            else WorkerCapability.BACKGROUND
        )

        return TaskRequest(
            kind=kind,
            priority=TaskPriority.BACKGROUND,
            name=request.name,
            description=request.description,
            required_capabilities=(capability,),
            resource_budgets=request.resource_budgets,
            background=True,
            interruptible=True,
            metadata={
                "background_task_id": request.background_task_id,
                "background_kind": request.kind.value,
            },
        )

    def _mark(
        self,
        *,
        descriptor: BackgroundTaskDescriptor,
        status: BackgroundTaskStatus,
        decision: BackgroundTaskDecision,
        reason: BackgroundTaskReason,
        success: bool,
        message: str,
        schedule_result: TaskScheduleResult | None = None,
        metadata: dict[str, object] | None = None,
    ) -> BackgroundTaskRuntimeResult:
        updated = descriptor.model_copy(
            update={
                "status": status,
                "updated_at": utc_now(),
            }
        )

        with self._lock:
            self._tasks[descriptor.task_id] = updated

        result = self._result(
            decision=decision,
            reason=reason,
            success=success,
            message=message,
            task_id=descriptor.task_id,
            descriptor=updated,
            schedule_result=schedule_result,
            metadata=metadata,
        )
        self._record(result)

        return result

    @staticmethod
    def _result(
        *,
        decision: BackgroundTaskDecision,
        reason: BackgroundTaskReason,
        success: bool,
        message: str,
        task_id: TaskId | None = None,
        descriptor: BackgroundTaskDescriptor | None = None,
        schedule_result: TaskScheduleResult | None = None,
        metadata: dict[str, object] | None = None,
    ) -> BackgroundTaskRuntimeResult:
        return BackgroundTaskRuntimeResult(
            decision=decision,
            reason=reason,
            success=success,
            message=message,
            task_id=task_id,
            descriptor=descriptor,
            schedule_result=schedule_result,
            metadata=metadata or {},
        )

    def _record(self, result: BackgroundTaskRuntimeResult) -> None:
        self._last_reason = result.reason

        if result.decision == BackgroundTaskDecision.SCHEDULED:
            self._scheduled_count += 1

        elif result.decision == BackgroundTaskDecision.YIELDED:
            self._yielded_count += 1

        elif result.decision == BackgroundTaskDecision.SHED:
            self._shed_count += 1

        elif result.decision == BackgroundTaskDecision.CANCELLED:
            self._cancelled_count += 1

        elif result.decision == BackgroundTaskDecision.REJECTED:
            self._rejected_count += 1


def memory_consolidation_task() -> BackgroundTaskRequest:
    return BackgroundTaskRequest(
        kind=BackgroundTaskKind.MEMORY_CONSOLIDATION,
        name="memory consolidation",
        description="Consolidate memory in the background.",
    )


def context_prefetch_task() -> BackgroundTaskRequest:
    return BackgroundTaskRequest(
        kind=BackgroundTaskKind.CONTEXT_PREFETCH,
        name="context prefetch",
        description="Prepare likely context in the background.",
    )


def workspace_scan_task() -> BackgroundTaskRequest:
    return BackgroundTaskRequest(
        kind=BackgroundTaskKind.WORKSPACE_SCAN,
        name="workspace scan",
        description="Scan workspace metadata in the background.",
    )


def health_check_task() -> BackgroundTaskRequest:
    return BackgroundTaskRequest(
        kind=BackgroundTaskKind.HEALTH_CHECK,
        name="health check",
        description="Check runtime health in the background.",
    )


def log_rotation_task() -> BackgroundTaskRequest:
    return BackgroundTaskRequest(
        kind=BackgroundTaskKind.LOG_ROTATION,
        name="log rotation",
        description="Rotate logs in the background.",
    )