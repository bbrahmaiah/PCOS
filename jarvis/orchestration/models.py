from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from jarvis.orchestration.ids import (
    JobId,
    OrchestrationId,
    TaskId,
    WorkerId,
    new_job_id,
    new_orchestration_id,
    new_task_id,
    new_worker_id,
    utc_now,
    validate_job_id,
    validate_orchestration_id,
    validate_task_id,
    validate_worker_id,
)


class OrchestrationModel(BaseModel):
    """
    Base immutable model for Phase 6 orchestration contracts.

    These models are contracts, not runtime side-effect objects.
    Runtime systems may create new versions, but should not mutate these
    objects invisibly.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,
    )


class TaskKind(StrEnum):
    """
    Type of work the orchestration runtime may coordinate.
    """

    CONVERSATION_TURN = "conversation_turn"
    PRESENCE_EVENT = "presence_event"
    COGNITION = "cognition"
    MEMORY_RETRIEVAL = "memory_retrieval"
    MEMORY_WRITE = "memory_write"
    TOOL_ACTION = "tool_action"
    BACKGROUND_MAINTENANCE = "background_maintenance"
    HEALTH_CHECK = "health_check"
    RECOVERY = "recovery"
    OBSERVABILITY = "observability"
    SYSTEM = "system"


class TaskPriority(IntEnum):
    """
    Scheduling priority.

    Lower numeric value means higher priority.
    """

    CRITICAL = 0
    HIGH = 10
    NORMAL = 20
    LOW = 30
    BACKGROUND = 40

    @property
    def is_foreground(self) -> bool:
        return self in {
            TaskPriority.CRITICAL,
            TaskPriority.HIGH,
            TaskPriority.NORMAL,
        }

    @property
    def is_background(self) -> bool:
        return self == TaskPriority.BACKGROUND


class TaskStatus(StrEnum):
    """
    Formal lifecycle for orchestrated tasks.
    """

    CREATED = "created"
    PENDING = "pending"
    SCHEDULED = "scheduled"
    WAITING_DEPENDENCIES = "waiting_dependencies"
    READY = "ready"
    RUNNING = "running"
    PAUSING = "pausing"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    BLOCKED = "blocked"

    @property
    def terminal(self) -> bool:
        return self in {
            TaskStatus.CANCELLED,
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.TIMED_OUT,
            TaskStatus.BLOCKED,
        }

    @property
    def active(self) -> bool:
        return self in {
            TaskStatus.SCHEDULED,
            TaskStatus.WAITING_DEPENDENCIES,
            TaskStatus.READY,
            TaskStatus.RUNNING,
            TaskStatus.PAUSING,
            TaskStatus.CANCELLING,
        }


class WorkerState(StrEnum):
    """
    Worker lifecycle state.
    """

    REGISTERED = "registered"
    STARTING = "starting"
    IDLE = "idle"
    BUSY = "busy"
    DRAINING = "draining"
    STOPPED = "stopped"
    FAILED = "failed"
    UNHEALTHY = "unhealthy"


class WorkerCapability(StrEnum):
    """
    Capabilities workers may expose to the Orchestration Kernel.
    """

    PRESENCE = "presence"
    CONVERSATION = "conversation"
    COGNITION = "cognition"
    MEMORY = "memory"
    TOOL_ACTION = "tool_action"
    ATTENTION = "attention"
    SCHEDULING = "scheduling"
    BACKGROUND = "background"
    RECOVERY = "recovery"
    OBSERVABILITY = "observability"


class ResourceKind(StrEnum):
    """
    Resource classes governed by orchestration budgets.
    """

    CPU_SLOT = "cpu_slot"
    WORKER_SLOT = "worker_slot"
    LLM_TOKEN = "llm_token"
    MEMORY_QUERY = "memory_query"
    MEMORY_WRITE = "memory_write"
    ACTION_SLOT = "action_slot"
    FILE_OPERATION = "file_operation"
    NETWORK_OPERATION = "network_operation"
    WALL_TIME_MS = "wall_time_ms"


class BudgetPolicy(StrEnum):
    """
    Resource budget enforcement mode.
    """

    ENFORCE = "enforce"
    WARN = "warn"
    SHED = "shed"


class OrchestratorState(StrEnum):
    """
    Formal Orchestration Kernel state.
    """

    STARTING = "starting"
    IDLE = "idle"
    COORDINATING = "coordinating"
    BUSY = "busy"
    LOAD_SHEDDING = "load_shedding"
    RECOVERING = "recovering"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


class TaskDependency(OrchestrationModel):
    """
    Dependency from one task to another.

    A task may run only when all dependencies reach allowed terminal states.
    """

    task_id: TaskId
    required_statuses: tuple[TaskStatus, ...] = (TaskStatus.SUCCEEDED,)

    @field_validator("task_id")
    @classmethod
    def _validate_task_id(cls, value: str) -> str:
        return validate_task_id(value)

    @model_validator(mode="after")
    def _validate_required_statuses(self) -> TaskDependency:
        if not self.required_statuses:
            raise ValueError("dependency requires at least one accepted status.")

        if not all(status.terminal for status in self.required_statuses):
            raise ValueError("dependency statuses must be terminal states.")

        return self


class ResourceBudget(OrchestrationModel):
    """
    Declared resource cost or limit.

    Step 0 law:
    no scheduled task may consume undeclared resources.
    """

    resource: ResourceKind
    amount: int = Field(gt=0)
    policy: BudgetPolicy = BudgetPolicy.ENFORCE
    window_seconds: int | None = Field(default=None, gt=0)
    reserved_for_conversation: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    def can_cover(self, requested: int) -> bool:
        """
        Return whether this budget can cover a requested amount.
        """

        return requested <= self.amount


class TaskRequest(OrchestrationModel):
    """
    Atomic unit of orchestrated work.

    A task is a runtime object. It does not execute itself.
    """

    task_id: TaskId = Field(default_factory=new_task_id)
    job_id: JobId | None = None
    kind: TaskKind
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.CREATED
    name: str
    description: str
    requested_by: str = "system"
    required_capabilities: tuple[WorkerCapability, ...]
    resource_budgets: tuple[ResourceBudget, ...]
    dependencies: tuple[TaskDependency, ...] = ()
    timeout_ms: int | None = Field(default=None, gt=0)
    interruptible: bool = True
    background: bool = False
    idempotent: bool = False
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("task_id")
    @classmethod
    def _validate_task_id(cls, value: str) -> str:
        return validate_task_id(value)

    @field_validator("job_id")
    @classmethod
    def _validate_job_id(cls, value: str | None) -> str | None:
        if value is None:
            return None

        return validate_job_id(value)

    @field_validator("name", "description", "requested_by")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @model_validator(mode="after")
    def _validate_contract(self) -> TaskRequest:
        if not self.required_capabilities:
            raise ValueError("task requires at least one worker capability.")

        if not self.resource_budgets:
            raise ValueError("task must declare resource budgets.")

        if self.background and self.priority != TaskPriority.BACKGROUND:
            raise ValueError("background tasks must use BACKGROUND priority.")

        if self.kind == TaskKind.CONVERSATION_TURN and self.background:
            raise ValueError("conversation tasks cannot be background tasks.")

        if self.kind == TaskKind.CONVERSATION_TURN:
            if self.priority not in {TaskPriority.CRITICAL, TaskPriority.HIGH}:
                raise ValueError("conversation tasks require high priority.")

        if self.priority in {TaskPriority.CRITICAL, TaskPriority.HIGH}:
            if self.timeout_ms is None:
                raise ValueError("high priority tasks require timeout_ms.")

        return self

    @property
    def ready_without_dependencies(self) -> bool:
        return not self.dependencies


class TaskResult(OrchestrationModel):
    """
    Observable result of an orchestrated task.
    """

    task_id: TaskId
    status: TaskStatus
    success: bool
    output: str = ""
    error: str | None = None
    worker_id: WorkerId | None = None
    started_at: object | None = None
    completed_at: object = Field(default_factory=utc_now)
    duration_ms: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

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

    @model_validator(mode="after")
    def _validate_result_shape(self) -> TaskResult:
        if not self.status.terminal:
            raise ValueError("task result must use a terminal status.")

        if self.success and self.status != TaskStatus.SUCCEEDED:
            raise ValueError("successful result must have SUCCEEDED status.")

        if not self.success and self.status == TaskStatus.SUCCEEDED:
            raise ValueError("failed result cannot use SUCCEEDED status.")

        return self


class JobRequest(OrchestrationModel):
    """
    Group of tasks coordinated as a job.

    Jobs are graphs. They do not execute themselves.
    """

    job_id: JobId = Field(default_factory=new_job_id)
    orchestration_id: OrchestrationId = Field(
        default_factory=new_orchestration_id
    )
    name: str
    description: str
    tasks: tuple[TaskRequest, ...]
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("job_id")
    @classmethod
    def _validate_job_id(cls, value: str) -> str:
        return validate_job_id(value)

    @field_validator("orchestration_id")
    @classmethod
    def _validate_orchestration_id(cls, value: str) -> str:
        return validate_orchestration_id(value)

    @field_validator("name", "description")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @model_validator(mode="after")
    def _validate_job_graph(self) -> JobRequest:
        if not self.tasks:
            raise ValueError("job requires at least one task.")

        task_ids = {task.task_id for task in self.tasks}

        if len(task_ids) != len(self.tasks):
            raise ValueError("job task ids must be unique.")

        for task in self.tasks:
            if task.job_id is not None and task.job_id != self.job_id:
                raise ValueError("task job_id must match parent job_id.")

            for dependency in task.dependencies:
                if dependency.task_id == task.task_id:
                    raise ValueError("task cannot depend on itself.")

                if dependency.task_id not in task_ids:
                    raise ValueError("task dependency must exist in same job.")

        self._validate_acyclic(task_ids=task_ids)

        return self

    def _validate_acyclic(self, *, task_ids: set[str]) -> None:
        adjacency: dict[str, set[str]] = {task_id: set() for task_id in task_ids}

        for task in self.tasks:
            for dependency in task.dependencies:
                adjacency[task.task_id].add(dependency.task_id)

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise ValueError("job task dependency graph cannot contain cycles.")

            if task_id in visited:
                return

            visiting.add(task_id)

            for dependency_id in adjacency[task_id]:
                visit(dependency_id)

            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in task_ids:
            visit(task_id)


class JobResult(OrchestrationModel):
    """
    Observable result of an orchestrated job.
    """

    job_id: JobId
    status: TaskStatus
    success: bool
    task_results: tuple[TaskResult, ...]
    completed_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("job_id")
    @classmethod
    def _validate_job_id(cls, value: str) -> str:
        return validate_job_id(value)

    @model_validator(mode="after")
    def _validate_job_result(self) -> JobResult:
        if not self.status.terminal:
            raise ValueError("job result must use a terminal status.")

        if self.success and self.status != TaskStatus.SUCCEEDED:
            raise ValueError("successful job must have SUCCEEDED status.")

        if self.success:
            if not self.task_results:
                raise ValueError("successful job requires task results.")

            if not all(result.success for result in self.task_results):
                raise ValueError("successful job requires all tasks to succeed.")

        return self


class WorkerContract(OrchestrationModel):
    """
    Registered worker contract.

    Workers execute. The Orchestration Kernel coordinates.
    """

    worker_id: WorkerId = Field(default_factory=new_worker_id)
    name: str
    state: WorkerState = WorkerState.REGISTERED
    capabilities: tuple[WorkerCapability, ...]
    accepted_task_kinds: tuple[TaskKind, ...]
    max_concurrent_tasks: int = Field(default=1, gt=0)
    resource_budgets: tuple[ResourceBudget, ...]
    restartable: bool = True
    heartbeat_interval_ms: int = Field(default=5_000, gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("worker_id")
    @classmethod
    def _validate_worker_id(cls, value: str) -> str:
        return validate_worker_id(value)

    @field_validator("name")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("worker name cannot be empty.")

        return cleaned

    @model_validator(mode="after")
    def _validate_worker_contract(self) -> WorkerContract:
        if not self.capabilities:
            raise ValueError("worker requires at least one capability.")

        if not self.accepted_task_kinds:
            raise ValueError("worker requires accepted task kinds.")

        if not self.resource_budgets:
            raise ValueError("worker requires resource budgets.")

        return self

    def can_accept(self, task: TaskRequest) -> bool:
        """
        Return whether this worker contract can accept the task type and
        capability requirements.
        """

        capabilities = set(self.capabilities)

        return (
            task.kind in self.accepted_task_kinds
            and set(task.required_capabilities).issubset(capabilities)
        )


class OrchestrationSnapshot(OrchestrationModel):
    """
    Minimal immutable snapshot of orchestration state.

    Later Phase 6 steps will expand this into live runtime diagnostics.
    """

    orchestration_id: OrchestrationId = Field(
        default_factory=new_orchestration_id
    )
    state: OrchestratorState
    active_task_count: int = Field(default=0, ge=0)
    active_job_count: int = Field(default=0, ge=0)
    registered_worker_count: int = Field(default=0, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("orchestration_id")
    @classmethod
    def _validate_orchestration_id(cls, value: str) -> str:
        return validate_orchestration_id(value)