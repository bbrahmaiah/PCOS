from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.orchestration.ids import (
    TaskId,
    WorkerId,
    utc_now,
    validate_task_id,
    validate_worker_id,
)
from jarvis.orchestration.models import (
    OrchestrationModel,
    TaskRequest,
    TaskResult,
    TaskStatus,
)
from jarvis.orchestration.registry import (
    WorkerHealthState,
    WorkerRegistry,
)
from jarvis.orchestration.scheduler import ScheduledTask


def new_coordination_message_id() -> str:
    return f"coordmsg_{uuid4().hex}"


def new_task_assignment_id() -> str:
    return f"assignment_{uuid4().hex}"


class CoordinationActorKind(StrEnum):
    """
    Actor category in the worker coordination protocol.
    """

    ORCHESTRATOR = "orchestrator"
    WORKER = "worker"


class WorkerCoordinationMessageKind(StrEnum):
    """
    Typed worker coordination message kinds.
    """

    TASK_ASSIGNED = "task_assigned"
    TASK_STARTED = "task_started"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    WORKER_HEALTH = "worker_health"


class WorkerCoordinationDecision(StrEnum):
    """
    Worker coordination decision.
    """

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    RECORDED = "recorded"


class WorkerCoordinationReason(StrEnum):
    """
    Machine-readable worker coordination reason.
    """

    TASK_ASSIGNED = "task_assigned"
    TASK_STARTED = "task_started"
    TASK_PROGRESS_RECORDED = "task_progress_recorded"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    WORKER_HEALTH_RECORDED = "worker_health_recorded"
    UNKNOWN_WORKER = "unknown_worker"
    TASK_NOT_ASSIGNED = "task_not_assigned"
    DUPLICATE_ASSIGNMENT = "duplicate_assignment"
    WORKER_MISMATCH = "worker_mismatch"
    DIRECT_WORKER_MESSAGE_REJECTED = "direct_worker_message_rejected"
    INVALID_MESSAGE_DIRECTION = "invalid_message_direction"
    RESULT_ALREADY_RECORDED = "result_already_recorded"


class WorkerCoordinationMessage(OrchestrationModel):
    """
    Typed coordination message.

    Workers may publish results to the orchestrator.
    The orchestrator may assign tasks to workers.
    Workers must never communicate directly with each other.
    """

    message_id: str = Field(default_factory=new_coordination_message_id)
    kind: WorkerCoordinationMessageKind
    sender_kind: CoordinationActorKind
    sender_id: str
    receiver_kind: CoordinationActorKind
    receiver_id: str
    task_id: TaskId | None = None
    worker_id: WorkerId | None = None
    payload: dict[str, object] = Field(default_factory=dict)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("message_id", "sender_id", "receiver_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("message_id")
    @classmethod
    def _validate_message_id(cls, value: str) -> str:
        if not value.startswith("coordmsg_"):
            raise ValueError("message_id must start with 'coordmsg_'.")

        return value

    @field_validator("task_id")
    @classmethod
    def _validate_task_id(cls, value: str | None) -> str | None:
        if value is None:
            return None

        return validate_task_id(value)

    @field_validator("worker_id")
    @classmethod
    def _validate_worker_id(cls, value: str | None) -> str | None:
        if value is None:
            return None

        return validate_worker_id(value)

    @model_validator(mode="after")
    def _validate_message_direction(self) -> WorkerCoordinationMessage:
        if (
            self.sender_kind == CoordinationActorKind.WORKER
            and self.receiver_kind == CoordinationActorKind.WORKER
        ):
            raise ValueError("worker-to-worker messages are forbidden.")

        if self.kind == WorkerCoordinationMessageKind.TASK_ASSIGNED:
            if self.sender_kind != CoordinationActorKind.ORCHESTRATOR:
                raise ValueError("task assignment must come from orchestrator.")

            if self.receiver_kind != CoordinationActorKind.WORKER:
                raise ValueError("task assignment must target a worker.")

        if self.kind in {
            WorkerCoordinationMessageKind.TASK_STARTED,
            WorkerCoordinationMessageKind.TASK_PROGRESS,
            WorkerCoordinationMessageKind.TASK_COMPLETED,
            WorkerCoordinationMessageKind.TASK_FAILED,
            WorkerCoordinationMessageKind.WORKER_HEALTH,
        }:
            if self.sender_kind != CoordinationActorKind.WORKER:
                raise ValueError("worker status messages must come from worker.")

            if self.receiver_kind != CoordinationActorKind.ORCHESTRATOR:
                raise ValueError("worker status messages must target orchestrator.")

        return self


class TaskAssignment(OrchestrationModel):
    """
    Assignment contract sent from orchestrator to one worker.

    This is not execution. It is a typed handoff.
    """

    assignment_id: str = Field(default_factory=new_task_assignment_id)
    scheduled_task: ScheduledTask
    task: TaskRequest
    assigned_by: str = "orchestration_kernel"
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("assignment_id", "assigned_by")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("assignment_id")
    @classmethod
    def _validate_assignment_id(cls, value: str) -> str:
        if not value.startswith("assignment_"):
            raise ValueError("assignment_id must start with 'assignment_'.")

        return value

    @model_validator(mode="after")
    def _validate_assignment(self) -> TaskAssignment:
        if self.scheduled_task.task_id != self.task.task_id:
            raise ValueError("scheduled task id must match task id.")

        return self

    @property
    def task_id(self) -> TaskId:
        return self.task.task_id

    @property
    def worker_id(self) -> WorkerId:
        return self.scheduled_task.worker_id


class WorkerProgressUpdate(OrchestrationModel):
    """
    Progress update emitted by a worker.
    """

    task_id: TaskId
    worker_id: WorkerId
    percent: int = Field(ge=0, le=100)
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("task_id")
    @classmethod
    def _validate_task_id(cls, value: str) -> str:
        return validate_task_id(value)

    @field_validator("worker_id")
    @classmethod
    def _validate_worker_id(cls, value: str) -> str:
        return validate_worker_id(value)

    @field_validator("message")
    @classmethod
    def _required_message(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("message cannot be empty.")

        return cleaned


class WorkerHealthBroadcast(OrchestrationModel):
    """
    Health broadcast emitted by a worker.
    """

    worker_id: WorkerId
    health: WorkerHealthState
    active_tasks: int = Field(default=0, ge=0)
    queued_tasks: int = Field(default=0, ge=0)
    message: str = "worker health broadcast"
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("worker_id")
    @classmethod
    def _validate_worker_id(cls, value: str) -> str:
        return validate_worker_id(value)

    @field_validator("message")
    @classmethod
    def _required_message(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("message cannot be empty.")

        return cleaned


class ResultCollection(OrchestrationModel):
    """
    Immutable collection of worker task results.
    """

    results: tuple[TaskResult, ...] = ()

    @property
    def task_ids(self) -> set[TaskId]:
        return {result.task_id for result in self.results}

    def contains(self, task_id: TaskId) -> bool:
        return validate_task_id(task_id) in self.task_ids

    def add(self, result: TaskResult) -> ResultCollection:
        if result.task_id in self.task_ids:
            raise ValueError("task result already recorded.")

        return self.model_copy(update={"results": self.results + (result,)})


class WorkerCoordinationResult(OrchestrationModel):
    """
    Result of a worker coordination operation.
    """

    decision: WorkerCoordinationDecision
    reason: WorkerCoordinationReason
    success: bool
    message: str
    coordination_message: WorkerCoordinationMessage | None = None
    assignment: TaskAssignment | None = None
    task_result: TaskResult | None = None
    progress: WorkerProgressUpdate | None = None
    health: WorkerHealthBroadcast | None = None
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
class WorkerCoordinatorConfig:
    """
    Worker coordinator configuration.
    """

    name: str = "worker_coordinator"

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")


@dataclass(frozen=True, slots=True)
class WorkerCoordinatorSnapshot:
    """
    Worker coordinator diagnostics.
    """

    name: str
    assignment_count: int
    message_count: int
    result_count: int
    progress_count: int
    health_count: int
    rejected_count: int
    last_reason: WorkerCoordinationReason | None


class WorkerCoordinator:
    """
    Phase 6 Worker Coordination Protocol.

    Responsibilities:
    - convert scheduler assignments into typed worker messages
    - collect typed worker progress, completion, failure, and health messages
    - reject worker-to-worker communication
    - keep workers isolated behind orchestration messages

    Non-responsibilities:
    - no task execution
    - no direct worker implementation
    - no scheduling decisions
    - no resource budgeting
    """

    def __init__(
        self,
        *,
        registry: WorkerRegistry,
        config: WorkerCoordinatorConfig | None = None,
    ) -> None:
        self._config = config or WorkerCoordinatorConfig()
        self._config.validate()

        self._registry = registry
        self._assignments: dict[TaskId, TaskAssignment] = {}
        self._messages: list[WorkerCoordinationMessage] = []
        self._progress: list[WorkerProgressUpdate] = []
        self._health: list[WorkerHealthBroadcast] = []
        self._results = ResultCollection()
        self._lock = RLock()

        self._rejected_count = 0
        self._last_reason: WorkerCoordinationReason | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def assign(
        self,
        *,
        scheduled_task: ScheduledTask,
        task: TaskRequest,
    ) -> WorkerCoordinationResult:
        """
        Send a typed assignment from orchestrator to worker.
        """

        with self._lock:
            if scheduled_task.task_id in self._assignments:
                return self._reject(
                    reason=WorkerCoordinationReason.DUPLICATE_ASSIGNMENT,
                    message="task already assigned",
                )

            lookup = self._registry.get(scheduled_task.worker_id)

            if lookup.descriptor is None:
                return self._reject(
                    reason=WorkerCoordinationReason.UNKNOWN_WORKER,
                    message="worker is not registered",
                )

            if not lookup.descriptor.contract.can_accept(task):
                return self._reject(
                    reason=WorkerCoordinationReason.INVALID_MESSAGE_DIRECTION,
                    message="worker contract cannot accept task",
                )

            assignment = TaskAssignment(
                scheduled_task=scheduled_task,
                task=task,
            )
            message = WorkerCoordinationMessage(
                kind=WorkerCoordinationMessageKind.TASK_ASSIGNED,
                sender_kind=CoordinationActorKind.ORCHESTRATOR,
                sender_id="orchestration_kernel",
                receiver_kind=CoordinationActorKind.WORKER,
                receiver_id=scheduled_task.worker_id,
                task_id=scheduled_task.task_id,
                worker_id=scheduled_task.worker_id,
                payload={
                    "assignment_id": assignment.assignment_id,
                    "task_kind": task.kind.value,
                },
            )
            self._assignments[assignment.task_id] = assignment
            self._messages.append(message)

            return self._accept(
                reason=WorkerCoordinationReason.TASK_ASSIGNED,
                message="task assignment message created",
                coordination_message=message,
                assignment=assignment,
            )

    def record_started(
        self,
        *,
        worker_id: WorkerId,
        task_id: TaskId,
    ) -> WorkerCoordinationResult:
        """
        Record worker task start event.
        """

        assignment = self._assignment_for_worker(
            worker_id=worker_id,
            task_id=task_id,
        )

        if isinstance(assignment, WorkerCoordinationResult):
            return assignment

        message = self._worker_message(
            kind=WorkerCoordinationMessageKind.TASK_STARTED,
            worker_id=worker_id,
            task_id=task_id,
            payload={"status": TaskStatus.RUNNING.value},
        )

        return self._record_message(
            message=message,
            reason=WorkerCoordinationReason.TASK_STARTED,
            result_message="task start recorded",
        )

    def record_progress(
        self,
        progress: WorkerProgressUpdate,
    ) -> WorkerCoordinationResult:
        """
        Record worker task progress event.
        """

        assignment = self._assignment_for_worker(
            worker_id=progress.worker_id,
            task_id=progress.task_id,
        )

        if isinstance(assignment, WorkerCoordinationResult):
            return assignment

        message = self._worker_message(
            kind=WorkerCoordinationMessageKind.TASK_PROGRESS,
            worker_id=progress.worker_id,
            task_id=progress.task_id,
            payload={
                "percent": progress.percent,
                "message": progress.message,
            },
        )

        with self._lock:
            self._progress.append(progress)

        return self._record_message(
            message=message,
            reason=WorkerCoordinationReason.TASK_PROGRESS_RECORDED,
            result_message="task progress recorded",
            progress=progress,
        )

    def record_result(
        self,
        *,
        worker_id: WorkerId,
        result: TaskResult,
    ) -> WorkerCoordinationResult:
        """
        Record worker task result event.
        """

        assignment = self._assignment_for_worker(
            worker_id=worker_id,
            task_id=result.task_id,
        )

        if isinstance(assignment, WorkerCoordinationResult):
            return assignment

        with self._lock:
            if self._results.contains(result.task_id):
                return self._reject(
                    reason=WorkerCoordinationReason.RESULT_ALREADY_RECORDED,
                    message="task result already recorded",
                )

            self._results = self._results.add(result)

        kind = (
            WorkerCoordinationMessageKind.TASK_COMPLETED
            if result.success
            else WorkerCoordinationMessageKind.TASK_FAILED
        )
        reason = (
            WorkerCoordinationReason.TASK_COMPLETED
            if result.success
            else WorkerCoordinationReason.TASK_FAILED
        )
        message = self._worker_message(
            kind=kind,
            worker_id=worker_id,
            task_id=result.task_id,
            payload={
                "status": result.status.value,
                "success": result.success,
                "output": result.output,
                "error": result.error,
            },
        )

        return self._record_message(
            message=message,
            reason=reason,
            result_message="task result recorded",
            task_result=result,
        )

    def record_health(
        self,
        broadcast: WorkerHealthBroadcast,
    ) -> WorkerCoordinationResult:
        """
        Record worker health broadcast.
        """

        lookup = self._registry.get(broadcast.worker_id)

        if lookup.descriptor is None:
            return self._reject(
                reason=WorkerCoordinationReason.UNKNOWN_WORKER,
                message="worker is not registered",
            )

        self._registry.update_health(
            broadcast.worker_id,
            broadcast.health,
        )
        self._registry.update_load(
            broadcast.worker_id,
            active_tasks=broadcast.active_tasks,
            queued_tasks=broadcast.queued_tasks,
        )
        message = self._worker_message(
            kind=WorkerCoordinationMessageKind.WORKER_HEALTH,
            worker_id=broadcast.worker_id,
            task_id=None,
            payload={
                "health": broadcast.health.value,
                "active_tasks": broadcast.active_tasks,
                "queued_tasks": broadcast.queued_tasks,
                "message": broadcast.message,
            },
        )

        with self._lock:
            self._health.append(broadcast)

        return self._record_message(
            message=message,
            reason=WorkerCoordinationReason.WORKER_HEALTH_RECORDED,
            result_message="worker health recorded",
            health=broadcast,
        )

    def submit_message(
        self,
        message: WorkerCoordinationMessage,
    ) -> WorkerCoordinationResult:
        """
        Submit a typed coordination message.

        This is the central enforcement point for no worker-to-worker calls.
        """

        if (
            message.sender_kind == CoordinationActorKind.WORKER
            and message.receiver_kind == CoordinationActorKind.WORKER
        ):
            return self._reject(
                reason=WorkerCoordinationReason.DIRECT_WORKER_MESSAGE_REJECTED,
                message="worker-to-worker messages are forbidden",
            )

        with self._lock:
            self._messages.append(message)

        return self._accept(
            reason=self._reason_for_message(message),
            message="coordination message recorded",
            coordination_message=message,
        )

    def assignment_for(self, task_id: TaskId) -> TaskAssignment | None:
        validated_task_id = validate_task_id(task_id)

        with self._lock:
            return self._assignments.get(validated_task_id)

    def result_collection(self) -> ResultCollection:
        with self._lock:
            return self._results

    def messages(self) -> tuple[WorkerCoordinationMessage, ...]:
        with self._lock:
            return tuple(self._messages)

    def progress_updates(self) -> tuple[WorkerProgressUpdate, ...]:
        with self._lock:
            return tuple(self._progress)

    def health_broadcasts(self) -> tuple[WorkerHealthBroadcast, ...]:
        with self._lock:
            return tuple(self._health)

    def snapshot(self) -> WorkerCoordinatorSnapshot:
        with self._lock:
            return WorkerCoordinatorSnapshot(
                name=self.name,
                assignment_count=len(self._assignments),
                message_count=len(self._messages),
                result_count=len(self._results.results),
                progress_count=len(self._progress),
                health_count=len(self._health),
                rejected_count=self._rejected_count,
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        with self._lock:
            self._assignments.clear()
            self._messages.clear()
            self._progress.clear()
            self._health.clear()
            self._results = ResultCollection()
            self._rejected_count = 0
            self._last_reason = None

    def _assignment_for_worker(
        self,
        *,
        worker_id: WorkerId,
        task_id: TaskId,
    ) -> TaskAssignment | WorkerCoordinationResult:
        validated_worker_id = validate_worker_id(worker_id)
        validated_task_id = validate_task_id(task_id)

        with self._lock:
            assignment = self._assignments.get(validated_task_id)

        if assignment is None:
            return self._reject(
                reason=WorkerCoordinationReason.TASK_NOT_ASSIGNED,
                message="task is not assigned",
            )

        if assignment.worker_id != validated_worker_id:
            return self._reject(
                reason=WorkerCoordinationReason.WORKER_MISMATCH,
                message="worker does not own task assignment",
            )

        return assignment

    @staticmethod
    def _worker_message(
        *,
        kind: WorkerCoordinationMessageKind,
        worker_id: WorkerId,
        task_id: TaskId | None,
        payload: dict[str, object],
    ) -> WorkerCoordinationMessage:
        return WorkerCoordinationMessage(
            kind=kind,
            sender_kind=CoordinationActorKind.WORKER,
            sender_id=worker_id,
            receiver_kind=CoordinationActorKind.ORCHESTRATOR,
            receiver_id="orchestration_kernel",
            task_id=task_id,
            worker_id=worker_id,
            payload=payload,
        )

    def _record_message(
        self,
        *,
        message: WorkerCoordinationMessage,
        reason: WorkerCoordinationReason,
        result_message: str,
        task_result: TaskResult | None = None,
        progress: WorkerProgressUpdate | None = None,
        health: WorkerHealthBroadcast | None = None,
    ) -> WorkerCoordinationResult:
        with self._lock:
            self._messages.append(message)

        return self._accept(
            reason=reason,
            message=result_message,
            coordination_message=message,
            task_result=task_result,
            progress=progress,
            health=health,
        )

    def _accept(
        self,
        *,
        reason: WorkerCoordinationReason,
        message: str,
        coordination_message: WorkerCoordinationMessage | None = None,
        assignment: TaskAssignment | None = None,
        task_result: TaskResult | None = None,
        progress: WorkerProgressUpdate | None = None,
        health: WorkerHealthBroadcast | None = None,
    ) -> WorkerCoordinationResult:
        self._last_reason = reason

        return WorkerCoordinationResult(
            decision=WorkerCoordinationDecision.ACCEPTED,
            reason=reason,
            success=True,
            message=message,
            coordination_message=coordination_message,
            assignment=assignment,
            task_result=task_result,
            progress=progress,
            health=health,
        )

    def _reject(
        self,
        *,
        reason: WorkerCoordinationReason,
        message: str,
    ) -> WorkerCoordinationResult:
        self._last_reason = reason
        self._rejected_count += 1

        return WorkerCoordinationResult(
            decision=WorkerCoordinationDecision.REJECTED,
            reason=reason,
            success=False,
            message=message,
        )

    @staticmethod
    def _reason_for_message(
        message: WorkerCoordinationMessage,
    ) -> WorkerCoordinationReason:
        return {
            WorkerCoordinationMessageKind.TASK_ASSIGNED: (
                WorkerCoordinationReason.TASK_ASSIGNED
            ),
            WorkerCoordinationMessageKind.TASK_STARTED: (
                WorkerCoordinationReason.TASK_STARTED
            ),
            WorkerCoordinationMessageKind.TASK_PROGRESS: (
                WorkerCoordinationReason.TASK_PROGRESS_RECORDED
            ),
            WorkerCoordinationMessageKind.TASK_COMPLETED: (
                WorkerCoordinationReason.TASK_COMPLETED
            ),
            WorkerCoordinationMessageKind.TASK_FAILED: (
                WorkerCoordinationReason.TASK_FAILED
            ),
            WorkerCoordinationMessageKind.WORKER_HEALTH: (
                WorkerCoordinationReason.WORKER_HEALTH_RECORDED
            ),
        }[message.kind]