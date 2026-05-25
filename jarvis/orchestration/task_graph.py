from __future__ import annotations

from enum import StrEnum

from pydantic import Field, field_validator, model_validator

from jarvis.orchestration.ids import (
    JobId,
    TaskId,
    utc_now,
    validate_job_id,
    validate_task_id,
)
from jarvis.orchestration.models import (
    JobRequest,
    JobResult,
    OrchestrationModel,
    TaskRequest,
    TaskResult,
    TaskStatus,
)


class TaskReadinessDecision(StrEnum):
    """
    Readiness decision for an orchestration task.
    """

    READY = "ready"
    WAITING_DEPENDENCIES = "waiting_dependencies"
    ALREADY_COMPLETED = "already_completed"
    BLOCKED_BY_FAILED_DEPENDENCY = "blocked_by_failed_dependency"
    UNKNOWN_TASK = "unknown_task"


class TaskGraphDecision(StrEnum):
    """
    Task graph mutation or validation decision.
    """

    ACCEPTED = "accepted"
    REJECTED = "rejected"


class TaskGraphReason(StrEnum):
    """
    Machine-readable task graph reason.
    """

    GRAPH_CREATED = "graph_created"
    GRAPH_VALID = "graph_valid"
    TASK_READY = "task_ready"
    TASK_WAITING_DEPENDENCIES = "task_waiting_dependencies"
    TASK_ALREADY_COMPLETED = "task_already_completed"
    TASK_UNKNOWN = "task_unknown"
    TASK_RESULT_ACCEPTED = "task_result_accepted"
    TASK_RESULT_DUPLICATE = "task_result_duplicate"
    TASK_RESULT_UNKNOWN = "task_result_unknown"
    FAILED_DEPENDENCY = "failed_dependency"
    JOB_INCOMPLETE = "job_incomplete"
    JOB_SUCCEEDED = "job_succeeded"
    JOB_FAILED = "job_failed"


class TaskGraphNode(OrchestrationModel):
    """
    Immutable graph node around a TaskRequest.

    The node does not execute. It exposes dependency metadata for schedulers.
    """

    task: TaskRequest

    @property
    def task_id(self) -> TaskId:
        return self.task.task_id

    @property
    def dependency_ids(self) -> tuple[TaskId, ...]:
        return tuple(dependency.task_id for dependency in self.task.dependencies)

    @property
    def has_dependencies(self) -> bool:
        return bool(self.task.dependencies)


class TaskReadinessResult(OrchestrationModel):
    """
    Result of checking whether a task can become READY.
    """

    task_id: TaskId
    decision: TaskReadinessDecision
    reason: TaskGraphReason
    ready: bool
    missing_dependencies: tuple[TaskId, ...] = ()
    failed_dependencies: tuple[TaskId, ...] = ()
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("task_id")
    @classmethod
    def _validate_task_id(cls, value: str) -> str:
        return validate_task_id(value)


class TaskGraphMutationResult(OrchestrationModel):
    """
    Result of attempting to add a task result to a graph.
    """

    task_id: TaskId
    decision: TaskGraphDecision
    reason: TaskGraphReason
    success: bool
    message: str
    graph: TaskGraph | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("task_id")
    @classmethod
    def _validate_task_id(cls, value: str) -> str:
        return validate_task_id(value)

    @field_validator("message")
    @classmethod
    def _required_message(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("message cannot be empty.")

        return cleaned


class TaskGraphValidationResult(OrchestrationModel):
    """
    Validation result for a TaskGraph.
    """

    job_id: JobId
    decision: TaskGraphDecision
    reason: TaskGraphReason
    valid: bool
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("job_id")
    @classmethod
    def _validate_job_id(cls, value: str) -> str:
        return validate_job_id(value)

    @field_validator("message")
    @classmethod
    def _required_message(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("message cannot be empty.")

        return cleaned


class TaskGraph(OrchestrationModel):
    """
    Immutable runtime graph for a JobRequest.

    A TaskGraph models dependency readiness and task results.
    It never executes tasks. It only tells the scheduler which tasks are ready,
    waiting, blocked, completed, or finished.

    Phase 6 law:
    Scheduler proposes execution order.
    Budget Runtime approves.
    Workers execute.
    TaskGraph only models correctness.
    """

    job: JobRequest
    results: tuple[TaskResult, ...] = ()
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_graph(self) -> TaskGraph:
        task_ids = self.task_ids
        result_ids = [result.task_id for result in self.results]

        if len(set(result_ids)) != len(result_ids):
            raise ValueError("task graph results must be unique per task.")

        for result_id in result_ids:
            if result_id not in task_ids:
                raise ValueError("task graph result must belong to graph job.")

        return self

    @property
    def job_id(self) -> JobId:
        return self.job.job_id

    @property
    def nodes(self) -> tuple[TaskGraphNode, ...]:
        return tuple(TaskGraphNode(task=task) for task in self.job.tasks)

    @property
    def task_ids(self) -> set[TaskId]:
        return {task.task_id for task in self.job.tasks}

    @property
    def completed_task_ids(self) -> set[TaskId]:
        return {result.task_id for result in self.results}

    @property
    def incomplete_task_ids(self) -> set[TaskId]:
        return self.task_ids - self.completed_task_ids

    @property
    def complete(self) -> bool:
        return not self.incomplete_task_ids

    @property
    def succeeded(self) -> bool:
        return self.complete and all(result.success for result in self.results)

    @property
    def failed(self) -> bool:
        return any(not result.success for result in self.results)

    def validate_graph(self) -> TaskGraphValidationResult:
        """
        Return a formal validation result.

        JobRequest already enforces local, acyclic dependencies.
        This method gives later runtime layers an observable validation object.
        """

        return TaskGraphValidationResult(
            job_id=self.job_id,
            decision=TaskGraphDecision.ACCEPTED,
            reason=TaskGraphReason.GRAPH_VALID,
            valid=True,
            message="task graph is valid",
            metadata={
                "task_count": len(self.job.tasks),
                "result_count": len(self.results),
            },
        )

    def task_for_id(self, task_id: TaskId) -> TaskRequest | None:
        """
        Return a task by id.
        """

        validated_task_id = validate_task_id(task_id)

        for task in self.job.tasks:
            if task.task_id == validated_task_id:
                return task

        return None

    def result_for_id(self, task_id: TaskId) -> TaskResult | None:
        """
        Return a task result by id.
        """

        validated_task_id = validate_task_id(task_id)

        for result in self.results:
            if result.task_id == validated_task_id:
                return result

        return None

    def readiness_for(self, task_id: TaskId) -> TaskReadinessResult:
        """
        Determine whether a task is ready based on completed dependencies.
        """

        validated_task_id = validate_task_id(task_id)
        task = self.task_for_id(validated_task_id)

        if task is None:
            return TaskReadinessResult(
                task_id=validated_task_id,
                decision=TaskReadinessDecision.UNKNOWN_TASK,
                reason=TaskGraphReason.TASK_UNKNOWN,
                ready=False,
            )

        if self.result_for_id(validated_task_id) is not None:
            return TaskReadinessResult(
                task_id=validated_task_id,
                decision=TaskReadinessDecision.ALREADY_COMPLETED,
                reason=TaskGraphReason.TASK_ALREADY_COMPLETED,
                ready=False,
            )

        missing: list[TaskId] = []
        failed: list[TaskId] = []

        for dependency in task.dependencies:
            dependency_result = self.result_for_id(dependency.task_id)

            if dependency_result is None:
                missing.append(dependency.task_id)
                continue

            if dependency_result.status not in dependency.required_statuses:
                failed.append(dependency.task_id)

        if failed:
            return TaskReadinessResult(
                task_id=validated_task_id,
                decision=TaskReadinessDecision.BLOCKED_BY_FAILED_DEPENDENCY,
                reason=TaskGraphReason.FAILED_DEPENDENCY,
                ready=False,
                failed_dependencies=tuple(failed),
            )

        if missing:
            return TaskReadinessResult(
                task_id=validated_task_id,
                decision=TaskReadinessDecision.WAITING_DEPENDENCIES,
                reason=TaskGraphReason.TASK_WAITING_DEPENDENCIES,
                ready=False,
                missing_dependencies=tuple(missing),
            )

        return TaskReadinessResult(
            task_id=validated_task_id,
            decision=TaskReadinessDecision.READY,
            reason=TaskGraphReason.TASK_READY,
            ready=True,
        )

    def ready_tasks(self) -> tuple[TaskRequest, ...]:
        """
        Return tasks ready to be scheduled.

        This does not schedule. It only exposes graph readiness.
        """

        return tuple(
            task
            for task in self.job.tasks
            if self.readiness_for(task.task_id).ready
        )

    def waiting_tasks(self) -> tuple[TaskRequest, ...]:
        """
        Return tasks waiting for dependencies.
        """

        return tuple(
            task
            for task in self.job.tasks
            if self.readiness_for(task.task_id).decision
            == TaskReadinessDecision.WAITING_DEPENDENCIES
        )

    def blocked_tasks(self) -> tuple[TaskRequest, ...]:
        """
        Return tasks blocked by failed dependencies.
        """

        return tuple(
            task
            for task in self.job.tasks
            if self.readiness_for(task.task_id).decision
            == TaskReadinessDecision.BLOCKED_BY_FAILED_DEPENDENCY
        )

    def add_result(self, result: TaskResult) -> TaskGraphMutationResult:
        """
        Return a new graph with a task result added.

        The graph remains immutable. This prevents hidden runtime mutation.
        """

        if result.task_id not in self.task_ids:
            return TaskGraphMutationResult(
                task_id=result.task_id,
                decision=TaskGraphDecision.REJECTED,
                reason=TaskGraphReason.TASK_RESULT_UNKNOWN,
                success=False,
                message="task result does not belong to graph",
                graph=self,
            )

        if self.result_for_id(result.task_id) is not None:
            return TaskGraphMutationResult(
                task_id=result.task_id,
                decision=TaskGraphDecision.REJECTED,
                reason=TaskGraphReason.TASK_RESULT_DUPLICATE,
                success=False,
                message="task result already recorded",
                graph=self,
            )

        next_graph = self.model_copy(
            update={
                "results": self.results + (result,),
                "updated_at": utc_now(),
            }
        )

        return TaskGraphMutationResult(
            task_id=result.task_id,
            decision=TaskGraphDecision.ACCEPTED,
            reason=TaskGraphReason.TASK_RESULT_ACCEPTED,
            success=True,
            message="task result accepted",
            graph=next_graph,
        )

    def to_job_result(self) -> JobResult:
        """
        Convert a complete graph to a JobResult.

        Incomplete graphs cannot become job results.
        """

        if not self.complete:
            raise ValueError("cannot build job result from incomplete task graph.")

        status = TaskStatus.SUCCEEDED if self.succeeded else TaskStatus.FAILED

        return JobResult(
            job_id=self.job_id,
            status=status,
            success=self.succeeded,
            task_results=self.results,
        )


class TaskGraphBuilder:
    """
    Factory for TaskGraph objects.

    Kept as a separate class so future Step 6 scheduler can depend on a narrow
    graph-building interface instead of manually constructing graphs.
    """

    @staticmethod
    def from_job(job: JobRequest) -> TaskGraph:
        """
        Build a graph from a validated JobRequest.
        """

        return TaskGraph(job=job)