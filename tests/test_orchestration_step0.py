from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.orchestration import (
    BudgetPolicy,
    JobRequest,
    JobResult,
    OrchestrationSnapshot,
    OrchestratorState,
    ResourceBudget,
    ResourceKind,
    TaskDependency,
    TaskKind,
    TaskPriority,
    TaskRequest,
    TaskResult,
    TaskStatus,
    WorkerCapability,
    WorkerContract,
    WorkerState,
    new_job_id,
    new_orchestration_id,
    new_task_id,
    new_worker_id,
    validate_job_id,
    validate_orchestration_id,
    validate_task_id,
    validate_worker_id,
)


def budget(
    resource: ResourceKind = ResourceKind.WORKER_SLOT,
    amount: int = 1,
) -> ResourceBudget:
    return ResourceBudget(resource=resource, amount=amount)


def task(
    *,
    task_id: str | None = None,
    job_id: str | None = None,
    kind: TaskKind = TaskKind.COGNITION,
    priority: TaskPriority = TaskPriority.NORMAL,
    background: bool = False,
    dependencies: tuple[TaskDependency, ...] = (),
    timeout_ms: int | None = None,
) -> TaskRequest:
    return TaskRequest(
        task_id=task_id or new_task_id(),
        job_id=job_id,
        kind=kind,
        priority=priority,
        name="test task",
        description="test orchestration task",
        required_capabilities=(WorkerCapability.COGNITION,),
        resource_budgets=(budget(),),
        dependencies=dependencies,
        timeout_ms=timeout_ms,
        background=background,
    )


def test_ids_have_stable_prefixes() -> None:
    assert new_task_id().startswith("task_")
    assert new_job_id().startswith("job_")
    assert new_worker_id().startswith("worker_")
    assert new_orchestration_id().startswith("orch_")


def test_id_validators_reject_wrong_prefixes() -> None:
    with pytest.raises(ValueError):
        validate_task_id("job_wrong")

    with pytest.raises(ValueError):
        validate_job_id("task_wrong")

    with pytest.raises(ValueError):
        validate_worker_id("orch_wrong")

    with pytest.raises(ValueError):
        validate_orchestration_id("worker_wrong")


def test_resource_budget_requires_positive_amount() -> None:
    with pytest.raises(ValidationError):
        ResourceBudget(resource=ResourceKind.CPU_SLOT, amount=0)


def test_resource_budget_can_cover_requested_amount() -> None:
    item = ResourceBudget(resource=ResourceKind.LLM_TOKEN, amount=100)

    assert item.can_cover(50) is True
    assert item.can_cover(101) is False


def test_task_request_requires_capability() -> None:
    with pytest.raises(ValidationError):
        TaskRequest(
            kind=TaskKind.COGNITION,
            priority=TaskPriority.NORMAL,
            name="task",
            description="task",
            required_capabilities=(),
            resource_budgets=(budget(),),
        )


def test_task_request_requires_budget() -> None:
    with pytest.raises(ValidationError):
        TaskRequest(
            kind=TaskKind.COGNITION,
            priority=TaskPriority.NORMAL,
            name="task",
            description="task",
            required_capabilities=(WorkerCapability.COGNITION,),
            resource_budgets=(),
        )


def test_high_priority_task_requires_timeout() -> None:
    with pytest.raises(ValidationError):
        task(priority=TaskPriority.HIGH)


def test_conversation_task_requires_high_priority() -> None:
    with pytest.raises(ValidationError):
        task(kind=TaskKind.CONVERSATION_TURN, priority=TaskPriority.NORMAL)


def test_conversation_task_cannot_be_background() -> None:
    with pytest.raises(ValidationError):
        task(
            kind=TaskKind.CONVERSATION_TURN,
            priority=TaskPriority.HIGH,
            timeout_ms=10_000,
            background=True,
        )


def test_background_task_must_use_background_priority() -> None:
    with pytest.raises(ValidationError):
        task(background=True, priority=TaskPriority.LOW)


def test_valid_background_task_contract() -> None:
    item = task(
        kind=TaskKind.BACKGROUND_MAINTENANCE,
        priority=TaskPriority.BACKGROUND,
        background=True,
    )

    assert item.background is True
    assert item.priority == TaskPriority.BACKGROUND


def test_dependency_requires_terminal_status() -> None:
    with pytest.raises(ValidationError):
        TaskDependency(
            task_id=new_task_id(),
            required_statuses=(TaskStatus.RUNNING,),
        )


def test_task_result_requires_terminal_status() -> None:
    with pytest.raises(ValidationError):
        TaskResult(
            task_id=new_task_id(),
            status=TaskStatus.RUNNING,
            success=False,
        )


def test_task_result_success_requires_succeeded_status() -> None:
    with pytest.raises(ValidationError):
        TaskResult(
            task_id=new_task_id(),
            status=TaskStatus.FAILED,
            success=True,
        )


def test_valid_task_result() -> None:
    result = TaskResult(
        task_id=new_task_id(),
        status=TaskStatus.SUCCEEDED,
        success=True,
    )

    assert result.success is True
    assert result.status == TaskStatus.SUCCEEDED


def test_job_requires_tasks() -> None:
    with pytest.raises(ValidationError):
        JobRequest(
            name="job",
            description="job",
            tasks=(),
        )


def test_job_rejects_duplicate_task_ids() -> None:
    task_id = new_task_id()

    with pytest.raises(ValidationError):
        JobRequest(
            name="job",
            description="job",
            tasks=(
                task(task_id=task_id),
                task(task_id=task_id),
            ),
        )


def test_job_rejects_missing_dependency() -> None:
    missing_task_id = new_task_id()
    dependent = task(
        dependencies=(TaskDependency(task_id=missing_task_id),),
    )

    with pytest.raises(ValidationError):
        JobRequest(
            name="job",
            description="job",
            tasks=(dependent,),
        )


def test_job_rejects_self_dependency() -> None:
    task_id = new_task_id()
    dependent = task(
        task_id=task_id,
        dependencies=(TaskDependency(task_id=task_id),),
    )

    with pytest.raises(ValidationError):
        JobRequest(
            name="job",
            description="job",
            tasks=(dependent,),
        )


def test_job_rejects_cycles() -> None:
    first_id = new_task_id()
    second_id = new_task_id()

    first = task(
        task_id=first_id,
        dependencies=(TaskDependency(task_id=second_id),),
    )
    second = task(
        task_id=second_id,
        dependencies=(TaskDependency(task_id=first_id),),
    )

    with pytest.raises(ValidationError):
        JobRequest(
            name="job",
            description="job",
            tasks=(first, second),
        )


def test_valid_job_request_with_dependency_graph() -> None:
    first_id = new_task_id()
    second_id = new_task_id()

    first = task(task_id=first_id)
    second = task(
        task_id=second_id,
        dependencies=(TaskDependency(task_id=first_id),),
    )
    job = JobRequest(
        name="job",
        description="job",
        tasks=(first, second),
    )

    assert job.tasks[1].dependencies[0].task_id == first_id


def test_job_result_requires_successful_task_results() -> None:
    failed_task = TaskResult(
        task_id=new_task_id(),
        status=TaskStatus.FAILED,
        success=False,
    )

    with pytest.raises(ValidationError):
        JobResult(
            job_id=new_job_id(),
            status=TaskStatus.SUCCEEDED,
            success=True,
            task_results=(failed_task,),
        )


def test_valid_job_result() -> None:
    succeeded_task = TaskResult(
        task_id=new_task_id(),
        status=TaskStatus.SUCCEEDED,
        success=True,
    )
    result = JobResult(
        job_id=new_job_id(),
        status=TaskStatus.SUCCEEDED,
        success=True,
        task_results=(succeeded_task,),
    )

    assert result.success is True


def test_worker_contract_requires_capabilities() -> None:
    with pytest.raises(ValidationError):
        WorkerContract(
            name="worker",
            capabilities=(),
            accepted_task_kinds=(TaskKind.COGNITION,),
            resource_budgets=(budget(),),
        )


def test_worker_contract_can_accept_matching_task() -> None:
    worker = WorkerContract(
        name="cognition worker",
        capabilities=(WorkerCapability.COGNITION,),
        accepted_task_kinds=(TaskKind.COGNITION,),
        resource_budgets=(budget(),),
    )
    item = task(kind=TaskKind.COGNITION)

    assert worker.can_accept(item) is True


def test_worker_contract_rejects_wrong_task_kind() -> None:
    worker = WorkerContract(
        name="memory worker",
        capabilities=(WorkerCapability.MEMORY,),
        accepted_task_kinds=(TaskKind.MEMORY_RETRIEVAL,),
        resource_budgets=(budget(),),
    )
    item = task(kind=TaskKind.COGNITION)

    assert worker.can_accept(item) is False


def test_orchestration_snapshot_contract() -> None:
    snapshot = OrchestrationSnapshot(
        state=OrchestratorState.IDLE,
        active_task_count=0,
        active_job_count=0,
        registered_worker_count=1,
    )

    assert snapshot.state == OrchestratorState.IDLE
    assert snapshot.registered_worker_count == 1


def test_enum_values_are_stable() -> None:
    assert TaskKind.COGNITION.value == "cognition"
    assert TaskPriority.CRITICAL.value == 0
    assert TaskStatus.RUNNING.value == "running"
    assert WorkerState.REGISTERED.value == "registered"
    assert WorkerCapability.TOOL_ACTION.value == "tool_action"
    assert ResourceKind.LLM_TOKEN.value == "llm_token"
    assert BudgetPolicy.ENFORCE.value == "enforce"
    assert OrchestratorState.LOAD_SHEDDING.value == "load_shedding"