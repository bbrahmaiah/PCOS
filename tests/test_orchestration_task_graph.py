from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.orchestration import (
    JobRequest,
    ResourceBudget,
    ResourceKind,
    TaskDependency,
    TaskGraph,
    TaskGraphBuilder,
    TaskGraphDecision,
    TaskGraphReason,
    TaskKind,
    TaskPriority,
    TaskReadinessDecision,
    TaskRequest,
    TaskResult,
    TaskStatus,
    WorkerCapability,
    new_job_id,
    new_task_id,
)


def budget() -> ResourceBudget:
    return ResourceBudget(resource=ResourceKind.WORKER_SLOT, amount=1)


def task(
    *,
    task_id: str | None = None,
    job_id: str | None = None,
    kind: TaskKind = TaskKind.COGNITION,
    dependencies: tuple[TaskDependency, ...] = (),
) -> TaskRequest:
    return TaskRequest(
        task_id=task_id or new_task_id(),
        job_id=job_id,
        kind=kind,
        priority=TaskPriority.NORMAL,
        name="graph task",
        description="graph task",
        required_capabilities=(WorkerCapability.COGNITION,),
        resource_budgets=(budget(),),
        dependencies=dependencies,
    )


def result(
    *,
    task_id: str,
    status: TaskStatus = TaskStatus.SUCCEEDED,
    success: bool = True,
) -> TaskResult:
    return TaskResult(
        task_id=task_id,
        status=status,
        success=success,
        output="done",
    )


def job(
    *,
    tasks: tuple[TaskRequest, ...],
) -> JobRequest:
    return JobRequest(
        name="graph job",
        description="graph job",
        tasks=tasks,
    )


def two_task_job() -> tuple[JobRequest, str, str]:
    first_id = new_task_id()
    second_id = new_task_id()

    first = task(task_id=first_id)
    second = task(
        task_id=second_id,
        dependencies=(TaskDependency(task_id=first_id),),
    )

    return job(tasks=(first, second)), first_id, second_id


def test_graph_builder_creates_graph() -> None:
    item, first_id, second_id = two_task_job()

    graph = TaskGraphBuilder.from_job(item)

    assert graph.job_id == item.job_id
    assert graph.task_ids == {first_id, second_id}
    assert graph.results == ()


def test_graph_nodes_expose_dependency_metadata() -> None:
    item, first_id, second_id = two_task_job()
    graph = TaskGraph(job=item)
    nodes = graph.nodes

    assert len(nodes) == 2
    assert nodes[0].task_id == first_id
    assert nodes[0].has_dependencies is False
    assert nodes[1].task_id == second_id
    assert nodes[1].dependency_ids == (first_id,)


def test_validate_graph_returns_observable_result() -> None:
    item, _, _ = two_task_job()
    graph = TaskGraph(job=item)

    validation = graph.validate_graph()

    assert validation.valid is True
    assert validation.decision == TaskGraphDecision.ACCEPTED
    assert validation.reason == TaskGraphReason.GRAPH_VALID


def test_first_task_ready_second_waiting() -> None:
    item, first_id, second_id = two_task_job()
    graph = TaskGraph(job=item)

    first = graph.readiness_for(first_id)
    second = graph.readiness_for(second_id)

    assert first.ready is True
    assert first.decision == TaskReadinessDecision.READY
    assert second.ready is False
    assert second.decision == TaskReadinessDecision.WAITING_DEPENDENCIES
    assert second.missing_dependencies == (first_id,)


def test_unknown_task_readiness() -> None:
    item, _, _ = two_task_job()
    graph = TaskGraph(job=item)
    unknown_id = new_task_id()

    readiness = graph.readiness_for(unknown_id)

    assert readiness.ready is False
    assert readiness.decision == TaskReadinessDecision.UNKNOWN_TASK
    assert readiness.reason == TaskGraphReason.TASK_UNKNOWN


def test_ready_tasks_returns_only_dependency_satisfied_tasks() -> None:
    item, first_id, _ = two_task_job()
    graph = TaskGraph(job=item)

    ready = graph.ready_tasks()

    assert len(ready) == 1
    assert ready[0].task_id == first_id


def test_add_result_returns_new_graph() -> None:
    item, first_id, _ = two_task_job()
    graph = TaskGraph(job=item)

    mutation = graph.add_result(result(task_id=first_id))

    assert mutation.success is True
    assert mutation.decision == TaskGraphDecision.ACCEPTED
    assert mutation.reason == TaskGraphReason.TASK_RESULT_ACCEPTED
    assert mutation.graph is not None
    assert graph.results == ()
    assert len(mutation.graph.results) == 1


def test_add_unknown_result_rejected() -> None:
    item, _, _ = two_task_job()
    graph = TaskGraph(job=item)
    unknown = result(task_id=new_task_id())

    mutation = graph.add_result(unknown)

    assert mutation.success is False
    assert mutation.reason == TaskGraphReason.TASK_RESULT_UNKNOWN
    assert mutation.graph == graph


def test_add_duplicate_result_rejected() -> None:
    item, first_id, _ = two_task_job()
    graph = TaskGraph(job=item)
    first_mutation = graph.add_result(result(task_id=first_id))

    assert first_mutation.graph is not None

    duplicate = first_mutation.graph.add_result(result(task_id=first_id))

    assert duplicate.success is False
    assert duplicate.reason == TaskGraphReason.TASK_RESULT_DUPLICATE


def test_second_task_ready_after_first_succeeds() -> None:
    item, first_id, second_id = two_task_job()
    graph = TaskGraph(job=item)
    mutation = graph.add_result(result(task_id=first_id))

    assert mutation.graph is not None

    readiness = mutation.graph.readiness_for(second_id)

    assert readiness.ready is True
    assert readiness.decision == TaskReadinessDecision.READY


def test_second_task_blocked_when_dependency_fails() -> None:
    item, first_id, second_id = two_task_job()
    graph = TaskGraph(job=item)
    mutation = graph.add_result(
        result(
            task_id=first_id,
            status=TaskStatus.FAILED,
            success=False,
        )
    )

    assert mutation.graph is not None

    readiness = mutation.graph.readiness_for(second_id)

    assert readiness.ready is False
    assert readiness.decision == TaskReadinessDecision.BLOCKED_BY_FAILED_DEPENDENCY
    assert readiness.failed_dependencies == (first_id,)


def test_waiting_and_blocked_task_views() -> None:
    item, first_id, second_id = two_task_job()
    graph = TaskGraph(job=item)

    assert tuple(task_item.task_id for task_item in graph.waiting_tasks()) == (
        second_id,
    )
    assert graph.blocked_tasks() == ()

    mutation = graph.add_result(
        result(
            task_id=first_id,
            status=TaskStatus.FAILED,
            success=False,
        )
    )

    assert mutation.graph is not None
    assert mutation.graph.waiting_tasks() == ()
    assert tuple(
        task_item.task_id for task_item in mutation.graph.blocked_tasks()
    ) == (second_id,)


def test_complete_graph_converts_to_successful_job_result() -> None:
    item, first_id, second_id = two_task_job()
    graph = TaskGraph(job=item)
    first_mutation = graph.add_result(result(task_id=first_id))

    assert first_mutation.graph is not None

    second_mutation = first_mutation.graph.add_result(result(task_id=second_id))

    assert second_mutation.graph is not None

    job_result = second_mutation.graph.to_job_result()

    assert job_result.success is True
    assert job_result.status == TaskStatus.SUCCEEDED
    assert len(job_result.task_results) == 2


def test_complete_graph_converts_to_failed_job_result() -> None:
    item, first_id, second_id = two_task_job()
    graph = TaskGraph(job=item)
    first_mutation = graph.add_result(result(task_id=first_id))

    assert first_mutation.graph is not None

    second_mutation = first_mutation.graph.add_result(
        result(
            task_id=second_id,
            status=TaskStatus.FAILED,
            success=False,
        )
    )

    assert second_mutation.graph is not None

    job_result = second_mutation.graph.to_job_result()

    assert job_result.success is False
    assert job_result.status == TaskStatus.FAILED


def test_incomplete_graph_cannot_convert_to_job_result() -> None:
    item, _, _ = two_task_job()
    graph = TaskGraph(job=item)

    with pytest.raises(ValueError):
        graph.to_job_result()


def test_graph_rejects_duplicate_results_at_construction() -> None:
    item, first_id, _ = two_task_job()
    first_result = result(task_id=first_id)

    with pytest.raises(ValidationError):
        TaskGraph(job=item, results=(first_result, first_result))


def test_graph_rejects_result_from_other_job() -> None:
    item, _, _ = two_task_job()
    other_result = result(task_id=new_task_id())

    with pytest.raises(ValidationError):
        TaskGraph(job=item, results=(other_result,))


def test_job_task_id_still_validates_parent_job_contract() -> None:
    job_id = new_job_id()
    first = task(job_id=job_id)
    item = JobRequest(
        job_id=job_id,
        name="job",
        description="job",
        tasks=(first,),
    )

    graph = TaskGraph(job=item)

    assert graph.job_id == job_id


def test_enum_values_are_stable() -> None:
    assert TaskReadinessDecision.READY.value == "ready"
    assert TaskReadinessDecision.WAITING_DEPENDENCIES.value
    assert TaskGraphDecision.ACCEPTED.value == "accepted"
    assert TaskGraphReason.TASK_RESULT_ACCEPTED.value == "task_result_accepted"