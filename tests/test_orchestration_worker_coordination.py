from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.orchestration import (
    CoordinationActorKind,
    ResourceBudget,
    ResourceKind,
    ScheduledTask,
    TaskAssignment,
    TaskKind,
    TaskPriority,
    TaskRequest,
    TaskResult,
    TaskStatus,
    WorkerCapability,
    WorkerContract,
    WorkerCoordinationDecision,
    WorkerCoordinationMessage,
    WorkerCoordinationMessageKind,
    WorkerCoordinationReason,
    WorkerCoordinator,
    WorkerCoordinatorConfig,
    WorkerHealthBroadcast,
    WorkerHealthState,
    WorkerProgressUpdate,
    WorkerRegistry,
    new_worker_id,
)


def budget() -> ResourceBudget:
    return ResourceBudget(resource=ResourceKind.WORKER_SLOT, amount=1)


def task() -> TaskRequest:
    return TaskRequest(
        kind=TaskKind.COGNITION,
        priority=TaskPriority.NORMAL,
        name="coordination task",
        description="coordination task",
        required_capabilities=(WorkerCapability.COGNITION,),
        resource_budgets=(budget(),),
    )


def worker_contract() -> WorkerContract:
    return WorkerContract(
        name="cognition worker",
        capabilities=(WorkerCapability.COGNITION,),
        accepted_task_kinds=(TaskKind.COGNITION,),
        resource_budgets=(budget(),),
    )


def registry_with_worker() -> tuple[WorkerRegistry, str]:
    registry = WorkerRegistry()
    contract = worker_contract()
    descriptor = WorkerRegistry.descriptor_from_contract(contract)
    registry.register(descriptor)

    return registry, contract.worker_id


def scheduled(task_item: TaskRequest, worker_id: str) -> ScheduledTask:
    return ScheduledTask(
        task_id=task_item.task_id,
        worker_id=worker_id,
        priority=task_item.priority,
    )


def coordinator() -> tuple[WorkerCoordinator, str, TaskRequest, ScheduledTask]:
    registry, worker_id = registry_with_worker()
    task_item = task()
    scheduled_task = scheduled(task_item, worker_id)

    return WorkerCoordinator(registry=registry), worker_id, task_item, scheduled_task


def test_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        WorkerCoordinatorConfig(name=" ").validate()


def test_message_rejects_worker_to_worker_directly() -> None:
    with pytest.raises(ValidationError):
        WorkerCoordinationMessage(
            kind=WorkerCoordinationMessageKind.TASK_PROGRESS,
            sender_kind=CoordinationActorKind.WORKER,
            sender_id=new_worker_id(),
            receiver_kind=CoordinationActorKind.WORKER,
            receiver_id=new_worker_id(),
        )


def test_assignment_validates_task_match() -> None:
    _, worker_id = registry_with_worker()
    first = task()
    second = task()

    with pytest.raises(ValidationError):
        TaskAssignment(
            scheduled_task=scheduled(first, worker_id),
            task=second,
        )


def test_assign_creates_orchestrator_to_worker_message() -> None:
    runtime, worker_id, task_item, scheduled_task = coordinator()

    result = runtime.assign(
        scheduled_task=scheduled_task,
        task=task_item,
    )

    assert result.success is True
    assert result.reason == WorkerCoordinationReason.TASK_ASSIGNED
    assert result.assignment is not None
    assert result.assignment.worker_id == worker_id
    assert result.coordination_message is not None
    assert (
        result.coordination_message.kind
        == WorkerCoordinationMessageKind.TASK_ASSIGNED
    )
    assert result.coordination_message.sender_kind == CoordinationActorKind.ORCHESTRATOR
    assert result.coordination_message.receiver_kind == CoordinationActorKind.WORKER


def test_assign_rejects_duplicate_assignment() -> None:
    runtime, _, task_item, scheduled_task = coordinator()

    first = runtime.assign(scheduled_task=scheduled_task, task=task_item)
    second = runtime.assign(scheduled_task=scheduled_task, task=task_item)

    assert first.success is True
    assert second.success is False
    assert second.reason == WorkerCoordinationReason.DUPLICATE_ASSIGNMENT


def test_assign_rejects_unknown_worker() -> None:
    registry = WorkerRegistry()
    runtime = WorkerCoordinator(registry=registry)
    task_item = task()
    scheduled_task = scheduled(task_item, new_worker_id())

    result = runtime.assign(
        scheduled_task=scheduled_task,
        task=task_item,
    )

    assert result.success is False
    assert result.reason == WorkerCoordinationReason.UNKNOWN_WORKER


def test_record_started_requires_assignment() -> None:
    runtime, worker_id, task_item, _ = coordinator()

    result = runtime.record_started(
        worker_id=worker_id,
        task_id=task_item.task_id,
    )

    assert result.success is False
    assert result.reason == WorkerCoordinationReason.TASK_NOT_ASSIGNED


def test_record_started_accepts_assigned_worker() -> None:
    runtime, worker_id, task_item, scheduled_task = coordinator()
    runtime.assign(scheduled_task=scheduled_task, task=task_item)

    result = runtime.record_started(
        worker_id=worker_id,
        task_id=task_item.task_id,
    )

    assert result.success is True
    assert result.reason == WorkerCoordinationReason.TASK_STARTED
    assert result.coordination_message is not None
    assert (
        result.coordination_message.kind
        == WorkerCoordinationMessageKind.TASK_STARTED
    )


def test_record_started_rejects_wrong_worker() -> None:
    runtime, _, task_item, scheduled_task = coordinator()
    runtime.assign(scheduled_task=scheduled_task, task=task_item)

    result = runtime.record_started(
        worker_id=new_worker_id(),
        task_id=task_item.task_id,
    )

    assert result.success is False
    assert result.reason == WorkerCoordinationReason.WORKER_MISMATCH


def test_record_progress_accepts_assigned_worker() -> None:
    runtime, worker_id, task_item, scheduled_task = coordinator()
    runtime.assign(scheduled_task=scheduled_task, task=task_item)
    progress = WorkerProgressUpdate(
        worker_id=worker_id,
        task_id=task_item.task_id,
        percent=50,
        message="half done",
    )

    result = runtime.record_progress(progress)

    assert result.success is True
    assert result.progress == progress
    assert len(runtime.progress_updates()) == 1


def test_record_result_collects_success() -> None:
    runtime, worker_id, task_item, scheduled_task = coordinator()
    runtime.assign(scheduled_task=scheduled_task, task=task_item)
    task_result = TaskResult(
        task_id=task_item.task_id,
        status=TaskStatus.SUCCEEDED,
        success=True,
        output="ok",
    )

    result = runtime.record_result(
        worker_id=worker_id,
        result=task_result,
    )

    assert result.success is True
    assert result.reason == WorkerCoordinationReason.TASK_COMPLETED
    assert runtime.result_collection().contains(task_item.task_id) is True


def test_record_result_collects_failure() -> None:
    runtime, worker_id, task_item, scheduled_task = coordinator()
    runtime.assign(scheduled_task=scheduled_task, task=task_item)
    task_result = TaskResult(
        task_id=task_item.task_id,
        status=TaskStatus.FAILED,
        success=False,
        error="failed",
    )

    result = runtime.record_result(
        worker_id=worker_id,
        result=task_result,
    )

    assert result.success is True
    assert result.reason == WorkerCoordinationReason.TASK_FAILED
    assert result.coordination_message is not None
    assert (
        result.coordination_message.kind
        == WorkerCoordinationMessageKind.TASK_FAILED
    )


def test_duplicate_result_rejected() -> None:
    runtime, worker_id, task_item, scheduled_task = coordinator()
    runtime.assign(scheduled_task=scheduled_task, task=task_item)
    task_result = TaskResult(
        task_id=task_item.task_id,
        status=TaskStatus.SUCCEEDED,
        success=True,
    )

    first = runtime.record_result(worker_id=worker_id, result=task_result)
    second = runtime.record_result(worker_id=worker_id, result=task_result)

    assert first.success is True
    assert second.success is False
    assert second.reason == WorkerCoordinationReason.RESULT_ALREADY_RECORDED


def test_health_broadcast_updates_registry() -> None:
    registry, worker_id = registry_with_worker()
    runtime = WorkerCoordinator(registry=registry)
    broadcast = WorkerHealthBroadcast(
        worker_id=worker_id,
        health=WorkerHealthState.DEGRADED,
        active_tasks=0,
        queued_tasks=1,
    )

    result = runtime.record_health(broadcast)
    lookup = registry.get(worker_id)

    assert result.success is True
    assert result.reason == WorkerCoordinationReason.WORKER_HEALTH_RECORDED
    assert lookup.descriptor is not None
    assert lookup.descriptor.health == WorkerHealthState.DEGRADED
    assert len(runtime.health_broadcasts()) == 1


def test_health_broadcast_rejects_unknown_worker() -> None:
    runtime = WorkerCoordinator(registry=WorkerRegistry())
    broadcast = WorkerHealthBroadcast(
        worker_id=new_worker_id(),
        health=WorkerHealthState.HEALTHY,
    )

    result = runtime.record_health(broadcast)

    assert result.success is False
    assert result.reason == WorkerCoordinationReason.UNKNOWN_WORKER


def test_submit_message_records_valid_worker_to_orchestrator_message() -> None:
    runtime, worker_id, task_item, scheduled_task = coordinator()
    runtime.assign(scheduled_task=scheduled_task, task=task_item)
    message = WorkerCoordinationMessage(
        kind=WorkerCoordinationMessageKind.TASK_PROGRESS,
        sender_kind=CoordinationActorKind.WORKER,
        sender_id=worker_id,
        receiver_kind=CoordinationActorKind.ORCHESTRATOR,
        receiver_id="orchestration_kernel",
        task_id=task_item.task_id,
        worker_id=worker_id,
        payload={"percent": 10},
    )

    result = runtime.submit_message(message)

    assert result.success is True
    assert result.reason == WorkerCoordinationReason.TASK_PROGRESS_RECORDED


def test_assignment_lookup_returns_assignment() -> None:
    runtime, _, task_item, scheduled_task = coordinator()
    runtime.assign(scheduled_task=scheduled_task, task=task_item)

    assert runtime.assignment_for(task_item.task_id) is not None


def test_snapshot_counts_coordination_state() -> None:
    runtime, worker_id, task_item, scheduled_task = coordinator()
    runtime.assign(scheduled_task=scheduled_task, task=task_item)
    runtime.record_started(worker_id=worker_id, task_id=task_item.task_id)

    snapshot = runtime.snapshot()

    assert snapshot.assignment_count == 1
    assert snapshot.message_count == 2
    assert snapshot.last_reason == WorkerCoordinationReason.TASK_STARTED


def test_reset_clears_coordination_state() -> None:
    runtime, _, task_item, scheduled_task = coordinator()
    runtime.assign(scheduled_task=scheduled_task, task=task_item)

    runtime.reset()

    snapshot = runtime.snapshot()

    assert snapshot.assignment_count == 0
    assert snapshot.message_count == 0
    assert snapshot.result_count == 0


def test_enum_values_are_stable() -> None:
    assert CoordinationActorKind.ORCHESTRATOR.value == "orchestrator"
    assert WorkerCoordinationMessageKind.TASK_ASSIGNED.value == "task_assigned"
    assert WorkerCoordinationDecision.ACCEPTED.value == "accepted"
    assert WorkerCoordinationReason.TASK_COMPLETED.value == "task_completed"