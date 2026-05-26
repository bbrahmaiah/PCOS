from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.orchestration import (
    AttentionContext,
    AttentionRuntime,
    BackgroundTaskDecision,
    BackgroundTaskKind,
    BackgroundTaskReason,
    BackgroundTaskRequest,
    BackgroundTaskRuntime,
    BackgroundTaskRuntimeConfig,
    BackgroundTaskStatus,
    OrchestrationStateContext,
    OrchestrationStateMachine,
    ResourceBudgetRuntime,
    ResourceKind,
    ResourcePool,
    TaskKind,
    TaskPriority,
    TaskScheduler,
    WorkerCapability,
    WorkerContract,
    WorkerRegistry,
    context_prefetch_task,
    health_check_task,
    log_rotation_task,
    memory_consolidation_task,
    new_task_id,
    worker_slots,
    workspace_scan_task,
)


def background_worker() -> WorkerContract:
    return WorkerContract(
        name="background worker",
        capabilities=(WorkerCapability.BACKGROUND,),
        accepted_task_kinds=(TaskKind.BACKGROUND_MAINTENANCE,),
        max_concurrent_tasks=4,
        resource_budgets=(worker_slots(1),),
    )


def health_worker() -> WorkerContract:
    return WorkerContract(
        name="health worker",
        capabilities=(WorkerCapability.OBSERVABILITY,),
        accepted_task_kinds=(TaskKind.HEALTH_CHECK,),
        max_concurrent_tasks=2,
        resource_budgets=(worker_slots(1),),
    )


def registry() -> WorkerRegistry:
    item = WorkerRegistry()
    item.register(
        WorkerRegistry.descriptor_from_contract(background_worker())
    )
    item.register(
        WorkerRegistry.descriptor_from_contract(health_worker())
    )

    return item


def budgets() -> ResourceBudgetRuntime:
    return ResourceBudgetRuntime(
        pools=(
            ResourcePool(
                resource=ResourceKind.WORKER_SLOT,
                capacity=8,
                conversation_reserved=2,
            ),
        )
    )


def state_machine() -> OrchestrationStateMachine:
    machine = OrchestrationStateMachine()
    machine.bootstrap_completed()

    return machine


def scheduler(
    *,
    registry_item: WorkerRegistry | None = None,
    attention: AttentionRuntime | None = None,
    budget_runtime: ResourceBudgetRuntime | None = None,
    machine: OrchestrationStateMachine | None = None,
) -> TaskScheduler:
    return TaskScheduler(
        registry=registry_item or registry(),
        attention=attention or AttentionRuntime(),
        budgets=budget_runtime or budgets(),
        state_machine=machine or state_machine(),
    )


def runtime(
    *,
    attention: AttentionRuntime | None = None,
    machine: OrchestrationStateMachine | None = None,
) -> BackgroundTaskRuntime:
    final_attention = attention or AttentionRuntime()
    final_machine = machine or state_machine()

    return BackgroundTaskRuntime(
        scheduler=scheduler(attention=final_attention, machine=final_machine),
        attention=final_attention,
        state_machine=final_machine,
    )


def test_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        BackgroundTaskRuntimeConfig(name=" ").validate()


def test_request_requires_cancellable() -> None:
    with pytest.raises(ValidationError):
        BackgroundTaskRequest(
            kind=BackgroundTaskKind.MEMORY_CONSOLIDATION,
            name="memory",
            description="memory",
            cancellable=False,
        )


def test_register_background_task() -> None:
    bg = runtime()

    result = bg.register(memory_consolidation_task())

    assert result.success is True
    assert result.reason == BackgroundTaskReason.TASK_REGISTERED
    assert result.descriptor is not None
    assert result.descriptor.task.priority == TaskPriority.BACKGROUND
    assert result.descriptor.task.background is True


def test_registered_background_task_is_lowest_priority() -> None:
    bg = runtime()

    result = bg.register(context_prefetch_task())

    assert result.descriptor is not None
    assert result.descriptor.task.priority == TaskPriority.BACKGROUND
    assert result.descriptor.task.background is True


def test_schedule_background_task_when_idle() -> None:
    bg = runtime()
    registered = bg.register(memory_consolidation_task())

    assert registered.task_id is not None

    result = bg.schedule_one(registered.task_id)

    assert result.success is True
    assert result.decision == BackgroundTaskDecision.SCHEDULED
    assert result.reason == BackgroundTaskReason.TASK_SCHEDULED
    assert result.descriptor is not None
    assert result.descriptor.status == BackgroundTaskStatus.SCHEDULED


def test_background_yields_when_user_speech_active() -> None:
    attention = AttentionRuntime(
        context=AttentionContext(
            active_conversation=True,
            speech_active=True,
        )
    )
    bg = runtime(attention=attention)
    registered = bg.register(workspace_scan_task())

    assert registered.task_id is not None

    result = bg.schedule_one(registered.task_id)

    assert result.success is False
    assert result.decision == BackgroundTaskDecision.YIELDED
    assert result.reason == BackgroundTaskReason.TASK_YIELDED_TO_ATTENTION


def test_background_sheds_under_load_shedding() -> None:
    machine = state_machine()
    machine.enter_busy(
        OrchestrationStateContext(
            registered_worker_count=1,
            active_task_count=1,
        )
    )
    machine.start_load_shedding(
        OrchestrationStateContext(
            registered_worker_count=1,
            active_task_count=1,
            resource_pressure=True,
        )
    )
    bg = runtime(machine=machine)
    registered = bg.register(log_rotation_task())

    assert registered.task_id is not None

    result = bg.schedule_one(registered.task_id)

    assert result.success is False
    assert result.decision == BackgroundTaskDecision.SHED
    assert result.reason == BackgroundTaskReason.TASK_SHED_DURING_LOAD_SHEDDING


def test_cancel_background_task() -> None:
    bg = runtime()
    registered = bg.register(context_prefetch_task())

    assert registered.task_id is not None

    result = bg.cancel(registered.task_id, reason="user spoke")

    assert result.success is True
    assert result.decision == BackgroundTaskDecision.CANCELLED
    assert result.descriptor is not None
    assert result.descriptor.cancelled is True


def test_cancel_unknown_task_rejected() -> None:
    bg = runtime()

    result = bg.cancel(new_task_id())

    assert result.success is False
    assert result.reason == BackgroundTaskReason.TASK_NOT_FOUND


def test_cancelled_task_does_not_schedule() -> None:
    bg = runtime()
    registered = bg.register(memory_consolidation_task())

    assert registered.task_id is not None

    bg.cancel(registered.task_id, reason="stop background")
    result = bg.schedule_one(registered.task_id)

    assert result.success is False
    assert result.decision == BackgroundTaskDecision.CANCELLED


def test_schedule_cycle_schedules_multiple_background_tasks() -> None:
    bg = runtime()
    bg.register(memory_consolidation_task())
    bg.register(context_prefetch_task())

    results = bg.schedule_cycle()

    assert len(results) == 2
    assert all(
        result.decision == BackgroundTaskDecision.SCHEDULED
        for result in results
    )


def test_schedule_cycle_respects_cycle_limit() -> None:
    attention = AttentionRuntime()
    machine = state_machine()
    bg = BackgroundTaskRuntime(
        scheduler=scheduler(attention=attention, machine=machine),
        attention=attention,
        state_machine=machine,
        config=BackgroundTaskRuntimeConfig(max_schedule_per_cycle=1),
    )
    bg.register(memory_consolidation_task())
    bg.register(context_prefetch_task())

    results = bg.schedule_cycle()

    assert len(results) == 1


def test_health_check_uses_observability_worker() -> None:
    bg = runtime()
    registered = bg.register(health_check_task())

    assert registered.task_id is not None
    assert registered.descriptor is not None
    assert registered.descriptor.task.kind == TaskKind.HEALTH_CHECK

    result = bg.schedule_one(registered.task_id)

    assert result.success is True


def test_scheduler_denial_yields_background_task() -> None:
    attention = AttentionRuntime()
    machine = state_machine()
    budget_runtime = ResourceBudgetRuntime(
        pools=(
            ResourcePool(
                resource=ResourceKind.WORKER_SLOT,
                capacity=1,
                reserved=1,
                conversation_reserved=0,
            ),
        )
    )
    bg = BackgroundTaskRuntime(
        scheduler=scheduler(
            attention=attention,
            budget_runtime=budget_runtime,
            machine=machine,
        ),
        attention=attention,
        state_machine=machine,
    )
    registered = bg.register(memory_consolidation_task())

    assert registered.task_id is not None

    result = bg.schedule_one(registered.task_id)

    assert result.success is False
    assert result.reason == BackgroundTaskReason.SCHEDULER_DENIED


def test_shed_all_marks_background_tasks_shed() -> None:
    bg = runtime()
    bg.register(memory_consolidation_task())
    bg.register(context_prefetch_task())

    results = bg.shed_all()

    assert len(results) == 2
    assert all(result.decision == BackgroundTaskDecision.SHED for result in results)


def test_descriptor_for_returns_registered_task() -> None:
    bg = runtime()
    registered = bg.register(workspace_scan_task())

    assert registered.task_id is not None
    assert bg.descriptor_for(registered.task_id) is not None


def test_snapshot_counts_runtime_state() -> None:
    bg = runtime()
    registered = bg.register(memory_consolidation_task())

    assert registered.task_id is not None

    bg.schedule_one(registered.task_id)
    snapshot = bg.snapshot()

    assert snapshot.registered_count == 1
    assert snapshot.scheduled_count == 1
    assert snapshot.last_reason == BackgroundTaskReason.TASK_SCHEDULED


def test_reset_clears_runtime_state() -> None:
    bg = runtime()
    bg.register(memory_consolidation_task())

    bg.reset()
    snapshot = bg.snapshot()

    assert snapshot.registered_count == 0
    assert snapshot.scheduled_count == 0
    assert snapshot.last_reason == BackgroundTaskReason.RUNTIME_RESET


def test_factory_helpers_create_expected_kinds() -> None:
    assert (
        memory_consolidation_task().kind
        == BackgroundTaskKind.MEMORY_CONSOLIDATION
    )
    assert context_prefetch_task().kind == BackgroundTaskKind.CONTEXT_PREFETCH
    assert workspace_scan_task().kind == BackgroundTaskKind.WORKSPACE_SCAN
    assert health_check_task().kind == BackgroundTaskKind.HEALTH_CHECK
    assert log_rotation_task().kind == BackgroundTaskKind.LOG_ROTATION


def test_enum_values_are_stable() -> None:
    assert BackgroundTaskKind.MEMORY_CONSOLIDATION.value == "memory_consolidation"
    assert BackgroundTaskStatus.SCHEDULED.value == "scheduled"
    assert BackgroundTaskDecision.YIELDED.value == "yielded"
    assert (
        BackgroundTaskReason.TASK_YIELDED_TO_ATTENTION.value
        == "task_yielded_to_attention"
    )