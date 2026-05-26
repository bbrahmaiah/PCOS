from __future__ import annotations

import pytest

from jarvis.orchestration import (
    AttentionContext,
    AttentionRuntime,
    JobRequest,
    OrchestrationStateMachine,
    ResourceBudgetRuntime,
    ResourceKind,
    ResourcePool,
    TaskDependency,
    TaskGraph,
    TaskKind,
    TaskPriority,
    TaskRequest,
    TaskResult,
    TaskScheduleDecision,
    TaskScheduler,
    TaskSchedulerConfig,
    TaskScheduleReason,
    TaskStatus,
    WorkerCapability,
    WorkerContract,
    WorkerHealthState,
    WorkerRegistry,
    new_task_id,
    worker_slots,
)


def task(
    *,
    task_id: str | None = None,
    kind: TaskKind = TaskKind.COGNITION,
    priority: TaskPriority = TaskPriority.NORMAL,
    capability: WorkerCapability = WorkerCapability.COGNITION,
    dependencies: tuple[TaskDependency, ...] = (),
    background: bool = False,
    timeout_ms: int | None = None,
) -> TaskRequest:
    return TaskRequest(
        task_id=task_id or new_task_id(),
        kind=kind,
        priority=priority,
        name="scheduler task",
        description="scheduler task",
        required_capabilities=(capability,),
        resource_budgets=(worker_slots(1),),
        dependencies=dependencies,
        background=background,
        timeout_ms=timeout_ms,
    )


def conversation_task() -> TaskRequest:
    return task(
        kind=TaskKind.CONVERSATION_TURN,
        priority=TaskPriority.CRITICAL,
        capability=WorkerCapability.CONVERSATION,
        timeout_ms=5_000,
    )


def background_task() -> TaskRequest:
    return task(
        kind=TaskKind.BACKGROUND_MAINTENANCE,
        priority=TaskPriority.BACKGROUND,
        capability=WorkerCapability.BACKGROUND,
        background=True,
    )


def worker(
    *,
    capability: WorkerCapability = WorkerCapability.COGNITION,
    task_kind: TaskKind = TaskKind.COGNITION,
    max_concurrent_tasks: int = 1,
) -> WorkerContract:
    return WorkerContract(
        name=f"{capability.value} worker",
        capabilities=(capability,),
        accepted_task_kinds=(task_kind,),
        max_concurrent_tasks=max_concurrent_tasks,
        resource_budgets=(worker_slots(1),),
    )


def job(*, tasks: tuple[TaskRequest, ...]) -> JobRequest:
    return JobRequest(
        name="scheduler job",
        description="scheduler job",
        tasks=tasks,
    )


def graph(*tasks: TaskRequest) -> TaskGraph:
    return TaskGraph(job=job(tasks=tasks))


def registry_with_worker(
    *,
    capability: WorkerCapability = WorkerCapability.COGNITION,
    task_kind: TaskKind = TaskKind.COGNITION,
    max_concurrent_tasks: int = 1,
    health: WorkerHealthState = WorkerHealthState.HEALTHY,
) -> WorkerRegistry:
    registry = WorkerRegistry()
    contract = worker(
        capability=capability,
        task_kind=task_kind,
        max_concurrent_tasks=max_concurrent_tasks,
    )
    registry.register(
        WorkerRegistry.descriptor_from_contract(
            contract,
            health=health,
        )
    )

    return registry


def budgets(*, capacity: int = 4, reserved: int = 0) -> ResourceBudgetRuntime:
    return ResourceBudgetRuntime(
        pools=(
            ResourcePool(
                resource=ResourceKind.WORKER_SLOT,
                capacity=capacity,
                reserved=reserved,
                conversation_reserved=0,
            ),
        )
    )


def ready_state_machine() -> OrchestrationStateMachine:
    machine = OrchestrationStateMachine()
    machine.bootstrap_completed()

    return machine


def scheduler(
    *,
    registry: WorkerRegistry | None = None,
    attention: AttentionRuntime | None = None,
    budget_runtime: ResourceBudgetRuntime | None = None,
    state_machine: OrchestrationStateMachine | None = None,
    config: TaskSchedulerConfig | None = None,
) -> TaskScheduler:
    return TaskScheduler(
        registry=registry or registry_with_worker(),
        attention=attention or AttentionRuntime(),
        budgets=budget_runtime or budgets(),
        state_machine=state_machine or ready_state_machine(),
        config=config,
    )


def test_config_rejects_invalid_cycle_limit() -> None:
    with pytest.raises(ValueError):
        TaskSchedulerConfig(max_tasks_per_cycle=0).validate()


def test_scheduler_defers_when_orchestrator_not_ready() -> None:
    item = task()
    machine = OrchestrationStateMachine()
    runtime = scheduler(state_machine=machine)

    result = runtime.schedule_ready(graph(item))

    assert result.deferred_count == 1
    assert result.results[0].reason == TaskScheduleReason.ORCHESTRATOR_NOT_READY


def test_scheduler_schedules_ready_task() -> None:
    item = task()
    runtime = scheduler()

    result = runtime.schedule_ready(graph(item))

    assert result.scheduled_count == 1
    assert result.results[0].decision == TaskScheduleDecision.SCHEDULED
    assert result.results[0].worker_id is not None
    assert runtime.assignment_for(item.task_id) is not None


def test_scheduler_uses_graph_readiness_not_guessing() -> None:
    first_id = new_task_id()
    second_id = new_task_id()
    first = task(task_id=first_id)
    second = task(
        task_id=second_id,
        dependencies=(TaskDependency(task_id=first_id),),
    )
    runtime = scheduler()

    result = runtime.schedule_ready(graph(first, second))

    assert result.scheduled_count == 1
    assert result.scheduled[0].task_id == first_id
    assert runtime.assignment_for(second_id) is None


def test_scheduler_schedules_downstream_after_dependency_result() -> None:
    first_id = new_task_id()
    second_id = new_task_id()
    first = task(task_id=first_id)
    second = task(
        task_id=second_id,
        dependencies=(TaskDependency(task_id=first_id),),
    )
    item_graph = graph(first, second)
    mutation = item_graph.add_result(
        TaskResult(
            task_id=first_id,
            status=TaskStatus.SUCCEEDED,
            success=True,
        )
    )

    assert mutation.graph is not None

    runtime = scheduler()

    result = runtime.schedule_ready(mutation.graph)

    assert result.scheduled_count == 1
    assert result.scheduled[0].task_id == second_id


def test_scheduler_defers_background_during_conversation() -> None:
    runtime = scheduler(
        registry=registry_with_worker(
            capability=WorkerCapability.BACKGROUND,
            task_kind=TaskKind.BACKGROUND_MAINTENANCE,
        ),
        attention=AttentionRuntime(
            context=AttentionContext(active_conversation=True)
        ),
    )

    result = runtime.schedule_ready(graph(background_task()))

    assert result.deferred_count == 1
    assert result.results[0].reason in {
        TaskScheduleReason.ATTENTION_DEFERRED,
        TaskScheduleReason.ATTENTION_SUPPRESSED,
    }


def test_scheduler_allows_conversation_task() -> None:
    runtime = scheduler(
        registry=registry_with_worker(
            capability=WorkerCapability.CONVERSATION,
            task_kind=TaskKind.CONVERSATION_TURN,
        ),
        attention=AttentionRuntime(
            context=AttentionContext(active_conversation=True)
        ),
    )

    result = runtime.schedule_ready(graph(conversation_task()))

    assert result.scheduled_count == 1


def test_scheduler_denies_when_budget_denied() -> None:
    item = task()
    runtime = scheduler(
        budget_runtime=budgets(capacity=1, reserved=1),
    )

    result = runtime.schedule_ready(graph(item))

    assert result.denied_count == 1
    assert result.results[0].reason == TaskScheduleReason.BUDGET_DENIED


def test_scheduler_defers_when_no_worker_available() -> None:
    item = task()
    runtime = scheduler(registry=WorkerRegistry())

    result = runtime.schedule_ready(graph(item))

    assert result.deferred_count == 1
    assert result.results[0].reason == TaskScheduleReason.NO_WORKER_AVAILABLE


def test_scheduler_excludes_unhealthy_workers() -> None:
    item = task()
    runtime = scheduler(
        registry=registry_with_worker(health=WorkerHealthState.UNHEALTHY)
    )

    result = runtime.schedule_ready(graph(item))

    assert result.deferred_count == 1
    assert result.results[0].reason == TaskScheduleReason.NO_WORKER_AVAILABLE


def test_scheduler_respects_priority_order() -> None:
    low = task(priority=TaskPriority.LOW)
    high = task(priority=TaskPriority.HIGH, timeout_ms=10_000)
    registry = registry_with_worker(max_concurrent_tasks=2)
    runtime = scheduler(registry=registry, budget_runtime=budgets(capacity=4))

    result = runtime.schedule_ready(graph(low, high))

    assert result.scheduled_count == 2
    assert result.scheduled[0].task_id == high.task_id
    assert result.scheduled[1].task_id == low.task_id


def test_scheduler_respects_max_tasks_per_cycle() -> None:
    first = task()
    second = task()
    registry = registry_with_worker(max_concurrent_tasks=2)
    runtime = scheduler(
        registry=registry,
        budget_runtime=budgets(capacity=4),
        config=TaskSchedulerConfig(max_tasks_per_cycle=1),
    )

    result = runtime.schedule_ready(graph(first, second))

    assert result.scheduled_count == 1


def test_scheduler_skips_duplicate_assignment() -> None:
    item = task()
    runtime = scheduler()

    first = runtime.schedule_ready(graph(item))
    second = runtime.schedule_ready(graph(item))

    assert first.scheduled_count == 1
    assert second.skipped_count == 1
    assert second.results[0].reason == TaskScheduleReason.TASK_ALREADY_SCHEDULED


def test_complete_releases_assignment_and_budget() -> None:
    item = task()
    budget_runtime = budgets(capacity=2)
    runtime = scheduler(budget_runtime=budget_runtime)

    scheduled = runtime.schedule_ready(graph(item))
    completed = runtime.complete(item.task_id)
    pool = budget_runtime.pool_for(ResourceKind.WORKER_SLOT)

    assert scheduled.scheduled_count == 1
    assert completed.success is True
    assert runtime.assignment_for(item.task_id) is None
    assert pool is not None
    assert pool.reserved == 0


def test_complete_unknown_assignment_skips() -> None:
    runtime = scheduler()
    result = runtime.complete(new_task_id())

    assert result.success is False
    assert result.decision == TaskScheduleDecision.SKIPPED


def test_snapshot_tracks_scheduler_metrics() -> None:
    item = task()
    runtime = scheduler()

    runtime.schedule_ready(graph(item))
    snapshot = runtime.snapshot()

    assert snapshot.scheduled_count == 1
    assert snapshot.active_assignment_count == 1
    assert snapshot.last_decision == TaskScheduleDecision.SCHEDULED


def test_reset_clears_scheduler_state() -> None:
    item = task()
    runtime = scheduler()
    runtime.schedule_ready(graph(item))

    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.scheduled_count == 0
    assert snapshot.active_assignment_count == 0


def test_no_ready_tasks_returns_empty_batch() -> None:
    first_id = new_task_id()
    second_id = new_task_id()
    first = task(task_id=first_id)
    second = task(
        task_id=second_id,
        dependencies=(TaskDependency(task_id=first_id),),
    )
    item_graph = graph(first, second)
    first_result = item_graph.add_result(
        TaskResult(
            task_id=first_id,
            status=TaskStatus.FAILED,
            success=False,
        )
    )

    assert first_result.graph is not None

    runtime = scheduler()
    result = runtime.schedule_ready(first_result.graph)

    assert result.results == ()
    assert result.scheduled_count == 0


def test_enum_values_are_stable() -> None:
    assert TaskScheduleDecision.SCHEDULED.value == "scheduled"
    assert TaskScheduleReason.TASK_SCHEDULED.value == "task_scheduled"