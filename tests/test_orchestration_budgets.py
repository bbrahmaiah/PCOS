from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.orchestration import (
    BudgetDecision,
    BudgetReason,
    BudgetReservationStatus,
    BudgetRuntimePolicy,
    ResourceBudget,
    ResourceBudgetRuntime,
    ResourceBudgetRuntimeConfig,
    ResourceKind,
    ResourcePool,
    TaskKind,
    TaskPriority,
    TaskRequest,
    WorkerCapability,
    action_slots,
    cpu_slots,
    file_operations,
    llm_tokens,
    memory_queries,
    memory_writes,
    network_operations,
    wall_time_ms,
    worker_slots,
)


def task(
    *,
    kind: TaskKind = TaskKind.COGNITION,
    priority: TaskPriority = TaskPriority.NORMAL,
    budgets: tuple[ResourceBudget, ...] | None = None,
    timeout_ms: int | None = None,
) -> TaskRequest:
    return TaskRequest(
        kind=kind,
        priority=priority,
        name="budget task",
        description="budget task",
        required_capabilities=(WorkerCapability.COGNITION,),
        resource_budgets=budgets or (worker_slots(1),),
        timeout_ms=timeout_ms,
    )


def conversation_task(
    *,
    budgets: tuple[ResourceBudget, ...] | None = None,
) -> TaskRequest:
    return TaskRequest(
        kind=TaskKind.CONVERSATION_TURN,
        priority=TaskPriority.CRITICAL,
        name="conversation task",
        description="conversation task",
        required_capabilities=(WorkerCapability.CONVERSATION,),
        resource_budgets=budgets or (worker_slots(1),),
        timeout_ms=5_000,
    )


def runtime_with_worker_slots(
    *,
    capacity: int = 2,
    reserved: int = 0,
    conversation_reserved: int = 1,
) -> ResourceBudgetRuntime:
    return ResourceBudgetRuntime(
        pools=(
            ResourcePool(
                resource=ResourceKind.WORKER_SLOT,
                capacity=capacity,
                reserved=reserved,
                conversation_reserved=conversation_reserved,
            ),
        )
    )


def test_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        ResourceBudgetRuntimeConfig(name=" ").validate()


def test_resource_pool_rejects_reserved_above_capacity() -> None:
    with pytest.raises(ValidationError):
        ResourcePool(
            resource=ResourceKind.CPU_SLOT,
            capacity=1,
            reserved=2,
        )


def test_resource_pool_rejects_conversation_reserve_above_capacity() -> None:
    with pytest.raises(ValidationError):
        ResourcePool(
            resource=ResourceKind.CPU_SLOT,
            capacity=1,
            conversation_reserved=2,
        )


def test_resource_pool_available_values() -> None:
    pool = ResourcePool(
        resource=ResourceKind.WORKER_SLOT,
        capacity=4,
        reserved=1,
        conversation_reserved=1,
    )

    assert pool.available == 3
    assert pool.non_reserved_available == 2


def test_default_pools_cover_core_resources() -> None:
    runtime = ResourceBudgetRuntime()
    resources = {pool.resource for pool in runtime.all_pools()}

    assert ResourceKind.CPU_SLOT in resources
    assert ResourceKind.WORKER_SLOT in resources
    assert ResourceKind.LLM_TOKEN in resources
    assert ResourceKind.MEMORY_QUERY in resources
    assert ResourceKind.MEMORY_WRITE in resources
    assert ResourceKind.ACTION_SLOT in resources
    assert ResourceKind.FILE_OPERATION in resources
    assert ResourceKind.NETWORK_OPERATION in resources
    assert ResourceKind.WALL_TIME_MS in resources


def test_evaluate_allows_available_budget() -> None:
    runtime = runtime_with_worker_slots(capacity=2)

    result = runtime.evaluate(task())

    assert result.allowed is True
    assert result.decision == BudgetDecision.ALLOW
    assert result.reason == BudgetReason.BUDGET_AVAILABLE


def test_evaluate_denies_when_resource_not_configured() -> None:
    runtime = runtime_with_worker_slots(capacity=2)

    result = runtime.evaluate(
        task(budgets=(llm_tokens(100),))
    )

    assert result.allowed is False
    assert result.decision == BudgetDecision.DENY
    assert result.reason == BudgetReason.RESOURCE_NOT_CONFIGURED


def test_evaluate_warns_for_unconfigured_resource_when_allowed() -> None:
    runtime = ResourceBudgetRuntime(
        policy=BudgetRuntimePolicy(allow_unconfigured_resources=True),
        pools=(
            ResourcePool(
                resource=ResourceKind.WORKER_SLOT,
                capacity=2,
            ),
        ),
    )

    result = runtime.evaluate(task(budgets=(llm_tokens(100),)))

    assert result.allowed is True
    assert result.decision == BudgetDecision.WARN
    assert result.reason == BudgetReason.BUDGET_WARNING


def test_conversation_reserve_blocks_background_consumption() -> None:
    runtime = runtime_with_worker_slots(
        capacity=2,
        reserved=1,
        conversation_reserved=1,
    )

    result = runtime.evaluate(task(budgets=(worker_slots(1),)))

    assert result.allowed is False
    assert result.reason == BudgetReason.CONVERSATION_RESERVE_PROTECTED


def test_conversation_task_can_use_conversation_reserve() -> None:
    runtime = runtime_with_worker_slots(
        capacity=2,
        reserved=1,
        conversation_reserved=1,
    )

    result = runtime.evaluate(conversation_task())

    assert result.allowed is True


def test_warning_threshold_returns_warning() -> None:
    runtime = ResourceBudgetRuntime(
        policy=BudgetRuntimePolicy(warning_threshold_percent=50),
        pools=(
            ResourcePool(
                resource=ResourceKind.WORKER_SLOT,
                capacity=2,
                reserved=0,
                conversation_reserved=0,
            ),
        ),
    )

    result = runtime.evaluate(task(budgets=(worker_slots(1),)))

    assert result.allowed is True
    assert result.decision == BudgetDecision.WARN
    assert result.warning_resources == (ResourceKind.WORKER_SLOT,)


def test_reserve_creates_reservation_and_updates_pool() -> None:
    runtime = runtime_with_worker_slots(capacity=2, conversation_reserved=0)
    item = task()

    result = runtime.reserve(item)
    pool = runtime.pool_for(ResourceKind.WORKER_SLOT)

    assert result.success is True
    assert result.reason == BudgetReason.RESERVATION_CREATED
    assert result.reservation is not None
    assert result.reservation.status == BudgetReservationStatus.RESERVED
    assert pool is not None
    assert pool.reserved == 1


def test_duplicate_reservation_returns_existing() -> None:
    runtime = runtime_with_worker_slots(capacity=2, conversation_reserved=0)
    item = task()

    first = runtime.reserve(item)
    second = runtime.reserve(item)

    assert first.success is True
    assert second.success is True
    assert second.reservation == first.reservation


def test_reserve_denied_when_budget_unavailable() -> None:
    runtime = runtime_with_worker_slots(
        capacity=1,
        reserved=1,
        conversation_reserved=0,
    )

    result = runtime.reserve(task())

    assert result.success is False
    assert result.decision == BudgetDecision.DENY


def test_release_restores_pool_capacity() -> None:
    runtime = runtime_with_worker_slots(capacity=2, conversation_reserved=0)
    item = task()

    reserve = runtime.reserve(item)
    release = runtime.release(item.task_id)
    pool = runtime.pool_for(ResourceKind.WORKER_SLOT)

    assert reserve.success is True
    assert release.success is True
    assert release.reason == BudgetReason.RESERVATION_RELEASED
    assert release.reservation is not None
    assert release.reservation.status == BudgetReservationStatus.RELEASED
    assert pool is not None
    assert pool.reserved == 0


def test_release_unknown_reservation_denied() -> None:
    item = task()
    runtime = runtime_with_worker_slots(capacity=2)

    result = runtime.release(item.task_id)

    assert result.success is False
    assert result.reason == BudgetReason.RESERVATION_NOT_FOUND


def test_reservation_for_returns_active_reservation() -> None:
    runtime = runtime_with_worker_slots(capacity=2, conversation_reserved=0)
    item = task()

    runtime.reserve(item)

    assert runtime.reservation_for(item.task_id) is not None


def test_snapshot_counts_runtime_state() -> None:
    runtime = runtime_with_worker_slots(capacity=2, conversation_reserved=0)
    item = task()

    runtime.evaluate(item)
    runtime.reserve(item)
    snapshot = runtime.snapshot()

    assert snapshot.pool_count == 1
    assert snapshot.reservation_count == 1
    assert snapshot.total_capacity == 2
    assert snapshot.total_reserved == 1
    assert snapshot.evaluation_count >= 1
    assert snapshot.allow_count >= 1


def test_reset_metrics_does_not_clear_reservations() -> None:
    runtime = runtime_with_worker_slots(capacity=2, conversation_reserved=0)
    item = task()

    runtime.reserve(item)
    runtime.reset_metrics()
    snapshot = runtime.snapshot()

    assert snapshot.evaluation_count == 0
    assert snapshot.reservation_count == 1


def test_budget_helpers_create_correct_resources() -> None:
    assert cpu_slots(1).resource == ResourceKind.CPU_SLOT
    assert worker_slots(1).resource == ResourceKind.WORKER_SLOT
    assert llm_tokens(1).resource == ResourceKind.LLM_TOKEN
    assert memory_queries(1).resource == ResourceKind.MEMORY_QUERY
    assert memory_writes(1).resource == ResourceKind.MEMORY_WRITE
    assert action_slots(1).resource == ResourceKind.ACTION_SLOT
    assert file_operations(1).resource == ResourceKind.FILE_OPERATION
    assert network_operations(1).resource == ResourceKind.NETWORK_OPERATION
    assert wall_time_ms(1).resource == ResourceKind.WALL_TIME_MS


def test_enum_values_are_stable() -> None:
    assert BudgetDecision.ALLOW.value == "allow"
    assert BudgetReason.BUDGET_AVAILABLE.value == "budget_available"
    assert BudgetReservationStatus.RESERVED.value == "reserved"