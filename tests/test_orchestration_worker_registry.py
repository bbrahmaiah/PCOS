from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.orchestration import (
    ResourceBudget,
    ResourceKind,
    TaskKind,
    TaskPriority,
    TaskRequest,
    WorkerAvailability,
    WorkerCapability,
    WorkerContract,
    WorkerDescriptor,
    WorkerHealthState,
    WorkerLoad,
    WorkerRegistrationDecision,
    WorkerRegistry,
    WorkerRegistryConfig,
    WorkerRegistryReason,
    new_worker_id,
)


def budget(resource: ResourceKind = ResourceKind.WORKER_SLOT) -> ResourceBudget:
    return ResourceBudget(resource=resource, amount=1)


def worker_contract(
    *,
    name: str = "cognition worker",
    worker_id: str | None = None,
    capabilities: tuple[WorkerCapability, ...] = (
        WorkerCapability.COGNITION,
    ),
    accepted_task_kinds: tuple[TaskKind, ...] = (TaskKind.COGNITION,),
    max_concurrent_tasks: int = 1,
) -> WorkerContract:
    return WorkerContract(
        worker_id=worker_id or new_worker_id(),
        name=name,
        capabilities=capabilities,
        accepted_task_kinds=accepted_task_kinds,
        max_concurrent_tasks=max_concurrent_tasks,
        resource_budgets=(budget(),),
    )


def descriptor(
    *,
    contract: WorkerContract | None = None,
    health: WorkerHealthState = WorkerHealthState.HEALTHY,
    availability: WorkerAvailability | None = None,
    enabled: bool = True,
) -> WorkerDescriptor:
    final_contract = contract or worker_contract()

    return WorkerRegistry.descriptor_from_contract(
        final_contract,
        health=health,
        availability=availability,
        enabled=enabled,
    )


def task(
    *,
    kind: TaskKind = TaskKind.COGNITION,
    capabilities: tuple[WorkerCapability, ...] = (
        WorkerCapability.COGNITION,
    ),
) -> TaskRequest:
    return TaskRequest(
        kind=kind,
        priority=TaskPriority.NORMAL,
        name="test task",
        description="test task",
        required_capabilities=capabilities,
        resource_budgets=(budget(),),
    )


def test_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        WorkerRegistryConfig(name=" ").validate()


def test_worker_load_rejects_impossible_load_factor() -> None:
    with pytest.raises(ValidationError):
        WorkerLoad(
            active_tasks=1,
            queued_tasks=0,
            max_concurrent_tasks=2,
            load_factor=0.1,
        )


def test_worker_load_reports_capacity() -> None:
    load = WorkerLoad(
        active_tasks=2,
        queued_tasks=0,
        max_concurrent_tasks=2,
        load_factor=1.0,
    )

    assert load.at_capacity is True
    assert load.has_capacity is False


def test_descriptor_requires_disabled_health_when_disabled() -> None:
    contract = worker_contract()

    with pytest.raises(ValidationError):
        WorkerDescriptor(
            contract=contract,
            health=WorkerHealthState.HEALTHY,
            availability=WorkerAvailability.DISABLED,
            enabled=False,
            load=WorkerLoad(
                active_tasks=0,
                queued_tasks=0,
                max_concurrent_tasks=contract.max_concurrent_tasks,
            ),
        )


def test_descriptor_from_contract_creates_schedulable_worker() -> None:
    item = descriptor()

    assert item.enabled is True
    assert item.healthy is True
    assert item.schedulable is True


def test_register_worker_success() -> None:
    registry = WorkerRegistry()
    item = descriptor()

    result = registry.register(item)

    assert result.success is True
    assert result.decision == WorkerRegistrationDecision.REGISTERED
    assert result.worker_id == item.worker_id
    assert registry.snapshot().worker_count == 1


def test_duplicate_worker_rejected_by_default() -> None:
    registry = WorkerRegistry()
    item = descriptor()

    first = registry.register(item)
    second = registry.register(item)

    assert first.success is True
    assert second.success is False
    assert second.reason == WorkerRegistryReason.DUPLICATE_WORKER_REJECTED
    assert registry.snapshot().worker_count == 1


def test_duplicate_worker_can_replace_when_configured() -> None:
    registry = WorkerRegistry(
        config=WorkerRegistryConfig(allow_replace=True)
    )
    item = descriptor()
    replacement = item.model_copy(
        update={"health": WorkerHealthState.DEGRADED}
    )

    first = registry.register(item)
    second = registry.register(replacement)

    assert first.success is True
    assert second.success is True
    assert second.decision == WorkerRegistrationDecision.UPDATED
    assert registry.snapshot().worker_count == 1


def test_get_worker_found() -> None:
    registry = WorkerRegistry()
    item = descriptor()
    registry.register(item)

    result = registry.get(item.worker_id)

    assert result.found is True
    assert result.descriptor is not None
    assert result.descriptor.worker_id == item.worker_id


def test_get_worker_not_found() -> None:
    registry = WorkerRegistry()
    worker_id = new_worker_id()

    result = registry.get(worker_id)

    assert result.found is False
    assert result.descriptor is None
    assert result.reason == WorkerRegistryReason.WORKER_NOT_FOUND


def test_enabled_healthy_and_schedulable_views() -> None:
    registry = WorkerRegistry()
    healthy = descriptor()
    unhealthy = descriptor(
        contract=worker_contract(name="memory worker"),
        health=WorkerHealthState.UNHEALTHY,
    )
    disabled = descriptor(
        contract=worker_contract(name="tool worker"),
        enabled=False,
    )

    registry.register(healthy)
    registry.register(unhealthy)
    registry.register(disabled)

    assert len(registry.all_workers()) == 3
    assert len(registry.enabled_workers()) == 2
    assert len(registry.healthy_workers()) == 1
    assert len(registry.schedulable_workers()) == 1


def test_find_by_capability_returns_schedulable_matches_only() -> None:
    registry = WorkerRegistry()
    cognition = descriptor()
    memory = descriptor(
        contract=worker_contract(
            name="memory worker",
            capabilities=(WorkerCapability.MEMORY,),
            accepted_task_kinds=(TaskKind.MEMORY_RETRIEVAL,),
        )
    )
    disabled_tool = descriptor(
        contract=worker_contract(
            name="tool worker",
            capabilities=(WorkerCapability.TOOL_ACTION,),
            accepted_task_kinds=(TaskKind.TOOL_ACTION,),
        ),
        enabled=False,
    )

    registry.register(cognition)
    registry.register(memory)
    registry.register(disabled_tool)

    matches = registry.find_by_capability(WorkerCapability.MEMORY)

    assert len(matches) == 1
    assert matches[0].name == "memory worker"


def test_find_for_task_returns_contract_matches() -> None:
    registry = WorkerRegistry()
    cognition = descriptor()
    memory = descriptor(
        contract=worker_contract(
            name="memory worker",
            capabilities=(WorkerCapability.MEMORY,),
            accepted_task_kinds=(TaskKind.MEMORY_RETRIEVAL,),
        )
    )
    registry.register(cognition)
    registry.register(memory)

    matches = registry.find_for_task(task(kind=TaskKind.COGNITION))

    assert len(matches) == 1
    assert matches[0].worker_id == cognition.worker_id


def test_find_for_task_excludes_busy_worker() -> None:
    registry = WorkerRegistry()
    item = descriptor()
    registry.register(item)
    registry.update_load(item.worker_id, active_tasks=1)

    matches = registry.find_for_task(task(kind=TaskKind.COGNITION))

    assert matches == ()


def test_update_health_changes_availability() -> None:
    registry = WorkerRegistry()
    item = descriptor(health=WorkerHealthState.UNKNOWN)
    registry.register(item)

    result = registry.update_health(item.worker_id, WorkerHealthState.HEALTHY)

    assert result.success is True
    assert result.descriptor is not None
    assert result.descriptor.health == WorkerHealthState.HEALTHY
    assert result.descriptor.availability == WorkerAvailability.AVAILABLE


def test_update_health_unknown_worker_rejected() -> None:
    registry = WorkerRegistry()

    result = registry.update_health(
        new_worker_id(),
        WorkerHealthState.HEALTHY,
    )

    assert result.success is False
    assert result.reason == WorkerRegistryReason.WORKER_NOT_FOUND


def test_update_load_marks_worker_busy_at_capacity() -> None:
    registry = WorkerRegistry()
    contract = worker_contract(max_concurrent_tasks=2)
    item = descriptor(contract=contract)
    registry.register(item)

    result = registry.update_load(
        item.worker_id,
        active_tasks=2,
        queued_tasks=1,
    )

    assert result.success is True
    assert result.descriptor is not None
    assert result.descriptor.availability == WorkerAvailability.BUSY
    assert result.descriptor.load.at_capacity is True


def test_disable_worker_removes_from_schedulable() -> None:
    registry = WorkerRegistry()
    item = descriptor()
    registry.register(item)

    result = registry.disable(item.worker_id)

    assert result.success is True
    assert result.descriptor is not None
    assert result.descriptor.enabled is False
    assert result.descriptor.health == WorkerHealthState.DISABLED
    assert registry.schedulable_workers() == ()


def test_snapshot_counts_registry_state() -> None:
    registry = WorkerRegistry()
    healthy = descriptor()
    busy = descriptor(contract=worker_contract(max_concurrent_tasks=1))
    unhealthy = descriptor(
        contract=worker_contract(name="bad worker"),
        health=WorkerHealthState.FAILED,
    )
    disabled = descriptor(
        contract=worker_contract(name="disabled worker"),
        enabled=False,
    )

    registry.register(healthy)
    registry.register(busy)
    registry.update_load(busy.worker_id, active_tasks=1)
    registry.register(unhealthy)
    registry.register(disabled)

    snapshot = registry.snapshot()

    assert snapshot.worker_count == 4
    assert snapshot.enabled_count == 3
    assert snapshot.healthy_count == 2
    assert snapshot.schedulable_count == 1
    assert snapshot.busy_count == 1
    assert snapshot.disabled_count == 1
    assert snapshot.unhealthy_count == 1


def test_reset_clears_registry() -> None:
    registry = WorkerRegistry()
    registry.register(descriptor())

    registry.reset()

    assert registry.snapshot().worker_count == 0
    assert registry.all_workers() == ()


def test_enum_values_are_stable() -> None:
    assert WorkerHealthState.HEALTHY.value == "healthy"
    assert WorkerAvailability.AVAILABLE.value == "available"
    assert WorkerRegistrationDecision.REGISTERED.value == "registered"
    assert WorkerRegistryReason.WORKER_AT_CAPACITY.value == "worker_at_capacity"