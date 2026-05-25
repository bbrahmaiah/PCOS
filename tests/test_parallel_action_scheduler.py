from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.tools import (
    ActionKind,
    ActionPlan,
    ActionRisk,
    ActionScheduleDecision,
    ActionSchedulePriority,
    ActionScheduleReason,
    ActionScheduleState,
    ActionScope,
    ActionStatus,
    ParallelActionScheduler,
    ParallelActionSchedulerConfig,
    PermissionDecision,
    ResourceLockKind,
    ResourceLockMode,
    ScheduledAction,
    ScheduledActionLock,
    ToolCapability,
)
from jarvis.tools.models import ActionStep


def step(
    *,
    action_id: str,
    order: int = 0,
    kind: ActionKind = ActionKind.READ,
    scope: ActionScope = ActionScope.WORKSPACE,
    risk: ActionRisk = ActionRisk.LOW,
    arguments: dict[str, object] | None = None,
) -> ActionStep:
    timeout_ms = 30_000 if risk in {ActionRisk.HIGH, ActionRisk.CRITICAL} else None

    return ActionStep(
        action_id=action_id,
        order=order,
        kind=kind,
        capability=ToolCapability.READ_FILE,
        scope=scope,
        risk=risk,
        description="test step",
        arguments=arguments or {},
        timeout_ms=timeout_ms,
        interruptible=True,
        rollback_supported=False,
    )


def plan(
    *,
    action_id: str,
    risk: ActionRisk = ActionRisk.LOW,
    scope: ActionScope = ActionScope.WORKSPACE,
    requires_approval: bool = False,
    steps: tuple[ActionStep, ...] | None = None,
) -> ActionPlan:
    return ActionPlan(
        action_id=action_id,
        goal="test action",
        steps=steps or (step(action_id=action_id, scope=scope, risk=risk),),
        risk=risk,
        scope=scope,
        requires_approval=requires_approval,
        permission_decision=(
            PermissionDecision.REQUIRE_APPROVAL
            if requires_approval
            else PermissionDecision.ALLOW
        ),
        status=ActionStatus.PLANNED,
    )


def test_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        ParallelActionSchedulerConfig(name=" ").validate()

    with pytest.raises(ValueError):
        ParallelActionSchedulerConfig(max_parallel_actions=0).validate()

    with pytest.raises(ValueError):
        ParallelActionSchedulerConfig(max_parallel_shell_actions=0).validate()


def test_lock_requires_key() -> None:
    with pytest.raises(ValidationError):
        ScheduledActionLock(kind=ResourceLockKind.FILE_PATH, key=" ")


def test_read_locks_do_not_conflict() -> None:
    left = ScheduledActionLock(
        kind=ResourceLockKind.FILE_PATH,
        key="a.py",
        mode=ResourceLockMode.READ,
    )
    right = ScheduledActionLock(
        kind=ResourceLockKind.FILE_PATH,
        key="a.py",
        mode=ResourceLockMode.READ,
    )

    assert left.conflicts_with(right) is False


def test_write_lock_conflicts_with_read() -> None:
    left = ScheduledActionLock(
        kind=ResourceLockKind.FILE_PATH,
        key="a.py",
        mode=ResourceLockMode.WRITE,
    )
    right = ScheduledActionLock(
        kind=ResourceLockKind.FILE_PATH,
        key="a.py",
        mode=ResourceLockMode.READ,
    )

    assert left.conflicts_with(right) is True


def test_scheduled_action_requires_matching_action_id() -> None:
    with pytest.raises(ValidationError):
        ScheduledAction(
            action_id="action-1",
            plan=plan(action_id="action-2"),
        )


def test_submit_low_risk_validated_action_is_ready() -> None:
    scheduler = ParallelActionScheduler()

    result = scheduler.submit(
        plan(action_id="action-1"),
        validated=True,
    )
    scheduled = scheduler.scheduled_action("action-1")

    assert result.decision == ActionScheduleDecision.ACCEPTED
    assert result.next_state == ActionScheduleState.READY
    assert scheduled is not None
    assert scheduled.cancellation_token is not None


def test_submit_waits_for_validation() -> None:
    scheduler = ParallelActionScheduler()

    result = scheduler.submit(plan(action_id="action-1"))

    assert result.next_state == ActionScheduleState.WAITING_VALIDATION


def test_validate_action_moves_to_ready() -> None:
    scheduler = ParallelActionScheduler()
    scheduler.submit(plan(action_id="action-1"))

    result = scheduler.validate("action-1")

    assert result.decision == ActionScheduleDecision.READY
    assert result.next_state == ActionScheduleState.READY


def test_high_risk_waits_for_approval() -> None:
    scheduler = ParallelActionScheduler()

    result = scheduler.submit(
        plan(
            action_id="action-1",
            risk=ActionRisk.HIGH,
            requires_approval=True,
        ),
        validated=True,
    )

    assert result.next_state == ActionScheduleState.WAITING_APPROVAL


def test_approve_high_risk_then_ready() -> None:
    scheduler = ParallelActionScheduler()
    scheduler.submit(
        plan(
            action_id="action-1",
            risk=ActionRisk.HIGH,
            requires_approval=True,
        ),
        validated=True,
    )

    result = scheduler.approve("action-1")

    assert result.next_state == ActionScheduleState.READY


def test_unknown_dependency_rejected() -> None:
    scheduler = ParallelActionScheduler()

    result = scheduler.submit(
        plan(action_id="action-2"),
        dependencies=("missing",),
    )

    assert result.decision == ActionScheduleDecision.REJECTED
    assert result.reason == ActionScheduleReason.INVALID_DEPENDENCY


def test_dependency_waits_until_success() -> None:
    scheduler = ParallelActionScheduler()
    scheduler.submit(plan(action_id="action-1"), validated=True)
    result = scheduler.submit(
        plan(action_id="action-2"),
        dependencies=("action-1",),
        validated=True,
    )

    assert result.next_state == ActionScheduleState.WAITING_DEPENDENCIES

    scheduler.start_action("action-1")
    scheduler.complete("action-1")
    ready = scheduler.refresh_ready()

    assert ready[0].action_id == "action-2"


def test_failed_dependency_blocks_dependent_action() -> None:
    scheduler = ParallelActionScheduler()
    scheduler.submit(plan(action_id="action-1"), validated=True)
    scheduler.submit(
        plan(action_id="action-2"),
        dependencies=("action-1",),
        validated=True,
    )

    scheduler.start_action("action-1")
    scheduler.complete("action-1", success=False)
    scheduler.refresh_ready()
    scheduled = scheduler.scheduled_action("action-2")

    assert scheduled is not None
    assert scheduled.state == ActionScheduleState.BLOCKED


def test_start_ready_respects_global_concurrency_limit() -> None:
    scheduler = ParallelActionScheduler(
        config=ParallelActionSchedulerConfig(max_parallel_actions=1)
    )
    scheduler.submit(plan(action_id="action-1"), validated=True)
    scheduler.submit(plan(action_id="action-2"), validated=True)

    results = scheduler.start_ready()
    running = scheduler.running_actions()

    assert len(running) == 1
    assert results[0].decision == ActionScheduleDecision.STARTED
    assert results[1].decision == ActionScheduleDecision.DEFERRED
    assert results[1].reason == ActionScheduleReason.CONCURRENCY_LIMIT_REACHED


def test_start_ready_respects_priority() -> None:
    scheduler = ParallelActionScheduler(
        config=ParallelActionSchedulerConfig(max_parallel_actions=1)
    )
    scheduler.submit(
        plan(action_id="low"),
        priority=ActionSchedulePriority.LOW,
        validated=True,
    )
    scheduler.submit(
        plan(action_id="high"),
        priority=ActionSchedulePriority.HIGH,
        validated=True,
    )

    scheduler.start_ready()
    running = scheduler.running_actions()

    assert running[0].action_id == "high"


def test_shell_actions_are_serialized() -> None:
    scheduler = ParallelActionScheduler(
        config=ParallelActionSchedulerConfig(max_parallel_actions=3)
    )
    shell_steps_1 = (
        step(
            action_id="action-1",
            kind=ActionKind.SHELL_COMMAND,
            scope=ActionScope.SHELL,
            arguments={"command": "pytest"},
        ),
    )
    shell_steps_2 = (
        step(
            action_id="action-2",
            kind=ActionKind.SHELL_COMMAND,
            scope=ActionScope.SHELL,
            arguments={"command": "mypy ."},
        ),
    )
    scheduler.submit(
        plan(action_id="action-1", scope=ActionScope.SHELL, steps=shell_steps_1),
        validated=True,
    )
    scheduler.submit(
        plan(action_id="action-2", scope=ActionScope.SHELL, steps=shell_steps_2),
        validated=True,
    )

    scheduler.start_ready()
    running = scheduler.running_actions()

    assert len(running) == 1
    assert running[0].plan.scope == ActionScope.SHELL


def test_file_write_collision_blocks_parallel_start() -> None:
    scheduler = ParallelActionScheduler(
        config=ParallelActionSchedulerConfig(max_parallel_actions=3)
    )
    write_step_1 = (
        step(
            action_id="action-1",
            kind=ActionKind.WRITE,
            arguments={"path": "a.py"},
        ),
    )
    write_step_2 = (
        step(
            action_id="action-2",
            kind=ActionKind.WRITE,
            arguments={"path": "a.py"},
        ),
    )

    scheduler.submit(plan(action_id="action-1", steps=write_step_1), validated=True)
    scheduler.submit(plan(action_id="action-2", steps=write_step_2), validated=True)
    results = scheduler.start_ready()
    running = scheduler.running_actions()

    assert len(running) == 1
    assert results[1].reason == ActionScheduleReason.RESOURCE_COLLISION


def test_two_file_reads_can_run_parallel() -> None:
    scheduler = ParallelActionScheduler(
        config=ParallelActionSchedulerConfig(max_parallel_actions=3)
    )
    read_step_1 = (
        step(
            action_id="action-1",
            kind=ActionKind.READ,
            arguments={"path": "a.py"},
        ),
    )
    read_step_2 = (
        step(
            action_id="action-2",
            kind=ActionKind.READ,
            arguments={"path": "a.py"},
        ),
    )

    scheduler.submit(plan(action_id="action-1", steps=read_step_1), validated=True)
    scheduler.submit(plan(action_id="action-2", steps=read_step_2), validated=True)
    scheduler.start_ready()

    assert len(scheduler.running_actions()) == 2


def test_cancel_action_updates_state() -> None:
    scheduler = ParallelActionScheduler()
    scheduler.submit(plan(action_id="action-1"), validated=True)

    result = scheduler.cancel("action-1")
    scheduled = scheduler.scheduled_action("action-1")

    assert result.decision == ActionScheduleDecision.CANCELLED
    assert scheduled is not None
    assert scheduled.state == ActionScheduleState.CANCELLED


def test_complete_action_updates_state() -> None:
    scheduler = ParallelActionScheduler()
    scheduler.submit(plan(action_id="action-1"), validated=True)
    scheduler.start_action("action-1")

    result = scheduler.complete("action-1")
    scheduled = scheduler.scheduled_action("action-1")

    assert result.decision == ActionScheduleDecision.COMPLETED
    assert scheduled is not None
    assert scheduled.state == ActionScheduleState.SUCCEEDED


def test_unknown_action_returns_not_found() -> None:
    scheduler = ParallelActionScheduler()

    result = scheduler.start_action("missing")

    assert result.decision == ActionScheduleDecision.REJECTED
    assert result.reason == ActionScheduleReason.ACTION_NOT_FOUND


def test_snapshot_and_reset() -> None:
    scheduler = ParallelActionScheduler()
    scheduler.submit(plan(action_id="action-1"), validated=True)
    scheduler.start_action("action-1")

    snapshot = scheduler.snapshot()

    assert snapshot.total_count == 1
    assert snapshot.running_count == 1

    scheduler.reset()
    reset_snapshot = scheduler.snapshot()

    assert reset_snapshot.total_count == 0
    assert reset_snapshot.last_decision is None


def test_enum_values_are_stable() -> None:
    assert ActionSchedulePriority.HIGH.value == "high"
    assert ActionScheduleState.RUNNING.value == "running"
    assert ResourceLockKind.FILE_PATH.value == "file_path"
    assert ResourceLockMode.EXCLUSIVE.value == "exclusive"