from __future__ import annotations

import pytest

from jarvis.cognitive import (
    Goal,
    GoalPriority,
    GoalStatus,
    PlanCreateRequest,
    PlanIntentKind,
    PlanningOperation,
    PlanningRuntime,
    PlanningRuntimeStatus,
    PlanRecallRequest,
    PlanRisk,
    PlanStepStatus,
    PlanStepUpdateRequest,
)
from jarvis.cognitive.contracts import utc_now


def _goal(
    *,
    status: GoalStatus = GoalStatus.ACTIVE,
    priority: GoalPriority = GoalPriority.HIGH,
) -> Goal:
    now = utc_now()
    return Goal(
        goal_id="goal_1",
        title="Build Phase 9",
        description="Build cognitive presence.",
        status=status,
        priority=priority,
        created_at=now,
        updated_at=now,
        tags=("phase9",),
    )


def test_plan_create_request_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        PlanCreateRequest(goal=_goal(), title=" ")

    with pytest.raises(ValueError):
        PlanCreateRequest(goal=_goal(), max_steps=0)


def test_planning_runtime_creates_developer_plan() -> None:
    runtime = PlanningRuntime()

    result = runtime.create_plan(
        PlanCreateRequest(
            goal=_goal(),
            intent_kind=PlanIntentKind.DEVELOPER,
        )
    )

    assert result.status == PlanningRuntimeStatus.READY
    assert result.operation == PlanningOperation.CREATE_PLAN
    assert result.plan is not None
    assert result.plan.goal_id == "goal_1"
    assert result.state.active_plan_id == result.plan.plan_id
    assert result.plan.steps
    assert result.plan.steps[0].status == PlanStepStatus.READY
    assert result.plan.steps[-1].metadata["verification"] is True


def test_planning_runtime_blocks_completed_goal() -> None:
    runtime = PlanningRuntime()

    result = runtime.create_plan(
        PlanCreateRequest(goal=_goal(status=GoalStatus.COMPLETED))
    )

    assert result.status == PlanningRuntimeStatus.BLOCKED
    assert result.succeeded is False
    assert result.plan is None


def test_planning_runtime_creates_custom_steps() -> None:
    runtime = PlanningRuntime()

    result = runtime.create_plan(
        PlanCreateRequest(
            goal=_goal(),
            requested_steps=(
                "Inspect contracts.",
                "Review risks.",
            ),
            max_steps=4,
        )
    )

    assert result.plan is not None
    assert len(result.plan.steps) == 3
    assert result.plan.steps[0].description == "Inspect contracts."
    assert result.plan.steps[-1].metadata["verification"] is True


def test_planning_runtime_updates_step() -> None:
    runtime = PlanningRuntime()
    created = runtime.create_plan(PlanCreateRequest(goal=_goal()))
    assert created.plan is not None
    step = created.plan.steps[0]

    updated = runtime.update_step(
        PlanStepUpdateRequest(
            plan_id=created.plan.plan_id,
            step_id=step.step_id,
            status=PlanStepStatus.RUNNING,
            title="Inspect context",
            description="Inspect current context.",
            risk=PlanRisk.MEDIUM,
            requires_approval=True,
        )
    )

    assert updated.status == PlanningRuntimeStatus.READY
    assert updated.operation == PlanningOperation.UPDATE_STEP
    assert updated.plan is not None
    updated_step = updated.plan.steps[0]
    assert updated_step.status == PlanStepStatus.RUNNING
    assert updated_step.title == "Inspect context"
    assert updated_step.risk == PlanRisk.MEDIUM
    assert updated_step.requires_approval is True


def test_planning_runtime_blocks_missing_plan_or_step() -> None:
    runtime = PlanningRuntime()

    missing_plan = runtime.update_step(
        PlanStepUpdateRequest(
            plan_id="missing",
            step_id="step",
            status=PlanStepStatus.RUNNING,
        )
    )

    assert missing_plan.status == PlanningRuntimeStatus.BLOCKED

    created = runtime.create_plan(PlanCreateRequest(goal=_goal()))
    assert created.plan is not None

    missing_step = runtime.update_step(
        PlanStepUpdateRequest(
            plan_id=created.plan.plan_id,
            step_id="missing",
            status=PlanStepStatus.RUNNING,
        )
    )

    assert missing_step.status == PlanningRuntimeStatus.BLOCKED


def test_planning_runtime_block_and_complete_step() -> None:
    runtime = PlanningRuntime()
    created = runtime.create_plan(PlanCreateRequest(goal=_goal()))
    assert created.plan is not None
    step_id = created.plan.steps[0].step_id

    blocked = runtime.block_step(
        plan_id=created.plan.plan_id,
        step_id=step_id,
        reason="Needs approval",
    )

    assert blocked.operation == PlanningOperation.BLOCK_STEP
    assert blocked.plan is not None
    assert blocked.plan.steps[0].status == PlanStepStatus.BLOCKED
    assert blocked.plan.steps[0].metadata["blocked_reason"] == "Needs approval"

    completed = runtime.complete_step(
        plan_id=created.plan.plan_id,
        step_id=step_id,
    )

    assert completed.operation == PlanningOperation.COMPLETE_STEP
    assert completed.plan is not None
    assert completed.plan.steps[0].status == PlanStepStatus.COMPLETED


def test_planning_runtime_block_step_requires_reason() -> None:
    runtime = PlanningRuntime()
    created = runtime.create_plan(PlanCreateRequest(goal=_goal()))
    assert created.plan is not None

    with pytest.raises(ValueError):
        runtime.block_step(
            plan_id=created.plan.plan_id,
            step_id=created.plan.steps[0].step_id,
            reason=" ",
        )


def test_planning_runtime_cancel_plan() -> None:
    runtime = PlanningRuntime()
    first = runtime.create_plan(PlanCreateRequest(goal=_goal()))
    second = runtime.create_plan(
        PlanCreateRequest(
            goal=_goal(),
            title="Second plan",
        )
    )
    assert first.plan is not None
    assert second.plan is not None

    cancelled = runtime.cancel_plan(second.plan.plan_id)

    assert cancelled.status == PlanningRuntimeStatus.READY
    assert cancelled.operation == PlanningOperation.CANCEL_PLAN
    assert cancelled.plan is not None
    assert cancelled.plan.metadata["cancelled"] is True
    assert runtime.state.active_plan_id == first.plan.plan_id


def test_planning_runtime_recall_by_goal_query_and_active_only() -> None:
    runtime = PlanningRuntime()
    first = runtime.create_plan(
        PlanCreateRequest(
            goal=_goal(),
            title="Developer Phase 9 plan",
        )
    )
    second_goal = _goal()
    second_goal = second_goal.__class__(
        goal_id="goal_2",
        title="Research Engine",
        description="Build research.",
        status=GoalStatus.ACTIVE,
        priority=GoalPriority.NORMAL,
        created_at=second_goal.created_at,
        updated_at=second_goal.updated_at,
    )
    runtime.create_plan(
        PlanCreateRequest(
            goal=second_goal,
            title="Research plan",
        )
    )
    assert first.plan is not None

    recalled = runtime.recall(
        PlanRecallRequest(
            goal_id="goal_1",
            query="Developer",
        )
    )

    assert recalled.operation == PlanningOperation.RECALL
    assert len(recalled.plans) == 1
    assert recalled.plans[0].title == "Developer Phase 9 plan"

    active = runtime.recall(PlanRecallRequest(active_only=True))

    assert len(active.plans) == 1
    assert active.plans[0].plan_id == runtime.state.active_plan_id


def test_planning_runtime_safety_plan_requires_approval() -> None:
    runtime = PlanningRuntime()

    result = runtime.create_plan(
        PlanCreateRequest(
            goal=_goal(priority=GoalPriority.CRITICAL),
            intent_kind=PlanIntentKind.SAFETY,
        )
    )

    assert result.plan is not None
    assert result.plan.requires_approval is True
    assert any(step.risk == PlanRisk.MEDIUM for step in result.plan.steps)


def test_planning_runtime_clear_resets_state() -> None:
    runtime = PlanningRuntime()
    runtime.create_plan(PlanCreateRequest(goal=_goal()))

    result = runtime.clear()

    assert result.operation == PlanningOperation.CLEAR
    assert result.state.plans == ()
    assert result.state.active_plan_id is None


def test_planning_runtime_snapshot_tracks_counts() -> None:
    runtime = PlanningRuntime()
    created = runtime.create_plan(PlanCreateRequest(goal=_goal()))
    assert created.plan is not None
    step_id = created.plan.steps[0].step_id
    runtime.complete_step(plan_id=created.plan.plan_id, step_id=step_id)

    snapshot = runtime.snapshot()

    assert snapshot.status == PlanningRuntimeStatus.READY
    assert snapshot.plan_count == 1
    assert snapshot.completed_step_count == 1
    assert snapshot.operation_count == 2


def test_planning_request_validation() -> None:
    with pytest.raises(ValueError):
        PlanStepUpdateRequest(
            plan_id=" ",
            step_id="step",
            status=PlanStepStatus.RUNNING,
        )

    with pytest.raises(ValueError):
        PlanRecallRequest(limit=0)


def test_planning_enum_values_are_stable() -> None:
    assert PlanningRuntimeStatus.READY.value == "ready"
    assert PlanningOperation.CREATE_PLAN.value == "create_plan"
    assert PlanIntentKind.DEVELOPER.value == "developer"