from __future__ import annotations

import pytest

from jarvis.cognitive import (
    GoalCreateRequest,
    GoalOperation,
    GoalPriority,
    GoalRecallRequest,
    GoalRuntime,
    GoalRuntimeStatus,
    GoalStatus,
    GoalUpdateRequest,
)


def test_goal_create_request_rejects_empty_title() -> None:
    with pytest.raises(ValueError):
        GoalCreateRequest(
            title=" ",
            description="Build Phase 9",
        )


def test_goal_runtime_creates_active_goal() -> None:
    runtime = GoalRuntime()

    result = runtime.create(
        GoalCreateRequest(
            title="Build Phase 9",
            description="Build cognitive presence.",
            priority=GoalPriority.HIGH,
            tags=("Phase9", "JARVIS"),
        )
    )

    assert result.status == GoalRuntimeStatus.READY
    assert result.operation == GoalOperation.CREATE
    assert result.goal is not None
    assert result.goal.status == GoalStatus.ACTIVE
    assert result.goal.priority == GoalPriority.HIGH
    assert result.goal.tags == ("phase9", "jarvis")
    assert result.state.active_goal_id == result.goal.goal_id


def test_goal_runtime_updates_goal_title_and_priority() -> None:
    runtime = GoalRuntime()
    created = runtime.create(
        GoalCreateRequest(
            title="Old title",
            description="Old description.",
        )
    )
    assert created.goal is not None

    updated = runtime.update(
        GoalUpdateRequest(
            goal_id=created.goal.goal_id,
            title="Build Goal Runtime",
            priority=GoalPriority.CRITICAL,
        )
    )

    assert updated.status == GoalRuntimeStatus.READY
    assert updated.operation == GoalOperation.UPDATE
    assert updated.goal is not None
    assert updated.goal.title == "Build Goal Runtime"
    assert updated.goal.priority == GoalPriority.CRITICAL


def test_goal_runtime_blocks_missing_goal_update() -> None:
    runtime = GoalRuntime()

    result = runtime.update(
        GoalUpdateRequest(
            goal_id="missing",
            title="No goal",
        )
    )

    assert result.status == GoalRuntimeStatus.BLOCKED
    assert result.succeeded is False
    assert result.goal is None


def test_goal_runtime_pause_resume_block_complete_cancel() -> None:
    runtime = GoalRuntime()
    created = runtime.create(
        GoalCreateRequest(
            title="Build runtime",
            description="Build goal lifecycle.",
        )
    )
    assert created.goal is not None
    goal_id = created.goal.goal_id

    paused = runtime.pause(goal_id)
    assert paused.operation == GoalOperation.PAUSE
    assert paused.goal is not None
    assert paused.goal.status == GoalStatus.PAUSED

    resumed = runtime.resume(goal_id)
    assert resumed.operation == GoalOperation.RESUME
    assert resumed.goal is not None
    assert resumed.goal.status == GoalStatus.ACTIVE

    blocked = runtime.block(goal_id, reason="Needs dependency")
    assert blocked.operation == GoalOperation.BLOCK
    assert blocked.goal is not None
    assert blocked.goal.status == GoalStatus.BLOCKED
    assert blocked.goal.metadata["blocked_reason"] == "Needs dependency"

    completed = runtime.complete(goal_id)
    assert completed.operation == GoalOperation.COMPLETE
    assert completed.goal is not None
    assert completed.goal.status == GoalStatus.COMPLETED

    cancelled = runtime.cancel(goal_id)
    assert cancelled.operation == GoalOperation.CANCEL
    assert cancelled.goal is not None
    assert cancelled.goal.status == GoalStatus.CANCELLED


def test_goal_runtime_block_requires_reason() -> None:
    runtime = GoalRuntime()
    created = runtime.create(
        GoalCreateRequest(
            title="Goal",
            description="Goal description.",
        )
    )
    assert created.goal is not None

    with pytest.raises(ValueError):
        runtime.block(created.goal.goal_id, reason=" ")


def test_goal_runtime_recall_by_query_status_priority_and_tag() -> None:
    runtime = GoalRuntime()
    runtime.create(
        GoalCreateRequest(
            title="Build Phase 9",
            description="Build cognitive runtime.",
            priority=GoalPriority.HIGH,
            tags=("phase9",),
        )
    )
    second = runtime.create(
        GoalCreateRequest(
            title="Build research engine",
            description="Build future research runtime.",
            priority=GoalPriority.NORMAL,
            tags=("research",),
        )
    )
    assert second.goal is not None
    runtime.pause(second.goal.goal_id)

    result = runtime.recall(
        GoalRecallRequest(
            query="Phase 9",
            statuses=(GoalStatus.ACTIVE,),
            priorities=(GoalPriority.HIGH,),
            tags=("phase9",),
        )
    )

    assert result.operation == GoalOperation.RECALL
    assert len(result.goals) == 1
    assert result.goals[0].title == "Build Phase 9"


def test_goal_runtime_parent_child_filter() -> None:
    runtime = GoalRuntime()
    parent = runtime.create(
        GoalCreateRequest(
            title="Phase 9",
            description="Build Phase 9.",
        )
    )
    assert parent.goal is not None
    child = runtime.create(
        GoalCreateRequest(
            title="Attention Runtime",
            description="Build attention.",
            parent_goal_id=parent.goal.goal_id,
        )
    )
    assert child.goal is not None

    result = runtime.recall(
        GoalRecallRequest(include_children=False)
    )

    assert all(goal.parent_goal_id is None for goal in result.goals)


def test_goal_runtime_clear_resets_state() -> None:
    runtime = GoalRuntime()
    runtime.create(
        GoalCreateRequest(
            title="Build Phase 9",
            description="Build cognitive runtime.",
        )
    )

    result = runtime.clear()

    assert result.operation == GoalOperation.CLEAR
    assert result.state.goals == ()
    assert result.state.active_goal_id is None


def test_goal_runtime_snapshot_tracks_counts() -> None:
    runtime = GoalRuntime()
    first = runtime.create(
        GoalCreateRequest(
            title="Active",
            description="Active goal.",
        )
    )
    assert first.goal is not None
    second = runtime.create(
        GoalCreateRequest(
            title="Blocked",
            description="Blocked goal.",
        )
    )
    assert second.goal is not None
    runtime.block(second.goal.goal_id, reason="Waiting")
    runtime.complete(first.goal.goal_id)

    snapshot = runtime.snapshot()

    assert snapshot.status == GoalRuntimeStatus.READY
    assert snapshot.goal_count == 2
    assert snapshot.blocked_count == 1
    assert snapshot.completed_count == 1
    assert snapshot.operation_count == 4


def test_goal_runtime_active_goal_moves_when_current_goal_paused() -> None:
    runtime = GoalRuntime()
    first = runtime.create(
        GoalCreateRequest(
            title="First",
            description="First goal.",
            priority=GoalPriority.NORMAL,
        )
    )
    second = runtime.create(
        GoalCreateRequest(
            title="Second",
            description="Second goal.",
            priority=GoalPriority.HIGH,
        )
    )
    assert first.goal is not None
    assert second.goal is not None
    assert runtime.state.active_goal_id == second.goal.goal_id

    runtime.pause(second.goal.goal_id)

    assert runtime.state.active_goal_id == first.goal.goal_id


def test_goal_request_validation() -> None:
    with pytest.raises(ValueError):
        GoalUpdateRequest(goal_id=" ")

    with pytest.raises(ValueError):
        GoalRecallRequest(limit=0)


def test_goal_enum_values_are_stable() -> None:
    assert GoalRuntimeStatus.READY.value == "ready"
    assert GoalOperation.CREATE.value == "create"
    assert GoalStatus.ACTIVE.value == "active"