from __future__ import annotations

from datetime import UTC, datetime

import pytest

from jarvis.cognitive import (
    AttentionDecision,
    AttentionItem,
    AttentionItemKind,
    AttentionPriority,
    AttentionState,
    BehaviorPolicy,
    Goal,
    GoalPriority,
    GoalState,
    GoalStatus,
    Phase9CheckKind,
    Phase9DesignGate,
    Phase9GateStatus,
    Plan,
    PlanningState,
    PlanRisk,
    PlanStep,
    PlanStepStatus,
    WorkingMemoryItem,
    WorkingMemoryKind,
    WorkingMemoryState,
    default_cognitive_session,
)


def _now() -> datetime:
    return datetime.now(UTC)


def test_attention_item_rejects_empty_title() -> None:
    with pytest.raises(ValueError):
        AttentionItem(
            item_id="att",
            kind=AttentionItemKind.SAFETY,
            title=" ",
            summary="battery critical",
            priority=AttentionPriority.CRITICAL,
            source="test",
            decision=AttentionDecision.INTERRUPT_NOW,
            created_at=_now(),
        )


def test_attention_state_tracks_critical_interrupts() -> None:
    item = AttentionItem(
        item_id="att_1",
        kind=AttentionItemKind.SYSTEM_HEALTH,
        title="Battery critical",
        summary="Battery is below safe threshold.",
        priority=AttentionPriority.CRITICAL,
        source="system",
        decision=AttentionDecision.INTERRUPT_NOW,
        created_at=_now(),
    )
    state = AttentionState(items=(item,), focused_item_id=item.item_id)

    assert state.has_focus is True
    assert state.critical_items == (item,)
    assert state.interrupt_items == (item,)


def test_working_memory_gets_item_by_key() -> None:
    item = WorkingMemoryItem(
        item_id="wm_1",
        kind=WorkingMemoryKind.OBJECTIVE,
        key="current_goal",
        value="Build Phase 9",
        importance=AttentionPriority.HIGH,
        source="test",
        created_at=_now(),
    )
    state = WorkingMemoryState(items=(item,))

    assert state.get("CURRENT_GOAL") == item
    assert state.high_importance_items == (item,)


def test_goal_state_tracks_active_and_blocked_goals() -> None:
    active = Goal(
        goal_id="goal_1",
        title="Build Phase 9",
        description="Build cognitive state layer.",
        status=GoalStatus.ACTIVE,
        priority=GoalPriority.HIGH,
        created_at=_now(),
        updated_at=_now(),
    )
    blocked = Goal(
        goal_id="goal_2",
        title="Live voice gate",
        description="Needs real adapter validation.",
        status=GoalStatus.BLOCKED,
        priority=GoalPriority.NORMAL,
        created_at=_now(),
        updated_at=_now(),
    )

    state = GoalState(goals=(active, blocked), active_goal_id=active.goal_id)

    assert state.has_active_goal is True
    assert state.active_goals == (active,)
    assert state.blocked_goals == (blocked,)


def test_plan_tracks_approval_requirement() -> None:
    step = PlanStep(
        step_id="step_1",
        title="Run validation",
        description="Run safe validation command.",
        status=PlanStepStatus.READY,
        risk=PlanRisk.MEDIUM,
        requires_approval=True,
        created_at=_now(),
    )
    plan = Plan(
        plan_id="plan_1",
        goal_id="goal_1",
        title="Validation plan",
        steps=(step,),
        created_at=_now(),
    )
    state = PlanningState(plans=(plan,), active_plan_id=plan.plan_id)

    assert plan.requires_approval is True
    assert state.active_plan == plan


def test_behavior_policy_rejects_invalid_reply_limit() -> None:
    with pytest.raises(ValueError):
        BehaviorPolicy(
            max_reply_sentences=0,
            interrupt_only_when_important=True,
            ask_when_instruction_incomplete=True,
            allow_dry_humor=True,
            truth_over_comfort=True,
            created_at=_now(),
        )


def test_default_cognitive_session_passes_design_gate() -> None:
    session = default_cognitive_session(user_label="Balu")
    report = Phase9DesignGate().validate(session)

    assert report.status == Phase9GateStatus.PASSED
    assert report.passed is True
    assert report.failed_count == 0

    kinds = {check.kind for check in report.checks}

    assert Phase9CheckKind.ATTENTION_STATE in kinds
    assert Phase9CheckKind.WORKING_MEMORY_STATE in kinds
    assert Phase9CheckKind.GOAL_STATE in kinds
    assert Phase9CheckKind.PLANNING_STATE in kinds
    assert Phase9CheckKind.PERSONALITY_PROFILE in kinds
    assert Phase9CheckKind.BEHAVIOR_POLICY in kinds
    assert Phase9CheckKind.COGNITIVE_SESSION in kinds


def test_design_gate_fails_when_behavior_is_not_concise() -> None:
    session = default_cognitive_session(user_label="Balu")
    noisy_policy = BehaviorPolicy(
        max_reply_sentences=10,
        interrupt_only_when_important=True,
        ask_when_instruction_incomplete=True,
        allow_dry_humor=True,
        truth_over_comfort=True,
        created_at=_now(),
    )
    broken_session = session.__class__(
        session_id=session.session_id,
        user_label=session.user_label,
        attention=session.attention,
        working_memory=session.working_memory,
        goals=session.goals,
        planning=session.planning,
        personality=session.personality,
        behavior_policy=noisy_policy,
        created_at=session.created_at,
        updated_at=session.updated_at,
        metadata=session.metadata,
    )

    report = Phase9DesignGate().validate(broken_session)

    assert report.status == Phase9GateStatus.FAILED
    assert report.failed_count == 1


def test_phase9_enum_values_are_stable() -> None:
    assert AttentionPriority.CRITICAL.value == "critical"
    assert WorkingMemoryKind.OBJECTIVE.value == "objective"
    assert GoalStatus.ACTIVE.value == "active"
    assert Phase9GateStatus.PASSED.value == "passed"