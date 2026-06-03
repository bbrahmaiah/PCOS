from __future__ import annotations

import pytest

from jarvis.cognitive import (
    AttentionDecision,
    AttentionItemKind,
    AttentionSignalSource,
    AttentionSignalUrgency,
    BehaviorIntent,
    BehaviorRisk,
    CognitiveSessionGoalRequest,
    CognitiveSessionOperation,
    CognitiveSessionResponseRequest,
    CognitiveSessionRuntime,
    CognitiveSessionRuntimeStatus,
    CognitiveSessionStartRequest,
    CognitiveSessionUpdateRequest,
    GoalPriority,
    PlanIntentKind,
    WorkingMemoryKind,
    make_attention_signal,
    make_working_memory_entry,
)


def test_cognitive_session_start_rejects_empty_user_label() -> None:
    with pytest.raises(ValueError):
        CognitiveSessionStartRequest(user_label=" ")


def test_cognitive_session_runtime_starts_session() -> None:
    runtime = CognitiveSessionRuntime()

    result = runtime.start(CognitiveSessionStartRequest(user_label="Balu"))

    assert result.status == CognitiveSessionRuntimeStatus.READY
    assert result.operation == CognitiveSessionOperation.START
    assert result.session.user_label == "Balu"
    assert result.session.personality.name == "JARVIS"


def test_cognitive_session_update_combines_attention_and_working_memory() -> None:
    runtime = CognitiveSessionRuntime()
    runtime.start(CognitiveSessionStartRequest(user_label="Balu"))

    signal = make_attention_signal(
        source=AttentionSignalSource.SAFETY,
        kind=AttentionItemKind.SAFETY,
        title="Battery critical",
        summary="Battery is below safe threshold.",
        urgency=AttentionSignalUrgency.EMERGENCY,
    )
    memory = make_working_memory_entry(
        kind=WorkingMemoryKind.RISK,
        key="battery",
        value="Battery is critically low",
    )

    result = runtime.update(
        CognitiveSessionUpdateRequest(
            attention_signals=(signal,),
            working_memory_entries=(memory,),
            assistant_is_speaking=True,
        )
    )

    assert result.status == CognitiveSessionRuntimeStatus.READY
    assert result.attention_result is not None
    assert result.attention_result.decision == AttentionDecision.INTERRUPT_NOW
    assert result.working_memory_result is not None
    assert result.session.attention.interrupt_items
    assert result.session.working_memory.get("battery") is not None


def test_cognitive_session_creates_goal_and_plan() -> None:
    runtime = CognitiveSessionRuntime()
    runtime.start(CognitiveSessionStartRequest(user_label="Balu"))

    result = runtime.create_goal(
        CognitiveSessionGoalRequest(
            title="Build Phase 9",
            description="Build cognitive session runtime.",
            priority=GoalPriority.HIGH,
            tags=("phase9",),
            create_plan=True,
            intent_kind=PlanIntentKind.DEVELOPER,
        )
    )

    assert result.status == CognitiveSessionRuntimeStatus.READY
    assert result.goal_result is not None
    assert result.goal_result.goal is not None
    assert result.planning_result is not None
    assert result.planning_result.plan is not None
    assert result.session.goals.has_active_goal is True
    assert result.session.planning.active_plan is not None


def test_cognitive_session_blocks_invalid_goal_priority() -> None:
    runtime = CognitiveSessionRuntime()

    result = runtime.create_goal(
        CognitiveSessionGoalRequest(
            title="Invalid",
            description="Invalid priority.",
            priority="high",
        )
    )

    assert result.status == CognitiveSessionRuntimeStatus.BLOCKED
    assert result.succeeded is False


def test_cognitive_session_response_uses_personality_runtime() -> None:
    runtime = CognitiveSessionRuntime()
    runtime.start(CognitiveSessionStartRequest(user_label="Balu"))

    result = runtime.respond(
        CognitiveSessionResponseRequest(
            intent=BehaviorIntent.CONFIRMATION,
            message="Running validation.",
        )
    )

    assert result.status == CognitiveSessionRuntimeStatus.READY
    assert result.behavior_result is not None
    assert result.behavior_result.text == "Certainly, sir. Running validation."


def test_cognitive_session_warning_response_is_protective() -> None:
    runtime = CognitiveSessionRuntime()

    result = runtime.respond(
        CognitiveSessionResponseRequest(
            intent=BehaviorIntent.WARNING,
            message="That action may delete project files.",
            risk=BehaviorRisk.HIGH,
        )
    )

    assert result.behavior_result is not None
    assert result.behavior_result.directive.should_warn is True
    assert "I would advise caution." in result.behavior_result.text


def test_cognitive_session_clear_resets_organs() -> None:
    runtime = CognitiveSessionRuntime()
    runtime.start(CognitiveSessionStartRequest(user_label="Balu"))
    runtime.create_goal(
        CognitiveSessionGoalRequest(
            title="Build",
            description="Build session.",
            priority=GoalPriority.NORMAL,
        )
    )

    result = runtime.clear()

    assert result.operation == CognitiveSessionOperation.CLEAR
    assert result.session.attention.items == ()
    assert result.session.working_memory.items == ()
    assert result.session.goals.goals == ()
    assert result.session.planning.plans == ()


def test_cognitive_session_snapshot_tracks_counts() -> None:
    runtime = CognitiveSessionRuntime()
    runtime.start(CognitiveSessionStartRequest(user_label="Balu"))
    runtime.respond(
        CognitiveSessionResponseRequest(
            intent=BehaviorIntent.CONFIRMATION,
            message="Ready.",
        )
    )

    snapshot = runtime.snapshot()

    assert snapshot.status == CognitiveSessionRuntimeStatus.READY
    assert snapshot.user_label == "Balu"
    assert snapshot.update_count == 2
    assert snapshot.behavior_decisions == 1


def test_cognitive_session_request_validation() -> None:
    with pytest.raises(ValueError):
        CognitiveSessionGoalRequest(
            title=" ",
            description="description",
            priority=GoalPriority.NORMAL,
        )


def test_cognitive_session_enum_values_are_stable() -> None:
    assert CognitiveSessionRuntimeStatus.READY.value == "ready"
    assert CognitiveSessionOperation.START.value == "start"