from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    BlockedState,
    ContextAwareUIReasoningRuntime,
    DualInputStream,
    EnvironmentFusionRuntime,
    FusedContext,
    GoalPriority,
    GraphDelta,
    GraphDeltaKind,
    GraphNodeKind,
    IntentLifecycleState,
    IntentPersistenceReason,
    IntentPersistenceRuntime,
    LastVerifiedState,
    PartialCompletionState,
    PausedWorkflowState,
    PlannerHintKind,
    ResumeStrategy,
    SubgoalState,
    TrustPolicyClassification,
    UIContextChain,
    UIReasoningIntentKind,
    UIReasoningRequest,
    UIReasoningResult,
    VoiceInputFrame,
    WorkspaceCognitiveGraph,
    WorkspaceGraphRuntime,
    graph_node,
)


def test_runtime_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        IntentPersistenceRuntime(name=" ")


def test_create_session() -> None:
    runtime = IntentPersistenceRuntime()
    session = runtime.create_session(workspace_id="workspace")

    assert session.workspace_id == "workspace"
    assert runtime.snapshot().session_count == 1


def test_persistent_intent_rejects_missing_active_subgoal_reference() -> None:
    runtime = IntentPersistenceRuntime()
    session = runtime.create_session(workspace_id="workspace")
    intent = runtime.create_intent(
        session_id=session.session_id,
        goal_description="debug project",
    )

    with pytest.raises(ValidationError):
        type(intent)(
            goal=intent.goal,
            subgoals=intent.subgoals,
            active_subgoal_id="missing",
            trust=intent.trust,
        )


def test_create_intent_stores_goal_and_subgoal() -> None:
    runtime = IntentPersistenceRuntime()
    session = runtime.create_session(workspace_id="workspace")
    subgoal = SubgoalState(
        description="inspect visible error",
        intent_kind=UIReasoningIntentKind.FIX,
        planner_hint=PlannerHintKind.INSPECT_ERROR,
    )

    intent = runtime.create_intent(
        session_id=session.session_id,
        goal_description="debug project",
        priority=GoalPriority.HIGH,
        subgoals=(subgoal,),
    )

    assert intent.goal.description == "debug project"
    assert intent.goal.priority == GoalPriority.HIGH
    assert intent.active_subgoal_id == subgoal.subgoal_id
    assert runtime.active_intent(session_id=session.session_id) == intent


def test_create_from_reasoning_builds_subgoal() -> None:
    reasoning = _reasoning_result()
    runtime = IntentPersistenceRuntime()
    session = runtime.create_session(workspace_id="workspace")

    intent = runtime.create_from_reasoning(
        session_id=session.session_id,
        reasoning=reasoning,
        goal_description="fix visible error",
    )

    assert intent.goal.description == "fix visible error"
    assert intent.subgoals
    assert intent.subgoals[0].planner_hint == PlannerHintKind.INSPECT_ERROR


def test_create_from_fused_context_records_verified_state() -> None:
    runtime = IntentPersistenceRuntime()
    session = runtime.create_session(workspace_id="workspace")
    fused = _fused_context()

    intent = runtime.create_from_fused_context(
        session_id=session.session_id,
        fused_context=fused,
    )

    assert intent.last_verified is not None
    assert intent.last_verified.visible_error_count == 1
    assert intent.metadata["fused_context_id"] == fused.context_id


def test_add_and_activate_subgoal() -> None:
    runtime = IntentPersistenceRuntime()
    session = runtime.create_session(workspace_id="workspace")
    intent = runtime.create_intent(
        session_id=session.session_id,
        goal_description="debug project",
    )
    subgoal = SubgoalState(description="open browser research")

    updated = runtime.add_subgoal(
        session_id=session.session_id,
        intent_id=intent.intent_id,
        subgoal=subgoal,
        activate=True,
    )

    assert updated.active_subgoal_id == subgoal.subgoal_id


def test_activate_missing_subgoal_fails() -> None:
    runtime = IntentPersistenceRuntime()
    session = runtime.create_session(workspace_id="workspace")
    intent = runtime.create_intent(
        session_id=session.session_id,
        goal_description="debug project",
    )

    with pytest.raises(ValueError):
        runtime.activate_subgoal(
            session_id=session.session_id,
            intent_id=intent.intent_id,
            subgoal_id="missing",
        )


def test_record_partial_progress() -> None:
    runtime = IntentPersistenceRuntime()
    session = runtime.create_session(workspace_id="workspace")
    intent = runtime.create_intent(
        session_id=session.session_id,
        goal_description="debug project",
    )

    updated = runtime.record_partial_progress(
        session_id=session.session_id,
        intent_id=intent.intent_id,
        partial=PartialCompletionState(
            summary="found TypeError on line 47",
            completed_steps=("ran tests", "opened failing file"),
            remaining_steps=("patch line 47", "rerun tests"),
            progress_ratio=0.45,
        ),
    )

    assert updated.status == IntentLifecycleState.PARTIAL
    assert updated.partial is not None
    assert updated.partial.progress_ratio == 0.45


def test_block_intent() -> None:
    runtime = IntentPersistenceRuntime()
    session = runtime.create_session(workspace_id="workspace")
    intent = runtime.create_intent(
        session_id=session.session_id,
        goal_description="debug project",
    )

    updated = runtime.block_intent(
        session_id=session.session_id,
        intent_id=intent.intent_id,
        blocked=BlockedState(
            reason="waiting for browser research",
            blocked_by="context_switch",
            requires_user_input=False,
        ),
    )

    assert updated.status == IntentLifecycleState.BLOCKED
    assert updated.blocked is not None
    assert updated.blocked.reason == "waiting for browser research"


def test_require_approval() -> None:
    runtime = IntentPersistenceRuntime()
    session = runtime.create_session(workspace_id="workspace")
    intent = runtime.create_intent(
        session_id=session.session_id,
        goal_description="modify file",
    )

    updated = runtime.require_approval(
        session_id=session.session_id,
        intent_id=intent.intent_id,
        reason="file write requires approval",
    )

    assert updated.status == IntentLifecycleState.WAITING_APPROVAL
    assert updated.blocked is not None
    assert updated.blocked.requires_user_input is True


def test_pause_creates_resume_token() -> None:
    runtime = IntentPersistenceRuntime()
    session = runtime.create_session(workspace_id="workspace")
    subgoal = SubgoalState(description="fix TypeError line 47")
    intent = runtime.create_intent(
        session_id=session.session_id,
        goal_description="debug project",
        subgoals=(subgoal,),
    )

    paused = runtime.pause_intent(
        session_id=session.session_id,
        intent_id=intent.intent_id,
        pause=PausedWorkflowState(
            paused_subgoal_id=subgoal.subgoal_id,
            reason="user switched to browser research",
            resume_plan=("return to TypeError", "patch line 47", "rerun tests"),
        ),
    )

    assert paused.status == IntentLifecycleState.PAUSED
    assert paused.resume_token is not None
    assert "debug project" in paused.resume_token.resume_prompt


def test_resume_intent_with_token() -> None:
    runtime = IntentPersistenceRuntime()
    session = runtime.create_session(workspace_id="workspace")
    intent = runtime.create_intent(
        session_id=session.session_id,
        goal_description="debug project",
    )
    paused = runtime.pause_intent(
        session_id=session.session_id,
        intent_id=intent.intent_id,
        pause=PausedWorkflowState(reason="user interrupted"),
    )

    assert paused.resume_token is not None

    resumed = runtime.resume_intent(
        session_id=session.session_id,
        intent_id=intent.intent_id,
        token_id=paused.resume_token.token_id,
    )

    assert resumed.status == IntentLifecycleState.ACTIVE
    assert resumed.paused is None


def test_resume_with_invalid_token_fails() -> None:
    runtime = IntentPersistenceRuntime()
    session = runtime.create_session(workspace_id="workspace")
    intent = runtime.create_intent(
        session_id=session.session_id,
        goal_description="debug project",
    )

    with pytest.raises(ValueError):
        runtime.resume_intent(
            session_id=session.session_id,
            intent_id=intent.intent_id,
            token_id="bad-token",
        )


def test_record_verified_state() -> None:
    runtime = IntentPersistenceRuntime()
    session = runtime.create_session(workspace_id="workspace")
    intent = runtime.create_intent(
        session_id=session.session_id,
        goal_description="debug project",
    )

    updated = runtime.record_verified_state(
        session_id=session.session_id,
        intent_id=intent.intent_id,
        verified=LastVerifiedState(
            summary="VS Code error still visible",
            graph_node_count=3,
            visible_error_count=1,
            policy=TrustPolicyClassification.VERIFY_FIRST,
        ),
    )

    assert updated.last_verified is not None
    assert updated.last_verified.visible_error_count == 1


def test_complete_cancel_and_fail_intent() -> None:
    runtime = IntentPersistenceRuntime()
    session = runtime.create_session(workspace_id="workspace")
    complete = runtime.create_intent(
        session_id=session.session_id,
        goal_description="complete me",
    )
    cancel = runtime.create_intent(
        session_id=session.session_id,
        goal_description="cancel me",
    )
    fail = runtime.create_intent(
        session_id=session.session_id,
        goal_description="fail me",
    )

    completed = runtime.complete_intent(
        session_id=session.session_id,
        intent_id=complete.intent_id,
        summary="done",
    )
    cancelled = runtime.cancel_intent(
        session_id=session.session_id,
        intent_id=cancel.intent_id,
        reason="user cancelled",
    )
    failed = runtime.fail_intent(
        session_id=session.session_id,
        intent_id=fail.intent_id,
        reason="tool unavailable",
    )

    assert completed.status == IntentLifecycleState.COMPLETED
    assert cancelled.status == IntentLifecycleState.CANCELLED
    assert failed.status == IntentLifecycleState.FAILED


def test_create_resume_token_manually() -> None:
    runtime = IntentPersistenceRuntime()
    session = runtime.create_session(workspace_id="workspace")
    intent = runtime.create_intent(
        session_id=session.session_id,
        goal_description="debug project",
    )

    token = runtime.create_resume_token(
        session_id=session.session_id,
        intent_id=intent.intent_id,
        strategy=ResumeStrategy.VERIFY_THEN_CONTINUE,
    )

    assert token.strategy == ResumeStrategy.VERIFY_THEN_CONTINUE
    assert token.intent_id == intent.intent_id


def test_missing_session_and_intent_fail() -> None:
    runtime = IntentPersistenceRuntime()

    with pytest.raises(ValueError):
        runtime.create_intent(
            session_id="missing",
            goal_description="debug project",
        )

    session = runtime.create_session(workspace_id="workspace")

    with pytest.raises(ValueError):
        runtime.block_intent(
            session_id=session.session_id,
            intent_id="missing",
            blocked=BlockedState(reason="missing"),
        )


def test_snapshot_tracks_counts() -> None:
    runtime = IntentPersistenceRuntime()
    session = runtime.create_session(workspace_id="workspace")
    active = runtime.create_intent(
        session_id=session.session_id,
        goal_description="active",
    )
    paused = runtime.create_intent(
        session_id=session.session_id,
        goal_description="paused",
    )
    blocked = runtime.create_intent(
        session_id=session.session_id,
        goal_description="blocked",
    )
    partial = runtime.create_intent(
        session_id=session.session_id,
        goal_description="partial",
    )
    completed = runtime.create_intent(
        session_id=session.session_id,
        goal_description="completed",
    )

    runtime.pause_intent(
        session_id=session.session_id,
        intent_id=paused.intent_id,
        pause=PausedWorkflowState(reason="pause"),
    )
    runtime.block_intent(
        session_id=session.session_id,
        intent_id=blocked.intent_id,
        blocked=BlockedState(reason="blocked"),
    )
    runtime.record_partial_progress(
        session_id=session.session_id,
        intent_id=partial.intent_id,
        partial=PartialCompletionState(summary="partial"),
    )
    runtime.complete_intent(
        session_id=session.session_id,
        intent_id=completed.intent_id,
        summary="done",
    )

    snapshot = runtime.snapshot()

    assert active.status == IntentLifecycleState.ACTIVE
    assert snapshot.intent_count == 5
    assert snapshot.active_count == 1
    assert snapshot.paused_count == 1
    assert snapshot.blocked_count == 1
    assert snapshot.partial_count == 1
    assert snapshot.completed_count == 1
    assert snapshot.resume_token_count == 1


def test_reset_clears_runtime() -> None:
    runtime = IntentPersistenceRuntime()
    runtime.create_session(workspace_id="workspace")

    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.runtime_event_count == 1
    assert snapshot.last_reason == IntentPersistenceReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert IntentLifecycleState.ACTIVE.value == "active"
    assert ResumeStrategy.VERIFY_THEN_CONTINUE.value == "verify_then_continue"
    assert GoalPriority.CRITICAL.value == "critical"


def _reasoning_result() -> UIReasoningResult:
    runtime = ContextAwareUIReasoningRuntime()
    session = runtime.create_session(workspace_id="workspace")

    return runtime.reason(
        UIReasoningRequest(
            session_id=session.session_id,
            utterance="fix that",
            context_chain=UIContextChain(
                app_kind=None,
                workspace_graph=_graph_with_error(),
            ),
        )
    )


def _fused_context() -> FusedContext:
    fusion = EnvironmentFusionRuntime()
    session = fusion.create_session(workspace_id="workspace")

    return fusion.fuse(
        session_id=session.session_id,
        stream=DualInputStream(
            voice=VoiceInputFrame(text="fix that"),
            workspace_graph=_graph_with_error(),
        ),
    )


def _graph_with_error() -> WorkspaceCognitiveGraph:
    runtime = WorkspaceGraphRuntime()
    session = runtime.create_session(workspace_id="workspace")
    error = graph_node(kind=GraphNodeKind.ERROR, label="AssertionError")

    runtime.apply_delta(
        session_id=session.session_id,
        delta=GraphDelta(
            kind=GraphDeltaKind.GRAPH_REBUILT,
            added_nodes=(error,),
            reason="error graph",
        ),
    )
    graph = runtime.graph_for(session.session_id)

    assert graph is not None

    return graph