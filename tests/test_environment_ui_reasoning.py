from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    ContentClass,
    ContentClassification,
    ContextAwareUIReasoningRuntime,
    DetectedAppKind,
    EnvironmentSource,
    GraphDelta,
    GraphDeltaKind,
    GraphEdgeKind,
    GraphNodeKind,
    GroundingIntentKind,
    GroundingQuery,
    InterfaceKind,
    PlannerHintKind,
    PrivacyClassification,
    SemanticScene,
    SemanticSceneKind,
    SensitiveUIDetection,
    SensitivityLevel,
    TrustCalibration,
    TrustPolicyClassification,
    UIContext,
    UIContextChain,
    UIReasoningDecision,
    UIReasoningIntentKind,
    UIReasoningReason,
    UIReasoningRequest,
    UIReasoningResult,
    UIReasoningStatus,
    UISemanticReason,
    UISemanticStatus,
    VisualGroundingResult,
    VisualGroundingRuntime,
    WorkspaceCognitiveGraph,
    WorkspaceGraphRuntime,
    graph_edge,
    graph_node,
)


def test_runtime_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        ContextAwareUIReasoningRuntime(name=" ")


def test_create_session() -> None:
    runtime = ContextAwareUIReasoningRuntime()
    session = runtime.create_session(workspace_id="workspace")

    assert session.workspace_id == "workspace"
    assert runtime.snapshot().session_count == 1


def test_resolved_result_requires_target() -> None:
    runtime = ContextAwareUIReasoningRuntime()
    session = runtime.create_session(workspace_id="workspace")
    result = runtime.reason(
        UIReasoningRequest(
            session_id=session.session_id,
            utterance="unknown",
            context_chain=UIContextChain(),
        )
    )

    with pytest.raises(ValidationError):
        UIReasoningResult(
            status=UIReasoningStatus.RESOLVED,
            reason=UIReasoningReason.INTENT_RESOLVED,
            decision=UIReasoningDecision.RESOLVED,
            request_id=result.request_id,
            intent=result.intent,
            context_kind=result.context_kind,
            resolved_target=None,
            safe_for_action_planning=False,
            message="invalid",
        )


def test_fix_that_in_vscode_resolves_visible_error() -> None:
    runtime = ContextAwareUIReasoningRuntime()
    session = runtime.create_session(workspace_id="workspace")
    graph = _graph_with_error()

    result = runtime.reason(
        UIReasoningRequest(
            session_id=session.session_id,
            utterance="fix that",
            context_chain=UIContextChain(
                app_kind=DetectedAppKind.IDE,
                workspace_graph=graph,
            ),
        )
    )

    assert result.status == UIReasoningStatus.RESOLVED
    assert result.intent.kind == UIReasoningIntentKind.FIX
    assert result.resolved_target is not None
    assert result.resolved_target.graph_node is not None
    assert result.resolved_target.graph_node.kind == GraphNodeKind.ERROR
    assert result.decision == UIReasoningDecision.VERIFY_FIRST
    assert result.planner_hints[0].kind == PlannerHintKind.INSPECT_ERROR


def test_fix_that_from_error_scene() -> None:
    runtime = ContextAwareUIReasoningRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.reason(
        UIReasoningRequest(
            session_id=session.session_id,
            utterance="fix that",
            context_chain=UIContextChain(
                app_kind=DetectedAppKind.IDE,
                ui_context=_ui_context(SemanticSceneKind.ERROR_DIALOG),
            ),
        )
    )

    assert result.status == UIReasoningStatus.RESOLVED
    assert result.resolved_target is not None
    assert result.resolved_target.scene_kind == SemanticSceneKind.ERROR_DIALOG


def test_open_it_in_file_context_resolves_selected_file() -> None:
    runtime = ContextAwareUIReasoningRuntime()
    session = runtime.create_session(workspace_id="workspace")
    graph, file_node_id = _graph_with_file()

    result = runtime.reason(
        UIReasoningRequest(
            session_id=session.session_id,
            utterance="open it",
            context_chain=UIContextChain(
                app_kind=DetectedAppKind.SYSTEM_APP,
                workspace_graph=graph,
                selected_node_id=file_node_id,
            ),
        )
    )

    assert result.intent.kind == UIReasoningIntentKind.OPEN
    assert result.resolved_target is not None
    assert result.resolved_target.graph_node is not None
    assert result.resolved_target.graph_node.kind == GraphNodeKind.FILE
    assert result.planner_hints[0].kind == PlannerHintKind.OPEN_FILE


def test_run_it_in_terminal_resolves_current_command() -> None:
    runtime = ContextAwareUIReasoningRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.reason(
        UIReasoningRequest(
            session_id=session.session_id,
            utterance="run it",
            context_chain=UIContextChain(
                app_kind=DetectedAppKind.TERMINAL,
                current_command="pytest tests/test_environment_ui_reasoning.py",
            ),
        )
    )

    assert result.intent.kind == UIReasoningIntentKind.RUN
    assert result.resolved_target is not None
    assert "pytest" in result.resolved_target.label
    assert result.planner_hints[0].kind == PlannerHintKind.RUN_COMMAND
    assert result.decision == UIReasoningDecision.VERIFY_FIRST


def test_run_it_in_project_context_resolves_project() -> None:
    runtime = ContextAwareUIReasoningRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.reason(
        UIReasoningRequest(
            session_id=session.session_id,
            utterance="run it",
            context_chain=UIContextChain(
                app_kind=DetectedAppKind.IDE,
                current_project="JARVIS_OS",
            ),
        )
    )

    assert result.resolved_target is not None
    assert result.resolved_target.label == "JARVIS_OS"


def test_copy_this_in_browser_resolves_selected_content() -> None:
    runtime = ContextAwareUIReasoningRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.reason(
        UIReasoningRequest(
            session_id=session.session_id,
            utterance="copy this",
            context_chain=UIContextChain(
                app_kind=DetectedAppKind.BROWSER,
                browser_selection="selected article paragraph",
            ),
        )
    )

    assert result.intent.kind == UIReasoningIntentKind.COPY
    assert result.resolved_target is not None
    assert result.resolved_target.text_selection == "selected article paragraph"
    assert result.safe_for_action_planning is True
    assert result.planner_hints[0].kind == PlannerHintKind.COPY_SELECTION


def test_general_focus_uses_visual_grounding_result() -> None:
    runtime = ContextAwareUIReasoningRuntime()
    session = runtime.create_session(workspace_id="workspace")
    grounding = _ground_terminal()

    result = runtime.reason(
        UIReasoningRequest(
            session_id=session.session_id,
            utterance="focus that",
            context_chain=UIContextChain(
                app_kind=DetectedAppKind.TERMINAL,
                grounding_result=grounding,
            ),
        )
    )

    assert result.resolved_target is not None
    assert result.resolved_target.grounding_result is not None
    assert result.planner_hints[0].kind == PlannerHintKind.FOCUS_TARGET


def test_unknown_intent_not_enough_context() -> None:
    runtime = ContextAwareUIReasoningRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.reason(
        UIReasoningRequest(
            session_id=session.session_id,
            utterance="please do the thing",
            context_chain=UIContextChain(),
        )
    )

    assert result.status == UIReasoningStatus.UNRESOLVED
    assert result.reason == UIReasoningReason.NOT_ENOUGH_CONTEXT
    assert result.decision == UIReasoningDecision.NOT_ENOUGH_CONTEXT


def test_missing_target_asks_user() -> None:
    runtime = ContextAwareUIReasoningRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.reason(
        UIReasoningRequest(
            session_id=session.session_id,
            utterance="open it",
            context_chain=UIContextChain(),
        )
    )

    assert result.status == UIReasoningStatus.AMBIGUOUS
    assert result.decision == UIReasoningDecision.ASK_USER
    assert result.planner_hints[0].kind == PlannerHintKind.ASK_CLARIFICATION


def test_missing_session_fails() -> None:
    runtime = ContextAwareUIReasoningRuntime()

    result = runtime.reason(
        UIReasoningRequest(
            session_id="missing",
            utterance="fix that",
            context_chain=UIContextChain(),
        )
    )

    assert result.status == UIReasoningStatus.FAILED
    assert result.reason == UIReasoningReason.SESSION_NOT_FOUND
    assert result.safe_for_action_planning is False


def test_snapshot_tracks_counts() -> None:
    runtime = ContextAwareUIReasoningRuntime()
    session = runtime.create_session(workspace_id="workspace")

    runtime.reason(
        UIReasoningRequest(
            session_id=session.session_id,
            utterance="copy this",
            context_chain=UIContextChain(
                app_kind=DetectedAppKind.BROWSER,
                browser_selection="selected paragraph",
            ),
        )
    )
    runtime.reason(
        UIReasoningRequest(
            session_id=session.session_id,
            utterance="open it",
            context_chain=UIContextChain(),
        )
    )
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.result_count == 2
    assert snapshot.resolved_count == 1
    assert snapshot.ambiguous_count == 1
    assert snapshot.safe_planning_count == 1


def test_reset_clears_runtime() -> None:
    runtime = ContextAwareUIReasoningRuntime()
    runtime.create_session(workspace_id="workspace")

    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.runtime_event_count == 1
    assert snapshot.last_reason == UIReasoningReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert UIReasoningIntentKind.FIX.value == "fix"
    assert UIReasoningDecision.VERIFY_FIRST.value == "verify_first"
    assert PlannerHintKind.RUN_COMMAND.value == "run_command"


def _ui_context(scene_kind: SemanticSceneKind) -> UIContext:
    scene = SemanticScene(
        kind=scene_kind,
        interface_kind=InterfaceKind.DIALOG,
        confidence=0.90,
        summary=scene_kind.value,
        trust=TrustCalibration(
            confidence=0.90,
            stability=0.90,
            ambiguity=0.0,
            source=EnvironmentSource.OS_OBSERVER,
            reason="test scene",
        ),
    )

    return UIContext(
        status=UISemanticStatus.UNDERSTOOD,
        reason=UISemanticReason.SCENE_UNDERSTOOD,
        request_id="semantic-request",
        scene=scene,
        content=ContentClassification(
            content_class=ContentClass.ERROR,
            interface_kind=InterfaceKind.DIALOG,
            confidence=0.90,
            reason="test content",
        ),
        sensitive=SensitiveUIDetection(
            sensitive=False,
            level=SensitivityLevel.NONE,
            reason="not sensitive",
            privacy=PrivacyClassification.WORKSPACE,
        ),
        policy_classification=TrustPolicyClassification.VERIFY_FIRST,
        safe_for_reasoning=True,
        safe_for_action=False,
        message="test context",
    )


def _graph_with_error() -> WorkspaceCognitiveGraph:
    runtime = WorkspaceGraphRuntime()
    session = runtime.create_session(workspace_id="workspace")
    app = graph_node(kind=GraphNodeKind.APP, label="VS Code")
    error = graph_node(kind=GraphNodeKind.ERROR, label="AssertionError")

    runtime.apply_delta(
        session_id=session.session_id,
        delta=GraphDelta(
            kind=GraphDeltaKind.GRAPH_REBUILT,
            added_nodes=(app, error),
            added_edges=(
                graph_edge(
                    kind=GraphEdgeKind.CONTAINS,
                    source_node_id=app.node_id,
                    target_node_id=error.node_id,
                ),
            ),
            reason="error graph",
        ),
    )
    graph = runtime.graph_for(session.session_id)

    assert graph is not None

    return graph


def _graph_with_file() -> tuple[WorkspaceCognitiveGraph, str]:
    runtime = WorkspaceGraphRuntime()
    session = runtime.create_session(workspace_id="workspace")
    file_node = graph_node(kind=GraphNodeKind.FILE, label="report.pdf")

    runtime.apply_delta(
        session_id=session.session_id,
        delta=GraphDelta(
            kind=GraphDeltaKind.GRAPH_REBUILT,
            added_nodes=(file_node,),
            reason="file graph",
        ),
    )
    graph = runtime.graph_for(session.session_id)

    assert graph is not None

    return graph, file_node.node_id


def _ground_terminal() -> VisualGroundingResult:
    visual = VisualGroundingRuntime()
    session = visual.create_session(workspace_id="workspace")
    graph_runtime = WorkspaceGraphRuntime()
    graph_session = graph_runtime.create_session(workspace_id="workspace")
    terminal = graph_node(kind=GraphNodeKind.TERMINAL, label="Terminal")

    graph_runtime.apply_delta(
        session_id=graph_session.session_id,
        delta=GraphDelta(
            kind=GraphDeltaKind.GRAPH_REBUILT,
            added_nodes=(terminal,),
            reason="terminal graph",
        ),
    )
    graph = graph_runtime.graph_for(graph_session.session_id)

    assert graph is not None

    return visual.ground(
        GroundingQuery(
            session_id=session.session_id,
            text="focus terminal",
            intent=GroundingIntentKind.FOCUS,
            workspace_graph=graph,
        )
    )