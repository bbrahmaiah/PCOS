from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    AmbiguityResolver,
    DetectedUIElement,
    EnvironmentSource,
    GraphDelta,
    GraphDeltaKind,
    GraphEdgeKind,
    GraphNodeKind,
    GroundingCandidate,
    GroundingDecision,
    GroundingIntentKind,
    GroundingQuery,
    GroundingReason,
    GroundingStatus,
    GroundingStrategy,
    GroundingTargetKind,
    OCRSourceKind,
    OCRTextKind,
    OCRTextRegion,
    PrivacyClassification,
    ScreenRegion,
    SemanticScene,
    SemanticSceneKind,
    TargetTrustPolicy,
    TextConfidenceScore,
    TrustCalibration,
    TrustPolicyClassification,
    UIContext,
    UIDetectionRequest,
    UIDetectionRuntime,
    UISemanticReason,
    UISemanticStatus,
    VisualGroundingRuntime,
    WorkspaceCognitiveGraph,
    WorkspaceGraphRuntime,
    graph_edge,
    graph_node,
)


def test_runtime_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        VisualGroundingRuntime(name=" ")


def test_create_session() -> None:
    runtime = VisualGroundingRuntime()
    session = runtime.create_session(workspace_id="workspace")

    assert session.workspace_id == "workspace"
    assert runtime.snapshot().session_count == 1


def test_policy_rejects_bad_threshold_order() -> None:
    from jarvis.environment import TargetTrustPolicy

    with pytest.raises(ValidationError):
        TargetTrustPolicy(
            minimum_grounding_confidence=0.90,
            verify_first_confidence=0.80,
        )


def test_workspace_graph_grounding_first() -> None:
    runtime = VisualGroundingRuntime()
    session = runtime.create_session(workspace_id="workspace")
    graph = _graph()

    result = runtime.ground(
        GroundingQuery(
            session_id=session.session_id,
            text="focus terminal",
            intent=GroundingIntentKind.FOCUS,
            workspace_graph=graph,
        )
    )

    assert result.status == GroundingStatus.GROUNDED
    assert result.selected is not None
    assert result.selected.strategy == GroundingStrategy.WORKSPACE_GRAPH_QUERY
    assert result.selected.graph_node is not None
    assert result.selected.graph_node.kind == GraphNodeKind.TERMINAL


def test_accessibility_exact_match_grounding() -> None:
    runtime = VisualGroundingRuntime()
    session = runtime.create_session(workspace_id="workspace")
    element = _run_button()

    result = runtime.ground(
        GroundingQuery(
            session_id=session.session_id,
            text="click Run",
            intent=GroundingIntentKind.CLICK,
            ui_elements=(element,),
        )
    )

    assert result.status == GroundingStatus.GROUNDED
    assert result.selected is not None
    assert result.selected.strategy == GroundingStrategy.ACCESSIBILITY_EXACT_MATCH
    assert result.selected.ui_element is not None
    assert result.safe_for_action_planning is True


def test_ocr_exact_match_grounding() -> None:
    runtime = VisualGroundingRuntime()
    session = runtime.create_session(workspace_id="workspace")
    region = _ocr("AssertionError")

    result = runtime.ground(
        GroundingQuery(
            session_id=session.session_id,
            text="inspect AssertionError",
            intent=GroundingIntentKind.INSPECT,
            text_regions=(region,),
        )
    )

    assert result.status == GroundingStatus.GROUNDED
    assert result.selected is not None
    assert result.selected.strategy == GroundingStrategy.OCR_EXACT_MATCH


def test_fuzzy_text_grounding_requires_verify_first() -> None:
    runtime = VisualGroundingRuntime()
    session = runtime.create_session(workspace_id="workspace")
    region = _ocr("AssertionError in test file")

    result = runtime.ground(
        GroundingQuery(
            session_id=session.session_id,
            text="inspect assertion error",
            intent=GroundingIntentKind.INSPECT,
            text_regions=(region,),
        )
    )

    assert result.candidates
    assert result.candidates[0].strategy == GroundingStrategy.FUZZY_TEXT_MATCH
    assert result.status == GroundingStatus.LOW_CONFIDENCE
    assert result.decision == GroundingDecision.ASK_USER
    assert result.selected is None
    assert result.safe_for_action_planning is False


def test_semantic_icon_match_for_run_button() -> None:
    runtime = VisualGroundingRuntime()
    session = runtime.create_session(workspace_id="workspace")
    element = _run_button()

    result = runtime.ground(
        GroundingQuery(
            session_id=session.session_id,
            text="click play",
            intent=GroundingIntentKind.CLICK,
            ui_elements=(element,),
        )
    )

    assert result.selected is not None
    assert result.selected.strategy == GroundingStrategy.SEMANTIC_ICON_MATCH


def test_spatial_reference_low_confidence_asks() -> None:
    runtime = VisualGroundingRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.ground(
        GroundingQuery(
            session_id=session.session_id,
            text="click this",
            intent=GroundingIntentKind.CLICK,
            active_region=ScreenRegion(x=10, y=10, width=50, height=50),
        )
    )

    assert result.status == GroundingStatus.LOW_CONFIDENCE
    assert result.decision == GroundingDecision.ASK_USER
    assert result.safe_for_action_planning is False


def test_context_inference_for_error_scene() -> None:
    runtime = VisualGroundingRuntime()
    session = runtime.create_session(workspace_id="workspace")
    context = _error_context()

    result = runtime.ground(
        GroundingQuery(
            session_id=session.session_id,
            text="inspect the error",
            intent=GroundingIntentKind.INSPECT,
            ui_context=context,
        )
    )

    assert result.selected is not None
    assert result.selected.strategy == GroundingStrategy.CONTEXT_INFERENCE
    assert result.selected.scene_kind == SemanticSceneKind.ERROR_DIALOG


def test_blocked_policy_blocks_grounding() -> None:
    runtime = VisualGroundingRuntime()
    session = runtime.create_session(workspace_id="workspace")
    region = _ocr(
        "password",
        policy=TrustPolicyClassification.BLOCKED,
    )

    result = runtime.ground(
        GroundingQuery(
            session_id=session.session_id,
            text="read password",
            intent=GroundingIntentKind.READ,
            text_regions=(region,),
        )
    )

    assert result.status == GroundingStatus.BLOCKED
    assert result.decision == GroundingDecision.BLOCKED
    assert result.safe_for_action_planning is False


def test_ambiguous_targets_ask_user() -> None:
    resolver = AmbiguityResolver()
    first = GroundingCandidate(
        target_kind=GroundingTargetKind.OCR_TEXT_REGION,
        label="Run tests",
        confidence=0.80,
        strategy=GroundingStrategy.FUZZY_TEXT_MATCH,
        policy=TrustPolicyClassification.SAFE,
        trust=TrustCalibration(
            confidence=0.80,
            stability=0.80,
            ambiguity=0.20,
            source=EnvironmentSource.OCR,
            reason="test candidate",
        ),
    )
    second = GroundingCandidate(
        target_kind=GroundingTargetKind.OCR_TEXT_REGION,
        label="Run build",
        confidence=0.76,
        strategy=GroundingStrategy.FUZZY_TEXT_MATCH,
        policy=TrustPolicyClassification.SAFE,
        trust=TrustCalibration(
            confidence=0.76,
            stability=0.76,
            ambiguity=0.24,
            source=EnvironmentSource.OCR,
            reason="test candidate",
        ),
    )

    result = resolver.resolve(
        candidates=(first, second),
        policy=TargetTrustPolicy(ambiguity_margin=0.08),
    )

    assert result.ambiguous is True
    assert result.ask_user_message is not None
    assert len(result.top_candidates) == 2


def test_not_found_result() -> None:
    runtime = VisualGroundingRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.ground(
        GroundingQuery(
            session_id=session.session_id,
            text="click nonexistent target",
            intent=GroundingIntentKind.CLICK,
        )
    )

    assert result.status == GroundingStatus.NOT_FOUND
    assert result.reason == GroundingReason.TARGET_NOT_FOUND


def test_missing_session_fails() -> None:
    runtime = VisualGroundingRuntime()

    result = runtime.ground(
        GroundingQuery(
            session_id="missing",
            text="click run",
            intent=GroundingIntentKind.CLICK,
        )
    )

    assert result.status == GroundingStatus.FAILED
    assert result.reason == GroundingReason.SESSION_NOT_FOUND


def test_snapshot_tracks_counts() -> None:
    runtime = VisualGroundingRuntime()
    session = runtime.create_session(workspace_id="workspace")

    runtime.ground(
        GroundingQuery(
            session_id=session.session_id,
            text="click Run",
            intent=GroundingIntentKind.CLICK,
            ui_elements=(_run_button(),),
        )
    )
    runtime.ground(
        GroundingQuery(
            session_id=session.session_id,
            text="click unknown",
            intent=GroundingIntentKind.CLICK,
        )
    )
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.result_count == 2
    assert snapshot.grounded_count == 1
    assert snapshot.not_found_count == 1


def test_reset_clears_runtime() -> None:
    runtime = VisualGroundingRuntime()
    runtime.create_session(workspace_id="workspace")

    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.runtime_event_count == 1
    assert snapshot.last_reason == GroundingReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert GroundingStrategy.WORKSPACE_GRAPH_QUERY.value == "workspace_graph_query"
    assert GroundingDecision.ASK_USER.value == "ask_user"
    assert GroundingStatus.GROUNDED.value == "grounded"


def _run_button() -> DetectedUIElement:
    detector = UIDetectionRuntime()
    session = detector.create_session(workspace_id="workspace")
    result = detector.detect_elements(
        UIDetectionRequest(
            session_id=session.session_id,
            region=ScreenRegion(x=0, y=0, width=500, height=300),
        )
    )

    assert result.elements

    return result.elements[0]


def _ocr(
    text: str,
    *,
    policy: TrustPolicyClassification = TrustPolicyClassification.SAFE,
) -> OCRTextRegion:
    return OCRTextRegion(
        text=text,
        bounds=ScreenRegion(x=0, y=0, width=300, height=20),
        kind=OCRTextKind.PLAIN_TEXT,
        source_kind=OCRSourceKind.GENERIC_OCR,
        source=EnvironmentSource.OCR,
        confidence=TextConfidenceScore(
            confidence=0.90,
            stability=0.90,
            ambiguity=0.0,
            source_kind=OCRSourceKind.GENERIC_OCR,
            accepted=True,
            reason="test confidence",
        ),
        privacy=PrivacyClassification.WORKSPACE,
        trust=TrustCalibration(
            confidence=0.90,
            stability=0.90,
            ambiguity=0.0,
            source=EnvironmentSource.OCR,
            reason="test OCR trust",
        ),
        policy_classification=policy,
        capture_id="capture",
    )


def _error_context() -> UIContext:
    from jarvis.environment import (
        ContentClass,
        ContentClassification,
        InterfaceKind,
        SensitiveUIDetection,
        SensitivityLevel,
    )

    scene = SemanticScene(
        kind=SemanticSceneKind.ERROR_DIALOG,
        interface_kind=InterfaceKind.DIALOG,
        confidence=0.90,
        summary="error dialog requiring attention",
        region=ScreenRegion(x=0, y=0, width=300, height=200),
        trust=TrustCalibration(
            confidence=0.90,
            stability=0.90,
            ambiguity=0.0,
            source=EnvironmentSource.OS_OBSERVER,
            reason="test error scene",
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
            reason="error content",
        ),
        sensitive=SensitiveUIDetection(
            sensitive=False,
            level=SensitivityLevel.NONE,
            reason="not sensitive",
        ),
        policy_classification=TrustPolicyClassification.VERIFY_FIRST,
        safe_for_reasoning=True,
        safe_for_action=False,
        message="error context",
    )


def _graph() -> WorkspaceCognitiveGraph:
    runtime = WorkspaceGraphRuntime()
    session = runtime.create_session(workspace_id="workspace")
    app = graph_node(kind=GraphNodeKind.APP, label="VS Code")
    terminal = graph_node(kind=GraphNodeKind.TERMINAL, label="Terminal")
    editor = graph_node(kind=GraphNodeKind.EDITOR, label="Editor")

    runtime.apply_delta(
        session_id=session.session_id,
        delta=GraphDelta(
            kind=GraphDeltaKind.GRAPH_REBUILT,
            added_nodes=(app, terminal, editor),
            added_edges=(
                graph_edge(
                    kind=GraphEdgeKind.CONTAINS,
                    source_node_id=app.node_id,
                    target_node_id=terminal.node_id,
                ),
                graph_edge(
                    kind=GraphEdgeKind.CONTAINS,
                    source_node_id=app.node_id,
                    target_node_id=editor.node_id,
                ),
            ),
            reason="test graph",
        ),
    )
    graph = runtime.graph_for(session.session_id)

    assert graph is not None

    return graph