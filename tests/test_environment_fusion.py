from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    ActiveIntentFrame,
    ConversationStateFrame,
    DualInputStream,
    EnvironmentFusionRuntime,
    FusedContext,
    FusionDecision,
    FusionInputSource,
    FusionMode,
    FusionReason,
    FusionStatus,
    GraphDelta,
    GraphDeltaKind,
    GraphNodeKind,
    GroundTruthReconciliationRuntime,
    MemoryContextFrame,
    TrustPolicyClassification,
    UIContext,
    UISemanticRuntime,
    VoiceInputFrame,
    WorkspaceCognitiveGraph,
    WorkspaceGraphRuntime,
    graph_node,
)


def test_runtime_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        EnvironmentFusionRuntime(name=" ")


def test_create_session() -> None:
    runtime = EnvironmentFusionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    assert session.workspace_id == "workspace"
    assert runtime.snapshot().session_count == 1


def test_blocked_context_cannot_be_action_plannable() -> None:
    runtime = EnvironmentFusionRuntime()
    session = runtime.create_session(workspace_id="workspace")
    context = runtime.fuse(
        session_id=session.session_id,
        stream=DualInputStream(voice=VoiceInputFrame(text="hello")),
    )

    with pytest.raises(ValidationError):
        FusedContext(
            status=FusionStatus.BLOCKED,
            reason=FusionReason.FUSION_BLOCKED,
            decision=FusionDecision.BLOCK_COGNITION,
            mode=FusionMode.PASSIVE_AWARENESS,
            stream=context.stream,
            visual_injection=context.visual_injection,
            enrichment=context.enrichment,
            bridge=context.bridge.model_copy(
                update={"safe_for_action_planning": True}
            ),
            trust=context.trust,
            policy=TrustPolicyClassification.BLOCKED,
            message="invalid",
        )


def test_voice_only_context_is_partial() -> None:
    runtime = EnvironmentFusionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    context = runtime.fuse(
        session_id=session.session_id,
        stream=DualInputStream(
            voice=VoiceInputFrame(text="what are we doing?")
        ),
    )

    assert context.status == FusionStatus.PARTIAL
    assert context.decision == FusionDecision.ASK_USER
    assert context.bridge.safe_for_response_generation is True
    assert context.bridge.safe_for_action_planning is False


def test_environment_and_conversation_fuse() -> None:
    runtime = EnvironmentFusionRuntime()
    session = runtime.create_session(workspace_id="workspace")
    graph = _graph_with_error()
    semantic = _semantic_context("fix that")

    context = runtime.fuse(
        session_id=session.session_id,
        stream=DualInputStream(
            voice=VoiceInputFrame(text="fix that"),
            conversation=ConversationStateFrame(active_topic="debugging"),
            workspace_graph=graph,
            semantic_context=semantic,
        ),
    )

    assert context.status in {FusionStatus.FUSED, FusionStatus.PARTIAL}
    assert FusionInputSource.VOICE in context.bridge.allowed_context_sources
    assert FusionInputSource.WORKSPACE_GRAPH in context.bridge.allowed_context_sources
    assert context.visual_injection.visible_error_labels == ("AssertionError",)
    assert "that=AssertionError" in context.enrichment.inferred_references


def test_active_intent_enriches_context() -> None:
    runtime = EnvironmentFusionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    context = runtime.fuse(
        session_id=session.session_id,
        stream=DualInputStream(
            voice=VoiceInputFrame(text="continue it"),
            active_intent=ActiveIntentFrame(
                goal="fix failing Phase 8 test",
                subgoal="inspect error",
            ),
            workspace_graph=_graph_with_error(),
        ),
    )

    assert context.mode == FusionMode.TASK_CONTINUITY
    assert any(
        item == "active_goal=fix failing Phase 8 test"
        for item in context.enrichment.inferred_references
    )


def test_memory_context_is_allowed_source() -> None:
    runtime = EnvironmentFusionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    context = runtime.fuse(
        session_id=session.session_id,
        stream=DualInputStream(
            voice=VoiceInputFrame(text="continue debugging"),
            memory_context=MemoryContextFrame(
                summaries=("current project is JARVIS_OS",),
                reasons=("recent active project memory",),
            ),
            workspace_graph=_graph_with_error(),
        ),
    )

    assert FusionInputSource.MEMORY in context.bridge.allowed_context_sources
    assert "current project is JARVIS_OS" in context.bridge.fused_summary


def test_sensitive_semantic_context_blocks_fusion() -> None:
    runtime = EnvironmentFusionRuntime()
    session = runtime.create_session(workspace_id="workspace")
    semantic_runtime = UISemanticRuntime()
    semantic_session = semantic_runtime.create_session(workspace_id="workspace")
    sensitive = semantic_runtime.understand(
        request=__import__("jarvis.environment").environment.UIContextRequest(
            session_id=semantic_session.session_id,
            text_regions=(),
            metadata={"test": True},
        )
    )
    sensitive = sensitive.model_copy(
        update={
            "safe_for_reasoning": False,
            "safe_for_action": False,
            "policy_classification": TrustPolicyClassification.BLOCKED,
        }
    )

    context = runtime.fuse(
        session_id=session.session_id,
        stream=DualInputStream(
            voice=VoiceInputFrame(text="read this"),
            semantic_context=sensitive,
        ),
    )

    assert context.status == FusionStatus.BLOCKED
    assert context.decision == FusionDecision.BLOCK_COGNITION
    assert context.bridge.safe_for_response_generation is False
    assert context.bridge.safe_for_action_planning is False


def test_divergence_degrades_fusion_and_blocks_grounding_source() -> None:
    runtime = EnvironmentFusionRuntime()
    session = runtime.create_session(workspace_id="workspace")
    ground = GroundTruthReconciliationRuntime()
    ground_session = ground.create_session(workspace_id="workspace")
    report = ground.reconcile(
        session_id=ground_session.session_id,
        belief=__import__("jarvis.environment").environment.BeliefStateSnapshot(
            workspace_id="workspace",
            active_app_id="vscode",
        ),
        observed=__import__("jarvis.environment").environment.ObservedRealitySnapshot(
            workspace_id="workspace",
            active_app_id="chrome",
        ),
    )

    context = runtime.fuse(
        session_id=session.session_id,
        stream=DualInputStream(
            voice=VoiceInputFrame(text="click that"),
            workspace_graph=_graph_with_error(),
            divergence_report=report,
        ),
    )

    assert context.status == FusionStatus.DEGRADED
    assert context.decision == FusionDecision.REFRESH_ENVIRONMENT
    assert FusionInputSource.GROUNDING in context.bridge.blocked_context_sources
    assert context.mode == FusionMode.RECOVERY_CONTEXT


def test_missing_deictic_reference_records_missing_context() -> None:
    runtime = EnvironmentFusionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    context = runtime.fuse(
        session_id=session.session_id,
        stream=DualInputStream(
            voice=VoiceInputFrame(text="fix that"),
            conversation=ConversationStateFrame(active_topic="debugging"),
        ),
    )

    assert "deictic reference target" in context.enrichment.missing_context
    assert context.enrichment.confidence == 0.45


def test_missing_session_fails() -> None:
    runtime = EnvironmentFusionRuntime()

    context = runtime.fuse(
        session_id="missing",
        stream=DualInputStream(voice=VoiceInputFrame(text="hello")),
    )

    assert context.status == FusionStatus.FAILED
    assert context.reason == FusionReason.SESSION_NOT_FOUND
    assert context.bridge.safe_for_response_generation is False


def test_snapshot_tracks_counts() -> None:
    runtime = EnvironmentFusionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    runtime.fuse(
        session_id=session.session_id,
        stream=DualInputStream(voice=VoiceInputFrame(text="hello")),
    )
    runtime.fuse(
        session_id=session.session_id,
        stream=DualInputStream(
            voice=VoiceInputFrame(text="fix that"),
            workspace_graph=_graph_with_error(),
            semantic_context=_semantic_context("fix that"),
        ),
    )
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.fused_context_count == 2
    assert snapshot.partial_count >= 1
    assert snapshot.runtime_event_count >= 3


def test_reset_clears_runtime() -> None:
    runtime = EnvironmentFusionRuntime()
    runtime.create_session(workspace_id="workspace")

    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.runtime_event_count == 1
    assert snapshot.last_reason == FusionReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert FusionInputSource.VOICE.value == "voice"
    assert FusionStatus.FUSED.value == "fused"
    assert FusionDecision.REFRESH_ENVIRONMENT.value == "refresh_environment"


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


def _semantic_context(text: str) -> UIContext:
    semantic = UISemanticRuntime()
    session = semantic.create_session(workspace_id="workspace")
    env = __import__("jarvis.environment").environment

    context = semantic.understand(
        env.UIContextRequest(
            session_id=session.session_id,
            text_regions=(
                env.OCRTextRegion(
                    text="Error: AssertionError",
                    bounds=env.ScreenRegion(x=0, y=0, width=300, height=20),
                    kind=env.OCRTextKind.ERROR,
                    source_kind=env.OCRSourceKind.GENERIC_OCR,
                    source=env.EnvironmentSource.OCR,
                    confidence=env.TextConfidenceScore(
                        confidence=0.90,
                        stability=0.90,
                        ambiguity=0.0,
                        source_kind=env.OCRSourceKind.GENERIC_OCR,
                        accepted=True,
                        reason="test confidence",
                    ),
                    privacy=env.PrivacyClassification.WORKSPACE,
                    trust=env.TrustCalibration(
                        confidence=0.90,
                        stability=0.90,
                        ambiguity=0.0,
                        source=env.EnvironmentSource.OCR,
                        reason="test OCR trust",
                    ),
                    policy_classification=env.TrustPolicyClassification.SAFE,
                    capture_id="capture",
                ),
            ),
            modal_present=True,
            metadata={"text": text},
        )
    )

    return context