from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    DualInputStream,
    EnvironmentFusionRuntime,
    FusedContext,
    GraphDelta,
    GraphDeltaKind,
    GraphNodeKind,
    TrustPolicyClassification,
    UIContext,
    UIContextRequest,
    UISemanticRuntime,
    VisualContextCacheEntry,
    VisualContextCachePolicy,
    VisualContextFragment,
    VisualContextFragmentKind,
    VisualContextFreshness,
    VisualContextPrefetchReason,
    VisualContextPrefetchRequest,
    VisualContextPrefetchRuntime,
    VisualContextStreamStatus,
    VoiceInputFrame,
    WorkspaceCognitiveGraph,
    WorkspaceGraphRuntime,
    graph_node,
)


def test_runtime_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        VisualContextPrefetchRuntime(name=" ")


def test_policy_rejects_bad_ttl_order() -> None:
    with pytest.raises(ValidationError):
        VisualContextCachePolicy(hot_ttl_turns=10, warm_ttl_turns=3)


def test_create_session() -> None:
    runtime = VisualContextPrefetchRuntime()
    session = runtime.create_session(workspace_id="workspace")

    assert session.workspace_id == "workspace"
    assert runtime.snapshot().session_count == 1


def test_cache_entry_expiry_and_stale() -> None:
    entry = VisualContextCacheEntry(
        key="key",
        value="value",
        freshness=VisualContextFreshness.HOT,
        created_turn=1,
        last_access_turn=1,
        ttl_turns=3,
    )

    assert entry.expired(current_turn=3) is False
    assert entry.expired(current_turn=4) is True
    assert entry.stale(current_turn=5, stale_after_turns=3) is True


def test_prefetch_caches_semantic_graph_fused_and_fragments() -> None:
    runtime = VisualContextPrefetchRuntime()
    session = runtime.create_session(workspace_id="workspace")
    semantic = _semantic_context()
    graph = _graph_with_error()
    fused = _fused_context(graph=graph)

    result = runtime.prefetch(
        VisualContextPrefetchRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            current_turn=1,
            environment_snapshot={"active": True},
            semantic_context=semantic,
            workspace_graph=graph,
            fused_context=fused,
            memory_hint="current project is JARVIS_OS",
            active_intent_hint="debug failing test",
        )
    )

    snapshot = runtime.snapshot()

    assert result.status == VisualContextStreamStatus.READY
    assert result.cached_snapshot is True
    assert result.cached_semantic_scene is True
    assert result.cached_graph is True
    assert result.cached_fused_context is True
    assert result.fragment_count >= 5
    assert snapshot.snapshot_cache_count == 1
    assert snapshot.semantic_cache_count == 1
    assert snapshot.graph_cache_count == 1
    assert snapshot.fragment_cache_count >= 5


def test_stream_returns_hot_fragments_without_rebuild() -> None:
    runtime = VisualContextPrefetchRuntime()
    session = runtime.create_session(workspace_id="workspace")
    graph = _graph_with_error()

    runtime.prefetch(
        VisualContextPrefetchRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            current_turn=1,
            workspace_graph=graph,
            active_intent_hint="fix visible error",
        )
    )
    stream = runtime.stream(
        session_id=session.session_id,
        workspace_id="workspace",
        current_turn=2,
    )

    assert stream.status in {
        VisualContextStreamStatus.READY,
        VisualContextStreamStatus.PARTIAL,
    }
    assert stream.fragments
    assert any(
        fragment.kind == VisualContextFragmentKind.ACTIVE_ERRORS
        for fragment in stream.fragments
    )
    assert stream.safe_for_cognition is True


def test_stream_empty_when_no_prefetch() -> None:
    runtime = VisualContextPrefetchRuntime()
    session = runtime.create_session(workspace_id="workspace")

    stream = runtime.stream(
        session_id=session.session_id,
        workspace_id="workspace",
        current_turn=1,
    )

    assert stream.status == VisualContextStreamStatus.EMPTY
    assert stream.safe_for_cognition is True
    assert stream.safe_for_action_planning is False


def test_missing_session_prefetch_blocks() -> None:
    runtime = VisualContextPrefetchRuntime()

    result = runtime.prefetch(
        VisualContextPrefetchRequest(
            session_id="missing",
            workspace_id="workspace",
            current_turn=1,
        )
    )

    assert result.status == VisualContextStreamStatus.BLOCKED
    assert result.reason == VisualContextPrefetchReason.SESSION_NOT_FOUND


def test_missing_session_stream_blocks() -> None:
    runtime = VisualContextPrefetchRuntime()

    stream = runtime.stream(
        session_id="missing",
        workspace_id="workspace",
        current_turn=1,
    )

    assert stream.status == VisualContextStreamStatus.BLOCKED
    assert stream.safe_for_cognition is False


def test_cache_getters_return_entries() -> None:
    runtime = VisualContextPrefetchRuntime()
    session = runtime.create_session(workspace_id="workspace")
    semantic = _semantic_context()
    graph = _graph_with_error()

    runtime.prefetch(
        VisualContextPrefetchRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            current_turn=1,
            environment_snapshot={"active": True},
            semantic_context=semantic,
            workspace_graph=graph,
        )
    )

    assert runtime.snapshot_entry(workspace_id="workspace", current_turn=2)
    assert runtime.graph_entry(workspace_id="workspace", current_turn=2)
    assert runtime.semantic_entry(
        workspace_id="workspace",
        scene=semantic.scene.kind,
        current_turn=2,
    )


def test_blocked_fragment_blocks_stream() -> None:
    fragment = VisualContextFragment(
        kind=VisualContextFragmentKind.SEMANTIC_SCENE,
        content="blocked secret UI",
        source_key="secret",
        confidence=0.90,
        policy=TrustPolicyClassification.BLOCKED,
        priority=100,
        trust=__import__("jarvis.environment").environment.TrustCalibration(
            confidence=0.90,
            stability=0.90,
            ambiguity=0.0,
            source=__import__("jarvis.environment").environment.EnvironmentSource.OS_OBSERVER,
            reason="blocked fragment",
        ),
    )
    runtime = VisualContextPrefetchRuntime()
    session = runtime.create_session(workspace_id="workspace")

    runtime.prefetch(
        VisualContextPrefetchRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            current_turn=1,
        )
    )
    runtime._fragment_cache.put(
        workspace_id="workspace",
        fragment=fragment,
        current_turn=1,
    )
    stream = runtime.stream(
        session_id=session.session_id,
        workspace_id="workspace",
        current_turn=2,
    )

    assert stream.status == VisualContextStreamStatus.BLOCKED
    assert stream.safe_for_cognition is False


def test_cache_expiry_removes_old_graph() -> None:
    policy = VisualContextCachePolicy(hot_ttl_turns=1, warm_ttl_turns=2)
    runtime = VisualContextPrefetchRuntime(policy=policy)
    session = runtime.create_session(workspace_id="workspace")

    runtime.prefetch(
        VisualContextPrefetchRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            current_turn=1,
            workspace_graph=_graph_with_error(),
        )
    )

    assert runtime.graph_entry(workspace_id="workspace", current_turn=3) is None


def test_snapshot_tracks_streams_and_events() -> None:
    runtime = VisualContextPrefetchRuntime()
    session = runtime.create_session(workspace_id="workspace")

    runtime.prefetch(
        VisualContextPrefetchRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            current_turn=1,
            workspace_graph=_graph_with_error(),
        )
    )
    runtime.stream(
        session_id=session.session_id,
        workspace_id="workspace",
        current_turn=2,
    )
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.stream_count == 1
    assert snapshot.runtime_event_count >= 3
    assert snapshot.last_reason == VisualContextPrefetchReason.FUSED_CONTEXT_STREAMED


def test_reset_clears_runtime() -> None:
    runtime = VisualContextPrefetchRuntime()
    runtime.create_session(workspace_id="workspace")

    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.runtime_event_count == 1
    assert snapshot.last_reason == VisualContextPrefetchReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert VisualContextFragmentKind.FUSED_CONTEXT.value == "fused_context"
    assert VisualContextFreshness.HOT.value == "hot"
    assert VisualContextStreamStatus.READY.value == "ready"


def _semantic_context() -> UIContext:
    runtime = UISemanticRuntime()
    session = runtime.create_session(workspace_id="workspace")

    return runtime.understand(
        UIContextRequest(
            session_id=session.session_id,
            text_regions=(),
            modal_present=True,
            metadata={"test": True},
        )
    )


def _fused_context(*, graph: WorkspaceCognitiveGraph) -> FusedContext:
    runtime = EnvironmentFusionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    return runtime.fuse(
        session_id=session.session_id,
        stream=DualInputStream(
            voice=VoiceInputFrame(text="fix that"),
            workspace_graph=graph,
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