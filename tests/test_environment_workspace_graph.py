from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    GraphDelta,
    GraphDeltaKind,
    GraphEdgeKind,
    GraphNode,
    GraphNodeKind,
    GraphQuery,
    GraphQueryDirection,
    GraphQueryKind,
    WorkspaceGraphReason,
    WorkspaceGraphRuntime,
    graph_edge,
    graph_node,
)


def test_runtime_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        WorkspaceGraphRuntime(name=" ")


def test_create_session() -> None:
    runtime = WorkspaceGraphRuntime()
    session = runtime.create_session(workspace_id="workspace")

    assert session.workspace_id == "workspace"
    assert runtime.snapshot().session_count == 1
    assert runtime.graph_for(session.session_id) is not None


def test_node_requires_label() -> None:
    with pytest.raises(ValidationError):
        graph_node(kind=GraphNodeKind.APP, label=" ")


def test_edge_rejects_self_loop() -> None:
    node = graph_node(kind=GraphNodeKind.APP, label="VS Code")

    with pytest.raises(ValidationError):
        graph_edge(
            kind=GraphEdgeKind.CONTAINS,
            source_node_id=node.node_id,
            target_node_id=node.node_id,
        )


def test_add_node() -> None:
    runtime = WorkspaceGraphRuntime()
    session = runtime.create_session(workspace_id="workspace")
    node = graph_node(kind=GraphNodeKind.APP, label="VS Code")

    runtime.add_node(session_id=session.session_id, node=node)
    graph = runtime.graph_for(session.session_id)

    assert graph is not None
    assert node.node_id in graph.nodes
    assert graph.history.entries


def test_add_edge() -> None:
    runtime = WorkspaceGraphRuntime()
    session = runtime.create_session(workspace_id="workspace")
    app = graph_node(kind=GraphNodeKind.APP, label="VS Code")
    window = graph_node(kind=GraphNodeKind.WINDOW, label="Main Window")

    runtime.add_node(session_id=session.session_id, node=app)
    runtime.add_node(session_id=session.session_id, node=window)
    edge = graph_edge(
        kind=GraphEdgeKind.CONTAINS,
        source_node_id=app.node_id,
        target_node_id=window.node_id,
    )
    runtime.add_edge(session_id=session.session_id, edge=edge)
    graph = runtime.graph_for(session.session_id)

    assert graph is not None
    assert edge.edge_id in graph.edges


def test_add_edge_rejects_missing_nodes() -> None:
    runtime = WorkspaceGraphRuntime()
    session = runtime.create_session(workspace_id="workspace")
    edge = graph_edge(
        kind=GraphEdgeKind.CONTAINS,
        source_node_id="missing-a",
        target_node_id="missing-b",
    )

    with pytest.raises(ValueError):
        runtime.add_edge(session_id=session.session_id, edge=edge)


def test_update_node() -> None:
    runtime = WorkspaceGraphRuntime()
    session = runtime.create_session(workspace_id="workspace")
    node = graph_node(kind=GraphNodeKind.FILE, label="old.py")

    runtime.add_node(session_id=session.session_id, node=node)
    updated = node.model_copy(update={"label": "new.py"})
    runtime.update_node(session_id=session.session_id, node=updated)
    graph = runtime.graph_for(session.session_id)

    assert graph is not None
    assert graph.nodes[node.node_id].label == "new.py"


def test_remove_node_deactivates_node_and_edges() -> None:
    runtime = WorkspaceGraphRuntime()
    session = runtime.create_session(workspace_id="workspace")
    app = graph_node(kind=GraphNodeKind.APP, label="VS Code")
    window = graph_node(kind=GraphNodeKind.WINDOW, label="Main Window")
    edge = graph_edge(
        kind=GraphEdgeKind.CONTAINS,
        source_node_id=app.node_id,
        target_node_id=window.node_id,
    )

    runtime.add_node(session_id=session.session_id, node=app)
    runtime.add_node(session_id=session.session_id, node=window)
    runtime.add_edge(session_id=session.session_id, edge=edge)
    runtime.remove_node(session_id=session.session_id, node_id=window.node_id)
    graph = runtime.graph_for(session.session_id)

    assert graph is not None
    assert graph.nodes[window.node_id].active is False
    assert graph.edges[edge.edge_id].active is False


def test_set_focus() -> None:
    runtime = WorkspaceGraphRuntime()
    session = runtime.create_session(workspace_id="workspace")
    node = graph_node(kind=GraphNodeKind.EDITOR, label="main.py")

    runtime.add_node(session_id=session.session_id, node=node)
    runtime.set_focus(session_id=session.session_id, node_id=node.node_id)
    graph = runtime.graph_for(session.session_id)

    assert graph is not None
    assert graph.focused_node_id == node.node_id


def test_query_by_kind() -> None:
    runtime, session_id, nodes = _sample_graph()

    result = runtime.query(
        session_id=session_id,
        query=GraphQuery(
            kind=GraphQueryKind.BY_KIND,
            node_kind=GraphNodeKind.FILE,
        ),
    )

    assert nodes["file"].node_id in {node.node_id for node in result.nodes}


def test_query_by_label() -> None:
    runtime, session_id, _ = _sample_graph()

    result = runtime.query(
        session_id=session_id,
        query=GraphQuery(
            kind=GraphQueryKind.BY_LABEL,
            label_contains="main",
        ),
    )

    assert result.nodes
    assert "main" in result.nodes[0].label.lower()


def test_query_neighbors() -> None:
    runtime, session_id, nodes = _sample_graph()

    result = runtime.query(
        session_id=session_id,
        query=GraphQuery(
            kind=GraphQueryKind.NEIGHBORS,
            node_id=nodes["app"].node_id,
            direction=GraphQueryDirection.OUTGOING,
        ),
    )

    assert nodes["window"].node_id in {node.node_id for node in result.nodes}


def test_query_descendants() -> None:
    runtime, session_id, nodes = _sample_graph()

    result = runtime.query(
        session_id=session_id,
        query=GraphQuery(
            kind=GraphQueryKind.DESCENDANTS,
            node_id=nodes["app"].node_id,
            edge_kind=GraphEdgeKind.CONTAINS,
            max_depth=4,
        ),
    )

    result_ids = {node.node_id for node in result.nodes}

    assert nodes["window"].node_id in result_ids
    assert nodes["file"].node_id in result_ids


def test_query_ancestors() -> None:
    runtime, session_id, nodes = _sample_graph()

    result = runtime.query(
        session_id=session_id,
        query=GraphQuery(
            kind=GraphQueryKind.ANCESTORS,
            node_id=nodes["file"].node_id,
            edge_kind=GraphEdgeKind.CONTAINS,
            max_depth=4,
        ),
    )

    result_ids = {node.node_id for node in result.nodes}

    assert nodes["editor"].node_id in result_ids
    assert nodes["app"].node_id in result_ids


def test_query_blockers() -> None:
    runtime, session_id, nodes = _sample_graph()

    result = runtime.query(
        session_id=session_id,
        query=GraphQuery(
            kind=GraphQueryKind.BLOCKERS,
            node_id=nodes["task"].node_id,
        ),
    )

    assert nodes["error"].node_id in {node.node_id for node in result.nodes}


def test_query_recent_changes() -> None:
    runtime, session_id, _ = _sample_graph()

    result = runtime.query(
        session_id=session_id,
        query=GraphQuery(kind=GraphQueryKind.RECENT_CHANGES),
    )

    assert result.nodes or result.edges


def test_query_focused_context() -> None:
    runtime, session_id, nodes = _sample_graph()

    runtime.set_focus(session_id=session_id, node_id=nodes["editor"].node_id)
    result = runtime.query(
        session_id=session_id,
        query=GraphQuery(kind=GraphQueryKind.FOCUSED_CONTEXT),
    )

    assert result.nodes
    assert result.nodes[0].node_id == nodes["editor"].node_id


def test_changed_after_and_failed_after_edges() -> None:
    runtime = WorkspaceGraphRuntime()
    session = runtime.create_session(workspace_id="workspace")
    command = graph_node(kind=GraphNodeKind.COMMAND, label="pytest")
    file_node = graph_node(kind=GraphNodeKind.FILE, label="main.py")
    error = graph_node(kind=GraphNodeKind.ERROR, label="AssertionError")

    runtime.apply_delta(
        session_id=session.session_id,
        delta=GraphDelta(
            kind=GraphDeltaKind.GRAPH_REBUILT,
            added_nodes=(command, file_node, error),
            reason="sample temporal graph",
        ),
    )
    changed_edge = graph_edge(
        kind=GraphEdgeKind.CHANGED_AFTER,
        source_node_id=file_node.node_id,
        target_node_id=command.node_id,
    )
    failed_edge = graph_edge(
        kind=GraphEdgeKind.FAILED_AFTER,
        source_node_id=error.node_id,
        target_node_id=command.node_id,
    )
    runtime.apply_delta(
        session_id=session.session_id,
        delta=GraphDelta(
            kind=GraphDeltaKind.EDGE_ADDED,
            added_edges=(changed_edge, failed_edge),
            reason="temporal edges added",
        ),
    )
    graph = runtime.graph_for(session.session_id)

    assert graph is not None
    assert graph.edges[changed_edge.edge_id].kind == GraphEdgeKind.CHANGED_AFTER
    assert graph.edges[failed_edge.edge_id].kind == GraphEdgeKind.FAILED_AFTER


def test_missing_session_raises() -> None:
    runtime = WorkspaceGraphRuntime()

    with pytest.raises(ValueError):
        runtime.query(
            session_id="missing",
            query=GraphQuery(kind=GraphQueryKind.RECENT_CHANGES),
        )


def test_snapshot_tracks_counts() -> None:
    runtime, session_id, _ = _sample_graph()

    runtime.query(
        session_id=session_id,
        query=GraphQuery(kind=GraphQueryKind.RECENT_CHANGES),
    )
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.graph_count == 1
    assert snapshot.node_count >= 6
    assert snapshot.edge_count >= 5
    assert snapshot.history_count >= 1
    assert snapshot.query_count == 1


def test_reset_clears_runtime() -> None:
    runtime = WorkspaceGraphRuntime()
    runtime.create_session(workspace_id="workspace")

    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.runtime_event_count == 1
    assert snapshot.last_reason == WorkspaceGraphReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert GraphNodeKind.APP.value == "app"
    assert GraphEdgeKind.CONTAINS.value == "contains"
    assert GraphEdgeKind.CHANGED_AFTER.value == "changed_after"
    assert GraphQueryKind.FOCUSED_CONTEXT.value == "focused_context"


def _sample_graph() -> tuple[WorkspaceGraphRuntime, str, dict[str, GraphNode]]:
    runtime = WorkspaceGraphRuntime()
    session = runtime.create_session(workspace_id="workspace")
    app = graph_node(kind=GraphNodeKind.APP, label="VS Code")
    window = graph_node(kind=GraphNodeKind.WINDOW, label="Main Window")
    editor = graph_node(kind=GraphNodeKind.EDITOR, label="Editor")
    file_node = graph_node(kind=GraphNodeKind.FILE, label="main.py")
    task = graph_node(kind=GraphNodeKind.TASK, label="Fix failing test")
    error = graph_node(kind=GraphNodeKind.ERROR, label="AssertionError")

    runtime.apply_delta(
        session_id=session.session_id,
        delta=GraphDelta(
            kind=GraphDeltaKind.GRAPH_REBUILT,
            added_nodes=(app, window, editor, file_node, task, error),
            added_edges=(
                graph_edge(
                    kind=GraphEdgeKind.CONTAINS,
                    source_node_id=app.node_id,
                    target_node_id=window.node_id,
                ),
                graph_edge(
                    kind=GraphEdgeKind.CONTAINS,
                    source_node_id=window.node_id,
                    target_node_id=editor.node_id,
                ),
                graph_edge(
                    kind=GraphEdgeKind.CONTAINS,
                    source_node_id=editor.node_id,
                    target_node_id=file_node.node_id,
                ),
                graph_edge(
                    kind=GraphEdgeKind.REFERENCES,
                    source_node_id=task.node_id,
                    target_node_id=file_node.node_id,
                ),
                graph_edge(
                    kind=GraphEdgeKind.BLOCKS,
                    source_node_id=error.node_id,
                    target_node_id=task.node_id,
                ),
            ),
            reason="sample graph",
        ),
    )

    return (
        runtime,
        session.session_id,
        {
            "app": app,
            "window": window,
            "editor": editor,
            "file": file_node,
            "task": task,
            "error": error,
        },
    )