from __future__ import annotations

from enum import StrEnum
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.environment.models import (
    EnvironmentSource,
    PrivacyClassification,
    ScreenRegion,
    TrustCalibration,
)
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class GraphNodeKind(StrEnum):
    APP = "app"
    WINDOW = "window"
    PANEL = "panel"
    FILE = "file"
    TAB = "tab"
    TERMINAL = "terminal"
    EDITOR = "editor"
    DIALOG = "dialog"
    ERROR = "error"
    COMMAND = "command"
    PROJECT = "project"
    TASK = "task"
    WORKFLOW = "workflow"
    TEXT_REGION = "text_region"
    UI_ELEMENT = "ui_element"
    UNKNOWN = "unknown"


class GraphEdgeKind(StrEnum):
    CONTAINS = "contains"
    FOCUSES = "focuses"
    OPENS = "opens"
    REFERENCES = "references"
    SPAWNED_BY = "spawned_by"
    DEPENDS_ON = "depends_on"
    BLOCKS = "blocks"
    LAST_USED = "last_used"
    CHANGED_AFTER = "changed_after"
    FAILED_AFTER = "failed_after"


class GraphDeltaKind(StrEnum):
    NODE_ADDED = "node_added"
    NODE_UPDATED = "node_updated"
    NODE_REMOVED = "node_removed"
    EDGE_ADDED = "edge_added"
    EDGE_REMOVED = "edge_removed"
    FOCUS_CHANGED = "focus_changed"
    GRAPH_REBUILT = "graph_rebuilt"


class GraphQueryKind(StrEnum):
    BY_KIND = "by_kind"
    BY_LABEL = "by_label"
    NEIGHBORS = "neighbors"
    DESCENDANTS = "descendants"
    ANCESTORS = "ancestors"
    BLOCKERS = "blockers"
    RECENT_CHANGES = "recent_changes"
    FOCUSED_CONTEXT = "focused_context"


class GraphQueryDirection(StrEnum):
    OUTGOING = "outgoing"
    INCOMING = "incoming"
    BOTH = "both"


class WorkspaceGraphReason(StrEnum):
    SESSION_CREATED = "session_created"
    GRAPH_NODE_ADDED = "graph_node_added"
    GRAPH_NODE_UPDATED = "graph_node_updated"
    GRAPH_NODE_REMOVED = "graph_node_removed"
    GRAPH_EDGE_ADDED = "graph_edge_added"
    GRAPH_EDGE_REMOVED = "graph_edge_removed"
    GRAPH_DELTA_APPLIED = "graph_delta_applied"
    GRAPH_QUERY_EXECUTED = "graph_query_executed"
    FOCUS_CHANGED = "focus_changed"
    HISTORY_RECORDED = "history_recorded"
    SESSION_NOT_FOUND = "session_not_found"
    RUNTIME_RESET = "runtime_reset"


class WorkspaceGraphEventKind(StrEnum):
    SESSION_CREATED = "session_created"
    GRAPH_MUTATED = "graph_mutated"
    DELTA_APPLIED = "delta_applied"
    QUERY_EXECUTED = "query_executed"
    HISTORY_RECORDED = "history_recorded"
    RUNTIME_RESET = "runtime_reset"


class GraphNode(OrchestrationModel):
    """
    Cognitive graph node.

    A node is a thing JARVIS can reason about:
    app, window, panel, file, tab, terminal, editor, dialog, error,
    command, project, task, workflow, text region, or UI element.
    """

    node_id: str = Field(default_factory=lambda: f"graph_node_{uuid4().hex}")
    kind: GraphNodeKind
    label: str
    trust: TrustCalibration
    privacy: PrivacyClassification = PrivacyClassification.WORKSPACE
    region: ScreenRegion | None = None
    source: EnvironmentSource = EnvironmentSource.OS_OBSERVER
    confidence: float = Field(default=0.90, ge=0.0, le=1.0)
    stable: bool = True
    active: bool = True
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("node_id", "label")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class GraphEdge(OrchestrationModel):
    """
    Cognitive graph edge.

    Edges make the workspace meaningful:
    contains, focuses, opens, references, spawned_by, depends_on,
    blocks, last_used, changed_after, failed_after.
    """

    edge_id: str = Field(default_factory=lambda: f"graph_edge_{uuid4().hex}")
    kind: GraphEdgeKind
    source_node_id: str
    target_node_id: str
    trust: TrustCalibration
    confidence: float = Field(default=0.90, ge=0.0, le=1.0)
    active: bool = True
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("edge_id", "source_node_id", "target_node_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _edge_cannot_self_loop(self) -> GraphEdge:
        if self.source_node_id == self.target_node_id:
            raise ValueError("graph edge cannot point to itself.")

        return self


class GraphDelta(OrchestrationModel):
    """
    Graph delta.

    This captures what changed, so the graph can reason temporally.
    """

    delta_id: str = Field(default_factory=lambda: f"graph_delta_{uuid4().hex}")
    kind: GraphDeltaKind
    added_nodes: tuple[GraphNode, ...] = ()
    updated_nodes: tuple[GraphNode, ...] = ()
    removed_node_ids: tuple[str, ...] = ()
    added_edges: tuple[GraphEdge, ...] = ()
    removed_edge_ids: tuple[str, ...] = ()
    focused_node_id: str | None = None
    reason: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("delta_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class GraphQuery(OrchestrationModel):
    """
    Query against the workspace cognitive graph.
    """

    query_id: str = Field(default_factory=lambda: f"graph_query_{uuid4().hex}")
    kind: GraphQueryKind
    node_id: str | None = None
    node_kind: GraphNodeKind | None = None
    edge_kind: GraphEdgeKind | None = None
    label_contains: str | None = None
    direction: GraphQueryDirection = GraphQueryDirection.OUTGOING
    max_depth: int = Field(default=2, ge=1, le=10)
    limit: int = Field(default=25, ge=1, le=100)
    include_inactive: bool = False
    created_at: object = Field(default_factory=utc_now)

    @field_validator("query_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class GraphQueryResult(OrchestrationModel):
    """
    Result of a graph query.
    """

    result_id: str = Field(default_factory=lambda: f"graph_query_result_{uuid4().hex}")
    query: GraphQuery
    nodes: tuple[GraphNode, ...] = ()
    edges: tuple[GraphEdge, ...] = ()
    created_at: object = Field(default_factory=utc_now)

    @field_validator("result_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class GraphHistoryEntry(OrchestrationModel):
    """
    One graph history entry.

    This is the temporal memory of the graph.
    """

    history_id: str = Field(default_factory=lambda: f"graph_history_{uuid4().hex}")
    delta: GraphDelta
    node_count_after: int = Field(ge=0)
    edge_count_after: int = Field(ge=0)
    created_at: object = Field(default_factory=utc_now)

    @field_validator("history_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class GraphHistory(OrchestrationModel):
    """
    Recent graph history.
    """

    entries: tuple[GraphHistoryEntry, ...] = ()
    max_entries: int = Field(default=100, ge=1, le=1000)

    @model_validator(mode="after")
    def _history_limit(self) -> GraphHistory:
        if len(self.entries) > self.max_entries:
            raise ValueError("graph history exceeds max_entries.")

        return self


class WorkspaceCognitiveGraph(OrchestrationModel):
    """
    Current cognitive graph state.
    """

    graph_id: str = Field(default_factory=lambda: f"workspace_graph_{uuid4().hex}")
    workspace_id: str
    nodes: dict[str, GraphNode] = Field(default_factory=dict)
    edges: dict[str, GraphEdge] = Field(default_factory=dict)
    focused_node_id: str | None = None
    history: GraphHistory = Field(default_factory=GraphHistory)
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)

    @field_validator("graph_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class WorkspaceGraphSession(OrchestrationModel):
    """
    Workspace graph runtime session.
    """

    session_id: str = Field(default_factory=lambda: f"graph_session_{uuid4().hex}")
    workspace_id: str
    graph: WorkspaceCognitiveGraph
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class WorkspaceGraphRuntimeEvent(OrchestrationModel):
    """
    Workspace graph runtime event.
    """

    event_id: str = Field(default_factory=lambda: f"graph_event_{uuid4().hex}")
    kind: WorkspaceGraphEventKind
    reason: WorkspaceGraphReason
    session_id: str | None = None
    node_id: str | None = None
    edge_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class WorkspaceGraphRuntimeSnapshot(OrchestrationModel):
    """
    Diagnostics for Phase 8 Step 13.
    """

    name: str
    session_count: int = Field(ge=0)
    graph_count: int = Field(ge=0)
    node_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    active_node_count: int = Field(ge=0)
    active_edge_count: int = Field(ge=0)
    history_count: int = Field(ge=0)
    query_count: int = Field(ge=0)
    runtime_event_count: int = Field(ge=0)
    last_reason: WorkspaceGraphReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        return _clean_required(value)


class WorkspaceGraphRuntime:
    """
    Phase 8 Step 13 Workspace Cognitive Graph Runtime.

    Responsibilities:
    - maintain a cognitive workspace graph
    - add/update/remove graph nodes
    - add/remove graph edges
    - apply graph deltas
    - track temporal history
    - answer graph queries
    - model changed-after and failed-after relationships

    Non-responsibilities:
    - no screen capture
    - no OCR
    - no action execution
    - no low-level coordinate automation
    """

    def __init__(
        self,
        *,
        name: str = "workspace_graph_runtime",
    ) -> None:
        cleaned = name.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._sessions: dict[str, WorkspaceGraphSession] = {}
        self._query_results: list[GraphQueryResult] = []
        self._events: list[WorkspaceGraphRuntimeEvent] = []
        self._lock = RLock()
        self._last_reason: WorkspaceGraphReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def create_session(
        self,
        *,
        workspace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceGraphSession:
        graph = WorkspaceCognitiveGraph(workspace_id=workspace_id)
        session = WorkspaceGraphSession(
            workspace_id=workspace_id,
            graph=graph,
            metadata=metadata or {},
        )
        event = self._event(
            kind=WorkspaceGraphEventKind.SESSION_CREATED,
            reason=WorkspaceGraphReason.SESSION_CREATED,
            session_id=session.session_id,
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._events.append(event)
            self._last_reason = event.reason

        return session

    def add_node(
        self,
        *,
        session_id: str,
        node: GraphNode,
    ) -> GraphNode:
        self.apply_delta(
            session_id=session_id,
            delta=GraphDelta(
                kind=GraphDeltaKind.NODE_ADDED,
                added_nodes=(node,),
                reason="node added",
            ),
        )

        return node

    def update_node(
        self,
        *,
        session_id: str,
        node: GraphNode,
    ) -> GraphNode:
        updated = node.model_copy(update={"updated_at": utc_now()})
        self.apply_delta(
            session_id=session_id,
            delta=GraphDelta(
                kind=GraphDeltaKind.NODE_UPDATED,
                updated_nodes=(updated,),
                reason="node updated",
            ),
        )

        return updated

    def remove_node(
        self,
        *,
        session_id: str,
        node_id: str,
    ) -> None:
        self.apply_delta(
            session_id=session_id,
            delta=GraphDelta(
                kind=GraphDeltaKind.NODE_REMOVED,
                removed_node_ids=(node_id,),
                reason="node removed",
            ),
        )

    def add_edge(
        self,
        *,
        session_id: str,
        edge: GraphEdge,
    ) -> GraphEdge:
        self.apply_delta(
            session_id=session_id,
            delta=GraphDelta(
                kind=GraphDeltaKind.EDGE_ADDED,
                added_edges=(edge,),
                reason="edge added",
            ),
        )

        return edge

    def remove_edge(
        self,
        *,
        session_id: str,
        edge_id: str,
    ) -> None:
        self.apply_delta(
            session_id=session_id,
            delta=GraphDelta(
                kind=GraphDeltaKind.EDGE_REMOVED,
                removed_edge_ids=(edge_id,),
                reason="edge removed",
            ),
        )

    def set_focus(
        self,
        *,
        session_id: str,
        node_id: str,
    ) -> None:
        self.apply_delta(
            session_id=session_id,
            delta=GraphDelta(
                kind=GraphDeltaKind.FOCUS_CHANGED,
                focused_node_id=node_id,
                reason="focus changed",
            ),
        )

    def apply_delta(
        self,
        *,
        session_id: str,
        delta: GraphDelta,
    ) -> WorkspaceCognitiveGraph:
        session = self._session_or_raise(session_id)
        graph = session.graph

        nodes = dict(graph.nodes)
        edges = dict(graph.edges)
        focused_node_id = graph.focused_node_id

        for node in delta.added_nodes:
            nodes[node.node_id] = node

        for node in delta.updated_nodes:
            if node.node_id not in nodes:
                raise ValueError(f"graph node not found: {node.node_id}")

            nodes[node.node_id] = node

        for node_id in delta.removed_node_ids:
            if node_id in nodes:
                nodes[node_id] = nodes[node_id].model_copy(
                    update={"active": False, "updated_at": utc_now()}
                )

            edges = {
                edge_id: edge.model_copy(update={"active": False})
                if edge.source_node_id == node_id or edge.target_node_id == node_id
                else edge
                for edge_id, edge in edges.items()
            }

        for edge in delta.added_edges:
            if edge.source_node_id not in nodes:
                raise ValueError(f"source graph node not found: {edge.source_node_id}")

            if edge.target_node_id not in nodes:
                raise ValueError(f"target graph node not found: {edge.target_node_id}")

            edges[edge.edge_id] = edge

        for edge_id in delta.removed_edge_ids:
            if edge_id in edges:
                edges[edge_id] = edges[edge_id].model_copy(update={"active": False})

        if delta.focused_node_id is not None:
            if delta.focused_node_id not in nodes:
                raise ValueError(
                    f"focused graph node not found: {delta.focused_node_id}"
            )

            focused_node_id = delta.focused_node_id

        history_entry = GraphHistoryEntry(
            delta=delta,
            node_count_after=len(nodes),
            edge_count_after=len(edges),
        )
        history_entries = (
            *graph.history.entries,
            history_entry,
        )[-graph.history.max_entries :]
        updated_graph = graph.model_copy(
            update={
                "nodes": nodes,
                "edges": edges,
                "focused_node_id": focused_node_id,
                "history": GraphHistory(
                    entries=history_entries,
                    max_entries=graph.history.max_entries,
                ),
                "updated_at": utc_now(),
            }
        )
        updated_session = session.model_copy(
            update={"graph": updated_graph, "updated_at": utc_now()}
        )
        event = self._event(
            kind=WorkspaceGraphEventKind.DELTA_APPLIED,
            reason=WorkspaceGraphReason.GRAPH_DELTA_APPLIED,
            session_id=session_id,
            metadata={"delta_kind": delta.kind.value},
        )

        with self._lock:
            self._sessions[session_id] = updated_session
            self._events.append(event)
            self._last_reason = event.reason

        return updated_graph

    def query(
        self,
        *,
        session_id: str,
        query: GraphQuery,
    ) -> GraphQueryResult:
        graph = self._session_or_raise(session_id).graph

        if query.kind == GraphQueryKind.BY_KIND:
            result = self._query_by_kind(graph=graph, query=query)
        elif query.kind == GraphQueryKind.BY_LABEL:
            result = self._query_by_label(graph=graph, query=query)
        elif query.kind == GraphQueryKind.NEIGHBORS:
            result = self._query_neighbors(graph=graph, query=query)
        elif query.kind == GraphQueryKind.DESCENDANTS:
            result = self._query_walk(
                graph=graph,
                query=query,
                reverse=False,
            )
        elif query.kind == GraphQueryKind.ANCESTORS:
            result = self._query_walk(
                graph=graph,
                query=query,
                reverse=True,
            )
        elif query.kind == GraphQueryKind.BLOCKERS:
            result = self._query_blockers(graph=graph, query=query)
        elif query.kind == GraphQueryKind.RECENT_CHANGES:
            result = self._query_recent_changes(graph=graph, query=query)
        else:
            result = self._query_focused_context(graph=graph, query=query)

        event = self._event(
            kind=WorkspaceGraphEventKind.QUERY_EXECUTED,
            reason=WorkspaceGraphReason.GRAPH_QUERY_EXECUTED,
            session_id=session_id,
            metadata={"query_kind": query.kind.value},
        )

        with self._lock:
            self._query_results.append(result)
            self._events.append(event)
            self._last_reason = event.reason

        return result

    def graph_for(self, session_id: str) -> WorkspaceCognitiveGraph | None:
        with self._lock:
            session = self._sessions.get(session_id)

            return session.graph if session is not None else None

    def session_for(self, session_id: str) -> WorkspaceGraphSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def query_results(self) -> tuple[GraphQueryResult, ...]:
        with self._lock:
            return tuple(self._query_results)

    def events(self) -> tuple[WorkspaceGraphRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def snapshot(self) -> WorkspaceGraphRuntimeSnapshot:
        with self._lock:
            graphs = [session.graph for session in self._sessions.values()]
            nodes = [node for graph in graphs for node in graph.nodes.values()]
            edges = [edge for graph in graphs for edge in graph.edges.values()]
            history_count = sum(
                len(graph.history.entries) for graph in graphs
            )

            return WorkspaceGraphRuntimeSnapshot(
                name=self.name,
                session_count=len(self._sessions),
                graph_count=len(graphs),
                node_count=len(nodes),
                edge_count=len(edges),
                active_node_count=sum(1 for node in nodes if node.active),
                active_edge_count=sum(1 for edge in edges if edge.active),
                history_count=history_count,
                query_count=len(self._query_results),
                runtime_event_count=len(self._events),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        event = self._event(
            kind=WorkspaceGraphEventKind.RUNTIME_RESET,
            reason=WorkspaceGraphReason.RUNTIME_RESET,
        )

        with self._lock:
            self._sessions.clear()
            self._query_results.clear()
            self._events.clear()
            self._events.append(event)
            self._last_reason = event.reason

    def _session_or_raise(self, session_id: str) -> WorkspaceGraphSession:
        with self._lock:
            session = self._sessions.get(session_id)

        if session is None:
            raise ValueError(f"workspace graph session not found: {session_id}")

        return session

    @staticmethod
    def _query_by_kind(
        *,
        graph: WorkspaceCognitiveGraph,
        query: GraphQuery,
    ) -> GraphQueryResult:
        nodes = tuple(
            node
            for node in graph.nodes.values()
            if (query.include_inactive or node.active)
            and node.kind == query.node_kind
        )[: query.limit]

        return GraphQueryResult(query=query, nodes=nodes)

    @staticmethod
    def _query_by_label(
        *,
        graph: WorkspaceCognitiveGraph,
        query: GraphQuery,
    ) -> GraphQueryResult:
        needle = (query.label_contains or "").lower()
        nodes = tuple(
            node
            for node in graph.nodes.values()
            if (query.include_inactive or node.active)
            and needle in node.label.lower()
        )[: query.limit]

        return GraphQueryResult(query=query, nodes=nodes)

    def _query_neighbors(
        self,
        *,
        graph: WorkspaceCognitiveGraph,
        query: GraphQuery,
    ) -> GraphQueryResult:
        if query.node_id is None:
            return GraphQueryResult(query=query)

        edges = self._matching_edges(graph=graph, query=query)
        node_ids = _neighbor_ids(
            node_id=query.node_id,
            edges=edges,
            direction=query.direction,
        )
        nodes = tuple(
            graph.nodes[node_id]
            for node_id in node_ids
            if node_id in graph.nodes
        )[: query.limit]

        return GraphQueryResult(query=query, nodes=nodes, edges=edges[: query.limit])

    def _query_walk(
        self,
        *,
        graph: WorkspaceCognitiveGraph,
        query: GraphQuery,
        reverse: bool,
    ) -> GraphQueryResult:
        if query.node_id is None:
            return GraphQueryResult(query=query)

        visited: set[str] = set()
        frontier: list[tuple[str, int]] = [(query.node_id, 0)]
        collected_nodes: list[GraphNode] = []
        collected_edges: list[GraphEdge] = []

        while frontier and len(collected_nodes) < query.limit:
            current, depth = frontier.pop(0)

            if current in visited or depth >= query.max_depth:
                continue

            visited.add(current)

            edges = tuple(
                edge
                for edge in graph.edges.values()
                if edge.active
                and (query.edge_kind is None or edge.kind == query.edge_kind)
                and (
                    edge.target_node_id == current
                    if reverse
                    else edge.source_node_id == current
                )
            )

            for edge in edges:
                next_id = edge.source_node_id if reverse else edge.target_node_id

                if next_id in graph.nodes and next_id not in visited:
                    collected_edges.append(edge)
                    collected_nodes.append(graph.nodes[next_id])
                    frontier.append((next_id, depth + 1))

        return GraphQueryResult(
            query=query,
            nodes=tuple(collected_nodes[: query.limit]),
            edges=tuple(collected_edges[: query.limit]),
        )

    @staticmethod
    def _query_blockers(
        *,
        graph: WorkspaceCognitiveGraph,
        query: GraphQuery,
    ) -> GraphQueryResult:
        edges = tuple(
            edge
            for edge in graph.edges.values()
            if edge.active
            and edge.kind == GraphEdgeKind.BLOCKS
            and (query.node_id is None or edge.target_node_id == query.node_id)
        )[: query.limit]
        nodes = tuple(
            graph.nodes[edge.source_node_id]
            for edge in edges
            if edge.source_node_id in graph.nodes
        )

        return GraphQueryResult(query=query, nodes=nodes, edges=edges)

    @staticmethod
    def _query_recent_changes(
        *,
        graph: WorkspaceCognitiveGraph,
        query: GraphQuery,
    ) -> GraphQueryResult:
        entries = graph.history.entries[-query.limit :]
        node_ids: set[str] = set()
        edge_ids: set[str] = set()

        for entry in entries:
            for node in entry.delta.added_nodes:
                node_ids.add(node.node_id)

            for node in entry.delta.updated_nodes:
                node_ids.add(node.node_id)

            node_ids.update(entry.delta.removed_node_ids)

            for edge in entry.delta.added_edges:
                edge_ids.add(edge.edge_id)

            edge_ids.update(entry.delta.removed_edge_ids)

        nodes = tuple(
            graph.nodes[node_id]
            for node_id in node_ids
            if node_id in graph.nodes
        )
        edges = tuple(
            graph.edges[edge_id]
            for edge_id in edge_ids
            if edge_id in graph.edges
        )

        return GraphQueryResult(query=query, nodes=nodes, edges=edges)

    def _query_focused_context(
        self,
        *,
        graph: WorkspaceCognitiveGraph,
        query: GraphQuery,
    ) -> GraphQueryResult:
        if graph.focused_node_id is None:
            return GraphQueryResult(query=query)

        focus_node = graph.nodes.get(graph.focused_node_id)

        if focus_node is None:
            return GraphQueryResult(query=query)

        neighbor_query = query.model_copy(
            update={
                "kind": GraphQueryKind.NEIGHBORS,
                "node_id": focus_node.node_id,
                "direction": GraphQueryDirection.BOTH,
            }
        )
        neighbors = self._query_neighbors(graph=graph, query=neighbor_query)

        return GraphQueryResult(
            query=query,
            nodes=(focus_node, *neighbors.nodes)[: query.limit],
            edges=neighbors.edges[: query.limit],
        )

    @staticmethod
    def _matching_edges(
        *,
        graph: WorkspaceCognitiveGraph,
        query: GraphQuery,
    ) -> tuple[GraphEdge, ...]:
        if query.node_id is None:
            return ()

        edges: list[GraphEdge] = []

        for edge in graph.edges.values():
            if not query.include_inactive and not edge.active:
                continue

            if query.edge_kind is not None and edge.kind != query.edge_kind:
                continue

            if query.direction == GraphQueryDirection.OUTGOING:
                if edge.source_node_id == query.node_id:
                    edges.append(edge)
            elif query.direction == GraphQueryDirection.INCOMING:
                if edge.target_node_id == query.node_id:
                    edges.append(edge)
            else:
                if query.node_id in {edge.source_node_id, edge.target_node_id}:
                    edges.append(edge)

        return tuple(edges)

    @staticmethod
    def _event(
        *,
        kind: WorkspaceGraphEventKind,
        reason: WorkspaceGraphReason,
        session_id: str | None = None,
        node_id: str | None = None,
        edge_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceGraphRuntimeEvent:
        return WorkspaceGraphRuntimeEvent(
            kind=kind,
            reason=reason,
            session_id=session_id,
            node_id=node_id,
            edge_id=edge_id,
            metadata=metadata or {},
        )


def _neighbor_ids(
    *,
    node_id: str,
    edges: tuple[GraphEdge, ...],
    direction: GraphQueryDirection,
) -> tuple[str, ...]:
    ids: list[str] = []

    for edge in edges:
        if direction == GraphQueryDirection.OUTGOING:
            ids.append(edge.target_node_id)
        elif direction == GraphQueryDirection.INCOMING:
            ids.append(edge.source_node_id)
        else:
            if edge.source_node_id == node_id:
                ids.append(edge.target_node_id)
            else:
                ids.append(edge.source_node_id)

    return tuple(dict.fromkeys(ids))


def graph_trust(reason: str = "workspace graph") -> TrustCalibration:
    return TrustCalibration(
        confidence=0.95,
        stability=0.95,
        ambiguity=0.0,
        source=EnvironmentSource.OS_OBSERVER,
        reason=reason,
    )


def graph_node(
    *,
    kind: GraphNodeKind,
    label: str,
    node_id: str | None = None,
    region: ScreenRegion | None = None,
    metadata: dict[str, Any] | None = None,
) -> GraphNode:
    return GraphNode(
        node_id=node_id or f"graph_node_{uuid4().hex}",
        kind=kind,
        label=label,
        region=region,
        trust=graph_trust(f"{kind.value} node"),
        metadata=metadata or {},
    )


def graph_edge(
    *,
    kind: GraphEdgeKind,
    source_node_id: str,
    target_node_id: str,
    edge_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> GraphEdge:
    return GraphEdge(
        edge_id=edge_id or f"graph_edge_{uuid4().hex}",
        kind=kind,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        trust=graph_trust(f"{kind.value} edge"),
        metadata=metadata or {},
    )


def _clean_required(value: str) -> str:
    cleaned = value.strip()

    if not cleaned:
        raise ValueError("field cannot be empty.")

    return cleaned