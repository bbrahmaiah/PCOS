from __future__ import annotations

from enum import StrEnum
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.environment.environment_fusion import FusedContext, FusionStatus
from jarvis.environment.models import EnvironmentSource, TrustCalibration
from jarvis.environment.trust_runtime import TrustPolicyClassification
from jarvis.environment.ui_semantics import SemanticSceneKind, UIContext
from jarvis.environment.workspace_graph import GraphNodeKind, WorkspaceCognitiveGraph
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class VisualContextFragmentKind(StrEnum):
    ENVIRONMENT_SUMMARY = "environment_summary"
    SEMANTIC_SCENE = "semantic_scene"
    WORKSPACE_GRAPH = "workspace_graph"
    FUSED_CONTEXT = "fused_context"
    ACTIVE_ERRORS = "active_errors"
    ACTIVE_TARGET = "active_target"
    MEMORY_HINT = "memory_hint"
    INTENT_HINT = "intent_hint"


class VisualContextFreshness(StrEnum):
    HOT = "hot"
    WARM = "warm"
    STALE = "stale"
    EXPIRED = "expired"


class VisualContextStreamStatus(StrEnum):
    READY = "ready"
    PARTIAL = "partial"
    STALE = "stale"
    BLOCKED = "blocked"
    EMPTY = "empty"


class VisualContextPrefetchReason(StrEnum):
    SESSION_CREATED = "session_created"
    PREFETCH_REQUESTED = "prefetch_requested"
    ENVIRONMENT_SNAPSHOT_CACHED = "environment_snapshot_cached"
    SEMANTIC_SCENE_CACHED = "semantic_scene_cached"
    WORKSPACE_GRAPH_CACHED = "workspace_graph_cached"
    FRAGMENT_CACHED = "fragment_cached"
    FUSED_CONTEXT_STREAMED = "fused_context_streamed"
    CACHE_MISS = "cache_miss"
    CACHE_EXPIRED = "cache_expired"
    SESSION_NOT_FOUND = "session_not_found"
    RUNTIME_RESET = "runtime_reset"


class VisualContextEventKind(StrEnum):
    SESSION_CREATED = "session_created"
    PREFETCH_COMPLETED = "prefetch_completed"
    STREAM_COMPLETED = "stream_completed"
    CACHE_MUTATED = "cache_mutated"
    RUNTIME_RESET = "runtime_reset"


class VisualContextCachePolicy(OrchestrationModel):
    """
    Cache policy for Phase 7-compatible visual context.

    All values are intentionally small. This is a hot context cache, not
    long-term memory.
    """

    hot_ttl_turns: int = Field(default=3, ge=1, le=20)
    warm_ttl_turns: int = Field(default=8, ge=1, le=50)
    max_fragments: int = Field(default=64, ge=1, le=512)
    max_graphs: int = Field(default=8, ge=1, le=64)
    max_scenes: int = Field(default=16, ge=1, le=128)
    max_snapshots: int = Field(default=16, ge=1, le=128)
    max_stream_fragments: int = Field(default=8, ge=1, le=64)
    stale_after_turns: int = Field(default=6, ge=1, le=100)

    @model_validator(mode="after")
    def _ttl_order(self) -> VisualContextCachePolicy:
        if self.warm_ttl_turns < self.hot_ttl_turns:
            raise ValueError("warm_ttl_turns must be >= hot_ttl_turns.")

        return self


class VisualContextCacheEntry(OrchestrationModel):
    """
    Generic cache entry.

    Values are deliberately object-typed so individual caches can store
    snapshots, scenes, graphs, fragments, or fused contexts without coupling.
    """

    entry_id: str = Field(default_factory=lambda: f"vcache_{uuid4().hex}")
    key: str
    value: object
    freshness: VisualContextFreshness
    confidence: float = Field(default=0.80, ge=0.0, le=1.0)
    policy: TrustPolicyClassification = TrustPolicyClassification.REVIEW
    created_turn: int = Field(default=0, ge=0)
    last_access_turn: int = Field(default=0, ge=0)
    ttl_turns: int = Field(default=3, ge=1)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("entry_id", "key")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    def expired(self, *, current_turn: int) -> bool:
        return current_turn - self.created_turn >= self.ttl_turns

    def stale(self, *, current_turn: int, stale_after_turns: int) -> bool:
        return current_turn - self.last_access_turn >= stale_after_turns


class VisualContextFragment(OrchestrationModel):
    """
    Small streamable visual context fragment.

    This is what cognition should receive incrementally instead of waiting for
    a giant environment context build.
    """

    fragment_id: str = Field(default_factory=lambda: f"vfrag_{uuid4().hex}")
    kind: VisualContextFragmentKind
    content: str
    source_key: str
    confidence: float = Field(default=0.80, ge=0.0, le=1.0)
    policy: TrustPolicyClassification = TrustPolicyClassification.REVIEW
    priority: int = Field(default=50, ge=0, le=100)
    trust: TrustCalibration
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("fragment_id", "content", "source_key")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class VisualContextPrefetchRequest(OrchestrationModel):
    """
    Request to precompute hot visual context before cognition asks for it.
    """

    request_id: str = Field(default_factory=lambda: f"vctx_prefetch_{uuid4().hex}")
    session_id: str
    workspace_id: str
    current_turn: int = Field(default=0, ge=0)
    environment_snapshot: object | None = None
    semantic_context: UIContext | None = None
    workspace_graph: WorkspaceCognitiveGraph | None = None
    fused_context: FusedContext | None = None
    memory_hint: str | None = None
    active_intent_hint: str | None = None
    reason: VisualContextPrefetchReason = (
        VisualContextPrefetchReason.PREFETCH_REQUESTED
    )
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("request_id", "session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class VisualContextPrefetchResult(OrchestrationModel):
    """
    Prefetch result.
    """

    result_id: str = Field(
        default_factory=lambda: f"vctx_prefetch_result_{uuid4().hex}"
    )
    request_id: str
    cached_snapshot: bool = False
    cached_semantic_scene: bool = False
    cached_graph: bool = False
    cached_fused_context: bool = False
    fragment_count: int = Field(default=0, ge=0)
    status: VisualContextStreamStatus
    reason: VisualContextPrefetchReason
    message: str
    created_at: object = Field(default_factory=utc_now)

    @field_validator("result_id", "request_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class FusedContextStream(OrchestrationModel):
    """
    Stream-ready context package for cognition.

    This is intentionally small and fragment-based to avoid blocking the LLM.
    """

    stream_id: str = Field(default_factory=lambda: f"fused_stream_{uuid4().hex}")
    session_id: str
    workspace_id: str
    current_turn: int = Field(ge=0)
    status: VisualContextStreamStatus
    fragments: tuple[VisualContextFragment, ...]
    source_keys: tuple[str, ...]
    safe_for_cognition: bool
    safe_for_action_planning: bool
    message: str
    created_at: object = Field(default_factory=utc_now)

    @field_validator("stream_id", "session_id", "workspace_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class VisualContextStreamingSession(OrchestrationModel):
    """
    Visual context streaming runtime session.
    """

    session_id: str = Field(default_factory=lambda: f"vctx_session_{uuid4().hex}")
    workspace_id: str
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class VisualContextStreamingRuntimeEvent(OrchestrationModel):
    """
    Runtime event for visual context streaming.
    """

    event_id: str = Field(default_factory=lambda: f"vctx_event_{uuid4().hex}")
    kind: VisualContextEventKind
    reason: VisualContextPrefetchReason
    session_id: str | None = None
    result_id: str | None = None
    stream_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class VisualContextStreamingRuntimeSnapshot(OrchestrationModel):
    """
    Diagnostics for Phase 8 Step 21.
    """

    name: str
    session_count: int = Field(ge=0)
    snapshot_cache_count: int = Field(ge=0)
    semantic_cache_count: int = Field(ge=0)
    graph_cache_count: int = Field(ge=0)
    fragment_cache_count: int = Field(ge=0)
    stream_count: int = Field(ge=0)
    hot_fragment_count: int = Field(ge=0)
    stale_fragment_count: int = Field(ge=0)
    runtime_event_count: int = Field(ge=0)
    last_reason: VisualContextPrefetchReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        return _clean_required(value)


class _BoundedCache:
    """
    Small bounded cache used by all visual context caches.
    """

    def __init__(self, *, max_items: int) -> None:
        self._max_items = max_items
        self._items: dict[str, VisualContextCacheEntry] = {}

    def put(self, entry: VisualContextCacheEntry) -> None:
        self._items[entry.key] = entry
        self._trim()

    def get(
        self,
        key: str,
        *,
        current_turn: int,
        stale_after_turns: int,
    ) -> VisualContextCacheEntry | None:
        entry = self._items.get(key)

        if entry is None:
            return None

        if entry.expired(current_turn=current_turn):
            self._items.pop(key, None)
            return None

        freshness = (
            VisualContextFreshness.STALE
            if entry.stale(
                current_turn=current_turn,
                stale_after_turns=stale_after_turns,
            )
            else entry.freshness
        )
        updated = entry.model_copy(
            update={
                "last_access_turn": current_turn,
                "freshness": freshness,
            }
        )
        self._items[key] = updated

        return updated

    def values(self) -> tuple[VisualContextCacheEntry, ...]:
        return tuple(self._items.values())

    def clear(self) -> None:
        self._items.clear()

    def count(self) -> int:
        return len(self._items)

    def _trim(self) -> None:
        if len(self._items) <= self._max_items:
            return

        ordered = sorted(
            self._items.values(),
            key=lambda item: (item.last_access_turn, str(item.created_at)),
        )

        for item in ordered[: len(self._items) - self._max_items]:
            self._items.pop(item.key, None)


class EnvironmentSnapshotCache:
    """
    Hot cache for environment snapshots.
    """

    def __init__(self, *, policy: VisualContextCachePolicy) -> None:
        self._policy = policy
        self._cache = _BoundedCache(max_items=policy.max_snapshots)

    def put(
        self,
        *,
        workspace_id: str,
        snapshot: object,
        current_turn: int,
    ) -> str:
        key = f"snapshot:{workspace_id}"
        self._cache.put(
            VisualContextCacheEntry(
                key=key,
                value=snapshot,
                freshness=VisualContextFreshness.HOT,
                confidence=0.85,
                created_turn=current_turn,
                last_access_turn=current_turn,
                ttl_turns=self._policy.warm_ttl_turns,
            )
        )

        return key

    def get(
        self,
        *,
        workspace_id: str,
        current_turn: int,
    ) -> VisualContextCacheEntry | None:
        return self._cache.get(
            f"snapshot:{workspace_id}",
            current_turn=current_turn,
            stale_after_turns=self._policy.stale_after_turns,
        )

    def count(self) -> int:
        return self._cache.count()

    def clear(self) -> None:
        self._cache.clear()


class SemanticSceneCache:
    """
    Hot cache for semantic scenes.
    """

    def __init__(self, *, policy: VisualContextCachePolicy) -> None:
        self._policy = policy
        self._cache = _BoundedCache(max_items=policy.max_scenes)

    def put(
        self,
        *,
        workspace_id: str,
        context: UIContext,
        current_turn: int,
    ) -> str:
        key = f"semantic:{workspace_id}:{context.scene.kind.value}"
        self._cache.put(
            VisualContextCacheEntry(
                key=key,
                value=context,
                freshness=VisualContextFreshness.HOT,
                confidence=context.scene.confidence,
                policy=context.policy_classification,
                created_turn=current_turn,
                last_access_turn=current_turn,
                ttl_turns=self._policy.hot_ttl_turns,
            )
        )

        return key

    def get(
        self,
        *,
        workspace_id: str,
        scene: SemanticSceneKind,
        current_turn: int,
    ) -> VisualContextCacheEntry | None:
        return self._cache.get(
            f"semantic:{workspace_id}:{scene.value}",
            current_turn=current_turn,
            stale_after_turns=self._policy.stale_after_turns,
        )

    def count(self) -> int:
        return self._cache.count()

    def clear(self) -> None:
        self._cache.clear()


class GraphHotCache:
    """
    Hot cache for active workspace graph.
    """

    def __init__(self, *, policy: VisualContextCachePolicy) -> None:
        self._policy = policy
        self._cache = _BoundedCache(max_items=policy.max_graphs)

    def put(
        self,
        *,
        workspace_id: str,
        graph: WorkspaceCognitiveGraph,
        current_turn: int,
    ) -> str:
        key = f"graph:{workspace_id}"
        self._cache.put(
            VisualContextCacheEntry(
                key=key,
                value=graph,
                freshness=VisualContextFreshness.HOT,
                confidence=0.88,
                created_turn=current_turn,
                last_access_turn=current_turn,
                ttl_turns=self._policy.hot_ttl_turns,
                metadata={
                    "node_count": len(graph.nodes),
                    "edge_count": len(graph.edges),
                },
            )
        )

        return key

    def get(
        self,
        *,
        workspace_id: str,
        current_turn: int,
    ) -> VisualContextCacheEntry | None:
        return self._cache.get(
            f"graph:{workspace_id}",
            current_turn=current_turn,
            stale_after_turns=self._policy.stale_after_turns,
        )

    def count(self) -> int:
        return self._cache.count()

    def clear(self) -> None:
        self._cache.clear()


class VisualContextFragmentCache:
    """
    Hot cache for streamable visual context fragments.
    """

    def __init__(self, *, policy: VisualContextCachePolicy) -> None:
        self._policy = policy
        self._cache = _BoundedCache(max_items=policy.max_fragments)

    def put(
        self,
        *,
        workspace_id: str,
        fragment: VisualContextFragment,
        current_turn: int,
    ) -> str:
        key = f"fragment:{workspace_id}:{fragment.kind.value}:{fragment.source_key}"
        self._cache.put(
            VisualContextCacheEntry(
                key=key,
                value=fragment,
                freshness=VisualContextFreshness.HOT,
                confidence=fragment.confidence,
                policy=fragment.policy,
                created_turn=current_turn,
                last_access_turn=current_turn,
                ttl_turns=self._policy.hot_ttl_turns,
            )
        )

        return key

    def fragments_for(
        self,
        *,
        workspace_id: str,
        current_turn: int,
    ) -> tuple[VisualContextFragment, ...]:
        prefix = f"fragment:{workspace_id}:"
        fragments: list[VisualContextFragment] = []

        for entry in self._cache.values():
            if not entry.key.startswith(prefix):
                continue

            cached = self._cache.get(
                entry.key,
                current_turn=current_turn,
                stale_after_turns=self._policy.stale_after_turns,
            )

            if cached is None:
                continue

            if isinstance(cached.value, VisualContextFragment):
                fragments.append(cached.value)

        return tuple(
            sorted(fragments, key=lambda item: item.priority, reverse=True)
        )

    def count(self) -> int:
        return self._cache.count()

    def hot_count(self) -> int:
        return sum(
            1
            for entry in self._cache.values()
            if entry.freshness == VisualContextFreshness.HOT
        )

    def stale_count(self) -> int:
        return sum(
            1
            for entry in self._cache.values()
            if entry.freshness == VisualContextFreshness.STALE
        )

    def clear(self) -> None:
        self._cache.clear()


class VisualContextFragmentBuilder:
    """
    Builds streamable fragments from Phase 8 context.
    """

    def build(
        self,
        request: VisualContextPrefetchRequest
    ) -> tuple[VisualContextFragment, ...]:
        fragments: list[VisualContextFragment] = []

        if request.semantic_context is not None:
            fragments.append(_semantic_fragment(request.semantic_context))

        if request.workspace_graph is not None:
            fragments.extend(_graph_fragments(request.workspace_graph))

        if request.fused_context is not None:
            fragments.append(_fused_fragment(request.fused_context))

        if request.memory_hint:
            fragments.append(
                _fragment(
                    kind=VisualContextFragmentKind.MEMORY_HINT,
                    content=request.memory_hint,
                    source_key="memory",
                    confidence=0.80,
                    policy=TrustPolicyClassification.SAFE,
                    priority=55,
                )
            )

        if request.active_intent_hint:
            fragments.append(
                _fragment(
                    kind=VisualContextFragmentKind.INTENT_HINT,
                    content=request.active_intent_hint,
                    source_key="intent",
                    confidence=0.82,
                    policy=TrustPolicyClassification.REVIEW,
                    priority=65,
                )
            )

        return tuple(fragments)


class FusedContextStreamer:
    """
    Streams cached visual context fragments into cognition.

    It never rebuilds heavy perception paths. It only streams hot fragments.
    """

    def __init__(
        self,
        *,
        policy: VisualContextCachePolicy,
        fragment_cache: VisualContextFragmentCache,
    ) -> None:
        self._policy = policy
        self._fragment_cache = fragment_cache

    def stream(
        self,
        *,
        session_id: str,
        workspace_id: str,
        current_turn: int,
    ) -> FusedContextStream:
        fragments = self._fragment_cache.fragments_for(
            workspace_id=workspace_id,
            current_turn=current_turn,
        )
        selected = fragments[: self._policy.max_stream_fragments]

        if not selected:
            return FusedContextStream(
                session_id=session_id,
                workspace_id=workspace_id,
                current_turn=current_turn,
                status=VisualContextStreamStatus.EMPTY,
                fragments=(),
                source_keys=(),
                safe_for_cognition=True,
                safe_for_action_planning=False,
                message="no hot visual context fragments available",
            )

        blocked = any(
            fragment.policy == TrustPolicyClassification.BLOCKED
            for fragment in selected
        )
        verify = any(
            fragment.policy
            in {
                TrustPolicyClassification.REVIEW,
                TrustPolicyClassification.VERIFY_FIRST,
            }
            for fragment in selected
        )
        status = (
            VisualContextStreamStatus.BLOCKED
            if blocked
            else VisualContextStreamStatus.PARTIAL
            if verify
            else VisualContextStreamStatus.READY
        )

        return FusedContextStream(
            session_id=session_id,
            workspace_id=workspace_id,
            current_turn=current_turn,
            status=status,
            fragments=tuple(selected),
            source_keys=tuple(fragment.source_key for fragment in selected),
            safe_for_cognition=not blocked,
            safe_for_action_planning=status == VisualContextStreamStatus.READY,
            message=f"streamed {len(selected)} visual context fragments",
        )


class VisualContextPrefetchRuntime:
    """
    Phase 8 Step 21 Visual Context Streaming / Phase 7 Integration.

    Responsibilities:
    - precompute screen summary cache
    - cache active workspace graph
    - cache semantic scenes
    - build hot streamable visual fragments
    - stream visual context into cognition without blocking

    Non-responsibilities:
    - no capture
    - no OCR
    - no graph rebuild
    - no LLM call
    - no action execution
    """

    def __init__(
        self,
        *,
        name: str = "visual_context_prefetch_runtime",
        policy: VisualContextCachePolicy | None = None,
        fragment_builder: VisualContextFragmentBuilder | None = None,
    ) -> None:
        cleaned = name.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._policy = policy or VisualContextCachePolicy()
        self._snapshot_cache = EnvironmentSnapshotCache(policy=self._policy)
        self._semantic_cache = SemanticSceneCache(policy=self._policy)
        self._graph_cache = GraphHotCache(policy=self._policy)
        self._fragment_cache = VisualContextFragmentCache(policy=self._policy)
        self._fragment_builder = fragment_builder or VisualContextFragmentBuilder()
        self._streamer = FusedContextStreamer(
            policy=self._policy,
            fragment_cache=self._fragment_cache,
        )
        self._sessions: dict[str, VisualContextStreamingSession] = {}
        self._prefetch_results: list[VisualContextPrefetchResult] = []
        self._streams: list[FusedContextStream] = []
        self._events: list[VisualContextStreamingRuntimeEvent] = []
        self._lock = RLock()
        self._last_reason: VisualContextPrefetchReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def create_session(
        self,
        *,
        workspace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> VisualContextStreamingSession:
        session = VisualContextStreamingSession(
            workspace_id=workspace_id,
            metadata=metadata or {},
        )
        event = self._event(
            kind=VisualContextEventKind.SESSION_CREATED,
            reason=VisualContextPrefetchReason.SESSION_CREATED,
            session_id=session.session_id,
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._events.append(event)
            self._last_reason = event.reason

        return session

    def prefetch(
        self,
        request: VisualContextPrefetchRequest,
    ) -> VisualContextPrefetchResult:
        if self.session_for(request.session_id) is None:
            result = VisualContextPrefetchResult(
                request_id=request.request_id,
                status=VisualContextStreamStatus.BLOCKED,
                reason=VisualContextPrefetchReason.SESSION_NOT_FOUND,
                message="visual context streaming session not found",
            )
            self._record_prefetch(result, request.session_id)
            return result

        cached_snapshot = False
        cached_semantic = False
        cached_graph = False
        cached_fused = False

        if request.environment_snapshot is not None:
            self._snapshot_cache.put(
                workspace_id=request.workspace_id,
                snapshot=request.environment_snapshot,
                current_turn=request.current_turn,
            )
            cached_snapshot = True

        if request.semantic_context is not None:
            self._semantic_cache.put(
                workspace_id=request.workspace_id,
                context=request.semantic_context,
                current_turn=request.current_turn,
            )
            cached_semantic = True

        if request.workspace_graph is not None:
            self._graph_cache.put(
                workspace_id=request.workspace_id,
                graph=request.workspace_graph,
                current_turn=request.current_turn,
            )
            cached_graph = True

        fragments = self._fragment_builder.build(request)

        for fragment in fragments:
            self._fragment_cache.put(
                workspace_id=request.workspace_id,
                fragment=fragment,
                current_turn=request.current_turn,
            )

        cached_fused = request.fused_context is not None

        status = (
            VisualContextStreamStatus.READY
            if fragments or cached_snapshot or cached_semantic or cached_graph
            else VisualContextStreamStatus.EMPTY
        )
        result = VisualContextPrefetchResult(
            request_id=request.request_id,
            cached_snapshot=cached_snapshot,
            cached_semantic_scene=cached_semantic,
            cached_graph=cached_graph,
            cached_fused_context=cached_fused,
            fragment_count=len(fragments),
            status=status,
            reason=VisualContextPrefetchReason.PREFETCH_REQUESTED,
            message=f"prefetched {len(fragments)} visual context fragments",
        )

        self._record_prefetch(result, request.session_id)
        self._touch_session(request.session_id)

        return result

    def stream(
        self,
        *,
        session_id: str,
        workspace_id: str,
        current_turn: int,
    ) -> FusedContextStream:
        if self.session_for(session_id) is None:
            stream = FusedContextStream(
                session_id=session_id,
                workspace_id=workspace_id,
                current_turn=current_turn,
                status=VisualContextStreamStatus.BLOCKED,
                fragments=(),
                source_keys=(),
                safe_for_cognition=False,
                safe_for_action_planning=False,
                message="visual context streaming session not found",
            )
            self._record_stream(
                stream,
                reason=VisualContextPrefetchReason.SESSION_NOT_FOUND,
            )
            return stream

        stream = self._streamer.stream(
            session_id=session_id,
            workspace_id=workspace_id,
            current_turn=current_turn,
        )
        self._record_stream(
            stream,
            reason=VisualContextPrefetchReason.FUSED_CONTEXT_STREAMED,
        )
        self._touch_session(session_id)

        return stream

    def snapshot_entry(
        self,
        *,
        workspace_id: str,
        current_turn: int,
    ) -> VisualContextCacheEntry | None:
        return self._snapshot_cache.get(
            workspace_id=workspace_id,
            current_turn=current_turn,
        )

    def graph_entry(
        self,
        *,
        workspace_id: str,
        current_turn: int,
    ) -> VisualContextCacheEntry | None:
        return self._graph_cache.get(
            workspace_id=workspace_id,
            current_turn=current_turn,
        )

    def semantic_entry(
        self,
        *,
        workspace_id: str,
        scene: SemanticSceneKind,
        current_turn: int,
    ) -> VisualContextCacheEntry | None:
        return self._semantic_cache.get(
            workspace_id=workspace_id,
            scene=scene,
            current_turn=current_turn,
        )

    def session_for(
        self,
        session_id: str,
    ) -> VisualContextStreamingSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def prefetch_results(self) -> tuple[VisualContextPrefetchResult, ...]:
        with self._lock:
            return tuple(self._prefetch_results)

    def streams(self) -> tuple[FusedContextStream, ...]:
        with self._lock:
            return tuple(self._streams)

    def events(self) -> tuple[VisualContextStreamingRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def snapshot(self) -> VisualContextStreamingRuntimeSnapshot:
        with self._lock:
            return VisualContextStreamingRuntimeSnapshot(
                name=self.name,
                session_count=len(self._sessions),
                snapshot_cache_count=self._snapshot_cache.count(),
                semantic_cache_count=self._semantic_cache.count(),
                graph_cache_count=self._graph_cache.count(),
                fragment_cache_count=self._fragment_cache.count(),
                stream_count=len(self._streams),
                hot_fragment_count=self._fragment_cache.hot_count(),
                stale_fragment_count=self._fragment_cache.stale_count(),
                runtime_event_count=len(self._events),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        event = self._event(
            kind=VisualContextEventKind.RUNTIME_RESET,
            reason=VisualContextPrefetchReason.RUNTIME_RESET,
        )

        with self._lock:
            self._sessions.clear()
            self._prefetch_results.clear()
            self._streams.clear()
            self._events.clear()
            self._snapshot_cache.clear()
            self._semantic_cache.clear()
            self._graph_cache.clear()
            self._fragment_cache.clear()
            self._events.append(event)
            self._last_reason = event.reason

    def _record_prefetch(
        self,
        result: VisualContextPrefetchResult,
        session_id: str,
    ) -> None:
        event = self._event(
            kind=VisualContextEventKind.PREFETCH_COMPLETED,
            reason=result.reason,
            session_id=session_id,
            result_id=result.result_id,
            metadata={"status": result.status.value},
        )

        with self._lock:
            self._prefetch_results.append(result)
            self._events.append(event)
            self._last_reason = result.reason

    def _record_stream(
        self,
        stream: FusedContextStream,
        *,
        reason: VisualContextPrefetchReason,
    ) -> None:
        event = self._event(
            kind=VisualContextEventKind.STREAM_COMPLETED,
            reason=reason,
            session_id=stream.session_id,
            stream_id=stream.stream_id,
            metadata={"status": stream.status.value},
        )

        with self._lock:
            self._streams.append(stream)
            self._events.append(event)
            self._last_reason = reason

    def _touch_session(self, session_id: str) -> None:
        session = self._sessions.get(session_id)

        if session is None:
            return

        self._sessions[session_id] = session.model_copy(
            update={"updated_at": utc_now()}
        )

    @staticmethod
    def _event(
        *,
        kind: VisualContextEventKind,
        reason: VisualContextPrefetchReason,
        session_id: str | None = None,
        result_id: str | None = None,
        stream_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VisualContextStreamingRuntimeEvent:
        return VisualContextStreamingRuntimeEvent(
            kind=kind,
            reason=reason,
            session_id=session_id,
            result_id=result_id,
            stream_id=stream_id,
            metadata=metadata or {},
        )


def _semantic_fragment(context: UIContext) -> VisualContextFragment:
    return _fragment(
        kind=VisualContextFragmentKind.SEMANTIC_SCENE,
        content=f"semantic scene: {context.scene.kind.value}",
        source_key=context.context_id,
        confidence=context.scene.confidence,
        policy=context.policy_classification,
        priority=80,
    )


def _graph_fragments(
    graph: WorkspaceCognitiveGraph
) -> tuple[VisualContextFragment, ...]:
    fragments = [
        _fragment(
            kind=VisualContextFragmentKind.WORKSPACE_GRAPH,
            content=(
                f"workspace graph nodes={len(graph.nodes)} "
                f"edges={len(graph.edges)}"
            ),
            source_key=graph.graph_id,
            confidence=0.88,
            policy=TrustPolicyClassification.SAFE,
            priority=70,
        )
    ]
    errors = [
        node.label
        for node in graph.nodes.values()
        if node.active and node.kind == GraphNodeKind.ERROR
    ]

    if errors:
        fragments.append(
            _fragment(
                kind=VisualContextFragmentKind.ACTIVE_ERRORS,
                content=f"visible errors: {', '.join(errors)}",
                source_key=f"{graph.graph_id}:errors",
                confidence=0.90,
                policy=TrustPolicyClassification.VERIFY_FIRST,
                priority=95,
            )
        )

    return tuple(fragments)


def _fused_fragment(context: FusedContext) -> VisualContextFragment:
    return _fragment(
        kind=VisualContextFragmentKind.FUSED_CONTEXT,
        content=context.bridge.fused_summary,
        source_key=context.context_id,
        confidence=context.trust.confidence,
        policy=context.policy,
        priority=100 if context.status == FusionStatus.FUSED else 75,
    )


def _fragment(
    *,
    kind: VisualContextFragmentKind,
    content: str,
    source_key: str,
    confidence: float,
    policy: TrustPolicyClassification,
    priority: int,
) -> VisualContextFragment:
    return VisualContextFragment(
        kind=kind,
        content=content,
        source_key=source_key,
        confidence=confidence,
        policy=policy,
        priority=priority,
        trust=TrustCalibration(
            confidence=confidence,
            stability=max(0.0, min(1.0, confidence + 0.04)),
            ambiguity=1.0 - confidence,
            source=EnvironmentSource.OS_OBSERVER,
            reason=f"visual context fragment: {kind.value}",
        ),
    )


def _clean_required(value: str) -> str:
    cleaned = value.strip()

    if not cleaned:
        raise ValueError("field cannot be empty.")

    return cleaned