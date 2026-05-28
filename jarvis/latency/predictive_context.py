from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from uuid import uuid4

from pydantic import Field, field_validator

from jarvis.latency.models import LatencyOperation, LatencySubsystem
from jarvis.latency.profiler import (
    PipelineLatencyProfiler,
    PipelineProfilerConfig,
    PipelineProfilerReport,
    PipelineStage,
)
from jarvis.latency.speculative_execution import (
    SpeculativeBranch,
)
from jarvis.latency.streaming_memory import (
    MemoryResultSource,
    MemoryStreamKind,
    MemoryStreamResult,
)
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class PredictiveContextStatus(StrEnum):
    """
    Predictive context lifecycle status.
    """

    CREATED = "created"
    BUILDING = "building"
    READY = "ready"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ContextFragmentKind(StrEnum):
    """
    Context fragment kind.
    """

    USER_PROFILE = "user_profile"
    WORKSPACE = "workspace"
    RECENT_TURN = "recent_turn"
    PREVIOUS_TURN = "previous_turn"
    EPISODIC_MEMORY = "episodic_memory"
    SEMANTIC_MEMORY = "semantic_memory"
    SPECULATIVE_BRANCH = "speculative_branch"
    SUMMARY = "summary"
    SYSTEM = "system"


class ContextBuildStrategy(StrEnum):
    """
    Context building strategy.
    """

    INCREMENTAL_EXTENSION = "incremental_extension"
    FULL_REBUILD = "full_rebuild"
    STREAMING_ASSEMBLY = "streaming_assembly"
    CACHE_HIT = "cache_hit"


class ContextCompressionPolicy(StrEnum):
    """
    Compression policy.

    Current turn and previous turn must never be compressed.
    """

    NEVER = "never"
    BACKGROUND_ELIGIBLE = "background_eligible"
    COMPRESSED = "compressed"


class PredictiveContextEventKind(StrEnum):
    """
    Predictive context event kind.
    """

    SESSION_CREATED = "session_created"
    SESSION_STARTED = "session_started"
    FRAGMENT_ADDED = "fragment_added"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    MEMORY_RESULT_CONSUMED = "memory_result_consumed"
    CONTEXT_READY = "context_ready"
    SEMANTIC_APPENDED = "semantic_appended"
    FRAGMENT_COMPRESSED = "fragment_compressed"
    CACHE_INVALIDATED = "cache_invalidated"
    SESSION_COMPLETED = "session_completed"
    SESSION_CANCELLED = "session_cancelled"
    SESSION_FAILED = "session_failed"


class PredictiveContextReason(StrEnum):
    """
    Machine-readable predictive context reasons.
    """

    SESSION_CREATED = "session_created"
    SESSION_STARTED = "session_started"
    FRAGMENT_ADDED = "fragment_added"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    MEMORY_RESULT_CONSUMED = "memory_result_consumed"
    PROFILE_EPISODIC_READY = "profile_episodic_ready"
    CONTEXT_READY = "context_ready"
    SEMANTIC_APPENDED = "semantic_appended"
    FRAGMENT_COMPRESSED = "fragment_compressed"
    MEMORY_WRITE_INVALIDATED_CACHE = "memory_write_invalidated_cache"
    SESSION_COMPLETED = "session_completed"
    SESSION_CANCELLED = "session_cancelled"
    SESSION_FAILED = "session_failed"
    SESSION_NOT_FOUND = "session_not_found"
    SESSION_NOT_ACTIVE = "session_not_active"
    EMPTY_CONTEXT = "empty_context"
    RUNTIME_RESET = "runtime_reset"


class ContextFragment(OrchestrationModel):
    """
    Reusable context fragment.

    Fragments are the unit of incremental context construction.
    """

    fragment_id: str = Field(default_factory=lambda: uuid4().hex)
    kind: ContextFragmentKind
    text: str
    priority: int = Field(ge=0)
    token_estimate: int = Field(ge=0)
    source_id: str | None = None
    compression_policy: ContextCompressionPolicy = ContextCompressionPolicy.NEVER
    created_at_ns: int = Field(default_factory=time.perf_counter_ns, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("fragment_id", "text")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class ContextSnapshot(OrchestrationModel):
    """
    Immutable context snapshot.

    Each new turn should extend this snapshot rather than rebuilding everything.
    """

    snapshot_id: str = Field(default_factory=lambda: uuid4().hex)
    turn_id: str
    fragments: tuple[ContextFragment, ...] = ()
    summary: str | None = None
    token_estimate: int = Field(default=0, ge=0)
    context_confidence: float = Field(default=0.0, ge=0, le=1)
    created_at_ns: int = Field(default_factory=time.perf_counter_ns, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("snapshot_id", "turn_id")
    @classmethod
    def _required_ids(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class ContextBuildRequest(OrchestrationModel):
    """
    Request to build predictive context.
    """

    request_id: str = Field(default_factory=lambda: uuid4().hex)
    turn_id: str
    user_text: str
    previous_snapshot: ContextSnapshot | None = None
    speculative_branch: SpeculativeBranch | None = None
    max_tokens: int = Field(default=4096, gt=0)
    created_at_ns: int = Field(default_factory=time.perf_counter_ns, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("request_id", "turn_id", "user_text")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class PredictiveContextEvent(OrchestrationModel):
    """
    Predictive context event.
    """

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    kind: PredictiveContextEventKind
    reason: PredictiveContextReason
    fragment_id: str | None = None
    latency_ms: float | None = None
    confidence: float | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("event_id", "session_id")
    @classmethod
    def _required_ids(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class PredictiveContextSessionState(OrchestrationModel):
    """
    Runtime state for one predictive context build session.
    """

    session_id: str = Field(default_factory=lambda: uuid4().hex)
    trace_id: str = Field(default_factory=lambda: uuid4().hex)
    request: ContextBuildRequest
    status: PredictiveContextStatus = PredictiveContextStatus.CREATED
    strategy: ContextBuildStrategy = ContextBuildStrategy.INCREMENTAL_EXTENSION
    started_at_ns: int | None = None
    ready_at_ns: int | None = None
    completed_at_ns: int | None = None
    cancelled_at_ns: int | None = None
    failed_at_ns: int | None = None
    fragment_count: int = Field(default=0, ge=0)
    memory_result_count: int = Field(default=0, ge=0)
    cache_hit_count: int = Field(default=0, ge=0)
    cache_miss_count: int = Field(default=0, ge=0)
    context_confidence: float = Field(default=0.0, ge=0, le=1)
    ready_for_llm: bool = False
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("session_id", "trace_id")
    @classmethod
    def _required_ids(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    def ready_latency_ms(self) -> float | None:
        if self.started_at_ns is None or self.ready_at_ns is None:
            return None

        return (self.ready_at_ns - self.started_at_ns) / 1_000_000.0

    def total_latency_ms(self) -> float | None:
        if self.started_at_ns is None or self.completed_at_ns is None:
            return None

        return (self.completed_at_ns - self.started_at_ns) / 1_000_000.0


class PredictiveContextResult(OrchestrationModel):
    """
    Result from predictive context runtime operation.
    """

    success: bool
    reason: PredictiveContextReason
    session_id: str
    status: PredictiveContextStatus
    fragment: ContextFragment | None = None
    snapshot: ContextSnapshot | None = None
    event: PredictiveContextEvent | None = None
    state: PredictiveContextSessionState | None = None
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("session_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class PredictiveContextReport(OrchestrationModel):
    """
    Final report for predictive context building.
    """

    session_id: str
    trace_id: str
    turn_id: str
    status: PredictiveContextStatus
    strategy: ContextBuildStrategy
    ready_for_llm: bool
    context_confidence: float = Field(ge=0, le=1)
    fragment_count: int = Field(ge=0)
    memory_result_count: int = Field(ge=0)
    cache_hit_count: int = Field(ge=0)
    cache_miss_count: int = Field(ge=0)
    ready_latency_ms: float | None = None
    total_latency_ms: float | None = None
    snapshot: ContextSnapshot
    events: tuple[PredictiveContextEvent, ...]
    profiler_report: PipelineProfilerReport | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("session_id", "trace_id", "turn_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class PredictiveContextRuntimeSnapshot(OrchestrationModel):
    """
    Runtime diagnostics for Phase 7 Step 10.
    """

    name: str
    session_count: int = Field(ge=0)
    building_count: int = Field(ge=0)
    ready_count: int = Field(ge=0)
    completed_count: int = Field(ge=0)
    cancelled_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    fragment_count: int = Field(ge=0)
    report_count: int = Field(ge=0)
    cache_size: int = Field(ge=0)
    last_reason: PredictiveContextReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class ContextFragmentCacheConfig:
    """
    LRU cache for reusable context fragments.
    """

    max_entries: int = 20

    def validate(self) -> None:
        if self.max_entries < 1:
            raise ValueError("max_entries must be positive.")


class ContextFragmentCache:
    """
    LRU fragment cache.

    Cache hit target: <10ms.
    Invalidated on memory writes.
    """

    def __init__(self, *, config: ContextFragmentCacheConfig | None = None) -> None:
        self._config = config or ContextFragmentCacheConfig()
        self._config.validate()
        self._items: OrderedDict[str, ContextFragment] = OrderedDict()
        self._lock = RLock()

    def get(self, key: str) -> ContextFragment | None:
        cache_key = self._key(key)

        with self._lock:
            item = self._items.get(cache_key)

            if item is None:
                return None

            self._items.move_to_end(cache_key)
            return item

    def set(self, key: str, fragment: ContextFragment) -> None:
        cache_key = self._key(key)

        with self._lock:
            self._items[cache_key] = fragment
            self._items.move_to_end(cache_key)

            while len(self._items) > self._config.max_entries:
                self._items.popitem(last=False)

    def invalidate_all(self) -> None:
        with self._lock:
            self._items.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._items)

    @staticmethod
    def _key(value: str) -> str:
        return value.strip().lower()


@dataclass(frozen=True, slots=True)
class PredictiveContextRuntimeConfig:
    """
    Predictive context builder configuration.
    """

    name: str = "predictive_context_builder"
    ready_confidence_threshold: float = 0.80
    target_delta_build_ms: float = 40.0
    target_cache_hit_ms: float = 10.0
    max_previous_turns_uncompressed: int = 2
    profile_predictive_context: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if not 0 <= self.ready_confidence_threshold <= 1:
            raise ValueError("ready_confidence_threshold must be within 0..1.")

        if self.target_delta_build_ms <= 0:
            raise ValueError("target_delta_build_ms must be positive.")

        if self.target_cache_hit_ms <= 0:
            raise ValueError("target_cache_hit_ms must be positive.")

        if self.max_previous_turns_uncompressed < 2:
            raise ValueError("must keep current and previous turn uncompressed.")


class PredictiveContextBuilderRuntime:
    """
    Phase 7 Step 10 Predictive Context Builder.

    Responsibilities:
    - build context incrementally from previous snapshot + new turn
    - consume memory results as they stream in
    - cache reusable fragments aggressively
    - compress older context fragments safely
    - mark context ready when profile + episodic confidence is sufficient
    - append semantic results if they arrive in time
    - never mutate memory or execute tools/actions

    Non-responsibilities:
    - no LLM call execution
    - no memory writes
    - no tool execution
    - no compression of current/previous turn
    """

    def __init__(
        self,
        *,
        config: PredictiveContextRuntimeConfig | None = None,
        cache: ContextFragmentCache | None = None,
        profiler: PipelineLatencyProfiler | None = None,
    ) -> None:
        self._config = config or PredictiveContextRuntimeConfig()
        self._config.validate()

        self._cache = cache or ContextFragmentCache()
        self._profiler = profiler or PipelineLatencyProfiler(
            config=PipelineProfilerConfig(record_to_latency_runtime=True)
        )
        self._states: dict[str, PredictiveContextSessionState] = {}
        self._fragments: dict[str, list[ContextFragment]] = {}
        self._events: dict[str, list[PredictiveContextEvent]] = {}
        self._reports: list[PredictiveContextReport] = []
        self._lock = RLock()
        self._last_reason: PredictiveContextReason | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def create_session(
        self,
        *,
        request: ContextBuildRequest,
        trace_id: str | None = None,
    ) -> PredictiveContextSessionState:
        strategy = (
            ContextBuildStrategy.INCREMENTAL_EXTENSION
            if request.previous_snapshot is not None
            else ContextBuildStrategy.FULL_REBUILD
        )
        state = PredictiveContextSessionState(
            trace_id=trace_id or uuid4().hex,
            request=request,
            strategy=strategy,
        )
        event = self._event(
            session_id=state.session_id,
            kind=PredictiveContextEventKind.SESSION_CREATED,
            reason=PredictiveContextReason.SESSION_CREATED,
        )

        with self._lock:
            self._states[state.session_id] = state
            self._fragments[state.session_id] = []
            self._events[state.session_id] = [event]
            self._last_reason = PredictiveContextReason.SESSION_CREATED

        self._profiler.start_trace(name="predictive_context", trace_id=state.trace_id)

        return state

    def start_session(self, session_id: str) -> PredictiveContextResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            if state.status != PredictiveContextStatus.CREATED:
                return self._failure(
                    session_id=session_id,
                    reason=PredictiveContextReason.SESSION_NOT_ACTIVE,
                    status=state.status,
                    message="context session cannot start from current state",
                    state=state,
                )

            started = state.model_copy(
                update={
                    "status": PredictiveContextStatus.BUILDING,
                    "started_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = started
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=PredictiveContextEventKind.SESSION_STARTED,
                    reason=PredictiveContextReason.SESSION_STARTED,
                )
            )
            self._last_reason = PredictiveContextReason.SESSION_STARTED

        self._seed_initial_fragments(session_id)

        return PredictiveContextResult(
            success=True,
            reason=PredictiveContextReason.SESSION_STARTED,
            session_id=session_id,
            status=PredictiveContextStatus.BUILDING,
            state=self.state_for(session_id),
            message="predictive context session started",
        )

    def consume_memory_result(
        self,
        *,
        session_id: str,
        result: MemoryStreamResult,
    ) -> PredictiveContextResult:
        state = self.state_for(session_id)

        if state is None:
            return self._missing_session(session_id)

        if state.status not in {
            PredictiveContextStatus.BUILDING,
            PredictiveContextStatus.READY,
        }:
            return self._failure(
                session_id=session_id,
                reason=PredictiveContextReason.SESSION_NOT_ACTIVE,
                status=state.status,
                message="context session is not accepting memory results",
                state=state,
            )

        fragment = self._fragment_from_memory(result)
        self._add_fragment(session_id=session_id, fragment=fragment)

        with self._lock:
            current = self._states[session_id]
            confidence = self._context_confidence(self._fragments[session_id])
            ready = self._ready_for_llm(self._fragments[session_id], confidence)
            update: dict[str, object] = {
                "memory_result_count": current.memory_result_count + 1,
                "context_confidence": confidence,
                "fragment_count": len(self._fragments[session_id]),
            }

            if ready and not current.ready_for_llm:
                update["ready_for_llm"] = True
                update["status"] = PredictiveContextStatus.READY
                update["ready_at_ns"] = time.perf_counter_ns()

            updated = current.model_copy(update=update)
            self._states[session_id] = updated

            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=PredictiveContextEventKind.MEMORY_RESULT_CONSUMED,
                    reason=PredictiveContextReason.MEMORY_RESULT_CONSUMED,
                    fragment_id=fragment.fragment_id,
                    confidence=confidence,
                )
            )

            if ready and not current.ready_for_llm:
                self._events[session_id].append(
                    self._event(
                        session_id=session_id,
                        kind=PredictiveContextEventKind.CONTEXT_READY,
                        reason=PredictiveContextReason.PROFILE_EPISODIC_READY,
                        latency_ms=updated.ready_latency_ms(),
                        confidence=confidence,
                    )
                )

            if (
                result.stream_kind == MemoryStreamKind.SEMANTIC
                and current.ready_for_llm
            ):
                self._events[session_id].append(
                    self._event(
                        session_id=session_id,
                        kind=PredictiveContextEventKind.SEMANTIC_APPENDED,
                        reason=PredictiveContextReason.SEMANTIC_APPENDED,
                        fragment_id=fragment.fragment_id,
                        confidence=confidence,
                    )
                )

            self._last_reason = PredictiveContextReason.MEMORY_RESULT_CONSUMED

        return PredictiveContextResult(
            success=True,
            reason=PredictiveContextReason.MEMORY_RESULT_CONSUMED,
            session_id=session_id,
            status=self._states[session_id].status,
            fragment=fragment,
            state=self._states[session_id],
            message="memory result consumed into predictive context",
        )

    def add_fragment(
        self,
        *,
        session_id: str,
        fragment: ContextFragment,
    ) -> PredictiveContextResult:
        state = self.state_for(session_id)

        if state is None:
            return self._missing_session(session_id)

        if state.status not in {
            PredictiveContextStatus.BUILDING,
            PredictiveContextStatus.READY,
        }:
            return self._failure(
                session_id=session_id,
                reason=PredictiveContextReason.SESSION_NOT_ACTIVE,
                status=state.status,
                message="context session is not active",
                state=state,
            )

        self._add_fragment(session_id=session_id, fragment=fragment)

        return PredictiveContextResult(
            success=True,
            reason=PredictiveContextReason.FRAGMENT_ADDED,
            session_id=session_id,
            status=self._states[session_id].status,
            fragment=fragment,
            state=self._states[session_id],
            message="context fragment added",
        )

    def complete_session(self, session_id: str) -> PredictiveContextReport:
        state = self.state_for(session_id)

        if state is None:
            raise ValueError(f"predictive context session not found: {session_id}")

        if state.status not in {
            PredictiveContextStatus.BUILDING,
            PredictiveContextStatus.READY,
        }:
            raise ValueError("context session cannot complete from current state")

        snapshot = self._build_snapshot(session_id)

        with self._lock:
            current = self._states[session_id]
            completed = current.model_copy(
                update={
                    "status": PredictiveContextStatus.COMPLETED,
                    "completed_at_ns": time.perf_counter_ns(),
                    "fragment_count": len(snapshot.fragments),
                    "context_confidence": snapshot.context_confidence,
                }
            )
            self._states[session_id] = completed
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=PredictiveContextEventKind.SESSION_COMPLETED,
                    reason=PredictiveContextReason.SESSION_COMPLETED,
                )
            )
            self._last_reason = PredictiveContextReason.SESSION_COMPLETED

        self._record_context_span(completed)

        profiler_report = None

        if self._config.profile_predictive_context:
            profiler_report = self._profiler.complete_trace(completed.trace_id)

        report = PredictiveContextReport(
            session_id=session_id,
            trace_id=completed.trace_id,
            turn_id=completed.request.turn_id,
            status=completed.status,
            strategy=completed.strategy,
            ready_for_llm=completed.ready_for_llm,
            context_confidence=completed.context_confidence,
            fragment_count=completed.fragment_count,
            memory_result_count=completed.memory_result_count,
            cache_hit_count=completed.cache_hit_count,
            cache_miss_count=completed.cache_miss_count,
            ready_latency_ms=completed.ready_latency_ms(),
            total_latency_ms=completed.total_latency_ms(),
            snapshot=snapshot,
            events=self.events_for(session_id),
            profiler_report=profiler_report,
        )

        with self._lock:
            self._reports.append(report)

        return report

    def compress_older_fragments(
        self,
        *,
        snapshot: ContextSnapshot,
    ) -> ContextSnapshot:
        """
        Compress old context.

        Never compresses current turn or previous turn. This method is safe for
        a Phase 6 background worker to call later.
        """

        fragments = list(snapshot.fragments)

        if len(fragments) <= self._config.max_previous_turns_uncompressed:
            return snapshot

        protected = fragments[-self._config.max_previous_turns_uncompressed :]
        compressible = fragments[: -self._config.max_previous_turns_uncompressed]

        summary_text = " ".join(fragment.text for fragment in compressible)
        summary = ContextFragment(
            kind=ContextFragmentKind.SUMMARY,
            text=self._compress_text(summary_text),
            priority=40,
            token_estimate=max(1, len(summary_text.split()) // 4),
            compression_policy=ContextCompressionPolicy.COMPRESSED,
            metadata={"compressed_fragment_count": len(compressible)},
        )
        new_fragments = (summary, *protected)

        return snapshot.model_copy(
            update={
                "fragments": new_fragments,
                "summary": summary.text,
                "token_estimate": sum(
                    fragment.token_estimate for fragment in new_fragments
                ),
            }
        )

    def invalidate_cache_on_memory_write(self) -> None:
        self._cache.invalidate_all()

        with self._lock:
            for session_id in self._events:
                self._events[session_id].append(
                    self._event(
                        session_id=session_id,
                        kind=PredictiveContextEventKind.CACHE_INVALIDATED,
                        reason=PredictiveContextReason.MEMORY_WRITE_INVALIDATED_CACHE,
                    )
                )

            self._last_reason = PredictiveContextReason.MEMORY_WRITE_INVALIDATED_CACHE

    def cancel_session(
        self,
        session_id: str,
        *,
        reason: str = "cancelled",
    ) -> PredictiveContextResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            if state.status in {
                PredictiveContextStatus.CANCELLED,
                PredictiveContextStatus.COMPLETED,
                PredictiveContextStatus.FAILED,
            }:
                return self._failure(
                    session_id=session_id,
                    reason=PredictiveContextReason.SESSION_NOT_ACTIVE,
                    status=state.status,
                    message="context session already terminal",
                    state=state,
                )

            cancelled = state.model_copy(
                update={
                    "status": PredictiveContextStatus.CANCELLED,
                    "cancelled_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = cancelled
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=PredictiveContextEventKind.SESSION_CANCELLED,
                    reason=PredictiveContextReason.SESSION_CANCELLED,
                    metadata={"cancel_reason": reason},
                )
            )
            self._last_reason = PredictiveContextReason.SESSION_CANCELLED

        return PredictiveContextResult(
            success=True,
            reason=PredictiveContextReason.SESSION_CANCELLED,
            session_id=session_id,
            status=PredictiveContextStatus.CANCELLED,
            state=cancelled,
            message="predictive context session cancelled",
        )

    def fail_session(
        self,
        session_id: str,
        *,
        error: str,
    ) -> PredictiveContextResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            failed = state.model_copy(
                update={
                    "status": PredictiveContextStatus.FAILED,
                    "failed_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = failed
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=PredictiveContextEventKind.SESSION_FAILED,
                    reason=PredictiveContextReason.SESSION_FAILED,
                    metadata={"error": error},
                )
            )
            self._last_reason = PredictiveContextReason.SESSION_FAILED

        return PredictiveContextResult(
            success=True,
            reason=PredictiveContextReason.SESSION_FAILED,
            session_id=session_id,
            status=PredictiveContextStatus.FAILED,
            state=failed,
            message="predictive context session failed",
        )

    def state_for(self, session_id: str) -> PredictiveContextSessionState | None:
        with self._lock:
            return self._states.get(session_id)

    def fragments_for(self, session_id: str) -> tuple[ContextFragment, ...]:
        with self._lock:
            fragments = tuple(self._fragments.get(session_id, ()))

        return tuple(sorted(fragments, key=lambda item: -item.priority))

    def events_for(self, session_id: str) -> tuple[PredictiveContextEvent, ...]:
        with self._lock:
            return tuple(self._events.get(session_id, ()))

    def reports(self) -> tuple[PredictiveContextReport, ...]:
        with self._lock:
            return tuple(self._reports)

    def latest_report(self) -> PredictiveContextReport | None:
        with self._lock:
            if not self._reports:
                return None

            return self._reports[-1]

    def snapshot(self) -> PredictiveContextRuntimeSnapshot:
        with self._lock:
            states = tuple(self._states.values())

            return PredictiveContextRuntimeSnapshot(
                name=self.name,
                session_count=len(states),
                building_count=sum(
                    1
                    for state in states
                    if state.status == PredictiveContextStatus.BUILDING
                ),
                ready_count=sum(
                    1
                    for state in states
                    if state.status == PredictiveContextStatus.READY
                ),
                completed_count=sum(
                    1
                    for state in states
                    if state.status == PredictiveContextStatus.COMPLETED
                ),
                cancelled_count=sum(
                    1
                    for state in states
                    if state.status == PredictiveContextStatus.CANCELLED
                ),
                failed_count=sum(
                    1 for state in states 
                    if state.status == PredictiveContextStatus.FAILED
                ),
                fragment_count=sum(len(items) for items in self._fragments.values()),
                report_count=len(self._reports),
                cache_size=self._cache.size(),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        with self._lock:
            self._states.clear()
            self._fragments.clear()
            self._events.clear()
            self._reports.clear()
            self._last_reason = PredictiveContextReason.RUNTIME_RESET

    def _seed_initial_fragments(self, session_id: str) -> None:
        state = self._states[session_id]
        previous = state.request.previous_snapshot

        if previous is not None:
            for fragment in previous.fragments:
                self._add_fragment(session_id=session_id, fragment=fragment)

        self._add_cached_or_new_fragment(
            session_id=session_id,
            cache_key="user_profile",
            kind=ContextFragmentKind.USER_PROFILE,
            text="User is building a real-time personal cognition JARVIS OS.",
            priority=100,
        )
        self._add_cached_or_new_fragment(
            session_id=session_id,
            cache_key="workspace_info",
            kind=ContextFragmentKind.WORKSPACE,
            text="Current project uses typed runtimes, contracts, and tests.",
            priority=80,
        )

        if state.request.speculative_branch is not None:
            branch = state.request.speculative_branch
            self._add_fragment(
                session_id=session_id,
                fragment=ContextFragment(
                    kind=ContextFragmentKind.SPECULATIVE_BRANCH,
                    text=f"Speculative intent: {branch.candidate.intent.value}",
                    priority=75,
                    token_estimate=6,
                    source_id=branch.branch_id,
                ),
            )

        self._add_fragment(
            session_id=session_id,
            fragment=ContextFragment(
                kind=ContextFragmentKind.RECENT_TURN,
                text=state.request.user_text,
                priority=95,
                token_estimate=max(1, len(state.request.user_text.split())),
            ),
        )

    def _add_cached_or_new_fragment(
        self,
        *,
        session_id: str,
        cache_key: str,
        kind: ContextFragmentKind,
        text: str,
        priority: int,
    ) -> None:
        cached = self._cache.get(cache_key)

        if cached is not None:
            self._add_fragment(session_id=session_id, fragment=cached)

            with self._lock:
                current = self._states[session_id]
                self._states[session_id] = current.model_copy(
                    update={"cache_hit_count": current.cache_hit_count + 1}
                )
                self._events[session_id].append(
                    self._event(
                        session_id=session_id,
                        kind=PredictiveContextEventKind.CACHE_HIT,
                        reason=PredictiveContextReason.CACHE_HIT,
                        fragment_id=cached.fragment_id,
                    )
                )

            return

        fragment = ContextFragment(
            kind=kind,
            text=text,
            priority=priority,
            token_estimate=max(1, len(text.split())),
        )
        self._cache.set(cache_key, fragment)
        self._add_fragment(session_id=session_id, fragment=fragment)

        with self._lock:
            current = self._states[session_id]
            self._states[session_id] = current.model_copy(
                update={"cache_miss_count": current.cache_miss_count + 1}
            )
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=PredictiveContextEventKind.CACHE_MISS,
                    reason=PredictiveContextReason.CACHE_MISS,
                    fragment_id=fragment.fragment_id,
                )
            )

    def _add_fragment(
        self,
        *,
        session_id: str,
        fragment: ContextFragment,
    ) -> None:
        with self._lock:
            self._fragments[session_id].append(fragment)
            current = self._states[session_id]
            updated = current.model_copy(
                update={"fragment_count": len(self._fragments[session_id])}
            )
            self._states[session_id] = updated
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=PredictiveContextEventKind.FRAGMENT_ADDED,
                    reason=PredictiveContextReason.FRAGMENT_ADDED,
                    fragment_id=fragment.fragment_id,
                )
            )
            self._last_reason = PredictiveContextReason.FRAGMENT_ADDED

    @staticmethod
    def _fragment_from_memory(result: MemoryStreamResult) -> ContextFragment:
        kind = ContextFragmentKind.SEMANTIC_MEMORY

        if result.source == MemoryResultSource.USER_PROFILE:
            kind = ContextFragmentKind.USER_PROFILE
        elif result.source == MemoryResultSource.EPISODIC_MEMORY:
            kind = ContextFragmentKind.EPISODIC_MEMORY

        return ContextFragment(
            kind=kind,
            text=result.text,
            priority=result.priority,
            token_estimate=max(1, len(result.text.split())),
            source_id=result.result_id,
            metadata={
                "stream_kind": result.stream_kind.value,
                "confidence": result.confidence,
                "relevance": result.relevance,
            },
        )

    def _build_snapshot(self, session_id: str) -> ContextSnapshot:
        state = self._states[session_id]
        fragments = self.fragments_for(session_id)
        token_estimate = sum(fragment.token_estimate for fragment in fragments)

        if token_estimate > state.request.max_tokens:
            fragments = self._trim_to_budget(
                fragments=fragments,
                max_tokens=state.request.max_tokens,
            )
            token_estimate = sum(fragment.token_estimate for fragment in fragments)

        return ContextSnapshot(
            turn_id=state.request.turn_id,
            fragments=fragments,
            token_estimate=token_estimate,
            context_confidence=state.context_confidence,
            metadata={"session_id": session_id, "strategy": state.strategy.value},
        )

    @staticmethod
    def _trim_to_budget(
        *,
        fragments: tuple[ContextFragment, ...],
        max_tokens: int,
    ) -> tuple[ContextFragment, ...]:
        selected: list[ContextFragment] = []
        total = 0

        for fragment in fragments:
            if total + fragment.token_estimate > max_tokens:
                continue

            selected.append(fragment)
            total += fragment.token_estimate

        return tuple(selected)

    def _ready_for_llm(
        self,
        fragments: list[ContextFragment],
        confidence: float,
    ) -> bool:
        has_profile = any(
            fragment.kind == ContextFragmentKind.USER_PROFILE
            for fragment in fragments
        )
        has_episodic = any(
            fragment.kind == ContextFragmentKind.EPISODIC_MEMORY
            for fragment in fragments
        )

        return (
            has_profile
            and has_episodic
            and confidence >= self._config.ready_confidence_threshold
        )

    @staticmethod
    def _context_confidence(fragments: list[ContextFragment]) -> float:
        profile = 0.0
        episodic = 0.0
        semantic = 0.0

        for fragment in fragments:
            confidence_value = fragment.metadata.get("confidence", 0.0)

            if isinstance(confidence_value, int | float | str):
                confidence = float(confidence_value)
            else:
                confidence = 0.0

            if fragment.kind == ContextFragmentKind.USER_PROFILE:
                profile = max(profile, confidence or 0.92)
            elif fragment.kind == ContextFragmentKind.EPISODIC_MEMORY:
                episodic = max(episodic, confidence)
            elif fragment.kind == ContextFragmentKind.SEMANTIC_MEMORY:
                semantic = max(semantic, confidence)

        base = (profile * 0.35) + (episodic * 0.45) + (semantic * 0.20)

        if profile > 0 and episodic > 0:
            base += 0.12

        return min(1.0, base)

    @staticmethod
    def _compress_text(text: str) -> str:
        words = text.split()

        if len(words) <= 24:
            return text

        return " ".join(words[:24]) + " ..."

    def _record_context_span(self, state: PredictiveContextSessionState) -> None:
        if state.started_at_ns is None or state.completed_at_ns is None:
            return

        self._profiler.record_stage(
            trace_id=state.trace_id,
            stage=PipelineStage.CONTEXT_BUILD,
            operation=LatencyOperation.CONTEXT_BUILD,
            subsystem=LatencySubsystem.COGNITION,
            start_ns=state.started_at_ns,
            end_ns=state.completed_at_ns,
            metadata={"session_id": state.session_id, "strategy": state.strategy.value},
        )

    @staticmethod
    def _event(
        *,
        session_id: str,
        kind: PredictiveContextEventKind,
        reason: PredictiveContextReason,
        fragment_id: str | None = None,
        latency_ms: float | None = None,
        confidence: float | None = None,
        metadata: dict[str, object] | None = None,
    ) -> PredictiveContextEvent:
        return PredictiveContextEvent(
            session_id=session_id,
            kind=kind,
            reason=reason,
            fragment_id=fragment_id,
            latency_ms=latency_ms,
            confidence=confidence,
            metadata=metadata or {},
        )

    @staticmethod
    def _missing_session(session_id: str) -> PredictiveContextResult:
        return PredictiveContextResult(
            success=False,
            reason=PredictiveContextReason.SESSION_NOT_FOUND,
            session_id=session_id,
            status=PredictiveContextStatus.FAILED,
            message="predictive context session not found",
        )

    @staticmethod
    def _failure(
        *,
        session_id: str,
        reason: PredictiveContextReason,
        status: PredictiveContextStatus,
        message: str,
        state: PredictiveContextSessionState | None = None,
    ) -> PredictiveContextResult:
        return PredictiveContextResult(
            success=False,
            reason=reason,
            session_id=session_id,
            status=status,
            state=state,
            message=message,
        )