from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from typing import Protocol
from uuid import uuid4

from pydantic import Field, field_validator

from jarvis.latency.models import LatencyOperation, LatencySubsystem
from jarvis.latency.profiler import (
    PipelineLatencyProfiler,
    PipelineProfilerConfig,
    PipelineProfilerReport,
    PipelineStage,
)
from jarvis.latency.streaming_stt import SpeculativeWorkHint, SpeculativeWorkKind
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class MemoryStreamKind(StrEnum):
    """
    Parallel memory retrieval streams.

    Profile is small/fast, episodic is recent/high-priority, semantic is
    broader/topic-relevant and may be cancelled by early stopping.
    """

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROFILE = "profile"


class MemoryStreamStatus(StrEnum):
    """
    Memory stream lifecycle status.
    """

    CREATED = "created"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class MemoryResultSource(StrEnum):
    """
    Source of a streamed memory result.
    """

    EPISODIC_MEMORY = "episodic_memory"
    SEMANTIC_MEMORY = "semantic_memory"
    USER_PROFILE = "user_profile"


class StreamingMemoryEventKind(StrEnum):
    """
    Streaming memory event kind.
    """

    SESSION_CREATED = "session_created"
    SESSION_STARTED = "session_started"
    STREAM_STARTED = "stream_started"
    RESULT_EMITTED = "result_emitted"
    FIRST_RESULT_READY = "first_result_ready"
    CONTEXT_CONFIDENCE_UPDATED = "context_confidence_updated"
    EARLY_STOP_TRIGGERED = "early_stop_triggered"
    STREAM_CANCELLED = "stream_cancelled"
    SESSION_COMPLETED = "session_completed"
    SESSION_CANCELLED = "session_cancelled"
    SESSION_FAILED = "session_failed"


class StreamingMemoryReason(StrEnum):
    """
    Machine-readable Streaming Memory reasons.
    """

    SESSION_CREATED = "session_created"
    SESSION_STARTED = "session_started"
    STREAM_STARTED = "stream_started"
    RESULT_ACCEPTED = "result_accepted"
    FIRST_RESULT_RECORDED = "first_result_recorded"
    CONTEXT_CONFIDENCE_UPDATED = "context_confidence_updated"
    EARLY_STOP_CONFIDENCE_REACHED = "early_stop_confidence_reached"
    SEMANTIC_STREAM_CANCELLED = "semantic_stream_cancelled"
    SESSION_COMPLETED = "session_completed"
    SESSION_CANCELLED = "session_cancelled"
    SESSION_FAILED = "session_failed"
    SESSION_NOT_FOUND = "session_not_found"
    SESSION_NOT_ACTIVE = "session_not_active"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    RUNTIME_RESET = "runtime_reset"


class MemoryRetrievalQuery(OrchestrationModel):
    """
    Query for streaming memory retrieval.
    """

    query_id: str = Field(default_factory=lambda: uuid4().hex)
    text: str
    trace_id: str = Field(default_factory=lambda: uuid4().hex)
    speculative: bool = True
    source_hint_id: str | None = None
    created_at_ns: int = Field(default_factory=time.perf_counter_ns, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("query_id", "text", "trace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class MemoryStreamResult(OrchestrationModel):
    """
    One memory result emitted incrementally.
    """

    result_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    query_id: str
    stream_kind: MemoryStreamKind
    source: MemoryResultSource
    text: str
    priority: int = Field(ge=0)
    relevance: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    emitted_at_ns: int = Field(default_factory=time.perf_counter_ns, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("result_id", "session_id", "query_id", "text")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class MemoryStreamState(OrchestrationModel):
    """
    State for one parallel retrieval stream.
    """

    stream_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    kind: MemoryStreamKind
    status: MemoryStreamStatus = MemoryStreamStatus.CREATED
    started_at_ns: int | None = None
    completed_at_ns: int | None = None
    cancelled_at_ns: int | None = None
    failed_at_ns: int | None = None
    result_count: int = Field(default=0, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("stream_id", "session_id")
    @classmethod
    def _required_ids(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class StreamingMemoryEvent(OrchestrationModel):
    """
    Streaming memory event.
    """

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    kind: StreamingMemoryEventKind
    reason: StreamingMemoryReason
    stream_kind: MemoryStreamKind | None = None
    result_id: str | None = None
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


class StreamingMemorySessionState(OrchestrationModel):
    """
    Runtime state for one streaming memory retrieval session.
    """

    session_id: str = Field(default_factory=lambda: uuid4().hex)
    query: MemoryRetrievalQuery
    status: MemoryStreamStatus = MemoryStreamStatus.CREATED
    started_at_ns: int | None = None
    first_result_at_ns: int | None = None
    completed_at_ns: int | None = None
    cancelled_at_ns: int | None = None
    failed_at_ns: int | None = None
    result_count: int = Field(default=0, ge=0)
    context_confidence: float = Field(default=0.0, ge=0, le=1)
    early_stopped: bool = False
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("session_id")
    @classmethod
    def _required_session_id(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("session_id cannot be empty.")

        return cleaned

    def first_result_latency_ms(self) -> float | None:
        if self.started_at_ns is None or self.first_result_at_ns is None:
            return None

        return (self.first_result_at_ns - self.started_at_ns) / 1_000_000.0

    def total_latency_ms(self) -> float | None:
        if self.started_at_ns is None or self.completed_at_ns is None:
            return None

        return (self.completed_at_ns - self.started_at_ns) / 1_000_000.0


class StreamingMemoryResult(OrchestrationModel):
    """
    Result from streaming memory runtime operation.
    """

    success: bool
    reason: StreamingMemoryReason
    session_id: str
    status: MemoryStreamStatus
    memory_result: MemoryStreamResult | None = None
    event: StreamingMemoryEvent | None = None
    state: StreamingMemorySessionState | None = None
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


class StreamingMemoryReport(OrchestrationModel):
    """
    Final streaming memory retrieval report.
    """

    session_id: str
    query_id: str
    trace_id: str
    status: MemoryStreamStatus
    result_count: int = Field(ge=0)
    episodic_count: int = Field(ge=0)
    semantic_count: int = Field(ge=0)
    profile_count: int = Field(ge=0)
    context_confidence: float = Field(ge=0, le=1)
    early_stopped: bool
    first_result_latency_ms: float | None = None
    total_latency_ms: float | None = None
    results: tuple[MemoryStreamResult, ...]
    events: tuple[StreamingMemoryEvent, ...]
    profiler_report: PipelineProfilerReport | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("session_id", "query_id", "trace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class StreamingMemoryRuntimeSnapshot(OrchestrationModel):
    """
    Runtime diagnostics for Phase 7 Step 8.
    """

    name: str
    session_count: int = Field(ge=0)
    active_count: int = Field(ge=0)
    completed_count: int = Field(ge=0)
    cancelled_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    result_count: int = Field(ge=0)
    event_count: int = Field(ge=0)
    report_count: int = Field(ge=0)
    last_reason: StreamingMemoryReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        return cleaned


class StreamingMemoryBackend(Protocol):
    """
    Backend protocol for streaming memory sources.
    """

    def retrieve(
        self,
        *,
        session_id: str,
        query: MemoryRetrievalQuery,
        stream_kind: MemoryStreamKind,
    ) -> tuple[MemoryStreamResult, ...]:
        """
        Return available results for a stream kind.

        Real backends should emit incrementally. Step 8 models the contract
        deterministically for tests.
        """


class FakeStreamingMemoryBackend:
    """
    Deterministic backend used for tests and runtime smoke validation.
    """

    def retrieve(
        self,
        *,
        session_id: str,
        query: MemoryRetrievalQuery,
        stream_kind: MemoryStreamKind,
    ) -> tuple[MemoryStreamResult, ...]:
        if stream_kind == MemoryStreamKind.PROFILE:
            return (
                MemoryStreamResult(
                    session_id=session_id,
                    query_id=query.query_id,
                    stream_kind=stream_kind,
                    source=MemoryResultSource.USER_PROFILE,
                    text="User is building JARVIS OS.",
                    priority=100,
                    relevance=0.95,
                    confidence=0.92,
                ),
            )

        if stream_kind == MemoryStreamKind.EPISODIC:
            return (
                MemoryStreamResult(
                    session_id=session_id,
                    query_id=query.query_id,
                    stream_kind=stream_kind,
                    source=MemoryResultSource.EPISODIC_MEMORY,
                    text=f"Recent discussion related to: {query.text}",
                    priority=90,
                    relevance=0.90,
                    confidence=0.88,
                ),
            )

        return (
            MemoryStreamResult(
                session_id=session_id,
                query_id=query.query_id,
                stream_kind=stream_kind,
                source=MemoryResultSource.SEMANTIC_MEMORY,
                text=f"Semantic concept relevant to: {query.text}",
                priority=70,
                relevance=0.82,
                confidence=0.80,
            ),
        )


@dataclass(frozen=True, slots=True)
class EmbeddingCacheConfig:
    """
    LRU cache for recent query embeddings.

    Step 8 models embedding cache behavior without requiring a vector library.
    """

    max_entries: int = 50

    def validate(self) -> None:
        if self.max_entries < 1:
            raise ValueError("max_entries must be positive.")


class EmbeddingCache:
    """
    Tiny deterministic LRU embedding cache.

    Real embedding vectors can plug in later. This cache preserves the Phase 7
    contract: common recent turns must avoid repeated embedding work.
    """

    def __init__(self, *, config: EmbeddingCacheConfig | None = None) -> None:
        self._config = config or EmbeddingCacheConfig()
        self._config.validate()
        self._items: OrderedDict[str, tuple[float, ...]] = OrderedDict()
        self._lock = RLock()

    def get(self, text: str) -> tuple[float, ...] | None:
        key = text.strip().lower()

        with self._lock:
            value = self._items.get(key)

            if value is None:
                return None

            self._items.move_to_end(key)
            return value

    def set(self, text: str, embedding: tuple[float, ...]) -> None:
        key = text.strip().lower()

        with self._lock:
            self._items[key] = embedding
            self._items.move_to_end(key)

            while len(self._items) > self._config.max_entries:
                self._items.popitem(last=False)

    def get_or_create(self, text: str) -> tuple[float, ...]:
        cached = self.get(text)

        if cached is not None:
            return cached

        embedding = self._fake_embedding(text)
        self.set(text, embedding)

        return embedding

    def size(self) -> int:
        with self._lock:
            return len(self._items)

    @staticmethod
    def _fake_embedding(text: str) -> tuple[float, ...]:
        normalized = text.strip().lower()
        length = float(len(normalized))
        word_count = float(len(normalized.split()))
        checksum = float(sum(ord(char) for char in normalized) % 997)

        return (length, word_count, checksum)


@dataclass(frozen=True, slots=True)
class StreamingMemoryRuntimeConfig:
    """
    Streaming memory retrieval configuration.
    """

    name: str = "streaming_memory_runtime"
    first_result_target_ms: float = 80.0
    full_context_target_ms: float = 150.0
    early_stop_confidence: float = 0.80
    profile_first: bool = True
    profile_streaming_memory: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.first_result_target_ms <= 0:
            raise ValueError("first_result_target_ms must be positive.")

        if self.full_context_target_ms <= 0:
            raise ValueError("full_context_target_ms must be positive.")

        if not 0 <= self.early_stop_confidence <= 1:
            raise ValueError("early_stop_confidence must be within 0..1.")


class StreamingMemoryRuntime:
    """
    Phase 7 Step 8 Streaming Memory Retrieval.

    Responsibilities:
    - start profile/episodic/semantic retrieval streams
    - emit results as they arrive
    - priority-order result consumption
    - update context confidence incrementally
    - early-stop/cancel semantic stream when enough context exists
    - expose first-result and full-context latency
    - feed latency profiler

    Non-responsibilities:
    - no direct cognition mutation
    - no direct context mutation
    - no direct tool/action execution
    - no memory writes
    """

    def __init__(
        self,
        *,
        config: StreamingMemoryRuntimeConfig | None = None,
        backend: StreamingMemoryBackend | None = None,
        embedding_cache: EmbeddingCache | None = None,
        profiler: PipelineLatencyProfiler | None = None,
    ) -> None:
        self._config = config or StreamingMemoryRuntimeConfig()
        self._config.validate()

        self._backend = backend or FakeStreamingMemoryBackend()
        self._embedding_cache = embedding_cache or EmbeddingCache()
        self._profiler = profiler or PipelineLatencyProfiler(
            config=PipelineProfilerConfig(record_to_latency_runtime=True)
        )
        self._states: dict[str, StreamingMemorySessionState] = {}
        self._streams: dict[str, dict[MemoryStreamKind, MemoryStreamState]] = {}
        self._results: dict[str, list[MemoryStreamResult]] = {}
        self._events: dict[str, list[StreamingMemoryEvent]] = {}
        self._reports: list[StreamingMemoryReport] = []
        self._lock = RLock()
        self._last_reason: StreamingMemoryReason | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def create_session(
        self,
        *,
        query_text: str,
        speculative: bool = True,
        trace_id: str | None = None,
        source_hint: SpeculativeWorkHint | None = None,
        metadata: dict[str, object] | None = None,
    ) -> StreamingMemorySessionState:
        query = MemoryRetrievalQuery(
            text=query_text,
            trace_id=trace_id or uuid4().hex,
            speculative=speculative,
            source_hint_id=source_hint.hint_id if source_hint is not None else None,
            metadata=metadata or {},
        )
        state = StreamingMemorySessionState(query=query)
        event = self._event(
            session_id=state.session_id,
            kind=StreamingMemoryEventKind.SESSION_CREATED,
            reason=StreamingMemoryReason.SESSION_CREATED,
        )

        with self._lock:
            self._states[state.session_id] = state
            self._streams[state.session_id] = {}
            self._results[state.session_id] = []
            self._events[state.session_id] = [event]
            self._last_reason = StreamingMemoryReason.SESSION_CREATED

        self._profiler.start_trace(
            name="streaming_memory_retrieval",
            trace_id=query.trace_id,
        )

        return state

    def start_session(self, session_id: str) -> StreamingMemoryResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            if state.status != MemoryStreamStatus.CREATED:
                return self._failure(
                    session_id=session_id,
                    reason=StreamingMemoryReason.SESSION_NOT_ACTIVE,
                    status=state.status,
                    message="memory session cannot start from current state",
                    state=state,
                )

            started = state.model_copy(
                update={
                    "status": MemoryStreamStatus.ACTIVE,
                    "started_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = started
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=StreamingMemoryEventKind.SESSION_STARTED,
                    reason=StreamingMemoryReason.SESSION_STARTED,
                )
            )
            self._last_reason = StreamingMemoryReason.SESSION_STARTED

        self._embedding_cache.get_or_create(started.query.text)

        return StreamingMemoryResult(
            success=True,
            reason=StreamingMemoryReason.SESSION_STARTED,
            session_id=session_id,
            status=MemoryStreamStatus.ACTIVE,
            state=started,
            message="streaming memory session started",
        )

    def run_available_streams(
        self,
        session_id: str,
    ) -> tuple[StreamingMemoryResult, ...]:
        state = self.state_for(session_id)

        if state is None:
            return (self._missing_session(session_id),)

        if state.status != MemoryStreamStatus.ACTIVE:
            return (
                self._failure(
                    session_id=session_id,
                    reason=StreamingMemoryReason.SESSION_NOT_ACTIVE,
                    status=state.status,
                    message="memory session is not active",
                    state=state,
                ),
            )

        order = (
            (
                MemoryStreamKind.PROFILE,
                MemoryStreamKind.EPISODIC,
                MemoryStreamKind.SEMANTIC,
            )
            if self._config.profile_first
            else (
                MemoryStreamKind.EPISODIC,
                MemoryStreamKind.PROFILE,
                MemoryStreamKind.SEMANTIC,
            )
        )
        outputs: list[StreamingMemoryResult] = []

        for stream_kind in order:
            current = self.state_for(session_id)

            if current is None or current.status != MemoryStreamStatus.ACTIVE:
                break

            if current.early_stopped and stream_kind == MemoryStreamKind.SEMANTIC:
                self._cancel_stream(
                    session_id=session_id,
                    stream_kind=stream_kind,
                    reason=StreamingMemoryReason.SEMANTIC_STREAM_CANCELLED,
                )
                continue

            outputs.extend(
                self._run_stream(
                    session_id=session_id,
                    stream_kind=stream_kind,
                )
            )

            current = self.state_for(session_id)

            if (
                current is not None
                and current.context_confidence >= self._config.early_stop_confidence
                and self._has_profile_and_episodic_results(session_id)
            ):
                self._trigger_early_stop(session_id)
                self._cancel_stream(
                    session_id=session_id,
                    stream_kind=MemoryStreamKind.SEMANTIC,
                    reason=StreamingMemoryReason.SEMANTIC_STREAM_CANCELLED,
            )

        return tuple(outputs)

    def complete_session(self, session_id: str) -> StreamingMemoryReport:
        state = self.state_for(session_id)

        if state is None:
            raise ValueError(f"streaming memory session not found: {session_id}")

        if state.status not in {
            MemoryStreamStatus.ACTIVE,
            MemoryStreamStatus.CREATED,
        }:
            raise ValueError("memory session cannot complete from current state")

        with self._lock:
            current = self._states[session_id]
            completed = current.model_copy(
                update={
                    "status": MemoryStreamStatus.COMPLETED,
                    "completed_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = completed
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=StreamingMemoryEventKind.SESSION_COMPLETED,
                    reason=StreamingMemoryReason.SESSION_COMPLETED,
                )
            )
            self._last_reason = StreamingMemoryReason.SESSION_COMPLETED

        self._record_memory_span(completed)

        profiler_report = None

        if self._config.profile_streaming_memory:
            profiler_report = self._profiler.complete_trace(completed.query.trace_id)

        results = self.results_for(session_id)
        report = StreamingMemoryReport(
            session_id=session_id,
            query_id=completed.query.query_id,
            trace_id=completed.query.trace_id,
            status=completed.status,
            result_count=len(results),
            episodic_count=self._count_kind(results, MemoryStreamKind.EPISODIC),
            semantic_count=self._count_kind(results, MemoryStreamKind.SEMANTIC),
            profile_count=self._count_kind(results, MemoryStreamKind.PROFILE),
            context_confidence=completed.context_confidence,
            early_stopped=completed.early_stopped,
            first_result_latency_ms=completed.first_result_latency_ms(),
            total_latency_ms=completed.total_latency_ms(),
            results=results,
            events=self.events_for(session_id),
            profiler_report=profiler_report,
        )

        with self._lock:
            self._reports.append(report)

        return report

    def cancel_session(
        self,
        session_id: str,
        *,
        reason: str = "cancelled",
    ) -> StreamingMemoryResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            if state.status in {
                MemoryStreamStatus.CANCELLED,
                MemoryStreamStatus.COMPLETED,
                MemoryStreamStatus.FAILED,
            }:
                return self._failure(
                    session_id=session_id,
                    reason=StreamingMemoryReason.SESSION_NOT_ACTIVE,
                    status=state.status,
                    message="memory session already terminal",
                    state=state,
                )

            cancelled = state.model_copy(
                update={
                    "status": MemoryStreamStatus.CANCELLED,
                    "cancelled_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = cancelled
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=StreamingMemoryEventKind.SESSION_CANCELLED,
                    reason=StreamingMemoryReason.SESSION_CANCELLED,
                    metadata={"cancel_reason": reason},
                )
            )
            self._last_reason = StreamingMemoryReason.SESSION_CANCELLED

        return StreamingMemoryResult(
            success=True,
            reason=StreamingMemoryReason.SESSION_CANCELLED,
            session_id=session_id,
            status=MemoryStreamStatus.CANCELLED,
            state=cancelled,
            message="streaming memory session cancelled",
        )

    def fail_session(
        self,
        session_id: str,
        *,
        error: str,
    ) -> StreamingMemoryResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            failed = state.model_copy(
                update={
                    "status": MemoryStreamStatus.FAILED,
                    "failed_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = failed
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=StreamingMemoryEventKind.SESSION_FAILED,
                    reason=StreamingMemoryReason.SESSION_FAILED,
                    metadata={"error": error},
                )
            )
            self._last_reason = StreamingMemoryReason.SESSION_FAILED

        return StreamingMemoryResult(
            success=True,
            reason=StreamingMemoryReason.SESSION_FAILED,
            session_id=session_id,
            status=MemoryStreamStatus.FAILED,
            state=failed,
            message="streaming memory session failed",
        )

    def state_for(self, session_id: str) -> StreamingMemorySessionState | None:
        with self._lock:
            return self._states.get(session_id)

    def results_for(self, session_id: str) -> tuple[MemoryStreamResult, ...]:
        with self._lock:
            items = tuple(self._results.get(session_id, ()))

        return tuple(sorted(items, key=lambda item: (-item.priority, -item.confidence)))

    def events_for(self, session_id: str) -> tuple[StreamingMemoryEvent, ...]:
        with self._lock:
            return tuple(self._events.get(session_id, ()))

    def reports(self) -> tuple[StreamingMemoryReport, ...]:
        with self._lock:
            return tuple(self._reports)

    def latest_report(self) -> StreamingMemoryReport | None:
        with self._lock:
            if not self._reports:
                return None

            return self._reports[-1]

    def snapshot(self) -> StreamingMemoryRuntimeSnapshot:
        with self._lock:
            states = tuple(self._states.values())

            return StreamingMemoryRuntimeSnapshot(
                name=self.name,
                session_count=len(states),
                active_count=sum(
                    1 for state in states if state.status == MemoryStreamStatus.ACTIVE
                ),
                completed_count=sum(
                    1
                    for state in states
                    if state.status == MemoryStreamStatus.COMPLETED
                ),
                cancelled_count=sum(
                    1
                    for state in states
                    if state.status == MemoryStreamStatus.CANCELLED
                ),
                failed_count=sum(
                    1 for state in states if state.status == MemoryStreamStatus.FAILED
                ),
                result_count=sum(len(items) for items in self._results.values()),
                event_count=sum(len(items) for items in self._events.values()),
                report_count=len(self._reports),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        with self._lock:
            self._states.clear()
            self._streams.clear()
            self._results.clear()
            self._events.clear()
            self._reports.clear()
            self._last_reason = StreamingMemoryReason.RUNTIME_RESET

    def _run_stream(
        self,
        *,
        session_id: str,
        stream_kind: MemoryStreamKind,
    ) -> tuple[StreamingMemoryResult, ...]:
        state = self._states[session_id]
        stream_state = MemoryStreamState(
            session_id=session_id,
            kind=stream_kind,
            status=MemoryStreamStatus.ACTIVE,
            started_at_ns=time.perf_counter_ns(),
        )

        with self._lock:
            self._streams[session_id][stream_kind] = stream_state
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=StreamingMemoryEventKind.STREAM_STARTED,
                    reason=StreamingMemoryReason.STREAM_STARTED,
                    stream_kind=stream_kind,
                )
            )

        results = self._backend.retrieve(
            session_id=session_id,
            query=state.query,
            stream_kind=stream_kind,
        )
        outputs: list[StreamingMemoryResult] = []

        for result in results:
            outputs.append(self._accept_result(result))

        with self._lock:
            current_stream = self._streams[session_id][stream_kind]
            self._streams[session_id][stream_kind] = current_stream.model_copy(
                update={
                    "status": MemoryStreamStatus.COMPLETED,
                    "completed_at_ns": time.perf_counter_ns(),
                    "result_count": len(results),
                }
            )

        return tuple(outputs)

    def _accept_result(self, result: MemoryStreamResult) -> StreamingMemoryResult:
        with self._lock:
            state = self._states[result.session_id]
            first_result_at_ns = state.first_result_at_ns or result.emitted_at_ns
            self._results[result.session_id].append(result)
            confidence = self._combined_confidence(self._results[result.session_id])

            updated = state.model_copy(
                update={
                    "first_result_at_ns": first_result_at_ns,
                    "result_count": state.result_count + 1,
                    "context_confidence": confidence,
                }
            )
            self._states[result.session_id] = updated

            latency = updated.first_result_latency_ms()
            self._events[result.session_id].append(
                self._event(
                    session_id=result.session_id,
                    kind=StreamingMemoryEventKind.RESULT_EMITTED,
                    reason=StreamingMemoryReason.RESULT_ACCEPTED,
                    stream_kind=result.stream_kind,
                    result_id=result.result_id,
                    latency_ms=latency,
                    confidence=result.confidence,
                )
            )

            if state.first_result_at_ns is None:
                self._events[result.session_id].append(
                    self._event(
                        session_id=result.session_id,
                        kind=StreamingMemoryEventKind.FIRST_RESULT_READY,
                        reason=StreamingMemoryReason.FIRST_RESULT_RECORDED,
                        stream_kind=result.stream_kind,
                        result_id=result.result_id,
                        latency_ms=latency,
                        confidence=result.confidence,
                    )
                )

            self._events[result.session_id].append(
                self._event(
                    session_id=result.session_id,
                    kind=StreamingMemoryEventKind.CONTEXT_CONFIDENCE_UPDATED,
                    reason=StreamingMemoryReason.CONTEXT_CONFIDENCE_UPDATED,
                    confidence=confidence,
                )
            )
            self._last_reason = StreamingMemoryReason.RESULT_ACCEPTED

        return StreamingMemoryResult(
            success=True,
            reason=StreamingMemoryReason.RESULT_ACCEPTED,
            session_id=result.session_id,
            status=MemoryStreamStatus.ACTIVE,
            memory_result=result,
            state=updated,
            message="streaming memory result accepted",
        )

    def _trigger_early_stop(self, session_id: str) -> None:
        with self._lock:
            state = self._states[session_id]

            if state.early_stopped:
                return

            updated = state.model_copy(update={"early_stopped": True})
            self._states[session_id] = updated
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=StreamingMemoryEventKind.EARLY_STOP_TRIGGERED,
                    reason=StreamingMemoryReason.EARLY_STOP_CONFIDENCE_REACHED,
                    confidence=updated.context_confidence,
                )
            )

    def _cancel_stream(
        self,
        *,
        session_id: str,
        stream_kind: MemoryStreamKind,
        reason: StreamingMemoryReason,
    ) -> None:
        with self._lock:
            existing = self._streams[session_id].get(stream_kind)

            if existing is None:
                existing = MemoryStreamState(session_id=session_id, kind=stream_kind)

            self._streams[session_id][stream_kind] = existing.model_copy(
                update={
                    "status": MemoryStreamStatus.CANCELLED,
                    "cancelled_at_ns": time.perf_counter_ns(),
                }
            )
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=StreamingMemoryEventKind.STREAM_CANCELLED,
                    reason=reason,
                    stream_kind=stream_kind,
                )
            )

    def _record_memory_span(self, state: StreamingMemorySessionState) -> None:
        if state.started_at_ns is None or state.completed_at_ns is None:
            return

        self._profiler.record_stage(
            trace_id=state.query.trace_id,
            stage=PipelineStage.MEMORY_RETRIEVAL,
            operation=LatencyOperation.MEMORY_RETRIEVAL,
            subsystem=LatencySubsystem.MEMORY,
            start_ns=state.started_at_ns,
            end_ns=state.completed_at_ns,
            metadata={"session_id": state.session_id},
        )

    @staticmethod
    def _combined_confidence(results: list[MemoryStreamResult]) -> float:
        if not results:
            return 0.0

        best_by_stream: dict[MemoryStreamKind, float] = {}

        for result in results:
            existing = best_by_stream.get(result.stream_kind, 0.0)
            best_by_stream[result.stream_kind] = max(existing, result.confidence)

        episodic = best_by_stream.get(MemoryStreamKind.EPISODIC, 0.0)
        semantic = best_by_stream.get(MemoryStreamKind.SEMANTIC, 0.0)
        profile = best_by_stream.get(MemoryStreamKind.PROFILE, 0.0)

        combined = (episodic * 0.45) + (semantic * 0.25) + (profile * 0.30)

        if episodic > 0 and profile > 0:
            combined += 0.12

        return min(1.0, combined)

    def _has_profile_and_episodic_results(self, session_id: str) -> bool:
        with self._lock:
            results = tuple(self._results.get(session_id, ()))

        has_profile = any(
            result.stream_kind == MemoryStreamKind.PROFILE for result in results
        )
        has_episodic = any(
            result.stream_kind == MemoryStreamKind.EPISODIC for result in results
        )

        return has_profile and has_episodic

    @staticmethod
    def _count_kind(
        results: tuple[MemoryStreamResult, ...],
        kind: MemoryStreamKind,
    ) -> int:
        return sum(1 for result in results if result.stream_kind == kind)

    @staticmethod
    def _event(
        *,
        session_id: str,
        kind: StreamingMemoryEventKind,
        reason: StreamingMemoryReason,
        stream_kind: MemoryStreamKind | None = None,
        result_id: str | None = None,
        latency_ms: float | None = None,
        confidence: float | None = None,
        metadata: dict[str, object] | None = None,
    ) -> StreamingMemoryEvent:
        return StreamingMemoryEvent(
            session_id=session_id,
            kind=kind,
            reason=reason,
            stream_kind=stream_kind,
            result_id=result_id,
            latency_ms=latency_ms,
            confidence=confidence,
            metadata=metadata or {},
        )

    @staticmethod
    def _missing_session(session_id: str) -> StreamingMemoryResult:
        return StreamingMemoryResult(
            success=False,
            reason=StreamingMemoryReason.SESSION_NOT_FOUND,
            session_id=session_id,
            status=MemoryStreamStatus.FAILED,
            message="streaming memory session not found",
        )

    @staticmethod
    def _failure(
        *,
        session_id: str,
        reason: StreamingMemoryReason,
        status: MemoryStreamStatus,
        message: str,
        state: StreamingMemorySessionState | None = None,
    ) -> StreamingMemoryResult:
        return StreamingMemoryResult(
            success=False,
            reason=reason,
            session_id=session_id,
            status=status,
            state=state,
            message=message,
        )


def memory_query_from_hint(
    *,
    text: str,
    hint: SpeculativeWorkHint,
    trace_id: str | None = None,
) -> MemoryRetrievalQuery:
    """
    Convert a speculative STT hint into a memory retrieval query.

    This preserves the rule: STT does not retrieve memory directly. It emits
    hints; memory runtime owns retrieval.
    """

    if hint.kind != SpeculativeWorkKind.MEMORY_PREFETCH:
        raise ValueError("memory query can only be created from memory prefetch hint")

    return MemoryRetrievalQuery(
        text=text,
        trace_id=trace_id or uuid4().hex,
        speculative=True,
        source_hint_id=hint.hint_id,
    )