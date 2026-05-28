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
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class PrewarmTarget(StrEnum):
    """
    Runtime targets that may be safely pre-warmed.

    Prewarming prepares capacity only. It must never execute user actions.
    """

    LLM_CONNECTION = "llm_connection"
    TTS_ENGINE = "tts_engine"
    AUDIO_PLAYBACK = "audio_playback"
    MEMORY_HOT_CACHE = "memory_hot_cache"
    WORKSPACE_INDEX = "workspace_index"


class PrewarmStatus(StrEnum):
    """
    Prewarm lifecycle status.
    """

    COLD = "cold"
    WARMING = "warming"
    WARM = "warm"
    DEGRADED = "degraded"
    CANCELLED = "cancelled"
    FAILED = "failed"


class WarmResourceRole(StrEnum):
    """
    Resource role inside a pool.
    """

    PRIMARY = "primary"
    STANDBY = "standby"
    WORKER = "worker"
    BUFFER = "buffer"
    CACHE = "cache"


class PrewarmEventKind(StrEnum):
    """
    Prewarming event kind.
    """

    SESSION_CREATED = "session_created"
    TARGET_WARMING_STARTED = "target_warming_started"
    RESOURCE_WARMED = "resource_warmed"
    IDLE_PING_SENT = "idle_ping_sent"
    CONNECTION_STANDBY_READY = "connection_standby_ready"
    TTS_POOL_READY = "tts_pool_ready"
    AUDIO_BUFFER_READY = "audio_buffer_ready"
    MEMORY_CACHE_READY = "memory_cache_ready"
    WORKSPACE_INDEX_READY = "workspace_index_ready"
    SESSION_COMPLETED = "session_completed"
    SESSION_CANCELLED = "session_cancelled"
    SESSION_FAILED = "session_failed"
    CACHE_INVALIDATED = "cache_invalidated"


class PrewarmReason(StrEnum):
    """
    Machine-readable prewarming reasons.
    """

    SESSION_CREATED = "session_created"
    TARGET_WARMING_STARTED = "target_warming_started"
    LLM_CONNECTION_READY = "llm_connection_ready"
    STANDBY_CONNECTION_READY = "standby_connection_ready"
    IDLE_PING_SENT = "idle_ping_sent"
    TTS_ENGINE_READY = "tts_engine_ready"
    TTS_WORKER_READY = "tts_worker_ready"
    PHONEME_CACHE_READY = "phoneme_cache_ready"
    AUDIO_STREAM_READY = "audio_stream_ready"
    AUDIO_PREBUFFER_READY = "audio_prebuffer_ready"
    MEMORY_PROFILE_READY = "memory_profile_ready"
    MEMORY_TURNS_READY = "memory_turns_ready"
    WORKSPACE_INDEX_READY = "workspace_index_ready"
    SESSION_COMPLETED = "session_completed"
    SESSION_CANCELLED = "session_cancelled"
    SESSION_FAILED = "session_failed"
    SESSION_NOT_FOUND = "session_not_found"
    CACHE_INVALIDATED = "cache_invalidated"
    RUNTIME_RESET = "runtime_reset"


class WarmResource(OrchestrationModel):
    """
    One warmed resource.

    Examples:
    - LLM primary connection
    - LLM standby connection
    - TTS synthesis worker
    - audio prebuffer
    - user profile hot cache
    """

    resource_id: str = Field(default_factory=lambda: uuid4().hex)
    target: PrewarmTarget
    role: WarmResourceRole
    name: str
    status: PrewarmStatus = PrewarmStatus.WARM
    warmed_at_ns: int = Field(default_factory=time.perf_counter_ns, ge=0)
    last_ping_at_ns: int | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("resource_id", "name")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class PrewarmPoolState(OrchestrationModel):
    """
    Pool state for one prewarm target.
    """

    pool_id: str = Field(default_factory=lambda: uuid4().hex)
    target: PrewarmTarget
    status: PrewarmStatus = PrewarmStatus.COLD
    resources: tuple[WarmResource, ...] = ()
    created_at_ns: int = Field(default_factory=time.perf_counter_ns, ge=0)
    warmed_at_ns: int | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("pool_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("pool_id cannot be empty.")

        return cleaned

    @property
    def warm_count(self) -> int:
        return sum(
            1 for resource in self.resources if resource.status == PrewarmStatus.WARM
        )


class PrewarmEvent(OrchestrationModel):
    """
    Prewarming runtime event.
    """

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    kind: PrewarmEventKind
    reason: PrewarmReason
    target: PrewarmTarget | None = None
    resource_id: str | None = None
    latency_ms: float | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("event_id", "session_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class PrewarmSessionState(OrchestrationModel):
    """
    State for one prewarming session.
    """

    session_id: str = Field(default_factory=lambda: uuid4().hex)
    trace_id: str = Field(default_factory=lambda: uuid4().hex)
    status: PrewarmStatus = PrewarmStatus.COLD
    started_at_ns: int = Field(default_factory=time.perf_counter_ns, ge=0)
    completed_at_ns: int | None = None
    cancelled_at_ns: int | None = None
    failed_at_ns: int | None = None
    warmed_target_count: int = Field(default=0, ge=0)
    warmed_resource_count: int = Field(default=0, ge=0)
    ping_count: int = Field(default=0, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("session_id", "trace_id")
    @classmethod
    def _required_ids(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    def total_latency_ms(self) -> float | None:
        if self.completed_at_ns is None:
            return None

        return (self.completed_at_ns - self.started_at_ns) / 1_000_000.0


class PrewarmResult(OrchestrationModel):
    """
    Result from a prewarming runtime operation.
    """

    success: bool
    reason: PrewarmReason
    session_id: str
    status: PrewarmStatus
    pool: PrewarmPoolState | None = None
    resource: WarmResource | None = None
    event: PrewarmEvent | None = None
    state: PrewarmSessionState | None = None
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


class PrewarmReport(OrchestrationModel):
    """
    Final report for a prewarming session.
    """

    session_id: str
    trace_id: str
    status: PrewarmStatus
    warmed_target_count: int = Field(ge=0)
    warmed_resource_count: int = Field(ge=0)
    ping_count: int = Field(ge=0)
    total_latency_ms: float | None = None
    pools: tuple[PrewarmPoolState, ...]
    events: tuple[PrewarmEvent, ...]
    profiler_report: PipelineProfilerReport | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("session_id", "trace_id")
    @classmethod
    def _required_ids(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class PrewarmRuntimeSnapshot(OrchestrationModel):
    """
    Runtime diagnostics for Phase 7 Step 11.
    """

    name: str
    session_count: int = Field(ge=0)
    warm_pool_count: int = Field(ge=0)
    warm_resource_count: int = Field(ge=0)
    report_count: int = Field(ge=0)
    cache_size: int = Field(ge=0)
    last_reason: PrewarmReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class HotCacheConfig:
    """
    LRU hot cache configuration.
    """

    max_entries: int = 5

    def validate(self) -> None:
        if self.max_entries < 1:
            raise ValueError("max_entries must be positive.")


class HotCache:
    """
    Small deterministic LRU hot cache.

    Used for last turns, profile fragments, phoneme patterns, and workspace hints.
    """

    def __init__(self, *, config: HotCacheConfig | None = None) -> None:
        self._config = config or HotCacheConfig()
        self._config.validate()
        self._items: OrderedDict[str, str] = OrderedDict()
        self._lock = RLock()

    def get(self, key: str) -> str | None:
        cache_key = self._key(key)

        with self._lock:
            value = self._items.get(cache_key)

            if value is None:
                return None

            self._items.move_to_end(cache_key)
            return value

    def set(self, key: str, value: str) -> None:
        cache_key = self._key(key)

        with self._lock:
            self._items[cache_key] = value
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
class PrewarmRuntimeConfig:
    """
    Phase 7 Step 11 prewarming configuration.
    """

    name: str = "prewarm_connection_pool_runtime"
    llm_connection_pool_size: int = 2
    tts_worker_pool_size: int = 2
    idle_ping_interval_ms: float = 30_000.0
    audio_prebuffer_ms: float = 200.0
    last_turn_cache_size: int = 5
    profile_pinned: bool = True
    profile_prewarming: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.llm_connection_pool_size < 2:
            raise ValueError("llm_connection_pool_size must be at least 2.")

        if self.tts_worker_pool_size < 1:
            raise ValueError("tts_worker_pool_size must be positive.")

        if self.idle_ping_interval_ms <= 0:
            raise ValueError("idle_ping_interval_ms must be positive.")

        if self.audio_prebuffer_ms <= 0:
            raise ValueError("audio_prebuffer_ms must be positive.")

        if self.last_turn_cache_size < 1:
            raise ValueError("last_turn_cache_size must be positive.")


class PrewarmConnectionPoolRuntime:
    """
    Phase 7 Step 11 Pre-warming & Connection Pool Runtime.

    Responsibilities:
    - maintain warm LLM connection slots
    - keep standby context prepared with system prompt metadata
    - keep TTS synthesis workers warm
    - keep common phoneme patterns cached
    - keep audio output stream and prebuffer ready
    - keep profile + last turns hot in memory
    - pre-index common workspace paths
    - send idle pings to prevent connection closure

    Non-responsibilities:
    - no real LLM generation
    - no real TTS synthesis
    - no real audio playback
    - no tool/action execution
    - no memory writes
    """

    def __init__(
        self,
        *,
        config: PrewarmRuntimeConfig | None = None,
        hot_cache: HotCache | None = None,
        profiler: PipelineLatencyProfiler | None = None,
    ) -> None:
        self._config = config or PrewarmRuntimeConfig()
        self._config.validate()

        self._hot_cache = hot_cache or HotCache(
            config=HotCacheConfig(max_entries=self._config.last_turn_cache_size)
        )
        self._profiler = profiler or PipelineLatencyProfiler(
            config=PipelineProfilerConfig(record_to_latency_runtime=True)
        )
        self._sessions: dict[str, PrewarmSessionState] = {}
        self._pools: dict[PrewarmTarget, PrewarmPoolState] = {}
        self._events: dict[str, list[PrewarmEvent]] = {}
        self._reports: list[PrewarmReport] = []
        self._lock = RLock()
        self._last_reason: PrewarmReason | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def create_session(
        self,
        *,
        trace_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> PrewarmSessionState:
        state = PrewarmSessionState(
            trace_id=trace_id or uuid4().hex,
            status=PrewarmStatus.WARMING,
            metadata=metadata or {},
        )
        event = self._event(
            session_id=state.session_id,
            kind=PrewarmEventKind.SESSION_CREATED,
            reason=PrewarmReason.SESSION_CREATED,
        )

        with self._lock:
            self._sessions[state.session_id] = state
            self._events[state.session_id] = [event]
            self._last_reason = PrewarmReason.SESSION_CREATED

        self._profiler.start_trace(
            name="prewarm_connection_pool",
            trace_id=state.trace_id,
        )

        return state

    def warm_all(self, session_id: str) -> tuple[PrewarmResult, ...]:
        state = self.state_for(session_id)

        if state is None:
            return (self._missing_session(session_id),)

        if state.status != PrewarmStatus.WARMING:
            return (
                self._failure(
                    session_id=session_id,
                    reason=PrewarmReason.SESSION_NOT_FOUND,
                    status=state.status,
                    message="prewarm session is not warming",
                    state=state,
                ),
            )

        results = (
            self.warm_llm_connections(session_id),
            self.warm_tts_pool(session_id),
            self.warm_audio_playback(session_id),
            self.warm_memory_cache(session_id),
            self.warm_workspace_index(session_id),
        )

        return results

    def warm_llm_connections(self, session_id: str) -> PrewarmResult:
        state = self.state_for(session_id)

        if state is None:
            return self._missing_session(session_id)

        resources: list[WarmResource] = []

        for index in range(self._config.llm_connection_pool_size):
            role = WarmResourceRole.PRIMARY if index == 0 else WarmResourceRole.STANDBY
            resource = WarmResource(
                target=PrewarmTarget.LLM_CONNECTION,
                role=role,
                name=f"llm_connection_{role.value}_{index}",
                metadata={
                    "keep_alive": True,
                    "system_prompt_loaded": role == WarmResourceRole.STANDBY,
                    "real_generation": False,
                },
            )
            resources.append(resource)

        pool = self._pool(
            target=PrewarmTarget.LLM_CONNECTION,
            resources=tuple(resources),
        )
        self._store_pool(session_id=session_id, pool=pool)

        standby = next(
            item for item in resources if item.role == WarmResourceRole.STANDBY
        )

        self._append_event(
            session_id=session_id,
            kind=PrewarmEventKind.CONNECTION_STANDBY_READY,
            reason=PrewarmReason.STANDBY_CONNECTION_READY,
            target=PrewarmTarget.LLM_CONNECTION,
            resource_id=standby.resource_id,
        )

        return self._success(
            session_id=session_id,
            reason=PrewarmReason.LLM_CONNECTION_READY,
            pool=pool,
            message="LLM primary and standby connections are warm",
        )

    def warm_tts_pool(self, session_id: str) -> PrewarmResult:
        state = self.state_for(session_id)

        if state is None:
            return self._missing_session(session_id)

        resources = tuple(
            WarmResource(
                target=PrewarmTarget.TTS_ENGINE,
                role=WarmResourceRole.WORKER,
                name=f"tts_worker_{index}",
                metadata={
                    "model_loaded": True,
                    "thread_ready": True,
                    "real_synthesis": False,
                },
            )
            for index in range(self._config.tts_worker_pool_size)
        )
        pool = self._pool(target=PrewarmTarget.TTS_ENGINE, resources=resources)
        self._store_pool(session_id=session_id, pool=pool)

        for pattern in ("hello", "okay", "yes", "one moment"):
            self._hot_cache.set(f"phoneme:{pattern}", f"PHONEME::{pattern}")

        self._append_event(
            session_id=session_id,
            kind=PrewarmEventKind.TTS_POOL_READY,
            reason=PrewarmReason.TTS_ENGINE_READY,
            target=PrewarmTarget.TTS_ENGINE,
        )
        self._append_event(
            session_id=session_id,
            kind=PrewarmEventKind.RESOURCE_WARMED,
            reason=PrewarmReason.PHONEME_CACHE_READY,
            target=PrewarmTarget.TTS_ENGINE,
        )

        return self._success(
            session_id=session_id,
            reason=PrewarmReason.TTS_ENGINE_READY,
            pool=pool,
            message="TTS model and synthesis workers are warm",
        )

    def warm_audio_playback(self, session_id: str) -> PrewarmResult:
        state = self.state_for(session_id)

        if state is None:
            return self._missing_session(session_id)

        resources = (
            WarmResource(
                target=PrewarmTarget.AUDIO_PLAYBACK,
                role=WarmResourceRole.BUFFER,
                name="audio_output_stream",
                metadata={
                    "stream_open": True,
                    "prebuffer_ms": self._config.audio_prebuffer_ms,
                    "sample_rate_conversion_precomputed": True,
                    "real_playback": False,
                },
            ),
        )
        pool = self._pool(target=PrewarmTarget.AUDIO_PLAYBACK, resources=resources)
        self._store_pool(session_id=session_id, pool=pool)

        self._append_event(
            session_id=session_id,
            kind=PrewarmEventKind.AUDIO_BUFFER_READY,
            reason=PrewarmReason.AUDIO_PREBUFFER_READY,
            target=PrewarmTarget.AUDIO_PLAYBACK,
            resource_id=resources[0].resource_id,
        )

        return self._success(
            session_id=session_id,
            reason=PrewarmReason.AUDIO_STREAM_READY,
            pool=pool,
            message="audio playback stream and prebuffer are warm",
        )

    def warm_memory_cache(self, session_id: str) -> PrewarmResult:
        state = self.state_for(session_id)

        if state is None:
            return self._missing_session(session_id)

        if self._config.profile_pinned:
            self._hot_cache.set("profile:user", "User profile hot cache pinned.")

        for index in range(self._config.last_turn_cache_size):
            self._hot_cache.set(f"turn:{index}", f"recent turn {index}")

        resources = (
            WarmResource(
                target=PrewarmTarget.MEMORY_HOT_CACHE,
                role=WarmResourceRole.CACHE,
                name="memory_hot_cache",
                metadata={
                    "profile_pinned": self._config.profile_pinned,
                    "last_turns": self._config.last_turn_cache_size,
                    "disk_fetch_per_turn": False,
                },
            ),
        )
        pool = self._pool(target=PrewarmTarget.MEMORY_HOT_CACHE, resources=resources)
        self._store_pool(session_id=session_id, pool=pool)

        self._append_event(
            session_id=session_id,
            kind=PrewarmEventKind.MEMORY_CACHE_READY,
            reason=PrewarmReason.MEMORY_PROFILE_READY,
            target=PrewarmTarget.MEMORY_HOT_CACHE,
            resource_id=resources[0].resource_id,
        )
        self._append_event(
            session_id=session_id,
            kind=PrewarmEventKind.MEMORY_CACHE_READY,
            reason=PrewarmReason.MEMORY_TURNS_READY,
            target=PrewarmTarget.MEMORY_HOT_CACHE,
            resource_id=resources[0].resource_id,
        )

        return self._success(
            session_id=session_id,
            reason=PrewarmReason.MEMORY_PROFILE_READY,
            pool=pool,
            message="user profile and last turns are hot",
        )

    def warm_workspace_index(self, session_id: str) -> PrewarmResult:
        state = self.state_for(session_id)

        if state is None:
            return self._missing_session(session_id)

        for path in ("jarvis/", "tests/", "scripts/"):
            self._hot_cache.set(f"workspace:{path}", f"INDEXED::{path}")

        resources = (
            WarmResource(
                target=PrewarmTarget.WORKSPACE_INDEX,
                role=WarmResourceRole.CACHE,
                name="workspace_path_index",
                metadata={
                    "paths": ("jarvis/", "tests/", "scripts/"),
                    "preindexed": True,
                    "real_file_scan": False,
                },
            ),
        )
        pool = self._pool(target=PrewarmTarget.WORKSPACE_INDEX, resources=resources)
        self._store_pool(session_id=session_id, pool=pool)

        self._append_event(
            session_id=session_id,
            kind=PrewarmEventKind.WORKSPACE_INDEX_READY,
            reason=PrewarmReason.WORKSPACE_INDEX_READY,
            target=PrewarmTarget.WORKSPACE_INDEX,
            resource_id=resources[0].resource_id,
        )

        return self._success(
            session_id=session_id,
            reason=PrewarmReason.WORKSPACE_INDEX_READY,
            pool=pool,
            message="workspace paths are pre-indexed",
        )

    def send_idle_ping(self, session_id: str) -> PrewarmResult:
        state = self.state_for(session_id)

        if state is None:
            return self._missing_session(session_id)

        pool = self._pools.get(PrewarmTarget.LLM_CONNECTION)

        if pool is None:
            return self._failure(
                session_id=session_id,
                reason=PrewarmReason.SESSION_NOT_FOUND,
                status=state.status,
                message="LLM connection pool is not warm",
                state=state,
            )

        now_ns = time.perf_counter_ns()
        resources = tuple(
            resource.model_copy(update={"last_ping_at_ns": now_ns})
            for resource in pool.resources
        )
        updated_pool = pool.model_copy(update={"resources": resources})
        self._pools[PrewarmTarget.LLM_CONNECTION] = updated_pool

        with self._lock:
            current = self._sessions[session_id]
            self._sessions[session_id] = current.model_copy(
                update={"ping_count": current.ping_count + len(resources)}
            )

        self._append_event(
            session_id=session_id,
            kind=PrewarmEventKind.IDLE_PING_SENT,
            reason=PrewarmReason.IDLE_PING_SENT,
            target=PrewarmTarget.LLM_CONNECTION,
            metadata={"pinged_connections": len(resources)},
        )

        return self._success(
            session_id=session_id,
            reason=PrewarmReason.IDLE_PING_SENT,
            pool=updated_pool,
            message="idle keep-alive ping sent",
        )

    def complete_session(self, session_id: str) -> PrewarmReport:
        state = self.state_for(session_id)

        if state is None:
            raise ValueError(f"prewarm session not found: {session_id}")

        if state.status not in {PrewarmStatus.WARMING, PrewarmStatus.WARM}:
            raise ValueError("prewarm session cannot complete from current state")

        with self._lock:
            current = self._sessions[session_id]
            completed = current.model_copy(
                update={
                    "status": PrewarmStatus.WARM,
                    "completed_at_ns": time.perf_counter_ns(),
                    "warmed_target_count": len(self._pools),
                    "warmed_resource_count": sum(
                        len(pool.resources) for pool in self._pools.values()
                    ),
                }
            )
            self._sessions[session_id] = completed
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=PrewarmEventKind.SESSION_COMPLETED,
                    reason=PrewarmReason.SESSION_COMPLETED,
                )
            )
            self._last_reason = PrewarmReason.SESSION_COMPLETED

        self._record_prewarm_span(completed)

        profiler_report = None

        if self._config.profile_prewarming:
            profiler_report = self._profiler.complete_trace(completed.trace_id)

        report = PrewarmReport(
            session_id=session_id,
            trace_id=completed.trace_id,
            status=completed.status,
            warmed_target_count=completed.warmed_target_count,
            warmed_resource_count=completed.warmed_resource_count,
            ping_count=completed.ping_count,
            total_latency_ms=completed.total_latency_ms(),
            pools=self.pools(),
            events=self.events_for(session_id),
            profiler_report=profiler_report,
        )

        with self._lock:
            self._reports.append(report)

        return report

    def invalidate_hot_cache(self, session_id: str) -> PrewarmResult:
        state = self.state_for(session_id)

        if state is None:
            return self._missing_session(session_id)

        self._hot_cache.invalidate_all()
        self._append_event(
            session_id=session_id,
            kind=PrewarmEventKind.CACHE_INVALIDATED,
            reason=PrewarmReason.CACHE_INVALIDATED,
        )

        return self._success(
            session_id=session_id,
            reason=PrewarmReason.CACHE_INVALIDATED,
            message="hot cache invalidated",
        )

    def cancel_session(self, session_id: str) -> PrewarmResult:
        with self._lock:
            state = self._sessions.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            cancelled = state.model_copy(
                update={
                    "status": PrewarmStatus.CANCELLED,
                    "cancelled_at_ns": time.perf_counter_ns(),
                }
            )
            self._sessions[session_id] = cancelled
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=PrewarmEventKind.SESSION_CANCELLED,
                    reason=PrewarmReason.SESSION_CANCELLED,
                )
            )
            self._last_reason = PrewarmReason.SESSION_CANCELLED

        return PrewarmResult(
            success=True,
            reason=PrewarmReason.SESSION_CANCELLED,
            session_id=session_id,
            status=PrewarmStatus.CANCELLED,
            state=cancelled,
            message="prewarm session cancelled",
        )

    def fail_session(self, session_id: str, *, error: str) -> PrewarmResult:
        with self._lock:
            state = self._sessions.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            failed = state.model_copy(
                update={
                    "status": PrewarmStatus.FAILED,
                    "failed_at_ns": time.perf_counter_ns(),
                }
            )
            self._sessions[session_id] = failed
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=PrewarmEventKind.SESSION_FAILED,
                    reason=PrewarmReason.SESSION_FAILED,
                    metadata={"error": error},
                )
            )
            self._last_reason = PrewarmReason.SESSION_FAILED

        return PrewarmResult(
            success=True,
            reason=PrewarmReason.SESSION_FAILED,
            session_id=session_id,
            status=PrewarmStatus.FAILED,
            state=failed,
            message="prewarm session failed",
        )

    def state_for(self, session_id: str) -> PrewarmSessionState | None:
        with self._lock:
            return self._sessions.get(session_id)

    def pool_for(self, target: PrewarmTarget) -> PrewarmPoolState | None:
        with self._lock:
            return self._pools.get(target)

    def pools(self) -> tuple[PrewarmPoolState, ...]:
        with self._lock:
            return tuple(self._pools.values())

    def events_for(self, session_id: str) -> tuple[PrewarmEvent, ...]:
        with self._lock:
            return tuple(self._events.get(session_id, ()))

    def reports(self) -> tuple[PrewarmReport, ...]:
        with self._lock:
            return tuple(self._reports)

    def latest_report(self) -> PrewarmReport | None:
        with self._lock:
            if not self._reports:
                return None

            return self._reports[-1]

    def hot_cache_size(self) -> int:
        return self._hot_cache.size()

    def snapshot(self) -> PrewarmRuntimeSnapshot:
        with self._lock:
            return PrewarmRuntimeSnapshot(
                name=self.name,
                session_count=len(self._sessions),
                warm_pool_count=sum(
                    1 
                    for pool in self._pools.values()
                    if pool.status == PrewarmStatus.WARM
                ),
                warm_resource_count=sum(
                    pool.warm_count for pool in self._pools.values()
                ),
                report_count=len(self._reports),
                cache_size=self._hot_cache.size(),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        with self._lock:
            self._sessions.clear()
            self._pools.clear()
            self._events.clear()
            self._reports.clear()
            self._hot_cache.invalidate_all()
            self._last_reason = PrewarmReason.RUNTIME_RESET

    def _store_pool(self, *, session_id: str, pool: PrewarmPoolState) -> None:
        with self._lock:
            self._pools[pool.target] = pool
            current = self._sessions[session_id]
            self._sessions[session_id] = current.model_copy(
                update={
                    "status": PrewarmStatus.WARMING,
                    "warmed_target_count": len(self._pools),
                    "warmed_resource_count": sum(
                        len(item.resources) for item in self._pools.values()
                    ),
                }
            )
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=PrewarmEventKind.TARGET_WARMING_STARTED,
                    reason=PrewarmReason.TARGET_WARMING_STARTED,
                    target=pool.target,
                )
            )

            for resource in pool.resources:
                self._events[session_id].append(
                    self._event(
                        session_id=session_id,
                        kind=PrewarmEventKind.RESOURCE_WARMED,
                        reason=self._reason_for_target(pool.target),
                        target=pool.target,
                        resource_id=resource.resource_id,
                    )
                )

            self._last_reason = self._reason_for_target(pool.target)

    @staticmethod
    def _pool(
        *,
        target: PrewarmTarget,
        resources: tuple[WarmResource, ...],
    ) -> PrewarmPoolState:
        return PrewarmPoolState(
            target=target,
            status=PrewarmStatus.WARM,
            resources=resources,
            warmed_at_ns=time.perf_counter_ns(),
            metadata={"resource_count": len(resources)},
        )

    def _success(
        self,
        *,
        session_id: str,
        reason: PrewarmReason,
        message: str,
        pool: PrewarmPoolState | None = None,
        resource: WarmResource | None = None,
    ) -> PrewarmResult:
        return PrewarmResult(
            success=True,
            reason=reason,
            session_id=session_id,
            status=PrewarmStatus.WARM,
            pool=pool,
            resource=resource,
            state=self.state_for(session_id),
            message=message,
        )

    @staticmethod
    def _reason_for_target(target: PrewarmTarget) -> PrewarmReason:
        if target == PrewarmTarget.LLM_CONNECTION:
            return PrewarmReason.LLM_CONNECTION_READY

        if target == PrewarmTarget.TTS_ENGINE:
            return PrewarmReason.TTS_ENGINE_READY

        if target == PrewarmTarget.AUDIO_PLAYBACK:
            return PrewarmReason.AUDIO_STREAM_READY

        if target == PrewarmTarget.MEMORY_HOT_CACHE:
            return PrewarmReason.MEMORY_PROFILE_READY

        return PrewarmReason.WORKSPACE_INDEX_READY

    def _append_event(
        self,
        *,
        session_id: str,
        kind: PrewarmEventKind,
        reason: PrewarmReason,
        target: PrewarmTarget | None = None,
        resource_id: str | None = None,
        latency_ms: float | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        with self._lock:
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=kind,
                    reason=reason,
                    target=target,
                    resource_id=resource_id,
                    latency_ms=latency_ms,
                    metadata=metadata,
                )
            )
            self._last_reason = reason

    def _record_prewarm_span(self, state: PrewarmSessionState) -> None:
        if state.completed_at_ns is None:
            return

        self._profiler.record_stage(
            trace_id=state.trace_id,
            stage=PipelineStage.CACHE_LOOKUP,
            operation=LatencyOperation.CONTEXT_BUILD,
            subsystem=LatencySubsystem.COGNITION,
            start_ns=state.started_at_ns,
            end_ns=state.completed_at_ns,
            metadata={"session_id": state.session_id, "prewarm": True},
        )

    @staticmethod
    def _event(
        *,
        session_id: str,
        kind: PrewarmEventKind,
        reason: PrewarmReason,
        target: PrewarmTarget | None = None,
        resource_id: str | None = None,
        latency_ms: float | None = None,
        metadata: dict[str, object] | None = None,
    ) -> PrewarmEvent:
        return PrewarmEvent(
            session_id=session_id,
            kind=kind,
            reason=reason,
            target=target,
            resource_id=resource_id,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )

    @staticmethod
    def _missing_session(session_id: str) -> PrewarmResult:
        return PrewarmResult(
            success=False,
            reason=PrewarmReason.SESSION_NOT_FOUND,
            session_id=session_id,
            status=PrewarmStatus.FAILED,
            message="prewarm session not found",
        )

    @staticmethod
    def _failure(
        *,
        session_id: str,
        reason: PrewarmReason,
        status: PrewarmStatus,
        message: str,
        state: PrewarmSessionState | None = None,
    ) -> PrewarmResult:
        return PrewarmResult(
            success=False,
            reason=reason,
            session_id=session_id,
            status=status,
            state=state,
            message=message,
        )