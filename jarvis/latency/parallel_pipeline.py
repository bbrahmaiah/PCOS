from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from uuid import uuid4

from pydantic import Field, field_validator

from jarvis.latency.models import LatencyOperation, LatencySubsystem
from jarvis.latency.predictive_context import (
    ContextBuildRequest,
    PredictiveContextBuilderRuntime,
)
from jarvis.latency.profiler import (
    PipelineLatencyProfiler,
    PipelineProfilerConfig,
    PipelineProfilerReport,
    PipelineStage,
)
from jarvis.latency.streaming_memory import MemoryStreamResult, StreamingMemoryRuntime
from jarvis.latency.streaming_stt import (
    AudioChunkKind,
    PartialIntent,
    StreamingSTTRuntime,
    audio_chunk_metadata,
)
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class ParallelPipelineStatus(StrEnum):
    """
    Parallel pipeline lifecycle status.
    """

    CREATED = "created"
    RUNNING = "running"
    WAITING_FOR_CONTEXT = "waiting_for_context"
    READY_FOR_LLM = "ready_for_llm"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ParallelPipelineStageKind(StrEnum):
    """
    Major overlapped pipeline stage.
    """

    STT_STREAMING = "stt_streaming"
    INTENT_CLASSIFICATION = "intent_classification"
    MEMORY_RETRIEVAL = "memory_retrieval"
    CONTEXT_BUILD = "context_build"
    LLM_FIRST_TOKEN = "llm_first_token"


class ParallelPipelineEventKind(StrEnum):
    """
    Parallel pipeline event kind.
    """

    SESSION_CREATED = "session_created"
    SESSION_STARTED = "session_started"
    STT_PARTIAL_READY = "stt_partial_ready"
    PARALLEL_BRANCHES_STARTED = "parallel_branches_started"
    INTENT_READY = "intent_ready"
    MEMORY_READY = "memory_ready"
    CONTEXT_READY = "context_ready"
    LLM_STARTED = "llm_started"
    LLM_FIRST_TOKEN_READY = "llm_first_token_ready"
    LATENCY_SAVINGS_COMPUTED = "latency_savings_computed"
    SESSION_COMPLETED = "session_completed"
    SESSION_CANCELLED = "session_cancelled"
    SESSION_FAILED = "session_failed"


class ParallelPipelineReason(StrEnum):
    """
    Machine-readable parallel pipeline reasons.
    """

    SESSION_CREATED = "session_created"
    SESSION_STARTED = "session_started"
    STT_PARTIAL_CONFIDENCE_LOW = "stt_partial_confidence_low"
    STT_PARTIAL_CONFIDENCE_ACCEPTED = "stt_partial_confidence_accepted"
    PARALLEL_BRANCHES_STARTED = "parallel_branches_started"
    INTENT_READY = "intent_ready"
    MEMORY_STREAM_READY = "memory_stream_ready"
    CONTEXT_READY = "context_ready"
    LLM_STARTED_WITH_PREBUILT_CONTEXT = "llm_started_with_prebuilt_context"
    LLM_FIRST_TOKEN_READY = "llm_first_token_ready"
    LATENCY_SAVINGS_COMPUTED = "latency_savings_computed"
    SESSION_COMPLETED = "session_completed"
    SESSION_CANCELLED = "session_cancelled"
    SESSION_FAILED = "session_failed"
    SESSION_NOT_FOUND = "session_not_found"
    SESSION_NOT_ACTIVE = "session_not_active"
    RUNTIME_RESET = "runtime_reset"


class PipelineBranchStatus(StrEnum):
    """
    Branch status for overlapped work.
    """

    NOT_STARTED = "not_started"
    RUNNING = "running"
    READY = "ready"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ParallelStageTiming(OrchestrationModel):
    """
    Timing for one pipeline stage.
    """

    timing_id: str = Field(default_factory=lambda: uuid4().hex)
    stage: ParallelPipelineStageKind
    status: PipelineBranchStatus = PipelineBranchStatus.NOT_STARTED
    started_at_ns: int | None = None
    completed_at_ns: int | None = None
    expected_serial_ms: float = Field(default=0.0, ge=0)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("timing_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("timing_id cannot be empty.")

        return cleaned

    def duration_ms(self) -> float | None:
        if self.started_at_ns is None or self.completed_at_ns is None:
            return None

        return (self.completed_at_ns - self.started_at_ns) / 1_000_000.0


class ParallelPipelineEvent(OrchestrationModel):
    """
    Event emitted by the parallel pipeline runtime.
    """

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    kind: ParallelPipelineEventKind
    reason: ParallelPipelineReason
    stage: ParallelPipelineStageKind | None = None
    latency_ms: float | None = None
    confidence: float | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("event_id", "session_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class ParallelPipelineSessionState(OrchestrationModel):
    """
    State for one parallel pipeline run.
    """

    session_id: str = Field(default_factory=lambda: uuid4().hex)
    trace_id: str = Field(default_factory=lambda: uuid4().hex)
    turn_id: str
    status: ParallelPipelineStatus = ParallelPipelineStatus.CREATED
    started_at_ns: int | None = None
    partial_ready_at_ns: int | None = None
    branches_started_at_ns: int | None = None
    context_ready_at_ns: int | None = None
    llm_started_at_ns: int | None = None
    first_token_at_ns: int | None = None
    completed_at_ns: int | None = None
    cancelled_at_ns: int | None = None
    failed_at_ns: int | None = None
    partial_confidence: float = Field(default=0.0, ge=0, le=1)
    ready_for_llm: bool = False
    serial_latency_ms: float = Field(default=0.0, ge=0)
    parallel_latency_ms: float = Field(default=0.0, ge=0)
    savings_ms: float = Field(default=0.0, ge=0)
    savings_ratio: float = Field(default=0.0, ge=0, le=1)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("session_id", "trace_id", "turn_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    def first_token_latency_ms(self) -> float | None:
        if self.started_at_ns is None or self.first_token_at_ns is None:
            return None

        return (self.first_token_at_ns - self.started_at_ns) / 1_000_000.0


class ParallelPipelineResult(OrchestrationModel):
    """
    Result from a parallel pipeline operation.
    """

    success: bool
    reason: ParallelPipelineReason
    session_id: str
    status: ParallelPipelineStatus
    event: ParallelPipelineEvent | None = None
    state: ParallelPipelineSessionState | None = None
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


class ParallelPipelineReport(OrchestrationModel):
    """
    Final report for a parallel pipeline run.
    """

    session_id: str
    trace_id: str
    turn_id: str
    status: ParallelPipelineStatus
    partial_confidence: float = Field(ge=0, le=1)
    ready_for_llm: bool
    serial_latency_ms: float = Field(ge=0)
    parallel_latency_ms: float = Field(ge=0)
    savings_ms: float = Field(ge=0)
    savings_ratio: float = Field(ge=0, le=1)
    stage_timings: tuple[ParallelStageTiming, ...]
    events: tuple[ParallelPipelineEvent, ...]
    profiler_report: PipelineProfilerReport | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("session_id", "trace_id", "turn_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class ParallelPipelineRuntimeSnapshot(OrchestrationModel):
    """
    Runtime diagnostics for Phase 7 Step 13.
    """

    name: str
    session_count: int = Field(ge=0)
    running_count: int = Field(ge=0)
    ready_count: int = Field(ge=0)
    completed_count: int = Field(ge=0)
    cancelled_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    report_count: int = Field(ge=0)
    average_savings_ms: float = Field(ge=0)
    last_reason: ParallelPipelineReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class ParallelPipelineRuntimeConfig:
    """
    Phase 7 Step 13 parallel pipeline configuration.
    """

    name: str = "parallel_pipeline_orchestration"
    partial_confidence_threshold: float = 0.60
    simulated_stt_ms: float = 150.0
    simulated_memory_ms: float = 150.0
    simulated_context_ms: float = 80.0
    simulated_llm_first_token_ms: float = 300.0
    profile_parallel_pipeline: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if not 0 <= self.partial_confidence_threshold <= 1:
            raise ValueError("partial_confidence_threshold must be within 0..1.")

        for field_name, value in (
            ("simulated_stt_ms", self.simulated_stt_ms),
            ("simulated_memory_ms", self.simulated_memory_ms),
            ("simulated_context_ms", self.simulated_context_ms),
            ("simulated_llm_first_token_ms", self.simulated_llm_first_token_ms),
        ):
            if value <= 0:
                raise ValueError(f"{field_name} must be positive.")


class ParallelPipelineRuntime:
    """
    Phase 7 Step 13 Parallel Pipeline Orchestration.

    Responsibilities:
    - start STT streaming
    - gate speculative parallel work on partial confidence
    - overlap memory retrieval, context building, and intent classification
    - start LLM first-token path when prebuilt context is ready
    - compute serial-vs-parallel latency savings
    - expose deterministic reports for regression tests

    Non-responsibilities:
    - no real LLM calls
    - no real audio capture
    - no real tool/action execution
    - no hidden memory writes
    """

    def __init__(
        self,
        *,
        config: ParallelPipelineRuntimeConfig | None = None,
        stt_runtime: StreamingSTTRuntime | None = None,
        memory_runtime: StreamingMemoryRuntime | None = None,
        context_runtime: PredictiveContextBuilderRuntime | None = None,
        profiler: PipelineLatencyProfiler | None = None,
    ) -> None:
        self._config = config or ParallelPipelineRuntimeConfig()
        self._config.validate()

        self._stt_runtime = stt_runtime or StreamingSTTRuntime()
        self._memory_runtime = memory_runtime or StreamingMemoryRuntime()
        self._context_runtime = context_runtime or PredictiveContextBuilderRuntime()
        self._profiler = profiler or PipelineLatencyProfiler(
            config=PipelineProfilerConfig(record_to_latency_runtime=True)
        )

        self._states: dict[str, ParallelPipelineSessionState] = {}
        self._stage_timings: dict[
            str,
            dict[ParallelPipelineStageKind, ParallelStageTiming],
        ]
        self._stage_timings = {}
        self._events: dict[str, list[ParallelPipelineEvent]] = {}
        self._reports: list[ParallelPipelineReport] = []
        self._lock = RLock()
        self._last_reason: ParallelPipelineReason | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def create_session(
        self,
        *,
        turn_id: str,
        trace_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> ParallelPipelineSessionState:
        state = ParallelPipelineSessionState(
            trace_id=trace_id or uuid4().hex,
            turn_id=turn_id,
            metadata=metadata or {},
        )
        event = self._event(
            session_id=state.session_id,
            kind=ParallelPipelineEventKind.SESSION_CREATED,
            reason=ParallelPipelineReason.SESSION_CREATED,
        )

        with self._lock:
            self._states[state.session_id] = state
            self._stage_timings[state.session_id] = self._initial_timings()
            self._events[state.session_id] = [event]
            self._last_reason = ParallelPipelineReason.SESSION_CREATED

        self._profiler.start_trace(
            name="parallel_pipeline_orchestration",
            trace_id=state.trace_id,
        )

        return state

    def start_session(self, session_id: str) -> ParallelPipelineResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            if state.status != ParallelPipelineStatus.CREATED:
                return self._failure(
                    session_id=session_id,
                    reason=ParallelPipelineReason.SESSION_NOT_ACTIVE,
                    status=state.status,
                    message="parallel pipeline cannot start from current state",
                    state=state,
                )

            started = state.model_copy(
                update={
                    "status": ParallelPipelineStatus.RUNNING,
                    "started_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = started
            self._start_stage(session_id, ParallelPipelineStageKind.STT_STREAMING)
            event = self._event(
                session_id=session_id,
                kind=ParallelPipelineEventKind.SESSION_STARTED,
                reason=ParallelPipelineReason.SESSION_STARTED,
            )
            self._events[session_id].append(event)
            self._last_reason = ParallelPipelineReason.SESSION_STARTED

        return ParallelPipelineResult(
            success=True,
            reason=ParallelPipelineReason.SESSION_STARTED,
            session_id=session_id,
            status=ParallelPipelineStatus.RUNNING,
            event=event,
            state=started,
            message="parallel pipeline session started",
        )

    def accept_partial_transcript(
        self,
        *,
        session_id: str,
        partial_text: str,
        confidence: float,
    ) -> ParallelPipelineResult:
        state = self.state_for(session_id)

        if state is None:
            return self._missing_session(session_id)

        if state.status != ParallelPipelineStatus.RUNNING:
            return self._failure(
                session_id=session_id,
                reason=ParallelPipelineReason.SESSION_NOT_ACTIVE,
                status=state.status,
                message="parallel pipeline is not running",
                state=state,
            )

        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be within 0..1.")

        now_ns = time.perf_counter_ns()
        reason = (
            ParallelPipelineReason.STT_PARTIAL_CONFIDENCE_ACCEPTED
            if confidence >= self._config.partial_confidence_threshold
            else ParallelPipelineReason.STT_PARTIAL_CONFIDENCE_LOW
        )

        with self._lock:
            current = self._states[session_id]
            updated = current.model_copy(
                update={
                    "partial_ready_at_ns": now_ns,
                    "partial_confidence": confidence,
                }
            )
            self._states[session_id] = updated
            self._complete_stage(session_id, ParallelPipelineStageKind.STT_STREAMING)
            event = self._event(
                session_id=session_id,
                kind=ParallelPipelineEventKind.STT_PARTIAL_READY,
                reason=reason,
                stage=ParallelPipelineStageKind.STT_STREAMING,
                confidence=confidence,
            )
            self._events[session_id].append(event)
            self._last_reason = reason

        if confidence < self._config.partial_confidence_threshold:
            return ParallelPipelineResult(
                success=False,
                reason=reason,
                session_id=session_id,
                status=ParallelPipelineStatus.RUNNING,
                event=event,
                state=updated,
                message="partial transcript confidence below parallel start gate",
            )

        return self._start_parallel_branches(
            session_id=session_id,
            partial_text=partial_text,
        )

    def complete_llm_first_token(self, session_id: str) -> ParallelPipelineResult:
        state = self.state_for(session_id)

        if state is None:
            return self._missing_session(session_id)

        if state.status != ParallelPipelineStatus.READY_FOR_LLM:
            return self._failure(
                session_id=session_id,
                reason=ParallelPipelineReason.SESSION_NOT_ACTIVE,
                status=state.status,
                message="pipeline is not ready for LLM first token",
                state=state,
            )

        with self._lock:
            current = self._states[session_id]
            now_ns = time.perf_counter_ns()
            self._complete_stage(
                session_id,
                ParallelPipelineStageKind.LLM_FIRST_TOKEN,
            )
            serial_ms = self._serial_latency_ms()
            parallel_ms = self._parallel_latency_ms(session_id)
            savings_ms = max(0.0, serial_ms - parallel_ms)
            savings_ratio = savings_ms / serial_ms if serial_ms > 0 else 0.0
            updated = current.model_copy(
                update={
                    "first_token_at_ns": now_ns,
                    "serial_latency_ms": serial_ms,
                    "parallel_latency_ms": parallel_ms,
                    "savings_ms": savings_ms,
                    "savings_ratio": savings_ratio,
                }
            )
            self._states[session_id] = updated
            event = self._event(
                session_id=session_id,
                kind=ParallelPipelineEventKind.LLM_FIRST_TOKEN_READY,
                reason=ParallelPipelineReason.LLM_FIRST_TOKEN_READY,
                stage=ParallelPipelineStageKind.LLM_FIRST_TOKEN,
                latency_ms=updated.first_token_latency_ms(),
            )
            self._events[session_id].append(event)
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=ParallelPipelineEventKind.LATENCY_SAVINGS_COMPUTED,
                    reason=ParallelPipelineReason.LATENCY_SAVINGS_COMPUTED,
                    metadata={
                        "serial_latency_ms": serial_ms,
                        "parallel_latency_ms": parallel_ms,
                        "savings_ms": savings_ms,
                        "savings_ratio": savings_ratio,
                    },
                )
            )
            self._last_reason = ParallelPipelineReason.LLM_FIRST_TOKEN_READY

        return ParallelPipelineResult(
            success=True,
            reason=ParallelPipelineReason.LLM_FIRST_TOKEN_READY,
            session_id=session_id,
            status=ParallelPipelineStatus.READY_FOR_LLM,
            event=event,
            state=updated,
            message="LLM first token ready with prebuilt context",
        )

    def complete_session(self, session_id: str) -> ParallelPipelineReport:
        state = self.state_for(session_id)

        if state is None:
            raise ValueError(f"parallel pipeline session not found: {session_id}")

        if state.status != ParallelPipelineStatus.READY_FOR_LLM:
            raise ValueError("parallel pipeline cannot complete from current state")

        with self._lock:
            current = self._states[session_id]
            completed = current.model_copy(
                update={
                    "status": ParallelPipelineStatus.COMPLETED,
                    "completed_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = completed
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=ParallelPipelineEventKind.SESSION_COMPLETED,
                    reason=ParallelPipelineReason.SESSION_COMPLETED,
                )
            )
            self._last_reason = ParallelPipelineReason.SESSION_COMPLETED

        self._record_pipeline_span(completed)

        profiler_report = None

        if self._config.profile_parallel_pipeline:
            profiler_report = self._profiler.complete_trace(completed.trace_id)

        report = ParallelPipelineReport(
            session_id=session_id,
            trace_id=completed.trace_id,
            turn_id=completed.turn_id,
            status=completed.status,
            partial_confidence=completed.partial_confidence,
            ready_for_llm=completed.ready_for_llm,
            serial_latency_ms=completed.serial_latency_ms,
            parallel_latency_ms=completed.parallel_latency_ms,
            savings_ms=completed.savings_ms,
            savings_ratio=completed.savings_ratio,
            stage_timings=self.stage_timings_for(session_id),
            events=self.events_for(session_id),
            profiler_report=profiler_report,
        )

        with self._lock:
            self._reports.append(report)

        return report

    def cancel_session(self, session_id: str) -> ParallelPipelineResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            cancelled = state.model_copy(
                update={
                    "status": ParallelPipelineStatus.CANCELLED,
                    "cancelled_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = cancelled

            for stage, timing in self._stage_timings[session_id].items():
                if timing.status == PipelineBranchStatus.RUNNING:
                    self._stage_timings[session_id][stage] = timing.model_copy(
                        update={"status": PipelineBranchStatus.CANCELLED}
                    )

            event = self._event(
                session_id=session_id,
                kind=ParallelPipelineEventKind.SESSION_CANCELLED,
                reason=ParallelPipelineReason.SESSION_CANCELLED,
            )
            self._events[session_id].append(event)
            self._last_reason = ParallelPipelineReason.SESSION_CANCELLED

        return ParallelPipelineResult(
            success=True,
            reason=ParallelPipelineReason.SESSION_CANCELLED,
            session_id=session_id,
            status=ParallelPipelineStatus.CANCELLED,
            event=event,
            state=cancelled,
            message="parallel pipeline session cancelled",
        )

    def fail_session(
        self,
        session_id: str,
        *,
        error: str,
    ) -> ParallelPipelineResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            failed = state.model_copy(
                update={
                    "status": ParallelPipelineStatus.FAILED,
                    "failed_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = failed
            event = self._event(
                session_id=session_id,
                kind=ParallelPipelineEventKind.SESSION_FAILED,
                reason=ParallelPipelineReason.SESSION_FAILED,
                metadata={"error": error},
            )
            self._events[session_id].append(event)
            self._last_reason = ParallelPipelineReason.SESSION_FAILED

        return ParallelPipelineResult(
            success=True,
            reason=ParallelPipelineReason.SESSION_FAILED,
            session_id=session_id,
            status=ParallelPipelineStatus.FAILED,
            event=event,
            state=failed,
            message="parallel pipeline session failed",
        )

    def state_for(self, session_id: str) -> ParallelPipelineSessionState | None:
        with self._lock:
            return self._states.get(session_id)

    def stage_timings_for(
        self,
        session_id: str,
    ) -> tuple[ParallelStageTiming, ...]:
        with self._lock:
            return tuple(self._stage_timings.get(session_id, {}).values())

    def events_for(self, session_id: str) -> tuple[ParallelPipelineEvent, ...]:
        with self._lock:
            return tuple(self._events.get(session_id, ()))

    def reports(self) -> tuple[ParallelPipelineReport, ...]:
        with self._lock:
            return tuple(self._reports)

    def latest_report(self) -> ParallelPipelineReport | None:
        with self._lock:
            if not self._reports:
                return None

            return self._reports[-1]

    def snapshot(self) -> ParallelPipelineRuntimeSnapshot:
        with self._lock:
            states = tuple(self._states.values())
            reports = tuple(self._reports)
            average_savings = (
                sum(report.savings_ms for report in reports) / len(reports)
                if reports
                else 0.0
            )

            return ParallelPipelineRuntimeSnapshot(
                name=self.name,
                session_count=len(states),
                running_count=sum(
                    1 
                    for state in states 
                    if state.status == ParallelPipelineStatus.RUNNING
                ),
                ready_count=sum(
                    1
                    for state in states
                    if state.status == ParallelPipelineStatus.READY_FOR_LLM
                ),
                completed_count=sum(
                    1
                    for state in states
                    if state.status == ParallelPipelineStatus.COMPLETED
                ),
                cancelled_count=sum(
                    1
                    for state in states
                    if state.status == ParallelPipelineStatus.CANCELLED
                ),
                failed_count=sum(
                    1 
                    for state in states 
                    if state.status == ParallelPipelineStatus.FAILED
                ),
                report_count=len(reports),
                average_savings_ms=average_savings,
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        with self._lock:
            self._states.clear()
            self._stage_timings.clear()
            self._events.clear()
            self._reports.clear()
            self._last_reason = ParallelPipelineReason.RUNTIME_RESET

    def _start_parallel_branches(
        self,
        *,
        session_id: str,
        partial_text: str,
    ) -> ParallelPipelineResult:
        branch_start_ns = time.perf_counter_ns()

        with self._lock:
            current = self._states[session_id]
            updated = current.model_copy(
                update={
                    "status": ParallelPipelineStatus.WAITING_FOR_CONTEXT,
                    "branches_started_at_ns": branch_start_ns,
                }
            )
            self._states[session_id] = updated
            self._start_stage(
                session_id,
                ParallelPipelineStageKind.INTENT_CLASSIFICATION,
            )
            self._start_stage(session_id, ParallelPipelineStageKind.MEMORY_RETRIEVAL)
            self._start_stage(session_id, ParallelPipelineStageKind.CONTEXT_BUILD)
            event = self._event(
                session_id=session_id,
                kind=ParallelPipelineEventKind.PARALLEL_BRANCHES_STARTED,
                reason=ParallelPipelineReason.PARALLEL_BRANCHES_STARTED,
                confidence=updated.partial_confidence,
            )
            self._events[session_id].append(event)

        intent = self._run_intent_branch(
            session_id=session_id,
            partial_text=partial_text,
        )
        memory_results = self._run_memory_branch(
            session_id=session_id,
            partial_text=partial_text,
        )
        self._run_context_branch(
            session_id=session_id,
            partial_text=partial_text,
            memory_results=memory_results,
        )

        with self._lock:
            current = self._states[session_id]
            llm_started_at_ns = time.perf_counter_ns()
            self._start_stage(session_id, ParallelPipelineStageKind.LLM_FIRST_TOKEN)
            ready = current.model_copy(
                update={
                    "status": ParallelPipelineStatus.READY_FOR_LLM,
                    "ready_for_llm": True,
                    "llm_started_at_ns": llm_started_at_ns,
                }
            )
            self._states[session_id] = ready
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=ParallelPipelineEventKind.LLM_STARTED,
                    reason=ParallelPipelineReason.LLM_STARTED_WITH_PREBUILT_CONTEXT,
                    stage=ParallelPipelineStageKind.LLM_FIRST_TOKEN,
                    metadata={"intent": intent.kind.value},
                )
            )
            self._last_reason = ParallelPipelineReason.LLM_STARTED_WITH_PREBUILT_CONTEXT

        return ParallelPipelineResult(
            success=True,
            reason=ParallelPipelineReason.PARALLEL_BRANCHES_STARTED,
            session_id=session_id,
            status=ParallelPipelineStatus.READY_FOR_LLM,
            event=event,
            state=ready,
            message="parallel memory/context/intent branches completed",
        )

    def _run_intent_branch(
        self,
        *,
        session_id: str,
        partial_text: str,
    ) -> PartialIntent:
        stt_session = self._stt_runtime.create_session(name=f"stt-{session_id}")
        self._stt_runtime.start_session(stt_session.session_id)
        result = self._stt_runtime.accept_audio_chunk(
            session_id=stt_session.session_id,
            duration_ms=120.0,
            kind=AudioChunkKind.SPEECH,
            metadata=audio_chunk_metadata(
                transcript=partial_text,
                confidence=self._states[session_id].partial_confidence,
            ),
        )

        if result.intent is None:
            raise RuntimeError("intent branch did not produce intent")

        with self._lock:
            self._complete_stage(
                session_id,
                ParallelPipelineStageKind.INTENT_CLASSIFICATION,
            )
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=ParallelPipelineEventKind.INTENT_READY,
                    reason=ParallelPipelineReason.INTENT_READY,
                    stage=ParallelPipelineStageKind.INTENT_CLASSIFICATION,
                    confidence=result.intent.confidence,
                    metadata={"intent": result.intent.kind.value},
                )
            )

        return result.intent

    def _run_memory_branch(
        self,
        *,
        session_id: str,
        partial_text: str,
    ) -> tuple[MemoryStreamResult, ...]:
        memory_session = self._memory_runtime.create_session(query_text=partial_text)
        self._memory_runtime.start_session(memory_session.session_id)
        self._memory_runtime.run_available_streams(memory_session.session_id)
        report = self._memory_runtime.complete_session(memory_session.session_id)

        with self._lock:
            self._complete_stage(session_id, ParallelPipelineStageKind.MEMORY_RETRIEVAL)
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=ParallelPipelineEventKind.MEMORY_READY,
                    reason=ParallelPipelineReason.MEMORY_STREAM_READY,
                    stage=ParallelPipelineStageKind.MEMORY_RETRIEVAL,
                    confidence=report.context_confidence,
                    metadata={"result_count": report.result_count},
                )
            )

        return report.results

    def _run_context_branch(
        self,
        *,
        session_id: str,
        partial_text: str,
        memory_results: tuple[MemoryStreamResult, ...],
    ) -> None:
        state = self._states[session_id]
        request = ContextBuildRequest(
            turn_id=state.turn_id,
            user_text=partial_text,
        )
        context_session = self._context_runtime.create_session(request=request)
        self._context_runtime.start_session(context_session.session_id)

        for memory_result in memory_results:
            self._context_runtime.consume_memory_result(
                session_id=context_session.session_id,
                result=memory_result,
            )

        report = self._context_runtime.complete_session(context_session.session_id)

        with self._lock:
            self._complete_stage(session_id, ParallelPipelineStageKind.CONTEXT_BUILD)
            now_ns = time.perf_counter_ns()
            current = self._states[session_id]
            self._states[session_id] = current.model_copy(
                update={"context_ready_at_ns": now_ns}
            )
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=ParallelPipelineEventKind.CONTEXT_READY,
                    reason=ParallelPipelineReason.CONTEXT_READY,
                    stage=ParallelPipelineStageKind.CONTEXT_BUILD,
                    confidence=report.context_confidence,
                    metadata={"fragment_count": report.fragment_count},
                )
            )

    def _initial_timings(self) -> dict[ParallelPipelineStageKind, ParallelStageTiming]:
        return {
            ParallelPipelineStageKind.STT_STREAMING: ParallelStageTiming(
                stage=ParallelPipelineStageKind.STT_STREAMING,
                expected_serial_ms=self._config.simulated_stt_ms,
            ),
            ParallelPipelineStageKind.INTENT_CLASSIFICATION: ParallelStageTiming(
                stage=ParallelPipelineStageKind.INTENT_CLASSIFICATION,
                expected_serial_ms=0.0,
            ),
            ParallelPipelineStageKind.MEMORY_RETRIEVAL: ParallelStageTiming(
                stage=ParallelPipelineStageKind.MEMORY_RETRIEVAL,
                expected_serial_ms=self._config.simulated_memory_ms,
            ),
            ParallelPipelineStageKind.CONTEXT_BUILD: ParallelStageTiming(
                stage=ParallelPipelineStageKind.CONTEXT_BUILD,
                expected_serial_ms=self._config.simulated_context_ms,
            ),
            ParallelPipelineStageKind.LLM_FIRST_TOKEN: ParallelStageTiming(
                stage=ParallelPipelineStageKind.LLM_FIRST_TOKEN,
                expected_serial_ms=self._config.simulated_llm_first_token_ms,
            ),
        }

    def _start_stage(
        self,
        session_id: str,
        stage: ParallelPipelineStageKind,
    ) -> None:
        timing = self._stage_timings[session_id][stage]
        self._stage_timings[session_id][stage] = timing.model_copy(
            update={
                "status": PipelineBranchStatus.RUNNING,
                "started_at_ns": time.perf_counter_ns(),
            }
        )

    def _complete_stage(
        self,
        session_id: str,
        stage: ParallelPipelineStageKind,
    ) -> None:
        timing = self._stage_timings[session_id][stage]
        started_at = timing.started_at_ns or time.perf_counter_ns()

        self._stage_timings[session_id][stage] = timing.model_copy(
            update={
                "status": PipelineBranchStatus.READY,
                "started_at_ns": started_at,
                "completed_at_ns": time.perf_counter_ns(),
            }
        )

    def _serial_latency_ms(self) -> float:
        return (
            self._config.simulated_stt_ms
            + self._config.simulated_memory_ms
            + self._config.simulated_context_ms
            + self._config.simulated_llm_first_token_ms
        )

    def _parallel_latency_ms(self, session_id: str) -> float:
        timings = self._stage_timings[session_id]
        pre_llm = max(
            self._config.simulated_stt_ms,
            self._config.simulated_memory_ms,
            self._config.simulated_context_ms,
        )
        llm = timings[
            ParallelPipelineStageKind.LLM_FIRST_TOKEN
        ].expected_serial_ms

        return pre_llm + llm

    def _record_pipeline_span(self, state: ParallelPipelineSessionState) -> None:
        if state.started_at_ns is None or state.completed_at_ns is None:
            return

        self._profiler.record_stage(
            trace_id=state.trace_id,
            stage=PipelineStage.STT_FIRST_PARTIAL,
            operation=LatencyOperation.STT_FIRST_TOKEN,
            subsystem=LatencySubsystem.PRESENCE,
            start_ns=state.started_at_ns,
            end_ns=state.completed_at_ns,
            metadata={
                "session_id": state.session_id,
                "parallel": True,
                "savings_ms": state.savings_ms,
            },
        )

    @staticmethod
    def _event(
        *,
        session_id: str,
        kind: ParallelPipelineEventKind,
        reason: ParallelPipelineReason,
        stage: ParallelPipelineStageKind | None = None,
        latency_ms: float | None = None,
        confidence: float | None = None,
        metadata: dict[str, object] | None = None,
    ) -> ParallelPipelineEvent:
        return ParallelPipelineEvent(
            session_id=session_id,
            kind=kind,
            reason=reason,
            stage=stage,
            latency_ms=latency_ms,
            confidence=confidence,
            metadata=metadata or {},
        )

    @staticmethod
    def _missing_session(session_id: str) -> ParallelPipelineResult:
        return ParallelPipelineResult(
            success=False,
            reason=ParallelPipelineReason.SESSION_NOT_FOUND,
            session_id=session_id,
            status=ParallelPipelineStatus.FAILED,
            message="parallel pipeline session not found",
        )

    @staticmethod
    def _failure(
        *,
        session_id: str,
        reason: ParallelPipelineReason,
        status: ParallelPipelineStatus,
        message: str,
        state: ParallelPipelineSessionState | None = None,
    ) -> ParallelPipelineResult:
        return ParallelPipelineResult(
            success=False,
            reason=reason,
            session_id=session_id,
            status=status,
            state=state,
            message=message,
        )