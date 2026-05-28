from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.latency.predictive_context import (
    ContextFragment,
    ContextFragmentKind,
    ContextSnapshot,
)
from jarvis.latency.profiler import (
    PipelineLatencyProfiler,
    PipelineProfilerConfig,
    PipelineProfilerReport,
    PipelineStage,
)
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class InterruptionRecoveryStatus(StrEnum):
    """
    Interruption recovery lifecycle.
    """

    CREATED = "created"
    TRACKING = "tracking"
    INTERRUPTED = "interrupted"
    RECONSTRUCTING = "reconstructing"
    READY = "ready"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class InterruptionRecoveryEventKind(StrEnum):
    """
    Interruption recovery event kind.
    """

    SESSION_CREATED = "session_created"
    SESSION_STARTED = "session_started"
    SNAPSHOT_CAPTURED = "snapshot_captured"
    INTERRUPT_DETECTED = "interrupt_detected"
    TTS_STOPPED = "tts_stopped"
    LLM_CANCELLED = "llm_cancelled"
    INTERRUPTION_CONTEXT_CAPTURED = "interruption_context_captured"
    SNAPSHOT_RESTORED = "snapshot_restored"
    DELTA_APPLIED = "delta_applied"
    CONTEXT_RECONSTRUCTED = "context_reconstructed"
    FIRST_NEW_WORD_READY = "first_new_word_ready"
    SESSION_COMPLETED = "session_completed"
    SESSION_CANCELLED = "session_cancelled"
    SESSION_FAILED = "session_failed"


class InterruptionRecoveryReason(StrEnum):
    """
    Machine-readable interruption recovery reasons.
    """

    SESSION_CREATED = "session_created"
    SESSION_STARTED = "session_started"
    SNAPSHOT_CAPTURED = "snapshot_captured"
    INTERRUPT_DETECTED = "interrupt_detected"
    TTS_STOPPED_WITHIN_BUDGET = "tts_stopped_within_budget"
    TTS_STOPPED_OVER_BUDGET = "tts_stopped_over_budget"
    LLM_CANCELLED_WITHIN_BUDGET = "llm_cancelled_within_budget"
    LLM_CANCELLED_OVER_BUDGET = "llm_cancelled_over_budget"
    CONTEXT_CAPTURED_WITHIN_BUDGET = "context_captured_within_budget"
    CONTEXT_CAPTURED_OVER_BUDGET = "context_captured_over_budget"
    SNAPSHOT_RESTORED = "snapshot_restored"
    DELTA_APPLIED = "delta_applied"
    CONTEXT_READY_WITHIN_BUDGET = "context_ready_within_budget"
    CONTEXT_READY_OVER_BUDGET = "context_ready_over_budget"
    FIRST_NEW_WORD_WITHIN_BUDGET = "first_new_word_within_budget"
    FIRST_NEW_WORD_OVER_BUDGET = "first_new_word_over_budget"
    SESSION_COMPLETED = "session_completed"
    SESSION_CANCELLED = "session_cancelled"
    SESSION_FAILED = "session_failed"
    SESSION_NOT_FOUND = "session_not_found"
    SESSION_NOT_ACTIVE = "session_not_active"
    NO_SNAPSHOT_AVAILABLE = "no_snapshot_available"
    RUNTIME_RESET = "runtime_reset"


class RecoverySnapshotKind(StrEnum):
    """
    Snapshot type.
    """

    PERIODIC = "periodic"
    INTERRUPT_CAPTURE = "interrupt_capture"
    RESTORED = "restored"


class InterruptionContextDelta(OrchestrationModel):
    """
    Delta applied after restoring nearest generation snapshot.
    """

    delta_id: str = Field(default_factory=lambda: uuid4().hex)
    interrupted_at_text: str
    user_new_utterance: str
    partial_assistant_utterance: str
    created_at_ns: int = Field(default_factory=time.perf_counter_ns, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("delta_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("delta_id cannot be empty.")

        return cleaned

    @model_validator(mode="after")
    def _requires_interruption_context(self) -> InterruptionContextDelta:
        if not self.interrupted_at_text.strip():
            raise ValueError("interrupted_at_text cannot be empty.")

        if not self.user_new_utterance.strip():
            raise ValueError("user_new_utterance cannot be empty.")

        return self


class RecoveryContextSnapshot(OrchestrationModel):
    """
    Runtime snapshot captured during active generation.

    Captured every 500ms while generation is active. On interrupt, the nearest
    snapshot is restored and patched with an interruption delta.
    """

    recovery_snapshot_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    turn_id: str
    kind: RecoverySnapshotKind = RecoverySnapshotKind.PERIODIC
    context_snapshot: ContextSnapshot
    assistant_partial_text: str = ""
    generation_sequence: int = Field(ge=0)
    captured_at_ns: int = Field(default_factory=time.perf_counter_ns, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("recovery_snapshot_id", "session_id", "turn_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class ReconstructedInterruptionContext(OrchestrationModel):
    """
    Context reconstructed after interruption.
    """

    reconstructed_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    source_snapshot_id: str
    delta_id: str
    context_snapshot: ContextSnapshot
    ready_at_ns: int = Field(default_factory=time.perf_counter_ns, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("reconstructed_id", "session_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class InterruptionRecoveryEvent(OrchestrationModel):
    """
    Typed interruption recovery event.
    """

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    kind: InterruptionRecoveryEventKind
    reason: InterruptionRecoveryReason
    latency_ms: float | None = None
    snapshot_id: str | None = None
    delta_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("event_id", "session_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class InterruptionRecoverySessionState(OrchestrationModel):
    """
    State for one interruption recovery session.
    """

    session_id: str = Field(default_factory=lambda: uuid4().hex)
    trace_id: str = Field(default_factory=lambda: uuid4().hex)
    turn_id: str
    status: InterruptionRecoveryStatus = InterruptionRecoveryStatus.CREATED
    started_at_ns: int | None = None
    interrupt_detected_at_ns: int | None = None
    tts_stopped_at_ns: int | None = None
    llm_cancelled_at_ns: int | None = None
    context_captured_at_ns: int | None = None
    context_ready_at_ns: int | None = None
    first_new_word_at_ns: int | None = None
    completed_at_ns: int | None = None
    cancelled_at_ns: int | None = None
    failed_at_ns: int | None = None
    snapshot_count: int = Field(default=0, ge=0)
    delta_count: int = Field(default=0, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("session_id", "trace_id", "turn_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    def tts_stop_latency_ms(self) -> float | None:
        if self.interrupt_detected_at_ns is None or self.tts_stopped_at_ns is None:
            return None

        return (self.tts_stopped_at_ns - self.interrupt_detected_at_ns) / 1_000_000.0

    def llm_cancel_latency_ms(self) -> float | None:
        if self.interrupt_detected_at_ns is None or self.llm_cancelled_at_ns is None:
            return None

        return (
            self.llm_cancelled_at_ns - self.interrupt_detected_at_ns
        ) / 1_000_000.0

    def context_capture_latency_ms(self) -> float | None:
        if self.interrupt_detected_at_ns is None or self.context_captured_at_ns is None:
            return None

        return (
            self.context_captured_at_ns - self.interrupt_detected_at_ns
        ) / 1_000_000.0

    def context_ready_latency_ms(self) -> float | None:
        if self.interrupt_detected_at_ns is None or self.context_ready_at_ns is None:
            return None

        return (
            self.context_ready_at_ns - self.interrupt_detected_at_ns
        ) / 1_000_000.0

    def first_new_word_latency_ms(self) -> float | None:
        if self.interrupt_detected_at_ns is None or self.first_new_word_at_ns is None:
            return None

        return (
            self.first_new_word_at_ns - self.interrupt_detected_at_ns
        ) / 1_000_000.0


class InterruptionRecoveryResult(OrchestrationModel):
    """
    Result from interruption recovery operation.
    """

    success: bool
    reason: InterruptionRecoveryReason
    session_id: str
    status: InterruptionRecoveryStatus
    snapshot: RecoveryContextSnapshot | None = None
    delta: InterruptionContextDelta | None = None
    reconstructed: ReconstructedInterruptionContext | None = None
    event: InterruptionRecoveryEvent | None = None
    state: InterruptionRecoverySessionState | None = None
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


class InterruptionRecoveryReport(OrchestrationModel):
    """
    Final interruption recovery report.
    """

    session_id: str
    trace_id: str
    turn_id: str
    status: InterruptionRecoveryStatus
    snapshot_count: int = Field(ge=0)
    delta_count: int = Field(ge=0)
    tts_stop_latency_ms: float | None = None
    llm_cancel_latency_ms: float | None = None
    context_capture_latency_ms: float | None = None
    context_ready_latency_ms: float | None = None
    first_new_word_latency_ms: float | None = None
    reconstructed_context: ReconstructedInterruptionContext | None = None
    events: tuple[InterruptionRecoveryEvent, ...]
    profiler_report: PipelineProfilerReport | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("session_id", "trace_id", "turn_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class InterruptionRecoveryRuntimeSnapshot(OrchestrationModel):
    """
    Runtime diagnostics for Phase 7 Step 12.
    """

    name: str
    session_count: int = Field(ge=0)
    tracking_count: int = Field(ge=0)
    ready_count: int = Field(ge=0)
    completed_count: int = Field(ge=0)
    cancelled_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    snapshot_count: int = Field(ge=0)
    report_count: int = Field(ge=0)
    last_reason: InterruptionRecoveryReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class InterruptionRecoveryRuntimeConfig:
    """
    Phase 7 Step 12 interruption recovery configuration.
    """

    name: str = "interruption_recovery_runtime"
    snapshot_interval_ms: float = 500.0
    tts_stop_budget_ms: float = 10.0
    llm_cancel_budget_ms: float = 30.0
    context_capture_budget_ms: float = 20.0
    context_ready_budget_ms: float = 80.0
    first_new_word_budget_ms: float = 300.0
    profile_recovery: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        for field_name, value in (
            ("snapshot_interval_ms", self.snapshot_interval_ms),
            ("tts_stop_budget_ms", self.tts_stop_budget_ms),
            ("llm_cancel_budget_ms", self.llm_cancel_budget_ms),
            ("context_capture_budget_ms", self.context_capture_budget_ms),
            ("context_ready_budget_ms", self.context_ready_budget_ms),
            ("first_new_word_budget_ms", self.first_new_word_budget_ms),
        ):
            if value <= 0:
                raise ValueError(f"{field_name} must be positive.")


class InterruptionRecoveryRuntime:
    """
    Phase 7 Step 12 Interruption Recovery Optimization.

    Responsibilities:
    - capture generation snapshots every 500ms
    - stop TTS playback quickly
    - cancel LLM stream quickly
    - capture interruption context quickly
    - reconstruct from nearest snapshot + delta
    - report first-new-word latency
    - avoid full context rebuild after interrupt

    Non-responsibilities:
    - no direct audio device control
    - no direct LLM execution
    - no real TTS synthesis
    - no tool/action execution
    """

    def __init__(
        self,
        *,
        config: InterruptionRecoveryRuntimeConfig | None = None,
        profiler: PipelineLatencyProfiler | None = None,
    ) -> None:
        self._config = config or InterruptionRecoveryRuntimeConfig()
        self._config.validate()

        self._profiler = profiler or PipelineLatencyProfiler(
            config=PipelineProfilerConfig(record_to_latency_runtime=True)
        )
        self._states: dict[str, InterruptionRecoverySessionState] = {}
        self._snapshots: dict[str, list[RecoveryContextSnapshot]] = {}
        self._deltas: dict[str, list[InterruptionContextDelta]] = {}
        self._reconstructed: dict[str, ReconstructedInterruptionContext] = {}
        self._events: dict[str, list[InterruptionRecoveryEvent]] = {}
        self._reports: list[InterruptionRecoveryReport] = []
        self._lock = RLock()
        self._last_reason: InterruptionRecoveryReason | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def create_session(
        self,
        *,
        turn_id: str,
        trace_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> InterruptionRecoverySessionState:
        state = InterruptionRecoverySessionState(
            trace_id=trace_id or uuid4().hex,
            turn_id=turn_id,
            metadata=metadata or {},
        )
        event = self._event(
            session_id=state.session_id,
            kind=InterruptionRecoveryEventKind.SESSION_CREATED,
            reason=InterruptionRecoveryReason.SESSION_CREATED,
        )

        with self._lock:
            self._states[state.session_id] = state
            self._snapshots[state.session_id] = []
            self._deltas[state.session_id] = []
            self._events[state.session_id] = [event]
            self._last_reason = InterruptionRecoveryReason.SESSION_CREATED

        self._profiler.start_trace(
            name="interruption_recovery",
            trace_id=state.trace_id,
        )

        return state

    def start_tracking(self, session_id: str) -> InterruptionRecoveryResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            if state.status != InterruptionRecoveryStatus.CREATED:
                return self._failure(
                    session_id=session_id,
                    reason=InterruptionRecoveryReason.SESSION_NOT_ACTIVE,
                    status=state.status,
                    message="recovery session cannot start from current state",
                    state=state,
                )

            started = state.model_copy(
                update={
                    "status": InterruptionRecoveryStatus.TRACKING,
                    "started_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = started
            event = self._event(
                session_id=session_id,
                kind=InterruptionRecoveryEventKind.SESSION_STARTED,
                reason=InterruptionRecoveryReason.SESSION_STARTED,
            )
            self._events[session_id].append(event)
            self._last_reason = InterruptionRecoveryReason.SESSION_STARTED

        return InterruptionRecoveryResult(
            success=True,
            reason=InterruptionRecoveryReason.SESSION_STARTED,
            session_id=session_id,
            status=InterruptionRecoveryStatus.TRACKING,
            event=event,
            state=started,
            message="interruption recovery tracking started",
        )

    def capture_generation_snapshot(
        self,
        *,
        session_id: str,
        context_snapshot: ContextSnapshot,
        assistant_partial_text: str,
        generation_sequence: int,
        force: bool = False,
    ) -> InterruptionRecoveryResult:
        state = self.state_for(session_id)

        if state is None:
            return self._missing_session(session_id)

        if state.status != InterruptionRecoveryStatus.TRACKING:
            return self._failure(
                session_id=session_id,
                reason=InterruptionRecoveryReason.SESSION_NOT_ACTIVE,
                status=state.status,
                message="session is not tracking generation",
                state=state,
            )

        if not force and not self._snapshot_due(session_id):
            latest = self._latest_snapshot(session_id)

            return InterruptionRecoveryResult(
                success=True,
                reason=InterruptionRecoveryReason.SNAPSHOT_CAPTURED,
                session_id=session_id,
                status=state.status,
                snapshot=latest,
                state=state,
                message="snapshot interval not reached; latest snapshot reused",
            )

        snapshot = RecoveryContextSnapshot(
            session_id=session_id,
            turn_id=state.turn_id,
            context_snapshot=context_snapshot,
            assistant_partial_text=assistant_partial_text,
            generation_sequence=generation_sequence,
        )

        with self._lock:
            self._snapshots[session_id].append(snapshot)
            current = self._states[session_id]
            updated = current.model_copy(
                update={"snapshot_count": len(self._snapshots[session_id])}
            )
            self._states[session_id] = updated
            event = self._event(
                session_id=session_id,
                kind=InterruptionRecoveryEventKind.SNAPSHOT_CAPTURED,
                reason=InterruptionRecoveryReason.SNAPSHOT_CAPTURED,
                snapshot_id=snapshot.recovery_snapshot_id,
            )
            self._events[session_id].append(event)
            self._last_reason = InterruptionRecoveryReason.SNAPSHOT_CAPTURED

        return InterruptionRecoveryResult(
            success=True,
            reason=InterruptionRecoveryReason.SNAPSHOT_CAPTURED,
            session_id=session_id,
            status=InterruptionRecoveryStatus.TRACKING,
            snapshot=snapshot,
            event=event,
            state=updated,
            message="generation recovery snapshot captured",
        )

    def detect_interrupt(
        self,
        *,
        session_id: str,
        partial_assistant_utterance: str,
        user_new_utterance: str,
    ) -> InterruptionRecoveryResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            if state.status != InterruptionRecoveryStatus.TRACKING:
                return self._failure(
                    session_id=session_id,
                    reason=InterruptionRecoveryReason.SESSION_NOT_ACTIVE,
                    status=state.status,
                    message="session is not tracking interruption",
                    state=state,
                )

            detected_at = time.perf_counter_ns()
            interrupted = state.model_copy(
                update={
                    "status": InterruptionRecoveryStatus.INTERRUPTED,
                    "interrupt_detected_at_ns": detected_at,
                }
            )
            self._states[session_id] = interrupted
            event = self._event(
                session_id=session_id,
                kind=InterruptionRecoveryEventKind.INTERRUPT_DETECTED,
                reason=InterruptionRecoveryReason.INTERRUPT_DETECTED,
                metadata={
                    "partial_assistant_utterance": partial_assistant_utterance,
                    "user_new_utterance": user_new_utterance,
                },
            )
            self._events[session_id].append(event)
            self._last_reason = InterruptionRecoveryReason.INTERRUPT_DETECTED

        return InterruptionRecoveryResult(
            success=True,
            reason=InterruptionRecoveryReason.INTERRUPT_DETECTED,
            session_id=session_id,
            status=InterruptionRecoveryStatus.INTERRUPTED,
            event=event,
            state=interrupted,
            message="interrupt detected",
        )

    def stop_tts_playback(self, session_id: str) -> InterruptionRecoveryResult:
        return self._mark_timed_step(
            session_id=session_id,
            status=InterruptionRecoveryStatus.INTERRUPTED,
            timestamp_field="tts_stopped_at_ns",
            event_kind=InterruptionRecoveryEventKind.TTS_STOPPED,
            within_reason=InterruptionRecoveryReason.TTS_STOPPED_WITHIN_BUDGET,
            over_reason=InterruptionRecoveryReason.TTS_STOPPED_OVER_BUDGET,
            budget_ms=self._config.tts_stop_budget_ms,
            message="TTS playback stopped",
        )

    def cancel_llm_stream(self, session_id: str) -> InterruptionRecoveryResult:
        return self._mark_timed_step(
            session_id=session_id,
            status=InterruptionRecoveryStatus.INTERRUPTED,
            timestamp_field="llm_cancelled_at_ns",
            event_kind=InterruptionRecoveryEventKind.LLM_CANCELLED,
            within_reason=InterruptionRecoveryReason.LLM_CANCELLED_WITHIN_BUDGET,
            over_reason=InterruptionRecoveryReason.LLM_CANCELLED_OVER_BUDGET,
            budget_ms=self._config.llm_cancel_budget_ms,
            message="LLM stream cancelled",
        )

    def capture_interruption_context(
        self,
        *,
        session_id: str,
        partial_assistant_utterance: str,
        user_new_utterance: str,
    ) -> InterruptionRecoveryResult:
        state = self.state_for(session_id)

        if state is None:
            return self._missing_session(session_id)

        if state.status != InterruptionRecoveryStatus.INTERRUPTED:
            return self._failure(
                session_id=session_id,
                reason=InterruptionRecoveryReason.SESSION_NOT_ACTIVE,
                status=state.status,
                message="session is not interrupted",
                state=state,
            )

        now_ns = time.perf_counter_ns()
        delta = InterruptionContextDelta(
            interrupted_at_text=f"User interrupted at: {partial_assistant_utterance}",
            user_new_utterance=f"User now says: {user_new_utterance}",
            partial_assistant_utterance=partial_assistant_utterance,
        )
        reason = self._budget_reason(
            latency_ms=self._latency_from_interrupt(state, now_ns),
            budget_ms=self._config.context_capture_budget_ms,
            within=InterruptionRecoveryReason.CONTEXT_CAPTURED_WITHIN_BUDGET,
            over=InterruptionRecoveryReason.CONTEXT_CAPTURED_OVER_BUDGET,
        )

        with self._lock:
            self._deltas[session_id].append(delta)
            current = self._states[session_id]
            updated = current.model_copy(
                update={
                    "status": InterruptionRecoveryStatus.RECONSTRUCTING,
                    "context_captured_at_ns": now_ns,
                    "delta_count": len(self._deltas[session_id]),
                }
            )
            self._states[session_id] = updated
            event = self._event(
                session_id=session_id,
                kind=InterruptionRecoveryEventKind.INTERRUPTION_CONTEXT_CAPTURED,
                reason=reason,
                delta_id=delta.delta_id,
                latency_ms=self._latency_from_interrupt(current, now_ns),
            )
            self._events[session_id].append(event)
            self._last_reason = reason

        return InterruptionRecoveryResult(
            success=True,
            reason=reason,
            session_id=session_id,
            status=InterruptionRecoveryStatus.RECONSTRUCTING,
            delta=delta,
            event=event,
            state=updated,
            message="interruption context captured",
        )

    def reconstruct_context(self, session_id: str) -> InterruptionRecoveryResult:
        state = self.state_for(session_id)

        if state is None:
            return self._missing_session(session_id)

        if state.status != InterruptionRecoveryStatus.RECONSTRUCTING:
            return self._failure(
                session_id=session_id,
                reason=InterruptionRecoveryReason.SESSION_NOT_ACTIVE,
                status=state.status,
                message="session is not reconstructing",
                state=state,
            )

        snapshot = self._latest_snapshot(session_id)

        if snapshot is None:
            return self._failure(
                session_id=session_id,
                reason=InterruptionRecoveryReason.NO_SNAPSHOT_AVAILABLE,
                status=state.status,
                message="no interruption recovery snapshot available",
                state=state,
            )

        delta = self._latest_delta(session_id)

        if delta is None:
            return self._failure(
                session_id=session_id,
                reason=InterruptionRecoveryReason.SESSION_NOT_ACTIVE,
                status=state.status,
                message="no interruption delta available",
                state=state,
            )

        restored_snapshot = self._apply_delta(snapshot=snapshot, delta=delta)
        ready_ns = time.perf_counter_ns()
        reconstructed = ReconstructedInterruptionContext(
            session_id=session_id,
            source_snapshot_id=snapshot.recovery_snapshot_id,
            delta_id=delta.delta_id,
            context_snapshot=restored_snapshot,
            ready_at_ns=ready_ns,
        )
        latency_ms = self._latency_from_interrupt(state, ready_ns)
        reason = self._budget_reason(
            latency_ms=latency_ms,
            budget_ms=self._config.context_ready_budget_ms,
            within=InterruptionRecoveryReason.CONTEXT_READY_WITHIN_BUDGET,
            over=InterruptionRecoveryReason.CONTEXT_READY_OVER_BUDGET,
        )

        with self._lock:
            self._reconstructed[session_id] = reconstructed
            current = self._states[session_id]
            updated = current.model_copy(
                update={
                    "status": InterruptionRecoveryStatus.READY,
                    "context_ready_at_ns": ready_ns,
                }
            )
            self._states[session_id] = updated
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=InterruptionRecoveryEventKind.SNAPSHOT_RESTORED,
                    reason=InterruptionRecoveryReason.SNAPSHOT_RESTORED,
                    snapshot_id=snapshot.recovery_snapshot_id,
                )
            )
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=InterruptionRecoveryEventKind.DELTA_APPLIED,
                    reason=InterruptionRecoveryReason.DELTA_APPLIED,
                    delta_id=delta.delta_id,
                )
            )
            event = self._event(
                session_id=session_id,
                kind=InterruptionRecoveryEventKind.CONTEXT_RECONSTRUCTED,
                reason=reason,
                snapshot_id=snapshot.recovery_snapshot_id,
                delta_id=delta.delta_id,
                latency_ms=latency_ms,
            )
            self._events[session_id].append(event)
            self._last_reason = reason

        return InterruptionRecoveryResult(
            success=True,
            reason=reason,
            session_id=session_id,
            status=InterruptionRecoveryStatus.READY,
            reconstructed=reconstructed,
            event=event,
            state=updated,
            message="interruption context reconstructed from snapshot plus delta",
        )

    def mark_first_new_word_ready(self, session_id: str) -> InterruptionRecoveryResult:
        state = self.state_for(session_id)

        if state is None:
            return self._missing_session(session_id)

        if state.status != InterruptionRecoveryStatus.READY:
            return self._failure(
                session_id=session_id,
                reason=InterruptionRecoveryReason.SESSION_NOT_ACTIVE,
                status=state.status,
                message="context is not ready for new response",
                state=state,
            )

        now_ns = time.perf_counter_ns()
        latency_ms = self._latency_from_interrupt(state, now_ns)
        reason = self._budget_reason(
            latency_ms=latency_ms,
            budget_ms=self._config.first_new_word_budget_ms,
            within=InterruptionRecoveryReason.FIRST_NEW_WORD_WITHIN_BUDGET,
            over=InterruptionRecoveryReason.FIRST_NEW_WORD_OVER_BUDGET,
        )

        with self._lock:
            current = self._states[session_id]
            updated = current.model_copy(update={"first_new_word_at_ns": now_ns})
            self._states[session_id] = updated
            event = self._event(
                session_id=session_id,
                kind=InterruptionRecoveryEventKind.FIRST_NEW_WORD_READY,
                reason=reason,
                latency_ms=latency_ms,
            )
            self._events[session_id].append(event)
            self._last_reason = reason

        return InterruptionRecoveryResult(
            success=True,
            reason=reason,
            session_id=session_id,
            status=InterruptionRecoveryStatus.READY,
            event=event,
            state=updated,
            message="first new word ready after interruption",
        )

    def complete_session(self, session_id: str) -> InterruptionRecoveryReport:
        state = self.state_for(session_id)

        if state is None:
            raise ValueError(f"interruption recovery session not found: {session_id}")

        if state.status != InterruptionRecoveryStatus.READY:
            raise ValueError("recovery session cannot complete from current state")

        with self._lock:
            current = self._states[session_id]
            completed = current.model_copy(
                update={
                    "status": InterruptionRecoveryStatus.COMPLETED,
                    "completed_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = completed
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=InterruptionRecoveryEventKind.SESSION_COMPLETED,
                    reason=InterruptionRecoveryReason.SESSION_COMPLETED,
                )
            )
            self._last_reason = InterruptionRecoveryReason.SESSION_COMPLETED

        self._record_recovery_span(completed)

        profiler_report = None

        if self._config.profile_recovery:
            profiler_report = self._profiler.complete_trace(completed.trace_id)

        report = InterruptionRecoveryReport(
            session_id=session_id,
            trace_id=completed.trace_id,
            turn_id=completed.turn_id,
            status=completed.status,
            snapshot_count=completed.snapshot_count,
            delta_count=completed.delta_count,
            tts_stop_latency_ms=completed.tts_stop_latency_ms(),
            llm_cancel_latency_ms=completed.llm_cancel_latency_ms(),
            context_capture_latency_ms=completed.context_capture_latency_ms(),
            context_ready_latency_ms=completed.context_ready_latency_ms(),
            first_new_word_latency_ms=completed.first_new_word_latency_ms(),
            reconstructed_context=self.reconstructed_for(session_id),
            events=self.events_for(session_id),
            profiler_report=profiler_report,
        )

        with self._lock:
            self._reports.append(report)

        return report

    def cancel_session(self, session_id: str) -> InterruptionRecoveryResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            cancelled = state.model_copy(
                update={
                    "status": InterruptionRecoveryStatus.CANCELLED,
                    "cancelled_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = cancelled
            event = self._event(
                session_id=session_id,
                kind=InterruptionRecoveryEventKind.SESSION_CANCELLED,
                reason=InterruptionRecoveryReason.SESSION_CANCELLED,
            )
            self._events[session_id].append(event)
            self._last_reason = InterruptionRecoveryReason.SESSION_CANCELLED

        return InterruptionRecoveryResult(
            success=True,
            reason=InterruptionRecoveryReason.SESSION_CANCELLED,
            session_id=session_id,
            status=InterruptionRecoveryStatus.CANCELLED,
            event=event,
            state=cancelled,
            message="interruption recovery session cancelled",
        )

    def fail_session(
        self,
        session_id: str,
        *,
        error: str,
    ) -> InterruptionRecoveryResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            failed = state.model_copy(
                update={
                    "status": InterruptionRecoveryStatus.FAILED,
                    "failed_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = failed
            event = self._event(
                session_id=session_id,
                kind=InterruptionRecoveryEventKind.SESSION_FAILED,
                reason=InterruptionRecoveryReason.SESSION_FAILED,
                metadata={"error": error},
            )
            self._events[session_id].append(event)
            self._last_reason = InterruptionRecoveryReason.SESSION_FAILED

        return InterruptionRecoveryResult(
            success=True,
            reason=InterruptionRecoveryReason.SESSION_FAILED,
            session_id=session_id,
            status=InterruptionRecoveryStatus.FAILED,
            event=event,
            state=failed,
            message="interruption recovery session failed",
        )

    def state_for(
        self,
        session_id: str,
    ) -> InterruptionRecoverySessionState | None:
        with self._lock:
            return self._states.get(session_id)

    def snapshots_for(self, session_id: str) -> tuple[RecoveryContextSnapshot, ...]:
        with self._lock:
            return tuple(self._snapshots.get(session_id, ()))

    def deltas_for(self, session_id: str) -> tuple[InterruptionContextDelta, ...]:
        with self._lock:
            return tuple(self._deltas.get(session_id, ()))

    def reconstructed_for(
        self,
        session_id: str,
    ) -> ReconstructedInterruptionContext | None:
        with self._lock:
            return self._reconstructed.get(session_id)

    def events_for(self, session_id: str) -> tuple[InterruptionRecoveryEvent, ...]:
        with self._lock:
            return tuple(self._events.get(session_id, ()))

    def reports(self) -> tuple[InterruptionRecoveryReport, ...]:
        with self._lock:
            return tuple(self._reports)

    def latest_report(self) -> InterruptionRecoveryReport | None:
        with self._lock:
            if not self._reports:
                return None

            return self._reports[-1]

    def snapshot(self) -> InterruptionRecoveryRuntimeSnapshot:
        with self._lock:
            states = tuple(self._states.values())

            return InterruptionRecoveryRuntimeSnapshot(
                name=self.name,
                session_count=len(states),
                tracking_count=sum(
                    1
                    for state in states
                    if state.status == InterruptionRecoveryStatus.TRACKING
                ),
                ready_count=sum(
                    1
                    for state in states
                    if state.status == InterruptionRecoveryStatus.READY
                ),
                completed_count=sum(
                    1
                    for state in states
                    if state.status == InterruptionRecoveryStatus.COMPLETED
                ),
                cancelled_count=sum(
                    1
                    for state in states
                    if state.status == InterruptionRecoveryStatus.CANCELLED
                ),
                failed_count=sum(
                    1
                    for state in states
                    if state.status == InterruptionRecoveryStatus.FAILED
                ),
                snapshot_count=sum(
                    len(items) for items in self._snapshots.values()
                ),
                report_count=len(self._reports),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        with self._lock:
            self._states.clear()
            self._snapshots.clear()
            self._deltas.clear()
            self._reconstructed.clear()
            self._events.clear()
            self._reports.clear()
            self._last_reason = InterruptionRecoveryReason.RUNTIME_RESET

    def _mark_timed_step(
        self,
        *,
        session_id: str,
        status: InterruptionRecoveryStatus,
        timestamp_field: str,
        event_kind: InterruptionRecoveryEventKind,
        within_reason: InterruptionRecoveryReason,
        over_reason: InterruptionRecoveryReason,
        budget_ms: float,
        message: str,
    ) -> InterruptionRecoveryResult:
        state = self.state_for(session_id)

        if state is None:
            return self._missing_session(session_id)

        if state.status != status:
            return self._failure(
                session_id=session_id,
                reason=InterruptionRecoveryReason.SESSION_NOT_ACTIVE,
                status=state.status,
                message="recovery session is not in expected state",
                state=state,
            )

        now_ns = time.perf_counter_ns()
        latency_ms = self._latency_from_interrupt(state, now_ns)
        reason = self._budget_reason(
            latency_ms=latency_ms,
            budget_ms=budget_ms,
            within=within_reason,
            over=over_reason,
        )

        with self._lock:
            current = self._states[session_id]
            updated = current.model_copy(update={timestamp_field: now_ns})
            self._states[session_id] = updated
            event = self._event(
                session_id=session_id,
                kind=event_kind,
                reason=reason,
                latency_ms=latency_ms,
            )
            self._events[session_id].append(event)
            self._last_reason = reason

        return InterruptionRecoveryResult(
            success=True,
            reason=reason,
            session_id=session_id,
            status=updated.status,
            event=event,
            state=updated,
            message=message,
        )

    def _snapshot_due(self, session_id: str) -> bool:
        latest = self._latest_snapshot(session_id)

        if latest is None:
            return True

        elapsed_ms = (
            time.perf_counter_ns() - latest.captured_at_ns
        ) / 1_000_000.0

        return elapsed_ms >= self._config.snapshot_interval_ms

    def _latest_snapshot(self, session_id: str) -> RecoveryContextSnapshot | None:
        with self._lock:
            snapshots = self._snapshots.get(session_id, ())

            if not snapshots:
                return None

            return snapshots[-1]

    def _latest_delta(self, session_id: str) -> InterruptionContextDelta | None:
        with self._lock:
            deltas = self._deltas.get(session_id, ())

            if not deltas:
                return None

            return deltas[-1]

    @staticmethod
    def _apply_delta(
        *,
        snapshot: RecoveryContextSnapshot,
        delta: InterruptionContextDelta,
    ) -> ContextSnapshot:
        delta_fragments = (
            ContextFragment(
                kind=ContextFragmentKind.RECENT_TURN,
                text=delta.interrupted_at_text,
                priority=98,
                token_estimate=max(1, len(delta.interrupted_at_text.split())),
                source_id=delta.delta_id,
            ),
            ContextFragment(
                kind=ContextFragmentKind.RECENT_TURN,
                text=delta.user_new_utterance,
                priority=100,
                token_estimate=max(1, len(delta.user_new_utterance.split())),
                source_id=delta.delta_id,
            ),
        )
        fragments = (*snapshot.context_snapshot.fragments, *delta_fragments)
        token_estimate = sum(fragment.token_estimate for fragment in fragments)

        return snapshot.context_snapshot.model_copy(
            update={
                "fragments": fragments,
                "token_estimate": token_estimate,
                "metadata": {
                    **snapshot.context_snapshot.metadata,
                    "interruption_delta_id": delta.delta_id,
                    "source_recovery_snapshot_id": snapshot.recovery_snapshot_id,
                },
            }
        )

    @staticmethod
    def _budget_reason(
        *,
        latency_ms: float | None,
        budget_ms: float,
        within: InterruptionRecoveryReason,
        over: InterruptionRecoveryReason,
    ) -> InterruptionRecoveryReason:
        if latency_ms is None:
            return over

        return within if latency_ms <= budget_ms else over

    @staticmethod
    def _latency_from_interrupt(
        state: InterruptionRecoverySessionState,
        now_ns: int,
    ) -> float | None:
        if state.interrupt_detected_at_ns is None:
            return None

        return (now_ns - state.interrupt_detected_at_ns) / 1_000_000.0

    def _record_recovery_span(
        self,
        state: InterruptionRecoverySessionState,
    ) -> None:
        if state.interrupt_detected_at_ns is None:
            return

        end_ns = state.first_new_word_at_ns or state.completed_at_ns

        if end_ns is None:
            return

        self._profiler.record_stage(
            trace_id=state.trace_id,
            stage=PipelineStage.INTERRUPT_RECOVERY,
            start_ns=state.interrupt_detected_at_ns,
            end_ns=end_ns,
            metadata={
                "session_id": state.session_id,
                "turn_id": state.turn_id,
            },
        )

    @staticmethod
    def _event(
        *,
        session_id: str,
        kind: InterruptionRecoveryEventKind,
        reason: InterruptionRecoveryReason,
        latency_ms: float | None = None,
        snapshot_id: str | None = None,
        delta_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> InterruptionRecoveryEvent:
        return InterruptionRecoveryEvent(
            session_id=session_id,
            kind=kind,
            reason=reason,
            latency_ms=latency_ms,
            snapshot_id=snapshot_id,
            delta_id=delta_id,
            metadata=metadata or {},
        )

    @staticmethod
    def _missing_session(session_id: str) -> InterruptionRecoveryResult:
        return InterruptionRecoveryResult(
            success=False,
            reason=InterruptionRecoveryReason.SESSION_NOT_FOUND,
            session_id=session_id,
            status=InterruptionRecoveryStatus.FAILED,
            message="interruption recovery session not found",
        )

    @staticmethod
    def _failure(
        *,
        session_id: str,
        reason: InterruptionRecoveryReason,
        status: InterruptionRecoveryStatus,
        message: str,
        state: InterruptionRecoverySessionState | None = None,
    ) -> InterruptionRecoveryResult:
        return InterruptionRecoveryResult(
            success=False,
            reason=reason,
            session_id=session_id,
            status=status,
            state=state,
            message=message,
        )