from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.latency.profiler import (
    PipelineLatencyProfiler,
    PipelineProfilerConfig,
    PipelineProfilerReport,
    PipelineStage,
)
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class ActionFeedbackType(StrEnum):
    """
    User-facing feedback type.

    These map to natural conversation/TTS updates, not raw system logs.
    """

    STARTED = "started"
    PROGRESS = "progress"
    MILESTONE = "milestone"
    COMPLETED = "completed"
    ERROR = "error"


class ActionFeedbackStatus(StrEnum):
    """
    Feedback stream lifecycle.
    """

    CREATED = "created"
    STREAMING = "streaming"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ActionFeedbackEventKind(StrEnum):
    """
    Internal event kinds for streaming action feedback.
    """

    SESSION_CREATED = "session_created"
    STREAM_STARTED = "stream_started"
    FEEDBACK_EMITTED = "feedback_emitted"
    PROGRESS_TICK = "progress_tick"
    MILESTONE_REACHED = "milestone_reached"
    COMPLETION_SUMMARY = "completion_summary"
    ERROR_SUMMARY = "error_summary"
    SESSION_COMPLETED = "session_completed"
    SESSION_CANCELLED = "session_cancelled"
    SESSION_FAILED = "session_failed"


class ActionFeedbackReason(StrEnum):
    """
    Machine-readable feedback reasons.
    """

    SESSION_CREATED = "session_created"
    STREAM_STARTED = "stream_started"
    START_ACK_EMITTED = "start_ack_emitted"
    START_ACK_WITHIN_BUDGET = "start_ack_within_budget"
    START_ACK_OVER_BUDGET = "start_ack_over_budget"
    PROGRESS_EMITTED = "progress_emitted"
    PROGRESS_NOT_DUE = "progress_not_due"
    MILESTONE_EMITTED = "milestone_emitted"
    COMPLETION_EMITTED = "completion_emitted"
    ERROR_EMITTED = "error_emitted"
    SESSION_COMPLETED = "session_completed"
    SESSION_CANCELLED = "session_cancelled"
    SESSION_FAILED = "session_failed"
    SESSION_NOT_FOUND = "session_not_found"
    SESSION_NOT_STREAMING = "session_not_streaming"
    INVALID_FEEDBACK_MESSAGE = "invalid_feedback_message"
    RUNTIME_RESET = "runtime_reset"


class FeedbackDeliveryTarget(StrEnum):
    """
    Target for feedback delivery.
    """

    CONVERSATION = "conversation"
    TTS = "tts"
    DEBUG = "debug"


class ActionFeedbackChunk(OrchestrationModel):
    """
    One user-facing feedback chunk.

    The `spoken_text` field is deliberately natural. It should sound like
    JARVIS working alongside the user, not like a system log.
    """

    chunk_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    action_id: str
    feedback_type: ActionFeedbackType
    spoken_text: str
    progress_current: int | None = Field(default=None, ge=0)
    progress_total: int | None = Field(default=None, ge=0)
    target: FeedbackDeliveryTarget = FeedbackDeliveryTarget.CONVERSATION
    tts_ready: bool = True
    emitted_at_ns: int = Field(default_factory=time.perf_counter_ns, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("chunk_id", "session_id", "action_id", "spoken_text")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @model_validator(mode="after")
    def _validate_progress(self) -> ActionFeedbackChunk:
        if self.progress_current is None and self.progress_total is None:
            return self

        if self.progress_current is None or self.progress_total is None:
            raise ValueError("progress_current and progress_total must pair.")

        if self.progress_current > self.progress_total:
            raise ValueError("progress_current cannot exceed progress_total.")

        return self


class ActionFeedbackEvent(OrchestrationModel):
    """
    Typed event for feedback observability.
    """

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    action_id: str
    kind: ActionFeedbackEventKind
    reason: ActionFeedbackReason
    feedback_type: ActionFeedbackType | None = None
    chunk_id: str | None = None
    latency_ms: float | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("event_id", "session_id", "action_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class ActionFeedbackSessionState(OrchestrationModel):
    """
    Runtime state for one action feedback stream.
    """

    session_id: str = Field(default_factory=lambda: uuid4().hex)
    trace_id: str = Field(default_factory=lambda: uuid4().hex)
    action_id: str
    action_name: str
    status: ActionFeedbackStatus = ActionFeedbackStatus.CREATED
    started_at_ns: int | None = None
    first_feedback_at_ns: int | None = None
    last_feedback_at_ns: int | None = None
    completed_at_ns: int | None = None
    cancelled_at_ns: int | None = None
    failed_at_ns: int | None = None
    chunk_count: int = Field(default=0, ge=0)
    progress_count: int = Field(default=0, ge=0)
    milestone_count: int = Field(default=0, ge=0)
    error_count: int = Field(default=0, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("session_id", "trace_id", "action_id", "action_name")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    def first_feedback_latency_ms(self) -> float | None:
        if self.started_at_ns is None or self.first_feedback_at_ns is None:
            return None

        return (self.first_feedback_at_ns - self.started_at_ns) / 1_000_000.0

    def total_latency_ms(self) -> float | None:
        if self.started_at_ns is None or self.completed_at_ns is None:
            return None

        return (self.completed_at_ns - self.started_at_ns) / 1_000_000.0


class ActionFeedbackResult(OrchestrationModel):
    """
    Result from action feedback runtime operation.
    """

    success: bool
    reason: ActionFeedbackReason
    session_id: str
    action_id: str
    status: ActionFeedbackStatus
    chunk: ActionFeedbackChunk | None = None
    event: ActionFeedbackEvent | None = None
    state: ActionFeedbackSessionState | None = None
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("session_id", "action_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class ActionFeedbackReport(OrchestrationModel):
    """
    Final report for one action feedback stream.
    """

    session_id: str
    trace_id: str
    action_id: str
    action_name: str
    status: ActionFeedbackStatus
    chunk_count: int = Field(ge=0)
    progress_count: int = Field(ge=0)
    milestone_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    first_feedback_latency_ms: float | None = None
    total_latency_ms: float | None = None
    chunks: tuple[ActionFeedbackChunk, ...]
    events: tuple[ActionFeedbackEvent, ...]
    profiler_report: PipelineProfilerReport | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("session_id", "trace_id", "action_id", "action_name")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class ActionFeedbackRuntimeSnapshot(OrchestrationModel):
    """
    Runtime diagnostics for Phase 7 Step 14.
    """

    name: str
    session_count: int = Field(ge=0)
    streaming_count: int = Field(ge=0)
    completed_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    cancelled_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    report_count: int = Field(ge=0)
    last_reason: ActionFeedbackReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class StreamingActionFeedbackConfig:
    """
    Streaming action feedback configuration.
    """

    name: str = "streaming_action_feedback_runtime"
    start_ack_budget_ms: float = 200.0
    progress_interval_ms: float = 2_000.0
    profile_feedback: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.start_ack_budget_ms <= 0:
            raise ValueError("start_ack_budget_ms must be positive.")

        if self.progress_interval_ms <= 0:
            raise ValueError("progress_interval_ms must be positive.")


class StreamingActionFeedbackRuntime:
    """
    Phase 7 Step 14 Streaming Action Feedback.

    Responsibilities:
    - emit immediate action acknowledgement
    - emit periodic progress updates
    - emit meaningful milestone updates immediately
    - emit completion/error summaries
    - produce natural TTS-ready conversation chunks
    - prevent silent waiting during long operations

    Non-responsibilities:
    - no direct tool execution
    - no shell execution
    - no TTS synthesis
    - no action approval decisions
    """

    def __init__(
        self,
        *,
        config: StreamingActionFeedbackConfig | None = None,
        profiler: PipelineLatencyProfiler | None = None,
    ) -> None:
        self._config = config or StreamingActionFeedbackConfig()
        self._config.validate()

        self._profiler = profiler or PipelineLatencyProfiler(
            config=PipelineProfilerConfig(record_to_latency_runtime=True)
        )
        self._states: dict[str, ActionFeedbackSessionState] = {}
        self._chunks: dict[str, list[ActionFeedbackChunk]] = {}
        self._events: dict[str, list[ActionFeedbackEvent]] = {}
        self._reports: list[ActionFeedbackReport] = []
        self._lock = RLock()
        self._last_reason: ActionFeedbackReason | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def create_session(
        self,
        *,
        action_id: str,
        action_name: str,
        trace_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> ActionFeedbackSessionState:
        state = ActionFeedbackSessionState(
            trace_id=trace_id or uuid4().hex,
            action_id=action_id,
            action_name=action_name,
            metadata=metadata or {},
        )
        event = self._event(
            session_id=state.session_id,
            action_id=action_id,
            kind=ActionFeedbackEventKind.SESSION_CREATED,
            reason=ActionFeedbackReason.SESSION_CREATED,
        )

        with self._lock:
            self._states[state.session_id] = state
            self._chunks[state.session_id] = []
            self._events[state.session_id] = [event]
            self._last_reason = ActionFeedbackReason.SESSION_CREATED

        self._profiler.start_trace(
            name="streaming_action_feedback",
            trace_id=state.trace_id,
        )

        return state

    def start_stream(
        self,
        session_id: str,
        *,
        message: str | None = None,
    ) -> ActionFeedbackResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            if state.status != ActionFeedbackStatus.CREATED:
                return self._failure(
                    session_id=session_id,
                    action_id=state.action_id,
                    reason=ActionFeedbackReason.SESSION_NOT_STREAMING,
                    status=state.status,
                    message="feedback stream cannot start from current state",
                    state=state,
                )

            started = state.model_copy(
                update={
                    "status": ActionFeedbackStatus.STREAMING,
                    "started_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = started
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    action_id=started.action_id,
                    kind=ActionFeedbackEventKind.STREAM_STARTED,
                    reason=ActionFeedbackReason.STREAM_STARTED,
                )
            )
            self._last_reason = ActionFeedbackReason.STREAM_STARTED

        return self.emit_started(session_id=session_id, message=message)

    def emit_started(
        self,
        *,
        session_id: str,
        message: str | None = None,
    ) -> ActionFeedbackResult:
        state = self.state_for(session_id)

        if state is None:
            return self._missing_session(session_id)

        if state.status != ActionFeedbackStatus.STREAMING:
            return self._failure(
                session_id=session_id,
                action_id=state.action_id,
                reason=ActionFeedbackReason.SESSION_NOT_STREAMING,
                status=state.status,
                message="feedback stream is not active",
                state=state,
            )

        spoken = message or f"{state.action_name} now..."
        result = self._emit_chunk(
            session_id=session_id,
            feedback_type=ActionFeedbackType.STARTED,
            spoken_text=spoken,
            reason=ActionFeedbackReason.START_ACK_EMITTED,
            event_kind=ActionFeedbackEventKind.FEEDBACK_EMITTED,
        )

        latest = self.state_for(session_id)

        if latest is not None:
            latency = latest.first_feedback_latency_ms()
            budget_reason = (
                ActionFeedbackReason.START_ACK_WITHIN_BUDGET
                if latency is not None and latency <= self._config.start_ack_budget_ms
                else ActionFeedbackReason.START_ACK_OVER_BUDGET
            )
            self._append_event(
                session_id=session_id,
                action_id=latest.action_id,
                kind=ActionFeedbackEventKind.FEEDBACK_EMITTED,
                reason=budget_reason,
                feedback_type=ActionFeedbackType.STARTED,
                chunk_id=result.chunk.chunk_id if result.chunk else None,
                latency_ms=latency,
            )

        return result

    def emit_progress(
        self,
        *,
        session_id: str,
        message: str,
        progress_current: int | None = None,
        progress_total: int | None = None,
        force: bool = False,
    ) -> ActionFeedbackResult:
        state = self.state_for(session_id)

        if state is None:
            return self._missing_session(session_id)

        if state.status != ActionFeedbackStatus.STREAMING:
            return self._failure(
                session_id=session_id,
                action_id=state.action_id,
                reason=ActionFeedbackReason.SESSION_NOT_STREAMING,
                status=state.status,
                message="feedback stream is not active",
                state=state,
            )

        if not force and not self._progress_due(state):
            return ActionFeedbackResult(
                success=False,
                reason=ActionFeedbackReason.PROGRESS_NOT_DUE,
                session_id=session_id,
                action_id=state.action_id,
                status=state.status,
                state=state,
                message="progress interval not reached",
            )

        return self._emit_chunk(
            session_id=session_id,
            feedback_type=ActionFeedbackType.PROGRESS,
            spoken_text=message,
            reason=ActionFeedbackReason.PROGRESS_EMITTED,
            event_kind=ActionFeedbackEventKind.PROGRESS_TICK,
            progress_current=progress_current,
            progress_total=progress_total,
        )

    def emit_milestone(
        self,
        *,
        session_id: str,
        message: str,
    ) -> ActionFeedbackResult:
        return self._emit_chunk(
            session_id=session_id,
            feedback_type=ActionFeedbackType.MILESTONE,
            spoken_text=message,
            reason=ActionFeedbackReason.MILESTONE_EMITTED,
            event_kind=ActionFeedbackEventKind.MILESTONE_REACHED,
        )

    def emit_completed(
        self,
        *,
        session_id: str,
        message: str,
    ) -> ActionFeedbackResult:
        result = self._emit_chunk(
            session_id=session_id,
            feedback_type=ActionFeedbackType.COMPLETED,
            spoken_text=message,
            reason=ActionFeedbackReason.COMPLETION_EMITTED,
            event_kind=ActionFeedbackEventKind.COMPLETION_SUMMARY,
        )

        if not result.success:
            return result

        with self._lock:
            state = self._states[session_id]
            self._states[session_id] = state.model_copy(
                update={
                    "status": ActionFeedbackStatus.COMPLETED,
                    "completed_at_ns": time.perf_counter_ns(),
                }
            )

        return result

    def emit_error(
        self,
        *,
        session_id: str,
        message: str,
    ) -> ActionFeedbackResult:
        result = self._emit_chunk(
            session_id=session_id,
            feedback_type=ActionFeedbackType.ERROR,
            spoken_text=message,
            reason=ActionFeedbackReason.ERROR_EMITTED,
            event_kind=ActionFeedbackEventKind.ERROR_SUMMARY,
        )

        if not result.success:
            return result

        with self._lock:
            state = self._states[session_id]
            self._states[session_id] = state.model_copy(
                update={
                    "status": ActionFeedbackStatus.ERROR,
                    "completed_at_ns": time.perf_counter_ns(),
                    "error_count": state.error_count + 1,
                }
            )

        return result

    def complete_session(self, session_id: str) -> ActionFeedbackReport:
        state = self.state_for(session_id)

        if state is None:
            raise ValueError(f"action feedback session not found: {session_id}")

        if state.status not in {
            ActionFeedbackStatus.COMPLETED,
            ActionFeedbackStatus.ERROR,
        }:
            raise ValueError("feedback session cannot complete from current state")

        with self._lock:
            current = self._states[session_id]
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    action_id=current.action_id,
                    kind=ActionFeedbackEventKind.SESSION_COMPLETED,
                    reason=ActionFeedbackReason.SESSION_COMPLETED,
                )
            )
            self._last_reason = ActionFeedbackReason.SESSION_COMPLETED

        self._record_feedback_span(current)

        profiler_report = None

        if self._config.profile_feedback:
            profiler_report = self._profiler.complete_trace(current.trace_id)

        report = ActionFeedbackReport(
            session_id=session_id,
            trace_id=current.trace_id,
            action_id=current.action_id,
            action_name=current.action_name,
            status=current.status,
            chunk_count=current.chunk_count,
            progress_count=current.progress_count,
            milestone_count=current.milestone_count,
            error_count=current.error_count,
            first_feedback_latency_ms=current.first_feedback_latency_ms(),
            total_latency_ms=current.total_latency_ms(),
            chunks=self.chunks_for(session_id),
            events=self.events_for(session_id),
            profiler_report=profiler_report,
        )

        with self._lock:
            self._reports.append(report)

        return report

    def cancel_session(self, session_id: str) -> ActionFeedbackResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            cancelled = state.model_copy(
                update={
                    "status": ActionFeedbackStatus.CANCELLED,
                    "cancelled_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = cancelled
            event = self._event(
                session_id=session_id,
                action_id=cancelled.action_id,
                kind=ActionFeedbackEventKind.SESSION_CANCELLED,
                reason=ActionFeedbackReason.SESSION_CANCELLED,
            )
            self._events[session_id].append(event)
            self._last_reason = ActionFeedbackReason.SESSION_CANCELLED

        return ActionFeedbackResult(
            success=True,
            reason=ActionFeedbackReason.SESSION_CANCELLED,
            session_id=session_id,
            action_id=cancelled.action_id,
            status=ActionFeedbackStatus.CANCELLED,
            event=event,
            state=cancelled,
            message="action feedback stream cancelled",
        )

    def fail_session(
        self,
        session_id: str,
        *,
        error: str,
    ) -> ActionFeedbackResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            failed = state.model_copy(
                update={
                    "status": ActionFeedbackStatus.FAILED,
                    "failed_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = failed
            event = self._event(
                session_id=session_id,
                action_id=failed.action_id,
                kind=ActionFeedbackEventKind.SESSION_FAILED,
                reason=ActionFeedbackReason.SESSION_FAILED,
                metadata={"error": error},
            )
            self._events[session_id].append(event)
            self._last_reason = ActionFeedbackReason.SESSION_FAILED

        return ActionFeedbackResult(
            success=True,
            reason=ActionFeedbackReason.SESSION_FAILED,
            session_id=session_id,
            action_id=failed.action_id,
            status=ActionFeedbackStatus.FAILED,
            event=event,
            state=failed,
            message="action feedback stream failed",
        )

    def state_for(self, session_id: str) -> ActionFeedbackSessionState | None:
        with self._lock:
            return self._states.get(session_id)

    def chunks_for(self, session_id: str) -> tuple[ActionFeedbackChunk, ...]:
        with self._lock:
            return tuple(self._chunks.get(session_id, ()))

    def events_for(self, session_id: str) -> tuple[ActionFeedbackEvent, ...]:
        with self._lock:
            return tuple(self._events.get(session_id, ()))

    def reports(self) -> tuple[ActionFeedbackReport, ...]:
        with self._lock:
            return tuple(self._reports)

    def latest_report(self) -> ActionFeedbackReport | None:
        with self._lock:
            if not self._reports:
                return None

            return self._reports[-1]

    def snapshot(self) -> ActionFeedbackRuntimeSnapshot:
        with self._lock:
            states = tuple(self._states.values())

            return ActionFeedbackRuntimeSnapshot(
                name=self.name,
                session_count=len(states),
                streaming_count=sum(
                    1
                    for state in states
                    if state.status == ActionFeedbackStatus.STREAMING
                ),
                completed_count=sum(
                    1
                    for state in states
                    if state.status == ActionFeedbackStatus.COMPLETED
                ),
                error_count=sum(
                    1 
                    for state in states 
                    if state.status == ActionFeedbackStatus.ERROR
                ),
                cancelled_count=sum(
                    1
                    for state in states
                    if state.status == ActionFeedbackStatus.CANCELLED
                ),
                failed_count=sum(
                    1 
                    for state in states 
                    if state.status == ActionFeedbackStatus.FAILED
                ),
                chunk_count=sum(len(items) for items in self._chunks.values()),
                report_count=len(self._reports),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        with self._lock:
            self._states.clear()
            self._chunks.clear()
            self._events.clear()
            self._reports.clear()
            self._last_reason = ActionFeedbackReason.RUNTIME_RESET

    def _emit_chunk(
        self,
        *,
        session_id: str,
        feedback_type: ActionFeedbackType,
        spoken_text: str,
        reason: ActionFeedbackReason,
        event_kind: ActionFeedbackEventKind,
        progress_current: int | None = None,
        progress_total: int | None = None,
    ) -> ActionFeedbackResult:
        state = self.state_for(session_id)

        if state is None:
            return self._missing_session(session_id)

        if state.status != ActionFeedbackStatus.STREAMING:
            return self._failure(
                session_id=session_id,
                action_id=state.action_id,
                reason=ActionFeedbackReason.SESSION_NOT_STREAMING,
                status=state.status,
                message="feedback stream is not active",
                state=state,
            )

        if not spoken_text.strip():
            return self._failure(
                session_id=session_id,
                action_id=state.action_id,
                reason=ActionFeedbackReason.INVALID_FEEDBACK_MESSAGE,
                status=state.status,
                message="feedback message cannot be empty",
                state=state,
            )

        chunk = ActionFeedbackChunk(
            session_id=session_id,
            action_id=state.action_id,
            feedback_type=feedback_type,
            spoken_text=spoken_text,
            progress_current=progress_current,
            progress_total=progress_total,
        )
        emitted_at = chunk.emitted_at_ns

        with self._lock:
            current = self._states[session_id]
            first_feedback_at = current.first_feedback_at_ns or emitted_at
            update: dict[str, object] = {
                "first_feedback_at_ns": first_feedback_at,
                "last_feedback_at_ns": emitted_at,
                "chunk_count": current.chunk_count + 1,
            }

            if feedback_type == ActionFeedbackType.PROGRESS:
                update["progress_count"] = current.progress_count + 1
            elif feedback_type == ActionFeedbackType.MILESTONE:
                update["milestone_count"] = current.milestone_count + 1

            updated = current.model_copy(update=update)
            self._states[session_id] = updated
            self._chunks[session_id].append(chunk)

            latency = updated.first_feedback_latency_ms()
            event = self._event(
                session_id=session_id,
                action_id=state.action_id,
                kind=event_kind,
                reason=reason,
                feedback_type=feedback_type,
                chunk_id=chunk.chunk_id,
                latency_ms=latency,
            )
            self._events[session_id].append(event)
            self._last_reason = reason

        return ActionFeedbackResult(
            success=True,
            reason=reason,
            session_id=session_id,
            action_id=state.action_id,
            status=updated.status,
            chunk=chunk,
            event=event,
            state=updated,
            message="action feedback emitted",
        )

    def _append_event(
        self,
        *,
        session_id: str,
        action_id: str,
        kind: ActionFeedbackEventKind,
        reason: ActionFeedbackReason,
        feedback_type: ActionFeedbackType | None = None,
        chunk_id: str | None = None,
        latency_ms: float | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        with self._lock:
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    action_id=action_id,
                    kind=kind,
                    reason=reason,
                    feedback_type=feedback_type,
                    chunk_id=chunk_id,
                    latency_ms=latency_ms,
                    metadata=metadata,
                )
            )
            self._last_reason = reason

    def _progress_due(self, state: ActionFeedbackSessionState) -> bool:
        if state.last_feedback_at_ns is None:
            return True

        elapsed_ms = (
            time.perf_counter_ns() - state.last_feedback_at_ns
        ) / 1_000_000.0

        return elapsed_ms >= self._config.progress_interval_ms

    def _record_feedback_span(self, state: ActionFeedbackSessionState) -> None:
        if state.started_at_ns is None or state.completed_at_ns is None:
            return

        self._profiler.record_stage(
            trace_id=state.trace_id,
            stage=PipelineStage.TOOL_FIRST_FEEDBACK,
            start_ns=state.started_at_ns,
            end_ns=state.completed_at_ns,
            metadata={
                "session_id": state.session_id,
                "action_id": state.action_id,
                "chunk_count": state.chunk_count,
            },
        )

    @staticmethod
    def _event(
        *,
        session_id: str,
        action_id: str,
        kind: ActionFeedbackEventKind,
        reason: ActionFeedbackReason,
        feedback_type: ActionFeedbackType | None = None,
        chunk_id: str | None = None,
        latency_ms: float | None = None,
        metadata: dict[str, object] | None = None,
    ) -> ActionFeedbackEvent:
        return ActionFeedbackEvent(
            session_id=session_id,
            action_id=action_id,
            kind=kind,
            reason=reason,
            feedback_type=feedback_type,
            chunk_id=chunk_id,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )

    @staticmethod
    def _missing_session(session_id: str) -> ActionFeedbackResult:
        return ActionFeedbackResult(
            success=False,
            reason=ActionFeedbackReason.SESSION_NOT_FOUND,
            session_id=session_id,
            action_id="unknown",
            status=ActionFeedbackStatus.FAILED,
            message="action feedback session not found",
        )

    @staticmethod
    def _failure(
        *,
        session_id: str,
        action_id: str,
        reason: ActionFeedbackReason,
        status: ActionFeedbackStatus,
        message: str,
        state: ActionFeedbackSessionState | None = None,
    ) -> ActionFeedbackResult:
        return ActionFeedbackResult(
            success=False,
            reason=reason,
            session_id=session_id,
            action_id=action_id,
            status=status,
            state=state,
            message=message,
        )