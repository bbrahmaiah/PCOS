from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from uuid import uuid4

from pydantic import Field, field_validator

from jarvis.latency.models import (
    LatencyOperation,
    LatencySubsystem,
)
from jarvis.latency.profiler import (
    PipelineLatencyProfiler,
    PipelineProfilerConfig,
    PipelineProfilerReport,
    PipelineStage,
)
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class TokenStreamStatus(StrEnum):
    """
    Token stream lifecycle status.
    """

    CREATED = "created"
    ACTIVE = "active"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


class TokenStreamChunkKind(StrEnum):
    """
    Token stream chunk kind.
    """

    TOKEN = "token"
    WORD = "word"
    SENTENCE = "sentence"
    CONTROL = "control"
    FINAL = "final"


class TokenStreamEventKind(StrEnum):
    """
    Token streaming event kind.
    """

    STREAM_CREATED = "stream_created"
    STREAM_STARTED = "stream_started"
    FIRST_TOKEN_EMITTED = "first_token_emitted"
    CHUNK_EMITTED = "chunk_emitted"
    SENTENCE_STABILIZED = "sentence_stabilized"
    STREAM_CANCELLED = "stream_cancelled"
    STREAM_COMPLETED = "stream_completed"
    STREAM_FAILED = "stream_failed"
    BACKPRESSURE_APPLIED = "backpressure_applied"


class TokenStability(StrEnum):
    """
    Stability level of streamed text.

    UNSTABLE:
        Token may still be revised or buffered.

    STABLE:
        Safe for UI display.

    SENTENCE_STABLE:
        Safe boundary for TTS handoff in Step 6.
    """

    UNSTABLE = "unstable"
    STABLE = "stable"
    SENTENCE_STABLE = "sentence_stable"


class TokenStreamReason(StrEnum):
    """
    Machine-readable token streaming reasons.
    """

    STREAM_CREATED = "stream_created"
    STREAM_STARTED = "stream_started"
    CHUNK_ACCEPTED = "chunk_accepted"
    FIRST_TOKEN_RECORDED = "first_token_recorded"
    SENTENCE_STABILIZED = "sentence_stabilized"
    STREAM_COMPLETED = "stream_completed"
    STREAM_CANCELLED = "stream_cancelled"
    STREAM_FAILED = "stream_failed"
    STREAM_NOT_FOUND = "stream_not_found"
    STREAM_NOT_ACTIVE = "stream_not_active"
    BACKPRESSURE_LIMIT_REACHED = "backpressure_limit_reached"
    RUNTIME_RESET = "runtime_reset"


class TokenChunk(OrchestrationModel):
    """
    One token-stream chunk.

    Chunks are immutable typed facts. They can represent individual tokens,
    words, stable sentences, control messages, or final response text.
    """

    chunk_id: str = Field(default_factory=lambda: uuid4().hex)
    stream_id: str
    sequence: int = Field(ge=0)
    text: str
    kind: TokenStreamChunkKind = TokenStreamChunkKind.TOKEN
    stability: TokenStability = TokenStability.UNSTABLE
    emitted_at_ns: int = Field(default_factory=time.perf_counter_ns, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("chunk_id", "stream_id")
    @classmethod
    def _required_ids(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("text")
    @classmethod
    def _text_may_not_be_none(cls, value: str) -> str:
        if value is None:
            raise ValueError("text cannot be None.")

        return value


class TokenStreamEvent(OrchestrationModel):
    """
    Event emitted by the token streaming runtime.
    """

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    stream_id: str
    kind: TokenStreamEventKind
    reason: TokenStreamReason
    sequence: int | None = None
    text: str | None = None
    latency_ms: float | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("event_id", "stream_id")
    @classmethod
    def _required_ids(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class TokenStreamState(OrchestrationModel):
    """
    Runtime state for one active/completed token stream.
    """

    stream_id: str = Field(default_factory=lambda: uuid4().hex)
    trace_id: str = Field(default_factory=lambda: uuid4().hex)
    name: str
    status: TokenStreamStatus = TokenStreamStatus.CREATED
    started_at_ns: int | None = None
    first_token_at_ns: int | None = None
    completed_at_ns: int | None = None
    cancelled_at_ns: int | None = None
    failed_at_ns: int | None = None
    chunk_count: int = Field(default=0, ge=0)
    stable_sentence_count: int = Field(default=0, ge=0)
    final_text: str = ""
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("stream_id", "trace_id", "name")
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

    def total_latency_ms(self) -> float | None:
        if self.started_at_ns is None or self.completed_at_ns is None:
            return None

        return (self.completed_at_ns - self.started_at_ns) / 1_000_000.0


class TokenStreamResult(OrchestrationModel):
    """
    Result returned from stream operations.
    """

    success: bool
    reason: TokenStreamReason
    stream_id: str
    status: TokenStreamStatus
    chunk: TokenChunk | None = None
    event: TokenStreamEvent | None = None
    state: TokenStreamState | None = None
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("stream_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class TokenStreamReport(OrchestrationModel):
    """
    Final report for one token stream.
    """

    stream_id: str
    trace_id: str
    name: str
    status: TokenStreamStatus
    chunk_count: int = Field(ge=0)
    stable_sentence_count: int = Field(ge=0)
    first_token_latency_ms: float | None = None
    total_latency_ms: float | None = None
    final_text: str
    events: tuple[TokenStreamEvent, ...]
    profiler_report: PipelineProfilerReport | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("stream_id", "trace_id", "name")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class TokenStreamRuntimeSnapshot(OrchestrationModel):
    """
    Runtime diagnostics for Step 5 token streaming.
    """

    name: str
    stream_count: int = Field(ge=0)
    active_count: int = Field(ge=0)
    completed_count: int = Field(ge=0)
    cancelled_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    event_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    report_count: int = Field(ge=0)
    last_reason: TokenStreamReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class TokenStreamRuntimeConfig:
    """
    Token stream runtime configuration.
    """

    name: str = "token_stream_runtime"
    max_buffered_chunks_per_stream: int = 4096
    stabilize_sentences: bool = True
    profile_streaming: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.max_buffered_chunks_per_stream < 1:
            raise ValueError("max_buffered_chunks_per_stream must be positive.")


class TokenStreamRuntime:
    """
    Phase 7 Step 5 Token Streaming Pipeline.

    Responsibilities:
    - create and manage token streams
    - emit token chunks immediately
    - detect first-token latency
    - stabilize sentence boundaries
    - support cancellation
    - assemble final text
    - emit typed stream events
    - feed latency profiler

    Non-responsibilities:
    - no direct LLM calls
    - no TTS synthesis
    - no tool execution
    - no unsafe action execution
    """

    def __init__(
        self,
        *,
        config: TokenStreamRuntimeConfig | None = None,
        profiler: PipelineLatencyProfiler | None = None,
    ) -> None:
        self._config = config or TokenStreamRuntimeConfig()
        self._config.validate()

        self._profiler = profiler or PipelineLatencyProfiler(
            config=PipelineProfilerConfig(record_to_latency_runtime=True)
        )
        self._states: dict[str, TokenStreamState] = {}
        self._chunks: dict[str, list[TokenChunk]] = {}
        self._events: dict[str, list[TokenStreamEvent]] = {}
        self._reports: list[TokenStreamReport] = []
        self._lock = RLock()
        self._last_reason: TokenStreamReason | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def create_stream(
        self,
        *,
        name: str,
        stream_id: str | None = None,
        trace_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> TokenStreamState:
        state = TokenStreamState(
            stream_id=stream_id or uuid4().hex,
            trace_id=trace_id or uuid4().hex,
            name=name,
            metadata=metadata or {},
        )
        event = self._event(
            stream_id=state.stream_id,
            kind=TokenStreamEventKind.STREAM_CREATED,
            reason=TokenStreamReason.STREAM_CREATED,
        )

        with self._lock:
            self._states[state.stream_id] = state
            self._chunks[state.stream_id] = []
            self._events[state.stream_id] = [event]
            self._last_reason = TokenStreamReason.STREAM_CREATED

        self._profiler.start_trace(name=name, trace_id=state.trace_id)

        return state

    def start_stream(self, stream_id: str) -> TokenStreamResult:
        with self._lock:
            state = self._states.get(stream_id)

            if state is None:
                return self._missing_stream(stream_id)

            if state.status != TokenStreamStatus.CREATED:
                return self._failure(
                    stream_id=stream_id,
                    reason=TokenStreamReason.STREAM_NOT_ACTIVE,
                    status=state.status,
                    message="stream cannot be started from current state",
                    state=state,
                )

            now_ns = time.perf_counter_ns()
            updated = state.model_copy(
                update={
                    "status": TokenStreamStatus.ACTIVE,
                    "started_at_ns": now_ns,
                }
            )
            self._states[stream_id] = updated
            event = self._event(
                stream_id=stream_id,
                kind=TokenStreamEventKind.STREAM_STARTED,
                reason=TokenStreamReason.STREAM_STARTED,
            )
            self._events[stream_id].append(event)
            self._last_reason = TokenStreamReason.STREAM_STARTED

        return TokenStreamResult(
            success=True,
            reason=TokenStreamReason.STREAM_STARTED,
            stream_id=stream_id,
            status=TokenStreamStatus.ACTIVE,
            event=event,
            state=updated,
            message="token stream started",
        )

    def emit_text(
        self,
        *,
        stream_id: str,
        text: str,
        kind: TokenStreamChunkKind = TokenStreamChunkKind.TOKEN,
        stability: TokenStability = TokenStability.UNSTABLE,
        metadata: dict[str, object] | None = None,
    ) -> TokenStreamResult:
        with self._lock:
            state = self._states.get(stream_id)

            if state is None:
                return self._missing_stream(stream_id)

            if state.status != TokenStreamStatus.ACTIVE:
                return self._failure(
                    stream_id=stream_id,
                    reason=TokenStreamReason.STREAM_NOT_ACTIVE,
                    status=state.status,
                    message="stream is not active",
                    state=state,
                )

            buffered = self._chunks[stream_id]

            if len(buffered) >= self._config.max_buffered_chunks_per_stream:
                event = self._event(
                    stream_id=stream_id,
                    kind=TokenStreamEventKind.BACKPRESSURE_APPLIED,
                    reason=TokenStreamReason.BACKPRESSURE_LIMIT_REACHED,
                )
                self._events[stream_id].append(event)
                self._last_reason = TokenStreamReason.BACKPRESSURE_LIMIT_REACHED

                return TokenStreamResult(
                    success=False,
                    reason=TokenStreamReason.BACKPRESSURE_LIMIT_REACHED,
                    stream_id=stream_id,
                    status=state.status,
                    event=event,
                    state=state,
                    message="stream buffer limit reached",
                )

            now_ns = time.perf_counter_ns()
            first_token_at_ns = state.first_token_at_ns or now_ns
            sequence = state.chunk_count

            chunk = TokenChunk(
                stream_id=stream_id,
                sequence=sequence,
                text=text,
                kind=kind,
                stability=stability,
                emitted_at_ns=now_ns,
                metadata=metadata or {},
            )
            buffered.append(chunk)

            first_token_latency_ms = None

            if state.first_token_at_ns is None and state.started_at_ns is not None:
                first_token_latency_ms = (
                    now_ns - state.started_at_ns
                ) / 1_000_000.0

            stable_sentence_count = state.stable_sentence_count

            if stability == TokenStability.SENTENCE_STABLE:
                stable_sentence_count += 1

            updated = state.model_copy(
                update={
                    "first_token_at_ns": first_token_at_ns,
                    "chunk_count": state.chunk_count + 1,
                    "stable_sentence_count": stable_sentence_count,
                    "final_text": self._assemble_text(buffered),
                }
            )
            self._states[stream_id] = updated

            event_kind = (
                TokenStreamEventKind.FIRST_TOKEN_EMITTED
                if sequence == 0
                else TokenStreamEventKind.CHUNK_EMITTED
            )
            reason = (
                TokenStreamReason.FIRST_TOKEN_RECORDED
                if sequence == 0
                else TokenStreamReason.CHUNK_ACCEPTED
            )
            event = self._event(
                stream_id=stream_id,
                kind=event_kind,
                reason=reason,
                sequence=sequence,
                text=text,
                latency_ms=first_token_latency_ms,
            )
            self._events[stream_id].append(event)

            if stability == TokenStability.SENTENCE_STABLE:
                self._events[stream_id].append(
                    self._event(
                        stream_id=stream_id,
                        kind=TokenStreamEventKind.SENTENCE_STABILIZED,
                        reason=TokenStreamReason.SENTENCE_STABILIZED,
                        sequence=sequence,
                        text=text,
                    )
                )

            self._last_reason = reason

        if sequence == 0 and state.started_at_ns is not None:
            self._record_first_token_span(
                state=state,
                first_token_ns=now_ns,
            )

        return TokenStreamResult(
            success=True,
            reason=reason,
            stream_id=stream_id,
            status=TokenStreamStatus.ACTIVE,
            chunk=chunk,
            event=event,
            state=updated,
            message="token chunk accepted",
        )

    def emit_many(
        self,
        *,
        stream_id: str,
        chunks: Iterable[str],
        stabilize_sentences: bool | None = None,
    ) -> tuple[TokenStreamResult, ...]:
        should_stabilize = (
            self._config.stabilize_sentences
            if stabilize_sentences is None
            else stabilize_sentences
        )
        results: list[TokenStreamResult] = []
        sentence_buffer = ""

        for text in chunks:
            sentence_buffer += text
            stability = TokenStability.UNSTABLE
            kind = TokenStreamChunkKind.TOKEN

            if should_stabilize and self._is_sentence_boundary(sentence_buffer):
                stability = TokenStability.SENTENCE_STABLE
                kind = TokenStreamChunkKind.SENTENCE
                sentence_buffer = ""

            result = self.emit_text(
                stream_id=stream_id,
                text=text,
                kind=kind,
                stability=stability,
            )
            results.append(result)

            if not result.success:
                break

        return tuple(results)

    def complete_stream(self, stream_id: str) -> TokenStreamReport:
        with self._lock:
            state = self._states.get(stream_id)

            if state is None:
                raise ValueError(f"stream not found: {stream_id}")

            if state.status not in {
                TokenStreamStatus.ACTIVE,
                TokenStreamStatus.CREATED,
            }:
                raise ValueError("stream cannot be completed from current state")

            now_ns = time.perf_counter_ns()
            chunks = tuple(self._chunks.get(stream_id, ()))
            final_text = self._assemble_text(chunks)
            updated = state.model_copy(
                update={
                    "status": TokenStreamStatus.COMPLETED,
                    "completed_at_ns": now_ns,
                    "final_text": final_text,
                    "chunk_count": len(chunks),
                }
            )
            self._states[stream_id] = updated
            event = self._event(
                stream_id=stream_id,
                kind=TokenStreamEventKind.STREAM_COMPLETED,
                reason=TokenStreamReason.STREAM_COMPLETED,
            )
            self._events[stream_id].append(event)
            self._last_reason = TokenStreamReason.STREAM_COMPLETED

        self._record_full_stream_span(updated)

        profiler_report = None

        if self._config.profile_streaming:
            profiler_report = self._profiler.complete_trace(updated.trace_id)

        report = TokenStreamReport(
            stream_id=updated.stream_id,
            trace_id=updated.trace_id,
            name=updated.name,
            status=updated.status,
            chunk_count=updated.chunk_count,
            stable_sentence_count=updated.stable_sentence_count,
            first_token_latency_ms=updated.first_token_latency_ms(),
            total_latency_ms=updated.total_latency_ms(),
            final_text=updated.final_text,
            events=self.events_for(stream_id),
            profiler_report=profiler_report,
        )

        with self._lock:
            self._reports.append(report)

        return report

    def cancel_stream(
        self,
        stream_id: str,
        *,
        reason: str = "cancelled",
    ) -> TokenStreamResult:
        with self._lock:
            state = self._states.get(stream_id)

            if state is None:
                return self._missing_stream(stream_id)

            if state.status in {
                TokenStreamStatus.COMPLETED,
                TokenStreamStatus.CANCELLED,
                TokenStreamStatus.FAILED,
            }:
                return self._failure(
                    stream_id=stream_id,
                    reason=TokenStreamReason.STREAM_NOT_ACTIVE,
                    status=state.status,
                    message="stream is already terminal",
                    state=state,
                )

            updated = state.model_copy(
                update={
                    "status": TokenStreamStatus.CANCELLED,
                    "cancelled_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[stream_id] = updated
            event = self._event(
                stream_id=stream_id,
                kind=TokenStreamEventKind.STREAM_CANCELLED,
                reason=TokenStreamReason.STREAM_CANCELLED,
                metadata={"cancel_reason": reason},
            )
            self._events[stream_id].append(event)
            self._last_reason = TokenStreamReason.STREAM_CANCELLED

        return TokenStreamResult(
            success=True,
            reason=TokenStreamReason.STREAM_CANCELLED,
            stream_id=stream_id,
            status=TokenStreamStatus.CANCELLED,
            event=event,
            state=updated,
            message="token stream cancelled",
        )

    def fail_stream(
        self,
        stream_id: str,
        *,
        error: str,
    ) -> TokenStreamResult:
        with self._lock:
            state = self._states.get(stream_id)

            if state is None:
                return self._missing_stream(stream_id)

            updated = state.model_copy(
                update={
                    "status": TokenStreamStatus.FAILED,
                    "failed_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[stream_id] = updated
            event = self._event(
                stream_id=stream_id,
                kind=TokenStreamEventKind.STREAM_FAILED,
                reason=TokenStreamReason.STREAM_FAILED,
                metadata={"error": error},
            )
            self._events[stream_id].append(event)
            self._last_reason = TokenStreamReason.STREAM_FAILED

        return TokenStreamResult(
            success=True,
            reason=TokenStreamReason.STREAM_FAILED,
            stream_id=stream_id,
            status=TokenStreamStatus.FAILED,
            event=event,
            state=updated,
            message="token stream failed",
        )

    def state_for(self, stream_id: str) -> TokenStreamState | None:
        with self._lock:
            return self._states.get(stream_id)

    def chunks_for(self, stream_id: str) -> tuple[TokenChunk, ...]:
        with self._lock:
            return tuple(self._chunks.get(stream_id, ()))

    def events_for(self, stream_id: str) -> tuple[TokenStreamEvent, ...]:
        with self._lock:
            return tuple(self._events.get(stream_id, ()))

    def reports(self) -> tuple[TokenStreamReport, ...]:
        with self._lock:
            return tuple(self._reports)

    def latest_report(self) -> TokenStreamReport | None:
        with self._lock:
            if not self._reports:
                return None

            return self._reports[-1]

    def snapshot(self) -> TokenStreamRuntimeSnapshot:
        with self._lock:
            states = tuple(self._states.values())

            return TokenStreamRuntimeSnapshot(
                name=self.name,
                stream_count=len(states),
                active_count=sum(
                    1 for state in states if state.status == TokenStreamStatus.ACTIVE
                ),
                completed_count=sum(
                    1
                    for state in states
                    if state.status == TokenStreamStatus.COMPLETED
                ),
                cancelled_count=sum(
                    1
                    for state in states
                    if state.status == TokenStreamStatus.CANCELLED
                ),
                failed_count=sum(
                    1 for state in states if state.status == TokenStreamStatus.FAILED
                ),
                event_count=sum(len(events) for events in self._events.values()),
                chunk_count=sum(len(chunks) for chunks in self._chunks.values()),
                report_count=len(self._reports),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        with self._lock:
            self._states.clear()
            self._chunks.clear()
            self._events.clear()
            self._reports.clear()
            self._last_reason = TokenStreamReason.RUNTIME_RESET

    def _record_first_token_span(
        self,
        *,
        state: TokenStreamState,
        first_token_ns: int,
    ) -> None:
        if state.started_at_ns is None:
            return

        self._profiler.record_stage(
            trace_id=state.trace_id,
            stage=PipelineStage.LLM_FIRST_TOKEN,
            operation=LatencyOperation.LLM_FIRST_TOKEN,
            subsystem=LatencySubsystem.COGNITION,
            start_ns=state.started_at_ns,
            end_ns=first_token_ns,
            metadata={"stream_id": state.stream_id},
        )

    def _record_full_stream_span(self, state: TokenStreamState) -> None:
        if state.started_at_ns is None or state.completed_at_ns is None:
            return

        self._profiler.record_stage(
            trace_id=state.trace_id,
            stage=PipelineStage.LLM_STREAMING,
            operation=LatencyOperation.LLM_FULL_RESPONSE,
            subsystem=LatencySubsystem.COGNITION,
            start_ns=state.started_at_ns,
            end_ns=state.completed_at_ns,
            metadata={"stream_id": state.stream_id},
        )

    @staticmethod
    def _assemble_text(chunks: Iterable[TokenChunk]) -> str:
        return "".join(chunk.text for chunk in chunks)

    @staticmethod
    def _is_sentence_boundary(text: str) -> bool:
        stripped = text.rstrip()

        return stripped.endswith((".", "!", "?"))

    @staticmethod
    def _event(
        *,
        stream_id: str,
        kind: TokenStreamEventKind,
        reason: TokenStreamReason,
        sequence: int | None = None,
        text: str | None = None,
        latency_ms: float | None = None,
        metadata: dict[str, object] | None = None,
    ) -> TokenStreamEvent:
        return TokenStreamEvent(
            stream_id=stream_id,
            kind=kind,
            reason=reason,
            sequence=sequence,
            text=text,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )

    @staticmethod
    def _missing_stream(stream_id: str) -> TokenStreamResult:
        return TokenStreamResult(
            success=False,
            reason=TokenStreamReason.STREAM_NOT_FOUND,
            stream_id=stream_id,
            status=TokenStreamStatus.FAILED,
            message="token stream not found",
        )

    @staticmethod
    def _failure(
        *,
        stream_id: str,
        reason: TokenStreamReason,
        status: TokenStreamStatus,
        message: str,
        state: TokenStreamState | None = None,
    ) -> TokenStreamResult:
        return TokenStreamResult(
            success=False,
            reason=reason,
            stream_id=stream_id,
            status=status,
            state=state,
            message=message,
        )