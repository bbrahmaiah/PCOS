from __future__ import annotations

import re
import time
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
from jarvis.latency.token_streaming import (
    TokenChunk,
    TokenStability,
    TokenStreamChunkKind,
)
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class StreamingTTSStatus(StrEnum):
    """
    Streaming TTS lifecycle status.
    """

    CREATED = "created"
    ACTIVE = "active"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


class SpeechBoundaryKind(StrEnum):
    """
    Reason a speech chunk was flushed.
    """

    SENTENCE_BOUNDARY = "sentence_boundary"
    PAUSE_MARKER = "pause_marker"
    MAX_WAIT_FLUSH = "max_wait_flush"
    FINAL_FLUSH = "final_flush"
    MANUAL_FLUSH = "manual_flush"


class StreamingTTSEventKind(StrEnum):
    """
    Streaming TTS event kind.
    """

    SESSION_CREATED = "session_created"
    SESSION_STARTED = "session_started"
    TOKEN_ACCEPTED = "token_accepted"
    SPEECH_CHUNK_CREATED = "speech_chunk_created"
    AUDIO_CHUNK_READY = "audio_chunk_ready"
    FIRST_AUDIO_READY = "first_audio_ready"
    PREBUFFER_READY = "prebuffer_ready"
    SESSION_CANCELLED = "session_cancelled"
    SESSION_COMPLETED = "session_completed"
    SESSION_FAILED = "session_failed"


class StreamingTTSReason(StrEnum):
    """
    Machine-readable Streaming TTS reasons.
    """

    SESSION_CREATED = "session_created"
    SESSION_STARTED = "session_started"
    TOKEN_BUFFERED = "token_buffered"
    BOUNDARY_DETECTED = "boundary_detected"
    MAX_WAIT_REACHED = "max_wait_reached"
    AUDIO_SYNTHESIZED = "audio_synthesized"
    FIRST_AUDIO_RECORDED = "first_audio_recorded"
    PREBUFFER_READY = "prebuffer_ready"
    SESSION_COMPLETED = "session_completed"
    SESSION_CANCELLED = "session_cancelled"
    SESSION_FAILED = "session_failed"
    SESSION_NOT_FOUND = "session_not_found"
    SESSION_NOT_ACTIVE = "session_not_active"
    EMPTY_FLUSH_SKIPPED = "empty_flush_skipped"
    RUNTIME_RESET = "runtime_reset"


class SpeechChunk(OrchestrationModel):
    """
    Text chunk sent to TTS.

    This is not the full LLM response. It is a natural speech segment triggered
    by sentence boundary, pause marker, maximum wait, or final flush.
    """

    speech_chunk_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    sequence: int = Field(ge=0)
    text: str
    boundary_kind: SpeechBoundaryKind
    word_count: int = Field(ge=0)
    source_token_count: int = Field(ge=0)
    created_at_ns: int = Field(default_factory=time.perf_counter_ns, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("speech_chunk_id", "session_id", "text")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class AudioChunk(OrchestrationModel):
    """
    Synthesized audio chunk ready for progressive playback.
    """

    audio_chunk_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    speech_chunk_id: str
    sequence: int = Field(ge=0)
    text: str
    audio_bytes: bytes
    estimated_duration_ms: float = Field(ge=0)
    ready_at_ns: int = Field(default_factory=time.perf_counter_ns, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("audio_chunk_id", "session_id", "speech_chunk_id", "text")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @property
    def byte_count(self) -> int:
        return len(self.audio_bytes)


class StreamingTTSEvent(OrchestrationModel):
    """
    Typed event emitted by the Streaming TTS runtime.
    """

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    kind: StreamingTTSEventKind
    reason: StreamingTTSReason
    sequence: int | None = None
    text: str | None = None
    latency_ms: float | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("event_id", "session_id")
    @classmethod
    def _required_ids(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class StreamingTTSSessionState(OrchestrationModel):
    """
    Runtime state for one streaming TTS session.
    """

    session_id: str = Field(default_factory=lambda: uuid4().hex)
    trace_id: str = Field(default_factory=lambda: uuid4().hex)
    name: str
    status: StreamingTTSStatus = StreamingTTSStatus.CREATED
    started_at_ns: int | None = None
    first_audio_at_ns: int | None = None
    completed_at_ns: int | None = None
    cancelled_at_ns: int | None = None
    failed_at_ns: int | None = None
    token_count: int = Field(default=0, ge=0)
    speech_chunk_count: int = Field(default=0, ge=0)
    audio_chunk_count: int = Field(default=0, ge=0)
    buffered_text: str = ""
    buffered_token_count: int = Field(default=0, ge=0)
    audio_buffer_ms: float = Field(default=0.0, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("session_id", "trace_id", "name")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    def first_audio_latency_ms(self) -> float | None:
        if self.started_at_ns is None or self.first_audio_at_ns is None:
            return None

        return (self.first_audio_at_ns - self.started_at_ns) / 1_000_000.0

    def total_latency_ms(self) -> float | None:
        if self.started_at_ns is None or self.completed_at_ns is None:
            return None

        return (self.completed_at_ns - self.started_at_ns) / 1_000_000.0


class StreamingTTSResult(OrchestrationModel):
    """
    Result from a Streaming TTS operation.
    """

    success: bool
    reason: StreamingTTSReason
    session_id: str
    status: StreamingTTSStatus
    speech_chunk: SpeechChunk | None = None
    audio_chunk: AudioChunk | None = None
    event: StreamingTTSEvent | None = None
    state: StreamingTTSSessionState | None = None
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


class StreamingTTSReport(OrchestrationModel):
    """
    Final report for one streaming TTS session.
    """

    session_id: str
    trace_id: str
    name: str
    status: StreamingTTSStatus
    token_count: int = Field(ge=0)
    speech_chunk_count: int = Field(ge=0)
    audio_chunk_count: int = Field(ge=0)
    first_audio_latency_ms: float | None = None
    total_latency_ms: float | None = None
    audio_buffer_ms: float = Field(ge=0)
    speech_chunks: tuple[SpeechChunk, ...]
    audio_chunks: tuple[AudioChunk, ...]
    events: tuple[StreamingTTSEvent, ...]
    profiler_report: PipelineProfilerReport | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("session_id", "trace_id", "name")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class StreamingTTSRuntimeSnapshot(OrchestrationModel):
    """
    Runtime diagnostics for Phase 7 Step 6.
    """

    name: str
    session_count: int = Field(ge=0)
    active_count: int = Field(ge=0)
    completed_count: int = Field(ge=0)
    cancelled_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    speech_chunk_count: int = Field(ge=0)
    audio_chunk_count: int = Field(ge=0)
    event_count: int = Field(ge=0)
    report_count: int = Field(ge=0)
    last_reason: StreamingTTSReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        return cleaned


class TextToSpeechAdapter(Protocol):
    """
    TTS adapter boundary.

    Real TTS engines will implement this later. Step 6 uses a fake adapter in
    tests so the streaming runtime is deterministic.
    """

    def synthesize(self, text: str) -> bytes:
        """
        Convert text into audio bytes.
        """


class FakeStreamingTTSAdapter:
    """
    Deterministic fake TTS adapter.

    This avoids real audio dependencies while preserving streaming contracts.
    """

    def synthesize(self, text: str) -> bytes:
        return f"AUDIO::{text}".encode()


@dataclass(frozen=True, slots=True)
class StreamingTTSRuntimeConfig:
    """
    Streaming TTS runtime configuration.
    """

    name: str = "streaming_tts_runtime"
    minimum_words_per_chunk: int = 4
    maximum_wait_ms: float = 300.0
    prebuffer_sentence_count: int = 2
    estimated_ms_per_word: float = 120.0
    profile_streaming_tts: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.minimum_words_per_chunk < 1:
            raise ValueError("minimum_words_per_chunk must be positive.")

        if self.maximum_wait_ms <= 0:
            raise ValueError("maximum_wait_ms must be positive.")

        if self.prebuffer_sentence_count < 1:
            raise ValueError("prebuffer_sentence_count must be positive.")

        if self.estimated_ms_per_word <= 0:
            raise ValueError("estimated_ms_per_word must be positive.")


class StreamingTTSRuntime:
    """
    Phase 7 Step 6 Streaming TTS Integration.

    Responsibilities:
    - consume streamed token chunks
    - accumulate natural speech text
    - detect sentence/pause/max-wait boundaries
    - synthesize progressive audio chunks
    - maintain small audio buffer ahead of token generation
    - track first-audio latency
    - support cancellation/failure
    - feed the latency profiler

    Non-responsibilities:
    - no real playback device control
    - no direct LLM calls
    - no tool execution
    - no unsafe action execution
    """

    def __init__(
        self,
        *,
        config: StreamingTTSRuntimeConfig | None = None,
        adapter: TextToSpeechAdapter | None = None,
        profiler: PipelineLatencyProfiler | None = None,
    ) -> None:
        self._config = config or StreamingTTSRuntimeConfig()
        self._config.validate()

        self._adapter = adapter or FakeStreamingTTSAdapter()
        self._profiler = profiler or PipelineLatencyProfiler(
            config=PipelineProfilerConfig(record_to_latency_runtime=True)
        )
        self._states: dict[str, StreamingTTSSessionState] = {}
        self._speech_chunks: dict[str, list[SpeechChunk]] = {}
        self._audio_chunks: dict[str, list[AudioChunk]] = {}
        self._events: dict[str, list[StreamingTTSEvent]] = {}
        self._reports: list[StreamingTTSReport] = []
        self._lock = RLock()
        self._last_reason: StreamingTTSReason | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def create_session(
        self,
        *,
        name: str,
        session_id: str | None = None,
        trace_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> StreamingTTSSessionState:
        state = StreamingTTSSessionState(
            session_id=session_id or uuid4().hex,
            trace_id=trace_id or uuid4().hex,
            name=name,
            metadata=metadata or {},
        )
        event = self._event(
            session_id=state.session_id,
            kind=StreamingTTSEventKind.SESSION_CREATED,
            reason=StreamingTTSReason.SESSION_CREATED,
        )

        with self._lock:
            self._states[state.session_id] = state
            self._speech_chunks[state.session_id] = []
            self._audio_chunks[state.session_id] = []
            self._events[state.session_id] = [event]
            self._last_reason = StreamingTTSReason.SESSION_CREATED

        self._profiler.start_trace(name=name, trace_id=state.trace_id)

        return state

    def start_session(self, session_id: str) -> StreamingTTSResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            if state.status != StreamingTTSStatus.CREATED:
                return self._failure(
                    session_id=session_id,
                    reason=StreamingTTSReason.SESSION_NOT_ACTIVE,
                    status=state.status,
                    message="session cannot be started from current state",
                    state=state,
                )

            updated = state.model_copy(
                update={
                    "status": StreamingTTSStatus.ACTIVE,
                    "started_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = updated
            event = self._event(
                session_id=session_id,
                kind=StreamingTTSEventKind.SESSION_STARTED,
                reason=StreamingTTSReason.SESSION_STARTED,
            )
            self._events[session_id].append(event)
            self._last_reason = StreamingTTSReason.SESSION_STARTED

        return StreamingTTSResult(
            success=True,
            reason=StreamingTTSReason.SESSION_STARTED,
            session_id=session_id,
            status=StreamingTTSStatus.ACTIVE,
            event=event,
            state=updated,
            message="streaming TTS session started",
        )

    def accept_token_chunk(
        self,
        *,
        session_id: str,
        chunk: TokenChunk,
    ) -> StreamingTTSResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            if state.status != StreamingTTSStatus.ACTIVE:
                return self._failure(
                    session_id=session_id,
                    reason=StreamingTTSReason.SESSION_NOT_ACTIVE,
                    status=state.status,
                    message="streaming TTS session is not active",
                    state=state,
                )

            buffered_text = state.buffered_text + chunk.text
            buffered_count = state.buffered_token_count + 1
            updated = state.model_copy(
                update={
                    "buffered_text": buffered_text,
                    "buffered_token_count": buffered_count,
                    "token_count": state.token_count + 1,
                }
            )
            self._states[session_id] = updated

            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=StreamingTTSEventKind.TOKEN_ACCEPTED,
                    reason=StreamingTTSReason.TOKEN_BUFFERED,
                    sequence=chunk.sequence,
                    text=chunk.text,
                )
            )

        boundary = self._detect_boundary(
            text=buffered_text,
            token_stability=chunk.stability,
            now_ns=chunk.emitted_at_ns,
            state=updated,
        )

        if boundary is None:
            return StreamingTTSResult(
                success=True,
                reason=StreamingTTSReason.TOKEN_BUFFERED,
                session_id=session_id,
                status=StreamingTTSStatus.ACTIVE,
                state=updated,
                message="token buffered for streaming TTS",
            )

        return self.flush(session_id=session_id, boundary_kind=boundary)

    def flush(
        self,
        *,
        session_id: str,
        boundary_kind: SpeechBoundaryKind = SpeechBoundaryKind.MANUAL_FLUSH,
    ) -> StreamingTTSResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            if state.status != StreamingTTSStatus.ACTIVE:
                return self._failure(
                    session_id=session_id,
                    reason=StreamingTTSReason.SESSION_NOT_ACTIVE,
                    status=state.status,
                    message="streaming TTS session is not active",
                    state=state,
                )

            text = self._normalize_text(state.buffered_text)

            if not text:
                return StreamingTTSResult(
                    success=False,
                    reason=StreamingTTSReason.EMPTY_FLUSH_SKIPPED,
                    session_id=session_id,
                    status=state.status,
                    state=state,
                    message="empty TTS flush skipped",
                )

            speech_sequence = state.speech_chunk_count
            speech_chunk = SpeechChunk(
                session_id=session_id,
                sequence=speech_sequence,
                text=text,
                boundary_kind=boundary_kind,
                word_count=self._word_count(text),
                source_token_count=state.buffered_token_count,
            )
            self._speech_chunks[session_id].append(speech_chunk)

        synth_start_ns = time.perf_counter_ns()
        audio_bytes = self._adapter.synthesize(text)
        synth_end_ns = time.perf_counter_ns()

        audio_chunk = AudioChunk(
            session_id=session_id,
            speech_chunk_id=speech_chunk.speech_chunk_id,
            sequence=speech_chunk.sequence,
            text=text,
            audio_bytes=audio_bytes,
            estimated_duration_ms=self._estimated_audio_ms(text),
            ready_at_ns=synth_end_ns,
        )

        with self._lock:
            current = self._states[session_id]
            first_audio_at_ns = current.first_audio_at_ns or synth_end_ns
            audio_buffer_ms = current.audio_buffer_ms + (
                audio_chunk.estimated_duration_ms
            )
            updated = current.model_copy(
                update={
                    "first_audio_at_ns": first_audio_at_ns,
                    "speech_chunk_count": current.speech_chunk_count + 1,
                    "audio_chunk_count": current.audio_chunk_count + 1,
                    "buffered_text": "",
                    "buffered_token_count": 0,
                    "audio_buffer_ms": audio_buffer_ms,
                }
            )
            self._states[session_id] = updated
            self._audio_chunks[session_id].append(audio_chunk)

            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=StreamingTTSEventKind.SPEECH_CHUNK_CREATED,
                    reason=StreamingTTSReason.BOUNDARY_DETECTED,
                    sequence=speech_chunk.sequence,
                    text=speech_chunk.text,
                )
            )
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=StreamingTTSEventKind.AUDIO_CHUNK_READY,
                    reason=StreamingTTSReason.AUDIO_SYNTHESIZED,
                    sequence=audio_chunk.sequence,
                    text=audio_chunk.text,
                    latency_ms=(synth_end_ns - synth_start_ns) / 1_000_000.0,
                )
            )

            if current.first_audio_at_ns is None:
                self._events[session_id].append(
                    self._event(
                        session_id=session_id,
                        kind=StreamingTTSEventKind.FIRST_AUDIO_READY,
                        reason=StreamingTTSReason.FIRST_AUDIO_RECORDED,
                        sequence=audio_chunk.sequence,
                        latency_ms=updated.first_audio_latency_ms(),
                    )
                )

            if updated.audio_chunk_count >= self._config.prebuffer_sentence_count:
                self._events[session_id].append(
                    self._event(
                        session_id=session_id,
                        kind=StreamingTTSEventKind.PREBUFFER_READY,
                        reason=StreamingTTSReason.PREBUFFER_READY,
                        metadata={"audio_buffer_ms": audio_buffer_ms},
                    )
                )

            self._last_reason = StreamingTTSReason.AUDIO_SYNTHESIZED

        self._record_tts_span(
            state=updated,
            start_ns=synth_start_ns,
            end_ns=synth_end_ns,
            first_audio=current.first_audio_at_ns is None,
        )

        return StreamingTTSResult(
            success=True,
            reason=StreamingTTSReason.AUDIO_SYNTHESIZED,
            session_id=session_id,
            status=StreamingTTSStatus.ACTIVE,
            speech_chunk=speech_chunk,
            audio_chunk=audio_chunk,
            state=updated,
            message="streaming TTS audio chunk ready",
        )

    def complete_session(self, session_id: str) -> StreamingTTSReport:
        state = self.state_for(session_id)

        if state is None:
            raise ValueError(f"streaming TTS session not found: {session_id}")

        if state.status not in {
            StreamingTTSStatus.ACTIVE,
            StreamingTTSStatus.CREATED,
        }:
            raise ValueError("session cannot be completed from current state")

        if state.status == StreamingTTSStatus.ACTIVE and state.buffered_text.strip():
            self.flush(
                session_id=session_id,
                boundary_kind=SpeechBoundaryKind.FINAL_FLUSH,
            )

        with self._lock:
            current = self._states[session_id]
            completed = current.model_copy(
                update={
                    "status": StreamingTTSStatus.COMPLETED,
                    "completed_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = completed
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=StreamingTTSEventKind.SESSION_COMPLETED,
                    reason=StreamingTTSReason.SESSION_COMPLETED,
                )
            )
            self._last_reason = StreamingTTSReason.SESSION_COMPLETED

        profiler_report = None

        if self._config.profile_streaming_tts:
            profiler_report = self._profiler.complete_trace(completed.trace_id)

        report = StreamingTTSReport(
            session_id=completed.session_id,
            trace_id=completed.trace_id,
            name=completed.name,
            status=completed.status,
            token_count=completed.token_count,
            speech_chunk_count=completed.speech_chunk_count,
            audio_chunk_count=completed.audio_chunk_count,
            first_audio_latency_ms=completed.first_audio_latency_ms(),
            total_latency_ms=completed.total_latency_ms(),
            audio_buffer_ms=completed.audio_buffer_ms,
            speech_chunks=self.speech_chunks_for(session_id),
            audio_chunks=self.audio_chunks_for(session_id),
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
    ) -> StreamingTTSResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            if state.status in {
                StreamingTTSStatus.CANCELLED,
                StreamingTTSStatus.COMPLETED,
                StreamingTTSStatus.FAILED,
            }:
                return self._failure(
                    session_id=session_id,
                    reason=StreamingTTSReason.SESSION_NOT_ACTIVE,
                    status=state.status,
                    message="session already terminal",
                    state=state,
                )

            updated = state.model_copy(
                update={
                    "status": StreamingTTSStatus.CANCELLED,
                    "cancelled_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = updated
            event = self._event(
                session_id=session_id,
                kind=StreamingTTSEventKind.SESSION_CANCELLED,
                reason=StreamingTTSReason.SESSION_CANCELLED,
                metadata={"cancel_reason": reason},
            )
            self._events[session_id].append(event)
            self._last_reason = StreamingTTSReason.SESSION_CANCELLED

        return StreamingTTSResult(
            success=True,
            reason=StreamingTTSReason.SESSION_CANCELLED,
            session_id=session_id,
            status=StreamingTTSStatus.CANCELLED,
            event=event,
            state=updated,
            message="streaming TTS session cancelled",
        )

    def fail_session(
        self,
        session_id: str,
        *,
        error: str,
    ) -> StreamingTTSResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            updated = state.model_copy(
                update={
                    "status": StreamingTTSStatus.FAILED,
                    "failed_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = updated
            event = self._event(
                session_id=session_id,
                kind=StreamingTTSEventKind.SESSION_FAILED,
                reason=StreamingTTSReason.SESSION_FAILED,
                metadata={"error": error},
            )
            self._events[session_id].append(event)
            self._last_reason = StreamingTTSReason.SESSION_FAILED

        return StreamingTTSResult(
            success=True,
            reason=StreamingTTSReason.SESSION_FAILED,
            session_id=session_id,
            status=StreamingTTSStatus.FAILED,
            event=event,
            state=updated,
            message="streaming TTS session failed",
        )

    def state_for(self, session_id: str) -> StreamingTTSSessionState | None:
        with self._lock:
            return self._states.get(session_id)

    def speech_chunks_for(self, session_id: str) -> tuple[SpeechChunk, ...]:
        with self._lock:
            return tuple(self._speech_chunks.get(session_id, ()))

    def audio_chunks_for(self, session_id: str) -> tuple[AudioChunk, ...]:
        with self._lock:
            return tuple(self._audio_chunks.get(session_id, ()))

    def events_for(self, session_id: str) -> tuple[StreamingTTSEvent, ...]:
        with self._lock:
            return tuple(self._events.get(session_id, ()))

    def reports(self) -> tuple[StreamingTTSReport, ...]:
        with self._lock:
            return tuple(self._reports)

    def latest_report(self) -> StreamingTTSReport | None:
        with self._lock:
            if not self._reports:
                return None

            return self._reports[-1]

    def snapshot(self) -> StreamingTTSRuntimeSnapshot:
        with self._lock:
            states = tuple(self._states.values())

            return StreamingTTSRuntimeSnapshot(
                name=self.name,
                session_count=len(states),
                active_count=sum(
                    1 for state in states if state.status == StreamingTTSStatus.ACTIVE
                ),
                completed_count=sum(
                    1
                    for state in states
                    if state.status == StreamingTTSStatus.COMPLETED
                ),
                cancelled_count=sum(
                    1
                    for state in states
                    if state.status == StreamingTTSStatus.CANCELLED
                ),
                failed_count=sum(
                    1 for state in states if state.status == StreamingTTSStatus.FAILED
                ),
                speech_chunk_count=sum(
                    len(chunks) for chunks in self._speech_chunks.values()
                ),
                audio_chunk_count=sum(
                    len(chunks) for chunks in self._audio_chunks.values()
                ),
                event_count=sum(len(events) for events in self._events.values()),
                report_count=len(self._reports),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        with self._lock:
            self._states.clear()
            self._speech_chunks.clear()
            self._audio_chunks.clear()
            self._events.clear()
            self._reports.clear()
            self._last_reason = StreamingTTSReason.RUNTIME_RESET

    def _detect_boundary(
        self,
        *,
        text: str,
        token_stability: TokenStability,
        now_ns: int,
        state: StreamingTTSSessionState,
    ) -> SpeechBoundaryKind | None:
        normalized = self._normalize_text(text)
        word_count = self._word_count(normalized)

        if word_count == 0:
            return None

        if self._has_pause_marker(normalized) and word_count >= 2:
            return SpeechBoundaryKind.PAUSE_MARKER

        if (
            self._has_sentence_boundary(normalized)
            and word_count >= self._config.minimum_words_per_chunk
        ):
            return SpeechBoundaryKind.SENTENCE_BOUNDARY

        if token_stability == TokenStability.SENTENCE_STABLE:
            return SpeechBoundaryKind.SENTENCE_BOUNDARY

        if state.started_at_ns is not None:
            elapsed_ms = (now_ns - state.started_at_ns) / 1_000_000.0

            if (
                elapsed_ms >= self._config.maximum_wait_ms
                and word_count >= self._config.minimum_words_per_chunk
            ):
                return SpeechBoundaryKind.MAX_WAIT_FLUSH

        return None

    def _record_tts_span(
        self,
        *,
        state: StreamingTTSSessionState,
        start_ns: int,
        end_ns: int,
        first_audio: bool,
    ) -> None:
        stage = (
            PipelineStage.TTS_FIRST_AUDIO
            if first_audio
            else PipelineStage.TTS_STREAMING
        )
        operation = LatencyOperation.TTS_FIRST_AUDIO

        self._profiler.record_stage(
            trace_id=state.trace_id,
            stage=stage,
            operation=operation,
            subsystem=LatencySubsystem.PRESENCE,
            start_ns=start_ns,
            end_ns=end_ns,
            metadata={"session_id": state.session_id},
        )

    def _estimated_audio_ms(self, text: str) -> float:
        return self._word_count(text) * self._config.estimated_ms_per_word

    @staticmethod
    def _word_count(text: str) -> int:
        return len(re.findall(r"\b\w+\b", text))

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _has_sentence_boundary(text: str) -> bool:
        return text.rstrip().endswith((".", "?", "!"))

    @staticmethod
    def _has_pause_marker(text: str) -> bool:
        return "—" in text or "..." in text

    @staticmethod
    def _event(
        *,
        session_id: str,
        kind: StreamingTTSEventKind,
        reason: StreamingTTSReason,
        sequence: int | None = None,
        text: str | None = None,
        latency_ms: float | None = None,
        metadata: dict[str, object] | None = None,
    ) -> StreamingTTSEvent:
        return StreamingTTSEvent(
            session_id=session_id,
            kind=kind,
            reason=reason,
            sequence=sequence,
            text=text,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )

    @staticmethod
    def _missing_session(session_id: str) -> StreamingTTSResult:
        return StreamingTTSResult(
            success=False,
            reason=StreamingTTSReason.SESSION_NOT_FOUND,
            session_id=session_id,
            status=StreamingTTSStatus.FAILED,
            message="streaming TTS session not found",
        )

    @staticmethod
    def _failure(
        *,
        session_id: str,
        reason: StreamingTTSReason,
        status: StreamingTTSStatus,
        message: str,
        state: StreamingTTSSessionState | None = None,
    ) -> StreamingTTSResult:
        return StreamingTTSResult(
            success=False,
            reason=reason,
            session_id=session_id,
            status=status,
            state=state,
            message=message,
        )


def token_chunk(
    *,
    stream_id: str = "token-stream",
    sequence: int = 0,
    text: str,
    stability: TokenStability = TokenStability.UNSTABLE,
) -> TokenChunk:
    """
    Test/helper factory for feeding TTS from token streams.
    """

    return TokenChunk(
        stream_id=stream_id,
        sequence=sequence,
        text=text,
        kind=TokenStreamChunkKind.TOKEN,
        stability=stability,
    )