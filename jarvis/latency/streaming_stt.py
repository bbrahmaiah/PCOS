from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from typing import Protocol
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.latency.models import LatencyOperation, LatencySubsystem
from jarvis.latency.profiler import (
    PipelineLatencyProfiler,
    PipelineProfilerConfig,
    PipelineProfilerReport,
    PipelineStage,
)
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class StreamingSTTStatus(StrEnum):
    """
    Streaming STT session lifecycle status.
    """

    CREATED = "created"
    ACTIVE = "active"
    FINALIZED = "finalized"
    CANCELLED = "cancelled"
    FAILED = "failed"


class AudioChunkKind(StrEnum):
    """
    Audio chunk kind.
    """

    SPEECH = "speech"
    SILENCE = "silence"
    NOISE = "noise"
    END_OF_TURN = "end_of_turn"


class PartialTranscriptStability(StrEnum):
    """
    Partial transcript stability level.
    """

    UNSTABLE = "unstable"
    STABLE = "stable"
    FINAL = "final"


class PartialIntentKind(StrEnum):
    """
    Lightweight partial intent classification.

    This is intentionally small and local. It is not full cognition.
    """

    UNKNOWN = "unknown"
    QUESTION = "question"
    COMMAND = "command"
    DEBUGGING = "debugging"
    RESEARCH = "research"
    MEMORY_RECALL = "memory_recall"
    TOOL_USE = "tool_use"
    CONVERSATION = "conversation"


class SpeculativeWorkKind(StrEnum):
    """
    Early work that may begin from partial transcript.

    This work must be discardable if the final transcript differs.
    """

    MEMORY_PREFETCH = "memory_prefetch"
    CONTEXT_PREWARM = "context_prewarm"
    TOOL_PLANNER_HINT = "tool_planner_hint"
    LLM_CONTEXT_PREWARM = "llm_context_prewarm"


class SpeculativeWorkStatus(StrEnum):
    """
    Speculative work lifecycle.
    """

    PROPOSED = "proposed"
    STARTED = "started"
    CANCELLED = "cancelled"
    CONFIRMED = "confirmed"
    DISCARDED = "discarded"


class StreamingSTTEventKind(StrEnum):
    """
    Streaming STT event kind.
    """

    SESSION_CREATED = "session_created"
    SESSION_STARTED = "session_started"
    AUDIO_ACCEPTED = "audio_accepted"
    PARTIAL_TRANSCRIPT_EMITTED = "partial_transcript_emitted"
    PARTIAL_INTENT_DETECTED = "partial_intent_detected"
    SPECULATIVE_WORK_STARTED = "speculative_work_started"
    SPECULATIVE_WORK_CANCELLED = "speculative_work_cancelled"
    FINAL_TRANSCRIPT_READY = "final_transcript_ready"
    SESSION_CANCELLED = "session_cancelled"
    SESSION_FAILED = "session_failed"


class StreamingSTTReason(StrEnum):
    """
    Machine-readable Streaming STT reasons.
    """

    SESSION_CREATED = "session_created"
    SESSION_STARTED = "session_started"
    AUDIO_CHUNK_ACCEPTED = "audio_chunk_accepted"
    PARTIAL_TRANSCRIPT_READY = "partial_transcript_ready"
    FIRST_PARTIAL_RECORDED = "first_partial_recorded"
    INTENT_CONFIDENCE_LOW = "intent_confidence_low"
    INTENT_CONFIDENCE_HIGH = "intent_confidence_high"
    SPECULATIVE_WORK_STARTED = "speculative_work_started"
    SPECULATIVE_WORK_CANCELLED = "speculative_work_cancelled"
    SPECULATIVE_WORK_CONFIRMED = "speculative_work_confirmed"
    SPECULATIVE_WORK_DISCARDED = "speculative_work_discarded"
    FINAL_TRANSCRIPT_READY = "final_transcript_ready"
    FINAL_TRANSCRIPT_DIVERGED = "final_transcript_diverged"
    SESSION_CANCELLED = "session_cancelled"
    SESSION_FAILED = "session_failed"
    SESSION_NOT_FOUND = "session_not_found"
    SESSION_NOT_ACTIVE = "session_not_active"
    RUNTIME_RESET = "runtime_reset"


class AudioChunk(OrchestrationModel):
    """
    Streaming audio chunk.

    Chunk size target: 100-200ms audio windows.
    """

    audio_chunk_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    sequence: int = Field(ge=0)
    duration_ms: float = Field(gt=0)
    data: bytes = b""
    kind: AudioChunkKind = AudioChunkKind.SPEECH
    received_at_ns: int = Field(default_factory=time.perf_counter_ns, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("audio_chunk_id", "session_id")
    @classmethod
    def _required_ids(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class PartialTranscript(OrchestrationModel):
    """
    Partial transcript emitted while the user is still speaking.
    """

    transcript_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    sequence: int = Field(ge=0)
    text: str
    stability: PartialTranscriptStability = PartialTranscriptStability.UNSTABLE
    confidence: float = Field(ge=0, le=1)
    source_audio_sequence: int = Field(ge=0)
    emitted_at_ns: int = Field(default_factory=time.perf_counter_ns, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("transcript_id", "session_id")
    @classmethod
    def _required_ids(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("text")
    @classmethod
    def _allow_empty_but_not_none(cls, value: str) -> str:
        if value is None:
            raise ValueError("text cannot be None.")

        return value


class PartialIntent(OrchestrationModel):
    """
    Lightweight intent classification from partial transcript.
    """

    intent_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    transcript_id: str
    kind: PartialIntentKind
    confidence: float = Field(ge=0, le=1)
    text: str
    created_at_ns: int = Field(default_factory=time.perf_counter_ns, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("intent_id", "session_id", "transcript_id")
    @classmethod
    def _required_ids(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class SpeculativeWorkHint(OrchestrationModel):
    """
    Speculative work hint started from partial transcript.

    These are hints/contracts, not direct execution.
    """

    hint_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    intent_id: str
    kind: SpeculativeWorkKind
    status: SpeculativeWorkStatus = SpeculativeWorkStatus.PROPOSED
    reason: str
    confidence: float = Field(ge=0, le=1)
    discardable: bool = True
    cancellable: bool = True
    created_at_ns: int = Field(default_factory=time.perf_counter_ns, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("hint_id", "session_id", "intent_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @model_validator(mode="after")
    def _must_be_safe(self) -> SpeculativeWorkHint:
        if not self.discardable:
            raise ValueError("speculative work must be discardable.")

        if not self.cancellable:
            raise ValueError("speculative work must be cancellable.")

        return self


class StreamingSTTEvent(OrchestrationModel):
    """
    Streaming STT runtime event.
    """

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    kind: StreamingSTTEventKind
    reason: StreamingSTTReason
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


class StreamingSTTSessionState(OrchestrationModel):
    """
    Runtime state for one streaming STT session.
    """

    session_id: str = Field(default_factory=lambda: uuid4().hex)
    trace_id: str = Field(default_factory=lambda: uuid4().hex)
    name: str
    status: StreamingSTTStatus = StreamingSTTStatus.CREATED
    started_at_ns: int | None = None
    first_partial_at_ns: int | None = None
    finalized_at_ns: int | None = None
    cancelled_at_ns: int | None = None
    failed_at_ns: int | None = None
    audio_chunk_count: int = Field(default=0, ge=0)
    partial_count: int = Field(default=0, ge=0)
    intent_count: int = Field(default=0, ge=0)
    speculative_hint_count: int = Field(default=0, ge=0)
    latest_partial_text: str = ""
    final_transcript: str = ""
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("session_id", "trace_id", "name")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    def first_partial_latency_ms(self) -> float | None:
        if self.started_at_ns is None or self.first_partial_at_ns is None:
            return None

        return (self.first_partial_at_ns - self.started_at_ns) / 1_000_000.0

    def finalization_latency_ms(self) -> float | None:
        if self.started_at_ns is None or self.finalized_at_ns is None:
            return None

        return (self.finalized_at_ns - self.started_at_ns) / 1_000_000.0


class StreamingSTTResult(OrchestrationModel):
    """
    Result from a Streaming STT operation.
    """

    success: bool
    reason: StreamingSTTReason
    session_id: str
    status: StreamingSTTStatus
    audio_chunk: AudioChunk | None = None
    partial: PartialTranscript | None = None
    intent: PartialIntent | None = None
    hints: tuple[SpeculativeWorkHint, ...] = ()
    event: StreamingSTTEvent | None = None
    state: StreamingSTTSessionState | None = None
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


class StreamingSTTReport(OrchestrationModel):
    """
    Final report for one streaming STT session.
    """

    session_id: str
    trace_id: str
    name: str
    status: StreamingSTTStatus
    audio_chunk_count: int = Field(ge=0)
    partial_count: int = Field(ge=0)
    intent_count: int = Field(ge=0)
    speculative_hint_count: int = Field(ge=0)
    confirmed_hint_count: int = Field(ge=0)
    discarded_hint_count: int = Field(ge=0)
    first_partial_latency_ms: float | None = None
    finalization_latency_ms: float | None = None
    final_transcript: str
    partials: tuple[PartialTranscript, ...]
    intents: tuple[PartialIntent, ...]
    hints: tuple[SpeculativeWorkHint, ...]
    events: tuple[StreamingSTTEvent, ...]
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


class StreamingSTTRuntimeSnapshot(OrchestrationModel):
    """
    Runtime diagnostics for Phase 7 Step 7.
    """

    name: str
    session_count: int = Field(ge=0)
    active_count: int = Field(ge=0)
    finalized_count: int = Field(ge=0)
    cancelled_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    audio_chunk_count: int = Field(ge=0)
    partial_count: int = Field(ge=0)
    intent_count: int = Field(ge=0)
    speculative_hint_count: int = Field(ge=0)
    report_count: int = Field(ge=0)
    last_reason: StreamingSTTReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        return cleaned


class StreamingSpeechToTextAdapter(Protocol):
    """
    STT adapter boundary.

    Real STT engines will implement this later. Step 7 uses a fake adapter in
    tests so the streaming contracts are deterministic.
    """

    def transcribe_partial(self, audio: AudioChunk) -> PartialTranscript:
        """
        Convert one audio chunk into a partial transcript.
        """


class LightweightPartialIntentClassifier(Protocol):
    """
    Lightweight local classifier for partial transcripts.
    """

    def classify(self, partial: PartialTranscript) -> PartialIntent:
        """
        Classify partial transcript intent.
        """


class FakeStreamingSTTAdapter:
    """
    Deterministic fake streaming STT adapter.

    It reads text from chunk metadata["transcript"] when present.
    """

    def transcribe_partial(self, audio: AudioChunk) -> PartialTranscript:
        text_value = audio.metadata.get("transcript", "")
        confidence_value = audio.metadata.get("confidence", 0.9)
        stability_value_raw = audio.metadata.get(
            "stability",
             PartialTranscriptStability.UNSTABLE.value,
        )

        text = str(text_value)

        if isinstance(confidence_value, int | float | str):
            confidence = float(confidence_value)
        else:
            confidence = 0.9

        stability_value = str(stability_value_raw)
        stability = PartialTranscriptStability(stability_value)

        if audio.kind == AudioChunkKind.END_OF_TURN:
            stability = PartialTranscriptStability.FINAL

        return PartialTranscript(
            session_id=audio.session_id,
            sequence=audio.sequence,
            text=text,
            confidence=confidence,
            stability=stability,
            source_audio_sequence=audio.sequence,
            emitted_at_ns=time.perf_counter_ns(),
        )


class KeywordPartialIntentClassifier:
    """
    Small local keyword classifier.

    This is deliberately lightweight. It gives early hints, not final cognition.
    """

    def classify(self, partial: PartialTranscript) -> PartialIntent:
        text = partial.text.lower().strip()
        kind = PartialIntentKind.CONVERSATION
        confidence = 0.55

        if not text:
            kind = PartialIntentKind.UNKNOWN
            confidence = 0.0
        elif any(word in text for word in ("debug", "error", "traceback", "fix")):
            kind = PartialIntentKind.DEBUGGING
            confidence = 0.86
        elif any(word in text for word in ("research", "latest", "find", "search")):
            kind = PartialIntentKind.RESEARCH
            confidence = 0.82
        elif any(word in text for word in ("remember", "recall", "memory")):
            kind = PartialIntentKind.MEMORY_RECALL
            confidence = 0.82
        elif any(word in text for word in ("open", "run", "create", "start", "stop")):
            kind = PartialIntentKind.COMMAND
            confidence = 0.78
        elif "?" in text or text.startswith(("what", "why", "how", "when", "where")):
            kind = PartialIntentKind.QUESTION
            confidence = 0.76
        elif any(word in text for word in ("tool", "browser", "file", "terminal")):
            kind = PartialIntentKind.TOOL_USE
            confidence = 0.76

        return PartialIntent(
            session_id=partial.session_id,
            transcript_id=partial.transcript_id,
            kind=kind,
            confidence=confidence,
            text=partial.text,
        )


@dataclass(frozen=True, slots=True)
class StreamingSTTRuntimeConfig:
    """
    Streaming STT runtime configuration.
    """

    name: str = "streaming_stt_runtime"
    audio_window_min_ms: float = 100.0
    audio_window_max_ms: float = 200.0
    speculative_confidence_threshold: float = 0.70
    speculative_discard_similarity_threshold: float = 0.60
    profile_streaming_stt: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.audio_window_min_ms <= 0:
            raise ValueError("audio_window_min_ms must be positive.")

        if self.audio_window_max_ms < self.audio_window_min_ms:
            raise ValueError(
                "audio_window_max_ms must be greater than or equal to min."
            )

        if not 0 <= self.speculative_confidence_threshold <= 1:
            raise ValueError("speculative_confidence_threshold must be within 0..1.")

        if not 0 <= self.speculative_discard_similarity_threshold <= 1:
            raise ValueError(
                "speculative_discard_similarity_threshold must be within 0..1."
            )


class StreamingSTTRuntime:
    """
    Phase 7 Step 7 Streaming STT Pipeline.

    Responsibilities:
    - accept 100-200ms audio chunks
    - emit partial transcripts while user is still speaking
    - classify partial intent locally
    - start speculative memory/context/tool hints
    - finalize transcript and reconcile speculative work
    - cancel/discard early work when final transcript diverges
    - feed latency profiler

    Non-responsibilities:
    - no direct memory retrieval execution
    - no direct tool execution
    - no final cognition decision
    - no autonomous action
    """

    def __init__(
        self,
        *,
        config: StreamingSTTRuntimeConfig | None = None,
        stt_adapter: StreamingSpeechToTextAdapter | None = None,
        intent_classifier: LightweightPartialIntentClassifier | None = None,
        profiler: PipelineLatencyProfiler | None = None,
    ) -> None:
        self._config = config or StreamingSTTRuntimeConfig()
        self._config.validate()

        self._stt_adapter = stt_adapter or FakeStreamingSTTAdapter()
        self._intent_classifier = intent_classifier or KeywordPartialIntentClassifier()
        self._profiler = profiler or PipelineLatencyProfiler(
            config=PipelineProfilerConfig(record_to_latency_runtime=True)
        )
        self._states: dict[str, StreamingSTTSessionState] = {}
        self._audio_chunks: dict[str, list[AudioChunk]] = {}
        self._partials: dict[str, list[PartialTranscript]] = {}
        self._intents: dict[str, list[PartialIntent]] = {}
        self._hints: dict[str, list[SpeculativeWorkHint]] = {}
        self._events: dict[str, list[StreamingSTTEvent]] = {}
        self._reports: list[StreamingSTTReport] = []
        self._lock = RLock()
        self._last_reason: StreamingSTTReason | None = None

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
    ) -> StreamingSTTSessionState:
        state = StreamingSTTSessionState(
            session_id=session_id or uuid4().hex,
            trace_id=trace_id or uuid4().hex,
            name=name,
            metadata=metadata or {},
        )
        event = self._event(
            session_id=state.session_id,
            kind=StreamingSTTEventKind.SESSION_CREATED,
            reason=StreamingSTTReason.SESSION_CREATED,
        )

        with self._lock:
            self._states[state.session_id] = state
            self._audio_chunks[state.session_id] = []
            self._partials[state.session_id] = []
            self._intents[state.session_id] = []
            self._hints[state.session_id] = []
            self._events[state.session_id] = [event]
            self._last_reason = StreamingSTTReason.SESSION_CREATED

        self._profiler.start_trace(name=name, trace_id=state.trace_id)

        return state

    def start_session(self, session_id: str) -> StreamingSTTResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            if state.status != StreamingSTTStatus.CREATED:
                return self._failure(
                    session_id=session_id,
                    reason=StreamingSTTReason.SESSION_NOT_ACTIVE,
                    status=state.status,
                    message="STT session cannot be started from current state",
                    state=state,
                )

            updated = state.model_copy(
                update={
                    "status": StreamingSTTStatus.ACTIVE,
                    "started_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = updated
            event = self._event(
                session_id=session_id,
                kind=StreamingSTTEventKind.SESSION_STARTED,
                reason=StreamingSTTReason.SESSION_STARTED,
            )
            self._events[session_id].append(event)
            self._last_reason = StreamingSTTReason.SESSION_STARTED

        return StreamingSTTResult(
            success=True,
            reason=StreamingSTTReason.SESSION_STARTED,
            session_id=session_id,
            status=StreamingSTTStatus.ACTIVE,
            event=event,
            state=updated,
            message="streaming STT session started",
        )

    def accept_audio_chunk(
        self,
        *,
        session_id: str,
        duration_ms: float,
        data: bytes = b"",
        kind: AudioChunkKind = AudioChunkKind.SPEECH,
        metadata: dict[str, object] | None = None,
    ) -> StreamingSTTResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            if state.status != StreamingSTTStatus.ACTIVE:
                return self._failure(
                    session_id=session_id,
                    reason=StreamingSTTReason.SESSION_NOT_ACTIVE,
                    status=state.status,
                    message="streaming STT session is not active",
                    state=state,
                )

            sequence = state.audio_chunk_count
            audio = AudioChunk(
                session_id=session_id,
                sequence=sequence,
                duration_ms=duration_ms,
                data=data,
                kind=kind,
                metadata=metadata or {},
            )
            self._audio_chunks[session_id].append(audio)

            audio_event = self._event(
                session_id=session_id,
                kind=StreamingSTTEventKind.AUDIO_ACCEPTED,
                reason=StreamingSTTReason.AUDIO_CHUNK_ACCEPTED,
                sequence=sequence,
            )
            self._events[session_id].append(audio_event)

        partial = self._stt_adapter.transcribe_partial(audio)
        intent = self._intent_classifier.classify(partial)
        hints = self._speculative_hints_for(intent)

        with self._lock:
            current = self._states[session_id]
            first_partial_at_ns = current.first_partial_at_ns or partial.emitted_at_ns
            updated = current.model_copy(
                update={
                    "first_partial_at_ns": first_partial_at_ns,
                    "audio_chunk_count": current.audio_chunk_count + 1,
                    "partial_count": current.partial_count + 1,
                    "intent_count": current.intent_count + 1,
                    "speculative_hint_count": (
                        current.speculative_hint_count + len(hints)
                    ),
                    "latest_partial_text": partial.text,
                }
            )
            self._states[session_id] = updated
            self._partials[session_id].append(partial)
            self._intents[session_id].append(intent)
            self._hints[session_id].extend(hints)

            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=StreamingSTTEventKind.PARTIAL_TRANSCRIPT_EMITTED,
                    reason=(
                        StreamingSTTReason.FIRST_PARTIAL_RECORDED
                        if current.first_partial_at_ns is None
                        else StreamingSTTReason.PARTIAL_TRANSCRIPT_READY
                    ),
                    sequence=partial.sequence,
                    text=partial.text,
                    latency_ms=updated.first_partial_latency_ms(),
                )
            )
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=StreamingSTTEventKind.PARTIAL_INTENT_DETECTED,
                    reason=(
                        StreamingSTTReason.INTENT_CONFIDENCE_HIGH
                        if intent.confidence
                        >= self._config.speculative_confidence_threshold
                        else StreamingSTTReason.INTENT_CONFIDENCE_LOW
                    ),
                    sequence=partial.sequence,
                    text=partial.text,
                    metadata={
                        "intent": intent.kind.value,
                        "confidence": intent.confidence,
                    },
                )
            )

            for hint in hints:
                self._events[session_id].append(
                    self._event(
                        session_id=session_id,
                        kind=StreamingSTTEventKind.SPECULATIVE_WORK_STARTED,
                        reason=StreamingSTTReason.SPECULATIVE_WORK_STARTED,
                        sequence=partial.sequence,
                        text=partial.text,
                        metadata={
                            "hint_id": hint.hint_id,
                            "hint_kind": hint.kind.value,
                        },
                    )
                )

            self._last_reason = StreamingSTTReason.PARTIAL_TRANSCRIPT_READY

        self._record_stt_span(
            state=updated,
            audio=audio,
            partial=partial,
            first_partial=current.first_partial_at_ns is None,
        )

        return StreamingSTTResult(
            success=True,
            reason=StreamingSTTReason.PARTIAL_TRANSCRIPT_READY,
            session_id=session_id,
            status=StreamingSTTStatus.ACTIVE,
            audio_chunk=audio,
            partial=partial,
            intent=intent,
            hints=tuple(hints),
            state=updated,
            message="audio chunk processed into partial transcript",
        )

    def finalize_transcript(
        self,
        *,
        session_id: str,
        final_text: str | None = None,
    ) -> StreamingSTTReport:
        state = self.state_for(session_id)

        if state is None:
            raise ValueError(f"streaming STT session not found: {session_id}")

        if state.status != StreamingSTTStatus.ACTIVE:
            raise ValueError("streaming STT session cannot finalize from current state")

        partials = self.partials_for(session_id)
        resolved_final = (
            final_text
            if final_text is not None
            else (partials[-1].text if partials else "")
        )
        similarity = self._similarity(state.latest_partial_text, resolved_final)
        diverged = similarity < self._config.speculative_discard_similarity_threshold

        with self._lock:
            hints = tuple(self._hints[session_id])
            reconciled_hints = tuple(
                hint.model_copy(
                    update={
                        "status": (
                            SpeculativeWorkStatus.DISCARDED
                            if diverged
                            else SpeculativeWorkStatus.CONFIRMED
                        )
                    }
                )
                for hint in hints
            )
            self._hints[session_id] = list(reconciled_hints)

            current = self._states[session_id]
            finalized = current.model_copy(
                update={
                    "status": StreamingSTTStatus.FINALIZED,
                    "finalized_at_ns": time.perf_counter_ns(),
                    "final_transcript": resolved_final,
                }
            )
            self._states[session_id] = finalized

            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=StreamingSTTEventKind.FINAL_TRANSCRIPT_READY,
                    reason=(
                        StreamingSTTReason.FINAL_TRANSCRIPT_DIVERGED
                        if diverged
                        else StreamingSTTReason.FINAL_TRANSCRIPT_READY
                    ),
                    text=resolved_final,
                    latency_ms=finalized.finalization_latency_ms(),
                    metadata={"similarity": similarity},
                )
            )

            for hint in reconciled_hints:
                self._events[session_id].append(
                    self._event(
                        session_id=session_id,
                        kind=(
                            StreamingSTTEventKind.SPECULATIVE_WORK_CANCELLED
                            if diverged
                            else StreamingSTTEventKind.SPECULATIVE_WORK_STARTED
                        ),
                        reason=(
                            StreamingSTTReason.SPECULATIVE_WORK_DISCARDED
                            if diverged
                            else StreamingSTTReason.SPECULATIVE_WORK_CONFIRMED
                        ),
                        metadata={
                            "hint_id": hint.hint_id,
                            "hint_kind": hint.kind.value,
                            "status": hint.status.value,
                        },
                    )
                )

            self._last_reason = (
                StreamingSTTReason.FINAL_TRANSCRIPT_DIVERGED
                if diverged
                else StreamingSTTReason.FINAL_TRANSCRIPT_READY
            )

        self._record_finalization_span(finalized)

        profiler_report = None

        if self._config.profile_streaming_stt:
            profiler_report = self._profiler.complete_trace(finalized.trace_id)

        report = StreamingSTTReport(
            session_id=finalized.session_id,
            trace_id=finalized.trace_id,
            name=finalized.name,
            status=finalized.status,
            audio_chunk_count=finalized.audio_chunk_count,
            partial_count=finalized.partial_count,
            intent_count=finalized.intent_count,
            speculative_hint_count=finalized.speculative_hint_count,
            confirmed_hint_count=sum(
                1
                for hint in self.hints_for(session_id)
                if hint.status == SpeculativeWorkStatus.CONFIRMED
            ),
            discarded_hint_count=sum(
                1
                for hint in self.hints_for(session_id)
                if hint.status == SpeculativeWorkStatus.DISCARDED
            ),
            first_partial_latency_ms=finalized.first_partial_latency_ms(),
            finalization_latency_ms=finalized.finalization_latency_ms(),
            final_transcript=resolved_final,
            partials=self.partials_for(session_id),
            intents=self.intents_for(session_id),
            hints=self.hints_for(session_id),
            events=self.events_for(session_id),
            profiler_report=profiler_report,
            metadata={"diverged": diverged, "similarity": similarity},
        )

        with self._lock:
            self._reports.append(report)

        return report

    def cancel_session(
        self,
        session_id: str,
        *,
        reason: str = "cancelled",
    ) -> StreamingSTTResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            if state.status in {
                StreamingSTTStatus.CANCELLED,
                StreamingSTTStatus.FINALIZED,
                StreamingSTTStatus.FAILED,
            }:
                return self._failure(
                    session_id=session_id,
                    reason=StreamingSTTReason.SESSION_NOT_ACTIVE,
                    status=state.status,
                    message="streaming STT session already terminal",
                    state=state,
                )

            cancelled_hints = [
                hint.model_copy(update={"status": SpeculativeWorkStatus.CANCELLED})
                for hint in self._hints[session_id]
            ]
            self._hints[session_id] = cancelled_hints

            updated = state.model_copy(
                update={
                    "status": StreamingSTTStatus.CANCELLED,
                    "cancelled_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = updated

            event = self._event(
                session_id=session_id,
                kind=StreamingSTTEventKind.SESSION_CANCELLED,
                reason=StreamingSTTReason.SESSION_CANCELLED,
                metadata={"cancel_reason": reason},
            )
            self._events[session_id].append(event)

            for hint in cancelled_hints:
                self._events[session_id].append(
                    self._event(
                        session_id=session_id,
                        kind=StreamingSTTEventKind.SPECULATIVE_WORK_CANCELLED,
                        reason=StreamingSTTReason.SPECULATIVE_WORK_CANCELLED,
                        metadata={
                            "hint_id": hint.hint_id,
                            "hint_kind": hint.kind.value,
                        },
                    )
                )

            self._last_reason = StreamingSTTReason.SESSION_CANCELLED

        return StreamingSTTResult(
            success=True,
            reason=StreamingSTTReason.SESSION_CANCELLED,
            session_id=session_id,
            status=StreamingSTTStatus.CANCELLED,
            event=event,
            state=updated,
            message="streaming STT session cancelled",
        )

    def fail_session(
        self,
        session_id: str,
        *,
        error: str,
    ) -> StreamingSTTResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            updated = state.model_copy(
                update={
                    "status": StreamingSTTStatus.FAILED,
                    "failed_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = updated
            event = self._event(
                session_id=session_id,
                kind=StreamingSTTEventKind.SESSION_FAILED,
                reason=StreamingSTTReason.SESSION_FAILED,
                metadata={"error": error},
            )
            self._events[session_id].append(event)
            self._last_reason = StreamingSTTReason.SESSION_FAILED

        return StreamingSTTResult(
            success=True,
            reason=StreamingSTTReason.SESSION_FAILED,
            session_id=session_id,
            status=StreamingSTTStatus.FAILED,
            event=event,
            state=updated,
            message="streaming STT session failed",
        )

    def state_for(self, session_id: str) -> StreamingSTTSessionState | None:
        with self._lock:
            return self._states.get(session_id)

    def audio_chunks_for(self, session_id: str) -> tuple[AudioChunk, ...]:
        with self._lock:
            return tuple(self._audio_chunks.get(session_id, ()))

    def partials_for(self, session_id: str) -> tuple[PartialTranscript, ...]:
        with self._lock:
            return tuple(self._partials.get(session_id, ()))

    def intents_for(self, session_id: str) -> tuple[PartialIntent, ...]:
        with self._lock:
            return tuple(self._intents.get(session_id, ()))

    def hints_for(self, session_id: str) -> tuple[SpeculativeWorkHint, ...]:
        with self._lock:
            return tuple(self._hints.get(session_id, ()))

    def events_for(self, session_id: str) -> tuple[StreamingSTTEvent, ...]:
        with self._lock:
            return tuple(self._events.get(session_id, ()))

    def reports(self) -> tuple[StreamingSTTReport, ...]:
        with self._lock:
            return tuple(self._reports)

    def latest_report(self) -> StreamingSTTReport | None:
        with self._lock:
            if not self._reports:
                return None

            return self._reports[-1]

    def snapshot(self) -> StreamingSTTRuntimeSnapshot:
        with self._lock:
            states = tuple(self._states.values())

            return StreamingSTTRuntimeSnapshot(
                name=self.name,
                session_count=len(states),
                active_count=sum(
                    1 for state in states if state.status == StreamingSTTStatus.ACTIVE
                ),
                finalized_count=sum(
                    1
                    for state in states
                    if state.status == StreamingSTTStatus.FINALIZED
                ),
                cancelled_count=sum(
                    1
                    for state in states
                    if state.status == StreamingSTTStatus.CANCELLED
                ),
                failed_count=sum(
                    1 for state in states if state.status == StreamingSTTStatus.FAILED
                ),
                audio_chunk_count=sum(
                    len(chunks) for chunks in self._audio_chunks.values()
                ),
                partial_count=sum(len(items) for items in self._partials.values()),
                intent_count=sum(len(items) for items in self._intents.values()),
                speculative_hint_count=sum(
                    len(items) for items in self._hints.values()
                ),
                report_count=len(self._reports),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        with self._lock:
            self._states.clear()
            self._audio_chunks.clear()
            self._partials.clear()
            self._intents.clear()
            self._hints.clear()
            self._events.clear()
            self._reports.clear()
            self._last_reason = StreamingSTTReason.RUNTIME_RESET

    def _speculative_hints_for(
        self,
        intent: PartialIntent,
    ) -> tuple[SpeculativeWorkHint, ...]:
        if intent.confidence < self._config.speculative_confidence_threshold:
            return ()

        hints: list[SpeculativeWorkHint] = [
            SpeculativeWorkHint(
                session_id=intent.session_id,
                intent_id=intent.intent_id,
                kind=SpeculativeWorkKind.MEMORY_PREFETCH,
                status=SpeculativeWorkStatus.STARTED,
                reason="partial transcript confidence exceeded threshold",
                confidence=intent.confidence,
                metadata={"intent": intent.kind.value},
            ),
            SpeculativeWorkHint(
                session_id=intent.session_id,
                intent_id=intent.intent_id,
                kind=SpeculativeWorkKind.CONTEXT_PREWARM,
                status=SpeculativeWorkStatus.STARTED,
                reason="partial transcript confidence exceeded threshold",
                confidence=intent.confidence,
                metadata={"intent": intent.kind.value},
            ),
        ]

        if intent.kind in {
            PartialIntentKind.COMMAND,
            PartialIntentKind.DEBUGGING,
            PartialIntentKind.RESEARCH,
            PartialIntentKind.TOOL_USE,
        }:
            hints.append(
                SpeculativeWorkHint(
                    session_id=intent.session_id,
                    intent_id=intent.intent_id,
                    kind=SpeculativeWorkKind.TOOL_PLANNER_HINT,
                    status=SpeculativeWorkStatus.STARTED,
                    reason="partial intent suggests possible tool planning",
                    confidence=intent.confidence,
                    metadata={"intent": intent.kind.value},
                )
            )

        if intent.kind in {
            PartialIntentKind.QUESTION,
            PartialIntentKind.DEBUGGING,
            PartialIntentKind.RESEARCH,
            PartialIntentKind.CONVERSATION,
        }:
            hints.append(
                SpeculativeWorkHint(
                    session_id=intent.session_id,
                    intent_id=intent.intent_id,
                    kind=SpeculativeWorkKind.LLM_CONTEXT_PREWARM,
                    status=SpeculativeWorkStatus.STARTED,
                    reason="partial intent suggests likely response generation",
                    confidence=intent.confidence,
                    metadata={"intent": intent.kind.value},
                )
            )

        return tuple(hints)

    def _record_stt_span(
        self,
        *,
        state: StreamingSTTSessionState,
        audio: AudioChunk,
        partial: PartialTranscript,
        first_partial: bool,
    ) -> None:
        stage = (
            PipelineStage.STT_FIRST_PARTIAL
            if first_partial
            else PipelineStage.STT_FINALIZATION
        )
        operation = (
            LatencyOperation.STT_FIRST_TOKEN
            if first_partial
            else LatencyOperation.STT_FINALIZATION
        )

        self._profiler.record_stage(
            trace_id=state.trace_id,
            stage=stage,
            operation=operation,
            subsystem=LatencySubsystem.PRESENCE,
            start_ns=audio.received_at_ns,
            end_ns=partial.emitted_at_ns,
            metadata={
                "session_id": state.session_id,
                "partial_sequence": partial.sequence,
            },
        )

    def _record_finalization_span(self, state: StreamingSTTSessionState) -> None:
        if state.started_at_ns is None or state.finalized_at_ns is None:
            return

        self._profiler.record_stage(
            trace_id=state.trace_id,
            stage=PipelineStage.STT_FINALIZATION,
            operation=LatencyOperation.STT_FINALIZATION,
            subsystem=LatencySubsystem.PRESENCE,
            start_ns=state.started_at_ns,
            end_ns=state.finalized_at_ns,
            metadata={"session_id": state.session_id},
        )

    @staticmethod
    def _similarity(left: str, right: str) -> float:
        left_words = set(left.lower().split())
        right_words = set(right.lower().split())

        if not left_words and not right_words:
            return 1.0

        if not left_words or not right_words:
            return 0.0

        return len(left_words & right_words) / len(left_words | right_words)

    @staticmethod
    def _event(
        *,
        session_id: str,
        kind: StreamingSTTEventKind,
        reason: StreamingSTTReason,
        sequence: int | None = None,
        text: str | None = None,
        latency_ms: float | None = None,
        metadata: dict[str, object] | None = None,
    ) -> StreamingSTTEvent:
        return StreamingSTTEvent(
            session_id=session_id,
            kind=kind,
            reason=reason,
            sequence=sequence,
            text=text,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )

    @staticmethod
    def _missing_session(session_id: str) -> StreamingSTTResult:
        return StreamingSTTResult(
            success=False,
            reason=StreamingSTTReason.SESSION_NOT_FOUND,
            session_id=session_id,
            status=StreamingSTTStatus.FAILED,
            message="streaming STT session not found",
        )

    @staticmethod
    def _failure(
        *,
        session_id: str,
        reason: StreamingSTTReason,
        status: StreamingSTTStatus,
        message: str,
        state: StreamingSTTSessionState | None = None,
    ) -> StreamingSTTResult:
        return StreamingSTTResult(
            success=False,
            reason=reason,
            session_id=session_id,
            status=status,
            state=state,
            message=message,
        )


def audio_chunk_metadata(
    *,
    transcript: str,
    confidence: float = 0.9,
    stability: PartialTranscriptStability = PartialTranscriptStability.UNSTABLE,
) -> dict[str, object]:
    """
    Helper metadata for fake streaming STT tests.
    """

    return {
        "transcript": transcript,
        "confidence": confidence,
        "stability": stability.value,
    }