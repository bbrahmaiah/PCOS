from __future__ import annotations

import re
import time
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class NaturalnessStatus(StrEnum):
    """
    Response naturalness lifecycle status.
    """

    CREATED = "created"
    OPTIMIZING = "optimizing"
    OPTIMIZED = "optimized"
    CANCELLED = "cancelled"
    FAILED = "failed"


class NaturalnessEventKind(StrEnum):
    """
    Naturalness optimizer event kind.
    """

    SESSION_CREATED = "session_created"
    OPTIMIZATION_STARTED = "optimization_started"
    SENTENCE_SPLIT = "sentence_split"
    PAUSE_INSERTED = "pause_inserted"
    FILLER_SELECTED = "filler_selected"
    PROSODY_APPLIED = "prosody_applied"
    CHUNK_CREATED = "chunk_created"
    OPTIMIZATION_COMPLETED = "optimization_completed"
    SESSION_CANCELLED = "session_cancelled"
    SESSION_FAILED = "session_failed"


class NaturalnessReason(StrEnum):
    """
    Machine-readable naturalness reasons.
    """

    SESSION_CREATED = "session_created"
    OPTIMIZATION_STARTED = "optimization_started"
    SENTENCE_SPLIT_FOR_VOICE = "sentence_split_for_voice"
    PAUSE_INSERTED_AT_SEMANTIC_BOUNDARY = (
        "pause_inserted_at_semantic_boundary"
    )
    FILLER_SELECTED = "filler_selected"
    FILLER_SUPPRESSED = "filler_suppressed"
    QUESTION_PROSODY_APPLIED = "question_prosody_applied"
    LIST_PROSODY_APPLIED = "list_prosody_applied"
    CHUNK_CREATED = "chunk_created"
    RESPONSE_OPTIMIZED = "response_optimized"
    SESSION_CANCELLED = "session_cancelled"
    SESSION_FAILED = "session_failed"
    SESSION_NOT_FOUND = "session_not_found"
    SESSION_NOT_ACTIVE = "session_not_active"
    EMPTY_RESPONSE = "empty_response"
    RUNTIME_RESET = "runtime_reset"


class SpeechChunkKind(StrEnum):
    """
    Natural speech chunk kind.
    """

    FILLER = "filler"
    SENTENCE = "sentence"
    LIST_ITEM = "list_item"
    SUMMARY = "summary"


class ProsodyHintKind(StrEnum):
    """
    Prosody hint kind.
    """

    NONE = "none"
    PAUSE = "pause"
    QUESTION_RISE = "question_rise"
    LIST_PACING = "list_pacing"
    EMPHASIS = "emphasis"


class FillerKind(StrEnum):
    """
    Thinking filler kind.
    """

    CHECKING = "checking"
    ONE_MOMENT = "one_moment"
    LET_ME_CHECK = "let_me_check"
    QUICK_LOOK = "quick_look"


class NaturalnessRequest(OrchestrationModel):
    """
    Request to optimize a generated response for spoken delivery.
    """

    request_id: str = Field(default_factory=lambda: uuid4().hex)
    raw_text: str
    voice_mode: bool = True
    allow_filler: bool = True
    user_waiting: bool = False
    created_at_ns: int = Field(default_factory=time.perf_counter_ns, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("request_id", "raw_text")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class SpeechPause(OrchestrationModel):
    """
    Natural pause inserted into speech.
    """

    pause_id: str = Field(default_factory=lambda: uuid4().hex)
    duration_ms: int = Field(ge=0)
    reason: NaturalnessReason
    created_at_ns: int = Field(default_factory=time.perf_counter_ns, ge=0)
    created_at: object = Field(default_factory=utc_now)

    @field_validator("pause_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("pause_id cannot be empty.")

        return cleaned


class NaturalSpeechChunk(OrchestrationModel):
    """
    TTS-ready spoken chunk.

    Chunks should be short, natural, and stable enough for streaming TTS.
    """

    chunk_id: str = Field(default_factory=lambda: uuid4().hex)
    kind: SpeechChunkKind
    text: str
    ssml_text: str
    word_count: int = Field(ge=0)
    pause_after_ms: int = Field(default=0, ge=0)
    prosody_hint: ProsodyHintKind = ProsodyHintKind.NONE
    tts_ready: bool = True
    created_at_ns: int = Field(default_factory=time.perf_counter_ns, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("chunk_id", "text", "ssml_text")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @model_validator(mode="after")
    def _word_count_matches_text(self) -> NaturalSpeechChunk:
        if self.word_count != len(self.text.split()):
            raise ValueError("word_count must match text word count.")

        return self


class ThinkingFiller(OrchestrationModel):
    """
    Short filler used to mask generation latency naturally.
    """

    filler_id: str = Field(default_factory=lambda: uuid4().hex)
    kind: FillerKind
    text: str
    duration_ms: int = Field(ge=0)
    ssml_text: str
    created_at_ns: int = Field(default_factory=time.perf_counter_ns, ge=0)
    created_at: object = Field(default_factory=utc_now)

    @field_validator("filler_id", "text", "ssml_text")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class NaturalnessEvent(OrchestrationModel):
    """
    Naturalness optimization event.
    """

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    kind: NaturalnessEventKind
    reason: NaturalnessReason
    chunk_id: str | None = None
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


class NaturalnessSessionState(OrchestrationModel):
    """
    Runtime state for one naturalness optimization session.
    """

    session_id: str = Field(default_factory=lambda: uuid4().hex)
    trace_id: str = Field(default_factory=lambda: uuid4().hex)
    status: NaturalnessStatus = NaturalnessStatus.CREATED
    request: NaturalnessRequest
    started_at_ns: int | None = None
    completed_at_ns: int | None = None
    cancelled_at_ns: int | None = None
    failed_at_ns: int | None = None
    chunk_count: int = Field(default=0, ge=0)
    pause_count: int = Field(default=0, ge=0)
    split_count: int = Field(default=0, ge=0)
    filler_count: int = Field(default=0, ge=0)
    prosody_count: int = Field(default=0, ge=0)
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
        if self.started_at_ns is None or self.completed_at_ns is None:
            return None

        return (self.completed_at_ns - self.started_at_ns) / 1_000_000.0


class NaturalnessResult(OrchestrationModel):
    """
    Result from one naturalness optimizer operation.
    """

    success: bool
    reason: NaturalnessReason
    session_id: str
    status: NaturalnessStatus
    chunk: NaturalSpeechChunk | None = None
    filler: ThinkingFiller | None = None
    event: NaturalnessEvent | None = None
    state: NaturalnessSessionState | None = None
    message: str
    created_at: object = Field(default_factory=utc_now)

    @field_validator("session_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class NaturalnessReport(OrchestrationModel):
    """
    Final naturalness optimization report.
    """

    session_id: str
    trace_id: str
    status: NaturalnessStatus
    raw_text: str
    optimized_text: str
    optimized_ssml: str
    chunk_count: int = Field(ge=0)
    pause_count: int = Field(ge=0)
    split_count: int = Field(ge=0)
    filler_count: int = Field(ge=0)
    prosody_count: int = Field(ge=0)
    total_latency_ms: float | None = None
    chunks: tuple[NaturalSpeechChunk, ...]
    fillers: tuple[ThinkingFiller, ...]
    events: tuple[NaturalnessEvent, ...]
    created_at: object = Field(default_factory=utc_now)

    @field_validator("session_id", "trace_id", "raw_text", "optimized_text")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class NaturalnessRuntimeSnapshot(OrchestrationModel):
    """
    Runtime diagnostics for Phase 7 Step 16.
    """

    name: str
    session_count: int = Field(ge=0)
    optimized_count: int = Field(ge=0)
    cancelled_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    report_count: int = Field(ge=0)
    last_reason: NaturalnessReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class ResponseNaturalnessConfig:
    """
    Response naturalness optimizer configuration.
    """

    name: str = "response_naturalness_optimizer"
    max_words_per_sentence: int = 15
    min_pause_ms: int = 150
    max_pause_ms: int = 300
    default_pause_ms: int = 220
    filler_min_ms: int = 800
    filler_max_ms: int = 1_200

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.max_words_per_sentence < 4:
            raise ValueError("max_words_per_sentence must be at least 4.")

        if self.min_pause_ms < 0:
            raise ValueError("min_pause_ms cannot be negative.")

        if self.max_pause_ms < self.min_pause_ms:
            raise ValueError("max_pause_ms must be >= min_pause_ms.")

        if self.default_pause_ms < self.min_pause_ms:
            raise ValueError("default_pause_ms below min_pause_ms.")

        if self.default_pause_ms > self.max_pause_ms:
            raise ValueError("default_pause_ms above max_pause_ms.")

        if self.filler_min_ms <= 0:
            raise ValueError("filler_min_ms must be positive.")

        if self.filler_max_ms < self.filler_min_ms:
            raise ValueError("filler_max_ms must be >= filler_min_ms.")


class ResponseNaturalnessOptimizerRuntime:
    """
    Phase 7 Step 16 Response Naturalness Optimizer.

    Responsibilities:
    - split long spoken sentences
    - insert natural breathing pauses
    - select non-repeating thinking fillers
    - apply SSML/prosody hints for TTS
    - produce short TTS-ready chunks
    - make fast responses sound natural

    Non-responsibilities:
    - no TTS synthesis
    - no LLM generation
    - no tool execution
    - no memory writes
    """

    _fillers: tuple[tuple[FillerKind, str], ...] = (
        (FillerKind.LET_ME_CHECK, "Let me check..."),
        (FillerKind.ONE_MOMENT, "One moment..."),
        (FillerKind.CHECKING, "Checking that now..."),
        (FillerKind.QUICK_LOOK, "I’ll take a quick look..."),
    )

    def __init__(
        self,
        *,
        config: ResponseNaturalnessConfig | None = None,
    ) -> None:
        self._config = config or ResponseNaturalnessConfig()
        self._config.validate()

        self._states: dict[str, NaturalnessSessionState] = {}
        self._chunks: dict[str, list[NaturalSpeechChunk]] = {}
        self._fillers_by_session: dict[str, list[ThinkingFiller]] = {}
        self._events: dict[str, list[NaturalnessEvent]] = {}
        self._reports: list[NaturalnessReport] = []
        self._last_filler_kind: FillerKind | None = None
        self._last_reason: NaturalnessReason | None = None
        self._lock = RLock()

    @property
    def name(self) -> str:
        return self._config.name

    def create_session(
        self,
        *,
        request: NaturalnessRequest,
        trace_id: str | None = None,
    ) -> NaturalnessSessionState:
        state = NaturalnessSessionState(
            trace_id=trace_id or uuid4().hex,
            request=request,
        )
        event = self._event(
            session_id=state.session_id,
            kind=NaturalnessEventKind.SESSION_CREATED,
            reason=NaturalnessReason.SESSION_CREATED,
        )

        with self._lock:
            self._states[state.session_id] = state
            self._chunks[state.session_id] = []
            self._fillers_by_session[state.session_id] = []
            self._events[state.session_id] = [event]
            self._last_reason = NaturalnessReason.SESSION_CREATED

        return state

    def optimize(self, session_id: str) -> NaturalnessReport:
        state = self.state_for(session_id)

        if state is None:
            raise ValueError(f"naturalness session not found: {session_id}")

        if state.status != NaturalnessStatus.CREATED:
            raise ValueError("naturalness session cannot optimize from current state")

        with self._lock:
            started = state.model_copy(
                update={
                    "status": NaturalnessStatus.OPTIMIZING,
                    "started_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = started
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=NaturalnessEventKind.OPTIMIZATION_STARTED,
                    reason=NaturalnessReason.OPTIMIZATION_STARTED,
                )
            )

        if started.request.allow_filler and started.request.user_waiting:
            self.select_filler(session_id)

        sentences = self._split_for_voice(started.request.raw_text)
        chunks = tuple(self._chunk_from_sentence(sentence) for sentence in sentences)

        with self._lock:
            self._chunks[session_id].extend(chunks)

            for chunk in chunks:
                self._events[session_id].append(
                    self._event(
                        session_id=session_id,
                        kind=NaturalnessEventKind.CHUNK_CREATED,
                        reason=NaturalnessReason.CHUNK_CREATED,
                        chunk_id=chunk.chunk_id,
                    )
                )

                if chunk.pause_after_ms > 0:
                    self._events[session_id].append(
                        self._event(
                            session_id=session_id,
                            kind=NaturalnessEventKind.PAUSE_INSERTED,
                            reason=(
                                NaturalnessReason
                                .PAUSE_INSERTED_AT_SEMANTIC_BOUNDARY
                            ),
                            chunk_id=chunk.chunk_id,
                        )
                    )

                if chunk.prosody_hint != ProsodyHintKind.NONE:
                    self._events[session_id].append(
                        self._event(
                            session_id=session_id,
                            kind=NaturalnessEventKind.PROSODY_APPLIED,
                            reason=self._reason_for_prosody(chunk.prosody_hint),
                            chunk_id=chunk.chunk_id,
                        )
                    )

            current = self._states[session_id]
            pause_count = sum(1 for chunk in chunks if chunk.pause_after_ms > 0)
            prosody_count = sum(
                1 for chunk in chunks if chunk.prosody_hint != ProsodyHintKind.NONE
            )
            split_count = max(0, len(chunks) - len(self._raw_sentence_units(
                started.request.raw_text
            )))
            completed = current.model_copy(
                update={
                    "status": NaturalnessStatus.OPTIMIZED,
                    "completed_at_ns": time.perf_counter_ns(),
                    "chunk_count": len(self._chunks[session_id]),
                    "pause_count": pause_count,
                    "split_count": split_count,
                    "prosody_count": prosody_count,
                }
            )
            self._states[session_id] = completed
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=NaturalnessEventKind.OPTIMIZATION_COMPLETED,
                    reason=NaturalnessReason.RESPONSE_OPTIMIZED,
                    latency_ms=completed.total_latency_ms(),
                )
            )
            self._last_reason = NaturalnessReason.RESPONSE_OPTIMIZED

        report = self._build_report(session_id)

        with self._lock:
            self._reports.append(report)

        return report

    def select_filler(self, session_id: str) -> NaturalnessResult:
        state = self.state_for(session_id)

        if state is None:
            return self._missing_session(session_id)

        index = 0

        if self._last_filler_kind is not None:
            previous_indexes = [
                idx
                for idx, (kind, _) in enumerate(self._fillers)
                if kind == self._last_filler_kind
            ]

            if previous_indexes:
                index = (previous_indexes[0] + 1) % len(self._fillers)

        kind, text = self._fillers[index]
        duration = self._config.filler_min_ms
        ssml = self._with_pause(
            text=text,
            pause_ms=duration,
        )
        filler = ThinkingFiller(
            kind=kind,
            text=text,
            duration_ms=duration,
            ssml_text=ssml,
        )

        with self._lock:
            self._fillers_by_session[session_id].append(filler)
            current = self._states[session_id]
            updated = current.model_copy(
                update={"filler_count": current.filler_count + 1}
            )
            self._states[session_id] = updated
            event = self._event(
                session_id=session_id,
                kind=NaturalnessEventKind.FILLER_SELECTED,
                reason=NaturalnessReason.FILLER_SELECTED,
            )
            self._events[session_id].append(event)
            self._last_filler_kind = kind
            self._last_reason = NaturalnessReason.FILLER_SELECTED

        return NaturalnessResult(
            success=True,
            reason=NaturalnessReason.FILLER_SELECTED,
            session_id=session_id,
            status=updated.status,
            filler=filler,
            event=event,
            state=updated,
            message="thinking filler selected",
        )

    def cancel_session(self, session_id: str) -> NaturalnessResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            cancelled = state.model_copy(
                update={
                    "status": NaturalnessStatus.CANCELLED,
                    "cancelled_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = cancelled
            event = self._event(
                session_id=session_id,
                kind=NaturalnessEventKind.SESSION_CANCELLED,
                reason=NaturalnessReason.SESSION_CANCELLED,
            )
            self._events[session_id].append(event)
            self._last_reason = NaturalnessReason.SESSION_CANCELLED

        return NaturalnessResult(
            success=True,
            reason=NaturalnessReason.SESSION_CANCELLED,
            session_id=session_id,
            status=NaturalnessStatus.CANCELLED,
            event=event,
            state=cancelled,
            message="naturalness session cancelled",
        )

    def fail_session(
        self,
        session_id: str,
        *,
        error: str,
    ) -> NaturalnessResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            failed = state.model_copy(
                update={
                    "status": NaturalnessStatus.FAILED,
                    "failed_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = failed
            event = self._event(
                session_id=session_id,
                kind=NaturalnessEventKind.SESSION_FAILED,
                reason=NaturalnessReason.SESSION_FAILED,
                metadata={"error": error},
            )
            self._events[session_id].append(event)
            self._last_reason = NaturalnessReason.SESSION_FAILED

        return NaturalnessResult(
            success=True,
            reason=NaturalnessReason.SESSION_FAILED,
            session_id=session_id,
            status=NaturalnessStatus.FAILED,
            event=event,
            state=failed,
            message="naturalness session failed",
        )

    def state_for(self, session_id: str) -> NaturalnessSessionState | None:
        with self._lock:
            return self._states.get(session_id)

    def chunks_for(self, session_id: str) -> tuple[NaturalSpeechChunk, ...]:
        with self._lock:
            return tuple(self._chunks.get(session_id, ()))

    def fillers_for(self, session_id: str) -> tuple[ThinkingFiller, ...]:
        with self._lock:
            return tuple(self._fillers_by_session.get(session_id, ()))

    def events_for(self, session_id: str) -> tuple[NaturalnessEvent, ...]:
        with self._lock:
            return tuple(self._events.get(session_id, ()))

    def reports(self) -> tuple[NaturalnessReport, ...]:
        with self._lock:
            return tuple(self._reports)

    def latest_report(self) -> NaturalnessReport | None:
        with self._lock:
            if not self._reports:
                return None

            return self._reports[-1]

    def snapshot(self) -> NaturalnessRuntimeSnapshot:
        with self._lock:
            states = tuple(self._states.values())

            return NaturalnessRuntimeSnapshot(
                name=self.name,
                session_count=len(states),
                optimized_count=sum(
                    1
                    for state in states
                    if state.status == NaturalnessStatus.OPTIMIZED
                ),
                cancelled_count=sum(
                    1
                    for state in states
                    if state.status == NaturalnessStatus.CANCELLED
                ),
                failed_count=sum(
                    1
                    for state in states
                    if state.status == NaturalnessStatus.FAILED
                ),
                report_count=len(self._reports),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        with self._lock:
            self._states.clear()
            self._chunks.clear()
            self._fillers_by_session.clear()
            self._events.clear()
            self._reports.clear()
            self._last_reason = NaturalnessReason.RUNTIME_RESET
            self._last_filler_kind = None

    def _build_report(self, session_id: str) -> NaturalnessReport:
        state = self._states[session_id]
        chunks = self.chunks_for(session_id)
        fillers = self.fillers_for(session_id)
        optimized_text = " ".join(chunk.text for chunk in chunks)
        optimized_ssml_parts = [filler.ssml_text for filler in fillers]
        optimized_ssml_parts.extend(chunk.ssml_text for chunk in chunks)
        optimized_ssml = " ".join(optimized_ssml_parts)

        return NaturalnessReport(
            session_id=session_id,
            trace_id=state.trace_id,
            status=state.status,
            raw_text=state.request.raw_text,
            optimized_text=optimized_text,
            optimized_ssml=optimized_ssml,
            chunk_count=state.chunk_count,
            pause_count=state.pause_count,
            split_count=state.split_count,
            filler_count=state.filler_count,
            prosody_count=state.prosody_count,
            total_latency_ms=state.total_latency_ms(),
            chunks=chunks,
            fillers=fillers,
            events=self.events_for(session_id),
        )

    def _split_for_voice(self, text: str) -> tuple[str, ...]:
        raw_units = self._raw_sentence_units(text)
        output: list[str] = []

        for unit in raw_units:
            output.extend(self._split_long_sentence(unit))

        return tuple(output)

    @staticmethod
    def _raw_sentence_units(text: str) -> tuple[str, ...]:
        normalized = re.sub(r"\s+", " ", text.strip())

        if ResponseNaturalnessOptimizerRuntime._looks_like_list_item(normalized):
            return (normalized,)

        protected = re.sub(
            r"\b(\d+)\.\s+",
            r"__LIST_MARKER_\1__ ",
            normalized,
        )
        parts = re.split(r"(?<=[.!?])\s+", protected)
        restored = tuple(
            re.sub(r"__LIST_MARKER_(\d+)__\s+", r"\1. ", part).strip()
            for part in parts
            if part.strip()
        )

        return restored

    def _split_long_sentence(self, sentence: str) -> tuple[str, ...]:
        words = sentence.split()

        if len(words) <= self._config.max_words_per_sentence:
            return (sentence,)

        chunks: list[str] = []
        current: list[str] = []

        for word in words:
            current.append(word)
            should_split = len(current) >= self._config.max_words_per_sentence
            boundary = self._is_soft_boundary(word)

            if should_split or boundary:
                chunks.append(" ".join(current).strip())
                current = []

        if current:
            chunks.append(" ".join(current).strip())

        return tuple(chunks)

    @staticmethod
    def _is_soft_boundary(word: str) -> bool:
        return word.endswith((",", ";", ":", "—"))

    def _chunk_from_sentence(self, sentence: str) -> NaturalSpeechChunk:
        text = sentence.strip()
        word_count = len(text.split())
        pause_ms = self._pause_for_text(text)
        prosody = self._prosody_for_text(text)
        ssml = self._ssml_for_text(
            text=text,
            pause_ms=pause_ms,
            prosody=prosody,
        )
        kind = (
            SpeechChunkKind.LIST_ITEM
            if self._looks_like_list_item(text)
            else SpeechChunkKind.SENTENCE
        )

        return NaturalSpeechChunk(
            kind=kind,
            text=text,
            ssml_text=ssml,
            word_count=word_count,
            pause_after_ms=pause_ms,
            prosody_hint=prosody,
        )

    def _pause_for_text(self, text: str) -> int:
        if "—" in text or ":" in text or ";" in text:
            return self._config.default_pause_ms

        if self._looks_like_list_item(text):
            return self._config.min_pause_ms

        if text.endswith((".", "?", "!")):
            return self._config.min_pause_ms

        return 0

    @staticmethod
    def _prosody_for_text(text: str) -> ProsodyHintKind:
        if text.endswith("?"):
            return ProsodyHintKind.QUESTION_RISE

        if ResponseNaturalnessOptimizerRuntime._looks_like_list_item(text):
            return ProsodyHintKind.LIST_PACING

        if "important" in text.lower() or "critical" in text.lower():
            return ProsodyHintKind.EMPHASIS

        if "—" in text:
            return ProsodyHintKind.PAUSE

        return ProsodyHintKind.NONE

    @staticmethod
    def _looks_like_list_item(text: str) -> bool:
        return bool(re.match(r"^(\d+\.|-|\*)\s+", text.strip()))

    def _ssml_for_text(
        self,
        *,
        text: str,
        pause_ms: int,
        prosody: ProsodyHintKind,
    ) -> str:
        escaped = self._escape_ssml(text)

        if prosody == ProsodyHintKind.QUESTION_RISE:
            escaped = f'<prosody pitch="+8%">{escaped}</prosody>'
        elif prosody == ProsodyHintKind.EMPHASIS:
            escaped = f"<emphasis level=\"moderate\">{escaped}</emphasis>"

        if pause_ms > 0:
            return self._with_pause(text=escaped, pause_ms=pause_ms)

        return escaped

    @staticmethod
    def _with_pause(*, text: str, pause_ms: int) -> str:
        return f'{text} <break time="{pause_ms}ms"/>'

    @staticmethod
    def _escape_ssml(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    @staticmethod
    def _reason_for_prosody(
        prosody: ProsodyHintKind,
    ) -> NaturalnessReason:
        if prosody == ProsodyHintKind.QUESTION_RISE:
            return NaturalnessReason.QUESTION_PROSODY_APPLIED

        if prosody == ProsodyHintKind.LIST_PACING:
            return NaturalnessReason.LIST_PROSODY_APPLIED

        return NaturalnessReason.PAUSE_INSERTED_AT_SEMANTIC_BOUNDARY

    @staticmethod
    def _event(
        *,
        session_id: str,
        kind: NaturalnessEventKind,
        reason: NaturalnessReason,
        chunk_id: str | None = None,
        latency_ms: float | None = None,
        metadata: dict[str, object] | None = None,
    ) -> NaturalnessEvent:
        return NaturalnessEvent(
            session_id=session_id,
            kind=kind,
            reason=reason,
            chunk_id=chunk_id,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )

    @staticmethod
    def _missing_session(session_id: str) -> NaturalnessResult:
        return NaturalnessResult(
            success=False,
            reason=NaturalnessReason.SESSION_NOT_FOUND,
            session_id=session_id,
            status=NaturalnessStatus.FAILED,
            message="naturalness session not found",
        )