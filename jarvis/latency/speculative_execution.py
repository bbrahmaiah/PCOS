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
from jarvis.latency.streaming_memory import MemoryRetrievalQuery
from jarvis.latency.streaming_stt import PartialIntentKind
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class SpeculationTrigger(StrEnum):
    """
    Runtime trigger for speculative work.
    """

    VAD_SPEECH_ENDING = "vad_speech_ending"
    USER_PAUSE = "user_pause"
    ACTION_COMPLETED = "action_completed"
    PARTIAL_TRANSCRIPT = "partial_transcript"
    CONVERSATION_CONTINUATION = "conversation_continuation"


class SpeculationStatus(StrEnum):
    """
    Speculative branch lifecycle status.
    """

    PROPOSED = "proposed"
    PREWARMING = "prewarming"
    READY = "ready"
    CONFIRMED = "confirmed"
    DISCARDED = "discarded"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


class SpeculativeWorkType(StrEnum):
    """
    Types of speculative work.

    These are preparation only. None may execute real-world actions.
    """

    MEMORY_PREFETCH = "memory_prefetch"
    LLM_CONTEXT_PREWARM = "llm_context_prewarm"
    TOOL_PLANNER_HINT = "tool_planner_hint"
    ACTION_PREVALIDATION = "action_prevalidation"


class SpeculationAggressiveness(StrEnum):
    """
    Adaptive speculation aggressiveness level.

    Accuracy below 40% reduces aggressiveness.
    Accuracy above 70% increases lookahead depth.
    """

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class SpeculativeExecutionEventKind(StrEnum):
    """
    Speculative execution event kind.
    """

    SESSION_CREATED = "session_created"
    SESSION_STARTED = "session_started"
    INTENT_PREDICTED = "intent_predicted"
    BRANCH_CREATED = "branch_created"
    WORK_PREWARMED = "work_prewarmed"
    BRANCH_READY = "branch_ready"
    BRANCH_CONFIRMED = "branch_confirmed"
    BRANCH_DISCARDED = "branch_discarded"
    BRANCH_CANCELLED = "branch_cancelled"
    ACCURACY_UPDATED = "accuracy_updated"
    AGGRESSIVENESS_UPDATED = "aggressiveness_updated"
    SESSION_COMPLETED = "session_completed"
    SESSION_CANCELLED = "session_cancelled"
    SESSION_FAILED = "session_failed"


class SpeculativeExecutionReason(StrEnum):
    """
    Machine-readable speculative execution reasons.
    """

    SESSION_CREATED = "session_created"
    SESSION_STARTED = "session_started"
    PREDICTION_READY = "prediction_ready"
    CANDIDATE_ACCEPTED = "candidate_accepted"
    CANDIDATE_REJECTED_LOW_PROBABILITY = "candidate_rejected_low_probability"
    BRANCH_CREATED = "branch_created"
    MEMORY_PREFETCH_PREPARED = "memory_prefetch_prepared"
    LLM_CONTEXT_PREWARMED = "llm_context_prewarmed"
    TOOL_PLANNER_HINT_PREPARED = "tool_planner_hint_prepared"
    ACTION_PREVALIDATED = "action_prevalidated"
    BRANCH_READY = "branch_ready"
    BRANCH_CONFIRMED = "branch_confirmed"
    BRANCH_DISCARDED = "branch_discarded"
    BRANCH_CANCELLED = "branch_cancelled"
    ACCURACY_LOW_REDUCED_AGGRESSIVENESS = "accuracy_low_reduced_aggressiveness"
    ACCURACY_HIGH_INCREASED_AGGRESSIVENESS = "accuracy_high_increased_aggressiveness"
    ACCURACY_NORMAL_MAINTAINED = "accuracy_normal_maintained"
    SESSION_COMPLETED = "session_completed"
    SESSION_CANCELLED = "session_cancelled"
    SESSION_FAILED = "session_failed"
    SESSION_NOT_FOUND = "session_not_found"
    SESSION_NOT_ACTIVE = "session_not_active"
    RUNTIME_RESET = "runtime_reset"


class SpeculativeCandidate(OrchestrationModel):
    """
    One predicted likely user intent.
    """

    candidate_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    intent: PartialIntentKind
    probability: float = Field(ge=0, le=1)
    rank: int = Field(ge=1)
    prompt_hint: str
    created_at_ns: int = Field(default_factory=time.perf_counter_ns, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("candidate_id", "session_id", "prompt_hint")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class SpeculativeWorkItem(OrchestrationModel):
    """
    One speculative preparation item.

    Work items are safe by contract:
    - no real action execution
    - cancellable
    - discardable
    - observable
    """

    work_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    branch_id: str
    candidate_id: str
    work_type: SpeculativeWorkType
    status: SpeculationStatus = SpeculationStatus.PROPOSED
    description: str
    cancellable: bool = True
    discardable: bool = True
    action_execution_allowed: bool = False
    created_at_ns: int = Field(default_factory=time.perf_counter_ns, ge=0)
    completed_at_ns: int | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator(
        "work_id",
         "session_id",
         "branch_id",
         "candidate_id",
         "description",
    )
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @model_validator(mode="after")
    def _validate_safety_contract(self) -> SpeculativeWorkItem:
        if not self.cancellable:
            raise ValueError("speculative work must be cancellable.")

        if not self.discardable:
            raise ValueError("speculative work must be discardable.")

        if self.action_execution_allowed:
            raise ValueError("speculative work must never execute actions.")

        return self


class SpeculativeBranch(OrchestrationModel):
    """
    A speculative branch for one likely intent.
    """

    branch_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    candidate: SpeculativeCandidate
    status: SpeculationStatus = SpeculationStatus.PROPOSED
    work_items: tuple[SpeculativeWorkItem, ...] = ()
    created_at_ns: int = Field(default_factory=time.perf_counter_ns, ge=0)
    ready_at_ns: int | None = None
    terminal_at_ns: int | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("branch_id", "session_id")
    @classmethod
    def _required_ids(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    def ready_latency_ms(self) -> float | None:
        if self.ready_at_ns is None:
            return None

        return (self.ready_at_ns - self.created_at_ns) / 1_000_000.0


class SpeculativeExecutionEvent(OrchestrationModel):
    """
    Event emitted by the speculative execution runtime.
    """

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    kind: SpeculativeExecutionEventKind
    reason: SpeculativeExecutionReason
    branch_id: str | None = None
    candidate_id: str | None = None
    work_id: str | None = None
    latency_ms: float | None = None
    probability: float | None = None
    accuracy: float | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("event_id", "session_id")
    @classmethod
    def _required_ids(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class SpeculativeSessionState(OrchestrationModel):
    """
    Runtime state for one speculative execution session.
    """

    session_id: str = Field(default_factory=lambda: uuid4().hex)
    trace_id: str = Field(default_factory=lambda: uuid4().hex)
    trigger: SpeculationTrigger
    source_text: str
    status: SpeculationStatus = SpeculationStatus.PROPOSED
    aggressiveness: SpeculationAggressiveness = SpeculationAggressiveness.NORMAL
    started_at_ns: int | None = None
    completed_at_ns: int | None = None
    cancelled_at_ns: int | None = None
    failed_at_ns: int | None = None
    candidate_count: int = Field(default=0, ge=0)
    branch_count: int = Field(default=0, ge=0)
    confirmed_count: int = Field(default=0, ge=0)
    discarded_count: int = Field(default=0, ge=0)
    cancelled_count: int = Field(default=0, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("session_id", "trace_id", "source_text")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    def total_latency_ms(self) -> float | None:
        if self.started_at_ns is None or self.completed_at_ns is None:
            return None

        return (self.completed_at_ns - self.started_at_ns) / 1_000_000.0


class SpeculationAccuracySnapshot(OrchestrationModel):
    """
    Rolling speculation accuracy diagnostics.
    """

    total_predictions: int = Field(ge=0)
    confirmed_predictions: int = Field(ge=0)
    discarded_predictions: int = Field(ge=0)
    accuracy: float = Field(ge=0, le=1)
    aggressiveness: SpeculationAggressiveness
    lookahead_depth: int = Field(ge=1)
    created_at: object = Field(default_factory=utc_now)


class SpeculativeExecutionResult(OrchestrationModel):
    """
    Result from speculative execution runtime operation.
    """

    success: bool
    reason: SpeculativeExecutionReason
    session_id: str
    status: SpeculationStatus
    state: SpeculativeSessionState | None = None
    branch: SpeculativeBranch | None = None
    event: SpeculativeExecutionEvent | None = None
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


class SpeculativeExecutionReport(OrchestrationModel):
    """
    Final report for one speculative execution session.
    """

    session_id: str
    trace_id: str
    trigger: SpeculationTrigger
    status: SpeculationStatus
    aggressiveness: SpeculationAggressiveness
    source_text: str
    candidate_count: int = Field(ge=0)
    branch_count: int = Field(ge=0)
    confirmed_count: int = Field(ge=0)
    discarded_count: int = Field(ge=0)
    cancelled_count: int = Field(ge=0)
    total_latency_ms: float | None = None
    candidates: tuple[SpeculativeCandidate, ...]
    branches: tuple[SpeculativeBranch, ...]
    events: tuple[SpeculativeExecutionEvent, ...]
    accuracy_snapshot: SpeculationAccuracySnapshot
    profiler_report: PipelineProfilerReport | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("session_id", "trace_id", "source_text")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class SpeculativeExecutionRuntimeSnapshot(OrchestrationModel):
    """
    Runtime diagnostics for Phase 7 Step 9.
    """

    name: str
    session_count: int = Field(ge=0)
    active_count: int = Field(ge=0)
    completed_count: int = Field(ge=0)
    cancelled_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    branch_count: int = Field(ge=0)
    report_count: int = Field(ge=0)
    accuracy: float = Field(ge=0, le=1)
    aggressiveness: SpeculationAggressiveness
    lookahead_depth: int = Field(ge=1)
    last_reason: SpeculativeExecutionReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class SpeculativeExecutionRuntimeConfig:
    """
    Speculative execution runtime configuration.
    """

    name: str = "speculative_execution_runtime"
    candidate_probability_threshold: float = 0.30
    max_candidates_normal: int = 2
    max_candidates_low: int = 1
    max_candidates_high: int = 3
    low_accuracy_threshold: float = 0.40
    high_accuracy_threshold: float = 0.70
    profile_speculation: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if not 0 <= self.candidate_probability_threshold <= 1:
            raise ValueError("candidate_probability_threshold must be within 0..1.")

        if self.max_candidates_normal < 1:
            raise ValueError("max_candidates_normal must be positive.")

        if self.max_candidates_low < 1:
            raise ValueError("max_candidates_low must be positive.")

        if self.max_candidates_high < self.max_candidates_normal:
            raise ValueError("max_candidates_high must be >= normal.")

        if not 0 <= self.low_accuracy_threshold <= 1:
            raise ValueError("low_accuracy_threshold must be within 0..1.")

        if not 0 <= self.high_accuracy_threshold <= 1:
            raise ValueError("high_accuracy_threshold must be within 0..1.")


class LightweightIntentPredictor:
    """
    Lightweight heuristic intent predictor.

    This is intentionally local and cheap. It predicts likely next intent; it
    does not perform final cognition.
    """

    def predict(
        self,
        *,
        trigger: SpeculationTrigger,
        source_text: str,
        session_id: str,
    ) -> tuple[SpeculativeCandidate, ...]:
        lowered = source_text.lower()
        candidates: list[tuple[PartialIntentKind, float, str]] = []

        if trigger == SpeculationTrigger.ACTION_COMPLETED:
            candidates.extend(
                (
                    (PartialIntentKind.QUESTION, 0.45, "likely follow-up question"),
                    (PartialIntentKind.COMMAND, 0.35, "likely next action command"),
                    (PartialIntentKind.CONVERSATION, 0.20, "acknowledgement"),
                )
            )
        elif any(word in lowered for word in ("debug", "error", "test", "failed")):
            candidates.extend(
                (
                    (PartialIntentKind.DEBUGGING, 0.62, "continue debugging context"),
                    (PartialIntentKind.TOOL_USE, 0.36, "likely tool/log inspection"),
                    (PartialIntentKind.QUESTION, 0.22, "possible explanation request"),
                )
            )
        elif any(word in lowered for word in ("research", "latest", "find")):
            candidates.extend(
                (
                    (PartialIntentKind.RESEARCH, 0.64, "continue research topic"),
                    (PartialIntentKind.MEMORY_RECALL, 0.33, "retrieve related context"),
                    (PartialIntentKind.QUESTION, 0.28, "answer follow-up question"),
                )
            )
        elif any(word in lowered for word in ("remember", "memory", "recall")):
            candidates.extend(
                (
                    (PartialIntentKind.MEMORY_RECALL, 0.60, "memory follow-up"),
                    (PartialIntentKind.QUESTION, 0.32, "memory explanation"),
                    (PartialIntentKind.CONVERSATION, 0.20, "conversation continuation"),
                )
            )
        else:
            candidates.extend(
                (
                    (PartialIntentKind.QUESTION, 0.42, "likely question continuation"),
                    (PartialIntentKind.CONVERSATION, 0.35, "conversation continuation"),
                    (PartialIntentKind.MEMORY_RECALL, 0.18, "possible memory context"),
                )
            )

        return tuple(
            SpeculativeCandidate(
                session_id=session_id,
                intent=intent,
                probability=probability,
                rank=index + 1,
                prompt_hint=hint,
            )
            for index, (intent, probability, hint) in enumerate(candidates)
        )


class SpeculativeExecutionRuntime:
    """
    Phase 7 Step 9 Speculative Execution Runtime.

    Responsibilities:
    - predict likely next user intent
    - select top candidates above threshold
    - create speculative branches
    - prewarm memory/context/tool/action validation hints
    - confirm matching branches
    - discard/cancel incorrect branches cleanly
    - track accuracy and adjust aggressiveness

    Non-responsibilities:
    - no real action execution
    - no direct tool execution
    - no direct LLM generation
    - no memory mutation
    """

    def __init__(
        self,
        *,
        config: SpeculativeExecutionRuntimeConfig | None = None,
        predictor: LightweightIntentPredictor | None = None,
        profiler: PipelineLatencyProfiler | None = None,
    ) -> None:
        self._config = config or SpeculativeExecutionRuntimeConfig()
        self._config.validate()

        self._predictor = predictor or LightweightIntentPredictor()
        self._profiler = profiler or PipelineLatencyProfiler(
            config=PipelineProfilerConfig(record_to_latency_runtime=True)
        )
        self._states: dict[str, SpeculativeSessionState] = {}
        self._candidates: dict[str, list[SpeculativeCandidate]] = {}
        self._branches: dict[str, list[SpeculativeBranch]] = {}
        self._events: dict[str, list[SpeculativeExecutionEvent]] = {}
        self._reports: list[SpeculativeExecutionReport] = []
        self._lock = RLock()
        self._last_reason: SpeculativeExecutionReason | None = None
        self._total_predictions = 0
        self._confirmed_predictions = 0
        self._discarded_predictions = 0
        self._aggressiveness = SpeculationAggressiveness.NORMAL

    @property
    def name(self) -> str:
        return self._config.name

    def create_session(
        self,
        *,
        trigger: SpeculationTrigger,
        source_text: str,
        trace_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> SpeculativeSessionState:
        state = SpeculativeSessionState(
            trace_id=trace_id or uuid4().hex,
            trigger=trigger,
            source_text=source_text,
            aggressiveness=self._aggressiveness,
            metadata=metadata or {},
        )
        event = self._event(
            session_id=state.session_id,
            kind=SpeculativeExecutionEventKind.SESSION_CREATED,
            reason=SpeculativeExecutionReason.SESSION_CREATED,
        )

        with self._lock:
            self._states[state.session_id] = state
            self._candidates[state.session_id] = []
            self._branches[state.session_id] = []
            self._events[state.session_id] = [event]
            self._last_reason = SpeculativeExecutionReason.SESSION_CREATED

        self._profiler.start_trace(
            name="speculative_execution",
            trace_id=state.trace_id,
        )

        return state

    def start_session(self, session_id: str) -> SpeculativeExecutionResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            if state.status != SpeculationStatus.PROPOSED:
                return self._failure(
                    session_id=session_id,
                    reason=SpeculativeExecutionReason.SESSION_NOT_ACTIVE,
                    status=state.status,
                    message="speculation session cannot start from current state",
                    state=state,
                )

            started = state.model_copy(
                update={
                    "status": SpeculationStatus.PREWARMING,
                    "started_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = started
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=SpeculativeExecutionEventKind.SESSION_STARTED,
                    reason=SpeculativeExecutionReason.SESSION_STARTED,
                )
            )
            self._last_reason = SpeculativeExecutionReason.SESSION_STARTED

        return SpeculativeExecutionResult(
            success=True,
            reason=SpeculativeExecutionReason.SESSION_STARTED,
            session_id=session_id,
            status=SpeculationStatus.PREWARMING,
            state=started,
            message="speculative execution session started",
        )

    def predict_and_prewarm(
        self,
        session_id: str,
    ) -> tuple[SpeculativeExecutionResult, ...]:
        state = self.state_for(session_id)

        if state is None:
            return (self._missing_session(session_id),)

        if state.status != SpeculationStatus.PREWARMING:
            return (
                self._failure(
                    session_id=session_id,
                    reason=SpeculativeExecutionReason.SESSION_NOT_ACTIVE,
                    status=state.status,
                    message="speculation session is not prewarming",
                    state=state,
                ),
            )

        candidates = self._select_candidates(
            self._predictor.predict(
                trigger=state.trigger,
                source_text=state.source_text,
                session_id=session_id,
            )
        )
        results: list[SpeculativeExecutionResult] = []

        with self._lock:
            self._candidates[session_id].extend(candidates)
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=SpeculativeExecutionEventKind.INTENT_PREDICTED,
                    reason=SpeculativeExecutionReason.PREDICTION_READY,
                    metadata={"candidate_count": len(candidates)},
                )
            )

        for candidate in candidates:
            branch = self._create_branch(session_id=session_id, candidate=candidate)
            ready_branch = self._prewarm_branch(branch)

            with self._lock:
                self._branches[session_id].append(ready_branch)

            results.append(
                SpeculativeExecutionResult(
                    success=True,
                    reason=SpeculativeExecutionReason.BRANCH_READY,
                    session_id=session_id,
                    status=ready_branch.status,
                    branch=ready_branch,
                    state=self.state_for(session_id),
                    message="speculative branch ready",
                )
            )

        with self._lock:
            current = self._states[session_id]
            updated = current.model_copy(
                update={
                    "candidate_count": len(self._candidates[session_id]),
                    "branch_count": len(self._branches[session_id]),
                }
            )
            self._states[session_id] = updated
            self._last_reason = SpeculativeExecutionReason.BRANCH_READY

        return tuple(results)

    def confirm(
        self,
        *,
        session_id: str,
        actual_intent: PartialIntentKind,
    ) -> SpeculativeExecutionResult:
        state = self.state_for(session_id)

        if state is None:
            return self._missing_session(session_id)

        branches = list(self.branches_for(session_id))
        matched: SpeculativeBranch | None = None

        for branch in branches:
            if branch.candidate.intent == actual_intent:
                matched = branch
                break

        with self._lock:
            updated_branches: list[SpeculativeBranch] = []
            confirmed_count = 0
            discarded_count = 0

            for branch in branches:
                if matched is not None and branch.branch_id == matched.branch_id:
                    updated = branch.model_copy(
                        update={
                            "status": SpeculationStatus.CONFIRMED,
                            "terminal_at_ns": time.perf_counter_ns(),
                        }
                    )
                    confirmed_count += 1
                    self._events[session_id].append(
                        self._event(
                            session_id=session_id,
                            kind=SpeculativeExecutionEventKind.BRANCH_CONFIRMED,
                            reason=SpeculativeExecutionReason.BRANCH_CONFIRMED,
                            branch_id=branch.branch_id,
                            candidate_id=branch.candidate.candidate_id,
                            probability=branch.candidate.probability,
                        )
                    )
                else:
                    updated = branch.model_copy(
                        update={
                            "status": SpeculationStatus.DISCARDED,
                            "terminal_at_ns": time.perf_counter_ns(),
                        }
                    )
                    discarded_count += 1
                    self._events[session_id].append(
                        self._event(
                            session_id=session_id,
                            kind=SpeculativeExecutionEventKind.BRANCH_DISCARDED,
                            reason=SpeculativeExecutionReason.BRANCH_DISCARDED,
                            branch_id=branch.branch_id,
                            candidate_id=branch.candidate.candidate_id,
                            probability=branch.candidate.probability,
                        )
                    )

                updated_branches.append(updated)

            self._branches[session_id] = updated_branches
            self._total_predictions += 1

            if confirmed_count > 0:
                self._confirmed_predictions += 1
            else:
                self._discarded_predictions += 1

            current = self._states[session_id]
            updated_state = current.model_copy(
                update={
                    "confirmed_count": current.confirmed_count + confirmed_count,
                    "discarded_count": current.discarded_count + discarded_count,
                }
            )
            self._states[session_id] = updated_state
            self._update_aggressiveness_locked(session_id)
            self._last_reason = (
                SpeculativeExecutionReason.BRANCH_CONFIRMED
                if confirmed_count > 0
                else SpeculativeExecutionReason.BRANCH_DISCARDED
            )

        return SpeculativeExecutionResult(
            success=confirmed_count > 0,
            reason=(
                SpeculativeExecutionReason.BRANCH_CONFIRMED
                if confirmed_count > 0
                else SpeculativeExecutionReason.BRANCH_DISCARDED
            ),
            session_id=session_id,
            status=(
                SpeculationStatus.CONFIRMED
                if confirmed_count > 0
                else SpeculationStatus.DISCARDED
            ),
            branch=matched,
            state=self.state_for(session_id),
            message="speculative branch reconciliation complete",
        )

    def cancel_session(
        self,
        session_id: str,
        *,
        reason: str = "cancelled",
    ) -> SpeculativeExecutionResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            if state.status in {
                SpeculationStatus.CANCELLED,
                SpeculationStatus.COMPLETED,
                SpeculationStatus.FAILED,
            }:
                return self._failure(
                    session_id=session_id,
                    reason=SpeculativeExecutionReason.SESSION_NOT_ACTIVE,
                    status=state.status,
                    message="speculation session already terminal",
                    state=state,
                )

            cancelled_branches = tuple(
                branch.model_copy(
                    update={
                        "status": SpeculationStatus.CANCELLED,
                        "terminal_at_ns": time.perf_counter_ns(),
                    }
                )
                for branch in self._branches[session_id]
            )
            self._branches[session_id] = list(cancelled_branches)

            cancelled = state.model_copy(
                update={
                    "status": SpeculationStatus.CANCELLED,
                    "cancelled_at_ns": time.perf_counter_ns(),
                    "cancelled_count": state.cancelled_count
                    + len(cancelled_branches),
                }
            )
            self._states[session_id] = cancelled
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=SpeculativeExecutionEventKind.SESSION_CANCELLED,
                    reason=SpeculativeExecutionReason.SESSION_CANCELLED,
                    metadata={"cancel_reason": reason},
                )
            )
            self._last_reason = SpeculativeExecutionReason.SESSION_CANCELLED

        return SpeculativeExecutionResult(
            success=True,
            reason=SpeculativeExecutionReason.SESSION_CANCELLED,
            session_id=session_id,
            status=SpeculationStatus.CANCELLED,
            state=cancelled,
            message="speculative execution session cancelled",
        )

    def complete_session(self, session_id: str) -> SpeculativeExecutionReport:
        state = self.state_for(session_id)

        if state is None:
            raise ValueError(f"speculation session not found: {session_id}")

        if state.status in {
            SpeculationStatus.CANCELLED,
            SpeculationStatus.FAILED,
            SpeculationStatus.COMPLETED,
        }:
            raise ValueError("speculation session cannot complete from current state")

        with self._lock:
            current = self._states[session_id]
            completed = current.model_copy(
                update={
                    "status": SpeculationStatus.COMPLETED,
                    "completed_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = completed
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=SpeculativeExecutionEventKind.SESSION_COMPLETED,
                    reason=SpeculativeExecutionReason.SESSION_COMPLETED,
                )
            )
            self._last_reason = SpeculativeExecutionReason.SESSION_COMPLETED

        self._record_speculation_span(completed)

        profiler_report = None

        if self._config.profile_speculation:
            profiler_report = self._profiler.complete_trace(completed.trace_id)

        report = SpeculativeExecutionReport(
            session_id=completed.session_id,
            trace_id=completed.trace_id,
            trigger=completed.trigger,
            status=completed.status,
            aggressiveness=completed.aggressiveness,
            source_text=completed.source_text,
            candidate_count=completed.candidate_count,
            branch_count=completed.branch_count,
            confirmed_count=completed.confirmed_count,
            discarded_count=completed.discarded_count,
            cancelled_count=completed.cancelled_count,
            total_latency_ms=completed.total_latency_ms(),
            candidates=self.candidates_for(session_id),
            branches=self.branches_for(session_id),
            events=self.events_for(session_id),
            accuracy_snapshot=self.accuracy_snapshot(),
            profiler_report=profiler_report,
        )

        with self._lock:
            self._reports.append(report)

        return report

    def fail_session(
        self,
        session_id: str,
        *,
        error: str,
    ) -> SpeculativeExecutionResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            failed = state.model_copy(
                update={
                    "status": SpeculationStatus.FAILED,
                    "failed_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = failed
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=SpeculativeExecutionEventKind.SESSION_FAILED,
                    reason=SpeculativeExecutionReason.SESSION_FAILED,
                    metadata={"error": error},
                )
            )
            self._last_reason = SpeculativeExecutionReason.SESSION_FAILED

        return SpeculativeExecutionResult(
            success=True,
            reason=SpeculativeExecutionReason.SESSION_FAILED,
            session_id=session_id,
            status=SpeculationStatus.FAILED,
            state=failed,
            message="speculative execution session failed",
        )

    def state_for(self, session_id: str) -> SpeculativeSessionState | None:
        with self._lock:
            return self._states.get(session_id)

    def candidates_for(self, session_id: str) -> tuple[SpeculativeCandidate, ...]:
        with self._lock:
            return tuple(self._candidates.get(session_id, ()))

    def branches_for(self, session_id: str) -> tuple[SpeculativeBranch, ...]:
        with self._lock:
            return tuple(self._branches.get(session_id, ()))

    def events_for(self, session_id: str) -> tuple[SpeculativeExecutionEvent, ...]:
        with self._lock:
            return tuple(self._events.get(session_id, ()))

    def reports(self) -> tuple[SpeculativeExecutionReport, ...]:
        with self._lock:
            return tuple(self._reports)

    def latest_report(self) -> SpeculativeExecutionReport | None:
        with self._lock:
            if not self._reports:
                return None

            return self._reports[-1]

    def accuracy_snapshot(self) -> SpeculationAccuracySnapshot:
        with self._lock:
            accuracy = (
                self._confirmed_predictions / self._total_predictions
                if self._total_predictions > 0
                else 0.0
            )
            lookahead = self._lookahead_depth()

            return SpeculationAccuracySnapshot(
                total_predictions=self._total_predictions,
                confirmed_predictions=self._confirmed_predictions,
                discarded_predictions=self._discarded_predictions,
                accuracy=accuracy,
                aggressiveness=self._aggressiveness,
                lookahead_depth=lookahead,
            )

    def snapshot(self) -> SpeculativeExecutionRuntimeSnapshot:
        with self._lock:
            states = tuple(self._states.values())
            accuracy = (
                self._confirmed_predictions / self._total_predictions
                if self._total_predictions > 0
                else 0.0
            )

            return SpeculativeExecutionRuntimeSnapshot(
                name=self.name,
                session_count=len(states),
                active_count=sum(
                    1
                    for state in states
                    if state.status == SpeculationStatus.PREWARMING
                ),
                completed_count=sum(
                    1
                    for state in states
                    if state.status == SpeculationStatus.COMPLETED
                ),
                cancelled_count=sum(
                    1
                    for state in states
                    if state.status == SpeculationStatus.CANCELLED
                ),
                failed_count=sum(
                    1 for state in states 
                    if state.status == SpeculationStatus.FAILED
                ),
                candidate_count=sum(len(items) for items in self._candidates.values()),
                branch_count=sum(len(items) for items in self._branches.values()),
                report_count=len(self._reports),
                accuracy=accuracy,
                aggressiveness=self._aggressiveness,
                lookahead_depth=self._lookahead_depth(),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        with self._lock:
            self._states.clear()
            self._candidates.clear()
            self._branches.clear()
            self._events.clear()
            self._reports.clear()
            self._last_reason = SpeculativeExecutionReason.RUNTIME_RESET
            self._total_predictions = 0
            self._confirmed_predictions = 0
            self._discarded_predictions = 0
            self._aggressiveness = SpeculationAggressiveness.NORMAL

    def _select_candidates(
        self,
        candidates: tuple[SpeculativeCandidate, ...],
    ) -> tuple[SpeculativeCandidate, ...]:
        selected = [
            candidate
            for candidate in candidates
            if candidate.probability >= self._config.candidate_probability_threshold
        ]
        selected.sort(key=lambda item: (-item.probability, item.rank))

        return tuple(selected[: self._lookahead_depth()])

    def _lookahead_depth(self) -> int:
        if self._aggressiveness == SpeculationAggressiveness.LOW:
            return self._config.max_candidates_low

        if self._aggressiveness == SpeculationAggressiveness.HIGH:
            return self._config.max_candidates_high

        return self._config.max_candidates_normal

    def _create_branch(
        self,
        *,
        session_id: str,
        candidate: SpeculativeCandidate,
    ) -> SpeculativeBranch:
        branch = SpeculativeBranch(
            session_id=session_id,
            candidate=candidate,
            status=SpeculationStatus.PREWARMING,
        )

        with self._lock:
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=SpeculativeExecutionEventKind.BRANCH_CREATED,
                    reason=SpeculativeExecutionReason.BRANCH_CREATED,
                    branch_id=branch.branch_id,
                    candidate_id=candidate.candidate_id,
                    probability=candidate.probability,
                )
            )

        return branch

    def _prewarm_branch(self, branch: SpeculativeBranch) -> SpeculativeBranch:
        work_items = self._work_items_for(branch)
        completed_items = tuple(
            item.model_copy(
                update={
                    "status": SpeculationStatus.READY,
                    "completed_at_ns": time.perf_counter_ns(),
                }
            )
            for item in work_items
        )
        ready = branch.model_copy(
            update={
                "status": SpeculationStatus.READY,
                "work_items": completed_items,
                "ready_at_ns": time.perf_counter_ns(),
            }
        )

        with self._lock:
            for item in completed_items:
                self._events[branch.session_id].append(
                    self._event(
                        session_id=branch.session_id,
                        kind=SpeculativeExecutionEventKind.WORK_PREWARMED,
                        reason=self._reason_for_work(item.work_type),
                        branch_id=branch.branch_id,
                        candidate_id=branch.candidate.candidate_id,
                        work_id=item.work_id,
                    )
                )

            self._events[branch.session_id].append(
                self._event(
                    session_id=branch.session_id,
                    kind=SpeculativeExecutionEventKind.BRANCH_READY,
                    reason=SpeculativeExecutionReason.BRANCH_READY,
                    branch_id=branch.branch_id,
                    candidate_id=branch.candidate.candidate_id,
                    latency_ms=ready.ready_latency_ms(),
                    probability=branch.candidate.probability,
                )
            )

        return ready

    def _work_items_for(
        self,
        branch: SpeculativeBranch,
    ) -> tuple[SpeculativeWorkItem, ...]:
        base: list[SpeculativeWorkItem] = [
            SpeculativeWorkItem(
                session_id=branch.session_id,
                branch_id=branch.branch_id,
                candidate_id=branch.candidate.candidate_id,
                work_type=SpeculativeWorkType.MEMORY_PREFETCH,
                description="prepare memory query for likely continuation",
                metadata={
                    "memory_query": MemoryRetrievalQuery(
                        text=branch.candidate.prompt_hint,
                        trace_id=uuid4().hex,
                        speculative=True,
                    ).model_dump(mode="json")
                },
            ),
            SpeculativeWorkItem(
                session_id=branch.session_id,
                branch_id=branch.branch_id,
                candidate_id=branch.candidate.candidate_id,
                work_type=SpeculativeWorkType.LLM_CONTEXT_PREWARM,
                description="prewarm LLM context envelope without generation",
            ),
        ]

        if branch.candidate.intent in {
            PartialIntentKind.COMMAND,
            PartialIntentKind.DEBUGGING,
            PartialIntentKind.RESEARCH,
            PartialIntentKind.TOOL_USE,
        }:
            base.append(
                SpeculativeWorkItem(
                    session_id=branch.session_id,
                    branch_id=branch.branch_id,
                    candidate_id=branch.candidate.candidate_id,
                    work_type=SpeculativeWorkType.TOOL_PLANNER_HINT,
                    description="prepare possible tool planner hint",
                )
            )

        if branch.candidate.intent == PartialIntentKind.COMMAND:
            base.append(
                SpeculativeWorkItem(
                    session_id=branch.session_id,
                    branch_id=branch.branch_id,
                    candidate_id=branch.candidate.candidate_id,
                    work_type=SpeculativeWorkType.ACTION_PREVALIDATION,
                    description="pre-validate likely action envelope without execution",
                )
            )

        return tuple(base)

    def _update_aggressiveness_locked(self, session_id: str) -> None:
        accuracy = (
            self._confirmed_predictions / self._total_predictions
            if self._total_predictions > 0
            else 0.0
        )

        previous = self._aggressiveness

        if accuracy < self._config.low_accuracy_threshold:
            self._aggressiveness = SpeculationAggressiveness.LOW
            reason = SpeculativeExecutionReason.ACCURACY_LOW_REDUCED_AGGRESSIVENESS
        elif accuracy > self._config.high_accuracy_threshold:
            self._aggressiveness = SpeculationAggressiveness.HIGH
            reason = SpeculativeExecutionReason.ACCURACY_HIGH_INCREASED_AGGRESSIVENESS
        else:
            self._aggressiveness = SpeculationAggressiveness.NORMAL
            reason = SpeculativeExecutionReason.ACCURACY_NORMAL_MAINTAINED

        self._events[session_id].append(
            self._event(
                session_id=session_id,
                kind=SpeculativeExecutionEventKind.ACCURACY_UPDATED,
                reason=reason,
                accuracy=accuracy,
                metadata={
                    "previous_aggressiveness": previous.value,
                    "new_aggressiveness": self._aggressiveness.value,
                },
            )
        )

        if previous != self._aggressiveness:
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=SpeculativeExecutionEventKind.AGGRESSIVENESS_UPDATED,
                    reason=reason,
                    accuracy=accuracy,
                    metadata={
                        "previous_aggressiveness": previous.value,
                        "new_aggressiveness": self._aggressiveness.value,
                        "lookahead_depth": self._lookahead_depth(),
                    },
                )
            )

    def _record_speculation_span(self, state: SpeculativeSessionState) -> None:
        if state.started_at_ns is None or state.completed_at_ns is None:
            return

        self._profiler.record_stage(
            trace_id=state.trace_id,
            stage=PipelineStage.CONTEXT_BUILD,
            start_ns=state.started_at_ns,
            end_ns=state.completed_at_ns,
            metadata={"session_id": state.session_id, "speculative": True},
        )

    @staticmethod
    def _reason_for_work(
        work_type: SpeculativeWorkType,
    ) -> SpeculativeExecutionReason:
        if work_type == SpeculativeWorkType.MEMORY_PREFETCH:
            return SpeculativeExecutionReason.MEMORY_PREFETCH_PREPARED

        if work_type == SpeculativeWorkType.LLM_CONTEXT_PREWARM:
            return SpeculativeExecutionReason.LLM_CONTEXT_PREWARMED

        if work_type == SpeculativeWorkType.TOOL_PLANNER_HINT:
            return SpeculativeExecutionReason.TOOL_PLANNER_HINT_PREPARED

        return SpeculativeExecutionReason.ACTION_PREVALIDATED

    @staticmethod
    def _event(
        *,
        session_id: str,
        kind: SpeculativeExecutionEventKind,
        reason: SpeculativeExecutionReason,
        branch_id: str | None = None,
        candidate_id: str | None = None,
        work_id: str | None = None,
        latency_ms: float | None = None,
        probability: float | None = None,
        accuracy: float | None = None,
        metadata: dict[str, object] | None = None,
    ) -> SpeculativeExecutionEvent:
        return SpeculativeExecutionEvent(
            session_id=session_id,
            kind=kind,
            reason=reason,
            branch_id=branch_id,
            candidate_id=candidate_id,
            work_id=work_id,
            latency_ms=latency_ms,
            probability=probability,
            accuracy=accuracy,
            metadata=metadata or {},
        )

    @staticmethod
    def _missing_session(session_id: str) -> SpeculativeExecutionResult:
        return SpeculativeExecutionResult(
            success=False,
            reason=SpeculativeExecutionReason.SESSION_NOT_FOUND,
            session_id=session_id,
            status=SpeculationStatus.FAILED,
            message="speculative execution session not found",
        )

    @staticmethod
    def _failure(
        *,
        session_id: str,
        reason: SpeculativeExecutionReason,
        status: SpeculationStatus,
        message: str,
        state: SpeculativeSessionState | None = None,
    ) -> SpeculativeExecutionResult:
        return SpeculativeExecutionResult(
            success=False,
            reason=reason,
            session_id=session_id,
            status=status,
            state=state,
            message=message,
        )