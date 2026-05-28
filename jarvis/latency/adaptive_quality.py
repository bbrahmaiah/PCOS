from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from uuid import uuid4

from pydantic import Field, field_validator

from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class QualityLatencyProfile(StrEnum):
    """
    Runtime quality profile selected from latency pressure.

    Profile switching is internal. The user should experience a smooth response,
    not visible mode changes.
    """

    FULL_QUALITY = "full_quality"
    BALANCED = "balanced"
    FAST_MODE = "fast_mode"


class QualityLatencyStatus(StrEnum):
    """
    Adaptive quality runtime status.
    """

    CREATED = "created"
    ACTIVE = "active"
    SWITCHED = "switched"
    STABLE = "stable"
    CANCELLED = "cancelled"
    FAILED = "failed"


class QualityLatencyReason(StrEnum):
    """
    Machine-readable adaptive quality reasons.
    """

    SESSION_CREATED = "session_created"
    PROFILE_SELECTED_FULL_QUALITY = "profile_selected_full_quality"
    PROFILE_SELECTED_BALANCED = "profile_selected_balanced"
    PROFILE_SELECTED_FAST_MODE = "profile_selected_fast_mode"
    PROFILE_UNCHANGED = "profile_unchanged"
    PROFILE_SWITCHED = "profile_switched"
    SWITCH_WITHIN_BUDGET = "switch_within_budget"
    SWITCH_OVER_BUDGET = "switch_over_budget"
    SESSION_CANCELLED = "session_cancelled"
    SESSION_FAILED = "session_failed"
    SESSION_NOT_FOUND = "session_not_found"
    RUNTIME_RESET = "runtime_reset"


class QualityLatencyEventKind(StrEnum):
    """
    Adaptive quality event kind.
    """

    SESSION_CREATED = "session_created"
    PRESSURE_EVALUATED = "pressure_evaluated"
    PROFILE_SELECTED = "profile_selected"
    PROFILE_SWITCHED = "profile_switched"
    SESSION_CANCELLED = "session_cancelled"
    SESSION_FAILED = "session_failed"


class MemoryRetrievalMode(StrEnum):
    """
    Memory retrieval depth selected by profile.
    """

    ALL_STREAMS = "all_streams"
    EPISODIC_PROFILE_ONLY = "episodic_profile_only"
    PROFILE_ONLY = "profile_only"


class SpeculationMode(StrEnum):
    """
    Speculation depth selected by profile.
    """

    FULL = "full"
    TOP_ONE = "top_one"
    DISABLED = "disabled"


class ContextDepthMode(StrEnum):
    """
    Context depth selected by profile.
    """

    MAXIMUM = "maximum"
    COMPRESSED_PREVIOUS_TURNS = "compressed_previous_turns"
    MINIMAL_LAST_TWO_PLUS_PROFILE = "minimal_last_two_plus_profile"


class TTSQualityMode(StrEnum):
    """
    TTS quality selected by profile.
    """

    HIGH_QUALITY = "high_quality"
    STANDARD = "standard"
    FASTEST = "fastest"


class QualityLatencyDecision(OrchestrationModel):
    """
    Applied runtime decision for one profile selection.
    """

    decision_id: str = Field(default_factory=lambda: uuid4().hex)
    profile: QualityLatencyProfile
    memory_mode: MemoryRetrievalMode
    speculation_mode: SpeculationMode
    context_mode: ContextDepthMode
    tts_mode: TTSQualityMode
    max_context_tokens: int = Field(gt=0)
    speculation_candidates: int = Field(ge=0)
    use_semantic_memory: bool
    use_episodic_memory: bool
    use_profile_memory: bool = True
    compress_previous_turns: bool
    switch_latency_ms: float = Field(default=0.0, ge=0)
    created_at_ns: int = Field(default_factory=time.perf_counter_ns, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("decision_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("decision_id cannot be empty.")

        return cleaned


class QualityLatencyPressureSample(OrchestrationModel):
    """
    Latency pressure sample.

    budget_used_ratio is the core input:
    - < 0.40 → FULL_QUALITY
    - 0.40..0.70 → BALANCED
    - > 0.70 → FAST_MODE
    """

    sample_id: str = Field(default_factory=lambda: uuid4().hex)
    budget_used_ratio: float = Field(ge=0)
    observed_latency_ms: float = Field(ge=0)
    budget_ms: float = Field(gt=0)
    queue_depth: int = Field(default=0, ge=0)
    active_workers: int = Field(default=0, ge=0)
    created_at_ns: int = Field(default_factory=time.perf_counter_ns, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("sample_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("sample_id cannot be empty.")

        return cleaned


class QualityLatencyEvent(OrchestrationModel):
    """
    Typed event for adaptive quality observability.
    """

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    kind: QualityLatencyEventKind
    reason: QualityLatencyReason
    profile: QualityLatencyProfile | None = None
    previous_profile: QualityLatencyProfile | None = None
    budget_used_ratio: float | None = None
    switch_latency_ms: float | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("event_id", "session_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class QualityLatencySessionState(OrchestrationModel):
    """
    State for one adaptive quality session.
    """

    session_id: str = Field(default_factory=lambda: uuid4().hex)
    trace_id: str = Field(default_factory=lambda: uuid4().hex)
    status: QualityLatencyStatus = QualityLatencyStatus.CREATED
    current_profile: QualityLatencyProfile = QualityLatencyProfile.FULL_QUALITY
    started_at_ns: int = Field(default_factory=time.perf_counter_ns, ge=0)
    last_switch_at_ns: int | None = None
    cancelled_at_ns: int | None = None
    failed_at_ns: int | None = None
    sample_count: int = Field(default=0, ge=0)
    switch_count: int = Field(default=0, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("session_id", "trace_id")
    @classmethod
    def _required_ids(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class QualityLatencyResult(OrchestrationModel):
    """
    Result from adaptive quality runtime operation.
    """

    success: bool
    reason: QualityLatencyReason
    session_id: str
    status: QualityLatencyStatus
    decision: QualityLatencyDecision | None = None
    event: QualityLatencyEvent | None = None
    state: QualityLatencySessionState | None = None
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


class QualityLatencyReport(OrchestrationModel):
    """
    Runtime report for adaptive quality.
    """

    session_id: str
    trace_id: str
    status: QualityLatencyStatus
    current_profile: QualityLatencyProfile
    sample_count: int = Field(ge=0)
    switch_count: int = Field(ge=0)
    decisions: tuple[QualityLatencyDecision, ...]
    events: tuple[QualityLatencyEvent, ...]
    created_at: object = Field(default_factory=utc_now)

    @field_validator("session_id", "trace_id")
    @classmethod
    def _required_ids(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class QualityLatencyRuntimeSnapshot(OrchestrationModel):
    """
    Diagnostics for Phase 7 Step 15.
    """

    name: str
    session_count: int = Field(ge=0)
    active_count: int = Field(ge=0)
    cancelled_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    decision_count: int = Field(ge=0)
    report_count: int = Field(ge=0)
    last_reason: QualityLatencyReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class AdaptiveQualityLatencyConfig:
    """
    Adaptive quality-latency runtime configuration.
    """

    name: str = "adaptive_quality_latency_runtime"
    full_quality_max_ratio: float = 0.40
    balanced_max_ratio: float = 0.70
    switch_budget_ms: float = 50.0
    full_quality_context_tokens: int = 8192
    balanced_context_tokens: int = 4096
    fast_context_tokens: int = 1536

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if not 0 <= self.full_quality_max_ratio <= 1:
            raise ValueError("full_quality_max_ratio must be within 0..1.")

        if not 0 <= self.balanced_max_ratio <= 1:
            raise ValueError("balanced_max_ratio must be within 0..1.")

        if self.full_quality_max_ratio >= self.balanced_max_ratio:
            raise ValueError("full threshold must be lower than balanced threshold.")

        if self.switch_budget_ms <= 0:
            raise ValueError("switch_budget_ms must be positive.")

        for field_name, value in (
            ("full_quality_context_tokens", self.full_quality_context_tokens),
            ("balanced_context_tokens", self.balanced_context_tokens),
            ("fast_context_tokens", self.fast_context_tokens),
        ):
            if value <= 0:
                raise ValueError(f"{field_name} must be positive.")


class AdaptiveQualityLatencyRuntime:
    """
    Phase 7 Step 15 Adaptive Quality-Latency Tradeoff.

    Responsibilities:
    - evaluate latency pressure
    - select invisible quality profile
    - produce runtime knobs for memory, speculation, context, and TTS
    - switch profiles under 50ms
    - preserve correctness while reducing context depth under pressure

    Non-responsibilities:
    - no user-visible mode announcements
    - no direct model calls
    - no direct memory retrieval
    - no direct TTS synthesis
    - no action execution
    """

    def __init__(
        self,
        *,
        config: AdaptiveQualityLatencyConfig | None = None,
    ) -> None:
        self._config = config or AdaptiveQualityLatencyConfig()
        self._config.validate()

        self._states: dict[str, QualityLatencySessionState] = {}
        self._samples: dict[str, list[QualityLatencyPressureSample]] = {}
        self._decisions: dict[str, list[QualityLatencyDecision]] = {}
        self._events: dict[str, list[QualityLatencyEvent]] = {}
        self._reports: list[QualityLatencyReport] = []
        self._lock = RLock()
        self._last_reason: QualityLatencyReason | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def create_session(
        self,
        *,
        trace_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> QualityLatencySessionState:
        state = QualityLatencySessionState(
            trace_id=trace_id or uuid4().hex,
            status=QualityLatencyStatus.ACTIVE,
            metadata=metadata or {},
        )
        event = self._event(
            session_id=state.session_id,
            kind=QualityLatencyEventKind.SESSION_CREATED,
            reason=QualityLatencyReason.SESSION_CREATED,
            profile=state.current_profile,
        )

        with self._lock:
            self._states[state.session_id] = state
            self._samples[state.session_id] = []
            self._decisions[state.session_id] = []
            self._events[state.session_id] = [event]
            self._last_reason = QualityLatencyReason.SESSION_CREATED

        return state

    def evaluate_pressure(
        self,
        *,
        session_id: str,
        sample: QualityLatencyPressureSample,
    ) -> QualityLatencyResult:
        state = self.state_for(session_id)

        if state is None:
            return self._missing_session(session_id)

        if state.status != QualityLatencyStatus.ACTIVE:
            return self._failure(
                session_id=session_id,
                reason=QualityLatencyReason.SESSION_NOT_FOUND,
                status=state.status,
                message="quality session is not active",
                state=state,
            )

        start_ns = time.perf_counter_ns()
        selected = self._profile_for_ratio(sample.budget_used_ratio)
        decision = self._decision_for_profile(
            profile=selected,
            switch_latency_ms=0.0,
        )
        switch_latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        decision = decision.model_copy(update={"switch_latency_ms": switch_latency_ms})

        switched = selected != state.current_profile
        reason = self._reason_for_profile(selected)

        if switched:
            reason = QualityLatencyReason.PROFILE_SWITCHED

        switch_budget_reason = (
            QualityLatencyReason.SWITCH_WITHIN_BUDGET
            if switch_latency_ms <= self._config.switch_budget_ms
            else QualityLatencyReason.SWITCH_OVER_BUDGET
        )

        with self._lock:
            current = self._states[session_id]
            updated = current.model_copy(
                update={
                    "status": (
                        QualityLatencyStatus.SWITCHED
                        if switched
                        else QualityLatencyStatus.STABLE
                    ),
                    "current_profile": selected,
                    "last_switch_at_ns": (
                        time.perf_counter_ns()
                        if switched
                        else current.last_switch_at_ns
                    ),
                    "sample_count": current.sample_count + 1,
                    "switch_count": current.switch_count + (1 if switched else 0),
                }
            )
            active_state = updated.model_copy(
                update={"status": QualityLatencyStatus.ACTIVE}
            )
            self._states[session_id] = active_state
            self._samples[session_id].append(sample)
            self._decisions[session_id].append(decision)
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=QualityLatencyEventKind.PRESSURE_EVALUATED,
                    reason=reason if switched else self._reason_for_profile(selected),
                    profile=selected,
                    previous_profile=state.current_profile,
                    budget_used_ratio=sample.budget_used_ratio,
                    switch_latency_ms=switch_latency_ms,
                )
            )
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=QualityLatencyEventKind.PROFILE_SELECTED,
                    reason=switch_budget_reason,
                    profile=selected,
                    previous_profile=state.current_profile,
                    budget_used_ratio=sample.budget_used_ratio,
                    switch_latency_ms=switch_latency_ms,
                )
            )

            if switched:
                self._events[session_id].append(
                    self._event(
                        session_id=session_id,
                        kind=QualityLatencyEventKind.PROFILE_SWITCHED,
                        reason=QualityLatencyReason.PROFILE_SWITCHED,
                        profile=selected,
                        previous_profile=state.current_profile,
                        budget_used_ratio=sample.budget_used_ratio,
                        switch_latency_ms=switch_latency_ms,
                    )
                )

            self._last_reason = switch_budget_reason

        return QualityLatencyResult(
            success=True,
            reason=switch_budget_reason,
            session_id=session_id,
            status=QualityLatencyStatus.ACTIVE,
            decision=decision,
            state=active_state,
            message="quality-latency profile evaluated",
        )

    def current_decision(
        self,
        session_id: str,
    ) -> QualityLatencyDecision | None:
        with self._lock:
            decisions = self._decisions.get(session_id, ())

            if not decisions:
                return None

            return decisions[-1]

    def build_report(self, session_id: str) -> QualityLatencyReport:
        state = self.state_for(session_id)

        if state is None:
            raise ValueError(f"quality-latency session not found: {session_id}")

        report = QualityLatencyReport(
            session_id=session_id,
            trace_id=state.trace_id,
            status=state.status,
            current_profile=state.current_profile,
            sample_count=state.sample_count,
            switch_count=state.switch_count,
            decisions=self.decisions_for(session_id),
            events=self.events_for(session_id),
        )

        with self._lock:
            self._reports.append(report)

        return report

    def cancel_session(self, session_id: str) -> QualityLatencyResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            cancelled = state.model_copy(
                update={
                    "status": QualityLatencyStatus.CANCELLED,
                    "cancelled_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = cancelled
            event = self._event(
                session_id=session_id,
                kind=QualityLatencyEventKind.SESSION_CANCELLED,
                reason=QualityLatencyReason.SESSION_CANCELLED,
                profile=cancelled.current_profile,
            )
            self._events[session_id].append(event)
            self._last_reason = QualityLatencyReason.SESSION_CANCELLED

        return QualityLatencyResult(
            success=True,
            reason=QualityLatencyReason.SESSION_CANCELLED,
            session_id=session_id,
            status=QualityLatencyStatus.CANCELLED,
            event=event,
            state=cancelled,
            message="quality-latency session cancelled",
        )

    def fail_session(
        self,
        session_id: str,
        *,
        error: str,
    ) -> QualityLatencyResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            failed = state.model_copy(
                update={
                    "status": QualityLatencyStatus.FAILED,
                    "failed_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = failed
            event = self._event(
                session_id=session_id,
                kind=QualityLatencyEventKind.SESSION_FAILED,
                reason=QualityLatencyReason.SESSION_FAILED,
                profile=failed.current_profile,
                metadata={"error": error},
            )
            self._events[session_id].append(event)
            self._last_reason = QualityLatencyReason.SESSION_FAILED

        return QualityLatencyResult(
            success=True,
            reason=QualityLatencyReason.SESSION_FAILED,
            session_id=session_id,
            status=QualityLatencyStatus.FAILED,
            event=event,
            state=failed,
            message="quality-latency session failed",
        )

    def state_for(self, session_id: str) -> QualityLatencySessionState | None:
        with self._lock:
            return self._states.get(session_id)

    def samples_for(
        self,
        session_id: str,
    ) -> tuple[QualityLatencyPressureSample, ...]:
        with self._lock:
            return tuple(self._samples.get(session_id, ()))

    def decisions_for(
        self,
        session_id: str,
    ) -> tuple[QualityLatencyDecision, ...]:
        with self._lock:
            return tuple(self._decisions.get(session_id, ()))

    def events_for(self, session_id: str) -> tuple[QualityLatencyEvent, ...]:
        with self._lock:
            return tuple(self._events.get(session_id, ()))

    def reports(self) -> tuple[QualityLatencyReport, ...]:
        with self._lock:
            return tuple(self._reports)

    def latest_report(self) -> QualityLatencyReport | None:
        with self._lock:
            if not self._reports:
                return None

            return self._reports[-1]

    def snapshot(self) -> QualityLatencyRuntimeSnapshot:
        with self._lock:
            states = tuple(self._states.values())

            return QualityLatencyRuntimeSnapshot(
                name=self.name,
                session_count=len(states),
                active_count=sum(
                    1 
                    for state in states 
                    if state.status == QualityLatencyStatus.ACTIVE
                ),
                cancelled_count=sum(
                    1
                    for state in states
                    if state.status == QualityLatencyStatus.CANCELLED
                ),
                failed_count=sum(
                    1 
                    for state in states 
                    if state.status == QualityLatencyStatus.FAILED
                ),
                decision_count=sum(len(items) for items in self._decisions.values()),
                report_count=len(self._reports),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        with self._lock:
            self._states.clear()
            self._samples.clear()
            self._decisions.clear()
            self._events.clear()
            self._reports.clear()
            self._last_reason = QualityLatencyReason.RUNTIME_RESET

    def _profile_for_ratio(self, ratio: float) -> QualityLatencyProfile:
        if ratio < self._config.full_quality_max_ratio:
            return QualityLatencyProfile.FULL_QUALITY

        if ratio <= self._config.balanced_max_ratio:
            return QualityLatencyProfile.BALANCED

        return QualityLatencyProfile.FAST_MODE

    def _decision_for_profile(
        self,
        *,
        profile: QualityLatencyProfile,
        switch_latency_ms: float,
    ) -> QualityLatencyDecision:
        if profile == QualityLatencyProfile.FULL_QUALITY:
            return QualityLatencyDecision(
                profile=profile,
                memory_mode=MemoryRetrievalMode.ALL_STREAMS,
                speculation_mode=SpeculationMode.FULL,
                context_mode=ContextDepthMode.MAXIMUM,
                tts_mode=TTSQualityMode.HIGH_QUALITY,
                max_context_tokens=self._config.full_quality_context_tokens,
                speculation_candidates=2,
                use_semantic_memory=True,
                use_episodic_memory=True,
                compress_previous_turns=False,
                switch_latency_ms=switch_latency_ms,
            )

        if profile == QualityLatencyProfile.BALANCED:
            return QualityLatencyDecision(
                profile=profile,
                memory_mode=MemoryRetrievalMode.EPISODIC_PROFILE_ONLY,
                speculation_mode=SpeculationMode.TOP_ONE,
                context_mode=ContextDepthMode.COMPRESSED_PREVIOUS_TURNS,
                tts_mode=TTSQualityMode.STANDARD,
                max_context_tokens=self._config.balanced_context_tokens,
                speculation_candidates=1,
                use_semantic_memory=False,
                use_episodic_memory=True,
                compress_previous_turns=True,
                switch_latency_ms=switch_latency_ms,
            )

        return QualityLatencyDecision(
            profile=profile,
            memory_mode=MemoryRetrievalMode.PROFILE_ONLY,
            speculation_mode=SpeculationMode.DISABLED,
            context_mode=ContextDepthMode.MINIMAL_LAST_TWO_PLUS_PROFILE,
            tts_mode=TTSQualityMode.FASTEST,
            max_context_tokens=self._config.fast_context_tokens,
            speculation_candidates=0,
            use_semantic_memory=False,
            use_episodic_memory=False,
            compress_previous_turns=True,
            switch_latency_ms=switch_latency_ms,
        )

    @staticmethod
    def _reason_for_profile(
        profile: QualityLatencyProfile,
    ) -> QualityLatencyReason:
        if profile == QualityLatencyProfile.FULL_QUALITY:
            return QualityLatencyReason.PROFILE_SELECTED_FULL_QUALITY

        if profile == QualityLatencyProfile.BALANCED:
            return QualityLatencyReason.PROFILE_SELECTED_BALANCED

        return QualityLatencyReason.PROFILE_SELECTED_FAST_MODE

    @staticmethod
    def _event(
        *,
        session_id: str,
        kind: QualityLatencyEventKind,
        reason: QualityLatencyReason,
        profile: QualityLatencyProfile | None = None,
        previous_profile: QualityLatencyProfile | None = None,
        budget_used_ratio: float | None = None,
        switch_latency_ms: float | None = None,
        metadata: dict[str, object] | None = None,
    ) -> QualityLatencyEvent:
        return QualityLatencyEvent(
            session_id=session_id,
            kind=kind,
            reason=reason,
            profile=profile,
            previous_profile=previous_profile,
            budget_used_ratio=budget_used_ratio,
            switch_latency_ms=switch_latency_ms,
            metadata=metadata or {},
        )

    @staticmethod
    def _missing_session(session_id: str) -> QualityLatencyResult:
        return QualityLatencyResult(
            success=False,
            reason=QualityLatencyReason.SESSION_NOT_FOUND,
            session_id=session_id,
            status=QualityLatencyStatus.FAILED,
            message="quality-latency session not found",
        )

    @staticmethod
    def _failure(
        *,
        session_id: str,
        reason: QualityLatencyReason,
        status: QualityLatencyStatus,
        message: str,
        state: QualityLatencySessionState | None = None,
    ) -> QualityLatencyResult:
        return QualityLatencyResult(
            success=False,
            reason=reason,
            session_id=session_id,
            status=status,
            state=state,
            message=message,
        )