from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class LoadScenarioKind(StrEnum):
    """
    Phase 7 load validation scenarios.
    """

    LONG_CONVERSATION = "long_conversation"
    BACKGROUND_TASK_PRESSURE = "background_task_pressure"
    MEMORY_PRESSURE = "memory_pressure"
    INTERRUPTION_STORM = "interruption_storm"


class LoadValidationStatus(StrEnum):
    """
    Load validation lifecycle status.
    """

    CREATED = "created"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class LoadValidationReason(StrEnum):
    """
    Machine-readable load validation reasons.
    """

    SESSION_CREATED = "session_created"
    SESSION_STARTED = "session_started"
    SCENARIO_STARTED = "scenario_started"
    SAMPLE_RECORDED = "sample_recorded"
    DEGRADATION_ACTIVATED = "degradation_activated"
    LOAD_SHEDDING_ACTIVATED = "load_shedding_activated"
    SCENARIO_PASSED = "scenario_passed"
    SCENARIO_FAILED_LATENCY_DRIFT = "scenario_failed_latency_drift"
    SCENARIO_FAILED_BUDGET = "scenario_failed_budget"
    SCENARIO_FAILED_DEGRADATION = "scenario_failed_degradation"
    SCENARIO_FAILED_INTERRUPTION_RECOVERY = (
        "scenario_failed_interruption_recovery"
    )
    REPORT_BUILT = "report_built"
    SESSION_CANCELLED = "session_cancelled"
    SESSION_NOT_FOUND = "session_not_found"
    SESSION_NOT_RUNNING = "session_not_running"
    RUNTIME_RESET = "runtime_reset"


class LoadValidationEventKind(StrEnum):
    """
    Load validation event kind.
    """

    SESSION_CREATED = "session_created"
    SESSION_STARTED = "session_started"
    SCENARIO_STARTED = "scenario_started"
    SAMPLE_RECORDED = "sample_recorded"
    DEGRADATION_ACTIVATED = "degradation_activated"
    LOAD_SHEDDING_ACTIVATED = "load_shedding_activated"
    SCENARIO_EVALUATED = "scenario_evaluated"
    REPORT_BUILT = "report_built"
    SESSION_CANCELLED = "session_cancelled"


class DegradationMode(StrEnum):
    """
    Graceful degradation mode under load.
    """

    NONE = "none"
    COMPRESS_CONTEXT = "compress_context"
    SHED_BACKGROUND = "shed_background"
    FAST_MEMORY = "fast_memory"
    FAST_TTS = "fast_tts"
    RECOVERY_PRIORITY = "recovery_priority"


class LoadScenarioConfig(OrchestrationModel):
    """
    Configuration for one load scenario.
    """

    scenario_id: str = Field(default_factory=lambda: uuid4().hex)
    kind: LoadScenarioKind
    target_latency_ms: float = Field(gt=0)
    max_latency_drift_ratio: float = Field(default=0.10, ge=0)
    required_samples: int = Field(default=5, ge=1)
    require_degradation: bool = True
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("scenario_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("scenario_id cannot be empty.")

        return cleaned


class LoadSample(OrchestrationModel):
    """
    One load validation sample.
    """

    sample_id: str = Field(default_factory=lambda: uuid4().hex)
    scenario: LoadScenarioKind
    latency_ms: float = Field(ge=0)
    turn_index: int = Field(default=0, ge=0)
    memory_entries: int = Field(default=0, ge=0)
    background_tasks: int = Field(default=0, ge=0)
    interruption_index: int = Field(default=0, ge=0)
    degradation_mode: DegradationMode = DegradationMode.NONE
    load_shedding_active: bool = False
    recovery_clean: bool = True
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


class LoadScenarioEvaluation(OrchestrationModel):
    """
    Result for one load scenario.
    """

    evaluation_id: str = Field(default_factory=lambda: uuid4().hex)
    scenario: LoadScenarioKind
    status: LoadValidationStatus
    reason: LoadValidationReason
    sample_count: int = Field(ge=0)
    p50_latency_ms: float = Field(default=0.0, ge=0)
    p95_latency_ms: float = Field(default=0.0, ge=0)
    first_latency_ms: float = Field(default=0.0, ge=0)
    last_latency_ms: float = Field(default=0.0, ge=0)
    latency_drift_ratio: float = Field(default=0.0, ge=0)
    degradation_active: bool
    load_shedding_active: bool
    all_recoveries_clean: bool
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("evaluation_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("evaluation_id cannot be empty.")

        return cleaned


class LoadValidationEvent(OrchestrationModel):
    """
    Typed load validation event.
    """

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    kind: LoadValidationEventKind
    reason: LoadValidationReason
    scenario: LoadScenarioKind | None = None
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


class LoadValidationSessionState(OrchestrationModel):
    """
    State for one load validation run.
    """

    session_id: str = Field(default_factory=lambda: uuid4().hex)
    trace_id: str = Field(default_factory=lambda: uuid4().hex)
    status: LoadValidationStatus = LoadValidationStatus.CREATED
    started_at_ns: int | None = None
    completed_at_ns: int | None = None
    cancelled_at_ns: int | None = None
    scenario_count: int = Field(default=0, ge=0)
    sample_count: int = Field(default=0, ge=0)
    failed_count: int = Field(default=0, ge=0)
    degradation_count: int = Field(default=0, ge=0)
    load_shedding_count: int = Field(default=0, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("session_id", "trace_id")
    @classmethod
    def _required_ids(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class LoadValidationResult(OrchestrationModel):
    """
    Result from load validation operation.
    """

    success: bool
    reason: LoadValidationReason
    session_id: str
    status: LoadValidationStatus
    event: LoadValidationEvent | None = None
    state: LoadValidationSessionState | None = None
    message: str
    created_at: object = Field(default_factory=utc_now)

    @field_validator("session_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class LoadValidationReport(OrchestrationModel):
    """
    Final load validation report.
    """

    session_id: str
    trace_id: str
    status: LoadValidationStatus
    scenario_count: int = Field(ge=0)
    sample_count: int = Field(ge=0)
    passed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    degradation_count: int = Field(ge=0)
    load_shedding_count: int = Field(ge=0)
    evaluations: tuple[LoadScenarioEvaluation, ...]
    events: tuple[LoadValidationEvent, ...]
    created_at: object = Field(default_factory=utc_now)

    @field_validator("session_id", "trace_id")
    @classmethod
    def _required_ids(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @model_validator(mode="after")
    def _counts_match(self) -> LoadValidationReport:
        if self.passed_count + self.failed_count != len(self.evaluations):
            raise ValueError("passed_count + failed_count must match evaluations.")

        return self


class LoadValidationRuntimeSnapshot(OrchestrationModel):
    """
    Runtime diagnostics for Phase 7 Step 19.
    """

    name: str
    session_count: int = Field(ge=0)
    running_count: int = Field(ge=0)
    passed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    cancelled_count: int = Field(ge=0)
    sample_count: int = Field(ge=0)
    report_count: int = Field(ge=0)
    last_reason: LoadValidationReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class LoadDegradationRuntimeConfig:
    """
    Phase 7 Step 19 load degradation validation configuration.
    """

    name: str = "load_degradation_validation_runtime"
    conversation_target_ms: float = 800.0
    memory_retrieval_target_ms: float = 150.0
    interruption_recovery_target_ms: float = 300.0
    max_turn_drift_ratio: float = 0.10
    long_conversation_turns: int = 50
    memory_pressure_entries: int = 10_000
    interruption_storm_count: int = 5
    interruption_storm_window_seconds: int = 10

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        for field_name, value in (
            ("conversation_target_ms", self.conversation_target_ms),
            ("memory_retrieval_target_ms", self.memory_retrieval_target_ms),
            ("interruption_recovery_target_ms", self.interruption_recovery_target_ms),
        ):
            if value <= 0:
                raise ValueError(f"{field_name} must be positive.")

        if self.max_turn_drift_ratio < 0:
            raise ValueError("max_turn_drift_ratio cannot be negative.")

        if self.long_conversation_turns < 50:
            raise ValueError("long_conversation_turns must be at least 50.")

        if self.memory_pressure_entries < 10_000:
            raise ValueError("memory_pressure_entries must be at least 10,000.")

        if self.interruption_storm_count < 5:
            raise ValueError("interruption_storm_count must be at least 5.")

        if self.interruption_storm_window_seconds <= 0:
            raise ValueError("interruption_storm_window_seconds must be positive.")


class LoadDegradationValidationRuntime:
    """
    Phase 7 Step 19 Load Testing & Degradation Validation.

    Responsibilities:
    - validate latency over 50+ turn conversations
    - validate background task pressure protection
    - validate memory retrieval under 10,000+ memory entries
    - validate rapid interruption storm recovery
    - require graceful degradation when load pressure exists
    - require load shedding to protect conversation

    Non-responsibilities:
    - no real shell/test execution
    - no real vector database load
    - no real memory writes
    - no real audio generation
    """

    def __init__(
        self,
        *,
        config: LoadDegradationRuntimeConfig | None = None,
    ) -> None:
        self._config = config or LoadDegradationRuntimeConfig()
        self._config.validate()

        self._states: dict[str, LoadValidationSessionState] = {}
        self._scenarios: dict[str, list[LoadScenarioConfig]] = {}
        self._samples: dict[str, list[LoadSample]] = {}
        self._events: dict[str, list[LoadValidationEvent]] = {}
        self._reports: list[LoadValidationReport] = []
        self._lock = RLock()
        self._last_reason: LoadValidationReason | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def create_session(
        self,
        *,
        trace_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> LoadValidationSessionState:
        scenarios = self.default_scenarios()
        state = LoadValidationSessionState(
            trace_id=trace_id or uuid4().hex,
            scenario_count=len(scenarios),
            metadata=metadata or {},
        )
        event = self._event(
            session_id=state.session_id,
            kind=LoadValidationEventKind.SESSION_CREATED,
            reason=LoadValidationReason.SESSION_CREATED,
        )

        with self._lock:
            self._states[state.session_id] = state
            self._scenarios[state.session_id] = list(scenarios)
            self._samples[state.session_id] = []
            self._events[state.session_id] = [event]
            self._last_reason = LoadValidationReason.SESSION_CREATED

        return state

    def start_session(self, session_id: str) -> LoadValidationResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            if state.status != LoadValidationStatus.CREATED:
                return self._failure(
                    session_id=session_id,
                    reason=LoadValidationReason.SESSION_NOT_RUNNING,
                    status=state.status,
                    message="load validation session cannot start",
                    state=state,
                )

            started = state.model_copy(
                update={
                    "status": LoadValidationStatus.RUNNING,
                    "started_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = started
            event = self._event(
                session_id=session_id,
                kind=LoadValidationEventKind.SESSION_STARTED,
                reason=LoadValidationReason.SESSION_STARTED,
            )
            self._events[session_id].append(event)
            self._last_reason = LoadValidationReason.SESSION_STARTED

        return LoadValidationResult(
            success=True,
            reason=LoadValidationReason.SESSION_STARTED,
            session_id=session_id,
            status=LoadValidationStatus.RUNNING,
            event=event,
            state=started,
            message="load degradation validation started",
        )

    def record_sample(
        self,
        *,
        session_id: str,
        sample: LoadSample,
    ) -> LoadValidationResult:
        state = self.state_for(session_id)

        if state is None:
            return self._missing_session(session_id)

        if state.status != LoadValidationStatus.RUNNING:
            return self._failure(
                session_id=session_id,
                reason=LoadValidationReason.SESSION_NOT_RUNNING,
                status=state.status,
                message="load validation session is not running",
                state=state,
            )

        with self._lock:
            self._samples[session_id].append(sample)
            current = self._states[session_id]
            degradation_count = current.degradation_count
            load_shedding_count = current.load_shedding_count

            if sample.degradation_mode != DegradationMode.NONE:
                degradation_count += 1
                self._events[session_id].append(
                    self._event(
                        session_id=session_id,
                        kind=LoadValidationEventKind.DEGRADATION_ACTIVATED,
                        reason=LoadValidationReason.DEGRADATION_ACTIVATED,
                        scenario=sample.scenario,
                        latency_ms=sample.latency_ms,
                    )
                )

            if sample.load_shedding_active:
                load_shedding_count += 1
                self._events[session_id].append(
                    self._event(
                        session_id=session_id,
                        kind=LoadValidationEventKind.LOAD_SHEDDING_ACTIVATED,
                        reason=LoadValidationReason.LOAD_SHEDDING_ACTIVATED,
                        scenario=sample.scenario,
                        latency_ms=sample.latency_ms,
                    )
                )

            updated = current.model_copy(
                update={
                    "sample_count": current.sample_count + 1,
                    "degradation_count": degradation_count,
                    "load_shedding_count": load_shedding_count,
                }
            )
            self._states[session_id] = updated
            event = self._event(
                session_id=session_id,
                kind=LoadValidationEventKind.SAMPLE_RECORDED,
                reason=LoadValidationReason.SAMPLE_RECORDED,
                scenario=sample.scenario,
                latency_ms=sample.latency_ms,
            )
            self._events[session_id].append(event)
            self._last_reason = LoadValidationReason.SAMPLE_RECORDED

        return LoadValidationResult(
            success=True,
            reason=LoadValidationReason.SAMPLE_RECORDED,
            session_id=session_id,
            status=LoadValidationStatus.RUNNING,
            event=event,
            state=updated,
            message="load validation sample recorded",
        )

    def run_simulated_suite(
        self,
        *,
        session_id: str,
        failing: bool = False,
    ) -> LoadValidationReport:
        state = self.state_for(session_id)

        if state is None:
            raise ValueError(f"load validation session not found: {session_id}")

        if state.status != LoadValidationStatus.RUNNING:
            raise ValueError("load validation session is not running")

        for scenario in self.scenarios_for(session_id):
            self._append_event(
                session_id=session_id,
                kind=LoadValidationEventKind.SCENARIO_STARTED,
                reason=LoadValidationReason.SCENARIO_STARTED,
                scenario=scenario.kind,
            )

            for sample in self._simulated_samples_for(
                scenario=scenario,
                failing=failing,
            ):
                self.record_sample(session_id=session_id, sample=sample)

        return self.build_report(session_id)

    def build_report(self, session_id: str) -> LoadValidationReport:
        state = self.state_for(session_id)

        if state is None:
            raise ValueError(f"load validation session not found: {session_id}")

        evaluations = tuple(
            self._evaluate_scenario(
                scenario=scenario,
                samples=self._samples_for_scenario(
                    session_id=session_id,
                    scenario=scenario.kind,
                ),
            )
            for scenario in self.scenarios_for(session_id)
        )
        failed_count = sum(
            1
            for evaluation in evaluations
            if evaluation.status == LoadValidationStatus.FAILED
        )
        passed_count = len(evaluations) - failed_count
        final_status = (
            LoadValidationStatus.FAILED
            if failed_count > 0
            else LoadValidationStatus.PASSED
        )

        with self._lock:
            current = self._states[session_id]
            completed = current.model_copy(
                update={
                    "status": final_status,
                    "completed_at_ns": time.perf_counter_ns(),
                    "failed_count": failed_count,
                }
            )
            self._states[session_id] = completed

            for evaluation in evaluations:
                self._events[session_id].append(
                    self._event(
                        session_id=session_id,
                        kind=LoadValidationEventKind.SCENARIO_EVALUATED,
                        reason=evaluation.reason,
                        scenario=evaluation.scenario,
                        latency_ms=evaluation.p95_latency_ms,
                    )
                )

            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=LoadValidationEventKind.REPORT_BUILT,
                    reason=LoadValidationReason.REPORT_BUILT,
                )
            )
            self._last_reason = LoadValidationReason.REPORT_BUILT

        report = LoadValidationReport(
            session_id=session_id,
            trace_id=state.trace_id,
            status=final_status,
            scenario_count=len(evaluations),
            sample_count=len(self.samples_for(session_id)),
            passed_count=passed_count,
            failed_count=failed_count,
            degradation_count=completed.degradation_count,
            load_shedding_count=completed.load_shedding_count,
            evaluations=evaluations,
            events=self.events_for(session_id),
        )

        with self._lock:
            self._reports.append(report)

        return report

    def cancel_session(self, session_id: str) -> LoadValidationResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            cancelled = state.model_copy(
                update={
                    "status": LoadValidationStatus.CANCELLED,
                    "cancelled_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = cancelled
            event = self._event(
                session_id=session_id,
                kind=LoadValidationEventKind.SESSION_CANCELLED,
                reason=LoadValidationReason.SESSION_CANCELLED,
            )
            self._events[session_id].append(event)
            self._last_reason = LoadValidationReason.SESSION_CANCELLED

        return LoadValidationResult(
            success=True,
            reason=LoadValidationReason.SESSION_CANCELLED,
            session_id=session_id,
            status=LoadValidationStatus.CANCELLED,
            event=event,
            state=cancelled,
            message="load validation session cancelled",
        )

    def default_scenarios(self) -> tuple[LoadScenarioConfig, ...]:
        return (
            LoadScenarioConfig(
                kind=LoadScenarioKind.LONG_CONVERSATION,
                target_latency_ms=self._config.conversation_target_ms,
                max_latency_drift_ratio=self._config.max_turn_drift_ratio,
                required_samples=self._config.long_conversation_turns,
                require_degradation=True,
            ),
            LoadScenarioConfig(
                kind=LoadScenarioKind.BACKGROUND_TASK_PRESSURE,
                target_latency_ms=self._config.conversation_target_ms,
                required_samples=8,
                require_degradation=True,
            ),
            LoadScenarioConfig(
                kind=LoadScenarioKind.MEMORY_PRESSURE,
                target_latency_ms=self._config.memory_retrieval_target_ms,
                required_samples=8,
                require_degradation=True,
            ),
            LoadScenarioConfig(
                kind=LoadScenarioKind.INTERRUPTION_STORM,
                target_latency_ms=self._config.interruption_recovery_target_ms,
                required_samples=self._config.interruption_storm_count,
                require_degradation=True,
            ),
        )

    def state_for(self, session_id: str) -> LoadValidationSessionState | None:
        with self._lock:
            return self._states.get(session_id)

    def scenarios_for(self, session_id: str) -> tuple[LoadScenarioConfig, ...]:
        with self._lock:
            return tuple(self._scenarios.get(session_id, ()))

    def samples_for(self, session_id: str) -> tuple[LoadSample, ...]:
        with self._lock:
            return tuple(self._samples.get(session_id, ()))

    def events_for(self, session_id: str) -> tuple[LoadValidationEvent, ...]:
        with self._lock:
            return tuple(self._events.get(session_id, ()))

    def reports(self) -> tuple[LoadValidationReport, ...]:
        with self._lock:
            return tuple(self._reports)

    def latest_report(self) -> LoadValidationReport | None:
        with self._lock:
            if not self._reports:
                return None

            return self._reports[-1]

    def snapshot(self) -> LoadValidationRuntimeSnapshot:
        with self._lock:
            states = tuple(self._states.values())

            return LoadValidationRuntimeSnapshot(
                name=self.name,
                session_count=len(states),
                running_count=sum(
                    1
                    for state in states
                    if state.status == LoadValidationStatus.RUNNING
                ),
                passed_count=sum(
                    1
                    for state in states
                    if state.status == LoadValidationStatus.PASSED
                ),
                failed_count=sum(
                    1
                    for state in states
                    if state.status == LoadValidationStatus.FAILED
                ),
                cancelled_count=sum(
                    1
                    for state in states
                    if state.status == LoadValidationStatus.CANCELLED
                ),
                sample_count=sum(len(items) for items in self._samples.values()),
                report_count=len(self._reports),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        with self._lock:
            self._states.clear()
            self._scenarios.clear()
            self._samples.clear()
            self._events.clear()
            self._reports.clear()
            self._last_reason = LoadValidationReason.RUNTIME_RESET

    def _samples_for_scenario(
        self,
        *,
        session_id: str,
        scenario: LoadScenarioKind,
    ) -> tuple[LoadSample, ...]:
        return tuple(
            sample 
            for sample in self.samples_for(session_id) 
            if sample.scenario == scenario
        )

    def _evaluate_scenario(
        self,
        *,
        scenario: LoadScenarioConfig,
        samples: tuple[LoadSample, ...],
    ) -> LoadScenarioEvaluation:
        latencies = tuple(sample.latency_ms for sample in samples)
        p50 = statistics.median(latencies) if latencies else 0.0
        p95 = self._percentile(latencies, 95.0)
        first = latencies[0] if latencies else 0.0
        last = latencies[-1] if latencies else 0.0
        drift = ((last - first) / first) if first > 0 and last > first else 0.0
        degradation_active = any(
            sample.degradation_mode != DegradationMode.NONE for sample in samples
        )
        load_shedding_active = any(sample.load_shedding_active for sample in samples)
        recoveries_clean = all(sample.recovery_clean for sample in samples)

        status = LoadValidationStatus.PASSED
        reason = LoadValidationReason.SCENARIO_PASSED

        if len(samples) < scenario.required_samples:
            status = LoadValidationStatus.FAILED
            reason = LoadValidationReason.SCENARIO_FAILED_BUDGET
        elif p95 > scenario.target_latency_ms:
            status = LoadValidationStatus.FAILED
            reason = LoadValidationReason.SCENARIO_FAILED_BUDGET
        elif drift > scenario.max_latency_drift_ratio:
            status = LoadValidationStatus.FAILED
            reason = LoadValidationReason.SCENARIO_FAILED_LATENCY_DRIFT
        elif scenario.require_degradation and not degradation_active:
            status = LoadValidationStatus.FAILED
            reason = LoadValidationReason.SCENARIO_FAILED_DEGRADATION
        elif (
            scenario.kind == LoadScenarioKind.BACKGROUND_TASK_PRESSURE
            and not load_shedding_active
        ):
            status = LoadValidationStatus.FAILED
            reason = LoadValidationReason.SCENARIO_FAILED_DEGRADATION
        elif ( 
             scenario.kind == LoadScenarioKind.INTERRUPTION_STORM 
             and not recoveries_clean
        ):
            status = LoadValidationStatus.FAILED
            reason = LoadValidationReason.SCENARIO_FAILED_INTERRUPTION_RECOVERY

        return LoadScenarioEvaluation(
            scenario=scenario.kind,
            status=status,
            reason=reason,
            sample_count=len(samples),
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            first_latency_ms=first,
            last_latency_ms=last,
            latency_drift_ratio=drift,
            degradation_active=degradation_active,
            load_shedding_active=load_shedding_active,
            all_recoveries_clean=recoveries_clean,
        )

    def _simulated_samples_for(
        self,
        *,
        scenario: LoadScenarioConfig,
        failing: bool,
    ) -> tuple[LoadSample, ...]:
        if scenario.kind == LoadScenarioKind.LONG_CONVERSATION:
            return self._long_conversation_samples(scenario=scenario, failing=failing)

        if scenario.kind == LoadScenarioKind.BACKGROUND_TASK_PRESSURE:
            return self._background_pressure_samples(scenario=scenario, failing=failing)

        if scenario.kind == LoadScenarioKind.MEMORY_PRESSURE:
            return self._memory_pressure_samples(scenario=scenario, failing=failing)

        return self._interruption_storm_samples(scenario=scenario, failing=failing)

    def _long_conversation_samples(
        self,
        *,
        scenario: LoadScenarioConfig,
        failing: bool,
    ) -> tuple[LoadSample, ...]:
        samples: list[LoadSample] = []
        base_latency = scenario.target_latency_ms * 0.65

        for turn in range(1, self._config.long_conversation_turns + 1):
            drift = 0.08 if not failing else 0.25
            latency = base_latency * (1.0 + drift * (turn / 50.0))
            samples.append(
                LoadSample(
                    scenario=LoadScenarioKind.LONG_CONVERSATION,
                    latency_ms=latency,
                    turn_index=turn,
                    degradation_mode=(
                        DegradationMode.COMPRESS_CONTEXT
                        if turn >= 20
                        else DegradationMode.NONE
                    ),
                    load_shedding_active=False,
                )
            )

        return tuple(samples)

    def _background_pressure_samples(
        self,
        *,
        scenario: LoadScenarioConfig,
        failing: bool,
    ) -> tuple[LoadSample, ...]:
        return tuple(
            LoadSample(
                scenario=LoadScenarioKind.BACKGROUND_TASK_PRESSURE,
                latency_ms=scenario.target_latency_ms * (0.70 if not failing else 1.20),
                background_tasks=3,
                degradation_mode=(
                    DegradationMode.SHED_BACKGROUND
                    if not failing
                    else DegradationMode.NONE
                ),
                load_shedding_active=not failing,
            )
            for _ in range(scenario.required_samples)
        )

    def _memory_pressure_samples(
        self,
        *,
        scenario: LoadScenarioConfig,
        failing: bool,
    ) -> tuple[LoadSample, ...]:
        return tuple(
            LoadSample(
                scenario=LoadScenarioKind.MEMORY_PRESSURE,
                latency_ms=scenario.target_latency_ms * (0.80 if not failing else 1.30),
                memory_entries=self._config.memory_pressure_entries,
                degradation_mode=(
                    DegradationMode.FAST_MEMORY
                    if not failing
                    else DegradationMode.NONE
                ),
            )
            for _ in range(scenario.required_samples)
        )

    def _interruption_storm_samples(
        self,
        *,
        scenario: LoadScenarioConfig,
        failing: bool,
    ) -> tuple[LoadSample, ...]:
        return tuple(
            LoadSample(
                scenario=LoadScenarioKind.INTERRUPTION_STORM,
                latency_ms=scenario.target_latency_ms * (0.70 if not failing else 1.20),
                interruption_index=index + 1,
                degradation_mode=(
                    DegradationMode.RECOVERY_PRIORITY
                    if not failing
                    else DegradationMode.NONE
                ),
                recovery_clean=not failing,
            )
            for index in range(self._config.interruption_storm_count)
        )

    @staticmethod
    def _percentile(values: tuple[float, ...], percentile: float) -> float:
        if not values:
            return 0.0

        if len(values) == 1:
            return values[0]

        sorted_values = sorted(values)
        rank = (percentile / 100.0) * (len(sorted_values) - 1)
        lower_index = int(rank)
        upper_index = min(lower_index + 1, len(sorted_values) - 1)
        fraction = rank - lower_index

        return (
            sorted_values[lower_index] * (1.0 - fraction)
            + sorted_values[upper_index] * fraction
        )

    def _append_event(
        self,
        *,
        session_id: str,
        kind: LoadValidationEventKind,
        reason: LoadValidationReason,
        scenario: LoadScenarioKind | None = None,
        latency_ms: float | None = None,
    ) -> None:
        with self._lock:
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=kind,
                    reason=reason,
                    scenario=scenario,
                    latency_ms=latency_ms,
                )
            )
            self._last_reason = reason

    @staticmethod
    def _event(
        *,
        session_id: str,
        kind: LoadValidationEventKind,
        reason: LoadValidationReason,
        scenario: LoadScenarioKind | None = None,
        latency_ms: float | None = None,
        metadata: dict[str, object] | None = None,
    ) -> LoadValidationEvent:
        return LoadValidationEvent(
            session_id=session_id,
            kind=kind,
            reason=reason,
            scenario=scenario,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )

    @staticmethod
    def _missing_session(session_id: str) -> LoadValidationResult:
        return LoadValidationResult(
            success=False,
            reason=LoadValidationReason.SESSION_NOT_FOUND,
            session_id=session_id,
            status=LoadValidationStatus.FAILED,
            message="load validation session not found",
        )

    @staticmethod
    def _failure(
        *,
        session_id: str,
        reason: LoadValidationReason,
        status: LoadValidationStatus,
        message: str,
        state: LoadValidationSessionState | None = None,
    ) -> LoadValidationResult:
        return LoadValidationResult(
            success=False,
            reason=reason,
            session_id=session_id,
            status=status,
            state=state,
            message=message,
        )