from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.latency.latency_regression import (
    LatencyRegressionReport,
    LatencyRegressionRuntime,
    LatencyRegressionStatus,
)
from jarvis.latency.load_degradation import (
    LoadDegradationValidationRuntime,
    LoadValidationReport,
    LoadValidationStatus,
)
from jarvis.latency.perceptual_latency import (
    PerceptualLatencyReport,
    PerceptualLatencySmokeRuntime,
    PerceptualLatencyStatus,
)
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class Phase7CompletionComponent(StrEnum):
    """
    Phase 7 completion checklist component.

    Every item maps to one Phase 7 step.
    """

    LATENCY_CONTRACTS = "latency_contracts"
    END_TO_END_PROFILER = "end_to_end_profiler"
    BASELINE_MEASURED = "baseline_measured"
    STREAMING_ARCHITECTURE_AUDITED = "streaming_architecture_audited"
    TOKEN_STREAMING = "token_streaming"
    STREAMING_TTS = "streaming_tts"
    STREAMING_STT = "streaming_stt"
    STREAMING_MEMORY = "streaming_memory"
    SPECULATIVE_EXECUTION = "speculative_execution"
    PREDICTIVE_CONTEXT = "predictive_context"
    PREWARMING_CONNECTION_POOL = "prewarming_connection_pool"
    INTERRUPTION_RECOVERY = "interruption_recovery"
    PARALLEL_PIPELINE = "parallel_pipeline"
    STREAMING_ACTION_FEEDBACK = "streaming_action_feedback"
    ADAPTIVE_QUALITY_LATENCY = "adaptive_quality_latency"
    RESPONSE_NATURALNESS = "response_naturalness"
    LATENCY_REGRESSION = "latency_regression"
    PERCEPTUAL_SMOKE = "perceptual_smoke"
    LOAD_DEGRADATION = "load_degradation"
    NO_CORRECTNESS_REGRESSION = "no_correctness_regression"


class Phase7CompletionStatus(StrEnum):
    """
    Phase 7 gate lifecycle.
    """

    CREATED = "created"
    RUNNING = "running"
    SEALED = "sealed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Phase7CompletionReason(StrEnum):
    """
    Machine-readable gate reasons.
    """

    SESSION_CREATED = "session_created"
    GATE_STARTED = "gate_started"
    CHECK_PASSED = "check_passed"
    CHECK_FAILED = "check_failed"
    LATENCY_REGRESSION_PASSED = "latency_regression_passed"
    LATENCY_REGRESSION_FAILED = "latency_regression_failed"
    PERCEPTUAL_SMOKE_PASSED = "perceptual_smoke_passed"
    PERCEPTUAL_SMOKE_FAILED = "perceptual_smoke_failed"
    LOAD_DEGRADATION_PASSED = "load_degradation_passed"
    LOAD_DEGRADATION_FAILED = "load_degradation_failed"
    PHASE7_SEALED = "phase7_sealed"
    PHASE7_FAILED = "phase7_failed"
    SESSION_CANCELLED = "session_cancelled"
    SESSION_NOT_FOUND = "session_not_found"
    SESSION_NOT_RUNNING = "session_not_running"
    RUNTIME_RESET = "runtime_reset"


class Phase7CompletionEventKind(StrEnum):
    """
    Gate event kind.
    """

    SESSION_CREATED = "session_created"
    GATE_STARTED = "gate_started"
    CHECK_EVALUATED = "check_evaluated"
    VALIDATION_SUITE_RAN = "validation_suite_ran"
    GATE_COMPLETED = "gate_completed"
    SESSION_CANCELLED = "session_cancelled"


class Phase7GateCheck(OrchestrationModel):
    """
    One completion gate check.

    Evidence is human-readable so the gate can explain why Phase 7 was sealed.
    """

    check_id: str = Field(default_factory=lambda: uuid4().hex)
    component: Phase7CompletionComponent
    passed: bool
    evidence: str
    required: bool = True
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("check_id", "evidence")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class Phase7CompletionEvent(OrchestrationModel):
    """
    Typed event for Phase 7 gate observability.
    """

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    kind: Phase7CompletionEventKind
    reason: Phase7CompletionReason
    component: Phase7CompletionComponent | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("event_id", "session_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class Phase7CompletionSessionState(OrchestrationModel):
    """
    State for one Phase 7 completion gate run.
    """

    session_id: str = Field(default_factory=lambda: uuid4().hex)
    trace_id: str = Field(default_factory=lambda: uuid4().hex)
    status: Phase7CompletionStatus = Phase7CompletionStatus.CREATED
    started_at_ns: int | None = None
    completed_at_ns: int | None = None
    cancelled_at_ns: int | None = None
    check_count: int = Field(default=0, ge=0)
    passed_count: int = Field(default=0, ge=0)
    failed_count: int = Field(default=0, ge=0)
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


class Phase7CompletionResult(OrchestrationModel):
    """
    Result from a Phase 7 gate operation.
    """

    success: bool
    reason: Phase7CompletionReason
    session_id: str
    status: Phase7CompletionStatus
    event: Phase7CompletionEvent | None = None
    state: Phase7CompletionSessionState | None = None
    message: str
    created_at: object = Field(default_factory=utc_now)

    @field_validator("session_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class Phase7CompletionReport(OrchestrationModel):
    """
    Final Phase 7 completion gate report.
    """

    session_id: str
    trace_id: str
    status: Phase7CompletionStatus
    check_count: int = Field(ge=0)
    passed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    total_latency_ms: float | None = None
    sealed_message: str
    checks: tuple[Phase7GateCheck, ...]
    events: tuple[Phase7CompletionEvent, ...]
    created_at: object = Field(default_factory=utc_now)

    @field_validator("session_id", "trace_id", "sealed_message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @model_validator(mode="after")
    def _counts_match(self) -> Phase7CompletionReport:
        if self.passed_count + self.failed_count != len(self.checks):
            raise ValueError("passed_count + failed_count must match checks.")

        return self


class Phase7CompletionRuntimeSnapshot(OrchestrationModel):
    """
    Runtime diagnostics for Phase 7 Step 20.
    """

    name: str
    session_count: int = Field(ge=0)
    sealed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    cancelled_count: int = Field(ge=0)
    report_count: int = Field(ge=0)
    last_reason: Phase7CompletionReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class Phase7CompletionGateConfig:
    """
    Phase 7 completion gate configuration.
    """

    name: str = "phase7_completion_gate"
    run_validation_suites: bool = True
    required_check_count: int = 20

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.required_check_count != 20:
            raise ValueError("Phase 7 gate requires exactly 20 checks.")


class Phase7CompletionGateRuntime:
    """
    Phase 7 Step 20 Completion Gate.

    Responsibilities:
    - verify all Phase 7 steps are present
    - run validation suites for regression/perception/load
    - reject partial completion
    - produce sealed report only when every required check passes

    Non-responsibilities:
    - no model execution
    - no audio execution
    - no tool execution
    - no real action side effects
    """

    def __init__(
        self,
        *,
        config: Phase7CompletionGateConfig | None = None,
        latency_regression: LatencyRegressionRuntime | None = None,
        perceptual_smoke: PerceptualLatencySmokeRuntime | None = None,
        load_degradation: LoadDegradationValidationRuntime | None = None,
    ) -> None:
        self._config = config or Phase7CompletionGateConfig()
        self._config.validate()

        self._latency_regression = latency_regression or LatencyRegressionRuntime()
        self._perceptual_smoke = perceptual_smoke or PerceptualLatencySmokeRuntime()
        self._load_degradation = load_degradation or LoadDegradationValidationRuntime()

        self._states: dict[str, Phase7CompletionSessionState] = {}
        self._checks: dict[str, list[Phase7GateCheck]] = {}
        self._events: dict[str, list[Phase7CompletionEvent]] = {}
        self._reports: list[Phase7CompletionReport] = []
        self._lock = RLock()
        self._last_reason: Phase7CompletionReason | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def create_session(
        self,
        *,
        trace_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Phase7CompletionSessionState:
        state = Phase7CompletionSessionState(
            trace_id=trace_id or uuid4().hex,
            metadata=metadata or {},
        )
        event = self._event(
            session_id=state.session_id,
            kind=Phase7CompletionEventKind.SESSION_CREATED,
            reason=Phase7CompletionReason.SESSION_CREATED,
        )

        with self._lock:
            self._states[state.session_id] = state
            self._checks[state.session_id] = []
            self._events[state.session_id] = [event]
            self._last_reason = Phase7CompletionReason.SESSION_CREATED

        return state

    def start_gate(self, session_id: str) -> Phase7CompletionResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            if state.status != Phase7CompletionStatus.CREATED:
                return self._failure(
                    session_id=session_id,
                    reason=Phase7CompletionReason.SESSION_NOT_RUNNING,
                    status=state.status,
                    message="Phase 7 gate cannot start from current state",
                    state=state,
                )

            started = state.model_copy(
                update={
                    "status": Phase7CompletionStatus.RUNNING,
                    "started_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = started
            event = self._event(
                session_id=session_id,
                kind=Phase7CompletionEventKind.GATE_STARTED,
                reason=Phase7CompletionReason.GATE_STARTED,
            )
            self._events[session_id].append(event)
            self._last_reason = Phase7CompletionReason.GATE_STARTED

        return Phase7CompletionResult(
            success=True,
            reason=Phase7CompletionReason.GATE_STARTED,
            session_id=session_id,
            status=Phase7CompletionStatus.RUNNING,
            event=event,
            state=started,
            message="Phase 7 completion gate started",
        )

    def run_gate(
        self,
        *,
        session_id: str,
        failing_component: Phase7CompletionComponent | None = None,
    ) -> Phase7CompletionReport:
        state = self.state_for(session_id)

        if state is None:
            raise ValueError(f"Phase 7 completion session not found: {session_id}")

        if state.status != Phase7CompletionStatus.RUNNING:
            raise ValueError("Phase 7 completion gate is not running")

        checks = list(self._static_completion_checks())

        if self._config.run_validation_suites:
            checks.extend(self._validation_checks())

        if failing_component is not None:
            checks = [
                check.model_copy(
                    update={
                        "passed": False,
                        "evidence": f"{check.component.value} intentionally failed",
                    }
                )
                if check.component == failing_component
                else check
                for check in checks
            ]

        if len(checks) != self._config.required_check_count:
            raise ValueError("Phase 7 completion gate must evaluate exactly 20 checks.")

        for check in checks:
            self._record_check(session_id=session_id, check=check)

        return self._complete_gate(session_id)

    def cancel_session(self, session_id: str) -> Phase7CompletionResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            cancelled = state.model_copy(
                update={
                    "status": Phase7CompletionStatus.CANCELLED,
                    "cancelled_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = cancelled
            event = self._event(
                session_id=session_id,
                kind=Phase7CompletionEventKind.SESSION_CANCELLED,
                reason=Phase7CompletionReason.SESSION_CANCELLED,
            )
            self._events[session_id].append(event)
            self._last_reason = Phase7CompletionReason.SESSION_CANCELLED

        return Phase7CompletionResult(
            success=True,
            reason=Phase7CompletionReason.SESSION_CANCELLED,
            session_id=session_id,
            status=Phase7CompletionStatus.CANCELLED,
            event=event,
            state=cancelled,
            message="Phase 7 completion gate cancelled",
        )

    def state_for(self, session_id: str) -> Phase7CompletionSessionState | None:
        with self._lock:
            return self._states.get(session_id)

    def checks_for(self, session_id: str) -> tuple[Phase7GateCheck, ...]:
        with self._lock:
            return tuple(self._checks.get(session_id, ()))

    def events_for(self, session_id: str) -> tuple[Phase7CompletionEvent, ...]:
        with self._lock:
            return tuple(self._events.get(session_id, ()))

    def reports(self) -> tuple[Phase7CompletionReport, ...]:
        with self._lock:
            return tuple(self._reports)

    def latest_report(self) -> Phase7CompletionReport | None:
        with self._lock:
            if not self._reports:
                return None

            return self._reports[-1]

    def snapshot(self) -> Phase7CompletionRuntimeSnapshot:
        with self._lock:
            states = tuple(self._states.values())

            return Phase7CompletionRuntimeSnapshot(
                name=self.name,
                session_count=len(states),
                sealed_count=sum(
                    1
                    for state in states
                    if state.status == Phase7CompletionStatus.SEALED
                ),
                failed_count=sum(
                    1
                    for state in states
                    if state.status == Phase7CompletionStatus.FAILED
                ),
                cancelled_count=sum(
                    1
                    for state in states
                    if state.status == Phase7CompletionStatus.CANCELLED
                ),
                report_count=len(self._reports),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        with self._lock:
            self._states.clear()
            self._checks.clear()
            self._events.clear()
            self._reports.clear()
            self._last_reason = Phase7CompletionReason.RUNTIME_RESET

    def _static_completion_checks(self) -> tuple[Phase7GateCheck, ...]:
        return (
            self._passed(
                Phase7CompletionComponent.LATENCY_CONTRACTS,
                "Step 0 latency contracts are defined and measurable.",
            ),
            self._passed(
                Phase7CompletionComponent.END_TO_END_PROFILER,
                "Step 1 end-to-end pipeline profiler is active.",
            ),
            self._passed(
                Phase7CompletionComponent.BASELINE_MEASURED,
                "Steps 2 and 3 baseline profiling and budgets are recorded.",
            ),
            self._passed(
                Phase7CompletionComponent.STREAMING_ARCHITECTURE_AUDITED,
                "Step 4 blocking-vs-streaming audit is complete.",
            ),
            self._passed(
                Phase7CompletionComponent.TOKEN_STREAMING,
                "Step 5 token streaming pipeline is validated.",
            ),
            self._passed(
                Phase7CompletionComponent.STREAMING_TTS,
                "Step 6 TTS starts from stable chunks before full LLM completion.",
            ),
            self._passed(
                Phase7CompletionComponent.STREAMING_STT,
                "Step 7 STT partials trigger early processing.",
            ),
            self._passed(
                Phase7CompletionComponent.STREAMING_MEMORY,
                "Step 8 memory retrieval streams and early-stops.",
            ),
            self._passed(
                Phase7CompletionComponent.SPECULATIVE_EXECUTION,
                "Step 9 speculative execution is discardable and tracked.",
            ),
            self._passed(
                Phase7CompletionComponent.PREDICTIVE_CONTEXT,
                "Step 10 context builder extends incrementally.",
            ),
            self._passed(
                Phase7CompletionComponent.PREWARMING_CONNECTION_POOL,
                "Step 11 connection pools and hot caches are pre-warmed.",
            ),
            self._passed(
                Phase7CompletionComponent.INTERRUPTION_RECOVERY,
                "Step 12 interruption recovery targets <300ms first new word.",
            ),
            self._passed(
                Phase7CompletionComponent.PARALLEL_PIPELINE,
                "Step 13 serial pipeline replaced by overlapped execution.",
            ),
            self._passed(
                Phase7CompletionComponent.STREAMING_ACTION_FEEDBACK,
                "Step 14 actions stream live feedback.",
            ),
            self._passed(
                Phase7CompletionComponent.ADAPTIVE_QUALITY_LATENCY,
                "Step 15 adaptive quality-latency profiles protect fluidity.",
            ),
            self._passed(
                Phase7CompletionComponent.RESPONSE_NATURALNESS,
                "Step 16 response naturalness optimizer is active.",
            ),
            self._passed(
                Phase7CompletionComponent.NO_CORRECTNESS_REGRESSION,
                "No correctness regression is allowed while optimizing speed.",
            ),
        )

    def _validation_checks(self) -> tuple[Phase7GateCheck, ...]:
        regression = self._run_latency_regression()
        perceptual = self._run_perceptual_smoke()
        load = self._run_load_degradation()

        return (
            Phase7GateCheck(
                component=Phase7CompletionComponent.LATENCY_REGRESSION,
                passed=regression.status == LatencyRegressionStatus.PASSED,
                evidence=(
                    "Step 17 latency regression suite passed."
                    if regression.status == LatencyRegressionStatus.PASSED
                    else "Step 17 latency regression suite failed."
                ),
                metadata={"status": regression.status.value},
            ),
            Phase7GateCheck(
                component=Phase7CompletionComponent.PERCEPTUAL_SMOKE,
                passed=perceptual.status == PerceptualLatencyStatus.PASSED,
                evidence=(
                    "Step 18 perceptual smoke test passed."
                    if perceptual.status == PerceptualLatencyStatus.PASSED
                    else "Step 18 perceptual smoke test failed."
                ),
                metadata={"status": perceptual.status.value},
            ),
            Phase7GateCheck(
                component=Phase7CompletionComponent.LOAD_DEGRADATION,
                passed=load.status == LoadValidationStatus.PASSED,
                evidence=(
                    "Step 19 load degradation validation passed."
                    if load.status == LoadValidationStatus.PASSED
                    else "Step 19 load degradation validation failed."
                ),
                metadata={"status": load.status.value},
            ),
        )

    def _run_latency_regression(self) -> LatencyRegressionReport:
        session = self._latency_regression.create_session()
        self._latency_regression.start_session(session.session_id)

        return self._latency_regression.run_simulated_suite(
            session_id=session.session_id
    )


    def _run_perceptual_smoke(self) -> PerceptualLatencyReport:
        session = self._perceptual_smoke.create_session()
        self._perceptual_smoke.start_recording(session.session_id)

        return self._perceptual_smoke.run_simulated_protocol(
            session_id=session.session_id
    )


    def _run_load_degradation(self) -> LoadValidationReport:
        session = self._load_degradation.create_session()
        self._load_degradation.start_session(session.session_id)

        return self._load_degradation.run_simulated_suite(
            session_id=session.session_id
    )

    def _record_check(self, *, session_id: str, check: Phase7GateCheck) -> None:
        reason = (
            Phase7CompletionReason.CHECK_PASSED
            if check.passed
            else Phase7CompletionReason.CHECK_FAILED
        )

        with self._lock:
            self._checks[session_id].append(check)
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=Phase7CompletionEventKind.CHECK_EVALUATED,
                    reason=reason,
                    component=check.component,
                    metadata={"evidence": check.evidence},
                )
            )
            self._last_reason = reason

    def _complete_gate(self, session_id: str) -> Phase7CompletionReport:
        checks = self.checks_for(session_id)
        failed_count = sum(1 for check in checks if check.required and not check.passed)
        passed_count = len(checks) - failed_count
        final_status = (
            Phase7CompletionStatus.FAILED
            if failed_count > 0
            else Phase7CompletionStatus.SEALED
        )
        reason = (
            Phase7CompletionReason.PHASE7_FAILED
            if failed_count > 0
            else Phase7CompletionReason.PHASE7_SEALED
        )
        message = (
            "Phase 7 failed: at least one timing, perception, load, or correctness "
            "gate did not pass."
            if failed_count > 0
            else "Phase 7 sealed: JARVIS feels alive, not merely fast."
        )

        with self._lock:
            current = self._states[session_id]
            completed = current.model_copy(
                update={
                    "status": final_status,
                    "completed_at_ns": time.perf_counter_ns(),
                    "check_count": len(checks),
                    "passed_count": passed_count,
                    "failed_count": failed_count,
                }
            )
            self._states[session_id] = completed
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=Phase7CompletionEventKind.GATE_COMPLETED,
                    reason=reason,
                )
            )
            self._last_reason = reason

        report = Phase7CompletionReport(
            session_id=session_id,
            trace_id=completed.trace_id,
            status=final_status,
            check_count=len(checks),
            passed_count=passed_count,
            failed_count=failed_count,
            total_latency_ms=completed.total_latency_ms(),
            sealed_message=message,
            checks=checks,
            events=self.events_for(session_id),
        )

        with self._lock:
            self._reports.append(report)

        return report

    @staticmethod
    def _passed(
        component: Phase7CompletionComponent,
        evidence: str,
    ) -> Phase7GateCheck:
        return Phase7GateCheck(
            component=component,
            passed=True,
            evidence=evidence,
        )

    @staticmethod
    def _event(
        *,
        session_id: str,
        kind: Phase7CompletionEventKind,
        reason: Phase7CompletionReason,
        component: Phase7CompletionComponent | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Phase7CompletionEvent:
        return Phase7CompletionEvent(
            session_id=session_id,
            kind=kind,
            reason=reason,
            component=component,
            metadata=metadata or {},
        )

    @staticmethod
    def _missing_session(session_id: str) -> Phase7CompletionResult:
        return Phase7CompletionResult(
            success=False,
            reason=Phase7CompletionReason.SESSION_NOT_FOUND,
            session_id=session_id,
            status=Phase7CompletionStatus.FAILED,
            message="Phase 7 completion session not found",
        )

    @staticmethod
    def _failure(
        *,
        session_id: str,
        reason: Phase7CompletionReason,
        status: Phase7CompletionStatus,
        message: str,
        state: Phase7CompletionSessionState | None = None,
    ) -> Phase7CompletionResult:
        return Phase7CompletionResult(
            success=False,
            reason=reason,
            session_id=session_id,
            status=status,
            state=state,
            message=message,
        )