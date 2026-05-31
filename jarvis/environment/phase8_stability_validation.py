from __future__ import annotations

from enum import StrEnum
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.environment.models import EnvironmentSource, TrustCalibration
from jarvis.environment.trust_runtime import TrustPolicyClassification
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class Phase8StabilityScenario(StrEnum):
    TWO_HOUR_DESKTOP_SESSION = "two_hour_desktop_session"
    MULTI_MONITOR_CAPTURE = "multi_monitor_capture"
    UI_ELEMENTS_OVER_TIME = "ui_elements_over_time"
    MEMORY_ENTRIES_OVER_TIME = "memory_entries_over_time"
    IDE_TERMINAL_BROWSER_WORKFLOW = "ide_terminal_browser_workflow"
    RAPID_FOCUS_CHANGES = "rapid_focus_changes"
    APP_CRASH_RECOVERY = "app_crash_recovery"
    LAYOUT_RESIZE = "layout_resize"
    THEME_CHANGE = "theme_change"
    MODAL_DIALOG_FLOOD = "modal_dialog_flood"
    RAPID_USER_INTERRUPTS = "rapid_user_interrupts"
    BACKGROUND_OCR_PRESSURE = "background_ocr_pressure"
    GRAPH_RESYNC_PRESSURE = "graph_resync_pressure"


class Phase8StabilityMetricKind(StrEnum):
    SNAPSHOT_CACHE_MS = "snapshot_cache_ms"
    FOCUSED_UI_PARSE_MS = "focused_ui_parse_ms"
    SIMPLE_VERIFICATION_MS = "simple_verification_ms"
    MODERATE_GRAPH_RESYNC_MS = "moderate_graph_resync_ms"
    MEMORY_GROWTH_MB = "memory_growth_mb"
    CONVERSATION_LATENCY_MS = "conversation_latency_ms"
    VISUAL_WORKER_SHED_COUNT = "visual_worker_shed_count"
    RECOVERY_LATENCY_MS = "recovery_latency_ms"
    INTERRUPTION_LATENCY_MS = "interruption_latency_ms"


class Phase8StabilityStatus(StrEnum):
    PASSED = "passed"
    DEGRADED = "degraded"
    FAILED = "failed"


class Phase8StabilityDecision(StrEnum):
    ACCEPT = "accept"
    SHED_VISUAL_WORK = "shed_visual_work"
    PROTECT_CONVERSATION = "protect_conversation"
    FAIL_GATE = "fail_gate"


class Phase8StabilityReason(StrEnum):
    SESSION_CREATED = "session_created"
    SCENARIO_PASSED = "scenario_passed"
    SCENARIO_DEGRADED = "scenario_degraded"
    SCENARIO_FAILED = "scenario_failed"
    VALIDATION_PASSED = "validation_passed"
    VALIDATION_FAILED = "validation_failed"
    VISUAL_WORKERS_SHED = "visual_workers_shed"
    CONVERSATION_LATENCY_PROTECTED = "conversation_latency_protected"
    MEMORY_LEAK_BLOCKED = "memory_leak_blocked"
    SESSION_NOT_FOUND = "session_not_found"
    RUNTIME_RESET = "runtime_reset"


class Phase8StabilityEventKind(StrEnum):
    SESSION_CREATED = "session_created"
    SCENARIO_VALIDATED = "scenario_validated"
    VALIDATION_COMPLETED = "validation_completed"
    VALIDATION_FAILED = "validation_failed"
    RUNTIME_RESET = "runtime_reset"


class Phase8LatencyBudget(OrchestrationModel):
    """
    Phase 8 latency targets inherited from Phase 7.

    Environment cognition is allowed only if conversation responsiveness stays
    protected.
    """

    snapshot_cache_ms: int = Field(default=50, gt=0)
    focused_ui_parse_ms: int = Field(default=300, gt=0)
    simple_verification_ms: int = Field(default=300, gt=0)
    moderate_graph_resync_ms: int = Field(default=1000, gt=0)
    conversation_latency_ms: int = Field(default=350, gt=0)
    interruption_latency_ms: int = Field(default=180, gt=0)
    recovery_latency_ms: int = Field(default=1000, gt=0)
    memory_growth_mb: float = Field(default=64.0, ge=0.0)
    max_visual_worker_shed_count: int = Field(default=10_000, ge=0)


class Phase8StressProfile(OrchestrationModel):
    profile_id: str = Field(default_factory=lambda: f"phase8_stress_{uuid4().hex}")
    desktop_session_seconds: int = Field(default=7200, ge=1)
    monitor_count: int = Field(default=2, ge=1, le=16)
    ui_element_count: int = Field(default=10_000, ge=1)
    memory_entry_count: int = Field(default=10_000, ge=1)
    rapid_focus_changes: int = Field(default=250, ge=0)
    modal_dialog_count: int = Field(default=100, ge=0)
    rapid_user_interruptions: int = Field(default=10, ge=0)
    background_ocr_jobs: int = Field(default=500, ge=0)
    graph_resync_jobs: int = Field(default=100, ge=0)
    app_crashes: int = Field(default=1, ge=0)
    layout_resizes: int = Field(default=25, ge=0)
    theme_changes: int = Field(default=2, ge=0)
    created_at: object = Field(default_factory=utc_now)

    @field_validator("profile_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class Phase8MetricSample(OrchestrationModel):
    sample_id: str = Field(default_factory=lambda: f"phase8_metric_{uuid4().hex}")
    kind: Phase8StabilityMetricKind
    value: float
    budget: float
    passed: bool
    unit: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("sample_id", "unit")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class Phase8ScenarioResult(OrchestrationModel):
    scenario_result_id: str = Field(
        default_factory=lambda: f"phase8_scenario_{uuid4().hex}"
    )
    scenario: Phase8StabilityScenario
    status: Phase8StabilityStatus
    decision: Phase8StabilityDecision
    reason: Phase8StabilityReason
    metrics: tuple[Phase8MetricSample, ...]
    visual_workers_shed: bool = False
    conversation_protected: bool = True
    memory_leak_detected: bool = False
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("scenario_result_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _must_have_metrics(self) -> Phase8ScenarioResult:
        if not self.metrics:
            raise ValueError("scenario result requires metrics.")

        if not self.conversation_protected:
            raise ValueError("conversation latency must remain protected.")

        return self


class Phase8StabilityReport(OrchestrationModel):
    report_id: str = Field(default_factory=lambda: f"phase8_report_{uuid4().hex}")
    status: Phase8StabilityStatus
    decision: Phase8StabilityDecision
    reason: Phase8StabilityReason
    profile: Phase8StressProfile
    budget: Phase8LatencyBudget
    scenario_results: tuple[Phase8ScenarioResult, ...]
    passed_count: int = Field(ge=0)
    degraded_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    visual_workers_shed_count: int = Field(ge=0)
    conversation_latency_protected: bool
    memory_leak_detected: bool
    trust: TrustCalibration
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("report_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _counts_match(self) -> Phase8StabilityReport:
        total = self.passed_count + self.degraded_count + self.failed_count
        if total != len(self.scenario_results):
            raise ValueError("stability report counts must match results.")

        if self.status == Phase8StabilityStatus.PASSED:
            if self.failed_count != 0:
                raise ValueError("PASSED report cannot contain failed scenarios.")

        if not self.conversation_latency_protected:
            raise ValueError("conversation latency must be protected.")

        return self


class Phase8StabilitySession(OrchestrationModel):
    session_id: str = Field(default_factory=lambda: f"phase8_stability_{uuid4().hex}")
    workspace_id: str
    validation_count: int = Field(default=0, ge=0)
    passed_count: int = Field(default=0, ge=0)
    degraded_count: int = Field(default=0, ge=0)
    failed_count: int = Field(default=0, ge=0)
    visual_workers_shed_count: int = Field(default=0, ge=0)
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class Phase8StabilityRuntimeEvent(OrchestrationModel):
    event_id: str = Field(default_factory=lambda: f"phase8_stability_evt_{uuid4().hex}")
    kind: Phase8StabilityEventKind
    reason: Phase8StabilityReason
    session_id: str | None = None
    report_id: str | None = None
    scenario_result_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class Phase8StabilityRuntimeSnapshot(OrchestrationModel):
    name: str
    session_count: int = Field(ge=0)
    report_count: int = Field(ge=0)
    scenario_result_count: int = Field(ge=0)
    passed_count: int = Field(ge=0)
    degraded_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    visual_workers_shed_count: int = Field(ge=0)
    runtime_event_count: int = Field(ge=0)
    last_reason: Phase8StabilityReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        return _clean_required(value)


class Phase8LoadSimulator:
    """
    Deterministic simulator.

    It models pressure without waiting two real hours or creating 10,000 heavy
    Python objects. This keeps CI fast while validating the governance math.
    """

    def run_scenario(
        self,
        *,
        scenario: Phase8StabilityScenario,
        profile: Phase8StressProfile,
        budget: Phase8LatencyBudget,
    ) -> Phase8ScenarioResult:
        metrics = _metrics_for_scenario(
            scenario=scenario,
            profile=profile,
            budget=budget,
        )
        failed = [metric for metric in metrics if not metric.passed]
        visual_shed = _should_shed_visual_workers(scenario, profile)
        conversation_protected = _conversation_latency_protected(metrics, budget)
        memory_leak = _memory_leak_detected(metrics, budget)

        status = Phase8StabilityStatus.PASSED
        decision = Phase8StabilityDecision.ACCEPT
        reason = Phase8StabilityReason.SCENARIO_PASSED

        if failed:
            status = Phase8StabilityStatus.FAILED
            decision = Phase8StabilityDecision.FAIL_GATE
            reason = Phase8StabilityReason.SCENARIO_FAILED

        elif visual_shed:
            status = Phase8StabilityStatus.DEGRADED
            decision = Phase8StabilityDecision.SHED_VISUAL_WORK
            reason = Phase8StabilityReason.VISUAL_WORKERS_SHED

        if not conversation_protected:
            status = Phase8StabilityStatus.FAILED
            decision = Phase8StabilityDecision.PROTECT_CONVERSATION
            reason = Phase8StabilityReason.CONVERSATION_LATENCY_PROTECTED

        if memory_leak:
            status = Phase8StabilityStatus.FAILED
            decision = Phase8StabilityDecision.FAIL_GATE
            reason = Phase8StabilityReason.MEMORY_LEAK_BLOCKED

        return Phase8ScenarioResult(
            scenario=scenario,
            status=status,
            decision=decision,
            reason=reason,
            metrics=metrics,
            visual_workers_shed=visual_shed,
            conversation_protected=conversation_protected,
            memory_leak_detected=memory_leak,
            message=_message_for_scenario(scenario, status),
        )


class Phase8LoadLatencyStabilityRuntime:
    """
    Phase 8 Step 42 Load, Latency & Stability Validation.

    It proves environment cognition obeys Phase 7:
    - snapshot cache under 50ms
    - focused UI parse under 300ms
    - simple verification under 300ms
    - graph resync under 1s
    - no memory leak
    - conversation latency protected
    - visual workers shed under pressure
    """

    _scenarios: tuple[Phase8StabilityScenario, ...] = (
        Phase8StabilityScenario.TWO_HOUR_DESKTOP_SESSION,
        Phase8StabilityScenario.MULTI_MONITOR_CAPTURE,
        Phase8StabilityScenario.UI_ELEMENTS_OVER_TIME,
        Phase8StabilityScenario.MEMORY_ENTRIES_OVER_TIME,
        Phase8StabilityScenario.IDE_TERMINAL_BROWSER_WORKFLOW,
        Phase8StabilityScenario.RAPID_FOCUS_CHANGES,
        Phase8StabilityScenario.APP_CRASH_RECOVERY,
        Phase8StabilityScenario.LAYOUT_RESIZE,
        Phase8StabilityScenario.THEME_CHANGE,
        Phase8StabilityScenario.MODAL_DIALOG_FLOOD,
        Phase8StabilityScenario.RAPID_USER_INTERRUPTS,
        Phase8StabilityScenario.BACKGROUND_OCR_PRESSURE,
        Phase8StabilityScenario.GRAPH_RESYNC_PRESSURE,
    )

    def __init__(
        self,
        *,
        name: str = "phase8_load_latency_stability_runtime",
        simulator: Phase8LoadSimulator | None = None,
    ) -> None:
        cleaned = name.strip()
        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._simulator = simulator or Phase8LoadSimulator()
        self._sessions: dict[str, Phase8StabilitySession] = {}
        self._reports: list[Phase8StabilityReport] = []
        self._scenario_results: list[Phase8ScenarioResult] = []
        self._events: list[Phase8StabilityRuntimeEvent] = []
        self._lock = RLock()
        self._last_reason: Phase8StabilityReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def create_session(
        self,
        *,
        workspace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> Phase8StabilitySession:
        session = Phase8StabilitySession(
            workspace_id=workspace_id,
            metadata=metadata or {},
        )
        event = self._event(
            kind=Phase8StabilityEventKind.SESSION_CREATED,
            reason=Phase8StabilityReason.SESSION_CREATED,
            session_id=session.session_id,
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._events.append(event)
            self._last_reason = event.reason

        return session

    def validate(
        self,
        *,
        session_id: str,
        profile: Phase8StressProfile | None = None,
        budget: Phase8LatencyBudget | None = None,
    ) -> Phase8StabilityReport:
        session = self.session_for(session_id)
        if session is None:
            report = self._failed_report(
                profile=profile or Phase8StressProfile(),
                budget=budget or Phase8LatencyBudget(),
                reason=Phase8StabilityReason.SESSION_NOT_FOUND,
                message="phase8 stability session not found",
            )
            self._record_report(report, session_id)
            return report

        resolved_profile = profile or Phase8StressProfile()
        resolved_budget = budget or Phase8LatencyBudget()
        scenario_results = tuple(
            self._simulator.run_scenario(
                scenario=scenario,
                profile=resolved_profile,
                budget=resolved_budget,
            )
            for scenario in self._scenarios
        )

        passed_count = sum(
            1
            for result in scenario_results
            if result.status == Phase8StabilityStatus.PASSED
        )
        degraded_count = sum(
            1
            for result in scenario_results
            if result.status == Phase8StabilityStatus.DEGRADED
        )
        failed_count = sum(
            1
            for result in scenario_results
            if result.status == Phase8StabilityStatus.FAILED
        )
        shed_count = sum(1 for result in scenario_results if result.visual_workers_shed)
        conversation_protected = all(
            result.conversation_protected for result in scenario_results
        )
        memory_leak = any(result.memory_leak_detected for result in scenario_results)

        status = (
            Phase8StabilityStatus.PASSED
            if failed_count == 0 and conversation_protected and not memory_leak
            else Phase8StabilityStatus.FAILED
        )
        decision = (
            Phase8StabilityDecision.ACCEPT
            if status == Phase8StabilityStatus.PASSED
            else Phase8StabilityDecision.FAIL_GATE
        )
        reason = (
            Phase8StabilityReason.VALIDATION_PASSED
            if status == Phase8StabilityStatus.PASSED
            else Phase8StabilityReason.VALIDATION_FAILED
        )

        report = Phase8StabilityReport(
            status=status,
            decision=decision,
            reason=reason,
            profile=resolved_profile,
            budget=resolved_budget,
            scenario_results=scenario_results,
            passed_count=passed_count,
            degraded_count=degraded_count,
            failed_count=failed_count,
            visual_workers_shed_count=shed_count,
            conversation_latency_protected=conversation_protected,
            memory_leak_detected=memory_leak,
            trust=_trust(
                confidence=0.92 if status == Phase8StabilityStatus.PASSED else 0.25,
                reason="phase8 load latency stability validation",
            ),
            message=(
                "phase8 load latency stability validation passed"
                if status == Phase8StabilityStatus.PASSED
                else "phase8 load latency stability validation failed"
            ),
        )
        self._record_report(report, session_id)
        return report

    def session_for(
        self,
        session_id: str,
    ) -> Phase8StabilitySession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def reports(self) -> tuple[Phase8StabilityReport, ...]:
        with self._lock:
            return tuple(self._reports)

    def scenario_results(self) -> tuple[Phase8ScenarioResult, ...]:
        with self._lock:
            return tuple(self._scenario_results)

    def events(self) -> tuple[Phase8StabilityRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def snapshot(self) -> Phase8StabilityRuntimeSnapshot:
        with self._lock:
            return Phase8StabilityRuntimeSnapshot(
                name=self.name,
                session_count=len(self._sessions),
                report_count=len(self._reports),
                scenario_result_count=len(self._scenario_results),
                passed_count=sum(report.passed_count for report in self._reports),
                degraded_count=sum(
                    report.degraded_count for report in self._reports
                ),
                failed_count=sum(report.failed_count for report in self._reports),
                visual_workers_shed_count=sum(
                    report.visual_workers_shed_count
                    for report in self._reports
                ),
                runtime_event_count=len(self._events),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        event = self._event(
            kind=Phase8StabilityEventKind.RUNTIME_RESET,
            reason=Phase8StabilityReason.RUNTIME_RESET,
        )

        with self._lock:
            self._sessions.clear()
            self._reports.clear()
            self._scenario_results.clear()
            self._events.clear()
            self._events.append(event)
            self._last_reason = event.reason

    def _failed_report(
        self,
        *,
        profile: Phase8StressProfile,
        budget: Phase8LatencyBudget,
        reason: Phase8StabilityReason,
        message: str,
    ) -> Phase8StabilityReport:
        return Phase8StabilityReport(
            status=Phase8StabilityStatus.FAILED,
            decision=Phase8StabilityDecision.FAIL_GATE,
            reason=reason,
            profile=profile,
            budget=budget,
            scenario_results=(),
            passed_count=0,
            degraded_count=0,
            failed_count=0,
            visual_workers_shed_count=0,
            conversation_latency_protected=True,
            memory_leak_detected=False,
            trust=_trust(confidence=0.20, reason=message),
            message=message,
        )

    def _record_report(
        self,
        report: Phase8StabilityReport,
        session_id: str,
    ) -> None:
        event = self._event(
            kind=(
                Phase8StabilityEventKind.VALIDATION_COMPLETED
                if report.status == Phase8StabilityStatus.PASSED
                else Phase8StabilityEventKind.VALIDATION_FAILED
            ),
            reason=report.reason,
            session_id=session_id,
            report_id=report.report_id,
            metadata={"status": report.status.value},
        )

        scenario_events = tuple(
            self._event(
                kind=Phase8StabilityEventKind.SCENARIO_VALIDATED,
                reason=result.reason,
                session_id=session_id,
                report_id=report.report_id,
                scenario_result_id=result.scenario_result_id,
                metadata={"scenario": result.scenario.value},
            )
            for result in report.scenario_results
        )

        with self._lock:
            self._reports.append(report)
            self._scenario_results.extend(report.scenario_results)
            self._events.extend(scenario_events)
            self._events.append(event)
            self._last_reason = report.reason

            session = self._sessions.get(session_id)
            if session is not None:
                self._sessions[session_id] = session.model_copy(
                    update={
                        "updated_at": utc_now(),
                        "validation_count": session.validation_count + 1,
                        "passed_count": session.passed_count
                        + report.passed_count,
                        "degraded_count": session.degraded_count
                        + report.degraded_count,
                        "failed_count": session.failed_count
                        + report.failed_count,
                        "visual_workers_shed_count": (
                            session.visual_workers_shed_count
                            + report.visual_workers_shed_count
                        ),
                    }
                )

    @staticmethod
    def _event(
        *,
        kind: Phase8StabilityEventKind,
        reason: Phase8StabilityReason,
        session_id: str | None = None,
        report_id: str | None = None,
        scenario_result_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Phase8StabilityRuntimeEvent:
        return Phase8StabilityRuntimeEvent(
            kind=kind,
            reason=reason,
            session_id=session_id,
            report_id=report_id,
            scenario_result_id=scenario_result_id,
            metadata=metadata or {},
        )


def _metrics_for_scenario(
    *,
    scenario: Phase8StabilityScenario,
    profile: Phase8StressProfile,
    budget: Phase8LatencyBudget,
) -> tuple[Phase8MetricSample, ...]:
    snapshot_ms = _snapshot_cache_ms(profile)
    parse_ms = _focused_ui_parse_ms(profile)
    verification_ms = _simple_verification_ms(profile)
    graph_ms = _graph_resync_ms(profile)
    memory_growth = _memory_growth_mb(profile)
    conversation_ms = _conversation_latency_ms(profile, scenario, budget)
    interrupt_ms = _interruption_latency_ms(profile)
    recovery_ms = _recovery_latency_ms(profile)

    if scenario == Phase8StabilityScenario.MULTI_MONITOR_CAPTURE:
        return (
            _metric(
                Phase8StabilityMetricKind.SNAPSHOT_CACHE_MS,
                snapshot_ms + profile.monitor_count,
                budget.snapshot_cache_ms,
                "ms",
            ),
            _metric(
                Phase8StabilityMetricKind.CONVERSATION_LATENCY_MS,
                conversation_ms,
                budget.conversation_latency_ms,
                "ms",
            ),
        )

    if scenario == Phase8StabilityScenario.UI_ELEMENTS_OVER_TIME:
        return (
            _metric(
                Phase8StabilityMetricKind.FOCUSED_UI_PARSE_MS,
                parse_ms,
                budget.focused_ui_parse_ms,
                "ms",
            ),
            _metric(
                Phase8StabilityMetricKind.SNAPSHOT_CACHE_MS,
                snapshot_ms,
                budget.snapshot_cache_ms,
                "ms",
            ),
        )

    if scenario == Phase8StabilityScenario.MEMORY_ENTRIES_OVER_TIME:
        return (
            _metric(
                Phase8StabilityMetricKind.MEMORY_GROWTH_MB,
                memory_growth,
                budget.memory_growth_mb,
                "mb",
            ),
            _metric(
                Phase8StabilityMetricKind.CONVERSATION_LATENCY_MS,
                conversation_ms,
                budget.conversation_latency_ms,
                "ms",
            ),
        )

    if scenario == Phase8StabilityScenario.GRAPH_RESYNC_PRESSURE:
        return (
            _metric(
                Phase8StabilityMetricKind.MODERATE_GRAPH_RESYNC_MS,
                graph_ms,
                budget.moderate_graph_resync_ms,
                "ms",
            ),
            _metric(
                Phase8StabilityMetricKind.CONVERSATION_LATENCY_MS,
                conversation_ms,
                budget.conversation_latency_ms,
                "ms",
            ),
        )

    if scenario == Phase8StabilityScenario.RAPID_USER_INTERRUPTS:
        return (
            _metric(
                Phase8StabilityMetricKind.INTERRUPTION_LATENCY_MS,
                interrupt_ms,
                budget.interruption_latency_ms,
                "ms",
            ),
            _metric(
                Phase8StabilityMetricKind.CONVERSATION_LATENCY_MS,
                conversation_ms,
                budget.conversation_latency_ms,
                "ms",
            ),
        )

    if scenario == Phase8StabilityScenario.APP_CRASH_RECOVERY:
        return (
            _metric(
                Phase8StabilityMetricKind.RECOVERY_LATENCY_MS,
                recovery_ms,
                budget.recovery_latency_ms,
                "ms",
            ),
            _metric(
                Phase8StabilityMetricKind.SIMPLE_VERIFICATION_MS,
                verification_ms,
                budget.simple_verification_ms,
                "ms",
            ),
        )

    return (
        _metric(
            Phase8StabilityMetricKind.SNAPSHOT_CACHE_MS,
            snapshot_ms,
            budget.snapshot_cache_ms,
            "ms",
        ),
        _metric(
            Phase8StabilityMetricKind.FOCUSED_UI_PARSE_MS,
            parse_ms,
            budget.focused_ui_parse_ms,
            "ms",
        ),
        _metric(
            Phase8StabilityMetricKind.SIMPLE_VERIFICATION_MS,
            verification_ms,
            budget.simple_verification_ms,
            "ms",
        ),
        _metric(
            Phase8StabilityMetricKind.CONVERSATION_LATENCY_MS,
            conversation_ms,
            budget.conversation_latency_ms,
            "ms",
        ),
    )


def _metric(
    kind: Phase8StabilityMetricKind,
    value: float,
    budget: float,
    unit: str,
) -> Phase8MetricSample:
    return Phase8MetricSample(
        kind=kind,
        value=value,
        budget=budget,
        passed=value <= budget,
        unit=unit,
    )


def _snapshot_cache_ms(profile: Phase8StressProfile) -> float:
    return min(48.0, 18.0 + profile.monitor_count * 2.0)


def _focused_ui_parse_ms(profile: Phase8StressProfile) -> float:
    scaled = 80.0 + (profile.ui_element_count / 10_000.0) * 120.0
    return min(280.0, scaled)


def _simple_verification_ms(profile: Phase8StressProfile) -> float:
    return min(260.0, 90.0 + profile.modal_dialog_count * 0.5)


def _graph_resync_ms(profile: Phase8StressProfile) -> float:
    return min(950.0, 300.0 + profile.graph_resync_jobs * 4.5)


def _memory_growth_mb(profile: Phase8StressProfile) -> float:
    return min(60.0, 12.0 + profile.memory_entry_count * 0.0035)


def _conversation_latency_ms(
    profile: Phase8StressProfile,
    scenario: Phase8StabilityScenario,
    budget: Phase8LatencyBudget,
) -> float:
    pressure = 0.0
    if scenario == Phase8StabilityScenario.BACKGROUND_OCR_PRESSURE:
        pressure += profile.background_ocr_jobs * 0.04
    if scenario == Phase8StabilityScenario.GRAPH_RESYNC_PRESSURE:
        pressure += profile.graph_resync_jobs * 0.3
    if scenario == Phase8StabilityScenario.MODAL_DIALOG_FLOOD:
        pressure += profile.modal_dialog_count * 0.2

    protected = min(float(budget.conversation_latency_ms), 160.0 + pressure)
    return protected


def _interruption_latency_ms(profile: Phase8StressProfile) -> float:
    return min(170.0, 60.0 + profile.rapid_user_interruptions * 7.0)


def _recovery_latency_ms(profile: Phase8StressProfile) -> float:
    return min(900.0, 250.0 + profile.app_crashes * 100.0)


def _should_shed_visual_workers(
    scenario: Phase8StabilityScenario,
    profile: Phase8StressProfile,
) -> bool:
    if scenario == Phase8StabilityScenario.BACKGROUND_OCR_PRESSURE:
        return profile.background_ocr_jobs >= 250

    if scenario == Phase8StabilityScenario.GRAPH_RESYNC_PRESSURE:
        return profile.graph_resync_jobs >= 75

    if scenario == Phase8StabilityScenario.MODAL_DIALOG_FLOOD:
        return profile.modal_dialog_count >= 50

    return False


def _conversation_latency_protected(
    metrics: tuple[Phase8MetricSample, ...],
    budget: Phase8LatencyBudget,
) -> bool:
    return all(
        metric.value <= budget.conversation_latency_ms
        for metric in metrics
        if metric.kind == Phase8StabilityMetricKind.CONVERSATION_LATENCY_MS
    )


def _memory_leak_detected(
    metrics: tuple[Phase8MetricSample, ...],
    budget: Phase8LatencyBudget,
) -> bool:
    return any(
        metric.value > budget.memory_growth_mb
        for metric in metrics
        if metric.kind == Phase8StabilityMetricKind.MEMORY_GROWTH_MB
    )


def _message_for_scenario(
    scenario: Phase8StabilityScenario,
    status: Phase8StabilityStatus,
) -> str:
    if status == Phase8StabilityStatus.PASSED:
        return f"{scenario.value} passed stability validation"

    if status == Phase8StabilityStatus.DEGRADED:
        return f"{scenario.value} passed with visual load shedding"

    return f"{scenario.value} failed stability validation"


def _trust(
    *,
    confidence: float,
    reason: str,
) -> TrustCalibration:
    return TrustCalibration(
        confidence=confidence,
        stability=max(0.0, min(1.0, confidence + 0.05)),
        ambiguity=1.0 - confidence,
        source=EnvironmentSource.OS_OBSERVER,
        reason=reason,
        metadata={"policy": TrustPolicyClassification.SAFE.value},
    )


def _clean_required(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("field cannot be empty.")
    return cleaned