from __future__ import annotations

from enum import StrEnum
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.environment.models import EnvironmentSource, TrustCalibration
from jarvis.environment.workspace_graph import (
    GraphDelta,
    GraphDeltaKind,
    GraphNodeKind,
    WorkspaceGraphRuntime,
    graph_node,
)
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class DivergenceCause(StrEnum):
    USER_ACTION = "user_action"
    APP_STATE_CHANGE = "app_state_change"
    OS_NOTIFICATION = "os_notification"
    POPUP_APPEARED = "popup_appeared"
    LAYOUT_CHANGED = "layout_changed"
    FILE_CHANGED_EXTERNALLY = "file_changed_externally"
    BROWSER_NAVIGATED = "browser_navigated"
    TERMINAL_COMPLETED = "terminal_completed"
    APP_CRASHED = "app_crashed"
    APP_UNRESPONSIVE = "app_unresponsive"
    FOCUS_CHANGED = "focus_changed"
    UNKNOWN = "unknown"


class DivergenceSeverity(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReconciliationDecision(StrEnum):
    CONTINUE = "continue"
    RESYNC = "resync"
    VERIFY_FIRST = "verify_first"
    RECOVER = "recover"
    BLOCK = "block"


class SyncRecoveryAction(StrEnum):
    NONE = "none"
    REFRESH_GRAPH = "refresh_graph"
    RECAPTURE_FOCUS = "recapture_focus"
    REBUILD_APP_STATE = "rebuild_app_state"
    CLEAR_STALE_FOCUS = "clear_stale_focus"
    WAIT_FOR_STABLE_STATE = "wait_for_stable_state"
    ESCALATE_TO_RECOVERY = "escalate_to_recovery"
    ASK_USER = "ask_user"


class GroundTruthStatus(StrEnum):
    CONSISTENT = "consistent"
    DIVERGED = "diverged"
    RESYNCED = "resynced"
    RECOVERY_REQUIRED = "recovery_required"
    BLOCKED = "blocked"


class GroundTruthReason(StrEnum):
    SESSION_CREATED = "session_created"
    SNAPSHOT_RECORDED = "snapshot_recorded"
    STATE_COMPARED = "state_compared"
    DIVERGENCE_DETECTED = "divergence_detected"
    NO_DIVERGENCE = "no_divergence"
    GRAPH_RESYNCED = "graph_resynced"
    RECOVERY_PLANNED = "recovery_planned"
    RESYNC_REQUIRED_BEFORE_CONTINUE = "resync_required_before_continue"
    SESSION_NOT_FOUND = "session_not_found"
    RUNTIME_RESET = "runtime_reset"


class GroundTruthEventKind(StrEnum):
    SESSION_CREATED = "session_created"
    SNAPSHOT_RECORDED = "snapshot_recorded"
    COMPARISON_COMPLETED = "comparison_completed"
    DIVERGENCE_COMPLETED = "divergence_completed"
    GRAPH_RESYNCED = "graph_resynced"
    RECOVERY_PLANNED = "recovery_planned"
    RUNTIME_RESET = "runtime_reset"


class BeliefStateSnapshot(OrchestrationModel):
    """
    JARVIS' belief about the workspace.

    This is derived from the cognitive graph and runtime state.
    """

    snapshot_id: str = Field(default_factory=lambda: f"belief_{uuid4().hex}")
    workspace_id: str
    focused_node_id: str | None = None
    active_app_id: str | None = None
    active_window_title: str | None = None
    layout_hash: str | None = None
    file_revision_hash: str | None = None
    browser_url: str | None = None
    terminal_status: str | None = None
    app_responsive: bool = True
    app_crashed: bool = False
    popup_count: int = Field(default=0, ge=0)
    notification_count: int = Field(default=0, ge=0)
    graph_node_count: int = Field(default=0, ge=0)
    graph_edge_count: int = Field(default=0, ge=0)
    captured_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("snapshot_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class ObservedRealitySnapshot(OrchestrationModel):
    """
    Observed environment truth.

    This comes from observers, app identity, timeline, capture, OCR,
    UI detection, file watchers, browser adapters, terminal adapters, etc.
    """

    snapshot_id: str = Field(default_factory=lambda: f"reality_{uuid4().hex}")
    workspace_id: str
    focused_node_id: str | None = None
    active_app_id: str | None = None
    active_window_title: str | None = None
    layout_hash: str | None = None
    file_revision_hash: str | None = None
    browser_url: str | None = None
    terminal_status: str | None = None
    app_responsive: bool = True
    app_crashed: bool = False
    popup_count: int = Field(default=0, ge=0)
    notification_count: int = Field(default=0, ge=0)
    user_action_counter: int = Field(default=0, ge=0)
    observed_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("snapshot_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class DivergenceSignal(OrchestrationModel):
    """
    One detected difference between belief and reality.
    """

    signal_id: str = Field(default_factory=lambda: f"divergence_{uuid4().hex}")
    cause: DivergenceCause
    severity: DivergenceSeverity
    description: str
    belief_value: object | None = None
    observed_value: object | None = None
    confidence: float = Field(default=0.90, ge=0.0, le=1.0)
    trust: TrustCalibration
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("signal_id", "description")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class DivergenceReport(OrchestrationModel):
    """
    Full divergence report.
    """

    report_id: str = Field(default_factory=lambda: f"divergence_report_{uuid4().hex}")
    belief: BeliefStateSnapshot
    observed: ObservedRealitySnapshot
    signals: tuple[DivergenceSignal, ...] = ()
    status: GroundTruthStatus
    decision: ReconciliationDecision
    reason: GroundTruthReason
    message: str
    created_at: object = Field(default_factory=utc_now)

    @field_validator("report_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @property
    def diverged(self) -> bool:
        return bool(self.signals)


class StateHistoryComparatorResult(OrchestrationModel):
    """
    State history comparison result.

    Used to detect changes that are only visible over time.
    """

    result_id: str = Field(default_factory=lambda: f"history_cmp_{uuid4().hex}")
    previous: ObservedRealitySnapshot | None = None
    current: ObservedRealitySnapshot
    signals: tuple[DivergenceSignal, ...] = ()
    created_at: object = Field(default_factory=utc_now)

    @field_validator("result_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class SyncRecovery(OrchestrationModel):
    """
    Recovery plan after divergence.

    This does not execute OS actions. It tells the orchestrator what kind of
    resync/recovery is required.
    """

    recovery_id: str = Field(default_factory=lambda: f"sync_recovery_{uuid4().hex}")
    decision: ReconciliationDecision
    actions: tuple[SyncRecoveryAction, ...]
    reason: GroundTruthReason
    safe_to_continue: bool
    requires_user_confirmation: bool = False
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("recovery_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _blocked_cannot_continue(self) -> SyncRecovery:
        if self.decision == ReconciliationDecision.BLOCK and self.safe_to_continue:
            raise ValueError("blocked recovery cannot be safe_to_continue.")

        return self


class GraphResyncResult(OrchestrationModel):
    """
    Result of graph resync.
    """

    result_id: str = Field(default_factory=lambda: f"graph_resync_{uuid4().hex}")
    applied: bool
    delta: GraphDelta | None = None
    recovery: SyncRecovery
    message: str
    created_at: object = Field(default_factory=utc_now)

    @field_validator("result_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class GroundTruthSession(OrchestrationModel):
    """
    Ground truth reconciliation session.
    """

    session_id: str = Field(default_factory=lambda: f"ground_truth_{uuid4().hex}")
    workspace_id: str
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class GroundTruthRuntimeEvent(OrchestrationModel):
    """
    Ground truth runtime event.
    """

    event_id: str = Field(default_factory=lambda: f"ground_truth_event_{uuid4().hex}")
    kind: GroundTruthEventKind
    reason: GroundTruthReason
    session_id: str | None = None
    report_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class GroundTruthRuntimeSnapshot(OrchestrationModel):
    """
    Diagnostics for Phase 8 Step 14.
    """

    name: str
    session_count: int = Field(ge=0)
    belief_snapshot_count: int = Field(ge=0)
    reality_snapshot_count: int = Field(ge=0)
    report_count: int = Field(ge=0)
    divergence_count: int = Field(ge=0)
    recovery_count: int = Field(ge=0)
    resync_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    runtime_event_count: int = Field(ge=0)
    last_reason: GroundTruthReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        return _clean_required(value)


class StateHistoryComparator:
    """
    Compares observed reality snapshots over time.
    """

    def compare(
        self,
        *,
        previous: ObservedRealitySnapshot | None,
        current: ObservedRealitySnapshot,
    ) -> StateHistoryComparatorResult:
        if previous is None:
            return StateHistoryComparatorResult(previous=None, current=current)

        signals: list[DivergenceSignal] = []

        if current.user_action_counter > previous.user_action_counter:
            signals.append(
                _signal(
                    cause=DivergenceCause.USER_ACTION,
                    severity=DivergenceSeverity.MEDIUM,
                    description="user manually changed environment state",
                    belief_value=previous.user_action_counter,
                    observed_value=current.user_action_counter,
                )
            )

        if previous.browser_url and current.browser_url:
            if previous.browser_url != current.browser_url:
                signals.append(
                    _signal(
                        cause=DivergenceCause.BROWSER_NAVIGATED,
                        severity=DivergenceSeverity.MEDIUM,
                        description="browser navigated since last observation",
                        belief_value=previous.browser_url,
                        observed_value=current.browser_url,
                    )
                )

        if previous.terminal_status and current.terminal_status:
            if previous.terminal_status != current.terminal_status:
                signals.append(
                    _signal(
                        cause=DivergenceCause.TERMINAL_COMPLETED,
                        severity=DivergenceSeverity.MEDIUM,
                        description="terminal status changed",
                        belief_value=previous.terminal_status,
                        observed_value=current.terminal_status,
                    )
                )

        return StateHistoryComparatorResult(
            previous=previous,
            current=current,
            signals=tuple(signals),
        )


class DivergenceDetector:
    """
    Detects divergence between graph belief and observed reality.
    """

    def detect(
        self,
        *,
        belief: BeliefStateSnapshot,
        observed: ObservedRealitySnapshot,
        history_result: StateHistoryComparatorResult | None = None,
    ) -> DivergenceReport:
        signals: list[DivergenceSignal] = []
        signals.extend(history_result.signals if history_result is not None else ())

        if belief.focused_node_id != observed.focused_node_id:
            signals.append(
                _signal(
                    cause=DivergenceCause.FOCUS_CHANGED,
                    severity=DivergenceSeverity.LOW,
                    description="focused graph node differs from observed focus",
                    belief_value=belief.focused_node_id,
                    observed_value=observed.focused_node_id,
                )
            )

        if belief.active_app_id != observed.active_app_id:
            signals.append(
                _signal(
                    cause=DivergenceCause.APP_STATE_CHANGE,
                    severity=DivergenceSeverity.HIGH,
                    description="active app changed unexpectedly",
                    belief_value=belief.active_app_id,
                    observed_value=observed.active_app_id,
                )
            )

        if belief.active_window_title != observed.active_window_title:
            signals.append(
                _signal(
                    cause=DivergenceCause.APP_STATE_CHANGE,
                    severity=DivergenceSeverity.MEDIUM,
                    description="active window title changed",
                    belief_value=belief.active_window_title,
                    observed_value=observed.active_window_title,
                )
            )

        if belief.layout_hash and observed.layout_hash:
            if belief.layout_hash != observed.layout_hash:
                signals.append(
                    _signal(
                        cause=DivergenceCause.LAYOUT_CHANGED,
                        severity=DivergenceSeverity.MEDIUM,
                        description="layout hash changed",
                        belief_value=belief.layout_hash,
                        observed_value=observed.layout_hash,
                    )
                )

        if belief.file_revision_hash and observed.file_revision_hash:
            if belief.file_revision_hash != observed.file_revision_hash:
                signals.append(
                    _signal(
                        cause=DivergenceCause.FILE_CHANGED_EXTERNALLY,
                        severity=DivergenceSeverity.HIGH,
                        description="file changed outside graph belief",
                        belief_value=belief.file_revision_hash,
                        observed_value=observed.file_revision_hash,
                    )
                )

        if belief.browser_url and observed.browser_url:
            if belief.browser_url != observed.browser_url:
                signals.append(
                    _signal(
                        cause=DivergenceCause.BROWSER_NAVIGATED,
                        severity=DivergenceSeverity.MEDIUM,
                        description="browser URL changed",
                        belief_value=belief.browser_url,
                        observed_value=observed.browser_url,
                    )
                )

        if belief.terminal_status and observed.terminal_status:
            if belief.terminal_status != observed.terminal_status:
                signals.append(
                    _signal(
                        cause=DivergenceCause.TERMINAL_COMPLETED,
                        severity=DivergenceSeverity.MEDIUM,
                        description="terminal command status changed",
                        belief_value=belief.terminal_status,
                        observed_value=observed.terminal_status,
                    )
                )

        if observed.popup_count > belief.popup_count:
            signals.append(
                _signal(
                    cause=DivergenceCause.POPUP_APPEARED,
                    severity=DivergenceSeverity.HIGH,
                    description="popup or modal appeared unexpectedly",
                    belief_value=belief.popup_count,
                    observed_value=observed.popup_count,
                )
            )

        if observed.notification_count > belief.notification_count:
            signals.append(
                _signal(
                    cause=DivergenceCause.OS_NOTIFICATION,
                    severity=DivergenceSeverity.LOW,
                    description="OS notification count increased",
                    belief_value=belief.notification_count,
                    observed_value=observed.notification_count,
                )
            )

        if belief.app_responsive and not observed.app_responsive:
            signals.append(
                _signal(
                    cause=DivergenceCause.APP_UNRESPONSIVE,
                    severity=DivergenceSeverity.HIGH,
                    description="app became unresponsive",
                    belief_value=belief.app_responsive,
                    observed_value=observed.app_responsive,
                )
            )

        if observed.app_crashed and not belief.app_crashed:
            signals.append(
                _signal(
                    cause=DivergenceCause.APP_CRASHED,
                    severity=DivergenceSeverity.CRITICAL,
                    description="app crashed after graph belief was built",
                    belief_value=belief.app_crashed,
                    observed_value=observed.app_crashed,
                )
            )

        decision = _decision_for(signals)
        status = GroundTruthStatus.DIVERGED if signals else GroundTruthStatus.CONSISTENT
        reason = (
            GroundTruthReason.DIVERGENCE_DETECTED
            if signals
            else GroundTruthReason.NO_DIVERGENCE
        )

        return DivergenceReport(
            belief=belief,
            observed=observed,
            signals=tuple(signals),
            status=status,
            decision=decision,
            reason=reason,
            message=(
                "ground truth divergence detected"
                if signals
                else "belief and reality are consistent"
            ),
        )


class GroundTruthEngine:
    """
    Coordinates history comparison and divergence detection.
    """

    def __init__(
        self,
        *,
        comparator: StateHistoryComparator | None = None,
        detector: DivergenceDetector | None = None,
    ) -> None:
        self._comparator = comparator or StateHistoryComparator()
        self._detector = detector or DivergenceDetector()

    def reconcile(
        self,
        *,
        belief: BeliefStateSnapshot,
        observed: ObservedRealitySnapshot,
        previous_observed: ObservedRealitySnapshot | None = None,
    ) -> DivergenceReport:
        history = self._comparator.compare(
            previous=previous_observed,
            current=observed,
        )

        return self._detector.detect(
            belief=belief,
            observed=observed,
            history_result=history,
        )


class GraphResyncRuntime:
    """
    Converts divergence reports into graph resync deltas.

    This never lies to the graph. Reality wins.
    """

    def plan_recovery(self, report: DivergenceReport) -> SyncRecovery:
        if not report.diverged:
            return SyncRecovery(
                decision=ReconciliationDecision.CONTINUE,
                actions=(SyncRecoveryAction.NONE,),
                reason=GroundTruthReason.NO_DIVERGENCE,
                safe_to_continue=True,
                message="belief and reality consistent",
            )

        causes = {signal.cause for signal in report.signals}
        severities = {signal.severity for signal in report.signals}

        if DivergenceSeverity.CRITICAL in severities:
            return SyncRecovery(
                decision=ReconciliationDecision.RECOVER,
                actions=(
                    SyncRecoveryAction.ESCALATE_TO_RECOVERY,
                    SyncRecoveryAction.REBUILD_APP_STATE,
                    SyncRecoveryAction.REFRESH_GRAPH,
                ),
                reason=GroundTruthReason.RECOVERY_PLANNED,
                safe_to_continue=False,
                message="critical divergence requires recovery before continuing",
            )

        if DivergenceCause.POPUP_APPEARED in causes:
            return SyncRecovery(
                decision=ReconciliationDecision.VERIFY_FIRST,
                actions=(
                    SyncRecoveryAction.RECAPTURE_FOCUS,
                    SyncRecoveryAction.REFRESH_GRAPH,
                    SyncRecoveryAction.ASK_USER,
                ),
                reason=GroundTruthReason.RECOVERY_PLANNED,
                safe_to_continue=False,
                requires_user_confirmation=True,
                message="popup appeared; user verification required",
            )

        if DivergenceCause.APP_UNRESPONSIVE in causes:
            return SyncRecovery(
                decision=ReconciliationDecision.BLOCK,
                actions=(
                    SyncRecoveryAction.WAIT_FOR_STABLE_STATE,
                    SyncRecoveryAction.REBUILD_APP_STATE,
                ),
                reason=GroundTruthReason.RECOVERY_PLANNED,
                safe_to_continue=False,
                message="app unresponsive; block continuation until stable",
            )

        return SyncRecovery(
            decision=ReconciliationDecision.RESYNC,
            actions=(
                SyncRecoveryAction.REFRESH_GRAPH,
                SyncRecoveryAction.RECAPTURE_FOCUS,
            ),
            reason=GroundTruthReason.RESYNC_REQUIRED_BEFORE_CONTINUE,
            safe_to_continue=False,
            message="graph resync required before continuing",
        )

    def resync(
        self,
        *,
        graph_runtime: WorkspaceGraphRuntime,
        graph_session_id: str,
        report: DivergenceReport,
    ) -> GraphResyncResult:
        recovery = self.plan_recovery(report)

        if recovery.decision == ReconciliationDecision.CONTINUE:
            return GraphResyncResult(
                applied=False,
                recovery=recovery,
                message="no graph resync required",
            )

        if recovery.decision in {
            ReconciliationDecision.BLOCK,
            ReconciliationDecision.RECOVER,
            ReconciliationDecision.VERIFY_FIRST,
        }:
            return GraphResyncResult(
                applied=False,
                recovery=recovery,
                message="graph resync deferred until verification/recovery",
            )

        delta = _delta_from_report(report)
        graph_runtime.apply_delta(session_id=graph_session_id, delta=delta)

        return GraphResyncResult(
            applied=True,
            delta=delta,
            recovery=recovery,
            message="graph resynced from observed reality",
        )


class GroundTruthReconciliationRuntime:
    """
    Phase 8 Step 14 Ground Truth Reconciliation Runtime.

    Responsibilities:
    - compare graph belief against observed reality
    - detect divergence causes
    - record state history
    - plan sync recovery
    - resync graph only when safe
    - force resync before continuation when reality diverges

    Non-responsibilities:
    - no screen capture
    - no OCR
    - no direct OS action
    - no unsafe continuation after divergence
    """

    def __init__(
        self,
        *,
        name: str = "ground_truth_reconciliation_runtime",
        engine: GroundTruthEngine | None = None,
        graph_resync: GraphResyncRuntime | None = None,
    ) -> None:
        cleaned = name.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._engine = engine or GroundTruthEngine()
        self._graph_resync = graph_resync or GraphResyncRuntime()
        self._sessions: dict[str, GroundTruthSession] = {}
        self._beliefs: dict[str, list[BeliefStateSnapshot]] = {}
        self._reality: dict[str, list[ObservedRealitySnapshot]] = {}
        self._reports: list[DivergenceReport] = []
        self._recoveries: list[SyncRecovery] = []
        self._resyncs: list[GraphResyncResult] = []
        self._events: list[GroundTruthRuntimeEvent] = []
        self._lock = RLock()
        self._last_reason: GroundTruthReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def create_session(
        self,
        *,
        workspace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> GroundTruthSession:
        session = GroundTruthSession(
            workspace_id=workspace_id,
            metadata=metadata or {},
        )
        event = self._event(
            kind=GroundTruthEventKind.SESSION_CREATED,
            reason=GroundTruthReason.SESSION_CREATED,
            session_id=session.session_id,
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._beliefs[session.session_id] = []
            self._reality[session.session_id] = []
            self._events.append(event)
            self._last_reason = event.reason

        return session

    def record_belief(
        self,
        *,
        session_id: str,
        belief: BeliefStateSnapshot,
    ) -> BeliefStateSnapshot:
        self._session_or_raise(session_id)
        event = self._event(
            kind=GroundTruthEventKind.SNAPSHOT_RECORDED,
            reason=GroundTruthReason.SNAPSHOT_RECORDED,
            session_id=session_id,
        )

        with self._lock:
            self._beliefs[session_id].append(belief)
            self._events.append(event)
            self._touch_session(session_id)
            self._last_reason = event.reason

        return belief

    def record_observation(
        self,
        *,
        session_id: str,
        observed: ObservedRealitySnapshot,
    ) -> ObservedRealitySnapshot:
        self._session_or_raise(session_id)
        event = self._event(
            kind=GroundTruthEventKind.SNAPSHOT_RECORDED,
            reason=GroundTruthReason.SNAPSHOT_RECORDED,
            session_id=session_id,
        )

        with self._lock:
            self._reality[session_id].append(observed)
            self._events.append(event)
            self._touch_session(session_id)
            self._last_reason = event.reason

        return observed

    def reconcile(
        self,
        *,
        session_id: str,
        belief: BeliefStateSnapshot,
        observed: ObservedRealitySnapshot,
    ) -> DivergenceReport:
        self._session_or_raise(session_id)

        with self._lock:
            previous = (
                self._reality[session_id][-1]
                if self._reality[session_id]
                else None
            ) 

        report = self._engine.reconcile(
            belief=belief,
            observed=observed,
            previous_observed=previous,
        )
        event = self._event(
            kind=GroundTruthEventKind.DIVERGENCE_COMPLETED,
            reason=report.reason,
            session_id=session_id,
            report_id=report.report_id,
            metadata={"decision": report.decision.value},
        )

        with self._lock:
            self._beliefs[session_id].append(belief)
            self._reality[session_id].append(observed)
            self._reports.append(report)
            self._events.append(event)
            self._touch_session(session_id)
            self._last_reason = report.reason

        return report

    def plan_recovery(self, report: DivergenceReport) -> SyncRecovery:
        recovery = self._graph_resync.plan_recovery(report)
        event = self._event(
            kind=GroundTruthEventKind.RECOVERY_PLANNED,
            reason=recovery.reason,
            report_id=report.report_id,
            metadata={"decision": recovery.decision.value},
        )

        with self._lock:
            self._recoveries.append(recovery)
            self._events.append(event)
            self._last_reason = recovery.reason

        return recovery

    def resync_graph(
        self,
        *,
        graph_runtime: WorkspaceGraphRuntime,
        graph_session_id: str,
        report: DivergenceReport,
    ) -> GraphResyncResult:
        result = self._graph_resync.resync(
            graph_runtime=graph_runtime,
            graph_session_id=graph_session_id,
            report=report,
        )
        event = self._event(
            kind=GroundTruthEventKind.GRAPH_RESYNCED,
            reason=(
                GroundTruthReason.GRAPH_RESYNCED
                if result.applied
                else result.recovery.reason
            ),
            report_id=report.report_id,
            metadata={
                "applied": result.applied,
                "decision": result.recovery.decision.value,
            },
        )

        with self._lock:
            self._resyncs.append(result)
            self._events.append(event)
            self._last_reason = event.reason

        return result

    def reports(self) -> tuple[DivergenceReport, ...]:
        with self._lock:
            return tuple(self._reports)

    def recoveries(self) -> tuple[SyncRecovery, ...]:
        with self._lock:
            return tuple(self._recoveries)

    def resyncs(self) -> tuple[GraphResyncResult, ...]:
        with self._lock:
            return tuple(self._resyncs)

    def events(self) -> tuple[GroundTruthRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def session_for(self, session_id: str) -> GroundTruthSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def snapshot(self) -> GroundTruthRuntimeSnapshot:
        with self._lock:
            return GroundTruthRuntimeSnapshot(
                name=self.name,
                session_count=len(self._sessions),
                belief_snapshot_count=sum(
                    len(items) for items in self._beliefs.values()
                ),
                reality_snapshot_count=sum(
                    len(items) for items in self._reality.values()
                ),
                report_count=len(self._reports),
                divergence_count=sum(1 for report in self._reports if report.diverged),
                recovery_count=len(self._recoveries),
                resync_count=sum(1 for result in self._resyncs if result.applied),
                blocked_count=sum(
                    1
                    for recovery in self._recoveries
                    if recovery.decision == ReconciliationDecision.BLOCK
                ),
                runtime_event_count=len(self._events),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        event = self._event(
            kind=GroundTruthEventKind.RUNTIME_RESET,
            reason=GroundTruthReason.RUNTIME_RESET,
        )

        with self._lock:
            self._sessions.clear()
            self._beliefs.clear()
            self._reality.clear()
            self._reports.clear()
            self._recoveries.clear()
            self._resyncs.clear()
            self._events.clear()
            self._events.append(event)
            self._last_reason = event.reason

    def _session_or_raise(self, session_id: str) -> GroundTruthSession:
        with self._lock:
            session = self._sessions.get(session_id)

        if session is None:
            raise ValueError(f"ground truth session not found: {session_id}")

        return session

    def _touch_session(self, session_id: str) -> None:
        session = self._sessions.get(session_id)

        if session is None:
            return

        self._sessions[session_id] = session.model_copy(
            update={"updated_at": utc_now()}
        )

    @staticmethod
    def _event(
        *,
        kind: GroundTruthEventKind,
        reason: GroundTruthReason,
        session_id: str | None = None,
        report_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GroundTruthRuntimeEvent:
        return GroundTruthRuntimeEvent(
            kind=kind,
            reason=reason,
            session_id=session_id,
            report_id=report_id,
            metadata=metadata or {},
        )


def _decision_for(
    signals: list[DivergenceSignal],
) -> ReconciliationDecision:
    if not signals:
        return ReconciliationDecision.CONTINUE

    severities = {signal.severity for signal in signals}
    causes = {signal.cause for signal in signals}

    if DivergenceSeverity.CRITICAL in severities:
        return ReconciliationDecision.RECOVER

    if DivergenceCause.APP_UNRESPONSIVE in causes:
        return ReconciliationDecision.BLOCK

    if DivergenceCause.POPUP_APPEARED in causes:
        return ReconciliationDecision.VERIFY_FIRST

    return ReconciliationDecision.RESYNC


def _delta_from_report(report: DivergenceReport) -> GraphDelta:
    nodes = tuple(
        graph_node(
            kind=_node_kind_for(signal),
            label=signal.description,
            metadata={
                "cause": signal.cause.value,
                "severity": signal.severity.value,
                "observed_value": signal.observed_value,
            },
        )
        for signal in report.signals
    )

    return GraphDelta(
        kind=GraphDeltaKind.GRAPH_REBUILT,
        added_nodes=nodes,
        reason="ground truth resync from observed reality",
        metadata={"report_id": report.report_id},
    )


def _node_kind_for(signal: DivergenceSignal) -> GraphNodeKind:
    if signal.cause == DivergenceCause.POPUP_APPEARED:
        return GraphNodeKind.DIALOG

    if signal.cause in {
        DivergenceCause.APP_CRASHED,
        DivergenceCause.APP_UNRESPONSIVE,
    }:
        return GraphNodeKind.ERROR

    if signal.cause == DivergenceCause.TERMINAL_COMPLETED:
        return GraphNodeKind.COMMAND

    if signal.cause == DivergenceCause.FILE_CHANGED_EXTERNALLY:
        return GraphNodeKind.FILE

    return GraphNodeKind.UNKNOWN


def _signal(
    *,
    cause: DivergenceCause,
    severity: DivergenceSeverity,
    description: str,
    belief_value: object | None = None,
    observed_value: object | None = None,
) -> DivergenceSignal:
    return DivergenceSignal(
        cause=cause,
        severity=severity,
        description=description,
        belief_value=belief_value,
        observed_value=observed_value,
        trust=TrustCalibration(
            confidence=0.94,
            stability=0.90,
            ambiguity=0.08,
            source=EnvironmentSource.OS_OBSERVER,
            reason=description,
        ),
    )


def _clean_required(value: str) -> str:
    cleaned = value.strip()

    if not cleaned:
        raise ValueError("field cannot be empty.")

    return cleaned