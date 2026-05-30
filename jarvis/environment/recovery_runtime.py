from __future__ import annotations

from enum import StrEnum
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.environment.models import EnvironmentSource, TrustCalibration
from jarvis.environment.trust_runtime import TrustPolicyClassification
from jarvis.environment.verification_runtime import (
    VerificationResult,
    VerificationStatus,
)
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class RecoveryAttemptKind(StrEnum):
    RETRY_SAME = "retry_same"
    RETRY_ADJUSTED = "retry_adjusted"
    ALTERNATIVE_PATH = "alternative_path"
    PARTIAL_SUCCESS_REPORT = "partial_success_report"
    ROLLBACK = "rollback"
    ESCALATE_TO_USER = "escalate_to_user"


class RecoveryStatus(StrEnum):
    PLANNED = "planned"
    RETRY_READY = "retry_ready"
    ROLLBACK_READY = "rollback_ready"
    ESCALATION_REQUIRED = "escalation_required"
    BLOCKED = "blocked"
    FAILED = "failed"


class RecoveryDecision(StrEnum):
    RETRY = "retry"
    TRY_ALTERNATIVE = "try_alternative"
    REPORT_PARTIAL_SUCCESS = "report_partial_success"
    ROLLBACK = "rollback"
    ESCALATE = "escalate"
    BLOCK = "block"


class RecoveryReason(StrEnum):
    SESSION_CREATED = "session_created"
    RETRY_SAME_SELECTED = "retry_same_selected"
    RETRY_ADJUSTED_SELECTED = "retry_adjusted_selected"
    ALTERNATIVE_PATH_SELECTED = "alternative_path_selected"
    PARTIAL_SUCCESS_REPORT_SELECTED = "partial_success_report_selected"
    ROLLBACK_SELECTED = "rollback_selected"
    ESCALATION_SELECTED = "escalation_selected"
    VERIFICATION_ALREADY_PASSED = "verification_already_passed"
    STUCK_DETECTED = "stuck_detected"
    RETRY_BUDGET_EXHAUSTED = "retry_budget_exhausted"
    ROLLBACK_REQUIRED = "rollback_required"
    SESSION_NOT_FOUND = "session_not_found"
    RUNTIME_RESET = "runtime_reset"


class RecoveryEventKind(StrEnum):
    SESSION_CREATED = "session_created"
    RECOVERY_PLANNED = "recovery_planned"
    RECOVERY_BLOCKED = "recovery_blocked"
    RUNTIME_RESET = "runtime_reset"


class RecoveryRiskLevel(StrEnum):
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryEscalationLevel(StrEnum):
    NONE = "none"
    SOFT_NOTIFY = "soft_notify"
    ASK_USER = "ask_user"
    REQUIRE_APPROVAL = "require_approval"
    HAND_OFF = "hand_off"


class RollbackTriggerKind(StrEnum):
    NONE = "none"
    IRREVERSIBLE_RISK = "irreversible_risk"
    REPEATED_FAILURE = "repeated_failure"
    STATE_MISMATCH = "state_mismatch"
    USER_APPROVAL_REQUIRED = "user_approval_required"
    LOW_TRUST = "low_trust"


class RecoveryStrategyState(StrEnum):
    AVAILABLE = "available"
    EXHAUSTED = "exhausted"
    BLOCKED = "blocked"


class RetryPolicy(OrchestrationModel):
    """
    Retry policy for recovery planning.

    Retry is bounded. It must never loop silently.
    """

    max_same_retries: int = Field(default=1, ge=0, le=10)
    max_adjusted_retries: int = Field(default=2, ge=0, le=10)
    max_alternative_paths: int = Field(default=1, ge=0, le=10)
    max_total_recovery_steps: int = Field(default=5, ge=1, le=25)
    allow_partial_success_report: bool = True
    allow_rollback: bool = True

    @model_validator(mode="after")
    def _total_must_cover_one_stage(self) -> RetryPolicy:
        if self.max_total_recovery_steps < 1:
            raise ValueError("max_total_recovery_steps must be positive.")

        return self


class EscalationPolicy(OrchestrationModel):
    """
    Escalation policy.

    JARVIS must escalate before silently failing.
    """

    escalate_after_stuck: bool = True
    require_user_for_high_risk: bool = True
    require_user_for_rollback: bool = True
    soft_notify_on_partial_success: bool = True
    handoff_on_critical_risk: bool = True


class RollbackTrigger(OrchestrationModel):
    trigger_id: str = Field(default_factory=lambda: f"rollback_trigger_{uuid4().hex}")
    kind: RollbackTriggerKind
    triggered: bool
    reason: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("trigger_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class RecoveryStrategy(OrchestrationModel):
    strategy_id: str = Field(default_factory=lambda: f"recovery_strategy_{uuid4().hex}")
    kind: RecoveryAttemptKind
    state: RecoveryStrategyState
    priority: int = Field(ge=0)
    description: str
    risk: RecoveryRiskLevel
    requires_user_approval: bool = False
    recovery_order_index: int = Field(ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("strategy_id", "description")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class RecoveryHistoryEntry(OrchestrationModel):
    entry_id: str = Field(default_factory=lambda: f"recovery_history_{uuid4().hex}")
    verification_result_id: str
    strategy_kind: RecoveryAttemptKind
    success: bool = False
    attempt_number: int = Field(default=1, ge=1)
    reason: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("entry_id", "verification_result_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class RecoveryRequest(OrchestrationModel):
    request_id: str = Field(default_factory=lambda: f"recovery_req_{uuid4().hex}")
    session_id: str
    workspace_id: str
    action_id: str
    verification: VerificationResult
    history: tuple[RecoveryHistoryEntry, ...] = ()
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    escalation_policy: EscalationPolicy = Field(default_factory=EscalationPolicy)
    irreversible_risk: bool = False
    partial_success_available: bool = False
    rollback_available: bool = False
    alternative_path_available: bool = False
    user_initiated: bool = True
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("request_id", "session_id", "workspace_id", "action_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class StuckReport(OrchestrationModel):
    report_id: str = Field(default_factory=lambda: f"stuck_report_{uuid4().hex}")
    stuck: bool
    same_retry_count: int = Field(ge=0)
    adjusted_retry_count: int = Field(ge=0)
    alternative_count: int = Field(ge=0)
    total_attempt_count: int = Field(ge=0)
    reason: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("report_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class RecoveryPlanStep(OrchestrationModel):
    step_id: str = Field(default_factory=lambda: f"recovery_step_{uuid4().hex}")
    order: int = Field(ge=0)
    strategy: RecoveryStrategy
    expected_effect: str
    requires_verification_after: bool = True
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("step_id", "expected_effect")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class RecoveryPlan(OrchestrationModel):
    plan_id: str = Field(default_factory=lambda: f"recovery_plan_{uuid4().hex}")
    action_id: str
    selected_strategy: RecoveryStrategy
    steps: tuple[RecoveryPlanStep, ...]
    rollback_trigger: RollbackTrigger
    escalation_level: RecoveryEscalationLevel
    recovery_order: tuple[RecoveryAttemptKind, ...]
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("plan_id", "action_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _requires_steps(self) -> RecoveryPlan:
        if not self.steps:
            raise ValueError("recovery plan requires at least one step.")

        return self


class RecoveryAuditRecord(OrchestrationModel):
    audit_id: str = Field(default_factory=lambda: f"recovery_audit_{uuid4().hex}")
    request_id: str
    action_id: str
    status: RecoveryStatus
    decision: RecoveryDecision
    reason: RecoveryReason
    selected_strategy: RecoveryAttemptKind | None = None
    recovery_needed: bool
    escalated: bool = False
    rollback_triggered: bool = False
    silent_failure: bool = False
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("audit_id", "request_id", "action_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _never_silent_failure(self) -> RecoveryAuditRecord:
        if self.silent_failure:
            raise ValueError("recovery runtime must never fail silently.")

        return self


class RecoveryResult(OrchestrationModel):
    result_id: str = Field(default_factory=lambda: f"recovery_result_{uuid4().hex}")
    status: RecoveryStatus
    decision: RecoveryDecision
    reason: RecoveryReason
    request: RecoveryRequest
    stuck_report: StuckReport
    plan: RecoveryPlan | None = None
    audit: RecoveryAuditRecord
    trust: TrustCalibration
    retry_allowed: bool = False
    rollback_required: bool = False
    escalation_required: bool = False
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("result_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _decision_flags_consistent(self) -> RecoveryResult:
        if self.decision == RecoveryDecision.RETRY and not self.retry_allowed:
            raise ValueError("RETRY decision requires retry_allowed.")

        if self.decision == RecoveryDecision.ROLLBACK and not self.rollback_required:
            raise ValueError("ROLLBACK decision requires rollback_required.")

        if self.decision == RecoveryDecision.ESCALATE and not self.escalation_required:
            raise ValueError("ESCALATE decision requires escalation_required.")

        return self


class RecoveryRuntimeSession(OrchestrationModel):
    session_id: str = Field(default_factory=lambda: f"recovery_session_{uuid4().hex}")
    workspace_id: str
    recovery_count: int = Field(default=0, ge=0)
    rollback_count: int = Field(default=0, ge=0)
    escalation_count: int = Field(default=0, ge=0)
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class RecoveryRuntimeEvent(OrchestrationModel):
    event_id: str = Field(default_factory=lambda: f"recovery_event_{uuid4().hex}")
    kind: RecoveryEventKind
    reason: RecoveryReason
    session_id: str | None = None
    result_id: str | None = None
    audit_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class RecoveryRuntimeSnapshot(OrchestrationModel):
    name: str
    session_count: int = Field(ge=0)
    result_count: int = Field(ge=0)
    retry_count: int = Field(ge=0)
    rollback_count: int = Field(ge=0)
    escalation_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    audit_count: int = Field(ge=0)
    runtime_event_count: int = Field(ge=0)
    last_reason: RecoveryReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        return _clean_required(value)


class StuckDetector:
    def detect(
        self,
        *,
        history: tuple[RecoveryHistoryEntry, ...],
        retry_policy: RetryPolicy,
    ) -> StuckReport:
        same_count = _count(history, RecoveryAttemptKind.RETRY_SAME)
        adjusted_count = _count(history, RecoveryAttemptKind.RETRY_ADJUSTED)
        alternative_count = _count(history, RecoveryAttemptKind.ALTERNATIVE_PATH)
        total_count = len(history)

        stuck = (
            same_count >= retry_policy.max_same_retries
            and adjusted_count >= retry_policy.max_adjusted_retries
            and alternative_count >= retry_policy.max_alternative_paths
        ) or total_count >= retry_policy.max_total_recovery_steps

        reason = "recovery path still available"
        if stuck:
            reason = "recovery retry budget exhausted or loop detected"

        return StuckReport(
            stuck=stuck,
            same_retry_count=same_count,
            adjusted_retry_count=adjusted_count,
            alternative_count=alternative_count,
            total_attempt_count=total_count,
            reason=reason,
        )


class RecoveryStrategySelector:
    """
    Selects recovery strategy in the required order:

    retry same → retry adjusted → alternative path → partial success report
    → rollback → escalate to user
    """

    recovery_order: tuple[RecoveryAttemptKind, ...] = (
        RecoveryAttemptKind.RETRY_SAME,
        RecoveryAttemptKind.RETRY_ADJUSTED,
        RecoveryAttemptKind.ALTERNATIVE_PATH,
        RecoveryAttemptKind.PARTIAL_SUCCESS_REPORT,
        RecoveryAttemptKind.ROLLBACK,
        RecoveryAttemptKind.ESCALATE_TO_USER,
    )

    def select(
        self,
        *,
        request: RecoveryRequest,
        stuck: StuckReport,
    ) -> RecoveryStrategy:
        if not request.verification.recovery_needed:
            return _strategy(
                kind=RecoveryAttemptKind.PARTIAL_SUCCESS_REPORT,
                state=RecoveryStrategyState.BLOCKED,
                priority=99,
                description="verification already passed; recovery is not needed",
                risk=RecoveryRiskLevel.SAFE,
                order_index=3,
            )

        if request.irreversible_risk:
            return _strategy(
                kind=RecoveryAttemptKind.ROLLBACK,
                state=(
                    RecoveryStrategyState.AVAILABLE
                    if request.rollback_available
                    else RecoveryStrategyState.BLOCKED
                ),
                priority=0,
                description="irreversible risk requires rollback path",
                risk=RecoveryRiskLevel.CRITICAL,
                requires_user_approval=True,
                order_index=4,
            )

        if not stuck.stuck:
            same_count = _count(
                request.history,
                RecoveryAttemptKind.RETRY_SAME,
            )
            adjusted_count = _count(
                request.history,
                RecoveryAttemptKind.RETRY_ADJUSTED,
            )
            alternative_count = _count(
                request.history,
                RecoveryAttemptKind.ALTERNATIVE_PATH,
            )

            if same_count < request.retry_policy.max_same_retries:
                return _strategy(
                    kind=RecoveryAttemptKind.RETRY_SAME,
                    state=RecoveryStrategyState.AVAILABLE,
                    priority=1,
                    description="retry the same action once with verification",
                    risk=RecoveryRiskLevel.LOW,
                    order_index=0,
                )

            if adjusted_count < request.retry_policy.max_adjusted_retries:
                return _strategy(
                    kind=RecoveryAttemptKind.RETRY_ADJUSTED,
                    state=RecoveryStrategyState.AVAILABLE,
                    priority=2,
                    description="retry with adjusted target/context",
                    risk=RecoveryRiskLevel.MEDIUM,
                    order_index=1,
                )

            if (
                request.alternative_path_available
                and alternative_count < request.retry_policy.max_alternative_paths
            ):
                return _strategy(
                    kind=RecoveryAttemptKind.ALTERNATIVE_PATH,
                    state=RecoveryStrategyState.AVAILABLE,
                    priority=3,
                    description="try an alternative recovery path",
                    risk=RecoveryRiskLevel.MEDIUM,
                    order_index=2,
                )

        if (
            request.partial_success_available
            and request.retry_policy.allow_partial_success_report
        ):
            return _strategy(
                kind=RecoveryAttemptKind.PARTIAL_SUCCESS_REPORT,
                state=RecoveryStrategyState.AVAILABLE,
                priority=4,
                description="report partial success before stronger recovery",
                risk=RecoveryRiskLevel.LOW,
                order_index=3,
            )

        if request.rollback_available and request.retry_policy.allow_rollback:
            return _strategy(
                kind=RecoveryAttemptKind.ROLLBACK,
                state=RecoveryStrategyState.AVAILABLE,
                priority=5,
                description="rollback to last verified safe state",
                risk=RecoveryRiskLevel.HIGH,
                requires_user_approval=request.escalation_policy.require_user_for_rollback,
                order_index=4,
            )

        return _strategy(
            kind=RecoveryAttemptKind.ESCALATE_TO_USER,
            state=RecoveryStrategyState.AVAILABLE,
            priority=6,
            description="escalate recovery decision to user",
            risk=RecoveryRiskLevel.MEDIUM,
            requires_user_approval=True,
            order_index=5,
        )


class RecoveryRuntime:
    """
    Phase 8 Step 30 Recovery & Retry Runtime.

    Responsibilities:
    - consume Step 29 verification result
    - choose recovery strategy in strict order
    - detect stuck/retry exhaustion
    - trigger rollback when necessary
    - escalate before failing
    - audit every recovery decision

    Non-responsibilities:
    - does not execute retry
    - does not execute rollback
    - does not hide failure
    """

    def __init__(
        self,
        *,
        name: str = "recovery_runtime",
        stuck_detector: StuckDetector | None = None,
        selector: RecoveryStrategySelector | None = None,
    ) -> None:
        cleaned = name.strip()
        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._stuck_detector = stuck_detector or StuckDetector()
        self._selector = selector or RecoveryStrategySelector()
        self._sessions: dict[str, RecoveryRuntimeSession] = {}
        self._results: list[RecoveryResult] = []
        self._audits: list[RecoveryAuditRecord] = []
        self._events: list[RecoveryRuntimeEvent] = []
        self._lock = RLock()
        self._last_reason: RecoveryReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def create_session(
        self,
        *,
        workspace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> RecoveryRuntimeSession:
        session = RecoveryRuntimeSession(
            workspace_id=workspace_id,
            metadata=metadata or {},
        )
        event = self._event(
            kind=RecoveryEventKind.SESSION_CREATED,
            reason=RecoveryReason.SESSION_CREATED,
            session_id=session.session_id,
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._events.append(event)
            self._last_reason = event.reason

        return session

    def recover(self, request: RecoveryRequest) -> RecoveryResult:
        if self.session_for(request.session_id) is None:
            result = _blocked_result(
                request=request,
                stuck_report=_empty_stuck_report(),
                reason=RecoveryReason.SESSION_NOT_FOUND,
                message="recovery runtime session not found",
            )
            self._record_result(result)
            return result

        if request.verification.status == VerificationStatus.PASSED:
            result = _blocked_result(
                request=request,
                stuck_report=_empty_stuck_report(),
                reason=RecoveryReason.VERIFICATION_ALREADY_PASSED,
                message="verification already passed; recovery is not needed",
            )
            self._record_result(result)
            return result

        stuck = self._stuck_detector.detect(
            history=request.history,
            retry_policy=request.retry_policy,
        )
        strategy = self._selector.select(request=request, stuck=stuck)
        result = _build_recovery_result(
            request=request,
            stuck_report=stuck,
            strategy=strategy,
        )
        self._record_result(result)

        return result

    def session_for(self, session_id: str) -> RecoveryRuntimeSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def results(self) -> tuple[RecoveryResult, ...]:
        with self._lock:
            return tuple(self._results)

    def audits(self) -> tuple[RecoveryAuditRecord, ...]:
        with self._lock:
            return tuple(self._audits)

    def events(self) -> tuple[RecoveryRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def snapshot(self) -> RecoveryRuntimeSnapshot:
        with self._lock:
            return RecoveryRuntimeSnapshot(
                name=self.name,
                session_count=len(self._sessions),
                result_count=len(self._results),
                retry_count=sum(
                    1
                    for result in self._results
                    if result.decision == RecoveryDecision.RETRY
                ),
                rollback_count=sum(
                    1
                    for result in self._results
                    if result.decision == RecoveryDecision.ROLLBACK
                ),
                escalation_count=sum(
                    1
                    for result in self._results
                    if result.decision == RecoveryDecision.ESCALATE
                ),
                blocked_count=sum(
                    1
                    for result in self._results
                    if result.status
                    in {
                        RecoveryStatus.BLOCKED,
                        RecoveryStatus.FAILED,
                    }
                ),
                audit_count=len(self._audits),
                runtime_event_count=len(self._events),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        event = self._event(
            kind=RecoveryEventKind.RUNTIME_RESET,
            reason=RecoveryReason.RUNTIME_RESET,
        )

        with self._lock:
            self._sessions.clear()
            self._results.clear()
            self._audits.clear()
            self._events.clear()
            self._events.append(event)
            self._last_reason = event.reason

    def _record_result(self, result: RecoveryResult) -> None:
        event = self._event(
            kind=(
                RecoveryEventKind.RECOVERY_PLANNED
                if result.status
                in {
                    RecoveryStatus.PLANNED,
                    RecoveryStatus.RETRY_READY,
                    RecoveryStatus.ROLLBACK_READY,
                    RecoveryStatus.ESCALATION_REQUIRED,
                }
                else RecoveryEventKind.RECOVERY_BLOCKED
            ),
            reason=result.reason,
            session_id=result.request.session_id,
            result_id=result.result_id,
            audit_id=result.audit.audit_id,
            metadata={
                "status": result.status.value,
                "decision": result.decision.value,
            },
        )

        with self._lock:
            self._results.append(result)
            self._audits.append(result.audit)
            self._events.append(event)
            self._last_reason = result.reason
            self._touch_session(result)

    def _touch_session(self, result: RecoveryResult) -> None:
        session = self._sessions.get(result.request.session_id)
        if session is None:
            return

        self._sessions[result.request.session_id] = session.model_copy(
            update={
                "updated_at": utc_now(),
                "recovery_count": session.recovery_count + 1,
                "rollback_count": session.rollback_count
                + (1 if result.rollback_required else 0),
                "escalation_count": session.escalation_count
                + (1 if result.decision == RecoveryDecision.ESCALATE else 0),
            }
        )

    @staticmethod
    def _event(
        *,
        kind: RecoveryEventKind,
        reason: RecoveryReason,
        session_id: str | None = None,
        result_id: str | None = None,
        audit_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RecoveryRuntimeEvent:
        return RecoveryRuntimeEvent(
            kind=kind,
            reason=reason,
            session_id=session_id,
            result_id=result_id,
            audit_id=audit_id,
            metadata=metadata or {},
        )


def _build_recovery_result(
    *,
    request: RecoveryRequest,
    stuck_report: StuckReport,
    strategy: RecoveryStrategy,
) -> RecoveryResult:
    if strategy.kind == RecoveryAttemptKind.RETRY_SAME:
        return _planned_result(
            request=request,
            stuck_report=stuck_report,
            strategy=strategy,
            status=RecoveryStatus.RETRY_READY,
            decision=RecoveryDecision.RETRY,
            reason=RecoveryReason.RETRY_SAME_SELECTED,
            retry_allowed=True,
            message="retry same action with verification",
        )

    if strategy.kind == RecoveryAttemptKind.RETRY_ADJUSTED:
        return _planned_result(
            request=request,
            stuck_report=stuck_report,
            strategy=strategy,
            status=RecoveryStatus.RETRY_READY,
            decision=RecoveryDecision.RETRY,
            reason=RecoveryReason.RETRY_ADJUSTED_SELECTED,
            retry_allowed=True,
            message="retry action with adjusted target/context",
        )

    if strategy.kind == RecoveryAttemptKind.ALTERNATIVE_PATH:
        return _planned_result(
            request=request,
            stuck_report=stuck_report,
            strategy=strategy,
            status=RecoveryStatus.PLANNED,
            decision=RecoveryDecision.TRY_ALTERNATIVE,
            reason=RecoveryReason.ALTERNATIVE_PATH_SELECTED,
            message="try alternative recovery path",
        )

    if strategy.kind == RecoveryAttemptKind.PARTIAL_SUCCESS_REPORT:
        return _planned_result(
            request=request,
            stuck_report=stuck_report,
            strategy=strategy,
            status=RecoveryStatus.PLANNED,
            decision=RecoveryDecision.REPORT_PARTIAL_SUCCESS,
            reason=RecoveryReason.PARTIAL_SUCCESS_REPORT_SELECTED,
            message="report partial success before escalation",
        )

    if strategy.kind == RecoveryAttemptKind.ROLLBACK:
        if strategy.state != RecoveryStrategyState.AVAILABLE:
            return _planned_result(
                request=request,
                stuck_report=stuck_report,
                strategy=strategy,
                status=RecoveryStatus.ESCALATION_REQUIRED,
                decision=RecoveryDecision.ESCALATE,
                reason=RecoveryReason.ESCALATION_SELECTED,
                escalation_required=True,
                message="rollback required but unavailable; escalate to user",
            )

        return _planned_result(
            request=request,
            stuck_report=stuck_report,
            strategy=strategy,
            status=RecoveryStatus.ROLLBACK_READY,
            decision=RecoveryDecision.ROLLBACK,
            reason=RecoveryReason.ROLLBACK_SELECTED,
            rollback_required=True,
            escalation_required=strategy.requires_user_approval,
            message="rollback to last verified safe state",
        )

    return _planned_result(
        request=request,
        stuck_report=stuck_report,
        strategy=strategy,
        status=RecoveryStatus.ESCALATION_REQUIRED,
        decision=RecoveryDecision.ESCALATE,
        reason=(
            RecoveryReason.STUCK_DETECTED
            if stuck_report.stuck
            else RecoveryReason.ESCALATION_SELECTED
        ),
        escalation_required=True,
        message="escalate recovery decision to user",
    )


def _planned_result(
    *,
    request: RecoveryRequest,
    stuck_report: StuckReport,
    strategy: RecoveryStrategy,
    status: RecoveryStatus,
    decision: RecoveryDecision,
    reason: RecoveryReason,
    message: str,
    retry_allowed: bool = False,
    rollback_required: bool = False,
    escalation_required: bool = False,
) -> RecoveryResult:
    rollback_trigger = _rollback_trigger_for(
        request=request,
        strategy=strategy,
        rollback_required=rollback_required,
    )
    plan = RecoveryPlan(
        action_id=request.action_id,
        selected_strategy=strategy,
        steps=(
            RecoveryPlanStep(
                order=0,
                strategy=strategy,
                expected_effect=_expected_effect_for(strategy.kind),
            ),
        ),
        rollback_trigger=rollback_trigger,
        escalation_level=_escalation_level_for(
            request=request,
            strategy=strategy,
            escalation_required=escalation_required,
        ),
        recovery_order=RecoveryStrategySelector.recovery_order,
    )
    audit = RecoveryAuditRecord(
        request_id=request.request_id,
        action_id=request.action_id,
        status=status,
        decision=decision,
        reason=reason,
        selected_strategy=strategy.kind,
        recovery_needed=request.verification.recovery_needed,
        escalated=escalation_required,
        rollback_triggered=rollback_required,
    )

    return RecoveryResult(
        status=status,
        decision=decision,
        reason=reason,
        request=request,
        stuck_report=stuck_report,
        plan=plan,
        audit=audit,
        trust=_trust_for(strategy=strategy, stuck=stuck_report),
        retry_allowed=retry_allowed,
        rollback_required=rollback_required,
        escalation_required=escalation_required,
        message=message,
    )


def _blocked_result(
    *,
    request: RecoveryRequest,
    stuck_report: StuckReport,
    reason: RecoveryReason,
    message: str,
) -> RecoveryResult:
    audit = RecoveryAuditRecord(
        request_id=request.request_id,
        action_id=request.action_id,
        status=RecoveryStatus.BLOCKED,
        decision=RecoveryDecision.BLOCK,
        reason=reason,
        selected_strategy=None,
        recovery_needed=request.verification.recovery_needed,
    )

    return RecoveryResult(
        status=RecoveryStatus.BLOCKED,
        decision=RecoveryDecision.BLOCK,
        reason=reason,
        request=request,
        stuck_report=stuck_report,
        plan=None,
        audit=audit,
        trust=_trust(
            confidence=0.20,
            stability=0.20,
            ambiguity=0.80,
            reason=message,
        ),
        retry_allowed=False,
        rollback_required=False,
        escalation_required=False,
        message=message,
    )


def _rollback_trigger_for(
    *,
    request: RecoveryRequest,
    strategy: RecoveryStrategy,
    rollback_required: bool,
) -> RollbackTrigger:
    if rollback_required:
        return RollbackTrigger(
            kind=RollbackTriggerKind.STATE_MISMATCH,
            triggered=True,
            reason="selected strategy requires rollback",
            confidence=0.90,
        )

    if request.irreversible_risk:
        return RollbackTrigger(
            kind=RollbackTriggerKind.IRREVERSIBLE_RISK,
            triggered=True,
            reason="irreversible risk detected",
            confidence=0.95,
        )

    if strategy.risk in {
        RecoveryRiskLevel.HIGH,
        RecoveryRiskLevel.CRITICAL,
    }:
        return RollbackTrigger(
            kind=RollbackTriggerKind.USER_APPROVAL_REQUIRED,
            triggered=False,
            reason="high risk strategy may require rollback later",
            confidence=0.75,
        )

    return RollbackTrigger(
        kind=RollbackTriggerKind.NONE,
        triggered=False,
        reason="rollback not currently required",
        confidence=0.80,
    )


def _escalation_level_for(
    *,
    request: RecoveryRequest,
    strategy: RecoveryStrategy,
    escalation_required: bool,
) -> RecoveryEscalationLevel:
    if escalation_required:
        if strategy.risk == RecoveryRiskLevel.CRITICAL:
            return RecoveryEscalationLevel.HAND_OFF

        if strategy.requires_user_approval:
            return RecoveryEscalationLevel.REQUIRE_APPROVAL

        return RecoveryEscalationLevel.ASK_USER

    if (
        strategy.kind == RecoveryAttemptKind.PARTIAL_SUCCESS_REPORT
        and request.escalation_policy.soft_notify_on_partial_success
    ):
        return RecoveryEscalationLevel.SOFT_NOTIFY

    return RecoveryEscalationLevel.NONE


def _expected_effect_for(kind: RecoveryAttemptKind) -> str:
    if kind == RecoveryAttemptKind.RETRY_SAME:
        return "same action re-attempted and verified"

    if kind == RecoveryAttemptKind.RETRY_ADJUSTED:
        return "adjusted action re-attempted and verified"

    if kind == RecoveryAttemptKind.ALTERNATIVE_PATH:
        return "alternative path attempted and verified"

    if kind == RecoveryAttemptKind.PARTIAL_SUCCESS_REPORT:
        return "partial success reported without pretending completion"

    if kind == RecoveryAttemptKind.ROLLBACK:
        return "state returned to last verified safe point"

    return "user receives clear escalation with recovery context"


def _strategy(
    *,
    kind: RecoveryAttemptKind,
    state: RecoveryStrategyState,
    priority: int,
    description: str,
    risk: RecoveryRiskLevel,
    order_index: int,
    requires_user_approval: bool = False,
) -> RecoveryStrategy:
    return RecoveryStrategy(
        kind=kind,
        state=state,
        priority=priority,
        description=description,
        risk=risk,
        requires_user_approval=requires_user_approval,
        recovery_order_index=order_index,
    )


def _trust_for(
    *,
    strategy: RecoveryStrategy,
    stuck: StuckReport,
) -> TrustCalibration:
    confidence = 0.82
    ambiguity = 0.18

    if stuck.stuck:
        confidence = 0.60
        ambiguity = 0.40

    if strategy.risk in {
        RecoveryRiskLevel.HIGH,
        RecoveryRiskLevel.CRITICAL,
    }:
        confidence = min(confidence, 0.70)
        ambiguity = max(ambiguity, 0.30)

    return _trust(
        confidence=confidence,
        stability=max(0.0, min(1.0, confidence + 0.05)),
        ambiguity=ambiguity,
        reason=f"recovery strategy selected: {strategy.kind.value}",
    )


def _trust(
    *,
    confidence: float,
    stability: float,
    ambiguity: float,
    reason: str,
) -> TrustCalibration:
    return TrustCalibration(
        confidence=confidence,
        stability=stability,
        ambiguity=ambiguity,
        source=EnvironmentSource.OS_OBSERVER,
        reason=reason,
        metadata={"policy": TrustPolicyClassification.REVIEW.value},
    )


def _empty_stuck_report() -> StuckReport:
    return StuckReport(
        stuck=False,
        same_retry_count=0,
        adjusted_retry_count=0,
        alternative_count=0,
        total_attempt_count=0,
        reason="stuck detection not applicable",
    )


def _count(
    history: tuple[RecoveryHistoryEntry, ...],
    kind: RecoveryAttemptKind,
) -> int:
    return sum(1 for entry in history if entry.strategy_kind == kind)


def _clean_required(value: str) -> str:
    cleaned = value.strip()

    if not cleaned:
        raise ValueError("field cannot be empty.")

    return cleaned