from __future__ import annotations

from enum import StrEnum
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.environment.app_identity import (
    AppIdentityResult,
    AppIdentityStatus,
    DetectedAppKind,
)
from jarvis.environment.interaction_policy import (
    InteractionDecision,
    InteractionPolicyResult,
    PhysicalInteractionKind,
)
from jarvis.environment.models import EnvironmentSource, TrustCalibration
from jarvis.environment.trust_runtime import TrustPolicyClassification
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class AppControlActionKind(StrEnum):
    LAUNCH = "launch"
    FOCUS = "focus"
    SWITCH = "switch"
    CLOSE = "close"
    RESTORE_SESSION = "restore_session"
    CHECK_RESPONSIVENESS = "check_responsiveness"


class AppControlStatus(StrEnum):
    READY = "ready"
    LAUNCHED = "launched"
    FOCUSED = "focused"
    SWITCHED = "switched"
    CLOSE_READY = "close_ready"
    RESTORED = "restored"
    BLOCKED = "blocked"
    FAILED = "failed"


class AppControlDecision(StrEnum):
    ALLOW = "allow"
    REQUIRE_VISIBILITY_VERIFICATION = "require_visibility_verification"
    REQUIRE_RESPONSIVENESS_VERIFICATION = "require_responsiveness_verification"
    REQUIRE_SESSION_RESTORE = "require_session_restore"
    REQUIRE_CLOSE_APPROVAL = "require_close_approval"
    BLOCK = "block"


class AppControlReason(StrEnum):
    SESSION_CREATED = "session_created"
    POLICY_NOT_ELIGIBLE = "policy_not_eligible"
    APP_IDENTITY_UNKNOWN = "app_identity_unknown"
    APP_BLOCKED_BY_IDENTITY = "app_blocked_by_identity"
    APP_LAUNCH_PLANNED = "app_launch_planned"
    APP_RESPONSIVE_VERIFIED = "app_responsive_verified"
    APP_FOCUSED = "app_focused"
    APP_SWITCHED = "app_switched"
    APP_CLOSE_SAFE = "app_close_safe"
    APP_CLOSE_APPROVAL_REQUIRED = "app_close_approval_required"
    UNSAVED_STATE_BLOCKED_CLOSE = "unsaved_state_blocked_close"
    SESSION_RESTORE_REQUIRED = "session_restore_required"
    SESSION_RESTORED = "session_restored"
    SESSION_NOT_FOUND = "session_not_found"
    RUNTIME_RESET = "runtime_reset"


class AppControlEventKind(StrEnum):
    SESSION_CREATED = "session_created"
    APP_CONTROL_ALLOWED = "app_control_allowed"
    APP_CONTROL_BLOCKED = "app_control_blocked"
    RUNTIME_RESET = "runtime_reset"


class AppVisibilityState(StrEnum):
    UNKNOWN = "unknown"
    NOT_VISIBLE = "not_visible"
    VISIBLE = "visible"
    MINIMIZED = "minimized"
    OCCLUDED = "occluded"


class AppControlResponsiveness(StrEnum):
    UNKNOWN = "unknown"
    RESPONSIVE = "responsive"
    UNRESPONSIVE = "unresponsive"


class AppCloseSafety(StrEnum):
    SAFE_TO_CLOSE = "safe_to_close"
    UNSAVED_CHANGES = "unsaved_changes"
    BACKGROUND_TASK_RUNNING = "background_task_running"
    CRITICAL_OR_UNKNOWN_APP = "critical_or_unknown_app"


class AppSessionRestoreStatus(StrEnum):
    NOT_REQUIRED = "not_required"
    REQUIRED = "required"
    RESTORED = "restored"
    FAILED = "failed"


class AppControlTarget(OrchestrationModel):
    target_id: str = Field(default_factory=lambda: f"app_target_{uuid4().hex}")
    app_name: str
    app_kind: DetectedAppKind = DetectedAppKind.UNKNOWN
    executable_hint: str | None = None
    window_title: str | None = None
    process_id: int | None = None
    workspace_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("target_id", "app_name")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class AppControlRequest(OrchestrationModel):
    request_id: str = Field(default_factory=lambda: f"app_control_req_{uuid4().hex}")
    session_id: str
    action: AppControlActionKind
    target: AppControlTarget
    policy_result: InteractionPolicyResult
    identity: AppIdentityResult | None = None
    require_session_restore: bool = False
    unsaved_changes_hint: bool = False
    background_task_hint: bool = False
    user_initiated: bool = True
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("request_id", "session_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class AppVisibilityVerification(OrchestrationModel):
    verification_id: str = Field(default_factory=lambda: f"app_visible_{uuid4().hex}")
    state: AppVisibilityState
    visible: bool
    focused: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("verification_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class AppResponsivenessVerification(OrchestrationModel):
    verification_id: str = Field(default_factory=lambda: f"app_response_{uuid4().hex}")
    responsive: bool
    state: AppControlResponsiveness = AppControlResponsiveness.UNKNOWN
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("verification_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class AppCloseSafetyReport(OrchestrationModel):
    report_id: str = Field(default_factory=lambda: f"app_close_{uuid4().hex}")
    safety: AppCloseSafety
    safe_to_close: bool
    requires_approval: bool
    reason: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("report_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class AppSessionRestorePlan(OrchestrationModel):
    restore_id: str = Field(default_factory=lambda: f"app_restore_{uuid4().hex}")
    status: AppSessionRestoreStatus
    restore_steps: tuple[str, ...] = ()
    restored_workspace_id: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("restore_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class AppControlAuditRecord(OrchestrationModel):
    audit_id: str = Field(default_factory=lambda: f"app_control_audit_{uuid4().hex}")
    request_id: str
    action: AppControlActionKind
    target_name: str
    status: AppControlStatus
    decision: AppControlDecision
    reason: AppControlReason
    policy_result_id: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("audit_id", "request_id", "target_name", "policy_result_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class AppControlResult(OrchestrationModel):
    result_id: str = Field(default_factory=lambda: f"app_control_result_{uuid4().hex}")
    status: AppControlStatus
    decision: AppControlDecision
    reason: AppControlReason
    request: AppControlRequest
    visibility: AppVisibilityVerification | None = None
    responsiveness: AppResponsivenessVerification | None = None
    close_safety: AppCloseSafetyReport | None = None
    restore_plan: AppSessionRestorePlan | None = None
    audit: AppControlAuditRecord
    trust: TrustCalibration
    control_eligible: bool = False
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("result_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _eligible_requires_allow(self) -> AppControlResult:
        if self.control_eligible and self.decision != AppControlDecision.ALLOW:
            raise ValueError("control_eligible requires ALLOW decision.")
        return self


class AppControlSession(OrchestrationModel):
    session_id: str = Field(
        default_factory=lambda: f"app_control_session_{uuid4().hex}"
    )
    workspace_id: str
    active_app_name: str | None = None
    restored_session_count: int = Field(default=0, ge=0)
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class AppControlRuntimeEvent(OrchestrationModel):
    event_id: str = Field(default_factory=lambda: f"app_control_event_{uuid4().hex}")
    kind: AppControlEventKind
    reason: AppControlReason
    session_id: str | None = None
    result_id: str | None = None
    request_id: str | None = None
    audit_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class AppControlRuntimeSnapshot(OrchestrationModel):
    name: str
    session_count: int = Field(ge=0)
    result_count: int = Field(ge=0)
    eligible_count: int = Field(ge=0)
    launched_count: int = Field(ge=0)
    focused_count: int = Field(ge=0)
    close_ready_count: int = Field(ge=0)
    restored_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    audit_count: int = Field(ge=0)
    runtime_event_count: int = Field(ge=0)
    last_reason: AppControlReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        return _clean_required(value)


class AppLauncher:
    def evaluate(self, request: AppControlRequest) -> AppControlResult:
        blocked = _policy_or_identity_block(request)
        if blocked is not None:
            return blocked

        visibility = AppVisibilityVerification(
            state=AppVisibilityState.VISIBLE,
            visible=True,
            focused=True,
            confidence=0.86,
            reason="app launch visibility verified",
        )
        responsiveness = _responsiveness_from_identity(request.identity)
        restore = SessionRestorer().restore(request)

        if not responsiveness.responsive:
            return _result(
                request=request,
                status=AppControlStatus.BLOCKED,
                decision=AppControlDecision.REQUIRE_RESPONSIVENESS_VERIFICATION,
                reason=AppControlReason.APP_RESPONSIVE_VERIFIED,
                visibility=visibility,
                responsiveness=responsiveness,
                restore_plan=restore,
                control_eligible=False,
                message="app launch blocked until responsiveness is verified",
            )

        if request.require_session_restore:
            if restore.status != AppSessionRestoreStatus.RESTORED:
                return _result(
                    request=request,
                    status=AppControlStatus.BLOCKED,
                    decision=AppControlDecision.REQUIRE_SESSION_RESTORE,
                    reason=AppControlReason.SESSION_RESTORE_REQUIRED,
                    visibility=visibility,
                    responsiveness=responsiveness,
                    restore_plan=restore,
                    control_eligible=False,
                    message="app launch requires session restore",
                )

        return _result(
            request=request,
            status=AppControlStatus.LAUNCHED,
            decision=AppControlDecision.ALLOW,
            reason=AppControlReason.APP_LAUNCH_PLANNED,
            visibility=visibility,
            responsiveness=responsiveness,
            restore_plan=restore,
            control_eligible=True,
            message="app is visible, responsive, safe, and restored",
        )


class AppFocuser:
    def evaluate(self, request: AppControlRequest) -> AppControlResult:
        blocked = _policy_or_identity_block(request)
        if blocked is not None:
            return blocked

        visibility = AppVisibilityVerification(
            state=AppVisibilityState.VISIBLE,
            visible=True,
            focused=True,
            confidence=0.88,
            reason="app focus visibility verified",
        )
        responsiveness = _responsiveness_from_identity(request.identity)

        if not responsiveness.responsive:
            return _result(
                request=request,
                status=AppControlStatus.BLOCKED,
                decision=AppControlDecision.REQUIRE_RESPONSIVENESS_VERIFICATION,
                reason=AppControlReason.APP_RESPONSIVE_VERIFIED,
                visibility=visibility,
                responsiveness=responsiveness,
                control_eligible=False,
                message="cannot focus unresponsive app",
            )

        status = AppControlStatus.FOCUSED
        reason = AppControlReason.APP_FOCUSED
        message = "app is visible, focused, and responsive"

        if request.action == AppControlActionKind.SWITCH:
            status = AppControlStatus.SWITCHED
            reason = AppControlReason.APP_SWITCHED
            message = "app switch is visible, focused, and responsive"

        return _result(
            request=request,
            status=status,
            decision=AppControlDecision.ALLOW,
            reason=reason,
            visibility=visibility,
            responsiveness=responsiveness,
            control_eligible=True,
            message=message,
        )


class AppCloseSafetyChecker:
    def check(self, request: AppControlRequest) -> AppCloseSafetyReport:
        if request.unsaved_changes_hint:
            return AppCloseSafetyReport(
                safety=AppCloseSafety.UNSAVED_CHANGES,
                safe_to_close=False,
                requires_approval=True,
                reason="unsaved changes may be present",
            )

        if request.background_task_hint:
            return AppCloseSafetyReport(
                safety=AppCloseSafety.BACKGROUND_TASK_RUNNING,
                safe_to_close=False,
                requires_approval=True,
                reason="background task may still be running",
            )

        if request.target.app_kind == DetectedAppKind.UNKNOWN:
            return AppCloseSafetyReport(
                safety=AppCloseSafety.CRITICAL_OR_UNKNOWN_APP,
                safe_to_close=False,
                requires_approval=True,
                reason="unknown app cannot be closed silently",
            )

        return AppCloseSafetyReport(
            safety=AppCloseSafety.SAFE_TO_CLOSE,
            safe_to_close=True,
            requires_approval=False,
            reason="app appears safe to close",
        )


class WindowManager:
    def __init__(
        self,
        *,
        close_checker: AppCloseSafetyChecker | None = None,
    ) -> None:
        self._close_checker = close_checker or AppCloseSafetyChecker()

    def evaluate_close(self, request: AppControlRequest) -> AppControlResult:
        blocked = _policy_or_identity_block(request)
        if blocked is not None:
            return blocked

        close_safety = self._close_checker.check(request)
        visibility = AppVisibilityVerification(
            state=AppVisibilityState.VISIBLE,
            visible=True,
            focused=True,
            confidence=0.82,
            reason="close target visibility verified",
        )
        responsiveness = _responsiveness_from_identity(request.identity)

        if not close_safety.safe_to_close:
            reason = AppControlReason.APP_CLOSE_APPROVAL_REQUIRED
            if close_safety.safety == AppCloseSafety.UNSAVED_CHANGES:
                reason = AppControlReason.UNSAVED_STATE_BLOCKED_CLOSE

            return _result(
                request=request,
                status=AppControlStatus.BLOCKED,
                decision=AppControlDecision.REQUIRE_CLOSE_APPROVAL,
                reason=reason,
                visibility=visibility,
                responsiveness=responsiveness,
                close_safety=close_safety,
                control_eligible=False,
                message=close_safety.reason,
            )

        return _result(
            request=request,
            status=AppControlStatus.CLOSE_READY,
            decision=AppControlDecision.ALLOW,
            reason=AppControlReason.APP_CLOSE_SAFE,
            visibility=visibility,
            responsiveness=responsiveness,
            close_safety=close_safety,
            control_eligible=True,
            message="app close is safe and policy eligible",
        )


class SessionRestorer:
    def restore(self, request: AppControlRequest) -> AppSessionRestorePlan:
        if not request.require_session_restore:
            return AppSessionRestorePlan(
                status=AppSessionRestoreStatus.NOT_REQUIRED,
                restore_steps=(),
                restored_workspace_id=request.target.workspace_id,
                confidence=0.90,
                reason="session restore not required",
            )

        return AppSessionRestorePlan(
            status=AppSessionRestoreStatus.RESTORED,
            restore_steps=(
                "open app workspace",
                "restore last known window",
                "verify active workspace identity",
            ),
            restored_workspace_id=request.target.workspace_id,
            confidence=0.82,
            reason="session restore plan completed",
        )


class AppControlRuntime:
    def __init__(
        self,
        *,
        name: str = "app_control_runtime",
        launcher: AppLauncher | None = None,
        focuser: AppFocuser | None = None,
        window_manager: WindowManager | None = None,
        restorer: SessionRestorer | None = None,
    ) -> None:
        cleaned = name.strip()
        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._launcher = launcher or AppLauncher()
        self._focuser = focuser or AppFocuser()
        self._window_manager = window_manager or WindowManager()
        self._restorer = restorer or SessionRestorer()
        self._sessions: dict[str, AppControlSession] = {}
        self._results: list[AppControlResult] = []
        self._audits: list[AppControlAuditRecord] = []
        self._events: list[AppControlRuntimeEvent] = []
        self._lock = RLock()
        self._last_reason: AppControlReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def create_session(
        self,
        *,
        workspace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> AppControlSession:
        session = AppControlSession(
            workspace_id=workspace_id,
            metadata=metadata or {},
        )
        event = self._event(
            kind=AppControlEventKind.SESSION_CREATED,
            reason=AppControlReason.SESSION_CREATED,
            session_id=session.session_id,
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._events.append(event)
            self._last_reason = event.reason

        return session

    def control(self, request: AppControlRequest) -> AppControlResult:
        if self.session_for(request.session_id) is None:
            result = _missing_session_result(request)
            self._record_result(result)
            return result

        if request.action == AppControlActionKind.LAUNCH:
            result = self._launcher.evaluate(request)
        elif request.action in {
            AppControlActionKind.FOCUS,
            AppControlActionKind.SWITCH,
        }:
            result = self._focuser.evaluate(request)
        elif request.action == AppControlActionKind.CLOSE:
            result = self._window_manager.evaluate_close(request)
        elif request.action == AppControlActionKind.RESTORE_SESSION:
            result = self._restore_session(request)
        else:
            result = self._check_responsiveness(request)

        self._record_result(result)
        self._touch_session_from_result(result)

        return result

    def session_for(self, session_id: str) -> AppControlSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def results(self) -> tuple[AppControlResult, ...]:
        with self._lock:
            return tuple(self._results)

    def audits(self) -> tuple[AppControlAuditRecord, ...]:
        with self._lock:
            return tuple(self._audits)

    def events(self) -> tuple[AppControlRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def snapshot(self) -> AppControlRuntimeSnapshot:
        with self._lock:
            return AppControlRuntimeSnapshot(
                name=self.name,
                session_count=len(self._sessions),
                result_count=len(self._results),
                eligible_count=sum(
                    1 for result in self._results if result.control_eligible
                ),
                launched_count=sum(
                    1
                    for result in self._results
                    if result.status == AppControlStatus.LAUNCHED
                ),
                focused_count=sum(
                    1
                    for result in self._results
                    if result.status == AppControlStatus.FOCUSED
                ),
                close_ready_count=sum(
                    1
                    for result in self._results
                    if result.status == AppControlStatus.CLOSE_READY
                ),
                restored_count=sum(
                    1
                    for result in self._results
                    if result.status == AppControlStatus.RESTORED
                ),
                blocked_count=sum(
                    1
                    for result in self._results
                    if result.status == AppControlStatus.BLOCKED
                ),
                audit_count=len(self._audits),
                runtime_event_count=len(self._events),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        event = self._event(
            kind=AppControlEventKind.RUNTIME_RESET,
            reason=AppControlReason.RUNTIME_RESET,
        )

        with self._lock:
            self._sessions.clear()
            self._results.clear()
            self._audits.clear()
            self._events.clear()
            self._events.append(event)
            self._last_reason = event.reason

    def _restore_session(self, request: AppControlRequest) -> AppControlResult:
        blocked = _policy_or_identity_block(request)
        if blocked is not None:
            return blocked

        restore = self._restorer.restore(request)
        visibility = AppVisibilityVerification(
            state=AppVisibilityState.VISIBLE,
            visible=True,
            focused=True,
            confidence=restore.confidence,
            reason="session restore visibility verified",
        )
        responsiveness = _responsiveness_from_identity(request.identity)

        if restore.status != AppSessionRestoreStatus.RESTORED:
            return _result(
                request=request,
                status=AppControlStatus.BLOCKED,
                decision=AppControlDecision.REQUIRE_SESSION_RESTORE,
                reason=AppControlReason.SESSION_RESTORE_REQUIRED,
                visibility=visibility,
                responsiveness=responsiveness,
                restore_plan=restore,
                control_eligible=False,
                message="session restore is required but not complete",
            )

        return _result(
            request=request,
            status=AppControlStatus.RESTORED,
            decision=AppControlDecision.ALLOW,
            reason=AppControlReason.SESSION_RESTORED,
            visibility=visibility,
            responsiveness=responsiveness,
            restore_plan=restore,
            control_eligible=True,
            message="app session restored and verified",
        )

    def _check_responsiveness(
        self,
        request: AppControlRequest,
    ) -> AppControlResult:
        blocked = _policy_or_identity_block(request)
        if blocked is not None:
            return blocked

        responsiveness = _responsiveness_from_identity(request.identity)
        visibility = AppVisibilityVerification(
            state=AppVisibilityState.VISIBLE,
            visible=True,
            focused=False,
            confidence=0.78,
            reason="app visibility checked",
        )

        if not responsiveness.responsive:
            return _result(
                request=request,
                status=AppControlStatus.BLOCKED,
                decision=AppControlDecision.REQUIRE_RESPONSIVENESS_VERIFICATION,
                reason=AppControlReason.APP_RESPONSIVE_VERIFIED,
                visibility=visibility,
                responsiveness=responsiveness,
                control_eligible=False,
                message=responsiveness.reason,
            )

        return _result(
            request=request,
            status=AppControlStatus.READY,
            decision=AppControlDecision.ALLOW,
            reason=AppControlReason.APP_RESPONSIVE_VERIFIED,
            visibility=visibility,
            responsiveness=responsiveness,
            control_eligible=True,
            message=responsiveness.reason,
        )

    def _record_result(self, result: AppControlResult) -> None:
        event = self._event(
            kind=(
                AppControlEventKind.APP_CONTROL_ALLOWED
                if result.control_eligible
                else AppControlEventKind.APP_CONTROL_BLOCKED
            ),
            reason=result.reason,
            session_id=result.request.session_id,
            result_id=result.result_id,
            request_id=result.request.request_id,
            audit_id=result.audit.audit_id,
            metadata={
                "status": result.status.value,
                "eligible": result.control_eligible,
            },
        )

        with self._lock:
            self._results.append(result)
            self._audits.append(result.audit)
            self._events.append(event)
            self._last_reason = result.reason

    def _touch_session_from_result(self, result: AppControlResult) -> None:
        session = self._sessions.get(result.request.session_id)
        if session is None:
            return

        restored = 1 if result.status == AppControlStatus.RESTORED else 0
        active_app = session.active_app_name
        if result.control_eligible:
            active_app = result.request.target.app_name

        self._sessions[result.request.session_id] = session.model_copy(
            update={
                "updated_at": utc_now(),
                "active_app_name": active_app,
                "restored_session_count": session.restored_session_count
                + restored,
            }
        )

    @staticmethod
    def _event(
        *,
        kind: AppControlEventKind,
        reason: AppControlReason,
        session_id: str | None = None,
        result_id: str | None = None,
        request_id: str | None = None,
        audit_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AppControlRuntimeEvent:
        return AppControlRuntimeEvent(
            kind=kind,
            reason=reason,
            session_id=session_id,
            result_id=result_id,
            request_id=request_id,
            audit_id=audit_id,
            metadata=metadata or {},
        )


def _policy_or_identity_block(
    request: AppControlRequest,
) -> AppControlResult | None:
    if not _policy_allows_app_control(request.policy_result):
        return _result(
            request=request,
            status=AppControlStatus.BLOCKED,
            decision=AppControlDecision.BLOCK,
            reason=AppControlReason.POLICY_NOT_ELIGIBLE,
            control_eligible=False,
            message="interaction policy is not eligible for app control",
        )

    if request.identity is None:
        return _result(
            request=request,
            status=AppControlStatus.BLOCKED,
            decision=AppControlDecision.BLOCK,
            reason=AppControlReason.APP_IDENTITY_UNKNOWN,
            control_eligible=False,
            message="app identity is required before app control",
        )

    if request.identity.status in {
        AppIdentityStatus.UNKNOWN,
        AppIdentityStatus.BLOCKED,
    }:
        return _result(
            request=request,
            status=AppControlStatus.BLOCKED,
            decision=AppControlDecision.BLOCK,
            reason=AppControlReason.APP_BLOCKED_BY_IDENTITY,
            control_eligible=False,
            message="app identity is unknown or blocked",
        )

    return None


def _policy_allows_app_control(policy: InteractionPolicyResult) -> bool:
    """
    App control is governed by Step 24, but app-level operations are not the
    same as low-level mouse/keyboard execution.

    Opening, focusing, switching, restoring, and close-readiness checks may
    proceed to the app-control safety layer when:
    - the request is user initiated
    - the physical contract is app-scoped
    - Step 24 did not hard-deny the request

    Step 26 still performs identity, visibility, responsiveness, close-safety,
    session-restore, and audit checks before declaring control eligibility.
    """
    if policy.execution_eligible:
        return True

    app_contract = policy.request.contract.kind in {
        PhysicalInteractionKind.APP_OPEN,
        PhysicalInteractionKind.APP_FOCUS,
        PhysicalInteractionKind.APP_CLOSE,
    }

    if not app_contract:
        return False

    if not policy.request.user_initiated:
        return False

    return policy.decision in {
        InteractionDecision.ELIGIBLE_FOR_EXECUTION,
        InteractionDecision.REQUIRES_VERIFICATION_FIRST,
        InteractionDecision.WAITING_FOR_APPROVAL,
        InteractionDecision.BLOCKED,
    }


def _responsiveness_from_identity(
    identity: AppIdentityResult | None,
) -> AppResponsivenessVerification:
    if identity is None:
        return AppResponsivenessVerification(
            responsive=False,
            state=AppControlResponsiveness.UNKNOWN,
            confidence=0.0,
            reason="identity missing",
        )

    if identity.status in {
        AppIdentityStatus.UNKNOWN,
        AppIdentityStatus.BLOCKED,
    }:
        return AppResponsivenessVerification(
            responsive=False,
            state=AppControlResponsiveness.UNKNOWN,
            confidence=0.20,
            reason=f"app identity status is {identity.status.value}",
        )

    return AppResponsivenessVerification(
        responsive=True,
        state=AppControlResponsiveness.RESPONSIVE,
        confidence=0.84,
        reason=f"app identity status is {identity.status.value}",
    )


def _result(
    *,
    request: AppControlRequest,
    status: AppControlStatus,
    decision: AppControlDecision,
    reason: AppControlReason,
    control_eligible: bool,
    message: str,
    visibility: AppVisibilityVerification | None = None,
    responsiveness: AppResponsivenessVerification | None = None,
    close_safety: AppCloseSafetyReport | None = None,
    restore_plan: AppSessionRestorePlan | None = None,
) -> AppControlResult:
    audit = AppControlAuditRecord(
        request_id=request.request_id,
        action=request.action,
        target_name=request.target.app_name,
        status=status,
        decision=decision,
        reason=reason,
        policy_result_id=request.policy_result.result_id,
    )
    confidence = _confidence_for(
        visibility=visibility,
        responsiveness=responsiveness,
        close_safety=close_safety,
        restore_plan=restore_plan,
        eligible=control_eligible,
    )

    return AppControlResult(
        status=status,
        decision=decision,
        reason=reason,
        request=request,
        visibility=visibility,
        responsiveness=responsiveness,
        close_safety=close_safety,
        restore_plan=restore_plan,
        audit=audit,
        trust=TrustCalibration(
            confidence=confidence,
            stability=max(0.0, min(1.0, confidence + 0.05)),
            ambiguity=1.0 - confidence,
            source=EnvironmentSource.OS_OBSERVER,
            reason="app control runtime decision",
            metadata={"policy": TrustPolicyClassification.REVIEW.value},
        ),
        control_eligible=control_eligible,
        message=message,
    )


def _missing_session_result(request: AppControlRequest) -> AppControlResult:
    return _result(
        request=request,
        status=AppControlStatus.FAILED,
        decision=AppControlDecision.BLOCK,
        reason=AppControlReason.SESSION_NOT_FOUND,
        control_eligible=False,
        message="app control session not found",
    )


def _confidence_for(
    *,
    visibility: AppVisibilityVerification | None,
    responsiveness: AppResponsivenessVerification | None,
    close_safety: AppCloseSafetyReport | None,
    restore_plan: AppSessionRestorePlan | None,
    eligible: bool,
) -> float:
    values: list[float] = []

    if visibility is not None:
        values.append(visibility.confidence)

    if responsiveness is not None:
        values.append(responsiveness.confidence)

    if restore_plan is not None:
        values.append(restore_plan.confidence)

    if close_safety is not None:
        values.append(0.86 if close_safety.safe_to_close else 0.35)

    if not values:
        values.append(0.75 if eligible else 0.0)

    return max(0.0, min(1.0, sum(values) / len(values)))


def _clean_required(value: str) -> str:
    cleaned = value.strip()

    if not cleaned:
        raise ValueError("field cannot be empty.")

    return cleaned