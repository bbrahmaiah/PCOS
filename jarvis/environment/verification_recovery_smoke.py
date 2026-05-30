from __future__ import annotations

from enum import StrEnum
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.environment.models import EnvironmentSource, TrustCalibration
from jarvis.environment.recovery_runtime import (
    RecoveryAttemptKind,
    RecoveryDecision,
    RecoveryHistoryEntry,
    RecoveryRequest,
    RecoveryRuntime,
    RecoveryStatus,
)
from jarvis.environment.trust_runtime import TrustPolicyClassification
from jarvis.environment.undo_rollback import (
    MutatingActionKind,
    ReversibilityContract,
    ReversibilityLevel,
    UndoActionKind,
    UndoRollbackRuntime,
    UndoRollbackStatus,
)
from jarvis.environment.verification_runtime import (
    VerificationContract,
    VerificationDecision,
    VerificationResult,
    VerificationRuntime,
    VerificationStateKind,
    VerificationStatus,
    VerificationTargetKind,
    expected_bool_state,
    expected_hash_state,
    observed_bool_state,
    observed_hash_state,
)
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class VerificationRecoverySmokeCheckKind(StrEnum):
    BUTTON_CLICK_VERIFIED = "button_click_verified"
    TEXT_TYPED_VERIFIED = "text_typed_verified"
    APP_LAUNCH_VERIFIED = "app_launch_verified"
    WRONG_TARGET_DETECTED = "wrong_target_detected"
    PARTIAL_SUCCESS_IS_NOT_SUCCESS = "partial_success_is_not_success"
    DIVERGENCE_DETECTED = "divergence_detected"
    RESYNC_WORKS = "resync_works"
    RECOVERY_RETRIES_SAFELY = "recovery_retries_safely"
    ROLLBACK_WORKS = "rollback_works"
    HALLUCINATED_SUCCESS_IMPOSSIBLE = "hallucinated_success_impossible"


class VerificationRecoverySmokeStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"


class VerificationRecoverySmokeReason(StrEnum):
    CHECK_PASSED = "check_passed"
    CHECK_FAILED = "check_failed"
    GATE_PASSED = "gate_passed"
    GATE_FAILED = "gate_failed"
    SESSION_CREATED = "session_created"
    RUNTIME_RESET = "runtime_reset"


class VerificationRecoverySmokeEventKind(StrEnum):
    SESSION_CREATED = "session_created"
    CHECK_RECORDED = "check_recorded"
    GATE_COMPLETED = "gate_completed"
    RUNTIME_RESET = "runtime_reset"


class VerificationRecoverySmokeCheckResult(OrchestrationModel):
    check_id: str = Field(default_factory=lambda: f"vr_smoke_check_{uuid4().hex}")
    kind: VerificationRecoverySmokeCheckKind
    status: VerificationRecoverySmokeStatus
    reason: VerificationRecoverySmokeReason
    message: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    trust: TrustCalibration
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("check_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class VerificationRecoverySmokeReport(OrchestrationModel):
    report_id: str = Field(default_factory=lambda: f"vr_smoke_report_{uuid4().hex}")
    checks: tuple[VerificationRecoverySmokeCheckResult, ...]
    passed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    sealed: bool
    reason: VerificationRecoverySmokeReason
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("report_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _counts_match_checks(self) -> VerificationRecoverySmokeReport:
        passed = sum(
            1
            for check in self.checks
            if check.status == VerificationRecoverySmokeStatus.PASSED
        )
        failed = sum(
            1
            for check in self.checks
            if check.status == VerificationRecoverySmokeStatus.FAILED
        )
        blocked = sum(
            1
            for check in self.checks
            if check.status == VerificationRecoverySmokeStatus.BLOCKED
        )

        if self.passed_count != passed:
            raise ValueError("passed_count does not match checks.")

        if self.failed_count != failed:
            raise ValueError("failed_count does not match checks.")

        if self.blocked_count != blocked:
            raise ValueError("blocked_count does not match checks.")

        if self.sealed and (failed > 0 or blocked > 0):
            raise ValueError("sealed smoke report cannot contain failures.")

        return self


class VerificationRecoverySmokeSession(OrchestrationModel):
    session_id: str = Field(default_factory=lambda: f"vr_smoke_session_{uuid4().hex}")
    workspace_id: str
    report_count: int = Field(default=0, ge=0)
    check_count: int = Field(default=0, ge=0)
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class VerificationRecoverySmokeRuntimeEvent(OrchestrationModel):
    event_id: str = Field(default_factory=lambda: f"vr_smoke_event_{uuid4().hex}")
    kind: VerificationRecoverySmokeEventKind
    reason: VerificationRecoverySmokeReason
    session_id: str | None = None
    report_id: str | None = None
    check_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class VerificationRecoverySmokeSnapshot(OrchestrationModel):
    name: str
    session_count: int = Field(ge=0)
    report_count: int = Field(ge=0)
    check_count: int = Field(ge=0)
    passed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    sealed_report_count: int = Field(ge=0)
    runtime_event_count: int = Field(ge=0)
    last_reason: VerificationRecoverySmokeReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        return _clean_required(value)


class VerificationRecoverySmokeGateRuntime:
    """
    Phase 8 Step 32 Verification & Recovery Smoke Gate.

    This runtime proves the verification/recovery/rollback chain before later
    real actions trust it.

    It validates:
    - button click verified
    - typed text verified
    - app launch verified
    - wrong target detected
    - partial success is not success
    - divergence detected
    - resync works
    - recovery retries safely
    - rollback works
    - hallucinated success impossible

    Non-responsibilities:
    - does not execute real UI actions
    - does not mutate files
    - does not perform physical input
    """

    def __init__(
        self,
        *,
        name: str = "verification_recovery_smoke_gate",
    ) -> None:
        cleaned = name.strip()
        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._sessions: dict[str, VerificationRecoverySmokeSession] = {}
        self._reports: list[VerificationRecoverySmokeReport] = []
        self._checks: list[VerificationRecoverySmokeCheckResult] = []
        self._events: list[VerificationRecoverySmokeRuntimeEvent] = []
        self._lock = RLock()
        self._last_reason: VerificationRecoverySmokeReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def create_session(
        self,
        *,
        workspace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> VerificationRecoverySmokeSession:
        session = VerificationRecoverySmokeSession(
            workspace_id=workspace_id,
            metadata=metadata or {},
        )
        event = self._event(
            kind=VerificationRecoverySmokeEventKind.SESSION_CREATED,
            reason=VerificationRecoverySmokeReason.SESSION_CREATED,
            session_id=session.session_id,
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._events.append(event)
            self._last_reason = event.reason

        return session

    def run(
        self,
        *,
        session_id: str,
    ) -> VerificationRecoverySmokeReport:
        if self.session_for(session_id) is None:
            check = _check(
                kind=(
                    VerificationRecoverySmokeCheckKind
                    .HALLUCINATED_SUCCESS_IMPOSSIBLE
                ),
                passed=False,
                message="smoke session not found",
                blocked=True,
            )
            report = self._report_from_checks((check,))
            self._record_report(report, session_id)
            return report

        checks = (
            self.check_button_click_verified(),
            self.check_text_typed_verified(),
            self.check_app_launch_verified(),
            self.check_wrong_target_detected(),
            self.check_partial_success_is_not_success(),
            self.check_divergence_detected(),
            self.check_resync_works(),
            self.check_recovery_retries_safely(),
            self.check_rollback_works(),
            self.check_hallucinated_success_impossible(),
        )
        report = self._report_from_checks(checks)
        self._record_report(report, session_id)

        return report

    def check_button_click_verified(self) -> VerificationRecoverySmokeCheckResult:
        result = _verify_bool(
            key="button.run.clicked",
            expected=True,
            observed=True,
            kind=VerificationStateKind.STATUS_EQUALS,
            target=VerificationTargetKind.UI_ELEMENT,
            description="button click should be verified",
        )

        return _check(
            kind=VerificationRecoverySmokeCheckKind.BUTTON_CLICK_VERIFIED,
            passed=result.action_complete,
            message="button click verification passed",
            metadata={"verification_status": result.status.value},
        )

    def check_text_typed_verified(self) -> VerificationRecoverySmokeCheckResult:
        runtime = VerificationRuntime()
        session = runtime.create_session(workspace_id="workspace")
        contract = VerificationContract(
            action_id="type_action",
            workspace_id="workspace",
            expected_states=(
                expected_hash_state(
                    key="field.text",
                    value="hello jarvis",
                    kind=VerificationStateKind.HASH_EQUALS,
                    target=VerificationTargetKind.UI_ELEMENT,
                    description="typed text hash should match",
                ),
            ),
        )
        result = runtime.verify(
            session_id=session.session_id,
            contract=contract,
            observed_states=(
                observed_hash_state(
                    key="field.text",
                    value="hello jarvis",
                    kind=VerificationStateKind.HASH_EQUALS,
                    target=VerificationTargetKind.UI_ELEMENT,
                    description="typed text hash matched",
                ),
            ),
        )

        return _check(
            kind=VerificationRecoverySmokeCheckKind.TEXT_TYPED_VERIFIED,
            passed=result.action_complete,
            message="typed text verification passed",
            metadata={"verification_status": result.status.value},
        )

    def check_app_launch_verified(self) -> VerificationRecoverySmokeCheckResult:
        result = _verify_bool(
            key="app.visible",
            expected=True,
            observed=True,
            kind=VerificationStateKind.VISIBLE,
            target=VerificationTargetKind.APP,
            description="app launch should be visible",
        )

        return _check(
            kind=VerificationRecoverySmokeCheckKind.APP_LAUNCH_VERIFIED,
            passed=result.action_complete,
            message="app launch verification passed",
            metadata={"verification_status": result.status.value},
        )

    def check_wrong_target_detected(self) -> VerificationRecoverySmokeCheckResult:
        runtime = VerificationRuntime()
        session = runtime.create_session(workspace_id="workspace")
        contract = VerificationContract(
            action_id="wrong_target_action",
            workspace_id="workspace",
            expected_states=(
                expected_bool_state(
                    key="target.clicked",
                    value=True,
                    kind=VerificationStateKind.STATUS_EQUALS,
                    target=VerificationTargetKind.UI_ELEMENT,
                    description="correct UI element should be clicked",
                ),
            ),
        )
        result = runtime.verify(
            session_id=session.session_id,
            contract=contract,
            observed_states=(
                observed_bool_state(
                    key="target.clicked",
                    value=True,
                    kind=VerificationStateKind.STATUS_EQUALS,
                    target=VerificationTargetKind.WINDOW,
                    description="wrong target observed",
                ),
            ),
        )

        passed = (
            result.status == VerificationStatus.RECOVERY_NEEDED
            and result.recovery_needed
            and not result.action_complete
        )

        return _check(
            kind=VerificationRecoverySmokeCheckKind.WRONG_TARGET_DETECTED,
            passed=passed,
            message="wrong target was detected and blocked from success",
            metadata={"verification_status": result.status.value},
        )

    def check_partial_success_is_not_success(
        self,
    ) -> VerificationRecoverySmokeCheckResult:
        recovery = RecoveryRuntime()
        session = recovery.create_session(workspace_id="workspace")
        result = recovery.recover(
            RecoveryRequest(
                session_id=session.session_id,
                workspace_id="workspace",
                action_id="partial_action",
                verification=_recovery_needed_verification(),
                history=_exhausted_history(),
                partial_success_available=True,
                rollback_available=True,
            )
        )

        passed = (
            result.decision == RecoveryDecision.REPORT_PARTIAL_SUCCESS
            and result.status == RecoveryStatus.PLANNED
            and not result.retry_allowed
            and not result.rollback_required
        )

        return _check(
            kind=(
                VerificationRecoverySmokeCheckKind
                .PARTIAL_SUCCESS_IS_NOT_SUCCESS
            ),
            passed=passed,
            message="partial success is reported without pretending completion",
            metadata={"recovery_decision": result.decision.value},
        )

    def check_divergence_detected(self) -> VerificationRecoverySmokeCheckResult:
        runtime = VerificationRuntime()
        session = runtime.create_session(workspace_id="workspace")
        contract = VerificationContract(
            action_id="divergence_action",
            workspace_id="workspace",
            expected_states=(
                expected_hash_state(
                    key="workspace.state",
                    value="expected-state",
                    kind=VerificationStateKind.HASH_EQUALS,
                    target=VerificationTargetKind.WORKSPACE_GRAPH,
                    description="workspace state should match expectation",
                ),
            ),
        )
        result = runtime.verify(
            session_id=session.session_id,
            contract=contract,
            observed_states=(
                observed_hash_state(
                    key="workspace.state",
                    value="observed-divergent-state",
                    kind=VerificationStateKind.HASH_EQUALS,
                    target=VerificationTargetKind.WORKSPACE_GRAPH,
                    description="workspace diverged from belief",
                ),
            ),
        )

        passed = result.recovery_needed and not result.action_complete

        return _check(
            kind=VerificationRecoverySmokeCheckKind.DIVERGENCE_DETECTED,
            passed=passed,
            message="divergence was detected through verification mismatch",
            metadata={"verification_status": result.status.value},
        )

    def check_resync_works(self) -> VerificationRecoverySmokeCheckResult:
        runtime = VerificationRuntime()
        session = runtime.create_session(workspace_id="workspace")

        first_contract = VerificationContract(
            action_id="resync_action",
            workspace_id="workspace",
            expected_states=(
                expected_hash_state(
                    key="workspace.state",
                    value="old-belief",
                    kind=VerificationStateKind.HASH_EQUALS,
                    target=VerificationTargetKind.WORKSPACE_GRAPH,
                    description="old belief should match",
                ),
            ),
        )
        first = runtime.verify(
            session_id=session.session_id,
            contract=first_contract,
            observed_states=(
                observed_hash_state(
                    key="workspace.state",
                    value="new-reality",
                    kind=VerificationStateKind.HASH_EQUALS,
                    target=VerificationTargetKind.WORKSPACE_GRAPH,
                    description="new reality observed",
                ),
            ),
        )

        resynced_contract = VerificationContract(
            action_id="resync_action",
            workspace_id="workspace",
            expected_states=(
                expected_hash_state(
                    key="workspace.state",
                    value="new-reality",
                    kind=VerificationStateKind.HASH_EQUALS,
                    target=VerificationTargetKind.WORKSPACE_GRAPH,
                    description="belief updated to reality",
                ),
            ),
        )
        second = runtime.verify(
            session_id=session.session_id,
            contract=resynced_contract,
            observed_states=(
                observed_hash_state(
                    key="workspace.state",
                    value="new-reality",
                    kind=VerificationStateKind.HASH_EQUALS,
                    target=VerificationTargetKind.WORKSPACE_GRAPH,
                    description="resynced reality verified",
                ),
            ),
        )

        passed = first.recovery_needed and second.action_complete

        return _check(
            kind=VerificationRecoverySmokeCheckKind.RESYNC_WORKS,
            passed=passed,
            message="resync updated belief to observed reality",
            metadata={
                "before": first.status.value,
                "after": second.status.value,
            },
        )

    def check_recovery_retries_safely(self) -> VerificationRecoverySmokeCheckResult:
        recovery = RecoveryRuntime()
        session = recovery.create_session(workspace_id="workspace")
        result = recovery.recover(
            RecoveryRequest(
                session_id=session.session_id,
                workspace_id="workspace",
                action_id="retry_action",
                verification=_recovery_needed_verification(),
            )
        )

        passed = (
            result.decision == RecoveryDecision.RETRY
            and result.retry_allowed
            and result.plan is not None
            and result.plan.selected_strategy.kind
            == RecoveryAttemptKind.RETRY_SAME
        )

        return _check(
            kind=VerificationRecoverySmokeCheckKind.RECOVERY_RETRIES_SAFELY,
            passed=passed,
            message="recovery selected bounded safe retry",
            metadata={"recovery_reason": result.reason.value},
        )

    def check_rollback_works(self) -> VerificationRecoverySmokeCheckResult:
        undo = UndoRollbackRuntime()
        undo_session = undo.create_session(workspace_id="workspace")
        undo.declare_undo(
            session_id=undo_session.session_id,
            contract=ReversibilityContract(
                action_id="rollback_action",
                workspace_id="workspace",
                mutation_kind=MutatingActionKind.TEXT_EDIT,
                reversibility=ReversibilityLevel.REVERSIBLE,
                undo_kind=UndoActionKind.TEXT_REVERT,
                undo_description="restore previous text snapshot",
                backup_required=True,
                backup_reference="backup://rollback_action",
            ),
        )

        recovery = RecoveryRuntime()
        recovery_session = recovery.create_session(workspace_id="workspace")
        recovery_result = recovery.recover(
            RecoveryRequest(
                session_id=recovery_session.session_id,
                workspace_id="workspace",
                action_id="rollback_action",
                verification=_recovery_needed_verification(),
                history=_exhausted_history(),
                rollback_available=True,
            )
        )

        verification_contract = VerificationContract(
            action_id="rollback_action",
            workspace_id="workspace",
            expected_states=(
                expected_bool_state(
                    key="rollback.verified",
                    value=True,
                    kind=VerificationStateKind.STATUS_EQUALS,
                    target=VerificationTargetKind.WORKSPACE_GRAPH,
                    description="rollback must be verified",
                ),
            ),
        )
        planned = undo.plan_rollback(
            session_id=undo_session.session_id,
            recovery=recovery_result,
            verification_contract=verification_contract,
        )

        verification = _verify_bool(
            key="rollback.verified",
            expected=True,
            observed=True,
            kind=VerificationStateKind.STATUS_EQUALS,
            target=VerificationTargetKind.WORKSPACE_GRAPH,
            description="rollback verified",
        )
        verified = undo.verify_rollback(
            session_id=undo_session.session_id,
            rollback_result=planned,
            verification_result=verification,
        )

        passed = (
            planned.rollback_ready
            and verified.rollback_verified
            and verified.status == UndoRollbackStatus.DECLARED
        )

        return _check(
            kind=VerificationRecoverySmokeCheckKind.ROLLBACK_WORKS,
            passed=passed,
            message="rollback plan was created and verified",
            metadata={
                "planned_status": planned.status.value,
                "verified_status": verified.status.value,
            },
        )

    def check_hallucinated_success_impossible(
        self,
    ) -> VerificationRecoverySmokeCheckResult:
        result = _verify_bool(
            key="file.saved",
            expected=True,
            observed=False,
            kind=VerificationStateKind.STATUS_EQUALS,
            target=VerificationTargetKind.FILE,
            description="file save must be verified",
        )

        passed = (
            result.status == VerificationStatus.RECOVERY_NEEDED
            and not result.action_complete
            and result.decision == VerificationDecision.REQUIRE_RECOVERY
        )

        return _check(
            kind=(
                VerificationRecoverySmokeCheckKind
                .HALLUCINATED_SUCCESS_IMPOSSIBLE
            ),
            passed=passed,
            message="mismatched state cannot be marked complete",
            metadata={"verification_status": result.status.value},
        )

    def session_for(
        self,
        session_id: str,
    ) -> VerificationRecoverySmokeSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def reports(self) -> tuple[VerificationRecoverySmokeReport, ...]:
        with self._lock:
            return tuple(self._reports)

    def checks(self) -> tuple[VerificationRecoverySmokeCheckResult, ...]:
        with self._lock:
            return tuple(self._checks)

    def events(self) -> tuple[VerificationRecoverySmokeRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def snapshot(self) -> VerificationRecoverySmokeSnapshot:
        with self._lock:
            return VerificationRecoverySmokeSnapshot(
                name=self.name,
                session_count=len(self._sessions),
                report_count=len(self._reports),
                check_count=len(self._checks),
                passed_count=sum(
                    1
                    for check in self._checks
                    if check.status == VerificationRecoverySmokeStatus.PASSED
                ),
                failed_count=sum(
                    1
                    for check in self._checks
                    if check.status == VerificationRecoverySmokeStatus.FAILED
                ),
                blocked_count=sum(
                    1
                    for check in self._checks
                    if check.status == VerificationRecoverySmokeStatus.BLOCKED
                ),
                sealed_report_count=sum(
                    1 for report in self._reports if report.sealed
                ),
                runtime_event_count=len(self._events),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        event = self._event(
            kind=VerificationRecoverySmokeEventKind.RUNTIME_RESET,
            reason=VerificationRecoverySmokeReason.RUNTIME_RESET,
        )

        with self._lock:
            self._sessions.clear()
            self._reports.clear()
            self._checks.clear()
            self._events.clear()
            self._events.append(event)
            self._last_reason = event.reason

    def _report_from_checks(
        self,
        checks: tuple[VerificationRecoverySmokeCheckResult, ...],
    ) -> VerificationRecoverySmokeReport:
        passed = sum(
            1
            for check in checks
            if check.status == VerificationRecoverySmokeStatus.PASSED
        )
        failed = sum(
            1
            for check in checks
            if check.status == VerificationRecoverySmokeStatus.FAILED
        )
        blocked = sum(
            1
            for check in checks
            if check.status == VerificationRecoverySmokeStatus.BLOCKED
        )
        sealed = failed == 0 and blocked == 0

        return VerificationRecoverySmokeReport(
            checks=checks,
            passed_count=passed,
            failed_count=failed,
            blocked_count=blocked,
            sealed=sealed,
            reason=(
                VerificationRecoverySmokeReason.GATE_PASSED
                if sealed
                else VerificationRecoverySmokeReason.GATE_FAILED
            ),
            message=(
                "verification recovery smoke gate passed"
                if sealed
                else "verification recovery smoke gate failed"
            ),
        )

    def _record_report(
        self,
        report: VerificationRecoverySmokeReport,
        session_id: str,
    ) -> None:
        event = self._event(
            kind=VerificationRecoverySmokeEventKind.GATE_COMPLETED,
            reason=report.reason,
            session_id=session_id,
            report_id=report.report_id,
            metadata={
                "sealed": report.sealed,
                "passed": report.passed_count,
                "failed": report.failed_count,
                "blocked": report.blocked_count,
            },
        )

        with self._lock:
            self._reports.append(report)
            self._checks.extend(report.checks)
            self._events.append(event)
            self._last_reason = report.reason

            session = self._sessions.get(session_id)
            if session is not None:
                self._sessions[session_id] = session.model_copy(
                    update={
                        "updated_at": utc_now(),
                        "report_count": session.report_count + 1,
                        "check_count": session.check_count + len(report.checks),
                    }
                )

    @staticmethod
    def _event(
        *,
        kind: VerificationRecoverySmokeEventKind,
        reason: VerificationRecoverySmokeReason,
        session_id: str | None = None,
        report_id: str | None = None,
        check_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VerificationRecoverySmokeRuntimeEvent:
        return VerificationRecoverySmokeRuntimeEvent(
            kind=kind,
            reason=reason,
            session_id=session_id,
            report_id=report_id,
            check_id=check_id,
            metadata=metadata or {},
        )


def _verify_bool(
    *,
    key: str,
    expected: bool,
    observed: bool,
    kind: VerificationStateKind,
    target: VerificationTargetKind,
    description: str,
) -> VerificationResult:
    runtime = VerificationRuntime()
    session = runtime.create_session(workspace_id="workspace")
    contract = VerificationContract(
        action_id=f"verify_{key}",
        workspace_id="workspace",
        expected_states=(
            expected_bool_state(
                key=key,
                value=expected,
                kind=kind,
                target=target,
                description=description,
            ),
        ),
    )

    return runtime.verify(
        session_id=session.session_id,
        contract=contract,
        observed_states=(
            observed_bool_state(
                key=key,
                value=observed,
                kind=kind,
                target=target,
                description=description,
            ),
        ),
    )


def _recovery_needed_verification() -> VerificationResult:
    return _verify_bool(
        key="action.completed",
        expected=True,
        observed=False,
        kind=VerificationStateKind.STATUS_EQUALS,
        target=VerificationTargetKind.WORKSPACE_GRAPH,
        description="action should have completed",
    )


def _exhausted_history() -> tuple[RecoveryHistoryEntry, ...]:
    return (
        RecoveryHistoryEntry(
            verification_result_id="verification_result_test",
            strategy_kind=RecoveryAttemptKind.RETRY_SAME,
            success=False,
            reason="same retry failed",
        ),
        RecoveryHistoryEntry(
            verification_result_id="verification_result_test",
            strategy_kind=RecoveryAttemptKind.RETRY_ADJUSTED,
            success=False,
            reason="adjusted retry failed",
        ),
        RecoveryHistoryEntry(
            verification_result_id="verification_result_test",
            strategy_kind=RecoveryAttemptKind.RETRY_ADJUSTED,
            success=False,
            reason="adjusted retry failed again",
        ),
        RecoveryHistoryEntry(
            verification_result_id="verification_result_test",
            strategy_kind=RecoveryAttemptKind.ALTERNATIVE_PATH,
            success=False,
            reason="alternative path failed",
        ),
    )


def _check(
    *,
    kind: VerificationRecoverySmokeCheckKind,
    passed: bool,
    message: str,
    blocked: bool = False,
    metadata: dict[str, Any] | None = None,
) -> VerificationRecoverySmokeCheckResult:
    status = VerificationRecoverySmokeStatus.PASSED
    if blocked:
        status = VerificationRecoverySmokeStatus.BLOCKED
    elif not passed:
        status = VerificationRecoverySmokeStatus.FAILED

    return VerificationRecoverySmokeCheckResult(
        kind=kind,
        status=status,
        reason=(
            VerificationRecoverySmokeReason.CHECK_PASSED
            if passed and not blocked
            else VerificationRecoverySmokeReason.CHECK_FAILED
        ),
        message=message,
        confidence=1.0 if passed else 0.0,
        trust=TrustCalibration(
            confidence=1.0 if passed else 0.0,
            stability=1.0 if passed else 0.0,
            ambiguity=0.0 if passed else 1.0,
            source=EnvironmentSource.OS_OBSERVER,
            reason=message,
            metadata={"policy": TrustPolicyClassification.REVIEW.value},
        ),
        metadata=metadata or {},
    )


def _clean_required(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("field cannot be empty.")
    return cleaned