from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from threading import RLock

from pydantic import Field, field_validator, model_validator

from jarvis.tools.audit import (
    ActionAuditActor,
    ActionAuditEventKind,
    ActionAuditLog,
    ActionAuditOutcome,
)
from jarvis.tools.ids import new_action_id, new_action_result_id, utc_now
from jarvis.tools.models import (
    ActionKind,
    ActionRisk,
    ActionScope,
    PermissionDecision,
    ToolCapability,
    ToolModel,
)


class ApprovalRequirement(StrEnum):
    """
    Required human approval level.
    """

    NONE = "none"
    SOFT_CONFIRMATION = "soft_confirmation"
    EXPLICIT_APPROVAL = "explicit_approval"
    ADMIN_APPROVAL = "admin_approval"
    BLOCKED = "blocked"


class ApprovalDecision(StrEnum):
    """
    Human approval decision state.
    """

    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"
    BLOCKED = "blocked"


class ApprovalScope(StrEnum):
    """
    Scope of approval.

    Narrow approvals are safer. Broad approvals must be rare and auditable.
    """

    ONE_TIME = "one_time"
    SESSION = "session"
    ACTION_KIND = "action_kind"
    TOOL_CAPABILITY = "tool_capability"
    RUNTIME = "runtime"


class ApprovalReason(StrEnum):
    """
    Machine-readable approval reason.
    """

    NO_APPROVAL_REQUIRED = "no_approval_required"
    SOFT_CONFIRMATION_REQUIRED = "soft_confirmation_required"
    EXPLICIT_APPROVAL_REQUIRED = "explicit_approval_required"
    ADMIN_APPROVAL_REQUIRED = "admin_approval_required"
    ACTION_BLOCKED = "action_blocked"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"
    APPROVAL_EXPIRED = "approval_expired"
    APPROVAL_NOT_FOUND = "approval_not_found"
    APPROVAL_SCOPE_MISMATCH = "approval_scope_mismatch"
    ADMIN_REQUIRED = "admin_required"
    SESSION_APPROVAL_ACTIVE = "session_approval_active"
    ONE_TIME_APPROVAL_CONSUMED = "one_time_approval_consumed"


class ApprovalEvaluation(ToolModel):
    """
    Result of evaluating whether an action requires approval.
    """

    evaluation_id: str = Field(default_factory=new_action_result_id)
    action_id: str
    requirement: ApprovalRequirement
    decision: ApprovalDecision
    reason: ApprovalReason
    risk: ActionRisk
    permission_decision: PermissionDecision
    message: str
    requires_human: bool = False
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("evaluation_id", "action_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class ApprovalRequest(ToolModel):
    """
    First-class approval request.

    This is what JARVIS should create before high-risk execution.
    """

    approval_id: str = Field(default_factory=new_action_result_id)
    action_id: str = Field(default_factory=new_action_id)
    requirement: ApprovalRequirement
    scope: ApprovalScope = ApprovalScope.ONE_TIME
    risk: ActionRisk
    reason: ApprovalReason
    message: str
    requested_by: str = "jarvis"
    action_kind: ActionKind | None = None
    capability: ToolCapability | None = None
    action_scope: ActionScope | None = None
    runtime_name: str | None = None
    expires_at: datetime
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("approval_id", "action_id", "message", "requested_by")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("runtime_name")
    @classmethod
    def _clean_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None

    @model_validator(mode="after")
    def _validate_blocked_request(self) -> ApprovalRequest:
        if self.requirement == ApprovalRequirement.NONE:
            raise ValueError("approval requests cannot use NONE requirement.")

        if self.requirement == ApprovalRequirement.BLOCKED:
            raise ValueError("blocked actions must not create approval requests.")

        return self

    @property
    def expired(self) -> bool:
        return utc_now() >= self.expires_at


class ApprovalRecord(ToolModel):
    """
    Immutable approval decision record.
    """

    record_id: str = Field(default_factory=new_action_result_id)
    approval_id: str
    action_id: str
    decision: ApprovalDecision
    requirement: ApprovalRequirement
    scope: ApprovalScope
    reason: ApprovalReason
    decided_by: str
    evidence: str
    approved_until: datetime | None = None
    action_kind: ActionKind | None = None
    capability: ToolCapability | None = None
    action_scope: ActionScope | None = None
    runtime_name: str | None = None
    consumed: bool = False
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator(
        "record_id",
        "approval_id",
        "action_id",
        "decided_by",
        "evidence",
    )
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("runtime_name")
    @classmethod
    def _clean_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None

    @property
    def approved(self) -> bool:
        return self.decision == ApprovalDecision.APPROVED

    @property
    def expired(self) -> bool:
        if self.approved_until is None:
            return False

        return utc_now() >= self.approved_until


class ApprovalCheckResult(ToolModel):
    """
    Result of checking whether an action has valid approval.
    """

    check_id: str = Field(default_factory=new_action_result_id)
    action_id: str
    decision: ApprovalDecision
    reason: ApprovalReason
    approved: bool
    approval_id: str | None = None
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("check_id", "action_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class HumanApprovalRuntimeConfig:
    """
    Human approval runtime configuration.
    """

    name: str = "human_approval_runtime"
    default_expiration_minutes: int = 10
    session_expiration_minutes: int = 60
    allow_session_approvals: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.default_expiration_minutes <= 0:
            raise ValueError("default_expiration_minutes must be positive.")

        if self.session_expiration_minutes <= 0:
            raise ValueError("session_expiration_minutes must be positive.")


@dataclass(frozen=True, slots=True)
class HumanApprovalRuntimeSnapshot:
    """
    Human approval runtime diagnostics.
    """

    name: str
    request_count: int
    approval_count: int
    denial_count: int
    blocked_count: int
    expired_count: int
    active_approval_count: int
    last_decision: ApprovalDecision | None
    last_reason: ApprovalReason | None
    last_error: str | None


class HumanApprovalRuntime:
    """
    First-class human approval runtime.

    Responsibilities:
    - evaluate approval requirements from risk and permission
    - create expiring approval requests
    - record approvals and denials with evidence
    - support one-time and session approvals
    - audit approval lifecycle events
    - prevent high-risk action bypass

    Non-responsibilities:
    - no action execution
    - no policy replacement
    - no validation replacement
    - no UI implementation
    """

    def __init__(
        self,
        *,
        config: HumanApprovalRuntimeConfig | None = None,
        audit_log: ActionAuditLog | None = None,
    ) -> None:
        self._config = config or HumanApprovalRuntimeConfig()
        self._config.validate()

        self._audit_log = audit_log
        self._lock = RLock()

        self._requests: dict[str, ApprovalRequest] = {}
        self._records: dict[str, ApprovalRecord] = {}

        self._request_count = 0
        self._approval_count = 0
        self._denial_count = 0
        self._blocked_count = 0
        self._expired_count = 0
        self._last_decision: ApprovalDecision | None = None
        self._last_reason: ApprovalReason | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def evaluate(
        self,
        *,
        action_id: str,
        risk: ActionRisk,
        permission_decision: PermissionDecision,
    ) -> ApprovalEvaluation:
        """
        Evaluate whether an action requires human approval.
        """

        requirement, decision, reason, requires_human = self._requirement_for(
            risk=risk,
            permission_decision=permission_decision,
        )

        message = self._message_for(
            requirement=requirement,
            risk=risk,
            permission_decision=permission_decision,
        )

        evaluation = ApprovalEvaluation(
            action_id=action_id,
            requirement=requirement,
            decision=decision,
            reason=reason,
            risk=risk,
            permission_decision=permission_decision,
            message=message,
            requires_human=requires_human,
            metadata={"runtime": self.name},
        )

        with self._lock:
            self._last_decision = decision
            self._last_reason = reason

            if decision == ApprovalDecision.BLOCKED:
                self._blocked_count += 1

        return evaluation

    def request_approval(
        self,
        *,
        action_id: str,
        requirement: ApprovalRequirement,
        risk: ActionRisk,
        reason: ApprovalReason,
        message: str,
        scope: ApprovalScope = ApprovalScope.ONE_TIME,
        requested_by: str = "jarvis",
        action_kind: ActionKind | None = None,
        capability: ToolCapability | None = None,
        action_scope: ActionScope | None = None,
        runtime_name: str | None = None,
        expires_in_minutes: int | None = None,
        metadata: dict[str, object] | None = None,
    ) -> ApprovalRequest:
        """
        Create an expiring approval request.

        BLOCKED and NONE requirements cannot become approval requests.
        """

        if scope == ApprovalScope.SESSION and not self._config.allow_session_approvals:
            raise ValueError("session approvals are disabled.")

        minutes = expires_in_minutes or (
            self._config.session_expiration_minutes
            if scope == ApprovalScope.SESSION
            else self._config.default_expiration_minutes
        )
        expires_at = utc_now() + timedelta(minutes=minutes)

        request = ApprovalRequest(
            action_id=action_id,
            requirement=requirement,
            scope=scope,
            risk=risk,
            reason=reason,
            message=message,
            requested_by=requested_by,
            action_kind=action_kind,
            capability=capability,
            action_scope=action_scope,
            runtime_name=runtime_name,
            expires_at=expires_at,
            metadata=metadata or {},
        )

        with self._lock:
            self._requests[request.approval_id] = request
            self._request_count += 1
            self._last_decision = ApprovalDecision.PENDING
            self._last_reason = ApprovalReason.APPROVAL_REQUESTED

        self._audit(
            action_id=action_id,
            event_kind=ActionAuditEventKind.APPROVAL_REQUESTED,
            outcome=ActionAuditOutcome.APPROVAL_REQUIRED,
            message=message,
            risk=risk,
            data={
                "approval_id": request.approval_id,
                "requirement": requirement.value,
                "scope": scope.value,
                "expires_at": expires_at.isoformat(),
            },
        )

        return request

    def approve(
        self,
        *,
        approval_id: str,
        decided_by: str,
        evidence: str,
        admin: bool = False,
    ) -> ApprovalRecord:
        """
        Approve an approval request.
        """

        with self._lock:
            request = self._requests.get(approval_id)

        if request is None:
            raise ValueError("approval request not found.")

        if request.expired:
            with self._lock:
                self._expired_count += 1
                self._last_decision = ApprovalDecision.EXPIRED
                self._last_reason = ApprovalReason.APPROVAL_EXPIRED

            raise ValueError("approval request expired.")

        if request.requirement == ApprovalRequirement.ADMIN_APPROVAL and not admin:
            raise ValueError("admin approval is required.")

        approved_until = (
            request.expires_at if request.scope == ApprovalScope.SESSION else None
        )

        record = ApprovalRecord(
            approval_id=request.approval_id,
            action_id=request.action_id,
            decision=ApprovalDecision.APPROVED,
            requirement=request.requirement,
            scope=request.scope,
            reason=ApprovalReason.APPROVAL_GRANTED,
            decided_by=decided_by,
            evidence=evidence,
            approved_until=approved_until,
            action_kind=request.action_kind,
            capability=request.capability,
            action_scope=request.action_scope,
            runtime_name=request.runtime_name,
            metadata={
                "admin": admin,
                "runtime": self.name,
            },
        )

        with self._lock:
            self._records[record.approval_id] = record
            self._approval_count += 1
            self._last_decision = ApprovalDecision.APPROVED
            self._last_reason = ApprovalReason.APPROVAL_GRANTED

        self._audit(
            action_id=request.action_id,
            event_kind=ActionAuditEventKind.APPROVAL_GRANTED,
            outcome=ActionAuditOutcome.ALLOW,
            message="human approval granted",
            risk=request.risk,
            data={
                "approval_id": approval_id,
                "decided_by": decided_by,
                "scope": request.scope.value,
                "requirement": request.requirement.value,
            },
        )

        return record

    def deny(
        self,
        *,
        approval_id: str,
        decided_by: str,
        evidence: str,
        reason: str,
    ) -> ApprovalRecord:
        """
        Deny an approval request.
        """

        with self._lock:
            request = self._requests.get(approval_id)

        if request is None:
            raise ValueError("approval request not found.")

        record = ApprovalRecord(
            approval_id=request.approval_id,
            action_id=request.action_id,
            decision=ApprovalDecision.DENIED,
            requirement=request.requirement,
            scope=request.scope,
            reason=ApprovalReason.APPROVAL_DENIED,
            decided_by=decided_by,
            evidence=evidence,
            action_kind=request.action_kind,
            capability=request.capability,
            action_scope=request.action_scope,
            runtime_name=request.runtime_name,
            metadata={
                "denial_reason": reason,
                "runtime": self.name,
            },
        )

        with self._lock:
            self._records[record.approval_id] = record
            self._denial_count += 1
            self._last_decision = ApprovalDecision.DENIED
            self._last_reason = ApprovalReason.APPROVAL_DENIED

        self._audit(
            action_id=request.action_id,
            event_kind=ActionAuditEventKind.APPROVAL_DENIED,
            outcome=ActionAuditOutcome.DENY,
            message=reason,
            risk=request.risk,
            data={
                "approval_id": approval_id,
                "decided_by": decided_by,
            },
        )

        return record

    def check_approval(
        self,
        *,
        action_id: str,
        action_kind: ActionKind | None = None,
        capability: ToolCapability | None = None,
        action_scope: ActionScope | None = None,
        runtime_name: str | None = None,
    ) -> ApprovalCheckResult:
        """
        Check whether an action has valid approval.

        One-time approvals are consumed after a successful check.
        """

        with self._lock:
            candidates = tuple(self._records.values())

        for record in candidates:
            if not record.approved:
                continue

            if record.expired:
                continue

            if record.consumed and record.scope == ApprovalScope.ONE_TIME:
                continue

            if not self._record_matches(
                record=record,
                action_id=action_id,
                action_kind=action_kind,
                capability=capability,
                action_scope=action_scope,
                runtime_name=runtime_name,
            ):
                continue

            if record.scope == ApprovalScope.ONE_TIME:
                self._consume(record.approval_id)

                return ApprovalCheckResult(
                    action_id=action_id,
                    decision=ApprovalDecision.APPROVED,
                    reason=ApprovalReason.ONE_TIME_APPROVAL_CONSUMED,
                    approved=True,
                    approval_id=record.approval_id,
                    message="one-time approval is valid and consumed",
                )

            return ApprovalCheckResult(
                action_id=action_id,
                decision=ApprovalDecision.APPROVED,
                reason=ApprovalReason.SESSION_APPROVAL_ACTIVE,
                approved=True,
                approval_id=record.approval_id,
                message="session approval is active",
            )

        return ApprovalCheckResult(
            action_id=action_id,
            decision=ApprovalDecision.DENIED,
            reason=ApprovalReason.APPROVAL_NOT_FOUND,
            approved=False,
            message="no valid approval found",
        )

    def approval_request(self, approval_id: str) -> ApprovalRequest | None:
        with self._lock:
            return self._requests.get(approval_id)

    def approval_record(self, approval_id: str) -> ApprovalRecord | None:
        with self._lock:
            return self._records.get(approval_id)

    def active_approvals(self) -> tuple[ApprovalRecord, ...]:
        with self._lock:
            return tuple(
                record
                for record in self._records.values()
                if record.approved and not record.expired and not record.consumed
            )

    def snapshot(self) -> HumanApprovalRuntimeSnapshot:
        with self._lock:
            return HumanApprovalRuntimeSnapshot(
                name=self.name,
                request_count=self._request_count,
                approval_count=self._approval_count,
                denial_count=self._denial_count,
                blocked_count=self._blocked_count,
                expired_count=self._expired_count,
                active_approval_count=len(self.active_approvals()),
                last_decision=self._last_decision,
                last_reason=self._last_reason,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        with self._lock:
            self._requests.clear()
            self._records.clear()
            self._request_count = 0
            self._approval_count = 0
            self._denial_count = 0
            self._blocked_count = 0
            self._expired_count = 0
            self._last_decision = None
            self._last_reason = None
            self._last_error = None

    def _consume(self, approval_id: str) -> None:
        with self._lock:
            record = self._records.get(approval_id)

            if record is None:
                return

            self._records[approval_id] = record.model_copy(
                update={"consumed": True}
            )

    @staticmethod
    def _requirement_for(
        *,
        risk: ActionRisk,
        permission_decision: PermissionDecision,
    ) -> tuple[ApprovalRequirement, ApprovalDecision, ApprovalReason, bool]:
        if permission_decision == PermissionDecision.DENY:
            return (
                ApprovalRequirement.BLOCKED,
                ApprovalDecision.BLOCKED,
                ApprovalReason.ACTION_BLOCKED,
                False,
            )

        if risk == ActionRisk.CRITICAL:
            return (
                ApprovalRequirement.ADMIN_APPROVAL,
                ApprovalDecision.PENDING,
                ApprovalReason.ADMIN_APPROVAL_REQUIRED,
                True,
            )

        if risk == ActionRisk.HIGH:
            return (
                ApprovalRequirement.EXPLICIT_APPROVAL,
                ApprovalDecision.PENDING,
                ApprovalReason.EXPLICIT_APPROVAL_REQUIRED,
                True,
            )

        if permission_decision == PermissionDecision.REQUIRE_APPROVAL:
            return (
                ApprovalRequirement.EXPLICIT_APPROVAL,
                ApprovalDecision.PENDING,
                ApprovalReason.EXPLICIT_APPROVAL_REQUIRED,
                True,
            )

        if permission_decision == PermissionDecision.REQUIRE_CONFIRMATION:
            return (
                ApprovalRequirement.SOFT_CONFIRMATION,
                ApprovalDecision.PENDING,
                ApprovalReason.SOFT_CONFIRMATION_REQUIRED,
                True,
            )

        if risk == ActionRisk.MEDIUM:
            return (
                ApprovalRequirement.SOFT_CONFIRMATION,
                ApprovalDecision.PENDING,
                ApprovalReason.SOFT_CONFIRMATION_REQUIRED,
                True,
            )

        return (
            ApprovalRequirement.NONE,
            ApprovalDecision.NOT_REQUIRED,
            ApprovalReason.NO_APPROVAL_REQUIRED,
            False,
        )

    @staticmethod
    def _message_for(
        *,
        requirement: ApprovalRequirement,
        risk: ActionRisk,
        permission_decision: PermissionDecision,
    ) -> str:
        if requirement == ApprovalRequirement.NONE:
            return "approval is not required for this low-risk action"

        if requirement == ApprovalRequirement.SOFT_CONFIRMATION:
            return "soft confirmation is required before this action"

        if requirement == ApprovalRequirement.EXPLICIT_APPROVAL:
            return "explicit human approval is required before this action"

        if requirement == ApprovalRequirement.ADMIN_APPROVAL:
            return "admin-level approval is required before this action"

        return (
            "action is blocked by approval policy "
            f"(risk={risk.value}, permission={permission_decision.value})"
        )

    @staticmethod
    def _record_matches(
        *,
        record: ApprovalRecord,
        action_id: str,
        action_kind: ActionKind | None,
        capability: ToolCapability | None,
        action_scope: ActionScope | None,
        runtime_name: str | None,
    ) -> bool:
        if record.scope == ApprovalScope.ONE_TIME:
            return record.action_id == action_id

        if record.scope == ApprovalScope.SESSION:
            return True

        if record.scope == ApprovalScope.ACTION_KIND:
            return record.action_kind is not None and record.action_kind == action_kind

        if record.scope == ApprovalScope.TOOL_CAPABILITY:
            return record.capability is not None and record.capability == capability

        if record.scope == ApprovalScope.RUNTIME:
            return (
                record.runtime_name is not None
                and record.runtime_name == runtime_name
            )

        return record.action_scope is not None and record.action_scope == action_scope

    def _audit(
        self,
        *,
        action_id: str,
        event_kind: ActionAuditEventKind,
        outcome: ActionAuditOutcome,
        message: str,
        risk: ActionRisk,
        data: dict[str, object],
    ) -> None:
        if self._audit_log is None:
            return

        self._audit_log.record(
            action_id=action_id,
            event_kind=event_kind,
            actor=ActionAuditActor.SYSTEM,
            outcome=outcome,
            message=message,
            risk=risk,
            source_runtime=self.name,
            data=data,
        )