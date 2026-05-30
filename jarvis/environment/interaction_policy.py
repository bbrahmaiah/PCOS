from __future__ import annotations

from enum import StrEnum
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.environment.environment_action_planning import (
    ApprovalRequirement,
    EnvironmentPlanRiskLevel,
    PolicyAwareActionPlan,
)
from jarvis.environment.environment_simulation import (
    EnvironmentSimulationResult,
    SimulatedActionKind,
)
from jarvis.environment.models import EnvironmentSource, TrustCalibration
from jarvis.environment.trust_runtime import TrustPolicyClassification
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class PhysicalInteractionKind(StrEnum):
    MOUSE_CLICK = "mouse_click"
    MOUSE_DOUBLE_CLICK = "mouse_double_click"
    MOUSE_MOVE = "mouse_move"
    KEYBOARD_TYPE = "keyboard_type"
    KEYBOARD_SHORTCUT = "keyboard_shortcut"
    APP_FOCUS = "app_focus"
    APP_OPEN = "app_open"
    APP_CLOSE = "app_close"
    CLIPBOARD_READ = "clipboard_read"
    CLIPBOARD_WRITE = "clipboard_write"
    FILE_MOVE = "file_move"
    FILE_DELETE = "file_delete"
    FORM_SUBMIT = "form_submit"
    SETTING_CHANGE = "setting_change"
    UNKNOWN = "unknown"


class InteractionRisk(StrEnum):
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    BLOCKED = "blocked"


class InteractionPermission(StrEnum):
    ALLOW = "allow"
    ALLOW_WITH_VERIFICATION = "allow_with_verification"
    REQUIRE_APPROVAL = "require_approval"
    REQUIRE_ADMIN_APPROVAL = "require_admin_approval"
    DENY = "deny"


class InteractionValidationStatus(StrEnum):
    VALID = "valid"
    NEEDS_VERIFICATION = "needs_verification"
    NEEDS_APPROVAL = "needs_approval"
    INVALID = "invalid"
    BLOCKED = "blocked"


class InteractionDecision(StrEnum):
    ELIGIBLE_FOR_EXECUTION = "eligible_for_execution"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    REQUIRES_VERIFICATION_FIRST = "requires_verification_first"
    BLOCKED = "blocked"
    INVALID = "invalid"


class InteractionPolicyReason(StrEnum):
    SESSION_CREATED = "session_created"
    REQUEST_ACCEPTED = "request_accepted"
    POLICY_ALLOWED = "policy_allowed"
    POLICY_REQUIRES_VERIFICATION = "policy_requires_verification"
    POLICY_REQUIRES_APPROVAL = "policy_requires_approval"
    POLICY_REQUIRES_ADMIN_APPROVAL = "policy_requires_admin_approval"
    POLICY_DENIED = "policy_denied"
    VALIDATION_FAILED = "validation_failed"
    SIMULATION_REQUIRED = "simulation_required"
    PLAN_NOT_EXECUTION_ELIGIBLE = "plan_not_execution_eligible"
    APPROVAL_RECORDED = "approval_recorded"
    AUDIT_RECORDED = "audit_recorded"
    SESSION_NOT_FOUND = "session_not_found"
    RUNTIME_RESET = "runtime_reset"


class InteractionEventKind(StrEnum):
    SESSION_CREATED = "session_created"
    POLICY_EVALUATED = "policy_evaluated"
    APPROVAL_RECORDED = "approval_recorded"
    AUDIT_RECORDED = "audit_recorded"
    INTERACTION_BLOCKED = "interaction_blocked"
    RUNTIME_RESET = "runtime_reset"


class InteractionApprovalStatus(StrEnum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"


class InteractionUndoRequirement(StrEnum):
    NOT_REQUIRED = "not_required"
    CAPTURE_PRE_STATE = "capture_pre_state"
    REGISTER_UNDO_ACTION = "register_undo_action"
    REQUIRE_BACKUP = "require_backup"
    NOT_UNDOABLE = "not_undoable"


class InteractionVerificationRequirement(StrEnum):
    NONE = "none"
    VERIFY_TARGET_STILL_VALID = "verify_target_still_valid"
    VERIFY_EXPECTED_STATE = "verify_expected_state"
    VERIFY_AND_RECONCILE = "verify_and_reconcile"


class PhysicalActionContract(OrchestrationModel):
    """
    Contract for future physical action runtimes.

    This is the final shape allowed to cross into mouse/keyboard/app runtimes.
    It is not execution.
    """

    contract_id: str = Field(default_factory=lambda: f"physical_contract_{uuid4().hex}")
    kind: PhysicalInteractionKind
    description: str
    target_label: str | None = None
    text_payload_present: bool = False
    source_plan_id: str | None = None
    source_simulation_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("contract_id", "description")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class InteractionRequest(OrchestrationModel):
    """
    Request to evaluate whether a physical interaction may be requested.

    This does not perform the action.
    """

    request_id: str = Field(default_factory=lambda: f"interaction_req_{uuid4().hex}")
    session_id: str
    workspace_id: str
    contract: PhysicalActionContract
    plan: PolicyAwareActionPlan | None = None
    simulation: EnvironmentSimulationResult | None = None
    user_initiated: bool = True
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("request_id", "session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class InteractionRiskAnalysis(OrchestrationModel):
    """
    Risk analysis for physical interaction.
    """

    analysis_id: str = Field(default_factory=lambda: f"interaction_risk_{uuid4().hex}")
    risk: InteractionRisk
    reason: str
    plan_risk: EnvironmentPlanRiskLevel | None = None
    approval_requirement: ApprovalRequirement = ApprovalRequirement.NONE
    confidence: float = Field(default=0.80, ge=0.0, le=1.0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("analysis_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class Phase5PermissionDecision(OrchestrationModel):
    """
    Phase 5 permission-policy adapter result.

    This keeps Phase 8 aligned with the Phase 5 law:
    LLM proposes; policy validates; runtime executes only after permission.
    """

    permission_id: str = Field(default_factory=lambda: f"phase5_perm_{uuid4().hex}")
    permission: InteractionPermission
    reason: str
    policy_classification: TrustPolicyClassification
    requires_user_approval: bool = False
    requires_admin_approval: bool = False
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("permission_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class InteractionValidationResult(OrchestrationModel):
    """
    Validation result before physical interaction eligibility.
    """

    validation_id: str = Field(default_factory=lambda: f"interaction_val_{uuid4().hex}")
    status: InteractionValidationStatus
    reason: str
    valid_contract: bool
    has_simulation: bool
    has_policy_plan: bool
    target_known: bool
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("validation_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class ApprovalGateDecision(OrchestrationModel):
    """
    Approval gate output.

    Approval may be pending even when the plan is otherwise valid.
    """

    approval_id: str = Field(default_factory=lambda: f"approval_gate_{uuid4().hex}")
    status: InteractionApprovalStatus
    requirement: ApprovalRequirement
    prompt: str | None = None
    approved_by: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("approval_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class InteractionAuditRecord(OrchestrationModel):
    """
    Audit record for physical interaction eligibility.

    This is created before execution so future physical runtimes cannot be
    invisible.
    """

    audit_id: str = Field(default_factory=lambda: f"interaction_audit_{uuid4().hex}")
    request_id: str
    contract_id: str
    decision: InteractionDecision
    risk: InteractionRisk
    permission: InteractionPermission
    approval_status: InteractionApprovalStatus
    reason: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("audit_id", "request_id", "contract_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class InteractionPolicyResult(OrchestrationModel):
    """
    Final Step 24 result.

    Later physical runtimes may only proceed if:
    - decision == ELIGIBLE_FOR_EXECUTION
    - execution_eligible == True
    - audit exists
    """

    result_id: str = Field(default_factory=lambda: f"interaction_result_{uuid4().hex}")
    status: InteractionValidationStatus
    decision: InteractionDecision
    reason: InteractionPolicyReason
    request: InteractionRequest
    risk: InteractionRiskAnalysis
    permission: Phase5PermissionDecision
    validation: InteractionValidationResult
    approval: ApprovalGateDecision
    verification_requirement: InteractionVerificationRequirement
    undo_requirement: InteractionUndoRequirement
    audit: InteractionAuditRecord
    trust: TrustCalibration
    execution_eligible: bool = False
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("result_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _eligible_requires_audit_and_allow(self) -> InteractionPolicyResult:
        if self.execution_eligible:
            if self.decision != InteractionDecision.ELIGIBLE_FOR_EXECUTION:
                raise ValueError("execution_eligible requires eligible decision.")

            if self.audit.decision != InteractionDecision.ELIGIBLE_FOR_EXECUTION:
                raise ValueError("execution_eligible requires matching audit.")

        return self


class InteractionPolicySession(OrchestrationModel):
    """
    Interaction policy runtime session.
    """

    session_id: str = Field(
        default_factory=lambda: f"interaction_session_{uuid4().hex}"
    )
    workspace_id: str
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class InteractionPolicyRuntimeEvent(OrchestrationModel):
    """
    Runtime event for Step 24.
    """

    event_id: str = Field(default_factory=lambda: f"interaction_event_{uuid4().hex}")
    kind: InteractionEventKind
    reason: InteractionPolicyReason
    session_id: str | None = None
    request_id: str | None = None
    result_id: str | None = None
    audit_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class InteractionPolicyRuntimeSnapshot(OrchestrationModel):
    """
    Diagnostics for Phase 8 Step 24.
    """

    name: str
    session_count: int = Field(ge=0)
    result_count: int = Field(ge=0)
    eligible_count: int = Field(ge=0)
    approval_pending_count: int = Field(ge=0)
    verification_required_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    audit_count: int = Field(ge=0)
    runtime_event_count: int = Field(ge=0)
    last_reason: InteractionPolicyReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        return _clean_required(value)


class InteractionRiskClassifier:
    """
    Converts policy-aware plan + physical contract into interaction risk.
    """

    def classify(self, request: InteractionRequest) -> InteractionRiskAnalysis:
        if request.plan is None:
            return InteractionRiskAnalysis(
                risk=InteractionRisk.BLOCKED,
                reason="physical interaction requires policy-aware action plan",
                confidence=0.0,
            )

        plan_risk = request.plan.risk.risk_level
        interaction_risk = _interaction_risk_from_plan(plan_risk)

        if request.contract.kind in {
            PhysicalInteractionKind.FILE_DELETE,
            PhysicalInteractionKind.FORM_SUBMIT,
            PhysicalInteractionKind.SETTING_CHANGE,
        }:
            interaction_risk = max_risk(interaction_risk, InteractionRisk.HIGH)

        if request.contract.kind == PhysicalInteractionKind.UNKNOWN:
            interaction_risk = InteractionRisk.BLOCKED

        return InteractionRiskAnalysis(
            risk=interaction_risk,
            reason=request.plan.risk.reason,
            plan_risk=plan_risk,
            approval_requirement=request.plan.risk.approval_requirement,
            confidence=request.plan.risk.confidence,
            metadata={"plan_id": request.plan.plan_id},
        )


class Phase5PermissionPolicyAdapter:
    """
    Adapter representing Phase 5 permission policy.

    This does not call tools. It mirrors the permission decision that future
    execution runtimes must respect.
    """

    def evaluate(
        self,
        request: InteractionRequest,
        risk: InteractionRiskAnalysis,
    ) -> Phase5PermissionDecision:
        if request.plan is None:
            return Phase5PermissionDecision(
                permission=InteractionPermission.DENY,
                reason="missing policy-aware action plan",
                policy_classification=TrustPolicyClassification.BLOCKED,
            )

        if not request.plan.safe_to_request_execution:
            if request.plan.requires_approval:
                return Phase5PermissionDecision(
                    permission=InteractionPermission.REQUIRE_APPROVAL,
                    reason="plan requires user approval before execution request",
                    policy_classification=TrustPolicyClassification.VERIFY_FIRST,
                    requires_user_approval=True,
                )

            return Phase5PermissionDecision(
                permission=InteractionPermission.DENY,
                reason="plan is not safe to request execution",
                policy_classification=TrustPolicyClassification.BLOCKED,
            )

        if risk.risk == InteractionRisk.BLOCKED:
            return Phase5PermissionDecision(
                permission=InteractionPermission.DENY,
                reason="interaction risk is blocked",
                policy_classification=TrustPolicyClassification.BLOCKED,
            )

        if risk.risk == InteractionRisk.CRITICAL:
            return Phase5PermissionDecision(
                permission=InteractionPermission.REQUIRE_ADMIN_APPROVAL,
                reason="critical physical interaction requires admin approval",
                policy_classification=TrustPolicyClassification.VERIFY_FIRST,
                requires_user_approval=True,
                requires_admin_approval=True,
            )

        if risk.risk in {InteractionRisk.HIGH, InteractionRisk.MEDIUM}:
            return Phase5PermissionDecision(
                permission=InteractionPermission.REQUIRE_APPROVAL,
                reason="physical interaction requires user approval",
                policy_classification=TrustPolicyClassification.VERIFY_FIRST,
                requires_user_approval=True,
            )

        if request.plan.requires_verification:
            return Phase5PermissionDecision(
                permission=InteractionPermission.ALLOW_WITH_VERIFICATION,
                reason="interaction allowed only with verification",
                policy_classification=TrustPolicyClassification.REVIEW,
            )

        return Phase5PermissionDecision(
            permission=InteractionPermission.ALLOW,
            reason="interaction allowed by permission policy",
            policy_classification=TrustPolicyClassification.SAFE,
        )


class InteractionValidator:
    """
    Validates interaction request before approval/execution eligibility.
    """

    def validate(
        self,
        request: InteractionRequest,
        permission: Phase5PermissionDecision,
    ) -> InteractionValidationResult:
        has_plan = request.plan is not None
        has_simulation = request.simulation is not None or (
            request.plan is not None and request.plan.simulation is not None
        )
        target_known = (
            request.contract.target_label is not None 
            or request.contract.kind
            in {
                PhysicalInteractionKind.KEYBOARD_SHORTCUT,
                PhysicalInteractionKind.CLIPBOARD_READ,
                PhysicalInteractionKind.APP_OPEN,
            }
        )
        valid_contract = request.contract.kind != PhysicalInteractionKind.UNKNOWN

        if permission.permission == InteractionPermission.DENY:
            return InteractionValidationResult(
                status=InteractionValidationStatus.BLOCKED,
                reason=permission.reason,
                valid_contract=valid_contract,
                has_simulation=has_simulation,
                has_policy_plan=has_plan,
                target_known=target_known,
            )

        if not has_plan:
            return InteractionValidationResult(
                status=InteractionValidationStatus.INVALID,
                reason="interaction request missing policy-aware plan",
                valid_contract=valid_contract,
                has_simulation=has_simulation,
                has_policy_plan=False,
                target_known=target_known,
            )

        if not has_simulation:
            return InteractionValidationResult(
                status=InteractionValidationStatus.INVALID,
                reason="interaction request missing simulation",
                valid_contract=valid_contract,
                has_simulation=False,
                has_policy_plan=has_plan,
                target_known=target_known,
            )

        if not valid_contract:
            return InteractionValidationResult(
                status=InteractionValidationStatus.INVALID,
                reason="physical action contract is invalid",
                valid_contract=False,
                has_simulation=has_simulation,
                has_policy_plan=has_plan,
                target_known=target_known,
            )

        if not target_known:
            return InteractionValidationResult(
                status=InteractionValidationStatus.NEEDS_VERIFICATION,
                reason="target must be verified before physical interaction",
                valid_contract=True,
                has_simulation=has_simulation,
                has_policy_plan=has_plan,
                target_known=False,
            )

        if permission.permission in {
            InteractionPermission.REQUIRE_APPROVAL,
            InteractionPermission.REQUIRE_ADMIN_APPROVAL,
        }:
            return InteractionValidationResult(
                status=InteractionValidationStatus.NEEDS_APPROVAL,
                reason="approval required by permission policy",
                valid_contract=True,
                has_simulation=has_simulation,
                has_policy_plan=has_plan,
                target_known=target_known,
            )

        if permission.permission == InteractionPermission.ALLOW_WITH_VERIFICATION:
            return InteractionValidationResult(
                status=InteractionValidationStatus.NEEDS_VERIFICATION,
                reason="verification required by permission policy",
                valid_contract=True,
                has_simulation=has_simulation,
                has_policy_plan=has_plan,
                target_known=target_known,
            )

        return InteractionValidationResult(
            status=InteractionValidationStatus.VALID,
            reason="interaction request validated",
            valid_contract=True,
            has_simulation=has_simulation,
            has_policy_plan=has_plan,
            target_known=target_known,
        )


class ApprovalGate:
    """
    Approval gate for physical interactions.
    """

    def evaluate(
        self,
        request: InteractionRequest,
        permission: Phase5PermissionDecision,
    ) -> ApprovalGateDecision:
        if permission.permission == InteractionPermission.ALLOW:
            return ApprovalGateDecision(
                status=InteractionApprovalStatus.NOT_REQUIRED,
                requirement=ApprovalRequirement.NONE,
            )

        if permission.permission == InteractionPermission.ALLOW_WITH_VERIFICATION:
            return ApprovalGateDecision(
                status=InteractionApprovalStatus.NOT_REQUIRED,
                requirement=ApprovalRequirement.NONE,
            )

        if permission.permission == InteractionPermission.REQUIRE_ADMIN_APPROVAL:
            return ApprovalGateDecision(
                status=InteractionApprovalStatus.PENDING,
                requirement=ApprovalRequirement.ADMIN_CONFIRMATION,
                prompt=f"Admin approval required: {request.contract.description}",
            )

        if permission.permission == InteractionPermission.REQUIRE_APPROVAL:
            return ApprovalGateDecision(
                status=InteractionApprovalStatus.PENDING,
                requirement=ApprovalRequirement.EXPLICIT_CONFIRMATION,
                prompt=f"Approve physical interaction: {request.contract.description}",
            )

        return ApprovalGateDecision(
            status=InteractionApprovalStatus.DENIED,
            requirement=ApprovalRequirement.BLOCKED,
            prompt="Interaction denied by policy.",
        )


class InteractionAudit:
    """
    Audit builder for interaction policy chain.
    """

    def build(
        self,
        *,
        request: InteractionRequest,
        decision: InteractionDecision,
        risk: InteractionRiskAnalysis,
        permission: Phase5PermissionDecision,
        approval: ApprovalGateDecision,
        reason: str,
    ) -> InteractionAuditRecord:
        return InteractionAuditRecord(
            request_id=request.request_id,
            contract_id=request.contract.contract_id,
            decision=decision,
            risk=risk.risk,
            permission=permission.permission,
            approval_status=approval.status,
            reason=reason,
            metadata={
                "workspace_id": request.workspace_id,
                "contract_kind": request.contract.kind.value,
                "plan_id": request.plan.plan_id if request.plan else None,
            },
        )


class InteractionPolicy:
    """
    Full Step 24 policy chain coordinator.
    """

    def __init__(
        self,
        *,
        risk_classifier: InteractionRiskClassifier | None = None,
        permission_policy: Phase5PermissionPolicyAdapter | None = None,
        validator: InteractionValidator | None = None,
        approval_gate: ApprovalGate | None = None,
        audit: InteractionAudit | None = None,
    ) -> None:
        self._risk_classifier = risk_classifier or InteractionRiskClassifier()
        self._permission_policy = permission_policy or Phase5PermissionPolicyAdapter()
        self._validator = validator or InteractionValidator()
        self._approval_gate = approval_gate or ApprovalGate()
        self._audit = audit or InteractionAudit()

    def evaluate(self, request: InteractionRequest) -> InteractionPolicyResult:
        risk = self._risk_classifier.classify(request)
        permission = self._permission_policy.evaluate(request, risk)
        validation = self._validator.validate(request, permission)
        approval = self._approval_gate.evaluate(request, permission)
        decision, reason = _decision_reason_for(
            validation=validation,
            permission=permission,
            approval=approval,
        )
        verification = _verification_requirement_for(
            request=request,
            validation=validation,
        )
        undo = _undo_requirement_for(request=request, risk=risk)
        audit = self._audit.build(
            request=request,
            decision=decision,
            risk=risk,
            permission=permission,
            approval=approval,
            reason=reason.value,
        )
        execution_eligible = decision == InteractionDecision.ELIGIBLE_FOR_EXECUTION
        trust = TrustCalibration(
            confidence=risk.confidence,
            stability=max(0.0, min(1.0, risk.confidence + 0.05)),
            ambiguity=1.0 - risk.confidence,
            source=EnvironmentSource.OS_OBSERVER,
            reason="interaction policy chain",
        )

        return InteractionPolicyResult(
            status=validation.status,
            decision=decision,
            reason=reason,
            request=request,
            risk=risk,
            permission=permission,
            validation=validation,
            approval=approval,
            verification_requirement=verification,
            undo_requirement=undo,
            audit=audit,
            trust=trust,
            execution_eligible=execution_eligible,
            message=_message_for(decision=decision, reason=reason),
        )


class InteractionPolicyRuntime:
    """
    Phase 8 Step 24 Interaction Contracts & Full Policy Chain.

    Responsibilities:
    - evaluate physical action contract
    - require simulation-backed plan
    - classify interaction risk
    - apply Phase 5 permission policy
    - require approval where needed
    - declare verification/undo/audit requirements
    - decide if future execution may be requested

    Non-responsibilities:
    - no mouse movement
    - no clicking
    - no typing
    - no app control
    - no clipboard mutation
    - no file mutation
    """

    def __init__(
        self,
        *,
        name: str = "interaction_policy_runtime",
        policy: InteractionPolicy | None = None,
    ) -> None:
        cleaned = name.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._policy = policy or InteractionPolicy()
        self._sessions: dict[str, InteractionPolicySession] = {}
        self._results: list[InteractionPolicyResult] = []
        self._audits: list[InteractionAuditRecord] = []
        self._events: list[InteractionPolicyRuntimeEvent] = []
        self._lock = RLock()
        self._last_reason: InteractionPolicyReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def create_session(
        self,
        *,
        workspace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> InteractionPolicySession:
        session = InteractionPolicySession(
            workspace_id=workspace_id,
            metadata=metadata or {},
        )
        event = self._event(
            kind=InteractionEventKind.SESSION_CREATED,
            reason=InteractionPolicyReason.SESSION_CREATED,
            session_id=session.session_id,
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._events.append(event)
            self._last_reason = event.reason

        return session

    def evaluate(self, request: InteractionRequest) -> InteractionPolicyResult:
        if self.session_for(request.session_id) is None:
            result = self._missing_session_result(request)
            self._record_result(result)
            return result

        result = self._policy.evaluate(request)
        self._record_result(result)
        self._touch_session(request.session_id)

        return result

    def session_for(self, session_id: str) -> InteractionPolicySession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def results(self) -> tuple[InteractionPolicyResult, ...]:
        with self._lock:
            return tuple(self._results)

    def audits(self) -> tuple[InteractionAuditRecord, ...]:
        with self._lock:
            return tuple(self._audits)

    def events(self) -> tuple[InteractionPolicyRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def snapshot(self) -> InteractionPolicyRuntimeSnapshot:
        with self._lock:
            return InteractionPolicyRuntimeSnapshot(
                name=self.name,
                session_count=len(self._sessions),
                result_count=len(self._results),
                eligible_count=sum(
                    1 for result in self._results if result.execution_eligible
                ),
                approval_pending_count=sum(
                    1
                    for result in self._results
                    if result.approval.status == InteractionApprovalStatus.PENDING
                ),
                verification_required_count=sum(
                    1
                    for result in self._results
                    if result.verification_requirement
                    != InteractionVerificationRequirement.NONE
                ),
                blocked_count=sum(
                    1
                    for result in self._results
                    if result.decision
                    in {
                        InteractionDecision.BLOCKED,
                        InteractionDecision.INVALID,
                    }
                ),
                audit_count=len(self._audits),
                runtime_event_count=len(self._events),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        event = self._event(
            kind=InteractionEventKind.RUNTIME_RESET,
            reason=InteractionPolicyReason.RUNTIME_RESET,
        )

        with self._lock:
            self._sessions.clear()
            self._results.clear()
            self._audits.clear()
            self._events.clear()
            self._events.append(event)
            self._last_reason = event.reason

    def _missing_session_result(
        self,
        request: InteractionRequest,
    ) -> InteractionPolicyResult:
        risk = InteractionRiskAnalysis(
            risk=InteractionRisk.BLOCKED,
            reason="interaction policy session not found",
            confidence=0.0,
        )
        permission = Phase5PermissionDecision(
            permission=InteractionPermission.DENY,
            reason="interaction policy session not found",
            policy_classification=TrustPolicyClassification.BLOCKED,
        )
        validation = InteractionValidationResult(
            status=InteractionValidationStatus.BLOCKED,
            reason="interaction policy session not found",
            valid_contract=False,
            has_simulation=False,
            has_policy_plan=False,
            target_known=False,
        )
        approval = ApprovalGateDecision(
            status=InteractionApprovalStatus.DENIED,
            requirement=ApprovalRequirement.BLOCKED,
            prompt="Interaction blocked because session was not found.",
        )
        audit = InteractionAuditRecord(
            request_id=request.request_id,
            contract_id=request.contract.contract_id,
            decision=InteractionDecision.BLOCKED,
            risk=InteractionRisk.BLOCKED,
            permission=InteractionPermission.DENY,
            approval_status=InteractionApprovalStatus.DENIED,
            reason="interaction policy session not found",
        )

        return InteractionPolicyResult(
            status=InteractionValidationStatus.BLOCKED,
            decision=InteractionDecision.BLOCKED,
            reason=InteractionPolicyReason.SESSION_NOT_FOUND,
            request=request,
            risk=risk,
            permission=permission,
            validation=validation,
            approval=approval,
            verification_requirement=InteractionVerificationRequirement.VERIFY_AND_RECONCILE,
            undo_requirement=InteractionUndoRequirement.NOT_UNDOABLE,
            audit=audit,
            trust=TrustCalibration(
                confidence=0.0,
                stability=0.0,
                ambiguity=1.0,
                source=EnvironmentSource.OS_OBSERVER,
                reason="interaction policy session missing",
            ),
            execution_eligible=False,
            message="interaction policy session not found",
        )

    def _record_result(self, result: InteractionPolicyResult) -> None:
        event = self._event(
            kind=(
                InteractionEventKind.POLICY_EVALUATED
                if result.execution_eligible
                else InteractionEventKind.INTERACTION_BLOCKED
            ),
            reason=result.reason,
            session_id=result.request.session_id,
            request_id=result.request.request_id,
            result_id=result.result_id,
            audit_id=result.audit.audit_id,
            metadata={
                "decision": result.decision.value,
                "eligible": result.execution_eligible,
            },
        )

        with self._lock:
            self._results.append(result)
            self._audits.append(result.audit)
            self._events.append(event)
            self._last_reason = result.reason

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
        kind: InteractionEventKind,
        reason: InteractionPolicyReason,
        session_id: str | None = None,
        request_id: str | None = None,
        result_id: str | None = None,
        audit_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> InteractionPolicyRuntimeEvent:
        return InteractionPolicyRuntimeEvent(
            kind=kind,
            reason=reason,
            session_id=session_id,
            request_id=request_id,
            result_id=result_id,
            audit_id=audit_id,
            metadata=metadata or {},
        )


def contract_from_plan(plan: PolicyAwareActionPlan) -> PhysicalActionContract:
    """
    Convert policy-aware plan into physical action contract.

    Later physical runtimes should receive only this contract after Step 24
    approves eligibility.
    """

    action = plan.candidate.action

    return PhysicalActionContract(
        kind=_physical_kind_from(action.kind),
        description=action.description,
        target_label=action.target_label,
        text_payload_present=action.text_payload is not None,
        source_plan_id=plan.plan_id,
        source_simulation_id=(
            plan.simulation.result_id if plan.simulation is not None else None
        ),
        metadata={"action_id": action.action_id},
    )


def _interaction_risk_from_plan(
    risk: EnvironmentPlanRiskLevel,
) -> InteractionRisk:
    if risk == EnvironmentPlanRiskLevel.BLOCKED:
        return InteractionRisk.BLOCKED

    if risk == EnvironmentPlanRiskLevel.CRITICAL:
        return InteractionRisk.CRITICAL

    if risk == EnvironmentPlanRiskLevel.HIGH:
        return InteractionRisk.HIGH

    if risk == EnvironmentPlanRiskLevel.MEDIUM:
        return InteractionRisk.MEDIUM

    if risk == EnvironmentPlanRiskLevel.LOW:
        return InteractionRisk.LOW

    return InteractionRisk.SAFE


def max_risk(first: InteractionRisk, second: InteractionRisk) -> InteractionRisk:
    order = {
        InteractionRisk.SAFE: 0,
        InteractionRisk.LOW: 1,
        InteractionRisk.MEDIUM: 2,
        InteractionRisk.HIGH: 3,
        InteractionRisk.CRITICAL: 4,
        InteractionRisk.BLOCKED: 5,
    }

    return first if order[first] >= order[second] else second


def _physical_kind_from(action: SimulatedActionKind) -> PhysicalInteractionKind:
    if action == SimulatedActionKind.CLICK:
        return PhysicalInteractionKind.MOUSE_CLICK

    if action == SimulatedActionKind.TYPE_TEXT:
        return PhysicalInteractionKind.KEYBOARD_TYPE

    if action == SimulatedActionKind.DELETE:
        return PhysicalInteractionKind.FILE_DELETE

    if action == SimulatedActionKind.SUBMIT:
        return PhysicalInteractionKind.FORM_SUBMIT

    if action == SimulatedActionKind.CLOSE:
        return PhysicalInteractionKind.APP_CLOSE

    if action == SimulatedActionKind.MOVE_FILE:
        return PhysicalInteractionKind.FILE_MOVE

    if action == SimulatedActionKind.CHANGE_SETTING:
        return PhysicalInteractionKind.SETTING_CHANGE

    if action == SimulatedActionKind.OPEN:
        return PhysicalInteractionKind.APP_OPEN

    if action == SimulatedActionKind.FOCUS:
        return PhysicalInteractionKind.APP_FOCUS

    if action == SimulatedActionKind.COPY:
        return PhysicalInteractionKind.CLIPBOARD_WRITE

    return PhysicalInteractionKind.UNKNOWN


def _decision_reason_for(
    *,
    validation: InteractionValidationResult,
    permission: Phase5PermissionDecision,
    approval: ApprovalGateDecision,
) -> tuple[InteractionDecision, InteractionPolicyReason]:
    if validation.status in {
        InteractionValidationStatus.BLOCKED,
        InteractionValidationStatus.INVALID,
    }:
        return InteractionDecision.BLOCKED, InteractionPolicyReason.VALIDATION_FAILED

    if permission.permission == InteractionPermission.DENY:
        return InteractionDecision.BLOCKED, InteractionPolicyReason.POLICY_DENIED

    if permission.permission == InteractionPermission.REQUIRE_ADMIN_APPROVAL:
        return (
            InteractionDecision.WAITING_FOR_APPROVAL,
            InteractionPolicyReason.POLICY_REQUIRES_ADMIN_APPROVAL,
        )

    if permission.permission == InteractionPermission.REQUIRE_APPROVAL:
        if approval.status == InteractionApprovalStatus.APPROVED:
            return (
                InteractionDecision.ELIGIBLE_FOR_EXECUTION,
                InteractionPolicyReason.POLICY_ALLOWED,
            )

        return (
            InteractionDecision.WAITING_FOR_APPROVAL,
            InteractionPolicyReason.POLICY_REQUIRES_APPROVAL,
        )

    if permission.permission == InteractionPermission.ALLOW_WITH_VERIFICATION:
        return (
            InteractionDecision.REQUIRES_VERIFICATION_FIRST,
            InteractionPolicyReason.POLICY_REQUIRES_VERIFICATION,
        )

    return (
        InteractionDecision.ELIGIBLE_FOR_EXECUTION,
        InteractionPolicyReason.POLICY_ALLOWED,
    )


def _verification_requirement_for(
    *,
    request: InteractionRequest,
    validation: InteractionValidationResult,
) -> InteractionVerificationRequirement:
    if validation.status == InteractionValidationStatus.NEEDS_VERIFICATION:
        return InteractionVerificationRequirement.VERIFY_TARGET_STILL_VALID

    if request.plan is not None and request.plan.requires_verification:
        return InteractionVerificationRequirement.VERIFY_EXPECTED_STATE

    if request.plan is not None and request.plan.expected_state_plan is not None:
        return InteractionVerificationRequirement.VERIFY_EXPECTED_STATE

    return InteractionVerificationRequirement.NONE


def _undo_requirement_for(
    *,
    request: InteractionRequest,
    risk: InteractionRiskAnalysis,
) -> InteractionUndoRequirement:
    if risk.risk == InteractionRisk.CRITICAL:
        return InteractionUndoRequirement.REQUIRE_BACKUP

    if risk.risk == InteractionRisk.HIGH:
        return InteractionUndoRequirement.REGISTER_UNDO_ACTION

    if request.contract.kind in {
        PhysicalInteractionKind.KEYBOARD_TYPE,
        PhysicalInteractionKind.CLIPBOARD_WRITE,
        PhysicalInteractionKind.SETTING_CHANGE,
    }:
        return InteractionUndoRequirement.CAPTURE_PRE_STATE

    if risk.risk == InteractionRisk.BLOCKED:
        return InteractionUndoRequirement.NOT_UNDOABLE

    return InteractionUndoRequirement.NOT_REQUIRED


def _message_for(
    *,
    decision: InteractionDecision,
    reason: InteractionPolicyReason,
) -> str:
    return f"interaction decision={decision.value}; reason={reason.value}"


def _clean_required(value: str) -> str:
    cleaned = value.strip()

    if not cleaned:
        raise ValueError("field cannot be empty.")

    return cleaned