from __future__ import annotations

from enum import StrEnum
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.environment.models import EnvironmentSource, TrustCalibration
from jarvis.environment.recovery_runtime import (
    RecoveryDecision,
    RecoveryResult,
)
from jarvis.environment.trust_runtime import TrustPolicyClassification
from jarvis.environment.verification_runtime import (
    VerificationContract,
    VerificationDecision,
    VerificationResult,
    VerificationStatus,
)
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class UndoActionKind(StrEnum):
    FILE_RESTORE = "file_restore"
    TEXT_REVERT = "text_revert"
    APP_STATE_RESTORE = "app_state_restore"
    CLIPBOARD_RESTORE = "clipboard_restore"
    WORKSPACE_RESTORE = "workspace_restore"
    COMMAND_COMPENSATION = "command_compensation"
    MANUAL_RECOVERY = "manual_recovery"


class MutatingActionKind(StrEnum):
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    TEXT_EDIT = "text_edit"
    CLIPBOARD_PASTE = "clipboard_paste"
    APP_CLOSE = "app_close"
    SETTINGS_CHANGE = "settings_change"
    COMMAND_EXECUTION = "command_execution"
    WORKSPACE_CHANGE = "workspace_change"


class ReversibilityLevel(StrEnum):
    REVERSIBLE = "reversible"
    COMPENSATABLE = "compensatable"
    PARTIAL = "partial"
    IRREVERSIBLE = "irreversible"
    UNKNOWN = "unknown"


class UndoRollbackStatus(StrEnum):
    DECLARED = "declared"
    STACKED = "stacked"
    ROLLBACK_READY = "rollback_ready"
    VERIFICATION_REQUIRED = "verification_required"
    APPROVAL_REQUIRED = "approval_required"
    BLOCKED = "blocked"
    FAILED = "failed"


class UndoRollbackDecision(StrEnum):
    ALLOW_MUTATION = "allow_mutation"
    REQUIRE_APPROVAL = "require_approval"
    REQUIRE_UNDO_DECLARATION = "require_undo_declaration"
    PREPARE_ROLLBACK = "prepare_rollback"
    REQUIRE_VERIFICATION = "require_verification"
    BLOCK = "block"


class UndoRollbackReason(StrEnum):
    SESSION_CREATED = "session_created"
    UNDO_DECLARED = "undo_declared"
    MUTATION_ALLOWED = "mutation_allowed"
    UNDO_DECLARATION_MISSING = "undo_declaration_missing"
    IRREVERSIBLE_REQUIRES_APPROVAL = "irreversible_requires_approval"
    UNKNOWN_REVERSIBILITY_BLOCKED = "unknown_reversibility_blocked"
    UNDO_STACK_PUSHED = "undo_stack_pushed"
    ROLLBACK_PLAN_CREATED = "rollback_plan_created"
    ROLLBACK_VERIFICATION_REQUIRED = "rollback_verification_required"
    ROLLBACK_VERIFIED = "rollback_verified"
    ROLLBACK_VERIFICATION_FAILED = "rollback_verification_failed"
    RECOVERY_NOT_ROLLBACK = "recovery_not_rollback"
    SESSION_NOT_FOUND = "session_not_found"
    RUNTIME_RESET = "runtime_reset"


class UndoRollbackEventKind(StrEnum):
    SESSION_CREATED = "session_created"
    UNDO_DECLARED = "undo_declared"
    STACK_UPDATED = "stack_updated"
    ROLLBACK_PLANNED = "rollback_planned"
    ROLLBACK_VERIFIED = "rollback_verified"
    OPERATION_BLOCKED = "operation_blocked"
    RUNTIME_RESET = "runtime_reset"


class UndoApprovalKind(StrEnum):
    NONE = "none"
    SOFT_CONFIRMATION = "soft_confirmation"
    EXPLICIT_CONFIRMATION = "explicit_confirmation"
    MANUAL_APPROVAL = "manual_approval"


class UndoVerificationRequirement(StrEnum):
    REQUIRED = "required"
    REQUIRED_WITH_RECONCILIATION = "required_with_reconciliation"
    MANUAL = "manual"


class ReversibilityContract(OrchestrationModel):
    """
    Required before every mutating action.

    It declares how the action can be undone, compensated, or why it cannot be
    safely reversed.
    """

    contract_id: str = Field(
        default_factory=lambda: f"reversibility_contract_{uuid4().hex}"
    )
    action_id: str
    workspace_id: str
    mutation_kind: MutatingActionKind
    reversibility: ReversibilityLevel
    undo_kind: UndoActionKind | None = None
    undo_description: str | None = None
    backup_required: bool = False
    backup_reference: str | None = None
    irreversible_reason: str | None = None
    requires_approval: bool = False
    verification_required: bool = True
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("contract_id", "action_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _validate_reversibility(self) -> ReversibilityContract:
        if self.reversibility in {
            ReversibilityLevel.REVERSIBLE,
            ReversibilityLevel.COMPENSATABLE,
            ReversibilityLevel.PARTIAL,
        }:
            if self.undo_kind is None:
                raise ValueError("reversible mutation requires undo_kind.")

            if self.undo_description is None:
                raise ValueError("reversible mutation requires undo_description.")

        if self.reversibility == ReversibilityLevel.IRREVERSIBLE:
            if self.irreversible_reason is None:
                raise ValueError("irreversible mutation requires reason.")

            if not self.requires_approval:
                raise ValueError("irreversible mutation requires approval.")

        if self.reversibility == ReversibilityLevel.UNKNOWN:
            if not self.requires_approval:
                raise ValueError("unknown reversibility requires approval.")

        if self.backup_required and self.backup_reference is None:
            raise ValueError("backup_required requires backup_reference.")

        return self


class UndoPolicy(OrchestrationModel):
    """
    Policy for allowing state-changing actions.
    """

    require_undo_for_mutation: bool = True
    require_approval_for_irreversible: bool = True
    block_unknown_reversibility: bool = True
    require_verification_after_undo: bool = True
    max_undo_stack_depth: int = Field(default=100, ge=1, le=1000)


class UndoDeclarationResult(OrchestrationModel):
    result_id: str = Field(default_factory=lambda: f"undo_decl_{uuid4().hex}")
    status: UndoRollbackStatus
    decision: UndoRollbackDecision
    reason: UndoRollbackReason
    contract: ReversibilityContract | None = None
    approval: UndoApprovalKind
    mutation_allowed: bool
    trust: TrustCalibration
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("result_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _allowed_requires_contract(self) -> UndoDeclarationResult:
        if self.mutation_allowed and self.contract is None:
            raise ValueError("mutation_allowed requires reversibility contract.")

        return self


class UndoStackEntry(OrchestrationModel):
    entry_id: str = Field(default_factory=lambda: f"undo_stack_{uuid4().hex}")
    action_id: str
    workspace_id: str
    contract: ReversibilityContract
    pushed_at: object = Field(default_factory=utc_now)
    consumed: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("entry_id", "action_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class UndoStackSnapshot(OrchestrationModel):
    stack_id: str
    depth: int = Field(ge=0)
    active_count: int = Field(ge=0)
    consumed_count: int = Field(ge=0)
    top_action_id: str | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("stack_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class UndoStack(OrchestrationModel):
    stack_id: str = Field(default_factory=lambda: f"undo_stack_{uuid4().hex}")
    workspace_id: str
    entries: tuple[UndoStackEntry, ...] = ()
    max_depth: int = Field(default=100, ge=1, le=1000)
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)

    @field_validator("stack_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    def push(self, entry: UndoStackEntry) -> UndoStack:
        next_entries = (entry,) + self.entries
        if len(next_entries) > self.max_depth:
            next_entries = next_entries[: self.max_depth]

        return self.model_copy(
            update={
                "entries": next_entries,
                "updated_at": utc_now(),
            }
        )

    def top(self) -> UndoStackEntry | None:
        for entry in self.entries:
            if not entry.consumed:
                return entry

        return None

    def snapshot(self) -> UndoStackSnapshot:
        top = self.top()
        consumed = sum(1 for entry in self.entries if entry.consumed)

        return UndoStackSnapshot(
            stack_id=self.stack_id,
            depth=len(self.entries),
            active_count=len(self.entries) - consumed,
            consumed_count=consumed,
            top_action_id=top.action_id if top is not None else None,
        )


class RollbackPlanStep(OrchestrationModel):
    step_id: str = Field(default_factory=lambda: f"rollback_step_{uuid4().hex}")
    order: int = Field(ge=0)
    undo_kind: UndoActionKind
    description: str
    verification_required: bool = True
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("step_id", "description")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class RollbackPlan(OrchestrationModel):
    rollback_id: str = Field(default_factory=lambda: f"rollback_plan_{uuid4().hex}")
    action_id: str
    workspace_id: str
    contract: ReversibilityContract
    steps: tuple[RollbackPlanStep, ...]
    verification_contract: VerificationContract | None = None
    requires_user_approval: bool = False
    verification_requirement: UndoVerificationRequirement = (
        UndoVerificationRequirement.REQUIRED
    )
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("rollback_id", "action_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _requires_steps_and_verification(self) -> RollbackPlan:
        if not self.steps:
            raise ValueError("rollback plan requires steps.")

        if self.verification_contract is None:
            raise ValueError("rollback plan requires verification contract.")

        return self


class RollbackAudit(OrchestrationModel):
    audit_id: str = Field(default_factory=lambda: f"rollback_audit_{uuid4().hex}")
    action_id: str
    rollback_id: str | None = None
    status: UndoRollbackStatus
    decision: UndoRollbackDecision
    reason: UndoRollbackReason
    requires_approval: bool = False
    verified: bool = False
    audit_preserved: bool = True
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("audit_id", "action_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _audit_must_be_preserved(self) -> RollbackAudit:
        if not self.audit_preserved:
            raise ValueError("rollback must preserve audit trail.")

        return self


class RollbackResult(OrchestrationModel):
    result_id: str = Field(default_factory=lambda: f"rollback_result_{uuid4().hex}")
    status: UndoRollbackStatus
    decision: UndoRollbackDecision
    reason: UndoRollbackReason
    recovery: RecoveryResult | None = None
    stack_entry: UndoStackEntry | None = None
    rollback_plan: RollbackPlan | None = None
    verification_result: VerificationResult | None = None
    audit: RollbackAudit
    trust: TrustCalibration
    rollback_ready: bool = False
    rollback_verified: bool = False
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("result_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _ready_requires_plan(self) -> RollbackResult:
        if self.rollback_ready and self.rollback_plan is None:
            raise ValueError("rollback_ready requires rollback_plan.")

        if self.rollback_verified and self.verification_result is None:
            raise ValueError("rollback_verified requires verification_result.")

        return self


class UndoRollbackRuntimeSession(OrchestrationModel):
    session_id: str = Field(default_factory=lambda: f"undo_session_{uuid4().hex}")
    workspace_id: str
    declaration_count: int = Field(default=0, ge=0)
    rollback_plan_count: int = Field(default=0, ge=0)
    verified_rollback_count: int = Field(default=0, ge=0)
    blocked_count: int = Field(default=0, ge=0)
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class UndoRollbackRuntimeEvent(OrchestrationModel):
    event_id: str = Field(default_factory=lambda: f"undo_event_{uuid4().hex}")
    kind: UndoRollbackEventKind
    reason: UndoRollbackReason
    session_id: str | None = None
    result_id: str | None = None
    audit_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class UndoRollbackRuntimeSnapshot(OrchestrationModel):
    name: str
    session_count: int = Field(ge=0)
    stack_count: int = Field(ge=0)
    declaration_count: int = Field(ge=0)
    rollback_result_count: int = Field(ge=0)
    rollback_ready_count: int = Field(ge=0)
    verified_rollback_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    audit_count: int = Field(ge=0)
    runtime_event_count: int = Field(ge=0)
    last_reason: UndoRollbackReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class UndoRollbackRuntime:
    """
    Phase 8 Step 31 Undo & Rollback System.

    Responsibilities:
    - require undo declaration before mutating action
    - block unknown reversibility
    - approval-gate irreversible actions
    - push reversible actions into undo stack
    - create rollback plans from undo stack entries
    - require rollback verification
    - preserve audit trail

    Non-responsibilities:
    - does not execute OS rollback
    - does not modify files directly
    - does not silently mark rollback complete
    """

    def __init__(
        self,
        *,
        name: str = "undo_rollback_runtime",
        policy: UndoPolicy | None = None,
    ) -> None:
        cleaned = name.strip()
        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._policy = policy or UndoPolicy()
        self._sessions: dict[str, UndoRollbackRuntimeSession] = {}
        self._stacks: dict[str, UndoStack] = {}
        self._declarations: list[UndoDeclarationResult] = []
        self._rollback_results: list[RollbackResult] = []
        self._audits: list[RollbackAudit] = []
        self._events: list[UndoRollbackRuntimeEvent] = []
        self._lock = RLock()
        self._last_reason: UndoRollbackReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def create_session(
        self,
        *,
        workspace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> UndoRollbackRuntimeSession:
        session = UndoRollbackRuntimeSession(
            workspace_id=workspace_id,
            metadata=metadata or {},
        )
        stack = UndoStack(
            workspace_id=workspace_id,
            max_depth=self._policy.max_undo_stack_depth,
        )
        event = self._event(
            kind=UndoRollbackEventKind.SESSION_CREATED,
            reason=UndoRollbackReason.SESSION_CREATED,
            session_id=session.session_id,
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._stacks[session.session_id] = stack
            self._events.append(event)
            self._last_reason = event.reason

        return session

    def declare_undo(
        self,
        *,
        session_id: str,
        contract: ReversibilityContract | None,
    ) -> UndoDeclarationResult:
        if self.session_for(session_id) is None:
            result = _declaration_result(
                contract=contract,
                status=UndoRollbackStatus.FAILED,
                decision=UndoRollbackDecision.BLOCK,
                reason=UndoRollbackReason.SESSION_NOT_FOUND,
                approval=UndoApprovalKind.NONE,
                mutation_allowed=False,
                message="undo rollback session not found",
            )
            self._record_declaration(result, session_id)
            return result

        result = self._evaluate_contract(contract)
        self._record_declaration(result, session_id)

        if result.mutation_allowed and result.contract is not None:
            self._push_undo_entry(session_id=session_id, contract=result.contract)

        return result

    def plan_rollback(
        self,
        *,
        session_id: str,
        recovery: RecoveryResult,
        verification_contract: VerificationContract,
    ) -> RollbackResult:
        if self.session_for(session_id) is None:
            result = _rollback_result(
                status=UndoRollbackStatus.FAILED,
                decision=UndoRollbackDecision.BLOCK,
                reason=UndoRollbackReason.SESSION_NOT_FOUND,
                recovery=recovery,
                stack_entry=None,
                plan=None,
                verification_result=None,
                message="undo rollback session not found",
            )
            self._record_rollback(result, session_id)
            return result

        if recovery.decision != RecoveryDecision.ROLLBACK:
            result = _rollback_result(
                status=UndoRollbackStatus.BLOCKED,
                decision=UndoRollbackDecision.BLOCK,
                reason=UndoRollbackReason.RECOVERY_NOT_ROLLBACK,
                recovery=recovery,
                stack_entry=None,
                plan=None,
                verification_result=None,
                message="recovery result did not request rollback",
            )
            self._record_rollback(result, session_id)
            return result

        stack = self._stacks[session_id]
        entry = stack.top()

        if entry is None:
            result = _rollback_result(
                status=UndoRollbackStatus.BLOCKED,
                decision=UndoRollbackDecision.BLOCK,
                reason=UndoRollbackReason.UNDO_DECLARATION_MISSING,
                recovery=recovery,
                stack_entry=None,
                plan=None,
                verification_result=None,
                message="rollback requires undo stack entry",
            )
            self._record_rollback(result, session_id)
            return result

        plan = _build_rollback_plan(
            entry=entry,
            verification_contract=verification_contract,
        )
        result = _rollback_result(
            status=UndoRollbackStatus.ROLLBACK_READY,
            decision=UndoRollbackDecision.PREPARE_ROLLBACK,
            reason=UndoRollbackReason.ROLLBACK_PLAN_CREATED,
            recovery=recovery,
            stack_entry=entry,
            plan=plan,
            verification_result=None,
            rollback_ready=True,
            message="rollback plan created and requires verification",
        )
        self._record_rollback(result, session_id)

        return result

    def verify_rollback(
        self,
        *,
        session_id: str,
        rollback_result: RollbackResult,
        verification_result: VerificationResult,
    ) -> RollbackResult:
        if self.session_for(session_id) is None:
            result = _rollback_result(
                status=UndoRollbackStatus.FAILED,
                decision=UndoRollbackDecision.BLOCK,
                reason=UndoRollbackReason.SESSION_NOT_FOUND,
                recovery=rollback_result.recovery,
                stack_entry=rollback_result.stack_entry,
                plan=rollback_result.rollback_plan,
                verification_result=verification_result,
                message="undo rollback session not found",
            )
            self._record_rollback(result, session_id)
            return result

        if rollback_result.rollback_plan is None:
            result = _rollback_result(
                status=UndoRollbackStatus.BLOCKED,
                decision=UndoRollbackDecision.BLOCK,
                reason=UndoRollbackReason.UNDO_DECLARATION_MISSING,
                recovery=rollback_result.recovery,
                stack_entry=rollback_result.stack_entry,
                plan=None,
                verification_result=verification_result,
                message="rollback verification requires rollback plan",
            )
            self._record_rollback(result, session_id)
            return result

        if (
            verification_result.status == VerificationStatus.PASSED
            and verification_result.decision == VerificationDecision.COMPLETE
        ):
            result = _rollback_result(
                status=UndoRollbackStatus.DECLARED,
                decision=UndoRollbackDecision.ALLOW_MUTATION,
                reason=UndoRollbackReason.ROLLBACK_VERIFIED,
                recovery=rollback_result.recovery,
                stack_entry=rollback_result.stack_entry,
                plan=rollback_result.rollback_plan,
                verification_result=verification_result,
                rollback_ready=False,
                rollback_verified=True,
                message="rollback verified successfully",
            )
            self._record_rollback(result, session_id)
            return result

        result = _rollback_result(
            status=UndoRollbackStatus.VERIFICATION_REQUIRED,
            decision=UndoRollbackDecision.REQUIRE_VERIFICATION,
            reason=UndoRollbackReason.ROLLBACK_VERIFICATION_FAILED,
            recovery=rollback_result.recovery,
            stack_entry=rollback_result.stack_entry,
            plan=rollback_result.rollback_plan,
            verification_result=verification_result,
            rollback_ready=False,
            rollback_verified=False,
            message="rollback verification failed or needs review",
        )
        self._record_rollback(result, session_id)

        return result

    def session_for(
        self,
        session_id: str,
    ) -> UndoRollbackRuntimeSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def stack_for(self, session_id: str) -> UndoStack | None:
        with self._lock:
            return self._stacks.get(session_id)

    def declarations(self) -> tuple[UndoDeclarationResult, ...]:
        with self._lock:
            return tuple(self._declarations)

    def rollback_results(self) -> tuple[RollbackResult, ...]:
        with self._lock:
            return tuple(self._rollback_results)

    def audits(self) -> tuple[RollbackAudit, ...]:
        with self._lock:
            return tuple(self._audits)

    def events(self) -> tuple[UndoRollbackRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def snapshot(self) -> UndoRollbackRuntimeSnapshot:
        with self._lock:
            return UndoRollbackRuntimeSnapshot(
                name=self.name,
                session_count=len(self._sessions),
                stack_count=len(self._stacks),
                declaration_count=len(self._declarations),
                rollback_result_count=len(self._rollback_results),
                rollback_ready_count=sum(
                    1
                    for result in self._rollback_results
                    if result.rollback_ready
                ),
                verified_rollback_count=sum(
                    1
                    for result in self._rollback_results
                    if result.rollback_verified
                ),
                blocked_count=sum(
                    1
                    for result in self._rollback_results
                    if result.status
                    in {
                        UndoRollbackStatus.BLOCKED,
                        UndoRollbackStatus.FAILED,
                    }
                ),
                audit_count=len(self._audits),
                runtime_event_count=len(self._events),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        event = self._event(
            kind=UndoRollbackEventKind.RUNTIME_RESET,
            reason=UndoRollbackReason.RUNTIME_RESET,
        )

        with self._lock:
            self._sessions.clear()
            self._stacks.clear()
            self._declarations.clear()
            self._rollback_results.clear()
            self._audits.clear()
            self._events.clear()
            self._events.append(event)
            self._last_reason = event.reason

    def _evaluate_contract(
        self,
        contract: ReversibilityContract | None,
    ) -> UndoDeclarationResult:
        if contract is None:
            return _declaration_result(
                contract=None,
                status=UndoRollbackStatus.BLOCKED,
                decision=UndoRollbackDecision.REQUIRE_UNDO_DECLARATION,
                reason=UndoRollbackReason.UNDO_DECLARATION_MISSING,
                approval=UndoApprovalKind.NONE,
                mutation_allowed=False,
                message="mutating action requires undo declaration",
            )

        if contract.reversibility == ReversibilityLevel.UNKNOWN:
            return _declaration_result(
                contract=contract,
                status=UndoRollbackStatus.BLOCKED,
                decision=UndoRollbackDecision.BLOCK,
                reason=UndoRollbackReason.UNKNOWN_REVERSIBILITY_BLOCKED,
                approval=UndoApprovalKind.EXPLICIT_CONFIRMATION,
                mutation_allowed=False,
                message="unknown reversibility is blocked",
            )

        if contract.reversibility == ReversibilityLevel.IRREVERSIBLE:
            return _declaration_result(
                contract=contract,
                status=UndoRollbackStatus.APPROVAL_REQUIRED,
                decision=UndoRollbackDecision.REQUIRE_APPROVAL,
                reason=UndoRollbackReason.IRREVERSIBLE_REQUIRES_APPROVAL,
                approval=UndoApprovalKind.EXPLICIT_CONFIRMATION,
                mutation_allowed=False,
                message="irreversible action requires explicit approval",
            )

        return _declaration_result(
            contract=contract,
            status=UndoRollbackStatus.DECLARED,
            decision=UndoRollbackDecision.ALLOW_MUTATION,
            reason=UndoRollbackReason.MUTATION_ALLOWED,
            approval=UndoApprovalKind.NONE,
            mutation_allowed=True,
            message="undo declared; mutation may proceed",
        )

    def _push_undo_entry(
        self,
        *,
        session_id: str,
        contract: ReversibilityContract,
    ) -> None:
        stack = self._stacks[session_id]
        entry = UndoStackEntry(
            action_id=contract.action_id,
            workspace_id=contract.workspace_id,
            contract=contract,
        )
        self._stacks[session_id] = stack.push(entry)

    def _record_declaration(
        self,
        result: UndoDeclarationResult,
        session_id: str,
    ) -> None:
        event = self._event(
            kind=(
                UndoRollbackEventKind.UNDO_DECLARED
                if result.mutation_allowed
                else UndoRollbackEventKind.OPERATION_BLOCKED
            ),
            reason=result.reason,
            session_id=session_id,
            result_id=result.result_id,
        )

        with self._lock:
            self._declarations.append(result)
            self._events.append(event)
            self._last_reason = result.reason
            session = self._sessions.get(session_id)
            if session is not None:
                self._sessions[session_id] = session.model_copy(
                    update={
                        "updated_at": utc_now(),
                        "declaration_count": session.declaration_count + 1,
                        "blocked_count": session.blocked_count
                        + (0 if result.mutation_allowed else 1),
                    }
                )

    def _record_rollback(
        self,
        result: RollbackResult,
        session_id: str,
    ) -> None:
        event = self._event(
            kind=(
                UndoRollbackEventKind.ROLLBACK_VERIFIED
                if result.rollback_verified
                else UndoRollbackEventKind.ROLLBACK_PLANNED
                if result.rollback_ready
                else UndoRollbackEventKind.OPERATION_BLOCKED
            ),
            reason=result.reason,
            session_id=session_id,
            result_id=result.result_id,
            audit_id=result.audit.audit_id,
        )

        with self._lock:
            self._rollback_results.append(result)
            self._audits.append(result.audit)
            self._events.append(event)
            self._last_reason = result.reason
            session = self._sessions.get(session_id)
            if session is not None:
                self._sessions[session_id] = session.model_copy(
                    update={
                        "updated_at": utc_now(),
                        "rollback_plan_count": session.rollback_plan_count
                        + (1 if result.rollback_ready else 0),
                        "verified_rollback_count": session.verified_rollback_count
                        + (1 if result.rollback_verified else 0),
                        "blocked_count": session.blocked_count
                        + (
                            1
                            if result.status
                            in {
                                UndoRollbackStatus.BLOCKED,
                                UndoRollbackStatus.FAILED,
                            }
                            else 0
                        ),
                    }
                )

    @staticmethod
    def _event(
        *,
        kind: UndoRollbackEventKind,
        reason: UndoRollbackReason,
        session_id: str | None = None,
        result_id: str | None = None,
        audit_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UndoRollbackRuntimeEvent:
        return UndoRollbackRuntimeEvent(
            kind=kind,
            reason=reason,
            session_id=session_id,
            result_id=result_id,
            audit_id=audit_id,
            metadata=metadata or {},
        )


def _declaration_result(
    *,
    contract: ReversibilityContract | None,
    status: UndoRollbackStatus,
    decision: UndoRollbackDecision,
    reason: UndoRollbackReason,
    approval: UndoApprovalKind,
    mutation_allowed: bool,
    message: str,
) -> UndoDeclarationResult:
    confidence = 0.86 if mutation_allowed else 0.40

    return UndoDeclarationResult(
        status=status,
        decision=decision,
        reason=reason,
        contract=contract,
        approval=approval,
        mutation_allowed=mutation_allowed,
        trust=_trust(confidence=confidence, reason=message),
        message=message,
    )


def _build_rollback_plan(
    *,
    entry: UndoStackEntry,
    verification_contract: VerificationContract,
) -> RollbackPlan:
    contract = entry.contract
    undo_kind = contract.undo_kind or UndoActionKind.MANUAL_RECOVERY
    description = contract.undo_description or "manual rollback required"

    return RollbackPlan(
        action_id=entry.action_id,
        workspace_id=entry.workspace_id,
        contract=contract,
        steps=(
            RollbackPlanStep(
                order=0,
                undo_kind=undo_kind,
                description=description,
                verification_required=True,
            ),
        ),
        verification_contract=verification_contract,
        requires_user_approval=contract.requires_approval,
        verification_requirement=UndoVerificationRequirement.REQUIRED,
    )


def _rollback_result(
    *,
    status: UndoRollbackStatus,
    decision: UndoRollbackDecision,
    reason: UndoRollbackReason,
    recovery: RecoveryResult | None,
    stack_entry: UndoStackEntry | None,
    plan: RollbackPlan | None,
    verification_result: VerificationResult | None,
    message: str,
    rollback_ready: bool = False,
    rollback_verified: bool = False,
) -> RollbackResult:
    action_id = "unknown_action"
    rollback_id = None
    requires_approval = False

    if recovery is not None:
        action_id = recovery.request.action_id

    if plan is not None:
        rollback_id = plan.rollback_id
        requires_approval = plan.requires_user_approval

    audit = RollbackAudit(
        action_id=action_id,
        rollback_id=rollback_id,
        status=status,
        decision=decision,
        reason=reason,
        requires_approval=requires_approval,
        verified=rollback_verified,
        audit_preserved=True,
    )

    confidence = 0.84
    if status in {
        UndoRollbackStatus.BLOCKED,
        UndoRollbackStatus.FAILED,
    }:
        confidence = 0.30

    if rollback_verified:
        confidence = 0.92

    return RollbackResult(
        status=status,
        decision=decision,
        reason=reason,
        recovery=recovery,
        stack_entry=stack_entry,
        rollback_plan=plan,
        verification_result=verification_result,
        audit=audit,
        trust=_trust(confidence=confidence, reason=message),
        rollback_ready=rollback_ready,
        rollback_verified=rollback_verified,
        message=message,
    )


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
        metadata={"policy": TrustPolicyClassification.REVIEW.value},
    )


def _clean_required(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("field cannot be empty.")
    return cleaned