from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    ActionVerificationStatus,
    EnvironmentRecoveryDecision,
    EnvironmentRecoveryResult,
    EnvironmentRecoveryStatus,
    MutatingActionKind,
    ReversibilityContract,
    ReversibilityLevel,
    UndoActionKind,
    UndoRollbackDecision,
    UndoRollbackReason,
    UndoRollbackRuntime,
    UndoRollbackStatus,
    VerificationContract,
    VerificationDecision,
    VerificationStateKind,
    VerificationTargetKind,
    expected_bool_state,
)
from jarvis.environment.undo_rollback import RollbackAudit
from jarvis.environment.verification_runtime import VerificationResult


def test_runtime_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        UndoRollbackRuntime(name=" ")


def test_reversible_contract_requires_undo_kind() -> None:
    with pytest.raises(ValidationError):
        ReversibilityContract(
            action_id="action",
            workspace_id="workspace",
            mutation_kind=MutatingActionKind.TEXT_EDIT,
            reversibility=ReversibilityLevel.REVERSIBLE,
            undo_description="revert text",
        )


def test_reversible_contract_requires_undo_description() -> None:
    with pytest.raises(ValidationError):
        ReversibilityContract(
            action_id="action",
            workspace_id="workspace",
            mutation_kind=MutatingActionKind.TEXT_EDIT,
            reversibility=ReversibilityLevel.REVERSIBLE,
            undo_kind=UndoActionKind.TEXT_REVERT,
        )


def test_irreversible_contract_requires_reason_and_approval() -> None:
    with pytest.raises(ValidationError):
        ReversibilityContract(
            action_id="action",
            workspace_id="workspace",
            mutation_kind=MutatingActionKind.FILE_DELETE,
            reversibility=ReversibilityLevel.IRREVERSIBLE,
        )

    with pytest.raises(ValidationError):
        ReversibilityContract(
            action_id="action",
            workspace_id="workspace",
            mutation_kind=MutatingActionKind.FILE_DELETE,
            reversibility=ReversibilityLevel.IRREVERSIBLE,
            irreversible_reason="permanent delete",
            requires_approval=False,
        )


def test_unknown_reversibility_requires_approval() -> None:
    with pytest.raises(ValidationError):
        ReversibilityContract(
            action_id="action",
            workspace_id="workspace",
            mutation_kind=MutatingActionKind.SETTINGS_CHANGE,
            reversibility=ReversibilityLevel.UNKNOWN,
        )


def test_backup_required_requires_reference() -> None:
    with pytest.raises(ValidationError):
        ReversibilityContract(
            action_id="action",
            workspace_id="workspace",
            mutation_kind=MutatingActionKind.FILE_WRITE,
            reversibility=ReversibilityLevel.REVERSIBLE,
            undo_kind=UndoActionKind.FILE_RESTORE,
            undo_description="restore file backup",
            backup_required=True,
        )


def test_rollback_audit_requires_preserved_trail() -> None:
    with pytest.raises(ValidationError):
        RollbackAudit(
            action_id="action",
            status=UndoRollbackStatus.BLOCKED,
            decision=UndoRollbackDecision.BLOCK,
            reason=UndoRollbackReason.UNDO_DECLARATION_MISSING,
            audit_preserved=False,
        )


def test_create_session() -> None:
    runtime = UndoRollbackRuntime()
    session = runtime.create_session(workspace_id="workspace")

    assert session.workspace_id == "workspace"
    assert runtime.snapshot().session_count == 1
    assert runtime.stack_for(session.session_id) is not None


def test_missing_undo_declaration_blocks_mutation() -> None:
    runtime = UndoRollbackRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.declare_undo(
        session_id=session.session_id,
        contract=None,
    )

    assert result.status == UndoRollbackStatus.BLOCKED
    assert result.decision == UndoRollbackDecision.REQUIRE_UNDO_DECLARATION
    assert result.reason == UndoRollbackReason.UNDO_DECLARATION_MISSING
    assert result.mutation_allowed is False


def test_reversible_action_declares_undo_and_pushes_stack() -> None:
    runtime = UndoRollbackRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.declare_undo(
        session_id=session.session_id,
        contract=_contract(),
    )

    stack = runtime.stack_for(session.session_id)

    assert result.status == UndoRollbackStatus.DECLARED
    assert result.decision == UndoRollbackDecision.ALLOW_MUTATION
    assert result.mutation_allowed is True
    assert stack is not None
    assert stack.snapshot().active_count == 1
    assert stack.snapshot().top_action_id == "action_1"


def test_irreversible_action_requires_approval_and_does_not_push_stack() -> None:
    runtime = UndoRollbackRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.declare_undo(
        session_id=session.session_id,
        contract=ReversibilityContract(
            action_id="action_1",
            workspace_id="workspace",
            mutation_kind=MutatingActionKind.FILE_DELETE,
            reversibility=ReversibilityLevel.IRREVERSIBLE,
            irreversible_reason="permanent delete",
            requires_approval=True,
        ),
    )

    stack = runtime.stack_for(session.session_id)

    assert result.status == UndoRollbackStatus.APPROVAL_REQUIRED
    assert result.decision == UndoRollbackDecision.REQUIRE_APPROVAL
    assert result.mutation_allowed is False
    assert stack is not None
    assert stack.snapshot().active_count == 0


def test_unknown_reversibility_blocks() -> None:
    runtime = UndoRollbackRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.declare_undo(
        session_id=session.session_id,
        contract=ReversibilityContract(
            action_id="action_1",
            workspace_id="workspace",
            mutation_kind=MutatingActionKind.SETTINGS_CHANGE,
            reversibility=ReversibilityLevel.UNKNOWN,
            requires_approval=True,
        ),
    )

    assert result.status == UndoRollbackStatus.BLOCKED
    assert result.reason == UndoRollbackReason.UNKNOWN_REVERSIBILITY_BLOCKED


def test_plan_rollback_requires_recovery_decision() -> None:
    runtime = UndoRollbackRuntime()
    session = runtime.create_session(workspace_id="workspace")
    runtime.declare_undo(session_id=session.session_id, contract=_contract())

    result = runtime.plan_rollback(
        session_id=session.session_id,
        recovery=_recovery(decision=EnvironmentRecoveryDecision.RETRY),
        verification_contract=_verification_contract(),
    )

    assert result.status == UndoRollbackStatus.BLOCKED
    assert result.reason == UndoRollbackReason.RECOVERY_NOT_ROLLBACK


def test_plan_rollback_requires_stack_entry() -> None:
    runtime = UndoRollbackRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.plan_rollback(
        session_id=session.session_id,
        recovery=_recovery(decision=EnvironmentRecoveryDecision.ROLLBACK),
        verification_contract=_verification_contract(),
    )

    assert result.status == UndoRollbackStatus.BLOCKED
    assert result.reason == UndoRollbackReason.UNDO_DECLARATION_MISSING


def test_plan_rollback_creates_plan_from_stack() -> None:
    runtime = UndoRollbackRuntime()
    session = runtime.create_session(workspace_id="workspace")
    runtime.declare_undo(session_id=session.session_id, contract=_contract())

    result = runtime.plan_rollback(
        session_id=session.session_id,
        recovery=_recovery(decision=EnvironmentRecoveryDecision.ROLLBACK),
        verification_contract=_verification_contract(),
    )

    assert result.status == UndoRollbackStatus.ROLLBACK_READY
    assert result.decision == UndoRollbackDecision.PREPARE_ROLLBACK
    assert result.rollback_ready is True
    assert result.rollback_plan is not None
    assert result.rollback_plan.verification_contract is not None
    assert result.audit.audit_preserved is True


def test_verify_rollback_passes_only_when_verified() -> None:
    runtime = UndoRollbackRuntime()
    session = runtime.create_session(workspace_id="workspace")
    runtime.declare_undo(session_id=session.session_id, contract=_contract())
    planned = runtime.plan_rollback(
        session_id=session.session_id,
        recovery=_recovery(decision=EnvironmentRecoveryDecision.ROLLBACK),
        verification_contract=_verification_contract(),
    )

    verified = runtime.verify_rollback(
        session_id=session.session_id,
        rollback_result=planned,
        verification_result=_verification_result(
            status=ActionVerificationStatus.PASSED,
            decision=VerificationDecision.COMPLETE,
        ),
    )

    assert verified.status == UndoRollbackStatus.DECLARED
    assert verified.reason == UndoRollbackReason.ROLLBACK_VERIFIED
    assert verified.rollback_verified is True
    assert verified.audit.verified is True


def test_verify_rollback_requires_review_when_verification_fails() -> None:
    runtime = UndoRollbackRuntime()
    session = runtime.create_session(workspace_id="workspace")
    runtime.declare_undo(session_id=session.session_id, contract=_contract())
    planned = runtime.plan_rollback(
        session_id=session.session_id,
        recovery=_recovery(decision=EnvironmentRecoveryDecision.ROLLBACK),
        verification_contract=_verification_contract(),
    )

    verified = runtime.verify_rollback(
        session_id=session.session_id,
        rollback_result=planned,
        verification_result=_verification_result(
            status=ActionVerificationStatus.RECOVERY_NEEDED,
            decision=VerificationDecision.REQUIRE_RECOVERY,
        ),
    )

    assert verified.status == UndoRollbackStatus.VERIFICATION_REQUIRED
    assert verified.decision == UndoRollbackDecision.REQUIRE_VERIFICATION
    assert verified.rollback_verified is False


def test_missing_session_blocks_declaration_and_rollback() -> None:
    runtime = UndoRollbackRuntime()

    declaration = runtime.declare_undo(
        session_id="missing",
        contract=_contract(),
    )
    rollback = runtime.plan_rollback(
        session_id="missing",
        recovery=_recovery(decision=EnvironmentRecoveryDecision.ROLLBACK),
        verification_contract=_verification_contract(),
    )

    assert declaration.status == UndoRollbackStatus.FAILED
    assert declaration.reason == UndoRollbackReason.SESSION_NOT_FOUND
    assert rollback.status == UndoRollbackStatus.FAILED
    assert rollback.reason == UndoRollbackReason.SESSION_NOT_FOUND


def test_snapshot_tracks_counts() -> None:
    runtime = UndoRollbackRuntime()
    session = runtime.create_session(workspace_id="workspace")
    runtime.declare_undo(session_id=session.session_id, contract=_contract())
    planned = runtime.plan_rollback(
        session_id=session.session_id,
        recovery=_recovery(decision=EnvironmentRecoveryDecision.ROLLBACK),
        verification_contract=_verification_contract(),
    )
    runtime.verify_rollback(
        session_id=session.session_id,
        rollback_result=planned,
        verification_result=_verification_result(
            status=ActionVerificationStatus.PASSED,
            decision=VerificationDecision.COMPLETE,
        ),
    )

    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.declaration_count == 1
    assert snapshot.rollback_result_count == 2
    assert snapshot.rollback_ready_count == 1
    assert snapshot.verified_rollback_count == 1
    assert snapshot.audit_count == 2


def test_session_tracks_counts() -> None:
    runtime = UndoRollbackRuntime()
    session = runtime.create_session(workspace_id="workspace")
    runtime.declare_undo(session_id=session.session_id, contract=_contract())
    planned = runtime.plan_rollback(
        session_id=session.session_id,
        recovery=_recovery(decision=EnvironmentRecoveryDecision.ROLLBACK),
        verification_contract=_verification_contract(),
    )
    runtime.verify_rollback(
        session_id=session.session_id,
        rollback_result=planned,
        verification_result=_verification_result(
            status=ActionVerificationStatus.PASSED,
            decision=VerificationDecision.COMPLETE,
        ),
    )

    stored = runtime.session_for(session.session_id)

    assert stored is not None
    assert stored.declaration_count == 1
    assert stored.rollback_plan_count == 1
    assert stored.verified_rollback_count == 1


def test_reset_clears_runtime() -> None:
    runtime = UndoRollbackRuntime()
    runtime.create_session(workspace_id="workspace")

    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.runtime_event_count == 1
    assert snapshot.last_reason == UndoRollbackReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert UndoActionKind.TEXT_REVERT.value == "text_revert"
    assert MutatingActionKind.FILE_WRITE.value == "file_write"
    assert ReversibilityLevel.REVERSIBLE.value == "reversible"


def _contract() -> ReversibilityContract:
    return ReversibilityContract(
        action_id="action_1",
        workspace_id="workspace",
        mutation_kind=MutatingActionKind.TEXT_EDIT,
        reversibility=ReversibilityLevel.REVERSIBLE,
        undo_kind=UndoActionKind.TEXT_REVERT,
        undo_description="restore previous text snapshot",
        backup_required=True,
        backup_reference="backup://action_1",
    )


def _verification_contract() -> VerificationContract:
    expected = expected_bool_state(
        key="rollback.verified",
        value=True,
        kind=VerificationStateKind.STATUS_EQUALS,
        target=VerificationTargetKind.WORKSPACE_GRAPH,
        description="rollback should be verified",
    )

    return VerificationContract(
        action_id="action_1",
        workspace_id="workspace",
        expected_states=(expected,),
    )


def _recovery(
    *,
    decision: EnvironmentRecoveryDecision,
) -> EnvironmentRecoveryResult:
    retry_allowed = decision == EnvironmentRecoveryDecision.RETRY
    rollback_required = decision == EnvironmentRecoveryDecision.ROLLBACK
    escalation_required = decision == EnvironmentRecoveryDecision.ESCALATE

    status = EnvironmentRecoveryStatus.ROLLBACK_READY
    if retry_allowed:
        status = EnvironmentRecoveryStatus.RETRY_READY

    return EnvironmentRecoveryResult.model_construct(
        status=status,
        decision=decision,
        request=type(
            "RecoveryRequestStub",
            (),
            {"action_id": "action_1"},
        )(),
        retry_allowed=retry_allowed,
        rollback_required=rollback_required,
        escalation_required=escalation_required,
    )


def _verification_result(
    *,
    status: ActionVerificationStatus,
    decision: VerificationDecision,
) -> VerificationResult:
    return VerificationResult.model_construct(
        status=status,
        decision=decision,
    )