from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    EnvironmentActionPlanDecision,
    EnvironmentActionPlanningRequest,
    EnvironmentActionPlanningRuntime,
    EnvironmentActionPlanStatus,
    InteractionApprovalStatus,
    InteractionDecision,
    InteractionPermission,
    InteractionPolicyReason,
    InteractionPolicyRuntime,
    InteractionUndoRequirement,
    InteractionValidationStatus,
    InteractionVerificationRequirement,
    PhysicalActionContract,
    PhysicalInteractionKind,
    PhysicalInteractionRequest,
    PhysicalInteractionRisk,
    PolicyAwareActionPlan,
    SimulatedActionKind,
    contract_from_plan,
)


def test_runtime_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        InteractionPolicyRuntime(name=" ")


def test_create_session() -> None:
    runtime = InteractionPolicyRuntime()
    session = runtime.create_session(workspace_id="workspace")

    assert session.workspace_id == "workspace"
    assert runtime.snapshot().session_count == 1


def test_execution_eligible_requires_matching_decision() -> None:
    policy = InteractionPolicyRuntime()
    session = policy.create_session(workspace_id="workspace")
    plan = _plan(
        action=SimulatedActionKind.FOCUS,
        user_intent="focus terminal",
    )
    request = PhysicalInteractionRequest(
        session_id=session.session_id,
        workspace_id="workspace",
        contract=contract_from_plan(plan),
        plan=plan,
    )
    result = policy.evaluate(request)

    with pytest.raises(ValidationError):
        type(result)(
            status=result.status,
            decision=InteractionDecision.BLOCKED,
            reason=result.reason,
            request=result.request,
            risk=result.risk,
            permission=result.permission,
            validation=result.validation,
            approval=result.approval,
            verification_requirement=result.verification_requirement,
            undo_requirement=result.undo_requirement,
            audit=result.audit,
            trust=result.trust,
            execution_eligible=True,
            message="invalid",
        )


def test_focus_contract_requires_verification_first() -> None:
    runtime = InteractionPolicyRuntime()
    session = runtime.create_session(workspace_id="workspace")
    plan = _plan(
        action=SimulatedActionKind.FOCUS,
        user_intent="focus terminal",
    )
    request = PhysicalInteractionRequest(
        session_id=session.session_id,
        workspace_id="workspace",
        contract=contract_from_plan(plan),
        plan=plan,
    )

    result = runtime.evaluate(request)

    assert result.decision == InteractionDecision.REQUIRES_VERIFICATION_FIRST
    assert result.execution_eligible is False
    assert result.permission.permission == (
        InteractionPermission.ALLOW_WITH_VERIFICATION
    )
    assert result.verification_requirement in {
        InteractionVerificationRequirement.VERIFY_TARGET_STILL_VALID,
        InteractionVerificationRequirement.VERIFY_EXPECTED_STATE,
    }
    assert result.audit is not None


def test_type_contract_requires_pre_state_and_verification() -> None:
    runtime = InteractionPolicyRuntime()
    session = runtime.create_session(workspace_id="workspace")
    plan = _plan(
        action=SimulatedActionKind.TYPE_TEXT,
        user_intent="type hello",
        text_payload="hello",
    )
    request = PhysicalInteractionRequest(
        session_id=session.session_id,
        workspace_id="workspace",
        contract=contract_from_plan(plan),
        plan=plan,
    )

    result = runtime.evaluate(request)

    assert result.undo_requirement == InteractionUndoRequirement.CAPTURE_PRE_STATE
    assert result.execution_eligible is False
    assert result.verification_requirement != InteractionVerificationRequirement.NONE


def test_delete_contract_waits_for_approval() -> None:
    runtime = InteractionPolicyRuntime()
    session = runtime.create_session(workspace_id="workspace")
    plan = _plan(
        action=SimulatedActionKind.DELETE,
        user_intent="delete old log",
    )
    request = PhysicalInteractionRequest(
        session_id=session.session_id,
        workspace_id="workspace",
        contract=contract_from_plan(plan),
        plan=plan,
    )

    result = runtime.evaluate(request)

    assert result.decision in {
        InteractionDecision.WAITING_FOR_APPROVAL,
        InteractionDecision.BLOCKED,
    }
    assert result.execution_eligible is False
    assert result.audit.risk in {
        PhysicalInteractionRisk.CRITICAL,
        PhysicalInteractionRisk.BLOCKED,
    }


def test_submit_contract_waits_for_approval() -> None:
    runtime = InteractionPolicyRuntime()
    session = runtime.create_session(workspace_id="workspace")
    plan = _plan(
        action=SimulatedActionKind.SUBMIT,
        user_intent="submit form",
    )
    request = PhysicalInteractionRequest(
        session_id=session.session_id,
        workspace_id="workspace",
        contract=contract_from_plan(plan),
        plan=plan,
    )

    result = runtime.evaluate(request)

    assert result.decision in {
        InteractionDecision.WAITING_FOR_APPROVAL,
        InteractionDecision.BLOCKED,
    }
    assert result.execution_eligible is False


def test_missing_plan_blocks_interaction() -> None:
    runtime = InteractionPolicyRuntime()
    session = runtime.create_session(workspace_id="workspace")
    contract = PhysicalActionContract(
        kind=PhysicalInteractionKind.MOUSE_CLICK,
        description="click without plan",
        target_label="Run",
    )
    request = PhysicalInteractionRequest(
        session_id=session.session_id,
        workspace_id="workspace",
        contract=contract,
        plan=None,
    )

    result = runtime.evaluate(request)

    assert result.decision == InteractionDecision.BLOCKED
    assert result.permission.permission == InteractionPermission.DENY
    assert result.validation.status == InteractionValidationStatus.BLOCKED


def test_unknown_contract_blocks() -> None:
    runtime = InteractionPolicyRuntime()
    session = runtime.create_session(workspace_id="workspace")
    plan = _plan(
        action=SimulatedActionKind.UNKNOWN,
        user_intent="unknown risky thing",
    )
    request = PhysicalInteractionRequest(
        session_id=session.session_id,
        workspace_id="workspace",
        contract=PhysicalActionContract(
            kind=PhysicalInteractionKind.UNKNOWN,
            description="unknown risky thing",
            source_plan_id=plan.plan_id,
        ),
        plan=plan,
    )

    result = runtime.evaluate(request)

    assert result.decision == InteractionDecision.BLOCKED
    assert result.execution_eligible is False


def test_missing_session_blocks() -> None:
    runtime = InteractionPolicyRuntime()
    plan = _plan(
        action=SimulatedActionKind.FOCUS,
        user_intent="focus terminal",
    )
    request = PhysicalInteractionRequest(
        session_id="missing",
        workspace_id="workspace",
        contract=contract_from_plan(plan),
        plan=plan,
    )

    result = runtime.evaluate(request)

    assert result.decision == InteractionDecision.BLOCKED
    assert result.reason == InteractionPolicyReason.SESSION_NOT_FOUND
    assert result.execution_eligible is False


def test_contract_from_plan_maps_kinds() -> None:
    delete_plan = _plan(
        action=SimulatedActionKind.DELETE,
        user_intent="delete file",
    )
    type_plan = _plan(
        action=SimulatedActionKind.TYPE_TEXT,
        user_intent="type hello",
        text_payload="hello",
    )

    assert contract_from_plan(delete_plan).kind == PhysicalInteractionKind.FILE_DELETE
    assert contract_from_plan(type_plan).kind == PhysicalInteractionKind.KEYBOARD_TYPE
    assert contract_from_plan(type_plan).text_payload_present is True


def test_snapshot_tracks_results_and_audits() -> None:
    runtime = InteractionPolicyRuntime()
    session = runtime.create_session(workspace_id="workspace")
    focus_plan = _plan(
        action=SimulatedActionKind.FOCUS,
        user_intent="focus terminal",
    )
    delete_plan = _plan(
        action=SimulatedActionKind.DELETE,
        user_intent="delete old log",
    )

    runtime.evaluate(
        PhysicalInteractionRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            contract=contract_from_plan(focus_plan),
            plan=focus_plan,
        )
    )
    runtime.evaluate(
        PhysicalInteractionRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            contract=contract_from_plan(delete_plan),
            plan=delete_plan,
        )
    )
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.result_count == 2
    assert snapshot.audit_count == 2
    assert snapshot.verification_required_count >= 1
    assert snapshot.blocked_count >= 0


def test_reset_clears_runtime() -> None:
    runtime = InteractionPolicyRuntime()
    runtime.create_session(workspace_id="workspace")

    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.runtime_event_count == 1
    assert snapshot.last_reason == InteractionPolicyReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert PhysicalInteractionKind.MOUSE_CLICK.value == "mouse_click"
    assert InteractionPermission.REQUIRE_APPROVAL.value == "require_approval"
    assert InteractionApprovalStatus.PENDING.value == "pending"


def _plan(
    *,
    action: SimulatedActionKind,
    user_intent: str,
    text_payload: str | None = None,
) -> PolicyAwareActionPlan:
    runtime = EnvironmentActionPlanningRuntime()
    session = runtime.create_session(workspace_id="workspace")

    plan = runtime.plan(
        EnvironmentActionPlanningRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            user_intent=user_intent,
            proposed_action_kind=action,
            text_payload=text_payload,
        )
    )

    if action in {SimulatedActionKind.FOCUS, SimulatedActionKind.TYPE_TEXT}:
        assert plan.status in {
            EnvironmentActionPlanStatus.SIMULATED,
            EnvironmentActionPlanStatus.READY_FOR_EXECUTION,
        }
        assert plan.decision in {
            EnvironmentActionPlanDecision.REQUIRE_VERIFICATION,
            EnvironmentActionPlanDecision.ALLOW_EXECUTION_REQUEST,
        }

    return plan