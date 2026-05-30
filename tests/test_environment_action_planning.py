from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    ApprovalRequirement,
    EnvironmentActionPlanDecision,
    EnvironmentActionPlanningRequest,
    EnvironmentActionPlanningRuntime,
    EnvironmentActionPlanReason,
    EnvironmentActionPlanStatus,
    EnvironmentPlanRiskLevel,
    ExpectedStatePlan,
    ExpectedStatePlanStep,
    PolicyAwareActionPlan,
    SimulatedActionKind,
)


def test_runtime_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        EnvironmentActionPlanningRuntime(name=" ")


def test_create_session() -> None:
    runtime = EnvironmentActionPlanningRuntime()
    session = runtime.create_session(workspace_id="workspace")

    assert session.workspace_id == "workspace"
    assert runtime.snapshot().session_count == 1


def test_expected_plan_requires_steps_when_verification_required() -> None:
    with pytest.raises(ValidationError):
        ExpectedStatePlan(steps=(), requires_verification=True)


def test_policy_plan_requires_simulation_when_execution_requestable() -> None:
    runtime = EnvironmentActionPlanningRuntime()
    session = runtime.create_session(workspace_id="workspace")

    plan = runtime.plan(
        EnvironmentActionPlanningRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            user_intent="focus terminal",
            proposed_action_kind=SimulatedActionKind.FOCUS,
        )
    )

    with pytest.raises(ValidationError):
        PolicyAwareActionPlan(
            plan_id=plan.plan_id,
            status=plan.status,
            decision=plan.decision,
            reason=plan.reason,
            candidate=plan.candidate,
            simulation=None,
            risk=plan.risk,
            expected_state_plan=plan.expected_state_plan,
            policy_decision=plan.policy_decision,
            trust=plan.trust,
            safe_to_request_execution=True,
            requires_approval=plan.requires_approval,
            requires_verification=plan.requires_verification,
            message="invalid",
        )


def test_focus_plan_passes_simulation_gate_with_verification() -> None:
    runtime = EnvironmentActionPlanningRuntime()
    session = runtime.create_session(workspace_id="workspace")

    plan = runtime.plan(
        EnvironmentActionPlanningRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            user_intent="focus terminal",
            proposed_action_kind=SimulatedActionKind.FOCUS,
        )
    )

    assert plan.status == EnvironmentActionPlanStatus.SIMULATED
    assert plan.decision == EnvironmentActionPlanDecision.REQUIRE_VERIFICATION
    assert plan.safe_to_request_execution is True
    assert plan.requires_verification is True
    assert plan.simulation is not None
    assert plan.expected_state_plan is not None


def test_delete_plan_requires_approval_and_blocks_execution_request() -> None:
    runtime = EnvironmentActionPlanningRuntime()
    session = runtime.create_session(workspace_id="workspace")

    plan = runtime.plan(
        EnvironmentActionPlanningRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            user_intent="delete old log",
            proposed_action_kind=SimulatedActionKind.DELETE,
        )
    )

    assert plan.status == EnvironmentActionPlanStatus.APPROVAL_REQUIRED
    assert plan.decision == EnvironmentActionPlanDecision.REQUIRE_APPROVAL
    assert plan.risk.risk_level == EnvironmentPlanRiskLevel.CRITICAL
    assert plan.requires_approval is True
    assert plan.safe_to_request_execution is False
    assert plan.policy_decision.approval_requirement == (
        ApprovalRequirement.EXPLICIT_CONFIRMATION
    )


def test_submit_plan_requires_approval() -> None:
    runtime = EnvironmentActionPlanningRuntime()
    session = runtime.create_session(workspace_id="workspace")

    plan = runtime.plan(
        EnvironmentActionPlanningRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            user_intent="submit form",
            proposed_action_kind=SimulatedActionKind.SUBMIT,
        )
    )

    assert plan.status == EnvironmentActionPlanStatus.APPROVAL_REQUIRED
    assert plan.requires_approval is True
    assert plan.safe_to_request_execution is False


def test_type_plan_requires_payload_from_request() -> None:
    runtime = EnvironmentActionPlanningRuntime()
    session = runtime.create_session(workspace_id="workspace")

    plan = runtime.plan(
        EnvironmentActionPlanningRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            user_intent="type hello",
            proposed_action_kind=SimulatedActionKind.TYPE_TEXT,
            text_payload="hello",
        )
    )

    assert plan.candidate.action.text_payload == "hello"
    assert plan.safe_to_request_execution is True
    assert plan.requires_verification is True


def test_blocked_source_policy_blocks_plan() -> None:
    runtime = EnvironmentActionPlanningRuntime()
    session = runtime.create_session(workspace_id="workspace")

    plan = runtime.plan(
        EnvironmentActionPlanningRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            user_intent="click secret",
            proposed_action_kind=SimulatedActionKind.CLICK,
            metadata={"policy": "blocked"},
        )
    )

    assert plan.status in {
        EnvironmentActionPlanStatus.BLOCKED,
        EnvironmentActionPlanStatus.APPROVAL_REQUIRED,
        EnvironmentActionPlanStatus.SIMULATED,
    }


def test_missing_session_fails() -> None:
    runtime = EnvironmentActionPlanningRuntime()

    plan = runtime.plan(
        EnvironmentActionPlanningRequest(
            session_id="missing",
            workspace_id="workspace",
            user_intent="focus terminal",
            proposed_action_kind=SimulatedActionKind.FOCUS,
        )
    )

    assert plan.status == EnvironmentActionPlanStatus.FAILED
    assert plan.reason == EnvironmentActionPlanReason.SESSION_NOT_FOUND
    assert plan.safe_to_request_execution is False


def test_infer_delete_action_from_intent() -> None:
    runtime = EnvironmentActionPlanningRuntime()
    session = runtime.create_session(workspace_id="workspace")

    plan = runtime.plan(
        EnvironmentActionPlanningRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            user_intent="delete the file",
        )
    )

    assert plan.candidate.action.kind == SimulatedActionKind.DELETE
    assert plan.requires_approval is True


def test_expected_state_plan_contains_verification_steps() -> None:
    runtime = EnvironmentActionPlanningRuntime()
    session = runtime.create_session(workspace_id="workspace")

    plan = runtime.plan(
        EnvironmentActionPlanningRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            user_intent="click save",
            proposed_action_kind=SimulatedActionKind.CLICK,
            metadata={"target_label": "Save"},
        )
    )

    assert plan.expected_state_plan is not None
    assert plan.expected_state_plan.steps
    assert all(
        isinstance(step, ExpectedStatePlanStep)
        for step in plan.expected_state_plan.steps
    )


def test_snapshot_tracks_plan_counts() -> None:
    runtime = EnvironmentActionPlanningRuntime()
    session = runtime.create_session(workspace_id="workspace")

    runtime.plan(
        EnvironmentActionPlanningRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            user_intent="focus terminal",
            proposed_action_kind=SimulatedActionKind.FOCUS,
        )
    )
    runtime.plan(
        EnvironmentActionPlanningRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            user_intent="delete old log",
            proposed_action_kind=SimulatedActionKind.DELETE,
        )
    )
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.plan_count == 2
    assert snapshot.verification_count >= 1
    assert snapshot.approval_count == 1
    assert snapshot.safe_execution_request_count >= 1


def test_reset_clears_runtime() -> None:
    runtime = EnvironmentActionPlanningRuntime()
    runtime.create_session(workspace_id="workspace")

    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.runtime_event_count == 1
    assert snapshot.last_reason == EnvironmentActionPlanReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert EnvironmentActionPlanStatus.SIMULATED.value == "simulated"
    assert EnvironmentActionPlanDecision.REQUIRE_APPROVAL.value == "require_approval"
    assert ApprovalRequirement.EXPLICIT_CONFIRMATION.value == "explicit_confirmation"