from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.tools import (
    ActionKind,
    ActionPlanningDecision,
    ActionPlanningIntentKind,
    ActionPlanningReason,
    ActionPlanningRequest,
    ActionRisk,
    ActionScope,
    MultiStepActionPlanner,
    MultiStepActionPlannerConfig,
    PlannedStepRole,
    PlannerStep,
    ToolCapability,
)


def test_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        MultiStepActionPlannerConfig(name=" ").validate()

    with pytest.raises(ValueError):
        MultiStepActionPlannerConfig(default_test_command=" ").validate()

    with pytest.raises(ValueError):
        MultiStepActionPlannerConfig(quality_gate_commands=()).validate()


def test_request_requires_intent() -> None:
    with pytest.raises(ValidationError):
        ActionPlanningRequest(user_intent=" ")


def test_executable_planner_step_requires_tool_role() -> None:
    with pytest.raises(ValidationError):
        PlannerStep(
            order=0,
            role=PlannedStepRole.COGNITIVE,
            title="Think",
            description="Think",
            executable=True,
            action_kind=ActionKind.READ,
            capability=ToolCapability.READ_FILE,
        )


def test_run_tests_and_summarize_plan() -> None:
    planner = MultiStepActionPlanner()

    proposal = planner.propose(
        ActionPlanningRequest(user_intent="run tests and summarize failures")
    )

    assert proposal.decision == ActionPlanningDecision.PROPOSED
    assert proposal.reason == ActionPlanningReason.TEST_PLAN_PROPOSED
    assert proposal.intent_kind == ActionPlanningIntentKind.RUN_TESTS_AND_SUMMARIZE
    assert proposal.action_plan is not None
    assert proposal.action_plan.scope == ActionScope.SHELL
    assert proposal.action_plan.steps[0].kind == ActionKind.SHELL_COMMAND
    assert proposal.action_plan.steps[0].arguments["command"] == "pytest"
    assert len(proposal.planner_steps) == 5
    assert proposal.planner_steps[0].role == PlannedStepRole.TOOL
    assert proposal.planner_steps[1].role == PlannedStepRole.COGNITIVE
    assert proposal.metadata["planner_proposes_only"] is True
    assert proposal.metadata["runtime_execution"] is False


def test_run_tests_uses_preferred_command() -> None:
    planner = MultiStepActionPlanner()

    proposal = planner.propose(
        ActionPlanningRequest(
            user_intent="run pytest",
            preferred_test_command="pytest tests/test_tools_models.py",
        )
    )

    assert proposal.action_plan is not None
    assert proposal.action_plan.steps[0].arguments["command"] == (
        "pytest tests/test_tools_models.py"
    )


def test_quality_gate_plan() -> None:
    planner = MultiStepActionPlanner()

    proposal = planner.propose(
        ActionPlanningRequest(user_intent="run quality gate")
    )

    assert proposal.decision == ActionPlanningDecision.PROPOSED
    assert proposal.intent_kind == ActionPlanningIntentKind.RUN_QUALITY_GATE
    assert proposal.action_plan is not None
    assert len(proposal.action_plan.steps) == 3
    assert proposal.action_plan.steps[0].arguments["command"] == "ruff check ."
    assert proposal.action_plan.steps[1].arguments["command"] == "mypy ."
    assert proposal.action_plan.steps[2].arguments["command"] == "pytest"


def test_open_file_plan_from_target_path() -> None:
    planner = MultiStepActionPlanner()

    proposal = planner.propose(
        ActionPlanningRequest(
            user_intent="open this file",
            target_path="jarvis/tools/planner.py",
        )
    )

    assert proposal.decision == ActionPlanningDecision.PROPOSED
    assert proposal.intent_kind == ActionPlanningIntentKind.OPEN_FILE
    assert proposal.action_plan is not None
    assert proposal.action_plan.steps[0].kind == ActionKind.IDE_OPEN_FILE
    assert proposal.action_plan.steps[0].arguments["path"] == (
        "jarvis/tools/planner.py"
    )


def test_open_file_plan_extracts_path() -> None:
    planner = MultiStepActionPlanner()

    proposal = planner.propose(
        ActionPlanningRequest(
            user_intent="open jarvis/tools/planner.py",
        )
    )

    assert proposal.decision == ActionPlanningDecision.PROPOSED
    assert proposal.action_plan is not None
    assert proposal.action_plan.steps[0].arguments["path"] == (
        "jarvis/tools/planner.py"
    )


def test_open_file_needs_path() -> None:
    planner = MultiStepActionPlanner()

    proposal = planner.propose(
        ActionPlanningRequest(user_intent="open file")
    )

    assert proposal.decision == ActionPlanningDecision.NEEDS_CLARIFICATION
    assert proposal.reason == ActionPlanningReason.MISSING_TARGET
    assert proposal.action_plan is None


def test_search_code_plan() -> None:
    planner = MultiStepActionPlanner()

    proposal = planner.propose(
        ActionPlanningRequest(
            user_intent="search for AdaptiveTurnDetector",
        )
    )

    assert proposal.decision == ActionPlanningDecision.PROPOSED
    assert proposal.intent_kind == ActionPlanningIntentKind.SEARCH_CODE
    assert proposal.action_plan is not None
    assert proposal.action_plan.steps[0].kind == ActionKind.SEARCH
    assert proposal.action_plan.steps[0].arguments["query"] == (
        "AdaptiveTurnDetector"
    )


def test_search_code_uses_explicit_query() -> None:
    planner = MultiStepActionPlanner()

    proposal = planner.propose(
        ActionPlanningRequest(
            user_intent="find code",
            search_query="MemoryGateway",
        )
    )

    assert proposal.action_plan is not None
    assert proposal.action_plan.steps[0].arguments["query"] == "MemoryGateway"


def test_prepare_patch_plan_does_not_apply_patch() -> None:
    planner = MultiStepActionPlanner()

    proposal = planner.propose(
        ActionPlanningRequest(
            user_intent="prepare patch",
            target_path="a.py",
            old_text="old",
            new_text="new",
        )
    )

    assert proposal.decision == ActionPlanningDecision.PROPOSED
    assert proposal.intent_kind == ActionPlanningIntentKind.PREPARE_PATCH
    assert proposal.risk == ActionRisk.MEDIUM
    assert proposal.action_plan is not None
    assert proposal.action_plan.steps[0].kind == ActionKind.PATCH
    assert proposal.action_plan.steps[0].rollback_supported is False
    assert proposal.planner_steps[1].role == PlannedStepRole.USER
    assert proposal.planner_steps[1].requires_approval is True


def test_prepare_patch_requires_target() -> None:
    planner = MultiStepActionPlanner()

    proposal = planner.propose(
        ActionPlanningRequest(
            user_intent="prepare patch",
            old_text="old",
            new_text="new",
        )
    )

    assert proposal.decision == ActionPlanningDecision.NEEDS_CLARIFICATION
    assert proposal.reason == ActionPlanningReason.MISSING_TARGET


def test_prepare_patch_requires_old_and_new_text() -> None:
    planner = MultiStepActionPlanner()

    proposal = planner.propose(
        ActionPlanningRequest(
            user_intent="prepare patch",
            target_path="a.py",
        )
    )

    assert proposal.decision == ActionPlanningDecision.NEEDS_CLARIFICATION
    assert proposal.reason == ActionPlanningReason.MISSING_PATCH_TEXT


def test_unknown_intent_needs_clarification() -> None:
    planner = MultiStepActionPlanner()

    proposal = planner.propose(
        ActionPlanningRequest(user_intent="do something magical")
    )

    assert proposal.decision == ActionPlanningDecision.NEEDS_CLARIFICATION
    assert proposal.reason == ActionPlanningReason.UNSUPPORTED_INTENT
    assert proposal.action_plan is None


def test_snapshot_and_reset() -> None:
    planner = MultiStepActionPlanner()

    planner.propose(
        ActionPlanningRequest(user_intent="run tests and summarize failures")
    )
    snapshot = planner.snapshot()

    assert snapshot.plan_count == 1
    assert snapshot.proposed_count == 1
    assert snapshot.last_decision == ActionPlanningDecision.PROPOSED

    planner.reset()
    reset_snapshot = planner.snapshot()

    assert reset_snapshot.plan_count == 0
    assert reset_snapshot.last_decision is None


def test_enum_values_are_stable() -> None:
    assert ActionPlanningIntentKind.RUN_TESTS_AND_SUMMARIZE.value == (
        "run_tests_and_summarize"
    )
    assert ActionPlanningDecision.PROPOSED.value == "proposed"
    assert PlannedStepRole.COGNITIVE.value == "cognitive"