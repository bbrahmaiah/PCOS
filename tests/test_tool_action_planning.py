from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.cognition import (
    CognitionRequest,
    ToolActionParameter,
    ToolActionPermissionMode,
    ToolActionPlanner,
    ToolActionPlannerConfig,
    ToolActionRiskLevel,
    ToolActionTargetKind,
    ToolActionType,
)


def make_request(
    *,
    request_id: str = "request-1",
    text: str = "open notepad",
) -> CognitionRequest:
    return CognitionRequest(
        request_id=request_id,
        text=text,
    )


def test_tool_action_planner_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        ToolActionPlannerConfig(name=" ").validate()

    with pytest.raises(ValueError):
        ToolActionPlannerConfig(dangerous_phrases=("",)).validate()

    with pytest.raises(ValueError):
        ToolActionPlannerConfig(high_risk_phrases=("",)).validate()

    with pytest.raises(ValueError):
        ToolActionPlannerConfig(action_phrases=("",)).validate()


def test_tool_action_parameter_rejects_invalid_values() -> None:
    with pytest.raises(ValidationError):
        ToolActionParameter(name=" ", value="hello")

    with pytest.raises(ValidationError):
        ToolActionParameter(name="target", value=" ")


def test_tool_action_planner_creates_open_application_plan() -> None:
    planner = ToolActionPlanner()

    plan = planner.plan(make_request(text="open notepad"))
    proposal = plan.proposals[0]

    assert proposal.action_type == ToolActionType.OPEN_APPLICATION
    assert proposal.target_kind == ToolActionTargetKind.APPLICATION
    assert proposal.target == "notepad"
    assert proposal.risk_level == ToolActionRiskLevel.LOW
    assert proposal.permission_mode == ToolActionPermissionMode.CONFIRMATION_REQUIRED
    assert proposal.executable is True
    assert plan.executable is True
    assert plan.blocked is False
    assert plan.metadata["llm_direct_execution_allowed"] is False


def test_tool_action_planner_can_auto_allow_low_risk_readonly() -> None:
    planner = ToolActionPlanner(
        config=ToolActionPlannerConfig(auto_allow_low_risk_readonly=True)
    )

    plan = planner.plan(make_request(text="open calculator"))

    assert plan.safety.allowed is True
    assert plan.safety.permission_mode == ToolActionPermissionMode.AUTO_ALLOWED
    assert plan.executable is True


def test_tool_action_planner_creates_open_file_plan() -> None:
    planner = ToolActionPlanner()

    plan = planner.plan(make_request(text="open file notes.txt"))
    proposal = plan.proposals[0]

    assert proposal.action_type == ToolActionType.OPEN_FILE
    assert proposal.target_kind == ToolActionTargetKind.FILE
    assert proposal.target == "file notes.txt"


def test_tool_action_planner_creates_read_file_plan() -> None:
    planner = ToolActionPlanner()

    plan = planner.plan(make_request(text="read report.pdf"))
    proposal = plan.proposals[0]

    assert proposal.action_type == ToolActionType.READ_FILE
    assert proposal.target_kind == ToolActionTargetKind.FILE
    assert proposal.risk_level == ToolActionRiskLevel.LOW


def test_tool_action_planner_creates_write_file_plan() -> None:
    planner = ToolActionPlanner()

    plan = planner.plan(make_request(text="create file notes.txt"))
    proposal = plan.proposals[0]

    assert proposal.action_type == ToolActionType.WRITE_FILE
    assert proposal.risk_level == ToolActionRiskLevel.MEDIUM
    assert proposal.permission_mode == ToolActionPermissionMode.CONFIRMATION_REQUIRED


def test_tool_action_planner_marks_delete_as_high_risk() -> None:
    planner = ToolActionPlanner()

    plan = planner.plan(make_request(text="delete old logs"))
    proposal = plan.proposals[0]

    assert proposal.action_type == ToolActionType.DELETE_FILE
    assert proposal.risk_level == ToolActionRiskLevel.HIGH
    assert proposal.permission_mode == (
        ToolActionPermissionMode.ELEVATED_CONFIRMATION_REQUIRED
    )
    assert plan.safety.allowed is True


def test_tool_action_planner_blocks_dangerous_request() -> None:
    planner = ToolActionPlanner()

    plan = planner.plan(make_request(text="delete system32"))

    assert plan.blocked is True
    assert plan.executable is False
    assert plan.safety.allowed is False
    assert plan.safety.risk_level == ToolActionRiskLevel.CRITICAL
    assert plan.safety.permission_mode == ToolActionPermissionMode.BLOCKED
    assert "request matched blocked dangerous phrase" in plan.safety.reasons


def test_tool_action_planner_blocks_terminal_by_default() -> None:
    planner = ToolActionPlanner()

    plan = planner.plan(make_request(text="run powershell Get-Process"))

    assert plan.blocked is True
    assert plan.safety.allowed is False
    assert plan.safety.permission_mode == ToolActionPermissionMode.BLOCKED
    assert "terminal command execution is disabled" in plan.safety.reasons


def test_tool_action_planner_allows_terminal_when_configured() -> None:
    planner = ToolActionPlanner(
        config=ToolActionPlannerConfig(allow_terminal_commands=True)
    )

    plan = planner.plan(make_request(text="run echo hello"))

    assert plan.blocked is False
    assert plan.safety.allowed is True
    assert plan.safety.risk_level == ToolActionRiskLevel.HIGH
    assert plan.safety.permission_mode == (
        ToolActionPermissionMode.ELEVATED_CONFIRMATION_REQUIRED
    )


def test_tool_action_planner_blocks_system_control_by_default() -> None:
    planner = ToolActionPlanner()

    plan = planner.plan(make_request(text="shutdown laptop"))

    assert plan.blocked is True
    assert plan.safety.allowed is False
    assert "system control execution is disabled" in plan.safety.reasons


def test_tool_action_planner_search_web_plan() -> None:
    planner = ToolActionPlanner()

    plan = planner.plan(make_request(text="search phase 3 cognition"))
    proposal = plan.proposals[0]

    assert proposal.action_type == ToolActionType.SEARCH_WEB
    assert proposal.target_kind == ToolActionTargetKind.URL
    assert proposal.target == "phase 3 cognition"


def test_tool_action_planner_send_message_plan() -> None:
    planner = ToolActionPlanner()

    plan = planner.plan(make_request(text="send message to Bala"))
    proposal = plan.proposals[0]

    assert proposal.action_type == ToolActionType.SEND_MESSAGE
    assert proposal.target_kind == ToolActionTargetKind.PERSON
    assert proposal.risk_level == ToolActionRiskLevel.MEDIUM


def test_tool_action_planner_schedule_event_plan() -> None:
    planner = ToolActionPlanner()

    plan = planner.plan(make_request(text="schedule study session"))
    proposal = plan.proposals[0]

    assert proposal.action_type == ToolActionType.SCHEDULE_EVENT
    assert proposal.target_kind == ToolActionTargetKind.CALENDAR
    assert proposal.risk_level == ToolActionRiskLevel.MEDIUM


def test_tool_action_planner_unknown_plan_is_safe_contract() -> None:
    planner = ToolActionPlanner()

    plan = planner.plan(make_request(text="think about this"))

    assert plan.proposals[0].action_type == ToolActionType.UNKNOWN
    assert plan.safety.allowed is True
    assert plan.executable is True
    assert plan.metadata["llm_direct_execution_allowed"] is False


def test_tool_action_planner_snapshot_counts() -> None:
    planner = ToolActionPlanner()

    planner.plan(make_request(text="open notepad"))
    planner.plan(make_request(request_id="request-2", text="delete system32"))
    planner.plan(make_request(request_id="request-3", text="create file notes.txt"))

    snapshot = planner.snapshot()

    assert snapshot.planned_count == 3
    assert snapshot.proposal_count == 3
    assert snapshot.allowed_count == 2
    assert snapshot.blocked_count == 1
    assert snapshot.confirmation_required_count == 2
    assert snapshot.last_request_id == "request-3"
    assert snapshot.last_risk_level == ToolActionRiskLevel.MEDIUM


def test_tool_action_planner_reset_clears_counters() -> None:
    planner = ToolActionPlanner()

    planner.plan(make_request())

    planner.reset()
    snapshot = planner.snapshot()

    assert snapshot.planned_count == 0
    assert snapshot.proposal_count == 0
    assert snapshot.allowed_count == 0
    assert snapshot.blocked_count == 0
    assert snapshot.confirmation_required_count == 0
    assert snapshot.last_request_id is None
    assert snapshot.last_error is None


def test_tool_action_enum_values_are_stable() -> None:
    assert ToolActionType.OPEN_APPLICATION.value == "open_application"
    assert ToolActionType.RUN_TERMINAL_COMMAND.value == "run_terminal_command"
    assert ToolActionTargetKind.APPLICATION.value == "application"
    assert ToolActionRiskLevel.CRITICAL.value == "critical"
    assert ToolActionPermissionMode.BLOCKED.value == "blocked"