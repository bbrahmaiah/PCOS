from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.tools import (
    ActionContext,
    ActionError,
    ActionKind,
    ActionPlan,
    ActionRequest,
    ActionResult,
    ActionRisk,
    ActionScope,
    ActionStatus,
    ActionStep,
    PermissionDecision,
    PolicyDecisionRecord,
    ToolCapability,
    ToolId,
    ToolValidationError,
    new_action_id,
    new_action_plan_id,
    new_action_step_id,
    new_tool_id,
)


def test_tool_ids_are_prefixed_and_unique() -> None:
    first = new_tool_id()
    second = new_tool_id()

    assert first.startswith("tool_")
    assert second.startswith("tool_")
    assert first != second


def test_action_ids_are_prefixed_and_unique() -> None:
    first = new_action_id()
    second = new_action_id()

    assert first.startswith("action_")
    assert second.startswith("action_")
    assert first != second


def test_tool_id_model_rejects_empty_value() -> None:
    with pytest.raises(ValidationError):
        ToolId(value=" ")


def test_action_context_cleans_optional_fields() -> None:
    context = ActionContext(
        user_id=" user ",
        session_id=" session ",
        requested_by=" user ",
        memory_context_refs=(" memory-1 ", " memory-2 "),
    )

    assert context.user_id == "user"
    assert context.session_id == "session"
    assert context.requested_by == "user"
    assert context.memory_context_refs == ("memory-1", "memory-2")


def test_action_context_rejects_duplicate_memory_refs() -> None:
    with pytest.raises(ValidationError):
        ActionContext(memory_context_refs=("memory-1", "memory-1"))


def test_action_error_requires_code_and_message() -> None:
    with pytest.raises(ValidationError):
        ActionError(code=" ", message="valid")

    with pytest.raises(ValidationError):
        ActionError(code="VALID", message=" ")


def test_low_risk_action_request_is_valid() -> None:
    request = ActionRequest(
        kind=ActionKind.READ,
        requested_capability=ToolCapability.READ_FILE,
        intent="Read project README",
        scope=ActionScope.WORKSPACE,
        risk=ActionRisk.LOW,
        arguments={"path": "README.md"},
    )

    assert request.action_id.startswith("action_")
    assert request.kind == ActionKind.READ
    assert request.requires_approval is False


def test_high_risk_action_request_requires_approval() -> None:
    with pytest.raises(ValidationError):
        ActionRequest(
            kind=ActionKind.DELETE,
            requested_capability=ToolCapability.DELETE_FILE,
            intent="Delete important file",
            scope=ActionScope.FILE_SYSTEM,
            risk=ActionRisk.HIGH,
        )


def test_high_risk_action_request_valid_with_approval() -> None:
    request = ActionRequest(
        kind=ActionKind.DELETE,
        requested_capability=ToolCapability.DELETE_FILE,
        intent="Delete generated temp file",
        scope=ActionScope.FILE_SYSTEM,
        risk=ActionRisk.HIGH,
        requires_approval=True,
    )

    assert request.requires_approval is True


def test_action_step_requires_description() -> None:
    with pytest.raises(ValidationError):
        ActionStep(
            action_id=new_action_id(),
            order=0,
            kind=ActionKind.READ,
            capability=ToolCapability.READ_FILE,
            scope=ActionScope.WORKSPACE,
            description=" ",
        )


def test_mutating_action_step_must_be_interruptible() -> None:
    with pytest.raises(ValidationError):
        ActionStep(
            action_id=new_action_id(),
            order=0,
            kind=ActionKind.WRITE,
            capability=ToolCapability.WRITE_FILE,
            scope=ActionScope.WORKSPACE,
            description="Write file",
            interruptible=False,
        )


def test_high_risk_action_step_requires_timeout() -> None:
    with pytest.raises(ValidationError):
        ActionStep(
            action_id=new_action_id(),
            order=0,
            kind=ActionKind.SHELL_COMMAND,
            capability=ToolCapability.RUN_SHELL_COMMAND,
            scope=ActionScope.SHELL,
            risk=ActionRisk.HIGH,
            description="Run risky command",
            interruptible=True,
        )


def test_action_step_accepts_timeout_for_high_risk() -> None:
    step = ActionStep(
        action_id=new_action_id(),
        order=0,
        kind=ActionKind.SHELL_COMMAND,
        capability=ToolCapability.RUN_SHELL_COMMAND,
        scope=ActionScope.SHELL,
        risk=ActionRisk.HIGH,
        description="Run approved command",
        interruptible=True,
        timeout_ms=30_000,
    )

    assert step.timeout_ms == 30_000


def test_action_step_rejects_duplicate_dependencies() -> None:
    with pytest.raises(ValidationError):
        ActionStep(
            action_id=new_action_id(),
            order=0,
            kind=ActionKind.READ,
            capability=ToolCapability.READ_FILE,
            scope=ActionScope.WORKSPACE,
            description="Read file",
            depends_on=("step-1", "step-1"),
        )


def test_action_plan_requires_steps() -> None:
    with pytest.raises(ValidationError):
        ActionPlan(
            action_id=new_action_id(),
            goal="Do something",
            steps=(),
            risk=ActionRisk.LOW,
            scope=ActionScope.WORKSPACE,
        )


def test_action_plan_requires_matching_action_ids() -> None:
    action_id = new_action_id()
    step = ActionStep(
        action_id=new_action_id(),
        order=0,
        kind=ActionKind.READ,
        capability=ToolCapability.READ_FILE,
        scope=ActionScope.WORKSPACE,
        description="Read file",
    )

    with pytest.raises(ValidationError):
        ActionPlan(
            action_id=action_id,
            goal="Read file",
            steps=(step,),
            risk=ActionRisk.LOW,
            scope=ActionScope.WORKSPACE,
        )


def test_action_plan_requires_unique_contiguous_step_order() -> None:
    action_id = new_action_id()
    first = ActionStep(
        action_id=action_id,
        order=0,
        kind=ActionKind.READ,
        capability=ToolCapability.READ_FILE,
        scope=ActionScope.WORKSPACE,
        description="Read first file",
    )
    second = ActionStep(
        action_id=action_id,
        order=2,
        kind=ActionKind.READ,
        capability=ToolCapability.READ_FILE,
        scope=ActionScope.WORKSPACE,
        description="Read second file",
    )

    with pytest.raises(ValidationError):
        ActionPlan(
            action_id=action_id,
            goal="Read files",
            steps=(first, second),
            risk=ActionRisk.LOW,
            scope=ActionScope.WORKSPACE,
        )


def test_valid_action_plan() -> None:
    action_id = new_action_id()
    first = ActionStep(
        action_id=action_id,
        order=0,
        kind=ActionKind.READ,
        capability=ToolCapability.READ_FILE,
        scope=ActionScope.WORKSPACE,
        description="Read file",
    )
    second = ActionStep(
        action_id=action_id,
        order=1,
        kind=ActionKind.SEARCH,
        capability=ToolCapability.SEARCH_FILES,
        scope=ActionScope.WORKSPACE,
        description="Search files",
        depends_on=(first.step_id,),
    )

    plan = ActionPlan(
        plan_id=new_action_plan_id(),
        action_id=action_id,
        goal="Inspect workspace",
        steps=(first, second),
        risk=ActionRisk.LOW,
        scope=ActionScope.WORKSPACE,
    )

    assert plan.status == ActionStatus.PLANNED
    assert len(plan.steps) == 2


def test_high_risk_action_plan_requires_approval() -> None:
    action_id = new_action_id()
    step = ActionStep(
        action_id=action_id,
        order=0,
        kind=ActionKind.DELETE,
        capability=ToolCapability.DELETE_FILE,
        scope=ActionScope.FILE_SYSTEM,
        risk=ActionRisk.HIGH,
        description="Delete file",
        timeout_ms=30_000,
    )

    with pytest.raises(ValidationError):
        ActionPlan(
            action_id=action_id,
            goal="Delete file",
            steps=(step,),
            risk=ActionRisk.HIGH,
            scope=ActionScope.FILE_SYSTEM,
            requires_approval=False,
        )


def test_denied_action_plan_must_be_blocked() -> None:
    action_id = new_action_id()
    step = ActionStep(
        action_id=action_id,
        order=0,
        kind=ActionKind.READ,
        capability=ToolCapability.READ_FILE,
        scope=ActionScope.WORKSPACE,
        description="Read file",
    )

    with pytest.raises(ValidationError):
        ActionPlan(
            action_id=action_id,
            goal="Read file",
            steps=(step,),
            risk=ActionRisk.LOW,
            scope=ActionScope.WORKSPACE,
            permission_decision=PermissionDecision.DENY,
            status=ActionStatus.PLANNED,
        )


def test_successful_action_result() -> None:
    result = ActionResult(
        action_id=new_action_id(),
        status=ActionStatus.SUCCEEDED,
        success=True,
        output="Tests passed",
        duration_ms=100,
    )

    assert result.result_id.startswith("result_")
    assert result.success is True
    assert result.output == "Tests passed"


def test_failed_action_result_requires_error() -> None:
    with pytest.raises(ValidationError):
        ActionResult(
            action_id=new_action_id(),
            status=ActionStatus.FAILED,
            success=False,
        )


def test_failed_action_result_with_error() -> None:
    error = ActionError(
        code="COMMAND_FAILED",
        message="Command exited with non-zero status",
        recoverable=True,
        retryable=False,
    )
    result = ActionResult(
        action_id=new_action_id(),
        status=ActionStatus.FAILED,
        success=False,
        error=error,
    )

    assert result.error is not None
    assert result.error.code == "COMMAND_FAILED"


def test_success_result_cannot_use_failed_status() -> None:
    with pytest.raises(ValidationError):
        ActionResult(
            action_id=new_action_id(),
            status=ActionStatus.FAILED,
            success=True,
            error=ActionError(code="X", message="X"),
        )


def test_failed_result_cannot_use_success_status() -> None:
    with pytest.raises(ValidationError):
        ActionResult(
            action_id=new_action_id(),
            status=ActionStatus.SUCCEEDED,
            success=False,
            error=ActionError(code="X", message="X"),
        )


def test_action_result_rejects_duplicate_artifacts() -> None:
    with pytest.raises(ValidationError):
        ActionResult(
            action_id=new_action_id(),
            status=ActionStatus.SUCCEEDED,
            success=True,
            artifacts=("a.txt", "a.txt"),
        )


def test_policy_decision_record() -> None:
    action_id = new_action_id()
    record = PolicyDecisionRecord(
        action_id=action_id,
        decision=PermissionDecision.REQUIRE_APPROVAL,
        risk=ActionRisk.HIGH,
        reason="High risk action requires approval",
    )

    assert record.action_id == action_id
    assert record.decision == PermissionDecision.REQUIRE_APPROVAL
    assert record.decision_id.startswith("policy_")


def test_errors_import_smoke() -> None:
    error = ToolValidationError("validation failed")

    assert str(error) == "validation failed"


def test_enum_values_are_stable() -> None:
    assert ActionKind.SHELL_COMMAND.value == "shell_command"
    assert ActionRisk.CRITICAL.value == "critical"
    assert ActionStatus.WAITING_FOR_APPROVAL.value == "waiting_for_approval"
    assert ActionScope.WORKSPACE.value == "workspace"
    assert PermissionDecision.REQUIRE_APPROVAL.value == "require_approval"
    assert ToolCapability.RUN_SHELL_COMMAND.value == "run_shell_command"


def test_action_step_id_prefix() -> None:
    assert new_action_step_id().startswith("step_")