from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.tools import (
    ActionKind,
    ActionPlan,
    ActionRisk,
    ActionScope,
    ActionStatus,
    ActionStep,
    ActionValidationDecision,
    ActionValidationFinding,
    ActionValidationReason,
    ActionValidationSeverity,
    ActionValidator,
    ActionValidatorConfig,
    PermissionDecision,
    PermissionPolicy,
    ToolAvailability,
    ToolCapability,
    ToolDescriptor,
    ToolHealth,
    ToolRegistry,
    new_action_id,
    new_action_plan_id,
)


def registry_with_tool(
    *,
    capability: ToolCapability = ToolCapability.READ_FILE,
    kind: ActionKind = ActionKind.READ,
    scope: ActionScope = ActionScope.WORKSPACE,
    risk: ActionRisk = ActionRisk.LOW,
    enabled: bool = True,
    availability: ToolAvailability = ToolAvailability.AVAILABLE,
    health: ToolHealth = ToolHealth.HEALTHY,
) -> ToolRegistry:
    registry = ToolRegistry()
    permission = (
        PermissionDecision.ALLOW
        if risk in {ActionRisk.NONE, ActionRisk.LOW, ActionRisk.MEDIUM}
        else PermissionDecision.REQUIRE_APPROVAL
    )
    registry.register(
        ToolDescriptor(
            tool_id="tool_validation_test",
            name="validation test tool",
            description="validation test descriptor",
            capabilities=(capability,),
            supported_action_kinds=(kind,),
            scopes=(scope,),
            max_risk=risk,
            required_permission=permission,
            enabled=enabled,
            availability=availability,
            health=health,
        )
    )

    return registry


def make_step(
    *,
    action_id: str,
    order: int = 0,
    kind: ActionKind = ActionKind.READ,
    capability: ToolCapability = ToolCapability.READ_FILE,
    scope: ActionScope = ActionScope.WORKSPACE,
    risk: ActionRisk = ActionRisk.LOW,
    arguments: dict[str, object] | None = None,
    timeout_ms: int | None = None,
    rollback_supported: bool = False,
    interruptible: bool = True,
) -> ActionStep:
    return ActionStep(
        action_id=action_id,
        order=order,
        kind=kind,
        capability=capability,
        scope=scope,
        risk=risk,
        description="validation step",
        arguments={"path": "README.md"} if arguments is None else arguments,
        timeout_ms=timeout_ms,
        rollback_supported=rollback_supported,
        interruptible=interruptible,
    )


def make_plan(
    *,
    step: ActionStep,
    risk: ActionRisk = ActionRisk.LOW,
    scope: ActionScope = ActionScope.WORKSPACE,
    requires_approval: bool = False,
    permission_decision: PermissionDecision | None = None,
    status: ActionStatus = ActionStatus.PLANNED,
) -> ActionPlan:
    return ActionPlan(
        action_id=step.action_id,
        goal="validate action plan",
        steps=(step,),
        risk=risk,
        scope=scope,
        requires_approval=requires_approval,
        permission_decision=permission_decision,
        status=status,
    )


def test_validator_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        ActionValidatorConfig(name=" ").validate()

    with pytest.raises(ValueError):
        ActionValidatorConfig(max_argument_string_length=0).validate()


def test_validation_finding_requires_message() -> None:
    with pytest.raises(ValidationError):
        ActionValidationFinding(
            action_id="action",
            severity=ActionValidationSeverity.ERROR,
            reason=ActionValidationReason.INVALID_ARGUMENTS,
            message=" ",
        )


def test_valid_read_plan_allowed() -> None:
    action_id = new_action_id()
    step = make_step(action_id=action_id)
    plan = make_plan(step=step)
    validator = ActionValidator(registry=registry_with_tool())

    result = validator.validate_plan(plan)

    assert result.valid is True
    assert result.blocked is False
    assert result.decision == ActionValidationDecision.ALLOW


def test_unknown_tool_blocks_plan() -> None:
    action_id = new_action_id()
    step = make_step(action_id=action_id)
    plan = make_plan(step=step)
    validator = ActionValidator(registry=ToolRegistry())

    result = validator.validate_plan(plan)

    assert result.blocked is True
    assert any(
        finding.reason == ActionValidationReason.UNKNOWN_TOOL
        for finding in result.findings
    )


def test_disabled_tool_blocks_plan() -> None:
    action_id = new_action_id()
    step = make_step(action_id=action_id)
    plan = make_plan(step=step)
    validator = ActionValidator(
        registry=registry_with_tool(
            enabled=False,
            availability=ToolAvailability.DISABLED,
        )
    )

    result = validator.validate_plan(plan)

    assert result.blocked is True
    assert any(
        finding.reason == ActionValidationReason.TOOL_DISABLED
        for finding in result.findings
    )


def test_unavailable_tool_blocks_plan() -> None:
    action_id = new_action_id()
    step = make_step(action_id=action_id)
    plan = make_plan(step=step)
    validator = ActionValidator(
        registry=registry_with_tool(availability=ToolAvailability.UNAVAILABLE)
    )

    result = validator.validate_plan(plan)

    assert result.blocked is True
    assert any(
        finding.reason == ActionValidationReason.TOOL_UNAVAILABLE
        for finding in result.findings
    )


def test_tool_capability_mismatch_blocks_plan() -> None:
    action_id = new_action_id()
    step = make_step(
        action_id=action_id,
        capability=ToolCapability.WRITE_FILE,
        kind=ActionKind.WRITE,
        risk=ActionRisk.MEDIUM,
    )
    plan = make_plan(step=step, risk=ActionRisk.MEDIUM)
    validator = ActionValidator(
        registry=registry_with_tool(capability=ToolCapability.READ_FILE)
    )

    result = validator.validate_plan(plan)

    assert result.blocked is True
    assert any(
        finding.reason == ActionValidationReason.UNKNOWN_TOOL
        or finding.reason == ActionValidationReason.TOOL_CAPABILITY_MISMATCH
        for finding in result.findings
    )


def test_path_traversal_blocks_plan() -> None:
    action_id = new_action_id()
    step = make_step(
        action_id=action_id,
        arguments={"path": "../secret.txt"},
    )
    plan = make_plan(step=step)
    validator = ActionValidator(registry=registry_with_tool())

    result = validator.validate_plan(plan)

    assert result.blocked is True
    assert any(
        finding.reason == ActionValidationReason.PATH_TRAVERSAL_BLOCKED
        for finding in result.findings
    )


def test_absolute_path_blocks_plan() -> None:
    action_id = new_action_id()
    step = make_step(
        action_id=action_id,
        arguments={"path": "C:\\Users\\Admin\\secret.txt"},
    )
    plan = make_plan(step=step)
    validator = ActionValidator(registry=registry_with_tool())

    result = validator.validate_plan(plan)

    assert result.blocked is True
    assert any(
        finding.reason == ActionValidationReason.ABSOLUTE_PATH_BLOCKED
        for finding in result.findings
    )


def test_missing_path_blocks_file_action() -> None:
    action_id = new_action_id()
    step = make_step(action_id=action_id, arguments={})
    plan = make_plan(step=step)
    validator = ActionValidator(registry=registry_with_tool())

    result = validator.validate_plan(plan)

    assert result.blocked is True
    assert any(
        finding.reason == ActionValidationReason.MISSING_PATH_ARGUMENT
        for finding in result.findings
    )


def test_dangerous_shell_command_blocks_plan() -> None:
    action_id = new_action_id()
    step = make_step(
        action_id=action_id,
        kind=ActionKind.SHELL_COMMAND,
        capability=ToolCapability.RUN_SHELL_COMMAND,
        scope=ActionScope.SHELL,
        risk=ActionRisk.HIGH,
        arguments={"command": "rm -rf /"},
        timeout_ms=30_000,
    )
    plan = make_plan(
        step=step,
        risk=ActionRisk.HIGH,
        scope=ActionScope.SHELL,
        requires_approval=True,
    )
    validator = ActionValidator(
        registry=registry_with_tool(
            capability=ToolCapability.RUN_SHELL_COMMAND,
            kind=ActionKind.SHELL_COMMAND,
            scope=ActionScope.SHELL,
            risk=ActionRisk.HIGH,
        )
    )

    result = validator.validate_plan(plan)

    assert result.blocked is True
    assert any(
        finding.reason == ActionValidationReason.UNSAFE_COMMAND
        for finding in result.findings
    )


def test_missing_command_argument_blocks_shell_action() -> None:
    action_id = new_action_id()
    step = make_step(
        action_id=action_id,
        kind=ActionKind.SHELL_COMMAND,
        capability=ToolCapability.RUN_SHELL_COMMAND,
        scope=ActionScope.SHELL,
        risk=ActionRisk.HIGH,
        arguments={},
        timeout_ms=30_000,
    )
    plan = make_plan(
        step=step,
        risk=ActionRisk.HIGH,
        scope=ActionScope.SHELL,
        requires_approval=True,
    )
    validator = ActionValidator(
        registry=registry_with_tool(
            capability=ToolCapability.RUN_SHELL_COMMAND,
            kind=ActionKind.SHELL_COMMAND,
            scope=ActionScope.SHELL,
            risk=ActionRisk.HIGH,
        )
    )

    result = validator.validate_plan(plan)

    assert result.blocked is True
    assert any(
        finding.reason == ActionValidationReason.MISSING_COMMAND_ARGUMENT
        for finding in result.findings
    )


def test_write_plan_requires_confirmation() -> None:
    action_id = new_action_id()
    step = make_step(
        action_id=action_id,
        kind=ActionKind.WRITE,
        capability=ToolCapability.WRITE_FILE,
        risk=ActionRisk.MEDIUM,
        arguments={"path": "notes.txt"},
    )
    plan = make_plan(step=step, risk=ActionRisk.MEDIUM)
    validator = ActionValidator(
        registry=registry_with_tool(
            capability=ToolCapability.WRITE_FILE,
            kind=ActionKind.WRITE,
            risk=ActionRisk.MEDIUM,
        )
    )

    result = validator.validate_plan(plan)

    assert result.valid is True
    assert result.decision == ActionValidationDecision.REQUIRE_CONFIRMATION
    assert result.requires_confirmation is True


def test_delete_plan_requires_approval() -> None:
    action_id = new_action_id()
    step = make_step(
        action_id=action_id,
        kind=ActionKind.DELETE,
        capability=ToolCapability.DELETE_FILE,
        scope=ActionScope.FILE_SYSTEM,
        risk=ActionRisk.HIGH,
        arguments={"path": "generated.tmp"},
        timeout_ms=30_000,
    )
    plan = make_plan(
        step=step,
        risk=ActionRisk.HIGH,
        scope=ActionScope.FILE_SYSTEM,
        requires_approval=True,
    )
    validator = ActionValidator(
        registry=registry_with_tool(
            capability=ToolCapability.DELETE_FILE,
            kind=ActionKind.DELETE,
            scope=ActionScope.FILE_SYSTEM,
            risk=ActionRisk.HIGH,
        )
    )

    result = validator.validate_plan(plan)

    assert result.valid is True
    assert result.decision == ActionValidationDecision.REQUIRE_APPROVAL


def test_missing_approval_blocks_when_policy_requires_it() -> None:
    action_id = new_action_id()
    step = make_step(
        action_id=action_id,
        kind=ActionKind.DELETE,
        capability=ToolCapability.DELETE_FILE,
        scope=ActionScope.FILE_SYSTEM,
        risk=ActionRisk.HIGH,
        arguments={"path": "generated.tmp"},
        timeout_ms=30_000,
    )
    plan = ActionPlan.model_construct(
        plan_id=new_action_plan_id(),
        action_id=action_id,
        goal="validate corrupted high-risk action plan",
        steps=(step,),
        risk=ActionRisk.HIGH,
        scope=ActionScope.FILE_SYSTEM,
        status=ActionStatus.PLANNED,
        permission_decision=None,
        requires_approval=False,
        metadata={},
    )
    validator = ActionValidator(
        registry=registry_with_tool(
            capability=ToolCapability.DELETE_FILE,
            kind=ActionKind.DELETE,
            scope=ActionScope.FILE_SYSTEM,
            risk=ActionRisk.HIGH,
        )
    )

    result = validator.validate_plan(plan)

    assert result.blocked is True
    assert any(
        finding.reason == ActionValidationReason.APPROVAL_REQUIRED
        for finding in result.findings
    )


def test_policy_denied_blocks_plan() -> None:
    action_id = new_action_id()
    step = make_step(action_id=action_id)
    plan = make_plan(
        step=step,
        permission_decision=PermissionDecision.DENY,
        status=ActionStatus.BLOCKED,
    )
    validator = ActionValidator(registry=registry_with_tool())

    result = validator.validate_plan(plan)

    assert result.blocked is True
    assert any(
        finding.reason == ActionValidationReason.POLICY_DENIED
        for finding in result.findings
    )


def test_policy_bypass_attempt_blocks_plan() -> None:
    action_id = new_action_id()
    step = make_step(
        action_id=action_id,
        kind=ActionKind.WRITE,
        capability=ToolCapability.WRITE_FILE,
        risk=ActionRisk.MEDIUM,
        arguments={"path": "notes.txt"},
    )
    plan = make_plan(
        step=step,
        risk=ActionRisk.MEDIUM,
        permission_decision=PermissionDecision.ALLOW,
    )
    validator = ActionValidator(
        registry=registry_with_tool(
            capability=ToolCapability.WRITE_FILE,
            kind=ActionKind.WRITE,
            risk=ActionRisk.MEDIUM,
        )
    )

    result = validator.validate_plan(plan)

    assert result.blocked is True
    assert any(
        finding.reason == ActionValidationReason.POLICY_BYPASS_ATTEMPT
        for finding in result.findings
    )


def test_argument_string_length_blocks_plan() -> None:
    action_id = new_action_id()
    step = make_step(
        action_id=action_id,
        arguments={"path": "a" * 10},
    )
    plan = make_plan(step=step)
    validator = ActionValidator(
        config=ActionValidatorConfig(max_argument_string_length=5),
        registry=registry_with_tool(),
    )

    result = validator.validate_plan(plan)

    assert result.blocked is True
    assert any(
        finding.reason == ActionValidationReason.INVALID_ARGUMENTS
        for finding in result.findings
    )


def test_snapshot_and_reset() -> None:
    action_id = new_action_id()
    step = make_step(action_id=action_id)
    plan = make_plan(step=step)
    validator = ActionValidator(registry=registry_with_tool())

    validator.validate_plan(plan)
    snapshot = validator.snapshot()

    assert snapshot.validation_count == 1
    assert snapshot.allowed_count == 1
    assert snapshot.last_decision == ActionValidationDecision.ALLOW

    validator.reset()
    reset_snapshot = validator.snapshot()

    assert reset_snapshot.validation_count == 0
    assert reset_snapshot.last_decision is None


def test_validation_enum_values_are_stable() -> None:
    assert ActionValidationDecision.ALLOW.value == "allow"
    assert ActionValidationSeverity.CRITICAL.value == "critical"
    assert ActionValidationReason.UNSAFE_COMMAND.value == "unsafe_command"


def test_validator_imports_policy_type() -> None:
    validator = ActionValidator(policy=PermissionPolicy())

    assert validator.name == "action_validator"