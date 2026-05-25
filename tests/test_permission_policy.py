from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.tools import (
    ActionKind,
    ActionPlan,
    ActionRequest,
    ActionRisk,
    ActionScope,
    ActionStatus,
    ActionStep,
    PermissionDecision,
    PermissionPolicy,
    PermissionPolicyConfig,
    PermissionPolicyEvaluation,
    PermissionReason,
    PermissionScope,
    RiskClassifier,
    ToolAvailability,
    ToolCapability,
    ToolDescriptor,
    ToolHealth,
    new_action_id,
)


def request(
    *,
    kind: ActionKind = ActionKind.READ,
    capability: ToolCapability = ToolCapability.READ_FILE,
    scope: ActionScope = ActionScope.WORKSPACE,
    risk: ActionRisk = ActionRisk.LOW,
    approval: bool = False,
) -> ActionRequest:
    return ActionRequest(
        kind=kind,
        requested_capability=capability,
        intent="test action",
        scope=scope,
        risk=risk,
        requires_approval=approval,
    )


def step(
    *,
    action_id: str,
    kind: ActionKind = ActionKind.READ,
    capability: ToolCapability = ToolCapability.READ_FILE,
    scope: ActionScope = ActionScope.WORKSPACE,
    risk: ActionRisk = ActionRisk.LOW,
) -> ActionStep:
    return ActionStep(
        action_id=action_id,
        order=0,
        kind=kind,
        capability=capability,
        scope=scope,
        risk=risk,
        description="test step",
        timeout_ms=30_000 if risk in {ActionRisk.HIGH, ActionRisk.CRITICAL} else None,
    )


def tool(
    *,
    capability: ToolCapability = ToolCapability.READ_FILE,
    kind: ActionKind = ActionKind.READ,
    scope: ActionScope = ActionScope.WORKSPACE,
    risk: ActionRisk = ActionRisk.LOW,
    enabled: bool = True,
    availability: ToolAvailability = ToolAvailability.AVAILABLE,
    health: ToolHealth = ToolHealth.HEALTHY,
) -> ToolDescriptor:
    return ToolDescriptor(
        tool_id="tool_policy_test",
        name="policy test tool",
        description="policy test descriptor",
        capabilities=(capability,),
        supported_action_kinds=(kind,),
        scopes=(scope,),
        max_risk=risk,
        required_permission=PermissionDecision.ALLOW
        if risk in {ActionRisk.NONE, ActionRisk.LOW, ActionRisk.MEDIUM}
        else PermissionDecision.REQUIRE_APPROVAL,
        enabled=enabled,
        availability=availability,
        health=health,
    )


def test_policy_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        PermissionPolicyConfig(name=" ").validate()


def test_policy_evaluation_requires_text() -> None:
    with pytest.raises(ValidationError):
        PermissionPolicyEvaluation(
            action_id=" ",
            decision=PermissionDecision.ALLOW,
            scope=PermissionScope.WORKSPACE_BOUNDED,
            risk=ActionRisk.LOW,
            reason=PermissionReason.SAFE_READ_ALLOWED,
            explanation="valid",
            allowed=True,
            blocked=False,
        )


def test_workspace_read_is_allowed() -> None:
    policy = PermissionPolicy()
    evaluation = policy.evaluate_request(request())

    assert evaluation.decision == PermissionDecision.ALLOW
    assert evaluation.allowed is True
    assert evaluation.reason == PermissionReason.WORKSPACE_BOUNDED_ALLOWED


def test_project_read_is_allowed() -> None:
    policy = PermissionPolicy()
    evaluation = policy.evaluate_request(
        request(scope=ActionScope.PROJECT)
    )

    assert evaluation.decision == PermissionDecision.ALLOW
    assert evaluation.scope == PermissionScope.PROJECT_BOUNDED


def test_memory_read_requires_gateway_reason() -> None:
    policy = PermissionPolicy()
    evaluation = policy.evaluate_request(
        request(scope=ActionScope.MEMORY)
    )

    assert evaluation.decision == PermissionDecision.ALLOW
    assert evaluation.reason == PermissionReason.MEMORY_GATEWAY_REQUIRED


def test_write_requires_confirmation() -> None:
    policy = PermissionPolicy()
    evaluation = policy.evaluate_request(
        request(
            kind=ActionKind.WRITE,
            capability=ToolCapability.WRITE_FILE,
            scope=ActionScope.WORKSPACE,
            risk=ActionRisk.MEDIUM,
        )
    )

    assert evaluation.decision == PermissionDecision.REQUIRE_CONFIRMATION
    assert evaluation.requires_confirmation is True


def test_patch_requires_confirmation() -> None:
    policy = PermissionPolicy()
    evaluation = policy.evaluate_request(
        request(
            kind=ActionKind.PATCH,
            capability=ToolCapability.PATCH_FILE,
            scope=ActionScope.WORKSPACE,
            risk=ActionRisk.MEDIUM,
        )
    )

    assert evaluation.decision == PermissionDecision.REQUIRE_CONFIRMATION
    assert evaluation.reason == PermissionReason.PATCH_REQUIRES_CONFIRMATION


def test_delete_requires_approval() -> None:
    policy = PermissionPolicy()
    evaluation = policy.evaluate_request(
        request(
            kind=ActionKind.DELETE,
            capability=ToolCapability.DELETE_FILE,
            scope=ActionScope.FILE_SYSTEM,
            risk=ActionRisk.HIGH,
            approval=True,
        )
    )

    assert evaluation.decision == PermissionDecision.REQUIRE_APPROVAL
    assert evaluation.requires_approval is True


def test_shell_requires_policy_handling() -> None:
    policy = PermissionPolicy()
    evaluation = policy.evaluate_request(
        request(
            kind=ActionKind.SHELL_COMMAND,
            capability=ToolCapability.RUN_SHELL_COMMAND,
            scope=ActionScope.SHELL,
            risk=ActionRisk.HIGH,
            approval=True,
        )
    )

    assert evaluation.decision == PermissionDecision.REQUIRE_APPROVAL
    assert evaluation.reason == PermissionReason.HIGH_RISK_REQUIRES_APPROVAL


def test_low_risk_shell_uses_configured_decision() -> None:
    policy = PermissionPolicy(
        config=PermissionPolicyConfig(
            shell_low_risk_decision=PermissionDecision.READ_ONLY_ONLY
        )
    )
    evaluation = policy.evaluate_request(
        request(
            kind=ActionKind.SHELL_COMMAND,
            capability=ToolCapability.RUN_SHELL_COMMAND,
            scope=ActionScope.SHELL,
            risk=ActionRisk.LOW,
        )
    )

    assert evaluation.decision == PermissionDecision.REQUIRE_APPROVAL


def test_browser_requires_confirmation() -> None:
    policy = PermissionPolicy()
    evaluation = policy.evaluate_request(
        request(
            kind=ActionKind.BROWSER_SEARCH,
            capability=ToolCapability.SEARCH_WEB,
            scope=ActionScope.BROWSER,
            risk=ActionRisk.LOW,
        )
    )

    assert evaluation.decision == PermissionDecision.REQUIRE_CONFIRMATION
    assert evaluation.reason == PermissionReason.BROWSER_REQUIRES_CONFIRMATION


def test_system_scope_high_risk_requires_approval() -> None:
    policy = PermissionPolicy()
    evaluation = policy.evaluate_request(
        request(
            kind=ActionKind.SYSTEM_QUERY,
            capability=ToolCapability.QUERY_DESKTOP_STATE,
            scope=ActionScope.SYSTEM,
            risk=ActionRisk.HIGH,
            approval=True,
        )
    )

    assert evaluation.decision == PermissionDecision.REQUIRE_APPROVAL


def test_critical_risk_denied_by_default() -> None:
    policy = PermissionPolicy()
    evaluation = policy.evaluate_request(
        request(
            kind=ActionKind.DELETE,
            capability=ToolCapability.DELETE_FILE,
            scope=ActionScope.SYSTEM,
            risk=ActionRisk.CRITICAL,
            approval=True,
        )
    )

    assert evaluation.decision == PermissionDecision.DENY
    assert evaluation.blocked is True
    assert evaluation.reason == PermissionReason.CRITICAL_RISK_DENIED


def test_tool_disabled_denies() -> None:
    policy = PermissionPolicy()
    item = tool(
        enabled=False,
        availability=ToolAvailability.DISABLED,
    )
    evaluation = policy.evaluate_request(request(), tool=item)

    assert evaluation.decision == PermissionDecision.DENY
    assert evaluation.reason == PermissionReason.TOOL_DISABLED


def test_tool_unavailable_denies() -> None:
    policy = PermissionPolicy()
    item = tool(availability=ToolAvailability.UNAVAILABLE)
    evaluation = policy.evaluate_request(request(), tool=item)

    assert evaluation.decision == PermissionDecision.DENY
    assert evaluation.reason == PermissionReason.TOOL_UNAVAILABLE


def test_tool_unhealthy_denies() -> None:
    policy = PermissionPolicy()
    item = tool(
        health=ToolHealth.UNHEALTHY,
        availability=ToolAvailability.UNAVAILABLE,
    )
    evaluation = policy.evaluate_request(request(), tool=item)

    assert evaluation.decision == PermissionDecision.DENY
    assert evaluation.reason == PermissionReason.TOOL_UNAVAILABLE


def test_tool_capability_mismatch_denies() -> None:
    policy = PermissionPolicy()
    item = tool(capability=ToolCapability.SEARCH_FILES)
    evaluation = policy.evaluate_request(request(), tool=item)

    assert evaluation.decision == PermissionDecision.DENY
    assert evaluation.reason == PermissionReason.TOOL_CAPABILITY_MISMATCH


def test_tool_action_kind_mismatch_denies() -> None:
    policy = PermissionPolicy()
    item = tool(kind=ActionKind.SEARCH)
    evaluation = policy.evaluate_request(request(), tool=item)

    assert evaluation.decision == PermissionDecision.DENY
    assert evaluation.reason == PermissionReason.TOOL_ACTION_KIND_MISMATCH


def test_tool_scope_mismatch_denies() -> None:
    policy = PermissionPolicy()
    item = tool(scope=ActionScope.PROJECT)
    evaluation = policy.evaluate_request(request(), tool=item)

    assert evaluation.decision == PermissionDecision.DENY
    assert evaluation.reason == PermissionReason.TOOL_SCOPE_MISMATCH


def test_tool_risk_exceeded_denies() -> None:
    policy = PermissionPolicy()
    item = tool(risk=ActionRisk.LOW)
    evaluation = policy.evaluate_request(
        request(
            kind=ActionKind.WRITE,
            capability=ToolCapability.WRITE_FILE,
            risk=ActionRisk.MEDIUM,
        ),
        tool=item,
    )

    assert evaluation.decision == PermissionDecision.DENY
    assert evaluation.reason == PermissionReason.TOOL_CAPABILITY_MISMATCH


def test_evaluate_step() -> None:
    policy = PermissionPolicy()
    action_id = new_action_id()
    evaluation = policy.evaluate_step(
        step(
            action_id=action_id,
            kind=ActionKind.WRITE,
            capability=ToolCapability.WRITE_FILE,
            risk=ActionRisk.MEDIUM,
        )
    )

    assert evaluation.action_id == action_id
    assert evaluation.decision == PermissionDecision.REQUIRE_CONFIRMATION


def test_evaluate_plan() -> None:
    policy = PermissionPolicy()
    action_id = new_action_id()
    item = step(action_id=action_id)
    plan = ActionPlan(
        action_id=action_id,
        goal="read file",
        steps=(item,),
        risk=ActionRisk.LOW,
        scope=ActionScope.WORKSPACE,
    )

    evaluation = policy.evaluate_plan(plan)

    assert evaluation.action_id == action_id
    assert evaluation.decision == PermissionDecision.ALLOW


def test_denied_plan_returns_denied() -> None:
    policy = PermissionPolicy()
    action_id = new_action_id()
    item = step(action_id=action_id)
    plan = ActionPlan(
        action_id=action_id,
        goal="read file",
        steps=(item,),
        risk=ActionRisk.LOW,
        scope=ActionScope.WORKSPACE,
        permission_decision=PermissionDecision.DENY,
        status=ActionStatus.BLOCKED,
    )

    evaluation = policy.evaluate_plan(plan)

    assert evaluation.decision == PermissionDecision.DENY


def test_risk_classifier_classifies_delete_as_high() -> None:
    classifier = RiskClassifier()
    result = classifier.classify_request(
        request(
            kind=ActionKind.DELETE,
            capability=ToolCapability.DELETE_FILE,
            scope=ActionScope.FILE_SYSTEM,
            risk=ActionRisk.LOW,
        )
    )

    assert result.risk == ActionRisk.HIGH
    assert result.score >= 0.8


def test_risk_classifier_classifies_write_as_medium() -> None:
    classifier = RiskClassifier()
    result = classifier.classify_request(
        request(
            kind=ActionKind.WRITE,
            capability=ToolCapability.WRITE_FILE,
            scope=ActionScope.WORKSPACE,
            risk=ActionRisk.LOW,
        )
    )

    assert result.risk == ActionRisk.MEDIUM


def test_snapshot_and_reset() -> None:
    policy = PermissionPolicy()

    policy.evaluate_request(request())
    snapshot = policy.snapshot()

    assert snapshot.evaluation_count == 1
    assert snapshot.allowed_count == 1
    assert snapshot.last_decision == PermissionDecision.ALLOW

    policy.reset()
    reset_snapshot = policy.snapshot()

    assert reset_snapshot.evaluation_count == 0
    assert reset_snapshot.last_decision is None


def test_permission_enum_values_are_stable() -> None:
    assert PermissionScope.WORKSPACE_BOUNDED.value == "workspace_bounded"
    assert PermissionReason.CRITICAL_RISK_DENIED.value == "critical_risk_denied"