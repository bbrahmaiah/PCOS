from __future__ import annotations

from datetime import timedelta

import pytest
from pydantic import ValidationError

from jarvis.tools import (
    ActionAuditEventKind,
    ActionAuditLog,
    ActionKind,
    ActionRisk,
    ActionScope,
    ApprovalDecision,
    ApprovalReason,
    ApprovalRequest,
    ApprovalRequirement,
    ApprovalScope,
    HumanApprovalRuntime,
    HumanApprovalRuntimeConfig,
    PermissionDecision,
    ToolCapability,
    utc_now,
)


def test_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        HumanApprovalRuntimeConfig(name=" ").validate()

    with pytest.raises(ValueError):
        HumanApprovalRuntimeConfig(default_expiration_minutes=0).validate()

    with pytest.raises(ValueError):
        HumanApprovalRuntimeConfig(session_expiration_minutes=0).validate()


def test_approval_request_rejects_none_requirement() -> None:
    with pytest.raises(ValidationError):
        ApprovalRequest(
            requirement=ApprovalRequirement.NONE,
            risk=ActionRisk.LOW,
            reason=ApprovalReason.NO_APPROVAL_REQUIRED,
            message="none",
            expires_at=utc_now() + timedelta(minutes=1),
        )


def test_approval_request_rejects_blocked_requirement() -> None:
    with pytest.raises(ValidationError):
        ApprovalRequest(
            requirement=ApprovalRequirement.BLOCKED,
            risk=ActionRisk.HIGH,
            reason=ApprovalReason.ACTION_BLOCKED,
            message="blocked",
            expires_at=utc_now() + timedelta(minutes=1),
        )


def test_low_risk_allow_requires_no_approval() -> None:
    runtime = HumanApprovalRuntime()

    result = runtime.evaluate(
        action_id="action-1",
        risk=ActionRisk.LOW,
        permission_decision=PermissionDecision.ALLOW,
    )

    assert result.requirement == ApprovalRequirement.NONE
    assert result.decision == ApprovalDecision.NOT_REQUIRED
    assert result.requires_human is False


def test_medium_risk_requires_confirmation() -> None:
    runtime = HumanApprovalRuntime()

    result = runtime.evaluate(
        action_id="action-1",
        risk=ActionRisk.MEDIUM,
        permission_decision=PermissionDecision.ALLOW,
    )

    assert result.requirement == ApprovalRequirement.SOFT_CONFIRMATION
    assert result.reason == ApprovalReason.SOFT_CONFIRMATION_REQUIRED
    assert result.requires_human is True


def test_permission_confirmation_requires_confirmation() -> None:
    runtime = HumanApprovalRuntime()

    result = runtime.evaluate(
        action_id="action-1",
        risk=ActionRisk.LOW,
        permission_decision=PermissionDecision.REQUIRE_CONFIRMATION,
    )

    assert result.requirement == ApprovalRequirement.SOFT_CONFIRMATION


def test_high_risk_requires_explicit_approval() -> None:
    runtime = HumanApprovalRuntime()

    result = runtime.evaluate(
        action_id="action-1",
        risk=ActionRisk.HIGH,
        permission_decision=PermissionDecision.ALLOW,
    )

    assert result.requirement == ApprovalRequirement.EXPLICIT_APPROVAL
    assert result.reason == ApprovalReason.EXPLICIT_APPROVAL_REQUIRED


def test_critical_risk_requires_admin_approval() -> None:
    runtime = HumanApprovalRuntime()

    result = runtime.evaluate(
        action_id="action-1",
        risk=ActionRisk.CRITICAL,
        permission_decision=PermissionDecision.ALLOW,
    )

    assert result.requirement == ApprovalRequirement.ADMIN_APPROVAL
    assert result.reason == ApprovalReason.ADMIN_APPROVAL_REQUIRED


def test_permission_deny_blocks_action() -> None:
    runtime = HumanApprovalRuntime()

    result = runtime.evaluate(
        action_id="action-1",
        risk=ActionRisk.LOW,
        permission_decision=PermissionDecision.DENY,
    )

    assert result.requirement == ApprovalRequirement.BLOCKED
    assert result.decision == ApprovalDecision.BLOCKED
    assert result.requires_human is False


def test_request_approval_creates_pending_request() -> None:
    runtime = HumanApprovalRuntime()

    request = runtime.request_approval(
        action_id="action-1",
        requirement=ApprovalRequirement.EXPLICIT_APPROVAL,
        risk=ActionRisk.HIGH,
        reason=ApprovalReason.EXPLICIT_APPROVAL_REQUIRED,
        message="delete file requires approval",
        action_kind=ActionKind.DELETE,
        capability=ToolCapability.DELETE_FILE,
        action_scope=ActionScope.WORKSPACE,
    )

    assert request.action_id == "action-1"
    assert request.requirement == ApprovalRequirement.EXPLICIT_APPROVAL
    assert request.scope == ApprovalScope.ONE_TIME
    assert request.expired is False


def test_approve_one_time_request_and_consume() -> None:
    runtime = HumanApprovalRuntime()
    request = runtime.request_approval(
        action_id="action-1",
        requirement=ApprovalRequirement.EXPLICIT_APPROVAL,
        risk=ActionRisk.HIGH,
        reason=ApprovalReason.EXPLICIT_APPROVAL_REQUIRED,
        message="delete file requires approval",
        action_kind=ActionKind.DELETE,
        capability=ToolCapability.DELETE_FILE,
    )

    record = runtime.approve(
        approval_id=request.approval_id,
        decided_by="Bala",
        evidence="User said yes",
    )
    check = runtime.check_approval(action_id="action-1")
    second_check = runtime.check_approval(action_id="action-1")

    assert record.approved is True
    assert check.approved is True
    assert check.reason == ApprovalReason.ONE_TIME_APPROVAL_CONSUMED
    assert second_check.approved is False


def test_deny_request_records_denial() -> None:
    runtime = HumanApprovalRuntime()
    request = runtime.request_approval(
        action_id="action-1",
        requirement=ApprovalRequirement.EXPLICIT_APPROVAL,
        risk=ActionRisk.HIGH,
        reason=ApprovalReason.EXPLICIT_APPROVAL_REQUIRED,
        message="delete file requires approval",
    )

    record = runtime.deny(
        approval_id=request.approval_id,
        decided_by="Bala",
        evidence="User said no",
        reason="unsafe delete",
    )

    assert record.decision == ApprovalDecision.DENIED
    assert record.metadata["denial_reason"] == "unsafe delete"


def test_admin_approval_requires_admin_flag() -> None:
    runtime = HumanApprovalRuntime()
    request = runtime.request_approval(
        action_id="action-1",
        requirement=ApprovalRequirement.ADMIN_APPROVAL,
        risk=ActionRisk.CRITICAL,
        reason=ApprovalReason.ADMIN_APPROVAL_REQUIRED,
        message="critical action requires admin",
    )

    with pytest.raises(ValueError):
        runtime.approve(
            approval_id=request.approval_id,
            decided_by="Bala",
            evidence="regular approval",
        )

    record = runtime.approve(
        approval_id=request.approval_id,
        decided_by="Bala",
        evidence="admin approval",
        admin=True,
    )

    assert record.approved is True
    assert record.metadata["admin"] is True


def test_expired_request_cannot_be_approved() -> None:
    runtime = HumanApprovalRuntime()
    request = runtime.request_approval(
        action_id="action-1",
        requirement=ApprovalRequirement.EXPLICIT_APPROVAL,
        risk=ActionRisk.HIGH,
        reason=ApprovalReason.EXPLICIT_APPROVAL_REQUIRED,
        message="approval",
        expires_in_minutes=1,
    )
    expired = request.model_copy(
        update={"expires_at": utc_now() - timedelta(seconds=1)}
    )
    runtime._requests[request.approval_id] = expired  # noqa: SLF001

    with pytest.raises(ValueError):
        runtime.approve(
            approval_id=request.approval_id,
            decided_by="Bala",
            evidence="too late",
        )


def test_session_approval_applies_beyond_single_action() -> None:
    runtime = HumanApprovalRuntime()
    request = runtime.request_approval(
        action_id="action-1",
        requirement=ApprovalRequirement.EXPLICIT_APPROVAL,
        risk=ActionRisk.HIGH,
        reason=ApprovalReason.EXPLICIT_APPROVAL_REQUIRED,
        message="session approval",
        scope=ApprovalScope.SESSION,
    )

    runtime.approve(
        approval_id=request.approval_id,
        decided_by="Bala",
        evidence="approve session",
    )

    check_one = runtime.check_approval(action_id="action-1")
    check_two = runtime.check_approval(action_id="action-2")

    assert check_one.approved is True
    assert check_two.approved is True
    assert check_two.reason == ApprovalReason.SESSION_APPROVAL_ACTIVE


def test_action_kind_scope_matches_only_same_kind() -> None:
    runtime = HumanApprovalRuntime()
    request = runtime.request_approval(
        action_id="action-1",
        requirement=ApprovalRequirement.EXPLICIT_APPROVAL,
        risk=ActionRisk.HIGH,
        reason=ApprovalReason.EXPLICIT_APPROVAL_REQUIRED,
        message="approve delete kind",
        scope=ApprovalScope.ACTION_KIND,
        action_kind=ActionKind.DELETE,
    )

    runtime.approve(
        approval_id=request.approval_id,
        decided_by="Bala",
        evidence="approve delete kind",
    )

    matching = runtime.check_approval(
        action_id="action-2",
        action_kind=ActionKind.DELETE,
    )
    non_matching = runtime.check_approval(
        action_id="action-3",
        action_kind=ActionKind.READ,
    )

    assert matching.approved is True
    assert non_matching.approved is False


def test_capability_scope_matches_only_same_capability() -> None:
    runtime = HumanApprovalRuntime()
    request = runtime.request_approval(
        action_id="action-1",
        requirement=ApprovalRequirement.EXPLICIT_APPROVAL,
        risk=ActionRisk.HIGH,
        reason=ApprovalReason.EXPLICIT_APPROVAL_REQUIRED,
        message="approve delete capability",
        scope=ApprovalScope.TOOL_CAPABILITY,
        capability=ToolCapability.DELETE_FILE,
    )

    runtime.approve(
        approval_id=request.approval_id,
        decided_by="Bala",
        evidence="approve delete capability",
    )

    matching = runtime.check_approval(
        action_id="action-2",
        capability=ToolCapability.DELETE_FILE,
    )
    non_matching = runtime.check_approval(
        action_id="action-3",
        capability=ToolCapability.READ_FILE,
    )

    assert matching.approved is True
    assert non_matching.approved is False


def test_session_approvals_can_be_disabled() -> None:
    runtime = HumanApprovalRuntime(
        config=HumanApprovalRuntimeConfig(allow_session_approvals=False)
    )

    with pytest.raises(ValueError):
        runtime.request_approval(
            action_id="action-1",
            requirement=ApprovalRequirement.EXPLICIT_APPROVAL,
            risk=ActionRisk.HIGH,
            reason=ApprovalReason.EXPLICIT_APPROVAL_REQUIRED,
            message="session approval",
            scope=ApprovalScope.SESSION,
        )


def test_audit_integration_records_request_grant_and_denial() -> None:
    audit_log = ActionAuditLog()
    runtime = HumanApprovalRuntime(audit_log=audit_log)

    request = runtime.request_approval(
        action_id="action-1",
        requirement=ApprovalRequirement.EXPLICIT_APPROVAL,
        risk=ActionRisk.HIGH,
        reason=ApprovalReason.EXPLICIT_APPROVAL_REQUIRED,
        message="approval needed",
    )
    runtime.approve(
        approval_id=request.approval_id,
        decided_by="Bala",
        evidence="approved",
    )

    events = [record.event_kind for record in audit_log.all_records()]

    assert ActionAuditEventKind.APPROVAL_REQUESTED in events
    assert ActionAuditEventKind.APPROVAL_GRANTED in events


def test_snapshot_and_reset() -> None:
    runtime = HumanApprovalRuntime()
    request = runtime.request_approval(
        action_id="action-1",
        requirement=ApprovalRequirement.EXPLICIT_APPROVAL,
        risk=ActionRisk.HIGH,
        reason=ApprovalReason.EXPLICIT_APPROVAL_REQUIRED,
        message="approval",
    )
    runtime.approve(
        approval_id=request.approval_id,
        decided_by="Bala",
        evidence="yes",
    )

    snapshot = runtime.snapshot()

    assert snapshot.request_count == 1
    assert snapshot.approval_count == 1
    assert snapshot.last_decision == ApprovalDecision.APPROVED

    runtime.reset()
    reset_snapshot = runtime.snapshot()

    assert reset_snapshot.request_count == 0
    assert reset_snapshot.last_decision is None


def test_enum_values_are_stable() -> None:
    assert ApprovalRequirement.EXPLICIT_APPROVAL.value == "explicit_approval"
    assert ApprovalDecision.APPROVED.value == "approved"
    assert ApprovalScope.ONE_TIME.value == "one_time"
    assert ApprovalReason.ADMIN_APPROVAL_REQUIRED.value == (
        "admin_approval_required"
    )