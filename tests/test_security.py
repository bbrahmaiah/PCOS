from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.runtime.events import EventBus
from jarvis.runtime.security import (
    AuditLogger,
    AuditRecord,
    IdentityManager,
    PermissionEngine,
    PermissionPolicy,
    PermissionRequest,
    PolicyEngine,
)
from jarvis.runtime.shared.enums import (
    EventType,
    PermissionDecision,
    RiskLevel,
)


def test_identity_manager_authenticates_and_logs_out() -> None:
    manager = IdentityManager()

    identity = manager.authenticate_local_user(
        user_id="bala",
        display_name="Bala",
    )

    assert identity.authenticated is True
    assert identity.user_id == "bala"
    assert manager.current_identity() == identity

    manager.logout()

    assert manager.current_identity() is None


def test_identity_manager_requires_identity() -> None:
    manager = IdentityManager()

    with pytest.raises(PermissionError):
        manager.require_identity()


def test_permission_request_rejects_empty_values() -> None:
    with pytest.raises(ValueError):
        PermissionRequest(action="   ", requested_by="test")

    with pytest.raises(ValueError):
        PermissionRequest(action="browser.open_url", requested_by="   ")


def test_policy_engine_allows_known_safe_action_with_identity() -> None:
    identity = IdentityManager().authenticate_local_user()
    engine = PolicyEngine()

    request = PermissionRequest(
        action="browser.open_url",
        requested_by="test",
    )

    result = engine.evaluate(request, identity=identity)

    assert result.allowed is True
    assert result.decision == PermissionDecision.ALLOW
    assert result.risk == RiskLevel.SAFE


def test_policy_engine_denies_known_action_without_identity() -> None:
    engine = PolicyEngine()

    request = PermissionRequest(
        action="browser.open_url",
        requested_by="test",
    )

    result = engine.evaluate(request, identity=None)

    assert result.allowed is False
    assert result.decision == PermissionDecision.DENY


def test_policy_engine_denies_unknown_action_by_default() -> None:
    engine = PolicyEngine()

    request = PermissionRequest(
        action="unknown.dangerous_action",
        requested_by="test",
    )

    result = engine.evaluate(request)

    assert result.allowed is False
    assert result.decision == PermissionDecision.DENY
    assert result.risk == RiskLevel.HIGH


def test_policy_engine_can_require_confirmation_for_unknown_actions() -> None:
    engine = PolicyEngine(deny_unknown_actions=False)

    request = PermissionRequest(
        action="unknown.action",
        requested_by="test",
    )

    result = engine.evaluate(request)

    assert result.requires_confirmation is True
    assert result.decision == PermissionDecision.REQUIRE_CONFIRMATION


def test_policy_engine_custom_policy_matches_wildcard() -> None:
    engine = PolicyEngine()
    identity = IdentityManager().authenticate_local_user()

    engine.add_policy(
        PermissionPolicy(
            action_pattern="custom.*",
            decision=PermissionDecision.ALLOW,
            risk=RiskLevel.LOW,
            reason="Custom test actions are allowed.",
        )
    )

    result = engine.evaluate(
        PermissionRequest(
            action="custom.demo",
            requested_by="test",
        ),
        identity=identity,
    )

    assert result.allowed is True


def test_audit_logger_records_result(tmp_path: Path) -> None:
    identity = IdentityManager().authenticate_local_user()
    policy = PolicyEngine()
    request = PermissionRequest(
        action="browser.search",
        requested_by="test",
    )
    result = policy.evaluate(request, identity=identity)

    logger = AuditLogger(
        log_file=tmp_path / "audit.jsonl",
        memory_limit=10,
    )

    record = logger.record_result(result)

    assert record.action == "browser.search"
    assert len(logger.records()) == 1
    assert logger.log_file.exists()
    assert "browser.search" in logger.log_file.read_text(encoding="utf-8")


def test_audit_logger_rejects_invalid_memory_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        AuditLogger(
            log_file=tmp_path / "audit.jsonl",
            memory_limit=0,
        )


def test_audit_logger_rejects_invalid_record(tmp_path: Path) -> None:
    logger = AuditLogger(log_file=tmp_path / "audit.jsonl")

    with pytest.raises(TypeError):
        logger.record("not a record")  # type: ignore[arg-type]


def test_audit_record_to_dict() -> None:
    record = AuditRecord(
        action="browser.search",
        decision=PermissionDecision.ALLOW,
        risk=RiskLevel.SAFE,
        reason="Allowed.",
        request_id="request-1",
    )

    data = record.to_dict()

    assert data["action"] == "browser.search"
    assert data["decision"] == "allow"
    assert data["risk"] == "safe"


def test_permission_engine_grants_allowed_action(tmp_path: Path) -> None:
    bus = EventBus(name="test_bus")
    audit_logger = AuditLogger(log_file=tmp_path / "audit.jsonl")
    engine = PermissionEngine(
        event_bus=bus,
        audit_logger=audit_logger,
    )

    engine.authenticate_local_user()

    result = engine.request_permission(
        action="browser.search",
        requested_by="test",
        payload={"query": "JARVIS"},
    )

    assert result.allowed is True

    event_types = [event.event_type for event in bus.history()]

    assert EventType.PERMISSION_REQUESTED in event_types
    assert EventType.POLICY_EVALUATED in event_types
    assert EventType.AUDIT_RECORDED in event_types
    assert EventType.PERMISSION_GRANTED in event_types


def test_permission_engine_denies_without_identity(tmp_path: Path) -> None:
    bus = EventBus(name="test_bus")
    audit_logger = AuditLogger(log_file=tmp_path / "audit.jsonl")
    engine = PermissionEngine(
        event_bus=bus,
        audit_logger=audit_logger,
    )

    result = engine.request_permission(
        action="system.open_app",
        requested_by="test",
    )

    assert result.allowed is False
    assert result.decision == PermissionDecision.DENY

    event_types = [event.event_type for event in bus.history()]

    assert EventType.PERMISSION_DENIED in event_types


def test_permission_engine_requires_confirmation_for_dangerous_action(
    tmp_path: Path,
) -> None:
    bus = EventBus(name="test_bus")
    audit_logger = AuditLogger(log_file=tmp_path / "audit.jsonl")
    engine = PermissionEngine(
        event_bus=bus,
        audit_logger=audit_logger,
    )

    engine.authenticate_local_user()

    result = engine.request_permission(
        action="filesystem.delete",
        requested_by="test",
    )

    assert result.allowed is False
    assert result.requires_confirmation is True
    assert result.decision == PermissionDecision.REQUIRE_DOUBLE_CONFIRMATION