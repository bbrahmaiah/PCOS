from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from jarvis.tools import (
    ActionAuditActor,
    ActionAuditEventKind,
    ActionAuditLog,
    ActionAuditLogConfig,
    ActionAuditOutcome,
    ActionAuditRecord,
    ActionAuditSensitivity,
    ActionRisk,
    ActionStatus,
)


def test_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        ActionAuditLogConfig(name=" ").validate()

    with pytest.raises(ValueError):
        ActionAuditLogConfig(log_path=" ").validate()

    with pytest.raises(ValueError):
        ActionAuditLogConfig(max_records_in_memory=0).validate()


def test_record_requires_message() -> None:
    with pytest.raises(ValidationError):
        ActionAuditRecord(
            sequence=0,
            action_id="action-1",
            event_kind=ActionAuditEventKind.INTENT_RECEIVED,
            actor=ActionAuditActor.USER,
            outcome=ActionAuditOutcome.INFO,
            message=" ",
        )


def test_record_intent() -> None:
    log = ActionAuditLog()

    record = log.record_intent(
        action_id="action-1",
        user_intent="run tests and summarize failures",
    )

    assert record.sequence == 0
    assert record.action_id == "action-1"
    assert record.event_kind == ActionAuditEventKind.INTENT_RECEIVED
    assert record.actor == ActionAuditActor.USER
    assert record.outcome == ActionAuditOutcome.INFO
    assert record.record_hash is not None
    assert record.data["user_intent"] == "run tests and summarize failures"


def test_hash_chain_links_records() -> None:
    log = ActionAuditLog()

    first = log.record_intent(
        action_id="action-1",
        user_intent="run tests",
    )
    second = log.record_plan_proposed(
        action_id="action-1",
        summary="plan proposed",
        plan_steps=5,
        risk=ActionRisk.LOW,
        requires_approval=False,
    )

    assert first.record_hash is not None
    assert second.previous_hash == first.record_hash
    assert second.record_hash is not None
    assert second.record_hash != first.record_hash


def test_records_for_action_filters_records() -> None:
    log = ActionAuditLog()

    log.record_intent(action_id="action-1", user_intent="run tests")
    log.record_intent(action_id="action-2", user_intent="open file")

    records = log.records_for_action("action-1")

    assert len(records) == 1
    assert records[0].action_id == "action-1"


def test_record_plan_proposed_without_approval() -> None:
    log = ActionAuditLog()

    record = log.record_plan_proposed(
        action_id="action-1",
        summary="safe test plan",
        plan_steps=5,
        risk=ActionRisk.LOW,
        requires_approval=False,
    )

    assert record.event_kind == ActionAuditEventKind.PLAN_PROPOSED
    assert record.actor == ActionAuditActor.PLANNER
    assert record.outcome == ActionAuditOutcome.INFO
    assert record.data["plan_steps"] == 5


def test_record_plan_proposed_with_approval_required() -> None:
    log = ActionAuditLog()

    record = log.record_plan_proposed(
        action_id="action-1",
        summary="risky plan",
        plan_steps=2,
        risk=ActionRisk.HIGH,
        requires_approval=True,
    )

    assert record.outcome == ActionAuditOutcome.APPROVAL_REQUIRED
    assert record.data["requires_approval"] is True


def test_record_execution_completed_success() -> None:
    log = ActionAuditLog()

    record = log.record_execution_completed(
        action_id="action-1",
        runtime="safe_shell_runtime",
        success=True,
        output_summary="tests passed",
        status=ActionStatus.SUCCEEDED,
    )

    assert record.event_kind == ActionAuditEventKind.EXECUTION_COMPLETED
    assert record.outcome == ActionAuditOutcome.SUCCEEDED
    assert record.source_runtime == "safe_shell_runtime"
    assert record.terminal is True


def test_record_execution_completed_failure() -> None:
    log = ActionAuditLog()

    record = log.record_execution_completed(
        action_id="action-1",
        runtime="safe_shell_runtime",
        success=False,
        output_summary="tests failed",
        status=ActionStatus.FAILED,
    )

    assert record.event_kind == ActionAuditEventKind.EXECUTION_FAILED
    assert record.outcome == ActionAuditOutcome.FAILED
    assert record.terminal is True


def test_record_interruption_cancelled() -> None:
    log = ActionAuditLog()

    record = log.record_interruption(
        action_id="action-1",
        message="user cancelled action",
        cancelled=True,
    )

    assert record.event_kind == ActionAuditEventKind.ACTION_CANCELLED
    assert record.outcome == ActionAuditOutcome.CANCELLED
    assert record.terminal is True


def test_record_rollback_success() -> None:
    log = ActionAuditLog()

    record = log.record_rollback(
        action_id="action-1",
        success=True,
        message="backup restored",
    )

    assert record.event_kind == ActionAuditEventKind.ROLLBACK_COMPLETED
    assert record.outcome == ActionAuditOutcome.ROLLED_BACK


def test_record_rollback_failure() -> None:
    log = ActionAuditLog()

    record = log.record_rollback(
        action_id="action-1",
        success=False,
        message="rollback failed",
    )

    assert record.event_kind == ActionAuditEventKind.ROLLBACK_FAILED
    assert record.outcome == ActionAuditOutcome.FAILED


def test_sensitive_data_is_redacted() -> None:
    log = ActionAuditLog()

    record = log.record(
        action_id="action-1",
        event_kind=ActionAuditEventKind.SECURITY_DECISION,
        actor=ActionAuditActor.SYSTEM,
        outcome=ActionAuditOutcome.BLOCKED,
        message="blocked sensitive operation",
        sensitivity=ActionAuditSensitivity.SENSITIVE,
        data={
            "username": "bala",
            "password": "secret",
            "nested": {
                "api_key": "abc",
            },
        },
    )

    assert record.data["username"] == "bala"
    assert record.data["password"] == "[REDACTED]"
    assert record.data["nested"]["api_key"] == "[REDACTED]"
    assert "data.password" in record.redacted_fields
    assert "data.nested.api_key" in record.redacted_fields


def test_redaction_can_be_disabled() -> None:
    log = ActionAuditLog(
        config=ActionAuditLogConfig(redact_sensitive_values=False)
    )

    record = log.record(
        action_id="action-1",
        event_kind=ActionAuditEventKind.SECURITY_DECISION,
        actor=ActionAuditActor.SYSTEM,
        outcome=ActionAuditOutcome.INFO,
        message="debug",
        data={"password": "secret"},
    )

    assert record.data["password"] == "secret"
    assert record.redacted_fields == ()


def test_persist_jsonl(tmp_path: Path) -> None:
    log_path = tmp_path / "audit" / "actions.jsonl"
    log = ActionAuditLog(
        config=ActionAuditLogConfig(
            persist_jsonl=True,
            log_path=str(log_path),
        )
    )

    record = log.record_intent(
        action_id="action-1",
        user_intent="run tests",
    )

    assert log_path.exists()

    lines = log_path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[0])

    assert payload["audit_id"] == record.audit_id
    assert payload["action_id"] == "action-1"


def test_max_records_trims_memory() -> None:
    log = ActionAuditLog(config=ActionAuditLogConfig(max_records_in_memory=2))

    log.record_intent(action_id="action-1", user_intent="one")
    log.record_intent(action_id="action-2", user_intent="two")
    log.record_intent(action_id="action-3", user_intent="three")

    records = log.all_records()

    assert len(records) == 2
    assert records[0].action_id == "action-2"
    assert records[1].action_id == "action-3"


def test_snapshot_and_reset() -> None:
    log = ActionAuditLog()

    log.record_intent(action_id="action-1", user_intent="run tests")
    snapshot = log.snapshot()

    assert snapshot.record_count == 1
    assert snapshot.last_sequence == 0
    assert snapshot.last_event_kind == ActionAuditEventKind.INTENT_RECEIVED
    assert snapshot.last_action_id == "action-1"
    assert snapshot.last_hash is not None

    log.reset()
    reset_snapshot = log.snapshot()

    assert reset_snapshot.record_count == 0
    assert reset_snapshot.last_sequence == -1
    assert reset_snapshot.last_hash is None


def test_enum_values_are_stable() -> None:
    assert ActionAuditEventKind.PLAN_PROPOSED.value == "plan_proposed"
    assert ActionAuditOutcome.APPROVAL_REQUIRED.value == "approval_required"
    assert ActionAuditActor.PLANNER.value == "planner"
    assert ActionAuditSensitivity.SENSITIVE.value == "sensitive"