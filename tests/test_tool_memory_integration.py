from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.tools import (
    ActionAuditEventKind,
    ActionAuditLog,
    ActionRisk,
    ActionStatus,
    ToolMemoryDecision,
    ToolMemoryEvent,
    ToolMemoryEventKind,
    ToolMemoryImportance,
    ToolMemoryIntegrationConfig,
    ToolMemoryIntegrationRuntime,
    ToolMemoryPolicy,
    ToolMemoryPolicyClass,
    ToolMemoryReason,
    ToolMemoryWriteProposal,
)


class FakeMemoryGateway:
    def __init__(self) -> None:
        self.writes: list[dict[str, object]] = []

    def write_tool_memory(
        self,
        *,
        content: str,
        source: str,
        confidence: float,
        policy_class: str,
        reason: str,
        tags: tuple[str, ...],
        metadata: dict[str, object],
    ) -> str:
        self.writes.append(
            {
                "content": content,
                "source": source,
                "confidence": confidence,
                "policy_class": policy_class,
                "reason": reason,
                "tags": tags,
                "metadata": metadata,
            }
        )

        return "memory-1"


def event(
    *,
    kind: ToolMemoryEventKind = ToolMemoryEventKind.EXECUTION_COMPLETED,
    summary: str = "pytest completed successfully",
    risk: ActionRisk = ActionRisk.LOW,
    status: ActionStatus | None = ActionStatus.SUCCEEDED,
    source_runtime: str | None = "safe_shell_runtime",
    data: dict[str, object] | None = None,
    user_visible: bool = True,
    approved_by_user: bool = False,
) -> ToolMemoryEvent:
    return ToolMemoryEvent(
        action_id="action-1",
        kind=kind,
        summary=summary,
        risk=risk,
        status=status,
        source_runtime=source_runtime,
        data=data or {},
        user_visible=user_visible,
        approved_by_user=approved_by_user,
    )


def test_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        ToolMemoryIntegrationConfig(name=" ").validate()


def test_event_requires_summary() -> None:
    with pytest.raises(ValidationError):
        ToolMemoryEvent(
            action_id="action-1",
            kind=ToolMemoryEventKind.EXECUTION_COMPLETED,
            summary=" ",
        )


def test_write_proposal_rejects_store_with_do_not_store() -> None:
    with pytest.raises(ValidationError):
        ToolMemoryWriteProposal(
            action_id="action-1",
            event_id="event-1",
            decision=ToolMemoryDecision.STORE,
            reason=ToolMemoryReason.SAFE_ACTION_SUMMARY,
            policy_class=ToolMemoryPolicyClass.DO_NOT_STORE,
            importance=ToolMemoryImportance.LOW,
            content="bad",
        )


def test_policy_skips_low_value_events_by_default() -> None:
    proposal = ToolMemoryPolicy().evaluate(
        event(kind=ToolMemoryEventKind.ACTION_REQUESTED),
        store_low_value_events=False,
    )

    assert proposal.decision == ToolMemoryDecision.SKIP
    assert proposal.reason == ToolMemoryReason.LOW_VALUE_EVENT_SKIPPED


def test_policy_stores_successful_action_summary() -> None:
    proposal = ToolMemoryPolicy().evaluate(
        event(),
        store_low_value_events=False,
    )

    assert proposal.decision == ToolMemoryDecision.STORE
    assert proposal.reason == ToolMemoryReason.SAFE_ACTION_SUMMARY
    assert proposal.policy_class == ToolMemoryPolicyClass.USER_PRIVATE
    assert "pytest completed successfully" in proposal.content


def test_policy_classifies_workspace_runtime() -> None:
    proposal = ToolMemoryPolicy().evaluate(
        event(source_runtime="file_system_runtime"),
        store_low_value_events=False,
    )

    assert proposal.policy_class == ToolMemoryPolicyClass.WORKSPACE


def test_policy_redacts_sensitive_data() -> None:
    proposal = ToolMemoryPolicy().evaluate(
        event(data={"password": "secret", "file": "a.py"}),
        store_low_value_events=False,
    )

    assert proposal.decision == ToolMemoryDecision.REDACT_AND_STORE
    assert proposal.reason == ToolMemoryReason.SENSITIVE_DATA_REDACTED
    assert proposal.policy_class == ToolMemoryPolicyClass.SENSITIVE_REDACTED
    assert "data.password" in proposal.redacted_fields
    assert "password=[REDACTED]" in proposal.content


def test_policy_blocks_sensitive_summary_without_structured_redaction() -> None:
    proposal = ToolMemoryPolicy().evaluate(
        event(summary="user entered password secret"),
        store_low_value_events=False,
    )

    assert proposal.decision == ToolMemoryDecision.BLOCK
    assert proposal.reason == ToolMemoryReason.SENSITIVE_DATA_BLOCKED


def test_policy_blocks_critical_risk_event() -> None:
    proposal = ToolMemoryPolicy().evaluate(
        event(risk=ActionRisk.CRITICAL),
        store_low_value_events=False,
    )

    assert proposal.decision == ToolMemoryDecision.BLOCK
    assert proposal.reason == ToolMemoryReason.HIGH_RISK_EVENT_BLOCKED


def test_user_preference_has_high_importance() -> None:
    proposal = ToolMemoryPolicy().evaluate(
        event(
            kind=ToolMemoryEventKind.USER_PREFERENCE_LEARNED,
            summary="User prefers ruff before pytest",
            approved_by_user=True,
        ),
        store_low_value_events=False,
    )

    assert proposal.decision == ToolMemoryDecision.STORE
    assert proposal.reason == ToolMemoryReason.SAFE_USER_PREFERENCE
    assert proposal.importance == ToolMemoryImportance.HIGH
    assert proposal.confidence == 0.95


def test_runtime_blocks_hidden_event() -> None:
    runtime = ToolMemoryIntegrationRuntime()

    proposal = runtime.propose(event(user_visible=False))

    assert proposal.decision == ToolMemoryDecision.BLOCK
    assert proposal.policy_class == ToolMemoryPolicyClass.DO_NOT_STORE


def test_runtime_write_without_gateway_fails_safely() -> None:
    runtime = ToolMemoryIntegrationRuntime()
    proposal = runtime.propose(event())

    result = runtime.write(proposal)

    assert result.success is False
    assert result.stored is False
    assert result.reason == ToolMemoryReason.MEMORY_WRITE_FAILED


def test_runtime_write_disabled() -> None:
    runtime = ToolMemoryIntegrationRuntime(
        config=ToolMemoryIntegrationConfig(allow_memory_writes=False)
    )
    proposal = runtime.propose(event())

    result = runtime.write(proposal)

    assert result.success is False
    assert result.stored is False
    assert result.reason == ToolMemoryReason.MEMORY_GATEWAY_UNAVAILABLE


def test_runtime_write_skip_does_not_call_gateway() -> None:
    gateway = FakeMemoryGateway()
    runtime = ToolMemoryIntegrationRuntime(memory_gateway=gateway)
    proposal = runtime.propose(event(kind=ToolMemoryEventKind.ACTION_REQUESTED))

    result = runtime.write(proposal)

    assert result.success is True
    assert result.stored is False
    assert gateway.writes == []


def test_runtime_write_block_does_not_call_gateway() -> None:
    gateway = FakeMemoryGateway()
    runtime = ToolMemoryIntegrationRuntime(memory_gateway=gateway)
    proposal = runtime.propose(event(risk=ActionRisk.CRITICAL))

    result = runtime.write(proposal)

    assert result.success is False
    assert result.stored is False
    assert gateway.writes == []


def test_runtime_stores_through_gateway() -> None:
    gateway = FakeMemoryGateway()
    runtime = ToolMemoryIntegrationRuntime(memory_gateway=gateway)
    proposal = runtime.propose(event())

    result = runtime.write(proposal)

    assert result.success is True
    assert result.stored is True
    assert result.memory_id == "memory-1"
    assert gateway.writes
    assert gateway.writes[0]["source"] == "tool_memory_integration"


def test_runtime_process_propose_and_write() -> None:
    gateway = FakeMemoryGateway()
    runtime = ToolMemoryIntegrationRuntime(memory_gateway=gateway)

    result = runtime.process(event())

    assert result.success is True
    assert result.stored is True


def test_audit_integration_records_memory_decision() -> None:
    audit = ActionAuditLog()
    runtime = ToolMemoryIntegrationRuntime(audit_log=audit)

    runtime.propose(event())

    records = audit.all_records()

    assert len(records) == 1
    assert records[0].event_kind == ActionAuditEventKind.MEMORY_CONTEXT_USED


def test_snapshot_and_reset() -> None:
    gateway = FakeMemoryGateway()
    runtime = ToolMemoryIntegrationRuntime(memory_gateway=gateway)

    runtime.process(event())
    snapshot = runtime.snapshot()

    assert snapshot.event_count == 1
    assert snapshot.proposal_count == 1
    assert snapshot.write_count == 1
    assert snapshot.stored_count == 1

    runtime.reset()
    reset_snapshot = runtime.snapshot()

    assert reset_snapshot.event_count == 0
    assert reset_snapshot.last_decision is None


def test_enum_values_are_stable() -> None:
    assert ToolMemoryEventKind.EXECUTION_COMPLETED.value == "execution_completed"
    assert ToolMemoryDecision.REDACT_AND_STORE.value == "redact_and_store"
    assert ToolMemoryPolicyClass.DO_NOT_STORE.value == "do_not_store"
    assert ToolMemoryReason.MEMORY_WRITE_SUCCEEDED.value == "memory_write_succeeded"