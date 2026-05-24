from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.memory import (
    GovernedMemoryGateway,
    InMemoryMemoryStore,
    MemoryKind,
    MemoryPolicyClassification,
    MemoryRetention,
    MemorySensitivity,
    MemorySource,
    MemoryWriteDecisionKind,
    MemoryWritePolicy,
    MemoryWritePolicyConfig,
    MemoryWritePolicyDecision,
    MemoryWriteRequest,
    MemoryWriteRiskLevel,
)


def make_request(
    *,
    text: str = "User is building governed memory write policy.",
    sensitivity: MemorySensitivity = MemorySensitivity.PRIVATE,
    source: MemorySource = MemorySource.CONVERSATION,
    retention: MemoryRetention = MemoryRetention.PERSISTENT,
    confidence: float = 1.0,
) -> MemoryWriteRequest:
    return MemoryWriteRequest(
        kind=MemoryKind.PROJECT,
        text=text,
        sensitivity=sensitivity,
        source=source,
        retention=retention,
        confidence=confidence,
    )


def test_memory_write_policy_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        MemoryWritePolicyConfig(name=" ").validate()

    with pytest.raises(ValueError):
        MemoryWritePolicyConfig(min_write_confidence=-0.1).validate()

    with pytest.raises(ValueError):
        MemoryWritePolicyConfig(min_write_confidence=1.1).validate()


def test_memory_write_policy_decision_requires_reason() -> None:
    request = make_request()

    with pytest.raises(ValidationError):
        MemoryWritePolicyDecision(
            request=request,
            decision=MemoryWriteDecisionKind.ALLOW,
            risk_level=MemoryWriteRiskLevel.LOW,
            allowed=True,
            reason=" ",
            policy_classification=MemoryPolicyClassification.ALLOWED,
        )


def test_memory_write_policy_allows_private_memory() -> None:
    policy = MemoryWritePolicy()
    request = make_request()

    decision = policy.evaluate(request)
    snapshot = policy.snapshot()

    assert decision.allowed is True
    assert decision.blocked is False
    assert decision.effective_request == request
    assert decision.decision == MemoryWriteDecisionKind.ALLOW
    assert decision.risk_level == MemoryWriteRiskLevel.MEDIUM
    assert decision.policy_classification == MemoryPolicyClassification.ALLOWED
    assert snapshot.evaluated_count == 1
    assert snapshot.allowed_count == 1


def test_memory_write_policy_blocks_sensitive_by_default() -> None:
    policy = MemoryWritePolicy()
    request = make_request(sensitivity=MemorySensitivity.SENSITIVE)

    decision = policy.evaluate(request)
    snapshot = policy.snapshot()

    assert decision.allowed is False
    assert decision.blocked is True
    assert decision.effective_request is None
    assert decision.decision == MemoryWriteDecisionKind.BLOCK
    assert decision.policy_classification == MemoryPolicyClassification.BLOCKED
    assert decision.reason == "blocked sensitive memory write by gateway policy"
    assert snapshot.blocked_count == 1
    assert snapshot.last_error == "blocked sensitive memory write by gateway policy"


def test_memory_write_policy_can_allow_sensitive_when_configured() -> None:
    policy = MemoryWritePolicy(
        config=MemoryWritePolicyConfig(allow_sensitive_writes=True)
    )
    request = make_request(sensitivity=MemorySensitivity.SENSITIVE)

    decision = policy.evaluate(request)

    assert decision.allowed is True
    assert decision.blocked is False
    assert decision.policy_classification == MemoryPolicyClassification.RESTRICTED
    assert decision.risk_level == MemoryWriteRiskLevel.HIGH


def test_memory_write_policy_blocks_low_confidence_by_default() -> None:
    policy = MemoryWritePolicy()
    request = make_request(confidence=0.1)

    decision = policy.evaluate(request)

    assert decision.allowed is False
    assert decision.blocked is True
    assert decision.reason == "blocked low-confidence memory write by policy"


def test_memory_write_policy_can_allow_low_confidence_when_configured() -> None:
    policy = MemoryWritePolicy(
        config=MemoryWritePolicyConfig(allow_low_confidence_writes=True)
    )
    request = make_request(confidence=0.1)

    decision = policy.evaluate(request)

    assert decision.allowed is True
    assert decision.blocked is False


def test_memory_write_policy_blocks_implicit_pinned_memory() -> None:
    policy = MemoryWritePolicy()
    request = make_request(
        retention=MemoryRetention.PINNED,
        source=MemorySource.CONVERSATION,
    )

    decision = policy.evaluate(request)

    assert decision.allowed is False
    assert decision.blocked is True
    assert decision.reason == "blocked pinned memory without explicit user source"


def test_memory_write_policy_allows_explicit_pinned_memory() -> None:
    policy = MemoryWritePolicy()
    request = make_request(
        retention=MemoryRetention.PINNED,
        source=MemorySource.USER_EXPLICIT,
    )

    decision = policy.evaluate(request)

    assert decision.allowed is True
    assert decision.risk_level == MemoryWriteRiskLevel.HIGH


def test_memory_write_policy_downgrades_system_private_to_internal() -> None:
    policy = MemoryWritePolicy()
    request = make_request(
        sensitivity=MemorySensitivity.PRIVATE,
        source=MemorySource.SYSTEM,
    )

    decision = policy.evaluate(request)

    assert decision.allowed is True
    assert decision.decision == MemoryWriteDecisionKind.DOWNGRADE
    assert decision.effective_request is not None
    assert decision.effective_request.sensitivity == MemorySensitivity.INTERNAL
    assert decision.effective_request.metadata["write_policy_downgraded"] is True


def test_memory_write_policy_reset_clears_counters() -> None:
    policy = MemoryWritePolicy()

    policy.evaluate(make_request())
    policy.reset()
    snapshot = policy.snapshot()

    assert snapshot.evaluated_count == 0
    assert snapshot.allowed_count == 0
    assert snapshot.blocked_count == 0
    assert snapshot.last_request_id is None
    assert snapshot.last_error is None


def test_memory_gateway_uses_write_policy() -> None:
    gateway = GovernedMemoryGateway(store=InMemoryMemoryStore())

    result = gateway.remember(
        make_request(sensitivity=MemorySensitivity.SENSITIVE)
    )
    snapshot = gateway.snapshot()

    assert result.allowed is False
    assert result.blocked is True
    assert result.reason == "blocked sensitive memory write by gateway policy"
    assert snapshot.write_blocked_count == 1
    assert snapshot.write_policy_snapshot.blocked_count == 1
    assert snapshot.store_snapshot.record_count == 0


def test_memory_gateway_stores_effective_policy_request() -> None:
    gateway = GovernedMemoryGateway(store=InMemoryMemoryStore())

    result = gateway.remember(
        make_request(
            sensitivity=MemorySensitivity.PRIVATE,
            source=MemorySource.SYSTEM,
        )
    )

    assert result.allowed is True
    assert result.record is not None
    assert result.record.sensitivity == MemorySensitivity.INTERNAL
    assert result.record.metadata["write_policy_downgraded"] is True


def test_memory_write_policy_enum_values_are_stable() -> None:
    assert MemoryWriteDecisionKind.ALLOW.value == "allow"
    assert MemoryWriteDecisionKind.BLOCK.value == "block"
    assert MemoryWriteDecisionKind.DOWNGRADE.value == "downgrade"
    assert MemoryWriteRiskLevel.CRITICAL.value == "critical"