from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.memory import (
    MemoryKind,
    MemoryPolicyClassification,
    MemoryPrivacyDecisionKind,
    MemoryPrivacyPolicy,
    MemoryPrivacyPolicyConfig,
    MemoryPrivacyPolicyDecision,
    MemoryPrivacyRiskLevel,
    MemoryPrivacySubject,
    MemoryRecord,
    MemorySensitivity,
    MemoryWriteRequest,
)


def make_request(
    *,
    text: str = "User prefers direct engineering explanations.",
    sensitivity: MemorySensitivity = MemorySensitivity.PRIVATE,
) -> MemoryWriteRequest:
    return MemoryWriteRequest(
        kind=MemoryKind.PREFERENCE,
        text=text,
        sensitivity=sensitivity,
    )


def test_memory_privacy_policy_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        MemoryPrivacyPolicyConfig(name=" ").validate()

    with pytest.raises(ValueError):
        MemoryPrivacyPolicyConfig(secret_terms=(" ",)).validate()

    with pytest.raises(ValueError):
        MemoryPrivacyPolicyConfig(critical_terms=(" ",)).validate()

    with pytest.raises(ValueError):
        MemoryPrivacyPolicyConfig(private_terms=(" ",)).validate()


def test_memory_privacy_policy_decision_requires_text_and_reason() -> None:
    with pytest.raises(ValidationError):
        MemoryPrivacyPolicyDecision(
            subject=MemoryPrivacySubject.WRITE_REQUEST,
            text=" ",
            decision=MemoryPrivacyDecisionKind.ALLOW,
            risk_level=MemoryPrivacyRiskLevel.LOW,
            sensitivity=MemorySensitivity.PUBLIC,
            policy_classification=MemoryPolicyClassification.ALLOWED,
            allowed=True,
            reasons=("valid",),
        )

    with pytest.raises(ValidationError):
        MemoryPrivacyPolicyDecision(
            subject=MemoryPrivacySubject.WRITE_REQUEST,
            text="valid text",
            decision=MemoryPrivacyDecisionKind.ALLOW,
            risk_level=MemoryPrivacyRiskLevel.LOW,
            sensitivity=MemorySensitivity.PUBLIC,
            policy_classification=MemoryPolicyClassification.ALLOWED,
            allowed=True,
            reasons=(" ",),
        )


def test_privacy_policy_allows_private_memory() -> None:
    policy = MemoryPrivacyPolicy()

    decision = policy.evaluate_write_request(make_request())

    assert decision.allowed is True
    assert decision.blocked is False
    assert decision.decision == MemoryPrivacyDecisionKind.ALLOW
    assert decision.risk_level == MemoryPrivacyRiskLevel.MEDIUM
    assert decision.sensitivity == MemorySensitivity.PRIVATE
    assert decision.policy_classification == MemoryPolicyClassification.ALLOWED


def test_privacy_policy_restricts_secret_like_content() -> None:
    policy = MemoryPrivacyPolicy()

    decision = policy.evaluate_write_request(
        make_request(text="My password is temporary123.")
    )

    assert decision.allowed is True
    assert decision.blocked is False
    assert decision.decision == MemoryPrivacyDecisionKind.RESTRICT
    assert decision.risk_level == MemoryPrivacyRiskLevel.HIGH
    assert decision.sensitivity == MemorySensitivity.SENSITIVE
    assert decision.policy_classification == MemoryPolicyClassification.RESTRICTED
    assert "password" in decision.matched_terms


def test_privacy_policy_blocks_critical_content_by_default() -> None:
    policy = MemoryPrivacyPolicy()

    decision = policy.evaluate_write_request(
        make_request(text="My seed phrase is alpha beta gamma.")
    )

    assert decision.allowed is False
    assert decision.blocked is True
    assert decision.decision == MemoryPrivacyDecisionKind.BLOCK
    assert decision.risk_level == MemoryPrivacyRiskLevel.CRITICAL
    assert decision.policy_classification == MemoryPolicyClassification.BLOCKED
    assert "seed phrase" in decision.matched_terms


def test_privacy_policy_can_redact_high_risk_content() -> None:
    policy = MemoryPrivacyPolicy(
        config=MemoryPrivacyPolicyConfig(redact_high_risk_content=True)
    )

    decision = policy.evaluate_write_request(
        make_request(text="The password is temporary123.")
    )

    assert decision.allowed is True
    assert decision.decision == MemoryPrivacyDecisionKind.REDACT
    assert decision.redacted_text is not None
    assert "[REDACTED]" in decision.redacted_text


def test_privacy_policy_restricts_declared_sensitive_memory() -> None:
    policy = MemoryPrivacyPolicy()

    decision = policy.evaluate_write_request(
        make_request(
            text="This is declared sensitive.",
            sensitivity=MemorySensitivity.SENSITIVE,
        )
    )

    assert decision.allowed is True
    assert decision.decision == MemoryPrivacyDecisionKind.RESTRICT
    assert decision.policy_classification == MemoryPolicyClassification.RESTRICTED


def test_privacy_policy_evaluates_record() -> None:
    policy = MemoryPrivacyPolicy()
    record = MemoryRecord(
        kind=MemoryKind.USER_PROFILE,
        text="User email is example@example.com.",
        sensitivity=MemorySensitivity.PRIVATE,
    )

    decision = policy.evaluate_record(record)

    assert decision.subject == MemoryPrivacySubject.RECORD
    assert decision.allowed is True
    assert decision.metadata["memory_id"] == record.memory_id


def test_privacy_policy_snapshot_counts() -> None:
    policy = MemoryPrivacyPolicy()

    policy.evaluate_write_request(make_request())
    policy.evaluate_write_request(make_request(text="My password is abc."))
    policy.evaluate_write_request(make_request(text="My seed phrase is abc."))

    snapshot = policy.snapshot()

    assert snapshot.evaluated_count == 3
    assert snapshot.allowed_count == 1
    assert snapshot.restricted_count == 1
    assert snapshot.blocked_count == 1
    assert snapshot.last_decision == MemoryPrivacyDecisionKind.BLOCK
    assert snapshot.last_risk_level == MemoryPrivacyRiskLevel.CRITICAL


def test_privacy_policy_reset() -> None:
    policy = MemoryPrivacyPolicy()

    policy.evaluate_write_request(make_request())
    policy.reset()

    snapshot = policy.snapshot()

    assert snapshot.evaluated_count == 0
    assert snapshot.allowed_count == 0
    assert snapshot.restricted_count == 0
    assert snapshot.blocked_count == 0
    assert snapshot.last_decision is None
    assert snapshot.last_error is None


def test_memory_privacy_enum_values_are_stable() -> None:
    assert MemoryPrivacyDecisionKind.ALLOW.value == "allow"
    assert MemoryPrivacyDecisionKind.RESTRICT.value == "restrict"
    assert MemoryPrivacyDecisionKind.REDACT.value == "redact"
    assert MemoryPrivacyDecisionKind.BLOCK.value == "block"
    assert MemoryPrivacyRiskLevel.CRITICAL.value == "critical"
    assert MemoryPrivacySubject.WRITE_REQUEST.value == "write_request"