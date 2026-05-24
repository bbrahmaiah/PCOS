from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from jarvis.memory import (
    InMemoryMemoryStore,
    MemoryImportance,
    MemoryKind,
    MemoryLifecycleDecision,
    MemoryLifecycleDecisionKind,
    MemoryLifecyclePolicy,
    MemoryLifecyclePolicyConfig,
    MemoryLifecycleReason,
    MemoryRecord,
    MemoryRetention,
    MemorySensitivity,
)


def make_record(
    *,
    retention: MemoryRetention = MemoryRetention.PERSISTENT,
    importance: MemoryImportance = MemoryImportance.NORMAL,
    sensitivity: MemorySensitivity = MemorySensitivity.PRIVATE,
    confidence: float = 1.0,
    updated_at: datetime | None = None,
    expires_at: datetime | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        kind=MemoryKind.PROJECT,
        text="Lifecycle managed memory.",
        retention=retention,
        importance=importance,
        sensitivity=sensitivity,
        confidence=confidence,
        updated_at=updated_at or datetime.now(UTC),
        expires_at=expires_at,
    )


def test_memory_lifecycle_policy_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        MemoryLifecyclePolicyConfig(name=" ").validate()

    with pytest.raises(ValueError):
        MemoryLifecyclePolicyConfig(temporary_ttl_seconds=0).validate()

    with pytest.raises(ValueError):
        MemoryLifecyclePolicyConfig(session_ttl_seconds=0).validate()

    with pytest.raises(ValueError):
        MemoryLifecyclePolicyConfig(low_confidence_stale_seconds=0).validate()

    with pytest.raises(ValueError):
        MemoryLifecyclePolicyConfig(sensitive_stale_seconds=0).validate()


def test_memory_lifecycle_decision_requires_text_fields() -> None:
    with pytest.raises(ValidationError):
        MemoryLifecycleDecision(
            memory_id=" ",
            decision=MemoryLifecycleDecisionKind.KEEP,
            reason=MemoryLifecycleReason.DEFAULT_KEEP,
            detail="valid",
        )

    with pytest.raises(ValidationError):
        MemoryLifecycleDecision(
            memory_id="memory-1",
            decision=MemoryLifecycleDecisionKind.KEEP,
            reason=MemoryLifecycleReason.DEFAULT_KEEP,
            detail=" ",
        )


def test_lifecycle_keeps_pinned_memory() -> None:
    policy = MemoryLifecyclePolicy()
    record = make_record(retention=MemoryRetention.PINNED)

    decision = policy.evaluate(record)

    assert decision.decision == MemoryLifecycleDecisionKind.PIN
    assert decision.reason == MemoryLifecycleReason.PINNED_MEMORY
    assert decision.delete_recommended is False
    assert decision.expire_recommended is False


def test_lifecycle_keeps_critical_memory() -> None:
    policy = MemoryLifecyclePolicy()
    record = make_record(importance=MemoryImportance.CRITICAL)

    decision = policy.evaluate(record)

    assert decision.decision == MemoryLifecycleDecisionKind.KEEP
    assert decision.reason == MemoryLifecycleReason.CRITICAL_MEMORY


def test_lifecycle_expires_record_with_explicit_expiration() -> None:
    policy = MemoryLifecyclePolicy()
    record = make_record(
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )

    decision = policy.evaluate(record)

    assert decision.decision == MemoryLifecycleDecisionKind.EXPIRE
    assert decision.reason == MemoryLifecycleReason.ALREADY_EXPIRED
    assert decision.expire_recommended is True


def test_lifecycle_delete_decision_when_delete_expired_enabled() -> None:
    policy = MemoryLifecyclePolicy(
        config=MemoryLifecyclePolicyConfig(delete_expired=True)
    )
    record = make_record(
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )

    decision = policy.evaluate(record)

    assert decision.decision == MemoryLifecycleDecisionKind.DELETE
    assert decision.delete_recommended is True


def test_lifecycle_expires_temporary_memory_after_ttl() -> None:
    policy = MemoryLifecyclePolicy(
        config=MemoryLifecyclePolicyConfig(temporary_ttl_seconds=10)
    )
    record = make_record(
        retention=MemoryRetention.TEMPORARY,
        updated_at=datetime.now(UTC) - timedelta(seconds=20),
    )

    decision = policy.evaluate(record)

    assert decision.decision == MemoryLifecycleDecisionKind.EXPIRE
    assert decision.reason == MemoryLifecycleReason.TEMPORARY_EXPIRED


def test_lifecycle_expires_session_memory_after_ttl() -> None:
    policy = MemoryLifecyclePolicy(
        config=MemoryLifecyclePolicyConfig(session_ttl_seconds=10)
    )
    record = make_record(
        retention=MemoryRetention.SESSION,
        updated_at=datetime.now(UTC) - timedelta(seconds=20),
    )

    decision = policy.evaluate(record)

    assert decision.decision == MemoryLifecycleDecisionKind.EXPIRE
    assert decision.reason == MemoryLifecycleReason.SESSION_EXPIRED


def test_lifecycle_expires_low_confidence_stale_memory() -> None:
    policy = MemoryLifecyclePolicy(
        config=MemoryLifecyclePolicyConfig(low_confidence_stale_seconds=10)
    )
    record = make_record(
        confidence=0.2,
        updated_at=datetime.now(UTC) - timedelta(seconds=20),
    )

    decision = policy.evaluate(record)

    assert decision.decision == MemoryLifecycleDecisionKind.EXPIRE
    assert decision.reason == MemoryLifecycleReason.LOW_CONFIDENCE_STALE


def test_lifecycle_expires_sensitive_stale_memory() -> None:
    policy = MemoryLifecyclePolicy(
        config=MemoryLifecyclePolicyConfig(sensitive_stale_seconds=10)
    )
    record = make_record(
        sensitivity=MemorySensitivity.SENSITIVE,
        updated_at=datetime.now(UTC) - timedelta(seconds=20),
    )

    decision = policy.evaluate(record)

    assert decision.decision == MemoryLifecycleDecisionKind.EXPIRE
    assert decision.reason == MemoryLifecycleReason.SENSITIVE_STALE


def test_lifecycle_keeps_active_persistent_memory() -> None:
    policy = MemoryLifecyclePolicy()
    record = make_record(retention=MemoryRetention.PERSISTENT)

    decision = policy.evaluate(record)

    assert decision.decision == MemoryLifecycleDecisionKind.KEEP
    assert decision.reason == MemoryLifecycleReason.PERSISTENT_ACTIVE


def test_lifecycle_sweep_without_delete() -> None:
    store = InMemoryMemoryStore()
    expired = store.put(
        make_record(
            retention=MemoryRetention.TEMPORARY,
            updated_at=datetime.now(UTC) - timedelta(seconds=20),
        )
    )
    active = store.put(make_record())

    policy = MemoryLifecyclePolicy(
        config=MemoryLifecyclePolicyConfig(temporary_ttl_seconds=10)
    )

    result = policy.sweep(store)

    assert result.evaluated_count == 2
    assert result.expired_count == 1
    assert result.deleted_count == 0
    assert store.get(expired.memory_id) is not None
    assert store.get(active.memory_id) is not None


def test_lifecycle_sweep_with_delete() -> None:
    store = InMemoryMemoryStore()
    expired = store.put(
        make_record(
            expires_at=datetime.now(UTC) - timedelta(seconds=1),
        )
    )

    policy = MemoryLifecyclePolicy(
        config=MemoryLifecyclePolicyConfig(delete_expired=True)
    )

    result = policy.sweep(store)

    assert result.evaluated_count == 1
    assert result.expired_count == 1
    assert result.deleted_count == 1
    assert store.get(expired.memory_id) is None


def test_lifecycle_snapshot_counts() -> None:
    policy = MemoryLifecyclePolicy()

    policy.evaluate(make_record())
    policy.evaluate(make_record(retention=MemoryRetention.PINNED))

    snapshot = policy.snapshot()

    assert snapshot.evaluated_count == 2
    assert snapshot.kept_count == 2
    assert snapshot.last_memory_id is not None
    assert snapshot.last_decision == MemoryLifecycleDecisionKind.PIN


def test_lifecycle_reset() -> None:
    policy = MemoryLifecyclePolicy()

    policy.evaluate(make_record())
    policy.reset()

    snapshot = policy.snapshot()

    assert snapshot.evaluated_count == 0
    assert snapshot.sweep_count == 0
    assert snapshot.kept_count == 0
    assert snapshot.last_memory_id is None
    assert snapshot.last_error is None


def test_memory_lifecycle_enum_values_are_stable() -> None:
    assert MemoryLifecycleDecisionKind.KEEP.value == "keep"
    assert MemoryLifecycleDecisionKind.EXPIRE.value == "expire"
    assert MemoryLifecycleDecisionKind.DELETE.value == "delete"
    assert MemoryLifecycleReason.PINNED_MEMORY.value == "pinned_memory"
    assert MemoryLifecycleReason.SENSITIVE_STALE.value == "sensitive_stale"