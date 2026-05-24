from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.memory import (
    GovernedMemoryGateway,
    InMemoryMemoryStore,
    MemoryGateway,
    MemoryGatewayConfig,
    MemoryGatewayRetrievalResult,
    MemoryGatewayWriteResult,
    MemoryKind,
    MemoryPolicyClassification,
    MemoryQuery,
    MemorySensitivity,
    MemoryWriteRequest,
)


def make_write(
    *,
    text: str = "User is building a personal JARVIS memory gateway.",
    sensitivity: MemorySensitivity = MemorySensitivity.PRIVATE,
) -> MemoryWriteRequest:
    return MemoryWriteRequest(
        kind=MemoryKind.PROJECT,
        text=text,
        sensitivity=sensitivity,
        tags=("jarvis", "memory"),
    )


def test_memory_gateway_config_rejects_invalid_name() -> None:
    with pytest.raises(ValueError):
        MemoryGatewayConfig(name=" ").validate()


def test_memory_gateway_protocol() -> None:
    gateway: MemoryGateway = GovernedMemoryGateway(
        store=InMemoryMemoryStore(),
    )

    result = gateway.remember(make_write())

    assert gateway.name == "memory_gateway"
    assert result.allowed is True


def test_memory_gateway_write_result_requires_reason() -> None:
    with pytest.raises(ValidationError):
        MemoryGatewayWriteResult(
            request=make_write(),
            allowed=True,
            reason=" ",
            policy_classification=MemoryPolicyClassification.ALLOWED,
        )


def test_memory_gateway_retrieval_result_requires_reason() -> None:
    gateway = GovernedMemoryGateway(store=InMemoryMemoryStore())
    write_result = gateway.remember(make_write())

    assert write_result.record is not None

    retrieval = gateway.retrieve(MemoryQuery(text="gateway"))

    with pytest.raises(ValidationError):
        MemoryGatewayRetrievalResult(
            query=retrieval.query,
            retrieval=retrieval.retrieval,
            allowed=True,
            reason=" ",
            policy_classification=MemoryPolicyClassification.ALLOWED,
        )


def test_memory_gateway_remember_allows_private_memory() -> None:
    gateway = GovernedMemoryGateway(store=InMemoryMemoryStore())

    result = gateway.remember(make_write())
    snapshot = gateway.snapshot()

    assert result.allowed is True
    assert result.blocked is False
    assert result.record is not None
    assert result.policy_classification == MemoryPolicyClassification.ALLOWED
    assert snapshot.write_count == 1
    assert snapshot.write_allowed_count == 1
    assert snapshot.store_snapshot.record_count == 1


def test_memory_gateway_blocks_sensitive_write_by_default() -> None:
    gateway = GovernedMemoryGateway(store=InMemoryMemoryStore())

    result = gateway.remember(
        make_write(
            text="Sensitive private secret.",
            sensitivity=MemorySensitivity.SENSITIVE,
        )
    )
    snapshot = gateway.snapshot()

    assert result.allowed is False
    assert result.blocked is True
    assert result.record is None
    assert result.policy_classification == MemoryPolicyClassification.BLOCKED
    assert snapshot.write_count == 1
    assert snapshot.write_blocked_count == 1
    assert snapshot.store_snapshot.record_count == 0


def test_memory_gateway_can_allow_sensitive_write_when_configured() -> None:
    gateway = GovernedMemoryGateway(
        store=InMemoryMemoryStore(),
        config=MemoryGatewayConfig(allow_sensitive_writes=True),
    )

    result = gateway.remember(
        make_write(
            text="Sensitive memory allowed for test.",
            sensitivity=MemorySensitivity.SENSITIVE,
        )
    )

    assert result.allowed is True
    assert result.record is not None
    assert result.record.sensitivity == MemorySensitivity.SENSITIVE


def test_memory_gateway_retrieve_returns_explainable_results() -> None:
    gateway = GovernedMemoryGateway(store=InMemoryMemoryStore())
    gateway.remember(make_write(text="JARVIS memory gateway is active."))

    result = gateway.retrieve(MemoryQuery(text="memory gateway"))

    assert result.allowed is True
    assert result.blocked is False
    assert result.result_count == 1
    assert result.records[0].text == "JARVIS memory gateway is active."

    explanation = result.results[0].explanation

    assert explanation.source == result.records[0].source
    assert explanation.reason
    assert explanation.confidence > 0
    assert explanation.retrieved_at.tzinfo is not None
    assert explanation.policy_classification == MemoryPolicyClassification.ALLOWED


def test_memory_gateway_filters_sensitive_retrieval_by_default() -> None:
    gateway = GovernedMemoryGateway(
        store=InMemoryMemoryStore(),
        config=MemoryGatewayConfig(allow_sensitive_writes=True),
    )
    gateway.remember(
        make_write(
            text="Sensitive memory.",
            sensitivity=MemorySensitivity.SENSITIVE,
        )
    )
    gateway.remember(make_write(text="Private memory."))

    result = gateway.retrieve(
        MemoryQuery(
            include_sensitive=True,
        )
    )

    assert result.allowed is True
    assert result.query.include_sensitive is False
    assert result.result_count == 1
    assert result.records[0].text == "Private memory."
    assert result.reason == "memory retrieval allowed with sensitive results filtered"


def test_memory_gateway_can_allow_sensitive_retrieval_when_configured() -> None:
    gateway = GovernedMemoryGateway(
        store=InMemoryMemoryStore(),
        config=MemoryGatewayConfig(
            allow_sensitive_writes=True,
            allow_sensitive_retrieval=True,
        ),
    )
    gateway.remember(
        make_write(
            text="Sensitive memory.",
            sensitivity=MemorySensitivity.SENSITIVE,
        )
    )

    result = gateway.retrieve(MemoryQuery(include_sensitive=True))

    assert result.query.include_sensitive is True
    assert result.result_count == 1
    assert result.results[0].policy_classification == (
        MemoryPolicyClassification.RESTRICTED
    )


def test_memory_gateway_get_and_delete() -> None:
    gateway = GovernedMemoryGateway(store=InMemoryMemoryStore())
    write_result = gateway.remember(make_write())

    assert write_result.record is not None

    memory_id = write_result.record.memory_id

    assert gateway.get(memory_id) == write_result.record
    assert gateway.delete(memory_id) is True
    assert gateway.get(memory_id) is None
    assert gateway.snapshot().delete_count == 1


def test_memory_gateway_delete_can_be_blocked() -> None:
    gateway = GovernedMemoryGateway(
        store=InMemoryMemoryStore(),
        config=MemoryGatewayConfig(allow_delete=False),
    )
    write_result = gateway.remember(make_write())

    assert write_result.record is not None

    deleted = gateway.delete(write_result.record.memory_id)
    snapshot = gateway.snapshot()

    assert deleted is False
    assert snapshot.delete_blocked_count == 1
    assert snapshot.last_error == "delete blocked by gateway policy"


def test_memory_gateway_clear_blocked_by_default() -> None:
    gateway = GovernedMemoryGateway(store=InMemoryMemoryStore())

    gateway.remember(make_write())
    gateway.clear()
    snapshot = gateway.snapshot()

    assert snapshot.clear_count == 0
    assert snapshot.clear_blocked_count == 1
    assert snapshot.store_snapshot.record_count == 1


def test_memory_gateway_clear_allowed_when_configured() -> None:
    gateway = GovernedMemoryGateway(
        store=InMemoryMemoryStore(),
        config=MemoryGatewayConfig(allow_clear=True),
    )

    gateway.remember(make_write())
    gateway.clear()
    snapshot = gateway.snapshot()

    assert snapshot.clear_count == 1
    assert snapshot.store_snapshot.record_count == 0