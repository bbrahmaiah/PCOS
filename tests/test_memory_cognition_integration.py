from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.memory import (
    ExtractiveMemorySummarizer,
    GovernedMemoryGateway,
    InMemoryMemoryStore,
    MemoryCognitionBridge,
    MemoryCognitionBridgeConfig,
    MemoryCognitionQuery,
    MemoryContextBuildStatus,
    MemoryImportance,
    MemoryKind,
    MemoryPolicyClassification,
    MemorySensitivity,
    MemoryWriteRequest,
)


def make_gateway() -> GovernedMemoryGateway:
    return GovernedMemoryGateway(store=InMemoryMemoryStore())


def seed_memory(gateway: GovernedMemoryGateway) -> None:
    gateway.remember(
        MemoryWriteRequest(
            kind=MemoryKind.PROJECT,
            text="JARVIS memory gateway is the only cognition-facing entry point.",
            importance=MemoryImportance.CRITICAL,
            tags=("jarvis", "memory", "gateway"),
        )
    )
    gateway.remember(
        MemoryWriteRequest(
            kind=MemoryKind.SEMANTIC,
            text="Memory context builder prepares bounded cognition context.",
            importance=MemoryImportance.HIGH,
            tags=("jarvis", "context"),
        )
    )


def test_memory_cognition_bridge_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        MemoryCognitionBridgeConfig(name=" ").validate()

    with pytest.raises(ValueError):
        MemoryCognitionBridgeConfig(default_max_results=0).validate()

    with pytest.raises(ValueError):
        MemoryCognitionBridgeConfig(default_max_context_items=0).validate()

    with pytest.raises(ValueError):
        MemoryCognitionBridgeConfig(default_max_context_chars=79).validate()


def test_memory_cognition_query_requires_text() -> None:
    with pytest.raises(ValidationError):
        MemoryCognitionQuery(text=" ")


def test_memory_cognition_query_to_memory_query() -> None:
    query = MemoryCognitionQuery(
        text="memory gateway",
        kinds=(MemoryKind.PROJECT,),
        max_results=5,
        include_sensitive=True,
        include_expired=True,
    )
    memory_query = query.to_memory_query()

    assert memory_query.text == "memory gateway"
    assert memory_query.kinds == (MemoryKind.PROJECT,)
    assert memory_query.max_results == 5
    assert memory_query.include_sensitive is True
    assert memory_query.include_expired is True
    assert memory_query.metadata["cognition_memory_query"] is True


def test_memory_cognition_bridge_builds_context() -> None:
    gateway = make_gateway()
    seed_memory(gateway)
    bridge = MemoryCognitionBridge(gateway=gateway)

    result = bridge.build_context_from_text("memory gateway")
    snapshot = bridge.snapshot()

    assert result.allowed is True
    assert result.blocked is False
    assert result.context_item_count >= 1
    assert result.context.status == MemoryContextBuildStatus.BUILT
    assert result.context.items[0].reason
    assert result.context.items[0].confidence > 0
    assert result.context.items[0].timestamp.tzinfo is not None
    assert result.context.items[0].policy_classification == (
        MemoryPolicyClassification.ALLOWED
    )
    assert snapshot.build_count == 1
    assert snapshot.built_count == 1


def test_memory_cognition_bridge_empty_context_when_no_match() -> None:
    gateway = make_gateway()
    seed_memory(gateway)
    bridge = MemoryCognitionBridge(gateway=gateway)

    result = bridge.build_context_from_text("unmatched query")

    assert result.allowed is True
    assert result.empty is True
    assert result.context.status == MemoryContextBuildStatus.EMPTY
    assert bridge.snapshot().empty_count == 1


def test_memory_cognition_bridge_respects_kind_filter() -> None:
    gateway = make_gateway()
    seed_memory(gateway)
    bridge = MemoryCognitionBridge(gateway=gateway)

    result = bridge.build_context_from_text(
        "memory gateway",
        kinds=(MemoryKind.PROJECT,),
    )

    assert result.allowed is True
    assert result.context_item_count == 1
    assert result.context.items[0].memory_kind == MemoryKind.PROJECT


def test_memory_cognition_bridge_supports_summary() -> None:
    gateway = make_gateway()
    seed_memory(gateway)
    bridge = MemoryCognitionBridge(
        gateway=gateway,
        summarizer=ExtractiveMemorySummarizer(),
    )

    result = bridge.build_context(
        MemoryCognitionQuery(
            text="memory",
            include_summary=True,
            max_context_items=3,
        )
    )
    snapshot = bridge.snapshot()

    assert result.allowed is True
    assert result.summary_result is not None
    assert result.summary_result.succeeded is True
    assert result.context_item_count >= 1
    assert snapshot.summary_count == 1


def test_memory_cognition_bridge_can_disable_summarization() -> None:
    gateway = make_gateway()
    seed_memory(gateway)
    bridge = MemoryCognitionBridge(
        gateway=gateway,
        summarizer=ExtractiveMemorySummarizer(),
        config=MemoryCognitionBridgeConfig(enable_summarization=False),
    )

    result = bridge.build_context(
        MemoryCognitionQuery(
            text="memory",
            include_summary=True,
        )
    )

    assert result.allowed is True
    assert result.summary_result is None
    assert bridge.snapshot().summary_count == 0


def test_memory_cognition_bridge_filters_restricted_context_by_default() -> None:
    gateway = GovernedMemoryGateway(
        store=InMemoryMemoryStore(),
    )
    gateway.store.put(
        MemoryWriteRequest(
            kind=MemoryKind.PROJECT,
            text="Sensitive memory gateway note.",
            sensitivity=MemorySensitivity.SENSITIVE,
        ).to_record()
    )
    bridge = MemoryCognitionBridge(gateway=gateway)

    result = bridge.build_context(
        MemoryCognitionQuery(
            text="memory gateway",
            include_sensitive=True,
        )
    )

    assert result.allowed is True
    assert result.empty is True


def test_memory_cognition_bridge_can_include_restricted_context() -> None:
    store = InMemoryMemoryStore()
    store.put(
        MemoryWriteRequest(
            kind=MemoryKind.PROJECT,
            text="Sensitive memory gateway note.",
            sensitivity=MemorySensitivity.SENSITIVE,
        ).to_record()
    )
    gateway = GovernedMemoryGateway(
        store=store,
        config=__import__("jarvis.memory").memory.MemoryGatewayConfig(
            allow_sensitive_retrieval=True
        ),
    )
    bridge = MemoryCognitionBridge(gateway=gateway)

    result = bridge.build_context(
        MemoryCognitionQuery(
            text="memory gateway",
            include_sensitive=True,
            include_restricted_context=True,
        )
    )

    assert result.allowed is True
    assert result.context_item_count == 1
    assert result.context.items[0].policy_classification == (
        MemoryPolicyClassification.RESTRICTED
    )


def test_memory_cognition_result_metadata() -> None:
    gateway = make_gateway()
    seed_memory(gateway)
    bridge = MemoryCognitionBridge(gateway=gateway)

    result = bridge.build_context_from_text("memory gateway")
    metadata = result.as_cognition_metadata()

    assert metadata["memory_context_id"] == result.context.context_id
    assert metadata["memory_context_item_count"] == result.context.item_count
    assert metadata["memory_retrieval_allowed"] is True
    assert metadata["memory_query_text"] == "memory gateway"


def test_memory_cognition_bridge_snapshot_and_reset() -> None:
    gateway = make_gateway()
    seed_memory(gateway)
    bridge = MemoryCognitionBridge(gateway=gateway)

    bridge.build_context_from_text("memory gateway")
    snapshot = bridge.snapshot()

    assert snapshot.build_count == 1
    assert snapshot.last_context_id is not None
    assert snapshot.last_query_text == "memory gateway"

    bridge.reset()
    reset_snapshot = bridge.snapshot()

    assert reset_snapshot.build_count == 0
    assert reset_snapshot.built_count == 0
    assert reset_snapshot.last_context_id is None
    assert reset_snapshot.last_query_text is None