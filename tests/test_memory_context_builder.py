from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.memory import (
    ExtractiveMemorySummarizer,
    GovernedMemoryGateway,
    InMemoryMemoryStore,
    MemoryContext,
    MemoryContextBuilder,
    MemoryContextBuilderConfig,
    MemoryContextBuilderProtocol,
    MemoryContextBuildRequest,
    MemoryContextBuildStatus,
    MemoryContextItem,
    MemoryContextItemKind,
    MemoryImportance,
    MemoryKind,
    MemoryPolicyClassification,
    MemoryQuery,
    MemoryRecord,
    MemorySensitivity,
    MemorySource,
    MemorySummaryRequest,
    MemoryWriteRequest,
)


def make_gateway() -> GovernedMemoryGateway:
    return GovernedMemoryGateway(store=InMemoryMemoryStore())


def make_record(
    *,
    text: str = "User is building a real-time personal cognition OS.",
    importance: MemoryImportance = MemoryImportance.HIGH,
    sensitivity: MemorySensitivity = MemorySensitivity.PRIVATE,
) -> MemoryRecord:
    return MemoryRecord(
        kind=MemoryKind.PROJECT,
        text=text,
        importance=importance,
        sensitivity=sensitivity,
        tags=("jarvis", "memory"),
    )


def test_memory_context_builder_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        MemoryContextBuilderConfig(name=" ").validate()

    with pytest.raises(ValueError):
        MemoryContextBuilderConfig(summary_score=-0.1).validate()

    with pytest.raises(ValueError):
        MemoryContextBuilderConfig(summary_score=1.1).validate()


def test_memory_context_item_requires_text_fields() -> None:
    with pytest.raises(ValidationError):
        MemoryContextItem(
            item_id=" ",
            item_kind=MemoryContextItemKind.RETRIEVED_MEMORY,
            text="valid",
            source=MemorySource.CONVERSATION,
            reason="valid",
            confidence=1.0,
            policy_classification=MemoryPolicyClassification.ALLOWED,
        )

    with pytest.raises(ValidationError):
        MemoryContextItem(
            item_kind=MemoryContextItemKind.RETRIEVED_MEMORY,
            text=" ",
            source=MemorySource.CONVERSATION,
            reason="valid",
            confidence=1.0,
            policy_classification=MemoryPolicyClassification.ALLOWED,
        )


def test_memory_context_as_text_block() -> None:
    item = MemoryContextItem(
        item_kind=MemoryContextItemKind.RETRIEVED_MEMORY,
        text="Memory gateway is the only entry point.",
        source=MemorySource.COGNITION,
        reason="matched gateway rule",
        confidence=0.9,
        policy_classification=MemoryPolicyClassification.ALLOWED,
    )
    context = MemoryContext(items=(item,), total_chars=item.char_count)

    block = context.as_text_block()

    assert "Memory gateway is the only entry point." in block
    assert "source=cognition" in block
    assert "confidence=0.90" in block
    assert context.item_count == 1
    assert context.empty is False


def test_memory_context_builder_satisfies_protocol() -> None:
    builder: MemoryContextBuilderProtocol = MemoryContextBuilder()

    context = builder.build(MemoryContextBuildRequest())

    assert context.empty is True


def test_context_builder_returns_empty_context_for_no_inputs() -> None:
    builder = MemoryContextBuilder()

    context = builder.build(MemoryContextBuildRequest())
    snapshot = builder.snapshot()

    assert context.status == MemoryContextBuildStatus.EMPTY
    assert context.item_count == 0
    assert snapshot.build_count == 1
    assert snapshot.empty_count == 1


def test_context_builder_builds_from_gateway_retrieval() -> None:
    gateway = make_gateway()
    gateway.remember(
        MemoryWriteRequest(
            kind=MemoryKind.PROJECT,
            text="JARVIS memory context builder prepares cognition context.",
            importance=MemoryImportance.HIGH,
            tags=("jarvis", "context"),
        )
    )
    retrieval = gateway.retrieve(MemoryQuery(text="memory context"))
    builder = MemoryContextBuilder()

    context = builder.build(
        MemoryContextBuildRequest(retrievals=(retrieval,))
    )

    assert context.status == MemoryContextBuildStatus.BUILT
    assert context.item_count == 1
    assert context.items[0].item_kind == MemoryContextItemKind.RETRIEVED_MEMORY
    assert context.items[0].memory_kind == MemoryKind.PROJECT
    assert context.items[0].source_memory_id is not None
    assert context.items[0].reason
    assert context.items[0].confidence > 0
    assert context.items[0].timestamp.tzinfo is not None
    assert context.items[0].policy_classification == (
        MemoryPolicyClassification.ALLOWED
    )


def test_context_builder_builds_from_summary() -> None:
    record = make_record()
    summarizer = ExtractiveMemorySummarizer()
    summary_result = summarizer.summarize(
        MemorySummaryRequest(records=(record,))
    )
    builder = MemoryContextBuilder()

    context = builder.build(
        MemoryContextBuildRequest(summaries=(summary_result,))
    )

    assert context.item_count == 1
    assert context.items[0].item_kind == MemoryContextItemKind.SUMMARY
    assert context.items[0].source_memory_id is not None
    assert context.items[0].source_memory_id.startswith("summary:")
    assert "source_memory_ids" in context.items[0].metadata


def test_context_builder_prefers_summary_when_configured() -> None:
    gateway = make_gateway()
    record = make_record(text="Retrieved memory item.")
    gateway.store.put(record)
    retrieval = gateway.retrieve(MemoryQuery(text="retrieved memory"))

    summary_result = ExtractiveMemorySummarizer().summarize(
        MemorySummaryRequest(
            records=(make_record(text="Summary memory item."),)
        )
    )

    builder = MemoryContextBuilder(
        config=MemoryContextBuilderConfig(prefer_summaries=True)
    )

    context = builder.build(
        MemoryContextBuildRequest(
            retrievals=(retrieval,),
            summaries=(summary_result,),
            max_items=2,
        )
    )

    assert context.item_count == 2
    assert context.items[0].item_kind == MemoryContextItemKind.SUMMARY


def test_context_builder_filters_restricted_by_default() -> None:
    gateway = GovernedMemoryGateway(
        store=InMemoryMemoryStore(),
    )
    gateway.store.put(
        make_record(
            text="Sensitive memory.",
            sensitivity=MemorySensitivity.SENSITIVE,
        )
    )
    retrieval = gateway.retrieve(MemoryQuery(include_sensitive=True))
    builder = MemoryContextBuilder()

    context = builder.build(
        MemoryContextBuildRequest(retrievals=(retrieval,))
    )

    assert context.empty is True


def test_context_builder_can_include_restricted() -> None:
    gateway = GovernedMemoryGateway(
        store=InMemoryMemoryStore(),
        config=None,
    )
    gateway.store.put(
        make_record(
            text="Sensitive memory.",
            sensitivity=MemorySensitivity.SENSITIVE,
        )
    )
    gateway = GovernedMemoryGateway(
        store=gateway.store,
        config=__import__("jarvis.memory").memory.MemoryGatewayConfig(
            allow_sensitive_retrieval=True
        ),
    )
    retrieval = gateway.retrieve(MemoryQuery(include_sensitive=True))
    builder = MemoryContextBuilder()

    context = builder.build(
        MemoryContextBuildRequest(
            retrievals=(retrieval,),
            include_restricted=True,
        )
    )

    assert context.item_count == 1
    assert context.items[0].policy_classification == (
        MemoryPolicyClassification.RESTRICTED
    )


def test_context_builder_deduplicates_memory_ids() -> None:
    gateway = make_gateway()
    record = make_record(text="Duplicate memory.")
    gateway.store.put(record)

    first = gateway.retrieve(MemoryQuery(text="duplicate"))
    second = gateway.retrieve(MemoryQuery(text="memory"))
    builder = MemoryContextBuilder()

    context = builder.build(
        MemoryContextBuildRequest(retrievals=(first, second))
    )

    assert context.item_count == 1
    assert context.items[0].source_memory_id == record.memory_id


def test_context_builder_applies_item_and_char_budget() -> None:
    gateway = make_gateway()

    gateway.store.put(make_record(text="A" * 120))
    gateway.store.put(make_record(text="B" * 120))

    retrieval = gateway.retrieve(MemoryQuery(max_results=2))
    builder = MemoryContextBuilder()

    context = builder.build(
        MemoryContextBuildRequest(
            retrievals=(retrieval,),
            max_items=1,
            max_chars=80,
        )
    )

    assert context.item_count == 1
    assert context.total_chars <= 80
    assert context.items[0].text.endswith("...")
    assert context.items[0].metadata["truncated"] is True


def test_context_builder_snapshot_and_reset() -> None:
    builder = MemoryContextBuilder()

    builder.build(MemoryContextBuildRequest())
    snapshot = builder.snapshot()

    assert snapshot.build_count == 1
    assert snapshot.empty_count == 1
    assert snapshot.last_context_id is not None

    builder.reset()
    reset_snapshot = builder.snapshot()

    assert reset_snapshot.build_count == 0
    assert reset_snapshot.empty_count == 0
    assert reset_snapshot.last_context_id is None


def test_memory_context_enum_values_are_stable() -> None:
    assert MemoryContextItemKind.RETRIEVED_MEMORY.value == "retrieved_memory"
    assert MemoryContextItemKind.SUMMARY.value == "summary"
    assert MemoryContextBuildStatus.BUILT.value == "built"
    assert MemoryContextBuildStatus.EMPTY.value == "empty"