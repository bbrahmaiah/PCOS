from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from jarvis.cognition import (
    CognitionContext,
    CognitionContextItem,
    CognitionRequest,
    InMemoryShortTermMemoryConfig,
    InMemoryShortTermMemoryStore,
    ShortTermMemoryItem,
    ShortTermMemoryKind,
    ShortTermMemoryPriority,
    ShortTermMemoryQuery,
    ShortTermMemoryStore,
)


def make_request(
    *,
    text: str = "What did we build for cognition?",
) -> CognitionRequest:
    return CognitionRequest(
        request_id="request-1",
        text=text,
        turn_id="turn-1",
        context=CognitionContext(
            session_id="session-1",
            items=(
                CognitionContextItem(
                    kind="conversation_user",
                    text="We discussed JARVIS.",
                ),
            ),
        ),
    )


def test_short_term_memory_item_rejects_invalid_values() -> None:
    with pytest.raises(ValidationError):
        ShortTermMemoryItem(text=" ")

    with pytest.raises(ValidationError):
        ShortTermMemoryItem(text="hello", source=" ")

    with pytest.raises(ValidationError):
        ShortTermMemoryItem(text="hello", confidence=1.1)


def test_short_term_memory_item_expiration() -> None:
    expired = ShortTermMemoryItem(
        text="old memory",
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    active = ShortTermMemoryItem(
        text="fresh memory",
        expires_at=datetime.now(UTC) + timedelta(seconds=60),
    )

    assert expired.expired() is True
    assert active.expired() is False


def test_short_term_memory_query_cleans_empty_query_text() -> None:
    query = ShortTermMemoryQuery(query_text=" ")

    assert query.query_text is None


def test_short_term_memory_query_rejects_invalid_bounds() -> None:
    with pytest.raises(ValidationError):
        ShortTermMemoryQuery(max_items=0)

    with pytest.raises(ValidationError):
        ShortTermMemoryQuery(min_confidence=-0.1)


def test_in_memory_short_term_memory_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        InMemoryShortTermMemoryConfig(name=" ").validate()

    with pytest.raises(ValueError):
        InMemoryShortTermMemoryConfig(max_items=0).validate()

    with pytest.raises(ValueError):
        InMemoryShortTermMemoryConfig(max_context_items=0).validate()

    with pytest.raises(ValueError):
        InMemoryShortTermMemoryConfig(max_context_item_chars=0).validate()


def test_in_memory_short_term_memory_store_protocol() -> None:
    store: ShortTermMemoryStore = InMemoryShortTermMemoryStore()

    item = store.remember(
        ShortTermMemoryItem(
            text="User is building JARVIS cognition.",
        )
    )
    result = store.retrieve(ShortTermMemoryQuery(query_text="JARVIS"))

    assert store.name == "in_memory_short_term_memory"
    assert item.memory_id
    assert result.item_count == 1
    assert store.forget(item.memory_id) is True


def test_short_term_memory_remember_text() -> None:
    store = InMemoryShortTermMemoryStore()

    item = store.remember_text(
        "User prefers elite engineering steps.",
        kind=ShortTermMemoryKind.USER_PREFERENCE,
        priority=ShortTermMemoryPriority.HIGH,
    )
    snapshot = store.snapshot()

    assert item.kind == ShortTermMemoryKind.USER_PREFERENCE
    assert item.priority == ShortTermMemoryPriority.HIGH
    assert snapshot.item_count == 1
    assert snapshot.remembered_count == 1
    assert snapshot.last_memory_id == item.memory_id


def test_short_term_memory_retrieve_by_query_text() -> None:
    store = InMemoryShortTermMemoryStore()

    store.remember_text("JARVIS cognition engine is complete.")
    store.remember_text("Weather is sunny today.")

    result = store.retrieve(ShortTermMemoryQuery(query_text="cognition"))

    assert result.item_count == 1
    assert result.items[0].text == "JARVIS cognition engine is complete."


def test_short_term_memory_retrieve_by_kind() -> None:
    store = InMemoryShortTermMemoryStore()

    store.remember_text(
        "User wants concise spoken answers.",
        kind=ShortTermMemoryKind.USER_PREFERENCE,
    )
    store.remember_text(
        "Adapter failed yesterday.",
        kind=ShortTermMemoryKind.ERROR_CONTEXT,
    )

    result = store.retrieve(
        ShortTermMemoryQuery(
            kinds=(ShortTermMemoryKind.ERROR_CONTEXT,),
        )
    )

    assert result.item_count == 1
    assert result.items[0].kind == ShortTermMemoryKind.ERROR_CONTEXT


def test_short_term_memory_retrieve_filters_confidence() -> None:
    store = InMemoryShortTermMemoryStore()

    store.remember_text("low confidence", confidence=0.2)
    store.remember_text("high confidence", confidence=0.9)

    result = store.retrieve(ShortTermMemoryQuery(min_confidence=0.8))

    assert result.item_count == 1
    assert result.items[0].text == "high confidence"


def test_short_term_memory_retrieve_excludes_expired_by_default() -> None:
    store = InMemoryShortTermMemoryStore()

    store.remember_text(
        "expired memory",
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    store.remember_text("active memory")

    result = store.retrieve(ShortTermMemoryQuery())

    assert result.item_count == 1
    assert result.items[0].text == "active memory"


def test_short_term_memory_retrieve_can_include_expired() -> None:
    store = InMemoryShortTermMemoryStore()

    store.remember_text(
        "expired memory",
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )

    result = store.retrieve(ShortTermMemoryQuery(include_expired=True))

    assert result.item_count == 1
    assert result.items[0].text == "expired memory"


def test_short_term_memory_ranks_priority_before_confidence() -> None:
    store = InMemoryShortTermMemoryStore()

    low = store.remember_text(
        "cognition low priority",
        priority=ShortTermMemoryPriority.LOW,
        confidence=1.0,
    )
    high = store.remember_text(
        "cognition high priority",
        priority=ShortTermMemoryPriority.HIGH,
        confidence=0.5,
    )

    result = store.retrieve(ShortTermMemoryQuery(query_text="cognition"))

    assert result.items[0].memory_id == high.memory_id
    assert result.items[1].memory_id == low.memory_id


def test_short_term_memory_bounds_capacity() -> None:
    store = InMemoryShortTermMemoryStore(
        config=InMemoryShortTermMemoryConfig(max_items=2)
    )

    store.remember_text("one", priority=ShortTermMemoryPriority.LOW)
    store.remember_text("two", priority=ShortTermMemoryPriority.NORMAL)
    store.remember_text("three", priority=ShortTermMemoryPriority.HIGH)

    items = store.items()

    assert len(items) == 2
    assert {item.text for item in items} == {"two", "three"}


def test_short_term_memory_builds_context() -> None:
    store = InMemoryShortTermMemoryStore(
        config=InMemoryShortTermMemoryConfig(max_context_items=2)
    )
    request = make_request(text="What did we build for cognition?")

    store.remember_text(
        "Cognition engine is complete.",
        kind=ShortTermMemoryKind.PROJECT_CONTEXT,
    )
    store.remember_text(
        "Streaming token pipeline is complete.",
        kind=ShortTermMemoryKind.PROJECT_CONTEXT,
    )

    context = store.build_context(request=request)

    assert context.session_id == "session-1"
    assert context.turn_id == "turn-1"
    assert context.item_count == 2
    assert context.items[0].kind.startswith("short_term_memory_")
    assert context.metadata["memory_item_count"] == 2


def test_short_term_memory_enriches_request_preserving_existing_context() -> None:
    store = InMemoryShortTermMemoryStore()
    request = make_request(text="What did we build for cognition?")

    store.remember_text("Cognition engine is complete.")
    enriched = store.enrich_request(request)

    assert enriched.request_id == request.request_id
    assert enriched.context.item_count == 2
    assert enriched.context.items[0].text == "We discussed JARVIS."
    assert enriched.context.items[1].text == "Cognition engine is complete."
    assert enriched.metadata["short_term_memory_enriched"] is True


def test_short_term_memory_forget() -> None:
    store = InMemoryShortTermMemoryStore()
    item = store.remember_text("temporary memory")

    assert store.forget(item.memory_id) is True
    assert store.forget(item.memory_id) is False
    assert store.snapshot().forgotten_count == 1


def test_short_term_memory_forget_expired() -> None:
    store = InMemoryShortTermMemoryStore()

    store.remember_text(
        "expired memory",
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    store.remember_text("active memory")

    forgotten_count = store.forget_expired()

    assert forgotten_count == 1
    assert store.snapshot().item_count == 1


def test_short_term_memory_clear() -> None:
    store = InMemoryShortTermMemoryStore()

    store.remember_text("temporary memory")
    store.clear()

    snapshot = store.snapshot()

    assert snapshot.item_count == 0
    assert snapshot.cleared_count == 1


def test_short_term_memory_context_item_text_is_bounded() -> None:
    store = InMemoryShortTermMemoryStore(
        config=InMemoryShortTermMemoryConfig(max_context_item_chars=10)
    )

    store.remember_text("This is a long memory item.")

    context = store.build_context()

    assert context.item_count == 1
    assert len(context.items[0].text) <= 10
    assert context.items[0].text.endswith("...")


def test_short_term_memory_enum_values_are_stable() -> None:
    assert ShortTermMemoryKind.USER_PREFERENCE.value == "user_preference"
    assert ShortTermMemoryKind.PROJECT_CONTEXT.value == "project_context"
    assert ShortTermMemoryPriority.LOW.value == "low"
    assert ShortTermMemoryPriority.CRITICAL.value == "critical"