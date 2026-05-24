from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from jarvis.memory import (
    InMemoryMemoryStore,
    InMemoryMemoryStoreConfig,
    MemoryImportance,
    MemoryKind,
    MemoryPolicyClassification,
    MemoryQuery,
    MemoryRecord,
    MemoryRetention,
    MemoryScope,
    MemorySensitivity,
    MemorySource,
    MemoryStore,
    MemoryWriteRequest,
)


def make_write_request(
    *,
    text: str = "User is building a local JARVIS memory runtime.",
    kind: MemoryKind = MemoryKind.PROJECT,
    importance: MemoryImportance = MemoryImportance.NORMAL,
    sensitivity: MemorySensitivity = MemorySensitivity.PRIVATE,
    tags: tuple[str, ...] = ("jarvis",),
) -> MemoryWriteRequest:
    return MemoryWriteRequest(
        kind=kind,
        text=text,
        importance=importance,
        sensitivity=sensitivity,
        tags=tags,
    )


def test_in_memory_store_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        InMemoryMemoryStoreConfig(name=" ").validate()

    with pytest.raises(ValueError):
        InMemoryMemoryStoreConfig(max_records=0).validate()


def test_in_memory_store_satisfies_protocol() -> None:
    store: MemoryStore = InMemoryMemoryStore()

    record = store.write(make_write_request())
    fetched = store.get(record.memory_id)

    assert store.name == "in_memory_memory_store"
    assert fetched == record


def test_in_memory_store_writes_memory_request() -> None:
    store = InMemoryMemoryStore()

    record = store.write(
        make_write_request(
            text="User prefers direct engineering answers.",
            kind=MemoryKind.PREFERENCE,
            importance=MemoryImportance.HIGH,
        )
    )
    snapshot = store.snapshot()

    assert record.kind == MemoryKind.PREFERENCE
    assert record.importance == MemoryImportance.HIGH
    assert record.metadata["write_request_id"]
    assert snapshot.record_count == 1
    assert snapshot.write_count == 1
    assert snapshot.last_memory_id == record.memory_id


def test_in_memory_store_puts_existing_record() -> None:
    store = InMemoryMemoryStore()
    record = MemoryRecord(
        kind=MemoryKind.SEMANTIC,
        text="Memory records are storage independent.",
        source=MemorySource.SYSTEM,
    )

    stored = store.put(record)

    assert stored == record
    assert store.get(record.memory_id) == record


def test_in_memory_store_get_rejects_empty_id() -> None:
    store = InMemoryMemoryStore()

    assert store.get(" ") is None


def test_in_memory_store_retrieves_by_text() -> None:
    store = InMemoryMemoryStore()

    store.write(make_write_request(text="JARVIS memory runtime is active."))
    store.write(make_write_request(text="Weather is clear today."))

    result = store.retrieve(MemoryQuery(text="memory runtime"))

    assert result.result_count == 1
    assert result.records[0].text == "JARVIS memory runtime is active."


def test_in_memory_store_retrieves_by_kind_scope_and_tags() -> None:
    store = InMemoryMemoryStore()

    store.write(
        MemoryWriteRequest(
            kind=MemoryKind.USER_PROFILE,
            scope=MemoryScope.USER,
            text="User studies engineering.",
            tags=("profile", "education"),
        )
    )
    store.write(
        MemoryWriteRequest(
            kind=MemoryKind.PROJECT,
            scope=MemoryScope.PROJECT,
            text="Project memory runtime.",
            tags=("project",),
        )
    )

    result = store.retrieve(
        MemoryQuery(
            kinds=(MemoryKind.USER_PROFILE,),
            scopes=(MemoryScope.USER,),
            tags=("profile",),
        )
    )

    assert result.result_count == 1
    assert result.records[0].kind == MemoryKind.USER_PROFILE


def test_in_memory_store_filters_expired_records() -> None:
    store = InMemoryMemoryStore()
    expired = MemoryRecord(
        kind=MemoryKind.EPISODIC,
        text="Expired memory.",
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    active = MemoryRecord(
        kind=MemoryKind.EPISODIC,
        text="Active memory.",
        expires_at=datetime.now(UTC) + timedelta(seconds=60),
    )

    store.put(expired)
    store.put(active)

    result = store.retrieve(MemoryQuery())

    assert result.result_count == 1
    assert result.records[0].text == "Active memory."


def test_in_memory_store_can_include_expired_records() -> None:
    store = InMemoryMemoryStore()
    expired = MemoryRecord(
        kind=MemoryKind.EPISODIC,
        text="Expired memory.",
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )

    store.put(expired)

    result = store.retrieve(MemoryQuery(include_expired=True))

    assert result.result_count == 1
    assert result.records[0] == expired


def test_in_memory_store_filters_sensitive_by_default() -> None:
    store = InMemoryMemoryStore()

    store.write(
        make_write_request(
            text="Sensitive memory.",
            sensitivity=MemorySensitivity.SENSITIVE,
        )
    )
    store.write(make_write_request(text="Private memory."))

    result = store.retrieve(MemoryQuery())

    assert result.result_count == 1
    assert result.records[0].text == "Private memory."


def test_in_memory_store_can_include_sensitive_records() -> None:
    store = InMemoryMemoryStore()

    store.write(
        make_write_request(
            text="Sensitive memory.",
            sensitivity=MemorySensitivity.SENSITIVE,
        )
    )

    result = store.retrieve(MemoryQuery(include_sensitive=True))

    assert result.result_count == 1
    assert result.records[0].sensitivity == MemorySensitivity.SENSITIVE
    assert result.results[0].policy_classification == (
        MemoryPolicyClassification.RESTRICTED
    )


def test_in_memory_store_filters_by_min_confidence() -> None:
    store = InMemoryMemoryStore()

    store.write(
        MemoryWriteRequest(
            kind=MemoryKind.SEMANTIC,
            text="Low confidence memory.",
            confidence=0.2,
        )
    )
    store.write(
        MemoryWriteRequest(
            kind=MemoryKind.SEMANTIC,
            text="High confidence memory.",
            confidence=0.9,
        )
    )

    result = store.retrieve(MemoryQuery(min_confidence=0.8))

    assert result.result_count == 1
    assert result.records[0].text == "High confidence memory."


def test_in_memory_store_ranks_by_importance_and_relevance() -> None:
    store = InMemoryMemoryStore()

    low = store.write(
        make_write_request(
            text="JARVIS memory low priority.",
            importance=MemoryImportance.LOW,
        )
    )
    high = store.write(
        make_write_request(
            text="JARVIS memory high priority.",
            importance=MemoryImportance.HIGH,
        )
    )

    result = store.retrieve(MemoryQuery(text="JARVIS memory"))

    assert result.result_count == 2
    assert result.records[0].memory_id == high.memory_id
    assert result.records[1].memory_id == low.memory_id


def test_in_memory_store_retrieval_explanations_are_auditable() -> None:
    store = InMemoryMemoryStore()
    store.write(
        make_write_request(
            text="JARVIS memory retrieval must explain itself.",
            importance=MemoryImportance.HIGH,
        )
    )

    result = store.retrieve(MemoryQuery(text="memory retrieval"))

    explanation = result.results[0].explanation

    assert explanation.source == MemorySource.CONVERSATION
    assert "matched query terms" in explanation.reason
    assert explanation.confidence > 0
    assert explanation.retrieved_at.tzinfo is not None
    assert explanation.policy_classification == MemoryPolicyClassification.ALLOWED


def test_in_memory_store_bounds_capacity() -> None:
    store = InMemoryMemoryStore(
        config=InMemoryMemoryStoreConfig(max_records=2)
    )

    store.write(
        make_write_request(
            text="low memory",
            importance=MemoryImportance.LOW,
        )
    )
    store.write(
        make_write_request(
            text="normal memory",
            importance=MemoryImportance.NORMAL,
        )
    )
    store.write(
        make_write_request(
            text="critical memory",
            importance=MemoryImportance.CRITICAL,
        )
    )

    records = store.records()

    assert len(records) == 2
    assert {record.text for record in records} == {
        "normal memory",
        "critical memory",
    }


def test_in_memory_store_delete() -> None:
    store = InMemoryMemoryStore()
    record = store.write(make_write_request())

    assert store.delete(record.memory_id) is True
    assert store.delete(record.memory_id) is False
    assert store.get(record.memory_id) is None
    assert store.snapshot().delete_count == 1


def test_in_memory_store_delete_rejects_empty_id() -> None:
    store = InMemoryMemoryStore()

    assert store.delete(" ") is False


def test_in_memory_store_clear() -> None:
    store = InMemoryMemoryStore()

    store.write(make_write_request())
    store.clear()

    snapshot = store.snapshot()

    assert snapshot.record_count == 0
    assert snapshot.clear_count == 1
    assert snapshot.last_memory_id is None


def test_in_memory_store_snapshot_counts_expired_records() -> None:
    store = InMemoryMemoryStore()
    expired = MemoryRecord(
        kind=MemoryKind.EPISODIC,
        text="Expired memory.",
        retention=MemoryRetention.TEMPORARY,
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )

    store.put(expired)

    snapshot = store.snapshot()

    assert snapshot.record_count == 1
    assert snapshot.active_record_count == 0
    assert snapshot.expired_record_count == 1


def test_in_memory_store_retrieve_updates_snapshot() -> None:
    store = InMemoryMemoryStore()

    store.write(make_write_request())
    query = MemoryQuery(text="JARVIS")
    store.retrieve(query)

    snapshot = store.snapshot()

    assert snapshot.retrieve_count == 1
    assert snapshot.last_query_id == query.query_id