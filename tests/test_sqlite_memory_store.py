from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from jarvis.memory import (
    GovernedMemoryGateway,
    MemoryImportance,
    MemoryKind,
    MemoryPolicyClassification,
    MemoryQuery,
    MemoryRecord,
    MemoryRetention,
    MemoryScope,
    MemorySensitivity,
    MemoryStore,
    MemoryWriteRequest,
    SQLiteMemoryStore,
    SQLiteMemoryStoreConfig,
)


def make_store(path: Path) -> SQLiteMemoryStore:
    return SQLiteMemoryStore(
        config=SQLiteMemoryStoreConfig(path=path),
    )


def make_write_request(
    *,
    text: str = "SQLite memory store persists JARVIS memory records.",
    kind: MemoryKind = MemoryKind.PROJECT,
    importance: MemoryImportance = MemoryImportance.NORMAL,
    sensitivity: MemorySensitivity = MemorySensitivity.PRIVATE,
    tags: tuple[str, ...] = ("jarvis", "sqlite"),
) -> MemoryWriteRequest:
    return MemoryWriteRequest(
        kind=kind,
        text=text,
        importance=importance,
        sensitivity=sensitivity,
        tags=tags,
    )


def test_sqlite_store_config_rejects_invalid_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        SQLiteMemoryStoreConfig(path=tmp_path / "memory.db", name=" ").validate()

    with pytest.raises(ValueError):
        SQLiteMemoryStoreConfig(
            path=tmp_path / "memory.db",
            timeout_seconds=0,
        ).validate()


def test_sqlite_store_satisfies_memory_store_protocol(tmp_path: Path) -> None:
    store: MemoryStore = make_store(tmp_path / "memory.db")

    record = store.write(make_write_request())

    assert store.name == "sqlite_memory_store"
    assert store.get(record.memory_id) == record


def test_sqlite_store_writes_and_reads_memory(tmp_path: Path) -> None:
    store = make_store(tmp_path / "memory.db")

    record = store.write(
        make_write_request(
            text="Persistent memory survives store recreation.",
            importance=MemoryImportance.HIGH,
        )
    )
    recreated = make_store(tmp_path / "memory.db")

    loaded = recreated.get(record.memory_id)

    assert loaded == record
    assert loaded is not None
    assert loaded.importance == MemoryImportance.HIGH
    assert recreated.snapshot().record_count == 1


def test_sqlite_store_put_updates_existing_record(tmp_path: Path) -> None:
    store = make_store(tmp_path / "memory.db")
    record = MemoryRecord(
        kind=MemoryKind.SEMANTIC,
        text="Original memory text.",
    )
    updated = record.model_copy(update={"text": "Updated memory text."})

    store.put(record)
    store.put(updated)

    loaded = store.get(record.memory_id)

    assert loaded == updated
    assert store.snapshot().record_count == 1
    assert store.snapshot().write_count == 2


def test_sqlite_store_get_rejects_empty_id(tmp_path: Path) -> None:
    store = make_store(tmp_path / "memory.db")

    assert store.get(" ") is None


def test_sqlite_store_retrieves_by_text(tmp_path: Path) -> None:
    store = make_store(tmp_path / "memory.db")

    store.write(make_write_request(text="JARVIS persistent memory is active."))
    store.write(make_write_request(text="Weather is clear today."))

    result = store.retrieve(MemoryQuery(text="persistent memory"))

    assert result.result_count == 1
    assert result.records[0].text == "JARVIS persistent memory is active."
    assert result.results[0].explanation.reason
    assert result.results[0].explanation.retrieved_at.tzinfo is not None


def test_sqlite_store_retrieves_by_kind_scope_and_tags(tmp_path: Path) -> None:
    store = make_store(tmp_path / "memory.db")

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


def test_sqlite_store_filters_expired_records(tmp_path: Path) -> None:
    store = make_store(tmp_path / "memory.db")
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


def test_sqlite_store_can_include_expired_records(tmp_path: Path) -> None:
    store = make_store(tmp_path / "memory.db")
    expired = MemoryRecord(
        kind=MemoryKind.EPISODIC,
        text="Expired memory.",
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )

    store.put(expired)

    result = store.retrieve(MemoryQuery(include_expired=True))

    assert result.result_count == 1
    assert result.records[0] == expired


def test_sqlite_store_filters_sensitive_by_default(tmp_path: Path) -> None:
    store = make_store(tmp_path / "memory.db")

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


def test_sqlite_store_can_include_sensitive_records(tmp_path: Path) -> None:
    store = make_store(tmp_path / "memory.db")

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


def test_sqlite_store_ranks_results(tmp_path: Path) -> None:
    store = make_store(tmp_path / "memory.db")

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


def test_sqlite_store_delete(tmp_path: Path) -> None:
    store = make_store(tmp_path / "memory.db")
    record = store.write(make_write_request())

    assert store.delete(record.memory_id) is True
    assert store.delete(record.memory_id) is False
    assert store.get(record.memory_id) is None
    assert store.snapshot().delete_count == 1


def test_sqlite_store_delete_rejects_empty_id(tmp_path: Path) -> None:
    store = make_store(tmp_path / "memory.db")

    assert store.delete(" ") is False


def test_sqlite_store_clear(tmp_path: Path) -> None:
    store = make_store(tmp_path / "memory.db")

    store.write(make_write_request())
    store.clear()

    snapshot = store.snapshot()

    assert snapshot.record_count == 0
    assert snapshot.clear_count == 1
    assert snapshot.last_memory_id is None


def test_sqlite_store_snapshot_counts_expired_records(tmp_path: Path) -> None:
    store = make_store(tmp_path / "memory.db")
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


def test_sqlite_store_works_behind_gateway(tmp_path: Path) -> None:
    store = make_store(tmp_path / "memory.db")
    gateway = GovernedMemoryGateway(store=store)

    write = gateway.remember(make_write_request(text="Gateway uses SQLite store."))
    retrieval = gateway.retrieve(MemoryQuery(text="SQLite store"))

    assert write.allowed is True
    assert retrieval.allowed is True
    assert retrieval.result_count == 1
    assert retrieval.records[0].text == "Gateway uses SQLite store."


def test_sqlite_store_close_is_safe(tmp_path: Path) -> None:
    store = make_store(tmp_path / "memory.db")

    store.close()

    assert store.snapshot().record_count == 0