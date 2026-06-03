from __future__ import annotations

import pytest

from jarvis.cognitive import (
    AttentionPriority,
    WorkingMemoryKind,
    WorkingMemoryOperation,
    WorkingMemoryRecallRequest,
    WorkingMemoryRetention,
    WorkingMemoryRuntime,
    WorkingMemoryRuntimeStatus,
    WorkingMemoryUpdateRequest,
    make_working_memory_entry,
)


def test_working_memory_runtime_rejects_invalid_max_items() -> None:
    with pytest.raises(ValueError):
        WorkingMemoryRuntime(max_items=0)


def test_working_memory_entry_rejects_empty_key() -> None:
    with pytest.raises(ValueError):
        make_working_memory_entry(
            kind=WorkingMemoryKind.OBJECTIVE,
            key=" ",
            value="Build Phase 9",
        )


def test_working_memory_update_adds_context_items() -> None:
    runtime = WorkingMemoryRuntime()
    entry = make_working_memory_entry(
        kind=WorkingMemoryKind.OBJECTIVE,
        key="current_objective",
        value="Build working memory runtime",
        importance=AttentionPriority.HIGH,
    )

    result = runtime.update(WorkingMemoryUpdateRequest(entries=(entry,)))

    assert result.status == WorkingMemoryRuntimeStatus.READY
    assert result.operation == WorkingMemoryOperation.UPSERT
    assert result.state.get("current_objective") is not None
    assert result.state.high_importance_items


def test_working_memory_recall_by_query() -> None:
    runtime = WorkingMemoryRuntime()
    runtime.update(
        WorkingMemoryUpdateRequest(
            entries=(
                make_working_memory_entry(
                    kind=WorkingMemoryKind.PROJECT,
                    key="project",
                    value="JARVIS OS",
                    importance=AttentionPriority.HIGH,
                ),
                make_working_memory_entry(
                    kind=WorkingMemoryKind.CONVERSATION,
                    key="topic",
                    value="Phase 9 working memory",
                    importance=AttentionPriority.NORMAL,
                ),
            )
        )
    )

    result = runtime.recall(
        WorkingMemoryRecallRequest(query="working memory")
    )

    assert result.status == WorkingMemoryRuntimeStatus.READY
    assert result.operation == WorkingMemoryOperation.RECALL
    assert len(result.items) == 1
    assert result.items[0].key == "topic"


def test_working_memory_recall_by_kind_and_importance() -> None:
    runtime = WorkingMemoryRuntime()
    runtime.update(
        WorkingMemoryUpdateRequest(
            entries=(
                make_working_memory_entry(
                    kind=WorkingMemoryKind.RISK,
                    key="battery",
                    value="Battery is low",
                    importance=AttentionPriority.CRITICAL,
                ),
                make_working_memory_entry(
                    kind=WorkingMemoryKind.TASK,
                    key="background_task",
                    value="Build watch running",
                    importance=AttentionPriority.NORMAL,
                ),
            )
        )
    )

    result = runtime.recall(
        WorkingMemoryRecallRequest(
            kinds=(WorkingMemoryKind.RISK,),
            minimum_importance=AttentionPriority.HIGH,
        )
    )

    assert len(result.items) == 1
    assert result.items[0].key == "battery"


def test_working_memory_upsert_prefers_higher_importance() -> None:
    runtime = WorkingMemoryRuntime()
    runtime.update(
        WorkingMemoryUpdateRequest(
            entries=(
                make_working_memory_entry(
                    kind=WorkingMemoryKind.OBJECTIVE,
                    key="objective",
                    value="Old objective",
                    importance=AttentionPriority.NORMAL,
                ),
            )
        )
    )
    runtime.update(
        WorkingMemoryUpdateRequest(
            entries=(
                make_working_memory_entry(
                    kind=WorkingMemoryKind.OBJECTIVE,
                    key="objective",
                    value="Critical objective",
                    importance=AttentionPriority.CRITICAL,
                ),
            )
        )
    )

    item = runtime.state.get("objective")

    assert item is not None
    assert item.value == "Critical objective"
    assert item.importance == AttentionPriority.CRITICAL


def test_working_memory_remove_key() -> None:
    runtime = WorkingMemoryRuntime()
    runtime.update(
        WorkingMemoryUpdateRequest(
            entries=(
                make_working_memory_entry(
                    kind=WorkingMemoryKind.TASK,
                    key="task",
                    value="Run tests",
                ),
            )
        )
    )

    runtime.update(WorkingMemoryUpdateRequest(remove_keys=("task",)))

    assert runtime.state.get("task") is None


def test_working_memory_clear_resets_state() -> None:
    runtime = WorkingMemoryRuntime()
    runtime.update(
        WorkingMemoryUpdateRequest(
            entries=(
                make_working_memory_entry(
                    kind=WorkingMemoryKind.PROJECT,
                    key="project",
                    value="JARVIS",
                ),
            )
        )
    )

    result = runtime.update(WorkingMemoryUpdateRequest(clear=True))

    assert result.operation == WorkingMemoryOperation.CLEAR
    assert result.state.items == ()


def test_working_memory_compacts_to_max_items() -> None:
    runtime = WorkingMemoryRuntime(max_items=3)

    entries = tuple(
        make_working_memory_entry(
            kind=WorkingMemoryKind.TASK,
            key=f"task_{index}",
            value=f"Task {index}",
            importance=AttentionPriority.NORMAL,
        )
        for index in range(10)
    )
    runtime.update(WorkingMemoryUpdateRequest(entries=entries))
    result = runtime.compact()

    assert result.operation == WorkingMemoryOperation.COMPACT
    assert len(result.state.items) == 3


def test_working_memory_expired_items_are_removed_on_compact() -> None:
    runtime = WorkingMemoryRuntime()
    runtime.update(
        WorkingMemoryUpdateRequest(
            entries=(
                make_working_memory_entry(
                    kind=WorkingMemoryKind.ASSUMPTION,
                    key="temporary",
                    value="Short lived assumption",
                    retention=WorkingMemoryRetention.SHORT,
                    ttl_seconds=1,
                ),
            )
        )
    )

    item = runtime.state.get("temporary")

    assert item is not None

    expired_item = item.__class__(
        item_id=item.item_id,
        kind=item.kind,
        key=item.key,
        value=item.value,
        importance=item.importance,
        source=item.source,
        created_at=item.created_at,
        metadata={
            **item.metadata,
            "expires_at": "2000-01-01T00:00:00+00:00",
        },
    )
    runtime._state = runtime.state.__class__(items=(expired_item,))  # noqa: SLF001

    result = runtime.compact()

    assert result.state.items == ()


def test_working_memory_snapshot_tracks_counts() -> None:
    runtime = WorkingMemoryRuntime()
    runtime.update(
        WorkingMemoryUpdateRequest(
            entries=(
                make_working_memory_entry(
                    kind=WorkingMemoryKind.OBJECTIVE,
                    key="objective",
                    value="Build Phase 9",
                ),
            )
        )
    )
    runtime.recall(WorkingMemoryRecallRequest(query="Phase 9"))
    snapshot = runtime.snapshot()

    assert snapshot.status == WorkingMemoryRuntimeStatus.READY
    assert snapshot.item_count == 1
    assert snapshot.update_count == 1
    assert snapshot.recall_count == 1


def test_working_memory_request_validation() -> None:
    with pytest.raises(ValueError):
        WorkingMemoryUpdateRequest(max_items=0)

    with pytest.raises(ValueError):
        WorkingMemoryRecallRequest(limit=0)


def test_working_memory_enum_values_are_stable() -> None:
    assert WorkingMemoryRuntimeStatus.READY.value == "ready"
    assert WorkingMemoryOperation.UPSERT.value == "upsert"
    assert WorkingMemoryRetention.SESSION.value == "session"