from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.orchestration import (
    ContextSnapshot,
    ContextSnapshotReason,
    ContextSnapshotRuntime,
    ContextSnapshotRuntimeConfig,
    ContextSnapshotStatus,
    ContextWriteDisposition,
    ContextWriteKind,
    ConversationTurnContext,
    PendingContextWrite,
    SnapshotPolicy,
    new_task_id,
    new_turn_id,
)


def turn_context(
    *,
    turn_id: str | None = None,
    user_text: str = "hello jarvis",
) -> ConversationTurnContext:
    return ConversationTurnContext(
        turn_id=turn_id or new_turn_id(),
        session_id="session-1",
        user_text=user_text,
        topic="testing",
        active_task_ids=(new_task_id(),),
        metadata={"stable": True},
    )


def write(
    *,
    turn_id: str,
    key: str = "memory_result",
    value: object = "queued memory",
    kind: ContextWriteKind = ContextWriteKind.MEMORY_RESULT,
) -> PendingContextWrite:
    return PendingContextWrite(
        turn_id=turn_id,
        kind=kind,
        key=key,
        value=value,
        source="test",
    )


def test_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        ContextSnapshotRuntimeConfig(name=" ").validate()


def test_turn_context_requires_user_text() -> None:
    with pytest.raises(ValidationError):
        ConversationTurnContext(user_text=" ")


def test_turn_context_validates_task_ids() -> None:
    with pytest.raises(ValidationError):
        ConversationTurnContext(
            user_text="hello",
            active_task_ids=("bad-task-id",),
        )


def test_snapshot_requires_snapshot_prefix() -> None:
    with pytest.raises(ValidationError):
        ContextSnapshot(
            snapshot_id="bad",
            turn_context=turn_context(),
        )


def test_pending_write_requires_write_prefix() -> None:
    with pytest.raises(ValidationError):
        PendingContextWrite(
            write_id="bad",
            turn_id=new_turn_id(),
            kind=ContextWriteKind.MEMORY_RESULT,
            key="memory",
            value="value",
            source="test",
        )


def test_begin_turn_creates_active_snapshot() -> None:
    runtime = ContextSnapshotRuntime()
    context = turn_context()

    result = runtime.begin_turn(context)

    assert result.success is True
    assert result.reason == ContextSnapshotReason.SNAPSHOT_CREATED
    assert result.snapshot is not None
    assert result.snapshot.active is True
    assert runtime.active_snapshot() == result.snapshot


def test_begin_turn_rejects_when_snapshot_already_active() -> None:
    runtime = ContextSnapshotRuntime()
    context = turn_context()

    first = runtime.begin_turn(context)
    second = runtime.begin_turn(turn_context())

    assert first.success is True
    assert second.success is False
    assert second.reason == ContextSnapshotReason.SNAPSHOT_ALREADY_ACTIVE


def test_active_snapshot_is_immutable_model() -> None:
    runtime = ContextSnapshotRuntime()
    context = turn_context()
    result = runtime.begin_turn(context)

    assert result.snapshot is not None

    with pytest.raises(ValidationError):
        result.snapshot.turn_context.user_text = "mutated"


def test_queue_background_write_for_active_turn() -> None:
    runtime = ContextSnapshotRuntime()
    context = turn_context()
    runtime.begin_turn(context)

    result = runtime.queue_background_write(write(turn_id=context.turn_id))

    assert result.success is True
    assert result.disposition == ContextWriteDisposition.QUEUED
    assert result.reason == ContextSnapshotReason.WRITE_QUEUED_FOR_NEXT_TURN
    assert len(runtime.queued_writes()) == 1


def test_queue_background_write_rejects_without_active_snapshot() -> None:
    runtime = ContextSnapshotRuntime()
    context = turn_context()

    result = runtime.queue_background_write(write(turn_id=context.turn_id))

    assert result.success is False
    assert result.reason == ContextSnapshotReason.NO_ACTIVE_SNAPSHOT


def test_queue_background_write_rejects_turn_mismatch() -> None:
    runtime = ContextSnapshotRuntime()
    context = turn_context()
    runtime.begin_turn(context)

    result = runtime.queue_background_write(write(turn_id=new_turn_id()))

    assert result.success is False
    assert result.reason == ContextSnapshotReason.WRITE_REJECTED_TURN_MISMATCH


def test_queue_background_write_rejects_when_queue_full() -> None:
    runtime = ContextSnapshotRuntime(
        policy=SnapshotPolicy(max_queued_writes=1)
    )
    context = turn_context()
    runtime.begin_turn(context)

    first = runtime.queue_background_write(
        write(turn_id=context.turn_id, key="first")
    )
    second = runtime.queue_background_write(
        write(turn_id=context.turn_id, key="second")
    )

    assert first.success is True
    assert second.success is False
    assert second.reason == ContextSnapshotReason.WRITE_REJECTED_QUEUE_FULL


def test_queue_background_write_rejects_when_policy_disabled() -> None:
    runtime = ContextSnapshotRuntime(
        policy=SnapshotPolicy(queue_background_writes=False)
    )
    context = turn_context()
    runtime.begin_turn(context)

    result = runtime.queue_background_write(write(turn_id=context.turn_id))

    assert result.success is False
    assert (
        result.reason
        == ContextSnapshotReason.WRITE_REJECTED_ACTIVE_SNAPSHOT_IMMUTABLE
    )


def test_reject_active_mutation() -> None:
    runtime = ContextSnapshotRuntime()
    context = turn_context()
    runtime.begin_turn(context)

    result = runtime.reject_active_mutation(write(turn_id=context.turn_id))

    assert result.success is False
    assert result.disposition == ContextWriteDisposition.REJECTED
    assert (
        result.reason
        == ContextSnapshotReason.WRITE_REJECTED_ACTIVE_SNAPSHOT_IMMUTABLE
    )


def test_seal_turn_closes_active_snapshot() -> None:
    runtime = ContextSnapshotRuntime()
    context = turn_context()
    runtime.begin_turn(context)

    result = runtime.seal_turn()

    assert result.success is True
    assert result.snapshot is not None
    assert result.snapshot.status == ContextSnapshotStatus.SEALED
    assert runtime.active_snapshot() is None


def test_seal_turn_rejects_without_active_snapshot() -> None:
    runtime = ContextSnapshotRuntime()

    result = runtime.seal_turn()

    assert result.success is False
    assert result.reason == ContextSnapshotReason.NO_ACTIVE_SNAPSHOT


def test_expire_turn_closes_active_snapshot() -> None:
    runtime = ContextSnapshotRuntime()
    context = turn_context()
    runtime.begin_turn(context)

    result = runtime.expire_turn()

    assert result.success is True
    assert result.snapshot is not None
    assert result.snapshot.status == ContextSnapshotStatus.EXPIRED
    assert runtime.active_snapshot() is None


def test_apply_queued_writes_to_next_context() -> None:
    runtime = ContextSnapshotRuntime()
    context = turn_context()
    runtime.begin_turn(context)
    runtime.queue_background_write(
        write(turn_id=context.turn_id, key="memory", value="result")
    )
    runtime.seal_turn()

    next_context = turn_context(user_text="next turn")
    updated, writes = runtime.apply_queued_writes(next_context)

    assert len(writes) == 1
    assert updated.metadata["memory"] == "result"
    assert runtime.queued_writes() == ()


def test_apply_queued_writes_does_not_mutate_original_context() -> None:
    runtime = ContextSnapshotRuntime()
    context = turn_context()
    runtime.begin_turn(context)
    runtime.queue_background_write(
        write(turn_id=context.turn_id, key="memory", value="result")
    )
    runtime.seal_turn()

    next_context = turn_context(user_text="next turn")
    updated, _ = runtime.apply_queued_writes(next_context)

    assert "memory" not in next_context.metadata
    assert updated.metadata["memory"] == "result"


def test_snapshot_reports_runtime_state() -> None:
    runtime = ContextSnapshotRuntime()
    context = turn_context()
    runtime.begin_turn(context)
    runtime.queue_background_write(write(turn_id=context.turn_id))

    snapshot = runtime.snapshot()

    assert snapshot.has_active_snapshot is True
    assert snapshot.active_turn_id == context.turn_id
    assert snapshot.queued_write_count == 1
    assert snapshot.snapshot_count == 1
    assert snapshot.queued_count == 1


def test_reset_clears_runtime_state() -> None:
    runtime = ContextSnapshotRuntime()
    context = turn_context()
    runtime.begin_turn(context)
    runtime.queue_background_write(write(turn_id=context.turn_id))

    runtime.reset()

    snapshot = runtime.snapshot()

    assert snapshot.has_active_snapshot is False
    assert snapshot.queued_write_count == 0
    assert snapshot.snapshot_count == 0


def test_enum_values_are_stable() -> None:
    assert ContextSnapshotStatus.ACTIVE.value == "active"
    assert ContextWriteKind.MEMORY_RESULT.value == "memory_result"
    assert ContextWriteDisposition.QUEUED.value == "queued"
    assert ContextSnapshotReason.SNAPSHOT_CREATED.value == "snapshot_created"