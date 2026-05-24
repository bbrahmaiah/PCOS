from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.memory import (
    EpisodicMemoryActor,
    EpisodicMemoryEvent,
    EpisodicMemoryEventKind,
    EpisodicMemoryQuery,
    EpisodicMemoryRuntime,
    EpisodicMemoryRuntimeConfig,
    GovernedMemoryGateway,
    InMemoryMemoryStore,
    MemoryImportance,
    MemoryKind,
    MemoryRetention,
    MemorySensitivity,
)


def make_runtime() -> EpisodicMemoryRuntime:
    return EpisodicMemoryRuntime(
        gateway=GovernedMemoryGateway(store=InMemoryMemoryStore()),
    )


def make_event(
    *,
    event_id: str = "event-1",
    text: str = "User completed Phase 4 lifecycle policy.",
    kind: EpisodicMemoryEventKind = EpisodicMemoryEventKind.MILESTONE,
    actor: EpisodicMemoryActor = EpisodicMemoryActor.USER,
    sensitivity: MemorySensitivity = MemorySensitivity.PRIVATE,
) -> EpisodicMemoryEvent:
    return EpisodicMemoryEvent(
        event_id=event_id,
        kind=kind,
        actor=actor,
        text=text,
        sensitivity=sensitivity,
        tags=("phase4", "memory"),
    )


def test_episodic_memory_runtime_config_rejects_invalid_name() -> None:
    with pytest.raises(ValueError):
        EpisodicMemoryRuntimeConfig(name=" ").validate()


def test_episodic_event_rejects_empty_fields() -> None:
    with pytest.raises(ValidationError):
        EpisodicMemoryEvent(
            event_id=" ",
            kind=EpisodicMemoryEventKind.MILESTONE,
            actor=EpisodicMemoryActor.USER,
            text="valid",
        )

    with pytest.raises(ValidationError):
        EpisodicMemoryEvent(
            event_id="event-1",
            kind=EpisodicMemoryEventKind.MILESTONE,
            actor=EpisodicMemoryActor.USER,
            text=" ",
        )


def test_episodic_event_cleans_tags() -> None:
    event = EpisodicMemoryEvent(
        event_id="event-1",
        kind=EpisodicMemoryEventKind.MILESTONE,
        actor=EpisodicMemoryActor.USER,
        text="Phase milestone.",
        tags=(" Phase4 ", "phase4", " Memory "),
    )

    assert event.tags == ("phase4", "memory")


def test_episodic_event_to_write_request() -> None:
    event = make_event()
    request = event.to_write_request()

    assert request.kind == MemoryKind.EPISODIC
    assert request.text == event.text
    assert request.importance == event.importance
    assert request.sensitivity == event.sensitivity
    assert "episodic" in request.tags
    assert EpisodicMemoryEventKind.MILESTONE.value in request.tags
    assert EpisodicMemoryActor.USER.value in request.tags
    assert request.metadata["event_id"] == "event-1"


def test_episodic_query_to_memory_query() -> None:
    query = EpisodicMemoryQuery(
        text="lifecycle policy",
        event_kinds=(EpisodicMemoryEventKind.MILESTONE,),
        actors=(EpisodicMemoryActor.USER,),
        tags=("phase4",),
        max_results=5,
    )
    memory_query = query.to_memory_query()

    assert memory_query.text == "lifecycle policy"
    assert memory_query.kinds == (MemoryKind.EPISODIC,)
    assert memory_query.max_results == 5
    assert "episodic" in memory_query.tags
    assert "milestone" in memory_query.tags
    assert "user" in memory_query.tags
    assert "phase4" in memory_query.tags


def test_episodic_runtime_captures_event() -> None:
    runtime = make_runtime()
    result = runtime.capture(make_event())
    snapshot = runtime.snapshot()

    assert result.allowed is True
    assert result.record is not None
    assert result.record.kind == MemoryKind.EPISODIC
    assert snapshot.captured_count == 1
    assert snapshot.captured_allowed_count == 1
    assert snapshot.last_event_id == "event-1"


def test_episodic_runtime_blocks_sensitive_event_by_gateway_policy() -> None:
    runtime = make_runtime()
    result = runtime.capture(
        make_event(
            text="Sensitive event.",
            sensitivity=MemorySensitivity.SENSITIVE,
        )
    )
    snapshot = runtime.snapshot()

    assert result.allowed is False
    assert result.blocked is True
    assert snapshot.captured_count == 1
    assert snapshot.captured_blocked_count == 1
    assert snapshot.last_error == result.reason


def test_episodic_runtime_capture_text() -> None:
    runtime = make_runtime()

    result = runtime.capture_text(
        "Assistant fixed a memory bug.",
        event_id="event-2",
        kind=EpisodicMemoryEventKind.DEBUGGING,
        actor=EpisodicMemoryActor.ASSISTANT,
        importance=MemoryImportance.HIGH,
        retention=MemoryRetention.PERSISTENT,
        tags=("debug",),
    )

    assert result.allowed is True
    assert result.record is not None
    assert "debugging" in result.record.tags
    assert "assistant" in result.record.tags
    assert result.record.importance == MemoryImportance.HIGH


def test_episodic_runtime_retrieves_events() -> None:
    runtime = make_runtime()

    runtime.capture(make_event())
    runtime.capture_text(
        "Unrelated system event.",
        event_id="event-2",
        kind=EpisodicMemoryEventKind.SYSTEM_EVENT,
        actor=EpisodicMemoryActor.SYSTEM,
    )

    result = runtime.retrieve(
        EpisodicMemoryQuery(
            text="lifecycle policy",
            event_kinds=(EpisodicMemoryEventKind.MILESTONE,),
            actors=(EpisodicMemoryActor.USER,),
            tags=("phase4",),
        )
    )

    assert result.allowed is True
    assert result.result_count == 1
    assert result.records[0].text == "User completed Phase 4 lifecycle policy."
    assert result.results[0].explanation.reason


def test_episodic_runtime_snapshot_and_reset() -> None:
    runtime = make_runtime()

    runtime.capture(make_event())
    runtime.retrieve(EpisodicMemoryQuery(text="lifecycle"))

    snapshot = runtime.snapshot()

    assert snapshot.captured_count == 1
    assert snapshot.retrieved_count == 1

    runtime.reset()
    reset_snapshot = runtime.snapshot()

    assert reset_snapshot.captured_count == 0
    assert reset_snapshot.retrieved_count == 0
    assert reset_snapshot.last_event_id is None


def test_episodic_enum_values_are_stable() -> None:
    assert EpisodicMemoryEventKind.USER_MESSAGE.value == "user_message"
    assert EpisodicMemoryEventKind.MILESTONE.value == "milestone"
    assert EpisodicMemoryEventKind.DEBUGGING.value == "debugging"
    assert EpisodicMemoryActor.USER.value == "user"
    assert EpisodicMemoryActor.ASSISTANT.value == "assistant"