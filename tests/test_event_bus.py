from __future__ import annotations

from datetime import timedelta
from time import sleep

import pytest

from jarvis.runtime.events import EventBus, RuntimeEvent, utc_now
from jarvis.runtime.shared.enums import EventCategory, EventPriority, EventType


def test_event_bus_publish_sync_delivers_event() -> None:
    bus = EventBus(name="test_bus")
    received: list[RuntimeEvent] = []

    bus.subscribe(
        EventType.USER_SPOKE,
        "test_subscriber",
        lambda event: received.append(event),
    )

    event = RuntimeEvent(
        event_type=EventType.USER_SPOKE,
        category=EventCategory.PRESENCE,
        source="test",
        payload={"text": "hello"},
    )

    delivered = bus.publish_sync(event)

    assert delivered == 1
    assert received == [event]


def test_event_bus_unsubscribe_removes_subscription() -> None:
    bus = EventBus(name="test_bus")
    received: list[RuntimeEvent] = []

    subscription = bus.subscribe(
        EventType.USER_SPOKE,
        "test_subscriber",
        lambda event: received.append(event),
    )

    removed = bus.unsubscribe(subscription.subscription_id)

    event = RuntimeEvent(
        event_type=EventType.USER_SPOKE,
        category=EventCategory.PRESENCE,
        source="test",
    )

    delivered = bus.publish_sync(event)

    assert removed is True
    assert delivered == 0
    assert received == []


def test_event_bus_unsubscribe_unknown_returns_false() -> None:
    bus = EventBus(name="test_bus")

    assert bus.unsubscribe("missing") is False


def test_event_bus_records_dead_letter_on_callback_failure() -> None:
    bus = EventBus(name="test_bus")

    def broken_callback(event: RuntimeEvent) -> None:
        _ = event
        raise RuntimeError("boom")

    bus.subscribe(EventType.USER_SPOKE, "broken_subscriber", broken_callback)

    event = RuntimeEvent(
        event_type=EventType.USER_SPOKE,
        category=EventCategory.PRESENCE,
        source="test",
    )

    delivered = bus.publish_sync(event)
    dead_letters = bus.dead_letters()

    assert delivered == 0
    assert len(dead_letters) == 1
    assert dead_letters[0].reason == "subscriber_callback_failed"
    assert dead_letters[0].subscriber_name == "broken_subscriber"


def test_event_bus_records_dead_letter_for_expired_event() -> None:
    bus = EventBus(name="test_bus")
    received: list[RuntimeEvent] = []

    bus.subscribe(
        EventType.USER_SPOKE,
        "test_subscriber",
        lambda event: received.append(event),
    )

    event = RuntimeEvent(
        event_type=EventType.USER_SPOKE,
        category=EventCategory.PRESENCE,
        source="test",
        deadline_at=utc_now() + timedelta(milliseconds=1),
    )

    sleep(0.01)

    delivered = bus.publish_sync(event)
    dead_letters = bus.dead_letters()

    assert delivered == 0
    assert received == []
    assert len(dead_letters) == 1
    assert dead_letters[0].reason == "event_expired"


def test_event_bus_background_dispatch_delivers_event() -> None:
    bus = EventBus(name="test_bus")
    received: list[RuntimeEvent] = []

    bus.subscribe(
        EventType.USER_SPOKE,
        "test_subscriber",
        lambda event: received.append(event),
    )

    bus.start()

    try:
        event = RuntimeEvent(
            event_type=EventType.USER_SPOKE,
            category=EventCategory.PRESENCE,
            source="test",
        )

        bus.publish(event)

        assert bus.drain(timeout_seconds=2.0) is True

    finally:
        bus.stop()

    assert received == [event]


def test_event_bus_dispatches_critical_before_low_priority() -> None:
    bus = EventBus(name="test_bus")
    received: list[EventType] = []

    bus.subscribe(
        EventType.RUNTIME_TICK,
        "low_subscriber",
        lambda event: received.append(event.event_type),
    )

    bus.subscribe(
        EventType.INTERRUPT_REQUESTED,
        "critical_subscriber",
        lambda event: received.append(event.event_type),
    )

    low_event = RuntimeEvent(
        event_type=EventType.RUNTIME_TICK,
        category=EventCategory.RUNTIME,
        source="test",
    )

    critical_event = RuntimeEvent(
        event_type=EventType.INTERRUPT_REQUESTED,
        category=EventCategory.PRESENCE,
        source="test",
    )

    bus.publish(low_event)
    bus.publish(critical_event)

    bus.start()

    try:
        assert bus.drain(timeout_seconds=2.0) is True
    finally:
        bus.stop()

    assert received == [
        EventType.INTERRUPT_REQUESTED,
        EventType.RUNTIME_TICK,
    ]


def test_event_bus_respects_explicit_priority_override() -> None:
    bus = EventBus(name="test_bus")
    received: list[EventType] = []

    bus.subscribe(
        EventType.RUNTIME_TICK,
        "tick_subscriber",
        lambda event: received.append(event.event_type),
    )

    bus.subscribe(
        EventType.STATE_UPDATED,
        "state_subscriber",
        lambda event: received.append(event.event_type),
    )

    normal_event = RuntimeEvent(
        event_type=EventType.STATE_UPDATED,
        category=EventCategory.STATE,
        source="test",
        priority=EventPriority.NORMAL,
    )

    promoted_event = RuntimeEvent(
        event_type=EventType.RUNTIME_TICK,
        category=EventCategory.RUNTIME,
        source="test",
        priority=EventPriority.CRITICAL,
    )

    bus.publish(normal_event)
    bus.publish(promoted_event)

    bus.start()

    try:
        assert bus.drain(timeout_seconds=2.0) is True
    finally:
        bus.stop()

    assert received == [
        EventType.RUNTIME_TICK,
        EventType.STATE_UPDATED,
    ]


def test_event_bus_snapshot_reports_state() -> None:
    bus = EventBus(name="test_bus")

    bus.subscribe(
        EventType.USER_SPOKE,
        "test_subscriber",
        lambda event: None,
    )

    event = RuntimeEvent(
        event_type=EventType.USER_SPOKE,
        category=EventCategory.PRESENCE,
        source="test",
    )

    bus.publish_sync(event)
    snapshot = bus.snapshot()

    assert snapshot.name == "test_bus"
    assert snapshot.subscription_count == 1
    assert snapshot.history_size == 1
    assert snapshot.published_count == 1
    assert snapshot.delivered_count == 1


def test_event_bus_rejects_invalid_publish_type() -> None:
    bus = EventBus(name="test_bus")

    with pytest.raises(TypeError):
        bus.publish("not an event")  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        bus.publish_sync("not an event")  # type: ignore[arg-type]


def test_event_bus_rejects_empty_subscription_id() -> None:
    bus = EventBus(name="test_bus")

    with pytest.raises(ValueError):
        bus.unsubscribe("   ")


def test_event_bus_clear_removes_history_and_dead_letters() -> None:
    bus = EventBus(name="test_bus")

    def broken_callback(event: RuntimeEvent) -> None:
        _ = event
        raise RuntimeError("boom")

    bus.subscribe(EventType.USER_SPOKE, "broken", broken_callback)

    event = RuntimeEvent(
        event_type=EventType.USER_SPOKE,
        category=EventCategory.PRESENCE,
        source="test",
    )

    bus.publish_sync(event)

    assert bus.history()
    assert bus.dead_letters()

    bus.clear()

    assert bus.history() == ()
    assert bus.dead_letters() == ()


def test_event_bus_start_stop_are_idempotent() -> None:
    bus = EventBus(name="test_bus")

    bus.start()
    bus.start()

    assert bus.snapshot().running is True

    bus.stop()
    bus.stop()

    assert bus.snapshot().running is False


def test_event_bus_constructor_validation() -> None:
    with pytest.raises(ValueError):
        EventBus(name="   ")

    with pytest.raises(ValueError):
        EventBus(history_limit=0)

    with pytest.raises(ValueError):
        EventBus(dead_letter_limit=0)


def test_event_bus_drain_rejects_invalid_timeout() -> None:
    bus = EventBus(name="test_bus")

    with pytest.raises(ValueError):
        bus.drain(timeout_seconds=0)


def test_event_bus_stop_rejects_invalid_timeout() -> None:
    bus = EventBus(name="test_bus")

    with pytest.raises(ValueError):
        bus.stop(timeout_seconds=0)