from __future__ import annotations

from datetime import timedelta

import pytest
from pydantic import ValidationError

from jarvis.runtime.events.event_models import RuntimeEvent, utc_now
from jarvis.runtime.events.priorities import priority_for_event
from jarvis.runtime.events.subscriptions import EventSubscription
from jarvis.runtime.shared.enums import EventCategory, EventPriority, EventType


def test_runtime_event_has_ids_and_timestamp() -> None:
    event = RuntimeEvent(
        event_type=EventType.RUNTIME_STARTED,
        category=EventCategory.RUNTIME,
        source="test",
    )

    assert event.event_id
    assert event.correlation_id
    assert event.created_at.tzinfo is not None


def test_event_assigns_default_priority_before_validation() -> None:
    event = RuntimeEvent(
        event_type=EventType.INTERRUPT_REQUESTED,
        category=EventCategory.PRESENCE,
        source="test",
    )

    assert event.priority == EventPriority.CRITICAL


def test_event_allows_explicit_priority_override() -> None:
    event = RuntimeEvent(
        event_type=EventType.INTERRUPT_REQUESTED,
        category=EventCategory.PRESENCE,
        source="test",
        priority=EventPriority.NORMAL,
    )

    assert event.priority == EventPriority.NORMAL


def test_priority_for_normal_event() -> None:
    assert priority_for_event(EventType.STATE_UPDATED) == EventPriority.NORMAL


def test_event_rejects_empty_source() -> None:
    with pytest.raises(ValidationError):
        RuntimeEvent(
            event_type=EventType.RUNTIME_STARTED,
            category=EventCategory.RUNTIME,
            source="",
        )


def test_event_rejects_empty_ids() -> None:
    with pytest.raises(ValidationError):
        RuntimeEvent(
            event_id="",
            event_type=EventType.RUNTIME_STARTED,
            category=EventCategory.RUNTIME,
            source="test",
        )

    with pytest.raises(ValidationError):
        RuntimeEvent(
            correlation_id="",
            event_type=EventType.RUNTIME_STARTED,
            category=EventCategory.RUNTIME,
            source="test",
        )


def test_event_rejects_empty_causation_id_when_provided() -> None:
    with pytest.raises(ValidationError):
        RuntimeEvent(
            causation_id="",
            event_type=EventType.RUNTIME_STARTED,
            category=EventCategory.RUNTIME,
            source="test",
        )


def test_event_rejects_empty_schema_version() -> None:
    with pytest.raises(ValidationError):
        RuntimeEvent(
            event_type=EventType.RUNTIME_STARTED,
            category=EventCategory.RUNTIME,
            source="test",
            schema_version="",
        )


def test_event_rejects_past_deadline() -> None:
    with pytest.raises(ValidationError):
        RuntimeEvent(
            event_type=EventType.RUNTIME_STARTED,
            category=EventCategory.RUNTIME,
            source="test",
            deadline_at=utc_now() - timedelta(seconds=1),
        )


def test_child_event_preserves_correlation_and_sets_causation() -> None:
    parent = RuntimeEvent(
        event_type=EventType.USER_SPOKE,
        category=EventCategory.PRESENCE,
        source="presence_worker",
        payload={"text": "open chrome"},
    )

    child = parent.child(
        event_type=EventType.INTENT_DETECTED,
        category=EventCategory.ROUTER,
        source="router_worker",
        payload={"intent": "open_app"},
    )

    assert child.correlation_id == parent.correlation_id
    assert child.causation_id == parent.event_id
    assert child.payload["intent"] == "open_app"


def test_child_event_can_override_cancellable() -> None:
    parent = RuntimeEvent(
        event_type=EventType.USER_SPOKE,
        category=EventCategory.PRESENCE,
        source="presence_worker",
        cancellable=True,
    )

    child = parent.child(
        event_type=EventType.ACTION_STARTED,
        category=EventCategory.ACTION,
        source="action_worker",
        cancellable=False,
    )

    assert child.cancellable is False


def test_event_expired_detection_false_for_future_deadline() -> None:
    event = RuntimeEvent(
        event_type=EventType.RUNTIME_TICK,
        category=EventCategory.RUNTIME,
        source="kernel",
        deadline_at=utc_now() + timedelta(seconds=1),
    )

    assert event.is_expired is False


def test_subscription_trims_subscriber_name() -> None:
    def callback(event: RuntimeEvent) -> None:
        _ = event

    subscription = EventSubscription(
        event_type=EventType.RUNTIME_STARTED,
        subscriber_name="  kernel_worker  ",
        callback=callback,
    )

    assert subscription.subscriber_name == "kernel_worker"


def test_subscription_rejects_empty_subscriber_name() -> None:
    def callback(event: RuntimeEvent) -> None:
        _ = event

    with pytest.raises(ValueError):
        EventSubscription(
            event_type=EventType.RUNTIME_STARTED,
            subscriber_name="   ",
            callback=callback,
        )


def test_subscription_rejects_non_callable_callback() -> None:
    with pytest.raises(TypeError):
        EventSubscription(
            event_type=EventType.RUNTIME_STARTED,
            subscriber_name="kernel_worker",
            callback="not_callable",  # type: ignore[arg-type]
        )