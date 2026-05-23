from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from jarvis.cognition import (
    CognitionCancelWorker,
    CognitionCancelWorkerConfig,
    CognitionRequest,
    CognitionRunState,
    CognitionWorker,
    FakeCognitionAdapter,
    PresenceCognitionInterruptBridgeConfig,
    PresenceCognitionInterruptBridgeWorker,
)
from jarvis.runtime.shared.enums import EventCategory, EventType


@dataclass(frozen=True, slots=True)
class FakeRuntimeEvent:
    event_type: EventType
    category: EventCategory
    source: str
    payload: dict[str, Any]
    event_id: str = "event-1"
    correlation_id: str = "correlation-1"
    causation_id: str | None = None

    def child(
        self,
        event_type: EventType,
        category: EventCategory,
        source: str,
        payload: dict[str, Any] | None = None,
        priority: Any | None = None,
    ) -> FakeRuntimeEvent:
        return FakeRuntimeEvent(
            event_type=event_type,
            category=category,
            source=source,
            payload=payload or {},
            correlation_id=self.correlation_id,
            causation_id=self.event_id,
        )


@dataclass(slots=True)
class FakeEventBus:
    subscriptions: list[tuple[EventType, str]] = field(default_factory=list)
    published: list[FakeRuntimeEvent] = field(default_factory=list)

    def subscribe(
        self,
        event_type: EventType,
        callback: Callable[[Any], Any],
        subscriber_name: str,
    ) -> str:
        self.subscriptions.append((event_type, subscriber_name))
        return f"subscription-{subscriber_name}"

    def publish_sync(self, event: FakeRuntimeEvent) -> None:
        self.published.append(event)


def make_interrupt_event(
    *,
    payload: dict[str, Any] | None = None,
    event_type: EventType = EventType.PRESENCE_INTERRUPT_REQUESTED,
) -> FakeRuntimeEvent:
    return FakeRuntimeEvent(
        event_type=event_type,
        category=EventCategory.PRESENCE,
        source="presence_interruption_worker",
        payload=payload
        or {
            "interrupt_id": "interrupt-1",
            "cognition_request_id": "request-1",
            "reason": "user started speaking",
            "turn_id": "turn-1",
            "source": "barge_in",
        },
    )


def make_cancel_event(
    *,
    payload: dict[str, Any] | None = None,
    event_type: EventType = EventType.COGNITION_CANCEL_REQUESTED,
) -> FakeRuntimeEvent:
    return FakeRuntimeEvent(
        event_type=event_type,
        category=EventCategory.COGNITION,
        source="presence_cognition_interrupt_bridge_worker",
        payload=payload
        or {
            "request_id": "request-1",
            "reason": "user started speaking",
        },
    )


def make_request() -> CognitionRequest:
    return CognitionRequest(
        request_id="request-1",
        text="explain the cognition runtime",
    )


def test_presence_cognition_interrupt_bridge_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        PresenceCognitionInterruptBridgeConfig(name=" ").validate()

    with pytest.raises(ValueError):
        PresenceCognitionInterruptBridgeConfig(source=" ").validate()

    with pytest.raises(ValueError):
        PresenceCognitionInterruptBridgeConfig(default_reason=" ").validate()


def test_presence_cognition_interrupt_bridge_start_subscribes_once() -> None:
    bus = FakeEventBus()
    bridge = PresenceCognitionInterruptBridgeWorker(event_bus=bus)

    bridge.on_start()
    bridge.on_start()

    snapshot = bridge.snapshot()

    assert snapshot.started is True
    assert snapshot.subscribed is True
    assert bus.subscriptions == [
        (
            EventType.PRESENCE_INTERRUPT_REQUESTED,
            "presence_cognition_interrupt_bridge_worker",
        )
    ]


def test_presence_cognition_interrupt_bridge_stop_is_idempotent() -> None:
    bridge = PresenceCognitionInterruptBridgeWorker(event_bus=FakeEventBus())

    bridge.on_start()
    bridge.on_stop()
    bridge.on_stop()

    snapshot = bridge.snapshot()

    assert snapshot.started is False
    assert snapshot.subscribed is False


def test_presence_cognition_interrupt_bridge_rejects_when_not_started() -> None:
    bus = FakeEventBus()
    bridge = PresenceCognitionInterruptBridgeWorker(event_bus=bus)

    result = bridge.process_presence_interrupt_requested(make_interrupt_event())
    snapshot = bridge.snapshot()

    assert result.rejected is True
    assert result.reason == "bridge is not started"
    assert snapshot.rejected_count == 1
    assert bus.published == []


def test_presence_cognition_interrupt_bridge_publishes_cancel_requested() -> None:
    bus = FakeEventBus()
    bridge = PresenceCognitionInterruptBridgeWorker(event_bus=bus)

    bridge.on_start()

    result = bridge.process_presence_interrupt_requested(make_interrupt_event())
    snapshot = bridge.snapshot()

    assert result.accepted is True
    assert result.request_id == "request-1"
    assert result.reason == "user started speaking"
    assert len(bus.published) == 1

    published = bus.published[0]

    assert published.event_type == EventType.COGNITION_CANCEL_REQUESTED
    assert published.category == EventCategory.COGNITION
    assert published.source == "presence_cognition_interrupt_bridge_worker"
    assert published.correlation_id == "correlation-1"
    assert published.causation_id == "event-1"
    assert published.payload["request_id"] == "request-1"
    assert published.payload["reason"] == "user started speaking"
    assert published.payload["metadata"]["interrupt_id"] == "interrupt-1"

    assert snapshot.processed_count == 1
    assert snapshot.published_count == 1
    assert snapshot.last_cancel_request_id == "request-1"
    assert snapshot.last_reason == "user started speaking"


def test_presence_cognition_interrupt_bridge_uses_default_reason() -> None:
    bus = FakeEventBus()
    bridge = PresenceCognitionInterruptBridgeWorker(event_bus=bus)

    bridge.on_start()

    result = bridge.process_presence_interrupt_requested(
        make_interrupt_event(
            payload={
                "cognition_request_id": "request-2",
            }
        )
    )

    assert result.accepted is True
    assert result.reason == "user interrupted"
    assert bus.published[0].payload["reason"] == "user interrupted"


def test_presence_cognition_interrupt_bridge_allows_missing_request_id() -> None:
    bus = FakeEventBus()
    bridge = PresenceCognitionInterruptBridgeWorker(event_bus=bus)

    bridge.on_start()

    result = bridge.process_presence_interrupt_requested(
        make_interrupt_event(payload={"reason": "barge in"})
    )

    assert result.accepted is True
    assert result.request_id is None
    assert bus.published[0].payload["request_id"] is None


def test_presence_cognition_interrupt_bridge_rejects_wrong_event_type() -> None:
    bus = FakeEventBus()
    bridge = PresenceCognitionInterruptBridgeWorker(event_bus=bus)

    bridge.on_start()

    result = bridge.process_presence_interrupt_requested(
        make_interrupt_event(event_type=EventType.COGNITION_CANCEL_REQUESTED)
    )

    assert result.rejected is True
    assert result.reason == "unsupported event type"
    assert bus.published == []


def test_cognition_cancel_worker_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        CognitionCancelWorkerConfig(name=" ").validate()


def test_cognition_cancel_worker_start_subscribes_once() -> None:
    bus = FakeEventBus()
    cognition_worker = CognitionWorker(adapter=FakeCognitionAdapter())
    cancel_worker = CognitionCancelWorker(
        event_bus=bus,
        cognition_worker=cognition_worker,
    )

    cancel_worker.on_start()
    cancel_worker.on_start()

    snapshot = cancel_worker.snapshot()

    assert snapshot.started is True
    assert snapshot.subscribed is True
    assert bus.subscriptions == [
        (
            EventType.COGNITION_CANCEL_REQUESTED,
            "cognition_cancel_worker",
        )
    ]


def test_cognition_cancel_worker_rejects_when_not_started() -> None:
    bus = FakeEventBus()
    cognition_worker = CognitionWorker(adapter=FakeCognitionAdapter())
    cancel_worker = CognitionCancelWorker(
        event_bus=bus,
        cognition_worker=cognition_worker,
    )

    result = cancel_worker.process_cognition_cancel_requested(make_cancel_event())
    snapshot = cancel_worker.snapshot()

    assert result.rejected is True
    assert result.reason == "cancel worker is not started"
    assert snapshot.rejected_count == 1


def test_cognition_cancel_worker_applies_cancel_to_active_request() -> None:
    bus = FakeEventBus()
    cognition_worker = CognitionWorker(adapter=FakeCognitionAdapter())
    cancel_worker = CognitionCancelWorker(
        event_bus=bus,
        cognition_worker=cognition_worker,
    )
    request = make_request()

    cognition_worker.on_start()
    assert cognition_worker.state_store.start_request(request).accepted is True

    cancel_worker.on_start()
    result = cancel_worker.process_cognition_cancel_requested(make_cancel_event())

    cancel_snapshot = cancel_worker.snapshot()
    cognition_snapshot = cognition_worker.snapshot()

    assert result.accepted is True
    assert result.request_id == "request-1"
    assert result.reason == "user started speaking"
    assert cancel_snapshot.accepted_count == 1
    assert cognition_snapshot.state.state == CognitionRunState.CANCELLING
    assert cognition_snapshot.state.cancelling is True
    assert cognition_snapshot.adapter.cancelled_count == 1


def test_cognition_cancel_worker_rejects_without_active_request() -> None:
    bus = FakeEventBus()
    cognition_worker = CognitionWorker(adapter=FakeCognitionAdapter())
    cancel_worker = CognitionCancelWorker(
        event_bus=bus,
        cognition_worker=cognition_worker,
    )

    cognition_worker.on_start()
    cancel_worker.on_start()

    result = cancel_worker.process_cognition_cancel_requested(make_cancel_event())
    snapshot = cancel_worker.snapshot()

    assert result.rejected is True
    assert snapshot.rejected_count == 1
    assert snapshot.last_error == "cognition worker rejected cancellation"


def test_cognition_cancel_worker_rejects_wrong_event_type() -> None:
    bus = FakeEventBus()
    cognition_worker = CognitionWorker(adapter=FakeCognitionAdapter())
    cancel_worker = CognitionCancelWorker(
        event_bus=bus,
        cognition_worker=cognition_worker,
    )

    cancel_worker.on_start()

    result = cancel_worker.process_cognition_cancel_requested(
        make_cancel_event(event_type=EventType.PRESENCE_INTERRUPT_REQUESTED)
    )

    assert result.rejected is True
    assert result.reason == "unsupported event type"


def test_cognition_interrupt_full_bridge_to_cancel_path() -> None:
    bus = FakeEventBus()
    cognition_worker = CognitionWorker(adapter=FakeCognitionAdapter())
    interrupt_bridge = PresenceCognitionInterruptBridgeWorker(event_bus=bus)
    cancel_worker = CognitionCancelWorker(
        event_bus=bus,
        cognition_worker=cognition_worker,
    )
    request = make_request()

    cognition_worker.on_start()
    assert cognition_worker.state_store.start_request(request).accepted is True

    interrupt_bridge.on_start()
    cancel_worker.on_start()

    bridge_result = interrupt_bridge.process_presence_interrupt_requested(
        make_interrupt_event()
    )

    assert bridge_result.accepted is True
    assert len(bus.published) == 1

    cancel_result = cancel_worker.process_cognition_cancel_requested(
        bus.published[0]
    )

    assert cancel_result.accepted is True
    assert cognition_worker.snapshot().state.state == CognitionRunState.CANCELLING