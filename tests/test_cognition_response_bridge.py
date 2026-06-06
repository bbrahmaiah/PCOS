from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from jarvis.cognition import (
    CognitionDialogueBridgeConfig,
    CognitionDialogueBridgeWorker,
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


def make_completed_event(
    *,
    payload: dict[str, Any] | None = None,
) -> FakeRuntimeEvent:
    return FakeRuntimeEvent(
        event_type=EventType.COGNITION_COMPLETED,
        category=EventCategory.COGNITION,
        source="cognition_worker",
        payload=payload
        or {
            "request_id": "cognition-request-1",
            "response_id": "response-1",
            "text": "Yes sir. I am listening.",
            "kind": "spoken_reply",
            "confidence": 1.0,
        },
    )


def make_failed_event(
    *,
    payload: dict[str, Any] | None = None,
) -> FakeRuntimeEvent:
    return FakeRuntimeEvent(
        event_type=EventType.COGNITION_FAILED,
        category=EventCategory.COGNITION,
        source="cognition_worker",
        payload=payload
        or {
            "request_id": "cognition-request-1",
            "failure_id": "failure-1",
            "message": "adapter failed",
            "kind": "adapter_error",
        },
    )


def test_cognition_dialogue_bridge_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        CognitionDialogueBridgeConfig(name=" ").validate()

    with pytest.raises(ValueError):
        CognitionDialogueBridgeConfig(source=" ").validate()

    with pytest.raises(ValueError):
        CognitionDialogueBridgeConfig(failure_fallback_text=" ").validate()

    with pytest.raises(ValueError):
        CognitionDialogueBridgeConfig(max_response_chars=0).validate()


def test_cognition_dialogue_bridge_start_subscribes_once() -> None:
    bus = FakeEventBus()
    worker = CognitionDialogueBridgeWorker(event_bus=bus)

    worker.on_start()
    worker.on_start()

    snapshot = worker.snapshot()

    assert snapshot.started is True
    assert snapshot.subscribed is True
    assert bus.subscriptions == [
        (
            EventType.COGNITION_COMPLETED,
            "cognition_dialogue_bridge_worker",
        ),
        (
            EventType.COGNITION_FAILED,
            "cognition_dialogue_bridge_worker",
        ),
    ]


def test_cognition_dialogue_bridge_stop_is_idempotent() -> None:
    worker = CognitionDialogueBridgeWorker(event_bus=FakeEventBus())

    worker.on_start()
    worker.on_stop()
    worker.on_stop()

    snapshot = worker.snapshot()

    assert snapshot.started is False
    assert snapshot.subscribed is False


def test_cognition_dialogue_bridge_rejects_when_not_started() -> None:
    bus = FakeEventBus()
    worker = CognitionDialogueBridgeWorker(event_bus=bus)

    result = worker.process_cognition_completed(make_completed_event())
    snapshot = worker.snapshot()

    assert result.rejected is True
    assert result.reason == "bridge is not started"
    assert snapshot.rejected_count == 1
    assert bus.published == []


def test_cognition_dialogue_bridge_publishes_dialogue_response_ready() -> None:
    bus = FakeEventBus()
    worker = CognitionDialogueBridgeWorker(event_bus=bus)

    worker.on_start()

    result = worker.process_cognition_completed(make_completed_event())
    snapshot = worker.snapshot()

    assert result.accepted is True
    assert result.response_text == "Yes sir. I am listening."
    assert len(bus.published) == 1

    published = bus.published[0]

    assert published.event_type == EventType.DIALOGUE_RESPONSE_READY
    assert published.category == EventCategory.DIALOGUE
    assert published.source == "cognition_dialogue_bridge_worker"
    assert published.correlation_id == "correlation-1"
    assert published.causation_id == "event-1"
    assert published.payload["text"] == "Yes sir. I am listening."
    assert published.payload["source"] == "cognition"
    assert published.payload["fallback"] is False
    assert published.payload["cognition_request_id"] == "cognition-request-1"
    assert published.payload["cognition_response_id"] == "response-1"
    assert published.payload["response_id"] == "dialogue-response-response-1"

    assert snapshot.completed_processed_count == 1
    assert snapshot.published_count == 1
    assert snapshot.rejected_count == 0
    assert snapshot.last_cognition_request_id == "cognition-request-1"
    assert snapshot.last_cognition_response_id == "response-1"
    assert snapshot.last_dialogue_response_id == "dialogue-response-response-1"


def test_cognition_dialogue_bridge_accepts_nested_response_payload() -> None:
    bus = FakeEventBus()
    worker = CognitionDialogueBridgeWorker(event_bus=bus)

    worker.on_start()

    result = worker.process_cognition_completed(
        make_completed_event(
            payload={
                "response": {
                    "request_id": "request-2",
                    "response_id": "response-2",
                    "text": "Nested response ready.",
                    "confidence": 0.9,
                    "kind": "spoken_reply",
                }
            }
        )
    )

    assert result.accepted is True
    assert bus.published[0].payload["text"] == "Nested response ready."
    assert bus.published[0].payload["cognition_request_id"] == "request-2"
    assert bus.published[0].payload["cognition_response_id"] == "response-2"


def test_cognition_dialogue_bridge_rejects_missing_response_text() -> None:
    bus = FakeEventBus()
    worker = CognitionDialogueBridgeWorker(event_bus=bus)

    worker.on_start()

    result = worker.process_cognition_completed(
        make_completed_event(payload={"request_id": "request-3"})
    )
    snapshot = worker.snapshot()

    assert result.rejected is True
    assert result.reason == "cognition payload does not contain response text."
    assert snapshot.rejected_count == 1
    assert bus.published == []


def test_cognition_dialogue_bridge_rejects_wrong_completed_event_type() -> None:
    bus = FakeEventBus()
    worker = CognitionDialogueBridgeWorker(event_bus=bus)

    worker.on_start()

    result = worker.process_cognition_completed(
        FakeRuntimeEvent(
            event_type=EventType.COGNITION_FAILED,
            category=EventCategory.COGNITION,
            source="cognition_worker",
            payload={},
        )
    )

    assert result.rejected is True
    assert result.reason == "unsupported event type"
    assert bus.published == []


def test_cognition_dialogue_bridge_publishes_failure_fallback() -> None:
    bus = FakeEventBus()
    worker = CognitionDialogueBridgeWorker(event_bus=bus)

    worker.on_start()

    result = worker.process_cognition_failed(make_failed_event())
    snapshot = worker.snapshot()

    assert result.accepted is True
    assert result.response_text == "cognition_response_bridge_failure"
    assert len(bus.published) == 1

    published = bus.published[0]

    assert published.event_type == EventType.DIALOGUE_RESPONSE_READY
    assert published.payload["fallback"] is True
    assert published.payload["text"] == "cognition_response_bridge_failure"
    assert published.payload["cognition_request_id"] == "cognition-request-1"
    assert published.payload["cognition_response_id"] == "failure-1"
    assert published.payload["response_id"] == "dialogue-fallback-failure-1"

    assert snapshot.failed_processed_count == 1
    assert snapshot.published_count == 1


def test_cognition_dialogue_bridge_accepts_nested_failure_payload() -> None:
    bus = FakeEventBus()
    worker = CognitionDialogueBridgeWorker(event_bus=bus)

    worker.on_start()

    result = worker.process_cognition_failed(
        make_failed_event(
            payload={
                "failure": {
                    "request_id": "request-4",
                    "failure_id": "failure-4",
                    "message": "failed",
                    "kind": "adapter_error",
                }
            }
        )
    )

    assert result.accepted is True
    assert bus.published[0].payload["cognition_request_id"] == "request-4"
    assert bus.published[0].payload["cognition_response_id"] == "failure-4"


def test_cognition_dialogue_bridge_can_disable_failure_fallback() -> None:
    bus = FakeEventBus()
    worker = CognitionDialogueBridgeWorker(
        event_bus=bus,
        config=CognitionDialogueBridgeConfig(
            publish_failure_fallback=False,
        ),
    )

    worker.on_start()

    result = worker.process_cognition_failed(make_failed_event())

    assert result.rejected is True
    assert result.reason == "failure fallback publishing is disabled"
    assert bus.published == []


def test_cognition_dialogue_bridge_rejects_wrong_failed_event_type() -> None:
    bus = FakeEventBus()
    worker = CognitionDialogueBridgeWorker(event_bus=bus)

    worker.on_start()

    result = worker.process_cognition_failed(make_completed_event())

    assert result.rejected is True
    assert result.reason == "unsupported event type"
    assert bus.published == []


def test_cognition_dialogue_bridge_truncates_long_response() -> None:
    bus = FakeEventBus()
    worker = CognitionDialogueBridgeWorker(
        event_bus=bus,
        config=CognitionDialogueBridgeConfig(max_response_chars=10),
    )

    worker.on_start()

    result = worker.process_cognition_completed(
        make_completed_event(
            payload={
                "request_id": "request-5",
                "response_id": "response-5",
                "text": "This response is too long.",
            }
        )
    )

    assert result.accepted is True
    assert bus.published[0].payload["text"] == "This respo"