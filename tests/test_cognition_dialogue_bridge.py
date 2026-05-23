from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from jarvis.cognition import (
    DialogueCognitionBridgeConfig,
    DialogueCognitionBridgeWorker,
    SpokenResponseStyle,
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


def make_event(
    *,
    payload: dict[str, Any] | None = None,
    event_type: EventType = EventType.DIALOGUE_RESPONSE_REQUESTED,
) -> FakeRuntimeEvent:
    return FakeRuntimeEvent(
        event_type=event_type,
        category=EventCategory.DIALOGUE,
        source="dialogue_bridge_worker",
        payload=payload
        or {
            "request_id": "dialogue-1",
            "text": "hello jarvis",
            "turn_id": "turn-1",
            "transcript_id": "transcript-1",
            "segment_id": "segment-1",
            "language": "en",
            "confidence": 0.98,
        },
    )


def test_dialogue_cognition_bridge_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        DialogueCognitionBridgeConfig(name=" ").validate()

    with pytest.raises(ValueError):
        DialogueCognitionBridgeConfig(source=" ").validate()

    with pytest.raises(ValueError):
        DialogueCognitionBridgeConfig(default_timeout_ms=0).validate()

    with pytest.raises(ValueError):
        DialogueCognitionBridgeConfig(max_response_chars=0).validate()


def test_dialogue_cognition_bridge_start_subscribes_once() -> None:
    bus = FakeEventBus()
    worker = DialogueCognitionBridgeWorker(event_bus=bus)

    worker.on_start()
    worker.on_start()

    snapshot = worker.snapshot()

    assert snapshot.started is True
    assert snapshot.subscribed is True
    assert bus.subscriptions == [
        (
            EventType.DIALOGUE_RESPONSE_REQUESTED,
            "dialogue_cognition_bridge_worker",
        )
    ]


def test_dialogue_cognition_bridge_stop_is_idempotent() -> None:
    worker = DialogueCognitionBridgeWorker(event_bus=FakeEventBus())

    worker.on_start()
    worker.on_stop()
    worker.on_stop()

    snapshot = worker.snapshot()

    assert snapshot.started is False
    assert snapshot.subscribed is False


def test_dialogue_cognition_bridge_rejects_when_not_started() -> None:
    bus = FakeEventBus()
    worker = DialogueCognitionBridgeWorker(event_bus=bus)

    result = worker.process_dialogue_response_requested(make_event())
    snapshot = worker.snapshot()

    assert result.rejected is True
    assert result.reason == "bridge is not started"
    assert snapshot.rejected_count == 1
    assert bus.published == []


def test_dialogue_cognition_bridge_publishes_cognition_request() -> None:
    bus = FakeEventBus()
    worker = DialogueCognitionBridgeWorker(event_bus=bus)

    worker.on_start()

    result = worker.process_dialogue_response_requested(make_event())
    snapshot = worker.snapshot()

    assert result.accepted is True
    assert result.cognition_request is not None
    assert result.cognition_request.request_id == "cognition-dialogue-1"
    assert result.cognition_request.text == "hello jarvis"
    assert result.cognition_request.turn_id == "turn-1"
    assert result.cognition_request.transcript_id == "transcript-1"
    assert result.cognition_request.correlation_id == "correlation-1"

    assert len(bus.published) == 1
    published = bus.published[0]

    assert published.event_type == EventType.COGNITION_REQUESTED
    assert published.category == EventCategory.COGNITION
    assert published.source == "dialogue_cognition_bridge_worker"
    assert published.correlation_id == "correlation-1"
    assert published.causation_id == "event-1"
    assert published.payload["request_id"] == "cognition-dialogue-1"
    assert published.payload["text"] == "hello jarvis"

    assert snapshot.processed_count == 1
    assert snapshot.published_count == 1
    assert snapshot.rejected_count == 0
    assert snapshot.last_dialogue_request_id == "dialogue-1"
    assert snapshot.last_cognition_request_id == "cognition-dialogue-1"


def test_dialogue_cognition_bridge_builds_policy_from_config() -> None:
    config = DialogueCognitionBridgeConfig(
        streaming_enabled=True,
        allow_memory_lookup=True,
        allow_tools=True,
        default_timeout_ms=10_000,
        max_response_chars=500,
        spoken_style=SpokenResponseStyle.NORMAL,
    )
    worker = DialogueCognitionBridgeWorker(
        event_bus=FakeEventBus(),
        config=config,
    )

    request = worker.build_cognition_request(make_event())

    assert request.policy.streaming_enabled is True
    assert request.policy.allow_memory_lookup is True
    assert request.policy.allow_tools is True
    assert request.policy.timeout_ms == 10_000
    assert request.policy.max_response_chars == 500
    assert request.policy.spoken_style == SpokenResponseStyle.NORMAL


def test_dialogue_cognition_bridge_accepts_alternate_text_keys() -> None:
    bus = FakeEventBus()
    worker = DialogueCognitionBridgeWorker(event_bus=bus)

    worker.on_start()

    result = worker.process_dialogue_response_requested(
        make_event(
            payload={
                "request_id": "dialogue-2",
                "transcript_text": "what is phase three",
            }
        )
    )

    assert result.accepted is True
    assert result.cognition_request is not None
    assert result.cognition_request.text == "what is phase three"
    assert bus.published[0].payload["request_id"] == "cognition-dialogue-2"


def test_dialogue_cognition_bridge_rejects_missing_text() -> None:
    bus = FakeEventBus()
    worker = DialogueCognitionBridgeWorker(event_bus=bus)

    worker.on_start()

    result = worker.process_dialogue_response_requested(
        make_event(payload={"request_id": "dialogue-3"})
    )
    snapshot = worker.snapshot()

    assert result.rejected is True
    assert result.reason == "dialogue payload does not contain response text."
    assert snapshot.rejected_count == 1
    assert bus.published == []


def test_dialogue_cognition_bridge_rejects_wrong_event_type() -> None:
    bus = FakeEventBus()
    worker = DialogueCognitionBridgeWorker(event_bus=bus)

    worker.on_start()

    result = worker.process_dialogue_response_requested(
        make_event(event_type=EventType.COGNITION_REQUESTED)
    )

    assert result.rejected is True
    assert result.reason == "unsupported event type"
    assert bus.published == []


def test_dialogue_cognition_bridge_payload_contains_metadata() -> None:
    bus = FakeEventBus()
    worker = DialogueCognitionBridgeWorker(event_bus=bus)

    worker.on_start()
    worker.process_dialogue_response_requested(make_event())

    payload = bus.published[0].payload

    assert payload["metadata"]["dialogue_request_id"] == "dialogue-1"
    assert payload["metadata"]["segment_id"] == "segment-1"
    assert payload["metadata"]["language"] == "en"
    assert payload["metadata"]["confidence"] == 0.98
    assert payload["policy"]["cancellable"] is True


def test_dialogue_cognition_bridge_generates_request_id_without_id() -> None:
    bus = FakeEventBus()
    worker = DialogueCognitionBridgeWorker(event_bus=bus)

    worker.on_start()

    result = worker.process_dialogue_response_requested(
        make_event(
            payload={"text": "hello"},
            event_type=EventType.DIALOGUE_RESPONSE_REQUESTED,
        )
    )

    assert result.accepted is True
    assert result.cognition_request is not None
    assert result.cognition_request.request_id == "cognition-event-1"