from __future__ import annotations

import pytest

from jarvis.presence.models import Transcript, TranscriptKind
from jarvis.presence.workers import DialogueBridgePolicy, DialogueBridgeWorker
from jarvis.runtime.events import EventBus, RuntimeEvent
from jarvis.runtime.shared.enums import EventCategory, EventPriority, EventType


def make_transcript(
    *,
    text: str = "hello jarvis",
    kind: TranscriptKind = TranscriptKind.FINAL,
) -> Transcript:
    return Transcript(
        segment_id="segment-1",
        text=text,
        kind=kind,
        confidence=0.98,
        language="en",
    )


def make_transcript_event(transcript: Transcript) -> RuntimeEvent:
    return RuntimeEvent(
        event_type=EventType.PRESENCE_TRANSCRIPT_FINAL,
        category=EventCategory.PRESENCE,
        source="test_stt_worker",
        payload={
            "transcript": transcript,
            "transcript_id": transcript.transcript_id,
            "text": transcript.text,
        },
    )


def test_dialogue_bridge_requires_positive_tick_interval() -> None:
    bus = EventBus(name="presence_test_bus")

    with pytest.raises(ValueError):
        DialogueBridgeWorker(
            event_bus=bus,
            tick_interval_seconds=0,
        )


def test_dialogue_bridge_subscribes_on_start() -> None:
    bus = EventBus(name="presence_test_bus")
    worker = DialogueBridgeWorker(event_bus=bus)

    worker.on_start()

    assert worker.dialogue_snapshot().subscribed is True

    worker.on_stop()


def test_dialogue_bridge_ignores_non_transcript_event() -> None:
    bus = EventBus(name="presence_test_bus")
    worker = DialogueBridgeWorker(event_bus=bus)

    event = RuntimeEvent(
        event_type=EventType.RUNTIME_TICK,
        category=EventCategory.RUNTIME,
        source="test",
    )

    worker.handle_transcript_event(event)

    snapshot = worker.dialogue_snapshot()

    assert snapshot.ignored_events == 1
    assert snapshot.response_requests == 0


def test_dialogue_bridge_ignores_event_without_transcript_object() -> None:
    bus = EventBus(name="presence_test_bus")
    worker = DialogueBridgeWorker(event_bus=bus)

    event = RuntimeEvent(
        event_type=EventType.PRESENCE_TRANSCRIPT_FINAL,
        category=EventCategory.PRESENCE,
        source="test",
        payload={"text": "hello jarvis"},
    )

    worker.handle_transcript_event(event)

    snapshot = worker.dialogue_snapshot()

    assert snapshot.ignored_events == 1
    assert snapshot.response_requests == 0


def test_dialogue_bridge_publishes_response_request() -> None:
    bus = EventBus(name="presence_test_bus")
    worker = DialogueBridgeWorker(event_bus=bus)
    transcript = make_transcript(text="open the browser")

    request_id = worker.process_transcript(transcript=transcript)

    history = bus.history()
    snapshot = worker.dialogue_snapshot()

    assert request_id is not None
    assert len(history) == 1
    assert history[0].event_type == EventType.ASSISTANT_RESPONSE_REQUESTED
    assert history[0].category == EventCategory.DIALOGUE
    assert history[0].priority == EventPriority.HIGH
    assert history[0].source == "dialogue_bridge_worker"
    assert history[0].payload["request_id"] == request_id
    assert history[0].payload["text"] == "open the browser"
    assert history[0].payload["response_style"] == "concise_human"
    assert history[0].payload["interruptible"] is True
    assert snapshot.processed_transcripts == 1
    assert snapshot.response_requests == 1
    assert snapshot.last_text == "open the browser"


def test_dialogue_bridge_uses_custom_policy() -> None:
    bus = EventBus(name="presence_test_bus")
    policy = DialogueBridgePolicy(
        response_style="brief_operator",
        max_response_words_hint=20,
        allow_clarifying_question=False,
        interruptible=True,
        prefer_short_spoken_response=True,
    )
    worker = DialogueBridgeWorker(event_bus=bus, policy=policy)

    worker.process_transcript(transcript=make_transcript())

    event = bus.history()[0]

    assert event.payload["response_style"] == "brief_operator"
    assert event.payload["max_response_words_hint"] == 20
    assert event.payload["allow_clarifying_question"] is False


def test_dialogue_bridge_rejects_partial_transcript() -> None:
    bus = EventBus(name="presence_test_bus")
    worker = DialogueBridgeWorker(event_bus=bus)
    transcript = make_transcript(text="hello", kind=TranscriptKind.PARTIAL)

    request_id = worker.process_transcript(transcript=transcript)

    snapshot = worker.dialogue_snapshot()

    assert request_id is None
    assert bus.history() == ()
    assert snapshot.rejected_transcripts == 1
    assert snapshot.response_requests == 0
    assert snapshot.last_transcript_id == transcript.transcript_id


def test_dialogue_bridge_can_process_transcript_event() -> None:
    bus = EventBus(name="presence_test_bus")
    worker = DialogueBridgeWorker(event_bus=bus)
    transcript = make_transcript(text="what time is it")

    worker.handle_transcript_event(make_transcript_event(transcript))

    assert worker.dialogue_snapshot().response_requests == 1
    assert len(bus.history()) == 1
    assert bus.history()[0].payload["text"] == "what time is it"


def test_dialogue_bridge_subscribed_event_flow_with_publish_sync() -> None:
    bus = EventBus(name="presence_test_bus")
    worker = DialogueBridgeWorker(event_bus=bus)
    transcript = make_transcript(text="continue")

    worker.on_start()
    bus.publish_sync(make_transcript_event(transcript))

    assert worker.dialogue_snapshot().response_requests == 1
    assert len(bus.history()) == 2

    worker.on_stop()


def test_dialogue_bridge_rejects_empty_worker_name() -> None:
    bus = EventBus(name="presence_test_bus")

    with pytest.raises(ValueError):
        DialogueBridgeWorker(
            event_bus=bus,
            name="   ",
        )