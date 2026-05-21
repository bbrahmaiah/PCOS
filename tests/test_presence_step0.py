from __future__ import annotations

from jarvis.presence import PRESENCE_PACKAGE_NAME
from jarvis.runtime.events import RuntimeEvent
from jarvis.runtime.events.priorities import priority_for_event
from jarvis.runtime.shared.enums import EventCategory, EventPriority, EventType


def test_presence_package_imports() -> None:
    assert PRESENCE_PACKAGE_NAME == "jarvis.presence"


def test_presence_event_types_exist() -> None:
    assert EventType.PRESENCE_STARTED.value == "presence.started"
    assert EventType.PRESENCE_WAKE_DETECTED.value == "presence.wake_detected"
    assert (
        EventType.PRESENCE_USER_STARTED_SPEAKING.value
        == "presence.user_started_speaking"
    )
    assert EventType.PRESENCE_TRANSCRIPT_FINAL.value == "presence.transcript_final"
    assert (
        EventType.PRESENCE_ASSISTANT_SPEECH_CANCELLED.value
        == "presence.assistant_speech_cancelled"
    )
    assert EventType.AUDIO_FRAME_CAPTURED.value == "audio.frame_captured"
    assert EventType.AUDIO_PLAYBACK_FAILED.value == "audio.playback_failed"


def test_presence_events_can_create_runtime_events() -> None:
    event = RuntimeEvent(
        event_type=EventType.PRESENCE_WAKE_DETECTED,
        category=EventCategory.PRESENCE,
        source="presence.step0.test",
        payload={
            "wake_word": "jarvis",
            "confidence": 0.98,
        },
    )

    assert event.event_type == EventType.PRESENCE_WAKE_DETECTED
    assert event.category == EventCategory.PRESENCE
    assert event.priority == EventPriority.HIGH
    assert event.payload["wake_word"] == "jarvis"


def test_presence_interruption_is_critical_priority() -> None:
    event = RuntimeEvent(
        event_type=EventType.PRESENCE_USER_INTERRUPTED,
        category=EventCategory.PRESENCE,
        source="presence.step0.test",
    )

    assert event.priority == EventPriority.CRITICAL


def test_presence_audio_frame_event_is_normal_priority() -> None:
    event = RuntimeEvent(
        event_type=EventType.AUDIO_FRAME_CAPTURED,
        category=EventCategory.PRESENCE,
        source="presence.step0.test",
        payload={
            "sample_rate": 16_000,
            "channels": 1,
        },
    )

    assert event.priority == EventPriority.NORMAL


def test_presence_priority_mapping() -> None:
    assert (
        priority_for_event(EventType.PRESENCE_USER_INTERRUPTED)
        == EventPriority.CRITICAL
    )
    assert priority_for_event(EventType.PRESENCE_WAKE_DETECTED) == EventPriority.HIGH
    assert priority_for_event(EventType.PRESENCE_TRANSCRIPT_FINAL) == EventPriority.HIGH
    assert priority_for_event(EventType.AUDIO_FRAME_CAPTURED) == EventPriority.NORMAL