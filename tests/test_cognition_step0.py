from __future__ import annotations

from jarvis.cognition import COGNITION_PACKAGE_NAME
from jarvis.runtime.events.priorities import priority_for_event
from jarvis.runtime.shared.enums import EventPriority, EventType


def test_cognition_package_exports_stable_name() -> None:
    assert COGNITION_PACKAGE_NAME == "jarvis.cognition"


def test_cognition_event_types_exist() -> None:
    assert EventType.COGNITION_REQUESTED.value == "cognition.requested"
    assert EventType.COGNITION_STARTED.value == "cognition.started"
    assert EventType.COGNITION_TOKEN_STREAMED.value == "cognition.token_streamed"
    assert EventType.COGNITION_COMPLETED.value == "cognition.completed"
    assert EventType.COGNITION_FAILED.value == "cognition.failed"
    assert EventType.COGNITION_CANCEL_REQUESTED.value == "cognition.cancel_requested"
    assert EventType.COGNITION_CANCELLED.value == "cognition.cancelled"


def test_cognition_event_values_are_unique() -> None:
    cognition_values = {
        EventType.COGNITION_REQUESTED.value,
        EventType.COGNITION_STARTED.value,
        EventType.COGNITION_TOKEN_STREAMED.value,
        EventType.COGNITION_COMPLETED.value,
        EventType.COGNITION_FAILED.value,
        EventType.COGNITION_CANCEL_REQUESTED.value,
        EventType.COGNITION_CANCELLED.value,
    }

    assert len(cognition_values) == 7


def test_cognition_request_completion_and_failure_are_high_priority() -> None:
    assert priority_for_event(EventType.COGNITION_REQUESTED) in {
        EventPriority.HIGH,
        EventPriority.CRITICAL,
    }
    assert priority_for_event(EventType.COGNITION_COMPLETED) in {
        EventPriority.HIGH,
        EventPriority.CRITICAL,
    }
    assert priority_for_event(EventType.COGNITION_FAILED) in {
        EventPriority.HIGH,
        EventPriority.CRITICAL,
    }


def test_cognition_cancel_events_are_not_normal_priority() -> None:
    assert priority_for_event(EventType.COGNITION_CANCEL_REQUESTED) in {
        EventPriority.HIGH,
        EventPriority.CRITICAL,
    }
    assert priority_for_event(EventType.COGNITION_CANCELLED) in {
        EventPriority.HIGH,
        EventPriority.CRITICAL,
    }


def test_cognition_started_is_not_critical_priority() -> None:
    assert priority_for_event(EventType.COGNITION_STARTED) != EventPriority.CRITICAL


def test_cognition_streaming_tokens_do_not_use_critical_priority() -> None:
    assert (
        priority_for_event(EventType.COGNITION_TOKEN_STREAMED)
        != EventPriority.CRITICAL
    )