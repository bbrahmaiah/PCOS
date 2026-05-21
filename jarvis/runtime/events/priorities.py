from __future__ import annotations

from jarvis.runtime.shared.enums import EventPriority, EventType


CRITICAL_EVENTS: set[EventType] = {
    EventType.INTERRUPT_REQUESTED,
    EventType.RUNTIME_STOPPING,
    EventType.RUNTIME_STOPPED,
    EventType.RUNTIME_FAILED,
    EventType.WORKER_FAILED,
    EventType.PERMISSION_DENIED,
    EventType.ACTION_FAILED,
    EventType.ACTION_CANCELLED,
}


HIGH_PRIORITY_EVENTS: set[EventType] = {
    EventType.USER_SPOKE,
    EventType.WAKE_WORD_DETECTED,
    EventType.PERMISSION_REQUESTED,
    EventType.PERMISSION_GRANTED,
    EventType.ACTION_REQUESTED,
    EventType.ACTION_STARTED,
    EventType.ACTION_COMPLETED,
    EventType.ASSISTANT_RESPONSE_REQUESTED,
    EventType.ASSISTANT_RESPONSE_READY,
}


LOW_PRIORITY_EVENTS: set[EventType] = {
    EventType.RUNTIME_TICK,
    EventType.METRIC_RECORDED,
    EventType.LATENCY_RECORDED,
    EventType.DIAGNOSTIC_REPORTED,
}


def priority_for_event(event_type: EventType) -> EventPriority:
    if event_type in CRITICAL_EVENTS:
        return EventPriority.CRITICAL

    if event_type in HIGH_PRIORITY_EVENTS:
        return EventPriority.HIGH

    if event_type in LOW_PRIORITY_EVENTS:
        return EventPriority.LOW

    return EventPriority.NORMAL