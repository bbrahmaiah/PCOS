from __future__ import annotations

from jarvis.runtime.shared.enums import EventPriority, EventType

CRITICAL_EVENTS: set[EventType] = {
    # Runtime critical lifecycle
    EventType.RUNTIME_STOPPING,
    EventType.RUNTIME_STOPPED,
    EventType.RUNTIME_FAILED,

    # Worker failures
    EventType.WORKER_FAILED,

    # Security/action critical outcomes
    EventType.PERMISSION_DENIED,
    EventType.ACTION_FAILED,
    EventType.ACTION_CANCELLED,

    # Legacy/simple presence interruption
    EventType.INTERRUPT_REQUESTED,

    # Phase 2 presence interruption
    EventType.PRESENCE_USER_INTERRUPTED,
    EventType.PRESENCE_ASSISTANT_SPEECH_CANCELLED,
}


HIGH_PRIORITY_EVENTS: set[EventType] = {
    # Legacy/simple presence commands
    EventType.USER_SPOKE,
    EventType.WAKE_WORD_DETECTED,

    # Security/action request flow
    EventType.PERMISSION_REQUESTED,
    EventType.PERMISSION_GRANTED,
    EventType.ACTION_REQUESTED,
    EventType.ACTION_STARTED,
    EventType.ACTION_COMPLETED,

    # Dialogue request flow
    EventType.ASSISTANT_RESPONSE_REQUESTED,
    EventType.ASSISTANT_RESPONSE_READY,

    # Phase 2 presence realtime flow
    EventType.PRESENCE_WAKE_DETECTED,
    EventType.PRESENCE_USER_STARTED_SPEAKING,
    EventType.PRESENCE_USER_STOPPED_SPEAKING,
    EventType.PRESENCE_TRANSCRIPT_FINAL,

    # Audio failure must surface quickly
    EventType.AUDIO_PLAYBACK_FAILED,
}


LOW_PRIORITY_EVENTS: set[EventType] = {
    # Runtime housekeeping
    EventType.RUNTIME_TICK,

    # Observability / diagnostics
    EventType.METRIC_RECORDED,
    EventType.LATENCY_RECORDED,
    EventType.DIAGNOSTIC_REPORTED,
}


NORMAL_PRIORITY_EVENTS: set[EventType] = {
    # Phase 2 presence lifecycle
    EventType.PRESENCE_STARTED,
    EventType.PRESENCE_STOPPED,
    EventType.PRESENCE_STATE_CHANGED,

    # Phase 2 listening lifecycle
    EventType.PRESENCE_SLEEP_REQUESTED,
    EventType.PRESENCE_LISTEN_STARTED,
    EventType.PRESENCE_LISTEN_STOPPED,
    EventType.PRESENCE_LISTEN_TIMEOUT,

    # Phase 2 transcript lifecycle
    EventType.PRESENCE_TRANSCRIPT_PARTIAL,
    EventType.PRESENCE_TRANSCRIPT_REJECTED,

    # Phase 2 assistant speech lifecycle
    EventType.PRESENCE_ASSISTANT_SPEAKING_STARTED,
    EventType.PRESENCE_ASSISTANT_SPEAKING_STOPPED,

    # Phase 2 audio lifecycle
    EventType.AUDIO_FRAME_CAPTURED,
    EventType.AUDIO_SPEECH_SEGMENT_STARTED,
    EventType.AUDIO_SPEECH_SEGMENT_COMPLETED,
    EventType.AUDIO_PLAYBACK_STARTED,
    EventType.AUDIO_PLAYBACK_STOPPED,
}


def priority_for_event(event_type: EventType) -> EventPriority:
    """
    Return the runtime priority for an event type.

    Design:
    - critical: shutdown, failure, denial, cancellation, interruption
    - high: user speech, wake, permission/action/response request flow
    - low: metrics, ticks, diagnostics
    - normal: default lifecycle/state/background flow
    """

    if event_type in CRITICAL_EVENTS:
        return EventPriority.CRITICAL

    if event_type in HIGH_PRIORITY_EVENTS:
        return EventPriority.HIGH

    if event_type in LOW_PRIORITY_EVENTS:
        return EventPriority.LOW

    return EventPriority.NORMAL