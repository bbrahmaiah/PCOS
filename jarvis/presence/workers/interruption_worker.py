from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any
from uuid import uuid4

from jarvis.presence.state import PresenceStateStore
from jarvis.runtime.events import EventBus, RuntimeEvent
from jarvis.runtime.observability.structured_logger import get_logger
from jarvis.runtime.shared.enums import EventCategory, EventType
from jarvis.runtime.workers import BaseWorker


def new_interruption_id() -> str:
    return uuid4().hex


@dataclass(frozen=True, slots=True)
class InterruptionWorkerSnapshot:
    """
    Immutable diagnostic snapshot for InterruptionWorker.
    """

    name: str
    subscribed: bool
    processed_user_speech_events: int
    interruptions_requested: int
    ignored_events: int
    non_interrupting_speech_events: int
    state_transition_failures: int
    last_interruption_id: str | None
    last_request_id: str | None
    last_segment_id: str | None
    last_error: str | None


class InterruptionWorker(BaseWorker):
    """
    Event-driven barge-in/interruption detector.

    Design:
    - consumes presence.user_started_speaking events
    - checks PresenceStateStore for assistant_speaking
    - emits presence.interrupt_requested
    - updates PresenceStateStore through interruption_detected()
    - does not capture audio
    - does not run VAD
    - does not stop audio hardware directly
    - does not perform STT/TTS/cognition

    This worker makes conversation feel real-time:
    if the user speaks while JARVIS is speaking, JARVIS must stop.
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        presence_store: PresenceStateStore,
        name: str = "interruption_worker",
        tick_interval_seconds: float = 0.05,
        daemon: bool = True,
        auto_subscribe: bool = True,
    ) -> None:
        clean_name = name.strip()

        if not clean_name:
            raise ValueError("InterruptionWorker name cannot be empty.")

        if tick_interval_seconds <= 0:
            raise ValueError("tick_interval_seconds must be greater than zero.")

        super().__init__(
            name=clean_name,
            event_bus=event_bus,
            tick_interval_seconds=tick_interval_seconds,
            daemon=daemon,
        )

        self._presence_store = presence_store
        self._auto_subscribe = auto_subscribe

        self._lock = RLock()
        self._subscribed = False
        self._processed_user_speech_events = 0
        self._interruptions_requested = 0
        self._ignored_events = 0
        self._non_interrupting_speech_events = 0
        self._state_transition_failures = 0
        self._last_interruption_id: str | None = None
        self._last_request_id: str | None = None
        self._last_segment_id: str | None = None
        self._last_error: str | None = None

        self._logger = get_logger("presence.interruption_worker")

    @property
    def presence_store(self) -> PresenceStateStore:
        return self._presence_store

    def on_start(self) -> None:
        """
        Subscribe to user speech start events when the worker starts.
        """

        if not self._auto_subscribe:
            return

        with self._lock:
            if self._subscribed:
                return

            self.event_bus.subscribe(
                event_type=EventType.PRESENCE_USER_STARTED_SPEAKING,
                subscriber_name=self.name,
                callback=self.handle_user_started_speaking_event,
            )
            self._subscribed = True

        self._logger.info(
            "interruption_worker_subscribed",
            worker=self.name,
            event_type=EventType.PRESENCE_USER_STARTED_SPEAKING.value,
        )

    def on_stop(self) -> None:
        """
        Event-driven worker has no external adapter to stop.
        """

        self._logger.info("interruption_worker_stopped", worker=self.name)

    def run_once(self) -> None:
        """
        Event-driven worker loop placeholder.

        Interruption work happens through handle_user_started_speaking_event().
        """

    def handle_user_started_speaking_event(self, event: RuntimeEvent) -> None:
        """
        Consume one presence.user_started_speaking event.
        """

        if event.event_type != EventType.PRESENCE_USER_STARTED_SPEAKING:
            self._record_ignored_event()
            return

        segment_id = self._extract_optional_string(event, "segment_id")
        frame_id = self._extract_optional_string(event, "frame_id")

        self.process_user_started_speaking(
            segment_id=segment_id,
            frame_id=frame_id,
            source_event=event,
        )

    def process_user_started_speaking(
        self,
        *,
        segment_id: str | None = None,
        frame_id: str | None = None,
        source_event: RuntimeEvent | None = None,
    ) -> str | None:
        """
        Request interruption if the assistant is currently speaking.
        """

        state = self._presence_store.current_state()

        with self._lock:
            self._processed_user_speech_events += 1

        if not state.assistant_speaking:
            with self._lock:
                self._non_interrupting_speech_events += 1
                self._last_error = None
            return None

        interruption_id = new_interruption_id()
        request_id = state.active_speech_request_id
        cancellation_token_id = state.active_cancellation_token_id

        try:
            transition = self._presence_store.interruption_detected(
                cancellation_token_id=cancellation_token_id,
                metadata={
                    "source": self.name,
                    "interruption_id": interruption_id,
                    "request_id": request_id,
                    "segment_id": segment_id,
                    "frame_id": frame_id,
                },
            )

        except Exception as exc:
            self._record_state_transition_failure(exc)
            raise

        if not transition.changed:
            with self._lock:
                self._non_interrupting_speech_events += 1
                self._last_error = transition.reason
            return None

        payload = {
            "interruption_id": interruption_id,
            "request_id": request_id,
            "cancellation_token_id": cancellation_token_id,
            "segment_id": segment_id,
            "frame_id": frame_id,
            "reason": "user_barge_in",
            "source": self.name,
        }

        self._publish_interrupt_requested(
            payload=payload,
            source_event=source_event,
        )

        with self._lock:
            self._interruptions_requested += 1
            self._last_interruption_id = interruption_id
            self._last_request_id = request_id
            self._last_segment_id = segment_id
            self._last_error = None

        self._logger.info(
            "interruption_requested",
            worker=self.name,
            interruption_id=interruption_id,
            request_id=request_id,
            segment_id=segment_id,
            frame_id=frame_id,
        )

        return interruption_id

    def interruption_snapshot(self) -> InterruptionWorkerSnapshot:
        """
        Return InterruptionWorker-specific diagnostics.
        """

        with self._lock:
            return InterruptionWorkerSnapshot(
                name=self.name,
                subscribed=self._subscribed,
                processed_user_speech_events=self._processed_user_speech_events,
                interruptions_requested=self._interruptions_requested,
                ignored_events=self._ignored_events,
                non_interrupting_speech_events=(
                    self._non_interrupting_speech_events
                ),
                state_transition_failures=self._state_transition_failures,
                last_interruption_id=self._last_interruption_id,
                last_request_id=self._last_request_id,
                last_segment_id=self._last_segment_id,
                last_error=self._last_error,
            )

    def _publish_interrupt_requested(
        self,
        *,
        payload: dict[str, Any],
        source_event: RuntimeEvent | None,
    ) -> None:
        """
        Publish interruption synchronously.

        Interruption is latency-critical. If the user speaks while JARVIS is
        speaking, playback must stop immediately in the same event turn.
        """

        if source_event is None:
            event = RuntimeEvent(
                event_type=EventType.INTERRUPT_REQUESTED,
                category=EventCategory.PRESENCE,
                source=self.name,
                payload=payload,
            )
        else:
            event = RuntimeEvent(
                event_type=EventType.INTERRUPT_REQUESTED,
                category=EventCategory.PRESENCE,
                source=self.name,
                correlation_id=source_event.correlation_id,
                payload=payload,
            )

        self.event_bus.publish_sync(event)

    def _record_ignored_event(self) -> None:
        with self._lock:
            self._ignored_events += 1

    def _record_state_transition_failure(self, exc: Exception) -> None:
        error = f"{type(exc).__name__}: {exc}"

        with self._lock:
            self._state_transition_failures += 1
            self._last_error = error

        self._logger.error(
            "interruption_state_transition_failed",
            worker=self.name,
            error=error,
        )

    @staticmethod
    def _extract_optional_string(
        event: RuntimeEvent,
        key: str,
    ) -> str | None:
        value = event.payload.get(key)

        if isinstance(value, str) and value.strip():
            return value

        return None