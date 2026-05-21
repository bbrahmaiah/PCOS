from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from threading import RLock
from typing import Any

from jarvis.presence.models import PresenceMode, PresenceState, TurnPhase
from jarvis.presence.state.turn_state_machine import (
    TurnStateMachine,
    TurnTransitionResult,
    TurnTrigger,
)
from jarvis.runtime.events import EventBus, RuntimeEvent
from jarvis.runtime.observability.structured_logger import get_logger
from jarvis.runtime.shared.enums import EventCategory, EventType


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class PresenceTransitionRecord:
    """
    Immutable audit record for one Presence state transition.
    """

    trigger: TurnTrigger
    changed: bool
    previous_mode: PresenceMode
    next_mode: PresenceMode
    previous_phase: TurnPhase
    next_phase: TurnPhase
    current_turn_id: str | None
    reason: str | None
    recorded_at: datetime


@dataclass(frozen=True, slots=True)
class PresenceStoreSnapshot:
    """
    Immutable diagnostic snapshot of the Presence state store.
    """

    state: PresenceState
    transition_count: int
    last_transition: PresenceTransitionRecord | None


class PresenceStateStore:
    """
    Thread-safe owner of Presence runtime state.

    Design:
    - owns the current PresenceState
    - uses TurnStateMachine for all valid transitions
    - protects mutation with RLock
    - records bounded transition history
    - emits RuntimeEvent when state changes
    - safe for parallel Presence workers

    This class does not:
    - capture audio
    - transcribe speech
    - play audio
    - start/stop workers
    - perform cognition
    """

    def __init__(
        self,
        *,
        event_bus: EventBus | None = None,
        initial_state: PresenceState | None = None,
        machine: TurnStateMachine | None = None,
        history_limit: int = 500,
        event_source: str = "presence_state_store",
    ) -> None:
        if history_limit < 1:
            raise ValueError("history_limit must be greater than zero.")

        clean_source = event_source.strip()

        if not clean_source:
            raise ValueError("event_source cannot be empty.")

        self._event_bus = event_bus
        self._state = initial_state or PresenceState()
        self._machine = machine or TurnStateMachine()
        self._history_limit = history_limit
        self._event_source = clean_source

        self._lock = RLock()
        self._history: list[PresenceTransitionRecord] = []
        self._logger = get_logger("presence.state_store")

    @property
    def event_source(self) -> str:
        return self._event_source

    def snapshot(self) -> PresenceStoreSnapshot:
        with self._lock:
            last_transition = self._history[-1] if self._history else None

            return PresenceStoreSnapshot(
                state=self._state,
                transition_count=len(self._history),
                last_transition=last_transition,
            )

    def current_state(self) -> PresenceState:
        with self._lock:
            return self._state

    def history(self) -> tuple[PresenceTransitionRecord, ...]:
        with self._lock:
            return tuple(self._history)

    def transition(
        self,
        trigger: TurnTrigger,
        *,
        turn_id: str | None = None,
        speech_request_id: str | None = None,
        cancellation_token_id: str | None = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TurnTransitionResult:
        """
        Apply one Presence transition safely.

        If the transition changes state, a presence.state_changed event is
        emitted after internal state is updated.
        """

        with self._lock:
            previous_state = self._state

            result = self._machine.transition(
                previous_state,
                trigger,
                turn_id=turn_id,
                speech_request_id=speech_request_id,
                cancellation_token_id=cancellation_token_id,
                error=error,
            )

            record = PresenceTransitionRecord(
                trigger=trigger,
                changed=result.changed,
                previous_mode=previous_state.mode,
                next_mode=result.next_state.mode,
                previous_phase=previous_state.turn_phase,
                next_phase=result.next_state.turn_phase,
                current_turn_id=result.next_state.current_turn_id,
                reason=result.reason,
                recorded_at=utc_now(),
            )

            self._append_history(record)

            if not result.changed:
                self._logger.info(
                    "presence_transition_rejected",
                    trigger=trigger.value,
                    mode=previous_state.mode.value,
                    phase=previous_state.turn_phase.value,
                    reason=result.reason,
                )
                return result

            self._state = result.next_state

            event_payload = self._build_event_payload(
                record=record,
                metadata=metadata,
            )

        self._emit_state_changed(event_payload)

        self._logger.info(
            "presence_state_changed",
            trigger=trigger.value,
            previous_mode=record.previous_mode.value,
            next_mode=record.next_mode.value,
            previous_phase=record.previous_phase.value,
            next_phase=record.next_phase.value,
            current_turn_id=record.current_turn_id,
        )

        return result

    def start_listening(
        self,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> TurnTransitionResult:
        return self.transition(
            TurnTrigger.START_LISTENING,
            metadata=metadata,
        )

    def wake_detected(
        self,
        *,
        turn_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TurnTransitionResult:
        return self.transition(
            TurnTrigger.WAKE_DETECTED,
            turn_id=turn_id,
            metadata=metadata,
        )

    def user_speech_started(
        self,
        *,
        turn_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TurnTransitionResult:
        return self.transition(
            TurnTrigger.USER_SPEECH_STARTED,
            turn_id=turn_id,
            metadata=metadata,
        )

    def user_speech_ended(
        self,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> TurnTransitionResult:
        return self.transition(
            TurnTrigger.USER_SPEECH_ENDED,
            metadata=metadata,
        )

    def transcript_ready(
        self,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> TurnTransitionResult:
        return self.transition(
            TurnTrigger.TRANSCRIPT_READY,
            metadata=metadata,
        )

    def assistant_response_started(
        self,
        *,
        speech_request_id: str,
        cancellation_token_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TurnTransitionResult:
        return self.transition(
            TurnTrigger.ASSISTANT_RESPONSE_STARTED,
            speech_request_id=speech_request_id,
            cancellation_token_id=cancellation_token_id,
            metadata=metadata,
        )

    def assistant_response_finished(
        self,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> TurnTransitionResult:
        return self.transition(
            TurnTrigger.ASSISTANT_RESPONSE_FINISHED,
            metadata=metadata,
        )

    def interruption_detected(
        self,
        *,
        cancellation_token_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TurnTransitionResult:
        return self.transition(
            TurnTrigger.INTERRUPTION_DETECTED,
            cancellation_token_id=cancellation_token_id,
            metadata=metadata,
        )

    def sleep_requested(
        self,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> TurnTransitionResult:
        return self.transition(
            TurnTrigger.SLEEP_REQUESTED,
            metadata=metadata,
        )

    def reset(
        self,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> TurnTransitionResult:
        return self.transition(
            TurnTrigger.RESET,
            metadata=metadata,
        )

    def error_occurred(
        self,
        *,
        error: str,
        metadata: dict[str, Any] | None = None,
    ) -> TurnTransitionResult:
        return self.transition(
            TurnTrigger.ERROR_OCCURRED,
            error=error,
            metadata=metadata,
        )

    def recover(
        self,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> TurnTransitionResult:
        return self.transition(
            TurnTrigger.RECOVER,
            metadata=metadata,
        )

    def _append_history(self, record: PresenceTransitionRecord) -> None:
        self._history.append(record)

        if len(self._history) > self._history_limit:
            del self._history[: len(self._history) - self._history_limit]

    def _build_event_payload(
        self,
        *,
        record: PresenceTransitionRecord,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "trigger": record.trigger.value,
            "changed": record.changed,
            "previous_mode": record.previous_mode.value,
            "next_mode": record.next_mode.value,
            "previous_phase": record.previous_phase.value,
            "next_phase": record.next_phase.value,
            "current_turn_id": record.current_turn_id,
            "reason": record.reason,
            "metadata": metadata or {},
        }

    def _emit_state_changed(self, payload: dict[str, Any]) -> None:
        if self._event_bus is None:
            return

        event = RuntimeEvent(
            event_type=EventType.PRESENCE_STATE_CHANGED,
            category=EventCategory.PRESENCE,
            source=self._event_source,
            payload=payload,
        )

        self._event_bus.publish(event)