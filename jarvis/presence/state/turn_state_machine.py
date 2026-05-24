from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Final
from uuid import uuid4

from jarvis.presence.models import PresenceMode, PresenceState, TurnPhase


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_turn_id() -> str:
    return uuid4().hex


class TurnTrigger(StrEnum):
    """
    External trigger applied to the Presence turn state machine.
    """

    START_LISTENING = "start_listening"
    WAKE_DETECTED = "wake_detected"
    USER_SPEECH_STARTED = "user_speech_started"
    USER_SPEECH_ENDED = "user_speech_ended"
    TRANSCRIPT_READY = "transcript_ready"
    ASSISTANT_RESPONSE_STARTED = "assistant_response_started"
    ASSISTANT_RESPONSE_FINISHED = "assistant_response_finished"
    INTERRUPTION_DETECTED = "interruption_detected"
    SLEEP_REQUESTED = "sleep_requested"
    RESET = "reset"
    ERROR_OCCURRED = "error_occurred"
    RECOVER = "recover"


@dataclass(frozen=True, slots=True)
class TurnTransition:
    """
    Static transition rule.
    """

    from_mode: PresenceMode
    from_phase: TurnPhase
    trigger: TurnTrigger
    to_mode: PresenceMode
    to_phase: TurnPhase


@dataclass(frozen=True, slots=True)
class TurnTransitionResult:
    """
    Result produced by a state transition.
    """

    previous_state: PresenceState
    next_state: PresenceState
    trigger: TurnTrigger
    changed: bool
    reason: str | None = None

    @property
    def allowed(self) -> bool:
        return self.changed


TransitionKey = tuple[PresenceMode, TurnPhase, TurnTrigger]

_TRANSITIONS: Final[dict[TransitionKey, tuple[PresenceMode, TurnPhase]]] = {
    (
        PresenceMode.IDLE,
        TurnPhase.NONE,
        TurnTrigger.START_LISTENING,
    ): (
        PresenceMode.LISTENING,
        TurnPhase.LISTENING_FOR_USER,
    ),
    (
        PresenceMode.IDLE,
        TurnPhase.NONE,
        TurnTrigger.WAKE_DETECTED,
    ): (
        PresenceMode.LISTENING,
        TurnPhase.LISTENING_FOR_USER,
    ),
    (
        PresenceMode.SLEEPING,
        TurnPhase.WAITING_FOR_WAKE,
        TurnTrigger.WAKE_DETECTED,
    ): (
        PresenceMode.LISTENING,
        TurnPhase.LISTENING_FOR_USER,
    ),
    (
        PresenceMode.LISTENING,
        TurnPhase.LISTENING_FOR_USER,
        TurnTrigger.USER_SPEECH_STARTED,
    ): (
        PresenceMode.USER_SPEAKING,
        TurnPhase.CAPTURING_USER_SPEECH,
    ),
    (
        PresenceMode.USER_SPEAKING,
        TurnPhase.CAPTURING_USER_SPEECH,
        TurnTrigger.USER_SPEECH_ENDED,
    ): (
        PresenceMode.PROCESSING_SPEECH,
        TurnPhase.TRANSCRIBING,
    ),
    (
        PresenceMode.PROCESSING_SPEECH,
        TurnPhase.TRANSCRIBING,
        TurnTrigger.TRANSCRIPT_READY,
    ): (
        PresenceMode.PROCESSING_SPEECH,
        TurnPhase.WAITING_FOR_RESPONSE,
    ),
    (
        PresenceMode.PROCESSING_SPEECH,
        TurnPhase.WAITING_FOR_RESPONSE,
        TurnTrigger.ASSISTANT_RESPONSE_STARTED,
    ): (
        PresenceMode.ASSISTANT_SPEAKING,
        TurnPhase.SPEAKING_RESPONSE,
    ),
    (
        PresenceMode.ASSISTANT_SPEAKING,
        TurnPhase.SPEAKING_RESPONSE,
        TurnTrigger.ASSISTANT_RESPONSE_FINISHED,
    ): (
        PresenceMode.LISTENING,
        TurnPhase.LISTENING_FOR_USER,
    ),
    (
        PresenceMode.ASSISTANT_SPEAKING,
        TurnPhase.SPEAKING_RESPONSE,
        TurnTrigger.INTERRUPTION_DETECTED,
    ): (
        PresenceMode.INTERRUPTED,
        TurnPhase.INTERRUPTED,
    ),
    (
        PresenceMode.INTERRUPTED,
        TurnPhase.INTERRUPTED,
        TurnTrigger.USER_SPEECH_STARTED,
    ): (
        PresenceMode.USER_SPEAKING,
        TurnPhase.CAPTURING_USER_SPEECH,
    ),
    (
        PresenceMode.INTERRUPTED,
        TurnPhase.INTERRUPTED,
        TurnTrigger.START_LISTENING,
    ): (
        PresenceMode.LISTENING,
        TurnPhase.LISTENING_FOR_USER,
    ),
    (
        PresenceMode.ERROR,
        TurnPhase.FAILED,
        TurnTrigger.RECOVER,
    ): (
        PresenceMode.IDLE,
        TurnPhase.NONE,
    ),
}


class TurnStateMachine:
    """
    Pure state transition engine for real-time Presence.

    It owns transition rules only. It does not capture audio, transcribe speech,
    speak audio, call workers, mutate shared state, or publish events.
    """

    def transition(
        self,
        state: PresenceState,
        trigger: TurnTrigger,
        *,
        turn_id: str | None = None,
        speech_request_id: str | None = None,
        cancellation_token_id: str | None = None,
        error: str | None = None,
    ) -> TurnTransitionResult:
        if trigger == TurnTrigger.RESET:
            return self._reset(state)

        if trigger == TurnTrigger.SLEEP_REQUESTED:
            return self._sleep(state)

        if trigger == TurnTrigger.ERROR_OCCURRED:
            return self._error(state, error=error)

        key = (state.mode, state.turn_phase, trigger)
        target = _TRANSITIONS.get(key)

        if target is None:
            return TurnTransitionResult(
                previous_state=state,
                next_state=state,
                trigger=trigger,
                changed=False,
                reason=(
                    "Transition not allowed: "
                    f"{state.mode.value}/{state.turn_phase.value} "
                    f"with trigger {trigger.value}."
                ),
            )

        next_mode, next_phase = target
        next_state = self._build_state(
            previous=state,
            trigger=trigger,
            mode=next_mode,
            turn_phase=next_phase,
            turn_id=turn_id,
            speech_request_id=speech_request_id,
            cancellation_token_id=cancellation_token_id,
        )

        return TurnTransitionResult(
            previous_state=state,
            next_state=next_state,
            trigger=trigger,
            changed=next_state != state,
        )

    def start_listening(self, state: PresenceState) -> TurnTransitionResult:
        return self.transition(state, TurnTrigger.START_LISTENING)

    def wake_detected(
        self,
        state: PresenceState,
        *,
        turn_id: str | None = None,
    ) -> TurnTransitionResult:
        return self.transition(
            state,
            TurnTrigger.WAKE_DETECTED,
            turn_id=turn_id,
        )

    def user_speech_started(
        self,
        state: PresenceState,
        *,
        turn_id: str | None = None,
    ) -> TurnTransitionResult:
        return self.transition(
            state,
            TurnTrigger.USER_SPEECH_STARTED,
            turn_id=turn_id,
        )

    def user_speech_ended(self, state: PresenceState) -> TurnTransitionResult:
        return self.transition(state, TurnTrigger.USER_SPEECH_ENDED)

    def transcript_ready(self, state: PresenceState) -> TurnTransitionResult:
        return self.transition(state, TurnTrigger.TRANSCRIPT_READY)

    def assistant_response_started(
        self,
        state: PresenceState,
        *,
        speech_request_id: str,
        cancellation_token_id: str | None = None,
    ) -> TurnTransitionResult:
        return self.transition(
            state,
            TurnTrigger.ASSISTANT_RESPONSE_STARTED,
            speech_request_id=speech_request_id,
            cancellation_token_id=cancellation_token_id,
        )

    def assistant_response_finished(
        self,
        state: PresenceState,
    ) -> TurnTransitionResult:
        return self.transition(state, TurnTrigger.ASSISTANT_RESPONSE_FINISHED)

    def interruption_detected(
        self,
        state: PresenceState,
        *,
        cancellation_token_id: str | None = None,
    ) -> TurnTransitionResult:
        return self.transition(
            state,
            TurnTrigger.INTERRUPTION_DETECTED,
            cancellation_token_id=cancellation_token_id,
        )

    def sleep_requested(self, state: PresenceState) -> TurnTransitionResult:
        return self.transition(state, TurnTrigger.SLEEP_REQUESTED)

    def reset(self, state: PresenceState) -> TurnTransitionResult:
        return self.transition(state, TurnTrigger.RESET)

    def error_occurred(
        self,
        state: PresenceState,
        *,
        error: str,
    ) -> TurnTransitionResult:
        return self.transition(
            state,
            TurnTrigger.ERROR_OCCURRED,
            error=error,
        )

    def recover(self, state: PresenceState) -> TurnTransitionResult:
        return self.transition(state, TurnTrigger.RECOVER)

    def _reset(self, state: PresenceState) -> TurnTransitionResult:
        """
        Reset must erase all runtime presence state.

        Keep this intentionally strict. Reset returns the exact default
        PresenceState so tests and runtime invariants can trust that no old
        turn id, speech id, cancellation token, timestamps, or error survive.
        """

        next_state = PresenceState()

        return TurnTransitionResult(
            previous_state=state,
            next_state=next_state,
            trigger=TurnTrigger.RESET,
            changed=state != next_state,
            reason="presence state reset to default",
        )

    def _sleep(self, state: PresenceState) -> TurnTransitionResult:
        next_state = PresenceState(
            mode=PresenceMode.SLEEPING,
            turn_phase=TurnPhase.WAITING_FOR_WAKE,
            awake=False,
            listening=False,
            user_speaking=False,
            assistant_speaking=False,
            updated_at=utc_now(),
        )

        return TurnTransitionResult(
            previous_state=state,
            next_state=next_state,
            trigger=TurnTrigger.SLEEP_REQUESTED,
            changed=next_state != state,
        )

    def _error(
        self,
        state: PresenceState,
        *,
        error: str | None,
    ) -> TurnTransitionResult:
        clean_error = (error or "Unknown Presence error.").strip()

        if not clean_error:
            clean_error = "Unknown Presence error."

        next_state = PresenceState(
            mode=PresenceMode.ERROR,
            turn_phase=TurnPhase.FAILED,
            awake=state.awake,
            listening=False,
            user_speaking=False,
            assistant_speaking=False,
            current_turn_id=state.current_turn_id,
            active_speech_request_id=state.active_speech_request_id,
            active_cancellation_token_id=state.active_cancellation_token_id,
            last_wake_at=state.last_wake_at,
            last_user_speech_at=state.last_user_speech_at,
            last_assistant_speech_at=state.last_assistant_speech_at,
            updated_at=utc_now(),
            last_error=clean_error,
        )

        return TurnTransitionResult(
            previous_state=state,
            next_state=next_state,
            trigger=TurnTrigger.ERROR_OCCURRED,
            changed=True,
            reason=clean_error,
        )

    def _build_state(
        self,
        *,
        previous: PresenceState,
        trigger: TurnTrigger,
        mode: PresenceMode,
        turn_phase: TurnPhase,
        turn_id: str | None,
        speech_request_id: str | None,
        cancellation_token_id: str | None,
    ) -> PresenceState:
        now = utc_now()
        next_turn_id = self._resolve_turn_id(
            previous=previous,
            trigger=trigger,
            provided_turn_id=turn_id,
        )

        awake = mode not in {PresenceMode.IDLE, PresenceMode.SLEEPING}
        listening = mode in {PresenceMode.LISTENING, PresenceMode.INTERRUPTED}
        user_speaking = mode == PresenceMode.USER_SPEAKING
        assistant_speaking = mode == PresenceMode.ASSISTANT_SPEAKING

        active_speech_request_id = previous.active_speech_request_id
        active_cancellation_token_id = previous.active_cancellation_token_id

        if trigger == TurnTrigger.ASSISTANT_RESPONSE_STARTED:
            active_speech_request_id = self._require_non_empty(
                speech_request_id,
                "speech_request_id",
            )
            active_cancellation_token_id = cancellation_token_id

        if trigger == TurnTrigger.ASSISTANT_RESPONSE_FINISHED:
            active_speech_request_id = None
            active_cancellation_token_id = None

        if trigger == TurnTrigger.INTERRUPTION_DETECTED:
            active_cancellation_token_id = (
                cancellation_token_id or active_cancellation_token_id
            )

        return PresenceState(
            mode=mode,
            turn_phase=turn_phase,
            awake=awake,
            listening=listening,
            user_speaking=user_speaking,
            assistant_speaking=assistant_speaking,
            current_turn_id=next_turn_id,
            active_speech_request_id=active_speech_request_id,
            active_cancellation_token_id=active_cancellation_token_id,
            last_wake_at=now
            if trigger == TurnTrigger.WAKE_DETECTED
            else previous.last_wake_at,
            last_user_speech_at=now
            if trigger == TurnTrigger.USER_SPEECH_STARTED
            else previous.last_user_speech_at,
            last_assistant_speech_at=now
            if trigger == TurnTrigger.ASSISTANT_RESPONSE_STARTED
            else previous.last_assistant_speech_at,
            updated_at=now,
            last_error=None,
        )

    @staticmethod
    def _resolve_turn_id(
        *,
        previous: PresenceState,
        trigger: TurnTrigger,
        provided_turn_id: str | None,
    ) -> str | None:
        if trigger in {
            TurnTrigger.WAKE_DETECTED,
            TurnTrigger.USER_SPEECH_STARTED,
        }:
            if provided_turn_id is not None:
                return TurnStateMachine._require_non_empty(
                    provided_turn_id,
                    "turn_id",
                )

            return previous.current_turn_id or new_turn_id()

        return previous.current_turn_id

    @staticmethod
    def _require_non_empty(value: str | None, field_name: str) -> str:
        if value is None:
            raise ValueError(f"{field_name} is required.")

        clean_value = value.strip()

        if not clean_value:
            raise ValueError(f"{field_name} cannot be empty.")

        return clean_value