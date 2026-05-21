from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class PresenceMode(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    USER_SPEAKING = "user_speaking"
    PROCESSING_SPEECH = "processing_speech"
    ASSISTANT_SPEAKING = "assistant_speaking"
    INTERRUPTED = "interrupted"
    SLEEPING = "sleeping"
    ERROR = "error"


class TurnPhase(StrEnum):
    NONE = "none"
    WAITING_FOR_WAKE = "waiting_for_wake"
    LISTENING_FOR_USER = "listening_for_user"
    CAPTURING_USER_SPEECH = "capturing_user_speech"
    TRANSCRIBING = "transcribing"
    WAITING_FOR_RESPONSE = "waiting_for_response"
    SPEAKING_RESPONSE = "speaking_response"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


class PresenceState(BaseModel):
    """
    Immutable high-level Presence state snapshot.

    This model does not mutate itself. The future PresenceStateStore will own
    transitions, validation, locking, and event emission.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: PresenceMode = PresenceMode.IDLE
    turn_phase: TurnPhase = TurnPhase.NONE

    awake: bool = False
    listening: bool = False
    user_speaking: bool = False
    assistant_speaking: bool = False

    current_turn_id: str | None = None
    active_speech_request_id: str | None = None
    active_cancellation_token_id: str | None = None

    last_wake_at: datetime | None = None
    last_user_speech_at: datetime | None = None
    last_assistant_speech_at: datetime | None = None
    updated_at: datetime = Field(default_factory=utc_now)

    last_error: str | None = None

    @property
    def interruptible(self) -> bool:
        return (
            self.assistant_speaking
            or self.mode == PresenceMode.ASSISTANT_SPEAKING
        )

    @property
    def active(self) -> bool:
        return (
            self.awake
            or self.listening
            or self.user_speaking
            or self.assistant_speaking
        )