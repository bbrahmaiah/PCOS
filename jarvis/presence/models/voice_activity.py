from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_voice_activity_id() -> str:
    return uuid4().hex


class VoiceActivityState(StrEnum):
    SILENCE = "silence"
    SPEECH_STARTED = "speech_started"
    SPEECH_CONTINUING = "speech_continuing"
    SPEECH_ENDED = "speech_ended"


class VoiceActivity(BaseModel):
    """
    Voice activity detection result.

    VAD workers will emit this after analyzing AudioFrame objects.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    activity_id: str = Field(default_factory=new_voice_activity_id)
    frame_id: str
    state: VoiceActivityState

    is_speech: bool
    confidence: float = Field(ge=0.0, le=1.0)
    energy: float = Field(default=0.0, ge=0.0)

    started_at: datetime | None = None
    ended_at: datetime | None = None
    detected_at: datetime = Field(default_factory=utc_now)

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("activity_id", "frame_id")
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("voice activity string fields cannot be empty.")

        return cleaned