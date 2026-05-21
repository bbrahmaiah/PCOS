from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_speech_request_id() -> str:
    return uuid4().hex


def new_speech_chunk_id() -> str:
    return uuid4().hex


class SpeechPriority(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class SpeechRequest(BaseModel):
    """
    Request for JARVIS to speak.

    Dialogue/Cognition will later create these. TTS workers consume them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str = Field(default_factory=new_speech_request_id)
    text: str

    voice_id: str = "default"
    priority: SpeechPriority = SpeechPriority.NORMAL

    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    interruptible: bool = True

    correlation_id: str | None = None
    cancellation_token_id: str | None = None

    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("request_id", "voice_id")
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("speech request string fields cannot be empty.")

        return cleaned

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("speech request text cannot be empty.")

        return cleaned


class SpeechChunk(BaseModel):
    """
    TTS output chunk.

    TTS workers create SpeechChunk objects. Playback workers consume them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str = Field(default_factory=new_speech_chunk_id)
    request_id: str
    audio_data: bytes

    sample_rate: int = Field(default=24_000, ge=8_000, le=192_000)
    channels: int = Field(default=1, ge=1, le=8)
    sample_width_bytes: int = Field(default=2, ge=1, le=8)

    chunk_index: int = Field(default=0, ge=0)
    final: bool = False
    created_at: datetime = Field(default_factory=utc_now)

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("chunk_id", "request_id")
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("speech chunk string fields cannot be empty.")

        return cleaned

    @field_validator("audio_data")
    @classmethod
    def validate_audio_data(cls, value: bytes) -> bytes:
        if not value:
            raise ValueError("speech chunk audio_data cannot be empty.")

        return value

    @property
    def byte_count(self) -> int:
        return len(self.audio_data)