from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_audio_frame_id() -> str:
    return uuid4().hex


class AudioFrame(BaseModel):
    """
    Immutable raw audio frame contract.

    This is the lowest-level Presence data unit. Real microphone adapters,
    fake adapters, VAD, wake detection, and interruption logic will all pass
    AudioFrame objects through the runtime.

    Design:
    - no hardware dependency
    - strict validation
    - immutable top-level model
    - ready for real-time audio pipelines
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    frame_id: str = Field(default_factory=new_audio_frame_id)
    source: str
    audio_data: bytes

    sample_rate: int = Field(default=16_000, ge=8_000, le=192_000)
    channels: int = Field(default=1, ge=1, le=8)
    sample_width_bytes: int = Field(default=2, ge=1, le=8)

    frame_index: int = Field(default=0, ge=0)
    captured_at: datetime = Field(default_factory=utc_now)

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("frame_id", "source")
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("audio frame string fields cannot be empty.")

        return cleaned

    @field_validator("audio_data")
    @classmethod
    def validate_audio_data(cls, value: bytes) -> bytes:
        if not value:
            raise ValueError("audio_data cannot be empty.")

        return value

    @property
    def byte_count(self) -> int:
        return len(self.audio_data)

    @property
    def sample_count(self) -> float:
        bytes_per_sample_frame = self.channels * self.sample_width_bytes
        return self.byte_count / bytes_per_sample_frame

    @property
    def duration_ms(self) -> float:
        return (self.sample_count / self.sample_rate) * 1000