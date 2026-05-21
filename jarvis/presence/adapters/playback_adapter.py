from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from jarvis.presence.models import SpeechChunk


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_playback_result_id() -> str:
    return uuid4().hex


class PlaybackStatus(StrEnum):
    STARTED = "started"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


class PlaybackResult(BaseModel):
    """
    Hardware-independent playback result.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    result_id: str = Field(default_factory=new_playback_result_id)
    chunk_id: str
    request_id: str
    status: PlaybackStatus
    played_at: datetime = Field(default_factory=utc_now)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("result_id", "chunk_id", "request_id")
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("playback result string fields cannot be empty.")

        return cleaned


class AudioPlaybackAdapter(ABC):
    """
    Abstract audio playback boundary.

    Implementations may use sounddevice, PyAudio, Windows audio APIs, or fake
    deterministic test playback.
    """

    @property
    @abstractmethod
    def is_playing(self) -> bool:
        """Return whether audio is currently playing."""

    @abstractmethod
    def play(self, chunk: SpeechChunk) -> PlaybackResult:
        """Play one speech chunk."""

    @abstractmethod
    def stop(self, *, request_id: str | None = None) -> PlaybackResult | None:
        """
        Stop playback.

        Return None if nothing was playing.
        """