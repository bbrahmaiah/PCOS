from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from jarvis.presence.models import AudioFrame


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_wake_detection_id() -> str:
    return uuid4().hex


class WakeWordDetection(BaseModel):
    """
    Wake-word detection result.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    detection_id: str = Field(default_factory=new_wake_detection_id)
    frame_id: str
    wake_word: str
    confidence: float = Field(ge=0.0, le=1.0)
    detected_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("detection_id", "frame_id", "wake_word")
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("wake detection string fields cannot be empty.")

        return cleaned


class WakeWordAdapter(ABC):
    """
    Abstract wake-word boundary.

    Implementations may use openWakeWord, Porcupine, or a fake adapter.
    """

    @abstractmethod
    def detect(self, frame: AudioFrame) -> WakeWordDetection | None:
        """
        Detect wake word in an audio frame.

        Return None when no wake word is detected.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset internal detector state."""