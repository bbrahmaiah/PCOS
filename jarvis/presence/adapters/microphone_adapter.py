from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field, field_validator

from jarvis.presence.models import AudioFrame


class MicrophoneDevice(BaseModel):
    """
    Hardware-independent microphone device descriptor.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    device_id: str
    name: str
    sample_rate: int = Field(default=16_000, ge=8_000, le=192_000)
    channels: int = Field(default=1, ge=1, le=8)
    is_default: bool = False

    @field_validator("device_id", "name")
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("microphone device string fields cannot be empty.")

        return cleaned


class MicrophoneAdapter(ABC):
    """
    Abstract microphone boundary.

    Real implementations may use sounddevice, PyAudio, WASAPI, or another
    backend. Workers only depend on this interface.
    """

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """Return whether microphone capture is active."""

    @abstractmethod
    def list_devices(self) -> tuple[MicrophoneDevice, ...]:
        """Return available microphone devices."""

    @abstractmethod
    def start(self) -> None:
        """Start microphone capture."""

    @abstractmethod
    def stop(self) -> None:
        """Stop microphone capture."""

    @abstractmethod
    def read_frame(self) -> AudioFrame | None:
        """
        Read one audio frame.

        Return None when no frame is currently available.
        """