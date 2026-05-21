from __future__ import annotations

from abc import ABC, abstractmethod

from jarvis.presence.models import SpeechChunk, SpeechRequest


class TextToSpeechAdapter(ABC):
    """
    Abstract text-to-speech boundary.

    Implementations may use local TTS, cloud TTS, streaming TTS, or fake test
    adapters.
    """

    @abstractmethod
    def synthesize(self, request: SpeechRequest) -> tuple[SpeechChunk, ...]:
        """Convert a speech request into one or more audio chunks."""

    @abstractmethod
    def reset(self) -> None:
        """Reset synthesis state."""