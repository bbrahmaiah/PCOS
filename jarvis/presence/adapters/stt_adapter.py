from __future__ import annotations

from abc import ABC, abstractmethod

from jarvis.presence.models import AudioFrame, Transcript


class SpeechToTextAdapter(ABC):
    """
    Abstract speech-to-text boundary.

    Implementations may use faster-whisper, local models, cloud STT, or fake
    deterministic test adapters.
    """

    @abstractmethod
    def transcribe(self, frames: tuple[AudioFrame, ...]) -> Transcript:
        """Transcribe a complete speech segment."""

    @abstractmethod
    def reset(self) -> None:
        """Reset transcription state."""