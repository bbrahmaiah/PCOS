from __future__ import annotations

from abc import ABC, abstractmethod

from jarvis.presence.models import AudioFrame, VoiceActivity


class VoiceActivityAdapter(ABC):
    """
    Abstract voice-activity-detection boundary.

    Implementations may use WebRTC VAD, Silero VAD, or a fake adapter.
    """

    @abstractmethod
    def analyze(self, frame: AudioFrame) -> VoiceActivity:
        """Analyze one audio frame and return voice activity."""

    @abstractmethod
    def reset(self) -> None:
        """Reset VAD stream state."""