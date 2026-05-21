from __future__ import annotations

from jarvis.presence.adapters.microphone_adapter import (
    MicrophoneAdapter,
    MicrophoneDevice,
)
from jarvis.presence.adapters.playback_adapter import (
    AudioPlaybackAdapter,
    PlaybackResult,
    PlaybackStatus,
)
from jarvis.presence.adapters.stt_adapter import SpeechToTextAdapter
from jarvis.presence.adapters.tts_adapter import TextToSpeechAdapter
from jarvis.presence.adapters.vad_adapter import VoiceActivityAdapter
from jarvis.presence.adapters.wake_word_adapter import (
    WakeWordAdapter,
    WakeWordDetection,
)

__all__ = [
    "AudioPlaybackAdapter",
    "MicrophoneAdapter",
    "MicrophoneDevice",
    "PlaybackResult",
    "PlaybackStatus",
    "SpeechToTextAdapter",
    "TextToSpeechAdapter",
    "VoiceActivityAdapter",
    "WakeWordAdapter",
    "WakeWordDetection",
]