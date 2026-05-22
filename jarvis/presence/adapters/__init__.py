from __future__ import annotations

from jarvis.presence.adapters.fake_adapters import (
    FakeAudioPlaybackAdapter,
    FakeMicrophoneAdapter,
    FakeSpeechToTextAdapter,
    FakeTextToSpeechAdapter,
    FakeVoiceActivityAdapter,
    FakeWakeWordAdapter,
    make_fake_audio_frame,
)
from jarvis.presence.adapters.microphone_adapter import (
    MicrophoneAdapter,
    MicrophoneDevice,
)
from jarvis.presence.adapters.playback_adapter import (
    AudioPlaybackAdapter,
    PlaybackResult,
    PlaybackStatus,
)
from jarvis.presence.adapters.real_microphone_adapter import (
    RawMicrophoneBackend,
    RealMicrophoneAdapter,
    RealMicrophoneConfig,
    SoundDeviceRawMicrophoneBackend,
)
from jarvis.presence.adapters.stt_adapter import SpeechToTextAdapter
from jarvis.presence.adapters.tts_adapter import TextToSpeechAdapter
from jarvis.presence.adapters.vad_adapter import VoiceActivityAdapter
from jarvis.presence.adapters.wake_word_adapter import (
    WakeWordAdapter,
    WakeWordDetection,
)

__all__ = [

    "RawMicrophoneBackend",
    "RealMicrophoneAdapter",
    "RealMicrophoneConfig",
    "SoundDeviceRawMicrophoneBackend",
    "AudioPlaybackAdapter",
    "FakeAudioPlaybackAdapter",
    "FakeMicrophoneAdapter",
    "FakeSpeechToTextAdapter",
    "FakeTextToSpeechAdapter",
    "FakeVoiceActivityAdapter",
    "FakeWakeWordAdapter",
    "MicrophoneAdapter",
    "MicrophoneDevice",
    "PlaybackResult",
    "PlaybackStatus",
    "SpeechToTextAdapter",
    "TextToSpeechAdapter",
    "VoiceActivityAdapter",
    "WakeWordAdapter",
    "WakeWordDetection",
    "make_fake_audio_frame",
]