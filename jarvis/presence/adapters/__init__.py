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
from jarvis.presence.adapters.real_playback_adapter import (
    AudioPlaybackBackend,
    RealAudioPlaybackAdapter,
    RealAudioPlaybackConfig,
    SoundDeviceAudioPlaybackBackend,
)
from jarvis.presence.adapters.real_stt_adapter import (
    FasterWhisperSpeechToTextBackend,
    RealSpeechToTextAdapter,
    RealSpeechToTextConfig,
    SpeechToTextBackend,
    log_probability_to_confidence,
    pcm_to_wav_bytes,
)
from jarvis.presence.adapters.real_tts_adapter import (
    Pyttsx3TextToSpeechBackend,
    RealTextToSpeechAdapter,
    RealTextToSpeechConfig,
    TextToSpeechBackend,
    wav_bytes_to_pcm,
)
from jarvis.presence.adapters.real_tts_adapter import (
    pcm_to_wav_bytes as tts_pcm_to_wav_bytes,
)
from jarvis.presence.adapters.real_voice_activity_adapter import (
    AudioEnergyFeatures,
    EnergyVoiceActivityAdapter,
    EnergyVoiceActivityConfig,
    EnergyWakeWordAdapter,
    EnergyWakeWordConfig,
    extract_int16_audio_features,
)
from jarvis.presence.adapters.stt_adapter import SpeechToTextAdapter
from jarvis.presence.adapters.tts_adapter import TextToSpeechAdapter
from jarvis.presence.adapters.vad_adapter import VoiceActivityAdapter
from jarvis.presence.adapters.wake_word_adapter import (
    WakeWordAdapter,
    WakeWordDetection,
)

__all__ = [

    "AudioPlaybackBackend",
    "Pyttsx3TextToSpeechBackend",
    "RealAudioPlaybackAdapter",
    "RealAudioPlaybackConfig",
    "RealTextToSpeechAdapter",
    "RealTextToSpeechConfig",
    "SoundDeviceAudioPlaybackBackend",
    "TextToSpeechBackend",
    "tts_pcm_to_wav_bytes",
    "wav_bytes_to_pcm",
    "FasterWhisperSpeechToTextBackend",
    "RealSpeechToTextAdapter",
    "RealSpeechToTextConfig",
    "SpeechToTextBackend",
    "log_probability_to_confidence",
    "pcm_to_wav_bytes",
    "AudioEnergyFeatures",
    "EnergyVoiceActivityAdapter",
    "EnergyVoiceActivityConfig",
    "EnergyWakeWordAdapter",
    "EnergyWakeWordConfig",
    "extract_int16_audio_features",
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