from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import NewType
from uuid import uuid4

VoiceSessionId = NewType("VoiceSessionId", str)
VoiceFrameId = NewType("VoiceFrameId", str)
VoiceSegmentId = NewType("VoiceSegmentId", str)
VoiceTranscriptId = NewType("VoiceTranscriptId", str)
VoiceTTSChunkId = NewType("VoiceTTSChunkId", str)
VoicePlaybackId = NewType("VoicePlaybackId", str)
VoiceInterruptId = NewType("VoiceInterruptId", str)


class VoiceRuntimeStatus(StrEnum):
    CREATED = "created"
    STARTING = "starting"
    LISTENING = "listening"
    USER_SPEAKING = "user_speaking"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    RECOVERING = "recovering"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class VoiceRuntimeMode(StrEnum):
    SAFE_SIMULATION = "safe_simulation"
    REAL_VOICE = "real_voice"


class VoiceInputFrameKind(StrEnum):
    PCM16_MONO = "pcm16_mono"
    FLOAT32_MONO = "float32_mono"


class VoiceSpeechSegmentStatus(StrEnum):
    STARTED = "started"
    ACTIVE = "active"
    ENDED = "ended"
    CANCELLED = "cancelled"


class VoiceTranscriptKind(StrEnum):
    PARTIAL = "partial"
    FINAL = "final"
    INTERRUPTION = "interruption"


class VoiceTTSChunkStatus(StrEnum):
    CREATED = "created"
    SYNTHESIZED = "synthesized"
    QUEUED = "queued"
    PLAYED = "played"
    CANCELLED = "cancelled"
    FAILED = "failed"


class VoicePlaybackStatus(StrEnum):
    IDLE = "idle"
    QUEUED = "queued"
    PLAYING = "playing"
    PAUSED = "paused"
    STOPPED = "stopped"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


class VoiceInterruptKind(StrEnum):
    STOP = "stop"
    PAUSE = "pause"
    CANCEL = "cancel"
    BARGE_IN = "barge_in"
    USER_CORRECTION = "user_correction"
    UNKNOWN = "unknown"


class VoiceDeviceHealth(StrEnum):
    READY = "ready"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class VoiceRuntimeConfig:
    mode: VoiceRuntimeMode = VoiceRuntimeMode.REAL_VOICE
    assistant_name: str = "JARVIS"
    user_label: str = "Balu"
    wake_word: str = "jarvis"
    sample_rate_hz: int = 16_000
    channels: int = 1
    frame_duration_ms: int = 20
    stt_language: str = "en"
    tts_voice: str = "default"
    interruption_stop_ms: int = 200
    max_silence_ms: int = 1_200
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.assistant_name.strip():
            raise ValueError("assistant_name cannot be empty.")
        if not self.user_label.strip():
            raise ValueError("user_label cannot be empty.")
        if not self.wake_word.strip():
            raise ValueError("wake_word cannot be empty.")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive.")
        if self.channels != 1:
            raise ValueError("voice runtime currently requires mono audio.")
        if self.frame_duration_ms <= 0:
            raise ValueError("frame_duration_ms must be positive.")
        if self.interruption_stop_ms <= 0:
            raise ValueError("interruption_stop_ms must be positive.")
        if self.max_silence_ms <= 0:
            raise ValueError("max_silence_ms must be positive.")


@dataclass(frozen=True, slots=True)
class VoiceRuntimeState:
    session_id: VoiceSessionId
    status: VoiceRuntimeStatus
    mode: VoiceRuntimeMode
    microphone_health: VoiceDeviceHealth
    stt_health: VoiceDeviceHealth
    tts_health: VoiceDeviceHealth
    playback_health: VoiceDeviceHealth
    listening: bool
    user_speaking: bool
    assistant_speaking: bool
    current_segment_id: VoiceSegmentId | None
    last_transcript_id: VoiceTranscriptId | None
    last_playback_id: VoicePlaybackId | None
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VoiceInputFrame:
    frame_id: VoiceFrameId
    session_id: VoiceSessionId
    kind: VoiceInputFrameKind
    sample_rate_hz: int
    channels: int
    data: bytes
    captured_at: datetime
    duration_ms: int
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive.")
        if self.channels != 1:
            raise ValueError("voice input frame must be mono.")
        if self.duration_ms <= 0:
            raise ValueError("duration_ms must be positive.")
        if not self.data:
            raise ValueError("voice input frame data cannot be empty.")


@dataclass(frozen=True, slots=True)
class VoiceSpeechSegment:
    segment_id: VoiceSegmentId
    session_id: VoiceSessionId
    status: VoiceSpeechSegmentStatus
    started_at: datetime
    ended_at: datetime | None = None
    confidence: float = 0.0
    frame_count: int = 0
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("speech segment confidence must be 0..1.")
        if self.frame_count < 0:
            raise ValueError("frame_count cannot be negative.")
        if self.ended_at is not None and self.ended_at < self.started_at:
            raise ValueError("ended_at cannot be before started_at.")


@dataclass(frozen=True, slots=True)
class VoiceTranscript:
    transcript_id: VoiceTranscriptId
    session_id: VoiceSessionId
    segment_id: VoiceSegmentId
    kind: VoiceTranscriptKind
    text: str
    confidence: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("voice transcript text cannot be empty.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("voice transcript confidence must be 0..1.")


@dataclass(frozen=True, slots=True)
class VoiceTTSRequest:
    session_id: VoiceSessionId
    text: str
    voice: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("voice TTS request text cannot be empty.")
        if not self.voice.strip():
            raise ValueError("voice TTS request voice cannot be empty.")


@dataclass(frozen=True, slots=True)
class VoiceTTSChunk:
    chunk_id: VoiceTTSChunkId
    session_id: VoiceSessionId
    status: VoiceTTSChunkStatus
    audio: bytes
    sample_rate_hz: int
    duration_ms: int
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.audio:
            raise ValueError("voice TTS chunk audio cannot be empty.")
        if self.sample_rate_hz <= 0:
            raise ValueError("voice TTS chunk sample_rate_hz must be positive.")
        if self.duration_ms <= 0:
            raise ValueError("voice TTS chunk duration_ms must be positive.")


@dataclass(frozen=True, slots=True)
class VoicePlaybackState:
    playback_id: VoicePlaybackId
    session_id: VoiceSessionId
    status: VoicePlaybackStatus
    chunk_id: VoiceTTSChunkId | None
    started_at: datetime | None
    stopped_at: datetime | None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            self.started_at is not None
            and self.stopped_at is not None
            and self.stopped_at < self.started_at
        ):
            raise ValueError("playback stopped_at cannot be before started_at.")


@dataclass(frozen=True, slots=True)
class VoiceInterruptSignal:
    interrupt_id: VoiceInterruptId
    session_id: VoiceSessionId
    kind: VoiceInterruptKind
    text: str
    confidence: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("voice interrupt text cannot be empty.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("voice interrupt confidence must be 0..1.")


@dataclass(frozen=True, slots=True)
class VoiceRuntimeSnapshot:
    state: VoiceRuntimeState
    captured_frames: int
    speech_segments: int
    partial_transcripts: int
    final_transcripts: int
    tts_chunks: int
    interruptions: int
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VoiceLauncherResult:
    status: VoiceRuntimeStatus
    state: VoiceRuntimeState
    message: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise ValueError("voice launcher result message cannot be empty.")


def utc_now() -> datetime:
    return datetime.now(UTC)


def make_voice_session_id() -> VoiceSessionId:
    return VoiceSessionId(f"voice_session_{uuid4().hex}")


def make_voice_frame_id() -> VoiceFrameId:
    return VoiceFrameId(f"voice_frame_{uuid4().hex}")


def make_voice_segment_id() -> VoiceSegmentId:
    return VoiceSegmentId(f"voice_segment_{uuid4().hex}")


def make_voice_transcript_id() -> VoiceTranscriptId:
    return VoiceTranscriptId(f"voice_transcript_{uuid4().hex}")


def make_voice_tts_chunk_id() -> VoiceTTSChunkId:
    return VoiceTTSChunkId(f"voice_tts_chunk_{uuid4().hex}")


def make_voice_playback_id() -> VoicePlaybackId:
    return VoicePlaybackId(f"voice_playback_{uuid4().hex}")


def make_voice_interrupt_id() -> VoiceInterruptId:
    return VoiceInterruptId(f"voice_interrupt_{uuid4().hex}")


def default_voice_runtime_config() -> VoiceRuntimeConfig:
    return VoiceRuntimeConfig()


def default_voice_runtime_state(
    config: VoiceRuntimeConfig | None = None,
) -> VoiceRuntimeState:
    resolved = config or default_voice_runtime_config()
    now = utc_now()
    return VoiceRuntimeState(
        session_id=make_voice_session_id(),
        status=VoiceRuntimeStatus.CREATED,
        mode=resolved.mode,
        microphone_health=VoiceDeviceHealth.READY,
        stt_health=VoiceDeviceHealth.READY,
        tts_health=VoiceDeviceHealth.READY,
        playback_health=VoiceDeviceHealth.READY,
        listening=False,
        user_speaking=False,
        assistant_speaking=False,
        current_segment_id=None,
        last_transcript_id=None,
        last_playback_id=None,
        created_at=now,
        updated_at=now,
    )