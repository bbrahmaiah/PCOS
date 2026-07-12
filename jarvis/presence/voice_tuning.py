from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from jarvis.presence.adapters import (
    EnergyVoiceActivityAdapter,
    EnergyVoiceActivityConfig,
    RealSpeechToTextAdapter,
    RealSpeechToTextConfig,
)
from jarvis.presence.full_voice_smoke import (
    FullVoiceSmokeConfig,
    FullVoiceSmokeReport,
)
from jarvis.presence.models import Transcript


class VoiceRuntimePreset(StrEnum):
    """
    Named real-time voice profiles.

    FAST:
        Lowest latency. Best for smoke tests and weak CPUs.

    BALANCED:
        Better speech boundaries with still-good latency.

    ACCURATE:
        Better transcription model, higher latency.
    """

    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"


@dataclass(frozen=True, slots=True)
class VoiceRuntimeProfile:
    """
    Real-time voice tuning profile for live JARVIS smoke runs.
    """

    preset: VoiceRuntimePreset
    stt_model: str
    vad_threshold: float
    vad_silence_threshold: float
    speech_start_frames: int
    speech_end_frames: int
    adaptive_vad: bool
    duration_seconds: float
    max_frames: int
    response_text: str

    def validate(self) -> None:
        if not self.stt_model.strip():
            raise ValueError("stt_model cannot be empty.")

        if self.vad_threshold <= 0:
            raise ValueError("vad_threshold must be greater than zero.")

        if self.vad_silence_threshold < 0:
            raise ValueError("vad_silence_threshold cannot be negative.")

        if self.speech_start_frames <= 0:
            raise ValueError("speech_start_frames must be greater than zero.")

        if self.speech_end_frames <= 0:
            raise ValueError("speech_end_frames must be greater than zero.")

        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be greater than zero.")

        if self.max_frames <= 0:
            raise ValueError("max_frames must be greater than zero.")

        if not self.response_text.strip():
            raise ValueError("response_text cannot be empty.")


VOICE_RUNTIME_PROFILES: dict[VoiceRuntimePreset, VoiceRuntimeProfile] = {
    VoiceRuntimePreset.FAST: VoiceRuntimeProfile(
        preset=VoiceRuntimePreset.FAST,
        stt_model="small",
        vad_threshold=120.0,
        vad_silence_threshold=60.0,
        speech_start_frames=2,
        speech_end_frames=10,
        adaptive_vad=False,
        duration_seconds=35.0,
        max_frames=4_000,
        response_text="Yes sir. I heard you.",
    ),
    VoiceRuntimePreset.BALANCED: VoiceRuntimeProfile(
        preset=VoiceRuntimePreset.BALANCED,
        stt_model="base",
        vad_threshold=120.0,
        vad_silence_threshold=60.0,
        speech_start_frames=2,
        speech_end_frames=10,
        adaptive_vad=False,
        duration_seconds=35.0,
        max_frames=4_000,
        response_text="Yes sir. I heard you.",
    ),
    VoiceRuntimePreset.ACCURATE: VoiceRuntimeProfile(
        preset=VoiceRuntimePreset.ACCURATE,
        stt_model="small",
        vad_threshold=150.0,
        vad_silence_threshold=75.0,
        speech_start_frames=3,
        speech_end_frames=12,
        adaptive_vad=False,
        duration_seconds=45.0,
        max_frames=5_000,
        response_text="Yes sir. I heard you.",
    ),
}


@dataclass(frozen=True, slots=True)
class WarmupSpeechSegment:
    """
    Silent warmup segment used to load STT before microphone capture.

    This prevents the first real user utterance from being delayed by model
    loading. The transcript may be empty; the purpose is model warmup.
    """

    segment_id: str = "warmup-segment"
    audio_data: bytes = b"\x00\x00" * 16_000
    sample_rate: int = 16_000
    channels: int = 1
    metadata: dict[str, Any] = field(default_factory=lambda: {"source": "warmup"})


@runtime_checkable
class WarmableSpeechToTextAdapter(Protocol):
    """
    Minimal STT contract for warmup.
    """

    def transcribe(self, segment: WarmupSpeechSegment) -> Transcript | None:
        """Warm or transcribe one segment."""


@dataclass(frozen=True, slots=True)
class VoiceRuntimeWarmupResult:
    """
    STT warmup result.
    """

    completed: bool
    transcript_returned: bool
    error: str | None


def get_voice_runtime_profile(
    preset: VoiceRuntimePreset | str,
) -> VoiceRuntimeProfile:
    """
    Return a copy-safe profile by preset name.
    """

    clean_preset = (
        preset
        if isinstance(preset, VoiceRuntimePreset)
        else VoiceRuntimePreset(str(preset).strip().lower())
    )

    return VOICE_RUNTIME_PROFILES[clean_preset]


def build_vad_adapter(
    profile: VoiceRuntimeProfile,
) -> EnergyVoiceActivityAdapter:
    """
    Build a tuned VAD adapter from a runtime profile.
    """

    profile.validate()

    return EnergyVoiceActivityAdapter(
        config=EnergyVoiceActivityConfig(
            speech_rms_threshold=profile.vad_threshold,
            silence_rms_threshold=profile.vad_silence_threshold,
            speech_start_frames=profile.speech_start_frames,
            speech_end_frames=profile.speech_end_frames,
            min_zero_crossing_rate=0.0,
            max_zero_crossing_rate=1.0,
            adaptive_noise_floor=profile.adaptive_vad,
        )
    )


def build_stt_adapter(
    profile: VoiceRuntimeProfile,
) -> RealSpeechToTextAdapter:
    """
    Build a tuned real STT adapter from a runtime profile.
    """

    profile.validate()

    return RealSpeechToTextAdapter(
        config=RealSpeechToTextConfig(
            model_size=profile.stt_model,
            device="cpu",
            compute_type="int8",
            language="en",
        )
    )


def build_full_voice_smoke_config(
    profile: VoiceRuntimeProfile,
    *,
    require_wake: bool,
    keep_listening: bool,
    duration_seconds: float | None = None,
    response_text: str | None = None,
) -> FullVoiceSmokeConfig:
    """
    Build a full voice smoke config from a runtime profile.
    """

    profile.validate()

    return FullVoiceSmokeConfig(
        duration_seconds=duration_seconds or profile.duration_seconds,
        max_frames=profile.max_frames,
        require_wake=require_wake,
        stop_after_first_turn=not keep_listening,
        response_text=response_text or profile.response_text,
    )


def warm_stt_model(
    stt: WarmableSpeechToTextAdapter,
) -> VoiceRuntimeWarmupResult:
    """
    Warm the STT model before microphone capture starts.

    Empty transcript is acceptable. Errors are returned instead of raised so
    the script can print a clear failure message.
    """

    try:
        transcript = stt.transcribe(WarmupSpeechSegment())

    except Exception as exc:
        return VoiceRuntimeWarmupResult(
            completed=False,
            transcript_returned=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    return VoiceRuntimeWarmupResult(
        completed=True,
        transcript_returned=transcript is not None,
        error=None,
    )


def format_full_voice_report(report: FullVoiceSmokeReport) -> str:
    """
    Format a human-readable full voice smoke summary.
    """

    lines = [
        "",
        "JARVIS Tuned Full Voice Smoke",
        "-----------------------------",
        f"Passed: {report.passed}",
        f"Duration: {report.duration_ms:.2f} ms",
        f"Frames read: {report.frames_read}",
        f"Wake detected: {report.wake_detected}",
        f"Speech completed: {report.speech_completed}",
        f"Turns: {report.turn_count}",
        f"Playback results: {report.playback_count}",
    ]

    for index, turn in enumerate(report.turns, start=1):
        lines.extend(
            [
                "",
                f"Turn {index}",
                f"  heard: {turn.transcript.text}",
                f"  response: {turn.response_text}",
                f"  chunks: {len(turn.chunks)}",
            ]
        )

        for result in turn.playback_results:
            lines.append(f"  playback: {result.status.value}")

    if report.errors:
        lines.append("")
        lines.append("Errors:")

        for error in report.errors:
            lines.append(f" - {error}")

    if not report.speech_completed and not report.errors:
        lines.extend(
            [
                "",
                "Tuning hint:",
                " - Speak clearly after microphone start.",
                " - Move closer to the microphone.",
                " - Try preset fast first.",
                " - Try lowering VAD threshold if speech is not detected.",
            ]
        )

    return "\n".join(lines)
