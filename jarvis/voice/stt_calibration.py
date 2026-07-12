from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from jarvis.voice.contracts import VoiceInputFrame, utc_now
from jarvis.voice.stt_runtime import VoiceSTTResult
from jarvis.voice.transcript_attention_gate import normalize_transcript_text


class VoiceSTTCalibrationScenarioKind(StrEnum):
    SILENCE = "silence"
    NOISE = "noise"
    WAKE_WORD = "wake_word"


class VoiceSTTCalibrationStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class VoiceSTTCalibrationPolicy:
    wake_words: tuple[str, ...] = ("jarvis", "jervis", "jarves")
    max_false_transcripts: int = 0
    min_wake_confidence: float = 0.70
    min_wake_acceptance_rate: float = 0.90
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.wake_words:
            raise ValueError("wake_words cannot be empty.")
        if self.max_false_transcripts < 0:
            raise ValueError("max_false_transcripts cannot be negative.")
        if not 0.0 <= self.min_wake_confidence <= 1.0:
            raise ValueError("min_wake_confidence must be 0..1.")
        if not 0.0 <= self.min_wake_acceptance_rate <= 1.0:
            raise ValueError("min_wake_acceptance_rate must be 0..1.")


@dataclass(frozen=True, slots=True)
class VoiceSTTCalibrationSample:
    kind: VoiceSTTCalibrationScenarioKind
    label: str
    frames: tuple[VoiceInputFrame, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("calibration sample label cannot be empty.")
        if not self.frames:
            raise ValueError("calibration sample frames cannot be empty.")


@dataclass(frozen=True, slots=True)
class VoiceSTTCalibrationSampleResult:
    sample: VoiceSTTCalibrationSample
    passed: bool
    transcript_text: str | None
    confidence: float | None
    rejection_reason: str | None
    message: str
    stt_result: VoiceSTTResult
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VoiceSTTCalibrationReport:
    status: VoiceSTTCalibrationStatus
    sample_results: tuple[VoiceSTTCalibrationSampleResult, ...]
    false_transcripts: int
    wake_attempts: int
    wake_passes: int
    wake_acceptance_rate: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == VoiceSTTCalibrationStatus.PASSED


@dataclass(frozen=True, slots=True)
class VoiceCaptureProfile:
    vad_min_energy: float
    vad_speech_start_ratio: float
    vad_start_trigger_frames: int
    vad_min_speech_ms: int
    vad_end_silence_ms: int
    stt_min_partial_confidence: float
    stt_min_final_confidence: float
    stt_max_no_speech_prob: float
    stt_min_avg_logprob: float
    stt_max_compression_ratio: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.vad_min_energy <= 0:
            raise ValueError("vad_min_energy must be positive.")
        if self.vad_speech_start_ratio <= 1.0:
            raise ValueError("vad_speech_start_ratio must be greater than 1.")
        if self.vad_start_trigger_frames < 1:
            raise ValueError("vad_start_trigger_frames must be positive.")
        if self.vad_min_speech_ms <= 0:
            raise ValueError("vad_min_speech_ms must be positive.")
        if self.vad_end_silence_ms <= 0:
            raise ValueError("vad_end_silence_ms must be positive.")
        if not 0.0 <= self.stt_min_partial_confidence <= 1.0:
            raise ValueError("stt_min_partial_confidence must be 0..1.")
        if not 0.0 <= self.stt_min_final_confidence <= 1.0:
            raise ValueError("stt_min_final_confidence must be 0..1.")
        if not 0.0 <= self.stt_max_no_speech_prob <= 1.0:
            raise ValueError("stt_max_no_speech_prob must be 0..1.")
        if self.stt_max_compression_ratio <= 0:
            raise ValueError("stt_max_compression_ratio must be positive.")

    def to_dict(self) -> dict[str, object]:
        return {
            "vad_min_energy": self.vad_min_energy,
            "vad_speech_start_ratio": self.vad_speech_start_ratio,
            "vad_start_trigger_frames": self.vad_start_trigger_frames,
            "vad_min_speech_ms": self.vad_min_speech_ms,
            "vad_end_silence_ms": self.vad_end_silence_ms,
            "stt_min_partial_confidence": self.stt_min_partial_confidence,
            "stt_min_final_confidence": self.stt_min_final_confidence,
            "stt_max_no_speech_prob": self.stt_max_no_speech_prob,
            "stt_min_avg_logprob": self.stt_min_avg_logprob,
            "stt_max_compression_ratio": self.stt_max_compression_ratio,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> VoiceCaptureProfile:
        created_at_raw = data.get("created_at")
        created_at = (
            datetime.fromisoformat(created_at_raw)
            if isinstance(created_at_raw, str) and created_at_raw.strip()
            else utc_now()
        )
        metadata = data.get("metadata")
        return cls(
            vad_min_energy=_float_field(data, "vad_min_energy"),
            vad_speech_start_ratio=_float_field(data, "vad_speech_start_ratio"),
            vad_start_trigger_frames=_int_field(data, "vad_start_trigger_frames"),
            vad_min_speech_ms=_int_field(data, "vad_min_speech_ms"),
            vad_end_silence_ms=_int_field(data, "vad_end_silence_ms"),
            stt_min_partial_confidence=_float_field(
                data,
                "stt_min_partial_confidence",
            ),
            stt_min_final_confidence=_float_field(
                data,
                "stt_min_final_confidence",
            ),
            stt_max_no_speech_prob=_float_field(data, "stt_max_no_speech_prob"),
            stt_min_avg_logprob=_float_field(data, "stt_min_avg_logprob"),
            stt_max_compression_ratio=_float_field(
                data,
                "stt_max_compression_ratio",
            ),
            created_at=created_at,
            metadata=metadata if isinstance(metadata, dict) else {},
        )


class VoiceSTTCalibrationSTT(Protocol):
    def transcribe_final(
        self,
        frames: tuple[VoiceInputFrame, ...],
        *,
        allow_action_candidate: bool = False,
    ) -> VoiceSTTResult:
        raise NotImplementedError


class VoiceSTTCalibrationRuntime:
    """
    Live-room STT calibration gate.

    This runtime does not tune thresholds by guessing. It measures the current
    audio setup and reports whether silence/noise are leaking accepted
    transcripts and whether the wake word is reliably recognized.
    """

    def __init__(
        self,
        *,
        stt: VoiceSTTCalibrationSTT,
        policy: VoiceSTTCalibrationPolicy | None = None,
    ) -> None:
        self._stt = stt
        self._policy = policy or VoiceSTTCalibrationPolicy()

    def run_samples(
        self,
        samples: tuple[VoiceSTTCalibrationSample, ...],
    ) -> VoiceSTTCalibrationReport:
        if not samples:
            raise ValueError("STT calibration requires at least one sample.")

        sample_results = tuple(self._evaluate_sample(sample) for sample in samples)
        false_transcripts = sum(
            1
            for result in sample_results
            if result.sample.kind
            in {
                VoiceSTTCalibrationScenarioKind.SILENCE,
                VoiceSTTCalibrationScenarioKind.NOISE,
            }
            and result.transcript_text is not None
        )
        wake_results = tuple(
            result
            for result in sample_results
            if result.sample.kind == VoiceSTTCalibrationScenarioKind.WAKE_WORD
        )
        wake_attempts = len(wake_results)
        wake_passes = sum(1 for result in wake_results if result.passed)
        wake_acceptance_rate = (
            1.0 if wake_attempts == 0 else wake_passes / wake_attempts
        )
        passed = (
            all(result.passed for result in sample_results)
            and false_transcripts <= self._policy.max_false_transcripts
            and wake_acceptance_rate >= self._policy.min_wake_acceptance_rate
        )
        return VoiceSTTCalibrationReport(
            status=(
                VoiceSTTCalibrationStatus.PASSED
                if passed
                else VoiceSTTCalibrationStatus.FAILED
            ),
            sample_results=sample_results,
            false_transcripts=false_transcripts,
            wake_attempts=wake_attempts,
            wake_passes=wake_passes,
            wake_acceptance_rate=wake_acceptance_rate,
            created_at=utc_now(),
            metadata={
                "wake_words": self._policy.wake_words,
                "max_false_transcripts": self._policy.max_false_transcripts,
                "min_wake_confidence": self._policy.min_wake_confidence,
                "min_wake_acceptance_rate": self._policy.min_wake_acceptance_rate,
                **self._policy.metadata,
            },
        )

    def _evaluate_sample(
        self,
        sample: VoiceSTTCalibrationSample,
    ) -> VoiceSTTCalibrationSampleResult:
        result = self._stt.transcribe_final(
            sample.frames,
            allow_action_candidate=False,
        )
        transcript = result.transcript
        transcript_text = None if transcript is None else transcript.text
        confidence = None if transcript is None else transcript.confidence
        rejection_reason = _rejection_reason(result)

        if sample.kind in {
            VoiceSTTCalibrationScenarioKind.SILENCE,
            VoiceSTTCalibrationScenarioKind.NOISE,
        }:
            passed = transcript is None
            message = (
                "no accepted transcript"
                if passed
                else "false transcript accepted during non-speech sample"
            )
        else:
            passed = (
                transcript is not None
                and transcript.confidence >= self._policy.min_wake_confidence
                and _contains_wake_word(transcript.text, self._policy.wake_words)
            )
            message = (
                "wake word accepted"
                if passed
                else "wake word was not accepted with enough confidence"
            )

        return VoiceSTTCalibrationSampleResult(
            sample=sample,
            passed=passed,
            transcript_text=transcript_text,
            confidence=confidence,
            rejection_reason=rejection_reason,
            message=message,
            stt_result=result,
            created_at=utc_now(),
            metadata={
                "stt_status": result.status.value,
                "stt_message": result.message,
                **result.metadata,
            },
        )


def _contains_wake_word(text: str, wake_words: tuple[str, ...]) -> bool:
    normalized = f" {normalize_transcript_text(text)} "
    return any(
        f" {normalize_transcript_text(wake_word)} " in normalized
        for wake_word in wake_words
    )


def _rejection_reason(result: VoiceSTTResult) -> str | None:
    value = result.metadata.get("rejection_reason")
    if isinstance(value, str) and value.strip():
        return value
    return None


def build_voice_capture_profile(
    report: VoiceSTTCalibrationReport,
) -> VoiceCaptureProfile:
    silence_energies = _sample_energies(
        sample.frames
        for sample_result in report.sample_results
        if sample_result.sample.kind
        in {
            VoiceSTTCalibrationScenarioKind.SILENCE,
            VoiceSTTCalibrationScenarioKind.NOISE,
        }
        for sample in (sample_result.sample,)
    )
    wake_energies = _sample_energies(
        sample.frames
        for sample_result in report.sample_results
        if sample_result.sample.kind == VoiceSTTCalibrationScenarioKind.WAKE_WORD
        for sample in (sample_result.sample,)
    )

    noise_floor = _percentile(silence_energies, 0.90, default=220.0)
    wake_floor = _percentile(wake_energies, 0.30, default=noise_floor * 5.0)
    vad_min_energy = max(450.0, noise_floor * 3.2)
    if wake_floor > noise_floor:
        vad_min_energy = min(vad_min_energy, max(450.0, wake_floor * 0.45))

    false_transcripts = report.false_transcripts
    wake_rate = report.wake_acceptance_rate
    strict = false_transcripts > 0 or wake_rate < 0.90

    return VoiceCaptureProfile(
        vad_min_energy=round(vad_min_energy, 2),
        vad_speech_start_ratio=4.0 if strict else 3.2,
        vad_start_trigger_frames=6 if strict else 4,
        vad_min_speech_ms=420 if strict else 320,
        vad_end_silence_ms=780 if strict else 700,
        stt_min_partial_confidence=0.45 if strict else 0.40,
        stt_min_final_confidence=0.76 if strict else 0.70,
        stt_max_no_speech_prob=0.45 if strict else 0.55,
        stt_min_avg_logprob=-1.05 if strict else -1.15,
        stt_max_compression_ratio=2.20 if strict else 2.35,
        created_at=utc_now(),
        metadata={
            "source": "stt_calibration",
            "calibration_status": report.status.value,
            "false_transcripts": false_transcripts,
            "wake_attempts": report.wake_attempts,
            "wake_passes": report.wake_passes,
            "wake_acceptance_rate": wake_rate,
            "noise_energy_p90": noise_floor,
            "wake_energy_p30": wake_floor,
            "strict_profile": strict,
        },
    )


def save_voice_capture_profile(
    profile: VoiceCaptureProfile,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(profile.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_voice_capture_profile(path: Path) -> VoiceCaptureProfile | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("voice capture profile must be a JSON object.")
    return VoiceCaptureProfile.from_dict(payload)


def _sample_energies(
    sample_frame_groups: Iterable[tuple[VoiceInputFrame, ...]],
) -> tuple[float, ...]:
    energies: list[float] = []
    for frames in sample_frame_groups:
        for frame in frames:
            try:
                energies.append(_pcm16_rms(frame.data))
            except ValueError:
                continue
    return tuple(energies)


def _pcm16_rms(data: bytes) -> float:
    if not data:
        raise ValueError("audio data cannot be empty.")
    if len(data) % 2 != 0:
        raise ValueError("pcm16 audio data length must be even.")

    total = 0
    samples = len(data) // 2
    for index in range(0, len(data), 2):
        sample = int.from_bytes(
            data[index:index + 2],
            byteorder="little",
            signed=True,
        )
        total += sample * sample
    return math.sqrt(total / samples)


def _percentile(
    values: tuple[float, ...],
    percentile: float,
    *,
    default: float,
) -> float:
    if not values:
        return default
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, int(round((len(ordered) - 1) * percentile))),
    )
    return ordered[index]


def _float_field(data: dict[str, object], key: str) -> float:
    value = data.get(key)
    if not isinstance(value, int | float):
        raise ValueError(f"{key} must be numeric.")
    return float(value)


def _int_field(data: dict[str, object], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer.")
    return value
