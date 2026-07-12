from __future__ import annotations

import io
import math
import wave
from dataclasses import dataclass
from threading import Lock
from typing import Any, Protocol, runtime_checkable

from jarvis.presence.models.transcript import Transcript, TranscriptKind
from jarvis.runtime.observability.structured_logger import get_logger


@runtime_checkable
class SpeechAudioSegment(Protocol):
    """
    Minimal speech-segment contract required by RealSpeechToTextAdapter.

    Read-only properties keep the adapter compatible with frozen dataclasses,
    Pydantic models, and mutable test stubs. STT only reads segment data; it
    must never mutate the segment.
    """

    @property
    def segment_id(self) -> str:
        """Stable segment identifier."""

    @property
    def audio_data(self) -> bytes:
        """Raw PCM audio bytes."""

    @property
    def sample_rate(self) -> int:
        """Audio sample rate."""

    @property
    def channels(self) -> int:
        """Audio channel count."""

    @property
    def metadata(self) -> dict[str, Any]:
        """Segment metadata."""


@runtime_checkable
class SpeechToTextBackend(Protocol):
    """
    Minimal backend contract for real STT.
    """

    def transcribe_pcm(
        self,
        *,
        audio_data: bytes,
        sample_rate: int,
        channels: int,
        language: str | None,
        prompt: str | None,
    ) -> tuple[str, float, str | None, dict[str, Any]]:
        """
        Transcribe raw PCM audio and return:
        text, confidence, language, metadata.
        """


@dataclass(frozen=True, slots=True)
class RealSpeechToTextConfig:
    """
    Configuration for local real STT.
    """

    model_size: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str | None = "en"
    beam_size: int = 1
    vad_filter: bool = False
    temperature: float = 0.0
    min_text_length: int = 1
    transcript_kind: TranscriptKind = TranscriptKind.FINAL

    def validate(self) -> None:
        if not self.model_size.strip():
            raise ValueError("model_size cannot be empty.")

        if not self.device.strip():
            raise ValueError("device cannot be empty.")

        if not self.compute_type.strip():
            raise ValueError("compute_type cannot be empty.")

        if self.beam_size <= 0:
            raise ValueError("beam_size must be greater than zero.")

        if self.temperature < 0:
            raise ValueError("temperature cannot be negative.")

        if self.min_text_length < 0:
            raise ValueError("min_text_length cannot be negative.")


class FasterWhisperSpeechToTextBackend:
    """
    faster-whisper backend.

    Import and model loading are lazy. This prevents import-time crashes and
    keeps the rest of JARVIS usable even before STT dependencies are installed.
    """

    def __init__(self, config: RealSpeechToTextConfig) -> None:
        self._config = config
        self._model: Any | None = None
        self._lock = Lock()
        self._logger = get_logger("presence.real_stt.faster_whisper")

    def transcribe_pcm(
        self,
        *,
        audio_data: bytes,
        sample_rate: int,
        channels: int,
        language: str | None,
        prompt: str | None,
    ) -> tuple[str, float, str | None, dict[str, Any]]:
        model = self._get_model()
        wav_bytes = pcm_to_wav_bytes(
            audio_data=audio_data,
            sample_rate=sample_rate,
            channels=channels,
        )

        segments, info = model.transcribe(
            io.BytesIO(wav_bytes),
            language=language,
            beam_size=self._config.beam_size,
            vad_filter=self._config.vad_filter,
            temperature=self._config.temperature,
            initial_prompt=prompt,
        )

        texts: list[str] = []
        probabilities: list[float] = []
        segment_count = 0

        for segment in segments:
            segment_count += 1
            no_speech_prob = float(getattr(segment, "no_speech_prob", 0.0))
            if no_speech_prob > 0.6:
                continue
            text = str(getattr(segment, "text", "")).strip()
            if len(text.split()) < 2:
                continue
            if text:
                texts.append(text)
            avg_logprob = float(getattr(segment, "avg_logprob", 0.0))
            probabilities.append(log_probability_to_confidence(avg_logprob))

        joined_text = " ".join(texts).strip()
        confidence = (
            sum(probabilities) / len(probabilities)
            if probabilities
            else 0.0
        )
        detected_language = getattr(info, "language", None)

        metadata = {
            "backend": "faster-whisper",
            "model_size": self._config.model_size,
            "device": self._config.device,
            "compute_type": self._config.compute_type,
            "segment_count": segment_count,
            "language_probability": float(
                getattr(info, "language_probability", 0.0)
            ),
        }

        self._logger.info(
            "faster_whisper_transcription_completed",
            text_length=len(joined_text),
            confidence=round(confidence, 4),
            language=detected_language,
            segment_count=segment_count,
        )

        return joined_text, confidence, detected_language, metadata

    def _get_model(self) -> Any:
        with self._lock:
            if self._model is not None:
                return self._model

            try:
                from faster_whisper import WhisperModel  # type: ignore[import-untyped]

            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "faster-whisper is required for RealSpeechToTextAdapter. "
                    "Install it with: pip install faster-whisper"
                ) from exc

            self._model = WhisperModel(
                self._config.model_size,
                device=self._config.device,
                compute_type=self._config.compute_type,
            )

            self._logger.info(
                "faster_whisper_model_loaded",
                model_size=self._config.model_size,
                device=self._config.device,
                compute_type=self._config.compute_type,
            )

            return self._model


class RealSpeechToTextAdapter:
    """
    Production speech-to-text adapter for Presence.

    Responsibilities:
    - consume completed speech-segment-like audio objects
    - call a real STT backend
    - return Transcript objects
    - stay backend-injectable for tests

    Non-responsibilities:
    - no microphone capture
    - no wake detection
    - no VAD
    - no event publishing
    - no cognition
    """

    def __init__(
        self,
        *,
        config: RealSpeechToTextConfig | None = None,
        backend: SpeechToTextBackend | None = None,
    ) -> None:
        self._config = config or RealSpeechToTextConfig()
        self._config.validate()

        self._backend = backend or FasterWhisperSpeechToTextBackend(self._config)

        self._lock = Lock()
        self._transcription_count = 0
        self._empty_transcriptions = 0
        self._last_text: str | None = None
        self._last_confidence: float | None = None
        self._last_error: str | None = None

        self._logger = get_logger("presence.real_stt")

    @property
    def config(self) -> RealSpeechToTextConfig:
        return self._config

    @property
    def transcription_count(self) -> int:
        with self._lock:
            return self._transcription_count

    @property
    def empty_transcriptions(self) -> int:
        with self._lock:
            return self._empty_transcriptions

    @property
    def last_text(self) -> str | None:
        with self._lock:
            return self._last_text

    @property
    def last_confidence(self) -> float | None:
        with self._lock:
            return self._last_confidence

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    def reset(self) -> None:
        with self._lock:
            self._transcription_count = 0
            self._empty_transcriptions = 0
            self._last_text = None
            self._last_confidence = None
            self._last_error = None

        self._logger.info("real_stt_reset")

    def transcribe(self, segment: SpeechAudioSegment) -> Transcript | None:
        """
        Transcribe one completed speech segment.
        """

        if not segment.audio_data:
            self._record_empty()
            return None

        try:
            text, confidence, language, metadata = self._backend.transcribe_pcm(
                audio_data=segment.audio_data,
                sample_rate=segment.sample_rate,
                channels=segment.channels,
                language=self._config.language,
                prompt=self._extract_prompt(segment),
            )

        except Exception as exc:
            self._record_error(exc)
            raise

        clean_text = text.strip()

        if len(clean_text) < self._config.min_text_length:
            self._record_empty()
            return None

        transcript_language = language or self._config.language or "unknown"

        transcript = Transcript(
            segment_id=segment.segment_id,
            text=clean_text,
            kind=self._config.transcript_kind,
            confidence=confidence,
            language=transcript_language,
            metadata={
                **metadata,
                "adapter": "real_stt",
                "sample_rate": segment.sample_rate,
                "channels": segment.channels,
                "byte_count": len(segment.audio_data),
            },
        )

        with self._lock:
            self._transcription_count += 1
            self._last_text = clean_text
            self._last_confidence = confidence
            self._last_error = None

        self._logger.info(
            "real_stt_transcription_completed",
            segment_id=segment.segment_id,
            text_length=len(clean_text),
            confidence=round(confidence, 4),
            language=transcript.language,
        )

        return transcript

    def _record_empty(self) -> None:
        with self._lock:
            self._empty_transcriptions += 1
            self._last_text = None
            self._last_confidence = None
            self._last_error = None

    def _record_error(self, exc: Exception) -> None:
        error = f"{type(exc).__name__}: {exc}"

        with self._lock:
            self._last_error = error

        self._logger.error(
            "real_stt_error",
            error=error,
        )

    @staticmethod
    def _extract_prompt(segment: SpeechAudioSegment) -> str | None:
        prompt = segment.metadata.get("prompt")

        if isinstance(prompt, str) and prompt.strip():
            return prompt

        return None


def pcm_to_wav_bytes(
    *,
    audio_data: bytes,
    sample_rate: int,
    channels: int,
    sample_width_bytes: int = 2,
) -> bytes:
    """
    Wrap raw PCM bytes into an in-memory WAV container.
    """

    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than zero.")

    if channels <= 0:
        raise ValueError("channels must be greater than zero.")

    if sample_width_bytes <= 0:
        raise ValueError("sample_width_bytes must be greater than zero.")

    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width_bytes)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)

    return buffer.getvalue()


def log_probability_to_confidence(avg_logprob: float) -> float:
    """
    Convert Whisper average log probability to a bounded confidence estimate.
    """

    try:
        probability = math.exp(avg_logprob)
    except OverflowError:
        return 0.0

    return max(0.0, min(1.0, probability))