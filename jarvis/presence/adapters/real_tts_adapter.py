from __future__ import annotations

import io
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from jarvis.presence.models import SpeechChunk
from jarvis.runtime.observability.structured_logger import get_logger


@runtime_checkable
class TextToSpeechBackend(Protocol):
    """
    Minimal backend contract for real TTS.

    Backend returns PCM WAV bytes. The adapter converts them into SpeechChunk.
    """

    def synthesize_to_wav(
        self,
        *,
        text: str,
        voice_id: str,
        rate: int,
        volume: float,
    ) -> bytes:
        """Synthesize text into WAV bytes."""


@dataclass(frozen=True, slots=True)
class RealTextToSpeechConfig:
    """
    Configuration for local real TTS.
    """

    voice_id: str = "default"
    rate: int = 180
    volume: float = 1.0
    sample_rate_fallback: int = 22_050
    channels_fallback: int = 1
    max_text_chars: int = 4_000

    def validate(self) -> None:
        if not self.voice_id.strip():
            raise ValueError("voice_id cannot be empty.")

        if self.rate <= 0:
            raise ValueError("rate must be greater than zero.")

        if not 0.0 <= self.volume <= 1.0:
            raise ValueError("volume must be between 0.0 and 1.0.")

        if self.sample_rate_fallback <= 0:
            raise ValueError("sample_rate_fallback must be greater than zero.")

        if self.channels_fallback <= 0:
            raise ValueError("channels_fallback must be greater than zero.")

        if self.max_text_chars <= 0:
            raise ValueError("max_text_chars must be greater than zero.")


class Pyttsx3TextToSpeechBackend:
    """
    pyttsx3 local/offline TTS backend.

    pyttsx3 is lazy-imported and initialized only when synthesis is requested.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._logger = get_logger("presence.real_tts.pyttsx3")

    def synthesize_to_wav(
        self,
        *,
        text: str,
        voice_id: str,
        rate: int,
        volume: float,
    ) -> bytes:
        try:
            import pyttsx3  # type: ignore[import-untyped]

        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "pyttsx3 is required for RealTextToSpeechAdapter. "
                "Install it with: pip install pyttsx3"
            ) from exc

        with self._lock:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / "jarvis_tts.wav"

                engine = pyttsx3.init()
                engine.setProperty("rate", rate)
                engine.setProperty("volume", volume)

                if voice_id != "default":
                    self._try_set_voice(
                        engine=engine,
                        voice_id=voice_id,
                    )

                engine.save_to_file(text, str(output_path))
                engine.runAndWait()
                engine.stop()

                wav_bytes = output_path.read_bytes()

        self._logger.info(
            "pyttsx3_tts_completed",
            text_length=len(text),
            byte_count=len(wav_bytes),
            voice_id=voice_id,
            rate=rate,
        )

        return wav_bytes

    @staticmethod
    def _try_set_voice(*, engine: Any, voice_id: str) -> None:
        voices = engine.getProperty("voices")

        for voice in voices:
            candidate_id = str(getattr(voice, "id", ""))

            if candidate_id == voice_id or voice_id.lower() in candidate_id.lower():
                engine.setProperty("voice", candidate_id)
                return


class RealTextToSpeechAdapter:
    """
    Production TTS adapter for Presence.

    Responsibilities:
    - consume text
    - synthesize speech through a backend
    - return SpeechChunk objects

    Non-responsibilities:
    - no playback
    - no microphone
    - no STT
    - no event publishing
    """

    def __init__(
        self,
        *,
        config: RealTextToSpeechConfig | None = None,
        backend: TextToSpeechBackend | None = None,
    ) -> None:
        self._config = config or RealTextToSpeechConfig()
        self._config.validate()

        self._backend = backend or Pyttsx3TextToSpeechBackend()
        self._lock = Lock()
        self._synthesis_count = 0
        self._empty_requests = 0
        self._last_text: str | None = None
        self._last_error: str | None = None

        self._logger = get_logger("presence.real_tts")

    @property
    def config(self) -> RealTextToSpeechConfig:
        return self._config

    @property
    def synthesis_count(self) -> int:
        with self._lock:
            return self._synthesis_count

    @property
    def empty_requests(self) -> int:
        with self._lock:
            return self._empty_requests

    @property
    def last_text(self) -> str | None:
        with self._lock:
            return self._last_text

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    def reset(self) -> None:
        with self._lock:
            self._synthesis_count = 0
            self._empty_requests = 0
            self._last_text = None
            self._last_error = None

        self._logger.info("real_tts_reset")

    def synthesize(
        self,
        *,
        text: str,
        request_id: str,
        voice_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[SpeechChunk, ...]:
        clean_text = text.strip()

        if not clean_text:
            with self._lock:
                self._empty_requests += 1
            return ()

        if len(clean_text) > self._config.max_text_chars:
            raise ValueError(
                f"text is too long: {len(clean_text)} > "
                f"{self._config.max_text_chars}"
            )

        selected_voice_id = voice_id or self._config.voice_id

        try:
            wav_bytes = self._backend.synthesize_to_wav(
                text=clean_text,
                voice_id=selected_voice_id,
                rate=self._config.rate,
                volume=self._config.volume,
            )
            audio_data, sample_rate, channels = wav_bytes_to_pcm(wav_bytes)

        except Exception as exc:
            self._record_error(exc)
            raise

        chunk = SpeechChunk(
            request_id=request_id,
            chunk_id=uuid4().hex,
            audio_data=audio_data,
            sample_rate=sample_rate or self._config.sample_rate_fallback,
            channels=channels or self._config.channels_fallback,
            final=True,
            metadata={
                **(metadata or {}),
                "adapter": "real_tts",
                "backend": type(self._backend).__name__,
                "text": clean_text,
                "voice_id": selected_voice_id,
                "sequence_index": 0,
                "wav_byte_count": len(wav_bytes),
                "pcm_byte_count": len(audio_data),
            },
        )

        with self._lock:
            self._synthesis_count += 1
            self._last_text = clean_text
            self._last_error = None

        self._logger.info(
            "real_tts_synthesis_completed",
            request_id=request_id,
            chunk_id=chunk.chunk_id,
            text_length=len(clean_text),
            sample_rate=chunk.sample_rate,
            channels=chunk.channels,
            byte_count=len(chunk.audio_data),
        )

        return (chunk,)

    def _record_error(self, exc: Exception) -> None:
        error = f"{type(exc).__name__}: {exc}"

        with self._lock:
            self._last_error = error

        self._logger.error("real_tts_error", error=error)


def wav_bytes_to_pcm(wav_bytes: bytes) -> tuple[bytes, int, int]:
    """
    Convert WAV bytes into raw PCM audio, sample_rate, channels.
    """

    if not wav_bytes:
        raise ValueError("wav_bytes cannot be empty.")

    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        audio_data = wav_file.readframes(frame_count)

    return audio_data, sample_rate, channels


def pcm_to_wav_bytes(
    *,
    audio_data: bytes,
    sample_rate: int,
    channels: int,
    sample_width_bytes: int = 2,
) -> bytes:
    """
    Wrap raw PCM bytes into WAV bytes.
    Useful for tests and local TTS backends.
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