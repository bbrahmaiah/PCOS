from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Protocol, runtime_checkable
from uuid import uuid4

from jarvis.presence.adapters import PlaybackResult, PlaybackStatus
from jarvis.presence.models import SpeechChunk
from jarvis.runtime.observability.structured_logger import get_logger


@runtime_checkable
class AudioPlaybackBackend(Protocol):
    """
    Minimal backend contract for speaker playback.
    """

    @property
    def active(self) -> bool:
        """Whether audio is currently playing."""

    def play_pcm(
        self,
        *,
        audio_data: bytes,
        sample_rate: int,
        channels: int,
    ) -> None:
        """Play raw PCM audio."""

    def stop(self) -> None:
        """Stop current playback immediately."""


@dataclass(frozen=True, slots=True)
class RealAudioPlaybackConfig:
    """
    Real speaker playback configuration.
    """

    dtype: str = "int16"
    block_until_finished: bool = False
    reject_empty_audio: bool = True

    def validate(self) -> None:
        if not self.dtype.strip():
            raise ValueError("dtype cannot be empty.")


class SoundDeviceAudioPlaybackBackend:
    """
    sounddevice-backed speaker output.

    sounddevice is lazy-imported so tests and non-audio runtime paths do not
    require speaker hardware.
    """

    def __init__(
        self,
        *,
        dtype: str = "int16",
        block_until_finished: bool = False,
    ) -> None:
        self._dtype = dtype
        self._block_until_finished = block_until_finished
        self._active = False
        self._lock = Lock()
        self._logger = get_logger("presence.real_playback.sounddevice")

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    def play_pcm(
        self,
        *,
        audio_data: bytes,
        sample_rate: int,
        channels: int,
    ) -> None:
        try:
            import numpy as np
            import sounddevice as sd  # type: ignore[import-untyped] 

        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "sounddevice and numpy are required for "
                "RealAudioPlaybackAdapter. Install with: pip install sounddevice numpy"
            ) from exc

        if sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero.")

        if channels <= 0:
            raise ValueError("channels must be greater than zero.")

        with self._lock:
            self._active = True

        try:
            samples = np.frombuffer(audio_data, dtype=np.int16)

            if channels > 1:
                samples = samples.reshape(-1, channels)

            sd.play(
                samples,
                samplerate=sample_rate,
                blocking=self._block_until_finished,
            )

            if self._block_until_finished:
                with self._lock:
                    self._active = False

        except Exception:
            with self._lock:
                self._active = False
            raise

        self._logger.info(
            "sounddevice_playback_started",
            sample_rate=sample_rate,
            channels=channels,
            byte_count=len(audio_data),
        )

    def stop(self) -> None:
        try:
            import sounddevice as sd

        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "sounddevice is required for RealAudioPlaybackAdapter. "
                "Install it with: pip install sounddevice"
            ) from exc

        sd.stop()

        with self._lock:
            self._active = False

        self._logger.info("sounddevice_playback_stopped")


class RealAudioPlaybackAdapter:
    """
    Production playback adapter for Presence.

    Responsibilities:
    - consume SpeechChunk audio
    - play PCM audio through speaker backend
    - stop playback immediately on interruption

    Non-responsibilities:
    - no TTS synthesis
    - no event publishing
    - no state transitions
    """

    def __init__(
        self,
        *,
        config: RealAudioPlaybackConfig | None = None,
        backend: AudioPlaybackBackend | None = None,
    ) -> None:
        self._config = config or RealAudioPlaybackConfig()
        self._config.validate()

        self._backend = backend or SoundDeviceAudioPlaybackBackend(
            dtype=self._config.dtype,
            block_until_finished=self._config.block_until_finished,
        )

        self._lock = Lock()
        self._play_count = 0
        self._stop_count = 0
        self._last_chunk_id: str | None = None
        self._last_error: str | None = None

        self._logger = get_logger("presence.real_playback")

    @property
    def is_playing(self) -> bool:
        return self._backend.active

    @property
    def play_count(self) -> int:
        with self._lock:
            return self._play_count

    @property
    def stop_count(self) -> int:
        with self._lock:
            return self._stop_count

    @property
    def last_chunk_id(self) -> str | None:
        with self._lock:
            return self._last_chunk_id

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    def reset(self) -> None:
        with self._lock:
            self._play_count = 0
            self._stop_count = 0
            self._last_chunk_id = None
            self._last_error = None

        self._logger.info("real_playback_reset")

    def play(self, chunk: SpeechChunk) -> PlaybackResult:
        if self._config.reject_empty_audio and not chunk.audio_data:
            return PlaybackResult(
                result_id=uuid4().hex,
                chunk_id=chunk.chunk_id,
                request_id=chunk.request_id,
                status=PlaybackStatus.FAILED,
                error="Empty audio cannot be played.",
                metadata={
                    "adapter": "real_playback",
                    "reason": "empty_audio",
                },
            )

        try:
            self._backend.play_pcm(
                audio_data=chunk.audio_data,
                sample_rate=chunk.sample_rate,
                channels=chunk.channels,
            )

        except Exception as exc:
            self._record_error(exc)

            return PlaybackResult(
                result_id=uuid4().hex,
                chunk_id=chunk.chunk_id,
                request_id=chunk.request_id,
                status=PlaybackStatus.FAILED,
                error=f"{type(exc).__name__}: {exc}",
                metadata={
                    "adapter": "real_playback",
                },
            )

        with self._lock:
            self._play_count += 1
            self._last_chunk_id = chunk.chunk_id
            self._last_error = None

        self._logger.info(
            "real_playback_started",
            request_id=chunk.request_id,
            chunk_id=chunk.chunk_id,
            sample_rate=chunk.sample_rate,
            channels=chunk.channels,
            byte_count=len(chunk.audio_data),
        )

        return PlaybackResult(
            result_id=uuid4().hex,
            chunk_id=chunk.chunk_id,
            request_id=chunk.request_id,
            status=PlaybackStatus.STARTED,
            metadata={
                "adapter": "real_playback",
                "backend": type(self._backend).__name__,
            },
        )

    def stop(self) -> None:
        try:
            self._backend.stop()

        except Exception as exc:
            self._record_error(exc)
            raise

        with self._lock:
            self._stop_count += 1

        self._logger.info("real_playback_stopped")

    def _record_error(self, exc: Exception) -> None:
        error = f"{type(exc).__name__}: {exc}"

        with self._lock:
            self._last_error = error

        self._logger.error("real_playback_error", error=error)