from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from typing import Any, Protocol

from jarvis.presence.models import AudioFrame
from jarvis.runtime.observability.structured_logger import get_logger

AudioCallback = Callable[[bytes], None]


class RawMicrophoneBackend(Protocol):
    @property
    def active(self) -> bool:
        """Whether the backend stream is active."""

    def start(self, callback: AudioCallback) -> None:
        """Start raw audio capture."""

    def stop(self) -> None:
        """Stop raw audio capture."""

    def close(self) -> None:
        """Release backend resources."""


@dataclass(frozen=True, slots=True)
class RealMicrophoneConfig:
    sample_rate: int = 16_000
    channels: int = 1
    frame_duration_ms: int = 30
    dtype: str = "int16"
    device: int | str | None = None
    queue_max_frames: int = 128
    overflow_policy: str = "drop_oldest"

    @property
    def samples_per_frame(self) -> int:
        return int(self.sample_rate * (self.frame_duration_ms / 1000.0))

    @property
    def bytes_per_sample(self) -> int:
        if self.dtype == "int16":
            return 2

        if self.dtype == "float32":
            return 4

        raise ValueError(f"Unsupported microphone dtype: {self.dtype!r}")

    @property
    def expected_frame_bytes(self) -> int:
        return self.samples_per_frame * self.channels * self.bytes_per_sample

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero.")

        if self.channels <= 0:
            raise ValueError("channels must be greater than zero.")

        if self.frame_duration_ms <= 0:
            raise ValueError("frame_duration_ms must be greater than zero.")

        if self.queue_max_frames <= 0:
            raise ValueError("queue_max_frames must be greater than zero.")

        if self.overflow_policy not in {"drop_oldest", "drop_newest", "raise"}:
            raise ValueError(
                "overflow_policy must be one of: "
                "'drop_oldest', 'drop_newest', 'raise'."
            )

        _ = self.bytes_per_sample


class SoundDeviceRawMicrophoneBackend:
    def __init__(self, config: RealMicrophoneConfig) -> None:
        self._config = config
        self._stream: Any | None = None
        self._callback: AudioCallback | None = None
        self._lock = Lock()
        self._logger = get_logger("presence.real_microphone.sounddevice")

    @property
    def active(self) -> bool:
        with self._lock:
            stream = self._stream

        if stream is None:
            return False

        return bool(getattr(stream, "active", False))

    def start(self, callback: AudioCallback) -> None:
        with self._lock:
            if self._stream is not None and self.active:
                return

            self._callback = callback
            self._stream = self._create_stream()
            self._stream.start()

        self._logger.info(
            "sounddevice_microphone_started",
            sample_rate=self._config.sample_rate,
            channels=self._config.channels,
            frame_duration_ms=self._config.frame_duration_ms,
            device=self._config.device,
        )

    def stop(self) -> None:
        with self._lock:
            stream = self._stream

        if stream is None:
            return

        try:
            stream.stop()
        finally:
            self._logger.info("sounddevice_microphone_stopped")

    def close(self) -> None:
        with self._lock:
            stream = self._stream
            self._stream = None
            self._callback = None

        if stream is None:
            return

        stream.close()
        self._logger.info("sounddevice_microphone_closed")

    def _create_stream(self) -> Any:
        try:
            import sounddevice as sd  # type: ignore[import-untyped]

        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "sounddevice is required for RealMicrophoneAdapter. "
                "Install it with: pip install sounddevice"
            ) from exc

        return sd.RawInputStream(
            samplerate=self._config.sample_rate,
            blocksize=self._config.samples_per_frame,
            channels=self._config.channels,
            dtype=self._config.dtype,
            device=self._config.device,
            callback=self._on_audio,
        )

    def _on_audio(
        self,
        indata: Any,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        del frames, time_info

        if status:
            self._logger.warning(
                "sounddevice_microphone_status",
                status=str(status),
            )

        callback = self._callback

        if callback is None:
            return

        callback(bytes(indata))


class RealMicrophoneAdapter:
    def __init__(
        self,
        *,
        config: RealMicrophoneConfig | None = None,
        backend: RawMicrophoneBackend | None = None,
    ) -> None:
        self._config = config or RealMicrophoneConfig()
        self._config.validate()

        self._backend = backend or SoundDeviceRawMicrophoneBackend(self._config)

        self._lock = Lock()
        self._queue: deque[bytes] = deque()
        self._started = False
        self._closed = False
        self._frame_index = 0
        self._dropped_frames = 0
        self._captured_frames = 0
        self._read_frames = 0
        self._last_error: str | None = None

        self._logger = get_logger("presence.real_microphone")

    @property
    def config(self) -> RealMicrophoneConfig:
        return self._config

    @property
    def is_active(self) -> bool:
        return self._started and self._backend.active

    @property
    def started(self) -> bool:
        return self._started

    @property
    def dropped_frames(self) -> int:
        with self._lock:
            return self._dropped_frames

    @property
    def captured_frames(self) -> int:
        with self._lock:
            return self._captured_frames

    @property
    def read_frames(self) -> int:
        with self._lock:
            return self._read_frames

    @property
    def pending_frames(self) -> int:
        with self._lock:
            return len(self._queue)

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot start a closed microphone adapter.")

            if self._started:
                return

        try:
            self._backend.start(self._on_raw_audio)

        except Exception as exc:
            self._record_error(exc)
            raise

        with self._lock:
            self._started = True
            self._last_error = None

        self._logger.info(
            "real_microphone_started",
            sample_rate=self._config.sample_rate,
            channels=self._config.channels,
            frame_duration_ms=self._config.frame_duration_ms,
            queue_max_frames=self._config.queue_max_frames,
        )

    def stop(self) -> None:
        with self._lock:
            if not self._started:
                return

        try:
            self._backend.stop()

        except Exception as exc:
            self._record_error(exc)
            raise

        with self._lock:
            self._started = False

        self._logger.info("real_microphone_stopped")

    def close(self) -> None:
        try:
            self.stop()
            self._backend.close()

        except Exception as exc:
            self._record_error(exc)
            raise

        with self._lock:
            self._closed = True
            self._queue.clear()

        self._logger.info("real_microphone_closed")

    def reset(self) -> None:
        with self._lock:
            self._queue.clear()
            self._frame_index = 0
            self._dropped_frames = 0
            self._captured_frames = 0
            self._read_frames = 0
            self._last_error = None

        self._logger.info("real_microphone_reset")

    def read_frame(self) -> AudioFrame | None:
        with self._lock:
            if not self._queue:
                return None

            audio_data = self._queue.popleft()
            frame_index = self._frame_index
            self._frame_index += 1
            self._read_frames += 1

        return AudioFrame(
            audio_data=audio_data,
            sample_rate=self._config.sample_rate,
            channels=self._config.channels,
            frame_index=frame_index,
            source="real_microphone_adapter",
        )

    def _on_raw_audio(self, audio_data: bytes) -> None:
        if not audio_data:
            return

        with self._lock:
            if len(self._queue) >= self._config.queue_max_frames:
                self._handle_queue_overflow_locked()

            if len(self._queue) < self._config.queue_max_frames:
                self._queue.append(audio_data)
                self._captured_frames += 1

    def _handle_queue_overflow_locked(self) -> None:
        if self._config.overflow_policy == "drop_oldest":
            self._queue.popleft()
            self._dropped_frames += 1
            return

        if self._config.overflow_policy == "drop_newest":
            self._dropped_frames += 1
            return

        raise RuntimeError("Real microphone audio queue overflow.")

    def _record_error(self, exc: Exception) -> None:
        error = f"{type(exc).__name__}: {exc}"

        with self._lock:
            self._last_error = error

        self._logger.error(
            "real_microphone_error",
            error=error,
        )