from __future__ import annotations

import importlib
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol

from jarvis.voice.contracts import (
    VoiceDeviceHealth,
    VoiceInputFrame,
    VoiceInputFrameKind,
    VoiceRuntimeConfig,
    default_voice_runtime_config,
    make_voice_frame_id,
    make_voice_session_id,
    utc_now,
)


class VoiceMicrophoneCaptureStatus(StrEnum):
    CREATED = "created"
    READY = "ready"
    CAPTURING = "capturing"
    STOPPED = "stopped"
    DEGRADED = "degraded"
    FAILED = "failed"


class VoiceMicrophoneCaptureOperation(StrEnum):
    PREPARE = "prepare"
    CAPTURE_ONCE = "capture_once"
    START = "start"
    STOP = "stop"
    SNAPSHOT = "snapshot"


@dataclass(frozen=True, slots=True)
class VoiceMicrophoneDeviceInfo:
    name: str
    index: int | None
    sample_rate_hz: int
    channels: int
    health: VoiceDeviceHealth
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("microphone device name cannot be empty.")
        if self.sample_rate_hz <= 0:
            raise ValueError("microphone sample_rate_hz must be positive.")
        if self.channels != 1:
            raise ValueError("microphone capture requires mono audio.")


@dataclass(frozen=True, slots=True)
class VoiceMicrophoneCapturePolicy:
    max_consecutive_failures: int = 3
    allow_overflow_recovery: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_consecutive_failures < 1:
            raise ValueError("max_consecutive_failures must be positive.")


@dataclass(frozen=True, slots=True)
class VoiceMicrophoneCaptureResult:
    status: VoiceMicrophoneCaptureStatus
    operation: VoiceMicrophoneCaptureOperation
    frame: VoiceInputFrame | None
    device: VoiceMicrophoneDeviceInfo | None
    message: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status in {
            VoiceMicrophoneCaptureStatus.READY,
            VoiceMicrophoneCaptureStatus.CAPTURING,
            VoiceMicrophoneCaptureStatus.STOPPED,
        }


@dataclass(frozen=True, slots=True)
class VoiceMicrophoneCaptureSnapshot:
    status: VoiceMicrophoneCaptureStatus
    device: VoiceMicrophoneDeviceInfo | None
    captured_frames: int
    captured_bytes: int
    consecutive_failures: int
    last_error: str | None
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


class VoiceMicrophoneAdapter(Protocol):
    def prepare(self, config: VoiceRuntimeConfig) -> VoiceMicrophoneDeviceInfo:
        raise NotImplementedError

    def start(self) -> None:
        raise NotImplementedError

    def read_frame(self, frame_bytes: int) -> bytes:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


class PyAudioMicrophoneAdapter:
    """
    Real Windows microphone adapter using PyAudio.

    Import is lazy so tests and non-audio environments do not crash.
    """

    def __init__(self, *, device_index: int | None = None) -> None:
        self._device_index = device_index
        self._pyaudio: Any | None = None
        self._pa: Any | None = None
        self._stream: Any | None = None
        self._config: VoiceRuntimeConfig | None = None
        self._samples_per_frame = 0

    def prepare(self, config: VoiceRuntimeConfig) -> VoiceMicrophoneDeviceInfo:
        module = importlib.import_module("pyaudio")
        self._pyaudio = module
        self._pa = module.PyAudio()
        self._config = config
        self._samples_per_frame = int(
            config.sample_rate_hz * config.frame_duration_ms / 1000
        )

        name = "Default microphone"
        if self._device_index is not None:
            info = self._pa.get_device_info_by_index(self._device_index)
            name = str(info.get("name", name))
        else:
            info = self._pa.get_default_input_device_info()
            name = str(info.get("name", name))
            self._device_index = int(info.get("index", 0))

        return VoiceMicrophoneDeviceInfo(
            name=name,
            index=self._device_index,
            sample_rate_hz=config.sample_rate_hz,
            channels=config.channels,
            health=VoiceDeviceHealth.READY,
            metadata={"adapter": "pyaudio"},
        )

    def start(self) -> None:
        if self._pyaudio is None or self._pa is None or self._config is None:
            raise RuntimeError("microphone adapter must be prepared first.")

        self._stream = self._pa.open(
            format=self._pyaudio.paInt16,
            channels=self._config.channels,
            rate=self._config.sample_rate_hz,
            input=True,
            input_device_index=self._device_index,
            frames_per_buffer=self._samples_per_frame,
        )

    def read_frame(self, frame_bytes: int) -> bytes:
        if self._stream is None:
            raise RuntimeError("microphone stream is not started.")

        data = self._stream.read(
            frame_bytes,
            exception_on_overflow=False,
        )
        if not isinstance(data, bytes):
            raise RuntimeError("microphone adapter returned non-bytes audio.")
        return data

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

        if self._pa is not None:
            self._pa.terminate()
            self._pa = None


class VoiceMicrophoneCaptureRuntime:
    """
    Step 51B real microphone capture loop.

    This runtime only captures audio frames. It does not transcribe,
    reason, speak, call Ollama, or execute tools.
    """

    def __init__(
        self,
        *,
        config: VoiceRuntimeConfig | None = None,
        adapter: VoiceMicrophoneAdapter | None = None,
        policy: VoiceMicrophoneCapturePolicy | None = None,
    ) -> None:
        self._config = config or default_voice_runtime_config()
        self._adapter = adapter or PyAudioMicrophoneAdapter()
        self._policy = policy or VoiceMicrophoneCapturePolicy()
        self._status = VoiceMicrophoneCaptureStatus.CREATED
        self._device: VoiceMicrophoneDeviceInfo | None = None
        self._session_id = make_voice_session_id()
        self._captured_frames = 0
        self._captured_bytes = 0
        self._consecutive_failures = 0
        self._last_error: str | None = None
        self._lock = threading.RLock()

    def prepare(self) -> VoiceMicrophoneCaptureResult:
        with self._lock:
            try:
                self._device = self._adapter.prepare(self._config)
                self._status = VoiceMicrophoneCaptureStatus.READY
                self._consecutive_failures = 0
                self._last_error = None
                return self._result(
                    operation=VoiceMicrophoneCaptureOperation.PREPARE,
                    status=self._status,
                    message="microphone capture prepared",
                    device=self._device,
                )
            except Exception as exc:
                self._status = VoiceMicrophoneCaptureStatus.FAILED
                self._last_error = str(exc)
                self._consecutive_failures += 1
                return self._result(
                    operation=VoiceMicrophoneCaptureOperation.PREPARE,
                    status=self._status,
                    message="microphone capture prepare failed",
                    metadata={"error": str(exc)},
                )

    def start(self) -> VoiceMicrophoneCaptureResult:
        with self._lock:
            if self._device is None:
                prepared = self.prepare()
                if not prepared.succeeded:
                    return prepared

            try:
                self._adapter.start()
                self._status = VoiceMicrophoneCaptureStatus.CAPTURING
                self._consecutive_failures = 0
                self._last_error = None
                return self._result(
                    operation=VoiceMicrophoneCaptureOperation.START,
                    status=self._status,
                    message="microphone capture started",
                    device=self._device,
                )
            except Exception as exc:
                self._status = VoiceMicrophoneCaptureStatus.FAILED
                self._last_error = str(exc)
                self._consecutive_failures += 1
                return self._result(
                    operation=VoiceMicrophoneCaptureOperation.START,
                    status=self._status,
                    message="microphone capture start failed",
                    device=self._device,
                    metadata={"error": str(exc)},
                )

    def capture_once(self) -> VoiceMicrophoneCaptureResult:
        with self._lock:
            if self._status not in {
                VoiceMicrophoneCaptureStatus.CAPTURING,
                VoiceMicrophoneCaptureStatus.DEGRADED,
            }:
                return self._result(
                    operation=VoiceMicrophoneCaptureOperation.CAPTURE_ONCE,
                    status=VoiceMicrophoneCaptureStatus.DEGRADED,
                    message="microphone capture is not running",
                    device=self._device,
                )

            samples_per_frame = int(
                self._config.sample_rate_hz
                * self._config.frame_duration_ms
                / 1000
            )

            try:
                data = self._adapter.read_frame(samples_per_frame)
                frame = VoiceInputFrame(
                    frame_id=make_voice_frame_id(),
                    session_id=self._session_id,
                    kind=VoiceInputFrameKind.PCM16_MONO,
                    sample_rate_hz=self._config.sample_rate_hz,
                    channels=self._config.channels,
                    data=data,
                    captured_at=utc_now(),
                    duration_ms=self._config.frame_duration_ms,
                    metadata={"source": "microphone"},
                )
                self._captured_frames += 1
                self._captured_bytes += len(data)
                self._consecutive_failures = 0
                self._last_error = None
                self._status = VoiceMicrophoneCaptureStatus.CAPTURING
                return self._result(
                    operation=VoiceMicrophoneCaptureOperation.CAPTURE_ONCE,
                    status=VoiceMicrophoneCaptureStatus.CAPTURING,
                    message="microphone frame captured",
                    frame=frame,
                    device=self._device,
                    metadata={"bytes": len(data)},
                )
            except Exception as exc:
                self._consecutive_failures += 1
                self._last_error = str(exc)
                self._status = _status_from_failures(
                    failures=self._consecutive_failures,
                    policy=self._policy,
                )
                return self._result(
                    operation=VoiceMicrophoneCaptureOperation.CAPTURE_ONCE,
                    status=self._status,
                    message="microphone capture failed",
                    device=self._device,
                    metadata={"error": str(exc)},
                )

    def stop(self) -> VoiceMicrophoneCaptureResult:
        with self._lock:
            try:
                self._adapter.stop()
                self._status = VoiceMicrophoneCaptureStatus.STOPPED
                return self._result(
                    operation=VoiceMicrophoneCaptureOperation.STOP,
                    status=self._status,
                    message="microphone capture stopped",
                    device=self._device,
                )
            except Exception as exc:
                self._status = VoiceMicrophoneCaptureStatus.FAILED
                self._last_error = str(exc)
                return self._result(
                    operation=VoiceMicrophoneCaptureOperation.STOP,
                    status=self._status,
                    message="microphone capture stop failed",
                    device=self._device,
                    metadata={"error": str(exc)},
                )

    def snapshot(self) -> VoiceMicrophoneCaptureSnapshot:
        with self._lock:
            return VoiceMicrophoneCaptureSnapshot(
                status=self._status,
                device=self._device,
                captured_frames=self._captured_frames,
                captured_bytes=self._captured_bytes,
                consecutive_failures=self._consecutive_failures,
                last_error=self._last_error,
                created_at=utc_now(),
            )

    def _result(
        self,
        *,
        operation: VoiceMicrophoneCaptureOperation,
        status: VoiceMicrophoneCaptureStatus,
        message: str,
        frame: VoiceInputFrame | None = None,
        device: VoiceMicrophoneDeviceInfo | None = None,
        metadata: dict[str, object] | None = None,
    ) -> VoiceMicrophoneCaptureResult:
        return VoiceMicrophoneCaptureResult(
            status=status,
            operation=operation,
            frame=frame,
            device=device,
            message=message,
            created_at=utc_now(),
            metadata=metadata or {},
        )


def _status_from_failures(
    *,
    failures: int,
    policy: VoiceMicrophoneCapturePolicy,
) -> VoiceMicrophoneCaptureStatus:
    if failures >= policy.max_consecutive_failures:
        return VoiceMicrophoneCaptureStatus.FAILED
    return VoiceMicrophoneCaptureStatus.DEGRADED