from __future__ import annotations

import importlib
import io
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol

from jarvis.voice.contracts import (
    VoiceDeviceHealth,
    VoicePlaybackState,
    VoicePlaybackStatus,
    VoiceRuntimeConfig,
    VoiceTTSChunk,
    VoiceTTSChunkStatus,
    default_voice_runtime_config,
    make_voice_playback_id,
    utc_now,
)


class VoicePlaybackRuntimeStatus(StrEnum):
    CREATED = "created"
    READY = "ready"
    QUEUED = "queued"
    PLAYING = "playing"
    STOPPING = "stopping"
    STOPPED = "stopped"
    DEGRADED = "degraded"
    FAILED = "failed"


class VoicePlaybackOperation(StrEnum):
    PREPARE = "prepare"
    ENQUEUE = "enqueue"
    PLAY_NEXT = "play_next"
    PLAY_ALL = "play_all"
    STOP = "stop"
    CLEAR = "clear"
    RESET = "reset"
    SNAPSHOT = "snapshot"


@dataclass(frozen=True, slots=True)
class VoiceSpeakerDeviceInfo:
    provider: str
    name: str
    sample_rate_hz: int
    channels: int
    health: VoiceDeviceHealth
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("speaker provider cannot be empty.")
        if not self.name.strip():
            raise ValueError("speaker name cannot be empty.")
        if self.sample_rate_hz <= 0:
            raise ValueError("speaker sample_rate_hz must be positive.")
        if self.channels <= 0:
            raise ValueError("speaker channels must be positive.")


@dataclass(frozen=True, slots=True)
class VoicePlaybackPolicy:
    provider: str = "sounddevice"
    output_device_index: int | None = None
    blocking_playback: bool = True
    max_queue_chunks: int = 32
    target_start_latency_ms: int = 120
    stop_timeout_ms: int = 250
    allow_sample_rate_mismatch: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("playback provider cannot be empty.")
        if self.max_queue_chunks < 1:
            raise ValueError("max_queue_chunks must be positive.")
        if self.target_start_latency_ms <= 0:
            raise ValueError("target_start_latency_ms must be positive.")
        if self.stop_timeout_ms <= 0:
            raise ValueError("stop_timeout_ms must be positive.")


@dataclass(frozen=True, slots=True)
class VoicePlaybackAdapterReport:
    played: bool
    sample_rate_hz: int
    duration_ms: int
    latency_ms: float
    bytes_played: int
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("playback report sample_rate_hz must be positive.")
        if self.duration_ms < 0:
            raise ValueError("playback report duration_ms cannot be negative.")
        if self.latency_ms < 0:
            raise ValueError("playback report latency_ms cannot be negative.")
        if self.bytes_played < 0:
            raise ValueError("playback report bytes_played cannot be negative.")


@dataclass(frozen=True, slots=True)
class VoicePlaybackResult:
    status: VoicePlaybackRuntimeStatus
    operation: VoicePlaybackOperation
    playback_state: VoicePlaybackState | None
    played_chunks: tuple[VoiceTTSChunk, ...]
    queued_chunks: int
    speaker: VoiceSpeakerDeviceInfo | None
    message: str
    latency_ms: float
    first_audio_latency_ms: float | None
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status in {
            VoicePlaybackRuntimeStatus.READY,
            VoicePlaybackRuntimeStatus.QUEUED,
            VoicePlaybackRuntimeStatus.PLAYING,
            VoicePlaybackRuntimeStatus.STOPPED,
        }


@dataclass(frozen=True, slots=True)
class VoicePlaybackSnapshot:
    status: VoicePlaybackRuntimeStatus
    speaker: VoiceSpeakerDeviceInfo | None
    queued_chunks: int
    played_chunks: int
    failed_chunks: int
    stopped_count: int
    current_playback: VoicePlaybackState | None
    last_latency_ms: float | None
    last_first_audio_latency_ms: float | None
    last_error: str | None
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


class VoiceSpeakerAdapter(Protocol):
    def prepare(
        self,
        config: VoiceRuntimeConfig,
        policy: VoicePlaybackPolicy,
    ) -> VoiceSpeakerDeviceInfo:
        raise NotImplementedError

    def play(
        self,
        chunk: VoiceTTSChunk,
        policy: VoicePlaybackPolicy,
    ) -> VoicePlaybackAdapterReport:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class SoundDeviceSpeakerAdapter:
    """
    Real speaker adapter using sounddevice + soundfile.

    It accepts WAV bytes from VoiceTTSChunk and plays them through the system
    speaker output. It does not generate or modify conversational content.
    """

    def __init__(self) -> None:
        self._sounddevice: Any | None = None
        self._soundfile: Any | None = None

    def prepare(
        self,
        config: VoiceRuntimeConfig,
        policy: VoicePlaybackPolicy,
    ) -> VoiceSpeakerDeviceInfo:
        self._sounddevice = importlib.import_module("sounddevice")
        self._soundfile = importlib.import_module("soundfile")

        device_name = "Default speaker"
        channels = 1
        sample_rate = 44_100

        try:
            device = self._sounddevice.query_devices(
                policy.output_device_index,
                "output",
            )
            device_name = str(device.get("name", device_name))
            channels = int(device.get("max_output_channels", channels))
            sample_rate = int(device.get("default_samplerate", sample_rate))
        except Exception:
            device_name = "Default speaker"

        return VoiceSpeakerDeviceInfo(
            provider="sounddevice",
            name=device_name,
            sample_rate_hz=sample_rate,
            channels=max(1, channels),
            health=VoiceDeviceHealth.READY,
            metadata={
                "output_device_index": policy.output_device_index,
                "blocking_playback": policy.blocking_playback,
            },
        )

    def play(
        self,
        chunk: VoiceTTSChunk,
        policy: VoicePlaybackPolicy,
    ) -> VoicePlaybackAdapterReport:
        if self._sounddevice is None or self._soundfile is None:
            raise RuntimeError("speaker adapter must be prepared first.")

        started = time.perf_counter()
        audio_file = io.BytesIO(chunk.audio)
        data, sample_rate = self._soundfile.read(
            audio_file,
            dtype="float32",
            always_2d=True,
        )

        self._sounddevice.play(
            data,
            samplerate=sample_rate,
            device=policy.output_device_index,
            blocking=policy.blocking_playback,
        )

        latency_ms = (time.perf_counter() - started) * 1000.0

        return VoicePlaybackAdapterReport(
            played=True,
            sample_rate_hz=int(sample_rate),
            duration_ms=chunk.duration_ms,
            latency_ms=latency_ms,
            bytes_played=len(chunk.audio),
            metadata={"provider": "sounddevice"},
        )

    def stop(self) -> None:
        if self._sounddevice is not None:
            self._sounddevice.stop()

    def close(self) -> None:
        self.stop()


class VoicePlaybackRuntime:
    """
    Step 51G speaker playback runtime.

    Plays VoiceTTSChunk audio through speakers.

    It does not:
    - generate response text
    - call Ollama
    - synthesize TTS
    - listen to microphone
    - bypass interruption controls

    It is interrupt-ready through stop(), which 51H will use for barge-in.
    """

    def __init__(
        self,
        *,
        config: VoiceRuntimeConfig | None = None,
        adapter: VoiceSpeakerAdapter | None = None,
        policy: VoicePlaybackPolicy | None = None,
    ) -> None:
        self._config = config or default_voice_runtime_config()
        self._adapter = adapter or SoundDeviceSpeakerAdapter()
        self._policy = policy or VoicePlaybackPolicy()
        self._status = VoicePlaybackRuntimeStatus.CREATED
        self._speaker: VoiceSpeakerDeviceInfo | None = None
        self._queue: deque[VoiceTTSChunk] = deque()
        self._current_playback: VoicePlaybackState | None = None
        self._played_chunks = 0
        self._failed_chunks = 0
        self._stopped_count = 0
        self._last_latency_ms: float | None = None
        self._last_first_audio_latency_ms: float | None = None
        self._last_error: str | None = None
        self._stop_requested = False

    def prepare(self) -> VoicePlaybackResult:
        started = time.perf_counter()
        try:
            self._speaker = self._adapter.prepare(self._config, self._policy)
            self._status = VoicePlaybackRuntimeStatus.READY
            self._last_error = None
            return self._result(
                operation=VoicePlaybackOperation.PREPARE,
                status=self._status,
                playback_state=None,
                played_chunks=(),
                message="speaker playback prepared",
                started=started,
            )
        except Exception as exc:
            self._status = VoicePlaybackRuntimeStatus.FAILED
            self._last_error = str(exc)
            return self._result(
                operation=VoicePlaybackOperation.PREPARE,
                status=self._status,
                playback_state=None,
                played_chunks=(),
                message="speaker playback prepare failed",
                started=started,
                metadata={"error": str(exc)},
            )

    def enqueue_chunk(
        self,
        chunk: VoiceTTSChunk,
    ) -> VoicePlaybackResult:
        return self.enqueue_chunks((chunk,))

    def enqueue_chunks(
        self,
        chunks: tuple[VoiceTTSChunk, ...],
    ) -> VoicePlaybackResult:
        started = time.perf_counter()

        if not chunks:
            self._status = VoicePlaybackRuntimeStatus.DEGRADED
            return self._result(
                operation=VoicePlaybackOperation.ENQUEUE,
                status=self._status,
                playback_state=None,
                played_chunks=(),
                message="no playback chunks supplied",
                started=started,
            )

        if len(self._queue) + len(chunks) > self._policy.max_queue_chunks:
            self._status = VoicePlaybackRuntimeStatus.DEGRADED
            return self._result(
                operation=VoicePlaybackOperation.ENQUEUE,
                status=self._status,
                playback_state=None,
                played_chunks=(),
                message="playback queue capacity exceeded",
                started=started,
                metadata={
                    "queue_size": len(self._queue),
                    "incoming": len(chunks),
                    "max_queue_chunks": self._policy.max_queue_chunks,
                },
            )

        for chunk in chunks:
            _validate_playback_chunk(chunk)
            self._queue.append(chunk)

        self._status = VoicePlaybackRuntimeStatus.QUEUED
        return self._result(
            operation=VoicePlaybackOperation.ENQUEUE,
            status=self._status,
            playback_state=None,
            played_chunks=(),
            message="voice chunks queued for playback",
            started=started,
            metadata={"queued": len(chunks)},
        )

    def play_next(self) -> VoicePlaybackResult:
        started = time.perf_counter()

        if self._speaker is None:
            prepared = self.prepare()
            if prepared.status == VoicePlaybackRuntimeStatus.FAILED:
                return prepared

        if not self._queue:
            self._status = VoicePlaybackRuntimeStatus.READY
            return self._result(
                operation=VoicePlaybackOperation.PLAY_NEXT,
                status=self._status,
                playback_state=None,
                played_chunks=(),
                message="playback queue empty",
                started=started,
            )

        if self._stop_requested:
            self._status = VoicePlaybackRuntimeStatus.STOPPED
            return self._result(
                operation=VoicePlaybackOperation.PLAY_NEXT,
                status=self._status,
                playback_state=None,
                played_chunks=(),
                message="playback stopped before next chunk",
                started=started,
            )

        chunk = self._queue.popleft()
        playback_state = VoicePlaybackState(
            playback_id=make_voice_playback_id(),
            session_id=chunk.session_id,
            status=VoicePlaybackStatus.PLAYING,
            chunk_id=chunk.chunk_id,
            started_at=utc_now(),
            stopped_at=None,
            metadata={"source": "voice_tts_chunk"},
        )
        self._current_playback = playback_state
        self._status = VoicePlaybackRuntimeStatus.PLAYING

        try:
            report = self._adapter.play(chunk, self._policy)
        except Exception as exc:
            self._failed_chunks += 1
            self._status = VoicePlaybackRuntimeStatus.FAILED
            self._last_error = str(exc)
            failed_state = VoicePlaybackState(
                playback_id=playback_state.playback_id,
                session_id=playback_state.session_id,
                status=VoicePlaybackStatus.FAILED,
                chunk_id=chunk.chunk_id,
                started_at=playback_state.started_at,
                stopped_at=utc_now(),
                metadata={"error": str(exc)},
            )
            self._current_playback = failed_state
            return self._result(
                operation=VoicePlaybackOperation.PLAY_NEXT,
                status=self._status,
                playback_state=failed_state,
                played_chunks=(),
                message="speaker playback failed",
                started=started,
                metadata={"error": str(exc)},
            )

        stopped_state = VoicePlaybackState(
            playback_id=playback_state.playback_id,
            session_id=playback_state.session_id,
            status=VoicePlaybackStatus.STOPPED,
            chunk_id=chunk.chunk_id,
            started_at=playback_state.started_at,
            stopped_at=utc_now(),
            metadata={
                "bytes_played": report.bytes_played,
                "duration_ms": report.duration_ms,
                "adapter_latency_ms": report.latency_ms,
                **report.metadata,
            },
        )
        self._current_playback = stopped_state
        self._played_chunks += 1
        self._last_error = None
        self._status = (
            VoicePlaybackRuntimeStatus.QUEUED
            if self._queue
            else VoicePlaybackRuntimeStatus.READY
        )

        return self._result(
            operation=VoicePlaybackOperation.PLAY_NEXT,
            status=VoicePlaybackRuntimeStatus.PLAYING,
            playback_state=stopped_state,
            played_chunks=(chunk,),
            message="voice chunk played",
            started=started,
            first_audio_latency_ms=report.latency_ms,
            metadata={
                "remaining_queue": len(self._queue),
                "adapter_latency_ms": report.latency_ms,
            },
        )

    def play_all(self) -> VoicePlaybackResult:
        started = time.perf_counter()
        played: list[VoiceTTSChunk] = []
        first_audio_latency_ms: float | None = None

        if self._speaker is None:
            prepared = self.prepare()
            if prepared.status == VoicePlaybackRuntimeStatus.FAILED:
                return prepared

        while self._queue:
            if self._stop_requested:
                self._status = VoicePlaybackRuntimeStatus.STOPPED
                break

            result = self.play_next()
            played.extend(result.played_chunks)

            if first_audio_latency_ms is None:
                first_audio_latency_ms = result.first_audio_latency_ms

            if result.status == VoicePlaybackRuntimeStatus.FAILED:
                return self._result(
                    operation=VoicePlaybackOperation.PLAY_ALL,
                    status=result.status,
                    playback_state=result.playback_state,
                    played_chunks=tuple(played),
                    message="playback failed during queue drain",
                    started=started,
                    first_audio_latency_ms=first_audio_latency_ms,
                    metadata=result.metadata,
                )

        status = (
            VoicePlaybackRuntimeStatus.STOPPED
            if self._stop_requested
            else VoicePlaybackRuntimeStatus.READY
        )
        self._status = status

        return self._result(
            operation=VoicePlaybackOperation.PLAY_ALL,
            status=status,
            playback_state=self._current_playback,
            played_chunks=tuple(played),
            message="playback queue drained",
            started=started,
            first_audio_latency_ms=first_audio_latency_ms,
            metadata={"played_count": len(played)},
        )

    def stop(self) -> VoicePlaybackResult:
        started = time.perf_counter()
        self._stop_requested = True
        self._status = VoicePlaybackRuntimeStatus.STOPPING

        try:
            self._adapter.stop()
        except Exception as exc:
            self._status = VoicePlaybackRuntimeStatus.DEGRADED
            self._last_error = str(exc)
            return self._result(
                operation=VoicePlaybackOperation.STOP,
                status=self._status,
                playback_state=self._current_playback,
                played_chunks=(),
                message="playback stop degraded",
                started=started,
                metadata={"error": str(exc)},
            )

        self._queue.clear()
        self._stopped_count += 1

        if self._current_playback is not None:
            self._current_playback = VoicePlaybackState(
                playback_id=self._current_playback.playback_id,
                session_id=self._current_playback.session_id,
                status=VoicePlaybackStatus.INTERRUPTED,
                chunk_id=self._current_playback.chunk_id,
                started_at=self._current_playback.started_at,
                stopped_at=utc_now(),
                metadata={"stop_requested": True},
            )

        self._status = VoicePlaybackRuntimeStatus.STOPPED
        return self._result(
            operation=VoicePlaybackOperation.STOP,
            status=self._status,
            playback_state=self._current_playback,
            played_chunks=(),
            message="playback stopped",
            started=started,
        )

    def clear(self) -> VoicePlaybackResult:
        started = time.perf_counter()
        self._queue.clear()
        self._stop_requested = False
        self._status = VoicePlaybackRuntimeStatus.READY
        return self._result(
            operation=VoicePlaybackOperation.CLEAR,
            status=self._status,
            playback_state=self._current_playback,
            played_chunks=(),
            message="playback queue cleared",
            started=started,
        )

    def reset(self) -> VoicePlaybackResult:
        started = time.perf_counter()
        self._queue.clear()
        self._adapter.close()
        self._speaker = None
        self._current_playback = None
        self._stop_requested = False
        self._status = VoicePlaybackRuntimeStatus.CREATED
        self._last_error = None
        return self._result(
            operation=VoicePlaybackOperation.RESET,
            status=self._status,
            playback_state=None,
            played_chunks=(),
            message="playback runtime reset",
            started=started,
        )

    def snapshot(self) -> VoicePlaybackSnapshot:
        return VoicePlaybackSnapshot(
            status=self._status,
            speaker=self._speaker,
            queued_chunks=len(self._queue),
            played_chunks=self._played_chunks,
            failed_chunks=self._failed_chunks,
            stopped_count=self._stopped_count,
            current_playback=self._current_playback,
            last_latency_ms=self._last_latency_ms,
            last_first_audio_latency_ms=self._last_first_audio_latency_ms,
            last_error=self._last_error,
            created_at=utc_now(),
        )

    def _result(
        self,
        *,
        operation: VoicePlaybackOperation,
        status: VoicePlaybackRuntimeStatus,
        playback_state: VoicePlaybackState | None,
        played_chunks: tuple[VoiceTTSChunk, ...],
        message: str,
        started: float,
        first_audio_latency_ms: float | None = None,
        metadata: dict[str, object] | None = None,
    ) -> VoicePlaybackResult:
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._last_latency_ms = latency_ms
        if first_audio_latency_ms is not None:
            self._last_first_audio_latency_ms = first_audio_latency_ms

        return VoicePlaybackResult(
            status=status,
            operation=operation,
            playback_state=playback_state,
            played_chunks=played_chunks,
            queued_chunks=len(self._queue),
            speaker=self._speaker,
            message=message,
            latency_ms=latency_ms,
            first_audio_latency_ms=first_audio_latency_ms,
            created_at=utc_now(),
            metadata=metadata or {},
        )


def _validate_playback_chunk(chunk: VoiceTTSChunk) -> None:
    if chunk.status not in {
        VoiceTTSChunkStatus.SYNTHESIZED,
        VoiceTTSChunkStatus.QUEUED,
    }:
        raise ValueError("only synthesized/queued TTS chunks can be played.")
    if not chunk.audio:
        raise ValueError("playback chunk audio cannot be empty.")
    if chunk.sample_rate_hz <= 0:
        raise ValueError("playback chunk sample_rate_hz must be positive.")
    if chunk.duration_ms <= 0:
        raise ValueError("playback chunk duration_ms must be positive.")