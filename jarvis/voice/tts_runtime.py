from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from jarvis.live import LiveResponse
from jarvis.voice.contracts import (
    VoiceDeviceHealth,
    VoiceRuntimeConfig,
    VoiceSessionId,
    VoiceTTSChunk,
    VoiceTTSChunkStatus,
    VoiceTTSRequest,
    default_voice_runtime_config,
    make_voice_tts_chunk_id,
    utc_now,
)


class VoiceTTSRuntimeStatus(StrEnum):
    CREATED = "created"
    READY = "ready"
    SYNTHESIZING = "synthesizing"
    DEGRADED = "degraded"
    FAILED = "failed"


class VoiceTTSOperation(StrEnum):
    PREPARE = "prepare"
    SYNTHESIZE_TEXT = "synthesize_text"
    SYNTHESIZE_RESPONSE = "synthesize_response"
    RESET = "reset"
    SNAPSHOT = "snapshot"


class VoiceTTSAudioFormat(StrEnum):
    WAV = "wav"


@dataclass(frozen=True, slots=True)
class VoiceTTSVoiceInfo:
    provider: str
    voice_name: str
    sample_rate_hz: int
    audio_format: VoiceTTSAudioFormat
    health: VoiceDeviceHealth
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("TTS provider cannot be empty.")
        if not self.voice_name.strip():
            raise ValueError("TTS voice_name cannot be empty.")
        if self.sample_rate_hz <= 0:
            raise ValueError("TTS sample_rate_hz must be positive.")


@dataclass(frozen=True, slots=True)
class VoiceTTSAudioData:
    audio: bytes
    sample_rate_hz: int
    duration_ms: int
    audio_format: VoiceTTSAudioFormat
    latency_ms: float
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.audio:
            raise ValueError("TTS audio cannot be empty.")
        if self.sample_rate_hz <= 0:
            raise ValueError("TTS audio sample_rate_hz must be positive.")
        if self.duration_ms <= 0:
            raise ValueError("TTS audio duration_ms must be positive.")
        if self.latency_ms < 0:
            raise ValueError("TTS latency_ms cannot be negative.")


@dataclass(frozen=True, slots=True)
class VoiceTTSChunkPlan:
    text: str
    index: int
    estimated_chars: int
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("TTS chunk plan text cannot be empty.")
        if self.index < 0:
            raise ValueError("TTS chunk plan index cannot be negative.")
        if self.estimated_chars <= 0:
            raise ValueError("TTS chunk plan estimated_chars must be positive.")


@dataclass(frozen=True, slots=True)
class VoiceTTSPolicy:
    provider: str = "piper"
    voice_name: str = "default"
    piper_executable: str = "piper"
    piper_model_path: Path | None = None
    piper_config_path: Path | None = None
    sample_rate_hz: int = 22_050
    max_chars_per_chunk: int = 220
    max_total_chars: int = 2_000
    timeout_seconds: int = 20
    max_retries: int = 1
    target_first_chunk_ms: int = 800
    normalize_whitespace: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("TTS provider cannot be empty.")
        if not self.voice_name.strip():
            raise ValueError("TTS voice_name cannot be empty.")
        if not self.piper_executable.strip():
            raise ValueError("piper_executable cannot be empty.")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive.")
        if self.max_chars_per_chunk < 40:
            raise ValueError("max_chars_per_chunk must be at least 40.")
        if self.max_total_chars < self.max_chars_per_chunk:
            raise ValueError("max_total_chars must exceed max_chars_per_chunk.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative.")
        if self.target_first_chunk_ms <= 0:
            raise ValueError("target_first_chunk_ms must be positive.")


@dataclass(frozen=True, slots=True)
class VoiceTTSResult:
    status: VoiceTTSRuntimeStatus
    operation: VoiceTTSOperation
    request: VoiceTTSRequest | None
    chunks: tuple[VoiceTTSChunk, ...]
    plans: tuple[VoiceTTSChunkPlan, ...]
    voice: VoiceTTSVoiceInfo | None
    message: str
    latency_ms: float
    first_chunk_latency_ms: float | None
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return (
            self.status
            in {
                VoiceTTSRuntimeStatus.READY,
                VoiceTTSRuntimeStatus.SYNTHESIZING,
            }
            and bool(self.chunks)
        )


@dataclass(frozen=True, slots=True)
class VoiceTTSSnapshot:
    status: VoiceTTSRuntimeStatus
    voice: VoiceTTSVoiceInfo | None
    synthesized_requests: int
    synthesized_chunks: int
    failed_requests: int
    degraded_requests: int
    last_text: str | None
    last_latency_ms: float | None
    last_first_chunk_latency_ms: float | None
    last_error: str | None
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


class VoiceTTSAdapter(Protocol):
    def prepare(
        self,
        config: VoiceRuntimeConfig,
        policy: VoiceTTSPolicy,
    ) -> VoiceTTSVoiceInfo:
        raise NotImplementedError

    def synthesize(
        self,
        request: VoiceTTSRequest,
        plan: VoiceTTSChunkPlan,
        config: VoiceRuntimeConfig,
        policy: VoiceTTSPolicy,
    ) -> VoiceTTSAudioData:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class PiperCliTTSAdapter:
    """
    Piper CLI adapter.

    This adapter only converts already-generated text into audio.
    It never creates conversational content.
    """

    def prepare(
        self,
        config: VoiceRuntimeConfig,
        policy: VoiceTTSPolicy,
    ) -> VoiceTTSVoiceInfo:
        executable = shutil.which(policy.piper_executable)
        if executable is None:
            raise RuntimeError(
                "Piper executable not found. Install/configure Piper first."
            )

        if policy.piper_model_path is None:
            raise RuntimeError("piper_model_path is required for real TTS.")

        if not policy.piper_model_path.exists():
            raise RuntimeError(
                f"Piper model does not exist: {policy.piper_model_path}"
            )

        if (
            policy.piper_config_path is not None
            and not policy.piper_config_path.exists()
        ):
            raise RuntimeError(
                f"Piper config does not exist: {policy.piper_config_path}"
            )

        return VoiceTTSVoiceInfo(
            provider="piper",
            voice_name=policy.voice_name,
            sample_rate_hz=policy.sample_rate_hz,
            audio_format=VoiceTTSAudioFormat.WAV,
            health=VoiceDeviceHealth.READY,
            metadata={
                "executable": executable,
                "model_path": str(policy.piper_model_path),
                "config_path": (
                    str(policy.piper_config_path)
                    if policy.piper_config_path is not None
                    else None
                ),
            },
        )

    def synthesize(
        self,
        request: VoiceTTSRequest,
        plan: VoiceTTSChunkPlan,
        config: VoiceRuntimeConfig,
        policy: VoiceTTSPolicy,
    ) -> VoiceTTSAudioData:
        executable = shutil.which(policy.piper_executable)
        if executable is None:
            raise RuntimeError("Piper executable unavailable.")

        if policy.piper_model_path is None:
            raise RuntimeError("piper_model_path is required.")

        started = time.perf_counter()

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as tmp:
            output_path = Path(tmp.name)

        command = [
            executable,
            "--model",
            str(policy.piper_model_path),
            "--output_file",
            str(output_path),
        ]

        if policy.piper_config_path is not None:
            command.extend(["--config", str(policy.piper_config_path)])

        try:
            completed = subprocess.run(
                command,
                input=plan.text,
                text=True,
                capture_output=True,
                timeout=policy.timeout_seconds,
                check=False,
            )

            if completed.returncode != 0:
                stderr = completed.stderr.strip()
                raise RuntimeError(
                    f"Piper synthesis failed: {stderr or 'unknown error'}"
                )

            audio = output_path.read_bytes()
            sample_rate, duration_ms = _read_wav_info(output_path)
            latency_ms = (time.perf_counter() - started) * 1000.0

            return VoiceTTSAudioData(
                audio=audio,
                sample_rate_hz=sample_rate,
                duration_ms=duration_ms,
                audio_format=VoiceTTSAudioFormat.WAV,
                latency_ms=latency_ms,
                metadata={
                    "provider": "piper",
                    "chunk_index": plan.index,
                },
            )
        finally:
            output_path.unlink(missing_ok=True)

    def close(self) -> None:
        return None


class VoiceTTSRuntime:
    """
    Step 51F TTS runtime.

    Converts generated response text into streaming-ready audio chunks.

    It does not:
    - generate responses
    - call Ollama
    - access tools
    - play audio
    - bypass Step 50 response boundary
    """

    def __init__(
        self,
        *,
        config: VoiceRuntimeConfig | None = None,
        adapter: VoiceTTSAdapter | None = None,
        policy: VoiceTTSPolicy | None = None,
    ) -> None:
        self._config = config or default_voice_runtime_config()
        self._adapter = adapter or PiperCliTTSAdapter()
        self._policy = policy or VoiceTTSPolicy()
        self._status = VoiceTTSRuntimeStatus.CREATED
        self._voice: VoiceTTSVoiceInfo | None = None
        self._synthesized_requests = 0
        self._synthesized_chunks = 0
        self._failed_requests = 0
        self._degraded_requests = 0
        self._last_text: str | None = None
        self._last_latency_ms: float | None = None
        self._last_first_chunk_latency_ms: float | None = None
        self._last_error: str | None = None

    def prepare(self) -> VoiceTTSResult:
        started = time.perf_counter()
        try:
            self._voice = self._adapter.prepare(self._config, self._policy)
            self._status = VoiceTTSRuntimeStatus.READY
            self._last_error = None
            return self._result(
                operation=VoiceTTSOperation.PREPARE,
                status=self._status,
                request=None,
                chunks=(),
                plans=(),
                message="TTS runtime prepared",
                started=started,
            )
        except Exception as exc:
            self._status = VoiceTTSRuntimeStatus.FAILED
            self._failed_requests += 1
            self._last_error = str(exc)
            return self._result(
                operation=VoiceTTSOperation.PREPARE,
                status=self._status,
                request=None,
                chunks=(),
                plans=(),
                message="TTS runtime prepare failed",
                started=started,
                metadata={"error": str(exc)},
            )

    def synthesize_response(
        self,
        *,
        response: LiveResponse,
        session_id: VoiceSessionId,
        voice: str | None = None,
    ) -> VoiceTTSResult:
        return self.synthesize_text(
            text=response.text,
            session_id=session_id,
            voice=voice,
            operation=VoiceTTSOperation.SYNTHESIZE_RESPONSE,
            metadata={
                "source": "live_response",
                "response_id": str(response.response_id),
            },
        )

    def synthesize_text(
        self,
        *,
        text: str,
        session_id: VoiceSessionId,
        voice: str | None = None,
        operation: VoiceTTSOperation = VoiceTTSOperation.SYNTHESIZE_TEXT,
        metadata: dict[str, object] | None = None,
    ) -> VoiceTTSResult:
        started = time.perf_counter()
        normalized = _normalize_tts_text(
            text=text,
            policy=self._policy,
        )

        if not normalized:
            self._status = VoiceTTSRuntimeStatus.DEGRADED
            self._degraded_requests += 1
            return self._result(
                operation=operation,
                status=self._status,
                request=None,
                chunks=(),
                plans=(),
                message="TTS text cannot be empty",
                started=started,
            )

        if len(normalized) > self._policy.max_total_chars:
            normalized = normalized[: self._policy.max_total_chars].rstrip()
            self._degraded_requests += 1

        if self._voice is None:
            prepared = self.prepare()
            if prepared.status == VoiceTTSRuntimeStatus.FAILED:
                return prepared

        request = VoiceTTSRequest(
            session_id=session_id,
            text=normalized,
            voice=voice or self._policy.voice_name,
            created_at=utc_now(),
            metadata=metadata or {},
        )

        plans = _plan_low_latency_chunks(
            text=normalized,
            max_chars=self._policy.max_chars_per_chunk,
        )

        self._last_text = normalized
        self._status = VoiceTTSRuntimeStatus.SYNTHESIZING

        chunks: list[VoiceTTSChunk] = []
        first_chunk_latency_ms: float | None = None

        for plan in plans:
            chunk_started = time.perf_counter()
            audio_data = self._synthesize_with_recovery(
                request=request,
                plan=plan,
            )

            if audio_data is None:
                self._status = VoiceTTSRuntimeStatus.FAILED
                self._failed_requests += 1
                return self._result(
                    operation=operation,
                    status=self._status,
                    request=request,
                    chunks=tuple(chunks),
                    plans=plans,
                    message="TTS synthesis failed",
                    started=started,
                    first_chunk_latency_ms=first_chunk_latency_ms,
                )

            chunk_latency_ms = (time.perf_counter() - chunk_started) * 1000.0
            if first_chunk_latency_ms is None:
                first_chunk_latency_ms = chunk_latency_ms

            chunk = VoiceTTSChunk(
                chunk_id=make_voice_tts_chunk_id(),
                session_id=session_id,
                status=VoiceTTSChunkStatus.SYNTHESIZED,
                audio=audio_data.audio,
                sample_rate_hz=audio_data.sample_rate_hz,
                duration_ms=audio_data.duration_ms,
                created_at=utc_now(),
                metadata={
                    "chunk_index": plan.index,
                    "text": plan.text,
                    "audio_format": audio_data.audio_format.value,
                    "latency_ms": audio_data.latency_ms,
                    "streaming_ready": True,
                    **audio_data.metadata,
                },
            )
            chunks.append(chunk)

        latency_ms = (time.perf_counter() - started) * 1000.0
        self._synthesized_requests += 1
        self._synthesized_chunks += len(chunks)
        self._last_latency_ms = latency_ms
        self._last_first_chunk_latency_ms = first_chunk_latency_ms
        self._last_error = None
        self._status = VoiceTTSRuntimeStatus.READY

        return VoiceTTSResult(
            status=VoiceTTSRuntimeStatus.SYNTHESIZING,
            operation=operation,
            request=request,
            chunks=tuple(chunks),
            plans=plans,
            voice=self._voice,
            message="TTS audio chunks synthesized",
            latency_ms=latency_ms,
            first_chunk_latency_ms=first_chunk_latency_ms,
            created_at=utc_now(),
            metadata={
                "chunk_count": len(chunks),
                "streaming_ready": True,
                "target_first_chunk_ms": self._policy.target_first_chunk_ms,
                "first_chunk_within_target": (
                    first_chunk_latency_ms is not None
                    and first_chunk_latency_ms
                    <= self._policy.target_first_chunk_ms
                ),
                "degraded_truncation": (
                    len(text) > self._policy.max_total_chars
                ),
            },
        )

    def reset(self) -> VoiceTTSResult:
        started = time.perf_counter()
        self._adapter.close()
        self._voice = None
        self._status = VoiceTTSRuntimeStatus.CREATED
        self._last_text = None
        self._last_latency_ms = None
        self._last_first_chunk_latency_ms = None
        self._last_error = None
        return self._result(
            operation=VoiceTTSOperation.RESET,
            status=self._status,
            request=None,
            chunks=(),
            plans=(),
            message="TTS runtime reset",
            started=started,
        )

    def snapshot(self) -> VoiceTTSSnapshot:
        return VoiceTTSSnapshot(
            status=self._status,
            voice=self._voice,
            synthesized_requests=self._synthesized_requests,
            synthesized_chunks=self._synthesized_chunks,
            failed_requests=self._failed_requests,
            degraded_requests=self._degraded_requests,
            last_text=self._last_text,
            last_latency_ms=self._last_latency_ms,
            last_first_chunk_latency_ms=self._last_first_chunk_latency_ms,
            last_error=self._last_error,
            created_at=utc_now(),
        )

    def _synthesize_with_recovery(
        self,
        *,
        request: VoiceTTSRequest,
        plan: VoiceTTSChunkPlan,
    ) -> VoiceTTSAudioData | None:
        attempts = self._policy.max_retries + 1
        for attempt in range(attempts):
            try:
                return self._adapter.synthesize(
                    request,
                    plan,
                    self._config,
                    self._policy,
                )
            except Exception as exc:
                self._last_error = str(exc)
                if attempt >= attempts - 1:
                    return None
                self._degraded_requests += 1
        return None

    def _result(
        self,
        *,
        operation: VoiceTTSOperation,
        status: VoiceTTSRuntimeStatus,
        request: VoiceTTSRequest | None,
        chunks: tuple[VoiceTTSChunk, ...],
        plans: tuple[VoiceTTSChunkPlan, ...],
        message: str,
        started: float,
        first_chunk_latency_ms: float | None = None,
        metadata: dict[str, object] | None = None,
    ) -> VoiceTTSResult:
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._last_latency_ms = latency_ms
        if first_chunk_latency_ms is not None:
            self._last_first_chunk_latency_ms = first_chunk_latency_ms

        return VoiceTTSResult(
            status=status,
            operation=operation,
            request=request,
            chunks=chunks,
            plans=plans,
            voice=self._voice,
            message=message,
            latency_ms=latency_ms,
            first_chunk_latency_ms=first_chunk_latency_ms,
            created_at=utc_now(),
            metadata=metadata or {},
        )


def _normalize_tts_text(
    *,
    text: str,
    policy: VoiceTTSPolicy,
) -> str:
    normalized = text.strip()
    if policy.normalize_whitespace:
        normalized = " ".join(normalized.split())
    return normalized


def _plan_low_latency_chunks(
    *,
    text: str,
    max_chars: int,
) -> tuple[VoiceTTSChunkPlan, ...]:
    sentences = _split_sentences(text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)

        if len(sentence) <= max_chars:
            current = sentence
        else:
            chunks.extend(_split_long_text(sentence, max_chars))
            current = ""

    if current:
        chunks.append(current)

    return tuple(
        VoiceTTSChunkPlan(
            text=chunk,
            index=index,
            estimated_chars=len(chunk),
        )
        for index, chunk in enumerate(chunks)
        if chunk.strip()
    )


def _split_sentences(text: str) -> tuple[str, ...]:
    sentences: list[str] = []
    current = ""

    for char in text:
        current += char
        if char in {".", "?", "!"}:
            sentences.append(current.strip())
            current = ""

    if current.strip():
        sentences.append(current.strip())

    return tuple(sentences)


def _split_long_text(
    text: str,
    max_chars: int,
) -> tuple[str, ...]:
    words = text.split()
    chunks: list[str] = []
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = word

    if current:
        chunks.append(current)

    return tuple(chunks)


def _read_wav_info(path: Path) -> tuple[int, int]:
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        duration_ms = int((frames / sample_rate) * 1000)
    return sample_rate, max(1, duration_ms)