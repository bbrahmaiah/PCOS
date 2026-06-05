from __future__ import annotations

import importlib
import tempfile
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from jarvis.voice.contracts import (
    VoiceDeviceHealth,
    VoiceInputFrame,
    VoiceRuntimeConfig,
    VoiceTranscript,
    VoiceTranscriptKind,
    default_voice_runtime_config,
    make_voice_segment_id,
    make_voice_transcript_id,
    utc_now,
)


class VoiceSTTRuntimeStatus(StrEnum):
    CREATED = "created"
    READY = "ready"
    TRANSCRIBING = "transcribing"
    DEGRADED = "degraded"
    FAILED = "failed"


class VoiceSTTOperation(StrEnum):
    PREPARE = "prepare"
    TRANSCRIBE = "transcribe"
    TRANSCRIBE_PARTIAL = "transcribe_partial"
    TRANSCRIBE_FINAL = "transcribe_final"
    RESET = "reset"
    SNAPSHOT = "snapshot"


class VoiceSTTMode(StrEnum):
    FAST_PARTIAL = "fast_partial"
    ACCURATE_FINAL = "accurate_final"


class VoiceSTTTranscriptSafety(StrEnum):
    PREDICTION_ONLY = "prediction_only"
    SAFE_FOR_DIALOGUE = "safe_for_dialogue"
    SAFE_FOR_ACTION = "safe_for_action"
    NEEDS_CLARIFICATION = "needs_clarification"


@dataclass(frozen=True, slots=True)
class VoiceSTTModelInfo:
    provider: str
    model_name: str
    device: str
    compute_type: str
    language: str
    mode: VoiceSTTMode
    health: VoiceDeviceHealth
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("STT provider cannot be empty.")
        if not self.model_name.strip():
            raise ValueError("STT model_name cannot be empty.")
        if not self.device.strip():
            raise ValueError("STT device cannot be empty.")
        if not self.compute_type.strip():
            raise ValueError("STT compute_type cannot be empty.")
        if not self.language.strip():
            raise ValueError("STT language cannot be empty.")


@dataclass(frozen=True, slots=True)
class VoiceSTTPolicy:
    partial_model_name: str = "tiny.en"
    final_model_name: str = "base.en"
    device: str = "cpu"
    compute_type: str = "int8"
    partial_beam_size: int = 1
    final_beam_size: int = 3
    min_partial_confidence: float = 0.30
    min_final_confidence: float = 0.45
    min_action_confidence: float = 0.70
    allow_partial_for_actions: bool = False
    prewarm_on_prepare: bool = True
    max_empty_results_before_degraded: int = 3
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.partial_model_name.strip():
            raise ValueError("partial_model_name cannot be empty.")
        if not self.final_model_name.strip():
            raise ValueError("final_model_name cannot be empty.")
        if not self.device.strip():
            raise ValueError("STT device cannot be empty.")
        if not self.compute_type.strip():
            raise ValueError("STT compute_type cannot be empty.")
        if self.partial_beam_size < 1:
            raise ValueError("partial_beam_size must be positive.")
        if self.final_beam_size < 1:
            raise ValueError("final_beam_size must be positive.")
        if not 0.0 <= self.min_partial_confidence <= 1.0:
            raise ValueError("min_partial_confidence must be 0..1.")
        if not 0.0 <= self.min_final_confidence <= 1.0:
            raise ValueError("min_final_confidence must be 0..1.")
        if not 0.0 <= self.min_action_confidence <= 1.0:
            raise ValueError("min_action_confidence must be 0..1.")
        if self.max_empty_results_before_degraded < 1:
            raise ValueError(
                "max_empty_results_before_degraded must be positive."
            )


@dataclass(frozen=True, slots=True)
class VoiceSTTRequest:
    frames: tuple[VoiceInputFrame, ...]
    mode: VoiceSTTMode
    allow_action_candidate: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("STT request frames cannot be empty.")


@dataclass(frozen=True, slots=True)
class VoiceSTTTranscriptCandidate:
    text: str
    confidence: float
    mode: VoiceSTTMode
    safety: VoiceSTTTranscriptSafety
    safe_for_action: bool
    latency_ms: float
    model_name: str
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("STT candidate text cannot be empty.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("STT candidate confidence must be 0..1.")
        if self.latency_ms < 0:
            raise ValueError("STT candidate latency_ms cannot be negative.")
        if not self.model_name.strip():
            raise ValueError("STT candidate model_name cannot be empty.")


@dataclass(frozen=True, slots=True)
class VoiceSTTResult:
    status: VoiceSTTRuntimeStatus
    operation: VoiceSTTOperation
    transcript: VoiceTranscript | None
    candidate: VoiceSTTTranscriptCandidate | None
    model: VoiceSTTModelInfo | None
    message: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return (
            self.status
            in {
                VoiceSTTRuntimeStatus.READY,
                VoiceSTTRuntimeStatus.TRANSCRIBING,
            }
            and self.transcript is not None
            and self.candidate is not None
        )

    @property
    def safe_for_action(self) -> bool:
        return self.candidate is not None and self.candidate.safe_for_action


@dataclass(frozen=True, slots=True)
class VoiceSTTSnapshot:
    status: VoiceSTTRuntimeStatus
    partial_model: VoiceSTTModelInfo | None
    final_model: VoiceSTTModelInfo | None
    partial_transcripts: int
    final_transcripts: int
    empty_results: int
    failed_results: int
    low_confidence_results: int
    last_text: str | None
    last_latency_ms: float | None
    last_safety: VoiceSTTTranscriptSafety | None
    last_error: str | None
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


class VoiceSTTAdapter(Protocol):
    def prepare(
        self,
        config: VoiceRuntimeConfig,
        policy: VoiceSTTPolicy,
    ) -> tuple[VoiceSTTModelInfo, VoiceSTTModelInfo]:
        raise NotImplementedError

    def transcribe(
        self,
        request: VoiceSTTRequest,
        config: VoiceRuntimeConfig,
        policy: VoiceSTTPolicy,
    ) -> tuple[str, float, str]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class FasterWhisperSTTAdapter:
    """
    Production local STT adapter using faster-whisper.

    Dual-lane design:
    - FAST_PARTIAL: low-latency early understanding while user speaks.
    - ACCURATE_FINAL: higher-confidence final transcript after speech end.
    """

    def __init__(self) -> None:
        self._partial_model: Any | None = None
        self._final_model: Any | None = None

    def prepare(
        self,
        config: VoiceRuntimeConfig,
        policy: VoiceSTTPolicy,
    ) -> tuple[VoiceSTTModelInfo, VoiceSTTModelInfo]:
        module = importlib.import_module("faster_whisper")
        model_cls = module.WhisperModel

        if policy.prewarm_on_prepare:
            self._partial_model = model_cls(
                policy.partial_model_name,
                device=policy.device,
                compute_type=policy.compute_type,
            )
            self._final_model = model_cls(
                policy.final_model_name,
                device=policy.device,
                compute_type=policy.compute_type,
            )

        partial = VoiceSTTModelInfo(
            provider="faster_whisper",
            model_name=policy.partial_model_name,
            device=policy.device,
            compute_type=policy.compute_type,
            language=config.stt_language,
            mode=VoiceSTTMode.FAST_PARTIAL,
            health=VoiceDeviceHealth.READY,
        )
        final = VoiceSTTModelInfo(
            provider="faster_whisper",
            model_name=policy.final_model_name,
            device=policy.device,
            compute_type=policy.compute_type,
            language=config.stt_language,
            mode=VoiceSTTMode.ACCURATE_FINAL,
            health=VoiceDeviceHealth.READY,
        )
        return partial, final

    def transcribe(
        self,
        request: VoiceSTTRequest,
        config: VoiceRuntimeConfig,
        policy: VoiceSTTPolicy,
    ) -> tuple[str, float, str]:
        module = importlib.import_module("faster_whisper")
        model_cls = module.WhisperModel

        if request.mode == VoiceSTTMode.FAST_PARTIAL:
            model_name = policy.partial_model_name
            beam_size = policy.partial_beam_size
            if self._partial_model is None:
                self._partial_model = model_cls(
                    model_name,
                    device=policy.device,
                    compute_type=policy.compute_type,
                )
            model = self._partial_model
        else:
            model_name = policy.final_model_name
            beam_size = policy.final_beam_size
            if self._final_model is None:
                self._final_model = model_cls(
                    model_name,
                    device=policy.device,
                    compute_type=policy.compute_type,
                )
            model = self._final_model

        wav_path = _frames_to_temp_wav(request.frames, config)
        try:
            segments, info = model.transcribe(
                str(wav_path),
                language=config.stt_language,
                beam_size=beam_size,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            text_parts: list[str] = []
            for segment in segments:
                text = str(getattr(segment, "text", "")).strip()
                if text:
                    text_parts.append(text)

            text = " ".join(text_parts).strip()
            confidence = float(getattr(info, "language_probability", 0.5))
            return text, max(0.0, min(1.0, confidence)), model_name
        finally:
            wav_path.unlink(missing_ok=True)

    def close(self) -> None:
        self._partial_model = None
        self._final_model = None


class VoiceSTTRuntime:
    """
    Step 51D Dual-Lane STT Runtime.

    This is the voice understanding front door.

    It does not:
    - call Ollama
    - generate responses
    - call TTS
    - execute tools
    - allow partial transcripts to execute actions
    """

    def __init__(
        self,
        *,
        config: VoiceRuntimeConfig | None = None,
        adapter: VoiceSTTAdapter | None = None,
        policy: VoiceSTTPolicy | None = None,
    ) -> None:
        self._config = config or default_voice_runtime_config()
        self._adapter = adapter or FasterWhisperSTTAdapter()
        self._policy = policy or VoiceSTTPolicy()
        self._status = VoiceSTTRuntimeStatus.CREATED
        self._partial_model: VoiceSTTModelInfo | None = None
        self._final_model: VoiceSTTModelInfo | None = None
        self._partial_transcripts = 0
        self._final_transcripts = 0
        self._empty_results = 0
        self._failed_results = 0
        self._low_confidence_results = 0
        self._last_text: str | None = None
        self._last_latency_ms: float | None = None
        self._last_safety: VoiceSTTTranscriptSafety | None = None
        self._last_error: str | None = None

    def prepare(self) -> VoiceSTTResult:
        try:
            partial, final = self._adapter.prepare(
                self._config,
                self._policy,
            )
            self._partial_model = partial
            self._final_model = final
            self._status = VoiceSTTRuntimeStatus.READY
            self._last_error = None
            return self._result(
                operation=VoiceSTTOperation.PREPARE,
                status=self._status,
                transcript=None,
                candidate=None,
                message="dual-lane STT runtime prepared",
            )
        except Exception as exc:
            self._status = VoiceSTTRuntimeStatus.FAILED
            self._failed_results += 1
            self._last_error = str(exc)
            return self._result(
                operation=VoiceSTTOperation.PREPARE,
                status=self._status,
                transcript=None,
                candidate=None,
                message="dual-lane STT prepare failed",
                metadata={"error": str(exc)},
            )

    def transcribe_partial(
        self,
        frames: tuple[VoiceInputFrame, ...],
    ) -> VoiceSTTResult:
        return self.transcribe(
            VoiceSTTRequest(
                frames=frames,
                mode=VoiceSTTMode.FAST_PARTIAL,
                allow_action_candidate=False,
            )
        )

    def transcribe_final(
        self,
        frames: tuple[VoiceInputFrame, ...],
        *,
        allow_action_candidate: bool = False,
    ) -> VoiceSTTResult:
        return self.transcribe(
            VoiceSTTRequest(
                frames=frames,
                mode=VoiceSTTMode.ACCURATE_FINAL,
                allow_action_candidate=allow_action_candidate,
            )
        )

    def transcribe(self, request: VoiceSTTRequest) -> VoiceSTTResult:
        if self._partial_model is None or self._final_model is None:
            prepared = self.prepare()
            if prepared.status == VoiceSTTRuntimeStatus.FAILED:
                return prepared

        operation = _operation_for_mode(request.mode)
        started = time.perf_counter()

        try:
            self._status = VoiceSTTRuntimeStatus.TRANSCRIBING
            text, confidence, model_name = self._adapter.transcribe(
                request,
                self._config,
                self._policy,
            )
        except Exception as exc:
            self._status = VoiceSTTRuntimeStatus.FAILED
            self._failed_results += 1
            self._last_error = str(exc)
            return self._result(
                operation=operation,
                status=self._status,
                transcript=None,
                candidate=None,
                message="STT transcription failed",
                metadata={"error": str(exc)},
            )

        latency_ms = (time.perf_counter() - started) * 1000.0
        self._last_latency_ms = latency_ms

        if not text.strip():
            return self._empty_result(operation=operation)

        min_confidence = _min_confidence_for_mode(
            mode=request.mode,
            policy=self._policy,
        )
        if confidence < min_confidence:
            self._low_confidence_results += 1
            self._status = VoiceSTTRuntimeStatus.DEGRADED
            return self._result(
                operation=operation,
                status=self._status,
                transcript=None,
                candidate=None,
                message="STT confidence below policy threshold",
                metadata={
                    "mode": request.mode.value,
                    "confidence": confidence,
                    "min_confidence": min_confidence,
                    "text": text,
                    "latency_ms": latency_ms,
                },
            )

        safety = _safety_for_transcript(
            mode=request.mode,
            confidence=confidence,
            allow_action_candidate=request.allow_action_candidate,
            policy=self._policy,
        )
        safe_for_action = safety == VoiceSTTTranscriptSafety.SAFE_FOR_ACTION

        candidate = VoiceSTTTranscriptCandidate(
            text=text,
            confidence=confidence,
            mode=request.mode,
            safety=safety,
            safe_for_action=safe_for_action,
            latency_ms=latency_ms,
            model_name=model_name,
        )

        transcript = VoiceTranscript(
            transcript_id=make_voice_transcript_id(),
            session_id=request.frames[0].session_id,
            segment_id=make_voice_segment_id(),
            kind=_transcript_kind_for_mode(request.mode),
            text=text,
            confidence=confidence,
            created_at=utc_now(),
            metadata={
                "mode": request.mode.value,
                "model_name": model_name,
                "latency_ms": latency_ms,
                "safe_for_action": safe_for_action,
                "safety": safety.value,
                "frame_count": len(request.frames),
            },
        )

        if request.mode == VoiceSTTMode.FAST_PARTIAL:
            self._partial_transcripts += 1
        else:
            self._final_transcripts += 1

        self._last_text = text
        self._last_safety = safety
        self._last_error = None
        self._status = VoiceSTTRuntimeStatus.READY

        return self._result(
            operation=operation,
            status=VoiceSTTRuntimeStatus.TRANSCRIBING,
            transcript=transcript,
            candidate=candidate,
            message="STT transcript produced",
            metadata={
                "mode": request.mode.value,
                "confidence": confidence,
                "latency_ms": latency_ms,
                "safety": safety.value,
                "safe_for_action": safe_for_action,
            },
        )

    def reset(self) -> VoiceSTTResult:
        self._adapter.close()
        self._partial_model = None
        self._final_model = None
        self._status = VoiceSTTRuntimeStatus.CREATED
        self._last_text = None
        self._last_latency_ms = None
        self._last_safety = None
        self._last_error = None
        return self._result(
            operation=VoiceSTTOperation.RESET,
            status=self._status,
            transcript=None,
            candidate=None,
            message="STT runtime reset",
        )

    def snapshot(self) -> VoiceSTTSnapshot:
        return VoiceSTTSnapshot(
            status=self._status,
            partial_model=self._partial_model,
            final_model=self._final_model,
            partial_transcripts=self._partial_transcripts,
            final_transcripts=self._final_transcripts,
            empty_results=self._empty_results,
            failed_results=self._failed_results,
            low_confidence_results=self._low_confidence_results,
            last_text=self._last_text,
            last_latency_ms=self._last_latency_ms,
            last_safety=self._last_safety,
            last_error=self._last_error,
            created_at=utc_now(),
        )

    def _empty_result(
        self,
        *,
        operation: VoiceSTTOperation,
    ) -> VoiceSTTResult:
        self._empty_results += 1
        self._status = (
            VoiceSTTRuntimeStatus.DEGRADED
            if self._empty_results
            >= self._policy.max_empty_results_before_degraded
            else VoiceSTTRuntimeStatus.READY
        )
        return self._result(
            operation=operation,
            status=self._status,
            transcript=None,
            candidate=None,
            message="STT returned empty transcript",
        )

    def _result(
        self,
        *,
        operation: VoiceSTTOperation,
        status: VoiceSTTRuntimeStatus,
        transcript: VoiceTranscript | None,
        candidate: VoiceSTTTranscriptCandidate | None,
        message: str,
        metadata: dict[str, object] | None = None,
    ) -> VoiceSTTResult:
        model = _model_for_operation(
            operation=operation,
            partial_model=self._partial_model,
            final_model=self._final_model,
        )
        return VoiceSTTResult(
            status=status,
            operation=operation,
            transcript=transcript,
            candidate=candidate,
            model=model,
            message=message,
            created_at=utc_now(),
            metadata=metadata or {},
        )


def _model_for_operation(
    *,
    operation: VoiceSTTOperation,
    partial_model: VoiceSTTModelInfo | None,
    final_model: VoiceSTTModelInfo | None,
) -> VoiceSTTModelInfo | None:
    if operation == VoiceSTTOperation.TRANSCRIBE_PARTIAL:
        return partial_model
    if operation == VoiceSTTOperation.TRANSCRIBE_FINAL:
        return final_model
    return None


def _operation_for_mode(mode: VoiceSTTMode) -> VoiceSTTOperation:
    if mode == VoiceSTTMode.FAST_PARTIAL:
        return VoiceSTTOperation.TRANSCRIBE_PARTIAL
    return VoiceSTTOperation.TRANSCRIBE_FINAL


def _transcript_kind_for_mode(mode: VoiceSTTMode) -> VoiceTranscriptKind:
    if mode == VoiceSTTMode.FAST_PARTIAL:
        return VoiceTranscriptKind.PARTIAL
    return VoiceTranscriptKind.FINAL


def _min_confidence_for_mode(
    *,
    mode: VoiceSTTMode,
    policy: VoiceSTTPolicy,
) -> float:
    if mode == VoiceSTTMode.FAST_PARTIAL:
        return policy.min_partial_confidence
    return policy.min_final_confidence


def _safety_for_transcript(
    *,
    mode: VoiceSTTMode,
    confidence: float,
    allow_action_candidate: bool,
    policy: VoiceSTTPolicy,
) -> VoiceSTTTranscriptSafety:
    if mode == VoiceSTTMode.FAST_PARTIAL:
        return VoiceSTTTranscriptSafety.PREDICTION_ONLY

    if confidence < policy.min_final_confidence:
        return VoiceSTTTranscriptSafety.NEEDS_CLARIFICATION

    if allow_action_candidate and confidence >= policy.min_action_confidence:
        return VoiceSTTTranscriptSafety.SAFE_FOR_ACTION

    return VoiceSTTTranscriptSafety.SAFE_FOR_DIALOGUE


def _frames_to_temp_wav(
    frames: tuple[VoiceInputFrame, ...],
    config: VoiceRuntimeConfig,
) -> Path:
    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    ) as tmp:
        path = Path(tmp.name)

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(config.channels)
        wav.setsampwidth(2)
        wav.setframerate(config.sample_rate_hz)
        wav.writeframes(b"".join(frame.data for frame in frames))

    return path