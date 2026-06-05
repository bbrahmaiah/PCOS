from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from jarvis.voice.contracts import (
    VoiceInputFrame,
    VoiceInputFrameKind,
    VoiceSegmentId,
    VoiceSpeechSegment,
    VoiceSpeechSegmentStatus,
    make_voice_segment_id,
    utc_now,
)


class VoiceActivityRuntimeStatus(StrEnum):
    CREATED = "created"
    READY = "ready"
    LISTENING = "listening"
    SPEECH_ACTIVE = "speech_active"
    SEGMENT_ENDED = "segment_ended"
    DEGRADED = "degraded"
    FAILED = "failed"


class VoiceActivityOperation(StrEnum):
    ANALYZE_FRAME = "analyze_frame"
    RESET = "reset"
    SNAPSHOT = "snapshot"


class VoiceActivityDecision(StrEnum):
    SILENCE = "silence"
    NOISE = "noise"
    SPEECH_STARTED = "speech_started"
    SPEECH_CONTINUED = "speech_continued"
    HOLDING_FOR_COMPLETION = "holding_for_completion"
    SPEECH_ENDED = "speech_ended"


@dataclass(frozen=True, slots=True)
class VoiceActivityPolicy:
    min_energy: float = 350.0
    speech_start_ratio: float = 3.0
    start_trigger_frames: int = 2
    min_speech_ms: int = 240
    end_silence_ms: int = 900
    max_segment_ms: int = 30_000
    noise_adaptation_rate: float = 0.05
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.min_energy <= 0:
            raise ValueError("min_energy must be positive.")
        if self.speech_start_ratio <= 1.0:
            raise ValueError("speech_start_ratio must be greater than 1.")
        if self.start_trigger_frames < 1:
            raise ValueError("start_trigger_frames must be positive.")
        if self.min_speech_ms <= 0:
            raise ValueError("min_speech_ms must be positive.")
        if self.end_silence_ms <= 0:
            raise ValueError("end_silence_ms must be positive.")
        if self.max_segment_ms <= self.min_speech_ms:
            raise ValueError("max_segment_ms must exceed min_speech_ms.")
        if not 0.0 < self.noise_adaptation_rate <= 1.0:
            raise ValueError("noise_adaptation_rate must be 0..1.")


@dataclass(frozen=True, slots=True)
class VoiceActivityResult:
    status: VoiceActivityRuntimeStatus
    operation: VoiceActivityOperation
    decision: VoiceActivityDecision
    segment: VoiceSpeechSegment | None
    energy: float
    threshold: float
    message: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def speech_started(self) -> bool:
        return self.decision == VoiceActivityDecision.SPEECH_STARTED

    @property
    def speech_ended(self) -> bool:
        return self.decision == VoiceActivityDecision.SPEECH_ENDED

    @property
    def holding_for_completion(self) -> bool:
        return self.decision == VoiceActivityDecision.HOLDING_FOR_COMPLETION


@dataclass(frozen=True, slots=True)
class VoiceActivitySnapshot:
    status: VoiceActivityRuntimeStatus
    current_segment_id: VoiceSegmentId | None
    analyzed_frames: int
    speech_segments: int
    speech_ms: int
    silence_ms: int
    pending_start_frames: int
    noise_floor: float
    last_energy: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


class VoiceActivityRuntime:
    """
    Step 51C VAD / speech activity runtime.

    This runtime detects speech start/end from microphone frames.

    It does not:
    - transcribe speech
    - call Ollama
    - call TTS
    - generate responses
    - execute tools

    It only decides when a speech segment starts, remains active,
    waits through natural pauses, and ends safely.
    """

    def __init__(
        self,
        *,
        policy: VoiceActivityPolicy | None = None,
    ) -> None:
        self._policy = policy or VoiceActivityPolicy()
        self._status = VoiceActivityRuntimeStatus.READY
        self._current_segment_id: VoiceSegmentId | None = None
        self._segment_started_at: datetime | None = None
        self._analyzed_frames = 0
        self._speech_segments = 0
        self._speech_ms = 0
        self._silence_ms = 0
        self._pending_start_frames = 0
        self._last_energy = 0.0
        self._noise_floor = self._policy.min_energy / 2.0

    def analyze_frame(self, frame: VoiceInputFrame) -> VoiceActivityResult:
        self._analyzed_frames += 1

        if frame.kind != VoiceInputFrameKind.PCM16_MONO:
            self._status = VoiceActivityRuntimeStatus.DEGRADED
            return self._result(
                decision=VoiceActivityDecision.NOISE,
                segment=None,
                energy=0.0,
                threshold=self._threshold(),
                message="unsupported voice frame kind",
                metadata={"frame_kind": frame.kind.value},
            )

        try:
            energy = _pcm16_rms(frame.data)
        except ValueError as exc:
            self._status = VoiceActivityRuntimeStatus.DEGRADED
            return self._result(
                decision=VoiceActivityDecision.NOISE,
                segment=None,
                energy=0.0,
                threshold=self._threshold(),
                message="invalid voice frame",
                metadata={"error": str(exc)},
            )

        self._last_energy = energy
        threshold = self._threshold()
        is_speech = energy >= threshold

        if self._current_segment_id is None:
            return self._analyze_in_listening_state(
                frame=frame,
                energy=energy,
                threshold=threshold,
                is_speech=is_speech,
            )

        return self._analyze_active_segment(
            frame=frame,
            energy=energy,
            threshold=threshold,
            is_speech=is_speech,
        )

    def reset(self) -> VoiceActivityResult:
        self._status = VoiceActivityRuntimeStatus.READY
        self._current_segment_id = None
        self._segment_started_at = None
        self._speech_ms = 0
        self._silence_ms = 0
        self._pending_start_frames = 0

        return self._result(
            decision=VoiceActivityDecision.SILENCE,
            segment=None,
            energy=self._last_energy,
            threshold=self._threshold(),
            message="voice activity runtime reset",
        )

    def snapshot(self) -> VoiceActivitySnapshot:
        return VoiceActivitySnapshot(
            status=self._status,
            current_segment_id=self._current_segment_id,
            analyzed_frames=self._analyzed_frames,
            speech_segments=self._speech_segments,
            speech_ms=self._speech_ms,
            silence_ms=self._silence_ms,
            pending_start_frames=self._pending_start_frames,
            noise_floor=self._noise_floor,
            last_energy=self._last_energy,
            created_at=utc_now(),
        )

    def _analyze_in_listening_state(
        self,
        *,
        frame: VoiceInputFrame,
        energy: float,
        threshold: float,
        is_speech: bool,
    ) -> VoiceActivityResult:
        self._status = VoiceActivityRuntimeStatus.LISTENING

        if not is_speech:
            self._pending_start_frames = 0
            self._adapt_noise_floor(energy)
            return self._result(
                decision=VoiceActivityDecision.SILENCE,
                segment=None,
                energy=energy,
                threshold=threshold,
                message="silence detected",
            )

        self._pending_start_frames += 1

        if self._pending_start_frames < self._policy.start_trigger_frames:
            return self._result(
                decision=VoiceActivityDecision.NOISE,
                segment=None,
                energy=energy,
                threshold=threshold,
                message="possible speech waiting for confirmation",
                metadata={
                    "pending_start_frames": self._pending_start_frames,
                },
            )

        segment_id = make_voice_segment_id()
        now = utc_now()
        self._current_segment_id = segment_id
        self._segment_started_at = now
        self._speech_segments += 1
        self._speech_ms = frame.duration_ms * self._pending_start_frames
        self._silence_ms = 0
        self._status = VoiceActivityRuntimeStatus.SPEECH_ACTIVE

        segment = VoiceSpeechSegment(
            segment_id=segment_id,
            session_id=frame.session_id,
            status=VoiceSpeechSegmentStatus.STARTED,
            started_at=now,
            confidence=_confidence(energy, threshold),
            frame_count=self._pending_start_frames,
        )

        return self._result(
            decision=VoiceActivityDecision.SPEECH_STARTED,
            segment=segment,
            energy=energy,
            threshold=threshold,
            message="speech started",
        )

    def _analyze_active_segment(
        self,
        *,
        frame: VoiceInputFrame,
        energy: float,
        threshold: float,
        is_speech: bool,
    ) -> VoiceActivityResult:
        if self._current_segment_id is None:
            raise RuntimeError("active segment missing segment id.")

        if is_speech:
            self._speech_ms += frame.duration_ms
            self._silence_ms = 0
            self._status = VoiceActivityRuntimeStatus.SPEECH_ACTIVE

            segment = VoiceSpeechSegment(
                segment_id=self._current_segment_id,
                session_id=frame.session_id,
                status=VoiceSpeechSegmentStatus.ACTIVE,
                started_at=self._segment_started_at or utc_now(),
                confidence=_confidence(energy, threshold),
                frame_count=max(1, self._speech_ms // frame.duration_ms),
            )

            return self._result(
                decision=VoiceActivityDecision.SPEECH_CONTINUED,
                segment=segment,
                energy=energy,
                threshold=threshold,
                message="speech continued",
            )

        self._silence_ms += frame.duration_ms
        self._adapt_noise_floor(energy)

        if self._should_end_segment():
            return self._end_segment(
                frame=frame,
                energy=energy,
                threshold=threshold,
                reason="speech ended after natural silence",
            )

        if self._speech_ms >= self._policy.max_segment_ms:
            return self._end_segment(
                frame=frame,
                energy=energy,
                threshold=threshold,
                reason="speech ended by maximum segment duration",
            )

        segment = VoiceSpeechSegment(
            segment_id=self._current_segment_id,
            session_id=frame.session_id,
            status=VoiceSpeechSegmentStatus.ACTIVE,
            started_at=self._segment_started_at or utc_now(),
            confidence=max(0.0, _confidence(energy, threshold) * 0.5),
            frame_count=max(1, self._speech_ms // frame.duration_ms),
            metadata={"silence_ms": self._silence_ms},
        )

        return self._result(
            decision=VoiceActivityDecision.HOLDING_FOR_COMPLETION,
            segment=segment,
            energy=energy,
            threshold=threshold,
            message="holding for natural speech completion",
            metadata={"silence_ms": self._silence_ms},
        )

    def _end_segment(
        self,
        *,
        frame: VoiceInputFrame,
        energy: float,
        threshold: float,
        reason: str,
    ) -> VoiceActivityResult:
        if self._current_segment_id is None:
            raise RuntimeError("cannot end missing voice segment.")

        ended_segment = VoiceSpeechSegment(
            segment_id=self._current_segment_id,
            session_id=frame.session_id,
            status=VoiceSpeechSegmentStatus.ENDED,
            started_at=self._segment_started_at or utc_now(),
            ended_at=utc_now(),
            confidence=_confidence(energy, threshold),
            frame_count=max(1, self._speech_ms // frame.duration_ms),
            metadata={
                "speech_ms": self._speech_ms,
                "silence_ms": self._silence_ms,
            },
        )

        self._current_segment_id = None
        self._segment_started_at = None
        self._speech_ms = 0
        self._silence_ms = 0
        self._pending_start_frames = 0
        self._status = VoiceActivityRuntimeStatus.SEGMENT_ENDED

        return self._result(
            decision=VoiceActivityDecision.SPEECH_ENDED,
            segment=ended_segment,
            energy=energy,
            threshold=threshold,
            message=reason,
        )

    def _should_end_segment(self) -> bool:
        return (
            self._speech_ms >= self._policy.min_speech_ms
            and self._silence_ms >= self._policy.end_silence_ms
        )

    def _threshold(self) -> float:
        return max(
            self._policy.min_energy,
            self._noise_floor * self._policy.speech_start_ratio,
        )

    def _adapt_noise_floor(self, energy: float) -> None:
        rate = self._policy.noise_adaptation_rate
        self._noise_floor = (self._noise_floor * (1.0 - rate)) + (
            energy * rate
        )

    def _result(
        self,
        *,
        decision: VoiceActivityDecision,
        segment: VoiceSpeechSegment | None,
        energy: float,
        threshold: float,
        message: str,
        metadata: dict[str, object] | None = None,
    ) -> VoiceActivityResult:
        return VoiceActivityResult(
            status=self._status,
            operation=VoiceActivityOperation.ANALYZE_FRAME,
            decision=decision,
            segment=segment,
            energy=energy,
            threshold=threshold,
            message=message,
            created_at=utc_now(),
            metadata=metadata or {},
        )


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


def _confidence(energy: float, threshold: float) -> float:
    if threshold <= 0:
        return 0.0
    return max(0.0, min(1.0, energy / (threshold * 2.0)))