from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from jarvis.presence.adapters import (
    EnergyVoiceActivityAdapter,
    EnergyWakeWordAdapter,
    RealMicrophoneAdapter,
    RealSpeechToTextAdapter,
    WakeWordDetection,
)
from jarvis.presence.models import AudioFrame, Transcript, VoiceActivity
from jarvis.presence.models.voice_activity import VoiceActivityState
from jarvis.runtime.observability.structured_logger import get_logger


@runtime_checkable
class SmokeMicrophoneAdapter(Protocol):
    """
    Minimal microphone contract required by the smoke harness.
    """

    def start(self) -> None:
        """Start microphone capture."""

    def stop(self) -> None:
        """Stop microphone capture."""

    def read_frame(self) -> AudioFrame | None:
        """Read one audio frame if available."""


@runtime_checkable
class SmokeWakeWordAdapter(Protocol):
    """
    Minimal wake adapter contract required by the smoke harness.
    """

    def detect(self, frame: AudioFrame) -> WakeWordDetection | None:
        """Return wake detection when wake is detected."""


@runtime_checkable
class SmokeVoiceActivityAdapter(Protocol):
    """
    Minimal VAD adapter contract required by the smoke harness.
    """

    def detect(self, frame: AudioFrame) -> VoiceActivity:
        """Return voice activity for one frame."""


@runtime_checkable
class SmokeSpeechToTextAdapter(Protocol):
    """
    Minimal STT adapter contract required by the smoke harness.
    """

    def transcribe(self, segment: CompletedSpeechSegment) -> Transcript | None:
        """Transcribe one completed speech segment."""


@dataclass(frozen=True, slots=True)
class CompletedSpeechSegment:
    """
    Completed speech audio segment for real-listening smoke tests.

    This intentionally matches the structural contract expected by
    RealSpeechToTextAdapter without depending on a concrete VAD model class.
    """

    segment_id: str
    audio_data: bytes
    sample_rate: int
    channels: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RealPresenceListeningSmokeConfig:
    """
    Controlled real-listening smoke configuration.

    Defaults are intentionally conservative:
    - short run
    - bounded frame count
    - wake required
    - stop after first transcript
    """

    duration_seconds: float = 10.0
    max_frames: int = 2_000
    frame_sleep_seconds: float = 0.005
    require_wake: bool = True
    stop_after_first_transcript: bool = True
    min_segment_frames: int = 1

    def validate(self) -> None:
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be greater than zero.")

        if self.max_frames <= 0:
            raise ValueError("max_frames must be greater than zero.")

        if self.frame_sleep_seconds < 0:
            raise ValueError("frame_sleep_seconds cannot be negative.")

        if self.min_segment_frames <= 0:
            raise ValueError("min_segment_frames must be greater than zero.")


@dataclass(frozen=True, slots=True)
class RealPresenceListeningSmokeReport:
    """
    Report for one controlled real-listening smoke run.
    """

    passed: bool
    started_at: datetime
    finished_at: datetime
    duration_ms: float
    frames_read: int
    wake_detected: bool
    speech_started: bool
    speech_completed: bool
    transcripts: tuple[Transcript, ...]
    errors: tuple[str, ...]

    @property
    def transcript_count(self) -> int:
        return len(self.transcripts)


class RealPresenceListeningSmokeHarness:
    """
    Controlled real-world listening harness.

    Responsibilities:
    - start microphone
    - process frames through wake/VAD/STT
    - produce a transcript report

    Non-responsibilities:
    - no cognition
    - no TTS
    - no playback
    - no event publishing
    - no laptop control
    """

    def __init__(
        self,
        *,
        config: RealPresenceListeningSmokeConfig | None = None,
        microphone: SmokeMicrophoneAdapter | None = None,
        wake_word: SmokeWakeWordAdapter | None = None,
        vad: SmokeVoiceActivityAdapter | None = None,
        stt: SmokeSpeechToTextAdapter | None = None,
    ) -> None:
        self._config = config or RealPresenceListeningSmokeConfig()
        self._config.validate()

        self._microphone = microphone or RealMicrophoneAdapter()
        self._wake_word = wake_word or EnergyWakeWordAdapter()
        self._vad = vad or EnergyVoiceActivityAdapter()
        self._stt = stt or RealSpeechToTextAdapter()

        self._logger = get_logger("presence.real_smoke")

    def run(self) -> RealPresenceListeningSmokeReport:
        started_at = datetime.now(UTC)
        started_perf = time.perf_counter()

        frames_read = 0
        wake_detected = not self._config.require_wake
        speech_started = False
        speech_completed = False
        errors: list[str] = []
        transcripts: list[Transcript] = []
        segment_frames: list[AudioFrame] = []

        try:
            self._microphone.start()
            deadline = time.monotonic() + self._config.duration_seconds

            while (
                time.monotonic() < deadline
                and frames_read < self._config.max_frames
            ):
                frame = self._microphone.read_frame()

                if frame is None:
                    if self._config.frame_sleep_seconds > 0:
                        time.sleep(self._config.frame_sleep_seconds)
                    continue

                frames_read += 1

                if not wake_detected:
                    detection = self._wake_word.detect(frame)

                    if detection is None:
                        continue

                    wake_detected = True
                    self._logger.info(
                        "real_smoke_wake_detected",
                        frame_id=frame.frame_id,
                        wake_word=detection.wake_word,
                        confidence=detection.confidence,
                    )

                activity = self._vad.detect(frame)

                if activity.state == VoiceActivityState.SPEECH_STARTED:
                    speech_started = True
                    segment_frames = [frame]
                    continue

                if activity.state == VoiceActivityState.SPEECH_CONTINUING:
                    if speech_started:
                        segment_frames.append(frame)
                    continue

                if activity.state == VoiceActivityState.SPEECH_ENDED:
                    if speech_started:
                        segment_frames.append(frame)
                        speech_completed = True
                        transcript = self._transcribe_segment(segment_frames)

                        if transcript is not None:
                            transcripts.append(transcript)

                        segment_frames = []
                        speech_started = False

                        if (
                            transcript is not None
                            and self._config.stop_after_first_transcript
                        ):
                            break

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            errors.append(error)
            self._logger.error("real_smoke_failed", error=error)

        finally:
            try:
                self._microphone.stop()
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")

        finished_at = datetime.now(UTC)
        duration_ms = (time.perf_counter() - started_perf) * 1000.0
        passed = not errors and len(transcripts) > 0

        report = RealPresenceListeningSmokeReport(
            passed=passed,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            frames_read=frames_read,
            wake_detected=wake_detected,
            speech_started=speech_started,
            speech_completed=speech_completed,
            transcripts=tuple(transcripts),
            errors=tuple(errors),
        )

        self._logger.info(
            "real_smoke_completed",
            passed=report.passed,
            frames_read=report.frames_read,
            wake_detected=report.wake_detected,
            speech_completed=report.speech_completed,
            transcript_count=report.transcript_count,
            duration_ms=round(report.duration_ms, 3),
        )

        return report

    def _transcribe_segment(
        self,
        frames: list[AudioFrame],
    ) -> Transcript | None:
        if len(frames) < self._config.min_segment_frames:
            return None

        first_frame = frames[0]
        audio_data = b"".join(frame.audio_data for frame in frames)

        segment = CompletedSpeechSegment(
            segment_id=uuid4().hex,
            audio_data=audio_data,
            sample_rate=first_frame.sample_rate,
            channels=first_frame.channels,
            metadata={
                "source": "real_presence_smoke",
                "frame_count": len(frames),
                "first_frame_id": first_frame.frame_id,
                "last_frame_id": frames[-1].frame_id,
            },
        )

        return self._stt.transcribe(segment)