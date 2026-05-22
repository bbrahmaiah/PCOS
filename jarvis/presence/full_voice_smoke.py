from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from jarvis.presence.adapters import (
    EnergyVoiceActivityAdapter,
    EnergyWakeWordAdapter,
    PlaybackResult,
    RealAudioPlaybackAdapter,
    RealMicrophoneAdapter,
    RealSpeechToTextAdapter,
    RealTextToSpeechAdapter,
    WakeWordDetection,
)
from jarvis.presence.models import AudioFrame, SpeechChunk, Transcript, VoiceActivity
from jarvis.presence.models.transcript import TranscriptKind
from jarvis.presence.models.voice_activity import VoiceActivityState
from jarvis.runtime.observability.structured_logger import get_logger


@runtime_checkable
class FullVoiceMicrophoneAdapter(Protocol):
    def start(self) -> None:
        """Start microphone capture."""

    def stop(self) -> None:
        """Stop microphone capture."""

    def read_frame(self) -> AudioFrame | None:
        """Read one audio frame if available."""


@runtime_checkable
class FullVoiceWakeWordAdapter(Protocol):
    def detect(self, frame: AudioFrame) -> WakeWordDetection | None:
        """Return wake detection when wake is detected."""


@runtime_checkable
class FullVoiceActivityAdapter(Protocol):
    def detect(self, frame: AudioFrame) -> VoiceActivity:
        """Return voice activity for one frame."""


@runtime_checkable
class FullVoiceSpeechToTextAdapter(Protocol):
    def transcribe(self, segment: FullVoiceSpeechSegment) -> Transcript | None:
        """Transcribe one completed speech segment."""


@runtime_checkable
class FullVoiceTextToSpeechAdapter(Protocol):
    def synthesize(
        self,
        *,
        text: str,
        request_id: str,
        voice_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[SpeechChunk, ...]:
        """Synthesize response text into speech chunks."""


@runtime_checkable
class FullVoicePlaybackAdapter(Protocol):
    @property
    def is_playing(self) -> bool:
        """Whether playback is active."""

    def play(self, chunk: SpeechChunk) -> PlaybackResult:
        """Play one speech chunk."""

    def stop(self) -> None:
        """Stop playback."""


@dataclass(frozen=True, slots=True)
class FullVoiceSpeechSegment:
    """
    Completed speech segment for full voice smoke tests.
    """

    segment_id: str
    audio_data: bytes
    sample_rate: int
    channels: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FullVoiceSmokeTurn:
    """
    One completed full voice smoke turn.
    """

    transcript: Transcript
    response_text: str
    chunks: tuple[SpeechChunk, ...]
    playback_results: tuple[PlaybackResult, ...]


@dataclass(frozen=True, slots=True)
class FullVoiceSmokeConfig:
    """
    Controlled full voice smoke configuration.

    This harness is intentionally safe: it uses a fixed canned response instead
    of cognition/LLM/action execution.
    """

    duration_seconds: float = 15.0
    max_frames: int = 3_000
    frame_sleep_seconds: float = 0.005
    require_wake: bool = True
    stop_after_first_turn: bool = True
    min_segment_frames: int = 1
    response_text: str = "Yes sir. I heard you."

    def validate(self) -> None:
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be greater than zero.")

        if self.max_frames <= 0:
            raise ValueError("max_frames must be greater than zero.")

        if self.frame_sleep_seconds < 0:
            raise ValueError("frame_sleep_seconds cannot be negative.")

        if self.min_segment_frames <= 0:
            raise ValueError("min_segment_frames must be greater than zero.")

        if not self.response_text.strip():
            raise ValueError("response_text cannot be empty.")


@dataclass(frozen=True, slots=True)
class FullVoiceSmokeReport:
    """
    Report for one controlled full voice smoke run.
    """

    passed: bool
    started_at: datetime
    finished_at: datetime
    duration_ms: float
    frames_read: int
    wake_detected: bool
    speech_completed: bool
    turns: tuple[FullVoiceSmokeTurn, ...]
    errors: tuple[str, ...]

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def transcript_count(self) -> int:
        return len(self.turns)

    @property
    def playback_count(self) -> int:
        return sum(len(turn.playback_results) for turn in self.turns)


class FullVoiceSmokeHarness:
    """
    Controlled real full voice loop harness.

    Responsibilities:
    - capture real frames
    - pass through wake/VAD/STT
    - use a fixed safe response
    - synthesize response
    - play response

    Non-responsibilities:
    - no cognition
    - no LLM
    - no action execution
    - no laptop control
    - no permission elevation
    """

    def __init__(
        self,
        *,
        config: FullVoiceSmokeConfig | None = None,
        microphone: FullVoiceMicrophoneAdapter | None = None,
        wake_word: FullVoiceWakeWordAdapter | None = None,
        vad: FullVoiceActivityAdapter | None = None,
        stt: FullVoiceSpeechToTextAdapter | None = None,
        tts: FullVoiceTextToSpeechAdapter | None = None,
        playback: FullVoicePlaybackAdapter | None = None,
    ) -> None:
        self._config = config or FullVoiceSmokeConfig()
        self._config.validate()

        self._microphone = microphone or RealMicrophoneAdapter()
        self._wake_word = wake_word or EnergyWakeWordAdapter()
        self._vad = vad or EnergyVoiceActivityAdapter()
        self._stt = stt or RealSpeechToTextAdapter()
        self._tts = tts or RealTextToSpeechAdapter()
        self._playback = playback or RealAudioPlaybackAdapter()

        self._logger = get_logger("presence.full_voice_smoke")

    def run(self) -> FullVoiceSmokeReport:
        started_at = datetime.now(UTC)
        started_perf = time.perf_counter()

        frames_read = 0
        wake_detected = not self._config.require_wake
        speech_started = False
        speech_completed = False
        segment_frames: list[AudioFrame] = []
        turns: list[FullVoiceSmokeTurn] = []
        errors: list[str] = []

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
                        "full_voice_smoke_wake_detected",
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
                    if not speech_started:
                        continue

                    segment_frames.append(frame)
                    speech_completed = True

                    turn = self._complete_turn(segment_frames)

                    if turn is not None:
                        turns.append(turn)

                    segment_frames = []
                    speech_started = False

                    if turn is not None and self._config.stop_after_first_turn:
                        break

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            errors.append(error)
            self._logger.error("full_voice_smoke_failed", error=error)

        finally:
            try:
                if self._playback.is_playing:
                    self._playback.stop()
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")

            try:
                self._microphone.stop()
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")

        finished_at = datetime.now(UTC)
        duration_ms = (time.perf_counter() - started_perf) * 1000.0
        passed = not errors and len(turns) > 0 and all(
            turn.playback_results for turn in turns
        )

        report = FullVoiceSmokeReport(
            passed=passed,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            frames_read=frames_read,
            wake_detected=wake_detected,
            speech_completed=speech_completed,
            turns=tuple(turns),
            errors=tuple(errors),
        )

        self._logger.info(
            "full_voice_smoke_completed",
            passed=report.passed,
            frames_read=report.frames_read,
            wake_detected=report.wake_detected,
            speech_completed=report.speech_completed,
            turn_count=report.turn_count,
            playback_count=report.playback_count,
            duration_ms=round(report.duration_ms, 3),
        )

        return report

    def _complete_turn(
        self,
        frames: list[AudioFrame],
    ) -> FullVoiceSmokeTurn | None:
        if len(frames) < self._config.min_segment_frames:
            return None

        transcript = self._transcribe_frames(frames)

        if transcript is None:
            return None

        response_text = self._config.response_text.strip()
        request_id = uuid4().hex

        chunks = self._tts.synthesize(
            text=response_text,
            request_id=request_id,
            metadata={
                "source": "full_voice_smoke",
                "transcript_id": transcript.segment_id,
                "transcript_text": transcript.text,
            },
        )

        playback_results = tuple(self._playback.play(chunk) for chunk in chunks)

        return FullVoiceSmokeTurn(
            transcript=transcript,
            response_text=response_text,
            chunks=chunks,
            playback_results=playback_results,
        )

    def _transcribe_frames(
        self,
        frames: list[AudioFrame],
    ) -> Transcript | None:
        first_frame = frames[0]
        audio_data = b"".join(frame.audio_data for frame in frames)

        segment = FullVoiceSpeechSegment(
            segment_id=uuid4().hex,
            audio_data=audio_data,
            sample_rate=first_frame.sample_rate,
            channels=first_frame.channels,
            metadata={
                "source": "full_voice_smoke",
                "frame_count": len(frames),
                "first_frame_id": first_frame.frame_id,
                "last_frame_id": frames[-1].frame_id,
            },
        )

        return self._stt.transcribe(segment)


def make_canned_transcript(
    *,
    segment_id: str = "segment-1",
    text: str = "hello jarvis",
) -> Transcript:
    """
    Test/helper factory for a final transcript.
    """

    return Transcript(
        segment_id=segment_id,
        text=text,
        kind=TranscriptKind.FINAL,
        confidence=0.98,
        language="en",
        metadata={"source": "full_voice_smoke"},
    )