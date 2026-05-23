from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Protocol, runtime_checkable

from pydantic import Field, field_validator

from jarvis.cognition.models import CognitionModel, SpokenResponseStyle
from jarvis.cognition.runtime import CognitionRuntime, CognitionRuntimeTurnResult
from jarvis.runtime.observability.structured_logger import get_logger


class VoiceCognitionTranscript(CognitionModel):
    """
    Transcript captured from a voice input path.

    This model stores text only. Audio remains owned by Presence/voice adapters.
    """

    text: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = "voice"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source")
    @classmethod
    def _source_required(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("source cannot be empty.")

        return cleaned


class VoiceCognitionPlaybackResult(CognitionModel):
    """
    Result of speaking a cognition response.
    """

    text: str
    started: bool = False
    completed: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def _text_required(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("text cannot be empty.")

        return cleaned


@runtime_checkable
class VoiceCognitionIO(Protocol):
    """
    Minimal voice I/O contract for real voice + cognition smoke.

    Real implementations can use microphone/STT/TTS/playback.
    Tests can use deterministic fakes.
    """

    @property
    def name(self) -> str:
        """Stable I/O name."""

    def listen_once(self) -> VoiceCognitionTranscript:
        """Capture one utterance and return a transcript."""

    def speak(self, text: str) -> VoiceCognitionPlaybackResult:
        """Speak one response."""


@dataclass(frozen=True, slots=True)
class VoiceCognitionSmokeConfig:
    """
    Configuration for the real voice + cognition smoke runner.
    """

    name: str = "voice_cognition_smoke"
    streaming: bool = False
    allow_tools: bool = False
    allow_memory_lookup: bool = True
    spoken_style: SpokenResponseStyle = SpokenResponseStyle.CONCISE
    fail_on_empty_transcript: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")


@dataclass(frozen=True, slots=True)
class VoiceCognitionSmokeReport:
    """
    Full report for one real voice + cognition smoke turn.
    """

    passed: bool
    transcript: VoiceCognitionTranscript | None
    runtime_result: CognitionRuntimeTurnResult | None
    playback: VoiceCognitionPlaybackResult | None
    reason: str | None = None

    @property
    def heard_text(self) -> str | None:
        if self.transcript is None:
            return None

        return self.transcript.text

    @property
    def response_text(self) -> str | None:
        if self.runtime_result is None or self.runtime_result.response is None:
            return None

        return self.runtime_result.response.text


class VoiceCognitionSmokeRunner:
    """
    Real voice + cognition smoke runner.

    Responsibilities:
    - receive one transcript from voice I/O
    - pass text through the assembled CognitionRuntime
    - shape/generated response through runtime spoken policy
    - send final text to voice output
    - prove safety: no direct laptop execution happens here

    Non-responsibilities:
    - no laptop action execution
    - no shell commands
    - no file writes/deletes
    - no permanent memory
    """

    def __init__(
        self,
        *,
        runtime: CognitionRuntime,
        voice_io: VoiceCognitionIO,
        config: VoiceCognitionSmokeConfig | None = None,
    ) -> None:
        self._config = config or VoiceCognitionSmokeConfig()
        self._config.validate()

        self._runtime = runtime
        self._voice_io = voice_io
        self._lock = RLock()
        self._logger = get_logger("cognition.voice_cognition_smoke")

        self._run_count = 0
        self._passed_count = 0
        self._failed_count = 0
        self._last_reason: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def run_once(self) -> VoiceCognitionSmokeReport:
        """
        Run one voice → cognition → voice smoke turn.
        """

        with self._lock:
            self._run_count += 1
            self._last_reason = None

        transcript = self._voice_io.listen_once()
        transcript_text = transcript.text.strip()

        if self._config.fail_on_empty_transcript and not transcript_text:
            return self._fail(
                transcript=transcript,
                runtime_result=None,
                playback=None,
                reason="empty transcript",
            )

        metadata = {
            "voice_io": self._voice_io.name,
            "voice_confidence": transcript.confidence,
            "voice_source": transcript.source,
        }

        if self._config.streaming:
            runtime_result = self._runtime.process_text_streaming(
                transcript_text,
                allow_tools=self._config.allow_tools,
                allow_memory_lookup=self._config.allow_memory_lookup,
                spoken_style=self._config.spoken_style,
                metadata=metadata,
            )

        else:
            runtime_result = self._runtime.process_text(
                transcript_text,
                allow_tools=self._config.allow_tools,
                allow_memory_lookup=self._config.allow_memory_lookup,
                spoken_style=self._config.spoken_style,
                metadata=metadata,
            )

        if runtime_result.response is None:
            return self._fail(
                transcript=transcript,
                runtime_result=runtime_result,
                playback=None,
                reason="cognition did not produce response",
            )

        playback = self._voice_io.speak(runtime_result.response.text)

        if not playback.completed:
            return self._fail(
                transcript=transcript,
                runtime_result=runtime_result,
                playback=playback,
                reason="playback did not complete",
            )

        report = VoiceCognitionSmokeReport(
            passed=True,
            transcript=transcript,
            runtime_result=runtime_result,
            playback=playback,
        )

        with self._lock:
            self._passed_count += 1
            self._last_reason = None

        self._logger.info(
            "voice_cognition_smoke_passed",
            runner=self.name,
            voice_io=self._voice_io.name,
            streaming=self._config.streaming,
            allow_tools=self._config.allow_tools,
            heard_text=transcript_text,
            response_text=runtime_result.response.text,
            action_plan_created=runtime_result.action_plan is not None,
        )

        return report

    def snapshot(self) -> dict[str, int | str | None]:
        """
        Return runner diagnostics.
        """

        with self._lock:
            return {
                "run_count": self._run_count,
                "passed_count": self._passed_count,
                "failed_count": self._failed_count,
                "last_reason": self._last_reason,
            }

    def _fail(
        self,
        *,
        transcript: VoiceCognitionTranscript | None,
        runtime_result: CognitionRuntimeTurnResult | None,
        playback: VoiceCognitionPlaybackResult | None,
        reason: str,
    ) -> VoiceCognitionSmokeReport:
        with self._lock:
            self._failed_count += 1
            self._last_reason = reason

        self._logger.error(
            "voice_cognition_smoke_failed",
            runner=self.name,
            voice_io=self._voice_io.name,
            reason=reason,
        )

        return VoiceCognitionSmokeReport(
            passed=False,
            transcript=transcript,
            runtime_result=runtime_result,
            playback=playback,
            reason=reason,
        )