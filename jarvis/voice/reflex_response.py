from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from jarvis.voice.contracts import VoiceTranscript, VoiceTranscriptKind
from jarvis.voice.transcript_attention_gate import normalize_transcript_text


class VoiceReflexResponseKind(StrEnum):
    NONE = "none"
    AVAILABILITY = "availability"
    STOP_PLAYBACK = "stop_playback"
    SHUTDOWN_SESSION = "shutdown_session"


@dataclass(frozen=True, slots=True)
class VoiceReflexResponsePolicy:
    enabled: bool = True
    min_confidence: float = 0.70
    wake_words: tuple[str, ...] = ("jarvis", "jervis", "jarves")
    availability_phrases: tuple[str, ...] = (
        "jarvis",
        "wake up jarvis",
        "jarvis wake up",
        "hey jarvis",
        "hello jarvis",
        "jarvis are you online",
        "jarvis are you there",
        "jarvis can you hear me",
        "are you online",
        "are you there",
        "can you hear me",
    )
    stop_phrases: tuple[str, ...] = (
        "stop",
        "wait",
        "pause",
        "quiet",
        "hold",
        "hold on",
        "enough",
        "stop talking",
        "be quiet",
    )
    shutdown_phrases: tuple[str, ...] = (
        "shutdown jarvis",
        "shut down jarvis",
        "jarvis shutdown",
        "jarvis shut down",
        "jarvis go offline",
        "go offline jarvis",
        "power down jarvis",
        "jarvis power down",
        "terminate jarvis",
        "exit jarvis",
    )
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be 0..1.")
        if not self.wake_words:
            raise ValueError("wake_words cannot be empty.")
        if not self.availability_phrases:
            raise ValueError("availability_phrases cannot be empty.")
        if not self.stop_phrases:
            raise ValueError("stop_phrases cannot be empty.")
        if not self.shutdown_phrases:
            raise ValueError("shutdown_phrases cannot be empty.")


@dataclass(frozen=True, slots=True)
class VoiceReflexResponseDecision:
    accepted: bool
    kind: VoiceReflexResponseKind
    normalized_text: str
    response_text: str | None
    should_speak: bool
    should_stop_playback: bool
    should_shutdown_session: bool
    should_continue_to_cognition: bool
    confidence: float
    reason: str
    metadata: dict[str, object] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "kind": self.kind.value,
            "normalized_text": self.normalized_text,
            "response_text": self.response_text,
            "should_speak": self.should_speak,
            "should_stop_playback": self.should_stop_playback,
            "should_shutdown_session": self.should_shutdown_session,
            "should_continue_to_cognition": self.should_continue_to_cognition,
            "confidence": self.confidence,
            "reason": self.reason,
            **self.metadata,
        }


class VoiceReflexResponseRuntime:
    """
    Narrow operational reflex lane for live voice.

    This is intentionally not a conversational answer generator. It only handles
    high-confidence local control phrases where waiting for the LLM would make
    the assistant feel unresponsive: availability pings and stop/playback
    interruption.
    """

    def __init__(
        self,
        *,
        policy: VoiceReflexResponsePolicy | None = None,
    ) -> None:
        self._policy = policy or VoiceReflexResponsePolicy()

    def evaluate(
        self,
        transcript: VoiceTranscript,
        *,
        assistant_speaking: bool,
    ) -> VoiceReflexResponseDecision:
        normalized = normalize_transcript_text(transcript.text)
        confidence = float(transcript.confidence)
        metadata = {
            "assistant_speaking": assistant_speaking,
            "operational_reflex_only": True,
            **self._policy.metadata,
        }

        if not self._policy.enabled:
            return self._rejected(
                normalized=normalized,
                confidence=confidence,
                reason="reflex_disabled",
                metadata=metadata,
            )
        if transcript.kind != VoiceTranscriptKind.FINAL:
            return self._rejected(
                normalized=normalized,
                confidence=confidence,
                reason="non_final_transcript",
                metadata=metadata,
            )
        if not normalized:
            return self._rejected(
                normalized=normalized,
                confidence=confidence,
                reason="empty_transcript",
                metadata=metadata,
            )
        if confidence < self._policy.min_confidence:
            return self._rejected(
                normalized=normalized,
                confidence=confidence,
                reason="low_confidence",
                metadata={
                    **metadata,
                    "min_confidence": self._policy.min_confidence,
                },
            )

        if normalized in self._normalized_shutdown_phrases():
            return VoiceReflexResponseDecision(
                accepted=True,
                kind=VoiceReflexResponseKind.SHUTDOWN_SESSION,
                normalized_text=normalized,
                response_text=None,
                should_speak=False,
                should_stop_playback=True,
                should_shutdown_session=True,
                should_continue_to_cognition=False,
                confidence=confidence,
                reason="shutdown_session_reflex",
                metadata={
                    **metadata,
                    "wake_detected": self._wake_detected(normalized),
                },
            )

        if normalized in self._normalized_stop_phrases():
            return VoiceReflexResponseDecision(
                accepted=True,
                kind=VoiceReflexResponseKind.STOP_PLAYBACK,
                normalized_text=normalized,
                response_text=None,
                should_speak=False,
                should_stop_playback=True,
                should_shutdown_session=False,
                should_continue_to_cognition=False,
                confidence=confidence,
                reason="stop_playback_reflex",
                metadata=metadata,
            )

        if normalized in self._normalized_availability_phrases():
            return self._rejected(
                normalized=normalized,
                confidence=confidence,
                reason="availability_requires_cognition",
                metadata={
                    **metadata,
                    "wake_detected": self._wake_detected(normalized),
                    "fixed_spoken_response_blocked": True,
                },
            )

        return self._rejected(
            normalized=normalized,
            confidence=confidence,
            reason="not_operational_reflex",
            metadata=metadata,
        )

    def _rejected(
        self,
        *,
        normalized: str,
        confidence: float,
        reason: str,
        metadata: dict[str, object],
    ) -> VoiceReflexResponseDecision:
        return VoiceReflexResponseDecision(
            accepted=False,
            kind=VoiceReflexResponseKind.NONE,
            normalized_text=normalized,
            response_text=None,
            should_speak=False,
            should_stop_playback=False,
            should_shutdown_session=False,
            should_continue_to_cognition=True,
            confidence=confidence,
            reason=reason,
            metadata=metadata,
        )

    def _normalized_availability_phrases(self) -> frozenset[str]:
        return frozenset(
            normalize_transcript_text(phrase)
            for phrase in self._policy.availability_phrases
        )

    def _normalized_stop_phrases(self) -> frozenset[str]:
        return frozenset(
            normalize_transcript_text(phrase)
            for phrase in self._policy.stop_phrases
        )

    def _normalized_shutdown_phrases(self) -> frozenset[str]:
        return frozenset(
            normalize_transcript_text(phrase)
            for phrase in self._policy.shutdown_phrases
        )

    def _wake_detected(self, normalized_text: str) -> bool:
        padded = f" {normalized_text} "
        return any(
            f" {normalize_transcript_text(wake_word)} " in padded
            for wake_word in self._policy.wake_words
        )
