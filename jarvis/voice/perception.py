from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from jarvis.voice.contracts import (
    VoiceTranscript,
    make_voice_transcript_id,
    utc_now,
)
from jarvis.voice.transcript_attention_gate import normalize_transcript_text


class VoicePerceptionIntentState(StrEnum):
    NOISE = "noise"
    WAKE_CANDIDATE = "wake_candidate"
    CAPTURING = "capturing"
    STABILIZING = "stabilizing"
    READY_FOR_ROUTING = "ready_for_routing"
    INTERRUPTION = "interruption"


@dataclass(frozen=True, slots=True)
class VoicePerceptionPolicy:
    wake_words: frozenset[str] = frozenset({"jarvis", "jervis", "jarves"})
    known_noise_texts: frozenset[str] = frozenset(
        {
            "boom",
            "thank you",
            "thanks",
            "bye",
            "buh bye",
            "goodbye",
            "happy holidays",
            "i am sorry",
            "i'm sorry",
        }
    )
    partial_history_limit: int = 5
    min_noise_confidence: float = 0.28
    min_ready_stability: float = 0.68
    min_ready_confidence: float = 0.55
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.partial_history_limit < 1:
            raise ValueError("partial_history_limit must be positive.")
        if not 0.0 <= self.min_noise_confidence <= 1.0:
            raise ValueError("min_noise_confidence must be 0..1.")
        if not 0.0 <= self.min_ready_stability <= 1.0:
            raise ValueError("min_ready_stability must be 0..1.")
        if not 0.0 <= self.min_ready_confidence <= 1.0:
            raise ValueError("min_ready_confidence must be 0..1.")


@dataclass(frozen=True, slots=True)
class VoicePerceptionPacket:
    transcript: VoiceTranscript
    normalized_text: str
    confidence: float
    stability: float
    intent_state: VoicePerceptionIntentState
    final: bool
    wake_detected: bool
    repeated_partial_count: int
    observed_partial_count: int
    reason: str
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def ready_for_routing(self) -> bool:
        return self.intent_state == VoicePerceptionIntentState.READY_FOR_ROUTING

    def to_metadata(self) -> dict[str, object]:
        return {
            "normalized_text": self.normalized_text,
            "confidence": self.confidence,
            "stability": self.stability,
            "intent_state": self.intent_state.value,
            "final": self.final,
            "wake_detected": self.wake_detected,
            "repeated_partial_count": self.repeated_partial_count,
            "observed_partial_count": self.observed_partial_count,
            "reason": self.reason,
            **self.metadata,
        }


class VoicePerceptionRuntime:
    """
    Lightweight real-time perception layer for voice.

    It never calls an LLM and never executes tools. Its job is to convert
    raw STT transcripts into evidence the cognitive router can trust.
    """

    def __init__(
        self,
        *,
        policy: VoicePerceptionPolicy | None = None,
    ) -> None:
        self._policy = policy or VoicePerceptionPolicy()
        self._partial_history: list[str] = []
        self._observed_partials = 0
        self._packets = 0
        self._noise_packets = 0
        self._ready_packets = 0
        self._interruption_packets = 0
        self._last_packet: VoicePerceptionPacket | None = None

    def observe_partial(
        self,
        transcript: VoiceTranscript,
        *,
        assistant_speaking: bool = False,
    ) -> VoicePerceptionPacket:
        self._observed_partials += 1
        normalized = normalize_transcript_text(transcript.text)
        self._partial_history.append(normalized)
        if len(self._partial_history) > self._policy.partial_history_limit:
            self._partial_history = self._partial_history[
                -self._policy.partial_history_limit :
            ]

        return self._remember(
            self._make_packet(
                transcript=transcript,
                normalized_text=normalized,
                final=False,
                assistant_speaking=assistant_speaking,
            )
        )

    def observe_final(
        self,
        transcript: VoiceTranscript,
        *,
        assistant_speaking: bool = False,
    ) -> VoicePerceptionPacket:
        normalized = normalize_transcript_text(transcript.text)
        packet = self._remember(
            self._make_packet(
                transcript=transcript,
                normalized_text=normalized,
                final=True,
                assistant_speaking=assistant_speaking,
            )
        )
        self._partial_history = []
        self._observed_partials = 0
        return packet

    def reset(self) -> None:
        self._partial_history = []
        self._observed_partials = 0
        self._last_packet = None

    def snapshot(self) -> dict[str, object]:
        return {
            "packets": self._packets,
            "noise_packets": self._noise_packets,
            "ready_packets": self._ready_packets,
            "interruption_packets": self._interruption_packets,
            "partial_history_size": len(self._partial_history),
            "observed_partial_count": self._observed_partials,
            "last_packet": (
                None if self._last_packet is None else self._last_packet.to_metadata()
            ),
        }

    def _make_packet(
        self,
        *,
        transcript: VoiceTranscript,
        normalized_text: str,
        final: bool,
        assistant_speaking: bool,
    ) -> VoicePerceptionPacket:
        confidence = max(0.0, min(1.0, transcript.confidence))
        wake_detected = _contains_wake(
            normalized_text=normalized_text,
            wake_words=self._policy.wake_words,
        )
        repeated_partial_count = self._repeated_partial_count(normalized_text)
        stability = self._stability_score(
            transcript=transcript,
            normalized_text=normalized_text,
            confidence=confidence,
            final=final,
            wake_detected=wake_detected,
            repeated_partial_count=repeated_partial_count,
        )
        intent_state, reason = self._intent_state(
            normalized_text=normalized_text,
            confidence=confidence,
            stability=stability,
            final=final,
            wake_detected=wake_detected,
            assistant_speaking=assistant_speaking,
            repeated_partial_count=repeated_partial_count,
        )

        return VoicePerceptionPacket(
            transcript=transcript,
            normalized_text=normalized_text,
            confidence=confidence,
            stability=stability,
            intent_state=intent_state,
            final=final,
            wake_detected=wake_detected,
            repeated_partial_count=repeated_partial_count,
            observed_partial_count=self._observed_partials,
            reason=reason,
        )

    def _stability_score(
        self,
        *,
        transcript: VoiceTranscript,
        normalized_text: str,
        confidence: float,
        final: bool,
        wake_detected: bool,
        repeated_partial_count: int,
    ) -> float:
        explicit = _metadata_float(transcript.metadata, "stability")
        if explicit is None:
            explicit = _metadata_float(transcript.metadata, "transcript_stability")
        if explicit is not None:
            return explicit

        word_count = len(normalized_text.split())
        word_factor = min(1.0, word_count / 4.0)
        repeat_factor = min(1.0, repeated_partial_count / 3.0)
        final_bonus = 0.08 if final else 0.0
        wake_bonus = 0.10 if wake_detected else 0.0
        noise_penalty = (
            0.30 if normalized_text in self._policy.known_noise_texts else 0.0
        )

        return max(
            0.0,
            min(
                1.0,
                (confidence * 0.54)
                + (word_factor * 0.20)
                + (repeat_factor * 0.12)
                + final_bonus
                + wake_bonus
                - noise_penalty,
            ),
        )

    def _intent_state(
        self,
        *,
        normalized_text: str,
        confidence: float,
        stability: float,
        final: bool,
        wake_detected: bool,
        assistant_speaking: bool,
        repeated_partial_count: int,
    ) -> tuple[VoicePerceptionIntentState, str]:
        if confidence < self._policy.min_noise_confidence:
            return VoicePerceptionIntentState.NOISE, "below_noise_confidence"
        if normalized_text in self._policy.known_noise_texts and not wake_detected:
            return VoicePerceptionIntentState.NOISE, "known_background_or_stt_noise"
        if assistant_speaking:
            return (
                VoicePerceptionIntentState.INTERRUPTION,
                "assistant_speaking_user_input_detected",
            )
        if not final and wake_detected:
            return VoicePerceptionIntentState.WAKE_CANDIDATE, "wake_word_partial"
        if not final:
            if repeated_partial_count >= 2:
                return (
                    VoicePerceptionIntentState.STABILIZING,
                    "partial_hypothesis_repeating",
                )
            return VoicePerceptionIntentState.CAPTURING, "partial_hypothesis_capturing"
        if (
            confidence >= self._policy.min_ready_confidence
            and stability >= self._policy.min_ready_stability
        ):
            return VoicePerceptionIntentState.READY_FOR_ROUTING, "final_stable_enough"
        return VoicePerceptionIntentState.STABILIZING, "final_not_stable_enough"

    def _repeated_partial_count(self, normalized_text: str) -> int:
        if not normalized_text:
            return 0
        return sum(1 for item in self._partial_history if item == normalized_text)

    def _remember(
        self,
        packet: VoicePerceptionPacket,
    ) -> VoicePerceptionPacket:
        self._packets += 1
        self._last_packet = packet
        if packet.intent_state == VoicePerceptionIntentState.NOISE:
            self._noise_packets += 1
        elif packet.intent_state == VoicePerceptionIntentState.READY_FOR_ROUTING:
            self._ready_packets += 1
        elif packet.intent_state == VoicePerceptionIntentState.INTERRUPTION:
            self._interruption_packets += 1
        return packet


def enrich_transcript_with_perception(
    transcript: VoiceTranscript,
    packet: VoicePerceptionPacket,
) -> VoiceTranscript:
    metadata = {
        **transcript.metadata,
        "perception": packet.to_metadata(),
        "perception_intent_state": packet.intent_state.value,
        "perception_confidence": packet.confidence,
        "perception_stability": packet.stability,
    }
    return VoiceTranscript(
        transcript_id=make_voice_transcript_id(),
        session_id=transcript.session_id,
        segment_id=transcript.segment_id,
        kind=transcript.kind,
        text=transcript.text,
        confidence=transcript.confidence,
        created_at=utc_now(),
        metadata=metadata,
    )


def _contains_wake(
    *,
    normalized_text: str,
    wake_words: frozenset[str],
) -> bool:
    padded = f" {normalized_text} "
    return any(f" {normalize_transcript_text(wake)} " in padded for wake in wake_words)


def _metadata_float(metadata: dict[str, object], key: str) -> float | None:
    value = metadata.get(key)
    if not isinstance(value, int | float):
        return None
    return max(0.0, min(1.0, float(value)))
