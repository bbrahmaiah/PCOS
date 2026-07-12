from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol


class TranscriptLike(Protocol):
    @property
    def text(self) -> str:
        raise NotImplementedError

    @property
    def confidence(self) -> float:
        raise NotImplementedError

    @property
    def kind(self) -> Any:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class TranscriptGateDecision:
    accepted: bool
    reason: str
    normalized_text: str
    wake_detected: bool
    attention_active: bool
    confidence: float
    word_count: int
    metadata: dict[str, object] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "normalized_text": self.normalized_text,
            "wake_detected": self.wake_detected,
            "attention_active": self.attention_active,
            "confidence": self.confidence,
            "word_count": self.word_count,
            **self.metadata,
        }


@dataclass(frozen=True, slots=True)
class TranscriptAttentionGatePolicy:
    wake_words: tuple[str, ...] = ("jarvis", "bob")
    attention_window_seconds: float = 45.0
    min_final_confidence: float = 0.65
    min_words_without_wake: int = 3
    min_words_when_attention_active: int = 1
    extend_attention_on_accept: bool = True
    duplicate_window_seconds: float = 12.0
    max_duplicate_count: int = 2
    require_attention_for_promoted_partials: bool = True
    require_wake_or_attention: bool = True
    known_silence_hallucinations: frozenset[str] = frozenset(
        {
            "thank you",
            "thank you bye bye",
            "on this case",
            "alexa",
            "caleb",
            "bye bye",
            "good luck yall",
            "good luck you all",
            "the child",
            "to the child",
            "its",
            "it s",
            "i lip",
            "i lit",
            "i lit back",
            "i lift",
            "i lift back",
            "we ll see you next time",
            "bless you",
            "good job",
            "good job paul",
            "good job john",
            "god",
            "so i ll see you soon",
            "service",
            "that will be something",
            "i m going to do something",
            "future",
            "done",
            "happy holidays",
            "i m sorry",
            "i miss up",
            "miss up",
            "buh bye",
            "bye",
            "goodbye",
            "i ll be right back",
            "i ll be right back bye",
            "ill be right back",
            "ill be right back bye",
            "john",
            "out of this car",
            "out of this country",
            "we ve got to risk it",
            "always at",
        }
    )

    def __post_init__(self) -> None:
        if not self.wake_words:
            raise ValueError("wake_words cannot be empty.")
        if self.attention_window_seconds <= 0:
            raise ValueError("attention_window_seconds must be positive.")
        if not 0.0 <= self.min_final_confidence <= 1.0:
            raise ValueError("min_final_confidence must be 0..1.")
        if self.min_words_without_wake < 1:
            raise ValueError("min_words_without_wake must be positive.")
        if self.min_words_when_attention_active < 1:
            raise ValueError("min_words_when_attention_active must be positive.")
        if self.duplicate_window_seconds <= 0:
            raise ValueError("duplicate_window_seconds must be positive.")
        if self.max_duplicate_count < 1:
            raise ValueError("max_duplicate_count must be positive.")


class TranscriptAttentionGate:
    def __init__(self, policy: TranscriptAttentionGatePolicy | None = None) -> None:
        self._policy = policy or TranscriptAttentionGatePolicy()
        self._attention_active_until = 0.0
        self._recent: deque[tuple[float, str]] = deque(maxlen=32)

    def evaluate(
        self,
        transcript: TranscriptLike,
        *,
        now: float | None = None,
    ) -> TranscriptGateDecision:
        current_time = time.perf_counter() if now is None else now
        raw_text = str(getattr(transcript, "text", "") or "")
        confidence = float(getattr(transcript, "confidence", 0.0) or 0.0)
        kind = _kind_value(getattr(transcript, "kind", ""))
        raw_metadata = getattr(transcript, "metadata", {}) or {}
        transcript_metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        promoted_from_partial = bool(
            transcript_metadata.get("promoted_from_partial")
        )

        normalized = normalize_transcript_text(raw_text)
        words = normalized.split()
        wake_detected = self._wake_detected(normalized)
        attention_active = current_time <= self._attention_active_until

        base_metadata: dict[str, object] = {
            "kind": kind,
            "promoted_from_partial": promoted_from_partial,
            "raw_text": raw_text,
        }

        if kind and kind != "final":
            return self._decision(
                False,
                "non_final_transcript",
                normalized,
                wake_detected,
                attention_active,
                confidence,
                len(words),
                base_metadata,
            )

        if not normalized:
            return self._decision(
                False,
                "empty_transcript",
                normalized,
                wake_detected,
                attention_active,
                confidence,
                0,
                base_metadata,
            )

        if confidence < self._policy.min_final_confidence:
            return self._decision(
                False,
                "low_confidence",
                normalized,
                wake_detected,
                attention_active,
                confidence,
                len(words),
                {
                    **base_metadata,
                    "min_confidence": self._policy.min_final_confidence,
                },
            )

        if self._is_known_silence_hallucination(normalized):
            return self._decision(
                False,
                "known_silence_hallucination",
                normalized,
                wake_detected,
                attention_active,
                confidence,
                len(words),
                base_metadata,
            )

        if self._looks_like_repeated_noise(words):
            return self._decision(
                False,
                "repeated_noise_pattern",
                normalized,
                wake_detected,
                attention_active,
                confidence,
                len(words),
                base_metadata,
            )

        duplicate_count = self._duplicate_count(current_time, normalized)
        if duplicate_count >= self._policy.max_duplicate_count and not wake_detected:
            return self._decision(
                False,
                "duplicate_transcript_noise",
                normalized,
                wake_detected,
                attention_active,
                confidence,
                len(words),
                {**base_metadata, "duplicate_count": duplicate_count},
            )

        if (
            self._policy.require_attention_for_promoted_partials
            and promoted_from_partial
            and not wake_detected
            and not attention_active
        ):
            return self._decision(
                False,
                "promoted_partial_without_attention",
                normalized,
                wake_detected,
                attention_active,
                confidence,
                len(words),
                base_metadata,
            )

        min_words = (
            self._policy.min_words_when_attention_active
            if attention_active
            else self._policy.min_words_without_wake
        )
        if len(words) < min_words and not wake_detected:
            return self._decision(
                False,
                "too_short_without_wake_word",
                normalized,
                wake_detected,
                attention_active,
                confidence,
                len(words),
                {
                    **base_metadata,
                    "min_words_without_wake": min_words,
                },
            )

        if (
            self._policy.require_wake_or_attention
            and not wake_detected
            and not attention_active
        ):
            return self._decision(
                False,
                "requires_wake_or_active_attention",
                normalized,
                wake_detected,
                attention_active,
                confidence,
                len(words),
                base_metadata,
            )

        self._remember(current_time, normalized)

        attention_extended = wake_detected or self._policy.extend_attention_on_accept
        if attention_extended:
            self._attention_active_until = (
                current_time + self._policy.attention_window_seconds
            )

        return self._decision(
            True,
            "accepted",
            normalized,
            wake_detected,
            attention_active,
            confidence,
            len(words),
            {
                **base_metadata,
                "attention_active_until": self._attention_active_until,
                "attention_extended": attention_extended,
            },
        )

    def reset(self) -> None:
        self._attention_active_until = 0.0
        self._recent.clear()

    def _wake_detected(self, normalized: str) -> bool:
        padded = f" {normalized} "
        return any(f" {wake} " in padded for wake in self._policy.wake_words)

    def _is_known_silence_hallucination(self, normalized: str) -> bool:
        if normalized in self._policy.known_silence_hallucinations:
            return True
        if normalized.startswith("thank you") and len(normalized.split()) <= 5:
            return True
        if normalized.startswith("good luck") and len(normalized.split()) <= 5:
            return True
        return False

    def _looks_like_repeated_noise(self, words: list[str]) -> bool:
        if len(words) < 3:
            return False
        if len(set(words)) == 1:
            return True

        compact = " ".join(words)
        repeated_pairs = ("i lift", "i lit", "i lip", "thank you")
        return any(compact.count(pair) >= 2 for pair in repeated_pairs)

    def _duplicate_count(self, now: float, normalized: str) -> int:
        self._prune_recent(now)
        return sum(1 for _, text in self._recent if text == normalized)

    def _remember(self, now: float, normalized: str) -> None:
        self._prune_recent(now)
        self._recent.append((now, normalized))

    def _prune_recent(self, now: float) -> None:
        cutoff = now - self._policy.duplicate_window_seconds
        while self._recent and self._recent[0][0] < cutoff:
            self._recent.popleft()

    def _decision(
        self,
        accepted: bool,
        reason: str,
        normalized: str,
        wake_detected: bool,
        attention_active: bool,
        confidence: float,
        word_count: int,
        metadata: dict[str, object],
    ) -> TranscriptGateDecision:
        return TranscriptGateDecision(
            accepted=accepted,
            reason=reason,
            normalized_text=normalized,
            wake_detected=wake_detected,
            attention_active=attention_active,
            confidence=confidence,
            word_count=word_count,
            metadata=metadata,
        )


def normalize_transcript_text(text: str) -> str:
    cleaned = text.strip().lower()
    cleaned = cleaned.replace("â€™", "'")
    cleaned = cleaned.replace("y'all", "yall")
    cleaned = cleaned.replace("we'll", "we ll")
    cleaned = cleaned.replace("i'm", "i m")
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _kind_value(kind: object) -> str:
    value = getattr(kind, "value", kind)
    return str(value or "").strip().lower()
