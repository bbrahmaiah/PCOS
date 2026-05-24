from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from jarvis.conversation.models import (
    ConversationMode,
    InterruptionIntent,
    TranscriptCompleteness,
    TurnDecisionKind,
    TurnDetectionDecision,
    TurnDetectionInput,
    TurnEndpointReason,
    TurnUrgency,
)
from jarvis.runtime.observability.structured_logger import get_logger


@dataclass(frozen=True, slots=True)
class AdaptiveTurnDetectorConfig:
    """
    Configuration for adaptive endpointing.

    These are not fixed endpoint rules. They are timing bands used with
    semantic completion, intent, speech state, and transcript stability.
    """

    name: str = "adaptive_turn_detector"

    interrupt_silence_ms: int = 120
    command_silence_ms: int = 450
    question_silence_ms: int = 850
    discussion_silence_ms: int = 1_400
    dictation_silence_ms: int = 1_800
    incomplete_patience_ms: int = 1_700
    max_wait_ms: int = 2_600

    maybe_complete_ratio: float = 0.7
    min_speech_ms: int = 120
    min_vad_confidence: float = 0.35
    stable_transcript_confidence: float = 0.65

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        values = (
            self.interrupt_silence_ms,
            self.command_silence_ms,
            self.question_silence_ms,
            self.discussion_silence_ms,
            self.dictation_silence_ms,
            self.incomplete_patience_ms,
            self.max_wait_ms,
            self.min_speech_ms,
        )

        if any(value < 0 for value in values):
            raise ValueError("timing values cannot be negative.")

        if self.max_wait_ms <= self.command_silence_ms:
            raise ValueError("max_wait_ms must exceed command_silence_ms.")

        if self.maybe_complete_ratio <= 0.0 or self.maybe_complete_ratio >= 1.0:
            raise ValueError("maybe_complete_ratio must be between 0 and 1.")

        if self.min_vad_confidence < 0.0 or self.min_vad_confidence > 1.0:
            raise ValueError("min_vad_confidence must be between 0 and 1.")

        if (
            self.stable_transcript_confidence < 0.0
            or self.stable_transcript_confidence > 1.0
        ):
            raise ValueError(
                "stable_transcript_confidence must be between 0 and 1."
            )


@dataclass(frozen=True, slots=True)
class AdaptiveTurnDetectorSnapshot:
    """
    Observable diagnostics for turn detector.
    """

    name: str
    evaluation_count: int
    finalized_count: int
    wait_count: int
    maybe_complete_count: int
    interrupt_count: int
    last_turn_id: str | None
    last_decision: TurnDecisionKind | None
    last_reason: TurnEndpointReason | None
    last_error: str | None


class AdaptiveTurnDetector:
    """
    Adaptive turn detector for real-time conversation.

    Responsibilities:
    - detect whether user is still speaking
    - avoid false execution from thinking pauses
    - finalize fast commands quickly
    - wait longer for incomplete discussion
    - detect interruption / cancel intent
    - produce observable, typed decisions

    Non-responsibilities:
    - no STT
    - no VAD implementation
    - no TTS control
    - no LLM calls
    - no direct action execution
    """

    _INTERRUPT_STOP = {
        "stop",
        "jarvis stop",
        "bob stop",
        "cancel",
        "cancel it",
        "stop it",
        "pause",
        "pause it",
        "wait",
        "wait jarvis",
        "hold on",
        "never mind",
        "nevermind",
    }

    _CANCEL_PHRASES = {
        "cancel",
        "cancel it",
        "never mind",
        "nevermind",
        "forget it",
    }

    _SHORT_COMMANDS = {
        "stop",
        "pause",
        "cancel",
        "continue",
        "yes",
        "no",
        "okay",
        "ok",
        "run tests",
        "open terminal",
        "explain this",
        "repeat",
    }

    _INCOMPLETE_ENDINGS = {
        "and",
        "but",
        "because",
        "so",
        "then",
        "like",
        "also",
        "or",
        "if",
        "when",
        "while",
        "before",
        "after",
        "for example",
        "i mean",
        "i want to",
        "can you",
        "could you",
        "what if",
        "let me",
    }

    def __init__(
        self,
        *,
        config: AdaptiveTurnDetectorConfig | None = None,
    ) -> None:
        self._config = config or AdaptiveTurnDetectorConfig()
        self._config.validate()

        self._lock = RLock()
        self._logger = get_logger("conversation.turn_detection")

        self._evaluation_count = 0
        self._finalized_count = 0
        self._wait_count = 0
        self._maybe_complete_count = 0
        self._interrupt_count = 0
        self._last_turn_id: str | None = None
        self._last_decision: TurnDecisionKind | None = None
        self._last_reason: TurnEndpointReason | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def evaluate(
        self,
        signal: TurnDetectionInput,
    ) -> TurnDetectionDecision:
        """
        Evaluate one conversational turn snapshot.
        """

        with self._lock:
            self._evaluation_count += 1
            self._last_turn_id = signal.turn_id
            self._last_error = None

        try:
            decision = self._evaluate(signal)
            self._record_decision(decision)

            self._logger.info(
                "conversation_turn_detection_evaluated",
                detector=self.name,
                turn_id=signal.turn_id,
                decision=decision.decision.value,
                reason=decision.reason.value,
                completeness=decision.completeness.value,
                silence_ms=signal.silence_ms,
                transcript=signal.transcript,
            )

            return decision

        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"

            raise

    def snapshot(self) -> AdaptiveTurnDetectorSnapshot:
        """
        Return turn detector diagnostics.
        """

        with self._lock:
            return AdaptiveTurnDetectorSnapshot(
                name=self.name,
                evaluation_count=self._evaluation_count,
                finalized_count=self._finalized_count,
                wait_count=self._wait_count,
                maybe_complete_count=self._maybe_complete_count,
                interrupt_count=self._interrupt_count,
                last_turn_id=self._last_turn_id,
                last_decision=self._last_decision,
                last_reason=self._last_reason,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Reset detector diagnostics.
        """

        with self._lock:
            self._evaluation_count = 0
            self._finalized_count = 0
            self._wait_count = 0
            self._maybe_complete_count = 0
            self._interrupt_count = 0
            self._last_turn_id = None
            self._last_decision = None
            self._last_reason = None
            self._last_error = None

        self._logger.info("adaptive_turn_detector_reset", detector=self.name)

    def _evaluate(
        self,
        signal: TurnDetectionInput,
    ) -> TurnDetectionDecision:
        transcript = signal.transcript.strip()
        normalized = transcript.casefold()
        intent = self._interruption_intent(normalized)
        mode = self._conversation_mode(signal, normalized)
        completeness = self._completeness(normalized, mode)
        urgency = self._urgency(
            intent=intent,
            mode=mode,
            transcript=normalized,
        )

        if not transcript:
            return self._wait_decision(
                signal=signal,
                transcript=transcript,
                completeness=TranscriptCompleteness.EMPTY,
                reason=TurnEndpointReason.EMPTY_TRANSCRIPT,
                urgency=urgency,
            )

        if self._is_barge_in(signal):
            return TurnDetectionDecision(
                turn_id=signal.turn_id,
                decision=TurnDecisionKind.INTERRUPT,
                reason=TurnEndpointReason.BARGE_IN,
                transcript=transcript,
                completeness=completeness,
                interruption_intent=InterruptionIntent.BARGE_IN,
                urgency=TurnUrgency.HIGH,
                confidence=0.9,
                should_start_cognition=False,
                should_cancel_response=True,
                should_keep_listening=True,
                endpoint_delay_ms=0,
                metadata={
                    "detector": self.name,
                    "conversation_mode": mode.value,
                },
            )

        if intent != InterruptionIntent.NONE:
            return self._interrupt_decision(
                signal=signal,
                transcript=transcript,
                intent=intent,
                urgency=urgency,
            )

        if signal.is_speech_active:
            return self._wait_decision(
                signal=signal,
                transcript=transcript,
                completeness=completeness,
                reason=TurnEndpointReason.SPEECH_ACTIVE,
                urgency=urgency,
            )

        if signal.silence_ms >= self._config.max_wait_ms:
            return self._finalize_decision(
                signal=signal,
                transcript=transcript,
                completeness=completeness,
                reason=TurnEndpointReason.MAX_WAIT_REACHED,
                urgency=urgency,
                confidence=0.78,
            )

        if completeness == TranscriptCompleteness.INCOMPLETE:
            return self._handle_incomplete(
                signal=signal,
                transcript=transcript,
                urgency=urgency,
            )

        threshold = self._endpoint_threshold_ms(
            mode=mode,
            completeness=completeness,
            normalized=normalized,
        )

        if signal.silence_ms >= threshold:
            return self._finalize_decision(
                signal=signal,
                transcript=transcript,
                completeness=completeness,
                reason=self._finalize_reason(mode),
                urgency=urgency,
                confidence=self._finalize_confidence(signal, completeness),
            )

        maybe_threshold = int(threshold * self._config.maybe_complete_ratio)

        if signal.silence_ms >= maybe_threshold:
            return TurnDetectionDecision(
                turn_id=signal.turn_id,
                decision=TurnDecisionKind.MAYBE_COMPLETE,
                reason=TurnEndpointReason.MAYBE_COMPLETE,
                transcript=transcript,
                completeness=completeness,
                interruption_intent=InterruptionIntent.NONE,
                urgency=urgency,
                confidence=0.62,
                should_start_cognition=False,
                should_cancel_response=False,
                should_keep_listening=True,
                endpoint_delay_ms=threshold - signal.silence_ms,
                metadata={
                    "detector": self.name,
                    "conversation_mode": mode.value,
                    "threshold_ms": threshold,
                },
            )

        return self._wait_decision(
            signal=signal,
            transcript=transcript,
            completeness=completeness,
            reason=TurnEndpointReason.LOW_SILENCE,
            urgency=urgency,
            endpoint_delay_ms=threshold - signal.silence_ms,
        )

    def _handle_incomplete(
        self,
        *,
        signal: TurnDetectionInput,
        transcript: str,
        urgency: TurnUrgency,
    ) -> TurnDetectionDecision:
        if signal.silence_ms >= self._config.incomplete_patience_ms:
            return TurnDetectionDecision(
                turn_id=signal.turn_id,
                decision=TurnDecisionKind.MAYBE_COMPLETE,
                reason=TurnEndpointReason.INCOMPLETE_TRANSCRIPT,
                transcript=transcript,
                completeness=TranscriptCompleteness.INCOMPLETE,
                interruption_intent=InterruptionIntent.NONE,
                urgency=urgency,
                confidence=0.45,
                should_start_cognition=False,
                should_cancel_response=False,
                should_keep_listening=True,
                endpoint_delay_ms=(
                    self._config.max_wait_ms - signal.silence_ms
                ),
                metadata={
                    "detector": self.name,
                    "safety": "waiting_due_to_incomplete_transcript",
                },
            )

        return self._wait_decision(
            signal=signal,
            transcript=transcript,
            completeness=TranscriptCompleteness.INCOMPLETE,
            reason=TurnEndpointReason.INCOMPLETE_TRANSCRIPT,
            urgency=urgency,
            endpoint_delay_ms=(
                self._config.incomplete_patience_ms - signal.silence_ms
            ),
        )

    def _interrupt_decision(
        self,
        *,
        signal: TurnDetectionInput,
        transcript: str,
        intent: InterruptionIntent,
        urgency: TurnUrgency,
    ) -> TurnDetectionDecision:
        decision = (
            TurnDecisionKind.CANCEL
            if intent == InterruptionIntent.CANCEL
            else TurnDecisionKind.INTERRUPT
        )
        reason = (
            TurnEndpointReason.CANCEL_INTENT
            if intent == InterruptionIntent.CANCEL
            else TurnEndpointReason.INTERRUPTION_INTENT
        )

        return TurnDetectionDecision(
            turn_id=signal.turn_id,
            decision=decision,
            reason=reason,
            transcript=transcript,
            completeness=TranscriptCompleteness.COMPLETE,
            interruption_intent=intent,
            urgency=urgency,
            confidence=0.96,
            should_start_cognition=False,
            should_cancel_response=True,
            should_keep_listening=True,
            endpoint_delay_ms=0,
            metadata={
                "detector": self.name,
                "fast_path": "interruption",
            },
        )

    def _finalize_decision(
        self,
        *,
        signal: TurnDetectionInput,
        transcript: str,
        completeness: TranscriptCompleteness,
        reason: TurnEndpointReason,
        urgency: TurnUrgency,
        confidence: float,
    ) -> TurnDetectionDecision:
        return TurnDetectionDecision(
            turn_id=signal.turn_id,
            decision=TurnDecisionKind.FINALIZE,
            reason=reason,
            transcript=transcript,
            completeness=completeness,
            interruption_intent=InterruptionIntent.NONE,
            urgency=urgency,
            confidence=confidence,
            should_start_cognition=True,
            should_cancel_response=False,
            should_keep_listening=True,
            endpoint_delay_ms=0,
            metadata={
                "detector": self.name,
            },
        )

    def _wait_decision(
        self,
        *,
        signal: TurnDetectionInput,
        transcript: str,
        completeness: TranscriptCompleteness,
        reason: TurnEndpointReason,
        urgency: TurnUrgency,
        endpoint_delay_ms: int = 0,
    ) -> TurnDetectionDecision:
        return TurnDetectionDecision(
            turn_id=signal.turn_id,
            decision=TurnDecisionKind.WAIT,
            reason=reason,
            transcript=transcript,
            completeness=completeness,
            interruption_intent=InterruptionIntent.NONE,
            urgency=urgency,
            confidence=0.75,
            should_start_cognition=False,
            should_cancel_response=False,
            should_keep_listening=True,
            endpoint_delay_ms=max(0, endpoint_delay_ms),
            metadata={
                "detector": self.name,
            },
        )

    def _record_decision(self, decision: TurnDetectionDecision) -> None:
        with self._lock:
            self._last_decision = decision.decision
            self._last_reason = decision.reason

            if decision.decision == TurnDecisionKind.FINALIZE:
                self._finalized_count += 1

            elif decision.decision == TurnDecisionKind.MAYBE_COMPLETE:
                self._maybe_complete_count += 1

            elif decision.interrupting:
                self._interrupt_count += 1

            else:
                self._wait_count += 1

    def _is_barge_in(self, signal: TurnDetectionInput) -> bool:
        if not signal.is_assistant_speaking:
            return False

        return (
            signal.is_speech_active
            and signal.speech_ms >= self._config.min_speech_ms
            and signal.vad_confidence >= self._config.min_vad_confidence
        )

    def _conversation_mode(
        self,
        signal: TurnDetectionInput,
        normalized: str,
    ) -> ConversationMode:
        if signal.conversation_mode != ConversationMode.UNKNOWN:
            return signal.conversation_mode

        if not normalized:
            return ConversationMode.UNKNOWN

        if normalized in self._SHORT_COMMANDS:
            return ConversationMode.COMMAND

        if normalized.endswith("?"):
            return ConversationMode.QUESTION

        question_starts = (
            "what ",
            "why ",
            "how ",
            "when ",
            "where ",
            "can you ",
            "could you ",
            "should i ",
            "do you ",
        )

        if normalized.startswith(question_starts):
            return ConversationMode.QUESTION

        discussion_starts = (
            "i think",
            "i want",
            "i am thinking",
            "let me explain",
            "for example",
            "the thing is",
        )

        if normalized.startswith(discussion_starts):
            return ConversationMode.DISCUSSION

        return ConversationMode.DISCUSSION

    def _completeness(
        self,
        normalized: str,
        mode: ConversationMode,
    ) -> TranscriptCompleteness:
        if not normalized:
            return TranscriptCompleteness.EMPTY

        if self._ends_in_incomplete_phrase(normalized):
            return TranscriptCompleteness.INCOMPLETE

        if normalized.endswith((".", "?", "!")):
            return TranscriptCompleteness.COMPLETE

        if normalized in self._SHORT_COMMANDS:
            return TranscriptCompleteness.COMPLETE

        word_count = len(normalized.split())

        if mode == ConversationMode.COMMAND and word_count <= 6:
            return TranscriptCompleteness.LIKELY_COMPLETE

        if mode == ConversationMode.QUESTION and word_count >= 4:
            return TranscriptCompleteness.LIKELY_COMPLETE

        if mode == ConversationMode.DISCUSSION and word_count >= 8:
            return TranscriptCompleteness.LIKELY_COMPLETE

        return TranscriptCompleteness.INCOMPLETE

    def _ends_in_incomplete_phrase(self, normalized: str) -> bool:
        words = normalized.split()

        if not words:
            return False

        last_word = words[-1]

        if last_word in self._INCOMPLETE_ENDINGS:
            return True

        return any(
            normalized.endswith(ending)
            for ending in self._INCOMPLETE_ENDINGS
            if " " in ending
        )

    def _interruption_intent(self, normalized: str) -> InterruptionIntent:
        if not normalized:
            return InterruptionIntent.NONE

        if normalized in self._CANCEL_PHRASES:
            return InterruptionIntent.CANCEL

        if normalized in self._INTERRUPT_STOP:
            if "pause" in normalized:
                return InterruptionIntent.PAUSE

            if "wait" in normalized or "hold on" in normalized:
                return InterruptionIntent.WAIT

            return InterruptionIntent.STOP

        if normalized.startswith(("no wait", "wait no", "actually")):
            return InterruptionIntent.CORRECTION

        return InterruptionIntent.NONE

    def _urgency(
        self,
        *,
        intent: InterruptionIntent,
        mode: ConversationMode,
        transcript: str,
    ) -> TurnUrgency:
        if intent in {
            InterruptionIntent.STOP,
            InterruptionIntent.CANCEL,
            InterruptionIntent.PAUSE,
        }:
            return TurnUrgency.CRITICAL

        if intent != InterruptionIntent.NONE:
            return TurnUrgency.HIGH

        if mode == ConversationMode.COMMAND:
            return TurnUrgency.HIGH

        if transcript in self._SHORT_COMMANDS:
            return TurnUrgency.HIGH

        if mode == ConversationMode.DISCUSSION:
            return TurnUrgency.NORMAL

        return TurnUrgency.NORMAL

    def _endpoint_threshold_ms(
        self,
        *,
        mode: ConversationMode,
        completeness: TranscriptCompleteness,
        normalized: str,
    ) -> int:
        if normalized in self._SHORT_COMMANDS:
            return self._config.command_silence_ms

        if mode == ConversationMode.COMMAND:
            return self._config.command_silence_ms

        if mode == ConversationMode.QUESTION:
            return self._config.question_silence_ms

        if mode == ConversationMode.DICTATION:
            return self._config.dictation_silence_ms

        if completeness == TranscriptCompleteness.COMPLETE:
            return self._config.discussion_silence_ms

        return self._config.incomplete_patience_ms

    @staticmethod
    def _finalize_reason(mode: ConversationMode) -> TurnEndpointReason:
        if mode == ConversationMode.COMMAND:
            return TurnEndpointReason.COMPLETE_COMMAND

        if mode == ConversationMode.QUESTION:
            return TurnEndpointReason.COMPLETE_QUESTION

        return TurnEndpointReason.COMPLETE_DISCUSSION

    def _finalize_confidence(
        self,
        signal: TurnDetectionInput,
        completeness: TranscriptCompleteness,
    ) -> float:
        base = 0.82

        if completeness == TranscriptCompleteness.COMPLETE:
            base += 0.08

        if signal.transcript_stability >= self._config.stable_transcript_confidence:
            base += 0.05

        if signal.vad_confidence >= self._config.min_vad_confidence:
            base += 0.03

        return min(1.0, base)