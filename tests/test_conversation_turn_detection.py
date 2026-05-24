from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.conversation import (
    AdaptiveTurnDetector,
    AdaptiveTurnDetectorConfig,
    ConversationMode,
    InterruptionIntent,
    TranscriptCompleteness,
    TurnDecisionKind,
    TurnDetectionInput,
    TurnEndpointReason,
    TurnUrgency,
)


def test_turn_detector_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        AdaptiveTurnDetectorConfig(name=" ").validate()

    with pytest.raises(ValueError):
        AdaptiveTurnDetectorConfig(command_silence_ms=-1).validate()

    with pytest.raises(ValueError):
        AdaptiveTurnDetectorConfig(
            command_silence_ms=500,
            max_wait_ms=500,
        ).validate()

    with pytest.raises(ValueError):
        AdaptiveTurnDetectorConfig(maybe_complete_ratio=1.2).validate()


def test_turn_detection_input_cleans_transcript() -> None:
    signal = TurnDetectionInput(
        turn_id="turn-1",
        transcript="  Jarvis    explain   this   ",
    )

    assert signal.transcript == "Jarvis explain this"


def test_turn_detection_input_rejects_empty_turn_id() -> None:
    with pytest.raises(ValidationError):
        TurnDetectionInput(turn_id=" ")


def test_empty_transcript_waits() -> None:
    detector = AdaptiveTurnDetector()

    decision = detector.evaluate(
        TurnDetectionInput(
            turn_id="turn-1",
            transcript="",
            silence_ms=2_000,
        )
    )

    assert decision.decision == TurnDecisionKind.WAIT
    assert decision.reason == TurnEndpointReason.EMPTY_TRANSCRIPT
    assert decision.completeness == TranscriptCompleteness.EMPTY
    assert decision.should_start_cognition is False


def test_active_speech_waits_even_with_complete_sentence() -> None:
    detector = AdaptiveTurnDetector()

    decision = detector.evaluate(
        TurnDetectionInput(
            turn_id="turn-1",
            transcript="Jarvis explain the error.",
            is_speech_active=True,
            silence_ms=0,
            vad_confidence=0.9,
        )
    )

    assert decision.decision == TurnDecisionKind.WAIT
    assert decision.reason == TurnEndpointReason.SPEECH_ACTIVE
    assert decision.should_start_cognition is False


def test_short_interrupt_command_interrupts_immediately() -> None:
    detector = AdaptiveTurnDetector()

    decision = detector.evaluate(
        TurnDetectionInput(
            turn_id="turn-1",
            transcript="stop",
            is_assistant_speaking=True,
            is_speech_active=True,
            speech_ms=80,
            vad_confidence=0.9,
        )
    )

    assert decision.decision == TurnDecisionKind.INTERRUPT
    assert decision.interruption_intent == InterruptionIntent.STOP
    assert decision.urgency == TurnUrgency.CRITICAL
    assert decision.should_cancel_response is True


def test_cancel_command_returns_cancel_decision() -> None:
    detector = AdaptiveTurnDetector()

    decision = detector.evaluate(
        TurnDetectionInput(
            turn_id="turn-1",
            transcript="cancel",
        )
    )

    assert decision.decision == TurnDecisionKind.CANCEL
    assert decision.reason == TurnEndpointReason.CANCEL_INTENT
    assert decision.should_cancel_response is True


def test_barge_in_interrupts_when_assistant_is_speaking() -> None:
    detector = AdaptiveTurnDetector()

    decision = detector.evaluate(
        TurnDetectionInput(
            turn_id="turn-1",
            transcript="actually explain first",
            is_assistant_speaking=True,
            is_speech_active=True,
            speech_ms=300,
            vad_confidence=0.8,
        )
    )

    assert decision.decision == TurnDecisionKind.INTERRUPT
    assert decision.reason == TurnEndpointReason.BARGE_IN
    assert decision.interruption_intent == InterruptionIntent.BARGE_IN
    assert decision.should_cancel_response is True


def test_complete_short_command_finalizes_fast() -> None:
    detector = AdaptiveTurnDetector()

    decision = detector.evaluate(
        TurnDetectionInput(
            turn_id="turn-1",
            transcript="run tests",
            silence_ms=500,
            transcript_stability=0.9,
            conversation_mode=ConversationMode.COMMAND,
        )
    )

    assert decision.decision == TurnDecisionKind.FINALIZE
    assert decision.reason == TurnEndpointReason.COMPLETE_COMMAND
    assert decision.should_start_cognition is True


def test_complete_question_finalizes_after_question_threshold() -> None:
    detector = AdaptiveTurnDetector()

    decision = detector.evaluate(
        TurnDetectionInput(
            turn_id="turn-1",
            transcript="How does memory gateway work?",
            silence_ms=900,
            transcript_stability=0.9,
        )
    )

    assert decision.decision == TurnDecisionKind.FINALIZE
    assert decision.reason == TurnEndpointReason.COMPLETE_QUESTION
    assert decision.should_start_cognition is True


def test_question_waits_before_threshold() -> None:
    detector = AdaptiveTurnDetector()

    decision = detector.evaluate(
        TurnDetectionInput(
            turn_id="turn-1",
            transcript="How does memory gateway work?",
            silence_ms=300,
            transcript_stability=0.9,
        )
    )

    assert decision.decision == TurnDecisionKind.WAIT
    assert decision.should_start_cognition is False


def test_incomplete_sentence_does_not_finalize_at_700ms() -> None:
    detector = AdaptiveTurnDetector()

    decision = detector.evaluate(
        TurnDetectionInput(
            turn_id="turn-1",
            transcript="Jarvis I want to",
            silence_ms=700,
            transcript_stability=0.8,
            conversation_mode=ConversationMode.DISCUSSION,
        )
    )

    assert decision.decision == TurnDecisionKind.WAIT
    assert decision.reason == TurnEndpointReason.INCOMPLETE_TRANSCRIPT
    assert decision.completeness == TranscriptCompleteness.INCOMPLETE
    assert decision.should_start_cognition is False


def test_incomplete_sentence_becomes_maybe_complete_after_patience() -> None:
    detector = AdaptiveTurnDetector()

    decision = detector.evaluate(
        TurnDetectionInput(
            turn_id="turn-1",
            transcript="Jarvis I want to",
            silence_ms=1_800,
            transcript_stability=0.8,
            conversation_mode=ConversationMode.DISCUSSION,
        )
    )

    assert decision.decision == TurnDecisionKind.MAYBE_COMPLETE
    assert decision.reason == TurnEndpointReason.INCOMPLETE_TRANSCRIPT
    assert decision.should_start_cognition is False


def test_max_wait_finalizes_incomplete_transcript_safely() -> None:
    detector = AdaptiveTurnDetector()

    decision = detector.evaluate(
        TurnDetectionInput(
            turn_id="turn-1",
            transcript="Jarvis I want to",
            silence_ms=2_700,
            transcript_stability=0.8,
            conversation_mode=ConversationMode.DISCUSSION,
        )
    )

    assert decision.decision == TurnDecisionKind.FINALIZE
    assert decision.reason == TurnEndpointReason.MAX_WAIT_REACHED
    assert decision.should_start_cognition is True


def test_discussion_likely_complete_waits_longer_than_command() -> None:
    detector = AdaptiveTurnDetector()

    decision = detector.evaluate(
        TurnDetectionInput(
            turn_id="turn-1",
            transcript=(
                "I think the memory runtime should retrieve context "
                "before cognition answers"
            ),
            silence_ms=700,
            transcript_stability=0.8,
            conversation_mode=ConversationMode.DISCUSSION,
        )
    )

    assert decision.decision in {
        TurnDecisionKind.WAIT,
        TurnDecisionKind.MAYBE_COMPLETE,
    }
    assert decision.should_start_cognition is False


def test_discussion_finalizes_after_discussion_threshold() -> None:
    detector = AdaptiveTurnDetector()

    decision = detector.evaluate(
        TurnDetectionInput(
            turn_id="turn-1",
            transcript=(
                "I think the memory runtime should retrieve context "
                "before cognition answers."
            ),
            silence_ms=1_500,
            transcript_stability=0.9,
            conversation_mode=ConversationMode.DISCUSSION,
        )
    )

    assert decision.decision == TurnDecisionKind.FINALIZE
    assert decision.reason == TurnEndpointReason.COMPLETE_DISCUSSION
    assert decision.should_start_cognition is True


def test_detector_reports_maybe_complete_before_finalize() -> None:
    detector = AdaptiveTurnDetector()

    decision = detector.evaluate(
        TurnDetectionInput(
            turn_id="turn-1",
            transcript="Can you explain the memory gateway?",
            silence_ms=650,
            transcript_stability=0.8,
        )
    )

    assert decision.decision == TurnDecisionKind.MAYBE_COMPLETE
    assert decision.reason == TurnEndpointReason.MAYBE_COMPLETE
    assert decision.should_start_cognition is False


def test_detector_snapshot_and_reset() -> None:
    detector = AdaptiveTurnDetector()

    detector.evaluate(
        TurnDetectionInput(
            turn_id="turn-1",
            transcript="run tests",
            silence_ms=500,
            conversation_mode=ConversationMode.COMMAND,
        )
    )

    snapshot = detector.snapshot()

    assert snapshot.evaluation_count == 1
    assert snapshot.finalized_count == 1
    assert snapshot.last_turn_id == "turn-1"
    assert snapshot.last_decision == TurnDecisionKind.FINALIZE

    detector.reset()
    reset_snapshot = detector.snapshot()

    assert reset_snapshot.evaluation_count == 0
    assert reset_snapshot.last_turn_id is None


def test_turn_decision_properties() -> None:
    detector = AdaptiveTurnDetector()

    finalized = detector.evaluate(
        TurnDetectionInput(
            turn_id="turn-1",
            transcript="run tests",
            silence_ms=500,
            conversation_mode=ConversationMode.COMMAND,
        )
    )
    interrupt = detector.evaluate(
        TurnDetectionInput(
            turn_id="turn-2",
            transcript="stop",
        )
    )
    waiting = detector.evaluate(
        TurnDetectionInput(
            turn_id="turn-3",
            transcript="Jarvis I want to",
            silence_ms=200,
        )
    )

    assert finalized.finalized is True
    assert interrupt.interrupting is True
    assert waiting.waiting is True


def test_turn_detection_enum_values_are_stable() -> None:
    assert TurnDecisionKind.WAIT.value == "wait"
    assert TurnDecisionKind.MAYBE_COMPLETE.value == "maybe_complete"
    assert TurnDecisionKind.FINALIZE.value == "finalize"
    assert TurnDecisionKind.INTERRUPT.value == "interrupt"
    assert TranscriptCompleteness.INCOMPLETE.value == "incomplete"
    assert InterruptionIntent.BARGE_IN.value == "barge_in"