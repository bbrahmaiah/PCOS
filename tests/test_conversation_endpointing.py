from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.conversation import (
    AdaptiveEndpointingEngine,
    AdaptiveEndpointingEngineConfig,
    ConversationMode,
    ConversationState,
    EndpointAction,
    EndpointConfidenceBand,
    EndpointingDecision,
    EndpointingInput,
    EndpointPacing,
    TurnDetectionInput,
)


def request(
    *,
    transcript: str,
    state: ConversationState = ConversationState.LISTENING,
    silence_ms: int = 0,
    is_speech_active: bool = False,
    is_assistant_speaking: bool = False,
    speech_ms: int = 0,
    vad_confidence: float = 0.0,
    transcript_stability: float = 0.0,
    conversation_mode: ConversationMode = ConversationMode.UNKNOWN,
    consecutive_maybe_complete_count: int = 0,
) -> EndpointingInput:
    return EndpointingInput(
        signal=TurnDetectionInput(
            turn_id="turn-1",
            transcript=transcript,
            is_speech_active=is_speech_active,
            is_assistant_speaking=is_assistant_speaking,
            silence_ms=silence_ms,
            speech_ms=speech_ms,
            vad_confidence=vad_confidence,
            transcript_stability=transcript_stability,
            conversation_mode=conversation_mode,
        ),
        conversation_state=state,
        consecutive_maybe_complete_count=consecutive_maybe_complete_count,
    )


def test_endpointing_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        AdaptiveEndpointingEngineConfig(name=" ").validate()

    with pytest.raises(ValueError):
        AdaptiveEndpointingEngineConfig(
            maybe_complete_to_prepare_count=0
        ).validate()

    with pytest.raises(ValueError):
        AdaptiveEndpointingEngineConfig(thinking_pause_extra_ms=-1).validate()

    with pytest.raises(ValueError):
        AdaptiveEndpointingEngineConfig(
            speaking_barge_in_min_confidence=1.5
        ).validate()


def test_endpointing_input_cleans_previous_transcript() -> None:
    item = EndpointingInput(
        signal=TurnDetectionInput(transcript="hello"),
        conversation_state=ConversationState.LISTENING,
        previous_transcript="  hello    jarvis ",
    )

    assert item.previous_transcript == "hello jarvis"


def test_endpointing_decision_requires_reason() -> None:
    engine = AdaptiveEndpointingEngine()
    decision = engine.evaluate(request(transcript="run tests", silence_ms=500))
    data = decision.model_dump(mode="python")
    data["reason"] = " "

    with pytest.raises(ValidationError):
        EndpointingDecision.model_validate(data)


def test_empty_transcript_keeps_listening() -> None:
    engine = AdaptiveEndpointingEngine()

    decision = engine.evaluate(request(transcript="", silence_ms=2_000))

    assert decision.action == EndpointAction.KEEP_LISTENING
    assert decision.waiting is True
    assert decision.should_start_cognition is False


def test_active_speech_keeps_listening() -> None:
    engine = AdaptiveEndpointingEngine()

    decision = engine.evaluate(
        request(
            transcript="Jarvis explain this.",
            is_speech_active=True,
            vad_confidence=0.9,
        )
    )

    assert decision.action == EndpointAction.KEEP_LISTENING
    assert decision.should_wait_for_more_audio is True


def test_short_command_starts_cognition_fast() -> None:
    engine = AdaptiveEndpointingEngine()

    decision = engine.evaluate(
        request(
            transcript="run tests",
            silence_ms=500,
            transcript_stability=0.9,
            conversation_mode=ConversationMode.COMMAND,
        )
    )

    assert decision.action == EndpointAction.START_COGNITION
    assert decision.finalized is True
    assert decision.pacing == EndpointPacing.FAST_COMMAND
    assert decision.should_start_cognition is True


def test_question_starts_cognition_after_threshold() -> None:
    engine = AdaptiveEndpointingEngine()

    decision = engine.evaluate(
        request(
            transcript="How does memory gateway work?",
            silence_ms=900,
            transcript_stability=0.9,
        )
    )

    assert decision.action == EndpointAction.START_COGNITION
    assert decision.pacing == EndpointPacing.NORMAL_QUESTION
    assert decision.confidence_band in {
        EndpointConfidenceBand.HIGH,
        EndpointConfidenceBand.CRITICAL,
    }


def test_incomplete_thinking_pause_waits() -> None:
    engine = AdaptiveEndpointingEngine()

    decision = engine.evaluate(
        request(
            transcript="Jarvis I want to",
            state=ConversationState.USER_THINKING,
            silence_ms=700,
            transcript_stability=0.8,
            conversation_mode=ConversationMode.DISCUSSION,
        )
    )

    assert decision.action == EndpointAction.WAIT_FOR_USER
    assert decision.pacing == EndpointPacing.THINKING_PAUSE
    assert decision.should_start_cognition is False
    assert decision.endpoint_delay_ms > 0


def test_incomplete_sentence_does_not_execute_at_700ms() -> None:
    engine = AdaptiveEndpointingEngine()

    decision = engine.evaluate(
        request(
            transcript="Jarvis I want to",
            silence_ms=700,
            conversation_mode=ConversationMode.DISCUSSION,
        )
    )

    assert decision.action == EndpointAction.KEEP_LISTENING
    assert decision.should_start_cognition is False
    assert decision.waiting is True


def test_maybe_complete_waits_for_more_evidence() -> None:
    engine = AdaptiveEndpointingEngine()

    decision = engine.evaluate(
        request(
            transcript="Can you explain the memory gateway?",
            silence_ms=650,
            transcript_stability=0.8,
            conversation_mode=ConversationMode.QUESTION,
        )
    )

    assert decision.action == EndpointAction.WAIT_FOR_USER
    assert decision.reason == "maybe complete; wait for more evidence before responding"


def test_repeated_maybe_complete_can_prepare_response() -> None:
    engine = AdaptiveEndpointingEngine()

    decision = engine.evaluate(
        request(
            transcript="Can you explain the memory gateway?",
            silence_ms=650,
            transcript_stability=0.8,
            conversation_mode=ConversationMode.QUESTION,
            consecutive_maybe_complete_count=2,
        )
    )

    assert decision.action == EndpointAction.PREPARE_RESPONSE
    assert decision.finalized is True
    assert decision.should_start_cognition is False


def test_speaking_barge_in_interrupts_response() -> None:
    engine = AdaptiveEndpointingEngine()

    decision = engine.evaluate(
        request(
            transcript="actually explain first",
            state=ConversationState.SPEAKING,
            is_assistant_speaking=True,
            is_speech_active=True,
            speech_ms=300,
            vad_confidence=0.9,
        )
    )

    assert decision.action == EndpointAction.INTERRUPT_RESPONSE
    assert decision.interrupting is True
    assert decision.should_cancel_response is True
    assert decision.pacing == EndpointPacing.INTERRUPT_IMMEDIATE


def test_cancel_command_cancels_response() -> None:
    engine = AdaptiveEndpointingEngine()

    decision = engine.evaluate(
        request(
            transcript="cancel",
            state=ConversationState.SPEAKING,
        )
    )

    assert decision.action == EndpointAction.CANCEL_RESPONSE
    assert decision.interrupting is True
    assert decision.should_cancel_response is True


def test_finalized_turn_during_speaking_interrupts() -> None:
    engine = AdaptiveEndpointingEngine()

    decision = engine.evaluate(
        request(
            transcript="Explain the memory gateway.",
            state=ConversationState.SPEAKING,
            silence_ms=1_500,
            transcript_stability=0.9,
            conversation_mode=ConversationMode.DISCUSSION,
        )
    )

    assert decision.action == EndpointAction.INTERRUPT_RESPONSE
    assert decision.should_cancel_response is True


def test_finalized_turn_in_thinking_waits() -> None:
    engine = AdaptiveEndpointingEngine()

    decision = engine.evaluate(
        request(
            transcript="Explain the memory gateway.",
            state=ConversationState.THINKING,
            silence_ms=1_500,
            transcript_stability=0.9,
            conversation_mode=ConversationMode.DISCUSSION,
        )
    )

    assert decision.action == EndpointAction.WAIT_FOR_USER
    assert decision.should_start_cognition is False


def test_endpointing_snapshot_and_reset() -> None:
    engine = AdaptiveEndpointingEngine()

    engine.evaluate(
        request(
            transcript="run tests",
            silence_ms=500,
            conversation_mode=ConversationMode.COMMAND,
        )
    )
    snapshot = engine.snapshot()

    assert snapshot.evaluation_count == 1
    assert snapshot.finalized_count == 1
    assert snapshot.last_action == EndpointAction.START_COGNITION

    engine.reset()
    reset_snapshot = engine.snapshot()

    assert reset_snapshot.evaluation_count == 0
    assert reset_snapshot.last_action is None


def test_endpointing_enum_values_are_stable() -> None:
    assert EndpointAction.KEEP_LISTENING.value == "keep_listening"
    assert EndpointAction.WAIT_FOR_USER.value == "wait_for_user"
    assert EndpointAction.START_COGNITION.value == "start_cognition"
    assert EndpointAction.INTERRUPT_RESPONSE.value == "interrupt_response"
    assert EndpointAction.CANCEL_RESPONSE.value == "cancel_response"
    assert EndpointPacing.FAST_COMMAND.value == "fast_command"
    assert EndpointPacing.THINKING_PAUSE.value == "thinking_pause"