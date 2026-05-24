from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.conversation import (
    ConversationMode,
    ConversationState,
    RealConversationInput,
    RealConversationRuntime,
    RealConversationRuntimeAction,
    RealConversationRuntimeConfig,
    RealConversationRuntimeOutput,
    RealConversationRuntimeStatus,
    StreamingLifecycle,
    TurnInputSource,
)


def user_input(
    *,
    transcript: str,
    silence_ms: int = 0,
    is_speech_active: bool = False,
    is_assistant_speaking: bool = False,
    speech_ms: int = 0,
    vad_confidence: float = 0.0,
    transcript_stability: float = 0.9,
    mode: ConversationMode = ConversationMode.UNKNOWN,
    consecutive_maybe_complete_count: int = 0,
) -> RealConversationInput:
    return RealConversationInput(
        turn_id="turn-1",
        transcript=transcript,
        silence_ms=silence_ms,
        is_speech_active=is_speech_active,
        is_assistant_speaking=is_assistant_speaking,
        speech_ms=speech_ms,
        vad_confidence=vad_confidence,
        transcript_stability=transcript_stability,
        conversation_mode=mode,
        consecutive_maybe_complete_count=consecutive_maybe_complete_count,
    )


def test_runtime_config_rejects_invalid_name() -> None:
    with pytest.raises(ValueError):
        RealConversationRuntimeConfig(name=" ").validate()


def test_runtime_input_requires_turn_id() -> None:
    with pytest.raises(ValidationError):
        RealConversationInput(turn_id=" ")


def test_runtime_input_cleans_transcript() -> None:
    item = RealConversationInput(
        transcript="  Jarvis   run   tests ",
        previous_transcript="  Jarvis   run ",
    )

    assert item.transcript == "Jarvis run tests"
    assert item.previous_transcript == "Jarvis run"


def test_runtime_starts_ready() -> None:
    runtime = RealConversationRuntime()

    snapshot = runtime.snapshot()

    assert snapshot.status == RealConversationRuntimeStatus.READY
    assert snapshot.current_state == ConversationState.IDLE
    assert snapshot.streaming_lifecycle == StreamingLifecycle.IDLE


def test_runtime_keeps_listening_for_active_speech() -> None:
    runtime = RealConversationRuntime()

    output = runtime.accept_input(
        user_input(
            transcript="Jarvis explain this",
            is_speech_active=True,
            vad_confidence=0.9,
        )
    )

    assert output.should_keep_listening is True
    assert output.should_start_cognition is False
    assert output.status in {
        RealConversationRuntimeStatus.LISTENING,
        RealConversationRuntimeStatus.USER_THINKING,
    }


def test_runtime_starts_cognition_for_complete_command() -> None:
    runtime = RealConversationRuntime()

    output = runtime.accept_input(
        user_input(
            transcript="run tests",
            silence_ms=500,
            mode=ConversationMode.COMMAND,
        )
    )
    snapshot = runtime.snapshot()

    assert output.should_start_cognition is True
    assert RealConversationRuntimeAction.START_COGNITION in output.actions
    assert output.status == RealConversationRuntimeStatus.COGNITION_READY
    assert snapshot.current_state == ConversationState.THINKING
    assert snapshot.cognition_start_count == 1


def test_runtime_updates_session_and_attention_on_finalized_turn() -> None:
    runtime = RealConversationRuntime()

    output = runtime.accept_input(
        user_input(
            transcript="How does memory gateway work?",
            silence_ms=900,
            mode=ConversationMode.QUESTION,
        )
    )

    assert output.session_snapshot is not None
    assert output.attention_decision is not None
    assert output.cognition_context_block is not None
    assert "Conversation session continuity:" in output.cognition_context_block
    assert "Attention runtime:" in output.cognition_context_block


def test_runtime_waits_for_incomplete_sentence() -> None:
    runtime = RealConversationRuntime()

    output = runtime.accept_input(
        user_input(
            transcript="Jarvis I want to",
            silence_ms=700,
            mode=ConversationMode.DISCUSSION,
        )
    )

    assert output.should_start_cognition is False
    assert output.should_keep_listening is True
    assert RealConversationRuntimeAction.KEEP_LISTENING in output.actions


def test_runtime_prepares_response_for_repeated_maybe_complete() -> None:
    runtime = RealConversationRuntime()

    output = runtime.accept_input(
        user_input(
            transcript="Can you explain the memory gateway?",
            silence_ms=650,
            mode=ConversationMode.QUESTION,
            consecutive_maybe_complete_count=2,
        )
    )

    assert output.status in {
        RealConversationRuntimeStatus.USER_THINKING,
        RealConversationRuntimeStatus.COGNITION_READY,
    }
    assert output.streaming_output is not None


def test_runtime_interrupts_active_response() -> None:
    runtime = RealConversationRuntime()

    output = runtime.accept_input(
        user_input(
            transcript="stop",
            is_assistant_speaking=True,
            is_speech_active=True,
            speech_ms=300,
            vad_confidence=0.9,
            mode=ConversationMode.COMMAND,
        )
    )

    assert output.should_cancel_active_work is True
    assert output.interrupt_decision is not None
    assert RealConversationRuntimeAction.CANCEL_ACTIVE_WORK in output.actions
    assert output.status == RealConversationRuntimeStatus.INTERRUPTED
    assert runtime.snapshot().interrupt_count == 1


def test_runtime_streams_cognition_token_and_starts_tts() -> None:
    runtime = RealConversationRuntime()

    runtime.accept_input(
        user_input(
            transcript="run tests",
            silence_ms=500,
            mode=ConversationMode.COMMAND,
        )
    )
    output = runtime.add_cognition_token(
        turn_id="turn-1",
        token="Memory is active and ready.",
    )

    assert output.streaming_output is not None
    assert output.should_start_tts is True
    assert RealConversationRuntimeAction.START_TTS in output.actions


def test_runtime_completes_cognition_with_final_chunk() -> None:
    runtime = RealConversationRuntime()

    runtime.accept_input(
        user_input(
            transcript="run tests",
            silence_ms=500,
            mode=ConversationMode.COMMAND,
        )
    )
    runtime.add_cognition_token(
        turn_id="turn-1",
        token="Short answer",
    )
    output = runtime.complete_cognition(turn_id="turn-1")

    assert output.streaming_output is not None
    assert output.status == RealConversationRuntimeStatus.COMPLETED
    assert RealConversationRuntimeAction.COMPLETE_RESPONSE in output.actions


def test_runtime_adds_assistant_response_to_session() -> None:
    runtime = RealConversationRuntime()

    runtime.accept_input(
        user_input(
            transcript="How does endpointing work?",
            silence_ms=900,
            mode=ConversationMode.QUESTION,
        )
    )
    output = runtime.add_assistant_response(
        "Endpointing decides when the user has finished speaking.",
        expects_follow_up=True,
    )

    assert output.session_snapshot is not None
    assert output.attention_decision is not None
    assert RealConversationRuntimeAction.UPDATE_SESSION in output.actions
    assert RealConversationRuntimeAction.UPDATE_ATTENTION in output.actions


def test_speech_chunk_started_and_completed_update_state() -> None:
    runtime = RealConversationRuntime()

    runtime.accept_input(
        user_input(
            transcript="run tests",
            silence_ms=500,
            mode=ConversationMode.COMMAND,
        )
    )

    started = runtime.speech_chunk_started(turn_id="turn-1")
    completed = runtime.speech_chunk_completed(turn_id="turn-1")

    assert started.status == RealConversationRuntimeStatus.SPEAKING
    assert RealConversationRuntimeAction.START_TTS in started.actions
    assert completed.should_keep_listening is True


def test_runtime_context_block_combines_session_and_attention() -> None:
    runtime = RealConversationRuntime()

    runtime.accept_input(
        user_input(
            transcript="How does attention runtime work?",
            silence_ms=900,
            mode=ConversationMode.QUESTION,
        )
    )
    block = runtime.context_block()

    assert "Conversation session continuity:" in block
    assert "Attention runtime:" in block


def test_runtime_reset_resets_all_diagnostics() -> None:
    runtime = RealConversationRuntime()

    runtime.accept_input(
        user_input(
            transcript="run tests",
            silence_ms=500,
            mode=ConversationMode.COMMAND,
        )
    )
    output = runtime.reset()
    snapshot = runtime.snapshot()

    assert output.status == RealConversationRuntimeStatus.RESET
    assert snapshot.input_count == 0
    assert snapshot.current_state == ConversationState.IDLE
    assert snapshot.streaming_lifecycle == StreamingLifecycle.IDLE


def test_runtime_output_requires_reason() -> None:
    output = RealConversationRuntimeOutput(
        turn_id="turn-1",
        status=RealConversationRuntimeStatus.READY,
        actions=(RealConversationRuntimeAction.KEEP_LISTENING,),
        reason="valid",
    )
    data = output.model_dump(mode="python")
    data["reason"] = " "

    with pytest.raises(ValidationError):
        RealConversationRuntimeOutput.model_validate(data)


def test_runtime_source_enum_path() -> None:
    item = RealConversationInput(
        transcript="hello",
        source=TurnInputSource.MICROPHONE,
    )

    assert item.source == TurnInputSource.MICROPHONE


def test_runtime_enum_values_are_stable() -> None:
    assert RealConversationRuntimeAction.START_COGNITION.value == (
        "start_cognition"
    )
    assert RealConversationRuntimeAction.CANCEL_ACTIVE_WORK.value == (
        "cancel_active_work"
    )
    assert RealConversationRuntimeStatus.COGNITION_READY.value == (
        "cognition_ready"
    )
    assert RealConversationRuntimeStatus.INTERRUPTED.value == "interrupted"