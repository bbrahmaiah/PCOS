from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.conversation import (
    AdaptiveEndpointingEngine,
    ConversationMode,
    ConversationState,
    EndpointAction,
    EndpointingDecision,
    EndpointingInput,
    SpeechChunkKind,
    StreamingConversationCoordinator,
    StreamingConversationCoordinatorConfig,
    StreamingConversationEvent,
    StreamingCoordinatorAction,
    StreamingEventKind,
    StreamingLifecycle,
    StreamingSpeechChunk,
    TurnDetectionInput,
)


def event(
    kind: StreamingEventKind,
    *,
    text: str = "",
    sequence: int = 0,
    turn_id: str = "turn-1",
) -> StreamingConversationEvent:
    return StreamingConversationEvent(
        turn_id=turn_id,
        kind=kind,
        text=text,
        sequence=sequence,
    )


def endpoint_decision(
    *,
    transcript: str = "run tests",
    state: ConversationState = ConversationState.LISTENING,
    silence_ms: int = 500,
    mode: ConversationMode = ConversationMode.COMMAND,
) -> EndpointingDecision:
    engine = AdaptiveEndpointingEngine()

    return engine.evaluate(
        EndpointingInput(
            signal=TurnDetectionInput(
                turn_id="turn-1",
                transcript=transcript,
                silence_ms=silence_ms,
                conversation_mode=mode,
                transcript_stability=0.9,
            ),
            conversation_state=state,
        )
    )


def test_streaming_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        StreamingConversationCoordinatorConfig(name=" ").validate()

    with pytest.raises(ValueError):
        StreamingConversationCoordinatorConfig(
            min_speech_chunk_chars=0
        ).validate()

    with pytest.raises(ValueError):
        StreamingConversationCoordinatorConfig(
            min_speech_chunk_chars=50,
            max_speech_chunk_chars=40,
        ).validate()

    with pytest.raises(ValueError):
        StreamingConversationCoordinatorConfig(sentence_endings=()).validate()


def test_streaming_event_requires_turn_id() -> None:
    with pytest.raises(ValidationError):
        StreamingConversationEvent(
            turn_id=" ",
            kind=StreamingEventKind.STT_PARTIAL,
        )


def test_speech_chunk_requires_text() -> None:
    with pytest.raises(ValidationError):
        StreamingSpeechChunk(
            turn_id="turn-1",
            text=" ",
            sequence=0,
            kind=SpeechChunkKind.PARTIAL,
        )


def test_stt_partial_keeps_listening() -> None:
    coordinator = StreamingConversationCoordinator()

    output = coordinator.accept_event(
        event(StreamingEventKind.STT_PARTIAL, text="Jarvis explain")
    )

    assert output.lifecycle == StreamingLifecycle.LISTENING
    assert output.actions == (StreamingCoordinatorAction.KEEP_LISTENING,)
    assert output.should_keep_listening is True


def test_endpointing_start_cognition_starts_streaming_cognition() -> None:
    coordinator = StreamingConversationCoordinator()
    decision = endpoint_decision()

    output = coordinator.accept_endpointing_decision(decision)
    snapshot = coordinator.snapshot()

    assert decision.action == EndpointAction.START_COGNITION
    assert output.lifecycle == StreamingLifecycle.THINKING
    assert output.should_start_cognition is True
    assert StreamingCoordinatorAction.START_COGNITION in output.actions
    assert snapshot.cognition_start_count == 1


def test_prepare_response_starts_cognition_without_final_turn_event() -> None:
    coordinator = StreamingConversationCoordinator()
    decision = endpoint_decision(
        transcript="Can you explain the memory gateway?",
        silence_ms=650,
        mode=ConversationMode.QUESTION,
    )
    prepared = decision.model_copy(
        update={
            "action": EndpointAction.PREPARE_RESPONSE,
            "should_start_cognition": False,
            "reason": "stable maybe-complete turn can prepare response",
        }
    )

    output = coordinator.accept_endpointing_decision(prepared)

    assert output.lifecycle == StreamingLifecycle.THINKING
    assert output.should_start_cognition is True
    assert output.actions == (StreamingCoordinatorAction.START_COGNITION,)


def test_cognition_tokens_buffer_until_boundary() -> None:
    coordinator = StreamingConversationCoordinator(
        config=StreamingConversationCoordinatorConfig(
            min_speech_chunk_chars=40,
            max_speech_chunk_chars=200,
        )
    )
    coordinator.accept_event(event(StreamingEventKind.USER_TURN_FINALIZED))

    output = coordinator.add_cognition_token(
        turn_id="turn-1",
        token="Memory is active",
    )

    assert output.lifecycle == StreamingLifecycle.RESPONSE_STREAMING
    assert output.actions == (StreamingCoordinatorAction.BUFFER_TOKEN,)
    assert output.speech_chunk is None


def test_cognition_token_emits_speech_chunk_on_sentence_boundary() -> None:
    coordinator = StreamingConversationCoordinator(
        config=StreamingConversationCoordinatorConfig(
            min_speech_chunk_chars=10,
            max_speech_chunk_chars=200,
        )
    )
    coordinator.accept_event(event(StreamingEventKind.USER_TURN_FINALIZED))

    output = coordinator.add_cognition_token(
        turn_id="turn-1",
        token="Memory is active now.",
    )
    snapshot = coordinator.snapshot()

    assert output.lifecycle == StreamingLifecycle.RESPONSE_STREAMING
    assert StreamingCoordinatorAction.EMIT_SPEECH_CHUNK in output.actions
    assert StreamingCoordinatorAction.START_TTS in output.actions
    assert output.speech_chunk is not None
    assert output.speech_chunk.kind == SpeechChunkKind.PARTIAL
    assert output.should_start_tts is True
    assert snapshot.speech_chunk_count == 1


def test_cognition_token_emits_when_max_chars_reached() -> None:
    coordinator = StreamingConversationCoordinator(
        config=StreamingConversationCoordinatorConfig(
            min_speech_chunk_chars=10,
            max_speech_chunk_chars=20,
            emit_on_sentence_boundary=True,
        )
    )
    coordinator.accept_event(event(StreamingEventKind.USER_TURN_FINALIZED))

    output = coordinator.add_cognition_token(
        turn_id="turn-1",
        token="This response is long enough to emit early",
    )

    assert output.speech_chunk is not None
    assert output.speech_chunk.kind == SpeechChunkKind.PARTIAL


def test_cognition_completed_emits_final_buffer() -> None:
    coordinator = StreamingConversationCoordinator(
        config=StreamingConversationCoordinatorConfig(
            min_speech_chunk_chars=100,
            max_speech_chunk_chars=200,
        )
    )
    coordinator.accept_event(event(StreamingEventKind.USER_TURN_FINALIZED))
    coordinator.add_cognition_token(turn_id="turn-1", token="Final short answer")

    output = coordinator.complete_cognition(turn_id="turn-1")

    assert output.lifecycle == StreamingLifecycle.COMPLETED
    assert StreamingCoordinatorAction.COMPLETE_RESPONSE in output.actions
    assert output.speech_chunk is not None
    assert output.speech_chunk.kind == SpeechChunkKind.FINAL


def test_cognition_completed_without_buffer_completes_cleanly() -> None:
    coordinator = StreamingConversationCoordinator()
    coordinator.accept_event(event(StreamingEventKind.USER_TURN_FINALIZED))

    output = coordinator.complete_cognition(turn_id="turn-1")

    assert output.lifecycle == StreamingLifecycle.COMPLETED
    assert output.speech_chunk is None
    assert output.actions == (StreamingCoordinatorAction.COMPLETE_RESPONSE,)


def test_speech_chunk_started_sets_speaking() -> None:
    coordinator = StreamingConversationCoordinator()

    output = coordinator.accept_event(
        event(StreamingEventKind.SPEECH_CHUNK_STARTED)
    )

    assert output.lifecycle == StreamingLifecycle.SPEAKING
    assert coordinator.lifecycle == StreamingLifecycle.SPEAKING


def test_interrupt_cancels_streams_and_clears_buffer() -> None:
    coordinator = StreamingConversationCoordinator()
    coordinator.accept_event(event(StreamingEventKind.USER_TURN_FINALIZED))
    coordinator.add_cognition_token(turn_id="turn-1", token="Partial response")

    output = coordinator.accept_event(
        event(StreamingEventKind.INTERRUPT_REQUESTED, text="stop")
    )
    snapshot = coordinator.snapshot()

    assert output.lifecycle == StreamingLifecycle.INTERRUPTED
    assert output.should_cancel_streams is True
    assert output.actions == (StreamingCoordinatorAction.CANCEL_STREAMS,)
    assert snapshot.buffered_chars == 0
    assert snapshot.interrupt_count == 1


def test_cancel_cancels_streams_and_clears_buffer() -> None:
    coordinator = StreamingConversationCoordinator()
    coordinator.accept_event(event(StreamingEventKind.USER_TURN_FINALIZED))
    coordinator.add_cognition_token(turn_id="turn-1", token="Partial response")

    output = coordinator.accept_event(
        event(StreamingEventKind.CANCEL_REQUESTED, text="cancel")
    )
    snapshot = coordinator.snapshot()

    assert output.lifecycle == StreamingLifecycle.CANCELLED
    assert output.should_cancel_streams is True
    assert snapshot.buffered_chars == 0
    assert snapshot.cancel_count == 1


def test_endpointing_interrupt_routes_to_cancel_streams() -> None:
    coordinator = StreamingConversationCoordinator()
    decision = endpoint_decision(
        transcript="stop",
        state=ConversationState.SPEAKING,
    )

    output = coordinator.accept_endpointing_decision(decision)

    assert output.should_cancel_streams is True
    assert output.lifecycle in {
        StreamingLifecycle.INTERRUPTED,
        StreamingLifecycle.CANCELLED,
    }


def test_reset_event_resets_lifecycle() -> None:
    coordinator = StreamingConversationCoordinator()
    coordinator.accept_event(event(StreamingEventKind.USER_TURN_FINALIZED))

    output = coordinator.accept_event(event(StreamingEventKind.RESET))
    snapshot = coordinator.snapshot()

    assert output.lifecycle == StreamingLifecycle.IDLE
    assert StreamingCoordinatorAction.RESET_SESSION in output.actions
    assert snapshot.lifecycle == StreamingLifecycle.IDLE


def test_snapshot_and_reset() -> None:
    coordinator = StreamingConversationCoordinator()

    coordinator.accept_event(event(StreamingEventKind.USER_TURN_FINALIZED))
    snapshot = coordinator.snapshot()

    assert snapshot.event_count == 1
    assert snapshot.cognition_start_count == 1
    assert snapshot.current_turn_id == "turn-1"

    coordinator.reset()
    reset_snapshot = coordinator.snapshot()

    assert reset_snapshot.event_count == 0
    assert reset_snapshot.current_turn_id is None
    assert reset_snapshot.lifecycle == StreamingLifecycle.IDLE


def test_streaming_output_requires_reason() -> None:
    coordinator = StreamingConversationCoordinator()
    output = coordinator.accept_event(event(StreamingEventKind.STT_PARTIAL))
    data = output.model_dump(mode="python")
    data["reason"] = " "

    with pytest.raises(ValidationError):
        type(output).model_validate(data)


def test_streaming_enum_values_are_stable() -> None:
    assert StreamingLifecycle.IDLE.value == "idle"
    assert StreamingLifecycle.RESPONSE_STREAMING.value == "response_streaming"
    assert StreamingEventKind.COGNITION_TOKEN.value == "cognition_token"
    assert StreamingCoordinatorAction.EMIT_SPEECH_CHUNK.value == (
        "emit_speech_chunk"
    )
    assert SpeechChunkKind.PARTIAL.value == "partial"
    assert SpeechChunkKind.FINAL.value == "final"