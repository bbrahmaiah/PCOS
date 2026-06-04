from __future__ import annotations

from jarvis.live import (
    LiveDialogueRequest,
    LiveDialogueRuntime,
    LiveDialogueRuntimeStatus,
    LiveResponseDraft,
    LiveResponseGenerationRequest,
    LiveResponseGenerationSource,
    LiveResponseIntent,
    LiveResponseSafety,
    LiveSessionConfig,
    LiveSessionMode,
    LiveSessionStateRuntime,
    LiveTranscriptKind,
    make_live_transcript,
)


class ContextualDialogueGenerator:
    def generate(
        self,
        request: LiveResponseGenerationRequest,
    ) -> LiveResponseDraft:
        context = request.context
        pieces = [
            context.live_state.user_label,
            request.intent.value,
            context.user_text,
            " ".join(context.memory_context),
            " ".join(context.goal_context),
            " ".join(context.environment_context),
        ]
        text = " | ".join(piece for piece in pieces if piece.strip())
        return LiveResponseDraft(
            text=text,
            generation_source=LiveResponseGenerationSource.RESPONSE_GENERATOR,
            token_count=len(text.split()),
            metadata={"test_generator": "contextual"},
        )


class InvalidDialogueGenerator:
    def generate(
        self,
        request: LiveResponseGenerationRequest,
    ) -> LiveResponseDraft:
        return LiveResponseDraft(
            text="Invalid source text.",
            generation_source=LiveResponseGenerationSource.DIAGNOSTIC_SYSTEM,
        )


def _state() -> LiveSessionStateRuntime:
    state = LiveSessionStateRuntime(
        config=LiveSessionConfig(
            mode=LiveSessionMode.REAL_VOICE,
            real_microphone_enabled=True,
            real_stt_enabled=True,
            real_tts_enabled=True,
        )
    )
    state.start()
    state.mark_ready()
    return state


def _runtime() -> LiveDialogueRuntime:
    return LiveDialogueRuntime(
        live_state=_state(),
        response_generator=ContextualDialogueGenerator(),
    )


def test_live_dialogue_start_turn_creates_turn_id() -> None:
    runtime = _runtime()

    result = runtime.start_turn()

    assert result.status == LiveDialogueRuntimeStatus.READY
    assert result.turn is not None
    assert result.turn.turn_id is not None


def test_live_dialogue_processes_transcript_into_generated_response() -> None:
    runtime = _runtime()
    turn = runtime.start_turn()
    assert turn.turn is not None

    transcript = make_live_transcript(
        turn_id=turn.turn.turn_id,
        kind=LiveTranscriptKind.FINAL,
        text="Jarvis teach me control systems.",
        confidence=0.95,
    )
    result = runtime.process_transcript(
        LiveDialogueRequest(
            transcript=transcript,
            metadata={
                "memory": "Last topic was Step 50.",
                "goal": "Build real live JARVIS.",
                "environment": "VS Code is active.",
            },
        )
    )

    assert result.status == LiveDialogueRuntimeStatus.READY
    assert result.turn is not None
    assert result.turn.response is not None
    assert "Balu" in result.turn.response.text
    assert "Jarvis teach me control systems." in result.turn.response.text
    assert "Build real live JARVIS." in result.turn.response.text


def test_live_dialogue_blocks_without_generator() -> None:
    runtime = LiveDialogueRuntime(live_state=_state())
    turn = runtime.start_turn()
    assert turn.turn is not None

    transcript = make_live_transcript(
        turn_id=turn.turn.turn_id,
        kind=LiveTranscriptKind.FINAL,
        text="Jarvis respond.",
        confidence=0.95,
    )
    result = runtime.process_transcript(
        LiveDialogueRequest(transcript=transcript)
    )

    assert result.status == LiveDialogueRuntimeStatus.BLOCKED
    assert result.response_boundary_result is not None


def test_live_dialogue_blocks_invalid_generator_source() -> None:
    runtime = LiveDialogueRuntime(
        live_state=_state(),
        response_generator=InvalidDialogueGenerator(),
    )
    turn = runtime.start_turn()
    assert turn.turn is not None

    transcript = make_live_transcript(
        turn_id=turn.turn.turn_id,
        kind=LiveTranscriptKind.FINAL,
        text="Jarvis respond.",
        confidence=0.95,
    )
    result = runtime.process_transcript(
        LiveDialogueRequest(transcript=transcript)
    )

    assert result.status == LiveDialogueRuntimeStatus.BLOCKED
    assert result.response_boundary_result is not None


def test_live_dialogue_blocks_safety_blocked_response() -> None:
    runtime = _runtime()
    turn = runtime.start_turn()
    assert turn.turn is not None

    transcript = make_live_transcript(
        turn_id=turn.turn.turn_id,
        kind=LiveTranscriptKind.FINAL,
        text="Jarvis respond.",
        confidence=0.95,
    )
    result = runtime.process_transcript(
        LiveDialogueRequest(
            transcript=transcript,
            safety=LiveResponseSafety.BLOCKED,
        )
    )

    assert result.status == LiveDialogueRuntimeStatus.BLOCKED


def test_live_dialogue_finish_response_updates_turn() -> None:
    runtime = _runtime()
    turn = runtime.start_turn()
    assert turn.turn is not None

    transcript = make_live_transcript(
        turn_id=turn.turn.turn_id,
        kind=LiveTranscriptKind.FINAL,
        text="Jarvis explain PID.",
        confidence=0.95,
    )
    processed = runtime.process_transcript(
        LiveDialogueRequest(transcript=transcript)
    )
    assert processed.status == LiveDialogueRuntimeStatus.READY

    finished = runtime.finish_response()

    assert finished.status == LiveDialogueRuntimeStatus.READY
    assert finished.turn is not None
    assert finished.turn.status.value == "response_finished"


def test_live_dialogue_intent_repeat_continue_learning_question() -> None:
    runtime = _runtime()

    examples = (
        ("repeat that", LiveResponseIntent.REPEAT),
        ("continue", LiveResponseIntent.CONTINUE),
        ("teach me thermodynamics", LiveResponseIntent.LEARNING),
        ("what is feedback?", LiveResponseIntent.ANSWER),
    )

    for text, expected in examples:
        turn = runtime.start_turn()
        assert turn.turn is not None
        transcript = make_live_transcript(
            turn_id=turn.turn.turn_id,
            kind=LiveTranscriptKind.FINAL,
            text=text,
            confidence=0.95,
        )
        result = runtime.process_transcript(
            LiveDialogueRequest(transcript=transcript)
        )

        assert result.status == LiveDialogueRuntimeStatus.READY
        assert result.turn is not None
        assert result.turn.metadata["intent"] == expected.value
        runtime.finish_response()


def test_live_dialogue_snapshot_tracks_counts() -> None:
    runtime = _runtime()
    turn = runtime.start_turn()
    assert turn.turn is not None
    transcript = make_live_transcript(
        turn_id=turn.turn.turn_id,
        kind=LiveTranscriptKind.FINAL,
        text="Jarvis continue.",
        confidence=0.95,
    )

    runtime.process_transcript(LiveDialogueRequest(transcript=transcript))
    runtime.finish_response()
    snapshot = runtime.snapshot()

    assert snapshot.status == LiveDialogueRuntimeStatus.READY
    assert snapshot.completed_turns == 1
    assert snapshot.generated_responses == 1


def test_live_dialogue_enum_values_are_stable() -> None:
    assert LiveDialogueRuntimeStatus.READY.value == "ready"