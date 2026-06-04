from __future__ import annotations

import pytest

from jarvis.live import (
    LiveDialogueRuntime,
    LiveInterruptedContext,
    LiveInterruptionDisposition,
    LiveInterruptionKind,
    LiveInterruptionOperation,
    LiveInterruptionPolicy,
    LiveInterruptionRequest,
    LiveInterruptionRuntime,
    LiveInterruptionRuntimeStatus,
    LiveResponseDraft,
    LiveResponseGenerationRequest,
    LiveResponseGenerationSource,
    LiveResponseKind,
    LiveResponseSafety,
    LiveSessionConfig,
    LiveSessionMode,
    LiveSessionStateRuntime,
    LiveTranscriptKind,
    make_live_response,
    make_live_transcript,
)


class InterruptionDialogueGenerator:
    def generate(
        self,
        request: LiveResponseGenerationRequest,
    ) -> LiveResponseDraft:
        context = request.context
        pieces = (
            context.user_text,
            " ".join(context.memory_context),
            " ".join(context.goal_context),
            str(context.metadata.get("interrupted_context", "")),
            str(context.metadata.get("repeat", "")),
            str(context.metadata.get("resume", "")),
        )
        text = " | ".join(piece for piece in pieces if piece.strip())
        return LiveResponseDraft(
            text=text or "generated interruption response",
            generation_source=LiveResponseGenerationSource.RESPONSE_GENERATOR,
            token_count=len(text.split()),
            metadata={"test_generator": "interruption"},
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


def _runtime() -> LiveInterruptionRuntime:
    state = _state()
    dialogue = LiveDialogueRuntime(
        live_state=state,
        response_generator=InterruptionDialogueGenerator(),
    )
    return LiveInterruptionRuntime(
        live_state=state,
        dialogue=dialogue,
    )


def _seed_response(runtime: LiveInterruptionRuntime) -> None:
    turn = runtime.live_state.start_user_turn()
    assert turn.state.current_turn_id is not None

    transcript = make_live_transcript(
        turn_id=turn.state.current_turn_id,
        kind=LiveTranscriptKind.FINAL,
        text="Jarvis explain PID control.",
        confidence=0.95,
    )
    runtime.live_state.transcript_ready(transcript)
    response = make_live_response(
        turn_id=turn.state.current_turn_id,
        kind=LiveResponseKind.CONVERSATIONAL,
        text="Generated PID explanation.",
        generation_source=LiveResponseGenerationSource.RESPONSE_GENERATOR,
        safety=LiveResponseSafety.SAFE_TO_SPEAK,
    )
    runtime.live_state.start_speaking(response)


def test_interruption_policy_validation() -> None:
    with pytest.raises(ValueError):
        LiveInterruptionPolicy(min_confidence=2.0)


def test_interruption_runtime_captures_context() -> None:
    runtime = _runtime()
    _seed_response(runtime)

    result = runtime.capture_context(reason="user said wait")

    assert result.status == LiveInterruptionRuntimeStatus.READY
    assert result.interrupted_context is not None
    assert result.interrupted_context.has_resume_point is True
    assert "interruption=user said wait" in (
        result.interrupted_context.resume_summary
    )


def test_interruption_runtime_blocks_low_confidence() -> None:
    runtime = _runtime()

    result = runtime.request_interrupt(
        LiveInterruptionRequest(
            text="stop",
            confidence=0.1,
            assistant_is_speaking=True,
        )
    )

    assert result.status == LiveInterruptionRuntimeStatus.BLOCKED


def test_interruption_runtime_requests_stop() -> None:
    runtime = _runtime()
    _seed_response(runtime)

    result = runtime.request_interrupt(
        LiveInterruptionRequest(
            text="stop",
            confidence=0.95,
            assistant_is_speaking=True,
        )
    )

    assert result.status == LiveInterruptionRuntimeStatus.READY
    assert result.kind == LiveInterruptionKind.STOP
    assert result.disposition == LiveInterruptionDisposition.STOP_OUTPUT
    assert result.bridge_result is not None
    assert result.bridge_result.should_interrupt is True
    assert result.interrupted_context is not None


def test_interruption_runtime_requests_pause() -> None:
    runtime = _runtime()

    result = runtime.request_interrupt(
        LiveInterruptionRequest(
            text="wait",
            confidence=0.95,
            assistant_is_speaking=True,
        )
    )

    assert result.status == LiveInterruptionRuntimeStatus.READY
    assert result.kind == LiveInterruptionKind.PAUSE
    assert result.disposition == LiveInterruptionDisposition.PAUSE_AND_LISTEN


def test_interruption_runtime_routes_question_through_dialogue() -> None:
    runtime = _runtime()
    _seed_response(runtime)
    interrupt = runtime.request_interrupt(
        LiveInterruptionRequest(
            text="wait",
            confidence=0.95,
            assistant_is_speaking=True,
        )
    )
    assert interrupt.interrupted_context is not None
    assert interrupt.interrupted_context.turn_id is not None

    transcript = make_live_transcript(
        turn_id=interrupt.interrupted_context.turn_id,
        kind=LiveTranscriptKind.INTERRUPTION,
        text="what does integral mean?",
        confidence=0.95,
    )
    result = runtime.handle_interrupt_transcript(transcript)

    assert result.status == LiveInterruptionRuntimeStatus.READY
    assert result.kind == LiveInterruptionKind.QUESTION
    assert result.dialogue_result is not None
    assert result.dialogue_result.turn is not None
    assert result.dialogue_result.turn.response is not None
    assert "interrupted_context" in result.dialogue_result.turn.metadata


def test_interruption_runtime_repeats_through_dialogue() -> None:
    runtime = _runtime()
    _seed_response(runtime)
    captured = runtime.capture_context(reason="repeat requested")
    assert captured.interrupted_context is not None
    assert captured.interrupted_context.turn_id is not None

    transcript = make_live_transcript(
        turn_id=captured.interrupted_context.turn_id,
        kind=LiveTranscriptKind.INTERRUPTION,
        text="repeat that",
        confidence=0.95,
    )
    result = runtime.handle_interrupt_transcript(transcript)

    assert result.status == LiveInterruptionRuntimeStatus.READY
    assert result.kind == LiveInterruptionKind.REPEAT
    assert result.dialogue_result is not None


def test_interruption_runtime_resumes_previous_context() -> None:
    runtime = _runtime()
    _seed_response(runtime)
    captured = runtime.capture_context(reason="pause")
    assert captured.interrupted_context is not None

    result = runtime.resume()

    assert result.status == LiveInterruptionRuntimeStatus.READY
    assert result.kind == LiveInterruptionKind.CONTINUE
    assert result.should_resume is True
    assert result.dialogue_result is not None


def test_interruption_runtime_blocks_resume_without_context() -> None:
    runtime = _runtime()

    result = runtime.resume()

    assert result.status == LiveInterruptionRuntimeStatus.BLOCKED


def test_interruption_runtime_cancel_resume_clears_context() -> None:
    runtime = _runtime()
    _seed_response(runtime)
    runtime.capture_context(reason="pause")

    result = runtime.cancel_resume(reason="user changed topic")

    assert result.status == LiveInterruptionRuntimeStatus.READY
    assert result.operation == LiveInterruptionOperation.CANCEL_RESUME
    assert runtime.active_context is None


def test_interruption_runtime_snapshot_tracks_counts() -> None:
    runtime = _runtime()
    _seed_response(runtime)

    runtime.request_interrupt(
        LiveInterruptionRequest(
            text="wait",
            confidence=0.95,
            assistant_is_speaking=True,
        )
    )
    runtime.resume()
    runtime.cancel_resume(reason="done")

    snapshot = runtime.snapshot()

    assert snapshot.status == LiveInterruptionRuntimeStatus.READY
    assert snapshot.interruption_count == 1
    assert snapshot.resume_count == 1
    assert snapshot.cancelled_count == 1


def test_interruption_runtime_enum_values_are_stable() -> None:
    assert LiveInterruptionRuntimeStatus.READY.value == "ready"
    assert LiveInterruptionKind.STOP.value == "stop"
    assert LiveInterruptionDisposition.RESUME_PREVIOUS.value == (
        "resume_previous"
    )
    assert LiveInterruptedContext.__name__ == "LiveInterruptedContext"