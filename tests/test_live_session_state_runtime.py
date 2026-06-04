from __future__ import annotations

import pytest

from jarvis.live import (
    LiveAudioState,
    LiveHealthStatus,
    LiveInteractionState,
    LiveResponseGenerationSource,
    LiveResponseKind,
    LiveResponseSafety,
    LiveSessionConfig,
    LiveSessionMode,
    LiveSessionPhase,
    LiveSessionStateOperation,
    LiveSessionStateRuntime,
    LiveSessionStateRuntimeStatus,
    LiveSessionStatus,
    LiveShutdownReason,
    LiveSubsystem,
    LiveSubsystemState,
    LiveSubsystemStatus,
    LiveTranscriptKind,
    make_live_response,
    make_live_transcript,
    make_live_turn_id,
    utc_now,
)


def _real_voice_runtime() -> LiveSessionStateRuntime:
    return LiveSessionStateRuntime(
        config=LiveSessionConfig(
            mode=LiveSessionMode.REAL_VOICE,
            real_microphone_enabled=True,
            real_stt_enabled=True,
            real_tts_enabled=True,
        )
    )


def test_live_session_state_runtime_starts_and_marks_ready() -> None:
    runtime = _real_voice_runtime()

    started = runtime.start()
    ready = runtime.mark_ready()

    assert started.status == LiveSessionStateRuntimeStatus.READY
    assert started.operation == LiveSessionStateOperation.START
    assert ready.status == LiveSessionStateRuntimeStatus.READY
    assert ready.state.status == LiveSessionStatus.RUNNING
    assert ready.state.phase == LiveSessionPhase.READY
    assert ready.state.conversation_active is True
    assert len(runtime.events) == 2


def test_live_session_state_blocks_double_start() -> None:
    runtime = _real_voice_runtime()

    runtime.start()
    blocked = runtime.start()

    assert blocked.status == LiveSessionStateRuntimeStatus.BLOCKED
    assert blocked.succeeded is False


def test_live_session_state_enters_listening_only_with_audio_ready() -> None:
    safe = LiveSessionStateRuntime()
    safe.start()
    safe.mark_ready()

    blocked = safe.enter_listening()

    assert blocked.status == LiveSessionStateRuntimeStatus.BLOCKED

    runtime = _real_voice_runtime()
    runtime.start()
    runtime.mark_ready()

    listening = runtime.enter_listening()

    assert listening.status == LiveSessionStateRuntimeStatus.READY
    assert listening.state.phase == LiveSessionPhase.LISTENING
    assert listening.state.interaction_state == LiveInteractionState.LISTENING
    assert listening.state.audio_state == LiveAudioState.STREAMING_INPUT


def test_live_session_state_user_turn_transcript_and_thinking() -> None:
    runtime = _real_voice_runtime()
    runtime.start()
    runtime.mark_ready()
    runtime.enter_listening()

    turn = runtime.start_user_turn()
    assert turn.state.current_turn_id is not None

    transcript = make_live_transcript(
        turn_id=turn.state.current_turn_id,
        kind=LiveTranscriptKind.FINAL,
        text="Jarvis continue step 50.",
        confidence=0.95,
    )
    accepted = runtime.transcript_ready(transcript)
    thinking = runtime.start_thinking()

    assert accepted.status == LiveSessionStateRuntimeStatus.READY
    assert accepted.state.last_transcript == transcript
    assert thinking.state.phase == LiveSessionPhase.THINKING
    assert thinking.state.interaction_state == LiveInteractionState.THINKING


def test_live_session_state_rejects_wrong_turn_transcript() -> None:
    runtime = _real_voice_runtime()
    runtime.start()
    runtime.mark_ready()
    runtime.start_user_turn()

    transcript = make_live_transcript(
        turn_id=make_live_turn_id(),
        kind=LiveTranscriptKind.FINAL,
        text="Wrong turn.",
        confidence=0.9,
    )
    result = runtime.transcript_ready(transcript)

    assert result.status == LiveSessionStateRuntimeStatus.BLOCKED


def test_live_session_state_speaking_requires_tts_and_response() -> None:
    safe = LiveSessionStateRuntime()
    safe.start()
    safe.mark_ready()
    response = make_live_response(
        turn_id=make_live_turn_id(),
        kind=LiveResponseKind.CONVERSATIONAL,
        text="Generated response for test.",
        generation_source=LiveResponseGenerationSource.RESPONSE_GENERATOR,
        safety=LiveResponseSafety.SAFE_TO_SPEAK,
    )

    blocked = safe.start_speaking(response)

    assert blocked.status == LiveSessionStateRuntimeStatus.BLOCKED

    runtime = _real_voice_runtime()
    runtime.start()
    runtime.mark_ready()

    speaking = runtime.start_speaking(response)

    assert speaking.status == LiveSessionStateRuntimeStatus.READY
    assert speaking.state.phase == LiveSessionPhase.SPEAKING
    assert speaking.state.assistant_speaking is True
    assert speaking.state.last_response == response


def test_live_session_state_blocks_safety_blocked_response() -> None:
    runtime = _real_voice_runtime()
    runtime.start()
    runtime.mark_ready()
    response = make_live_response(
        turn_id=make_live_turn_id(),
        kind=LiveResponseKind.CONVERSATIONAL,
        text="Blocked response.",
        generation_source=LiveResponseGenerationSource.RESPONSE_GENERATOR,
        safety=LiveResponseSafety.BLOCKED,
    )

    result = runtime.start_speaking(response)

    assert result.status == LiveSessionStateRuntimeStatus.BLOCKED


def test_live_session_state_finish_speaking_returns_to_ready() -> None:
    runtime = _real_voice_runtime()
    runtime.start()
    runtime.mark_ready()
    response = make_live_response(
        turn_id=make_live_turn_id(),
        kind=LiveResponseKind.CONVERSATIONAL,
        text="Generated response for test.",
        generation_source=LiveResponseGenerationSource.RESPONSE_GENERATOR,
        safety=LiveResponseSafety.SAFE_TO_SPEAK,
    )

    runtime.start_speaking(response)
    finished = runtime.finish_speaking()

    assert finished.status == LiveSessionStateRuntimeStatus.READY
    assert finished.state.assistant_speaking is False
    assert finished.state.interaction_state == (
        LiveInteractionState.WAITING_FOR_USER
    )


def test_live_session_state_interrupts_without_generating_speech() -> None:
    runtime = _real_voice_runtime()
    runtime.start()
    runtime.mark_ready()

    result = runtime.interrupt(reason="user barge-in")

    assert result.status == LiveSessionStateRuntimeStatus.READY
    assert result.state.phase == LiveSessionPhase.INTERRUPTED
    assert result.state.assistant_speaking is False
    assert result.event is not None
    assert result.event.source == LiveSubsystem.INTERRUPTION


def test_live_session_state_interrupt_requires_reason() -> None:
    runtime = _real_voice_runtime()
    runtime.start()
    runtime.mark_ready()

    with pytest.raises(ValueError):
        runtime.interrupt(reason=" ")


def test_live_session_state_recovery_cycle() -> None:
    runtime = _real_voice_runtime()
    runtime.start()
    runtime.mark_ready()

    recovering = runtime.enter_recovery(
        subsystem=LiveSubsystem.STT,
        reason="STT adapter timeout.",
    )

    assert recovering.state.status == LiveSessionStatus.DEGRADED
    assert recovering.state.phase == LiveSessionPhase.RECOVERING
    assert recovering.state.health_status == LiveHealthStatus.DEGRADED

    recovered = runtime.finish_recovery()

    assert recovered.state.status == LiveSessionStatus.RUNNING
    assert recovered.state.phase == LiveSessionPhase.READY
    assert recovered.state.health_status == LiveHealthStatus.HEALTHY


def test_live_session_state_updates_subsystem_health() -> None:
    runtime = _real_voice_runtime()
    runtime.start()
    runtime.mark_ready()

    result = runtime.update_subsystem(
        LiveSubsystemState(
            subsystem=LiveSubsystem.STT,
            status=LiveSubsystemStatus.FAILED,
            message="STT failed.",
            updated_at=utc_now(),
        )
    )

    assert result.state.status == LiveSessionStatus.FAILED
    assert result.state.health_status == LiveHealthStatus.FAILED
    assert result.event is not None
    assert result.event.priority.value == "critical"


def test_live_session_state_update_health_can_fail_session() -> None:
    runtime = _real_voice_runtime()
    runtime.start()
    runtime.mark_ready()

    result = runtime.update_health(LiveHealthStatus.FAILED)

    assert result.state.status == LiveSessionStatus.FAILED
    assert result.state.health_status == LiveHealthStatus.FAILED


def test_live_session_state_stop_disables_live_activity() -> None:
    runtime = _real_voice_runtime()
    runtime.start()
    runtime.mark_ready()

    stopped = runtime.stop(reason=LiveShutdownReason.USER_REQUEST)

    assert stopped.status == LiveSessionStateRuntimeStatus.READY
    assert stopped.state.status == LiveSessionStatus.STOPPED
    assert stopped.state.microphone_active is False
    assert stopped.state.stt_active is False
    assert stopped.state.tts_active is False
    assert stopped.state.playback_active is False
    assert stopped.state.shutdown_reason == LiveShutdownReason.USER_REQUEST


def test_live_session_state_snapshot_and_clear_events() -> None:
    runtime = _real_voice_runtime()
    runtime.start()
    runtime.mark_ready()

    snapshot = runtime.snapshot()

    assert snapshot.status == LiveSessionStateRuntimeStatus.READY
    assert snapshot.event_count == 2
    assert snapshot.transition_count == 2
    assert snapshot.uptime_seconds >= 0.0

    cleared = runtime.clear_events()

    assert cleared.status == LiveSessionStateRuntimeStatus.READY
    assert runtime.events == ()


def test_live_session_state_enum_values_are_stable() -> None:
    assert LiveSessionStateRuntimeStatus.READY.value == "ready"
    assert LiveSessionStateOperation.START.value == "start"
    assert LiveSessionStateOperation.INTERRUPT.value == "interrupt"