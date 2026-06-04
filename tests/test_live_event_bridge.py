from __future__ import annotations

from jarvis.live import (
    LiveEventBridgeRequest,
    LiveEventBridgeRuntime,
    LiveEventBridgeStatus,
    LiveEventKind,
    LiveEventPriority,
    LiveResponseBridgeRequest,
    LiveResponseGenerationSource,
    LiveResponseKind,
    LiveResponseSafety,
    LiveSessionConfig,
    LiveSessionMode,
    LiveSessionStateRuntime,
    LiveSessionStatus,
    LiveSubsystem,
    LiveTranscriptBridgeRequest,
    LiveTranscriptKind,
    make_live_event,
    make_live_response,
    make_live_transcript,
)


def _bridge() -> LiveEventBridgeRuntime:
    state = LiveSessionStateRuntime(
        config=LiveSessionConfig(
            mode=LiveSessionMode.REAL_VOICE,
            real_microphone_enabled=True,
            real_stt_enabled=True,
            real_tts_enabled=True,
        )
    )
    return LiveEventBridgeRuntime(live_state=state)


def test_live_event_bridge_starts_and_marks_ready() -> None:
    bridge = _bridge()
    start = make_live_event(
        kind=LiveEventKind.SESSION_START_REQUESTED,
        priority=LiveEventPriority.NORMAL,
        source=LiveSubsystem.RUNTIME_KERNEL,
        title="Start",
        summary="Start live session.",
    )
    started = bridge.bridge_event(LiveEventBridgeRequest(event=start))

    assert started.status == LiveEventBridgeStatus.READY
    assert started.live_result is not None
    assert started.live_result.state.status == LiveSessionStatus.STARTING

    ready = make_live_event(
        kind=LiveEventKind.SESSION_STARTED,
        priority=LiveEventPriority.NORMAL,
        source=LiveSubsystem.RUNTIME_KERNEL,
        title="Ready",
        summary="Live session ready.",
    )
    result = bridge.bridge_event(LiveEventBridgeRequest(event=ready))

    assert result.status == LiveEventBridgeStatus.READY
    assert result.live_result is not None
    assert result.live_result.state.status == LiveSessionStatus.RUNNING


def test_live_event_bridge_transcript_updates_cognitive_session() -> None:
    bridge = _bridge()
    bridge.live_state.start()
    bridge.live_state.mark_ready()
    turn = bridge.live_state.start_user_turn()
    assert turn.state.current_turn_id is not None

    transcript = make_live_transcript(
        turn_id=turn.state.current_turn_id,
        kind=LiveTranscriptKind.FINAL,
        text="Jarvis continue step 50.",
        confidence=0.95,
    )
    result = bridge.bridge_transcript(
        LiveTranscriptBridgeRequest(transcript=transcript)
    )

    assert result.status == LiveEventBridgeStatus.READY
    assert result.live_result is not None
    assert result.cognitive_result is not None
    assert result.cognitive_result.processed_events
    assert result.live_result.state.last_transcript == transcript


def test_live_event_bridge_blocks_wrong_turn_transcript() -> None:
    bridge = _bridge()
    bridge.live_state.start()
    bridge.live_state.mark_ready()
    bridge.live_state.start_user_turn()

    transcript = make_live_transcript(
        turn_id=make_live_transcript(
            turn_id=bridge.live_state.state.current_turn_id,  # type: ignore[arg-type]
            kind=LiveTranscriptKind.FINAL,
            text="temporary",
            confidence=0.9,
        ).turn_id,
        kind=LiveTranscriptKind.FINAL,
        text="Wrong turn.",
        confidence=0.9,
    )

    # force mismatch by starting a new turn after making transcript
    bridge.live_state.start_user_turn()
    result = bridge.bridge_transcript(
        LiveTranscriptBridgeRequest(transcript=transcript)
    )

    assert result.status == LiveEventBridgeStatus.BLOCKED


def test_live_event_bridge_interruption_routes_to_state_and_cognition() -> None:
    bridge = _bridge()
    bridge.live_state.start()
    bridge.live_state.mark_ready()
    event = make_live_event(
        kind=LiveEventKind.INTERRUPTION_REQUESTED,
        priority=LiveEventPriority.CRITICAL,
        source=LiveSubsystem.INTERRUPTION,
        title="Interrupt",
        summary="User interrupted current response.",
    )

    result = bridge.bridge_event(LiveEventBridgeRequest(event=event))

    assert result.status == LiveEventBridgeStatus.READY
    assert result.should_interrupt is True
    assert result.live_result is not None
    assert result.cognitive_result is not None


def test_live_event_bridge_developer_signal_updates_cognitive_memory() -> None:
    bridge = _bridge()
    event = make_live_event(
        kind=LiveEventKind.DEVELOPER_SIGNAL_RECEIVED,
        priority=LiveEventPriority.HIGH,
        source=LiveSubsystem.DEVELOPER_PACK,
        title="Build failed",
        summary="Mypy failed in live event bridge.",
    )

    result = bridge.bridge_event(LiveEventBridgeRequest(event=event))

    assert result.status == LiveEventBridgeStatus.READY
    assert result.cognitive_result is not None
    session = result.cognitive_result.session_result.session
    assert session.working_memory.items


def test_live_event_bridge_environment_signal_updates_cognitive_context() -> None:
    bridge = _bridge()
    event = make_live_event(
        kind=LiveEventKind.ENVIRONMENT_CONTEXT_UPDATED,
        priority=LiveEventPriority.NORMAL,
        source=LiveSubsystem.ENVIRONMENT,
        title="Screen context",
        summary="VS Code is active.",
    )

    result = bridge.bridge_event(LiveEventBridgeRequest(event=event))

    assert result.status == LiveEventBridgeStatus.READY
    assert result.cognitive_result is not None
    assert result.cognitive_result.processed_events


def test_live_event_bridge_response_requires_boundary_validation() -> None:
    bridge = _bridge()
    bridge.live_state.start()
    bridge.live_state.mark_ready()

    response = make_live_response(
        turn_id=bridge.live_state.state.current_turn_id
        or make_live_transcript(
            turn_id=bridge.live_state.start_user_turn().state.current_turn_id,  # type: ignore[arg-type]
            kind=LiveTranscriptKind.FINAL,
            text="temporary",
            confidence=0.9,
        ).turn_id,
        kind=LiveResponseKind.CONVERSATIONAL,
        text="Generated response.",
        generation_source=LiveResponseGenerationSource.RESPONSE_GENERATOR,
        safety=LiveResponseSafety.SAFE_TO_SPEAK,
    )

    result = bridge.bridge_response(LiveResponseBridgeRequest(response=response))

    assert result.status == LiveEventBridgeStatus.READY
    assert result.response_boundary_result is not None
    assert result.live_result is not None


def test_live_event_bridge_blocks_scripted_conversational_response() -> None:
    bridge = _bridge()
    bridge.live_state.start()
    bridge.live_state.mark_ready()
    bridge.live_state.start_user_turn()
    assert bridge.live_state.state.current_turn_id is not None

    response = make_live_response(
        turn_id=bridge.live_state.state.current_turn_id,
        kind=LiveResponseKind.CONVERSATIONAL,
        text="Scripted diagnostic conversation.",
        generation_source=LiveResponseGenerationSource.DIAGNOSTIC_SYSTEM,
        safety=LiveResponseSafety.SAFE_TO_SPEAK,
    )

    result = bridge.bridge_response(LiveResponseBridgeRequest(response=response))

    assert result.status == LiveEventBridgeStatus.BLOCKED


def test_live_event_bridge_snapshot_tracks_counts() -> None:
    bridge = _bridge()
    event = make_live_event(
        kind=LiveEventKind.DEVELOPER_SIGNAL_RECEIVED,
        priority=LiveEventPriority.NORMAL,
        source=LiveSubsystem.DEVELOPER_PACK,
        title="Build passed",
        summary="All checks passed.",
    )

    bridge.bridge_event(LiveEventBridgeRequest(event=event))
    snapshot = bridge.snapshot()

    assert snapshot.status == LiveEventBridgeStatus.READY
    assert snapshot.bridged_event_count == 1
    assert snapshot.cognitive_session_id


def test_live_event_bridge_enum_values_are_stable() -> None:
    assert LiveEventBridgeStatus.READY.value == "ready"