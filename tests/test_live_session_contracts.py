from __future__ import annotations

import pytest

from jarvis.live import (
    LiveAudioFrameKind,
    LiveAudioState,
    LiveEventKind,
    LiveEventPriority,
    LiveHealthStatus,
    LiveInteractionState,
    LiveResponseGenerationSource,
    LiveResponseKind,
    LiveResponseSafety,
    LiveSessionConfig,
    LiveSessionDesignGate,
    LiveSessionMode,
    LiveSessionPhase,
    LiveSessionSnapshot,
    LiveSessionStatus,
    LiveSubsystem,
    LiveSubsystemStatus,
    LiveTranscriptKind,
    LiveWakeState,
    default_live_session_config,
    default_live_session_state,
    make_live_audio_frame,
    make_live_event,
    make_live_response,
    make_live_transcript,
    make_live_turn_id,
    utc_now,
)


def test_live_session_config_rejects_empty_user_label() -> None:
    with pytest.raises(ValueError):
        LiveSessionConfig(user_label=" ")


def test_live_session_config_rejects_invalid_session_seconds() -> None:
    with pytest.raises(ValueError):
        LiveSessionConfig(max_session_seconds=0)


def test_default_live_session_config_is_safe_simulation() -> None:
    config = default_live_session_config()

    assert config.user_label == "Balu"
    assert config.assistant_name == "JARVIS"
    assert config.mode == LiveSessionMode.SAFE_SIMULATION
    assert config.real_microphone_enabled is False
    assert config.real_stt_enabled is False
    assert config.real_tts_enabled is False
    assert config.interruption_enabled is True


def test_default_live_session_state_contains_all_organs() -> None:
    state = default_live_session_state()
    subsystems = {item.subsystem for item in state.subsystem_states}

    assert LiveSubsystem.RUNTIME_KERNEL in subsystems
    assert LiveSubsystem.EVENT_BUS in subsystems
    assert LiveSubsystem.PRESENCE in subsystems
    assert LiveSubsystem.CONVERSATION in subsystems
    assert LiveSubsystem.COGNITION in subsystems
    assert LiveSubsystem.MEMORY in subsystems
    assert LiveSubsystem.TOOLS in subsystems
    assert LiveSubsystem.ORCHESTRATION in subsystems
    assert LiveSubsystem.LATENCY in subsystems
    assert LiveSubsystem.ENVIRONMENT in subsystems
    assert LiveSubsystem.DEVELOPER_PACK in subsystems
    assert LiveSubsystem.COGNITIVE_SESSION in subsystems
    assert LiveSubsystem.MICROPHONE in subsystems
    assert LiveSubsystem.STT in subsystems
    assert LiveSubsystem.TTS in subsystems
    assert LiveSubsystem.PLAYBACK in subsystems
    assert LiveSubsystem.WAKE in subsystems
    assert LiveSubsystem.INTERRUPTION in subsystems
    assert LiveSubsystem.HEALTH_MONITOR in subsystems
    assert LiveSubsystem.RECOVERY in subsystems
    assert LiveSubsystem.RESPONSE_GENERATOR in subsystems


def test_safe_simulation_disables_real_audio_but_keeps_contracts() -> None:
    state = default_live_session_state()
    by_subsystem = {item.subsystem: item for item in state.subsystem_states}

    assert by_subsystem[LiveSubsystem.MICROPHONE].status == (
        LiveSubsystemStatus.DISABLED
    )
    assert by_subsystem[LiveSubsystem.STT].status == LiveSubsystemStatus.DISABLED
    assert by_subsystem[LiveSubsystem.TTS].status == LiveSubsystemStatus.DISABLED
    assert by_subsystem[LiveSubsystem.PLAYBACK].status == (
        LiveSubsystemStatus.DISABLED
    )
    assert state.microphone_active is False
    assert state.stt_active is False
    assert state.tts_active is False


def test_real_voice_config_marks_audio_as_active() -> None:
    config = LiveSessionConfig(
        mode=LiveSessionMode.REAL_VOICE,
        real_microphone_enabled=True,
        real_stt_enabled=True,
        real_tts_enabled=True,
    )
    state = default_live_session_state(config=config)
    by_subsystem = {item.subsystem: item for item in state.subsystem_states}

    assert by_subsystem[LiveSubsystem.MICROPHONE].status == (
        LiveSubsystemStatus.READY
    )
    assert by_subsystem[LiveSubsystem.STT].status == LiveSubsystemStatus.READY
    assert by_subsystem[LiveSubsystem.TTS].status == LiveSubsystemStatus.READY
    assert by_subsystem[LiveSubsystem.PLAYBACK].status == (
        LiveSubsystemStatus.READY
    )
    assert state.microphone_active is True
    assert state.stt_active is True
    assert state.tts_active is True


def test_live_session_state_capability_properties() -> None:
    config = LiveSessionConfig(
        mode=LiveSessionMode.REAL_VOICE,
        real_microphone_enabled=True,
        real_stt_enabled=True,
        real_tts_enabled=True,
    )
    created = default_live_session_state(config=config)
    running = created.__class__(
        session_id=created.session_id,
        status=LiveSessionStatus.RUNNING,
        phase=LiveSessionPhase.LISTENING,
        mode=created.mode,
        interaction_state=LiveInteractionState.LISTENING,
        wake_state=created.wake_state,
        audio_state=created.audio_state,
        health_status=created.health_status,
        user_label=created.user_label,
        assistant_name=created.assistant_name,
        started_at=utc_now(),
        updated_at=utc_now(),
        user_present=True,
        microphone_active=created.microphone_active,
        stt_active=created.stt_active,
        tts_active=created.tts_active,
        playback_active=created.playback_active,
        assistant_speaking=False,
        conversation_active=True,
        interruption_enabled=created.interruption_enabled,
        wake_enabled=created.wake_enabled,
        environment_enabled=created.environment_enabled,
        memory_enabled=created.memory_enabled,
        goal_tracking_enabled=created.goal_tracking_enabled,
        developer_pack_enabled=created.developer_pack_enabled,
        tools_enabled=created.tools_enabled,
        subsystem_states=created.subsystem_states,
    )

    assert running.is_running is True
    assert running.can_listen is True
    assert running.can_speak is True
    assert running.can_interrupt is True


def test_live_session_event_validation() -> None:
    with pytest.raises(ValueError):
        make_live_event(
            kind=LiveEventKind.ERROR,
            priority=LiveEventPriority.CRITICAL,
            source=LiveSubsystem.HEALTH_MONITOR,
            title=" ",
            summary="error",
        )


def test_make_live_event_creates_valid_event() -> None:
    event = make_live_event(
        kind=LiveEventKind.SESSION_STARTED,
        priority=LiveEventPriority.NORMAL,
        source=LiveSubsystem.RUNTIME_KERNEL,
        title="Session started",
        summary="Live session started.",
    )

    assert event.event_id.startswith("live_evt_")
    assert event.kind == LiveEventKind.SESSION_STARTED
    assert event.source == LiveSubsystem.RUNTIME_KERNEL


def test_live_audio_frame_contract_validation() -> None:
    frame = make_live_audio_frame(
        kind=LiveAudioFrameKind.INPUT,
        sample_rate_hz=16000,
        channels=1,
        duration_ms=20,
        rms=0.1,
        speech_probability=0.8,
    )

    assert frame.frame_id.startswith("audio_")
    assert frame.kind == LiveAudioFrameKind.INPUT
    assert frame.sample_rate_hz == 16000

    with pytest.raises(ValueError):
        make_live_audio_frame(
            kind=LiveAudioFrameKind.INPUT,
            sample_rate_hz=0,
            channels=1,
            duration_ms=20,
        )


def test_live_transcript_contract_validation() -> None:
    turn_id = make_live_turn_id()
    transcript = make_live_transcript(
        turn_id=turn_id,
        kind=LiveTranscriptKind.FINAL,
        text="Jarvis continue the live session.",
        confidence=0.95,
    )

    assert transcript.turn_id == turn_id
    assert transcript.kind == LiveTranscriptKind.FINAL
    assert transcript.confidence == 0.95

    with pytest.raises(ValueError):
        make_live_transcript(
            turn_id=turn_id,
            kind=LiveTranscriptKind.FINAL,
            text=" ",
            confidence=0.9,
        )


def test_live_response_contract_separates_generated_conversation() -> None:
    turn_id = make_live_turn_id()
    response = make_live_response(
        turn_id=turn_id,
        kind=LiveResponseKind.CONVERSATIONAL,
        text="Generated response placeholder for test.",
        generation_source=LiveResponseGenerationSource.RESPONSE_GENERATOR,
        safety=LiveResponseSafety.SAFE_TO_SPEAK,
    )

    assert response.turn_id == turn_id
    assert response.is_conversational is True
    assert response.generated_by_cognition is True
    assert response.deterministic_system_response is False


def test_live_diagnostic_response_is_not_conversational() -> None:
    turn_id = make_live_turn_id()
    response = make_live_response(
        turn_id=turn_id,
        kind=LiveResponseKind.DIAGNOSTIC,
        text="Microphone unavailable.",
        generation_source=LiveResponseGenerationSource.DIAGNOSTIC_SYSTEM,
        safety=LiveResponseSafety.SAFE_TO_SPEAK,
    )

    assert response.is_conversational is False
    assert response.generated_by_cognition is False
    assert response.deterministic_system_response is True


def test_live_session_design_gate_passes_default_state() -> None:
    state = default_live_session_state()
    report = LiveSessionDesignGate().validate(state)

    assert report.passed is True
    assert report.failed_count == 0
    assert report.passed_count == len(report.checks)


def test_live_session_snapshot_reports_running_state() -> None:
    state = default_live_session_state()
    snapshot = LiveSessionSnapshot(
        state=state,
        event_count=0,
        uptime_seconds=0.0,
        created_at=utc_now(),
    )

    assert snapshot.running is False
    assert snapshot.event_count == 0


def test_live_enum_values_are_stable() -> None:
    assert LiveSessionStatus.RUNNING.value == "running"
    assert LiveSessionMode.SAFE_SIMULATION.value == "safe_simulation"
    assert LiveSessionPhase.LISTENING.value == "listening"
    assert LiveInteractionState.LISTENING.value == "listening"
    assert LiveWakeState.LISTENING_FOR_WAKE.value == "listening_for_wake"
    assert LiveAudioState.STT_READY.value == "stt_ready"
    assert LiveHealthStatus.HEALTHY.value == "healthy"
    assert LiveAudioFrameKind.INPUT.value == "input"
    assert LiveTranscriptKind.FINAL.value == "final"
    assert LiveResponseKind.CONVERSATIONAL.value == "conversational"
    assert LiveResponseGenerationSource.RESPONSE_GENERATOR.value == (
        "response_generator"
    )
    assert LiveResponseSafety.SAFE_TO_SPEAK.value == "safe_to_speak"