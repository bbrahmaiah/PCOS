from __future__ import annotations

from jarvis.live import (
    LiveResponseDraft,
    LiveResponseGenerationRequest,
    LiveResponseGenerationSource,
    LiveSessionConfig,
    LiveSessionMode,
    LiveSessionRunner,
    LiveSessionRunnerConfig,
    LiveSessionRunnerStatus,
    LiveShutdownReason,
)


class RunnerFakeGenerator:
    def generate(
        self,
        request: LiveResponseGenerationRequest,
    ) -> LiveResponseDraft:
        context = request.context
        pieces = (
            context.live_state.user_label,
            request.intent.value,
            context.user_text,
            " ".join(context.memory_context),
            " ".join(context.goal_context),
            " ".join(context.environment_context),
        )
        text = " | ".join(piece for piece in pieces if piece.strip())
        return LiveResponseDraft(
            text=text,
            generation_source=LiveResponseGenerationSource.RESPONSE_GENERATOR,
            token_count=len(text.split()),
            metadata={"test_generator": "runner"},
        )


def _runner() -> LiveSessionRunner:
    return LiveSessionRunner(
        config=LiveSessionRunnerConfig(
            session_config=LiveSessionConfig(
                mode=LiveSessionMode.REAL_VOICE,
                real_microphone_enabled=True,
                real_stt_enabled=True,
                real_tts_enabled=True,
            ),
            auto_prepare_audio=False,
            auto_health_check=False,
            auto_recover=False,
        ),
        response_generator=RunnerFakeGenerator(),
    )


def test_live_session_runner_starts() -> None:
    runner = _runner()

    result = runner.start()

    assert result.status == LiveSessionRunnerStatus.RUNNING
    assert result.state_result is not None
    assert runner.live_state.state.conversation_active is True


def test_live_session_runner_ignores_empty_text() -> None:
    runner = _runner()
    runner.start()

    result = runner.ingest_text(text=" ")

    assert result.status == LiveSessionRunnerStatus.RUNNING
    assert result.dialogue_result is None


def test_live_session_runner_ingests_wake_text_through_dialogue() -> None:
    runner = _runner()
    runner.start()

    result = runner.ingest_text(
        text="Jarvis teach me control systems.",
        metadata={
            "memory": "Previous focus was Step 50.",
            "goal": "Build real live JARVIS.",
            "environment": "VS Code is active.",
        },
    )

    assert result.status == LiveSessionRunnerStatus.RUNNING
    assert result.wake_result is not None
    assert result.dialogue_result is not None
    assert result.dialogue_result.turn is not None
    assert result.dialogue_result.turn.response is not None
    assert "Build real live JARVIS." in (
        result.dialogue_result.turn.response.text
    )


def test_live_session_runner_does_not_process_background_text() -> None:
    runner = _runner()
    runner.start()

    result = runner.ingest_text(
        text="background speech without wake word",
        speech_probability=0.95,
    )

    assert result.dialogue_result is None


def test_live_session_runner_handles_interruption() -> None:
    runner = _runner()
    runner.start()
    runner.ingest_text(text="Jarvis explain PID.")

    result = runner.handle_interrupt(text="wait", confidence=0.95)

    assert result.interruption_result is not None
    assert result.interruption_result.succeeded is True


def test_live_session_runner_health_check() -> None:
    runner = _runner()
    runner.start()

    result = runner.check_health()

    assert result.health_result is not None


def test_live_session_runner_recover_runs_without_speech() -> None:
    runner = _runner()
    runner.start()

    result = runner.recover()

    assert result.recovery_result is not None
    assert result.dialogue_result is None


def test_live_session_runner_shutdown_stops_state() -> None:
    runner = _runner()
    runner.start()

    result = runner.shutdown(reason=LiveShutdownReason.USER_REQUEST)

    assert result.status == LiveSessionRunnerStatus.STOPPED
    assert runner.live_state.state.conversation_active is False


def test_live_session_runner_snapshot() -> None:
    runner = _runner()
    runner.start()
    runner.ingest_text(text="Jarvis continue.")

    snapshot = runner.snapshot()

    assert snapshot.status in {
        LiveSessionRunnerStatus.RUNNING,
        LiveSessionRunnerStatus.DEGRADED,
    }
    assert snapshot.state_status


def test_live_session_runner_blocks_without_generator() -> None:
    runner = LiveSessionRunner(
        config=LiveSessionRunnerConfig(
            session_config=LiveSessionConfig(
                mode=LiveSessionMode.REAL_VOICE,
                real_microphone_enabled=True,
                real_stt_enabled=True,
                real_tts_enabled=True,
            ),
            auto_prepare_audio=False,
            auto_health_check=False,
        )
    )
    runner.start()

    result = runner.ingest_text(text="Jarvis respond.")

    assert result.status == LiveSessionRunnerStatus.DEGRADED
    assert result.dialogue_result is not None


def test_live_session_runner_enum_values_are_stable() -> None:
    assert LiveSessionRunnerStatus.RUNNING.value == "running"