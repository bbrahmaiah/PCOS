from __future__ import annotations

from pathlib import Path

from jarvis.cognition import (
    Phase3CompletionGate,
    Phase3CompletionVoiceIO,
    VoiceCognitionIO,
    complete_phase3_cognition,
)


def test_phase3_completion_gate_passes() -> None:
    report = complete_phase3_cognition(project_root=Path.cwd())

    assert report.passed is True
    assert report.failed_count == 0
    assert report.passed_count == report.total_count
    assert report.total_count >= 8


def test_phase3_completion_check_names_are_stable() -> None:
    report = Phase3CompletionGate(project_root=Path.cwd()).complete()
    check_names = {check.name for check in report.checks}

    assert "phase3_cognition_validation_passed" in check_names
    assert "cognition_runtime_text_turn_passed" in check_names
    assert "cognition_runtime_streaming_turn_passed" in check_names
    assert "voice_cognition_smoke_path_passed" in check_names
    assert "tool_planning_is_plan_only" in check_names
    assert "ollama_backend_boundary_available" in check_names
    assert "real_local_llm_smoke_script_exists" in check_names
    assert "real_voice_cognition_smoke_script_exists" in check_names


def test_phase3_completion_voice_io_satisfies_protocol() -> None:
    voice_io: VoiceCognitionIO = Phase3CompletionVoiceIO(
        transcript="Hello Jarvis.",
    )

    transcript = voice_io.listen_once()
    playback = voice_io.speak("Yes sir.")

    assert transcript.text == "Hello Jarvis."
    assert playback.completed is True


def test_phase3_completion_gate_stores_last_report() -> None:
    gate = Phase3CompletionGate(project_root=Path.cwd())

    assert gate.last_report() is None

    report = gate.complete()

    assert gate.last_report() == report