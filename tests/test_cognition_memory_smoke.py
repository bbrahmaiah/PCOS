from __future__ import annotations

from pathlib import Path

from scripts.smoke_cognition_memory import (
    CognitionMemorySmokeConfig,
    main,
    run_cognition_memory_smoke,
)


def test_cognition_memory_smoke_passes_with_in_memory_backend() -> None:
    report = run_cognition_memory_smoke(
        CognitionMemorySmokeConfig(
            prompt="What should JARVIS remember about memory gateway?",
            backend="in-memory",
            validate_phase4=False,
        )
    )

    assert report.passed is True
    assert report.retrieval_results > 0
    assert report.memory_context_items > 0
    assert report.captured_response is True
    assert "Memory is active" in report.response_text


def test_cognition_memory_smoke_passes_with_sqlite_backend(tmp_path: Path) -> None:
    report = run_cognition_memory_smoke(
        CognitionMemorySmokeConfig(
            prompt="What should JARVIS remember about cognition OS memory?",
            backend="sqlite",
            sqlite_path=tmp_path / "memory.db",
            validate_phase4=False,
        )
    )

    assert report.passed is True
    assert report.backend == "sqlite"
    assert report.retrieval_results > 0
    assert report.memory_context_items > 0


def test_cognition_memory_smoke_can_disable_summary() -> None:
    report = run_cognition_memory_smoke(
        CognitionMemorySmokeConfig(
            prompt="memory gateway",
            include_summary=False,
            validate_phase4=False,
        )
    )

    assert report.passed is True
    assert report.memory_context_items > 0


def test_cognition_memory_smoke_can_disable_capture() -> None:
    report = run_cognition_memory_smoke(
        CognitionMemorySmokeConfig(
            prompt="memory gateway",
            capture_response=False,
            validate_phase4=False,
        )
    )

    assert report.passed is True
    assert report.captured_response is False


def test_cognition_memory_smoke_can_run_phase4_validation(
    tmp_path: Path,
) -> None:
    report = run_cognition_memory_smoke(
        CognitionMemorySmokeConfig(
            prompt="memory gateway",
            backend="sqlite",
            sqlite_path=tmp_path / "memory.db",
            validate_phase4=True,
        )
    )

    assert report.passed is True
    assert report.phase4_validation_status is not None


def test_cognition_memory_smoke_cli_passes() -> None:
    exit_code = main(
        [
            "--prompt",
            "memory gateway",
            "--backend",
            "in-memory",
            "--no-validation",
        ]
    )

    assert exit_code == 0