from __future__ import annotations

from pathlib import Path

from scripts.smoke_memory_search import (
    MemorySearchSmokeConfig,
    main,
    run_memory_search_smoke,
)


def test_memory_search_smoke_passes_with_in_memory_backend() -> None:
    report = run_memory_search_smoke(
        MemorySearchSmokeConfig(
            query_text="memory gateway",
            backend="in-memory",
        )
    )

    assert report.passed is True
    assert report.lexical_result_count > 0
    assert report.context_item_count > 0
    assert report.vector_enabled is True
    assert report.vector_result_count > 0


def test_memory_search_smoke_passes_with_sqlite_backend(tmp_path: Path) -> None:
    report = run_memory_search_smoke(
        MemorySearchSmokeConfig(
            query_text="cognition OS memory",
            backend="sqlite",
            sqlite_path=tmp_path / "memory.db",
        )
    )

    assert report.passed is True
    assert report.backend == "sqlite"
    assert report.lexical_result_count > 0
    assert report.context_item_count > 0


def test_memory_search_smoke_can_disable_vector() -> None:
    report = run_memory_search_smoke(
        MemorySearchSmokeConfig(
            query_text="memory context",
            backend="in-memory",
            include_vector=False,
        )
    )

    assert report.passed is True
    assert report.vector_enabled is False
    assert report.vector_result_count == 0


def test_memory_search_smoke_can_run_serial() -> None:
    report = run_memory_search_smoke(
        MemorySearchSmokeConfig(
            query_text="vector boundary",
            backend="in-memory",
            parallel=False,
        )
    )

    assert report.passed is True
    assert report.context_item_count > 0


def test_memory_search_smoke_cli_passes() -> None:
    exit_code = main(
        [
            "--backend",
            "in-memory",
            "--query",
            "memory gateway",
            "--no-vector",
            "--serial",
        ]
    )

    assert exit_code == 0