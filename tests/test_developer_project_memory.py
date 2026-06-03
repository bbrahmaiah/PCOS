from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.developer import (
    CodeContextEngine,
    CodeContextRequest,
    ErrorIntelligenceEngine,
    ErrorIntelligenceRequest,
    ErrorSourceKind,
    FixSuggestionEngine,
    FixSuggestionRequest,
    JsonProjectMemoryStore,
    ProjectMemoryEngine,
    ProjectMemoryImportance,
    ProjectMemoryKind,
    ProjectMemoryRecallRequest,
    ProjectMemoryStatus,
    ProjectMemoryStoreRequest,
)
from jarvis.developer import (
    TestCommandKind as CommandKind,
)
from jarvis.developer import (
    TestCommandPlan as CommandPlan,
)
from jarvis.developer import (
    TestCommandSafety as CommandSafety,
)
from jarvis.developer import (
    TestRunResult as RunResult,
)
from jarvis.developer import (
    TestRunStatus as RunStatus,
)


def test_project_memory_store_request_rejects_empty_title(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        ProjectMemoryStoreRequest(
            project_root=tmp_path,
            kind=ProjectMemoryKind.NOTE,
            title=" ",
            summary="summary",
        )


def test_project_memory_blocks_missing_root(tmp_path: Path) -> None:
    result = ProjectMemoryEngine().snapshot(tmp_path / "missing")

    assert result.status == ProjectMemoryStatus.BLOCKED
    assert result.succeeded is False


def test_project_memory_stores_and_recalls_note(tmp_path: Path) -> None:
    engine = ProjectMemoryEngine()

    stored = engine.store(
        ProjectMemoryStoreRequest(
            project_root=tmp_path,
            kind=ProjectMemoryKind.NOTE,
            title="Current task",
            summary="Build project memory engine",
            details="Remember active development progress.",
            importance=ProjectMemoryImportance.HIGH,
            tags=("Memory", "Developer"),
        )
    )

    assert stored.status == ProjectMemoryStatus.STORED
    assert stored.record is not None
    assert stored.record.tags == ("memory", "developer")
    assert stored.snapshot is not None
    assert stored.snapshot.record_count == 1

    recalled = engine.recall(
        ProjectMemoryRecallRequest(
            project_root=tmp_path,
            query="project memory",
        )
    )

    assert recalled.status == ProjectMemoryStatus.LOADED
    assert len(recalled.records) == 1
    assert recalled.records[0].title == "Current task"


def test_project_memory_persists_to_json(tmp_path: Path) -> None:
    first = ProjectMemoryEngine()
    first.store(
        ProjectMemoryStoreRequest(
            project_root=tmp_path,
            kind=ProjectMemoryKind.TODO,
            title="Next step",
            summary="Build build watcher",
            tags=("todo",),
        )
    )

    second = ProjectMemoryEngine()
    recalled = second.recall(
        ProjectMemoryRecallRequest(
            project_root=tmp_path,
            tags=("todo",),
        )
    )

    assert len(recalled.records) == 1
    assert recalled.records[0].summary == "Build build watcher"

    memory_path = JsonProjectMemoryStore().memory_path(tmp_path)

    assert memory_path.exists()
    assert ".jarvis" in str(memory_path)


def test_project_memory_clear_removes_json_file(tmp_path: Path) -> None:
    engine = ProjectMemoryEngine()
    engine.store(
        ProjectMemoryStoreRequest(
            project_root=tmp_path,
            kind=ProjectMemoryKind.NOTE,
            title="Temporary",
            summary="Temporary memory",
        )
    )

    memory_path = JsonProjectMemoryStore().memory_path(tmp_path)

    assert memory_path.exists()

    result = engine.clear(tmp_path)

    assert result.status == ProjectMemoryStatus.CLEARED
    assert not memory_path.exists()


def test_project_memory_remembers_code_context(tmp_path: Path) -> None:
    _write(tmp_path / "pyproject.toml", "[project]\n")
    _write(tmp_path / "jarvis" / "core.py", "x = 1\n")
    _write(tmp_path / "tests" / "test_core.py", "def test_core(): pass\n")

    context = CodeContextEngine().build_context(
        CodeContextRequest(
            project_root=tmp_path,
            active_file=tmp_path / "jarvis" / "core.py",
        )
    )
    result = ProjectMemoryEngine().remember_code_context(
        project_root=tmp_path,
        code_context=context,
    )

    assert result.status == ProjectMemoryStatus.STORED
    assert result.record is not None
    assert result.record.kind == ProjectMemoryKind.CODE_CONTEXT
    assert result.snapshot is not None
    assert result.snapshot.active_files == (str(Path("jarvis") / "core.py"),)


def test_project_memory_remembers_test_result(tmp_path: Path) -> None:
    result = ProjectMemoryEngine().remember_test_result(
        project_root=tmp_path,
        test_result=_test_result(status=RunStatus.FAILED),
    )

    assert result.status == ProjectMemoryStatus.STORED
    assert result.record is not None
    assert result.record.kind == ProjectMemoryKind.TEST_RESULT
    assert result.record.importance == ProjectMemoryImportance.HIGH


def test_project_memory_remembers_error_report(tmp_path: Path) -> None:
    error_report = ErrorIntelligenceEngine().analyze(
        ErrorIntelligenceRequest(
            stdout="jarvis/core.py:1: error: bad type [assignment]\n",
            source=ErrorSourceKind.MYPY,
            exit_code=1,
        )
    )

    result = ProjectMemoryEngine().remember_error_report(
        project_root=tmp_path,
        error_report=error_report,
    )

    assert result.status == ProjectMemoryStatus.STORED
    assert result.record is not None
    assert result.record.kind == ProjectMemoryKind.ERROR_DIAGNOSIS
    assert result.snapshot is not None
    assert result.snapshot.recent_errors


def test_project_memory_remembers_fix_report(tmp_path: Path) -> None:
    error_report = ErrorIntelligenceEngine().analyze(
        ErrorIntelligenceRequest(
            stdout="jarvis/core.py:1: error: bad type [assignment]\n",
            source=ErrorSourceKind.MYPY,
            exit_code=1,
        )
    )
    fix_report = FixSuggestionEngine().suggest(
        FixSuggestionRequest(error_report=error_report)
    )

    result = ProjectMemoryEngine().remember_fix_report(
        project_root=tmp_path,
        fix_report=fix_report,
    )

    assert result.status == ProjectMemoryStatus.STORED
    assert result.record is not None
    assert result.record.kind == ProjectMemoryKind.FIX_SUGGESTION
    assert result.snapshot is not None
    assert result.snapshot.recent_fixes


def test_project_memory_recall_filters_by_kind_and_limit(
    tmp_path: Path,
) -> None:
    engine = ProjectMemoryEngine()

    for index in range(5):
        engine.store(
            ProjectMemoryStoreRequest(
                project_root=tmp_path,
                kind=ProjectMemoryKind.NOTE,
                title=f"Note {index}",
                summary=f"Summary {index}",
            )
        )

    recalled = engine.recall(
        ProjectMemoryRecallRequest(
            project_root=tmp_path,
            kinds=(ProjectMemoryKind.NOTE,),
            limit=2,
        )
    )

    assert len(recalled.records) == 2
    assert all(record.kind == ProjectMemoryKind.NOTE for record in recalled.records)


def test_project_memory_enum_values_are_stable() -> None:
    assert ProjectMemoryStatus.STORED.value == "stored"
    assert ProjectMemoryKind.ERROR_DIAGNOSIS.value == "error_diagnosis"
    assert ProjectMemoryImportance.HIGH.value == "high"


def _test_result(*, status: RunStatus) -> RunResult:
    return RunResult(
        status=status,
        project_root="project",
        plan=CommandPlan(
            command=("python", "-m", "pytest"),
            kind=CommandKind.PYTHON_MODULE_PYTEST,
            safety=CommandSafety.SAFE,
            reason="test",
        ),
        exit_code=1 if status == RunStatus.FAILED else 0,
        stdout="",
        stderr="failed" if status == RunStatus.FAILED else "",
        duration_ms=10.0,
        summary="Tests failed." if status == RunStatus.FAILED else "Tests passed.",
        created_at=ProjectMemoryEngine().store(
            ProjectMemoryStoreRequest(
                project_root=Path.cwd(),
                kind=ProjectMemoryKind.NOTE,
                title="clock",
                summary="clock",
            )
        ).created_at,
    )


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")