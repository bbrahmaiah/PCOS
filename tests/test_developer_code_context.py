from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.developer import (
    CodeContextEngine,
    CodeContextRequest,
    CodeContextRisk,
    CodeContextSignalKind,
    CodeContextStatus,
    CodeLanguage,
)


def test_code_context_request_rejects_invalid_max_files(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        CodeContextRequest(
            project_root=tmp_path,
            max_files=0,
        )


def test_code_context_blocks_missing_project_root(
    tmp_path: Path,
) -> None:
    engine = CodeContextEngine()
    snapshot = engine.build_context(
        CodeContextRequest(
            project_root=tmp_path / "missing",
        )
    )

    assert snapshot.status == CodeContextStatus.BLOCKED
    assert snapshot.summary is None
    assert snapshot.risk == CodeContextRisk.HIGH


def test_code_context_builds_python_project_snapshot(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "pyproject.toml", "[tool.pytest.ini_options]\n")
    _write(tmp_path / "README.md", "# Demo\n")
    _write(tmp_path / "jarvis" / "__init__.py", "")
    _write(tmp_path / "jarvis" / "core.py", "def run():\n    return 1\n")
    _write(tmp_path / "tests" / "test_core.py", "def test_core():\n    pass\n")
    _write(tmp_path / "logs" / "runtime.log", "ready\n")

    engine = CodeContextEngine()
    snapshot = engine.build_context(
        CodeContextRequest(
            project_root=tmp_path,
            active_file=tmp_path / "jarvis" / "core.py",
            max_files=50,
        )
    )

    assert snapshot.status == CodeContextStatus.READY
    assert snapshot.ready is True
    assert snapshot.summary is not None
    assert snapshot.summary.project_name == tmp_path.name
    assert snapshot.summary.primary_language == CodeLanguage.PYTHON
    assert snapshot.summary.source_count >= 1
    assert snapshot.summary.test_count == 1
    assert snapshot.summary.config_count >= 1
    assert snapshot.summary.documentation_count >= 1
    assert snapshot.summary.log_count >= 1
    assert snapshot.summary.active_file == str(Path("jarvis") / "core.py")
    assert snapshot.risk in {
        CodeContextRisk.NONE,
        CodeContextRisk.LOW,
    }

    kinds = {signal.kind for signal in snapshot.signals}

    assert CodeContextSignalKind.ACTIVE_FILE in kinds
    assert CodeContextSignalKind.SOURCE_FILE in kinds
    assert CodeContextSignalKind.TEST_FILE in kinds
    assert CodeContextSignalKind.CONFIG_FILE in kinds
    assert CodeContextSignalKind.DOCUMENTATION_FILE in kinds
    assert CodeContextSignalKind.LOG_FILE in kinds


def test_code_context_can_exclude_logs_and_docs(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "main.py", "print('hello')\n")
    _write(tmp_path / "README.md", "# Docs\n")
    _write(tmp_path / "logs" / "runtime.log", "ready\n")

    engine = CodeContextEngine()
    snapshot = engine.build_context(
        CodeContextRequest(
            project_root=tmp_path,
            include_logs=False,
            include_docs=False,
        )
    )

    kinds = {signal.kind for signal in snapshot.signals}

    assert CodeContextSignalKind.LOG_FILE not in kinds
    assert CodeContextSignalKind.DOCUMENTATION_FILE not in kinds


def test_code_context_detects_no_tests_risk(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "main.py", "print('hello')\n")
    _write(tmp_path / "pyproject.toml", "[project]\n")

    engine = CodeContextEngine()
    snapshot = engine.build_context(CodeContextRequest(project_root=tmp_path))

    assert snapshot.summary is not None
    assert "source files exist but no tests were detected" in (
        snapshot.summary.risks
    )
    assert snapshot.risk == CodeContextRisk.MEDIUM


def test_code_context_respects_max_files(
    tmp_path: Path,
) -> None:
    for index in range(20):
        _write(tmp_path / f"file_{index}.py", "x = 1\n")

    engine = CodeContextEngine()
    snapshot = engine.build_context(
        CodeContextRequest(
            project_root=tmp_path,
            max_files=5,
        )
    )

    assert len(snapshot.signals) == 5
    assert snapshot.summary is not None
    assert snapshot.summary.file_count == 5


def test_code_context_ignores_virtual_environment(
    tmp_path: Path,
) -> None:
    _write(tmp_path / ".venv" / "Lib" / "site.py", "ignored = True\n")
    _write(tmp_path / "main.py", "print('hello')\n")

    engine = CodeContextEngine()
    snapshot = engine.build_context(
        CodeContextRequest(
            project_root=tmp_path,
            include_hidden=True,
            max_files=20,
        )
    )

    paths = {signal.path for signal in snapshot.signals}

    assert "main.py" in paths
    assert not any(".venv" in path for path in paths)


def test_code_context_enum_values_are_stable() -> None:
    assert CodeContextStatus.READY.value == "ready"
    assert CodeLanguage.PYTHON.value == "python"
    assert CodeContextSignalKind.ACTIVE_FILE.value == "active_file"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")