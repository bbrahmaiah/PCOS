from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.developer import (
    CodeNavigationEngine,
    CodeNavigationQueryKind,
    CodeNavigationRequest,
    CodeNavigationStatus,
    CodeSymbolKind,
)


def test_code_navigation_request_rejects_invalid_limits(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        CodeNavigationRequest(project_root=tmp_path, max_files=0)

    with pytest.raises(ValueError):
        CodeNavigationRequest(project_root=tmp_path, max_results=0)


def test_code_navigation_blocks_missing_project_root(
    tmp_path: Path,
) -> None:
    report = CodeNavigationEngine().search(
        CodeNavigationRequest(
            project_root=tmp_path / "missing",
            query="run",
        )
    )

    assert report.status == CodeNavigationStatus.BLOCKED
    assert report.index is None


def test_code_navigation_builds_python_symbol_index(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "jarvis" / "core.py",
        """
import os
from pathlib import Path

class RuntimeKernel:
    def start(self):
        return None

def build_system(config):
    return RuntimeKernel()
""",
    )

    report = CodeNavigationEngine().build_index(
        CodeNavigationRequest(project_root=tmp_path)
    )

    assert report.status == CodeNavigationStatus.READY
    assert report.index is not None
    assert report.index.file_count == 1

    symbols = {(symbol.name, symbol.kind) for symbol in report.index.symbols}
    imports = {(symbol.name, symbol.kind) for symbol in report.index.imports}

    assert ("RuntimeKernel", CodeSymbolKind.CLASS) in symbols
    assert ("start", CodeSymbolKind.METHOD) in symbols
    assert ("build_system", CodeSymbolKind.FUNCTION) in symbols
    assert ("os", CodeSymbolKind.IMPORT) in imports
    assert ("Path", CodeSymbolKind.IMPORT) in imports


def test_code_navigation_finds_symbol(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "jarvis" / "assembly.py",
        """
class JarvisSystem:
    def start(self):
        return None
""",
    )

    report = CodeNavigationEngine().search(
        CodeNavigationRequest(
            project_root=tmp_path,
            query="JarvisSystem",
            query_kind=CodeNavigationQueryKind.SYMBOL,
        )
    )

    assert report.status == CodeNavigationStatus.READY
    assert report.found is True
    assert report.matches[0].symbol is not None
    assert report.matches[0].symbol.name == "JarvisSystem"
    assert report.matches[0].line == 1


def test_code_navigation_finds_file(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "jarvis" / "developer" / "build_watch.py", "x = 1\n")
    _write(tmp_path / "README.md", "# Demo\n")

    report = CodeNavigationEngine().search(
        CodeNavigationRequest(
            project_root=tmp_path,
            query="build_watch",
            query_kind=CodeNavigationQueryKind.FILE,
        )
    )

    assert report.status == CodeNavigationStatus.READY
    assert report.matches[0].file_path == str(
        Path("jarvis") / "developer" / "build_watch.py"
    )


def test_code_navigation_finds_import(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "jarvis" / "core.py",
        "from pathlib import Path\nimport json\n",
    )

    report = CodeNavigationEngine().search(
        CodeNavigationRequest(
            project_root=tmp_path,
            query="pathlib",
            query_kind=CodeNavigationQueryKind.IMPORT,
        )
    )

    assert report.status == CodeNavigationStatus.READY
    assert report.matches[0].symbol is not None
    assert report.matches[0].symbol.kind == CodeSymbolKind.IMPORT


def test_code_navigation_finds_references(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "jarvis" / "core.py",
        """
def build_system():
    return JarvisSystem()

class JarvisSystem:
    pass
""",
    )

    report = CodeNavigationEngine().search(
        CodeNavigationRequest(
            project_root=tmp_path,
            query="JarvisSystem",
            query_kind=CodeNavigationQueryKind.REFERENCE,
        )
    )

    assert report.status == CodeNavigationStatus.READY
    assert len(report.matches) >= 2
    expected_path = str(Path("jarvis") / "core.py")

    assert all(
        match.file_path == expected_path
        for match in report.matches
    )


def test_code_navigation_structure_map_prioritizes_important_files(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "pyproject.toml", "[project]\n")
    _write(tmp_path / "README.md", "# Demo\n")
    _write(tmp_path / "jarvis" / "__init__.py", "")

    report = CodeNavigationEngine().search(
        CodeNavigationRequest(
            project_root=tmp_path,
            query_kind=CodeNavigationQueryKind.STRUCTURE,
        )
    )

    assert report.status == CodeNavigationStatus.READY
    assert report.matches[0].score >= 0.9


def test_code_navigation_can_exclude_tests(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "tests" / "test_core.py", "def test_core(): pass\n")
    _write(tmp_path / "jarvis" / "core.py", "def run(): pass\n")

    report = CodeNavigationEngine().build_index(
        CodeNavigationRequest(
            project_root=tmp_path,
            include_tests=False,
        )
    )

    assert report.index is not None
    assert str(Path("tests") / "test_core.py") not in report.index.files
    assert str(Path("jarvis") / "core.py") in report.index.files


def test_code_navigation_empty_query_returns_not_found(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "jarvis" / "core.py", "def run(): pass\n")

    report = CodeNavigationEngine().search(
        CodeNavigationRequest(project_root=tmp_path, query="")
    )

    assert report.status == CodeNavigationStatus.NOT_FOUND


def test_code_navigation_does_not_mutate_files(
    tmp_path: Path,
) -> None:
    target = tmp_path / "jarvis" / "core.py"
    original = "def run():\n    return 1\n"
    _write(target, original)

    CodeNavigationEngine().search(
        CodeNavigationRequest(
            project_root=tmp_path,
            query="run",
            query_kind=CodeNavigationQueryKind.SYMBOL,
        )
    )

    assert target.read_text(encoding="utf-8") == original


def test_code_navigation_enum_values_are_stable() -> None:
    assert CodeNavigationStatus.READY.value == "ready"
    assert CodeNavigationQueryKind.SYMBOL.value == "symbol"
    assert CodeSymbolKind.CLASS.value == "class"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.lstrip(), encoding="utf-8")