from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path


def utc_now() -> datetime:
    return datetime.now(UTC)


class CodeNavigationStatus(StrEnum):
    READY = "ready"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    NOT_FOUND = "not_found"


class CodeSymbolKind(StrEnum):
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    IMPORT = "import"
    VARIABLE = "variable"
    UNKNOWN = "unknown"


class CodeNavigationQueryKind(StrEnum):
    FILE = "file"
    SYMBOL = "symbol"
    IMPORT = "import"
    REFERENCE = "reference"
    STRUCTURE = "structure"


@dataclass(frozen=True, slots=True)
class CodeNavigationRequest:
    project_root: Path
    query: str = ""
    query_kind: CodeNavigationQueryKind = CodeNavigationQueryKind.SYMBOL
    max_files: int = 500
    max_results: int = 25
    include_tests: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_files < 1:
            raise ValueError("max_files must be at least 1.")
        if self.max_results < 1:
            raise ValueError("max_results must be at least 1.")


@dataclass(frozen=True, slots=True)
class CodeSymbol:
    name: str
    kind: CodeSymbolKind
    file_path: str
    line: int | None
    column: int | None
    signature: str | None = None
    parent: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("code symbol name cannot be empty.")
        if not self.file_path.strip():
            raise ValueError("code symbol file_path cannot be empty.")


@dataclass(frozen=True, slots=True)
class CodeNavigationMatch:
    symbol: CodeSymbol | None
    file_path: str
    line: int | None
    column: int | None
    score: float
    reason: str
    excerpt: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.file_path.strip():
            raise ValueError("navigation match file_path cannot be empty.")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("navigation match score must be between 0 and 1.")
        if not self.reason.strip():
            raise ValueError("navigation match reason cannot be empty.")


@dataclass(frozen=True, slots=True)
class CodeNavigationIndex:
    project_root: str
    files: tuple[str, ...]
    symbols: tuple[CodeSymbol, ...]
    imports: tuple[CodeSymbol, ...]
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def symbol_count(self) -> int:
        return len(self.symbols)


@dataclass(frozen=True, slots=True)
class CodeNavigationReport:
    status: CodeNavigationStatus
    query: str
    query_kind: CodeNavigationQueryKind
    matches: tuple[CodeNavigationMatch, ...]
    index: CodeNavigationIndex | None
    reason: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def found(self) -> bool:
        return self.status == CodeNavigationStatus.READY and bool(self.matches)


class CodeNavigationEngine:
    """
    Step 47G Code Navigation Engine.

    Read-only codebase navigation:
    - indexes files
    - indexes Python classes/functions/methods/imports
    - finds files
    - finds symbols
    - finds imports
    - finds references
    - maps project structure

    It never mutates files.
    It never executes commands.
    It never opens IDE files automatically.
    """

    def build_index(
        self,
        request: CodeNavigationRequest,
    ) -> CodeNavigationReport:
        root = request.project_root.expanduser().resolve()

        if not root.exists():
            return _blocked(
                request=request,
                reason=f"project root does not exist: {root}",
            )

        if not root.is_dir():
            return _blocked(
                request=request,
                reason=f"project root is not a directory: {root}",
            )

        files = _collect_files(
            root=root,
            max_files=request.max_files,
            include_tests=request.include_tests,
        )
        symbols: list[CodeSymbol] = []
        imports: list[CodeSymbol] = []

        for path in files:
            file_symbols, file_imports = _symbols_for_file(root=root, path=path)
            symbols.extend(file_symbols)
            imports.extend(file_imports)

        index = CodeNavigationIndex(
            project_root=str(root),
            files=tuple(_relative(root=root, path=path) for path in files),
            symbols=tuple(symbols),
            imports=tuple(imports),
            created_at=utc_now(),
            metadata={
                **request.metadata,
                "max_files": request.max_files,
                "include_tests": request.include_tests,
            },
        )

        return CodeNavigationReport(
            status=(
                CodeNavigationStatus.READY
                if index.files
                else CodeNavigationStatus.PARTIAL
            ),
            query=request.query,
            query_kind=request.query_kind,
            matches=(),
            index=index,
            reason="code navigation index built",
            created_at=utc_now(),
            metadata=request.metadata,
        )

    def search(
        self,
        request: CodeNavigationRequest,
    ) -> CodeNavigationReport:
        index_report = self.build_index(request)

        if index_report.index is None:
            return index_report

        query = request.query.strip()
        if not query and request.query_kind != CodeNavigationQueryKind.STRUCTURE:
            return CodeNavigationReport(
                status=CodeNavigationStatus.NOT_FOUND,
                query=query,
                query_kind=request.query_kind,
                matches=(),
                index=index_report.index,
                reason="query is empty",
                created_at=utc_now(),
                metadata=request.metadata,
            )

        if request.query_kind == CodeNavigationQueryKind.FILE:
            matches = _find_files(index=index_report.index, query=query)
        elif request.query_kind == CodeNavigationQueryKind.IMPORT:
            matches = _find_imports(index=index_report.index, query=query)
        elif request.query_kind == CodeNavigationQueryKind.REFERENCE:
            matches = _find_references(
                root=Path(index_report.index.project_root),
                index=index_report.index,
                query=query,
            )
        elif request.query_kind == CodeNavigationQueryKind.STRUCTURE:
            matches = _structure_matches(index=index_report.index)
        else:
            matches = _find_symbols(index=index_report.index, query=query)

        limited = tuple(matches[: request.max_results])

        return CodeNavigationReport(
            status=(
                CodeNavigationStatus.READY
                if limited
                else CodeNavigationStatus.NOT_FOUND
            ),
            query=query,
            query_kind=request.query_kind,
            matches=limited,
            index=index_report.index,
            reason=(
                "navigation matches found"
                if limited
                else "no navigation matches found"
            ),
            created_at=utc_now(),
            metadata={
                **request.metadata,
                "match_count": len(limited),
            },
        )


def _collect_files(
    *,
    root: Path,
    max_files: int,
    include_tests: bool,
) -> tuple[Path, ...]:
    ignored_dirs = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        "dist",
        "build",
    }
    files: list[Path] = []

    for path in root.rglob("*"):
        if len(files) >= max_files:
            break

        relative_parts = path.relative_to(root).parts

        if any(part in ignored_dirs for part in relative_parts):
            continue

        if not path.is_file():
            continue

        if not include_tests and _is_test_path(path):
            continue

        if path.suffix.lower() not in {".py", ".md", ".txt", ".toml", ".json"}:
            continue

        files.append(path)

    files.sort(key=lambda item: str(item.relative_to(root)).lower())
    return tuple(files)


def _symbols_for_file(
    *,
    root: Path,
    path: Path,
) -> tuple[tuple[CodeSymbol, ...], tuple[CodeSymbol, ...]]:
    if path.suffix.lower() != ".py":
        return (), ()

    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return (), ()

    relative = _relative(root=root, path=path)
    symbols: list[CodeSymbol] = [
        CodeSymbol(
            name=Path(relative).stem,
            kind=CodeSymbolKind.MODULE,
            file_path=relative,
            line=1,
            column=0,
            signature=None,
            parent=None,
        )
    ]
    imports: list[CodeSymbol] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            symbols.append(
                CodeSymbol(
                    name=node.name,
                    kind=CodeSymbolKind.CLASS,
                    file_path=relative,
                    line=node.lineno,
                    column=node.col_offset,
                    signature=f"class {node.name}",
                    parent=None,
                )
            )
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.append(
                        CodeSymbol(
                            name=child.name,
                            kind=CodeSymbolKind.METHOD,
                            file_path=relative,
                            line=child.lineno,
                            column=child.col_offset,
                            signature=_function_signature(child),
                            parent=node.name,
                        )
                    )

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not _is_nested_function(tree=tree, target=node):
                symbols.append(
                    CodeSymbol(
                        name=node.name,
                        kind=CodeSymbolKind.FUNCTION,
                        file_path=relative,
                        line=node.lineno,
                        column=node.col_offset,
                        signature=_function_signature(node),
                        parent=None,
                    )
                )

        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name
                imports.append(
                    CodeSymbol(
                        name=name,
                        kind=CodeSymbolKind.IMPORT,
                        file_path=relative,
                        line=node.lineno,
                        column=node.col_offset,
                        signature=f"import {alias.name}",
                        parent=None,
                        metadata={"module": alias.name},
                    )
                )

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                name = alias.asname or alias.name
                imports.append(
                    CodeSymbol(
                        name=name,
                        kind=CodeSymbolKind.IMPORT,
                        file_path=relative,
                        line=node.lineno,
                        column=node.col_offset,
                        signature=f"from {module} import {alias.name}",
                        parent=None,
                        metadata={"module": module, "symbol": alias.name},
                    )
                )

    return tuple(_dedupe_symbols(symbols)), tuple(_dedupe_symbols(imports))


def _is_nested_function(
    *,
    tree: ast.AST,
    target: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    for node in ast.walk(tree):
        if node is target:
            continue

        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue

        for child in body:
            if child is target:
                return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))

    return False


def _function_signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    args = [arg.arg for arg in node.args.args]
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}({', '.join(args)})"


def _find_files(
    *,
    index: CodeNavigationIndex,
    query: str,
) -> list[CodeNavigationMatch]:
    q = query.lower()
    matches: list[CodeNavigationMatch] = []

    for file_path in index.files:
        score = _score_text(file_path, q)
        if score <= 0:
            continue

        matches.append(
            CodeNavigationMatch(
                symbol=None,
                file_path=file_path,
                line=None,
                column=None,
                score=score,
                reason="file path matched query",
                excerpt=file_path,
            )
        )

    return _sort_matches(matches)


def _find_symbols(
    *,
    index: CodeNavigationIndex,
    query: str,
) -> list[CodeNavigationMatch]:
    q = query.lower()
    matches: list[CodeNavigationMatch] = []

    for symbol in index.symbols:
        text = " ".join(
            part
            for part in (
                symbol.name,
                symbol.signature or "",
                symbol.file_path,
                symbol.parent or "",
            )
            if part
        )
        score = _score_text(text, q)
        if score <= 0:
            continue

        matches.append(
            CodeNavigationMatch(
                symbol=symbol,
                file_path=symbol.file_path,
                line=symbol.line,
                column=symbol.column,
                score=score,
                reason=f"{symbol.kind.value} matched query",
                excerpt=symbol.signature or symbol.name,
            )
        )

    return _sort_matches(matches)


def _find_imports(
    *,
    index: CodeNavigationIndex,
    query: str,
) -> list[CodeNavigationMatch]:
    q = query.lower()
    matches: list[CodeNavigationMatch] = []

    for symbol in index.imports:
        text = " ".join(
            str(value)
            for value in (
                symbol.name,
                symbol.signature or "",
                symbol.file_path,
                symbol.metadata.get("module", ""),
                symbol.metadata.get("symbol", ""),
            )
        )
        score = _score_text(text, q)
        if score <= 0:
            continue

        matches.append(
            CodeNavigationMatch(
                symbol=symbol,
                file_path=symbol.file_path,
                line=symbol.line,
                column=symbol.column,
                score=score,
                reason="import matched query",
                excerpt=symbol.signature or symbol.name,
            )
        )

    return _sort_matches(matches)


def _find_references(
    *,
    root: Path,
    index: CodeNavigationIndex,
    query: str,
) -> list[CodeNavigationMatch]:
    q = query.strip()
    if not q:
        return []

    pattern = re.compile(rf"\b{re.escape(q)}\b")
    matches: list[CodeNavigationMatch] = []

    for file_path in index.files:
        path = root / file_path
        if path.suffix.lower() != ".py":
            continue

        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue

        for line_index, line in enumerate(lines, start=1):
            if not pattern.search(line):
                continue

            matches.append(
                CodeNavigationMatch(
                    symbol=None,
                    file_path=file_path,
                    line=line_index,
                    column=line.find(q),
                    score=0.75,
                    reason="reference text matched query",
                    excerpt=line.strip(),
                )
            )

    return _sort_matches(matches)


def _structure_matches(
    *,
    index: CodeNavigationIndex,
) -> list[CodeNavigationMatch]:
    matches: list[CodeNavigationMatch] = []

    for file_path in index.files:
        score = 0.7
        if Path(file_path).name in {
            "pyproject.toml",
            "README.md",
            "__init__.py",
            "main.py",
            "bootstrap.py",
        }:
            score = 0.9

        matches.append(
            CodeNavigationMatch(
                symbol=None,
                file_path=file_path,
                line=None,
                column=None,
                score=score,
                reason="project structure file",
                excerpt=file_path,
            )
        )

    return _sort_matches(matches)


def _score_text(text: str, query: str) -> float:
    lowered = text.lower()

    if not query:
        return 0.0

    if lowered == query:
        return 1.0

    if Path(lowered).name == query:
        return 0.98

    if query in lowered:
        return 0.85

    query_parts = [part for part in re.split(r"[\W_]+", query) if part]
    if query_parts and all(part in lowered for part in query_parts):
        return 0.7

    return 0.0


def _dedupe_symbols(symbols: list[CodeSymbol]) -> tuple[CodeSymbol, ...]:
    seen: set[tuple[str, CodeSymbolKind, str, int | None]] = set()
    unique: list[CodeSymbol] = []

    for symbol in symbols:
        key = (symbol.name, symbol.kind, symbol.file_path, symbol.line)
        if key in seen:
            continue
        seen.add(key)
        unique.append(symbol)

    return tuple(unique)


def _sort_matches(
    matches: list[CodeNavigationMatch],
) -> list[CodeNavigationMatch]:
    return sorted(
        matches,
        key=lambda match: (
            -match.score,
            match.file_path,
            match.line or 0,
        ),
    )


def _is_test_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    return "tests" in parts or name.startswith("test_")


def _relative(
    *,
    root: Path,
    path: Path,
) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _blocked(
    *,
    request: CodeNavigationRequest,
    reason: str,
) -> CodeNavigationReport:
    return CodeNavigationReport(
        status=CodeNavigationStatus.BLOCKED,
        query=request.query,
        query_kind=request.query_kind,
        matches=(),
        index=None,
        reason=reason,
        created_at=utc_now(),
        metadata=request.metadata,
    )