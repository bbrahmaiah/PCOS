from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path


def utc_now() -> datetime:
    return datetime.now(UTC)


class CodeContextStatus(StrEnum):
    READY = "ready"
    PARTIAL = "partial"
    BLOCKED = "blocked"


class CodeLanguage(StrEnum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    MARKDOWN = "markdown"
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    TEXT = "text"
    UNKNOWN = "unknown"


class CodeContextSignalKind(StrEnum):
    PROJECT_ROOT = "project_root"
    ACTIVE_FILE = "active_file"
    RECENT_FILE = "recent_file"
    TEST_FILE = "test_file"
    CONFIG_FILE = "config_file"
    SOURCE_FILE = "source_file"
    LOG_FILE = "log_file"
    DOCUMENTATION_FILE = "documentation_file"


class CodeContextRisk(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class CodeFileSignal:
    path: str
    kind: CodeContextSignalKind
    language: CodeLanguage
    size_bytes: int
    modified_at: datetime | None
    reason: str
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.path.strip():
            raise ValueError("code file signal path cannot be empty.")
        if not self.reason.strip():
            raise ValueError("code file signal reason cannot be empty.")


@dataclass(frozen=True, slots=True)
class CodeContextRequest:
    project_root: Path
    active_file: Path | None = None
    max_files: int = 200
    include_hidden: bool = False
    include_logs: bool = True
    include_docs: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_files < 1:
            raise ValueError("max_files must be at least 1.")


@dataclass(frozen=True, slots=True)
class CodeContextSummary:
    project_name: str
    project_root: str
    primary_language: CodeLanguage
    file_count: int
    source_count: int
    test_count: int
    config_count: int
    documentation_count: int
    log_count: int
    active_file: str | None
    recent_files: tuple[str, ...]
    important_files: tuple[str, ...]
    risks: tuple[str, ...]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class CodeContextSnapshot:
    status: CodeContextStatus
    summary: CodeContextSummary | None
    signals: tuple[CodeFileSignal, ...]
    risk: CodeContextRisk
    reason: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == CodeContextStatus.READY


class CodeContextEngine:
    """
    Step 47A Code Context Engine.

    This is a read-only developer awareness subsystem.

    It observes the project tree and produces a structured coding context
    snapshot for future Error Intelligence, Test Runner, Fix Suggestions,
    and Project Memory.

    It never mutates files.
    It never runs shell commands.
    It never applies fixes.
    """

    def build_context(
        self,
        request: CodeContextRequest,
    ) -> CodeContextSnapshot:
        root = request.project_root.expanduser().resolve()

        if not root.exists():
            return _blocked(
                reason=f"project root does not exist: {root}",
                metadata=request.metadata,
            )

        if not root.is_dir():
            return _blocked(
                reason=f"project root is not a directory: {root}",
                metadata=request.metadata,
            )

        files = _collect_files(request=request, root=root)
        signals = _signals_for_files(
            root=root,
            files=files,
            request=request,
        )

        if request.active_file is not None:
            active_signal = _active_file_signal(
                root=root,
                active_file=request.active_file,
            )
            if active_signal is not None:
                signals = (active_signal, *signals)

        summary = _summary(
            root=root,
            request=request,
            signals=signals,
        )
        risk = _risk_for_summary(summary=summary, root=root)

        return CodeContextSnapshot(
            status=(
                CodeContextStatus.READY
                if signals
                else CodeContextStatus.PARTIAL
            ),
            summary=summary,
            signals=signals,
            risk=risk,
            reason="code context built successfully",
            created_at=utc_now(),
            metadata={
                **request.metadata,
                "max_files": request.max_files,
                "include_hidden": request.include_hidden,
                "include_logs": request.include_logs,
                "include_docs": request.include_docs,
            },
        )


def _collect_files(
    *,
    request: CodeContextRequest,
    root: Path,
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
        ".idea",
    }

    collected: list[Path] = []

    for path in root.rglob("*"):
        if len(collected) >= request.max_files:
            break

        if not request.include_hidden and any(
            part.startswith(".") for part in path.relative_to(root).parts
        ):
            continue

        if any(part in ignored_dirs for part in path.relative_to(root).parts):
            continue

        if not path.is_file():
            continue

        if not request.include_logs and _is_log_file(path):
            continue

        if not request.include_docs and _is_documentation_file(path):
            continue

        collected.append(path)

    collected.sort(key=lambda item: _mtime(item), reverse=True)
    return tuple(collected[: request.max_files])


def _signals_for_files(
    *,
    root: Path,
    files: tuple[Path, ...],
    request: CodeContextRequest,
) -> tuple[CodeFileSignal, ...]:
    signals: list[CodeFileSignal] = []

    for path in files:
        kind = _signal_kind(path)
        language = _language_for_path(path)

        if kind == CodeContextSignalKind.LOG_FILE and not request.include_logs:
            continue

        if (
            kind == CodeContextSignalKind.DOCUMENTATION_FILE
            and not request.include_docs
        ):
            continue

        signals.append(
            CodeFileSignal(
                path=_relative(root=root, path=path),
                kind=kind,
                language=language,
                size_bytes=_size(path),
                modified_at=_modified_at(path),
                reason=_reason_for_signal(kind=kind, path=path),
                metadata={
                    "suffix": path.suffix.lower(),
                    "name": path.name,
                },
            )
        )

    return tuple(signals)


def _active_file_signal(
    *,
    root: Path,
    active_file: Path,
) -> CodeFileSignal | None:
    path = active_file.expanduser().resolve()

    if not path.exists() or not path.is_file():
        return None

    try:
        path.relative_to(root)
    except ValueError:
        return None

    return CodeFileSignal(
        path=_relative(root=root, path=path),
        kind=CodeContextSignalKind.ACTIVE_FILE,
        language=_language_for_path(path),
        size_bytes=_size(path),
        modified_at=_modified_at(path),
        reason="active editor file",
        metadata={
            "suffix": path.suffix.lower(),
            "name": path.name,
        },
    )


def _summary(
    *,
    root: Path,
    request: CodeContextRequest,
    signals: tuple[CodeFileSignal, ...],
) -> CodeContextSummary:
    source = tuple(
        signal
        for signal in signals
        if signal.kind == CodeContextSignalKind.SOURCE_FILE
    )
    tests = tuple(
        signal
        for signal in signals
        if signal.kind == CodeContextSignalKind.TEST_FILE
    )
    configs = tuple(
        signal
        for signal in signals
        if signal.kind == CodeContextSignalKind.CONFIG_FILE
    )
    docs = tuple(
        signal
        for signal in signals
        if signal.kind == CodeContextSignalKind.DOCUMENTATION_FILE
    )
    logs = tuple(
        signal
        for signal in signals
        if signal.kind == CodeContextSignalKind.LOG_FILE
    )

    recent_files = tuple(signal.path for signal in signals[:10])
    important_files = _important_files(signals)
    active_file = (
        _relative(root=root, path=request.active_file.expanduser().resolve())
        if request.active_file is not None
        and request.active_file.expanduser().resolve().exists()
        else None
    )

    return CodeContextSummary(
        project_name=root.name,
        project_root=str(root),
        primary_language=_primary_language(signals),
        file_count=len(signals),
        source_count=len(source),
        test_count=len(tests),
        config_count=len(configs),
        documentation_count=len(docs),
        log_count=len(logs),
        active_file=active_file,
        recent_files=recent_files,
        important_files=important_files,
        risks=_summary_risks(
            test_count=len(tests),
            config_count=len(configs),
            source_count=len(source),
        ),
        created_at=utc_now(),
    )


def _important_files(
    signals: tuple[CodeFileSignal, ...],
) -> tuple[str, ...]:
    important_names = {
        "pyproject.toml",
        "requirements.txt",
        "README.md",
        "bootstrap.py",
        "main.py",
        "package.json",
        "tsconfig.json",
        "pytest.ini",
        ".env",
    }

    selected: list[str] = []

    for signal in signals:
        name = Path(signal.path).name
        if name in important_names:
            selected.append(signal.path)

    return tuple(selected[:20])


def _summary_risks(
    *,
    test_count: int,
    config_count: int,
    source_count: int,
) -> tuple[str, ...]:
    risks: list[str] = []

    if source_count > 0 and test_count == 0:
        risks.append("source files exist but no tests were detected")

    if config_count == 0:
        risks.append("no project config files were detected")

    return tuple(risks)


def _risk_for_summary(
    *,
    summary: CodeContextSummary,
    root: Path,
) -> CodeContextRisk:
    if ".venv" in {path.name for path in root.iterdir()}:
        return CodeContextRisk.LOW

    if summary.source_count > 0 and summary.test_count == 0:
        return CodeContextRisk.MEDIUM

    if summary.config_count == 0:
        return CodeContextRisk.LOW

    return CodeContextRisk.NONE


def _signal_kind(path: Path) -> CodeContextSignalKind:
    name = path.name.lower()
    suffix = path.suffix.lower()

    if _is_test_file(path):
        return CodeContextSignalKind.TEST_FILE

    if name in {
        "pyproject.toml",
        "requirements.txt",
        "package.json",
        "tsconfig.json",
        "pytest.ini",
        "mypy.ini",
        "ruff.toml",
        ".env",
        ".gitignore",
    }:
        return CodeContextSignalKind.CONFIG_FILE

    if _is_log_file(path):
        return CodeContextSignalKind.LOG_FILE

    if _is_documentation_file(path):
        return CodeContextSignalKind.DOCUMENTATION_FILE

    if suffix in {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".java",
        ".rs",
        ".go",
        ".cpp",
        ".c",
        ".h",
    }:
        return CodeContextSignalKind.SOURCE_FILE

    return CodeContextSignalKind.RECENT_FILE


def _language_for_path(path: Path) -> CodeLanguage:
    suffix = path.suffix.lower()

    if suffix == ".py":
        return CodeLanguage.PYTHON
    if suffix in {".js", ".jsx"}:
        return CodeLanguage.JAVASCRIPT
    if suffix in {".ts", ".tsx"}:
        return CodeLanguage.TYPESCRIPT
    if suffix in {".md", ".rst"}:
        return CodeLanguage.MARKDOWN
    if suffix == ".json":
        return CodeLanguage.JSON
    if suffix in {".yaml", ".yml"}:
        return CodeLanguage.YAML
    if suffix == ".toml":
        return CodeLanguage.TOML
    if suffix in {".txt", ".log"}:
        return CodeLanguage.TEXT

    return CodeLanguage.UNKNOWN


def _primary_language(
    signals: tuple[CodeFileSignal, ...],
) -> CodeLanguage:
    counts: dict[CodeLanguage, int] = {}

    for signal in signals:
        if signal.language == CodeLanguage.UNKNOWN:
            continue

        counts[signal.language] = counts.get(signal.language, 0) + 1

    if not counts:
        return CodeLanguage.UNKNOWN

    return max(counts.items(), key=lambda item: item[1])[0]


def _reason_for_signal(
    *,
    kind: CodeContextSignalKind,
    path: Path,
) -> str:
    if kind == CodeContextSignalKind.TEST_FILE:
        return "test file detected"
    if kind == CodeContextSignalKind.CONFIG_FILE:
        return "project configuration file detected"
    if kind == CodeContextSignalKind.LOG_FILE:
        return "log file detected"
    if kind == CodeContextSignalKind.DOCUMENTATION_FILE:
        return "documentation file detected"
    if kind == CodeContextSignalKind.SOURCE_FILE:
        return "source code file detected"

    return f"recent project file detected: {path.name}"


def _is_test_file(path: Path) -> bool:
    name = path.name.lower()
    parts = {part.lower() for part in path.parts}

    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or "tests" in parts
    )


def _is_log_file(path: Path) -> bool:
    return path.suffix.lower() == ".log" or "logs" in {
        part.lower() for part in path.parts
    }


def _is_documentation_file(path: Path) -> bool:
    return path.suffix.lower() in {".md", ".rst"} or path.name.lower() in {
        "readme",
        "readme.md",
    }


def _relative(
    *,
    root: Path,
    path: Path,
) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _modified_at(path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    except OSError:
        return None


def _blocked(
    *,
    reason: str,
    metadata: dict[str, object],
) -> CodeContextSnapshot:
    return CodeContextSnapshot(
        status=CodeContextStatus.BLOCKED,
        summary=None,
        signals=(),
        risk=CodeContextRisk.HIGH,
        reason=reason,
        created_at=utc_now(),
        metadata=metadata,
    )