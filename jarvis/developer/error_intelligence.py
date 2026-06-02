from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path


def utc_now() -> datetime:
    return datetime.now(UTC)


class ErrorIntelligenceStatus(StrEnum):
    ANALYZED = "analyzed"
    PARTIAL = "partial"
    NO_ERROR = "no_error"
    BLOCKED = "blocked"


class ErrorSourceKind(StrEnum):
    PYTEST = "pytest"
    MYPY = "mypy"
    RUFF = "ruff"
    PYTHON_TRACEBACK = "python_traceback"
    TERMINAL = "terminal"
    UNKNOWN = "unknown"


class ErrorSeverity(StrEnum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(StrEnum):
    TEST_FAILURE = "test_failure"
    TYPE_ERROR = "type_error"
    LINT_ERROR = "lint_error"
    IMPORT_ERROR = "import_error"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    ASSERTION_ERROR = "assertion_error"
    TIMEOUT = "timeout"
    CONFIGURATION_ERROR = "configuration_error"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ErrorIntelligenceRequest:
    stdout: str = ""
    stderr: str = ""
    source: ErrorSourceKind = ErrorSourceKind.UNKNOWN
    exit_code: int | None = None
    project_root: Path | None = None
    command: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.stdout and not self.stderr:
            raise ValueError("stdout or stderr is required for error analysis.")


@dataclass(frozen=True, slots=True)
class ErrorLocation:
    file_path: str | None
    line: int | None
    column: int | None
    symbol: str | None = None


@dataclass(frozen=True, slots=True)
class ErrorSignal:
    category: ErrorCategory
    severity: ErrorSeverity
    source: ErrorSourceKind
    message: str
    location: ErrorLocation | None
    evidence: str
    confidence: float
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise ValueError("error signal message cannot be empty.")
        if not self.evidence.strip():
            raise ValueError("error signal evidence cannot be empty.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("error signal confidence must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class ErrorDiagnosis:
    primary_category: ErrorCategory
    severity: ErrorSeverity
    title: str
    summary: str
    likely_cause: str
    next_action: str
    confidence: float
    evidence: tuple[str, ...]
    affected_files: tuple[str, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("diagnosis title cannot be empty.")
        if not self.summary.strip():
            raise ValueError("diagnosis summary cannot be empty.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("diagnosis confidence must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class ErrorIntelligenceReport:
    status: ErrorIntelligenceStatus
    source: ErrorSourceKind
    signals: tuple[ErrorSignal, ...]
    diagnosis: ErrorDiagnosis | None
    raw_excerpt: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        return self.status in {
            ErrorIntelligenceStatus.ANALYZED,
            ErrorIntelligenceStatus.PARTIAL,
        } and bool(self.signals)


class ErrorIntelligenceEngine:
    """
    Step 47C Error Intelligence Engine.

    Read-only diagnostic engine for developer workflows.

    It parses:
    - pytest failures
    - mypy errors
    - ruff errors
    - Python tracebacks
    - terminal/test output

    It never mutates files.
    It never executes commands.
    It never applies fixes.
    """

    def analyze(
        self,
        request: ErrorIntelligenceRequest,
    ) -> ErrorIntelligenceReport:
        text = _combined_output(request)

        if not text.strip():
            return ErrorIntelligenceReport(
                status=ErrorIntelligenceStatus.NO_ERROR,
                source=request.source,
                signals=(),
                diagnosis=None,
                raw_excerpt="",
                created_at=utc_now(),
                metadata=request.metadata,
            )

        inferred_source = (
            request.source
            if request.source != ErrorSourceKind.UNKNOWN
            else _infer_source(text)
        )

        signals = _parse_signals(
            text=text,
            source=inferred_source,
        )

        if not signals:
            if request.exit_code == 0:
                status = ErrorIntelligenceStatus.NO_ERROR
            else:
                status = ErrorIntelligenceStatus.PARTIAL

            diagnosis = (
                _diagnosis_from_unknown_output(
                    text=text,
                    source=inferred_source,
                    exit_code=request.exit_code,
                )
                if status == ErrorIntelligenceStatus.PARTIAL
                else None
            )

            return ErrorIntelligenceReport(
                status=status,
                source=inferred_source,
                signals=(),
                diagnosis=diagnosis,
                raw_excerpt=_excerpt(text),
                created_at=utc_now(),
                metadata={
                    **request.metadata,
                    "exit_code": request.exit_code,
                    "command": request.command,
                },
            )

        diagnosis = _diagnose(signals=signals, source=inferred_source)

        return ErrorIntelligenceReport(
            status=ErrorIntelligenceStatus.ANALYZED,
            source=inferred_source,
            signals=signals,
            diagnosis=diagnosis,
            raw_excerpt=_excerpt(text),
            created_at=utc_now(),
            metadata={
                **request.metadata,
                "exit_code": request.exit_code,
                "command": request.command,
                "signal_count": len(signals),
            },
        )


def _combined_output(request: ErrorIntelligenceRequest) -> str:
    chunks = []

    if request.stdout.strip():
        chunks.append(request.stdout)

    if request.stderr.strip():
        chunks.append(request.stderr)

    return "\n".join(chunks)


def _infer_source(text: str) -> ErrorSourceKind:
    lower = text.lower()

    if "traceback (most recent call last)" in lower:
        return ErrorSourceKind.PYTHON_TRACEBACK

    if "mypy" in lower or re.search(r":\d+: error:", text):
        return ErrorSourceKind.MYPY

    if "ruff" in lower or re.search(r"^[A-Z]\d{3}\s", text, re.MULTILINE):
        return ErrorSourceKind.RUFF

    if "pytest" in lower or "failed" in lower and "passed" in lower:
        return ErrorSourceKind.PYTEST

    return ErrorSourceKind.TERMINAL


def _parse_signals(
    *,
    text: str,
    source: ErrorSourceKind,
) -> tuple[ErrorSignal, ...]:
    if source == ErrorSourceKind.MYPY:
        return _parse_mypy(text)

    if source == ErrorSourceKind.RUFF:
        return _parse_ruff(text)

    if source == ErrorSourceKind.PYTHON_TRACEBACK:
        return _parse_traceback(text)

    if source == ErrorSourceKind.PYTEST:
        return _parse_pytest(text)

    return _parse_terminal(text)


def _parse_mypy(text: str) -> tuple[ErrorSignal, ...]:
    pattern = re.compile(
        r"^(?P<file>.*?):(?P<line>\d+):(?:(?P<column>\d+):)? "
        r"error: (?P<message>.*?)(?:\s+\[(?P<code>[^\]]+)\])?$",
        re.MULTILINE,
    )
    signals: list[ErrorSignal] = []

    for match in pattern.finditer(text):
        file_path = match.group("file")
        line = _to_int(match.group("line"))
        column = _to_int(match.group("column"))
        message = match.group("message").strip()
        code = match.group("code")

        signals.append(
            ErrorSignal(
                category=ErrorCategory.TYPE_ERROR,
                severity=ErrorSeverity.MEDIUM,
                source=ErrorSourceKind.MYPY,
                message=message,
                location=ErrorLocation(
                    file_path=file_path,
                    line=line,
                    column=column,
                ),
                evidence=match.group(0),
                confidence=0.95,
                metadata={"code": code or ""},
            )
        )

    return tuple(signals)


def _parse_ruff(text: str) -> tuple[ErrorSignal, ...]:
    signals: list[ErrorSignal] = []

    # Ruff common format:
    # E501 Line too long (92 > 88)
    #    --> path.py:10:89
    block_pattern = re.compile(
        r"(?P<code>[A-Z]\d{3}) (?P<message>.*?)\n\s+-->\s+"
        r"(?P<file>.*?):(?P<line>\d+):(?P<column>\d+)",
        re.DOTALL,
    )

    for match in block_pattern.finditer(text):
        message = f"{match.group('code')} {match.group('message').strip()}"
        signals.append(
            ErrorSignal(
                category=ErrorCategory.LINT_ERROR,
                severity=ErrorSeverity.LOW,
                source=ErrorSourceKind.RUFF,
                message=message,
                location=ErrorLocation(
                    file_path=match.group("file"),
                    line=_to_int(match.group("line")),
                    column=_to_int(match.group("column")),
                ),
                evidence=match.group(0).strip(),
                confidence=0.95,
                metadata={"code": match.group("code")},
            )
        )

    simple_pattern = re.compile(
        r"^(?P<file>.*?):(?P<line>\d+):(?P<column>\d+): "
        r"(?P<code>[A-Z]\d{3}) (?P<message>.*?)$",
        re.MULTILINE,
    )

    for match in simple_pattern.finditer(text):
        signals.append(
            ErrorSignal(
                category=ErrorCategory.LINT_ERROR,
                severity=ErrorSeverity.LOW,
                source=ErrorSourceKind.RUFF,
                message=(
                    f"{match.group('code')} {match.group('message').strip()}"
                ),
                location=ErrorLocation(
                    file_path=match.group("file"),
                    line=_to_int(match.group("line")),
                    column=_to_int(match.group("column")),
                ),
                evidence=match.group(0),
                confidence=0.9,
                metadata={"code": match.group("code")},
            )
        )

    return tuple(_dedupe_signals(signals))


def _parse_traceback(text: str) -> tuple[ErrorSignal, ...]:
    exception_line = _last_nonempty_line(text)
    category = _category_from_exception(exception_line)
    location = _traceback_location(text)

    return (
        ErrorSignal(
            category=category,
            severity=(
                ErrorSeverity.HIGH
                if category in {
                    ErrorCategory.IMPORT_ERROR,
                    ErrorCategory.SYNTAX_ERROR,
                    ErrorCategory.RUNTIME_ERROR,
                }
                else ErrorSeverity.MEDIUM
            ),
            source=ErrorSourceKind.PYTHON_TRACEBACK,
            message=exception_line or "Python traceback detected",
            location=location,
            evidence=_excerpt(text, max_chars=1200),
            confidence=0.9,
            metadata={},
        ),
    )


def _parse_pytest(text: str) -> tuple[ErrorSignal, ...]:
    signals: list[ErrorSignal] = []

    failure_header = re.compile(
        r"^_{5,}\s+(?P<name>.*?)\s+_{5,}$",
        re.MULTILINE,
    )
    failed_line = re.compile(
        r"^(?P<file>.*?):(?P<line>\d+): (?P<error>.*)$",
        re.MULTILINE,
    )

    for match in failure_header.finditer(text):
        test_name = match.group("name").strip()
        signals.append(
            ErrorSignal(
                category=ErrorCategory.TEST_FAILURE,
                severity=ErrorSeverity.MEDIUM,
                source=ErrorSourceKind.PYTEST,
                message=f"pytest failure: {test_name}",
                location=None,
                evidence=match.group(0),
                confidence=0.85,
                metadata={"test_name": test_name},
            )
        )

    for match in failed_line.finditer(text):
        error = match.group("error").strip()
        category = (
            ErrorCategory.ASSERTION_ERROR
            if "assert" in error.lower()
            else ErrorCategory.TEST_FAILURE
        )
        signals.append(
            ErrorSignal(
                category=category,
                severity=ErrorSeverity.MEDIUM,
                source=ErrorSourceKind.PYTEST,
                message=error,
                location=ErrorLocation(
                    file_path=match.group("file"),
                    line=_to_int(match.group("line")),
                    column=None,
                ),
                evidence=match.group(0),
                confidence=0.85,
                metadata={},
            )
        )

    if not signals and "failed" in text.lower():
        signals.append(
            ErrorSignal(
                category=ErrorCategory.TEST_FAILURE,
                severity=ErrorSeverity.MEDIUM,
                source=ErrorSourceKind.PYTEST,
                message="pytest reported failing tests",
                location=None,
                evidence=_excerpt(text),
                confidence=0.7,
                metadata={},
            )
        )

    return tuple(_dedupe_signals(signals))


def _parse_terminal(text: str) -> tuple[ErrorSignal, ...]:
    lower = text.lower()

    if "timed out" in lower or "timeout" in lower:
        return (
            ErrorSignal(
                category=ErrorCategory.TIMEOUT,
                severity=ErrorSeverity.MEDIUM,
                source=ErrorSourceKind.TERMINAL,
                message="terminal output indicates timeout",
                location=None,
                evidence=_excerpt(text),
                confidence=0.75,
                metadata={},
            ),
        )

    if "error" in lower or "failed" in lower:
        return (
            ErrorSignal(
                category=ErrorCategory.UNKNOWN,
                severity=ErrorSeverity.MEDIUM,
                source=ErrorSourceKind.TERMINAL,
                message="terminal output indicates an error",
                location=None,
                evidence=_excerpt(text),
                confidence=0.65,
                metadata={},
            ),
        )

    return ()


def _diagnose(
    *,
    signals: tuple[ErrorSignal, ...],
    source: ErrorSourceKind,
) -> ErrorDiagnosis:
    primary = _primary_signal(signals)
    affected_files = _affected_files(signals)
    evidence = tuple(signal.evidence for signal in signals[:5])

    return ErrorDiagnosis(
        primary_category=primary.category,
        severity=_max_severity(signals),
        title=_title(primary),
        summary=_summary(signals=signals, source=source),
        likely_cause=_likely_cause(primary),
        next_action=_next_action(primary),
        confidence=round(
            sum(signal.confidence for signal in signals) / len(signals),
            3,
        ),
        evidence=evidence,
        affected_files=affected_files,
        metadata={
            "signal_count": len(signals),
            "source": source.value,
        },
    )


def _diagnosis_from_unknown_output(
    *,
    text: str,
    source: ErrorSourceKind,
    exit_code: int | None,
) -> ErrorDiagnosis:
    return ErrorDiagnosis(
        primary_category=ErrorCategory.UNKNOWN,
        severity=ErrorSeverity.MEDIUM,
        title="Command failed but no structured error was detected",
        summary=(
            "The command returned a failing status, but the output did not "
             "match known parser formats."
        ),
        likely_cause=(
            "The failure may use an unsupported output format or incomplete "
            "logs."
        ),
        next_action=(
            "Inspect the raw excerpt and add a parser if this format appears "
            "often."
        ),
        confidence=0.45,
        evidence=(_excerpt(text),),
        affected_files=(),
        metadata={
            "source": source.value,
            "exit_code": exit_code,
        },
    )


def _primary_signal(signals: tuple[ErrorSignal, ...]) -> ErrorSignal:
    return max(
        signals,
        key=lambda signal: (
            _severity_rank(signal.severity),
            signal.confidence,
        ),
    )


def _max_severity(signals: tuple[ErrorSignal, ...]) -> ErrorSeverity:
    return max(signals, key=lambda signal: _severity_rank(signal.severity)).severity


def _severity_rank(severity: ErrorSeverity) -> int:
    ranks = {
        ErrorSeverity.INFO: 0,
        ErrorSeverity.LOW: 1,
        ErrorSeverity.MEDIUM: 2,
        ErrorSeverity.HIGH: 3,
        ErrorSeverity.CRITICAL: 4,
    }
    return ranks[severity]


def _title(signal: ErrorSignal) -> str:
    titles = {
        ErrorCategory.TEST_FAILURE: "Test failure detected",
        ErrorCategory.TYPE_ERROR: "Type checking error detected",
        ErrorCategory.LINT_ERROR: "Lint error detected",
        ErrorCategory.IMPORT_ERROR: "Import error detected",
        ErrorCategory.SYNTAX_ERROR: "Syntax error detected",
        ErrorCategory.RUNTIME_ERROR: "Runtime error detected",
        ErrorCategory.ASSERTION_ERROR: "Assertion failure detected",
        ErrorCategory.TIMEOUT: "Timeout detected",
        ErrorCategory.CONFIGURATION_ERROR: "Configuration error detected",
        ErrorCategory.UNKNOWN: "Unknown error detected",
    }
    return titles[signal.category]


def _summary(
    *,
    signals: tuple[ErrorSignal, ...],
    source: ErrorSourceKind,
) -> str:
    files = _affected_files(signals)
    file_text = (
        f" affecting {len(files)} file(s)"
        if files
        else " with no specific file location"
    )

    return (
        f"{len(signals)} {source.value} error signal(s) detected"
        f"{file_text}."
    )


def _likely_cause(signal: ErrorSignal) -> str:
    if signal.category == ErrorCategory.TYPE_ERROR:
        return (
            "A type annotation, function signature, or incompatible value "
            "likely violates the type contract."
        )

    if signal.category == ErrorCategory.LINT_ERROR:
        return "A formatting, style, or static lint rule was violated."

    if signal.category == ErrorCategory.TEST_FAILURE:
        return (
            "A test assertion or expected behavior no longer matches the "
            "implementation."
        )

    if signal.category == ErrorCategory.ASSERTION_ERROR:
        return (
            "An assertion failed because observed behavior differs from "
            "expected behavior."
        )

    if signal.category == ErrorCategory.IMPORT_ERROR:
        return "A module, symbol, or dependency could not be imported."

    if signal.category == ErrorCategory.SYNTAX_ERROR:
        return "Python could not parse the file due to invalid syntax."

    if signal.category == ErrorCategory.TIMEOUT:
        return "The operation exceeded its allowed time budget."

    if signal.category == ErrorCategory.RUNTIME_ERROR:
        return "The program raised an exception during execution."

    return "The output indicates failure, but the root cause needs more context."


def _next_action(signal: ErrorSignal) -> str:
    location = ""
    if signal.location is not None and signal.location.file_path is not None:
        location = f" Start at {signal.location.file_path}"
        if signal.location.line is not None:
            location += f":{signal.location.line}"
        location += "."

    if signal.category == ErrorCategory.TYPE_ERROR:
        return (
            "Inspect the reported type contract and align the value, "
            "annotation, or call signature."
            + location
        )

    if signal.category == ErrorCategory.LINT_ERROR:
        return "Apply the reported lint rule or formatting correction." + location

    if signal.category in {
        ErrorCategory.TEST_FAILURE,
        ErrorCategory.ASSERTION_ERROR,
    }:
        return (
            "Open the failing test, compare expected vs observed behavior, "
            "then inspect the implementation path."
            + location
        )

    if signal.category == ErrorCategory.IMPORT_ERROR:
        return (
            "Verify the import path, exported symbol, and dependency "
            "availability."
            + location
        )

    if signal.category == ErrorCategory.SYNTAX_ERROR:
        return (
            "Fix the syntax at the reported location before running deeper "
            "checks."
            + location
        )

    if signal.category == ErrorCategory.TIMEOUT:
        return (
            "Check for blocking loops, slow external calls, or missing "
            "cancellation boundaries."
        )

    return "Inspect the raw output and nearest changed code before applying a fix."


def _affected_files(signals: tuple[ErrorSignal, ...]) -> tuple[str, ...]:
    files: list[str] = []

    for signal in signals:
        if (
            signal.location is not None
            and signal.location.file_path is not None
            and signal.location.file_path not in files
        ):
            files.append(signal.location.file_path)

    return tuple(files)


def _traceback_location(text: str) -> ErrorLocation | None:
    frame_pattern = re.compile(
        r'File "(?P<file>.*?)", line (?P<line>\d+), in (?P<symbol>.*)'
    )
    matches = list(frame_pattern.finditer(text))

    if not matches:
        return None

    match = matches[-1]
    return ErrorLocation(
        file_path=match.group("file"),
        line=_to_int(match.group("line")),
        column=None,
        symbol=match.group("symbol").strip(),
    )


def _category_from_exception(exception_line: str) -> ErrorCategory:
    if exception_line.startswith("ImportError") or exception_line.startswith(
        "ModuleNotFoundError"
    ):
        return ErrorCategory.IMPORT_ERROR

    if exception_line.startswith("SyntaxError"):
        return ErrorCategory.SYNTAX_ERROR

    if exception_line.startswith("AssertionError"):
        return ErrorCategory.ASSERTION_ERROR

    if exception_line:
        return ErrorCategory.RUNTIME_ERROR

    return ErrorCategory.UNKNOWN


def _dedupe_signals(signals: list[ErrorSignal]) -> tuple[ErrorSignal, ...]:
    seen: set[tuple[ErrorCategory, str, str | None, int | None]] = set()
    unique: list[ErrorSignal] = []

    for signal in signals:
        key = (
            signal.category,
            signal.message,
            signal.location.file_path if signal.location else None,
            signal.location.line if signal.location else None,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(signal)

    return tuple(unique)


def _excerpt(text: str, *, max_chars: int = 2000) -> str:
    cleaned = text.strip()

    if len(cleaned) <= max_chars:
        return cleaned

    return cleaned[:max_chars] + "\n...[truncated]"


def _last_nonempty_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if not lines:
        return ""

    return lines[-1]


def _to_int(value: str | None) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except ValueError:
        return None