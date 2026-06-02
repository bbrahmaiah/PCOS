from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from typing import Protocol


def utc_now() -> datetime:
    return datetime.now(UTC)


class TestDiscoveryStatus(StrEnum):
    READY = "ready"
    PARTIAL = "partial"
    BLOCKED = "blocked"


class TestRunStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"


class TestCommandKind(StrEnum):
    PYTEST = "pytest"
    PYTHON_MODULE_PYTEST = "python_module_pytest"
    NPM_TEST = "npm_test"
    UNKNOWN = "unknown"


class TestCommandSafety(StrEnum):
    SAFE = "safe"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class TestDiscoveryRequest:
    project_root: Path
    max_files: int = 300
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_files < 1:
            raise ValueError("max_files must be at least 1.")


@dataclass(frozen=True, slots=True)
class TestRunRequest:
    project_root: Path
    command: tuple[str, ...] | None = None
    timeout_seconds: float = 120.0
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")


@dataclass(frozen=True, slots=True)
class TestCommandPlan:
    command: tuple[str, ...]
    kind: TestCommandKind
    safety: TestCommandSafety
    reason: str
    blocked_reason: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def safe(self) -> bool:
        return self.safety == TestCommandSafety.SAFE


@dataclass(frozen=True, slots=True)
class TestDiscoveryReport:
    status: TestDiscoveryStatus
    project_root: str
    test_files: tuple[str, ...]
    config_files: tuple[str, ...]
    suggested_plan: TestCommandPlan | None
    reason: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == TestDiscoveryStatus.READY


@dataclass(frozen=True, slots=True)
class TestExecutionRequest:
    project_root: Path
    command: tuple[str, ...]
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class TestExecutionResult:
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    duration_ms: float


@dataclass(frozen=True, slots=True)
class TestRunResult:
    status: TestRunStatus
    project_root: str
    plan: TestCommandPlan
    exit_code: int | None
    stdout: str
    stderr: str
    duration_ms: float
    summary: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == TestRunStatus.PASSED


class TestExecutionAdapter(Protocol):
    def execute(
        self,
        request: TestExecutionRequest,
    ) -> TestExecutionResult:
        ...


class SubprocessTestExecutionAdapter:
    """
    Safe subprocess executor for test commands.

    This adapter never uses shell=True. The command is already validated by
    TestRunnerEngine before execution.
    """

    def execute(
        self,
        request: TestExecutionRequest,
    ) -> TestExecutionResult:
        started = perf_counter()

        try:
            completed = subprocess.run(
                request.command,
                cwd=request.project_root,
                capture_output=True,
                text=True,
                timeout=request.timeout_seconds,
                shell=False,
                check=False,
            )
            return TestExecutionResult(
                exit_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                timed_out=False,
                duration_ms=_elapsed_ms(started),
            )

        except subprocess.TimeoutExpired as exc:
            return TestExecutionResult(
                exit_code=None,
                stdout=exc.stdout if isinstance(exc.stdout, str) else "",
                stderr=exc.stderr if isinstance(exc.stderr, str) else "",
                timed_out=True,
                duration_ms=_elapsed_ms(started),
            )


class TestRunnerEngine:
    """
    Step 47B Test Runner Engine.

    Disciplined developer capability:
    - discovers test files/config
    - builds safe test command plans
    - blocks unsafe terminal commands
    - executes only allowlisted test commands
    - captures stdout/stderr/exit code/duration
    - summarizes result for JARVIS cognition/voice later

    It does not run arbitrary shell commands.
    It does not mutate files.
    It does not apply fixes.
    """

    def __init__(
        self,
        *,
        executor: TestExecutionAdapter | None = None,
    ) -> None:
        self._executor = executor or SubprocessTestExecutionAdapter()

    def discover(
        self,
        request: TestDiscoveryRequest,
    ) -> TestDiscoveryReport:
        root = request.project_root.expanduser().resolve()

        if not root.exists():
            return TestDiscoveryReport(
                status=TestDiscoveryStatus.BLOCKED,
                project_root=str(root),
                test_files=(),
                config_files=(),
                suggested_plan=None,
                reason=f"project root does not exist: {root}",
                created_at=utc_now(),
                metadata=request.metadata,
            )

        if not root.is_dir():
            return TestDiscoveryReport(
                status=TestDiscoveryStatus.BLOCKED,
                project_root=str(root),
                test_files=(),
                config_files=(),
                suggested_plan=None,
                reason=f"project root is not a directory: {root}",
                created_at=utc_now(),
                metadata=request.metadata,
            )

        files = _collect_files(root=root, max_files=request.max_files)
        test_files = tuple(
            _relative(root=root, path=path)
            for path in files
            if _is_test_file(path)
        )
        config_files = tuple(
            _relative(root=root, path=path)
            for path in files
            if _is_test_config(path)
        )
        suggested_plan = _suggest_plan(
            root=root,
            test_files=test_files,
            config_files=config_files,
        )

        status = (
            TestDiscoveryStatus.READY
            if suggested_plan is not None and test_files
            else TestDiscoveryStatus.PARTIAL
        )

        return TestDiscoveryReport(
            status=status,
            project_root=str(root),
            test_files=test_files,
            config_files=config_files,
            suggested_plan=suggested_plan,
            reason=(
                "test workflow discovered"
                if suggested_plan is not None
                else "no supported test workflow discovered"
            ),
            created_at=utc_now(),
            metadata={
                **request.metadata,
                "file_count": len(files),
            },
        )

    def plan(
        self,
        request: TestRunRequest,
    ) -> TestCommandPlan:
        root = request.project_root.expanduser().resolve()

        if request.command is not None:
            return _validate_command(command=request.command)

        discovery = self.discover(
            TestDiscoveryRequest(
                project_root=root,
                metadata=request.metadata,
            )
        )

        if discovery.suggested_plan is None:
            return TestCommandPlan(
                command=(),
                kind=TestCommandKind.UNKNOWN,
                safety=TestCommandSafety.BLOCKED,
                reason="no supported test command discovered",
                blocked_reason="no pytest/npm test workflow detected",
            )

        return discovery.suggested_plan

    def run(
        self,
        request: TestRunRequest,
    ) -> TestRunResult:
        root = request.project_root.expanduser().resolve()

        if not root.exists() or not root.is_dir():
            plan = TestCommandPlan(
                command=request.command or (),
                kind=TestCommandKind.UNKNOWN,
                safety=TestCommandSafety.BLOCKED,
                reason="project root invalid",
                blocked_reason=f"project root is not a directory: {root}",
            )
            return _blocked_result(
                root=root,
                plan=plan,
                metadata=request.metadata,
            )

        plan = self.plan(request)

        if not plan.safe:
            return _blocked_result(
                root=root,
                plan=plan,
                metadata=request.metadata,
            )

        execution = self._executor.execute(
            TestExecutionRequest(
                project_root=root,
                command=plan.command,
                timeout_seconds=request.timeout_seconds,
            )
        )

        if execution.timed_out:
            status = TestRunStatus.TIMEOUT
        elif execution.exit_code == 0:
            status = TestRunStatus.PASSED
        else:
            status = TestRunStatus.FAILED

        return TestRunResult(
            status=status,
            project_root=str(root),
            plan=plan,
            exit_code=execution.exit_code,
            stdout=execution.stdout,
            stderr=execution.stderr,
            duration_ms=execution.duration_ms,
            summary=_summary_for_execution(
                status=status,
                execution=execution,
            ),
            created_at=utc_now(),
            metadata=request.metadata,
        )


def _suggest_plan(
    *,
    root: Path,
    test_files: tuple[str, ...],
    config_files: tuple[str, ...],
) -> TestCommandPlan | None:
    package_json = root / "package.json"

    if package_json.exists():
        return _validate_command(command=("npm", "test"))

    if test_files or any(
        Path(config).name
        in {
            "pyproject.toml",
            "pytest.ini",
            "tox.ini",
            "setup.cfg",
        }
        for config in config_files
    ):
        return _validate_command(command=(sys.executable, "-m", "pytest"))

    return None


def _validate_command(command: tuple[str, ...]) -> TestCommandPlan:
    if not command:
        return TestCommandPlan(
            command=command,
            kind=TestCommandKind.UNKNOWN,
            safety=TestCommandSafety.BLOCKED,
            reason="empty command",
            blocked_reason="command cannot be empty",
        )

    unsafe_reason = _unsafe_command_reason(command)
    if unsafe_reason is not None:
        return TestCommandPlan(
            command=command,
            kind=TestCommandKind.UNKNOWN,
            safety=TestCommandSafety.BLOCKED,
            reason="unsafe command rejected",
            blocked_reason=unsafe_reason,
        )

    kind = _command_kind(command)

    if kind == TestCommandKind.UNKNOWN:
        return TestCommandPlan(
            command=command,
            kind=kind,
            safety=TestCommandSafety.BLOCKED,
            reason="command is not an allowlisted test command",
            blocked_reason="only pytest, python -m pytest, and npm test are allowed",
        )

    return TestCommandPlan(
        command=command,
        kind=kind,
        safety=TestCommandSafety.SAFE,
        reason="allowlisted test command",
        blocked_reason=None,
    )


def _command_kind(command: tuple[str, ...]) -> TestCommandKind:
    executable = Path(command[0]).name.lower()

    if executable in {"pytest", "pytest.exe"}:
        return TestCommandKind.PYTEST

    if (
        _is_python_executable(executable)
        and len(command) >= 3
        and command[1] == "-m"
        and command[2] == "pytest"
    ):
        return TestCommandKind.PYTHON_MODULE_PYTEST

    if executable in {"npm", "npm.cmd", "npm.exe"} and len(command) >= 2:
        if command[1] == "test":
            return TestCommandKind.NPM_TEST

    return TestCommandKind.UNKNOWN


def _unsafe_command_reason(command: tuple[str, ...]) -> str | None:
    dangerous_tokens = {
        "rm",
        "del",
        "erase",
        "format",
        "shutdown",
        "restart",
        "curl",
        "wget",
        "scp",
        "ssh",
        "powershell",
        "cmd",
        "bash",
    }
    shell_markers = {
        ";",
        "&&",
        "||",
        "|",
        ">",
        "<",
        "`",
        "$(",
    }

    lowered = tuple(part.lower() for part in command)

    for part in lowered:
        if part in dangerous_tokens:
            return f"dangerous command token blocked: {part}"

        if any(marker in part for marker in shell_markers):
            return f"shell control marker blocked: {part}"

    return None


def _is_python_executable(executable: str) -> bool:
    return executable in {
        "python",
        "python.exe",
        "python3",
        "python3.exe",
    } or executable.startswith("python")


def _collect_files(
    *,
    root: Path,
    max_files: int,
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

        files.append(path)

    files.sort(key=lambda item: _mtime(item), reverse=True)
    return tuple(files)


def _is_test_file(path: Path) -> bool:
    name = path.name.lower()
    parts = {part.lower() for part in path.parts}

    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or name.endswith(".test.ts")
        or name.endswith(".test.js")
        or "tests" in parts
    )


def _is_test_config(path: Path) -> bool:
    return path.name.lower() in {
        "pyproject.toml",
        "pytest.ini",
        "tox.ini",
        "setup.cfg",
        "package.json",
    }


def _summary_for_execution(
    *,
    status: TestRunStatus,
    execution: TestExecutionResult,
) -> str:
    if status == TestRunStatus.PASSED:
        return "Tests passed."

    if status == TestRunStatus.TIMEOUT:
        return "Test run timed out before completion."

    if status == TestRunStatus.FAILED:
        return "Tests failed. Review captured output for details."

    return "Test run did not complete successfully."


def _blocked_result(
    *,
    root: Path,
    plan: TestCommandPlan,
    metadata: dict[str, object],
) -> TestRunResult:
    return TestRunResult(
        status=TestRunStatus.BLOCKED,
        project_root=str(root),
        plan=plan,
        exit_code=None,
        stdout="",
        stderr="",
        duration_ms=0.0,
        summary=plan.blocked_reason or "test run blocked",
        created_at=utc_now(),
        metadata=metadata,
    )


def _relative(
    *,
    root: Path,
    path: Path,
) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000.0, 3)


