from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from jarvis.developer import (
    BuildWatchDecision,
    BuildWatchEngine,
    BuildWatchEventKind,
    BuildWatchRequest,
    BuildWatchStatus,
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
    TestExecutionRequest as ExecutionRequest,
)
from jarvis.developer import (
    TestExecutionResult as ExecutionResult,
)
from jarvis.developer import (
    TestRunnerEngine as RunnerEngine,
)
from jarvis.developer import (
    TestRunStatus as RunStatus,
)


class FakeTestExecutionAdapter:
    def __init__(
        self,
        *,
        exit_code: int | None = 0,
        stdout: str = "1 passed",
        stderr: str = "",
        timed_out: bool = False,
    ) -> None:
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = timed_out
        self.requests: list[ExecutionRequest] = []

    def execute(
        self,
        request: ExecutionRequest,
    ) -> ExecutionResult:
        self.requests.append(request)
        return ExecutionResult(
            exit_code=self.exit_code,
            stdout=self.stdout,
            stderr=self.stderr,
            timed_out=self.timed_out,
            duration_ms=8.0,
        )


def test_build_watch_request_rejects_invalid_values(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        BuildWatchRequest(
            project_root=tmp_path,
            timeout_seconds=0,
        )

    with pytest.raises(ValueError):
        BuildWatchRequest(
            project_root=tmp_path,
            max_files=0,
        )


def test_build_watch_blocks_missing_project_root(
    tmp_path: Path,
) -> None:
    snapshot = BuildWatchEngine().check_once(
        BuildWatchRequest(project_root=tmp_path / "missing")
    )

    assert snapshot.status == BuildWatchStatus.BLOCKED
    assert snapshot.decision == BuildWatchDecision.BLOCKED
    assert snapshot.code_context is None


def test_build_watch_context_only_cycle(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "pyproject.toml", "[tool.pytest.ini_options]\n")
    _write(tmp_path / "tests" / "test_demo.py", "def test_demo(): pass\n")
    _write(tmp_path / "jarvis" / "core.py", "x = 1\n")

    snapshot = BuildWatchEngine().check_once(
        BuildWatchRequest(
            project_root=tmp_path,
            run_tests=False,
            remember=True,
        )
    )

    assert snapshot.status == BuildWatchStatus.HEALTHY
    assert snapshot.decision == BuildWatchDecision.REMEMBER_ONLY
    assert snapshot.code_context is not None
    assert snapshot.test_result is None
    assert snapshot.memory_results
    assert snapshot.fingerprint


def test_build_watch_passes_when_tests_pass(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "pyproject.toml", "[tool.pytest.ini_options]\n")
    _write(tmp_path / "tests" / "test_demo.py", "def test_demo(): pass\n")

    adapter = FakeTestExecutionAdapter(exit_code=0, stdout="1 passed")
    engine = BuildWatchEngine(
        test_runner=RunnerEngine(executor=adapter),
    )

    snapshot = engine.check_once(BuildWatchRequest(project_root=tmp_path))

    assert snapshot.status == BuildWatchStatus.HEALTHY
    assert snapshot.decision == BuildWatchDecision.RUN_TESTS
    assert snapshot.test_result is not None
    assert snapshot.test_result.status == RunStatus.PASSED
    assert snapshot.error_report is None
    assert snapshot.fix_report is None
    assert len(adapter.requests) == 1


def test_build_watch_analyzes_failure_and_suggests_fix(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "pyproject.toml", "[tool.pytest.ini_options]\n")
    _write(tmp_path / "tests" / "test_demo.py", "def test_demo(): pass\n")

    failing_output = """________________ test_example ________________

    def test_example():
>       assert 1 == 2
E       assert 1 == 2

tests/test_demo.py:3: AssertionError
"""

    adapter = FakeTestExecutionAdapter(
        exit_code=1,
        stdout=failing_output,
        stderr="",
    )
    engine = BuildWatchEngine(test_runner=RunnerEngine(executor=adapter))

    snapshot = engine.check_once(BuildWatchRequest(project_root=tmp_path))

    assert snapshot.status == BuildWatchStatus.FAILING
    assert snapshot.decision == BuildWatchDecision.SUGGEST_FIX
    assert snapshot.test_result is not None
    assert snapshot.error_report is not None
    assert snapshot.error_report.has_errors is True
    assert snapshot.fix_report is not None
    assert snapshot.fix_report.has_suggestions is True
    assert snapshot.memory_results

    event_kinds = {event.kind for event in snapshot.events}

    assert BuildWatchEventKind.CONTEXT_CAPTURED in event_kinds
    assert BuildWatchEventKind.TEST_RUN_COMPLETED in event_kinds
    assert BuildWatchEventKind.ERROR_ANALYZED in event_kinds
    assert BuildWatchEventKind.FIX_SUGGESTED in event_kinds
    assert BuildWatchEventKind.MEMORY_UPDATED in event_kinds


def test_build_watch_blocks_when_no_test_workflow(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "main.py", "print('hello')\n")

    adapter = FakeTestExecutionAdapter()
    engine = BuildWatchEngine(test_runner=RunnerEngine(executor=adapter))

    snapshot = engine.check_once(BuildWatchRequest(project_root=tmp_path))

    assert snapshot.status == BuildWatchStatus.BLOCKED
    assert snapshot.decision == BuildWatchDecision.BLOCKED
    assert snapshot.test_result is not None
    assert snapshot.test_result.status == RunStatus.BLOCKED
    assert len(adapter.requests) == 0


def test_build_watch_detects_unchanged_fingerprint(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "pyproject.toml", "[tool.pytest.ini_options]\n")
    _write(tmp_path / "tests" / "test_demo.py", "def test_demo(): pass\n")

    adapter = FakeTestExecutionAdapter(exit_code=0, stdout="1 passed")
    engine = BuildWatchEngine(test_runner=RunnerEngine(executor=adapter))

    first = engine.check_once(BuildWatchRequest(project_root=tmp_path))
    second = engine.check_once(
        BuildWatchRequest(
            project_root=tmp_path,
            previous_fingerprint=first.fingerprint,
        )
    )

    assert first.changed is True
    assert second.changed is False


def test_build_watch_does_not_mutate_source_files(
    tmp_path: Path,
) -> None:
    source = tmp_path / "jarvis" / "core.py"
    original = "x = 1\n"
    _write(source, original)
    _write(tmp_path / "pyproject.toml", "[tool.pytest.ini_options]\n")
    _write(tmp_path / "tests" / "test_demo.py", "def test_demo(): pass\n")

    adapter = FakeTestExecutionAdapter(exit_code=1, stdout="failed")
    engine = BuildWatchEngine(test_runner=RunnerEngine(executor=adapter))

    snapshot = engine.check_once(BuildWatchRequest(project_root=tmp_path))

    assert snapshot.status in {
        BuildWatchStatus.FAILING,
        BuildWatchStatus.DEGRADED,
    }
    assert source.read_text(encoding="utf-8") == original


def test_build_watch_enum_values_are_stable() -> None:
    assert BuildWatchStatus.HEALTHY.value == "healthy"
    assert BuildWatchDecision.SUGGEST_FIX.value == "suggest_fix"
    assert BuildWatchEventKind.MEMORY_UPDATED.value == "memory_updated"


def _command_plan(status: RunStatus) -> CommandPlan:
    return CommandPlan(
        command=("python", "-m", "pytest"),
        kind=CommandKind.PYTHON_MODULE_PYTEST,
        safety=CommandSafety.SAFE,
        reason=status.value,
    )


def _now() -> datetime:
    return datetime.now(UTC)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")