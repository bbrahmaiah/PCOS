from __future__ import annotations

import sys
from pathlib import Path

import pytest

from jarvis.developer import (
    TestCommandKind as CommandKind,
)
from jarvis.developer import (
    TestCommandSafety as CommandSafety,
)
from jarvis.developer import (
    TestDiscoveryRequest as DiscoveryRequest,
)
from jarvis.developer import (
    TestDiscoveryStatus as DiscoveryStatus,
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
    TestRunRequest as RunRequest,
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
            duration_ms=12.5,
        )


def test_test_discovery_request_rejects_invalid_max_files(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        DiscoveryRequest(
            project_root=tmp_path,
            max_files=0,
        )


def test_test_run_request_rejects_invalid_timeout(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        RunRequest(
            project_root=tmp_path,
            timeout_seconds=0,
        )


def test_discovery_blocks_missing_project_root(
    tmp_path: Path,
) -> None:
    engine = RunnerEngine()
    report = engine.discover(
        DiscoveryRequest(project_root=tmp_path / "missing")
    )

    assert report.status == DiscoveryStatus.BLOCKED
    assert report.suggested_plan is None


def test_discovery_detects_pytest_project(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "pyproject.toml", "[tool.pytest.ini_options]\n")
    _write(tmp_path / "tests" / "test_core.py", "def test_core(): pass\n")

    engine = RunnerEngine()
    report = engine.discover(DiscoveryRequest(project_root=tmp_path))

    assert report.status == DiscoveryStatus.READY
    assert report.suggested_plan is not None
    assert report.suggested_plan.safe is True
    assert report.suggested_plan.kind == CommandKind.PYTHON_MODULE_PYTEST
    assert str(Path("tests") / "test_core.py") in report.test_files


def test_plan_blocks_unsafe_command(
    tmp_path: Path,
) -> None:
    engine = RunnerEngine()
    plan = engine.plan(
        RunRequest(
            project_root=tmp_path,
            command=("rm", "-rf", "."),
        )
    )

    assert plan.safety == CommandSafety.BLOCKED
    assert plan.blocked_reason is not None


def test_plan_allows_python_module_pytest(
    tmp_path: Path,
) -> None:
    engine = RunnerEngine()
    plan = engine.plan(
        RunRequest(
            project_root=tmp_path,
            command=(sys.executable, "-m", "pytest"),
        )
    )

    assert plan.safe is True
    assert plan.kind == CommandKind.PYTHON_MODULE_PYTEST


def test_run_executes_safe_plan_and_passes(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "tests" / "test_core.py", "def test_core(): pass\n")
    adapter = FakeTestExecutionAdapter(exit_code=0, stdout="1 passed")
    engine = RunnerEngine(executor=adapter)

    result = engine.run(RunRequest(project_root=tmp_path))

    assert result.status == RunStatus.PASSED
    assert result.passed is True
    assert result.exit_code == 0
    assert result.summary == "Tests passed."
    assert len(adapter.requests) == 1
    assert adapter.requests[0].command[1:] == ("-m", "pytest")


def test_run_reports_failed_tests(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "tests" / "test_core.py", "def test_core(): assert False\n")
    adapter = FakeTestExecutionAdapter(
        exit_code=1,
        stdout="",
        stderr="failed",
    )
    engine = RunnerEngine(executor=adapter)

    result = engine.run(RunRequest(project_root=tmp_path))

    assert result.status == RunStatus.FAILED
    assert result.passed is False
    assert result.exit_code == 1
    assert "failed" in result.stderr


def test_run_reports_timeout(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "tests" / "test_core.py", "def test_core(): pass\n")
    adapter = FakeTestExecutionAdapter(
        exit_code=None,
        stdout="",
        stderr="",
        timed_out=True,
    )
    engine = RunnerEngine(executor=adapter)

    result = engine.run(
        RunRequest(
            project_root=tmp_path,
            timeout_seconds=1,
        )
    )

    assert result.status == RunStatus.TIMEOUT
    assert result.exit_code is None


def test_run_blocks_when_no_test_workflow_detected(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "main.py", "print('hello')\n")
    adapter = FakeTestExecutionAdapter()
    engine = RunnerEngine(executor=adapter)

    result = engine.run(RunRequest(project_root=tmp_path))

    assert result.status == RunStatus.BLOCKED
    assert len(adapter.requests) == 0


def test_run_blocks_invalid_project_root(
    tmp_path: Path,
) -> None:
    engine = RunnerEngine()

    result = engine.run(RunRequest(project_root=tmp_path / "missing"))

    assert result.status == RunStatus.BLOCKED
    assert result.exit_code is None


def test_test_runner_enum_values_are_stable() -> None:
    assert RunStatus.PASSED.value == "passed"
    assert CommandKind.PYTHON_MODULE_PYTEST.value == "python_module_pytest"
    assert CommandSafety.SAFE.value == "safe"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")