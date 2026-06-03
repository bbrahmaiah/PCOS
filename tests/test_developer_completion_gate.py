from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.developer import (
    DeveloperFeaturePackCheckKind,
    DeveloperFeaturePackCompletionGate,
    DeveloperFeaturePackGateConfig,
    DeveloperFeaturePackGateStatus,
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


class FakeExecutionAdapter:
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

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        self.requests.append(request)
        return ExecutionResult(
            exit_code=self.exit_code,
            stdout=self.stdout,
            stderr=self.stderr,
            timed_out=self.timed_out,
            duration_ms=10.0,
        )


def test_developer_gate_config_rejects_bad_values(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        DeveloperFeaturePackGateConfig(project_root=tmp_path, max_files=0)

    with pytest.raises(ValueError):
        DeveloperFeaturePackGateConfig(
            project_root=tmp_path,
            timeout_seconds=0,
        )


def test_developer_gate_blocks_missing_project_root(
    tmp_path: Path,
) -> None:
    gate = DeveloperFeaturePackCompletionGate(
        config=DeveloperFeaturePackGateConfig(
            project_root=tmp_path / "missing",
        )
    )

    report = gate.run()

    assert report.status == DeveloperFeaturePackGateStatus.FAILED
    assert report.passed is False
    assert report.error is not None

    kinds = {check.kind for check in report.checks}

    assert DeveloperFeaturePackCheckKind.PROJECT_VALID in kinds


def test_developer_gate_passes_integrated_pack(
    tmp_path: Path,
) -> None:
    _build_project(tmp_path)

    adapter = FakeExecutionAdapter(exit_code=0, stdout="1 passed")
    test_runner = RunnerEngine(executor=adapter)

    gate = DeveloperFeaturePackCompletionGate(
        config=DeveloperFeaturePackGateConfig(
            project_root=tmp_path,
            active_file=tmp_path / "jarvis" / "core.py",
            navigation_query="run",
            remember=True,
        ),
        test_runner=test_runner,
    )

    report = gate.run()

    assert report.status == DeveloperFeaturePackGateStatus.PASSED
    assert report.passed is True
    assert report.failed_count == 0
    assert len(adapter.requests) >= 2

    kinds = {check.kind for check in report.checks}

    assert DeveloperFeaturePackCheckKind.PROJECT_VALID in kinds
    assert DeveloperFeaturePackCheckKind.CODE_CONTEXT in kinds
    assert DeveloperFeaturePackCheckKind.CODE_NAVIGATION_INDEX in kinds
    assert DeveloperFeaturePackCheckKind.CODE_NAVIGATION_SEARCH in kinds
    assert DeveloperFeaturePackCheckKind.TEST_DISCOVERY in kinds
    assert DeveloperFeaturePackCheckKind.TEST_RUNNER in kinds
    assert DeveloperFeaturePackCheckKind.ERROR_INTELLIGENCE in kinds
    assert DeveloperFeaturePackCheckKind.FIX_SUGGESTION in kinds
    assert DeveloperFeaturePackCheckKind.PROJECT_MEMORY in kinds
    assert DeveloperFeaturePackCheckKind.BUILD_WATCH in kinds
    assert DeveloperFeaturePackCheckKind.SOURCE_IMMUTABILITY in kinds


def test_developer_gate_handles_failing_tests(
    tmp_path: Path,
) -> None:
    _build_project(tmp_path)
    failing_output = """________________ test_example ________________

    def test_example():
>       assert 1 == 2
E       assert 1 == 2

tests/test_core.py:3: AssertionError
"""

    adapter = FakeExecutionAdapter(exit_code=1, stdout=failing_output)
    gate = DeveloperFeaturePackCompletionGate(
        config=DeveloperFeaturePackGateConfig(project_root=tmp_path),
        test_runner=RunnerEngine(executor=adapter),
    )

    report = gate.run()

    assert report.status == DeveloperFeaturePackGateStatus.PASSED
    assert report.failed_count == 0


def test_developer_gate_does_not_mutate_source_files(
    tmp_path: Path,
) -> None:
    _build_project(tmp_path)
    source = tmp_path / "jarvis" / "core.py"
    before = source.read_text(encoding="utf-8")

    gate = DeveloperFeaturePackCompletionGate(
        config=DeveloperFeaturePackGateConfig(project_root=tmp_path),
        test_runner=RunnerEngine(
            executor=FakeExecutionAdapter(exit_code=0, stdout="1 passed")
        ),
    )

    report = gate.run()

    assert report.status == DeveloperFeaturePackGateStatus.PASSED
    assert source.read_text(encoding="utf-8") == before


def test_developer_gate_enum_values_are_stable() -> None:
    assert DeveloperFeaturePackGateStatus.PASSED.value == "passed"
    assert DeveloperFeaturePackCheckKind.BUILD_WATCH.value == "build_watch"


def _build_project(root: Path) -> None:
    _write(root / "pyproject.toml", "[tool.pytest.ini_options]\n")
    _write(root / "README.md", "# Demo\n")
    _write(
        root / "jarvis" / "core.py",
        """
def run():
    return 1

class DeveloperRuntime:
    def start(self):
        return run()
""",
    )
    _write(
        root / "tests" / "test_core.py",
        """
from jarvis.core import run

def test_core():
    assert run() == 1
""",
    )


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.lstrip(), encoding="utf-8")