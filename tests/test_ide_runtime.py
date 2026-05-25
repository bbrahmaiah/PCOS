from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from jarvis.tools import (
    ActionStatus,
    FileSystemRuntime,
    FileSystemRuntimeConfig,
    IdeActionDecision,
    IdeActionKind,
    IdeActionReason,
    IdeActionRequest,
    IdePolicy,
    IdeRuntime,
    IdeRuntimeConfig,
    SafeShellRuntime,
    SafeShellRuntimeConfig,
    ShellProcessOutcome,
)


class FakeEditorLauncher:
    def __init__(self, *, opened: bool = True) -> None:
        self.opened = opened
        self.files: list[str] = []
        self.symbols: list[tuple[str, str | None]] = []

    def open_file(self, path: str) -> bool:
        self.files.append(path)

        return self.opened

    def open_symbol(self, symbol: str, path: str | None = None) -> bool:
        self.symbols.append((symbol, path))

        return self.opened


class FakeShellRunner:
    def __init__(self, *, outcome: ShellProcessOutcome | None = None) -> None:
        self.outcome = outcome or ShellProcessOutcome(
            exit_code=0,
            stdout="tests passed",
            stderr="",
        )
        self.commands: list[tuple[str, ...]] = []

    def run(self, **kwargs: object) -> ShellProcessOutcome:
        argv = kwargs["argv"]

        assert isinstance(argv, tuple)

        self.commands.append(argv)

        return self.outcome


def runtime(tmp_path: Path) -> IdeRuntime:
    return IdeRuntime(
        config=IdeRuntimeConfig(workspace_root=str(tmp_path)),
        file_runtime=FileSystemRuntime(
            config=FileSystemRuntimeConfig(workspace_root=str(tmp_path))
        ),
        shell_runtime=SafeShellRuntime(
            config=SafeShellRuntimeConfig(workspace_root=str(tmp_path)),
            runner=FakeShellRunner(),
        ),
        editor_launcher=FakeEditorLauncher(),
    )




def test_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        IdeRuntimeConfig(name=" ").validate()

    with pytest.raises(ValueError):
        IdeRuntimeConfig(workspace_root=" ").validate()

    with pytest.raises(ValueError):
        IdeRuntimeConfig(allowed_test_commands=()).validate()


def test_request_requires_path_for_open_file() -> None:
    with pytest.raises(ValidationError):
        IdeActionRequest(kind=IdeActionKind.OPEN_FILE)


def test_request_requires_symbol_for_open_symbol() -> None:
    with pytest.raises(ValidationError):
        IdeActionRequest(kind=IdeActionKind.OPEN_SYMBOL)


def test_request_requires_patch_payload() -> None:
    with pytest.raises(ValidationError):
        IdeActionRequest(
            kind=IdeActionKind.PREPARE_PATCH,
            path="a.py",
        )


def test_request_requires_test_command() -> None:
    with pytest.raises(ValidationError):
        IdeActionRequest(kind=IdeActionKind.RUN_TESTS)


def test_policy_allows_open_file() -> None:
    policy = IdePolicy(allowed_test_commands=("pytest",))

    result = policy.evaluate(
        IdeActionRequest(
            kind=IdeActionKind.OPEN_FILE,
            path="a.py",
        )
    )

    assert result.decision == IdeActionDecision.ALLOW
    assert result.reason == IdeActionReason.SAFE_OPEN_ALLOWED


def test_policy_blocks_unknown_test_command() -> None:
    policy = IdePolicy(allowed_test_commands=("pytest",))

    result = policy.evaluate(
        IdeActionRequest(
            kind=IdeActionKind.RUN_TESTS,
            test_command="npm install something",
        )
    )

    assert result.decision == IdeActionDecision.DENY
    assert result.reason == IdeActionReason.TEST_COMMAND_NOT_ALLOWED


def test_apply_patch_requires_approval() -> None:
    policy = IdePolicy(allowed_test_commands=("pytest",))

    result = policy.evaluate(
        IdeActionRequest(
            kind=IdeActionKind.APPLY_PATCH,
            path="a.py",
            old_text="old",
            new_text="new",
        )
    )

    assert result.decision == IdeActionDecision.REQUIRE_APPROVAL
    assert result.reason == IdeActionReason.PATCH_REQUIRES_APPROVAL


def test_open_file_uses_editor_launcher(tmp_path: Path) -> None:
    launcher = FakeEditorLauncher()
    ide = IdeRuntime(
        config=IdeRuntimeConfig(workspace_root=str(tmp_path)),
        editor_launcher=launcher,
    )

    result = ide.execute(
        IdeActionRequest(
            kind=IdeActionKind.OPEN_FILE,
            path="a.py",
        )
    )

    assert result.success is True
    assert launcher.files == ["a.py"]


def test_open_symbol_uses_editor_launcher(tmp_path: Path) -> None:
    launcher = FakeEditorLauncher()
    ide = IdeRuntime(
        config=IdeRuntimeConfig(workspace_root=str(tmp_path)),
        editor_launcher=launcher,
    )

    result = ide.execute(
        IdeActionRequest(
            kind=IdeActionKind.OPEN_SYMBOL,
            symbol="AdaptiveTurnDetector",
            path="jarvis/conversation/turn_detection.py",
        )
    )

    assert result.success is True
    assert launcher.symbols == [
        ("AdaptiveTurnDetector", "jarvis/conversation/turn_detection.py")
    ]


def test_show_diagnostics_reads_file(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("print('debug')\n# TODO fix\n", encoding="utf-8")
    ide = runtime(tmp_path)

    result = ide.execute(
        IdeActionRequest(
            kind=IdeActionKind.SHOW_DIAGNOSTICS,
            path="a.py",
        )
    )

    assert result.success is True
    assert "print statement found" in result.output
    assert "TODO fix" in result.output


def test_prepare_patch_returns_diff_without_modifying_file(tmp_path: Path) -> None:
    target = tmp_path / "a.py"
    target.write_text("old", encoding="utf-8")
    ide = runtime(tmp_path)

    result = ide.execute(
        IdeActionRequest(
            kind=IdeActionKind.PREPARE_PATCH,
            path="a.py",
            old_text="old",
            new_text="new",
        )
    )

    assert result.success is True
    assert result.diff is not None
    assert "-old" in result.diff
    assert "+new" in result.diff
    assert target.read_text(encoding="utf-8") == "old"


def test_apply_patch_without_approval_is_blocked(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("old", encoding="utf-8")
    ide = runtime(tmp_path)

    result = ide.execute(
        IdeActionRequest(
            kind=IdeActionKind.APPLY_PATCH,
            path="a.py",
            old_text="old",
            new_text="new",
        )
    )

    assert result.success is False
    assert result.status == ActionStatus.BLOCKED
    assert result.decision == IdeActionDecision.REQUIRE_APPROVAL


def test_apply_patch_with_approval_uses_file_runtime(tmp_path: Path) -> None:
    target = tmp_path / "a.py"
    target.write_text("old", encoding="utf-8")
    ide = runtime(tmp_path)

    result = ide.execute(
        IdeActionRequest(
            kind=IdeActionKind.APPLY_PATCH,
            path="a.py",
            old_text="old",
            new_text="new",
            approved=True,
        )
    )

    assert result.success is True
    assert target.read_text(encoding="utf-8") == "new"
    assert result.file_result is not None
    assert result.file_result.backup_path is not None


def test_run_tests_uses_safe_shell_runtime(tmp_path: Path) -> None:
    runner = FakeShellRunner()
    ide = IdeRuntime(
        config=IdeRuntimeConfig(workspace_root=str(tmp_path)),
        file_runtime=FileSystemRuntime(
            config=FileSystemRuntimeConfig(workspace_root=str(tmp_path))
        ),
        shell_runtime=SafeShellRuntime(
            config=SafeShellRuntimeConfig(workspace_root=str(tmp_path)),
            runner=runner,
        ),
        editor_launcher=FakeEditorLauncher(),
    )

    result = ide.execute(
        IdeActionRequest(
            kind=IdeActionKind.RUN_TESTS,
            test_command="pytest",
        )
    )

    assert result.success is True
    assert runner.commands == [("pytest",)]


def test_navigate_project_uses_file_search(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    ide = runtime(tmp_path)

    result = ide.execute(
        IdeActionRequest(
            kind=IdeActionKind.NAVIGATE_PROJECT,
            path=".",
        )
    )

    assert result.success is True
    assert "a.py" in result.output
    assert "b.txt" not in result.output


def test_editor_launcher_failure_returns_failed(tmp_path: Path) -> None:
    ide = IdeRuntime(
        config=IdeRuntimeConfig(workspace_root=str(tmp_path)),
        editor_launcher=FakeEditorLauncher(opened=False),
    )

    result = ide.execute(
        IdeActionRequest(
            kind=IdeActionKind.OPEN_FILE,
            path="a.py",
        )
    )

    assert result.success is False
    assert result.status == ActionStatus.FAILED


def test_snapshot_and_reset(tmp_path: Path) -> None:
    ide = runtime(tmp_path)

    ide.execute(
        IdeActionRequest(
            kind=IdeActionKind.OPEN_FILE,
            path="a.py",
        )
    )
    snapshot = ide.snapshot()

    assert snapshot.action_count == 1
    assert snapshot.success_count == 1

    ide.reset()
    reset_snapshot = ide.snapshot()

    assert reset_snapshot.action_count == 0
    assert reset_snapshot.last_status is None


def test_enum_values_are_stable() -> None:
    assert IdeActionKind.OPEN_FILE.value == "open_file"
    assert IdeActionDecision.REQUIRE_APPROVAL.value == "require_approval"
    assert IdeActionReason.PATCH_REQUIRES_APPROVAL.value == (
        "patch_requires_approval"
    )