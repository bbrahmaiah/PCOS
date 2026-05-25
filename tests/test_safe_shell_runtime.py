from __future__ import annotations

from pathlib import Path
from threading import Event

import pytest
from pydantic import ValidationError

from jarvis.tools import (
    ActionStatus,
    SafeShellRuntime,
    SafeShellRuntimeConfig,
    ShellCommandDecision,
    ShellCommandPolicy,
    ShellCommandReason,
    ShellCommandRequest,
    ShellProcessOutcome,
)


class FakeShellRunner:
    def __init__(
        self,
        *,
        outcome: ShellProcessOutcome | None = None,
    ) -> None:
        self.outcome = outcome or ShellProcessOutcome(
            exit_code=0,
            stdout="ok",
            stderr="",
        )
        self.calls: list[tuple[tuple[str, ...], Path, int]] = []

    def run(
        self,
        *,
        argv: tuple[str, ...],
        cwd: Path,
        timeout_ms: int,
        cancellation_event: Event | None,
    ) -> ShellProcessOutcome:
        del cancellation_event

        self.calls.append((argv, cwd, timeout_ms))

        return self.outcome


def test_shell_runtime_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        SafeShellRuntimeConfig(name=" ").validate()

    with pytest.raises(ValueError):
        SafeShellRuntimeConfig(max_output_chars=0).validate()


def test_shell_command_request_requires_command() -> None:
    with pytest.raises(ValidationError):
        ShellCommandRequest(command=" ")


def test_policy_allows_pytest() -> None:
    policy = ShellCommandPolicy()
    result = policy.evaluate(ShellCommandRequest(command="pytest"))

    assert result.decision == ShellCommandDecision.ALLOW
    assert result.reason == ShellCommandReason.ALLOWED_TEST_COMMAND
    assert result.argv == ("pytest",)


def test_policy_allows_ruff_check() -> None:
    result = ShellCommandPolicy().evaluate(
        ShellCommandRequest(command="ruff check .")
    )

    assert result.decision == ShellCommandDecision.ALLOW
    assert result.argv == ("ruff", "check", ".")


def test_policy_allows_mypy() -> None:
    result = ShellCommandPolicy().evaluate(
        ShellCommandRequest(command="mypy .")
    )

    assert result.decision == ShellCommandDecision.ALLOW
    assert result.argv == ("mypy", ".")


def test_policy_allows_python_scripts() -> None:
    result = ShellCommandPolicy().evaluate(
        ShellCommandRequest(command="python scripts/complete_phase45.py")
    )

    assert result.decision == ShellCommandDecision.ALLOW
    assert result.reason == ShellCommandReason.ALLOWED_PYTHON_SCRIPT


def test_policy_allows_git_status() -> None:
    result = ShellCommandPolicy().evaluate(
        ShellCommandRequest(command="git status")
    )

    assert result.decision == ShellCommandDecision.ALLOW
    assert result.reason == ShellCommandReason.ALLOWED_GIT_STATUS


def test_policy_allows_read_only_cmd_builtins() -> None:
    result = ShellCommandPolicy().evaluate(ShellCommandRequest(command="dir"))

    assert result.decision == ShellCommandDecision.ALLOW
    assert result.reason == ShellCommandReason.ALLOWED_READ_ONLY_CMD_BUILTIN


def test_policy_blocks_shell_metacharacters() -> None:
    result = ShellCommandPolicy().evaluate(
        ShellCommandRequest(command="pytest && del important.txt")
    )

    assert result.decision == ShellCommandDecision.DENY
    assert result.reason == ShellCommandReason.UNSAFE_TOKEN


def test_policy_blocks_dangerous_command() -> None:
    result = ShellCommandPolicy().evaluate(
        ShellCommandRequest(command="shutdown /s")
    )

    assert result.decision == ShellCommandDecision.DENY
    assert result.reason == ShellCommandReason.DANGEROUS_COMMAND


def test_policy_requires_approval_for_unknown_command() -> None:
    result = ShellCommandPolicy().evaluate(
        ShellCommandRequest(command="unknown-tool --danger")
    )

    assert result.decision == ShellCommandDecision.REQUIRE_APPROVAL
    assert result.reason == ShellCommandReason.UNKNOWN_COMMAND


def test_policy_requires_approval_for_python_outside_scripts() -> None:
    result = ShellCommandPolicy().evaluate(
        ShellCommandRequest(command="python -c print(1)")
    )

    assert result.decision == ShellCommandDecision.REQUIRE_APPROVAL
    assert result.reason == ShellCommandReason.APPROVAL_REQUIRED


def test_runtime_executes_allowed_command_with_fake_runner() -> None:
    runner = FakeShellRunner()
    runtime = SafeShellRuntime(runner=runner)

    result = runtime.execute(ShellCommandRequest(command="pytest"))

    assert result.status == ActionStatus.SUCCEEDED
    assert result.success is True
    assert result.stdout == "ok"
    assert runner.calls[0][0] == ("pytest",)


def test_runtime_blocks_unknown_command() -> None:
    runner = FakeShellRunner()
    runtime = SafeShellRuntime(runner=runner)

    result = runtime.execute(
        ShellCommandRequest(command="unknown-tool --danger")
    )

    assert result.status == ActionStatus.BLOCKED
    assert result.success is False
    assert not runner.calls


def test_runtime_blocks_dangerous_command() -> None:
    runner = FakeShellRunner()
    runtime = SafeShellRuntime(runner=runner)

    result = runtime.execute(ShellCommandRequest(command="rm -rf /"))

    assert result.status == ActionStatus.BLOCKED
    assert result.policy_result.reason == ShellCommandReason.DANGEROUS_COMMAND
    assert not runner.calls


def test_runtime_blocks_workspace_escape() -> None:
    runner = FakeShellRunner()
    runtime = SafeShellRuntime(
        config=SafeShellRuntimeConfig(workspace_root="."),
        runner=runner,
    )

    result = runtime.execute(
        ShellCommandRequest(command="pytest", working_directory="..")
    )

    assert result.status == ActionStatus.FAILED
    assert result.success is False
    assert not runner.calls


def test_runtime_records_failed_exit_code() -> None:
    runner = FakeShellRunner(
        outcome=ShellProcessOutcome(
            exit_code=1,
            stdout="",
            stderr="failed",
        )
    )
    runtime = SafeShellRuntime(runner=runner)

    result = runtime.execute(ShellCommandRequest(command="pytest"))

    assert result.status == ActionStatus.FAILED
    assert result.success is False
    assert result.stderr == "failed"


def test_runtime_records_timeout() -> None:
    runner = FakeShellRunner(
        outcome=ShellProcessOutcome(
            exit_code=None,
            stdout="",
            stderr="timeout",
            timed_out=True,
        )
    )
    runtime = SafeShellRuntime(runner=runner)

    result = runtime.execute(ShellCommandRequest(command="pytest"))

    assert result.status == ActionStatus.FAILED
    assert result.timed_out is True


def test_runtime_records_cancellation() -> None:
    runner = FakeShellRunner(
        outcome=ShellProcessOutcome(
            exit_code=None,
            stdout="",
            stderr="cancelled",
            cancelled=True,
        )
    )
    runtime = SafeShellRuntime(runner=runner)

    result = runtime.execute(ShellCommandRequest(command="pytest"))

    assert result.status == ActionStatus.CANCELLED
    assert result.cancelled is True


def test_runtime_truncates_output() -> None:
    runner = FakeShellRunner(
        outcome=ShellProcessOutcome(
            exit_code=0,
            stdout="abcdef",
            stderr="",
        )
    )
    runtime = SafeShellRuntime(
        config=SafeShellRuntimeConfig(max_output_chars=3),
        runner=runner,
    )

    result = runtime.execute(ShellCommandRequest(command="pytest"))

    assert result.stdout == "abc\n...[truncated]"


def test_snapshot_and_reset() -> None:
    runner = FakeShellRunner()
    runtime = SafeShellRuntime(runner=runner)

    runtime.execute(ShellCommandRequest(command="pytest"))
    snapshot = runtime.snapshot()

    assert snapshot.execution_count == 1
    assert snapshot.success_count == 1
    assert snapshot.last_status == ActionStatus.SUCCEEDED

    runtime.reset()
    reset_snapshot = runtime.snapshot()

    assert reset_snapshot.execution_count == 0
    assert reset_snapshot.last_status is None


def test_enum_values_are_stable() -> None:
    assert ShellCommandDecision.ALLOW.value == "allow"
    assert ShellCommandReason.DANGEROUS_COMMAND.value == "dangerous_command"