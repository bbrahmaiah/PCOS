from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.tools import (
    ActionAuditLog,
    ActionPlanningRequest,
    ActionStatus,
    FileSystemRuntime,
    FileSystemRuntimeConfig,
    IdeRuntime,
    IdeRuntimeConfig,
    RealActionSmokeReason,
    RealActionSmokeRequest,
    RealActionSmokeRuntime,
    RealActionSmokeRuntimeConfig,
    RealActionSmokeStatus,
    SafeShellRuntime,
    SafeShellRuntimeConfig,
    ShellProcessOutcome,
    SmokeDispatchKind,
)


class FakeShellRunner:
    def __init__(
        self,
        *,
        exit_code: int = 0,
        stdout: str = "tests passed",
        stderr: str = "",
    ) -> None:
        self.outcome = ShellProcessOutcome(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
        )
        self.commands: list[tuple[str, ...]] = []

    def run(self, **kwargs: object) -> ShellProcessOutcome:
        argv = kwargs["argv"]

        assert isinstance(argv, tuple)

        self.commands.append(argv)

        return self.outcome


class FakeMemoryGateway:
    def __init__(self) -> None:
        self.writes: list[dict[str, object]] = []

    def write_tool_memory(
        self,
        *,
        content: str,
        source: str,
        confidence: float,
        policy_class: str,
        reason: str,
        tags: tuple[str, ...],
        metadata: dict[str, object],
    ) -> str:
        self.writes.append(
            {
                "content": content,
                "source": source,
                "confidence": confidence,
                "policy_class": policy_class,
                "reason": reason,
                "tags": tags,
                "metadata": metadata,
            }
        )

        return "memory-1"


class FakeEditorLauncher:
    def __init__(self) -> None:
        self.files: list[str] = []

    def open_file(self, path: str) -> bool:
        self.files.append(path)

        return True

    def open_symbol(self, symbol: str, path: str | None = None) -> bool:
        del symbol, path

        return True


def runtime(
    tmp_path: Path,
    *,
    runner: FakeShellRunner | None = None,
    audit_log: ActionAuditLog | None = None,
) -> RealActionSmokeRuntime:
    shell = SafeShellRuntime(
        config=SafeShellRuntimeConfig(workspace_root=str(tmp_path)),
        runner=runner or FakeShellRunner(),
    )
    file_runtime = FileSystemRuntime(
        config=FileSystemRuntimeConfig(workspace_root=str(tmp_path))
    )
    ide_runtime = IdeRuntime(
        config=IdeRuntimeConfig(workspace_root=str(tmp_path)),
        file_runtime=file_runtime,
        shell_runtime=shell,
        editor_launcher=FakeEditorLauncher(),
    )

    return RealActionSmokeRuntime(
        shell_runtime=shell,
        file_runtime=file_runtime,
        ide_runtime=ide_runtime,
        audit_log=audit_log or ActionAuditLog(),
    )


def request(intent: str) -> RealActionSmokeRequest:
    return RealActionSmokeRequest(
        planning_request=ActionPlanningRequest(user_intent=intent)
    )


def test_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        RealActionSmokeRuntimeConfig(name=" ").validate()


def test_run_tests_pipeline_success(tmp_path: Path) -> None:
    runner = FakeShellRunner(stdout="1410 passed")
    smoke = runtime(tmp_path, runner=runner)

    result = smoke.run(request("run tests and summarize failures"))

    assert result.success is True
    assert result.status == RealActionSmokeStatus.SUCCEEDED
    assert result.reason == RealActionSmokeReason.PIPELINE_SUCCEEDED
    assert result.proposal is not None
    assert result.validation_result is not None
    assert len(result.dispatch_results) == 1
    assert result.dispatch_results[0].kind == SmokeDispatchKind.SHELL
    assert runner.commands == [("pytest",)]
    assert result.memory_proposal is not None
    assert result.memory_result is None


def test_run_tests_pipeline_failure(tmp_path: Path) -> None:
    runner = FakeShellRunner(
        exit_code=1,
        stdout="",
        stderr="test failed",
    )
    smoke = runtime(tmp_path, runner=runner)

    result = smoke.run(request("run tests and summarize failures"))

    assert result.success is False
    assert result.status == RealActionSmokeStatus.FAILED
    assert result.reason == RealActionSmokeReason.DISPATCH_FAILED
    assert result.dispatch_results[0].status == ActionStatus.FAILED


def test_quality_gate_runs_multiple_shell_steps(tmp_path: Path) -> None:
    runner = FakeShellRunner(stdout="ok")
    smoke = runtime(tmp_path, runner=runner)

    result = smoke.run(request("run quality gate"))

    assert result.success is True
    assert len(result.dispatch_results) == 3
    assert runner.commands == [
        ("ruff", "check", "."),
        ("mypy", "."),
        ("pytest",),
    ]


def test_open_file_pipeline_success(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("print('ok')", encoding="utf-8")
    smoke = runtime(tmp_path)

    result = smoke.run(
        RealActionSmokeRequest(
            planning_request=ActionPlanningRequest(
                user_intent="open this file",
                target_path="a.py",
            )
        )
    )

    assert result.success is True
    assert result.dispatch_results[0].kind == SmokeDispatchKind.IDE
    assert result.dispatch_results[0].output == "file open requested: a.py"


def test_search_project_pipeline_success(tmp_path: Path) -> None:
    (tmp_path / "memory_gateway.py").write_text("x", encoding="utf-8")
    (tmp_path / "other.txt").write_text("y", encoding="utf-8")
    smoke = runtime(tmp_path)

    result = smoke.run(
        RealActionSmokeRequest(
            planning_request=ActionPlanningRequest(
                user_intent="search for memory_gateway",
            )
        )
    )

    assert result.success is True
    assert "memory_gateway.py" in result.dispatch_results[0].output


def test_unknown_intent_fails_before_execution(tmp_path: Path) -> None:
    smoke = runtime(tmp_path)

    result = smoke.run(request("do something magical"))

    assert result.success is False
    assert result.status == RealActionSmokeStatus.FAILED
    assert result.reason == RealActionSmokeReason.PLANNING_FAILED
    assert result.dispatch_results == ()


def test_medium_risk_patch_blocked_by_default(tmp_path: Path) -> None:
    smoke = runtime(tmp_path)

    result = smoke.run(
        RealActionSmokeRequest(
            planning_request=ActionPlanningRequest(
                user_intent="prepare patch",
                target_path="a.py",
                old_text="old",
                new_text="new",
            )
        )
    )

    assert result.success is False
    assert result.status == RealActionSmokeStatus.APPROVAL_REQUIRED
    assert result.reason == RealActionSmokeReason.APPROVAL_REQUIRED


def test_medium_risk_patch_allowed_when_configured(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("old", encoding="utf-8")
    smoke = RealActionSmokeRuntime(
        config=RealActionSmokeRuntimeConfig(allow_medium_risk_smoke=True),
        file_runtime=FileSystemRuntime(
            config=FileSystemRuntimeConfig(workspace_root=str(tmp_path))
        ),
        shell_runtime=SafeShellRuntime(
            config=SafeShellRuntimeConfig(workspace_root=str(tmp_path)),
            runner=FakeShellRunner(),
        ),
        ide_runtime=IdeRuntime(
            config=IdeRuntimeConfig(workspace_root=str(tmp_path)),
            file_runtime=FileSystemRuntime(
                config=FileSystemRuntimeConfig(workspace_root=str(tmp_path))
            ),
            shell_runtime=SafeShellRuntime(
                config=SafeShellRuntimeConfig(workspace_root=str(tmp_path)),
                runner=FakeShellRunner(),
            ),
            editor_launcher=FakeEditorLauncher(),
        ),
    )

    result = smoke.run(
        RealActionSmokeRequest(
            planning_request=ActionPlanningRequest(
                user_intent="prepare patch",
                target_path="a.py",
                old_text="old",
                new_text="new",
            ),
            approved=True,
        )
    )

    assert result.success is True
    assert result.dispatch_results[0].kind == SmokeDispatchKind.IDE
    assert "patch prepared" in result.dispatch_results[0].output
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "old"


def test_audit_log_records_intent_plan_and_execution(tmp_path: Path) -> None:
    audit_log = ActionAuditLog()
    smoke = runtime(tmp_path, audit_log=audit_log)

    result = smoke.run(request("run tests and summarize failures"))
    records = audit_log.all_records()
    event_kinds = [record.event_kind.value for record in records]

    assert result.success is True
    assert "intent_received" in event_kinds
    assert "plan_proposed" in event_kinds
    assert "execution_completed" in event_kinds


def test_snapshot_and_reset(tmp_path: Path) -> None:
    smoke = runtime(tmp_path)

    smoke.run(request("run tests and summarize failures"))
    snapshot = smoke.snapshot()

    assert snapshot.request_count == 1
    assert snapshot.success_count == 1
    assert snapshot.last_status == RealActionSmokeStatus.SUCCEEDED

    smoke.reset()
    reset_snapshot = smoke.snapshot()

    assert reset_snapshot.request_count == 0
    assert reset_snapshot.last_status is None


def test_enum_values_are_stable() -> None:
    assert RealActionSmokeStatus.SUCCEEDED.value == "succeeded"
    assert RealActionSmokeReason.PIPELINE_SUCCEEDED.value == "pipeline_succeeded"
    assert SmokeDispatchKind.SHELL.value == "shell"