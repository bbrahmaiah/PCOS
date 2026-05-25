from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from threading import Event, RLock
from typing import Protocol

from pydantic import Field, field_validator

from jarvis.tools.ids import new_action_id, new_action_result_id, utc_now
from jarvis.tools.models import (
    ActionKind,
    ActionPlan,
    ActionRisk,
    ActionScope,
    ActionStatus,
    ActionStep,
    PermissionDecision,
    ToolCapability,
    ToolModel,
)
from jarvis.tools.registry import (
    ToolAvailability,
    ToolDescriptor,
    ToolHealth,
    ToolRegistry,
)
from jarvis.tools.validation import (
    ActionValidationDecision,
    ActionValidationResult,
    ActionValidator,
    ActionValidatorConfig,
)


class ShellCommandDecision(StrEnum):
    """
    Shell-specific policy decision.
    """

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


class ShellCommandReason(StrEnum):
    """
    Machine-readable shell policy reason.
    """

    ALLOWED_TEST_COMMAND = "allowed_test_command"
    ALLOWED_QUALITY_COMMAND = "allowed_quality_command"
    ALLOWED_PYTHON_SCRIPT = "allowed_python_script"
    ALLOWED_GIT_STATUS = "allowed_git_status"
    ALLOWED_READ_ONLY_CMD_BUILTIN = "allowed_read_only_cmd_builtin"
    EMPTY_COMMAND = "empty_command"
    UNSAFE_TOKEN = "unsafe_token"
    DANGEROUS_COMMAND = "dangerous_command"
    UNKNOWN_COMMAND = "unknown_command"
    APPROVAL_REQUIRED = "approval_required"
    WORKSPACE_OUT_OF_BOUNDS = "workspace_out_of_bounds"
    VALIDATION_BLOCKED = "validation_blocked"
    TIMEOUT_OCCURRED = "timeout_occurred"
    CANCELLED = "cancelled"
    EXECUTION_FAILED = "execution_failed"
    EXECUTION_SUCCEEDED = "execution_succeeded"


class ShellCommandRequest(ToolModel):
    """
    Request to execute a safe shell command.

    This is not raw command execution. SafeShellRuntime still applies shell
    policy, typed action planning, validation, timeout, cancellation, and
    observable result capture.
    """

    action_id: str = Field(default_factory=new_action_id)
    command: str
    working_directory: str | None = None
    timeout_ms: int = Field(default=30_000, ge=1, le=300_000)
    cancellation_token_id: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("action_id", "command")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("working_directory", "cancellation_token_id")
    @classmethod
    def _clean_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None


class ShellCommandPolicyResult(ToolModel):
    """
    Result of shell-specific policy evaluation.
    """

    decision: ShellCommandDecision
    permission_decision: PermissionDecision
    reason: ShellCommandReason
    risk: ActionRisk
    executable: str | None = None
    argv: tuple[str, ...] = ()
    explanation: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("explanation")
    @classmethod
    def _explanation_required(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("explanation cannot be empty.")

        return cleaned


class ShellCommandResult(ToolModel):
    """
    Observable result of a governed shell command.
    """

    result_id: str = Field(default_factory=new_action_result_id)
    action_id: str
    command: str
    status: ActionStatus
    success: bool
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    cancelled: bool = False
    policy_result: ShellCommandPolicyResult
    validation_result: ActionValidationResult | None = None
    started_at: object = Field(default_factory=utc_now)
    completed_at: object = Field(default_factory=utc_now)
    duration_ms: int = Field(default=0, ge=0)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("result_id", "action_id", "command")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class ShellProcessOutcome:
    """
    Low-level process outcome returned by a shell process runner.
    """

    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    cancelled: bool = False


class ShellProcessRunner(Protocol):
    """
    Process runner protocol.

    Tests can inject a fake runner. Production uses SubprocessShellRunner.
    """

    def run(
        self,
        *,
        argv: tuple[str, ...],
        cwd: Path,
        timeout_ms: int,
        cancellation_event: Event | None,
    ) -> ShellProcessOutcome:
        ...


@dataclass(frozen=True, slots=True)
class SafeShellRuntimeConfig:
    """
    Configuration for SafeShellRuntime.
    """

    name: str = "safe_shell_runtime"
    workspace_root: str = "."
    max_output_chars: int = 20_000
    register_default_shell_tool: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.max_output_chars <= 0:
            raise ValueError("max_output_chars must be positive.")


@dataclass(frozen=True, slots=True)
class SafeShellRuntimeSnapshot:
    """
    Observable diagnostics for SafeShellRuntime.
    """

    name: str
    execution_count: int
    success_count: int
    blocked_count: int
    failed_count: int
    timeout_count: int
    cancelled_count: int
    last_status: ActionStatus | None
    last_reason: ShellCommandReason | None
    last_error: str | None


class ShellCommandPolicy:
    """
    Conservative shell command policy.

    This allows only known early commands and blocks or gates everything else.
    """

    _BLOCKED_PATTERNS = (
        r"\brm\s+-rf\b",
        r"\bdel\b",
        r"\brmdir\b",
        r"\bformat\b",
        r"\bshutdown\b",
        r"\brestart-computer\b",
        r"\bstop-computer\b",
        r"\bset-executionpolicy\b",
        r"\breg\s+(add|delete|import|restore)\b",
        r"\bdiskpart\b",
        r"\bbcdedit\b",
        r"\binvoke-expression\b",
        r"\biex\b",
        r"\binvoke-webrequest\b",
        r"\biwr\b",
        r"\bcurl\b",
        r"\bwget\b",
        r"\bstart-process\s+-verb\s+runas\b",
    )
    _UNSAFE_TOKENS = ("&&", "||", "|", ";", ">", "<", "`", "$(")

    def evaluate(self, request: ShellCommandRequest) -> ShellCommandPolicyResult:
        command = request.command.strip()

        if not command:
            return self._deny(
                reason=ShellCommandReason.EMPTY_COMMAND,
                explanation="shell command cannot be empty",
            )

        lowered = command.casefold()

        if any(token in command for token in self._UNSAFE_TOKENS):
            return self._deny(
                reason=ShellCommandReason.UNSAFE_TOKEN,
                explanation="shell metacharacters are blocked",
            )

        for pattern in self._BLOCKED_PATTERNS:
            if re.search(pattern, lowered):
                return self._deny(
                    reason=ShellCommandReason.DANGEROUS_COMMAND,
                    explanation="dangerous shell command pattern is blocked",
                    metadata={"pattern": pattern},
                )

        argv = self._parse(command)

        if not argv:
            return self._deny(
                reason=ShellCommandReason.EMPTY_COMMAND,
                explanation="shell command did not parse into argv",
            )

        executable = argv[0].casefold()

        if executable == "pytest":
            return self._allow(
                argv=argv,
                reason=ShellCommandReason.ALLOWED_TEST_COMMAND,
                explanation="pytest command is allowed",
            )

        if executable == "ruff" and len(argv) >= 2 and argv[1] == "check":
            return self._allow(
                argv=argv,
                reason=ShellCommandReason.ALLOWED_QUALITY_COMMAND,
                explanation="ruff check command is allowed",
            )

        if executable == "mypy":
            return self._allow(
                argv=argv,
                reason=ShellCommandReason.ALLOWED_QUALITY_COMMAND,
                explanation="mypy command is allowed",
            )

        if executable in {"python", "python.exe", "py"}:
            return self._python_decision(argv)

        if executable == "git" and len(argv) >= 2 and argv[1] == "status":
            return self._allow(
                argv=argv,
                reason=ShellCommandReason.ALLOWED_GIT_STATUS,
                explanation="git status is allowed",
            )

        if executable in {"dir", "tree", "type"}:
            return self._allow(
                argv=argv,
                reason=ShellCommandReason.ALLOWED_READ_ONLY_CMD_BUILTIN,
                explanation="read-only Windows shell builtin is allowed",
            )

        return ShellCommandPolicyResult(
            decision=ShellCommandDecision.REQUIRE_APPROVAL,
            permission_decision=PermissionDecision.REQUIRE_APPROVAL,
            reason=ShellCommandReason.UNKNOWN_COMMAND,
            risk=ActionRisk.HIGH,
            executable=argv[0],
            argv=tuple(argv),
            explanation="unknown shell command requires approval",
        )

    def _python_decision(self, argv: list[str]) -> ShellCommandPolicyResult:
        if len(argv) >= 2 and self._is_safe_script_path(argv[1]):
            return self._allow(
                argv=argv,
                reason=ShellCommandReason.ALLOWED_PYTHON_SCRIPT,
                explanation="python scripts/*.py command is allowed",
            )

        return ShellCommandPolicyResult(
            decision=ShellCommandDecision.REQUIRE_APPROVAL,
            permission_decision=PermissionDecision.REQUIRE_APPROVAL,
            reason=ShellCommandReason.APPROVAL_REQUIRED,
            risk=ActionRisk.HIGH,
            executable=argv[0],
            argv=tuple(argv),
            explanation="python command outside scripts/*.py requires approval",
        )

    @staticmethod
    def _is_safe_script_path(value: str) -> bool:
        normalized = value.replace("\\", "/").lstrip("./")

        return normalized.startswith("scripts/") and normalized.endswith(".py")

    @staticmethod
    def _parse(command: str) -> list[str]:
        return shlex.split(command, posix=os.name != "nt")

    @staticmethod
    def _allow(
        *,
        argv: list[str],
        reason: ShellCommandReason,
        explanation: str,
    ) -> ShellCommandPolicyResult:
        return ShellCommandPolicyResult(
            decision=ShellCommandDecision.ALLOW,
            permission_decision=PermissionDecision.ALLOW,
            reason=reason,
            risk=ActionRisk.LOW,
            executable=argv[0],
            argv=tuple(argv),
            explanation=explanation,
        )

    @staticmethod
    def _deny(
        *,
        reason: ShellCommandReason,
        explanation: str,
        metadata: dict[str, object] | None = None,
    ) -> ShellCommandPolicyResult:
        return ShellCommandPolicyResult(
            decision=ShellCommandDecision.DENY,
            permission_decision=PermissionDecision.DENY,
            reason=reason,
            risk=ActionRisk.CRITICAL,
            executable=None,
            argv=(),
            explanation=explanation,
            metadata=metadata or {},
        )


class SubprocessShellRunner:
    """
    Subprocess-based runner with timeout and cooperative cancellation.

    This is used only after SafeShellRuntime applies policy and validation.
    """

    def run(
        self,
        *,
        argv: tuple[str, ...],
        cwd: Path,
        timeout_ms: int,
        cancellation_event: Event | None,
    ) -> ShellProcessOutcome:
        start = time.monotonic()
        timeout_seconds = timeout_ms / 1000

        process = subprocess.Popen(
            self._platform_argv(argv),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
        )

        while process.poll() is None:
            if cancellation_event is not None and cancellation_event.is_set():
                process.terminate()

                try:
                    stdout, stderr = process.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = process.communicate()

                return ShellProcessOutcome(
                    exit_code=process.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    cancelled=True,
                )

            if time.monotonic() - start > timeout_seconds:
                process.kill()
                stdout, stderr = process.communicate()

                return ShellProcessOutcome(
                    exit_code=process.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    timed_out=True,
                )

            time.sleep(0.05)

        stdout, stderr = process.communicate()

        return ShellProcessOutcome(
            exit_code=process.returncode,
            stdout=stdout,
            stderr=stderr,
        )

    @staticmethod
    def _platform_argv(argv: tuple[str, ...]) -> tuple[str, ...]:
        if sys.platform == "win32" and argv[0].casefold() in {"dir", "tree", "type"}:
            return ("cmd", "/d", "/c", *argv)

        return argv


class SafeShellRuntime:
    """
    Governed shell runtime.

    Responsibilities:
    - accept a typed ShellCommandRequest
    - enforce shell-specific command policy
    - convert command into typed ActionPlan
    - validate with ActionValidator
    - execute only allowed and validated commands
    - enforce timeout and cancellation
    - return observable ShellCommandResult

    Non-responsibilities:
    - no arbitrary shell execution
    - no approval UI
    - no hidden subprocess calls from cognition
    - no file mutation bypass
    - no policy bypass
    """

    def __init__(
        self,
        *,
        config: SafeShellRuntimeConfig | None = None,
        registry: ToolRegistry | None = None,
        shell_policy: ShellCommandPolicy | None = None,
        validator: ActionValidator | None = None,
        runner: ShellProcessRunner | None = None,
    ) -> None:
        self._config = config or SafeShellRuntimeConfig()
        self._config.validate()

        self._registry = registry or ToolRegistry()
        self._shell_policy = shell_policy or ShellCommandPolicy()
        self._runner = runner or SubprocessShellRunner()

        if self._config.register_default_shell_tool:
            self._register_default_shell_tool()

        self._validator = validator or ActionValidator(
            config=ActionValidatorConfig(require_policy_evaluation=False),
            registry=self._registry,
        )
        self._lock = RLock()
        self._execution_count = 0
        self._success_count = 0
        self._blocked_count = 0
        self._failed_count = 0
        self._timeout_count = 0
        self._cancelled_count = 0
        self._last_status: ActionStatus | None = None
        self._last_reason: ShellCommandReason | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def execute(
        self,
        request: ShellCommandRequest,
        *,
        cancellation_event: Event | None = None,
    ) -> ShellCommandResult:
        """
        Execute a shell command only after policy and validation pass.
        """

        with self._lock:
            self._execution_count += 1
            self._last_error = None

        started = utc_now()
        monotonic_start = time.monotonic()

        try:
            cwd = self._resolve_working_directory(request)
            policy_result = self._shell_policy.evaluate(request)

            if policy_result.decision != ShellCommandDecision.ALLOW:
                result = self._blocked_result(
                    request=request,
                    policy_result=policy_result,
                    status=ActionStatus.BLOCKED,
                    reason=policy_result.reason,
                    started_at=started,
                    monotonic_start=monotonic_start,
                )
                self._record(result)

                return result

            plan = self._build_plan(
                request=request,
                policy_result=policy_result,
            )
            validation = self._validator.validate_plan(plan)

            if validation.decision != ActionValidationDecision.ALLOW:
                result = self._blocked_result(
                    request=request,
                    policy_result=policy_result,
                    status=ActionStatus.BLOCKED,
                    reason=ShellCommandReason.VALIDATION_BLOCKED,
                    started_at=started,
                    monotonic_start=monotonic_start,
                    validation_result=validation,
                )
                self._record(result)

                return result

            outcome = self._runner.run(
                argv=policy_result.argv,
                cwd=cwd,
                timeout_ms=request.timeout_ms,
                cancellation_event=cancellation_event,
            )
            result = self._result_from_outcome(
                request=request,
                policy_result=policy_result,
                validation_result=validation,
                outcome=outcome,
                started_at=started,
                monotonic_start=monotonic_start,
            )
            self._record(result)

            return result

        except Exception as exc:
            policy_result = ShellCommandPolicyResult(
                decision=ShellCommandDecision.DENY,
                permission_decision=PermissionDecision.DENY,
                reason=ShellCommandReason.EXECUTION_FAILED,
                risk=ActionRisk.HIGH,
                explanation=f"{type(exc).__name__}: {exc}",
            )
            result = self._blocked_result(
                request=request,
                policy_result=policy_result,
                status=ActionStatus.FAILED,
                reason=ShellCommandReason.EXECUTION_FAILED,
                started_at=started,
                monotonic_start=monotonic_start,
                stderr=f"{type(exc).__name__}: {exc}",
            )
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._record(result)

            return result

    def snapshot(self) -> SafeShellRuntimeSnapshot:
        """
        Return runtime diagnostics.
        """

        with self._lock:
            return SafeShellRuntimeSnapshot(
                name=self.name,
                execution_count=self._execution_count,
                success_count=self._success_count,
                blocked_count=self._blocked_count,
                failed_count=self._failed_count,
                timeout_count=self._timeout_count,
                cancelled_count=self._cancelled_count,
                last_status=self._last_status,
                last_reason=self._last_reason,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Reset diagnostics only.
        """

        with self._lock:
            self._execution_count = 0
            self._success_count = 0
            self._blocked_count = 0
            self._failed_count = 0
            self._timeout_count = 0
            self._cancelled_count = 0
            self._last_status = None
            self._last_reason = None
            self._last_error = None

    def _register_default_shell_tool(self) -> None:
        self._registry.register(
            ToolDescriptor(
                tool_id="tool_safe_shell",
                name="safe shell runtime",
                description="Governed shell execution runtime",
                capabilities=(ToolCapability.RUN_SHELL_COMMAND,),
                supported_action_kinds=(ActionKind.SHELL_COMMAND,),
                scopes=(ActionScope.SHELL,),
                max_risk=ActionRisk.HIGH,
                required_permission=PermissionDecision.REQUIRE_APPROVAL,
                availability=ToolAvailability.AVAILABLE,
                health=ToolHealth.HEALTHY,
                enabled=True,
            )
        )

    def _build_plan(
        self,
        *,
        request: ShellCommandRequest,
        policy_result: ShellCommandPolicyResult,
    ) -> ActionPlan:
        step = ActionStep(
            action_id=request.action_id,
            order=0,
            kind=ActionKind.SHELL_COMMAND,
            capability=ToolCapability.RUN_SHELL_COMMAND,
            scope=ActionScope.SHELL,
            risk=policy_result.risk,
            description="execute governed safe shell command",
            arguments={"command": request.command},
            timeout_ms=request.timeout_ms,
            interruptible=True,
            rollback_supported=False,
        )

        return ActionPlan(
            action_id=request.action_id,
            goal=f"execute safe shell command: {request.command}",
            steps=(step,),
            risk=policy_result.risk,
            scope=ActionScope.SHELL,
            requires_approval=False,
            permission_decision=policy_result.permission_decision,
            status=ActionStatus.PLANNED,
        )

    def _resolve_working_directory(self, request: ShellCommandRequest) -> Path:
        root = Path(self._config.workspace_root).resolve()
        cwd = Path(request.working_directory or root).resolve()

        if not self._is_within_root(cwd, root):
            raise ValueError("working directory must stay inside workspace root.")

        return cwd

    @staticmethod
    def _is_within_root(path: Path, root: Path) -> bool:
        return path == root or root in path.parents

    def _blocked_result(
        self,
        *,
        request: ShellCommandRequest,
        policy_result: ShellCommandPolicyResult,
        status: ActionStatus,
        reason: ShellCommandReason,
        started_at: object,
        monotonic_start: float,
        validation_result: ActionValidationResult | None = None,
        stderr: str = "",
    ) -> ShellCommandResult:
        return ShellCommandResult(
            action_id=request.action_id,
            command=request.command,
            status=status,
            success=False,
            exit_code=None,
            stdout="",
            stderr=stderr,
            policy_result=policy_result,
            validation_result=validation_result,
            started_at=started_at,
            completed_at=utc_now(),
            duration_ms=self._duration_ms(monotonic_start),
            metadata={
                "runtime": self.name,
                "reason": reason.value,
            },
        )

    def _result_from_outcome(
        self,
        *,
        request: ShellCommandRequest,
        policy_result: ShellCommandPolicyResult,
        validation_result: ActionValidationResult,
        outcome: ShellProcessOutcome,
        started_at: object,
        monotonic_start: float,
    ) -> ShellCommandResult:
        if outcome.cancelled:
            status = ActionStatus.CANCELLED
            reason = ShellCommandReason.CANCELLED

        elif outcome.timed_out:
            status = ActionStatus.FAILED
            reason = ShellCommandReason.TIMEOUT_OCCURRED

        elif outcome.exit_code == 0:
            status = ActionStatus.SUCCEEDED
            reason = ShellCommandReason.EXECUTION_SUCCEEDED

        else:
            status = ActionStatus.FAILED
            reason = ShellCommandReason.EXECUTION_FAILED

        return ShellCommandResult(
            action_id=request.action_id,
            command=request.command,
            status=status,
            success=status == ActionStatus.SUCCEEDED,
            exit_code=outcome.exit_code,
            stdout=self._truncate(outcome.stdout),
            stderr=self._truncate(outcome.stderr),
            timed_out=outcome.timed_out,
            cancelled=outcome.cancelled,
            policy_result=policy_result,
            validation_result=validation_result,
            started_at=started_at,
            completed_at=utc_now(),
            duration_ms=self._duration_ms(monotonic_start),
            metadata={
                "runtime": self.name,
                "reason": reason.value,
            },
        )

    def _record(self, result: ShellCommandResult) -> None:
        with self._lock:
            self._last_status = result.status
            self._last_reason = ShellCommandReason(
                str(result.metadata.get("reason", ShellCommandReason.EXECUTION_FAILED))
            )

            if result.success:
                self._success_count += 1

            elif result.status == ActionStatus.BLOCKED:
                self._blocked_count += 1

            elif result.cancelled:
                self._cancelled_count += 1

            elif result.timed_out:
                self._timeout_count += 1

            else:
                self._failed_count += 1

    def _truncate(self, value: str) -> str:
        if len(value) <= self._config.max_output_chars:
            return value

        return value[: self._config.max_output_chars] + "\n...[truncated]"

    @staticmethod
    def _duration_ms(start: float) -> int:
        return max(0, int((time.monotonic() - start) * 1000))