from __future__ import annotations

import difflib
import re
import time
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from typing import Protocol

from pydantic import Field, field_validator, model_validator

from jarvis.tools.filesystem import (
    FileOperationKind,
    FileOperationRequest,
    FileOperationResult,
    FileSystemRuntime,
)
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
from jarvis.tools.shell import (
    SafeShellRuntime,
    ShellCommandRequest,
    ShellCommandResult,
)
from jarvis.tools.validation import (
    ActionValidationDecision,
    ActionValidationResult,
    ActionValidator,
    ActionValidatorConfig,
)


class IdeActionKind(StrEnum):
    """
    Supported governed IDE/editor action kinds.
    """

    OPEN_FILE = "open_file"
    OPEN_SYMBOL = "open_symbol"
    SHOW_DIAGNOSTICS = "show_diagnostics"
    PREPARE_PATCH = "prepare_patch"
    APPLY_PATCH = "apply_patch"
    RUN_TESTS = "run_tests"
    NAVIGATE_PROJECT = "navigate_project"


class IdeActionDecision(StrEnum):
    """
    IDE runtime decision.
    """

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_CONFIRMATION = "require_confirmation"
    REQUIRE_APPROVAL = "require_approval"


class IdeActionReason(StrEnum):
    """
    Machine-readable IDE runtime reason.
    """

    SAFE_OPEN_ALLOWED = "safe_open_allowed"
    SAFE_SYMBOL_LOOKUP_ALLOWED = "safe_symbol_lookup_allowed"
    SAFE_DIAGNOSTICS_ALLOWED = "safe_diagnostics_allowed"
    SAFE_NAVIGATION_ALLOWED = "safe_navigation_allowed"
    PATCH_PREPARED = "patch_prepared"
    PATCH_REQUIRES_APPROVAL = "patch_requires_approval"
    PATCH_APPLIED = "patch_applied"
    TESTS_STARTED = "tests_started"
    TEST_COMMAND_NOT_ALLOWED = "test_command_not_allowed"
    APPROVAL_MISSING = "approval_missing"
    CONFIRMATION_MISSING = "confirmation_missing"
    PATH_REQUIRED = "path_required"
    SYMBOL_REQUIRED = "symbol_required"
    PATCH_PAYLOAD_REQUIRED = "patch_payload_required"
    VALIDATION_BLOCKED = "validation_blocked"
    ACTION_SUCCEEDED = "action_succeeded"
    ACTION_FAILED = "action_failed"


class IdeActionRequest(ToolModel):
    """
    Typed request for a governed IDE/editor action.

    This runtime does not silently edit files. Patch application is approval
    gated and delegated to FileSystemRuntime.
    """

    action_id: str = Field(default_factory=new_action_id)
    kind: IdeActionKind
    path: str | None = None
    symbol: str | None = None
    old_text: str | None = None
    new_text: str | None = None
    test_command: str | None = None
    recursive: bool = False
    confirmed: bool = False
    approved: bool = False
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("action_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("action_id cannot be empty.")

        return cleaned

    @field_validator("path", "symbol", "old_text", "new_text", "test_command")
    @classmethod
    def _clean_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None

    @model_validator(mode="after")
    def _validate_payload(self) -> IdeActionRequest:
        if self.kind in {
            IdeActionKind.OPEN_FILE,
            IdeActionKind.SHOW_DIAGNOSTICS,
            IdeActionKind.PREPARE_PATCH,
            IdeActionKind.APPLY_PATCH,
        } and self.path is None:
            raise ValueError("path is required for this IDE action.")

        if self.kind == IdeActionKind.OPEN_SYMBOL and self.symbol is None:
            raise ValueError("symbol is required for open symbol.")

        if self.kind in {
            IdeActionKind.PREPARE_PATCH,
            IdeActionKind.APPLY_PATCH,
        } and (self.old_text is None or self.new_text is None):
            raise ValueError("old_text and new_text are required for patch actions.")

        if self.kind == IdeActionKind.RUN_TESTS and self.test_command is None:
            raise ValueError("test_command is required for run tests.")

        return self


class IdeActionPolicyResult(ToolModel):
    """
    IDE policy result.
    """

    decision: IdeActionDecision
    permission_decision: PermissionDecision
    reason: IdeActionReason
    risk: ActionRisk
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

    @property
    def allowed(self) -> bool:
        return self.decision == IdeActionDecision.ALLOW


class IdeActionResult(ToolModel):
    """
    Observable result of a governed IDE/editor action.
    """

    result_id: str = Field(default_factory=new_action_result_id)
    action_id: str
    kind: IdeActionKind
    status: ActionStatus
    success: bool
    decision: IdeActionDecision
    reason: IdeActionReason
    output: str
    path: str | None = None
    symbol: str | None = None
    diff: str | None = None
    file_result: FileOperationResult | None = None
    shell_result: ShellCommandResult | None = None
    policy_result: IdeActionPolicyResult
    validation_result: ActionValidationResult | None = None
    started_at: object = Field(default_factory=utc_now)
    completed_at: object = Field(default_factory=utc_now)
    duration_ms: int = Field(default=0, ge=0)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("result_id", "action_id", "output")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class IdeRuntimeConfig:
    """
    Configuration for IdeRuntime.
    """

    name: str = "ide_runtime"
    workspace_root: str = "."
    register_default_ide_tool: bool = True
    allowed_test_commands: tuple[str, ...] = (
        "pytest",
        "ruff check .",
        "mypy .",
    )

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if not self.workspace_root.strip():
            raise ValueError("workspace_root cannot be empty.")

        if not self.allowed_test_commands:
            raise ValueError("allowed_test_commands cannot be empty.")


@dataclass(frozen=True, slots=True)
class IdeRuntimeSnapshot:
    """
    Observable diagnostics for IdeRuntime.
    """

    name: str
    action_count: int
    success_count: int
    blocked_count: int
    failed_count: int
    approval_required_count: int
    confirmation_required_count: int
    last_status: ActionStatus | None
    last_reason: IdeActionReason | None
    last_error: str | None


class EditorLauncher(Protocol):
    """
    Editor launcher protocol.

    Tests can inject a fake launcher. Production uses LoggingEditorLauncher,
    which intentionally does not perform hidden editor control.
    """

    def open_file(self, path: str) -> bool:
        ...

    def open_symbol(self, symbol: str, path: str | None = None) -> bool:
        ...


class LoggingEditorLauncher:
    """
    Safe placeholder editor launcher.

    It records intent through the runtime result, but does not secretly control
    an editor. A real VS Code adapter can be added later behind this protocol.
    """

    def open_file(self, path: str) -> bool:
        del path

        return True

    def open_symbol(self, symbol: str, path: str | None = None) -> bool:
        del symbol, path

        return True


class IdePolicy:
    """
    Conservative IDE/editor policy.

    Read/navigation/diagnostics are allowed. Patch preparation is allowed
    because it only produces a diff. Patch application requires approval.
    Tests are allowed only if they match the configured safe test commands.
    """

    def __init__(self, *, allowed_test_commands: tuple[str, ...]) -> None:
        self._allowed_test_commands = tuple(
            command.strip() for command in allowed_test_commands
        )

    def evaluate(self, request: IdeActionRequest) -> IdeActionPolicyResult:
        if request.kind == IdeActionKind.OPEN_FILE:
            return self._allow(
                reason=IdeActionReason.SAFE_OPEN_ALLOWED,
                explanation="opening a workspace file is read-oriented",
            )

        if request.kind == IdeActionKind.OPEN_SYMBOL:
            return self._allow(
                reason=IdeActionReason.SAFE_SYMBOL_LOOKUP_ALLOWED,
                explanation="symbol lookup is read-oriented",
            )

        if request.kind == IdeActionKind.SHOW_DIAGNOSTICS:
            return self._allow(
                reason=IdeActionReason.SAFE_DIAGNOSTICS_ALLOWED,
                explanation="diagnostics are read-oriented",
            )

        if request.kind == IdeActionKind.NAVIGATE_PROJECT:
            return self._allow(
                reason=IdeActionReason.SAFE_NAVIGATION_ALLOWED,
                explanation="project navigation is read-oriented",
            )

        if request.kind == IdeActionKind.PREPARE_PATCH:
            return self._allow(
                reason=IdeActionReason.PATCH_PREPARED,
                explanation="patch preparation only produces a diff",
                risk=ActionRisk.MEDIUM,
                permission=PermissionDecision.REQUIRE_CONFIRMATION,
            )

        if request.kind == IdeActionKind.APPLY_PATCH:
            if not request.approved:
                return IdeActionPolicyResult(
                    decision=IdeActionDecision.REQUIRE_APPROVAL,
                    permission_decision=PermissionDecision.REQUIRE_APPROVAL,
                    reason=IdeActionReason.PATCH_REQUIRES_APPROVAL,
                    risk=ActionRisk.HIGH,
                    explanation="applying a patch requires explicit approval",
                )

            return self._allow(
                reason=IdeActionReason.PATCH_APPLIED,
                explanation="approved patch application is allowed",
                risk=ActionRisk.HIGH,
                permission=PermissionDecision.REQUIRE_APPROVAL,
            )

        if request.kind == IdeActionKind.RUN_TESTS:
            command = request.test_command or ""

            if command not in self._allowed_test_commands:
                return IdeActionPolicyResult(
                    decision=IdeActionDecision.DENY,
                    permission_decision=PermissionDecision.DENY,
                    reason=IdeActionReason.TEST_COMMAND_NOT_ALLOWED,
                    risk=ActionRisk.HIGH,
                    explanation="test command is not in the IDE allowlist",
                )

            return self._allow(
                reason=IdeActionReason.TESTS_STARTED,
                explanation="safe test command is allowed through SafeShellRuntime",
            )

        return IdeActionPolicyResult(
            decision=IdeActionDecision.DENY,
            permission_decision=PermissionDecision.DENY,
            reason=IdeActionReason.ACTION_FAILED,
            risk=ActionRisk.HIGH,
            explanation="unsupported IDE action",
        )

    @staticmethod
    def _allow(
        *,
        reason: IdeActionReason,
        explanation: str,
        risk: ActionRisk = ActionRisk.LOW,
        permission: PermissionDecision = PermissionDecision.ALLOW,
    ) -> IdeActionPolicyResult:
        return IdeActionPolicyResult(
            decision=IdeActionDecision.ALLOW,
            permission_decision=permission,
            reason=reason,
            risk=risk,
            explanation=explanation,
        )


class IdeRuntime:
    """
    Governed IDE/editor runtime.

    Responsibilities:
    - open files and symbols visibly
    - show diagnostics through safe read/search paths
    - prepare patches as diffs
    - apply patches only through FileSystemRuntime with approval
    - run tests only through SafeShellRuntime
    - produce observable audit-ready results

    Non-responsibilities:
    - no hidden editing
    - no direct subprocess execution
    - no raw filesystem writes
    - no bypass of policy or validation
    """

    def __init__(
        self,
        *,
        config: IdeRuntimeConfig | None = None,
        registry: ToolRegistry | None = None,
        policy: IdePolicy | None = None,
        validator: ActionValidator | None = None,
        file_runtime: FileSystemRuntime | None = None,
        shell_runtime: SafeShellRuntime | None = None,
        editor_launcher: EditorLauncher | None = None,
    ) -> None:
        self._config = config or IdeRuntimeConfig()
        self._config.validate()

        self._registry = registry or ToolRegistry()
        self._policy = policy or IdePolicy(
            allowed_test_commands=self._config.allowed_test_commands
        )
        self._file_runtime = file_runtime or FileSystemRuntime()
        self._shell_runtime = shell_runtime or SafeShellRuntime()
        self._editor_launcher: EditorLauncher = (
            editor_launcher or LoggingEditorLauncher()
        )

        if self._config.register_default_ide_tool:
            self._register_default_ide_tool()

        self._validator = validator or ActionValidator(
            config=ActionValidatorConfig(require_policy_evaluation=False),
            registry=self._registry,
        )
        self._lock = RLock()

        self._action_count = 0
        self._success_count = 0
        self._blocked_count = 0
        self._failed_count = 0
        self._approval_required_count = 0
        self._confirmation_required_count = 0
        self._last_status: ActionStatus | None = None
        self._last_reason: IdeActionReason | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def execute(self, request: IdeActionRequest) -> IdeActionResult:
        """
        Execute one governed IDE/editor action.
        """

        with self._lock:
            self._action_count += 1
            self._last_error = None

        started = utc_now()
        monotonic_start = time.monotonic()

        try:
            policy_result = self._policy.evaluate(request)

            if policy_result.decision != IdeActionDecision.ALLOW:
                result = self._blocked_result(
                    request=request,
                    policy_result=policy_result,
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

            if validation.decision == ActionValidationDecision.BLOCK:
                result = self._blocked_result(
                    request=request,
                    policy_result=policy_result,
                    validation_result=validation,
                    reason=IdeActionReason.VALIDATION_BLOCKED,
                    started_at=started,
                    monotonic_start=monotonic_start,
                )
                self._record(result)

                return result

            result = self._execute_allowed(
                request=request,
                policy_result=policy_result,
                validation_result=validation,
                started_at=started,
                monotonic_start=monotonic_start,
            )
            self._record(result)

            return result

        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            policy_result = IdeActionPolicyResult(
                decision=IdeActionDecision.DENY,
                permission_decision=PermissionDecision.DENY,
                reason=IdeActionReason.ACTION_FAILED,
                risk=ActionRisk.HIGH,
                explanation=f"{type(exc).__name__}: {exc}",
            )
            result = self._failed_result(
                request=request,
                policy_result=policy_result,
                output=f"{type(exc).__name__}: {exc}",
                started_at=started,
                monotonic_start=monotonic_start,
            )
            self._record(result)

            return result

    def snapshot(self) -> IdeRuntimeSnapshot:
        """
        Return runtime diagnostics.
        """

        with self._lock:
            return IdeRuntimeSnapshot(
                name=self.name,
                action_count=self._action_count,
                success_count=self._success_count,
                blocked_count=self._blocked_count,
                failed_count=self._failed_count,
                approval_required_count=self._approval_required_count,
                confirmation_required_count=self._confirmation_required_count,
                last_status=self._last_status,
                last_reason=self._last_reason,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Reset runtime diagnostics.
        """

        with self._lock:
            self._action_count = 0
            self._success_count = 0
            self._blocked_count = 0
            self._failed_count = 0
            self._approval_required_count = 0
            self._confirmation_required_count = 0
            self._last_status = None
            self._last_reason = None
            self._last_error = None

    def _execute_allowed(
        self,
        *,
        request: IdeActionRequest,
        policy_result: IdeActionPolicyResult,
        validation_result: ActionValidationResult,
        started_at: object,
        monotonic_start: float,
    ) -> IdeActionResult:
        if request.kind == IdeActionKind.OPEN_FILE:
            return self._open_file(
                request=request,
                policy_result=policy_result,
                validation_result=validation_result,
                started_at=started_at,
                monotonic_start=monotonic_start,
            )

        if request.kind == IdeActionKind.OPEN_SYMBOL:
            return self._open_symbol(
                request=request,
                policy_result=policy_result,
                validation_result=validation_result,
                started_at=started_at,
                monotonic_start=monotonic_start,
            )

        if request.kind == IdeActionKind.SHOW_DIAGNOSTICS:
            return self._show_diagnostics(
                request=request,
                policy_result=policy_result,
                validation_result=validation_result,
                started_at=started_at,
                monotonic_start=monotonic_start,
            )

        if request.kind == IdeActionKind.PREPARE_PATCH:
            return self._prepare_patch(
                request=request,
                policy_result=policy_result,
                validation_result=validation_result,
                started_at=started_at,
                monotonic_start=monotonic_start,
            )

        if request.kind == IdeActionKind.APPLY_PATCH:
            return self._apply_patch(
                request=request,
                policy_result=policy_result,
                validation_result=validation_result,
                started_at=started_at,
                monotonic_start=monotonic_start,
            )

        if request.kind == IdeActionKind.RUN_TESTS:
            return self._run_tests(
                request=request,
                policy_result=policy_result,
                validation_result=validation_result,
                started_at=started_at,
                monotonic_start=monotonic_start,
            )

        if request.kind == IdeActionKind.NAVIGATE_PROJECT:
            return self._navigate_project(
                request=request,
                policy_result=policy_result,
                validation_result=validation_result,
                started_at=started_at,
                monotonic_start=monotonic_start,
            )

        raise ValueError(f"unsupported IDE action: {request.kind.value}")

    def _open_file(
        self,
        *,
        request: IdeActionRequest,
        policy_result: IdeActionPolicyResult,
        validation_result: ActionValidationResult,
        started_at: object,
        monotonic_start: float,
    ) -> IdeActionResult:
        path = request.path or ""
        opened = self._editor_launcher.open_file(path)

        if not opened:
            return self._failed_result(
                request=request,
                policy_result=policy_result,
                validation_result=validation_result,
                output="editor launcher failed to open file",
                started_at=started_at,
                monotonic_start=monotonic_start,
            )

        return self._success_result(
            request=request,
            policy_result=policy_result,
            validation_result=validation_result,
            output=f"file open requested: {path}",
            started_at=started_at,
            monotonic_start=monotonic_start,
            path=path,
        )

    def _open_symbol(
        self,
        *,
        request: IdeActionRequest,
        policy_result: IdeActionPolicyResult,
        validation_result: ActionValidationResult,
        started_at: object,
        monotonic_start: float,
    ) -> IdeActionResult:
        symbol = request.symbol or ""
        opened = self._editor_launcher.open_symbol(symbol, request.path)

        if not opened:
            return self._failed_result(
                request=request,
                policy_result=policy_result,
                validation_result=validation_result,
                output="editor launcher failed to open symbol",
                started_at=started_at,
                monotonic_start=monotonic_start,
            )

        return self._success_result(
            request=request,
            policy_result=policy_result,
            validation_result=validation_result,
            output=f"symbol open requested: {symbol}",
            started_at=started_at,
            monotonic_start=monotonic_start,
            path=request.path,
            symbol=symbol,
        )

    def _show_diagnostics(
        self,
        *,
        request: IdeActionRequest,
        policy_result: IdeActionPolicyResult,
        validation_result: ActionValidationResult,
        started_at: object,
        monotonic_start: float,
    ) -> IdeActionResult:
        file_result = self._file_runtime.execute(
            FileOperationRequest(
                action_id=request.action_id,
                kind=FileOperationKind.READ_FILE,
                path=request.path or "",
            )
        )
        diagnostics = self._diagnostics_from_text(file_result.content or "")

        return self._success_result(
            request=request,
            policy_result=policy_result,
            validation_result=validation_result,
            output=diagnostics,
            started_at=started_at,
            monotonic_start=monotonic_start,
            path=request.path,
            file_result=file_result,
        )

    def _prepare_patch(
        self,
        *,
        request: IdeActionRequest,
        policy_result: IdeActionPolicyResult,
        validation_result: ActionValidationResult,
        started_at: object,
        monotonic_start: float,
    ) -> IdeActionResult:
        diff = self._diff(
            before=request.old_text or "",
            after=request.new_text or "",
            path=request.path or "",
        )

        return self._success_result(
            request=request,
            policy_result=policy_result,
            validation_result=validation_result,
            output="patch prepared; no file was modified",
            started_at=started_at,
            monotonic_start=monotonic_start,
            path=request.path,
            diff=diff,
        )

    def _apply_patch(
        self,
        *,
        request: IdeActionRequest,
        policy_result: IdeActionPolicyResult,
        validation_result: ActionValidationResult,
        started_at: object,
        monotonic_start: float,
    ) -> IdeActionResult:
        file_result = self._file_runtime.execute(
            FileOperationRequest(
                action_id=request.action_id,
                kind=FileOperationKind.PATCH_FILE,
                path=request.path or "",
                old_text=request.old_text,
                new_text=request.new_text,
                confirmed=True,
                approved=True,
            )
        )

        if not file_result.success:
            return self._failed_result(
                request=request,
                policy_result=policy_result,
                validation_result=validation_result,
                output=file_result.output,
                started_at=started_at,
                monotonic_start=monotonic_start,
                file_result=file_result,
            )

        return self._success_result(
            request=request,
            policy_result=policy_result,
            validation_result=validation_result,
            output="approved patch applied through FileSystemRuntime",
            started_at=started_at,
            monotonic_start=monotonic_start,
            path=request.path,
            diff=file_result.diff,
            file_result=file_result,
        )

    def _run_tests(
        self,
        *,
        request: IdeActionRequest,
        policy_result: IdeActionPolicyResult,
        validation_result: ActionValidationResult,
        started_at: object,
        monotonic_start: float,
    ) -> IdeActionResult:
        shell_result = self._shell_runtime.execute(
            ShellCommandRequest(
                action_id=request.action_id,
                command=request.test_command or "",
            )
        )

        if not shell_result.success:
            return self._failed_result(
                request=request,
                policy_result=policy_result,
                validation_result=validation_result,
                output=shell_result.stderr or shell_result.stdout,
                started_at=started_at,
                monotonic_start=monotonic_start,
                shell_result=shell_result,
            )

        return self._success_result(
            request=request,
            policy_result=policy_result,
            validation_result=validation_result,
            output=shell_result.stdout or "tests completed successfully",
            started_at=started_at,
            monotonic_start=monotonic_start,
            shell_result=shell_result,
        )

    def _navigate_project(
        self,
        *,
        request: IdeActionRequest,
        policy_result: IdeActionPolicyResult,
        validation_result: ActionValidationResult,
        started_at: object,
        monotonic_start: float,
    ) -> IdeActionResult:
        file_result = self._file_runtime.execute(
            FileOperationRequest(
                action_id=request.action_id,
                kind=FileOperationKind.SEARCH_FILES,
                path=request.path or ".",
                pattern="*.py",
                recursive=request.recursive,
            )
        )

        return self._success_result(
            request=request,
            policy_result=policy_result,
            validation_result=validation_result,
            output=file_result.output or "project navigation completed",
            started_at=started_at,
            monotonic_start=monotonic_start,
            path=request.path,
            file_result=file_result,
        )

    def _build_plan(
        self,
        *,
        request: IdeActionRequest,
        policy_result: IdeActionPolicyResult,
    ) -> ActionPlan:
        kind = self._action_kind(request.kind)
        capability = self._capability(request.kind)
        risk = policy_result.risk
        timeout_ms = 30_000 if risk in {ActionRisk.HIGH, ActionRisk.CRITICAL} else None
        requires_approval = (
            request.approved
            or policy_result.decision == IdeActionDecision.REQUIRE_APPROVAL
            or policy_result.permission_decision == PermissionDecision.REQUIRE_APPROVAL
            or risk in {ActionRisk.HIGH, ActionRisk.CRITICAL}
        )
        arguments = self._arguments_for_request(request)

        step = ActionStep(
            action_id=request.action_id,
            order=0,
            kind=kind,
            capability=capability,
            scope=ActionScope.IDE,
            risk=risk,
            description=f"execute governed IDE action: {request.kind.value}",
            arguments=arguments,
            timeout_ms=timeout_ms,
            interruptible=True,
            rollback_supported=request.kind == IdeActionKind.APPLY_PATCH,
        )

        return ActionPlan(
            action_id=request.action_id,
            goal=f"execute IDE action: {request.kind.value}",
            steps=(step,),
            risk=risk,
            scope=ActionScope.IDE,
            requires_approval=requires_approval,
            permission_decision=policy_result.permission_decision,
            status=ActionStatus.PLANNED,
        )

    def _register_default_ide_tool(self) -> None:
        self._registry.register(
            ToolDescriptor(
                tool_id="tool_ide_runtime",
                name="ide runtime",
                description="Governed IDE/editor assistance runtime",
                capabilities=(
                    ToolCapability.READ_FILE,
                    ToolCapability.SEARCH_FILES,
                    ToolCapability.PATCH_FILE,
                    ToolCapability.APPLY_IDE_PATCH,
                    ToolCapability.RUN_SHELL_COMMAND,
                ),
                supported_action_kinds=(
                    ActionKind.READ,
                    ActionKind.SEARCH,
                    ActionKind.PATCH,
                    ActionKind.IDE_OPEN_FILE,
                    ActionKind.IDE_APPLY_PATCH,
                    ActionKind.SHELL_COMMAND,
                ),
                scopes=(ActionScope.IDE,),
                max_risk=ActionRisk.HIGH,
                required_permission=PermissionDecision.REQUIRE_APPROVAL,
                availability=ToolAvailability.AVAILABLE,
                health=ToolHealth.HEALTHY,
                enabled=True,
            )
        )

    @staticmethod
    def _action_kind(kind: IdeActionKind) -> ActionKind:
        return {
            IdeActionKind.OPEN_FILE: ActionKind.IDE_OPEN_FILE,
            IdeActionKind.OPEN_SYMBOL: ActionKind.READ,
            IdeActionKind.SHOW_DIAGNOSTICS: ActionKind.READ,
            IdeActionKind.PREPARE_PATCH: ActionKind.PATCH,
            IdeActionKind.APPLY_PATCH: ActionKind.IDE_APPLY_PATCH,
            IdeActionKind.RUN_TESTS: ActionKind.SHELL_COMMAND,
            IdeActionKind.NAVIGATE_PROJECT: ActionKind.SEARCH,
        }[kind]

    @staticmethod
    def _capability(kind: IdeActionKind) -> ToolCapability:
        return {
            IdeActionKind.OPEN_FILE: ToolCapability.READ_FILE,
            IdeActionKind.OPEN_SYMBOL: ToolCapability.SEARCH_FILES,
            IdeActionKind.SHOW_DIAGNOSTICS: ToolCapability.READ_FILE,
            IdeActionKind.PREPARE_PATCH: ToolCapability.PATCH_FILE,
            IdeActionKind.APPLY_PATCH: ToolCapability.APPLY_IDE_PATCH,
            IdeActionKind.RUN_TESTS: ToolCapability.RUN_SHELL_COMMAND,
            IdeActionKind.NAVIGATE_PROJECT: ToolCapability.SEARCH_FILES,
        }[kind]

    @staticmethod
    def _arguments_for_request(request: IdeActionRequest) -> dict[str, object]:
        arguments: dict[str, object] = {}

        if request.path is not None:
            arguments["path"] = request.path

        if request.symbol is not None:
            arguments["symbol"] = request.symbol

        if request.test_command is not None:
            arguments["command"] = request.test_command

        return arguments

    @staticmethod
    def _diagnostics_from_text(text: str) -> str:
        if not text:
            return "No diagnostics: file content was empty or unavailable."

        diagnostics: list[str] = []

        for line_number, line in enumerate(text.splitlines(), start=1):
            lowered = line.casefold()

            if "todo" in lowered or "fixme" in lowered:
                diagnostics.append(f"line {line_number}: {line.strip()}")

            if re.search(r"\bprint\(", line):
                diagnostics.append(f"line {line_number}: print statement found")

        if not diagnostics:
            return "No lightweight diagnostics found."

        return "\n".join(diagnostics)

    @staticmethod
    def _diff(*, before: str, after: str, path: str) -> str:
        return "\n".join(
            difflib.unified_diff(
                before.splitlines(),
                after.splitlines(),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                lineterm="",
            )
        )

    def _success_result(
        self,
        *,
        request: IdeActionRequest,
        policy_result: IdeActionPolicyResult,
        validation_result: ActionValidationResult,
        output: str,
        started_at: object,
        monotonic_start: float,
        path: str | None = None,
        symbol: str | None = None,
        diff: str | None = None,
        file_result: FileOperationResult | None = None,
        shell_result: ShellCommandResult | None = None,
    ) -> IdeActionResult:
        return IdeActionResult(
            action_id=request.action_id,
            kind=request.kind,
            status=ActionStatus.SUCCEEDED,
            success=True,
            decision=IdeActionDecision.ALLOW,
            reason=IdeActionReason.ACTION_SUCCEEDED,
            output=output,
            path=path,
            symbol=symbol,
            diff=diff,
            file_result=file_result,
            shell_result=shell_result,
            policy_result=policy_result,
            validation_result=validation_result,
            started_at=started_at,
            completed_at=utc_now(),
            duration_ms=self._duration_ms(monotonic_start),
            metadata={
                "runtime": self.name,
                "reason": IdeActionReason.ACTION_SUCCEEDED.value,
            },
        )

    def _blocked_result(
        self,
        *,
        request: IdeActionRequest,
        policy_result: IdeActionPolicyResult,
        started_at: object,
        monotonic_start: float,
        validation_result: ActionValidationResult | None = None,
        reason: IdeActionReason | None = None,
    ) -> IdeActionResult:
        final_reason = reason or policy_result.reason

        return IdeActionResult(
            action_id=request.action_id,
            kind=request.kind,
            status=ActionStatus.BLOCKED,
            success=False,
            decision=policy_result.decision,
            reason=final_reason,
            output=policy_result.explanation,
            path=request.path,
            symbol=request.symbol,
            policy_result=policy_result,
            validation_result=validation_result,
            started_at=started_at,
            completed_at=utc_now(),
            duration_ms=self._duration_ms(monotonic_start),
            metadata={
                "runtime": self.name,
                "reason": final_reason.value,
            },
        )

    def _failed_result(
        self,
        *,
        request: IdeActionRequest,
        policy_result: IdeActionPolicyResult,
        output: str,
        started_at: object,
        monotonic_start: float,
        validation_result: ActionValidationResult | None = None,
        file_result: FileOperationResult | None = None,
        shell_result: ShellCommandResult | None = None,
    ) -> IdeActionResult:
        return IdeActionResult(
            action_id=request.action_id,
            kind=request.kind,
            status=ActionStatus.FAILED,
            success=False,
            decision=IdeActionDecision.DENY,
            reason=IdeActionReason.ACTION_FAILED,
            output=output or "IDE action failed",
            path=request.path,
            symbol=request.symbol,
            file_result=file_result,
            shell_result=shell_result,
            policy_result=policy_result,
            validation_result=validation_result,
            started_at=started_at,
            completed_at=utc_now(),
            duration_ms=self._duration_ms(monotonic_start),
            metadata={
                "runtime": self.name,
                "reason": IdeActionReason.ACTION_FAILED.value,
            },
        )

    def _record(self, result: IdeActionResult) -> None:
        with self._lock:
            self._last_status = result.status
            self._last_reason = result.reason

            if result.success:
                self._success_count += 1

            elif result.decision == IdeActionDecision.REQUIRE_APPROVAL:
                self._approval_required_count += 1
                self._blocked_count += 1

            elif result.decision == IdeActionDecision.REQUIRE_CONFIRMATION:
                self._confirmation_required_count += 1
                self._blocked_count += 1

            elif result.status == ActionStatus.BLOCKED:
                self._blocked_count += 1

            else:
                self._failed_count += 1

    @staticmethod
    def _duration_ms(start: float) -> int:
        return max(0, int((time.monotonic() - start) * 1000))