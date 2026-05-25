from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import PurePath
from threading import RLock

from pydantic import Field, field_validator

from jarvis.tools.ids import new_action_result_id, utc_now
from jarvis.tools.models import (
    ActionKind,
    ActionPlan,
    ActionRisk,
    ActionScope,
    ActionStep,
    PermissionDecision,
    ToolModel,
)
from jarvis.tools.policy import (
    PermissionPolicy,
    PermissionPolicyEvaluation,
)
from jarvis.tools.registry import (
    ToolDescriptor,
    ToolLookupStatus,
    ToolRegistry,
)


class ActionValidationSeverity(StrEnum):
    """
    Severity of a validation finding.
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ActionValidationReason(StrEnum):
    """
    Machine-readable validation reason.
    """

    VALIDATION_PASSED = "validation_passed"
    EMPTY_PLAN = "empty_plan"
    INVALID_STEP_ORDER = "invalid_step_order"
    ACTION_ID_MISMATCH = "action_id_mismatch"
    UNKNOWN_TOOL = "unknown_tool"
    TOOL_DISABLED = "tool_disabled"
    TOOL_UNAVAILABLE = "tool_unavailable"
    TOOL_CAPABILITY_MISMATCH = "tool_capability_mismatch"
    TOOL_SCOPE_MISMATCH = "tool_scope_mismatch"
    TOOL_ACTION_KIND_MISMATCH = "tool_action_kind_mismatch"
    TOOL_RISK_EXCEEDED = "tool_risk_exceeded"
    POLICY_DENIED = "policy_denied"
    POLICY_BYPASS_ATTEMPT = "policy_bypass_attempt"
    APPROVAL_REQUIRED = "approval_required"
    CONFIRMATION_REQUIRED = "confirmation_required"
    UNSAFE_PATH = "unsafe_path"
    ABSOLUTE_PATH_BLOCKED = "absolute_path_blocked"
    PATH_TRAVERSAL_BLOCKED = "path_traversal_blocked"
    MISSING_PATH_ARGUMENT = "missing_path_argument"
    UNSAFE_COMMAND = "unsafe_command"
    MISSING_COMMAND_ARGUMENT = "missing_command_argument"
    INVALID_ARGUMENTS = "invalid_arguments"
    RISK_ESCALATION = "risk_escalation"
    MISSING_TIMEOUT = "missing_timeout"
    NON_INTERRUPTIBLE_UNSAFE_ACTION = "non_interruptible_unsafe_action"
    MISSING_ROLLBACK = "missing_rollback"


class ActionValidationDecision(StrEnum):
    """
    Final validation decision.
    """

    ALLOW = "allow"
    BLOCK = "block"
    REQUIRE_APPROVAL = "require_approval"
    REQUIRE_CONFIRMATION = "require_confirmation"


class ActionValidationFinding(ToolModel):
    """
    One validation finding.

    Findings are structured so future observability/audit systems can explain
    exactly why an action was allowed, blocked, or gated.
    """

    finding_id: str = Field(default_factory=new_action_result_id)
    action_id: str
    step_id: str | None = None
    severity: ActionValidationSeverity
    reason: ActionValidationReason
    message: str
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("finding_id", "action_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("step_id")
    @classmethod
    def _clean_step_id(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None


class ActionValidationResult(ToolModel):
    """
    Final validation result for one action plan.
    """

    validation_id: str = Field(default_factory=new_action_result_id)
    action_id: str
    plan_id: str
    decision: ActionValidationDecision
    valid: bool
    blocked: bool
    requires_approval: bool = False
    requires_confirmation: bool = False
    findings: tuple[ActionValidationFinding, ...]
    policy_evaluation: PermissionPolicyEvaluation | None = None
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("validation_id", "action_id", "plan_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @property
    def error_count(self) -> int:
        return sum(
            1
            for finding in self.findings
            if finding.severity
            in {
                ActionValidationSeverity.ERROR,
                ActionValidationSeverity.CRITICAL,
            }
        )

    @property
    def warning_count(self) -> int:
        return sum(
            1
            for finding in self.findings
            if finding.severity == ActionValidationSeverity.WARNING
        )


@dataclass(frozen=True, slots=True)
class ActionValidatorConfig:
    """
    Configuration for the action validation layer.

    Defaults are conservative because this layer protects future execution.
    """

    name: str = "action_validator"
    block_absolute_paths: bool = True
    block_path_traversal: bool = True
    require_tool_registration: bool = True
    require_policy_evaluation: bool = True
    require_timeout_for_high_risk: bool = True
    require_rollback_for_destructive: bool = False
    require_interruptible_mutations: bool = True
    max_argument_string_length: int = 8_000

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.max_argument_string_length <= 0:
            raise ValueError("max_argument_string_length must be positive.")


@dataclass(frozen=True, slots=True)
class ActionValidatorSnapshot:
    """
    Observable diagnostics for ActionValidator.
    """

    name: str
    validation_count: int
    allowed_count: int
    blocked_count: int
    approval_required_count: int
    confirmation_required_count: int
    last_decision: ActionValidationDecision | None
    last_error: str | None


class ActionValidator:
    """
    Validation wall before action execution.

    Responsibilities:
    - validate typed action plans before execution
    - verify registered tool availability and compatibility
    - enforce policy decisions
    - block dangerous shell commands
    - block unsafe paths
    - detect risk escalation
    - enforce timeout, approval, rollback, and interruptibility requirements

    Non-responsibilities:
    - no tool execution
    - no shell execution
    - no file mutation
    - no approval UI
    - no policy replacement
    """

    _PATH_ARGUMENT_KEYS = (
        "path",
        "file",
        "target",
        "target_path",
        "source",
        "source_path",
        "destination",
        "destination_path",
        "workspace_path",
        "project_path",
    )

    _SHELL_ARGUMENT_KEYS = (
        "command",
        "cmd",
        "script",
    )

    _DANGEROUS_COMMAND_PATTERNS = (
        r"\brm\s+-rf\b",
        r"\brm\s+/[A-Za-z0-9_./\\-]*",
        r"\bdel\s+/[fqsa]\b",
        r"\brmdir\s+/s\b",
        r"\bformat\b",
        r"\bshutdown\b",
        r"\brestart-computer\b",
        r"\bstop-computer\b",
        r"\bset-executionpolicy\b",
        r"\breg\s+(add|delete|import|restore)\b",
        r"\btakeown\b",
        r"\bicacls\b",
        r"\bdiskpart\b",
        r"\bbcdedit\b",
        r"\binvoke-expression\b",
        r"\biex\b",
        r"\binvoke-webrequest\b",
        r"\biwr\b",
        r"\bcurl\b",
        r"\bwget\b",
        r">\s*\$profile\b",
    )

    def __init__(
        self,
        *,
        config: ActionValidatorConfig | None = None,
        registry: ToolRegistry | None = None,
        policy: PermissionPolicy | None = None,
    ) -> None:
        self._config = config or ActionValidatorConfig()
        self._config.validate()

        self._registry = registry or ToolRegistry()
        self._policy = policy or PermissionPolicy()
        self._lock = RLock()

        self._validation_count = 0
        self._allowed_count = 0
        self._blocked_count = 0
        self._approval_required_count = 0
        self._confirmation_required_count = 0
        self._last_decision: ActionValidationDecision | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def validate_plan(self, plan: ActionPlan) -> ActionValidationResult:
        """
        Validate a full action plan before execution.

        This is the main entrypoint for Step 4.
        """

        with self._lock:
            self._validation_count += 1
            self._last_error = None

        try:
            findings: list[ActionValidationFinding] = []

            findings.extend(self._validate_plan_shape(plan))

            tool_by_step = self._resolve_tools(plan=plan, findings=findings)

            for step in plan.steps:
                tool = tool_by_step.get(step.step_id)
                findings.extend(self._validate_step(step=step, tool=tool))

            policy_evaluation = None

            if self._config.require_policy_evaluation:
                policy_evaluation = self._policy.evaluate_plan(plan)
                findings.extend(
                    self._findings_from_policy(
                        plan=plan,
                        policy_evaluation=policy_evaluation,
                    )
                )

            result = self._build_result(
                plan=plan,
                findings=tuple(findings),
                policy_evaluation=policy_evaluation,
            )
            self._record(result)

            return result

        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"

            raise

    def snapshot(self) -> ActionValidatorSnapshot:
        """
        Return validator diagnostics.
        """

        with self._lock:
            return ActionValidatorSnapshot(
                name=self.name,
                validation_count=self._validation_count,
                allowed_count=self._allowed_count,
                blocked_count=self._blocked_count,
                approval_required_count=self._approval_required_count,
                confirmation_required_count=self._confirmation_required_count,
                last_decision=self._last_decision,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Reset validator diagnostics.
        """

        with self._lock:
            self._validation_count = 0
            self._allowed_count = 0
            self._blocked_count = 0
            self._approval_required_count = 0
            self._confirmation_required_count = 0
            self._last_decision = None
            self._last_error = None

    def _validate_plan_shape(
        self,
        plan: ActionPlan,
    ) -> tuple[ActionValidationFinding, ...]:
        findings: list[ActionValidationFinding] = []

        if not plan.steps:
            findings.append(
                self._finding(
                    action_id=plan.action_id,
                    severity=ActionValidationSeverity.CRITICAL,
                    reason=ActionValidationReason.EMPTY_PLAN,
                    message="action plan has no steps",
                )
            )

        orders = [step.order for step in plan.steps]

        if sorted(orders) != list(range(len(plan.steps))):
            findings.append(
                self._finding(
                    action_id=plan.action_id,
                    severity=ActionValidationSeverity.ERROR,
                    reason=ActionValidationReason.INVALID_STEP_ORDER,
                    message="action plan step order must be contiguous from zero",
                )
            )

        for step in plan.steps:
            if step.action_id != plan.action_id:
                findings.append(
                    self._finding(
                        action_id=plan.action_id,
                        step_id=step.step_id,
                        severity=ActionValidationSeverity.CRITICAL,
                        reason=ActionValidationReason.ACTION_ID_MISMATCH,
                        message="plan step action_id does not match plan action_id",
                    )
                )

        return tuple(findings)

    def _resolve_tools(
        self,
        *,
        plan: ActionPlan,
        findings: list[ActionValidationFinding],
    ) -> dict[str, ToolDescriptor]:
        resolved: dict[str, ToolDescriptor] = {}

        for step in plan.steps:
            matches = self._registry.find_by_capability(
                step.capability,
                available_only=False,
            )

            if not matches:
                if self._config.require_tool_registration:
                    findings.append(
                        self._finding(
                            action_id=plan.action_id,
                            step_id=step.step_id,
                            severity=ActionValidationSeverity.ERROR,
                            reason=ActionValidationReason.UNKNOWN_TOOL,
                            message=(
                                "no registered tool supports requested "
                                f"capability {step.capability.value}"
                            ),
                        )
                    )

                continue

            compatible = self._first_compatible_tool(step=step, tools=matches)

            if compatible is None:
                findings.append(
                    self._finding(
                        action_id=plan.action_id,
                        step_id=step.step_id,
                        severity=ActionValidationSeverity.ERROR,
                        reason=ActionValidationReason.TOOL_CAPABILITY_MISMATCH,
                        message="registered tools do not match action kind/scope/risk",
                    )
                )

                continue

            resolved[step.step_id] = compatible
            findings.extend(self._validate_tool_state(step=step, tool=compatible))

        return resolved

    def _validate_step(
        self,
        *,
        step: ActionStep,
        tool: ToolDescriptor | None,
    ) -> tuple[ActionValidationFinding, ...]:
        del tool

        findings: list[ActionValidationFinding] = []

        findings.extend(self._validate_arguments(step))
        findings.extend(self._validate_path_arguments(step))
        findings.extend(self._validate_shell_arguments(step))
        findings.extend(self._validate_risk_controls(step))

        return tuple(findings)

    def _validate_tool_state(
        self,
        *,
        step: ActionStep,
        tool: ToolDescriptor,
    ) -> tuple[ActionValidationFinding, ...]:
        findings: list[ActionValidationFinding] = []

        lookup = self._registry.get(tool.tool_id)

        if lookup.status == ToolLookupStatus.DISABLED:
            findings.append(
                self._finding(
                    action_id=step.action_id,
                    step_id=step.step_id,
                    severity=ActionValidationSeverity.ERROR,
                    reason=ActionValidationReason.TOOL_DISABLED,
                    message="registered tool is disabled",
                )
            )

        if lookup.status == ToolLookupStatus.UNAVAILABLE:
            findings.append(
                self._finding(
                    action_id=step.action_id,
                    step_id=step.step_id,
                    severity=ActionValidationSeverity.ERROR,
                    reason=ActionValidationReason.TOOL_UNAVAILABLE,
                    message="registered tool is unavailable or degraded",
                )
            )

        if self._risk_rank(step.risk) > self._risk_rank(tool.max_risk):
            findings.append(
                self._finding(
                    action_id=step.action_id,
                    step_id=step.step_id,
                    severity=ActionValidationSeverity.CRITICAL,
                    reason=ActionValidationReason.TOOL_RISK_EXCEEDED,
                    message="step risk exceeds registered tool max risk",
                )
            )

        return tuple(findings)

    def _validate_arguments(
        self,
        step: ActionStep,
    ) -> tuple[ActionValidationFinding, ...]:
        findings: list[ActionValidationFinding] = []

        for key, value in step.arguments.items():
            if not isinstance(key, str) or not key.strip():
                findings.append(
                    self._finding(
                        action_id=step.action_id,
                        step_id=step.step_id,
                        severity=ActionValidationSeverity.ERROR,
                        reason=ActionValidationReason.INVALID_ARGUMENTS,
                        message="argument keys must be non-empty strings",
                    )
                )

            if isinstance(value, str):
                if len(value) > self._config.max_argument_string_length:
                    findings.append(
                        self._finding(
                            action_id=step.action_id,
                            step_id=step.step_id,
                            severity=ActionValidationSeverity.ERROR,
                            reason=ActionValidationReason.INVALID_ARGUMENTS,
                            message="argument string is too long",
                            metadata={
                                "argument": key,
                            },
                        )
                    )

        return tuple(findings)

    def _validate_path_arguments(
        self,
        step: ActionStep,
    ) -> tuple[ActionValidationFinding, ...]:
        if step.scope not in {
            ActionScope.WORKSPACE,
            ActionScope.PROJECT,
            ActionScope.FILE_SYSTEM,
            ActionScope.IDE,
        }:
            return ()

        path_values = self._extract_string_arguments(
            arguments=step.arguments,
            keys=self._PATH_ARGUMENT_KEYS,
        )

        if step.kind in {
            ActionKind.READ,
            ActionKind.SEARCH,
            ActionKind.WRITE,
            ActionKind.PATCH,
            ActionKind.DELETE,
            ActionKind.MOVE,
            ActionKind.COPY,
            ActionKind.IDE_OPEN_FILE,
            ActionKind.IDE_APPLY_PATCH,
        } and not path_values:
            return (
                self._finding(
                    action_id=step.action_id,
                    step_id=step.step_id,
                    severity=ActionValidationSeverity.ERROR,
                    reason=ActionValidationReason.MISSING_PATH_ARGUMENT,
                    message="file or workspace action requires a path argument",
                ),
            )

        findings: list[ActionValidationFinding] = []

        for key, value in path_values:
            normalized = value.replace("\\", "/")
            parts = PurePath(normalized).parts

            if self._config.block_path_traversal and ".." in parts:
                findings.append(
                    self._finding(
                        action_id=step.action_id,
                        step_id=step.step_id,
                        severity=ActionValidationSeverity.CRITICAL,
                        reason=ActionValidationReason.PATH_TRAVERSAL_BLOCKED,
                        message="path traversal is blocked",
                        metadata={
                            "argument": key,
                            "path": value,
                        },
                    )
                )

            if self._config.block_absolute_paths and self._is_absolute_path(value):
                findings.append(
                    self._finding(
                        action_id=step.action_id,
                        step_id=step.step_id,
                        severity=ActionValidationSeverity.ERROR,
                        reason=ActionValidationReason.ABSOLUTE_PATH_BLOCKED,
                        message="absolute paths are blocked before boundary validation",
                        metadata={
                            "argument": key,
                            "path": value,
                        },
                    )
                )

            if "\x00" in value:
                findings.append(
                    self._finding(
                        action_id=step.action_id,
                        step_id=step.step_id,
                        severity=ActionValidationSeverity.CRITICAL,
                        reason=ActionValidationReason.UNSAFE_PATH,
                        message="path contains null byte",
                        metadata={
                            "argument": key,
                        },
                    )
                )

        return tuple(findings)

    def _validate_shell_arguments(
        self,
        step: ActionStep,
    ) -> tuple[ActionValidationFinding, ...]:
        if step.kind != ActionKind.SHELL_COMMAND:
            return ()

        command_values = self._extract_string_arguments(
            arguments=step.arguments,
            keys=self._SHELL_ARGUMENT_KEYS,
        )

        if not command_values:
            return (
                self._finding(
                    action_id=step.action_id,
                    step_id=step.step_id,
                    severity=ActionValidationSeverity.ERROR,
                    reason=ActionValidationReason.MISSING_COMMAND_ARGUMENT,
                    message="shell command action requires a command argument",
                ),
            )

        findings: list[ActionValidationFinding] = []

        for key, value in command_values:
            normalized = " ".join(value.casefold().strip().split())

            for pattern in self._DANGEROUS_COMMAND_PATTERNS:
                if re.search(pattern, normalized):
                    findings.append(
                        self._finding(
                            action_id=step.action_id,
                            step_id=step.step_id,
                            severity=ActionValidationSeverity.CRITICAL,
                            reason=ActionValidationReason.UNSAFE_COMMAND,
                            message="dangerous shell command pattern is blocked",
                            metadata={
                                "argument": key,
                                "pattern": pattern,
                            },
                        )
                    )

        return tuple(findings)

    def _validate_risk_controls(
        self,
        step: ActionStep,
    ) -> tuple[ActionValidationFinding, ...]:
        findings: list[ActionValidationFinding] = []

        mutating = step.kind in {
            ActionKind.WRITE,
            ActionKind.PATCH,
            ActionKind.DELETE,
            ActionKind.MOVE,
            ActionKind.COPY,
            ActionKind.SHELL_COMMAND,
            ActionKind.IDE_APPLY_PATCH,
        }

        destructive = step.kind in {
            ActionKind.DELETE,
            ActionKind.SHELL_COMMAND,
            ActionKind.IDE_APPLY_PATCH,
        }

        if (
            self._config.require_timeout_for_high_risk
            and step.risk in {ActionRisk.HIGH, ActionRisk.CRITICAL}
            and step.timeout_ms is None
        ):
            findings.append(
                self._finding(
                    action_id=step.action_id,
                    step_id=step.step_id,
                    severity=ActionValidationSeverity.ERROR,
                    reason=ActionValidationReason.MISSING_TIMEOUT,
                    message="high and critical risk steps require timeout_ms",
                )
            )

        if (
            self._config.require_interruptible_mutations
            and mutating
            and not step.interruptible
        ):
            findings.append(
                self._finding(
                    action_id=step.action_id,
                    step_id=step.step_id,
                    severity=ActionValidationSeverity.CRITICAL,
                    reason=ActionValidationReason.NON_INTERRUPTIBLE_UNSAFE_ACTION,
                    message="mutating steps must be interruptible",
                )
            )

        if (
            self._config.require_rollback_for_destructive
            and destructive
            and not step.rollback_supported
        ):
            findings.append(
                self._finding(
                    action_id=step.action_id,
                    step_id=step.step_id,
                    severity=ActionValidationSeverity.WARNING,
                    reason=ActionValidationReason.MISSING_ROLLBACK,
                    message="destructive step has no rollback support",
                )
            )

        return tuple(findings)

    def _findings_from_policy(
        self,
        *,
        plan: ActionPlan,
        policy_evaluation: PermissionPolicyEvaluation,
    ) -> tuple[ActionValidationFinding, ...]:
        findings: list[ActionValidationFinding] = []

        if policy_evaluation.decision == PermissionDecision.DENY:
            findings.append(
                self._finding(
                    action_id=plan.action_id,
                    severity=ActionValidationSeverity.CRITICAL,
                    reason=ActionValidationReason.POLICY_DENIED,
                    message=(
                        "permission policy denied action: "
                        f"{policy_evaluation.reason.value}"
                    ),
                    metadata={
                        "policy_reason": policy_evaluation.reason.value,
                    },
                )
            )

        if policy_evaluation.requires_approval and not plan.requires_approval:
            findings.append(
                self._finding(
                    action_id=plan.action_id,
                    severity=ActionValidationSeverity.ERROR,
                    reason=ActionValidationReason.APPROVAL_REQUIRED,
                    message=(
                        "policy requires approval but plan does not declare "
                        "approval"
                    ),
                )
            )

        if policy_evaluation.requires_confirmation:
            findings.append(
                self._finding(
                    action_id=plan.action_id,
                    severity=ActionValidationSeverity.WARNING,
                    reason=ActionValidationReason.CONFIRMATION_REQUIRED,
                    message="policy requires user confirmation before execution",
                )
            )

        if (
            plan.permission_decision == PermissionDecision.ALLOW
            and policy_evaluation.decision != PermissionDecision.ALLOW
        ):
            findings.append(
                self._finding(
                    action_id=plan.action_id,
                    severity=ActionValidationSeverity.CRITICAL,
                    reason=ActionValidationReason.POLICY_BYPASS_ATTEMPT,
                    message="plan marked allow but policy requires stronger decision",
                )
            )

        return tuple(findings)

    def _build_result(
        self,
        *,
        plan: ActionPlan,
        findings: tuple[ActionValidationFinding, ...],
        policy_evaluation: PermissionPolicyEvaluation | None,
    ) -> ActionValidationResult:
        blocked = any(
            finding.severity
            in {
                ActionValidationSeverity.ERROR,
                ActionValidationSeverity.CRITICAL,
            }
            for finding in findings
        )
        requires_approval = any(
            finding.reason == ActionValidationReason.APPROVAL_REQUIRED
            for finding in findings
        )
        requires_confirmation = any(
            finding.reason == ActionValidationReason.CONFIRMATION_REQUIRED
            for finding in findings
        )

        if policy_evaluation is not None:
            requires_approval = (
                requires_approval or policy_evaluation.requires_approval
            )
            requires_confirmation = (
                requires_confirmation
                or policy_evaluation.requires_confirmation
            )

        if blocked:
            decision = ActionValidationDecision.BLOCK

        elif requires_approval:
            decision = ActionValidationDecision.REQUIRE_APPROVAL

        elif requires_confirmation:
            decision = ActionValidationDecision.REQUIRE_CONFIRMATION

        else:
            decision = ActionValidationDecision.ALLOW

        final_findings = findings

        if not final_findings:
            final_findings = (
                self._finding(
                    action_id=plan.action_id,
                    severity=ActionValidationSeverity.INFO,
                    reason=ActionValidationReason.VALIDATION_PASSED,
                    message="action plan validation passed",
                ),
            )

        return ActionValidationResult(
            action_id=plan.action_id,
            plan_id=plan.plan_id,
            decision=decision,
            valid=decision != ActionValidationDecision.BLOCK,
            blocked=decision == ActionValidationDecision.BLOCK,
            requires_approval=decision == ActionValidationDecision.REQUIRE_APPROVAL,
            requires_confirmation=(
                decision == ActionValidationDecision.REQUIRE_CONFIRMATION
            ),
            findings=final_findings,
            policy_evaluation=policy_evaluation,
            metadata={
                "validator": self.name,
                "plan_status": plan.status.value,
                "step_count": len(plan.steps),
            },
        )

    def _record(self, result: ActionValidationResult) -> None:
        with self._lock:
            self._last_decision = result.decision

            if result.blocked:
                self._blocked_count += 1

            elif result.requires_approval:
                self._approval_required_count += 1

            elif result.requires_confirmation:
                self._confirmation_required_count += 1

            else:
                self._allowed_count += 1

    @staticmethod
    def _first_compatible_tool(
        *,
        step: ActionStep,
        tools: tuple[ToolDescriptor, ...],
    ) -> ToolDescriptor | None:
        for tool in tools:
            if not tool.supports_action_kind(step.kind):
                continue

            if not tool.supports_scope(step.scope):
                continue

            if ActionValidator._risk_rank(step.risk) > ActionValidator._risk_rank(
                tool.max_risk
            ):
                continue

            return tool

        return None

    @staticmethod
    def _extract_string_arguments(
        *,
        arguments: dict[str, object],
        keys: tuple[str, ...],
    ) -> tuple[tuple[str, str], ...]:
        key_set = {key.casefold() for key in keys}
        values: list[tuple[str, str]] = []

        for key, value in arguments.items():
            if key.casefold() not in key_set:
                continue

            if isinstance(value, str):
                values.append((key, value))

        return tuple(values)

    @staticmethod
    def _is_absolute_path(value: str) -> bool:
        normalized = value.strip()

        if normalized.startswith(("/", "\\")):
            return True

        if re.match(r"^[a-zA-Z]:[\\/]", normalized):
            return True

        return False

    @staticmethod
    def _risk_rank(risk: ActionRisk) -> int:
        return {
            ActionRisk.NONE: 0,
            ActionRisk.LOW: 1,
            ActionRisk.MEDIUM: 2,
            ActionRisk.HIGH: 3,
            ActionRisk.CRITICAL: 4,
        }[risk]

    @staticmethod
    def _finding(
        *,
        action_id: str,
        severity: ActionValidationSeverity,
        reason: ActionValidationReason,
        message: str,
        step_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> ActionValidationFinding:
        return ActionValidationFinding(
            action_id=action_id,
            step_id=step_id,
            severity=severity,
            reason=reason,
            message=message,
            metadata=metadata or {},
        )