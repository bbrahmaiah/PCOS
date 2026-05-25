from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from jarvis.tools.ids import (
    new_action_error_id,
    new_action_id,
    new_action_plan_id,
    new_action_result_id,
    new_action_step_id,
    new_policy_decision_id,
    new_tool_id,
    utc_now,
)


class ToolModel(BaseModel):
    """
    Base model for all Tool & Action Runtime contracts.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,
    )


class ToolId(ToolModel):
    """
    Stable identifier for a tool.

    This is a model instead of a raw string so future registry/runtime layers can
    attach stronger validation without changing public contracts.
    """

    value: str = Field(default_factory=new_tool_id)

    @field_validator("value")
    @classmethod
    def _valid_tool_id(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("tool id cannot be empty.")

        return cleaned


class ActionId(ToolModel):
    """
    Stable identifier for one action request/execution.
    """

    value: str = Field(default_factory=new_action_id)

    @field_validator("value")
    @classmethod
    def _valid_action_id(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("action id cannot be empty.")

        return cleaned


class ActionKind(StrEnum):
    """
    High-level action kind.

    These are intentionally broad. Concrete tools add details through
    capability, arguments, and policy metadata.
    """

    READ = "read"
    SEARCH = "search"
    WRITE = "write"
    PATCH = "patch"
    DELETE = "delete"
    MOVE = "move"
    COPY = "copy"
    SHELL_COMMAND = "shell_command"
    OPEN_APPLICATION = "open_application"
    BROWSER_OPEN = "browser_open"
    BROWSER_SEARCH = "browser_search"
    IDE_OPEN_FILE = "ide_open_file"
    IDE_APPLY_PATCH = "ide_apply_patch"
    SYSTEM_QUERY = "system_query"
    COMPOSITE = "composite"


class ActionRisk(StrEnum):
    """
    Risk classification for an action.

    Policy runtime will later use this to decide allow/deny/approval/sandbox.
    """

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionStatus(StrEnum):
    """
    Lifecycle status for governed actions.
    """

    CREATED = "created"
    PLANNED = "planned"
    VALIDATING = "validating"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    APPROVED = "approved"
    BLOCKED = "blocked"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class ActionScope(StrEnum):
    """
    Boundary/scope where an action may operate.
    """

    MEMORY = "memory"
    WORKSPACE = "workspace"
    PROJECT = "project"
    FILE_SYSTEM = "file_system"
    SHELL = "shell"
    BROWSER = "browser"
    IDE = "ide"
    DESKTOP = "desktop"
    NETWORK = "network"
    SYSTEM = "system"


class PermissionDecision(StrEnum):
    """
    Policy result for an action.

    This is a contract only. The full permission runtime comes in Step 3.
    """

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"
    REQUIRE_CONFIRMATION = "require_confirmation"
    SANDBOX_ONLY = "sandbox_only"
    READ_ONLY_ONLY = "read_only_only"


class ToolCapability(StrEnum):
    """
    Capability exposed by a tool.

    Registry/runtime layers will map capabilities to concrete tool descriptors.
    """

    READ_FILE = "read_file"
    LIST_DIRECTORY = "list_directory"
    SEARCH_FILES = "search_files"
    WRITE_FILE = "write_file"
    PATCH_FILE = "patch_file"
    DELETE_FILE = "delete_file"
    RUN_SHELL_COMMAND = "run_shell_command"
    OPEN_BROWSER = "open_browser"
    SEARCH_WEB = "search_web"
    OPEN_IDE_FILE = "open_ide_file"
    APPLY_IDE_PATCH = "apply_ide_patch"
    QUERY_DESKTOP_STATE = "query_desktop_state"
    CONTROL_APPLICATION = "control_application"


class ActionContext(ToolModel):
    """
    Runtime context passed into action planning/execution.

    This does not grant permission. It only describes context. Policy still
    decides whether an action may proceed.
    """

    user_id: str | None = None
    session_id: str | None = None
    conversation_id: str | None = None
    turn_id: str | None = None
    workspace_root: str | None = None
    project_root: str | None = None
    working_directory: str | None = None
    requested_by: str = "user"
    memory_context_refs: tuple[str, ...] = ()
    cancellation_token_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "user_id",
        "session_id",
        "conversation_id",
        "turn_id",
        "workspace_root",
        "project_root",
        "working_directory",
        "requested_by",
        "cancellation_token_id",
    )
    @classmethod
    def _clean_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None

    @field_validator("memory_context_refs")
    @classmethod
    def _clean_memory_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(item.strip() for item in value if item.strip())

        if len(cleaned) != len(set(cleaned)):
            raise ValueError("memory_context_refs cannot contain duplicates.")

        return cleaned


class ActionError(ToolModel):
    """
    Structured action error.

    Errors are data, not just exceptions, because action failures must be logged,
    audited, explained, retried, and sometimes rolled back.
    """

    error_id: str = Field(default_factory=new_action_error_id)
    code: str
    message: str
    recoverable: bool = False
    retryable: bool = False
    caused_by: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("error_id", "code", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("caused_by")
    @classmethod
    def _clean_caused_by(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None


class ActionRequest(ToolModel):
    """
    User/cognition request for a real-world action.

    This is not executable by itself. It must become a plan, pass policy,
    validation, approval if needed, and then enter execution runtime.
    """

    action_id: str = Field(default_factory=new_action_id)
    kind: ActionKind
    requested_capability: ToolCapability
    intent: str
    scope: ActionScope
    risk: ActionRisk = ActionRisk.LOW
    context: ActionContext = Field(default_factory=ActionContext)
    arguments: dict[str, Any] = Field(default_factory=dict)
    requires_approval: bool = False
    idempotency_key: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("action_id", "intent")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("idempotency_key")
    @classmethod
    def _clean_idempotency_key(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None

    @model_validator(mode="after")
    def _approval_required_for_high_risk(self) -> ActionRequest:
        if self.risk in {ActionRisk.HIGH, ActionRisk.CRITICAL}:
            if not self.requires_approval:
                raise ValueError("high and critical risk actions require approval.")

        return self


class ActionStep(ToolModel):
    """
    One step inside an action plan.

    A step is still not direct execution. It is a typed, policy-visible unit
    that future execution runtime may run after validation.
    """

    step_id: str = Field(default_factory=new_action_step_id)
    action_id: str
    order: int = Field(ge=0)
    kind: ActionKind
    capability: ToolCapability
    scope: ActionScope
    risk: ActionRisk = ActionRisk.LOW
    description: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    timeout_ms: int | None = Field(default=None, ge=1)
    rollback_supported: bool = False
    interruptible: bool = True
    status: ActionStatus = ActionStatus.PLANNED
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("step_id", "action_id", "description")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("depends_on")
    @classmethod
    def _clean_depends_on(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(item.strip() for item in value if item.strip())

        if len(cleaned) != len(set(cleaned)):
            raise ValueError("depends_on cannot contain duplicates.")

        return cleaned

    @model_validator(mode="after")
    def _validate_mutating_steps(self) -> ActionStep:
        mutating_kinds = {
            ActionKind.WRITE,
            ActionKind.PATCH,
            ActionKind.DELETE,
            ActionKind.MOVE,
            ActionKind.COPY,
            ActionKind.IDE_APPLY_PATCH,
            ActionKind.SHELL_COMMAND,
        }

        if self.kind in mutating_kinds and not self.interruptible:
            raise ValueError("mutating action steps must be interruptible.")

        if self.risk in {ActionRisk.HIGH, ActionRisk.CRITICAL}:
            if not self.timeout_ms:
                raise ValueError("high and critical risk steps require timeout_ms.")

        return self


class ActionPlan(ToolModel):
    """
    Typed action plan.

    The planner creates this. Policy and validation inspect this. Execution
    runtime later executes validated steps. The plan itself never executes.
    """

    plan_id: str = Field(default_factory=new_action_plan_id)
    action_id: str
    goal: str
    steps: tuple[ActionStep, ...]
    risk: ActionRisk
    scope: ActionScope
    status: ActionStatus = ActionStatus.PLANNED
    permission_decision: PermissionDecision | None = None
    requires_approval: bool = False
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("plan_id", "action_id", "goal")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @model_validator(mode="after")
    def _validate_plan(self) -> ActionPlan:
        if not self.steps:
            raise ValueError("action plan must contain at least one step.")

        action_ids = {step.action_id for step in self.steps}

        if action_ids != {self.action_id}:
            raise ValueError("all plan steps must share the plan action_id.")

        orders = [step.order for step in self.steps]

        if len(orders) != len(set(orders)):
            raise ValueError("action step order values must be unique.")

        sorted_orders = sorted(orders)
        expected_orders = list(range(len(self.steps)))

        if sorted_orders != expected_orders:
            raise ValueError("action step order must be contiguous from zero.")

        if self.risk in {ActionRisk.HIGH, ActionRisk.CRITICAL}:
            if not self.requires_approval:
                raise ValueError("high and critical risk plans require approval.")

        if self.permission_decision == PermissionDecision.DENY:
            if self.status != ActionStatus.BLOCKED:
                raise ValueError("denied action plans must have BLOCKED status.")

        return self


class ActionResult(ToolModel):
    """
    Result of a governed action or action step.

    This is the structured result returned back to cognition/conversation after
    execution runtime completes, fails, or cancels work.
    """

    result_id: str = Field(default_factory=new_action_result_id)
    action_id: str
    status: ActionStatus
    success: bool
    output: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    error: ActionError | None = None
    started_at: datetime | None = None
    completed_at: datetime = Field(default_factory=utc_now)
    duration_ms: int | None = Field(default=None, ge=0)
    artifacts: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("result_id", "action_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("output")
    @classmethod
    def _clean_output(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None

    @field_validator("artifacts")
    @classmethod
    def _clean_artifacts(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(item.strip() for item in value if item.strip())

        if len(cleaned) != len(set(cleaned)):
            raise ValueError("artifacts cannot contain duplicates.")

        return cleaned

    @model_validator(mode="after")
    def _validate_result_consistency(self) -> ActionResult:
        success_statuses = {
            ActionStatus.SUCCEEDED,
            ActionStatus.ROLLED_BACK,
        }
        failure_statuses = {
            ActionStatus.FAILED,
            ActionStatus.BLOCKED,
            ActionStatus.CANCELLED,
        }

        if self.success and self.status not in success_statuses:
            raise ValueError("successful results must use a success status.")

        if not self.success and self.status in success_statuses:
            raise ValueError("failed results cannot use a success status.")

        if self.status in failure_statuses and self.error is None:
            raise ValueError("failed, blocked, or cancelled results require error.")

        return self


class PolicyDecisionRecord(ToolModel):
    """
    Step-0 policy decision record.

    The full Permission Policy Runtime is Step 3. This model exists now because
    action contracts need a stable place to carry policy decisions.
    """

    decision_id: str = Field(default_factory=new_policy_decision_id)
    action_id: str
    decision: PermissionDecision
    risk: ActionRisk
    reason: str
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("decision_id", "action_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned