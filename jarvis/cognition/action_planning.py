from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from typing import Any

from pydantic import Field, field_validator

from jarvis.cognition.models import (
    CognitionModel,
    CognitionRequest,
    new_id,
)
from jarvis.runtime.observability.structured_logger import get_logger


class ToolActionType(StrEnum):
    """
    High-level action type proposed by cognition.

    These are proposals only. No action is executed by this module.
    """

    OPEN_APPLICATION = "open_application"
    OPEN_FILE = "open_file"
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    DELETE_FILE = "delete_file"
    MOVE_FILE = "move_file"
    RUN_TERMINAL_COMMAND = "run_terminal_command"
    SEARCH_WEB = "search_web"
    SEND_MESSAGE = "send_message"
    SCHEDULE_EVENT = "schedule_event"
    SYSTEM_CONTROL = "system_control"
    UNKNOWN = "unknown"


class ToolActionTargetKind(StrEnum):
    """
    Target kind for a proposed action.
    """

    APPLICATION = "application"
    FILE = "file"
    DIRECTORY = "directory"
    TERMINAL = "terminal"
    URL = "url"
    PERSON = "person"
    CALENDAR = "calendar"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class ToolActionRiskLevel(StrEnum):
    """
    Safety risk level for a proposed action.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ToolActionPermissionMode(StrEnum):
    """
    Permission requirement before execution.
    """

    AUTO_ALLOWED = "auto_allowed"
    CONFIRMATION_REQUIRED = "confirmation_required"
    ELEVATED_CONFIRMATION_REQUIRED = "elevated_confirmation_required"
    BLOCKED = "blocked"


class ToolActionParameter(CognitionModel):
    """
    One structured parameter for a proposed action.
    """

    name: str
    value: str
    sensitive: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "value")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class ToolActionProposal(CognitionModel):
    """
    A proposed tool/laptop action.

    This is a plan, not execution. Later tool runtime can execute only after
    permission and policy validation.
    """

    proposal_id: str = Field(default_factory=new_id)
    request_id: str
    action_type: ToolActionType = ToolActionType.UNKNOWN
    target_kind: ToolActionTargetKind = ToolActionTargetKind.UNKNOWN
    target: str | None = None
    parameters: tuple[ToolActionParameter, ...] = ()
    risk_level: ToolActionRiskLevel = ToolActionRiskLevel.LOW
    permission_mode: ToolActionPermissionMode = (
        ToolActionPermissionMode.CONFIRMATION_REQUIRED
    )
    rationale: str = "Action proposed from user request."
    executable: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("request_id", "rationale")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("target")
    @classmethod
    def _clean_optional_target(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None


class ToolActionSafetyDecision(CognitionModel):
    """
    Safety decision for a set of proposed actions.
    """

    decision_id: str = Field(default_factory=new_id)
    request_id: str
    permission_mode: ToolActionPermissionMode
    risk_level: ToolActionRiskLevel
    allowed: bool
    reasons: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("request_id")
    @classmethod
    def _request_id_required(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("request_id cannot be empty.")

        return cleaned


class ToolActionPlan(CognitionModel):
    """
    Final action plan produced by cognition.

    It can be passed to a future tool runtime, but this module never executes it.
    """

    plan_id: str = Field(default_factory=new_id)
    request_id: str
    proposals: tuple[ToolActionProposal, ...] = ()
    safety: ToolActionSafetyDecision
    executable: bool = False
    blocked: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("request_id")
    @classmethod
    def _request_id_required(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("request_id cannot be empty.")

        return cleaned

    @property
    def proposal_count(self) -> int:
        return len(self.proposals)


@dataclass(frozen=True, slots=True)
class ToolActionPlannerConfig:
    """
    Configuration for ToolActionPlanner.

    Defaults are intentionally conservative.
    """

    name: str = "tool_action_planner"
    auto_allow_low_risk_readonly: bool = False
    allow_terminal_commands: bool = False
    allow_system_control: bool = False
    require_confirmation_for_medium_risk: bool = True
    dangerous_phrases: tuple[str, ...] = (
        "format drive",
        "delete system32",
        "disable antivirus",
        "bypass password",
        "steal password",
        "keylogger",
        "malware",
        "wipe disk",
        "remove all files",
    )
    high_risk_phrases: tuple[str, ...] = (
        "delete",
        "remove",
        "overwrite",
        "shutdown",
        "restart",
        "powershell",
        "command prompt",
        "terminal",
        "run command",
        "execute command",
    )
    action_phrases: tuple[str, ...] = (
        "open",
        "read",
        "write",
        "create",
        "delete",
        "move",
        "copy",
        "run",
        "execute",
        "send",
        "schedule",
        "search",
        "close",
        "shutdown",
        "restart",
    )

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        self._validate_phrases("dangerous_phrases", self.dangerous_phrases)
        self._validate_phrases("high_risk_phrases", self.high_risk_phrases)
        self._validate_phrases("action_phrases", self.action_phrases)

    @staticmethod
    def _validate_phrases(
        name: str,
        phrases: tuple[str, ...],
    ) -> None:
        for phrase in phrases:
            if not phrase.strip():
                raise ValueError(f"{name} cannot contain empty phrases.")


@dataclass(frozen=True, slots=True)
class ToolActionPlannerSnapshot:
    """
    Observable tool action planner diagnostics.
    """

    name: str
    planned_count: int
    proposal_count: int
    allowed_count: int
    blocked_count: int
    confirmation_required_count: int
    last_request_id: str | None
    last_risk_level: ToolActionRiskLevel | None
    last_permission_mode: ToolActionPermissionMode | None
    last_error: str | None


class ToolActionPlanner:
    """
    Safe action planning contract layer.

    Responsibilities:
    - detect whether a user request appears action-oriented
    - create structured ToolActionProposal objects
    - assign risk level and permission mode
    - block clearly dangerous actions
    - keep LLM separate from execution

    Non-responsibilities:
    - no actual laptop control
    - no shell execution
    - no file writes/deletes
    - no API calls
    - no permission UI
    """

    def __init__(
        self,
        *,
        config: ToolActionPlannerConfig | None = None,
    ) -> None:
        self._config = config or ToolActionPlannerConfig()
        self._config.validate()

        self._lock = RLock()
        self._logger = get_logger("cognition.tool_action_planner")

        self._planned_count = 0
        self._proposal_count = 0
        self._allowed_count = 0
        self._blocked_count = 0
        self._confirmation_required_count = 0
        self._last_request_id: str | None = None
        self._last_risk_level: ToolActionRiskLevel | None = None
        self._last_permission_mode: ToolActionPermissionMode | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def plan(self, request: CognitionRequest) -> ToolActionPlan:
        """
        Create a safe action plan for one cognition request.

        This never executes anything.
        """

        normalized_text = self._normalize(request.text)
        proposal = self._proposal_for_request(
            request=request,
            normalized_text=normalized_text,
        )
        safety = self._safety_decision(
            request=request,
            proposal=proposal,
            normalized_text=normalized_text,
        )

        plan = ToolActionPlan(
            request_id=request.request_id,
            proposals=(proposal,),
            safety=safety,
            executable=safety.allowed and proposal.executable,
            blocked=not safety.allowed,
            metadata={
                "planner": self.name,
                "llm_direct_execution_allowed": False,
            },
        )

        self._record_plan(plan)

        self._logger.info(
            "tool_action_plan_created",
            planner=self.name,
            request_id=request.request_id,
            action_type=proposal.action_type.value,
            risk_level=safety.risk_level.value,
            permission_mode=safety.permission_mode.value,
            executable=plan.executable,
            blocked=plan.blocked,
        )

        return plan

    def snapshot(self) -> ToolActionPlannerSnapshot:
        """
        Return planner diagnostics.
        """

        with self._lock:
            return ToolActionPlannerSnapshot(
                name=self.name,
                planned_count=self._planned_count,
                proposal_count=self._proposal_count,
                allowed_count=self._allowed_count,
                blocked_count=self._blocked_count,
                confirmation_required_count=self._confirmation_required_count,
                last_request_id=self._last_request_id,
                last_risk_level=self._last_risk_level,
                last_permission_mode=self._last_permission_mode,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Reset planner counters.
        """

        with self._lock:
            self._planned_count = 0
            self._proposal_count = 0
            self._allowed_count = 0
            self._blocked_count = 0
            self._confirmation_required_count = 0
            self._last_request_id = None
            self._last_risk_level = None
            self._last_permission_mode = None
            self._last_error = None

        self._logger.info("tool_action_planner_reset", planner=self.name)

    def _proposal_for_request(
        self,
        *,
        request: CognitionRequest,
        normalized_text: str,
    ) -> ToolActionProposal:
        action_type = self._action_type_for(normalized_text)
        target_kind = self._target_kind_for(action_type, normalized_text)
        risk_level = self._risk_level_for(normalized_text, action_type)
        permission_mode = self._permission_mode_for(risk_level, action_type)
        executable = permission_mode != ToolActionPermissionMode.BLOCKED
        target = self._target_for(normalized_text, action_type)

        return ToolActionProposal(
            request_id=request.request_id,
            action_type=action_type,
            target_kind=target_kind,
            target=target,
            parameters=(
                ToolActionParameter(
                    name="utterance",
                    value=request.text,
                    sensitive=False,
                ),
            ),
            risk_level=risk_level,
            permission_mode=permission_mode,
            rationale="Prepared as a safe proposal. No execution performed.",
            executable=executable,
            metadata={
                "planner": self.name,
                "requires_runtime_permission": True,
            },
        )

    def _safety_decision(
        self,
        *,
        request: CognitionRequest,
        proposal: ToolActionProposal,
        normalized_text: str,
    ) -> ToolActionSafetyDecision:
        reasons: list[str] = []

        if self._contains_any(normalized_text, self._config.dangerous_phrases):
            reasons.append("request matched blocked dangerous phrase")

            return ToolActionSafetyDecision(
                request_id=request.request_id,
                permission_mode=ToolActionPermissionMode.BLOCKED,
                risk_level=ToolActionRiskLevel.CRITICAL,
                allowed=False,
                reasons=tuple(reasons),
                metadata={
                    "planner": self.name,
                },
            )

        if (
            proposal.action_type == ToolActionType.RUN_TERMINAL_COMMAND
            and not self._config.allow_terminal_commands
        ):
            reasons.append("terminal command execution is disabled")

            return ToolActionSafetyDecision(
                request_id=request.request_id,
                permission_mode=ToolActionPermissionMode.BLOCKED,
                risk_level=ToolActionRiskLevel.HIGH,
                allowed=False,
                reasons=tuple(reasons),
                metadata={
                    "planner": self.name,
                },
            )

        if (
            proposal.action_type == ToolActionType.SYSTEM_CONTROL
            and not self._config.allow_system_control
        ):
            reasons.append("system control execution is disabled")

            return ToolActionSafetyDecision(
                request_id=request.request_id,
                permission_mode=ToolActionPermissionMode.BLOCKED,
                risk_level=ToolActionRiskLevel.HIGH,
                allowed=False,
                reasons=tuple(reasons),
                metadata={
                    "planner": self.name,
                },
            )

        reasons.append("proposal requires tool runtime validation")

        return ToolActionSafetyDecision(
            request_id=request.request_id,
            permission_mode=proposal.permission_mode,
            risk_level=proposal.risk_level,
            allowed=proposal.permission_mode != ToolActionPermissionMode.BLOCKED,
            reasons=tuple(reasons),
            metadata={
                "planner": self.name,
            },
        )

    def _record_plan(self, plan: ToolActionPlan) -> None:
        with self._lock:
            self._planned_count += 1
            self._proposal_count += plan.proposal_count
            self._last_request_id = plan.request_id
            self._last_risk_level = plan.safety.risk_level
            self._last_permission_mode = plan.safety.permission_mode
            self._last_error = None

            if plan.safety.allowed:
                self._allowed_count += 1

            else:
                self._blocked_count += 1

            if plan.safety.permission_mode in {
                ToolActionPermissionMode.CONFIRMATION_REQUIRED,
                ToolActionPermissionMode.ELEVATED_CONFIRMATION_REQUIRED,
            }:
                self._confirmation_required_count += 1

    def _action_type_for(
        self,
        normalized_text: str,
    ) -> ToolActionType:
        if normalized_text.startswith("open "):
            if any(token in normalized_text for token in ("file", "folder")):
                return ToolActionType.OPEN_FILE

            return ToolActionType.OPEN_APPLICATION

        if normalized_text.startswith(("read ", "show ")):
            return ToolActionType.READ_FILE

        if normalized_text.startswith(("write ", "create ")):
            return ToolActionType.WRITE_FILE

        if normalized_text.startswith(("delete ", "remove ")):
            return ToolActionType.DELETE_FILE

        if normalized_text.startswith(("move ", "copy ")):
            return ToolActionType.MOVE_FILE

        if normalized_text.startswith(("run ", "execute ")):
            return ToolActionType.RUN_TERMINAL_COMMAND

        if normalized_text.startswith("search "):
            return ToolActionType.SEARCH_WEB

        if normalized_text.startswith("send "):
            return ToolActionType.SEND_MESSAGE

        if normalized_text.startswith("schedule "):
            return ToolActionType.SCHEDULE_EVENT

        if normalized_text.startswith(("shutdown", "restart", "close ")):
            return ToolActionType.SYSTEM_CONTROL

        if self._contains_any(normalized_text, self._config.action_phrases):
            return ToolActionType.UNKNOWN

        return ToolActionType.UNKNOWN

    @staticmethod
    def _target_kind_for(
        action_type: ToolActionType,
        normalized_text: str,
    ) -> ToolActionTargetKind:
        if action_type == ToolActionType.OPEN_APPLICATION:
            return ToolActionTargetKind.APPLICATION

        if action_type in {
            ToolActionType.OPEN_FILE,
            ToolActionType.READ_FILE,
            ToolActionType.WRITE_FILE,
            ToolActionType.DELETE_FILE,
            ToolActionType.MOVE_FILE,
        }:
            if "folder" in normalized_text or "directory" in normalized_text:
                return ToolActionTargetKind.DIRECTORY

            return ToolActionTargetKind.FILE

        if action_type == ToolActionType.RUN_TERMINAL_COMMAND:
            return ToolActionTargetKind.TERMINAL

        if action_type == ToolActionType.SEARCH_WEB:
            return ToolActionTargetKind.URL

        if action_type == ToolActionType.SEND_MESSAGE:
            return ToolActionTargetKind.PERSON

        if action_type == ToolActionType.SCHEDULE_EVENT:
            return ToolActionTargetKind.CALENDAR

        if action_type == ToolActionType.SYSTEM_CONTROL:
            return ToolActionTargetKind.SYSTEM

        return ToolActionTargetKind.UNKNOWN

    def _risk_level_for(
        self,
        normalized_text: str,
        action_type: ToolActionType,
    ) -> ToolActionRiskLevel:
        if self._contains_any(normalized_text, self._config.dangerous_phrases):
            return ToolActionRiskLevel.CRITICAL

        if action_type in {
            ToolActionType.DELETE_FILE,
            ToolActionType.RUN_TERMINAL_COMMAND,
            ToolActionType.SYSTEM_CONTROL,
        }:
            return ToolActionRiskLevel.HIGH

        if self._contains_any(normalized_text, self._config.high_risk_phrases):
            return ToolActionRiskLevel.HIGH

        if action_type in {
            ToolActionType.WRITE_FILE,
            ToolActionType.MOVE_FILE,
            ToolActionType.SEND_MESSAGE,
            ToolActionType.SCHEDULE_EVENT,
        }:
            return ToolActionRiskLevel.MEDIUM

        return ToolActionRiskLevel.LOW

    def _permission_mode_for(
        self,
        risk_level: ToolActionRiskLevel,
        action_type: ToolActionType,
    ) -> ToolActionPermissionMode:
        if risk_level == ToolActionRiskLevel.CRITICAL:
            return ToolActionPermissionMode.BLOCKED

        if risk_level == ToolActionRiskLevel.HIGH:
            return ToolActionPermissionMode.ELEVATED_CONFIRMATION_REQUIRED

        if risk_level == ToolActionRiskLevel.MEDIUM:
            if self._config.require_confirmation_for_medium_risk:
                return ToolActionPermissionMode.CONFIRMATION_REQUIRED

            return ToolActionPermissionMode.AUTO_ALLOWED

        if (
            risk_level == ToolActionRiskLevel.LOW
            and self._config.auto_allow_low_risk_readonly
            and action_type
            in {
                ToolActionType.OPEN_APPLICATION,
                ToolActionType.OPEN_FILE,
                ToolActionType.READ_FILE,
                ToolActionType.SEARCH_WEB,
            }
        ):
            return ToolActionPermissionMode.AUTO_ALLOWED

        return ToolActionPermissionMode.CONFIRMATION_REQUIRED

    @staticmethod
    def _target_for(
        normalized_text: str,
        action_type: ToolActionType,
    ) -> str | None:
        prefixes = {
            ToolActionType.OPEN_APPLICATION: "open ",
            ToolActionType.OPEN_FILE: "open ",
            ToolActionType.READ_FILE: "read ",
            ToolActionType.WRITE_FILE: "write ",
            ToolActionType.DELETE_FILE: "delete ",
            ToolActionType.MOVE_FILE: "move ",
            ToolActionType.RUN_TERMINAL_COMMAND: "run ",
            ToolActionType.SEARCH_WEB: "search ",
            ToolActionType.SEND_MESSAGE: "send ",
            ToolActionType.SCHEDULE_EVENT: "schedule ",
        }

        prefix = prefixes.get(action_type)

        if prefix is None or not normalized_text.startswith(prefix):
            return None

        target = normalized_text.removeprefix(prefix).strip()

        return target or None

    @staticmethod
    def _contains_any(
        text: str,
        phrases: tuple[str, ...],
    ) -> bool:
        return any(phrase.casefold() in text for phrase in phrases)

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.casefold().split())