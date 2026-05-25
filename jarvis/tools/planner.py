from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from pydantic import Field, field_validator, model_validator

from jarvis.tools.ids import new_action_id, new_action_plan_id, utc_now
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


class ActionPlanningIntentKind(StrEnum):
    """
    High-level intent categories the planner can understand.

    This is intentionally conservative. Unknown intent must not become hidden
    execution.
    """

    RUN_TESTS_AND_SUMMARIZE = "run_tests_and_summarize"
    RUN_QUALITY_GATE = "run_quality_gate"
    OPEN_FILE = "open_file"
    SEARCH_CODE = "search_code"
    PREPARE_PATCH = "prepare_patch"
    UNKNOWN = "unknown"


class ActionPlanningDecision(StrEnum):
    """
    Planner decision.

    PROPOSED means the plan is ready for policy/validation.
    NEEDS_CLARIFICATION means the intent is not executable enough.
    REJECTED means the planner refuses to create a plan.
    """

    PROPOSED = "proposed"
    NEEDS_CLARIFICATION = "needs_clarification"
    REJECTED = "rejected"


class ActionPlanningReason(StrEnum):
    """
    Machine-readable planning reason.
    """

    TEST_PLAN_PROPOSED = "test_plan_proposed"
    QUALITY_GATE_PLAN_PROPOSED = "quality_gate_plan_proposed"
    OPEN_FILE_PLAN_PROPOSED = "open_file_plan_proposed"
    SEARCH_CODE_PLAN_PROPOSED = "search_code_plan_proposed"
    PATCH_PLAN_PROPOSED = "patch_plan_proposed"
    UNSUPPORTED_INTENT = "unsupported_intent"
    MISSING_TARGET = "missing_target"
    MISSING_PATCH_TEXT = "missing_patch_text"
    PLAN_REQUIRES_APPROVAL = "plan_requires_approval"


class PlannedStepRole(StrEnum):
    """
    Role of a planner step.

    TOOL steps are executable by a runtime later.
    COGNITIVE steps are reasoning/summarization steps, not direct tools.
    USER steps require user approval or clarification.
    """

    TOOL = "tool"
    COGNITIVE = "cognitive"
    USER = "user"


class PlannerStep(ToolModel):
    """
    Human-readable planner step.

    This is separate from ActionStep because not every plan step maps to a
    runtime tool. Some steps are cognitive, such as parsing failures.
    """

    order: int = Field(ge=0)
    role: PlannedStepRole
    title: str
    description: str
    action_kind: ActionKind | None = None
    capability: ToolCapability | None = None
    risk: ActionRisk = ActionRisk.LOW
    executable: bool = False
    requires_approval: bool = False
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("title", "description")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @model_validator(mode="after")
    def _validate_executable_shape(self) -> PlannerStep:
        if self.executable and self.role != PlannedStepRole.TOOL:
            raise ValueError("only tool planner steps can be executable.")

        if self.executable and self.action_kind is None:
            raise ValueError("executable planner steps require action_kind.")

        if self.executable and self.capability is None:
            raise ValueError("executable planner steps require capability.")

        return self


class ActionPlanningRequest(ToolModel):
    """
    Request to create a safe multi-step action plan.

    This is still only planning. It does not execute shell, file, browser, IDE,
    desktop, or OS actions.
    """

    request_id: str = Field(default_factory=new_action_id)
    user_intent: str
    target_path: str | None = None
    search_query: str | None = None
    old_text: str | None = None
    new_text: str | None = None
    preferred_test_command: str | None = None
    approved: bool = False
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("request_id", "user_intent")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator(
        "target_path",
        "search_query",
        "old_text",
        "new_text",
        "preferred_test_command",
    )
    @classmethod
    def _clean_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None


class ActionPlanProposal(ToolModel):
    """
    Planner output.

    The action_plan is the executable proposal that policy/validation/runtime
    can consume later. planner_steps preserve the full human-readable chain.
    """

    proposal_id: str = Field(default_factory=new_action_plan_id)
    request_id: str
    decision: ActionPlanningDecision
    reason: ActionPlanningReason
    intent_kind: ActionPlanningIntentKind
    summary: str
    planner_steps: tuple[PlannerStep, ...] = ()
    action_plan: ActionPlan | None = None
    requires_approval: bool = False
    risk: ActionRisk = ActionRisk.LOW
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("proposal_id", "request_id", "summary")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @model_validator(mode="after")
    def _validate_proposal(self) -> ActionPlanProposal:
        if self.decision == ActionPlanningDecision.PROPOSED:
            if self.action_plan is None:
                raise ValueError("proposed planning decisions require action_plan.")

            if not self.planner_steps:
                raise ValueError("proposed planning decisions require planner_steps.")

        if self.decision != ActionPlanningDecision.PROPOSED:
            if self.action_plan is not None:
                raise ValueError("non-proposed planning decisions cannot include plan.")

        return self


@dataclass(frozen=True, slots=True)
class MultiStepActionPlannerConfig:
    """
    Configuration for MultiStepActionPlanner.
    """

    name: str = "multi_step_action_planner"
    default_test_command: str = "pytest"
    quality_gate_commands: tuple[str, ...] = (
        "ruff check .",
        "mypy .",
        "pytest",
    )

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if not self.default_test_command.strip():
            raise ValueError("default_test_command cannot be empty.")

        if not self.quality_gate_commands:
            raise ValueError("quality_gate_commands cannot be empty.")


@dataclass(frozen=True, slots=True)
class MultiStepActionPlannerSnapshot:
    """
    Planner diagnostics.
    """

    name: str
    plan_count: int
    proposed_count: int
    clarification_count: int
    rejected_count: int
    last_intent_kind: ActionPlanningIntentKind | None
    last_decision: ActionPlanningDecision | None


class MultiStepActionPlanner:
    """
    Conservative multi-step action planner.

    Responsibilities:
    - classify a safe user intent
    - create ordered planner steps
    - create an ActionPlan proposal for executable runtime steps
    - explain approval/risk
    - never execute actions

    Non-responsibilities:
    - no shell execution
    - no file writes
    - no browser actions
    - no IDE edits
    - no policy bypass
    """

    def __init__(
        self,
        *,
        config: MultiStepActionPlannerConfig | None = None,
    ) -> None:
        self._config = config or MultiStepActionPlannerConfig()
        self._config.validate()

        self._plan_count = 0
        self._proposed_count = 0
        self._clarification_count = 0
        self._rejected_count = 0
        self._last_intent_kind: ActionPlanningIntentKind | None = None
        self._last_decision: ActionPlanningDecision | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def propose(self, request: ActionPlanningRequest) -> ActionPlanProposal:
        """
        Create a multi-step plan proposal.

        This method never executes the plan.
        """

        self._plan_count += 1
        intent_kind = self._classify_request(request)
        self._last_intent_kind = intent_kind

        if intent_kind == ActionPlanningIntentKind.RUN_TESTS_AND_SUMMARIZE:
            proposal = self._plan_tests_and_summary(request)

        elif intent_kind == ActionPlanningIntentKind.RUN_QUALITY_GATE:
            proposal = self._plan_quality_gate(request)

        elif intent_kind == ActionPlanningIntentKind.OPEN_FILE:
            proposal = self._plan_open_file(request)

        elif intent_kind == ActionPlanningIntentKind.SEARCH_CODE:
            proposal = self._plan_search_code(request)

        elif intent_kind == ActionPlanningIntentKind.PREPARE_PATCH:
            proposal = self._plan_prepare_patch(request)

        else:
            proposal = self._needs_clarification(
                request=request,
                intent_kind=ActionPlanningIntentKind.UNKNOWN,
                reason=ActionPlanningReason.UNSUPPORTED_INTENT,
                summary=(
                    "I could not safely convert this intent into a governed "
                    "action plan."
                ),
            )

        self._record(proposal)

        return proposal

    def snapshot(self) -> MultiStepActionPlannerSnapshot:
        """
        Return planner diagnostics.
        """

        return MultiStepActionPlannerSnapshot(
            name=self.name,
            plan_count=self._plan_count,
            proposed_count=self._proposed_count,
            clarification_count=self._clarification_count,
            rejected_count=self._rejected_count,
            last_intent_kind=self._last_intent_kind,
            last_decision=self._last_decision,
        )

    def reset(self) -> None:
        """
        Reset planner diagnostics only.
        """

        self._plan_count = 0
        self._proposed_count = 0
        self._clarification_count = 0
        self._rejected_count = 0
        self._last_intent_kind = None
        self._last_decision = None

    def _plan_tests_and_summary(
        self,
        request: ActionPlanningRequest,
    ) -> ActionPlanProposal:
        command = request.preferred_test_command or self._config.default_test_command
        action_id = new_action_id()

        planner_steps = (
            PlannerStep(
                order=0,
                role=PlannedStepRole.TOOL,
                title="Run test suite",
                description=f"Run safe test command: {command}",
                action_kind=ActionKind.SHELL_COMMAND,
                capability=ToolCapability.RUN_SHELL_COMMAND,
                executable=True,
                metadata={"command": command},
            ),
            PlannerStep(
                order=1,
                role=PlannedStepRole.COGNITIVE,
                title="Capture output",
                description="Capture stdout and stderr from the shell runtime.",
                executable=False,
            ),
            PlannerStep(
                order=2,
                role=PlannedStepRole.COGNITIVE,
                title="Parse failures",
                description="Identify failed tests, error messages, and files.",
                executable=False,
            ),
            PlannerStep(
                order=3,
                role=PlannedStepRole.COGNITIVE,
                title="Summarize failures",
                description="Summarize failures clearly for the user.",
                executable=False,
            ),
            PlannerStep(
                order=4,
                role=PlannedStepRole.COGNITIVE,
                title="Suggest next action",
                description="Suggest the next safest debugging step.",
                executable=False,
            ),
        )
        steps = (
            ActionStep(
                action_id=action_id,
                order=0,
                kind=ActionKind.SHELL_COMMAND,
                capability=ToolCapability.RUN_SHELL_COMMAND,
                scope=ActionScope.SHELL,
                risk=ActionRisk.LOW,
                description=f"run tests using governed shell command: {command}",
                arguments={"command": command},
                timeout_ms=120_000,
                interruptible=True,
                rollback_supported=False,
            ),
        )

        return self._proposal(
            request=request,
            intent_kind=ActionPlanningIntentKind.RUN_TESTS_AND_SUMMARIZE,
            reason=ActionPlanningReason.TEST_PLAN_PROPOSED,
            summary="Proposed a safe test execution and failure-summary plan.",
            action_id=action_id,
            planner_steps=planner_steps,
            steps=steps,
            risk=ActionRisk.LOW,
            scope=ActionScope.SHELL,
        )

    def _plan_quality_gate(
        self,
        request: ActionPlanningRequest,
    ) -> ActionPlanProposal:
        action_id = new_action_id()
        planner_steps: list[PlannerStep] = []
        action_steps: list[ActionStep] = []

        for index, command in enumerate(self._config.quality_gate_commands):
            planner_steps.append(
                PlannerStep(
                    order=index,
                    role=PlannedStepRole.TOOL,
                    title=f"Run quality command {index + 1}",
                    description=f"Run safe quality command: {command}",
                    action_kind=ActionKind.SHELL_COMMAND,
                    capability=ToolCapability.RUN_SHELL_COMMAND,
                    executable=True,
                    metadata={"command": command},
                )
            )
            action_steps.append(
                ActionStep(
                    action_id=action_id,
                    order=index,
                    kind=ActionKind.SHELL_COMMAND,
                    capability=ToolCapability.RUN_SHELL_COMMAND,
                    scope=ActionScope.SHELL,
                    risk=ActionRisk.LOW,
                    description=f"run quality command: {command}",
                    arguments={"command": command},
                    timeout_ms=120_000,
                    interruptible=True,
                    rollback_supported=False,
                )
            )

        planner_steps.append(
            PlannerStep(
                order=len(planner_steps),
                role=PlannedStepRole.COGNITIVE,
                title="Summarize quality result",
                description="Summarize ruff, mypy, and pytest outcomes.",
                executable=False,
            )
        )

        return self._proposal(
            request=request,
            intent_kind=ActionPlanningIntentKind.RUN_QUALITY_GATE,
            reason=ActionPlanningReason.QUALITY_GATE_PLAN_PROPOSED,
            summary="Proposed a safe quality-gate plan.",
            action_id=action_id,
            planner_steps=tuple(planner_steps),
            steps=tuple(action_steps),
            risk=ActionRisk.LOW,
            scope=ActionScope.SHELL,
        )

    def _plan_open_file(self, request: ActionPlanningRequest) -> ActionPlanProposal:
        target_path = request.target_path or self._extract_path(request.user_intent)

        if target_path is None:
            return self._needs_clarification(
                request=request,
                intent_kind=ActionPlanningIntentKind.OPEN_FILE,
                reason=ActionPlanningReason.MISSING_TARGET,
                summary="Opening a file requires a target path.",
            )

        action_id = new_action_id()
        planner_steps = (
            PlannerStep(
                order=0,
                role=PlannedStepRole.TOOL,
                title="Open file",
                description=f"Open workspace file visibly: {target_path}",
                action_kind=ActionKind.IDE_OPEN_FILE,
                capability=ToolCapability.READ_FILE,
                executable=True,
                metadata={"path": target_path},
            ),
        )
        steps = (
            ActionStep(
                action_id=action_id,
                order=0,
                kind=ActionKind.IDE_OPEN_FILE,
                capability=ToolCapability.READ_FILE,
                scope=ActionScope.IDE,
                risk=ActionRisk.LOW,
                description=f"open file in governed IDE runtime: {target_path}",
                arguments={"path": target_path},
                timeout_ms=None,
                interruptible=True,
                rollback_supported=False,
            ),
        )

        return self._proposal(
            request=request,
            intent_kind=ActionPlanningIntentKind.OPEN_FILE,
            reason=ActionPlanningReason.OPEN_FILE_PLAN_PROPOSED,
            summary="Proposed a safe file-open plan.",
            action_id=action_id,
            planner_steps=planner_steps,
            steps=steps,
            risk=ActionRisk.LOW,
            scope=ActionScope.IDE,
        )

    def _plan_search_code(self, request: ActionPlanningRequest) -> ActionPlanProposal:
        query = request.search_query or self._extract_search_query(request.user_intent)

        if query is None:
            return self._needs_clarification(
                request=request,
                intent_kind=ActionPlanningIntentKind.SEARCH_CODE,
                reason=ActionPlanningReason.MISSING_TARGET,
                summary="Searching code requires a search query.",
            )

        action_id = new_action_id()
        planner_steps = (
            PlannerStep(
                order=0,
                role=PlannedStepRole.TOOL,
                title="Search project files",
                description=f"Search workspace code for: {query}",
                action_kind=ActionKind.SEARCH,
                capability=ToolCapability.SEARCH_FILES,
                executable=True,
                metadata={"query": query},
            ),
            PlannerStep(
                order=1,
                role=PlannedStepRole.COGNITIVE,
                title="Summarize matches",
                description="Summarize relevant files and possible next steps.",
                executable=False,
            ),
        )
        steps = (
            ActionStep(
                action_id=action_id,
                order=0,
                kind=ActionKind.SEARCH,
                capability=ToolCapability.SEARCH_FILES,
                scope=ActionScope.WORKSPACE,
                risk=ActionRisk.LOW,
                description=f"search project files for: {query}",
                arguments={"path": ".", "query": query},
                timeout_ms=30_000,
                interruptible=True,
                rollback_supported=False,
            ),
        )

        return self._proposal(
            request=request,
            intent_kind=ActionPlanningIntentKind.SEARCH_CODE,
            reason=ActionPlanningReason.SEARCH_CODE_PLAN_PROPOSED,
            summary="Proposed a safe project-search plan.",
            action_id=action_id,
            planner_steps=planner_steps,
            steps=steps,
            risk=ActionRisk.LOW,
            scope=ActionScope.WORKSPACE,
        )

    def _plan_prepare_patch(self, request: ActionPlanningRequest) -> ActionPlanProposal:
        if request.target_path is None:
            return self._needs_clarification(
                request=request,
                intent_kind=ActionPlanningIntentKind.PREPARE_PATCH,
                reason=ActionPlanningReason.MISSING_TARGET,
                summary="Preparing a patch requires a target path.",
            )

        if request.old_text is None or request.new_text is None:
            return self._needs_clarification(
                request=request,
                intent_kind=ActionPlanningIntentKind.PREPARE_PATCH,
                reason=ActionPlanningReason.MISSING_PATCH_TEXT,
                summary="Preparing a patch requires old_text and new_text.",
            )

        action_id = new_action_id()
        planner_steps = (
            PlannerStep(
                order=0,
                role=PlannedStepRole.TOOL,
                title="Prepare patch",
                description="Prepare a diff only. Do not modify the file yet.",
                action_kind=ActionKind.PATCH,
                capability=ToolCapability.PATCH_FILE,
                risk=ActionRisk.MEDIUM,
                executable=True,
                requires_approval=False,
                metadata={"path": request.target_path},
            ),
            PlannerStep(
                order=1,
                role=PlannedStepRole.USER,
                title="Review patch",
                description="User reviews patch before any application.",
                risk=ActionRisk.MEDIUM,
                executable=False,
                requires_approval=True,
            ),
        )
        steps = (
            ActionStep(
                action_id=action_id,
                order=0,
                kind=ActionKind.PATCH,
                capability=ToolCapability.PATCH_FILE,
                scope=ActionScope.IDE,
                risk=ActionRisk.MEDIUM,
                description="prepare patch diff without applying it",
                arguments={
                    "path": request.target_path,
                    "old_text": request.old_text,
                    "new_text": request.new_text,
                },
                timeout_ms=30_000,
                interruptible=True,
                rollback_supported=False,
            ),
        )

        return self._proposal(
            request=request,
            intent_kind=ActionPlanningIntentKind.PREPARE_PATCH,
            reason=ActionPlanningReason.PATCH_PLAN_PROPOSED,
            summary="Proposed a patch-preparation plan. No file modification.",
            action_id=action_id,
            planner_steps=planner_steps,
            steps=steps,
            risk=ActionRisk.MEDIUM,
            scope=ActionScope.IDE,
        )

    def _proposal(
        self,
        *,
        request: ActionPlanningRequest,
        intent_kind: ActionPlanningIntentKind,
        reason: ActionPlanningReason,
        summary: str,
        action_id: str,
        planner_steps: tuple[PlannerStep, ...],
        steps: tuple[ActionStep, ...],
        risk: ActionRisk,
        scope: ActionScope,
    ) -> ActionPlanProposal:
        requires_approval = risk in {ActionRisk.HIGH, ActionRisk.CRITICAL}

        action_plan = ActionPlan(
            action_id=action_id,
            goal=request.user_intent,
            steps=steps,
            risk=risk,
            scope=scope,
            requires_approval=requires_approval,
            permission_decision=(
                PermissionDecision.REQUIRE_APPROVAL
                if requires_approval
                else PermissionDecision.ALLOW
            ),
            status=ActionStatus.PLANNED,
            metadata={
                "planner": self.name,
                "request_id": request.request_id,
                "intent_kind": intent_kind.value,
            },
        )

        return ActionPlanProposal(
            request_id=request.request_id,
            decision=ActionPlanningDecision.PROPOSED,
            reason=reason,
            intent_kind=intent_kind,
            summary=summary,
            planner_steps=planner_steps,
            action_plan=action_plan,
            requires_approval=requires_approval,
            risk=risk,
            metadata={
                "planner": self.name,
                "planner_proposes_only": True,
                "runtime_execution": False,
            },
        )

    @staticmethod
    def _needs_clarification(
        *,
        request: ActionPlanningRequest,
        intent_kind: ActionPlanningIntentKind,
        reason: ActionPlanningReason,
        summary: str,
    ) -> ActionPlanProposal:
        return ActionPlanProposal(
            request_id=request.request_id,
            decision=ActionPlanningDecision.NEEDS_CLARIFICATION,
            reason=reason,
            intent_kind=intent_kind,
            summary=summary,
            action_plan=None,
            planner_steps=(),
            risk=ActionRisk.LOW,
        )

    def _record(self, proposal: ActionPlanProposal) -> None:
        self._last_decision = proposal.decision

        if proposal.decision == ActionPlanningDecision.PROPOSED:
            self._proposed_count += 1

        elif proposal.decision == ActionPlanningDecision.NEEDS_CLARIFICATION:
            self._clarification_count += 1

        else:
            self._rejected_count += 1

    @staticmethod
    def _classify_request(
        request: ActionPlanningRequest,
    ) -> ActionPlanningIntentKind:
        normalized = request.user_intent.casefold()

        if "quality" in normalized and "gate" in normalized:
            return ActionPlanningIntentKind.RUN_QUALITY_GATE

        if "ruff" in normalized and "mypy" in normalized and "pytest" in normalized:
            return ActionPlanningIntentKind.RUN_QUALITY_GATE

        if "run tests" in normalized or "pytest" in normalized:
            return ActionPlanningIntentKind.RUN_TESTS_AND_SUMMARIZE

        if "open" in normalized and (
            "file" in normalized
            or request.target_path is not None
            or MultiStepActionPlanner._extract_path(request.user_intent) is not None
        ):
            return ActionPlanningIntentKind.OPEN_FILE

        if "search" in normalized or "find" in normalized:
            return ActionPlanningIntentKind.SEARCH_CODE

        if (
            "patch" in normalized
            or "change" in normalized
            or "replace" in normalized
        ):
            return ActionPlanningIntentKind.PREPARE_PATCH

        return ActionPlanningIntentKind.UNKNOWN

    @staticmethod
    def _extract_path(intent: str) -> str | None:
        match = re.search(r"([A-Za-z0-9_./\\-]+\.(py|md|txt|json|yaml|yml))", intent)

        if match is None:
            return None

        return match.group(1).replace("\\", "/")

    @staticmethod
    def _extract_search_query(intent: str) -> str | None:
        cleaned = intent.strip()

        for prefix in ("search for", "search", "find usages of", "find"):
            if cleaned.casefold().startswith(prefix):
                query = cleaned[len(prefix) :].strip(" :")
                return query or None

        return cleaned or None