from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import RLock

from pydantic import Field, field_validator

from jarvis.tools.ids import new_policy_decision_id, utc_now
from jarvis.tools.models import (
    ActionKind,
    ActionPlan,
    ActionRequest,
    ActionRisk,
    ActionScope,
    ActionStep,
    PermissionDecision,
    ToolCapability,
    ToolModel,
)
from jarvis.tools.registry import (
    ToolAvailability,
    ToolDescriptor,
    ToolHealth,
)


class PermissionScope(StrEnum):
    """
    Policy-level scope classification.

    This is separate from ActionScope because policy may reduce or constrain a
    broad action scope into a safer permission category.
    """

    READ_ONLY = "read_only"
    WORKSPACE_BOUNDED = "workspace_bounded"
    PROJECT_BOUNDED = "project_bounded"
    FILE_MUTATION = "file_mutation"
    SHELL_EXECUTION = "shell_execution"
    BROWSER_NETWORK = "browser_network"
    IDE_CONTROL = "ide_control"
    DESKTOP_CONTROL = "desktop_control"
    SYSTEM_CONTROL = "system_control"
    MEMORY_ACCESS = "memory_access"
    UNKNOWN = "unknown"


class PermissionReason(StrEnum):
    """
    Machine-readable reason for a permission decision.
    """

    SAFE_READ_ALLOWED = "safe_read_allowed"
    SAFE_SEARCH_ALLOWED = "safe_search_allowed"
    WORKSPACE_BOUNDED_ALLOWED = "workspace_bounded_allowed"
    MEMORY_GATEWAY_REQUIRED = "memory_gateway_required"
    WRITE_REQUIRES_CONFIRMATION = "write_requires_confirmation"
    PATCH_REQUIRES_CONFIRMATION = "patch_requires_confirmation"
    DELETE_REQUIRES_APPROVAL = "delete_requires_approval"
    MOVE_REQUIRES_CONFIRMATION = "move_requires_confirmation"
    COPY_REQUIRES_CONFIRMATION = "copy_requires_confirmation"
    SHELL_REQUIRES_APPROVAL = "shell_requires_approval"
    SHELL_READ_ONLY_ONLY = "shell_read_only_only"
    BROWSER_REQUIRES_CONFIRMATION = "browser_requires_confirmation"
    NETWORK_REQUIRES_APPROVAL = "network_requires_approval"
    IDE_EDIT_REQUIRES_CONFIRMATION = "ide_edit_requires_confirmation"
    DESKTOP_REQUIRES_APPROVAL = "desktop_requires_approval"
    SYSTEM_REQUIRES_APPROVAL = "system_requires_approval"
    HIGH_RISK_REQUIRES_APPROVAL = "high_risk_requires_approval"
    CRITICAL_RISK_DENIED = "critical_risk_denied"
    TOOL_NOT_REGISTERED = "tool_not_registered"
    TOOL_DISABLED = "tool_disabled"
    TOOL_UNAVAILABLE = "tool_unavailable"
    TOOL_UNHEALTHY = "tool_unhealthy"
    TOOL_CAPABILITY_MISMATCH = "tool_capability_mismatch"
    TOOL_SCOPE_MISMATCH = "tool_scope_mismatch"
    TOOL_ACTION_KIND_MISMATCH = "tool_action_kind_mismatch"
    TOOL_RISK_EXCEEDED = "tool_risk_exceeded"
    POLICY_DEFAULT_DENY = "policy_default_deny"


class RiskClassifierResult(ToolModel):
    """
    Result produced by the RiskClassifier.
    """

    risk: ActionRisk
    score: float = Field(ge=0.0, le=1.0)
    reasons: tuple[PermissionReason, ...]
    metadata: dict[str, object] = Field(default_factory=dict)


class PermissionPolicyEvaluation(ToolModel):
    """
    Final permission-policy result for an action request, step, or plan.
    """

    decision_id: str = Field(default_factory=new_policy_decision_id)
    action_id: str
    decision: PermissionDecision
    scope: PermissionScope
    risk: ActionRisk
    reason: PermissionReason
    explanation: str
    allowed: bool
    blocked: bool
    requires_approval: bool = False
    requires_confirmation: bool = False
    sandbox_only: bool = False
    read_only_only: bool = False
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("decision_id", "action_id", "explanation")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class PermissionPolicyConfig:
    """
    Configuration for PermissionPolicy.

    Conservative defaults protect the runtime before real execution exists.
    """

    name: str = "permission_policy"
    allow_workspace_reads: bool = True
    allow_project_reads: bool = True
    allow_memory_context_refs: bool = True
    allow_low_risk_system_query: bool = False
    allow_critical_actions: bool = False
    shell_low_risk_decision: PermissionDecision = (
        PermissionDecision.REQUIRE_CONFIRMATION
    )
    browser_low_risk_decision: PermissionDecision = (
        PermissionDecision.REQUIRE_CONFIRMATION
    )
    default_decision: PermissionDecision = PermissionDecision.DENY

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")


@dataclass(frozen=True, slots=True)
class PermissionPolicySnapshot:
    """
    Observable diagnostics for PermissionPolicy.
    """

    name: str
    evaluation_count: int
    allowed_count: int
    denied_count: int
    approval_required_count: int
    confirmation_required_count: int
    sandbox_only_count: int
    read_only_only_count: int
    last_decision: PermissionDecision | None
    last_reason: PermissionReason | None
    last_error: str | None


class RiskClassifier:
    """
    Deterministic risk classifier for action contracts.

    This classifier does not execute anything. It only estimates conservative
    risk from action kind, scope, capability, and explicit risk labels.
    """

    def classify_request(self, request: ActionRequest) -> RiskClassifierResult:
        return self.classify(
            kind=request.kind,
            scope=request.scope,
            capability=request.requested_capability,
            declared_risk=request.risk,
        )

    def classify_step(self, step: ActionStep) -> RiskClassifierResult:
        return self.classify(
            kind=step.kind,
            scope=step.scope,
            capability=step.capability,
            declared_risk=step.risk,
        )

    def classify_plan(self, plan: ActionPlan) -> RiskClassifierResult:
        step_results = tuple(self.classify_step(step) for step in plan.steps)
        risks = [plan.risk, *(result.risk for result in step_results)]
        risk = self._max_risk(risks)
        score = max((result.score for result in step_results), default=0.0)
        reasons = tuple(
            reason
            for result in step_results
            for reason in result.reasons
        )

        return RiskClassifierResult(
            risk=risk,
            score=max(score, self._score_for_risk(risk)),
            reasons=reasons or (self._reason_for_risk(risk),),
            metadata={
                "source": "action_plan",
                "step_count": len(plan.steps),
            },
        )

    def classify(
        self,
        *,
        kind: ActionKind,
        scope: ActionScope,
        capability: ToolCapability,
        declared_risk: ActionRisk,
    ) -> RiskClassifierResult:
        inferred_risk = self._infer_risk(
            kind=kind,
            scope=scope,
            capability=capability,
        )
        risk = self._max_risk((declared_risk, inferred_risk))
        reason = self._reason_for_risk(risk)

        return RiskClassifierResult(
            risk=risk,
            score=self._score_for_risk(risk),
            reasons=(reason,),
            metadata={
                "kind": kind.value,
                "scope": scope.value,
                "capability": capability.value,
                "declared_risk": declared_risk.value,
                "inferred_risk": inferred_risk.value,
            },
        )

    @staticmethod
    def _infer_risk(
        *,
        kind: ActionKind,
        scope: ActionScope,
        capability: ToolCapability,
    ) -> ActionRisk:
        if kind == ActionKind.DELETE:
            return ActionRisk.HIGH

        if kind == ActionKind.SHELL_COMMAND:
            return ActionRisk.HIGH

        if kind in {ActionKind.WRITE, ActionKind.PATCH, ActionKind.MOVE}:
            return ActionRisk.MEDIUM

        if kind == ActionKind.IDE_APPLY_PATCH:
            return ActionRisk.MEDIUM

        if scope in {ActionScope.SYSTEM, ActionScope.DESKTOP}:
            return ActionRisk.HIGH

        if scope == ActionScope.NETWORK:
            return ActionRisk.MEDIUM

        if capability in {
            ToolCapability.DELETE_FILE,
            ToolCapability.RUN_SHELL_COMMAND,
            ToolCapability.CONTROL_APPLICATION,
        }:
            return ActionRisk.HIGH

        if capability in {
            ToolCapability.WRITE_FILE,
            ToolCapability.PATCH_FILE,
            ToolCapability.APPLY_IDE_PATCH,
        }:
            return ActionRisk.MEDIUM

        return ActionRisk.LOW

    @staticmethod
    def _max_risk(risks: tuple[ActionRisk, ...] | list[ActionRisk]) -> ActionRisk:
        order = {
            ActionRisk.NONE: 0,
            ActionRisk.LOW: 1,
            ActionRisk.MEDIUM: 2,
            ActionRisk.HIGH: 3,
            ActionRisk.CRITICAL: 4,
        }

        return max(risks, key=lambda risk: order[risk])

    @staticmethod
    def _score_for_risk(risk: ActionRisk) -> float:
        return {
            ActionRisk.NONE: 0.0,
            ActionRisk.LOW: 0.25,
            ActionRisk.MEDIUM: 0.55,
            ActionRisk.HIGH: 0.82,
            ActionRisk.CRITICAL: 1.0,
        }[risk]

    @staticmethod
    def _reason_for_risk(risk: ActionRisk) -> PermissionReason:
        if risk == ActionRisk.CRITICAL:
            return PermissionReason.CRITICAL_RISK_DENIED

        if risk == ActionRisk.HIGH:
            return PermissionReason.HIGH_RISK_REQUIRES_APPROVAL

        if risk == ActionRisk.MEDIUM:
            return PermissionReason.WRITE_REQUIRES_CONFIRMATION

        return PermissionReason.SAFE_READ_ALLOWED


class PermissionPolicy:
    """
    Permission policy runtime for governed actions.

    Responsibilities:
    - classify action risk conservatively
    - validate registered tool constraints
    - decide allow/deny/approval/confirmation/sandbox/read-only
    - produce observable policy decisions

    Non-responsibilities:
    - no tool execution
    - no shell execution
    - no file writes
    - no approval UI
    - no validation-layer replacement
    """

    def __init__(
        self,
        *,
        config: PermissionPolicyConfig | None = None,
        risk_classifier: RiskClassifier | None = None,
    ) -> None:
        self._config = config or PermissionPolicyConfig()
        self._config.validate()

        self._risk_classifier = risk_classifier or RiskClassifier()
        self._lock = RLock()

        self._evaluation_count = 0
        self._allowed_count = 0
        self._denied_count = 0
        self._approval_required_count = 0
        self._confirmation_required_count = 0
        self._sandbox_only_count = 0
        self._read_only_only_count = 0
        self._last_decision: PermissionDecision | None = None
        self._last_reason: PermissionReason | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def evaluate_request(
        self,
        request: ActionRequest,
        *,
        tool: ToolDescriptor | None = None,
    ) -> PermissionPolicyEvaluation:
        classification = self._risk_classifier.classify_request(request)
        evaluation = self._evaluate(
            action_id=request.action_id,
            kind=request.kind,
            capability=request.requested_capability,
            scope=request.scope,
            risk=classification.risk,
            tool=tool,
            metadata={
                "source": "action_request",
                "intent": request.intent,
                "classifier_score": classification.score,
            },
        )
        self._record(evaluation)

        return evaluation

    def evaluate_step(
        self,
        step: ActionStep,
        *,
        tool: ToolDescriptor | None = None,
    ) -> PermissionPolicyEvaluation:
        classification = self._risk_classifier.classify_step(step)
        evaluation = self._evaluate(
            action_id=step.action_id,
            kind=step.kind,
            capability=step.capability,
            scope=step.scope,
            risk=classification.risk,
            tool=tool,
            metadata={
                "source": "action_step",
                "step_id": step.step_id,
                "description": step.description,
                "classifier_score": classification.score,
            },
        )
        self._record(evaluation)

        return evaluation

    def evaluate_plan(
        self,
        plan: ActionPlan,
        *,
        tool: ToolDescriptor | None = None,
    ) -> PermissionPolicyEvaluation:
        classification = self._risk_classifier.classify_plan(plan)

        if plan.permission_decision == PermissionDecision.DENY:
            evaluation = self._build_evaluation(
                action_id=plan.action_id,
                decision=PermissionDecision.DENY,
                scope=self._permission_scope(plan.scope),
                risk=classification.risk,
                reason=PermissionReason.POLICY_DEFAULT_DENY,
                explanation="plan is already marked as denied",
                metadata={
                    "source": "action_plan",
                    "plan_id": plan.plan_id,
                },
            )
            self._record(evaluation)

            return evaluation

        step_evaluations = tuple(
            self._evaluate(
                action_id=step.action_id,
                kind=step.kind,
                capability=step.capability,
                scope=step.scope,
                risk=self._risk_classifier.classify_step(step).risk,
                tool=tool,
                metadata={
                    "source": "action_plan_step",
                    "plan_id": plan.plan_id,
                    "step_id": step.step_id,
                    "step_order": step.order,
                    "description": step.description,
                },
            )
            for step in plan.steps
        )
        strongest = self._strongest_evaluation(
            action_id=plan.action_id,
            plan=plan,
            risk=classification.risk,
            evaluations=step_evaluations,
        )
        self._record(strongest)

        return strongest

    def _strongest_evaluation(
        self,
        *,
        action_id: str,
        plan: ActionPlan,
        risk: ActionRisk,
        evaluations: tuple[PermissionPolicyEvaluation, ...],
    ) -> PermissionPolicyEvaluation:
        if not evaluations:
            return self._build_evaluation(
                action_id=action_id,
                decision=PermissionDecision.DENY,
                scope=self._permission_scope(plan.scope),
                risk=risk,
                reason=PermissionReason.POLICY_DEFAULT_DENY,
                explanation="empty action plan denied by policy",
                metadata={
                    "source": "action_plan",
                    "plan_id": plan.plan_id,
                },
            )

        strongest = max(
            evaluations,
            key=lambda evaluation: self._decision_rank(evaluation.decision),
        )

        return self._build_evaluation(
            action_id=action_id,
            decision=strongest.decision,
            scope=self._permission_scope(plan.scope),
            risk=risk,
            reason=strongest.reason,
            explanation=(
                "plan permission derived from strongest step decision: "
                f"{strongest.explanation}"
            ),
            metadata={
                "source": "action_plan",
                "plan_id": plan.plan_id,
                "step_count": len(plan.steps),
                "strongest_step_decision": strongest.decision.value,
                "strongest_step_reason": strongest.reason.value,
            },
        )

    def snapshot(self) -> PermissionPolicySnapshot:
        with self._lock:
            return PermissionPolicySnapshot(
                name=self.name,
                evaluation_count=self._evaluation_count,
                allowed_count=self._allowed_count,
                denied_count=self._denied_count,
                approval_required_count=self._approval_required_count,
                confirmation_required_count=self._confirmation_required_count,
                sandbox_only_count=self._sandbox_only_count,
                read_only_only_count=self._read_only_only_count,
                last_decision=self._last_decision,
                last_reason=self._last_reason,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        with self._lock:
            self._evaluation_count = 0
            self._allowed_count = 0
            self._denied_count = 0
            self._approval_required_count = 0
            self._confirmation_required_count = 0
            self._sandbox_only_count = 0
            self._read_only_only_count = 0
            self._last_decision = None
            self._last_reason = None
            self._last_error = None

    def _evaluate(
        self,
        *,
        action_id: str,
        kind: ActionKind,
        capability: ToolCapability,
        scope: ActionScope,
        risk: ActionRisk,
        tool: ToolDescriptor | None,
        metadata: dict[str, object],
    ) -> PermissionPolicyEvaluation:
        tool_guard = self._tool_guard(
            kind=kind,
            capability=capability,
            scope=scope,
            risk=risk,
            tool=tool,
        )

        if tool_guard is not None:
            return self._build_evaluation(
                action_id=action_id,
                decision=tool_guard[0],
                scope=self._permission_scope(scope),
                risk=risk,
                reason=tool_guard[1],
                explanation=tool_guard[2],
                metadata=metadata,
            )

        if risk == ActionRisk.CRITICAL and not self._config.allow_critical_actions:
            return self._build_evaluation(
                action_id=action_id,
                decision=PermissionDecision.DENY,
                scope=self._permission_scope(scope),
                risk=risk,
                reason=PermissionReason.CRITICAL_RISK_DENIED,
                explanation="critical risk actions are denied by policy",
                metadata=metadata,
            )

        if risk == ActionRisk.HIGH:
            return self._build_evaluation(
                action_id=action_id,
                decision=PermissionDecision.REQUIRE_APPROVAL,
                scope=self._permission_scope(scope),
                risk=risk,
                reason=PermissionReason.HIGH_RISK_REQUIRES_APPROVAL,
                explanation="high risk action requires explicit approval",
                metadata=metadata,
            )

        return self._decision_by_kind(
            action_id=action_id,
            kind=kind,
            scope=scope,
            risk=risk,
            metadata=metadata,
        )

    def _decision_by_kind(
        self,
        *,
        action_id: str,
        kind: ActionKind,
        scope: ActionScope,
        risk: ActionRisk,
        metadata: dict[str, object],
    ) -> PermissionPolicyEvaluation:
        permission_scope = self._permission_scope(scope)

        if kind in {ActionKind.READ, ActionKind.SEARCH, ActionKind.SYSTEM_QUERY}:
            return self._read_decision(
                action_id=action_id,
                kind=kind,
                scope=scope,
                risk=risk,
                metadata=metadata,
            )

        if kind == ActionKind.WRITE:
            return self._build_evaluation(
                action_id=action_id,
                decision=PermissionDecision.REQUIRE_CONFIRMATION,
                scope=permission_scope,
                risk=risk,
                reason=PermissionReason.WRITE_REQUIRES_CONFIRMATION,
                explanation="write action requires confirmation",
                metadata=metadata,
            )

        if kind in {ActionKind.PATCH, ActionKind.IDE_APPLY_PATCH}:
            return self._build_evaluation(
                action_id=action_id,
                decision=PermissionDecision.REQUIRE_CONFIRMATION,
                scope=permission_scope,
                risk=risk,
                reason=PermissionReason.PATCH_REQUIRES_CONFIRMATION,
                explanation="patch action requires confirmation",
                metadata=metadata,
            )

        if kind == ActionKind.DELETE:
            return self._build_evaluation(
                action_id=action_id,
                decision=PermissionDecision.REQUIRE_APPROVAL,
                scope=permission_scope,
                risk=risk,
                reason=PermissionReason.DELETE_REQUIRES_APPROVAL,
                explanation="delete action requires explicit approval",
                metadata=metadata,
            )

        if kind == ActionKind.MOVE:
            return self._build_evaluation(
                action_id=action_id,
                decision=PermissionDecision.REQUIRE_CONFIRMATION,
                scope=permission_scope,
                risk=risk,
                reason=PermissionReason.MOVE_REQUIRES_CONFIRMATION,
                explanation="move action requires confirmation",
                metadata=metadata,
            )

        if kind == ActionKind.COPY:
            return self._build_evaluation(
                action_id=action_id,
                decision=PermissionDecision.REQUIRE_CONFIRMATION,
                scope=permission_scope,
                risk=risk,
                reason=PermissionReason.COPY_REQUIRES_CONFIRMATION,
                explanation="copy action requires confirmation",
                metadata=metadata,
            )

        if kind == ActionKind.SHELL_COMMAND:
            return self._shell_decision(
                action_id=action_id,
                scope=permission_scope,
                risk=risk,
                metadata=metadata,
            )

        if kind in {ActionKind.BROWSER_OPEN, ActionKind.BROWSER_SEARCH}:
            return self._build_evaluation(
                action_id=action_id,
                decision=self._config.browser_low_risk_decision,
                scope=permission_scope,
                risk=risk,
                reason=PermissionReason.BROWSER_REQUIRES_CONFIRMATION,
                explanation="browser or network action requires policy check",
                metadata=metadata,
            )

        if kind == ActionKind.IDE_OPEN_FILE:
            return self._build_evaluation(
                action_id=action_id,
                decision=PermissionDecision.ALLOW,
                scope=permission_scope,
                risk=risk,
                reason=PermissionReason.SAFE_READ_ALLOWED,
                explanation="opening an IDE file is read-oriented",
                metadata=metadata,
            )

        return self._build_evaluation(
            action_id=action_id,
            decision=self._config.default_decision,
            scope=permission_scope,
            risk=risk,
            reason=PermissionReason.POLICY_DEFAULT_DENY,
            explanation="policy default decision applied",
            metadata=metadata,
        )

    def _read_decision(
        self,
        *,
        action_id: str,
        kind: ActionKind,
        scope: ActionScope,
        risk: ActionRisk,
        metadata: dict[str, object],
    ) -> PermissionPolicyEvaluation:
        permission_scope = self._permission_scope(scope)

        if scope == ActionScope.MEMORY:
            decision = (
                PermissionDecision.ALLOW
                if self._config.allow_memory_context_refs
                else PermissionDecision.READ_ONLY_ONLY
            )

            return self._build_evaluation(
                action_id=action_id,
                decision=decision,
                scope=permission_scope,
                risk=risk,
                reason=PermissionReason.MEMORY_GATEWAY_REQUIRED,
                explanation="memory access must use memory gateway references",
                metadata=metadata,
            )

        if scope == ActionScope.WORKSPACE and self._config.allow_workspace_reads:
            return self._build_evaluation(
                action_id=action_id,
                decision=PermissionDecision.ALLOW,
                scope=permission_scope,
                risk=risk,
                reason=PermissionReason.WORKSPACE_BOUNDED_ALLOWED,
                explanation="workspace bounded read/search is allowed",
                metadata=metadata,
            )

        if scope == ActionScope.PROJECT and self._config.allow_project_reads:
            return self._build_evaluation(
                action_id=action_id,
                decision=PermissionDecision.ALLOW,
                scope=permission_scope,
                risk=risk,
                reason=PermissionReason.WORKSPACE_BOUNDED_ALLOWED,
                explanation="project bounded read/search is allowed",
                metadata=metadata,
            )

        if kind == ActionKind.SYSTEM_QUERY and self._config.allow_low_risk_system_query:
            return self._build_evaluation(
                action_id=action_id,
                decision=PermissionDecision.ALLOW,
                scope=permission_scope,
                risk=risk,
                reason=PermissionReason.SAFE_READ_ALLOWED,
                explanation="low risk system query is allowed",
                metadata=metadata,
            )

        return self._build_evaluation(
            action_id=action_id,
            decision=PermissionDecision.READ_ONLY_ONLY,
            scope=permission_scope,
            risk=risk,
            reason=PermissionReason.SAFE_READ_ALLOWED,
            explanation="read action is constrained to read-only mode",
            metadata=metadata,
        )

    def _shell_decision(
        self,
        *,
        action_id: str,
        scope: PermissionScope,
        risk: ActionRisk,
        metadata: dict[str, object],
    ) -> PermissionPolicyEvaluation:
        if self._config.shell_low_risk_decision == PermissionDecision.READ_ONLY_ONLY:
            reason = PermissionReason.SHELL_READ_ONLY_ONLY
            explanation = "shell action is constrained to read-only command mode"
        else:
            reason = PermissionReason.SHELL_REQUIRES_APPROVAL
            explanation = "shell command requires explicit policy handling"

        return self._build_evaluation(
            action_id=action_id,
            decision=self._config.shell_low_risk_decision,
            scope=scope,
            risk=risk,
            reason=reason,
            explanation=explanation,
            metadata=metadata,
        )

    def _tool_guard(
        self,
        *,
        kind: ActionKind,
        capability: ToolCapability,
        scope: ActionScope,
        risk: ActionRisk,
        tool: ToolDescriptor | None,
    ) -> tuple[PermissionDecision, PermissionReason, str] | None:
        if tool is None:
            return None

        if not tool.enabled:
            return (
                PermissionDecision.DENY,
                PermissionReason.TOOL_DISABLED,
                "registered tool is disabled",
            )

        if tool.availability != ToolAvailability.AVAILABLE:
            return (
                PermissionDecision.DENY,
                PermissionReason.TOOL_UNAVAILABLE,
                "registered tool is unavailable",
            )

        if tool.health == ToolHealth.UNHEALTHY:
            return (
                PermissionDecision.DENY,
                PermissionReason.TOOL_UNHEALTHY,
                "registered tool is unhealthy",
            )

        if not tool.supports_capability(capability):
            return (
                PermissionDecision.DENY,
                PermissionReason.TOOL_CAPABILITY_MISMATCH,
                "registered tool does not support requested capability",
            )

        if not tool.supports_action_kind(kind) and kind != ActionKind.COMPOSITE:
            return (
                PermissionDecision.DENY,
                PermissionReason.TOOL_ACTION_KIND_MISMATCH,
                "registered tool does not support requested action kind",
            )

        if not tool.supports_scope(scope):
            return (
                PermissionDecision.DENY,
                PermissionReason.TOOL_SCOPE_MISMATCH,
                "registered tool does not support requested scope",
            )

        if self._risk_rank(risk) > self._risk_rank(tool.max_risk):
            return (
                PermissionDecision.DENY,
                PermissionReason.TOOL_RISK_EXCEEDED,
                "requested action risk exceeds registered tool max risk",
            )

        return None

    def _build_evaluation(
        self,
        *,
        action_id: str,
        decision: PermissionDecision,
        scope: PermissionScope,
        risk: ActionRisk,
        reason: PermissionReason,
        explanation: str,
        metadata: dict[str, object],
    ) -> PermissionPolicyEvaluation:
        return PermissionPolicyEvaluation(
            action_id=action_id,
            decision=decision,
            scope=scope,
            risk=risk,
            reason=reason,
            explanation=explanation,
            allowed=decision == PermissionDecision.ALLOW,
            blocked=decision == PermissionDecision.DENY,
            requires_approval=decision == PermissionDecision.REQUIRE_APPROVAL,
            requires_confirmation=(
                decision == PermissionDecision.REQUIRE_CONFIRMATION
            ),
            sandbox_only=decision == PermissionDecision.SANDBOX_ONLY,
            read_only_only=decision == PermissionDecision.READ_ONLY_ONLY,
            metadata={
                "policy": self.name,
                **metadata,
            },
        )

    def _record(self, evaluation: PermissionPolicyEvaluation) -> None:
        with self._lock:
            self._evaluation_count += 1
            self._last_decision = evaluation.decision
            self._last_reason = evaluation.reason
            self._last_error = None

            if evaluation.allowed:
                self._allowed_count += 1

            if evaluation.blocked:
                self._denied_count += 1

            if evaluation.requires_approval:
                self._approval_required_count += 1

            if evaluation.requires_confirmation:
                self._confirmation_required_count += 1

            if evaluation.sandbox_only:
                self._sandbox_only_count += 1

            if evaluation.read_only_only:
                self._read_only_only_count += 1

    @staticmethod
    def _permission_scope(scope: ActionScope) -> PermissionScope:
        return {
            ActionScope.MEMORY: PermissionScope.MEMORY_ACCESS,
            ActionScope.WORKSPACE: PermissionScope.WORKSPACE_BOUNDED,
            ActionScope.PROJECT: PermissionScope.PROJECT_BOUNDED,
            ActionScope.FILE_SYSTEM: PermissionScope.FILE_MUTATION,
            ActionScope.SHELL: PermissionScope.SHELL_EXECUTION,
            ActionScope.BROWSER: PermissionScope.BROWSER_NETWORK,
            ActionScope.IDE: PermissionScope.IDE_CONTROL,
            ActionScope.DESKTOP: PermissionScope.DESKTOP_CONTROL,
            ActionScope.NETWORK: PermissionScope.BROWSER_NETWORK,
            ActionScope.SYSTEM: PermissionScope.SYSTEM_CONTROL,
        }.get(scope, PermissionScope.UNKNOWN)

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
    def _decision_rank(decision: PermissionDecision) -> int:
        return {
            PermissionDecision.ALLOW: 0,
            PermissionDecision.READ_ONLY_ONLY: 1,
            PermissionDecision.SANDBOX_ONLY: 2,
            PermissionDecision.REQUIRE_CONFIRMATION: 3,
            PermissionDecision.REQUIRE_APPROVAL: 4,
            PermissionDecision.DENY: 5,
        }[decision]