from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from typing import Protocol

from pydantic import Field, field_validator, model_validator

from jarvis.tools.audit import (
    ActionAuditActor,
    ActionAuditEventKind,
    ActionAuditLog,
    ActionAuditOutcome,
)
from jarvis.tools.autonomy import (
    AutonomousTaskRequest,
    AutonomousTaskResult,
    SafeAutonomousTaskRuntime,
)
from jarvis.tools.ids import new_action_result_id, utc_now
from jarvis.tools.models import ActionRisk, ToolModel
from jarvis.tools.planner import (
    ActionPlanningDecision,
    ActionPlanningRequest,
    ActionPlanProposal,
    MultiStepActionPlanner,
)
from jarvis.tools.smoke_runtime import (
    RealActionSmokeRequest,
    RealActionSmokeResult,
    RealActionSmokeRuntime,
)


class CognitionToolMode(StrEnum):
    """
    How far cognition is allowed to send a tool request.

    PLAN_ONLY is safest.
    SMOKE_EXECUTION uses the Step 16 governed pipeline.
    AUTONOMOUS_TASK uses the Step 17 bounded autonomy runtime.
    """

    PLAN_ONLY = "plan_only"
    SMOKE_EXECUTION = "smoke_execution"
    AUTONOMOUS_TASK = "autonomous_task"


class CognitionToolDecision(StrEnum):
    """
    Cognition-tool bridge decision.
    """

    PROPOSED = "proposed"
    EXECUTED = "executed"
    AUTONOMOUS_EXECUTED = "autonomous_executed"
    NEEDS_CLARIFICATION = "needs_clarification"
    BLOCKED = "blocked"
    FAILED = "failed"


class CognitionToolReason(StrEnum):
    """
    Machine-readable bridge reason.
    """

    PLAN_PROPOSED = "plan_proposed"
    SMOKE_EXECUTION_SUCCEEDED = "smoke_execution_succeeded"
    SMOKE_EXECUTION_FAILED = "smoke_execution_failed"
    AUTONOMOUS_TASK_SUCCEEDED = "autonomous_task_succeeded"
    AUTONOMOUS_TASK_FAILED = "autonomous_task_failed"
    EXECUTION_NOT_ALLOWED = "execution_not_allowed"
    AUTONOMY_NOT_ALLOWED = "autonomy_not_allowed"
    PLANNING_FAILED = "planning_failed"
    MISSING_AUTONOMOUS_TASK = "missing_autonomous_task"
    UNSUPPORTED_MODE = "unsupported_mode"
    BRIDGE_FAILED = "bridge_failed"


class CognitionToolSource(StrEnum):
    """
    Source of the tool request.
    """

    USER_TURN = "user_turn"
    COGNITION_PLAN = "cognition_plan"
    MEMORY_CONTEXT = "memory_context"
    BACKGROUND_REASONING = "background_reasoning"
    SYSTEM = "system"


class CognitionToolIntent(ToolModel):
    """
    Tool intent proposed by cognition.

    This is not execution. It is a typed request into the governed action
    pipeline.
    """

    intent_id: str = Field(default_factory=new_action_result_id)
    source: CognitionToolSource = CognitionToolSource.COGNITION_PLAN
    mode: CognitionToolMode = CognitionToolMode.PLAN_ONLY
    user_text: str
    goal: str
    planning_request: ActionPlanningRequest | None = None
    autonomous_task: AutonomousTaskRequest | None = None
    allow_execution: bool = False
    allow_autonomy: bool = False
    approved: bool = False
    write_memory: bool = False
    risk_ceiling: ActionRisk = ActionRisk.LOW
    correlation_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("intent_id", "user_text", "goal")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("correlation_id")
    @classmethod
    def _clean_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None

    @model_validator(mode="after")
    def _validate_mode_shape(self) -> CognitionToolIntent:
        if self.mode == CognitionToolMode.AUTONOMOUS_TASK:
            if self.autonomous_task is None:
                raise ValueError("autonomous mode requires autonomous_task.")

        return self

    def resolved_planning_request(self) -> ActionPlanningRequest:
        """
        Return explicit planning request or derive one from user_text.
        """

        if self.planning_request is not None:
            return self.planning_request

        return ActionPlanningRequest(user_intent=self.user_text)


class CognitionToolBridgeResult(ToolModel):
    """
    Result returned to cognition.

    This tells cognition what happened without letting cognition bypass the
    action pipeline.
    """

    result_id: str = Field(default_factory=new_action_result_id)
    intent_id: str
    decision: CognitionToolDecision
    reason: CognitionToolReason
    success: bool
    message: str
    proposal: ActionPlanProposal | None = None
    smoke_result: RealActionSmokeResult | None = None
    autonomous_result: AutonomousTaskResult | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("result_id", "intent_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class CognitionToolBridgeConfig:
    """
    Cognition-tool bridge configuration.
    """

    name: str = "cognition_tool_bridge"
    allow_smoke_execution: bool = True
    allow_autonomous_tasks: bool = False
    max_risk_without_approval: ActionRisk = ActionRisk.LOW

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")


@dataclass(frozen=True, slots=True)
class CognitionToolBridgeSnapshot:
    """
    Bridge diagnostics.
    """

    name: str
    request_count: int
    proposed_count: int
    executed_count: int
    autonomous_count: int
    blocked_count: int
    failed_count: int
    last_decision: CognitionToolDecision | None
    last_reason: CognitionToolReason | None
    last_error: str | None


class PlannerRunner(Protocol):
    """
    Narrow planner protocol.
    """

    def propose(self, request: ActionPlanningRequest) -> ActionPlanProposal:
        ...


class SmokeRunner(Protocol):
    """
    Narrow smoke runtime protocol.
    """

    def run(self, request: RealActionSmokeRequest) -> RealActionSmokeResult:
        ...


class AutonomousRunner(Protocol):
    """
    Narrow autonomous runtime protocol.
    """

    def run(self, request: AutonomousTaskRequest) -> AutonomousTaskResult:
        ...


class CognitionToolBridge:
    """
    Safe bridge from Cognition Runtime into Tool Runtime.

    Responsibilities:
    - accept typed cognition tool intent
    - convert intent into governed planner/smoke/autonomy requests
    - enforce no direct execution by cognition
    - enforce execution permissions
    - return observable result to cognition
    - audit every bridge decision

    Non-responsibilities:
    - no direct shell execution
    - no direct file execution
    - no direct browser/IDE/desktop control
    - no memory writes outside tool memory integration
    - no approval bypass
    - no autonomy bypass
    """

    def __init__(
        self,
        *,
        config: CognitionToolBridgeConfig | None = None,
        planner: PlannerRunner | None = None,
        smoke_runtime: SmokeRunner | None = None,
        autonomous_runtime: AutonomousRunner | None = None,
        audit_log: ActionAuditLog | None = None,
    ) -> None:
        self._config = config or CognitionToolBridgeConfig()
        self._config.validate()

        self._planner = planner or MultiStepActionPlanner()
        self._smoke_runtime = smoke_runtime or RealActionSmokeRuntime()
        self._autonomous_runtime = (
            autonomous_runtime or SafeAutonomousTaskRuntime()
        )
        self._audit_log = audit_log or ActionAuditLog()
        self._lock = RLock()

        self._request_count = 0
        self._proposed_count = 0
        self._executed_count = 0
        self._autonomous_count = 0
        self._blocked_count = 0
        self._failed_count = 0
        self._last_decision: CognitionToolDecision | None = None
        self._last_reason: CognitionToolReason | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def handle(self, intent: CognitionToolIntent) -> CognitionToolBridgeResult:
        """
        Handle a cognition-proposed tool intent.

        This method is the only safe entry point from cognition into Phase 5.
        """

        with self._lock:
            self._request_count += 1
            self._last_error = None

        self._audit_intent(intent)

        try:
            if intent.mode == CognitionToolMode.PLAN_ONLY:
                result = self._handle_plan_only(intent)

            elif intent.mode == CognitionToolMode.SMOKE_EXECUTION:
                result = self._handle_smoke_execution(intent)

            elif intent.mode == CognitionToolMode.AUTONOMOUS_TASK:
                result = self._handle_autonomous_task(intent)

            else:
                result = self._result(
                    intent=intent,
                    decision=CognitionToolDecision.BLOCKED,
                    reason=CognitionToolReason.UNSUPPORTED_MODE,
                    success=False,
                    message="unsupported cognition tool mode",
                )

            self._record(result)

            return result

        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"

            result = self._result(
                intent=intent,
                decision=CognitionToolDecision.FAILED,
                reason=CognitionToolReason.BRIDGE_FAILED,
                success=False,
                message=f"{type(exc).__name__}: {exc}",
            )
            self._record(result)

            return result

    def snapshot(self) -> CognitionToolBridgeSnapshot:
        """
        Return bridge diagnostics.
        """

        with self._lock:
            return CognitionToolBridgeSnapshot(
                name=self.name,
                request_count=self._request_count,
                proposed_count=self._proposed_count,
                executed_count=self._executed_count,
                autonomous_count=self._autonomous_count,
                blocked_count=self._blocked_count,
                failed_count=self._failed_count,
                last_decision=self._last_decision,
                last_reason=self._last_reason,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Reset diagnostics only.
        """

        with self._lock:
            self._request_count = 0
            self._proposed_count = 0
            self._executed_count = 0
            self._autonomous_count = 0
            self._blocked_count = 0
            self._failed_count = 0
            self._last_decision = None
            self._last_reason = None
            self._last_error = None

    def _handle_plan_only(
        self,
        intent: CognitionToolIntent,
    ) -> CognitionToolBridgeResult:
        proposal = self._planner.propose(intent.resolved_planning_request())

        if proposal.decision != ActionPlanningDecision.PROPOSED:
            return self._result(
                intent=intent,
                decision=CognitionToolDecision.NEEDS_CLARIFICATION,
                reason=CognitionToolReason.PLANNING_FAILED,
                success=False,
                message=proposal.summary,
                proposal=proposal,
            )

        self._audit_plan(intent=intent, proposal=proposal)

        return self._result(
            intent=intent,
            decision=CognitionToolDecision.PROPOSED,
            reason=CognitionToolReason.PLAN_PROPOSED,
            success=True,
            message="cognition tool plan proposed",
            proposal=proposal,
        )

    def _handle_smoke_execution(
        self,
        intent: CognitionToolIntent,
    ) -> CognitionToolBridgeResult:
        if not self._config.allow_smoke_execution or not intent.allow_execution:
            return self._result(
                intent=intent,
                decision=CognitionToolDecision.BLOCKED,
                reason=CognitionToolReason.EXECUTION_NOT_ALLOWED,
                success=False,
                message="smoke execution is not allowed for this intent",
            )

        smoke_request = RealActionSmokeRequest(
            planning_request=intent.resolved_planning_request(),
            approved=intent.approved,
            write_memory=intent.write_memory,
        )
        smoke_result = self._smoke_runtime.run(smoke_request)

        return self._result(
            intent=intent,
            decision=(
                CognitionToolDecision.EXECUTED
                if smoke_result.success
                else CognitionToolDecision.FAILED
            ),
            reason=(
                CognitionToolReason.SMOKE_EXECUTION_SUCCEEDED
                if smoke_result.success
                else CognitionToolReason.SMOKE_EXECUTION_FAILED
            ),
            success=smoke_result.success,
            message=smoke_result.message,
            proposal=smoke_result.proposal,
            smoke_result=smoke_result,
        )

    def _handle_autonomous_task(
        self,
        intent: CognitionToolIntent,
    ) -> CognitionToolBridgeResult:
        if not self._config.allow_autonomous_tasks or not intent.allow_autonomy:
            return self._result(
                intent=intent,
                decision=CognitionToolDecision.BLOCKED,
                reason=CognitionToolReason.AUTONOMY_NOT_ALLOWED,
                success=False,
                message="autonomous task execution is not allowed",
            )

        if intent.autonomous_task is None:
            return self._result(
                intent=intent,
                decision=CognitionToolDecision.BLOCKED,
                reason=CognitionToolReason.MISSING_AUTONOMOUS_TASK,
                success=False,
                message="autonomous task request is missing",
            )

        autonomous_result = self._autonomous_runtime.run(intent.autonomous_task)

        return self._result(
            intent=intent,
            decision=(
                CognitionToolDecision.AUTONOMOUS_EXECUTED
                if autonomous_result.success
                else CognitionToolDecision.FAILED
            ),
            reason=(
                CognitionToolReason.AUTONOMOUS_TASK_SUCCEEDED
                if autonomous_result.success
                else CognitionToolReason.AUTONOMOUS_TASK_FAILED
            ),
            success=autonomous_result.success,
            message=autonomous_result.message,
            autonomous_result=autonomous_result,
        )

    def _audit_intent(self, intent: CognitionToolIntent) -> None:
        self._audit_log.record(
            action_id=intent.intent_id,
            event_kind=ActionAuditEventKind.INTENT_RECEIVED,
            actor=ActionAuditActor.COGNITION,
            outcome=ActionAuditOutcome.INFO,
            message="cognition proposed tool intent",
            risk=intent.risk_ceiling,
            source_runtime=self.name,
            correlation_id=intent.correlation_id,
            data={
                "source": intent.source.value,
                "mode": intent.mode.value,
                "goal": intent.goal,
                "allow_execution": intent.allow_execution,
                "allow_autonomy": intent.allow_autonomy,
            },
        )

    def _audit_plan(
        self,
        *,
        intent: CognitionToolIntent,
        proposal: ActionPlanProposal,
    ) -> None:
        self._audit_log.record(
            action_id=intent.intent_id,
            event_kind=ActionAuditEventKind.PLAN_PROPOSED,
            actor=ActionAuditActor.PLANNER,
            outcome=ActionAuditOutcome.INFO,
            message=proposal.summary,
            risk=proposal.risk,
            source_runtime=self.name,
            correlation_id=intent.correlation_id,
            data={
                "proposal_id": proposal.proposal_id,
                "intent_kind": proposal.intent_kind.value,
                "requires_approval": proposal.requires_approval,
            },
        )

    @staticmethod
    def _result(
        *,
        intent: CognitionToolIntent,
        decision: CognitionToolDecision,
        reason: CognitionToolReason,
        success: bool,
        message: str,
        proposal: ActionPlanProposal | None = None,
        smoke_result: RealActionSmokeResult | None = None,
        autonomous_result: AutonomousTaskResult | None = None,
    ) -> CognitionToolBridgeResult:
        return CognitionToolBridgeResult(
            intent_id=intent.intent_id,
            decision=decision,
            reason=reason,
            success=success,
            message=message,
            proposal=proposal,
            smoke_result=smoke_result,
            autonomous_result=autonomous_result,
            metadata={
                "source": intent.source.value,
                "mode": intent.mode.value,
            },
        )

    def _record(self, result: CognitionToolBridgeResult) -> None:
        with self._lock:
            self._last_decision = result.decision
            self._last_reason = result.reason

            if result.decision == CognitionToolDecision.PROPOSED:
                self._proposed_count += 1

            elif result.decision == CognitionToolDecision.EXECUTED:
                self._executed_count += 1

            elif result.decision == CognitionToolDecision.AUTONOMOUS_EXECUTED:
                self._autonomous_count += 1

            elif result.decision == CognitionToolDecision.BLOCKED:
                self._blocked_count += 1

            elif result.decision in {
                CognitionToolDecision.FAILED,
                CognitionToolDecision.NEEDS_CLARIFICATION,
            }:
                self._failed_count += 1