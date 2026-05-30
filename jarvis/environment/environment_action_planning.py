from __future__ import annotations

from enum import StrEnum
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.environment.environment_simulation import (
    EnvironmentSimulationRequest,
    EnvironmentSimulationResult,
    EnvironmentSimulationRuntime,
    SimulatedAction,
    SimulatedActionKind,
    SimulationDecision,
    SimulationRiskLevel,
)
from jarvis.environment.models import (
    EnvironmentSource,
    ScreenRegion,
    TrustCalibration,
)
from jarvis.environment.trust_runtime import TrustPolicyClassification
from jarvis.environment.ui_reasoning import (
    UIReasoningIntentKind,
    UIReasoningResult,
)
from jarvis.environment.visual_grounding import (
    GroundingDecision,
    VisualGroundingResult,
)
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class EnvironmentActionPlanStatus(StrEnum):
    CANDIDATE = "candidate"
    SIMULATED = "simulated"
    APPROVAL_REQUIRED = "approval_required"
    BLOCKED = "blocked"
    READY_FOR_EXECUTION = "ready_for_execution"
    FAILED = "failed"


class EnvironmentActionPlanDecision(StrEnum):
    ALLOW_EXECUTION_REQUEST = "allow_execution_request"
    REQUIRE_VERIFICATION = "require_verification"
    REQUIRE_APPROVAL = "require_approval"
    BLOCK = "block"
    ASK_USER = "ask_user"


class EnvironmentActionPlanReason(StrEnum):
    SESSION_CREATED = "session_created"
    CANDIDATE_PLAN_CREATED = "candidate_plan_created"
    SIMULATION_ALLOWED_PLAN = "simulation_allowed_plan"
    SIMULATION_REQUIRES_VERIFICATION = "simulation_requires_verification"
    SIMULATION_REQUIRES_APPROVAL = "simulation_requires_approval"
    SIMULATION_BLOCKED_PLAN = "simulation_blocked_plan"
    LOW_CONFIDENCE_GROUNDING = "low_confidence_grounding"
    POLICY_BLOCKED_PLAN = "policy_blocked_plan"
    APPROVAL_REQUIRED = "approval_required"
    SESSION_NOT_FOUND = "session_not_found"
    RUNTIME_RESET = "runtime_reset"


class EnvironmentActionPlanEventKind(StrEnum):
    SESSION_CREATED = "session_created"
    PLAN_CREATED = "plan_created"
    PLAN_BLOCKED = "plan_blocked"
    RUNTIME_RESET = "runtime_reset"


class EnvironmentPlanRiskLevel(StrEnum):
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    BLOCKED = "blocked"


class ApprovalRequirement(StrEnum):
    NONE = "none"
    SOFT_CONFIRMATION = "soft_confirmation"
    EXPLICIT_CONFIRMATION = "explicit_confirmation"
    ADMIN_CONFIRMATION = "admin_confirmation"
    BLOCKED = "blocked"


class ExpectedStatePlanStep(OrchestrationModel):
    """
    One expected state verification step after execution.

    This is for later verification runtime. It does not verify now.
    """

    step_id: str = Field(default_factory=lambda: f"expected_step_{uuid4().hex}")
    description: str
    verification_hint: str
    required: bool = True
    confidence: float = Field(default=0.80, ge=0.0, le=1.0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("step_id", "description", "verification_hint")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class ExpectedStatePlan(OrchestrationModel):
    """
    Expected state plan produced from simulation.

    Later Step 29 Verification Runtime will consume this.
    """

    expected_plan_id: str = Field(
        default_factory=lambda: f"expected_plan_{uuid4().hex}"
    )
    steps: tuple[ExpectedStatePlanStep, ...]
    requires_verification: bool = True
    confidence: float = Field(default=0.80, ge=0.0, le=1.0)
    created_at: object = Field(default_factory=utc_now)

    @field_validator("expected_plan_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _must_have_steps_when_verification_required(self) -> ExpectedStatePlan:
        if self.requires_verification and not self.steps:
            raise ValueError("verification-required expected plan needs steps.")

        return self


class PlanRiskAnalysis(OrchestrationModel):
    """
    Risk analysis for candidate action plan.
    """

    analysis_id: str = Field(default_factory=lambda: f"plan_risk_{uuid4().hex}")
    risk_level: EnvironmentPlanRiskLevel
    approval_requirement: ApprovalRequirement
    reason: str
    simulation_risk: SimulationRiskLevel | None = None
    confidence: float = Field(default=0.80, ge=0.0, le=1.0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("analysis_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class PolicyDecision(OrchestrationModel):
    """
    Final policy decision for action planning.

    This gates whether the plan can be handed to execution layer later.
    """

    decision_id: str = Field(default_factory=lambda: f"policy_decision_{uuid4().hex}")
    decision: EnvironmentActionPlanDecision
    policy: TrustPolicyClassification
    approval_requirement: ApprovalRequirement
    reason: str
    allow_execution_request: bool = False
    require_verification: bool = True
    created_at: object = Field(default_factory=utc_now)

    @field_validator("decision_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class CandidateActionPlan(OrchestrationModel):
    """
    Candidate plan before execution.

    This contains exactly one simulated environment action for now.
    Multi-step action graphs come later.
    """

    plan_id: str = Field(default_factory=lambda: f"env_action_plan_{uuid4().hex}")
    action: SimulatedAction
    grounding_result: VisualGroundingResult | None = None
    reasoning_result: UIReasoningResult | None = None
    created_from: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("plan_id", "created_from")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class PolicyAwareActionPlan(OrchestrationModel):
    """
    Final plan output of Step 23.

    Still not executed. Later runtimes consume only if allowed.
    """

    plan_id: str
    status: EnvironmentActionPlanStatus
    decision: EnvironmentActionPlanDecision
    reason: EnvironmentActionPlanReason
    candidate: CandidateActionPlan
    simulation: EnvironmentSimulationResult | None = None
    risk: PlanRiskAnalysis
    expected_state_plan: ExpectedStatePlan | None = None
    policy_decision: PolicyDecision
    trust: TrustCalibration
    safe_to_request_execution: bool
    requires_approval: bool
    requires_verification: bool
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("plan_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _execution_request_requires_simulation(self) -> PolicyAwareActionPlan:
        if self.safe_to_request_execution and self.simulation is None:
            raise ValueError("execution-requestable plan requires simulation.")

        return self


class EnvironmentActionPlanningRequest(OrchestrationModel):
    """
    Request to build policy-aware action plan.
    """

    request_id: str = Field(default_factory=lambda: f"env_plan_req_{uuid4().hex}")
    session_id: str
    workspace_id: str
    user_intent: str
    grounding_result: VisualGroundingResult | None = None
    reasoning_result: UIReasoningResult | None = None
    proposed_action_kind: SimulatedActionKind | None = None
    text_payload: str | None = None
    current_state_summary: str | None = None
    user_initiated: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: object = Field(default_factory=utc_now)

    @field_validator("request_id", "session_id", "workspace_id", "user_intent")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class EnvironmentActionPlanningSession(OrchestrationModel):
    """
    Action planning runtime session.
    """

    session_id: str = Field(default_factory=lambda: f"env_plan_session_{uuid4().hex}")
    workspace_id: str
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class EnvironmentActionPlanningRuntimeEvent(OrchestrationModel):
    """
    Runtime event for Step 23.
    """

    event_id: str = Field(default_factory=lambda: f"env_plan_event_{uuid4().hex}")
    kind: EnvironmentActionPlanEventKind
    reason: EnvironmentActionPlanReason
    session_id: str | None = None
    plan_id: str | None = None
    request_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class EnvironmentActionPlanningRuntimeSnapshot(OrchestrationModel):
    """
    Diagnostics for Phase 8 Step 23.
    """

    name: str
    session_count: int = Field(ge=0)
    plan_count: int = Field(ge=0)
    ready_count: int = Field(ge=0)
    approval_count: int = Field(ge=0)
    verification_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    safe_execution_request_count: int = Field(ge=0)
    runtime_event_count: int = Field(ge=0)
    last_reason: EnvironmentActionPlanReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        return _clean_required(value)


class EnvironmentActionPlanner:
    """
    Builds candidate action plan from user intent, grounding, and reasoning.

    This does not simulate and does not execute.
    """

    def build_candidate(
        self,
        request: EnvironmentActionPlanningRequest,
    ) -> CandidateActionPlan:
        action_kind = request.proposed_action_kind or _infer_action_kind(request)
        target_label = _target_label_from(request)
        source_policy = _source_policy_from(request)
        description = _description_for(
            action_kind=action_kind,
            user_intent=request.user_intent,
            target_label=target_label,
        )

        action = SimulatedAction(
            kind=action_kind,
            description=description,
            target_label=target_label,
            target_region=_target_region_from(request),
            text_payload=request.text_payload,
            source_policy=source_policy,
            user_initiated=request.user_initiated,
            metadata={"request_id": request.request_id},
        )

        return CandidateActionPlan(
            action=action,
            grounding_result=request.grounding_result,
            reasoning_result=request.reasoning_result,
            created_from=request.user_intent,
            metadata={"workspace_id": request.workspace_id},
        )


class SimulationGate:
    """
    Gates candidate action plan through EnvironmentSimulationRuntime.
    """

    def __init__(
        self,
        *,
        simulation_runtime: EnvironmentSimulationRuntime | None = None,
    ) -> None:
        self._simulation_runtime = (
            simulation_runtime or EnvironmentSimulationRuntime()
        )
        self._simulation_session_by_workspace: dict[str, str] = {}

    def simulate(
        self,
        *,
        request: EnvironmentActionPlanningRequest,
        candidate: CandidateActionPlan,
    ) -> EnvironmentSimulationResult:
        simulation_session_id = self._session_for(request.workspace_id)

        return self._simulation_runtime.simulate(
            EnvironmentSimulationRequest(
                session_id=simulation_session_id,
                workspace_id=request.workspace_id,
                action=candidate.action,
                grounding_result=request.grounding_result,
                reasoning_result=request.reasoning_result,
                current_state_summary=request.current_state_summary,
            )
        )

    def _session_for(self, workspace_id: str) -> str:
        session_id = self._simulation_session_by_workspace.get(workspace_id)

        if session_id is not None:
            return session_id

        session = self._simulation_runtime.create_session(workspace_id=workspace_id)
        self._simulation_session_by_workspace[workspace_id] = session.session_id

        return session.session_id


class PlanRiskAnalyzer:
    """
    Converts simulation result into action planning risk analysis.
    """

    def analyze(
        self,
        simulation: EnvironmentSimulationResult,
    ) -> PlanRiskAnalysis:
        if simulation.outcome is None:
            return PlanRiskAnalysis(
                risk_level=EnvironmentPlanRiskLevel.BLOCKED,
                approval_requirement=ApprovalRequirement.BLOCKED,
                reason="simulation did not produce outcome",
                confidence=0.0,
            )

        sim_risk = simulation.outcome.rollback_risk.risk_level
        approval = _approval_for_simulation(simulation)
        risk_level = _plan_risk_from_simulation(sim_risk)
        confidence = simulation.outcome.confidence.effective_score()

        return PlanRiskAnalysis(
            risk_level=risk_level,
            approval_requirement=approval,
            reason=simulation.outcome.rollback_risk.reason,
            simulation_risk=sim_risk,
            confidence=confidence,
            metadata={
                "simulation_status": simulation.status.value,
                "simulation_decision": simulation.decision.value,
            },
        )


class ExpectedStatePlanBuilder:
    """
    Converts simulation expected changes into verification plan.
    """

    def build(
        self,
        simulation: EnvironmentSimulationResult,
    ) -> ExpectedStatePlan | None:
        if simulation.outcome is None:
            return None

        steps = tuple(
            ExpectedStatePlanStep(
                description=change.description,
                verification_hint=change.verification_hint,
                required=True,
                confidence=change.confidence,
                metadata={"expected_kind": change.kind.value},
            )
            for change in simulation.outcome.expected_changes
        )
        confidence = simulation.outcome.confidence.state_confidence

        return ExpectedStatePlan(
            steps=steps,
            requires_verification=simulation.requires_verification,
            confidence=confidence,
        )


class PolicyAwarePlanBuilder:
    """
    Builds final policy-aware plan from candidate + simulation + risk.
    """

    def build(
        self,
        *,
        candidate: CandidateActionPlan,
        simulation: EnvironmentSimulationResult,
        risk: PlanRiskAnalysis,
        expected_state_plan: ExpectedStatePlan | None,
    ) -> PolicyAwareActionPlan:
        policy_decision = _policy_decision_for(simulation=simulation, risk=risk)
        status, reason = _status_reason_for(policy_decision)
        trust = _trust_for(simulation=simulation, risk=risk)
        requires_approval = (
            policy_decision.decision == EnvironmentActionPlanDecision.REQUIRE_APPROVAL
        )

        return PolicyAwareActionPlan(
            plan_id=candidate.plan_id,
            status=status,
            decision=policy_decision.decision,
            reason=reason,
            candidate=candidate,
            simulation=simulation,
            risk=risk,
            expected_state_plan=expected_state_plan,
            policy_decision=policy_decision,
            trust=trust,
            safe_to_request_execution=policy_decision.allow_execution_request,
            requires_approval=requires_approval,
            requires_verification=policy_decision.require_verification,
            message=_message_for(policy_decision),
        )


class EnvironmentActionPlanningRuntime:
    """
    Phase 8 Step 23 Action Planning With Simulation Gate.

    Responsibilities:
    - build candidate action plan
    - force simulation before plan can proceed
    - analyze risk
    - create expected state verification plan
    - apply policy and approval gate

    Non-responsibilities:
    - no execution
    - no clicking
    - no typing
    - no file mutation
    - no policy bypass
    """

    def __init__(
        self,
        *,
        name: str = "environment_action_planning_runtime",
        planner: EnvironmentActionPlanner | None = None,
        simulation_gate: SimulationGate | None = None,
        risk_analyzer: PlanRiskAnalyzer | None = None,
        expected_state_builder: ExpectedStatePlanBuilder | None = None,
        policy_builder: PolicyAwarePlanBuilder | None = None,
    ) -> None:
        cleaned = name.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._planner = planner or EnvironmentActionPlanner()
        self._simulation_gate = simulation_gate or SimulationGate()
        self._risk_analyzer = risk_analyzer or PlanRiskAnalyzer()
        self._expected_state_builder = (
            expected_state_builder or ExpectedStatePlanBuilder()
        )
        self._policy_builder = policy_builder or PolicyAwarePlanBuilder()
        self._sessions: dict[str, EnvironmentActionPlanningSession] = {}
        self._plans: list[PolicyAwareActionPlan] = []
        self._events: list[EnvironmentActionPlanningRuntimeEvent] = []
        self._lock = RLock()
        self._last_reason: EnvironmentActionPlanReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def create_session(
        self,
        *,
        workspace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> EnvironmentActionPlanningSession:
        session = EnvironmentActionPlanningSession(
            workspace_id=workspace_id,
            metadata=metadata or {},
        )
        event = self._event(
            kind=EnvironmentActionPlanEventKind.SESSION_CREATED,
            reason=EnvironmentActionPlanReason.SESSION_CREATED,
            session_id=session.session_id,
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._events.append(event)
            self._last_reason = event.reason

        return session

    def plan(
        self,
        request: EnvironmentActionPlanningRequest,
    ) -> PolicyAwareActionPlan:
        if self.session_for(request.session_id) is None:
            plan = self._failed_plan(request)
            self._record_plan(plan, request.session_id, request.request_id)
            return plan

        candidate = self._planner.build_candidate(request)
        simulation = self._simulation_gate.simulate(
            request=request,
            candidate=candidate,
        )
        risk = self._risk_analyzer.analyze(simulation)
        expected_state_plan = self._expected_state_builder.build(simulation)
        plan = self._policy_builder.build(
            candidate=candidate,
            simulation=simulation,
            risk=risk,
            expected_state_plan=expected_state_plan,
        )

        self._record_plan(plan, request.session_id, request.request_id)
        self._touch_session(request.session_id)

        return plan

    def session_for(
        self,
        session_id: str,
    ) -> EnvironmentActionPlanningSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def plans(self) -> tuple[PolicyAwareActionPlan, ...]:
        with self._lock:
            return tuple(self._plans)

    def events(self) -> tuple[EnvironmentActionPlanningRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def snapshot(self) -> EnvironmentActionPlanningRuntimeSnapshot:
        with self._lock:
            return EnvironmentActionPlanningRuntimeSnapshot(
                name=self.name,
                session_count=len(self._sessions),
                plan_count=len(self._plans),
                ready_count=sum(
                    1
                    for plan in self._plans
                    if plan.status == EnvironmentActionPlanStatus.READY_FOR_EXECUTION
                ),
                approval_count=sum(1 for plan in self._plans if plan.requires_approval),
                verification_count=sum(
                    1 for plan in self._plans if plan.requires_verification
                ),
                blocked_count=sum(
                    1
                    for plan in self._plans
                    if plan.status
                    in {
                        EnvironmentActionPlanStatus.BLOCKED,
                        EnvironmentActionPlanStatus.FAILED,
                    }
                ),
                safe_execution_request_count=sum(
                    1 for plan in self._plans if plan.safe_to_request_execution
                ),
                runtime_event_count=len(self._events),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        event = self._event(
            kind=EnvironmentActionPlanEventKind.RUNTIME_RESET,
            reason=EnvironmentActionPlanReason.RUNTIME_RESET,
        )

        with self._lock:
            self._sessions.clear()
            self._plans.clear()
            self._events.clear()
            self._events.append(event)
            self._last_reason = event.reason

    def _failed_plan(
        self,
        request: EnvironmentActionPlanningRequest,
    ) -> PolicyAwareActionPlan:
        action = SimulatedAction(
            kind=request.proposed_action_kind or SimulatedActionKind.UNKNOWN,
            description=request.user_intent,
            text_payload=request.text_payload
            if request.proposed_action_kind == SimulatedActionKind.TYPE_TEXT
            else None,
            source_policy=TrustPolicyClassification.BLOCKED,
        )
        candidate = CandidateActionPlan(
            action=action,
            grounding_result=request.grounding_result,
            reasoning_result=request.reasoning_result,
            created_from=request.user_intent,
        )
        risk = PlanRiskAnalysis(
            risk_level=EnvironmentPlanRiskLevel.BLOCKED,
            approval_requirement=ApprovalRequirement.BLOCKED,
            reason="planning session not found",
            confidence=0.0,
        )
        policy = PolicyDecision(
            decision=EnvironmentActionPlanDecision.BLOCK,
            policy=TrustPolicyClassification.BLOCKED,
            approval_requirement=ApprovalRequirement.BLOCKED,
            reason="planning session not found",
            allow_execution_request=False,
            require_verification=True,
        )

        return PolicyAwareActionPlan(
            plan_id=candidate.plan_id,
            status=EnvironmentActionPlanStatus.FAILED,
            decision=EnvironmentActionPlanDecision.BLOCK,
            reason=EnvironmentActionPlanReason.SESSION_NOT_FOUND,
            candidate=candidate,
            simulation=None,
            risk=risk,
            expected_state_plan=None,
            policy_decision=policy,
            trust=TrustCalibration(
                confidence=0.0,
                stability=0.0,
                ambiguity=1.0,
                source=EnvironmentSource.OS_OBSERVER,
                reason="planning session missing",
            ),
            safe_to_request_execution=False,
            requires_approval=True,
            requires_verification=True,
            message="environment action planning session not found",
        )

    def _record_plan(
        self,
        plan: PolicyAwareActionPlan,
        session_id: str,
        request_id: str,
    ) -> None:
        event = self._event(
            kind=(
                EnvironmentActionPlanEventKind.PLAN_CREATED
                if plan.safe_to_request_execution
                else EnvironmentActionPlanEventKind.PLAN_BLOCKED
            ),
            reason=plan.reason,
            session_id=session_id,
            plan_id=plan.plan_id,
            request_id=request_id,
            metadata={
                "status": plan.status.value,
                "decision": plan.decision.value,
            },
        )

        with self._lock:
            self._plans.append(plan)
            self._events.append(event)
            self._last_reason = plan.reason

    def _touch_session(self, session_id: str) -> None:
        session = self._sessions.get(session_id)

        if session is None:
            return

        self._sessions[session_id] = session.model_copy(
            update={"updated_at": utc_now()}
        )

    @staticmethod
    def _event(
        *,
        kind: EnvironmentActionPlanEventKind,
        reason: EnvironmentActionPlanReason,
        session_id: str | None = None,
        plan_id: str | None = None,
        request_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EnvironmentActionPlanningRuntimeEvent:
        return EnvironmentActionPlanningRuntimeEvent(
            kind=kind,
            reason=reason,
            session_id=session_id,
            plan_id=plan_id,
            request_id=request_id,
            metadata=metadata or {},
        )


def _infer_action_kind(
    request: EnvironmentActionPlanningRequest,
) -> SimulatedActionKind:
    text = request.user_intent.lower()

    if request.reasoning_result is not None:
        intent = request.reasoning_result.intent.kind

        if intent == UIReasoningIntentKind.OPEN:
            return SimulatedActionKind.OPEN

        if intent == UIReasoningIntentKind.RUN:
            return SimulatedActionKind.SUBMIT

        if intent == UIReasoningIntentKind.COPY:
            return SimulatedActionKind.COPY

        if intent == UIReasoningIntentKind.FOCUS:
            return SimulatedActionKind.FOCUS

    if "delete" in text or "remove" in text:
        return SimulatedActionKind.DELETE

    if "type" in text or "write" in text:
        return SimulatedActionKind.TYPE_TEXT

    if "submit" in text or "send" in text:
        return SimulatedActionKind.SUBMIT

    if "close" in text:
        return SimulatedActionKind.CLOSE

    if "move" in text:
        return SimulatedActionKind.MOVE_FILE

    if "setting" in text or "disable" in text or "enable" in text:
        return SimulatedActionKind.CHANGE_SETTING

    if "open" in text:
        return SimulatedActionKind.OPEN

    if "focus" in text:
        return SimulatedActionKind.FOCUS

    if "copy" in text:
        return SimulatedActionKind.COPY

    return SimulatedActionKind.CLICK


def _target_label_from(
    request: EnvironmentActionPlanningRequest,
) -> str | None:
    if request.grounding_result is not None:
        if request.grounding_result.selected is not None:
            return request.grounding_result.selected.label

    if request.reasoning_result is not None:
        if request.reasoning_result.resolved_target is not None:
            return request.reasoning_result.resolved_target.label

    return None


def _target_region_from(
    request: EnvironmentActionPlanningRequest,
) -> ScreenRegion | None:
    if request.grounding_result is None:
        return None

    if request.grounding_result.selected is None:
        return None

    return request.grounding_result.selected.region


def _source_policy_from(
    request: EnvironmentActionPlanningRequest,
) -> TrustPolicyClassification:
    if request.grounding_result is not None:
        if request.grounding_result.decision == GroundingDecision.BLOCKED:
            return TrustPolicyClassification.BLOCKED

        if not request.grounding_result.safe_for_action_planning:
            return TrustPolicyClassification.VERIFY_FIRST

    if request.reasoning_result is not None:
        if not request.reasoning_result.safe_for_action_planning:
            return TrustPolicyClassification.VERIFY_FIRST

    return TrustPolicyClassification.SAFE


def _description_for(
    *,
    action_kind: SimulatedActionKind,
    user_intent: str,
    target_label: str | None,
) -> str:
    target = target_label or "unknown target"

    return f"{action_kind.value}: {user_intent} -> {target}"


def _approval_for_simulation(
    simulation: EnvironmentSimulationResult,
) -> ApprovalRequirement:
    if simulation.outcome is None:
        return ApprovalRequirement.BLOCKED

    if simulation.decision == SimulationDecision.BLOCK_ACTION:
        return ApprovalRequirement.BLOCKED

    if simulation.decision == SimulationDecision.REQUIRE_APPROVAL:
        if (
            simulation.outcome.rollback_risk.risk_level
            == SimulationRiskLevel.IRREVERSIBLE
        ):
            return ApprovalRequirement.EXPLICIT_CONFIRMATION

        return ApprovalRequirement.SOFT_CONFIRMATION

    if simulation.decision == SimulationDecision.VERIFY_FIRST:
        return ApprovalRequirement.SOFT_CONFIRMATION

    return ApprovalRequirement.NONE


def _plan_risk_from_simulation(
    risk: SimulationRiskLevel,
) -> EnvironmentPlanRiskLevel:
    if risk == SimulationRiskLevel.BLOCKED:
        return EnvironmentPlanRiskLevel.BLOCKED

    if risk == SimulationRiskLevel.IRREVERSIBLE:
        return EnvironmentPlanRiskLevel.CRITICAL

    if risk == SimulationRiskLevel.HIGH:
        return EnvironmentPlanRiskLevel.HIGH

    if risk == SimulationRiskLevel.MEDIUM:
        return EnvironmentPlanRiskLevel.MEDIUM

    if risk == SimulationRiskLevel.LOW:
        return EnvironmentPlanRiskLevel.LOW

    return EnvironmentPlanRiskLevel.SAFE


def _policy_decision_for(
    *,
    simulation: EnvironmentSimulationResult,
    risk: PlanRiskAnalysis,
) -> PolicyDecision:
    if simulation.outcome is None:
        return PolicyDecision(
            decision=EnvironmentActionPlanDecision.BLOCK,
            policy=TrustPolicyClassification.BLOCKED,
            approval_requirement=ApprovalRequirement.BLOCKED,
            reason="missing simulation outcome",
            allow_execution_request=False,
            require_verification=True,
        )

    if simulation.decision == SimulationDecision.BLOCK_ACTION:
        return PolicyDecision(
            decision=EnvironmentActionPlanDecision.BLOCK,
            policy=TrustPolicyClassification.BLOCKED,
            approval_requirement=ApprovalRequirement.BLOCKED,
            reason="simulation blocked action",
            allow_execution_request=False,
            require_verification=True,
        )

    if simulation.decision == SimulationDecision.REQUIRE_APPROVAL:
        return PolicyDecision(
            decision=EnvironmentActionPlanDecision.REQUIRE_APPROVAL,
            policy=TrustPolicyClassification.VERIFY_FIRST,
            approval_requirement=risk.approval_requirement,
            reason="simulation requires approval",
            allow_execution_request=False,
            require_verification=True,
        )

    if simulation.decision == SimulationDecision.ASK_USER:
        return PolicyDecision(
            decision=EnvironmentActionPlanDecision.ASK_USER,
            policy=TrustPolicyClassification.REVIEW,
            approval_requirement=ApprovalRequirement.SOFT_CONFIRMATION,
            reason="simulation confidence too low",
            allow_execution_request=False,
            require_verification=True,
        )

    if simulation.decision == SimulationDecision.VERIFY_FIRST:
        return PolicyDecision(
            decision=EnvironmentActionPlanDecision.REQUIRE_VERIFICATION,
            policy=TrustPolicyClassification.REVIEW,
            approval_requirement=risk.approval_requirement,
            reason="simulation requires verification",
            allow_execution_request=True,
            require_verification=True,
        )

    return PolicyDecision(
        decision=EnvironmentActionPlanDecision.ALLOW_EXECUTION_REQUEST,
        policy=TrustPolicyClassification.SAFE,
        approval_requirement=ApprovalRequirement.NONE,
        reason="simulation allows planning",
        allow_execution_request=True,
        require_verification=False,
    )


def _status_reason_for(
    policy: PolicyDecision,
) -> tuple[EnvironmentActionPlanStatus, EnvironmentActionPlanReason]:
    if policy.decision == EnvironmentActionPlanDecision.ALLOW_EXECUTION_REQUEST:
        return (
            EnvironmentActionPlanStatus.READY_FOR_EXECUTION,
            EnvironmentActionPlanReason.SIMULATION_ALLOWED_PLAN,
        )

    if policy.decision == EnvironmentActionPlanDecision.REQUIRE_VERIFICATION:
        return (
            EnvironmentActionPlanStatus.SIMULATED,
            EnvironmentActionPlanReason.SIMULATION_REQUIRES_VERIFICATION,
        )

    if policy.decision == EnvironmentActionPlanDecision.REQUIRE_APPROVAL:
        return (
            EnvironmentActionPlanStatus.APPROVAL_REQUIRED,
            EnvironmentActionPlanReason.SIMULATION_REQUIRES_APPROVAL,
        )

    if policy.decision == EnvironmentActionPlanDecision.ASK_USER:
        return (
            EnvironmentActionPlanStatus.BLOCKED,
            EnvironmentActionPlanReason.LOW_CONFIDENCE_GROUNDING,
        )

    return (
        EnvironmentActionPlanStatus.BLOCKED,
        EnvironmentActionPlanReason.SIMULATION_BLOCKED_PLAN,
    )


def _trust_for(
    *,
    simulation: EnvironmentSimulationResult,
    risk: PlanRiskAnalysis,
) -> TrustCalibration:
    confidence = risk.confidence

    if simulation.outcome is not None:
        confidence = min(confidence, simulation.outcome.confidence.effective_score())

    return TrustCalibration(
        confidence=confidence,
        stability=max(0.0, min(1.0, confidence + 0.05)),
        ambiguity=1.0 - confidence,
        source=EnvironmentSource.OS_OBSERVER,
        reason="policy-aware environment action plan",
    )


def _message_for(policy: PolicyDecision) -> str:
    return (
        f"action plan decision={policy.decision.value}; "
        f"approval={policy.approval_requirement.value}; "
        f"verification_required={policy.require_verification}"
    )


def _clean_required(value: str) -> str:
    cleaned = value.strip()

    if not cleaned:
        raise ValueError("field cannot be empty.")

    return cleaned