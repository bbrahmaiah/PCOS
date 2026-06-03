from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from uuid import uuid4

from jarvis.cognitive.contracts import (
    Goal,
    GoalPriority,
    GoalStatus,
    Plan,
    PlanningState,
    PlanRisk,
    PlanStep,
    PlanStepStatus,
    utc_now,
)


class PlanningRuntimeStatus(StrEnum):
    READY = "ready"
    DEGRADED = "degraded"
    BLOCKED = "blocked"


class PlanningOperation(StrEnum):
    CREATE_PLAN = "create_plan"
    UPDATE_STEP = "update_step"
    ACTIVATE_PLAN = "activate_plan"
    BLOCK_STEP = "block_step"
    COMPLETE_STEP = "complete_step"
    CANCEL_PLAN = "cancel_plan"
    RECALL = "recall"
    CLEAR = "clear"


class PlanIntentKind(StrEnum):
    GENERAL = "general"
    DEVELOPER = "developer"
    RESEARCH = "research"
    SYSTEM_RECOVERY = "system_recovery"
    SAFETY = "safety"
    COMMUNICATION = "communication"
    SIMULATION = "simulation"


@dataclass(frozen=True, slots=True)
class PlanCreateRequest:
    goal: Goal
    intent_kind: PlanIntentKind = PlanIntentKind.GENERAL
    title: str | None = None
    requested_steps: tuple[str, ...] = ()
    max_steps: int = 8
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.title is not None and not self.title.strip():
            raise ValueError("plan create title cannot be empty.")
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1.")


@dataclass(frozen=True, slots=True)
class PlanStepUpdateRequest:
    plan_id: str
    step_id: str
    status: PlanStepStatus
    title: str | None = None
    description: str | None = None
    risk: PlanRisk | None = None
    requires_approval: bool | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.plan_id.strip():
            raise ValueError("plan step update plan_id cannot be empty.")
        if not self.step_id.strip():
            raise ValueError("plan step update step_id cannot be empty.")
        if self.title is not None and not self.title.strip():
            raise ValueError("plan step update title cannot be empty.")
        if self.description is not None and not self.description.strip():
            raise ValueError("plan step update description cannot be empty.")


@dataclass(frozen=True, slots=True)
class PlanRecallRequest:
    goal_id: str | None = None
    query: str = ""
    active_only: bool = False
    include_cancelled: bool = False
    limit: int = 10
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.goal_id is not None and not self.goal_id.strip():
            raise ValueError("plan recall goal_id cannot be empty.")
        if self.limit < 1:
            raise ValueError("plan recall limit must be at least 1.")


@dataclass(frozen=True, slots=True)
class PlanningRuntimeResult:
    status: PlanningRuntimeStatus
    operation: PlanningOperation
    state: PlanningState
    plan: Plan | None
    plans: tuple[Plan, ...]
    reason: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == PlanningRuntimeStatus.READY


@dataclass(frozen=True, slots=True)
class PlanningRuntimeSnapshot:
    status: PlanningRuntimeStatus
    state: PlanningState
    plan_count: int
    active_plan_id: str | None
    pending_step_count: int
    ready_step_count: int
    blocked_step_count: int
    completed_step_count: int
    approval_required_count: int
    operation_count: int
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


class PlanningRuntime:
    """
    Phase 9 / Step 49D Planning Runtime.

    Planning Runtime converts goals into safe, typed, reviewable plans:
    - goal -> plan
    - plan -> steps
    - steps -> risk
    - risk -> approval requirement
    - plan -> verification step

    It intentionally does not execute:
    - no tool calls
    - no laptop control
    - no memory writes
    - no autonomous execution
    """

    def __init__(self) -> None:
        self._state = PlanningState()
        self._operation_count = 0

    @property
    def state(self) -> PlanningState:
        return self._state

    def create_plan(
        self,
        request: PlanCreateRequest,
    ) -> PlanningRuntimeResult:
        self._operation_count += 1

        if request.goal.status in {
            GoalStatus.COMPLETED,
            GoalStatus.CANCELLED,
        }:
            return PlanningRuntimeResult(
                status=PlanningRuntimeStatus.BLOCKED,
                operation=PlanningOperation.CREATE_PLAN,
                state=self._state,
                plan=None,
                plans=(),
                reason=(
                    "cannot create plan for completed or cancelled goal"
                ),
                created_at=utc_now(),
                metadata={
                    **request.metadata,
                    "goal_id": request.goal.goal_id,
                    "goal_status": request.goal.status.value,
                },
            )

        plan = _build_plan(request)
        plans = (plan, *self._state.plans)
        self._state = PlanningState(
            plans=plans,
            active_plan_id=plan.plan_id,
            created_at=utc_now(),
        )

        return PlanningRuntimeResult(
            status=PlanningRuntimeStatus.READY,
            operation=PlanningOperation.CREATE_PLAN,
            state=self._state,
            plan=plan,
            plans=(plan,),
            reason="plan created and activated",
            created_at=utc_now(),
            metadata={
                **request.metadata,
                "intent_kind": request.intent_kind.value,
                "step_count": len(plan.steps),
                "requires_approval": plan.requires_approval,
            },
        )

    def update_step(
        self,
        request: PlanStepUpdateRequest,
    ) -> PlanningRuntimeResult:
        self._operation_count += 1
        plan = _find_plan(self._state.plans, request.plan_id)

        if plan is None:
            return PlanningRuntimeResult(
                status=PlanningRuntimeStatus.BLOCKED,
                operation=PlanningOperation.UPDATE_STEP,
                state=self._state,
                plan=None,
                plans=(),
                reason=f"plan not found: {request.plan_id}",
                created_at=utc_now(),
                metadata=request.metadata,
            )

        step = _find_step(plan.steps, request.step_id)
        if step is None:
            return PlanningRuntimeResult(
                status=PlanningRuntimeStatus.BLOCKED,
                operation=PlanningOperation.UPDATE_STEP,
                state=self._state,
                plan=plan,
                plans=(plan,),
                reason=f"plan step not found: {request.step_id}",
                created_at=utc_now(),
                metadata=request.metadata,
            )

        updated_step = PlanStep(
            step_id=step.step_id,
            title=request.title.strip() if request.title else step.title,
            description=(
                request.description.strip()
                if request.description
                else step.description
            ),
            status=request.status,
            risk=request.risk or step.risk,
            requires_approval=(
                request.requires_approval
                if request.requires_approval is not None
                else step.requires_approval
            ),
            created_at=step.created_at,
            metadata={**step.metadata, **request.metadata},
        )
        updated_plan = Plan(
            plan_id=plan.plan_id,
            goal_id=plan.goal_id,
            title=plan.title,
            steps=_replace_step(plan.steps, updated_step),
            created_at=plan.created_at,
            metadata=plan.metadata,
        )
        self._state = PlanningState(
            plans=_replace_plan(self._state.plans, updated_plan),
            active_plan_id=self._state.active_plan_id,
            created_at=utc_now(),
        )

        return PlanningRuntimeResult(
            status=PlanningRuntimeStatus.READY,
            operation=_operation_for_step_status(request.status),
            state=self._state,
            plan=updated_plan,
            plans=(updated_plan,),
            reason="plan step updated",
            created_at=utc_now(),
            metadata={
                **request.metadata,
                "step_id": request.step_id,
                "step_status": request.status.value,
            },
        )

    def activate_plan(self, plan_id: str) -> PlanningRuntimeResult:
        self._operation_count += 1

        if not plan_id.strip():
            raise ValueError("activate plan_id cannot be empty.")

        plan = _find_plan(self._state.plans, plan_id)
        if plan is None:
            return PlanningRuntimeResult(
                status=PlanningRuntimeStatus.BLOCKED,
                operation=PlanningOperation.ACTIVATE_PLAN,
                state=self._state,
                plan=None,
                plans=(),
                reason=f"plan not found: {plan_id}",
                created_at=utc_now(),
            )

        self._state = PlanningState(
            plans=self._state.plans,
            active_plan_id=plan.plan_id,
            created_at=utc_now(),
        )
        return PlanningRuntimeResult(
            status=PlanningRuntimeStatus.READY,
            operation=PlanningOperation.ACTIVATE_PLAN,
            state=self._state,
            plan=plan,
            plans=(plan,),
            reason="plan activated",
            created_at=utc_now(),
        )

    def block_step(
        self,
        *,
        plan_id: str,
        step_id: str,
        reason: str,
    ) -> PlanningRuntimeResult:
        if not reason.strip():
            raise ValueError("block step reason cannot be empty.")

        return self.update_step(
            PlanStepUpdateRequest(
                plan_id=plan_id,
                step_id=step_id,
                status=PlanStepStatus.BLOCKED,
                metadata={"blocked_reason": reason.strip()},
            )
        )

    def complete_step(
        self,
        *,
        plan_id: str,
        step_id: str,
    ) -> PlanningRuntimeResult:
        return self.update_step(
            PlanStepUpdateRequest(
                plan_id=plan_id,
                step_id=step_id,
                status=PlanStepStatus.COMPLETED,
            )
        )

    def cancel_plan(self, plan_id: str) -> PlanningRuntimeResult:
        self._operation_count += 1

        if not plan_id.strip():
            raise ValueError("cancel plan_id cannot be empty.")

        plan = _find_plan(self._state.plans, plan_id)
        if plan is None:
            return PlanningRuntimeResult(
                status=PlanningRuntimeStatus.BLOCKED,
                operation=PlanningOperation.CANCEL_PLAN,
                state=self._state,
                plan=None,
                plans=(),
                reason=f"plan not found: {plan_id}",
                created_at=utc_now(),
            )

        cancelled_steps = tuple(
            PlanStep(
                step_id=step.step_id,
                title=step.title,
                description=step.description,
                status=(
                    PlanStepStatus.CANCELLED
                    if step.status
                    not in {
                        PlanStepStatus.COMPLETED,
                        PlanStepStatus.FAILED,
                    }
                    else step.status
                ),
                risk=step.risk,
                requires_approval=step.requires_approval,
                created_at=step.created_at,
                metadata=step.metadata,
            )
            for step in plan.steps
        )
        cancelled_plan = Plan(
            plan_id=plan.plan_id,
            goal_id=plan.goal_id,
            title=plan.title,
            steps=cancelled_steps,
            created_at=plan.created_at,
            metadata={**plan.metadata, "cancelled": True},
        )
        plans = _replace_plan(self._state.plans, cancelled_plan)
        self._state = PlanningState(
            plans=plans,
            active_plan_id=_next_active_plan_id(
                plans=plans,
                excluded_plan_id=plan_id,
            ),
            created_at=utc_now(),
        )

        return PlanningRuntimeResult(
            status=PlanningRuntimeStatus.READY,
            operation=PlanningOperation.CANCEL_PLAN,
            state=self._state,
            plan=cancelled_plan,
            plans=(cancelled_plan,),
            reason="plan cancelled",
            created_at=utc_now(),
        )

    def recall(
        self,
        request: PlanRecallRequest,
    ) -> PlanningRuntimeResult:
        self._operation_count += 1
        plans = _filter_plans(
            plans=self._state.plans,
            query=request.query,
            goal_id=request.goal_id,
            active_plan_id=self._state.active_plan_id,
            active_only=request.active_only,
            include_cancelled=request.include_cancelled,
            limit=request.limit,
        )

        return PlanningRuntimeResult(
            status=PlanningRuntimeStatus.READY,
            operation=PlanningOperation.RECALL,
            state=self._state,
            plan=plans[0] if plans else None,
            plans=plans,
            reason="plan recall completed",
            created_at=utc_now(),
            metadata={
                **request.metadata,
                "query": request.query,
                "match_count": len(plans),
            },
        )

    def clear(self) -> PlanningRuntimeResult:
        self._operation_count += 1
        self._state = PlanningState()

        return PlanningRuntimeResult(
            status=PlanningRuntimeStatus.READY,
            operation=PlanningOperation.CLEAR,
            state=self._state,
            plan=None,
            plans=(),
            reason="planning state cleared",
            created_at=utc_now(),
        )

    def snapshot(self) -> PlanningRuntimeSnapshot:
        steps = tuple(step for plan in self._state.plans for step in plan.steps)
        return PlanningRuntimeSnapshot(
            status=PlanningRuntimeStatus.READY,
            state=self._state,
            plan_count=len(self._state.plans),
            active_plan_id=self._state.active_plan_id,
            pending_step_count=len(
                tuple(step for step in steps if step.status == PlanStepStatus.PENDING)
            ),
            ready_step_count=len(
                tuple(step for step in steps if step.status == PlanStepStatus.READY)
            ),
            blocked_step_count=len(
                tuple(step for step in steps if step.status == PlanStepStatus.BLOCKED)
            ),
            completed_step_count=len(
                tuple(
                    step
                    for step in steps
                    if step.status == PlanStepStatus.COMPLETED
                )
            ),
            approval_required_count=len(
                tuple(step for step in steps if step.requires_approval)
            ),
            operation_count=self._operation_count,
            created_at=utc_now(),
        )


def _build_plan(request: PlanCreateRequest) -> Plan:
    now = utc_now()
    raw_steps = (
        request.requested_steps
        if request.requested_steps
        else _default_steps_for_intent(
            goal=request.goal,
            intent_kind=request.intent_kind,
        )
    )
    steps = tuple(
        _plan_step_from_text(
            text=text,
            intent_kind=request.intent_kind,
            goal=request.goal,
            index=index,
            created_at=now,
        )
        for index, text in enumerate(raw_steps[: request.max_steps], start=1)
    )
    verification = _verification_step(
        intent_kind=request.intent_kind,
        created_at=now,
    )
    if len(steps) < request.max_steps:
        steps = (*steps, verification)

    return Plan(
        plan_id=f"plan_{uuid4().hex}",
        goal_id=request.goal.goal_id,
        title=request.title.strip()
        if request.title is not None
        else f"Plan for {request.goal.title}",
        steps=steps,
        created_at=now,
        metadata={
            **request.metadata,
            "intent_kind": request.intent_kind.value,
            "goal_priority": request.goal.priority.value,
        },
    )


def _default_steps_for_intent(
    *,
    goal: Goal,
    intent_kind: PlanIntentKind,
) -> tuple[str, ...]:
    if intent_kind == PlanIntentKind.DEVELOPER:
        return (
            "Inspect current code context.",
            "Identify relevant files and symbols.",
            "Run safe validation checks.",
            "Analyze any failure output.",
            "Suggest reviewable fixes without applying them.",
        )

    if intent_kind == PlanIntentKind.RESEARCH:
        return (
            "Clarify the research question.",
            "Collect relevant sources.",
            "Rank and summarize source evidence.",
            "Identify contradictions or uncertainty.",
            "Prepare concise findings.",
        )

    if intent_kind == PlanIntentKind.SAFETY:
        return (
            "Identify the safety signal.",
            "Classify severity and urgency.",
            "Stop or defer risky work if required.",
            "Ask for approval before any risky action.",
        )

    if intent_kind == PlanIntentKind.SYSTEM_RECOVERY:
        return (
            "Identify failed subsystem.",
            "Collect diagnostic state.",
            "Choose safe recovery path.",
            "Verify subsystem health after recovery.",
        )

    return (
        f"Clarify goal: {goal.title}.",
        "Gather relevant context.",
        "Choose safest next step.",
        "Prepare verification before completion.",
    )


def _plan_step_from_text(
    *,
    text: str,
    intent_kind: PlanIntentKind,
    goal: Goal,
    index: int,
    created_at: datetime,
) -> PlanStep:
    risk = _risk_for_step(text=text, intent_kind=intent_kind, goal=goal)
    return PlanStep(
        step_id=f"step_{uuid4().hex}",
        title=f"Step {index}",
        description=text.strip(),
        status=PlanStepStatus.READY if index == 1 else PlanStepStatus.PENDING,
        risk=risk,
        requires_approval=_requires_approval(risk),
        created_at=created_at,
        metadata={"intent_kind": intent_kind.value, "index": index},
    )


def _verification_step(
    *,
    intent_kind: PlanIntentKind,
    created_at: datetime,
) -> PlanStep:
    return PlanStep(
        step_id=f"step_{uuid4().hex}",
        title="Verify result",
        description="Verify the result before marking the plan complete.",
        status=PlanStepStatus.PENDING,
        risk=PlanRisk.LOW,
        requires_approval=False,
        created_at=created_at,
        metadata={"intent_kind": intent_kind.value, "verification": True},
    )


def _risk_for_step(
    *,
    text: str,
    intent_kind: PlanIntentKind,
    goal: Goal,
) -> PlanRisk:
    lowered = text.lower()

    if intent_kind in {
        PlanIntentKind.SAFETY,
        PlanIntentKind.SYSTEM_RECOVERY,
        PlanIntentKind.COMMUNICATION,
        PlanIntentKind.SIMULATION,
    }:
        return PlanRisk.MEDIUM

    if goal.priority == GoalPriority.CRITICAL:
        return PlanRisk.MEDIUM

    risky_words = {
        "delete",
        "modify",
        "write",
        "send",
        "execute",
        "control",
        "install",
        "remove",
        "shutdown",
        "recover",
    }
    if any(word in lowered for word in risky_words):
        return PlanRisk.MEDIUM

    return PlanRisk.LOW


def _requires_approval(risk: PlanRisk) -> bool:
    return risk in {PlanRisk.MEDIUM, PlanRisk.HIGH}


def _operation_for_step_status(status: PlanStepStatus) -> PlanningOperation:
    if status == PlanStepStatus.BLOCKED:
        return PlanningOperation.BLOCK_STEP
    if status == PlanStepStatus.COMPLETED:
        return PlanningOperation.COMPLETE_STEP
    return PlanningOperation.UPDATE_STEP


def _find_plan(plans: tuple[Plan, ...], plan_id: str) -> Plan | None:
    for plan in plans:
        if plan.plan_id == plan_id:
            return plan
    return None


def _find_step(steps: tuple[PlanStep, ...], step_id: str) -> PlanStep | None:
    for step in steps:
        if step.step_id == step_id:
            return step
    return None


def _replace_step(
    steps: tuple[PlanStep, ...],
    updated: PlanStep,
) -> tuple[PlanStep, ...]:
    return tuple(updated if step.step_id == updated.step_id else step for step in steps)


def _replace_plan(
    plans: tuple[Plan, ...],
    updated: Plan,
) -> tuple[Plan, ...]:
    return tuple(updated if plan.plan_id == updated.plan_id else plan for plan in plans)


def _next_active_plan_id(
    *,
    plans: tuple[Plan, ...],
    excluded_plan_id: str,
) -> str | None:
    for plan in plans:
        if plan.plan_id == excluded_plan_id:
            continue
        if not _plan_cancelled(plan):
            return plan.plan_id
    return None


def _filter_plans(
    *,
    plans: tuple[Plan, ...],
    query: str,
    goal_id: str | None,
    active_plan_id: str | None,
    active_only: bool,
    include_cancelled: bool,
    limit: int,
) -> tuple[Plan, ...]:
    query_text = query.strip().lower()
    filtered: list[Plan] = []

    for plan in plans:
        if goal_id is not None and plan.goal_id != goal_id:
            continue

        if active_only and plan.plan_id != active_plan_id:
            continue

        if not include_cancelled and _plan_cancelled(plan):
            continue

        if query_text and query_text not in _search_text(plan):
            continue

        filtered.append(plan)

    filtered.sort(
        key=lambda plan: (
            0 if plan.plan_id == active_plan_id else 1,
            plan.created_at,
            plan.title,
        )
    )
    return tuple(filtered[:limit])


def _search_text(plan: Plan) -> str:
    return " ".join(
        (
            plan.title,
            plan.goal_id,
            " ".join(step.title for step in plan.steps),
            " ".join(step.description for step in plan.steps),
        )
    ).lower()


def _plan_cancelled(plan: Plan) -> bool:
    return bool(plan.metadata.get("cancelled"))