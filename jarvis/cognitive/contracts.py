from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(UTC)


class AttentionPriority(StrEnum):
    BACKGROUND = "background"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class AttentionItemKind(StrEnum):
    VOICE = "voice"
    SCREEN = "screen"
    NOTIFICATION = "notification"
    SYSTEM_HEALTH = "system_health"
    PROJECT = "project"
    MEMORY_RECALL = "memory_recall"
    ACTIVE_TASK = "active_task"
    SAFETY = "safety"
    RESEARCH = "research"
    USER_COMMAND = "user_command"


class AttentionDecision(StrEnum):
    IGNORE = "ignore"
    TRACK = "track"
    FOCUS = "focus"
    INTERRUPT_NOW = "interrupt_now"


class WorkingMemoryKind(StrEnum):
    CONVERSATION = "conversation"
    PROJECT = "project"
    OBJECTIVE = "objective"
    SCREEN_CONTEXT = "screen_context"
    RECENT_ACTION = "recent_action"
    TASK = "task"
    RISK = "risk"
    ASSUMPTION = "assumption"
    USER_PREFERENCE = "user_preference"


class GoalStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class GoalPriority(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class PlanStepStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PlanRisk(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class BehaviorTone(StrEnum):
    CALM = "calm"
    CONCISE = "concise"
    WARNING = "warning"
    CLARIFYING = "clarifying"
    HUMOROUS = "humorous"
    PROTECTIVE = "protective"


class Phase9GateStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


class Phase9CheckKind(StrEnum):
    ATTENTION_STATE = "attention_state"
    WORKING_MEMORY_STATE = "working_memory_state"
    GOAL_STATE = "goal_state"
    PLANNING_STATE = "planning_state"
    PERSONALITY_PROFILE = "personality_profile"
    BEHAVIOR_POLICY = "behavior_policy"
    COGNITIVE_SESSION = "cognitive_session"


@dataclass(frozen=True, slots=True)
class AttentionItem:
    item_id: str
    kind: AttentionItemKind
    title: str
    summary: str
    priority: AttentionPriority
    source: str
    decision: AttentionDecision
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.item_id.strip():
            raise ValueError("attention item_id cannot be empty.")
        if not self.title.strip():
            raise ValueError("attention title cannot be empty.")
        if not self.summary.strip():
            raise ValueError("attention summary cannot be empty.")
        if not self.source.strip():
            raise ValueError("attention source cannot be empty.")


@dataclass(frozen=True, slots=True)
class AttentionState:
    items: tuple[AttentionItem, ...] = ()
    focused_item_id: str | None = None
    created_at: datetime = field(default_factory=utc_now)

    @property
    def critical_items(self) -> tuple[AttentionItem, ...]:
        return tuple(
            item
            for item in self.items
            if item.priority == AttentionPriority.CRITICAL
        )

    @property
    def interrupt_items(self) -> tuple[AttentionItem, ...]:
        return tuple(
            item
            for item in self.items
            if item.decision == AttentionDecision.INTERRUPT_NOW
        )

    @property
    def has_focus(self) -> bool:
        return self.focused_item_id is not None


@dataclass(frozen=True, slots=True)
class WorkingMemoryItem:
    item_id: str
    kind: WorkingMemoryKind
    key: str
    value: str
    importance: AttentionPriority
    source: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.item_id.strip():
            raise ValueError("working memory item_id cannot be empty.")
        if not self.key.strip():
            raise ValueError("working memory key cannot be empty.")
        if not self.value.strip():
            raise ValueError("working memory value cannot be empty.")
        if not self.source.strip():
            raise ValueError("working memory source cannot be empty.")


@dataclass(frozen=True, slots=True)
class WorkingMemoryState:
    items: tuple[WorkingMemoryItem, ...] = ()
    created_at: datetime = field(default_factory=utc_now)

    def get(self, key: str) -> WorkingMemoryItem | None:
        normalized = key.strip().lower()
        for item in self.items:
            if item.key.strip().lower() == normalized:
                return item
        return None

    @property
    def high_importance_items(self) -> tuple[WorkingMemoryItem, ...]:
        return tuple(
            item
            for item in self.items
            if item.importance in {
                AttentionPriority.HIGH,
                AttentionPriority.CRITICAL,
            }
        )


@dataclass(frozen=True, slots=True)
class Goal:
    goal_id: str
    title: str
    description: str
    status: GoalStatus
    priority: GoalPriority
    created_at: datetime
    updated_at: datetime
    parent_goal_id: str | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.goal_id.strip():
            raise ValueError("goal_id cannot be empty.")
        if not self.title.strip():
            raise ValueError("goal title cannot be empty.")
        if not self.description.strip():
            raise ValueError("goal description cannot be empty.")


@dataclass(frozen=True, slots=True)
class GoalState:
    goals: tuple[Goal, ...] = ()
    active_goal_id: str | None = None
    created_at: datetime = field(default_factory=utc_now)

    @property
    def active_goals(self) -> tuple[Goal, ...]:
        return tuple(goal for goal in self.goals if goal.status == GoalStatus.ACTIVE)

    @property
    def blocked_goals(self) -> tuple[Goal, ...]:
        return tuple(goal for goal in self.goals if goal.status == GoalStatus.BLOCKED)

    @property
    def has_active_goal(self) -> bool:
        return bool(self.active_goals)


@dataclass(frozen=True, slots=True)
class PlanStep:
    step_id: str
    title: str
    description: str
    status: PlanStepStatus
    risk: PlanRisk
    requires_approval: bool
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.step_id.strip():
            raise ValueError("plan step_id cannot be empty.")
        if not self.title.strip():
            raise ValueError("plan step title cannot be empty.")
        if not self.description.strip():
            raise ValueError("plan step description cannot be empty.")


@dataclass(frozen=True, slots=True)
class Plan:
    plan_id: str
    goal_id: str
    title: str
    steps: tuple[PlanStep, ...]
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.plan_id.strip():
            raise ValueError("plan_id cannot be empty.")
        if not self.goal_id.strip():
            raise ValueError("plan goal_id cannot be empty.")
        if not self.title.strip():
            raise ValueError("plan title cannot be empty.")

    @property
    def requires_approval(self) -> bool:
        return any(step.requires_approval for step in self.steps)


@dataclass(frozen=True, slots=True)
class PlanningState:
    plans: tuple[Plan, ...] = ()
    active_plan_id: str | None = None
    created_at: datetime = field(default_factory=utc_now)

    @property
    def active_plan(self) -> Plan | None:
        if self.active_plan_id is None:
            return None

        for plan in self.plans:
            if plan.plan_id == self.active_plan_id:
                return plan

        return None


@dataclass(frozen=True, slots=True)
class PersonalityProfile:
    name: str
    traits: tuple[str, ...]
    default_tone: BehaviorTone
    confirmation_phrase: str
    warning_phrase: str
    clarification_phrase: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("personality name cannot be empty.")
        if not self.traits:
            raise ValueError("personality traits cannot be empty.")
        # Empty confirmation_phrase is allowed.
        # Production speech must be generated by cognition/Ollama, not fixed text.


@dataclass(frozen=True, slots=True)
class BehaviorPolicy:
    max_reply_sentences: int
    interrupt_only_when_important: bool
    ask_when_instruction_incomplete: bool
    allow_dry_humor: bool
    truth_over_comfort: bool
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_reply_sentences < 1:
            raise ValueError("max_reply_sentences must be at least 1.")


@dataclass(frozen=True, slots=True)
class CognitiveSessionState:
    session_id: str
    user_label: str
    attention: AttentionState
    working_memory: WorkingMemoryState
    goals: GoalState
    planning: PlanningState
    personality: PersonalityProfile
    behavior_policy: BehaviorPolicy
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("session_id cannot be empty.")
        if not self.user_label.strip():
            raise ValueError("user_label cannot be empty.")


@dataclass(frozen=True, slots=True)
class Phase9DesignCheck:
    kind: Phase9CheckKind
    passed: bool
    message: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Phase9DesignGateReport:
    status: Phase9GateStatus
    checks: tuple[Phase9DesignCheck, ...]
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == Phase9GateStatus.PASSED

    @property
    def failed_count(self) -> int:
        return sum(1 for check in self.checks if not check.passed)


class Phase9DesignGate:
    """
    Step 49 Phase 9 Design Gate.

    This validates the cognitive contracts before Attention Runtime,
    Working Memory Runtime, Goal Runtime, Planning Runtime, Personality
    Runtime, and Cognitive Session Runtime are implemented.

    It performs no actions, no memory writes, no tool execution, and no
    autonomous behavior.
    """

    def validate(
        self,
        session: CognitiveSessionState,
    ) -> Phase9DesignGateReport:
        checks = (
            _check_attention(session.attention),
            _check_working_memory(session.working_memory),
            _check_goals(session.goals),
            _check_planning(session.planning),
            _check_personality(session.personality),
            _check_behavior_policy(session.behavior_policy),
            _check_session(session),
        )

        return Phase9DesignGateReport(
            status=(
                Phase9GateStatus.PASSED
                if all(check.passed for check in checks)
                else Phase9GateStatus.FAILED
            ),
            checks=checks,
            created_at=utc_now(),
            metadata={
                "session_id": session.session_id,
                "user_label": session.user_label,
            },
        )


def default_cognitive_session(
    *,
    user_label: str = "Balu",
) -> CognitiveSessionState:
    now = utc_now()
    session_id = f"cog_{uuid4().hex}"

    attention_item = AttentionItem(
        item_id=f"att_{uuid4().hex}",
        kind=AttentionItemKind.USER_COMMAND,
        title="Current user focus",
        summary="User is building JARVIS Phase 9 cognitive runtime.",
        priority=AttentionPriority.HIGH,
        source="phase9_design_gate",
        decision=AttentionDecision.FOCUS,
        created_at=now,
    )
    working_memory_item = WorkingMemoryItem(
        item_id=f"wm_{uuid4().hex}",
        kind=WorkingMemoryKind.OBJECTIVE,
        key="current_objective",
        value="Build Phase 9 as persistent cognitive context.",
        importance=AttentionPriority.HIGH,
        source="phase9_design_gate",
        created_at=now,
    )
    goal = Goal(
        goal_id=f"goal_{uuid4().hex}",
        title="Build Phase 9 cognitive presence",
        description=(
            "Implement attention, working memory, goals, planning, "
            "personality, and cognitive session runtime."
        ),
        status=GoalStatus.ACTIVE,
        priority=GoalPriority.HIGH,
        created_at=now,
        updated_at=now,
        tags=("phase9", "cognitive-runtime"),
    )
    step = PlanStep(
        step_id=f"step_{uuid4().hex}",
        title="Validate Phase 9 contracts",
        description="Confirm all Phase 9 cognitive contracts are present.",
        status=PlanStepStatus.READY,
        risk=PlanRisk.LOW,
        requires_approval=False,
        created_at=now,
    )
    plan = Plan(
        plan_id=f"plan_{uuid4().hex}",
        goal_id=goal.goal_id,
        title="Phase 9 implementation plan",
        steps=(step,),
        created_at=now,
    )
    personality = PersonalityProfile(
        name="JARVIS",
        traits=(
            "calm",
            "respectful",
            "concise",
            "protective",
            "truthful",
            "slightly_witty",
        ),
        default_tone=BehaviorTone.CALM,
        confirmation_phrase="",
        warning_phrase="I would advise caution.",
        clarification_phrase="I need one detail before proceeding.",
        created_at=now,
    )
    behavior = BehaviorPolicy(
        max_reply_sentences=3,
        interrupt_only_when_important=True,
        ask_when_instruction_incomplete=True,
        allow_dry_humor=True,
        truth_over_comfort=True,
        created_at=now,
    )

    return CognitiveSessionState(
        session_id=session_id,
        user_label=user_label,
        attention=AttentionState(
            items=(attention_item,),
            focused_item_id=attention_item.item_id,
            created_at=now,
        ),
        working_memory=WorkingMemoryState(
            items=(working_memory_item,),
            created_at=now,
        ),
        goals=GoalState(
            goals=(goal,),
            active_goal_id=goal.goal_id,
            created_at=now,
        ),
        planning=PlanningState(
            plans=(plan,),
            active_plan_id=plan.plan_id,
            created_at=now,
        ),
        personality=personality,
        behavior_policy=behavior,
        created_at=now,
        updated_at=now,
        metadata={"phase": "phase9_design_gate"},
    )


def _check_attention(state: AttentionState) -> Phase9DesignCheck:
    return Phase9DesignCheck(
        kind=Phase9CheckKind.ATTENTION_STATE,
        passed=bool(state.items),
        message="attention state contains at least one attention item",
        created_at=utc_now(),
        metadata={
            "item_count": len(state.items),
            "has_focus": state.has_focus,
            "critical_count": len(state.critical_items),
        },
    )


def _check_working_memory(state: WorkingMemoryState) -> Phase9DesignCheck:
    return Phase9DesignCheck(
        kind=Phase9CheckKind.WORKING_MEMORY_STATE,
        passed=bool(state.items),
        message="working memory state contains active context",
        created_at=utc_now(),
        metadata={
            "item_count": len(state.items),
            "high_importance_count": len(state.high_importance_items),
        },
    )


def _check_goals(state: GoalState) -> Phase9DesignCheck:
    return Phase9DesignCheck(
        kind=Phase9CheckKind.GOAL_STATE,
        passed=state.has_active_goal,
        message="goal state contains active goals",
        created_at=utc_now(),
        metadata={
            "goal_count": len(state.goals),
            "active_count": len(state.active_goals),
            "blocked_count": len(state.blocked_goals),
        },
    )


def _check_planning(state: PlanningState) -> Phase9DesignCheck:
    active_plan = state.active_plan
    return Phase9DesignCheck(
        kind=Phase9CheckKind.PLANNING_STATE,
        passed=active_plan is not None and bool(active_plan.steps),
        message="planning state contains an active plan with steps",
        created_at=utc_now(),
        metadata={
            "plan_count": len(state.plans),
            "active_plan": active_plan.plan_id if active_plan else "",
            "requires_approval": (
                active_plan.requires_approval if active_plan else False
            ),
        },
    )


def _check_personality(profile: PersonalityProfile) -> Phase9DesignCheck:
    return Phase9DesignCheck(
        kind=Phase9CheckKind.PERSONALITY_PROFILE,
        passed=bool(profile.traits),
        message="personality profile is configured",
        created_at=utc_now(),
        metadata={
            "name": profile.name,
            "trait_count": len(profile.traits),
            "default_tone": profile.default_tone.value,
        },
    )


def _check_behavior_policy(policy: BehaviorPolicy) -> Phase9DesignCheck:
    return Phase9DesignCheck(
        kind=Phase9CheckKind.BEHAVIOR_POLICY,
        passed=policy.max_reply_sentences <= 4,
        message="behavior policy enforces concise assistant behavior",
        created_at=utc_now(),
        metadata={
            "max_reply_sentences": policy.max_reply_sentences,
            "interrupt_only_when_important": (
                policy.interrupt_only_when_important
            ),
            "truth_over_comfort": policy.truth_over_comfort,
        },
    )


def _check_session(session: CognitiveSessionState) -> Phase9DesignCheck:
    return Phase9DesignCheck(
        kind=Phase9CheckKind.COGNITIVE_SESSION,
        passed=bool(session.session_id and session.user_label),
        message="cognitive session state is valid",
        created_at=utc_now(),
        metadata={
            "session_id": session.session_id,
            "user_label": session.user_label,
        },
    )