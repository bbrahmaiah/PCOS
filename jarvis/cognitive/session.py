from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from uuid import uuid4

from jarvis.cognitive.attention import (
    AttentionEvaluationRequest,
    AttentionEvaluationResult,
    AttentionRuntime,
    AttentionSignal,
)
from jarvis.cognitive.contracts import (
    AttentionState,
    BehaviorPolicy,
    CognitiveSessionState,
    GoalState,
    PersonalityProfile,
    PlanningState,
    WorkingMemoryState,
    utc_now,
)
from jarvis.cognitive.goals import (
    GoalCreateRequest,
    GoalRuntime,
    GoalRuntimeResult,
)
from jarvis.cognitive.personality import (
    BehaviorIntent,
    BehaviorRequest,
    BehaviorRisk,
    BehaviorRuntimeResult,
    PersonalityRuntime,
)
from jarvis.cognitive.planning import (
    PlanCreateRequest,
    PlanIntentKind,
    PlanningRuntime,
    PlanningRuntimeResult,
)
from jarvis.cognitive.working_memory import (
    WorkingMemoryEntry,
    WorkingMemoryRuntime,
    WorkingMemoryRuntimeResult,
    WorkingMemoryUpdateRequest,
)


class CognitiveSessionRuntimeStatus(StrEnum):
    READY = "ready"
    DEGRADED = "degraded"
    BLOCKED = "blocked"


class CognitiveSessionOperation(StrEnum):
    START = "start"
    UPDATE_ATTENTION = "update_attention"
    UPDATE_WORKING_MEMORY = "update_working_memory"
    CREATE_GOAL = "create_goal"
    CREATE_PLAN = "create_plan"
    RESPOND = "respond"
    SYNCHRONIZE = "synchronize"
    CLEAR = "clear"


@dataclass(frozen=True, slots=True)
class CognitiveSessionStartRequest:
    user_label: str = "Balu"
    session_id: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.user_label.strip():
            raise ValueError("cognitive session user_label cannot be empty.")
        if self.session_id is not None and not self.session_id.strip():
            raise ValueError("cognitive session session_id cannot be empty.")


@dataclass(frozen=True, slots=True)
class CognitiveSessionUpdateRequest:
    attention_signals: tuple[AttentionSignal, ...] = ()
    working_memory_entries: tuple[WorkingMemoryEntry, ...] = ()
    user_is_speaking: bool = False
    assistant_is_speaking: bool = False
    allow_interruptions: bool = True
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CognitiveSessionGoalRequest:
    title: str
    description: str
    priority: object
    tags: tuple[str, ...] = ()
    create_plan: bool = True
    intent_kind: PlanIntentKind = PlanIntentKind.GENERAL
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("cognitive session goal title cannot be empty.")
        if not self.description.strip():
            raise ValueError(
                "cognitive session goal description cannot be empty."
            )


@dataclass(frozen=True, slots=True)
class CognitiveSessionResponseRequest:
    intent: BehaviorIntent
    message: str = ""
    risk: BehaviorRisk = BehaviorRisk.NONE
    instruction_complete: bool = True
    user_is_busy: bool = False
    allow_humor: bool = False
    requires_truth_challenge: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CognitiveSessionRuntimeResult:
    status: CognitiveSessionRuntimeStatus
    operation: CognitiveSessionOperation
    session: CognitiveSessionState
    attention_result: AttentionEvaluationResult | None
    working_memory_result: WorkingMemoryRuntimeResult | None
    goal_result: GoalRuntimeResult | None
    planning_result: PlanningRuntimeResult | None
    behavior_result: BehaviorRuntimeResult | None
    reason: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == CognitiveSessionRuntimeStatus.READY


@dataclass(frozen=True, slots=True)
class CognitiveSessionRuntimeSnapshot:
    status: CognitiveSessionRuntimeStatus
    session: CognitiveSessionState
    session_id: str
    user_label: str
    update_count: int
    attention_items: int
    working_memory_items: int
    goal_count: int
    plan_count: int
    behavior_decisions: int
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


class CognitiveSessionRuntime:
    """
    Phase 9 / Step 49F Cognitive Session Runtime.

    This runtime coordinates the Phase 9 cognitive organs:
    - AttentionRuntime
    - WorkingMemoryRuntime
    - GoalRuntime
    - PlanningRuntime
    - PersonalityRuntime

    It creates one active CognitiveSessionState.

    It intentionally does not execute tools, mutate long-term memory,
    control the laptop, or bypass safety. It only coordinates cognitive state.
    """

    def __init__(
        self,
        *,
        attention: AttentionRuntime | None = None,
        working_memory: WorkingMemoryRuntime | None = None,
        goals: GoalRuntime | None = None,
        planning: PlanningRuntime | None = None,
        personality: PersonalityRuntime | None = None,
    ) -> None:
        self._attention = attention or AttentionRuntime()
        self._working_memory = working_memory or WorkingMemoryRuntime()
        self._goals = goals or GoalRuntime()
        self._planning = planning or PlanningRuntime()
        self._personality = personality or PersonalityRuntime()
        self._session = _build_session(
            session_id=f"cog_{uuid4().hex}",
            user_label="Balu",
            attention_state=self._attention.state,
            working_memory_state=self._working_memory.state,
            goal_state=self._goals.state,
            planning_state=self._planning.state,
            personality_profile=self._personality.profile,
            behavior_policy=self._personality.policy,
            metadata={},
        )
        self._update_count = 0

    @property
    def session(self) -> CognitiveSessionState:
        return self._session

    def start(
        self,
        request: CognitiveSessionStartRequest,
    ) -> CognitiveSessionRuntimeResult:
        self._update_count += 1
        self._session = _build_session(
            session_id=request.session_id or f"cog_{uuid4().hex}",
            user_label=request.user_label.strip(),
            attention_state=self._attention.state,
            working_memory_state=self._working_memory.state,
            goal_state=self._goals.state,
            planning_state=self._planning.state,
            personality_profile=self._personality.profile,
            behavior_policy=self._personality.policy,
            metadata=request.metadata,
        )

        return self._result(
            operation=CognitiveSessionOperation.START,
            reason="cognitive session started",
            metadata=request.metadata,
        )

    def update(
        self,
        request: CognitiveSessionUpdateRequest,
    ) -> CognitiveSessionRuntimeResult:
        self._update_count += 1

        attention_result = self._attention.evaluate(
            AttentionEvaluationRequest(
                signals=request.attention_signals,
                current_state=self._attention.state,
                user_is_speaking=request.user_is_speaking,
                assistant_is_speaking=request.assistant_is_speaking,
                allow_interruptions=request.allow_interruptions,
                metadata=request.metadata,
            )
        )
        working_memory_result = self._working_memory.update(
            WorkingMemoryUpdateRequest(
                entries=request.working_memory_entries,
                metadata=request.metadata,
            )
        )
        self._sync_session(metadata=request.metadata)

        return self._result(
            operation=CognitiveSessionOperation.SYNCHRONIZE,
            attention_result=attention_result,
            working_memory_result=working_memory_result,
            reason="cognitive session updated",
            metadata={
                **request.metadata,
                "attention_decision": attention_result.decision.value,
                "working_memory_items": len(
                    working_memory_result.state.items
                ),
            },
        )

    def create_goal(
        self,
        request: CognitiveSessionGoalRequest,
    ) -> CognitiveSessionRuntimeResult:
        self._update_count += 1

        from jarvis.cognitive.contracts import GoalPriority

        if not isinstance(request.priority, GoalPriority):
            return self._result(
                operation=CognitiveSessionOperation.CREATE_GOAL,
                status=CognitiveSessionRuntimeStatus.BLOCKED,
                reason="goal priority must be GoalPriority",
                metadata=request.metadata,
            )

        goal_result = self._goals.create(
            GoalCreateRequest(
                title=request.title,
                description=request.description,
                priority=request.priority,
                tags=request.tags,
                metadata=request.metadata,
            )
        )
        planning_result: PlanningRuntimeResult | None = None

        if request.create_plan and goal_result.goal is not None:
            planning_result = self._planning.create_plan(
                PlanCreateRequest(
                    goal=goal_result.goal,
                    intent_kind=request.intent_kind,
                    metadata=request.metadata,
                )
            )

        self._sync_session(metadata=request.metadata)

        return self._result(
            operation=CognitiveSessionOperation.CREATE_GOAL,
            goal_result=goal_result,
            planning_result=planning_result,
            reason="goal created in cognitive session",
            metadata={
                **request.metadata,
                "created_plan": planning_result is not None,
            },
        )

    def respond(
        self,
        request: CognitiveSessionResponseRequest,
    ) -> CognitiveSessionRuntimeResult:
        self._update_count += 1

        behavior_result = self._personality.respond(
            BehaviorRequest(
                intent=request.intent,
                message=request.message,
                risk=request.risk,
                instruction_complete=request.instruction_complete,
                user_is_busy=request.user_is_busy,
                allow_humor=request.allow_humor,
                requires_truth_challenge=request.requires_truth_challenge,
                metadata=request.metadata,
            )
        )
        self._sync_session(metadata=request.metadata)

        return self._result(
            operation=CognitiveSessionOperation.RESPOND,
            behavior_result=behavior_result,
            reason="behavior response generated from cognitive session",
            metadata=request.metadata,
        )

    def clear(self) -> CognitiveSessionRuntimeResult:
        self._update_count += 1
        self._attention.clear()
        self._working_memory.update(WorkingMemoryUpdateRequest(clear=True))
        self._goals.clear()
        self._planning.clear()
        self._sync_session(metadata={})

        return self._result(
            operation=CognitiveSessionOperation.CLEAR,
            reason="cognitive session cleared",
        )

    def snapshot(self) -> CognitiveSessionRuntimeSnapshot:
        behavior_snapshot = self._personality.snapshot()
        return CognitiveSessionRuntimeSnapshot(
            status=CognitiveSessionRuntimeStatus.READY,
            session=self._session,
            session_id=self._session.session_id,
            user_label=self._session.user_label,
            update_count=self._update_count,
            attention_items=len(self._session.attention.items),
            working_memory_items=len(self._session.working_memory.items),
            goal_count=len(self._session.goals.goals),
            plan_count=len(self._session.planning.plans),
            behavior_decisions=behavior_snapshot.decision_count,
            created_at=utc_now(),
        )

    def _sync_session(
        self,
        *,
        metadata: dict[str, object],
    ) -> None:
        self._session = _build_session(
            session_id=self._session.session_id,
            user_label=self._session.user_label,
            attention_state=self._attention.state,
            working_memory_state=self._working_memory.state,
            goal_state=self._goals.state,
            planning_state=self._planning.state,
            personality_profile=self._personality.profile,
            behavior_policy=self._personality.policy,
            metadata={**self._session.metadata, **metadata},
        )

    def _result(
        self,
        *,
        operation: CognitiveSessionOperation,
        reason: str,
        status: CognitiveSessionRuntimeStatus = CognitiveSessionRuntimeStatus.READY,
        attention_result: AttentionEvaluationResult | None = None,
        working_memory_result: WorkingMemoryRuntimeResult | None = None,
        goal_result: GoalRuntimeResult | None = None,
        planning_result: PlanningRuntimeResult | None = None,
        behavior_result: BehaviorRuntimeResult | None = None,
        metadata: dict[str, object] | None = None,
    ) -> CognitiveSessionRuntimeResult:
        return CognitiveSessionRuntimeResult(
            status=status,
            operation=operation,
            session=self._session,
            attention_result=attention_result,
            working_memory_result=working_memory_result,
            goal_result=goal_result,
            planning_result=planning_result,
            behavior_result=behavior_result,
            reason=reason,
            created_at=utc_now(),
            metadata=metadata or {},
        )


def _build_session(
    *,
    session_id: str,
    user_label: str,
    attention_state: AttentionState,
    working_memory_state: WorkingMemoryState,
    goal_state: GoalState,
    planning_state: PlanningState,
    personality_profile: PersonalityProfile,
    behavior_policy: BehaviorPolicy,
    metadata: dict[str, object],
) -> CognitiveSessionState:
    now = utc_now()
    return CognitiveSessionState(
        session_id=session_id,
        user_label=user_label,
        attention=attention_state,
        working_memory=working_memory_state,
        goals=goal_state,
        planning=planning_state,
        personality=personality_profile,
        behavior_policy=behavior_policy,
        created_at=now,
        updated_at=now,
        metadata=metadata,
    )