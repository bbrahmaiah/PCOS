from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from uuid import uuid4

from jarvis.cognitive.contracts import (
    Goal,
    GoalPriority,
    GoalState,
    GoalStatus,
    utc_now,
)


class GoalRuntimeStatus(StrEnum):
    READY = "ready"
    DEGRADED = "degraded"
    BLOCKED = "blocked"


class GoalOperation(StrEnum):
    CREATE = "create"
    UPDATE = "update"
    PAUSE = "pause"
    RESUME = "resume"
    BLOCK = "block"
    COMPLETE = "complete"
    CANCEL = "cancel"
    RECALL = "recall"
    CLEAR = "clear"


@dataclass(frozen=True, slots=True)
class GoalCreateRequest:
    title: str
    description: str
    priority: GoalPriority = GoalPriority.NORMAL
    parent_goal_id: str | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("goal create title cannot be empty.")
        if not self.description.strip():
            raise ValueError("goal create description cannot be empty.")


@dataclass(frozen=True, slots=True)
class GoalUpdateRequest:
    goal_id: str
    title: str | None = None
    description: str | None = None
    status: GoalStatus | None = None
    priority: GoalPriority | None = None
    tags: tuple[str, ...] | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.goal_id.strip():
            raise ValueError("goal update goal_id cannot be empty.")
        if self.title is not None and not self.title.strip():
            raise ValueError("goal update title cannot be empty.")
        if self.description is not None and not self.description.strip():
            raise ValueError("goal update description cannot be empty.")


@dataclass(frozen=True, slots=True)
class GoalRecallRequest:
    query: str = ""
    statuses: tuple[GoalStatus, ...] = ()
    priorities: tuple[GoalPriority, ...] = ()
    tags: tuple[str, ...] = ()
    include_children: bool = True
    limit: int = 10
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError("goal recall limit must be at least 1.")


@dataclass(frozen=True, slots=True)
class GoalRuntimeResult:
    status: GoalRuntimeStatus
    operation: GoalOperation
    state: GoalState
    goal: Goal | None
    goals: tuple[Goal, ...]
    reason: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == GoalRuntimeStatus.READY


@dataclass(frozen=True, slots=True)
class GoalRuntimeSnapshot:
    status: GoalRuntimeStatus
    state: GoalState
    goal_count: int
    active_count: int
    blocked_count: int
    completed_count: int
    operation_count: int
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


class GoalRuntime:
    """
    Phase 9 / Step 49C Goal State Runtime.

    Goal runtime tracks what JARVIS and Balu are trying to achieve:
    - current goal
    - subgoals
    - blocked goals
    - paused goals
    - completed goals
    - goal priority
    - next focus

    It is intentionally non-executing:
    - no tool calls
    - no long-term memory writes
    - no laptop control
    - no autonomous execution

    It only maintains goal state.
    """

    def __init__(self) -> None:
        self._state = GoalState()
        self._operation_count = 0

    @property
    def state(self) -> GoalState:
        return self._state

    def create(
        self,
        request: GoalCreateRequest,
    ) -> GoalRuntimeResult:
        self._operation_count += 1
        now = utc_now()
        goal = Goal(
            goal_id=f"goal_{uuid4().hex}",
            title=request.title.strip(),
            description=request.description.strip(),
            status=GoalStatus.ACTIVE,
            priority=request.priority,
            created_at=now,
            updated_at=now,
            parent_goal_id=request.parent_goal_id,
            tags=_clean_tags(request.tags),
            metadata=request.metadata,
        )
        goals = (goal, *self._state.goals)
        self._state = GoalState(
            goals=goals,
            active_goal_id=goal.goal_id,
            created_at=utc_now(),
        )

        return GoalRuntimeResult(
            status=GoalRuntimeStatus.READY,
            operation=GoalOperation.CREATE,
            state=self._state,
            goal=goal,
            goals=(goal,),
            reason="goal created and activated",
            created_at=utc_now(),
            metadata={"goal_count": len(goals)},
        )

    def update(
        self,
        request: GoalUpdateRequest,
    ) -> GoalRuntimeResult:
        self._operation_count += 1
        existing = _find_goal(self._state.goals, request.goal_id)

        if existing is None:
            return GoalRuntimeResult(
                status=GoalRuntimeStatus.BLOCKED,
                operation=GoalOperation.UPDATE,
                state=self._state,
                goal=None,
                goals=(),
                reason=f"goal not found: {request.goal_id}",
                created_at=utc_now(),
                metadata=request.metadata,
            )

        updated = Goal(
            goal_id=existing.goal_id,
            title=(
                request.title.strip()
                if request.title is not None
                else existing.title
            ),
            description=(
                request.description.strip()
                if request.description is not None
                else existing.description
            ),
            status=request.status or existing.status,
            priority=request.priority or existing.priority,
            created_at=existing.created_at,
            updated_at=utc_now(),
            parent_goal_id=existing.parent_goal_id,
            tags=(
                _clean_tags(request.tags)
                if request.tags is not None
                else existing.tags
            ),
            metadata={**existing.metadata, **request.metadata},
        )
        goals = _replace_goal(self._state.goals, updated)
        active_goal_id = _next_active_goal_id(
            goals=goals,
            preferred_goal_id=(
                updated.goal_id
                if updated.status == GoalStatus.ACTIVE
                else self._state.active_goal_id
            ),
        )
        self._state = GoalState(
            goals=goals,
            active_goal_id=active_goal_id,
            created_at=utc_now(),
        )

        return GoalRuntimeResult(
            status=GoalRuntimeStatus.READY,
            operation=_operation_for_status(request.status),
            state=self._state,
            goal=updated,
            goals=(updated,),
            reason="goal updated",
            created_at=utc_now(),
            metadata={"active_goal_id": active_goal_id or ""},
        )

    def pause(self, goal_id: str) -> GoalRuntimeResult:
        return self.update(
            GoalUpdateRequest(goal_id=goal_id, status=GoalStatus.PAUSED)
        )

    def resume(self, goal_id: str) -> GoalRuntimeResult:
        return self.update(
            GoalUpdateRequest(goal_id=goal_id, status=GoalStatus.ACTIVE)
        )

    def block(
        self,
        goal_id: str,
        *,
        reason: str,
    ) -> GoalRuntimeResult:
        if not reason.strip():
            raise ValueError("block reason cannot be empty.")

        return self.update(
            GoalUpdateRequest(
                goal_id=goal_id,
                status=GoalStatus.BLOCKED,
                metadata={"blocked_reason": reason.strip()},
            )
        )

    def complete(self, goal_id: str) -> GoalRuntimeResult:
        return self.update(
            GoalUpdateRequest(goal_id=goal_id, status=GoalStatus.COMPLETED)
        )

    def cancel(self, goal_id: str) -> GoalRuntimeResult:
        return self.update(
            GoalUpdateRequest(goal_id=goal_id, status=GoalStatus.CANCELLED)
        )

    def recall(
        self,
        request: GoalRecallRequest,
    ) -> GoalRuntimeResult:
        self._operation_count += 1
        goals = _filter_goals(
            goals=self._state.goals,
            query=request.query,
            statuses=request.statuses,
            priorities=request.priorities,
            tags=request.tags,
            include_children=request.include_children,
            limit=request.limit,
        )

        return GoalRuntimeResult(
            status=GoalRuntimeStatus.READY,
            operation=GoalOperation.RECALL,
            state=self._state,
            goal=goals[0] if goals else None,
            goals=goals,
            reason="goal recall completed",
            created_at=utc_now(),
            metadata={
                **request.metadata,
                "query": request.query,
                "match_count": len(goals),
            },
        )

    def clear(self) -> GoalRuntimeResult:
        self._operation_count += 1
        self._state = GoalState()

        return GoalRuntimeResult(
            status=GoalRuntimeStatus.READY,
            operation=GoalOperation.CLEAR,
            state=self._state,
            goal=None,
            goals=(),
            reason="goal state cleared",
            created_at=utc_now(),
        )

    def snapshot(self) -> GoalRuntimeSnapshot:
        return GoalRuntimeSnapshot(
            status=GoalRuntimeStatus.READY,
            state=self._state,
            goal_count=len(self._state.goals),
            active_count=len(self._state.active_goals),
            blocked_count=len(self._state.blocked_goals),
            completed_count=len(
                tuple(
                    goal
                    for goal in self._state.goals
                    if goal.status == GoalStatus.COMPLETED
                )
            ),
            operation_count=self._operation_count,
            created_at=utc_now(),
        )


def _operation_for_status(status: GoalStatus | None) -> GoalOperation:
    if status == GoalStatus.PAUSED:
        return GoalOperation.PAUSE
    if status == GoalStatus.ACTIVE:
        return GoalOperation.RESUME
    if status == GoalStatus.BLOCKED:
        return GoalOperation.BLOCK
    if status == GoalStatus.COMPLETED:
        return GoalOperation.COMPLETE
    if status == GoalStatus.CANCELLED:
        return GoalOperation.CANCEL

    return GoalOperation.UPDATE


def _find_goal(
    goals: tuple[Goal, ...],
    goal_id: str,
) -> Goal | None:
    for goal in goals:
        if goal.goal_id == goal_id:
            return goal

    return None


def _replace_goal(
    goals: tuple[Goal, ...],
    updated: Goal,
) -> tuple[Goal, ...]:
    return tuple(updated if goal.goal_id == updated.goal_id else goal for goal in goals)


def _next_active_goal_id(
    *,
    goals: tuple[Goal, ...],
    preferred_goal_id: str | None,
) -> str | None:
    if preferred_goal_id is not None:
        preferred = _find_goal(goals, preferred_goal_id)
        if preferred is not None and preferred.status == GoalStatus.ACTIVE:
            return preferred.goal_id

    active_goals = tuple(goal for goal in goals if goal.status == GoalStatus.ACTIVE)
    if not active_goals:
        return None

    return sorted(
        active_goals,
        key=lambda goal: (
            -_priority_rank(goal.priority),
            goal.created_at,
            goal.title,
        ),
    )[0].goal_id


def _filter_goals(
    *,
    goals: tuple[Goal, ...],
    query: str,
    statuses: tuple[GoalStatus, ...],
    priorities: tuple[GoalPriority, ...],
    tags: tuple[str, ...],
    include_children: bool,
    limit: int,
) -> tuple[Goal, ...]:
    query_text = query.strip().lower()
    wanted_tags = {tag.strip().lower() for tag in tags if tag.strip()}
    filtered: list[Goal] = []

    for goal in goals:
        if statuses and goal.status not in statuses:
            continue

        if priorities and goal.priority not in priorities:
            continue

        if wanted_tags and not wanted_tags.intersection(
            {tag.lower() for tag in goal.tags}
        ):
            continue

        if not include_children and goal.parent_goal_id is not None:
            continue

        if query_text and query_text not in _search_text(goal):
            continue

        filtered.append(goal)

    filtered.sort(
        key=lambda goal: (
            -_status_rank(goal.status),
            -_priority_rank(goal.priority),
            goal.updated_at,
            goal.title,
        )
    )
    return tuple(filtered[:limit])


def _search_text(goal: Goal) -> str:
    return " ".join(
        (
            goal.title,
            goal.description,
            goal.status.value,
            goal.priority.value,
            " ".join(goal.tags),
        )
    ).lower()


def _clean_tags(tags: tuple[str, ...]) -> tuple[str, ...]:
    cleaned: list[str] = []

    for tag in tags:
        value = tag.strip().lower()
        if value and value not in cleaned:
            cleaned.append(value)

    return tuple(cleaned)


def _priority_rank(priority: GoalPriority) -> int:
    ranks = {
        GoalPriority.LOW: 0,
        GoalPriority.NORMAL: 1,
        GoalPriority.HIGH: 2,
        GoalPriority.CRITICAL: 3,
    }
    return ranks[priority]


def _status_rank(status: GoalStatus) -> int:
    ranks = {
        GoalStatus.ACTIVE: 5,
        GoalStatus.BLOCKED: 4,
        GoalStatus.PAUSED: 3,
        GoalStatus.COMPLETED: 2,
        GoalStatus.CANCELLED: 1,
    }
    return ranks[status]