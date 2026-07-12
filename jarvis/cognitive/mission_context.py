from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from jarvis.cognitive.contracts import (
    AttentionItem,
    AttentionPriority,
    Goal,
    GoalPriority,
    GoalStatus,
    WorkingMemoryItem,
    WorkingMemoryKind,
    utc_now,
)

if TYPE_CHECKING:
    from jarvis.cognition.models import CognitionContextItem, CognitionRequest


class MissionContextStatus(StrEnum):
    READY = "ready"
    DEGRADED = "degraded"
    BLOCKED = "blocked"


class MissionContextUrgency(StrEnum):
    BACKGROUND = "background"
    NORMAL = "normal"
    IMPORTANT = "important"
    URGENT = "urgent"
    CRITICAL = "critical"


class MissionContextInterruptionPolicy(StrEnum):
    STAY_SILENT = "stay_silent"
    MENTION_WHEN_NATURAL = "mention_when_natural"
    RESPOND_NOW = "respond_now"
    INTERRUPT_NOW = "interrupt_now"


@dataclass(frozen=True, slots=True)
class MissionContextInput:
    user_label: str = "Balu"
    current_project: str | None = None
    current_goal: str | None = None
    current_task: str | None = None
    environment: str | None = None
    request_text: str = ""
    attention_items: tuple[AttentionItem, ...] = ()
    working_memory_items: tuple[WorkingMemoryItem, ...] = ()
    active_goals: tuple[Goal, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.user_label.strip():
            raise ValueError("mission context user_label cannot be empty.")


@dataclass(frozen=True, slots=True)
class MissionContextState:
    user_label: str
    current_project: str | None
    current_goal: str | None
    current_task: str | None
    environment: str | None
    urgency: MissionContextUrgency
    interruption_policy: MissionContextInterruptionPolicy
    focus_summary: str
    risk_summary: str | None
    open_questions: tuple[str, ...]
    updated_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def should_interrupt(self) -> bool:
        return (
            self.interruption_policy
            == MissionContextInterruptionPolicy.INTERRUPT_NOW
        )

    @property
    def should_respond_now(self) -> bool:
        return self.interruption_policy in {
            MissionContextInterruptionPolicy.RESPOND_NOW,
            MissionContextInterruptionPolicy.INTERRUPT_NOW,
        }


@dataclass(frozen=True, slots=True)
class MissionContextResult:
    status: MissionContextStatus
    state: MissionContextState
    reason: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == MissionContextStatus.READY


@dataclass(frozen=True, slots=True)
class MissionContextSnapshot:
    status: MissionContextStatus
    state: MissionContextState
    update_count: int
    last_reason: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


class MissionContextRuntime:
    """
    Protected mission context engine.

    Mission context is the meaning layer between raw facts and action:
    it tracks what Balu is doing, why it matters, what is urgent, and when
    JARVIS should interrupt or stay quiet. It does not execute tools, speak,
    or invent fixed responses.
    """

    def __init__(self) -> None:
        self._state = MissionContextState(
            user_label="Balu",
            current_project=None,
            current_goal=None,
            current_task=None,
            environment=None,
            urgency=MissionContextUrgency.NORMAL,
            interruption_policy=MissionContextInterruptionPolicy.MENTION_WHEN_NATURAL,
            focus_summary="No active mission context yet.",
            risk_summary=None,
            open_questions=(),
            updated_at=utc_now(),
            metadata={},
        )
        self._status = MissionContextStatus.READY
        self._update_count = 0
        self._last_reason = "mission context initialized"

    @property
    def state(self) -> MissionContextState:
        return self._state

    def update(self, request: MissionContextInput) -> MissionContextResult:
        self._update_count += 1

        project = _first_text(
            request.current_project,
            _working_memory_value(
                request.working_memory_items,
                WorkingMemoryKind.PROJECT,
            ),
            _metadata_text(request.metadata, "current_project"),
            self._state.current_project,
        )
        goal = _first_text(
            request.current_goal,
            _active_goal_title(request.active_goals),
            _working_memory_value(
                request.working_memory_items,
                WorkingMemoryKind.OBJECTIVE,
            ),
            _metadata_text(request.metadata, "current_goal"),
            self._state.current_goal,
        )
        task = _first_text(
            request.current_task,
            _working_memory_value(request.working_memory_items, WorkingMemoryKind.TASK),
            _metadata_text(request.metadata, "current_task"),
            _request_task_hint(request.request_text),
            self._state.current_task,
        )
        environment = _first_text(
            request.environment,
            _working_memory_value(
                request.working_memory_items,
                WorkingMemoryKind.SCREEN_CONTEXT,
            ),
            _metadata_text(request.metadata, "environment"),
            self._state.environment,
        )

        urgency = _derive_urgency(
            attention_items=request.attention_items,
            working_memory_items=request.working_memory_items,
            active_goals=request.active_goals,
            metadata=request.metadata,
            previous=self._state.urgency,
        )
        interruption_policy = _interruption_policy_for(
            urgency=urgency,
            request_text=request.request_text,
            metadata=request.metadata,
        )
        risk_summary = _risk_summary(
            attention_items=request.attention_items,
            working_memory_items=request.working_memory_items,
            metadata=request.metadata,
        )
        open_questions = _open_questions(
            project=project,
            goal=goal,
            task=task,
            request_text=request.request_text,
            metadata=request.metadata,
        )
        focus_summary = _focus_summary(
            user_label=request.user_label,
            project=project,
            goal=goal,
            task=task,
            environment=environment,
            urgency=urgency,
            risk_summary=risk_summary,
        )

        self._state = MissionContextState(
            user_label=request.user_label.strip(),
            current_project=project,
            current_goal=goal,
            current_task=task,
            environment=environment,
            urgency=urgency,
            interruption_policy=interruption_policy,
            focus_summary=focus_summary,
            risk_summary=risk_summary,
            open_questions=open_questions,
            updated_at=utc_now(),
            metadata={
                **request.metadata,
                "attention_items": len(request.attention_items),
                "working_memory_items": len(request.working_memory_items),
                "active_goals": len(request.active_goals),
            },
        )
        self._status = MissionContextStatus.READY
        self._last_reason = "mission context updated"

        return MissionContextResult(
            status=self._status,
            state=self._state,
            reason=self._last_reason,
            created_at=utc_now(),
            metadata={
                "urgency": urgency.value,
                "interruption_policy": interruption_policy.value,
                "open_question_count": len(open_questions),
            },
        )

    def enrich_request(self, request: CognitionRequest) -> CognitionRequest:
        item = self.to_context_item()
        existing_context = request.context
        context = existing_context.model_copy(
            update={
                "items": (item, *existing_context.items),
                "metadata": {
                    **existing_context.metadata,
                    "mission_context_urgency": self._state.urgency.value,
                    "mission_context_policy": self._state.interruption_policy.value,
                },
            }
        )
        return request.model_copy(
            update={
                "context": context,
                "metadata": {
                    **request.metadata,
                    "mission_context": self._state.focus_summary,
                    "mission_context_urgency": self._state.urgency.value,
                    "mission_context_policy": self._state.interruption_policy.value,
                },
            }
        )

    def to_context_item(self) -> CognitionContextItem:
        from jarvis.cognition.models import CognitionContextItem

        state = self._state
        lines = [
            f"User: {state.user_label}",
            f"Focus: {state.focus_summary}",
            f"Urgency: {state.urgency.value}",
            f"Interrupt policy: {state.interruption_policy.value}",
        ]
        if state.risk_summary is not None:
            lines.append(f"Risk: {state.risk_summary}")
        if state.open_questions:
            lines.append("Open questions: " + "; ".join(state.open_questions))

        return CognitionContextItem(
            kind="mission_context",
            text="\n".join(lines),
            score=1.0,
            source="mission_context_runtime",
            metadata={
                "current_project": state.current_project,
                "current_goal": state.current_goal,
                "current_task": state.current_task,
                "environment": state.environment,
                "urgency": state.urgency.value,
                "interruption_policy": state.interruption_policy.value,
            },
        )

    def snapshot(self) -> MissionContextSnapshot:
        return MissionContextSnapshot(
            status=self._status,
            state=self._state,
            update_count=self._update_count,
            last_reason=self._last_reason,
            created_at=utc_now(),
        )

    def clear(self) -> MissionContextResult:
        self._state = MissionContextState(
            user_label=self._state.user_label,
            current_project=None,
            current_goal=None,
            current_task=None,
            environment=None,
            urgency=MissionContextUrgency.NORMAL,
            interruption_policy=MissionContextInterruptionPolicy.MENTION_WHEN_NATURAL,
            focus_summary="No active mission context yet.",
            risk_summary=None,
            open_questions=(),
            updated_at=utc_now(),
            metadata={},
        )
        self._last_reason = "mission context cleared"
        return MissionContextResult(
            status=MissionContextStatus.READY,
            state=self._state,
            reason=self._last_reason,
            created_at=utc_now(),
        )


def mission_context_input_from_request(
    request: CognitionRequest,
    *,
    user_label: str = "Balu",
) -> MissionContextInput:
    return MissionContextInput(
        user_label=user_label,
        current_project=_metadata_text(request.metadata, "current_project"),
        current_goal=_metadata_text(request.metadata, "current_goal"),
        current_task=_metadata_text(request.metadata, "current_task"),
        environment=_metadata_text(request.metadata, "environment"),
        request_text=request.text,
        metadata=request.metadata,
    )


def _first_text(*values: str | None) -> str | None:
    for value in values:
        if value is not None and value.strip():
            return " ".join(value.split())
    return None


def _metadata_text(metadata: dict[str, object], key: str) -> str | None:
    value = metadata.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _working_memory_value(
    items: tuple[WorkingMemoryItem, ...],
    kind: WorkingMemoryKind,
) -> str | None:
    for item in items:
        if item.kind == kind and item.value.strip():
            return item.value.strip()
    return None


def _active_goal_title(goals: tuple[Goal, ...]) -> str | None:
    ordered = sorted(
        goals,
        key=lambda goal: _goal_priority_rank(goal.priority),
        reverse=True,
    )
    for goal in ordered:
        if goal.status == GoalStatus.ACTIVE and goal.title.strip():
            return goal.title.strip()
    return None


def _request_task_hint(text: str) -> str | None:
    clean = " ".join(text.split())
    if not clean:
        return None
    if len(clean) <= 140:
        return clean
    return f"{clean[:137].rstrip()}..."


def _derive_urgency(
    *,
    attention_items: tuple[AttentionItem, ...],
    working_memory_items: tuple[WorkingMemoryItem, ...],
    active_goals: tuple[Goal, ...],
    metadata: dict[str, object],
    previous: MissionContextUrgency,
) -> MissionContextUrgency:
    explicit = _metadata_text(metadata, "urgency")
    if explicit is not None:
        normalized = explicit.casefold()
        for urgency in MissionContextUrgency:
            if urgency.value == normalized:
                return urgency

    priorities = [item.priority for item in attention_items]
    priorities.extend(item.importance for item in working_memory_items)
    priorities.extend(
        _attention_priority_for_goal(goal.priority) for goal in active_goals
    )

    if AttentionPriority.CRITICAL in priorities:
        return MissionContextUrgency.CRITICAL
    if AttentionPriority.HIGH in priorities:
        return MissionContextUrgency.URGENT
    if AttentionPriority.NORMAL in priorities:
        return MissionContextUrgency.NORMAL
    if priorities:
        return MissionContextUrgency.IMPORTANT
    if previous != MissionContextUrgency.BACKGROUND:
        return previous
    return MissionContextUrgency.NORMAL


def _interruption_policy_for(
    *,
    urgency: MissionContextUrgency,
    request_text: str,
    metadata: dict[str, object],
) -> MissionContextInterruptionPolicy:
    explicit = _metadata_text(metadata, "interruption_policy")
    if explicit is not None:
        normalized = explicit.casefold()
        for policy in MissionContextInterruptionPolicy:
            if policy.value == normalized:
                return policy

    if urgency == MissionContextUrgency.CRITICAL:
        return MissionContextInterruptionPolicy.INTERRUPT_NOW
    if urgency == MissionContextUrgency.URGENT:
        return MissionContextInterruptionPolicy.RESPOND_NOW
    if request_text.strip():
        return MissionContextInterruptionPolicy.RESPOND_NOW
    if urgency == MissionContextUrgency.IMPORTANT:
        return MissionContextInterruptionPolicy.MENTION_WHEN_NATURAL
    return MissionContextInterruptionPolicy.STAY_SILENT


def _risk_summary(
    *,
    attention_items: tuple[AttentionItem, ...],
    working_memory_items: tuple[WorkingMemoryItem, ...],
    metadata: dict[str, object],
) -> str | None:
    explicit = _metadata_text(metadata, "risk_summary")
    if explicit is not None:
        return explicit

    risk_memory = _working_memory_value(working_memory_items, WorkingMemoryKind.RISK)
    if risk_memory is not None:
        return risk_memory

    critical = tuple(
        item
        for item in attention_items
        if item.priority in {AttentionPriority.HIGH, AttentionPriority.CRITICAL}
    )
    if critical:
        first = critical[0]
        return f"{first.title}: {first.summary}"
    return None


def _open_questions(
    *,
    project: str | None,
    goal: str | None,
    task: str | None,
    request_text: str,
    metadata: dict[str, object],
) -> tuple[str, ...]:
    explicit = metadata.get("open_questions")
    if isinstance(explicit, tuple) and all(isinstance(item, str) for item in explicit):
        return tuple(item.strip() for item in explicit if item.strip())

    questions: list[str] = []
    if project is None:
        questions.append("current project is unknown")
    if goal is None and request_text.strip():
        questions.append("current goal is not confirmed")
    if task is None and request_text.strip():
        questions.append("current task is not confirmed")
    return tuple(questions[:4])


def _focus_summary(
    *,
    user_label: str,
    project: str | None,
    goal: str | None,
    task: str | None,
    environment: str | None,
    urgency: MissionContextUrgency,
    risk_summary: str | None,
) -> str:
    parts = [f"{user_label.strip()}"]
    if project is not None:
        parts.append(f"is working on {project}")
    if goal is not None:
        parts.append(f"to achieve {goal}")
    if task is not None:
        parts.append(f"current task: {task}")
    if environment is not None:
        parts.append(f"environment: {environment}")
    parts.append(f"urgency is {urgency.value}")
    if risk_summary is not None:
        parts.append(f"risk: {risk_summary}")
    return "; ".join(parts) + "."


def _goal_priority_rank(priority: GoalPriority) -> int:
    return {
        GoalPriority.LOW: 1,
        GoalPriority.NORMAL: 2,
        GoalPriority.HIGH: 3,
        GoalPriority.CRITICAL: 4,
    }[priority]


def _attention_priority_for_goal(priority: GoalPriority) -> AttentionPriority:
    return {
        GoalPriority.LOW: AttentionPriority.LOW,
        GoalPriority.NORMAL: AttentionPriority.NORMAL,
        GoalPriority.HIGH: AttentionPriority.HIGH,
        GoalPriority.CRITICAL: AttentionPriority.CRITICAL,
    }[priority]
