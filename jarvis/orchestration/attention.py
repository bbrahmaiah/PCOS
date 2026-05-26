from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from threading import RLock

from pydantic import Field, field_validator, model_validator

from jarvis.orchestration.ids import TaskId, utc_now, validate_task_id
from jarvis.orchestration.models import (
    OrchestrationModel,
    TaskKind,
    TaskPriority,
    TaskRequest,
)


class AttentionFocusKind(StrEnum):
    """
    What currently owns human-facing attention.
    """

    NONE = "none"
    CONVERSATION = "conversation"
    USER_WAITING = "user_waiting"
    COGNITION = "cognition"
    TOOL_ACTION = "tool_action"
    MEMORY = "memory"
    BACKGROUND = "background"
    MAINTENANCE = "maintenance"
    RECOVERY = "recovery"


class AttentionDecision(StrEnum):
    """
    Scheduling attention decision.
    """

    ALLOW = "allow"
    ALLOW_WITH_YIELD = "allow_with_yield"
    DEFER = "defer"
    SUPPRESS = "suppress"
    PREEMPT_BACKGROUND = "preempt_background"


class AttentionReason(StrEnum):
    """
    Machine-readable reason for attention decisions.
    """

    CONVERSATION_PROTECTED = "conversation_protected"
    USER_WAITING_PROTECTED = "user_waiting_protected"
    CRITICAL_ALLOWED = "critical_allowed"
    FOREGROUND_ALLOWED = "foreground_allowed"
    BACKGROUND_ALLOWED = "background_allowed"
    BACKGROUND_DEFERRED = "background_deferred"
    MAINTENANCE_SUPPRESSED = "maintenance_suppressed"
    TOOL_ALLOWED_WITH_YIELD = "tool_allowed_with_yield"
    MEMORY_ALLOWED_WITH_YIELD = "memory_allowed_with_yield"
    LOW_PRIORITY_DEFERRED = "low_priority_deferred"
    POLICY_DISABLED = "policy_disabled"
    FOCUS_STACK_EMPTY = "focus_stack_empty"


class AttentionUrgency(IntEnum):
    """
    Human-facing urgency.

    Higher value means more urgent.
    """

    NONE = 0
    LOW = 10
    NORMAL = 20
    HIGH = 30
    CRITICAL = 40


class AttentionScore(OrchestrationModel):
    """
    Attention scoring result.

    Scores are not execution authority. They are scheduling guidance.
    """

    task_id: TaskId
    focus_kind: AttentionFocusKind
    urgency: AttentionUrgency
    priority: TaskPriority
    score: int = Field(ge=0, le=100)
    reason: AttentionReason
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("task_id")
    @classmethod
    def _validate_task_id(cls, value: str) -> str:
        return validate_task_id(value)


class FocusFrame(OrchestrationModel):
    """
    One frame in the attention focus stack.

    The top frame is the strongest current focus.
    """

    focus_id: str
    kind: AttentionFocusKind
    owner_task_id: TaskId | None = None
    urgency: AttentionUrgency = AttentionUrgency.NORMAL
    description: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("focus_id", "description")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("owner_task_id")
    @classmethod
    def _validate_owner_task_id(cls, value: str | None) -> str | None:
        if value is None:
            return None

        return validate_task_id(value)


class FocusStack(OrchestrationModel):
    """
    Immutable attention focus stack.

    Focus is explicit. Hidden focus is not allowed.
    """

    frames: tuple[FocusFrame, ...] = ()

    @property
    def empty(self) -> bool:
        return not self.frames

    @property
    def current(self) -> FocusFrame | None:
        if not self.frames:
            return None

        return self.frames[-1]

    def push(self, frame: FocusFrame) -> FocusStack:
        """
        Return a new stack with a focus frame pushed.
        """

        return self.model_copy(update={"frames": self.frames + (frame,)})

    def pop(self) -> FocusStack:
        """
        Return a new stack with the current focus removed.
        """

        if not self.frames:
            return self

        return self.model_copy(update={"frames": self.frames[:-1]})

    def clear(self) -> FocusStack:
        """
        Return an empty focus stack.
        """

        return self.model_copy(update={"frames": ()})

    def contains_kind(self, kind: AttentionFocusKind) -> bool:
        """
        Return whether the stack contains a focus kind.
        """

        return any(frame.kind == kind for frame in self.frames)


class AttentionContext(OrchestrationModel):
    """
    Current attention context.

    This is what matters right now.
    """

    active_conversation: bool = False
    user_waiting: bool = False
    speech_active: bool = False
    response_playing: bool = False
    active_foreground_tasks: int = Field(default=0, ge=0)
    active_background_tasks: int = Field(default=0, ge=0)
    focus_stack: FocusStack = Field(default_factory=FocusStack)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @property
    def conversation_protected(self) -> bool:
        return (
            self.active_conversation
            or self.user_waiting
            or self.speech_active
            or self.response_playing
            or self.focus_stack.contains_kind(AttentionFocusKind.CONVERSATION)
            or self.focus_stack.contains_kind(AttentionFocusKind.USER_WAITING)
        )


class AttentionPolicy(OrchestrationModel):
    """
    Attention policy.

    Conversation responsiveness is inviolable.
    """

    enabled: bool = True
    protect_conversation: bool = True
    defer_background_during_conversation: bool = True
    suppress_maintenance_during_conversation: bool = True
    allow_critical_during_conversation: bool = True
    max_background_tasks_during_conversation: int = Field(default=0, ge=0)
    max_background_tasks_idle: int = Field(default=2, ge=0)

    @model_validator(mode="after")
    def _validate_policy(self) -> AttentionPolicy:
        conversation_limit = self.max_background_tasks_during_conversation
        idle_limit = self.max_background_tasks_idle

        if conversation_limit > idle_limit:
            raise ValueError(
                "conversation background limit cannot exceed idle background limit."
            )

        return self


class AttentionEvaluation(OrchestrationModel):
    """
    Result of evaluating a task against attention context.
    """

    task_id: TaskId
    decision: AttentionDecision
    reason: AttentionReason
    allowed: bool
    score: AttentionScore
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("task_id")
    @classmethod
    def _validate_task_id(cls, value: str) -> str:
        return validate_task_id(value)


@dataclass(frozen=True, slots=True)
class AttentionRuntimeConfig:
    """
    Attention runtime configuration.
    """

    name: str = "attention_runtime"

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")


@dataclass(frozen=True, slots=True)
class AttentionRuntimeSnapshot:
    """
    Attention runtime diagnostics.
    """

    name: str
    evaluation_count: int
    allow_count: int
    defer_count: int
    suppress_count: int
    preempt_count: int
    current_focus: AttentionFocusKind | None
    conversation_protected: bool
    last_decision: AttentionDecision | None
    last_reason: AttentionReason | None


class AttentionRuntime:
    """
    Phase 6 Attention Runtime.

    Responsibilities:
    - protect conversation responsiveness
    - evaluate task attention priority
    - maintain explicit focus stack
    - tell the scheduler whether work may run now, yield, defer, or suppress

    Non-responsibilities:
    - no task execution
    - no resource budget approval
    - no worker coordination
    - no direct cancellation
    """

    def __init__(
        self,
        *,
        config: AttentionRuntimeConfig | None = None,
        policy: AttentionPolicy | None = None,
        context: AttentionContext | None = None,
    ) -> None:
        self._config = config or AttentionRuntimeConfig()
        self._config.validate()

        self._policy = policy or AttentionPolicy()
        self._context = context or AttentionContext()
        self._lock = RLock()

        self._evaluation_count = 0
        self._allow_count = 0
        self._defer_count = 0
        self._suppress_count = 0
        self._preempt_count = 0
        self._last_decision: AttentionDecision | None = None
        self._last_reason: AttentionReason | None = None

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def context(self) -> AttentionContext:
        return self._context

    @property
    def policy(self) -> AttentionPolicy:
        return self._policy

    def update_context(self, context: AttentionContext) -> None:
        """
        Replace the current attention context.
        """

        with self._lock:
            self._context = context

    def push_focus(self, frame: FocusFrame) -> AttentionContext:
        """
        Push an attention focus frame.
        """

        with self._lock:
            self._context = self._context.model_copy(
                update={"focus_stack": self._context.focus_stack.push(frame)}
            )

            return self._context

    def pop_focus(self) -> AttentionContext:
        """
        Pop the current attention focus frame.
        """

        with self._lock:
            self._context = self._context.model_copy(
                update={"focus_stack": self._context.focus_stack.pop()}
            )

            return self._context

    def clear_focus(self) -> AttentionContext:
        """
        Clear all focus frames.
        """

        with self._lock:
            self._context = self._context.model_copy(
                update={"focus_stack": self._context.focus_stack.clear()}
            )

            return self._context

    def evaluate(self, task: TaskRequest) -> AttentionEvaluation:
        """
        Evaluate a task against attention context.

        The scheduler may use this result, but this runtime never schedules
        directly.
        """

        with self._lock:
            context = self._context
            policy = self._policy

        score = self._score(task=task, context=context)

        if not policy.enabled:
            evaluation = self._evaluation(
                task=task,
                decision=AttentionDecision.ALLOW,
                reason=AttentionReason.POLICY_DISABLED,
                allowed=True,
                score=score,
            )
            self._record(evaluation)

            return evaluation

        if task.priority == TaskPriority.CRITICAL:
            evaluation = self._evaluation(
                task=task,
                decision=AttentionDecision.ALLOW,
                reason=AttentionReason.CRITICAL_ALLOWED,
                allowed=True,
                score=score,
            )
            self._record(evaluation)

            return evaluation

        if context.conversation_protected and policy.protect_conversation:
            evaluation = self._evaluate_during_conversation(
                task=task,
                context=context,
                policy=policy,
                score=score,
            )
            self._record(evaluation)

            return evaluation

        evaluation = self._evaluate_idle(
            task=task,
            context=context,
            policy=policy,
            score=score,
        )
        self._record(evaluation)

        return evaluation

    def snapshot(self) -> AttentionRuntimeSnapshot:
        """
        Return runtime diagnostics.
        """

        with self._lock:
            current = self._context.focus_stack.current

            return AttentionRuntimeSnapshot(
                name=self.name,
                evaluation_count=self._evaluation_count,
                allow_count=self._allow_count,
                defer_count=self._defer_count,
                suppress_count=self._suppress_count,
                preempt_count=self._preempt_count,
                current_focus=current.kind if current is not None else None,
                conversation_protected=self._context.conversation_protected,
                last_decision=self._last_decision,
                last_reason=self._last_reason,
            )

    def reset_metrics(self) -> None:
        """
        Reset metrics only.
        """

        with self._lock:
            self._evaluation_count = 0
            self._allow_count = 0
            self._defer_count = 0
            self._suppress_count = 0
            self._preempt_count = 0
            self._last_decision = None
            self._last_reason = None

    def _evaluate_during_conversation(
        self,
        *,
        task: TaskRequest,
        context: AttentionContext,
        policy: AttentionPolicy,
        score: AttentionScore,
    ) -> AttentionEvaluation:
        if task.kind == TaskKind.CONVERSATION_TURN:
            return self._evaluation(
                task=task,
                decision=AttentionDecision.ALLOW,
                reason=AttentionReason.CONVERSATION_PROTECTED,
                allowed=True,
                score=score,
            )

        if task.kind == TaskKind.TOOL_ACTION:
            return self._evaluation(
                task=task,
                decision=AttentionDecision.ALLOW_WITH_YIELD,
                reason=AttentionReason.TOOL_ALLOWED_WITH_YIELD,
                allowed=True,
                score=score,
            )

        if task.kind == TaskKind.MEMORY_RETRIEVAL:
            return self._evaluation(
                task=task,
                decision=AttentionDecision.ALLOW_WITH_YIELD,
                reason=AttentionReason.MEMORY_ALLOWED_WITH_YIELD,
                allowed=True,
                score=score,
            )

        if task.background or task.priority == TaskPriority.BACKGROUND:
            if (
                policy.defer_background_during_conversation
                and context.active_background_tasks
                >= policy.max_background_tasks_during_conversation
            ):
                return self._evaluation(
                    task=task,
                    decision=AttentionDecision.DEFER,
                    reason=AttentionReason.BACKGROUND_DEFERRED,
                    allowed=False,
                    score=score,
                )

        if task.kind == TaskKind.BACKGROUND_MAINTENANCE:
            if policy.suppress_maintenance_during_conversation:
                return self._evaluation(
                    task=task,
                    decision=AttentionDecision.SUPPRESS,
                    reason=AttentionReason.MAINTENANCE_SUPPRESSED,
                    allowed=False,
                    score=score,
                )

        return self._evaluation(
            task=task,
            decision=AttentionDecision.DEFER,
            reason=AttentionReason.LOW_PRIORITY_DEFERRED,
            allowed=False,
            score=score,
        )

    def _evaluate_idle(
        self,
        *,
        task: TaskRequest,
        context: AttentionContext,
        policy: AttentionPolicy,
        score: AttentionScore,
    ) -> AttentionEvaluation:
        if task.background or task.priority == TaskPriority.BACKGROUND:
            if context.active_background_tasks >= policy.max_background_tasks_idle:
                return self._evaluation(
                    task=task,
                    decision=AttentionDecision.DEFER,
                    reason=AttentionReason.BACKGROUND_DEFERRED,
                    allowed=False,
                    score=score,
                )

            return self._evaluation(
                task=task,
                decision=AttentionDecision.ALLOW,
                reason=AttentionReason.BACKGROUND_ALLOWED,
                allowed=True,
                score=score,
            )

        return self._evaluation(
            task=task,
            decision=AttentionDecision.ALLOW,
            reason=AttentionReason.FOREGROUND_ALLOWED,
            allowed=True,
            score=score,
        )

    def _score(
        self,
        *,
        task: TaskRequest,
        context: AttentionContext,
    ) -> AttentionScore:
        urgency = self._urgency_for_task(task)
        base = int(urgency)
        priority_boost = max(0, 40 - int(task.priority))
        conversation_boost = 20 if task.kind == TaskKind.CONVERSATION_TURN else 0
        background_penalty = 30 if task.background else 0
        protected_penalty = (
            20
            if context.conversation_protected
            and task.priority in {TaskPriority.LOW, TaskPriority.BACKGROUND}
            else 0
        )
        final_score = max(
            0,
            min(
                100,
                base + priority_boost + conversation_boost
                - background_penalty - protected_penalty,
            ),
        )

        return AttentionScore(
            task_id=task.task_id,
            focus_kind=self._focus_kind_for_task(task),
            urgency=urgency,
            priority=task.priority,
            score=final_score,
            reason=(
                AttentionReason.CONVERSATION_PROTECTED
                if task.kind == TaskKind.CONVERSATION_TURN
                else AttentionReason.FOREGROUND_ALLOWED
            ),
        )

    @staticmethod
    def _urgency_for_task(task: TaskRequest) -> AttentionUrgency:
        if task.priority == TaskPriority.CRITICAL:
            return AttentionUrgency.CRITICAL

        if task.kind == TaskKind.CONVERSATION_TURN:
            return AttentionUrgency.CRITICAL

        if task.priority == TaskPriority.HIGH:
            return AttentionUrgency.HIGH

        if task.priority == TaskPriority.NORMAL:
            return AttentionUrgency.NORMAL

        if task.priority == TaskPriority.LOW:
            return AttentionUrgency.LOW

        return AttentionUrgency.NONE

    @staticmethod
    def _focus_kind_for_task(task: TaskRequest) -> AttentionFocusKind:
        return {
            TaskKind.CONVERSATION_TURN: AttentionFocusKind.CONVERSATION,
            TaskKind.COGNITION: AttentionFocusKind.COGNITION,
            TaskKind.MEMORY_RETRIEVAL: AttentionFocusKind.MEMORY,
            TaskKind.MEMORY_WRITE: AttentionFocusKind.MEMORY,
            TaskKind.TOOL_ACTION: AttentionFocusKind.TOOL_ACTION,
            TaskKind.BACKGROUND_MAINTENANCE: AttentionFocusKind.BACKGROUND,
            TaskKind.HEALTH_CHECK: AttentionFocusKind.MAINTENANCE,
            TaskKind.RECOVERY: AttentionFocusKind.RECOVERY,
            TaskKind.PRESENCE_EVENT: AttentionFocusKind.CONVERSATION,
            TaskKind.OBSERVABILITY: AttentionFocusKind.MAINTENANCE,
            TaskKind.SYSTEM: AttentionFocusKind.MAINTENANCE,
        }[task.kind]

    @staticmethod
    def _evaluation(
        *,
        task: TaskRequest,
        decision: AttentionDecision,
        reason: AttentionReason,
        allowed: bool,
        score: AttentionScore,
    ) -> AttentionEvaluation:
        return AttentionEvaluation(
            task_id=task.task_id,
            decision=decision,
            reason=reason,
            allowed=allowed,
            score=score,
        )

    def _record(self, evaluation: AttentionEvaluation) -> None:
        with self._lock:
            self._evaluation_count += 1
            self._last_decision = evaluation.decision
            self._last_reason = evaluation.reason

            if evaluation.decision in {
                AttentionDecision.ALLOW,
                AttentionDecision.ALLOW_WITH_YIELD,
            }:
                self._allow_count += 1

            elif evaluation.decision == AttentionDecision.DEFER:
                self._defer_count += 1

            elif evaluation.decision == AttentionDecision.SUPPRESS:
                self._suppress_count += 1

            elif evaluation.decision == AttentionDecision.PREEMPT_BACKGROUND:
                self._preempt_count += 1