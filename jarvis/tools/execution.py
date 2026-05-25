from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from threading import RLock

from pydantic import Field, field_validator, model_validator

from jarvis.tools.ids import new_action_result_id, utc_now
from jarvis.tools.models import (
    ActionPlan,
    ActionRisk,
    ActionStatus,
    ToolModel,
)


class ActionExecutionPriority(StrEnum):
    """
    Priority for future scheduling/orchestration.

    Step 2 does not schedule work yet. It only records the priority contract
    so future schedulers can reason about action importance.
    """

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class ActionExecutionEventKind(StrEnum):
    """
    Event accepted by the ActionExecutionProtocol.

    These events describe lifecycle intent. They do not execute tools.
    """

    PLAN_ACCEPTED = "plan_accepted"
    VALIDATION_STARTED = "validation_started"
    VALIDATION_PASSED = "validation_passed"
    VALIDATION_FAILED = "validation_failed"
    POLICY_DENIED = "policy_denied"
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"
    START_REQUESTED = "start_requested"
    PAUSE_REQUESTED = "pause_requested"
    PAUSED = "paused"
    RESUME_REQUESTED = "resume_requested"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMEOUT_OCCURRED = "timeout_occurred"
    RETRY_REQUESTED = "retry_requested"
    ROLLBACK_COMPLETED = "rollback_completed"


class ActionExecutionDisposition(StrEnum):
    """
    Result disposition for a protocol transition.
    """

    APPLIED = "applied"
    REJECTED = "rejected"
    IGNORED = "ignored"


class ActionExecutionState(ToolModel):
    """
    Runtime state for one governed action.

    This is the core of Step 2: an action is now a managed runtime object with
    lifecycle state, attempts, priority, cancellation, pause, and rollback
    metadata.
    """

    execution_id: str = Field(default_factory=new_action_result_id)
    action_id: str
    plan_id: str
    status: ActionStatus = ActionStatus.CREATED
    priority: ActionExecutionPriority = ActionExecutionPriority.NORMAL
    risk: ActionRisk = ActionRisk.LOW
    current_step_index: int = Field(default=0, ge=0)
    total_steps: int = Field(ge=1)
    attempt: int = Field(default=1, ge=1)
    max_attempts: int = Field(default=1, ge=1)
    interruptible: bool = True
    cancellable: bool = True
    rollback_supported: bool = False
    cancellation_requested: bool = False
    pause_requested: bool = False
    timeout_ms: int | None = Field(default=None, ge=1)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("execution_id", "action_id", "plan_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @model_validator(mode="after")
    def _validate_step_index(self) -> ActionExecutionState:
        if self.current_step_index >= self.total_steps:
            if self.status not in {
                ActionStatus.SUCCEEDED,
                ActionStatus.FAILED,
                ActionStatus.CANCELLED,
                ActionStatus.BLOCKED,
                ActionStatus.ROLLED_BACK,
            }:
                raise ValueError(
                    "current_step_index must be less than total_steps "
                    "for active executions."
                )

        if self.attempt > self.max_attempts:
            raise ValueError("attempt cannot exceed max_attempts.")

        if self.completed_at is not None:
            terminal = {
                ActionStatus.SUCCEEDED,
                ActionStatus.FAILED,
                ActionStatus.CANCELLED,
                ActionStatus.BLOCKED,
                ActionStatus.ROLLED_BACK,
            }

            if self.status not in terminal:
                raise ValueError(
                    "completed_at is only valid for terminal statuses."
                )

        return self

    @property
    def terminal(self) -> bool:
        return self.status in {
            ActionStatus.SUCCEEDED,
            ActionStatus.FAILED,
            ActionStatus.CANCELLED,
            ActionStatus.BLOCKED,
            ActionStatus.ROLLED_BACK,
        }

    @property
    def active(self) -> bool:
        return self.status in {
            ActionStatus.VALIDATING,
            ActionStatus.APPROVED,
            ActionStatus.RUNNING,
            ActionStatus.PAUSING,
            ActionStatus.PAUSED,
            ActionStatus.CANCELLING,
        }

    @property
    def can_retry(self) -> bool:
        return self.status == ActionStatus.FAILED and self.attempt < self.max_attempts


class ActionExecutionEvent(ToolModel):
    """
    One lifecycle event applied to an action execution state.
    """

    event_id: str = Field(default_factory=new_action_result_id)
    execution_id: str
    action_id: str
    kind: ActionExecutionEventKind
    reason: str
    actor: str = "runtime"
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("event_id", "execution_id", "action_id", "reason", "actor")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class ActionExecutionTransition(ToolModel):
    """
    Result of applying one execution lifecycle event.
    """

    transition_id: str = Field(default_factory=new_action_result_id)
    event: ActionExecutionEvent
    previous_state: ActionExecutionState
    next_state: ActionExecutionState
    disposition: ActionExecutionDisposition
    changed: bool
    reason: str
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("transition_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @property
    def applied(self) -> bool:
        return self.disposition == ActionExecutionDisposition.APPLIED


@dataclass(frozen=True, slots=True)
class ActionExecutionProtocolConfig:
    """
    Configuration for the action execution protocol.

    This protocol validates lifecycle transitions only. It never runs tools.
    """

    name: str = "action_execution_protocol"
    allow_retry_from_failed: bool = True
    allow_rollback_from_cancelled: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")


@dataclass(frozen=True, slots=True)
class ActionExecutionProtocolSnapshot:
    """
    Observable diagnostics for the protocol.
    """

    name: str
    transition_count: int
    applied_count: int
    rejected_count: int
    ignored_count: int
    terminal_count: int
    last_status: ActionStatus | None
    last_event_kind: ActionExecutionEventKind | None
    last_error: str | None


class ActionExecutionProtocol:
    """
    Formal lifecycle protocol for governed actions.

    Responsibilities:
    - create execution state from typed action plans
    - enforce valid lifecycle transitions
    - support pause/resume/cancel/retry/rollback state transitions
    - reject unsafe or impossible transitions
    - expose diagnostics

    Non-responsibilities:
    - no tool execution
    - no shell execution
    - no file writes
    - no browser automation
    - no approval UI
    - no policy decision making
    """

    _TRANSITIONS: dict[
        tuple[ActionStatus, ActionExecutionEventKind],
        ActionStatus,
    ] = {
        (ActionStatus.CREATED, ActionExecutionEventKind.PLAN_ACCEPTED): (
            ActionStatus.PLANNED
        ),
        (ActionStatus.PLANNED, ActionExecutionEventKind.VALIDATION_STARTED): (
            ActionStatus.VALIDATING
        ),
        (ActionStatus.VALIDATING, ActionExecutionEventKind.VALIDATION_PASSED): (
            ActionStatus.APPROVED
        ),
        (ActionStatus.VALIDATING, ActionExecutionEventKind.APPROVAL_REQUIRED): (
            ActionStatus.WAITING_FOR_APPROVAL
        ),
        (ActionStatus.VALIDATING, ActionExecutionEventKind.VALIDATION_FAILED): (
            ActionStatus.BLOCKED
        ),
        (ActionStatus.VALIDATING, ActionExecutionEventKind.POLICY_DENIED): (
            ActionStatus.BLOCKED
        ),
        (
            ActionStatus.WAITING_FOR_APPROVAL,
            ActionExecutionEventKind.APPROVAL_GRANTED,
        ): ActionStatus.APPROVED,
        (
            ActionStatus.WAITING_FOR_APPROVAL,
            ActionExecutionEventKind.APPROVAL_DENIED,
        ): ActionStatus.BLOCKED,
        (ActionStatus.APPROVED, ActionExecutionEventKind.START_REQUESTED): (
            ActionStatus.RUNNING
        ),
        (ActionStatus.RUNNING, ActionExecutionEventKind.PAUSE_REQUESTED): (
            ActionStatus.PAUSING
        ),
        (ActionStatus.PAUSING, ActionExecutionEventKind.PAUSED): (
            ActionStatus.PAUSED
        ),
        (ActionStatus.PAUSED, ActionExecutionEventKind.RESUME_REQUESTED): (
            ActionStatus.RUNNING
        ),
        (ActionStatus.RUNNING, ActionExecutionEventKind.CANCEL_REQUESTED): (
            ActionStatus.CANCELLING
        ),
        (ActionStatus.PAUSED, ActionExecutionEventKind.CANCEL_REQUESTED): (
            ActionStatus.CANCELLING
        ),
        (ActionStatus.PAUSING, ActionExecutionEventKind.CANCEL_REQUESTED): (
            ActionStatus.CANCELLING
        ),
        (ActionStatus.CANCELLING, ActionExecutionEventKind.CANCELLED): (
            ActionStatus.CANCELLED
        ),
        (ActionStatus.RUNNING, ActionExecutionEventKind.SUCCEEDED): (
            ActionStatus.SUCCEEDED
        ),
        (ActionStatus.RUNNING, ActionExecutionEventKind.FAILED): (
            ActionStatus.FAILED
        ),
        (ActionStatus.RUNNING, ActionExecutionEventKind.TIMEOUT_OCCURRED): (
            ActionStatus.FAILED
        ),
        (ActionStatus.FAILED, ActionExecutionEventKind.RETRY_REQUESTED): (
            ActionStatus.PLANNED
        ),
        (ActionStatus.FAILED, ActionExecutionEventKind.ROLLBACK_COMPLETED): (
            ActionStatus.ROLLED_BACK
        ),
        (ActionStatus.BLOCKED, ActionExecutionEventKind.ROLLBACK_COMPLETED): (
            ActionStatus.ROLLED_BACK
        ),
        (ActionStatus.CANCELLED, ActionExecutionEventKind.ROLLBACK_COMPLETED): (
            ActionStatus.ROLLED_BACK
        ),
    }

    def __init__(
        self,
        *,
        config: ActionExecutionProtocolConfig | None = None,
    ) -> None:
        self._config = config or ActionExecutionProtocolConfig()
        self._config.validate()

        self._lock = RLock()
        self._transition_count = 0
        self._applied_count = 0
        self._rejected_count = 0
        self._ignored_count = 0
        self._terminal_count = 0
        self._last_status: ActionStatus | None = None
        self._last_event_kind: ActionExecutionEventKind | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def create_state(
        self,
        plan: ActionPlan,
        *,
        priority: ActionExecutionPriority = ActionExecutionPriority.NORMAL,
        max_attempts: int = 1,
        timeout_ms: int | None = None,
    ) -> ActionExecutionState:
        """
        Create a runtime execution state from a typed action plan.

        This still does not execute anything.
        """

        rollback_supported = any(step.rollback_supported for step in plan.steps)
        interruptible = all(step.interruptible for step in plan.steps)

        return ActionExecutionState(
            action_id=plan.action_id,
            plan_id=plan.plan_id,
            status=ActionStatus.CREATED,
            priority=priority,
            risk=plan.risk,
            total_steps=len(plan.steps),
            max_attempts=max_attempts,
            interruptible=interruptible,
            cancellable=True,
            rollback_supported=rollback_supported,
            timeout_ms=timeout_ms,
            metadata={
                "protocol": self.name,
                "plan_goal": plan.goal,
                "plan_status": plan.status.value,
            },
        )

    def event(
        self,
        state: ActionExecutionState,
        kind: ActionExecutionEventKind,
        *,
        reason: str,
        actor: str = "runtime",
        metadata: dict[str, object] | None = None,
    ) -> ActionExecutionEvent:
        """
        Create a typed execution event for a state.
        """

        return ActionExecutionEvent(
            execution_id=state.execution_id,
            action_id=state.action_id,
            kind=kind,
            reason=reason,
            actor=actor,
            metadata=metadata or {},
        )

    def apply(
        self,
        state: ActionExecutionState,
        event: ActionExecutionEvent,
    ) -> ActionExecutionTransition:
        """
        Apply one lifecycle event to an execution state.
        """

        with self._lock:
            self._transition_count += 1
            self._last_event_kind = event.kind
            self._last_error = None

        try:
            transition = self._apply_checked(state=state, event=event)
            self._record_transition(transition)

            return transition

        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"

            raise

    def transition(
        self,
        state: ActionExecutionState,
        kind: ActionExecutionEventKind,
        *,
        reason: str,
        actor: str = "runtime",
        metadata: dict[str, object] | None = None,
    ) -> ActionExecutionTransition:
        """
        Convenience method: create and apply one event.
        """

        return self.apply(
            state,
            self.event(
                state,
                kind,
                reason=reason,
                actor=actor,
                metadata=metadata,
            ),
        )

    def snapshot(self) -> ActionExecutionProtocolSnapshot:
        """
        Return protocol diagnostics.
        """

        with self._lock:
            return ActionExecutionProtocolSnapshot(
                name=self.name,
                transition_count=self._transition_count,
                applied_count=self._applied_count,
                rejected_count=self._rejected_count,
                ignored_count=self._ignored_count,
                terminal_count=self._terminal_count,
                last_status=self._last_status,
                last_event_kind=self._last_event_kind,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Reset diagnostics.
        """

        with self._lock:
            self._transition_count = 0
            self._applied_count = 0
            self._rejected_count = 0
            self._ignored_count = 0
            self._terminal_count = 0
            self._last_status = None
            self._last_event_kind = None
            self._last_error = None

    def _apply_checked(
        self,
        *,
        state: ActionExecutionState,
        event: ActionExecutionEvent,
    ) -> ActionExecutionTransition:
        if event.execution_id != state.execution_id:
            return self._rejected(
                state=state,
                event=event,
                reason="event execution_id does not match state execution_id",
            )

        if event.action_id != state.action_id:
            return self._rejected(
                state=state,
                event=event,
                reason="event action_id does not match state action_id",
            )

        if state.terminal:
            return self._terminal_transition(state=state, event=event)

        guard = self._guard(state=state, event=event)

        if guard is not None:
            return self._rejected(state=state, event=event, reason=guard)

        target_status = self._TRANSITIONS.get((state.status, event.kind))

        if target_status is None:
            return self._ignored(
                state=state,
                event=event,
                reason=(
                    "event is not valid for current action status: "
                    f"{state.status.value} + {event.kind.value}"
                ),
            )

        next_state = self._next_state(
            state=state,
            event=event,
            status=target_status,
        )

        return ActionExecutionTransition(
            event=event,
            previous_state=state,
            next_state=next_state,
            disposition=ActionExecutionDisposition.APPLIED,
            changed=next_state != state,
            reason="action execution transition applied",
        )

    def _guard(
        self,
        *,
        state: ActionExecutionState,
        event: ActionExecutionEvent,
    ) -> str | None:
        if event.kind == ActionExecutionEventKind.PAUSE_REQUESTED:
            if not state.interruptible:
                return "non-interruptible actions cannot be paused"

        if event.kind == ActionExecutionEventKind.CANCEL_REQUESTED:
            if not state.cancellable:
                return "non-cancellable actions cannot be cancelled"

        if event.kind == ActionExecutionEventKind.RETRY_REQUESTED:
            if not self._config.allow_retry_from_failed:
                return "retry transitions are disabled"

            if state.status != ActionStatus.FAILED:
                return "only failed actions can be retried"

            if not state.can_retry:
                return "action has no retry attempts remaining"

        if event.kind == ActionExecutionEventKind.ROLLBACK_COMPLETED:
            if state.status == ActionStatus.CANCELLED:
                if not self._config.allow_rollback_from_cancelled:
                    return "rollback from cancelled actions is disabled"

            if not state.rollback_supported:
                return "action does not support rollback"

        return None

    def _next_state(
        self,
        *,
        state: ActionExecutionState,
        event: ActionExecutionEvent,
        status: ActionStatus,
    ) -> ActionExecutionState:
        now = utc_now()
        updates: dict[str, object] = {
            "status": status,
            "updated_at": now,
        }

        if status == ActionStatus.RUNNING:
            updates["pause_requested"] = False
            updates["cancellation_requested"] = False

            if state.started_at is None:
                updates["started_at"] = now

        if event.kind == ActionExecutionEventKind.PAUSE_REQUESTED:
            updates["pause_requested"] = True

        if event.kind == ActionExecutionEventKind.CANCEL_REQUESTED:
            updates["cancellation_requested"] = True

        if event.kind == ActionExecutionEventKind.RETRY_REQUESTED:
            updates["attempt"] = state.attempt + 1
            updates["current_step_index"] = 0
            updates["started_at"] = None
            updates["completed_at"] = None
            updates["pause_requested"] = False
            updates["cancellation_requested"] = False

        if status in {
            ActionStatus.SUCCEEDED,
            ActionStatus.FAILED,
            ActionStatus.CANCELLED,
            ActionStatus.BLOCKED,
            ActionStatus.ROLLED_BACK,
        }:
            updates["completed_at"] = now

        return state.model_copy(update=updates)

    def _rejected(
        self,
        *,
        state: ActionExecutionState,
        event: ActionExecutionEvent,
        reason: str,
    ) -> ActionExecutionTransition:
        return ActionExecutionTransition(
            event=event,
            previous_state=state,
            next_state=state,
            disposition=ActionExecutionDisposition.REJECTED,
            changed=False,
            reason=reason,
        )

    def _ignored(
        self,
        *,
        state: ActionExecutionState,
        event: ActionExecutionEvent,
        reason: str,
    ) -> ActionExecutionTransition:
        return ActionExecutionTransition(
            event=event,
            previous_state=state,
            next_state=state,
            disposition=ActionExecutionDisposition.IGNORED,
            changed=False,
            reason=reason,
        )

    def _terminal_transition(
        self,
        *,
        state: ActionExecutionState,
        event: ActionExecutionEvent,
    ) -> ActionExecutionTransition:
        if (
            event.kind == ActionExecutionEventKind.RETRY_REQUESTED
            and state.status == ActionStatus.FAILED
        ):
            guard = self._guard(state=state, event=event)

            if guard is not None:
                return self._rejected(state=state, event=event, reason=guard)

            next_state = self._next_state(
                state=state,
                event=event,
                status=ActionStatus.PLANNED,
            )

            return ActionExecutionTransition(
                event=event,
                previous_state=state,
                next_state=next_state,
                disposition=ActionExecutionDisposition.APPLIED,
                changed=next_state != state,
                reason="failed action moved back to planned for retry",
            )

        if (
            event.kind == ActionExecutionEventKind.ROLLBACK_COMPLETED
            and state.status
            in {
                ActionStatus.FAILED,
                ActionStatus.BLOCKED,
                ActionStatus.CANCELLED,
            }
        ):
            guard = self._guard(state=state, event=event)

            if guard is not None:
                return self._rejected(state=state, event=event, reason=guard)

            next_state = self._next_state(
                state=state,
                event=event,
                status=ActionStatus.ROLLED_BACK,
            )

            return ActionExecutionTransition(
                event=event,
                previous_state=state,
                next_state=next_state,
                disposition=ActionExecutionDisposition.APPLIED,
                changed=next_state != state,
                reason="terminal action rolled back",
            )

        return self._ignored(
            state=state,
            event=event,
            reason="terminal action execution state cannot transition",
        )

    def _record_transition(self, transition: ActionExecutionTransition) -> None:
        with self._lock:
            self._last_status = transition.next_state.status

            if transition.disposition == ActionExecutionDisposition.APPLIED:
                self._applied_count += 1

            elif transition.disposition == ActionExecutionDisposition.REJECTED:
                self._rejected_count += 1

            else:
                self._ignored_count += 1

            if transition.next_state.terminal:
                self._terminal_count += 1