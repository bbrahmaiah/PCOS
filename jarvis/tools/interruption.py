from __future__ import annotations

import shutil
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from threading import Event, RLock
from typing import Protocol

from pydantic import Field, field_validator, model_validator

from jarvis.tools.ids import new_action_id, new_action_result_id, utc_now
from jarvis.tools.models import (
    ActionRisk,
    ToolModel,
)


class ActionInterruptKind(StrEnum):
    """
    Supported action interruption kinds.
    """

    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"
    TIMEOUT = "timeout"


class ActionInterruptReason(StrEnum):
    """
    Machine-readable interruption reason.
    """

    USER_REQUESTED = "user_requested"
    TIMEOUT_EXCEEDED = "timeout_exceeded"
    SAFETY_POLICY = "safety_policy"
    SYSTEM_SHUTDOWN = "system_shutdown"
    HIGHER_PRIORITY_ACTION = "higher_priority_action"
    UNKNOWN = "unknown"


class ActionInterruptDecision(StrEnum):
    """
    Interrupt controller decision.
    """

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    IGNORED = "ignored"


class ActionCancellationState(StrEnum):
    """
    Cancellation token state.
    """

    ACTIVE = "active"
    PAUSE_REQUESTED = "pause_requested"
    PAUSED = "paused"
    RESUME_REQUESTED = "resume_requested"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    COMPLETED = "completed"


class RollbackStepKind(StrEnum):
    """
    Supported rollback step kinds.
    """

    RESTORE_BACKUP = "restore_backup"
    DELETE_CREATED_FILE = "delete_created_file"
    MOVE_BACK = "move_back"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"
    NOOP = "noop"


class RollbackStatus(StrEnum):
    """
    Rollback status.
    """

    NOT_REQUIRED = "not_required"
    AVAILABLE = "available"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PARTIAL = "partial"
    UNSAFE = "unsafe"


class RollbackReason(StrEnum):
    """
    Machine-readable rollback reason.
    """

    BACKUP_RESTORED = "backup_restored"
    CREATED_FILE_REMOVED = "created_file_removed"
    FILE_MOVED_BACK = "file_moved_back"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"
    NO_ROLLBACK_NEEDED = "no_rollback_needed"
    ROLLBACK_UNAVAILABLE = "rollback_unavailable"
    ROLLBACK_FAILED = "rollback_failed"
    ROLLBACK_SUCCEEDED = "rollback_succeeded"


class ActionCancellationToken(ToolModel):
    """
    Cancellation token for one action.

    Long-running runtimes should check this token cooperatively. The token
    does not kill work by magic; it provides a safe shared cancellation signal.
    """

    token_id: str = Field(default_factory=new_action_result_id)
    action_id: str = Field(default_factory=new_action_id)
    state: ActionCancellationState = ActionCancellationState.ACTIVE
    reason: ActionInterruptReason = ActionInterruptReason.UNKNOWN
    interruptible: bool = True
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("token_id", "action_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @property
    def pause_requested(self) -> bool:
        return self.state == ActionCancellationState.PAUSE_REQUESTED

    @property
    def cancellation_requested(self) -> bool:
        return self.state in {
            ActionCancellationState.CANCEL_REQUESTED,
            ActionCancellationState.CANCELLED,
            ActionCancellationState.TIMED_OUT,
        }

    @property
    def terminal(self) -> bool:
        return self.state in {
            ActionCancellationState.CANCELLED,
            ActionCancellationState.TIMED_OUT,
            ActionCancellationState.COMPLETED,
        }


class ActionInterruptRequest(ToolModel):
    """
    Request to interrupt an action.
    """

    action_id: str
    kind: ActionInterruptKind
    reason: ActionInterruptReason = ActionInterruptReason.USER_REQUESTED
    requested_by: str = "user"
    force: bool = False
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("action_id", "requested_by")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class ActionInterruptResult(ToolModel):
    """
    Observable result of an interrupt request.
    """

    result_id: str = Field(default_factory=new_action_result_id)
    action_id: str
    decision: ActionInterruptDecision
    kind: ActionInterruptKind
    previous_state: ActionCancellationState | None = None
    next_state: ActionCancellationState | None = None
    reason: ActionInterruptReason
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("result_id", "action_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class RollbackStep(ToolModel):
    """
    One rollback step.

    Rollback is explicit and observable. A mutating action should produce a
    rollback plan or explain why rollback is unsafe/impossible.
    """

    step_id: str = Field(default_factory=new_action_result_id)
    kind: RollbackStepKind
    description: str
    target_path: str | None = None
    backup_path: str | None = None
    destination_path: str | None = None
    required: bool = True
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("step_id", "description")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("target_path", "backup_path", "destination_path")
    @classmethod
    def _clean_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None

    @model_validator(mode="after")
    def _validate_paths(self) -> RollbackStep:
        if self.kind == RollbackStepKind.RESTORE_BACKUP:
            if self.target_path is None or self.backup_path is None:
                raise ValueError("restore backup requires target_path and backup_path.")

        if self.kind == RollbackStepKind.DELETE_CREATED_FILE:
            if self.target_path is None:
                raise ValueError("delete created file requires target_path.")

        if self.kind == RollbackStepKind.MOVE_BACK:
            if self.target_path is None or self.destination_path is None:
                raise ValueError("move back requires target_path and destination_path.")

        return self


class RollbackPlan(ToolModel):
    """
    Rollback plan for a mutating action.
    """

    rollback_id: str = Field(default_factory=new_action_result_id)
    action_id: str
    status: RollbackStatus = RollbackStatus.AVAILABLE
    risk: ActionRisk = ActionRisk.MEDIUM
    steps: tuple[RollbackStep, ...] = ()
    explanation: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("rollback_id", "action_id", "explanation")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @model_validator(mode="after")
    def _validate_plan(self) -> RollbackPlan:
        if self.status == RollbackStatus.AVAILABLE and not self.steps:
            raise ValueError("available rollback plans require at least one step.")

        if self.status == RollbackStatus.UNSAFE and self.steps:
            raise ValueError("unsafe rollback plans must not include executable steps.")

        return self

    @property
    def rollback_supported(self) -> bool:
        return self.status == RollbackStatus.AVAILABLE and bool(self.steps)


class RollbackStepResult(ToolModel):
    """
    Result of one rollback step.
    """

    result_id: str = Field(default_factory=new_action_result_id)
    step_id: str
    kind: RollbackStepKind
    status: RollbackStatus
    reason: RollbackReason
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("result_id", "step_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class RollbackResult(ToolModel):
    """
    Final rollback result.
    """

    result_id: str = Field(default_factory=new_action_result_id)
    rollback_id: str
    action_id: str
    status: RollbackStatus
    reason: RollbackReason
    success: bool
    step_results: tuple[RollbackStepResult, ...] = ()
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("result_id", "rollback_id", "action_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class ActionInterruptControllerConfig:
    """
    Configuration for ActionInterruptController.
    """

    name: str = "action_interrupt_controller"

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")


@dataclass(frozen=True, slots=True)
class ActionInterruptControllerSnapshot:
    """
    Observable diagnostics for ActionInterruptController.
    """

    name: str
    token_count: int
    interrupt_count: int
    rollback_plan_count: int
    rollback_count: int
    last_interrupt_kind: ActionInterruptKind | None
    last_rollback_status: RollbackStatus | None
    last_error: str | None


class RollbackExecutor(Protocol):
    """
    Rollback executor protocol.

    Production uses FileRollbackExecutor. Tests can inject fake executors.
    """

    def execute(self, plan: RollbackPlan) -> RollbackResult:
        ...


class FileRollbackExecutor:
    """
    Workspace-bounded file rollback executor.

    This executor supports restoring backups, deleting created files, and moving
    files back. It does not execute shell commands.
    """

    def __init__(self, *, workspace_root: str = ".") -> None:
        self._workspace_root = Path(workspace_root).resolve()

    def execute(self, plan: RollbackPlan) -> RollbackResult:
        if not plan.rollback_supported:
            return RollbackResult(
                rollback_id=plan.rollback_id,
                action_id=plan.action_id,
                status=RollbackStatus.UNSAFE,
                reason=RollbackReason.ROLLBACK_UNAVAILABLE,
                success=False,
                message=plan.explanation,
            )

        results: list[RollbackStepResult] = []

        for step in plan.steps:
            results.append(self._execute_step(step))

        successful_statuses = {
            RollbackStatus.SUCCEEDED,
            RollbackStatus.NOT_REQUIRED,
        }

        failed = [
            result
            for result in results
            if result.status not in successful_statuses
        ]

        if not failed:
            return RollbackResult(
                rollback_id=plan.rollback_id,
                action_id=plan.action_id,
                status=RollbackStatus.SUCCEEDED,
                reason=RollbackReason.ROLLBACK_SUCCEEDED,
                success=True,
                step_results=tuple(results),
                message="rollback completed successfully",
            )

        if len(failed) == len(results):
            status = RollbackStatus.FAILED
        else:
            status = RollbackStatus.PARTIAL

        return RollbackResult(
            rollback_id=plan.rollback_id,
            action_id=plan.action_id,
            status=status,
            reason=RollbackReason.ROLLBACK_FAILED,
            success=False,
            step_results=tuple(results),
            message="rollback failed or completed partially",
        )

    def _execute_step(self, step: RollbackStep) -> RollbackStepResult:
        try:
            if step.kind == RollbackStepKind.NOOP:
                return self._step_result(
                    step=step,
                    status=RollbackStatus.NOT_REQUIRED,
                    reason=RollbackReason.NO_ROLLBACK_NEEDED,
                    message="no rollback required",
                )

            if step.kind == RollbackStepKind.MANUAL_REVIEW_REQUIRED:
                return self._step_result(
                    step=step,
                    status=RollbackStatus.UNSAFE,
                    reason=RollbackReason.MANUAL_REVIEW_REQUIRED,
                    message="manual review required",
                )

            if step.kind == RollbackStepKind.RESTORE_BACKUP:
                return self._restore_backup(step)

            if step.kind == RollbackStepKind.DELETE_CREATED_FILE:
                return self._delete_created_file(step)

            if step.kind == RollbackStepKind.MOVE_BACK:
                return self._move_back(step)

            return self._step_result(
                step=step,
                status=RollbackStatus.FAILED,
                reason=RollbackReason.ROLLBACK_FAILED,
                message="unsupported rollback step",
            )

        except Exception as exc:
            return self._step_result(
                step=step,
                status=RollbackStatus.FAILED,
                reason=RollbackReason.ROLLBACK_FAILED,
                message=f"{type(exc).__name__}: {exc}",
            )

    def _restore_backup(self, step: RollbackStep) -> RollbackStepResult:
        target = self._resolve(step.target_path)
        backup = self._resolve(step.backup_path)

        if not backup.is_file():
            return self._step_result(
                step=step,
                status=RollbackStatus.FAILED,
                reason=RollbackReason.ROLLBACK_FAILED,
                message="backup file not found",
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup, target)

        return self._step_result(
            step=step,
            status=RollbackStatus.SUCCEEDED,
            reason=RollbackReason.BACKUP_RESTORED,
            message="backup restored",
        )

    def _delete_created_file(self, step: RollbackStep) -> RollbackStepResult:
        target = self._resolve(step.target_path)

        if not target.exists():
            return self._step_result(
                step=step,
                status=RollbackStatus.NOT_REQUIRED,
                reason=RollbackReason.NO_ROLLBACK_NEEDED,
                message="created file already absent",
            )

        if not target.is_file():
            return self._step_result(
                step=step,
                status=RollbackStatus.FAILED,
                reason=RollbackReason.ROLLBACK_FAILED,
                message="target is not a file",
            )

        target.unlink()

        return self._step_result(
            step=step,
            status=RollbackStatus.SUCCEEDED,
            reason=RollbackReason.CREATED_FILE_REMOVED,
            message="created file removed",
        )

    def _move_back(self, step: RollbackStep) -> RollbackStepResult:
        source = self._resolve(step.target_path)
        destination = self._resolve(step.destination_path)

        if not source.exists():
            return self._step_result(
                step=step,
                status=RollbackStatus.FAILED,
                reason=RollbackReason.ROLLBACK_FAILED,
                message="move-back source missing",
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))

        return self._step_result(
            step=step,
            status=RollbackStatus.SUCCEEDED,
            reason=RollbackReason.FILE_MOVED_BACK,
            message="file moved back",
        )

    def _resolve(self, value: str | None) -> Path:
        if value is None:
            raise ValueError("path is required.")

        raw = value.strip()

        if not raw:
            raise ValueError("path cannot be empty.")

        candidate = (self._workspace_root / raw).resolve()

        if not self._is_within_root(candidate):
            raise ValueError("rollback path must stay inside workspace root.")

        return candidate

    def _is_within_root(self, path: Path) -> bool:
        return path == self._workspace_root or self._workspace_root in path.parents

    @staticmethod
    def _step_result(
        *,
        step: RollbackStep,
        status: RollbackStatus,
        reason: RollbackReason,
        message: str,
    ) -> RollbackStepResult:
        return RollbackStepResult(
            step_id=step.step_id,
            kind=step.kind,
            status=status,
            reason=reason,
            message=message,
        )


class ActionInterruptController:
    """
    Central action interruption and rollback controller.

    Responsibilities:
    - create and track cancellation tokens
    - request pause/resume/cancel/timeout
    - expose cancellation Event objects for runtimes that need them
    - register rollback plans
    - execute rollback through a bounded rollback executor

    Non-responsibilities:
    - no direct shell execution
    - no hidden file mutation outside rollback executor
    - no policy bypass
    - no magical preemption of unsafe non-cooperative actions
    """

    def __init__(
        self,
        *,
        config: ActionInterruptControllerConfig | None = None,
        rollback_executor: RollbackExecutor | None = None,
    ) -> None:
        self._config = config or ActionInterruptControllerConfig()
        self._config.validate()

        self._rollback_executor = rollback_executor or FileRollbackExecutor()
        self._lock = RLock()

        self._tokens: dict[str, ActionCancellationToken] = {}
        self._events: dict[str, Event] = {}
        self._rollback_plans: dict[str, RollbackPlan] = {}

        self._interrupt_count = 0
        self._rollback_count = 0
        self._last_interrupt_kind: ActionInterruptKind | None = None
        self._last_rollback_status: RollbackStatus | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def create_token(
        self,
        *,
        action_id: str | None = None,
        interruptible: bool = True,
        metadata: dict[str, object] | None = None,
    ) -> ActionCancellationToken:
        """
        Create a cancellation token for an action.
        """

        token = ActionCancellationToken(
            action_id=action_id or new_action_id(),
            interruptible=interruptible,
            metadata=metadata or {},
        )

        with self._lock:
            self._tokens[token.action_id] = token
            self._events[token.action_id] = Event()

        return token

    def get_token(self, action_id: str) -> ActionCancellationToken | None:
        """
        Return the current token for an action.
        """

        with self._lock:
            return self._tokens.get(action_id)

    def cancellation_event(self, action_id: str) -> Event | None:
        """
        Return the cancellation Event for runtimes such as SafeShellRuntime.
        """

        with self._lock:
            return self._events.get(action_id)

    def interrupt(self, request: ActionInterruptRequest) -> ActionInterruptResult:
        """
        Apply pause/resume/cancel/timeout request to an action token.
        """

        with self._lock:
            self._interrupt_count += 1
            self._last_interrupt_kind = request.kind
            self._last_error = None

            token = self._tokens.get(request.action_id)

            if token is None:
                return ActionInterruptResult(
                    action_id=request.action_id,
                    decision=ActionInterruptDecision.REJECTED,
                    kind=request.kind,
                    reason=request.reason,
                    message="action has no cancellation token",
                )

            if not token.interruptible and not request.force:
                return ActionInterruptResult(
                    action_id=request.action_id,
                    decision=ActionInterruptDecision.REJECTED,
                    kind=request.kind,
                    previous_state=token.state,
                    next_state=token.state,
                    reason=request.reason,
                    message="action is not interruptible",
                )

            if token.terminal:
                return ActionInterruptResult(
                    action_id=request.action_id,
                    decision=ActionInterruptDecision.IGNORED,
                    kind=request.kind,
                    previous_state=token.state,
                    next_state=token.state,
                    reason=request.reason,
                    message="action token is already terminal",
                )

            previous = token.state
            next_state = self._next_state(token.state, request.kind, request.reason)
            updated = token.model_copy(
                update={
                    "state": next_state,
                    "reason": request.reason,
                    "updated_at": utc_now(),
                }
            )
            self._tokens[request.action_id] = updated

            event = self._events.get(request.action_id)

            if event is not None and next_state in {
                ActionCancellationState.CANCEL_REQUESTED,
                ActionCancellationState.TIMED_OUT,
            }:
                event.set()

            return ActionInterruptResult(
                action_id=request.action_id,
                decision=ActionInterruptDecision.ACCEPTED,
                kind=request.kind,
                previous_state=previous,
                next_state=next_state,
                reason=request.reason,
                message=f"interrupt accepted: {request.kind.value}",
            )

    def mark_paused(self, action_id: str) -> ActionInterruptResult:
        """
        Mark action as paused after the runtime cooperatively stops work.
        """

        return self._mark_state(
            action_id=action_id,
            state=ActionCancellationState.PAUSED,
            kind=ActionInterruptKind.PAUSE,
            message="action marked paused",
        )

    def mark_completed(self, action_id: str) -> ActionInterruptResult:
        """
        Mark action as completed.
        """

        return self._mark_state(
            action_id=action_id,
            state=ActionCancellationState.COMPLETED,
            kind=ActionInterruptKind.RESUME,
            message="action marked completed",
        )

    def mark_cancelled(self, action_id: str) -> ActionInterruptResult:
        """
        Mark action as cancelled after runtime cleanup.
        """

        return self._mark_state(
            action_id=action_id,
            state=ActionCancellationState.CANCELLED,
            kind=ActionInterruptKind.CANCEL,
            message="action marked cancelled",
        )

    def register_rollback_plan(self, plan: RollbackPlan) -> None:
        """
        Register rollback plan for an action.
        """

        with self._lock:
            self._rollback_plans[plan.action_id] = plan

    def rollback_plan(self, action_id: str) -> RollbackPlan | None:
        """
        Return rollback plan for action.
        """

        with self._lock:
            return self._rollback_plans.get(action_id)

    def rollback(self, action_id: str) -> RollbackResult:
        """
        Execute rollback for an action if a safe plan exists.
        """

        with self._lock:
            self._rollback_count += 1
            self._last_error = None
            plan = self._rollback_plans.get(action_id)

        if plan is None:
            result = RollbackResult(
                rollback_id=new_action_result_id(),
                action_id=action_id,
                status=RollbackStatus.UNSAFE,
                reason=RollbackReason.ROLLBACK_UNAVAILABLE,
                success=False,
                message="no rollback plan registered",
            )

            with self._lock:
                self._last_rollback_status = result.status

            return result

        try:
            result = self._rollback_executor.execute(plan)

            with self._lock:
                self._last_rollback_status = result.status

            return result

        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._last_rollback_status = RollbackStatus.FAILED

            return RollbackResult(
                rollback_id=plan.rollback_id,
                action_id=action_id,
                status=RollbackStatus.FAILED,
                reason=RollbackReason.ROLLBACK_FAILED,
                success=False,
                message=f"{type(exc).__name__}: {exc}",
            )

    def snapshot(self) -> ActionInterruptControllerSnapshot:
        """
        Return controller diagnostics.
        """

        with self._lock:
            return ActionInterruptControllerSnapshot(
                name=self.name,
                token_count=len(self._tokens),
                interrupt_count=self._interrupt_count,
                rollback_plan_count=len(self._rollback_plans),
                rollback_count=self._rollback_count,
                last_interrupt_kind=self._last_interrupt_kind,
                last_rollback_status=self._last_rollback_status,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Reset controller state and diagnostics.
        """

        with self._lock:
            self._tokens.clear()
            self._events.clear()
            self._rollback_plans.clear()
            self._interrupt_count = 0
            self._rollback_count = 0
            self._last_interrupt_kind = None
            self._last_rollback_status = None
            self._last_error = None

    def _mark_state(
        self,
        *,
        action_id: str,
        state: ActionCancellationState,
        kind: ActionInterruptKind,
        message: str,
    ) -> ActionInterruptResult:
        with self._lock:
            token = self._tokens.get(action_id)

            if token is None:
                return ActionInterruptResult(
                    action_id=action_id,
                    decision=ActionInterruptDecision.REJECTED,
                    kind=kind,
                    reason=ActionInterruptReason.UNKNOWN,
                    message="action has no cancellation token",
                )

            previous = token.state
            updated = token.model_copy(
                update={
                    "state": state,
                    "updated_at": utc_now(),
                }
            )
            self._tokens[action_id] = updated

            return ActionInterruptResult(
                action_id=action_id,
                decision=ActionInterruptDecision.ACCEPTED,
                kind=kind,
                previous_state=previous,
                next_state=state,
                reason=token.reason,
                message=message,
            )

    @staticmethod
    def _next_state(
        current: ActionCancellationState,
        kind: ActionInterruptKind,
        reason: ActionInterruptReason,
    ) -> ActionCancellationState:
        del current

        if kind == ActionInterruptKind.PAUSE:
            return ActionCancellationState.PAUSE_REQUESTED

        if kind == ActionInterruptKind.RESUME:
            return ActionCancellationState.RESUME_REQUESTED

        if kind == ActionInterruptKind.CANCEL:
            return ActionCancellationState.CANCEL_REQUESTED

        if kind == ActionInterruptKind.TIMEOUT:
            if reason == ActionInterruptReason.TIMEOUT_EXCEEDED:
                return ActionCancellationState.TIMED_OUT

            return ActionCancellationState.CANCEL_REQUESTED

        return ActionCancellationState.CANCEL_REQUESTED