from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import RLock

from pydantic import Field, field_validator, model_validator

from jarvis.tools.ids import new_action_result_id, utc_now
from jarvis.tools.interruption import (
    ActionCancellationToken,
    ActionInterruptController,
)
from jarvis.tools.models import (
    ActionKind,
    ActionPlan,
    ActionRisk,
    ActionScope,
    ToolModel,
)


class ActionSchedulePriority(StrEnum):
    """
    Scheduler priority.

    Higher priority actions are dispatched first, but never bypass approval,
    validation, dependency, cancellation, or collision safety.
    """

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"


class ActionScheduleState(StrEnum):
    """
    Scheduled action lifecycle state.
    """

    QUEUED = "queued"
    WAITING_DEPENDENCIES = "waiting_dependencies"
    WAITING_APPROVAL = "waiting_approval"
    WAITING_VALIDATION = "waiting_validation"
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class ActionScheduleDecision(StrEnum):
    """
    Scheduler decision.
    """

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    DEFERRED = "deferred"
    READY = "ready"
    STARTED = "started"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class ActionScheduleReason(StrEnum):
    """
    Machine-readable scheduling reason.
    """

    ACTION_ACCEPTED = "action_accepted"
    ACTION_READY = "action_ready"
    ACTION_STARTED = "action_started"
    ACTION_COMPLETED = "action_completed"
    ACTION_FAILED = "action_failed"
    ACTION_CANCELLED = "action_cancelled"
    ACTION_BLOCKED = "action_blocked"
    WAITING_FOR_DEPENDENCIES = "waiting_for_dependencies"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    WAITING_FOR_VALIDATION = "waiting_for_validation"
    CONCURRENCY_LIMIT_REACHED = "concurrency_limit_reached"
    RESOURCE_COLLISION = "resource_collision"
    DEPENDENCY_FAILED = "dependency_failed"
    ACTION_NOT_FOUND = "action_not_found"
    INVALID_DEPENDENCY = "invalid_dependency"
    ALREADY_TERMINAL = "already_terminal"
    ALREADY_RUNNING = "already_running"


class ResourceLockKind(StrEnum):
    """
    Resource lock kind.

    Locks are descriptive scheduler contracts. They prevent unsafe parallel
    access to the same file, runtime, app, browser, shell, or scope.
    """

    FILE_PATH = "file_path"
    RUNTIME = "runtime"
    SCOPE = "scope"
    SHELL = "shell"
    BROWSER = "browser"
    IDE = "ide"
    WORKSPACE = "workspace"


class ResourceLockMode(StrEnum):
    """
    Lock mode.

    Multiple reads may run together. A write conflicts with every other lock
    on the same resource key.
    """

    READ = "read"
    WRITE = "write"
    EXCLUSIVE = "exclusive"


class ScheduledActionLock(ToolModel):
    """
    Lock required by a scheduled action.
    """

    kind: ResourceLockKind
    key: str
    mode: ResourceLockMode = ResourceLockMode.READ
    reason: str = "scheduler resource safety"

    @field_validator("key", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    def conflicts_with(self, other: ScheduledActionLock) -> bool:
        """
        Return True when two locks cannot run in parallel.
        """

        if self.kind != other.kind:
            return False

        if self.key != other.key:
            return False

        if self.mode == ResourceLockMode.READ and other.mode == ResourceLockMode.READ:
            return False

        return True


class ScheduledAction(ToolModel):
    """
    Scheduler-owned action record.

    This wraps an ActionPlan with orchestration metadata. It does not execute.
    """

    schedule_id: str = Field(default_factory=new_action_result_id)
    action_id: str
    plan: ActionPlan
    state: ActionScheduleState = ActionScheduleState.QUEUED
    priority: ActionSchedulePriority = ActionSchedulePriority.NORMAL
    dependencies: tuple[str, ...] = ()
    locks: tuple[ScheduledActionLock, ...] = ()
    approved: bool = False
    validated: bool = False
    cancellation_token: ActionCancellationToken | None = None
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("schedule_id", "action_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("dependencies")
    @classmethod
    def _clean_dependencies(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(item.strip() for item in value if item.strip())

    @model_validator(mode="after")
    def _validate_action_identity(self) -> ScheduledAction:
        if self.plan.action_id != self.action_id:
            raise ValueError("scheduled action_id must match plan.action_id.")

        return self

    @property
    def terminal(self) -> bool:
        return self.state in {
            ActionScheduleState.SUCCEEDED,
            ActionScheduleState.FAILED,
            ActionScheduleState.CANCELLED,
            ActionScheduleState.BLOCKED,
        }


class SchedulingResult(ToolModel):
    """
    Observable scheduler result.
    """

    result_id: str = Field(default_factory=new_action_result_id)
    action_id: str
    decision: ActionScheduleDecision
    reason: ActionScheduleReason
    previous_state: ActionScheduleState | None = None
    next_state: ActionScheduleState | None = None
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


@dataclass(frozen=True, slots=True)
class ParallelActionSchedulerConfig:
    """
    Configuration for ParallelActionScheduler.
    """

    name: str = "parallel_action_scheduler"
    max_parallel_actions: int = 3
    max_parallel_shell_actions: int = 1
    max_parallel_browser_actions: int = 1
    allow_parallel_mutations: bool = False
    require_validation_before_start: bool = True
    require_approval_for_high_risk: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.max_parallel_actions <= 0:
            raise ValueError("max_parallel_actions must be positive.")

        if self.max_parallel_shell_actions <= 0:
            raise ValueError("max_parallel_shell_actions must be positive.")

        if self.max_parallel_browser_actions <= 0:
            raise ValueError("max_parallel_browser_actions must be positive.")


@dataclass(frozen=True, slots=True)
class ParallelActionSchedulerSnapshot:
    """
    Scheduler diagnostics.
    """

    name: str
    queued_count: int
    ready_count: int
    running_count: int
    succeeded_count: int
    failed_count: int
    cancelled_count: int
    blocked_count: int
    total_count: int
    last_decision: ActionScheduleDecision | None
    last_reason: ActionScheduleReason | None
    last_error: str | None


class ParallelActionScheduler:
    """
    Governed parallel action scheduler.

    Responsibilities:
    - accept validated typed action plans
    - enforce dependencies
    - enforce approval gates
    - enforce validation gates
    - enforce concurrency limits
    - prevent resource collisions
    - create cancellation tokens
    - mark action lifecycle state

    Non-responsibilities:
    - no action execution
    - no shell/file/browser/IDE calls
    - no approval granting
    - no policy replacement
    - no validation replacement
    """

    def __init__(
        self,
        *,
        config: ParallelActionSchedulerConfig | None = None,
        interrupt_controller: ActionInterruptController | None = None,
    ) -> None:
        self._config = config or ParallelActionSchedulerConfig()
        self._config.validate()

        self._interrupt_controller = interrupt_controller or ActionInterruptController()
        self._lock = RLock()

        self._actions: dict[str, ScheduledAction] = {}
        self._last_decision: ActionScheduleDecision | None = None
        self._last_reason: ActionScheduleReason | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def submit(
        self,
        plan: ActionPlan,
        *,
        priority: ActionSchedulePriority = ActionSchedulePriority.NORMAL,
        dependencies: tuple[str, ...] = (),
        locks: tuple[ScheduledActionLock, ...] | None = None,
        approved: bool = False,
        validated: bool = False,
        metadata: dict[str, object] | None = None,
    ) -> SchedulingResult:
        """
        Submit an action plan to the scheduler.

        Submission does not execute the action.
        """

        with self._lock:
            if plan.action_id in self._actions:
                return self._result(
                    action_id=plan.action_id,
                    decision=ActionScheduleDecision.REJECTED,
                    reason=ActionScheduleReason.ACTION_BLOCKED,
                    message="action is already scheduled",
                )

            missing_dependencies = [
                item for item in dependencies if item not in self._actions
            ]

            if missing_dependencies:
                return self._result(
                    action_id=plan.action_id,
                    decision=ActionScheduleDecision.REJECTED,
                    reason=ActionScheduleReason.INVALID_DEPENDENCY,
                    message="scheduled action has unknown dependencies",
                    metadata={"missing_dependencies": tuple(missing_dependencies)},
                )

            token = self._interrupt_controller.create_token(
                action_id=plan.action_id,
                interruptible=True,
                metadata={"scheduler": self.name},
            )
            inferred_locks = locks or self._infer_locks(plan)
            initial_state = self._initial_state(
                plan=plan,
                dependencies=dependencies,
                approved=approved,
                validated=validated,
            )
            scheduled = ScheduledAction(
                action_id=plan.action_id,
                plan=plan,
                state=initial_state,
                priority=priority,
                dependencies=dependencies,
                locks=inferred_locks,
                approved=approved,
                validated=validated,
                cancellation_token=token,
                metadata=metadata or {},
            )
            self._actions[plan.action_id] = scheduled

            return self._result(
                action_id=plan.action_id,
                decision=ActionScheduleDecision.ACCEPTED,
                reason=ActionScheduleReason.ACTION_ACCEPTED,
                previous_state=None,
                next_state=initial_state,
                message="action accepted by scheduler",
            )

    def approve(self, action_id: str) -> SchedulingResult:
        """
        Mark a scheduled action as approved.
        """

        return self._update_gate(
            action_id=action_id,
            approved=True,
            validated=None,
            message="action approval recorded",
        )

    def validate(self, action_id: str) -> SchedulingResult:
        """
        Mark a scheduled action as validated.
        """

        return self._update_gate(
            action_id=action_id,
            approved=None,
            validated=True,
            message="action validation recorded",
        )

    def refresh_ready(self) -> tuple[ScheduledAction, ...]:
        """
        Recalculate readiness for all non-terminal, non-running actions.
        """

        with self._lock:
            refreshed: list[ScheduledAction] = []

            for action in tuple(self._actions.values()):
                if action.terminal or action.state == ActionScheduleState.RUNNING:
                    continue

                next_state = self._state_after_gates(action)
                updated = action.model_copy(
                    update={
                        "state": next_state,
                        "updated_at": utc_now(),
                    }
                )
                self._actions[action.action_id] = updated

                if next_state == ActionScheduleState.READY:
                    refreshed.append(updated)

            return tuple(refreshed)

    def start_ready(self) -> tuple[SchedulingResult, ...]:
        """
        Start as many ready actions as safely possible.

        This only changes scheduler state to RUNNING. A separate executor must
        perform the actual runtime work.
        """

        with self._lock:
            self.refresh_ready()
            results: list[SchedulingResult] = []

            candidates = sorted(
                (
                    action
                    for action in self._actions.values()
                    if action.state == ActionScheduleState.READY
                ),
                key=self._sort_key,
            )

            for action in candidates:
                result = self._start_locked(action.action_id)
                results.append(result)

            return tuple(results)

    def start_action(self, action_id: str) -> SchedulingResult:
        """
        Start one action if it is ready and collision-free.
        """

        with self._lock:
            self.refresh_ready()

            return self._start_locked(action_id)

    def complete(
        self,
        action_id: str,
        *,
        success: bool = True,
        message: str | None = None,
    ) -> SchedulingResult:
        """
        Mark a running action as succeeded or failed.
        """

        next_state = (
            ActionScheduleState.SUCCEEDED
            if success
            else ActionScheduleState.FAILED
        )
        decision = (
            ActionScheduleDecision.COMPLETED
            if success
            else ActionScheduleDecision.BLOCKED
        )
        reason = (
            ActionScheduleReason.ACTION_COMPLETED
            if success
            else ActionScheduleReason.ACTION_FAILED
        )

        with self._lock:
            action = self._actions.get(action_id)

            if action is None:
                return self._not_found(action_id)

            if action.terminal:
                return self._terminal_result(action)

            previous = action.state
            updated = action.model_copy(
                update={
                    "state": next_state,
                    "updated_at": utc_now(),
                }
            )
            self._actions[action_id] = updated
            self._interrupt_controller.mark_completed(action_id)

            return self._result(
                action_id=action_id,
                decision=decision,
                reason=reason,
                previous_state=previous,
                next_state=next_state,
                message=message or f"action marked {next_state.value}",
            )

    def cancel(
        self,
        action_id: str,
        *,
        message: str = "action cancelled by scheduler",
    ) -> SchedulingResult:
        """
        Cancel an action and notify the interrupt controller.
        """

        with self._lock:
            action = self._actions.get(action_id)

            if action is None:
                return self._not_found(action_id)

            if action.terminal:
                return self._terminal_result(action)

            previous = action.state
            updated = action.model_copy(
                update={
                    "state": ActionScheduleState.CANCELLED,
                    "updated_at": utc_now(),
                }
            )
            self._actions[action_id] = updated
            self._interrupt_controller.mark_cancelled(action_id)

            return self._result(
                action_id=action_id,
                decision=ActionScheduleDecision.CANCELLED,
                reason=ActionScheduleReason.ACTION_CANCELLED,
                previous_state=previous,
                next_state=ActionScheduleState.CANCELLED,
                message=message,
            )

    def block(
        self,
        action_id: str,
        *,
        message: str = "action blocked by scheduler",
    ) -> SchedulingResult:
        """
        Mark an action as blocked.
        """

        with self._lock:
            action = self._actions.get(action_id)

            if action is None:
                return self._not_found(action_id)

            if action.terminal:
                return self._terminal_result(action)

            previous = action.state
            updated = action.model_copy(
                update={
                    "state": ActionScheduleState.BLOCKED,
                    "updated_at": utc_now(),
                }
            )
            self._actions[action_id] = updated

            return self._result(
                action_id=action_id,
                decision=ActionScheduleDecision.BLOCKED,
                reason=ActionScheduleReason.ACTION_BLOCKED,
                previous_state=previous,
                next_state=ActionScheduleState.BLOCKED,
                message=message,
            )

    def scheduled_action(self, action_id: str) -> ScheduledAction | None:
        """
        Return one scheduled action.
        """

        with self._lock:
            return self._actions.get(action_id)

    def running_actions(self) -> tuple[ScheduledAction, ...]:
        """
        Return running actions.
        """

        with self._lock:
            return tuple(
                action
                for action in self._actions.values()
                if action.state == ActionScheduleState.RUNNING
            )

    def all_actions(self) -> tuple[ScheduledAction, ...]:
        """
        Return all scheduled actions.
        """

        with self._lock:
            return tuple(self._actions.values())

    def snapshot(self) -> ParallelActionSchedulerSnapshot:
        """
        Return scheduler diagnostics.
        """

        with self._lock:
            states = [action.state for action in self._actions.values()]

            return ParallelActionSchedulerSnapshot(
                name=self.name,
                queued_count=states.count(ActionScheduleState.QUEUED),
                ready_count=states.count(ActionScheduleState.READY),
                running_count=states.count(ActionScheduleState.RUNNING),
                succeeded_count=states.count(ActionScheduleState.SUCCEEDED),
                failed_count=states.count(ActionScheduleState.FAILED),
                cancelled_count=states.count(ActionScheduleState.CANCELLED),
                blocked_count=states.count(ActionScheduleState.BLOCKED),
                total_count=len(states),
                last_decision=self._last_decision,
                last_reason=self._last_reason,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Reset scheduler state.
        """

        with self._lock:
            self._actions.clear()
            self._last_decision = None
            self._last_reason = None
            self._last_error = None

    def _update_gate(
        self,
        *,
        action_id: str,
        approved: bool | None,
        validated: bool | None,
        message: str,
    ) -> SchedulingResult:
        with self._lock:
            action = self._actions.get(action_id)

            if action is None:
                return self._not_found(action_id)

            if action.terminal:
                return self._terminal_result(action)

            previous = action.state
            updated = action.model_copy(
                update={
                    "approved": action.approved if approved is None else approved,
                    "validated": (
                        action.validated if validated is None else validated
                    ),
                    "updated_at": utc_now(),
                }
            )
            next_state = self._state_after_gates(updated)
            updated = updated.model_copy(
                update={
                    "state": next_state,
                    "updated_at": utc_now(),
                }
            )
            self._actions[action_id] = updated

            return self._result(
                action_id=action_id,
                decision=(
                    ActionScheduleDecision.READY
                    if next_state == ActionScheduleState.READY
                    else ActionScheduleDecision.DEFERRED
                ),
                reason=self._reason_for_state(next_state),
                previous_state=previous,
                next_state=next_state,
                message=message,
            )

    def _start_locked(self, action_id: str) -> SchedulingResult:
        action = self._actions.get(action_id)

        if action is None:
            return self._not_found(action_id)

        if action.terminal:
            return self._terminal_result(action)

        if action.state == ActionScheduleState.RUNNING:
            return self._result(
                action_id=action_id,
                decision=ActionScheduleDecision.DEFERRED,
                reason=ActionScheduleReason.ALREADY_RUNNING,
                previous_state=action.state,
                next_state=action.state,
                message="action is already running",
            )

        gate_state = self._state_after_gates(action)

        if gate_state != ActionScheduleState.READY:
            updated = action.model_copy(
                update={
                    "state": gate_state,
                    "updated_at": utc_now(),
                }
            )
            self._actions[action_id] = updated

            return self._result(
                action_id=action_id,
                decision=ActionScheduleDecision.DEFERRED,
                reason=self._reason_for_state(gate_state),
                previous_state=action.state,
                next_state=gate_state,
                message="action is not ready to start",
            )

        collision_reason = self._collision_reason(action)

        if collision_reason is not None:
            return self._result(
                action_id=action_id,
                decision=ActionScheduleDecision.DEFERRED,
                reason=collision_reason,
                previous_state=action.state,
                next_state=action.state,
                message="action deferred by scheduler safety limits",
            )

        updated = action.model_copy(
            update={
                "state": ActionScheduleState.RUNNING,
                "updated_at": utc_now(),
            }
        )
        self._actions[action_id] = updated

        return self._result(
            action_id=action_id,
            decision=ActionScheduleDecision.STARTED,
            reason=ActionScheduleReason.ACTION_STARTED,
            previous_state=action.state,
            next_state=ActionScheduleState.RUNNING,
            message="action started by scheduler",
        )

    def _state_after_gates(self, action: ScheduledAction) -> ActionScheduleState:
        failed_dependency = self._failed_dependency(action)

        if failed_dependency is not None:
            return ActionScheduleState.BLOCKED

        if not self._dependencies_satisfied(action):
            return ActionScheduleState.WAITING_DEPENDENCIES

        if self._approval_required(action) and not action.approved:
            return ActionScheduleState.WAITING_APPROVAL

        if self._config.require_validation_before_start and not action.validated:
            return ActionScheduleState.WAITING_VALIDATION

        return ActionScheduleState.READY

    def _dependencies_satisfied(self, action: ScheduledAction) -> bool:
        for dependency in action.dependencies:
            dependency_action = self._actions.get(dependency)

            if dependency_action is None:
                return False

            if dependency_action.state != ActionScheduleState.SUCCEEDED:
                return False

        return True

    def _failed_dependency(self, action: ScheduledAction) -> str | None:
        for dependency in action.dependencies:
            dependency_action = self._actions.get(dependency)

            if dependency_action is None:
                continue

            if dependency_action.state in {
                ActionScheduleState.FAILED,
                ActionScheduleState.CANCELLED,
                ActionScheduleState.BLOCKED,
            }:
                return dependency

        return None

    def _approval_required(self, action: ScheduledAction) -> bool:
        if action.plan.requires_approval:
            return True

        if not self._config.require_approval_for_high_risk:
            return False

        return action.plan.risk in {ActionRisk.HIGH, ActionRisk.CRITICAL}

    def _collision_reason(
        self,
        action: ScheduledAction,
    ) -> ActionScheduleReason | None:
        running = self.running_actions()

        if len(running) >= self._config.max_parallel_actions:
            return ActionScheduleReason.CONCURRENCY_LIMIT_REACHED

        if self._scope_running_count(ActionScope.SHELL) >= (
            self._config.max_parallel_shell_actions
        ):
            if action.plan.scope == ActionScope.SHELL:
                return ActionScheduleReason.CONCURRENCY_LIMIT_REACHED

        if self._scope_running_count(ActionScope.BROWSER) >= (
            self._config.max_parallel_browser_actions
        ):
            if action.plan.scope == ActionScope.BROWSER:
                return ActionScheduleReason.CONCURRENCY_LIMIT_REACHED

        for running_action in running:
            if self._locks_conflict(action, running_action):
                return ActionScheduleReason.RESOURCE_COLLISION

        if not self._config.allow_parallel_mutations:
            if self._is_mutating(action) and any(
                self._is_mutating(running_action) for running_action in running
            ):
                return ActionScheduleReason.RESOURCE_COLLISION

        return None

    def _scope_running_count(self, scope: ActionScope) -> int:
        return sum(
            1
            for action in self._actions.values()
            if action.state == ActionScheduleState.RUNNING
            and action.plan.scope == scope
        )

    @staticmethod
    def _locks_conflict(
        left: ScheduledAction,
        right: ScheduledAction,
    ) -> bool:
        for left_lock in left.locks:
            for right_lock in right.locks:
                if left_lock.conflicts_with(right_lock):
                    return True

        return False

    @staticmethod
    def _is_mutating(action: ScheduledAction) -> bool:
        mutating_kinds = {
            ActionKind.WRITE,
            ActionKind.PATCH,
            ActionKind.COPY,
            ActionKind.MOVE,
            ActionKind.DELETE,
            ActionKind.IDE_APPLY_PATCH,
            ActionKind.SHELL_COMMAND,
        }

        return any(step.kind in mutating_kinds for step in action.plan.steps)

    @staticmethod
    def _infer_locks(plan: ActionPlan) -> tuple[ScheduledActionLock, ...]:
        locks: list[ScheduledActionLock] = [
            ScheduledActionLock(
                kind=ResourceLockKind.SCOPE,
                key=plan.scope.value,
                mode=ResourceLockMode.READ,
                reason="action scope lock",
            )
        ]

        for step in plan.steps:
            path = step.arguments.get("path")
            command = step.arguments.get("command")

            if isinstance(path, str) and path.strip():
                locks.append(
                    ScheduledActionLock(
                        kind=ResourceLockKind.FILE_PATH,
                        key=path.strip().replace("\\", "/"),
                        mode=(
                            ResourceLockMode.WRITE
                            if step.kind
                            in {
                                ActionKind.WRITE,
                                ActionKind.PATCH,
                                ActionKind.COPY,
                                ActionKind.MOVE,
                                ActionKind.DELETE,
                                ActionKind.IDE_APPLY_PATCH,
                            }
                            else ResourceLockMode.READ
                        ),
                        reason=f"path lock for {step.kind.value}",
                    )
                )

            if step.scope == ActionScope.SHELL or command is not None:
                locks.append(
                    ScheduledActionLock(
                        kind=ResourceLockKind.SHELL,
                        key="default",
                        mode=ResourceLockMode.EXCLUSIVE,
                        reason="shell actions run one at a time by default",
                    )
                )

            if step.scope == ActionScope.BROWSER:
                locks.append(
                    ScheduledActionLock(
                        kind=ResourceLockKind.BROWSER,
                        key="default",
                        mode=ResourceLockMode.EXCLUSIVE,
                        reason="browser actions are serialized by default",
                    )
                )

            if step.scope == ActionScope.IDE:
                locks.append(
                    ScheduledActionLock(
                        kind=ResourceLockKind.IDE,
                        key="default",
                        mode=ResourceLockMode.WRITE
                        if step.kind == ActionKind.IDE_APPLY_PATCH
                        else ResourceLockMode.READ,
                        reason="IDE context lock",
                    )
                )

        return tuple(locks)

    def _initial_state(
        self,
        *,
        plan: ActionPlan,
        dependencies: tuple[str, ...],
        approved: bool,
        validated: bool,
    ) -> ActionScheduleState:
        ghost_action = ScheduledAction.model_construct(
            action_id=plan.action_id,
            plan=plan,
            state=ActionScheduleState.QUEUED,
            priority=ActionSchedulePriority.NORMAL,
            dependencies=dependencies,
            locks=(),
            approved=approved,
            validated=validated,
            cancellation_token=None,
            metadata={},
        )

        return self._state_after_gates(ghost_action)

    @staticmethod
    def _priority_rank(priority: ActionSchedulePriority) -> int:
        return {
            ActionSchedulePriority.CRITICAL: 0,
            ActionSchedulePriority.HIGH: 1,
            ActionSchedulePriority.NORMAL: 2,
            ActionSchedulePriority.LOW: 3,
            ActionSchedulePriority.BACKGROUND: 4,
        }[priority]

    def _sort_key(self, action: ScheduledAction) -> tuple[int, str]:
        return (
            self._priority_rank(action.priority),
            str(action.created_at),
        )

    @staticmethod
    def _reason_for_state(state: ActionScheduleState) -> ActionScheduleReason:
        return {
            ActionScheduleState.WAITING_DEPENDENCIES: (
                ActionScheduleReason.WAITING_FOR_DEPENDENCIES
            ),
            ActionScheduleState.WAITING_APPROVAL: (
                ActionScheduleReason.WAITING_FOR_APPROVAL
            ),
            ActionScheduleState.WAITING_VALIDATION: (
                ActionScheduleReason.WAITING_FOR_VALIDATION
            ),
            ActionScheduleState.READY: ActionScheduleReason.ACTION_READY,
            ActionScheduleState.BLOCKED: ActionScheduleReason.DEPENDENCY_FAILED,
            ActionScheduleState.QUEUED: ActionScheduleReason.ACTION_ACCEPTED,
            ActionScheduleState.RUNNING: ActionScheduleReason.ALREADY_RUNNING,
            ActionScheduleState.SUCCEEDED: ActionScheduleReason.ACTION_COMPLETED,
            ActionScheduleState.FAILED: ActionScheduleReason.ACTION_FAILED,
            ActionScheduleState.CANCELLED: ActionScheduleReason.ACTION_CANCELLED,
        }[state]

    def _not_found(self, action_id: str) -> SchedulingResult:
        return self._result(
            action_id=action_id,
            decision=ActionScheduleDecision.REJECTED,
            reason=ActionScheduleReason.ACTION_NOT_FOUND,
            message="scheduled action not found",
        )

    def _terminal_result(self, action: ScheduledAction) -> SchedulingResult:
        return self._result(
            action_id=action.action_id,
            decision=ActionScheduleDecision.IGNORED
            if hasattr(ActionScheduleDecision, "IGNORED")
            else ActionScheduleDecision.DEFERRED,
            reason=ActionScheduleReason.ALREADY_TERMINAL,
            previous_state=action.state,
            next_state=action.state,
            message="action is already terminal",
        )

    def _result(
        self,
        *,
        action_id: str,
        decision: ActionScheduleDecision,
        reason: ActionScheduleReason,
        message: str,
        previous_state: ActionScheduleState | None = None,
        next_state: ActionScheduleState | None = None,
        metadata: dict[str, object] | None = None,
    ) -> SchedulingResult:
        self._last_decision = decision
        self._last_reason = reason

        return SchedulingResult(
            action_id=action_id,
            decision=decision,
            reason=reason,
            previous_state=previous_state,
            next_state=next_state,
            message=message,
            metadata=metadata or {},
        )