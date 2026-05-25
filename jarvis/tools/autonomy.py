from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from typing import Protocol

from pydantic import Field, field_validator, model_validator

from jarvis.tools.audit import (
    ActionAuditActor,
    ActionAuditEventKind,
    ActionAuditLog,
    ActionAuditOutcome,
)
from jarvis.tools.ids import new_action_result_id, utc_now
from jarvis.tools.interruption import (
    ActionInterruptController,
    ActionInterruptKind,
    ActionInterruptReason,
    ActionInterruptRequest,
)
from jarvis.tools.models import ActionRisk, ActionStatus, ToolModel
from jarvis.tools.planner import ActionPlanningRequest
from jarvis.tools.scheduler import ActionSchedulePriority
from jarvis.tools.smoke_runtime import (
    RealActionSmokeRequest,
    RealActionSmokeResult,
    RealActionSmokeRuntime,
    RealActionSmokeStatus,
)


class AutonomousTaskMode(StrEnum):
    """
    Autonomous task execution mode.

    Step 17 starts with read-only and low-risk autonomy only.
    """

    READ_ONLY = "read_only"
    LOW_RISK = "low_risk"


class AutonomousTaskState(StrEnum):
    """
    Autonomous task lifecycle state.
    """

    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"


class AutonomousTaskDecision(StrEnum):
    """
    Autonomous task runtime decision.
    """

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    STARTED = "started"
    STEP_COMPLETED = "step_completed"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    FAILED = "failed"


class AutonomousTaskReason(StrEnum):
    """
    Machine-readable autonomous task reason.
    """

    TASK_ACCEPTED = "task_accepted"
    TASK_STARTED = "task_started"
    STEP_SUCCEEDED = "step_succeeded"
    STEP_FAILED = "step_failed"
    TASK_SUCCEEDED = "task_succeeded"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"
    EMPTY_TASK_BLOCKED = "empty_task_blocked"
    TOO_MANY_STEPS_BLOCKED = "too_many_steps_blocked"
    HIGH_RISK_BLOCKED = "high_risk_blocked"
    MUTATION_BLOCKED = "mutation_blocked"
    APPROVAL_REQUIRED_BLOCKED = "approval_required_blocked"
    SMOKE_RUNTIME_BLOCKED = "smoke_runtime_blocked"
    CANCELLATION_REQUESTED = "cancellation_requested"
    UNSUPPORTED_MODE = "unsupported_mode"


class AutonomousStepKind(StrEnum):
    """
    Supported autonomous step kind.

    Step 17 only allows steps that can be routed through RealActionSmokeRuntime.
    """

    RUN_TESTS = "run_tests"
    RUN_QUALITY_GATE = "run_quality_gate"
    SEARCH_PROJECT = "search_project"
    OPEN_FILE = "open_file"
    READ_FILE = "read_file"


class AutonomousTaskStep(ToolModel):
    """
    One bounded autonomous task step.

    This is still a proposal for the smoke runtime. It is not direct execution.
    """

    step_id: str = Field(default_factory=new_action_result_id)
    order: int = Field(ge=0)
    kind: AutonomousStepKind
    instruction: str
    target_path: str | None = None
    search_query: str | None = None
    risk: ActionRisk = ActionRisk.LOW
    requires_approval: bool = False
    completed: bool = False
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("step_id", "instruction")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("target_path", "search_query")
    @classmethod
    def _clean_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None

    @model_validator(mode="after")
    def _validate_step_shape(self) -> AutonomousTaskStep:
        if self.kind in {AutonomousStepKind.OPEN_FILE, AutonomousStepKind.READ_FILE}:
            if self.target_path is None:
                raise ValueError("file-oriented autonomous steps require target_path.")

        if self.kind == AutonomousStepKind.SEARCH_PROJECT:
            if self.search_query is None:
                raise ValueError("search_project steps require search_query.")

        return self


class AutonomousTaskRequest(ToolModel):
    """
    Request to execute a bounded autonomous task.

    This is the Step 17 top-level input.
    """

    task_id: str = Field(default_factory=new_action_result_id)
    objective: str
    mode: AutonomousTaskMode = AutonomousTaskMode.READ_ONLY
    steps: tuple[AutonomousTaskStep, ...]
    priority: ActionSchedulePriority = ActionSchedulePriority.NORMAL
    allow_memory_write: bool = False
    approved: bool = False
    max_steps: int = Field(default=5, ge=1)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("task_id", "objective")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @model_validator(mode="after")
    def _validate_task_bounds(self) -> AutonomousTaskRequest:
        if not self.steps:
            raise ValueError("autonomous task requires at least one step.")

        if len(self.steps) > self.max_steps:
            raise ValueError("autonomous task exceeds max_steps.")

        orders = [step.order for step in self.steps]

        if len(set(orders)) != len(orders):
            raise ValueError("autonomous task step orders must be unique.")

        return self


class AutonomousStepResult(ToolModel):
    """
    Result for one autonomous step.
    """

    result_id: str = Field(default_factory=new_action_result_id)
    task_id: str
    step_id: str
    order: int = Field(ge=0)
    kind: AutonomousStepKind
    decision: AutonomousTaskDecision
    reason: AutonomousTaskReason
    success: bool
    message: str
    smoke_result: RealActionSmokeResult | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("result_id", "task_id", "step_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class AutonomousTaskResult(ToolModel):
    """
    Final autonomous task result.
    """

    result_id: str = Field(default_factory=new_action_result_id)
    task_id: str
    objective: str
    state: AutonomousTaskState
    decision: AutonomousTaskDecision
    reason: AutonomousTaskReason
    success: bool
    step_results: tuple[AutonomousStepResult, ...] = ()
    message: str
    started_at: object = Field(default_factory=utc_now)
    completed_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("result_id", "task_id", "objective", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class SafeAutonomousTaskRuntimeConfig:
    """
    Safe autonomous task runtime configuration.
    """

    name: str = "safe_autonomous_task_runtime"
    max_steps_per_task: int = 5
    stop_on_first_failure: bool = True
    allow_low_risk_mode: bool = True
    allow_memory_write: bool = False

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.max_steps_per_task <= 0:
            raise ValueError("max_steps_per_task must be positive.")


@dataclass(frozen=True, slots=True)
class SafeAutonomousTaskRuntimeSnapshot:
    """
    Runtime diagnostics.
    """

    name: str
    task_count: int
    success_count: int
    failed_count: int
    blocked_count: int
    cancelled_count: int
    last_state: AutonomousTaskState | None
    last_reason: AutonomousTaskReason | None
    last_error: str | None

class SmokeRuntimeRunner(Protocol):
    """
    Narrow runtime protocol for autonomous task execution.

    SafeAutonomousTaskRuntime only needs the Step 16 smoke runtime contract:
    run a governed RealActionSmokeRequest and return RealActionSmokeResult.
    This keeps tests and future adapters type-safe without requiring subclassing.
    """

    def run(self, request: RealActionSmokeRequest) -> RealActionSmokeResult:
        ...


class SafeAutonomousTaskRuntime:
    """
    Safe autonomous task runtime.

    Responsibilities:
    - run bounded low-risk autonomous task steps
    - route every executable step through RealActionSmokeRuntime
    - enforce max step count
    - block high-risk and approval-required autonomy
    - support cancellation through ActionInterruptController
    - audit task lifecycle
    - preserve no-hidden-autonomy discipline

    Non-responsibilities:
    - no direct shell/file/browser/IDE execution
    - no destructive autonomy
    - no unbounded loops
    - no bypass of policy, validation, approval, scheduler, audit, or memory
    """

    def __init__(
        self,
        *,
        config: SafeAutonomousTaskRuntimeConfig | None = None,
        smoke_runtime: SmokeRuntimeRunner | None = None,
        interrupt_controller: ActionInterruptController | None = None,
        audit_log: ActionAuditLog | None = None,
    ) -> None:
        self._config = config or SafeAutonomousTaskRuntimeConfig()
        self._config.validate()

        self._smoke_runtime = smoke_runtime or RealActionSmokeRuntime()
        self._interrupt_controller = (
            interrupt_controller or ActionInterruptController()
        )
        self._audit_log = audit_log or ActionAuditLog()
        self._lock = RLock()

        self._task_count = 0
        self._success_count = 0
        self._failed_count = 0
        self._blocked_count = 0
        self._cancelled_count = 0
        self._last_state: AutonomousTaskState | None = None
        self._last_reason: AutonomousTaskReason | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def run(self, request: AutonomousTaskRequest) -> AutonomousTaskResult:
        """
        Run one bounded autonomous task.

        Every step goes through RealActionSmokeRuntime.
        """

        with self._lock:
            self._task_count += 1
            self._last_error = None

        token = self._interrupt_controller.create_token(
            action_id=request.task_id,
            interruptible=True,
            metadata={"runtime": self.name},
        )

        self._audit_task_started(request)

        preflight = self._preflight(request)

        if preflight is not None:
            self._record_result(preflight)

            return preflight

        step_results: list[AutonomousStepResult] = []

        for step in sorted(request.steps, key=lambda item: item.order):
            current_token = self._interrupt_controller.get_token(request.task_id)

            if current_token is not None and current_token.cancellation_requested:
                result = self._cancelled_result(
                    request=request,
                    step_results=tuple(step_results),
                    message="autonomous task cancelled before next step",
                )
                self._record_result(result)

                return result

            step_result = self._run_step(
                request=request,
                step=step,
            )
            step_results.append(step_result)

            if not step_result.success and self._config.stop_on_first_failure:
                result = AutonomousTaskResult(
                    task_id=request.task_id,
                    objective=request.objective,
                    state=AutonomousTaskState.FAILED,
                    decision=AutonomousTaskDecision.FAILED,
                    reason=AutonomousTaskReason.STEP_FAILED,
                    success=False,
                    step_results=tuple(step_results),
                    message="autonomous task stopped after failed step",
                    metadata={
                        "runtime": self.name,
                        "token_id": token.token_id,
                    },
                )
                self._audit_task_finished(result)
                self._record_result(result)

                return result

        result = AutonomousTaskResult(
            task_id=request.task_id,
            objective=request.objective,
            state=AutonomousTaskState.SUCCEEDED,
            decision=AutonomousTaskDecision.COMPLETED,
            reason=AutonomousTaskReason.TASK_SUCCEEDED,
            success=True,
            step_results=tuple(step_results),
            message="autonomous task completed safely",
            metadata={
                "runtime": self.name,
                "token_id": token.token_id,
            },
        )
        self._interrupt_controller.mark_completed(request.task_id)
        self._audit_task_finished(result)
        self._record_result(result)

        return result

    def cancel(self, task_id: str) -> None:
        """
        Request cancellation for an autonomous task.
        """

        self._interrupt_controller.interrupt(
            ActionInterruptRequest(
                action_id=task_id,
                kind=ActionInterruptKind.CANCEL,
                reason=ActionInterruptReason.USER_REQUESTED,
            )
        )

    def snapshot(self) -> SafeAutonomousTaskRuntimeSnapshot:
        """
        Return runtime diagnostics.
        """

        with self._lock:
            return SafeAutonomousTaskRuntimeSnapshot(
                name=self.name,
                task_count=self._task_count,
                success_count=self._success_count,
                failed_count=self._failed_count,
                blocked_count=self._blocked_count,
                cancelled_count=self._cancelled_count,
                last_state=self._last_state,
                last_reason=self._last_reason,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Reset diagnostics only.
        """

        with self._lock:
            self._task_count = 0
            self._success_count = 0
            self._failed_count = 0
            self._blocked_count = 0
            self._cancelled_count = 0
            self._last_state = None
            self._last_reason = None
            self._last_error = None

    def _preflight(
        self,
        request: AutonomousTaskRequest,
    ) -> AutonomousTaskResult | None:
        if len(request.steps) > self._config.max_steps_per_task:
            return self._blocked_result(
                request=request,
                reason=AutonomousTaskReason.TOO_MANY_STEPS_BLOCKED,
                message="autonomous task exceeds runtime step limit",
            )

        if request.mode == AutonomousTaskMode.LOW_RISK:
            if not self._config.allow_low_risk_mode:
                return self._blocked_result(
                    request=request,
                    reason=AutonomousTaskReason.UNSUPPORTED_MODE,
                    message="low-risk autonomous mode is disabled",
                )

        for step in request.steps:
            step_block = self._preflight_step(request=request, step=step)

            if step_block is not None:
                return step_block

        return None

    def _preflight_step(
        self,
        *,
        request: AutonomousTaskRequest,
        step: AutonomousTaskStep,
    ) -> AutonomousTaskResult | None:
        if step.risk in {ActionRisk.HIGH, ActionRisk.CRITICAL}:
            return self._blocked_result(
                request=request,
                reason=AutonomousTaskReason.HIGH_RISK_BLOCKED,
                message="high-risk autonomous steps are blocked",
            )

        if step.requires_approval and not request.approved:
            return self._blocked_result(
                request=request,
                reason=AutonomousTaskReason.APPROVAL_REQUIRED_BLOCKED,
                message="approval-required autonomous step is blocked",
            )

        if request.mode == AutonomousTaskMode.READ_ONLY:
            if step.kind not in self._read_only_kinds():
                return self._blocked_result(
                    request=request,
                    reason=AutonomousTaskReason.MUTATION_BLOCKED,
                    message="read-only autonomous mode blocks mutating steps",
                )

        return None

    def _run_step(
        self,
        *,
        request: AutonomousTaskRequest,
        step: AutonomousTaskStep,
    ) -> AutonomousStepResult:
        planning_request = self._planning_request_for_step(step)
        smoke_request = RealActionSmokeRequest(
            planning_request=planning_request,
            priority=request.priority,
            approved=request.approved,
            write_memory=(
                request.allow_memory_write
                and self._config.allow_memory_write
            ),
        )
        smoke_result = self._smoke_runtime.run(smoke_request)

        if smoke_result.success:
            return AutonomousStepResult(
                task_id=request.task_id,
                step_id=step.step_id,
                order=step.order,
                kind=step.kind,
                decision=AutonomousTaskDecision.STEP_COMPLETED,
                reason=AutonomousTaskReason.STEP_SUCCEEDED,
                success=True,
                message="autonomous step completed through smoke runtime",
                smoke_result=smoke_result,
                metadata={"runtime": self.name},
            )

        return AutonomousStepResult(
            task_id=request.task_id,
            step_id=step.step_id,
            order=step.order,
            kind=step.kind,
            decision=AutonomousTaskDecision.FAILED,
            reason=(
                AutonomousTaskReason.SMOKE_RUNTIME_BLOCKED
                if smoke_result.status
                in {
                    RealActionSmokeStatus.APPROVAL_REQUIRED,
                    RealActionSmokeStatus.POLICY_BLOCKED,
                    RealActionSmokeStatus.VALIDATION_BLOCKED,
                }
                else AutonomousTaskReason.STEP_FAILED
            ),
            success=False,
            message=smoke_result.message,
            smoke_result=smoke_result,
            metadata={
                "runtime": self.name,
                "smoke_reason": smoke_result.reason.value,
            },
        )

    @staticmethod
    def _planning_request_for_step(
        step: AutonomousTaskStep,
    ) -> ActionPlanningRequest:
        if step.kind == AutonomousStepKind.RUN_TESTS:
            return ActionPlanningRequest(
                user_intent="run tests and summarize failures"
            )

        if step.kind == AutonomousStepKind.RUN_QUALITY_GATE:
            return ActionPlanningRequest(user_intent="run quality gate")

        if step.kind == AutonomousStepKind.SEARCH_PROJECT:
            return ActionPlanningRequest(
                user_intent=f"search for {step.search_query}",
                search_query=step.search_query,
            )

        if step.kind == AutonomousStepKind.OPEN_FILE:
            return ActionPlanningRequest(
                user_intent="open this file",
                target_path=step.target_path,
            )

        if step.kind == AutonomousStepKind.READ_FILE:
            return ActionPlanningRequest(
                user_intent="search for file content",
                search_query=step.target_path,
            )

        return ActionPlanningRequest(user_intent=step.instruction)

    @staticmethod
    def _read_only_kinds() -> set[AutonomousStepKind]:
        return {
            AutonomousStepKind.RUN_TESTS,
            AutonomousStepKind.RUN_QUALITY_GATE,
            AutonomousStepKind.SEARCH_PROJECT,
            AutonomousStepKind.OPEN_FILE,
            AutonomousStepKind.READ_FILE,
        }

    def _blocked_result(
        self,
        *,
        request: AutonomousTaskRequest,
        reason: AutonomousTaskReason,
        message: str,
    ) -> AutonomousTaskResult:
        result = AutonomousTaskResult(
            task_id=request.task_id,
            objective=request.objective,
            state=AutonomousTaskState.BLOCKED,
            decision=AutonomousTaskDecision.BLOCKED,
            reason=reason,
            success=False,
            message=message,
            metadata={"runtime": self.name},
        )
        self._audit_task_finished(result)

        return result

    def _cancelled_result(
        self,
        *,
        request: AutonomousTaskRequest,
        step_results: tuple[AutonomousStepResult, ...],
        message: str,
    ) -> AutonomousTaskResult:
        result = AutonomousTaskResult(
            task_id=request.task_id,
            objective=request.objective,
            state=AutonomousTaskState.CANCELLED,
            decision=AutonomousTaskDecision.CANCELLED,
            reason=AutonomousTaskReason.TASK_CANCELLED,
            success=False,
            step_results=step_results,
            message=message,
            metadata={"runtime": self.name},
        )
        self._audit_task_finished(result)

        return result

    def _audit_task_started(self, request: AutonomousTaskRequest) -> None:
        self._audit_log.record(
            action_id=request.task_id,
            event_kind=ActionAuditEventKind.EXECUTION_STARTED,
            actor=ActionAuditActor.RUNTIME,
            outcome=ActionAuditOutcome.INFO,
            message="safe autonomous task started",
            risk=ActionRisk.LOW,
            status=ActionStatus.RUNNING,
            source_runtime=self.name,
            data={
                "objective": request.objective,
                "mode": request.mode.value,
                "step_count": len(request.steps),
            },
        )

    def _audit_task_finished(self, result: AutonomousTaskResult) -> None:
        self._audit_log.record(
            action_id=result.task_id,
            event_kind=(
                ActionAuditEventKind.EXECUTION_COMPLETED
                if result.success
                else ActionAuditEventKind.EXECUTION_FAILED
            ),
            actor=ActionAuditActor.RUNTIME,
            outcome=(
                ActionAuditOutcome.SUCCEEDED
                if result.success
                else ActionAuditOutcome.FAILED
            ),
            message=result.message,
            risk=ActionRisk.LOW,
            status=ActionStatus.SUCCEEDED if result.success else ActionStatus.FAILED,
            source_runtime=self.name,
            data={
                "state": result.state.value,
                "reason": result.reason.value,
                "step_results": len(result.step_results),
            },
        )

    def _record_result(self, result: AutonomousTaskResult) -> None:
        with self._lock:
            self._last_state = result.state
            self._last_reason = result.reason

            if result.success:
                self._success_count += 1

            elif result.state == AutonomousTaskState.BLOCKED:
                self._blocked_count += 1

            elif result.state == AutonomousTaskState.CANCELLED:
                self._cancelled_count += 1

            else:
                self._failed_count += 1