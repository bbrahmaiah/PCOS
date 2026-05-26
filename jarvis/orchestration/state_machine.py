from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import RLock

from pydantic import Field, field_validator

from jarvis.orchestration.ids import (
    OrchestrationId,
    new_orchestration_id,
    utc_now,
    validate_orchestration_id,
)
from jarvis.orchestration.models import (
    OrchestrationModel,
    OrchestrationSnapshot,
    OrchestratorState,
)


class OrchestrationEventKind(StrEnum):
    """
    Events that may request orchestrator state transitions.
    """

    BOOTSTRAP_STARTED = "bootstrap_started"
    BOOTSTRAP_COMPLETED = "bootstrap_completed"
    COORDINATION_STARTED = "coordination_started"
    COORDINATION_COMPLETED = "coordination_completed"
    LOAD_INCREASED = "load_increased"
    LOAD_NORMALIZED = "load_normalized"
    LOAD_SHEDDING_STARTED = "load_shedding_started"
    LOAD_SHEDDING_COMPLETED = "load_shedding_completed"
    RECOVERY_STARTED = "recovery_started"
    RECOVERY_COMPLETED = "recovery_completed"
    SHUTDOWN_REQUESTED = "shutdown_requested"
    SHUTDOWN_COMPLETED = "shutdown_completed"
    RESET_REQUESTED = "reset_requested"


class OrchestrationTransitionDecision(StrEnum):
    """
    Result of a state transition request.
    """

    APPLIED = "applied"
    REJECTED = "rejected"
    IGNORED = "ignored"


class OrchestrationTransitionReason(StrEnum):
    """
    Machine-readable transition reason.
    """

    BOOTSTRAP_ACCEPTED = "bootstrap_accepted"
    BOOTSTRAP_FINISHED = "bootstrap_finished"
    COORDINATION_ACCEPTED = "coordination_accepted"
    COORDINATION_FINISHED = "coordination_finished"
    BUSY_ACCEPTED = "busy_accepted"
    IDLE_ACCEPTED = "idle_accepted"
    LOAD_SHEDDING_ACCEPTED = "load_shedding_accepted"
    LOAD_SHEDDING_FINISHED = "load_shedding_finished"
    RECOVERY_ACCEPTED = "recovery_accepted"
    RECOVERY_FINISHED = "recovery_finished"
    SHUTDOWN_ACCEPTED = "shutdown_accepted"
    SHUTDOWN_FINISHED = "shutdown_finished"
    RESET_ACCEPTED = "reset_accepted"
    SAME_STATE_IGNORED = "same_state_ignored"
    TERMINAL_STATE_REJECTED = "terminal_state_rejected"
    INVALID_TRANSITION_REJECTED = "invalid_transition_rejected"
    GUARD_REJECTED = "guard_rejected"


class OrchestrationGuardReason(StrEnum):
    """
    Guard condition reason.
    """

    GUARD_PASSED = "guard_passed"
    ACTIVE_TASKS_BLOCK_SHUTDOWN = "active_tasks_block_shutdown"
    ACTIVE_JOBS_BLOCK_SHUTDOWN = "active_jobs_block_shutdown"
    COORDINATION_REQUIRES_WORKERS = "coordination_requires_workers"
    BUSY_REQUIRES_ACTIVE_TASKS = "busy_requires_active_tasks"
    LOAD_SHEDDING_REQUIRES_PRESSURE = "load_shedding_requires_pressure"
    RECOVERY_REQUIRES_FAILURE = "recovery_requires_failure"


class OrchestrationStateContext(OrchestrationModel):
    """
    Runtime facts used by the state machine guards.

    The state machine does not inspect workers or tasks directly. It receives
    context from the orchestration kernel.
    """

    active_task_count: int = Field(default=0, ge=0)
    active_job_count: int = Field(default=0, ge=0)
    registered_worker_count: int = Field(default=0, ge=0)
    failed_worker_count: int = Field(default=0, ge=0)
    resource_pressure: bool = False
    shutdown_requested: bool = False
    metadata: dict[str, object] = Field(default_factory=dict)

    @property
    def has_active_work(self) -> bool:
        return self.active_task_count > 0 or self.active_job_count > 0

    @property
    def has_workers(self) -> bool:
        return self.registered_worker_count > 0

    @property
    def has_failure(self) -> bool:
        return self.failed_worker_count > 0


class OrchestrationState(OrchestrationModel):
    """
    Current orchestrator state.

    Exactly one OrchestratorState owns the orchestration kernel at any time.
    """

    orchestration_id: OrchestrationId = Field(default_factory=new_orchestration_id)
    state: OrchestratorState = OrchestratorState.STARTING
    context: OrchestrationStateContext = Field(
        default_factory=OrchestrationStateContext
    )
    version: int = Field(default=0, ge=0)
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("orchestration_id")
    @classmethod
    def _validate_orchestration_id(cls, value: str) -> str:
        return validate_orchestration_id(value)

    def snapshot(self) -> OrchestrationSnapshot:
        """
        Convert current state to a Step 0 snapshot contract.
        """

        return OrchestrationSnapshot(
            orchestration_id=self.orchestration_id,
            state=self.state,
            active_task_count=self.context.active_task_count,
            active_job_count=self.context.active_job_count,
            registered_worker_count=self.context.registered_worker_count,
            metadata={
                "version": self.version,
                "failed_worker_count": self.context.failed_worker_count,
                "resource_pressure": self.context.resource_pressure,
            },
        )


class OrchestrationTransition(OrchestrationModel):
    """
    One logged state transition.
    """

    orchestration_id: OrchestrationId
    event: OrchestrationEventKind
    from_state: OrchestratorState
    to_state: OrchestratorState
    decision: OrchestrationTransitionDecision
    reason: OrchestrationTransitionReason
    guard_reason: OrchestrationGuardReason
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("orchestration_id")
    @classmethod
    def _validate_orchestration_id(cls, value: str) -> str:
        return validate_orchestration_id(value)

    @field_validator("message")
    @classmethod
    def _required_message(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("message cannot be empty.")

        return cleaned


class OrchestrationTransitionResult(OrchestrationModel):
    """
    Result returned by the state machine.
    """

    state: OrchestrationState
    transition: OrchestrationTransition
    changed: bool

    @property
    def accepted(self) -> bool:
        return (
            self.transition.decision
            == OrchestrationTransitionDecision.APPLIED
        )


class OrchestrationStateMachineConfig(OrchestrationModel):
    """
    State machine configuration.
    """

    name: str = "orchestration_state_machine"
    allow_reset_from_stopped: bool = True
    require_no_active_work_for_shutdown: bool = True

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class OrchestrationStateMachineSnapshot:
    """
    State machine diagnostics.
    """

    name: str
    current_state: OrchestratorState
    transition_count: int
    applied_count: int
    rejected_count: int
    ignored_count: int
    last_decision: OrchestrationTransitionDecision | None
    last_reason: OrchestrationTransitionReason | None


class OrchestrationStateMachine:
    """
    Phase 6 Orchestration State Machine.

    Responsibilities:
    - keep exactly one orchestrator state
    - guard transitions
    - log every transition decision
    - expose current snapshot for later scheduler/kernel layers

    Non-responsibilities:
    - no task scheduling
    - no task execution
    - no worker coordination
    - no resource reservation
    """

    _ALLOWED_TRANSITIONS: dict[OrchestratorState, set[OrchestratorState]] = {
        OrchestratorState.STARTING: {
            OrchestratorState.IDLE,
            OrchestratorState.RECOVERING,
            OrchestratorState.SHUTTING_DOWN,
        },
        OrchestratorState.IDLE: {
            OrchestratorState.COORDINATING,
            OrchestratorState.BUSY,
            OrchestratorState.RECOVERING,
            OrchestratorState.SHUTTING_DOWN,
        },
        OrchestratorState.COORDINATING: {
            OrchestratorState.IDLE,
            OrchestratorState.BUSY,
            OrchestratorState.LOAD_SHEDDING,
            OrchestratorState.RECOVERING,
            OrchestratorState.SHUTTING_DOWN,
        },
        OrchestratorState.BUSY: {
            OrchestratorState.IDLE,
            OrchestratorState.COORDINATING,
            OrchestratorState.LOAD_SHEDDING,
            OrchestratorState.RECOVERING,
            OrchestratorState.SHUTTING_DOWN,
        },
        OrchestratorState.LOAD_SHEDDING: {
            OrchestratorState.IDLE,
            OrchestratorState.COORDINATING,
            OrchestratorState.BUSY,
            OrchestratorState.RECOVERING,
            OrchestratorState.SHUTTING_DOWN,
        },
        OrchestratorState.RECOVERING: {
            OrchestratorState.IDLE,
            OrchestratorState.COORDINATING,
            OrchestratorState.SHUTTING_DOWN,
        },
        OrchestratorState.SHUTTING_DOWN: {
            OrchestratorState.STOPPED,
            OrchestratorState.RECOVERING,
        },
        OrchestratorState.STOPPED: set(),
    }

    def __init__(
        self,
        *,
        config: OrchestrationStateMachineConfig | None = None,
        initial_state: OrchestrationState | None = None,
    ) -> None:
        self._config = config or OrchestrationStateMachineConfig()
        self._state = initial_state or OrchestrationState()
        self._transitions: list[OrchestrationTransition] = []
        self._lock = RLock()

        self._applied_count = 0
        self._rejected_count = 0
        self._ignored_count = 0
        self._last_decision: OrchestrationTransitionDecision | None = None
        self._last_reason: OrchestrationTransitionReason | None = None

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def state(self) -> OrchestrationState:
        return self._state

    def transition(
        self,
        event: OrchestrationEventKind,
        *,
        context: OrchestrationStateContext | None = None,
        message: str | None = None,
    ) -> OrchestrationTransitionResult:
        """
        Apply an orchestration event to the state machine.
        """

        with self._lock:
            current = self._state
            next_context = context or current.context
            target = self._target_for_event(
                event=event,
                current=current.state,
                context=next_context,
            )

            result = self._attempt_transition_locked(
                event=event,
                target=target,
                context=next_context,
                message=message,
            )

            return result

    def bootstrap_completed(
        self,
        context: OrchestrationStateContext | None = None,
    ) -> OrchestrationTransitionResult:
        return self.transition(
            OrchestrationEventKind.BOOTSTRAP_COMPLETED,
            context=context,
            message="orchestrator bootstrap completed",
        )

    def start_coordination(
        self,
        context: OrchestrationStateContext,
    ) -> OrchestrationTransitionResult:
        return self.transition(
            OrchestrationEventKind.COORDINATION_STARTED,
            context=context,
            message="orchestrator coordination started",
        )

    def complete_coordination(
        self,
        context: OrchestrationStateContext | None = None,
    ) -> OrchestrationTransitionResult:
        return self.transition(
            OrchestrationEventKind.COORDINATION_COMPLETED,
            context=context,
            message="orchestrator coordination completed",
        )

    def enter_busy(
        self,
        context: OrchestrationStateContext,
    ) -> OrchestrationTransitionResult:
        return self.transition(
            OrchestrationEventKind.LOAD_INCREASED,
            context=context,
            message="orchestrator entered busy state",
        )

    def normalize_load(
        self,
        context: OrchestrationStateContext | None = None,
    ) -> OrchestrationTransitionResult:
        return self.transition(
            OrchestrationEventKind.LOAD_NORMALIZED,
            context=context,
            message="orchestrator load normalized",
        )

    def start_load_shedding(
        self,
        context: OrchestrationStateContext,
    ) -> OrchestrationTransitionResult:
        return self.transition(
            OrchestrationEventKind.LOAD_SHEDDING_STARTED,
            context=context,
            message="orchestrator started load shedding",
        )

    def start_recovery(
        self,
        context: OrchestrationStateContext,
    ) -> OrchestrationTransitionResult:
        return self.transition(
            OrchestrationEventKind.RECOVERY_STARTED,
            context=context,
            message="orchestrator recovery started",
        )

    def complete_recovery(
        self,
        context: OrchestrationStateContext | None = None,
    ) -> OrchestrationTransitionResult:
        return self.transition(
            OrchestrationEventKind.RECOVERY_COMPLETED,
            context=context,
            message="orchestrator recovery completed",
        )

    def request_shutdown(
        self,
        context: OrchestrationStateContext | None = None,
    ) -> OrchestrationTransitionResult:
        return self.transition(
            OrchestrationEventKind.SHUTDOWN_REQUESTED,
            context=context,
            message="orchestrator shutdown requested",
        )

    def complete_shutdown(self) -> OrchestrationTransitionResult:
        return self.transition(
            OrchestrationEventKind.SHUTDOWN_COMPLETED,
            message="orchestrator shutdown completed",
        )

    def reset(self) -> OrchestrationTransitionResult:
        return self.transition(
            OrchestrationEventKind.RESET_REQUESTED,
            context=OrchestrationStateContext(),
            message="orchestrator state machine reset",
        )

    def transition_log(self) -> tuple[OrchestrationTransition, ...]:
        """
        Return immutable transition log.
        """

        with self._lock:
            return tuple(self._transitions)

    def snapshot(self) -> OrchestrationStateMachineSnapshot:
        """
        Return state machine diagnostics.
        """

        with self._lock:
            return OrchestrationStateMachineSnapshot(
                name=self.name,
                current_state=self._state.state,
                transition_count=len(self._transitions),
                applied_count=self._applied_count,
                rejected_count=self._rejected_count,
                ignored_count=self._ignored_count,
                last_decision=self._last_decision,
                last_reason=self._last_reason,
            )

    def runtime_snapshot(self) -> OrchestrationSnapshot:
        """
        Return Step 0 orchestration snapshot.
        """

        with self._lock:
            return self._state.snapshot()

    def _attempt_transition_locked(
        self,
        *,
        event: OrchestrationEventKind,
        target: OrchestratorState,
        context: OrchestrationStateContext,
        message: str | None,
    ) -> OrchestrationTransitionResult:
        current = self._state
        guard_reason = self._guard(
            current=current.state,
            target=target,
            context=context,
        )

        if current.state == target:
            transition = self._build_transition(
                event=event,
                from_state=current.state,
                to_state=target,
                decision=OrchestrationTransitionDecision.IGNORED,
                reason=OrchestrationTransitionReason.SAME_STATE_IGNORED,
                guard_reason=guard_reason,
                message=message or "orchestrator already in target state",
            )
            self._record_transition(transition)

            return OrchestrationTransitionResult(
                state=current,
                transition=transition,
                changed=False,
            )

        if current.state == OrchestratorState.STOPPED:
            if not (
                event == OrchestrationEventKind.RESET_REQUESTED
                and self._config.allow_reset_from_stopped
            ):
                transition = self._build_transition(
                    event=event,
                    from_state=current.state,
                    to_state=target,
                    decision=OrchestrationTransitionDecision.REJECTED,
                    reason=OrchestrationTransitionReason.TERMINAL_STATE_REJECTED,
                    guard_reason=guard_reason,
                    message="stopped orchestrator rejects non-reset transitions",
                )
                self._record_transition(transition)

                return OrchestrationTransitionResult(
                    state=current,
                    transition=transition,
                    changed=False,
                )

        if target not in self._ALLOWED_TRANSITIONS[current.state]:
            if not (
                current.state == OrchestratorState.STOPPED
                and target == OrchestratorState.STARTING
                and self._config.allow_reset_from_stopped
            ):
                transition = self._build_transition(
                    event=event,
                    from_state=current.state,
                    to_state=target,
                    decision=OrchestrationTransitionDecision.REJECTED,
                    reason=OrchestrationTransitionReason.INVALID_TRANSITION_REJECTED,
                    guard_reason=guard_reason,
                    message="orchestrator transition is not allowed",
                )
                self._record_transition(transition)

                return OrchestrationTransitionResult(
                    state=current,
                    transition=transition,
                    changed=False,
                )

        if guard_reason != OrchestrationGuardReason.GUARD_PASSED:
            transition = self._build_transition(
                event=event,
                from_state=current.state,
                to_state=target,
                decision=OrchestrationTransitionDecision.REJECTED,
                reason=OrchestrationTransitionReason.GUARD_REJECTED,
                guard_reason=guard_reason,
                message="orchestrator transition guard rejected",
            )
            self._record_transition(transition)

            return OrchestrationTransitionResult(
                state=current,
                transition=transition,
                changed=False,
            )

        next_state = OrchestrationState(
            orchestration_id=current.orchestration_id,
            state=target,
            context=context,
            version=current.version + 1,
            metadata=current.metadata,
        )
        transition = self._build_transition(
            event=event,
            from_state=current.state,
            to_state=target,
            decision=OrchestrationTransitionDecision.APPLIED,
            reason=self._reason_for_target(event=event, target=target),
            guard_reason=guard_reason,
            message=message or "orchestrator transition applied",
        )
        self._state = next_state
        self._record_transition(transition)

        return OrchestrationTransitionResult(
            state=next_state,
            transition=transition,
            changed=True,
        )

    def _guard(
        self,
        *,
        current: OrchestratorState,
        target: OrchestratorState,
        context: OrchestrationStateContext,
    ) -> OrchestrationGuardReason:
        del current

        if target == OrchestratorState.COORDINATING:
            if not context.has_workers:
                return OrchestrationGuardReason.COORDINATION_REQUIRES_WORKERS

        if target == OrchestratorState.BUSY:
            if context.active_task_count <= 0:
                return OrchestrationGuardReason.BUSY_REQUIRES_ACTIVE_TASKS

        if target == OrchestratorState.LOAD_SHEDDING:
            if not context.resource_pressure:
                return OrchestrationGuardReason.LOAD_SHEDDING_REQUIRES_PRESSURE

        if target == OrchestratorState.RECOVERING:
            if not context.has_failure:
                return OrchestrationGuardReason.RECOVERY_REQUIRES_FAILURE

        if target == OrchestratorState.SHUTTING_DOWN:
            if self._config.require_no_active_work_for_shutdown:
                if context.active_task_count > 0:
                    return OrchestrationGuardReason.ACTIVE_TASKS_BLOCK_SHUTDOWN

                if context.active_job_count > 0:
                    return OrchestrationGuardReason.ACTIVE_JOBS_BLOCK_SHUTDOWN

        return OrchestrationGuardReason.GUARD_PASSED

    @staticmethod
    def _target_for_event(
        *,
        event: OrchestrationEventKind,
        current: OrchestratorState,
        context: OrchestrationStateContext,
    ) -> OrchestratorState:
        if event == OrchestrationEventKind.BOOTSTRAP_STARTED:
            return OrchestratorState.STARTING

        if event == OrchestrationEventKind.BOOTSTRAP_COMPLETED:
            return OrchestratorState.IDLE

        if event == OrchestrationEventKind.COORDINATION_STARTED:
            return OrchestratorState.COORDINATING

        if event == OrchestrationEventKind.COORDINATION_COMPLETED:
            return (
                OrchestratorState.BUSY
                if context.has_active_work
                else OrchestratorState.IDLE
            )

        if event == OrchestrationEventKind.LOAD_INCREASED:
            return OrchestratorState.BUSY

        if event == OrchestrationEventKind.LOAD_NORMALIZED:
            return OrchestratorState.IDLE

        if event == OrchestrationEventKind.LOAD_SHEDDING_STARTED:
            return OrchestratorState.LOAD_SHEDDING

        if event == OrchestrationEventKind.LOAD_SHEDDING_COMPLETED:
            return (
                OrchestratorState.BUSY
                if context.has_active_work
                else OrchestratorState.IDLE
            )

        if event == OrchestrationEventKind.RECOVERY_STARTED:
            return OrchestratorState.RECOVERING

        if event == OrchestrationEventKind.RECOVERY_COMPLETED:
            return (
                OrchestratorState.COORDINATING
                if context.has_active_work
                else OrchestratorState.IDLE
            )

        if event == OrchestrationEventKind.SHUTDOWN_REQUESTED:
            return OrchestratorState.SHUTTING_DOWN

        if event == OrchestrationEventKind.SHUTDOWN_COMPLETED:
            return OrchestratorState.STOPPED

        if event == OrchestrationEventKind.RESET_REQUESTED:
            return OrchestratorState.STARTING

        return current

    @staticmethod
    def _reason_for_target(
        *,
        event: OrchestrationEventKind,
        target: OrchestratorState,
    ) -> OrchestrationTransitionReason:
        if event == OrchestrationEventKind.BOOTSTRAP_COMPLETED:
            return OrchestrationTransitionReason.BOOTSTRAP_FINISHED

        if event == OrchestrationEventKind.COORDINATION_STARTED:
            return OrchestrationTransitionReason.COORDINATION_ACCEPTED

        if event == OrchestrationEventKind.COORDINATION_COMPLETED:
            return OrchestrationTransitionReason.COORDINATION_FINISHED

        if event == OrchestrationEventKind.LOAD_INCREASED:
            return OrchestrationTransitionReason.BUSY_ACCEPTED

        if event == OrchestrationEventKind.LOAD_NORMALIZED:
            return OrchestrationTransitionReason.IDLE_ACCEPTED

        if event == OrchestrationEventKind.LOAD_SHEDDING_STARTED:
            return OrchestrationTransitionReason.LOAD_SHEDDING_ACCEPTED

        if event == OrchestrationEventKind.LOAD_SHEDDING_COMPLETED:
            return OrchestrationTransitionReason.LOAD_SHEDDING_FINISHED

        if event == OrchestrationEventKind.RECOVERY_STARTED:
            return OrchestrationTransitionReason.RECOVERY_ACCEPTED

        if event == OrchestrationEventKind.RECOVERY_COMPLETED:
            return OrchestrationTransitionReason.RECOVERY_FINISHED

        if event == OrchestrationEventKind.SHUTDOWN_REQUESTED:
            return OrchestrationTransitionReason.SHUTDOWN_ACCEPTED

        if event == OrchestrationEventKind.SHUTDOWN_COMPLETED:
            return OrchestrationTransitionReason.SHUTDOWN_FINISHED

        if event == OrchestrationEventKind.RESET_REQUESTED:
            return OrchestrationTransitionReason.RESET_ACCEPTED

        if target == OrchestratorState.STARTING:
            return OrchestrationTransitionReason.BOOTSTRAP_ACCEPTED

        return OrchestrationTransitionReason.IDLE_ACCEPTED

    def _build_transition(
        self,
        *,
        event: OrchestrationEventKind,
        from_state: OrchestratorState,
        to_state: OrchestratorState,
        decision: OrchestrationTransitionDecision,
        reason: OrchestrationTransitionReason,
        guard_reason: OrchestrationGuardReason,
        message: str,
    ) -> OrchestrationTransition:
        return OrchestrationTransition(
            orchestration_id=self._state.orchestration_id,
            event=event,
            from_state=from_state,
            to_state=to_state,
            decision=decision,
            reason=reason,
            guard_reason=guard_reason,
            message=message,
        )

    def _record_transition(self, transition: OrchestrationTransition) -> None:
        self._transitions.append(transition)
        self._last_decision = transition.decision
        self._last_reason = transition.reason

        if transition.decision == OrchestrationTransitionDecision.APPLIED:
            self._applied_count += 1

        elif transition.decision == OrchestrationTransitionDecision.REJECTED:
            self._rejected_count += 1

        else:
            self._ignored_count += 1