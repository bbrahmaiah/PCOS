from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum, StrEnum
from threading import RLock
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.orchestration.ids import (
    TaskId,
    WorkerId,
    utc_now,
    validate_task_id,
    validate_worker_id,
)
from jarvis.orchestration.models import OrchestrationModel, WorkerCapability


def new_interrupt_id() -> str:
    return f"interrupt_{uuid4().hex}"


def new_interrupt_dispatch_id() -> str:
    return f"intdispatch_{uuid4().hex}"


class InterruptKind(StrEnum):
    """
    Interrupt source kind.
    """

    USER_INTERRUPT = "user_interrupt"
    TIMEOUT = "timeout"
    SHUTDOWN = "shutdown"
    SAFETY = "safety"
    RECOVERY = "recovery"


class InterruptPropagationStatus(StrEnum):
    """
    Lifecycle status for interrupt propagation.
    """

    CREATED = "created"
    PROPAGATING = "propagating"
    WAITING_ACK = "waiting_ack"
    COMPLETED = "completed"
    ESCALATED = "escalated"
    FAILED = "failed"


class InterruptPropagationDecision(StrEnum):
    """
    Decision returned by InterruptPropagator.
    """

    STARTED = "started"
    DISPATCHED = "dispatched"
    ACKNOWLEDGED = "acknowledged"
    COMPLETED = "completed"
    ESCALATED = "escalated"
    REJECTED = "rejected"


class InterruptPropagationReason(StrEnum):
    """
    Machine-readable interrupt propagation reason.
    """

    INTERRUPT_STARTED = "interrupt_started"
    INTERRUPT_DISPATCHED = "interrupt_dispatched"
    INTERRUPT_ACKNOWLEDGED = "interrupt_acknowledged"
    INTERRUPT_COMPLETED = "interrupt_completed"
    INTERRUPT_ESCALATED_ORPHANED_TASKS = "interrupt_escalated_orphaned_tasks"
    INTERRUPT_NOT_FOUND = "interrupt_not_found"
    INTERRUPT_ALREADY_ACTIVE = "interrupt_already_active"
    INVALID_ACKNOWLEDGEMENT = "invalid_acknowledgement"
    WRONG_WORKER_ACKNOWLEDGED = "wrong_worker_acknowledged"
    WRONG_ORDER_ACKNOWLEDGED = "wrong_order_acknowledged"
    PROPAGATION_ORDER_EMPTY = "propagation_order_empty"
    DISPATCH_ALREADY_ACKNOWLEDGED = "dispatch_already_acknowledged"


class PropagationPhase(IntEnum):
    """
    Stable interrupt propagation order.

    Lower value dispatches first.
    """

    PRESENCE = 0
    COGNITION = 10
    TOOL = 20
    MEMORY = 30
    BACKGROUND = 40


_CAPABILITY_TO_PHASE: dict[WorkerCapability, PropagationPhase] = {
    WorkerCapability.PRESENCE: PropagationPhase.PRESENCE,
    WorkerCapability.CONVERSATION: PropagationPhase.PRESENCE,
    WorkerCapability.COGNITION: PropagationPhase.COGNITION,
    WorkerCapability.TOOL_ACTION: PropagationPhase.TOOL,
    WorkerCapability.MEMORY: PropagationPhase.MEMORY,
    WorkerCapability.BACKGROUND: PropagationPhase.BACKGROUND,
}


class InterruptEvent(OrchestrationModel):
    """
    User/runtime interrupt event.

    This event describes the need to interrupt. It does not broadcast directly.
    """

    interrupt_id: str = Field(default_factory=new_interrupt_id)
    kind: InterruptKind
    reason: str
    requested_by: str = "user"
    affected_task_ids: tuple[TaskId, ...] = ()
    timeout_ms: int = Field(default=2_000, gt=0)
    rollback_requested: bool = True
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("interrupt_id", "reason", "requested_by")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("interrupt_id")
    @classmethod
    def _validate_interrupt_id(cls, value: str) -> str:
        if not value.startswith("interrupt_"):
            raise ValueError("interrupt_id must start with 'interrupt_'.")

        return value

    @field_validator("affected_task_ids")
    @classmethod
    def _validate_task_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(validate_task_id(value) for value in values)


class PropagationTarget(OrchestrationModel):
    """
    One ordered interrupt target.

    Targets are worker/capability groups. The propagator dispatches one target
    at a time and waits for acknowledgement.
    """

    phase: PropagationPhase
    capability: WorkerCapability
    worker_id: WorkerId
    task_ids: tuple[TaskId, ...]
    rollback_supported: bool = False
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("worker_id")
    @classmethod
    def _validate_worker_id(cls, value: str) -> str:
        return validate_worker_id(value)

    @field_validator("task_ids")
    @classmethod
    def _validate_task_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            raise ValueError("propagation target requires task ids.")

        return tuple(validate_task_id(value) for value in values)

    @model_validator(mode="after")
    def _validate_phase_matches_capability(self) -> PropagationTarget:
        expected = _CAPABILITY_TO_PHASE.get(self.capability)

        if expected is None:
            raise ValueError("capability is not interrupt-propagation capable.")

        if self.phase != expected:
            raise ValueError("propagation phase must match worker capability.")

        return self


class PropagationOrder(OrchestrationModel):
    """
    Ordered propagation plan.

    Interrupts do not broadcast. They flow through this order.
    """

    targets: tuple[PropagationTarget, ...]

    @model_validator(mode="after")
    def _validate_order(self) -> PropagationOrder:
        if not self.targets:
            raise ValueError("propagation order cannot be empty.")

        phases = [int(target.phase) for target in self.targets]

        if phases != sorted(phases):
            raise ValueError("propagation targets must be sorted by phase.")

        seen_pairs: set[tuple[str, int]] = set()

        for target in self.targets:
            key = (target.worker_id, int(target.phase))

            if key in seen_pairs:
                raise ValueError("duplicate worker target in propagation order.")

            seen_pairs.add(key)

        return self

    @staticmethod
    def ordered(
        targets: tuple[PropagationTarget, ...],
    ) -> PropagationOrder:
        """
        Build a stable propagation order.
        """

        return PropagationOrder(
            targets=tuple(sorted(targets, key=lambda target: int(target.phase)))
        )

    @property
    def task_ids(self) -> set[TaskId]:
        found: set[TaskId] = set()

        for target in self.targets:
            found.update(target.task_ids)

        return found


class InterruptDispatch(OrchestrationModel):
    """
    Interrupt dispatch sent from orchestrator to one target.

    This is a typed message, not direct worker cancellation.
    """

    dispatch_id: str = Field(default_factory=new_interrupt_dispatch_id)
    interrupt_id: str
    target: PropagationTarget
    status: InterruptPropagationStatus = InterruptPropagationStatus.WAITING_ACK
    sent_at: object = Field(default_factory=utc_now)
    acknowledged_at: object | None = None
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("dispatch_id", "interrupt_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("dispatch_id")
    @classmethod
    def _validate_dispatch_id(cls, value: str) -> str:
        if not value.startswith("intdispatch_"):
            raise ValueError("dispatch_id must start with 'intdispatch_'.")

        return value

    @field_validator("interrupt_id")
    @classmethod
    def _validate_interrupt_id(cls, value: str) -> str:
        if not value.startswith("interrupt_"):
            raise ValueError("interrupt_id must start with 'interrupt_'.")

        return value

    @property
    def acknowledged(self) -> bool:
        return self.acknowledged_at is not None


class InterruptAcknowledgement(OrchestrationModel):
    """
    Worker acknowledgement for one interrupt dispatch.
    """

    interrupt_id: str
    dispatch_id: str
    worker_id: WorkerId
    capability: WorkerCapability
    task_ids: tuple[TaskId, ...]
    accepted: bool = True
    rollback_started: bool = False
    message: str = "interrupt acknowledged"
    acknowledged_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("interrupt_id", "dispatch_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("interrupt_id")
    @classmethod
    def _validate_interrupt_id(cls, value: str) -> str:
        if not value.startswith("interrupt_"):
            raise ValueError("interrupt_id must start with 'interrupt_'.")

        return value

    @field_validator("dispatch_id")
    @classmethod
    def _validate_dispatch_id(cls, value: str) -> str:
        if not value.startswith("intdispatch_"):
            raise ValueError("dispatch_id must start with 'intdispatch_'.")

        return value

    @field_validator("worker_id")
    @classmethod
    def _validate_worker_id(cls, value: str) -> str:
        return validate_worker_id(value)

    @field_validator("task_ids")
    @classmethod
    def _validate_task_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            raise ValueError("acknowledgement requires task ids.")

        return tuple(validate_task_id(value) for value in values)


class OrphanedInterrupt(OrchestrationModel):
    """
    Escalated dispatch that was not acknowledged in time.
    """

    interrupt_id: str
    dispatch_id: str
    worker_id: WorkerId
    task_ids: tuple[TaskId, ...]
    timeout_ms: int
    escalated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("interrupt_id", "dispatch_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("worker_id")
    @classmethod
    def _validate_worker_id(cls, value: str) -> str:
        return validate_worker_id(value)

    @field_validator("task_ids")
    @classmethod
    def _validate_task_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(validate_task_id(value) for value in values)


class InterruptPropagationRecord(OrchestrationModel):
    """
    Runtime record of an active interrupt propagation.
    """

    event: InterruptEvent
    order: PropagationOrder
    dispatches: tuple[InterruptDispatch, ...] = ()
    acknowledgements: tuple[InterruptAcknowledgement, ...] = ()
    status: InterruptPropagationStatus = InterruptPropagationStatus.CREATED
    current_index: int = Field(default=0, ge=0)
    started_at: object = Field(default_factory=utc_now)
    completed_at: object | None = None
    orphaned: tuple[OrphanedInterrupt, ...] = ()
    metadata: dict[str, object] = Field(default_factory=dict)

    @property
    def interrupt_id(self) -> str:
        return self.event.interrupt_id

    @property
    def complete(self) -> bool:
        return self.status == InterruptPropagationStatus.COMPLETED

    @property
    def current_dispatch(self) -> InterruptDispatch | None:
        if not self.dispatches:
            return None

        latest = self.dispatches[-1]

        if latest.acknowledged:
            return None

        return latest

    @property
    def next_target(self) -> PropagationTarget | None:
        if self.current_index >= len(self.order.targets):
            return None

        return self.order.targets[self.current_index]


class InterruptPropagationResult(OrchestrationModel):
    """
    Result returned by InterruptPropagator.
    """

    decision: InterruptPropagationDecision
    reason: InterruptPropagationReason
    success: bool
    message: str
    record: InterruptPropagationRecord | None = None
    dispatch: InterruptDispatch | None = None
    acknowledgement: InterruptAcknowledgement | None = None
    orphaned: OrphanedInterrupt | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("message")
    @classmethod
    def _required_message(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("message cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class InterruptPropagatorConfig:
    """
    Interrupt propagator configuration.
    """

    name: str = "interrupt_propagator"
    orphan_timeout_ms: int = 2_000

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.orphan_timeout_ms <= 0:
            raise ValueError("orphan_timeout_ms must be positive.")


@dataclass(frozen=True, slots=True)
class InterruptPropagatorSnapshot:
    """
    Interrupt propagator diagnostics.
    """

    name: str
    active_interrupt_count: int
    completed_count: int
    escalated_count: int
    rejected_count: int
    dispatch_count: int
    acknowledgement_count: int
    last_reason: InterruptPropagationReason | None


class InterruptPropagator:
    """
    Phase 6 Interrupt Propagation System.

    Responsibilities:
    - propagate interrupt in dependency order
    - dispatch to one worker target at a time
    - require acknowledgement before next dispatch
    - escalate orphaned dispatches after timeout
    - keep rollback intent visible for Phase 5 rollback contracts

    Non-responsibilities:
    - no direct worker cancellation
    - no broadcast cancellation storm
    - no direct rollback execution
    - no task execution
    """

    def __init__(
        self,
        *,
        config: InterruptPropagatorConfig | None = None,
    ) -> None:
        self._config = config or InterruptPropagatorConfig()
        self._config.validate()

        self._records: dict[str, InterruptPropagationRecord] = {}
        self._lock = RLock()

        self._completed_count = 0
        self._escalated_count = 0
        self._rejected_count = 0
        self._last_reason: InterruptPropagationReason | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def start(
        self,
        *,
        event: InterruptEvent,
        order: PropagationOrder,
    ) -> InterruptPropagationResult:
        """
        Start interrupt propagation and dispatch the first target.
        """

        with self._lock:
            if event.interrupt_id in self._records:
                return self._reject(
                    reason=InterruptPropagationReason.INTERRUPT_ALREADY_ACTIVE,
                    message="interrupt is already active",
                    record=self._records[event.interrupt_id],
                )

            if not order.targets:
                return self._reject(
                    reason=InterruptPropagationReason.PROPAGATION_ORDER_EMPTY,
                    message="propagation order is empty",
                    record=None,
                )

            record = InterruptPropagationRecord(
                event=event,
                order=order,
                status=InterruptPropagationStatus.PROPAGATING,
            )
            self._records[event.interrupt_id] = record

        return self._dispatch_next(event.interrupt_id)

    def acknowledge(
        self,
        acknowledgement: InterruptAcknowledgement,
    ) -> InterruptPropagationResult:
        """
        Acknowledge the current dispatch and dispatch the next target.
        """

        with self._lock:
            record = self._records.get(acknowledgement.interrupt_id)

            if record is None:
                return self._reject(
                    reason=InterruptPropagationReason.INTERRUPT_NOT_FOUND,
                    message="interrupt not found",
                    record=None,
                )

            current = record.current_dispatch

            if current is None:
                return self._reject(
                    reason=InterruptPropagationReason.INVALID_ACKNOWLEDGEMENT,
                    message="no dispatch is waiting for acknowledgement",
                    record=record,
                )

            if current.dispatch_id != acknowledgement.dispatch_id:
                return self._reject(
                    reason=InterruptPropagationReason.WRONG_ORDER_ACKNOWLEDGED,
                    message="acknowledgement does not match current dispatch",
                    record=record,
                )

            if current.target.worker_id != acknowledgement.worker_id:
                return self._reject(
                    reason=InterruptPropagationReason.WRONG_WORKER_ACKNOWLEDGED,
                    message="wrong worker acknowledged interrupt",
                    record=record,
                )

            if current.acknowledged:
                return self._reject(
                    reason=(
                        InterruptPropagationReason
                        .DISPATCH_ALREADY_ACKNOWLEDGED
                    ),
                    message="dispatch already acknowledged",
                    record=record,
                )

            acknowledged_dispatch = current.model_copy(
                update={
                    "status": InterruptPropagationStatus.COMPLETED,
                    "acknowledged_at": acknowledgement.acknowledged_at,
                }
            )
            updated_dispatches = record.dispatches[:-1] + (
                acknowledged_dispatch,
            )
            next_index = record.current_index + 1
            next_status = (
                InterruptPropagationStatus.COMPLETED
                if next_index >= len(record.order.targets)
                else InterruptPropagationStatus.PROPAGATING
            )
            completed_at = (
                utc_now()
                if next_status == InterruptPropagationStatus.COMPLETED
                else None
            )
            updated_record = record.model_copy(
                update={
                    "dispatches": updated_dispatches,
                    "acknowledgements": (
                        record.acknowledgements + (acknowledgement,)
                    ),
                    "current_index": next_index,
                    "status": next_status,
                    "completed_at": completed_at,
                }
            )
            self._records[record.interrupt_id] = updated_record

            if next_status == InterruptPropagationStatus.COMPLETED:
                self._completed_count += 1
                return self._accept(
                    decision=InterruptPropagationDecision.COMPLETED,
                    reason=InterruptPropagationReason.INTERRUPT_COMPLETED,
                    message="interrupt propagation completed",
                    record=updated_record,
                    acknowledgement=acknowledgement,
                )

        dispatched = self._dispatch_next(acknowledgement.interrupt_id)

        return InterruptPropagationResult(
            decision=InterruptPropagationDecision.ACKNOWLEDGED,
            reason=InterruptPropagationReason.INTERRUPT_ACKNOWLEDGED,
            success=True,
            message="interrupt acknowledged; next dispatch prepared",
            record=dispatched.record,
            dispatch=dispatched.dispatch,
            acknowledgement=acknowledgement,
        )

    def escalate_orphans(
        self,
        *,
        interrupt_id: str,
        now: datetime | None = None,
        force: bool = False,
    ) -> InterruptPropagationResult:
        """
        Escalate a dispatch that has not been acknowledged before timeout.
        """

        with self._lock:
            record = self._records.get(interrupt_id)

            if record is None:
                return self._reject(
                    reason=InterruptPropagationReason.INTERRUPT_NOT_FOUND,
                    message="interrupt not found",
                    record=None,
                )

            current = record.current_dispatch

            if current is None:
                return self._reject(
                    reason=InterruptPropagationReason.INVALID_ACKNOWLEDGEMENT,
                    message="no dispatch is waiting for acknowledgement",
                    record=record,
                )

            if not force and not self._dispatch_timed_out(
                event=record.event,
                dispatch=current,
                now=now,
            ):
                return self._reject(
                    reason=InterruptPropagationReason.INVALID_ACKNOWLEDGEMENT,
                    message="dispatch timeout has not elapsed",
                    record=record,
                )

            orphan = OrphanedInterrupt(
                interrupt_id=record.interrupt_id,
                dispatch_id=current.dispatch_id,
                worker_id=current.target.worker_id,
                task_ids=current.target.task_ids,
                timeout_ms=record.event.timeout_ms,
            )
            updated_record = record.model_copy(
                update={
                    "status": InterruptPropagationStatus.ESCALATED,
                    "orphaned": record.orphaned + (orphan,),
                    "completed_at": utc_now(),
                }
            )
            self._records[record.interrupt_id] = updated_record
            self._escalated_count += 1

            return self._accept(
                decision=InterruptPropagationDecision.ESCALATED,
                reason=(
                    InterruptPropagationReason
                    .INTERRUPT_ESCALATED_ORPHANED_TASKS
                ),
                message="interrupt orphaned tasks escalated",
                record=updated_record,
                dispatch=current,
                orphaned=orphan,
            )

    def record_for(
        self,
        interrupt_id: str,
    ) -> InterruptPropagationRecord | None:
        with self._lock:
            return self._records.get(interrupt_id)

    def active_records(self) -> tuple[InterruptPropagationRecord, ...]:
        with self._lock:
            return tuple(
                record
                for record in self._records.values()
                if record.status
                in {
                    InterruptPropagationStatus.PROPAGATING,
                    InterruptPropagationStatus.WAITING_ACK,
                }
            )

    def all_records(self) -> tuple[InterruptPropagationRecord, ...]:
        with self._lock:
            return tuple(self._records.values())

    def snapshot(self) -> InterruptPropagatorSnapshot:
        with self._lock:
            records = tuple(self._records.values())

            return InterruptPropagatorSnapshot(
                name=self.name,
                active_interrupt_count=len(self.active_records()),
                completed_count=self._completed_count,
                escalated_count=self._escalated_count,
                rejected_count=self._rejected_count,
                dispatch_count=sum(len(record.dispatches) for record in records),
                acknowledgement_count=sum(
                    len(record.acknowledgements) for record in records
                ),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self._completed_count = 0
            self._escalated_count = 0
            self._rejected_count = 0
            self._last_reason = None

    def _dispatch_next(
        self,
        interrupt_id: str,
    ) -> InterruptPropagationResult:
        with self._lock:
            record = self._records[interrupt_id]
            target = record.next_target

            if target is None:
                completed = record.model_copy(
                    update={
                        "status": InterruptPropagationStatus.COMPLETED,
                        "completed_at": utc_now(),
                    }
                )
                self._records[interrupt_id] = completed
                self._completed_count += 1

                return self._accept(
                    decision=InterruptPropagationDecision.COMPLETED,
                    reason=InterruptPropagationReason.INTERRUPT_COMPLETED,
                    message="interrupt propagation completed",
                    record=completed,
                )

            dispatch = InterruptDispatch(
                interrupt_id=record.interrupt_id,
                target=target,
                metadata={
                    "rollback_requested": record.event.rollback_requested,
                    "rollback_supported": target.rollback_supported,
                },
            )
            updated_record = record.model_copy(
                update={
                    "dispatches": record.dispatches + (dispatch,),
                    "status": InterruptPropagationStatus.WAITING_ACK,
                }
            )
            self._records[interrupt_id] = updated_record

            return self._accept(
                decision=InterruptPropagationDecision.DISPATCHED,
                reason=InterruptPropagationReason.INTERRUPT_DISPATCHED,
                message="interrupt dispatched to next worker target",
                record=updated_record,
                dispatch=dispatch,
            )

    @staticmethod
    def _dispatch_timed_out(
        *,
        event: InterruptEvent,
        dispatch: InterruptDispatch,
        now: datetime | None,
    ) -> bool:
        if not isinstance(dispatch.sent_at, datetime):
            return False

        final_now = now or utc_now()

        if not isinstance(final_now, datetime):
            return False

        elapsed_ms = int((final_now - dispatch.sent_at).total_seconds() * 1000)

        return elapsed_ms >= event.timeout_ms

    def _accept(
        self,
        *,
        decision: InterruptPropagationDecision,
        reason: InterruptPropagationReason,
        message: str,
        record: InterruptPropagationRecord | None = None,
        dispatch: InterruptDispatch | None = None,
        acknowledgement: InterruptAcknowledgement | None = None,
        orphaned: OrphanedInterrupt | None = None,
    ) -> InterruptPropagationResult:
        self._last_reason = reason

        return InterruptPropagationResult(
            decision=decision,
            reason=reason,
            success=True,
            message=message,
            record=record,
            dispatch=dispatch,
            acknowledgement=acknowledgement,
            orphaned=orphaned,
        )

    def _reject(
        self,
        *,
        reason: InterruptPropagationReason,
        message: str,
        record: InterruptPropagationRecord | None,
    ) -> InterruptPropagationResult:
        self._last_reason = reason
        self._rejected_count += 1

        return InterruptPropagationResult(
            decision=InterruptPropagationDecision.REJECTED,
            reason=reason,
            success=False,
            message=message,
            record=record,
        )


def propagation_target_for_capability(
    *,
    capability: WorkerCapability,
    worker_id: WorkerId,
    task_ids: tuple[TaskId, ...],
    rollback_supported: bool = False,
) -> PropagationTarget:
    """
    Build a propagation target with the correct phase for a capability.
    """

    phase = _CAPABILITY_TO_PHASE.get(capability)

    if phase is None:
        raise ValueError("capability is not interrupt-propagation capable.")

    return PropagationTarget(
        phase=phase,
        capability=capability,
        worker_id=worker_id,
        task_ids=task_ids,
        rollback_supported=rollback_supported,
    )