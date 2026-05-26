from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import RLock

from pydantic import Field, field_validator, model_validator

from jarvis.orchestration.ids import TaskId, utc_now, validate_task_id
from jarvis.orchestration.models import (
    BudgetPolicy,
    OrchestrationModel,
    ResourceBudget,
    ResourceKind,
    TaskKind,
    TaskRequest,
)


class BudgetDecision(StrEnum):
    """
    Resource budget decision.
    """

    ALLOW = "allow"
    WARN = "warn"
    DENY = "deny"
    SHED = "shed"


class BudgetReason(StrEnum):
    """
    Machine-readable budget reason.
    """

    BUDGET_AVAILABLE = "budget_available"
    BUDGET_WARNING = "budget_warning"
    BUDGET_EXCEEDED = "budget_exceeded"
    BUDGET_SHED_REQUIRED = "budget_shed_required"
    TASK_BUDGET_MISSING = "task_budget_missing"
    RESOURCE_NOT_CONFIGURED = "resource_not_configured"
    CONVERSATION_RESERVE_PROTECTED = "conversation_reserve_protected"
    INVALID_RELEASE = "invalid_release"
    RESERVATION_NOT_FOUND = "reservation_not_found"
    RESERVATION_CREATED = "reservation_created"
    RESERVATION_RELEASED = "reservation_released"


class BudgetReservationStatus(StrEnum):
    """
    Lifecycle state for a resource reservation.
    """

    RESERVED = "reserved"
    RELEASED = "released"
    DENIED = "denied"


class ResourcePool(OrchestrationModel):
    """
    Runtime pool for one resource kind.

    The pool tracks capacity, reserved amount, and optional conversation reserve.
    Conversation reserve protects real-time responsiveness.
    """

    resource: ResourceKind
    capacity: int = Field(gt=0)
    reserved: int = Field(default=0, ge=0)
    conversation_reserved: int = Field(default=0, ge=0)
    policy: BudgetPolicy = BudgetPolicy.ENFORCE
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_pool(self) -> ResourcePool:
        if self.reserved > self.capacity:
            raise ValueError("reserved amount cannot exceed capacity.")

        if self.conversation_reserved > self.capacity:
            raise ValueError("conversation reserve cannot exceed capacity.")

        return self

    @property
    def available(self) -> int:
        return max(0, self.capacity - self.reserved)

    @property
    def non_reserved_available(self) -> int:
        protected_available = self.capacity - self.conversation_reserved

        return max(0, protected_available - self.reserved)

    def can_reserve(
        self,
        amount: int,
        *,
        protect_conversation: bool,
        conversation_task: bool,
    ) -> bool:
        """
        Return whether the pool can reserve the requested amount.
        """

        if amount <= 0:
            return False

        if not protect_conversation or conversation_task:
            return amount <= self.available

        return amount <= self.non_reserved_available


class BudgetReservation(OrchestrationModel):
    """
    Concrete resource reservation for a task.

    Reservations are explicit and later released by task_id.
    """

    task_id: TaskId
    resources: tuple[ResourceBudget, ...]
    status: BudgetReservationStatus = BudgetReservationStatus.RESERVED
    created_at: object = Field(default_factory=utc_now)
    released_at: object | None = None
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("task_id")
    @classmethod
    def _validate_task_id(cls, value: str) -> str:
        return validate_task_id(value)

    @model_validator(mode="after")
    def _validate_reservation(self) -> BudgetReservation:
        if not self.resources:
            raise ValueError("reservation requires at least one resource.")

        if self.status == BudgetReservationStatus.RELEASED:
            if self.released_at is None:
                raise ValueError("released reservations require released_at.")

        return self


class BudgetEvaluation(OrchestrationModel):
    """
    Result of evaluating a task resource request.
    """

    task_id: TaskId
    decision: BudgetDecision
    reason: BudgetReason
    allowed: bool
    requested: tuple[ResourceBudget, ...]
    denied_resources: tuple[ResourceKind, ...] = ()
    warning_resources: tuple[ResourceKind, ...] = ()
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("task_id")
    @classmethod
    def _validate_task_id(cls, value: str) -> str:
        return validate_task_id(value)


class BudgetReservationResult(OrchestrationModel):
    """
    Result of reserving or releasing resources.
    """

    task_id: TaskId
    decision: BudgetDecision
    reason: BudgetReason
    success: bool
    message: str
    reservation: BudgetReservation | None = None
    pools: tuple[ResourcePool, ...] = ()
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("task_id")
    @classmethod
    def _validate_task_id(cls, value: str) -> str:
        return validate_task_id(value)

    @field_validator("message")
    @classmethod
    def _required_message(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("message cannot be empty.")

        return cleaned


class BudgetRuntimePolicy(OrchestrationModel):
    """
    Runtime-level budget policy.
    """

    protect_conversation_reserve: bool = True
    require_task_budgets: bool = True
    allow_unconfigured_resources: bool = False
    warning_threshold_percent: int = Field(default=80, ge=1, le=100)


@dataclass(frozen=True, slots=True)
class ResourceBudgetRuntimeConfig:
    """
    Resource budget runtime configuration.
    """

    name: str = "resource_budget_runtime"

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")


@dataclass(frozen=True, slots=True)
class ResourceBudgetRuntimeSnapshot:
    """
    Resource budget runtime diagnostics.
    """

    name: str
    pool_count: int
    reservation_count: int
    total_capacity: int
    total_reserved: int
    evaluation_count: int
    allow_count: int
    warn_count: int
    deny_count: int
    last_decision: BudgetDecision | None
    last_reason: BudgetReason | None


class ResourceBudgetRuntime:
    """
    Phase 6 Resource Budget Runtime.

    Responsibilities:
    - hold configured resource pools
    - evaluate task-declared resource budgets
    - reserve resources before scheduling
    - release resources after task completion/cancellation/failure
    - protect conversation reserve

    Non-responsibilities:
    - no task execution
    - no worker execution
    - no scheduling order
    - no attention decisions
    """

    def __init__(
        self,
        *,
        config: ResourceBudgetRuntimeConfig | None = None,
        policy: BudgetRuntimePolicy | None = None,
        pools: tuple[ResourcePool, ...] | None = None,
    ) -> None:
        self._config = config or ResourceBudgetRuntimeConfig()
        self._config.validate()

        self._policy = policy or BudgetRuntimePolicy()
        self._pools: dict[ResourceKind, ResourcePool] = {}
        self._reservations: dict[TaskId, BudgetReservation] = {}
        self._lock = RLock()

        self._evaluation_count = 0
        self._allow_count = 0
        self._warn_count = 0
        self._deny_count = 0
        self._last_decision: BudgetDecision | None = None
        self._last_reason: BudgetReason | None = None

        for pool in pools or self.default_pools():
            self._pools[pool.resource] = pool

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def policy(self) -> BudgetRuntimePolicy:
        return self._policy

    @staticmethod
    def default_pools() -> tuple[ResourcePool, ...]:
        """
        Conservative local defaults for the Phase 6 foundation.
        """

        return (
            ResourcePool(
                resource=ResourceKind.CPU_SLOT,
                capacity=4,
                conversation_reserved=1,
            ),
            ResourcePool(
                resource=ResourceKind.WORKER_SLOT,
                capacity=8,
                conversation_reserved=2,
            ),
            ResourcePool(
                resource=ResourceKind.LLM_TOKEN,
                capacity=32_000,
                conversation_reserved=8_000,
            ),
            ResourcePool(
                resource=ResourceKind.MEMORY_QUERY,
                capacity=16,
                conversation_reserved=4,
            ),
            ResourcePool(
                resource=ResourceKind.MEMORY_WRITE,
                capacity=4,
                conversation_reserved=1,
            ),
            ResourcePool(
                resource=ResourceKind.ACTION_SLOT,
                capacity=2,
                conversation_reserved=1,
            ),
            ResourcePool(
                resource=ResourceKind.FILE_OPERATION,
                capacity=8,
                conversation_reserved=1,
            ),
            ResourcePool(
                resource=ResourceKind.NETWORK_OPERATION,
                capacity=2,
                conversation_reserved=0,
            ),
            ResourcePool(
                resource=ResourceKind.WALL_TIME_MS,
                capacity=300_000,
                conversation_reserved=30_000,
            ),
        )

    def configure_pool(self, pool: ResourcePool) -> None:
        """
        Configure or replace a resource pool.
        """

        with self._lock:
            self._pools[pool.resource] = pool

    def pool_for(self, resource: ResourceKind) -> ResourcePool | None:
        """
        Return a resource pool by kind.
        """

        with self._lock:
            return self._pools.get(resource)

    def all_pools(self) -> tuple[ResourcePool, ...]:
        """
        Return all pools.
        """

        with self._lock:
            return tuple(self._pools.values())

    def evaluate(self, task: TaskRequest) -> BudgetEvaluation:
        """
        Evaluate whether a task's declared budgets can be reserved.
        """

        with self._lock:
            evaluation = self._evaluate_locked(task)

        self._record_evaluation(evaluation)

        return evaluation

    def reserve(self, task: TaskRequest) -> BudgetReservationResult:
        """
        Reserve resources for a task.

        The scheduler must call this before assignment.
        """

        with self._lock:
            evaluation = self._evaluate_locked(task)

            if not evaluation.allowed:
                result = BudgetReservationResult(
                    task_id=task.task_id,
                    decision=evaluation.decision,
                    reason=evaluation.reason,
                    success=False,
                    message="resource reservation denied",
                    pools=tuple(self._pools.values()),
                    metadata={
                        "denied_resources": tuple(
                            resource.value
                            for resource in evaluation.denied_resources
                        )
                    },
                )
                self._record_evaluation(evaluation)

                return result

            if task.task_id in self._reservations:
                existing = self._reservations[task.task_id]

                return BudgetReservationResult(
                    task_id=task.task_id,
                    decision=BudgetDecision.ALLOW,
                    reason=BudgetReason.RESERVATION_CREATED,
                    success=True,
                    message="resource reservation already exists",
                    reservation=existing,
                    pools=tuple(self._pools.values()),
                )

            next_pools = dict(self._pools)

            for requested in task.resource_budgets:
                current = next_pools[requested.resource]
                next_pools[requested.resource] = current.model_copy(
                    update={
                        "reserved": current.reserved + requested.amount,
                        "updated_at": utc_now(),
                    }
                )

            reservation = BudgetReservation(
                task_id=task.task_id,
                resources=task.resource_budgets,
            )
            self._pools = next_pools
            self._reservations[task.task_id] = reservation

            success_evaluation = BudgetEvaluation(
                task_id=task.task_id,
                decision=BudgetDecision.ALLOW,
                reason=BudgetReason.RESERVATION_CREATED,
                allowed=True,
                requested=task.resource_budgets,
            )
            self._record_evaluation(success_evaluation)

            return BudgetReservationResult(
                task_id=task.task_id,
                decision=BudgetDecision.ALLOW,
                reason=BudgetReason.RESERVATION_CREATED,
                success=True,
                message="resource reservation created",
                reservation=reservation,
                pools=tuple(self._pools.values()),
            )

    def release(self, task_id: TaskId) -> BudgetReservationResult:
        """
        Release resources reserved for a task.
        """

        validated_task_id = validate_task_id(task_id)

        with self._lock:
            reservation = self._reservations.get(validated_task_id)

            if reservation is None:
                return BudgetReservationResult(
                    task_id=validated_task_id,
                    decision=BudgetDecision.DENY,
                    reason=BudgetReason.RESERVATION_NOT_FOUND,
                    success=False,
                    message="resource reservation not found",
                    pools=tuple(self._pools.values()),
                )

            next_pools = dict(self._pools)

            for requested in reservation.resources:
                current = next_pools.get(requested.resource)

                if current is None or current.reserved < requested.amount:
                    return BudgetReservationResult(
                        task_id=validated_task_id,
                        decision=BudgetDecision.DENY,
                        reason=BudgetReason.INVALID_RELEASE,
                        success=False,
                        message="resource reservation cannot be released safely",
                        reservation=reservation,
                        pools=tuple(self._pools.values()),
                    )

                next_pools[requested.resource] = current.model_copy(
                    update={
                        "reserved": current.reserved - requested.amount,
                        "updated_at": utc_now(),
                    }
                )

            released = reservation.model_copy(
                update={
                    "status": BudgetReservationStatus.RELEASED,
                    "released_at": utc_now(),
                }
            )
            self._pools = next_pools
            self._reservations.pop(validated_task_id)

            return BudgetReservationResult(
                task_id=validated_task_id,
                decision=BudgetDecision.ALLOW,
                reason=BudgetReason.RESERVATION_RELEASED,
                success=True,
                message="resource reservation released",
                reservation=released,
                pools=tuple(self._pools.values()),
            )

    def reservation_for(self, task_id: TaskId) -> BudgetReservation | None:
        """
        Return an active reservation.
        """

        validated_task_id = validate_task_id(task_id)

        with self._lock:
            return self._reservations.get(validated_task_id)

    def snapshot(self) -> ResourceBudgetRuntimeSnapshot:
        """
        Return runtime diagnostics.
        """

        with self._lock:
            pools = tuple(self._pools.values())

            return ResourceBudgetRuntimeSnapshot(
                name=self.name,
                pool_count=len(pools),
                reservation_count=len(self._reservations),
                total_capacity=sum(pool.capacity for pool in pools),
                total_reserved=sum(pool.reserved for pool in pools),
                evaluation_count=self._evaluation_count,
                allow_count=self._allow_count,
                warn_count=self._warn_count,
                deny_count=self._deny_count,
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
            self._warn_count = 0
            self._deny_count = 0
            self._last_decision = None
            self._last_reason = None

    def _evaluate_locked(self, task: TaskRequest) -> BudgetEvaluation:
        if self._policy.require_task_budgets and not task.resource_budgets:
            return BudgetEvaluation(
                task_id=task.task_id,
                decision=BudgetDecision.DENY,
                reason=BudgetReason.TASK_BUDGET_MISSING,
                allowed=False,
                requested=(),
            )

        denied: list[ResourceKind] = []
        warnings: list[ResourceKind] = []
        conversation_task = task.kind == TaskKind.CONVERSATION_TURN

        for requested in task.resource_budgets:
            pool = self._pools.get(requested.resource)

            if pool is None:
                if self._policy.allow_unconfigured_resources:
                    warnings.append(requested.resource)
                    continue

                denied.append(requested.resource)
                continue

            can_reserve = pool.can_reserve(
                requested.amount,
                protect_conversation=self._policy.protect_conversation_reserve,
                conversation_task=conversation_task,
            )

            if not can_reserve:
                denied.append(requested.resource)
                continue

            projected_reserved = pool.reserved + requested.amount
            projected_percent = int((projected_reserved / pool.capacity) * 100)

            if projected_percent >= self._policy.warning_threshold_percent:
                warnings.append(requested.resource)

        if denied:
            return BudgetEvaluation(
                task_id=task.task_id,
                decision=BudgetDecision.DENY,
                reason=self._reason_for_denial(task=task, denied=tuple(denied)),
                allowed=False,
                requested=task.resource_budgets,
                denied_resources=tuple(denied),
                warning_resources=tuple(warnings),
            )

        if warnings:
            return BudgetEvaluation(
                task_id=task.task_id,
                decision=BudgetDecision.WARN,
                reason=BudgetReason.BUDGET_WARNING,
                allowed=True,
                requested=task.resource_budgets,
                warning_resources=tuple(warnings),
            )

        return BudgetEvaluation(
            task_id=task.task_id,
            decision=BudgetDecision.ALLOW,
            reason=BudgetReason.BUDGET_AVAILABLE,
            allowed=True,
            requested=task.resource_budgets,
        )

    def _reason_for_denial(
        self,
        *,
        task: TaskRequest,
        denied: tuple[ResourceKind, ...],
    ) -> BudgetReason:
        if any(resource not in self._pools for resource in denied):
            return BudgetReason.RESOURCE_NOT_CONFIGURED

        if task.kind != TaskKind.CONVERSATION_TURN:
            return BudgetReason.CONVERSATION_RESERVE_PROTECTED

        return BudgetReason.BUDGET_EXCEEDED

    def _record_evaluation(self, evaluation: BudgetEvaluation) -> None:
        with self._lock:
            self._evaluation_count += 1
            self._last_decision = evaluation.decision
            self._last_reason = evaluation.reason

            if evaluation.decision == BudgetDecision.ALLOW:
                self._allow_count += 1

            elif evaluation.decision == BudgetDecision.WARN:
                self._warn_count += 1

            else:
                self._deny_count += 1


def cpu_slots(amount: int) -> ResourceBudget:
    return ResourceBudget(resource=ResourceKind.CPU_SLOT, amount=amount)


def worker_slots(amount: int) -> ResourceBudget:
    return ResourceBudget(resource=ResourceKind.WORKER_SLOT, amount=amount)


def llm_tokens(amount: int) -> ResourceBudget:
    return ResourceBudget(resource=ResourceKind.LLM_TOKEN, amount=amount)


def memory_queries(amount: int) -> ResourceBudget:
    return ResourceBudget(resource=ResourceKind.MEMORY_QUERY, amount=amount)


def memory_writes(amount: int) -> ResourceBudget:
    return ResourceBudget(resource=ResourceKind.MEMORY_WRITE, amount=amount)


def action_slots(amount: int) -> ResourceBudget:
    return ResourceBudget(resource=ResourceKind.ACTION_SLOT, amount=amount)


def file_operations(amount: int) -> ResourceBudget:
    return ResourceBudget(resource=ResourceKind.FILE_OPERATION, amount=amount)


def network_operations(amount: int) -> ResourceBudget:
    return ResourceBudget(resource=ResourceKind.NETWORK_OPERATION, amount=amount)


def wall_time_ms(amount: int) -> ResourceBudget:
    return ResourceBudget(resource=ResourceKind.WALL_TIME_MS, amount=amount)