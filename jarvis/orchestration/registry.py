from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import RLock

from pydantic import Field, field_validator, model_validator

from jarvis.orchestration.ids import WorkerId, utc_now, validate_worker_id
from jarvis.orchestration.models import (
    OrchestrationModel,
    ResourceBudget,
    TaskKind,
    TaskRequest,
    WorkerCapability,
    WorkerContract,
)


class WorkerHealthState(StrEnum):
    """
    Observable health state for a registered orchestration worker.
    """

    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    FAILED = "failed"
    DISABLED = "disabled"


class WorkerAvailability(StrEnum):
    """
    Worker availability for scheduling.
    """

    AVAILABLE = "available"
    BUSY = "busy"
    DRAINING = "draining"
    UNAVAILABLE = "unavailable"
    DISABLED = "disabled"


class WorkerRegistrationDecision(StrEnum):
    """
    Registry mutation decision.
    """

    REGISTERED = "registered"
    UPDATED = "updated"
    REJECTED = "rejected"


class WorkerLookupDecision(StrEnum):
    """
    Registry lookup decision.
    """

    FOUND = "found"
    NOT_FOUND = "not_found"


class WorkerRegistryReason(StrEnum):
    """
    Machine-readable registry reason.
    """

    WORKER_REGISTERED = "worker_registered"
    WORKER_UPDATED = "worker_updated"
    WORKER_NOT_FOUND = "worker_not_found"
    WORKER_DISABLED = "worker_disabled"
    WORKER_UNHEALTHY = "worker_unhealthy"
    WORKER_CAPABILITY_MISMATCH = "worker_capability_mismatch"
    WORKER_TASK_KIND_MISMATCH = "worker_task_kind_mismatch"
    WORKER_AT_CAPACITY = "worker_at_capacity"
    DUPLICATE_WORKER_REJECTED = "duplicate_worker_rejected"
    INVALID_WORKER_REJECTED = "invalid_worker_rejected"


class WorkerLoad(OrchestrationModel):
    """
    Runtime load snapshot for a worker.

    Load is observable state. It is not execution authority.
    """

    active_tasks: int = Field(default=0, ge=0)
    queued_tasks: int = Field(default=0, ge=0)
    max_concurrent_tasks: int = Field(gt=0)
    load_factor: float = Field(default=0.0, ge=0.0, le=1.0)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_load(self) -> WorkerLoad:
        expected = min(1.0, self.active_tasks / self.max_concurrent_tasks)

        if self.load_factor < expected:
            raise ValueError("load_factor cannot be lower than active capacity use.")

        return self

    @property
    def at_capacity(self) -> bool:
        return self.active_tasks >= self.max_concurrent_tasks

    @property
    def has_capacity(self) -> bool:
        return not self.at_capacity


class WorkerDescriptor(OrchestrationModel):
    """
    Registered worker descriptor.

    This is the Step 1 runtime view of a worker. It combines the immutable
    worker contract from Step 0 with health, load, availability, and enablement.

    Workers execute. The Orchestration Kernel coordinates.
    """

    contract: WorkerContract
    health: WorkerHealthState = WorkerHealthState.UNKNOWN
    availability: WorkerAvailability = WorkerAvailability.UNAVAILABLE
    load: WorkerLoad
    enabled: bool = True
    registered_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    last_heartbeat_at: object | None = None
    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_descriptor(self) -> WorkerDescriptor:
        if not self.enabled:
            if self.health != WorkerHealthState.DISABLED:
                raise ValueError("disabled workers must have DISABLED health.")

            if self.availability != WorkerAvailability.DISABLED:
                raise ValueError("disabled workers must have DISABLED availability.")

        if self.load.max_concurrent_tasks != self.contract.max_concurrent_tasks:
            raise ValueError(
                "worker load max_concurrent_tasks must match worker contract."
            )

        return self

    @property
    def worker_id(self) -> WorkerId:
        return self.contract.worker_id

    @property
    def name(self) -> str:
        return self.contract.name

    @property
    def capabilities(self) -> tuple[WorkerCapability, ...]:
        return self.contract.capabilities

    @property
    def accepted_task_kinds(self) -> tuple[TaskKind, ...]:
        return self.contract.accepted_task_kinds

    @property
    def resource_budgets(self) -> tuple[ResourceBudget, ...]:
        return self.contract.resource_budgets

    @property
    def healthy(self) -> bool:
        return self.health == WorkerHealthState.HEALTHY

    @property
    def schedulable(self) -> bool:
        return (
            self.enabled
            and self.health == WorkerHealthState.HEALTHY
            and self.availability == WorkerAvailability.AVAILABLE
            and self.load.has_capacity
        )

    def can_accept(self, task: TaskRequest) -> bool:
        """
        Return whether this descriptor can accept a task right now.
        """

        return self.schedulable and self.contract.can_accept(task)


class WorkerRegistrationResult(OrchestrationModel):
    """
    Result of a worker registry mutation.
    """

    worker_id: WorkerId | None = None
    decision: WorkerRegistrationDecision
    reason: WorkerRegistryReason
    success: bool
    message: str
    descriptor: WorkerDescriptor | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("worker_id")
    @classmethod
    def _validate_worker_id(cls, value: str | None) -> str | None:
        if value is None:
            return None

        return validate_worker_id(value)

    @field_validator("message")
    @classmethod
    def _required_message(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("message cannot be empty.")

        return cleaned


class WorkerLookupResult(OrchestrationModel):
    """
    Result of a worker registry lookup.
    """

    worker_id: WorkerId
    decision: WorkerLookupDecision
    reason: WorkerRegistryReason
    found: bool
    descriptor: WorkerDescriptor | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("worker_id")
    @classmethod
    def _validate_worker_id(cls, value: str) -> str:
        return validate_worker_id(value)


@dataclass(frozen=True, slots=True)
class WorkerRegistryConfig:
    """
    Worker registry configuration.
    """

    name: str = "worker_registry"
    allow_replace: bool = False

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")


@dataclass(frozen=True, slots=True)
class WorkerRegistrySnapshot:
    """
    Observable worker registry diagnostics.
    """

    name: str
    worker_count: int
    enabled_count: int
    healthy_count: int
    schedulable_count: int
    busy_count: int
    disabled_count: int
    unhealthy_count: int
    last_updated_at: object | None


class WorkerRegistry:
    """
    Phase 6 Worker Registry Runtime.

    Responsibilities:
    - register every worker explicitly
    - expose worker capabilities, health, load, and availability
    - reject hidden or duplicate workers
    - find schedulable workers for tasks
    - provide observable registry snapshots

    Non-responsibilities:
    - no task execution
    - no scheduling decisions beyond capability matching
    - no worker-to-worker communication
    - no direct runtime side effects
    """

    def __init__(self, *, config: WorkerRegistryConfig | None = None) -> None:
        self._config = config or WorkerRegistryConfig()
        self._config.validate()

        self._workers: dict[WorkerId, WorkerDescriptor] = {}
        self._lock = RLock()
        self._last_updated_at: object | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def register(
        self,
        descriptor: WorkerDescriptor,
    ) -> WorkerRegistrationResult:
        """
        Register a worker descriptor.

        Duplicate workers are rejected unless allow_replace=True.
        """

        with self._lock:
            existing = self._workers.get(descriptor.worker_id)

            if existing is not None and not self._config.allow_replace:
                return WorkerRegistrationResult(
                    worker_id=descriptor.worker_id,
                    decision=WorkerRegistrationDecision.REJECTED,
                    reason=WorkerRegistryReason.DUPLICATE_WORKER_REJECTED,
                    success=False,
                    message="worker already registered",
                    descriptor=existing,
                )

            decision = (
                WorkerRegistrationDecision.UPDATED
                if existing is not None
                else WorkerRegistrationDecision.REGISTERED
            )
            reason = (
                WorkerRegistryReason.WORKER_UPDATED
                if existing is not None
                else WorkerRegistryReason.WORKER_REGISTERED
            )
            self._workers[descriptor.worker_id] = descriptor
            self._last_updated_at = utc_now()

            return WorkerRegistrationResult(
                worker_id=descriptor.worker_id,
                decision=decision,
                reason=reason,
                success=True,
                message="worker registered",
                descriptor=descriptor,
            )

    def get(self, worker_id: WorkerId) -> WorkerLookupResult:
        """
        Look up a worker by id.
        """

        validated_worker_id = validate_worker_id(worker_id)

        with self._lock:
            descriptor = self._workers.get(validated_worker_id)

        if descriptor is None:
            return WorkerLookupResult(
                worker_id=validated_worker_id,
                decision=WorkerLookupDecision.NOT_FOUND,
                reason=WorkerRegistryReason.WORKER_NOT_FOUND,
                found=False,
            )

        return WorkerLookupResult(
            worker_id=validated_worker_id,
            decision=WorkerLookupDecision.FOUND,
            reason=WorkerRegistryReason.WORKER_REGISTERED,
            found=True,
            descriptor=descriptor,
        )

    def all_workers(self) -> tuple[WorkerDescriptor, ...]:
        """
        Return all registered workers.
        """

        with self._lock:
            return tuple(self._workers.values())

    def enabled_workers(self) -> tuple[WorkerDescriptor, ...]:
        """
        Return enabled workers.
        """

        return tuple(worker for worker in self.all_workers() if worker.enabled)

    def healthy_workers(self) -> tuple[WorkerDescriptor, ...]:
        """
        Return healthy workers.
        """

        return tuple(worker for worker in self.all_workers() if worker.healthy)

    def schedulable_workers(self) -> tuple[WorkerDescriptor, ...]:
        """
        Return workers currently schedulable.
        """

        return tuple(worker for worker in self.all_workers() if worker.schedulable)

    def find_by_capability(
        self,
        capability: WorkerCapability,
    ) -> tuple[WorkerDescriptor, ...]:
        """
        Find schedulable workers exposing a capability.
        """

        return tuple(
            worker
            for worker in self.schedulable_workers()
            if capability in worker.capabilities
        )

    def find_for_task(
        self,
        task: TaskRequest,
    ) -> tuple[WorkerDescriptor, ...]:
        """
        Find schedulable workers that can accept the task.
        """

        return tuple(
            worker for worker in self.schedulable_workers() if worker.can_accept(task)
        )

    def update_health(
        self,
        worker_id: WorkerId,
        health: WorkerHealthState,
        *,
        availability: WorkerAvailability | None = None,
    ) -> WorkerRegistrationResult:
        """
        Update worker health and optional availability.
        """

        with self._lock:
            current = self._workers.get(validate_worker_id(worker_id))

            if current is None:
                return WorkerRegistrationResult(
                    worker_id=worker_id,
                    decision=WorkerRegistrationDecision.REJECTED,
                    reason=WorkerRegistryReason.WORKER_NOT_FOUND,
                    success=False,
                    message="worker not found",
                )

            next_availability = availability or self._availability_for_health(health)
            updated = current.model_copy(
                update={
                    "health": health,
                    "availability": next_availability,
                    "updated_at": utc_now(),
                    "last_heartbeat_at": utc_now(),
                }
            )
            self._workers[current.worker_id] = updated
            self._last_updated_at = utc_now()

            return WorkerRegistrationResult(
                worker_id=current.worker_id,
                decision=WorkerRegistrationDecision.UPDATED,
                reason=WorkerRegistryReason.WORKER_UPDATED,
                success=True,
                message="worker health updated",
                descriptor=updated,
            )

    def update_load(
        self,
        worker_id: WorkerId,
        *,
        active_tasks: int,
        queued_tasks: int = 0,
    ) -> WorkerRegistrationResult:
        """
        Update worker load.
        """

        with self._lock:
            current = self._workers.get(validate_worker_id(worker_id))

            if current is None:
                return WorkerRegistrationResult(
                    worker_id=worker_id,
                    decision=WorkerRegistrationDecision.REJECTED,
                    reason=WorkerRegistryReason.WORKER_NOT_FOUND,
                    success=False,
                    message="worker not found",
                )

            load_factor = min(
                1.0,
                active_tasks / current.contract.max_concurrent_tasks,
            )
            next_availability = (
                WorkerAvailability.BUSY
                if active_tasks >= current.contract.max_concurrent_tasks
                else self._availability_for_health(current.health)
            )
            updated_load = WorkerLoad(
                active_tasks=active_tasks,
                queued_tasks=queued_tasks,
                max_concurrent_tasks=current.contract.max_concurrent_tasks,
                load_factor=load_factor,
            )
            updated = current.model_copy(
                update={
                    "load": updated_load,
                    "availability": next_availability,
                    "updated_at": utc_now(),
                }
            )
            self._workers[current.worker_id] = updated
            self._last_updated_at = utc_now()

            return WorkerRegistrationResult(
                worker_id=current.worker_id,
                decision=WorkerRegistrationDecision.UPDATED,
                reason=WorkerRegistryReason.WORKER_UPDATED,
                success=True,
                message="worker load updated",
                descriptor=updated,
            )

    def disable(self, worker_id: WorkerId) -> WorkerRegistrationResult:
        """
        Disable a registered worker.
        """

        with self._lock:
            current = self._workers.get(validate_worker_id(worker_id))

            if current is None:
                return WorkerRegistrationResult(
                    worker_id=worker_id,
                    decision=WorkerRegistrationDecision.REJECTED,
                    reason=WorkerRegistryReason.WORKER_NOT_FOUND,
                    success=False,
                    message="worker not found",
                )

            updated = current.model_copy(
                update={
                    "enabled": False,
                    "health": WorkerHealthState.DISABLED,
                    "availability": WorkerAvailability.DISABLED,
                    "updated_at": utc_now(),
                }
            )
            self._workers[current.worker_id] = updated
            self._last_updated_at = utc_now()

            return WorkerRegistrationResult(
                worker_id=current.worker_id,
                decision=WorkerRegistrationDecision.UPDATED,
                reason=WorkerRegistryReason.WORKER_DISABLED,
                success=True,
                message="worker disabled",
                descriptor=updated,
            )

    def snapshot(self) -> WorkerRegistrySnapshot:
        """
        Return registry diagnostics.
        """

        workers = self.all_workers()

        return WorkerRegistrySnapshot(
            name=self.name,
            worker_count=len(workers),
            enabled_count=sum(1 for worker in workers if worker.enabled),
            healthy_count=sum(1 for worker in workers if worker.healthy),
            schedulable_count=sum(1 for worker in workers if worker.schedulable),
            busy_count=sum(
                1
                for worker in workers
                if worker.availability == WorkerAvailability.BUSY
            ),
            disabled_count=sum(1 for worker in workers if not worker.enabled),
            unhealthy_count=sum(
                1
                for worker in workers
                if worker.health
                in {
                    WorkerHealthState.DEGRADED,
                    WorkerHealthState.UNHEALTHY,
                    WorkerHealthState.FAILED,
                }
            ),
            last_updated_at=self._last_updated_at,
        )

    def reset(self) -> None:
        """
        Clear registered workers.

        Used for deterministic test and bootstrap lifecycle boundaries.
        """

        with self._lock:
            self._workers.clear()
            self._last_updated_at = None

    @staticmethod
    def descriptor_from_contract(
        contract: WorkerContract,
        *,
        health: WorkerHealthState = WorkerHealthState.HEALTHY,
        availability: WorkerAvailability | None = None,
        enabled: bool = True,
    ) -> WorkerDescriptor:
        """
        Build a descriptor from a Step 0 WorkerContract.
        """

        final_health = health if enabled else WorkerHealthState.DISABLED
        final_availability = (
            availability
            if availability is not None
            else WorkerRegistry._availability_for_health(final_health)
        )

        if not enabled:
            final_availability = WorkerAvailability.DISABLED

        return WorkerDescriptor(
            contract=contract,
            health=final_health,
            availability=final_availability,
            enabled=enabled,
            load=WorkerLoad(
                active_tasks=0,
                queued_tasks=0,
                max_concurrent_tasks=contract.max_concurrent_tasks,
                load_factor=0.0,
            ),
        )

    @staticmethod
    def _availability_for_health(
        health: WorkerHealthState,
    ) -> WorkerAvailability:
        if health == WorkerHealthState.HEALTHY:
            return WorkerAvailability.AVAILABLE

        if health == WorkerHealthState.DISABLED:
            return WorkerAvailability.DISABLED

        if health in {
            WorkerHealthState.UNKNOWN,
            WorkerHealthState.DEGRADED,
            WorkerHealthState.UNHEALTHY,
            WorkerHealthState.FAILED,
        }:
            return WorkerAvailability.UNAVAILABLE

        return WorkerAvailability.UNAVAILABLE