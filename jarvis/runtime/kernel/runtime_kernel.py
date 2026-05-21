from __future__ import annotations

from dataclasses import dataclass

from jarvis.runtime.config import RuntimeSettings, get_settings
from jarvis.runtime.events import EventBus, EventBusSnapshot
from jarvis.runtime.kernel.cancellation_manager import (
    CancellationManager,
    CancellationManagerSnapshot,
)
from jarvis.runtime.kernel.health_monitor import HealthMonitor, RuntimeHealthSnapshot
from jarvis.runtime.kernel.lifecycle_manager import LifecycleManager
from jarvis.runtime.kernel.scheduler import Scheduler, SchedulerSnapshot
from jarvis.runtime.observability.structured_logger import get_logger
from jarvis.runtime.security import (
    AuditLogger,
    IdentityManager,
    PermissionEngine,
    PolicyEngine,
)
from jarvis.runtime.state import StateEngine, StateSnapshot
from jarvis.runtime.workers import BaseWorker, WorkerManager, WorkerManagerSnapshot


@dataclass(frozen=True, slots=True)
class RuntimeKernelSnapshot:
    settings: RuntimeSettings
    event_bus: EventBusSnapshot
    state: StateSnapshot
    workers: WorkerManagerSnapshot
    scheduler: SchedulerSnapshot
    cancellations: CancellationManagerSnapshot
    health: RuntimeHealthSnapshot


class RuntimeKernel:
    """
    Central runtime kernel for JARVIS.

    This wires the core operating runtime:
    - configuration
    - EventBus
    - StateEngine
    - Security runtime
    - WorkerManager
    - Scheduler
    - CancellationManager
    - HealthMonitor
    - LifecycleManager
    """

    def __init__(
        self,
        *,
        settings: RuntimeSettings | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self.settings = settings or get_settings()

        self.event_bus = event_bus or EventBus(
            name="kernel_event_bus",
            history_limit=self.settings.workers.event_queue_size,
            dead_letter_limit=self.settings.workers.event_queue_size,
        )

        self.state_engine = StateEngine(event_bus=self.event_bus)

        self.identity_manager = IdentityManager()
        self.policy_engine = PolicyEngine(
            deny_unknown_actions=self.settings.security.deny_unknown_actions,
        )
        self.audit_logger = AuditLogger()
        self.permission_engine = PermissionEngine(
            event_bus=self.event_bus,
            identity_manager=self.identity_manager,
            policy_engine=self.policy_engine,
            audit_logger=self.audit_logger,
        )

        self.worker_manager = WorkerManager(event_bus=self.event_bus)

        self.scheduler = Scheduler(
            event_bus=self.event_bus,
            tick_interval_seconds=self.settings.observability.health_check_interval_seconds,
        )

        self.cancellation_manager = CancellationManager()
        self.health_monitor = HealthMonitor()

        self.lifecycle_manager = LifecycleManager(
            event_bus=self.event_bus,
            state_engine=self.state_engine,
            worker_manager=self.worker_manager,
        )

        self._logger = get_logger("kernel.runtime_kernel")

        self.worker_manager.register(self.scheduler)

    def start(self) -> None:
        self._logger.info("runtime_kernel_start_requested")
        self.lifecycle_manager.start()

    def stop(self) -> None:
        self._logger.info("runtime_kernel_stop_requested")
        self.lifecycle_manager.stop()

    def register_worker(self, worker: BaseWorker) -> None:
        self.worker_manager.register(worker)

    def snapshot(self) -> RuntimeKernelSnapshot:
        event_bus_snapshot = self.event_bus.snapshot()
        state_snapshot = self.state_engine.snapshot()
        worker_snapshot = self.worker_manager.snapshot()
        scheduler_snapshot = self.scheduler.scheduler_snapshot()
        cancellation_snapshot = self.cancellation_manager.snapshot()

        health_snapshot = self.health_monitor.check(
            event_bus_snapshot=event_bus_snapshot,
            worker_manager_snapshot=worker_snapshot,
            state_snapshot=state_snapshot,
        )

        return RuntimeKernelSnapshot(
            settings=self.settings,
            event_bus=event_bus_snapshot,
            state=state_snapshot,
            workers=worker_snapshot,
            scheduler=scheduler_snapshot,
            cancellations=cancellation_snapshot,
            health=health_snapshot,
        )