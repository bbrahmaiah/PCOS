from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from jarvis.runtime.events import EventBus
from jarvis.runtime.observability.structured_logger import get_logger
from jarvis.runtime.shared.enums import WorkerStatus
from jarvis.runtime.workers.worker import BaseWorker, WorkerSnapshot
from jarvis.runtime.workers.worker_registry import WorkerRegistry


@dataclass(frozen=True, slots=True)
class WorkerManagerSnapshot:
    worker_count: int
    running_count: int
    failed_count: int
    workers: tuple[WorkerSnapshot, ...]


class WorkerManager:
    """
    Coordinates worker registration and lifecycle.

    The manager owns orchestration.
    Individual workers own execution.
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        registry: WorkerRegistry | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.registry = registry or WorkerRegistry()
        self._lock = RLock()
        self._logger = get_logger("workers.manager")

    def register(self, worker: BaseWorker) -> None:
        self.registry.register(worker)

        self._logger.info(
            "worker_registered",
            worker=worker.name,
        )

    def unregister(self, worker_name: str, *, stop_first: bool = True) -> BaseWorker:
        worker = self.registry.require(worker_name)

        if stop_first:
            worker.stop()

        removed = self.registry.unregister(worker_name)

        self._logger.info(
            "worker_unregistered",
            worker=removed.name,
        )

        return removed

    def start_worker(self, worker_name: str) -> None:
        worker = self.registry.require(worker_name)
        worker.start()

        self._logger.info(
            "worker_start_dispatched",
            worker=worker.name,
        )

    def stop_worker(self, worker_name: str) -> None:
        worker = self.registry.require(worker_name)
        worker.stop()

        self._logger.info(
            "worker_stop_dispatched",
            worker=worker.name,
        )

    def start_all(self) -> None:
        with self._lock:
            workers = self.registry.workers()

        for worker in workers:
            worker.start()

        self._logger.info(
            "all_workers_start_dispatched",
            worker_count=len(workers),
        )

    def stop_all(self) -> None:
        with self._lock:
            workers = self.registry.workers()

        for worker in reversed(workers):
            worker.stop()

        self._logger.info(
            "all_workers_stop_dispatched",
            worker_count=len(workers),
        )

    def snapshot(self) -> WorkerManagerSnapshot:
        snapshots = self.registry.snapshots()

        running_count = sum(1 for snapshot in snapshots if snapshot.running)
        failed_count = sum(
            1 for snapshot in snapshots if snapshot.status == WorkerStatus.FAILED
        )

        return WorkerManagerSnapshot(
            worker_count=len(snapshots),
            running_count=running_count,
            failed_count=failed_count,
            workers=snapshots,
        )