from __future__ import annotations

from threading import RLock

from jarvis.runtime.workers.worker import BaseWorker, WorkerSnapshot


class WorkerRegistry:
    """
    Thread-safe registry of runtime workers.

    The registry owns lookup and uniqueness.
    It does not start or stop workers.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._workers: dict[str, BaseWorker] = {}

    def register(self, worker: BaseWorker) -> None:
        if not isinstance(worker, BaseWorker):
            raise TypeError("WorkerRegistry.register expects BaseWorker.")

        with self._lock:
            if worker.name in self._workers:
                raise ValueError(f"Worker already registered: {worker.name}")

            self._workers[worker.name] = worker

    def unregister(self, worker_name: str) -> BaseWorker:
        clean_name = worker_name.strip()

        if not clean_name:
            raise ValueError("worker_name cannot be empty.")

        with self._lock:
            try:
                return self._workers.pop(clean_name)
            except KeyError as exc:
                raise KeyError(f"Worker not registered: {clean_name}") from exc

    def get(self, worker_name: str) -> BaseWorker | None:
        clean_name = worker_name.strip()

        if not clean_name:
            raise ValueError("worker_name cannot be empty.")

        with self._lock:
            return self._workers.get(clean_name)

    def require(self, worker_name: str) -> BaseWorker:
        worker = self.get(worker_name)

        if worker is None:
            raise KeyError(f"Worker not registered: {worker_name}")

        return worker

    def names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._workers))

    def workers(self) -> tuple[BaseWorker, ...]:
        with self._lock:
            return tuple(self._workers.values())

    def snapshots(self) -> tuple[WorkerSnapshot, ...]:
        with self._lock:
            return tuple(worker.snapshot() for worker in self._workers.values())

    def clear(self) -> None:
        with self._lock:
            self._workers.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._workers)

    def __contains__(self, worker_name: object) -> bool:
        if not isinstance(worker_name, str):
            return False

        with self._lock:
            return worker_name in self._workers