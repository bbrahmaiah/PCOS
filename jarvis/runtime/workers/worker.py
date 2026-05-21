from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Event, RLock, Thread, current_thread
from time import sleep
from typing import Final

from jarvis.runtime.events import EventBus, RuntimeEvent
from jarvis.runtime.observability.performance_monitor import get_performance_monitor
from jarvis.runtime.observability.structured_logger import get_logger
from jarvis.runtime.shared.enums import EventCategory, EventType, WorkerStatus

_MIN_TICK_INTERVAL_SECONDS: Final[float] = 0.001


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class WorkerSnapshot:
    """
    Immutable diagnostic snapshot for one runtime worker.
    """

    name: str
    status: WorkerStatus
    running: bool
    started_at: datetime | None
    stopped_at: datetime | None
    last_heartbeat_at: datetime | None
    failure_count: int
    last_error: str | None


class BaseWorker(ABC):
    """
    Base class for all long-running JARVIS runtime workers.

    Workers are independent parallel subsystems:
    Presence, Awareness, Memory, Router, Action, Dialogue, and more.
    """

    def __init__(
        self,
        *,
        name: str,
        event_bus: EventBus,
        tick_interval_seconds: float = 0.05,
        daemon: bool = True,
    ) -> None:
        clean_name = name.strip()

        if not clean_name:
            raise ValueError("Worker name cannot be empty.")

        if tick_interval_seconds < _MIN_TICK_INTERVAL_SECONDS:
            raise ValueError(
                f"tick_interval_seconds must be >= {_MIN_TICK_INTERVAL_SECONDS}."
            )

        self.name = clean_name
        self.event_bus = event_bus
        self.tick_interval_seconds = tick_interval_seconds
        self.daemon = daemon

        self._lock = RLock()
        self._stop_requested = Event()
        self._thread: Thread | None = None

        self._status = WorkerStatus.CREATED
        self._started_at: datetime | None = None
        self._stopped_at: datetime | None = None
        self._last_heartbeat_at: datetime | None = None
        self._failure_count = 0
        self._last_error: str | None = None

        self._logger = get_logger(f"workers.{self.name}")
        self._performance = get_performance_monitor()

    @property
    def status(self) -> WorkerStatus:
        with self._lock:
            return self._status

    @property
    def running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """
        Start the worker thread.

        Safe to call multiple times.
        """

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return

            self._stop_requested.clear()
            self._status = WorkerStatus.STARTING
            self._started_at = _utc_now()
            self._stopped_at = None
            self._last_error = None

            self._thread = Thread(
                target=self._run_loop,
                name=f"{self.name}_worker",
                daemon=self.daemon,
            )

            self._emit_worker_event(EventType.WORKER_STARTING)
            self._thread.start()

        self._logger.info("worker_start_requested", worker=self.name)

    def stop(self, *, timeout_seconds: float = 5.0) -> None:
        """
        Request worker shutdown and wait for thread termination.

        Safe to call multiple times.
        """

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")

        with self._lock:
            thread = self._thread

            if thread is None:
                self._status = WorkerStatus.STOPPED
                self._stopped_at = _utc_now()
                return

            if self._status not in {WorkerStatus.STOPPED, WorkerStatus.FAILED}:
                self._status = WorkerStatus.STOPPING

            self._stop_requested.set()
            self._emit_worker_event(EventType.WORKER_STOPPING)

        if thread is not current_thread():
            thread.join(timeout=timeout_seconds)

        with self._lock:
            if thread.is_alive():
                self._logger.warning(
                    "worker_stop_timeout",
                    worker=self.name,
                    timeout_seconds=timeout_seconds,
                )
                return

            self._thread = None

            if self._status != WorkerStatus.FAILED:
                self._status = WorkerStatus.STOPPED

            self._stopped_at = _utc_now()

        self._emit_worker_event(EventType.WORKER_STOPPED)
        self._logger.info("worker_stopped", worker=self.name)

    def request_stop(self) -> None:
        """
        Signal the worker to stop without joining.

        Useful from inside run_once().
        """

        self._stop_requested.set()

    def should_stop(self) -> bool:
        return self._stop_requested.is_set()

    def join(self, *, timeout_seconds: float | None = None) -> None:
        with self._lock:
            thread = self._thread

        if thread is not None and thread is not current_thread():
            thread.join(timeout=timeout_seconds)

    def snapshot(self) -> WorkerSnapshot:
        with self._lock:
            return WorkerSnapshot(
                name=self.name,
                status=self._status,
                running=self.running,
                started_at=self._started_at,
                stopped_at=self._stopped_at,
                last_heartbeat_at=self._last_heartbeat_at,
                failure_count=self._failure_count,
                last_error=self._last_error,
            )

    def heartbeat(self) -> None:
        with self._lock:
            self._last_heartbeat_at = _utc_now()

        self._emit_worker_event(EventType.WORKER_HEALTH_UPDATED)

    def on_start(self) -> None:
        """
        Optional startup hook for subclasses.
        """

        return None

    def on_stop(self) -> None:
        """
        Optional shutdown hook for subclasses.
        """

        return None

    @abstractmethod
    def run_once(self) -> None:
        """
        Execute one unit of worker work.

        Subclasses must implement this method.
        """

        raise NotImplementedError

    def _run_loop(self) -> None:
        try:
            with self._lock:
                self._status = WorkerStatus.RUNNING

            self._emit_worker_event(EventType.WORKER_STARTED)
            self._logger.info("worker_started", worker=self.name)

            self.on_start()

            while not self._stop_requested.is_set():
                with self._lock:
                    self._status = WorkerStatus.BUSY

                with self._performance.measure(f"worker.{self.name}.run_once"):
                    self.run_once()

                self.heartbeat()

                with self._lock:
                    if not self._stop_requested.is_set():
                        self._status = WorkerStatus.IDLE

                sleep(self.tick_interval_seconds)

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

            with self._lock:
                self._failure_count += 1
                self._last_error = error
                self._status = WorkerStatus.FAILED
                self._stopped_at = _utc_now()

            self._emit_worker_event(
                EventType.WORKER_FAILED,
                payload={"error": error},
            )

            self._logger.exception(
                "worker_failed",
                worker=self.name,
                error=error,
            )

        finally:
            try:
                self.on_stop()
            finally:
                with self._lock:
                    if self._status != WorkerStatus.FAILED:
                        self._status = WorkerStatus.STOPPED
                        self._stopped_at = _utc_now()

    def _emit_worker_event(
        self,
        event_type: EventType,
        payload: dict[str, object] | None = None,
    ) -> None:
        event = RuntimeEvent(
            event_type=event_type,
            category=EventCategory.WORKER,
            source=self.name,
            payload={
                "worker_name": self.name,
                "status": self.status.value,
                **(payload or {}),
            },
        )

        self.event_bus.publish(event)