from __future__ import annotations

from jarvis.runtime.events import EventBus, RuntimeEvent
from jarvis.runtime.observability.performance_monitor import get_performance_monitor
from jarvis.runtime.observability.structured_logger import get_logger
from jarvis.runtime.shared.enums import EventCategory, EventType, RuntimeStatus
from jarvis.runtime.state import StateEngine
from jarvis.runtime.workers import WorkerManager


class LifecycleManager:
    """
    Coordinates runtime lifecycle.

    Start order:
    1. EventBus
    2. Runtime state STARTING
    3. Workers
    4. Runtime state RUNNING

    Stop order:
    1. Runtime state STOPPING
    2. Workers stop
    3. Runtime state STOPPED
    4. EventBus drains and stops
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        state_engine: StateEngine,
        worker_manager: WorkerManager,
    ) -> None:
        self.event_bus = event_bus
        self.state_engine = state_engine
        self.worker_manager = worker_manager

        self._logger = get_logger("kernel.lifecycle")
        self._performance = get_performance_monitor()

    def start(self) -> None:
        with self._performance.measure("kernel.lifecycle.start"):
            try:
                self.event_bus.start()

                self._emit_runtime_event(EventType.RUNTIME_STARTING)
                self.state_engine.set_runtime_status(RuntimeStatus.STARTING)

                self.worker_manager.start_all()

                self.state_engine.set_runtime_status(RuntimeStatus.RUNNING)
                self._emit_runtime_event(EventType.RUNTIME_STARTED)

                self._logger.info("runtime_lifecycle_started")

            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

                self.state_engine.set_runtime_status(
                    RuntimeStatus.FAILED,
                    last_error=error,
                )

                self._emit_runtime_event(
                    EventType.RUNTIME_FAILED,
                    payload={"error": error},
                )

                self._logger.exception(
                    "runtime_lifecycle_start_failed",
                    error=error,
                )

                raise

    def stop(self) -> None:
        with self._performance.measure("kernel.lifecycle.stop"):
            try:
                self._emit_runtime_event(EventType.RUNTIME_STOPPING)
                self.state_engine.set_runtime_status(RuntimeStatus.STOPPING)

                self.worker_manager.stop_all()

                self.state_engine.set_runtime_status(RuntimeStatus.STOPPED)
                self._emit_runtime_event(EventType.RUNTIME_STOPPED)

                self.event_bus.drain(timeout_seconds=5.0)
                self.event_bus.stop(timeout_seconds=5.0)

                self._logger.info("runtime_lifecycle_stopped")

            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

                self.state_engine.set_runtime_status(
                    RuntimeStatus.FAILED,
                    last_error=error,
                )

                self._emit_runtime_event(
                    EventType.RUNTIME_FAILED,
                    payload={"error": error},
                )

                self._logger.exception(
                    "runtime_lifecycle_stop_failed",
                    error=error,
                )

                raise

    def _emit_runtime_event(
        self,
        event_type: EventType,
        *,
        payload: dict[str, object] | None = None,
    ) -> None:
        self.event_bus.publish(
            RuntimeEvent(
                event_type=event_type,
                category=EventCategory.RUNTIME,
                source="lifecycle_manager",
                payload=payload or {},
            )
        )