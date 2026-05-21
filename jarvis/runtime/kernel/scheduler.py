from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from threading import RLock
from uuid import uuid4

from jarvis.runtime.events import EventBus, RuntimeEvent
from jarvis.runtime.observability.structured_logger import get_logger
from jarvis.runtime.shared.enums import EventCategory, EventType
from jarvis.runtime.workers import BaseWorker


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_task_id() -> str:
    return uuid4().hex


@dataclass(slots=True)
class ScheduledTask:
    name: str
    callback: Callable[[], None]
    interval_seconds: float
    run_once: bool = False
    task_id: str = field(default_factory=new_task_id)
    enabled: bool = True
    next_run_at: datetime = field(default_factory=utc_now)
    last_run_at: datetime | None = None
    run_count: int = 0
    failure_count: int = 0
    last_error: str | None = None

    def __post_init__(self) -> None:
        self.name = self.name.strip()

        if not self.name:
            raise ValueError("scheduled task name cannot be empty.")

        if self.interval_seconds < 0:
            raise ValueError("interval_seconds cannot be negative.")

        if not callable(self.callback):
            raise TypeError("scheduled task callback must be callable.")


@dataclass(frozen=True, slots=True)
class SchedulerSnapshot:
    task_count: int
    enabled_count: int
    tasks: tuple[ScheduledTask, ...]


class Scheduler(BaseWorker):
    """
    Kernel scheduler.

    Runs lightweight periodic jobs:
    - future heartbeat ticks
    - health checks
    - cleanup tasks
    - diagnostic events
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        name: str = "scheduler",
        tick_interval_seconds: float = 0.05,
    ) -> None:
        super().__init__(
            name=name,
            event_bus=event_bus,
            tick_interval_seconds=tick_interval_seconds,
        )

        self._lock = RLock()
        self._tasks: dict[str, ScheduledTask] = {}
        self._logger = get_logger("kernel.scheduler")

    def register_task(
        self,
        *,
        name: str,
        callback: Callable[[], None],
        interval_seconds: float,
        run_once: bool = False,
        start_immediately: bool = True,
    ) -> ScheduledTask:
        next_run_at = utc_now()

        if not start_immediately:
            next_run_at = next_run_at + timedelta(seconds=interval_seconds)

        task = ScheduledTask(
            name=name,
            callback=callback,
            interval_seconds=interval_seconds,
            run_once=run_once,
            next_run_at=next_run_at,
        )

        with self._lock:
            if task.name in (existing.name for existing in self._tasks.values()):
                raise ValueError(f"Scheduled task already exists: {task.name}")

            self._tasks[task.task_id] = task

        self._logger.info(
            "scheduled_task_registered",
            task_id=task.task_id,
            task_name=task.name,
            interval_seconds=task.interval_seconds,
            run_once=task.run_once,
        )

        return task

    def unregister_task(self, task_id: str) -> bool:
        clean_id = self._validate_task_id(task_id)

        with self._lock:
            existed = clean_id in self._tasks
            self._tasks.pop(clean_id, None)

        return existed

    def enable_task(self, task_id: str) -> None:
        task = self.require_task(task_id)
        task.enabled = True

    def disable_task(self, task_id: str) -> None:
        task = self.require_task(task_id)
        task.enabled = False

    def require_task(self, task_id: str) -> ScheduledTask:
        clean_id = self._validate_task_id(task_id)

        with self._lock:
            try:
                return self._tasks[clean_id]
            except KeyError as exc:
                raise KeyError(f"Scheduled task not found: {clean_id}") from exc

    def scheduler_snapshot(self) -> SchedulerSnapshot:
        with self._lock:
            tasks = tuple(self._tasks.values())

        return SchedulerSnapshot(
            task_count=len(tasks),
            enabled_count=sum(1 for task in tasks if task.enabled),
            tasks=tasks,
        )

    def run_due_tasks(self) -> int:
        now = utc_now()

        with self._lock:
            due_tasks = tuple(
                task
                for task in self._tasks.values()
                if task.enabled and task.next_run_at <= now
            )

        executed = 0

        for task in due_tasks:
            self._run_task(task)
            executed += 1

        return executed

    def run_once(self) -> None:
        executed = self.run_due_tasks()

        if executed:
            self.event_bus.publish(
                RuntimeEvent(
                    event_type=EventType.RUNTIME_TICK,
                    category=EventCategory.RUNTIME,
                    source=self.name,
                    payload={"scheduled_tasks_executed": executed},
                )
            )

    def _run_task(self, task: ScheduledTask) -> None:
        try:
            task.callback()
            task.last_error = None

        except Exception as exc:
            task.failure_count += 1
            task.last_error = f"{type(exc).__name__}: {exc}"

            self._logger.exception(
                "scheduled_task_failed",
                task_id=task.task_id,
                task_name=task.name,
                error=task.last_error,
            )

            self.event_bus.publish(
                RuntimeEvent(
                    event_type=EventType.DIAGNOSTIC_REPORTED,
                    category=EventCategory.OPERATIONS,
                    source=self.name,
                    payload={
                        "task_id": task.task_id,
                        "task_name": task.name,
                        "error": task.last_error,
                    },
                )
            )

        finally:
            task.run_count += 1
            task.last_run_at = utc_now()

            if task.run_once:
                self.unregister_task(task.task_id)
            else:
                task.next_run_at = utc_now() + timedelta(seconds=task.interval_seconds)

    @staticmethod
    def _validate_task_id(task_id: str) -> str:
        if not isinstance(task_id, str):
            raise TypeError("task_id must be a string.")

        clean_id = task_id.strip()

        if not clean_id:
            raise ValueError("task_id cannot be empty.")

        return clean_id