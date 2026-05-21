from __future__ import annotations

from threading import Event

import pytest

from jarvis.runtime.events import EventBus
from jarvis.runtime.kernel import (
    CancellationManager,
    HealthMonitor,
    LifecycleManager,
    RuntimeKernel,
    Scheduler,
)
from jarvis.runtime.shared.enums import RuntimeStatus
from jarvis.runtime.state import StateEngine
from jarvis.runtime.workers import WorkerManager


def test_cancellation_manager_creates_and_cancels_token() -> None:
    manager = CancellationManager()

    token = manager.create_token(
        token_id="task-1",
        metadata={"kind": "test"},
    )

    assert token.cancelled is False

    manager.cancel("task-1", reason="stop")

    assert token.cancelled is True
    assert token.reason == "stop"

    snapshot = manager.snapshot()

    assert snapshot.token_count == 1
    assert snapshot.cancelled_count == 1


def test_cancellation_manager_rejects_duplicate_token() -> None:
    manager = CancellationManager()

    manager.create_token(token_id="same")

    with pytest.raises(ValueError):
        manager.create_token(token_id="same")


def test_cancellation_token_raise_if_cancelled() -> None:
    manager = CancellationManager()
    token = manager.create_token(token_id="task-1")

    token.cancel("interrupted")

    with pytest.raises(RuntimeError):
        token.raise_if_cancelled()


def test_scheduler_runs_due_task_once() -> None:
    bus = EventBus(name="test_bus")
    scheduler = Scheduler(event_bus=bus)
    called = Event()

    scheduler.register_task(
        name="test_task",
        callback=called.set,
        interval_seconds=1.0,
        run_once=True,
    )

    executed = scheduler.run_due_tasks()

    assert executed == 1
    assert called.is_set() is True
    assert scheduler.scheduler_snapshot().task_count == 0


def test_scheduler_records_failed_task() -> None:
    bus = EventBus(name="test_bus")
    scheduler = Scheduler(event_bus=bus)

    def fail() -> None:
        raise RuntimeError("boom")

    task = scheduler.register_task(
        name="failing_task",
        callback=fail,
        interval_seconds=1.0,
    )

    executed = scheduler.run_due_tasks()

    assert executed == 1

    stored = scheduler.require_task(task.task_id)

    assert stored.failure_count == 1
    assert stored.last_error is not None
    assert "RuntimeError" in stored.last_error


def test_scheduler_rejects_invalid_task() -> None:
    bus = EventBus(name="test_bus")
    scheduler = Scheduler(event_bus=bus)

    with pytest.raises(ValueError):
        scheduler.register_task(
            name="   ",
            callback=lambda: None,
            interval_seconds=1.0,
        )

    with pytest.raises(ValueError):
        scheduler.register_task(
            name="bad",
            callback=lambda: None,
            interval_seconds=-1.0,
        )


def test_health_monitor_reports_healthy_runtime() -> None:
    bus = EventBus(name="test_bus")
    state = StateEngine(event_bus=bus)
    workers = WorkerManager(event_bus=bus)

    state.set_runtime_status(RuntimeStatus.RUNNING)
    bus.start()

    try:
        health = HealthMonitor().check(
            event_bus_snapshot=bus.snapshot(),
            worker_manager_snapshot=workers.snapshot(),
            state_snapshot=state.snapshot(),
        )
    finally:
        bus.stop()

    assert health.healthy is True
    assert health.runtime_status == RuntimeStatus.RUNNING


def test_health_monitor_reports_event_bus_stopped_when_runtime_running() -> None:
    bus = EventBus(name="test_bus")
    state = StateEngine(event_bus=bus)
    workers = WorkerManager(event_bus=bus)

    state.set_runtime_status(RuntimeStatus.RUNNING)

    health = HealthMonitor().check(
        event_bus_snapshot=bus.snapshot(),
        worker_manager_snapshot=workers.snapshot(),
        state_snapshot=state.snapshot(),
    )

    assert health.healthy is False
    assert "runtime_running_but_event_bus_stopped" in health.reasons


def test_lifecycle_manager_start_and_stop() -> None:
    bus = EventBus(name="test_bus")
    state = StateEngine(event_bus=bus)
    workers = WorkerManager(event_bus=bus)

    lifecycle = LifecycleManager(
        event_bus=bus,
        state_engine=state,
        worker_manager=workers,
    )

    lifecycle.start()

    assert state.runtime_state().status == RuntimeStatus.RUNNING
    assert bus.snapshot().running is True

    lifecycle.stop()

    assert state.runtime_state().status == RuntimeStatus.STOPPED
    assert bus.snapshot().running is False


def test_runtime_kernel_start_stop_snapshot() -> None:
    kernel = RuntimeKernel()

    kernel.start()

    try:
        snapshot = kernel.snapshot()

        assert snapshot.state.runtime.status == RuntimeStatus.RUNNING
        assert snapshot.event_bus.running is True
        assert snapshot.workers.worker_count >= 1
    finally:
        kernel.stop()

    stopped_snapshot = kernel.snapshot()

    assert stopped_snapshot.state.runtime.status == RuntimeStatus.STOPPED
    assert stopped_snapshot.event_bus.running is False