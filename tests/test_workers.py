from __future__ import annotations

from threading import Event
from time import sleep

import pytest

from jarvis.runtime.events import EventBus
from jarvis.runtime.shared.enums import WorkerStatus
from jarvis.runtime.workers import BaseWorker, WorkerManager, WorkerRegistry


class CountingWorker(BaseWorker):
    def __init__(self, *, event_bus: EventBus, name: str = "counting_worker") -> None:
        super().__init__(
            name=name,
            event_bus=event_bus,
            tick_interval_seconds=0.005,
        )
        self.count = 0
        self.started_hook_called = False
        self.stopped_hook_called = False

    def on_start(self) -> None:
        self.started_hook_called = True

    def on_stop(self) -> None:
        self.stopped_hook_called = True

    def run_once(self) -> None:
        self.count += 1

        if self.count >= 3:
            self.request_stop()


class FailingWorker(BaseWorker):
    def __init__(self, *, event_bus: EventBus, name: str = "failing_worker") -> None:
        super().__init__(
            name=name,
            event_bus=event_bus,
            tick_interval_seconds=0.005,
        )

    def run_once(self) -> None:
        raise RuntimeError("boom")


def wait_until(condition: object, *, timeout_seconds: float = 2.0) -> None:
    deadline = timeout_seconds

    while deadline > 0:
        if bool(condition):
            return

        sleep(0.01)
        deadline -= 0.01

    raise AssertionError("condition was not met before timeout")


def test_worker_start_runs_and_stops_cleanly() -> None:
    bus = EventBus(name="test_bus")
    worker = CountingWorker(event_bus=bus)

    worker.start()
    worker.join(timeout_seconds=2.0)
    worker.stop()

    snapshot = worker.snapshot()

    assert worker.count >= 3
    assert snapshot.status == WorkerStatus.STOPPED
    assert snapshot.failure_count == 0
    assert snapshot.last_error is None
    assert worker.started_hook_called is True
    assert worker.stopped_hook_called is True


def test_worker_failure_is_captured() -> None:
    bus = EventBus(name="test_bus")
    worker = FailingWorker(event_bus=bus)

    worker.start()
    worker.join(timeout_seconds=2.0)

    snapshot = worker.snapshot()

    assert snapshot.status == WorkerStatus.FAILED
    assert snapshot.failure_count == 1
    assert snapshot.last_error is not None
    assert "RuntimeError" in snapshot.last_error


def test_worker_rejects_invalid_name() -> None:
    bus = EventBus(name="test_bus")

    with pytest.raises(ValueError):
        CountingWorker(event_bus=bus, name="   ")


def test_worker_rejects_invalid_tick_interval() -> None:
    bus = EventBus(name="test_bus")

    class BadWorker(BaseWorker):
        def run_once(self) -> None:
            return None

    with pytest.raises(ValueError):
        BadWorker(
            name="bad_worker",
            event_bus=bus,
            tick_interval_seconds=0.0,
        )


def test_worker_stop_is_idempotent_before_start() -> None:
    bus = EventBus(name="test_bus")
    worker = CountingWorker(event_bus=bus)

    worker.stop()
    worker.stop()

    assert worker.snapshot().status == WorkerStatus.STOPPED


def test_worker_registry_registers_and_retrieves_worker() -> None:
    bus = EventBus(name="test_bus")
    registry = WorkerRegistry()
    worker = CountingWorker(event_bus=bus)

    registry.register(worker)

    assert len(registry) == 1
    assert "counting_worker" in registry
    assert registry.get("counting_worker") is worker
    assert registry.require("counting_worker") is worker
    assert registry.names() == ("counting_worker",)


def test_worker_registry_rejects_duplicate_worker() -> None:
    bus = EventBus(name="test_bus")
    registry = WorkerRegistry()

    registry.register(CountingWorker(event_bus=bus))

    with pytest.raises(ValueError):
        registry.register(CountingWorker(event_bus=bus))


def test_worker_registry_unregisters_worker() -> None:
    bus = EventBus(name="test_bus")
    registry = WorkerRegistry()
    worker = CountingWorker(event_bus=bus)

    registry.register(worker)
    removed = registry.unregister("counting_worker")

    assert removed is worker
    assert len(registry) == 0


def test_worker_registry_rejects_missing_worker() -> None:
    registry = WorkerRegistry()

    with pytest.raises(KeyError):
        registry.require("missing")

    with pytest.raises(KeyError):
        registry.unregister("missing")


def test_worker_registry_rejects_empty_worker_name() -> None:
    registry = WorkerRegistry()

    with pytest.raises(ValueError):
        registry.get("   ")

    with pytest.raises(ValueError):
        registry.unregister("   ")


def test_worker_manager_starts_and_stops_all_workers() -> None:
    bus = EventBus(name="test_bus")
    manager = WorkerManager(event_bus=bus)

    first = CountingWorker(event_bus=bus, name="first_worker")
    second = CountingWorker(event_bus=bus, name="second_worker")

    manager.register(first)
    manager.register(second)

    manager.start_all()

    first.join(timeout_seconds=2.0)
    second.join(timeout_seconds=2.0)

    manager.stop_all()

    snapshot = manager.snapshot()

    assert snapshot.worker_count == 2
    assert snapshot.running_count == 0
    assert snapshot.failed_count == 0
    assert first.count >= 3
    assert second.count >= 3


def test_worker_manager_unregister_stops_worker() -> None:
    bus = EventBus(name="test_bus")
    manager = WorkerManager(event_bus=bus)
    worker = CountingWorker(event_bus=bus)

    manager.register(worker)
    manager.start_worker("counting_worker")
    worker.join(timeout_seconds=2.0)

    removed = manager.unregister("counting_worker")

    assert removed is worker
    assert manager.snapshot().worker_count == 0


def test_worker_manager_reports_failed_worker() -> None:
    bus = EventBus(name="test_bus")
    manager = WorkerManager(event_bus=bus)
    worker = FailingWorker(event_bus=bus)

    manager.register(worker)
    manager.start_worker("failing_worker")
    worker.join(timeout_seconds=2.0)

    snapshot = manager.snapshot()

    assert snapshot.worker_count == 1
    assert snapshot.failed_count == 1


def test_worker_can_emit_lifecycle_events_to_event_bus() -> None:
    bus = EventBus(name="test_bus")
    received = Event()

    def callback(_event: object) -> None:
        received.set()

    bus.subscribe(
        event_type=__import__(
            "jarvis.runtime.shared.enums",
            fromlist=["EventType"],
        ).EventType.WORKER_STARTED,
        subscriber_name="test_listener",
        callback=callback,
    )

    bus.start()

    try:
        worker = CountingWorker(event_bus=bus)
        worker.start()
        worker.join(timeout_seconds=2.0)

        assert received.wait(timeout=2.0) is True
    finally:
        worker.stop()
        bus.stop()