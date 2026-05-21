from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from jarvis.runtime.events import EventBusSnapshot
from jarvis.runtime.shared.enums import RuntimeStatus
from jarvis.runtime.state import StateSnapshot
from jarvis.runtime.workers import WorkerManagerSnapshot


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class RuntimeHealthSnapshot:
    healthy: bool
    checked_at: datetime
    runtime_status: RuntimeStatus
    event_bus_running: bool
    worker_count: int
    running_workers: int
    failed_workers: int
    reasons: tuple[str, ...]


class HealthMonitor:
    """
    Kernel health monitor.

    It combines EventBus, WorkerManager, and StateEngine snapshots into
    a clear runtime health result.
    """

    def check(
        self,
        *,
        event_bus_snapshot: EventBusSnapshot,
        worker_manager_snapshot: WorkerManagerSnapshot,
        state_snapshot: StateSnapshot,
    ) -> RuntimeHealthSnapshot:
        reasons: list[str] = []

        if state_snapshot.runtime.status == RuntimeStatus.FAILED:
            reasons.append("runtime_status_failed")

        if worker_manager_snapshot.failed_count > 0:
            reasons.append("worker_failures_detected")

        if (
            state_snapshot.runtime.status == RuntimeStatus.RUNNING
            and not event_bus_snapshot.running
        ):
            reasons.append("runtime_running_but_event_bus_stopped")

        healthy = not reasons

        return RuntimeHealthSnapshot(
            healthy=healthy,
            checked_at=utc_now(),
            runtime_status=state_snapshot.runtime.status,
            event_bus_running=event_bus_snapshot.running,
            worker_count=worker_manager_snapshot.worker_count,
            running_workers=worker_manager_snapshot.running_count,
            failed_workers=worker_manager_snapshot.failed_count,
            reasons=tuple(reasons),
        )