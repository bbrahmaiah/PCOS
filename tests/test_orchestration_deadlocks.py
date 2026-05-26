from __future__ import annotations

from datetime import timedelta

import pytest
from pydantic import ValidationError

from jarvis.orchestration import (
    DeadlockDetectionDecision,
    DeadlockDetectionReason,
    DeadlockDetector,
    DeadlockDetectorConfig,
    DeadlockResolutionAction,
    ResourceLockKind,
    ResourceLockRequest,
    TaskPriority,
    WaitEdge,
    WaitGraph,
    new_task_id,
    new_worker_id,
)
from jarvis.orchestration.ids import utc_now


def wait_edge(
    *,
    waiting_worker_id: str | None = None,
    blocking_worker_id: str | None = None,
    waiting_task_id: str | None = None,
    blocking_task_id: str | None = None,
    waiting_priority: TaskPriority = TaskPriority.NORMAL,
    blocking_priority: TaskPriority = TaskPriority.NORMAL,
    resource: ResourceLockKind = ResourceLockKind.MEMORY,
    timeout_ms: int = 2_000,
) -> WaitEdge:
    return WaitEdge(
        waiting_worker_id=waiting_worker_id or new_worker_id(),
        blocking_worker_id=blocking_worker_id or new_worker_id(),
        waiting_task_id=waiting_task_id or new_task_id(),
        blocking_task_id=blocking_task_id or new_task_id(),
        resource=resource,
        waiting_task_priority=waiting_priority,
        blocking_task_priority=blocking_priority,
        timeout_ms=timeout_ms,
    )


def test_config_rejects_invalid_timeout() -> None:
    with pytest.raises(ValueError):
        DeadlockDetectorConfig(max_wait_ms=0).validate()


def test_lock_request_requires_locks() -> None:
    with pytest.raises(ValidationError):
        ResourceLockRequest(
            task_id=new_task_id(),
            worker_id=new_worker_id(),
            locks=(),
        )


def test_lock_order_valid() -> None:
    detector = DeadlockDetector()
    request = ResourceLockRequest(
        task_id=new_task_id(),
        worker_id=new_worker_id(),
        locks=(
            ResourceLockKind.ATTENTION,
            ResourceLockKind.MEMORY,
            ResourceLockKind.FILE_SYSTEM,
        ),
    )

    result = detector.validate_lock_order(request)

    assert result.success is True
    assert result.decision == DeadlockDetectionDecision.CLEAR
    assert result.reason == DeadlockDetectionReason.LOCK_ORDER_VALID


def test_lock_order_violation_rejected() -> None:
    detector = DeadlockDetector()
    request = ResourceLockRequest(
        task_id=new_task_id(),
        worker_id=new_worker_id(),
        locks=(
            ResourceLockKind.TOOL,
            ResourceLockKind.MEMORY,
        ),
    )

    result = detector.validate_lock_order(request)

    assert result.success is False
    assert result.decision == DeadlockDetectionDecision.REJECTED
    assert result.reason == DeadlockDetectionReason.LOCK_ORDER_VIOLATION


def test_wait_edge_rejects_self_wait() -> None:
    worker_id = new_worker_id()

    with pytest.raises(ValidationError):
        wait_edge(
            waiting_worker_id=worker_id,
            blocking_worker_id=worker_id,
        )


def test_wait_graph_add_and_remove_edge() -> None:
    edge = wait_edge()
    graph = WaitGraph()

    updated = graph.add_edge(edge)
    removed = updated.remove_edge(edge.edge_id)

    assert len(updated.edges) == 1
    assert removed.edges == ()


def test_wait_graph_remove_missing_edge_raises() -> None:
    graph = WaitGraph()

    with pytest.raises(ValueError):
        graph.remove_edge("waitedge_missing")


def test_no_deadlock_detected_for_linear_waits() -> None:
    worker_a = new_worker_id()
    worker_b = new_worker_id()
    worker_c = new_worker_id()
    detector = DeadlockDetector()

    detector.add_wait(
        wait_edge(waiting_worker_id=worker_a, blocking_worker_id=worker_b)
    )
    result = detector.add_wait(
        wait_edge(waiting_worker_id=worker_b, blocking_worker_id=worker_c)
    )

    assert result.success is True
    assert result.reason == DeadlockDetectionReason.WAIT_EDGE_ADDED
    assert detector.detect().reason == DeadlockDetectionReason.NO_DEADLOCK


def test_cycle_detection_two_workers() -> None:
    worker_a = new_worker_id()
    worker_b = new_worker_id()
    detector = DeadlockDetector()

    detector.add_wait(
        wait_edge(waiting_worker_id=worker_a, blocking_worker_id=worker_b)
    )
    result = detector.add_wait(
        wait_edge(waiting_worker_id=worker_b, blocking_worker_id=worker_a)
    )

    assert result.success is False
    assert result.decision == DeadlockDetectionDecision.DETECTED
    assert result.reason == DeadlockDetectionReason.CYCLE_DETECTED
    assert result.event is not None
    assert len(result.event.cycle) == 2


def test_cycle_detection_three_workers() -> None:
    worker_a = new_worker_id()
    worker_b = new_worker_id()
    worker_c = new_worker_id()
    detector = DeadlockDetector()

    detector.add_wait(
        wait_edge(waiting_worker_id=worker_a, blocking_worker_id=worker_b)
    )
    detector.add_wait(
        wait_edge(waiting_worker_id=worker_b, blocking_worker_id=worker_c)
    )
    result = detector.add_wait(
        wait_edge(waiting_worker_id=worker_c, blocking_worker_id=worker_a)
    )

    assert result.event is not None
    assert len(result.event.worker_ids) == 3


def test_deadlock_event_contains_wait_graph_snapshot() -> None:
    worker_a = new_worker_id()
    worker_b = new_worker_id()
    detector = DeadlockDetector()

    detector.add_wait(
        wait_edge(waiting_worker_id=worker_a, blocking_worker_id=worker_b)
    )
    result = detector.add_wait(
        wait_edge(waiting_worker_id=worker_b, blocking_worker_id=worker_a)
    )

    assert result.event is not None
    assert len(result.event.wait_graph_snapshot.edges) == 2


def test_resolve_deadlock_cancels_lowest_priority_task() -> None:
    worker_a = new_worker_id()
    worker_b = new_worker_id()
    low_task = new_task_id()
    high_task = new_task_id()
    detector = DeadlockDetector()

    detector.add_wait(
        wait_edge(
            waiting_worker_id=worker_a,
            blocking_worker_id=worker_b,
            waiting_task_id=low_task,
            blocking_task_id=high_task,
            waiting_priority=TaskPriority.LOW,
            blocking_priority=TaskPriority.HIGH,
        )
    )
    detected = detector.add_wait(
        wait_edge(
            waiting_worker_id=worker_b,
            blocking_worker_id=worker_a,
            waiting_task_id=high_task,
            blocking_task_id=low_task,
            waiting_priority=TaskPriority.HIGH,
            blocking_priority=TaskPriority.LOW,
        )
    )

    assert detected.event is not None

    resolved = detector.resolve(detected.event.deadlock_id)

    assert resolved.success is True
    assert resolved.resolution is not None
    assert resolved.resolution.action == DeadlockResolutionAction.CANCEL_AND_RETRY_TASK
    assert resolved.resolution.task_id == low_task


def test_resolve_unknown_deadlock_rejected() -> None:
    detector = DeadlockDetector()

    result = detector.resolve("deadlock_missing")

    assert result.success is False
    assert result.reason == DeadlockDetectionReason.DEADLOCK_NOT_FOUND


def test_detect_timeouts() -> None:
    detector = DeadlockDetector()
    edge = wait_edge(timeout_ms=1)
    old_edge = edge.model_copy(
        update={"created_at": utc_now() - timedelta(milliseconds=10)}
    )

    detector.add_wait(old_edge)
    results = detector.detect_timeouts()

    assert len(results) == 1
    assert results[0].reason == DeadlockDetectionReason.WAIT_TIMEOUT_DETECTED


def test_snapshot_counts_detector_state() -> None:
    worker_a = new_worker_id()
    worker_b = new_worker_id()
    detector = DeadlockDetector()

    detector.add_wait(
        wait_edge(waiting_worker_id=worker_a, blocking_worker_id=worker_b)
    )
    detected = detector.add_wait(
        wait_edge(waiting_worker_id=worker_b, blocking_worker_id=worker_a)
    )

    assert detected.event is not None

    detector.resolve(detected.event.deadlock_id)
    snapshot = detector.snapshot()

    assert snapshot.wait_edge_count == 2
    assert snapshot.detected_count == 1
    assert snapshot.resolved_count == 1


def test_reset_clears_detector_state() -> None:
    detector = DeadlockDetector()
    detector.add_wait(wait_edge())

    detector.reset()
    snapshot = detector.snapshot()

    assert snapshot.wait_edge_count == 0
    assert snapshot.detected_count == 0


def test_enum_values_are_stable() -> None:
    assert ResourceLockKind.MEMORY.value == "memory"
    assert DeadlockDetectionDecision.DETECTED.value == "detected"
    assert DeadlockDetectionReason.CYCLE_DETECTED.value == "cycle_detected"
    assert DeadlockResolutionAction.CANCEL_AND_RETRY_TASK.value