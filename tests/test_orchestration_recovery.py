from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from jarvis.orchestration import (
    RecoverableTaskRecord,
    RecoveryEvent,
    RecoveryEventType,
    RecoveryFailureKind,
    RecoveryManager,
    RecoveryReason,
    RecoveryRuntimeConfig,
    RecoveryStatus,
    RecoveryStrategy,
    RecoveryStrategySelector,
    RetryPolicy,
    StateReconstructor,
)


def record(
    *,
    task_id: str = "task-1",
    worker_id: str | None = "worker-1",
    failure_kind: RecoveryFailureKind = RecoveryFailureKind.TASK_FAILURE,
    failure_count: int = 0,
    max_attempts: int = 3,
    recoverable: bool = True,
    age_seconds: int = 0,
    stale_after_seconds: int = 120,
    interrupted: bool = False,
) -> RecoverableTaskRecord:
    return RecoverableTaskRecord(
        task_id=task_id,
        worker_id=worker_id,
        failure_kind=failure_kind,
        failure_count=failure_count,
        max_attempts=max_attempts,
        recoverable=recoverable,
        age_seconds=age_seconds,
        stale_after_seconds=stale_after_seconds,
        interrupted=interrupted,
    )


def test_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        RecoveryRuntimeConfig(name=" ").validate()


def test_config_rejects_invalid_backoff() -> None:
    config = RecoveryRuntimeConfig(
        backoff_base_seconds=10,
        backoff_max_seconds=5,
    )

    with pytest.raises(ValueError):
        config.validate()


def test_task_record_requires_task_id() -> None:
    with pytest.raises(ValidationError):
        RecoverableTaskRecord(task_id=" ")


def test_task_record_detects_stale_assignment() -> None:
    item = record(age_seconds=120, stale_after_seconds=120)

    assert item.is_stale is True


def test_task_record_attempts_remaining() -> None:
    item = record(failure_count=1, max_attempts=3)

    assert item.attempts_remaining == 2


def test_retry_policy_allows_recoverable_task() -> None:
    policy = RetryPolicy(
        config=RecoveryRuntimeConfig(
            backoff_base_seconds=2,
            backoff_max_seconds=60,
        )
    )

    decision = policy.decide(record(failure_count=1))

    assert decision.allowed is True
    assert decision.delay_seconds == 4
    assert decision.reason == RecoveryReason.RETRY_ALLOWED


def test_retry_policy_denies_unrecoverable_task() -> None:
    policy = RetryPolicy()

    decision = policy.decide(record(recoverable=False))

    assert decision.allowed is False
    assert decision.reason == RecoveryReason.TASK_UNRECOVERABLE


def test_retry_policy_denies_exhausted_attempts() -> None:
    policy = RetryPolicy()

    decision = policy.decide(record(failure_count=3, max_attempts=3))

    assert decision.allowed is False
    assert decision.reason == RecoveryReason.RETRY_DENIED


def test_selector_restarts_crashed_worker() -> None:
    selector = RecoveryStrategySelector()

    decision = selector.decide_for_task(
        record(failure_kind=RecoveryFailureKind.WORKER_CRASH)
    )

    assert decision.strategy == RecoveryStrategy.RESTART_WORKER
    assert decision.status == RecoveryStatus.RECOVERING
    assert decision.reason == RecoveryReason.WORKER_RESTART_REQUIRED


def test_selector_reconstructs_stale_assignment() -> None:
    selector = RecoveryStrategySelector()

    decision = selector.decide_for_task(
        record(age_seconds=130, stale_after_seconds=120)
    )

    assert decision.strategy == RecoveryStrategy.RECONSTRUCT_FROM_CHECKPOINT
    assert decision.reason == RecoveryReason.STALE_ASSIGNMENT_DETECTED


def test_selector_degrades_interrupted_recovery() -> None:
    selector = RecoveryStrategySelector()

    decision = selector.decide_for_task(record(interrupted=True))

    assert decision.strategy == RecoveryStrategy.DEGRADE_GRACEFULLY
    assert decision.status == RecoveryStatus.DEGRADED


def test_selector_quarantines_after_threshold() -> None:
    selector = RecoveryStrategySelector(
        config=RecoveryRuntimeConfig(quarantine_after_failures=2)
    )

    decision = selector.decide_for_task(record(failure_count=2))

    assert decision.status == RecoveryStatus.QUARANTINED
    assert decision.strategy == RecoveryStrategy.ABANDON_AND_LOG
    assert decision.user_visible is True
    assert decision.user_message == "I lost track of that task."


def test_selector_retries_recoverable_task() -> None:
    selector = RecoveryStrategySelector()

    decision = selector.decide_for_task(record(failure_count=1))

    assert decision.strategy == RecoveryStrategy.RETRY_WITH_BACKOFF
    assert decision.retry_delay_seconds > 0


def test_selector_abandons_unrecoverable_task() -> None:
    selector = RecoveryStrategySelector()

    decision = selector.decide_for_task(record(recoverable=False))

    assert decision.strategy == RecoveryStrategy.ABANDON_AND_LOG
    assert decision.status == RecoveryStatus.ABANDONED
    assert decision.user_visible is True


def test_reconstructor_returns_empty_without_checkpoint() -> None:
    reconstructor = StateReconstructor()

    reconstructed = reconstructor.reconstruct(checkpoint=None, events=())

    assert reconstructed.state == {}
    assert reconstructed.reason == RecoveryReason.NO_CHECKPOINT_AVAILABLE


def test_reconstructor_replays_events_after_checkpoint() -> None:
    manager = RecoveryManager()

    checkpoint_result = manager.checkpoint(
        sequence=1,
        state={"active_task": "task-1"},
        force=True,
    )
    assert checkpoint_result.checkpoint is not None

    events = (
        RecoveryEvent(
            sequence=2,
            event_type=RecoveryEventType.STATE_SET,
            payload={"key": "active_task", "value": "task-2"},
        ),
        RecoveryEvent(
            sequence=3,
            event_type=RecoveryEventType.STATE_SET,
            payload={"key": "mode", "value": "recovered"},
        ),
    )

    reconstructor = StateReconstructor()
    reconstructed = reconstructor.reconstruct(
        checkpoint=checkpoint_result.checkpoint,
        events=events,
    )

    assert reconstructed.state["active_task"] == "task-2"
    assert reconstructed.state["mode"] == "recovered"
    assert reconstructed.replayed_event_count == 2


def test_manager_skips_checkpoint_before_interval() -> None:
    manager = RecoveryManager(
        config=RecoveryRuntimeConfig(checkpoint_interval_seconds=30)
    )

    result = manager.checkpoint(
        sequence=1,
        state={"x": 1},
        elapsed_since_last_seconds=10,
    )

    assert result.reason == RecoveryReason.CHECKPOINT_SKIPPED
    assert manager.snapshot().checkpoint_count == 0


def test_manager_creates_checkpoint_when_forced() -> None:
    manager = RecoveryManager()

    result = manager.checkpoint(
        sequence=1,
        state={"x": 1},
        force=True,
    )

    assert result.success is True
    assert result.reason == RecoveryReason.CHECKPOINT_CREATED
    assert result.checkpoint is not None
    assert manager.snapshot().checkpoint_count == 1


def test_manager_appends_event() -> None:
    manager = RecoveryManager()

    result = manager.append_event(
        sequence=1,
        event_type=RecoveryEventType.STATE_SET,
        payload={"key": "x", "value": 2},
    )

    assert result.reason == RecoveryReason.EVENT_RECORDED
    assert manager.snapshot().event_count == 1


def test_manager_reconstructs_from_checkpoint_and_event_log() -> None:
    manager = RecoveryManager()

    manager.checkpoint(
        sequence=1,
        state={"active_task": "task-1", "old": True},
        force=True,
    )
    manager.append_event(
        sequence=2,
        event_type=RecoveryEventType.STATE_SET,
        payload={"key": "active_task", "value": "task-2"},
    )
    manager.append_event(
        sequence=3,
        event_type=RecoveryEventType.STATE_DELETE,
        payload={"key": "old"},
    )

    result = manager.reconstruct_last_known_good_state()

    assert result.success is True
    assert result.reconstructed_state is not None
    assert result.reconstructed_state.state["active_task"] == "task-2"
    assert "old" not in result.reconstructed_state.state
    assert result.reconstructed_state.replayed_event_count == 2


def test_manager_reconstruction_without_checkpoint_fails_safely() -> None:
    manager = RecoveryManager()

    result = manager.reconstruct_last_known_good_state()

    assert result.success is False
    assert result.reason == RecoveryReason.NO_CHECKPOINT_AVAILABLE
    assert result.reconstructed_state is not None
    assert result.reconstructed_state.state == {}


def test_manager_records_recovery_decision() -> None:
    manager = RecoveryManager()

    result = manager.recover_task(record(failure_count=1))

    assert result.success is True
    assert result.reason == RecoveryReason.RECOVERY_RECORDED
    assert result.decision is not None
    assert result.decision.strategy == RecoveryStrategy.RETRY_WITH_BACKOFF
    assert manager.snapshot().audit_count == 1


def test_manager_quarantines_bad_task() -> None:
    manager = RecoveryManager(
        config=RecoveryRuntimeConfig(quarantine_after_failures=2)
    )

    result = manager.recover_task(record(task_id="bad-task", failure_count=2))

    assert result.decision is not None
    assert result.decision.status == RecoveryStatus.QUARANTINED
    assert manager.is_quarantined("bad-task") is True
    assert manager.snapshot().quarantined_count == 1


def test_manager_manual_quarantine() -> None:
    manager = RecoveryManager()

    result = manager.quarantine_task(record(task_id="task-q"))

    assert result.decision is not None
    assert result.decision.status == RecoveryStatus.QUARANTINED
    assert manager.is_quarantined("task-q") is True


def test_manager_detects_stale_assignments() -> None:
    manager = RecoveryManager()

    stale = manager.detect_stale_assignments(
        (
            record(task_id="fresh", age_seconds=10),
            record(task_id="stale", age_seconds=130),
        )
    )

    assert len(stale) == 1
    assert stale[0].task_id == "stale"


def test_manager_snapshot_tracks_counts() -> None:
    manager = RecoveryManager()

    manager.checkpoint(sequence=1, state={"x": 1}, force=True)
    manager.append_event(
        sequence=2,
        event_type=RecoveryEventType.STATE_SET,
        payload={"key": "x", "value": 2},
    )
    manager.recover_task(record(failure_count=1))

    snapshot = manager.snapshot()

    assert snapshot.checkpoint_count == 1
    assert snapshot.event_count == 1
    assert snapshot.audit_count >= 2
    assert snapshot.last_reason is not None


def test_manager_reset_clears_state() -> None:
    manager = RecoveryManager()

    manager.checkpoint(sequence=1, state={"x": 1}, force=True)
    manager.append_event(
        sequence=2,
        event_type=RecoveryEventType.STATE_SET,
        payload={"key": "x", "value": 2},
    )
    manager.quarantine_task(record(task_id="q"))

    manager.reset()
    snapshot = manager.snapshot()

    assert snapshot.checkpoint_count == 0
    assert snapshot.event_count == 0
    assert snapshot.audit_count == 0
    assert snapshot.quarantined_count == 0
    assert snapshot.last_reason == RecoveryReason.RUNTIME_RESET


def test_sqlite_file_store_survives_manager_recreation(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "recovery.sqlite"

    first = RecoveryManager(
        config=RecoveryRuntimeConfig(sqlite_path=str(sqlite_path))
    )
    first.checkpoint(sequence=1, state={"mode": "before"}, force=True)
    first.append_event(
        sequence=2,
        event_type=RecoveryEventType.STATE_SET,
        payload={"key": "mode", "value": "after"},
    )
    first.close()

    second = RecoveryManager(
        config=RecoveryRuntimeConfig(sqlite_path=str(sqlite_path))
    )
    result = second.reconstruct_last_known_good_state()

    assert result.reconstructed_state is not None
    assert result.reconstructed_state.state["mode"] == "after"
    second.close()


def test_recovery_decision_requires_user_message_when_visible() -> None:
    from jarvis.orchestration import RecoveryDecision

    with pytest.raises(ValidationError):
        RecoveryDecision(
            strategy=RecoveryStrategy.ABANDON_AND_LOG,
            status=RecoveryStatus.ABANDONED,
            reason=RecoveryReason.TASK_UNRECOVERABLE,
            message="abandoned",
            user_visible=True,
        )


def test_enum_values_are_stable() -> None:
    assert RecoveryStrategy.RETRY_WITH_BACKOFF.value == "retry_with_backoff"
    assert RecoveryStatus.RECOVERED.value == "recovered"
    assert RecoveryFailureKind.WORKER_CRASH.value == "worker_crash"
    assert RecoveryReason.STATE_RECONSTRUCTED.value == "state_reconstructed"
    assert RecoveryEventType.STATE_SET.value == "state_set"