from __future__ import annotations

from datetime import timedelta

import pytest
from pydantic import ValidationError

from jarvis.orchestration import (
    InterruptAcknowledgement,
    InterruptEvent,
    InterruptKind,
    InterruptPropagationDecision,
    InterruptPropagationReason,
    InterruptPropagationStatus,
    InterruptPropagator,
    InterruptPropagatorConfig,
    PropagationOrder,
    PropagationPhase,
    PropagationTarget,
    WorkerCapability,
    new_task_id,
    new_worker_id,
    propagation_target_for_capability,
)
from jarvis.orchestration.ids import utc_now


def target(
    capability: WorkerCapability,
    *,
    task_id: str | None = None,
    worker_id: str | None = None,
    rollback_supported: bool = False,
) -> PropagationTarget:
    return propagation_target_for_capability(
        capability=capability,
        worker_id=worker_id or new_worker_id(),
        task_ids=(task_id or new_task_id(),),
        rollback_supported=rollback_supported,
    )


def event(*, timeout_ms: int = 2_000) -> InterruptEvent:
    return InterruptEvent(
        kind=InterruptKind.USER_INTERRUPT,
        reason="user interrupted speech",
        affected_task_ids=(new_task_id(),),
        timeout_ms=timeout_ms,
    )


def ack_for(
    *,
    interrupt_id: str,
    dispatch_id: str,
    target_item: PropagationTarget,
) -> InterruptAcknowledgement:
    return InterruptAcknowledgement(
        interrupt_id=interrupt_id,
        dispatch_id=dispatch_id,
        worker_id=target_item.worker_id,
        capability=target_item.capability,
        task_ids=target_item.task_ids,
        rollback_started=target_item.rollback_supported,
    )


def test_config_rejects_invalid_timeout() -> None:
    with pytest.raises(ValueError):
        InterruptPropagatorConfig(orphan_timeout_ms=0).validate()


def test_event_requires_reason() -> None:
    with pytest.raises(ValidationError):
        InterruptEvent(kind=InterruptKind.USER_INTERRUPT, reason=" ")


def test_target_phase_must_match_capability() -> None:
    with pytest.raises(ValidationError):
        PropagationTarget(
            phase=PropagationPhase.BACKGROUND,
            capability=WorkerCapability.COGNITION,
            worker_id=new_worker_id(),
            task_ids=(new_task_id(),),
        )


def test_propagation_order_sorts_targets() -> None:
    background = target(WorkerCapability.BACKGROUND)
    presence = target(WorkerCapability.PRESENCE)
    cognition = target(WorkerCapability.COGNITION)

    order = PropagationOrder.ordered((background, cognition, presence))

    assert order.targets[0].phase == PropagationPhase.PRESENCE
    assert order.targets[1].phase == PropagationPhase.COGNITION
    assert order.targets[2].phase == PropagationPhase.BACKGROUND


def test_start_dispatches_first_target_only() -> None:
    first = target(WorkerCapability.PRESENCE)
    second = target(WorkerCapability.COGNITION)
    interrupt = event()
    runtime = InterruptPropagator()

    result = runtime.start(
        event=interrupt,
        order=PropagationOrder.ordered((second, first)),
    )

    assert result.success is True
    assert result.decision == InterruptPropagationDecision.DISPATCHED
    assert result.dispatch is not None
    assert result.dispatch.target.phase == PropagationPhase.PRESENCE
    assert result.record is not None
    assert len(result.record.dispatches) == 1


def test_duplicate_interrupt_rejected() -> None:
    first = target(WorkerCapability.PRESENCE)
    interrupt = event()
    runtime = InterruptPropagator()

    runtime.start(event=interrupt, order=PropagationOrder.ordered((first,)))
    result = runtime.start(event=interrupt, order=PropagationOrder.ordered((first,)))

    assert result.success is False
    assert result.reason == InterruptPropagationReason.INTERRUPT_ALREADY_ACTIVE


def test_acknowledgement_dispatches_next_target() -> None:
    first = target(WorkerCapability.PRESENCE)
    second = target(WorkerCapability.COGNITION)
    interrupt = event()
    runtime = InterruptPropagator()

    started = runtime.start(
        event=interrupt,
        order=PropagationOrder.ordered((first, second)),
    )

    assert started.dispatch is not None

    acknowledged = runtime.acknowledge(
        ack_for(
            interrupt_id=interrupt.interrupt_id,
            dispatch_id=started.dispatch.dispatch_id,
            target_item=first,
        )
    )

    assert acknowledged.success is True
    assert acknowledged.decision == InterruptPropagationDecision.ACKNOWLEDGED
    assert acknowledged.dispatch is not None
    assert acknowledged.dispatch.target.phase == PropagationPhase.COGNITION


def test_final_ack_completes_interrupt() -> None:
    first = target(WorkerCapability.PRESENCE)
    interrupt = event()
    runtime = InterruptPropagator()

    started = runtime.start(
        event=interrupt,
        order=PropagationOrder.ordered((first,)),
    )

    assert started.dispatch is not None

    completed = runtime.acknowledge(
        ack_for(
            interrupt_id=interrupt.interrupt_id,
            dispatch_id=started.dispatch.dispatch_id,
            target_item=first,
        )
    )

    assert completed.success is True
    assert completed.decision == InterruptPropagationDecision.COMPLETED
    assert completed.record is not None
    assert completed.record.status == InterruptPropagationStatus.COMPLETED


def test_ack_unknown_interrupt_rejected() -> None:
    first = target(WorkerCapability.PRESENCE)
    runtime = InterruptPropagator()

    result = runtime.acknowledge(
        InterruptAcknowledgement(
            interrupt_id="interrupt_missing",
            dispatch_id="intdispatch_missing",
            worker_id=first.worker_id,
            capability=first.capability,
            task_ids=first.task_ids,
        )
    )

    assert result.success is False
    assert result.reason == InterruptPropagationReason.INTERRUPT_NOT_FOUND


def test_wrong_order_ack_rejected() -> None:
    first = target(WorkerCapability.PRESENCE)
    second = target(WorkerCapability.COGNITION)
    interrupt = event()
    runtime = InterruptPropagator()

    started = runtime.start(
        event=interrupt,
        order=PropagationOrder.ordered((first, second)),
    )

    assert started.dispatch is not None

    result = runtime.acknowledge(
        InterruptAcknowledgement(
            interrupt_id=interrupt.interrupt_id,
            dispatch_id="intdispatch_wrong",
            worker_id=first.worker_id,
            capability=first.capability,
            task_ids=first.task_ids,
        )
    )

    assert result.success is False
    assert result.reason == InterruptPropagationReason.WRONG_ORDER_ACKNOWLEDGED


def test_wrong_worker_ack_rejected() -> None:
    first = target(WorkerCapability.PRESENCE)
    interrupt = event()
    runtime = InterruptPropagator()

    started = runtime.start(event=interrupt, order=PropagationOrder.ordered((first,)))

    assert started.dispatch is not None

    result = runtime.acknowledge(
        InterruptAcknowledgement(
            interrupt_id=interrupt.interrupt_id,
            dispatch_id=started.dispatch.dispatch_id,
            worker_id=new_worker_id(),
            capability=first.capability,
            task_ids=first.task_ids,
        )
    )

    assert result.success is False
    assert result.reason == InterruptPropagationReason.WRONG_WORKER_ACKNOWLEDGED


def test_escalate_orphaned_dispatch_after_timeout() -> None:
    first = target(WorkerCapability.COGNITION, rollback_supported=True)
    interrupt = event(timeout_ms=2_000)
    runtime = InterruptPropagator()

    started = runtime.start(event=interrupt, order=PropagationOrder.ordered((first,)))

    assert started.dispatch is not None
    assert isinstance(started.dispatch.sent_at, type(utc_now()))

    future = started.dispatch.sent_at + timedelta(milliseconds=2_001)
    escalated = runtime.escalate_orphans(
        interrupt_id=interrupt.interrupt_id,
        now=future,
    )

    assert escalated.success is True
    assert escalated.decision == InterruptPropagationDecision.ESCALATED
    assert (
        escalated.reason
        == InterruptPropagationReason.INTERRUPT_ESCALATED_ORPHANED_TASKS
    )
    assert escalated.orphaned is not None
    assert escalated.orphaned.worker_id == first.worker_id


def test_escalate_before_timeout_rejected() -> None:
    first = target(WorkerCapability.COGNITION)
    interrupt = event(timeout_ms=2_000)
    runtime = InterruptPropagator()

    runtime.start(event=interrupt, order=PropagationOrder.ordered((first,)))
    result = runtime.escalate_orphans(interrupt_id=interrupt.interrupt_id)

    assert result.success is False
    assert result.reason == InterruptPropagationReason.INVALID_ACKNOWLEDGEMENT


def test_force_escalate_orphaned_dispatch() -> None:
    first = target(WorkerCapability.COGNITION)
    interrupt = event()
    runtime = InterruptPropagator()

    runtime.start(event=interrupt, order=PropagationOrder.ordered((first,)))
    result = runtime.escalate_orphans(
        interrupt_id=interrupt.interrupt_id,
        force=True,
    )

    assert result.success is True
    assert result.orphaned is not None


def test_rollback_metadata_is_visible_on_dispatch() -> None:
    first = target(WorkerCapability.TOOL_ACTION, rollback_supported=True)
    interrupt = event()
    runtime = InterruptPropagator()

    result = runtime.start(event=interrupt, order=PropagationOrder.ordered((first,)))

    assert result.dispatch is not None
    assert result.dispatch.metadata["rollback_requested"] is True
    assert result.dispatch.metadata["rollback_supported"] is True


def test_snapshot_counts_interrupt_activity() -> None:
    first = target(WorkerCapability.PRESENCE)
    interrupt = event()
    runtime = InterruptPropagator()

    started = runtime.start(event=interrupt, order=PropagationOrder.ordered((first,)))

    assert started.dispatch is not None

    runtime.acknowledge(
        ack_for(
            interrupt_id=interrupt.interrupt_id,
            dispatch_id=started.dispatch.dispatch_id,
            target_item=first,
        )
    )
    snapshot = runtime.snapshot()

    assert snapshot.completed_count == 1
    assert snapshot.dispatch_count == 1
    assert snapshot.acknowledgement_count == 1
    assert snapshot.last_reason == InterruptPropagationReason.INTERRUPT_COMPLETED


def test_active_records_excludes_completed() -> None:
    first = target(WorkerCapability.PRESENCE)
    interrupt = event()
    runtime = InterruptPropagator()

    started = runtime.start(event=interrupt, order=PropagationOrder.ordered((first,)))

    assert started.dispatch is not None
    assert len(runtime.active_records()) == 1

    runtime.acknowledge(
        ack_for(
            interrupt_id=interrupt.interrupt_id,
            dispatch_id=started.dispatch.dispatch_id,
            target_item=first,
        )
    )

    assert runtime.active_records() == ()


def test_reset_clears_state() -> None:
    first = target(WorkerCapability.PRESENCE)
    interrupt = event()
    runtime = InterruptPropagator()

    runtime.start(event=interrupt, order=PropagationOrder.ordered((first,)))
    runtime.reset()

    snapshot = runtime.snapshot()

    assert snapshot.active_interrupt_count == 0
    assert snapshot.dispatch_count == 0


def test_enum_values_are_stable() -> None:
    assert InterruptKind.USER_INTERRUPT.value == "user_interrupt"
    assert InterruptPropagationStatus.WAITING_ACK.value == "waiting_ack"
    assert InterruptPropagationDecision.DISPATCHED.value == "dispatched"
    assert InterruptPropagationReason.INTERRUPT_COMPLETED.value
    assert PropagationPhase.PRESENCE.value == 0