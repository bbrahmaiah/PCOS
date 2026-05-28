from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    EnvironmentBackpressureController,
    VisualArbitrationReason,
    VisualLoadLevel,
    VisualPriorityArbitrator,
    VisualTaskDecision,
    VisualTaskKind,
    VisualTaskPriority,
    VisualTaskRequest,
)


def test_arbitrator_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        VisualPriorityArbitrator(name=" ")


def test_task_request_requires_positive_budgets() -> None:
    with pytest.raises(ValidationError):
        VisualTaskRequest(
            kind=VisualTaskKind.CAPTURE,
            requested_latency_ms=0,
            requested_cpu_percent=5,
            requested_memory_mb=64,
        )

    with pytest.raises(ValidationError):
        VisualTaskRequest(
            kind=VisualTaskKind.OCR,
            requested_latency_ms=100,
            requested_cpu_percent=101,
            requested_memory_mb=64,
        )


def test_priority_order_for_critical_conversation_and_interruption() -> None:
    arbitrator = VisualPriorityArbitrator()

    assert arbitrator.priority_for(VisualTaskKind.CONVERSATION) == (
        VisualTaskPriority.CRITICAL
    )
    assert arbitrator.priority_for(VisualTaskKind.INTERRUPTION) == (
        VisualTaskPriority.CRITICAL
    )
    assert arbitrator.priority_for(VisualTaskKind.STT) == (
        VisualTaskPriority.CRITICAL
    )
    assert arbitrator.priority_for(VisualTaskKind.TTS) == (
        VisualTaskPriority.CRITICAL
    )


def test_priority_order_for_verification_and_recovery() -> None:
    arbitrator = VisualPriorityArbitrator()

    assert arbitrator.priority_for(VisualTaskKind.VERIFICATION) == (
        VisualTaskPriority.HIGH
    )
    assert arbitrator.priority_for(VisualTaskKind.RECOVERY) == (
        VisualTaskPriority.HIGH
    )


def test_priority_order_for_background_visual_work() -> None:
    arbitrator = VisualPriorityArbitrator()

    assert arbitrator.priority_for(VisualTaskKind.GRAPH_REFRESH) == (
        VisualTaskPriority.LOW
    )
    assert arbitrator.priority_for(VisualTaskKind.OCR) == (
        VisualTaskPriority.BACKGROUND
    )
    assert arbitrator.priority_for(VisualTaskKind.MEMORY_CONSOLIDATION) == (
        VisualTaskPriority.DEFERRED
    )


def test_normal_task_runs_when_no_pressure() -> None:
    arbitrator = VisualPriorityArbitrator()

    decision = arbitrator.arbitrate(_request(VisualTaskKind.CAPTURE))

    assert decision.decision == VisualTaskDecision.RUN_NOW
    assert decision.reason == VisualArbitrationReason.TASK_ACCEPTED
    assert decision.budget is not None


def test_conversation_blocks_background_ocr() -> None:
    arbitrator = VisualPriorityArbitrator()
    arbitrator.update_backpressure(
        EnvironmentBackpressureController(conversation_active=True)
    )

    decision = arbitrator.arbitrate(_request(VisualTaskKind.OCR))

    assert decision.decision == VisualTaskDecision.DEFER
    assert decision.reason == VisualArbitrationReason.TASK_BLOCKED_BY_CONVERSATION


def test_conversation_sheds_sheddable_background_work() -> None:
    arbitrator = VisualPriorityArbitrator()
    arbitrator.update_backpressure(
        EnvironmentBackpressureController(conversation_active=True)
    )

    decision = arbitrator.arbitrate(
        _request(VisualTaskKind.OCR, can_shed=True)
    )

    assert decision.decision == VisualTaskDecision.SHED
    assert decision.reason == VisualArbitrationReason.TASK_SHED_BY_BACKPRESSURE


def test_interruption_defers_non_critical_work() -> None:
    arbitrator = VisualPriorityArbitrator()
    arbitrator.update_backpressure(
        EnvironmentBackpressureController(interruption_active=True)
    )

    decision = arbitrator.arbitrate(_request(VisualTaskKind.CAPTURE))

    assert decision.decision == VisualTaskDecision.DEFER
    assert decision.reason == VisualArbitrationReason.TASK_BLOCKED_BY_INTERRUPTION


def test_interruption_allows_interruption_task() -> None:
    arbitrator = VisualPriorityArbitrator()
    arbitrator.update_backpressure(
        EnvironmentBackpressureController(interruption_active=True)
    )

    decision = arbitrator.arbitrate(_request(VisualTaskKind.INTERRUPTION))

    assert decision.decision == VisualTaskDecision.RUN_NOW
    assert decision.priority == VisualTaskPriority.CRITICAL


def test_critical_load_sheds_background_tasks() -> None:
    arbitrator = VisualPriorityArbitrator()
    arbitrator.update_backpressure(
        EnvironmentBackpressureController(load_level=VisualLoadLevel.CRITICAL)
    )

    decision = arbitrator.arbitrate(_request(VisualTaskKind.OCR))

    assert decision.decision == VisualTaskDecision.SHED
    assert decision.reason == VisualArbitrationReason.TASK_SHED_BY_BACKPRESSURE


def test_high_load_limits_degradable_normal_tasks() -> None:
    arbitrator = VisualPriorityArbitrator()
    arbitrator.update_backpressure(
        EnvironmentBackpressureController(load_level=VisualLoadLevel.HIGH)
    )

    decision = arbitrator.arbitrate(_request(VisualTaskKind.CAPTURE))

    assert decision.decision == VisualTaskDecision.RUN_LIMITED
    assert decision.reason == VisualArbitrationReason.TASK_LIMITED_BY_BUDGET
    assert decision.budget is not None
    assert decision.budget.latency_budget_ms < 50.0


def test_budget_overrun_limits_degradable_task() -> None:
    arbitrator = VisualPriorityArbitrator()

    decision = arbitrator.arbitrate(
        _request(
            VisualTaskKind.CAPTURE,
            latency_ms=200,
            cpu_percent=20,
            memory_mb=256,
        )
    )

    assert decision.decision == VisualTaskDecision.RUN_LIMITED
    assert decision.reason == VisualArbitrationReason.TASK_LIMITED_BY_BUDGET


def test_budget_overrun_defers_non_degradable_task() -> None:
    arbitrator = VisualPriorityArbitrator()

    decision = arbitrator.arbitrate(
        _request(
            VisualTaskKind.CAPTURE,
            latency_ms=200,
            cpu_percent=20,
            memory_mb=256,
            can_degrade=False,
        )
    )

    assert decision.decision == VisualTaskDecision.DEFER
    assert decision.reason == VisualArbitrationReason.TASK_DEFERRED_BY_PRIORITY


def test_verification_remains_high_priority_under_conversation() -> None:
    arbitrator = VisualPriorityArbitrator()
    arbitrator.update_backpressure(
        EnvironmentBackpressureController(conversation_active=True)
    )

    decision = arbitrator.arbitrate(_request(VisualTaskKind.VERIFICATION))

    assert decision.decision == VisualTaskDecision.RUN_NOW
    assert decision.priority == VisualTaskPriority.HIGH


def test_snapshot_tracks_decisions() -> None:
    arbitrator = VisualPriorityArbitrator()

    arbitrator.arbitrate(_request(VisualTaskKind.CAPTURE))
    arbitrator.update_backpressure(
        EnvironmentBackpressureController(conversation_active=True)
    )
    arbitrator.arbitrate(_request(VisualTaskKind.OCR))
    snapshot = arbitrator.snapshot()

    assert snapshot.decision_count == 2
    assert snapshot.run_now_count == 1
    assert snapshot.deferred_count == 1
    assert snapshot.conversation_active is True
    assert snapshot.event_count == 3


def test_reset_clears_runtime_state() -> None:
    arbitrator = VisualPriorityArbitrator()

    arbitrator.arbitrate(_request(VisualTaskKind.CAPTURE))
    arbitrator.reset()
    snapshot = arbitrator.snapshot()

    assert snapshot.decision_count == 0
    assert snapshot.event_count == 1
    assert snapshot.last_reason == VisualArbitrationReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert VisualTaskKind.CONVERSATION.value == "conversation"
    assert VisualTaskPriority.CRITICAL.value == "critical"
    assert VisualTaskDecision.RUN_LIMITED.value == "run_limited"
    assert VisualLoadLevel.SHEDDING.value == "shedding"


def _request(
    kind: VisualTaskKind,
    *,
    latency_ms: float = 20.0,
    cpu_percent: float = 4.0,
    memory_mb: float = 64.0,
    can_degrade: bool = True,
    can_shed: bool = False,
) -> VisualTaskRequest:
    return VisualTaskRequest(
        kind=kind,
        requested_latency_ms=latency_ms,
        requested_cpu_percent=cpu_percent,
        requested_memory_mb=memory_mb,
        can_degrade=can_degrade,
        can_shed=can_shed,
    )