from __future__ import annotations

import pytest

from jarvis.cognitive import (
    AttentionDecision,
    AttentionEvaluationRequest,
    AttentionItemKind,
    AttentionPriority,
    AttentionRuntime,
    AttentionRuntimeStatus,
    AttentionSignal,
    AttentionSignalSource,
    AttentionSignalUrgency,
    make_attention_signal,
)


def test_attention_signal_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError):
        make_attention_signal(
            source=AttentionSignalSource.SYSTEM,
            kind=AttentionItemKind.SYSTEM_HEALTH,
            title="Battery",
            summary="Battery critical",
            urgency=AttentionSignalUrgency.EMERGENCY,
            confidence=2.0,
        )


def test_attention_runtime_degrades_when_no_signals() -> None:
    runtime = AttentionRuntime()

    result = runtime.evaluate(AttentionEvaluationRequest(signals=()))

    assert result.status == AttentionRuntimeStatus.DEGRADED
    assert result.decision == AttentionDecision.IGNORE
    assert result.selected_item is None
    assert result.state.items == ()


def test_attention_runtime_tracks_normal_signal() -> None:
    runtime = AttentionRuntime()
    signal = make_attention_signal(
        source=AttentionSignalSource.PROJECT,
        kind=AttentionItemKind.PROJECT,
        title="Build finished",
        summary="Background build completed.",
        urgency=AttentionSignalUrgency.NORMAL,
        confidence=0.9,
    )

    result = runtime.evaluate(
        AttentionEvaluationRequest(signals=(signal,))
    )

    assert result.status == AttentionRuntimeStatus.READY
    assert result.decision == AttentionDecision.TRACK
    assert result.selected_item is not None
    assert result.selected_item.priority == AttentionPriority.NORMAL
    assert result.should_interrupt is False


def test_attention_runtime_focuses_important_signal() -> None:
    runtime = AttentionRuntime()
    signal = make_attention_signal(
        source=AttentionSignalSource.PROJECT,
        kind=AttentionItemKind.ACTIVE_TASK,
        title="Tests failed",
        summary="The current test suite failed.",
        urgency=AttentionSignalUrgency.IMPORTANT,
        confidence=0.95,
    )

    result = runtime.evaluate(
        AttentionEvaluationRequest(signals=(signal,))
    )

    assert result.decision == AttentionDecision.FOCUS
    assert result.selected_item is not None
    assert result.selected_item.priority == AttentionPriority.HIGH
    assert result.state.has_focus is True


def test_attention_runtime_interrupts_emergency_signal() -> None:
    runtime = AttentionRuntime()
    signal = make_attention_signal(
        source=AttentionSignalSource.SAFETY,
        kind=AttentionItemKind.SAFETY,
        title="Unsafe delete requested",
        summary="A destructive action requires immediate stop.",
        urgency=AttentionSignalUrgency.EMERGENCY,
        confidence=1.0,
    )

    result = runtime.evaluate(
        AttentionEvaluationRequest(
            signals=(signal,),
            assistant_is_speaking=True,
            allow_interruptions=True,
        )
    )

    assert result.decision == AttentionDecision.INTERRUPT_NOW
    assert result.should_interrupt is True
    assert result.selected_item is not None
    assert result.selected_item.priority == AttentionPriority.CRITICAL
    assert runtime.snapshot().interruption_count == 1


def test_attention_runtime_does_not_interrupt_when_disabled() -> None:
    runtime = AttentionRuntime()
    signal = make_attention_signal(
        source=AttentionSignalSource.SAFETY,
        kind=AttentionItemKind.SAFETY,
        title="Battery critical",
        summary="Battery is at 3 percent.",
        urgency=AttentionSignalUrgency.EMERGENCY,
        confidence=1.0,
    )

    result = runtime.evaluate(
        AttentionEvaluationRequest(
            signals=(signal,),
            allow_interruptions=False,
        )
    )

    assert result.decision == AttentionDecision.FOCUS
    assert result.should_interrupt is False


def test_attention_runtime_tracks_high_priority_while_user_speaks() -> None:
    runtime = AttentionRuntime()
    signal = make_attention_signal(
        source=AttentionSignalSource.NOTIFICATION,
        kind=AttentionItemKind.NOTIFICATION,
        title="Important message",
        summary="A project message arrived.",
        urgency=AttentionSignalUrgency.URGENT,
        confidence=0.8,
    )

    result = runtime.evaluate(
        AttentionEvaluationRequest(
            signals=(signal,),
            user_is_speaking=True,
        )
    )

    assert result.decision == AttentionDecision.TRACK
    assert result.should_interrupt is False
    assert "user is currently speaking" in result.reason


def test_attention_runtime_prefers_critical_over_normal() -> None:
    runtime = AttentionRuntime()
    normal = make_attention_signal(
        source=AttentionSignalSource.PROJECT,
        kind=AttentionItemKind.PROJECT,
        title="Build complete",
        summary="Build completed.",
        urgency=AttentionSignalUrgency.NORMAL,
        confidence=1.0,
    )
    critical = make_attention_signal(
        source=AttentionSignalSource.SYSTEM,
        kind=AttentionItemKind.SYSTEM_HEALTH,
        title="Battery critical",
        summary="Battery below safe threshold.",
        urgency=AttentionSignalUrgency.EMERGENCY,
        confidence=1.0,
    )

    result = runtime.evaluate(
        AttentionEvaluationRequest(signals=(normal, critical))
    )

    assert result.selected_item is not None
    assert result.selected_item.title == "Battery critical"
    assert result.decision == AttentionDecision.INTERRUPT_NOW


def test_attention_runtime_merges_duplicate_items_by_priority() -> None:
    runtime = AttentionRuntime()
    low = make_attention_signal(
        source=AttentionSignalSource.PROJECT,
        kind=AttentionItemKind.PROJECT,
        title="Build status",
        summary="Build is running.",
        urgency=AttentionSignalUrgency.NORMAL,
        confidence=0.8,
    )
    high = AttentionSignal(
        signal_id=low.signal_id,
        source=low.source,
        kind=low.kind,
        title=low.title,
        summary="Build failed.",
        urgency=AttentionSignalUrgency.URGENT,
        confidence=1.0,
        created_at=low.created_at,
        metadata={},
    )

    result = runtime.evaluate(
        AttentionEvaluationRequest(signals=(low, high))
    )

    assert len(result.state.items) == 1
    assert result.state.items[0].priority == AttentionPriority.HIGH


def test_attention_runtime_clear_resets_state() -> None:
    runtime = AttentionRuntime()
    signal = make_attention_signal(
        source=AttentionSignalSource.USER,
        kind=AttentionItemKind.USER_COMMAND,
        title="Run tests",
        summary="User requested test run.",
        urgency=AttentionSignalUrgency.IMPORTANT,
    )

    runtime.evaluate(AttentionEvaluationRequest(signals=(signal,)))
    result = runtime.clear()

    assert result.state.items == ()
    assert result.decision == AttentionDecision.IGNORE


def test_attention_runtime_snapshot_tracks_evaluations() -> None:
    runtime = AttentionRuntime()
    signal = make_attention_signal(
        source=AttentionSignalSource.USER,
        kind=AttentionItemKind.USER_COMMAND,
        title="Continue work",
        summary="User asked to continue.",
        urgency=AttentionSignalUrgency.IMPORTANT,
    )

    runtime.evaluate(AttentionEvaluationRequest(signals=(signal,)))
    snapshot = runtime.snapshot()

    assert snapshot.status == AttentionRuntimeStatus.READY
    assert snapshot.evaluated_count == 1
    assert snapshot.last_decision == AttentionDecision.FOCUS


def test_attention_enum_values_are_stable() -> None:
    assert AttentionSignalUrgency.EMERGENCY.value == "emergency"
    assert AttentionSignalSource.SAFETY.value == "safety"
    assert AttentionRuntimeStatus.READY.value == "ready"