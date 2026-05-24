from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.conversation import (
    AttentionDecision,
    AttentionDisposition,
    AttentionPriority,
    AttentionRuntime,
    AttentionRuntimeConfig,
    AttentionSignal,
    AttentionSignalKind,
    AttentionTarget,
    AttentionTargetKind,
    AttentionTargetStatus,
    ConversationFollowUpExpectation,
    ConversationMode,
    ConversationSessionRuntime,
    ConversationState,
    TurnUrgency,
)


def signal(
    *,
    kind: AttentionSignalKind = AttentionSignalKind.USER_TURN,
    text: str = "build adaptive conversation runtime",
    priority: AttentionPriority = AttentionPriority.NORMAL,
    urgency: TurnUrgency = TurnUrgency.NORMAL,
    state: ConversationState | None = None,
    target_id: str | None = None,
) -> AttentionSignal:
    return AttentionSignal(
        kind=kind,
        text=text,
        priority=priority,
        urgency=urgency,
        state=state,
        target_id=target_id,
    )


def test_attention_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        AttentionRuntimeConfig(name=" ").validate()

    with pytest.raises(ValueError):
        AttentionRuntimeConfig(max_targets=0).validate()

    with pytest.raises(ValueError):
        AttentionRuntimeConfig(focus_threshold=1.5).validate()

    with pytest.raises(ValueError):
        AttentionRuntimeConfig(
            focus_threshold=0.8,
            interrupt_threshold=0.7,
        ).validate()


def test_attention_signal_requires_text() -> None:
    with pytest.raises(ValidationError):
        AttentionSignal(
            kind=AttentionSignalKind.USER_TURN,
            text=" ",
        )


def test_attention_target_requires_label() -> None:
    with pytest.raises(ValidationError):
        AttentionTarget(
            kind=AttentionTargetKind.CONVERSATION,
            label=" ",
        )


def test_low_priority_background_task_is_backgrounded() -> None:
    runtime = AttentionRuntime()

    decision = runtime.submit_signal(
        signal(
            kind=AttentionSignalKind.BACKGROUND_TASK,
            text="background memory compaction",
            priority=AttentionPriority.LOW,
        )
    )

    assert decision.disposition == AttentionDisposition.BACKGROUND
    assert decision.should_start_cognition is False
    assert decision.should_interrupt is False


def test_normal_user_turn_becomes_monitor_or_focus() -> None:
    runtime = AttentionRuntime()

    decision = runtime.submit_signal(
        signal(priority=AttentionPriority.HIGH)
    )

    assert decision.disposition in {
        AttentionDisposition.FOCUS,
        AttentionDisposition.MONITOR,
    }
    assert decision.selected_target is not None


def test_high_priority_user_turn_sets_focus() -> None:
    runtime = AttentionRuntime()

    decision = runtime.submit_signal(
        signal(
            priority=AttentionPriority.HIGH,
            urgency=TurnUrgency.HIGH,
        )
    )
    focus = runtime.current_focus()

    assert decision.disposition == AttentionDisposition.FOCUS
    assert decision.should_start_cognition is True
    assert focus is not None
    assert focus.label == "build adaptive conversation runtime"


def test_critical_signal_interrupts() -> None:
    runtime = AttentionRuntime()

    decision = runtime.submit_signal(
        signal(
            kind=AttentionSignalKind.INTERRUPTION,
            text="stop",
            priority=AttentionPriority.CRITICAL,
            urgency=TurnUrgency.CRITICAL,
            state=ConversationState.INTERRUPTED,
        )
    )

    assert decision.disposition == AttentionDisposition.INTERRUPT
    assert decision.should_interrupt is True
    assert decision.should_start_cognition is True
    assert runtime.snapshot().interrupt_count == 1


def test_focus_change_background_existing_flag() -> None:
    runtime = AttentionRuntime()

    first = runtime.submit_signal(
        signal(
            text="memory runtime",
            priority=AttentionPriority.HIGH,
            target_id="target-1",
        )
    )
    second = runtime.submit_signal(
        signal(
            text="interrupt current answer",
            priority=AttentionPriority.CRITICAL,
            target_id="target-2",
        )
    )

    assert first.current_focus_id == "target-1"
    assert second.current_focus_id == "target-2"
    assert second.should_background_existing is True


def test_focus_target_explicitly() -> None:
    runtime = AttentionRuntime()
    target = AttentionTarget(
        target_id="objective-1",
        kind=AttentionTargetKind.USER_OBJECTIVE,
        label="build real conversation runtime",
        priority=AttentionPriority.HIGH,
        weight=0.8,
    )

    decision = runtime.focus_target(target)
    focus = runtime.current_focus()

    assert decision.selected_target is not None
    assert focus is not None
    assert focus.target_id == "objective-1"


def test_background_target_clears_current_focus() -> None:
    runtime = AttentionRuntime()

    runtime.submit_signal(
        signal(priority=AttentionPriority.HIGH, target_id="target-1")
    )
    backgrounded = runtime.background_target("target-1")

    assert backgrounded is not None
    assert backgrounded.status == AttentionTargetStatus.BACKGROUNDED
    assert runtime.current_focus() is None


def test_complete_target_clears_current_focus() -> None:
    runtime = AttentionRuntime()

    runtime.submit_signal(
        signal(priority=AttentionPriority.HIGH, target_id="target-1")
    )
    completed = runtime.complete_target("target-1")

    assert completed is not None
    assert completed.status == AttentionTargetStatus.COMPLETED
    assert runtime.current_focus() is None


def test_cancel_target_clears_current_focus() -> None:
    runtime = AttentionRuntime()

    runtime.submit_signal(
        signal(priority=AttentionPriority.HIGH, target_id="target-1")
    )
    cancelled = runtime.cancel_target("target-1")

    assert cancelled is not None
    assert cancelled.status == AttentionTargetStatus.CANCELLED
    assert runtime.current_focus() is None


def test_missing_target_updates_return_none() -> None:
    runtime = AttentionRuntime()

    assert runtime.background_target("missing") is None
    assert runtime.complete_target("missing") is None
    assert runtime.cancel_target("missing") is None


def test_update_from_session_focuses_follow_up_context() -> None:
    session = ConversationSessionRuntime(session_id="session-1")
    model = session.add_assistant_turn(
        "Should I continue?",
        expects_follow_up=True,
        topic="conversation runtime",
    )
    runtime = AttentionRuntime()

    decision = runtime.update_from_session(model)

    assert model.follow_up_expectation == ConversationFollowUpExpectation.REQUIRED
    assert decision.selected_target is not None
    assert decision.selected_target.target_id == "session-1"
    assert decision.disposition in {
        AttentionDisposition.FOCUS,
        AttentionDisposition.MONITOR,
    }


def test_attention_context_block_contains_focus() -> None:
    runtime = AttentionRuntime()

    runtime.submit_signal(
        signal(
            text="adaptive endpointing",
            priority=AttentionPriority.HIGH,
            urgency=TurnUrgency.HIGH,
        )
    )
    block = runtime.as_context_block()

    assert "Attention runtime:" in block
    assert "current_focus: adaptive endpointing" in block


def test_attention_snapshot_and_reset() -> None:
    runtime = AttentionRuntime()

    runtime.submit_signal(
        signal(priority=AttentionPriority.HIGH)
    )
    snapshot = runtime.snapshot()

    assert snapshot.signal_count == 1
    assert snapshot.decision_count == 1
    assert snapshot.current_focus_id is not None

    runtime.reset()
    reset_snapshot = runtime.snapshot()

    assert reset_snapshot.signal_count == 0
    assert reset_snapshot.current_focus_id is None


def test_attention_decision_requires_reason() -> None:
    runtime = AttentionRuntime()
    decision = runtime.submit_signal(signal(priority=AttentionPriority.HIGH))
    data = decision.model_dump(mode="python")
    data["reason"] = " "

    with pytest.raises(ValidationError):
        AttentionDecision.model_validate(data)


def test_attention_target_reuses_existing_weight() -> None:
    runtime = AttentionRuntime()

    runtime.submit_signal(
        signal(
            text="jarvis runtime",
            priority=AttentionPriority.HIGH,
            target_id="same",
        )
    )
    decision = runtime.submit_signal(
        signal(
            text="jarvis runtime",
            priority=AttentionPriority.NORMAL,
            target_id="same",
        )
    )

    assert decision.selected_target is not None
    assert decision.selected_target.target_id == "same"


def test_attention_trims_targets() -> None:
    runtime = AttentionRuntime(
        config=AttentionRuntimeConfig(max_targets=3),
    )

    for index in range(6):
        runtime.submit_signal(
            signal(
                text=f"target {index}",
                priority=AttentionPriority.HIGH,
                target_id=f"target-{index}",
            )
        )

    assert runtime.snapshot().target_count <= 4


def test_attention_enum_values_are_stable() -> None:
    assert AttentionSignalKind.USER_TURN.value == "user_turn"
    assert AttentionTargetKind.CONVERSATION.value == "conversation"
    assert AttentionPriority.CRITICAL.value == "critical"
    assert AttentionDisposition.INTERRUPT.value == "interrupt"
    assert AttentionTargetStatus.BACKGROUNDED.value == "backgrounded"


def test_session_import_smoke() -> None:
    session = ConversationSessionRuntime()
    model = session.add_user_turn(
        "How does attention runtime work?",
        conversation_mode=ConversationMode.QUESTION,
    )

    assert model.turn_count == 1