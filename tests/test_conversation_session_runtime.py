from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.conversation import (
    ConversationContinuityStatus,
    ConversationFollowUpExpectation,
    ConversationMode,
    ConversationSessionRuntime,
    ConversationSessionRuntimeConfig,
    ConversationSessionSnapshotModel,
    ConversationSessionTurn,
    ConversationState,
    ConversationTopicShift,
    ConversationTurnRole,
)


def test_session_runtime_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        ConversationSessionRuntimeConfig(name=" ").validate()

    with pytest.raises(ValueError):
        ConversationSessionRuntimeConfig(max_turns=0).validate()

    with pytest.raises(ValueError):
        ConversationSessionRuntimeConfig(max_summary_chars=0).validate()

    with pytest.raises(ValueError):
        ConversationSessionRuntimeConfig(topic_shift_threshold=1.5).validate()


def test_session_turn_requires_text() -> None:
    with pytest.raises(ValidationError):
        ConversationSessionTurn(
            role=ConversationTurnRole.USER,
            text=" ",
        )


def test_session_turn_cleans_text_topic_and_objective() -> None:
    turn = ConversationSessionTurn(
        role=ConversationTurnRole.USER,
        text="  build    jarvis  ",
        topic="  jarvis   runtime ",
        objective="  build ",
    )

    assert turn.text == "build jarvis"
    assert turn.topic == "jarvis runtime"
    assert turn.objective == "build"


def test_session_starts_empty() -> None:
    runtime = ConversationSessionRuntime(session_id="session-1")
    snapshot = runtime.snapshot()

    assert snapshot.session_id == "session-1"
    assert snapshot.status == ConversationContinuityStatus.EMPTY
    assert snapshot.turn_count == 0


def test_add_user_turn_creates_active_continuity() -> None:
    runtime = ConversationSessionRuntime(session_id="session-1")

    model = runtime.add_user_turn(
        "Jarvis build adaptive conversation runtime",
        conversation_mode=ConversationMode.DISCUSSION,
        state=ConversationState.LISTENING,
    )
    snapshot = runtime.snapshot()

    assert model.status == ConversationContinuityStatus.ACTIVE
    assert model.turn_count == 1
    assert model.last_user_turn is not None
    assert snapshot.user_turn_count == 1
    assert snapshot.active_topic is not None


def test_add_assistant_turn_tracks_last_assistant_response() -> None:
    runtime = ConversationSessionRuntime()

    runtime.add_user_turn("Jarvis explain memory gateway")
    model = runtime.add_assistant_turn("Memory gateway is the safe boundary.")

    assert model.last_assistant_turn is not None
    assert model.last_assistant_turn.text == "Memory gateway is the safe boundary."


def test_assistant_question_sets_follow_up_likely() -> None:
    runtime = ConversationSessionRuntime()

    model = runtime.add_assistant_turn("Should I continue?")

    assert model.status == ConversationContinuityStatus.WAITING_FOR_FOLLOW_UP
    assert model.follow_up_expectation == ConversationFollowUpExpectation.LIKELY


def test_assistant_required_follow_up_sets_required() -> None:
    runtime = ConversationSessionRuntime()

    model = runtime.add_assistant_turn(
        "Confirm before I continue.",
        expects_follow_up=True,
    )

    assert model.status == ConversationContinuityStatus.WAITING_FOR_FOLLOW_UP
    assert model.follow_up_expectation == ConversationFollowUpExpectation.REQUIRED


def test_user_question_sets_follow_up_likely() -> None:
    runtime = ConversationSessionRuntime()

    model = runtime.add_user_turn("How does endpointing work?")

    assert model.follow_up_expectation == ConversationFollowUpExpectation.LIKELY


def test_explicit_topic_and_objective_are_preserved() -> None:
    runtime = ConversationSessionRuntime()

    model = runtime.add_user_turn(
        "Let's work on the runtime.",
        topic="adaptive conversation",
        objective="build state runtime",
    )

    assert model.active_topic == "adaptive conversation"
    assert model.current_objective == "build state runtime"


def test_topic_shift_detects_high_shift() -> None:
    runtime = ConversationSessionRuntime()

    runtime.add_user_turn("Build adaptive conversation runtime")
    model = runtime.add_user_turn("Explain cricket match highlights")

    assert model.topic_shift == ConversationTopicShift.HIGH


def test_topic_shift_detects_low_shift_for_related_turn() -> None:
    runtime = ConversationSessionRuntime()

    runtime.add_user_turn("Build adaptive conversation runtime")
    model = runtime.add_user_turn("Adaptive conversation runtime needs streaming")

    assert model.topic_shift in {
        ConversationTopicShift.LOW,
        ConversationTopicShift.MEDIUM,
    }


def test_compacts_old_turns_when_max_turns_exceeded() -> None:
    runtime = ConversationSessionRuntime(
        config=ConversationSessionRuntimeConfig(max_turns=3),
    )

    runtime.add_user_turn("turn one about memory")
    runtime.add_assistant_turn("turn two response")
    runtime.add_user_turn("turn three about cognition")
    model = runtime.add_assistant_turn("turn four response")

    snapshot = runtime.snapshot()

    assert model.turn_count == 3
    assert model.summary is not None
    assert snapshot.compaction_count == 1


def test_context_block_contains_session_continuity() -> None:
    runtime = ConversationSessionRuntime(session_id="session-1")

    model = runtime.add_user_turn(
        "Jarvis explain the memory gateway",
        topic="memory gateway",
        objective="understand",
    )
    block = model.as_context_block()

    assert "Conversation session continuity:" in block
    assert "active_topic: memory gateway" in block
    assert "user: Jarvis explain the memory gateway" in block


def test_temporary_context_can_be_set() -> None:
    runtime = ConversationSessionRuntime()

    model = runtime.set_temporary_context("active_file", "main.py")

    assert model.temporary_context["active_file"] == "main.py"


def test_temporary_context_rejects_empty_key() -> None:
    runtime = ConversationSessionRuntime()

    with pytest.raises(ValueError):
        runtime.set_temporary_context(" ", "value")


def test_mark_interrupted_preserves_turns() -> None:
    runtime = ConversationSessionRuntime()

    runtime.add_user_turn("Jarvis explain endpointing")
    model = runtime.mark_interrupted(reason="user barge-in")

    assert model.status == ConversationContinuityStatus.INTERRUPTED
    assert model.turn_count == 1
    assert model.temporary_context["last_interrupt_reason"] == "user barge-in"


def test_pause_and_close() -> None:
    runtime = ConversationSessionRuntime()

    paused = runtime.pause(reason="thinking")
    closed = runtime.close(reason="done")

    assert paused.status == ConversationContinuityStatus.PAUSED
    assert closed.status == ConversationContinuityStatus.CLOSED
    assert closed.temporary_context["close_reason"] == "done"


def test_snapshot_model_requires_session_id() -> None:
    with pytest.raises(ValidationError):
        ConversationSessionSnapshotModel(
            session_id=" ",
            status=ConversationContinuityStatus.ACTIVE,
            follow_up_expectation=ConversationFollowUpExpectation.NONE,
        )


def test_runtime_snapshot_and_reset() -> None:
    runtime = ConversationSessionRuntime(session_id="session-1")

    runtime.add_user_turn("Jarvis build the session runtime")
    snapshot = runtime.snapshot()

    assert snapshot.turn_count == 1
    assert snapshot.update_count == 1
    assert snapshot.last_turn_id is not None

    runtime.reset()
    reset_snapshot = runtime.snapshot()

    assert reset_snapshot.turn_count == 0
    assert reset_snapshot.status == ConversationContinuityStatus.EMPTY
    assert reset_snapshot.last_turn_id is None


def test_session_enum_values_are_stable() -> None:
    assert ConversationTurnRole.USER.value == "user"
    assert ConversationTurnRole.ASSISTANT.value == "assistant"
    assert ConversationContinuityStatus.ACTIVE.value == "active"
    assert ConversationContinuityStatus.INTERRUPTED.value == "interrupted"
    assert ConversationFollowUpExpectation.REQUIRED.value == "required"
    assert ConversationTopicShift.HIGH.value == "high"