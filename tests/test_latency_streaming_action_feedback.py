from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.latency import (
    ActionFeedbackChunk,
    ActionFeedbackReason,
    ActionFeedbackStatus,
    ActionFeedbackType,
    StreamingActionFeedbackConfig,
    StreamingActionFeedbackRuntime,
)


def test_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        StreamingActionFeedbackConfig(name=" ").validate()


def test_config_rejects_invalid_budgets() -> None:
    with pytest.raises(ValueError):
        StreamingActionFeedbackConfig(start_ack_budget_ms=0).validate()

    with pytest.raises(ValueError):
        StreamingActionFeedbackConfig(progress_interval_ms=0).validate()


def test_chunk_requires_message() -> None:
    with pytest.raises(ValidationError):
        ActionFeedbackChunk(
            session_id="session",
            action_id="action",
            feedback_type=ActionFeedbackType.PROGRESS,
            spoken_text=" ",
        )


def test_chunk_progress_pair_required() -> None:
    with pytest.raises(ValidationError):
        ActionFeedbackChunk(
            session_id="session",
            action_id="action",
            feedback_type=ActionFeedbackType.PROGRESS,
            spoken_text="3 tests passing",
            progress_current=3,
        )


def test_chunk_progress_current_cannot_exceed_total() -> None:
    with pytest.raises(ValidationError):
        ActionFeedbackChunk(
            session_id="session",
            action_id="action",
            feedback_type=ActionFeedbackType.PROGRESS,
            spoken_text="bad progress",
            progress_current=4,
            progress_total=3,
        )


def test_runtime_creates_session() -> None:
    runtime = StreamingActionFeedbackRuntime()
    state = runtime.create_session(
        action_id="run-tests",
        action_name="Running tests",
    )

    assert state.status == ActionFeedbackStatus.CREATED
    assert runtime.snapshot().session_count == 1


def test_start_stream_emits_immediate_ack() -> None:
    runtime = StreamingActionFeedbackRuntime()
    state = runtime.create_session(
        action_id="run-tests",
        action_name="Running tests",
    )

    result = runtime.start_stream(state.session_id)

    assert result.success is True
    assert result.chunk is not None
    assert result.chunk.feedback_type == ActionFeedbackType.STARTED
    assert result.state is not None
    assert result.state.first_feedback_latency_ms() is not None


def test_start_stream_custom_message() -> None:
    runtime = StreamingActionFeedbackRuntime()
    state = runtime.create_session(
        action_id="run-tests",
        action_name="Running tests",
    )

    result = runtime.start_stream(
        state.session_id,
        message="Running tests now...",
    )

    assert result.chunk is not None
    assert result.chunk.spoken_text == "Running tests now..."


def test_progress_not_due_without_force() -> None:
    runtime = StreamingActionFeedbackRuntime()
    state = runtime.create_session(
        action_id="run-tests",
        action_name="Running tests",
    )

    runtime.start_stream(state.session_id)
    result = runtime.emit_progress(
        session_id=state.session_id,
        message="3 tests passing, checking more...",
    )

    assert result.success is False
    assert result.reason == ActionFeedbackReason.PROGRESS_NOT_DUE


def test_progress_emits_when_forced() -> None:
    runtime = StreamingActionFeedbackRuntime()
    state = runtime.create_session(
        action_id="run-tests",
        action_name="Running tests",
    )

    runtime.start_stream(state.session_id)
    result = runtime.emit_progress(
        session_id=state.session_id,
        message="3 tests passing, checking more...",
        progress_current=3,
        progress_total=10,
        force=True,
    )

    assert result.success is True
    assert result.chunk is not None
    assert result.chunk.feedback_type == ActionFeedbackType.PROGRESS
    assert result.chunk.progress_current == 3


def test_milestone_emits_immediately() -> None:
    runtime = StreamingActionFeedbackRuntime()
    state = runtime.create_session(
        action_id="run-tests",
        action_name="Running tests",
    )

    runtime.start_stream(state.session_id)
    result = runtime.emit_milestone(
        session_id=state.session_id,
        message="All unit tests passed, running integration...",
    )

    assert result.success is True
    assert result.chunk is not None
    assert result.chunk.feedback_type == ActionFeedbackType.MILESTONE


def test_completed_emits_summary_and_marks_completed() -> None:
    runtime = StreamingActionFeedbackRuntime()
    state = runtime.create_session(
        action_id="run-tests",
        action_name="Running tests",
    )

    runtime.start_stream(state.session_id)
    result = runtime.emit_completed(
        session_id=state.session_id,
        message="Done. 47 tests passed, 2 failed.",
    )

    assert result.success is True

    updated = runtime.state_for(state.session_id)

    assert updated is not None
    assert updated.status == ActionFeedbackStatus.COMPLETED


def test_error_emits_summary_and_marks_error() -> None:
    runtime = StreamingActionFeedbackRuntime()
    state = runtime.create_session(
        action_id="run-tests",
        action_name="Running tests",
    )

    runtime.start_stream(state.session_id)
    result = runtime.emit_error(
        session_id=state.session_id,
        message="Test runner hit an error — here's what happened...",
    )

    assert result.success is True

    updated = runtime.state_for(state.session_id)

    assert updated is not None
    assert updated.status == ActionFeedbackStatus.ERROR
    assert updated.error_count == 1


def test_complete_session_builds_report() -> None:
    runtime = StreamingActionFeedbackRuntime()
    state = runtime.create_session(
        action_id="run-tests",
        action_name="Running tests",
    )

    runtime.start_stream(state.session_id)
    runtime.emit_progress(
        session_id=state.session_id,
        message="3 tests passing, checking more...",
        force=True,
    )
    runtime.emit_milestone(
        session_id=state.session_id,
        message="All unit tests passed, running integration...",
    )
    runtime.emit_completed(
        session_id=state.session_id,
        message="Done. 47 tests passed, 2 failed.",
    )
    report = runtime.complete_session(state.session_id)

    assert report.status == ActionFeedbackStatus.COMPLETED
    assert report.chunk_count == 4
    assert report.progress_count == 1
    assert report.milestone_count == 1
    assert report.profiler_report is not None


def test_complete_rejects_missing_session() -> None:
    runtime = StreamingActionFeedbackRuntime()

    with pytest.raises(ValueError):
        runtime.complete_session("missing")


def test_complete_rejects_active_session() -> None:
    runtime = StreamingActionFeedbackRuntime()
    state = runtime.create_session(
        action_id="run-tests",
        action_name="Running tests",
    )

    runtime.start_stream(state.session_id)

    with pytest.raises(ValueError):
        runtime.complete_session(state.session_id)


def test_cancel_session() -> None:
    runtime = StreamingActionFeedbackRuntime()
    state = runtime.create_session(
        action_id="run-tests",
        action_name="Running tests",
    )

    runtime.start_stream(state.session_id)
    result = runtime.cancel_session(state.session_id)

    assert result.success is True
    assert result.status == ActionFeedbackStatus.CANCELLED


def test_fail_session() -> None:
    runtime = StreamingActionFeedbackRuntime()
    state = runtime.create_session(
        action_id="run-tests",
        action_name="Running tests",
    )

    result = runtime.fail_session(state.session_id, error="feedback failed")

    assert result.success is True
    assert result.status == ActionFeedbackStatus.FAILED


def test_reports_are_queryable() -> None:
    runtime = StreamingActionFeedbackRuntime()
    state = runtime.create_session(
        action_id="run-tests",
        action_name="Running tests",
    )

    runtime.start_stream(state.session_id)
    runtime.emit_completed(
        session_id=state.session_id,
        message="Done. 47 tests passed.",
    )
    report = runtime.complete_session(state.session_id)

    assert runtime.latest_report() == report
    assert runtime.reports() == (report,)


def test_snapshot_tracks_counts() -> None:
    runtime = StreamingActionFeedbackRuntime()
    state = runtime.create_session(
        action_id="run-tests",
        action_name="Running tests",
    )

    runtime.start_stream(state.session_id)
    runtime.emit_completed(
        session_id=state.session_id,
        message="Done. 47 tests passed.",
    )
    runtime.complete_session(state.session_id)

    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.completed_count == 1
    assert snapshot.chunk_count == 2
    assert snapshot.report_count == 1


def test_reset_clears_runtime_state() -> None:
    runtime = StreamingActionFeedbackRuntime()
    state = runtime.create_session(
        action_id="run-tests",
        action_name="Running tests",
    )

    runtime.start_stream(state.session_id)
    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.chunk_count == 0
    assert snapshot.last_reason == ActionFeedbackReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert ActionFeedbackType.STARTED.value == "started"
    assert ActionFeedbackStatus.STREAMING.value == "streaming"
    assert ActionFeedbackReason.PROGRESS_EMITTED.value == "progress_emitted"