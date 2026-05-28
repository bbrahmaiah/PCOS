from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.latency import (
    ContextFragment,
    ContextFragmentKind,
    ContextSnapshot,
    InterruptionContextDelta,
    InterruptionRecoveryReason,
    InterruptionRecoveryRuntime,
    InterruptionRecoveryRuntimeConfig,
    InterruptionRecoverySessionState,
    InterruptionRecoveryStatus,
)


def test_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        InterruptionRecoveryRuntimeConfig(name=" ").validate()


def test_config_rejects_invalid_budgets() -> None:
    with pytest.raises(ValueError):
        InterruptionRecoveryRuntimeConfig(snapshot_interval_ms=0).validate()

    with pytest.raises(ValueError):
        InterruptionRecoveryRuntimeConfig(first_new_word_budget_ms=0).validate()


def test_delta_requires_context() -> None:
    with pytest.raises(ValidationError):
        InterruptionContextDelta(
            interrupted_at_text=" ",
            user_new_utterance="new words",
            partial_assistant_utterance="partial",
        )


def test_runtime_creates_session() -> None:
    runtime = InterruptionRecoveryRuntime()
    state = runtime.create_session(turn_id="turn")

    assert state.status == InterruptionRecoveryStatus.CREATED
    assert runtime.snapshot().session_count == 1


def test_start_tracking() -> None:
    runtime = InterruptionRecoveryRuntime()
    state = runtime.create_session(turn_id="turn")

    result = runtime.start_tracking(state.session_id)

    assert result.success is True
    assert result.status == InterruptionRecoveryStatus.TRACKING


def test_capture_generation_snapshot() -> None:
    runtime = InterruptionRecoveryRuntime()
    state = runtime.create_session(turn_id="turn")

    runtime.start_tracking(state.session_id)
    result = runtime.capture_generation_snapshot(
        session_id=state.session_id,
        context_snapshot=_context_snapshot(),
        assistant_partial_text="I was saying",
        generation_sequence=1,
        force=True,
    )

    assert result.success is True
    assert result.snapshot is not None
    assert runtime.snapshot().snapshot_count == 1


def test_snapshot_interval_reuses_latest_without_force() -> None:
    runtime = InterruptionRecoveryRuntime()
    state = runtime.create_session(turn_id="turn")

    runtime.start_tracking(state.session_id)
    first = runtime.capture_generation_snapshot(
        session_id=state.session_id,
        context_snapshot=_context_snapshot(),
        assistant_partial_text="first",
        generation_sequence=1,
        force=True,
    )
    second = runtime.capture_generation_snapshot(
        session_id=state.session_id,
        context_snapshot=_context_snapshot(),
        assistant_partial_text="second",
        generation_sequence=2,
    )

    assert first.snapshot is not None
    assert second.snapshot == first.snapshot
    assert len(runtime.snapshots_for(state.session_id)) == 1


def test_detect_interrupt() -> None:
    runtime = InterruptionRecoveryRuntime()
    state = runtime.create_session(turn_id="turn")

    runtime.start_tracking(state.session_id)
    result = runtime.detect_interrupt(
        session_id=state.session_id,
        partial_assistant_utterance="I was saying",
        user_new_utterance="wait",
    )

    assert result.success is True
    assert result.status == InterruptionRecoveryStatus.INTERRUPTED


def test_stop_tts_and_cancel_llm_after_interrupt() -> None:
    runtime = InterruptionRecoveryRuntime()
    state = runtime.create_session(turn_id="turn")

    runtime.start_tracking(state.session_id)
    runtime.detect_interrupt(
        session_id=state.session_id,
        partial_assistant_utterance="I was saying",
        user_new_utterance="wait",
    )

    tts = runtime.stop_tts_playback(state.session_id)
    llm = runtime.cancel_llm_stream(state.session_id)

    assert tts.success is True
    assert llm.success is True
    assert tts.state is not None
    assert tts.state.tts_stop_latency_ms() is not None
    assert llm.state is not None
    assert llm.state.llm_cancel_latency_ms() is not None


def test_capture_interruption_context() -> None:
    runtime = InterruptionRecoveryRuntime()
    state = runtime.create_session(turn_id="turn")

    runtime.start_tracking(state.session_id)
    runtime.detect_interrupt(
        session_id=state.session_id,
        partial_assistant_utterance="I was saying",
        user_new_utterance="wait",
    )
    result = runtime.capture_interruption_context(
        session_id=state.session_id,
        partial_assistant_utterance="I was saying",
        user_new_utterance="wait",
    )

    assert result.success is True
    assert result.delta is not None
    assert result.status == InterruptionRecoveryStatus.RECONSTRUCTING


def test_reconstruct_requires_snapshot() -> None:
    runtime = InterruptionRecoveryRuntime()
    state = runtime.create_session(turn_id="turn")

    runtime.start_tracking(state.session_id)
    runtime.detect_interrupt(
        session_id=state.session_id,
        partial_assistant_utterance="I was saying",
        user_new_utterance="wait",
    )
    runtime.capture_interruption_context(
        session_id=state.session_id,
        partial_assistant_utterance="I was saying",
        user_new_utterance="wait",
    )

    result = runtime.reconstruct_context(state.session_id)

    assert result.success is False
    assert result.reason == InterruptionRecoveryReason.NO_SNAPSHOT_AVAILABLE


def test_reconstruct_context_from_snapshot_plus_delta() -> None:
    runtime = InterruptionRecoveryRuntime()
    state = runtime.create_session(turn_id="turn")

    runtime.start_tracking(state.session_id)
    runtime.capture_generation_snapshot(
        session_id=state.session_id,
        context_snapshot=_context_snapshot(),
        assistant_partial_text="I was saying",
        generation_sequence=1,
        force=True,
    )
    runtime.detect_interrupt(
        session_id=state.session_id,
        partial_assistant_utterance="I was saying",
        user_new_utterance="wait, change that",
    )
    runtime.stop_tts_playback(state.session_id)
    runtime.cancel_llm_stream(state.session_id)
    runtime.capture_interruption_context(
        session_id=state.session_id,
        partial_assistant_utterance="I was saying",
        user_new_utterance="wait, change that",
    )
    result = runtime.reconstruct_context(state.session_id)

    assert result.success is True
    assert result.status == InterruptionRecoveryStatus.READY
    assert result.reconstructed is not None

    texts = [
        fragment.text
        for fragment in result.reconstructed.context_snapshot.fragments
    ]

    assert any("User interrupted at:" in text for text in texts)
    assert any("User now says:" in text for text in texts)


def test_mark_first_new_word_ready() -> None:
    runtime = InterruptionRecoveryRuntime()
    state = _ready_runtime(runtime)

    result = runtime.mark_first_new_word_ready(state.session_id)

    assert result.success is True
    assert result.state is not None
    assert result.state.first_new_word_latency_ms() is not None


def test_complete_session_builds_report() -> None:
    runtime = InterruptionRecoveryRuntime()
    state = _ready_runtime(runtime)

    runtime.mark_first_new_word_ready(state.session_id)
    report = runtime.complete_session(state.session_id)

    assert report.status == InterruptionRecoveryStatus.COMPLETED
    assert report.snapshot_count == 1
    assert report.delta_count == 1
    assert report.reconstructed_context is not None
    assert report.first_new_word_latency_ms is not None
    assert report.profiler_report is not None


def test_complete_rejects_missing_session() -> None:
    runtime = InterruptionRecoveryRuntime()

    with pytest.raises(ValueError):
        runtime.complete_session("missing")


def test_cancel_session() -> None:
    runtime = InterruptionRecoveryRuntime()
    state = runtime.create_session(turn_id="turn")

    result = runtime.cancel_session(state.session_id)

    assert result.success is True
    assert result.status == InterruptionRecoveryStatus.CANCELLED


def test_fail_session() -> None:
    runtime = InterruptionRecoveryRuntime()
    state = runtime.create_session(turn_id="turn")

    result = runtime.fail_session(state.session_id, error="failed")

    assert result.success is True
    assert result.status == InterruptionRecoveryStatus.FAILED


def test_reports_are_queryable() -> None:
    runtime = InterruptionRecoveryRuntime()
    state = _ready_runtime(runtime)

    runtime.mark_first_new_word_ready(state.session_id)
    report = runtime.complete_session(state.session_id)

    assert runtime.latest_report() == report
    assert runtime.reports() == (report,)


def test_snapshot_tracks_counts() -> None:
    runtime = InterruptionRecoveryRuntime()
    state = _ready_runtime(runtime)

    runtime.mark_first_new_word_ready(state.session_id)
    runtime.complete_session(state.session_id)
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.completed_count == 1
    assert snapshot.snapshot_count == 1
    assert snapshot.report_count == 1


def test_reset_clears_state() -> None:
    runtime = InterruptionRecoveryRuntime()
    state = runtime.create_session(turn_id="turn")

    runtime.start_tracking(state.session_id)
    runtime.capture_generation_snapshot(
        session_id=state.session_id,
        context_snapshot=_context_snapshot(),
        assistant_partial_text="partial",
        generation_sequence=1,
        force=True,
    )
    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.snapshot_count == 0
    assert snapshot.last_reason == InterruptionRecoveryReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert InterruptionRecoveryStatus.TRACKING.value == "tracking"
    assert InterruptionRecoveryReason.INTERRUPT_DETECTED.value == (
        "interrupt_detected"
    )
    assert InterruptionRecoveryReason.FIRST_NEW_WORD_WITHIN_BUDGET.value == (
        "first_new_word_within_budget"
    )


def _context_snapshot() -> ContextSnapshot:
    fragment = ContextFragment(
        kind=ContextFragmentKind.RECENT_TURN,
        text="Existing context",
        priority=80,
        token_estimate=2,
    )

    return ContextSnapshot(
        turn_id="turn",
        fragments=(fragment,),
        token_estimate=2,
        context_confidence=0.9,
    )


def _ready_runtime(
    runtime: InterruptionRecoveryRuntime,
) -> InterruptionRecoverySessionState:
    state = runtime.create_session(turn_id="turn")
    runtime.start_tracking(state.session_id)
    runtime.capture_generation_snapshot(
        session_id=state.session_id,
        context_snapshot=_context_snapshot(),
        assistant_partial_text="I was saying",
        generation_sequence=1,
        force=True,
    )
    runtime.detect_interrupt(
        session_id=state.session_id,
        partial_assistant_utterance="I was saying",
        user_new_utterance="wait, change that",
    )
    runtime.stop_tts_playback(state.session_id)
    runtime.cancel_llm_stream(state.session_id)
    runtime.capture_interruption_context(
        session_id=state.session_id,
        partial_assistant_utterance="I was saying",
        user_new_utterance="wait, change that",
    )
    runtime.reconstruct_context(state.session_id)

    ready = runtime.state_for(state.session_id)
    assert ready is not None

    return ready