from __future__ import annotations

import pytest

from jarvis.latency import (
    ParallelPipelineReason,
    ParallelPipelineRuntime,
    ParallelPipelineRuntimeConfig,
    ParallelPipelineStageKind,
    ParallelPipelineStatus,
    PipelineBranchStatus,
)


def test_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        ParallelPipelineRuntimeConfig(name=" ").validate()


def test_config_rejects_invalid_confidence_threshold() -> None:
    with pytest.raises(ValueError):
        ParallelPipelineRuntimeConfig(partial_confidence_threshold=2).validate()


def test_config_rejects_invalid_stage_latency() -> None:
    with pytest.raises(ValueError):
        ParallelPipelineRuntimeConfig(simulated_stt_ms=0).validate()


def test_runtime_creates_session() -> None:
    runtime = ParallelPipelineRuntime()
    state = runtime.create_session(turn_id="turn")

    assert state.status == ParallelPipelineStatus.CREATED
    assert runtime.snapshot().session_count == 1


def test_runtime_starts_session_and_stt_stage() -> None:
    runtime = ParallelPipelineRuntime()
    state = runtime.create_session(turn_id="turn")

    result = runtime.start_session(state.session_id)
    timings = runtime.stage_timings_for(state.session_id)

    assert result.success is True
    assert result.status == ParallelPipelineStatus.RUNNING
    assert any(
        timing.stage == ParallelPipelineStageKind.STT_STREAMING
        and timing.status == PipelineBranchStatus.RUNNING
        for timing in timings
    )


def test_low_confidence_partial_does_not_start_parallel_branches() -> None:
    runtime = ParallelPipelineRuntime()
    state = runtime.create_session(turn_id="turn")

    runtime.start_session(state.session_id)
    result = runtime.accept_partial_transcript(
        session_id=state.session_id,
        partial_text="debug this error",
        confidence=0.40,
    )

    assert result.success is False
    assert result.reason == ParallelPipelineReason.STT_PARTIAL_CONFIDENCE_LOW
    assert result.state is not None
    assert result.state.status == ParallelPipelineStatus.RUNNING


def test_high_confidence_partial_starts_parallel_branches() -> None:
    runtime = ParallelPipelineRuntime()
    state = runtime.create_session(turn_id="turn")

    runtime.start_session(state.session_id)
    result = runtime.accept_partial_transcript(
        session_id=state.session_id,
        partial_text="debug this error",
        confidence=0.80,
    )

    assert result.success is True
    assert result.status == ParallelPipelineStatus.READY_FOR_LLM
    assert result.state is not None
    assert result.state.ready_for_llm is True


def test_parallel_branches_all_reach_ready() -> None:
    runtime = ParallelPipelineRuntime()
    state = runtime.create_session(turn_id="turn")

    runtime.start_session(state.session_id)
    runtime.accept_partial_transcript(
        session_id=state.session_id,
        partial_text="debug this error",
        confidence=0.80,
    )

    timings = runtime.stage_timings_for(state.session_id)
    relevant = {
        ParallelPipelineStageKind.STT_STREAMING,
        ParallelPipelineStageKind.INTENT_CLASSIFICATION,
        ParallelPipelineStageKind.MEMORY_RETRIEVAL,
        ParallelPipelineStageKind.CONTEXT_BUILD,
    }

    assert all(
        timing.status == PipelineBranchStatus.READY
        for timing in timings
        if timing.stage in relevant
    )


def test_llm_first_token_computes_savings() -> None:
    runtime = ParallelPipelineRuntime()
    state = runtime.create_session(turn_id="turn")

    runtime.start_session(state.session_id)
    runtime.accept_partial_transcript(
        session_id=state.session_id,
        partial_text="debug this error",
        confidence=0.80,
    )
    result = runtime.complete_llm_first_token(state.session_id)

    assert result.success is True
    assert result.state is not None
    assert result.state.serial_latency_ms == 680.0
    assert result.state.parallel_latency_ms == 450.0
    assert result.state.savings_ms == 230.0
    assert round(result.state.savings_ratio, 2) == 0.34


def test_complete_session_builds_report() -> None:
    runtime = ParallelPipelineRuntime()
    state = runtime.create_session(turn_id="turn")

    runtime.start_session(state.session_id)
    runtime.accept_partial_transcript(
        session_id=state.session_id,
        partial_text="debug this error",
        confidence=0.80,
    )
    runtime.complete_llm_first_token(state.session_id)
    report = runtime.complete_session(state.session_id)

    assert report.status == ParallelPipelineStatus.COMPLETED
    assert report.ready_for_llm is True
    assert report.savings_ms == 230.0
    assert report.profiler_report is not None


def test_complete_rejects_missing_session() -> None:
    runtime = ParallelPipelineRuntime()

    with pytest.raises(ValueError):
        runtime.complete_session("missing")


def test_complete_rejects_before_ready() -> None:
    runtime = ParallelPipelineRuntime()
    state = runtime.create_session(turn_id="turn")

    runtime.start_session(state.session_id)

    with pytest.raises(ValueError):
        runtime.complete_session(state.session_id)


def test_cancel_session_cancels_running_stage() -> None:
    runtime = ParallelPipelineRuntime()
    state = runtime.create_session(turn_id="turn")

    runtime.start_session(state.session_id)
    result = runtime.cancel_session(state.session_id)

    assert result.success is True
    assert result.status == ParallelPipelineStatus.CANCELLED
    assert any(
        timing.status == PipelineBranchStatus.CANCELLED
        for timing in runtime.stage_timings_for(state.session_id)
    )


def test_fail_session() -> None:
    runtime = ParallelPipelineRuntime()
    state = runtime.create_session(turn_id="turn")

    result = runtime.fail_session(state.session_id, error="failed")

    assert result.success is True
    assert result.status == ParallelPipelineStatus.FAILED


def test_reports_are_queryable() -> None:
    runtime = ParallelPipelineRuntime()
    state = runtime.create_session(turn_id="turn")

    runtime.start_session(state.session_id)
    runtime.accept_partial_transcript(
        session_id=state.session_id,
        partial_text="debug this error",
        confidence=0.80,
    )
    runtime.complete_llm_first_token(state.session_id)
    report = runtime.complete_session(state.session_id)

    assert runtime.latest_report() == report
    assert runtime.reports() == (report,)


def test_snapshot_tracks_average_savings() -> None:
    runtime = ParallelPipelineRuntime()
    state = runtime.create_session(turn_id="turn")

    runtime.start_session(state.session_id)
    runtime.accept_partial_transcript(
        session_id=state.session_id,
        partial_text="debug this error",
        confidence=0.80,
    )
    runtime.complete_llm_first_token(state.session_id)
    runtime.complete_session(state.session_id)

    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.completed_count == 1
    assert snapshot.report_count == 1
    assert snapshot.average_savings_ms == 230.0


def test_reset_clears_runtime_state() -> None:
    runtime = ParallelPipelineRuntime()
    state = runtime.create_session(turn_id="turn")

    runtime.start_session(state.session_id)
    runtime.reset()

    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.report_count == 0
    assert snapshot.last_reason == ParallelPipelineReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert ParallelPipelineStatus.RUNNING.value == "running"
    assert ParallelPipelineStageKind.MEMORY_RETRIEVAL.value == "memory_retrieval"
    assert PipelineBranchStatus.READY.value == "ready"
    assert ParallelPipelineReason.LATENCY_SAVINGS_COMPUTED.value == (
        "latency_savings_computed"
    )