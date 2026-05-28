from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.latency import (
    PartialIntentKind,
    SpeculationAggressiveness,
    SpeculationStatus,
    SpeculationTrigger,
    SpeculativeExecutionReason,
    SpeculativeExecutionRuntime,
    SpeculativeExecutionRuntimeConfig,
    SpeculativeWorkItem,
    SpeculativeWorkType,
)


def test_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        SpeculativeExecutionRuntimeConfig(name=" ").validate()


def test_config_rejects_invalid_probability_threshold() -> None:
    with pytest.raises(ValueError):
        SpeculativeExecutionRuntimeConfig(candidate_probability_threshold=2).validate()


def test_config_rejects_invalid_candidate_limits() -> None:
    with pytest.raises(ValueError):
        SpeculativeExecutionRuntimeConfig(max_candidates_normal=0).validate()

    with pytest.raises(ValueError):
        SpeculativeExecutionRuntimeConfig(
            max_candidates_normal=3,
            max_candidates_high=2,
        ).validate()


def test_speculative_work_must_be_cancellable_discardable_and_non_executing() -> None:
    with pytest.raises(ValidationError):
        SpeculativeWorkItem(
            session_id="session",
            branch_id="branch",
            candidate_id="candidate",
            work_type=SpeculativeWorkType.MEMORY_PREFETCH,
            description="bad",
            cancellable=False,
        )

    with pytest.raises(ValidationError):
        SpeculativeWorkItem(
            session_id="session",
            branch_id="branch",
            candidate_id="candidate",
            work_type=SpeculativeWorkType.MEMORY_PREFETCH,
            description="bad",
            discardable=False,
        )

    with pytest.raises(ValidationError):
        SpeculativeWorkItem(
            session_id="session",
            branch_id="branch",
            candidate_id="candidate",
            work_type=SpeculativeWorkType.ACTION_PREVALIDATION,
            description="bad",
            action_execution_allowed=True,
        )


def test_runtime_creates_session() -> None:
    runtime = SpeculativeExecutionRuntime()
    state = runtime.create_session(
        trigger=SpeculationTrigger.USER_PAUSE,
        source_text="debug this error",
    )

    assert state.status == SpeculationStatus.PROPOSED
    assert runtime.snapshot().session_count == 1


def test_runtime_starts_session() -> None:
    runtime = SpeculativeExecutionRuntime()
    state = runtime.create_session(
        trigger=SpeculationTrigger.USER_PAUSE,
        source_text="debug this error",
    )

    result = runtime.start_session(state.session_id)

    assert result.success is True
    assert result.status == SpeculationStatus.PREWARMING
    assert result.reason == SpeculativeExecutionReason.SESSION_STARTED


def test_runtime_rejects_missing_session_start() -> None:
    runtime = SpeculativeExecutionRuntime()

    result = runtime.start_session("missing")

    assert result.success is False
    assert result.reason == SpeculativeExecutionReason.SESSION_NOT_FOUND


def test_predict_and_prewarm_creates_top_candidates() -> None:
    runtime = SpeculativeExecutionRuntime()
    state = runtime.create_session(
        trigger=SpeculationTrigger.USER_PAUSE,
        source_text="debug this error",
    )

    runtime.start_session(state.session_id)
    results = runtime.predict_and_prewarm(state.session_id)

    assert len(results) == 2
    assert all(result.success for result in results)
    assert len(runtime.candidates_for(state.session_id)) == 2
    assert len(runtime.branches_for(state.session_id)) == 2


def test_candidates_below_threshold_are_rejected() -> None:
    runtime = SpeculativeExecutionRuntime(
        config=SpeculativeExecutionRuntimeConfig(candidate_probability_threshold=0.90)
    )
    state = runtime.create_session(
        trigger=SpeculationTrigger.USER_PAUSE,
        source_text="debug this error",
    )

    runtime.start_session(state.session_id)
    results = runtime.predict_and_prewarm(state.session_id)

    assert results == ()
    assert runtime.candidates_for(state.session_id) == ()


def test_branch_work_items_are_safe() -> None:
    runtime = SpeculativeExecutionRuntime()
    state = runtime.create_session(
        trigger=SpeculationTrigger.USER_PAUSE,
        source_text="run this command",
    )

    runtime.start_session(state.session_id)
    runtime.predict_and_prewarm(state.session_id)

    branches = runtime.branches_for(state.session_id)

    assert branches
    assert all(branch.status == SpeculationStatus.READY for branch in branches)

    for branch in branches:
        for item in branch.work_items:
            assert item.cancellable is True
            assert item.discardable is True
            assert item.action_execution_allowed is False


def test_command_branch_gets_action_prevalidation_without_execution() -> None:
    runtime = SpeculativeExecutionRuntime()
    state = runtime.create_session(
        trigger=SpeculationTrigger.ACTION_COMPLETED,
        source_text="action done",
    )

    runtime.start_session(state.session_id)
    runtime.predict_and_prewarm(state.session_id)

    command_branches = [
        branch
        for branch in runtime.branches_for(state.session_id)
        if branch.candidate.intent == PartialIntentKind.COMMAND
    ]

    assert command_branches
    assert any(
        item.work_type == SpeculativeWorkType.ACTION_PREVALIDATION
        for branch in command_branches
        for item in branch.work_items
    )


def test_confirm_matching_branch_discards_others() -> None:
    runtime = SpeculativeExecutionRuntime()
    state = runtime.create_session(
        trigger=SpeculationTrigger.USER_PAUSE,
        source_text="debug this error",
    )

    runtime.start_session(state.session_id)
    runtime.predict_and_prewarm(state.session_id)
    result = runtime.confirm(
        session_id=state.session_id,
        actual_intent=PartialIntentKind.DEBUGGING,
    )

    assert result.success is True
    assert result.reason == SpeculativeExecutionReason.BRANCH_CONFIRMED

    branches = runtime.branches_for(state.session_id)

    assert any(branch.status == SpeculationStatus.CONFIRMED for branch in branches)
    assert any(branch.status == SpeculationStatus.DISCARDED for branch in branches)


def test_confirm_without_match_discards_all() -> None:
    runtime = SpeculativeExecutionRuntime()
    state = runtime.create_session(
        trigger=SpeculationTrigger.USER_PAUSE,
        source_text="debug this error",
    )

    runtime.start_session(state.session_id)
    runtime.predict_and_prewarm(state.session_id)
    result = runtime.confirm(
        session_id=state.session_id,
        actual_intent=PartialIntentKind.MEMORY_RECALL,
    )

    assert result.success is False
    assert result.reason == SpeculativeExecutionReason.BRANCH_DISCARDED
    assert all(
        branch.status == SpeculationStatus.DISCARDED
        for branch in runtime.branches_for(state.session_id)
    )


def test_accuracy_low_reduces_aggressiveness() -> None:
    runtime = SpeculativeExecutionRuntime()
    state = runtime.create_session(
        trigger=SpeculationTrigger.USER_PAUSE,
        source_text="debug this error",
    )

    runtime.start_session(state.session_id)
    runtime.predict_and_prewarm(state.session_id)
    runtime.confirm(
        session_id=state.session_id,
        actual_intent=PartialIntentKind.MEMORY_RECALL,
    )

    snapshot = runtime.accuracy_snapshot()

    assert snapshot.accuracy == 0.0
    assert snapshot.aggressiveness == SpeculationAggressiveness.LOW
    assert snapshot.lookahead_depth == 1


def test_accuracy_high_increases_aggressiveness() -> None:
    runtime = SpeculativeExecutionRuntime()
    state = runtime.create_session(
        trigger=SpeculationTrigger.USER_PAUSE,
        source_text="debug this error",
    )

    runtime.start_session(state.session_id)
    runtime.predict_and_prewarm(state.session_id)
    runtime.confirm(
        session_id=state.session_id,
        actual_intent=PartialIntentKind.DEBUGGING,
    )

    snapshot = runtime.accuracy_snapshot()

    assert snapshot.accuracy > 0.70
    assert snapshot.aggressiveness == SpeculationAggressiveness.HIGH
    assert snapshot.lookahead_depth == 3


def test_complete_session_builds_report() -> None:
    runtime = SpeculativeExecutionRuntime()
    state = runtime.create_session(
        trigger=SpeculationTrigger.USER_PAUSE,
        source_text="debug this error",
    )

    runtime.start_session(state.session_id)
    runtime.predict_and_prewarm(state.session_id)
    runtime.confirm(
        session_id=state.session_id,
        actual_intent=PartialIntentKind.DEBUGGING,
    )
    report = runtime.complete_session(state.session_id)

    assert report.status == SpeculationStatus.COMPLETED
    assert report.branch_count == 2
    assert report.confirmed_count == 1
    assert report.profiler_report is not None


def test_cancel_session_cancels_branches() -> None:
    runtime = SpeculativeExecutionRuntime()
    state = runtime.create_session(
        trigger=SpeculationTrigger.USER_PAUSE,
        source_text="debug this error",
    )

    runtime.start_session(state.session_id)
    runtime.predict_and_prewarm(state.session_id)
    result = runtime.cancel_session(state.session_id)

    assert result.success is True
    assert result.status == SpeculationStatus.CANCELLED
    assert all(
        branch.status == SpeculationStatus.CANCELLED
        for branch in runtime.branches_for(state.session_id)
    )


def test_fail_session() -> None:
    runtime = SpeculativeExecutionRuntime()
    state = runtime.create_session(
        trigger=SpeculationTrigger.USER_PAUSE,
        source_text="debug this error",
    )

    runtime.start_session(state.session_id)
    result = runtime.fail_session(state.session_id, error="failed")

    assert result.success is True
    assert result.status == SpeculationStatus.FAILED


def test_snapshot_tracks_counts() -> None:
    runtime = SpeculativeExecutionRuntime()
    state = runtime.create_session(
        trigger=SpeculationTrigger.USER_PAUSE,
        source_text="debug this error",
    )

    runtime.start_session(state.session_id)
    runtime.predict_and_prewarm(state.session_id)
    runtime.confirm(
        session_id=state.session_id,
        actual_intent=PartialIntentKind.DEBUGGING,
    )
    runtime.complete_session(state.session_id)

    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.completed_count == 1
    assert snapshot.candidate_count == 2
    assert snapshot.branch_count == 2
    assert snapshot.report_count == 1


def test_reports_are_queryable() -> None:
    runtime = SpeculativeExecutionRuntime()
    state = runtime.create_session(
        trigger=SpeculationTrigger.USER_PAUSE,
        source_text="debug this error",
    )

    runtime.start_session(state.session_id)
    runtime.predict_and_prewarm(state.session_id)
    report = runtime.complete_session(state.session_id)

    assert runtime.latest_report() == report
    assert runtime.reports() == (report,)


def test_reset_clears_state() -> None:
    runtime = SpeculativeExecutionRuntime()
    state = runtime.create_session(
        trigger=SpeculationTrigger.USER_PAUSE,
        source_text="debug this error",
    )

    runtime.start_session(state.session_id)
    runtime.predict_and_prewarm(state.session_id)
    runtime.reset()

    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.branch_count == 0
    assert snapshot.accuracy == 0.0
    assert snapshot.aggressiveness == SpeculationAggressiveness.NORMAL
    assert snapshot.last_reason == SpeculativeExecutionReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert SpeculationTrigger.USER_PAUSE.value == "user_pause"
    assert SpeculationStatus.READY.value == "ready"
    assert SpeculationStatus.COMPLETED.value == "completed"
    assert SpeculativeWorkType.MEMORY_PREFETCH.value == "memory_prefetch"
    assert SpeculationAggressiveness.HIGH.value == "high"