from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.latency import (
    LatencyBudgetAllocationReason,
    LatencyBudgetAllocationStatus,
    LatencyBudgetAllocator,
    LatencyBudgetAllocatorConfig,
    LatencyBudgetPlan,
    LatencyBudgetSlice,
    LatencyBudgetSliceKind,
    PipelineStage,
    VoiceBaselineProfilerConfig,
    VoiceMicroLatencyProfile,
    VoicePipelineBaselineProfiler,
    build_synthetic_voice_sample,
    default_first_word_latency_budget_plan,
)


def test_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        LatencyBudgetAllocatorConfig(name=" ").validate()


def test_config_rejects_invalid_warning_ratio() -> None:
    with pytest.raises(ValueError):
        LatencyBudgetAllocatorConfig(warning_threshold_ratio=0).validate()

    with pytest.raises(ValueError):
        LatencyBudgetAllocatorConfig(warning_threshold_ratio=2).validate()


def test_config_rejects_invalid_critical_ratio() -> None:
    with pytest.raises(ValueError):
        LatencyBudgetAllocatorConfig(critical_threshold_ratio=1).validate()


def test_slice_requires_stage() -> None:
    with pytest.raises(ValidationError):
        LatencyBudgetSlice(
            kind=LatencyBudgetSliceKind.AUDIO_OUTPUT,
            name="Audio output",
            budget_ms=40,
            percent_of_total=5,
            owner="playback_runtime",
            stages=(),
        )


def test_plan_requires_slices() -> None:
    with pytest.raises(ValidationError):
        LatencyBudgetPlan(total_budget_ms=800, slices=())


def test_plan_requires_budget_sum_to_match_total() -> None:
    item = LatencyBudgetSlice(
        kind=LatencyBudgetSliceKind.AUDIO_OUTPUT,
        name="Audio output",
        budget_ms=40,
        percent_of_total=100,
        owner="playback_runtime",
        stages=(PipelineStage.PLAYBACK_STARTUP,),
    )

    with pytest.raises(ValidationError):
        LatencyBudgetPlan(total_budget_ms=800, slices=(item,))


def test_plan_requires_percentages_to_sum_to_100() -> None:
    item = LatencyBudgetSlice(
        kind=LatencyBudgetSliceKind.AUDIO_OUTPUT,
        name="Audio output",
        budget_ms=800,
        percent_of_total=20,
        owner="playback_runtime",
        stages=(PipelineStage.PLAYBACK_STARTUP,),
    )

    with pytest.raises(ValidationError):
        LatencyBudgetPlan(total_budget_ms=800, slices=(item,))


def test_default_plan_sums_to_800ms() -> None:
    plan = default_first_word_latency_budget_plan()

    assert plan.total_budget_ms == 800.0
    assert sum(item.budget_ms for item in plan.slices) == 800.0
    assert len(plan.slices) == 7


def test_default_plan_contains_expected_budget_slices() -> None:
    plan = default_first_word_latency_budget_plan()
    kinds = {item.kind for item in plan.slices}

    assert LatencyBudgetSliceKind.AUDIO_CAPTURE_VAD in kinds
    assert LatencyBudgetSliceKind.STT_FIRST_PARTIAL in kinds
    assert LatencyBudgetSliceKind.CONTEXT_BUILD in kinds
    assert LatencyBudgetSliceKind.MEMORY_RETRIEVAL in kinds
    assert LatencyBudgetSliceKind.LLM_FIRST_TOKEN in kinds
    assert LatencyBudgetSliceKind.TTS_FIRST_CHUNK in kinds
    assert LatencyBudgetSliceKind.AUDIO_OUTPUT in kinds


def test_plan_slice_lookup() -> None:
    plan = default_first_word_latency_budget_plan()

    item = plan.slice_for(LatencyBudgetSliceKind.LLM_FIRST_TOKEN)

    assert item is not None
    assert item.budget_ms == 300.0


def test_allocator_evaluates_profile_passed() -> None:
    profile = _budget_friendly_profile()
    allocator = LatencyBudgetAllocator(
        config=LatencyBudgetAllocatorConfig(warning_threshold_ratio=1.0)
    )

    report = allocator.evaluate_profile(profile)

    assert report.status == LatencyBudgetAllocationStatus.PASSED
    assert report.reason == LatencyBudgetAllocationReason.BUDGET_PASSED
    assert report.evaluation_count == 7
    assert report.total_budget_ms == 800.0


def test_allocator_evaluates_profile_warning() -> None:
    profile = _budget_friendly_profile()
    allocator = LatencyBudgetAllocator(
        config=LatencyBudgetAllocatorConfig(warning_threshold_ratio=0.50)
    )

    report = allocator.evaluate_profile(profile)

    assert report.status == LatencyBudgetAllocationStatus.WARNING
    assert report.warning_count > 0


def test_allocator_evaluates_profile_violation() -> None:
    profile = _budget_friendly_profile().model_copy(
        update={"llm_first_token_ms": 400.0}
    )
    allocator = LatencyBudgetAllocator()

    report = allocator.evaluate_profile(profile)

    assert report.status == LatencyBudgetAllocationStatus.VIOLATION
    assert report.violation_count > 0


def test_allocator_evaluates_profile_critical() -> None:
    profile = _budget_friendly_profile().model_copy(
        update={"llm_first_token_ms": 1000.0}
    )
    allocator = LatencyBudgetAllocator()

    report = allocator.evaluate_profile(profile)

    assert report.status == LatencyBudgetAllocationStatus.CRITICAL
    assert report.critical_count > 0


def test_allocator_identifies_over_budget_slice() -> None:
    profile = _budget_friendly_profile().model_copy(
        update={"memory_retrieval_ms": 250.0}
    )
    allocator = LatencyBudgetAllocator()

    report = allocator.evaluate_profile(profile)
    memory_eval = next(
        item
        for item in report.evaluations
        if item.slice_kind == LatencyBudgetSliceKind.MEMORY_RETRIEVAL
    )

    assert memory_eval.status == LatencyBudgetAllocationStatus.CRITICAL
    assert memory_eval.remaining_ms < 0


def test_allocator_evaluates_pipeline_report() -> None:
    profiler = VoicePipelineBaselineProfiler(
        config=VoiceBaselineProfilerConfig(record_to_pipeline_profiler=False)
    )
    sample = build_synthetic_voice_sample()
    profiler.record_sample(sample)
    baseline = profiler.create_baseline_report()

    allocator = LatencyBudgetAllocator()
    report = allocator.evaluate_profile(baseline.profiles[0])

    assert report.evaluation_count == 7
    assert report.metadata["source"] == "voice_micro_latency_profile"


def test_allocator_evaluates_voice_baseline_report() -> None:
    profiler = VoicePipelineBaselineProfiler(
        config=VoiceBaselineProfilerConfig(record_to_pipeline_profiler=False)
    )
    profiler.record_sample(build_synthetic_voice_sample(trace_id="one"))
    profiler.record_sample(build_synthetic_voice_sample(trace_id="two"))
    baseline = profiler.create_baseline_report()

    allocator = LatencyBudgetAllocator()
    reports = allocator.evaluate_voice_baseline_report(baseline)

    assert len(reports) == 2
    assert allocator.snapshot().report_count == 2
    assert (
        allocator.snapshot().last_reason
        == LatencyBudgetAllocationReason.VOICE_BASELINE_EVALUATED
    )


def test_latest_report_and_reports_are_queryable() -> None:
    allocator = LatencyBudgetAllocator()
    profile = _baseline_profile()

    report = allocator.evaluate_profile(profile)

    assert allocator.latest_report() == report
    assert allocator.reports() == (report,)


def test_allocator_snapshot_tracks_state() -> None:
    allocator = LatencyBudgetAllocator()
    profile = _baseline_profile()

    allocator.evaluate_profile(profile)
    snapshot = allocator.snapshot()

    assert snapshot.report_count == 1
    assert snapshot.last_status is not None
    assert snapshot.last_total_percent_used is not None


def test_allocator_reset_clears_reports() -> None:
    allocator = LatencyBudgetAllocator()
    profile = _baseline_profile()

    allocator.evaluate_profile(profile)
    allocator.reset()
    snapshot = allocator.snapshot()

    assert snapshot.report_count == 0
    assert snapshot.last_reason == LatencyBudgetAllocationReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert LatencyBudgetSliceKind.LLM_FIRST_TOKEN.value == "llm_first_token"
    assert LatencyBudgetAllocationStatus.PASSED.value == "passed"
    assert LatencyBudgetAllocationReason.BUDGET_PASSED.value == "budget_passed"


def _baseline_profile() -> VoiceMicroLatencyProfile:
    profiler = VoicePipelineBaselineProfiler(
        config=VoiceBaselineProfilerConfig(record_to_pipeline_profiler=False)
    )
    sample = build_synthetic_voice_sample()
    return profiler.record_sample(sample)

def _budget_friendly_profile() -> VoiceMicroLatencyProfile:
    return _baseline_profile().model_copy(
        update={
            "microphone_capture_ms": 20.0,
            "vad_detection_ms": 20.0,
            "stt_first_partial_ms": 100.0,
            "context_build_ms": 60.0,
            "memory_retrieval_ms": 80.0,
            "llm_first_token_ms": 250.0,
            "tts_first_audio_ms": 60.0,
            "playback_startup_ms": 30.0,
            "first_audio_wall_clock_ms": 620.0,
            "total_wall_clock_ms": 1200.0,
        }
    )