from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.latency import (
    PipelineStage,
    VoiceBaselineProfilerConfig,
    VoiceBaselineReason,
    VoiceBaselineSample,
    VoiceBaselineScenario,
    VoiceBaselineStatus,
    VoiceMicroLatencyKind,
    VoicePipelineBaselineProfiler,
    build_synthetic_voice_sample,
)


def test_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        VoiceBaselineProfilerConfig(name=" ").validate()


def test_config_rejects_bad_first_audio_thresholds() -> None:
    config = VoiceBaselineProfilerConfig(
        first_audio_degraded_ms=1000,
        first_audio_failed_ms=500,
    )

    with pytest.raises(ValueError):
        config.validate()


def test_config_rejects_bad_total_thresholds() -> None:
    config = VoiceBaselineProfilerConfig(
        total_degraded_ms=5000,
        total_failed_ms=4000,
    )

    with pytest.raises(ValueError):
        config.validate()


def test_config_rejects_invalid_sample_count() -> None:
    with pytest.raises(ValueError):
        VoiceBaselineProfilerConfig(minimum_samples_for_baseline=0).validate()


def test_sample_requires_spans() -> None:
    with pytest.raises(ValidationError):
        VoiceBaselineSample(
            scenario=VoiceBaselineScenario.BASIC_QUESTION,
            trace_id="trace",
            spans=(),
        )


def test_sample_rejects_span_trace_mismatch() -> None:
    sample = build_synthetic_voice_sample(trace_id="trace-1")

    with pytest.raises(ValidationError):
        VoiceBaselineSample(
            scenario=VoiceBaselineScenario.BASIC_QUESTION,
            trace_id="different",
            spans=sample.spans,
        )


def test_synthetic_sample_contains_voice_pipeline_stages() -> None:
    sample = build_synthetic_voice_sample()
    stages = {span.stage for span in sample.spans}

    assert PipelineStage.MICROPHONE_CAPTURE in stages
    assert PipelineStage.VAD_DETECTION in stages
    assert PipelineStage.STT_FIRST_PARTIAL in stages
    assert PipelineStage.STT_FINALIZATION in stages
    assert PipelineStage.MEMORY_RETRIEVAL in stages
    assert PipelineStage.LLM_FIRST_TOKEN in stages
    assert PipelineStage.TTS_FIRST_AUDIO in stages
    assert PipelineStage.PLAYBACK_STARTUP in stages


def test_profiler_records_sample_and_profile() -> None:
    profiler = VoicePipelineBaselineProfiler(
        config=VoiceBaselineProfilerConfig(record_to_pipeline_profiler=False)
    )
    sample = build_synthetic_voice_sample()

    profile = profiler.record_sample(sample)

    assert profile.trace_id == sample.trace_id
    assert profile.stt_first_partial_ms > 0
    assert profile.tts_first_audio_ms > 0
    assert profile.playback_startup_ms > 0
    assert profiler.snapshot().sample_count == 1


def test_profiler_creates_baseline_report() -> None:
    profiler = VoicePipelineBaselineProfiler(
        config=VoiceBaselineProfilerConfig(record_to_pipeline_profiler=False)
    )

    profiler.record_sample(build_synthetic_voice_sample())
    report = profiler.create_baseline_report()

    assert report.status == VoiceBaselineStatus.PASSED
    assert report.reason == VoiceBaselineReason.BASELINE_CREATED
    assert report.sample_count == 1
    assert report.profile_count == 1
    assert report.aggregate_count == 1


def test_profiler_fails_when_not_enough_samples() -> None:
    profiler = VoicePipelineBaselineProfiler(
        config=VoiceBaselineProfilerConfig(
            record_to_pipeline_profiler=False,
            minimum_samples_for_baseline=2,
        )
    )

    profiler.record_sample(build_synthetic_voice_sample())
    report = profiler.create_baseline_report()

    assert report.status == VoiceBaselineStatus.FAILED
    assert report.reason == VoiceBaselineReason.BASELINE_FAILED


def test_profiler_marks_degraded_baseline() -> None:
    profiler = VoicePipelineBaselineProfiler(
        config=VoiceBaselineProfilerConfig(
            record_to_pipeline_profiler=False,
            first_audio_degraded_ms=500,
            first_audio_failed_ms=5000,
            total_degraded_ms=2000,
            total_failed_ms=10000,
        )
    )

    profiler.record_sample(build_synthetic_voice_sample())
    report = profiler.create_baseline_report()

    assert report.status == VoiceBaselineStatus.DEGRADED
    assert report.degraded_count == 1


def test_profiler_marks_failed_baseline() -> None:
    profiler = VoicePipelineBaselineProfiler(
        config=VoiceBaselineProfilerConfig(
            record_to_pipeline_profiler=False,
            first_audio_degraded_ms=100,
            first_audio_failed_ms=200,
            total_degraded_ms=100,
            total_failed_ms=200,
        )
    )

    profiler.record_sample(build_synthetic_voice_sample(slow=True))
    report = profiler.create_baseline_report()

    assert report.status == VoiceBaselineStatus.FAILED
    assert report.failed_count == 1


def test_profiler_aggregates_multiple_samples() -> None:
    profiler = VoicePipelineBaselineProfiler(
        config=VoiceBaselineProfilerConfig(record_to_pipeline_profiler=False)
    )

    profiler.record_sample(
        build_synthetic_voice_sample(
            scenario=VoiceBaselineScenario.BASIC_QUESTION,
            trace_id="one",
        )
    )
    profiler.record_sample(
        build_synthetic_voice_sample(
            scenario=VoiceBaselineScenario.BASIC_QUESTION,
            trace_id="two",
            offset_ms=10,
        )
    )
    report = profiler.create_baseline_report()

    assert report.sample_count == 2
    assert report.aggregates[0].sample_count == 2
    assert report.aggregates[0].first_audio_p95_ms > 0
    assert report.aggregates[0].first_audio_worst_ms > 0


def test_profiler_groups_by_scenario() -> None:
    profiler = VoicePipelineBaselineProfiler(
        config=VoiceBaselineProfilerConfig(record_to_pipeline_profiler=False)
    )

    profiler.record_sample(
        build_synthetic_voice_sample(
            scenario=VoiceBaselineScenario.BASIC_QUESTION,
            trace_id="basic",
        )
    )
    profiler.record_sample(
        build_synthetic_voice_sample(
            scenario=VoiceBaselineScenario.MEMORY_QUESTION,
            trace_id="memory",
        )
    )
    report = profiler.create_baseline_report()

    assert report.aggregate_count == 2


def test_latest_report_and_reports_are_queryable() -> None:
    profiler = VoicePipelineBaselineProfiler(
        config=VoiceBaselineProfilerConfig(record_to_pipeline_profiler=False)
    )

    profiler.record_sample(build_synthetic_voice_sample())
    report = profiler.create_baseline_report()

    assert profiler.latest_report() == report
    assert profiler.reports() == (report,)


def test_reset_clears_runtime_state() -> None:
    profiler = VoicePipelineBaselineProfiler(
        config=VoiceBaselineProfilerConfig(record_to_pipeline_profiler=False)
    )

    profiler.record_sample(build_synthetic_voice_sample())
    profiler.create_baseline_report()
    profiler.reset()

    snapshot = profiler.snapshot()

    assert snapshot.sample_count == 0
    assert snapshot.report_count == 0
    assert snapshot.last_reason == VoiceBaselineReason.RUNTIME_RESET


def test_report_collects_pipeline_findings() -> None:
    profiler = VoicePipelineBaselineProfiler(
        config=VoiceBaselineProfilerConfig(record_to_pipeline_profiler=False)
    )

    profiler.record_sample(build_synthetic_voice_sample(slow=True))
    report = profiler.create_baseline_report()

    assert len(report.findings) > 0


def test_enum_values_are_stable() -> None:
    assert VoiceBaselineScenario.BASIC_QUESTION.value == "basic_question"
    assert VoiceBaselineStatus.PASSED.value == "passed"
    assert VoiceBaselineReason.BASELINE_CREATED.value == "baseline_created"
    assert VoiceMicroLatencyKind.VAD_DETECTION.value == "vad_detection"