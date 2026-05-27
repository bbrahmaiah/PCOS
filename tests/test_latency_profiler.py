from __future__ import annotations

import time

import pytest
from pydantic import ValidationError

from jarvis.latency import (
    LatencyOperation,
    LatencySubsystem,
    PipelineFindingKind,
    PipelineFindingSeverity,
    PipelineLatencyProfiler,
    PipelineProfilerConfig,
    PipelineProfilerReason,
    PipelineSpan,
    PipelineStage,
    PipelineStageTimer,
    PipelineTraceStatus,
)


def ns(ms: int) -> int:
    return ms * 1_000_000


def span(
    *,
    trace_id: str = "trace-1",
    stage: PipelineStage = PipelineStage.STT_FIRST_PARTIAL,
    start_ms: int = 0,
    end_ms: int = 100,
    cache_hit: bool | None = None,
) -> PipelineSpan:
    return PipelineSpan(
        trace_id=trace_id,
        stage=stage,
        operation=LatencyOperation.STT_FIRST_TOKEN,
        subsystem=LatencySubsystem.PRESENCE,
        start_ns=ns(start_ms),
        end_ns=ns(end_ms),
        cache_hit=cache_hit,
    )


def test_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        PipelineProfilerConfig(name=" ").validate()


def test_config_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError):
        PipelineProfilerConfig(slow_stage_threshold_ms=0).validate()

    with pytest.raises(ValueError):
        PipelineProfilerConfig(low_overlap_ratio_threshold=2).validate()


def test_span_rejects_empty_trace_id() -> None:
    with pytest.raises(ValidationError):
        PipelineSpan(
            trace_id=" ",
            stage=PipelineStage.STT_FIRST_PARTIAL,
            operation=LatencyOperation.STT_FIRST_TOKEN,
            subsystem=LatencySubsystem.PRESENCE,
            start_ns=0,
            end_ns=1,
        )


def test_span_rejects_invalid_time_order() -> None:
    with pytest.raises(ValidationError):
        PipelineSpan(
            trace_id="trace",
            stage=PipelineStage.STT_FIRST_PARTIAL,
            operation=LatencyOperation.STT_FIRST_TOKEN,
            subsystem=LatencySubsystem.PRESENCE,
            start_ns=10,
            end_ns=1,
        )


def test_span_duration_and_latency_conversion() -> None:
    item = span(start_ms=1, end_ms=4)

    assert item.duration_ms() == 3.0
    assert item.to_latency_span().duration_ms() == 3.0


def test_profiler_starts_trace() -> None:
    profiler = PipelineLatencyProfiler()

    trace = profiler.start_trace(name="voice_turn")

    assert trace.name == "voice_turn"
    assert trace.status == PipelineTraceStatus.OPEN
    assert profiler.snapshot().trace_count == 1


def test_profiler_records_span() -> None:
    profiler = PipelineLatencyProfiler()
    trace = profiler.start_trace(name="voice_turn")

    profiler.record_span(span(trace_id=trace.trace_id))

    assert len(profiler.spans_for(trace.trace_id)) == 1
    assert profiler.snapshot().span_count == 1


def test_profiler_record_stage_resolves_operation() -> None:
    profiler = PipelineLatencyProfiler()
    trace = profiler.start_trace(name="voice_turn")

    item = profiler.record_stage(
        trace_id=trace.trace_id,
        stage=PipelineStage.MEMORY_RETRIEVAL,
        start_ns=0,
        end_ns=ns(50),
    )

    assert item.operation == LatencyOperation.MEMORY_RETRIEVAL
    assert item.subsystem == LatencySubsystem.MEMORY


def test_profiler_rejects_missing_trace_on_complete() -> None:
    profiler = PipelineLatencyProfiler()

    with pytest.raises(ValueError):
        profiler.complete_trace("missing")


def test_profiler_rejects_empty_trace_report() -> None:
    profiler = PipelineLatencyProfiler()
    trace = profiler.start_trace(name="empty")

    with pytest.raises(ValueError):
        profiler.profile_trace(trace.trace_id)


def test_profiler_builds_report_for_serial_pipeline() -> None:
    profiler = PipelineLatencyProfiler(
        config=PipelineProfilerConfig(record_to_latency_runtime=False)
    )
    trace = profiler.start_trace(name="serial")

    profiler.record_span(
        span(
            trace_id=trace.trace_id,
            stage=PipelineStage.STT_FIRST_PARTIAL,
            start_ms=0,
            end_ms=100,
        )
    )
    profiler.record_span(
        span(
            trace_id=trace.trace_id,
            stage=PipelineStage.MEMORY_RETRIEVAL,
            start_ms=120,
            end_ms=220,
        )
    )

    report = profiler.complete_trace(trace.trace_id)

    assert report.span_count == 2
    assert report.wall_clock_ms == 220.0
    assert report.serial_duration_ms == 200.0
    assert report.idle_gap_ms == 20.0
    assert len(report.gaps) == 1


def test_profiler_detects_overlap() -> None:
    profiler = PipelineLatencyProfiler(
        config=PipelineProfilerConfig(record_to_latency_runtime=False)
    )
    trace = profiler.start_trace(name="parallel")

    profiler.record_span(
        span(
            trace_id=trace.trace_id,
            stage=PipelineStage.STT_FIRST_PARTIAL,
            start_ms=0,
            end_ms=200,
        )
    )
    profiler.record_span(
        span(
            trace_id=trace.trace_id,
            stage=PipelineStage.MEMORY_RETRIEVAL,
            start_ms=50,
            end_ms=180,
        )
    )

    report = profiler.complete_trace(trace.trace_id)

    assert report.wall_clock_ms == 200.0
    assert report.serial_duration_ms == 330.0
    assert report.overlap_saved_ms == 130.0
    assert len(report.overlaps) == 1
    assert report.overlap_ratio > 0


def test_profiler_detects_slow_stage() -> None:
    profiler = PipelineLatencyProfiler(
        config=PipelineProfilerConfig(
            record_to_latency_runtime=False,
            slow_stage_threshold_ms=100,
        )
    )
    trace = profiler.start_trace(name="slow")

    profiler.record_span(
        span(
            trace_id=trace.trace_id,
            stage=PipelineStage.LLM_FIRST_TOKEN,
            start_ms=0,
            end_ms=200,
        )
    )

    report = profiler.complete_trace(trace.trace_id)

    assert any(
        finding.kind == PipelineFindingKind.SLOW_STAGE
        for finding in report.findings
    )


def test_profiler_detects_queue_stall() -> None:
    profiler = PipelineLatencyProfiler(
        config=PipelineProfilerConfig(
            record_to_latency_runtime=False,
            queue_stall_threshold_ms=50,
        )
    )
    trace = profiler.start_trace(name="queue")

    profiler.record_stage(
        trace_id=trace.trace_id,
        stage=PipelineStage.SCHEDULER_QUEUE,
        start_ns=0,
        end_ns=ns(100),
        queue_name="conversation_queue",
    )

    report = profiler.complete_trace(trace.trace_id)

    assert any(
        finding.kind == PipelineFindingKind.QUEUE_STALL
        for finding in report.findings
    )


def test_profiler_detects_worker_wait() -> None:
    profiler = PipelineLatencyProfiler(
        config=PipelineProfilerConfig(
            record_to_latency_runtime=False,
            worker_wait_threshold_ms=50,
        )
    )
    trace = profiler.start_trace(name="wait")

    profiler.record_stage(
        trace_id=trace.trace_id,
        stage=PipelineStage.WORKER_WAIT,
        start_ns=0,
        end_ns=ns(100),
        worker_id="memory_worker",
    )

    report = profiler.complete_trace(trace.trace_id)

    assert any(
        finding.kind == PipelineFindingKind.WORKER_WAIT
        for finding in report.findings
    )


def test_profiler_detects_cache_miss() -> None:
    profiler = PipelineLatencyProfiler(
        config=PipelineProfilerConfig(record_to_latency_runtime=False)
    )
    trace = profiler.start_trace(name="cache")

    profiler.record_stage(
        trace_id=trace.trace_id,
        stage=PipelineStage.CACHE_LOOKUP,
        start_ns=0,
        end_ns=ns(10),
        cache_name="context_cache",
        cache_hit=False,
    )

    report = profiler.complete_trace(trace.trace_id)

    assert any(
        finding.kind == PipelineFindingKind.CACHE_MISS
        for finding in report.findings
    )


def test_profiler_detects_interrupt_delay() -> None:
    profiler = PipelineLatencyProfiler(
        config=PipelineProfilerConfig(
            record_to_latency_runtime=False,
            interrupt_delay_threshold_ms=50,
        )
    )
    trace = profiler.start_trace(name="interrupt")

    profiler.record_stage(
        trace_id=trace.trace_id,
        stage=PipelineStage.INTERRUPT_RECOVERY,
        start_ns=0,
        end_ns=ns(100),
    )

    report = profiler.complete_trace(trace.trace_id)

    assert any(
        finding.kind == PipelineFindingKind.INTERRUPT_DELAY
        and finding.severity == PipelineFindingSeverity.CRITICAL
        for finding in report.findings
    )


def test_profiler_detects_low_overlap() -> None:
    profiler = PipelineLatencyProfiler(
        config=PipelineProfilerConfig(
            record_to_latency_runtime=False,
            low_overlap_ratio_threshold=0.5,
        )
    )
    trace = profiler.start_trace(name="serial_low_overlap")

    profiler.record_span(span(trace_id=trace.trace_id, start_ms=0, end_ms=50))
    profiler.record_span(span(trace_id=trace.trace_id, start_ms=60, end_ms=100))

    report = profiler.complete_trace(trace.trace_id)

    assert any(
        finding.kind == PipelineFindingKind.LOW_OVERLAP
        for finding in report.findings
    )


def test_stage_durations_are_reported() -> None:
    profiler = PipelineLatencyProfiler(
        config=PipelineProfilerConfig(record_to_latency_runtime=False)
    )
    trace = profiler.start_trace(name="durations")

    profiler.record_span(
        span(
            trace_id=trace.trace_id,
            stage=PipelineStage.STT_FIRST_PARTIAL,
            start_ms=0,
            end_ms=100,
        )
    )
    profiler.record_span(
        span(
            trace_id=trace.trace_id,
            stage=PipelineStage.STT_FIRST_PARTIAL,
            start_ms=120,
            end_ms=170,
        )
    )

    report = profiler.complete_trace(trace.trace_id)

    assert report.stage_durations_ms[PipelineStage.STT_FIRST_PARTIAL.value] == 150.0


def test_latest_report_and_reports_are_queryable() -> None:
    profiler = PipelineLatencyProfiler(
        config=PipelineProfilerConfig(record_to_latency_runtime=False)
    )
    trace = profiler.start_trace(name="query")

    profiler.record_span(span(trace_id=trace.trace_id))
    report = profiler.complete_trace(trace.trace_id)

    assert profiler.latest_report() == report
    assert profiler.reports() == (report,)


def test_snapshot_tracks_bottlenecks() -> None:
    profiler = PipelineLatencyProfiler(
        config=PipelineProfilerConfig(
            record_to_latency_runtime=False,
            slow_stage_threshold_ms=50,
        )
    )
    trace = profiler.start_trace(name="snapshot")

    profiler.record_span(span(trace_id=trace.trace_id, start_ms=0, end_ms=100))
    profiler.complete_trace(trace.trace_id)

    snapshot = profiler.snapshot()

    assert snapshot.report_count == 1
    assert snapshot.bottleneck_count > 0


def test_reset_clears_profiler_state() -> None:
    profiler = PipelineLatencyProfiler(
        config=PipelineProfilerConfig(record_to_latency_runtime=False)
    )
    trace = profiler.start_trace(name="reset")

    profiler.record_span(span(trace_id=trace.trace_id))
    profiler.complete_trace(trace.trace_id)
    profiler.reset()

    snapshot = profiler.snapshot()

    assert snapshot.trace_count == 0
    assert snapshot.span_count == 0
    assert snapshot.last_reason == PipelineProfilerReason.RUNTIME_RESET


def test_pipeline_stage_timer_records_span() -> None:
    profiler = PipelineLatencyProfiler(
        config=PipelineProfilerConfig(record_to_latency_runtime=False)
    )
    trace = profiler.start_trace(name="timer")

    with PipelineStageTimer(
        profiler=profiler,
        trace_id=trace.trace_id,
        stage=PipelineStage.STT_FIRST_PARTIAL,
    ) as timer:
        time.sleep(0.001)

    assert timer.span is not None
    assert len(profiler.spans_for(trace.trace_id)) == 1


def test_enum_values_are_stable() -> None:
    assert PipelineStage.MICROPHONE_CAPTURE.value == "microphone_capture"
    assert PipelineTraceStatus.COMPLETED.value == "completed"
    assert PipelineFindingKind.CACHE_MISS.value == "cache_miss"
    assert PipelineFindingSeverity.BOTTLENECK.value == "bottleneck"
    assert PipelineProfilerReason.TRACE_PROFILED.value == "trace_profiled"