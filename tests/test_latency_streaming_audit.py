from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.latency import (
    CriticalPathRank,
    LatencyBudgetAllocator,
    PipelineLatencyProfiler,
    PipelineProfilerConfig,
    PipelineStage,
    StreamingArchitectureAuditConfig,
    StreamingArchitectureAuditRuntime,
    StreamingAuditFlow,
    StreamingAuditFlowSpec,
    StreamingDebtSeverity,
    StreamingReadiness,
    VoiceBaselineProfilerConfig,
    VoicePipelineBaselineProfiler,
    build_synthetic_voice_sample,
)


def test_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        StreamingArchitectureAuditConfig(name=" ").validate()


def test_flow_spec_requires_step_5_or_later() -> None:
    with pytest.raises(ValidationError):
        StreamingAuditFlowSpec(
            flow=StreamingAuditFlow.STT_PARTIALS,
            name="STT",
            owner="presence_runtime",
            critical_path_rank=CriticalPathRank.STT,
            expected_behavior="stream partials",
            blocking_failure="waits for final transcript",
            required_for_step=4,
        )


def test_audit_declared_all_streaming_passes() -> None:
    runtime = StreamingArchitectureAuditRuntime()
    report = runtime.audit_declared_flows(
        {flow: StreamingReadiness.STREAMING for flow in StreamingAuditFlow}
    )

    assert report.passed is True
    assert report.streaming_count == len(StreamingAuditFlow)
    assert report.debt_count == 0


def test_audit_declared_blocking_detects_debt() -> None:
    runtime = StreamingArchitectureAuditRuntime()
    report = runtime.audit_declared_flows(
        {flow: StreamingReadiness.BLOCKING for flow in StreamingAuditFlow}
    )

    assert report.passed is False
    assert report.blocking_count == len(StreamingAuditFlow)
    assert report.debt_count == len(StreamingAuditFlow)
    assert report.critical_debt_count > 0
    assert report.fix_order[0] == StreamingAuditFlow.STT_PARTIALS


def test_audit_declared_partial_detects_partial_debt() -> None:
    runtime = StreamingArchitectureAuditRuntime()
    report = runtime.audit_declared_flows(
        {flow: StreamingReadiness.PARTIAL for flow in StreamingAuditFlow}
    )

    assert report.status == StreamingReadiness.PARTIAL
    assert report.partial_count == len(StreamingAuditFlow)
    assert report.debt_count == len(StreamingAuditFlow)


def test_unknown_flows_default_to_blocking() -> None:
    runtime = StreamingArchitectureAuditRuntime()

    report = runtime.audit_declared_flows({})

    assert report.status == StreamingReadiness.BLOCKING
    assert report.blocking_count == len(StreamingAuditFlow)


def test_findings_are_created_for_debt() -> None:
    runtime = StreamingArchitectureAuditRuntime()
    report = runtime.audit_declared_flows(
        {
            StreamingAuditFlow.STT_PARTIALS: StreamingReadiness.BLOCKING,
            StreamingAuditFlow.LLM_TOKEN_OUTPUT: StreamingReadiness.STREAMING,
        }
    )

    stt_eval = next(
        item
        for item in report.evaluations
        if item.flow == StreamingAuditFlow.STT_PARTIALS
    )

    assert stt_eval.findings
    assert stt_eval.findings[0].is_debt is True


def test_fix_order_uses_critical_path_rank() -> None:
    runtime = StreamingArchitectureAuditRuntime()
    report = runtime.audit_declared_flows(
        {
            StreamingAuditFlow.TTS_SYNTHESIS: StreamingReadiness.BLOCKING,
            StreamingAuditFlow.STT_PARTIALS: StreamingReadiness.BLOCKING,
            StreamingAuditFlow.MEMORY_RETRIEVAL: StreamingReadiness.BLOCKING,
        }
    )

    assert report.fix_order[0] == StreamingAuditFlow.STT_PARTIALS
    assert StreamingAuditFlow.MEMORY_RETRIEVAL in report.fix_order
    assert StreamingAuditFlow.TTS_SYNTHESIS in report.fix_order


def test_critical_path_blocking_is_critical() -> None:
    runtime = StreamingArchitectureAuditRuntime()
    report = runtime.audit_declared_flows(
        {StreamingAuditFlow.LLM_TOKEN_OUTPUT: StreamingReadiness.BLOCKING}
    )

    llm_eval = next(
        item
        for item in report.evaluations
        if item.flow == StreamingAuditFlow.LLM_TOKEN_OUTPUT
    )

    assert llm_eval.severity == StreamingDebtSeverity.CRITICAL


def test_late_path_blocking_is_medium_or_high() -> None:
    runtime = StreamingArchitectureAuditRuntime()
    report = runtime.audit_declared_flows(
        {StreamingAuditFlow.TOOL_FEEDBACK: StreamingReadiness.BLOCKING}
    )

    tool_eval = next(
        item
        for item in report.evaluations
        if item.flow == StreamingAuditFlow.TOOL_FEEDBACK
    )

    assert tool_eval.severity in {
        StreamingDebtSeverity.MEDIUM,
        StreamingDebtSeverity.HIGH,
    }


def test_audit_pipeline_report_infers_readiness() -> None:
    profiler = PipelineLatencyProfiler(
        config=PipelineProfilerConfig(record_to_latency_runtime=False)
    )
    trace = profiler.start_trace(name="voice")
    profiler.record_stage(
        trace_id=trace.trace_id,
        stage=PipelineStage.STT_FIRST_PARTIAL,
        start_ns=0,
        end_ns=100_000_000,
    )
    profiler.record_stage(
        trace_id=trace.trace_id,
        stage=PipelineStage.LLM_FIRST_TOKEN,
        start_ns=100_000_000,
        end_ns=250_000_000,
    )
    report = profiler.complete_trace(trace.trace_id)

    audit = StreamingArchitectureAuditRuntime().audit_pipeline_report(report)

    assert audit.metadata["source"] == "pipeline_profiler_report"
    assert audit.flow_count == len(StreamingAuditFlow)


def test_audit_budget_report_infers_readiness() -> None:
    voice = VoicePipelineBaselineProfiler(
        config=VoiceBaselineProfilerConfig(record_to_pipeline_profiler=False)
    )
    profile = voice.record_sample(build_synthetic_voice_sample())
    allocation = LatencyBudgetAllocator().evaluate_profile(profile)

    audit = StreamingArchitectureAuditRuntime().audit_budget_report(allocation)

    assert audit.metadata["source"] == "latency_budget_allocation_report"
    assert audit.flow_count == len(StreamingAuditFlow)


def test_latest_report_and_reports_are_queryable() -> None:
    runtime = StreamingArchitectureAuditRuntime()
    report = runtime.audit_declared_flows(
        {flow: StreamingReadiness.STREAMING for flow in StreamingAuditFlow}
    )

    assert runtime.latest_report() == report
    assert runtime.reports() == (report,)


def test_snapshot_tracks_state() -> None:
    runtime = StreamingArchitectureAuditRuntime()

    runtime.audit_declared_flows(
        {StreamingAuditFlow.STT_PARTIALS: StreamingReadiness.BLOCKING}
    )
    snapshot = runtime.snapshot()

    assert snapshot.report_count == 1
    assert snapshot.last_status == StreamingReadiness.BLOCKING
    assert snapshot.last_debt_count is not None


def test_reset_clears_runtime_state() -> None:
    runtime = StreamingArchitectureAuditRuntime()

    runtime.audit_declared_flows(
        {StreamingAuditFlow.STT_PARTIALS: StreamingReadiness.BLOCKING}
    )
    runtime.reset()

    snapshot = runtime.snapshot()

    assert snapshot.report_count == 0
    assert snapshot.last_reason is not None


def test_enum_values_are_stable() -> None:
    assert StreamingAuditFlow.STT_PARTIALS.value == "stt_partials"
    assert StreamingReadiness.BLOCKING.value == "blocking"
    assert StreamingDebtSeverity.CRITICAL.value == "critical"
    assert CriticalPathRank.STT.value == 20