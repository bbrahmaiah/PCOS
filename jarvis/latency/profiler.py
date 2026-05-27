from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from types import TracebackType
from typing import Self
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.latency.measurements import LatencyMeasurementRuntime
from jarvis.latency.models import (
    LatencyOperation,
    LatencySpan,
    LatencySubsystem,
)
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class PipelineStage(StrEnum):
    """
    End-to-end stages in the Phase 7 real-time cognition pipeline.
    """

    MICROPHONE_CAPTURE = "microphone_capture"
    VAD_DETECTION = "vad_detection"
    STT_FIRST_PARTIAL = "stt_first_partial"
    STT_FINALIZATION = "stt_finalization"
    INTENT_CLASSIFICATION = "intent_classification"
    CONTEXT_BUILD = "context_build"
    MEMORY_RETRIEVAL = "memory_retrieval"
    SCHEDULER_QUEUE = "scheduler_queue"
    WORKER_WAIT = "worker_wait"
    CACHE_LOOKUP = "cache_lookup"
    LLM_FIRST_TOKEN = "llm_first_token"
    LLM_STREAMING = "llm_streaming"
    TTS_FIRST_AUDIO = "tts_first_audio"
    TTS_STREAMING = "tts_streaming"
    PLAYBACK_STARTUP = "playback_startup"
    INTERRUPT_DETECTION = "interrupt_detection"
    INTERRUPT_RECOVERY = "interrupt_recovery"
    TOOL_FIRST_FEEDBACK = "tool_first_feedback"
    ACTION_PROGRESS_FEEDBACK = "action_progress_feedback"


class PipelineTraceStatus(StrEnum):
    """
    Pipeline trace lifecycle status.
    """

    OPEN = "open"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineFindingKind(StrEnum):
    """
    Profiler finding kinds.

    These expose invisible latency problems.
    """

    SLOW_STAGE = "slow_stage"
    QUEUE_STALL = "queue_stall"
    CACHE_MISS = "cache_miss"
    WORKER_WAIT = "worker_wait"
    SCHEDULER_DELAY = "scheduler_delay"
    INTERRUPT_DELAY = "interrupt_delay"
    PIPELINE_GAP = "pipeline_gap"
    LOW_OVERLAP = "low_overlap"


class PipelineFindingSeverity(StrEnum):
    """
    Profiler finding severity.
    """

    INFO = "info"
    WARNING = "warning"
    BOTTLENECK = "bottleneck"
    CRITICAL = "critical"


class PipelineProfilerReason(StrEnum):
    """
    Machine-readable profiler reasons.
    """

    TRACE_STARTED = "trace_started"
    SPAN_RECORDED = "span_recorded"
    TRACE_PROFILED = "trace_profiled"
    TRACE_NOT_FOUND = "trace_not_found"
    TRACE_HAS_NO_SPANS = "trace_has_no_spans"
    BOTTLENECKS_DETECTED = "bottlenecks_detected"
    RUNTIME_RESET = "runtime_reset"


class PipelineSpan(OrchestrationModel):
    """
    One measured span in the end-to-end pipeline.

    This is richer than a raw LatencySpan because it includes pipeline stage,
    worker identity, queue metadata, cache metadata, and parent relationships.
    """

    span_id: str = Field(default_factory=lambda: uuid4().hex)
    trace_id: str
    stage: PipelineStage
    operation: LatencyOperation
    subsystem: LatencySubsystem
    start_ns: int = Field(ge=0)
    end_ns: int = Field(ge=0)
    parent_span_id: str | None = None
    worker_id: str | None = None
    queue_name: str | None = None
    cache_name: str | None = None
    cache_hit: bool | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("span_id", "trace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @model_validator(mode="after")
    def _validate_time_order(self) -> PipelineSpan:
        if self.end_ns < self.start_ns:
            raise ValueError("end_ns must be greater than or equal to start_ns.")

        return self

    def duration_ms(self) -> float:
        return (self.end_ns - self.start_ns) / 1_000_000.0

    def to_latency_span(self) -> LatencySpan:
        return LatencySpan(
            span_id=self.span_id,
            operation=self.operation,
            subsystem=self.subsystem,
            start_ns=self.start_ns,
            end_ns=self.end_ns,
            trace_id=self.trace_id,
            metadata={
                **self.metadata,
                "pipeline_stage": self.stage.value,
                "worker_id": self.worker_id,
                "queue_name": self.queue_name,
                "cache_name": self.cache_name,
                "cache_hit": self.cache_hit,
            },
        )


class PipelineOverlap(OrchestrationModel):
    """
    Overlap between two spans.

    Overlap is the heart of Phase 7. Serial work feels slow. Overlapped work
    creates the living JARVIS feeling.
    """

    first_span_id: str
    second_span_id: str
    first_stage: PipelineStage
    second_stage: PipelineStage
    overlap_ms: float = Field(ge=0)

    @field_validator("first_span_id", "second_span_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class PipelineGap(OrchestrationModel):
    """
    Idle gap between spans.

    Gaps are places where JARVIS was waiting when it could have been flowing.
    """

    previous_span_id: str
    next_span_id: str
    previous_stage: PipelineStage
    next_stage: PipelineStage
    gap_ms: float = Field(ge=0)

    @field_validator("previous_span_id", "next_span_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class PipelineFinding(OrchestrationModel):
    """
    Bottleneck or diagnostic finding from the profiler.
    """

    finding_id: str = Field(default_factory=lambda: uuid4().hex)
    trace_id: str
    kind: PipelineFindingKind
    severity: PipelineFindingSeverity
    message: str
    stage: PipelineStage | None = None
    span_id: str | None = None
    duration_ms: float | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("finding_id", "trace_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class PipelineTrace(OrchestrationModel):
    """
    End-to-end trace for one user-perceived interaction.

    Example:
    microphone → VAD → STT → memory/context → LLM → TTS → playback
    """

    trace_id: str = Field(default_factory=lambda: uuid4().hex)
    name: str
    status: PipelineTraceStatus = PipelineTraceStatus.OPEN
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("trace_id", "name")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class PipelineProfilerReport(OrchestrationModel):
    """
    Full profiler report for one trace.
    """

    trace_id: str
    trace_name: str
    status: PipelineTraceStatus
    span_count: int = Field(ge=0)
    wall_clock_ms: float = Field(ge=0)
    serial_duration_ms: float = Field(ge=0)
    overlap_saved_ms: float = Field(ge=0)
    idle_gap_ms: float = Field(ge=0)
    overlap_ratio: float = Field(ge=0)
    stage_durations_ms: dict[str, float] = Field(default_factory=dict)
    overlaps: tuple[PipelineOverlap, ...] = ()
    gaps: tuple[PipelineGap, ...] = ()
    findings: tuple[PipelineFinding, ...] = ()
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("trace_id", "trace_name")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @property
    def has_bottlenecks(self) -> bool:
        return any(
            finding.severity
            in {
                PipelineFindingSeverity.BOTTLENECK,
                PipelineFindingSeverity.CRITICAL,
            }
            for finding in self.findings
        )


class PipelineProfilerSnapshot(OrchestrationModel):
    """
    Runtime diagnostics for Step 1 profiler.
    """

    name: str
    trace_count: int = Field(ge=0)
    span_count: int = Field(ge=0)
    report_count: int = Field(ge=0)
    finding_count: int = Field(ge=0)
    bottleneck_count: int = Field(ge=0)
    last_reason: PipelineProfilerReason | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class PipelineProfilerConfig:
    """
    End-to-end latency profiler configuration.
    """

    name: str = "pipeline_latency_profiler"
    slow_stage_threshold_ms: float = 300.0
    queue_stall_threshold_ms: float = 100.0
    worker_wait_threshold_ms: float = 100.0
    interrupt_delay_threshold_ms: float = 250.0
    gap_threshold_ms: float = 40.0
    low_overlap_ratio_threshold: float = 0.10
    record_to_latency_runtime: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.slow_stage_threshold_ms <= 0:
            raise ValueError("slow_stage_threshold_ms must be positive.")

        if self.queue_stall_threshold_ms <= 0:
            raise ValueError("queue_stall_threshold_ms must be positive.")

        if self.worker_wait_threshold_ms <= 0:
            raise ValueError("worker_wait_threshold_ms must be positive.")

        if self.interrupt_delay_threshold_ms <= 0:
            raise ValueError("interrupt_delay_threshold_ms must be positive.")

        if self.gap_threshold_ms < 0:
            raise ValueError("gap_threshold_ms cannot be negative.")

        if not 0 <= self.low_overlap_ratio_threshold <= 1:
            raise ValueError("low_overlap_ratio_threshold must be within 0..1.")


class PipelineLatencyProfiler:
    """
    Phase 7 Step 1 End-to-End Latency Profiler.

    Responsibilities:
    - record cross-runtime pipeline spans
    - compute per-stage timing
    - detect overlap and idle gaps
    - expose queue stalls, cache misses, worker waits, scheduler delays
    - produce bottleneck reports
    - feed Step 0 latency measurement runtime

    Non-responsibilities:
    - no optimization
    - no scheduling changes
    - no streaming rewrites
    - no performance guessing
    """

    def __init__(
        self,
        *,
        config: PipelineProfilerConfig | None = None,
        latency_runtime: LatencyMeasurementRuntime | None = None,
    ) -> None:
        self._config = config or PipelineProfilerConfig()
        self._config.validate()

        self._latency_runtime = latency_runtime or LatencyMeasurementRuntime()
        self._traces: dict[str, PipelineTrace] = {}
        self._spans: dict[str, list[PipelineSpan]] = defaultdict(list)
        self._reports: list[PipelineProfilerReport] = []
        self._lock = RLock()
        self._last_reason: PipelineProfilerReason | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def start_trace(
        self,
        *,
        name: str,
        trace_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> PipelineTrace:
        trace = PipelineTrace(
            trace_id=trace_id or uuid4().hex,
            name=name,
            metadata=metadata or {},
        )

        with self._lock:
            self._traces[trace.trace_id] = trace
            self._last_reason = PipelineProfilerReason.TRACE_STARTED

        return trace

    def record_span(self, span: PipelineSpan) -> PipelineSpan:
        with self._lock:
            if span.trace_id not in self._traces:
                self._traces[span.trace_id] = PipelineTrace(
                    trace_id=span.trace_id,
                    name="implicit_trace",
                )

            self._spans[span.trace_id].append(span)
            self._last_reason = PipelineProfilerReason.SPAN_RECORDED

        if self._config.record_to_latency_runtime:
            self._latency_runtime.record_span(span.to_latency_span())

        return span

    def record_stage(
        self,
        *,
        trace_id: str,
        stage: PipelineStage,
        start_ns: int,
        end_ns: int,
        operation: LatencyOperation | None = None,
        subsystem: LatencySubsystem | None = None,
        parent_span_id: str | None = None,
        worker_id: str | None = None,
        queue_name: str | None = None,
        cache_name: str | None = None,
        cache_hit: bool | None = None,
        metadata: dict[str, object] | None = None,
    ) -> PipelineSpan:
        resolved_operation, resolved_subsystem = self._resolve_stage(stage)

        span = PipelineSpan(
            trace_id=trace_id,
            stage=stage,
            operation=operation or resolved_operation,
            subsystem=subsystem or resolved_subsystem,
            start_ns=start_ns,
            end_ns=end_ns,
            parent_span_id=parent_span_id,
            worker_id=worker_id,
            queue_name=queue_name,
            cache_name=cache_name,
            cache_hit=cache_hit,
            metadata=metadata or {},
        )

        return self.record_span(span)

    def complete_trace(self, trace_id: str) -> PipelineProfilerReport:
        with self._lock:
            trace = self._traces.get(trace_id)

            if trace is None:
                self._last_reason = PipelineProfilerReason.TRACE_NOT_FOUND
                raise ValueError(f"trace not found: {trace_id}")

            self._traces[trace_id] = PipelineTrace(
                trace_id=trace.trace_id,
                name=trace.name,
                status=PipelineTraceStatus.COMPLETED,
                metadata=dict(trace.metadata),
            )

        report = self.profile_trace(trace_id)

        return report

    def profile_trace(self, trace_id: str) -> PipelineProfilerReport:
        with self._lock:
            trace = self._traces.get(trace_id)
            spans = tuple(self._spans.get(trace_id, ()))

        if trace is None:
            with self._lock:
                self._last_reason = PipelineProfilerReason.TRACE_NOT_FOUND

            raise ValueError(f"trace not found: {trace_id}")

        if not spans:
            with self._lock:
                self._last_reason = PipelineProfilerReason.TRACE_HAS_NO_SPANS

            raise ValueError(f"trace has no spans: {trace_id}")

        report = self._build_report(trace=trace, spans=spans)

        with self._lock:
            self._reports.append(report)
            self._last_reason = (
                PipelineProfilerReason.BOTTLENECKS_DETECTED
                if report.has_bottlenecks
                else PipelineProfilerReason.TRACE_PROFILED
            )

        return report

    def spans_for(self, trace_id: str) -> tuple[PipelineSpan, ...]:
        with self._lock:
            return tuple(self._spans.get(trace_id, ()))

    def latest_report(self) -> PipelineProfilerReport | None:
        with self._lock:
            if not self._reports:
                return None

            return self._reports[-1]

    def reports(self) -> tuple[PipelineProfilerReport, ...]:
        with self._lock:
            return tuple(self._reports)

    def snapshot(self) -> PipelineProfilerSnapshot:
        with self._lock:
            reports = tuple(self._reports)
            span_count = sum(len(spans) for spans in self._spans.values())
            finding_count = sum(len(report.findings) for report in reports)
            bottleneck_count = sum(
                1
                for report in reports
                for finding in report.findings
                if finding.severity
                in {
                    PipelineFindingSeverity.BOTTLENECK,
                    PipelineFindingSeverity.CRITICAL,
                }
            )

            return PipelineProfilerSnapshot(
                name=self.name,
                trace_count=len(self._traces),
                span_count=span_count,
                report_count=len(reports),
                finding_count=finding_count,
                bottleneck_count=bottleneck_count,
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        with self._lock:
            self._traces.clear()
            self._spans.clear()
            self._reports.clear()
            self._last_reason = PipelineProfilerReason.RUNTIME_RESET

    def _build_report(
        self,
        *,
        trace: PipelineTrace,
        spans: tuple[PipelineSpan, ...],
    ) -> PipelineProfilerReport:
        sorted_spans = tuple(sorted(spans, key=lambda item: item.start_ns))
        start_ns = min(span.start_ns for span in sorted_spans)
        end_ns = max(span.end_ns for span in sorted_spans)
        wall_clock_ms = (end_ns - start_ns) / 1_000_000.0
        serial_duration_ms = sum(span.duration_ms() for span in sorted_spans)

        gaps = self._detect_gaps(sorted_spans)
        overlaps = self._detect_overlaps(sorted_spans)
        idle_gap_ms = sum(gap.gap_ms for gap in gaps)
        overlap_saved_ms = max(0.0, serial_duration_ms - wall_clock_ms)
        overlap_ratio = (
            overlap_saved_ms / serial_duration_ms
            if serial_duration_ms > 0
            else 0.0
        )
        stage_durations = self._stage_durations(sorted_spans)
        findings = self._findings(
            trace_id=trace.trace_id,
            spans=sorted_spans,
            gaps=gaps,
            overlap_ratio=overlap_ratio,
        )

        return PipelineProfilerReport(
            trace_id=trace.trace_id,
            trace_name=trace.name,
            status=trace.status,
            span_count=len(sorted_spans),
            wall_clock_ms=wall_clock_ms,
            serial_duration_ms=serial_duration_ms,
            overlap_saved_ms=overlap_saved_ms,
            idle_gap_ms=idle_gap_ms,
            overlap_ratio=overlap_ratio,
            stage_durations_ms=stage_durations,
            overlaps=overlaps,
            gaps=gaps,
            findings=findings,
        )

    def _findings(
        self,
        *,
        trace_id: str,
        spans: tuple[PipelineSpan, ...],
        gaps: tuple[PipelineGap, ...],
        overlap_ratio: float,
    ) -> tuple[PipelineFinding, ...]:
        findings: list[PipelineFinding] = []

        for span in spans:
            findings.extend(self._span_findings(trace_id=trace_id, span=span))

        for gap in gaps:
            if gap.gap_ms >= self._config.gap_threshold_ms:
                findings.append(
                    PipelineFinding(
                        trace_id=trace_id,
                        kind=PipelineFindingKind.PIPELINE_GAP,
                        severity=PipelineFindingSeverity.WARNING,
                        message="pipeline idle gap detected",
                        stage=gap.next_stage,
                        duration_ms=gap.gap_ms,
                        metadata={
                            "previous_stage": gap.previous_stage.value,
                            "next_stage": gap.next_stage.value,
                        },
                    )
                )

        if overlap_ratio < self._config.low_overlap_ratio_threshold and len(spans) > 1:
            findings.append(
                PipelineFinding(
                    trace_id=trace_id,
                    kind=PipelineFindingKind.LOW_OVERLAP,
                    severity=PipelineFindingSeverity.BOTTLENECK,
                    message="pipeline has low overlap and may be too sequential",
                    duration_ms=overlap_ratio,
                    metadata={
                        "overlap_ratio": overlap_ratio,
                        "threshold": self._config.low_overlap_ratio_threshold,
                    },
                )
            )

        return tuple(findings)

    def _span_findings(
        self,
        *,
        trace_id: str,
        span: PipelineSpan,
    ) -> tuple[PipelineFinding, ...]:
        findings: list[PipelineFinding] = []
        duration_ms = span.duration_ms()

        if duration_ms >= self._config.slow_stage_threshold_ms:
            findings.append(
                PipelineFinding(
                    trace_id=trace_id,
                    kind=PipelineFindingKind.SLOW_STAGE,
                    severity=PipelineFindingSeverity.BOTTLENECK,
                    message="slow pipeline stage detected",
                    stage=span.stage,
                    span_id=span.span_id,
                    duration_ms=duration_ms,
                )
            )

        if (
            span.stage == PipelineStage.SCHEDULER_QUEUE
            and duration_ms >= self._config.queue_stall_threshold_ms
        ):
            findings.append(
                PipelineFinding(
                    trace_id=trace_id,
                    kind=PipelineFindingKind.QUEUE_STALL,
                    severity=PipelineFindingSeverity.BOTTLENECK,
                    message="scheduler queue stall detected",
                    stage=span.stage,
                    span_id=span.span_id,
                    duration_ms=duration_ms,
                    metadata={"queue_name": span.queue_name},
                )
            )

        if (
            span.stage == PipelineStage.WORKER_WAIT
            and duration_ms >= self._config.worker_wait_threshold_ms
        ):
            findings.append(
                PipelineFinding(
                    trace_id=trace_id,
                    kind=PipelineFindingKind.WORKER_WAIT,
                    severity=PipelineFindingSeverity.BOTTLENECK,
                    message="worker wait state detected",
                    stage=span.stage,
                    span_id=span.span_id,
                    duration_ms=duration_ms,
                    metadata={"worker_id": span.worker_id},
                )
            )

        if span.stage == PipelineStage.CACHE_LOOKUP and span.cache_hit is False:
            findings.append(
                PipelineFinding(
                    trace_id=trace_id,
                    kind=PipelineFindingKind.CACHE_MISS,
                    severity=PipelineFindingSeverity.WARNING,
                    message="cache miss detected",
                    stage=span.stage,
                    span_id=span.span_id,
                    duration_ms=duration_ms,
                    metadata={"cache_name": span.cache_name},
                )
            )

        if (
            span.stage
            in {
                PipelineStage.INTERRUPT_DETECTION,
                PipelineStage.INTERRUPT_RECOVERY,
            }
            and duration_ms >= self._config.interrupt_delay_threshold_ms
        ):
            findings.append(
                PipelineFinding(
                    trace_id=trace_id,
                    kind=PipelineFindingKind.INTERRUPT_DELAY,
                    severity=PipelineFindingSeverity.CRITICAL,
                    message="interrupt latency exceeded profiler threshold",
                    stage=span.stage,
                    span_id=span.span_id,
                    duration_ms=duration_ms,
                )
            )

        return tuple(findings)

    @staticmethod
    def _stage_durations(
        spans: tuple[PipelineSpan, ...],
    ) -> dict[str, float]:
        durations: dict[str, float] = defaultdict(float)

        for span in spans:
            durations[span.stage.value] += span.duration_ms()

        return dict(durations)

    @staticmethod
    def _detect_gaps(
        spans: tuple[PipelineSpan, ...],
    ) -> tuple[PipelineGap, ...]:
        if len(spans) < 2:
            return ()

        gaps: list[PipelineGap] = []
        cursor = spans[0]

        for current in spans[1:]:
            if current.start_ns > cursor.end_ns:
                gaps.append(
                    PipelineGap(
                        previous_span_id=cursor.span_id,
                        next_span_id=current.span_id,
                        previous_stage=cursor.stage,
                        next_stage=current.stage,
                        gap_ms=(current.start_ns - cursor.end_ns) / 1_000_000.0,
                    )
                )

            if current.end_ns > cursor.end_ns:
                cursor = current

        return tuple(gaps)

    @staticmethod
    def _detect_overlaps(
        spans: tuple[PipelineSpan, ...],
    ) -> tuple[PipelineOverlap, ...]:
        overlaps: list[PipelineOverlap] = []

        for index, first in enumerate(spans):
            for second in spans[index + 1 :]:
                latest_start = max(first.start_ns, second.start_ns)
                earliest_end = min(first.end_ns, second.end_ns)

                if earliest_end > latest_start:
                    overlaps.append(
                        PipelineOverlap(
                            first_span_id=first.span_id,
                            second_span_id=second.span_id,
                            first_stage=first.stage,
                            second_stage=second.stage,
                            overlap_ms=(
                                earliest_end - latest_start
                            )
                            / 1_000_000.0,
                        )
                    )

        return tuple(overlaps)

    @staticmethod
    def _resolve_stage(
        stage: PipelineStage,
    ) -> tuple[LatencyOperation, LatencySubsystem]:
        mapping: dict[PipelineStage, tuple[LatencyOperation, LatencySubsystem]] = {
            PipelineStage.MICROPHONE_CAPTURE: (
                LatencyOperation.STT_FIRST_TOKEN,
                LatencySubsystem.PRESENCE,
            ),
            PipelineStage.VAD_DETECTION: (
                LatencyOperation.STT_FIRST_TOKEN,
                LatencySubsystem.PRESENCE,
            ),
            PipelineStage.STT_FIRST_PARTIAL: (
                LatencyOperation.STT_FIRST_TOKEN,
                LatencySubsystem.PRESENCE,
            ),
            PipelineStage.STT_FINALIZATION: (
                LatencyOperation.STT_FINALIZATION,
                LatencySubsystem.PRESENCE,
            ),
            PipelineStage.INTENT_CLASSIFICATION: (
                LatencyOperation.CONTEXT_BUILD,
                LatencySubsystem.COGNITION,
            ),
            PipelineStage.CONTEXT_BUILD: (
                LatencyOperation.CONTEXT_BUILD,
                LatencySubsystem.COGNITION,
            ),
            PipelineStage.MEMORY_RETRIEVAL: (
                LatencyOperation.MEMORY_RETRIEVAL,
                LatencySubsystem.MEMORY,
            ),
            PipelineStage.SCHEDULER_QUEUE: (
                LatencyOperation.INTERRUPT_RESPONSE,
                LatencySubsystem.ORCHESTRATION,
            ),
            PipelineStage.WORKER_WAIT: (
                LatencyOperation.INTERRUPT_RESPONSE,
                LatencySubsystem.ORCHESTRATION,
            ),
            PipelineStage.CACHE_LOOKUP: (
                LatencyOperation.CONTEXT_BUILD,
                LatencySubsystem.COGNITION,
            ),
            PipelineStage.LLM_FIRST_TOKEN: (
                LatencyOperation.LLM_FIRST_TOKEN,
                LatencySubsystem.COGNITION,
            ),
            PipelineStage.LLM_STREAMING: (
                LatencyOperation.LLM_FULL_RESPONSE,
                LatencySubsystem.COGNITION,
            ),
            PipelineStage.TTS_FIRST_AUDIO: (
                LatencyOperation.TTS_FIRST_AUDIO,
                LatencySubsystem.PRESENCE,
            ),
            PipelineStage.TTS_STREAMING: (
                LatencyOperation.TTS_FIRST_AUDIO,
                LatencySubsystem.PRESENCE,
            ),
            PipelineStage.PLAYBACK_STARTUP: (
                LatencyOperation.PLAYBACK_STARTUP,
                LatencySubsystem.PLAYBACK,
            ),
            PipelineStage.INTERRUPT_DETECTION: (
                LatencyOperation.INTERRUPT_RESPONSE,
                LatencySubsystem.ORCHESTRATION,
            ),
            PipelineStage.INTERRUPT_RECOVERY: (
                LatencyOperation.INTERRUPT_RESPONSE,
                LatencySubsystem.ORCHESTRATION,
            ),
            PipelineStage.TOOL_FIRST_FEEDBACK: (
                LatencyOperation.TOOL_FIRST_FEEDBACK,
                LatencySubsystem.TOOLS,
            ),
            PipelineStage.ACTION_PROGRESS_FEEDBACK: (
                LatencyOperation.ACTION_PROGRESS_FEEDBACK,
                LatencySubsystem.TOOLS,
            ),
        }

        return mapping[stage]


class PipelineStageTimer:
    """
    Context manager for timing one pipeline stage.
    """

    def __init__(
        self,
        *,
        profiler: PipelineLatencyProfiler,
        trace_id: str,
        stage: PipelineStage,
        operation: LatencyOperation | None = None,
        subsystem: LatencySubsystem | None = None,
        parent_span_id: str | None = None,
        worker_id: str | None = None,
        queue_name: str | None = None,
        cache_name: str | None = None,
        cache_hit: bool | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self._profiler = profiler
        self._trace_id = trace_id
        self._stage = stage
        self._operation = operation
        self._subsystem = subsystem
        self._parent_span_id = parent_span_id
        self._worker_id = worker_id
        self._queue_name = queue_name
        self._cache_name = cache_name
        self._cache_hit = cache_hit
        self._metadata = metadata or {}
        self._start_ns: int | None = None
        self._span: PipelineSpan | None = None

    @property
    def span(self) -> PipelineSpan | None:
        return self._span

    def __enter__(self) -> Self:
        self._start_ns = time.perf_counter_ns()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        end_ns = time.perf_counter_ns()

        if self._start_ns is None:
            raise RuntimeError("pipeline timer exited before entering")

        self._span = self._profiler.record_stage(
            trace_id=self._trace_id,
            stage=self._stage,
            start_ns=self._start_ns,
            end_ns=end_ns,
            operation=self._operation,
            subsystem=self._subsystem,
            parent_span_id=self._parent_span_id,
            worker_id=self._worker_id,
            queue_name=self._queue_name,
            cache_name=self._cache_name,
            cache_hit=self._cache_hit,
            metadata=dict(self._metadata),
        )