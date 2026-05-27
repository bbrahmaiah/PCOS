from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from statistics import mean
from threading import RLock
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.latency.models import (
    LatencyOperation,
    LatencyPercentile,
    LatencySubsystem,
)
from jarvis.latency.profiler import (
    PipelineFinding,
    PipelineLatencyProfiler,
    PipelineProfilerConfig,
    PipelineProfilerReport,
    PipelineSpan,
    PipelineStage,
)
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class VoiceBaselineScenario(StrEnum):
    """
    Voice baseline scenario.

    Phase 7 Step 2 focuses on measuring voice-path micro-latency before
    optimizing anything.
    """

    BASIC_QUESTION = "basic_question"
    MEMORY_QUESTION = "memory_question"
    TOOL_INTENT = "tool_intent"
    INTERRUPTED_RESPONSE = "interrupted_response"
    LONG_TURN = "long_turn"
    QUICK_COMMAND = "quick_command"


class VoiceBaselineStatus(StrEnum):
    """
    Baseline run status.
    """

    PASSED = "passed"
    DEGRADED = "degraded"
    FAILED = "failed"


class VoiceBaselineReason(StrEnum):
    """
    Machine-readable baseline profiling reasons.
    """

    SAMPLE_RECORDED = "sample_recorded"
    BASELINE_CREATED = "baseline_created"
    BASELINE_DEGRADED = "baseline_degraded"
    BASELINE_FAILED = "baseline_failed"
    SAMPLE_REJECTED = "sample_rejected"
    RUNTIME_RESET = "runtime_reset"


class VoiceMicroLatencyKind(StrEnum):
    """
    Human-perceived micro-latency points in the voice path.
    """

    WAKE_DETECTION = "wake_detection"
    MICROPHONE_CAPTURE = "microphone_capture"
    VAD_DETECTION = "vad_detection"
    STT_FIRST_PARTIAL = "stt_first_partial"
    STT_FINALIZATION = "stt_finalization"
    ROUTING_DECISION = "routing_decision"
    CONTEXT_BUILD = "context_build"
    MEMORY_RETRIEVAL = "memory_retrieval"
    LLM_FIRST_TOKEN = "llm_first_token"
    TTS_FIRST_AUDIO = "tts_first_audio"
    PLAYBACK_STARTUP = "playback_startup"
    INTERRUPT_RECOVERY = "interrupt_recovery"


class VoiceBaselineSample(OrchestrationModel):
    """
    One voice baseline sample.

    In real runs, this is produced by instrumented Presence/Cognition/Memory/TTS.
    In tests, synthetic spans make the baseline repeatable.
    """

    sample_id: str = Field(default_factory=lambda: uuid4().hex)
    scenario: VoiceBaselineScenario
    trace_id: str
    spans: tuple[PipelineSpan, ...]
    expected_first_audio_ms: float = Field(default=800.0, gt=0)
    expected_total_ms: float = Field(default=2500.0, gt=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("sample_id", "trace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @model_validator(mode="after")
    def _validate_spans(self) -> VoiceBaselineSample:
        if not self.spans:
            raise ValueError("voice baseline sample requires at least one span.")

        for span in self.spans:
            if span.trace_id != self.trace_id:
                raise ValueError("all sample spans must share the sample trace_id.")

        return self


class VoiceMicroLatencyProfile(OrchestrationModel):
    """
    Micro-latency profile extracted from one pipeline report.
    """

    trace_id: str
    scenario: VoiceBaselineScenario
    wake_detection_ms: float = Field(ge=0)
    microphone_capture_ms: float = Field(ge=0)
    vad_detection_ms: float = Field(ge=0)
    stt_first_partial_ms: float = Field(ge=0)
    stt_finalization_ms: float = Field(ge=0)
    routing_decision_ms: float = Field(ge=0)
    context_build_ms: float = Field(ge=0)
    memory_retrieval_ms: float = Field(ge=0)
    llm_first_token_ms: float = Field(ge=0)
    tts_first_audio_ms: float = Field(ge=0)
    playback_startup_ms: float = Field(ge=0)
    interrupt_recovery_ms: float = Field(ge=0)
    first_audio_wall_clock_ms: float = Field(ge=0)
    total_wall_clock_ms: float = Field(ge=0)
    overlap_ratio: float = Field(ge=0)
    idle_gap_ms: float = Field(ge=0)
    finding_count: int = Field(ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("trace_id")
    @classmethod
    def _required_trace_id(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("trace_id cannot be empty.")

        return cleaned


class VoiceBaselineAggregate(OrchestrationModel):
    """
    Aggregated baseline metrics for one scenario.
    """

    scenario: VoiceBaselineScenario
    sample_count: int = Field(ge=0)
    first_audio_p50_ms: float = Field(ge=0)
    first_audio_p90_ms: float = Field(ge=0)
    first_audio_p95_ms: float = Field(ge=0)
    first_audio_p99_ms: float = Field(ge=0)
    first_audio_worst_ms: float = Field(ge=0)
    total_p50_ms: float = Field(ge=0)
    total_p95_ms: float = Field(ge=0)
    total_worst_ms: float = Field(ge=0)
    average_overlap_ratio: float = Field(ge=0)
    average_idle_gap_ms: float = Field(ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)


class VoiceBaselineReport(OrchestrationModel):
    """
    Full voice pipeline baseline report.
    """

    report_id: str = Field(default_factory=lambda: uuid4().hex)
    status: VoiceBaselineStatus
    reason: VoiceBaselineReason
    summary: str
    sample_count: int = Field(ge=0)
    profile_count: int = Field(ge=0)
    aggregate_count: int = Field(ge=0)
    degraded_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    profiles: tuple[VoiceMicroLatencyProfile, ...]
    aggregates: tuple[VoiceBaselineAggregate, ...]
    findings: tuple[PipelineFinding, ...] = ()
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("report_id", "summary")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class VoiceBaselineProfilerConfig:
    """
    Voice baseline profiler configuration.

    These are not final optimization targets. They define when the baseline is
    considered degraded enough to highlight.
    """

    name: str = "voice_pipeline_baseline_profiler"
    first_audio_degraded_ms: float = 800.0
    first_audio_failed_ms: float = 1200.0
    total_degraded_ms: float = 3500.0
    total_failed_ms: float = 6000.0
    minimum_samples_for_baseline: int = 1
    record_to_pipeline_profiler: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.first_audio_degraded_ms <= 0:
            raise ValueError("first_audio_degraded_ms must be positive.")

        if self.first_audio_failed_ms < self.first_audio_degraded_ms:
            raise ValueError(
                "first_audio_failed_ms must be greater than or equal to degraded."
            )

        if self.total_degraded_ms <= 0:
            raise ValueError("total_degraded_ms must be positive.")

        if self.total_failed_ms < self.total_degraded_ms:
            raise ValueError(
                "total_failed_ms must be greater than or equal to degraded."
            )

        if self.minimum_samples_for_baseline < 1:
            raise ValueError("minimum_samples_for_baseline must be positive.")


@dataclass(frozen=True, slots=True)
class VoiceBaselineRuntimeSnapshot:
    """
    Runtime diagnostics for Step 2 voice baseline profiling.
    """

    name: str
    sample_count: int
    report_count: int
    last_status: VoiceBaselineStatus | None
    last_reason: VoiceBaselineReason | None


class VoicePipelineBaselineProfiler:
    """
    Phase 7 Step 2 Voice Pipeline Baseline Profiler.

    Responsibilities:
    - record baseline samples across the voice/cognition/audio path
    - extract micro-latency profiles
    - aggregate p50/p90/p95/p99/worst for first-audio and total latency
    - expose degraded/failed baseline status
    - identify voice-path bottlenecks before optimization begins

    Non-responsibilities:
    - no optimization
    - no scheduler mutation
    - no streaming pipeline rewrite
    - no latency masking
    """

    def __init__(
        self,
        *,
        config: VoiceBaselineProfilerConfig | None = None,
        pipeline_profiler: PipelineLatencyProfiler | None = None,
    ) -> None:
        self._config = config or VoiceBaselineProfilerConfig()
        self._config.validate()

        self._pipeline_profiler = pipeline_profiler or PipelineLatencyProfiler(
            config=PipelineProfilerConfig(record_to_latency_runtime=True)
        )
        self._samples: list[VoiceBaselineSample] = []
        self._reports: list[VoiceBaselineReport] = []
        self._lock = RLock()
        self._last_status: VoiceBaselineStatus | None = None
        self._last_reason: VoiceBaselineReason | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def record_sample(
        self,
        sample: VoiceBaselineSample,
    ) -> VoiceMicroLatencyProfile:
        report = self._profile_sample(sample)

        profile = self._profile_from_report(
            sample=sample,
            report=report,
        )

        with self._lock:
            self._samples.append(sample)
            self._last_status = VoiceBaselineStatus.PASSED
            self._last_reason = VoiceBaselineReason.SAMPLE_RECORDED

        return profile

    def create_baseline_report(self) -> VoiceBaselineReport:
        with self._lock:
            samples = tuple(self._samples)

        if len(samples) < self._config.minimum_samples_for_baseline:
            report = VoiceBaselineReport(
                status=VoiceBaselineStatus.FAILED,
                reason=VoiceBaselineReason.BASELINE_FAILED,
                summary="not enough samples to create voice baseline",
                sample_count=len(samples),
                profile_count=0,
                aggregate_count=0,
                degraded_count=0,
                failed_count=1,
                profiles=(),
                aggregates=(),
            )
            self._store_report(report)
            return report

        profiles: list[VoiceMicroLatencyProfile] = []
        findings: list[PipelineFinding] = []

        for sample in samples:
            pipeline_report = self._profile_sample(sample)
            profiles.append(
                self._profile_from_report(
                    sample=sample,
                    report=pipeline_report,
                )
            )
            findings.extend(pipeline_report.findings)

        aggregates = self._aggregate_profiles(tuple(profiles))
        degraded_count = sum(
            1 for profile in profiles if self._is_degraded(profile)
        )
        failed_count = sum(1 for profile in profiles if self._is_failed(profile))

        status = self._status_for(
            degraded_count=degraded_count,
            failed_count=failed_count,
        )
        reason = self._reason_for(status)

        report = VoiceBaselineReport(
            status=status,
            reason=reason,
            summary=self._summary_for(status),
            sample_count=len(samples),
            profile_count=len(profiles),
            aggregate_count=len(aggregates),
            degraded_count=degraded_count,
            failed_count=failed_count,
            profiles=tuple(profiles),
            aggregates=aggregates,
            findings=tuple(findings),
        )
        self._store_report(report)

        return report

    def samples(self) -> tuple[VoiceBaselineSample, ...]:
        with self._lock:
            return tuple(self._samples)

    def latest_report(self) -> VoiceBaselineReport | None:
        with self._lock:
            if not self._reports:
                return None

            return self._reports[-1]

    def reports(self) -> tuple[VoiceBaselineReport, ...]:
        with self._lock:
            return tuple(self._reports)

    def snapshot(self) -> VoiceBaselineRuntimeSnapshot:
        with self._lock:
            return VoiceBaselineRuntimeSnapshot(
                name=self.name,
                sample_count=len(self._samples),
                report_count=len(self._reports),
                last_status=self._last_status,
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._reports.clear()
            self._last_status = None
            self._last_reason = VoiceBaselineReason.RUNTIME_RESET

    def _store_report(self, report: VoiceBaselineReport) -> None:
        with self._lock:
            self._reports.append(report)
            self._last_status = report.status
            self._last_reason = report.reason

    def _profile_sample(
        self,
        sample: VoiceBaselineSample,
    ) -> PipelineProfilerReport:
        profiler = self._new_pipeline_profiler()
        profiler.start_trace(
            name=sample.scenario.value,
            trace_id=sample.trace_id,
            metadata=dict(sample.metadata),
        )

        for span in sample.spans:
            profiler.record_span(span)

        return profiler.complete_trace(sample.trace_id)

    def _new_pipeline_profiler(self) -> PipelineLatencyProfiler:
        if self._config.record_to_pipeline_profiler:
            return self._pipeline_profiler

        return PipelineLatencyProfiler(
            config=PipelineProfilerConfig(record_to_latency_runtime=False)
        )

    def _profile_from_report(
        self,
        *,
        sample: VoiceBaselineSample,
        report: PipelineProfilerReport,
    ) -> VoiceMicroLatencyProfile:
        durations = report.stage_durations_ms

        return VoiceMicroLatencyProfile(
            trace_id=sample.trace_id,
            scenario=sample.scenario,
            wake_detection_ms=durations.get(PipelineStage.VAD_DETECTION.value, 0.0),
            microphone_capture_ms=durations.get(
                PipelineStage.MICROPHONE_CAPTURE.value,
                0.0,
            ),
            vad_detection_ms=durations.get(PipelineStage.VAD_DETECTION.value, 0.0),
            stt_first_partial_ms=durations.get(
                PipelineStage.STT_FIRST_PARTIAL.value,
                0.0,
            ),
            stt_finalization_ms=durations.get(
                PipelineStage.STT_FINALIZATION.value,
                0.0,
            ),
            routing_decision_ms=durations.get(
                PipelineStage.INTENT_CLASSIFICATION.value,
                0.0,
            ),
            context_build_ms=durations.get(PipelineStage.CONTEXT_BUILD.value, 0.0),
            memory_retrieval_ms=durations.get(
                PipelineStage.MEMORY_RETRIEVAL.value,
                0.0,
            ),
            llm_first_token_ms=durations.get(
                PipelineStage.LLM_FIRST_TOKEN.value,
                0.0,
            ),
            tts_first_audio_ms=durations.get(
                PipelineStage.TTS_FIRST_AUDIO.value,
                0.0,
            ),
            playback_startup_ms=durations.get(
                PipelineStage.PLAYBACK_STARTUP.value,
                0.0,
            ),
            interrupt_recovery_ms=durations.get(
                PipelineStage.INTERRUPT_RECOVERY.value,
                0.0,
            ),
            first_audio_wall_clock_ms=self._first_audio_wall_clock_ms(sample),
            total_wall_clock_ms=report.wall_clock_ms,
            overlap_ratio=report.overlap_ratio,
            idle_gap_ms=report.idle_gap_ms,
            finding_count=len(report.findings),
        )

    @staticmethod
    def _first_audio_wall_clock_ms(sample: VoiceBaselineSample) -> float:
        first_start = min(span.start_ns for span in sample.spans)
        first_audio_end = max(
            (
                span.end_ns
                for span in sample.spans
                if span.stage
                in {
                    PipelineStage.TTS_FIRST_AUDIO,
                    PipelineStage.PLAYBACK_STARTUP,
                }
            ),
            default=max(span.end_ns for span in sample.spans),
        )

        return (first_audio_end - first_start) / 1_000_000.0

    def _aggregate_profiles(
        self,
        profiles: tuple[VoiceMicroLatencyProfile, ...],
    ) -> tuple[VoiceBaselineAggregate, ...]:
        grouped: dict[
            VoiceBaselineScenario,
            list[VoiceMicroLatencyProfile],
        ] = {}

        for profile in profiles:
            grouped.setdefault(profile.scenario, []).append(profile)

        return tuple(
            self._aggregate_scenario(scenario=scenario, profiles=tuple(items))
            for scenario, items in grouped.items()
        )

    def _aggregate_scenario(
        self,
        *,
        scenario: VoiceBaselineScenario,
        profiles: tuple[VoiceMicroLatencyProfile, ...],
    ) -> VoiceBaselineAggregate:
        first_audio_values = tuple(
            profile.first_audio_wall_clock_ms for profile in profiles
        )
        total_values = tuple(profile.total_wall_clock_ms for profile in profiles)
        overlap_values = tuple(profile.overlap_ratio for profile in profiles)
        idle_values = tuple(profile.idle_gap_ms for profile in profiles)

        return VoiceBaselineAggregate(
            scenario=scenario,
            sample_count=len(profiles),
            first_audio_p50_ms=self._percentile(
                first_audio_values,
                LatencyPercentile.P50,
            ),
            first_audio_p90_ms=self._percentile(
                first_audio_values,
                LatencyPercentile.P90,
            ),
            first_audio_p95_ms=self._percentile(
                first_audio_values,
                LatencyPercentile.P95,
            ),
            first_audio_p99_ms=self._percentile(
                first_audio_values,
                LatencyPercentile.P99,
            ),
            first_audio_worst_ms=max(first_audio_values) if first_audio_values else 0.0,
            total_p50_ms=self._percentile(total_values, LatencyPercentile.P50),
            total_p95_ms=self._percentile(total_values, LatencyPercentile.P95),
            total_worst_ms=max(total_values) if total_values else 0.0,
            average_overlap_ratio=mean(overlap_values) if overlap_values else 0.0,
            average_idle_gap_ms=mean(idle_values) if idle_values else 0.0,
        )

    @staticmethod
    def _percentile(
        values: tuple[float, ...],
        percentile: LatencyPercentile,
    ) -> float:
        ordered = sorted(values)

        if not ordered:
            return 0.0

        if len(ordered) == 1:
            return ordered[0]

        rank = (percentile.value / 100.0) * (len(ordered) - 1)
        lower_index = int(rank)
        upper_index = min(lower_index + 1, len(ordered) - 1)
        fraction = rank - lower_index

        return ordered[lower_index] + (
            ordered[upper_index] - ordered[lower_index]
        ) * fraction

    def _is_degraded(self, profile: VoiceMicroLatencyProfile) -> bool:
        return (
            profile.first_audio_wall_clock_ms
            >= self._config.first_audio_degraded_ms
            or profile.total_wall_clock_ms >= self._config.total_degraded_ms
        )

    def _is_failed(self, profile: VoiceMicroLatencyProfile) -> bool:
        return (
            profile.first_audio_wall_clock_ms >= self._config.first_audio_failed_ms
            or profile.total_wall_clock_ms >= self._config.total_failed_ms
        )

    @staticmethod
    def _status_for(
        *,
        degraded_count: int,
        failed_count: int,
    ) -> VoiceBaselineStatus:
        if failed_count > 0:
            return VoiceBaselineStatus.FAILED

        if degraded_count > 0:
            return VoiceBaselineStatus.DEGRADED

        return VoiceBaselineStatus.PASSED

    @staticmethod
    def _reason_for(status: VoiceBaselineStatus) -> VoiceBaselineReason:
        if status == VoiceBaselineStatus.PASSED:
            return VoiceBaselineReason.BASELINE_CREATED

        if status == VoiceBaselineStatus.DEGRADED:
            return VoiceBaselineReason.BASELINE_DEGRADED

        return VoiceBaselineReason.BASELINE_FAILED

    @staticmethod
    def _summary_for(status: VoiceBaselineStatus) -> str:
        if status == VoiceBaselineStatus.PASSED:
            return "voice pipeline baseline created"

        if status == VoiceBaselineStatus.DEGRADED:
            return "voice pipeline baseline created with degraded samples"

        return "voice pipeline baseline failed"


def build_synthetic_voice_sample(
    *,
    scenario: VoiceBaselineScenario = VoiceBaselineScenario.BASIC_QUESTION,
    trace_id: str | None = None,
    offset_ms: int = 0,
    slow: bool = False,
) -> VoiceBaselineSample:
    """
    Build a deterministic synthetic voice sample.

    This is not fake architecture. This creates repeatable baseline vectors
    before real microphone profiling is wired in.
    """

    actual_trace_id = trace_id or uuid4().hex
    multiplier = 3 if slow else 1

    def stage(
        pipeline_stage: PipelineStage,
        start_ms: int,
        end_ms: int,
        operation: LatencyOperation,
        subsystem: LatencySubsystem,
    ) -> PipelineSpan:
        return PipelineSpan(
            trace_id=actual_trace_id,
            stage=pipeline_stage,
            operation=operation,
            subsystem=subsystem,
            start_ns=(offset_ms + start_ms) * 1_000_000,
            end_ns=(offset_ms + end_ms * multiplier) * 1_000_000,
        )

    return VoiceBaselineSample(
        scenario=scenario,
        trace_id=actual_trace_id,
        spans=(
            stage(
                PipelineStage.MICROPHONE_CAPTURE,
                0,
                40,
                LatencyOperation.STT_FIRST_TOKEN,
                LatencySubsystem.PRESENCE,
            ),
            stage(
                PipelineStage.VAD_DETECTION,
                20,
                60,
                LatencyOperation.STT_FIRST_TOKEN,
                LatencySubsystem.PRESENCE,
            ),
            stage(
                PipelineStage.STT_FIRST_PARTIAL,
                50,
                160,
                LatencyOperation.STT_FIRST_TOKEN,
                LatencySubsystem.PRESENCE,
            ),
            stage(
                PipelineStage.STT_FINALIZATION,
                160,
                310,
                LatencyOperation.STT_FINALIZATION,
                LatencySubsystem.PRESENCE,
            ),
            stage(
                PipelineStage.INTENT_CLASSIFICATION,
                120,
                180,
                LatencyOperation.CONTEXT_BUILD,
                LatencySubsystem.COGNITION,
            ),
            stage(
                PipelineStage.MEMORY_RETRIEVAL,
                130,
                260,
                LatencyOperation.MEMORY_RETRIEVAL,
                LatencySubsystem.MEMORY,
            ),
            stage(
                PipelineStage.CONTEXT_BUILD,
                220,
                300,
                LatencyOperation.CONTEXT_BUILD,
                LatencySubsystem.COGNITION,
            ),
            stage(
                PipelineStage.LLM_FIRST_TOKEN,
                300,
                560,
                LatencyOperation.LLM_FIRST_TOKEN,
                LatencySubsystem.COGNITION,
            ),
            stage(
                PipelineStage.TTS_FIRST_AUDIO,
                540,
                660,
                LatencyOperation.TTS_FIRST_AUDIO,
                LatencySubsystem.PRESENCE,
            ),
            stage(
                PipelineStage.PLAYBACK_STARTUP,
                650,
                710,
                LatencyOperation.PLAYBACK_STARTUP,
                LatencySubsystem.PLAYBACK,
            ),
        ),
    )