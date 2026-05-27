from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.latency.profiler import (
    PipelineProfilerReport,
    PipelineStage,
)
from jarvis.latency.voice_baseline import (
    VoiceBaselineReport,
    VoiceMicroLatencyProfile,
)
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class LatencyBudgetSliceKind(StrEnum):
    """
    First-word latency budget slices.

    These are the human-perceived stages that decide whether JARVIS feels
    fast and fluid.
    """

    AUDIO_CAPTURE_VAD = "audio_capture_vad"
    STT_FIRST_PARTIAL = "stt_first_partial"
    CONTEXT_BUILD = "context_build"
    MEMORY_RETRIEVAL = "memory_retrieval"
    LLM_FIRST_TOKEN = "llm_first_token"
    TTS_FIRST_CHUNK = "tts_first_chunk"
    AUDIO_OUTPUT = "audio_output"


class LatencyBudgetAllocationStatus(StrEnum):
    """
    Status for one budget allocation evaluation.
    """

    PASSED = "passed"
    WARNING = "warning"
    VIOLATION = "violation"
    CRITICAL = "critical"


class LatencyBudgetAllocationReason(StrEnum):
    """
    Machine-readable budget allocation reasons.
    """

    PLAN_CREATED = "plan_created"
    PROFILE_EVALUATED = "profile_evaluated"
    PIPELINE_EVALUATED = "pipeline_evaluated"
    VOICE_BASELINE_EVALUATED = "voice_baseline_evaluated"
    BUDGET_PASSED = "budget_passed"
    BUDGET_WARNING = "budget_warning"
    BUDGET_VIOLATION = "budget_violation"
    BUDGET_CRITICAL = "budget_critical"
    INVALID_BUDGET_PLAN = "invalid_budget_plan"
    RUNTIME_RESET = "runtime_reset"


class LatencyBudgetSlice(OrchestrationModel):
    """
    One allocated slice of the first-word latency budget.

    Example:
    LLM_FIRST_TOKEN gets 300ms of the total 800ms first-word budget.
    """

    slice_id: str = Field(default_factory=lambda: uuid4().hex)
    kind: LatencyBudgetSliceKind
    name: str
    budget_ms: float = Field(gt=0)
    percent_of_total: float = Field(ge=0, le=100)
    stages: tuple[PipelineStage, ...]
    owner: str
    hard_budget: bool = True
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("slice_id", "name", "owner")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @model_validator(mode="after")
    def _validate_stages(self) -> LatencyBudgetSlice:
        if not self.stages:
            raise ValueError("budget slice must include at least one stage.")

        return self


class LatencyBudgetPlan(OrchestrationModel):
    """
    Complete first-word latency budget plan.

    This is the stage-level contract that keeps JARVIS fast and fluid.
    """

    plan_id: str = Field(default_factory=lambda: uuid4().hex)
    name: str = "phase7_first_word_budget"
    total_budget_ms: float = Field(gt=0)
    slices: tuple[LatencyBudgetSlice, ...]
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("plan_id", "name")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @model_validator(mode="after")
    def _validate_budget_sum(self) -> LatencyBudgetPlan:
        if not self.slices:
            raise ValueError("latency budget plan must include slices.")

        total = sum(item.budget_ms for item in self.slices)

        if abs(total - self.total_budget_ms) > 0.001:
            raise ValueError("budget slices must sum to total_budget_ms.")

        percent_total = sum(item.percent_of_total for item in self.slices)

        if abs(percent_total - 100.0) > 0.5:
            raise ValueError("budget slice percentages must sum to 100.")

        kinds = [item.kind for item in self.slices]

        if len(set(kinds)) != len(kinds):
            raise ValueError("budget slice kinds must be unique.")

        return self

    def slice_for(
        self,
        kind: LatencyBudgetSliceKind,
    ) -> LatencyBudgetSlice | None:
        for item in self.slices:
            if item.kind == kind:
                return item

        return None


class LatencyBudgetSliceEvaluation(OrchestrationModel):
    """
    Evaluation of one budget slice against measured latency.
    """

    slice_kind: LatencyBudgetSliceKind
    slice_name: str
    owner: str
    budget_ms: float = Field(gt=0)
    actual_ms: float = Field(ge=0)
    remaining_ms: float
    percent_used: float = Field(ge=0)
    status: LatencyBudgetAllocationStatus
    reason: LatencyBudgetAllocationReason
    message: str
    stages: tuple[PipelineStage, ...]
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("slice_name", "owner", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @property
    def passed(self) -> bool:
        return self.status == LatencyBudgetAllocationStatus.PASSED


class LatencyBudgetAllocationReport(OrchestrationModel):
    """
    Full first-word budget allocation evaluation report.
    """

    report_id: str = Field(default_factory=lambda: uuid4().hex)
    plan_id: str
    status: LatencyBudgetAllocationStatus
    reason: LatencyBudgetAllocationReason
    summary: str
    total_budget_ms: float = Field(gt=0)
    total_actual_ms: float = Field(ge=0)
    total_remaining_ms: float
    total_percent_used: float = Field(ge=0)
    evaluation_count: int = Field(ge=0)
    passed_count: int = Field(ge=0)
    warning_count: int = Field(ge=0)
    violation_count: int = Field(ge=0)
    critical_count: int = Field(ge=0)
    evaluations: tuple[LatencyBudgetSliceEvaluation, ...]
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("report_id", "plan_id", "summary")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @property
    def passed(self) -> bool:
        return self.status == LatencyBudgetAllocationStatus.PASSED


@dataclass(frozen=True, slots=True)
class LatencyBudgetAllocatorConfig:
    """
    Budget allocator configuration.

    warning_threshold_ratio:
        Warn when a slice uses this much of its budget.

    critical_threshold_ratio:
        Critical when a slice exceeds this multiple of its budget.
    """

    name: str = "phase7_latency_budget_allocator"
    warning_threshold_ratio: float = 0.90
    critical_threshold_ratio: float = 1.50

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if not 0 < self.warning_threshold_ratio <= 1:
            raise ValueError("warning_threshold_ratio must be within 0..1.")

        if self.critical_threshold_ratio <= 1:
            raise ValueError("critical_threshold_ratio must be greater than 1.")


@dataclass(frozen=True, slots=True)
class LatencyBudgetAllocatorSnapshot:
    """
    Runtime diagnostics for Step 3 budget allocation.
    """

    name: str
    report_count: int
    last_status: LatencyBudgetAllocationStatus | None
    last_reason: LatencyBudgetAllocationReason | None
    last_total_percent_used: float | None


class LatencyBudgetAllocator:
    """
    Phase 7 Step 3 Latency Budget Allocator.

    Responsibilities:
    - allocate first-word latency budget across pipeline stages
    - evaluate voice baseline profiles against budget slices
    - identify which stage silently destroys fluidity
    - provide optimization isolation evidence

    Non-responsibilities:
    - no optimization
    - no streaming changes
    - no scheduling mutation
    - no speculative execution
    """

    def __init__(
        self,
        *,
        config: LatencyBudgetAllocatorConfig | None = None,
        plan: LatencyBudgetPlan | None = None,
    ) -> None:
        self._config = config or LatencyBudgetAllocatorConfig()
        self._config.validate()

        self._plan = plan or default_first_word_latency_budget_plan()
        self._reports: list[LatencyBudgetAllocationReport] = []
        self._lock = RLock()
        self._last_status: LatencyBudgetAllocationStatus | None = None
        self._last_reason: LatencyBudgetAllocationReason | None = None

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def plan(self) -> LatencyBudgetPlan:
        return self._plan

    def evaluate_profile(
        self,
        profile: VoiceMicroLatencyProfile,
    ) -> LatencyBudgetAllocationReport:
        actuals = {
            LatencyBudgetSliceKind.AUDIO_CAPTURE_VAD: (
                profile.microphone_capture_ms + profile.vad_detection_ms
            ),
            LatencyBudgetSliceKind.STT_FIRST_PARTIAL: (
                profile.stt_first_partial_ms
            ),
            LatencyBudgetSliceKind.CONTEXT_BUILD: profile.context_build_ms,
            LatencyBudgetSliceKind.MEMORY_RETRIEVAL: profile.memory_retrieval_ms,
            LatencyBudgetSliceKind.LLM_FIRST_TOKEN: profile.llm_first_token_ms,
            LatencyBudgetSliceKind.TTS_FIRST_CHUNK: profile.tts_first_audio_ms,
            LatencyBudgetSliceKind.AUDIO_OUTPUT: profile.playback_startup_ms,
        }

        report = self._evaluate_actuals(
            actuals=actuals,
            metadata={
                "source": "voice_micro_latency_profile",
                "trace_id": profile.trace_id,
                "scenario": profile.scenario.value,
                "first_audio_wall_clock_ms": profile.first_audio_wall_clock_ms,
                "total_wall_clock_ms": profile.total_wall_clock_ms,
            },
        )
        self._store_report(report)

        return report

    def evaluate_pipeline_report(
        self,
        report: PipelineProfilerReport,
    ) -> LatencyBudgetAllocationReport:
        durations = report.stage_durations_ms
        actuals = {
            LatencyBudgetSliceKind.AUDIO_CAPTURE_VAD: (
                durations.get(PipelineStage.MICROPHONE_CAPTURE.value, 0.0)
                + durations.get(PipelineStage.VAD_DETECTION.value, 0.0)
            ),
            LatencyBudgetSliceKind.STT_FIRST_PARTIAL: durations.get(
                PipelineStage.STT_FIRST_PARTIAL.value,
                0.0,
            ),
            LatencyBudgetSliceKind.CONTEXT_BUILD: durations.get(
                PipelineStage.CONTEXT_BUILD.value,
                0.0,
            ),
            LatencyBudgetSliceKind.MEMORY_RETRIEVAL: durations.get(
                PipelineStage.MEMORY_RETRIEVAL.value,
                0.0,
            ),
            LatencyBudgetSliceKind.LLM_FIRST_TOKEN: durations.get(
                PipelineStage.LLM_FIRST_TOKEN.value,
                0.0,
            ),
            LatencyBudgetSliceKind.TTS_FIRST_CHUNK: durations.get(
                PipelineStage.TTS_FIRST_AUDIO.value,
                0.0,
            ),
            LatencyBudgetSliceKind.AUDIO_OUTPUT: durations.get(
                PipelineStage.PLAYBACK_STARTUP.value,
                0.0,
            ),
        }

        allocation_report = self._evaluate_actuals(
            actuals=actuals,
            metadata={
                "source": "pipeline_profiler_report",
                "trace_id": report.trace_id,
                "trace_name": report.trace_name,
                "wall_clock_ms": report.wall_clock_ms,
                "overlap_ratio": report.overlap_ratio,
            },
        )
        self._store_report(allocation_report)

        return allocation_report

    def evaluate_voice_baseline_report(
        self,
        baseline: VoiceBaselineReport,
    ) -> tuple[LatencyBudgetAllocationReport, ...]:
        reports = tuple(self.evaluate_profile(profile) for profile in baseline.profiles)

        with self._lock:
            self._last_reason = (
                LatencyBudgetAllocationReason.VOICE_BASELINE_EVALUATED
            )

        return reports

    def latest_report(self) -> LatencyBudgetAllocationReport | None:
        with self._lock:
            if not self._reports:
                return None

            return self._reports[-1]

    def reports(self) -> tuple[LatencyBudgetAllocationReport, ...]:
        with self._lock:
            return tuple(self._reports)

    def snapshot(self) -> LatencyBudgetAllocatorSnapshot:
        with self._lock:
            latest = self._reports[-1] if self._reports else None

            return LatencyBudgetAllocatorSnapshot(
                name=self.name,
                report_count=len(self._reports),
                last_status=self._last_status,
                last_reason=self._last_reason,
                last_total_percent_used=(
                    latest.total_percent_used if latest is not None else None
                ),
            )

    def reset(self) -> None:
        with self._lock:
            self._reports.clear()
            self._last_status = None
            self._last_reason = LatencyBudgetAllocationReason.RUNTIME_RESET

    def _store_report(self, report: LatencyBudgetAllocationReport) -> None:
        with self._lock:
            self._reports.append(report)
            self._last_status = report.status
            self._last_reason = report.reason

    def _evaluate_actuals(
        self,
        *,
        actuals: dict[LatencyBudgetSliceKind, float],
        metadata: dict[str, object],
    ) -> LatencyBudgetAllocationReport:
        evaluations = tuple(
            self._evaluate_slice(
                budget_slice=budget_slice,
                actual_ms=actuals.get(budget_slice.kind, 0.0),
            )
            for budget_slice in self._plan.slices
        )

        total_actual = sum(item.actual_ms for item in evaluations)
        total_remaining = self._plan.total_budget_ms - total_actual
        total_percent = (
            total_actual / self._plan.total_budget_ms
        ) * 100.0

        passed_count = sum(
            1
            for item in evaluations
            if item.status == LatencyBudgetAllocationStatus.PASSED
        )
        warning_count = sum(
            1
            for item in evaluations
            if item.status == LatencyBudgetAllocationStatus.WARNING
        )
        violation_count = sum(
            1
            for item in evaluations
            if item.status == LatencyBudgetAllocationStatus.VIOLATION
        )
        critical_count = sum(
            1
            for item in evaluations
            if item.status == LatencyBudgetAllocationStatus.CRITICAL
        )

        status = self._status_for(
            warning_count=warning_count,
            violation_count=violation_count,
            critical_count=critical_count,
            total_actual_ms=total_actual,
        )
        reason = self._reason_for(status)

        return LatencyBudgetAllocationReport(
            plan_id=self._plan.plan_id,
            status=status,
            reason=reason,
            summary=self._summary_for(status),
            total_budget_ms=self._plan.total_budget_ms,
            total_actual_ms=total_actual,
            total_remaining_ms=total_remaining,
            total_percent_used=total_percent,
            evaluation_count=len(evaluations),
            passed_count=passed_count,
            warning_count=warning_count,
            violation_count=violation_count,
            critical_count=critical_count,
            evaluations=evaluations,
            metadata=metadata,
        )

    def _evaluate_slice(
        self,
        *,
        budget_slice: LatencyBudgetSlice,
        actual_ms: float,
    ) -> LatencyBudgetSliceEvaluation:
        percent_used = (actual_ms / budget_slice.budget_ms) * 100.0
        remaining = budget_slice.budget_ms - actual_ms
        status = self._slice_status(
            actual_ms=actual_ms,
            budget_ms=budget_slice.budget_ms,
        )
        reason = self._reason_for(status)

        return LatencyBudgetSliceEvaluation(
            slice_kind=budget_slice.kind,
            slice_name=budget_slice.name,
            owner=budget_slice.owner,
            budget_ms=budget_slice.budget_ms,
            actual_ms=actual_ms,
            remaining_ms=remaining,
            percent_used=percent_used,
            status=status,
            reason=reason,
            message=self._slice_message(status),
            stages=budget_slice.stages,
        )

    def _slice_status(
        self,
        *,
        actual_ms: float,
        budget_ms: float,
    ) -> LatencyBudgetAllocationStatus:
        if actual_ms > budget_ms * self._config.critical_threshold_ratio:
            return LatencyBudgetAllocationStatus.CRITICAL

        if actual_ms > budget_ms:
            return LatencyBudgetAllocationStatus.VIOLATION

        if actual_ms >= budget_ms * self._config.warning_threshold_ratio:
            return LatencyBudgetAllocationStatus.WARNING

        return LatencyBudgetAllocationStatus.PASSED

    def _status_for(
        self,
        *,
        warning_count: int,
        violation_count: int,
        critical_count: int,
        total_actual_ms: float,
    ) -> LatencyBudgetAllocationStatus:
        if critical_count > 0 or total_actual_ms > (
            self._plan.total_budget_ms * self._config.critical_threshold_ratio
        ):
            return LatencyBudgetAllocationStatus.CRITICAL

        if violation_count > 0 or total_actual_ms > self._plan.total_budget_ms:
            return LatencyBudgetAllocationStatus.VIOLATION

        if warning_count > 0:
            return LatencyBudgetAllocationStatus.WARNING

        return LatencyBudgetAllocationStatus.PASSED

    @staticmethod
    def _reason_for(
        status: LatencyBudgetAllocationStatus,
    ) -> LatencyBudgetAllocationReason:
        if status == LatencyBudgetAllocationStatus.PASSED:
            return LatencyBudgetAllocationReason.BUDGET_PASSED

        if status == LatencyBudgetAllocationStatus.WARNING:
            return LatencyBudgetAllocationReason.BUDGET_WARNING

        if status == LatencyBudgetAllocationStatus.VIOLATION:
            return LatencyBudgetAllocationReason.BUDGET_VIOLATION

        return LatencyBudgetAllocationReason.BUDGET_CRITICAL

    @staticmethod
    def _summary_for(status: LatencyBudgetAllocationStatus) -> str:
        if status == LatencyBudgetAllocationStatus.PASSED:
            return "first-word latency budget passed"

        if status == LatencyBudgetAllocationStatus.WARNING:
            return "first-word latency budget is close to limit"

        if status == LatencyBudgetAllocationStatus.VIOLATION:
            return "first-word latency budget was exceeded"

        return "first-word latency budget is critically exceeded"

    @staticmethod
    def _slice_message(status: LatencyBudgetAllocationStatus) -> str:
        if status == LatencyBudgetAllocationStatus.PASSED:
            return "stage is within latency budget"

        if status == LatencyBudgetAllocationStatus.WARNING:
            return "stage is close to latency budget"

        if status == LatencyBudgetAllocationStatus.VIOLATION:
            return "stage exceeded latency budget and must be optimized"

        return "stage critically exceeded latency budget"


def default_first_word_latency_budget_plan() -> LatencyBudgetPlan:
    """
    Default Phase 7 first-word latency allocation.

    Total budget: 800ms.
    """

    return LatencyBudgetPlan(
        total_budget_ms=800.0,
        slices=(
            LatencyBudgetSlice(
                kind=LatencyBudgetSliceKind.AUDIO_CAPTURE_VAD,
                name="Audio capture + VAD",
                budget_ms=50.0,
                percent_of_total=6.25,
                owner="presence_runtime",
                stages=(
                    PipelineStage.MICROPHONE_CAPTURE,
                    PipelineStage.VAD_DETECTION,
                ),
            ),
            LatencyBudgetSlice(
                kind=LatencyBudgetSliceKind.STT_FIRST_PARTIAL,
                name="STT first partial",
                budget_ms=150.0,
                percent_of_total=18.75,
                owner="presence_runtime",
                stages=(PipelineStage.STT_FIRST_PARTIAL,),
            ),
            LatencyBudgetSlice(
                kind=LatencyBudgetSliceKind.CONTEXT_BUILD,
                name="Context build",
                budget_ms=80.0,
                percent_of_total=10.0,
                owner="cognition_runtime",
                stages=(PipelineStage.CONTEXT_BUILD,),
            ),
            LatencyBudgetSlice(
                kind=LatencyBudgetSliceKind.MEMORY_RETRIEVAL,
                name="Memory retrieval",
                budget_ms=100.0,
                percent_of_total=12.5,
                owner="memory_runtime",
                stages=(PipelineStage.MEMORY_RETRIEVAL,),
            ),
            LatencyBudgetSlice(
                kind=LatencyBudgetSliceKind.LLM_FIRST_TOKEN,
                name="LLM first token",
                budget_ms=300.0,
                percent_of_total=37.5,
                owner="cognition_runtime",
                stages=(PipelineStage.LLM_FIRST_TOKEN,),
            ),
            LatencyBudgetSlice(
                kind=LatencyBudgetSliceKind.TTS_FIRST_CHUNK,
                name="TTS first chunk",
                budget_ms=80.0,
                percent_of_total=10.0,
                owner="presence_runtime",
                stages=(PipelineStage.TTS_FIRST_AUDIO,),
            ),
            LatencyBudgetSlice(
                kind=LatencyBudgetSliceKind.AUDIO_OUTPUT,
                name="Audio output",
                budget_ms=40.0,
                percent_of_total=5.0,
                owner="playback_runtime",
                stages=(PipelineStage.PLAYBACK_STARTUP,),
            ),
        ),
        metadata={
            "phase": "phase7",
            "step": "step3",
            "law": "if every stage is within budget, experience feels instant",
        },
    )