from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from threading import RLock
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.latency.budget_allocation import (
    LatencyBudgetAllocationReport,
    LatencyBudgetAllocationStatus,
    LatencyBudgetSliceKind,
)
from jarvis.latency.profiler import (
    PipelineFindingKind,
    PipelineProfilerReport,
    PipelineStage,
)
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class StreamingAuditFlow(StrEnum):
    """
    Critical Phase 7 data flows audited before Streaming Core begins.
    """

    STT_PARTIALS = "stt_partials"
    TURN_FINALIZATION = "turn_finalization"
    MEMORY_RETRIEVAL = "memory_retrieval"
    CONTEXT_BUILD = "context_build"
    LLM_TOKEN_OUTPUT = "llm_token_output"
    TTS_SYNTHESIS = "tts_synthesis"
    AUDIO_PLAYBACK = "audio_playback"
    TOOL_FEEDBACK = "tool_feedback"
    ACTION_PROGRESS = "action_progress"
    INTERRUPT_RECOVERY = "interrupt_recovery"
    SCHEDULER_DISPATCH = "scheduler_dispatch"
    CACHE_LOOKUP = "cache_lookup"


class StreamingReadiness(StrEnum):
    """
    Streaming readiness classification.

    STREAMING:
        Already supports incremental output without blocking critical path.

    PARTIAL:
        Some incremental behavior exists but blocking waits remain.

    BLOCKING:
        Waits for full completion before downstream work can begin.
    """

    STREAMING = "streaming"
    PARTIAL = "partial"
    BLOCKING = "blocking"


class StreamingDebtSeverity(StrEnum):
    """
    Streaming architecture debt severity.
    """

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class StreamingAuditReason(StrEnum):
    """
    Machine-readable audit reasons.
    """

    AUDIT_CREATED = "audit_created"
    AUDIT_PASSED = "audit_passed"
    AUDIT_PARTIAL = "audit_partial"
    AUDIT_BLOCKED = "audit_blocked"
    FLOW_REGISTERED = "flow_registered"
    FLOW_EVALUATED = "flow_evaluated"
    BUDGET_REPORT_EVALUATED = "budget_report_evaluated"
    PROFILER_REPORT_EVALUATED = "profiler_report_evaluated"
    RUNTIME_RESET = "runtime_reset"


class CriticalPathRank(IntEnum):
    """
    Critical-path priority.

    Lower rank means earlier in the user-perceived path.
    Blocking flows must be fixed in this order.
    """

    AUDIO_CAPTURE = 10
    STT = 20
    TURN_FINALIZATION = 30
    MEMORY_CONTEXT = 40
    LLM = 50
    TTS = 60
    PLAYBACK = 70
    INTERRUPT = 80
    TOOLING = 90
    BACKGROUND = 100


class StreamingAuditFlowSpec(OrchestrationModel):
    """
    Expected streaming behavior for one flow.
    """

    flow_id: str = Field(default_factory=lambda: uuid4().hex)
    flow: StreamingAuditFlow
    name: str
    owner: str
    critical_path_rank: CriticalPathRank
    expected_behavior: str
    blocking_failure: str
    required_for_step: int
    stages: tuple[PipelineStage, ...] = ()
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator(
    "flow_id",
    "name",
    "owner",
    "expected_behavior",
    "blocking_failure",
    )
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @model_validator(mode="after")
    def _validate_required_step(self) -> StreamingAuditFlowSpec:
        if self.required_for_step < 5:
            raise ValueError("streaming audit flows must target Step 5 or later.")

        return self


class StreamingAuditFinding(OrchestrationModel):
    """
    One finding from the streaming architecture audit.
    """

    finding_id: str = Field(default_factory=lambda: uuid4().hex)
    flow: StreamingAuditFlow
    readiness: StreamingReadiness
    severity: StreamingDebtSeverity
    critical_path_rank: CriticalPathRank
    message: str
    evidence: dict[str, object] = Field(default_factory=dict)
    created_at: object = Field(default_factory=utc_now)

    @field_validator("finding_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @property
    def is_debt(self) -> bool:
        return self.readiness in {
            StreamingReadiness.BLOCKING,
            StreamingReadiness.PARTIAL,
        }


class StreamingAuditFlowEvaluation(OrchestrationModel):
    """
    Evaluation result for one critical data flow.
    """

    flow: StreamingAuditFlow
    name: str
    owner: str
    readiness: StreamingReadiness
    severity: StreamingDebtSeverity
    critical_path_rank: CriticalPathRank
    required_for_step: int
    message: str
    findings: tuple[StreamingAuditFinding, ...] = ()
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("name", "owner", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @property
    def passed(self) -> bool:
        return self.readiness == StreamingReadiness.STREAMING


class StreamingArchitectureAuditReport(OrchestrationModel):
    """
    Full Phase 7 Step 4 streaming architecture audit report.
    """

    report_id: str = Field(default_factory=lambda: uuid4().hex)
    status: StreamingReadiness
    reason: StreamingAuditReason
    summary: str
    flow_count: int = Field(ge=0)
    streaming_count: int = Field(ge=0)
    partial_count: int = Field(ge=0)
    blocking_count: int = Field(ge=0)
    debt_count: int = Field(ge=0)
    critical_debt_count: int = Field(ge=0)
    evaluations: tuple[StreamingAuditFlowEvaluation, ...]
    fix_order: tuple[StreamingAuditFlow, ...]
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("report_id", "summary")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @property
    def passed(self) -> bool:
        return self.status == StreamingReadiness.STREAMING


@dataclass(frozen=True, slots=True)
class StreamingArchitectureAuditConfig:
    """
    Streaming architecture audit configuration.
    """

    name: str = "phase7_streaming_architecture_audit"
    fail_on_blocking: bool = False

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")


@dataclass(frozen=True, slots=True)
class StreamingArchitectureAuditSnapshot:
    """
    Runtime diagnostics for Step 4.
    """

    name: str
    report_count: int
    last_status: StreamingReadiness | None
    last_reason: StreamingAuditReason | None
    last_debt_count: int | None
    last_critical_debt_count: int | None


class StreamingArchitectureAuditRuntime:
    """
    Phase 7 Step 4 Streaming Architecture Audit Runtime.

    Responsibilities:
    - audit Phases 1-6 critical data flows
    - classify flows as STREAMING / PARTIAL / BLOCKING
    - mark every synchronous wait as latency debt
    - order debts by critical-path rank
    - feed Streaming Core priorities for Steps 5-16

    Non-responsibilities:
    - no optimization
    - no pipeline rewrites
    - no scheduler mutation
    - no speculative execution
    """

    def __init__(
        self,
        *,
        config: StreamingArchitectureAuditConfig | None = None,
        flow_specs: tuple[StreamingAuditFlowSpec, ...] | None = None,
    ) -> None:
        self._config = config or StreamingArchitectureAuditConfig()
        self._config.validate()

        self._flow_specs = flow_specs or default_streaming_audit_flow_specs()
        self._reports: list[StreamingArchitectureAuditReport] = []
        self._lock = RLock()
        self._last_status: StreamingReadiness | None = None
        self._last_reason: StreamingAuditReason | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def audit_declared_flows(
        self,
        declared_readiness: dict[StreamingAuditFlow, StreamingReadiness],
    ) -> StreamingArchitectureAuditReport:
        evaluations = tuple(
            self._evaluate_declared_flow(
                spec=spec,
                readiness=declared_readiness.get(
                    spec.flow,
                    StreamingReadiness.BLOCKING,
                ),
            )
            for spec in self._flow_specs
        )
        report = self._build_report(
            evaluations=evaluations,
            reason=StreamingAuditReason.AUDIT_CREATED,
            metadata={"source": "declared_readiness"},
        )
        self._store_report(report)

        return report

    def audit_pipeline_report(
        self,
        report: PipelineProfilerReport,
    ) -> StreamingArchitectureAuditReport:
        readiness = self._infer_readiness_from_pipeline(report)
        audit = self.audit_declared_flows(readiness)

        with self._lock:
            self._last_reason = StreamingAuditReason.PROFILER_REPORT_EVALUATED

        return audit.model_copy(
            update={
                "reason": StreamingAuditReason.PROFILER_REPORT_EVALUATED,
                "metadata": {
                    **audit.metadata,
                    "trace_id": report.trace_id,
                    "trace_name": report.trace_name,
                    "source": "pipeline_profiler_report",
                },
            }
        )

    def audit_budget_report(
        self,
        report: LatencyBudgetAllocationReport,
    ) -> StreamingArchitectureAuditReport:
        readiness = self._infer_readiness_from_budget(report)
        audit = self.audit_declared_flows(readiness)

        with self._lock:
            self._last_reason = StreamingAuditReason.BUDGET_REPORT_EVALUATED

        return audit.model_copy(
            update={
                "reason": StreamingAuditReason.BUDGET_REPORT_EVALUATED,
                "metadata": {
                    **audit.metadata,
                    "budget_report_id": report.report_id,
                    "source": "latency_budget_allocation_report",
                },
            }
        )

    def latest_report(self) -> StreamingArchitectureAuditReport | None:
        with self._lock:
            if not self._reports:
                return None

            return self._reports[-1]

    def reports(self) -> tuple[StreamingArchitectureAuditReport, ...]:
        with self._lock:
            return tuple(self._reports)

    def snapshot(self) -> StreamingArchitectureAuditSnapshot:
        with self._lock:
            latest = self._reports[-1] if self._reports else None

            return StreamingArchitectureAuditSnapshot(
                name=self.name,
                report_count=len(self._reports),
                last_status=self._last_status,
                last_reason=self._last_reason,
                last_debt_count=latest.debt_count if latest is not None else None,
                last_critical_debt_count=(
                    latest.critical_debt_count if latest is not None else None
                ),
            )

    def reset(self) -> None:
        with self._lock:
            self._reports.clear()
            self._last_status = None
            self._last_reason = StreamingAuditReason.RUNTIME_RESET

    def _store_report(self, report: StreamingArchitectureAuditReport) -> None:
        with self._lock:
            self._reports.append(report)
            self._last_status = report.status
            self._last_reason = report.reason

    def _evaluate_declared_flow(
        self,
        *,
        spec: StreamingAuditFlowSpec,
        readiness: StreamingReadiness,
    ) -> StreamingAuditFlowEvaluation:
        severity = self._severity_for(
            readiness=readiness,
            rank=spec.critical_path_rank,
        )
        findings = self._findings_for(
            spec=spec,
            readiness=readiness,
            severity=severity,
        )

        return StreamingAuditFlowEvaluation(
            flow=spec.flow,
            name=spec.name,
            owner=spec.owner,
            readiness=readiness,
            severity=severity,
            critical_path_rank=spec.critical_path_rank,
            required_for_step=spec.required_for_step,
            message=self._message_for(readiness),
            findings=findings,
            metadata={
                "expected_behavior": spec.expected_behavior,
                "blocking_failure": spec.blocking_failure,
                "stages": [stage.value for stage in spec.stages],
            },
        )

    def _build_report(
        self,
        *,
        evaluations: tuple[StreamingAuditFlowEvaluation, ...],
        reason: StreamingAuditReason,
        metadata: dict[str, object],
    ) -> StreamingArchitectureAuditReport:
        streaming_count = sum(
            1 for item in evaluations if item.readiness == StreamingReadiness.STREAMING
        )
        partial_count = sum(
            1 for item in evaluations if item.readiness == StreamingReadiness.PARTIAL
        )
        blocking_count = sum(
            1 for item in evaluations if item.readiness == StreamingReadiness.BLOCKING
        )
        debt_count = partial_count + blocking_count
        critical_debt_count = sum(
            1 for item in evaluations if item.severity == StreamingDebtSeverity.CRITICAL
        )
        status = self._report_status(
            streaming_count=streaming_count,
            partial_count=partial_count,
            blocking_count=blocking_count,
        )
        fix_order = tuple(
            item.flow
            for item in sorted(
                (item for item in evaluations if not item.passed),
                key=lambda item: (int(item.critical_path_rank), item.required_for_step),
            )
        )

        return StreamingArchitectureAuditReport(
            status=status,
            reason=reason,
            summary=self._summary_for(status),
            flow_count=len(evaluations),
            streaming_count=streaming_count,
            partial_count=partial_count,
            blocking_count=blocking_count,
            debt_count=debt_count,
            critical_debt_count=critical_debt_count,
            evaluations=evaluations,
            fix_order=fix_order,
            metadata=metadata,
        )

    @staticmethod
    def _report_status(
        *,
        streaming_count: int,
        partial_count: int,
        blocking_count: int,
    ) -> StreamingReadiness:
        if blocking_count > 0:
            return StreamingReadiness.BLOCKING

        if partial_count > 0:
            return StreamingReadiness.PARTIAL

        if streaming_count > 0:
            return StreamingReadiness.STREAMING

        return StreamingReadiness.BLOCKING

    @staticmethod
    def _summary_for(status: StreamingReadiness) -> str:
        if status == StreamingReadiness.STREAMING:
            return "streaming architecture audit passed"

        if status == StreamingReadiness.PARTIAL:
            return "streaming architecture audit found partial streaming debt"

        return "streaming architecture audit found blocking latency debt"

    @staticmethod
    def _message_for(readiness: StreamingReadiness) -> str:
        if readiness == StreamingReadiness.STREAMING:
            return "flow is streaming-ready"

        if readiness == StreamingReadiness.PARTIAL:
            return "flow has partial streaming but still contains blocking waits"

        return "flow blocks downstream pipeline and must be audited before optimization"

    @staticmethod
    def _severity_for(
        *,
        readiness: StreamingReadiness,
        rank: CriticalPathRank,
    ) -> StreamingDebtSeverity:
        if readiness == StreamingReadiness.STREAMING:
            return StreamingDebtSeverity.NONE

        if readiness == StreamingReadiness.PARTIAL:
            if rank <= CriticalPathRank.LLM:
                return StreamingDebtSeverity.HIGH

            return StreamingDebtSeverity.MEDIUM

        if rank <= CriticalPathRank.LLM:
            return StreamingDebtSeverity.CRITICAL

        if rank <= CriticalPathRank.PLAYBACK:
            return StreamingDebtSeverity.HIGH

        return StreamingDebtSeverity.MEDIUM

    @staticmethod
    def _findings_for(
        *,
        spec: StreamingAuditFlowSpec,
        readiness: StreamingReadiness,
        severity: StreamingDebtSeverity,
    ) -> tuple[StreamingAuditFinding, ...]:
        if readiness == StreamingReadiness.STREAMING:
            return ()

        return (
            StreamingAuditFinding(
                flow=spec.flow,
                readiness=readiness,
                severity=severity,
                critical_path_rank=spec.critical_path_rank,
                message=spec.blocking_failure,
                evidence={
                    "owner": spec.owner,
                    "required_for_step": spec.required_for_step,
                    "expected_behavior": spec.expected_behavior,
                },
            ),
        )

    @staticmethod
    def _infer_readiness_from_pipeline(
        report: PipelineProfilerReport,
    ) -> dict[StreamingAuditFlow, StreamingReadiness]:
        readiness = _default_partial_readiness()

        stage_names = set(report.stage_durations_ms)
        finding_kinds = {finding.kind for finding in report.findings}

        if PipelineStage.STT_FIRST_PARTIAL.value in stage_names:
            readiness[StreamingAuditFlow.STT_PARTIALS] = StreamingReadiness.STREAMING

        if PipelineStage.MEMORY_RETRIEVAL.value in stage_names:
            readiness[StreamingAuditFlow.MEMORY_RETRIEVAL] = StreamingReadiness.PARTIAL

        if PipelineStage.CONTEXT_BUILD.value in stage_names:
            readiness[StreamingAuditFlow.CONTEXT_BUILD] = StreamingReadiness.PARTIAL

        if PipelineStage.LLM_FIRST_TOKEN.value in stage_names:
            readiness[StreamingAuditFlow.LLM_TOKEN_OUTPUT] = StreamingReadiness.PARTIAL

        if PipelineStage.TTS_FIRST_AUDIO.value in stage_names:
            readiness[StreamingAuditFlow.TTS_SYNTHESIS] = StreamingReadiness.PARTIAL

        if PipelineStage.PLAYBACK_STARTUP.value in stage_names:
            readiness[StreamingAuditFlow.AUDIO_PLAYBACK] = StreamingReadiness.PARTIAL

        if PipelineStage.INTERRUPT_RECOVERY.value in stage_names:
            readiness[StreamingAuditFlow.INTERRUPT_RECOVERY] = (
                StreamingReadiness.PARTIAL
            )

        if PipelineFindingKind.PIPELINE_GAP in finding_kinds:
            for flow in (
                StreamingAuditFlow.CONTEXT_BUILD,
                StreamingAuditFlow.MEMORY_RETRIEVAL,
                StreamingAuditFlow.LLM_TOKEN_OUTPUT,
                StreamingAuditFlow.TTS_SYNTHESIS,
            ):
                if readiness.get(flow) == StreamingReadiness.STREAMING:
                    readiness[flow] = StreamingReadiness.PARTIAL

        return readiness

    @staticmethod
    def _infer_readiness_from_budget(
        report: LatencyBudgetAllocationReport,
    ) -> dict[StreamingAuditFlow, StreamingReadiness]:
        readiness = _default_partial_readiness()

        mapping = {
            LatencyBudgetSliceKind.AUDIO_CAPTURE_VAD: StreamingAuditFlow.STT_PARTIALS,
            LatencyBudgetSliceKind.STT_FIRST_PARTIAL: StreamingAuditFlow.STT_PARTIALS,
            LatencyBudgetSliceKind.CONTEXT_BUILD: StreamingAuditFlow.CONTEXT_BUILD,
            LatencyBudgetSliceKind.MEMORY_RETRIEVAL: (
                StreamingAuditFlow.MEMORY_RETRIEVAL
            ),
            LatencyBudgetSliceKind.LLM_FIRST_TOKEN: (
                StreamingAuditFlow.LLM_TOKEN_OUTPUT
            ),
            LatencyBudgetSliceKind.TTS_FIRST_CHUNK: StreamingAuditFlow.TTS_SYNTHESIS,
            LatencyBudgetSliceKind.AUDIO_OUTPUT: StreamingAuditFlow.AUDIO_PLAYBACK,
        }

        for evaluation in report.evaluations:
            flow = mapping.get(evaluation.slice_kind)

            if flow is None:
                continue

            if evaluation.status == LatencyBudgetAllocationStatus.PASSED:
                readiness[flow] = StreamingReadiness.STREAMING
            elif evaluation.status == LatencyBudgetAllocationStatus.WARNING:
                readiness[flow] = StreamingReadiness.PARTIAL
            else:
                readiness[flow] = StreamingReadiness.BLOCKING

        return readiness


def _default_partial_readiness() -> dict[StreamingAuditFlow, StreamingReadiness]:
    """
    Conservative default.

    Unknown architecture is treated as partial/blocking debt, never assumed safe.
    """

    return {
        flow: StreamingReadiness.PARTIAL
        for flow in StreamingAuditFlow
    }


def default_streaming_audit_flow_specs() -> tuple[StreamingAuditFlowSpec, ...]:
    """
    Default Step 4 audit checklist.

    This mirrors the Phase 7 roadmap checklist exactly.
    """

    return (
        StreamingAuditFlowSpec(
            flow=StreamingAuditFlow.STT_PARTIALS,
            name="Streaming STT partials",
            owner="presence_runtime",
            critical_path_rank=CriticalPathRank.STT,
            required_for_step=7,
            expected_behavior="STT streams partial transcripts before final silence.",
            blocking_failure=(
                "STT waits for silence before downstream cognition begins."
            ), 
            stages=(PipelineStage.STT_FIRST_PARTIAL,),
        ),
        StreamingAuditFlowSpec(
            flow=StreamingAuditFlow.TURN_FINALIZATION,
            name="Turn finalization",
            owner="conversation_runtime",
            critical_path_rank=CriticalPathRank.TURN_FINALIZATION,
            required_for_step=7,
            expected_behavior="Turn finalization emits stable updates incrementally.",
            blocking_failure=(
                "Turn finalization blocks all cognition until full endpoint."
            ),
            stages=(PipelineStage.STT_FINALIZATION,),
        ),
        StreamingAuditFlowSpec(
            flow=StreamingAuditFlow.MEMORY_RETRIEVAL,
            name="Streaming memory retrieval",
            owner="memory_runtime",
            critical_path_rank=CriticalPathRank.MEMORY_CONTEXT,
            required_for_step=8,
            expected_behavior="Memory retrieval streams useful results as they arrive.",
            blocking_failure="Memory retrieval waits for all memory sources to finish.",
            stages=(PipelineStage.MEMORY_RETRIEVAL,),
        ),
        StreamingAuditFlowSpec(
            flow=StreamingAuditFlow.CONTEXT_BUILD,
            name="Incremental context build",
            owner="cognition_runtime",
            critical_path_rank=CriticalPathRank.MEMORY_CONTEXT,
            required_for_step=10,
            expected_behavior="Context builder consumes memory incrementally.",
            blocking_failure="Context builder waits for full memory retrieval.",
            stages=(PipelineStage.CONTEXT_BUILD,),
        ),
        StreamingAuditFlowSpec(
            flow=StreamingAuditFlow.LLM_TOKEN_OUTPUT,
            name="LLM token streaming",
            owner="cognition_runtime",
            critical_path_rank=CriticalPathRank.LLM,
            required_for_step=5,
            expected_behavior="LLM streams first token immediately.",
            blocking_failure="LLM waits for full response before downstream TTS.",
            stages=(PipelineStage.LLM_FIRST_TOKEN, PipelineStage.LLM_STREAMING),
        ),
        StreamingAuditFlowSpec(
            flow=StreamingAuditFlow.TTS_SYNTHESIS,
            name="Streaming TTS synthesis",
            owner="presence_runtime",
            critical_path_rank=CriticalPathRank.TTS,
            required_for_step=6,
            expected_behavior="TTS begins on first stable sentence/chunk.",
            blocking_failure="TTS waits for full response text before synthesis.",
            stages=(PipelineStage.TTS_FIRST_AUDIO, PipelineStage.TTS_STREAMING),
        ),
        StreamingAuditFlowSpec(
            flow=StreamingAuditFlow.AUDIO_PLAYBACK,
            name="Progressive audio playback",
            owner="playback_runtime",
            critical_path_rank=CriticalPathRank.PLAYBACK,
            required_for_step=6,
            expected_behavior=(
                "Audio playback starts as soon as first audio chunk is ready."
            ),
            blocking_failure="Playback waits for full synthesized audio.",
            stages=(PipelineStage.PLAYBACK_STARTUP,),
        ),
        StreamingAuditFlowSpec(
            flow=StreamingAuditFlow.TOOL_FEEDBACK,
            name="Streaming tool feedback",
            owner="tool_runtime",
            critical_path_rank=CriticalPathRank.TOOLING,
            required_for_step=14,
            expected_behavior="Tools emit progress before completion.",
            blocking_failure=(
                "Tool runtime waits until action completion to update user."
            ),
            stages=(PipelineStage.TOOL_FIRST_FEEDBACK,),
        ),
        StreamingAuditFlowSpec(
            flow=StreamingAuditFlow.ACTION_PROGRESS,
            name="Streaming action progress",
            owner="tool_runtime",
            critical_path_rank=CriticalPathRank.TOOLING,
            required_for_step=14,
            expected_behavior="Actions stream progress milestones.",
            blocking_failure="Action runtime hides progress until final result.",
            stages=(PipelineStage.ACTION_PROGRESS_FEEDBACK,),
        ),
        StreamingAuditFlowSpec(
            flow=StreamingAuditFlow.INTERRUPT_RECOVERY,
            name="Streaming interruption recovery",
            owner="orchestration_runtime",
            critical_path_rank=CriticalPathRank.INTERRUPT,
            required_for_step=12,
            expected_behavior="Interrupt recovery restores from snapshot/delta.",
            blocking_failure="Interrupt recovery rebuilds context synchronously.",
            stages=(PipelineStage.INTERRUPT_RECOVERY,),
        ),
        StreamingAuditFlowSpec(
            flow=StreamingAuditFlow.SCHEDULER_DISPATCH,
            name="Latency-aware scheduler dispatch",
            owner="orchestration_runtime",
            critical_path_rank=CriticalPathRank.BACKGROUND,
            required_for_step=13,
            expected_behavior=(
                "Scheduler dispatches foreground work without queue stalls."
            ),
            blocking_failure="Scheduler queue stalls foreground conversation.",
            stages=(PipelineStage.SCHEDULER_QUEUE,),
        ),
        StreamingAuditFlowSpec(
            flow=StreamingAuditFlow.CACHE_LOOKUP,
            name="Streaming cache lookup",
            owner="latency_runtime",
            critical_path_rank=CriticalPathRank.MEMORY_CONTEXT,
            required_for_step=11,
            expected_behavior=(
                "Caches return fast hits and miss without blocking pipeline."
            ),
            blocking_failure="Cache lookup blocks context build or retrieval.",
            stages=(PipelineStage.CACHE_LOOKUP,),
        ),
    )