from __future__ import annotations

from enum import IntEnum, StrEnum
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class LatencySubsystem(StrEnum):
    """
    Runtime owner for a latency operation.

    Phase 7 is cross-runtime. Latency ownership must be explicit so bottlenecks
    can be assigned to the correct nervous-system component.
    """

    PRESENCE = "presence"
    CONVERSATION = "conversation"
    COGNITION = "cognition"
    MEMORY = "memory"
    TOOLS = "tools"
    ORCHESTRATION = "orchestration"
    RECOVERY = "recovery"
    PLAYBACK = "playback"
    LATENCY = "latency"


class LatencyOperation(StrEnum):
    """
    Critical Phase 7 operations.

    These are the first timing contracts for the real-time cognition pipeline.
    """

    STT_FIRST_TOKEN = "stt_first_token"
    STT_FINALIZATION = "stt_finalization"
    MEMORY_RETRIEVAL = "memory_retrieval"
    CONTEXT_BUILD = "context_build"
    LLM_FIRST_TOKEN = "llm_first_token"
    LLM_FULL_RESPONSE = "llm_full_response"
    TTS_FIRST_AUDIO = "tts_first_audio"
    PLAYBACK_STARTUP = "playback_startup"
    INTERRUPT_RESPONSE = "interrupt_response"
    RECOVERY_RECONSTRUCT = "recovery_reconstruct"
    TOOL_FIRST_FEEDBACK = "tool_first_feedback"
    ACTION_PROGRESS_FEEDBACK = "action_progress_feedback"


class LatencyPercentile(IntEnum):
    """
    Percentiles tracked by Phase 7.

    Average latency is intentionally not the primary metric.
    """

    P50 = 50
    P90 = 90
    P95 = 95
    P99 = 99


class LatencySeverity(StrEnum):
    """
    Budget evaluation severity.
    """

    OK = "ok"
    WARNING = "warning"
    VIOLATION = "violation"
    CRITICAL = "critical"


class LatencyEventKind(StrEnum):
    """
    Measurement lifecycle event kind.
    """

    SPAN_STARTED = "span_started"
    SPAN_COMPLETED = "span_completed"
    MEASUREMENT_RECORDED = "measurement_recorded"
    BUDGET_WARNING = "budget_warning"
    BUDGET_VIOLATION = "budget_violation"
    BUDGET_CRITICAL = "budget_critical"


class LatencyTarget(OrchestrationModel):
    """
    SLA-style target for one operation at a specific percentile.

    Example:
    STT first token p95 target = 150ms, max = 250ms.
    """

    operation: LatencyOperation
    subsystem: LatencySubsystem
    percentile: LatencyPercentile = LatencyPercentile.P95
    target_ms: int = Field(gt=0)
    warning_ms: int = Field(gt=0)
    max_ms: int = Field(gt=0)
    description: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("description")
    @classmethod
    def _required_description(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("description cannot be empty.")

        return cleaned

    @model_validator(mode="after")
    def _validate_threshold_order(self) -> LatencyTarget:
        if self.warning_ms < self.target_ms:
            raise ValueError("warning_ms must be greater than or equal to target_ms.")

        if self.max_ms < self.warning_ms:
            raise ValueError("max_ms must be greater than or equal to warning_ms.")

        return self


class LatencyBudget(OrchestrationModel):
    """
    Runtime latency budget contract.

    A budget can hold multiple percentile targets for the same operation.
    """

    budget_id: str = Field(default_factory=lambda: uuid4().hex)
    operation: LatencyOperation
    subsystem: LatencySubsystem
    owner: str
    targets: tuple[LatencyTarget, ...]
    hard_realtime: bool = False
    user_visible: bool = True
    enabled: bool = True
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("budget_id", "owner")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @model_validator(mode="after")
    def _validate_targets(self) -> LatencyBudget:
        if not self.targets:
            raise ValueError("latency budget must include at least one target.")

        for target in self.targets:
            if target.operation != self.operation:
                raise ValueError("target operation must match budget operation.")

            if target.subsystem != self.subsystem:
                raise ValueError("target subsystem must match budget subsystem.")

        return self

    def target_for(
        self,
        percentile: LatencyPercentile,
    ) -> LatencyTarget | None:
        for target in self.targets:
            if target.percentile == percentile:
                return target

        return None


class LatencyMeasurement(OrchestrationModel):
    """
    One measured operation duration.

    This is the core timing fact emitted by Phase 7 instrumentation.
    """

    measurement_id: str = Field(default_factory=lambda: uuid4().hex)
    operation: LatencyOperation
    subsystem: LatencySubsystem
    duration_ms: float = Field(ge=0)
    trace_id: str | None = None
    span_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("measurement_id")
    @classmethod
    def _required_measurement_id(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("measurement_id cannot be empty.")

        return cleaned


class LatencyBudgetResult(OrchestrationModel):
    """
    Result of evaluating a measurement against a budget.
    """

    operation: LatencyOperation
    subsystem: LatencySubsystem
    duration_ms: float
    percentile: LatencyPercentile
    severity: LatencySeverity
    target_ms: int
    warning_ms: int
    max_ms: int
    within_target: bool
    within_max: bool
    message: str
    measurement_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("message")
    @classmethod
    def _required_message(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("message cannot be empty.")

        return cleaned


class LatencyEvent(OrchestrationModel):
    """
    Event emitted by latency instrumentation.

    These events later feed Step 1 profiler and observability.
    """

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    kind: LatencyEventKind
    operation: LatencyOperation
    subsystem: LatencySubsystem
    trace_id: str | None = None
    span_id: str | None = None
    duration_ms: float | None = None
    severity: LatencySeverity | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_event_id(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("event_id cannot be empty.")

        return cleaned


class LatencySpan(OrchestrationModel):
    """
    Completed measured span.

    Step 0 records completed spans. Step 1 will build full pipeline spans.
    """

    span_id: str = Field(default_factory=lambda: uuid4().hex)
    operation: LatencyOperation
    subsystem: LatencySubsystem
    start_ns: int = Field(ge=0)
    end_ns: int = Field(ge=0)
    trace_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("span_id")
    @classmethod
    def _required_span_id(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("span_id cannot be empty.")

        return cleaned

    @model_validator(mode="after")
    def _validate_time_order(self) -> LatencySpan:
        if self.end_ns < self.start_ns:
            raise ValueError("end_ns must be greater than or equal to start_ns.")

        return self

    def duration_ms(self) -> float:
        return (self.end_ns - self.start_ns) / 1_000_000.0

    def to_measurement(self) -> LatencyMeasurement:
        return LatencyMeasurement(
            operation=self.operation,
            subsystem=self.subsystem,
            duration_ms=self.duration_ms(),
            trace_id=self.trace_id,
            span_id=self.span_id,
            metadata=dict(self.metadata),
        )


class PercentileSnapshot(OrchestrationModel):
    """
    Percentile view for one operation.

    Worst-case is tracked because p95 alone can hide rare but painful stalls.
    """

    operation: LatencyOperation
    subsystem: LatencySubsystem
    sample_count: int = Field(ge=0)
    p50_ms: float = Field(ge=0)
    p90_ms: float = Field(ge=0)
    p95_ms: float = Field(ge=0)
    p99_ms: float = Field(ge=0)
    worst_ms: float = Field(ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)


class LatencyViolation(OrchestrationModel):
    """
    Stored violation/warning fact.

    Violations are not vague logs. They are typed evidence.
    """

    violation_id: str = Field(default_factory=lambda: uuid4().hex)
    operation: LatencyOperation
    subsystem: LatencySubsystem
    severity: LatencySeverity
    duration_ms: float
    max_ms: int
    message: str
    measurement_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("violation_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class LatencyRuntimeSnapshot(OrchestrationModel):
    """
    Step 0 runtime snapshot.

    This is the initial nervous-system timing diagnostic surface.
    """

    name: str
    budget_count: int = Field(ge=0)
    measurement_count: int = Field(ge=0)
    violation_count: int = Field(ge=0)
    warning_count: int = Field(ge=0)
    critical_count: int = Field(ge=0)
    percentile_count: int = Field(ge=0)
    last_severity: LatencySeverity | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        return cleaned