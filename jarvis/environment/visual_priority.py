from __future__ import annotations

from enum import StrEnum
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator

from jarvis.environment.worker_registry import EnvironmentWorkerKind
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class VisualTaskKind(StrEnum):
    """
    Work types competing for real-time attention.

    Phase 8 must arbitrate visual/environment work against conversation,
    interruption, tools, and action safety.
    """

    CONVERSATION = "conversation"
    INTERRUPTION = "interruption"
    VERIFICATION = "verification"
    RECOVERY = "recovery"
    ACTIVE_ACTION_TARGET = "active_action_target"
    FOCUSED_ENVIRONMENT_CONTEXT = "focused_environment_context"
    GROUNDING = "grounding"
    SEMANTIC_PARSING = "semantic_parsing"
    GRAPH_REFRESH = "graph_refresh"
    CAPTURE = "capture"
    OCR = "ocr"
    MEMORY_WRITE = "memory_write"
    MEMORY_CONSOLIDATION = "memory_consolidation"
    PASSIVE_OBSERVATION = "passive_observation"
    TTS = "tts"
    STT = "stt"
    TOOL_ACTION = "tool_action"


class VisualTaskPriority(StrEnum):
    """
    Priority class for environment arbitration.
    """

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"
    DEFERRED = "deferred"


class VisualTaskDecision(StrEnum):
    """
    Arbitration decision.
    """

    RUN_NOW = "run_now"
    RUN_LIMITED = "run_limited"
    DEFER = "defer"
    SHED = "shed"
    BLOCK = "block"


class VisualLoadLevel(StrEnum):
    """
    Runtime load level for visual/environment work.
    """

    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"
    SHEDDING = "shedding"


class VisualArbitrationReason(StrEnum):
    """
    Machine-readable arbitration reason.
    """

    TASK_ACCEPTED = "task_accepted"
    TASK_LIMITED_BY_BUDGET = "task_limited_by_budget"
    TASK_DEFERRED_BY_PRIORITY = "task_deferred_by_priority"
    TASK_SHED_BY_BACKPRESSURE = "task_shed_by_backpressure"
    TASK_BLOCKED_BY_CONVERSATION = "task_blocked_by_conversation"
    TASK_BLOCKED_BY_INTERRUPTION = "task_blocked_by_interruption"
    TASK_BLOCKED_BY_LOAD = "task_blocked_by_load"
    BUDGET_POLICY_UPDATED = "budget_policy_updated"
    BACKPRESSURE_UPDATED = "backpressure_updated"
    RUNTIME_RESET = "runtime_reset"


class VisualArbitrationEventKind(StrEnum):
    """
    Event kind emitted by the arbitrator.
    """

    TASK_ARBITRATED = "task_arbitrated"
    BUDGET_UPDATED = "budget_updated"
    BACKPRESSURE_UPDATED = "backpressure_updated"
    RUNTIME_RESET = "runtime_reset"


class VisualTaskBudget(OrchestrationModel):
    """
    Budget assigned to a task after arbitration.
    """

    latency_budget_ms: float = Field(gt=0)
    cpu_budget_percent: float = Field(gt=0, le=100)
    memory_budget_mb: float = Field(gt=0)
    queue_budget: int = Field(default=1, ge=1)
    max_frequency_hz: float = Field(default=1.0, gt=0)


class CaptureBudgetPolicy(OrchestrationModel):
    """
    Capture-specific budget policy.
    """

    focused_capture_ms: float = Field(default=50.0, gt=0)
    ambient_capture_ms: float = Field(default=120.0, gt=0)
    background_capture_ms: float = Field(default=250.0, gt=0)
    max_cpu_percent: float = Field(default=8.0, gt=0, le=100)
    memory_budget_mb: float = Field(default=128.0, gt=0)


class OCRBudgetPolicy(OrchestrationModel):
    """
    OCR-specific budget policy.
    """

    focused_ocr_ms: float = Field(default=300.0, gt=0)
    background_ocr_ms: float = Field(default=600.0, gt=0)
    max_cpu_percent: float = Field(default=20.0, gt=0, le=100)
    memory_budget_mb: float = Field(default=512.0, gt=0)


class GraphUpdateBudget(OrchestrationModel):
    """
    Workspace graph update budget.
    """

    foreground_update_ms: float = Field(default=200.0, gt=0)
    background_update_ms: float = Field(default=500.0, gt=0)
    max_cpu_percent: float = Field(default=10.0, gt=0, le=100)
    memory_budget_mb: float = Field(default=512.0, gt=0)


class VerificationPriorityPolicy(OrchestrationModel):
    """
    Verification and recovery priority policy.

    Verification and recovery protect correctness and safety, so they outrank
    background perception.
    """

    verification_ms: float = Field(default=300.0, gt=0)
    recovery_ms: float = Field(default=500.0, gt=0)
    max_cpu_percent: float = Field(default=12.0, gt=0, le=100)
    memory_budget_mb: float = Field(default=256.0, gt=0)


class EnvironmentBackpressureController(OrchestrationModel):
    """
    Backpressure state for visual/environment work.
    """

    load_level: VisualLoadLevel = VisualLoadLevel.NORMAL
    conversation_active: bool = False
    interruption_active: bool = False
    action_in_progress: bool = False
    cpu_pressure: float = Field(default=0.0, ge=0.0, le=1.0)
    memory_pressure: float = Field(default=0.0, ge=0.0, le=1.0)
    queue_pressure: float = Field(default=0.0, ge=0.0, le=1.0)

    def pressure_score(self) -> float:
        return max(self.cpu_pressure, self.memory_pressure, self.queue_pressure)


class VisualTaskRequest(OrchestrationModel):
    """
    Request to run one visual/environment task.

    This is a scheduling contract, not task execution.
    """

    task_id: str = Field(default_factory=lambda: f"visualtask_{uuid4().hex}")
    kind: VisualTaskKind
    worker_kind: EnvironmentWorkerKind | None = None
    requested_latency_ms: float = Field(gt=0)
    requested_cpu_percent: float = Field(gt=0, le=100)
    requested_memory_mb: float = Field(gt=0)
    can_run_in_background: bool = False
    can_degrade: bool = True
    can_shed: bool = False
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("task_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("task_id cannot be empty.")

        return cleaned


class VisualArbitrationDecision(OrchestrationModel):
    """
    Result of one visual arbitration decision.
    """

    decision_id: str = Field(default_factory=lambda: f"visualdecision_{uuid4().hex}")
    task_id: str
    task_kind: VisualTaskKind
    priority: VisualTaskPriority
    decision: VisualTaskDecision
    reason: VisualArbitrationReason
    budget: VisualTaskBudget | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("decision_id", "task_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class VisualArbitrationEvent(OrchestrationModel):
    """
    Typed arbitration event for observability.
    """

    event_id: str = Field(default_factory=lambda: f"visualevent_{uuid4().hex}")
    kind: VisualArbitrationEventKind
    reason: VisualArbitrationReason
    task_id: str | None = None
    task_kind: VisualTaskKind | None = None
    decision: VisualTaskDecision | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("event_id cannot be empty.")

        return cleaned


class VisualPrioritySnapshot(OrchestrationModel):
    """
    Runtime diagnostics for Phase 8 Step 2.
    """

    name: str
    decision_count: int = Field(ge=0)
    run_now_count: int = Field(ge=0)
    run_limited_count: int = Field(ge=0)
    deferred_count: int = Field(ge=0)
    shed_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    event_count: int = Field(ge=0)
    load_level: VisualLoadLevel
    conversation_active: bool
    interruption_active: bool
    last_reason: VisualArbitrationReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        return cleaned


class VisualPriorityArbitrator:
    """
    Phase 8 Step 2 Visual Priority Arbitration Runtime.

    Responsibilities:
    - protect Phase 7 responsiveness from Phase 8 visual workloads
    - arbitrate capture, OCR, graph, semantic, grounding, verification,
      recovery, memory writes, conversation, STT/TTS, and tool actions
    - enforce priority order
    - apply backpressure under load
    - limit/defer/shed visual work when needed

    Non-responsibilities:
    - no OCR
    - no capture
    - no graph building
    - no action execution
    """

    def __init__(
        self,
        *,
        name: str = "visual_priority_arbitrator",
        capture_policy: CaptureBudgetPolicy | None = None,
        ocr_policy: OCRBudgetPolicy | None = None,
        graph_policy: GraphUpdateBudget | None = None,
        verification_policy: VerificationPriorityPolicy | None = None,
    ) -> None:
        cleaned = name.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._capture_policy = capture_policy or CaptureBudgetPolicy()
        self._ocr_policy = ocr_policy or OCRBudgetPolicy()
        self._graph_policy = graph_policy or GraphUpdateBudget()
        self._verification_policy = (
            verification_policy or VerificationPriorityPolicy()
        )
        self._backpressure = EnvironmentBackpressureController()
        self._decisions: list[VisualArbitrationDecision] = []
        self._events: list[VisualArbitrationEvent] = []
        self._lock = RLock()
        self._last_reason: VisualArbitrationReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def update_backpressure(
        self,
        controller: EnvironmentBackpressureController,
    ) -> VisualArbitrationEvent:
        event = VisualArbitrationEvent(
            kind=VisualArbitrationEventKind.BACKPRESSURE_UPDATED,
            reason=VisualArbitrationReason.BACKPRESSURE_UPDATED,
            metadata={
                "load_level": controller.load_level.value,
                "conversation_active": controller.conversation_active,
                "interruption_active": controller.interruption_active,
                "pressure_score": controller.pressure_score(),
            },
        )

        with self._lock:
            self._backpressure = controller
            self._events.append(event)
            self._last_reason = event.reason

        return event

    def arbitrate(
        self,
        request: VisualTaskRequest,
    ) -> VisualArbitrationDecision:
        priority = self.priority_for(request.kind)
        decision, reason = self._decision_for(request=request, priority=priority)
        budget = self._budget_for(
            request=request,
            priority=priority,
            decision=decision,
        )

        arbitration = VisualArbitrationDecision(
            task_id=request.task_id,
            task_kind=request.kind,
            priority=priority,
            decision=decision,
            reason=reason,
            budget=budget,
        )
        event = VisualArbitrationEvent(
            kind=VisualArbitrationEventKind.TASK_ARBITRATED,
            reason=reason,
            task_id=request.task_id,
            task_kind=request.kind,
            decision=decision,
        )

        with self._lock:
            self._decisions.append(arbitration)
            self._events.append(event)
            self._last_reason = reason

        return arbitration

    def priority_for(self, kind: VisualTaskKind) -> VisualTaskPriority:
        """
        Canonical Phase 8 priority order.

        conversation > interruption > verification/recovery
        > active action target > focused environment context
        > graph refresh > OCR background > memory consolidation
        > passive observation
        """

        if kind in {
            VisualTaskKind.CONVERSATION,
            VisualTaskKind.STT,
            VisualTaskKind.TTS,
        }:
            return VisualTaskPriority.CRITICAL

        if kind == VisualTaskKind.INTERRUPTION:
            return VisualTaskPriority.CRITICAL

        if kind in {VisualTaskKind.VERIFICATION, VisualTaskKind.RECOVERY}:
            return VisualTaskPriority.HIGH

        if kind in {
            VisualTaskKind.ACTIVE_ACTION_TARGET,
            VisualTaskKind.GROUNDING,
            VisualTaskKind.TOOL_ACTION,
        }:
            return VisualTaskPriority.HIGH

        if kind in {
            VisualTaskKind.FOCUSED_ENVIRONMENT_CONTEXT,
            VisualTaskKind.CAPTURE,
            VisualTaskKind.SEMANTIC_PARSING,
        }:
            return VisualTaskPriority.NORMAL

        if kind == VisualTaskKind.GRAPH_REFRESH:
            return VisualTaskPriority.LOW

        if kind in {VisualTaskKind.OCR, VisualTaskKind.MEMORY_WRITE}:
            return VisualTaskPriority.BACKGROUND

        return VisualTaskPriority.DEFERRED

    def decisions(self) -> tuple[VisualArbitrationDecision, ...]:
        with self._lock:
            return tuple(self._decisions)

    def events(self) -> tuple[VisualArbitrationEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def snapshot(self) -> VisualPrioritySnapshot:
        with self._lock:
            decisions = tuple(self._decisions)

            return VisualPrioritySnapshot(
                name=self.name,
                decision_count=len(decisions),
                run_now_count=sum(
                    1
                    for decision in decisions
                    if decision.decision == VisualTaskDecision.RUN_NOW
                ),
                run_limited_count=sum(
                    1
                    for decision in decisions
                    if decision.decision == VisualTaskDecision.RUN_LIMITED
                ),
                deferred_count=sum(
                    1
                    for decision in decisions
                    if decision.decision == VisualTaskDecision.DEFER
                ),
                shed_count=sum(
                    1
                    for decision in decisions
                    if decision.decision == VisualTaskDecision.SHED
                ),
                blocked_count=sum(
                    1
                    for decision in decisions
                    if decision.decision == VisualTaskDecision.BLOCK
                ),
                event_count=len(self._events),
                load_level=self._backpressure.load_level,
                conversation_active=self._backpressure.conversation_active,
                interruption_active=self._backpressure.interruption_active,
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        event = VisualArbitrationEvent(
            kind=VisualArbitrationEventKind.RUNTIME_RESET,
            reason=VisualArbitrationReason.RUNTIME_RESET,
        )

        with self._lock:
            self._backpressure = EnvironmentBackpressureController()
            self._decisions.clear()
            self._events.clear()
            self._events.append(event)
            self._last_reason = VisualArbitrationReason.RUNTIME_RESET

    def _decision_for(
        self,
        *,
        request: VisualTaskRequest,
        priority: VisualTaskPriority,
    ) -> tuple[VisualTaskDecision, VisualArbitrationReason]:
        pressure = self._backpressure

        if pressure.interruption_active and request.kind != VisualTaskKind.INTERRUPTION:
            if priority in {VisualTaskPriority.BACKGROUND, VisualTaskPriority.DEFERRED}:
                return (
                    VisualTaskDecision.SHED,
                    VisualArbitrationReason.TASK_SHED_BY_BACKPRESSURE,
                )

            if priority != VisualTaskPriority.CRITICAL:
                return (
                    VisualTaskDecision.DEFER,
                    VisualArbitrationReason.TASK_BLOCKED_BY_INTERRUPTION,
                )

        if pressure.conversation_active and priority in {
            VisualTaskPriority.BACKGROUND,
            VisualTaskPriority.DEFERRED,
        }:
            if request.can_shed:
                return (
                    VisualTaskDecision.SHED,
                    VisualArbitrationReason.TASK_SHED_BY_BACKPRESSURE,
                )

            return (
                VisualTaskDecision.DEFER,
                VisualArbitrationReason.TASK_BLOCKED_BY_CONVERSATION,
            )

        if pressure.load_level in {VisualLoadLevel.CRITICAL, VisualLoadLevel.SHEDDING}:
            if priority in {VisualTaskPriority.BACKGROUND, VisualTaskPriority.DEFERRED}:
                return (
                    VisualTaskDecision.SHED,
                    VisualArbitrationReason.TASK_SHED_BY_BACKPRESSURE,
                )

            if priority == VisualTaskPriority.LOW:
                return (
                    VisualTaskDecision.DEFER,
                    VisualArbitrationReason.TASK_BLOCKED_BY_LOAD,
                )

        if pressure.load_level == VisualLoadLevel.HIGH and request.can_degrade:
            if priority in {
                VisualTaskPriority.NORMAL,
                VisualTaskPriority.LOW,
                VisualTaskPriority.BACKGROUND,
            }:
                return (
                    VisualTaskDecision.RUN_LIMITED,
                    VisualArbitrationReason.TASK_LIMITED_BY_BUDGET,
                )

        if self._exceeds_nominal_budget(request):
            if request.can_degrade:
                return (
                    VisualTaskDecision.RUN_LIMITED,
                    VisualArbitrationReason.TASK_LIMITED_BY_BUDGET,
                )

            return (
                VisualTaskDecision.DEFER,
                VisualArbitrationReason.TASK_DEFERRED_BY_PRIORITY,
            )

        return VisualTaskDecision.RUN_NOW, VisualArbitrationReason.TASK_ACCEPTED

    def _budget_for(
        self,
        *,
        request: VisualTaskRequest,
        priority: VisualTaskPriority,
        decision: VisualTaskDecision,
    ) -> VisualTaskBudget | None:
        if decision in {VisualTaskDecision.DEFER, VisualTaskDecision.SHED}:
            return None

        base = self._base_budget_for(request.kind)

        if decision == VisualTaskDecision.RUN_LIMITED:
            return VisualTaskBudget(
                latency_budget_ms=max(1.0, base.latency_budget_ms * 0.60),
                cpu_budget_percent=max(1.0, base.cpu_budget_percent * 0.50),
                memory_budget_mb=max(1.0, base.memory_budget_mb * 0.70),
                queue_budget=1,
                max_frequency_hz=max(0.1, base.max_frequency_hz * 0.50),
            )

        if priority == VisualTaskPriority.CRITICAL:
            return base.model_copy(update={"queue_budget": max(1, base.queue_budget)})

        return base

    def _base_budget_for(self, kind: VisualTaskKind) -> VisualTaskBudget:
        if kind == VisualTaskKind.CAPTURE:
            return VisualTaskBudget(
                latency_budget_ms=self._capture_policy.focused_capture_ms,
                cpu_budget_percent=self._capture_policy.max_cpu_percent,
                memory_budget_mb=self._capture_policy.memory_budget_mb,
                max_frequency_hz=10.0,
            )

        if kind == VisualTaskKind.OCR:
            return VisualTaskBudget(
                latency_budget_ms=self._ocr_policy.focused_ocr_ms,
                cpu_budget_percent=self._ocr_policy.max_cpu_percent,
                memory_budget_mb=self._ocr_policy.memory_budget_mb,
                max_frequency_hz=2.0,
            )

        if kind == VisualTaskKind.GRAPH_REFRESH:
            return VisualTaskBudget(
                latency_budget_ms=self._graph_policy.foreground_update_ms,
                cpu_budget_percent=self._graph_policy.max_cpu_percent,
                memory_budget_mb=self._graph_policy.memory_budget_mb,
                max_frequency_hz=2.0,
            )

        if kind == VisualTaskKind.VERIFICATION:
            return VisualTaskBudget(
                latency_budget_ms=self._verification_policy.verification_ms,
                cpu_budget_percent=self._verification_policy.max_cpu_percent,
                memory_budget_mb=self._verification_policy.memory_budget_mb,
                max_frequency_hz=5.0,
            )

        if kind == VisualTaskKind.RECOVERY:
            return VisualTaskBudget(
                latency_budget_ms=self._verification_policy.recovery_ms,
                cpu_budget_percent=self._verification_policy.max_cpu_percent,
                memory_budget_mb=self._verification_policy.memory_budget_mb,
                max_frequency_hz=3.0,
            )

        if kind in {
            VisualTaskKind.CONVERSATION,
            VisualTaskKind.STT,
            VisualTaskKind.TTS,
        }:
            return VisualTaskBudget(
                latency_budget_ms=80.0,
                cpu_budget_percent=15.0,
                memory_budget_mb=256.0,
                queue_budget=1,
                max_frequency_hz=20.0,
            )

        if kind == VisualTaskKind.INTERRUPTION:
            return VisualTaskBudget(
                latency_budget_ms=30.0,
                cpu_budget_percent=10.0,
                memory_budget_mb=128.0,
                queue_budget=1,
                max_frequency_hz=30.0,
            )

        return VisualTaskBudget(
            latency_budget_ms=200.0,
            cpu_budget_percent=10.0,
            memory_budget_mb=256.0,
            max_frequency_hz=1.0,
        )

    def _exceeds_nominal_budget(self, request: VisualTaskRequest) -> bool:
        budget = self._base_budget_for(request.kind)

        return (
            request.requested_latency_ms > budget.latency_budget_ms
            or request.requested_cpu_percent > budget.cpu_budget_percent
            or request.requested_memory_mb > budget.memory_budget_mb
        )