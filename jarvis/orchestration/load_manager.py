from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from threading import RLock

from pydantic import Field, field_validator, model_validator

from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel
from jarvis.orchestration.observability import (
    BottleneckSeverity,
    OrchestrationDashboard,
    OrchestrationHealth,
)


class CognitiveLoadLevel(IntEnum):
    """
    Cognitive orchestration load level.

    IntEnum is intentional: levels have ordered severity.
    """

    NORMAL = 0
    ELEVATED = 1
    HIGH = 2
    CRITICAL = 3
    SHEDDING = 4


class LoadSheddingTarget(StrEnum):
    """
    Ordered load-shedding targets.

    Conversation is present only as a protected target. It must never be shed.
    """

    BACKGROUND_MAINTENANCE = "background_maintenance"
    BACKGROUND_PREFETCH = "background_prefetch"
    NON_CRITICAL_MEMORY = "non_critical_memory"
    TOOL_PLANNING = "tool_planning"
    CONVERSATION = "conversation"


class LoadSheddingAction(StrEnum):
    """
    Action recommended by the load manager.
    """

    NONE = "none"
    MONITOR = "monitor"
    COMPRESS = "compress"
    DEFER = "defer"
    SHED = "shed"
    PAUSE = "pause"
    PROTECT = "protect"


class LoadDecisionReason(StrEnum):
    """
    Machine-readable load-management reasons.
    """

    LOAD_NORMAL = "load_normal"
    LOAD_ELEVATED = "load_elevated"
    LOAD_HIGH = "load_high"
    LOAD_CRITICAL = "load_critical"
    LOAD_SHEDDING = "load_shedding"
    HYSTERESIS_HOLD = "hysteresis_hold"
    BACKGROUND_MAINTENANCE_SHED = "background_maintenance_shed"
    BACKGROUND_PREFETCH_SHED = "background_prefetch_shed"
    NON_CRITICAL_MEMORY_COMPRESSED = "non_critical_memory_compressed"
    TOOL_PLANNING_DEFERRED = "tool_planning_deferred"
    CONVERSATION_PROTECTED = "conversation_protected"
    DASHBOARD_RECORDED = "dashboard_recorded"
    RUNTIME_RESET = "runtime_reset"


class LoadSignal(OrchestrationModel):
    """
    Normalized pressure signal derived from orchestration observability.

    This is the safe bridge from Step 13 Observability to Step 14 Load Manager.
    """

    worker_utilization_percent: int = Field(default=0, ge=0, le=100)
    resource_utilization_percent: int = Field(default=0, ge=0, le=100)
    queue_depth: int = Field(default=0, ge=0)
    interrupt_frequency: int = Field(default=0, ge=0)
    bottleneck_count: int = Field(default=0, ge=0)
    critical_bottleneck_count: int = Field(default=0, ge=0)
    dashboard_health: OrchestrationHealth = OrchestrationHealth.UNKNOWN
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @property
    def pressure_percent(self) -> int:
        """
        Conservative pressure score.

        Worker/resource pressure dominate. Queue, interrupt, and bottleneck
        pressure are capped because they are secondary pressure indicators.
        """

        queue_pressure = min(100, self.queue_depth * 5)
        interrupt_pressure = min(100, self.interrupt_frequency * 8)
        bottleneck_pressure = min(100, self.bottleneck_count * 20)
        critical_pressure = min(100, self.critical_bottleneck_count * 50)

        return max(
            self.worker_utilization_percent,
            self.resource_utilization_percent,
            queue_pressure,
            interrupt_pressure,
            bottleneck_pressure,
            critical_pressure,
        )

    @classmethod
    def from_dashboard(cls, dashboard: OrchestrationDashboard) -> LoadSignal:
        critical_bottlenecks = sum(
            1
            for item in dashboard.bottlenecks
            if item.detected and item.severity == BottleneckSeverity.CRITICAL
        )
        detected_bottlenecks = sum(
            1 for item in dashboard.bottlenecks if item.detected
        )

        return cls(
            worker_utilization_percent=(
                dashboard.metrics.worker_utilization_percent
            ),
            resource_utilization_percent=(
                dashboard.metrics.budget_consumption_percent
            ),
            queue_depth=dashboard.metrics.queue_depth,
            interrupt_frequency=dashboard.metrics.interrupt_frequency,
            bottleneck_count=detected_bottlenecks,
            critical_bottleneck_count=critical_bottlenecks,
            dashboard_health=dashboard.health,
            metadata={"dashboard_summary": dashboard.summary},
        )


class LoadSheddingDecision(OrchestrationModel):
    """
    Single load-shedding recommendation.

    The runtime recommends. Execution remains the responsibility of other
    governed runtimes.
    """

    target: LoadSheddingTarget
    action: LoadSheddingAction
    reason: LoadDecisionReason
    allowed: bool
    priority: int = Field(default=0, ge=0)
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("message")
    @classmethod
    def _required_message(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("message cannot be empty.")

        return cleaned

    @model_validator(mode="after")
    def _conversation_must_never_be_shed(self) -> LoadSheddingDecision:
        if (
            self.target == LoadSheddingTarget.CONVERSATION
            and self.action
            in {
                LoadSheddingAction.SHED,
                LoadSheddingAction.PAUSE,
                LoadSheddingAction.DEFER,
            }
        ):
            raise ValueError("conversation responsiveness must never be shed.")

        return self


class CognitiveLoadAssessment(OrchestrationModel):
    """
    Full load manager assessment.

    This is the queryable output used by future orchestration and debug UI.
    """

    level: CognitiveLoadLevel
    previous_level: CognitiveLoadLevel | None = None
    reason: LoadDecisionReason
    signal: LoadSignal
    decisions: tuple[LoadSheddingDecision, ...] = ()
    debug_mode: bool = False
    user_visible_events: tuple[str, ...] = ()
    conversation_protected: bool = True
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @property
    def should_shed(self) -> bool:
        return any(
            item.allowed
            and item.action
            in {
                LoadSheddingAction.SHED,
                LoadSheddingAction.PAUSE,
                LoadSheddingAction.DEFER,
                LoadSheddingAction.COMPRESS,
            }
            for item in self.decisions
        )

    @property
    def pressure_percent(self) -> int:
        return self.signal.pressure_percent


class LoadManagerResult(OrchestrationModel):
    """
    Result of a load-manager operation.
    """

    reason: LoadDecisionReason
    success: bool
    message: str
    assessment: CognitiveLoadAssessment | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("message")
    @classmethod
    def _required_message(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("message cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class CognitiveLoadManagerConfig:
    """
    Cognitive Load Manager configuration.

    Recovery is hysteretic: load must drop by recovery_margin_percent before
    returning to a lower level.
    """

    name: str = "cognitive_load_manager"
    elevated_threshold_percent: int = 60
    high_threshold_percent: int = 75
    critical_threshold_percent: int = 88
    shedding_threshold_percent: int = 95
    recovery_margin_percent: int = 20
    debug_mode: bool = False

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        thresholds = (
            self.elevated_threshold_percent,
            self.high_threshold_percent,
            self.critical_threshold_percent,
            self.shedding_threshold_percent,
        )

        if any(value < 1 or value > 100 for value in thresholds):
            raise ValueError("load thresholds must be within 1..100.")

        if thresholds != tuple(sorted(thresholds)):
            raise ValueError("load thresholds must be monotonically increasing.")

        if len(set(thresholds)) != len(thresholds):
            raise ValueError("load thresholds must be unique.")

        if self.recovery_margin_percent < 1:
            raise ValueError("recovery_margin_percent must be positive.")

        if self.recovery_margin_percent >= self.elevated_threshold_percent:
            raise ValueError(
                "recovery_margin_percent must be lower than elevated threshold."
            )


@dataclass(frozen=True, slots=True)
class CognitiveLoadManagerSnapshot:
    """
    Runtime diagnostics for the load manager.
    """

    name: str
    assessment_count: int
    last_level: CognitiveLoadLevel | None
    last_reason: LoadDecisionReason | None
    last_pressure_percent: int | None
    debug_mode: bool


class LoadPolicy:
    """
    Hysteretic load classification policy.

    Upward transitions are immediate.
    Downward transitions require pressure to drop below the current level's
    threshold minus recovery margin.
    """

    def __init__(
        self,
        *,
        config: CognitiveLoadManagerConfig | None = None,
    ) -> None:
        self._config = config or CognitiveLoadManagerConfig()
        self._config.validate()

    def classify(
        self,
        signal: LoadSignal,
        *,
        current_level: CognitiveLoadLevel | None = None,
    ) -> tuple[CognitiveLoadLevel, LoadDecisionReason]:
        raw_level = self._raw_level(signal)

        if current_level is None:
            return raw_level, self._reason_for_level(raw_level)

        if raw_level > current_level:
            return raw_level, self._reason_for_level(raw_level)

        if raw_level == current_level:
            return current_level, self._reason_for_level(current_level)

        if self._can_recover_from(current_level, signal.pressure_percent):
            return raw_level, self._reason_for_level(raw_level)

        return current_level, LoadDecisionReason.HYSTERESIS_HOLD

    def _raw_level(self, signal: LoadSignal) -> CognitiveLoadLevel:
        pressure = signal.pressure_percent

        if signal.critical_bottleneck_count > 0:
            return CognitiveLoadLevel.CRITICAL

        if signal.dashboard_health == OrchestrationHealth.CRITICAL:
            return CognitiveLoadLevel.CRITICAL

        if pressure >= self._config.shedding_threshold_percent:
            return CognitiveLoadLevel.SHEDDING

        if pressure >= self._config.critical_threshold_percent:
            return CognitiveLoadLevel.CRITICAL

        if pressure >= self._config.high_threshold_percent:
            return CognitiveLoadLevel.HIGH

        if pressure >= self._config.elevated_threshold_percent:
            return CognitiveLoadLevel.ELEVATED

        return CognitiveLoadLevel.NORMAL

    def _can_recover_from(
        self,
        current_level: CognitiveLoadLevel,
        pressure_percent: int,
    ) -> bool:
        recovery_threshold = max(
            0,
            self._entry_threshold(current_level)
            - self._config.recovery_margin_percent,
        )

        return pressure_percent <= recovery_threshold

    def _entry_threshold(self, level: CognitiveLoadLevel) -> int:
        return {
            CognitiveLoadLevel.NORMAL: 0,
            CognitiveLoadLevel.ELEVATED: (
                self._config.elevated_threshold_percent
            ),
            CognitiveLoadLevel.HIGH: self._config.high_threshold_percent,
            CognitiveLoadLevel.CRITICAL: self._config.critical_threshold_percent,
            CognitiveLoadLevel.SHEDDING: self._config.shedding_threshold_percent,
        }[level]

    @staticmethod
    def _reason_for_level(level: CognitiveLoadLevel) -> LoadDecisionReason:
        return {
            CognitiveLoadLevel.NORMAL: LoadDecisionReason.LOAD_NORMAL,
            CognitiveLoadLevel.ELEVATED: LoadDecisionReason.LOAD_ELEVATED,
            CognitiveLoadLevel.HIGH: LoadDecisionReason.LOAD_HIGH,
            CognitiveLoadLevel.CRITICAL: LoadDecisionReason.LOAD_CRITICAL,
            CognitiveLoadLevel.SHEDDING: LoadDecisionReason.LOAD_SHEDDING,
        }[level]


class LoadSheddingStrategy:
    """
    Ordered load-shedding strategy.

    Shedding order:
    1. background maintenance
    2. background prefetch
    3. non-critical memory
    4. tool planning
    5. conversation is protected, never shed
    """

    def decisions_for(
        self,
        level: CognitiveLoadLevel,
    ) -> tuple[LoadSheddingDecision, ...]:
        decisions: list[LoadSheddingDecision] = [
            self._conversation_protection_decision()
        ]

        if level == CognitiveLoadLevel.NORMAL:
            return tuple(decisions)

        decisions.append(
            LoadSheddingDecision(
                target=LoadSheddingTarget.BACKGROUND_MAINTENANCE,
                action=LoadSheddingAction.MONITOR,
                reason=LoadDecisionReason.LOAD_ELEVATED,
                allowed=True,
                priority=10,
                message="monitor background maintenance pressure",
            )
        )

        if level >= CognitiveLoadLevel.HIGH:
            decisions.append(
                LoadSheddingDecision(
                    target=LoadSheddingTarget.BACKGROUND_MAINTENANCE,
                    action=LoadSheddingAction.SHED,
                    reason=LoadDecisionReason.BACKGROUND_MAINTENANCE_SHED,
                    allowed=True,
                    priority=20,
                    message="shed background maintenance before user-facing work",
                )
            )
            decisions.append(
                LoadSheddingDecision(
                    target=LoadSheddingTarget.BACKGROUND_PREFETCH,
                    action=LoadSheddingAction.PAUSE,
                    reason=LoadDecisionReason.BACKGROUND_PREFETCH_SHED,
                    allowed=True,
                    priority=30,
                    message="pause background prefetch to reduce pressure",
                )
            )

        if level >= CognitiveLoadLevel.CRITICAL:
            decisions.append(
                LoadSheddingDecision(
                    target=LoadSheddingTarget.NON_CRITICAL_MEMORY,
                    action=LoadSheddingAction.COMPRESS,
                    reason=LoadDecisionReason.NON_CRITICAL_MEMORY_COMPRESSED,
                    allowed=True,
                    priority=40,
                    message="compress non-critical memory retrieval",
                )
            )

        if level >= CognitiveLoadLevel.SHEDDING:
            decisions.append(
                LoadSheddingDecision(
                    target=LoadSheddingTarget.TOOL_PLANNING,
                    action=LoadSheddingAction.DEFER,
                    reason=LoadDecisionReason.TOOL_PLANNING_DEFERRED,
                    allowed=True,
                    priority=50,
                    message="defer tool planning until load recovers",
                )
            )

        return tuple(sorted(decisions, key=lambda item: item.priority))

    @staticmethod
    def _conversation_protection_decision() -> LoadSheddingDecision:
        return LoadSheddingDecision(
            target=LoadSheddingTarget.CONVERSATION,
            action=LoadSheddingAction.PROTECT,
            reason=LoadDecisionReason.CONVERSATION_PROTECTED,
            allowed=True,
            priority=0,
            message="conversation responsiveness is protected",
        )


class CognitiveLoadMonitor:
    """
    Phase 6 Step 14 Cognitive Load Monitor.

    This is the public façade used by orchestration code when it wants to
    evaluate current cognitive load.
    """

    def __init__(
        self,
        *,
        policy: LoadPolicy | None = None,
        strategy: LoadSheddingStrategy | None = None,
        config: CognitiveLoadManagerConfig | None = None,
    ) -> None:
        self._config = config or CognitiveLoadManagerConfig()
        self._config.validate()
        self._policy = policy or LoadPolicy(config=self._config)
        self._strategy = strategy or LoadSheddingStrategy()

    def assess(
        self,
        signal: LoadSignal,
        *,
        previous_level: CognitiveLoadLevel | None = None,
    ) -> CognitiveLoadAssessment:
        level, reason = self._policy.classify(
            signal,
            current_level=previous_level,
        )
        decisions = self._strategy.decisions_for(level)
        user_visible_events = self._debug_events(level)

        return CognitiveLoadAssessment(
            level=level,
            previous_level=previous_level,
            reason=reason,
            signal=signal,
            decisions=decisions,
            debug_mode=self._config.debug_mode,
            user_visible_events=user_visible_events,
            conversation_protected=any(
                item.target == LoadSheddingTarget.CONVERSATION
                and item.action == LoadSheddingAction.PROTECT
                for item in decisions
            ),
        )

    def assess_dashboard(
        self,
        dashboard: OrchestrationDashboard,
        *,
        previous_level: CognitiveLoadLevel | None = None,
    ) -> CognitiveLoadAssessment:
        return self.assess(
            LoadSignal.from_dashboard(dashboard),
            previous_level=previous_level,
        )

    def _debug_events(
        self,
        level: CognitiveLoadLevel,
    ) -> tuple[str, ...]:
        if not self._config.debug_mode:
            return ()

        if level >= CognitiveLoadLevel.HIGH:
            return ("JARVIS is under load, background tasks paused",)

        if level == CognitiveLoadLevel.ELEVATED:
            return ("JARVIS load is elevated, background work is being monitored",)

        return ()


class CognitiveLoadManagerRuntime:
    """
    Phase 6 Step 14 Cognitive Load Manager Runtime.

    Responsibilities:
    - consume observability dashboards
    - classify load level
    - apply hysteretic recovery
    - produce ordered load-shedding recommendations
    - protect conversation responsiveness

    Non-responsibilities:
    - no direct task execution
    - no direct worker mutation
    - no direct memory/tool cancellation
    - no hidden background behavior
    """

    def __init__(
        self,
        *,
        config: CognitiveLoadManagerConfig | None = None,
        monitor: CognitiveLoadMonitor | None = None,
    ) -> None:
        self._config = config or CognitiveLoadManagerConfig()
        self._config.validate()

        self._monitor = monitor or CognitiveLoadMonitor(config=self._config)
        self._assessments: list[CognitiveLoadAssessment] = []
        self._last_level: CognitiveLoadLevel | None = None
        self._last_reason: LoadDecisionReason | None = None
        self._lock = RLock()

    @property
    def name(self) -> str:
        return self._config.name

    def record_signal(self, signal: LoadSignal) -> LoadManagerResult:
        with self._lock:
            previous_level = self._last_level

        assessment = self._monitor.assess(
            signal,
            previous_level=previous_level,
        )

        with self._lock:
            self._assessments.append(assessment)
            self._last_level = assessment.level
            self._last_reason = assessment.reason

        return LoadManagerResult(
            reason=assessment.reason,
            success=True,
            message="cognitive load signal recorded",
            assessment=assessment,
        )

    def record_dashboard(
        self,
        dashboard: OrchestrationDashboard,
    ) -> LoadManagerResult:
        signal = LoadSignal.from_dashboard(dashboard)
        result = self.record_signal(signal)

        return LoadManagerResult(
            reason=LoadDecisionReason.DASHBOARD_RECORDED,
            success=result.success,
            message="cognitive load dashboard recorded",
            assessment=result.assessment,
        )

    def latest_assessment(self) -> CognitiveLoadAssessment | None:
        with self._lock:
            if not self._assessments:
                return None

            return self._assessments[-1]

    def assessments(self) -> tuple[CognitiveLoadAssessment, ...]:
        with self._lock:
            return tuple(self._assessments)

    def snapshot(self) -> CognitiveLoadManagerSnapshot:
        with self._lock:
            latest = self._assessments[-1] if self._assessments else None

            return CognitiveLoadManagerSnapshot(
                name=self.name,
                assessment_count=len(self._assessments),
                last_level=self._last_level,
                last_reason=self._last_reason,
                last_pressure_percent=(
                    latest.pressure_percent if latest is not None else None
                ),
                debug_mode=self._config.debug_mode,
            )

    def reset(self) -> None:
        with self._lock:
            self._assessments.clear()
            self._last_level = None
            self._last_reason = LoadDecisionReason.RUNTIME_RESET