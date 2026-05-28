from __future__ import annotations

from enum import StrEnum
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class EnvironmentWorkerKind(StrEnum):
    """
    Phase 8 governed environment workers.

    No Phase 8 visual/environment worker may run unless it is registered here.
    """

    CAPTURE = "capture_worker"
    OCR = "ocr_worker"
    UI_DETECTION = "ui_detection_worker"
    ENVIRONMENT_OBSERVER = "environment_observer_worker"
    ENVIRONMENT_STATE = "environment_state_worker"
    ENVIRONMENT_TIMELINE = "environment_timeline_worker"
    TRUST_CALIBRATION = "trust_calibration_worker"
    VISUAL_PRIORITY = "visual_priority_worker"
    WORKSPACE_GRAPH = "workspace_graph_worker"
    UI_SEMANTIC = "ui_semantic_worker"
    VISUAL_GROUNDING = "visual_grounding_worker"
    INTENT_PERSISTENCE = "intent_persistence_worker"
    SIMULATION = "simulation_worker"
    INTERACTION = "interaction_worker"
    VERIFICATION = "verification_worker"
    RECOVERY = "recovery_worker"
    ENVIRONMENT_MEMORY = "environment_memory_worker"
    HUMAN_COLLABORATION = "human_collaboration_worker"
    SECURITY_AUDIT = "security_audit_worker"


class EnvironmentSubsystem(StrEnum):
    """
    Phase 8 subsystem ownership.
    """

    FOUNDATION = "foundation"
    VISUAL_PERCEPTION = "visual_perception"
    ENVIRONMENT_MAPPING = "environment_mapping"
    UI_COGNITION = "ui_cognition"
    FUSION_INTENT = "fusion_intent"
    SIMULATION_PLANNING = "simulation_planning"
    SAFE_INTERACTION = "safe_interaction"
    VERIFICATION_RECOVERY = "verification_recovery"
    MEMORY_COLLABORATION = "memory_collaboration"
    SECURITY = "security"


class EnvironmentWorkerCapability(StrEnum):
    """
    Declared capability of a Phase 8 worker.
    """

    CAPTURE_SCREEN = "capture_screen"
    EXTRACT_TEXT = "extract_text"
    DETECT_UI_ELEMENTS = "detect_ui_elements"
    OBSERVE_ENVIRONMENT = "observe_environment"
    BUILD_ENVIRONMENT_STATE = "build_environment_state"
    TRACK_TIMELINE = "track_timeline"
    CALIBRATE_TRUST = "calibrate_trust"
    ARBITRATE_PRIORITY = "arbitrate_priority"
    BUILD_WORKSPACE_GRAPH = "build_workspace_graph"
    UNDERSTAND_UI_SEMANTICS = "understand_ui_semantics"
    GROUND_VISUAL_TARGETS = "ground_visual_targets"
    PERSIST_INTENT = "persist_intent"
    SIMULATE_OUTCOME = "simulate_outcome"
    INTERACT_WITH_ENVIRONMENT = "interact_with_environment"
    VERIFY_ACTION = "verify_action"
    RECOVER_FROM_FAILURE = "recover_from_failure"
    STORE_ENVIRONMENT_MEMORY = "store_environment_memory"
    COLLABORATE_WITH_HUMAN = "collaborate_with_human"
    AUDIT_ENVIRONMENT_SECURITY = "audit_environment_security"


class EnvironmentWorkerHealth(StrEnum):
    """
    Worker health state.
    """

    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    DISABLED = "disabled"


class EnvironmentWorkerPriority(StrEnum):
    """
    Worker priority class.

    Conversation and interruption always remain above background visual work.
    """

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"


class EnvironmentAttentionRequirement(StrEnum):
    """
    Environment attention requirement.
    """

    NONE = "none"
    PERIPHERAL = "peripheral"
    AMBIENT = "ambient"
    FOCUSED = "focused"
    DEEP = "deep"


class EnvironmentRestartPolicy(StrEnum):
    """
    Worker restart behavior.
    """

    NEVER = "never"
    ON_FAILURE = "on_failure"
    ALWAYS = "always"
    MANUAL = "manual"


class EnvironmentCircuitBreakerPolicy(StrEnum):
    """
    Circuit breaker behavior for worker failure storms.
    """

    NONE = "none"
    OPEN_ON_FAILURE_RATE = "open_on_failure_rate"
    OPEN_ON_LATENCY_BREACH = "open_on_latency_breach"
    OPEN_ON_POLICY_VIOLATION = "open_on_policy_violation"


class EnvironmentFailurePolicy(StrEnum):
    """
    Failure behavior when worker cannot complete safely.
    """

    FAIL_FAST = "fail_fast"
    DEGRADE_GRACEFULLY = "degrade_gracefully"
    RETRY_THEN_DEGRADE = "retry_then_degrade"
    ESCALATE_TO_USER = "escalate_to_user"
    DISABLE_WORKER = "disable_worker"


class EnvironmentWorkerRegistrationStatus(StrEnum):
    """
    Registry operation status.
    """

    REGISTERED = "registered"
    UPDATED = "updated"
    REJECTED = "rejected"
    NOT_FOUND = "not_found"
    RESET = "reset"


class EnvironmentWorkerRegistrationReason(StrEnum):
    """
    Machine-readable registry reason.
    """

    WORKER_REGISTERED = "worker_registered"
    WORKER_UPDATED = "worker_updated"
    WORKER_DUPLICATE_REJECTED = "worker_duplicate_rejected"
    WORKER_NOT_FOUND = "worker_not_found"
    REQUIRED_WORKER_MISSING = "required_worker_missing"
    ALL_REQUIRED_WORKERS_REGISTERED = "all_required_workers_registered"
    RUNTIME_RESET = "runtime_reset"


class EnvironmentWorkerBudget(OrchestrationModel):
    """
    Resource budget declared by a worker.

    Budgets exist before execution so Phase 8 cannot accidentally destroy
    Phase 7 responsiveness.
    """

    latency_budget_ms: float = Field(gt=0)
    cpu_budget_percent: float = Field(gt=0, le=100)
    memory_budget_mb: float = Field(gt=0)
    queue_budget: int = Field(default=1, ge=1)
    background_allowed: bool = False


class EnvironmentWorkerDescriptor(OrchestrationModel):
    """
    Contract for one governed Phase 8 worker.
    """

    worker_id: str = Field(default_factory=lambda: f"envworker_{uuid4().hex}")
    kind: EnvironmentWorkerKind
    name: str
    subsystem: EnvironmentSubsystem
    capability: EnvironmentWorkerCapability
    budget: EnvironmentWorkerBudget
    attention_requirement: EnvironmentAttentionRequirement
    priority: EnvironmentWorkerPriority
    restart_policy: EnvironmentRestartPolicy
    circuit_breaker_policy: EnvironmentCircuitBreakerPolicy
    failure_policy: EnvironmentFailurePolicy
    health: EnvironmentWorkerHealth = EnvironmentWorkerHealth.UNKNOWN
    required: bool = True
    enabled: bool = True
    version: str = "1.0"
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("worker_id", "name", "version")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @model_validator(mode="after")
    def _background_workers_must_be_low_priority(
        self,
    ) -> EnvironmentWorkerDescriptor:
        if not self.budget.background_allowed:
            return self

        allowed = {
            EnvironmentWorkerPriority.LOW,
            EnvironmentWorkerPriority.BACKGROUND,
            EnvironmentWorkerPriority.NORMAL,
        }

        if self.priority not in allowed:
            raise ValueError("background workers cannot be CRITICAL or HIGH priority.")

        return self

    @model_validator(mode="after")
    def _critical_workers_need_circuit_breaker(
        self,
    ) -> EnvironmentWorkerDescriptor:
        if self.priority != EnvironmentWorkerPriority.CRITICAL:
            return self

        if self.circuit_breaker_policy == EnvironmentCircuitBreakerPolicy.NONE:
            raise ValueError("critical workers require circuit breaker policy.")

        return self


class EnvironmentWorkerRegistryEvent(OrchestrationModel):
    """
    Registry event for observability.
    """

    event_id: str = Field(default_factory=lambda: f"envworker_event_{uuid4().hex}")
    reason: EnvironmentWorkerRegistrationReason
    status: EnvironmentWorkerRegistrationStatus
    worker_kind: EnvironmentWorkerKind | None = None
    worker_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("event_id cannot be empty.")

        return cleaned


class EnvironmentWorkerRegistryResult(OrchestrationModel):
    """
    Result returned by registry operations.
    """

    success: bool
    reason: EnvironmentWorkerRegistrationReason
    status: EnvironmentWorkerRegistrationStatus
    worker: EnvironmentWorkerDescriptor | None = None
    event: EnvironmentWorkerRegistryEvent
    message: str

    @field_validator("message")
    @classmethod
    def _required_message(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("message cannot be empty.")

        return cleaned


class EnvironmentWorkerRegistrySnapshot(OrchestrationModel):
    """
    Diagnostics for Phase 8 Step 1.
    """

    name: str
    worker_count: int = Field(ge=0)
    enabled_count: int = Field(ge=0)
    required_count: int = Field(ge=0)
    healthy_count: int = Field(ge=0)
    degraded_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    background_count: int = Field(ge=0)
    event_count: int = Field(ge=0)
    last_reason: EnvironmentWorkerRegistrationReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        return cleaned


class EnvironmentWorkerRegistry:
    """
    Phase 8 Step 1 Environment Worker Registry.

    Responsibilities:
    - register every Phase 8 environment worker before runtime execution
    - enforce no hidden visual workers
    - declare subsystem, capability, budgets, priority, attention, restart,
      circuit breaker, and failure policy
    - provide health and readiness checks

    Non-responsibilities:
    - no capture
    - no OCR
    - no UI detection
    - no interaction execution
    - no worker scheduling
    """

    def __init__(self, *, name: str = "environment_worker_registry") -> None:
        cleaned = name.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._workers: dict[EnvironmentWorkerKind, EnvironmentWorkerDescriptor] = {}
        self._events: list[EnvironmentWorkerRegistryEvent] = []
        self._lock = RLock()
        self._last_reason: EnvironmentWorkerRegistrationReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def register(
        self,
        worker: EnvironmentWorkerDescriptor,
        *,
        replace: bool = False,
    ) -> EnvironmentWorkerRegistryResult:
        with self._lock:
            existing = self._workers.get(worker.kind)

            if existing is not None and not replace:
                event = self._event(
                    reason=(
                        EnvironmentWorkerRegistrationReason
                        .WORKER_DUPLICATE_REJECTED
                    ),
                    status=EnvironmentWorkerRegistrationStatus.REJECTED,
                    worker=worker,
                )
                self._events.append(event)
                self._last_reason = event.reason

                return EnvironmentWorkerRegistryResult(
                    success=False,
                    reason=event.reason,
                    status=event.status,
                    worker=existing,
                    event=event,
                    message="environment worker already registered",
                )

            status = (
                EnvironmentWorkerRegistrationStatus.UPDATED
                if existing is not None
                else EnvironmentWorkerRegistrationStatus.REGISTERED
            )
            reason = (
                EnvironmentWorkerRegistrationReason.WORKER_UPDATED
                if existing is not None
                else EnvironmentWorkerRegistrationReason.WORKER_REGISTERED
            )
            self._workers[worker.kind] = worker
            event = self._event(reason=reason, status=status, worker=worker)
            self._events.append(event)
            self._last_reason = reason

        return EnvironmentWorkerRegistryResult(
            success=True,
            reason=reason,
            status=status,
            worker=worker,
            event=event,
            message="environment worker registered",
        )

    def register_defaults(self) -> tuple[EnvironmentWorkerRegistryResult, ...]:
        return tuple(self.register(worker) for worker in default_environment_workers())

    def update_health(
        self,
        kind: EnvironmentWorkerKind,
        health: EnvironmentWorkerHealth,
    ) -> EnvironmentWorkerRegistryResult:
        with self._lock:
            worker = self._workers.get(kind)

            if worker is None:
                event = self._event(
                    reason=EnvironmentWorkerRegistrationReason.WORKER_NOT_FOUND,
                    status=EnvironmentWorkerRegistrationStatus.NOT_FOUND,
                    worker_kind=kind,
                )
                self._events.append(event)
                self._last_reason = event.reason

                return EnvironmentWorkerRegistryResult(
                    success=False,
                    reason=event.reason,
                    status=event.status,
                    event=event,
                    message="environment worker not found",
                )

            updated = worker.model_copy(update={"health": health})
            self._workers[kind] = updated
            event = self._event(
                reason=EnvironmentWorkerRegistrationReason.WORKER_UPDATED,
                status=EnvironmentWorkerRegistrationStatus.UPDATED,
                worker=updated,
            )
            self._events.append(event)
            self._last_reason = event.reason

        return EnvironmentWorkerRegistryResult(
            success=True,
            reason=event.reason,
            status=event.status,
            worker=updated,
            event=event,
            message="environment worker health updated",
        )

    def get(
        self,
        kind: EnvironmentWorkerKind,
    ) -> EnvironmentWorkerDescriptor | None:
        with self._lock:
            return self._workers.get(kind)

    def all_workers(self) -> tuple[EnvironmentWorkerDescriptor, ...]:
        with self._lock:
            return tuple(self._workers.values())

    def workers_for_subsystem(
        self,
        subsystem: EnvironmentSubsystem,
    ) -> tuple[EnvironmentWorkerDescriptor, ...]:
        with self._lock:
            return tuple(
                worker
                for worker in self._workers.values()
                if worker.subsystem == subsystem
            )

    def events(self) -> tuple[EnvironmentWorkerRegistryEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def missing_required_workers(self) -> tuple[EnvironmentWorkerKind, ...]:
        registered = {worker.kind for worker in self.all_workers()}
        required = {worker.kind for worker in default_environment_workers()}

        return tuple(kind for kind in required if kind not in registered)

    def readiness_event(self) -> EnvironmentWorkerRegistryEvent:
        missing = self.missing_required_workers()

        if missing:
            event = EnvironmentWorkerRegistryEvent(
                reason=EnvironmentWorkerRegistrationReason.REQUIRED_WORKER_MISSING,
                status=EnvironmentWorkerRegistrationStatus.REJECTED,
                metadata={"missing": [kind.value for kind in missing]},
            )
        else:
            event = EnvironmentWorkerRegistryEvent(
                reason=(
                    EnvironmentWorkerRegistrationReason
                    .ALL_REQUIRED_WORKERS_REGISTERED
                ),
                status=EnvironmentWorkerRegistrationStatus.REGISTERED,
            )

        with self._lock:
            self._events.append(event)
            self._last_reason = event.reason

        return event

    def is_ready(self) -> bool:
        return not self.missing_required_workers()

    def snapshot(self) -> EnvironmentWorkerRegistrySnapshot:
        with self._lock:
            workers = tuple(self._workers.values())

            return EnvironmentWorkerRegistrySnapshot(
                name=self.name,
                worker_count=len(workers),
                enabled_count=sum(1 for worker in workers if worker.enabled),
                required_count=sum(1 for worker in workers if worker.required),
                healthy_count=sum(
                    1
                    for worker in workers
                    if worker.health == EnvironmentWorkerHealth.HEALTHY
                ),
                degraded_count=sum(
                    1
                    for worker in workers
                    if worker.health == EnvironmentWorkerHealth.DEGRADED
                ),
                failed_count=sum(
                    1
                    for worker in workers
                    if worker.health == EnvironmentWorkerHealth.FAILED
                ),
                background_count=sum(
                    1 for worker in workers if worker.budget.background_allowed
                ),
                event_count=len(self._events),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        with self._lock:
            self._workers.clear()
            self._events.clear()
            self._last_reason = EnvironmentWorkerRegistrationReason.RUNTIME_RESET

    @staticmethod
    def _event(
        *,
        reason: EnvironmentWorkerRegistrationReason,
        status: EnvironmentWorkerRegistrationStatus,
        worker: EnvironmentWorkerDescriptor | None = None,
        worker_kind: EnvironmentWorkerKind | None = None,
    ) -> EnvironmentWorkerRegistryEvent:
        return EnvironmentWorkerRegistryEvent(
            reason=reason,
            status=status,
            worker_kind=worker.kind if worker is not None else worker_kind,
            worker_id=worker.worker_id if worker is not None else None,
        )


def default_environment_workers() -> tuple[EnvironmentWorkerDescriptor, ...]:
    """
    Canonical Phase 8 worker set.

    These are declarations only. They do not execute.
    """

    return (
        _worker(
            kind=EnvironmentWorkerKind.CAPTURE,
            name="CaptureWorker",
            subsystem=EnvironmentSubsystem.VISUAL_PERCEPTION,
            capability=EnvironmentWorkerCapability.CAPTURE_SCREEN,
            latency_ms=50,
            cpu_percent=8,
            memory_mb=128,
            attention=EnvironmentAttentionRequirement.FOCUSED,
            priority=EnvironmentWorkerPriority.HIGH,
            restart=EnvironmentRestartPolicy.ON_FAILURE,
            circuit=EnvironmentCircuitBreakerPolicy.OPEN_ON_LATENCY_BREACH,
            failure=EnvironmentFailurePolicy.DEGRADE_GRACEFULLY,
        ),
        _worker(
            kind=EnvironmentWorkerKind.OCR,
            name="OCRWorker",
            subsystem=EnvironmentSubsystem.VISUAL_PERCEPTION,
            capability=EnvironmentWorkerCapability.EXTRACT_TEXT,
            latency_ms=300,
            cpu_percent=20,
            memory_mb=512,
            attention=EnvironmentAttentionRequirement.FOCUSED,
            priority=EnvironmentWorkerPriority.NORMAL,
            restart=EnvironmentRestartPolicy.ON_FAILURE,
            circuit=EnvironmentCircuitBreakerPolicy.OPEN_ON_LATENCY_BREACH,
            failure=EnvironmentFailurePolicy.DEGRADE_GRACEFULLY,
            background=True,
        ),
        _worker(
            kind=EnvironmentWorkerKind.UI_DETECTION,
            name="UIDetectionWorker",
            subsystem=EnvironmentSubsystem.VISUAL_PERCEPTION,
            capability=EnvironmentWorkerCapability.DETECT_UI_ELEMENTS,
            latency_ms=250,
            cpu_percent=18,
            memory_mb=384,
            attention=EnvironmentAttentionRequirement.FOCUSED,
            priority=EnvironmentWorkerPriority.NORMAL,
            restart=EnvironmentRestartPolicy.ON_FAILURE,
            circuit=EnvironmentCircuitBreakerPolicy.OPEN_ON_FAILURE_RATE,
            failure=EnvironmentFailurePolicy.RETRY_THEN_DEGRADE,
            background=True,
        ),
        _worker(
            kind=EnvironmentWorkerKind.ENVIRONMENT_OBSERVER,
            name="EnvironmentObserverWorker",
            subsystem=EnvironmentSubsystem.FOUNDATION,
            capability=EnvironmentWorkerCapability.OBSERVE_ENVIRONMENT,
            latency_ms=30,
            cpu_percent=5,
            memory_mb=128,
            attention=EnvironmentAttentionRequirement.PERIPHERAL,
            priority=EnvironmentWorkerPriority.HIGH,
            restart=EnvironmentRestartPolicy.ALWAYS,
            circuit=EnvironmentCircuitBreakerPolicy.OPEN_ON_FAILURE_RATE,
            failure=EnvironmentFailurePolicy.DEGRADE_GRACEFULLY,
        ),
        _worker(
            kind=EnvironmentWorkerKind.ENVIRONMENT_STATE,
            name="EnvironmentStateWorker",
            subsystem=EnvironmentSubsystem.FOUNDATION,
            capability=EnvironmentWorkerCapability.BUILD_ENVIRONMENT_STATE,
            latency_ms=50,
            cpu_percent=6,
            memory_mb=192,
            attention=EnvironmentAttentionRequirement.AMBIENT,
            priority=EnvironmentWorkerPriority.HIGH,
            restart=EnvironmentRestartPolicy.ALWAYS,
            circuit=EnvironmentCircuitBreakerPolicy.OPEN_ON_FAILURE_RATE,
            failure=EnvironmentFailurePolicy.FAIL_FAST,
        ),
        _worker(
            kind=EnvironmentWorkerKind.ENVIRONMENT_TIMELINE,
            name="EnvironmentTimelineWorker",
            subsystem=EnvironmentSubsystem.FOUNDATION,
            capability=EnvironmentWorkerCapability.TRACK_TIMELINE,
            latency_ms=40,
            cpu_percent=5,
            memory_mb=256,
            attention=EnvironmentAttentionRequirement.AMBIENT,
            priority=EnvironmentWorkerPriority.NORMAL,
            restart=EnvironmentRestartPolicy.ON_FAILURE,
            circuit=EnvironmentCircuitBreakerPolicy.OPEN_ON_FAILURE_RATE,
            failure=EnvironmentFailurePolicy.DEGRADE_GRACEFULLY,
        ),
        _worker(
            kind=EnvironmentWorkerKind.TRUST_CALIBRATION,
            name="TrustCalibrationWorker",
            subsystem=EnvironmentSubsystem.FOUNDATION,
            capability=EnvironmentWorkerCapability.CALIBRATE_TRUST,
            latency_ms=20,
            cpu_percent=4,
            memory_mb=96,
            attention=EnvironmentAttentionRequirement.NONE,
            priority=EnvironmentWorkerPriority.HIGH,
            restart=EnvironmentRestartPolicy.ALWAYS,
            circuit=EnvironmentCircuitBreakerPolicy.OPEN_ON_POLICY_VIOLATION,
            failure=EnvironmentFailurePolicy.FAIL_FAST,
        ),
        _worker(
            kind=EnvironmentWorkerKind.VISUAL_PRIORITY,
            name="VisualPriorityWorker",
            subsystem=EnvironmentSubsystem.FOUNDATION,
            capability=EnvironmentWorkerCapability.ARBITRATE_PRIORITY,
            latency_ms=20,
            cpu_percent=4,
            memory_mb=96,
            attention=EnvironmentAttentionRequirement.NONE,
            priority=EnvironmentWorkerPriority.CRITICAL,
            restart=EnvironmentRestartPolicy.ALWAYS,
            circuit=EnvironmentCircuitBreakerPolicy.OPEN_ON_LATENCY_BREACH,
            failure=EnvironmentFailurePolicy.FAIL_FAST,
        ),
        _worker(
            kind=EnvironmentWorkerKind.WORKSPACE_GRAPH,
            name="WorkspaceGraphWorker",
            subsystem=EnvironmentSubsystem.ENVIRONMENT_MAPPING,
            capability=EnvironmentWorkerCapability.BUILD_WORKSPACE_GRAPH,
            latency_ms=200,
            cpu_percent=10,
            memory_mb=512,
            attention=EnvironmentAttentionRequirement.AMBIENT,
            priority=EnvironmentWorkerPriority.NORMAL,
            restart=EnvironmentRestartPolicy.ON_FAILURE,
            circuit=EnvironmentCircuitBreakerPolicy.OPEN_ON_LATENCY_BREACH,
            failure=EnvironmentFailurePolicy.DEGRADE_GRACEFULLY,
            background=True,
        ),
        _worker(
            kind=EnvironmentWorkerKind.UI_SEMANTIC,
            name="UISemanticWorker",
            subsystem=EnvironmentSubsystem.UI_COGNITION,
            capability=EnvironmentWorkerCapability.UNDERSTAND_UI_SEMANTICS,
            latency_ms=250,
            cpu_percent=16,
            memory_mb=384,
            attention=EnvironmentAttentionRequirement.FOCUSED,
            priority=EnvironmentWorkerPriority.NORMAL,
            restart=EnvironmentRestartPolicy.ON_FAILURE,
            circuit=EnvironmentCircuitBreakerPolicy.OPEN_ON_LATENCY_BREACH,
            failure=EnvironmentFailurePolicy.DEGRADE_GRACEFULLY,
            background=True,
        ),
        _worker(
            kind=EnvironmentWorkerKind.VISUAL_GROUNDING,
            name="VisualGroundingWorker",
            subsystem=EnvironmentSubsystem.UI_COGNITION,
            capability=EnvironmentWorkerCapability.GROUND_VISUAL_TARGETS,
            latency_ms=150,
            cpu_percent=10,
            memory_mb=256,
            attention=EnvironmentAttentionRequirement.FOCUSED,
            priority=EnvironmentWorkerPriority.HIGH,
            restart=EnvironmentRestartPolicy.ON_FAILURE,
            circuit=EnvironmentCircuitBreakerPolicy.OPEN_ON_POLICY_VIOLATION,
            failure=EnvironmentFailurePolicy.ESCALATE_TO_USER,
        ),
        _worker(
            kind=EnvironmentWorkerKind.INTENT_PERSISTENCE,
            name="IntentPersistenceWorker",
            subsystem=EnvironmentSubsystem.FUSION_INTENT,
            capability=EnvironmentWorkerCapability.PERSIST_INTENT,
            latency_ms=40,
            cpu_percent=4,
            memory_mb=128,
            attention=EnvironmentAttentionRequirement.NONE,
            priority=EnvironmentWorkerPriority.NORMAL,
            restart=EnvironmentRestartPolicy.ON_FAILURE,
            circuit=EnvironmentCircuitBreakerPolicy.OPEN_ON_FAILURE_RATE,
            failure=EnvironmentFailurePolicy.DEGRADE_GRACEFULLY,
        ),
        _worker(
            kind=EnvironmentWorkerKind.SIMULATION,
            name="SimulationWorker",
            subsystem=EnvironmentSubsystem.SIMULATION_PLANNING,
            capability=EnvironmentWorkerCapability.SIMULATE_OUTCOME,
            latency_ms=200,
            cpu_percent=12,
            memory_mb=256,
            attention=EnvironmentAttentionRequirement.FOCUSED,
            priority=EnvironmentWorkerPriority.HIGH,
            restart=EnvironmentRestartPolicy.ON_FAILURE,
            circuit=EnvironmentCircuitBreakerPolicy.OPEN_ON_POLICY_VIOLATION,
            failure=EnvironmentFailurePolicy.ESCALATE_TO_USER,
        ),
        _worker(
            kind=EnvironmentWorkerKind.INTERACTION,
            name="InteractionWorker",
            subsystem=EnvironmentSubsystem.SAFE_INTERACTION,
            capability=EnvironmentWorkerCapability.INTERACT_WITH_ENVIRONMENT,
            latency_ms=100,
            cpu_percent=6,
            memory_mb=128,
            attention=EnvironmentAttentionRequirement.FOCUSED,
            priority=EnvironmentWorkerPriority.CRITICAL,
            restart=EnvironmentRestartPolicy.NEVER,
            circuit=EnvironmentCircuitBreakerPolicy.OPEN_ON_POLICY_VIOLATION,
            failure=EnvironmentFailurePolicy.ESCALATE_TO_USER,
        ),
        _worker(
            kind=EnvironmentWorkerKind.VERIFICATION,
            name="VerificationWorker",
            subsystem=EnvironmentSubsystem.VERIFICATION_RECOVERY,
            capability=EnvironmentWorkerCapability.VERIFY_ACTION,
            latency_ms=300,
            cpu_percent=12,
            memory_mb=256,
            attention=EnvironmentAttentionRequirement.FOCUSED,
            priority=EnvironmentWorkerPriority.CRITICAL,
            restart=EnvironmentRestartPolicy.ON_FAILURE,
            circuit=EnvironmentCircuitBreakerPolicy.OPEN_ON_LATENCY_BREACH,
            failure=EnvironmentFailurePolicy.FAIL_FAST,
        ),
        _worker(
            kind=EnvironmentWorkerKind.RECOVERY,
            name="RecoveryWorker",
            subsystem=EnvironmentSubsystem.VERIFICATION_RECOVERY,
            capability=EnvironmentWorkerCapability.RECOVER_FROM_FAILURE,
            latency_ms=500,
            cpu_percent=10,
            memory_mb=256,
            attention=EnvironmentAttentionRequirement.FOCUSED,
            priority=EnvironmentWorkerPriority.HIGH,
            restart=EnvironmentRestartPolicy.ON_FAILURE,
            circuit=EnvironmentCircuitBreakerPolicy.OPEN_ON_FAILURE_RATE,
            failure=EnvironmentFailurePolicy.ESCALATE_TO_USER,
        ),
        _worker(
            kind=EnvironmentWorkerKind.ENVIRONMENT_MEMORY,
            name="EnvironmentMemoryWorker",
            subsystem=EnvironmentSubsystem.MEMORY_COLLABORATION,
            capability=EnvironmentWorkerCapability.STORE_ENVIRONMENT_MEMORY,
            latency_ms=100,
            cpu_percent=6,
            memory_mb=256,
            attention=EnvironmentAttentionRequirement.NONE,
            priority=EnvironmentWorkerPriority.LOW,
            restart=EnvironmentRestartPolicy.ON_FAILURE,
            circuit=EnvironmentCircuitBreakerPolicy.OPEN_ON_POLICY_VIOLATION,
            failure=EnvironmentFailurePolicy.DEGRADE_GRACEFULLY,
            background=True,
        ),
        _worker(
            kind=EnvironmentWorkerKind.HUMAN_COLLABORATION,
            name="HumanCollaborationWorker",
            subsystem=EnvironmentSubsystem.MEMORY_COLLABORATION,
            capability=EnvironmentWorkerCapability.COLLABORATE_WITH_HUMAN,
            latency_ms=80,
            cpu_percent=4,
            memory_mb=128,
            attention=EnvironmentAttentionRequirement.AMBIENT,
            priority=EnvironmentWorkerPriority.NORMAL,
            restart=EnvironmentRestartPolicy.ON_FAILURE,
            circuit=EnvironmentCircuitBreakerPolicy.OPEN_ON_FAILURE_RATE,
            failure=EnvironmentFailurePolicy.DEGRADE_GRACEFULLY,
        ),
        _worker(
            kind=EnvironmentWorkerKind.SECURITY_AUDIT,
            name="SecurityAuditWorker",
            subsystem=EnvironmentSubsystem.SECURITY,
            capability=EnvironmentWorkerCapability.AUDIT_ENVIRONMENT_SECURITY,
            latency_ms=150,
            cpu_percent=8,
            memory_mb=256,
            attention=EnvironmentAttentionRequirement.NONE,
            priority=EnvironmentWorkerPriority.HIGH,
            restart=EnvironmentRestartPolicy.ALWAYS,
            circuit=EnvironmentCircuitBreakerPolicy.OPEN_ON_POLICY_VIOLATION,
            failure=EnvironmentFailurePolicy.FAIL_FAST,
        ),
    )


def _worker(
    *,
    kind: EnvironmentWorkerKind,
    name: str,
    subsystem: EnvironmentSubsystem,
    capability: EnvironmentWorkerCapability,
    latency_ms: float,
    cpu_percent: float,
    memory_mb: float,
    attention: EnvironmentAttentionRequirement,
    priority: EnvironmentWorkerPriority,
    restart: EnvironmentRestartPolicy,
    circuit: EnvironmentCircuitBreakerPolicy,
    failure: EnvironmentFailurePolicy,
    background: bool = False,
) -> EnvironmentWorkerDescriptor:
    return EnvironmentWorkerDescriptor(
        kind=kind,
        name=name,
        subsystem=subsystem,
        capability=capability,
        budget=EnvironmentWorkerBudget(
            latency_budget_ms=latency_ms,
            cpu_budget_percent=cpu_percent,
            memory_budget_mb=memory_mb,
            background_allowed=background,
        ),
        attention_requirement=attention,
        priority=priority,
        restart_policy=restart,
        circuit_breaker_policy=circuit,
        failure_policy=failure,
        health=EnvironmentWorkerHealth.HEALTHY,
    )