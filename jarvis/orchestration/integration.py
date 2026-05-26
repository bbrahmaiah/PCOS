from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class IntegratedPhase(StrEnum):
    """
    Previous JARVIS phases that become orchestrated participants.

    These are not direct modules anymore. They are phase-level integration
    boundaries represented to Phase 6 through adapters.
    """

    PRESENCE = "presence"
    COGNITION = "cognition"
    MEMORY = "memory"
    TOOLS = "tools"
    ATTENTION = "attention"
    BACKGROUND = "background"


class IntegratedWorkerKind(StrEnum):
    """
    Worker identity exposed to the Orchestration Kernel.

    The kernel coordinates workers. It does not know raw phase internals.
    """

    PRESENCE_WORKER = "presence_worker"
    COGNITION_WORKER = "cognition_worker"
    MEMORY_WORKER = "memory_worker"
    TOOL_WORKER = "tool_worker"
    ATTENTION_WORKER = "attention_worker"
    BACKGROUND_WORKER = "background_worker"


class PhaseEventKind(StrEnum):
    """
    Cross-phase event kinds entering integration.

    These are phase-level signals before they become orchestration tasks.
    """

    USER_SPEECH_STARTED = "user_speech_started"
    USER_TURN_FINALIZED = "user_turn_finalized"
    USER_INTERRUPTED = "user_interrupted"
    COGNITION_REQUESTED = "cognition_requested"
    MEMORY_REQUESTED = "memory_requested"
    TOOL_REQUESTED = "tool_requested"
    ATTENTION_REQUESTED = "attention_requested"
    BACKGROUND_REQUESTED = "background_requested"
    WORKER_RESULT_READY = "worker_result_ready"


class IntegratedTaskKind(StrEnum):
    """
    Orchestration task kind produced by adapters.
    """

    PRESENCE_SIGNAL = "presence_signal"
    TURN_COORDINATION = "turn_coordination"
    INTERRUPT_COORDINATION = "interrupt_coordination"
    COGNITION_TASK = "cognition_task"
    MEMORY_TASK = "memory_task"
    TOOL_TASK = "tool_task"
    ATTENTION_TASK = "attention_task"
    BACKGROUND_TASK = "background_task"
    RESULT_COLLECTION = "result_collection"


class PhaseIntegrationStatus(StrEnum):
    """
    Phase integration lifecycle state.
    """

    UNREGISTERED = "unregistered"
    REGISTERED = "registered"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    ISOLATED = "isolated"
    DISABLED = "disabled"


class PhaseIntegrationReason(StrEnum):
    """
    Machine-readable integration reason.
    """

    ADAPTER_REGISTERED = "adapter_registered"
    ADAPTER_REPLACED = "adapter_replaced"
    ADAPTER_NOT_FOUND = "adapter_not_found"
    ADAPTER_DISABLED = "adapter_disabled"
    ADAPTER_HEALTHY = "adapter_healthy"
    ADAPTER_DEGRADED = "adapter_degraded"
    PHASE_EVENT_ACCEPTED = "phase_event_accepted"
    PHASE_EVENT_REJECTED = "phase_event_rejected"
    TASK_ENVELOPE_CREATED = "task_envelope_created"
    TASK_ENVELOPE_REJECTED = "task_envelope_rejected"
    BOUNDARY_PRESERVED = "boundary_preserved"
    DIRECT_EXECUTION_BLOCKED = "direct_execution_blocked"
    INTEGRATION_SNAPSHOT_CREATED = "integration_snapshot_created"
    DEFAULT_ADAPTERS_REGISTERED = "default_adapters_registered"
    RUNTIME_RESET = "runtime_reset"


class BoundaryViolationKind(StrEnum):
    """
    Direct-coupling violations Step 16 must prevent.
    """

    PRESENCE_TO_COGNITION_DIRECT = "presence_to_cognition_direct"
    COGNITION_TO_MEMORY_DIRECT = "cognition_to_memory_direct"
    COGNITION_TO_TOOL_DIRECT = "cognition_to_tool_direct"
    MEMORY_TO_CONTEXT_DIRECT = "memory_to_context_direct"
    TOOL_EXECUTION_DIRECT = "tool_execution_direct"
    WORKER_TO_WORKER_DIRECT = "worker_to_worker_direct"


class PhaseEvent(OrchestrationModel):
    """
    Input event from a previous phase.

    This is the first boundary object. Raw phases become events before they
    become orchestration tasks.
    """

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    source_phase: IntegratedPhase
    event_kind: PhaseEventKind
    payload: dict[str, Any] = Field(default_factory=dict)
    requires_context_snapshot: bool = True
    direct_execution_requested: bool = False
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_event_id(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("event_id cannot be empty.")

        return cleaned


class IntegratedTaskEnvelope(OrchestrationModel):
    """
    Task envelope sent into orchestration.

    This is not execution. It is a governed task request that downstream
    scheduler/budget/coordination runtimes may accept or reject.
    """

    envelope_id: str = Field(default_factory=lambda: uuid4().hex)
    source_event_id: str
    source_phase: IntegratedPhase
    target_worker: IntegratedWorkerKind
    task_kind: IntegratedTaskKind
    payload: dict[str, Any] = Field(default_factory=dict)
    requires_budget: bool = True
    requires_context_snapshot: bool = True
    requires_policy: bool = True
    interruptible: bool = True
    direct_execution_allowed: bool = False
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("envelope_id", "source_event_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @model_validator(mode="after")
    def _validate_boundary_rules(self) -> IntegratedTaskEnvelope:
        if self.direct_execution_allowed:
            raise ValueError("integrated task envelopes cannot execute directly.")

        if self.task_kind == IntegratedTaskKind.TOOL_TASK:
            if not self.requires_policy:
                raise ValueError("tool task envelopes must require policy.")

            if not self.requires_budget:
                raise ValueError("tool task envelopes must require budget.")

        if self.task_kind == IntegratedTaskKind.COGNITION_TASK:
            if not self.requires_context_snapshot:
                raise ValueError(
                    "cognition task envelopes must require context snapshot."
                )

        return self


class PhaseAdapterCapability(OrchestrationModel):
    """
    Declared adapter capabilities.

    Capabilities are static and inspectable. No hidden routing.
    """

    accepted_events: tuple[PhaseEventKind, ...]
    emitted_tasks: tuple[IntegratedTaskKind, ...]
    supports_interrupt: bool = True
    supports_recovery: bool = True
    supports_observability: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_capabilities(self) -> PhaseAdapterCapability:
        if not self.accepted_events:
            raise ValueError("adapter must accept at least one event kind.")

        if not self.emitted_tasks:
            raise ValueError("adapter must emit at least one task kind.")

        return self


class PhaseAdapterHealth(OrchestrationModel):
    """
    Adapter health view.
    """

    phase: IntegratedPhase
    worker_kind: IntegratedWorkerKind
    status: PhaseIntegrationStatus
    accepted_count: int = Field(default=0, ge=0)
    rejected_count: int = Field(default=0, ge=0)
    emitted_count: int = Field(default=0, ge=0)
    last_reason: PhaseIntegrationReason | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PhaseIntegrationResult(OrchestrationModel):
    """
    Result of an integration operation.
    """

    reason: PhaseIntegrationReason
    success: bool
    message: str
    event: PhaseEvent | None = None
    envelope: IntegratedTaskEnvelope | None = None
    health: PhaseAdapterHealth | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("message")
    @classmethod
    def _required_message(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("message cannot be empty.")

        return cleaned


class PhaseIntegrationSnapshot(OrchestrationModel):
    """
    Runtime integration diagnostics.
    """

    name: str
    adapter_count: int = Field(ge=0)
    healthy_count: int = Field(default=0, ge=0)
    degraded_count: int = Field(default=0, ge=0)
    isolated_count: int = Field(default=0, ge=0)
    routed_event_count: int = Field(default=0, ge=0)
    emitted_envelope_count: int = Field(default=0, ge=0)
    rejected_event_count: int = Field(default=0, ge=0)
    direct_execution_block_count: int = Field(default=0, ge=0)
    last_reason: PhaseIntegrationReason | None = None
    adapters: tuple[PhaseAdapterHealth, ...] = ()
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class PhaseIntegrationConfig:
    """
    Phase integration runtime configuration.
    """

    name: str = "phase_integration_runtime"
    auto_register_defaults: bool = True
    block_direct_execution: bool = True
    require_context_snapshot_for_cognition: bool = True
    require_policy_for_tools: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")


class PhaseWorkerAdapter:
    """
    Base adapter that converts phase events into orchestration task envelopes.

    Adapters are translators only:
    - no execution
    - no direct runtime calls
    - no cross-worker calls
    """

    def __init__(
        self,
        *,
        phase: IntegratedPhase,
        worker_kind: IntegratedWorkerKind,
        capabilities: PhaseAdapterCapability,
        event_task_map: dict[PhaseEventKind, IntegratedTaskKind],
        status: PhaseIntegrationStatus = PhaseIntegrationStatus.REGISTERED,
    ) -> None:
        self._phase = phase
        self._worker_kind = worker_kind
        self._capabilities = capabilities
        self._event_task_map = dict(event_task_map)
        self._status = status
        self._lock = RLock()

        self._accepted_count = 0
        self._rejected_count = 0
        self._emitted_count = 0
        self._last_reason: PhaseIntegrationReason | None = None

        self._validate_mapping()

    @property
    def phase(self) -> IntegratedPhase:
        return self._phase

    @property
    def worker_kind(self) -> IntegratedWorkerKind:
        return self._worker_kind

    @property
    def capabilities(self) -> PhaseAdapterCapability:
        return self._capabilities

    @property
    def status(self) -> PhaseIntegrationStatus:
        return self._status

    def can_accept(self, event: PhaseEvent) -> bool:
        return (
            event.source_phase == self._phase
            and event.event_kind in self._capabilities.accepted_events
            and self._status
            not in {
                PhaseIntegrationStatus.DISABLED,
                PhaseIntegrationStatus.ISOLATED,
            }
        )

    def build_envelope(
        self,
        event: PhaseEvent,
    ) -> PhaseIntegrationResult:
        if not self.can_accept(event):
            with self._lock:
                self._rejected_count += 1
                self._last_reason = PhaseIntegrationReason.PHASE_EVENT_REJECTED

            return PhaseIntegrationResult(
                reason=PhaseIntegrationReason.PHASE_EVENT_REJECTED,
                success=False,
                message="phase adapter rejected event",
                event=event,
                health=self.health(),
            )

        if event.direct_execution_requested:
            with self._lock:
                self._rejected_count += 1
                self._last_reason = (
                    PhaseIntegrationReason.DIRECT_EXECUTION_BLOCKED
                )

            return PhaseIntegrationResult(
                reason=PhaseIntegrationReason.DIRECT_EXECUTION_BLOCKED,
                success=False,
                message="direct execution request blocked by integration layer",
                event=event,
                health=self.health(),
                metadata={
                    "violation": self._violation_for_event(event).value,
                },
            )

        task_kind = self._event_task_map[event.event_kind]
        envelope = IntegratedTaskEnvelope(
            source_event_id=event.event_id,
            source_phase=event.source_phase,
            target_worker=self._worker_kind,
            task_kind=task_kind,
            payload=dict(event.payload),
            requires_context_snapshot=self._requires_snapshot(task_kind, event),
            requires_policy=self._requires_policy(task_kind),
            requires_budget=True,
            interruptible=True,
            direct_execution_allowed=False,
            metadata={
                "adapter_phase": self._phase.value,
                "boundary": "phase_adapter",
            },
        )

        with self._lock:
            self._accepted_count += 1
            self._emitted_count += 1
            self._last_reason = PhaseIntegrationReason.TASK_ENVELOPE_CREATED

        return PhaseIntegrationResult(
            reason=PhaseIntegrationReason.TASK_ENVELOPE_CREATED,
            success=True,
            message="phase event converted to orchestration task envelope",
            event=event,
            envelope=envelope,
            health=self.health(),
        )

    def mark_healthy(self) -> PhaseAdapterHealth:
        with self._lock:
            self._status = PhaseIntegrationStatus.HEALTHY
            self._last_reason = PhaseIntegrationReason.ADAPTER_HEALTHY

        return self.health()

    def mark_degraded(self) -> PhaseAdapterHealth:
        with self._lock:
            self._status = PhaseIntegrationStatus.DEGRADED
            self._last_reason = PhaseIntegrationReason.ADAPTER_DEGRADED

        return self.health()

    def isolate(self) -> PhaseAdapterHealth:
        with self._lock:
            self._status = PhaseIntegrationStatus.ISOLATED
            self._last_reason = PhaseIntegrationReason.ADAPTER_DISABLED

        return self.health()

    def disable(self) -> PhaseAdapterHealth:
        with self._lock:
            self._status = PhaseIntegrationStatus.DISABLED
            self._last_reason = PhaseIntegrationReason.ADAPTER_DISABLED

        return self.health()

    def health(self) -> PhaseAdapterHealth:
        with self._lock:
            return PhaseAdapterHealth(
                phase=self._phase,
                worker_kind=self._worker_kind,
                status=self._status,
                accepted_count=self._accepted_count,
                rejected_count=self._rejected_count,
                emitted_count=self._emitted_count,
                last_reason=self._last_reason,
            )

    def _validate_mapping(self) -> None:
        accepted = set(self._capabilities.accepted_events)
        emitted = set(self._capabilities.emitted_tasks)

        for event_kind, task_kind in self._event_task_map.items():
            if event_kind not in accepted:
                raise ValueError(
                    "event_task_map contains event not declared in capabilities."
                )

            if task_kind not in emitted:
                raise ValueError(
                    "event_task_map emits task not declared in capabilities."
                )

    @staticmethod
    def _requires_policy(task_kind: IntegratedTaskKind) -> bool:
        return task_kind == IntegratedTaskKind.TOOL_TASK

    @staticmethod
    def _requires_snapshot(
        task_kind: IntegratedTaskKind,
        event: PhaseEvent,
    ) -> bool:
        if task_kind in {
            IntegratedTaskKind.COGNITION_TASK,
            IntegratedTaskKind.MEMORY_TASK,
            IntegratedTaskKind.TOOL_TASK,
            IntegratedTaskKind.TURN_COORDINATION,
        }:
            return True

        return event.requires_context_snapshot

    @staticmethod
    def _violation_for_event(event: PhaseEvent) -> BoundaryViolationKind:
        if event.source_phase == IntegratedPhase.PRESENCE:
            return BoundaryViolationKind.PRESENCE_TO_COGNITION_DIRECT

        if event.source_phase == IntegratedPhase.COGNITION:
            if event.event_kind == PhaseEventKind.TOOL_REQUESTED:
                return BoundaryViolationKind.COGNITION_TO_TOOL_DIRECT

            return BoundaryViolationKind.COGNITION_TO_MEMORY_DIRECT

        if event.source_phase == IntegratedPhase.MEMORY:
            return BoundaryViolationKind.MEMORY_TO_CONTEXT_DIRECT

        if event.source_phase == IntegratedPhase.TOOLS:
            return BoundaryViolationKind.TOOL_EXECUTION_DIRECT

        return BoundaryViolationKind.WORKER_TO_WORKER_DIRECT


class PresencePhaseAdapter(PhaseWorkerAdapter):
    """
    Phase 2 Presence Runtime adapter.

    Converts voice/presence signals into orchestration events.
    """

    def __init__(self) -> None:
        super().__init__(
            phase=IntegratedPhase.PRESENCE,
            worker_kind=IntegratedWorkerKind.PRESENCE_WORKER,
            capabilities=PhaseAdapterCapability(
                accepted_events=(
                    PhaseEventKind.USER_SPEECH_STARTED,
                    PhaseEventKind.USER_TURN_FINALIZED,
                    PhaseEventKind.USER_INTERRUPTED,
                ),
                emitted_tasks=(
                    IntegratedTaskKind.PRESENCE_SIGNAL,
                    IntegratedTaskKind.TURN_COORDINATION,
                    IntegratedTaskKind.INTERRUPT_COORDINATION,
                ),
            ),
            event_task_map={
                PhaseEventKind.USER_SPEECH_STARTED: (
                    IntegratedTaskKind.PRESENCE_SIGNAL
                ),
                PhaseEventKind.USER_TURN_FINALIZED: (
                    IntegratedTaskKind.TURN_COORDINATION
                ),
                PhaseEventKind.USER_INTERRUPTED: (
                    IntegratedTaskKind.INTERRUPT_COORDINATION
                ),
            },
        )


class CognitionPhaseAdapter(PhaseWorkerAdapter):
    """
    Phase 3 Cognition Runtime adapter.

    Cognition becomes a worker task target, not a direct caller of memory/tools.
    """

    def __init__(self) -> None:
        super().__init__(
            phase=IntegratedPhase.COGNITION,
            worker_kind=IntegratedWorkerKind.COGNITION_WORKER,
            capabilities=PhaseAdapterCapability(
                accepted_events=(PhaseEventKind.COGNITION_REQUESTED,),
                emitted_tasks=(IntegratedTaskKind.COGNITION_TASK,),
            ),
            event_task_map={
                PhaseEventKind.COGNITION_REQUESTED: (
                    IntegratedTaskKind.COGNITION_TASK
                ),
            },
        )


class MemoryPhaseAdapter(PhaseWorkerAdapter):
    """
    Phase 4 Memory Runtime adapter.

    Memory access is represented as orchestration work and must respect
    snapshots and policy boundaries.
    """

    def __init__(self) -> None:
        super().__init__(
            phase=IntegratedPhase.MEMORY,
            worker_kind=IntegratedWorkerKind.MEMORY_WORKER,
            capabilities=PhaseAdapterCapability(
                accepted_events=(PhaseEventKind.MEMORY_REQUESTED,),
                emitted_tasks=(IntegratedTaskKind.MEMORY_TASK,),
            ),
            event_task_map={
                PhaseEventKind.MEMORY_REQUESTED: IntegratedTaskKind.MEMORY_TASK,
            },
        )


class ToolPhaseAdapter(PhaseWorkerAdapter):
    """
    Phase 5 Tool Runtime adapter.

    Tools are never invoked directly from cognition. They become policy-guarded
    orchestration task envelopes.
    """

    def __init__(self) -> None:
        super().__init__(
            phase=IntegratedPhase.TOOLS,
            worker_kind=IntegratedWorkerKind.TOOL_WORKER,
            capabilities=PhaseAdapterCapability(
                accepted_events=(PhaseEventKind.TOOL_REQUESTED,),
                emitted_tasks=(IntegratedTaskKind.TOOL_TASK,),
            ),
            event_task_map={
                PhaseEventKind.TOOL_REQUESTED: IntegratedTaskKind.TOOL_TASK,
            },
        )


class AttentionPhaseAdapter(PhaseWorkerAdapter):
    """
    Attention adapter.

    Attention participates as a worker-visible runtime while remaining the
    protection policy for active conversation.
    """

    def __init__(self) -> None:
        super().__init__(
            phase=IntegratedPhase.ATTENTION,
            worker_kind=IntegratedWorkerKind.ATTENTION_WORKER,
            capabilities=PhaseAdapterCapability(
                accepted_events=(PhaseEventKind.ATTENTION_REQUESTED,),
                emitted_tasks=(IntegratedTaskKind.ATTENTION_TASK,),
            ),
            event_task_map={
                PhaseEventKind.ATTENTION_REQUESTED: (
                    IntegratedTaskKind.ATTENTION_TASK
                ),
            },
        )


class BackgroundPhaseAdapter(PhaseWorkerAdapter):
    """
    Background adapter.

    Background work stays lowest priority and enters orchestration as
    cancellable work.
    """

    def __init__(self) -> None:
        super().__init__(
            phase=IntegratedPhase.BACKGROUND,
            worker_kind=IntegratedWorkerKind.BACKGROUND_WORKER,
            capabilities=PhaseAdapterCapability(
                accepted_events=(PhaseEventKind.BACKGROUND_REQUESTED,),
                emitted_tasks=(IntegratedTaskKind.BACKGROUND_TASK,),
            ),
            event_task_map={
                PhaseEventKind.BACKGROUND_REQUESTED: (
                    IntegratedTaskKind.BACKGROUND_TASK
                ),
            },
        )


class PhaseIntegrationRuntime:
    """
    Phase 6 Step 16 Full Phase 1-5 Integration Runtime.

    Responsibilities:
    - register phase adapters
    - convert phase events into orchestration task envelopes
    - enforce no direct execution
    - expose integration health and counts
    - preserve phase boundaries

    Non-responsibilities:
    - no STT/TTS execution
    - no LLM generation
    - no memory access
    - no tool execution
    - no scheduling
    - no worker-to-worker communication
    """

    def __init__(
        self,
        *,
        config: PhaseIntegrationConfig | None = None,
        adapters: tuple[PhaseWorkerAdapter, ...] = (),
    ) -> None:
        self._config = config or PhaseIntegrationConfig()
        self._config.validate()

        self._adapters: dict[IntegratedPhase, PhaseWorkerAdapter] = {}
        self._emitted_envelopes: list[IntegratedTaskEnvelope] = []
        self._lock = RLock()

        self._routed_event_count = 0
        self._rejected_event_count = 0
        self._direct_execution_block_count = 0
        self._last_reason: PhaseIntegrationReason | None = None

        if self._config.auto_register_defaults:
            self.register_default_adapters()

        for adapter in adapters:
            self.register_adapter(adapter)

    @property
    def name(self) -> str:
        return self._config.name

    def register_default_adapters(self) -> PhaseIntegrationResult:
        for adapter in (
            PresencePhaseAdapter(),
            CognitionPhaseAdapter(),
            MemoryPhaseAdapter(),
            ToolPhaseAdapter(),
            AttentionPhaseAdapter(),
            BackgroundPhaseAdapter(),
        ):
            self.register_adapter(adapter)

        with self._lock:
            self._last_reason = (
                PhaseIntegrationReason.DEFAULT_ADAPTERS_REGISTERED
            )

        return PhaseIntegrationResult(
            reason=PhaseIntegrationReason.DEFAULT_ADAPTERS_REGISTERED,
            success=True,
            message="default phase adapters registered",
        )

    def register_adapter(
        self,
        adapter: PhaseWorkerAdapter,
    ) -> PhaseIntegrationResult:
        with self._lock:
            replaced = adapter.phase in self._adapters
            self._adapters[adapter.phase] = adapter
            self._last_reason = (
                PhaseIntegrationReason.ADAPTER_REPLACED
                if replaced
                else PhaseIntegrationReason.ADAPTER_REGISTERED
            )

        adapter.mark_healthy()

        return PhaseIntegrationResult(
            reason=(
                PhaseIntegrationReason.ADAPTER_REPLACED
                if replaced
                else PhaseIntegrationReason.ADAPTER_REGISTERED
            ),
            success=True,
            message=(
                "phase adapter replaced"
                if replaced
                else "phase adapter registered"
            ),
            health=adapter.health(),
        )

    def route_event(self, event: PhaseEvent) -> PhaseIntegrationResult:
        if (
            self._config.block_direct_execution
            and event.direct_execution_requested
        ):
            with self._lock:
                self._rejected_event_count += 1
                self._direct_execution_block_count += 1
                self._last_reason = (
                    PhaseIntegrationReason.DIRECT_EXECUTION_BLOCKED
                )

            return PhaseIntegrationResult(
                reason=PhaseIntegrationReason.DIRECT_EXECUTION_BLOCKED,
                success=False,
                message="direct execution blocked before adapter routing",
                event=event,
            )

        with self._lock:
            adapter = self._adapters.get(event.source_phase)

        if adapter is None:
            with self._lock:
                self._rejected_event_count += 1
                self._last_reason = PhaseIntegrationReason.ADAPTER_NOT_FOUND

            return PhaseIntegrationResult(
                reason=PhaseIntegrationReason.ADAPTER_NOT_FOUND,
                success=False,
                message="no adapter registered for phase",
                event=event,
            )

        result = adapter.build_envelope(event)

        with self._lock:
            if result.success and result.envelope is not None:
                self._routed_event_count += 1
                self._emitted_envelopes.append(result.envelope)
                self._last_reason = PhaseIntegrationReason.TASK_ENVELOPE_CREATED
            else:
                self._rejected_event_count += 1
                self._last_reason = result.reason

                if (
                    result.reason
                    == PhaseIntegrationReason.DIRECT_EXECUTION_BLOCKED
                ):
                    self._direct_execution_block_count += 1

        return result

    def route_events(
        self,
        events: tuple[PhaseEvent, ...],
    ) -> tuple[PhaseIntegrationResult, ...]:
        return tuple(self.route_event(event) for event in events)

    def emitted_envelopes(self) -> tuple[IntegratedTaskEnvelope, ...]:
        with self._lock:
            return tuple(self._emitted_envelopes)

    def adapter_health(
        self,
        phase: IntegratedPhase,
    ) -> PhaseAdapterHealth | None:
        with self._lock:
            adapter = self._adapters.get(phase)

        if adapter is None:
            return None

        return adapter.health()

    def mark_adapter_degraded(
        self,
        phase: IntegratedPhase,
    ) -> PhaseIntegrationResult:
        with self._lock:
            adapter = self._adapters.get(phase)

        if adapter is None:
            return PhaseIntegrationResult(
                reason=PhaseIntegrationReason.ADAPTER_NOT_FOUND,
                success=False,
                message="no adapter registered for phase",
            )

        health = adapter.mark_degraded()

        with self._lock:
            self._last_reason = PhaseIntegrationReason.ADAPTER_DEGRADED

        return PhaseIntegrationResult(
            reason=PhaseIntegrationReason.ADAPTER_DEGRADED,
            success=True,
            message="phase adapter marked degraded",
            health=health,
        )

    def isolate_adapter(
        self,
        phase: IntegratedPhase,
    ) -> PhaseIntegrationResult:
        with self._lock:
            adapter = self._adapters.get(phase)

        if adapter is None:
            return PhaseIntegrationResult(
                reason=PhaseIntegrationReason.ADAPTER_NOT_FOUND,
                success=False,
                message="no adapter registered for phase",
            )

        health = adapter.isolate()

        with self._lock:
            self._last_reason = PhaseIntegrationReason.ADAPTER_DISABLED

        return PhaseIntegrationResult(
            reason=PhaseIntegrationReason.ADAPTER_DISABLED,
            success=True,
            message="phase adapter isolated",
            health=health,
        )

    def snapshot(self) -> PhaseIntegrationSnapshot:
        with self._lock:
            adapter_health = tuple(
                adapter.health() for adapter in self._adapters.values()
            )
            healthy_count = sum(
                1
                for item in adapter_health
                if item.status == PhaseIntegrationStatus.HEALTHY
            )
            degraded_count = sum(
                1
                for item in adapter_health
                if item.status == PhaseIntegrationStatus.DEGRADED
            )
            isolated_count = sum(
                1
                for item in adapter_health
                if item.status == PhaseIntegrationStatus.ISOLATED
            )

            return PhaseIntegrationSnapshot(
                name=self.name,
                adapter_count=len(self._adapters),
                healthy_count=healthy_count,
                degraded_count=degraded_count,
                isolated_count=isolated_count,
                routed_event_count=self._routed_event_count,
                emitted_envelope_count=len(self._emitted_envelopes),
                rejected_event_count=self._rejected_event_count,
                direct_execution_block_count=(
                    self._direct_execution_block_count
                ),
                last_reason=self._last_reason,
                adapters=adapter_health,
            )

    def reset(self) -> None:
        with self._lock:
            self._emitted_envelopes.clear()
            self._routed_event_count = 0
            self._rejected_event_count = 0
            self._direct_execution_block_count = 0
            self._last_reason = PhaseIntegrationReason.RUNTIME_RESET