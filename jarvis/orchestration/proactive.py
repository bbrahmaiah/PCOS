from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.integration import (
    IntegratedPhase,
    IntegratedTaskEnvelope,
    IntegratedTaskKind,
    IntegratedWorkerKind,
)
from jarvis.orchestration.models import OrchestrationModel


class ProactiveTriggerKind(StrEnum):
    """
    Safe triggers that may start proactive preparation.

    These are signals, not permissions to act.
    """

    USER_PAUSED = "user_paused"
    CONVERSATION_IDLE = "conversation_idle"
    BUILD_RUNNING = "build_running"
    TEST_RUNNING = "test_running"
    FILE_CHANGED = "file_changed"
    WORKSPACE_CHANGED = "workspace_changed"
    ERROR_PATTERN_SEEN = "error_pattern_seen"


class ProactiveWorkKind(StrEnum):
    """
    Low-risk proactive work types.

    All work here must remain read-only and cancellable.
    """

    MEMORY_PREFETCH = "memory_prefetch"
    CONTEXT_PREWARM = "context_prewarm"
    TOOL_PATH_PREWARM = "tool_path_prewarm"
    BUILD_MONITORING = "build_monitoring"
    TEST_MONITORING = "test_monitoring"
    WORKSPACE_NOTE = "workspace_note"
    ERROR_CONTEXT_PREPARE = "error_context_prepare"
    LOW_RISK_SUGGESTION = "low_risk_suggestion"


class ProactiveRiskLevel(IntEnum):
    """
    Proactive risk classification.

    Only LOW is executable as proactive orchestration preparation.
    ACTION is intentionally blocked.
    """

    LOW = 0
    MEDIUM = 1
    HIGH = 2
    ACTION = 3


class ProactiveStatus(StrEnum):
    """
    Proactive lifecycle status.
    """

    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    SUPPRESSED = "suppressed"
    BLOCKED = "blocked"
    QUEUED = "queued"
    CANCELLED = "cancelled"
    SURFACED = "surfaced"


class ProactiveReason(StrEnum):
    """
    Machine-readable proactive orchestration reason.
    """

    TRIGGER_ACCEPTED = "trigger_accepted"
    TRIGGER_SUPPRESSED = "trigger_suppressed"
    CONVERSATION_ACTIVE_SUPPRESSED = "conversation_active_suppressed"
    LOW_CONFIDENCE_SUPPRESSED = "low_confidence_suppressed"
    RISK_BLOCKED = "risk_blocked"
    ACTION_BLOCKED = "action_blocked"
    ENVELOPE_CREATED = "envelope_created"
    SUGGESTION_CREATED = "suggestion_created"
    SUGGESTION_SUPPRESSED = "suggestion_suppressed"
    PROACTIVE_BATCH_CREATED = "proactive_batch_created"
    PROACTIVE_CANCELLED = "proactive_cancelled"
    RUNTIME_RESET = "runtime_reset"


class ProactiveTrigger(OrchestrationModel):
    """
    Runtime signal that may produce proactive preparation.

    A trigger never executes work directly.
    """

    trigger_id: str = Field(default_factory=lambda: uuid4().hex)
    kind: ProactiveTriggerKind
    payload: dict[str, Any] = Field(default_factory=dict)
    confidence_percent: int = Field(default=50, ge=0, le=100)
    conversation_active: bool = False
    user_visible: bool = False
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("trigger_id")
    @classmethod
    def _required_trigger_id(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("trigger_id cannot be empty.")

        return cleaned


class ProactiveTaskEnvelope(OrchestrationModel):
    """
    Proactive preparation envelope.

    This is not execution. It is a governed task request that must still pass
    orchestration, budget, attention, and cancellation policies.
    """

    envelope_id: str = Field(default_factory=lambda: uuid4().hex)
    trigger_id: str
    work_kind: ProactiveWorkKind
    target_worker: IntegratedWorkerKind
    task_kind: IntegratedTaskKind
    payload: dict[str, Any] = Field(default_factory=dict)
    risk_level: ProactiveRiskLevel = ProactiveRiskLevel.LOW
    confidence_percent: int = Field(ge=0, le=100)
    read_only: bool = True
    cancellable: bool = True
    lower_than_reactive: bool = True
    action_allowed: bool = False
    requires_budget: bool = True
    requires_context_snapshot: bool = True
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("envelope_id", "trigger_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @model_validator(mode="after")
    def _enforce_proactive_law(self) -> ProactiveTaskEnvelope:
        if self.risk_level != ProactiveRiskLevel.LOW:
            raise ValueError("proactive work must be low risk.")

        if not self.read_only:
            raise ValueError("proactive work must be read-only.")

        if not self.cancellable:
            raise ValueError("proactive work must be cancellable.")

        if not self.lower_than_reactive:
            raise ValueError("proactive work must be lower than reactive work.")

        if self.action_allowed:
            raise ValueError("proactive orchestration can never allow actions.")

        if self.work_kind == ProactiveWorkKind.TOOL_PATH_PREWARM:
            if self.target_worker == IntegratedWorkerKind.TOOL_WORKER:
                raise ValueError(
                    "tool path prewarm cannot target tool execution worker."
                )

        return self

    def to_integrated_envelope(self) -> IntegratedTaskEnvelope:
        """
        Convert proactive envelope into Step 16 integration envelope.

        It remains non-executing, policy-bound, budget-bound, and interruptible.
        """

        return IntegratedTaskEnvelope(
            source_event_id=self.trigger_id,
            source_phase=IntegratedPhase.BACKGROUND,
            target_worker=self.target_worker,
            task_kind=self.task_kind,
            payload=dict(self.payload),
            requires_budget=self.requires_budget,
            requires_context_snapshot=self.requires_context_snapshot,
            requires_policy=True,
            interruptible=True,
            direct_execution_allowed=False,
            metadata={
                "proactive": True,
                "proactive_envelope_id": self.envelope_id,
                "work_kind": self.work_kind.value,
                "confidence_percent": self.confidence_percent,
            },
        )


class ProactiveSuggestion(OrchestrationModel):
    """
    Suggestion prepared proactively.

    Suggestions are surfaced only when confidence is high.
    """

    suggestion_id: str = Field(default_factory=lambda: uuid4().hex)
    trigger_id: str
    title: str
    message: str
    confidence_percent: int = Field(ge=0, le=100)
    surface_threshold_percent: int = Field(default=80, ge=0, le=100)
    risk_level: ProactiveRiskLevel = ProactiveRiskLevel.LOW
    read_only: bool = True
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("suggestion_id", "trigger_id", "title", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @model_validator(mode="after")
    def _validate_safe_suggestion(self) -> ProactiveSuggestion:
        if self.risk_level != ProactiveRiskLevel.LOW:
            raise ValueError("proactive suggestions must be low risk.")

        if not self.read_only:
            raise ValueError("proactive suggestions must be read-only.")

        return self

    @property
    def can_surface(self) -> bool:
        return self.confidence_percent >= self.surface_threshold_percent


class ProactiveDecision(OrchestrationModel):
    """
    Decision generated from a proactive trigger.
    """

    status: ProactiveStatus
    reason: ProactiveReason
    message: str
    trigger: ProactiveTrigger
    envelopes: tuple[ProactiveTaskEnvelope, ...] = ()
    suggestions: tuple[ProactiveSuggestion, ...] = ()
    user_visible_suggestions: tuple[ProactiveSuggestion, ...] = ()
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("message")
    @classmethod
    def _required_message(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("message cannot be empty.")

        return cleaned


class ProactiveResult(OrchestrationModel):
    """
    Result of proactive runtime operation.
    """

    reason: ProactiveReason
    success: bool
    message: str
    decision: ProactiveDecision | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("message")
    @classmethod
    def _required_message(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("message cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class ProactiveOrchestrationConfig:
    """
    Proactive Orchestration configuration.
    """

    name: str = "proactive_orchestration"
    minimum_trigger_confidence_percent: int = 50
    suggestion_surface_threshold_percent: int = 80
    max_envelopes_per_trigger: int = 3
    suppress_when_conversation_active: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if not 0 <= self.minimum_trigger_confidence_percent <= 100:
            raise ValueError(
                "minimum_trigger_confidence_percent must be within 0..100."
            )

        if not 0 <= self.suggestion_surface_threshold_percent <= 100:
            raise ValueError(
                "suggestion_surface_threshold_percent must be within 0..100."
            )

        if self.max_envelopes_per_trigger < 1:
            raise ValueError("max_envelopes_per_trigger must be positive.")


@dataclass(frozen=True, slots=True)
class ProactiveOrchestrationSnapshot:
    """
    Proactive runtime diagnostics.
    """

    name: str
    trigger_count: int
    decision_count: int
    envelope_count: int
    suggestion_count: int
    surfaced_suggestion_count: int
    suppressed_count: int
    blocked_count: int
    cancelled_count: int
    last_reason: ProactiveReason | None


class TriggerPolicy:
    """
    Converts safe triggers into proactive preparation work.

    The policy is intentionally conservative:
    - no actions
    - no writes
    - no direct tool execution
    - no high-risk work
    """

    def __init__(
        self,
        *,
        config: ProactiveOrchestrationConfig | None = None,
    ) -> None:
        self._config = config or ProactiveOrchestrationConfig()
        self._config.validate()

    def evaluate(self, trigger: ProactiveTrigger) -> ProactiveDecision:
        if (
            self._config.suppress_when_conversation_active
            and trigger.conversation_active
        ):
            return ProactiveDecision(
                status=ProactiveStatus.SUPPRESSED,
                reason=ProactiveReason.CONVERSATION_ACTIVE_SUPPRESSED,
                message="proactive work suppressed while conversation is active",
                trigger=trigger,
            )

        if (
            trigger.confidence_percent
            < self._config.minimum_trigger_confidence_percent
        ):
            return ProactiveDecision(
                status=ProactiveStatus.SUPPRESSED,
                reason=ProactiveReason.LOW_CONFIDENCE_SUPPRESSED,
                message="proactive trigger confidence is below threshold",
                trigger=trigger,
            )

        envelopes = self._build_envelopes(trigger)
        suggestions = self._build_suggestions(trigger)
        visible = tuple(
            item for item in suggestions if item.can_surface
        )

        return ProactiveDecision(
            status=ProactiveStatus.ACCEPTED,
            reason=ProactiveReason.PROACTIVE_BATCH_CREATED,
            message="proactive preparation batch created",
            trigger=trigger,
            envelopes=envelopes[: self._config.max_envelopes_per_trigger],
            suggestions=suggestions,
            user_visible_suggestions=visible,
            metadata={
                "envelope_count": len(envelopes),
                "visible_suggestion_count": len(visible),
            },
        )

    def _build_envelopes(
        self,
        trigger: ProactiveTrigger,
    ) -> tuple[ProactiveTaskEnvelope, ...]:
        if trigger.kind == ProactiveTriggerKind.USER_PAUSED:
            return (
                self._envelope(
                    trigger,
                    work_kind=ProactiveWorkKind.MEMORY_PREFETCH,
                    target_worker=IntegratedWorkerKind.MEMORY_WORKER,
                    task_kind=IntegratedTaskKind.MEMORY_TASK,
                    payload={"reason": "user paused; prefetch likely memory"},
                ),
                self._envelope(
                    trigger,
                    work_kind=ProactiveWorkKind.CONTEXT_PREWARM,
                    target_worker=IntegratedWorkerKind.BACKGROUND_WORKER,
                    task_kind=IntegratedTaskKind.BACKGROUND_TASK,
                    payload={"reason": "user paused; prewarm turn context"},
                ),
            )

        if trigger.kind == ProactiveTriggerKind.BUILD_RUNNING:
            return (
                self._envelope(
                    trigger,
                    work_kind=ProactiveWorkKind.BUILD_MONITORING,
                    target_worker=IntegratedWorkerKind.BACKGROUND_WORKER,
                    task_kind=IntegratedTaskKind.BACKGROUND_TASK,
                    payload={"reason": "monitor build and prepare summary"},
                ),
            )

        if trigger.kind == ProactiveTriggerKind.TEST_RUNNING:
            return (
                self._envelope(
                    trigger,
                    work_kind=ProactiveWorkKind.TEST_MONITORING,
                    target_worker=IntegratedWorkerKind.BACKGROUND_WORKER,
                    task_kind=IntegratedTaskKind.BACKGROUND_TASK,
                    payload={"reason": "monitor tests and prepare summary"},
                ),
            )

        if trigger.kind == ProactiveTriggerKind.FILE_CHANGED:
            return (
                self._envelope(
                    trigger,
                    work_kind=ProactiveWorkKind.WORKSPACE_NOTE,
                    target_worker=IntegratedWorkerKind.BACKGROUND_WORKER,
                    task_kind=IntegratedTaskKind.BACKGROUND_TASK,
                    payload={"reason": "file changed; note context for next turn"},
                ),
            )

        if trigger.kind == ProactiveTriggerKind.WORKSPACE_CHANGED:
            return (
                self._envelope(
                    trigger,
                    work_kind=ProactiveWorkKind.CONTEXT_PREWARM,
                    target_worker=IntegratedWorkerKind.BACKGROUND_WORKER,
                    task_kind=IntegratedTaskKind.BACKGROUND_TASK,
                    payload={"reason": "workspace changed; prepare context"},
                ),
                self._envelope(
                    trigger,
                    work_kind=ProactiveWorkKind.TOOL_PATH_PREWARM,
                    target_worker=IntegratedWorkerKind.BACKGROUND_WORKER,
                    task_kind=IntegratedTaskKind.BACKGROUND_TASK,
                    payload={"reason": "prewarm tool metadata, not execution"},
                ),
            )

        if trigger.kind == ProactiveTriggerKind.ERROR_PATTERN_SEEN:
            return (
                self._envelope(
                    trigger,
                    work_kind=ProactiveWorkKind.ERROR_CONTEXT_PREPARE,
                    target_worker=IntegratedWorkerKind.BACKGROUND_WORKER,
                    task_kind=IntegratedTaskKind.BACKGROUND_TASK,
                    payload={"reason": "prepare error context for possible help"},
                ),
            )

        return (
            self._envelope(
                trigger,
                work_kind=ProactiveWorkKind.CONTEXT_PREWARM,
                target_worker=IntegratedWorkerKind.BACKGROUND_WORKER,
                task_kind=IntegratedTaskKind.BACKGROUND_TASK,
                payload={"reason": "conversation idle; prepare context"},
            ),
        )

    def _build_suggestions(
        self,
        trigger: ProactiveTrigger,
    ) -> tuple[ProactiveSuggestion, ...]:
        threshold = self._config.suggestion_surface_threshold_percent

        if trigger.kind == ProactiveTriggerKind.BUILD_RUNNING:
            return (
                ProactiveSuggestion(
                    trigger_id=trigger.trigger_id,
                    title="Build monitoring ready",
                    message="I can summarize the build result when it finishes.",
                    confidence_percent=trigger.confidence_percent,
                    surface_threshold_percent=threshold,
                ),
            )

        if trigger.kind == ProactiveTriggerKind.ERROR_PATTERN_SEEN:
            return (
                ProactiveSuggestion(
                    trigger_id=trigger.trigger_id,
                    title="Possible error context prepared",
                    message="I noticed an error pattern and prepared context.",
                    confidence_percent=trigger.confidence_percent,
                    surface_threshold_percent=threshold,
                ),
            )

        if trigger.kind == ProactiveTriggerKind.FILE_CHANGED:
            return (
                ProactiveSuggestion(
                    trigger_id=trigger.trigger_id,
                    title="Workspace context updated",
                    message="I noted the file change for the next turn.",
                    confidence_percent=trigger.confidence_percent,
                    surface_threshold_percent=threshold,
                ),
            )

        return ()

    @staticmethod
    def _envelope(
        trigger: ProactiveTrigger,
        *,
        work_kind: ProactiveWorkKind,
        target_worker: IntegratedWorkerKind,
        task_kind: IntegratedTaskKind,
        payload: dict[str, Any],
    ) -> ProactiveTaskEnvelope:
        return ProactiveTaskEnvelope(
            trigger_id=trigger.trigger_id,
            work_kind=work_kind,
            target_worker=target_worker,
            task_kind=task_kind,
            payload={**payload, **trigger.payload},
            confidence_percent=trigger.confidence_percent,
            risk_level=ProactiveRiskLevel.LOW,
            read_only=True,
            cancellable=True,
            lower_than_reactive=True,
            action_allowed=False,
            requires_budget=True,
            requires_context_snapshot=True,
        )


class ProactiveEngine:
    """
    Phase 6 Step 17 Proactive Engine.

    Responsibilities:
    - accept safe proactive triggers
    - create low-risk read-only proactive task envelopes
    - suppress proactive work during active conversation
    - block action-like proactive work
    - surface suggestions only when confidence is high
    - provide queryable proactive diagnostics

    Non-responsibilities:
    - no tool execution
    - no file mutation
    - no memory mutation
    - no autonomous action
    - no direct worker execution
    """

    def __init__(
        self,
        *,
        config: ProactiveOrchestrationConfig | None = None,
        trigger_policy: TriggerPolicy | None = None,
    ) -> None:
        self._config = config or ProactiveOrchestrationConfig()
        self._config.validate()

        self._trigger_policy = trigger_policy or TriggerPolicy(
            config=self._config
        )
        self._lock = RLock()

        self._triggers: list[ProactiveTrigger] = []
        self._decisions: list[ProactiveDecision] = []
        self._cancelled_envelope_ids: set[str] = set()
        self._last_reason: ProactiveReason | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def handle_trigger(
        self,
        trigger: ProactiveTrigger,
    ) -> ProactiveResult:
        decision = self._trigger_policy.evaluate(trigger)

        with self._lock:
            self._triggers.append(trigger)
            self._decisions.append(decision)
            self._last_reason = decision.reason

        return ProactiveResult(
            reason=decision.reason,
            success=decision.status
            not in {
                ProactiveStatus.BLOCKED,
            },
            message=decision.message,
            decision=decision,
        )

    def cancel_envelope(self, envelope_id: str) -> ProactiveResult:
        cleaned = envelope_id.strip()

        if not cleaned:
            raise ValueError("envelope_id cannot be empty.")

        with self._lock:
            self._cancelled_envelope_ids.add(cleaned)
            self._last_reason = ProactiveReason.PROACTIVE_CANCELLED

        return ProactiveResult(
            reason=ProactiveReason.PROACTIVE_CANCELLED,
            success=True,
            message="proactive envelope cancelled",
            metadata={"envelope_id": cleaned},
        )

    def latest_decision(self) -> ProactiveDecision | None:
        with self._lock:
            if not self._decisions:
                return None

            return self._decisions[-1]

    def decisions(self) -> tuple[ProactiveDecision, ...]:
        with self._lock:
            return tuple(self._decisions)

    def pending_envelopes(self) -> tuple[ProactiveTaskEnvelope, ...]:
        with self._lock:
            cancelled = set(self._cancelled_envelope_ids)
            return tuple(
                envelope
                for decision in self._decisions
                for envelope in decision.envelopes
                if envelope.envelope_id not in cancelled
            )

    def integrated_envelopes(self) -> tuple[IntegratedTaskEnvelope, ...]:
        return tuple(
            envelope.to_integrated_envelope()
            for envelope in self.pending_envelopes()
        )

    def surfaced_suggestions(self) -> tuple[ProactiveSuggestion, ...]:
        with self._lock:
            return tuple(
                suggestion
                for decision in self._decisions
                for suggestion in decision.user_visible_suggestions
            )

    def snapshot(self) -> ProactiveOrchestrationSnapshot:
        with self._lock:
            envelope_count = sum(
                len(decision.envelopes) for decision in self._decisions
            )
            suggestion_count = sum(
                len(decision.suggestions) for decision in self._decisions
            )
            surfaced_count = sum(
                len(decision.user_visible_suggestions)
                for decision in self._decisions
            )
            suppressed_count = sum(
                1
                for decision in self._decisions
                if decision.status == ProactiveStatus.SUPPRESSED
            )
            blocked_count = sum(
                1
                for decision in self._decisions
                if decision.status == ProactiveStatus.BLOCKED
            )

            return ProactiveOrchestrationSnapshot(
                name=self.name,
                trigger_count=len(self._triggers),
                decision_count=len(self._decisions),
                envelope_count=envelope_count,
                suggestion_count=suggestion_count,
                surfaced_suggestion_count=surfaced_count,
                suppressed_count=suppressed_count,
                blocked_count=blocked_count,
                cancelled_count=len(self._cancelled_envelope_ids),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        with self._lock:
            self._triggers.clear()
            self._decisions.clear()
            self._cancelled_envelope_ids.clear()
            self._last_reason = ProactiveReason.RUNTIME_RESET