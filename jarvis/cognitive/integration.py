from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from uuid import uuid4

from jarvis.cognitive.attention import (
    AttentionSignal,
    AttentionSignalSource,
    AttentionSignalUrgency,
    make_attention_signal,
)
from jarvis.cognitive.contracts import (
    AttentionItemKind,
    AttentionPriority,
    GoalPriority,
    WorkingMemoryKind,
    utc_now,
)
from jarvis.cognitive.personality import BehaviorIntent, BehaviorRisk
from jarvis.cognitive.planning import PlanIntentKind
from jarvis.cognitive.session import (
    CognitiveSessionGoalRequest,
    CognitiveSessionResponseRequest,
    CognitiveSessionRuntime,
    CognitiveSessionRuntimeResult,
    CognitiveSessionRuntimeSnapshot,
    CognitiveSessionRuntimeStatus,
    CognitiveSessionStartRequest,
    CognitiveSessionUpdateRequest,
)
from jarvis.cognitive.working_memory import (
    WorkingMemoryEntry,
    make_working_memory_entry,
)


class CognitiveIntegrationStatus(StrEnum):
    READY = "ready"
    DEGRADED = "degraded"
    BLOCKED = "blocked"


class CognitiveIntegrationSource(StrEnum):
    PRESENCE = "presence"
    CONVERSATION = "conversation"
    MEMORY = "memory"
    COGNITION = "cognition"
    ORCHESTRATION = "orchestration"
    DEVELOPER = "developer"
    ENVIRONMENT = "environment"
    SYSTEM = "system"


class CognitiveIntegrationEventKind(StrEnum):
    USER_UTTERANCE = "user_utterance"
    ASSISTANT_RESPONSE = "assistant_response"
    INTERRUPTION = "interruption"
    MEMORY_RECALL = "memory_recall"
    COGNITION_RESULT = "cognition_result"
    ORCHESTRATION_ALERT = "orchestration_alert"
    SYSTEM_HEALTH = "system_health"
    SCREEN_CONTEXT = "screen_context"
    DEVELOPER_BUILD = "developer_build"
    DEVELOPER_ERROR = "developer_error"
    GOAL_REQUEST = "goal_request"
    WARNING = "warning"
    STATUS = "status"


@dataclass(frozen=True, slots=True)
class CognitiveIntegrationEvent:
    event_id: str
    source: CognitiveIntegrationSource
    kind: CognitiveIntegrationEventKind
    title: str
    summary: str
    urgency: AttentionSignalUrgency
    confidence: float = 1.0
    working_memory_kind: WorkingMemoryKind | None = None
    goal_title: str | None = None
    goal_description: str | None = None
    goal_priority: GoalPriority | None = None
    plan_intent_kind: PlanIntentKind = PlanIntentKind.GENERAL
    behavior_intent: BehaviorIntent | None = None
    behavior_risk: BehaviorRisk = BehaviorRisk.NONE
    user_is_speaking: bool = False
    assistant_is_speaking: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id.strip():
            raise ValueError("integration event_id cannot be empty.")
        if not self.title.strip():
            raise ValueError("integration event title cannot be empty.")
        if not self.summary.strip():
            raise ValueError("integration event summary cannot be empty.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("integration event confidence must be between 0 and 1.")
        if self.goal_title is not None and not self.goal_title.strip():
            raise ValueError("integration event goal_title cannot be empty.")
        if (
            self.goal_description is not None
            and not self.goal_description.strip()
        ):
            raise ValueError("integration event goal_description cannot be empty.")


@dataclass(frozen=True, slots=True)
class CognitiveIntegrationRequest:
    events: tuple[CognitiveIntegrationEvent, ...]
    user_label: str = "Balu"
    start_session: bool = False
    allow_interruptions: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.user_label.strip():
            raise ValueError("integration user_label cannot be empty.")


@dataclass(frozen=True, slots=True)
class CognitiveIntegrationResult:
    status: CognitiveIntegrationStatus
    session_result: CognitiveSessionRuntimeResult
    goal_results: tuple[CognitiveSessionRuntimeResult, ...]
    behavior_results: tuple[CognitiveSessionRuntimeResult, ...]
    processed_events: tuple[CognitiveIntegrationEvent, ...]
    reason: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == CognitiveIntegrationStatus.READY

    @property
    def should_interrupt(self) -> bool:
        attention_result = self.session_result.attention_result
        return bool(
            attention_result is not None
            and attention_result.should_interrupt
        )


@dataclass(frozen=True, slots=True)
class CognitiveIntegrationRuntimeSnapshot:
    status: CognitiveIntegrationStatus
    session_snapshot: CognitiveSessionRuntimeSnapshot
    ingest_count: int
    event_count: int
    interruption_count: int
    goal_event_count: int
    behavior_event_count: int
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


class CognitiveIntegrationRuntime:
    """
    Phase 9 / Step 49G Integration Runtime.

    This is the bridge between existing organs and Phase 9 cognitive state.

    Existing systems remain separate:
    - Presence emits voice/interruption events.
    - Conversation emits user-intent events.
    - Memory emits recall-context events.
    - Cognition emits reasoning/result events.
    - Orchestration emits priority/system events.
    - Developer Pack emits build/error/navigation events.
    - Environment emits screen/system context events.

    This runtime converts those into CognitiveSessionRuntime updates.

    It does not execute tools.
    It does not write long-term memory.
    It does not replace Phase 1–8.
    It does not duplicate Developer Pack.
    """

    def __init__(
        self,
        *,
        session_runtime: CognitiveSessionRuntime | None = None,
    ) -> None:
        self._session = session_runtime or CognitiveSessionRuntime()
        self._ingest_count = 0
        self._event_count = 0
        self._interruption_count = 0
        self._goal_event_count = 0
        self._behavior_event_count = 0

    @property
    def session_runtime(self) -> CognitiveSessionRuntime:
        return self._session

    def ingest(
        self,
        request: CognitiveIntegrationRequest,
    ) -> CognitiveIntegrationResult:
        self._ingest_count += 1
        self._event_count += len(request.events)

        if request.start_session:
            self._session.start(
                CognitiveSessionStartRequest(
                    user_label=request.user_label,
                    metadata=request.metadata,
                )
            )

        if not request.events:
            session_result = self._session.update(
                CognitiveSessionUpdateRequest(metadata=request.metadata)
            )
            return CognitiveIntegrationResult(
                status=CognitiveIntegrationStatus.DEGRADED,
                session_result=session_result,
                goal_results=(),
                behavior_results=(),
                processed_events=(),
                reason="no integration events provided",
                created_at=utc_now(),
                metadata=request.metadata,
            )

        attention_signals = tuple(
            _attention_signal_from_event(event)
            for event in request.events
        )
        working_memory_entries = tuple(
            entry
            for entry in (
                _working_memory_entry_from_event(event)
                for event in request.events
            )
            if entry is not None
        )
        user_is_speaking = any(event.user_is_speaking for event in request.events)
        assistant_is_speaking = any(
            event.assistant_is_speaking
            for event in request.events
        )

        session_result = self._session.update(
            CognitiveSessionUpdateRequest(
                attention_signals=attention_signals,
                working_memory_entries=working_memory_entries,
                user_is_speaking=user_is_speaking,
                assistant_is_speaking=assistant_is_speaking,
                allow_interruptions=request.allow_interruptions,
                metadata=request.metadata,
            )
        )

        if session_result.attention_result is not None:
            if session_result.attention_result.should_interrupt:
                self._interruption_count += 1

        goal_results: list[CognitiveSessionRuntimeResult] = []
        behavior_results: list[CognitiveSessionRuntimeResult] = []

        for event in request.events:
            goal_result = self._maybe_create_goal(event)
            if goal_result is not None:
                goal_results.append(goal_result)
                self._goal_event_count += 1

            behavior_result = self._maybe_respond(event)
            if behavior_result is not None:
                behavior_results.append(behavior_result)
                self._behavior_event_count += 1

            if goal_results or behavior_results:
                session_result = self._session.update(
                    CognitiveSessionUpdateRequest(metadata=request.metadata)
                )

        status = (
            CognitiveIntegrationStatus.READY
            if session_result.status == CognitiveSessionRuntimeStatus.READY
            else CognitiveIntegrationStatus.DEGRADED
        )

        return CognitiveIntegrationResult(
            status=status,
            session_result=session_result,
            goal_results=tuple(goal_results),
            behavior_results=tuple(behavior_results),
            processed_events=request.events,
            reason="integration events ingested into cognitive session",
            created_at=utc_now(),
            metadata={
                **request.metadata,
                "event_count": len(request.events),
                "working_memory_entries": len(working_memory_entries),
                "goal_results": len(goal_results),
                "behavior_results": len(behavior_results),
            },
        )

    def snapshot(self) -> CognitiveIntegrationRuntimeSnapshot:
        return CognitiveIntegrationRuntimeSnapshot(
            status=CognitiveIntegrationStatus.READY,
            session_snapshot=self._session.snapshot(),
            ingest_count=self._ingest_count,
            event_count=self._event_count,
            interruption_count=self._interruption_count,
            goal_event_count=self._goal_event_count,
            behavior_event_count=self._behavior_event_count,
            created_at=utc_now(),
        )

    def _maybe_create_goal(
        self,
        event: CognitiveIntegrationEvent,
    ) -> CognitiveSessionRuntimeResult | None:
        if event.kind != CognitiveIntegrationEventKind.GOAL_REQUEST:
            return None

        title = event.goal_title or event.title
        description = event.goal_description or event.summary
        priority = event.goal_priority or _goal_priority_from_urgency(
            event.urgency
        )

        return self._session.create_goal(
            CognitiveSessionGoalRequest(
                title=title,
                description=description,
                priority=priority,
                tags=(
                    event.source.value,
                    event.kind.value,
                ),
                create_plan=True,
                intent_kind=event.plan_intent_kind,
                metadata=event.metadata,
            )
        )

    def _maybe_respond(
        self,
        event: CognitiveIntegrationEvent,
    ) -> CognitiveSessionRuntimeResult | None:
        intent = event.behavior_intent or _behavior_intent_for_event(event)

        if intent is None:
            return None

        risk = (
            event.behavior_risk
            if event.behavior_risk != BehaviorRisk.NONE
            else _behavior_risk_from_urgency(event.urgency)
        )

        return self._session.respond(
            CognitiveSessionResponseRequest(
                intent=intent,
                message=event.summary,
                risk=risk,
                instruction_complete=True,
                user_is_busy=event.user_is_speaking,
                allow_humor=False,
                requires_truth_challenge=(
                    event.kind == CognitiveIntegrationEventKind.WARNING
                ),
                metadata=event.metadata,
            )
        )


def make_cognitive_integration_event(
    *,
    source: CognitiveIntegrationSource,
    kind: CognitiveIntegrationEventKind,
    title: str,
    summary: str,
    urgency: AttentionSignalUrgency,
    confidence: float = 1.0,
    working_memory_kind: WorkingMemoryKind | None = None,
    goal_title: str | None = None,
    goal_description: str | None = None,
    goal_priority: GoalPriority | None = None,
    plan_intent_kind: PlanIntentKind = PlanIntentKind.GENERAL,
    behavior_intent: BehaviorIntent | None = None,
    behavior_risk: BehaviorRisk = BehaviorRisk.NONE,
    user_is_speaking: bool = False,
    assistant_is_speaking: bool = False,
    metadata: dict[str, object] | None = None,
) -> CognitiveIntegrationEvent:
    return CognitiveIntegrationEvent(
        event_id=f"cie_{uuid4().hex}",
        source=source,
        kind=kind,
        title=title,
        summary=summary,
        urgency=urgency,
        confidence=confidence,
        working_memory_kind=working_memory_kind,
        goal_title=goal_title,
        goal_description=goal_description,
        goal_priority=goal_priority,
        plan_intent_kind=plan_intent_kind,
        behavior_intent=behavior_intent,
        behavior_risk=behavior_risk,
        user_is_speaking=user_is_speaking,
        assistant_is_speaking=assistant_is_speaking,
        metadata=metadata or {},
    )


def _attention_signal_from_event(
    event: CognitiveIntegrationEvent,
) -> AttentionSignal:
    return make_attention_signal(
        source=_attention_source_from_integration_source(event.source),
        kind=_attention_kind_from_event_kind(event.kind),
        title=event.title,
        summary=event.summary,
        urgency=event.urgency,
        confidence=event.confidence,
        metadata={
            **event.metadata,
            "integration_event_id": event.event_id,
            "integration_source": event.source.value,
            "integration_kind": event.kind.value,
        },
    )


def _working_memory_entry_from_event(
    event: CognitiveIntegrationEvent,
) -> WorkingMemoryEntry | None:
    kind = event.working_memory_kind or _working_memory_kind_for_event(event)
    if kind is None:
        return None

    return make_working_memory_entry(
        kind=kind,
        key=f"{event.source.value}:{event.kind.value}:{event.title}",
        value=event.summary,
        importance=_attention_priority_from_urgency(event.urgency),
        source=f"integration:{event.source.value}",
        metadata={
            **event.metadata,
            "integration_event_id": event.event_id,
            "integration_kind": event.kind.value,
        },
    )


def _attention_source_from_integration_source(
    source: CognitiveIntegrationSource,
) -> AttentionSignalSource:
    mapping = {
        CognitiveIntegrationSource.PRESENCE: AttentionSignalSource.VOICE,
        CognitiveIntegrationSource.CONVERSATION: AttentionSignalSource.USER,
        CognitiveIntegrationSource.MEMORY: AttentionSignalSource.MEMORY,
        CognitiveIntegrationSource.COGNITION: AttentionSignalSource.TASK,
        CognitiveIntegrationSource.ORCHESTRATION: AttentionSignalSource.TASK,
        CognitiveIntegrationSource.DEVELOPER: AttentionSignalSource.PROJECT,
        CognitiveIntegrationSource.ENVIRONMENT: AttentionSignalSource.SCREEN,
        CognitiveIntegrationSource.SYSTEM: AttentionSignalSource.SYSTEM,
    }
    return mapping[source]


def _attention_kind_from_event_kind(
    kind: CognitiveIntegrationEventKind,
) -> AttentionItemKind:
    mapping = {
        CognitiveIntegrationEventKind.USER_UTTERANCE: (
            AttentionItemKind.USER_COMMAND
        ),
        CognitiveIntegrationEventKind.ASSISTANT_RESPONSE: (
            AttentionItemKind.ACTIVE_TASK
        ),
        CognitiveIntegrationEventKind.INTERRUPTION: AttentionItemKind.SAFETY,
        CognitiveIntegrationEventKind.MEMORY_RECALL: (
            AttentionItemKind.MEMORY_RECALL
        ),
        CognitiveIntegrationEventKind.COGNITION_RESULT: (
            AttentionItemKind.ACTIVE_TASK
        ),
        CognitiveIntegrationEventKind.ORCHESTRATION_ALERT: (
            AttentionItemKind.ACTIVE_TASK
        ),
        CognitiveIntegrationEventKind.SYSTEM_HEALTH: (
            AttentionItemKind.SYSTEM_HEALTH
        ),
        CognitiveIntegrationEventKind.SCREEN_CONTEXT: AttentionItemKind.SCREEN,
        CognitiveIntegrationEventKind.DEVELOPER_BUILD: (
            AttentionItemKind.PROJECT
        ),
        CognitiveIntegrationEventKind.DEVELOPER_ERROR: (
            AttentionItemKind.PROJECT
        ),
        CognitiveIntegrationEventKind.GOAL_REQUEST: (
            AttentionItemKind.USER_COMMAND
        ),
        CognitiveIntegrationEventKind.WARNING: AttentionItemKind.SAFETY,
        CognitiveIntegrationEventKind.STATUS: AttentionItemKind.ACTIVE_TASK,
    }
    return mapping[kind]


def _working_memory_kind_for_event(
    event: CognitiveIntegrationEvent,
) -> WorkingMemoryKind | None:
    mapping = {
        CognitiveIntegrationEventKind.USER_UTTERANCE: (
            WorkingMemoryKind.CONVERSATION
        ),
        CognitiveIntegrationEventKind.MEMORY_RECALL: (
            WorkingMemoryKind.RECENT_ACTION
        ),
        CognitiveIntegrationEventKind.COGNITION_RESULT: (
            WorkingMemoryKind.RECENT_ACTION
        ),
        CognitiveIntegrationEventKind.SYSTEM_HEALTH: WorkingMemoryKind.RISK,
        CognitiveIntegrationEventKind.SCREEN_CONTEXT: (
            WorkingMemoryKind.SCREEN_CONTEXT
        ),
        CognitiveIntegrationEventKind.DEVELOPER_BUILD: WorkingMemoryKind.TASK,
        CognitiveIntegrationEventKind.DEVELOPER_ERROR: WorkingMemoryKind.RISK,
        CognitiveIntegrationEventKind.GOAL_REQUEST: WorkingMemoryKind.OBJECTIVE,
        CognitiveIntegrationEventKind.WARNING: WorkingMemoryKind.RISK,
        CognitiveIntegrationEventKind.STATUS: WorkingMemoryKind.RECENT_ACTION,
    }
    return mapping.get(event.kind)


def _goal_priority_from_urgency(
    urgency: AttentionSignalUrgency,
) -> GoalPriority:
    if urgency == AttentionSignalUrgency.EMERGENCY:
        return GoalPriority.CRITICAL
    if urgency == AttentionSignalUrgency.URGENT:
        return GoalPriority.HIGH
    if urgency == AttentionSignalUrgency.IMPORTANT:
        return GoalPriority.HIGH
    if urgency == AttentionSignalUrgency.NORMAL:
        return GoalPriority.NORMAL
    return GoalPriority.LOW


def _attention_priority_from_urgency(
    urgency: AttentionSignalUrgency,
) -> AttentionPriority:

    if urgency == AttentionSignalUrgency.EMERGENCY:
        return AttentionPriority.CRITICAL
    if urgency == AttentionSignalUrgency.URGENT:
        return AttentionPriority.HIGH
    if urgency == AttentionSignalUrgency.IMPORTANT:
        return AttentionPriority.HIGH
    if urgency == AttentionSignalUrgency.NORMAL:
        return AttentionPriority.NORMAL
    return AttentionPriority.BACKGROUND


def _behavior_intent_for_event(
    event: CognitiveIntegrationEvent,
) -> BehaviorIntent | None:
    if event.kind == CognitiveIntegrationEventKind.INTERRUPTION:
        return BehaviorIntent.INTERRUPTION
    if event.kind == CognitiveIntegrationEventKind.WARNING:
        return BehaviorIntent.WARNING
    if event.kind == CognitiveIntegrationEventKind.SYSTEM_HEALTH:
        return BehaviorIntent.WARNING
    if event.kind == CognitiveIntegrationEventKind.DEVELOPER_ERROR:
        return BehaviorIntent.WARNING
    if event.kind == CognitiveIntegrationEventKind.STATUS:
        return BehaviorIntent.STATUS
    if event.kind == CognitiveIntegrationEventKind.ASSISTANT_RESPONSE:
        return BehaviorIntent.STATUS
    return None


def _behavior_risk_from_urgency(
    urgency: AttentionSignalUrgency,
) -> BehaviorRisk:
    if urgency == AttentionSignalUrgency.EMERGENCY:
        return BehaviorRisk.CRITICAL
    if urgency == AttentionSignalUrgency.URGENT:
        return BehaviorRisk.HIGH
    if urgency == AttentionSignalUrgency.IMPORTANT:
        return BehaviorRisk.MEDIUM
    if urgency == AttentionSignalUrgency.NORMAL:
        return BehaviorRisk.LOW
    return BehaviorRisk.NONE