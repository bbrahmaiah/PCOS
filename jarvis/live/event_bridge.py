from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from jarvis.cognitive import (
    AttentionSignalUrgency,
    CognitiveIntegrationEventKind,
    CognitiveIntegrationRequest,
    CognitiveIntegrationResult,
    CognitiveIntegrationRuntime,
    CognitiveIntegrationSource,
    WorkingMemoryKind,
    make_cognitive_integration_event,
)
from jarvis.live.contracts import (
    LiveEventKind,
    LiveEventPriority,
    LiveResponse,
    LiveResponseGenerationSource,
    LiveResponseKind,
    LiveResponseSafety,
    LiveSessionEvent,
    LiveSubsystem,
    LiveTranscript,
    make_live_event,
    make_live_response,
    utc_now,
)
from jarvis.live.response_boundary import (
    LiveResponseBoundaryResult,
    LiveResponseBoundaryRuntime,
)
from jarvis.live.session_state import (
    LiveSessionStateRuntime,
    LiveSessionStateRuntimeResult,
    LiveSessionStateRuntimeStatus,
)


class LiveEventBridgeStatus(StrEnum):
    READY = "ready"
    DEGRADED = "degraded"
    BLOCKED = "blocked"


class LiveEventBridgeOperation(StrEnum):
    BRIDGE_EVENT = "bridge_event"
    BRIDGE_TRANSCRIPT = "bridge_transcript"
    BRIDGE_RESPONSE = "bridge_response"
    BRIDGE_SUBSYSTEM_SIGNAL = "bridge_subsystem_signal"
    SNAPSHOT = "snapshot"


class LiveEventBridgeRoute(StrEnum):
    LIVE_STATE_ONLY = "live_state_only"
    COGNITIVE_ONLY = "cognitive_only"
    LIVE_AND_COGNITIVE = "live_and_cognitive"
    RESPONSE_BOUNDARY = "response_boundary"


@dataclass(frozen=True, slots=True)
class LiveEventBridgeRequest:
    event: LiveSessionEvent
    update_live_state: bool = True
    update_cognitive_state: bool = True
    allow_interruptions: bool = True
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LiveTranscriptBridgeRequest:
    transcript: LiveTranscript
    user_is_speaking: bool = False
    assistant_is_speaking: bool = False
    allow_interruptions: bool = True
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LiveResponseBridgeRequest:
    response: LiveResponse
    validate_for_tts: bool = True
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LiveEventBridgeResult:
    status: LiveEventBridgeStatus
    operation: LiveEventBridgeOperation
    route: LiveEventBridgeRoute
    live_result: LiveSessionStateRuntimeResult | None
    cognitive_result: CognitiveIntegrationResult | None
    response_boundary_result: LiveResponseBoundaryResult | None
    reason: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == LiveEventBridgeStatus.READY

    @property
    def should_interrupt(self) -> bool:
        return bool(
            self.cognitive_result is not None
            and self.cognitive_result.should_interrupt
        )


@dataclass(frozen=True, slots=True)
class LiveEventBridgeSnapshot:
    status: LiveEventBridgeStatus
    bridged_event_count: int
    bridged_transcript_count: int
    bridged_response_count: int
    blocked_count: int
    live_event_count: int
    cognitive_session_id: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


class LiveEventBridgeRuntime:
    """
    Step 50C Live Event Bridge.

    This runtime bridges the already-built organs into the live session.

    It does not replace:
    - Presence Runtime
    - Conversation Runtime
    - Memory Runtime
    - Cognition Runtime
    - Orchestration Runtime
    - Environment Runtime
    - Developer Pack
    - Phase 9 Cognitive Session

    It only converts live events into typed state/cognitive integration updates.

    It never executes tools.
    It never accesses memory directly.
    It never calls TTS directly.
    It never creates scripted conversational speech.
    """

    def __init__(
        self,
        *,
        live_state: LiveSessionStateRuntime | None = None,
        cognitive_integration: CognitiveIntegrationRuntime | None = None,
        response_boundary: LiveResponseBoundaryRuntime | None = None,
    ) -> None:
        self._live_state = live_state or LiveSessionStateRuntime()
        self._cognitive = cognitive_integration or CognitiveIntegrationRuntime()
        self._response_boundary = response_boundary or LiveResponseBoundaryRuntime()
        self._bridged_event_count = 0
        self._bridged_transcript_count = 0
        self._bridged_response_count = 0
        self._blocked_count = 0

    @property
    def live_state(self) -> LiveSessionStateRuntime:
        return self._live_state

    @property
    def cognitive_integration(self) -> CognitiveIntegrationRuntime:
        return self._cognitive

    def bridge_event(
        self,
        request: LiveEventBridgeRequest,
    ) -> LiveEventBridgeResult:
        live_result = (
            self._apply_live_event_to_state(request.event)
            if request.update_live_state
            else None
        )
        cognitive_result = (
            self._apply_live_event_to_cognition(
                event=request.event,
                allow_interruptions=request.allow_interruptions,
                metadata={**request.metadata, **request.event.metadata},
            )
            if request.update_cognitive_state
            else None
        )

        status = _combined_status(live_result, cognitive_result)
        if status == LiveEventBridgeStatus.BLOCKED:
            self._blocked_count += 1
        else:
            self._bridged_event_count += 1

        return LiveEventBridgeResult(
            status=status,
            operation=LiveEventBridgeOperation.BRIDGE_EVENT,
            route=_route_for(live_result, cognitive_result, None),
            live_result=live_result,
            cognitive_result=cognitive_result,
            response_boundary_result=None,
            reason="live event bridged",
            created_at=utc_now(),
            metadata={
                **request.metadata,
                "event_kind": request.event.kind.value,
                "event_source": request.event.source.value,
            },
        )

    def bridge_transcript(
        self,
        request: LiveTranscriptBridgeRequest,
    ) -> LiveEventBridgeResult:
        live_result = self._live_state.transcript_ready(request.transcript)

        if live_result.status == LiveSessionStateRuntimeStatus.BLOCKED:
            self._blocked_count += 1
            return LiveEventBridgeResult(
                status=LiveEventBridgeStatus.BLOCKED,
                operation=LiveEventBridgeOperation.BRIDGE_TRANSCRIPT,
                route=LiveEventBridgeRoute.LIVE_STATE_ONLY,
                live_result=live_result,
                cognitive_result=None,
                response_boundary_result=None,
                reason="transcript rejected by live state runtime",
                created_at=utc_now(),
                metadata=request.metadata,
            )

        cognitive_event = make_cognitive_integration_event(
            source=CognitiveIntegrationSource.CONVERSATION,
            kind=CognitiveIntegrationEventKind.USER_UTTERANCE,
            title="Live transcript",
            summary=request.transcript.text,
            urgency=AttentionSignalUrgency.IMPORTANT,
            confidence=request.transcript.confidence,
            working_memory_kind=WorkingMemoryKind.CONVERSATION,
            user_is_speaking=request.user_is_speaking,
            assistant_is_speaking=request.assistant_is_speaking,
            metadata={
                **request.metadata,
                "turn_id": str(request.transcript.turn_id),
                "transcript_id": str(request.transcript.transcript_id),
                "transcript_kind": request.transcript.kind.value,
            },
        )
        cognitive_result = self._cognitive.ingest(
            CognitiveIntegrationRequest(
                events=(cognitive_event,),
                allow_interruptions=request.allow_interruptions,
                metadata=request.metadata,
            )
        )

        self._bridged_transcript_count += 1

        return LiveEventBridgeResult(
            status=LiveEventBridgeStatus.READY,
            operation=LiveEventBridgeOperation.BRIDGE_TRANSCRIPT,
            route=LiveEventBridgeRoute.LIVE_AND_COGNITIVE,
            live_result=live_result,
            cognitive_result=cognitive_result,
            response_boundary_result=None,
            reason="transcript bridged to conversation and cognitive session",
            created_at=utc_now(),
            metadata=request.metadata,
        )

    def bridge_response(
        self,
        request: LiveResponseBridgeRequest,
    ) -> LiveEventBridgeResult:
        boundary_result = (
            self._response_boundary.validate_for_tts(request.response)
            if request.validate_for_tts
            else None
        )

        if boundary_result is not None and not boundary_result.succeeded:
            self._blocked_count += 1
            return LiveEventBridgeResult(
                status=LiveEventBridgeStatus.BLOCKED,
                operation=LiveEventBridgeOperation.BRIDGE_RESPONSE,
                route=LiveEventBridgeRoute.RESPONSE_BOUNDARY,
                live_result=None,
                cognitive_result=None,
                response_boundary_result=boundary_result,
                reason="response rejected by live response boundary",
                created_at=utc_now(),
                metadata=request.metadata,
            )

        live_result = self._live_state.start_speaking(request.response)

        if live_result.status == LiveSessionStateRuntimeStatus.BLOCKED:
            self._blocked_count += 1
            return LiveEventBridgeResult(
                status=LiveEventBridgeStatus.BLOCKED,
                operation=LiveEventBridgeOperation.BRIDGE_RESPONSE,
                route=LiveEventBridgeRoute.RESPONSE_BOUNDARY,
                live_result=live_result,
                cognitive_result=None,
                response_boundary_result=boundary_result,
                reason="response rejected by live session state",
                created_at=utc_now(),
                metadata=request.metadata,
            )

        self._bridged_response_count += 1

        return LiveEventBridgeResult(
            status=LiveEventBridgeStatus.READY,
            operation=LiveEventBridgeOperation.BRIDGE_RESPONSE,
            route=LiveEventBridgeRoute.RESPONSE_BOUNDARY,
            live_result=live_result,
            cognitive_result=None,
            response_boundary_result=boundary_result,
            reason="generated live response accepted for speech state",
            created_at=utc_now(),
            metadata={
                **request.metadata,
                "response_id": str(request.response.response_id),
                "generation_source": request.response.generation_source.value,
            },
        )

    def bridge_subsystem_signal(
        self,
        *,
        source: LiveSubsystem,
        title: str,
        summary: str,
        priority: LiveEventPriority,
        metadata: dict[str, object] | None = None,
    ) -> LiveEventBridgeResult:
        event = make_live_event(
            kind=_event_kind_for_subsystem_signal(source),
            priority=priority,
            source=source,
            title=title,
            summary=summary,
            metadata=metadata,
        )
        return self.bridge_event(
            LiveEventBridgeRequest(
                event=event,
                update_live_state=True,
                update_cognitive_state=True,
                metadata=metadata or {},
            )
        )

    def snapshot(self) -> LiveEventBridgeSnapshot:
        cognitive_snapshot = self._cognitive.snapshot()
        return LiveEventBridgeSnapshot(
            status=LiveEventBridgeStatus.READY,
            bridged_event_count=self._bridged_event_count,
            bridged_transcript_count=self._bridged_transcript_count,
            bridged_response_count=self._bridged_response_count,
            blocked_count=self._blocked_count,
            live_event_count=len(self._live_state.events),
            cognitive_session_id=cognitive_snapshot.session_snapshot.session_id,
            created_at=utc_now(),
        )

    def _apply_live_event_to_state(
        self,
        event: LiveSessionEvent,
    ) -> LiveSessionStateRuntimeResult | None:
        if event.kind == LiveEventKind.SESSION_START_REQUESTED:
            return self._live_state.start()

        if event.kind == LiveEventKind.SESSION_STARTED:
            return self._live_state.mark_ready()

        if event.kind == LiveEventKind.WAKE_DETECTED:
            return self._live_state.enter_listening()

        if event.kind == LiveEventKind.USER_SPEECH_STARTED:
            return self._live_state.start_user_turn()

        if event.kind == LiveEventKind.INTERRUPTION_REQUESTED:
            return self._live_state.interrupt(reason=event.summary)

        if event.kind == LiveEventKind.RECOVERY_STARTED:
            return self._live_state.enter_recovery(
                subsystem=event.source,
                reason=event.summary,
            )

        if event.kind == LiveEventKind.RECOVERY_FINISHED:
            return self._live_state.finish_recovery()

        if event.kind == LiveEventKind.SESSION_STOP_REQUESTED:
            from jarvis.live.contracts import LiveShutdownReason

            return self._live_state.stop(reason=LiveShutdownReason.USER_REQUEST)

        return None

    def _apply_live_event_to_cognition(
        self,
        *,
        event: LiveSessionEvent,
        allow_interruptions: bool,
        metadata: dict[str, object],
    ) -> CognitiveIntegrationResult:
        cognitive_event = make_cognitive_integration_event(
            source=_cognitive_source_from_live_source(event.source),
            kind=_cognitive_kind_from_live_kind(event.kind),
            title=event.title,
            summary=event.summary,
            urgency=_urgency_from_priority(event.priority),
            confidence=1.0,
            working_memory_kind=_working_memory_kind_from_live_kind(event.kind),
            user_is_speaking=event.kind == LiveEventKind.USER_SPEECH_STARTED,
            assistant_is_speaking=event.kind
            == LiveEventKind.ASSISTANT_RESPONSE_STARTED,
            metadata={
                **metadata,
                "live_event_id": str(event.event_id),
                "live_event_kind": event.kind.value,
                "live_source": event.source.value,
            },
        )
        return self._cognitive.ingest(
            CognitiveIntegrationRequest(
                events=(cognitive_event,),
                allow_interruptions=allow_interruptions,
                metadata=metadata,
            )
        )


def make_generated_live_response_for_bridge_test(
    *,
    turn_id: LiveTranscript,
    text: str,
) -> LiveResponse:
    """
    Test helper only.

    This does not bypass the response boundary in production code.
    Production responses must be generated through 50A.5.
    """
    return make_live_response(
        turn_id=turn_id.turn_id,
        kind=LiveResponseKind.CONVERSATIONAL,
        text=text,
        generation_source=LiveResponseGenerationSource.RESPONSE_GENERATOR,
        safety=LiveResponseSafety.SAFE_TO_SPEAK,
    )


def _combined_status(
    live_result: LiveSessionStateRuntimeResult | None,
    cognitive_result: CognitiveIntegrationResult | None,
) -> LiveEventBridgeStatus:
    if live_result is not None:
        if live_result.status == LiveSessionStateRuntimeStatus.BLOCKED:
            return LiveEventBridgeStatus.BLOCKED

    if cognitive_result is not None:
        if not cognitive_result.succeeded:
            return LiveEventBridgeStatus.DEGRADED

    return LiveEventBridgeStatus.READY


def _route_for(
    live_result: LiveSessionStateRuntimeResult | None,
    cognitive_result: CognitiveIntegrationResult | None,
    boundary_result: LiveResponseBoundaryResult | None,
) -> LiveEventBridgeRoute:
    if boundary_result is not None:
        return LiveEventBridgeRoute.RESPONSE_BOUNDARY

    if live_result is not None and cognitive_result is not None:
        return LiveEventBridgeRoute.LIVE_AND_COGNITIVE

    if live_result is not None:
        return LiveEventBridgeRoute.LIVE_STATE_ONLY

    return LiveEventBridgeRoute.COGNITIVE_ONLY


def _urgency_from_priority(
    priority: LiveEventPriority,
) -> AttentionSignalUrgency:
    if priority == LiveEventPriority.CRITICAL:
        return AttentionSignalUrgency.EMERGENCY
    if priority == LiveEventPriority.HIGH:
        return AttentionSignalUrgency.URGENT
    if priority == LiveEventPriority.NORMAL:
        return AttentionSignalUrgency.NORMAL
    return AttentionSignalUrgency.BACKGROUND


def _cognitive_source_from_live_source(
    source: LiveSubsystem,
) -> CognitiveIntegrationSource:
    mapping = {
        LiveSubsystem.PRESENCE: CognitiveIntegrationSource.PRESENCE,
        LiveSubsystem.CONVERSATION: CognitiveIntegrationSource.CONVERSATION,
        LiveSubsystem.COGNITION: CognitiveIntegrationSource.COGNITION,
        LiveSubsystem.MEMORY: CognitiveIntegrationSource.MEMORY,
        LiveSubsystem.ORCHESTRATION: CognitiveIntegrationSource.ORCHESTRATION,
        LiveSubsystem.ENVIRONMENT: CognitiveIntegrationSource.ENVIRONMENT,
        LiveSubsystem.DEVELOPER_PACK: CognitiveIntegrationSource.DEVELOPER,
        LiveSubsystem.RUNTIME_KERNEL: CognitiveIntegrationSource.SYSTEM,
        LiveSubsystem.EVENT_BUS: CognitiveIntegrationSource.SYSTEM,
        LiveSubsystem.TOOLS: CognitiveIntegrationSource.ORCHESTRATION,
        LiveSubsystem.LATENCY: CognitiveIntegrationSource.ORCHESTRATION,
        LiveSubsystem.COGNITIVE_SESSION: CognitiveIntegrationSource.COGNITION,
        LiveSubsystem.MICROPHONE: CognitiveIntegrationSource.PRESENCE,
        LiveSubsystem.STT: CognitiveIntegrationSource.PRESENCE,
        LiveSubsystem.TTS: CognitiveIntegrationSource.PRESENCE,
        LiveSubsystem.PLAYBACK: CognitiveIntegrationSource.PRESENCE,
        LiveSubsystem.WAKE: CognitiveIntegrationSource.PRESENCE,
        LiveSubsystem.INTERRUPTION: CognitiveIntegrationSource.PRESENCE,
        LiveSubsystem.HEALTH_MONITOR: CognitiveIntegrationSource.SYSTEM,
        LiveSubsystem.RECOVERY: CognitiveIntegrationSource.SYSTEM,
        LiveSubsystem.RESPONSE_GENERATOR: CognitiveIntegrationSource.COGNITION,
    }
    return mapping[source]


def _cognitive_kind_from_live_kind(
    kind: LiveEventKind,
) -> CognitiveIntegrationEventKind:
    mapping = {
        LiveEventKind.SESSION_START_REQUESTED: (
            CognitiveIntegrationEventKind.STATUS
        ),
        LiveEventKind.SESSION_STARTED: CognitiveIntegrationEventKind.STATUS,
        LiveEventKind.SESSION_STOP_REQUESTED: (
            CognitiveIntegrationEventKind.STATUS
        ),
        LiveEventKind.SESSION_STOPPED: CognitiveIntegrationEventKind.STATUS,
        LiveEventKind.WAKE_DETECTED: (
            CognitiveIntegrationEventKind.USER_UTTERANCE
        ),
        LiveEventKind.USER_SPEECH_STARTED: (
            CognitiveIntegrationEventKind.USER_UTTERANCE
        ),
        LiveEventKind.USER_SPEECH_ENDED: (
            CognitiveIntegrationEventKind.USER_UTTERANCE
        ),
        LiveEventKind.TRANSCRIPT_READY: (
            CognitiveIntegrationEventKind.USER_UTTERANCE
        ),
        LiveEventKind.ASSISTANT_RESPONSE_STARTED: (
            CognitiveIntegrationEventKind.ASSISTANT_RESPONSE
        ),
        LiveEventKind.ASSISTANT_RESPONSE_FINISHED: (
            CognitiveIntegrationEventKind.ASSISTANT_RESPONSE
        ),
        LiveEventKind.INTERRUPTION_REQUESTED: (
            CognitiveIntegrationEventKind.INTERRUPTION
        ),
        LiveEventKind.INTERRUPTION_HANDLED: (
            CognitiveIntegrationEventKind.INTERRUPTION
        ),
        LiveEventKind.MEMORY_CONTEXT_UPDATED: (
            CognitiveIntegrationEventKind.MEMORY_RECALL
        ),
        LiveEventKind.GOAL_UPDATED: CognitiveIntegrationEventKind.GOAL_REQUEST,
        LiveEventKind.PLAN_UPDATED: CognitiveIntegrationEventKind.STATUS,
        LiveEventKind.ENVIRONMENT_CONTEXT_UPDATED: (
            CognitiveIntegrationEventKind.SCREEN_CONTEXT
        ),
        LiveEventKind.DEVELOPER_SIGNAL_RECEIVED: (
            CognitiveIntegrationEventKind.DEVELOPER_BUILD
        ),
        LiveEventKind.HEALTH_CHANGED: (
            CognitiveIntegrationEventKind.SYSTEM_HEALTH
        ),
        LiveEventKind.RECOVERY_STARTED: (
            CognitiveIntegrationEventKind.SYSTEM_HEALTH
        ),
        LiveEventKind.RECOVERY_FINISHED: (
            CognitiveIntegrationEventKind.SYSTEM_HEALTH
        ),
        LiveEventKind.ERROR: CognitiveIntegrationEventKind.WARNING,
    }
    return mapping[kind]


def _working_memory_kind_from_live_kind(
    kind: LiveEventKind,
) -> WorkingMemoryKind | None:
    mapping = {
        LiveEventKind.USER_SPEECH_STARTED: WorkingMemoryKind.CONVERSATION,
        LiveEventKind.USER_SPEECH_ENDED: WorkingMemoryKind.CONVERSATION,
        LiveEventKind.TRANSCRIPT_READY: WorkingMemoryKind.CONVERSATION,
        LiveEventKind.MEMORY_CONTEXT_UPDATED: WorkingMemoryKind.PROJECT,
        LiveEventKind.GOAL_UPDATED: WorkingMemoryKind.OBJECTIVE,
        LiveEventKind.PLAN_UPDATED: WorkingMemoryKind.TASK,
        LiveEventKind.ENVIRONMENT_CONTEXT_UPDATED: (
            WorkingMemoryKind.SCREEN_CONTEXT
        ),
        LiveEventKind.DEVELOPER_SIGNAL_RECEIVED: WorkingMemoryKind.TASK,
        LiveEventKind.HEALTH_CHANGED: WorkingMemoryKind.RISK,
        LiveEventKind.RECOVERY_STARTED: WorkingMemoryKind.RISK,
        LiveEventKind.ERROR: WorkingMemoryKind.RISK,
    }
    return mapping.get(kind)


def _event_kind_for_subsystem_signal(
    source: LiveSubsystem,
) -> LiveEventKind:
    if source == LiveSubsystem.DEVELOPER_PACK:
        return LiveEventKind.DEVELOPER_SIGNAL_RECEIVED
    if source == LiveSubsystem.ENVIRONMENT:
        return LiveEventKind.ENVIRONMENT_CONTEXT_UPDATED
    if source == LiveSubsystem.MEMORY:
        return LiveEventKind.MEMORY_CONTEXT_UPDATED
    if source == LiveSubsystem.HEALTH_MONITOR:
        return LiveEventKind.HEALTH_CHANGED
    return LiveEventKind.ERROR