from __future__ import annotations

from enum import StrEnum
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator

from jarvis.environment.models import (
    EnvironmentDelta,
    EnvironmentEvent,
    EnvironmentEventKind,
    EnvironmentSnapshot,
    EnvironmentSource,
    RecentStateHistory,
    TemporalWorkspaceState,
    TrustCalibration,
)
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class ChangeCause(StrEnum):
    """
    Why the environment changed.

    This is essential for debugging, recovery, intent persistence,
    and distinguishing user-driven changes from JARVIS-driven changes.
    """

    UNKNOWN = "unknown"
    USER_ACTION = "user_action"
    JARVIS_ACTION = "jarvis_action"
    APP_STATE_CHANGE = "app_state_change"
    OS_STATE_CHANGE = "os_state_change"
    FILESYSTEM_CHANGE = "filesystem_change"
    BROWSER_NAVIGATION = "browser_navigation"
    TERMINAL_OUTPUT = "terminal_output"
    TOOL_RESULT = "tool_result"
    ERROR_APPEARED = "error_appeared"
    ERROR_RESOLVED = "error_resolved"
    RECOVERY_ACTION = "recovery_action"
    INTERRUPTION = "interruption"
    VERIFICATION_RESULT = "verification_result"


class StateTransitionKind(StrEnum):
    """
    Type of temporal transition.
    """

    APPEARED = "appeared"
    DISAPPEARED = "disappeared"
    MOVED = "moved"
    CHANGED = "changed"
    FOCUSED = "focused"
    UNFOCUSED = "unfocused"
    FAILED = "failed"
    RECOVERED = "recovered"
    USER_DID = "user_did"
    JARVIS_DID = "jarvis_did"
    APP_CHANGED = "app_changed"
    SNAPSHOT_RECORDED = "snapshot_recorded"


class TimelineQueryKind(StrEnum):
    """
    Timeline query type.
    """

    RECENT_CHANGES = "recent_changes"
    CHANGES_SINCE_SNAPSHOT = "changes_since_snapshot"
    CHANGES_BY_CAUSE = "changes_by_cause"
    CHANGES_BY_TRANSITION = "changes_by_transition"
    FAILURES = "failures"
    RECOVERIES = "recoveries"
    USER_ACTIONS = "user_actions"
    JARVIS_ACTIONS = "jarvis_actions"
    APP_CHANGES = "app_changes"
    RELATED_TO_ENTITY = "related_to_entity"


class TimelineEventKind(StrEnum):
    """
    Runtime event kind.
    """

    SESSION_CREATED = "session_created"
    SNAPSHOT_RECORDED = "snapshot_recorded"
    DELTA_RECORDED = "delta_recorded"
    TRANSITION_RECORDED = "transition_recorded"
    EVENT_RECORDED = "event_recorded"
    QUERY_EXECUTED = "query_executed"
    SESSION_CLEARED = "session_cleared"
    RUNTIME_RESET = "runtime_reset"


class TimelineReason(StrEnum):
    """
    Machine-readable timeline runtime reasons.
    """

    SESSION_CREATED = "session_created"
    SNAPSHOT_RECORDED = "snapshot_recorded"
    DELTA_RECORDED = "delta_recorded"
    TRANSITION_RECORDED = "transition_recorded"
    EVENT_RECORDED = "event_recorded"
    QUERY_EXECUTED = "query_executed"
    SESSION_CLEARED = "session_cleared"
    SESSION_NOT_FOUND = "session_not_found"
    HISTORY_LIMIT_ENFORCED = "history_limit_enforced"
    RUNTIME_RESET = "runtime_reset"


class StateTransition(OrchestrationModel):
    """
    One interpreted environment change.

    Delta says what changed.
    StateTransition says what that change means.
    """

    transition_id: str = Field(default_factory=lambda: f"transition_{uuid4().hex}")
    session_id: str
    kind: StateTransitionKind
    cause: ChangeCause = ChangeCause.UNKNOWN
    entity_id: str | None = None
    entity_kind: str | None = None
    previous_snapshot_id: str | None = None
    current_snapshot_id: str | None = None
    delta_id: str | None = None
    summary: str
    actor: str | None = None
    occurred_at: object = Field(default_factory=utc_now)
    trust: TrustCalibration
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("transition_id", "session_id", "summary")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class TimelineQuery(OrchestrationModel):
    """
    Query over recent environment history.
    """

    query_id: str = Field(default_factory=lambda: f"timeline_query_{uuid4().hex}")
    session_id: str
    kind: TimelineQueryKind
    limit: int = Field(default=10, ge=1)
    cause: ChangeCause | None = None
    transition_kind: StateTransitionKind | None = None
    entity_id: str | None = None
    include_snapshots: bool = False
    include_deltas: bool = True
    include_events: bool = True
    include_transitions: bool = True
    created_at: object = Field(default_factory=utc_now)

    @field_validator("query_id", "session_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class TimelineQueryResult(OrchestrationModel):
    """
    Result of a timeline query.
    """

    query: TimelineQuery
    snapshots: tuple[EnvironmentSnapshot, ...] = ()
    deltas: tuple[EnvironmentDelta, ...] = ()
    events: tuple[EnvironmentEvent, ...] = ()
    transitions: tuple[StateTransition, ...] = ()
    summary: str
    created_at: object = Field(default_factory=utc_now)

    @field_validator("summary")
    @classmethod
    def _required_summary(cls, value: str) -> str:
        return _clean_required(value)


class EnvironmentTimelineSession(OrchestrationModel):
    """
    Timeline session for one workspace/environment context.
    """

    session_id: str = Field(default_factory=lambda: f"timeline_{uuid4().hex}")
    workflow_id: str
    max_history: int = Field(default=50, ge=1)
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "workflow_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class TimelineRuntimeEvent(OrchestrationModel):
    """
    Runtime event emitted by EnvironmentTimelineRuntime.
    """

    event_id: str = Field(default_factory=lambda: f"timeline_event_{uuid4().hex}")
    kind: TimelineEventKind
    reason: TimelineReason
    session_id: str | None = None
    entity_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class TimelineOperationResult(OrchestrationModel):
    """
    Result returned by timeline operations.
    """

    success: bool
    reason: TimelineReason
    session: EnvironmentTimelineSession | None = None
    event: TimelineRuntimeEvent
    message: str

    @field_validator("message")
    @classmethod
    def _required_message(cls, value: str) -> str:
        return _clean_required(value)


class TimelineRuntimeSnapshot(OrchestrationModel):
    """
    Diagnostics for Phase 8 Step 4.
    """

    name: str
    session_count: int = Field(ge=0)
    snapshot_count: int = Field(ge=0)
    delta_count: int = Field(ge=0)
    transition_count: int = Field(ge=0)
    environment_event_count: int = Field(ge=0)
    runtime_event_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    recovery_count: int = Field(ge=0)
    user_action_count: int = Field(ge=0)
    jarvis_action_count: int = Field(ge=0)
    last_reason: TimelineReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        return _clean_required(value)


class EnvironmentTimelineRuntime:
    """
    Phase 8 Step 4 Environment Timeline Runtime.

    Responsibilities:
    - track snapshots, deltas, transitions, and environment events over time
    - explain what changed and why
    - preserve recent history for recovery, debugging, interruption handling,
      and workflow continuity
    - answer timeline queries

    Non-responsibilities:
    - no screen capture
    - no OCR
    - no UI detection
    - no recovery execution
    - no physical action execution
    """

    def __init__(self, *, name: str = "environment_timeline_runtime") -> None:
        cleaned = name.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._sessions: dict[str, EnvironmentTimelineSession] = {}
        self._snapshots: dict[str, list[EnvironmentSnapshot]] = {}
        self._deltas: dict[str, list[EnvironmentDelta]] = {}
        self._transitions: dict[str, list[StateTransition]] = {}
        self._environment_events: dict[str, list[EnvironmentEvent]] = {}
        self._runtime_events: list[TimelineRuntimeEvent] = []
        self._lock = RLock()
        self._last_reason: TimelineReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def create_session(
        self,
        *,
        workflow_id: str,
        max_history: int = 50,
        metadata: dict[str, Any] | None = None,
    ) -> EnvironmentTimelineSession:
        session = EnvironmentTimelineSession(
            workflow_id=workflow_id,
            max_history=max_history,
            metadata=metadata or {},
        )
        event = self._event(
            kind=TimelineEventKind.SESSION_CREATED,
            reason=TimelineReason.SESSION_CREATED,
            session_id=session.session_id,
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._snapshots[session.session_id] = []
            self._deltas[session.session_id] = []
            self._transitions[session.session_id] = []
            self._environment_events[session.session_id] = []
            self._runtime_events.append(event)
            self._last_reason = event.reason

        return session

    def record_snapshot(
        self,
        *,
        session_id: str,
        snapshot: EnvironmentSnapshot,
    ) -> TimelineOperationResult:
        session = self.session_for(session_id)

        if session is None:
            return self._missing_session(session_id)

        transition = StateTransition(
            session_id=session_id,
            kind=StateTransitionKind.SNAPSHOT_RECORDED,
            cause=ChangeCause.UNKNOWN,
            current_snapshot_id=snapshot.snapshot_id,
            summary="environment snapshot recorded",
            trust=snapshot.trust,
        )
        event = self._event(
            kind=TimelineEventKind.SNAPSHOT_RECORDED,
            reason=TimelineReason.SNAPSHOT_RECORDED,
            session_id=session_id,
        )

        with self._lock:
            self._snapshots[session_id].append(snapshot)
            self._transitions[session_id].append(transition)
            self._runtime_events.append(event)
            self._touch_session(session_id)
            self._enforce_limit(session_id)
            self._last_reason = event.reason

        return TimelineOperationResult(
            success=True,
            reason=TimelineReason.SNAPSHOT_RECORDED,
            session=self.session_for(session_id),
            event=event,
            message="environment snapshot recorded",
        )

    def record_delta(
        self,
        *,
        session_id: str,
        delta: EnvironmentDelta,
        cause: ChangeCause = ChangeCause.UNKNOWN,
        summary: str | None = None,
        actor: str | None = None,
    ) -> TimelineOperationResult:
        session = self.session_for(session_id)

        if session is None:
            return self._missing_session(session_id)

        transition = self._transition_from_delta(
            session_id=session_id,
            delta=delta,
            cause=cause,
            summary=summary,
            actor=actor,
        )
        event = self._event(
            kind=TimelineEventKind.DELTA_RECORDED,
            reason=TimelineReason.DELTA_RECORDED,
            session_id=session_id,
            entity_id=transition.entity_id,
        )

        with self._lock:
            self._deltas[session_id].append(delta)
            self._transitions[session_id].append(transition)
            self._runtime_events.append(event)
            self._touch_session(session_id)
            self._enforce_limit(session_id)
            self._last_reason = event.reason

        return TimelineOperationResult(
            success=True,
            reason=TimelineReason.DELTA_RECORDED,
            session=self.session_for(session_id),
            event=event,
            message="environment delta recorded",
        )

    def record_transition(
        self,
        transition: StateTransition,
    ) -> TimelineOperationResult:
        session = self.session_for(transition.session_id)

        if session is None:
            return self._missing_session(transition.session_id)

        event = self._event(
            kind=TimelineEventKind.TRANSITION_RECORDED,
            reason=TimelineReason.TRANSITION_RECORDED,
            session_id=transition.session_id,
            entity_id=transition.entity_id,
        )

        with self._lock:
            self._transitions[transition.session_id].append(transition)
            self._runtime_events.append(event)
            self._touch_session(transition.session_id)
            self._enforce_limit(transition.session_id)
            self._last_reason = event.reason

        return TimelineOperationResult(
            success=True,
            reason=TimelineReason.TRANSITION_RECORDED,
            session=self.session_for(transition.session_id),
            event=event,
            message="state transition recorded",
        )

    def record_environment_event(
        self,
        *,
        session_id: str,
        environment_event: EnvironmentEvent,
    ) -> TimelineOperationResult:
        session = self.session_for(session_id)

        if session is None:
            return self._missing_session(session_id)

        transition = self._transition_from_event(
            session_id=session_id,
            event=environment_event,
        )
        runtime_event = self._event(
            kind=TimelineEventKind.EVENT_RECORDED,
            reason=TimelineReason.EVENT_RECORDED,
            session_id=session_id,
            entity_id=environment_event.element_id
            or environment_event.window_id
            or environment_event.app_id,
        )

        with self._lock:
            self._environment_events[session_id].append(environment_event)
            self._transitions[session_id].append(transition)
            self._runtime_events.append(runtime_event)
            self._touch_session(session_id)
            self._enforce_limit(session_id)
            self._last_reason = runtime_event.reason

        return TimelineOperationResult(
            success=True,
            reason=TimelineReason.EVENT_RECORDED,
            session=self.session_for(session_id),
            event=runtime_event,
            message="environment event recorded",
        )

    def query(self, query: TimelineQuery) -> TimelineQueryResult:
        session = self.session_for(query.session_id)

        if session is None:
            raise ValueError(f"timeline session not found: {query.session_id}")

        transitions = self._query_transitions(query)
        deltas = self._query_deltas(query)
        events = self._query_events(query)
        snapshots = self._query_snapshots(query)

        event = self._event(
            kind=TimelineEventKind.QUERY_EXECUTED,
            reason=TimelineReason.QUERY_EXECUTED,
            session_id=query.session_id,
            metadata={"query_kind": query.kind.value},
        )

        with self._lock:
            self._runtime_events.append(event)
            self._last_reason = event.reason

        return TimelineQueryResult(
            query=query,
            snapshots=snapshots,
            deltas=deltas,
            events=events,
            transitions=transitions,
            summary=self._summary_for_query(query, transitions, deltas, events),
        )

    def recent_state_history(self, session_id: str) -> RecentStateHistory | None:
        session = self.session_for(session_id)

        if session is None:
            return None

        with self._lock:
            return RecentStateHistory(
                snapshots=tuple(self._snapshots[session_id][-session.max_history :]),
                deltas=tuple(self._deltas[session_id][-session.max_history :]),
                max_items=session.max_history,
            )

    def temporal_workspace_state(
        self,
        session_id: str,
    ) -> TemporalWorkspaceState | None:
        session = self.session_for(session_id)

        if session is None:
            return None

        with self._lock:
            snapshots = self._snapshots[session_id]

            if not snapshots:
                return None

            return TemporalWorkspaceState(
                workflow_id=session.workflow_id,
                current_snapshot=snapshots[-1],
                recent_deltas=tuple(self._deltas[session_id][-session.max_history :]),
                last_user_action=self._last_action_summary(
                    session_id=session_id,
                    cause=ChangeCause.USER_ACTION,
                ),
                last_jarvis_action=self._last_action_summary(
                    session_id=session_id,
                    cause=ChangeCause.JARVIS_ACTION,
                ),
            )

    def session_for(self, session_id: str) -> EnvironmentTimelineSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def snapshots_for(
        self,
        session_id: str,
    ) -> tuple[EnvironmentSnapshot, ...]:
        with self._lock:
            return tuple(self._snapshots.get(session_id, ()))

    def deltas_for(self, session_id: str) -> tuple[EnvironmentDelta, ...]:
        with self._lock:
            return tuple(self._deltas.get(session_id, ()))

    def transitions_for(
        self,
        session_id: str,
    ) -> tuple[StateTransition, ...]:
        with self._lock:
            return tuple(self._transitions.get(session_id, ()))

    def environment_events_for(
        self,
        session_id: str,
    ) -> tuple[EnvironmentEvent, ...]:
        with self._lock:
            return tuple(self._environment_events.get(session_id, ()))

    def runtime_events(self) -> tuple[TimelineRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._runtime_events)

    def clear_session(self, session_id: str) -> TimelineOperationResult:
        session = self.session_for(session_id)

        if session is None:
            return self._missing_session(session_id)

        event = self._event(
            kind=TimelineEventKind.SESSION_CLEARED,
            reason=TimelineReason.SESSION_CLEARED,
            session_id=session_id,
        )

        with self._lock:
            self._snapshots[session_id].clear()
            self._deltas[session_id].clear()
            self._transitions[session_id].clear()
            self._environment_events[session_id].clear()
            self._runtime_events.append(event)
            self._touch_session(session_id)
            self._last_reason = event.reason

        return TimelineOperationResult(
            success=True,
            reason=TimelineReason.SESSION_CLEARED,
            session=self.session_for(session_id),
            event=event,
            message="timeline session cleared",
        )

    def snapshot(self) -> TimelineRuntimeSnapshot:
        with self._lock:
            transitions = [
                transition
                for items in self._transitions.values()
                for transition in items
            ]

            return TimelineRuntimeSnapshot(
                name=self.name,
                session_count=len(self._sessions),
                snapshot_count=sum(len(items) for items in self._snapshots.values()),
                delta_count=sum(len(items) for items in self._deltas.values()),
                transition_count=len(transitions),
                environment_event_count=sum(
                    len(items) for items in self._environment_events.values()
                ),
                runtime_event_count=len(self._runtime_events),
                failure_count=sum(
                    1
                    for transition in transitions
                    if transition.kind == StateTransitionKind.FAILED
                ),
                recovery_count=sum(
                    1
                    for transition in transitions
                    if transition.kind == StateTransitionKind.RECOVERED
                ),
                user_action_count=sum(
                    1
                    for transition in transitions
                    if transition.cause == ChangeCause.USER_ACTION
                ),
                jarvis_action_count=sum(
                    1
                    for transition in transitions
                    if transition.cause == ChangeCause.JARVIS_ACTION
                ),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        event = self._event(
            kind=TimelineEventKind.RUNTIME_RESET,
            reason=TimelineReason.RUNTIME_RESET,
        )

        with self._lock:
            self._sessions.clear()
            self._snapshots.clear()
            self._deltas.clear()
            self._transitions.clear()
            self._environment_events.clear()
            self._runtime_events.clear()
            self._runtime_events.append(event)
            self._last_reason = TimelineReason.RUNTIME_RESET

    def _transition_from_delta(
        self,
        *,
        session_id: str,
        delta: EnvironmentDelta,
        cause: ChangeCause,
        summary: str | None,
        actor: str | None,
    ) -> StateTransition:
        kind = self._transition_kind_for_delta(delta=delta, cause=cause)
        entity_id = self._entity_from_delta(delta)

        return StateTransition(
            session_id=session_id,
            kind=kind,
            cause=cause,
            entity_id=entity_id,
            previous_snapshot_id=delta.previous_snapshot_id,
            current_snapshot_id=delta.current_snapshot_id,
            delta_id=delta.delta_id,
            summary=summary or self._summary_for_delta(delta=delta, cause=cause),
            actor=actor,
            trust=delta.trust,
        )

    @staticmethod
    def _transition_kind_for_delta(
        *,
        delta: EnvironmentDelta,
        cause: ChangeCause,
    ) -> StateTransitionKind:
        if cause == ChangeCause.ERROR_APPEARED:
            return StateTransitionKind.FAILED

        if cause == ChangeCause.ERROR_RESOLVED:
            return StateTransitionKind.RECOVERED

        if delta.appeared_elements:
            return StateTransitionKind.APPEARED

        if delta.disappeared_elements:
            return StateTransitionKind.DISAPPEARED

        if (
            delta.changed_windows
            or delta.changed_elements
            or delta.changed_text_regions
        ):
            return StateTransitionKind.CHANGED

        return StateTransitionKind.CHANGED

    @staticmethod
    def _entity_from_delta(delta: EnvironmentDelta) -> str | None:
        for group in (
            delta.appeared_elements,
            delta.disappeared_elements,
            delta.changed_elements,
            delta.changed_windows,
            delta.changed_text_regions,
        ):
            if group:
                return group[0]

        return None

    @staticmethod
    def _summary_for_delta(
        *,
        delta: EnvironmentDelta,
        cause: ChangeCause,
    ) -> str:
        if delta.cause_hint:
            return delta.cause_hint

        if cause == ChangeCause.ERROR_APPEARED:
            return "environment failure appeared"

        if cause == ChangeCause.ERROR_RESOLVED:
            return "environment failure recovered"

        if delta.appeared_elements:
            return "environment element appeared"

        if delta.disappeared_elements:
            return "environment element disappeared"

        return "environment state changed"

    def _transition_from_event(
        self,
        *,
        session_id: str,
        event: EnvironmentEvent,
    ) -> StateTransition:
        kind = self._transition_kind_for_event(event.kind)
        cause = self._cause_for_event(event.kind)
        entity_id = event.element_id or event.window_id or event.app_id

        return StateTransition(
            session_id=session_id,
            kind=kind,
            cause=cause,
            entity_id=entity_id,
            entity_kind=event.kind.value,
            summary=f"environment event recorded: {event.kind.value}",
            trust=event.trust,
            metadata={"event_id": event.event_id, **event.payload},
        )

    @staticmethod
    def _transition_kind_for_event(
        kind: EnvironmentEventKind,
    ) -> StateTransitionKind:
        if kind in {
            EnvironmentEventKind.WINDOW_OPENED,
            EnvironmentEventKind.MODAL_OPENED,
        }:
            return StateTransitionKind.APPEARED

        if kind in {
            EnvironmentEventKind.WINDOW_CLOSED,
            EnvironmentEventKind.MODAL_CLOSED,
        }:
            return StateTransitionKind.DISAPPEARED

        if kind == EnvironmentEventKind.WINDOW_MOVED:
            return StateTransitionKind.MOVED

        if kind == EnvironmentEventKind.WINDOW_FOCUSED:
            return StateTransitionKind.FOCUSED

        if kind == EnvironmentEventKind.APP_CRASHED:
            return StateTransitionKind.FAILED

        if kind == EnvironmentEventKind.RECOVERY_REQUESTED:
            return StateTransitionKind.RECOVERED

        return StateTransitionKind.CHANGED

    @staticmethod
    def _cause_for_event(kind: EnvironmentEventKind) -> ChangeCause:
        if kind == EnvironmentEventKind.APP_CRASHED:
            return ChangeCause.APP_STATE_CHANGE

        if kind == EnvironmentEventKind.FILE_CHANGED:
            return ChangeCause.FILESYSTEM_CHANGE

        if kind == EnvironmentEventKind.CLIPBOARD_CHANGED:
            return ChangeCause.USER_ACTION

        if kind == EnvironmentEventKind.RECOVERY_REQUESTED:
            return ChangeCause.RECOVERY_ACTION

        if kind == EnvironmentEventKind.VERIFICATION_COMPLETED:
            return ChangeCause.VERIFICATION_RESULT

        return ChangeCause.APP_STATE_CHANGE

    def _query_transitions(
        self,
        query: TimelineQuery,
    ) -> tuple[StateTransition, ...]:
        if not query.include_transitions:
            return ()

        with self._lock:
            transitions = list(self._transitions.get(query.session_id, ()))

        filtered = self._filter_transitions(query=query, transitions=transitions)

        return tuple(filtered[-query.limit :])

    def _filter_transitions(
        self,
        *,
        query: TimelineQuery,
        transitions: list[StateTransition],
    ) -> list[StateTransition]:
        if query.kind == TimelineQueryKind.CHANGES_BY_CAUSE and query.cause:
            return [
                transition
                for transition in transitions
                if transition.cause == query.cause
            ]

        if (
            query.kind == TimelineQueryKind.CHANGES_BY_TRANSITION
            and query.transition_kind
        ):
            return [
                transition
                for transition in transitions
                if transition.kind == query.transition_kind
            ]

        if query.kind == TimelineQueryKind.FAILURES:
            return [
                transition
                for transition in transitions
                if transition.kind == StateTransitionKind.FAILED
            ]

        if query.kind == TimelineQueryKind.RECOVERIES:
            return [
                transition
                for transition in transitions
                if transition.kind == StateTransitionKind.RECOVERED
            ]

        if query.kind == TimelineQueryKind.USER_ACTIONS:
            return [
                transition
                for transition in transitions
                if transition.cause == ChangeCause.USER_ACTION
            ]

        if query.kind == TimelineQueryKind.JARVIS_ACTIONS:
            return [
                transition
                for transition in transitions
                if transition.cause == ChangeCause.JARVIS_ACTION
            ]

        if query.kind == TimelineQueryKind.RELATED_TO_ENTITY and query.entity_id:
            return [
                transition
                for transition in transitions
                if transition.entity_id == query.entity_id
            ]

        if query.kind == TimelineQueryKind.APP_CHANGES:
            return [
                transition
                for transition in transitions
                if transition.cause == ChangeCause.APP_STATE_CHANGE
            ]

        return transitions

    def _query_deltas(self, query: TimelineQuery) -> tuple[EnvironmentDelta, ...]:
        if not query.include_deltas:
            return ()

        with self._lock:
            return tuple(self._deltas.get(query.session_id, ())[-query.limit :])

    def _query_events(self, query: TimelineQuery) -> tuple[EnvironmentEvent, ...]:
        if not query.include_events:
            return ()

        with self._lock:
            return tuple(
                self._environment_events.get(query.session_id, ())[-query.limit :]
            )

    def _query_snapshots(
        self,
        query: TimelineQuery,
    ) -> tuple[EnvironmentSnapshot, ...]:
        if not query.include_snapshots:
            return ()

        with self._lock:
            return tuple(self._snapshots.get(query.session_id, ())[-query.limit :])

    @staticmethod
    def _summary_for_query(
        query: TimelineQuery,
        transitions: tuple[StateTransition, ...],
        deltas: tuple[EnvironmentDelta, ...],
        events: tuple[EnvironmentEvent, ...],
    ) -> str:
        return (
            f"timeline query {query.kind.value}: "
            f"{len(transitions)} transitions, "
            f"{len(deltas)} deltas, "
            f"{len(events)} events"
        )

    def _last_action_summary(
        self,
        *,
        session_id: str,
        cause: ChangeCause,
    ) -> str | None:
        with self._lock:
            for transition in reversed(self._transitions.get(session_id, ())):
                if transition.cause == cause:
                    return transition.summary

        return None

    def _touch_session(self, session_id: str) -> None:
        session = self._sessions[session_id]
        self._sessions[session_id] = session.model_copy(
            update={"updated_at": utc_now()}
        )

    def _enforce_limit(self, session_id: str) -> None:
        session = self._sessions[session_id]
        limit = session.max_history

        self._snapshots[session_id] = self._snapshots[session_id][-limit:]
        self._deltas[session_id] = self._deltas[session_id][-limit:]
        self._transitions[session_id] = self._transitions[session_id][-limit:]
        self._environment_events[session_id] = (
            self._environment_events[session_id][-limit:]
        )

    @staticmethod
    def _event(
        *,
        kind: TimelineEventKind,
        reason: TimelineReason,
        session_id: str | None = None,
        entity_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TimelineRuntimeEvent:
        return TimelineRuntimeEvent(
            kind=kind,
            reason=reason,
            session_id=session_id,
            entity_id=entity_id,
            metadata=metadata or {},
        )

    @staticmethod
    def _missing_session(session_id: str) -> TimelineOperationResult:
        event = TimelineRuntimeEvent(
            kind=TimelineEventKind.SESSION_CLEARED,
            reason=TimelineReason.SESSION_NOT_FOUND,
            session_id=session_id,
        )

        return TimelineOperationResult(
            success=False,
            reason=TimelineReason.SESSION_NOT_FOUND,
            event=event,
            message="timeline session not found",
        )


def _clean_required(value: str) -> str:
    cleaned = value.strip()

    if not cleaned:
        raise ValueError("field cannot be empty.")

    return cleaned


def trusted_timeline_observation(
    *,
    source: EnvironmentSource = EnvironmentSource.OS_OBSERVER,
    reason: str = "timeline observation",
) -> TrustCalibration:
    """
    Helper for timeline tests and fake-first runtime construction.
    """

    return TrustCalibration(
        confidence=0.95,
        stability=0.95,
        ambiguity=0.0,
        source=source,
        reason=reason,
    )