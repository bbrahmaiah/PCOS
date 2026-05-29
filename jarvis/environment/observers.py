from __future__ import annotations

from collections import deque
from enum import StrEnum
from threading import RLock
from typing import Any, Protocol
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.environment.models import (
    EnvironmentEvent,
    EnvironmentEventKind,
    EnvironmentSource,
    ScreenRegion,
    TrustCalibration,
)
from jarvis.environment.timeline import (
    EnvironmentTimelineRuntime,
)
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class EnvironmentObserverKind(StrEnum):
    """
    Phase 8 event-driven observer kinds.

    Observers report reality changes. They do not perform cognition,
    screen capture, OCR, or physical action.
    """

    ACTIVE_WINDOW = "active_window_observer"
    DISPLAY = "display_observer"
    CURSOR = "cursor_observer"
    CLIPBOARD = "clipboard_observer"
    FILESYSTEM = "filesystem_observer"
    PROCESS = "process_observer"
    BROWSER = "browser_observer"
    APP_LIFECYCLE = "app_lifecycle_observer"
    MODAL = "modal_observer"


class EnvironmentObserverStatus(StrEnum):
    """
    Observer lifecycle status.
    """

    CREATED = "created"
    STARTED = "started"
    STOPPED = "stopped"
    DEGRADED = "degraded"
    FAILED = "failed"
    DISABLED = "disabled"


class EnvironmentObserverHealth(StrEnum):
    """
    Observer health.
    """

    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    STALE = "stale"


class ObserverBackboneEventKind(StrEnum):
    """
    Backbone runtime event kind.
    """

    OBSERVER_REGISTERED = "observer_registered"
    OBSERVER_STARTED = "observer_started"
    OBSERVER_STOPPED = "observer_stopped"
    OBSERVER_FAILED = "observer_failed"
    OBSERVER_EVENT_EMITTED = "observer_event_emitted"
    OBSERVER_EVENT_DEDUPED = "observer_event_deduped"
    EVENT_ROUTED_TO_TIMELINE = "event_routed_to_timeline"
    RUNTIME_RESET = "runtime_reset"


class ObserverBackboneReason(StrEnum):
    """
    Machine-readable observer backbone reason.
    """

    OBSERVER_REGISTERED = "observer_registered"
    OBSERVER_DUPLICATE_REJECTED = "observer_duplicate_rejected"
    OBSERVER_STARTED = "observer_started"
    OBSERVER_STOPPED = "observer_stopped"
    OBSERVER_FAILED = "observer_failed"
    OBSERVER_NOT_FOUND = "observer_not_found"
    EVENT_ACCEPTED = "event_accepted"
    EVENT_DEDUPED = "event_deduped"
    EVENT_ROUTED_TO_TIMELINE = "event_routed_to_timeline"
    TIMELINE_SESSION_MISSING = "timeline_session_missing"
    RUNTIME_RESET = "runtime_reset"


class ObserverEventPriority(StrEnum):
    """
    Priority of observer events.

    This is not visual priority arbitration. This only classifies observer
    signal importance before routing.
    """

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"


class ObserverEmissionMode(StrEnum):
    """
    How an observer emits events.
    """

    EVENT_DRIVEN = "event_driven"
    SUBSCRIPTION = "subscription"
    CALLBACK = "callback"
    ADAPTER_PUSH = "adapter_push"


class EnvironmentObserverDescriptor(OrchestrationModel):
    """
    Static descriptor for one observer.

    Every observer must declare what it watches and how it emits events.
    """

    observer_id: str = Field(default_factory=lambda: f"observer_{uuid4().hex}")
    kind: EnvironmentObserverKind
    name: str
    watched_events: tuple[EnvironmentEventKind, ...]
    source: EnvironmentSource = EnvironmentSource.OS_OBSERVER
    emission_mode: ObserverEmissionMode = ObserverEmissionMode.EVENT_DRIVEN
    priority: ObserverEventPriority = ObserverEventPriority.NORMAL
    background_allowed: bool = True
    required: bool = True
    enabled: bool = True
    status: EnvironmentObserverStatus = EnvironmentObserverStatus.CREATED
    health: EnvironmentObserverHealth = EnvironmentObserverHealth.UNKNOWN
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("observer_id", "name")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _must_watch_events(self) -> EnvironmentObserverDescriptor:
        if not self.watched_events:
            raise ValueError("observer must watch at least one event kind.")

        return self


class ObserverSignal(OrchestrationModel):
    """
    Raw observer signal.

    This is the fake-first input contract used by test adapters and later
    OS-specific observers.
    """

    signal_id: str = Field(default_factory=lambda: f"observersignal_{uuid4().hex}")
    observer_kind: EnvironmentObserverKind
    event_kind: EnvironmentEventKind
    source: EnvironmentSource = EnvironmentSource.OS_OBSERVER
    app_id: str | None = None
    window_id: str | None = None
    element_id: str | None = None
    region: ScreenRegion | None = None
    confidence: float = Field(default=0.95, ge=0.0, le=1.0)
    stability: float = Field(default=0.95, ge=0.0, le=1.0)
    ambiguity: float = Field(default=0.0, ge=0.0, le=1.0)
    payload: dict[str, Any] = Field(default_factory=dict)
    emitted_at: object = Field(default_factory=utc_now)

    @field_validator("signal_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class ObserverBackboneRuntimeEvent(OrchestrationModel):
    """
    Observer backbone observability event.
    """

    event_id: str = Field(default_factory=lambda: f"observer_event_{uuid4().hex}")
    kind: ObserverBackboneEventKind
    reason: ObserverBackboneReason
    observer_kind: EnvironmentObserverKind | None = None
    environment_event_kind: EnvironmentEventKind | None = None
    timeline_session_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class ObserverBackboneResult(OrchestrationModel):
    """
    Result of observer backbone operation.
    """

    success: bool
    reason: ObserverBackboneReason
    event: ObserverBackboneRuntimeEvent
    environment_event: EnvironmentEvent | None = None
    observer: EnvironmentObserverDescriptor | None = None
    message: str

    @field_validator("message")
    @classmethod
    def _required_message(cls, value: str) -> str:
        return _clean_required(value)


class ObserverBackboneSnapshot(OrchestrationModel):
    """
    Diagnostics for Phase 8 Step 5.
    """

    name: str
    observer_count: int = Field(ge=0)
    started_count: int = Field(ge=0)
    stopped_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    degraded_count: int = Field(ge=0)
    emitted_event_count: int = Field(ge=0)
    deduped_event_count: int = Field(ge=0)
    timeline_routed_count: int = Field(ge=0)
    runtime_event_count: int = Field(ge=0)
    last_reason: ObserverBackboneReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        return _clean_required(value)


class EnvironmentObserverAdapter(Protocol):
    """
    Protocol for later OS/browser/filesystem-specific observer adapters.

    Step 5 defines the interface only. Real adapters come later.
    """

    @property
    def kind(self) -> EnvironmentObserverKind:
        ...

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def health(self) -> EnvironmentObserverHealth:
        ...


class EnvironmentObserverBackbone:
    """
    Phase 8 Step 5 Environment Event & Observer Backbone.

    Responsibilities:
    - register observer descriptors
    - start/stop observer lifecycle
    - accept observer signals
    - convert signals into trusted EnvironmentEvents
    - dedupe repeated reality events
    - route events into EnvironmentTimelineRuntime
    - expose runtime diagnostics

    Non-responsibilities:
    - no screen capture
    - no OCR
    - no UI detection
    - no polling loops
    - no physical action execution
    """

    def __init__(
        self,
        *,
        name: str = "environment_observer_backbone",
        timeline: EnvironmentTimelineRuntime | None = None,
        dedupe_window: int = 20,
    ) -> None:
        cleaned = name.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        if dedupe_window < 1:
            raise ValueError("dedupe_window must be positive.")

        self._name = cleaned
        self._timeline = timeline
        self._dedupe_window = dedupe_window
        self._observers: dict[
            EnvironmentObserverKind,
            EnvironmentObserverDescriptor
        ] = {}
        self._events: list[EnvironmentEvent] = []
        self._runtime_events: list[ObserverBackboneRuntimeEvent] = []
        self._recent_fingerprints: deque[str] = deque(maxlen=dedupe_window)
        self._deduped_count = 0
        self._timeline_routed_count = 0
        self._lock = RLock()
        self._last_reason: ObserverBackboneReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def register_observer(
        self,
        observer: EnvironmentObserverDescriptor,
        *,
        replace: bool = False,
    ) -> ObserverBackboneResult:
        with self._lock:
            existing = self._observers.get(observer.kind)

            if existing is not None and not replace:
                event = self._runtime_event(
                    kind=ObserverBackboneEventKind.OBSERVER_REGISTERED,
                    reason=ObserverBackboneReason.OBSERVER_DUPLICATE_REJECTED,
                    observer_kind=observer.kind,
                )
                self._runtime_events.append(event)
                self._last_reason = event.reason

                return ObserverBackboneResult(
                    success=False,
                    reason=event.reason,
                    event=event,
                    observer=existing,
                    message="observer already registered",
                )

            self._observers[observer.kind] = observer
            event = self._runtime_event(
                kind=ObserverBackboneEventKind.OBSERVER_REGISTERED,
                reason=ObserverBackboneReason.OBSERVER_REGISTERED,
                observer_kind=observer.kind,
            )
            self._runtime_events.append(event)
            self._last_reason = event.reason

        return ObserverBackboneResult(
            success=True,
            reason=ObserverBackboneReason.OBSERVER_REGISTERED,
            event=event,
            observer=observer,
            message="observer registered",
        )

    def register_defaults(self) -> tuple[ObserverBackboneResult, ...]:
        return tuple(
            self.register_observer(observer)
            for observer in default_environment_observers()
        )

    def start_observer(
        self,
        kind: EnvironmentObserverKind,
    ) -> ObserverBackboneResult:
        return self._set_observer_status(
            kind=kind,
            status=EnvironmentObserverStatus.STARTED,
            health=EnvironmentObserverHealth.HEALTHY,
            event_kind=ObserverBackboneEventKind.OBSERVER_STARTED,
            reason=ObserverBackboneReason.OBSERVER_STARTED,
            message="observer started",
        )

    def stop_observer(
        self,
        kind: EnvironmentObserverKind,
    ) -> ObserverBackboneResult:
        return self._set_observer_status(
            kind=kind,
            status=EnvironmentObserverStatus.STOPPED,
            health=EnvironmentObserverHealth.UNKNOWN,
            event_kind=ObserverBackboneEventKind.OBSERVER_STOPPED,
            reason=ObserverBackboneReason.OBSERVER_STOPPED,
            message="observer stopped",
        )

    def fail_observer(
        self,
        kind: EnvironmentObserverKind,
        *,
        reason_text: str,
    ) -> ObserverBackboneResult:
        return self._set_observer_status(
            kind=kind,
            status=EnvironmentObserverStatus.FAILED,
            health=EnvironmentObserverHealth.FAILED,
            event_kind=ObserverBackboneEventKind.OBSERVER_FAILED,
            reason=ObserverBackboneReason.OBSERVER_FAILED,
            message=reason_text,
        )

    def emit_signal(
        self,
        signal: ObserverSignal,
        *,
        timeline_session_id: str | None = None,
    ) -> ObserverBackboneResult:
        observer = self.observer_for(signal.observer_kind)

        if observer is None:
            event = self._runtime_event(
                kind=ObserverBackboneEventKind.OBSERVER_EVENT_EMITTED,
                reason=ObserverBackboneReason.OBSERVER_NOT_FOUND,
                observer_kind=signal.observer_kind,
                environment_event_kind=signal.event_kind,
            )

            with self._lock:
                self._runtime_events.append(event)
                self._last_reason = event.reason

            return ObserverBackboneResult(
                success=False,
                reason=ObserverBackboneReason.OBSERVER_NOT_FOUND,
                event=event,
                message="observer not registered",
            )

        if signal.event_kind not in observer.watched_events:
            event = self._runtime_event(
                kind=ObserverBackboneEventKind.OBSERVER_EVENT_EMITTED,
                reason=ObserverBackboneReason.OBSERVER_NOT_FOUND,
                observer_kind=signal.observer_kind,
                environment_event_kind=signal.event_kind,
                metadata={"detail": "observer does not watch this event kind"},
            )

            with self._lock:
                self._runtime_events.append(event)
                self._last_reason = event.reason

            return ObserverBackboneResult(
                success=False,
                reason=ObserverBackboneReason.OBSERVER_NOT_FOUND,
                event=event,
                observer=observer,
                message="observer does not watch this event kind",
            )

        environment_event = self._event_from_signal(signal)
        fingerprint = self._fingerprint(environment_event)

        with self._lock:
            if fingerprint in self._recent_fingerprints:
                runtime_event = self._runtime_event(
                    kind=ObserverBackboneEventKind.OBSERVER_EVENT_DEDUPED,
                    reason=ObserverBackboneReason.EVENT_DEDUPED,
                    observer_kind=signal.observer_kind,
                    environment_event_kind=signal.event_kind,
                )
                self._deduped_count += 1
                self._runtime_events.append(runtime_event)
                self._last_reason = runtime_event.reason

                return ObserverBackboneResult(
                    success=True,
                    reason=ObserverBackboneReason.EVENT_DEDUPED,
                    event=runtime_event,
                    environment_event=environment_event,
                    observer=observer,
                    message="observer event deduped",
                )

            self._recent_fingerprints.append(fingerprint)
            self._events.append(environment_event)
            runtime_event = self._runtime_event(
                kind=ObserverBackboneEventKind.OBSERVER_EVENT_EMITTED,
                reason=ObserverBackboneReason.EVENT_ACCEPTED,
                observer_kind=signal.observer_kind,
                environment_event_kind=signal.event_kind,
            )
            self._runtime_events.append(runtime_event)
            self._last_reason = runtime_event.reason

        if timeline_session_id is not None:
            return self.route_event_to_timeline(
                environment_event=environment_event,
                timeline_session_id=timeline_session_id,
                observer=observer,
            )

        return ObserverBackboneResult(
            success=True,
            reason=ObserverBackboneReason.EVENT_ACCEPTED,
            event=runtime_event,
            environment_event=environment_event,
            observer=observer,
            message="observer event accepted",
        )

    def route_event_to_timeline(
        self,
        *,
        environment_event: EnvironmentEvent,
        timeline_session_id: str,
        observer: EnvironmentObserverDescriptor | None = None,
    ) -> ObserverBackboneResult:
        if self._timeline is None:
            runtime_event = self._runtime_event(
                kind=ObserverBackboneEventKind.EVENT_ROUTED_TO_TIMELINE,
                reason=ObserverBackboneReason.TIMELINE_SESSION_MISSING,
                environment_event_kind=environment_event.kind,
                timeline_session_id=timeline_session_id,
                metadata={"detail": "timeline runtime not configured"},
            )

            with self._lock:
                self._runtime_events.append(runtime_event)
                self._last_reason = runtime_event.reason

            return ObserverBackboneResult(
                success=False,
                reason=ObserverBackboneReason.TIMELINE_SESSION_MISSING,
                event=runtime_event,
                environment_event=environment_event,
                observer=observer,
                message="timeline runtime not configured",
            )

        result = self._timeline.record_environment_event(
            session_id=timeline_session_id,
            environment_event=environment_event,
        )

        if not result.success:
            runtime_event = self._runtime_event(
                kind=ObserverBackboneEventKind.EVENT_ROUTED_TO_TIMELINE,
                reason=ObserverBackboneReason.TIMELINE_SESSION_MISSING,
                environment_event_kind=environment_event.kind,
                timeline_session_id=timeline_session_id,
            )

            with self._lock:
                self._runtime_events.append(runtime_event)
                self._last_reason = runtime_event.reason

            return ObserverBackboneResult(
                success=False,
                reason=ObserverBackboneReason.TIMELINE_SESSION_MISSING,
                event=runtime_event,
                environment_event=environment_event,
                observer=observer,
                message="timeline session missing",
            )

        runtime_event = self._runtime_event(
            kind=ObserverBackboneEventKind.EVENT_ROUTED_TO_TIMELINE,
            reason=ObserverBackboneReason.EVENT_ROUTED_TO_TIMELINE,
            environment_event_kind=environment_event.kind,
            timeline_session_id=timeline_session_id,
        )

        with self._lock:
            self._timeline_routed_count += 1
            self._runtime_events.append(runtime_event)
            self._last_reason = runtime_event.reason

        return ObserverBackboneResult(
            success=True,
            reason=ObserverBackboneReason.EVENT_ROUTED_TO_TIMELINE,
            event=runtime_event,
            environment_event=environment_event,
            observer=observer,
            message="observer event routed to timeline",
        )

    def observer_for(
        self,
        kind: EnvironmentObserverKind,
    ) -> EnvironmentObserverDescriptor | None:
        with self._lock:
            return self._observers.get(kind)

    def observers(self) -> tuple[EnvironmentObserverDescriptor, ...]:
        with self._lock:
            return tuple(self._observers.values())

    def environment_events(self) -> tuple[EnvironmentEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def runtime_events(self) -> tuple[ObserverBackboneRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._runtime_events)

    def is_ready(self) -> bool:
        registered = {observer.kind for observer in self.observers()}
        required = {observer.kind for observer in default_environment_observers()}

        return required.issubset(registered)

    def snapshot(self) -> ObserverBackboneSnapshot:
        with self._lock:
            observers = tuple(self._observers.values())

            return ObserverBackboneSnapshot(
                name=self.name,
                observer_count=len(observers),
                started_count=sum(
                    1
                    for observer in observers
                    if observer.status == EnvironmentObserverStatus.STARTED
                ),
                stopped_count=sum(
                    1
                    for observer in observers
                    if observer.status == EnvironmentObserverStatus.STOPPED
                ),
                failed_count=sum(
                    1
                    for observer in observers
                    if observer.status == EnvironmentObserverStatus.FAILED
                ),
                degraded_count=sum(
                    1
                    for observer in observers
                    if observer.health == EnvironmentObserverHealth.DEGRADED
                ),
                emitted_event_count=len(self._events),
                deduped_event_count=self._deduped_count,
                timeline_routed_count=self._timeline_routed_count,
                runtime_event_count=len(self._runtime_events),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        runtime_event = self._runtime_event(
            kind=ObserverBackboneEventKind.RUNTIME_RESET,
            reason=ObserverBackboneReason.RUNTIME_RESET,
        )

        with self._lock:
            self._observers.clear()
            self._events.clear()
            self._runtime_events.clear()
            self._recent_fingerprints.clear()
            self._deduped_count = 0
            self._timeline_routed_count = 0
            self._runtime_events.append(runtime_event)
            self._last_reason = runtime_event.reason

    def _set_observer_status(
        self,
        *,
        kind: EnvironmentObserverKind,
        status: EnvironmentObserverStatus,
        health: EnvironmentObserverHealth,
        event_kind: ObserverBackboneEventKind,
        reason: ObserverBackboneReason,
        message: str,
    ) -> ObserverBackboneResult:
        with self._lock:
            observer = self._observers.get(kind)

            if observer is None:
                event = self._runtime_event(
                    kind=event_kind,
                    reason=ObserverBackboneReason.OBSERVER_NOT_FOUND,
                    observer_kind=kind,
                )
                self._runtime_events.append(event)
                self._last_reason = event.reason

                return ObserverBackboneResult(
                    success=False,
                    reason=ObserverBackboneReason.OBSERVER_NOT_FOUND,
                    event=event,
                    message="observer not found",
                )

            updated = observer.model_copy(
                update={"status": status, "health": health}
            )
            self._observers[kind] = updated
            event = self._runtime_event(
                kind=event_kind,
                reason=reason,
                observer_kind=kind,
            )
            self._runtime_events.append(event)
            self._last_reason = event.reason

        return ObserverBackboneResult(
            success=True,
            reason=reason,
            event=event,
            observer=updated,
            message=message,
        )

    @staticmethod
    def _event_from_signal(signal: ObserverSignal) -> EnvironmentEvent:
        return EnvironmentEvent(
            kind=signal.event_kind,
            source=signal.source,
            app_id=signal.app_id,
            window_id=signal.window_id,
            element_id=signal.element_id,
            region=signal.region,
            trust=TrustCalibration(
                confidence=signal.confidence,
                stability=signal.stability,
                ambiguity=signal.ambiguity,
                source=signal.source,
                reason=f"observer signal from {signal.observer_kind.value}",
            ),
            payload=signal.payload,
        )

    @staticmethod
    def _fingerprint(event: EnvironmentEvent) -> str:
        return "|".join(
            (
                event.kind.value,
                event.source.value,
                event.app_id or "",
                event.window_id or "",
                event.element_id or "",
                str(event.payload),
            )
        )

    @staticmethod
    def _runtime_event(
        *,
        kind: ObserverBackboneEventKind,
        reason: ObserverBackboneReason,
        observer_kind: EnvironmentObserverKind | None = None,
        environment_event_kind: EnvironmentEventKind | None = None,
        timeline_session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ObserverBackboneRuntimeEvent:
        return ObserverBackboneRuntimeEvent(
            kind=kind,
            reason=reason,
            observer_kind=observer_kind,
            environment_event_kind=environment_event_kind,
            timeline_session_id=timeline_session_id,
            metadata=metadata or {},
        )


class ActiveWindowObserver:
    kind = EnvironmentObserverKind.ACTIVE_WINDOW


class DisplayObserver:
    kind = EnvironmentObserverKind.DISPLAY


class CursorObserver:
    kind = EnvironmentObserverKind.CURSOR


class ClipboardObserver:
    kind = EnvironmentObserverKind.CLIPBOARD


class FilesystemObserver:
    kind = EnvironmentObserverKind.FILESYSTEM


class ProcessObserver:
    kind = EnvironmentObserverKind.PROCESS


class BrowserObserver:
    kind = EnvironmentObserverKind.BROWSER


class AppLifecycleObserver:
    kind = EnvironmentObserverKind.APP_LIFECYCLE


class ModalObserver:
    kind = EnvironmentObserverKind.MODAL


def default_environment_observers() -> tuple[EnvironmentObserverDescriptor, ...]:
    """
    Canonical Phase 8 observer set.

    These are descriptors only. Real OS hooks come later.
    """

    return (
        EnvironmentObserverDescriptor(
            kind=EnvironmentObserverKind.ACTIVE_WINDOW,
            name="ActiveWindowObserver",
            watched_events=(
                EnvironmentEventKind.WINDOW_FOCUSED,
                EnvironmentEventKind.WINDOW_OPENED,
                EnvironmentEventKind.WINDOW_CLOSED,
                EnvironmentEventKind.WINDOW_MOVED,
            ),
            priority=ObserverEventPriority.HIGH,
        ),
        EnvironmentObserverDescriptor(
            kind=EnvironmentObserverKind.DISPLAY,
            name="DisplayObserver",
            watched_events=(EnvironmentEventKind.DISPLAY_CHANGED,),
            priority=ObserverEventPriority.HIGH,
        ),
        EnvironmentObserverDescriptor(
            kind=EnvironmentObserverKind.CURSOR,
            name="CursorObserver",
            watched_events=(EnvironmentEventKind.CURSOR_MOVED,),
            priority=ObserverEventPriority.LOW,
        ),
        EnvironmentObserverDescriptor(
            kind=EnvironmentObserverKind.CLIPBOARD,
            name="ClipboardObserver",
            watched_events=(EnvironmentEventKind.CLIPBOARD_CHANGED,),
            priority=ObserverEventPriority.HIGH,
        ),
        EnvironmentObserverDescriptor(
            kind=EnvironmentObserverKind.FILESYSTEM,
            name="FilesystemObserver",
            watched_events=(EnvironmentEventKind.FILE_CHANGED,),
            priority=ObserverEventPriority.NORMAL,
        ),
        EnvironmentObserverDescriptor(
            kind=EnvironmentObserverKind.PROCESS,
            name="ProcessObserver",
            watched_events=(
                EnvironmentEventKind.APP_STARTED,
                EnvironmentEventKind.APP_EXITED,
                EnvironmentEventKind.APP_CRASHED,
            ),
            priority=ObserverEventPriority.HIGH,
        ),
        EnvironmentObserverDescriptor(
            kind=EnvironmentObserverKind.BROWSER,
            name="BrowserObserver",
            watched_events=(EnvironmentEventKind.UI_CHANGED,),
            priority=ObserverEventPriority.NORMAL,
        ),
        EnvironmentObserverDescriptor(
            kind=EnvironmentObserverKind.APP_LIFECYCLE,
            name="AppLifecycleObserver",
            watched_events=(
                EnvironmentEventKind.APP_STARTED,
                EnvironmentEventKind.APP_EXITED,
                EnvironmentEventKind.APP_CRASHED,
            ),
            priority=ObserverEventPriority.HIGH,
        ),
        EnvironmentObserverDescriptor(
            kind=EnvironmentObserverKind.MODAL,
            name="ModalObserver",
            watched_events=(
                EnvironmentEventKind.MODAL_OPENED,
                EnvironmentEventKind.MODAL_CLOSED,
            ),
            priority=ObserverEventPriority.HIGH,
        ),
    )


def _clean_required(value: str) -> str:
    cleaned = value.strip()

    if not cleaned:
        raise ValueError("field cannot be empty.")

    return cleaned