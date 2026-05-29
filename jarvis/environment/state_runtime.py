from __future__ import annotations

from enum import StrEnum
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.environment.models import (
    AppKind,
    AppState,
    DisplayKind,
    DisplayState,
    EnvironmentEvent,
    EnvironmentEventKind,
    EnvironmentSnapshot,
    EnvironmentSource,
    EnvironmentState,
    ScreenPoint,
    ScreenRegion,
    TrustCalibration,
    WindowMode,
    WindowState,
)
from jarvis.environment.timeline import EnvironmentTimelineRuntime
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class ClipboardSensitivity(StrEnum):
    """
    Clipboard sensitivity hint.

    Raw clipboard content must not be stored in environment state.
    """

    UNKNOWN = "unknown"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    SECRET = "secret"


class ProcessStatus(StrEnum):
    """
    Process runtime status.
    """

    UNKNOWN = "unknown"
    RUNNING = "running"
    EXITED = "exited"
    CRASHED = "crashed"
    UNRESPONSIVE = "unresponsive"


class ModalKind(StrEnum):
    """
    Known modal kinds.
    """

    UNKNOWN = "unknown"
    CONFIRMATION = "confirmation"
    ERROR = "error"
    FILE_PICKER = "file_picker"
    SAVE_DIALOG = "save_dialog"
    LOGIN = "login"
    PERMISSION = "permission"


class AppResponsiveness(StrEnum):
    """
    App responsiveness state.
    """

    UNKNOWN = "unknown"
    RESPONSIVE = "responsive"
    SLOW = "slow"
    UNRESPONSIVE = "unresponsive"
    CRASHED = "crashed"


class EnvironmentStateReason(StrEnum):
    """
    Machine-readable state runtime reason.
    """

    SESSION_CREATED = "session_created"
    DISPLAY_REGISTERED = "display_registered"
    APP_REGISTERED = "app_registered"
    WINDOW_REGISTERED = "window_registered"
    WINDOW_FOCUSED = "window_focused"
    WINDOW_CLOSED = "window_closed"
    CURSOR_UPDATED = "cursor_updated"
    CLIPBOARD_UPDATED = "clipboard_updated"
    PROCESS_UPDATED = "process_updated"
    MODAL_OPENED = "modal_opened"
    MODAL_CLOSED = "modal_closed"
    RESPONSIVENESS_UPDATED = "responsiveness_updated"
    EVENT_APPLIED = "event_applied"
    SNAPSHOT_BUILT = "snapshot_built"
    ENVIRONMENT_STATE_BUILT = "environment_state_built"
    SESSION_NOT_FOUND = "session_not_found"
    RUNTIME_RESET = "runtime_reset"


class EnvironmentStateEventKind(StrEnum):
    """
    Runtime event kind.
    """

    SESSION_CREATED = "session_created"
    STATE_MUTATED = "state_mutated"
    EVENT_APPLIED = "event_applied"
    SNAPSHOT_BUILT = "snapshot_built"
    RUNTIME_RESET = "runtime_reset"


class LiveWindowRegistry(OrchestrationModel):
    """
    Current known windows.
    """

    windows: dict[str, WindowState] = Field(default_factory=dict)
    focused_window_id: str | None = None

    def visible_windows(self) -> tuple[WindowState, ...]:
        return tuple(window for window in self.windows.values() if window.visible)


class DisplayTopology(OrchestrationModel):
    """
    Current display topology.
    """

    displays: dict[str, DisplayState] = Field(default_factory=dict)
    primary_display_id: str | None = None

    @model_validator(mode="after")
    def _primary_must_exist(self) -> DisplayTopology:
        if self.primary_display_id is None:
            return self

        if self.primary_display_id not in self.displays:
            raise ValueError("primary_display_id must exist in displays.")

        return self


class CursorTracker(OrchestrationModel):
    """
    Current cursor state.
    """

    position: ScreenPoint | None = None
    last_significant_position: ScreenPoint | None = None
    moved_significantly: bool = False
    updated_at: object = Field(default_factory=utc_now)


class ClipboardState(OrchestrationModel):
    """
    Clipboard state without raw clipboard content.
    """

    content_hash: str | None = None
    hint: str | None = None
    sensitivity: ClipboardSensitivity = ClipboardSensitivity.UNKNOWN
    updated_at: object = Field(default_factory=utc_now)

    @field_validator("content_hash", "hint")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None


class ProcessState(OrchestrationModel):
    """
    Process state.
    """

    process_id: str
    app_id: str | None = None
    name: str
    status: ProcessStatus = ProcessStatus.UNKNOWN
    pid: int | None = Field(default=None, ge=0)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("process_id", "name")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class FocusedAppState(OrchestrationModel):
    """
    Focused app/window state.
    """

    app_id: str | None = None
    window_id: str | None = None
    focused_at: object = Field(default_factory=utc_now)


class ModalState(OrchestrationModel):
    """
    Current modal state.
    """

    modal_id: str = Field(default_factory=lambda: f"modal_{uuid4().hex}")
    window_id: str | None = None
    app_id: str | None = None
    kind: ModalKind = ModalKind.UNKNOWN
    title: str | None = None
    open: bool = True
    opened_at: object = Field(default_factory=utc_now)
    closed_at: object | None = None

    @field_validator("modal_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class AppResponsivenessState(OrchestrationModel):
    """
    App responsiveness registry.
    """

    states: dict[str, AppResponsiveness] = Field(default_factory=dict)

    def state_for(self, app_id: str) -> AppResponsiveness:
        return self.states.get(app_id, AppResponsiveness.UNKNOWN)


class DesktopSessionState(OrchestrationModel):
    """
    Current desktop session identity.
    """

    session_id: str = Field(default_factory=lambda: f"desktop_{uuid4().hex}")
    workspace_id: str
    active_project_path: str | None = None
    current_user_activity: str | None = None
    started_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)

    @field_validator("session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class EnvironmentStateSession(OrchestrationModel):
    """
    One state runtime session.
    """

    session_id: str = Field(default_factory=lambda: f"envstate_{uuid4().hex}")
    workspace_id: str
    desktop: DesktopSessionState
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class EnvironmentStateRuntimeEvent(OrchestrationModel):
    """
    State runtime event.
    """

    event_id: str = Field(default_factory=lambda: f"envstate_event_{uuid4().hex}")
    kind: EnvironmentStateEventKind
    reason: EnvironmentStateReason
    session_id: str | None = None
    entity_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class EnvironmentStateOperationResult(OrchestrationModel):
    """
    Result returned by state runtime operations.
    """

    success: bool
    reason: EnvironmentStateReason
    event: EnvironmentStateRuntimeEvent
    session: EnvironmentStateSession | None = None
    message: str

    @field_validator("message")
    @classmethod
    def _required_message(cls, value: str) -> str:
        return _clean_required(value)


class EnvironmentStateRuntimeSnapshot(OrchestrationModel):
    """
    Diagnostics for Phase 8 Step 6.
    """

    name: str
    session_count: int = Field(ge=0)
    display_count: int = Field(ge=0)
    app_count: int = Field(ge=0)
    window_count: int = Field(ge=0)
    visible_window_count: int = Field(ge=0)
    process_count: int = Field(ge=0)
    modal_count: int = Field(ge=0)
    open_modal_count: int = Field(ge=0)
    focused_app_id: str | None = None
    focused_window_id: str | None = None
    runtime_event_count: int = Field(ge=0)
    last_reason: EnvironmentStateReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        return _clean_required(value)


class EnvironmentStateRuntime:
    """
    Phase 8 Step 6 Environment State Runtime.

    Responsibilities:
    - maintain the live desktop world model
    - track displays, apps, windows, cursor, clipboard, processes, modals
    - apply observer EnvironmentEvents into current truth
    - build EnvironmentSnapshot and EnvironmentState contracts
    - optionally route built snapshots into timeline

    Non-responsibilities:
    - no screen capture
    - no OCR
    - no visual detection
    - no physical action execution
    """

    def __init__(
        self,
        *,
        name: str = "environment_state_runtime",
        timeline: EnvironmentTimelineRuntime | None = None,
    ) -> None:
        cleaned = name.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._timeline = timeline
        self._sessions: dict[str, EnvironmentStateSession] = {}
        self._windows: dict[str, LiveWindowRegistry] = {}
        self._displays: dict[str, DisplayTopology] = {}
        self._apps: dict[str, dict[str, AppState]] = {}
        self._cursor: dict[str, CursorTracker] = {}
        self._clipboard: dict[str, ClipboardState] = {}
        self._processes: dict[str, dict[str, ProcessState]] = {}
        self._focused: dict[str, FocusedAppState] = {}
        self._modals: dict[str, dict[str, ModalState]] = {}
        self._responsiveness: dict[str, AppResponsivenessState] = {}
        self._runtime_events: list[EnvironmentStateRuntimeEvent] = []
        self._lock = RLock()
        self._last_reason: EnvironmentStateReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def create_session(
        self,
        *,
        workspace_id: str,
        active_project_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EnvironmentStateSession:
        desktop = DesktopSessionState(
            workspace_id=workspace_id,
            active_project_path=active_project_path,
        )
        session = EnvironmentStateSession(
            workspace_id=workspace_id,
            desktop=desktop,
            metadata=metadata or {},
        )
        event = self._event(
            kind=EnvironmentStateEventKind.SESSION_CREATED,
            reason=EnvironmentStateReason.SESSION_CREATED,
            session_id=session.session_id,
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._windows[session.session_id] = LiveWindowRegistry()
            self._displays[session.session_id] = DisplayTopology()
            self._apps[session.session_id] = {}
            self._cursor[session.session_id] = CursorTracker()
            self._clipboard[session.session_id] = ClipboardState()
            self._processes[session.session_id] = {}
            self._focused[session.session_id] = FocusedAppState()
            self._modals[session.session_id] = {}
            self._responsiveness[session.session_id] = AppResponsivenessState()
            self._runtime_events.append(event)
            self._last_reason = event.reason

        return session

    def register_display(
        self,
        *,
        session_id: str,
        display: DisplayState,
    ) -> EnvironmentStateOperationResult:
        session = self.session_for(session_id)

        if session is None:
            return self._missing_session(session_id)

        with self._lock:
            topology = self._displays[session_id]
            displays = dict(topology.displays)
            displays[display.display_id] = display
            primary = topology.primary_display_id

            if display.primary or primary is None:
                primary = display.display_id

            self._displays[session_id] = DisplayTopology(
                displays=displays,
                primary_display_id=primary,
            )

        return self._mutated(
            session_id=session_id,
            reason=EnvironmentStateReason.DISPLAY_REGISTERED,
            entity_id=display.display_id,
            message="display registered",
        )

    def register_app(
        self,
        *,
        session_id: str,
        app: AppState,
    ) -> EnvironmentStateOperationResult:
        session = self.session_for(session_id)

        if session is None:
            return self._missing_session(session_id)

        with self._lock:
            self._apps[session_id][app.app_id] = app
            self._responsiveness[session_id].states[app.app_id] = (
                AppResponsiveness.RESPONSIVE
                if app.responsive
                else AppResponsiveness.UNRESPONSIVE
            )

        return self._mutated(
            session_id=session_id,
            reason=EnvironmentStateReason.APP_REGISTERED,
            entity_id=app.app_id,
            message="app registered",
        )

    def register_window(
        self,
        *,
        session_id: str,
        window: WindowState,
    ) -> EnvironmentStateOperationResult:
        session = self.session_for(session_id)

        if session is None:
            return self._missing_session(session_id)

        with self._lock:
            registry = self._windows[session_id]
            windows = dict(registry.windows)
            windows[window.window_id] = window
            focused_window_id = registry.focused_window_id

            if window.focused:
                focused_window_id = window.window_id
                self._focused[session_id] = FocusedAppState(
                    app_id=window.app_id,
                    window_id=window.window_id,
                )

            self._windows[session_id] = LiveWindowRegistry(
                windows=windows,
                focused_window_id=focused_window_id,
            )

        return self._mutated(
            session_id=session_id,
            reason=EnvironmentStateReason.WINDOW_REGISTERED,
            entity_id=window.window_id,
            message="window registered",
        )

    def focus_window(
        self,
        *,
        session_id: str,
        window_id: str,
    ) -> EnvironmentStateOperationResult:
        session = self.session_for(session_id)

        if session is None:
            return self._missing_session(session_id)

        with self._lock:
            registry = self._windows[session_id]

            if window_id not in registry.windows:
                return self._missing_session(session_id)

            windows: dict[str, WindowState] = {}

            for existing_id, window in registry.windows.items():
                windows[existing_id] = window.model_copy(
                    update={"focused": existing_id == window_id}
                )

            focused = windows[window_id]
            self._windows[session_id] = LiveWindowRegistry(
                windows=windows,
                focused_window_id=window_id,
            )
            self._focused[session_id] = FocusedAppState(
                app_id=focused.app_id,
                window_id=focused.window_id,
            )

        return self._mutated(
            session_id=session_id,
            reason=EnvironmentStateReason.WINDOW_FOCUSED,
            entity_id=window_id,
            message="window focused",
        )

    def close_window(
        self,
        *,
        session_id: str,
        window_id: str,
    ) -> EnvironmentStateOperationResult:
        session = self.session_for(session_id)

        if session is None:
            return self._missing_session(session_id)

        with self._lock:
            registry = self._windows[session_id]

            if window_id not in registry.windows:
                return self._missing_session(session_id)

            windows = dict(registry.windows)
            closed = windows[window_id].model_copy(
                update={"visible": False, "focused": False}
            )
            windows[window_id] = closed
            focused_window_id = registry.focused_window_id

            if focused_window_id == window_id:
                focused_window_id = None
                self._focused[session_id] = FocusedAppState()

            self._windows[session_id] = LiveWindowRegistry(
                windows=windows,
                focused_window_id=focused_window_id,
            )

        return self._mutated(
            session_id=session_id,
            reason=EnvironmentStateReason.WINDOW_CLOSED,
            entity_id=window_id,
            message="window closed",
        )

    def update_cursor(
        self,
        *,
        session_id: str,
        position: ScreenPoint,
        significant: bool = True,
    ) -> EnvironmentStateOperationResult:
        session = self.session_for(session_id)

        if session is None:
            return self._missing_session(session_id)

        with self._lock:
            previous = self._cursor[session_id]
            self._cursor[session_id] = CursorTracker(
                position=position,
                last_significant_position=(
                    position if significant else previous.last_significant_position
                ),
                moved_significantly=significant,
            )

        return self._mutated(
            session_id=session_id,
            reason=EnvironmentStateReason.CURSOR_UPDATED,
            entity_id=None,
            message="cursor updated",
        )

    def update_clipboard(
        self,
        *,
        session_id: str,
        content_hash: str,
        hint: str | None = None,
        sensitivity: ClipboardSensitivity = ClipboardSensitivity.UNKNOWN,
    ) -> EnvironmentStateOperationResult:
        session = self.session_for(session_id)

        if session is None:
            return self._missing_session(session_id)

        with self._lock:
            self._clipboard[session_id] = ClipboardState(
                content_hash=content_hash,
                hint=hint,
                sensitivity=sensitivity,
            )

        return self._mutated(
            session_id=session_id,
            reason=EnvironmentStateReason.CLIPBOARD_UPDATED,
            entity_id=None,
            message="clipboard updated",
        )

    def update_process(
        self,
        *,
        session_id: str,
        process: ProcessState,
    ) -> EnvironmentStateOperationResult:
        session = self.session_for(session_id)

        if session is None:
            return self._missing_session(session_id)

        with self._lock:
            self._processes[session_id][process.process_id] = process

            if process.app_id:
                if process.status == ProcessStatus.CRASHED:
                    self._responsiveness[session_id].states[process.app_id] = (
                        AppResponsiveness.CRASHED
                    )
                elif process.status == ProcessStatus.UNRESPONSIVE:
                    self._responsiveness[session_id].states[process.app_id] = (
                        AppResponsiveness.UNRESPONSIVE
                    )

        return self._mutated(
            session_id=session_id,
            reason=EnvironmentStateReason.PROCESS_UPDATED,
            entity_id=process.process_id,
            message="process updated",
        )

    def open_modal(
        self,
        *,
        session_id: str,
        modal: ModalState,
    ) -> EnvironmentStateOperationResult:
        session = self.session_for(session_id)

        if session is None:
            return self._missing_session(session_id)

        with self._lock:
            self._modals[session_id][modal.modal_id] = modal

        return self._mutated(
            session_id=session_id,
            reason=EnvironmentStateReason.MODAL_OPENED,
            entity_id=modal.modal_id,
            message="modal opened",
        )

    def close_modal(
        self,
        *,
        session_id: str,
        modal_id: str,
    ) -> EnvironmentStateOperationResult:
        session = self.session_for(session_id)

        if session is None:
            return self._missing_session(session_id)

        with self._lock:
            modal = self._modals[session_id].get(modal_id)

            if modal is None:
                return self._missing_session(session_id)

            self._modals[session_id][modal_id] = modal.model_copy(
                update={"open": False, "closed_at": utc_now()}
            )

        return self._mutated(
            session_id=session_id,
            reason=EnvironmentStateReason.MODAL_CLOSED,
            entity_id=modal_id,
            message="modal closed",
        )

    def update_responsiveness(
        self,
        *,
        session_id: str,
        app_id: str,
        responsiveness: AppResponsiveness,
    ) -> EnvironmentStateOperationResult:
        session = self.session_for(session_id)

        if session is None:
            return self._missing_session(session_id)

        with self._lock:
            states = dict(self._responsiveness[session_id].states)
            states[app_id] = responsiveness
            self._responsiveness[session_id] = AppResponsivenessState(
                states=states
            )

        return self._mutated(
            session_id=session_id,
            reason=EnvironmentStateReason.RESPONSIVENESS_UPDATED,
            entity_id=app_id,
            message="app responsiveness updated",
        )

    def apply_event(
        self,
        *,
        session_id: str,
        event: EnvironmentEvent,
    ) -> EnvironmentStateOperationResult:
        if event.kind == EnvironmentEventKind.WINDOW_FOCUSED and event.window_id:
            return self.focus_window(
                session_id=session_id,
                window_id=event.window_id,
            )

        if event.kind == EnvironmentEventKind.WINDOW_CLOSED and event.window_id:
            return self.close_window(
                session_id=session_id,
                window_id=event.window_id,
            )

        if event.kind == EnvironmentEventKind.CURSOR_MOVED and event.payload:
            x = int(event.payload.get("x", 0))
            y = int(event.payload.get("y", 0))

            return self.update_cursor(
                session_id=session_id,
                position=ScreenPoint(x=x, y=y),
                significant=True,
            )

        if event.kind == EnvironmentEventKind.CLIPBOARD_CHANGED:
            return self.update_clipboard(
                session_id=session_id,
                content_hash=str(event.payload.get("hash", "unknown")),
                hint=event.payload.get("hint"),
                sensitivity=ClipboardSensitivity(
                    event.payload.get("sensitivity", ClipboardSensitivity.UNKNOWN)
                ),
            )

        if event.kind == EnvironmentEventKind.APP_CRASHED and event.app_id:
            return self.update_responsiveness(
                session_id=session_id,
                app_id=event.app_id,
                responsiveness=AppResponsiveness.CRASHED,
            )

        if event.kind == EnvironmentEventKind.MODAL_OPENED:
            return self.open_modal(
                session_id=session_id,
                modal=ModalState(
                    window_id=event.window_id,
                    app_id=event.app_id,
                    kind=ModalKind(event.payload.get("modal_kind", "unknown")),
                    title=event.payload.get("title"),
                ),
            )

        if event.kind == EnvironmentEventKind.MODAL_CLOSED:
            modal_id = event.payload.get("modal_id")

            if isinstance(modal_id, str):
                return self.close_modal(session_id=session_id, modal_id=modal_id)

        return self._mutated(
            session_id=session_id,
            reason=EnvironmentStateReason.EVENT_APPLIED,
            entity_id=event.element_id or event.window_id or event.app_id,
            message="environment event applied",
        )

    def build_snapshot(self, session_id: str) -> EnvironmentSnapshot:
        session = self.session_for(session_id)

        if session is None:
            raise ValueError(f"environment state session not found: {session_id}")

        with self._lock:
            windows = tuple(self._windows[session_id].windows.values())
            apps = tuple(self._apps[session_id].values())
            displays = tuple(self._displays[session_id].displays.values())
            focused = self._focused[session_id]
            cursor = self._cursor[session_id]

        snapshot = EnvironmentSnapshot(
            displays=displays,
            apps=apps,
            windows=windows,
            focused_app_id=focused.app_id,
            focused_window_id=focused.window_id,
            cursor_position=cursor.position,
            trust=_trusted_state(reason="environment state snapshot built"),
        )
        runtime_event = self._event(
            kind=EnvironmentStateEventKind.SNAPSHOT_BUILT,
            reason=EnvironmentStateReason.SNAPSHOT_BUILT,
            session_id=session_id,
        )

        with self._lock:
            self._runtime_events.append(runtime_event)
            self._last_reason = runtime_event.reason

        if self._timeline is not None:
            self._timeline.record_snapshot(
                session_id=session.workspace_id,
                snapshot=snapshot,
            )

        return snapshot

    def build_environment_state(self, session_id: str) -> EnvironmentState:
        session = self.session_for(session_id)

        if session is None:
            raise ValueError(f"environment state session not found: {session_id}")

        snapshot = self.build_snapshot(session_id)
        history = (
            self._timeline.recent_state_history(session_id)
            if self._timeline is not None
            else None
        )
        focused = self.focused_state(session_id)

        state = EnvironmentState(
            current_snapshot=snapshot,
            recent_history=history,
            focused_app_id=focused.app_id if focused else None,
            focused_window_id=focused.window_id if focused else None,
            active_workflow_id=session.workspace_id,
            trust=_trusted_state(reason="environment state built"),
        )
        runtime_event = self._event(
            kind=EnvironmentStateEventKind.SNAPSHOT_BUILT,
            reason=EnvironmentStateReason.ENVIRONMENT_STATE_BUILT,
            session_id=session_id,
        )

        with self._lock:
            self._runtime_events.append(runtime_event)
            self._last_reason = runtime_event.reason

        return state

    def session_for(self, session_id: str) -> EnvironmentStateSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def window_registry(self, session_id: str) -> LiveWindowRegistry | None:
        with self._lock:
            return self._windows.get(session_id)

    def display_topology(self, session_id: str) -> DisplayTopology | None:
        with self._lock:
            return self._displays.get(session_id)

    def cursor_state(self, session_id: str) -> CursorTracker | None:
        with self._lock:
            return self._cursor.get(session_id)

    def clipboard_state(self, session_id: str) -> ClipboardState | None:
        with self._lock:
            return self._clipboard.get(session_id)

    def processes_for(self, session_id: str) -> tuple[ProcessState, ...]:
        with self._lock:
            return tuple(self._processes.get(session_id, {}).values())

    def focused_state(self, session_id: str) -> FocusedAppState | None:
        with self._lock:
            return self._focused.get(session_id)

    def modals_for(self, session_id: str) -> tuple[ModalState, ...]:
        with self._lock:
            return tuple(self._modals.get(session_id, {}).values())

    def responsiveness_for(
        self,
        session_id: str,
    ) -> AppResponsivenessState | None:
        with self._lock:
            return self._responsiveness.get(session_id)

    def runtime_events(self) -> tuple[EnvironmentStateRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._runtime_events)

    def snapshot(self) -> EnvironmentStateRuntimeSnapshot:
        with self._lock:
            focused_app_id: str | None = None
            focused_window_id: str | None = None

            for focused in self._focused.values():
                if focused.app_id or focused.window_id:
                    focused_app_id = focused.app_id
                    focused_window_id = focused.window_id
                    break

            windows = [
                window
                for registry in self._windows.values()
                for window in registry.windows.values()
            ]
            modals = [
                modal
                for registry in self._modals.values()
                for modal in registry.values()
            ]

            return EnvironmentStateRuntimeSnapshot(
                name=self.name,
                session_count=len(self._sessions),
                display_count=sum(
                    len(topology.displays) for topology in self._displays.values()
                ),
                app_count=sum(len(apps) for apps in self._apps.values()),
                window_count=len(windows),
                visible_window_count=sum(1 for window in windows if window.visible),
                process_count=sum(
                    len(processes) for processes in self._processes.values()
                ),
                modal_count=len(modals),
                open_modal_count=sum(1 for modal in modals if modal.open),
                focused_app_id=focused_app_id,
                focused_window_id=focused_window_id,
                runtime_event_count=len(self._runtime_events),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        runtime_event = self._event(
            kind=EnvironmentStateEventKind.RUNTIME_RESET,
            reason=EnvironmentStateReason.RUNTIME_RESET,
        )

        with self._lock:
            self._sessions.clear()
            self._windows.clear()
            self._displays.clear()
            self._apps.clear()
            self._cursor.clear()
            self._clipboard.clear()
            self._processes.clear()
            self._focused.clear()
            self._modals.clear()
            self._responsiveness.clear()
            self._runtime_events.clear()
            self._runtime_events.append(runtime_event)
            self._last_reason = runtime_event.reason

    def _mutated(
        self,
        *,
        session_id: str,
        reason: EnvironmentStateReason,
        entity_id: str | None,
        message: str,
    ) -> EnvironmentStateOperationResult:
        event = self._event(
            kind=EnvironmentStateEventKind.STATE_MUTATED,
            reason=reason,
            session_id=session_id,
            entity_id=entity_id,
        )

        with self._lock:
            self._touch_session(session_id)
            self._runtime_events.append(event)
            self._last_reason = reason

        return EnvironmentStateOperationResult(
            success=True,
            reason=reason,
            event=event,
            session=self.session_for(session_id),
            message=message,
        )

    def _touch_session(self, session_id: str) -> None:
        session = self._sessions[session_id]
        desktop = session.desktop.model_copy(update={"updated_at": utc_now()})
        self._sessions[session_id] = session.model_copy(
            update={"desktop": desktop, "updated_at": utc_now()}
        )

    @staticmethod
    def _event(
        *,
        kind: EnvironmentStateEventKind,
        reason: EnvironmentStateReason,
        session_id: str | None = None,
        entity_id: str | None = None,
    ) -> EnvironmentStateRuntimeEvent:
        return EnvironmentStateRuntimeEvent(
            kind=kind,
            reason=reason,
            session_id=session_id,
            entity_id=entity_id,
        )

    @staticmethod
    def _missing_session(session_id: str) -> EnvironmentStateOperationResult:
        event = EnvironmentStateRuntimeEvent(
            kind=EnvironmentStateEventKind.STATE_MUTATED,
            reason=EnvironmentStateReason.SESSION_NOT_FOUND,
            session_id=session_id,
        )

        return EnvironmentStateOperationResult(
            success=False,
            reason=EnvironmentStateReason.SESSION_NOT_FOUND,
            event=event,
            message="environment state session not found",
        )


def _clean_required(value: str) -> str:
    cleaned = value.strip()

    if not cleaned:
        raise ValueError("field cannot be empty.")

    return cleaned


def _trusted_state(reason: str) -> TrustCalibration:
    return TrustCalibration(
        confidence=0.95,
        stability=0.95,
        ambiguity=0.0,
        source=EnvironmentSource.OS_OBSERVER,
        reason=reason,
    )


def fake_display(
    *,
    display_id: str | None = None,
    primary: bool = True,
) -> DisplayState:
    return DisplayState(
        display_id=display_id or "display_primary",
        kind=DisplayKind.PRIMARY if primary else DisplayKind.SECONDARY,
        bounds=ScreenRegion(x=0, y=0, width=1920, height=1080),
        primary=primary,
        active=True,
    )


def fake_app(
    *,
    app_id: str = "app_vscode",
    name: str = "VS Code",
    kind: AppKind = AppKind.IDE,
    responsive: bool = True,
) -> AppState:
    return AppState(
        app_id=app_id,
        process_id=f"process_{app_id}",
        name=name,
        kind=kind,
        responsive=responsive,
        trusted_identity=True,
    )


def fake_window(
    *,
    window_id: str = "window_main",
    app_id: str = "app_vscode",
    title: str = "JARVIS_OS - VS Code",
    focused: bool = False,
) -> WindowState:
    return WindowState(
        window_id=window_id,
        app_id=app_id,
        title=title,
        bounds=ScreenRegion(x=0, y=0, width=1200, height=800),
        mode=WindowMode.NORMAL,
        focused=focused,
        visible=True,
        responsive=True,
        trust=_trusted_state(reason="fake window state"),
    )