from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from typing import Protocol

from pydantic import Field, field_validator, model_validator

from jarvis.tools.ids import new_action_result_id, utc_now
from jarvis.tools.models import ToolModel


class DesktopAwarenessKind(StrEnum):
    """
    Supported awareness categories.

    These are observation contracts only. They must not trigger desktop control.
    """

    ACTIVE_WINDOW = "active_window"
    FOCUSED_APP = "focused_app"
    WORKSPACE_STATE = "workspace_state"
    TERMINAL_STATE = "terminal_state"
    IDE_STATE = "ide_state"
    BROWSER_STATE = "browser_state"
    SCREEN_CONTEXT = "screen_context"
    DESKTOP_STATE = "desktop_state"


class DesktopAppKind(StrEnum):
    """
    High-level application kind.

    This avoids hard-coding behavior to process names too early.
    """

    UNKNOWN = "unknown"
    TERMINAL = "terminal"
    IDE = "ide"
    BROWSER = "browser"
    FILE_MANAGER = "file_manager"
    DOCUMENT_EDITOR = "document_editor"
    MEDIA_PLAYER = "media_player"
    COMMUNICATION = "communication"
    SYSTEM = "system"


class DesktopAwarenessSource(StrEnum):
    """
    Source of awareness data.
    """

    UNKNOWN = "unknown"
    FAKE_PROVIDER = "fake_provider"
    MANUAL_CONTEXT = "manual_context"
    OS_SNAPSHOT = "os_snapshot"
    APP_ADAPTER = "app_adapter"


class DesktopPrivacyLevel(StrEnum):
    """
    Privacy level for captured context.
    """

    PUBLIC = "public"
    WORKSPACE = "workspace"
    USER_PRIVATE = "user_private"
    SENSITIVE = "sensitive"


class AppCapabilityKind(StrEnum):
    """
    Safe capability descriptors.

    These describe what an app can be used for. They do not grant permission to
    control the app.
    """

    VIEW_ONLY = "view_only"
    READ_STATE = "read_state"
    OPEN_RESOURCE = "open_resource"
    SEARCH_CONTEXT = "search_context"
    EDITOR_NAVIGATION = "editor_navigation"
    TERMINAL_STATUS = "terminal_status"
    BROWSER_CONTEXT = "browser_context"


class AwarenessConfidence(StrEnum):
    """
    Confidence band for awareness inference.
    """

    UNKNOWN = "unknown"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class WindowBounds(ToolModel):
    """
    Screen-space window bounds.

    Coordinates are metadata only. They must not be used for mouse automation
    in this phase.
    """

    x: int = 0
    y: int = 0
    width: int = Field(default=0, ge=0)
    height: int = Field(default=0, ge=0)


class ActiveWindowSnapshot(ToolModel):
    """
    Active/focused window snapshot.
    """

    title: str = "Unknown"
    app_name: str = "Unknown"
    process_name: str | None = None
    bounds: WindowBounds | None = None
    privacy_level: DesktopPrivacyLevel = DesktopPrivacyLevel.USER_PRIVATE
    confidence: AwarenessConfidence = AwarenessConfidence.UNKNOWN
    captured_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("title", "app_name")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("process_name")
    @classmethod
    def _clean_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None


class FocusedAppSnapshot(ToolModel):
    """
    Focused application snapshot.
    """

    app_name: str = "Unknown"
    app_kind: DesktopAppKind = DesktopAppKind.UNKNOWN
    process_name: str | None = None
    window_title: str | None = None
    privacy_level: DesktopPrivacyLevel = DesktopPrivacyLevel.USER_PRIVATE
    confidence: AwarenessConfidence = AwarenessConfidence.UNKNOWN
    captured_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("app_name")
    @classmethod
    def _required_app_name(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("app_name cannot be empty.")

        return cleaned

    @field_validator("process_name", "window_title")
    @classmethod
    def _clean_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None


class WorkspaceStateSnapshot(ToolModel):
    """
    Workspace/project state snapshot.
    """

    workspace_root: str | None = None
    project_name: str | None = None
    active_file: str | None = None
    git_branch: str | None = None
    dirty_files_count: int = Field(default=0, ge=0)
    open_files: tuple[str, ...] = ()
    privacy_level: DesktopPrivacyLevel = DesktopPrivacyLevel.WORKSPACE
    confidence: AwarenessConfidence = AwarenessConfidence.UNKNOWN
    captured_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator(
        "workspace_root",
        "project_name",
        "active_file",
        "git_branch",
    )
    @classmethod
    def _clean_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None

    @field_validator("open_files")
    @classmethod
    def _clean_open_files(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(item.strip() for item in value if item.strip())


class TerminalStateSnapshot(ToolModel):
    """
    Terminal awareness snapshot.

    This is state awareness only. It must not execute commands.
    """

    working_directory: str | None = None
    shell_name: str | None = None
    last_command: str | None = None
    command_running: bool = False
    exit_code: int | None = None
    visible_output_tail: str | None = None
    privacy_level: DesktopPrivacyLevel = DesktopPrivacyLevel.USER_PRIVATE
    confidence: AwarenessConfidence = AwarenessConfidence.UNKNOWN
    captured_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator(
        "working_directory",
        "shell_name",
        "last_command",
        "visible_output_tail",
    )
    @classmethod
    def _clean_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None


class IdeStateSnapshot(ToolModel):
    """
    IDE/editor awareness snapshot.
    """

    ide_name: str | None = None
    active_file: str | None = None
    active_symbol: str | None = None
    diagnostics_count: int = Field(default=0, ge=0)
    tests_running: bool = False
    debug_session_active: bool = False
    privacy_level: DesktopPrivacyLevel = DesktopPrivacyLevel.WORKSPACE
    confidence: AwarenessConfidence = AwarenessConfidence.UNKNOWN
    captured_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("ide_name", "active_file", "active_symbol")
    @classmethod
    def _clean_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None


class BrowserStateSnapshot(ToolModel):
    """
    Browser awareness snapshot.
    """

    browser_name: str | None = None
    active_url: str | None = None
    active_title: str | None = None
    tab_count: int = Field(default=0, ge=0)
    download_active: bool = False
    form_detected: bool = False
    privacy_level: DesktopPrivacyLevel = DesktopPrivacyLevel.USER_PRIVATE
    confidence: AwarenessConfidence = AwarenessConfidence.UNKNOWN
    captured_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("browser_name", "active_url", "active_title")
    @classmethod
    def _clean_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None


class ScreenContextMetadata(ToolModel):
    """
    Screen-level metadata.

    This intentionally excludes OCR/image capture. Later vision runtime can add
    richer visual contracts behind policy.
    """

    monitor_count: int = Field(default=1, ge=0)
    active_monitor_index: int = Field(default=0, ge=0)
    screen_locked: bool = False
    user_present: bool = True
    privacy_level: DesktopPrivacyLevel = DesktopPrivacyLevel.USER_PRIVATE
    confidence: AwarenessConfidence = AwarenessConfidence.UNKNOWN
    captured_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)


class AppCapabilityDescriptor(ToolModel):
    """
    Safe descriptor of what an observed app may support.

    This is descriptive only. It is not permission to execute actions.
    """

    app_name: str
    app_kind: DesktopAppKind
    capabilities: tuple[AppCapabilityKind, ...]
    control_allowed: bool = False
    reason: str = "awareness only"

    @field_validator("app_name", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @model_validator(mode="after")
    def _deny_control_in_awareness_phase(self) -> AppCapabilityDescriptor:
        if self.control_allowed:
            raise ValueError("desktop awareness descriptors cannot allow control.")

        return self


class DesktopStateSnapshot(ToolModel):
    """
    Full desktop state snapshot.

    This is the Step 9 top-level awareness contract.
    """

    snapshot_id: str = Field(default_factory=new_action_result_id)
    active_window: ActiveWindowSnapshot | None = None
    focused_app: FocusedAppSnapshot | None = None
    workspace: WorkspaceStateSnapshot | None = None
    terminal: TerminalStateSnapshot | None = None
    ide: IdeStateSnapshot | None = None
    browser: BrowserStateSnapshot | None = None
    screen: ScreenContextMetadata | None = None
    app_capabilities: tuple[AppCapabilityDescriptor, ...] = ()
    source: DesktopAwarenessSource = DesktopAwarenessSource.UNKNOWN
    privacy_level: DesktopPrivacyLevel = DesktopPrivacyLevel.USER_PRIVATE
    confidence: AwarenessConfidence = AwarenessConfidence.UNKNOWN
    captured_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("snapshot_id")
    @classmethod
    def _required_snapshot_id(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("snapshot_id cannot be empty.")

        return cleaned

    @property
    def has_focus_context(self) -> bool:
        return self.active_window is not None or self.focused_app is not None

    @property
    def has_developer_context(self) -> bool:
        return (
            self.workspace is not None
            or self.terminal is not None
            or self.ide is not None
        )


@dataclass(frozen=True, slots=True)
class DesktopAwarenessRuntimeConfig:
    """
    Configuration for DesktopAwarenessRuntime.
    """

    name: str = "desktop_awareness_runtime"
    allow_control: bool = False

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.allow_control:
            raise ValueError("Step 9 is awareness only. Control is not allowed.")


@dataclass(frozen=True, slots=True)
class DesktopAwarenessRuntimeSnapshot:
    """
    Observable diagnostics for DesktopAwarenessRuntime.
    """

    name: str
    capture_count: int
    last_confidence: AwarenessConfidence | None
    last_source: DesktopAwarenessSource | None
    last_error: str | None


class DesktopAwarenessProvider(Protocol):
    """
    Provider protocol for desktop awareness.

    Real OS adapters can be added later. This protocol prevents the runtime from
    owning platform-specific code.
    """

    def capture(self) -> DesktopStateSnapshot:
        ...


class EmptyDesktopAwarenessProvider:
    """
    Safe default provider.

    It returns an explicit unknown snapshot instead of attempting unsafe or
    platform-specific desktop introspection.
    """

    def capture(self) -> DesktopStateSnapshot:
        return DesktopStateSnapshot(
            source=DesktopAwarenessSource.UNKNOWN,
            confidence=AwarenessConfidence.UNKNOWN,
            screen=ScreenContextMetadata(
                confidence=AwarenessConfidence.UNKNOWN,
            ),
            metadata={
                "provider": "empty_desktop_awareness_provider",
                "control_allowed": False,
            },
        )


class DesktopAwarenessRuntime:
    """
    Desktop/app awareness runtime.

    Responsibilities:
    - provide a single safe desktop state contract
    - preserve active window / focused app / workspace / terminal / IDE /
      browser context
    - describe app capabilities without granting control
    - expose diagnostics for observability

    Non-responsibilities:
    - no mouse automation
    - no keyboard automation
    - no window control
    - no screenshots
    - no hidden app interaction
    """

    def __init__(
        self,
        *,
        config: DesktopAwarenessRuntimeConfig | None = None,
        provider: DesktopAwarenessProvider | None = None,
    ) -> None:
        self._config = config or DesktopAwarenessRuntimeConfig()
        self._config.validate()

        self._provider = provider or EmptyDesktopAwarenessProvider()
        self._lock = RLock()

        self._capture_count = 0
        self._last_confidence: AwarenessConfidence | None = None
        self._last_source: DesktopAwarenessSource | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def capture(self) -> DesktopStateSnapshot:
        """
        Capture one desktop awareness snapshot.

        This method observes only. It never controls the desktop.
        """

        with self._lock:
            self._capture_count += 1
            self._last_error = None

        try:
            snapshot = self._provider.capture()
            self._assert_awareness_only(snapshot)

            with self._lock:
                self._last_confidence = snapshot.confidence
                self._last_source = snapshot.source

            return snapshot

        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"

            raise

    def snapshot(self) -> DesktopAwarenessRuntimeSnapshot:
        """
        Return runtime diagnostics.
        """

        with self._lock:
            return DesktopAwarenessRuntimeSnapshot(
                name=self.name,
                capture_count=self._capture_count,
                last_confidence=self._last_confidence,
                last_source=self._last_source,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Reset diagnostics only.
        """

        with self._lock:
            self._capture_count = 0
            self._last_confidence = None
            self._last_source = None
            self._last_error = None

    @staticmethod
    def _assert_awareness_only(snapshot: DesktopStateSnapshot) -> None:
        for descriptor in snapshot.app_capabilities:
            if descriptor.control_allowed:
                raise ValueError("desktop awareness cannot expose control permission.")