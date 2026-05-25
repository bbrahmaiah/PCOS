from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.tools import (
    ActiveWindowSnapshot,
    AppCapabilityDescriptor,
    AppCapabilityKind,
    AwarenessConfidence,
    BrowserStateSnapshot,
    DesktopAppKind,
    DesktopAwarenessKind,
    DesktopAwarenessRuntime,
    DesktopAwarenessRuntimeConfig,
    DesktopAwarenessSource,
    DesktopPrivacyLevel,
    DesktopStateSnapshot,
    EmptyDesktopAwarenessProvider,
    FocusedAppSnapshot,
    IdeStateSnapshot,
    ScreenContextMetadata,
    TerminalStateSnapshot,
    WindowBounds,
    WorkspaceStateSnapshot,
)


class FakeDesktopAwarenessProvider:
    def capture(self) -> DesktopStateSnapshot:
        return DesktopStateSnapshot(
            active_window=ActiveWindowSnapshot(
                title="JARVIS_OS - Visual Studio Code",
                app_name="Visual Studio Code",
                process_name="Code.exe",
                confidence=AwarenessConfidence.HIGH,
            ),
            focused_app=FocusedAppSnapshot(
                app_name="Visual Studio Code",
                app_kind=DesktopAppKind.IDE,
                process_name="Code.exe",
                confidence=AwarenessConfidence.HIGH,
            ),
            workspace=WorkspaceStateSnapshot(
                workspace_root="E:/JARVIS_OS",
                project_name="JARVIS_OS",
                active_file="jarvis/tools/ide.py",
                git_branch="main",
                dirty_files_count=1,
                open_files=("jarvis/tools/ide.py",),
                confidence=AwarenessConfidence.HIGH,
            ),
            terminal=TerminalStateSnapshot(
                working_directory="E:/JARVIS_OS",
                shell_name="PowerShell",
                last_command="pytest",
                command_running=False,
                exit_code=0,
                confidence=AwarenessConfidence.MEDIUM,
            ),
            ide=IdeStateSnapshot(
                ide_name="Visual Studio Code",
                active_file="jarvis/tools/ide.py",
                active_symbol="IdeRuntime",
                diagnostics_count=0,
                tests_running=False,
                confidence=AwarenessConfidence.HIGH,
            ),
            browser=BrowserStateSnapshot(
                browser_name="Chrome",
                active_url="https://docs.python.org",
                active_title="Python Docs",
                tab_count=3,
                confidence=AwarenessConfidence.MEDIUM,
            ),
            screen=ScreenContextMetadata(
                monitor_count=1,
                active_monitor_index=0,
                user_present=True,
                confidence=AwarenessConfidence.MEDIUM,
            ),
            app_capabilities=(
                AppCapabilityDescriptor(
                    app_name="Visual Studio Code",
                    app_kind=DesktopAppKind.IDE,
                    capabilities=(
                        AppCapabilityKind.VIEW_ONLY,
                        AppCapabilityKind.READ_STATE,
                        AppCapabilityKind.EDITOR_NAVIGATION,
                    ),
                ),
            ),
            source=DesktopAwarenessSource.FAKE_PROVIDER,
            confidence=AwarenessConfidence.HIGH,
        )


def test_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        DesktopAwarenessRuntimeConfig(name=" ").validate()

    with pytest.raises(ValueError):
        DesktopAwarenessRuntimeConfig(allow_control=True).validate()


def test_window_bounds_accepts_dimensions() -> None:
    bounds = WindowBounds(x=10, y=20, width=800, height=600)

    assert bounds.width == 800
    assert bounds.height == 600


def test_active_window_requires_title() -> None:
    with pytest.raises(ValidationError):
        ActiveWindowSnapshot(title=" ", app_name="Code")


def test_focused_app_requires_name() -> None:
    with pytest.raises(ValidationError):
        FocusedAppSnapshot(app_name=" ")


def test_workspace_open_files_are_cleaned() -> None:
    workspace = WorkspaceStateSnapshot(
        open_files=(" a.py ", "", " b.py "),
    )

    assert workspace.open_files == ("a.py", "b.py")


def test_app_descriptor_never_allows_control() -> None:
    with pytest.raises(ValidationError):
        AppCapabilityDescriptor(
            app_name="Code",
            app_kind=DesktopAppKind.IDE,
            capabilities=(AppCapabilityKind.VIEW_ONLY,),
            control_allowed=True,
        )


def test_desktop_state_properties() -> None:
    state = DesktopStateSnapshot(
        active_window=ActiveWindowSnapshot(
            title="Terminal",
            app_name="Windows Terminal",
        ),
        workspace=WorkspaceStateSnapshot(project_name="JARVIS_OS"),
    )

    assert state.has_focus_context is True
    assert state.has_developer_context is True


def test_empty_provider_returns_unknown_snapshot() -> None:
    snapshot = EmptyDesktopAwarenessProvider().capture()

    assert snapshot.source == DesktopAwarenessSource.UNKNOWN
    assert snapshot.confidence == AwarenessConfidence.UNKNOWN
    assert snapshot.screen is not None


def test_runtime_capture_from_fake_provider() -> None:
    runtime = DesktopAwarenessRuntime(provider=FakeDesktopAwarenessProvider())

    snapshot = runtime.capture()

    assert snapshot.active_window is not None
    assert snapshot.active_window.title == "JARVIS_OS - Visual Studio Code"
    assert snapshot.focused_app is not None
    assert snapshot.focused_app.app_kind == DesktopAppKind.IDE
    assert snapshot.workspace is not None
    assert snapshot.workspace.project_name == "JARVIS_OS"
    assert snapshot.terminal is not None
    assert snapshot.terminal.last_command == "pytest"
    assert snapshot.ide is not None
    assert snapshot.ide.active_symbol == "IdeRuntime"
    assert snapshot.browser is not None
    assert snapshot.browser.browser_name == "Chrome"
    assert snapshot.has_developer_context is True


def test_runtime_snapshot_and_reset() -> None:
    runtime = DesktopAwarenessRuntime(provider=FakeDesktopAwarenessProvider())

    runtime.capture()
    snapshot = runtime.snapshot()

    assert snapshot.capture_count == 1
    assert snapshot.last_confidence == AwarenessConfidence.HIGH
    assert snapshot.last_source == DesktopAwarenessSource.FAKE_PROVIDER

    runtime.reset()
    reset_snapshot = runtime.snapshot()

    assert reset_snapshot.capture_count == 0
    assert reset_snapshot.last_confidence is None
    assert reset_snapshot.last_source is None


def test_runtime_rejects_provider_control_descriptor() -> None:
    class UnsafeProvider:
        def capture(self) -> DesktopStateSnapshot:
            descriptor = AppCapabilityDescriptor.model_construct(
                app_name="Unsafe App",
                app_kind=DesktopAppKind.SYSTEM,
                capabilities=(AppCapabilityKind.VIEW_ONLY,),
                control_allowed=True,
                reason="unsafe test",
            )

            return DesktopStateSnapshot(app_capabilities=(descriptor,))

    runtime = DesktopAwarenessRuntime(provider=UnsafeProvider())

    with pytest.raises(ValueError):
        runtime.capture()


def test_privacy_defaults_are_conservative() -> None:
    active_window = ActiveWindowSnapshot(title="Email", app_name="Mail")
    terminal = TerminalStateSnapshot(last_command="pytest")
    workspace = WorkspaceStateSnapshot(project_name="JARVIS_OS")

    assert active_window.privacy_level == DesktopPrivacyLevel.USER_PRIVATE
    assert terminal.privacy_level == DesktopPrivacyLevel.USER_PRIVATE
    assert workspace.privacy_level == DesktopPrivacyLevel.WORKSPACE


def test_enum_values_are_stable() -> None:
    assert DesktopAwarenessKind.DESKTOP_STATE.value == "desktop_state"
    assert DesktopAppKind.IDE.value == "ide"
    assert DesktopAwarenessSource.FAKE_PROVIDER.value == "fake_provider"
    assert AppCapabilityKind.VIEW_ONLY.value == "view_only"