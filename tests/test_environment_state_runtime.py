from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    AppResponsiveness,
    AppResponsivenessState,
    ClipboardSensitivity,
    EnvironmentEvent,
    EnvironmentEventKind,
    EnvironmentSource,
    EnvironmentStateReason,
    EnvironmentStateRuntime,
    EnvironmentTimelineRuntime,
    ModalKind,
    ModalState,
    ProcessState,
    ProcessStatus,
    ScreenPoint,
    TrustCalibration,
    fake_app,
    fake_display,
    fake_window,
)


def test_runtime_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        EnvironmentStateRuntime(name=" ")


def test_session_requires_workspace_id() -> None:
    runtime = EnvironmentStateRuntime()

    with pytest.raises(ValidationError):
        runtime.create_session(workspace_id=" ")


def test_create_session_initializes_registries() -> None:
    runtime = EnvironmentStateRuntime()
    session = runtime.create_session(workspace_id="workspace")

    assert session.workspace_id == "workspace"
    assert runtime.window_registry(session.session_id) is not None
    assert runtime.display_topology(session.session_id) is not None
    assert runtime.snapshot().session_count == 1


def test_register_display() -> None:
    runtime = EnvironmentStateRuntime()
    session = runtime.create_session(workspace_id="workspace")
    display = fake_display()

    result = runtime.register_display(
        session_id=session.session_id,
        display=display,
    )
    topology = runtime.display_topology(session.session_id)

    assert result.success is True
    assert result.reason == EnvironmentStateReason.DISPLAY_REGISTERED
    assert topology is not None
    assert topology.primary_display_id == display.display_id


def test_register_app_updates_responsiveness() -> None:
    runtime = EnvironmentStateRuntime()
    session = runtime.create_session(workspace_id="workspace")
    app = fake_app()

    result = runtime.register_app(session_id=session.session_id, app=app)
    responsiveness = runtime.responsiveness_for(session.session_id)

    assert result.success is True
    assert responsiveness is not None
    assert responsiveness.state_for(app.app_id) == AppResponsiveness.RESPONSIVE


def test_register_window_and_focus() -> None:
    runtime = EnvironmentStateRuntime()
    session = runtime.create_session(workspace_id="workspace")
    app = fake_app()
    window = fake_window(app_id=app.app_id)

    runtime.register_app(session_id=session.session_id, app=app)
    runtime.register_window(session_id=session.session_id, window=window)
    result = runtime.focus_window(
        session_id=session.session_id,
        window_id=window.window_id,
    )
    focused = runtime.focused_state(session.session_id)

    assert result.success is True
    assert focused is not None
    assert focused.app_id == app.app_id
    assert focused.window_id == window.window_id


def test_close_focused_window_clears_focus() -> None:
    runtime = EnvironmentStateRuntime()
    session = runtime.create_session(workspace_id="workspace")
    window = fake_window(focused=True)

    runtime.register_window(session_id=session.session_id, window=window)
    runtime.close_window(
        session_id=session.session_id,
        window_id=window.window_id,
    )
    focused = runtime.focused_state(session.session_id)
    registry = runtime.window_registry(session.session_id)

    assert focused is not None
    assert focused.window_id is None
    assert registry is not None
    assert registry.windows[window.window_id].visible is False


def test_update_cursor() -> None:
    runtime = EnvironmentStateRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.update_cursor(
        session_id=session.session_id,
        position=ScreenPoint(x=10, y=20),
    )
    cursor = runtime.cursor_state(session.session_id)

    assert result.success is True
    assert cursor is not None
    assert cursor.position == ScreenPoint(x=10, y=20)
    assert cursor.moved_significantly is True


def test_update_clipboard_stores_hash_not_raw_content() -> None:
    runtime = EnvironmentStateRuntime()
    session = runtime.create_session(workspace_id="workspace")

    runtime.update_clipboard(
        session_id=session.session_id,
        content_hash="abc123",
        hint="text/plain",
        sensitivity=ClipboardSensitivity.LOW,
    )
    clipboard = runtime.clipboard_state(session.session_id)

    assert clipboard is not None
    assert clipboard.content_hash == "abc123"
    assert clipboard.hint == "text/plain"


def test_update_process_crash_marks_app_crashed() -> None:
    runtime = EnvironmentStateRuntime()
    session = runtime.create_session(workspace_id="workspace")
    app = fake_app()

    runtime.register_app(session_id=session.session_id, app=app)
    runtime.update_process(
        session_id=session.session_id,
        process=ProcessState(
            process_id="process",
            app_id=app.app_id,
            name="VS Code",
            status=ProcessStatus.CRASHED,
        ),
    )
    responsiveness = runtime.responsiveness_for(session.session_id)

    assert responsiveness is not None
    assert responsiveness.state_for(app.app_id) == AppResponsiveness.CRASHED


def test_open_and_close_modal() -> None:
    runtime = EnvironmentStateRuntime()
    session = runtime.create_session(workspace_id="workspace")
    modal = ModalState(kind=ModalKind.ERROR, title="Error")

    runtime.open_modal(session_id=session.session_id, modal=modal)
    runtime.close_modal(session_id=session.session_id, modal_id=modal.modal_id)
    modals = runtime.modals_for(session.session_id)

    assert len(modals) == 1
    assert modals[0].open is False


def test_update_responsiveness() -> None:
    runtime = EnvironmentStateRuntime()
    session = runtime.create_session(workspace_id="workspace")

    runtime.update_responsiveness(
        session_id=session.session_id,
        app_id="app",
        responsiveness=AppResponsiveness.UNRESPONSIVE,
    )
    state = runtime.responsiveness_for(session.session_id)

    assert isinstance(state, AppResponsivenessState)
    assert state.state_for("app") == AppResponsiveness.UNRESPONSIVE


def test_apply_window_focus_event() -> None:
    runtime = EnvironmentStateRuntime()
    session = runtime.create_session(workspace_id="workspace")
    window = fake_window()

    runtime.register_window(session_id=session.session_id, window=window)
    result = runtime.apply_event(
        session_id=session.session_id,
        event=_event(
            EnvironmentEventKind.WINDOW_FOCUSED,
            window_id=window.window_id,
            app_id=window.app_id,
        ),
    )

    assert result.success is True
    assert result.reason == EnvironmentStateReason.WINDOW_FOCUSED


def test_apply_cursor_event() -> None:
    runtime = EnvironmentStateRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.apply_event(
        session_id=session.session_id,
        event=_event(
            EnvironmentEventKind.CURSOR_MOVED,
            payload={"x": 42, "y": 99},
        ),
    )
    cursor = runtime.cursor_state(session.session_id)

    assert result.success is True
    assert cursor is not None
    assert cursor.position == ScreenPoint(x=42, y=99)


def test_apply_clipboard_event() -> None:
    runtime = EnvironmentStateRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.apply_event(
        session_id=session.session_id,
        event=_event(
            EnvironmentEventKind.CLIPBOARD_CHANGED,
            payload={
                "hash": "hash123",
                "hint": "text",
                "sensitivity": ClipboardSensitivity.MEDIUM.value,
            },
        ),
    )
    clipboard = runtime.clipboard_state(session.session_id)

    assert result.success is True
    assert clipboard is not None
    assert clipboard.sensitivity == ClipboardSensitivity.MEDIUM


def test_apply_app_crashed_event() -> None:
    runtime = EnvironmentStateRuntime()
    session = runtime.create_session(workspace_id="workspace")
    app = fake_app()

    runtime.register_app(session_id=session.session_id, app=app)
    result = runtime.apply_event(
        session_id=session.session_id,
        event=_event(EnvironmentEventKind.APP_CRASHED, app_id=app.app_id),
    )
    responsiveness = runtime.responsiveness_for(session.session_id)

    assert result.success is True
    assert responsiveness is not None
    assert responsiveness.state_for(app.app_id) == AppResponsiveness.CRASHED


def test_build_snapshot() -> None:
    runtime = EnvironmentStateRuntime()
    session = runtime.create_session(workspace_id="workspace")
    app = fake_app()
    window = fake_window(app_id=app.app_id, focused=True)

    runtime.register_display(session_id=session.session_id, display=fake_display())
    runtime.register_app(session_id=session.session_id, app=app)
    runtime.register_window(session_id=session.session_id, window=window)
    snapshot = runtime.build_snapshot(session.session_id)

    assert len(snapshot.displays) == 1
    assert len(snapshot.apps) == 1
    assert len(snapshot.windows) == 1
    assert snapshot.focused_app_id == app.app_id


def test_build_environment_state() -> None:
    runtime = EnvironmentStateRuntime()
    session = runtime.create_session(workspace_id="workspace")
    app = fake_app()
    window = fake_window(app_id=app.app_id, focused=True)

    runtime.register_app(session_id=session.session_id, app=app)
    runtime.register_window(session_id=session.session_id, window=window)
    state = runtime.build_environment_state(session.session_id)

    assert state.focused_app_id == app.app_id
    assert state.focused_window_id == window.window_id
    assert state.active_workflow_id == "workspace"


def test_build_snapshot_routes_to_timeline_when_configured() -> None:
    timeline = EnvironmentTimelineRuntime()
    timeline_session = timeline.create_session(workflow_id="workflow")
    runtime = EnvironmentStateRuntime(timeline=timeline)

    session = runtime.create_session(workspace_id=timeline_session.session_id)
    runtime.build_snapshot(session.session_id)

    assert timeline.snapshot().snapshot_count == 1


def test_missing_session_operation_fails() -> None:
    runtime = EnvironmentStateRuntime()

    result = runtime.register_app(session_id="missing", app=fake_app())

    assert result.success is False
    assert result.reason == EnvironmentStateReason.SESSION_NOT_FOUND


def test_snapshot_tracks_counts() -> None:
    runtime = EnvironmentStateRuntime()
    session = runtime.create_session(workspace_id="workspace")

    runtime.register_display(session_id=session.session_id, display=fake_display())
    runtime.register_app(session_id=session.session_id, app=fake_app())
    runtime.register_window(session_id=session.session_id, window=fake_window())
    runtime.open_modal(
        session_id=session.session_id,
        modal=ModalState(kind=ModalKind.ERROR),
    )
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.display_count == 1
    assert snapshot.app_count == 1
    assert snapshot.window_count == 1
    assert snapshot.open_modal_count == 1


def test_reset_clears_runtime() -> None:
    runtime = EnvironmentStateRuntime()
    runtime.create_session(workspace_id="workspace")

    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.runtime_event_count == 1
    assert snapshot.last_reason == EnvironmentStateReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert ClipboardSensitivity.SECRET.value == "secret"
    assert ProcessStatus.CRASHED.value == "crashed"
    assert ModalKind.ERROR.value == "error"
    assert AppResponsiveness.UNRESPONSIVE.value == "unresponsive"


def _event(
    kind: EnvironmentEventKind,
    *,
    app_id: str | None = None,
    window_id: str | None = None,
    payload: dict[str, object] | None = None,
) -> EnvironmentEvent:
    return EnvironmentEvent(
        kind=kind,
        source=EnvironmentSource.OS_OBSERVER,
        app_id=app_id,
        window_id=window_id,
        trust=TrustCalibration(
            confidence=0.95,
            stability=0.95,
            ambiguity=0.0,
            source=EnvironmentSource.OS_OBSERVER,
            reason="test event",
        ),
        payload=payload or {},
    )