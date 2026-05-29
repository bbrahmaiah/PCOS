from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    EnvironmentEventKind,
    EnvironmentObserverBackbone,
    EnvironmentObserverDescriptor,
    EnvironmentObserverHealth,
    EnvironmentObserverKind,
    EnvironmentObserverStatus,
    EnvironmentSource,
    EnvironmentTimelineRuntime,
    ObserverBackboneReason,
    ObserverEventPriority,
    ObserverSignal,
    default_environment_observers,
)


def test_backbone_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        EnvironmentObserverBackbone(name=" ")


def test_backbone_rejects_invalid_dedupe_window() -> None:
    with pytest.raises(ValueError):
        EnvironmentObserverBackbone(dedupe_window=0)


def test_observer_descriptor_requires_watched_events() -> None:
    with pytest.raises(ValidationError):
        EnvironmentObserverDescriptor(
            kind=EnvironmentObserverKind.ACTIVE_WINDOW,
            name="ActiveWindowObserver",
            watched_events=(),
        )


def test_default_observers_include_required_backbone() -> None:
    observers = default_environment_observers()
    kinds = {observer.kind for observer in observers}

    assert len(observers) == 9
    assert EnvironmentObserverKind.ACTIVE_WINDOW in kinds
    assert EnvironmentObserverKind.DISPLAY in kinds
    assert EnvironmentObserverKind.CURSOR in kinds
    assert EnvironmentObserverKind.CLIPBOARD in kinds
    assert EnvironmentObserverKind.FILESYSTEM in kinds
    assert EnvironmentObserverKind.PROCESS in kinds
    assert EnvironmentObserverKind.BROWSER in kinds
    assert EnvironmentObserverKind.APP_LIFECYCLE in kinds
    assert EnvironmentObserverKind.MODAL in kinds


def test_register_defaults_makes_backbone_ready() -> None:
    backbone = EnvironmentObserverBackbone()

    results = backbone.register_defaults()

    assert len(results) == 9
    assert all(result.success for result in results)
    assert backbone.is_ready() is True


def test_register_duplicate_rejected() -> None:
    backbone = EnvironmentObserverBackbone()
    observer = default_environment_observers()[0]

    backbone.register_observer(observer)
    result = backbone.register_observer(observer)

    assert result.success is False
    assert result.reason == ObserverBackboneReason.OBSERVER_DUPLICATE_REJECTED


def test_start_and_stop_observer() -> None:
    backbone = EnvironmentObserverBackbone()
    backbone.register_defaults()

    started = backbone.start_observer(EnvironmentObserverKind.ACTIVE_WINDOW)
    stopped = backbone.stop_observer(EnvironmentObserverKind.ACTIVE_WINDOW)
    observer = backbone.observer_for(EnvironmentObserverKind.ACTIVE_WINDOW)

    assert started.success is True
    assert stopped.success is True
    assert observer is not None
    assert observer.status == EnvironmentObserverStatus.STOPPED


def test_start_missing_observer_rejected() -> None:
    backbone = EnvironmentObserverBackbone()

    result = backbone.start_observer(EnvironmentObserverKind.ACTIVE_WINDOW)

    assert result.success is False
    assert result.reason == ObserverBackboneReason.OBSERVER_NOT_FOUND


def test_fail_observer_marks_failed() -> None:
    backbone = EnvironmentObserverBackbone()
    backbone.register_defaults()

    result = backbone.fail_observer(
        EnvironmentObserverKind.PROCESS,
        reason_text="process hook failed",
    )
    observer = backbone.observer_for(EnvironmentObserverKind.PROCESS)

    assert result.success is True
    assert observer is not None
    assert observer.health == EnvironmentObserverHealth.FAILED


def test_emit_signal_creates_environment_event() -> None:
    backbone = EnvironmentObserverBackbone()
    backbone.register_defaults()
    backbone.start_observer(EnvironmentObserverKind.ACTIVE_WINDOW)

    result = backbone.emit_signal(
        ObserverSignal(
            observer_kind=EnvironmentObserverKind.ACTIVE_WINDOW,
            event_kind=EnvironmentEventKind.WINDOW_FOCUSED,
            source=EnvironmentSource.OS_OBSERVER,
            window_id="window-1",
            app_id="app-1",
        )
    )

    assert result.success is True
    assert result.environment_event is not None
    assert result.environment_event.kind == EnvironmentEventKind.WINDOW_FOCUSED
    assert result.environment_event.window_id == "window-1"


def test_emit_signal_rejects_missing_observer() -> None:
    backbone = EnvironmentObserverBackbone()

    result = backbone.emit_signal(
        ObserverSignal(
            observer_kind=EnvironmentObserverKind.ACTIVE_WINDOW,
            event_kind=EnvironmentEventKind.WINDOW_FOCUSED,
        )
    )

    assert result.success is False
    assert result.reason == ObserverBackboneReason.OBSERVER_NOT_FOUND


def test_emit_signal_rejects_unwatched_event_kind() -> None:
    backbone = EnvironmentObserverBackbone()
    backbone.register_defaults()

    result = backbone.emit_signal(
        ObserverSignal(
            observer_kind=EnvironmentObserverKind.CURSOR,
            event_kind=EnvironmentEventKind.APP_CRASHED,
        )
    )

    assert result.success is False
    assert result.reason == ObserverBackboneReason.OBSERVER_NOT_FOUND


def test_duplicate_signal_is_deduped() -> None:
    backbone = EnvironmentObserverBackbone()
    backbone.register_defaults()
    signal = ObserverSignal(
        observer_kind=EnvironmentObserverKind.ACTIVE_WINDOW,
        event_kind=EnvironmentEventKind.WINDOW_FOCUSED,
        window_id="window-1",
    )

    first = backbone.emit_signal(signal)
    second = backbone.emit_signal(signal)

    assert first.reason == ObserverBackboneReason.EVENT_ACCEPTED
    assert second.reason == ObserverBackboneReason.EVENT_DEDUPED
    assert backbone.snapshot().deduped_event_count == 1


def test_signal_routes_to_timeline() -> None:
    timeline = EnvironmentTimelineRuntime()
    session = timeline.create_session(workflow_id="workflow")
    backbone = EnvironmentObserverBackbone(timeline=timeline)

    backbone.register_defaults()
    result = backbone.emit_signal(
        ObserverSignal(
            observer_kind=EnvironmentObserverKind.PROCESS,
            event_kind=EnvironmentEventKind.APP_CRASHED,
            app_id="app",
        ),
        timeline_session_id=session.session_id,
    )

    assert result.success is True
    assert result.reason == ObserverBackboneReason.EVENT_ROUTED_TO_TIMELINE
    assert backbone.snapshot().timeline_routed_count == 1
    assert timeline.snapshot().environment_event_count == 1
    assert timeline.snapshot().failure_count == 1


def test_timeline_missing_is_reported() -> None:
    backbone = EnvironmentObserverBackbone()
    backbone.register_defaults()

    result = backbone.emit_signal(
        ObserverSignal(
            observer_kind=EnvironmentObserverKind.PROCESS,
            event_kind=EnvironmentEventKind.APP_CRASHED,
            app_id="app",
        ),
        timeline_session_id="missing",
    )

    assert result.success is False
    assert result.reason == ObserverBackboneReason.TIMELINE_SESSION_MISSING


def test_snapshot_tracks_observers_and_events() -> None:
    backbone = EnvironmentObserverBackbone()

    backbone.register_defaults()
    backbone.start_observer(EnvironmentObserverKind.ACTIVE_WINDOW)
    backbone.emit_signal(
        ObserverSignal(
            observer_kind=EnvironmentObserverKind.ACTIVE_WINDOW,
            event_kind=EnvironmentEventKind.WINDOW_FOCUSED,
            window_id="window",
        )
    )
    snapshot = backbone.snapshot()

    assert snapshot.observer_count == 9
    assert snapshot.started_count == 1
    assert snapshot.emitted_event_count == 1
    assert snapshot.runtime_event_count >= 11


def test_reset_clears_backbone() -> None:
    backbone = EnvironmentObserverBackbone()

    backbone.register_defaults()
    backbone.reset()
    snapshot = backbone.snapshot()

    assert snapshot.observer_count == 0
    assert snapshot.emitted_event_count == 0
    assert snapshot.runtime_event_count == 1
    assert snapshot.last_reason == ObserverBackboneReason.RUNTIME_RESET


def test_default_observer_priorities() -> None:
    observers = {
        observer.kind: observer
        for observer in default_environment_observers()
    }

    assert observers[EnvironmentObserverKind.ACTIVE_WINDOW].priority == (
        ObserverEventPriority.HIGH
    )
    assert observers[EnvironmentObserverKind.CURSOR].priority == (
        ObserverEventPriority.LOW
    )
    assert observers[EnvironmentObserverKind.MODAL].priority == (
        ObserverEventPriority.HIGH
    )


def test_enum_values_are_stable() -> None:
    assert EnvironmentObserverKind.ACTIVE_WINDOW.value == "active_window_observer"
    assert EnvironmentObserverStatus.STARTED.value == "started"
    assert ObserverBackboneReason.EVENT_ACCEPTED.value == "event_accepted"