from __future__ import annotations

from typing import Any, cast

import pytest

from jarvis.runtime.events import EventBus
from jarvis.runtime.shared.enums import EventType, RuntimeStatus, SystemMode
from jarvis.runtime.state import GlobalContext, RuntimeState, StateEngine


def test_runtime_state_defaults_are_valid() -> None:
    state = RuntimeState()

    assert state.status == RuntimeStatus.CREATED
    assert state.mode == SystemMode.PASSIVE
    assert state.updated_at is not None


def test_global_context_set_get_delete_snapshot_clear() -> None:
    context = GlobalContext()

    context.set("active_window", {"title": "VS Code"})

    assert context.get("active_window") == {"title": "VS Code"}
    assert context.snapshot().size == 1

    assert context.delete("active_window") is True
    assert context.delete("active_window") is False

    context.set("x", 1)
    context.clear()

    assert context.snapshot().size == 0


def test_global_context_rejects_empty_key() -> None:
    context = GlobalContext()

    with pytest.raises(ValueError):
        context.set("   ", 1)

    with pytest.raises(ValueError):
        context.get("   ")


def test_global_context_rejects_non_string_key() -> None:
    context = GlobalContext()

    with pytest.raises(TypeError):
        context.set(cast(Any, 123), "bad")


def test_global_context_update_many() -> None:
    context = GlobalContext()

    context.update_many(
        {
            "current_app": "VS Code",
            "browser_url": "https://example.com",
        }
    )

    snapshot = context.snapshot()

    assert snapshot.size == 2
    assert snapshot.values["current_app"] == "VS Code"


def test_global_context_update_many_rejects_non_dict() -> None:
    context = GlobalContext()

    with pytest.raises(TypeError):
        context.update_many(cast(Any, ["not", "dict"]))


def test_global_context_defensive_copy() -> None:
    context = GlobalContext()
    value = {"items": [1, 2]}

    context.set("value", value)
    value["items"].append(3)

    stored = context.get("value")

    assert stored == {"items": [1, 2]}


def test_state_engine_initial_snapshot() -> None:
    bus = EventBus(name="test_bus")
    engine = StateEngine(event_bus=bus)

    snapshot = engine.snapshot()

    assert snapshot.runtime.status == RuntimeStatus.CREATED
    assert snapshot.runtime.mode == SystemMode.PASSIVE
    assert snapshot.session is None
    assert snapshot.context.size == 0


def test_state_engine_updates_runtime_status() -> None:
    bus = EventBus(name="test_bus")
    engine = StateEngine(event_bus=bus)

    state = engine.set_runtime_status(RuntimeStatus.RUNNING)

    assert state.status == RuntimeStatus.RUNNING
    assert state.started_at is not None

    history = bus.history()

    assert history[-1].event_type == EventType.STATE_UPDATED
    assert history[-1].payload["field"] == "runtime.status"


def test_state_engine_updates_system_mode() -> None:
    bus = EventBus(name="test_bus")
    engine = StateEngine(event_bus=bus)

    state = engine.set_system_mode(SystemMode.ACTIVE)

    assert state.mode == SystemMode.ACTIVE
    assert bus.history()[-1].payload["field"] == "runtime.mode"


def test_state_engine_session_lifecycle() -> None:
    bus = EventBus(name="test_bus")
    engine = StateEngine(event_bus=bus)

    session = engine.start_session(
        user_id="bala",
        active_goal="Build JARVIS",
        active_topic="State Engine",
    )

    assert session.active is True
    assert session.user_id == "bala"
    assert session.active_goal == "Build JARVIS"

    updated_goal = engine.update_session_goal("Improve runtime")
    updated_topic = engine.update_session_topic("Workers")

    assert updated_goal.active_goal == "Improve runtime"
    assert updated_topic.active_topic == "Workers"

    ended = engine.end_session()

    assert ended is not None
    assert ended.active is False
    assert ended.ended_at is not None


def test_state_engine_end_session_without_session_returns_none() -> None:
    bus = EventBus(name="test_bus")
    engine = StateEngine(event_bus=bus)

    assert engine.end_session() is None


def test_state_engine_session_update_requires_active_session() -> None:
    bus = EventBus(name="test_bus")
    engine = StateEngine(event_bus=bus)

    with pytest.raises(RuntimeError):
        engine.update_session_goal("No session")

    with pytest.raises(RuntimeError):
        engine.update_session_topic("No session")


def test_state_engine_context_operations_emit_events() -> None:
    bus = EventBus(name="test_bus")
    engine = StateEngine(event_bus=bus)

    engine.set_context("active_window", "VS Code")
    engine.update_context({"current_app": "VS Code", "cpu": 12.5})
    removed = engine.delete_context("cpu")
    engine.clear_context()

    assert removed is True
    assert engine.snapshot().context.size == 0

    event_types = [event.event_type for event in bus.history()]

    assert EventType.CONTEXT_UPDATED in event_types


def test_state_engine_context_get_default() -> None:
    bus = EventBus(name="test_bus")
    engine = StateEngine(event_bus=bus)

    assert engine.get_context("missing", "default") == "default"


def test_state_engine_runtime_failure_records_error() -> None:
    bus = EventBus(name="test_bus")
    engine = StateEngine(event_bus=bus)

    state = engine.set_runtime_status(
        RuntimeStatus.FAILED,
        last_error="boom",
    )

    assert state.status == RuntimeStatus.FAILED
    assert state.last_error == "boom"
    assert state.stopped_at is not None