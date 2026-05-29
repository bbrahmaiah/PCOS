from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    ChangeCause,
    EnvironmentDelta,
    EnvironmentEvent,
    EnvironmentEventKind,
    EnvironmentSnapshot,
    EnvironmentSource,
    EnvironmentTimelineRuntime,
    StateTransition,
    StateTransitionKind,
    TimelineQuery,
    TimelineQueryKind,
    TimelineReason,
    trusted_timeline_observation,
)


def test_runtime_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        EnvironmentTimelineRuntime(name=" ")


def test_session_requires_workflow_id() -> None:
    runtime = EnvironmentTimelineRuntime()

    with pytest.raises(ValidationError):
        runtime.create_session(workflow_id=" ")


def test_create_session() -> None:
    runtime = EnvironmentTimelineRuntime()
    session = runtime.create_session(workflow_id="workflow")

    assert session.workflow_id == "workflow"
    assert runtime.snapshot().session_count == 1


def test_record_snapshot_creates_snapshot_transition() -> None:
    runtime = EnvironmentTimelineRuntime()
    session = runtime.create_session(workflow_id="workflow")
    snapshot = _snapshot()

    result = runtime.record_snapshot(
        session_id=session.session_id,
        snapshot=snapshot,
    )

    transitions = runtime.transitions_for(session.session_id)

    assert result.success is True
    assert result.reason == TimelineReason.SNAPSHOT_RECORDED
    assert runtime.snapshots_for(session.session_id) == (snapshot,)
    assert transitions[0].kind == StateTransitionKind.SNAPSHOT_RECORDED


def test_record_snapshot_rejects_missing_session() -> None:
    runtime = EnvironmentTimelineRuntime()

    result = runtime.record_snapshot(
        session_id="missing",
        snapshot=_snapshot(),
    )

    assert result.success is False
    assert result.reason == TimelineReason.SESSION_NOT_FOUND


def test_record_delta_tracks_appeared_element() -> None:
    runtime = EnvironmentTimelineRuntime()
    session = runtime.create_session(workflow_id="workflow")
    delta = _delta(appeared=("error-panel",), hint="error panel appeared")

    result = runtime.record_delta(
        session_id=session.session_id,
        delta=delta,
        cause=ChangeCause.ERROR_APPEARED,
    )

    transition = runtime.transitions_for(session.session_id)[0]

    assert result.success is True
    assert transition.kind == StateTransitionKind.FAILED
    assert transition.cause == ChangeCause.ERROR_APPEARED
    assert transition.entity_id == "error-panel"


def test_record_delta_tracks_disappeared_element() -> None:
    runtime = EnvironmentTimelineRuntime()
    session = runtime.create_session(workflow_id="workflow")
    delta = _delta(disappeared=("dialog",))

    runtime.record_delta(session_id=session.session_id, delta=delta)
    transition = runtime.transitions_for(session.session_id)[0]

    assert transition.kind == StateTransitionKind.DISAPPEARED


def test_record_transition_for_user_action() -> None:
    runtime = EnvironmentTimelineRuntime()
    session = runtime.create_session(workflow_id="workflow")
    transition = StateTransition(
        session_id=session.session_id,
        kind=StateTransitionKind.USER_DID,
        cause=ChangeCause.USER_ACTION,
        entity_id="button-run",
        summary="user clicked run",
        actor="user",
        trust=trusted_timeline_observation(),
    )

    result = runtime.record_transition(transition)

    assert result.success is True
    assert runtime.snapshot().user_action_count == 1


def test_record_transition_for_jarvis_action() -> None:
    runtime = EnvironmentTimelineRuntime()
    session = runtime.create_session(workflow_id="workflow")
    transition = StateTransition(
        session_id=session.session_id,
        kind=StateTransitionKind.JARVIS_DID,
        cause=ChangeCause.JARVIS_ACTION,
        entity_id="line-47",
        summary="JARVIS edited line 47",
        actor="jarvis",
        trust=trusted_timeline_observation(),
    )

    runtime.record_transition(transition)

    assert runtime.snapshot().jarvis_action_count == 1


def test_record_environment_event_maps_to_transition() -> None:
    runtime = EnvironmentTimelineRuntime()
    session = runtime.create_session(workflow_id="workflow")
    event = EnvironmentEvent(
        kind=EnvironmentEventKind.APP_CRASHED,
        source=EnvironmentSource.OS_OBSERVER,
        app_id="app",
        trust=trusted_timeline_observation(),
    )

    result = runtime.record_environment_event(
        session_id=session.session_id,
        environment_event=event,
    )

    transition = runtime.transitions_for(session.session_id)[0]

    assert result.success is True
    assert transition.kind == StateTransitionKind.FAILED
    assert transition.cause == ChangeCause.APP_STATE_CHANGE


def test_query_recent_changes() -> None:
    runtime = EnvironmentTimelineRuntime()
    session = runtime.create_session(workflow_id="workflow")

    runtime.record_delta(
        session_id=session.session_id,
        delta=_delta(appeared=("terminal-output",)),
    )
    result = runtime.query(
        TimelineQuery(
            session_id=session.session_id,
            kind=TimelineQueryKind.RECENT_CHANGES,
        )
    )

    assert result.transitions
    assert result.deltas
    assert "timeline query recent_changes" in result.summary


def test_query_failures() -> None:
    runtime = EnvironmentTimelineRuntime()
    session = runtime.create_session(workflow_id="workflow")

    runtime.record_delta(
        session_id=session.session_id,
        delta=_delta(appeared=("error",)),
        cause=ChangeCause.ERROR_APPEARED,
    )
    result = runtime.query(
        TimelineQuery(
            session_id=session.session_id,
            kind=TimelineQueryKind.FAILURES,
        )
    )

    assert len(result.transitions) == 1
    assert result.transitions[0].kind == StateTransitionKind.FAILED


def test_query_user_actions() -> None:
    runtime = EnvironmentTimelineRuntime()
    session = runtime.create_session(workflow_id="workflow")
    transition = StateTransition(
        session_id=session.session_id,
        kind=StateTransitionKind.USER_DID,
        cause=ChangeCause.USER_ACTION,
        summary="user moved mouse",
        actor="user",
        trust=trusted_timeline_observation(),
    )

    runtime.record_transition(transition)
    result = runtime.query(
        TimelineQuery(
            session_id=session.session_id,
            kind=TimelineQueryKind.USER_ACTIONS,
        )
    )

    assert result.transitions[0].cause == ChangeCause.USER_ACTION


def test_query_related_to_entity() -> None:
    runtime = EnvironmentTimelineRuntime()
    session = runtime.create_session(workflow_id="workflow")

    runtime.record_delta(
        session_id=session.session_id,
        delta=_delta(changed=("line-47",)),
        cause=ChangeCause.JARVIS_ACTION,
    )
    result = runtime.query(
        TimelineQuery(
            session_id=session.session_id,
            kind=TimelineQueryKind.RELATED_TO_ENTITY,
            entity_id="line-47",
        )
    )

    assert result.transitions[0].entity_id == "line-47"


def test_recent_state_history_enforces_session_limit() -> None:
    runtime = EnvironmentTimelineRuntime()
    session = runtime.create_session(workflow_id="workflow", max_history=2)

    runtime.record_snapshot(session_id=session.session_id, snapshot=_snapshot())
    runtime.record_snapshot(session_id=session.session_id, snapshot=_snapshot())
    runtime.record_snapshot(session_id=session.session_id, snapshot=_snapshot())

    history = runtime.recent_state_history(session.session_id)

    assert history is not None
    assert len(history.snapshots) == 2


def test_temporal_workspace_state_tracks_last_actions() -> None:
    runtime = EnvironmentTimelineRuntime()
    session = runtime.create_session(workflow_id="workflow")

    runtime.record_snapshot(session_id=session.session_id, snapshot=_snapshot())
    runtime.record_transition(
        StateTransition(
            session_id=session.session_id,
            kind=StateTransitionKind.USER_DID,
            cause=ChangeCause.USER_ACTION,
            summary="user clicked run",
            actor="user",
            trust=trusted_timeline_observation(),
        )
    )
    runtime.record_transition(
        StateTransition(
            session_id=session.session_id,
            kind=StateTransitionKind.JARVIS_DID,
            cause=ChangeCause.JARVIS_ACTION,
            summary="JARVIS edited line 47",
            actor="jarvis",
            trust=trusted_timeline_observation(),
        )
    )
    state = runtime.temporal_workspace_state(session.session_id)

    assert state is not None
    assert state.last_user_action == "user clicked run"
    assert state.last_jarvis_action == "JARVIS edited line 47"


def test_clear_session_removes_history() -> None:
    runtime = EnvironmentTimelineRuntime()
    session = runtime.create_session(workflow_id="workflow")

    runtime.record_snapshot(session_id=session.session_id, snapshot=_snapshot())
    result = runtime.clear_session(session.session_id)

    assert result.success is True
    assert runtime.snapshots_for(session.session_id) == ()
    assert runtime.transitions_for(session.session_id) == ()


def test_snapshot_tracks_counts() -> None:
    runtime = EnvironmentTimelineRuntime()
    session = runtime.create_session(workflow_id="workflow")

    runtime.record_snapshot(session_id=session.session_id, snapshot=_snapshot())
    runtime.record_delta(
        session_id=session.session_id,
        delta=_delta(appeared=("error",)),
        cause=ChangeCause.ERROR_APPEARED,
    )
    runtime.record_delta(
        session_id=session.session_id,
        delta=_delta(disappeared=("error",)),
        cause=ChangeCause.ERROR_RESOLVED,
    )
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.snapshot_count == 1
    assert snapshot.delta_count == 2
    assert snapshot.failure_count == 1
    assert snapshot.recovery_count == 1


def test_reset_clears_runtime() -> None:
    runtime = EnvironmentTimelineRuntime()
    session = runtime.create_session(workflow_id="workflow")

    runtime.record_snapshot(session_id=session.session_id, snapshot=_snapshot())
    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.runtime_event_count == 1
    assert snapshot.last_reason == TimelineReason.RUNTIME_RESET


def test_query_missing_session_raises() -> None:
    runtime = EnvironmentTimelineRuntime()

    with pytest.raises(ValueError):
        runtime.query(
            TimelineQuery(
                session_id="missing",
                kind=TimelineQueryKind.RECENT_CHANGES,
            )
        )


def test_enum_values_are_stable() -> None:
    assert ChangeCause.JARVIS_ACTION.value == "jarvis_action"
    assert StateTransitionKind.APPEARED.value == "appeared"
    assert TimelineQueryKind.FAILURES.value == "failures"
    assert TimelineReason.SNAPSHOT_RECORDED.value == "snapshot_recorded"


def _snapshot() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(trust=trusted_timeline_observation())


def _delta(
    *,
    appeared: tuple[str, ...] = (),
    disappeared: tuple[str, ...] = (),
    changed: tuple[str, ...] = (),
    hint: str | None = None,
) -> EnvironmentDelta:
    return EnvironmentDelta(
        current_snapshot_id="snapshot_current",
        appeared_elements=appeared,
        disappeared_elements=disappeared,
        changed_elements=changed,
        cause_hint=hint,
        trust=trusted_timeline_observation(),
    )