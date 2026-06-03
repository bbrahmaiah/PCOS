from __future__ import annotations

import pytest

from jarvis.cognitive import (
    AttentionSignalUrgency,
    BehaviorIntent,
    CognitiveIntegrationEvent,
    CognitiveIntegrationEventKind,
    CognitiveIntegrationRequest,
    CognitiveIntegrationRuntime,
    CognitiveIntegrationSource,
    CognitiveIntegrationStatus,
    GoalPriority,
    PlanIntentKind,
    WorkingMemoryKind,
    make_cognitive_integration_event,
)


def test_cognitive_integration_event_rejects_empty_title() -> None:
    with pytest.raises(ValueError):
        make_cognitive_integration_event(
            source=CognitiveIntegrationSource.CONVERSATION,
            kind=CognitiveIntegrationEventKind.USER_UTTERANCE,
            title=" ",
            summary="Continue Phase 9.",
            urgency=AttentionSignalUrgency.IMPORTANT,
        )


def test_cognitive_integration_runtime_degrades_without_events() -> None:
    runtime = CognitiveIntegrationRuntime()

    result = runtime.ingest(CognitiveIntegrationRequest(events=()))

    assert result.status == CognitiveIntegrationStatus.DEGRADED
    assert result.processed_events == ()


def test_presence_interruption_updates_attention_and_behavior() -> None:
    runtime = CognitiveIntegrationRuntime()
    event = make_cognitive_integration_event(
        source=CognitiveIntegrationSource.PRESENCE,
        kind=CognitiveIntegrationEventKind.INTERRUPTION,
        title="User interrupted",
        summary="Stopping current speech and listening.",
        urgency=AttentionSignalUrgency.EMERGENCY,
        assistant_is_speaking=True,
    )

    result = runtime.ingest(
        CognitiveIntegrationRequest(
            events=(event,),
            start_session=True,
            user_label="Balu",
        )
    )

    assert result.status == CognitiveIntegrationStatus.READY
    assert result.should_interrupt is True
    assert result.session_result.session.attention.interrupt_items
    assert result.behavior_results
    assert result.behavior_results[0].behavior_result is not None
    assert result.behavior_results[0].behavior_result.text == (
        "Stopping. Listening now."
    )


def test_developer_build_event_updates_working_memory() -> None:
    runtime = CognitiveIntegrationRuntime()
    event = make_cognitive_integration_event(
        source=CognitiveIntegrationSource.DEVELOPER,
        kind=CognitiveIntegrationEventKind.DEVELOPER_BUILD,
        title="Build watch",
        summary="Tests passed successfully.",
        urgency=AttentionSignalUrgency.NORMAL,
    )

    result = runtime.ingest(CognitiveIntegrationRequest(events=(event,)))

    assert result.status == CognitiveIntegrationStatus.READY
    assert result.session_result.session.working_memory.items
    assert result.session_result.session.working_memory.items[0].kind == (
        WorkingMemoryKind.TASK
    )


def test_developer_error_event_warns_and_tracks_risk() -> None:
    runtime = CognitiveIntegrationRuntime()
    event = make_cognitive_integration_event(
        source=CognitiveIntegrationSource.DEVELOPER,
        kind=CognitiveIntegrationEventKind.DEVELOPER_ERROR,
        title="Test failure",
        summary="A type error remains in the cognitive integration layer.",
        urgency=AttentionSignalUrgency.URGENT,
    )

    result = runtime.ingest(CognitiveIntegrationRequest(events=(event,)))

    assert result.status == CognitiveIntegrationStatus.READY
    assert result.behavior_results
    assert result.behavior_results[0].behavior_result is not None
    assert result.behavior_results[0].behavior_result.directive.should_warn
    assert result.session_result.session.working_memory.items[0].kind == (
        WorkingMemoryKind.RISK
    )


def test_goal_request_creates_goal_and_plan() -> None:
    runtime = CognitiveIntegrationRuntime()
    event = make_cognitive_integration_event(
        source=CognitiveIntegrationSource.CONVERSATION,
        kind=CognitiveIntegrationEventKind.GOAL_REQUEST,
        title="Continue Phase 9",
        summary="Build Phase 9 integration.",
        urgency=AttentionSignalUrgency.IMPORTANT,
        goal_title="Build Phase 9 integration",
        goal_description="Connect cognitive session to runtime sources.",
        goal_priority=GoalPriority.HIGH,
        plan_intent_kind=PlanIntentKind.DEVELOPER,
    )

    result = runtime.ingest(CognitiveIntegrationRequest(events=(event,)))

    assert result.status == CognitiveIntegrationStatus.READY
    assert result.goal_results
    assert result.goal_results[0].goal_result is not None
    assert result.goal_results[0].goal_result.goal is not None
    assert result.goal_results[0].planning_result is not None
    assert result.session_result.session.goals.has_active_goal is True


def test_memory_recall_event_becomes_working_memory_context() -> None:
    runtime = CognitiveIntegrationRuntime()
    event = make_cognitive_integration_event(
        source=CognitiveIntegrationSource.MEMORY,
        kind=CognitiveIntegrationEventKind.MEMORY_RECALL,
        title="Project memory",
        summary="User is building JARVIS Phase 9.",
        urgency=AttentionSignalUrgency.IMPORTANT,
        working_memory_kind=WorkingMemoryKind.PROJECT,
    )

    result = runtime.ingest(CognitiveIntegrationRequest(events=(event,)))

    assert result.status == CognitiveIntegrationStatus.READY
    item = result.session_result.session.working_memory.items[0]

    assert item.kind == WorkingMemoryKind.PROJECT
    assert "Phase 9" in item.value


def test_system_health_emergency_interrupts() -> None:
    runtime = CognitiveIntegrationRuntime()
    event = make_cognitive_integration_event(
        source=CognitiveIntegrationSource.SYSTEM,
        kind=CognitiveIntegrationEventKind.SYSTEM_HEALTH,
        title="Battery critical",
        summary="Battery level is critically low.",
        urgency=AttentionSignalUrgency.EMERGENCY,
    )

    result = runtime.ingest(CognitiveIntegrationRequest(events=(event,)))

    assert result.should_interrupt is True
    assert result.behavior_results
    assert result.behavior_results[0].behavior_result is not None
    assert "I would advise caution." in result.behavior_results[
        0
    ].behavior_result.text


def test_status_event_can_use_explicit_behavior_intent() -> None:
    runtime = CognitiveIntegrationRuntime()
    event = make_cognitive_integration_event(
        source=CognitiveIntegrationSource.COGNITION,
        kind=CognitiveIntegrationEventKind.STATUS,
        title="Ready",
        summary="Phase 9 integration is ready.",
        urgency=AttentionSignalUrgency.NORMAL,
        behavior_intent=BehaviorIntent.CONFIRMATION,
    )

    result = runtime.ingest(CognitiveIntegrationRequest(events=(event,)))

    assert result.behavior_results
    behavior = result.behavior_results[0].behavior_result

    assert behavior is not None
    assert behavior.text.startswith("Certainly, sir.")


def test_integration_snapshot_tracks_counts() -> None:
    runtime = CognitiveIntegrationRuntime()
    event = make_cognitive_integration_event(
        source=CognitiveIntegrationSource.CONVERSATION,
        kind=CognitiveIntegrationEventKind.USER_UTTERANCE,
        title="Continue",
        summary="Continue the current task.",
        urgency=AttentionSignalUrgency.IMPORTANT,
    )

    runtime.ingest(CognitiveIntegrationRequest(events=(event,)))
    snapshot = runtime.snapshot()

    assert snapshot.status == CognitiveIntegrationStatus.READY
    assert snapshot.ingest_count == 1
    assert snapshot.event_count == 1


def test_integration_event_validation() -> None:
    with pytest.raises(ValueError):
        CognitiveIntegrationEvent(
            event_id=" ",
            source=CognitiveIntegrationSource.SYSTEM,
            kind=CognitiveIntegrationEventKind.STATUS,
            title="Status",
            summary="Status event.",
            urgency=AttentionSignalUrgency.NORMAL,
        )


def test_integration_enum_values_are_stable() -> None:
    assert CognitiveIntegrationStatus.READY.value == "ready"
    assert CognitiveIntegrationSource.DEVELOPER.value == "developer"
    assert CognitiveIntegrationEventKind.GOAL_REQUEST.value == "goal_request"