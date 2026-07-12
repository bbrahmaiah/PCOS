from __future__ import annotations

from jarvis.cognition import (
    FakeCognitionAdapter,
    FakeCognitionConfig,
    FakeCognitionMode,
)
from jarvis.cognition.action_planning import ToolActionPlanner
from jarvis.cognition.memory import InMemoryShortTermMemoryStore
from jarvis.cognition.planning import ResponsePlanner
from jarvis.cognition.runtime import (
    CognitionRuntime,
    CognitionRuntimeComponents,
    CognitionRuntimeConfig,
)
from jarvis.cognition.session_context import ConversationSessionStore
from jarvis.cognition.spoken_policy import SpokenDialoguePolicy
from jarvis.cognitive import (
    AttentionEvaluationRequest,
    AttentionItemKind,
    AttentionPriority,
    AttentionRuntime,
    AttentionSignalSource,
    AttentionSignalUrgency,
    MissionContextInput,
    MissionContextInterruptionPolicy,
    MissionContextRuntime,
    MissionContextUrgency,
    WorkingMemoryKind,
    WorkingMemoryRuntime,
    WorkingMemoryUpdateRequest,
    make_attention_signal,
    make_working_memory_entry,
)


def test_mission_context_tracks_project_goal_task_and_urgency() -> None:
    runtime = MissionContextRuntime()
    battery_signal = make_attention_signal(
        source=AttentionSignalSource.SAFETY,
        kind=AttentionItemKind.SAFETY,
        title="Battery critical",
        summary="Battery is below the safe threshold.",
        urgency=AttentionSignalUrgency.EMERGENCY,
    )
    attention = AttentionRuntime().evaluate(
        AttentionEvaluationRequest(signals=(battery_signal,))
    )

    result = runtime.update(
        MissionContextInput(
            current_project="Pure JARVIS",
            current_goal="Stabilize the voice spine",
            current_task="Run the connected launcher",
            environment="VS Code and PowerShell",
            attention_items=attention.state.items,
        )
    )

    assert result.succeeded is True
    assert result.state.current_project == "Pure JARVIS"
    assert result.state.current_goal == "Stabilize the voice spine"
    assert result.state.current_task == "Run the connected launcher"
    assert result.state.urgency == MissionContextUrgency.CRITICAL
    assert (
        result.state.interruption_policy
        == MissionContextInterruptionPolicy.INTERRUPT_NOW
    )
    assert result.state.should_interrupt is True
    assert "Battery critical" in (result.state.risk_summary or "")


def test_mission_context_uses_working_memory_when_direct_fields_are_missing() -> None:
    runtime = MissionContextRuntime()
    project_memory = make_working_memory_entry(
        kind=WorkingMemoryKind.PROJECT,
        key="project",
        value="JARVIS_OS",
        importance=AttentionPriority.HIGH,
    )
    task_memory = make_working_memory_entry(
        kind=WorkingMemoryKind.TASK,
        key="task",
        value="Build mission context engine",
    )
    memory = WorkingMemoryRuntime().update(
        WorkingMemoryUpdateRequest(entries=(project_memory, task_memory))
    )

    result = runtime.update(
        MissionContextInput(
            request_text="continue from where we stopped",
            working_memory_items=memory.state.items,
        )
    )

    assert result.state.current_project == "JARVIS_OS"
    assert result.state.current_task == "Build mission context engine"
    assert result.state.urgency == MissionContextUrgency.URGENT
    assert result.state.should_respond_now is True


def test_mission_context_enriches_cognition_runtime_every_turn() -> None:
    runtime = _make_cognition_runtime()

    result = runtime.process_text(
        "continue the JARVIS build",
        metadata={
            "current_project": "JARVIS_OS",
            "current_goal": "Build a PCOS foundation",
            "current_task": "Mission context integration",
            "environment": "PowerShell",
        },
    )

    assert result.succeeded is True
    context_items = result.enriched_request.context.items
    mission_items = tuple(
        item for item in context_items if item.kind == "mission_context"
    )
    assert len(mission_items) == 1
    assert "JARVIS_OS" in mission_items[0].text
    assert "Mission context integration" in mission_items[0].text
    assert (
        result.enriched_request.metadata["mission_context_policy"]
        == MissionContextInterruptionPolicy.RESPOND_NOW.value
    )

    snapshot = runtime.snapshot()
    assert snapshot.mission_context_update_count == 1
    assert snapshot.mission_context_policy == "respond_now"


def test_mission_context_can_be_disabled_for_runtime_boundaries() -> None:
    runtime = _make_cognition_runtime(
        config=CognitionRuntimeConfig(enable_mission_context=False)
    )

    result = runtime.process_text(
        "continue the JARVIS build",
        metadata={"current_project": "JARVIS_OS"},
    )

    assert result.succeeded is True
    assert all(
        item.kind != "mission_context"
        for item in result.enriched_request.context.items
    )


def _make_cognition_runtime(
    *,
    config: CognitionRuntimeConfig | None = None,
) -> CognitionRuntime:
    adapter = FakeCognitionAdapter(
        config=FakeCognitionConfig(mode=FakeCognitionMode.ECHO)
    )
    return CognitionRuntime(
        components=CognitionRuntimeComponents(
            adapter=adapter,
            session_store=ConversationSessionStore(),
            memory_store=InMemoryShortTermMemoryStore(),
            response_planner=ResponsePlanner(),
            action_planner=ToolActionPlanner(),
            spoken_policy=SpokenDialoguePolicy(),
            mission_context=MissionContextRuntime(),
        ),
        config=config,
    )
