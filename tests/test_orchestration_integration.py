from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.orchestration import (
    CognitionPhaseAdapter,
    IntegratedPhase,
    IntegratedTaskEnvelope,
    IntegratedTaskKind,
    IntegratedWorkerKind,
    MemoryPhaseAdapter,
    PhaseAdapterCapability,
    PhaseEvent,
    PhaseEventKind,
    PhaseIntegrationConfig,
    PhaseIntegrationReason,
    PhaseIntegrationRuntime,
    PhaseIntegrationStatus,
    PhaseWorkerAdapter,
    PresencePhaseAdapter,
    ToolPhaseAdapter,
)


def event(
    *,
    source_phase: IntegratedPhase = IntegratedPhase.PRESENCE,
    event_kind: PhaseEventKind = PhaseEventKind.USER_TURN_FINALIZED,
    direct_execution_requested: bool = False,
) -> PhaseEvent:
    return PhaseEvent(
        source_phase=source_phase,
        event_kind=event_kind,
        payload={"text": "hello"},
        direct_execution_requested=direct_execution_requested,
    )


def test_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        PhaseIntegrationConfig(name=" ").validate()


def test_phase_event_requires_event_id() -> None:
    with pytest.raises(ValidationError):
        PhaseEvent(
            event_id=" ",
            source_phase=IntegratedPhase.PRESENCE,
            event_kind=PhaseEventKind.USER_TURN_FINALIZED,
        )


def test_capability_requires_accepted_events() -> None:
    with pytest.raises(ValidationError):
        PhaseAdapterCapability(
            accepted_events=(),
            emitted_tasks=(IntegratedTaskKind.COGNITION_TASK,),
        )


def test_capability_requires_emitted_tasks() -> None:
    with pytest.raises(ValidationError):
        PhaseAdapterCapability(
            accepted_events=(PhaseEventKind.COGNITION_REQUESTED,),
            emitted_tasks=(),
        )


def test_integrated_envelope_blocks_direct_execution() -> None:
    with pytest.raises(ValidationError):
        IntegratedTaskEnvelope(
            source_event_id="event-1",
            source_phase=IntegratedPhase.TOOLS,
            target_worker=IntegratedWorkerKind.TOOL_WORKER,
            task_kind=IntegratedTaskKind.TOOL_TASK,
            direct_execution_allowed=True,
        )


def test_tool_envelope_requires_policy() -> None:
    with pytest.raises(ValidationError):
        IntegratedTaskEnvelope(
            source_event_id="event-1",
            source_phase=IntegratedPhase.TOOLS,
            target_worker=IntegratedWorkerKind.TOOL_WORKER,
            task_kind=IntegratedTaskKind.TOOL_TASK,
            requires_policy=False,
        )


def test_cognition_envelope_requires_snapshot() -> None:
    with pytest.raises(ValidationError):
        IntegratedTaskEnvelope(
            source_event_id="event-1",
            source_phase=IntegratedPhase.COGNITION,
            target_worker=IntegratedWorkerKind.COGNITION_WORKER,
            task_kind=IntegratedTaskKind.COGNITION_TASK,
            requires_context_snapshot=False,
        )


def test_presence_adapter_accepts_presence_event() -> None:
    adapter = PresencePhaseAdapter()

    result = adapter.build_envelope(event())

    assert result.success is True
    assert result.reason == PhaseIntegrationReason.TASK_ENVELOPE_CREATED
    assert result.envelope is not None
    assert result.envelope.task_kind == IntegratedTaskKind.TURN_COORDINATION
    assert result.envelope.target_worker == IntegratedWorkerKind.PRESENCE_WORKER


def test_presence_adapter_rejects_wrong_phase() -> None:
    adapter = PresencePhaseAdapter()

    result = adapter.build_envelope(
        event(
            source_phase=IntegratedPhase.COGNITION,
            event_kind=PhaseEventKind.COGNITION_REQUESTED,
        )
    )

    assert result.success is False
    assert result.reason == PhaseIntegrationReason.PHASE_EVENT_REJECTED


def test_presence_interrupt_maps_to_interrupt_coordination() -> None:
    adapter = PresencePhaseAdapter()

    result = adapter.build_envelope(
        event(event_kind=PhaseEventKind.USER_INTERRUPTED)
    )

    assert result.envelope is not None
    assert result.envelope.task_kind == IntegratedTaskKind.INTERRUPT_COORDINATION


def test_adapter_blocks_direct_execution_request() -> None:
    adapter = ToolPhaseAdapter()

    result = adapter.build_envelope(
        event(
            source_phase=IntegratedPhase.TOOLS,
            event_kind=PhaseEventKind.TOOL_REQUESTED,
            direct_execution_requested=True,
        )
    )

    assert result.success is False
    assert result.reason == PhaseIntegrationReason.DIRECT_EXECUTION_BLOCKED


def test_adapter_mapping_must_match_capabilities() -> None:
    with pytest.raises(ValueError):
        PhaseWorkerAdapter(
            phase=IntegratedPhase.COGNITION,
            worker_kind=IntegratedWorkerKind.COGNITION_WORKER,
            capabilities=PhaseAdapterCapability(
                accepted_events=(PhaseEventKind.COGNITION_REQUESTED,),
                emitted_tasks=(IntegratedTaskKind.COGNITION_TASK,),
            ),
            event_task_map={
                PhaseEventKind.MEMORY_REQUESTED: IntegratedTaskKind.MEMORY_TASK
            },
        )


def test_runtime_registers_default_adapters() -> None:
    runtime = PhaseIntegrationRuntime()

    snapshot = runtime.snapshot()

    assert snapshot.adapter_count == 6
    assert snapshot.healthy_count == 6


def test_runtime_can_disable_default_registration() -> None:
    runtime = PhaseIntegrationRuntime(
        config=PhaseIntegrationConfig(auto_register_defaults=False)
    )

    snapshot = runtime.snapshot()

    assert snapshot.adapter_count == 0


def test_runtime_registers_adapter() -> None:
    runtime = PhaseIntegrationRuntime(
        config=PhaseIntegrationConfig(auto_register_defaults=False)
    )

    result = runtime.register_adapter(CognitionPhaseAdapter())

    assert result.success is True
    assert result.reason == PhaseIntegrationReason.ADAPTER_REGISTERED
    assert runtime.snapshot().adapter_count == 1


def test_runtime_replaces_adapter() -> None:
    runtime = PhaseIntegrationRuntime(
        config=PhaseIntegrationConfig(auto_register_defaults=False)
    )

    runtime.register_adapter(CognitionPhaseAdapter())
    result = runtime.register_adapter(CognitionPhaseAdapter())

    assert result.success is True
    assert result.reason == PhaseIntegrationReason.ADAPTER_REPLACED


def test_runtime_routes_presence_turn_event() -> None:
    runtime = PhaseIntegrationRuntime()

    result = runtime.route_event(event())

    assert result.success is True
    assert result.envelope is not None
    assert result.envelope.task_kind == IntegratedTaskKind.TURN_COORDINATION
    assert runtime.snapshot().routed_event_count == 1
    assert runtime.snapshot().emitted_envelope_count == 1


def test_runtime_routes_cognition_request_to_cognition_worker() -> None:
    runtime = PhaseIntegrationRuntime()

    result = runtime.route_event(
        event(
            source_phase=IntegratedPhase.COGNITION,
            event_kind=PhaseEventKind.COGNITION_REQUESTED,
        )
    )

    assert result.envelope is not None
    assert result.envelope.target_worker == IntegratedWorkerKind.COGNITION_WORKER
    assert result.envelope.task_kind == IntegratedTaskKind.COGNITION_TASK
    assert result.envelope.requires_context_snapshot is True


def test_runtime_routes_memory_request_to_memory_worker() -> None:
    runtime = PhaseIntegrationRuntime()

    result = runtime.route_event(
        event(
            source_phase=IntegratedPhase.MEMORY,
            event_kind=PhaseEventKind.MEMORY_REQUESTED,
        )
    )

    assert result.envelope is not None
    assert result.envelope.target_worker == IntegratedWorkerKind.MEMORY_WORKER
    assert result.envelope.task_kind == IntegratedTaskKind.MEMORY_TASK


def test_runtime_routes_tool_request_to_policy_guarded_tool_task() -> None:
    runtime = PhaseIntegrationRuntime()

    result = runtime.route_event(
        event(
            source_phase=IntegratedPhase.TOOLS,
            event_kind=PhaseEventKind.TOOL_REQUESTED,
        )
    )

    assert result.envelope is not None
    assert result.envelope.target_worker == IntegratedWorkerKind.TOOL_WORKER
    assert result.envelope.task_kind == IntegratedTaskKind.TOOL_TASK
    assert result.envelope.requires_policy is True
    assert result.envelope.requires_budget is True


def test_runtime_blocks_direct_execution_before_adapter() -> None:
    runtime = PhaseIntegrationRuntime()

    result = runtime.route_event(
        event(
            source_phase=IntegratedPhase.TOOLS,
            event_kind=PhaseEventKind.TOOL_REQUESTED,
            direct_execution_requested=True,
        )
    )

    snapshot = runtime.snapshot()

    assert result.success is False
    assert result.reason == PhaseIntegrationReason.DIRECT_EXECUTION_BLOCKED
    assert snapshot.direct_execution_block_count == 1
    assert snapshot.rejected_event_count == 1


def test_runtime_rejects_missing_adapter() -> None:
    runtime = PhaseIntegrationRuntime(
        config=PhaseIntegrationConfig(auto_register_defaults=False)
    )

    result = runtime.route_event(
        event(
            source_phase=IntegratedPhase.MEMORY,
            event_kind=PhaseEventKind.MEMORY_REQUESTED,
        )
    )

    assert result.success is False
    assert result.reason == PhaseIntegrationReason.ADAPTER_NOT_FOUND
    assert runtime.snapshot().rejected_event_count == 1


def test_runtime_routes_many_events() -> None:
    runtime = PhaseIntegrationRuntime()

    results = runtime.route_events(
        (
            event(),
            event(
                source_phase=IntegratedPhase.COGNITION,
                event_kind=PhaseEventKind.COGNITION_REQUESTED,
            ),
            event(
                source_phase=IntegratedPhase.MEMORY,
                event_kind=PhaseEventKind.MEMORY_REQUESTED,
            ),
        )
    )

    assert len(results) == 3
    assert all(result.success for result in results)
    assert runtime.snapshot().routed_event_count == 3


def test_runtime_emitted_envelopes_are_queryable() -> None:
    runtime = PhaseIntegrationRuntime()

    runtime.route_event(event())
    envelopes = runtime.emitted_envelopes()

    assert len(envelopes) == 1
    assert envelopes[0].source_phase == IntegratedPhase.PRESENCE


def test_runtime_marks_adapter_degraded() -> None:
    runtime = PhaseIntegrationRuntime()

    result = runtime.mark_adapter_degraded(IntegratedPhase.COGNITION)

    assert result.success is True
    assert result.health is not None
    assert result.health.status == PhaseIntegrationStatus.DEGRADED
    assert runtime.snapshot().degraded_count == 1


def test_runtime_isolates_adapter_and_rejects_future_events() -> None:
    runtime = PhaseIntegrationRuntime()

    runtime.isolate_adapter(IntegratedPhase.MEMORY)
    result = runtime.route_event(
        event(
            source_phase=IntegratedPhase.MEMORY,
            event_kind=PhaseEventKind.MEMORY_REQUESTED,
        )
    )

    assert result.success is False
    assert result.reason == PhaseIntegrationReason.PHASE_EVENT_REJECTED
    assert runtime.snapshot().isolated_count == 1


def test_runtime_adapter_health_returns_none_for_missing_adapter() -> None:
    runtime = PhaseIntegrationRuntime(
        config=PhaseIntegrationConfig(auto_register_defaults=False)
    )

    assert runtime.adapter_health(IntegratedPhase.TOOLS) is None


def test_runtime_reset_clears_counts_but_keeps_adapters() -> None:
    runtime = PhaseIntegrationRuntime()

    runtime.route_event(event())
    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.adapter_count == 6
    assert snapshot.routed_event_count == 0
    assert snapshot.emitted_envelope_count == 0
    assert snapshot.last_reason == PhaseIntegrationReason.RUNTIME_RESET


def test_memory_adapter_is_separate_from_cognition_adapter() -> None:
    memory = MemoryPhaseAdapter()
    cognition = CognitionPhaseAdapter()

    assert memory.phase != cognition.phase
    assert memory.worker_kind != cognition.worker_kind


def test_tool_adapter_never_allows_direct_execution() -> None:
    adapter = ToolPhaseAdapter()

    result = adapter.build_envelope(
        event(
            source_phase=IntegratedPhase.TOOLS,
            event_kind=PhaseEventKind.TOOL_REQUESTED,
        )
    )

    assert result.envelope is not None
    assert result.envelope.direct_execution_allowed is False


def test_enum_values_are_stable() -> None:
    assert IntegratedPhase.PRESENCE.value == "presence"
    assert IntegratedWorkerKind.TOOL_WORKER.value == "tool_worker"
    assert PhaseEventKind.USER_INTERRUPTED.value == "user_interrupted"
    assert IntegratedTaskKind.TOOL_TASK.value == "tool_task"
    assert PhaseIntegrationReason.TASK_ENVELOPE_CREATED.value == (
        "task_envelope_created"
    )