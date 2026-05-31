from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    Phase8BridgeMode,
    Phase8FullIntegrationRuntime,
    Phase8IntegrationBoundary,
    Phase8IntegrationDecision,
    Phase8IntegrationPacket,
    Phase8IntegrationReason,
    Phase8IntegrationRequest,
    Phase8IntegrationStatus,
)


def test_runtime_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        Phase8FullIntegrationRuntime(name=" ")


def test_packet_blocks_direct_action_payload() -> None:
    with pytest.raises(ValidationError):
        Phase8IntegrationPacket(
            boundary=Phase8IntegrationBoundary.PHASE1_EVENT_BUS,
            source_phase="phase8",
            target_phase="phase1",
            workspace_id="workspace",
            payload={"direct_click": True},
            mode=Phase8BridgeMode.READ_ONLY,
        )


def test_memory_packet_requires_gateway() -> None:
    with pytest.raises(ValidationError):
        Phase8IntegrationPacket(
            boundary=Phase8IntegrationBoundary.PHASE4_MEMORY_GATEWAY,
            source_phase="phase8",
            target_phase="phase4",
            workspace_id="workspace",
            payload={"memory": "workflow"},
            mode=Phase8BridgeMode.POLICY_GATED,
            gateway_required=False,
        )


def test_action_packet_requires_policy() -> None:
    with pytest.raises(ValidationError):
        Phase8IntegrationPacket(
            boundary=Phase8IntegrationBoundary.PHASE5_ACTION_POLICY,
            source_phase="phase8",
            target_phase="phase5",
            workspace_id="workspace",
            payload={"action": "click"},
            mode=Phase8BridgeMode.POLICY_GATED,
            policy_required=False,
        )


def test_create_session() -> None:
    runtime = Phase8FullIntegrationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    assert session.workspace_id == "workspace"
    assert runtime.snapshot().session_count == 1


def test_phase1_event_bus_integration() -> None:
    runtime = Phase8FullIntegrationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.integrate(
        _request(
            session_id=session.session_id,
            boundary=Phase8IntegrationBoundary.PHASE1_EVENT_BUS,
            target_phase="phase1",
            mode=Phase8BridgeMode.READ_ONLY,
        )
    )

    assert result.status == Phase8IntegrationStatus.INTEGRATED
    assert result.reason == Phase8IntegrationReason.EVENT_BUS_BRIDGED
    assert result.packet is not None
    assert result.packet.payload["bus_contract"] == "EnvironmentEvents"


def test_phase2_voice_environment_fusion() -> None:
    runtime = Phase8FullIntegrationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.integrate(
        _request(
            session_id=session.session_id,
            boundary=Phase8IntegrationBoundary.PHASE2_VOICE_ENVIRONMENT_FUSION,
            target_phase="phase2",
            mode=Phase8BridgeMode.PREPARE_ONLY,
        )
    )

    assert result.status == Phase8IntegrationStatus.INTEGRATED
    assert result.reason == Phase8IntegrationReason.VOICE_ENVIRONMENT_FUSED
    assert result.packet is not None
    assert result.packet.payload["voice_safe"] is True


def test_phase3_cognition_receives_fused_context() -> None:
    runtime = Phase8FullIntegrationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.integrate(
        _request(
            session_id=session.session_id,
            boundary=Phase8IntegrationBoundary.PHASE3_COGNITION_FUSED_CONTEXT,
            target_phase="phase3",
            mode=Phase8BridgeMode.READ_ONLY,
        )
    )

    assert result.status == Phase8IntegrationStatus.INTEGRATED
    assert result.reason == Phase8IntegrationReason.COGNITION_CONTEXT_INJECTED
    assert result.packet is not None
    assert result.packet.payload["context_contract"] == "FusedContext"


def test_phase4_memory_uses_gateway() -> None:
    runtime = Phase8FullIntegrationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.integrate(
        _request(
            session_id=session.session_id,
            boundary=Phase8IntegrationBoundary.PHASE4_MEMORY_GATEWAY,
            target_phase="phase4",
            mode=Phase8BridgeMode.POLICY_GATED,
            gateway_required=True,
        )
    )

    assert result.status == Phase8IntegrationStatus.INTEGRATED
    assert result.reason == Phase8IntegrationReason.MEMORY_GATEWAY_BRIDGED
    assert result.packet is not None
    assert result.packet.gateway_required is True
    assert result.packet.payload["direct_memory_write"] is False


def test_phase4_memory_without_gateway_is_blocked() -> None:
    runtime = Phase8FullIntegrationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.integrate(
        _request(
            session_id=session.session_id,
            boundary=Phase8IntegrationBoundary.PHASE4_MEMORY_GATEWAY,
            target_phase="phase4",
            mode=Phase8BridgeMode.POLICY_GATED,
            gateway_required=False,
        )
    )

    assert result.status == Phase8IntegrationStatus.BLOCKED
    assert result.reason == Phase8IntegrationReason.POLICY_BYPASS_BLOCKED


def test_phase5_action_requires_policy() -> None:
    runtime = Phase8FullIntegrationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.integrate(
        _request(
            session_id=session.session_id,
            boundary=Phase8IntegrationBoundary.PHASE5_ACTION_POLICY,
            target_phase="phase5",
            mode=Phase8BridgeMode.POLICY_GATED,
            policy_required=True,
        )
    )

    assert result.status == Phase8IntegrationStatus.INTEGRATED
    assert result.reason == Phase8IntegrationReason.ACTION_POLICY_BRIDGED
    assert result.packet is not None
    assert result.packet.policy_required is True
    assert result.packet.payload["direct_execution"] is False


def test_phase5_direct_action_is_blocked() -> None:
    runtime = Phase8FullIntegrationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.integrate(
        _request(
            session_id=session.session_id,
            boundary=Phase8IntegrationBoundary.PHASE5_ACTION_POLICY,
            target_phase="phase5",
            mode=Phase8BridgeMode.POLICY_GATED,
            policy_required=True,
            payload={"direct_click": True},
        )
    )

    assert result.status == Phase8IntegrationStatus.BLOCKED
    assert result.reason == Phase8IntegrationReason.DIRECT_ACTION_BLOCKED


def test_phase6_visual_workers_scheduled_by_orchestration() -> None:
    runtime = Phase8FullIntegrationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.integrate(
        _request(
            session_id=session.session_id,
            boundary=Phase8IntegrationBoundary.PHASE6_ORCHESTRATION,
            target_phase="phase6",
            mode=Phase8BridgeMode.PREPARE_ONLY,
        )
    )

    assert result.status == Phase8IntegrationStatus.INTEGRATED
    assert result.reason == Phase8IntegrationReason.ORCHESTRATION_BRIDGED
    assert result.packet is not None
    assert result.packet.payload["scheduled_by_orchestration"] is True


def test_phase7_visual_context_streaming() -> None:
    runtime = Phase8FullIntegrationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.integrate(
        _request(
            session_id=session.session_id,
            boundary=Phase8IntegrationBoundary.PHASE7_VISUAL_STREAMING,
            target_phase="phase7",
            mode=Phase8BridgeMode.STREAMING,
        )
    )

    assert result.status == Phase8IntegrationStatus.INTEGRATED
    assert result.reason == Phase8IntegrationReason.VISUAL_STREAMING_BRIDGED
    assert result.packet is not None
    assert result.packet.mode == Phase8BridgeMode.STREAMING
    assert result.packet.payload["blocking_context_build"] is False


def test_phase8_awareness_does_not_replace_previous_phases() -> None:
    runtime = Phase8FullIntegrationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.integrate(
        _request(
            session_id=session.session_id,
            boundary=Phase8IntegrationBoundary.PHASE8_ENVIRONMENT_AWARENESS,
            target_phase="phase8",
            mode=Phase8BridgeMode.READ_ONLY,
        )
    )

    assert result.status == Phase8IntegrationStatus.INTEGRATED
    assert result.reason == Phase8IntegrationReason.ENVIRONMENT_AWARENESS_BRIDGED
    assert result.packet is not None
    assert result.packet.payload["replaces_previous_phases"] is False


def test_integrate_all_then_verify_full_integration() -> None:
    runtime = Phase8FullIntegrationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    results = runtime.integrate_all(
        session_id=session.session_id,
        workspace_id="workspace",
    )
    verified = runtime.verify_full_integration(session_id=session.session_id)

    assert len(results) == 8
    assert all(
        result.status == Phase8IntegrationStatus.INTEGRATED
        for result in results
    )
    assert verified.status == Phase8IntegrationStatus.VERIFIED
    assert verified.verification is not None
    assert verified.verification.verified is True
    assert verified.verification.missing_boundaries == ()


def test_verify_blocks_when_boundaries_missing() -> None:
    runtime = Phase8FullIntegrationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    verified = runtime.verify_full_integration(session_id=session.session_id)

    assert verified.status == Phase8IntegrationStatus.BLOCKED
    assert verified.verification is not None
    assert verified.verification.verified is False
    assert len(verified.verification.missing_boundaries) == 8


def test_missing_session_fails() -> None:
    runtime = Phase8FullIntegrationRuntime()

    result = runtime.integrate(
        _request(
            session_id="missing",
            boundary=Phase8IntegrationBoundary.PHASE1_EVENT_BUS,
            target_phase="phase1",
            mode=Phase8BridgeMode.READ_ONLY,
        )
    )

    assert result.status == Phase8IntegrationStatus.FAILED
    assert result.reason == Phase8IntegrationReason.SESSION_NOT_FOUND


def test_snapshot_tracks_counts() -> None:
    runtime = Phase8FullIntegrationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    runtime.integrate_all(
        session_id=session.session_id,
        workspace_id="workspace",
    )
    runtime.verify_full_integration(session_id=session.session_id)
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.integrated_count == 8
    assert snapshot.verified_count == 1
    assert snapshot.blocked_count == 0


def test_session_tracks_ready_flags() -> None:
    runtime = Phase8FullIntegrationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    runtime.integrate_all(
        session_id=session.session_id,
        workspace_id="workspace",
    )
    stored = runtime.session_for(session.session_id)

    assert stored is not None
    assert stored.phase1_event_bus_ready is True
    assert stored.phase2_voice_fusion_ready is True
    assert stored.phase3_cognition_context_ready is True
    assert stored.phase4_memory_gateway_ready is True
    assert stored.phase5_action_policy_ready is True
    assert stored.phase6_orchestration_ready is True
    assert stored.phase7_visual_streaming_ready is True
    assert stored.phase8_awareness_ready is True


def test_reset_clears_runtime() -> None:
    runtime = Phase8FullIntegrationRuntime()
    runtime.create_session(workspace_id="workspace")

    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.runtime_event_count == 1
    assert snapshot.last_reason == Phase8IntegrationReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert Phase8IntegrationBoundary.PHASE1_EVENT_BUS.value == "phase1_event_bus"
    assert Phase8IntegrationStatus.INTEGRATED.value == "integrated"
    assert Phase8IntegrationDecision.INTEGRATE.value == "integrate"


def _request(
    *,
    session_id: str,
    boundary: Phase8IntegrationBoundary,
    target_phase: str,
    mode: Phase8BridgeMode,
    policy_required: bool = False,
    gateway_required: bool = False,
    payload: dict[str, object] | None = None,
) -> Phase8IntegrationRequest:
    return Phase8IntegrationRequest(
        session_id=session_id,
        workspace_id="workspace",
        boundary=boundary,
        payload=payload or {"environment": "aware"},
        target_phase=target_phase,
        mode=mode,
        policy_required=policy_required,
        gateway_required=gateway_required,
    )