from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    AttentionGovernanceReason,
    CaptureFrequencyPolicy,
    CaptureGovernanceRequest,
    CapturePermission,
    EnvironmentAttentionMode,
    EnvironmentAttentionRuntime,
    EnvironmentBackpressureController,
    FocusRegionKind,
    InspectionDepth,
    PeripheralAwareness,
    PrivacyZone,
    PrivacyZoneDecision,
    PrivacyZonePolicy,
    ScreenRegion,
    VisualAttentionPolicy,
    VisualLoadLevel,
    depth_rank,
    fake_focus_region,
    min_depth,
)


def test_runtime_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        EnvironmentAttentionRuntime(name=" ")


def test_visual_policy_rejects_deep_without_permission() -> None:
    with pytest.raises(ValidationError):
        VisualAttentionPolicy(
            max_depth=InspectionDepth.DEEP,
            allow_deep_inspection=False,
        )


def test_frequency_policy_requires_increasing_depth() -> None:
    with pytest.raises(ValidationError):
        CaptureFrequencyPolicy(
            peripheral_hz=5,
            ambient_hz=1,
            focused_hz=2,
            deep_hz=3,
            max_burst_hz=10,
        )


def test_create_session() -> None:
    runtime = EnvironmentAttentionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    assert session.workspace_id == "workspace"
    assert session.mode == EnvironmentAttentionMode.PERIPHERAL
    assert runtime.snapshot().session_count == 1


def test_add_and_remove_focus_region() -> None:
    runtime = EnvironmentAttentionRuntime()
    session = runtime.create_session(workspace_id="workspace")
    region = fake_focus_region()

    added = runtime.add_focus_region(
        session_id=session.session_id,
        focus_region=region,
    )
    removed = runtime.remove_focus_region(
        session_id=session.session_id,
        focus_id=region.focus_id,
    )

    assert added.success is True
    assert removed.success is True
    assert runtime.active_focus_regions(session.session_id) == ()


def test_peripheral_awareness_detects_important_change() -> None:
    awareness = PeripheralAwareness(app_crashed=True)

    assert awareness.has_important_change() is True


def test_update_peripheral_awareness() -> None:
    runtime = EnvironmentAttentionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.update_peripheral_awareness(
        session_id=session.session_id,
        awareness=PeripheralAwareness(modal_present=True),
    )
    stored = runtime.peripheral_awareness(session.session_id)

    assert result.success is True
    assert stored is not None
    assert stored.modal_present is True


def test_deep_attention_requires_user_intent() -> None:
    runtime = EnvironmentAttentionRuntime(
        visual_policy=VisualAttentionPolicy(
            max_depth=InspectionDepth.DEEP,
            allow_deep_inspection=True,
            require_user_intent_for_deep=True,
        )
    )
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.set_attention_mode(
        session_id=session.session_id,
        mode=EnvironmentAttentionMode.DEEP,
        user_initiated=False,
    )

    assert result.success is False
    assert result.reason == AttentionGovernanceReason.CAPTURE_BLOCKED_BY_DEPTH


def test_deep_attention_allowed_with_user_intent() -> None:
    runtime = EnvironmentAttentionRuntime(
        visual_policy=VisualAttentionPolicy(
            max_depth=InspectionDepth.DEEP,
            allow_deep_inspection=True,
            require_user_intent_for_deep=True,
        )
    )
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.set_attention_mode(
        session_id=session.session_id,
        mode=EnvironmentAttentionMode.DEEP,
        user_initiated=True,
    )

    assert result.success is True
    assert result.session is not None
    assert result.session.mode == EnvironmentAttentionMode.DEEP


def test_capture_allowed_for_peripheral_request() -> None:
    runtime = EnvironmentAttentionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    decision = runtime.decide_capture(
        _request(
            session_id=session.session_id,
            depth=InspectionDepth.PERIPHERAL,
        )
    )

    assert decision.permission == CapturePermission.ALLOW
    assert decision.reason == AttentionGovernanceReason.CAPTURE_ALLOWED
    assert decision.frequency_hz > 0


def test_focused_capture_requires_focused_mode_or_user_intent() -> None:
    runtime = EnvironmentAttentionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    decision = runtime.decide_capture(
        _request(
            session_id=session.session_id,
            depth=InspectionDepth.FOCUSED,
            user_initiated=False,
        )
    )

    assert decision.permission == CapturePermission.BLOCK
    assert decision.reason == AttentionGovernanceReason.CAPTURE_BLOCKED_BY_DEPTH


def test_focused_capture_allowed_with_user_intent() -> None:
    runtime = EnvironmentAttentionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    decision = runtime.decide_capture(
        _request(
            session_id=session.session_id,
            depth=InspectionDepth.FOCUSED,
            user_initiated=True,
        )
    )

    assert decision.permission == CapturePermission.ALLOW


def test_conversation_defers_focused_capture() -> None:
    runtime = EnvironmentAttentionRuntime()
    session = runtime.create_session(
        workspace_id="workspace",
        mode=EnvironmentAttentionMode.FOCUSED,
    )
    runtime.update_backpressure(
        EnvironmentBackpressureController(conversation_active=True)
    )

    decision = runtime.decide_capture(
        _request(
            session_id=session.session_id,
            depth=InspectionDepth.FOCUSED,
            user_initiated=True,
        )
    )

    assert decision.permission == CapturePermission.DEFER
    assert decision.reason == AttentionGovernanceReason.CAPTURE_DEFERRED


def test_interruption_defers_capture() -> None:
    runtime = EnvironmentAttentionRuntime()
    session = runtime.create_session(workspace_id="workspace")
    runtime.update_backpressure(
        EnvironmentBackpressureController(interruption_active=True)
    )

    decision = runtime.decide_capture(
        _request(
            session_id=session.session_id,
            depth=InspectionDepth.PERIPHERAL,
        )
    )

    assert decision.permission == CapturePermission.DEFER
    assert decision.reason == AttentionGovernanceReason.CAPTURE_BLOCKED_BY_LOAD


def test_critical_load_defers_capture() -> None:
    runtime = EnvironmentAttentionRuntime()
    session = runtime.create_session(workspace_id="workspace")
    runtime.update_backpressure(
        EnvironmentBackpressureController(load_level=VisualLoadLevel.CRITICAL)
    )

    decision = runtime.decide_capture(
        _request(
            session_id=session.session_id,
            depth=InspectionDepth.PERIPHERAL,
        )
    )

    assert decision.permission == CapturePermission.DEFER


def test_privacy_zone_blocks_capture() -> None:
    zone = PrivacyZone(
        name="password manager",
        region=ScreenRegion(x=0, y=0, width=100, height=100),
        capture_allowed=False,
        ocr_allowed=False,
        reason="secret region",
    )
    runtime = EnvironmentAttentionRuntime(
        privacy_policy=PrivacyZonePolicy(zones=(zone,))
    )
    session = runtime.create_session(workspace_id="workspace")

    decision = runtime.decide_capture(
        _request(
            session_id=session.session_id,
            region=ScreenRegion(x=10, y=10, width=20, height=20),
            depth=InspectionDepth.PERIPHERAL,
        )
    )

    assert decision.permission == CapturePermission.BLOCK
    assert decision.privacy_decision == PrivacyZoneDecision.CAPTURE_BLOCKED


def test_privacy_zone_allows_capture_but_blocks_ocr() -> None:
    zone = PrivacyZone(
        name="private app",
        app_name="PrivateApp",
        capture_allowed=True,
        ocr_allowed=False,
        reason="capture allowed but OCR blocked",
    )
    runtime = EnvironmentAttentionRuntime(
        privacy_policy=PrivacyZonePolicy(zones=(zone,))
    )
    session = runtime.create_session(workspace_id="workspace")

    decision = runtime.decide_capture(
        _request(
            session_id=session.session_id,
            app_name="PrivateApp",
            depth=InspectionDepth.FOCUSED,
            user_initiated=True,
        )
    )

    assert decision.permission == CapturePermission.ALLOW_LIMITED
    assert decision.allowed_depth == InspectionDepth.PERIPHERAL
    assert decision.privacy_decision == PrivacyZoneDecision.OCR_BLOCKED


def test_redaction_limits_depth() -> None:
    region = ScreenRegion(x=0, y=0, width=100, height=100)
    zone = PrivacyZone(
        name="redacted region",
        region=region,
        capture_allowed=True,
        ocr_allowed=True,
        reason="redaction test",
    )
    policy = PrivacyZonePolicy(zones=(zone,))
    runtime = EnvironmentAttentionRuntime(privacy_policy=policy)
    session = runtime.create_session(
        workspace_id="workspace",
        mode=EnvironmentAttentionMode.FOCUSED,
    )

    decision = runtime.decide_capture(
        _request(
            session_id=session.session_id,
            region=region,
            depth=InspectionDepth.FOCUSED,
            user_initiated=True,
        )
    )

    assert decision.permission == CapturePermission.ALLOW


def test_snapshot_tracks_decisions() -> None:
    runtime = EnvironmentAttentionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    runtime.add_focus_region(
        session_id=session.session_id,
        focus_region=fake_focus_region(kind=FocusRegionKind.ACTIVE_WINDOW),
    )
    runtime.decide_capture(
        _request(
            session_id=session.session_id,
            depth=InspectionDepth.PERIPHERAL,
        )
    )
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.focus_region_count == 1
    assert snapshot.active_focus_region_count == 1
    assert snapshot.decision_count == 1
    assert snapshot.allowed_count == 1


def test_reset_clears_runtime() -> None:
    runtime = EnvironmentAttentionRuntime()
    runtime.create_session(workspace_id="workspace")

    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.runtime_event_count == 1
    assert snapshot.last_reason == AttentionGovernanceReason.RUNTIME_RESET


def test_depth_helpers() -> None:
    assert depth_rank(InspectionDepth.DEEP) > depth_rank(InspectionDepth.FOCUSED)
    assert min_depth(InspectionDepth.DEEP, InspectionDepth.AMBIENT) == (
        InspectionDepth.AMBIENT
    )


def test_enum_values_are_stable() -> None:
    assert InspectionDepth.PERIPHERAL.value == "peripheral"
    assert EnvironmentAttentionMode.DEEP.value == "deep"
    assert CapturePermission.ALLOW_LIMITED.value == "allow_limited"


def _request(
    *,
    session_id: str,
    depth: InspectionDepth,
    region: ScreenRegion | None = None,
    app_name: str | None = None,
    user_initiated: bool = False,
) -> CaptureGovernanceRequest:
    return CaptureGovernanceRequest(
        session_id=session_id,
        region=region or ScreenRegion(x=0, y=0, width=100, height=100),
        requested_depth=depth,
        reason="test capture governance",
        app_name=app_name,
        user_initiated=user_initiated,
    )