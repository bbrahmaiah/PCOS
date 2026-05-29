from __future__ import annotations

import pytest

from jarvis.environment import (
    CaptureMode,
    CapturePixelFormat,
    CaptureReason,
    CaptureStatus,
    EnvironmentAttentionMode,
    EnvironmentAttentionRuntime,
    EnvironmentAttentionSession,
    EnvironmentBackpressureController,
    FakeScreenCaptureAdapter,
    InspectionDepth,
    PrivacyZone,
    PrivacyZonePolicy,
    ScreenCaptureRequest,
    ScreenCaptureRuntime,
    ScreenRegion,
    VisualLoadLevel,
    VisualPriorityArbitrator,
    fake_display,
)


def test_runtime_rejects_empty_name() -> None:
    attention = EnvironmentAttentionRuntime()

    with pytest.raises(ValueError):
        ScreenCaptureRuntime(name=" ", attention_runtime=attention)


def test_payload_requires_positive_size() -> None:
    adapter = FakeScreenCaptureAdapter()

    payload = adapter.capture_region(ScreenRegion(x=0, y=0, width=10, height=20))

    assert payload.width == 10
    assert payload.height == 20
    assert payload.pixel_format == CapturePixelFormat.FAKE


def test_create_session() -> None:
    attention, attention_session = _attention()
    runtime = ScreenCaptureRuntime(attention_runtime=attention)

    session = runtime.create_session(
        attention_session_id=attention_session.session_id,
        workspace_id="workspace",
    )

    assert session.attention_session_id == attention_session.session_id
    assert runtime.snapshot().session_count == 1


def test_capture_region_allowed() -> None:
    attention, attention_session = _attention()
    runtime = ScreenCaptureRuntime(attention_runtime=attention)
    session = runtime.create_session(
        attention_session_id=attention_session.session_id,
        workspace_id="workspace",
    )

    result = runtime.capture_region(
        _request(
            session_id=session.session_id,
            depth=InspectionDepth.PERIPHERAL,
        )
    )

    assert result.success is True
    assert result.reason == CaptureReason.CAPTURED_REGION
    assert result.region_capture is not None
    assert result.region_capture.payload is not None


def test_capture_focused_window() -> None:
    attention, attention_session = _attention(mode=EnvironmentAttentionMode.FOCUSED)
    runtime = ScreenCaptureRuntime(attention_runtime=attention)
    session = runtime.create_session(
        attention_session_id=attention_session.session_id,
        workspace_id="workspace",
    )

    result = runtime.capture_focused_window(
        session_id=session.session_id,
        region=ScreenRegion(x=0, y=0, width=800, height=600),
        user_initiated=True,
    )

    assert result.success is True
    assert result.region_capture is not None
    assert result.region_capture.mode == CaptureMode.REGION


def test_capture_blocked_by_attention_depth() -> None:
    attention, attention_session = _attention()
    runtime = ScreenCaptureRuntime(attention_runtime=attention)
    session = runtime.create_session(
        attention_session_id=attention_session.session_id,
        workspace_id="workspace",
    )

    result = runtime.capture_region(
        _request(
            session_id=session.session_id,
            depth=InspectionDepth.FOCUSED,
            user_initiated=False,
        )
    )

    assert result.success is False
    assert result.status == CaptureStatus.BLOCKED
    assert result.reason == CaptureReason.CAPTURE_BLOCKED_BY_ATTENTION


def test_capture_deferred_by_conversation() -> None:
    attention, attention_session = _attention(mode=EnvironmentAttentionMode.FOCUSED)
    attention.update_backpressure(
        EnvironmentBackpressureController(conversation_active=True)
    )
    runtime = ScreenCaptureRuntime(attention_runtime=attention)
    session = runtime.create_session(
        attention_session_id=attention_session.session_id,
        workspace_id="workspace",
    )

    result = runtime.capture_region(
        _request(
            session_id=session.session_id,
            depth=InspectionDepth.FOCUSED,
            user_initiated=True,
        )
    )

    assert result.success is False
    assert result.status == CaptureStatus.DEFERRED
    assert result.reason == CaptureReason.CAPTURE_DEFERRED_BY_ATTENTION


def test_capture_blocked_by_privacy_zone() -> None:
    zone = PrivacyZone(
        name="secret region",
        region=ScreenRegion(x=0, y=0, width=200, height=200),
        capture_allowed=False,
        ocr_allowed=False,
        reason="private",
    )
    attention, attention_session = _attention(
        privacy_policy=PrivacyZonePolicy(zones=(zone,))
    )
    runtime = ScreenCaptureRuntime(attention_runtime=attention)
    session = runtime.create_session(
        attention_session_id=attention_session.session_id,
        workspace_id="workspace",
    )

    result = runtime.capture_region(
        _request(
            session_id=session.session_id,
            region=ScreenRegion(x=10, y=10, width=20, height=20),
            depth=InspectionDepth.PERIPHERAL,
        )
    )

    assert result.success is False
    assert result.reason == CaptureReason.CAPTURE_BLOCKED_BY_PRIVACY


def test_capture_limited_when_ocr_blocked_by_privacy() -> None:
    zone = PrivacyZone(
        name="private app",
        app_name="PrivateApp",
        capture_allowed=True,
        ocr_allowed=False,
        reason="OCR blocked",
    )
    attention, attention_session = _attention(
        privacy_policy=PrivacyZonePolicy(zones=(zone,))
    )
    runtime = ScreenCaptureRuntime(attention_runtime=attention)
    session = runtime.create_session(
        attention_session_id=attention_session.session_id,
        workspace_id="workspace",
    )

    result = runtime.capture_region(
        _request(
            session_id=session.session_id,
            app_name="PrivateApp",
            depth=InspectionDepth.FOCUSED,
            user_initiated=True,
        )
    )

    assert result.success is True
    assert result.status == CaptureStatus.LIMITED
    assert result.reason == CaptureReason.CAPTURE_LIMITED_BY_ATTENTION


def test_capture_deferred_by_priority_arbitrator() -> None:
    attention, attention_session = _attention(mode=EnvironmentAttentionMode.FOCUSED)
    priority = VisualPriorityArbitrator()
    priority.update_backpressure(
        EnvironmentBackpressureController(load_level=VisualLoadLevel.CRITICAL)
    )
    runtime = ScreenCaptureRuntime(
        attention_runtime=attention,
        priority_arbitrator=priority,
    )
    session = runtime.create_session(
        attention_session_id=attention_session.session_id,
        workspace_id="workspace",
    )

    result = runtime.capture_region(
        _request(
            session_id=session.session_id,
            depth=InspectionDepth.FOCUSED,
            user_initiated=True,
        )
    )

    assert result.success is False
    assert result.reason == CaptureReason.CAPTURE_DEFERRED_BY_PRIORITY


def test_delta_capture_records_changed_region() -> None:
    attention, attention_session = _attention()
    runtime = ScreenCaptureRuntime(attention_runtime=attention)
    session = runtime.create_session(
        attention_session_id=attention_session.session_id,
        workspace_id="workspace",
    )

    result = runtime.capture_delta(
        _request(
            session_id=session.session_id,
            depth=InspectionDepth.PERIPHERAL,
        )
    )

    assert result.success is True
    assert result.delta_capture is not None
    assert result.delta_capture.changed_regions


def test_multi_monitor_capture() -> None:
    attention, attention_session = _attention()
    runtime = ScreenCaptureRuntime(attention_runtime=attention)
    session = runtime.create_session(
        attention_session_id=attention_session.session_id,
        workspace_id="workspace",
    )

    result = runtime.capture_multi_monitor(
        session_id=session.session_id,
        displays=(fake_display(),),
    )

    assert result.success is True
    assert result.multi_monitor_capture is not None
    assert result.multi_monitor_capture.captures


def test_create_schedule_uses_attention_frequency() -> None:
    attention, attention_session = _attention()
    runtime = ScreenCaptureRuntime(attention_runtime=attention)
    session = runtime.create_session(
        attention_session_id=attention_session.session_id,
        workspace_id="workspace",
    )

    result = runtime.create_schedule(
        session_id=session.session_id,
        mode=CaptureMode.REGION,
        depth=InspectionDepth.PERIPHERAL,
    )

    assert result.success is True
    assert result.schedule is not None
    assert result.schedule.frequency_hz > 0


def test_missing_session_fails() -> None:
    attention, _ = _attention()
    runtime = ScreenCaptureRuntime(attention_runtime=attention)

    result = runtime.capture_region(
        _request(session_id="missing", depth=InspectionDepth.PERIPHERAL)
    )

    assert result.success is False
    assert result.reason == CaptureReason.SESSION_NOT_FOUND


def test_snapshot_tracks_counts() -> None:
    attention, attention_session = _attention()
    runtime = ScreenCaptureRuntime(attention_runtime=attention)
    session = runtime.create_session(
        attention_session_id=attention_session.session_id,
        workspace_id="workspace",
    )

    runtime.capture_region(
        _request(session_id=session.session_id, depth=InspectionDepth.PERIPHERAL)
    )
    runtime.capture_delta(
        _request(session_id=session.session_id, depth=InspectionDepth.PERIPHERAL)
    )
    runtime.create_schedule(
        session_id=session.session_id,
        mode=CaptureMode.REGION,
        depth=InspectionDepth.PERIPHERAL,
    )
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.region_capture_count >= 2
    assert snapshot.delta_capture_count == 1
    assert snapshot.schedule_count == 1


def test_reset_clears_runtime() -> None:
    attention, attention_session = _attention()
    runtime = ScreenCaptureRuntime(attention_runtime=attention)
    runtime.create_session(
        attention_session_id=attention_session.session_id,
        workspace_id="workspace",
    )

    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.runtime_event_count == 1
    assert snapshot.last_reason == CaptureReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert CaptureMode.REGION.value == "region"
    assert CaptureStatus.CAPTURED.value == "captured"
    assert CaptureReason.CAPTURED_REGION.value == "captured_region"
    assert CapturePixelFormat.FAKE.value == "fake"


def _attention(
    *,
    mode: EnvironmentAttentionMode = EnvironmentAttentionMode.PERIPHERAL,
    privacy_policy: PrivacyZonePolicy | None = None,
) -> tuple[EnvironmentAttentionRuntime, EnvironmentAttentionSession]:
    runtime = EnvironmentAttentionRuntime(privacy_policy=privacy_policy)
    session = runtime.create_session(workspace_id="workspace", mode=mode)

    return runtime, session


def _request(
    *,
    session_id: str,
    depth: InspectionDepth,
    region: ScreenRegion | None = None,
    app_name: str | None = None,
    user_initiated: bool = False,
) -> ScreenCaptureRequest:
    return ScreenCaptureRequest(
        session_id=session_id,
        mode=CaptureMode.REGION,
        region=region or ScreenRegion(x=0, y=0, width=100, height=100),
        depth=depth,
        reason="test screen capture",
        app_name=app_name,
        user_initiated=user_initiated,
    )