from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    CollaborationPhase,
    HumanSpeedActionKind,
    HumanSpeedActionRequest,
    HumanSpeedDecision,
    HumanSpeedInteractionRuntime,
    HumanSpeedReason,
    HumanSpeedStatus,
    HumanTimingModel,
    MotionCurveKind,
    MotionCurvePolicy,
    NarrationTone,
    NaturalnessLevel,
    ProgressNarration,
    ScreenPoint,
    SpeechActionSyncKind,
)


def test_runtime_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        HumanSpeedInteractionRuntime(name=" ")


def test_motion_curve_policy_rejects_bad_duration_order() -> None:
    with pytest.raises(ValidationError):
        MotionCurvePolicy(min_duration_ms=1000, max_duration_ms=100)


def test_timing_plan_rejects_invalid_total() -> None:
    from jarvis.environment import HumanTimingPlan

    model = HumanTimingModel()

    with pytest.raises(ValidationError):
        HumanTimingPlan(
            action_kind=HumanSpeedActionKind.WAIT,
            pre_action_delay_ms=1,
            action_duration_ms=1,
            post_action_delay_ms=1,
            total_duration_ms=99,
            reason="invalid",
            model=model,
        )


def test_create_session() -> None:
    runtime = HumanSpeedInteractionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    assert session.workspace_id == "workspace"
    assert runtime.snapshot().session_count == 1


def test_mouse_move_creates_human_curve() -> None:
    runtime = HumanSpeedInteractionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.optimize(
        HumanSpeedActionRequest(
            session_id=session.session_id,
            action_kind=HumanSpeedActionKind.MOUSE_MOVE,
            description="move mouse to button",
            start_point=ScreenPoint(x=10, y=10),
            target_point=ScreenPoint(x=300, y=220),
        )
    )

    assert result.status in {
        HumanSpeedStatus.OPTIMIZED,
        HumanSpeedStatus.NATURAL,
    }
    assert result.pacing_plan is not None
    assert result.pacing_plan.motion_curve is not None
    assert result.pacing_plan.motion_curve.curve_kind == MotionCurveKind.HUMAN_ARC
    assert len(result.pacing_plan.motion_curve.points) >= 2


def test_keyboard_typing_gets_human_duration() -> None:
    runtime = HumanSpeedInteractionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.optimize(
        HumanSpeedActionRequest(
            session_id=session.session_id,
            action_kind=HumanSpeedActionKind.KEYBOARD_TYPE,
            description="type into known field",
            text="hello jarvis",
        )
    )

    assert result.pacing_plan is not None
    assert result.pacing_plan.timing.action_duration_ms > 0
    assert result.pacing_plan.timing.total_duration_ms >= (
        result.pacing_plan.timing.action_duration_ms
    )


def test_speech_action_sync_waits_before_click() -> None:
    runtime = HumanSpeedInteractionRuntime()
    session = runtime.create_session(workspace_id="workspace")
    narration = ProgressNarration(
        phase=CollaborationPhase.APPLYING_CHANGE,
        text="I need your approval before clicking.",
        tone=NarrationTone.CALM,
    )

    result = runtime.optimize(
        HumanSpeedActionRequest(
            session_id=session.session_id,
            action_kind=HumanSpeedActionKind.MOUSE_CLICK,
            description="click submit",
            narration=narration,
        )
    )

    assert result.pacing_plan is not None
    assert result.pacing_plan.speech_sync is not None
    assert result.pacing_plan.speech_sync.kind == (
        SpeechActionSyncKind.SPEAK_BEFORE_ACTION
    )
    assert result.pacing_plan.speech_sync.action_should_wait_for_speech is True


def test_speech_during_verify_wait() -> None:
    runtime = HumanSpeedInteractionRuntime()
    session = runtime.create_session(workspace_id="workspace")
    narration = ProgressNarration(
        phase=CollaborationPhase.VERIFYING,
        text="I'm verifying the result now.",
        tone=NarrationTone.CALM,
    )

    result = runtime.optimize(
        HumanSpeedActionRequest(
            session_id=session.session_id,
            action_kind=HumanSpeedActionKind.VERIFY,
            description="verify state",
            narration=narration,
        )
    )

    assert result.pacing_plan is not None
    assert result.pacing_plan.speech_sync is not None
    assert result.pacing_plan.speech_sync.kind == (
        SpeechActionSyncKind.SPEAK_DURING_WAIT
    )


def test_too_robotic_timing_is_blocked_by_strict_model() -> None:
    runtime = HumanSpeedInteractionRuntime(
        timing_model=HumanTimingModel(
            min_action_gap_ms=0,
            min_pre_action_notice_ms=0,
            min_post_action_verify_ms=0,
            max_robotic_speed_score=0.10,
        )
    )
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.optimize(
        HumanSpeedActionRequest(
            session_id=session.session_id,
            action_kind=HumanSpeedActionKind.MOUSE_CLICK,
            description="instant click",
            requires_user_visibility=False,
        )
    )

    assert result.status == HumanSpeedStatus.BLOCKED
    assert result.decision == HumanSpeedDecision.SLOW_DOWN
    assert result.reason == HumanSpeedReason.ROBOTIC_PACING_BLOCKED


def test_naturalness_accepts_normal_timing() -> None:
    runtime = HumanSpeedInteractionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.optimize(
        HumanSpeedActionRequest(
            session_id=session.session_id,
            action_kind=HumanSpeedActionKind.KEYBOARD_TYPE,
            description="type visible text",
            text="This is a normal paced typing action.",
        )
    )

    assert result.evaluation is not None
    assert result.evaluation.accepted is True
    assert result.evaluation.level in {
        NaturalnessLevel.ACCEPTABLE,
        NaturalnessLevel.NATURAL,
        NaturalnessLevel.EXCELLENT,
    }


def test_missing_session_fails() -> None:
    runtime = HumanSpeedInteractionRuntime()

    result = runtime.optimize(
        HumanSpeedActionRequest(
            session_id="missing",
            action_kind=HumanSpeedActionKind.WAIT,
            description="wait",
        )
    )

    assert result.status == HumanSpeedStatus.FAILED
    assert result.reason == HumanSpeedReason.SESSION_NOT_FOUND


def test_snapshot_tracks_counts() -> None:
    runtime = HumanSpeedInteractionRuntime()
    session = runtime.create_session(workspace_id="workspace")
    narration = ProgressNarration(
        phase=CollaborationPhase.RUNNING_TESTS,
        text="The tests are running.",
        tone=NarrationTone.CALM,
    )

    runtime.optimize(
        HumanSpeedActionRequest(
            session_id=session.session_id,
            action_kind=HumanSpeedActionKind.MOUSE_MOVE,
            description="move",
            start_point=ScreenPoint(x=0, y=0),
            target_point=ScreenPoint(x=200, y=120),
        )
    )
    runtime.optimize(
        HumanSpeedActionRequest(
            session_id=session.session_id,
            action_kind=HumanSpeedActionKind.VERIFY,
            description="verify",
            narration=narration,
        )
    )

    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.result_count == 2
    assert snapshot.optimized_count == 2
    assert snapshot.motion_plan_count == 1
    assert snapshot.speech_sync_count == 1


def test_session_tracks_counts() -> None:
    runtime = HumanSpeedInteractionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    runtime.optimize(
        HumanSpeedActionRequest(
            session_id=session.session_id,
            action_kind=HumanSpeedActionKind.MOUSE_MOVE,
            description="move",
            start_point=ScreenPoint(x=0, y=0),
            target_point=ScreenPoint(x=120, y=90),
        )
    )

    stored = runtime.session_for(session.session_id)

    assert stored is not None
    assert stored.optimized_count == 1
    assert stored.motion_plan_count == 1


def test_reset_clears_runtime() -> None:
    runtime = HumanSpeedInteractionRuntime()
    runtime.create_session(workspace_id="workspace")

    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.runtime_event_count == 1
    assert snapshot.last_reason == HumanSpeedReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert HumanSpeedActionKind.MOUSE_MOVE.value == "mouse_move"
    assert HumanSpeedStatus.NATURAL.value == "natural"
    assert SpeechActionSyncKind.SPEAK_BEFORE_ACTION.value == "speak_before_action"