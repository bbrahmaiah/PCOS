from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    EnvironmentActionPlanningRequest,
    EnvironmentActionPlanningRuntime,
    HumanOverrideKind,
    HumanTimingPolicy,
    InteractionPolicyResult,
    KeyboardActionRequest,
    KeyboardShortcutKind,
    MouseActionRequest,
    MouseButton,
    NaturalMotionEngine,
    PhysicalInputReason,
    PhysicalInputRuntime,
    PhysicalInputStatus,
    PhysicalInteractionRequest,
    PhysicalOverrideSignal,
    ScreenRegion,
    SimulatedActionKind,
    contract_from_plan,
)
from jarvis.environment.interaction_policy import InteractionPolicyRuntime


def test_runtime_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        PhysicalInputRuntime(name=" ")


def test_timing_policy_rejects_bad_ranges() -> None:
    with pytest.raises(ValidationError):
        HumanTimingPolicy(min_move_duration_ms=500, max_move_duration_ms=100)

    with pytest.raises(ValidationError):
        HumanTimingPolicy(min_key_interval_ms=200, max_key_interval_ms=20)


def test_keyboard_request_requires_input() -> None:
    policy = _policy_result(
        action=SimulatedActionKind.TYPE_TEXT,
        user_intent="type hello",
        text_payload="hello",
    )

    with pytest.raises(ValidationError):
        KeyboardActionRequest(
            session_id="session",
            policy_result=policy,
        )


def test_create_session() -> None:
    runtime = PhysicalInputRuntime()
    session = runtime.create_session(workspace_id="workspace")

    assert session.workspace_id == "workspace"
    assert runtime.snapshot().session_count == 1


def test_natural_motion_engine_never_teleports() -> None:
    engine = NaturalMotionEngine()
    region = ScreenRegion(x=100, y=100, width=40, height=20)

    plan = engine.plan(start_x=0, start_y=0, target=region)

    assert plan.points
    assert plan.points[-1].x == 120
    assert plan.points[-1].y == 110
    assert plan.total_duration_ms >= 1
    assert plan.human_speed is True


def test_mouse_click_requires_policy_eligibility() -> None:
    runtime = PhysicalInputRuntime()
    session = runtime.create_session(workspace_id="workspace")
    policy = _policy_result(
        action=SimulatedActionKind.DELETE,
        user_intent="delete old log",
    )

    result = runtime.execute_mouse(
        MouseActionRequest(
            session_id=session.session_id,
            policy_result=policy,
            target_region=ScreenRegion(x=10, y=10, width=20, height=20),
        )
    )

    assert result.status == PhysicalInputStatus.BLOCKED
    assert result.reason == PhysicalInputReason.POLICY_NOT_ELIGIBLE
    assert result.executed is False


def test_mouse_click_requires_preclick_target_region() -> None:
    runtime = PhysicalInputRuntime()
    session = runtime.create_session(workspace_id="workspace")
    policy = _policy_result(
        action=SimulatedActionKind.FOCUS,
        user_intent="focus terminal",
        target_label="Terminal",
    )

    result = runtime.execute_mouse(
        MouseActionRequest(
            session_id=session.session_id,
            policy_result=policy,
            target_region=None,
        )
    )

    assert result.status == PhysicalInputStatus.BLOCKED
    assert result.reason == PhysicalInputReason.PRECLICK_FAILED
    assert result.executed is False


def test_mouse_click_executes_with_preclick_and_motion_plan() -> None:
    runtime = PhysicalInputRuntime()
    session = runtime.create_session(workspace_id="workspace")
    policy = _policy_result(
        action=SimulatedActionKind.FOCUS,
        user_intent="focus terminal",
        target_label="Terminal",
    )

    result = runtime.execute_mouse(
        MouseActionRequest(
            session_id=session.session_id,
            policy_result=policy,
            target_region=ScreenRegion(x=100, y=50, width=80, height=30),
            button=MouseButton.LEFT,
            current_x=0,
            current_y=0,
        )
    )

    assert result.status == PhysicalInputStatus.EXECUTED
    assert result.executed is True
    assert result.motion_plan is not None
    assert result.preclick is not None
    assert result.preclick.passed is True


def test_double_click_uses_double_click_kind() -> None:
    runtime = PhysicalInputRuntime()
    session = runtime.create_session(workspace_id="workspace")
    policy = _policy_result(
        action=SimulatedActionKind.FOCUS,
        user_intent="focus terminal",
        target_label="Terminal",
    )

    result = runtime.execute_mouse(
        MouseActionRequest(
            session_id=session.session_id,
            policy_result=policy,
            target_region=ScreenRegion(x=100, y=50, width=80, height=30),
            double_click=True,
        )
    )

    assert result.executed is True
    assert result.input_kind.value == "mouse_double_click"


def test_keyboard_typing_blocks_unknown_field() -> None:
    runtime = PhysicalInputRuntime()
    session = runtime.create_session(workspace_id="workspace")
    policy = _policy_result(
        action=SimulatedActionKind.TYPE_TEXT,
        user_intent="type hello",
        text_payload="hello",
        target_label="Search",
    )

    result = runtime.execute_keyboard(
        KeyboardActionRequest(
            session_id=session.session_id,
            policy_result=policy,
            text="hello",
            focus_known=True,
            target_field_known=False,
        )
    )

    assert result.status == PhysicalInputStatus.BLOCKED
    assert result.reason == PhysicalInputReason.UNKNOWN_FIELD
    assert result.executed is False


def test_keyboard_typing_blocks_uncertain_focus() -> None:
    runtime = PhysicalInputRuntime()
    session = runtime.create_session(workspace_id="workspace")
    policy = _policy_result(
        action=SimulatedActionKind.TYPE_TEXT,
        user_intent="type hello",
        text_payload="hello",
        target_label="Search",
    )

    result = runtime.execute_keyboard(
        KeyboardActionRequest(
            session_id=session.session_id,
            policy_result=policy,
            text="hello",
            focus_known=False,
            target_field_known=True,
        )
    )

    assert result.status == PhysicalInputStatus.BLOCKED
    assert result.reason == PhysicalInputReason.FOCUS_UNCERTAIN
    assert result.executed is False


def test_keyboard_typing_executes_when_focus_and_field_known() -> None:
    runtime = PhysicalInputRuntime()
    session = runtime.create_session(workspace_id="workspace")
    policy = _policy_result(
        action=SimulatedActionKind.TYPE_TEXT,
        user_intent="type hello",
        text_payload="hello",
        target_label="Search",
    )

    result = runtime.execute_keyboard(
        KeyboardActionRequest(
            session_id=session.session_id,
            policy_result=policy,
            text="hello",
            focus_known=True,
            target_field_known=True,
        )
    )

    assert result.status == PhysicalInputStatus.EXECUTED
    assert result.executed is True


def test_keyboard_shortcut_can_execute_without_field() -> None:
    runtime = PhysicalInputRuntime()
    session = runtime.create_session(workspace_id="workspace")
    policy = _policy_result(
        action=SimulatedActionKind.FOCUS,
        user_intent="focus terminal",
        target_label="Terminal",
    )

    result = runtime.execute_keyboard(
        KeyboardActionRequest(
            session_id=session.session_id,
            policy_result=policy,
            shortcut=KeyboardShortcutKind.ESCAPE,
        )
    )

    assert result.status == PhysicalInputStatus.EXECUTED
    assert result.executed is True
    assert result.input_kind.value == "keyboard_shortcut"


def test_user_mouse_override_pauses_jarvis() -> None:
    runtime = PhysicalInputRuntime()
    session = runtime.create_session(workspace_id="workspace")
    policy = _policy_result(
        action=SimulatedActionKind.FOCUS,
        user_intent="focus terminal",
        target_label="Terminal",
    )
    runtime.update_override(
        PhysicalOverrideSignal(
            kind=HumanOverrideKind.USER_MOUSE_MOVED,
            active=True,
            reason="user moved mouse",
        )
    )

    result = runtime.execute_mouse(
        MouseActionRequest(
            session_id=session.session_id,
            policy_result=policy,
            target_region=ScreenRegion(x=100, y=50, width=80, height=30),
        )
    )

    assert result.status == PhysicalInputStatus.PAUSED_BY_USER
    assert result.reason == PhysicalInputReason.HUMAN_OVERRIDE_DETECTED
    stored_session = runtime.session_for(session.session_id)

    assert stored_session is not None
    assert stored_session.paused_by_user is True


def test_stop_command_cancels_immediately() -> None:
    runtime = PhysicalInputRuntime()
    session = runtime.create_session(workspace_id="workspace")
    policy = _policy_result(
        action=SimulatedActionKind.TYPE_TEXT,
        user_intent="type hello",
        text_payload="hello",
        target_label="Search",
    )
    runtime.update_override(
        PhysicalOverrideSignal(
            kind=HumanOverrideKind.USER_STOP_COMMAND,
            active=True,
            reason="user said stop",
        )
    )

    result = runtime.execute_keyboard(
        KeyboardActionRequest(
            session_id=session.session_id,
            policy_result=policy,
            text="hello",
            focus_known=True,
            target_field_known=True,
        )
    )

    assert result.status == PhysicalInputStatus.CANCELLED
    assert result.reason == PhysicalInputReason.STOP_CANCELLED
    stored_session = runtime.session_for(session.session_id)

    assert stored_session is not None
    assert stored_session.cancelled is True


def test_missing_session_blocks() -> None:
    runtime = PhysicalInputRuntime()
    policy = _policy_result(
        action=SimulatedActionKind.FOCUS,
        user_intent="focus terminal",
        target_label="Terminal",
    )

    result = runtime.execute_mouse(
        MouseActionRequest(
            session_id="missing",
            policy_result=policy,
            target_region=ScreenRegion(x=10, y=10, width=20, height=20),
        )
    )

    assert result.status == PhysicalInputStatus.BLOCKED
    assert result.reason == PhysicalInputReason.SESSION_NOT_FOUND


def test_snapshot_tracks_counts() -> None:
    runtime = PhysicalInputRuntime()
    session = runtime.create_session(workspace_id="workspace")
    policy = _policy_result(
        action=SimulatedActionKind.FOCUS,
        user_intent="focus terminal",
        target_label="Terminal",
    )

    runtime.execute_mouse(
        MouseActionRequest(
            session_id=session.session_id,
            policy_result=policy,
            target_region=ScreenRegion(x=10, y=10, width=20, height=20),
        )
    )
    runtime.update_override(
        PhysicalOverrideSignal(
            kind=HumanOverrideKind.USER_MOUSE_MOVED,
            active=True,
            reason="user moved mouse",
        )
    )
    runtime.execute_mouse(
        MouseActionRequest(
            session_id=session.session_id,
            policy_result=policy,
            target_region=ScreenRegion(x=10, y=10, width=20, height=20),
        )
    )

    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.result_count == 2
    assert snapshot.executed_count == 1
    assert snapshot.paused_by_user_count == 1
    assert snapshot.motion_plan_count == 1


def test_reset_clears_runtime() -> None:
    runtime = PhysicalInputRuntime()
    runtime.create_session(workspace_id="workspace")

    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.runtime_event_count == 1
    assert snapshot.last_reason == PhysicalInputReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert PhysicalInputStatus.EXECUTED.value == "executed"
    assert PhysicalInputReason.STOP_CANCELLED.value == "stop_cancelled"
    assert HumanOverrideKind.USER_MOUSE_MOVED.value == "user_mouse_moved"


def _policy_result(
    *,
    action: SimulatedActionKind,
    user_intent: str,
    text_payload: str | None = None,
    target_label: str | None = None,
) -> InteractionPolicyResult:
    planning = EnvironmentActionPlanningRuntime()
    plan_session = planning.create_session(workspace_id="workspace")
    plan = planning.plan(
        EnvironmentActionPlanningRequest(
            session_id=plan_session.session_id,
            workspace_id="workspace",
            user_intent=user_intent,
            proposed_action_kind=action,
            text_payload=text_payload,
        )
    )

    if target_label is not None:
        plan = plan.model_copy(
            update={
                "candidate": plan.candidate.model_copy(
                    update={
                        "action": plan.candidate.action.model_copy(
                            update={"target_label": target_label}
                        )
                    }
                )
            }
        )

    policy = InteractionPolicyRuntime()
    policy_session = policy.create_session(workspace_id="workspace")
    contract = contract_from_plan(plan)
    if target_label is not None:
        contract = contract.model_copy(update={"target_label": target_label})

    return policy.evaluate(
        PhysicalInteractionRequest(
            session_id=policy_session.session_id,
            workspace_id="workspace",
            contract=contract,
            plan=plan,
        )
    )