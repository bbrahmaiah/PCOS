from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    AppCloseSafety,
    AppControlActionKind,
    AppControlDecision,
    AppControlReason,
    AppControlRequest,
    AppControlRuntime,
    AppControlStatus,
    AppControlTarget,
    AppIdentityResult,
    AppIdentityStatus,
    AppSessionRestoreStatus,
    AppVisibilityState,
    DetectedAppKind,
    EnvironmentActionPlanningRequest,
    EnvironmentActionPlanningRuntime,
    PhysicalInteractionRequest,
    SimulatedActionKind,
    contract_from_plan,
)
from jarvis.environment.interaction_policy import (
    InteractionPolicyResult,
    InteractionPolicyRuntime,
)


def test_runtime_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        AppControlRuntime(name=" ")


def test_target_rejects_empty_app_name() -> None:
    with pytest.raises(ValidationError):
        AppControlTarget(app_name=" ")


def test_create_session() -> None:
    runtime = AppControlRuntime()
    session = runtime.create_session(workspace_id="workspace")

    assert session.workspace_id == "workspace"
    assert runtime.snapshot().session_count == 1


def test_launch_requires_policy_eligibility() -> None:
    runtime = AppControlRuntime()
    session = runtime.create_session(workspace_id="workspace")
    policy = _policy(
        action=SimulatedActionKind.DELETE,
        user_intent="delete file",
    )

    result = runtime.control(
        AppControlRequest(
            session_id=session.session_id,
            action=AppControlActionKind.LAUNCH,
            target=_target(),
            policy_result=policy,
            identity=_identity(),
        )
    )

    assert result.status == AppControlStatus.BLOCKED
    assert result.reason == AppControlReason.POLICY_NOT_ELIGIBLE
    assert result.control_eligible is False


def test_launch_requires_identity() -> None:
    runtime = AppControlRuntime()
    session = runtime.create_session(workspace_id="workspace")
    policy = _policy(
        action=SimulatedActionKind.OPEN,
        user_intent="open vscode",
    )

    result = runtime.control(
        AppControlRequest(
            session_id=session.session_id,
            action=AppControlActionKind.LAUNCH,
            target=_target(),
            policy_result=policy,
            identity=None,
        )
    )

    assert result.status == AppControlStatus.BLOCKED
    assert result.reason == AppControlReason.APP_IDENTITY_UNKNOWN
    assert result.control_eligible is False


def test_launch_blocks_unknown_identity() -> None:
    runtime = AppControlRuntime()
    session = runtime.create_session(workspace_id="workspace")
    policy = _policy(
        action=SimulatedActionKind.OPEN,
        user_intent="open app",
    )

    result = runtime.control(
        AppControlRequest(
            session_id=session.session_id,
            action=AppControlActionKind.LAUNCH,
            target=_target(app_kind=DetectedAppKind.UNKNOWN),
            policy_result=policy,
            identity=_identity(status=AppIdentityStatus.UNKNOWN),
        )
    )

    assert result.status == AppControlStatus.BLOCKED
    assert result.reason == AppControlReason.APP_BLOCKED_BY_IDENTITY
    assert result.control_eligible is False


def test_launch_visible_responsive_and_restored() -> None:
    runtime = AppControlRuntime()
    session = runtime.create_session(workspace_id="workspace")
    policy = _policy(
        action=SimulatedActionKind.OPEN,
        user_intent="open vscode",
    )

    result = runtime.control(
        AppControlRequest(
            session_id=session.session_id,
            action=AppControlActionKind.LAUNCH,
            target=_target(app_name="VS Code", app_kind=DetectedAppKind.IDE),
            policy_result=policy,
            identity=_identity(),
            require_session_restore=True,
        )
    )

    assert result.status == AppControlStatus.LAUNCHED
    assert result.decision == AppControlDecision.ALLOW
    assert result.control_eligible is True

    assert result.visibility is not None
    assert result.visibility.visible is True
    assert result.visibility.focused is True
    assert result.visibility.state == AppVisibilityState.VISIBLE

    assert result.responsiveness is not None
    assert result.responsiveness.responsive is True

    assert result.restore_plan is not None
    assert result.restore_plan.status == AppSessionRestoreStatus.RESTORED


def test_focus_app_when_visible_and_responsive() -> None:
    runtime = AppControlRuntime()
    session = runtime.create_session(workspace_id="workspace")
    policy = _policy(
        action=SimulatedActionKind.FOCUS,
        user_intent="focus vscode",
    )

    result = runtime.control(
        AppControlRequest(
            session_id=session.session_id,
            action=AppControlActionKind.FOCUS,
            target=_target(app_name="VS Code", app_kind=DetectedAppKind.IDE),
            policy_result=policy,
            identity=_identity(),
        )
    )

    assert result.status == AppControlStatus.FOCUSED
    assert result.decision == AppControlDecision.ALLOW
    assert result.control_eligible is True

    assert result.visibility is not None
    assert result.visibility.state == AppVisibilityState.VISIBLE
    assert result.visibility.focused is True

    assert result.responsiveness is not None
    assert result.responsiveness.responsive is True


def test_switch_app() -> None:
    runtime = AppControlRuntime()
    session = runtime.create_session(workspace_id="workspace")
    policy = _policy(
        action=SimulatedActionKind.FOCUS,
        user_intent="switch to browser",
    )

    result = runtime.control(
        AppControlRequest(
            session_id=session.session_id,
            action=AppControlActionKind.SWITCH,
            target=_target(app_name="Chrome", app_kind=DetectedAppKind.BROWSER),
            policy_result=policy,
            identity=_identity(),
        )
    )

    assert result.status == AppControlStatus.SWITCHED
    assert result.decision == AppControlDecision.ALLOW
    assert result.control_eligible is True


def test_focus_blocks_blocked_identity() -> None:
    runtime = AppControlRuntime()
    session = runtime.create_session(workspace_id="workspace")
    policy = _policy(
        action=SimulatedActionKind.FOCUS,
        user_intent="focus browser",
    )

    result = runtime.control(
        AppControlRequest(
            session_id=session.session_id,
            action=AppControlActionKind.FOCUS,
            target=_target(app_name="Chrome", app_kind=DetectedAppKind.BROWSER),
            policy_result=policy,
            identity=_identity(status=AppIdentityStatus.BLOCKED),
        )
    )

    assert result.status == AppControlStatus.BLOCKED
    assert result.reason == AppControlReason.APP_BLOCKED_BY_IDENTITY
    assert result.control_eligible is False


def test_close_blocks_unsaved_changes() -> None:
    runtime = AppControlRuntime()
    session = runtime.create_session(workspace_id="workspace")
    policy = _policy(
        action=SimulatedActionKind.CLOSE,
        user_intent="close vscode",
    )

    result = runtime.control(
        AppControlRequest(
            session_id=session.session_id,
            action=AppControlActionKind.CLOSE,
            target=_target(app_name="VS Code", app_kind=DetectedAppKind.IDE),
            policy_result=policy,
            identity=_identity(),
            unsaved_changes_hint=True,
        )
    )

    assert result.status == AppControlStatus.BLOCKED
    assert result.reason == AppControlReason.UNSAVED_STATE_BLOCKED_CLOSE
    assert result.close_safety is not None
    assert result.close_safety.safety == AppCloseSafety.UNSAVED_CHANGES
    assert result.close_safety.requires_approval is True
    assert result.close_safety.safe_to_close is False


def test_close_blocks_background_task() -> None:
    runtime = AppControlRuntime()
    session = runtime.create_session(workspace_id="workspace")
    policy = _policy(
        action=SimulatedActionKind.CLOSE,
        user_intent="close terminal",
    )

    result = runtime.control(
        AppControlRequest(
            session_id=session.session_id,
            action=AppControlActionKind.CLOSE,
            target=_target(app_name="Terminal", app_kind=DetectedAppKind.TERMINAL),
            policy_result=policy,
            identity=_identity(),
            background_task_hint=True,
        )
    )

    assert result.status == AppControlStatus.BLOCKED
    assert result.reason == AppControlReason.APP_CLOSE_APPROVAL_REQUIRED
    assert result.close_safety is not None
    assert result.close_safety.safety == AppCloseSafety.BACKGROUND_TASK_RUNNING


def test_close_blocks_unknown_app() -> None:
    runtime = AppControlRuntime()
    session = runtime.create_session(workspace_id="workspace")
    policy = _policy(
        action=SimulatedActionKind.CLOSE,
        user_intent="close unknown app",
    )

    result = runtime.control(
        AppControlRequest(
            session_id=session.session_id,
            action=AppControlActionKind.CLOSE,
            target=_target(app_name="Unknown", app_kind=DetectedAppKind.UNKNOWN),
            policy_result=policy,
            identity=_identity(),
        )
    )

    assert result.status == AppControlStatus.BLOCKED
    assert result.reason == AppControlReason.APP_CLOSE_APPROVAL_REQUIRED
    assert result.close_safety is not None
    assert result.close_safety.safety == AppCloseSafety.CRITICAL_OR_UNKNOWN_APP


def test_close_safe_app() -> None:
    runtime = AppControlRuntime()
    session = runtime.create_session(workspace_id="workspace")
    policy = _policy(
        action=SimulatedActionKind.CLOSE,
        user_intent="close browser",
    )

    result = runtime.control(
        AppControlRequest(
            session_id=session.session_id,
            action=AppControlActionKind.CLOSE,
            target=_target(app_name="Chrome", app_kind=DetectedAppKind.BROWSER),
            policy_result=policy,
            identity=_identity(),
        )
    )

    assert result.status == AppControlStatus.CLOSE_READY
    assert result.decision == AppControlDecision.ALLOW
    assert result.control_eligible is True

    assert result.close_safety is not None
    assert result.close_safety.safety == AppCloseSafety.SAFE_TO_CLOSE
    assert result.close_safety.safe_to_close is True


def test_restore_session_action() -> None:
    runtime = AppControlRuntime()
    session = runtime.create_session(workspace_id="workspace")
    policy = _policy(
        action=SimulatedActionKind.OPEN,
        user_intent="restore vscode",
    )

    result = runtime.control(
        AppControlRequest(
            session_id=session.session_id,
            action=AppControlActionKind.RESTORE_SESSION,
            target=_target(app_name="VS Code", app_kind=DetectedAppKind.IDE),
            policy_result=policy,
            identity=_identity(),
            require_session_restore=True,
        )
    )

    assert result.status == AppControlStatus.RESTORED
    assert result.decision == AppControlDecision.ALLOW
    assert result.control_eligible is True
    assert result.restore_plan is not None
    assert result.restore_plan.status == AppSessionRestoreStatus.RESTORED


def test_check_responsiveness_action() -> None:
    runtime = AppControlRuntime()
    session = runtime.create_session(workspace_id="workspace")
    policy = _policy(
        action=SimulatedActionKind.FOCUS,
        user_intent="check vscode responsiveness",
    )

    result = runtime.control(
        AppControlRequest(
            session_id=session.session_id,
            action=AppControlActionKind.CHECK_RESPONSIVENESS,
            target=_target(app_name="VS Code", app_kind=DetectedAppKind.IDE),
            policy_result=policy,
            identity=_identity(),
        )
    )

    assert result.status == AppControlStatus.READY
    assert result.decision == AppControlDecision.ALLOW
    assert result.control_eligible is True
    assert result.responsiveness is not None
    assert result.responsiveness.responsive is True


def test_missing_session_fails() -> None:
    runtime = AppControlRuntime()
    policy = _policy(
        action=SimulatedActionKind.OPEN,
        user_intent="open vscode",
    )

    result = runtime.control(
        AppControlRequest(
            session_id="missing",
            action=AppControlActionKind.LAUNCH,
            target=_target(app_name="VS Code", app_kind=DetectedAppKind.IDE),
            policy_result=policy,
            identity=_identity(),
        )
    )

    assert result.status == AppControlStatus.FAILED
    assert result.reason == AppControlReason.SESSION_NOT_FOUND
    assert result.control_eligible is False


def test_snapshot_tracks_counts() -> None:
    runtime = AppControlRuntime()
    session = runtime.create_session(workspace_id="workspace")
    policy_open = _policy(
        action=SimulatedActionKind.OPEN,
        user_intent="open vscode",
    )
    policy_focus = _policy(
        action=SimulatedActionKind.FOCUS,
        user_intent="focus vscode",
    )

    runtime.control(
        AppControlRequest(
            session_id=session.session_id,
            action=AppControlActionKind.LAUNCH,
            target=_target(app_name="VS Code", app_kind=DetectedAppKind.IDE),
            policy_result=policy_open,
            identity=_identity(),
        )
    )
    runtime.control(
        AppControlRequest(
            session_id=session.session_id,
            action=AppControlActionKind.FOCUS,
            target=_target(app_name="VS Code", app_kind=DetectedAppKind.IDE),
            policy_result=policy_focus,
            identity=_identity(),
        )
    )
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.result_count == 2
    assert snapshot.eligible_count == 2
    assert snapshot.launched_count == 1
    assert snapshot.focused_count == 1
    assert snapshot.audit_count == 2


def test_session_tracks_active_app_and_restore_count() -> None:
    runtime = AppControlRuntime()
    session = runtime.create_session(workspace_id="workspace")
    policy = _policy(
        action=SimulatedActionKind.OPEN,
        user_intent="restore vscode",
    )

    runtime.control(
        AppControlRequest(
            session_id=session.session_id,
            action=AppControlActionKind.RESTORE_SESSION,
            target=_target(app_name="VS Code", app_kind=DetectedAppKind.IDE),
            policy_result=policy,
            identity=_identity(),
            require_session_restore=True,
        )
    )
    stored = runtime.session_for(session.session_id)

    assert stored is not None
    assert stored.active_app_name == "VS Code"
    assert stored.restored_session_count == 1


def test_reset_clears_runtime() -> None:
    runtime = AppControlRuntime()
    runtime.create_session(workspace_id="workspace")

    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.runtime_event_count == 1
    assert snapshot.last_reason == AppControlReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert AppControlActionKind.LAUNCH.value == "launch"
    assert AppControlStatus.LAUNCHED.value == "launched"
    assert AppCloseSafety.SAFE_TO_CLOSE.value == "safe_to_close"


def _target(
    *,
    app_name: str = "VS Code",
    app_kind: DetectedAppKind = DetectedAppKind.IDE,
) -> AppControlTarget:
    return AppControlTarget(
        app_name=app_name,
        app_kind=app_kind,
        window_title=app_name,
        workspace_id="workspace",
    )


def _identity(
    *,
    status: AppIdentityStatus = AppIdentityStatus.IDENTIFIED,
) -> AppIdentityResult:
    return AppIdentityResult.model_construct(status=status)


def _policy(
    *,
    action: SimulatedActionKind,
    user_intent: str,
) -> InteractionPolicyResult:
    planning = EnvironmentActionPlanningRuntime()
    plan_session = planning.create_session(workspace_id="workspace")
    plan = planning.plan(
        EnvironmentActionPlanningRequest(
            session_id=plan_session.session_id,
            workspace_id="workspace",
            user_intent=user_intent,
            proposed_action_kind=action,
        )
    )

    policy = InteractionPolicyRuntime()
    policy_session = policy.create_session(workspace_id="workspace")

    return policy.evaluate(
        PhysicalInteractionRequest(
            session_id=policy_session.session_id,
            workspace_id="workspace",
            contract=contract_from_plan(plan),
            plan=plan,
        )
    )