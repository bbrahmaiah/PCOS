from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    EnvironmentSimulationRequest,
    EnvironmentSimulationResult,
    EnvironmentSimulationRuntime,
    EnvironmentSimulationStatus,
    ExpectedStateKind,
    RollbackCapability,
    SimulatedAction,
    SimulatedActionKind,
    SimulationDecision,
    SimulationReason,
    SimulationRiskLevel,
    TrustPolicyClassification,
)


def test_runtime_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        EnvironmentSimulationRuntime(name=" ")


def test_create_session() -> None:
    runtime = EnvironmentSimulationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    assert session.workspace_id == "workspace"
    assert runtime.snapshot().session_count == 1


def test_type_text_requires_payload() -> None:
    with pytest.raises(ValidationError):
        SimulatedAction(
            kind=SimulatedActionKind.TYPE_TEXT,
            description="type into field",
        )


def test_safe_result_requires_outcome() -> None:
    action = SimulatedAction(
        kind=SimulatedActionKind.FOCUS,
        description="focus terminal",
    )

    with pytest.raises(ValidationError):
        EnvironmentSimulationResult(
            status=EnvironmentSimulationStatus.PREDICTED,
            decision=SimulationDecision.ALLOW_PLANNING,
            reason=SimulationReason.OUTCOME_PREDICTED,
            request_id="request",
            action=action,
            outcome=None,
            safe_for_planning=True,
            requires_verification=False,
            requires_user_approval=False,
            message="invalid",
        )


def test_click_save_predicts_dialog_close_and_timestamp() -> None:
    runtime = EnvironmentSimulationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.simulate(
        EnvironmentSimulationRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            action=SimulatedAction(
                kind=SimulatedActionKind.CLICK,
                description="click save",
                target_label="Save",
            ),
        )
    )

    assert result.outcome is not None
    kinds = {change.kind for change in result.outcome.expected_changes}
    assert ExpectedStateKind.DIALOG_CLOSES in kinds
    assert ExpectedStateKind.FILE_TIMESTAMP_UPDATES in kinds
    assert result.safe_for_planning is True
    assert result.requires_verification is True


def test_type_text_predicts_target_text_only() -> None:
    runtime = EnvironmentSimulationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.simulate(
        EnvironmentSimulationRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            action=SimulatedAction(
                kind=SimulatedActionKind.TYPE_TEXT,
                description="type into search field",
                target_label="search",
                text_payload="hello",
            ),
        )
    )

    assert result.outcome is not None
    assert result.outcome.expected_changes[0].kind == (
        ExpectedStateKind.TEXT_APPEARS_IN_TARGET
    )
    assert result.outcome.rollback_risk.risk_level == SimulationRiskLevel.MEDIUM
    assert result.decision == SimulationDecision.VERIFY_FIRST


def test_close_predicts_unsaved_prompt_risk() -> None:
    runtime = EnvironmentSimulationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.simulate(
        EnvironmentSimulationRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            action=SimulatedAction(
                kind=SimulatedActionKind.CLOSE,
                description="close editor",
                target_label="Close",
            ),
        )
    )

    assert result.outcome is not None
    assert result.outcome.expected_changes[0].kind == (
        ExpectedStateKind.UNSAVED_PROMPT_MAY_APPEAR
    )
    assert result.outcome.rollback_risk.requires_user_approval is True


def test_delete_is_irreversible_and_requires_approval() -> None:
    runtime = EnvironmentSimulationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.simulate(
        EnvironmentSimulationRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            action=SimulatedAction(
                kind=SimulatedActionKind.DELETE,
                description="delete file",
                target_label="old.log",
            ),
        )
    )

    assert result.status == EnvironmentSimulationStatus.HIGH_RISK
    assert result.decision == SimulationDecision.REQUIRE_APPROVAL
    assert result.reason == SimulationReason.IRREVERSIBLE_ACTION
    assert result.outcome is not None
    assert result.outcome.rollback_risk.risk_level == (
        SimulationRiskLevel.IRREVERSIBLE
    )
    assert result.outcome.rollback_risk.rollback_capability == (
        RollbackCapability.RESTORE_FILE_BACKUP
    )


def test_submit_requires_approval() -> None:
    runtime = EnvironmentSimulationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.simulate(
        EnvironmentSimulationRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            action=SimulatedAction(
                kind=SimulatedActionKind.SUBMIT,
                description="submit form",
                target_label="Submit",
            ),
        )
    )

    assert result.status == EnvironmentSimulationStatus.HIGH_RISK
    assert result.requires_user_approval is True
    assert result.safe_for_planning is False


def test_move_file_requires_backup_and_approval() -> None:
    runtime = EnvironmentSimulationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.simulate(
        EnvironmentSimulationRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            action=SimulatedAction(
                kind=SimulatedActionKind.MOVE_FILE,
                description="move file",
                target_label="report.pdf",
            ),
        )
    )

    assert result.outcome is not None
    assert result.outcome.rollback_risk.requires_backup is True
    assert result.requires_user_approval is True


def test_change_setting_high_risk() -> None:
    runtime = EnvironmentSimulationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.simulate(
        EnvironmentSimulationRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            action=SimulatedAction(
                kind=SimulatedActionKind.CHANGE_SETTING,
                description="disable setting",
                target_label="Auto Save",
            ),
        )
    )

    assert result.status == EnvironmentSimulationStatus.HIGH_RISK
    assert result.decision == SimulationDecision.REQUIRE_APPROVAL


def test_blocked_policy_blocks_action() -> None:
    runtime = EnvironmentSimulationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.simulate(
        EnvironmentSimulationRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            action=SimulatedAction(
                kind=SimulatedActionKind.CLICK,
                description="click blocked target",
                target_label="secret",
                source_policy=TrustPolicyClassification.BLOCKED,
            ),
        )
    )

    assert result.status == EnvironmentSimulationStatus.BLOCKED
    assert result.decision == SimulationDecision.BLOCK_ACTION
    assert result.reason == SimulationReason.ACTION_BLOCKED_BY_POLICY
    assert result.safe_for_planning is False


def test_focus_low_risk_needs_verification() -> None:
    runtime = EnvironmentSimulationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.simulate(
        EnvironmentSimulationRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            action=SimulatedAction(
                kind=SimulatedActionKind.FOCUS,
                description="focus terminal",
                target_label="Terminal",
            ),
        )
    )

    assert result.outcome is not None
    assert result.outcome.expected_changes[0].kind == ExpectedStateKind.APP_FOCUSED
    assert result.decision == SimulationDecision.VERIFY_FIRST
    assert result.safe_for_planning is True


def test_copy_predicts_clipboard_update() -> None:
    runtime = EnvironmentSimulationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.simulate(
        EnvironmentSimulationRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            action=SimulatedAction(
                kind=SimulatedActionKind.COPY,
                description="copy selected text",
                target_label="selection",
            ),
        )
    )

    assert result.outcome is not None
    assert result.outcome.expected_changes[0].kind == (
        ExpectedStateKind.CLIPBOARD_UPDATED
    )


def test_missing_session_fails() -> None:
    runtime = EnvironmentSimulationRuntime()

    result = runtime.simulate(
        EnvironmentSimulationRequest(
            session_id="missing",
            workspace_id="workspace",
            action=SimulatedAction(
                kind=SimulatedActionKind.CLICK,
                description="click run",
                target_label="Run",
            ),
        )
    )

    assert result.status == EnvironmentSimulationStatus.FAILED
    assert result.reason == SimulationReason.SESSION_NOT_FOUND


def test_snapshot_tracks_counts() -> None:
    runtime = EnvironmentSimulationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    runtime.simulate(
        EnvironmentSimulationRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            action=SimulatedAction(
                kind=SimulatedActionKind.FOCUS,
                description="focus terminal",
                target_label="Terminal",
            ),
        )
    )
    runtime.simulate(
        EnvironmentSimulationRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            action=SimulatedAction(
                kind=SimulatedActionKind.DELETE,
                description="delete file",
                target_label="old.log",
            ),
        )
    )
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.simulation_count == 2
    assert snapshot.verification_count == 1
    assert snapshot.high_risk_count == 1
    assert snapshot.safe_planning_count == 1


def test_reset_clears_runtime() -> None:
    runtime = EnvironmentSimulationRuntime()
    runtime.create_session(workspace_id="workspace")

    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.runtime_event_count == 1
    assert snapshot.last_reason == SimulationReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert SimulatedActionKind.DELETE.value == "delete"
    assert SimulationRiskLevel.IRREVERSIBLE.value == "irreversible"
    assert SimulationDecision.REQUIRE_APPROVAL.value == "require_approval"