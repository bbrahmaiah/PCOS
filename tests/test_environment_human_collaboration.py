from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    ApprovalDialogueRequest,
    ApprovalDialogueResponse,
    ApprovalDialogueStatus,
    ApprovalRisk,
    CollaborationDecision,
    CollaborationPhase,
    CollaborationReason,
    CollaborationRequest,
    CollaborationStatus,
    HumanCollaborationRuntime,
    NarrationPolicy,
    NarrationTone,
    ProgressNarration,
    UserOverrideKind,
    UserOverrideRuntime,
)


def test_runtime_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        HumanCollaborationRuntime(name=" ")


def test_narration_policy_validates_message_bounds() -> None:
    with pytest.raises(ValidationError):
        NarrationPolicy(max_message_chars=10)


def test_progress_narration_must_be_user_visible() -> None:
    with pytest.raises(ValidationError):
        ProgressNarration(
            phase=CollaborationPhase.FOUND_ERROR,
            text="I found the error.",
            user_visible=False,
            tone=NarrationTone.CALM,
        )


def test_create_session() -> None:
    runtime = HumanCollaborationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    assert session.workspace_id == "workspace"
    assert runtime.snapshot().session_count == 1


def test_narrates_found_error() -> None:
    runtime = HumanCollaborationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.narrate_progress(
        CollaborationRequest(
            session_id=session.session_id,
            phase=CollaborationPhase.FOUND_ERROR,
        )
    )

    assert result.status == CollaborationStatus.NARRATED
    assert result.narration is not None
    assert result.narration.text == "error_found"


def test_narrates_opening_file_with_target() -> None:
    runtime = HumanCollaborationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.narrate_progress(
        CollaborationRequest(
            session_id=session.session_id,
            phase=CollaborationPhase.OPENING_FILE,
            target="main.py",
        )
    )

    assert result.status == CollaborationStatus.NARRATED
    assert result.narration is not None
    assert "main.py" in result.narration.text


def test_narrates_tests_running() -> None:
    runtime = HumanCollaborationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.narrate_progress(
        CollaborationRequest(
            session_id=session.session_id,
            phase=CollaborationPhase.RUNNING_TESTS,
        )
    )

    assert result.status == CollaborationStatus.NARRATED
    assert result.narration is not None
    assert result.narration.text == "tests_running"


def test_narrates_issue_remains() -> None:
    runtime = HumanCollaborationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.narrate_progress(
        CollaborationRequest(
            session_id=session.session_id,
            phase=CollaborationPhase.ISSUE_REMAINS,
        )
    )

    assert result.status == CollaborationStatus.NARRATED
    assert result.narration is not None
    assert result.narration.text == "issue_remaining"


def test_silent_physical_control_is_blocked() -> None:
    runtime = HumanCollaborationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.narrate_progress(
        CollaborationRequest(
            session_id=session.session_id,
            phase=CollaborationPhase.APPLYING_CHANGE,
            silent_physical_control=True,
        )
    )

    assert result.status == CollaborationStatus.BLOCKED
    assert result.reason == CollaborationReason.SILENT_CONTROL_BLOCKED


def test_progress_request_with_approval_required_blocks_for_approval() -> None:
    runtime = HumanCollaborationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.narrate_progress(
        CollaborationRequest(
            session_id=session.session_id,
            phase=CollaborationPhase.APPLYING_CHANGE,
            requires_approval=True,
        )
    )

    assert result.status == CollaborationStatus.APPROVAL_REQUIRED
    assert result.decision == CollaborationDecision.ASK_APPROVAL


def test_request_approval_creates_dialogue() -> None:
    runtime = HumanCollaborationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.request_approval(
        ApprovalDialogueRequest(
            session_id=session.session_id,
            proposed_action="submitting the form",
            reason="this changes external state",
            risk=ApprovalRisk.HIGH,
            target="browser form",
        )
    )

    assert result.status == CollaborationStatus.APPROVAL_REQUIRED
    assert result.approval_dialogue is not None
    assert result.approval_dialogue.status == ApprovalDialogueStatus.PENDING
    assert result.approval_dialogue.requires_explicit_approval is True
    assert "approval" in result.message.lower()


def test_approval_response_approved() -> None:
    runtime = HumanCollaborationRuntime()
    session = runtime.create_session(workspace_id="workspace")
    requested = runtime.request_approval(
        ApprovalDialogueRequest(
            session_id=session.session_id,
            proposed_action="closing the app",
            reason="there may be unsaved work",
            risk=ApprovalRisk.HIGH,
        )
    )

    assert requested.approval_dialogue is not None

    result = runtime.respond_to_approval(
        session_id=session.session_id,
        response=ApprovalDialogueResponse(
            dialogue_id=requested.approval_dialogue.dialogue_id,
            approved=True,
            user_text="approve",
        ),
    )

    assert result.status == CollaborationStatus.APPROVED
    assert result.decision == CollaborationDecision.CONTINUE


def test_approval_response_denied() -> None:
    runtime = HumanCollaborationRuntime()
    session = runtime.create_session(workspace_id="workspace")
    requested = runtime.request_approval(
        ApprovalDialogueRequest(
            session_id=session.session_id,
            proposed_action="deleting the file",
            reason="delete is risky",
            risk=ApprovalRisk.CRITICAL,
            irreversible=True,
        )
    )

    assert requested.approval_dialogue is not None

    result = runtime.respond_to_approval(
        session_id=session.session_id,
        response=ApprovalDialogueResponse(
            dialogue_id=requested.approval_dialogue.dialogue_id,
            approved=False,
            user_text="deny",
        ),
    )

    assert result.status == CollaborationStatus.DENIED
    assert result.decision == CollaborationDecision.BLOCK


def test_user_override_runtime_detects_cancel_pause_takeover() -> None:
    runtime = UserOverrideRuntime()

    cancel = runtime.parse(session_id="session", text="stop")
    pause = runtime.parse(session_id="session", text="pause")
    takeover = runtime.parse(session_id="session", text="let me do it")

    assert cancel.kind == UserOverrideKind.CANCEL
    assert pause.kind == UserOverrideKind.PAUSE
    assert takeover.kind == UserOverrideKind.TAKE_OVER


def test_handle_user_cancel_override() -> None:
    runtime = HumanCollaborationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.handle_user_override(
        session_id=session.session_id,
        user_text="stop",
    )

    stored = runtime.session_for(session.session_id)

    assert result.status == CollaborationStatus.OVERRIDDEN
    assert result.decision == CollaborationDecision.CANCEL
    assert stored is not None
    assert stored.cancelled is True


def test_handle_user_pause_override() -> None:
    runtime = HumanCollaborationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.handle_user_override(
        session_id=session.session_id,
        user_text="pause",
    )

    stored = runtime.session_for(session.session_id)

    assert result.status == CollaborationStatus.OVERRIDDEN
    assert result.decision == CollaborationDecision.PAUSE
    assert stored is not None
    assert stored.paused is True


def test_missing_session_fails_operations() -> None:
    runtime = HumanCollaborationRuntime()

    narration = runtime.narrate_progress(
        CollaborationRequest(
            session_id="missing",
            phase=CollaborationPhase.FOUND_ERROR,
        )
    )
    approval = runtime.request_approval(
        ApprovalDialogueRequest(
            session_id="missing",
            proposed_action="submit form",
            reason="external state change",
            risk=ApprovalRisk.HIGH,
        )
    )
    override = runtime.handle_user_override(
        session_id="missing",
        user_text="stop",
    )

    assert narration.status == CollaborationStatus.FAILED
    assert approval.status == CollaborationStatus.FAILED
    assert override.status == CollaborationStatus.FAILED


def test_snapshot_tracks_counts() -> None:
    runtime = HumanCollaborationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    runtime.narrate_progress(
        CollaborationRequest(
            session_id=session.session_id,
            phase=CollaborationPhase.FOUND_ERROR,
        )
    )
    requested = runtime.request_approval(
        ApprovalDialogueRequest(
            session_id=session.session_id,
            proposed_action="submit form",
            reason="external state change",
            risk=ApprovalRisk.HIGH,
        )
    )
    assert requested.approval_dialogue is not None

    runtime.respond_to_approval(
        session_id=session.session_id,
        response=ApprovalDialogueResponse(
            dialogue_id=requested.approval_dialogue.dialogue_id,
            approved=True,
            user_text="approve",
        ),
    )
    runtime.handle_user_override(
        session_id=session.session_id,
        user_text="pause",
    )

    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.narration_count == 1
    assert snapshot.approval_count == 1
    assert snapshot.approved_count == 1
    assert snapshot.override_count == 1


def test_session_tracks_counts() -> None:
    runtime = HumanCollaborationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    runtime.narrate_progress(
        CollaborationRequest(
            session_id=session.session_id,
            phase=CollaborationPhase.FOUND_ERROR,
        )
    )
    runtime.handle_user_override(
        session_id=session.session_id,
        user_text="stop",
    )

    stored = runtime.session_for(session.session_id)

    assert stored is not None
    assert stored.narration_count == 1
    assert stored.override_count == 1
    assert stored.cancelled is True


def test_reset_clears_runtime() -> None:
    runtime = HumanCollaborationRuntime()
    runtime.create_session(workspace_id="workspace")

    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.runtime_event_count == 1
    assert snapshot.last_reason == CollaborationReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert CollaborationPhase.FOUND_ERROR.value == "found_error"
    assert CollaborationStatus.NARRATED.value == "narrated"
    assert ApprovalRisk.CRITICAL.value == "critical"