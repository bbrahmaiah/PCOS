from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from jarvis.tools import (
    ActionCancellationState,
    ActionInterruptController,
    ActionInterruptControllerConfig,
    ActionInterruptDecision,
    ActionInterruptKind,
    ActionInterruptReason,
    ActionInterruptRequest,
    ActionRisk,
    FileRollbackExecutor,
    RollbackPlan,
    RollbackReason,
    RollbackStatus,
    RollbackStep,
    RollbackStepKind,
)


def test_controller_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        ActionInterruptControllerConfig(name=" ").validate()


def test_cancellation_token_lifecycle_cancel() -> None:
    controller = ActionInterruptController()
    token = controller.create_token(action_id="action-1")

    result = controller.interrupt(
        ActionInterruptRequest(
            action_id="action-1",
            kind=ActionInterruptKind.CANCEL,
            reason=ActionInterruptReason.USER_REQUESTED,
        )
    )

    updated = controller.get_token("action-1")
    event = controller.cancellation_event("action-1")

    assert token.state == ActionCancellationState.ACTIVE
    assert result.decision == ActionInterruptDecision.ACCEPTED
    assert result.next_state == ActionCancellationState.CANCEL_REQUESTED
    assert updated is not None
    assert updated.cancellation_requested is True
    assert event is not None
    assert event.is_set()


def test_pause_resume_lifecycle() -> None:
    controller = ActionInterruptController()
    controller.create_token(action_id="action-1")

    pause = controller.interrupt(
        ActionInterruptRequest(
            action_id="action-1",
            kind=ActionInterruptKind.PAUSE,
        )
    )
    marked = controller.mark_paused("action-1")
    resume = controller.interrupt(
        ActionInterruptRequest(
            action_id="action-1",
            kind=ActionInterruptKind.RESUME,
        )
    )

    assert pause.next_state == ActionCancellationState.PAUSE_REQUESTED
    assert marked.next_state == ActionCancellationState.PAUSED
    assert resume.next_state == ActionCancellationState.RESUME_REQUESTED


def test_timeout_sets_timed_out_state() -> None:
    controller = ActionInterruptController()
    controller.create_token(action_id="action-1")

    result = controller.interrupt(
        ActionInterruptRequest(
            action_id="action-1",
            kind=ActionInterruptKind.TIMEOUT,
            reason=ActionInterruptReason.TIMEOUT_EXCEEDED,
        )
    )
    updated = controller.get_token("action-1")

    assert result.next_state == ActionCancellationState.TIMED_OUT
    assert updated is not None
    assert updated.terminal is True


def test_interrupt_rejects_unknown_action() -> None:
    controller = ActionInterruptController()

    result = controller.interrupt(
        ActionInterruptRequest(
            action_id="missing",
            kind=ActionInterruptKind.CANCEL,
        )
    )

    assert result.decision == ActionInterruptDecision.REJECTED
    assert result.message == "action has no cancellation token"


def test_interrupt_rejects_non_interruptible_without_force() -> None:
    controller = ActionInterruptController()
    controller.create_token(action_id="action-1", interruptible=False)

    result = controller.interrupt(
        ActionInterruptRequest(
            action_id="action-1",
            kind=ActionInterruptKind.CANCEL,
        )
    )

    assert result.decision == ActionInterruptDecision.REJECTED
    assert result.message == "action is not interruptible"


def test_interrupt_force_accepts_non_interruptible() -> None:
    controller = ActionInterruptController()
    controller.create_token(action_id="action-1", interruptible=False)

    result = controller.interrupt(
        ActionInterruptRequest(
            action_id="action-1",
            kind=ActionInterruptKind.CANCEL,
            force=True,
        )
    )

    assert result.decision == ActionInterruptDecision.ACCEPTED
    assert result.next_state == ActionCancellationState.CANCEL_REQUESTED


def test_terminal_token_interrupt_is_ignored() -> None:
    controller = ActionInterruptController()
    controller.create_token(action_id="action-1")
    controller.mark_completed("action-1")

    result = controller.interrupt(
        ActionInterruptRequest(
            action_id="action-1",
            kind=ActionInterruptKind.CANCEL,
        )
    )

    assert result.decision == ActionInterruptDecision.IGNORED


def test_rollback_step_validation() -> None:
    with pytest.raises(ValidationError):
        RollbackStep(
            kind=RollbackStepKind.RESTORE_BACKUP,
            description="restore",
            target_path="a.txt",
        )


def test_rollback_plan_requires_steps_when_available() -> None:
    with pytest.raises(ValidationError):
        RollbackPlan(
            action_id="action-1",
            status=RollbackStatus.AVAILABLE,
            explanation="available but empty",
        )


def test_unsafe_rollback_plan_cannot_have_steps() -> None:
    with pytest.raises(ValidationError):
        RollbackPlan(
            action_id="action-1",
            status=RollbackStatus.UNSAFE,
            explanation="unsafe",
            steps=(
                RollbackStep(
                    kind=RollbackStepKind.NOOP,
                    description="noop",
                ),
            ),
        )


def test_file_rollback_restores_backup(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    backup = tmp_path / ".jarvis_backups" / "a.txt.bak"
    target.write_text("new", encoding="utf-8")
    backup.parent.mkdir(parents=True)
    backup.write_text("old", encoding="utf-8")

    plan = RollbackPlan(
        action_id="action-1",
        risk=ActionRisk.MEDIUM,
        explanation="restore a.txt from backup",
        steps=(
            RollbackStep(
                kind=RollbackStepKind.RESTORE_BACKUP,
                description="restore backup",
                target_path="a.txt",
                backup_path=".jarvis_backups/a.txt.bak",
            ),
        ),
    )
    result = FileRollbackExecutor(workspace_root=str(tmp_path)).execute(plan)

    assert result.success is True
    assert result.status == RollbackStatus.SUCCEEDED
    assert target.read_text(encoding="utf-8") == "old"


def test_file_rollback_deletes_created_file(tmp_path: Path) -> None:
    created = tmp_path / "created.txt"
    created.write_text("created", encoding="utf-8")

    plan = RollbackPlan(
        action_id="action-1",
        explanation="delete created file",
        steps=(
            RollbackStep(
                kind=RollbackStepKind.DELETE_CREATED_FILE,
                description="delete created",
                target_path="created.txt",
            ),
        ),
    )

    result = FileRollbackExecutor(workspace_root=str(tmp_path)).execute(plan)

    assert result.success is True
    assert not created.exists()


def test_file_rollback_moves_file_back(tmp_path: Path) -> None:
    moved = tmp_path / "moved.txt"
    original = tmp_path / "original.txt"
    moved.write_text("content", encoding="utf-8")

    plan = RollbackPlan(
        action_id="action-1",
        explanation="move file back",
        steps=(
            RollbackStep(
                kind=RollbackStepKind.MOVE_BACK,
                description="move back",
                target_path="moved.txt",
                destination_path="original.txt",
            ),
        ),
    )

    result = FileRollbackExecutor(workspace_root=str(tmp_path)).execute(plan)

    assert result.success is True
    assert not moved.exists()
    assert original.read_text(encoding="utf-8") == "content"


def test_file_rollback_blocks_workspace_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("outside", encoding="utf-8")

    plan = RollbackPlan(
        action_id="action-1",
        explanation="unsafe path",
        steps=(
            RollbackStep(
                kind=RollbackStepKind.DELETE_CREATED_FILE,
                description="delete outside",
                target_path="../outside.txt",
            ),
        ),
    )

    result = FileRollbackExecutor(workspace_root=str(tmp_path)).execute(plan)

    assert result.success is False
    assert result.status == RollbackStatus.FAILED


def test_controller_registers_and_runs_rollback(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    backup = tmp_path / ".jarvis_backups" / "a.txt.bak"
    target.write_text("new", encoding="utf-8")
    backup.parent.mkdir(parents=True)
    backup.write_text("old", encoding="utf-8")

    controller = ActionInterruptController(
        rollback_executor=FileRollbackExecutor(workspace_root=str(tmp_path))
    )
    plan = RollbackPlan(
        action_id="action-1",
        explanation="restore backup",
        steps=(
            RollbackStep(
                kind=RollbackStepKind.RESTORE_BACKUP,
                description="restore",
                target_path="a.txt",
                backup_path=".jarvis_backups/a.txt.bak",
            ),
        ),
    )

    controller.register_rollback_plan(plan)
    result = controller.rollback("action-1")

    assert result.success is True
    assert target.read_text(encoding="utf-8") == "old"


def test_controller_rollback_without_plan_is_unsafe() -> None:
    controller = ActionInterruptController()

    result = controller.rollback("missing-action")

    assert result.success is False
    assert result.status == RollbackStatus.UNSAFE
    assert result.reason == RollbackReason.ROLLBACK_UNAVAILABLE


def test_snapshot_and_reset() -> None:
    controller = ActionInterruptController()
    controller.create_token(action_id="action-1")
    controller.interrupt(
        ActionInterruptRequest(
            action_id="action-1",
            kind=ActionInterruptKind.CANCEL,
        )
    )

    snapshot = controller.snapshot()

    assert snapshot.token_count == 1
    assert snapshot.interrupt_count == 1
    assert snapshot.last_interrupt_kind == ActionInterruptKind.CANCEL

    controller.reset()
    reset_snapshot = controller.snapshot()

    assert reset_snapshot.token_count == 0
    assert reset_snapshot.interrupt_count == 0


def test_enum_values_are_stable() -> None:
    assert ActionInterruptKind.CANCEL.value == "cancel"
    assert ActionCancellationState.CANCEL_REQUESTED.value == "cancel_requested"
    assert RollbackStepKind.RESTORE_BACKUP.value == "restore_backup"
    assert RollbackStatus.UNSAFE.value == "unsafe"