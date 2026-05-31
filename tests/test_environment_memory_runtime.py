from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    CursorPosition,
    EnvironmentMemoryAuditRecord,
    EnvironmentMemoryDecision,
    EnvironmentMemoryReason,
    EnvironmentMemoryRuntime,
    EnvironmentMemoryStatus,
    EnvironmentWorkspaceMemoryEntry,
    TrustCalibration,
    WorkflowMemoryGateway,
    WorkflowStage,
)


def test_runtime_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        EnvironmentMemoryRuntime(name=" ")


def test_cursor_position_requires_file_path() -> None:
    with pytest.raises(ValidationError):
        CursorPosition(file_path=" ", line=1)


def test_workspace_memory_entry_rejects_empty_semantic_state() -> None:
    with pytest.raises(ValidationError):
        EnvironmentWorkspaceMemoryEntry(
            session_id="session",
            workspace_id="workspace",
            app_name="VS Code",
            continuity_token="token",
            trust=_runtime_trust(),
        )


def test_workspace_memory_blocks_raw_screen_metadata() -> None:
    with pytest.raises(ValidationError):
        EnvironmentWorkspaceMemoryEntry(
            session_id="session",
            workspace_id="workspace",
            app_name="VS Code",
            active_files=("main.py",),
            continuity_token="token",
            trust=_runtime_trust(),
            metadata={"screenshot_bytes": b"not allowed"},
        )


def test_audit_rejects_raw_screen_logged() -> None:
    with pytest.raises(ValidationError):
        EnvironmentMemoryAuditRecord(
            status=EnvironmentMemoryStatus.STORED,
            decision=EnvironmentMemoryDecision.STORE,
            reason=EnvironmentMemoryReason.WORKFLOW_ENTRY_STORED,
            raw_screen_logged=True,
        )


def test_create_session() -> None:
    runtime = EnvironmentMemoryRuntime()
    session = runtime.create_session(workspace_id="workspace")

    assert session.workspace_id == "workspace"
    assert runtime.snapshot().session_count == 1


def test_store_semantic_workflow_memory() -> None:
    runtime = EnvironmentMemoryRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.store_workflow(
        session_id=session.session_id,
        app_name="VS Code",
        project_path="E:/JARVIS_OS",
        active_files=("jarvis/environment/app.py",),
        cursor_positions=(
            CursorPosition(
                file_path="jarvis/environment/app.py",
                line=42,
                column=5,
                symbol="EnvironmentRuntime",
            ),
        ),
        terminal_directory="E:/JARVIS_OS",
        recent_commands=("pytest",),
        visible_errors=("AssertionError on line 42",),
        pending_todos=("fix verification path",),
        workflow_stage=WorkflowStage.DEBUGGING,
    )

    assert result.status == EnvironmentMemoryStatus.STORED
    assert result.entry is not None
    assert result.entry.app_name == "VS Code"
    assert result.entry.project_path == "E:/JARVIS_OS"
    assert result.entry.workflow_stage == WorkflowStage.DEBUGGING
    assert runtime.snapshot().stored_entry_count == 1


def test_store_blocks_raw_screen_metadata() -> None:
    runtime = EnvironmentMemoryRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.store_workflow(
        session_id=session.session_id,
        app_name="VS Code",
        active_files=("main.py",),
        metadata={"raw_ocr_dump": "do not store"},
    )

    assert result.status == EnvironmentMemoryStatus.BLOCKED
    assert result.reason == EnvironmentMemoryReason.RAW_SCREEN_MEMORY_BLOCKED


def test_recent_commands_are_redacted_when_sensitive() -> None:
    runtime = EnvironmentMemoryRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.store_workflow(
        session_id=session.session_id,
        app_name="Terminal",
        terminal_directory="E:/JARVIS_OS",
        recent_commands=("export API_KEY=super-secret-value",),
        workflow_stage=WorkflowStage.TESTING,
    )

    assert result.entry is not None
    assert result.entry.recent_commands[0].startswith("<redacted-command:")
    assert "super-secret-value" not in result.entry.recent_commands[0]


def test_recall_session_memory() -> None:
    runtime = EnvironmentMemoryRuntime()
    session = runtime.create_session(workspace_id="workspace")
    runtime.store_workflow(
        session_id=session.session_id,
        app_name="VS Code",
        active_files=("main.py",),
        workflow_stage=WorkflowStage.CODING,
    )

    recalled = runtime.recall_session(session_id=session.session_id)

    assert recalled.status == EnvironmentMemoryStatus.RECALLED
    assert recalled.reason == EnvironmentMemoryReason.SESSION_MEMORY_RECALLED
    assert len(recalled.entries) == 1


def test_recall_project_memory_builds_project_summary() -> None:
    runtime = EnvironmentMemoryRuntime()
    session = runtime.create_session(workspace_id="workspace")
    runtime.store_workflow(
        session_id=session.session_id,
        app_name="VS Code",
        project_path="E:/JARVIS_OS",
        active_files=("main.py", "runtime.py"),
        visible_errors=("mypy failed",),
        pending_todos=("fix type error",),
        workflow_stage=WorkflowStage.DEBUGGING,
    )

    recalled = runtime.recall_project(
        session_id=session.session_id,
        project_path="E:/JARVIS_OS",
    )

    assert recalled.status == EnvironmentMemoryStatus.RECALLED
    assert recalled.project_memory is not None
    assert recalled.project_memory.project_path == "E:/JARVIS_OS"
    assert recalled.project_memory.entry_count == 1
    assert recalled.project_memory.workflow_stage == WorkflowStage.DEBUGGING


def test_continue_workflow_creates_session_continuity() -> None:
    runtime = EnvironmentMemoryRuntime()
    session = runtime.create_session(workspace_id="workspace")
    runtime.store_workflow(
        session_id=session.session_id,
        app_name="VS Code",
        project_path="E:/JARVIS_OS",
        active_files=("main.py",),
        visible_errors=("pytest failure",),
        pending_todos=("repair Step 33",),
        workflow_stage=WorkflowStage.DEBUGGING,
    )

    result = runtime.continue_workflow(session_id=session.session_id)

    assert result.status == EnvironmentMemoryStatus.CONTINUITY_READY
    assert result.continuity is not None
    assert result.continuity.workflow_stage == WorkflowStage.DEBUGGING
    assert "VS Code" in result.continuity.resume_summary
    assert result.continuity.continuity_token


def test_continue_workflow_blocks_when_empty() -> None:
    runtime = EnvironmentMemoryRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.continue_workflow(session_id=session.session_id)

    assert result.status == EnvironmentMemoryStatus.BLOCKED
    assert result.reason == EnvironmentMemoryReason.EMPTY_MEMORY_BLOCKED


def test_missing_session_fails_store_recall_and_continue() -> None:
    runtime = EnvironmentMemoryRuntime()

    store = runtime.store_workflow(
        session_id="missing",
        app_name="VS Code",
        active_files=("main.py",),
    )
    recall = runtime.recall_session(session_id="missing")
    continuity = runtime.continue_workflow(session_id="missing")

    assert store.status == EnvironmentMemoryStatus.FAILED
    assert recall.status == EnvironmentMemoryStatus.FAILED
    assert continuity.status == EnvironmentMemoryStatus.FAILED


def test_gateway_is_boundary_for_storage() -> None:
    gateway = WorkflowMemoryGateway()
    runtime = EnvironmentMemoryRuntime(gateway=gateway)
    session = runtime.create_session(workspace_id="workspace")

    runtime.store_workflow(
        session_id=session.session_id,
        app_name="VS Code",
        active_files=("main.py",),
        workflow_stage=WorkflowStage.CODING,
    )

    assert len(gateway.snapshot_entries()) == 1
    assert runtime.gateway_entries() == gateway.snapshot_entries()


def test_snapshot_tracks_counts() -> None:
    runtime = EnvironmentMemoryRuntime()
    session = runtime.create_session(workspace_id="workspace")

    runtime.store_workflow(
        session_id=session.session_id,
        app_name="VS Code",
        active_files=("main.py",),
        workflow_stage=WorkflowStage.CODING,
    )
    runtime.recall_session(session_id=session.session_id)
    runtime.continue_workflow(session_id=session.session_id)

    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.stored_entry_count == 1
    assert snapshot.store_count == 1
    assert snapshot.recall_count == 1
    assert snapshot.continuity_count == 1
    assert snapshot.audit_count == 3


def test_session_tracks_counts() -> None:
    runtime = EnvironmentMemoryRuntime()
    session = runtime.create_session(workspace_id="workspace")

    runtime.store_workflow(
        session_id=session.session_id,
        app_name="VS Code",
        active_files=("main.py",),
        workflow_stage=WorkflowStage.CODING,
    )
    runtime.recall_session(session_id=session.session_id)
    runtime.continue_workflow(session_id=session.session_id)

    stored = runtime.session_for(session.session_id)

    assert stored is not None
    assert stored.store_count == 1
    assert stored.recall_count == 1
    assert stored.continuity_count == 1


def test_reset_clears_runtime_state_but_not_gateway_history() -> None:
    gateway = WorkflowMemoryGateway()
    runtime = EnvironmentMemoryRuntime(gateway=gateway)
    session = runtime.create_session(workspace_id="workspace")

    runtime.store_workflow(
        session_id=session.session_id,
        app_name="VS Code",
        active_files=("main.py",),
        workflow_stage=WorkflowStage.CODING,
    )
    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.runtime_event_count == 1
    assert snapshot.last_reason == EnvironmentMemoryReason.RUNTIME_RESET
    assert len(gateway.snapshot_entries()) == 1


def test_enum_values_are_stable() -> None:
    assert WorkflowStage.DEBUGGING.value == "debugging"
    assert EnvironmentMemoryStatus.STORED.value == "stored"
    assert EnvironmentMemoryDecision.CONTINUE_WORKFLOW.value == "continue_workflow"


def _runtime_trust() -> TrustCalibration:
    result_runtime = EnvironmentMemoryRuntime()
    session = result_runtime.create_session(workspace_id="workspace")
    result = result_runtime.store_workflow(
        session_id=session.session_id,
        app_name="VS Code",
        active_files=("main.py",),
    )
    assert result.entry is not None
    return result.entry.trust