from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    CursorPosition,
    EnvironmentMemoryRuntime,
    EnvironmentWorkspaceMemoryEntry,
    TaskContinuity,
    WorkflowCognitionDecision,
    WorkflowCognitionReason,
    WorkflowCognitionRuntime,
    WorkflowCognitionStatus,
    WorkflowConfidenceBand,
    WorkflowContext,
    WorkflowIntentionModel,
    WorkflowKind,
    WorkflowResumePlan,
    WorkflowSignal,
    WorkflowSignalKind,
    WorkflowStage,
)


def test_runtime_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        WorkflowCognitionRuntime(name=" ")


def test_signal_rejects_empty_reason() -> None:
    with pytest.raises(ValidationError):
        WorkflowSignal(
            kind=WorkflowSignalKind.INTENT_SIGNAL,
            workflow=WorkflowKind.CODING,
            weight=0.5,
            reason=" ",
        )


def test_context_requires_signals() -> None:
    entry = _entry(stage=WorkflowStage.CODING)
    continuity = TaskContinuity(
        continuity_token=entry.continuity_token,
        workflow_stage=entry.workflow_stage,
    )

    with pytest.raises(ValidationError):
        WorkflowContext(
            workspace_id=entry.workspace_id,
            app_name=entry.app_name,
            memory_entry=entry,
            task_continuity=continuity,
            intention=_intention(),
            signals=(),
            summary="invalid",
        )


def test_resume_plan_requires_actions() -> None:
    with pytest.raises(ValidationError):
        WorkflowResumePlan(
            workflow=WorkflowKind.CODING,
            continuity_token="token",
            resume_summary="resume",
            suggested_next_actions=(),
            requires_user_confirmation=False,
            confidence=0.8,
        )


def test_create_session() -> None:
    runtime = WorkflowCognitionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    assert session.workspace_id == "workspace"
    assert runtime.snapshot().session_count == 1


def test_detects_debugging_from_visible_error() -> None:
    runtime = WorkflowCognitionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.analyze(
        session_id=session.session_id,
        memory_entry=_entry(
            stage=WorkflowStage.DEBUGGING,
            visible_errors=("AssertionError in test_environment.py",),
            pending_todos=("fix failing test",),
        ),
    )

    assert result.status == WorkflowCognitionStatus.UNDERSTOOD
    assert result.context is not None
    assert result.context.intention.primary_workflow == WorkflowKind.DEBUGGING
    assert result.prediction is not None
    assert result.prediction.next_workflow == WorkflowKind.TESTING


def test_detects_testing_from_commands() -> None:
    runtime = WorkflowCognitionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.analyze(
        session_id=session.session_id,
        memory_entry=_entry(
            stage=WorkflowStage.UNKNOWN,
            recent_commands=("pytest tests/test_environment.py -vv",),
            active_files=("tests/test_environment.py",),
        ),
    )

    assert result.context is not None
    assert result.context.intention.primary_workflow == WorkflowKind.TESTING


def test_detects_coding_from_source_file() -> None:
    runtime = WorkflowCognitionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.analyze(
        session_id=session.session_id,
        memory_entry=_entry(
            stage=WorkflowStage.UNKNOWN,
            active_files=("jarvis/environment/workflow_cognition.py",),
        ),
    )

    assert result.context is not None
    assert result.context.intention.primary_workflow == WorkflowKind.CODING


def test_detects_researching_from_user_intent() -> None:
    runtime = WorkflowCognitionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.analyze(
        session_id=session.session_id,
        memory_entry=_entry(stage=WorkflowStage.UNKNOWN),
        user_intent="research best latency pipeline",
    )

    assert result.context is not None
    assert result.context.intention.primary_workflow == WorkflowKind.RESEARCHING


def test_detects_writing_from_file_and_todo() -> None:
    runtime = WorkflowCognitionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.analyze(
        session_id=session.session_id,
        memory_entry=_entry(
            stage=WorkflowStage.WRITING,
            active_files=("design_notes.md",),
            pending_todos=("write final section",),
        ),
    )

    assert result.context is not None
    assert result.context.intention.primary_workflow == WorkflowKind.WRITING


def test_detects_reviewing_from_command() -> None:
    runtime = WorkflowCognitionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.analyze(
        session_id=session.session_id,
        memory_entry=_entry(
            stage=WorkflowStage.UNKNOWN,
            recent_commands=("git diff -- jarvis/environment",),
        ),
    )

    assert result.context is not None
    assert result.context.intention.primary_workflow == WorkflowKind.REVIEWING


def test_detects_deploying_from_command() -> None:
    runtime = WorkflowCognitionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.analyze(
        session_id=session.session_id,
        memory_entry=_entry(
            stage=WorkflowStage.UNKNOWN,
            recent_commands=("docker push jarvis:latest",),
        ),
    )

    assert result.context is not None
    assert result.context.intention.primary_workflow == WorkflowKind.DEPLOYING


def test_low_confidence_asks_user() -> None:
    runtime = WorkflowCognitionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.analyze(
        session_id=session.session_id,
        memory_entry=_entry(
            stage=WorkflowStage.UNKNOWN,
            active_files=("unknown.bin",),
        ),
    )

    assert result.status == WorkflowCognitionStatus.LOW_CONFIDENCE
    assert result.decision == WorkflowCognitionDecision.ASK_USER


def test_plan_resume_creates_resume_plan() -> None:
    runtime = WorkflowCognitionRuntime()
    session = runtime.create_session(workspace_id="workspace")
    analyzed = runtime.analyze(
        session_id=session.session_id,
        memory_entry=_entry(
            stage=WorkflowStage.DEBUGGING,
            visible_errors=("TypeError line 47",),
            active_files=("main.py",),
            cursor_positions=(
                CursorPosition(file_path="main.py", line=47, column=9),
            ),
        ),
    )

    assert analyzed.context is not None

    resumed = runtime.plan_resume(
        session_id=session.session_id,
        context=analyzed.context,
    )

    assert resumed.status == WorkflowCognitionStatus.RESUME_READY
    assert resumed.decision == WorkflowCognitionDecision.RESUME
    assert resumed.resume_plan is not None
    assert resumed.resume_plan.suggested_next_actions


def test_missing_session_fails() -> None:
    runtime = WorkflowCognitionRuntime()

    result = runtime.analyze(
        session_id="missing",
        memory_entry=_entry(stage=WorkflowStage.CODING),
    )

    assert result.status == WorkflowCognitionStatus.FAILED
    assert result.reason == WorkflowCognitionReason.SESSION_NOT_FOUND


def test_snapshot_tracks_counts() -> None:
    runtime = WorkflowCognitionRuntime()
    session = runtime.create_session(workspace_id="workspace")
    understood = runtime.analyze(
        session_id=session.session_id,
        memory_entry=_entry(stage=WorkflowStage.DEBUGGING),
    )

    assert understood.context is not None

    runtime.plan_resume(
        session_id=session.session_id,
        context=understood.context,
    )
    runtime.analyze(
        session_id=session.session_id,
        memory_entry=_entry(
            stage=WorkflowStage.UNKNOWN,
            active_files=("unknown.bin",),
        ),
    )
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.result_count == 3
    assert snapshot.understood_count == 1
    assert snapshot.resume_count == 1
    assert snapshot.low_confidence_count == 1


def test_session_tracks_counts() -> None:
    runtime = WorkflowCognitionRuntime()
    session = runtime.create_session(workspace_id="workspace")
    analyzed = runtime.analyze(
        session_id=session.session_id,
        memory_entry=_entry(stage=WorkflowStage.CODING),
    )

    assert analyzed.context is not None

    runtime.plan_resume(
        session_id=session.session_id,
        context=analyzed.context,
    )
    stored = runtime.session_for(session.session_id)

    assert stored is not None
    assert stored.analyze_count == 1
    assert stored.resume_count == 1


def test_reset_clears_runtime() -> None:
    runtime = WorkflowCognitionRuntime()
    runtime.create_session(workspace_id="workspace")

    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.runtime_event_count == 1
    assert snapshot.last_reason == WorkflowCognitionReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert WorkflowKind.DEBUGGING.value == "debugging"
    assert WorkflowCognitionStatus.UNDERSTOOD.value == "understood"
    assert WorkflowConfidenceBand.HIGH.value == "high"


def _entry(
    *,
    stage: WorkflowStage,
    active_files: tuple[str, ...] = (),
    cursor_positions: tuple[CursorPosition, ...] = (),
    recent_commands: tuple[str, ...] = (),
    visible_errors: tuple[str, ...] = (),
    pending_todos: tuple[str, ...] = (),
) -> EnvironmentWorkspaceMemoryEntry:
    memory = EnvironmentMemoryRuntime()
    session = memory.create_session(workspace_id="workspace")
    result = memory.store_workflow(
        session_id=session.session_id,
        app_name="VS Code",
        project_path="E:/JARVIS_OS",
        active_files=active_files,
        cursor_positions=cursor_positions,
        terminal_directory="E:/JARVIS_OS",
        recent_commands=recent_commands,
        visible_errors=visible_errors,
        pending_todos=pending_todos,
        workflow_stage=stage,
    )

    assert result.entry is not None
    return result.entry


def _intention() -> WorkflowIntentionModel:
    return WorkflowIntentionModel(
        primary_workflow=WorkflowKind.CODING,
        confidence=0.8,
        confidence_band=WorkflowConfidenceBand.HIGH,
        inferred_goal="continue coding",
    )