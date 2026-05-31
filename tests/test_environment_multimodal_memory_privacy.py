from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    EnvironmentMemoryRuntime,
    EnvironmentMemoryScope,
    EnvironmentMemoryStatus,
    EnvironmentWorkspaceMemoryEntry,
    MemoryPrivacyAuditRecord,
    MemoryPrivacyDecision,
    MemoryPrivacyReason,
    MemoryPrivacyStatus,
    MemoryRedactionPolicy,
    MemoryRetentionKind,
    MultimodalMemoryPrivacyRuntime,
    ProjectMemoryRetention,
    SensitiveUIClassifier,
    SensitiveUIKind,
    WorkflowStage,
)


def test_runtime_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        MultimodalMemoryPrivacyRuntime(name=" ")


def test_redaction_policy_rejects_empty_prefix() -> None:
    with pytest.raises(ValidationError):
        MemoryRedactionPolicy(replacement_prefix=" ")


def test_audit_rejects_raw_sensitive_logging() -> None:
    with pytest.raises(ValidationError):
        MemoryPrivacyAuditRecord(
            status=MemoryPrivacyStatus.BLOCKED,
            decision=MemoryPrivacyDecision.BLOCK,
            reason=MemoryPrivacyReason.PASSWORD_BLOCKED,
            raw_sensitive_logged=True,
        )


def test_lifecycle_record_requires_gateway_write() -> None:
    from jarvis.environment import WorkflowMemoryLifecycleRecord

    entry = _entry(visible_errors=("safe error",))

    with pytest.raises(ValidationError):
        WorkflowMemoryLifecycleRecord(
            entry=entry,
            scope=EnvironmentMemoryScope.SESSION,
            retention_kind=MemoryRetentionKind.SESSION,
            stored_through_gateway=False,
            classification=SensitiveUIClassifier().classify(entry),
        )


def test_project_retention_requires_project_path() -> None:
    with pytest.raises(ValidationError):
        ProjectMemoryRetention(
            workspace_id="workspace",
            project_path=" ",
        )


def test_create_session() -> None:
    runtime = MultimodalMemoryPrivacyRuntime()
    session = runtime.create_session(workspace_id="workspace")

    assert session.workspace_id == "workspace"
    assert runtime.snapshot().session_count == 1


def test_safe_session_memory_stored_through_gateway() -> None:
    runtime = MultimodalMemoryPrivacyRuntime()
    session = runtime.create_session(workspace_id="workspace")
    entry = _entry(visible_errors=("AssertionError line 42",))

    result = runtime.store_memory(
        session_id=session.session_id,
        entry=entry,
        retention_kind=MemoryRetentionKind.SESSION,
    )

    assert result.status == MemoryPrivacyStatus.STORED
    assert result.lifecycle is not None
    assert result.lifecycle.stored_through_gateway is True
    assert len(runtime.gateway_entries()) == 1


def test_password_memory_is_blocked_and_not_written() -> None:
    runtime = MultimodalMemoryPrivacyRuntime()
    session = runtime.create_session(workspace_id="workspace")
    entry = _entry(visible_errors=("password field visible",))

    result = runtime.store_memory(
        session_id=session.session_id,
        entry=entry,
        retention_kind=MemoryRetentionKind.SESSION,
    )

    assert result.status == MemoryPrivacyStatus.BLOCKED
    assert result.reason == MemoryPrivacyReason.PASSWORD_BLOCKED
    assert len(runtime.gateway_entries()) == 0


def test_payment_form_memory_is_blocked_and_not_written() -> None:
    runtime = MultimodalMemoryPrivacyRuntime()
    session = runtime.create_session(workspace_id="workspace")
    entry = _entry(pending_todos=("fill credit card cvv payment form",))

    result = runtime.store_memory(
        session_id=session.session_id,
        entry=entry,
        retention_kind=MemoryRetentionKind.SESSION,
    )

    assert result.status == MemoryPrivacyStatus.BLOCKED
    assert result.reason == MemoryPrivacyReason.PAYMENT_FORM_BLOCKED
    assert len(runtime.gateway_entries()) == 0


def test_private_ui_is_redacted_and_stored() -> None:
    runtime = MultimodalMemoryPrivacyRuntime()
    session = runtime.create_session(workspace_id="workspace")
    entry = _entry(
        visible_errors=("private message from user should be remembered only safely",),
    )

    result = runtime.store_memory(
        session_id=session.session_id,
        entry=entry,
        retention_kind=MemoryRetentionKind.SESSION,
    )

    assert result.status == MemoryPrivacyStatus.REDACTED_STORED
    assert result.entry is not None
    assert result.entry.visible_errors[0].startswith("<redacted:")
    assert "private message" not in result.entry.visible_errors[0]
    assert len(runtime.gateway_entries()) == 1


def test_api_token_is_redacted_and_stored() -> None:
    runtime = MultimodalMemoryPrivacyRuntime()
    session = runtime.create_session(workspace_id="workspace")
    entry = _entry(recent_commands=("curl -H 'Bearer abc.def.secret'",))

    result = runtime.store_memory(
        session_id=session.session_id,
        entry=entry,
        retention_kind=MemoryRetentionKind.PROJECT,
    )

    assert result.status == MemoryPrivacyStatus.REDACTED_STORED
    assert result.reason == MemoryPrivacyReason.TOKEN_REDACTED
    assert result.entry is not None
    assert result.entry.recent_commands[0].startswith("<redacted:")


def test_session_memory_expires() -> None:
    runtime = MultimodalMemoryPrivacyRuntime()
    session = runtime.create_session(workspace_id="workspace")
    entry = _entry(visible_errors=("safe session error",))

    runtime.store_memory(
        session_id=session.session_id,
        entry=entry,
        retention_kind=MemoryRetentionKind.SESSION,
    )
    result = runtime.expire_session_memories(
        session_id=session.session_id,
        now=datetime.max.replace(tzinfo=UTC),
    )

    assert result.status == MemoryPrivacyStatus.EXPIRED
    assert result.expired_count == 1
    assert runtime.snapshot().expired_count == 1
    assert len(runtime.active_lifecycle_records()) == 0


def test_project_memory_persists_until_cleared() -> None:
    runtime = MultimodalMemoryPrivacyRuntime()
    session = runtime.create_session(workspace_id="workspace")
    entry = _entry(visible_errors=("safe project error",))

    runtime.store_memory(
        session_id=session.session_id,
        entry=entry,
        retention_kind=MemoryRetentionKind.PROJECT,
    )
    runtime.expire_session_memories(
        session_id=session.session_id,
        now=datetime.max.replace(tzinfo=UTC),
    )

    assert len(runtime.active_lifecycle_records()) == 1
    assert runtime.snapshot().expired_count == 0


def test_user_can_clear_project_memory() -> None:
    runtime = MultimodalMemoryPrivacyRuntime()
    session = runtime.create_session(workspace_id="workspace")
    entry = _entry(
        project_path="E:/JARVIS_OS",
        visible_errors=("safe project error",),
    )

    runtime.store_memory(
        session_id=session.session_id,
        entry=entry,
        retention_kind=MemoryRetentionKind.PROJECT,
    )
    result = runtime.clear_project_memory(
        session_id=session.session_id,
        project_path="E:/JARVIS_OS",
    )

    assert result.status == MemoryPrivacyStatus.CLEARED
    assert result.cleared_count == 1
    assert len(runtime.active_lifecycle_records()) == 0


def test_missing_session_fails() -> None:
    runtime = MultimodalMemoryPrivacyRuntime()
    entry = _entry(visible_errors=("safe",))

    result = runtime.store_memory(
        session_id="missing",
        entry=entry,
        retention_kind=MemoryRetentionKind.SESSION,
    )

    assert result.status == MemoryPrivacyStatus.FAILED
    assert result.reason == MemoryPrivacyReason.SESSION_NOT_FOUND


def test_classifier_does_not_store_raw_sensitive_content_in_findings() -> None:
    classifier = SensitiveUIClassifier()
    entry = _entry(visible_errors=("password hunter2",))

    classification = classifier.classify(entry)

    assert classification.findings
    assert classification.findings[0].kind == SensitiveUIKind.PASSWORD
    assert classification.findings[0].content_hash is not None
    assert "hunter2" not in classification.findings[0].model_dump_json()


def test_snapshot_tracks_counts() -> None:
    runtime = MultimodalMemoryPrivacyRuntime()
    session = runtime.create_session(workspace_id="workspace")

    runtime.store_memory(
        session_id=session.session_id,
        entry=_entry(visible_errors=("safe",)),
        retention_kind=MemoryRetentionKind.SESSION,
    )
    runtime.store_memory(
        session_id=session.session_id,
        entry=_entry(visible_errors=("private message from friend",)),
        retention_kind=MemoryRetentionKind.SESSION,
    )
    runtime.store_memory(
        session_id=session.session_id,
        entry=_entry(visible_errors=("password field",)),
        retention_kind=MemoryRetentionKind.SESSION,
    )

    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.stored_count == 1
    assert snapshot.redacted_count == 1
    assert snapshot.blocked_count == 1
    assert snapshot.audit_count == 3


def test_session_tracks_counts() -> None:
    runtime = MultimodalMemoryPrivacyRuntime()
    session = runtime.create_session(workspace_id="workspace")

    runtime.store_memory(
        session_id=session.session_id,
        entry=_entry(visible_errors=("safe",)),
        retention_kind=MemoryRetentionKind.SESSION,
    )
    runtime.store_memory(
        session_id=session.session_id,
        entry=_entry(visible_errors=("private message",)),
        retention_kind=MemoryRetentionKind.SESSION,
    )
    runtime.store_memory(
        session_id=session.session_id,
        entry=_entry(visible_errors=("password field",)),
        retention_kind=MemoryRetentionKind.SESSION,
    )

    stored = runtime.session_for(session.session_id)

    assert stored is not None
    assert stored.store_count == 2
    assert stored.redaction_count == 1
    assert stored.blocked_count == 1


def test_reset_clears_runtime_state_not_gateway_history() -> None:
    runtime = MultimodalMemoryPrivacyRuntime()
    session = runtime.create_session(workspace_id="workspace")
    runtime.store_memory(
        session_id=session.session_id,
        entry=_entry(visible_errors=("safe",)),
        retention_kind=MemoryRetentionKind.PROJECT,
    )

    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.runtime_event_count == 1
    assert len(runtime.gateway_entries()) == 1


def test_enum_values_are_stable() -> None:
    assert MemoryPrivacyStatus.REDACTED_STORED.value == "redacted_stored"
    assert MemoryRetentionKind.PROJECT.value == "project"
    assert SensitiveUIKind.PAYMENT_FORM.value == "payment_form"


def _entry(
    *,
    project_path: str = "E:/JARVIS_OS",
    recent_commands: tuple[str, ...] = (),
    visible_errors: tuple[str, ...] = (),
    pending_todos: tuple[str, ...] = (),
) -> EnvironmentWorkspaceMemoryEntry:
    memory = EnvironmentMemoryRuntime()
    session = memory.create_session(workspace_id="workspace")
    result = memory.store_workflow(
        session_id=session.session_id,
        app_name="VS Code",
        project_path=project_path,
        active_files=("main.py",),
        recent_commands=recent_commands,
        visible_errors=visible_errors,
        pending_todos=pending_todos,
        workflow_stage=WorkflowStage.DEBUGGING,
    )

    assert result.status == EnvironmentMemoryStatus.STORED
    assert result.entry is not None
    return result.entry