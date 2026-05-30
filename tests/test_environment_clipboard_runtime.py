from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    ClipboardDecision,
    ClipboardHashPhase,
    ClipboardHijackStatus,
    ClipboardOperationKind,
    ClipboardReason,
    ClipboardRuntime,
    ClipboardSensitivityKind,
    ClipboardSensitivityLevel,
    ClipboardStatus,
)


def test_runtime_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        ClipboardRuntime(name=" ")


def test_create_session() -> None:
    runtime = ClipboardRuntime()
    session = runtime.create_session(workspace_id="workspace")

    assert session.workspace_id == "workspace"
    assert runtime.snapshot().session_count == 1


def test_record_hash_never_logs_raw_content() -> None:
    runtime = ClipboardRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.record_hash(
        session_id=session.session_id,
        clipboard_text="hello world",
    )

    assert result.status == ClipboardStatus.VERIFIED
    assert result.hash_record is not None
    assert result.hash_record.content_length == len("hello world")
    assert result.audit.raw_content_logged is False
    assert result.audit.content_hash == result.hash_record.content_hash


def test_empty_clipboard_blocks_hashing() -> None:
    runtime = ClipboardRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.record_hash(
        session_id=session.session_id,
        clipboard_text="",
    )

    assert result.status == ClipboardStatus.BLOCKED
    assert result.reason == ClipboardReason.EMPTY_CLIPBOARD_BLOCKED


def test_prepare_paste_allows_safe_content() -> None:
    runtime = ClipboardRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.prepare_paste(
        session_id=session.session_id,
        clipboard_text="normal safe text",
        target_field_known=True,
        focus_known=True,
    )

    assert result.status == ClipboardStatus.READY
    assert result.decision == ClipboardDecision.ALLOW
    assert result.safe_for_physical_paste is True
    assert result.scan is not None
    assert result.scan.highest_level == ClipboardSensitivityLevel.SAFE


def test_prepare_paste_blocks_unknown_field() -> None:
    runtime = ClipboardRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.prepare_paste(
        session_id=session.session_id,
        clipboard_text="normal safe text",
        target_field_known=False,
        focus_known=True,
    )

    assert result.status == ClipboardStatus.BLOCKED
    assert result.reason == ClipboardReason.UNKNOWN_FIELD_BLOCKED
    assert result.safe_for_physical_paste is False


def test_prepare_paste_blocks_uncertain_focus() -> None:
    runtime = ClipboardRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.prepare_paste(
        session_id=session.session_id,
        clipboard_text="normal safe text",
        target_field_known=True,
        focus_known=False,
    )

    assert result.status == ClipboardStatus.BLOCKED
    assert result.reason == ClipboardReason.FOCUS_UNCERTAIN_BLOCKED
    assert result.safe_for_physical_paste is False


def test_scanner_detects_api_key_without_raw_logging() -> None:
    runtime = ClipboardRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.prepare_paste(
        session_id=session.session_id,
        clipboard_text="api_key = abcdefghijklmnop1234567890",
        target_field_known=True,
        focus_known=True,
    )

    assert result.status == ClipboardStatus.BLOCKED
    assert result.reason == ClipboardReason.SENSITIVE_CONTENT_BLOCKED
    assert result.scan is not None
    assert result.scan.finding_count == 1
    assert result.scan.findings[0].kind == ClipboardSensitivityKind.API_KEY
    assert result.scan.findings[0].evidence_hash is not None
    assert result.audit.raw_content_logged is False


def test_scanner_detects_password() -> None:
    runtime = ClipboardRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.prepare_paste(
        session_id=session.session_id,
        clipboard_text="password = super-secret-value",
        target_field_known=True,
        focus_known=True,
    )

    assert result.status == ClipboardStatus.BLOCKED
    assert result.scan is not None
    assert result.scan.highest_level == ClipboardSensitivityLevel.HIGH


def test_scanner_detects_private_key_as_critical() -> None:
    runtime = ClipboardRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.prepare_paste(
        session_id=session.session_id,
        clipboard_text="-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----",
        target_field_known=True,
        focus_known=True,
    )

    assert result.status == ClipboardStatus.BLOCKED
    assert result.scan is not None
    assert result.scan.highest_level == ClipboardSensitivityLevel.CRITICAL


def test_sensitive_content_can_require_approval_when_explicitly_allowed() -> None:
    runtime = ClipboardRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.prepare_paste(
        session_id=session.session_id,
        clipboard_text="token = abcdefghijklmnopqrstuvwxyz123456",
        target_field_known=True,
        focus_known=True,
        allow_sensitive=True,
    )

    assert result.status == ClipboardStatus.NEEDS_APPROVAL
    assert result.decision == ClipboardDecision.REQUIRE_APPROVAL
    assert result.reason == ClipboardReason.SENSITIVE_CONTENT_REQUIRES_APPROVAL
    assert result.safe_for_physical_paste is False


def test_hash_verification_passes_when_unchanged() -> None:
    runtime = ClipboardRuntime()
    session = runtime.create_session(workspace_id="workspace")
    prepared = runtime.prepare_paste(
        session_id=session.session_id,
        clipboard_text="safe text",
        target_field_known=True,
        focus_known=True,
    )

    assert prepared.hash_record is not None

    result = runtime.verify_clipboard_hash(
        session_id=session.session_id,
        expected_hash=prepared.hash_record.content_hash,
        observed_clipboard_text="safe text",
        phase=ClipboardHashPhase.BEFORE_PASTE,
    )

    assert result.status == ClipboardStatus.VERIFIED
    assert result.reason == ClipboardReason.CLIPBOARD_HASH_VERIFIED
    assert result.hijack_report is not None
    assert result.hijack_report.status == ClipboardHijackStatus.CLEAN


def test_hash_verification_detects_hijack_before_paste() -> None:
    runtime = ClipboardRuntime()
    session = runtime.create_session(workspace_id="workspace")
    prepared = runtime.prepare_paste(
        session_id=session.session_id,
        clipboard_text="safe text",
        target_field_known=True,
        focus_known=True,
    )

    assert prepared.hash_record is not None

    result = runtime.verify_clipboard_hash(
        session_id=session.session_id,
        expected_hash=prepared.hash_record.content_hash,
        observed_clipboard_text="attacker replaced clipboard",
        phase=ClipboardHashPhase.BEFORE_PASTE,
    )

    assert result.status == ClipboardStatus.HIJACK_DETECTED
    assert result.decision == ClipboardDecision.ABORT
    assert result.reason == ClipboardReason.CLIPBOARD_HIJACK_DETECTED


def test_hash_verification_detects_hijack_after_paste() -> None:
    runtime = ClipboardRuntime()
    session = runtime.create_session(workspace_id="workspace")
    prepared = runtime.prepare_paste(
        session_id=session.session_id,
        clipboard_text="safe text",
        target_field_known=True,
        focus_known=True,
    )

    assert prepared.hash_record is not None

    result = runtime.verify_clipboard_hash(
        session_id=session.session_id,
        expected_hash=prepared.hash_record.content_hash,
        observed_clipboard_text="changed after paste",
        phase=ClipboardHashPhase.AFTER_PASTE,
    )

    assert result.status == ClipboardStatus.HIJACK_DETECTED
    assert result.hijack_report is not None
    assert result.hijack_report.phase == ClipboardHashPhase.AFTER_PASTE


def test_missing_session_fails() -> None:
    runtime = ClipboardRuntime()

    result = runtime.prepare_paste(
        session_id="missing",
        clipboard_text="safe text",
        target_field_known=True,
        focus_known=True,
    )

    assert result.status == ClipboardStatus.FAILED
    assert result.reason == ClipboardReason.SESSION_NOT_FOUND


def test_invalid_hash_record_rejects_bad_hash() -> None:
    from jarvis.environment import ClipboardHashRecord

    with pytest.raises(ValidationError):
        ClipboardHashRecord(content_hash="bad", content_length=3)


def test_audit_rejects_raw_content_logged() -> None:
    from jarvis.environment import ClipboardAuditRecord

    with pytest.raises(ValidationError):
        ClipboardAuditRecord(
            operation=ClipboardOperationKind.READ_HASH,
            status=ClipboardStatus.VERIFIED,
            decision=ClipboardDecision.ALLOW,
            reason=ClipboardReason.HASH_RECORDED,
            raw_content_logged=True,
        )


def test_snapshot_tracks_counts() -> None:
    runtime = ClipboardRuntime()
    session = runtime.create_session(workspace_id="workspace")

    runtime.prepare_paste(
        session_id=session.session_id,
        clipboard_text="safe text",
        target_field_known=True,
        focus_known=True,
    )
    runtime.prepare_paste(
        session_id=session.session_id,
        clipboard_text="password = secret",
        target_field_known=True,
        focus_known=True,
    )
    prepared = runtime.prepare_paste(
        session_id=session.session_id,
        clipboard_text="another safe text",
        target_field_known=True,
        focus_known=True,
    )

    assert prepared.hash_record is not None

    runtime.verify_clipboard_hash(
        session_id=session.session_id,
        expected_hash=prepared.hash_record.content_hash,
        observed_clipboard_text="hijacked",
        phase=ClipboardHashPhase.BEFORE_PASTE,
    )

    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.result_count == 4
    assert snapshot.safe_paste_count == 2
    assert snapshot.blocked_count == 2
    assert snapshot.hijack_count == 1
    assert snapshot.audit_count == 4


def test_session_tracks_operation_count_and_hijack_count() -> None:
    runtime = ClipboardRuntime()
    session = runtime.create_session(workspace_id="workspace")
    prepared = runtime.prepare_paste(
        session_id=session.session_id,
        clipboard_text="safe text",
        target_field_known=True,
        focus_known=True,
    )

    assert prepared.hash_record is not None

    runtime.verify_clipboard_hash(
        session_id=session.session_id,
        expected_hash=prepared.hash_record.content_hash,
        observed_clipboard_text="changed",
        phase=ClipboardHashPhase.BEFORE_PASTE,
    )

    stored = runtime.session_for(session.session_id)

    assert stored is not None
    assert stored.operation_count == 2
    assert stored.hijack_count == 1


def test_reset_clears_runtime() -> None:
    runtime = ClipboardRuntime()
    runtime.create_session(workspace_id="workspace")

    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.runtime_event_count == 1
    assert snapshot.last_reason == ClipboardReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert ClipboardOperationKind.PREPARE_PASTE.value == "prepare_paste"
    assert ClipboardStatus.HIJACK_DETECTED.value == "hijack_detected"
    assert ClipboardSensitivityKind.API_KEY.value == "api_key"