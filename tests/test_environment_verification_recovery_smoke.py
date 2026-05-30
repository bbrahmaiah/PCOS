from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    VerificationRecoverySmokeCheckKind,
    VerificationRecoverySmokeGateRuntime,
    VerificationRecoverySmokeReason,
    VerificationRecoverySmokeReport,
    VerificationRecoverySmokeStatus,
)


def test_runtime_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        VerificationRecoverySmokeGateRuntime(name=" ")


def test_create_session() -> None:
    runtime = VerificationRecoverySmokeGateRuntime()
    session = runtime.create_session(workspace_id="workspace")

    assert session.workspace_id == "workspace"
    assert runtime.snapshot().session_count == 1


def test_report_rejects_sealed_with_failure() -> None:
    runtime = VerificationRecoverySmokeGateRuntime()
    failed = runtime.check_hallucinated_success_impossible().model_copy(
        update={
            "status": VerificationRecoverySmokeStatus.FAILED,
            "reason": VerificationRecoverySmokeReason.CHECK_FAILED,
        }
    )

    with pytest.raises(ValidationError):
        VerificationRecoverySmokeReport(
            checks=(failed,),
            passed_count=0,
            failed_count=1,
            blocked_count=0,
            sealed=True,
            reason=VerificationRecoverySmokeReason.GATE_PASSED,
            message="invalid sealed report",
        )


def test_button_click_verified() -> None:
    runtime = VerificationRecoverySmokeGateRuntime()
    result = runtime.check_button_click_verified()

    assert result.kind == (
        VerificationRecoverySmokeCheckKind.BUTTON_CLICK_VERIFIED
    )
    assert result.status == VerificationRecoverySmokeStatus.PASSED


def test_text_typed_verified() -> None:
    runtime = VerificationRecoverySmokeGateRuntime()
    result = runtime.check_text_typed_verified()

    assert result.kind == VerificationRecoverySmokeCheckKind.TEXT_TYPED_VERIFIED
    assert result.status == VerificationRecoverySmokeStatus.PASSED


def test_app_launch_verified() -> None:
    runtime = VerificationRecoverySmokeGateRuntime()
    result = runtime.check_app_launch_verified()

    assert result.kind == VerificationRecoverySmokeCheckKind.APP_LAUNCH_VERIFIED
    assert result.status == VerificationRecoverySmokeStatus.PASSED


def test_wrong_target_detected() -> None:
    runtime = VerificationRecoverySmokeGateRuntime()
    result = runtime.check_wrong_target_detected()

    assert result.kind == VerificationRecoverySmokeCheckKind.WRONG_TARGET_DETECTED
    assert result.status == VerificationRecoverySmokeStatus.PASSED


def test_partial_success_is_not_success() -> None:
    runtime = VerificationRecoverySmokeGateRuntime()
    result = runtime.check_partial_success_is_not_success()

    assert result.kind == (
        VerificationRecoverySmokeCheckKind.PARTIAL_SUCCESS_IS_NOT_SUCCESS
    )
    assert result.status == VerificationRecoverySmokeStatus.PASSED


def test_divergence_detected() -> None:
    runtime = VerificationRecoverySmokeGateRuntime()
    result = runtime.check_divergence_detected()

    assert result.kind == VerificationRecoverySmokeCheckKind.DIVERGENCE_DETECTED
    assert result.status == VerificationRecoverySmokeStatus.PASSED


def test_resync_works() -> None:
    runtime = VerificationRecoverySmokeGateRuntime()
    result = runtime.check_resync_works()

    assert result.kind == VerificationRecoverySmokeCheckKind.RESYNC_WORKS
    assert result.status == VerificationRecoverySmokeStatus.PASSED


def test_recovery_retries_safely() -> None:
    runtime = VerificationRecoverySmokeGateRuntime()
    result = runtime.check_recovery_retries_safely()

    assert result.kind == (
        VerificationRecoverySmokeCheckKind.RECOVERY_RETRIES_SAFELY
    )
    assert result.status == VerificationRecoverySmokeStatus.PASSED


def test_rollback_works() -> None:
    runtime = VerificationRecoverySmokeGateRuntime()
    result = runtime.check_rollback_works()

    assert result.kind == VerificationRecoverySmokeCheckKind.ROLLBACK_WORKS
    assert result.status == VerificationRecoverySmokeStatus.PASSED


def test_hallucinated_success_impossible() -> None:
    runtime = VerificationRecoverySmokeGateRuntime()
    result = runtime.check_hallucinated_success_impossible()

    assert result.kind == (
        VerificationRecoverySmokeCheckKind.HALLUCINATED_SUCCESS_IMPOSSIBLE
    )
    assert result.status == VerificationRecoverySmokeStatus.PASSED


def test_run_seals_gate() -> None:
    runtime = VerificationRecoverySmokeGateRuntime()
    session = runtime.create_session(workspace_id="workspace")

    report = runtime.run(session_id=session.session_id)

    assert report.sealed is True
    assert report.passed_count == 10
    assert report.failed_count == 0
    assert report.blocked_count == 0
    assert len(report.checks) == 10


def test_missing_session_blocks_gate() -> None:
    runtime = VerificationRecoverySmokeGateRuntime()

    report = runtime.run(session_id="missing")

    assert report.sealed is False
    assert report.blocked_count == 1
    assert report.reason == VerificationRecoverySmokeReason.GATE_FAILED


def test_snapshot_tracks_reports_checks_and_events() -> None:
    runtime = VerificationRecoverySmokeGateRuntime()
    session = runtime.create_session(workspace_id="workspace")

    runtime.run(session_id=session.session_id)
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.report_count == 1
    assert snapshot.check_count == 10
    assert snapshot.passed_count == 10
    assert snapshot.failed_count == 0
    assert snapshot.blocked_count == 0
    assert snapshot.sealed_report_count == 1
    assert snapshot.runtime_event_count == 2


def test_session_tracks_report_and_check_counts() -> None:
    runtime = VerificationRecoverySmokeGateRuntime()
    session = runtime.create_session(workspace_id="workspace")

    runtime.run(session_id=session.session_id)
    stored = runtime.session_for(session.session_id)

    assert stored is not None
    assert stored.report_count == 1
    assert stored.check_count == 10


def test_reset_clears_runtime() -> None:
    runtime = VerificationRecoverySmokeGateRuntime()
    runtime.create_session(workspace_id="workspace")

    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.runtime_event_count == 1
    assert snapshot.last_reason == VerificationRecoverySmokeReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert (
        VerificationRecoverySmokeCheckKind.ROLLBACK_WORKS.value
        == "rollback_works"
    )
    assert VerificationRecoverySmokeStatus.PASSED.value == "passed"
    assert VerificationRecoverySmokeReason.GATE_PASSED.value == "gate_passed"