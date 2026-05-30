from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    ActionVerificationStatus,
    EnvironmentRecoveryDecision,
    EnvironmentRecoveryReason,
    EnvironmentRecoveryRequest,
    EnvironmentRecoveryRuntime,
    EnvironmentRecoveryStatus,
    EnvironmentRecoveryStrategyState,
    EnvironmentRetryPolicy,
    EnvironmentRollbackTriggerKind,
    EnvironmentStuckDetector,
    EnvironmentStuckReport,
    RecoveryAttemptKind,
)
from jarvis.environment.recovery_runtime import (
    RecoveryAuditRecord,
    RecoveryHistoryEntry,
)
from jarvis.environment.verification_runtime import (
    RecoveryNeededReason,
    VerificationDecision,
    VerificationResult,
)


def test_runtime_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        EnvironmentRecoveryRuntime(name=" ")


def test_retry_policy_rejects_invalid_total() -> None:
    with pytest.raises(ValidationError):
        EnvironmentRetryPolicy(max_total_recovery_steps=0)


def test_audit_rejects_silent_failure() -> None:
    with pytest.raises(ValidationError):
        RecoveryAuditRecord(
            request_id="request",
            action_id="action",
            status=EnvironmentRecoveryStatus.FAILED,
            decision=EnvironmentRecoveryDecision.BLOCK,
            reason=EnvironmentRecoveryReason.STUCK_DETECTED,
            recovery_needed=True,
            silent_failure=True,
        )


def test_create_session() -> None:
    runtime = EnvironmentRecoveryRuntime()
    session = runtime.create_session(workspace_id="workspace")

    assert session.workspace_id == "workspace"
    assert runtime.snapshot().session_count == 1


def test_verification_already_passed_blocks_recovery() -> None:
    runtime = EnvironmentRecoveryRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.recover(
        _request(
            session_id=session.session_id,
            verification=_verification(
                status=ActionVerificationStatus.PASSED,
                recovery_needed=False,
            ),
        )
    )

    assert result.status == EnvironmentRecoveryStatus.BLOCKED
    assert result.reason == EnvironmentRecoveryReason.VERIFICATION_ALREADY_PASSED
    assert result.retry_allowed is False


def test_first_failure_selects_retry_same() -> None:
    runtime = EnvironmentRecoveryRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.recover(_request(session_id=session.session_id))

    assert result.status == EnvironmentRecoveryStatus.RETRY_READY
    assert result.decision == EnvironmentRecoveryDecision.RETRY
    assert result.reason == EnvironmentRecoveryReason.RETRY_SAME_SELECTED
    assert result.retry_allowed is True
    assert result.plan is not None
    assert result.plan.selected_strategy.kind == RecoveryAttemptKind.RETRY_SAME


def test_after_same_retry_selects_adjusted_retry() -> None:
    runtime = EnvironmentRecoveryRuntime()
    session = runtime.create_session(workspace_id="workspace")
    history = (
        _history(RecoveryAttemptKind.RETRY_SAME),
    )

    result = runtime.recover(
        _request(session_id=session.session_id, history=history)
    )

    assert result.status == EnvironmentRecoveryStatus.RETRY_READY
    assert result.reason == EnvironmentRecoveryReason.RETRY_ADJUSTED_SELECTED
    assert result.plan is not None
    assert result.plan.selected_strategy.kind == RecoveryAttemptKind.RETRY_ADJUSTED


def test_after_adjusted_retry_selects_alternative_path() -> None:
    runtime = EnvironmentRecoveryRuntime()
    session = runtime.create_session(workspace_id="workspace")
    history = (
        _history(RecoveryAttemptKind.RETRY_SAME),
        _history(RecoveryAttemptKind.RETRY_ADJUSTED),
        _history(RecoveryAttemptKind.RETRY_ADJUSTED),
    )

    result = runtime.recover(
        _request(
            session_id=session.session_id,
            history=history,
            alternative_path_available=True,
        )
    )

    assert result.status == EnvironmentRecoveryStatus.PLANNED
    assert result.decision == EnvironmentRecoveryDecision.TRY_ALTERNATIVE
    assert result.reason == EnvironmentRecoveryReason.ALTERNATIVE_PATH_SELECTED


def test_partial_success_report_before_rollback() -> None:
    runtime = EnvironmentRecoveryRuntime()
    session = runtime.create_session(workspace_id="workspace")
    history = _exhausted_history()

    result = runtime.recover(
        _request(
            session_id=session.session_id,
            history=history,
            partial_success_available=True,
            rollback_available=True,
        )
    )

    assert result.status == EnvironmentRecoveryStatus.PLANNED
    assert result.decision == (
        EnvironmentRecoveryDecision.REPORT_PARTIAL_SUCCESS
    )
    assert result.reason == (
        EnvironmentRecoveryReason.PARTIAL_SUCCESS_REPORT_SELECTED
    )
    assert result.plan is not None
    assert result.plan.escalation_level.value == "soft_notify"


def test_rollback_selected_after_retry_exhaustion() -> None:
    runtime = EnvironmentRecoveryRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.recover(
        _request(
            session_id=session.session_id,
            history=_exhausted_history(),
            rollback_available=True,
        )
    )

    assert result.status == EnvironmentRecoveryStatus.ROLLBACK_READY
    assert result.decision == EnvironmentRecoveryDecision.ROLLBACK
    assert result.reason == EnvironmentRecoveryReason.ROLLBACK_SELECTED
    assert result.rollback_required is True
    assert result.plan is not None
    assert result.plan.rollback_trigger.triggered is True


def test_irreversible_risk_prefers_rollback() -> None:
    runtime = EnvironmentRecoveryRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.recover(
        _request(
            session_id=session.session_id,
            irreversible_risk=True,
            rollback_available=True,
        )
    )

    assert result.status == EnvironmentRecoveryStatus.ROLLBACK_READY
    assert result.decision == EnvironmentRecoveryDecision.ROLLBACK
    assert result.plan is not None
    assert result.plan.selected_strategy.requires_user_approval is True


def test_irreversible_risk_without_rollback_escalates() -> None:
    runtime = EnvironmentRecoveryRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.recover(
        _request(
            session_id=session.session_id,
            irreversible_risk=True,
            rollback_available=False,
        )
    )

    assert result.status == EnvironmentRecoveryStatus.ESCALATION_REQUIRED
    assert result.decision == EnvironmentRecoveryDecision.ESCALATE
    assert result.escalation_required is True


def test_escalates_when_no_recovery_path_available() -> None:
    runtime = EnvironmentRecoveryRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.recover(
        _request(
            session_id=session.session_id,
            history=_exhausted_history(),
            rollback_available=False,
            partial_success_available=False,
            alternative_path_available=False,
        )
    )

    assert result.status == EnvironmentRecoveryStatus.ESCALATION_REQUIRED
    assert result.decision == EnvironmentRecoveryDecision.ESCALATE
    assert result.escalation_required is True


def test_stuck_detector_reports_stuck() -> None:
    detector = EnvironmentStuckDetector()
    report = detector.detect(
        history=_exhausted_history(),
        retry_policy=EnvironmentRetryPolicy(),
    )

    assert isinstance(report, EnvironmentStuckReport)
    assert report.stuck is True


def test_missing_session_blocks() -> None:
    runtime = EnvironmentRecoveryRuntime()

    result = runtime.recover(_request(session_id="missing"))

    assert result.status == EnvironmentRecoveryStatus.BLOCKED
    assert result.reason == EnvironmentRecoveryReason.SESSION_NOT_FOUND


def test_snapshot_tracks_counts() -> None:
    runtime = EnvironmentRecoveryRuntime()
    session = runtime.create_session(workspace_id="workspace")

    runtime.recover(_request(session_id=session.session_id))
    runtime.recover(
        _request(
            session_id=session.session_id,
            history=_exhausted_history(),
            rollback_available=True,
        )
    )
    runtime.recover(
        _request(
            session_id=session.session_id,
            history=_exhausted_history(),
        )
    )

    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.result_count == 3
    assert snapshot.retry_count == 1
    assert snapshot.rollback_count == 1
    assert snapshot.escalation_count == 1
    assert snapshot.audit_count == 3


def test_session_tracks_rollback_and_escalation_counts() -> None:
    runtime = EnvironmentRecoveryRuntime()
    session = runtime.create_session(workspace_id="workspace")

    runtime.recover(
        _request(
            session_id=session.session_id,
            history=_exhausted_history(),
            rollback_available=True,
        )
    )
    runtime.recover(
        _request(
            session_id=session.session_id,
            history=_exhausted_history(),
        )
    )

    stored = runtime.session_for(session.session_id)

    assert stored is not None
    assert stored.recovery_count == 2
    assert stored.rollback_count == 1
    assert stored.escalation_count == 1


def test_reset_clears_runtime() -> None:
    runtime = EnvironmentRecoveryRuntime()
    runtime.create_session(workspace_id="workspace")

    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.runtime_event_count == 1
    assert snapshot.last_reason == EnvironmentRecoveryReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert RecoveryAttemptKind.RETRY_SAME.value == "retry_same"
    assert EnvironmentRecoveryStatus.RETRY_READY.value == "retry_ready"
    assert EnvironmentRollbackTriggerKind.STATE_MISMATCH.value == "state_mismatch"
    assert EnvironmentRecoveryStrategyState.AVAILABLE.value == "available"


def _request(
    *,
    session_id: str,
    verification: VerificationResult | None = None,
    history: tuple[RecoveryHistoryEntry, ...] = (),
    irreversible_risk: bool = False,
    partial_success_available: bool = False,
    rollback_available: bool = False,
    alternative_path_available: bool = False,
) -> EnvironmentRecoveryRequest:
    return EnvironmentRecoveryRequest(
        session_id=session_id,
        workspace_id="workspace",
        action_id="action_1",
        verification=verification or _verification(),
        history=history,
        irreversible_risk=irreversible_risk,
        partial_success_available=partial_success_available,
        rollback_available=rollback_available,
        alternative_path_available=alternative_path_available,
    )


def _verification(
    *,
    status: ActionVerificationStatus = ActionVerificationStatus.RECOVERY_NEEDED,
    recovery_needed: bool = True,
) -> VerificationResult:
    return VerificationResult.model_construct(
        result_id="verification_result_test",
        status=status,
        decision=(
            VerificationDecision.COMPLETE
            if status == ActionVerificationStatus.PASSED
            else VerificationDecision.REQUIRE_RECOVERY
        ),
        recovery_needed=recovery_needed,
        recovery_reason=(
            RecoveryNeededReason.NONE
            if not recovery_needed
            else RecoveryNeededReason.STATE_MISMATCH
        ),
    )


def _history(kind: RecoveryAttemptKind) -> RecoveryHistoryEntry:
    return RecoveryHistoryEntry(
        verification_result_id="verification_result_test",
        strategy_kind=kind,
        success=False,
        reason=f"{kind.value} failed",
    )


def _exhausted_history() -> tuple[RecoveryHistoryEntry, ...]:
    return (
        _history(RecoveryAttemptKind.RETRY_SAME),
        _history(RecoveryAttemptKind.RETRY_ADJUSTED),
        _history(RecoveryAttemptKind.RETRY_ADJUSTED),
        _history(RecoveryAttemptKind.ALTERNATIVE_PATH),
    )