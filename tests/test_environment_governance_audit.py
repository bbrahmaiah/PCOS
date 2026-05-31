from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    EnvironmentAttackVector,
    EnvironmentGovernanceAuditor,
    EnvironmentGovernancePolicy,
    EnvironmentSource,
    GovernanceAttackSample,
    GovernanceAuditDecision,
    GovernanceAuditReason,
    GovernanceAuditReport,
    GovernanceAuditStatus,
    GovernanceControlKind,
    GovernanceControlResult,
    GovernanceRiskLevel,
    SafetyEnvironmentGovernanceAuditRuntime,
    TrustCalibration,
    default_governance_attack_samples,
)


def test_runtime_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        SafetyEnvironmentGovernanceAuditRuntime(name=" ")


def test_attack_sample_requires_description() -> None:
    with pytest.raises(ValidationError):
        GovernanceAttackSample(
            vector=EnvironmentAttackVector.UI_SPOOFING,
            description=" ",
        )


def test_vector_result_requires_controls() -> None:
    from jarvis.environment import GovernanceVectorAuditResult

    sample = GovernanceAttackSample(
        vector=EnvironmentAttackVector.UI_SPOOFING,
        description="fake UI",
    )

    with pytest.raises(ValidationError):
        GovernanceVectorAuditResult(
            sample=sample,
            status=GovernanceAuditStatus.BLOCKED,
            decision=GovernanceAuditDecision.BLOCK,
            reason=GovernanceAuditReason.UI_SPOOFING_BLOCKED,
            controls=(),
            blocked=True,
            confidence=0.9,
            message="blocked",
        )


def test_expected_block_vector_cannot_be_unblocked() -> None:
    from jarvis.environment import GovernanceVectorAuditResult

    sample = GovernanceAttackSample(
        vector=EnvironmentAttackVector.CLIPBOARD_HIJACK,
        description="clipboard changed",
        expected_block=True,
    )

    with pytest.raises(ValidationError):
        GovernanceVectorAuditResult(
            sample=sample,
            status=GovernanceAuditStatus.FAILED,
            decision=GovernanceAuditDecision.FAIL,
            reason=GovernanceAuditReason.AUDIT_FAILED,
            controls=(
                GovernanceControlResult(
                    kind=GovernanceControlKind.POLICY_GATE,
                    passed=True,
                    reason="policy checked",
                ),
            ),
            blocked=False,
            confidence=0.1,
            message="not blocked",
        )


def test_report_rejects_bad_counts() -> None:
    with pytest.raises(ValidationError):
        GovernanceAuditReport(
            status=GovernanceAuditStatus.PASSED,
            decision=GovernanceAuditDecision.ALLOW,
            reason=GovernanceAuditReason.AUDIT_PASSED,
            vector_results=(),
            passed_count=1,
            blocked_count=0,
            failed_count=0,
            trust=_trust_for_test(),
            message="bad counts",
        )


def test_create_session() -> None:
    runtime = SafetyEnvironmentGovernanceAuditRuntime()
    session = runtime.create_session(workspace_id="workspace")

    assert session.workspace_id == "workspace"
    assert runtime.snapshot().session_count == 1


@pytest.mark.parametrize(
    ("vector", "reason"),
    [
        (
            EnvironmentAttackVector.VISUAL_PROMPT_INJECTION,
            GovernanceAuditReason.VISUAL_PROMPT_INJECTION_BLOCKED,
        ),
        (
            EnvironmentAttackVector.UI_SPOOFING,
            GovernanceAuditReason.UI_SPOOFING_BLOCKED,
        ),
        (
            EnvironmentAttackVector.COORDINATE_MANIPULATION,
            GovernanceAuditReason.COORDINATE_MANIPULATION_BLOCKED,
        ),
        (
            EnvironmentAttackVector.CLIPBOARD_HIJACK,
            GovernanceAuditReason.CLIPBOARD_HIJACK_BLOCKED,
        ),
        (
            EnvironmentAttackVector.APP_IMPERSONATION,
            GovernanceAuditReason.APP_IMPERSONATION_BLOCKED,
        ),
        (
            EnvironmentAttackVector.KEYSTROKE_INJECTION,
            GovernanceAuditReason.KEYSTROKE_INJECTION_BLOCKED,
        ),
        (
            EnvironmentAttackVector.FAKE_APPROVAL_DIALOG,
            GovernanceAuditReason.FAKE_APPROVAL_DIALOG_BLOCKED,
        ),
        (
            EnvironmentAttackVector.OCR_COMMAND_INJECTION,
            GovernanceAuditReason.OCR_COMMAND_INJECTION_BLOCKED,
        ),
        (
            EnvironmentAttackVector.PRIVACY_ZONE_VIOLATION,
            GovernanceAuditReason.PRIVACY_ZONE_VIOLATION_BLOCKED,
        ),
        (
            EnvironmentAttackVector.POLICY_BYPASS_VISUAL_CONTENT,
            GovernanceAuditReason.POLICY_BYPASS_VISUAL_CONTENT_BLOCKED,
        ),
        (
            EnvironmentAttackVector.MALICIOUS_MODAL_SPOOFING,
            GovernanceAuditReason.MALICIOUS_MODAL_SPOOFING_BLOCKED,
        ),
    ],
)
def test_each_attack_vector_is_blocked(
    vector: EnvironmentAttackVector,
    reason: GovernanceAuditReason,
) -> None:
    auditor = EnvironmentGovernanceAuditor()
    result = auditor.audit_vector(
        GovernanceAttackSample(
            vector=vector,
            description=f"attack sample for {vector.value}",
            risk_level=GovernanceRiskLevel.CRITICAL,
        )
    )

    assert result.status == GovernanceAuditStatus.BLOCKED
    assert result.decision == GovernanceAuditDecision.BLOCK
    assert result.reason == reason
    assert result.blocked is True
    assert result.controls


def test_full_audit_passes_all_default_vectors() -> None:
    runtime = SafetyEnvironmentGovernanceAuditRuntime()
    session = runtime.create_session(workspace_id="workspace")

    report = runtime.run_full_audit(session_id=session.session_id)

    assert report.status == GovernanceAuditStatus.PASSED
    assert report.decision == GovernanceAuditDecision.ALLOW
    assert report.reason == GovernanceAuditReason.AUDIT_PASSED
    assert report.failed_count == 0
    assert report.blocked_count == 11
    assert report.passed_count == 11


def test_default_attack_samples_cover_all_required_vectors() -> None:
    samples = default_governance_attack_samples(workspace_id="workspace")
    vectors = {sample.vector for sample in samples}

    assert vectors == {
        EnvironmentAttackVector.VISUAL_PROMPT_INJECTION,
        EnvironmentAttackVector.UI_SPOOFING,
        EnvironmentAttackVector.COORDINATE_MANIPULATION,
        EnvironmentAttackVector.CLIPBOARD_HIJACK,
        EnvironmentAttackVector.APP_IMPERSONATION,
        EnvironmentAttackVector.KEYSTROKE_INJECTION,
        EnvironmentAttackVector.FAKE_APPROVAL_DIALOG,
        EnvironmentAttackVector.OCR_COMMAND_INJECTION,
        EnvironmentAttackVector.PRIVACY_ZONE_VIOLATION,
        EnvironmentAttackVector.POLICY_BYPASS_VISUAL_CONTENT,
        EnvironmentAttackVector.MALICIOUS_MODAL_SPOOFING,
    }


def test_policy_can_fail_vector_when_control_disabled() -> None:
    auditor = EnvironmentGovernanceAuditor(
        policy=EnvironmentGovernancePolicy(block_clipboard_hijack=False)
    )
    sample = GovernanceAttackSample(
        vector=EnvironmentAttackVector.CLIPBOARD_HIJACK,
        description="clipboard changed after hash",
    )

    with pytest.raises(ValidationError):
        auditor.audit_vector(sample)


def test_missing_session_vector_audit_fails() -> None:
    runtime = SafetyEnvironmentGovernanceAuditRuntime()
    sample = GovernanceAttackSample(
        vector=EnvironmentAttackVector.UI_SPOOFING,
        description="spoofed UI",
    )

    result = runtime.audit_vector(session_id="missing", sample=sample)

    assert result.status == GovernanceAuditStatus.FAILED
    assert result.reason == GovernanceAuditReason.SESSION_NOT_FOUND


def test_missing_session_full_audit_fails() -> None:
    runtime = SafetyEnvironmentGovernanceAuditRuntime()

    report = runtime.run_full_audit(session_id="missing")

    assert report.status == GovernanceAuditStatus.FAILED
    assert report.reason == GovernanceAuditReason.SESSION_NOT_FOUND


def test_snapshot_tracks_counts() -> None:
    runtime = SafetyEnvironmentGovernanceAuditRuntime()
    session = runtime.create_session(workspace_id="workspace")

    runtime.run_full_audit(session_id=session.session_id)
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.report_count == 1
    assert snapshot.vector_result_count == 11
    assert snapshot.passed_count == 11
    assert snapshot.blocked_count == 11
    assert snapshot.failed_count == 0


def test_session_tracks_counts() -> None:
    runtime = SafetyEnvironmentGovernanceAuditRuntime()
    session = runtime.create_session(workspace_id="workspace")

    runtime.run_full_audit(session_id=session.session_id)
    stored = runtime.session_for(session.session_id)

    assert stored is not None
    assert stored.audit_count == 1
    assert stored.passed_count == 11
    assert stored.blocked_count == 11
    assert stored.failed_count == 0


def test_reset_clears_runtime() -> None:
    runtime = SafetyEnvironmentGovernanceAuditRuntime()
    runtime.create_session(workspace_id="workspace")

    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.runtime_event_count == 1
    assert snapshot.last_reason == GovernanceAuditReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert (
        EnvironmentAttackVector.VISUAL_PROMPT_INJECTION.value
        == "visual_prompt_injection"
    )
    assert GovernanceAuditStatus.PASSED.value == "passed"
    assert GovernanceRiskLevel.CRITICAL.value == "critical"


def _trust_for_test() -> TrustCalibration:
    return TrustCalibration(
        confidence=0.9,
        stability=0.9,
        ambiguity=0.1,
        source=EnvironmentSource.OS_OBSERVER,
        reason="test",
    )