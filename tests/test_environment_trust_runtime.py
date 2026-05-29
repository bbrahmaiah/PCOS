from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    AmbiguityScore,
    ConfidenceSignal,
    EnvironmentSource,
    EnvironmentTrustLevel,
    PrivacyClassification,
    SourceReliability,
    StabilityScore,
    TrustCalibrationRuntime,
    TrustDecisionKind,
    TrustPolicyClassification,
    TrustRuntimeReason,
    TrustSubjectKind,
)


def test_runtime_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        TrustCalibrationRuntime(name=" ")


def test_confidence_signal_requires_subject_and_reason() -> None:
    with pytest.raises(ValidationError):
        ConfidenceSignal(
            subject_id=" ",
            subject_kind=TrustSubjectKind.OCR_TEXT,
            source=EnvironmentSource.OCR,
            value=0.9,
            reason="ocr confidence",
        )


def test_stability_score_contract() -> None:
    score = StabilityScore(
        subject_id="element",
        subject_kind=TrustSubjectKind.UI_ELEMENT,
        value=0.94,
        sample_count=5,
        reason="stable across frames",
    )

    assert score.value == 0.94
    assert score.sample_count == 5


def test_ambiguity_score_contract() -> None:
    score = AmbiguityScore(
        subject_id="button",
        subject_kind=TrustSubjectKind.GROUNDING_TARGET,
        value=0.70,
        candidate_count=3,
        reason="three matching buttons",
    )

    assert score.candidate_count == 3


def test_source_reliability_updates() -> None:
    runtime = TrustCalibrationRuntime()
    reliability = SourceReliability(
        source=EnvironmentSource.OCR,
        reliability=0.90,
        reason="local OCR tuned for terminal text",
    )

    event = runtime.update_source_reliability(reliability)

    assert event.reason == TrustRuntimeReason.SOURCE_RELIABILITY_UPDATED
    assert runtime.source_reliability_for(EnvironmentSource.OCR) == reliability


def test_record_signal() -> None:
    runtime = TrustCalibrationRuntime()
    signal = ConfidenceSignal(
        subject_id="text-1",
        subject_kind=TrustSubjectKind.OCR_TEXT,
        source=EnvironmentSource.OCR,
        value=0.91,
        reason="clear OCR text",
    )

    event = runtime.record_signal(signal)

    assert event.reason == TrustRuntimeReason.SIGNAL_RECORDED
    assert runtime.signals() == (signal,)


def test_accessibility_observation_becomes_safe() -> None:
    runtime = TrustCalibrationRuntime()

    trust = runtime.calibrate_observation(
        subject_id="button-run",
        subject_kind=TrustSubjectKind.UI_ELEMENT,
        source=EnvironmentSource.ACCESSIBILITY,
        confidence=0.99,
        stability=0.99,
        ambiguity=0.0,
        reason="accessibility button target",
    )

    assert trust.calibration.level == EnvironmentTrustLevel.VERIFIED
    assert trust.policy_classification == TrustPolicyClassification.SAFE


def test_ocr_observation_can_be_review_due_to_source_reliability() -> None:
    runtime = TrustCalibrationRuntime()

    trust = runtime.calibrate_observation(
        subject_id="ocr-error",
        subject_kind=TrustSubjectKind.OCR_TEXT,
        source=EnvironmentSource.OCR,
        confidence=0.91,
        stability=0.92,
        ambiguity=0.05,
        reason="OCR error text",
    )

    assert trust.policy_classification in {
        TrustPolicyClassification.REVIEW,
        TrustPolicyClassification.SAFE,
    }
    assert trust.calibration.source == EnvironmentSource.OCR


def test_high_ambiguity_requires_user_clarification() -> None:
    runtime = TrustCalibrationRuntime()

    trust = runtime.calibrate_observation(
        subject_id="save-button",
        subject_kind=TrustSubjectKind.GROUNDING_TARGET,
        source=EnvironmentSource.VISUAL_DETECTION,
        confidence=0.90,
        stability=0.90,
        ambiguity=0.80,
        reason="multiple matching save buttons",
    )

    assert trust.policy_classification == TrustPolicyClassification.ASK_USER


def test_secret_privacy_is_blocked_even_with_high_confidence() -> None:
    runtime = TrustCalibrationRuntime()

    trust = runtime.calibrate_observation(
        subject_id="password-field",
        subject_kind=TrustSubjectKind.UI_ELEMENT,
        source=EnvironmentSource.ACCESSIBILITY,
        confidence=0.99,
        stability=0.99,
        ambiguity=0.0,
        privacy=PrivacyClassification.SECRET,
        reason="password manager field",
    )

    assert trust.policy_classification == TrustPolicyClassification.BLOCKED


def test_action_requires_verification_even_when_confident() -> None:
    runtime = TrustCalibrationRuntime()

    trust = runtime.calibrate_action(
        subject_id="run-button",
        subject_kind=TrustSubjectKind.INTERACTION,
        source=EnvironmentSource.ACCESSIBILITY,
        confidence=0.96,
        stability=0.96,
        ambiguity=0.0,
        reason="run button click candidate",
    )

    assert trust.policy_classification == TrustPolicyClassification.VERIFY_FIRST
    assert trust.decision == TrustDecisionKind.ACCEPT_WITH_VERIFICATION
    assert trust.requires_verification is True


def test_action_blocks_low_trust_target() -> None:
    runtime = TrustCalibrationRuntime()

    trust = runtime.calibrate_action(
        subject_id="unknown-button",
        subject_kind=TrustSubjectKind.INTERACTION,
        source=EnvironmentSource.VISUAL_DETECTION,
        confidence=0.70,
        stability=0.70,
        ambiguity=0.20,
        reason="low trust visual target",
    )

    assert trust.policy_classification == TrustPolicyClassification.BLOCKED
    assert trust.decision == TrustDecisionKind.BLOCK


def test_verification_trust_safe_when_verified() -> None:
    runtime = TrustCalibrationRuntime()

    trust = runtime.calibrate_verification(
        subject_id="save-complete",
        subject_kind=TrustSubjectKind.VERIFICATION,
        source=EnvironmentSource.VERIFICATION,
        confidence=0.97,
        stability=0.97,
        ambiguity=0.0,
        verified=True,
        reason="expected file timestamp observed",
    )

    assert trust.verified is True
    assert trust.policy_classification == TrustPolicyClassification.SAFE


def test_build_decision_from_calibration() -> None:
    runtime = TrustCalibrationRuntime()
    observation = runtime.calibrate_observation(
        subject_id="dialog",
        subject_kind=TrustSubjectKind.UI_ELEMENT,
        source=EnvironmentSource.ACCESSIBILITY,
        confidence=0.95,
        stability=0.95,
        ambiguity=0.0,
        reason="dialog detected",
    )

    decision = runtime.build_decision(
        subject_id="dialog",
        subject_kind=TrustSubjectKind.UI_ELEMENT,
        calibration=observation.calibration,
        reason="accept dialog observation",
    )

    assert decision.decision == TrustDecisionKind.ACCEPT
    assert decision.policy_classification == TrustPolicyClassification.SAFE


def test_snapshot_tracks_runtime_state() -> None:
    runtime = TrustCalibrationRuntime()

    runtime.record_signal(
        ConfidenceSignal(
            subject_id="text",
            subject_kind=TrustSubjectKind.OCR_TEXT,
            source=EnvironmentSource.OCR,
            value=0.91,
            reason="clear text",
        )
    )
    runtime.calibrate_observation(
        subject_id="button",
        subject_kind=TrustSubjectKind.UI_ELEMENT,
        source=EnvironmentSource.ACCESSIBILITY,
        confidence=0.96,
        stability=0.96,
        ambiguity=0.0,
        reason="button detected",
    )
    runtime.calibrate_action(
        subject_id="click",
        subject_kind=TrustSubjectKind.INTERACTION,
        source=EnvironmentSource.ACCESSIBILITY,
        confidence=0.96,
        stability=0.96,
        ambiguity=0.0,
        reason="click target",
    )
    snapshot = runtime.snapshot()

    assert snapshot.signal_count == 1
    assert snapshot.observation_count == 1
    assert snapshot.action_count == 1
    assert snapshot.source_count >= 1
    assert snapshot.verify_first_count == 1


def test_reset_clears_runtime_state() -> None:
    runtime = TrustCalibrationRuntime()

    runtime.calibrate_observation(
        subject_id="button",
        subject_kind=TrustSubjectKind.UI_ELEMENT,
        source=EnvironmentSource.ACCESSIBILITY,
        confidence=0.96,
        stability=0.96,
        ambiguity=0.0,
        reason="button detected",
    )
    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.observation_count == 0
    assert snapshot.event_count == 1
    assert snapshot.last_reason == TrustRuntimeReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert TrustSubjectKind.OCR_TEXT.value == "ocr_text"
    assert TrustPolicyClassification.VERIFY_FIRST.value == "verify_first"
    assert TrustDecisionKind.ACCEPT_WITH_VERIFICATION.value == (
        "accept_with_verification"
    )