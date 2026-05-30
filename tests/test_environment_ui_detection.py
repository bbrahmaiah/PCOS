from __future__ import annotations

import pytest

from jarvis.environment import (
    DetectedUIElement,
    DetectedUIElementKind,
    ElementClassifierConfig,
    ElementInteractionSafety,
    EnvironmentSource,
    OCRTextKind,
    OCRTextRegion,
    PrivacyClassification,
    ScreenRegion,
    TrustCalibration,
    UIDetectionReason,
    UIDetectionRequest,
    UIDetectionRuntime,
    UIDetectionSource,
    UIDetectionStatus,
)


def test_runtime_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        UIDetectionRuntime(name=" ")


def test_classifier_config_accepts_defaults() -> None:
    config = ElementClassifierConfig()

    assert config.accessibility_min_confidence < config.visual_fallback_min_confidence


def test_create_session() -> None:
    runtime = UIDetectionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    assert session.workspace_id == "workspace"
    assert runtime.snapshot().session_count == 1


def test_accessibility_detection_preferred() -> None:
    runtime = UIDetectionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.detect_elements(_request(session.session_id))

    assert result.status == UIDetectionStatus.DETECTED
    assert result.reason == UIDetectionReason.ACCESSIBILITY_ELEMENTS_DETECTED
    assert result.source_order[0] == UIDetectionSource.ACCESSIBILITY
    assert result.elements
    assert result.elements[0].detection_source == UIDetectionSource.ACCESSIBILITY


def test_detected_element_has_trust_and_policy() -> None:
    runtime = UIDetectionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.detect_elements(_request(session.session_id))
    element = result.elements[0]

    assert isinstance(element, DetectedUIElement)
    assert element.trust.effective_score() > 0.0
    assert element.policy_classification is not None
    assert element.confidence.accepted is True


def test_secret_privacy_blocks_detection() -> None:
    runtime = UIDetectionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.detect_elements(
        _request(
            session.session_id,
            privacy=PrivacyClassification.SECRET,
        )
    )

    assert result.status == UIDetectionStatus.PRIVACY_BLOCKED
    assert result.reason == UIDetectionReason.PRIVACY_BLOCKED
    assert not result.elements


def test_app_parser_uses_ocr_text_regions() -> None:
    runtime = UIDetectionRuntime()
    session = runtime.create_session(workspace_id="workspace")
    text_region = _ocr_text_region(kind=OCRTextKind.CODE)

    result = runtime.detect_elements(
        _request(
            session.session_id,
            prefer_accessibility=False,
            text_regions=(text_region,),
        )
    )

    assert result.status == UIDetectionStatus.DETECTED
    assert result.reason == UIDetectionReason.APP_ELEMENTS_DETECTED
    assert any(
        element.kind == DetectedUIElementKind.CODE_EDITOR
        for element in result.elements
    )


def test_visual_fallback_runs_when_no_accessibility_or_app_elements() -> None:
    runtime = UIDetectionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.detect_elements(
        _request(
            session.session_id,
            prefer_accessibility=False,
            text_regions=(),
            allow_visual_fallback=True,
        )
    )

    assert result.status in {
        UIDetectionStatus.DETECTED,
        UIDetectionStatus.LOW_CONFIDENCE,
    }
    assert UIDetectionSource.VISUAL_FALLBACK in result.source_order


def test_visual_fallback_can_be_disabled() -> None:
    runtime = UIDetectionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.detect_elements(
        _request(
            session.session_id,
            prefer_accessibility=False,
            text_regions=(),
            allow_visual_fallback=False,
        )
    )

    assert result.status == UIDetectionStatus.LOW_CONFIDENCE


def test_interactive_region_map_blocks_low_confidence_elements() -> None:
    runtime = UIDetectionRuntime(
        classifier_config=ElementClassifierConfig(
            visual_fallback_min_confidence=0.95,
            interactive_min_confidence=0.95,
        )
    )
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.detect_elements(
        _request(
            session.session_id,
            prefer_accessibility=False,
            text_regions=(),
            allow_visual_fallback=True,
        )
    )
    region_map = runtime.build_interactive_region_map(
        session_id=session.session_id,
        elements=result.elements + result.rejected_elements,
    )

    assert region_map.blocked_count >= 1


def test_interactive_region_map_accepts_accessibility_button() -> None:
    runtime = UIDetectionRuntime()
    session = runtime.create_session(workspace_id="workspace")
    result = runtime.detect_elements(_request(session.session_id))

    region_map = runtime.build_interactive_region_map(
        session_id=session.session_id,
        elements=result.elements,
    )

    assert region_map.safe_count >= 1


def test_missing_session_detection_fails() -> None:
    runtime = UIDetectionRuntime()

    result = runtime.detect_elements(_request("missing"))

    assert result.status == UIDetectionStatus.FAILED
    assert result.reason == UIDetectionReason.SESSION_NOT_FOUND


def test_build_map_missing_session_raises() -> None:
    runtime = UIDetectionRuntime()

    with pytest.raises(ValueError):
        runtime.build_interactive_region_map(
            session_id="missing",
            elements=(),
        )


def test_snapshot_tracks_counts() -> None:
    runtime = UIDetectionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.detect_elements(_request(session.session_id))
    runtime.build_interactive_region_map(
        session_id=session.session_id,
        elements=result.elements,
    )
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.detection_count == 1
    assert snapshot.element_count >= 1
    assert snapshot.interactive_map_count == 1


def test_reset_clears_runtime() -> None:
    runtime = UIDetectionRuntime()
    runtime.create_session(workspace_id="workspace")

    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.runtime_event_count == 1
    assert snapshot.last_reason == UIDetectionReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert UIDetectionSource.ACCESSIBILITY.value == "accessibility"
    assert UIDetectionStatus.DETECTED.value == "detected"
    assert DetectedUIElementKind.BUTTON.value == "button"
    assert ElementInteractionSafety.BLOCKED.value == "blocked"


def _request(
    session_id: str,
    *,
    prefer_accessibility: bool = True,
    allow_visual_fallback: bool = True,
    text_regions: tuple[OCRTextRegion, ...] = (),
    privacy: PrivacyClassification = PrivacyClassification.WORKSPACE,
) -> UIDetectionRequest:
    return UIDetectionRequest(
        session_id=session_id,
        region=ScreenRegion(x=0, y=0, width=500, height=300),
        text_regions=text_regions,
        prefer_accessibility=prefer_accessibility,
        allow_visual_fallback=allow_visual_fallback,
        privacy=privacy,
    )


def _ocr_text_region(kind: OCRTextKind) -> OCRTextRegion:
    from jarvis.environment import (
        OCRSourceKind,
        TextConfidenceScore,
        TrustPolicyClassification,
    )

    return OCRTextRegion(
        text="def main() -> None:",
        bounds=ScreenRegion(x=0, y=0, width=300, height=20),
        kind=kind,
        source_kind=OCRSourceKind.CODE,
        source=EnvironmentSource.OCR,
        confidence=TextConfidenceScore(
            confidence=0.90,
            stability=0.90,
            ambiguity=0.0,
            source_kind=OCRSourceKind.CODE,
            accepted=True,
            reason="test text confidence",
        ),
        privacy=PrivacyClassification.WORKSPACE,
        trust=TrustCalibration(
            confidence=0.90,
            stability=0.90,
            ambiguity=0.0,
            source=EnvironmentSource.OCR,
            reason="test OCR trust",
        ),
        policy_classification=TrustPolicyClassification.SAFE,
        capture_id="capture",
    )