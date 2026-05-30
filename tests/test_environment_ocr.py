from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    AttentionGovernanceReason,
    CaptureGovernanceDecision,
    CaptureMode,
    CapturePayload,
    CapturePermission,
    CapturePixelFormat,
    CaptureStatus,
    EnvironmentSource,
    InspectionDepth,
    OCRExtractionRequest,
    OCRReason,
    OCRRuntime,
    OCRSourceKind,
    OCRStatus,
    OCRTextKind,
    PrivacyClassification,
    PrivacyZoneDecision,
    RegionCapture,
    ScreenRegion,
    TerminalTextExtractor,
    TextConfidenceScorer,
    TextConfidenceScorerConfig,
    TrustCalibration,
)


def test_runtime_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        OCRRuntime(name=" ")


def test_confidence_config_requires_valid_thresholds() -> None:
    with pytest.raises(ValidationError):
        TextConfidenceScorerConfig(
            minimum_confidence=0.9,
            trusted_confidence=0.8,
        )


def test_confidence_scorer_accepts_clear_text() -> None:
    scorer = TextConfidenceScorer()
    score = scorer.score(
        text="pytest passed in 20s",
        source_kind=OCRSourceKind.TERMINAL,
        privacy=PrivacyClassification.WORKSPACE,
    )

    assert score.accepted is True
    assert score.confidence > 0.55


def test_confidence_scorer_rejects_secret_text() -> None:
    scorer = TextConfidenceScorer()
    score = scorer.score(
        text="password secret",
        source_kind=OCRSourceKind.GENERIC_OCR,
        privacy=PrivacyClassification.SECRET,
    )

    assert score.accepted is False
    assert score.confidence == 0.0


def test_create_session() -> None:
    runtime = OCRRuntime()
    session = runtime.create_session(workspace_id="workspace")

    assert session.workspace_id == "workspace"
    assert runtime.snapshot().session_count == 1


def test_extract_generic_text() -> None:
    runtime = OCRRuntime()
    session = runtime.create_session(workspace_id="workspace")
    request = _request(session_id=session.session_id)

    result = runtime.extract_text(request)

    assert result.status == OCRStatus.EXTRACTED
    assert result.reason == OCRReason.OCR_EXTRACTED
    assert result.text_regions
    assert result.text_regions[0].bounds.width == 100
    assert result.text_regions[0].confidence.accepted is True


def test_text_region_converts_to_environment_text_region() -> None:
    runtime = OCRRuntime()
    session = runtime.create_session(workspace_id="workspace")
    result = runtime.extract_text(_request(session_id=session.session_id))

    text_region = result.text_regions[0].to_environment_text_region()

    assert text_region.text
    assert result.text_regions[0].confidence.confidence > 0.0


def test_extract_code_text() -> None:
    runtime = OCRRuntime()
    session = runtime.create_session(workspace_id="workspace")
    request = _request(session_id=session.session_id)

    result = runtime.extract_code_text(request)

    assert result.status == OCRStatus.EXTRACTED
    assert result.reason == OCRReason.CODE_TEXT_EXTRACTED
    assert result.text_regions[0].kind == OCRTextKind.CODE


def test_extract_terminal_text() -> None:
    runtime = OCRRuntime()
    session = runtime.create_session(workspace_id="workspace")
    request = _request(session_id=session.session_id)
    extractor = TerminalTextExtractor()

    result = extractor.extract(runtime, request)

    assert result.status == OCRStatus.EXTRACTED
    assert result.reason == OCRReason.TERMINAL_TEXT_EXTRACTED
    assert result.text_regions[0].kind in {
        OCRTextKind.TERMINAL,
        OCRTextKind.ERROR,
    }


def test_extract_browser_text() -> None:
    runtime = OCRRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.extract_browser_text(_request(session_id=session.session_id))

    assert result.reason == OCRReason.BROWSER_TEXT_EXTRACTED
    assert result.text_regions[0].kind == OCRTextKind.BROWSER


def test_extract_document_text() -> None:
    runtime = OCRRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.extract_document_text(_request(session_id=session.session_id))

    assert result.reason == OCRReason.DOCUMENT_TEXT_EXTRACTED
    assert result.text_regions[0].kind == OCRTextKind.DOCUMENT


def test_low_confidence_text_is_rejected() -> None:
    runtime = OCRRuntime(
        scorer=TextConfidenceScorer(
            TextConfidenceScorerConfig(
                minimum_confidence=0.99,
                trusted_confidence=0.99,
            )
        )
    )
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.extract_text(_request(session_id=session.session_id))

    assert result.status == OCRStatus.LOW_CONFIDENCE
    assert result.reason == OCRReason.LOW_CONFIDENCE_TEXT_REJECTED
    assert result.rejected_regions


def test_secret_privacy_blocks_ocr() -> None:
    runtime = OCRRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.extract_text(
        _request(
            session_id=session.session_id,
            privacy=PrivacyClassification.SECRET,
        )
    )

    assert result.status == OCRStatus.PRIVACY_BLOCKED
    assert result.reason == OCRReason.PRIVACY_BLOCKED
    assert not result.text_regions


def test_invalid_capture_status_blocks_ocr() -> None:
    runtime = OCRRuntime()
    session = runtime.create_session(workspace_id="workspace")
    capture = _capture(status=CaptureStatus.BLOCKED)

    result = runtime.extract_text(
        OCRExtractionRequest(
            session_id=session.session_id,
            capture=capture,
        )
    )

    assert result.status == OCRStatus.CAPTURE_INVALID


def test_missing_session_fails() -> None:
    runtime = OCRRuntime()

    result = runtime.extract_text(_request(session_id="missing"))

    assert result.status == OCRStatus.FAILED
    assert result.reason == OCRReason.SESSION_NOT_FOUND


def test_snapshot_tracks_counts() -> None:
    runtime = OCRRuntime()
    session = runtime.create_session(workspace_id="workspace")

    runtime.extract_text(_request(session_id=session.session_id))
    runtime.extract_text(
        _request(
            session_id=session.session_id,
            privacy=PrivacyClassification.SECRET,
        )
    )
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.extraction_count == 2
    assert snapshot.accepted_text_region_count >= 1
    assert snapshot.blocked_count == 1


def test_reset_clears_runtime() -> None:
    runtime = OCRRuntime()
    runtime.create_session(workspace_id="workspace")

    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.runtime_event_count == 1
    assert snapshot.last_reason == OCRReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert OCRSourceKind.GENERIC_OCR.value == "generic_ocr"
    assert OCRStatus.EXTRACTED.value == "extracted"
    assert OCRTextKind.CODE.value == "code"
    assert OCRReason.OCR_EXTRACTED.value == "ocr_extracted"


def _request(
    *,
    session_id: str,
    privacy: PrivacyClassification = PrivacyClassification.WORKSPACE,
) -> OCRExtractionRequest:
    return OCRExtractionRequest(
        session_id=session_id,
        capture=_capture(),
        privacy=privacy,
    )


def _capture(
    *,
    status: CaptureStatus = CaptureStatus.CAPTURED,
) -> RegionCapture:
    region = ScreenRegion(x=0, y=0, width=100, height=40)

    return RegionCapture(
        mode=CaptureMode.REGION,
        status=status,
        region=region,
        payload=CapturePayload(
            width=100,
            height=40,
            pixel_format=CapturePixelFormat.FAKE,
            byte_count=4000,
            content_hash="fake",
            metadata={"text": "pytest passed in 20s"},
        ),
        governance=_governance(),
        trust=TrustCalibration(
            confidence=0.95,
            stability=0.95,
            ambiguity=0.0,
            source=EnvironmentSource.SCREEN_CAPTURE,
            reason="test capture",
        ),
    )


def _governance() -> CaptureGovernanceDecision:
    return CaptureGovernanceDecision(
        request_id="capture-request",
        permission=CapturePermission.ALLOW,
        reason=AttentionGovernanceReason.CAPTURE_ALLOWED,
        allowed_depth=InspectionDepth.PERIPHERAL,
        frequency_hz=0.5,
        privacy_decision=PrivacyZoneDecision.ALLOWED,
        region=ScreenRegion(x=0, y=0, width=100, height=40),
    )