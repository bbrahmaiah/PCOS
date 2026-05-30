from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    ContentClass,
    DetectedAppKind,
    EnvironmentSource,
    InterfaceKind,
    OCRSourceKind,
    OCRTextKind,
    OCRTextRegion,
    PrivacyClassification,
    ScreenRegion,
    SemanticSceneKind,
    SensitivityLevel,
    TextConfidenceScore,
    TrustCalibration,
    TrustPolicyClassification,
    UIContext,
    UIContextRequest,
    UISemanticReason,
    UISemanticRuntime,
    UISemanticStatus,
)


def test_runtime_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        UISemanticRuntime(name=" ")


def test_context_rejects_sensitive_safe_action() -> None:
    runtime = UISemanticRuntime()
    session = runtime.create_session(workspace_id="workspace")
    context = runtime.understand(
        UIContextRequest(
            session_id=session.session_id,
            text_regions=(_ocr_text("password", OCRTextKind.PLAIN_TEXT),),
        )
    )

    with pytest.raises(ValidationError):
        UIContext(
            status=context.status,
            reason=context.reason,
            request_id=context.request_id,
            scene=context.scene,
            content=context.content,
            sensitive=context.sensitive,
            policy_classification=context.policy_classification,
            safe_for_reasoning=True,
            safe_for_action=True,
            message="invalid",
        )


def test_create_session() -> None:
    runtime = UISemanticRuntime()
    session = runtime.create_session(workspace_id="workspace")

    assert session.workspace_id == "workspace"
    assert runtime.snapshot().session_count == 1


def test_code_session_scene() -> None:
    runtime = UISemanticRuntime()
    session = runtime.create_session(workspace_id="workspace")

    context = runtime.understand(
        UIContextRequest(
            session_id=session.session_id,
            text_regions=(_ocr_text("def main() -> None:", OCRTextKind.CODE),),
            app_kind=DetectedAppKind.IDE,
        )
    )

    assert context.status == UISemanticStatus.UNDERSTOOD
    assert context.scene.kind == SemanticSceneKind.CODE_SESSION
    assert context.content.content_class == ContentClass.CODE
    assert context.safe_for_reasoning is True


def test_terminal_running_scene() -> None:
    runtime = UISemanticRuntime()
    session = runtime.create_session(workspace_id="workspace")

    context = runtime.understand(
        UIContextRequest(
            session_id=session.session_id,
            text_regions=(_ocr_text("pytest running", OCRTextKind.TERMINAL),),
            app_kind=DetectedAppKind.TERMINAL,
        )
    )

    assert context.scene.kind == SemanticSceneKind.TERMINAL_RUNNING
    assert context.content.interface_kind == InterfaceKind.TERMINAL


def test_error_dialog_scene() -> None:
    runtime = UISemanticRuntime()
    session = runtime.create_session(workspace_id="workspace")

    context = runtime.understand(
        UIContextRequest(
            session_id=session.session_id,
            text_regions=(_ocr_text("Error: failed to save", OCRTextKind.ERROR),),
            modal_present=True,
        )
    )

    assert context.scene.kind == SemanticSceneKind.ERROR_DIALOG
    assert context.policy_classification == TrustPolicyClassification.VERIFY_FIRST
    assert context.safe_for_action is False


def test_browser_research_scene() -> None:
    runtime = UISemanticRuntime()
    session = runtime.create_session(workspace_id="workspace")

    context = runtime.understand(
        UIContextRequest(
            session_id=session.session_id,
            text_regions=(_ocr_text("documentation page", OCRTextKind.BROWSER),),
            app_kind=DetectedAppKind.BROWSER,
        )
    )

    assert context.scene.kind == SemanticSceneKind.BROWSER_RESEARCH
    assert context.content.content_class == ContentClass.RESEARCH_PAGE


def test_sensitive_form_blocks_action() -> None:
    runtime = UISemanticRuntime()
    session = runtime.create_session(workspace_id="workspace")

    context = runtime.understand(
        UIContextRequest(
            session_id=session.session_id,
            text_regions=(_ocr_text("password and OTP", OCRTextKind.PLAIN_TEXT),),
        )
    )

    assert context.status == UISemanticStatus.SENSITIVE_BLOCKED
    assert context.scene.kind == SemanticSceneKind.FORM_SENSITIVE
    assert context.sensitive.level == SensitivityLevel.SECRET
    assert context.policy_classification == TrustPolicyClassification.BLOCKED
    assert context.safe_for_action is False


def test_media_player_scene() -> None:
    runtime = UISemanticRuntime()
    session = runtime.create_session(workspace_id="workspace")

    context = runtime.understand(
        UIContextRequest(
            session_id=session.session_id,
            app_kind=DetectedAppKind.MEDIA_APP,
        )
    )

    assert context.scene.kind == SemanticSceneKind.MEDIA_PLAYER
    assert context.content.content_class == ContentClass.MEDIA


def test_app_loading_scene_blocks_action() -> None:
    runtime = UISemanticRuntime()
    session = runtime.create_session(workspace_id="workspace")

    context = runtime.understand(
        UIContextRequest(
            session_id=session.session_id,
            loading=True,
        )
    )

    assert context.scene.kind == SemanticSceneKind.APP_LOADING
    assert context.safe_for_action is False


def test_confirmation_dialog_requires_verify_first() -> None:
    runtime = UISemanticRuntime()
    session = runtime.create_session(workspace_id="workspace")

    context = runtime.understand(
        UIContextRequest(
            session_id=session.session_id,
            text_regions=(
                _ocr_text(
                    "Are you sure you want to delete?",
                     OCRTextKind.PLAIN_TEXT,
                ),
            ),
            modal_present=True,
        )
    )

    assert context.scene.kind == SemanticSceneKind.CONFIRMATION_DIALOG
    assert context.policy_classification == TrustPolicyClassification.VERIFY_FIRST
    assert context.safe_for_action is False


def test_unknown_scene_low_trust() -> None:
    runtime = UISemanticRuntime()
    session = runtime.create_session(workspace_id="workspace")

    context = runtime.understand(UIContextRequest(session_id=session.session_id))

    assert context.status == UISemanticStatus.UNKNOWN
    assert context.scene.kind == SemanticSceneKind.UNKNOWN
    assert context.reason == UISemanticReason.UNKNOWN_SCENE
    assert context.safe_for_action is False


def test_missing_session_fails() -> None:
    runtime = UISemanticRuntime()

    context = runtime.understand(UIContextRequest(session_id="missing"))

    assert context.status == UISemanticStatus.FAILED
    assert context.reason == UISemanticReason.SESSION_NOT_FOUND
    assert context.safe_for_action is False


def test_snapshot_tracks_counts() -> None:
    runtime = UISemanticRuntime()
    session = runtime.create_session(workspace_id="workspace")

    runtime.understand(
        UIContextRequest(
            session_id=session.session_id,
            text_regions=(_ocr_text("password", OCRTextKind.PLAIN_TEXT),),
        )
    )
    runtime.understand(UIContextRequest(session_id=session.session_id))
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.context_count == 2
    assert snapshot.sensitive_count == 1
    assert snapshot.blocked_count == 2
    assert snapshot.unknown_count == 1


def test_reset_clears_runtime() -> None:
    runtime = UISemanticRuntime()
    runtime.create_session(workspace_id="workspace")

    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.runtime_event_count == 1
    assert snapshot.last_reason == UISemanticReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert SemanticSceneKind.CODE_SESSION.value == "code_session"
    assert InterfaceKind.TERMINAL.value == "terminal"
    assert ContentClass.SENSITIVE_FORM.value == "sensitive_form"
    assert SensitivityLevel.SECRET.value == "secret"


def _ocr_text(text: str, kind: OCRTextKind) -> OCRTextRegion:
    return OCRTextRegion(
        text=text,
        bounds=ScreenRegion(x=0, y=0, width=300, height=20),
        kind=kind,
        source_kind=OCRSourceKind.GENERIC_OCR,
        source=EnvironmentSource.OCR,
        confidence=TextConfidenceScore(
            confidence=0.90,
            stability=0.90,
            ambiguity=0.0,
            source_kind=OCRSourceKind.GENERIC_OCR,
            accepted=True,
            reason="test confidence",
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