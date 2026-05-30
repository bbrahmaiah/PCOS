from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    ContentClass,
    ContentClassification,
    DetectedAppKind,
    EnvironmentSource,
    InterfaceKind,
    InterfacePatternRecognitionRuntime,
    OCRSourceKind,
    OCRTextKind,
    OCRTextRegion,
    PatternMatcher,
    PrivacyClassification,
    ScreenRegion,
    SemanticScene,
    SemanticSceneKind,
    SensitiveUIDetection,
    SensitivityLevel,
    TextConfidenceScore,
    TrustCalibration,
    TrustPolicyClassification,
    UIContext,
    UIPatternDefinition,
    UIPatternKind,
    UIPatternMatchRequest,
    UIPatternReason,
    UIPatternRisk,
    UIPatternSource,
    UIPatternStatus,
    UISemanticReason,
    UISemanticStatus,
)


def test_runtime_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        InterfacePatternRecognitionRuntime(name=" ")


def test_pattern_definition_requires_signal() -> None:
    with pytest.raises(ValidationError):
        UIPatternDefinition(
            kind=UIPatternKind.UNKNOWN,
            name="Broken",
            source=UIPatternSource.COMMON_LIBRARY,
        )


def test_default_patterns_loaded() -> None:
    runtime = InterfacePatternRecognitionRuntime()

    snapshot = runtime.library_snapshot()

    assert snapshot.total_pattern_count >= 10


def test_create_session() -> None:
    runtime = InterfacePatternRecognitionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    assert session.workspace_id == "workspace"
    assert runtime.snapshot().session_count == 1


def test_confirmation_dialog_recognized() -> None:
    runtime = InterfacePatternRecognitionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.recognize(
        UIPatternMatchRequest(
            session_id=session.session_id,
            ui_context=_context(
                scene=SemanticSceneKind.CONFIRMATION_DIALOG,
                content=ContentClass.CONFIRMATION,
            ),
            text_regions=(
                _ocr_text("Are you sure you want to delete?"),
            ),
        )
    )

    assert result.status == UIPatternStatus.MATCHED
    assert result.best_match is not None
    assert result.best_match.pattern.kind == UIPatternKind.CONFIRMATION_DIALOG
    assert result.safe_for_action is False


def test_error_dialog_recognized() -> None:
    runtime = InterfacePatternRecognitionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.recognize(
        UIPatternMatchRequest(
            session_id=session.session_id,
            ui_context=_context(
                scene=SemanticSceneKind.ERROR_DIALOG,
                content=ContentClass.ERROR,
            ),
            text_regions=(_ocr_text("Error: failed to save"),),
        )
    )

    assert result.best_match is not None
    assert result.best_match.pattern.kind == UIPatternKind.ERROR_DIALOG
    assert result.reason in {
        UIPatternReason.COMMON_PATTERN_MATCHED,
        UIPatternReason.DIALOG_PATTERN_MATCHED,
    }


def test_login_form_blocks_action() -> None:
    runtime = InterfacePatternRecognitionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.recognize(
        UIPatternMatchRequest(
            session_id=session.session_id,
            ui_context=_context(
                scene=SemanticSceneKind.FORM_SENSITIVE,
                content=ContentClass.SENSITIVE_FORM,
                blocked=True,
            ),
            text_regions=(_ocr_text("login password otp"),),
        )
    )

    assert result.best_match is not None
    assert result.best_match.pattern.kind == UIPatternKind.LOGIN_FORM
    assert result.status == UIPatternStatus.BLOCKED
    assert result.safe_for_action is False


def test_file_picker_recognized() -> None:
    runtime = InterfacePatternRecognitionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.recognize(
        UIPatternMatchRequest(
            session_id=session.session_id,
            ui_context=_context(
                scene=SemanticSceneKind.FILE_PICKER,
                content=ContentClass.FILE_SELECTION,
            ),
            text_regions=(_ocr_text("Open File Name Cancel"),),
        )
    )

    assert result.best_match is not None
    assert result.best_match.pattern.kind == UIPatternKind.FILE_PICKER


def test_loading_spinner_recognized() -> None:
    runtime = InterfacePatternRecognitionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.recognize(
        UIPatternMatchRequest(
            session_id=session.session_id,
            ui_context=_context(
                scene=SemanticSceneKind.APP_LOADING,
                content=ContentClass.LOADING,
            ),
            text_regions=(_ocr_text("Loading please wait"),),
            loading=True,
        )
    )

    assert result.best_match is not None
    assert result.best_match.pattern.kind == UIPatternKind.LOADING_SPINNER
    assert result.safe_for_action is False


def test_progress_bar_recognized() -> None:
    runtime = InterfacePatternRecognitionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.recognize(
        UIPatternMatchRequest(
            session_id=session.session_id,
            text_regions=(_ocr_text("Progress 70% completed"),),
        )
    )

    assert result.best_match is not None
    assert result.best_match.pattern.kind == UIPatternKind.PROGRESS_BAR
    assert result.safe_for_action is True


def test_terminal_prompt_recognized() -> None:
    runtime = InterfacePatternRecognitionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.recognize(
        UIPatternMatchRequest(
            session_id=session.session_id,
            app_kind=DetectedAppKind.TERMINAL,
            ui_context=_context(
                scene=SemanticSceneKind.TERMINAL_RUNNING,
                content=ContentClass.TERMINAL_OUTPUT,
            ),
            text_regions=(_ocr_text("PS E:\\JARVIS_OS> pytest"),),
        )
    )

    assert result.best_match is not None
    assert result.best_match.pattern.kind == UIPatternKind.TERMINAL_PROMPT


def test_vscode_command_palette_recognized() -> None:
    runtime = InterfacePatternRecognitionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.recognize(
        UIPatternMatchRequest(
            session_id=session.session_id,
            app_kind=DetectedAppKind.IDE,
            text_regions=(_ocr_text("> Show All Commands"),),
        )
    )

    assert result.best_match is not None
    assert result.best_match.pattern.kind == UIPatternKind.VSCODE_COMMAND_PALETTE


def test_chrome_warning_page_blocked() -> None:
    runtime = InterfacePatternRecognitionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.recognize(
        UIPatternMatchRequest(
            session_id=session.session_id,
            app_kind=DetectedAppKind.BROWSER,
            text_regions=(
                _ocr_text("Your connection is not private Advanced Back to safety"),
            ),
        )
    )

    assert result.status == UIPatternStatus.BLOCKED
    assert result.best_match is not None
    assert result.best_match.pattern.kind == UIPatternKind.CHROME_WARNING_PAGE
    assert result.safe_for_action is False


def test_unknown_pattern() -> None:
    runtime = InterfacePatternRecognitionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.recognize(
        UIPatternMatchRequest(
            session_id=session.session_id,
            text_regions=(_ocr_text("random unrelated text"),),
        )
    )

    assert result.status == UIPatternStatus.UNKNOWN
    assert result.reason == UIPatternReason.PATTERN_UNKNOWN


def test_missing_session_fails() -> None:
    runtime = InterfacePatternRecognitionRuntime()

    result = runtime.recognize(UIPatternMatchRequest(session_id="missing"))

    assert result.status == UIPatternStatus.FAILED
    assert result.reason == UIPatternReason.SESSION_NOT_FOUND


def test_register_custom_pattern() -> None:
    runtime = InterfacePatternRecognitionRuntime(load_defaults=False)
    session = runtime.create_session(workspace_id="workspace")
    pattern = UIPatternDefinition(
        kind=UIPatternKind.UNKNOWN,
        name="Media Player Pattern",
        source=UIPatternSource.COMMON_LIBRARY,
        text_terms=("play", "pause"),
        minimum_score=0.40,
        risk=UIPatternRisk.SAFE,
    )

    runtime.register_pattern(pattern)
    result = runtime.recognize(
        UIPatternMatchRequest(
            session_id=session.session_id,
            text_regions=(_ocr_text("play pause"),),
        )
    )

    assert result.best_match is not None
    assert result.best_match.pattern.kind == UIPatternKind.UNKNOWN


def test_pattern_matcher_scores_required_app_signal() -> None:
    pattern = UIPatternDefinition(
        kind=UIPatternKind.TERMINAL_PROMPT,
        name="Terminal",
        source=UIPatternSource.APP_SPECIFIC_LIBRARY,
        app_kind=DetectedAppKind.TERMINAL,
        text_terms=("pytest",),
        minimum_score=0.40,
    )
    matcher = PatternMatcher()

    matches = matcher.match(
        request=UIPatternMatchRequest(
            session_id="session",
            app_kind=DetectedAppKind.IDE,
            text_regions=(_ocr_text("pytest"),),
        ),
        patterns=(pattern,),
    )

    assert not matches


def test_snapshot_tracks_counts() -> None:
    runtime = InterfacePatternRecognitionRuntime()
    session = runtime.create_session(workspace_id="workspace")

    runtime.recognize(
        UIPatternMatchRequest(
            session_id=session.session_id,
            text_regions=(_ocr_text("Progress 70% completed"),),
        )
    )
    runtime.recognize(
        UIPatternMatchRequest(
            session_id=session.session_id,
            text_regions=(_ocr_text("random unrelated text"),),
        )
    )
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.result_count == 2
    assert snapshot.matched_count >= 1
    assert snapshot.unknown_count == 1


def test_reset_clears_runtime() -> None:
    runtime = InterfacePatternRecognitionRuntime()
    runtime.create_session(workspace_id="workspace")

    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.runtime_event_count == 1
    assert snapshot.last_reason == UIPatternReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert UIPatternKind.SAVE_DIALOG.value == "save_dialog"
    assert UIPatternStatus.MATCHED.value == "matched"
    assert UIPatternSource.COMMON_LIBRARY.value == "common_library"
    assert UIPatternRisk.BLOCKED.value == "blocked"


def _ocr_text(text: str) -> OCRTextRegion:
    return OCRTextRegion(
        text=text,
        bounds=ScreenRegion(x=0, y=0, width=300, height=20),
        kind=OCRTextKind.PLAIN_TEXT,
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


def _context(
    *,
    scene: SemanticSceneKind,
    content: ContentClass,
    blocked: bool = False,
) -> UIContext:
    sensitive = SensitiveUIDetection(
        sensitive=blocked,
        level=SensitivityLevel.SECRET if blocked else SensitivityLevel.NONE,
        reason="test sensitive" if blocked else "not sensitive",
        action_blocked=blocked,
    )
    classification = ContentClassification(
        content_class=content,
        interface_kind=InterfaceKind.DIALOG,
        confidence=0.90,
        reason="test classification",
        evidence=(content.value,),
    )
    semantic_scene = SemanticScene(
        kind=scene,
        interface_kind=InterfaceKind.DIALOG,
        confidence=0.90,
        summary=scene.value,
        trust=TrustCalibration(
            confidence=0.90,
            stability=0.90,
            ambiguity=0.0,
            source=EnvironmentSource.OS_OBSERVER,
            reason="test scene",
        ),
    )

    return UIContext(
        status=UISemanticStatus.SENSITIVE_BLOCKED
        if blocked
        else UISemanticStatus.UNDERSTOOD,
        reason=UISemanticReason.SENSITIVE_UI_DETECTED
        if blocked
        else UISemanticReason.SCENE_UNDERSTOOD,
        request_id="semantic-request",
        scene=semantic_scene,
        content=classification,
        sensitive=sensitive,
        policy_classification=TrustPolicyClassification.BLOCKED
        if blocked
        else TrustPolicyClassification.SAFE,
        safe_for_reasoning=True,
        safe_for_action=not blocked,
        message="test context",
    )