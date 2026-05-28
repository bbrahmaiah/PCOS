from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    AppKind,
    AppState,
    DisplayKind,
    DisplayState,
    EnvironmentDelta,
    EnvironmentEvent,
    EnvironmentEventKind,
    EnvironmentSnapshot,
    EnvironmentSource,
    EnvironmentState,
    EnvironmentTrustLevel,
    GroundingResult,
    IntentState,
    IntentStatus,
    InteractionKind,
    InteractionRequest,
    InteractionRisk,
    PrivacyClassification,
    PrivacyZone,
    RecentStateHistory,
    RecoveryPlan,
    RecoveryStrategy,
    ScreenPoint,
    ScreenRegion,
    SimulationResult,
    SimulationStatus,
    TemporalWorkspaceState,
    TextRegion,
    TextRegionKind,
    TrustCalibration,
    TrustScore,
    UIElement,
    UIElementKind,
    VerificationResult,
    VerificationStatus,
    VisualConfidence,
    WindowMode,
    WindowState,
    WorkspaceMemoryEntry,
)


def test_visual_confidence_computes_trust_score() -> None:
    confidence = VisualConfidence(
        confidence=0.90,
        stability=0.90,
        ambiguity=0.10,
        source=EnvironmentSource.SCREEN_CAPTURE,
        explanation="clear button region",
    )

    assert confidence.trust_score() == pytest.approx(0.729)


def test_trust_score_contract() -> None:
    score = TrustScore(
        score=0.92,
        level=EnvironmentTrustLevel.VERIFIED,
        confidence=0.96,
        stability=0.98,
        ambiguity=0.02,
        source=EnvironmentSource.ACCESSIBILITY,
        reason="accessibility target verified",
    )

    assert score.level == EnvironmentTrustLevel.VERIFIED


def test_recent_state_history_tracks_snapshots_and_deltas() -> None:
    snapshot = EnvironmentSnapshot(trust=_trust())
    delta = EnvironmentDelta(
        current_snapshot_id=snapshot.snapshot_id,
        changed_windows=("window_a",),
        trust=_trust(),
    )
    history = RecentStateHistory(
        snapshots=(snapshot,),
        deltas=(delta,),
        max_items=5,
    )

    assert history.snapshots[0].snapshot_id == snapshot.snapshot_id
    assert history.deltas[0].changed_windows == ("window_a",)


def test_recent_state_history_rejects_too_many_items() -> None:
    snapshots = tuple(EnvironmentSnapshot(trust=_trust()) for _ in range(3))

    with pytest.raises(ValidationError):
        RecentStateHistory(snapshots=snapshots, max_items=2)


def test_environment_state_contract() -> None:
    snapshot = EnvironmentSnapshot(trust=_trust())
    history = RecentStateHistory(snapshots=(snapshot,), max_items=5)
    state = EnvironmentState(
        current_snapshot=snapshot,
        recent_history=history,
        focused_app_id="app",
        focused_window_id="window",
        active_workflow_id="workflow",
        trust=_trust(),
    )

    assert state.current_snapshot.snapshot_id == snapshot.snapshot_id
    assert state.recent_history is not None

def test_trust_calibration_derives_verified_level() -> None:
    trust = _trust(confidence=0.98)

    assert trust.level == EnvironmentTrustLevel.VERIFIED
    assert trust.effective_score() > 0.90


def test_trust_calibration_derives_low_level_when_ambiguous() -> None:
    trust = TrustCalibration(
        confidence=0.80,
        stability=0.70,
        ambiguity=0.50,
        source=EnvironmentSource.OCR,
        reason="unclear text",
    )

    assert trust.level == EnvironmentTrustLevel.LOW


def test_screen_region_requires_positive_size() -> None:
    with pytest.raises(ValidationError):
        ScreenRegion(x=0, y=0, width=0, height=100)


def test_screen_region_contains_point() -> None:
    region = ScreenRegion(x=10, y=10, width=100, height=100)

    assert region.contains_point(ScreenPoint(x=50, y=50)) is True
    assert region.contains_point(ScreenPoint(x=500, y=500)) is False


def test_display_state_contract() -> None:
    region = ScreenRegion(x=0, y=0, width=1920, height=1080)
    display = DisplayState(
        kind=DisplayKind.PRIMARY,
        bounds=region,
        primary=True,
    )

    assert display.primary is True
    assert display.bounds.area == 1920 * 1080


def test_app_state_contract() -> None:
    app = AppState(
        name="VS Code",
        kind=AppKind.IDE,
        responsive=True,
        trusted_identity=True,
    )

    assert app.kind == AppKind.IDE
    assert app.privacy_classification == PrivacyClassification.WORKSPACE


def test_window_state_contract() -> None:
    app = AppState(name="VS Code", kind=AppKind.IDE)
    window = WindowState(
        app_id=app.app_id,
        title="main.py - VS Code",
        bounds=ScreenRegion(x=0, y=0, width=1200, height=800),
        mode=WindowMode.NORMAL,
        focused=True,
        trust=_trust(),
    )

    assert window.focused is True
    assert window.app_id == app.app_id


def test_text_region_requires_confidence_and_kind() -> None:
    text = TextRegion(
        text="TypeError: expected int",
        bounds=ScreenRegion(x=0, y=0, width=400, height=40),
        kind=TextRegionKind.ERROR,
        trust=_trust(confidence=0.90, source=EnvironmentSource.OCR),
    )

    assert text.kind == TextRegionKind.ERROR


def test_interactive_ui_element_requires_minimum_trust() -> None:
    with pytest.raises(ValidationError):
        UIElement(
            kind=UIElementKind.BUTTON,
            bounds=ScreenRegion(x=0, y=0, width=120, height=40),
            interactive=True,
            trust=_trust(confidence=0.50),
        )


def test_ui_element_contract_accepts_high_trust_interactive_target() -> None:
    element = _button()

    assert element.interactive is True
    assert element.kind == UIElementKind.BUTTON


def test_environment_snapshot_contract() -> None:
    app = AppState(name="VS Code", kind=AppKind.IDE)
    window = WindowState(
        app_id=app.app_id,
        title="JARVIS_OS",
        bounds=ScreenRegion(x=0, y=0, width=1000, height=700),
        trust=_trust(),
    )
    snapshot = EnvironmentSnapshot(
        apps=(app,),
        windows=(window,),
        focused_app_id=app.app_id,
        focused_window_id=window.window_id,
        trust=_trust(source=EnvironmentSource.SCREEN_CAPTURE),
    )

    assert snapshot.focused_app_id == app.app_id
    assert len(snapshot.windows) == 1


def test_environment_delta_tracks_temporal_change() -> None:
    delta = EnvironmentDelta(
        current_snapshot_id="snapshot_current",
        appeared_elements=("element_a",),
        cause_hint="dialog appeared after click",
        trust=_trust(source=EnvironmentSource.OS_OBSERVER),
    )

    assert delta.appeared_elements == ("element_a",)


def test_temporal_workspace_state_links_snapshot_and_deltas() -> None:
    snapshot = EnvironmentSnapshot(trust=_trust())
    delta = EnvironmentDelta(
        current_snapshot_id=snapshot.snapshot_id,
        changed_windows=("window_a",),
        trust=_trust(),
    )
    state = TemporalWorkspaceState(
        current_snapshot=snapshot,
        recent_deltas=(delta,),
        last_user_action="clicked run",
    )

    assert state.current_snapshot.snapshot_id == snapshot.snapshot_id
    assert state.recent_deltas[0].changed_windows == ("window_a",)


def test_environment_event_contract() -> None:
    event = EnvironmentEvent(
        kind=EnvironmentEventKind.WINDOW_FOCUSED,
        source=EnvironmentSource.OS_OBSERVER,
        window_id="window_123",
        trust=_trust(source=EnvironmentSource.OS_OBSERVER),
    )

    assert event.kind == EnvironmentEventKind.WINDOW_FOCUSED


def test_grounding_result_requires_selected_candidate_membership() -> None:
    selected = _button(text="Save")
    other = _button(text="Cancel")

    with pytest.raises(ValidationError):
        GroundingResult(
            query="save button",
            candidates=(other,),
            selected_element=selected,
            trust=_trust(),
            explanation="wrong candidate",
        )


def test_grounding_result_accepts_selected_candidate() -> None:
    button = _button(text="Save")
    result = GroundingResult(
        query="save button",
        candidates=(button,),
        selected_element=button,
        trust=_trust(),
        explanation="exact button text match",
    )

    assert result.selected_element == button


def test_intent_state_contract() -> None:
    intent = IntentState(
        user_goal="debug the failing tests",
        status=IntentStatus.ACTIVE,
        active_subgoal="inspect pytest output",
    )

    assert intent.user_goal == "debug the failing tests"


def test_simulation_result_contract() -> None:
    simulation = SimulationResult(
        status=SimulationStatus.PREDICTED,
        predicted_state_summary="Save dialog should close.",
        rollback_risk=InteractionRisk.LOW,
        trust=_trust(source=EnvironmentSource.SIMULATION),
    )

    assert simulation.status == SimulationStatus.PREDICTED


def test_interaction_requires_target_for_physical_action() -> None:
    with pytest.raises(ValidationError):
        InteractionRequest(
            kind=InteractionKind.CLICK,
            risk=InteractionRisk.LOW,
        )


def test_interaction_rejects_low_trust_target_for_non_safe_action() -> None:
    low_trust_target = UIElement(
        kind=UIElementKind.BUTTON,
        bounds=ScreenRegion(x=0, y=0, width=100, height=40),
        interactive=True,
        trust=_trust(confidence=0.65),
    )

    with pytest.raises(ValidationError):
        InteractionRequest(
            kind=InteractionKind.CLICK,
            risk=InteractionRisk.LOW,
            target_element=low_trust_target,
        )


def test_interaction_accepts_high_trust_target() -> None:
    request = InteractionRequest(
        kind=InteractionKind.CLICK,
        risk=InteractionRisk.LOW,
        target_element=_button(),
        reversible=True,
    )

    assert request.target_element is not None
    assert request.reversible is True


def test_verification_result_matched_requires_confirmed_status() -> None:
    with pytest.raises(ValidationError):
        VerificationResult(
            status=VerificationStatus.PARTIAL,
            expected_summary="dialog closed",
            observed_summary="dialog maybe closed",
            matched=True,
            confidence=0.70,
            trust=_trust(source=EnvironmentSource.VERIFICATION),
        )


def test_verification_result_confirmed() -> None:
    result = VerificationResult(
        status=VerificationStatus.CONFIRMED,
        expected_summary="file saved",
        observed_summary="modified timestamp updated",
        matched=True,
        confidence=0.95,
        trust=_trust(source=EnvironmentSource.VERIFICATION),
    )

    assert result.matched is True
    assert result.recovery_needed is False


def test_recovery_plan_rejects_retry_count_above_max() -> None:
    with pytest.raises(ValidationError):
        RecoveryPlan(
            strategy=RecoveryStrategy.RETRY_SAME,
            reason="temporary UI failure",
            reversible=True,
            retry_count=4,
            max_retries=3,
        )


def test_workspace_memory_rejects_blocked_classification() -> None:
    with pytest.raises(ValidationError):
        WorkspaceMemoryEntry(
            workflow_id="workflow",
            privacy_classification=PrivacyClassification.BLOCKED,
        )


def test_workspace_memory_contract() -> None:
    memory = WorkspaceMemoryEntry(
        workflow_id="workflow",
        app_id="vscode",
        project_path="E:/JARVIS_OS",
        active_files=("main.py",),
        recent_commands=("pytest",),
        visible_errors=("1 failing test",),
        workflow_stage="debugging",
    )

    assert memory.project_path == "E:/JARVIS_OS"


def test_privacy_zone_requires_scope() -> None:
    with pytest.raises(ValidationError):
        PrivacyZone(name="password area", reason="sensitive")


def test_privacy_zone_contract() -> None:
    zone = PrivacyZone(
        name="password manager",
        app_name="1Password",
        reason="secret app blocked from capture",
    )

    assert zone.capture_allowed is False
    assert zone.ocr_allowed is False


def test_enum_values_are_stable() -> None:
    assert UIElementKind.BUTTON.value == "button"
    assert InteractionRisk.BLOCKED.value == "blocked"
    assert VerificationStatus.CONFIRMED.value == "confirmed"
    assert RecoveryStrategy.ROLLBACK.value == "rollback"
    assert PrivacyClassification.SECRET.value == "secret"


def _trust(
    *,
    confidence: float = 0.95,
    source: EnvironmentSource = EnvironmentSource.ACCESSIBILITY,
) -> TrustCalibration:
    return TrustCalibration(
        confidence=confidence,
        stability=1.0,
        ambiguity=0.0,
        source=source,
        reason="test trust",
    )


def _button(text: str = "Run") -> UIElement:
    return UIElement(
        kind=UIElementKind.BUTTON,
        text=text,
        bounds=ScreenRegion(x=10, y=10, width=100, height=40),
        interactive=True,
        trust=_trust(confidence=0.90),
    )