from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    BeliefStateSnapshot,
    DivergenceCause,
    DivergenceDetector,
    DivergenceSeverity,
    GroundTruthReason,
    GroundTruthReconciliationRuntime,
    GroundTruthStatus,
    ObservedRealitySnapshot,
    ReconciliationDecision,
    StateHistoryComparator,
    SyncRecovery,
    SyncRecoveryAction,
    WorkspaceGraphRuntime,
)


def test_runtime_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        GroundTruthReconciliationRuntime(name=" ")


def test_create_session() -> None:
    runtime = GroundTruthReconciliationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    assert session.workspace_id == "workspace"
    assert runtime.snapshot().session_count == 1


def test_sync_recovery_rejects_blocked_continue() -> None:
    with pytest.raises(ValidationError):
        SyncRecovery(
            decision=ReconciliationDecision.BLOCK,
            actions=(SyncRecoveryAction.NONE,),
            reason=GroundTruthReason.RECOVERY_PLANNED,
            safe_to_continue=True,
            message="invalid",
        )


def test_no_divergence_allows_continue() -> None:
    detector = DivergenceDetector()
    belief = _belief()
    observed = _observed()

    report = detector.detect(belief=belief, observed=observed)

    assert report.status == GroundTruthStatus.CONSISTENT
    assert report.decision == ReconciliationDecision.CONTINUE
    assert report.reason == GroundTruthReason.NO_DIVERGENCE
    assert not report.diverged


def test_focus_change_detected() -> None:
    detector = DivergenceDetector()

    report = detector.detect(
        belief=_belief(focused_node_id="editor"),
        observed=_observed(focused_node_id="terminal"),
    )

    assert report.diverged
    assert any(
        signal.cause == DivergenceCause.FOCUS_CHANGED
        for signal in report.signals
    )


def test_user_manual_action_detected_from_history() -> None:
    comparator = StateHistoryComparator()
    previous = _observed(user_action_counter=1)
    current = _observed(user_action_counter=2)

    result = comparator.compare(previous=previous, current=current)

    assert any(signal.cause == DivergenceCause.USER_ACTION for signal in result.signals)


def test_app_changed_unexpectedly_detected() -> None:
    report = DivergenceDetector().detect(
        belief=_belief(active_app_id="vscode"),
        observed=_observed(active_app_id="chrome"),
    )

    assert any(
        signal.cause == DivergenceCause.APP_STATE_CHANGE
        for signal in report.signals
    )
    assert report.decision == ReconciliationDecision.RESYNC


def test_popup_requires_verify_first() -> None:
    runtime = GroundTruthReconciliationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    report = runtime.reconcile(
        session_id=session.session_id,
        belief=_belief(popup_count=0),
        observed=_observed(popup_count=1),
    )
    recovery = runtime.plan_recovery(report)

    assert report.decision == ReconciliationDecision.VERIFY_FIRST
    assert recovery.requires_user_confirmation is True
    assert SyncRecoveryAction.ASK_USER in recovery.actions


def test_layout_change_requires_resync() -> None:
    runtime = GroundTruthReconciliationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    report = runtime.reconcile(
        session_id=session.session_id,
        belief=_belief(layout_hash="layout-a"),
        observed=_observed(layout_hash="layout-b"),
    )
    recovery = runtime.plan_recovery(report)

    assert report.decision == ReconciliationDecision.RESYNC
    assert SyncRecoveryAction.REFRESH_GRAPH in recovery.actions


def test_file_changed_externally_detected() -> None:
    report = DivergenceDetector().detect(
        belief=_belief(file_revision_hash="rev-a"),
        observed=_observed(file_revision_hash="rev-b"),
    )

    assert any(
        signal.cause == DivergenceCause.FILE_CHANGED_EXTERNALLY
        for signal in report.signals
    )


def test_browser_navigated_detected() -> None:
    report = DivergenceDetector().detect(
        belief=_belief(browser_url="https://old.example"),
        observed=_observed(browser_url="https://new.example"),
    )

    assert any(
        signal.cause == DivergenceCause.BROWSER_NAVIGATED
        for signal in report.signals
    )


def test_terminal_completed_detected() -> None:
    report = DivergenceDetector().detect(
        belief=_belief(terminal_status="running"),
        observed=_observed(terminal_status="completed"),
    )

    assert any(
        signal.cause == DivergenceCause.TERMINAL_COMPLETED
        for signal in report.signals
    )


def test_app_crash_requires_recovery() -> None:
    runtime = GroundTruthReconciliationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    report = runtime.reconcile(
        session_id=session.session_id,
        belief=_belief(app_crashed=False),
        observed=_observed(app_crashed=True),
    )
    recovery = runtime.plan_recovery(report)

    assert report.decision == ReconciliationDecision.RECOVER
    assert any(
        signal.severity == DivergenceSeverity.CRITICAL
        for signal in report.signals
    )
    assert recovery.safe_to_continue is False
    assert SyncRecoveryAction.ESCALATE_TO_RECOVERY in recovery.actions


def test_app_unresponsive_blocks_continuation() -> None:
    runtime = GroundTruthReconciliationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    report = runtime.reconcile(
        session_id=session.session_id,
        belief=_belief(app_responsive=True),
        observed=_observed(app_responsive=False),
    )
    recovery = runtime.plan_recovery(report)

    assert report.decision == ReconciliationDecision.BLOCK
    assert recovery.safe_to_continue is False


def test_resync_graph_applies_delta_for_resyncable_divergence() -> None:
    ground = GroundTruthReconciliationRuntime()
    ground_session = ground.create_session(workspace_id="workspace")
    graph_runtime = WorkspaceGraphRuntime()
    graph_session = graph_runtime.create_session(workspace_id="workspace")

    report = ground.reconcile(
        session_id=ground_session.session_id,
        belief=_belief(layout_hash="layout-a"),
        observed=_observed(layout_hash="layout-b"),
    )
    result = ground.resync_graph(
        graph_runtime=graph_runtime,
        graph_session_id=graph_session.session_id,
        report=report,
    )
    graph = graph_runtime.graph_for(graph_session.session_id)

    assert result.applied is True
    assert result.delta is not None
    assert graph is not None
    assert graph.nodes


def test_resync_graph_defers_for_popup() -> None:
    ground = GroundTruthReconciliationRuntime()
    ground_session = ground.create_session(workspace_id="workspace")
    graph_runtime = WorkspaceGraphRuntime()
    graph_session = graph_runtime.create_session(workspace_id="workspace")

    report = ground.reconcile(
        session_id=ground_session.session_id,
        belief=_belief(popup_count=0),
        observed=_observed(popup_count=1),
    )
    result = ground.resync_graph(
        graph_runtime=graph_runtime,
        graph_session_id=graph_session.session_id,
        report=report,
    )

    assert result.applied is False
    assert result.recovery.decision == ReconciliationDecision.VERIFY_FIRST


def test_missing_session_raises() -> None:
    runtime = GroundTruthReconciliationRuntime()

    with pytest.raises(ValueError):
        runtime.reconcile(
            session_id="missing",
            belief=_belief(),
            observed=_observed(),
        )


def test_snapshot_tracks_counts() -> None:
    runtime = GroundTruthReconciliationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    report = runtime.reconcile(
        session_id=session.session_id,
        belief=_belief(layout_hash="a"),
        observed=_observed(layout_hash="b"),
    )
    runtime.plan_recovery(report)
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.belief_snapshot_count == 1
    assert snapshot.reality_snapshot_count == 1
    assert snapshot.report_count == 1
    assert snapshot.divergence_count == 1
    assert snapshot.recovery_count == 1


def test_reset_clears_runtime() -> None:
    runtime = GroundTruthReconciliationRuntime()
    runtime.create_session(workspace_id="workspace")

    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.runtime_event_count == 1
    assert snapshot.last_reason == GroundTruthReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert DivergenceCause.USER_ACTION.value == "user_action"
    assert ReconciliationDecision.RESYNC.value == "resync"
    assert GroundTruthStatus.DIVERGED.value == "diverged"


def _belief(
    *,
    focused_node_id: str | None = "editor",
    active_app_id: str | None = "vscode",
    active_window_title: str | None = "VS Code",
    layout_hash: str | None = "layout",
    file_revision_hash: str | None = "rev",
    browser_url: str | None = "https://example.com",
    terminal_status: str | None = "running",
    app_responsive: bool = True,
    app_crashed: bool = False,
    popup_count: int = 0,
    notification_count: int = 0,
) -> BeliefStateSnapshot:
    return BeliefStateSnapshot(
        workspace_id="workspace",
        focused_node_id=focused_node_id,
        active_app_id=active_app_id,
        active_window_title=active_window_title,
        layout_hash=layout_hash,
        file_revision_hash=file_revision_hash,
        browser_url=browser_url,
        terminal_status=terminal_status,
        app_responsive=app_responsive,
        app_crashed=app_crashed,
        popup_count=popup_count,
        notification_count=notification_count,
    )


def _observed(
    *,
    focused_node_id: str | None = "editor",
    active_app_id: str | None = "vscode",
    active_window_title: str | None = "VS Code",
    layout_hash: str | None = "layout",
    file_revision_hash: str | None = "rev",
    browser_url: str | None = "https://example.com",
    terminal_status: str | None = "running",
    app_responsive: bool = True,
    app_crashed: bool = False,
    popup_count: int = 0,
    notification_count: int = 0,
    user_action_counter: int = 0,
) -> ObservedRealitySnapshot:
    return ObservedRealitySnapshot(
        workspace_id="workspace",
        focused_node_id=focused_node_id,
        active_app_id=active_app_id,
        active_window_title=active_window_title,
        layout_hash=layout_hash,
        file_revision_hash=file_revision_hash,
        browser_url=browser_url,
        terminal_status=terminal_status,
        app_responsive=app_responsive,
        app_crashed=app_crashed,
        popup_count=popup_count,
        notification_count=notification_count,
        user_action_counter=user_action_counter,
    )