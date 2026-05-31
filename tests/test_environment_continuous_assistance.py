from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    AppCrashWatcher,
    AssistanceConfidenceBand,
    AssistanceDecision,
    AssistanceObservation,
    AssistanceObservationKind,
    AssistancePreparation,
    AssistancePreparationKind,
    AssistanceReason,
    AssistanceStatus,
    AssistanceSuggestion,
    AssistanceSuggestionKind,
    BuildErrorWatcher,
    ContinuousAssistanceModeRuntime,
    LongRunningTaskWatcher,
    ProactivePermission,
    RepeatedActionDetector,
    UserReturnDetector,
    WorkflowKind,
    WorkflowSuggestionPolicy,
)


def test_runtime_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        ContinuousAssistanceModeRuntime(name=" ")


def test_preparation_cannot_contain_action() -> None:
    observation = _observation()

    with pytest.raises(ValidationError):
        AssistancePreparation(
            kind=AssistancePreparationKind.ERROR_SUMMARY,
            observation=observation,
            prepared_summary="prepared",
            contains_action=True,
            confidence=0.9,
        )


def test_suggestion_cannot_request_action_execution() -> None:
    observation = _observation()

    with pytest.raises(ValidationError):
        AssistanceSuggestion(
            kind=AssistanceSuggestionKind.DEBUG_ERROR,
            observation=observation,
            message="Want help?",
            confidence=0.9,
            confidence_band=AssistanceConfidenceBand.HIGH,
            action_requested=True,
        )


def test_create_session() -> None:
    runtime = ContinuousAssistanceModeRuntime()
    session = runtime.create_session(workspace_id="workspace")

    assert session.workspace_id == "workspace"
    assert runtime.snapshot().session_count == 1


def test_build_error_watcher_detects_error() -> None:
    watcher = BuildErrorWatcher()

    observation = watcher.observe(
        workspace_id="workspace",
        build_output="FAILED tests/test_app.py::test_x AssertionError",
    )

    assert observation is not None
    assert observation.kind == AssistanceObservationKind.BUILD_ERROR
    assert observation.workflow == WorkflowKind.DEBUGGING


def test_build_error_watcher_ignores_clean_output() -> None:
    watcher = BuildErrorWatcher()

    observation = watcher.observe(
        workspace_id="workspace",
        build_output="3105 passed in 22s",
    )

    assert observation is None


def test_long_running_task_watcher_detects_threshold() -> None:
    watcher = LongRunningTaskWatcher()

    observation = watcher.observe(
        workspace_id="workspace",
        task_name="pytest",
        elapsed_seconds=400,
        threshold_seconds=300,
    )

    assert observation is not None
    assert observation.kind == AssistanceObservationKind.LONG_RUNNING_TASK


def test_repeated_action_detector_detects_repetition() -> None:
    detector = RepeatedActionDetector()

    observation = detector.observe(
        workspace_id="workspace",
        action_names=("scroll", "scroll", "click", "scroll"),
    )

    assert observation is not None
    assert observation.kind == AssistanceObservationKind.REPEATED_ACTION


def test_app_crash_watcher_detects_crash() -> None:
    watcher = AppCrashWatcher()

    observation = watcher.observe(
        workspace_id="workspace",
        app_name="VS Code",
        crashed=True,
    )

    assert observation is not None
    assert observation.kind == AssistanceObservationKind.APP_CRASH


def test_user_return_detector_detects_return() -> None:
    detector = UserReturnDetector()

    observation = detector.observe(
        workspace_id="workspace",
        away_seconds=120,
        active_workflow=WorkflowKind.DEBUGGING,
    )

    assert observation is not None
    assert observation.kind == AssistanceObservationKind.USER_RETURN
    assert observation.workflow == WorkflowKind.DEBUGGING


def test_observe_records_passive_awareness() -> None:
    runtime = ContinuousAssistanceModeRuntime()
    session = runtime.create_session(workspace_id="workspace")
    observation = _observation()

    result = runtime.observe(
        session_id=session.session_id,
        observation=observation,
    )

    assert result.status == AssistanceStatus.OBSERVED
    assert result.decision == AssistanceDecision.OBSERVE
    assert result.observation is not None


def test_prepare_records_safe_preparation() -> None:
    runtime = ContinuousAssistanceModeRuntime()
    session = runtime.create_session(workspace_id="workspace")
    observation = _observation()

    result = runtime.prepare(
        session_id=session.session_id,
        observation=observation,
    )

    assert result.status == AssistanceStatus.PREPARED
    assert result.decision == AssistanceDecision.PREPARE
    assert result.preparation is not None
    assert result.preparation.contains_action is False


def test_suggest_only_when_confident() -> None:
    runtime = ContinuousAssistanceModeRuntime()
    session = runtime.create_session(workspace_id="workspace")
    observation = _observation(confidence=0.90)

    prepared = runtime.prepare(
        session_id=session.session_id,
        observation=observation,
    )
    result = runtime.suggest(
        session_id=session.session_id,
        observation=observation,
        preparation=prepared.preparation,
    )

    assert result.status == AssistanceStatus.SUGGESTED
    assert result.decision == AssistanceDecision.SUGGEST
    assert result.suggestion is not None
    assert result.suggestion.requires_user_confirmation is True
    assert result.suggestion.action_requested is False


def test_low_confidence_suggestion_is_suppressed() -> None:
    runtime = ContinuousAssistanceModeRuntime()
    session = runtime.create_session(workspace_id="workspace")
    observation = _observation(confidence=0.40)

    result = runtime.suggest(
        session_id=session.session_id,
        observation=observation,
    )

    assert result.status == AssistanceStatus.SUPPRESSED
    assert result.reason == AssistanceReason.LOW_CONFIDENCE_SUPPRESSED


def test_policy_can_disable_suggestions() -> None:
    runtime = ContinuousAssistanceModeRuntime(
        policy=WorkflowSuggestionPolicy(allow_proactive_suggestion=False)
    )
    session = runtime.create_session(workspace_id="workspace")
    observation = _observation(confidence=0.95)

    result = runtime.suggest(
        session_id=session.session_id,
        observation=observation,
    )

    assert result.status == AssistanceStatus.SUPPRESSED


def test_proactive_action_is_blocked() -> None:
    runtime = ContinuousAssistanceModeRuntime()
    session = runtime.create_session(workspace_id="workspace")

    result = runtime.request_proactive_action(
        session_id=session.session_id,
        description="click the submit button",
    )

    assert result.status == AssistanceStatus.BLOCKED
    assert result.decision == AssistanceDecision.BLOCK_ACTION
    assert result.reason == AssistanceReason.PROACTIVE_ACTION_BLOCKED


def test_policy_decision_blocks_action_permission() -> None:
    runtime = ContinuousAssistanceModeRuntime()
    decision = runtime._policy_decision(
        permission=ProactivePermission.ACTION,
        confidence=1.0,
    )

    assert decision.allowed is False
    assert decision.reason == AssistanceReason.PROACTIVE_ACTION_BLOCKED


def test_missing_session_fails_operations() -> None:
    runtime = ContinuousAssistanceModeRuntime()
    observation = _observation()

    observed = runtime.observe(session_id="missing", observation=observation)
    prepared = runtime.prepare(session_id="missing", observation=observation)
    suggested = runtime.suggest(session_id="missing", observation=observation)
    action = runtime.request_proactive_action(
        session_id="missing",
        description="click",
    )

    assert observed.status == AssistanceStatus.FAILED
    assert prepared.status == AssistanceStatus.FAILED
    assert suggested.status == AssistanceStatus.FAILED
    assert action.status == AssistanceStatus.FAILED


def test_snapshot_tracks_counts() -> None:
    runtime = ContinuousAssistanceModeRuntime()
    session = runtime.create_session(workspace_id="workspace")
    confident = _observation(confidence=0.90)
    low = _observation(confidence=0.30)

    runtime.observe(session_id=session.session_id, observation=confident)
    prepared = runtime.prepare(
        session_id=session.session_id,
        observation=confident,
    )
    runtime.suggest(
        session_id=session.session_id,
        observation=confident,
        preparation=prepared.preparation,
    )
    runtime.suggest(session_id=session.session_id, observation=low)
    runtime.request_proactive_action(
        session_id=session.session_id,
        description="click button",
    )

    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.observation_count == 1
    assert snapshot.preparation_count == 1
    assert snapshot.suggestion_count == 1
    assert snapshot.suppressed_count == 1
    assert snapshot.blocked_action_count == 1


def test_session_tracks_counts() -> None:
    runtime = ContinuousAssistanceModeRuntime()
    session = runtime.create_session(workspace_id="workspace")
    observation = _observation(confidence=0.90)

    runtime.observe(session_id=session.session_id, observation=observation)
    runtime.prepare(session_id=session.session_id, observation=observation)
    runtime.suggest(session_id=session.session_id, observation=observation)
    runtime.request_proactive_action(
        session_id=session.session_id,
        description="click",
    )

    stored = runtime.session_for(session.session_id)

    assert stored is not None
    assert stored.observation_count == 1
    assert stored.preparation_count == 1
    assert stored.suggestion_count == 1
    assert stored.blocked_action_count == 1


def test_reset_clears_runtime() -> None:
    runtime = ContinuousAssistanceModeRuntime()
    runtime.create_session(workspace_id="workspace")

    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.runtime_event_count == 1
    assert snapshot.last_reason == AssistanceReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert AssistanceObservationKind.BUILD_ERROR.value == "build_error"
    assert AssistanceStatus.SUGGESTED.value == "suggested"
    assert ProactivePermission.ACTION.value == "action"


def _observation(
    *,
    confidence: float = 0.90,
) -> AssistanceObservation:
    return AssistanceObservation(
        kind=AssistanceObservationKind.BUILD_ERROR,
        workspace_id="workspace",
        summary="build failed with AssertionError",
        workflow=WorkflowKind.DEBUGGING,
        confidence=confidence,
    )