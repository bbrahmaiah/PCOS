from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.orchestration import (
    IntegratedTaskKind,
    IntegratedWorkerKind,
    ProactiveEngine,
    ProactiveOrchestrationConfig,
    ProactiveReason,
    ProactiveRiskLevel,
    ProactiveStatus,
    ProactiveSuggestion,
    ProactiveTaskEnvelope,
    ProactiveTrigger,
    ProactiveTriggerKind,
    ProactiveWorkKind,
    TriggerPolicy,
)


def trigger(
    *,
    kind: ProactiveTriggerKind = ProactiveTriggerKind.USER_PAUSED,
    confidence_percent: int = 90,
    conversation_active: bool = False,
) -> ProactiveTrigger:
    return ProactiveTrigger(
        kind=kind,
        confidence_percent=confidence_percent,
        conversation_active=conversation_active,
        payload={"topic": "jarvis"},
    )


def test_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        ProactiveOrchestrationConfig(name=" ").validate()


def test_config_rejects_invalid_max_envelopes() -> None:
    config = ProactiveOrchestrationConfig(max_envelopes_per_trigger=0)

    with pytest.raises(ValueError):
        config.validate()


def test_trigger_requires_id() -> None:
    with pytest.raises(ValidationError):
        ProactiveTrigger(
            trigger_id=" ",
            kind=ProactiveTriggerKind.USER_PAUSED,
        )


def test_task_envelope_blocks_medium_risk() -> None:
    with pytest.raises(ValidationError):
        ProactiveTaskEnvelope(
            trigger_id="trigger-1",
            work_kind=ProactiveWorkKind.MEMORY_PREFETCH,
            target_worker=IntegratedWorkerKind.MEMORY_WORKER,
            task_kind=IntegratedTaskKind.MEMORY_TASK,
            confidence_percent=90,
            risk_level=ProactiveRiskLevel.MEDIUM,
        )


def test_task_envelope_requires_read_only() -> None:
    with pytest.raises(ValidationError):
        ProactiveTaskEnvelope(
            trigger_id="trigger-1",
            work_kind=ProactiveWorkKind.MEMORY_PREFETCH,
            target_worker=IntegratedWorkerKind.MEMORY_WORKER,
            task_kind=IntegratedTaskKind.MEMORY_TASK,
            confidence_percent=90,
            read_only=False,
        )


def test_task_envelope_requires_cancellable() -> None:
    with pytest.raises(ValidationError):
        ProactiveTaskEnvelope(
            trigger_id="trigger-1",
            work_kind=ProactiveWorkKind.MEMORY_PREFETCH,
            target_worker=IntegratedWorkerKind.MEMORY_WORKER,
            task_kind=IntegratedTaskKind.MEMORY_TASK,
            confidence_percent=90,
            cancellable=False,
        )


def test_task_envelope_must_be_lower_than_reactive() -> None:
    with pytest.raises(ValidationError):
        ProactiveTaskEnvelope(
            trigger_id="trigger-1",
            work_kind=ProactiveWorkKind.MEMORY_PREFETCH,
            target_worker=IntegratedWorkerKind.MEMORY_WORKER,
            task_kind=IntegratedTaskKind.MEMORY_TASK,
            confidence_percent=90,
            lower_than_reactive=False,
        )


def test_task_envelope_blocks_actions() -> None:
    with pytest.raises(ValidationError):
        ProactiveTaskEnvelope(
            trigger_id="trigger-1",
            work_kind=ProactiveWorkKind.MEMORY_PREFETCH,
            target_worker=IntegratedWorkerKind.MEMORY_WORKER,
            task_kind=IntegratedTaskKind.MEMORY_TASK,
            confidence_percent=90,
            action_allowed=True,
        )


def test_tool_path_prewarm_cannot_target_tool_worker() -> None:
    with pytest.raises(ValidationError):
        ProactiveTaskEnvelope(
            trigger_id="trigger-1",
            work_kind=ProactiveWorkKind.TOOL_PATH_PREWARM,
            target_worker=IntegratedWorkerKind.TOOL_WORKER,
            task_kind=IntegratedTaskKind.TOOL_TASK,
            confidence_percent=90,
        )


def test_suggestion_surfaces_only_above_threshold() -> None:
    suggestion = ProactiveSuggestion(
        trigger_id="trigger-1",
        title="Ready",
        message="Prepared.",
        confidence_percent=85,
        surface_threshold_percent=80,
    )

    assert suggestion.can_surface is True


def test_suggestion_suppressed_below_threshold() -> None:
    suggestion = ProactiveSuggestion(
        trigger_id="trigger-1",
        title="Ready",
        message="Prepared.",
        confidence_percent=70,
        surface_threshold_percent=80,
    )

    assert suggestion.can_surface is False


def test_policy_suppresses_when_conversation_active() -> None:
    policy = TriggerPolicy()

    decision = policy.evaluate(trigger(conversation_active=True))

    assert decision.status == ProactiveStatus.SUPPRESSED
    assert decision.reason == ProactiveReason.CONVERSATION_ACTIVE_SUPPRESSED
    assert decision.envelopes == ()


def test_policy_suppresses_low_confidence_trigger() -> None:
    policy = TriggerPolicy(
        config=ProactiveOrchestrationConfig(
            minimum_trigger_confidence_percent=60
        )
    )

    decision = policy.evaluate(trigger(confidence_percent=40))

    assert decision.status == ProactiveStatus.SUPPRESSED
    assert decision.reason == ProactiveReason.LOW_CONFIDENCE_SUPPRESSED


def test_user_pause_creates_memory_prefetch_and_context_prewarm() -> None:
    policy = TriggerPolicy()

    decision = policy.evaluate(trigger(kind=ProactiveTriggerKind.USER_PAUSED))

    work_kinds = {item.work_kind for item in decision.envelopes}

    assert decision.status == ProactiveStatus.ACCEPTED
    assert ProactiveWorkKind.MEMORY_PREFETCH in work_kinds
    assert ProactiveWorkKind.CONTEXT_PREWARM in work_kinds


def test_build_running_creates_monitoring_envelope_and_suggestion() -> None:
    policy = TriggerPolicy()

    decision = policy.evaluate(
        trigger(kind=ProactiveTriggerKind.BUILD_RUNNING)
    )

    assert len(decision.envelopes) == 1
    assert decision.envelopes[0].work_kind == ProactiveWorkKind.BUILD_MONITORING
    assert len(decision.suggestions) == 1
    assert len(decision.user_visible_suggestions) == 1


def test_file_changed_creates_workspace_note() -> None:
    policy = TriggerPolicy()

    decision = policy.evaluate(
        trigger(kind=ProactiveTriggerKind.FILE_CHANGED)
    )

    assert decision.envelopes[0].work_kind == ProactiveWorkKind.WORKSPACE_NOTE


def test_workspace_changed_prewarms_context_and_tool_metadata_only() -> None:
    policy = TriggerPolicy()

    decision = policy.evaluate(
        trigger(kind=ProactiveTriggerKind.WORKSPACE_CHANGED)
    )

    work_kinds = {item.work_kind for item in decision.envelopes}

    assert ProactiveWorkKind.CONTEXT_PREWARM in work_kinds
    assert ProactiveWorkKind.TOOL_PATH_PREWARM in work_kinds
    assert all(
        item.target_worker != IntegratedWorkerKind.TOOL_WORKER
        for item in decision.envelopes
    )


def test_error_pattern_seen_prepares_error_context() -> None:
    policy = TriggerPolicy()

    decision = policy.evaluate(
        trigger(kind=ProactiveTriggerKind.ERROR_PATTERN_SEEN)
    )

    assert decision.envelopes[0].work_kind == (
        ProactiveWorkKind.ERROR_CONTEXT_PREPARE
    )


def test_max_envelopes_per_trigger_is_enforced() -> None:
    policy = TriggerPolicy(
        config=ProactiveOrchestrationConfig(max_envelopes_per_trigger=1)
    )

    decision = policy.evaluate(trigger(kind=ProactiveTriggerKind.USER_PAUSED))

    assert len(decision.envelopes) == 1


def test_engine_handles_trigger() -> None:
    engine = ProactiveEngine()

    result = engine.handle_trigger(trigger())

    assert result.success is True
    assert result.decision is not None
    assert result.decision.status == ProactiveStatus.ACCEPTED
    assert engine.snapshot().trigger_count == 1


def test_engine_pending_envelopes_are_queryable() -> None:
    engine = ProactiveEngine()

    engine.handle_trigger(trigger())
    envelopes = engine.pending_envelopes()

    assert len(envelopes) == 2
    assert all(item.cancellable for item in envelopes)


def test_engine_converts_to_integrated_envelopes() -> None:
    engine = ProactiveEngine()

    engine.handle_trigger(trigger())
    envelopes = engine.integrated_envelopes()

    assert len(envelopes) == 2
    assert all(item.direct_execution_allowed is False for item in envelopes)
    assert all(item.interruptible for item in envelopes)


def test_engine_cancel_envelope() -> None:
    engine = ProactiveEngine()

    engine.handle_trigger(trigger())
    envelope = engine.pending_envelopes()[0]

    result = engine.cancel_envelope(envelope.envelope_id)

    assert result.success is True
    assert result.reason == ProactiveReason.PROACTIVE_CANCELLED
    assert envelope.envelope_id not in {
        item.envelope_id for item in engine.pending_envelopes()
    }


def test_engine_surfaces_high_confidence_suggestions() -> None:
    engine = ProactiveEngine()

    engine.handle_trigger(trigger(kind=ProactiveTriggerKind.BUILD_RUNNING))
    suggestions = engine.surfaced_suggestions()

    assert len(suggestions) == 1
    assert suggestions[0].can_surface is True


def test_engine_does_not_surface_low_confidence_suggestions() -> None:
    engine = ProactiveEngine()

    engine.handle_trigger(
        trigger(
            kind=ProactiveTriggerKind.BUILD_RUNNING,
            confidence_percent=70,
        )
    )

    assert engine.surfaced_suggestions() == ()


def test_engine_snapshot_tracks_counts() -> None:
    engine = ProactiveEngine()

    engine.handle_trigger(trigger(kind=ProactiveTriggerKind.BUILD_RUNNING))
    snapshot = engine.snapshot()

    assert snapshot.trigger_count == 1
    assert snapshot.decision_count == 1
    assert snapshot.envelope_count == 1
    assert snapshot.suggestion_count == 1
    assert snapshot.surfaced_suggestion_count == 1


def test_engine_snapshot_tracks_suppressed() -> None:
    engine = ProactiveEngine()

    engine.handle_trigger(trigger(conversation_active=True))
    snapshot = engine.snapshot()

    assert snapshot.suppressed_count == 1


def test_engine_reset_clears_state() -> None:
    engine = ProactiveEngine()

    engine.handle_trigger(trigger())
    engine.reset()
    snapshot = engine.snapshot()

    assert snapshot.trigger_count == 0
    assert snapshot.decision_count == 0
    assert snapshot.last_reason == ProactiveReason.RUNTIME_RESET


def test_cancel_rejects_empty_id() -> None:
    engine = ProactiveEngine()

    with pytest.raises(ValueError):
        engine.cancel_envelope(" ")


def test_enum_values_are_stable() -> None:
    assert ProactiveTriggerKind.USER_PAUSED.value == "user_paused"
    assert ProactiveWorkKind.MEMORY_PREFETCH.value == "memory_prefetch"
    assert ProactiveRiskLevel.LOW.value == 0
    assert ProactiveStatus.ACCEPTED.value == "accepted"
    assert ProactiveReason.PROACTIVE_BATCH_CREATED.value == (
        "proactive_batch_created"
    )