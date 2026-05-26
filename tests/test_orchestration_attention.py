from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.orchestration import (
    AttentionContext,
    AttentionDecision,
    AttentionFocusKind,
    AttentionPolicy,
    AttentionReason,
    AttentionRuntime,
    AttentionRuntimeConfig,
    AttentionUrgency,
    FocusFrame,
    FocusStack,
    ResourceBudget,
    ResourceKind,
    TaskKind,
    TaskPriority,
    TaskRequest,
    WorkerCapability,
    new_task_id,
)


def budget() -> ResourceBudget:
    return ResourceBudget(resource=ResourceKind.WORKER_SLOT, amount=1)


def task(
    *,
    kind: TaskKind = TaskKind.COGNITION,
    priority: TaskPriority = TaskPriority.NORMAL,
    background: bool = False,
    timeout_ms: int | None = None,
) -> TaskRequest:
    return TaskRequest(
        kind=kind,
        priority=priority,
        name="attention task",
        description="attention task",
        required_capabilities=(WorkerCapability.COGNITION,),
        resource_budgets=(budget(),),
        timeout_ms=timeout_ms,
        background=background,
    )


def conversation_task() -> TaskRequest:
    return TaskRequest(
        kind=TaskKind.CONVERSATION_TURN,
        priority=TaskPriority.CRITICAL,
        name="conversation task",
        description="conversation task",
        required_capabilities=(WorkerCapability.CONVERSATION,),
        resource_budgets=(budget(),),
        timeout_ms=5_000,
    )


def background_task() -> TaskRequest:
    return TaskRequest(
        kind=TaskKind.BACKGROUND_MAINTENANCE,
        priority=TaskPriority.BACKGROUND,
        name="background task",
        description="background task",
        required_capabilities=(WorkerCapability.BACKGROUND,),
        resource_budgets=(budget(),),
        background=True,
    )


def memory_task() -> TaskRequest:
    return TaskRequest(
        kind=TaskKind.MEMORY_RETRIEVAL,
        priority=TaskPriority.NORMAL,
        name="memory task",
        description="memory task",
        required_capabilities=(WorkerCapability.MEMORY,),
        resource_budgets=(budget(),),
    )


def tool_task() -> TaskRequest:
    return TaskRequest(
        kind=TaskKind.TOOL_ACTION,
        priority=TaskPriority.NORMAL,
        name="tool task",
        description="tool task",
        required_capabilities=(WorkerCapability.TOOL_ACTION,),
        resource_budgets=(budget(),),
    )


def test_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        AttentionRuntimeConfig(name=" ").validate()


def test_focus_frame_requires_description() -> None:
    with pytest.raises(ValidationError):
        FocusFrame(
            focus_id="focus-1",
            kind=AttentionFocusKind.CONVERSATION,
            description=" ",
        )


def test_focus_frame_validates_owner_task_id() -> None:
    with pytest.raises(ValidationError):
        FocusFrame(
            focus_id="focus-1",
            kind=AttentionFocusKind.CONVERSATION,
            owner_task_id="bad-id",
            description="conversation",
        )


def test_focus_stack_push_pop_and_clear() -> None:
    frame = FocusFrame(
        focus_id="focus-1",
        kind=AttentionFocusKind.CONVERSATION,
        owner_task_id=new_task_id(),
        description="conversation",
    )
    stack = FocusStack()

    pushed = stack.push(frame)

    assert stack.empty is True
    assert pushed.empty is False
    assert pushed.current == frame
    assert pushed.contains_kind(AttentionFocusKind.CONVERSATION) is True
    assert pushed.pop().empty is True
    assert pushed.clear().empty is True


def test_attention_policy_rejects_invalid_background_limits() -> None:
    with pytest.raises(ValidationError):
        AttentionPolicy(
            max_background_tasks_during_conversation=2,
            max_background_tasks_idle=1,
        )


def test_context_detects_conversation_protection() -> None:
    context = AttentionContext(active_conversation=True)

    assert context.conversation_protected is True


def test_context_detects_focus_stack_conversation() -> None:
    frame = FocusFrame(
        focus_id="focus-1",
        kind=AttentionFocusKind.CONVERSATION,
        description="conversation",
    )
    context = AttentionContext(focus_stack=FocusStack().push(frame))

    assert context.conversation_protected is True


def test_conversation_task_allowed_during_conversation() -> None:
    runtime = AttentionRuntime(
        context=AttentionContext(active_conversation=True)
    )

    result = runtime.evaluate(conversation_task())

    assert result.allowed is True
    assert result.decision == AttentionDecision.ALLOW
    assert result.reason == AttentionReason.CRITICAL_ALLOWED


def test_memory_allowed_with_yield_during_conversation() -> None:
    runtime = AttentionRuntime(
        context=AttentionContext(active_conversation=True)
    )

    result = runtime.evaluate(memory_task())

    assert result.allowed is True
    assert result.decision == AttentionDecision.ALLOW_WITH_YIELD
    assert result.reason == AttentionReason.MEMORY_ALLOWED_WITH_YIELD


def test_tool_allowed_with_yield_during_conversation() -> None:
    runtime = AttentionRuntime(
        context=AttentionContext(user_waiting=True)
    )

    result = runtime.evaluate(tool_task())

    assert result.allowed is True
    assert result.decision == AttentionDecision.ALLOW_WITH_YIELD
    assert result.reason == AttentionReason.TOOL_ALLOWED_WITH_YIELD


def test_background_deferred_during_conversation() -> None:
    runtime = AttentionRuntime(
        context=AttentionContext(active_conversation=True)
    )

    result = runtime.evaluate(background_task())

    assert result.allowed is False
    assert result.decision == AttentionDecision.DEFER
    assert result.reason == AttentionReason.BACKGROUND_DEFERRED


def test_maintenance_suppressed_with_background_allowed() -> None:
    runtime = AttentionRuntime(
        policy=AttentionPolicy(max_background_tasks_during_conversation=1),
        context=AttentionContext(active_conversation=True),
    )

    result = runtime.evaluate(background_task())

    assert result.allowed is False
    assert result.decision == AttentionDecision.SUPPRESS
    assert result.reason == AttentionReason.MAINTENANCE_SUPPRESSED


def test_critical_task_allowed_even_during_conversation() -> None:
    runtime = AttentionRuntime(
        context=AttentionContext(active_conversation=True)
    )

    result = runtime.evaluate(
        task(priority=TaskPriority.CRITICAL, timeout_ms=5_000)
    )

    assert result.allowed is True
    assert result.reason == AttentionReason.CRITICAL_ALLOWED


def test_background_allowed_when_idle_under_limit() -> None:
    runtime = AttentionRuntime(context=AttentionContext())

    result = runtime.evaluate(background_task())

    assert result.allowed is True
    assert result.decision == AttentionDecision.ALLOW
    assert result.reason == AttentionReason.BACKGROUND_ALLOWED


def test_background_deferred_when_idle_limit_reached() -> None:
    runtime = AttentionRuntime(
        context=AttentionContext(active_background_tasks=2)
    )

    result = runtime.evaluate(background_task())

    assert result.allowed is False
    assert result.decision == AttentionDecision.DEFER
    assert result.reason == AttentionReason.BACKGROUND_DEFERRED


def test_foreground_allowed_when_idle() -> None:
    runtime = AttentionRuntime(context=AttentionContext())

    result = runtime.evaluate(task())

    assert result.allowed is True
    assert result.reason == AttentionReason.FOREGROUND_ALLOWED


def test_policy_disabled_allows_task() -> None:
    runtime = AttentionRuntime(
        policy=AttentionPolicy(enabled=False),
        context=AttentionContext(active_conversation=True),
    )

    result = runtime.evaluate(background_task())

    assert result.allowed is True
    assert result.reason == AttentionReason.POLICY_DISABLED


def test_score_prioritizes_conversation() -> None:
    runtime = AttentionRuntime()

    conversation = runtime.evaluate(conversation_task())
    background = runtime.evaluate(background_task())

    assert conversation.score.score > background.score.score
    assert conversation.score.urgency == AttentionUrgency.CRITICAL


def test_runtime_focus_operations_update_context() -> None:
    runtime = AttentionRuntime()
    frame = FocusFrame(
        focus_id="focus-1",
        kind=AttentionFocusKind.USER_WAITING,
        description="user waiting",
    )

    pushed = runtime.push_focus(frame)

    assert pushed.conversation_protected is True
    assert runtime.snapshot().current_focus == AttentionFocusKind.USER_WAITING

    popped = runtime.pop_focus()

    assert popped.focus_stack.empty is True


def test_update_context_replaces_context() -> None:
    runtime = AttentionRuntime()
    runtime.update_context(AttentionContext(active_conversation=True))

    assert runtime.context.conversation_protected is True


def test_snapshot_counts_evaluations() -> None:
    runtime = AttentionRuntime()

    runtime.evaluate(task())
    runtime.evaluate(background_task())

    snapshot = runtime.snapshot()

    assert snapshot.evaluation_count == 2
    assert snapshot.allow_count == 2
    assert snapshot.last_decision == AttentionDecision.ALLOW


def test_reset_metrics_clears_counts_only() -> None:
    runtime = AttentionRuntime(context=AttentionContext(active_conversation=True))
    runtime.evaluate(memory_task())

    runtime.reset_metrics()
    snapshot = runtime.snapshot()

    assert snapshot.evaluation_count == 0
    assert snapshot.conversation_protected is True


def test_enum_values_are_stable() -> None:
    assert AttentionFocusKind.CONVERSATION.value == "conversation"
    assert AttentionDecision.ALLOW_WITH_YIELD.value == "allow_with_yield"
    assert AttentionReason.BACKGROUND_DEFERRED.value == "background_deferred"
    assert AttentionUrgency.CRITICAL.value == 40