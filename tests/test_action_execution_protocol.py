from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.tools import (
    ActionExecutionDisposition,
    ActionExecutionEvent,
    ActionExecutionEventKind,
    ActionExecutionPriority,
    ActionExecutionProtocol,
    ActionExecutionProtocolConfig,
    ActionExecutionState,
    ActionKind,
    ActionPlan,
    ActionRisk,
    ActionScope,
    ActionStatus,
    ActionStep,
    ToolCapability,
    new_action_id,
)


def plan(
    *,
    risk: ActionRisk = ActionRisk.LOW,
    rollback_supported: bool = False,
    interruptible: bool = True,
) -> ActionPlan:
    action_id = new_action_id()
    step = ActionStep(
        action_id=action_id,
        order=0,
        kind=ActionKind.READ,
        capability=ToolCapability.READ_FILE,
        scope=ActionScope.WORKSPACE,
        risk=risk,
        description="Read file",
        rollback_supported=rollback_supported,
        interruptible=interruptible,
        timeout_ms=30_000 if risk in {ActionRisk.HIGH, ActionRisk.CRITICAL} else None,
    )

    return ActionPlan(
        action_id=action_id,
        goal="Read project file",
        steps=(step,),
        risk=risk,
        scope=ActionScope.WORKSPACE,
        requires_approval=risk in {ActionRisk.HIGH, ActionRisk.CRITICAL},
    )


def advance(
    protocol: ActionExecutionProtocol,
    state: ActionExecutionState,
    *events: ActionExecutionEventKind,
) -> ActionExecutionState:
    current = state

    for event in events:
        current = protocol.transition(
            current,
            event,
            reason=f"apply {event.value}",
        ).next_state

    return current


def test_execution_protocol_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        ActionExecutionProtocolConfig(name=" ").validate()


def test_create_state_from_plan() -> None:
    protocol = ActionExecutionProtocol()
    item = plan()

    state = protocol.create_state(
        item,
        priority=ActionExecutionPriority.HIGH,
        max_attempts=3,
        timeout_ms=30_000,
    )

    assert state.action_id == item.action_id
    assert state.plan_id == item.plan_id
    assert state.status == ActionStatus.CREATED
    assert state.priority == ActionExecutionPriority.HIGH
    assert state.total_steps == 1
    assert state.max_attempts == 3
    assert state.timeout_ms == 30_000


def test_execution_state_rejects_empty_ids() -> None:
    with pytest.raises(ValidationError):
        ActionExecutionState(
            action_id=" ",
            plan_id="plan",
            total_steps=1,
        )


def test_execution_state_rejects_invalid_attempt() -> None:
    with pytest.raises(ValidationError):
        ActionExecutionState(
            action_id="action",
            plan_id="plan",
            total_steps=1,
            attempt=2,
            max_attempts=1,
        )


def test_execution_event_requires_reason() -> None:
    state = ActionExecutionProtocol().create_state(plan())

    with pytest.raises(ValidationError):
        ActionExecutionEvent(
            execution_id=state.execution_id,
            action_id=state.action_id,
            kind=ActionExecutionEventKind.PLAN_ACCEPTED,
            reason=" ",
        )


def test_full_success_lifecycle() -> None:
    protocol = ActionExecutionProtocol()
    state = protocol.create_state(plan())

    state = advance(
        protocol,
        state,
        ActionExecutionEventKind.PLAN_ACCEPTED,
        ActionExecutionEventKind.VALIDATION_STARTED,
        ActionExecutionEventKind.VALIDATION_PASSED,
        ActionExecutionEventKind.START_REQUESTED,
        ActionExecutionEventKind.SUCCEEDED,
    )

    assert state.status == ActionStatus.SUCCEEDED
    assert state.terminal is True
    assert state.started_at is not None
    assert state.completed_at is not None


def test_approval_lifecycle() -> None:
    protocol = ActionExecutionProtocol()
    state = protocol.create_state(plan())

    state = advance(
        protocol,
        state,
        ActionExecutionEventKind.PLAN_ACCEPTED,
        ActionExecutionEventKind.VALIDATION_STARTED,
        ActionExecutionEventKind.APPROVAL_REQUIRED,
    )

    assert state.status == ActionStatus.WAITING_FOR_APPROVAL

    state = advance(
        protocol,
        state,
        ActionExecutionEventKind.APPROVAL_GRANTED,
        ActionExecutionEventKind.START_REQUESTED,
    )

    assert state.status == ActionStatus.RUNNING


def test_approval_denied_blocks_action() -> None:
    protocol = ActionExecutionProtocol()
    state = protocol.create_state(plan())

    state = advance(
        protocol,
        state,
        ActionExecutionEventKind.PLAN_ACCEPTED,
        ActionExecutionEventKind.VALIDATION_STARTED,
        ActionExecutionEventKind.APPROVAL_REQUIRED,
        ActionExecutionEventKind.APPROVAL_DENIED,
    )

    assert state.status == ActionStatus.BLOCKED
    assert state.terminal is True


def test_validation_failure_blocks_action() -> None:
    protocol = ActionExecutionProtocol()
    state = protocol.create_state(plan())

    state = advance(
        protocol,
        state,
        ActionExecutionEventKind.PLAN_ACCEPTED,
        ActionExecutionEventKind.VALIDATION_STARTED,
        ActionExecutionEventKind.VALIDATION_FAILED,
    )

    assert state.status == ActionStatus.BLOCKED


def test_policy_denied_blocks_action() -> None:
    protocol = ActionExecutionProtocol()
    state = protocol.create_state(plan())

    state = advance(
        protocol,
        state,
        ActionExecutionEventKind.PLAN_ACCEPTED,
        ActionExecutionEventKind.VALIDATION_STARTED,
        ActionExecutionEventKind.POLICY_DENIED,
    )

    assert state.status == ActionStatus.BLOCKED


def test_pause_and_resume_lifecycle() -> None:
    protocol = ActionExecutionProtocol()
    state = protocol.create_state(plan())

    state = advance(
        protocol,
        state,
        ActionExecutionEventKind.PLAN_ACCEPTED,
        ActionExecutionEventKind.VALIDATION_STARTED,
        ActionExecutionEventKind.VALIDATION_PASSED,
        ActionExecutionEventKind.START_REQUESTED,
        ActionExecutionEventKind.PAUSE_REQUESTED,
    )

    assert state.status == ActionStatus.PAUSING
    assert state.pause_requested is True

    state = advance(
        protocol,
        state,
        ActionExecutionEventKind.PAUSED,
        ActionExecutionEventKind.RESUME_REQUESTED,
    )

    assert state.status == ActionStatus.RUNNING
    assert state.pause_requested is False


def test_non_interruptible_action_cannot_pause() -> None:
    protocol = ActionExecutionProtocol()
    state = protocol.create_state(plan(interruptible=False))

    running = advance(
        protocol,
        state,
        ActionExecutionEventKind.PLAN_ACCEPTED,
        ActionExecutionEventKind.VALIDATION_STARTED,
        ActionExecutionEventKind.VALIDATION_PASSED,
        ActionExecutionEventKind.START_REQUESTED,
    )
    transition = protocol.transition(
        running,
        ActionExecutionEventKind.PAUSE_REQUESTED,
        reason="try pause",
    )

    assert transition.disposition == ActionExecutionDisposition.REJECTED
    assert transition.next_state.status == ActionStatus.RUNNING


def test_cancel_lifecycle() -> None:
    protocol = ActionExecutionProtocol()
    state = protocol.create_state(plan())

    state = advance(
        protocol,
        state,
        ActionExecutionEventKind.PLAN_ACCEPTED,
        ActionExecutionEventKind.VALIDATION_STARTED,
        ActionExecutionEventKind.VALIDATION_PASSED,
        ActionExecutionEventKind.START_REQUESTED,
        ActionExecutionEventKind.CANCEL_REQUESTED,
    )

    assert state.status == ActionStatus.CANCELLING
    assert state.cancellation_requested is True

    state = advance(protocol, state, ActionExecutionEventKind.CANCELLED)

    assert state.status == ActionStatus.CANCELLED
    assert state.terminal is True


def test_non_cancellable_action_cannot_cancel() -> None:
    protocol = ActionExecutionProtocol()
    state = protocol.create_state(plan()).model_copy(update={"cancellable": False})

    running = advance(
        protocol,
        state,
        ActionExecutionEventKind.PLAN_ACCEPTED,
        ActionExecutionEventKind.VALIDATION_STARTED,
        ActionExecutionEventKind.VALIDATION_PASSED,
        ActionExecutionEventKind.START_REQUESTED,
    )
    transition = protocol.transition(
        running,
        ActionExecutionEventKind.CANCEL_REQUESTED,
        reason="try cancel",
    )

    assert transition.disposition == ActionExecutionDisposition.REJECTED
    assert transition.next_state.status == ActionStatus.RUNNING


def test_timeout_marks_failed() -> None:
    protocol = ActionExecutionProtocol()
    state = protocol.create_state(plan())

    state = advance(
        protocol,
        state,
        ActionExecutionEventKind.PLAN_ACCEPTED,
        ActionExecutionEventKind.VALIDATION_STARTED,
        ActionExecutionEventKind.VALIDATION_PASSED,
        ActionExecutionEventKind.START_REQUESTED,
        ActionExecutionEventKind.TIMEOUT_OCCURRED,
    )

    assert state.status == ActionStatus.FAILED
    assert state.terminal is True


def test_retry_failed_action() -> None:
    protocol = ActionExecutionProtocol()
    state = protocol.create_state(plan(), max_attempts=2)

    failed = advance(
        protocol,
        state,
        ActionExecutionEventKind.PLAN_ACCEPTED,
        ActionExecutionEventKind.VALIDATION_STARTED,
        ActionExecutionEventKind.VALIDATION_PASSED,
        ActionExecutionEventKind.START_REQUESTED,
        ActionExecutionEventKind.FAILED,
    )

    transition = protocol.transition(
        failed,
        ActionExecutionEventKind.RETRY_REQUESTED,
        reason="retry after failure",
    )

    assert transition.disposition == ActionExecutionDisposition.APPLIED
    assert transition.next_state.status == ActionStatus.PLANNED
    assert transition.next_state.attempt == 2


def test_retry_rejected_when_attempts_exhausted() -> None:
    protocol = ActionExecutionProtocol()
    state = protocol.create_state(plan(), max_attempts=1)

    failed = advance(
        protocol,
        state,
        ActionExecutionEventKind.PLAN_ACCEPTED,
        ActionExecutionEventKind.VALIDATION_STARTED,
        ActionExecutionEventKind.VALIDATION_PASSED,
        ActionExecutionEventKind.START_REQUESTED,
        ActionExecutionEventKind.FAILED,
    )
    transition = protocol.transition(
        failed,
        ActionExecutionEventKind.RETRY_REQUESTED,
        reason="retry after failure",
    )

    assert transition.disposition == ActionExecutionDisposition.REJECTED


def test_retry_can_be_disabled() -> None:
    protocol = ActionExecutionProtocol(
        config=ActionExecutionProtocolConfig(allow_retry_from_failed=False)
    )
    state = protocol.create_state(plan(), max_attempts=2)

    failed = advance(
        protocol,
        state,
        ActionExecutionEventKind.PLAN_ACCEPTED,
        ActionExecutionEventKind.VALIDATION_STARTED,
        ActionExecutionEventKind.VALIDATION_PASSED,
        ActionExecutionEventKind.START_REQUESTED,
        ActionExecutionEventKind.FAILED,
    )
    transition = protocol.transition(
        failed,
        ActionExecutionEventKind.RETRY_REQUESTED,
        reason="retry after failure",
    )

    assert transition.disposition == ActionExecutionDisposition.REJECTED


def test_rollback_failed_action() -> None:
    protocol = ActionExecutionProtocol()
    state = protocol.create_state(plan(rollback_supported=True))

    failed = advance(
        protocol,
        state,
        ActionExecutionEventKind.PLAN_ACCEPTED,
        ActionExecutionEventKind.VALIDATION_STARTED,
        ActionExecutionEventKind.VALIDATION_PASSED,
        ActionExecutionEventKind.START_REQUESTED,
        ActionExecutionEventKind.FAILED,
    )
    transition = protocol.transition(
        failed,
        ActionExecutionEventKind.ROLLBACK_COMPLETED,
        reason="rollback applied",
    )

    assert transition.disposition == ActionExecutionDisposition.APPLIED
    assert transition.next_state.status == ActionStatus.ROLLED_BACK


def test_rollback_rejected_without_support() -> None:
    protocol = ActionExecutionProtocol()
    state = protocol.create_state(plan(rollback_supported=False))

    failed = advance(
        protocol,
        state,
        ActionExecutionEventKind.PLAN_ACCEPTED,
        ActionExecutionEventKind.VALIDATION_STARTED,
        ActionExecutionEventKind.VALIDATION_PASSED,
        ActionExecutionEventKind.START_REQUESTED,
        ActionExecutionEventKind.FAILED,
    )
    transition = protocol.transition(
        failed,
        ActionExecutionEventKind.ROLLBACK_COMPLETED,
        reason="rollback attempted",
    )

    assert transition.disposition == ActionExecutionDisposition.REJECTED


def test_invalid_transition_is_ignored() -> None:
    protocol = ActionExecutionProtocol()
    state = protocol.create_state(plan())

    transition = protocol.transition(
        state,
        ActionExecutionEventKind.START_REQUESTED,
        reason="invalid start",
    )

    assert transition.disposition == ActionExecutionDisposition.IGNORED
    assert transition.changed is False


def test_mismatched_event_is_rejected() -> None:
    protocol = ActionExecutionProtocol()
    state = protocol.create_state(plan())
    event = ActionExecutionEvent(
        execution_id="different",
        action_id=state.action_id,
        kind=ActionExecutionEventKind.PLAN_ACCEPTED,
        reason="wrong execution",
    )

    transition = protocol.apply(state, event)

    assert transition.disposition == ActionExecutionDisposition.REJECTED


def test_terminal_state_ignores_normal_transitions() -> None:
    protocol = ActionExecutionProtocol()
    state = protocol.create_state(plan())

    succeeded = advance(
        protocol,
        state,
        ActionExecutionEventKind.PLAN_ACCEPTED,
        ActionExecutionEventKind.VALIDATION_STARTED,
        ActionExecutionEventKind.VALIDATION_PASSED,
        ActionExecutionEventKind.START_REQUESTED,
        ActionExecutionEventKind.SUCCEEDED,
    )
    transition = protocol.transition(
        succeeded,
        ActionExecutionEventKind.CANCEL_REQUESTED,
        reason="too late",
    )

    assert transition.disposition == ActionExecutionDisposition.IGNORED


def test_snapshot_and_reset() -> None:
    protocol = ActionExecutionProtocol()
    state = protocol.create_state(plan())

    protocol.transition(
        state,
        ActionExecutionEventKind.PLAN_ACCEPTED,
        reason="plan accepted",
    )
    snapshot = protocol.snapshot()

    assert snapshot.transition_count == 1
    assert snapshot.applied_count == 1
    assert snapshot.last_status == ActionStatus.PLANNED

    protocol.reset()
    reset_snapshot = protocol.snapshot()

    assert reset_snapshot.transition_count == 0
    assert reset_snapshot.last_status is None


def test_execution_enum_values_are_stable() -> None:
    assert ActionExecutionPriority.CRITICAL.value == "critical"
    assert ActionExecutionEventKind.PLAN_ACCEPTED.value == "plan_accepted"
    assert ActionExecutionEventKind.CANCEL_REQUESTED.value == "cancel_requested"
    assert ActionExecutionDisposition.APPLIED.value == "applied"