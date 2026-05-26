from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.orchestration import (
    OrchestrationEventKind,
    OrchestrationGuardReason,
    OrchestrationState,
    OrchestrationStateContext,
    OrchestrationStateMachine,
    OrchestrationStateMachineConfig,
    OrchestrationTransitionDecision,
    OrchestrationTransitionReason,
    OrchestratorState,
)


def workers() -> OrchestrationStateContext:
    return OrchestrationStateContext(registered_worker_count=1)


def active_work() -> OrchestrationStateContext:
    return OrchestrationStateContext(
        registered_worker_count=1,
        active_task_count=1,
        active_job_count=1,
    )


def pressure() -> OrchestrationStateContext:
    return OrchestrationStateContext(
        registered_worker_count=1,
        active_task_count=1,
        resource_pressure=True,
    )


def failure() -> OrchestrationStateContext:
    return OrchestrationStateContext(
        registered_worker_count=1,
        failed_worker_count=1,
    )


def test_config_requires_name() -> None:
    with pytest.raises(ValidationError):
        OrchestrationStateMachineConfig(name=" ")


def test_initial_state_is_starting() -> None:
    machine = OrchestrationStateMachine()

    assert machine.state.state == OrchestratorState.STARTING
    assert machine.state.version == 0


def test_bootstrap_completed_moves_to_idle() -> None:
    machine = OrchestrationStateMachine()

    result = machine.bootstrap_completed()

    assert result.accepted is True
    assert result.changed is True
    assert result.state.state == OrchestratorState.IDLE
    assert result.transition.reason == OrchestrationTransitionReason.BOOTSTRAP_FINISHED


def test_same_state_transition_is_ignored() -> None:
    machine = OrchestrationStateMachine()

    result = machine.transition(OrchestrationEventKind.BOOTSTRAP_STARTED)

    assert result.changed is False
    assert result.transition.decision == OrchestrationTransitionDecision.IGNORED
    assert result.transition.reason == OrchestrationTransitionReason.SAME_STATE_IGNORED


def test_coordination_requires_workers() -> None:
    machine = OrchestrationStateMachine()
    machine.bootstrap_completed()

    result = machine.start_coordination(OrchestrationStateContext())

    assert result.accepted is False
    assert result.transition.decision == OrchestrationTransitionDecision.REJECTED
    assert (
        result.transition.guard_reason
        == OrchestrationGuardReason.COORDINATION_REQUIRES_WORKERS
    )


def test_start_coordination_from_idle() -> None:
    machine = OrchestrationStateMachine()
    machine.bootstrap_completed()

    result = machine.start_coordination(workers())

    assert result.accepted is True
    assert result.state.state == OrchestratorState.COORDINATING


def test_complete_coordination_returns_idle_without_active_work() -> None:
    machine = OrchestrationStateMachine()
    machine.bootstrap_completed()
    machine.start_coordination(workers())

    result = machine.complete_coordination(workers())

    assert result.accepted is True
    assert result.state.state == OrchestratorState.IDLE


def test_complete_coordination_moves_busy_with_active_work() -> None:
    machine = OrchestrationStateMachine()
    machine.bootstrap_completed()
    machine.start_coordination(workers())

    result = machine.complete_coordination(active_work())

    assert result.accepted is True
    assert result.state.state == OrchestratorState.BUSY


def test_busy_requires_active_tasks() -> None:
    machine = OrchestrationStateMachine()
    machine.bootstrap_completed()

    result = machine.enter_busy(workers())

    assert result.accepted is False
    assert (
        result.transition.guard_reason
        == OrchestrationGuardReason.BUSY_REQUIRES_ACTIVE_TASKS
    )


def test_enter_busy_from_idle_with_active_task() -> None:
    machine = OrchestrationStateMachine()
    machine.bootstrap_completed()

    result = machine.enter_busy(active_work())

    assert result.accepted is True
    assert result.state.state == OrchestratorState.BUSY


def test_load_shedding_requires_resource_pressure() -> None:
    machine = OrchestrationStateMachine()
    machine.bootstrap_completed()
    machine.enter_busy(active_work())

    result = machine.start_load_shedding(active_work())

    assert result.accepted is False
    assert (
        result.transition.guard_reason
        == OrchestrationGuardReason.LOAD_SHEDDING_REQUIRES_PRESSURE
    )


def test_load_shedding_from_busy_with_pressure() -> None:
    machine = OrchestrationStateMachine()
    machine.bootstrap_completed()
    machine.enter_busy(active_work())

    result = machine.start_load_shedding(pressure())

    assert result.accepted is True
    assert result.state.state == OrchestratorState.LOAD_SHEDDING


def test_recovery_requires_failure() -> None:
    machine = OrchestrationStateMachine()
    machine.bootstrap_completed()

    result = machine.start_recovery(workers())

    assert result.accepted is False
    assert (
        result.transition.guard_reason
        == OrchestrationGuardReason.RECOVERY_REQUIRES_FAILURE
    )


def test_recovery_from_idle_with_failure() -> None:
    machine = OrchestrationStateMachine()
    machine.bootstrap_completed()

    result = machine.start_recovery(failure())

    assert result.accepted is True
    assert result.state.state == OrchestratorState.RECOVERING


def test_complete_recovery_returns_idle_without_work() -> None:
    machine = OrchestrationStateMachine()
    machine.bootstrap_completed()
    machine.start_recovery(failure())

    result = machine.complete_recovery(workers())

    assert result.accepted is True
    assert result.state.state == OrchestratorState.IDLE


def test_shutdown_rejects_active_work_by_default() -> None:
    machine = OrchestrationStateMachine()
    machine.bootstrap_completed()
    machine.enter_busy(active_work())

    result = machine.request_shutdown(active_work())

    assert result.accepted is False
    assert (
        result.transition.guard_reason
        == OrchestrationGuardReason.ACTIVE_TASKS_BLOCK_SHUTDOWN
    )


def test_shutdown_then_stopped() -> None:
    machine = OrchestrationStateMachine()
    machine.bootstrap_completed()

    shutdown = machine.request_shutdown(workers())
    stopped = machine.complete_shutdown()

    assert shutdown.accepted is True
    assert shutdown.state.state == OrchestratorState.SHUTTING_DOWN
    assert stopped.accepted is True
    assert stopped.state.state == OrchestratorState.STOPPED


def test_stopped_rejects_non_reset_transition() -> None:
    machine = OrchestrationStateMachine()
    machine.bootstrap_completed()
    machine.request_shutdown(workers())
    machine.complete_shutdown()

    result = machine.bootstrap_completed()

    assert result.accepted is False
    assert (
        result.transition.reason
        == OrchestrationTransitionReason.TERMINAL_STATE_REJECTED
    )


def test_reset_from_stopped_returns_starting() -> None:
    machine = OrchestrationStateMachine()
    machine.bootstrap_completed()
    machine.request_shutdown(workers())
    machine.complete_shutdown()

    result = machine.reset()

    assert result.accepted is True
    assert result.state.state == OrchestratorState.STARTING


def test_transition_log_records_all_decisions() -> None:
    machine = OrchestrationStateMachine()

    machine.bootstrap_completed()
    machine.start_coordination(OrchestrationStateContext())
    log = machine.transition_log()

    assert len(log) == 2
    assert log[0].decision == OrchestrationTransitionDecision.APPLIED
    assert log[1].decision == OrchestrationTransitionDecision.REJECTED


def test_snapshot_counts_decisions() -> None:
    machine = OrchestrationStateMachine()

    machine.transition(OrchestrationEventKind.BOOTSTRAP_STARTED)
    machine.bootstrap_completed()
    machine.start_coordination(OrchestrationStateContext())

    snapshot = machine.snapshot()

    assert snapshot.transition_count == 3
    assert snapshot.ignored_count == 1
    assert snapshot.applied_count == 1
    assert snapshot.rejected_count == 1
    assert snapshot.current_state == OrchestratorState.IDLE


def test_runtime_snapshot_reflects_state_context() -> None:
    machine = OrchestrationStateMachine()
    machine.bootstrap_completed()
    machine.start_coordination(workers())

    snapshot = machine.runtime_snapshot()

    assert snapshot.state == OrchestratorState.COORDINATING
    assert snapshot.registered_worker_count == 1


def test_initial_state_can_be_injected() -> None:
    state = OrchestrationState(
        state=OrchestratorState.IDLE,
        context=workers(),
    )
    machine = OrchestrationStateMachine(initial_state=state)

    assert machine.state.state == OrchestratorState.IDLE


def test_enum_values_are_stable() -> None:
    assert OrchestrationEventKind.BOOTSTRAP_COMPLETED.value == "bootstrap_completed"
    assert OrchestrationTransitionDecision.APPLIED.value == "applied"
    assert OrchestrationTransitionReason.GUARD_REJECTED.value == "guard_rejected"
    assert OrchestrationGuardReason.GUARD_PASSED.value == "guard_passed"