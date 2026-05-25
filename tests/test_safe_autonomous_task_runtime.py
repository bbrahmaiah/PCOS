from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.tools import (
    ActionAuditEventKind,
    ActionAuditLog,
    ActionRisk,
    AutonomousStepKind,
    AutonomousTaskMode,
    AutonomousTaskReason,
    AutonomousTaskRequest,
    AutonomousTaskState,
    AutonomousTaskStep,
    RealActionSmokeReason,
    RealActionSmokeRequest,
    RealActionSmokeResult,
    RealActionSmokeStatus,
    SafeAutonomousTaskRuntime,
    SafeAutonomousTaskRuntimeConfig,
)


class FakeSmokeRuntime:
    def __init__(self, *, success: bool = True) -> None:
        self.success = success
        self.requests: list[RealActionSmokeRequest] = []

    def run(self, request: RealActionSmokeRequest) -> RealActionSmokeResult:
        self.requests.append(request)

        return RealActionSmokeResult(
            request_id=request.request_id,
            action_id="action-1",
            status=(
                RealActionSmokeStatus.SUCCEEDED
                if self.success
                else RealActionSmokeStatus.FAILED
            ),
            reason=(
                RealActionSmokeReason.PIPELINE_SUCCEEDED
                if self.success
                else RealActionSmokeReason.DISPATCH_FAILED
            ),
            success=self.success,
            message="ok" if self.success else "failed",
        )


def step(
    *,
    order: int = 0,
    kind: AutonomousStepKind = AutonomousStepKind.RUN_TESTS,
    instruction: str = "run tests",
    target_path: str | None = None,
    search_query: str | None = None,
    risk: ActionRisk = ActionRisk.LOW,
    requires_approval: bool = False,
) -> AutonomousTaskStep:
    return AutonomousTaskStep(
        order=order,
        kind=kind,
        instruction=instruction,
        target_path=target_path,
        search_query=search_query,
        risk=risk,
        requires_approval=requires_approval,
    )


def task(
    *,
    steps: tuple[AutonomousTaskStep, ...] | None = None,
    mode: AutonomousTaskMode = AutonomousTaskMode.READ_ONLY,
    approved: bool = False,
    max_steps: int = 5,
) -> AutonomousTaskRequest:
    return AutonomousTaskRequest(
        objective="autonomous smoke task",
        mode=mode,
        steps=steps or (step(),),
        approved=approved,
        max_steps=max_steps,
    )


def test_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        SafeAutonomousTaskRuntimeConfig(name=" ").validate()

    with pytest.raises(ValueError):
        SafeAutonomousTaskRuntimeConfig(max_steps_per_task=0).validate()


def test_step_requires_target_for_open_file() -> None:
    with pytest.raises(ValidationError):
        step(kind=AutonomousStepKind.OPEN_FILE, instruction="open file")


def test_step_requires_query_for_search() -> None:
    with pytest.raises(ValidationError):
        step(kind=AutonomousStepKind.SEARCH_PROJECT, instruction="search")


def test_task_rejects_empty_steps() -> None:
    with pytest.raises(ValidationError):
        AutonomousTaskRequest(
            objective="empty",
            steps=(),
        )


def test_task_rejects_too_many_steps() -> None:
    with pytest.raises(ValidationError):
        AutonomousTaskRequest(
            objective="too many",
            steps=(
                step(order=0),
                step(order=1),
            ),
            max_steps=1,
        )


def test_successful_autonomous_task() -> None:
    smoke = FakeSmokeRuntime(success=True)
    runtime = SafeAutonomousTaskRuntime(smoke_runtime=smoke)

    result = runtime.run(task())

    assert result.success is True
    assert result.state == AutonomousTaskState.SUCCEEDED
    assert result.reason == AutonomousTaskReason.TASK_SUCCEEDED
    assert len(result.step_results) == 1
    assert result.step_results[0].success is True
    assert len(smoke.requests) == 1


def test_multiple_steps_run_in_order() -> None:
    smoke = FakeSmokeRuntime(success=True)
    runtime = SafeAutonomousTaskRuntime(smoke_runtime=smoke)

    result = runtime.run(
        task(
            steps=(
                step(order=1, kind=AutonomousStepKind.RUN_TESTS),
                step(
                    order=0,
                    kind=AutonomousStepKind.SEARCH_PROJECT,
                    instruction="search memory",
                    search_query="MemoryGateway",
                ),
            )
        )
    )

    assert result.success is True
    assert len(smoke.requests) == 2
    assert smoke.requests[0].planning_request.search_query == "MemoryGateway"


def test_failed_step_stops_task() -> None:
    smoke = FakeSmokeRuntime(success=False)
    runtime = SafeAutonomousTaskRuntime(smoke_runtime=smoke)

    result = runtime.run(task())

    assert result.success is False
    assert result.state == AutonomousTaskState.FAILED
    assert result.reason == AutonomousTaskReason.STEP_FAILED


def test_runtime_blocks_too_many_steps() -> None:
    runtime = SafeAutonomousTaskRuntime(
        config=SafeAutonomousTaskRuntimeConfig(max_steps_per_task=1),
        smoke_runtime=FakeSmokeRuntime(),
    )

    result = runtime.run(
        task(
            steps=(
                step(order=0),
                step(order=1),
            ),
            max_steps=2,
        )
    )

    assert result.success is False
    assert result.state == AutonomousTaskState.BLOCKED
    assert result.reason == AutonomousTaskReason.TOO_MANY_STEPS_BLOCKED


def test_blocks_high_risk_autonomous_step() -> None:
    runtime = SafeAutonomousTaskRuntime(smoke_runtime=FakeSmokeRuntime())

    result = runtime.run(
        task(
            steps=(
                step(risk=ActionRisk.HIGH),
            )
        )
    )

    assert result.success is False
    assert result.state == AutonomousTaskState.BLOCKED
    assert result.reason == AutonomousTaskReason.HIGH_RISK_BLOCKED


def test_blocks_approval_required_without_approval() -> None:
    runtime = SafeAutonomousTaskRuntime(smoke_runtime=FakeSmokeRuntime())

    result = runtime.run(
        task(
            steps=(
                step(requires_approval=True),
            )
        )
    )

    assert result.success is False
    assert result.state == AutonomousTaskState.BLOCKED
    assert result.reason == AutonomousTaskReason.APPROVAL_REQUIRED_BLOCKED


def test_allows_approval_required_when_approved() -> None:
    runtime = SafeAutonomousTaskRuntime(smoke_runtime=FakeSmokeRuntime())

    result = runtime.run(
        task(
            approved=True,
            steps=(
                step(requires_approval=True),
            ),
        )
    )

    assert result.success is True


def test_low_risk_mode_can_be_disabled() -> None:
    runtime = SafeAutonomousTaskRuntime(
        config=SafeAutonomousTaskRuntimeConfig(allow_low_risk_mode=False),
        smoke_runtime=FakeSmokeRuntime(),
    )

    result = runtime.run(task(mode=AutonomousTaskMode.LOW_RISK))

    assert result.success is False
    assert result.reason == AutonomousTaskReason.UNSUPPORTED_MODE


def test_cancel_before_run_finishes() -> None:
    runtime = SafeAutonomousTaskRuntime(smoke_runtime=FakeSmokeRuntime())

    request = task(
        steps=(
            step(order=0),
            step(order=1),
        )
    )
    runtime._interrupt_controller.create_token(  # noqa: SLF001
        action_id=request.task_id
    )
    runtime.cancel(request.task_id)

    result = runtime.run(request)

    assert result.success is True


def test_audit_records_start_and_finish() -> None:
    audit = ActionAuditLog()
    runtime = SafeAutonomousTaskRuntime(
        smoke_runtime=FakeSmokeRuntime(),
        audit_log=audit,
    )

    result = runtime.run(task())
    event_kinds = [record.event_kind for record in audit.all_records()]

    assert result.success is True
    assert ActionAuditEventKind.EXECUTION_STARTED in event_kinds
    assert ActionAuditEventKind.EXECUTION_COMPLETED in event_kinds


def test_snapshot_and_reset() -> None:
    runtime = SafeAutonomousTaskRuntime(smoke_runtime=FakeSmokeRuntime())

    runtime.run(task())
    snapshot = runtime.snapshot()

    assert snapshot.task_count == 1
    assert snapshot.success_count == 1
    assert snapshot.last_state == AutonomousTaskState.SUCCEEDED

    runtime.reset()
    reset_snapshot = runtime.snapshot()

    assert reset_snapshot.task_count == 0
    assert reset_snapshot.last_state is None


def test_enum_values_are_stable() -> None:
    assert AutonomousTaskMode.READ_ONLY.value == "read_only"
    assert AutonomousTaskState.SUCCEEDED.value == "succeeded"
    assert AutonomousStepKind.RUN_TESTS.value == "run_tests"
    assert AutonomousTaskReason.TASK_SUCCEEDED.value == "task_succeeded"