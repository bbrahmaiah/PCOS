from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.tools import (
    ActionAuditEventKind,
    ActionAuditLog,
    AutonomousStepKind,
    AutonomousTaskDecision,
    AutonomousTaskMode,
    AutonomousTaskReason,
    AutonomousTaskRequest,
    AutonomousTaskResult,
    AutonomousTaskState,
    AutonomousTaskStep,
    CognitionToolBridge,
    CognitionToolBridgeConfig,
    CognitionToolDecision,
    CognitionToolIntent,
    CognitionToolMode,
    CognitionToolReason,
    RealActionSmokeReason,
    RealActionSmokeRequest,
    RealActionSmokeResult,
    RealActionSmokeStatus,
)


class FakeSmokeRunner:
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
            message="smoke ok" if self.success else "smoke failed",
        )


class FakeAutonomousRunner:
    def __init__(self, *, success: bool = True) -> None:
        self.success = success
        self.requests: list[AutonomousTaskRequest] = []

    def run(self, request: AutonomousTaskRequest) -> AutonomousTaskResult:
        self.requests.append(request)

        return AutonomousTaskResult(
            task_id=request.task_id,
            objective=request.objective,
            state=(
                AutonomousTaskState.SUCCEEDED
                if self.success
                else AutonomousTaskState.FAILED
            ),
            decision=(
                AutonomousTaskDecision.COMPLETED
                if self.success
                else AutonomousTaskDecision.FAILED
            ),
            reason=(
                AutonomousTaskReason.TASK_SUCCEEDED
                if self.success
                else AutonomousTaskReason.TASK_FAILED
            ),
            success=self.success,
            message="autonomous ok" if self.success else "autonomous failed",
        )


def intent(
    *,
    mode: CognitionToolMode = CognitionToolMode.PLAN_ONLY,
    allow_execution: bool = False,
    allow_autonomy: bool = False,
    autonomous_task: AutonomousTaskRequest | None = None,
) -> CognitionToolIntent:
    return CognitionToolIntent(
        mode=mode,
        user_text="run tests and summarize failures",
        goal="prove governed cognition-tool bridge",
        allow_execution=allow_execution,
        allow_autonomy=allow_autonomy,
        autonomous_task=autonomous_task,
    )


def autonomous_task() -> AutonomousTaskRequest:
    return AutonomousTaskRequest(
        objective="safe autonomous bridge task",
        mode=AutonomousTaskMode.READ_ONLY,
        steps=(
            AutonomousTaskStep(
                order=0,
                kind=AutonomousStepKind.RUN_TESTS,
                instruction="run tests",
            ),
        ),
    )


def test_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        CognitionToolBridgeConfig(name=" ").validate()


def test_intent_requires_text() -> None:
    with pytest.raises(ValidationError):
        CognitionToolIntent(
            user_text=" ",
            goal="test",
        )


def test_autonomous_mode_requires_task() -> None:
    with pytest.raises(ValidationError):
        CognitionToolIntent(
            mode=CognitionToolMode.AUTONOMOUS_TASK,
            user_text="run task",
            goal="task",
        )


def test_plan_only_proposes_action_plan() -> None:
    bridge = CognitionToolBridge()

    result = bridge.handle(intent())

    assert result.success is True
    assert result.decision == CognitionToolDecision.PROPOSED
    assert result.reason == CognitionToolReason.PLAN_PROPOSED
    assert result.proposal is not None


def test_unknown_plan_needs_clarification() -> None:
    bridge = CognitionToolBridge()
    request = CognitionToolIntent(
        user_text="do something magical",
        goal="unknown",
    )

    result = bridge.handle(request)

    assert result.success is False
    assert result.decision == CognitionToolDecision.NEEDS_CLARIFICATION
    assert result.reason == CognitionToolReason.PLANNING_FAILED


def test_smoke_execution_blocked_without_permission() -> None:
    smoke = FakeSmokeRunner()
    bridge = CognitionToolBridge(smoke_runtime=smoke)

    result = bridge.handle(
        intent(mode=CognitionToolMode.SMOKE_EXECUTION)
    )

    assert result.success is False
    assert result.decision == CognitionToolDecision.BLOCKED
    assert result.reason == CognitionToolReason.EXECUTION_NOT_ALLOWED
    assert smoke.requests == []


def test_smoke_execution_runs_when_allowed() -> None:
    smoke = FakeSmokeRunner(success=True)
    bridge = CognitionToolBridge(smoke_runtime=smoke)

    result = bridge.handle(
        intent(
            mode=CognitionToolMode.SMOKE_EXECUTION,
            allow_execution=True,
        )
    )

    assert result.success is True
    assert result.decision == CognitionToolDecision.EXECUTED
    assert result.reason == CognitionToolReason.SMOKE_EXECUTION_SUCCEEDED
    assert result.smoke_result is not None
    assert len(smoke.requests) == 1


def test_smoke_execution_failure_returns_failed() -> None:
    smoke = FakeSmokeRunner(success=False)
    bridge = CognitionToolBridge(smoke_runtime=smoke)

    result = bridge.handle(
        intent(
            mode=CognitionToolMode.SMOKE_EXECUTION,
            allow_execution=True,
        )
    )

    assert result.success is False
    assert result.decision == CognitionToolDecision.FAILED
    assert result.reason == CognitionToolReason.SMOKE_EXECUTION_FAILED


def test_smoke_execution_disabled_by_config() -> None:
    smoke = FakeSmokeRunner(success=True)
    bridge = CognitionToolBridge(
        config=CognitionToolBridgeConfig(allow_smoke_execution=False),
        smoke_runtime=smoke,
    )

    result = bridge.handle(
        intent(
            mode=CognitionToolMode.SMOKE_EXECUTION,
            allow_execution=True,
        )
    )

    assert result.success is False
    assert result.decision == CognitionToolDecision.BLOCKED
    assert smoke.requests == []


def test_autonomous_execution_blocked_by_default() -> None:
    autonomous = FakeAutonomousRunner()
    bridge = CognitionToolBridge(autonomous_runtime=autonomous)

    result = bridge.handle(
        intent(
            mode=CognitionToolMode.AUTONOMOUS_TASK,
            allow_autonomy=True,
            autonomous_task=autonomous_task(),
        )
    )

    assert result.success is False
    assert result.decision == CognitionToolDecision.BLOCKED
    assert result.reason == CognitionToolReason.AUTONOMY_NOT_ALLOWED
    assert autonomous.requests == []


def test_autonomous_execution_runs_when_enabled() -> None:
    autonomous = FakeAutonomousRunner(success=True)
    bridge = CognitionToolBridge(
        config=CognitionToolBridgeConfig(allow_autonomous_tasks=True),
        autonomous_runtime=autonomous,
    )

    result = bridge.handle(
        intent(
            mode=CognitionToolMode.AUTONOMOUS_TASK,
            allow_autonomy=True,
            autonomous_task=autonomous_task(),
        )
    )

    assert result.success is True
    assert result.decision == CognitionToolDecision.AUTONOMOUS_EXECUTED
    assert result.reason == CognitionToolReason.AUTONOMOUS_TASK_SUCCEEDED
    assert result.autonomous_result is not None
    assert len(autonomous.requests) == 1


def test_audit_records_intent_and_plan() -> None:
    audit = ActionAuditLog()
    bridge = CognitionToolBridge(audit_log=audit)

    result = bridge.handle(intent())
    event_kinds = [record.event_kind for record in audit.all_records()]

    assert result.success is True
    assert ActionAuditEventKind.INTENT_RECEIVED in event_kinds
    assert ActionAuditEventKind.PLAN_PROPOSED in event_kinds


def test_snapshot_and_reset() -> None:
    bridge = CognitionToolBridge()

    bridge.handle(intent())
    snapshot = bridge.snapshot()

    assert snapshot.request_count == 1
    assert snapshot.proposed_count == 1
    assert snapshot.last_decision == CognitionToolDecision.PROPOSED

    bridge.reset()
    reset_snapshot = bridge.snapshot()

    assert reset_snapshot.request_count == 0
    assert reset_snapshot.last_decision is None


def test_enum_values_are_stable() -> None:
    assert CognitionToolMode.PLAN_ONLY.value == "plan_only"
    assert CognitionToolDecision.EXECUTED.value == "executed"
    assert CognitionToolReason.PLAN_PROPOSED.value == "plan_proposed"