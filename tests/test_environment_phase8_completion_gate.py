from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    EnvironmentSource,
    Phase8ChecklistResult,
    Phase8CompletionCapability,
    Phase8CompletionChecklistItem,
    Phase8CompletionDecision,
    Phase8CompletionGateRuntime,
    Phase8CompletionReason,
    Phase8CompletionReport,
    Phase8CompletionStatus,
    Phase8GateKind,
    Phase8GateResult,
    TrustCalibration,
)


def test_runtime_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        Phase8CompletionGateRuntime(name=" ")


def test_completion_report_rejects_bad_counts() -> None:
    with pytest.raises(ValidationError):
        Phase8CompletionReport(
            status=Phase8CompletionStatus.SEALED,
            decision=Phase8CompletionDecision.SEAL_PHASE,
            reason=Phase8CompletionReason.PHASE8_SEALED,
            capabilities=(),
            checklist=(),
            gates=(),
            capability_passed_count=1,
            checklist_passed_count=0,
            gate_passed_count=0,
            failed_count=0,
            sealed=True,
            trust=_trust_for_test(),
            message="bad counts",
        )


def test_sealed_report_cannot_have_failures() -> None:
    gate = Phase8GateResult(
        gate=Phase8GateKind.INTEGRATION,
        status=Phase8CompletionStatus.BLOCKED,
        decision=Phase8CompletionDecision.BLOCK_SEAL,
        reason=Phase8CompletionReason.COMPLETION_BLOCKED,
        passed=False,
        message="blocked",
    )

    with pytest.raises(ValidationError):
        Phase8CompletionReport(
            status=Phase8CompletionStatus.SEALED,
            decision=Phase8CompletionDecision.SEAL_PHASE,
            reason=Phase8CompletionReason.PHASE8_SEALED,
            capabilities=(),
            checklist=(),
            gates=(gate,),
            capability_passed_count=0,
            checklist_passed_count=0,
            gate_passed_count=0,
            failed_count=1,
            sealed=True,
            trust=_trust_for_test(),
            message="invalid seal",
        )


def test_create_session() -> None:
    runtime = Phase8CompletionGateRuntime()
    session = runtime.create_session(workspace_id="workspace")

    assert session.workspace_id == "workspace"
    assert runtime.snapshot().session_count == 1


def test_completion_gate_seals_phase8() -> None:
    runtime = Phase8CompletionGateRuntime()
    session = runtime.create_session(workspace_id="workspace")

    report = runtime.run_completion_gate(session_id=session.session_id)

    assert report.status == Phase8CompletionStatus.SEALED
    assert report.decision == Phase8CompletionDecision.SEAL_PHASE
    assert report.reason == Phase8CompletionReason.PHASE8_SEALED
    assert report.sealed is True
    assert report.failed_count == 0


def test_completion_gate_confirms_all_capabilities() -> None:
    runtime = Phase8CompletionGateRuntime()
    session = runtime.create_session(workspace_id="workspace")

    report = runtime.run_completion_gate(session_id=session.session_id)
    capabilities = {result.capability for result in report.capabilities}

    assert capabilities == set(Phase8CompletionCapability)
    assert report.capability_passed_count == len(Phase8CompletionCapability)


def test_completion_gate_confirms_full_checklist() -> None:
    runtime = Phase8CompletionGateRuntime()
    session = runtime.create_session(workspace_id="workspace")

    report = runtime.run_completion_gate(session_id=session.session_id)
    checklist = {result.item for result in report.checklist}

    assert checklist == set(Phase8CompletionChecklistItem)
    assert report.checklist_passed_count == len(Phase8CompletionChecklistItem)


def test_completion_gate_runs_all_final_gates() -> None:
    runtime = Phase8CompletionGateRuntime()
    session = runtime.create_session(workspace_id="workspace")

    report = runtime.run_completion_gate(session_id=session.session_id)
    gates = {result.gate for result in report.gates}

    assert gates == {
        Phase8GateKind.INTEGRATION,
        Phase8GateKind.SECURITY_AUDIT,
        Phase8GateKind.SMOKE,
        Phase8GateKind.LOAD_STABILITY,
    }
    assert all(result.passed for result in report.gates)
    assert report.gate_passed_count == 4


def test_missing_session_fails_gate() -> None:
    runtime = Phase8CompletionGateRuntime()

    report = runtime.run_completion_gate(session_id="missing")

    assert report.status == Phase8CompletionStatus.FAILED
    assert report.reason == Phase8CompletionReason.SESSION_NOT_FOUND
    assert report.sealed is False


def test_snapshot_tracks_sealed_count() -> None:
    runtime = Phase8CompletionGateRuntime()
    session = runtime.create_session(workspace_id="workspace")

    runtime.run_completion_gate(session_id=session.session_id)
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.report_count == 1
    assert snapshot.sealed_count == 1
    assert snapshot.blocked_count == 0


def test_session_tracks_completion_count() -> None:
    runtime = Phase8CompletionGateRuntime()
    session = runtime.create_session(workspace_id="workspace")

    runtime.run_completion_gate(session_id=session.session_id)
    stored = runtime.session_for(session.session_id)

    assert stored is not None
    assert stored.completion_count == 1
    assert stored.sealed_count == 1
    assert stored.blocked_count == 0


def test_reset_clears_runtime() -> None:
    runtime = Phase8CompletionGateRuntime()
    runtime.create_session(workspace_id="workspace")

    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.runtime_event_count == 1
    assert snapshot.last_reason == Phase8CompletionReason.RUNTIME_RESET


def test_checklist_item_model() -> None:
    result = Phase8ChecklistResult(
        item=Phase8CompletionChecklistItem.CLIPBOARD_PROTECTED,
        passed=True,
        reason=Phase8CompletionReason.CHECKLIST_ITEM_PASSED,
        message="clipboard protected",
    )

    assert result.passed is True


def test_enum_values_are_stable() -> None:
    assert Phase8CompletionStatus.SEALED.value == "sealed"
    assert Phase8CompletionCapability.SEE.value == "see"
    assert Phase8GateKind.LOAD_STABILITY.value == "load_stability"


def _trust_for_test() -> TrustCalibration:
    return TrustCalibration(
        confidence=0.9,
        stability=0.9,
        ambiguity=0.1,
        source=EnvironmentSource.OS_OBSERVER,
        reason="test",
    )