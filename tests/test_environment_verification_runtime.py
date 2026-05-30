from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    ActionVerificationStatus,
    ExpectedState,
    ObservedState,
    RecoveryNeededReason,
    VerificationContract,
    VerificationDecision,
    VerificationDeltaKind,
    VerificationReason,
    VerificationRuntime,
    VerificationStateKind,
    VerificationTargetKind,
    expected_bool_state,
    expected_hash_state,
    observed_bool_state,
    observed_hash_state,
    state_hash,
)


def test_runtime_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        VerificationRuntime(name=" ")


def test_expected_state_requires_comparable_value() -> None:
    with pytest.raises(ValidationError):
        ExpectedState(
            key="window.visible",
            kind=VerificationStateKind.VISIBLE,
            target=VerificationTargetKind.WINDOW,
            description="window should be visible",
        )


def test_observed_state_requires_comparable_value() -> None:
    with pytest.raises(ValidationError):
        ObservedState(
            key="window.visible",
            kind=VerificationStateKind.VISIBLE,
            target=VerificationTargetKind.WINDOW,
            description="window is visible",
        )


def test_contract_requires_expected_states() -> None:
    with pytest.raises(ValidationError):
        VerificationContract(
            action_id="action",
            workspace_id="workspace",
            expected_states=(),
        )


def test_contract_rejects_duplicate_expected_keys() -> None:
    state = expected_bool_state(
        key="window.visible",
        value=True,
        kind=VerificationStateKind.VISIBLE,
        target=VerificationTargetKind.WINDOW,
        description="window should be visible",
    )

    with pytest.raises(ValidationError):
        VerificationContract(
            action_id="action",
            workspace_id="workspace",
            expected_states=(state, state),
        )


def test_create_session() -> None:
    runtime = VerificationRuntime()
    session = runtime.create_session(workspace_id="workspace")

    assert session.workspace_id == "workspace"
    assert runtime.snapshot().session_count == 1


def test_passes_when_expected_matches_observed() -> None:
    runtime = VerificationRuntime()
    session = runtime.create_session(workspace_id="workspace")
    contract = _contract(
        expected_bool_state(
            key="app.visible",
            value=True,
            kind=VerificationStateKind.VISIBLE,
            target=VerificationTargetKind.APP,
            description="app should be visible",
        )
    )

    result = runtime.verify(
        session_id=session.session_id,
        contract=contract,
        observed_states=(
            observed_bool_state(
                key="app.visible",
                value=True,
                kind=VerificationStateKind.VISIBLE,
                target=VerificationTargetKind.APP,
                description="app is visible",
            ),
        ),
    )

    assert result.status == ActionVerificationStatus.PASSED
    assert result.decision == VerificationDecision.COMPLETE
    assert result.action_complete is True
    assert result.recovery_needed is False
    assert result.recovery_reason == RecoveryNeededReason.NONE
    assert result.audit.raw_state_logged is False


def test_hash_state_passes_without_raw_value_logging() -> None:
    runtime = VerificationRuntime()
    session = runtime.create_session(workspace_id="workspace")
    contract = _contract(
        expected_hash_state(
            key="clipboard.hash",
            value="safe text",
            kind=VerificationStateKind.HASH_EQUALS,
            target=VerificationTargetKind.CLIPBOARD,
            description="clipboard hash should match",
        )
    )

    result = runtime.verify(
        session_id=session.session_id,
        contract=contract,
        observed_states=(
            observed_hash_state(
                key="clipboard.hash",
                value="safe text",
                kind=VerificationStateKind.HASH_EQUALS,
                target=VerificationTargetKind.CLIPBOARD,
                description="clipboard hash matched",
            ),
        ),
    )

    assert result.status == ActionVerificationStatus.PASSED
    assert result.observed_states[0].observed_hash == state_hash("safe text")
    assert result.audit.raw_state_logged is False


def test_missing_observed_state_requires_recovery() -> None:
    runtime = VerificationRuntime()
    session = runtime.create_session(workspace_id="workspace")
    contract = _contract(
        expected_bool_state(
            key="file.exists",
            value=True,
            kind=VerificationStateKind.EXISTS,
            target=VerificationTargetKind.FILE,
            description="file should exist",
        )
    )

    result = runtime.verify(
        session_id=session.session_id,
        contract=contract,
        observed_states=(),
    )

    assert result.status == ActionVerificationStatus.RECOVERY_NEEDED
    assert result.decision == VerificationDecision.REQUIRE_RECOVERY
    assert result.recovery_needed is True
    assert result.deltas[0].kind == VerificationDeltaKind.MISSING_OBSERVED_STATE


def test_value_mismatch_requires_recovery() -> None:
    runtime = VerificationRuntime()
    session = runtime.create_session(workspace_id="workspace")
    contract = _contract(
        expected_bool_state(
            key="terminal.done",
            value=True,
            kind=VerificationStateKind.COMMAND_COMPLETED,
            target=VerificationTargetKind.TERMINAL,
            description="command should complete",
        )
    )

    result = runtime.verify(
        session_id=session.session_id,
        contract=contract,
        observed_states=(
            observed_bool_state(
                key="terminal.done",
                value=False,
                kind=VerificationStateKind.COMMAND_COMPLETED,
                target=VerificationTargetKind.TERMINAL,
                description="command still running",
            ),
        ),
    )

    assert result.status == ActionVerificationStatus.RECOVERY_NEEDED
    assert result.recovery_needed is True
    assert result.deltas[0].kind == VerificationDeltaKind.VALUE_MISMATCH


def test_low_confidence_needs_review() -> None:
    runtime = VerificationRuntime()
    session = runtime.create_session(workspace_id="workspace")
    contract = _contract(
        expected_bool_state(
            key="dialog.closed",
            value=True,
            kind=VerificationStateKind.VISIBLE,
            target=VerificationTargetKind.WINDOW,
            description="dialog should be closed",
        )
    )

    result = runtime.verify(
        session_id=session.session_id,
        contract=contract,
        observed_states=(
            observed_bool_state(
                key="dialog.closed",
                value=True,
                kind=VerificationStateKind.VISIBLE,
                target=VerificationTargetKind.WINDOW,
                description="dialog appears closed",
                confidence=0.40,
            ),
        ),
    )

    assert result.status == ActionVerificationStatus.NEEDS_REVIEW
    assert result.decision == VerificationDecision.REQUIRE_REVIEW
    assert result.deltas[0].kind == VerificationDeltaKind.LOW_CONFIDENCE_OBSERVATION


def test_extra_observed_state_needs_review() -> None:
    runtime = VerificationRuntime()
    session = runtime.create_session(workspace_id="workspace")
    contract = _contract(
        expected_bool_state(
            key="app.focused",
            value=True,
            kind=VerificationStateKind.FOCUSED,
            target=VerificationTargetKind.APP,
            description="app should be focused",
        )
    )

    result = runtime.verify(
        session_id=session.session_id,
        contract=contract,
        observed_states=(
            observed_bool_state(
                key="app.focused",
                value=True,
                kind=VerificationStateKind.FOCUSED,
                target=VerificationTargetKind.APP,
                description="app focused",
            ),
            observed_bool_state(
                key="unexpected.popup",
                value=True,
                kind=VerificationStateKind.VISIBLE,
                target=VerificationTargetKind.WINDOW,
                description="unexpected popup appeared",
            ),
        ),
    )

    assert result.status == ActionVerificationStatus.NEEDS_REVIEW
    assert any(
        delta.kind == VerificationDeltaKind.EXTRA_OBSERVED_STATE
        for delta in result.deltas
    )


def test_missing_session_fails() -> None:
    runtime = VerificationRuntime()
    contract = _contract(
        expected_bool_state(
            key="app.visible",
            value=True,
            kind=VerificationStateKind.VISIBLE,
            target=VerificationTargetKind.APP,
            description="app should be visible",
        )
    )

    result = runtime.verify(
        session_id="missing",
        contract=contract,
        observed_states=(),
    )

    assert result.status == ActionVerificationStatus.FAILED
    assert result.reason == VerificationReason.SESSION_NOT_FOUND
    assert result.recovery_needed is True


def test_result_rejects_complete_with_recovery() -> None:
    runtime = VerificationRuntime()
    session = runtime.create_session(workspace_id="workspace")
    contract = _contract(
        expected_bool_state(
            key="app.visible",
            value=True,
            kind=VerificationStateKind.VISIBLE,
            target=VerificationTargetKind.APP,
            description="app should be visible",
        )
    )
    result = runtime.verify(
        session_id=session.session_id,
        contract=contract,
        observed_states=(),
    )

    with pytest.raises(ValidationError):
        type(result)(
            status=ActionVerificationStatus.RECOVERY_NEEDED,
            decision=result.decision,
            reason=result.reason,
            contract=result.contract,
            observed_states=result.observed_states,
            deltas=result.deltas,
            trust_score=result.trust_score,
            trust=result.trust,
            recovery_needed=True,
            recovery_reason=result.recovery_reason,
            audit=result.audit,
            action_complete=True,
            message="invalid",
        )


def test_snapshot_tracks_counts() -> None:
    runtime = VerificationRuntime()
    session = runtime.create_session(workspace_id="workspace")
    contract = _contract(
        expected_bool_state(
            key="app.visible",
            value=True,
            kind=VerificationStateKind.VISIBLE,
            target=VerificationTargetKind.APP,
            description="app should be visible",
        )
    )

    runtime.verify(
        session_id=session.session_id,
        contract=contract,
        observed_states=(
            observed_bool_state(
                key="app.visible",
                value=True,
                kind=VerificationStateKind.VISIBLE,
                target=VerificationTargetKind.APP,
                description="app visible",
            ),
        ),
    )
    runtime.verify(
        session_id=session.session_id,
        contract=contract,
        observed_states=(),
    )
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.result_count == 2
    assert snapshot.passed_count == 1
    assert snapshot.recovery_needed_count == 1
    assert snapshot.audit_count == 2


def test_session_tracks_verification_and_recovery_counts() -> None:
    runtime = VerificationRuntime()
    session = runtime.create_session(workspace_id="workspace")
    contract = _contract(
        expected_bool_state(
            key="app.visible",
            value=True,
            kind=VerificationStateKind.VISIBLE,
            target=VerificationTargetKind.APP,
            description="app should be visible",
        )
    )

    runtime.verify(
        session_id=session.session_id,
        contract=contract,
        observed_states=(),
    )
    stored = runtime.session_for(session.session_id)

    assert stored is not None
    assert stored.verification_count == 1
    assert stored.recovery_needed_count == 1


def test_reset_clears_runtime() -> None:
    runtime = VerificationRuntime()
    runtime.create_session(workspace_id="workspace")

    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.runtime_event_count == 1
    assert snapshot.last_reason == VerificationReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert VerificationTargetKind.APP.value == "app"
    assert VerificationStateKind.VISIBLE.value == "visible"
    assert ActionVerificationStatus.PASSED.value == "passed"


def _contract(*states: ExpectedState) -> VerificationContract:
    return VerificationContract(
        action_id="action_1",
        workspace_id="workspace",
        expected_states=states,
    )