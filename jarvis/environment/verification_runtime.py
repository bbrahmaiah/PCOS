from __future__ import annotations

from enum import StrEnum
from hashlib import sha256
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.environment.models import EnvironmentSource, TrustCalibration
from jarvis.environment.trust_runtime import TrustPolicyClassification
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class VerificationTargetKind(StrEnum):
    APP = "app"
    WINDOW = "window"
    UI_ELEMENT = "ui_element"
    FILE = "file"
    TERMINAL = "terminal"
    DOCUMENT = "document"
    CLIPBOARD = "clipboard"
    WORKSPACE_GRAPH = "workspace_graph"
    UNKNOWN = "unknown"


class VerificationStateKind(StrEnum):
    EXISTS = "exists"
    VISIBLE = "visible"
    FOCUSED = "focused"
    TEXT_EQUALS = "text_equals"
    TEXT_CONTAINS = "text_contains"
    HASH_EQUALS = "hash_equals"
    STATUS_EQUALS = "status_equals"
    COUNT_EQUALS = "count_equals"
    COMMAND_COMPLETED = "command_completed"
    FILE_TIMESTAMP_CHANGED = "file_timestamp_changed"


class VerificationDeltaKind(StrEnum):
    MATCHED = "matched"
    MISSING_OBSERVED_STATE = "missing_observed_state"
    VALUE_MISMATCH = "value_mismatch"
    LOW_CONFIDENCE_OBSERVATION = "low_confidence_observation"
    EXTRA_OBSERVED_STATE = "extra_observed_state"


class VerificationStatus(StrEnum):
    PASSED = "passed"
    NEEDS_REVIEW = "needs_review"
    RECOVERY_NEEDED = "recovery_needed"
    FAILED = "failed"


class VerificationDecision(StrEnum):
    COMPLETE = "complete"
    REQUIRE_REVIEW = "require_review"
    REQUIRE_RECOVERY = "require_recovery"
    BLOCKED = "blocked"


class VerificationReason(StrEnum):
    SESSION_CREATED = "session_created"
    VERIFICATION_PASSED = "verification_passed"
    VERIFICATION_NEEDS_REVIEW = "verification_needs_review"
    VERIFICATION_RECOVERY_NEEDED = "verification_recovery_needed"
    CONTRACT_INVALID = "contract_invalid"
    OBSERVED_STATE_MISSING = "observed_state_missing"
    SESSION_NOT_FOUND = "session_not_found"
    RUNTIME_RESET = "runtime_reset"


class VerificationEventKind(StrEnum):
    SESSION_CREATED = "session_created"
    VERIFICATION_COMPLETED = "verification_completed"
    VERIFICATION_BLOCKED = "verification_blocked"
    RUNTIME_RESET = "runtime_reset"


class RecoveryNeededReason(StrEnum):
    NONE = "none"
    MISSING_STATE = "missing_state"
    STATE_MISMATCH = "state_mismatch"
    LOW_TRUST = "low_trust"
    OBSERVATION_INCOMPLETE = "observation_incomplete"


class ExpectedState(OrchestrationModel):
    """
    State JARVIS expects after an action.

    Use hashes for sensitive values. Do not store raw secrets here.
    """

    state_id: str = Field(default_factory=lambda: f"expected_state_{uuid4().hex}")
    key: str
    kind: VerificationStateKind
    target: VerificationTargetKind
    description: str
    expected_hash: str | None = None
    expected_bool: bool | None = None
    expected_number: int | None = None
    required: bool = True
    confidence: float = Field(default=0.90, ge=0.0, le=1.0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("state_id", "key", "description")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _requires_comparable_value(self) -> ExpectedState:
        has_value = any(
            value is not None
            for value in (
                self.expected_hash,
                self.expected_bool,
                self.expected_number,
            )
        )

        if not has_value:
            raise ValueError("expected state requires a comparable value.")

        return self


class ObservedState(OrchestrationModel):
    """
    State JARVIS actually observes after an action.
    """

    state_id: str = Field(default_factory=lambda: f"observed_state_{uuid4().hex}")
    key: str
    kind: VerificationStateKind
    target: VerificationTargetKind
    description: str
    observed_hash: str | None = None
    observed_bool: bool | None = None
    observed_number: int | None = None
    confidence: float = Field(default=0.90, ge=0.0, le=1.0)
    source: EnvironmentSource = EnvironmentSource.OS_OBSERVER
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("state_id", "key", "description")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _requires_comparable_value(self) -> ObservedState:
        has_value = any(
            value is not None
            for value in (
                self.observed_hash,
                self.observed_bool,
                self.observed_number,
            )
        )

        if not has_value:
            raise ValueError("observed state requires a comparable value.")

        return self


class VerificationDelta(OrchestrationModel):
    delta_id: str = Field(default_factory=lambda: f"verification_delta_{uuid4().hex}")
    kind: VerificationDeltaKind
    key: str
    expected_state_id: str | None = None
    observed_state_id: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    recovery_needed: bool = False
    reason: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("delta_id", "key", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class VerificationTrustScore(OrchestrationModel):
    score_id: str = Field(default_factory=lambda: f"verification_trust_{uuid4().hex}")
    confidence: float = Field(ge=0.0, le=1.0)
    stability: float = Field(ge=0.0, le=1.0)
    ambiguity: float = Field(ge=0.0, le=1.0)
    matched_ratio: float = Field(ge=0.0, le=1.0)
    recovery_needed: bool
    reason: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("score_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    def effective_score(self) -> float:
        return max(
            0.0,
            min(
                1.0,
                (self.confidence * 0.45)
                + (self.stability * 0.35)
                + ((1.0 - self.ambiguity) * 0.20),
            ),
        )


class VerificationContract(OrchestrationModel):
    contract_id: str = Field(
        default_factory=lambda: f"verification_contract_{uuid4().hex}"
    )
    action_id: str
    workspace_id: str
    expected_states: tuple[ExpectedState, ...]
    minimum_confidence: float = Field(default=0.70, ge=0.0, le=1.0)
    require_all_required_states: bool = True
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("contract_id", "action_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _requires_expected_states(self) -> VerificationContract:
        if not self.expected_states:
            raise ValueError("verification contract requires expected states.")

        keys = [state.key for state in self.expected_states]
        if len(keys) != len(set(keys)):
            raise ValueError("expected state keys must be unique.")

        return self


class VerificationAuditRecord(OrchestrationModel):
    audit_id: str = Field(default_factory=lambda: f"verification_audit_{uuid4().hex}")
    contract_id: str
    action_id: str
    status: VerificationStatus
    decision: VerificationDecision
    reason: VerificationReason
    expected_count: int = Field(ge=0)
    observed_count: int = Field(ge=0)
    delta_count: int = Field(ge=0)
    recovery_needed: bool
    raw_state_logged: bool = False
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("audit_id", "contract_id", "action_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _never_log_raw_state(self) -> VerificationAuditRecord:
        if self.raw_state_logged:
            raise ValueError("verification audit must not log raw state.")

        return self


class VerificationResult(OrchestrationModel):
    result_id: str = Field(default_factory=lambda: f"verification_result_{uuid4().hex}")
    status: VerificationStatus
    decision: VerificationDecision
    reason: VerificationReason
    contract: VerificationContract
    observed_states: tuple[ObservedState, ...]
    deltas: tuple[VerificationDelta, ...]
    trust_score: VerificationTrustScore
    trust: TrustCalibration
    recovery_needed: bool
    recovery_reason: RecoveryNeededReason
    audit: VerificationAuditRecord
    action_complete: bool = False
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("result_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _complete_requires_passed(self) -> VerificationResult:
        if self.action_complete and self.status != VerificationStatus.PASSED:
            raise ValueError("action_complete requires PASSED verification.")

        if self.action_complete and self.recovery_needed:
            raise ValueError("action_complete cannot require recovery.")

        return self


class VerificationRuntimeSession(OrchestrationModel):
    session_id: str = Field(
        default_factory=lambda: f"verification_session_{uuid4().hex}"
    )
    workspace_id: str
    verification_count: int = Field(default=0, ge=0)
    recovery_needed_count: int = Field(default=0, ge=0)
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class VerificationRuntimeEvent(OrchestrationModel):
    event_id: str = Field(default_factory=lambda: f"verification_event_{uuid4().hex}")
    kind: VerificationEventKind
    reason: VerificationReason
    session_id: str | None = None
    result_id: str | None = None
    audit_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class VerificationRuntimeSnapshot(OrchestrationModel):
    name: str
    session_count: int = Field(ge=0)
    result_count: int = Field(ge=0)
    passed_count: int = Field(ge=0)
    review_count: int = Field(ge=0)
    recovery_needed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    audit_count: int = Field(ge=0)
    runtime_event_count: int = Field(ge=0)
    last_reason: VerificationReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        return _clean_required(value)


class StateHistoryComparator:
    """
    Compares expected state against observed state and produces deltas.
    """

    def compare(
        self,
        *,
        contract: VerificationContract,
        observed_states: tuple[ObservedState, ...],
    ) -> tuple[VerificationDelta, ...]:
        observed_by_key = {state.key: state for state in observed_states}
        expected_keys = {state.key for state in contract.expected_states}
        deltas: list[VerificationDelta] = []

        for expected in contract.expected_states:
            observed = observed_by_key.get(expected.key)

            if observed is None:
                deltas.append(
                    VerificationDelta(
                        kind=VerificationDeltaKind.MISSING_OBSERVED_STATE,
                        key=expected.key,
                        expected_state_id=expected.state_id,
                        confidence=0.0,
                        recovery_needed=expected.required,
                        reason="required observed state is missing",
                    )
                )
                continue

            if observed.confidence < contract.minimum_confidence:
                deltas.append(
                    VerificationDelta(
                        kind=VerificationDeltaKind.LOW_CONFIDENCE_OBSERVATION,
                        key=expected.key,
                        expected_state_id=expected.state_id,
                        observed_state_id=observed.state_id,
                        confidence=observed.confidence,
                        recovery_needed=False,
                        reason="observed state confidence below threshold",
                    )
                )
                continue

            if not _state_matches(expected, observed):
                deltas.append(
                    VerificationDelta(
                        kind=VerificationDeltaKind.VALUE_MISMATCH,
                        key=expected.key,
                        expected_state_id=expected.state_id,
                        observed_state_id=observed.state_id,
                        confidence=observed.confidence,
                        recovery_needed=expected.required,
                        reason="observed state does not match expected state",
                    )
                )
                continue

            deltas.append(
                VerificationDelta(
                    kind=VerificationDeltaKind.MATCHED,
                    key=expected.key,
                    expected_state_id=expected.state_id,
                    observed_state_id=observed.state_id,
                    confidence=observed.confidence,
                    recovery_needed=False,
                    reason="observed state matched expected state",
                )
            )

        for observed in observed_states:
            if observed.key not in expected_keys:
                deltas.append(
                    VerificationDelta(
                        kind=VerificationDeltaKind.EXTRA_OBSERVED_STATE,
                        key=observed.key,
                        observed_state_id=observed.state_id,
                        confidence=observed.confidence,
                        recovery_needed=False,
                        reason="observed state was not part of contract",
                    )
                )

        return tuple(deltas)


class VerificationRuntime:
    """
    Phase 8 Step 29 Verification Runtime.

    Responsibilities:
    - compare expected state against observed state
    - compute deltas
    - compute trust score
    - decide action completion
    - mark recovery_needed when reality disagrees
    - audit every verification

    Non-responsibilities:
    - no recovery execution
    - no retry execution
    - no undo execution
    """

    def __init__(
        self,
        *,
        name: str = "verification_runtime",
        comparator: StateHistoryComparator | None = None,
    ) -> None:
        cleaned = name.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._comparator = comparator or StateHistoryComparator()
        self._sessions: dict[str, VerificationRuntimeSession] = {}
        self._results: list[VerificationResult] = []
        self._audits: list[VerificationAuditRecord] = []
        self._events: list[VerificationRuntimeEvent] = []
        self._lock = RLock()
        self._last_reason: VerificationReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def create_session(
        self,
        *,
        workspace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> VerificationRuntimeSession:
        session = VerificationRuntimeSession(
            workspace_id=workspace_id,
            metadata=metadata or {},
        )
        event = self._event(
            kind=VerificationEventKind.SESSION_CREATED,
            reason=VerificationReason.SESSION_CREATED,
            session_id=session.session_id,
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._events.append(event)
            self._last_reason = event.reason

        return session

    def verify(
        self,
        *,
        session_id: str,
        contract: VerificationContract,
        observed_states: tuple[ObservedState, ...],
    ) -> VerificationResult:
        if self.session_for(session_id) is None:
            result = self._missing_session_result(
                contract=contract,
                observed_states=observed_states,
            )
            self._record_result(result, session_id)
            return result

        deltas = self._comparator.compare(
            contract=contract,
            observed_states=observed_states,
        )
        result = _build_result(
            contract=contract,
            observed_states=observed_states,
            deltas=deltas,
        )
        self._record_result(result, session_id)

        return result

    def session_for(
        self,
        session_id: str,
    ) -> VerificationRuntimeSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def results(self) -> tuple[VerificationResult, ...]:
        with self._lock:
            return tuple(self._results)

    def audits(self) -> tuple[VerificationAuditRecord, ...]:
        with self._lock:
            return tuple(self._audits)

    def events(self) -> tuple[VerificationRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def snapshot(self) -> VerificationRuntimeSnapshot:
        with self._lock:
            return VerificationRuntimeSnapshot(
                name=self.name,
                session_count=len(self._sessions),
                result_count=len(self._results),
                passed_count=sum(
                    1
                    for result in self._results
                    if result.status == VerificationStatus.PASSED
                ),
                review_count=sum(
                    1
                    for result in self._results
                    if result.status == VerificationStatus.NEEDS_REVIEW
                ),
                recovery_needed_count=sum(
                    1
                    for result in self._results
                    if result.status == VerificationStatus.RECOVERY_NEEDED
                ),
                failed_count=sum(
                    1
                    for result in self._results
                    if result.status == VerificationStatus.FAILED
                ),
                audit_count=len(self._audits),
                runtime_event_count=len(self._events),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        event = self._event(
            kind=VerificationEventKind.RUNTIME_RESET,
            reason=VerificationReason.RUNTIME_RESET,
        )

        with self._lock:
            self._sessions.clear()
            self._results.clear()
            self._audits.clear()
            self._events.clear()
            self._events.append(event)
            self._last_reason = event.reason

    def _missing_session_result(
        self,
        *,
        contract: VerificationContract,
        observed_states: tuple[ObservedState, ...],
    ) -> VerificationResult:
        audit = VerificationAuditRecord(
            contract_id=contract.contract_id,
            action_id=contract.action_id,
            status=VerificationStatus.FAILED,
            decision=VerificationDecision.BLOCKED,
            reason=VerificationReason.SESSION_NOT_FOUND,
            expected_count=len(contract.expected_states),
            observed_count=len(observed_states),
            delta_count=0,
            recovery_needed=True,
        )
        score = VerificationTrustScore(
            confidence=0.0,
            stability=0.0,
            ambiguity=1.0,
            matched_ratio=0.0,
            recovery_needed=True,
            reason="verification session not found",
        )

        return VerificationResult(
            status=VerificationStatus.FAILED,
            decision=VerificationDecision.BLOCKED,
            reason=VerificationReason.SESSION_NOT_FOUND,
            contract=contract,
            observed_states=observed_states,
            deltas=(),
            trust_score=score,
            trust=_trust_from_score(score),
            recovery_needed=True,
            recovery_reason=RecoveryNeededReason.OBSERVATION_INCOMPLETE,
            audit=audit,
            action_complete=False,
            message="verification session not found",
        )

    def _record_result(
        self,
        result: VerificationResult,
        session_id: str,
    ) -> None:
        event = self._event(
            kind=(
                VerificationEventKind.VERIFICATION_COMPLETED
                if result.status
                in {
                    VerificationStatus.PASSED,
                    VerificationStatus.NEEDS_REVIEW,
                    VerificationStatus.RECOVERY_NEEDED,
                }
                else VerificationEventKind.VERIFICATION_BLOCKED
            ),
            reason=result.reason,
            session_id=session_id,
            result_id=result.result_id,
            audit_id=result.audit.audit_id,
            metadata={
                "status": result.status.value,
                "decision": result.decision.value,
                "recovery_needed": result.recovery_needed,
            },
        )

        with self._lock:
            self._results.append(result)
            self._audits.append(result.audit)
            self._events.append(event)
            self._last_reason = result.reason
            self._touch_session(
                session_id=session_id,
                recovery_needed=result.recovery_needed,
            )

    def _touch_session(
        self,
        *,
        session_id: str,
        recovery_needed: bool,
    ) -> None:
        session = self._sessions.get(session_id)

        if session is None:
            return

        self._sessions[session_id] = session.model_copy(
            update={
                "updated_at": utc_now(),
                "verification_count": session.verification_count + 1,
                "recovery_needed_count": session.recovery_needed_count
                + (1 if recovery_needed else 0),
            }
        )

    @staticmethod
    def _event(
        *,
        kind: VerificationEventKind,
        reason: VerificationReason,
        session_id: str | None = None,
        result_id: str | None = None,
        audit_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VerificationRuntimeEvent:
        return VerificationRuntimeEvent(
            kind=kind,
            reason=reason,
            session_id=session_id,
            result_id=result_id,
            audit_id=audit_id,
            metadata=metadata or {},
        )


def state_hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def expected_hash_state(
    *,
    key: str,
    value: str,
    kind: VerificationStateKind,
    target: VerificationTargetKind,
    description: str,
    required: bool = True,
) -> ExpectedState:
    return ExpectedState(
        key=key,
        kind=kind,
        target=target,
        description=description,
        expected_hash=state_hash(value),
        required=required,
    )


def observed_hash_state(
    *,
    key: str,
    value: str,
    kind: VerificationStateKind,
    target: VerificationTargetKind,
    description: str,
    confidence: float = 0.90,
) -> ObservedState:
    return ObservedState(
        key=key,
        kind=kind,
        target=target,
        description=description,
        observed_hash=state_hash(value),
        confidence=confidence,
    )


def expected_bool_state(
    *,
    key: str,
    value: bool,
    kind: VerificationStateKind,
    target: VerificationTargetKind,
    description: str,
    required: bool = True,
) -> ExpectedState:
    return ExpectedState(
        key=key,
        kind=kind,
        target=target,
        description=description,
        expected_bool=value,
        required=required,
    )


def observed_bool_state(
    *,
    key: str,
    value: bool,
    kind: VerificationStateKind,
    target: VerificationTargetKind,
    description: str,
    confidence: float = 0.90,
) -> ObservedState:
    return ObservedState(
        key=key,
        kind=kind,
        target=target,
        description=description,
        observed_bool=value,
        confidence=confidence,
    )


def _build_result(
    *,
    contract: VerificationContract,
    observed_states: tuple[ObservedState, ...],
    deltas: tuple[VerificationDelta, ...],
) -> VerificationResult:
    recovery_needed = any(delta.recovery_needed for delta in deltas)
    blocking_deltas = tuple(
        delta
        for delta in deltas
        if delta.kind
        in {
            VerificationDeltaKind.MISSING_OBSERVED_STATE,
            VerificationDeltaKind.VALUE_MISMATCH,
        }
        and delta.recovery_needed
    )
    review_deltas = tuple(
        delta
        for delta in deltas
        if delta.kind
        in {
            VerificationDeltaKind.LOW_CONFIDENCE_OBSERVATION,
            VerificationDeltaKind.EXTRA_OBSERVED_STATE,
        }
    )
    score = _trust_score_for(
        contract=contract,
        observed_states=observed_states,
        deltas=deltas,
        recovery_needed=recovery_needed,
    )
    status, decision, reason, recovery_reason = _decision_for(
        score=score,
        blocking_delta_count=len(blocking_deltas),
        review_delta_count=len(review_deltas),
        recovery_needed=recovery_needed,
    )
    audit = VerificationAuditRecord(
        contract_id=contract.contract_id,
        action_id=contract.action_id,
        status=status,
        decision=decision,
        reason=reason,
        expected_count=len(contract.expected_states),
        observed_count=len(observed_states),
        delta_count=len(deltas),
        recovery_needed=recovery_needed,
    )

    return VerificationResult(
        status=status,
        decision=decision,
        reason=reason,
        contract=contract,
        observed_states=observed_states,
        deltas=deltas,
        trust_score=score,
        trust=_trust_from_score(score),
        recovery_needed=recovery_needed,
        recovery_reason=recovery_reason,
        audit=audit,
        action_complete=status == VerificationStatus.PASSED,
        message=_message_for(status=status, recovery_needed=recovery_needed),
    )


def _state_matches(expected: ExpectedState, observed: ObservedState) -> bool:
    if expected.kind != observed.kind:
        return False

    if expected.target != observed.target:
        return False

    if expected.expected_hash is not None:
        return expected.expected_hash == observed.observed_hash

    if expected.expected_bool is not None:
        return expected.expected_bool == observed.observed_bool

    if expected.expected_number is not None:
        return expected.expected_number == observed.observed_number

    return False


def _trust_score_for(
    *,
    contract: VerificationContract,
    observed_states: tuple[ObservedState, ...],
    deltas: tuple[VerificationDelta, ...],
    recovery_needed: bool,
) -> VerificationTrustScore:
    matched_count = sum(
        1 for delta in deltas if delta.kind == VerificationDeltaKind.MATCHED
    )
    expected_count = len(contract.expected_states)
    matched_ratio = matched_count / expected_count if expected_count else 0.0

    observed_confidence = 0.0
    if observed_states:
        observed_confidence = sum(
            state.confidence for state in observed_states
        ) / len(observed_states)

    delta_penalty = sum(
        1
        for delta in deltas
        if delta.kind != VerificationDeltaKind.MATCHED
    )
    ambiguity = min(1.0, delta_penalty / max(1, len(deltas)))
    confidence = min(observed_confidence, matched_ratio)
    stability = matched_ratio

    return VerificationTrustScore(
        confidence=confidence,
        stability=stability,
        ambiguity=ambiguity,
        matched_ratio=matched_ratio,
        recovery_needed=recovery_needed,
        reason="verification trust computed from observed state and deltas",
    )


def _decision_for(
    *,
    score: VerificationTrustScore,
    blocking_delta_count: int,
    review_delta_count: int,
    recovery_needed: bool,
) -> tuple[
    VerificationStatus,
    VerificationDecision,
    VerificationReason,
    RecoveryNeededReason,
]:
    if blocking_delta_count > 0 or recovery_needed:
        return (
            VerificationStatus.RECOVERY_NEEDED,
            VerificationDecision.REQUIRE_RECOVERY,
            VerificationReason.VERIFICATION_RECOVERY_NEEDED,
            RecoveryNeededReason.STATE_MISMATCH,
        )

    if score.effective_score() < 0.70 or review_delta_count > 0:
        return (
            VerificationStatus.NEEDS_REVIEW,
            VerificationDecision.REQUIRE_REVIEW,
            VerificationReason.VERIFICATION_NEEDS_REVIEW,
            RecoveryNeededReason.LOW_TRUST,
        )

    return (
        VerificationStatus.PASSED,
        VerificationDecision.COMPLETE,
        VerificationReason.VERIFICATION_PASSED,
        RecoveryNeededReason.NONE,
    )


def _trust_from_score(score: VerificationTrustScore) -> TrustCalibration:
    return TrustCalibration(
        confidence=score.confidence,
        stability=score.stability,
        ambiguity=score.ambiguity,
        source=EnvironmentSource.OS_OBSERVER,
        reason=score.reason,
        metadata={"policy": TrustPolicyClassification.REVIEW.value},
    )


def _message_for(
    *,
    status: VerificationStatus,
    recovery_needed: bool,
) -> str:
    return (
        f"verification status={status.value}; "
        f"recovery_needed={str(recovery_needed).lower()}"
    )


def _clean_required(value: str) -> str:
    cleaned = value.strip()

    if not cleaned:
        raise ValueError("field cannot be empty.")

    return cleaned