from __future__ import annotations

from enum import StrEnum
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.environment.models import (
    ScreenRegion,
)
from jarvis.environment.trust_runtime import TrustPolicyClassification
from jarvis.environment.ui_patterns import UIPatternRecognitionResult
from jarvis.environment.ui_reasoning import (
    UIReasoningResult,
)
from jarvis.environment.ui_semantics import UIContext
from jarvis.environment.visual_grounding import VisualGroundingResult
from jarvis.environment.workspace_graph import WorkspaceCognitiveGraph
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class SimulatedActionKind(StrEnum):
    CLICK = "click"
    TYPE_TEXT = "type_text"
    DELETE = "delete"
    SUBMIT = "submit"
    CLOSE = "close"
    MOVE_FILE = "move_file"
    CHANGE_SETTING = "change_setting"
    OPEN = "open"
    FOCUS = "focus"
    COPY = "copy"
    READ = "read"
    UNKNOWN = "unknown"


class SimulationRiskLevel(StrEnum):
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    IRREVERSIBLE = "irreversible"
    BLOCKED = "blocked"


class SimulationStatus(StrEnum):
    PREDICTED = "predicted"
    NEEDS_VERIFICATION = "needs_verification"
    HIGH_RISK = "high_risk"
    BLOCKED = "blocked"
    LOW_CONFIDENCE = "low_confidence"
    FAILED = "failed"


class SimulationDecision(StrEnum):
    ALLOW_PLANNING = "allow_planning"
    VERIFY_FIRST = "verify_first"
    REQUIRE_APPROVAL = "require_approval"
    BLOCK_ACTION = "block_action"
    ASK_USER = "ask_user"


class SimulationReason(StrEnum):
    SESSION_CREATED = "session_created"
    OUTCOME_PREDICTED = "outcome_predicted"
    EXPECTED_STATE_GENERATED = "expected_state_generated"
    ROLLBACK_RISK_ESTIMATED = "rollback_risk_estimated"
    HIGH_RISK_ACTION = "high_risk_action"
    IRREVERSIBLE_ACTION = "irreversible_action"
    LOW_CONFIDENCE_SIMULATION = "low_confidence_simulation"
    ACTION_BLOCKED_BY_POLICY = "action_blocked_by_policy"
    SESSION_NOT_FOUND = "session_not_found"
    RUNTIME_RESET = "runtime_reset"


class SimulationEventKind(StrEnum):
    SESSION_CREATED = "session_created"
    SIMULATION_COMPLETED = "simulation_completed"
    SIMULATION_BLOCKED = "simulation_blocked"
    RUNTIME_RESET = "runtime_reset"


class ExpectedStateKind(StrEnum):
    DIALOG_CLOSES = "dialog_closes"
    FILE_TIMESTAMP_UPDATES = "file_timestamp_updates"
    TEXT_APPEARS_IN_TARGET = "text_appears_in_target"
    UNSAVED_PROMPT_MAY_APPEAR = "unsaved_prompt_may_appear"
    FILE_REMOVED = "file_removed"
    FILE_MOVED = "file_moved"
    SETTING_CHANGED = "setting_changed"
    APP_FOCUSED = "app_focused"
    CLIPBOARD_UPDATED = "clipboard_updated"
    NO_VISIBLE_CHANGE = "no_visible_change"
    UNKNOWN = "unknown"


class RollbackCapability(StrEnum):
    NOT_NEEDED = "not_needed"
    SIMPLE_UNDO = "simple_undo"
    RESTORE_PREVIOUS_VALUE = "restore_previous_value"
    RESTORE_FILE_BACKUP = "restore_file_backup"
    MANUAL_RECOVERY = "manual_recovery"
    NOT_ROLLBACKABLE = "not_rollbackable"


class SimulationConfidence(OrchestrationModel):
    """
    Confidence for predicted environment outcome.
    """

    confidence_id: str = Field(default_factory=lambda: f"sim_conf_{uuid4().hex}")
    outcome_confidence: float = Field(ge=0.0, le=1.0)
    state_confidence: float = Field(ge=0.0, le=1.0)
    rollback_confidence: float = Field(ge=0.0, le=1.0)
    ambiguity: float = Field(ge=0.0, le=1.0)
    reason: str
    created_at: object = Field(default_factory=utc_now)

    @field_validator("confidence_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    def effective_score(self) -> float:
        return max(
            0.0,
            min(
                1.0,
                (
                    self.outcome_confidence
                    + self.state_confidence
                    + self.rollback_confidence
                    - self.ambiguity
                )
                / 3.0,
            ),
        )


class SimulatedAction(OrchestrationModel):
    """
    Action to simulate.

    This is not executable. It is an intent/action proposal for prediction only.
    """

    action_id: str = Field(default_factory=lambda: f"sim_action_{uuid4().hex}")
    kind: SimulatedActionKind
    description: str
    target_label: str | None = None
    target_region: ScreenRegion | None = None
    text_payload: str | None = None
    source_policy: TrustPolicyClassification = TrustPolicyClassification.REVIEW
    user_initiated: bool = True
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("action_id", "description")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _typing_requires_payload(self) -> SimulatedAction:
        if self.kind == SimulatedActionKind.TYPE_TEXT and not self.text_payload:
            raise ValueError("TYPE_TEXT simulation requires text_payload.")

        return self


class ExpectedStateChange(OrchestrationModel):
    """
    Expected state after simulated action.
    """

    expected_id: str = Field(default_factory=lambda: f"expected_state_{uuid4().hex}")
    kind: ExpectedStateKind
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    verification_hint: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("expected_id", "description", "verification_hint")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class RollbackRiskEstimate(OrchestrationModel):
    """
    Rollback risk estimate.
    """

    estimate_id: str = Field(default_factory=lambda: f"rollback_risk_{uuid4().hex}")
    risk_level: SimulationRiskLevel
    rollback_capability: RollbackCapability
    requires_backup: bool = False
    requires_user_approval: bool = False
    reason: str
    created_at: object = Field(default_factory=utc_now)

    @field_validator("estimate_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class PredictedActionOutcome(OrchestrationModel):
    """
    Predicted action outcome.
    """

    outcome_id: str = Field(default_factory=lambda: f"predicted_outcome_{uuid4().hex}")
    action_kind: SimulatedActionKind
    summary: str
    expected_changes: tuple[ExpectedStateChange, ...]
    rollback_risk: RollbackRiskEstimate
    confidence: SimulationConfidence
    policy: TrustPolicyClassification
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("outcome_id", "summary")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class EnvironmentSimulationRequest(OrchestrationModel):
    """
    Request to simulate an environment action before real interaction.
    """

    request_id: str = Field(default_factory=lambda: f"sim_req_{uuid4().hex}")
    session_id: str
    workspace_id: str
    action: SimulatedAction
    workspace_graph: WorkspaceCognitiveGraph | None = None
    semantic_context: UIContext | None = None
    grounding_result: VisualGroundingResult | None = None
    pattern_result: UIPatternRecognitionResult | None = None
    reasoning_result: UIReasoningResult | None = None
    current_state_summary: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("request_id", "session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class EnvironmentSimulationResult(OrchestrationModel):
    """
    Final simulation result.

    This result can be consumed by Step 23 planning gate.
    """

    result_id: str = Field(default_factory=lambda: f"sim_result_{uuid4().hex}")
    status: SimulationStatus
    decision: SimulationDecision
    reason: SimulationReason
    request_id: str
    action: SimulatedAction
    outcome: PredictedActionOutcome | None = None
    safe_for_planning: bool
    requires_verification: bool
    requires_user_approval: bool
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("result_id", "request_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _safe_requires_outcome(self) -> EnvironmentSimulationResult:
        if self.safe_for_planning and self.outcome is None:
            raise ValueError("safe simulation result requires predicted outcome.")

        return self


class EnvironmentSimulationSession(OrchestrationModel):
    """
    Simulation runtime session.
    """

    session_id: str = Field(default_factory=lambda: f"sim_session_{uuid4().hex}")
    workspace_id: str
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class EnvironmentSimulationRuntimeEvent(OrchestrationModel):
    """
    Simulation runtime event.
    """

    event_id: str = Field(default_factory=lambda: f"sim_event_{uuid4().hex}")
    kind: SimulationEventKind
    reason: SimulationReason
    session_id: str | None = None
    result_id: str | None = None
    action_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class EnvironmentSimulationRuntimeSnapshot(OrchestrationModel):
    """
    Diagnostics for Phase 8 Step 22.
    """

    name: str
    session_count: int = Field(ge=0)
    simulation_count: int = Field(ge=0)
    predicted_count: int = Field(ge=0)
    verification_count: int = Field(ge=0)
    high_risk_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    safe_planning_count: int = Field(ge=0)
    runtime_event_count: int = Field(ge=0)
    last_reason: SimulationReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        return _clean_required(value)


class ExpectedStateGenerator:
    """
    Generates expected state changes for a simulated action.
    """

    def generate(
        self,
        request: EnvironmentSimulationRequest,
    ) -> tuple[ExpectedStateChange, ...]:
        action = request.action

        if action.kind == SimulatedActionKind.CLICK:
            return self._click_expected(request)

        if action.kind == SimulatedActionKind.TYPE_TEXT:
            return (
                ExpectedStateChange(
                    kind=ExpectedStateKind.TEXT_APPEARS_IN_TARGET,
                    description="typed text should appear only in target field",
                    confidence=0.86,
                    verification_hint="verify target text equals typed payload",
                ),
            )

        if action.kind == SimulatedActionKind.CLOSE:
            return (
                ExpectedStateChange(
                    kind=ExpectedStateKind.UNSAVED_PROMPT_MAY_APPEAR,
                    description="closing may produce unsaved changes prompt",
                    confidence=0.72,
                    verification_hint="verify window closed or prompt appeared",
                ),
            )

        if action.kind == SimulatedActionKind.DELETE:
            return (
                ExpectedStateChange(
                    kind=ExpectedStateKind.FILE_REMOVED,
                    description="target file or item may be removed",
                    confidence=0.70,
                    verification_hint="verify item moved to trash or no longer visible",
                ),
            )

        if action.kind == SimulatedActionKind.MOVE_FILE:
            return (
                ExpectedStateChange(
                    kind=ExpectedStateKind.FILE_MOVED,
                    description="file should move to destination path",
                    confidence=0.78,
                    verification_hint="verify source absent and destination present",
                ),
            )

        if action.kind == SimulatedActionKind.CHANGE_SETTING:
            return (
                ExpectedStateChange(
                    kind=ExpectedStateKind.SETTING_CHANGED,
                    description="setting value should change",
                    confidence=0.76,
                    verification_hint="verify setting state after change",
                ),
            )

        if action.kind == SimulatedActionKind.SUBMIT:
            return (
                ExpectedStateChange(
                    kind=ExpectedStateKind.DIALOG_CLOSES,
                    description="submit may close dialog or navigate to next state",
                    confidence=0.74,
                    verification_hint="verify confirmation or next page state",
                ),
            )

        if action.kind == SimulatedActionKind.FOCUS:
            return (
                ExpectedStateChange(
                    kind=ExpectedStateKind.APP_FOCUSED,
                    description="target app/window should become focused",
                    confidence=0.88,
                    verification_hint="verify focused app/window equals target",
                ),
            )

        if action.kind == SimulatedActionKind.COPY:
            return (
                ExpectedStateChange(
                    kind=ExpectedStateKind.CLIPBOARD_UPDATED,
                    description="clipboard should update with selected content",
                    confidence=0.82,
                    verification_hint=(
                        "verify clipboard hash changed, not raw secret content"
                    ),
                ),
            )

        return (
            ExpectedStateChange(
                kind=ExpectedStateKind.UNKNOWN,
                description="unknown expected state",
                confidence=0.30,
                verification_hint="ask user or verify manually",
            ),
        )

    def _click_expected(
        self,
        request: EnvironmentSimulationRequest,
    ) -> tuple[ExpectedStateChange, ...]:
        label = (request.action.target_label or "").lower()

        if "save" in label:
            return (
                ExpectedStateChange(
                    kind=ExpectedStateKind.DIALOG_CLOSES,
                    description="save dialog may close",
                    confidence=0.82,
                    verification_hint="verify dialog closed",
                ),
                ExpectedStateChange(
                    kind=ExpectedStateKind.FILE_TIMESTAMP_UPDATES,
                    description="file timestamp may update",
                    confidence=0.76,
                    verification_hint="verify file modified timestamp changed",
                ),
            )

        if "close" in label or "x" == label.strip():
            return (
                ExpectedStateChange(
                    kind=ExpectedStateKind.UNSAVED_PROMPT_MAY_APPEAR,
                    description="close click may trigger unsaved prompt",
                    confidence=0.74,
                    verification_hint="verify closed window or unsaved prompt",
                ),
            )

        if "delete" in label or "remove" in label:
            return (
                ExpectedStateChange(
                    kind=ExpectedStateKind.FILE_REMOVED,
                    description="delete action may remove selected item",
                    confidence=0.68,
                    verification_hint="verify item removal or trash state",
                ),
            )

        return (
            ExpectedStateChange(
                kind=ExpectedStateKind.NO_VISIBLE_CHANGE,
                description="click may update focus or produce minor UI change",
                confidence=0.70,
                verification_hint="verify target-specific UI state changed",
            ),
        )


class RollbackRiskEstimator:
    """
    Estimates rollback risk before action planning.
    """

    def estimate(
        self,
        action: SimulatedAction,
        expected: tuple[ExpectedStateChange, ...],
    ) -> RollbackRiskEstimate:
        if action.source_policy == TrustPolicyClassification.BLOCKED:
            return RollbackRiskEstimate(
                risk_level=SimulationRiskLevel.BLOCKED,
                rollback_capability=RollbackCapability.NOT_ROLLBACKABLE,
                requires_user_approval=True,
                reason="action source policy is blocked",
            )

        if action.kind == SimulatedActionKind.DELETE:
            return RollbackRiskEstimate(
                risk_level=SimulationRiskLevel.IRREVERSIBLE,
                rollback_capability=RollbackCapability.RESTORE_FILE_BACKUP,
                requires_backup=True,
                requires_user_approval=True,
                reason="delete may be irreversible without backup",
            )

        if action.kind in {
            SimulatedActionKind.SUBMIT,
            SimulatedActionKind.CHANGE_SETTING,
            SimulatedActionKind.MOVE_FILE,
        }:
            return RollbackRiskEstimate(
                risk_level=SimulationRiskLevel.HIGH,
                rollback_capability=RollbackCapability.MANUAL_RECOVERY,
                requires_backup=action.kind == SimulatedActionKind.MOVE_FILE,
                requires_user_approval=True,
                reason="action changes persistent state",
            )

        if action.kind == SimulatedActionKind.TYPE_TEXT:
            return RollbackRiskEstimate(
                risk_level=SimulationRiskLevel.MEDIUM,
                rollback_capability=RollbackCapability.SIMPLE_UNDO,
                requires_user_approval=False,
                reason="typing can usually be undone but must target correct field",
            )

        if action.kind == SimulatedActionKind.CLOSE:
            return RollbackRiskEstimate(
                risk_level=SimulationRiskLevel.MEDIUM,
                rollback_capability=RollbackCapability.MANUAL_RECOVERY,
                requires_user_approval=True,
                reason="closing may lose unsaved state",
            )

        if action.kind in {
            SimulatedActionKind.CLICK,
            SimulatedActionKind.OPEN,
            SimulatedActionKind.FOCUS,
            SimulatedActionKind.COPY,
            SimulatedActionKind.READ,
        }:
            return RollbackRiskEstimate(
                risk_level=SimulationRiskLevel.LOW,
                rollback_capability=RollbackCapability.NOT_NEEDED,
                requires_user_approval=False,
                reason="action is low risk when target is verified",
            )

        return RollbackRiskEstimate(
            risk_level=SimulationRiskLevel.HIGH,
            rollback_capability=RollbackCapability.MANUAL_RECOVERY,
            requires_user_approval=True,
            reason="unknown action kind requires approval",
        )


class SimulationConfidenceEstimator:
    """
    Estimates confidence for the whole simulation.
    """

    def estimate(
        self,
        *,
        request: EnvironmentSimulationRequest,
        expected: tuple[ExpectedStateChange, ...],
        risk: RollbackRiskEstimate,
    ) -> SimulationConfidence:
        base = 0.75

        if request.grounding_result is not None:
            if request.grounding_result.selected is not None:
                base = min(base, request.grounding_result.selected.confidence)
            elif not request.grounding_result.safe_for_action_planning:
                base = min(base, 0.45)

        if request.semantic_context is not None:
            base = min(base, request.semantic_context.scene.confidence)

        if request.pattern_result is not None and request.pattern_result.best_match:
            base = min(base, request.pattern_result.best_match.score)

        if risk.risk_level in {
            SimulationRiskLevel.HIGH,
            SimulationRiskLevel.IRREVERSIBLE,
            SimulationRiskLevel.BLOCKED,
        }:
            base = min(base, 0.62)

        state_confidence = (
            sum(change.confidence for change in expected) / len(expected)
            if expected
            else 0.20
        )
        rollback_confidence = 0.80

        if risk.rollback_capability == RollbackCapability.NOT_ROLLBACKABLE:
            rollback_confidence = 0.20
        elif risk.rollback_capability == RollbackCapability.MANUAL_RECOVERY:
            rollback_confidence = 0.45

        ambiguity = 1.0 - min(base, state_confidence)

        return SimulationConfidence(
            outcome_confidence=max(0.0, min(1.0, base)),
            state_confidence=max(0.0, min(1.0, state_confidence)),
            rollback_confidence=rollback_confidence,
            ambiguity=max(0.0, min(1.0, ambiguity)),
            reason=(
                "simulation confidence estimated from target,"
                "state, and rollback risk"
            ),
        )


class ActionOutcomePredictor:
    """
    Predicts likely action outcome.
    """

    def __init__(
        self,
        *,
        expected_generator: ExpectedStateGenerator | None = None,
        rollback_estimator: RollbackRiskEstimator | None = None,
        confidence_estimator: SimulationConfidenceEstimator | None = None,
    ) -> None:
        self._expected_generator = expected_generator or ExpectedStateGenerator()
        self._rollback_estimator = rollback_estimator or RollbackRiskEstimator()
        self._confidence_estimator = (
            confidence_estimator or SimulationConfidenceEstimator()
        )

    def predict(
        self,
        request: EnvironmentSimulationRequest,
    ) -> PredictedActionOutcome:
        expected = self._expected_generator.generate(request)
        rollback = self._rollback_estimator.estimate(
            request.action,
            expected,
        )
        confidence = self._confidence_estimator.estimate(
            request=request,
            expected=expected,
            risk=rollback,
        )

        return PredictedActionOutcome(
            action_kind=request.action.kind,
            summary=_summary_for(request.action, expected, rollback),
            expected_changes=expected,
            rollback_risk=rollback,
            confidence=confidence,
            policy=_policy_for(rollback),
            metadata={
                "workspace_id": request.workspace_id,
                "current_state_summary": request.current_state_summary,
            },
        )


class StateTransitionSimulator:
    """
    Simulates high-level state transition.

    This is intentionally abstract. It predicts state shape, not raw pixels.
    """

    def simulate(
        self,
        request: EnvironmentSimulationRequest,
        outcome: PredictedActionOutcome,
    ) -> dict[str, Any]:
        return {
            "action": request.action.kind.value,
            "target": request.action.target_label,
            "expected": [change.kind.value for change in outcome.expected_changes],
            "risk": outcome.rollback_risk.risk_level.value,
            "policy": outcome.policy.value,
        }


class EnvironmentSimulationRuntime:
    """
    Phase 8 Step 22 Environment Simulation Runtime.

    Responsibilities:
    - predict likely action outcomes before touching desktop
    - generate expected state
    - estimate rollback risk
    - decide whether planning can continue

    Non-responsibilities:
    - no clicking
    - no typing
    - no file mutation
    - no tool execution
    """

    def __init__(
        self,
        *,
        name: str = "environment_simulation_runtime",
        predictor: ActionOutcomePredictor | None = None,
        transition_simulator: StateTransitionSimulator | None = None,
    ) -> None:
        cleaned = name.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._predictor = predictor or ActionOutcomePredictor()
        self._transition_simulator = transition_simulator or StateTransitionSimulator()
        self._sessions: dict[str, EnvironmentSimulationSession] = {}
        self._results: list[EnvironmentSimulationResult] = []
        self._events: list[EnvironmentSimulationRuntimeEvent] = []
        self._lock = RLock()
        self._last_reason: SimulationReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def create_session(
        self,
        *,
        workspace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> EnvironmentSimulationSession:
        session = EnvironmentSimulationSession(
            workspace_id=workspace_id,
            metadata=metadata or {},
        )
        event = self._event(
            kind=SimulationEventKind.SESSION_CREATED,
            reason=SimulationReason.SESSION_CREATED,
            session_id=session.session_id,
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._events.append(event)
            self._last_reason = event.reason

        return session

    def simulate(
        self,
        request: EnvironmentSimulationRequest,
    ) -> EnvironmentSimulationResult:
        if self.session_for(request.session_id) is None:
            result = EnvironmentSimulationResult(
                status=SimulationStatus.FAILED,
                decision=SimulationDecision.BLOCK_ACTION,
                reason=SimulationReason.SESSION_NOT_FOUND,
                request_id=request.request_id,
                action=request.action,
                outcome=None,
                safe_for_planning=False,
                requires_verification=True,
                requires_user_approval=True,
                message="simulation session not found",
            )
            self._record_result(result, request.session_id)
            return result

        outcome = self._predictor.predict(request)
        transition = self._transition_simulator.simulate(request, outcome)
        status, decision, reason = _status_decision_reason(outcome)
        result = EnvironmentSimulationResult(
            status=status,
            decision=decision,
            reason=reason,
            request_id=request.request_id,
            action=request.action,
            outcome=outcome,
            safe_for_planning=decision
            in {
                SimulationDecision.ALLOW_PLANNING,
                SimulationDecision.VERIFY_FIRST,
            },
            requires_verification=decision != SimulationDecision.ALLOW_PLANNING,
            requires_user_approval=decision
            in {
                SimulationDecision.REQUIRE_APPROVAL,
                SimulationDecision.BLOCK_ACTION,
            },
            message=_result_message(status=status, decision=decision, outcome=outcome),
            metadata={"transition": transition},
        )

        self._record_result(result, request.session_id)
        self._touch_session(request.session_id)

        return result

    def session_for(
        self,
        session_id: str,
    ) -> EnvironmentSimulationSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def results(self) -> tuple[EnvironmentSimulationResult, ...]:
        with self._lock:
            return tuple(self._results)

    def events(self) -> tuple[EnvironmentSimulationRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def snapshot(self) -> EnvironmentSimulationRuntimeSnapshot:
        with self._lock:
            return EnvironmentSimulationRuntimeSnapshot(
                name=self.name,
                session_count=len(self._sessions),
                simulation_count=len(self._results),
                predicted_count=sum(
                    1
                    for result in self._results
                    if result.status == SimulationStatus.PREDICTED
                ),
                verification_count=sum(
                    1
                    for result in self._results
                    if result.status == SimulationStatus.NEEDS_VERIFICATION
                ),
                high_risk_count=sum(
                    1
                    for result in self._results
                    if result.status == SimulationStatus.HIGH_RISK
                ),
                blocked_count=sum(
                    1
                    for result in self._results
                    if result.status
                    in {
                        SimulationStatus.BLOCKED,
                        SimulationStatus.FAILED,
                    }
                ),
                safe_planning_count=sum(
                    1 for result in self._results if result.safe_for_planning
                ),
                runtime_event_count=len(self._events),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        event = self._event(
            kind=SimulationEventKind.RUNTIME_RESET,
            reason=SimulationReason.RUNTIME_RESET,
        )

        with self._lock:
            self._sessions.clear()
            self._results.clear()
            self._events.clear()
            self._events.append(event)
            self._last_reason = event.reason

    def _record_result(
        self,
        result: EnvironmentSimulationResult,
        session_id: str,
    ) -> None:
        event = self._event(
            kind=(
                SimulationEventKind.SIMULATION_COMPLETED
                if result.safe_for_planning
                else SimulationEventKind.SIMULATION_BLOCKED
            ),
            reason=result.reason,
            session_id=session_id,
            result_id=result.result_id,
            action_id=result.action.action_id,
            metadata={"status": result.status.value},
        )

        with self._lock:
            self._results.append(result)
            self._events.append(event)
            self._last_reason = result.reason

    def _touch_session(self, session_id: str) -> None:
        session = self._sessions.get(session_id)

        if session is None:
            return

        self._sessions[session_id] = session.model_copy(
            update={"updated_at": utc_now()}
        )

    @staticmethod
    def _event(
        *,
        kind: SimulationEventKind,
        reason: SimulationReason,
        session_id: str | None = None,
        result_id: str | None = None,
        action_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EnvironmentSimulationRuntimeEvent:
        return EnvironmentSimulationRuntimeEvent(
            kind=kind,
            reason=reason,
            session_id=session_id,
            result_id=result_id,
            action_id=action_id,
            metadata=metadata or {},
        )


def _status_decision_reason(
    outcome: PredictedActionOutcome,
) -> tuple[SimulationStatus, SimulationDecision, SimulationReason]:
    risk = outcome.rollback_risk.risk_level
    score = outcome.confidence.effective_score()

    if outcome.policy == TrustPolicyClassification.BLOCKED:
        return (
            SimulationStatus.BLOCKED,
            SimulationDecision.BLOCK_ACTION,
            SimulationReason.ACTION_BLOCKED_BY_POLICY,
        )

    if risk == SimulationRiskLevel.IRREVERSIBLE:
        return (
            SimulationStatus.HIGH_RISK,
            SimulationDecision.REQUIRE_APPROVAL,
            SimulationReason.IRREVERSIBLE_ACTION,
        )

    if risk == SimulationRiskLevel.HIGH:
        return (
            SimulationStatus.HIGH_RISK,
            SimulationDecision.REQUIRE_APPROVAL,
            SimulationReason.HIGH_RISK_ACTION,
        )

    if score < 0.45:
        return (
            SimulationStatus.LOW_CONFIDENCE,
            SimulationDecision.ASK_USER,
            SimulationReason.LOW_CONFIDENCE_SIMULATION,
        )

    if risk in {SimulationRiskLevel.MEDIUM, SimulationRiskLevel.LOW}:
        return (
            SimulationStatus.NEEDS_VERIFICATION,
            SimulationDecision.VERIFY_FIRST,
            SimulationReason.OUTCOME_PREDICTED,
        )

    return (
        SimulationStatus.PREDICTED,
        SimulationDecision.ALLOW_PLANNING,
        SimulationReason.OUTCOME_PREDICTED,
    )


def _policy_for(risk: RollbackRiskEstimate) -> TrustPolicyClassification:
    if risk.risk_level == SimulationRiskLevel.BLOCKED:
        return TrustPolicyClassification.BLOCKED

    if risk.requires_user_approval:
        return TrustPolicyClassification.VERIFY_FIRST

    if risk.risk_level in {SimulationRiskLevel.MEDIUM, SimulationRiskLevel.LOW}:
        return TrustPolicyClassification.REVIEW

    return TrustPolicyClassification.SAFE


def _summary_for(
    action: SimulatedAction,
    expected: tuple[ExpectedStateChange, ...],
    rollback: RollbackRiskEstimate,
) -> str:
    expected_text = ", ".join(change.kind.value for change in expected)

    return (
        f"simulate {action.kind.value}: expected={expected_text}; "
        f"risk={rollback.risk_level.value}"
    )


def _result_message(
    *,
    status: SimulationStatus,
    decision: SimulationDecision,
    outcome: PredictedActionOutcome,
) -> str:
    return (
        f"simulation {status.value}; decision={decision.value}; "
        f"{outcome.summary}"
    )


def _clean_required(value: str) -> str:
    cleaned = value.strip()

    if not cleaned:
        raise ValueError("field cannot be empty.")

    return cleaned