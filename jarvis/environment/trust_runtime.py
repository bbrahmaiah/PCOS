from __future__ import annotations

from enum import StrEnum
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator

from jarvis.environment.models import (
    EnvironmentSource,
    EnvironmentTrustLevel,
    PrivacyClassification,
    TrustCalibration,
)
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class TrustSignalKind(StrEnum):
    """
    Type of trust signal entering the calibration runtime.
    """

    CONFIDENCE = "confidence"
    STABILITY = "stability"
    AMBIGUITY = "ambiguity"
    SOURCE_RELIABILITY = "source_reliability"
    OBSERVATION = "observation"
    ACTION = "action"
    VERIFICATION = "verification"


class TrustSubjectKind(StrEnum):
    """
    Subject being calibrated.
    """

    OCR_TEXT = "ocr_text"
    UI_ELEMENT = "ui_element"
    SCREEN_REGION = "screen_region"
    WINDOW_STATE = "window_state"
    APP_STATE = "app_state"
    WORKSPACE_GRAPH = "workspace_graph"
    GROUNDING_TARGET = "grounding_target"
    SIMULATION = "simulation"
    INTERACTION = "interaction"
    VERIFICATION = "verification"
    RECOVERY = "recovery"
    ENVIRONMENT_MEMORY = "environment_memory"


class TrustPolicyClassification(StrEnum):
    """
    Policy classification derived from trust.

    This does not execute policy. It gives later systems a clear decision hint.
    """

    SAFE = "safe"
    REVIEW = "review"
    ASK_USER = "ask_user"
    VERIFY_FIRST = "verify_first"
    BLOCKED = "blocked"


class TrustDecisionKind(StrEnum):
    """
    Final trust decision.
    """

    ACCEPT = "accept"
    ACCEPT_WITH_VERIFICATION = "accept_with_verification"
    ASK_FOR_CLARIFICATION = "ask_for_clarification"
    DEFER = "defer"
    BLOCK = "block"


class TrustRuntimeReason(StrEnum):
    """
    Machine-readable runtime reason.
    """

    SIGNAL_RECORDED = "signal_recorded"
    OBSERVATION_CALIBRATED = "observation_calibrated"
    ACTION_CALIBRATED = "action_calibrated"
    VERIFICATION_CALIBRATED = "verification_calibrated"
    TRUST_DECISION_BUILT = "trust_decision_built"
    SOURCE_RELIABILITY_UPDATED = "source_reliability_updated"
    LOW_CONFIDENCE_BLOCKED = "low_confidence_blocked"
    HIGH_AMBIGUITY_REQUIRES_CLARIFICATION = (
        "high_ambiguity_requires_clarification"
    )
    RUNTIME_RESET = "runtime_reset"


class TrustRuntimeEventKind(StrEnum):
    """
    Trust runtime event kind.
    """

    SIGNAL_RECORDED = "signal_recorded"
    TRUST_CALIBRATED = "trust_calibrated"
    DECISION_BUILT = "decision_built"
    SOURCE_RELIABILITY_UPDATED = "source_reliability_updated"
    RUNTIME_RESET = "runtime_reset"


class ConfidenceSignal(OrchestrationModel):
    """
    Confidence signal emitted by a Phase 8 subsystem.
    """

    signal_id: str = Field(default_factory=lambda: f"trustsignal_{uuid4().hex}")
    subject_id: str
    subject_kind: TrustSubjectKind
    source: EnvironmentSource
    value: float = Field(ge=0.0, le=1.0)
    reason: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("signal_id", "subject_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class StabilityScore(OrchestrationModel):
    """
    Stability score.

    Stable means the observed object has not been flickering, moving,
    contradicting itself, or changing too rapidly.
    """

    subject_id: str
    subject_kind: TrustSubjectKind
    value: float = Field(ge=0.0, le=1.0)
    sample_count: int = Field(default=1, ge=1)
    reason: str

    @field_validator("subject_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class AmbiguityScore(OrchestrationModel):
    """
    Ambiguity score.

    Higher ambiguity means multiple targets, uncertain text, spoofable UI,
    or conflicting observations.
    """

    subject_id: str
    subject_kind: TrustSubjectKind
    value: float = Field(ge=0.0, le=1.0)
    candidate_count: int = Field(default=1, ge=1)
    reason: str

    @field_validator("subject_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class SourceReliability(OrchestrationModel):
    """
    Reliability profile for one source.

    Accessibility is generally more reliable than OCR.
    Verification is stronger than simulation.
    """

    source: EnvironmentSource
    reliability: float = Field(ge=0.0, le=1.0)
    sample_count: int = Field(default=1, ge=1)
    last_verified_at: object = Field(default_factory=utc_now)
    reason: str

    @field_validator("reason")
    @classmethod
    def _required_reason(cls, value: str) -> str:
        return _clean_required(value)


class ObservationTrust(OrchestrationModel):
    """
    Calibrated trust for perception/observation results.
    """

    trust_id: str = Field(default_factory=lambda: f"obstrust_{uuid4().hex}")
    subject_id: str
    subject_kind: TrustSubjectKind
    source: EnvironmentSource
    calibration: TrustCalibration
    policy_classification: TrustPolicyClassification
    last_verified_at: object | None = None
    reason: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("trust_id", "subject_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class ActionTrust(OrchestrationModel):
    """
    Calibrated trust for an action candidate.

    This gates physical/environment interaction later.
    """

    trust_id: str = Field(default_factory=lambda: f"actiontrust_{uuid4().hex}")
    subject_id: str
    subject_kind: TrustSubjectKind
    source: EnvironmentSource
    calibration: TrustCalibration
    policy_classification: TrustPolicyClassification
    decision: TrustDecisionKind
    requires_verification: bool
    reason: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("trust_id", "subject_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class VerificationTrust(OrchestrationModel):
    """
    Calibrated trust for verification results.
    """

    trust_id: str = Field(default_factory=lambda: f"verifytrust_{uuid4().hex}")
    subject_id: str
    subject_kind: TrustSubjectKind
    source: EnvironmentSource
    calibration: TrustCalibration
    policy_classification: TrustPolicyClassification
    verified: bool
    reason: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("trust_id", "subject_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class TrustDecision(OrchestrationModel):
    """
    Final decision produced from calibrated trust.
    """

    decision_id: str = Field(default_factory=lambda: f"trustdecision_{uuid4().hex}")
    subject_id: str
    subject_kind: TrustSubjectKind
    decision: TrustDecisionKind
    policy_classification: TrustPolicyClassification
    calibration: TrustCalibration
    reason: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("decision_id", "subject_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class TrustRuntimeEvent(OrchestrationModel):
    """
    Trust calibration runtime event.
    """

    event_id: str = Field(default_factory=lambda: f"trustevent_{uuid4().hex}")
    kind: TrustRuntimeEventKind
    reason: TrustRuntimeReason
    subject_id: str | None = None
    subject_kind: TrustSubjectKind | None = None
    trust_level: EnvironmentTrustLevel | None = None
    policy_classification: TrustPolicyClassification | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class TrustRuntimeSnapshot(OrchestrationModel):
    """
    Diagnostics for Phase 8 Step 3.
    """

    name: str
    signal_count: int = Field(ge=0)
    observation_count: int = Field(ge=0)
    action_count: int = Field(ge=0)
    verification_count: int = Field(ge=0)
    decision_count: int = Field(ge=0)
    source_count: int = Field(ge=0)
    event_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    ask_user_count: int = Field(ge=0)
    verify_first_count: int = Field(ge=0)
    last_reason: TrustRuntimeReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        return _clean_required(value)


class TrustCalibrationRuntime:
    """
    Phase 8 Step 3 Trust Calibration Runtime.

    Responsibilities:
    - convert confidence/stability/ambiguity/source reliability into trust
    - produce observation trust, action trust, and verification trust
    - emit policy classification
    - never treat perception as certainty
    - make every Phase 8 subsystem probabilistic and auditable

    Non-responsibilities:
    - no OCR
    - no visual detection
    - no interaction execution
    - no verification execution
    """

    def __init__(self, *, name: str = "trust_calibration_runtime") -> None:
        cleaned = name.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._source_reliability: dict[EnvironmentSource, SourceReliability] = (
            self._default_source_reliability()
        )
        self._signals: list[ConfidenceSignal] = []
        self._observations: list[ObservationTrust] = []
        self._actions: list[ActionTrust] = []
        self._verifications: list[VerificationTrust] = []
        self._decisions: list[TrustDecision] = []
        self._events: list[TrustRuntimeEvent] = []
        self._lock = RLock()
        self._last_reason: TrustRuntimeReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def record_signal(self, signal: ConfidenceSignal) -> TrustRuntimeEvent:
        event = TrustRuntimeEvent(
            kind=TrustRuntimeEventKind.SIGNAL_RECORDED,
            reason=TrustRuntimeReason.SIGNAL_RECORDED,
            subject_id=signal.subject_id,
            subject_kind=signal.subject_kind,
            metadata={
                "source": signal.source.value,
                "value": signal.value,
                "reason": signal.reason,
            },
        )

        with self._lock:
            self._signals.append(signal)
            self._events.append(event)
            self._last_reason = event.reason

        return event

    def update_source_reliability(
        self,
        reliability: SourceReliability,
    ) -> TrustRuntimeEvent:
        event = TrustRuntimeEvent(
            kind=TrustRuntimeEventKind.SOURCE_RELIABILITY_UPDATED,
            reason=TrustRuntimeReason.SOURCE_RELIABILITY_UPDATED,
            metadata={
                "source": reliability.source.value,
                "reliability": reliability.reliability,
            },
        )

        with self._lock:
            self._source_reliability[reliability.source] = reliability
            self._events.append(event)
            self._last_reason = event.reason

        return event

    def calibrate_observation(
        self,
        *,
        subject_id: str,
        subject_kind: TrustSubjectKind,
        source: EnvironmentSource,
        confidence: float,
        stability: float,
        ambiguity: float,
        privacy: PrivacyClassification = PrivacyClassification.WORKSPACE,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> ObservationTrust:
        calibration = self._calibration(
            source=source,
            confidence=confidence,
            stability=stability,
            ambiguity=ambiguity,
            reason=reason,
            metadata=metadata,
        )
        policy = self._policy_for(
            calibration=calibration,
            privacy=privacy,
            action=False,
            verified=False,
        )
        observation = ObservationTrust(
            subject_id=subject_id,
            subject_kind=subject_kind,
            source=source,
            calibration=calibration,
            policy_classification=policy,
            last_verified_at=None,
            reason=reason,
            metadata=metadata or {},
        )

        self._append_calibration_event(
            subject_id=subject_id,
            subject_kind=subject_kind,
            calibration=calibration,
            policy=policy,
            reason=TrustRuntimeReason.OBSERVATION_CALIBRATED,
        )

        with self._lock:
            self._observations.append(observation)

        return observation

    def calibrate_action(
        self,
        *,
        subject_id: str,
        subject_kind: TrustSubjectKind,
        source: EnvironmentSource,
        confidence: float,
        stability: float,
        ambiguity: float,
        privacy: PrivacyClassification = PrivacyClassification.WORKSPACE,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> ActionTrust:
        calibration = self._calibration(
            source=source,
            confidence=confidence,
            stability=stability,
            ambiguity=ambiguity,
            reason=reason,
            metadata=metadata,
        )
        policy = self._policy_for(
            calibration=calibration,
            privacy=privacy,
            action=True,
            verified=False,
        )
        decision = self._decision_for(policy)
        action = ActionTrust(
            subject_id=subject_id,
            subject_kind=subject_kind,
            source=source,
            calibration=calibration,
            policy_classification=policy,
            decision=decision,
            requires_verification=policy
            in {
                TrustPolicyClassification.SAFE,
                TrustPolicyClassification.REVIEW,
                TrustPolicyClassification.VERIFY_FIRST,
            },
            reason=reason,
            metadata=metadata or {},
        )

        self._append_calibration_event(
            subject_id=subject_id,
            subject_kind=subject_kind,
            calibration=calibration,
            policy=policy,
            reason=TrustRuntimeReason.ACTION_CALIBRATED,
        )

        with self._lock:
            self._actions.append(action)

        return action

    def calibrate_verification(
        self,
        *,
        subject_id: str,
        subject_kind: TrustSubjectKind,
        source: EnvironmentSource,
        confidence: float,
        stability: float,
        ambiguity: float,
        verified: bool,
        privacy: PrivacyClassification = PrivacyClassification.WORKSPACE,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> VerificationTrust:
        calibration = self._calibration(
            source=source,
            confidence=confidence,
            stability=stability,
            ambiguity=ambiguity,
            reason=reason,
            metadata=metadata,
        )
        policy = self._policy_for(
            calibration=calibration,
            privacy=privacy,
            action=False,
            verified=verified,
        )
        verification = VerificationTrust(
            subject_id=subject_id,
            subject_kind=subject_kind,
            source=source,
            calibration=calibration,
            policy_classification=policy,
            verified=verified,
            reason=reason,
            metadata=metadata or {},
        )

        self._append_calibration_event(
            subject_id=subject_id,
            subject_kind=subject_kind,
            calibration=calibration,
            policy=policy,
            reason=TrustRuntimeReason.VERIFICATION_CALIBRATED,
        )

        with self._lock:
            self._verifications.append(verification)

        return verification

    def build_decision(
        self,
        *,
        subject_id: str,
        subject_kind: TrustSubjectKind,
        calibration: TrustCalibration,
        privacy: PrivacyClassification = PrivacyClassification.WORKSPACE,
        action: bool = False,
        verified: bool = False,
        reason: str,
    ) -> TrustDecision:
        policy = self._policy_for(
            calibration=calibration,
            privacy=privacy,
            action=action,
            verified=verified,
        )
        decision_kind = self._decision_for(policy)
        decision = TrustDecision(
            subject_id=subject_id,
            subject_kind=subject_kind,
            decision=decision_kind,
            policy_classification=policy,
            calibration=calibration,
            reason=reason,
        )
        event = TrustRuntimeEvent(
            kind=TrustRuntimeEventKind.DECISION_BUILT,
            reason=TrustRuntimeReason.TRUST_DECISION_BUILT,
            subject_id=subject_id,
            subject_kind=subject_kind,
            trust_level=calibration.level,
            policy_classification=policy,
        )

        with self._lock:
            self._decisions.append(decision)
            self._events.append(event)
            self._last_reason = event.reason

        return decision

    def source_reliability_for(
        self,
        source: EnvironmentSource,
    ) -> SourceReliability:
        with self._lock:
            return self._source_reliability[source]

    def signals(self) -> tuple[ConfidenceSignal, ...]:
        with self._lock:
            return tuple(self._signals)

    def observations(self) -> tuple[ObservationTrust, ...]:
        with self._lock:
            return tuple(self._observations)

    def actions(self) -> tuple[ActionTrust, ...]:
        with self._lock:
            return tuple(self._actions)

    def verifications(self) -> tuple[VerificationTrust, ...]:
        with self._lock:
            return tuple(self._verifications)

    def decisions(self) -> tuple[TrustDecision, ...]:
        with self._lock:
            return tuple(self._decisions)

    def events(self) -> tuple[TrustRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def snapshot(self) -> TrustRuntimeSnapshot:
        with self._lock:
            policies: list[TrustPolicyClassification] = []

            policies.extend(
                item.policy_classification for item in self._observations
            )
            policies.extend(
                item.policy_classification for item in self._actions
            )
            policies.extend(
                item.policy_classification for item in self._verifications
            )
            policies.extend(
                item.policy_classification for item in self._decisions
            )

            return TrustRuntimeSnapshot(
                name=self.name,
                signal_count=len(self._signals),
                observation_count=len(self._observations),
                action_count=len(self._actions),
                verification_count=len(self._verifications),
                decision_count=len(self._decisions),
                source_count=len(self._source_reliability),
                event_count=len(self._events),
                blocked_count=sum(
                    1 
                    for policy in policies 
                    if policy == TrustPolicyClassification.BLOCKED
                ),
                ask_user_count=sum(
                    1 
                    for policy in policies 
                    if policy == TrustPolicyClassification.ASK_USER
                ),
                verify_first_count=sum(
                    1
                    for policy in policies
                    if policy == TrustPolicyClassification.VERIFY_FIRST
                ),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        event = TrustRuntimeEvent(
            kind=TrustRuntimeEventKind.RUNTIME_RESET,
            reason=TrustRuntimeReason.RUNTIME_RESET,
        )

        with self._lock:
            self._signals.clear()
            self._observations.clear()
            self._actions.clear()
            self._verifications.clear()
            self._decisions.clear()
            self._events.clear()
            self._events.append(event)
            self._last_reason = TrustRuntimeReason.RUNTIME_RESET

    def _calibration(
        self,
        *,
        source: EnvironmentSource,
        confidence: float,
        stability: float,
        ambiguity: float,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> TrustCalibration:
        source_reliability = self.source_reliability_for(source)
        adjusted_confidence = confidence * source_reliability.reliability

        return TrustCalibration(
            confidence=adjusted_confidence,
            stability=stability,
            ambiguity=ambiguity,
            source=source,
            reason=reason,
            metadata=metadata or {},
        )

    def _policy_for(
        self,
        *,
        calibration: TrustCalibration,
        privacy: PrivacyClassification,
        action: bool,
        verified: bool,
    ) -> TrustPolicyClassification:
        score = calibration.effective_score()

        if privacy in {PrivacyClassification.BLOCKED, PrivacyClassification.SECRET}:
            return TrustPolicyClassification.BLOCKED

        if calibration.ambiguity >= 0.65:
            return TrustPolicyClassification.ASK_USER

        if action and score < 0.75:
            return TrustPolicyClassification.BLOCKED

        if score < 0.40:
            return TrustPolicyClassification.BLOCKED

        if score < 0.60:
            return TrustPolicyClassification.ASK_USER

        if action and not verified:
            return TrustPolicyClassification.VERIFY_FIRST

        if score < 0.80:
            return TrustPolicyClassification.REVIEW

        return TrustPolicyClassification.SAFE

    @staticmethod
    def _decision_for(
        policy: TrustPolicyClassification,
    ) -> TrustDecisionKind:
        if policy == TrustPolicyClassification.SAFE:
            return TrustDecisionKind.ACCEPT

        if policy in {
            TrustPolicyClassification.REVIEW,
            TrustPolicyClassification.VERIFY_FIRST,
        }:
            return TrustDecisionKind.ACCEPT_WITH_VERIFICATION

        if policy == TrustPolicyClassification.ASK_USER:
            return TrustDecisionKind.ASK_FOR_CLARIFICATION

        return TrustDecisionKind.BLOCK

    def _append_calibration_event(
        self,
        *,
        subject_id: str,
        subject_kind: TrustSubjectKind,
        calibration: TrustCalibration,
        policy: TrustPolicyClassification,
        reason: TrustRuntimeReason,
    ) -> None:
        event = TrustRuntimeEvent(
            kind=TrustRuntimeEventKind.TRUST_CALIBRATED,
            reason=reason,
            subject_id=subject_id,
            subject_kind=subject_kind,
            trust_level=calibration.level,
            policy_classification=policy,
        )

        with self._lock:
            self._events.append(event)
            self._last_reason = event.reason

    @staticmethod
    def _default_source_reliability() -> dict[EnvironmentSource, SourceReliability]:
        return {
            EnvironmentSource.ACCESSIBILITY: SourceReliability(
                source=EnvironmentSource.ACCESSIBILITY,
                reliability=0.98,
                reason="accessibility API is structured and high confidence",
            ),
            EnvironmentSource.SCREEN_CAPTURE: SourceReliability(
                source=EnvironmentSource.SCREEN_CAPTURE,
                reliability=0.85,
                reason="screen capture is direct but needs interpretation",
            ),
            EnvironmentSource.OCR: SourceReliability(
                source=EnvironmentSource.OCR,
                reliability=0.78,
                reason="OCR is useful but error-prone",
            ),
            EnvironmentSource.VISUAL_DETECTION: SourceReliability(
                source=EnvironmentSource.VISUAL_DETECTION,
                reliability=0.72,
                reason="visual detection is probabilistic",
            ),
            EnvironmentSource.OS_OBSERVER: SourceReliability(
                source=EnvironmentSource.OS_OBSERVER,
                reliability=0.90,
                reason="OS events are structured but may miss app internals",
            ),
            EnvironmentSource.APP_PROFILE: SourceReliability(
                source=EnvironmentSource.APP_PROFILE,
                reliability=0.88,
                reason="app profile is stable but app versions change",
            ),
            EnvironmentSource.USER_INPUT: SourceReliability(
                source=EnvironmentSource.USER_INPUT,
                reliability=0.95,
                reason="user input is direct intent evidence",
            ),
            EnvironmentSource.MEMORY: SourceReliability(
                source=EnvironmentSource.MEMORY,
                reliability=0.80,
                reason="memory is useful but may be stale",
            ),
            EnvironmentSource.SIMULATION: SourceReliability(
                source=EnvironmentSource.SIMULATION,
                reliability=0.70,
                reason="simulation predicts, it does not observe",
            ),
            EnvironmentSource.VERIFICATION: SourceReliability(
                source=EnvironmentSource.VERIFICATION,
                reliability=0.96,
                reason="verification compares expected and observed state",
            ),
        }


def _clean_required(value: str) -> str:
    cleaned = value.strip()

    if not cleaned:
        raise ValueError("field cannot be empty.")

    return cleaned