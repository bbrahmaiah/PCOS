from __future__ import annotations

from enum import StrEnum
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.environment.models import EnvironmentSource, TrustCalibration
from jarvis.environment.trust_runtime import TrustPolicyClassification
from jarvis.environment.workflow_cognition import WorkflowKind
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class AssistanceObservationKind(StrEnum):
    BUILD_ERROR = "build_error"
    LONG_RUNNING_TASK = "long_running_task"
    REPEATED_ACTION = "repeated_action"
    APP_CRASH = "app_crash"
    USER_RETURN = "user_return"
    WORKFLOW_IDLE = "workflow_idle"


class AssistancePreparationKind(StrEnum):
    ERROR_SUMMARY = "error_summary"
    LOG_CONTEXT = "log_context"
    TEST_SUMMARY = "test_summary"
    RESUME_CONTEXT = "resume_context"
    REPEATED_ACTION_HINT = "repeated_action_hint"
    CRASH_CONTEXT = "crash_context"


class AssistanceSuggestionKind(StrEnum):
    DEBUG_ERROR = "debug_error"
    CHECK_LONG_TASK = "check_long_task"
    AUTOMATE_REPEATED_ACTION = "automate_repeated_action"
    RESTORE_CRASHED_APP = "restore_crashed_app"
    RESUME_WORKFLOW = "resume_workflow"
    ASK_IF_HELP_NEEDED = "ask_if_help_needed"


class AssistanceStatus(StrEnum):
    OBSERVED = "observed"
    PREPARED = "prepared"
    SUGGESTED = "suggested"
    SUPPRESSED = "suppressed"
    BLOCKED = "blocked"
    FAILED = "failed"


class AssistanceDecision(StrEnum):
    OBSERVE = "observe"
    PREPARE = "prepare"
    SUGGEST = "suggest"
    SUPPRESS = "suppress"
    BLOCK_ACTION = "block_action"
    FAIL = "fail"


class AssistanceReason(StrEnum):
    SESSION_CREATED = "session_created"
    BUILD_ERROR_OBSERVED = "build_error_observed"
    LONG_RUNNING_TASK_OBSERVED = "long_running_task_observed"
    REPEATED_ACTION_OBSERVED = "repeated_action_observed"
    APP_CRASH_OBSERVED = "app_crash_observed"
    USER_RETURN_OBSERVED = "user_return_observed"
    PREPARATION_CREATED = "preparation_created"
    SUGGESTION_CREATED = "suggestion_created"
    LOW_CONFIDENCE_SUPPRESSED = "low_confidence_suppressed"
    PROACTIVE_ACTION_BLOCKED = "proactive_action_blocked"
    SESSION_NOT_FOUND = "session_not_found"
    RUNTIME_RESET = "runtime_reset"


class AssistanceEventKind(StrEnum):
    SESSION_CREATED = "session_created"
    OBSERVATION_RECORDED = "observation_recorded"
    PREPARATION_RECORDED = "preparation_recorded"
    SUGGESTION_RECORDED = "suggestion_recorded"
    SUGGESTION_SUPPRESSED = "suggestion_suppressed"
    PROACTIVE_ACTION_BLOCKED = "proactive_action_blocked"
    RUNTIME_RESET = "runtime_reset"


class ProactivePermission(StrEnum):
    OBSERVATION = "observation"
    PREPARATION = "preparation"
    SUGGESTION = "suggestion"
    ACTION = "action"


class AssistanceConfidenceBand(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AssistanceObservation(OrchestrationModel):
    observation_id: str = Field(
        default_factory=lambda: f"assist_obs_{uuid4().hex}"
    )
    kind: AssistanceObservationKind
    workspace_id: str
    summary: str
    workflow: WorkflowKind | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    source: EnvironmentSource = EnvironmentSource.OS_OBSERVER
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("observation_id", "workspace_id", "summary")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class AssistancePreparation(OrchestrationModel):
    preparation_id: str = Field(
        default_factory=lambda: f"assist_prep_{uuid4().hex}"
    )
    kind: AssistancePreparationKind
    observation: AssistanceObservation
    prepared_summary: str
    safe_to_show_user: bool = True
    contains_action: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("preparation_id", "prepared_summary")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _preparation_cannot_execute(self) -> AssistancePreparation:
        if self.contains_action:
            raise ValueError("proactive preparation cannot contain actions.")

        return self


class AssistanceSuggestion(OrchestrationModel):
    suggestion_id: str = Field(default_factory=lambda: f"assist_sugg_{uuid4().hex}")
    kind: AssistanceSuggestionKind
    observation: AssistanceObservation
    preparation: AssistancePreparation | None = None
    message: str
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_band: AssistanceConfidenceBand
    user_visible: bool = True
    requires_user_confirmation: bool = True
    action_requested: bool = False
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("suggestion_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _suggestion_not_action(self) -> AssistanceSuggestion:
        if self.action_requested:
            raise ValueError("proactive suggestion cannot request action execution.")

        if not self.user_visible:
            raise ValueError("proactive suggestion must be user-visible.")

        return self


class WorkflowSuggestionPolicy(OrchestrationModel):
    """
    Step 38 policy.

    Observation, preparation, and suggestions are allowed.
    Proactive action is blocked unless a later policy layer explicitly approves it.
    """

    allow_proactive_observation: bool = True
    allow_proactive_preparation: bool = True
    allow_proactive_suggestion: bool = True
    allow_proactive_action: bool = False
    min_suggestion_confidence: float = Field(default=0.70, ge=0.0, le=1.0)
    max_suggestions_per_session: int = Field(default=5, ge=1, le=100)


class AssistancePolicyDecision(OrchestrationModel):
    decision_id: str = Field(
        default_factory=lambda: f"assist_policy_{uuid4().hex}"
    )
    permission: ProactivePermission
    allowed: bool
    reason: AssistanceReason
    confidence: float = Field(ge=0.0, le=1.0)
    created_at: object = Field(default_factory=utc_now)

    @field_validator("decision_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class ContinuousAssistanceResult(OrchestrationModel):
    result_id: str = Field(default_factory=lambda: f"assist_result_{uuid4().hex}")
    status: AssistanceStatus
    decision: AssistanceDecision
    reason: AssistanceReason
    observation: AssistanceObservation | None = None
    preparation: AssistancePreparation | None = None
    suggestion: AssistanceSuggestion | None = None
    policy_decision: AssistancePolicyDecision | None = None
    trust: TrustCalibration
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("result_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _payload_matches_status(self) -> ContinuousAssistanceResult:
        if self.status == AssistanceStatus.OBSERVED and self.observation is None:
            raise ValueError("OBSERVED result requires observation.")

        if self.status == AssistanceStatus.PREPARED and self.preparation is None:
            raise ValueError("PREPARED result requires preparation.")

        if self.status == AssistanceStatus.SUGGESTED and self.suggestion is None:
            raise ValueError("SUGGESTED result requires suggestion.")

        return self


class ContinuousAssistanceSession(OrchestrationModel):
    session_id: str = Field(default_factory=lambda: f"assist_session_{uuid4().hex}")
    workspace_id: str
    observation_count: int = Field(default=0, ge=0)
    preparation_count: int = Field(default=0, ge=0)
    suggestion_count: int = Field(default=0, ge=0)
    suppressed_count: int = Field(default=0, ge=0)
    blocked_action_count: int = Field(default=0, ge=0)
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class ContinuousAssistanceRuntimeEvent(OrchestrationModel):
    event_id: str = Field(default_factory=lambda: f"assist_event_{uuid4().hex}")
    kind: AssistanceEventKind
    reason: AssistanceReason
    session_id: str | None = None
    result_id: str | None = None
    observation_id: str | None = None
    suggestion_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class ContinuousAssistanceRuntimeSnapshot(OrchestrationModel):
    name: str
    session_count: int = Field(ge=0)
    result_count: int = Field(ge=0)
    observation_count: int = Field(ge=0)
    preparation_count: int = Field(ge=0)
    suggestion_count: int = Field(ge=0)
    suppressed_count: int = Field(ge=0)
    blocked_action_count: int = Field(ge=0)
    runtime_event_count: int = Field(ge=0)
    last_reason: AssistanceReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        return _clean_required(value)


class BuildErrorWatcher:
    def observe(
        self,
        *,
        workspace_id: str,
        build_output: str,
    ) -> AssistanceObservation | None:
        lowered = build_output.lower()
        if not any(token in lowered for token in ("error", "failed", "traceback")):
            return None

        return AssistanceObservation(
            kind=AssistanceObservationKind.BUILD_ERROR,
            workspace_id=workspace_id,
            summary=_short(build_output),
            workflow=WorkflowKind.DEBUGGING,
            confidence=0.90,
            metadata={"watcher": "build_error"},
        )


class LongRunningTaskWatcher:
    def observe(
        self,
        *,
        workspace_id: str,
        task_name: str,
        elapsed_seconds: int,
        threshold_seconds: int = 300,
    ) -> AssistanceObservation | None:
        if elapsed_seconds < threshold_seconds:
            return None

        return AssistanceObservation(
            kind=AssistanceObservationKind.LONG_RUNNING_TASK,
            workspace_id=workspace_id,
            summary=f"{task_name} has been running for {elapsed_seconds}s",
            workflow=WorkflowKind.TESTING,
            confidence=0.78,
            metadata={
                "watcher": "long_running_task",
                "elapsed_seconds": elapsed_seconds,
            },
        )


class RepeatedActionDetector:
    def observe(
        self,
        *,
        workspace_id: str,
        action_names: tuple[str, ...],
    ) -> AssistanceObservation | None:
        if len(action_names) < 3:
            return None

        repeated = _most_common(action_names)
        if repeated is None:
            return None

        action, count = repeated
        if count < 3:
            return None

        return AssistanceObservation(
            kind=AssistanceObservationKind.REPEATED_ACTION,
            workspace_id=workspace_id,
            summary=f"Repeated action detected: {action} x{count}",
            workflow=WorkflowKind.CODING,
            confidence=0.82,
            metadata={
                "watcher": "repeated_action",
                "action": action,
                "count": count,
            },
        )


class AppCrashWatcher:
    def observe(
        self,
        *,
        workspace_id: str,
        app_name: str,
        crashed: bool,
    ) -> AssistanceObservation | None:
        if not crashed:
            return None

        return AssistanceObservation(
            kind=AssistanceObservationKind.APP_CRASH,
            workspace_id=workspace_id,
            summary=f"{app_name} appears to have crashed",
            workflow=WorkflowKind.DEBUGGING,
            confidence=0.86,
            metadata={"watcher": "app_crash", "app_name": app_name},
        )


class UserReturnDetector:
    def observe(
        self,
        *,
        workspace_id: str,
        away_seconds: int,
        active_workflow: WorkflowKind | None = None,
    ) -> AssistanceObservation | None:
        if away_seconds < 60:
            return None

        return AssistanceObservation(
            kind=AssistanceObservationKind.USER_RETURN,
            workspace_id=workspace_id,
            summary=f"User returned after {away_seconds}s away",
            workflow=active_workflow,
            confidence=0.74,
            metadata={"watcher": "user_return", "away_seconds": away_seconds},
        )


class ContinuousPreparationBuilder:
    def prepare(self, observation: AssistanceObservation) -> AssistancePreparation:
        if observation.kind == AssistanceObservationKind.BUILD_ERROR:
            return AssistancePreparation(
                kind=AssistancePreparationKind.ERROR_SUMMARY,
                observation=observation,
                prepared_summary=f"Prepared error summary: {observation.summary}",
                confidence=observation.confidence,
            )

        if observation.kind == AssistanceObservationKind.LONG_RUNNING_TASK:
            return AssistancePreparation(
                kind=AssistancePreparationKind.TEST_SUMMARY,
                observation=observation,
                prepared_summary=f"Prepared task status: {observation.summary}",
                confidence=observation.confidence,
            )

        if observation.kind == AssistanceObservationKind.REPEATED_ACTION:
            return AssistancePreparation(
                kind=AssistancePreparationKind.REPEATED_ACTION_HINT,
                observation=observation,
                prepared_summary=(
                    "Prepared repeated-action hint: "
                     f"{observation.summary}"
                ),
                confidence=observation.confidence,
            )

        if observation.kind == AssistanceObservationKind.APP_CRASH:
            return AssistancePreparation(
                kind=AssistancePreparationKind.CRASH_CONTEXT,
                observation=observation,
                prepared_summary=f"Prepared crash context: {observation.summary}",
                confidence=observation.confidence,
            )

        return AssistancePreparation(
            kind=AssistancePreparationKind.RESUME_CONTEXT,
            observation=observation,
            prepared_summary=f"Prepared resume context: {observation.summary}",
            confidence=observation.confidence,
        )


class ContinuousSuggestionBuilder:
    def suggest(
        self,
        *,
        observation: AssistanceObservation,
        preparation: AssistancePreparation | None,
    ) -> AssistanceSuggestion:
        kind = AssistanceSuggestionKind.ASK_IF_HELP_NEEDED
        message = "proactive_observation_available"

        if observation.kind == AssistanceObservationKind.BUILD_ERROR:
            kind = AssistanceSuggestionKind.DEBUG_ERROR
            message = "build_error_assistance_available"

        elif observation.kind == AssistanceObservationKind.LONG_RUNNING_TASK:
            kind = AssistanceSuggestionKind.CHECK_LONG_TASK
            message = "long_running_task_status_available"

        elif observation.kind == AssistanceObservationKind.REPEATED_ACTION:
            kind = AssistanceSuggestionKind.AUTOMATE_REPEATED_ACTION
            message = (
                "I noticed a repeated action. I can prepare a safer shortcut "
                "if you want."
            )

        elif observation.kind == AssistanceObservationKind.APP_CRASH:
            kind = AssistanceSuggestionKind.RESTORE_CRASHED_APP
            message = "app_crash_recovery_available"

        elif observation.kind == AssistanceObservationKind.USER_RETURN:
            kind = AssistanceSuggestionKind.RESUME_WORKFLOW
            message = "resume_summary_available"

        return AssistanceSuggestion(
            kind=kind,
            observation=observation,
            preparation=preparation,
            message=message,
            confidence=observation.confidence,
            confidence_band=_confidence_band(observation.confidence),
            user_visible=True,
            requires_user_confirmation=True,
            action_requested=False,
        )


class ContinuousAssistanceModeRuntime:
    """
    Phase 8 Step 38 Continuous Assistance Mode.

    Allowed:
    - proactive observation
    - proactive preparation
    - proactive suggestion with confidence

    Not allowed:
    - proactive action execution
    - hidden computer control
    - acting without policy/user approval
    """

    def __init__(
        self,
        *,
        name: str = "continuous_assistance_mode_runtime",
        policy: WorkflowSuggestionPolicy | None = None,
        preparation_builder: ContinuousPreparationBuilder | None = None,
        suggestion_builder: ContinuousSuggestionBuilder | None = None,
    ) -> None:
        cleaned = name.strip()
        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._policy = policy or WorkflowSuggestionPolicy()
        self._preparation_builder = (
            preparation_builder or ContinuousPreparationBuilder()
        )
        self._suggestion_builder = suggestion_builder or ContinuousSuggestionBuilder()
        self._sessions: dict[str, ContinuousAssistanceSession] = {}
        self._results: list[ContinuousAssistanceResult] = []
        self._events: list[ContinuousAssistanceRuntimeEvent] = []
        self._lock = RLock()
        self._last_reason: AssistanceReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def create_session(
        self,
        *,
        workspace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> ContinuousAssistanceSession:
        session = ContinuousAssistanceSession(
            workspace_id=workspace_id,
            metadata=metadata or {},
        )
        event = self._event(
            kind=AssistanceEventKind.SESSION_CREATED,
            reason=AssistanceReason.SESSION_CREATED,
            session_id=session.session_id,
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._events.append(event)
            self._last_reason = event.reason

        return session

    def observe(
        self,
        *,
        session_id: str,
        observation: AssistanceObservation,
    ) -> ContinuousAssistanceResult:
        if self.session_for(session_id) is None:
            result = _blocked_result(
                status=AssistanceStatus.FAILED,
                decision=AssistanceDecision.FAIL,
                reason=AssistanceReason.SESSION_NOT_FOUND,
                message="continuous assistance session not found",
            )
            self._record_result(result, session_id)
            return result

        policy = self._policy_decision(
            permission=ProactivePermission.OBSERVATION,
            confidence=observation.confidence,
        )
        if not policy.allowed:
            result = _blocked_result(
                status=AssistanceStatus.BLOCKED,
                decision=AssistanceDecision.BLOCK_ACTION,
                reason=policy.reason,
                message="proactive observation blocked by policy",
                policy_decision=policy,
            )
            self._record_result(result, session_id)
            return result

        result = ContinuousAssistanceResult(
            status=AssistanceStatus.OBSERVED,
            decision=AssistanceDecision.OBSERVE,
            reason=_reason_for_observation(observation.kind),
            observation=observation,
            policy_decision=policy,
            trust=_trust(
                confidence=observation.confidence,
                reason="proactive observation recorded",
            ),
            message=observation.summary,
        )
        self._record_result(result, session_id)
        return result

    def prepare(
        self,
        *,
        session_id: str,
        observation: AssistanceObservation,
    ) -> ContinuousAssistanceResult:
        if self.session_for(session_id) is None:
            result = _blocked_result(
                status=AssistanceStatus.FAILED,
                decision=AssistanceDecision.FAIL,
                reason=AssistanceReason.SESSION_NOT_FOUND,
                message="continuous assistance session not found",
            )
            self._record_result(result, session_id)
            return result

        policy = self._policy_decision(
            permission=ProactivePermission.PREPARATION,
            confidence=observation.confidence,
        )
        if not policy.allowed:
            result = _blocked_result(
                status=AssistanceStatus.BLOCKED,
                decision=AssistanceDecision.BLOCK_ACTION,
                reason=policy.reason,
                message="proactive preparation blocked by policy",
                policy_decision=policy,
            )
            self._record_result(result, session_id)
            return result

        preparation = self._preparation_builder.prepare(observation)
        result = ContinuousAssistanceResult(
            status=AssistanceStatus.PREPARED,
            decision=AssistanceDecision.PREPARE,
            reason=AssistanceReason.PREPARATION_CREATED,
            observation=observation,
            preparation=preparation,
            policy_decision=policy,
            trust=_trust(
                confidence=preparation.confidence,
                reason="proactive preparation recorded",
            ),
            message=preparation.prepared_summary,
        )
        self._record_result(result, session_id)
        return result

    def suggest(
        self,
        *,
        session_id: str,
        observation: AssistanceObservation,
        preparation: AssistancePreparation | None = None,
    ) -> ContinuousAssistanceResult:
        if self.session_for(session_id) is None:
            result = _blocked_result(
                status=AssistanceStatus.FAILED,
                decision=AssistanceDecision.FAIL,
                reason=AssistanceReason.SESSION_NOT_FOUND,
                message="continuous assistance session not found",
            )
            self._record_result(result, session_id)
            return result

        policy = self._policy_decision(
            permission=ProactivePermission.SUGGESTION,
            confidence=observation.confidence,
        )

        if not policy.allowed:
            result = ContinuousAssistanceResult(
                status=AssistanceStatus.SUPPRESSED,
                decision=AssistanceDecision.SUPPRESS,
                reason=policy.reason,
                observation=observation,
                preparation=preparation,
                policy_decision=policy,
                trust=_trust(
                    confidence=observation.confidence,
                    reason="proactive suggestion suppressed",
                ),
                message="suggestion suppressed by confidence or policy",
            )
            self._record_result(result, session_id)
            return result

        if observation.confidence < self._policy.min_suggestion_confidence:
            result = ContinuousAssistanceResult(
                status=AssistanceStatus.SUPPRESSED,
                decision=AssistanceDecision.SUPPRESS,
                reason=AssistanceReason.LOW_CONFIDENCE_SUPPRESSED,
                observation=observation,
                preparation=preparation,
                policy_decision=policy,
                trust=_trust(
                    confidence=observation.confidence,
                    reason="low-confidence suggestion suppressed",
                ),
                message="suggestion suppressed due to low confidence",
            )
            self._record_result(result, session_id)
            return result

        suggestion = self._suggestion_builder.suggest(
            observation=observation,
            preparation=preparation,
        )
        result = ContinuousAssistanceResult(
            status=AssistanceStatus.SUGGESTED,
            decision=AssistanceDecision.SUGGEST,
            reason=AssistanceReason.SUGGESTION_CREATED,
            observation=observation,
            preparation=preparation,
            suggestion=suggestion,
            policy_decision=policy,
            trust=_trust(
                confidence=suggestion.confidence,
                reason="proactive suggestion created",
            ),
            message=suggestion.message,
        )
        self._record_result(result, session_id)
        return result

    def request_proactive_action(
        self,
        *,
        session_id: str,
        description: str,
    ) -> ContinuousAssistanceResult:
        if self.session_for(session_id) is None:
            result = _blocked_result(
                status=AssistanceStatus.FAILED,
                decision=AssistanceDecision.FAIL,
                reason=AssistanceReason.SESSION_NOT_FOUND,
                message="continuous assistance session not found",
            )
            self._record_result(result, session_id)
            return result

        policy = self._policy_decision(
            permission=ProactivePermission.ACTION,
            confidence=1.0,
        )

        result = _blocked_result(
            status=AssistanceStatus.BLOCKED,
            decision=AssistanceDecision.BLOCK_ACTION,
            reason=AssistanceReason.PROACTIVE_ACTION_BLOCKED,
            message=f"proactive action blocked: {description}",
            policy_decision=policy,
        )
        self._record_result(result, session_id)
        return result

    def session_for(
        self,
        session_id: str,
    ) -> ContinuousAssistanceSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def results(self) -> tuple[ContinuousAssistanceResult, ...]:
        with self._lock:
            return tuple(self._results)

    def events(self) -> tuple[ContinuousAssistanceRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def snapshot(self) -> ContinuousAssistanceRuntimeSnapshot:
        with self._lock:
            return ContinuousAssistanceRuntimeSnapshot(
                name=self.name,
                session_count=len(self._sessions),
                result_count=len(self._results),
                observation_count=sum(
                    1
                    for result in self._results
                    if result.status == AssistanceStatus.OBSERVED
                ),
                preparation_count=sum(
                    1
                    for result in self._results
                    if result.status == AssistanceStatus.PREPARED
                ),
                suggestion_count=sum(
                    1
                    for result in self._results
                    if result.status == AssistanceStatus.SUGGESTED
                ),
                suppressed_count=sum(
                    1
                    for result in self._results
                    if result.status == AssistanceStatus.SUPPRESSED
                ),
                blocked_action_count=sum(
                    1
                    for result in self._results
                    if result.reason == AssistanceReason.PROACTIVE_ACTION_BLOCKED
                ),
                runtime_event_count=len(self._events),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        event = self._event(
            kind=AssistanceEventKind.RUNTIME_RESET,
            reason=AssistanceReason.RUNTIME_RESET,
        )

        with self._lock:
            self._sessions.clear()
            self._results.clear()
            self._events.clear()
            self._events.append(event)
            self._last_reason = event.reason

    def _policy_decision(
        self,
        *,
        permission: ProactivePermission,
        confidence: float,
    ) -> AssistancePolicyDecision:
        allowed = False
        reason = AssistanceReason.PROACTIVE_ACTION_BLOCKED

        if permission == ProactivePermission.OBSERVATION:
            allowed = self._policy.allow_proactive_observation
            reason = AssistanceReason.BUILD_ERROR_OBSERVED

        elif permission == ProactivePermission.PREPARATION:
            allowed = self._policy.allow_proactive_preparation
            reason = AssistanceReason.PREPARATION_CREATED

        elif permission == ProactivePermission.SUGGESTION:
            allowed = self._policy.allow_proactive_suggestion
            reason = (
                AssistanceReason.SUGGESTION_CREATED
                if confidence >= self._policy.min_suggestion_confidence
                else AssistanceReason.LOW_CONFIDENCE_SUPPRESSED
            )

        elif permission == ProactivePermission.ACTION:
            allowed = self._policy.allow_proactive_action
            reason = AssistanceReason.PROACTIVE_ACTION_BLOCKED

        return AssistancePolicyDecision(
            permission=permission,
            allowed=allowed,
            reason=reason,
            confidence=confidence,
        )

    def _record_result(
        self,
        result: ContinuousAssistanceResult,
        session_id: str,
    ) -> None:
        event = self._event(
            kind=_event_kind_for(result),
            reason=result.reason,
            session_id=session_id,
            result_id=result.result_id,
            observation_id=(
                result.observation.observation_id
                if result.observation is not None
                else None
            ),
            suggestion_id=(
                result.suggestion.suggestion_id
                if result.suggestion is not None
                else None
            ),
            metadata={"status": result.status.value},
        )

        with self._lock:
            self._results.append(result)
            self._events.append(event)
            self._last_reason = result.reason

            session = self._sessions.get(session_id)
            if session is not None:
                self._sessions[session_id] = session.model_copy(
                    update={
                        "updated_at": utc_now(),
                        "observation_count": session.observation_count
                        + (1 if result.status == AssistanceStatus.OBSERVED else 0),
                        "preparation_count": session.preparation_count
                        + (1 if result.status == AssistanceStatus.PREPARED else 0),
                        "suggestion_count": session.suggestion_count
                        + (1 if result.status == AssistanceStatus.SUGGESTED else 0),
                        "suppressed_count": session.suppressed_count
                        + (1 if result.status == AssistanceStatus.SUPPRESSED else 0),
                        "blocked_action_count": session.blocked_action_count
                        + (
                            1
                            if result.reason
                            == AssistanceReason.PROACTIVE_ACTION_BLOCKED
                            else 0
                        ),
                    }
                )

    @staticmethod
    def _event(
        *,
        kind: AssistanceEventKind,
        reason: AssistanceReason,
        session_id: str | None = None,
        result_id: str | None = None,
        observation_id: str | None = None,
        suggestion_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ContinuousAssistanceRuntimeEvent:
        return ContinuousAssistanceRuntimeEvent(
            kind=kind,
            reason=reason,
            session_id=session_id,
            result_id=result_id,
            observation_id=observation_id,
            suggestion_id=suggestion_id,
            metadata=metadata or {},
        )


def _reason_for_observation(kind: AssistanceObservationKind) -> AssistanceReason:
    mapping = {
        AssistanceObservationKind.BUILD_ERROR: AssistanceReason.BUILD_ERROR_OBSERVED,
        AssistanceObservationKind.LONG_RUNNING_TASK: (
            AssistanceReason.LONG_RUNNING_TASK_OBSERVED
        ),
        AssistanceObservationKind.REPEATED_ACTION: (
            AssistanceReason.REPEATED_ACTION_OBSERVED
        ),
        AssistanceObservationKind.APP_CRASH: AssistanceReason.APP_CRASH_OBSERVED,
        AssistanceObservationKind.USER_RETURN: AssistanceReason.USER_RETURN_OBSERVED,
    }
    return mapping.get(kind, AssistanceReason.BUILD_ERROR_OBSERVED)


def _event_kind_for(result: ContinuousAssistanceResult) -> AssistanceEventKind:
    if result.status == AssistanceStatus.OBSERVED:
        return AssistanceEventKind.OBSERVATION_RECORDED

    if result.status == AssistanceStatus.PREPARED:
        return AssistanceEventKind.PREPARATION_RECORDED

    if result.status == AssistanceStatus.SUGGESTED:
        return AssistanceEventKind.SUGGESTION_RECORDED

    if result.status == AssistanceStatus.SUPPRESSED:
        return AssistanceEventKind.SUGGESTION_SUPPRESSED

    if result.reason == AssistanceReason.PROACTIVE_ACTION_BLOCKED:
        return AssistanceEventKind.PROACTIVE_ACTION_BLOCKED

    return AssistanceEventKind.PROACTIVE_ACTION_BLOCKED


def _confidence_band(confidence: float) -> AssistanceConfidenceBand:
    if confidence >= 0.80:
        return AssistanceConfidenceBand.HIGH

    if confidence >= 0.50:
        return AssistanceConfidenceBand.MEDIUM

    return AssistanceConfidenceBand.LOW


def _most_common(values: tuple[str, ...]) -> tuple[str, int] | None:
    counts: dict[str, int] = {}
    for value in values:
        cleaned = value.strip()
        if not cleaned:
            continue
        counts[cleaned] = counts.get(cleaned, 0) + 1

    if not counts:
        return None

    return max(counts.items(), key=lambda item: item[1])


def _blocked_result(
    *,
    status: AssistanceStatus,
    decision: AssistanceDecision,
    reason: AssistanceReason,
    message: str,
    policy_decision: AssistancePolicyDecision | None = None,
) -> ContinuousAssistanceResult:
    return ContinuousAssistanceResult(
        status=status,
        decision=decision,
        reason=reason,
        policy_decision=policy_decision,
        trust=_trust(confidence=0.20, reason=message),
        message=message,
    )


def _trust(
    *,
    confidence: float,
    reason: str,
) -> TrustCalibration:
    return TrustCalibration(
        confidence=confidence,
        stability=max(0.0, min(1.0, confidence + 0.05)),
        ambiguity=1.0 - confidence,
        source=EnvironmentSource.OS_OBSERVER,
        reason=reason,
        metadata={"policy": TrustPolicyClassification.SAFE.value},
    )


def _short(value: str, *, limit: int = 220) -> str:
    cleaned = _clean_required(value)
    if len(cleaned) <= limit:
        return cleaned

    return cleaned[: limit - 3] + "..."


def _clean_required(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("field cannot be empty.")
    return cleaned