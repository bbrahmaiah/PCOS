from __future__ import annotations

from enum import StrEnum
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.environment.environment_fusion import FusedContext
from jarvis.environment.models import EnvironmentSource, TrustCalibration
from jarvis.environment.trust_runtime import TrustPolicyClassification
from jarvis.environment.ui_reasoning import (
    PlannerHintKind,
    UIReasoningIntentKind,
    UIReasoningResult,
)
from jarvis.environment.workspace_graph import GraphNodeKind
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class IntentLifecycleState(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    BLOCKED = "blocked"
    WAITING_APPROVAL = "waiting_approval"
    PARTIAL = "partial"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class GoalPriority(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class IntentPersistenceReason(StrEnum):
    SESSION_CREATED = "session_created"
    INTENT_CREATED = "intent_created"
    INTENT_UPDATED = "intent_updated"
    SUBGOAL_ADDED = "subgoal_added"
    SUBGOAL_ACTIVATED = "subgoal_activated"
    INTENT_PAUSED = "intent_paused"
    INTENT_RESUMED = "intent_resumed"
    INTENT_BLOCKED = "intent_blocked"
    APPROVAL_REQUIRED = "approval_required"
    PARTIAL_PROGRESS_RECORDED = "partial_progress_recorded"
    VERIFIED_STATE_RECORDED = "verified_state_recorded"
    INTENT_COMPLETED = "intent_completed"
    INTENT_CANCELLED = "intent_cancelled"
    INTENT_FAILED = "intent_failed"
    RESUME_TOKEN_CREATED = "resume_token_created"
    SESSION_NOT_FOUND = "session_not_found"
    INTENT_NOT_FOUND = "intent_not_found"
    RUNTIME_RESET = "runtime_reset"


class IntentPersistenceEventKind(StrEnum):
    SESSION_CREATED = "session_created"
    INTENT_MUTATED = "intent_mutated"
    RESUME_TOKEN_CREATED = "resume_token_created"
    RUNTIME_RESET = "runtime_reset"


class ResumeStrategy(StrEnum):
    CONTINUE_ACTIVE_SUBGOAL = "continue_active_subgoal"
    RESTORE_PAUSED_WORKFLOW = "restore_paused_workflow"
    VERIFY_THEN_CONTINUE = "verify_then_continue"
    ASK_USER_BEFORE_RESUME = "ask_user_before_resume"
    DO_NOT_RESUME = "do_not_resume"


class GoalState(OrchestrationModel):
    """
    Persistent top-level user goal.

    This is the thing JARVIS must not lose across interruptions.
    """

    goal_id: str = Field(default_factory=lambda: f"goal_{uuid4().hex}")
    description: str
    priority: GoalPriority = GoalPriority.NORMAL
    status: IntentLifecycleState = IntentLifecycleState.ACTIVE
    created_from: str | None = None
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("goal_id", "description")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class SubgoalState(OrchestrationModel):
    """
    Persistent subgoal.

    Examples:
    - inspect error
    - research browser docs
    - run tests
    - verify target
    """

    subgoal_id: str = Field(default_factory=lambda: f"subgoal_{uuid4().hex}")
    description: str
    intent_kind: UIReasoningIntentKind | None = None
    planner_hint: PlannerHintKind | None = None
    status: IntentLifecycleState = IntentLifecycleState.ACTIVE
    order: int = Field(default=0, ge=0)
    confidence: float = Field(default=0.80, ge=0.0, le=1.0)
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("subgoal_id", "description")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class BlockedState(OrchestrationModel):
    """
    Why the intent cannot continue right now.
    """

    blocked_id: str = Field(default_factory=lambda: f"blocked_{uuid4().hex}")
    reason: str
    blocked_by: str | None = None
    recoverable: bool = True
    requires_user_input: bool = False
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("blocked_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class PartialCompletionState(OrchestrationModel):
    """
    Partial progress made toward the intent.
    """

    partial_id: str = Field(default_factory=lambda: f"partial_{uuid4().hex}")
    summary: str
    completed_steps: tuple[str, ...] = ()
    remaining_steps: tuple[str, ...] = ()
    progress_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: tuple[str, ...] = ()
    recorded_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("partial_id", "summary")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class PausedWorkflowState(OrchestrationModel):
    """
    Paused workflow state.

    Used when user interrupts, switches task, or asks JARVIS to wait.
    """

    pause_id: str = Field(default_factory=lambda: f"paused_{uuid4().hex}")
    paused_subgoal_id: str | None = None
    reason: str
    resume_plan: tuple[str, ...] = ()
    paused_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("pause_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class LastVerifiedState(OrchestrationModel):
    """
    Last verified environment/workflow state.

    This is important because resume must verify reality before continuing.
    """

    verified_id: str = Field(default_factory=lambda: f"verified_{uuid4().hex}")
    summary: str
    graph_node_count: int = Field(default=0, ge=0)
    visible_error_count: int = Field(default=0, ge=0)
    policy: TrustPolicyClassification = TrustPolicyClassification.REVIEW
    verified_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("verified_id", "summary")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class IntentResumeToken(OrchestrationModel):
    """
    Token used to resume an interrupted or paused intent.

    This token is not permission to act. It is only permission to reconstruct
    context and propose continuation.
    """

    token_id: str = Field(default_factory=lambda: f"resume_{uuid4().hex}")
    intent_id: str
    goal_id: str
    active_subgoal_id: str | None = None
    strategy: ResumeStrategy
    resume_prompt: str
    requires_verification: bool = True
    expires_after_turns: int = Field(default=10, ge=1, le=100)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("token_id", "intent_id", "goal_id", "resume_prompt")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class PersistentIntentState(OrchestrationModel):
    """
    Full persistent intent state.

    This is the main object Step 20 preserves.
    """

    intent_id: str = Field(default_factory=lambda: f"intent_{uuid4().hex}")
    goal: GoalState
    subgoals: tuple[SubgoalState, ...] = ()
    active_subgoal_id: str | None = None
    blocked: BlockedState | None = None
    partial: PartialCompletionState | None = None
    paused: PausedWorkflowState | None = None
    last_verified: LastVerifiedState | None = None
    resume_token: IntentResumeToken | None = None
    status: IntentLifecycleState = IntentLifecycleState.ACTIVE
    policy: TrustPolicyClassification = TrustPolicyClassification.REVIEW
    trust: TrustCalibration
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("intent_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _active_subgoal_must_exist(self) -> PersistentIntentState:
        if self.active_subgoal_id is None:
            return self

        ids = {subgoal.subgoal_id for subgoal in self.subgoals}

        if self.active_subgoal_id not in ids:
            raise ValueError("active_subgoal_id must reference an existing subgoal.")

        return self


class IntentPersistenceSession(OrchestrationModel):
    """
    Intent persistence runtime session.
    """

    session_id: str = Field(default_factory=lambda: f"intent_session_{uuid4().hex}")
    workspace_id: str
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class IntentPersistenceRuntimeEvent(OrchestrationModel):
    """
    Intent persistence runtime event.
    """

    event_id: str = Field(default_factory=lambda: f"intent_event_{uuid4().hex}")
    kind: IntentPersistenceEventKind
    reason: IntentPersistenceReason
    session_id: str | None = None
    intent_id: str | None = None
    token_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class IntentPersistenceRuntimeSnapshot(OrchestrationModel):
    """
    Diagnostics for Phase 8 Step 20.
    """

    name: str
    session_count: int = Field(ge=0)
    intent_count: int = Field(ge=0)
    active_count: int = Field(ge=0)
    paused_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    partial_count: int = Field(ge=0)
    completed_count: int = Field(ge=0)
    resume_token_count: int = Field(ge=0)
    runtime_event_count: int = Field(ge=0)
    last_reason: IntentPersistenceReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        return _clean_required(value)


class IntentStateBuilder:
    """
    Builds PersistentIntentState from fused context and reasoning output.
    """

    def build_from_fused_context(
        self,
        *,
        fused_context: FusedContext,
        description: str | None = None,
    ) -> PersistentIntentState:
        goal_text = description or _goal_from_context(fused_context)
        goal = GoalState(
            description=goal_text,
            created_from=fused_context.enrichment.original_text,
            confidence=fused_context.trust.confidence,
        )
        subgoals = _subgoals_from_context(fused_context)
        active_subgoal_id = subgoals[0].subgoal_id if subgoals else None

        return PersistentIntentState(
            goal=goal,
            subgoals=subgoals,
            active_subgoal_id=active_subgoal_id,
            status=IntentLifecycleState.ACTIVE,
            policy=fused_context.policy,
            trust=fused_context.trust,
            last_verified=_verified_from_context(fused_context),
            metadata={
                "fused_context_id": fused_context.context_id,
                "fusion_status": fused_context.status.value,
            },
        )

    def build_from_reasoning(
        self,
        *,
        reasoning: UIReasoningResult,
        goal_description: str | None = None,
    ) -> PersistentIntentState:
        goal = GoalState(
            description=goal_description or reasoning.message,
            created_from=reasoning.intent.raw_text,
            confidence=reasoning.intent.confidence,
        )
        subgoals = tuple(
            SubgoalState(
                description=hint.description,
                intent_kind=reasoning.intent.kind,
                planner_hint=hint.kind,
                status=IntentLifecycleState.ACTIVE,
                order=index,
                confidence=hint.confidence,
                metadata={"requires_verification": hint.requires_verification},
            )
            for index, hint in enumerate(reasoning.planner_hints)
        )
        active_subgoal_id = subgoals[0].subgoal_id if subgoals else None
        trust = TrustCalibration(
            confidence=reasoning.intent.confidence,
            stability=0.82,
            ambiguity=1.0 - reasoning.intent.confidence,
            source=_environment_source(),
            reason="intent built from UI reasoning",
        )

        return PersistentIntentState(
            goal=goal,
            subgoals=subgoals,
            active_subgoal_id=active_subgoal_id,
            status=IntentLifecycleState.ACTIVE,
            policy=TrustPolicyClassification.REVIEW,
            trust=trust,
            metadata={"reasoning_result_id": reasoning.result_id},
        )


class IntentPersistenceRuntime:
    """
    Phase 8 Step 20 Intent Persistence Runtime.

    Responsibilities:
    - persist active goal and subgoals
    - preserve blocked/paused/partial state
    - create resume tokens
    - record last verified state
    - survive interruptions and context switches

    Non-responsibilities:
    - no action execution
    - no direct memory writes outside memory gateway
    - no LLM calls
    - no environment capture
    """

    def __init__(
        self,
        *,
        name: str = "intent_persistence_runtime",
        builder: IntentStateBuilder | None = None,
    ) -> None:
        cleaned = name.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._builder = builder or IntentStateBuilder()
        self._sessions: dict[str, IntentPersistenceSession] = {}
        self._intents: dict[str, dict[str, PersistentIntentState]] = {}
        self._events: list[IntentPersistenceRuntimeEvent] = []
        self._lock = RLock()
        self._last_reason: IntentPersistenceReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def create_session(
        self,
        *,
        workspace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> IntentPersistenceSession:
        session = IntentPersistenceSession(
            workspace_id=workspace_id,
            metadata=metadata or {},
        )
        event = self._event(
            kind=IntentPersistenceEventKind.SESSION_CREATED,
            reason=IntentPersistenceReason.SESSION_CREATED,
            session_id=session.session_id,
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._intents[session.session_id] = {}
            self._events.append(event)
            self._last_reason = event.reason

        return session

    def create_intent(
        self,
        *,
        session_id: str,
        goal_description: str,
        priority: GoalPriority = GoalPriority.NORMAL,
        subgoals: tuple[SubgoalState, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> PersistentIntentState:
        self._session_or_raise(session_id)
        goal = GoalState(description=goal_description, priority=priority)
        active_subgoal_id = subgoals[0].subgoal_id if subgoals else None
        state = PersistentIntentState(
            goal=goal,
            subgoals=subgoals,
            active_subgoal_id=active_subgoal_id,
            trust=TrustCalibration(
                confidence=goal.confidence,
                stability=0.84,
                ambiguity=1.0 - goal.confidence,
                source=_environment_source(),
                reason="intent created",
            ),
            metadata=metadata or {},
        )

        self._store(
            session_id=session_id,
            state=state,
            reason=IntentPersistenceReason.INTENT_CREATED,
        )

        return state

    def create_from_fused_context(
        self,
        *,
        session_id: str,
        fused_context: FusedContext,
        description: str | None = None,
    ) -> PersistentIntentState:
        self._session_or_raise(session_id)
        state = self._builder.build_from_fused_context(
            fused_context=fused_context,
            description=description,
        )

        self._store(
            session_id=session_id,
            state=state,
            reason=IntentPersistenceReason.INTENT_CREATED,
        )

        return state

    def create_from_reasoning(
        self,
        *,
        session_id: str,
        reasoning: UIReasoningResult,
        goal_description: str | None = None,
    ) -> PersistentIntentState:
        self._session_or_raise(session_id)
        state = self._builder.build_from_reasoning(
            reasoning=reasoning,
            goal_description=goal_description,
        )

        self._store(
            session_id=session_id,
            state=state,
            reason=IntentPersistenceReason.INTENT_CREATED,
        )

        return state

    def add_subgoal(
        self,
        *,
        session_id: str,
        intent_id: str,
        subgoal: SubgoalState,
        activate: bool = False,
    ) -> PersistentIntentState:
        state = self._intent_or_raise(session_id, intent_id)
        subgoals = (*state.subgoals, subgoal)
        active_subgoal_id = subgoal.subgoal_id if activate else state.active_subgoal_id
        updated = state.model_copy(
            update={
                "subgoals": subgoals,
                "active_subgoal_id": active_subgoal_id,
                "updated_at": utc_now(),
            }
        )

        self._store(
            session_id=session_id,
            state=updated,
            reason=IntentPersistenceReason.SUBGOAL_ADDED,
        )

        return updated

    def activate_subgoal(
        self,
        *,
        session_id: str,
        intent_id: str,
        subgoal_id: str,
    ) -> PersistentIntentState:
        state = self._intent_or_raise(session_id, intent_id)

        if subgoal_id not in {subgoal.subgoal_id for subgoal in state.subgoals}:
            raise ValueError(f"subgoal not found: {subgoal_id}")

        updated = state.model_copy(
            update={
                "active_subgoal_id": subgoal_id,
                "status": IntentLifecycleState.ACTIVE,
                "updated_at": utc_now(),
            }
        )

        self._store(
            session_id=session_id,
            state=updated,
            reason=IntentPersistenceReason.SUBGOAL_ACTIVATED,
        )

        return updated

    def record_partial_progress(
        self,
        *,
        session_id: str,
        intent_id: str,
        partial: PartialCompletionState,
    ) -> PersistentIntentState:
        state = self._intent_or_raise(session_id, intent_id)
        updated = state.model_copy(
            update={
                "partial": partial,
                "status": IntentLifecycleState.PARTIAL,
                "updated_at": utc_now(),
            }
        )

        self._store(
            session_id=session_id,
            state=updated,
            reason=IntentPersistenceReason.PARTIAL_PROGRESS_RECORDED,
        )

        return updated

    def block_intent(
        self,
        *,
        session_id: str,
        intent_id: str,
        blocked: BlockedState,
    ) -> PersistentIntentState:
        state = self._intent_or_raise(session_id, intent_id)
        updated = state.model_copy(
            update={
                "blocked": blocked,
                "status": IntentLifecycleState.BLOCKED,
                "updated_at": utc_now(),
            }
        )

        self._store(
            session_id=session_id,
            state=updated,
            reason=IntentPersistenceReason.INTENT_BLOCKED,
        )

        return updated

    def require_approval(
        self,
        *,
        session_id: str,
        intent_id: str,
        reason: str,
    ) -> PersistentIntentState:
        blocked = BlockedState(
            reason=reason,
            blocked_by="approval",
            recoverable=True,
            requires_user_input=True,
        )
        state = self._intent_or_raise(session_id, intent_id)
        updated = state.model_copy(
            update={
                "blocked": blocked,
                "status": IntentLifecycleState.WAITING_APPROVAL,
                "updated_at": utc_now(),
            }
        )

        self._store(
            session_id=session_id,
            state=updated,
            reason=IntentPersistenceReason.APPROVAL_REQUIRED,
        )

        return updated

    def pause_intent(
        self,
        *,
        session_id: str,
        intent_id: str,
        pause: PausedWorkflowState,
    ) -> PersistentIntentState:
        state = self._intent_or_raise(session_id, intent_id)
        token = _resume_token_for(
            state=state,
            strategy=ResumeStrategy.RESTORE_PAUSED_WORKFLOW,
            reason=pause.reason,
        )
        updated = state.model_copy(
            update={
                "paused": pause,
                "resume_token": token,
                "status": IntentLifecycleState.PAUSED,
                "updated_at": utc_now(),
            }
        )

        self._store(
            session_id=session_id,
            state=updated,
            reason=IntentPersistenceReason.INTENT_PAUSED,
            token_id=token.token_id,
        )

        return updated

    def resume_intent(
        self,
        *,
        session_id: str,
        intent_id: str,
        token_id: str,
    ) -> PersistentIntentState:
        state = self._intent_or_raise(session_id, intent_id)

        if state.resume_token is None or state.resume_token.token_id != token_id:
            raise ValueError("invalid resume token.")

        updated = state.model_copy(
            update={
                "paused": None,
                "blocked": None,
                "status": IntentLifecycleState.ACTIVE,
                "updated_at": utc_now(),
            }
        )

        self._store(
            session_id=session_id,
            state=updated,
            reason=IntentPersistenceReason.INTENT_RESUMED,
            token_id=token_id,
        )

        return updated

    def record_verified_state(
        self,
        *,
        session_id: str,
        intent_id: str,
        verified: LastVerifiedState,
    ) -> PersistentIntentState:
        state = self._intent_or_raise(session_id, intent_id)
        updated = state.model_copy(
            update={
                "last_verified": verified,
                "updated_at": utc_now(),
            }
        )

        self._store(
            session_id=session_id,
            state=updated,
            reason=IntentPersistenceReason.VERIFIED_STATE_RECORDED,
        )

        return updated

    def complete_intent(
        self,
        *,
        session_id: str,
        intent_id: str,
        summary: str,
    ) -> PersistentIntentState:
        state = self._intent_or_raise(session_id, intent_id)
        final_partial = PartialCompletionState(
            summary=summary,
            progress_ratio=1.0,
            completed_steps=(summary,),
        )
        updated_goal = state.goal.model_copy(
            update={
                "status": IntentLifecycleState.COMPLETED,
                "updated_at": utc_now(),
            }
        )
        updated = state.model_copy(
            update={
                "goal": updated_goal,
                "partial": final_partial,
                "status": IntentLifecycleState.COMPLETED,
                "updated_at": utc_now(),
            }
        )

        self._store(
            session_id=session_id,
            state=updated,
            reason=IntentPersistenceReason.INTENT_COMPLETED,
        )

        return updated

    def cancel_intent(
        self,
        *,
        session_id: str,
        intent_id: str,
        reason: str,
    ) -> PersistentIntentState:
        state = self._intent_or_raise(session_id, intent_id)
        updated_goal = state.goal.model_copy(
            update={
                "status": IntentLifecycleState.CANCELLED,
                "updated_at": utc_now(),
            }
        )
        updated = state.model_copy(
            update={
                "goal": updated_goal,
                "status": IntentLifecycleState.CANCELLED,
                "blocked": BlockedState(
                    reason=reason,
                    blocked_by="cancelled",
                    recoverable=False,
                ),
                "updated_at": utc_now(),
            }
        )

        self._store(
            session_id=session_id,
            state=updated,
            reason=IntentPersistenceReason.INTENT_CANCELLED,
        )

        return updated

    def fail_intent(
        self,
        *,
        session_id: str,
        intent_id: str,
        reason: str,
        recoverable: bool = True,
    ) -> PersistentIntentState:
        state = self._intent_or_raise(session_id, intent_id)
        updated = state.model_copy(
            update={
                "status": IntentLifecycleState.FAILED,
                "blocked": BlockedState(
                    reason=reason,
                    blocked_by="failure",
                    recoverable=recoverable,
                ),
                "updated_at": utc_now(),
            }
        )

        self._store(
            session_id=session_id,
            state=updated,
            reason=IntentPersistenceReason.INTENT_FAILED,
        )

        return updated

    def create_resume_token(
        self,
        *,
        session_id: str,
        intent_id: str,
        strategy: ResumeStrategy = ResumeStrategy.VERIFY_THEN_CONTINUE,
    ) -> IntentResumeToken:
        state = self._intent_or_raise(session_id, intent_id)
        token = _resume_token_for(
            state=state,
            strategy=strategy,
            reason="manual resume token request",
        )
        updated = state.model_copy(
            update={
                "resume_token": token,
                "updated_at": utc_now(),
            }
        )

        self._store(
            session_id=session_id,
            state=updated,
            reason=IntentPersistenceReason.RESUME_TOKEN_CREATED,
            token_id=token.token_id,
        )

        return token

    def active_intent(
        self,
        *,
        session_id: str,
    ) -> PersistentIntentState | None:
        with self._lock:
            intents = self._intents.get(session_id, {})

            active = [
                intent
                for intent in intents.values()
                if intent.status
                in {
                    IntentLifecycleState.ACTIVE,
                    IntentLifecycleState.PARTIAL,
                    IntentLifecycleState.BLOCKED,
                    IntentLifecycleState.WAITING_APPROVAL,
                    IntentLifecycleState.PAUSED,
                }
            ]

            if not active:
                return None

            return sorted(
                active,
                key=lambda item: str(item.updated_at),
                reverse=True,
            )[0]

    def intent_for(
        self,
        *,
        session_id: str,
        intent_id: str,
    ) -> PersistentIntentState | None:
        with self._lock:
            return self._intents.get(session_id, {}).get(intent_id)

    def intents_for(
        self,
        *,
        session_id: str,
    ) -> tuple[PersistentIntentState, ...]:
        with self._lock:
            return tuple(self._intents.get(session_id, {}).values())

    def events(self) -> tuple[IntentPersistenceRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def snapshot(self) -> IntentPersistenceRuntimeSnapshot:
        with self._lock:
            intents = [
                intent
                for session_intents in self._intents.values()
                for intent in session_intents.values()
            ]

            return IntentPersistenceRuntimeSnapshot(
                name=self.name,
                session_count=len(self._sessions),
                intent_count=len(intents),
                active_count=sum(
                    1
                    for intent in intents
                    if intent.status == IntentLifecycleState.ACTIVE
                ),
                paused_count=sum(
                    1
                    for intent in intents
                    if intent.status == IntentLifecycleState.PAUSED
                ),
                blocked_count=sum(
                    1
                    for intent in intents
                    if intent.status
                    in {
                        IntentLifecycleState.BLOCKED,
                        IntentLifecycleState.WAITING_APPROVAL,
                    }
                ),
                partial_count=sum(
                    1
                    for intent in intents
                    if intent.status == IntentLifecycleState.PARTIAL
                ),
                completed_count=sum(
                    1
                    for intent in intents
                    if intent.status == IntentLifecycleState.COMPLETED
                ),
                resume_token_count=sum(
                    1 for intent in intents if intent.resume_token is not None
                ),
                runtime_event_count=len(self._events),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        event = self._event(
            kind=IntentPersistenceEventKind.RUNTIME_RESET,
            reason=IntentPersistenceReason.RUNTIME_RESET,
        )

        with self._lock:
            self._sessions.clear()
            self._intents.clear()
            self._events.clear()
            self._events.append(event)
            self._last_reason = event.reason

    def _session_or_raise(self, session_id: str) -> IntentPersistenceSession:
        with self._lock:
            session = self._sessions.get(session_id)

        if session is None:
            raise ValueError(f"intent persistence session not found: {session_id}")

        return session

    def _intent_or_raise(
        self,
        session_id: str,
        intent_id: str,
    ) -> PersistentIntentState:
        with self._lock:
            state = self._intents.get(session_id, {}).get(intent_id)

        if state is None:
            raise ValueError(f"persistent intent not found: {intent_id}")

        return state

    def _store(
        self,
        *,
        session_id: str,
        state: PersistentIntentState,
        reason: IntentPersistenceReason,
        token_id: str | None = None,
    ) -> None:
        event = self._event(
            kind=IntentPersistenceEventKind.INTENT_MUTATED,
            reason=reason,
            session_id=session_id,
            intent_id=state.intent_id,
            token_id=token_id,
        )

        with self._lock:
            self._intents.setdefault(session_id, {})[state.intent_id] = state
            self._events.append(event)
            self._last_reason = reason
            self._touch_session(session_id)

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
        kind: IntentPersistenceEventKind,
        reason: IntentPersistenceReason,
        session_id: str | None = None,
        intent_id: str | None = None,
        token_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IntentPersistenceRuntimeEvent:
        return IntentPersistenceRuntimeEvent(
            kind=kind,
            reason=reason,
            session_id=session_id,
            intent_id=intent_id,
            token_id=token_id,
            metadata=metadata or {},
        )


def _goal_from_context(context: FusedContext) -> str:
    if context.stream.reasoning_result is not None:
        return context.stream.reasoning_result.message

    if context.stream.active_intent is not None:
        if context.stream.active_intent.goal:
            return context.stream.active_intent.goal

    return context.enrichment.enriched_text


def _subgoals_from_context(context: FusedContext) -> tuple[SubgoalState, ...]:
    if context.stream.reasoning_result is None:
        return ()

    return tuple(
        SubgoalState(
            description=hint.description,
            intent_kind=context.stream.reasoning_result.intent.kind,
            planner_hint=hint.kind,
            order=index,
            confidence=hint.confidence,
            metadata={
                "requires_verification": hint.requires_verification,
                "allowed_to_plan": hint.allowed_to_plan,
            },
        )
        for index, hint in enumerate(context.stream.reasoning_result.planner_hints)
    )


def _verified_from_context(context: FusedContext) -> LastVerifiedState | None:
    graph = context.stream.workspace_graph

    if graph is None:
        return None

    return LastVerifiedState(
        summary=context.bridge.fused_summary,
        graph_node_count=len(graph.nodes),
        visible_error_count=sum(
            1 for node in graph.nodes.values() if node.kind == GraphNodeKind.ERROR
        ),
        policy=context.policy,
    )


def _resume_token_for(
    *,
    state: PersistentIntentState,
    strategy: ResumeStrategy,
    reason: str,
) -> IntentResumeToken:
    active = _active_subgoal(state)
    resume_parts = [f"Resume goal: {state.goal.description}"]

    if active is not None:
        resume_parts.append(f"Continue subgoal: {active.description}")

    if state.blocked is not None:
        resume_parts.append(f"Previously blocked: {state.blocked.reason}")

    if state.partial is not None:
        resume_parts.append(f"Progress: {state.partial.summary}")

    return IntentResumeToken(
        intent_id=state.intent_id,
        goal_id=state.goal.goal_id,
        active_subgoal_id=state.active_subgoal_id,
        strategy=strategy,
        resume_prompt=f"{' | '.join(resume_parts)} | reason: {reason}",
        requires_verification=True,
    )


def _active_subgoal(state: PersistentIntentState) -> SubgoalState | None:
    if state.active_subgoal_id is None:
        return None

    for subgoal in state.subgoals:
        if subgoal.subgoal_id == state.active_subgoal_id:
            return subgoal

    return None


def _environment_source() -> EnvironmentSource:
    return EnvironmentSource.OS_OBSERVER


def _clean_required(value: str) -> str:
    cleaned = value.strip()

    if not cleaned:
        raise ValueError("field cannot be empty.")

    return cleaned