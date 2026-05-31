from __future__ import annotations

from enum import StrEnum
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.environment.environment_memory import (
    SessionContinuity,
    WorkflowStage,
    WorkspaceMemoryEntry,
)
from jarvis.environment.models import EnvironmentSource, TrustCalibration
from jarvis.environment.trust_runtime import TrustPolicyClassification
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class WorkflowKind(StrEnum):
    CODING = "coding"
    DEBUGGING = "debugging"
    RESEARCHING = "researching"
    WRITING = "writing"
    REVIEWING = "reviewing"
    TESTING = "testing"
    DEPLOYING = "deploying"
    UNKNOWN = "unknown"


class WorkflowCognitionStatus(StrEnum):
    UNDERSTOOD = "understood"
    RESUME_READY = "resume_ready"
    LOW_CONFIDENCE = "low_confidence"
    BLOCKED = "blocked"
    FAILED = "failed"


class WorkflowCognitionDecision(StrEnum):
    CONTINUE = "continue"
    RESUME = "resume"
    ASK_USER = "ask_user"
    BLOCK = "block"


class WorkflowCognitionReason(StrEnum):
    SESSION_CREATED = "session_created"
    WORKFLOW_UNDERSTOOD = "workflow_understood"
    WORKFLOW_LOW_CONFIDENCE = "workflow_low_confidence"
    RESUME_PLAN_CREATED = "resume_plan_created"
    CONTEXT_EMPTY_BLOCKED = "context_empty_blocked"
    SESSION_NOT_FOUND = "session_not_found"
    RUNTIME_RESET = "runtime_reset"


class WorkflowCognitionEventKind(StrEnum):
    SESSION_CREATED = "session_created"
    WORKFLOW_ANALYZED = "workflow_analyzed"
    RESUME_PLANNED = "resume_planned"
    OPERATION_BLOCKED = "operation_blocked"
    RUNTIME_RESET = "runtime_reset"


class WorkflowSignalKind(StrEnum):
    FILE_SIGNAL = "file_signal"
    COMMAND_SIGNAL = "command_signal"
    ERROR_SIGNAL = "error_signal"
    TODO_SIGNAL = "todo_signal"
    STAGE_SIGNAL = "stage_signal"
    INTENT_SIGNAL = "intent_signal"
    TERMINAL_SIGNAL = "terminal_signal"


class WorkflowConfidenceBand(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class WorkflowSignal(OrchestrationModel):
    signal_id: str = Field(default_factory=lambda: f"workflow_signal_{uuid4().hex}")
    kind: WorkflowSignalKind
    workflow: WorkflowKind
    weight: float = Field(ge=0.0, le=1.0)
    reason: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("signal_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class IntentionModel(OrchestrationModel):
    intention_id: str = Field(default_factory=lambda: f"intention_{uuid4().hex}")
    primary_workflow: WorkflowKind
    secondary_workflows: tuple[WorkflowKind, ...] = ()
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_band: WorkflowConfidenceBand
    user_intent: str | None = None
    inferred_goal: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("intention_id", "inferred_goal")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class TaskContinuity(OrchestrationModel):
    continuity_id: str = Field(default_factory=lambda: f"task_cont_{uuid4().hex}")
    continuity_token: str
    workflow_stage: WorkflowStage
    active_files: tuple[str, ...] = ()
    cursor_summary: str | None = None
    terminal_directory: str | None = None
    recent_commands: tuple[str, ...] = ()
    visible_errors: tuple[str, ...] = ()
    pending_todos: tuple[str, ...] = ()
    blocked: bool = False
    blocked_reason: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("continuity_id", "continuity_token")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class WorkflowContext(OrchestrationModel):
    context_id: str = Field(default_factory=lambda: f"workflow_ctx_{uuid4().hex}")
    workspace_id: str
    app_name: str
    project_path: str | None = None
    memory_entry: WorkspaceMemoryEntry
    task_continuity: TaskContinuity
    intention: IntentionModel
    signals: tuple[WorkflowSignal, ...]
    summary: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("context_id", "workspace_id", "app_name", "summary")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _requires_signals(self) -> WorkflowContext:
        if not self.signals:
            raise ValueError("workflow context requires signals.")

        return self


class WorkflowPrediction(OrchestrationModel):
    prediction_id: str = Field(
        default_factory=lambda: f"workflow_prediction_{uuid4().hex}"
    )
    next_workflow: WorkflowKind
    next_step: str
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    safe_to_prepare: bool = True
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("prediction_id", "next_step", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class WorkflowResumePlan(OrchestrationModel):
    resume_plan_id: str = Field(
        default_factory=lambda: f"workflow_resume_{uuid4().hex}"
    )
    workflow: WorkflowKind
    continuity_token: str
    resume_summary: str
    suggested_next_actions: tuple[str, ...]
    requires_user_confirmation: bool
    confidence: float = Field(ge=0.0, le=1.0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("resume_plan_id", "continuity_token", "resume_summary")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _requires_actions(self) -> WorkflowResumePlan:
        if not self.suggested_next_actions:
            raise ValueError("workflow resume plan requires suggested actions.")

        return self


class WorkflowCognitionResult(OrchestrationModel):
    result_id: str = Field(default_factory=lambda: f"workflow_result_{uuid4().hex}")
    status: WorkflowCognitionStatus
    decision: WorkflowCognitionDecision
    reason: WorkflowCognitionReason
    context: WorkflowContext | None = None
    prediction: WorkflowPrediction | None = None
    resume_plan: WorkflowResumePlan | None = None
    trust: TrustCalibration
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("result_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _resume_requires_plan(self) -> WorkflowCognitionResult:
        if self.decision == WorkflowCognitionDecision.RESUME:
            if self.resume_plan is None:
                raise ValueError("RESUME decision requires resume_plan.")

        if self.status == WorkflowCognitionStatus.UNDERSTOOD:
            if self.context is None:
                raise ValueError("UNDERSTOOD status requires context.")

        return self


class WorkflowCognitionSession(OrchestrationModel):
    session_id: str = Field(default_factory=lambda: f"workflow_session_{uuid4().hex}")
    workspace_id: str
    analyze_count: int = Field(default=0, ge=0)
    resume_count: int = Field(default=0, ge=0)
    blocked_count: int = Field(default=0, ge=0)
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class WorkflowCognitionRuntimeEvent(OrchestrationModel):
    event_id: str = Field(default_factory=lambda: f"workflow_event_{uuid4().hex}")
    kind: WorkflowCognitionEventKind
    reason: WorkflowCognitionReason
    session_id: str | None = None
    result_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class WorkflowCognitionRuntimeSnapshot(OrchestrationModel):
    name: str
    session_count: int = Field(ge=0)
    result_count: int = Field(ge=0)
    understood_count: int = Field(ge=0)
    resume_count: int = Field(ge=0)
    low_confidence_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    runtime_event_count: int = Field(ge=0)
    last_reason: WorkflowCognitionReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        return _clean_required(value)


class WorkflowPredictor:
    def predict(self, context: WorkflowContext) -> WorkflowPrediction:
        workflow = context.intention.primary_workflow

        if workflow == WorkflowKind.DEBUGGING:
            return WorkflowPrediction(
                next_workflow=WorkflowKind.TESTING,
                next_step="fix the visible error, then run targeted tests",
                confidence=0.84,
                reason="debugging usually proceeds into verification tests",
            )

        if workflow == WorkflowKind.CODING:
            return WorkflowPrediction(
                next_workflow=WorkflowKind.TESTING,
                next_step="finish the edit and run the relevant test file",
                confidence=0.78,
                reason="coding changes should be verified with tests",
            )

        if workflow == WorkflowKind.RESEARCHING:
            return WorkflowPrediction(
                next_workflow=WorkflowKind.REVIEWING,
                next_step="compare collected sources and extract conclusions",
                confidence=0.76,
                reason="research work usually needs source review",
            )

        if workflow == WorkflowKind.WRITING:
            return WorkflowPrediction(
                next_workflow=WorkflowKind.REVIEWING,
                next_step="review the draft for structure and clarity",
                confidence=0.74,
                reason="writing should be followed by review",
            )

        if workflow == WorkflowKind.TESTING:
            return WorkflowPrediction(
                next_workflow=WorkflowKind.DEBUGGING,
                next_step="inspect failures or mark the workflow complete",
                confidence=0.72,
                reason="test results determine debug or completion",
            )

        if workflow == WorkflowKind.DEPLOYING:
            return WorkflowPrediction(
                next_workflow=WorkflowKind.REVIEWING,
                next_step="verify deployment health and logs",
                confidence=0.80,
                reason="deployment requires post-deploy verification",
            )

        return WorkflowPrediction(
            next_workflow=WorkflowKind.UNKNOWN,
            next_step="ask the user what workflow to continue",
            confidence=0.30,
            reason="workflow is not clear enough to predict safely",
            safe_to_prepare=False,
        )


class WorkflowResumePlanner:
    def plan(
        self,
        *,
        context: WorkflowContext,
        continuity: SessionContinuity | None = None,
    ) -> WorkflowResumePlan:
        token = context.task_continuity.continuity_token
        if continuity is not None:
            token = continuity.continuity_token

        actions = _resume_actions_for(context)
        requires_confirmation = context.intention.confidence < 0.75

        return WorkflowResumePlan(
            workflow=context.intention.primary_workflow,
            continuity_token=token,
            resume_summary=context.summary,
            suggested_next_actions=actions,
            requires_user_confirmation=requires_confirmation,
            confidence=context.intention.confidence,
        )


class WorkflowCognitionRuntime:
    """
    Phase 8 Step 34 Workflow Cognition Runtime.

    This runtime understands the user's work, not just the visible screen.

    It consumes semantic workflow memory from Step 33 and infers:
    - coding
    - debugging
    - researching
    - writing
    - reviewing
    - testing
    - deploying
    """

    def __init__(
        self,
        *,
        name: str = "workflow_cognition_runtime",
        predictor: WorkflowPredictor | None = None,
        resume_planner: WorkflowResumePlanner | None = None,
    ) -> None:
        cleaned = name.strip()
        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._predictor = predictor or WorkflowPredictor()
        self._resume_planner = resume_planner or WorkflowResumePlanner()
        self._sessions: dict[str, WorkflowCognitionSession] = {}
        self._results: list[WorkflowCognitionResult] = []
        self._events: list[WorkflowCognitionRuntimeEvent] = []
        self._lock = RLock()
        self._last_reason: WorkflowCognitionReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def create_session(
        self,
        *,
        workspace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> WorkflowCognitionSession:
        session = WorkflowCognitionSession(
            workspace_id=workspace_id,
            metadata=metadata or {},
        )
        event = self._event(
            kind=WorkflowCognitionEventKind.SESSION_CREATED,
            reason=WorkflowCognitionReason.SESSION_CREATED,
            session_id=session.session_id,
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._events.append(event)
            self._last_reason = event.reason

        return session

    def analyze(
        self,
        *,
        session_id: str,
        memory_entry: WorkspaceMemoryEntry,
        user_intent: str | None = None,
    ) -> WorkflowCognitionResult:
        if self.session_for(session_id) is None:
            result = _blocked_result(
                reason=WorkflowCognitionReason.SESSION_NOT_FOUND,
                message="workflow cognition session not found",
            )
            self._record_result(result, session_id)
            return result

        signals = _signals_from_entry(memory_entry, user_intent)
        if not signals:
            signals = (
                WorkflowSignal(
                    kind=WorkflowSignalKind.FILE_SIGNAL,
                    workflow=WorkflowKind.UNKNOWN,
                    weight=0.20,
                    reason="semantic workflow context exists but workflow is unclear",
                ),
            )

        intention = _intention_from_signals(
            signals=signals,
            user_intent=user_intent,
        )
        continuity = _task_continuity_from_entry(memory_entry)
        context = WorkflowContext(
            workspace_id=memory_entry.workspace_id,
            app_name=memory_entry.app_name,
            project_path=memory_entry.project_path,
            memory_entry=memory_entry,
            task_continuity=continuity,
            intention=intention,
            signals=signals,
            summary=_context_summary(memory_entry, intention),
        )
        prediction = self._predictor.predict(context)

        if (
            intention.confidence < 0.45
            or intention.primary_workflow == WorkflowKind.UNKNOWN
        ):
            result = WorkflowCognitionResult(
                status=WorkflowCognitionStatus.LOW_CONFIDENCE,
                decision=WorkflowCognitionDecision.ASK_USER,
                reason=WorkflowCognitionReason.WORKFLOW_LOW_CONFIDENCE,
                context=context,
                prediction=prediction,
                trust=_trust(
                    confidence=intention.confidence,
                    reason="workflow cognition low confidence",
                ),
                message="workflow is unclear; ask user before continuing",
            )
            self._record_result(result, session_id)
            return result

        result = WorkflowCognitionResult(
            status=WorkflowCognitionStatus.UNDERSTOOD,
            decision=WorkflowCognitionDecision.CONTINUE,
            reason=WorkflowCognitionReason.WORKFLOW_UNDERSTOOD,
            context=context,
            prediction=prediction,
            trust=_trust(
                confidence=intention.confidence,
                reason="workflow cognition understood",
            ),
            message="workflow understood",
        )
        self._record_result(result, session_id)

        return result

    def plan_resume(
        self,
        *,
        session_id: str,
        context: WorkflowContext,
        continuity: SessionContinuity | None = None,
    ) -> WorkflowCognitionResult:
        if self.session_for(session_id) is None:
            result = _blocked_result(
                reason=WorkflowCognitionReason.SESSION_NOT_FOUND,
                message="workflow cognition session not found",
            )
            self._record_result(result, session_id)
            return result

        plan = self._resume_planner.plan(
            context=context,
            continuity=continuity,
        )
        result = WorkflowCognitionResult(
            status=WorkflowCognitionStatus.RESUME_READY,
            decision=WorkflowCognitionDecision.RESUME,
            reason=WorkflowCognitionReason.RESUME_PLAN_CREATED,
            context=context,
            prediction=self._predictor.predict(context),
            resume_plan=plan,
            trust=_trust(
                confidence=plan.confidence,
                reason="workflow resume plan created",
            ),
            message="workflow resume plan is ready",
        )
        self._record_result(result, session_id)

        return result

    def session_for(
        self,
        session_id: str,
    ) -> WorkflowCognitionSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def results(self) -> tuple[WorkflowCognitionResult, ...]:
        with self._lock:
            return tuple(self._results)

    def events(self) -> tuple[WorkflowCognitionRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def snapshot(self) -> WorkflowCognitionRuntimeSnapshot:
        with self._lock:
            return WorkflowCognitionRuntimeSnapshot(
                name=self.name,
                session_count=len(self._sessions),
                result_count=len(self._results),
                understood_count=sum(
                    1
                    for result in self._results
                    if result.status == WorkflowCognitionStatus.UNDERSTOOD
                ),
                resume_count=sum(
                    1
                    for result in self._results
                    if result.status == WorkflowCognitionStatus.RESUME_READY
                ),
                low_confidence_count=sum(
                    1
                    for result in self._results
                    if result.status == WorkflowCognitionStatus.LOW_CONFIDENCE
                ),
                blocked_count=sum(
                    1
                    for result in self._results
                    if result.status
                    in {
                        WorkflowCognitionStatus.BLOCKED,
                        WorkflowCognitionStatus.FAILED,
                    }
                ),
                runtime_event_count=len(self._events),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        event = self._event(
            kind=WorkflowCognitionEventKind.RUNTIME_RESET,
            reason=WorkflowCognitionReason.RUNTIME_RESET,
        )

        with self._lock:
            self._sessions.clear()
            self._results.clear()
            self._events.clear()
            self._events.append(event)
            self._last_reason = event.reason

    def _record_result(
        self,
        result: WorkflowCognitionResult,
        session_id: str,
    ) -> None:
        event = self._event(
            kind=_event_kind_for(result),
            reason=result.reason,
            session_id=session_id,
            result_id=result.result_id,
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
                        "analyze_count": session.analyze_count
                        + (
                            1
                            if result.status
                            in {
                                WorkflowCognitionStatus.UNDERSTOOD,
                                WorkflowCognitionStatus.LOW_CONFIDENCE,
                            }
                            else 0
                        ),
                        "resume_count": session.resume_count
                        + (
                            1
                            if result.status
                            == WorkflowCognitionStatus.RESUME_READY
                            else 0
                        ),
                        "blocked_count": session.blocked_count
                        + (
                            1
                            if result.status
                            in {
                                WorkflowCognitionStatus.BLOCKED,
                                WorkflowCognitionStatus.FAILED,
                            }
                            else 0
                        ),
                    }
                )

    @staticmethod
    def _event(
        *,
        kind: WorkflowCognitionEventKind,
        reason: WorkflowCognitionReason,
        session_id: str | None = None,
        result_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkflowCognitionRuntimeEvent:
        return WorkflowCognitionRuntimeEvent(
            kind=kind,
            reason=reason,
            session_id=session_id,
            result_id=result_id,
            metadata=metadata or {},
        )


def _signals_from_entry(
    entry: WorkspaceMemoryEntry,
    user_intent: str | None,
) -> tuple[WorkflowSignal, ...]:
    signals: list[WorkflowSignal] = []

    if entry.workflow_stage != WorkflowStage.UNKNOWN:
        workflow = _workflow_from_stage(entry.workflow_stage)
        signals.append(
            WorkflowSignal(
                kind=WorkflowSignalKind.STAGE_SIGNAL,
                workflow=workflow,
                weight=0.85,
                reason=f"workflow stage indicates {workflow.value}",
            )
        )

    for file_path in entry.active_files:
        signals.extend(_file_signals(file_path))

    for command in entry.recent_commands:
        signals.extend(_command_signals(command))

    for error in entry.visible_errors:
        signals.append(
            WorkflowSignal(
                kind=WorkflowSignalKind.ERROR_SIGNAL,
                workflow=WorkflowKind.DEBUGGING,
                weight=0.90,
                reason=f"visible error indicates debugging: {_short(error)}",
            )
        )

    for todo in entry.pending_todos:
        signals.extend(_todo_signals(todo))

        if entry.terminal_directory and (
            entry.active_files
            or entry.recent_commands
            or entry.visible_errors
            or entry.pending_todos
        ):
            signals.append(
                WorkflowSignal(
                    kind=WorkflowSignalKind.TERMINAL_SIGNAL,
                    workflow=WorkflowKind.CODING,
                    weight=0.15,
                    reason="terminal directory provides weak workspace context",
                )
            )

    if user_intent:
        signals.extend(_intent_signals(user_intent))

    return tuple(signals)


def _intention_from_signals(
    *,
    signals: tuple[WorkflowSignal, ...],
    user_intent: str | None,
) -> IntentionModel:
    scores: dict[WorkflowKind, float] = {}
    for signal in signals:
        scores[signal.workflow] = scores.get(signal.workflow, 0.0) + signal.weight

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    primary = ranked[0][0] if ranked else WorkflowKind.UNKNOWN

    total = sum(scores.values()) or 1.0
    confidence = min(1.0, ranked[0][1] / total) if ranked else 0.0

    strong_signals = [
        signal
        for signal in signals
        if signal.kind
        in {
            WorkflowSignalKind.INTENT_SIGNAL,
            WorkflowSignalKind.COMMAND_SIGNAL,
            WorkflowSignalKind.ERROR_SIGNAL,
            WorkflowSignalKind.STAGE_SIGNAL,
        }
        and signal.weight >= 0.85
    ]

    if strong_signals:
        strongest = max(strong_signals, key=lambda signal: signal.weight)
        primary = strongest.workflow
        confidence = max(confidence, strongest.weight)

    if primary == WorkflowKind.UNKNOWN:
        confidence = 0.20

    secondary = tuple(
        workflow
        for workflow, _ in ranked
        if workflow != primary
    )[:2]
    band = _confidence_band(confidence)

    return IntentionModel(
        primary_workflow=primary,
        secondary_workflows=secondary,
        confidence=confidence,
        confidence_band=band,
        user_intent=user_intent,
        inferred_goal=_goal_for(primary),
    )


def _task_continuity_from_entry(entry: WorkspaceMemoryEntry) -> TaskContinuity:
    cursor_summary = None
    if entry.cursor_positions:
        cursor = entry.cursor_positions[0]
        cursor_summary = (
            f"{cursor.file_path}:{cursor.line}:{cursor.column}"
            + (f" symbol={cursor.symbol}" if cursor.symbol else "")
        )

    return TaskContinuity(
        continuity_token=entry.continuity_token,
        workflow_stage=entry.workflow_stage,
        active_files=entry.active_files,
        cursor_summary=cursor_summary,
        terminal_directory=entry.terminal_directory,
        recent_commands=entry.recent_commands,
        visible_errors=entry.visible_errors,
        pending_todos=entry.pending_todos,
        blocked=bool(entry.visible_errors),
        blocked_reason=entry.visible_errors[0] if entry.visible_errors else None,
    )


def _file_signals(file_path: str) -> tuple[WorkflowSignal, ...]:
    lowered = file_path.lower()
    signals: list[WorkflowSignal] = []

    if lowered.endswith((".py", ".ts", ".tsx", ".js", ".go", ".rs", ".java")):
        signals.append(
            WorkflowSignal(
                kind=WorkflowSignalKind.FILE_SIGNAL,
                workflow=WorkflowKind.CODING,
                weight=0.70,
                reason=f"source file active: {file_path}",
            )
        )

    if "test" in lowered or lowered.endswith(("_test.py", ".spec.ts")):
        signals.append(
            WorkflowSignal(
                kind=WorkflowSignalKind.FILE_SIGNAL,
                workflow=WorkflowKind.TESTING,
                weight=0.75,
                reason=f"test file active: {file_path}",
            )
        )

    if lowered.endswith((".md", ".txt", ".docx")):
        signals.append(
            WorkflowSignal(
                kind=WorkflowSignalKind.FILE_SIGNAL,
                workflow=WorkflowKind.WRITING,
                weight=0.65,
                reason=f"writing file active: {file_path}",
            )
        )

    return tuple(signals)


def _command_signals(command: str) -> tuple[WorkflowSignal, ...]:
    lowered = command.lower()
    signals: list[WorkflowSignal] = []

    if any(term in lowered for term in ("pytest", "ruff", "mypy", "npm test")):
        signals.append(
            WorkflowSignal(
                kind=WorkflowSignalKind.COMMAND_SIGNAL,
                workflow=WorkflowKind.TESTING,
                weight=0.90,
                reason=f"test command observed: {_short(command)}",
            )
        )

    if any(term in lowered for term in ("git diff", "git status", "review")):
        signals.append(
            WorkflowSignal(
                kind=WorkflowSignalKind.COMMAND_SIGNAL,
                workflow=WorkflowKind.REVIEWING,
                weight=0.70,
                reason=f"review command observed: {_short(command)}",
            )
        )

    if any(term in lowered for term in ("deploy", "kubectl", "docker push")):
        signals.append(
            WorkflowSignal(
                kind=WorkflowSignalKind.COMMAND_SIGNAL,
                workflow=WorkflowKind.DEPLOYING,
                weight=0.90,
                reason=f"deployment command observed: {_short(command)}",
            )
        )

    return tuple(signals)


def _todo_signals(todo: str) -> tuple[WorkflowSignal, ...]:
    lowered = todo.lower()

    if any(term in lowered for term in ("bug", "fix", "error", "fail")):
        return (
            WorkflowSignal(
                kind=WorkflowSignalKind.TODO_SIGNAL,
                workflow=WorkflowKind.DEBUGGING,
                weight=0.65,
                reason=f"todo indicates debugging: {_short(todo)}",
            ),
        )

    if any(term in lowered for term in ("write", "draft", "document")):
        return (
            WorkflowSignal(
                kind=WorkflowSignalKind.TODO_SIGNAL,
                workflow=WorkflowKind.WRITING,
                weight=0.65,
                reason=f"todo indicates writing: {_short(todo)}",
            ),
        )

    return (
        WorkflowSignal(
            kind=WorkflowSignalKind.TODO_SIGNAL,
            workflow=WorkflowKind.CODING,
            weight=0.35,
            reason=f"todo indicates active work: {_short(todo)}",
        ),
    )


def _intent_signals(intent: str) -> tuple[WorkflowSignal, ...]:
    lowered = intent.lower()
    mapping = (
        (WorkflowKind.DEBUGGING, ("debug", "fix", "error", "traceback")),
        (WorkflowKind.RESEARCHING, ("research", "search", "find sources")),
        (WorkflowKind.WRITING, ("write", "draft", "compose")),
        (WorkflowKind.REVIEWING, ("review", "compare", "inspect")),
        (WorkflowKind.TESTING, ("test", "pytest", "verify")),
        (WorkflowKind.DEPLOYING, ("deploy", "release", "publish")),
        (WorkflowKind.CODING, ("code", "implement", "build")),
    )

    for workflow, terms in mapping:
        if any(term in lowered for term in terms):
            return (
                WorkflowSignal(
                    kind=WorkflowSignalKind.INTENT_SIGNAL,
                    workflow=workflow,
                    weight=0.95,
                    reason=f"user intent indicates {workflow.value}",
                ),
            )

    return ()


def _workflow_from_stage(stage: WorkflowStage) -> WorkflowKind:
    mapping = {
        WorkflowStage.CODING: WorkflowKind.CODING,
        WorkflowStage.DEBUGGING: WorkflowKind.DEBUGGING,
        WorkflowStage.TESTING: WorkflowKind.TESTING,
        WorkflowStage.RESEARCHING: WorkflowKind.RESEARCHING,
        WorkflowStage.WRITING: WorkflowKind.WRITING,
        WorkflowStage.REVIEWING: WorkflowKind.REVIEWING,
        WorkflowStage.BLOCKED: WorkflowKind.DEBUGGING,
    }
    return mapping.get(stage, WorkflowKind.UNKNOWN)


def _resume_actions_for(context: WorkflowContext) -> tuple[str, ...]:
    workflow = context.intention.primary_workflow

    if workflow == WorkflowKind.DEBUGGING:
        return (
            "open the most relevant visible error",
            "inspect the active file around the cursor",
            "run the smallest targeted test after the fix",
        )

    if workflow == WorkflowKind.TESTING:
        return (
            "parse the latest test output",
            "open the failing test or mark tests complete",
        )

    if workflow == WorkflowKind.RESEARCHING:
        return (
            "resume source comparison",
            "extract claims and contradictions",
        )

    if workflow == WorkflowKind.WRITING:
        return (
            "open the active draft",
            "continue from the last section",
        )

    if workflow == WorkflowKind.DEPLOYING:
        return (
            "verify deployment command output",
            "check health logs before declaring success",
        )

    return (
        "restore active files",
        "summarize current project state",
        "ask user what to continue",
    )


def _context_summary(
    entry: WorkspaceMemoryEntry,
    intention: IntentionModel,
) -> str:
    project = entry.project_path or "unknown project"
    files = ", ".join(entry.active_files[:3]) or "no active files"
    errors = "; ".join(entry.visible_errors[:2]) or "no visible errors"
    todos = "; ".join(entry.pending_todos[:2]) or "no pending todos"

    return (
        f"{intention.primary_workflow.value} workflow in {entry.app_name}; "
        f"project={project}; files={files}; "
        f"errors={errors}; todos={todos}"
    )


def _goal_for(workflow: WorkflowKind) -> str:
    goals = {
        WorkflowKind.CODING: "continue implementing the active code task",
        WorkflowKind.DEBUGGING: "identify and fix the current failure",
        WorkflowKind.RESEARCHING: "gather and compare useful information",
        WorkflowKind.WRITING: "continue writing the active document",
        WorkflowKind.REVIEWING: "inspect and improve the current work",
        WorkflowKind.TESTING: "verify the current work through tests",
        WorkflowKind.DEPLOYING: "ship safely and verify deployment health",
    }
    return goals.get(workflow, "clarify the current workflow")


def _confidence_band(confidence: float) -> WorkflowConfidenceBand:
    if confidence >= 0.75:
        return WorkflowConfidenceBand.HIGH

    if confidence >= 0.45:
        return WorkflowConfidenceBand.MEDIUM

    return WorkflowConfidenceBand.LOW


def _event_kind_for(
    result: WorkflowCognitionResult,
) -> WorkflowCognitionEventKind:
    if result.status == WorkflowCognitionStatus.RESUME_READY:
        return WorkflowCognitionEventKind.RESUME_PLANNED

    if result.status in {
        WorkflowCognitionStatus.BLOCKED,
        WorkflowCognitionStatus.FAILED,
    }:
        return WorkflowCognitionEventKind.OPERATION_BLOCKED

    return WorkflowCognitionEventKind.WORKFLOW_ANALYZED


def _blocked_result(
    *,
    reason: WorkflowCognitionReason,
    message: str,
) -> WorkflowCognitionResult:
    return WorkflowCognitionResult(
        status=WorkflowCognitionStatus.FAILED
        if reason == WorkflowCognitionReason.SESSION_NOT_FOUND
        else WorkflowCognitionStatus.BLOCKED,
        decision=WorkflowCognitionDecision.BLOCK,
        reason=reason,
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


def _short(value: str, *, limit: int = 80) -> str:
    cleaned = value.strip()
    if len(cleaned) <= limit:
        return cleaned

    return cleaned[: limit - 3] + "..."


def _clean_required(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("field cannot be empty.")
    return cleaned