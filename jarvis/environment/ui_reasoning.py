from __future__ import annotations

from enum import StrEnum
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.environment.app_identity import DetectedAppKind
from jarvis.environment.models import EnvironmentSource, TrustCalibration
from jarvis.environment.trust_runtime import TrustPolicyClassification
from jarvis.environment.ui_patterns import UIPatternKind, UIPatternRecognitionResult
from jarvis.environment.ui_semantics import SemanticSceneKind, UIContext
from jarvis.environment.visual_grounding import (
    GroundingStatus,
    VisualGroundingResult,
)
from jarvis.environment.workspace_graph import (
    GraphNode,
    GraphNodeKind,
    WorkspaceCognitiveGraph,
)
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class UIReasoningIntentKind(StrEnum):
    FIX = "fix"
    OPEN = "open"
    RUN = "run"
    COPY = "copy"
    FOCUS = "focus"
    INSPECT = "inspect"
    SELECT = "select"
    READ = "read"
    UNKNOWN = "unknown"


class UIReasoningContextKind(StrEnum):
    CODE_CONTEXT = "code_context"
    TERMINAL_CONTEXT = "terminal_context"
    BROWSER_CONTEXT = "browser_context"
    FILE_CONTEXT = "file_context"
    DIALOG_CONTEXT = "dialog_context"
    ERROR_CONTEXT = "error_context"
    LOADING_CONTEXT = "loading_context"
    UNKNOWN_CONTEXT = "unknown_context"


class UIReasoningDecision(StrEnum):
    RESOLVED = "resolved"
    VERIFY_FIRST = "verify_first"
    ASK_USER = "ask_user"
    BLOCKED = "blocked"
    NOT_ENOUGH_CONTEXT = "not_enough_context"


class UIReasoningStatus(StrEnum):
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    BLOCKED = "blocked"
    UNRESOLVED = "unresolved"
    FAILED = "failed"


class UIReasoningReason(StrEnum):
    SESSION_CREATED = "session_created"
    INTENT_RESOLVED = "intent_resolved"
    INTENT_AMBIGUOUS = "intent_ambiguous"
    INTENT_BLOCKED = "intent_blocked"
    NOT_ENOUGH_CONTEXT = "not_enough_context"
    SESSION_NOT_FOUND = "session_not_found"
    RUNTIME_RESET = "runtime_reset"


class UIReasoningEventKind(StrEnum):
    SESSION_CREATED = "session_created"
    REASONING_COMPLETED = "reasoning_completed"
    REASONING_BLOCKED = "reasoning_blocked"
    RUNTIME_RESET = "runtime_reset"


class PlannerHintKind(StrEnum):
    INSPECT_ERROR = "inspect_error"
    OPEN_FILE = "open_file"
    RUN_COMMAND = "run_command"
    COPY_SELECTION = "copy_selection"
    FOCUS_TARGET = "focus_target"
    VERIFY_DIALOG = "verify_dialog"
    WAIT_FOR_STABLE_STATE = "wait_for_stable_state"
    ASK_CLARIFICATION = "ask_clarification"
    NO_ACTION = "no_action"


class EnvironmentIntent(OrchestrationModel):
    """
    Parsed natural-language intent.

    This is environment intent, not executable action.
    """

    intent_id: str = Field(default_factory=lambda: f"env_intent_{uuid4().hex}")
    raw_text: str
    kind: UIReasoningIntentKind
    confidence: float = Field(ge=0.0, le=1.0)
    object_hint: str | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("intent_id", "raw_text")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class UIContextChain(OrchestrationModel):
    """
    Context chain used to resolve vague instructions.

    It combines semantic scene, pattern, visual grounding, app kind,
    graph state, selected/focused graph nodes, and recent metadata.
    """

    chain_id: str = Field(default_factory=lambda: f"ui_context_chain_{uuid4().hex}")
    app_kind: DetectedAppKind | None = None
    ui_context: UIContext | None = None
    pattern_result: UIPatternRecognitionResult | None = None
    grounding_result: VisualGroundingResult | None = None
    workspace_graph: WorkspaceCognitiveGraph | None = None
    focused_node_id: str | None = None
    selected_node_id: str | None = None
    selected_text: str | None = None
    current_command: str | None = None
    current_file: str | None = None
    current_project: str | None = None
    browser_selection: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("chain_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class ResolvedUITarget(OrchestrationModel):
    """
    Environment-aware target.

    This may reference graph node, grounded visual target, selection,
    semantic scene, or context inference.
    """

    target_id: str = Field(default_factory=lambda: f"resolved_target_{uuid4().hex}")
    label: str
    context_kind: UIReasoningContextKind
    confidence: float = Field(ge=0.0, le=1.0)
    policy: TrustPolicyClassification
    graph_node: GraphNode | None = None
    grounding_result: VisualGroundingResult | None = None
    scene_kind: SemanticSceneKind | None = None
    pattern_kind: UIPatternKind | None = None
    text_selection: str | None = None
    trust: TrustCalibration
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("target_id", "label")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class ActionPlannerHint(OrchestrationModel):
    """
    Hint for later action planning.

    This is not an action plan and never executes directly.
    """

    hint_id: str = Field(default_factory=lambda: f"planner_hint_{uuid4().hex}")
    kind: PlannerHintKind
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    requires_verification: bool = True
    allowed_to_plan: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("hint_id", "description")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class UIReasoningRequest(OrchestrationModel):
    """
    Request for context-aware UI reasoning.
    """

    request_id: str = Field(default_factory=lambda: f"ui_reason_req_{uuid4().hex}")
    session_id: str
    utterance: str
    context_chain: UIContextChain
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("request_id", "session_id", "utterance")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class UIReasoningResult(OrchestrationModel):
    """
    Final context-aware UI reasoning result.
    """

    result_id: str = Field(default_factory=lambda: f"ui_reason_result_{uuid4().hex}")
    status: UIReasoningStatus
    reason: UIReasoningReason
    decision: UIReasoningDecision
    request_id: str
    intent: EnvironmentIntent
    context_kind: UIReasoningContextKind
    resolved_target: ResolvedUITarget | None = None
    planner_hints: tuple[ActionPlannerHint, ...] = ()
    safe_for_action_planning: bool
    message: str
    created_at: object = Field(default_factory=utc_now)

    @field_validator("result_id", "request_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _resolved_requires_target(self) -> UIReasoningResult:
        if self.status == UIReasoningStatus.RESOLVED:
            if self.resolved_target is None:
                raise ValueError("resolved result requires resolved_target.")

        return self


class UIReasoningSession(OrchestrationModel):
    """
    UI reasoning runtime session.
    """

    session_id: str = Field(default_factory=lambda: f"ui_reasoning_{uuid4().hex}")
    workspace_id: str
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class UIReasoningRuntimeEvent(OrchestrationModel):
    """
    UI reasoning runtime event.
    """

    event_id: str = Field(default_factory=lambda: f"ui_reason_event_{uuid4().hex}")
    kind: UIReasoningEventKind
    reason: UIReasoningReason
    session_id: str | None = None
    result_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class UIReasoningRuntimeSnapshot(OrchestrationModel):
    """
    Diagnostics for Phase 8 Step 18.
    """

    name: str
    session_count: int = Field(ge=0)
    result_count: int = Field(ge=0)
    resolved_count: int = Field(ge=0)
    ambiguous_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    unresolved_count: int = Field(ge=0)
    safe_planning_count: int = Field(ge=0)
    runtime_event_count: int = Field(ge=0)
    last_reason: UIReasoningReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        return _clean_required(value)


class EnvironmentIntentResolver:
    """
    Resolves utterance into environment intent.
    """

    def resolve(self, utterance: str) -> EnvironmentIntent:
        text = _norm(utterance)

        if _contains_any(text, ("fix", "solve", "repair")):
            return _intent(utterance, UIReasoningIntentKind.FIX, 0.90)

        if _contains_any(text, ("open", "launch")):
            return _intent(utterance, UIReasoningIntentKind.OPEN, 0.88)

        if _contains_any(text, ("run", "execute", "start")):
            return _intent(utterance, UIReasoningIntentKind.RUN, 0.90)

        if _contains_any(text, ("copy", "clip")):
            return _intent(utterance, UIReasoningIntentKind.COPY, 0.88)

        if _contains_any(text, ("focus", "switch")):
            return _intent(utterance, UIReasoningIntentKind.FOCUS, 0.86)

        if _contains_any(text, ("inspect", "check", "look")):
            return _intent(utterance, UIReasoningIntentKind.INSPECT, 0.86)

        if _contains_any(text, ("select", "choose")):
            return _intent(utterance, UIReasoningIntentKind.SELECT, 0.84)

        if _contains_any(text, ("read", "tell")):
            return _intent(utterance, UIReasoningIntentKind.READ, 0.84)

        return _intent(utterance, UIReasoningIntentKind.UNKNOWN, 0.20)


class IntentResolver:
    """
    Resolves vague words like 'that', 'it', and 'this' using environment context.
    """

    def resolve_target(
        self,
        *,
        intent: EnvironmentIntent,
        chain: UIContextChain,
    ) -> ResolvedUITarget | None:
        if intent.kind == UIReasoningIntentKind.FIX:
            return self._resolve_fix(chain)

        if intent.kind == UIReasoningIntentKind.OPEN:
            return self._resolve_open(chain)

        if intent.kind == UIReasoningIntentKind.RUN:
            return self._resolve_run(chain)

        if intent.kind == UIReasoningIntentKind.COPY:
            return self._resolve_copy(chain)

        if intent.kind in {
            UIReasoningIntentKind.FOCUS,
            UIReasoningIntentKind.INSPECT,
            UIReasoningIntentKind.READ,
            UIReasoningIntentKind.SELECT,
        }:
            return self._resolve_general(intent=intent, chain=chain)

        return None

    def _resolve_fix(self, chain: UIContextChain) -> ResolvedUITarget | None:
        error_node = _find_node(chain, GraphNodeKind.ERROR)

        if error_node is not None:
            return _target(
                label=error_node.label,
                context_kind=UIReasoningContextKind.ERROR_CONTEXT,
                confidence=0.90,
                policy=TrustPolicyClassification.VERIFY_FIRST,
                graph_node=error_node,
                scene_kind=SemanticSceneKind.ERROR_DIALOG,
                metadata={"resolution": "visible error node"},
            )

        if chain.ui_context is not None:
            if chain.ui_context.scene.kind == SemanticSceneKind.ERROR_DIALOG:
                return _target(
                    label=chain.ui_context.scene.summary,
                    context_kind=UIReasoningContextKind.ERROR_CONTEXT,
                    confidence=chain.ui_context.scene.confidence,
                    policy=chain.ui_context.policy_classification,
                    scene_kind=chain.ui_context.scene.kind,
                    metadata={"resolution": "semantic error scene"},
                )

        return None

    def _resolve_open(self, chain: UIContextChain) -> ResolvedUITarget | None:
        selected = _selected_or_focused_node(chain)

        if selected is not None:
            return _target(
                label=selected.label,
                context_kind=UIReasoningContextKind.FILE_CONTEXT
                if selected.kind == GraphNodeKind.FILE
                else UIReasoningContextKind.UNKNOWN_CONTEXT,
                confidence=0.88,
                policy=TrustPolicyClassification.VERIFY_FIRST,
                graph_node=selected,
                metadata={"resolution": "selected or focused graph node"},
            )

        file_node = _find_node(chain, GraphNodeKind.FILE)

        if file_node is not None:
            return _target(
                label=file_node.label,
                context_kind=UIReasoningContextKind.FILE_CONTEXT,
                confidence=0.82,
                policy=TrustPolicyClassification.VERIFY_FIRST,
                graph_node=file_node,
                metadata={"resolution": "nearest file node"},
            )

        return None

    def _resolve_run(self, chain: UIContextChain) -> ResolvedUITarget | None:
        command = _find_node(chain, GraphNodeKind.COMMAND)

        if command is not None:
            return _target(
                label=command.label,
                context_kind=UIReasoningContextKind.TERMINAL_CONTEXT,
                confidence=0.90,
                policy=TrustPolicyClassification.VERIFY_FIRST,
                graph_node=command,
                metadata={"resolution": "current command graph node"},
            )

        if chain.current_command:
            return _target(
                label=chain.current_command,
                context_kind=UIReasoningContextKind.TERMINAL_CONTEXT,
                confidence=0.86,
                policy=TrustPolicyClassification.VERIFY_FIRST,
                metadata={"resolution": "current command string"},
            )

        if chain.current_project:
            return _target(
                label=chain.current_project,
                context_kind=UIReasoningContextKind.CODE_CONTEXT,
                confidence=0.78,
                policy=TrustPolicyClassification.VERIFY_FIRST,
                metadata={"resolution": "current project"},
            )

        return None

    def _resolve_copy(self, chain: UIContextChain) -> ResolvedUITarget | None:
        if chain.browser_selection:
            return _target(
                label="browser selection",
                context_kind=UIReasoningContextKind.BROWSER_CONTEXT,
                confidence=0.88,
                policy=TrustPolicyClassification.SAFE,
                text_selection=chain.browser_selection,
                metadata={"resolution": "browser selected content"},
            )

        if chain.selected_text:
            return _target(
                label="selected text",
                context_kind=UIReasoningContextKind.UNKNOWN_CONTEXT,
                confidence=0.84,
                policy=TrustPolicyClassification.SAFE,
                text_selection=chain.selected_text,
                metadata={"resolution": "generic selected text"},
            )

        return None

    def _resolve_general(
        self,
        *,
        intent: EnvironmentIntent,
        chain: UIContextChain,
    ) -> ResolvedUITarget | None:
        if chain.grounding_result is not None:
            if chain.grounding_result.status == GroundingStatus.GROUNDED:
                selected = chain.grounding_result.selected

                if selected is not None:
                    return _target(
                        label=selected.label,
                        context_kind=_context_kind_from_chain(chain),
                        confidence=selected.confidence,
                        policy=selected.policy,
                        grounding_result=chain.grounding_result,
                        metadata={"resolution": f"grounded {intent.kind.value}"},
                    )

        focused = _selected_or_focused_node(chain)

        if focused is not None:
            return _target(
                label=focused.label,
                context_kind=_context_kind_from_node(focused),
                confidence=0.76,
                policy=TrustPolicyClassification.REVIEW,
                graph_node=focused,
                metadata={"resolution": "focused graph node"},
            )

        return None


class ActionPlannerHints:
    """
    Converts resolved environment intent into safe planner hints.

    These are hints only. They do not execute.
    """

    def build(
        self,
        *,
        intent: EnvironmentIntent,
        target: ResolvedUITarget | None,
        decision: UIReasoningDecision,
    ) -> tuple[ActionPlannerHint, ...]:
        if target is None:
            return (
                ActionPlannerHint(
                    kind=PlannerHintKind.ASK_CLARIFICATION,
                    description="ask user to clarify target",
                    confidence=0.90,
                    requires_verification=False,
                    allowed_to_plan=False,
                ),
            )

        if decision in {
            UIReasoningDecision.ASK_USER,
            UIReasoningDecision.NOT_ENOUGH_CONTEXT,
        }:
            return (
                ActionPlannerHint(
                    kind=PlannerHintKind.ASK_CLARIFICATION,
                    description="ask user before planning action",
                    confidence=0.90,
                    requires_verification=False,
                    allowed_to_plan=False,
                ),
            )

        if decision == UIReasoningDecision.BLOCKED:
            return (
                ActionPlannerHint(
                    kind=PlannerHintKind.NO_ACTION,
                    description="context is blocked; do not plan action",
                    confidence=1.0,
                    requires_verification=False,
                    allowed_to_plan=False,
                ),
            )

        hint_kind = _hint_kind(intent=intent, target=target)
        allowed = (
            decision == UIReasoningDecision.RESOLVED
            and target.policy == TrustPolicyClassification.SAFE
        )

        return (
            ActionPlannerHint(
                kind=hint_kind,
                description=f"{hint_kind.value} for {target.label}",
                confidence=target.confidence,
                requires_verification=target.policy
                != TrustPolicyClassification.SAFE,
                allowed_to_plan=allowed,
                metadata={"target": target.label},
            ),
        )


class UIReasoningEngine:
    """
    Coordinates environment intent resolution and planner hint generation.
    """

    def __init__(
        self,
        *,
        environment_intent_resolver: EnvironmentIntentResolver | None = None,
        intent_resolver: IntentResolver | None = None,
        planner_hints: ActionPlannerHints | None = None,
    ) -> None:
        self._environment_intent_resolver = (
            environment_intent_resolver or EnvironmentIntentResolver()
        )
        self._intent_resolver = intent_resolver or IntentResolver()
        self._planner_hints = planner_hints or ActionPlannerHints()

    def reason(self, request: UIReasoningRequest) -> UIReasoningResult:
        intent = self._environment_intent_resolver.resolve(request.utterance)
        context_kind = _context_kind_from_chain(request.context_chain)
        target = self._intent_resolver.resolve_target(
            intent=intent,
            chain=request.context_chain,
        )
        decision = _decision_for(intent=intent, target=target)
        status, reason = _status_reason_for(decision)
        hints = self._planner_hints.build(
            intent=intent,
            target=target,
            decision=decision,
        )
        safe_for_action_planning = any(hint.allowed_to_plan for hint in hints)

        return UIReasoningResult(
            status=status,
            reason=reason,
            decision=decision,
            request_id=request.request_id,
            intent=intent,
            context_kind=context_kind,
            resolved_target=target,
            planner_hints=hints,
            safe_for_action_planning=safe_for_action_planning,
            message=_message_for(decision=decision, target=target),
        )


class ContextAwareUIReasoningRuntime:
    """
    Phase 8 Step 18 Context-Aware UI Reasoning Runtime.

    Responsibilities:
    - resolve vague utterances using UI context
    - resolve "that", "it", "this" from graph/semantic/grounding state
    - produce action planner hints
    - block unsafe or under-specified intent

    Non-responsibilities:
    - no clicking
    - no typing
    - no direct tool execution
    - no policy bypass
    """

    def __init__(
        self,
        *,
        name: str = "context_aware_ui_reasoning_runtime",
        engine: UIReasoningEngine | None = None,
    ) -> None:
        cleaned = name.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._engine = engine or UIReasoningEngine()
        self._sessions: dict[str, UIReasoningSession] = {}
        self._results: list[UIReasoningResult] = []
        self._events: list[UIReasoningRuntimeEvent] = []
        self._lock = RLock()
        self._last_reason: UIReasoningReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def create_session(
        self,
        *,
        workspace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> UIReasoningSession:
        session = UIReasoningSession(
            workspace_id=workspace_id,
            metadata=metadata or {},
        )
        event = self._event(
            kind=UIReasoningEventKind.SESSION_CREATED,
            reason=UIReasoningReason.SESSION_CREATED,
            session_id=session.session_id,
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._events.append(event)
            self._last_reason = event.reason

        return session

    def reason(self, request: UIReasoningRequest) -> UIReasoningResult:
        if self.session_for(request.session_id) is None:
            result = _failed_result(request)
            self._record_result(result, session_id=request.session_id)
            return result

        result = self._engine.reason(request)
        self._record_result(result, session_id=request.session_id)
        self._touch_session(request.session_id)

        return result

    def session_for(self, session_id: str) -> UIReasoningSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def results(self) -> tuple[UIReasoningResult, ...]:
        with self._lock:
            return tuple(self._results)

    def events(self) -> tuple[UIReasoningRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def snapshot(self) -> UIReasoningRuntimeSnapshot:
        with self._lock:
            return UIReasoningRuntimeSnapshot(
                name=self.name,
                session_count=len(self._sessions),
                result_count=len(self._results),
                resolved_count=sum(
                    1
                    for result in self._results
                    if result.status == UIReasoningStatus.RESOLVED
                ),
                ambiguous_count=sum(
                    1
                    for result in self._results
                    if result.status == UIReasoningStatus.AMBIGUOUS
                ),
                blocked_count=sum(
                    1
                    for result in self._results
                    if result.status == UIReasoningStatus.BLOCKED
                ),
                unresolved_count=sum(
                    1
                    for result in self._results
                    if result.status == UIReasoningStatus.UNRESOLVED
                ),
                safe_planning_count=sum(
                    1
                    for result in self._results
                    if result.safe_for_action_planning
                ),
                runtime_event_count=len(self._events),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        event = self._event(
            kind=UIReasoningEventKind.RUNTIME_RESET,
            reason=UIReasoningReason.RUNTIME_RESET,
        )

        with self._lock:
            self._sessions.clear()
            self._results.clear()
            self._events.clear()
            self._events.append(event)
            self._last_reason = event.reason

    def _record_result(
        self,
        result: UIReasoningResult,
        *,
        session_id: str,
    ) -> None:
        event = self._event(
            kind=(
                UIReasoningEventKind.REASONING_COMPLETED
                if result.status == UIReasoningStatus.RESOLVED
                else UIReasoningEventKind.REASONING_BLOCKED
            ),
            reason=result.reason,
            session_id=session_id,
            result_id=result.result_id,
            metadata={
                "status": result.status.value,
                "decision": result.decision.value,
            },
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
        kind: UIReasoningEventKind,
        reason: UIReasoningReason,
        session_id: str | None = None,
        result_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UIReasoningRuntimeEvent:
        return UIReasoningRuntimeEvent(
            kind=kind,
            reason=reason,
            session_id=session_id,
            result_id=result_id,
            metadata=metadata or {},
        )


def _intent(
    raw_text: str,
    kind: UIReasoningIntentKind,
    confidence: float,
) -> EnvironmentIntent:
    return EnvironmentIntent(
        raw_text=raw_text,
        kind=kind,
        confidence=confidence,
        object_hint=_object_hint(raw_text),
    )


def _object_hint(text: str) -> str | None:
    normalized = _norm(text)

    for word in ("that", "it", "this", "current", "selected"):
        if word in normalized:
            return word

    return None


def _decision_for(
    *,
    intent: EnvironmentIntent,
    target: ResolvedUITarget | None,
) -> UIReasoningDecision:
    if intent.kind == UIReasoningIntentKind.UNKNOWN:
        return UIReasoningDecision.NOT_ENOUGH_CONTEXT

    if target is None:
        return UIReasoningDecision.ASK_USER

    if target.policy == TrustPolicyClassification.BLOCKED:
        return UIReasoningDecision.BLOCKED

    if target.policy in {
        TrustPolicyClassification.REVIEW,
        TrustPolicyClassification.VERIFY_FIRST,
    }:
        return UIReasoningDecision.VERIFY_FIRST

    if target.confidence < 0.70:
        return UIReasoningDecision.ASK_USER

    return UIReasoningDecision.RESOLVED


def _status_reason_for(
    decision: UIReasoningDecision,
) -> tuple[UIReasoningStatus, UIReasoningReason]:
    if decision == UIReasoningDecision.RESOLVED:
        return UIReasoningStatus.RESOLVED, UIReasoningReason.INTENT_RESOLVED

    if decision == UIReasoningDecision.VERIFY_FIRST:
        return UIReasoningStatus.RESOLVED, UIReasoningReason.INTENT_RESOLVED

    if decision == UIReasoningDecision.ASK_USER:
        return UIReasoningStatus.AMBIGUOUS, UIReasoningReason.INTENT_AMBIGUOUS

    if decision == UIReasoningDecision.BLOCKED:
        return UIReasoningStatus.BLOCKED, UIReasoningReason.INTENT_BLOCKED

    return UIReasoningStatus.UNRESOLVED, UIReasoningReason.NOT_ENOUGH_CONTEXT


def _message_for(
    *,
    decision: UIReasoningDecision,
    target: ResolvedUITarget | None,
) -> str:
    if target is None:
        return "not enough UI context to resolve intent"

    if decision == UIReasoningDecision.RESOLVED:
        return f"resolved intent target: {target.label}"

    if decision == UIReasoningDecision.VERIFY_FIRST:
        return f"resolved target requires verification: {target.label}"

    if decision == UIReasoningDecision.BLOCKED:
        return f"target blocked by policy: {target.label}"

    return "intent requires clarification"


def _failed_result(request: UIReasoningRequest) -> UIReasoningResult:
    intent = EnvironmentIntent(
        raw_text=request.utterance,
        kind=UIReasoningIntentKind.UNKNOWN,
        confidence=0.0,
    )

    return UIReasoningResult(
        status=UIReasoningStatus.FAILED,
        reason=UIReasoningReason.SESSION_NOT_FOUND,
        decision=UIReasoningDecision.BLOCKED,
        request_id=request.request_id,
        intent=intent,
        context_kind=UIReasoningContextKind.UNKNOWN_CONTEXT,
        resolved_target=None,
        planner_hints=(
            ActionPlannerHint(
                kind=PlannerHintKind.NO_ACTION,
                description="session missing; do not plan action",
                confidence=1.0,
                requires_verification=False,
                allowed_to_plan=False,
            ),
        ),
        safe_for_action_planning=False,
        message="UI reasoning session not found",
    )


def _find_node(
    chain: UIContextChain,
    kind: GraphNodeKind,
) -> GraphNode | None:
    if chain.workspace_graph is None:
        return None

    for node in chain.workspace_graph.nodes.values():
        if node.active and node.kind == kind:
            return node

    return None


def _selected_or_focused_node(chain: UIContextChain) -> GraphNode | None:
    if chain.workspace_graph is None:
        return None

    for node_id in (chain.selected_node_id, chain.focused_node_id):
        if node_id is None:
            continue

        node = chain.workspace_graph.nodes.get(node_id)

        if node is not None and node.active:
            return node

    return None


def _target(
    *,
    label: str,
    context_kind: UIReasoningContextKind,
    confidence: float,
    policy: TrustPolicyClassification,
    graph_node: GraphNode | None = None,
    grounding_result: VisualGroundingResult | None = None,
    scene_kind: SemanticSceneKind | None = None,
    pattern_kind: UIPatternKind | None = None,
    text_selection: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ResolvedUITarget:
    return ResolvedUITarget(
        label=label,
        context_kind=context_kind,
        confidence=confidence,
        policy=policy,
        graph_node=graph_node,
        grounding_result=grounding_result,
        scene_kind=scene_kind,
        pattern_kind=pattern_kind,
        text_selection=text_selection,
        trust=TrustCalibration(
            confidence=confidence,
            stability=max(0.0, min(1.0, confidence + 0.05)),
            ambiguity=1.0 - confidence,
            source=EnvironmentSource.OS_OBSERVER,
            reason=f"resolved target: {label}",
        ),
        metadata=metadata or {},
    )


def _hint_kind(
    *,
    intent: EnvironmentIntent,
    target: ResolvedUITarget,
) -> PlannerHintKind:
    if intent.kind == UIReasoningIntentKind.FIX:
        return PlannerHintKind.INSPECT_ERROR

    if intent.kind == UIReasoningIntentKind.OPEN:
        return PlannerHintKind.OPEN_FILE

    if intent.kind == UIReasoningIntentKind.RUN:
        return PlannerHintKind.RUN_COMMAND

    if intent.kind == UIReasoningIntentKind.COPY:
        return PlannerHintKind.COPY_SELECTION

    if intent.kind == UIReasoningIntentKind.FOCUS:
        return PlannerHintKind.FOCUS_TARGET

    if target.context_kind == UIReasoningContextKind.DIALOG_CONTEXT:
        return PlannerHintKind.VERIFY_DIALOG

    return PlannerHintKind.ASK_CLARIFICATION


def _context_kind_from_chain(chain: UIContextChain) -> UIReasoningContextKind:
    if chain.ui_context is not None:
        scene = chain.ui_context.scene.kind

        if scene == SemanticSceneKind.CODE_SESSION:
            return UIReasoningContextKind.CODE_CONTEXT

        if scene == SemanticSceneKind.TERMINAL_RUNNING:
            return UIReasoningContextKind.TERMINAL_CONTEXT

        if scene == SemanticSceneKind.BROWSER_RESEARCH:
            return UIReasoningContextKind.BROWSER_CONTEXT

        if scene in {
            SemanticSceneKind.ERROR_DIALOG,
            SemanticSceneKind.CONFIRMATION_DIALOG,
        }:
            return UIReasoningContextKind.DIALOG_CONTEXT

        if scene == SemanticSceneKind.APP_LOADING:
            return UIReasoningContextKind.LOADING_CONTEXT

    if chain.app_kind == DetectedAppKind.IDE:
        return UIReasoningContextKind.CODE_CONTEXT

    if chain.app_kind == DetectedAppKind.TERMINAL:
        return UIReasoningContextKind.TERMINAL_CONTEXT

    if chain.app_kind == DetectedAppKind.BROWSER:
        return UIReasoningContextKind.BROWSER_CONTEXT

    return UIReasoningContextKind.UNKNOWN_CONTEXT


def _context_kind_from_node(node: GraphNode) -> UIReasoningContextKind:
    if node.kind in {GraphNodeKind.ERROR, GraphNodeKind.DIALOG}:
        return UIReasoningContextKind.ERROR_CONTEXT

    if node.kind in {GraphNodeKind.TERMINAL, GraphNodeKind.COMMAND}:
        return UIReasoningContextKind.TERMINAL_CONTEXT

    if node.kind in {GraphNodeKind.FILE, GraphNodeKind.EDITOR, GraphNodeKind.PROJECT}:
        return UIReasoningContextKind.CODE_CONTEXT

    return UIReasoningContextKind.UNKNOWN_CONTEXT


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _norm(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _clean_required(value: str) -> str:
    cleaned = value.strip()

    if not cleaned:
        raise ValueError("field cannot be empty.")

    return cleaned