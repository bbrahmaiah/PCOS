from __future__ import annotations

from enum import StrEnum
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.environment.app_identity import AppStateModel, DetectedAppKind
from jarvis.environment.ground_truth import DivergenceReport, GroundTruthStatus
from jarvis.environment.models import (
    EnvironmentSource,
    EnvironmentState,
    TemporalWorkspaceState,
    TrustCalibration,
)
from jarvis.environment.trust_runtime import TrustPolicyClassification
from jarvis.environment.ui_reasoning import (
    UIReasoningIntentKind,
    UIReasoningResult,
)
from jarvis.environment.ui_semantics import SemanticSceneKind, UIContext
from jarvis.environment.visual_grounding import VisualGroundingResult
from jarvis.environment.workspace_graph import (
    GraphNodeKind,
    WorkspaceCognitiveGraph,
)
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class FusionInputSource(StrEnum):
    VOICE = "voice"
    CONVERSATION = "conversation"
    ENVIRONMENT_STATE = "environment_state"
    WORKSPACE_GRAPH = "workspace_graph"
    SEMANTIC_SCENE = "semantic_scene"
    TIMELINE = "timeline"
    MEMORY = "memory"
    ACTIVE_INTENT = "active_intent"
    GROUNDING = "grounding"
    APP_STATE = "app_state"
    GROUND_TRUTH = "ground_truth"


class FusionMode(StrEnum):
    PASSIVE_AWARENESS = "passive_awareness"
    SCREEN_AWARE_CONVERSATION = "screen_aware_conversation"
    TASK_CONTINUITY = "task_continuity"
    DEBUGGING_ASSISTANCE = "debugging_assistance"
    ACTION_PREPARATION = "action_preparation"
    RECOVERY_CONTEXT = "recovery_context"


class FusionStatus(StrEnum):
    FUSED = "fused"
    PARTIAL = "partial"
    DEGRADED = "degraded"
    BLOCKED = "blocked"
    FAILED = "failed"


class FusionDecision(StrEnum):
    USE_FOR_COGNITION = "use_for_cognition"
    USE_WITH_VERIFICATION = "use_with_verification"
    ASK_USER = "ask_user"
    REFRESH_ENVIRONMENT = "refresh_environment"
    BLOCK_COGNITION = "block_cognition"


class FusionReason(StrEnum):
    SESSION_CREATED = "session_created"
    CONTEXT_FUSED = "context_fused"
    PARTIAL_CONTEXT_FUSED = "partial_context_fused"
    FUSION_DEGRADED = "fusion_degraded"
    FUSION_BLOCKED = "fusion_blocked"
    SESSION_NOT_FOUND = "session_not_found"
    RUNTIME_RESET = "runtime_reset"


class FusionEventKind(StrEnum):
    SESSION_CREATED = "session_created"
    FUSION_COMPLETED = "fusion_completed"
    FUSION_BLOCKED = "fusion_blocked"
    RUNTIME_RESET = "runtime_reset"


class VoiceInputFrame(OrchestrationModel):
    """
    Current voice/user input frame.

    This is the conversational half of cognition.
    """

    frame_id: str = Field(default_factory=lambda: f"voice_frame_{uuid4().hex}")
    text: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    is_final: bool = True
    speaker: str = "user"
    captured_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("frame_id", "text", "speaker")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class ConversationStateFrame(OrchestrationModel):
    """
    Conversation state frame.

    This carries active topic, last assistant response, turn count,
    and whether the user is currently in a task/workflow.
    """

    frame_id: str = Field(default_factory=lambda: f"conversation_frame_{uuid4().hex}")
    active_topic: str | None = None
    last_user_text: str | None = None
    last_assistant_summary: str | None = None
    turn_count: int = Field(default=0, ge=0)
    conversation_active: bool = True
    user_interrupted: bool = False
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("frame_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class MemoryContextFrame(OrchestrationModel):
    """
    Memory context frame.

    This must contain memory summaries/reasons only, not raw uncontrolled memory.
    """

    frame_id: str = Field(default_factory=lambda: f"memory_frame_{uuid4().hex}")
    summaries: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    confidence: float = Field(default=0.80, ge=0.0, le=1.0)
    policy: TrustPolicyClassification = TrustPolicyClassification.SAFE
    retrieved_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("frame_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class ActiveIntentFrame(OrchestrationModel):
    """
    Active intent frame.

    Carries current user goal/subgoal without executing anything.
    """

    frame_id: str = Field(default_factory=lambda: f"active_intent_{uuid4().hex}")
    goal: str | None = None
    subgoal: str | None = None
    intent_kind: UIReasoningIntentKind | None = None
    blocked: bool = False
    paused: bool = False
    confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("frame_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class DualInputStream(OrchestrationModel):
    """
    Dual input stream.

    From Phase 8 onward, cognition receives both:
    - voice/conversation stream
    - environment stream
    """

    stream_id: str = Field(default_factory=lambda: f"dual_stream_{uuid4().hex}")
    voice: VoiceInputFrame
    conversation: ConversationStateFrame | None = None
    environment_state: EnvironmentState | None = None
    app_state: AppStateModel | None = None
    workspace_graph: WorkspaceCognitiveGraph | None = None
    semantic_context: UIContext | None = None
    temporal_state: TemporalWorkspaceState | None = None
    memory_context: MemoryContextFrame | None = None
    active_intent: ActiveIntentFrame | None = None
    grounding_result: VisualGroundingResult | None = None
    reasoning_result: UIReasoningResult | None = None
    divergence_report: DivergenceReport | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("stream_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class VisualContextInjection(OrchestrationModel):
    """
    Visual context selected for cognition prompt/context assembly.

    This is a controlled injection, not raw screen dump.
    """

    injection_id: str = Field(default_factory=lambda: f"visual_injection_{uuid4().hex}")
    scene: SemanticSceneKind | None = None
    focused_app_kind: DetectedAppKind | None = None
    focused_node_labels: tuple[str, ...] = ()
    visible_error_labels: tuple[str, ...] = ()
    selected_target_label: str | None = None
    graph_summary: str | None = None
    timeline_summary: str | None = None
    trust: TrustCalibration
    policy: TrustPolicyClassification
    created_at: object = Field(default_factory=utc_now)

    @field_validator("injection_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class ConversationEnrichment(OrchestrationModel):
    """
    Conversation enrichment output.

    This tells cognition what the user likely means in context.
    """

    enrichment_id: str = Field(
        default_factory=lambda: f"conversation_enrich_{uuid4().hex}"
    )
    original_text: str
    enriched_text: str
    inferred_references: tuple[str, ...] = ()
    missing_context: tuple[str, ...] = ()
    confidence: float = Field(ge=0.0, le=1.0)
    created_at: object = Field(default_factory=utc_now)

    @field_validator("enrichment_id", "original_text", "enriched_text")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class ScreenAwareCognitionBridge(OrchestrationModel):
    """
    Bridge object handed to cognition.

    It is screen-aware, but still policy controlled.
    """

    bridge_id: str = Field(default_factory=lambda: f"screen_bridge_{uuid4().hex}")
    fused_summary: str
    cognition_instructions: tuple[str, ...]
    allowed_context_sources: tuple[FusionInputSource, ...]
    blocked_context_sources: tuple[FusionInputSource, ...] = ()
    require_verification_before_action: bool = True
    safe_for_response_generation: bool
    safe_for_action_planning: bool
    created_at: object = Field(default_factory=utc_now)

    @field_validator("bridge_id", "fused_summary")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class FusedContext(OrchestrationModel):
    """
    Final fused context for screen-aware cognition.

    This is the main artifact of Step 19.
    """

    context_id: str = Field(default_factory=lambda: f"fused_context_{uuid4().hex}")
    status: FusionStatus
    reason: FusionReason
    decision: FusionDecision
    mode: FusionMode
    stream: DualInputStream
    visual_injection: VisualContextInjection
    enrichment: ConversationEnrichment
    bridge: ScreenAwareCognitionBridge
    trust: TrustCalibration
    policy: TrustPolicyClassification
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("context_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _blocked_cannot_be_safe(self) -> FusedContext:
        if self.status == FusionStatus.BLOCKED:
            if self.bridge.safe_for_action_planning:
                raise ValueError("blocked fused context cannot be action-plannable.")

        return self


class EnvironmentFusionSession(OrchestrationModel):
    """
    Environment fusion runtime session.
    """

    session_id: str = Field(default_factory=lambda: f"env_fusion_{uuid4().hex}")
    workspace_id: str
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class EnvironmentFusionRuntimeEvent(OrchestrationModel):
    """
    Environment fusion runtime event.
    """

    event_id: str = Field(default_factory=lambda: f"env_fusion_event_{uuid4().hex}")
    kind: FusionEventKind
    reason: FusionReason
    session_id: str | None = None
    context_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class EnvironmentFusionRuntimeSnapshot(OrchestrationModel):
    """
    Diagnostics for Phase 8 Step 19.
    """

    name: str
    session_count: int = Field(ge=0)
    fused_context_count: int = Field(ge=0)
    full_fusion_count: int = Field(ge=0)
    partial_count: int = Field(ge=0)
    degraded_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    action_plannable_count: int = Field(ge=0)
    runtime_event_count: int = Field(ge=0)
    last_reason: FusionReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        return _clean_required(value)


class EnvironmentFusionPolicy:
    """
    Fusion policy.

    Protects cognition from stale, divergent, blocked, or unsafe environment
    input.
    """

    def evaluate(
        self, 
        stream: DualInputStream
        ) -> tuple[FusionStatus, FusionDecision, TrustPolicyClassification]:
        if stream.divergence_report is not None:
            if stream.divergence_report.status == GroundTruthStatus.DIVERGED:
                return (
                    FusionStatus.DEGRADED,
                    FusionDecision.REFRESH_ENVIRONMENT,
                    TrustPolicyClassification.REVIEW,
                )

        if stream.semantic_context is not None:
            if not stream.semantic_context.safe_for_reasoning:
                return (
                    FusionStatus.BLOCKED,
                    FusionDecision.BLOCK_COGNITION,
                    TrustPolicyClassification.BLOCKED,
                )

            if not stream.semantic_context.safe_for_action:
                return (
                    FusionStatus.PARTIAL,
                    FusionDecision.USE_WITH_VERIFICATION,
                    stream.semantic_context.policy_classification,
                )

        if stream.reasoning_result is not None:
            if not stream.reasoning_result.safe_for_action_planning:
                return (
                    FusionStatus.PARTIAL,
                    FusionDecision.USE_WITH_VERIFICATION,
                    TrustPolicyClassification.VERIFY_FIRST,
                )

        if stream.workspace_graph is None and stream.environment_state is None:
            return (
                FusionStatus.PARTIAL,
                FusionDecision.ASK_USER,
                TrustPolicyClassification.REVIEW,
            )

        return (
            FusionStatus.FUSED,
            FusionDecision.USE_FOR_COGNITION,
            TrustPolicyClassification.SAFE,
        )


class EnvironmentContextSummarizer:
    """
    Summarizes environment without dumping raw screen data.
    """

    def visual_injection_for(
        self,
        stream: DualInputStream,
        *,
        policy: TrustPolicyClassification,
    ) -> VisualContextInjection:
        scene = (
            stream.semantic_context.scene.kind
            if stream.semantic_context is not None
            else None
        )
        focused_app_kind = (
            stream.app_state.identity.classification.kind
            if stream.app_state is not None
            else None
        )
        focused_node_labels = _focused_node_labels(stream.workspace_graph)
        visible_error_labels = _node_labels(stream.workspace_graph, GraphNodeKind.ERROR)
        selected_target_label = (
            stream.grounding_result.selected.label
            if stream.grounding_result is not None
            and stream.grounding_result.selected is not None
            else None
        )

        return VisualContextInjection(
            scene=scene,
            focused_app_kind=focused_app_kind,
            focused_node_labels=focused_node_labels,
            visible_error_labels=visible_error_labels,
            selected_target_label=selected_target_label,
            graph_summary=_graph_summary(stream.workspace_graph),
            timeline_summary=_timeline_summary(stream.temporal_state),
            trust=TrustCalibration(
                confidence=_fusion_confidence(stream),
                stability=0.86,
                ambiguity=1.0 - _fusion_confidence(stream),
                source=_environment_source(),
                reason="environment visual context injection",
            ),
            policy=policy,
        )


class ConversationEnricher:
    """
    Enriches voice text with environment references.
    """

    def enrich(
        self,
        *,
        stream: DualInputStream,
        visual: VisualContextInjection,
    ) -> ConversationEnrichment:
        text = stream.voice.text
        references: list[str] = []
        missing: list[str] = []

        if _has_deictic_reference(text):
            if visual.selected_target_label:
                references.append(f"that={visual.selected_target_label}")
            elif visual.visible_error_labels:
                references.append(f"that={visual.visible_error_labels[0]}")
            elif visual.focused_node_labels:
                references.append(f"that={visual.focused_node_labels[0]}")
            else:
                missing.append("deictic reference target")

        if visual.scene is not None:
            references.append(f"scene={visual.scene.value}")

        if visual.focused_app_kind is not None:
            references.append(f"app={visual.focused_app_kind.value}")

        if stream.active_intent is not None and stream.active_intent.goal:
            references.append(f"active_goal={stream.active_intent.goal}")

        enriched = text

        if references:
            enriched = f"{text} | context: {', '.join(references)}"

        confidence = 0.90 if references else 0.65

        if missing:
            confidence = 0.45

        return ConversationEnrichment(
            original_text=text,
            enriched_text=enriched,
            inferred_references=tuple(references),
            missing_context=tuple(missing),
            confidence=confidence,
        )


class CognitionBridgeBuilder:
    """
    Builds the controlled bridge handed to cognition.
    """

    def build(
        self,
        *,
        stream: DualInputStream,
        visual: VisualContextInjection,
        enrichment: ConversationEnrichment,
        status: FusionStatus,
        decision: FusionDecision,
        policy: TrustPolicyClassification,
    ) -> ScreenAwareCognitionBridge:
        sources = _allowed_sources(stream)
        blocked = _blocked_sources(stream, policy)
        require_verification = decision in {
            FusionDecision.USE_WITH_VERIFICATION,
            FusionDecision.REFRESH_ENVIRONMENT,
            FusionDecision.ASK_USER,
            FusionDecision.BLOCK_COGNITION,
        }
        safe_response = status != FusionStatus.BLOCKED
        safe_action = (
            status == FusionStatus.FUSED
            and decision == FusionDecision.USE_FOR_COGNITION
            and policy == TrustPolicyClassification.SAFE
        )

        return ScreenAwareCognitionBridge(
            fused_summary=_fused_summary(
                stream=stream,
                visual=visual,
                enrichment=enrichment,
                status=status,
            ),
            cognition_instructions=_instructions_for(
                status=status,
                decision=decision,
                policy=policy,
            ),
            allowed_context_sources=sources,
            blocked_context_sources=blocked,
            require_verification_before_action=require_verification,
            safe_for_response_generation=safe_response,
            safe_for_action_planning=safe_action,
        )


class EnvironmentFusionRuntime:
    """
    Phase 8 Step 19 Environment-Conversation Fusion Runtime.

    Responsibilities:
    - fuse voice/conversation/environment into FusedContext
    - enrich conversation with screen/environment meaning
    - prepare screen-aware cognition bridge
    - prevent stale/divergent/unsafe environment from silently entering cognition

    Non-responsibilities:
    - no direct LLM call
    - no memory direct retrieval
    - no action execution
    - no screen capture
    """

    def __init__(
        self,
        *,
        name: str = "environment_fusion_runtime",
        policy: EnvironmentFusionPolicy | None = None,
        summarizer: EnvironmentContextSummarizer | None = None,
        enricher: ConversationEnricher | None = None,
        bridge_builder: CognitionBridgeBuilder | None = None,
    ) -> None:
        cleaned = name.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._policy = policy or EnvironmentFusionPolicy()
        self._summarizer = summarizer or EnvironmentContextSummarizer()
        self._enricher = enricher or ConversationEnricher()
        self._bridge_builder = bridge_builder or CognitionBridgeBuilder()
        self._sessions: dict[str, EnvironmentFusionSession] = {}
        self._contexts: list[FusedContext] = []
        self._events: list[EnvironmentFusionRuntimeEvent] = []
        self._lock = RLock()
        self._last_reason: FusionReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def create_session(
        self,
        *,
        workspace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> EnvironmentFusionSession:
        session = EnvironmentFusionSession(
            workspace_id=workspace_id,
            metadata=metadata or {},
        )
        event = self._event(
            kind=FusionEventKind.SESSION_CREATED,
            reason=FusionReason.SESSION_CREATED,
            session_id=session.session_id,
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._events.append(event)
            self._last_reason = event.reason

        return session

    def fuse(
        self,
        *,
        session_id: str,
        stream: DualInputStream,
        mode: FusionMode | None = None,
    ) -> FusedContext:
        if self.session_for(session_id) is None:
            context = self._failed_context(stream)
            self._record_context(context, session_id=session_id)
            return context

        status, decision, policy = self._policy.evaluate(stream)
        visual = self._summarizer.visual_injection_for(stream, policy=policy)
        enrichment = self._enricher.enrich(stream=stream, visual=visual)
        bridge = self._bridge_builder.build(
            stream=stream,
            visual=visual,
            enrichment=enrichment,
            status=status,
            decision=decision,
            policy=policy,
        )
        trust = TrustCalibration(
            confidence=min(visual.trust.confidence, enrichment.confidence),
            stability=visual.trust.stability,
            ambiguity=max(visual.trust.ambiguity, 1.0 - enrichment.confidence),
            source=_environment_source(),
            reason="environment conversation fusion",
        )
        context = FusedContext(
            status=status,
            reason=_reason_for(status=status),
            decision=decision,
            mode=mode or _mode_for(stream),
            stream=stream,
            visual_injection=visual,
            enrichment=enrichment,
            bridge=bridge,
            trust=trust,
            policy=policy,
            message=_message_for(status=status, decision=decision),
        )

        self._record_context(context, session_id=session_id)
        self._touch_session(session_id)

        return context

    def session_for(self, session_id: str) -> EnvironmentFusionSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def contexts(self) -> tuple[FusedContext, ...]:
        with self._lock:
            return tuple(self._contexts)

    def events(self) -> tuple[EnvironmentFusionRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def snapshot(self) -> EnvironmentFusionRuntimeSnapshot:
        with self._lock:
            return EnvironmentFusionRuntimeSnapshot(
                name=self.name,
                session_count=len(self._sessions),
                fused_context_count=len(self._contexts),
                full_fusion_count=sum(
                    1
                    for context in self._contexts
                    if context.status == FusionStatus.FUSED
                ),
                partial_count=sum(
                    1
                    for context in self._contexts
                    if context.status == FusionStatus.PARTIAL
                ),
                degraded_count=sum(
                    1
                    for context in self._contexts
                    if context.status == FusionStatus.DEGRADED
                ),
                blocked_count=sum(
                    1
                    for context in self._contexts
                    if context.status == FusionStatus.BLOCKED
                ),
                action_plannable_count=sum(
                    1
                    for context in self._contexts
                    if context.bridge.safe_for_action_planning
                ),
                runtime_event_count=len(self._events),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        event = self._event(
            kind=FusionEventKind.RUNTIME_RESET,
            reason=FusionReason.RUNTIME_RESET,
        )

        with self._lock:
            self._sessions.clear()
            self._contexts.clear()
            self._events.clear()
            self._events.append(event)
            self._last_reason = event.reason

    def _failed_context(self, stream: DualInputStream) -> FusedContext:
        visual = VisualContextInjection(
            trust=TrustCalibration(
                confidence=0.0,
                stability=0.0,
                ambiguity=1.0,
                source=_environment_source(),
                reason="fusion session not found",
            ),
            policy=TrustPolicyClassification.BLOCKED,
        )
        enrichment = ConversationEnrichment(
            original_text=stream.voice.text,
            enriched_text=stream.voice.text,
            confidence=0.0,
            missing_context=("fusion session",),
        )
        bridge = ScreenAwareCognitionBridge(
            fused_summary="environment fusion session not found",
            cognition_instructions=("Do not use environment context.",),
            allowed_context_sources=(FusionInputSource.VOICE,),
            blocked_context_sources=(
                FusionInputSource.ENVIRONMENT_STATE,
                FusionInputSource.WORKSPACE_GRAPH,
                FusionInputSource.SEMANTIC_SCENE,
            ),
            require_verification_before_action=True,
            safe_for_response_generation=False,
            safe_for_action_planning=False,
        )

        return FusedContext(
            status=FusionStatus.FAILED,
            reason=FusionReason.SESSION_NOT_FOUND,
            decision=FusionDecision.BLOCK_COGNITION,
            mode=FusionMode.PASSIVE_AWARENESS,
            stream=stream,
            visual_injection=visual,
            enrichment=enrichment,
            bridge=bridge,
            trust=visual.trust,
            policy=TrustPolicyClassification.BLOCKED,
            message="environment fusion session not found",
        )

    def _record_context(
        self,
        context: FusedContext,
        *,
        session_id: str,
    ) -> None:
        event = self._event(
            kind=(
                FusionEventKind.FUSION_BLOCKED
                if context.status in {FusionStatus.BLOCKED, FusionStatus.FAILED}
                else FusionEventKind.FUSION_COMPLETED
            ),
            reason=context.reason,
            session_id=session_id,
            context_id=context.context_id,
            metadata={
                "status": context.status.value,
                "decision": context.decision.value,
            },
        )

        with self._lock:
            self._contexts.append(context)
            self._events.append(event)
            self._last_reason = context.reason

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
        kind: FusionEventKind,
        reason: FusionReason,
        session_id: str | None = None,
        context_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EnvironmentFusionRuntimeEvent:
        return EnvironmentFusionRuntimeEvent(
            kind=kind,
            reason=reason,
            session_id=session_id,
            context_id=context_id,
            metadata=metadata or {},
        )


def _allowed_sources(stream: DualInputStream) -> tuple[FusionInputSource, ...]:
    sources = [FusionInputSource.VOICE]

    if stream.conversation is not None:
        sources.append(FusionInputSource.CONVERSATION)

    if stream.environment_state is not None:
        sources.append(FusionInputSource.ENVIRONMENT_STATE)

    if stream.app_state is not None:
        sources.append(FusionInputSource.APP_STATE)

    if stream.workspace_graph is not None:
        sources.append(FusionInputSource.WORKSPACE_GRAPH)

    if stream.semantic_context is not None:
        sources.append(FusionInputSource.SEMANTIC_SCENE)

    if stream.temporal_state is not None:
        sources.append(FusionInputSource.TIMELINE)

    if stream.memory_context is not None:
        sources.append(FusionInputSource.MEMORY)

    if stream.active_intent is not None:
        sources.append(FusionInputSource.ACTIVE_INTENT)

    if stream.grounding_result is not None:
        sources.append(FusionInputSource.GROUNDING)

    if stream.divergence_report is not None:
        sources.append(FusionInputSource.GROUND_TRUTH)

    return tuple(sources)


def _blocked_sources(
    stream: DualInputStream,
    policy: TrustPolicyClassification,
) -> tuple[FusionInputSource, ...]:
    blocked: list[FusionInputSource] = []

    if policy == TrustPolicyClassification.BLOCKED:
        blocked.extend(
            (
                FusionInputSource.ENVIRONMENT_STATE,
                FusionInputSource.WORKSPACE_GRAPH,
                FusionInputSource.SEMANTIC_SCENE,
                FusionInputSource.GROUNDING,
            )
        )

    if stream.divergence_report is not None:
        if stream.divergence_report.status == GroundTruthStatus.DIVERGED:
            blocked.append(FusionInputSource.GROUNDING)

    return tuple(dict.fromkeys(blocked))


def _focused_node_labels(
    graph: WorkspaceCognitiveGraph | None,
) -> tuple[str, ...]:
    if graph is None or graph.focused_node_id is None:
        return ()

    node = graph.nodes.get(graph.focused_node_id)

    if node is None:
        return ()

    return (node.label,)


def _node_labels(
    graph: WorkspaceCognitiveGraph | None,
    kind: GraphNodeKind,
) -> tuple[str, ...]:
    if graph is None:
        return ()

    return tuple(
        node.label
        for node in graph.nodes.values()
        if node.active and node.kind == kind
    )


def _graph_summary(graph: WorkspaceCognitiveGraph | None) -> str | None:
    if graph is None:
        return None

    app_count = sum(
        1 for node in graph.nodes.values() if node.kind == GraphNodeKind.APP
    )
    error_count = sum(
        1 for node in graph.nodes.values() if node.kind == GraphNodeKind.ERROR
    )

    return (
        f"workspace graph: nodes={len(graph.nodes)}, "
        f"edges={len(graph.edges)}, apps={app_count}, errors={error_count}"
    )


def _timeline_summary(state: TemporalWorkspaceState | None) -> str | None:
    if state is None:
        return None

    return "temporal workspace state available"


def _fusion_confidence(stream: DualInputStream) -> float:
    score = stream.voice.confidence

    if stream.semantic_context is not None:
        score = min(score, stream.semantic_context.scene.confidence)

    if stream.grounding_result is not None and stream.grounding_result.selected:
        score = min(score, stream.grounding_result.selected.confidence)

    if stream.divergence_report is not None:
        if stream.divergence_report.status == GroundTruthStatus.DIVERGED:
            score = min(score, 0.55)

    return max(0.0, min(1.0, score))


def _has_deictic_reference(text: str) -> bool:
    normalized = " ".join(text.lower().strip().split())

    return any(
        word in normalized.split()
        for word in ("that", "it", "this", "there", "current", "selected")
    )


def _fused_summary(
    *,
    stream: DualInputStream,
    visual: VisualContextInjection,
    enrichment: ConversationEnrichment,
    status: FusionStatus,
) -> str:
    parts = [
        f"user said: {stream.voice.text}",
        f"fusion status: {status.value}",
    ]

    if visual.focused_app_kind is not None:
        parts.append(f"focused app: {visual.focused_app_kind.value}")

    if visual.scene is not None:
        parts.append(f"scene: {visual.scene.value}")

    if visual.visible_error_labels:
        parts.append(f"visible errors: {', '.join(visual.visible_error_labels)}")

    if stream.memory_context is not None and stream.memory_context.summaries:
        parts.append(f"memory: {stream.memory_context.summaries[0]}")

    if enrichment.inferred_references:
        parts.append(f"references: {', '.join(enrichment.inferred_references)}")

    return " | ".join(parts)


def _instructions_for(
    *,
    status: FusionStatus,
    decision: FusionDecision,
    policy: TrustPolicyClassification,
) -> tuple[str, ...]:
    instructions = [
        "Use both conversation and environment context.",
        "Do not assume raw screen content beyond fused context.",
        "Do not execute actions directly.",
    ]

    if decision != FusionDecision.USE_FOR_COGNITION:
        instructions.append("Verify or clarify before action planning.")

    if status == FusionStatus.DEGRADED:
        instructions.append("Environment may be stale; refresh before acting.")

    if policy == TrustPolicyClassification.BLOCKED:
        instructions.append("Environment context is blocked by policy.")

    return tuple(instructions)


def _reason_for(status: FusionStatus) -> FusionReason:
    if status == FusionStatus.FUSED:
        return FusionReason.CONTEXT_FUSED

    if status == FusionStatus.PARTIAL:
        return FusionReason.PARTIAL_CONTEXT_FUSED

    if status == FusionStatus.DEGRADED:
        return FusionReason.FUSION_DEGRADED

    if status == FusionStatus.BLOCKED:
        return FusionReason.FUSION_BLOCKED

    return FusionReason.FUSION_BLOCKED


def _mode_for(stream: DualInputStream) -> FusionMode:
    if stream.divergence_report is not None:
        if stream.divergence_report.status == GroundTruthStatus.DIVERGED:
            return FusionMode.RECOVERY_CONTEXT

    if stream.reasoning_result is not None:
        if stream.reasoning_result.intent.kind == UIReasoningIntentKind.FIX:
            return FusionMode.DEBUGGING_ASSISTANCE

    if stream.active_intent is not None:
        return FusionMode.TASK_CONTINUITY

    if stream.workspace_graph is not None or stream.semantic_context is not None:
        return FusionMode.SCREEN_AWARE_CONVERSATION

    return FusionMode.PASSIVE_AWARENESS


def _message_for(
    *,
    status: FusionStatus,
    decision: FusionDecision,
) -> str:
    if status == FusionStatus.FUSED:
        return "conversation and environment fused for cognition"

    if status == FusionStatus.PARTIAL:
        return "partial environment fusion; verification required"

    if status == FusionStatus.DEGRADED:
        return "environment fusion degraded; refresh required"

    if status == FusionStatus.BLOCKED:
        return "environment fusion blocked by policy"

    return f"fusion result: {decision.value}"


def _environment_source() -> EnvironmentSource:
    return EnvironmentSource.OS_OBSERVER


def _clean_required(value: str) -> str:
    cleaned = value.strip()

    if not cleaned:
        raise ValueError("field cannot be empty.")

    return cleaned