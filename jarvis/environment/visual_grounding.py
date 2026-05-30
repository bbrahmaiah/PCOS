from __future__ import annotations

from difflib import SequenceMatcher
from enum import StrEnum
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.environment.models import EnvironmentSource, ScreenRegion, TrustCalibration
from jarvis.environment.ocr import OCRTextRegion
from jarvis.environment.trust_runtime import TrustPolicyClassification
from jarvis.environment.ui_detection import DetectedUIElement, DetectedUIElementKind
from jarvis.environment.ui_patterns import UIPatternKind, UIPatternRecognitionResult
from jarvis.environment.ui_semantics import SemanticSceneKind, UIContext
from jarvis.environment.workspace_graph import (
    GraphNode,
    GraphNodeKind,
    WorkspaceCognitiveGraph,
)
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class GroundingIntentKind(StrEnum):
    CLICK = "click"
    FOCUS = "focus"
    INSPECT = "inspect"
    OPEN = "open"
    READ = "read"
    SELECT = "select"
    TYPE_INTO = "type_into"
    UNKNOWN = "unknown"


class GroundingStrategy(StrEnum):
    WORKSPACE_GRAPH_QUERY = "workspace_graph_query"
    ACCESSIBILITY_EXACT_MATCH = "accessibility_exact_match"
    OCR_EXACT_MATCH = "ocr_exact_match"
    FUZZY_TEXT_MATCH = "fuzzy_text_match"
    SEMANTIC_ICON_MATCH = "semantic_icon_match"
    SPATIAL_REFERENCE = "spatial_reference"
    CONTEXT_INFERENCE = "context_inference"


class GroundingTargetKind(StrEnum):
    GRAPH_NODE = "graph_node"
    UI_ELEMENT = "ui_element"
    OCR_TEXT_REGION = "ocr_text_region"
    SEMANTIC_SCENE = "semantic_scene"
    PATTERN = "pattern"
    UNKNOWN = "unknown"


class GroundingDecision(StrEnum):
    GROUNDED = "grounded"
    ASK_USER = "ask_user"
    VERIFY_FIRST = "verify_first"
    BLOCKED = "blocked"
    NOT_FOUND = "not_found"


class GroundingStatus(StrEnum):
    GROUNDED = "grounded"
    AMBIGUOUS = "ambiguous"
    LOW_CONFIDENCE = "low_confidence"
    BLOCKED = "blocked"
    NOT_FOUND = "not_found"
    FAILED = "failed"


class GroundingReason(StrEnum):
    SESSION_CREATED = "session_created"
    TARGET_GROUNDED = "target_grounded"
    AMBIGUITY_DETECTED = "ambiguity_detected"
    LOW_CONFIDENCE_TARGET = "low_confidence_target"
    TARGET_BLOCKED_BY_POLICY = "target_blocked_by_policy"
    TARGET_NOT_FOUND = "target_not_found"
    SESSION_NOT_FOUND = "session_not_found"
    RUNTIME_RESET = "runtime_reset"


class GroundingEventKind(StrEnum):
    SESSION_CREATED = "session_created"
    GROUNDING_COMPLETED = "grounding_completed"
    GROUNDING_BLOCKED = "grounding_blocked"
    RUNTIME_RESET = "runtime_reset"


class GroundingEvidence(OrchestrationModel):
    """
    Evidence explaining why a target was selected.
    """

    evidence_id: str = Field(default_factory=lambda: f"grounding_evd_{uuid4().hex}")
    strategy: GroundingStrategy
    description: str
    score: float = Field(ge=0.0, le=1.0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("evidence_id", "description")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class TargetTrustPolicy(OrchestrationModel):
    """
    Trust policy for grounding.

    Low confidence means ask. Ambiguity means ask. Blocked policy means block.
    """

    minimum_grounding_confidence: float = Field(default=0.72, ge=0.0, le=1.0)
    verify_first_confidence: float = Field(default=0.82, ge=0.0, le=1.0)
    ambiguity_margin: float = Field(default=0.08, ge=0.0, le=1.0)
    require_trusted_interactive_target: bool = True

    @model_validator(mode="after")
    def _threshold_order(self) -> TargetTrustPolicy:
        if self.verify_first_confidence < self.minimum_grounding_confidence:
            raise ValueError(
                "verify_first_confidence must be >= minimum_grounding_confidence."
            )

        return self


class GroundingCandidate(OrchestrationModel):
    """
    One possible grounded target.

    This is evidence only. It is not an action.
    """

    candidate_id: str = Field(default_factory=lambda: f"grounding_cand_{uuid4().hex}")
    target_kind: GroundingTargetKind
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    strategy: GroundingStrategy
    policy: TrustPolicyClassification
    trust: TrustCalibration
    region: ScreenRegion | None = None
    graph_node: GraphNode | None = None
    ui_element: DetectedUIElement | None = None
    text_region: OCRTextRegion | None = None
    scene_kind: SemanticSceneKind | None = None
    pattern_kind: UIPatternKind | None = None
    evidence: tuple[GroundingEvidence, ...] = ()
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("candidate_id", "label")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class GroundingQuery(OrchestrationModel):
    """
    Natural language grounding query.

    Example:
    - click Run
    - focus terminal
    - inspect the error
    - open the file picker
    """

    query_id: str = Field(default_factory=lambda: f"grounding_query_{uuid4().hex}")
    session_id: str
    text: str
    intent: GroundingIntentKind = GroundingIntentKind.UNKNOWN
    workspace_graph: WorkspaceCognitiveGraph | None = None
    ui_elements: tuple[DetectedUIElement, ...] = ()
    text_regions: tuple[OCRTextRegion, ...] = ()
    ui_context: UIContext | None = None
    pattern_result: UIPatternRecognitionResult | None = None
    active_region: ScreenRegion | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("query_id", "session_id", "text")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class AmbiguityResolution(OrchestrationModel):
    """
    Ambiguity analysis for candidate targets.
    """

    resolution_id: str = Field(default_factory=lambda: f"ambiguity_{uuid4().hex}")
    ambiguous: bool
    top_candidates: tuple[GroundingCandidate, ...]
    reason: str
    ask_user_message: str | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("resolution_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class VisualGroundingResult(OrchestrationModel):
    """
    Final visual grounding result.

    This result may be consumed by later planning/action layers, but it does
    not execute anything.
    """

    result_id: str = Field(default_factory=lambda: f"grounding_result_{uuid4().hex}")
    status: GroundingStatus
    reason: GroundingReason
    decision: GroundingDecision
    query: GroundingQuery
    candidates: tuple[GroundingCandidate, ...] = ()
    selected: GroundingCandidate | None = None
    ambiguity: AmbiguityResolution
    safe_for_action_planning: bool
    message: str
    created_at: object = Field(default_factory=utc_now)

    @field_validator("result_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _selected_must_be_in_candidates(self) -> VisualGroundingResult:
        if self.selected is None:
            return self

        candidate_ids = {candidate.candidate_id for candidate in self.candidates}

        if self.selected.candidate_id not in candidate_ids:
            raise ValueError("selected candidate must be present in candidates.")

        return self


class VisualGroundingSession(OrchestrationModel):
    """
    Visual grounding runtime session.
    """

    session_id: str = Field(default_factory=lambda: f"visual_grounding_{uuid4().hex}")
    workspace_id: str
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class VisualGroundingRuntimeEvent(OrchestrationModel):
    """
    Visual grounding runtime event.
    """

    event_id: str = Field(default_factory=lambda: f"grounding_event_{uuid4().hex}")
    kind: GroundingEventKind
    reason: GroundingReason
    session_id: str | None = None
    result_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class VisualGroundingRuntimeSnapshot(OrchestrationModel):
    """
    Diagnostics for Phase 8 Step 17.
    """

    name: str
    session_count: int = Field(ge=0)
    result_count: int = Field(ge=0)
    grounded_count: int = Field(ge=0)
    ambiguous_count: int = Field(ge=0)
    low_confidence_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    not_found_count: int = Field(ge=0)
    safe_planning_count: int = Field(ge=0)
    runtime_event_count: int = Field(ge=0)
    last_reason: GroundingReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        return _clean_required(value)


class AmbiguityResolver:
    """
    Resolves candidate ambiguity.

    Silent guessing is prohibited.
    """

    def resolve(
        self,
        *,
        candidates: tuple[GroundingCandidate, ...],
        policy: TargetTrustPolicy,
    ) -> AmbiguityResolution:
        if not candidates:
            return AmbiguityResolution(
                ambiguous=False,
                top_candidates=(),
                reason="no grounding candidates",
                ask_user_message=None,
            )

        ordered = tuple(
            sorted(candidates, key=lambda candidate: candidate.confidence, reverse=True)
        )
        top = ordered[0]
        second = ordered[1] if len(ordered) > 1 else None

        if second is None:
            return AmbiguityResolution(
                ambiguous=False,
                top_candidates=(top,),
                reason="single best candidate",
            )

        if top.confidence - second.confidence <= policy.ambiguity_margin:
            return AmbiguityResolution(
                ambiguous=True,
                top_candidates=ordered[:3],
                reason="multiple similar grounding candidates",
                ask_user_message=_ambiguity_message(ordered[:3]),
            )

        return AmbiguityResolution(
            ambiguous=False,
            top_candidates=ordered[:3],
            reason="clear best candidate",
        )


class GroundingCandidateBuilder:
    """
    Builds candidates using the required grounding order.
    """

    def build(self, query: GroundingQuery) -> tuple[GroundingCandidate, ...]:
        candidates = [
            *self._from_workspace_graph(query),
            *self._from_accessibility_exact(query),
            *self._from_ocr_exact(query),
            *self._from_fuzzy_text(query),
            *self._from_semantic_icon(query),
            *self._from_spatial_reference(query),
            *self._from_context_inference(query),
        ]

        return _unique_candidates(tuple(candidates))

    def _from_workspace_graph(
        self,
        query: GroundingQuery,
    ) -> tuple[GroundingCandidate, ...]:
        if query.workspace_graph is None:
            return ()

        wanted = _query_terms(query.text)
        candidates: list[GroundingCandidate] = []

        for node in query.workspace_graph.nodes.values():
            if not node.active:
                continue

            label = node.label.lower()
            score = _term_score(wanted, label)

            if score <= 0.0:
                continue

            kind_bonus = _graph_kind_bonus(query=query, node=node)
            confidence = min(1.0, 0.58 + score * 0.30 + kind_bonus)

            candidates.append(
                GroundingCandidate(
                    target_kind=GroundingTargetKind.GRAPH_NODE,
                    label=node.label,
                    confidence=confidence,
                    strategy=GroundingStrategy.WORKSPACE_GRAPH_QUERY,
                    policy=TrustPolicyClassification.SAFE,
                    trust=node.trust,
                    region=node.region,
                    graph_node=node,
                    evidence=(
                        _evidence(
                            strategy=GroundingStrategy.WORKSPACE_GRAPH_QUERY,
                            description=f"graph node matched '{node.label}'",
                            score=confidence,
                        ),
                    ),
                )
            )

        return tuple(candidates)

    def _from_accessibility_exact(
        self,
        query: GroundingQuery,
    ) -> tuple[GroundingCandidate, ...]:
        wanted = _normalized_query(query.text)
        candidates: list[GroundingCandidate] = []

        for element in query.ui_elements:
            label = (element.label or "").strip()

            if not label:
                continue

            if label.lower() not in wanted and wanted not in label.lower():
                continue

            confidence = min(1.0, element.confidence.confidence + 0.08)
            candidates.append(
                GroundingCandidate(
                    target_kind=GroundingTargetKind.UI_ELEMENT,
                    label=label,
                    confidence=confidence,
                    strategy=GroundingStrategy.ACCESSIBILITY_EXACT_MATCH,
                    policy=element.policy_classification,
                    trust=element.trust,
                    region=element.element.bounds,
                    ui_element=element,
                    evidence=(
                        _evidence(
                            strategy=GroundingStrategy.ACCESSIBILITY_EXACT_MATCH,
                            description=(
                                f"accessibility label exactly matched '{label}'"
                            ),
                            score=confidence,
                        ),
                    ),
                )
            )

        return tuple(candidates)

    def _from_ocr_exact(
        self,
        query: GroundingQuery,
    ) -> tuple[GroundingCandidate, ...]:
        wanted = _normalized_query(query.text)
        candidates: list[GroundingCandidate] = []

        for region in query.text_regions:
            text = region.text.strip()

            if not text:
                continue

            if text.lower() not in wanted and wanted not in text.lower():
                continue

            confidence = min(1.0, region.confidence.confidence + 0.04)
            candidates.append(
                GroundingCandidate(
                    target_kind=GroundingTargetKind.OCR_TEXT_REGION,
                    label=text,
                    confidence=confidence,
                    strategy=GroundingStrategy.OCR_EXACT_MATCH,
                    policy=region.policy_classification,
                    trust=region.trust,
                    region=region.bounds,
                    text_region=region,
                    evidence=(
                        _evidence(
                            strategy=GroundingStrategy.OCR_EXACT_MATCH,
                            description=f"OCR text exactly matched '{text}'",
                            score=confidence,
                        ),
                    ),
                )
            )

        return tuple(candidates)

    def _from_fuzzy_text(
        self,
        query: GroundingQuery,
    ) -> tuple[GroundingCandidate, ...]:
        wanted = _normalized_query(query.text)
        candidates: list[GroundingCandidate] = []

        for region in query.text_regions:
            text = region.text.strip()

            if not text:
                continue

            ratio = SequenceMatcher(None, wanted, text.lower()).ratio()

            if ratio < 0.50:
                continue

            confidence = min(0.86, ratio * region.confidence.confidence)
            candidates.append(
                GroundingCandidate(
                    target_kind=GroundingTargetKind.OCR_TEXT_REGION,
                    label=text,
                    confidence=confidence,
                    strategy=GroundingStrategy.FUZZY_TEXT_MATCH,
                    policy=region.policy_classification,
                    trust=region.trust,
                    region=region.bounds,
                    text_region=region,
                    evidence=(
                        _evidence(
                            strategy=GroundingStrategy.FUZZY_TEXT_MATCH,
                            description=f"fuzzy OCR match '{text}'",
                            score=confidence,
                        ),
                    ),
                )
            )

        for element in query.ui_elements:
            label = (element.label or "").strip()

            if not label:
                continue

            ratio = SequenceMatcher(None, wanted, label.lower()).ratio()

            if ratio < 0.50:
                continue

            confidence = min(0.88, ratio * element.confidence.confidence)
            candidates.append(
                GroundingCandidate(
                    target_kind=GroundingTargetKind.UI_ELEMENT,
                    label=label,
                    confidence=confidence,
                    strategy=GroundingStrategy.FUZZY_TEXT_MATCH,
                    policy=element.policy_classification,
                    trust=element.trust,
                    region=element.element.bounds,
                    ui_element=element,
                    evidence=(
                        _evidence(
                            strategy=GroundingStrategy.FUZZY_TEXT_MATCH,
                            description=f"fuzzy UI label match '{label}'",
                            score=confidence,
                        ),
                    ),
                )
            )

        return tuple(candidates)

    def _from_semantic_icon(
        self,
        query: GroundingQuery,
    ) -> tuple[GroundingCandidate, ...]:
        text = _normalized_query(query.text)
        candidates: list[GroundingCandidate] = []
        icon_terms = {
            "run": ("run", "play", "start"),
            "search": ("search", "find"),
            "settings": ("settings", "gear"),
            "close": ("close", "x", "dismiss"),
        }

        for canonical, terms in icon_terms.items():
            if not any(term in text for term in terms):
                continue

            for element in query.ui_elements:
                label = (element.label or "").lower()
                kind_match = element.kind == DetectedUIElementKind.BUTTON
                label_match = any(term in label for term in terms)

                if not kind_match and not label_match:
                    continue

                confidence = min(0.84, element.confidence.confidence * 0.90)
                candidates.append(
                    GroundingCandidate(
                        target_kind=GroundingTargetKind.UI_ELEMENT,
                        label=element.label or canonical,
                        confidence=confidence,
                        strategy=GroundingStrategy.SEMANTIC_ICON_MATCH,
                        policy=element.policy_classification,
                        trust=element.trust,
                        region=element.element.bounds,
                        ui_element=element,
                        evidence=(
                            _evidence(
                                strategy=GroundingStrategy.SEMANTIC_ICON_MATCH,
                                description=f"semantic icon matched '{canonical}'",
                                score=confidence,
                            ),
                        ),
                    )
                )

        return tuple(candidates)

    def _from_spatial_reference(
        self,
        query: GroundingQuery,
    ) -> tuple[GroundingCandidate, ...]:
        text = _normalized_query(query.text)

        if query.active_region is None:
            return ()

        if not any(
            term in text
            for term in ("this", "that", "here", "current", "selected", "focused")
        ):
            return ()

        return (
            GroundingCandidate(
                target_kind=GroundingTargetKind.UNKNOWN,
                label="active visual region",
                confidence=0.64,
                strategy=GroundingStrategy.SPATIAL_REFERENCE,
                policy=TrustPolicyClassification.REVIEW,
                trust=TrustCalibration(
                    confidence=0.64,
                    stability=0.60,
                    ambiguity=0.36,
                    source=EnvironmentSource.OS_OBSERVER,
                    reason="spatial reference inferred from active region",
                ),
                region=query.active_region,
                evidence=(
                    _evidence(
                        strategy=GroundingStrategy.SPATIAL_REFERENCE,
                        description="spatial reference matched active region",
                        score=0.64,
                    ),
                ),
            ),
        )

    def _from_context_inference(
        self,
        query: GroundingQuery,
    ) -> tuple[GroundingCandidate, ...]:
        if query.ui_context is None:
            return ()

        text = _normalized_query(query.text)
        scene = query.ui_context.scene

        if "error" in text and scene.kind == SemanticSceneKind.ERROR_DIALOG:
            return (
                GroundingCandidate(
                    target_kind=GroundingTargetKind.SEMANTIC_SCENE,
                    label=scene.summary,
                    confidence=min(0.86, scene.confidence + 0.02),
                    strategy=GroundingStrategy.CONTEXT_INFERENCE,
                    policy=query.ui_context.policy_classification,
                    trust=scene.trust,
                    region=scene.region,
                    scene_kind=scene.kind,
                    evidence=(
                        _evidence(
                            strategy=GroundingStrategy.CONTEXT_INFERENCE,
                            description="semantic error scene matched intent",
                            score=scene.confidence,
                        ),
                    ),
                ),
            )

        if query.pattern_result is not None and query.pattern_result.best_match:
            best = query.pattern_result.best_match

            if any(term in text for term in ("dialog", "picker", "warning", "prompt")):
                return (
                    GroundingCandidate(
                        target_kind=GroundingTargetKind.PATTERN,
                        label=best.pattern.name,
                        confidence=best.score,
                        strategy=GroundingStrategy.CONTEXT_INFERENCE,
                        policy=best.policy,
                        trust=best.trust,
                        region=best.region,
                        pattern_kind=best.pattern.kind,
                        evidence=(
                            _evidence(
                                strategy=GroundingStrategy.CONTEXT_INFERENCE,
                                description=(
                                    f"pattern context matched {best.pattern.name}"
                                ),
                                score=best.score,
                            ),
                        ),
                    ),
                )

        return ()


class VisualGroundingRuntime:
    """
    Phase 8 Step 17 Visual Grounding Runtime.

    Responsibilities:
    - map natural language intent to trusted visual/graph targets
    - use strict grounding order
    - detect ambiguity
    - enforce target trust policy
    - ask instead of guessing when confidence is low

    Non-responsibilities:
    - no clicking
    - no typing
    - no raw coordinate automation
    - no action execution
    """

    def __init__(
        self,
        *,
        name: str = "visual_grounding_runtime",
        candidate_builder: GroundingCandidateBuilder | None = None,
        ambiguity_resolver: AmbiguityResolver | None = None,
        target_policy: TargetTrustPolicy | None = None,
    ) -> None:
        cleaned = name.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._candidate_builder = candidate_builder or GroundingCandidateBuilder()
        self._ambiguity_resolver = ambiguity_resolver or AmbiguityResolver()
        self._target_policy = target_policy or TargetTrustPolicy()
        self._sessions: dict[str, VisualGroundingSession] = {}
        self._results: list[VisualGroundingResult] = []
        self._events: list[VisualGroundingRuntimeEvent] = []
        self._lock = RLock()
        self._last_reason: GroundingReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def create_session(
        self,
        *,
        workspace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> VisualGroundingSession:
        session = VisualGroundingSession(
            workspace_id=workspace_id,
            metadata=metadata or {},
        )
        event = self._event(
            kind=GroundingEventKind.SESSION_CREATED,
            reason=GroundingReason.SESSION_CREATED,
            session_id=session.session_id,
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._events.append(event)
            self._last_reason = event.reason

        return session

    def ground(self, query: GroundingQuery) -> VisualGroundingResult:
        if self.session_for(query.session_id) is None:
            result = self._failed_result(query)
            self._record_result(result)
            return result

        candidates = self._candidate_builder.build(query)
        ordered = tuple(
            sorted(candidates, key=lambda candidate: candidate.confidence, reverse=True)
        )
        ambiguity = self._ambiguity_resolver.resolve(
            candidates=ordered,
            policy=self._target_policy,
        )
        result = self._result_from(
            query=query,
            candidates=ordered,
            ambiguity=ambiguity,
        )

        self._record_result(result)
        self._touch_session(query.session_id)

        return result

    def session_for(self, session_id: str) -> VisualGroundingSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def results(self) -> tuple[VisualGroundingResult, ...]:
        with self._lock:
            return tuple(self._results)

    def events(self) -> tuple[VisualGroundingRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def snapshot(self) -> VisualGroundingRuntimeSnapshot:
        with self._lock:
            return VisualGroundingRuntimeSnapshot(
                name=self.name,
                session_count=len(self._sessions),
                result_count=len(self._results),
                grounded_count=sum(
                    1
                    for result in self._results
                    if result.status == GroundingStatus.GROUNDED
                ),
                ambiguous_count=sum(
                    1
                    for result in self._results
                    if result.status == GroundingStatus.AMBIGUOUS
                ),
                low_confidence_count=sum(
                    1
                    for result in self._results
                    if result.status == GroundingStatus.LOW_CONFIDENCE
                ),
                blocked_count=sum(
                    1
                    for result in self._results
                    if result.status == GroundingStatus.BLOCKED
                ),
                not_found_count=sum(
                    1
                    for result in self._results
                    if result.status == GroundingStatus.NOT_FOUND
                ),
                safe_planning_count=sum(
                    1 for result in self._results if result.safe_for_action_planning
                ),
                runtime_event_count=len(self._events),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        event = self._event(
            kind=GroundingEventKind.RUNTIME_RESET,
            reason=GroundingReason.RUNTIME_RESET,
        )

        with self._lock:
            self._sessions.clear()
            self._results.clear()
            self._events.clear()
            self._events.append(event)
            self._last_reason = event.reason

    def _result_from(
        self,
        *,
        query: GroundingQuery,
        candidates: tuple[GroundingCandidate, ...],
        ambiguity: AmbiguityResolution,
    ) -> VisualGroundingResult:
        if not candidates:
            return VisualGroundingResult(
                status=GroundingStatus.NOT_FOUND,
                reason=GroundingReason.TARGET_NOT_FOUND,
                decision=GroundingDecision.NOT_FOUND,
                query=query,
                candidates=(),
                selected=None,
                ambiguity=ambiguity,
                safe_for_action_planning=False,
                message="no visual target found",
            )

        selected = candidates[0]

        if selected.policy == TrustPolicyClassification.BLOCKED:
            return VisualGroundingResult(
                status=GroundingStatus.BLOCKED,
                reason=GroundingReason.TARGET_BLOCKED_BY_POLICY,
                decision=GroundingDecision.BLOCKED,
                query=query,
                candidates=candidates,
                selected=selected,
                ambiguity=ambiguity,
                safe_for_action_planning=False,
                message="grounded target is blocked by policy",
            )

        if ambiguity.ambiguous:
            return VisualGroundingResult(
                status=GroundingStatus.AMBIGUOUS,
                reason=GroundingReason.AMBIGUITY_DETECTED,
                decision=GroundingDecision.ASK_USER,
                query=query,
                candidates=candidates,
                selected=None,
                ambiguity=ambiguity,
                safe_for_action_planning=False,
                message=ambiguity.ask_user_message or "ambiguous visual target",
            )

        if selected.confidence < self._target_policy.minimum_grounding_confidence:
            return VisualGroundingResult(
                status=GroundingStatus.LOW_CONFIDENCE,
                reason=GroundingReason.LOW_CONFIDENCE_TARGET,
                decision=GroundingDecision.ASK_USER,
                query=query,
                candidates=candidates,
                selected=None,
                ambiguity=ambiguity,
                safe_for_action_planning=False,
                message="low confidence grounding; ask user",
            )

        if selected.confidence < self._target_policy.verify_first_confidence:
            return VisualGroundingResult(
                status=GroundingStatus.GROUNDED,
                reason=GroundingReason.TARGET_GROUNDED,
                decision=GroundingDecision.VERIFY_FIRST,
                query=query,
                candidates=candidates,
                selected=selected,
                ambiguity=ambiguity,
                safe_for_action_planning=False,
                message="target grounded but verification required",
            )

        return VisualGroundingResult(
            status=GroundingStatus.GROUNDED,
            reason=GroundingReason.TARGET_GROUNDED,
            decision=GroundingDecision.GROUNDED,
            query=query,
            candidates=candidates,
            selected=selected,
            ambiguity=ambiguity,
            safe_for_action_planning=True,
            message=f"grounded target: {selected.label}",
        )

    def _failed_result(self, query: GroundingQuery) -> VisualGroundingResult:
        ambiguity = AmbiguityResolution(
            ambiguous=False,
            top_candidates=(),
            reason="session not found",
        )

        return VisualGroundingResult(
            status=GroundingStatus.FAILED,
            reason=GroundingReason.SESSION_NOT_FOUND,
            decision=GroundingDecision.BLOCKED,
            query=query,
            candidates=(),
            selected=None,
            ambiguity=ambiguity,
            safe_for_action_planning=False,
            message="visual grounding session not found",
        )

    def _record_result(self, result: VisualGroundingResult) -> None:
        event = self._event(
            kind=(
                GroundingEventKind.GROUNDING_COMPLETED
                if result.status == GroundingStatus.GROUNDED
                else GroundingEventKind.GROUNDING_BLOCKED
            ),
            reason=result.reason,
            session_id=result.query.session_id,
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
        kind: GroundingEventKind,
        reason: GroundingReason,
        session_id: str | None = None,
        result_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VisualGroundingRuntimeEvent:
        return VisualGroundingRuntimeEvent(
            kind=kind,
            reason=reason,
            session_id=session_id,
            result_id=result_id,
            metadata=metadata or {},
        )


def _evidence(
    *,
    strategy: GroundingStrategy,
    description: str,
    score: float,
) -> GroundingEvidence:
    return GroundingEvidence(
        strategy=strategy,
        description=description,
        score=max(0.0, min(1.0, score)),
    )


def _unique_candidates(
    candidates: tuple[GroundingCandidate, ...],
) -> tuple[GroundingCandidate, ...]:
    best: dict[tuple[GroundingTargetKind, str], GroundingCandidate] = {}

    for candidate in candidates:
        key = (candidate.target_kind, candidate.label.lower())
        existing = best.get(key)

        if existing is None or candidate.confidence > existing.confidence:
            best[key] = candidate

    return tuple(best.values())


def _normalized_query(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _query_terms(text: str) -> tuple[str, ...]:
    ignored = {
        "click",
        "focus",
        "inspect",
        "open",
        "read",
        "select",
        "the",
        "a",
        "an",
        "button",
        "target",
        "please",
        "on",
        "at",
    }

    return tuple(
        term
        for term in _normalized_query(text).split()
        if term not in ignored and len(term) > 1
    )


def _term_score(terms: tuple[str, ...], label: str) -> float:
    if not terms:
        return 0.0

    matched = sum(1 for term in terms if term in label.lower())

    return matched / len(terms)


def _graph_kind_bonus(
    *,
    query: GroundingQuery,
    node: GraphNode,
) -> float:
    text = _normalized_query(query.text)

    if "terminal" in text and node.kind == GraphNodeKind.TERMINAL:
        return 0.10

    if "error" in text and node.kind == GraphNodeKind.ERROR:
        return 0.10

    if "file" in text and node.kind == GraphNodeKind.FILE:
        return 0.08

    if "dialog" in text and node.kind == GraphNodeKind.DIALOG:
        return 0.08

    if query.intent == GroundingIntentKind.FOCUS and node.kind in {
        GraphNodeKind.TERMINAL,
        GraphNodeKind.EDITOR,
        GraphNodeKind.WINDOW,
    }:
        return 0.05

    return 0.0


def _ambiguity_message(candidates: tuple[GroundingCandidate, ...]) -> str:
    labels = ", ".join(candidate.label for candidate in candidates)

    return f"I found multiple possible targets: {labels}. Which one do you mean?"


def _clean_required(value: str) -> str:
    cleaned = value.strip()

    if not cleaned:
        raise ValueError("field cannot be empty.")

    return cleaned