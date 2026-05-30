from __future__ import annotations

from enum import StrEnum
from threading import RLock
from typing import Any, Protocol
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.environment.models import (
    EnvironmentSource,
    PrivacyClassification,
    ScreenRegion,
    TrustCalibration,
    UIElement,
)
from jarvis.environment.models import (
    UIElementKind as BaseUIElementKind,
)
from jarvis.environment.ocr import OCRTextKind, OCRTextRegion
from jarvis.environment.trust_runtime import (
    TrustCalibrationRuntime,
    TrustPolicyClassification,
    TrustSubjectKind,
)
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class UIDetectionSource(StrEnum):
    ACCESSIBILITY = "accessibility"
    APP_PARSER = "app_parser"
    VISUAL_FALLBACK = "visual_fallback"


class DetectedUIElementRole(StrEnum):
    UNKNOWN = "unknown"
    BUTTON = "button"
    INPUT = "input"
    TEXT = "text"
    MENU = "menu"
    MENU_ITEM = "menu_item"
    TAB = "tab"
    PANEL = "panel"
    TERMINAL = "terminal"
    CODE_EDITOR = "code_editor"
    BROWSER_CONTENT = "browser_content"
    DOCUMENT = "document"
    CHECKBOX = "checkbox"
    DROPDOWN = "dropdown"
    SCROLL_REGION = "scroll_region"


class DetectedUIElementKind(StrEnum):
    UNKNOWN = "unknown"
    BUTTON = "button"
    INPUT = "input"
    TEXT = "text"
    MENU = "menu"
    MENU_ITEM = "menu_item"
    TAB = "tab"
    PANEL = "panel"
    TERMINAL = "terminal"
    CODE_EDITOR = "code_editor"
    BROWSER_CONTENT = "browser_content"
    DOCUMENT = "document"
    CHECKBOX = "checkbox"
    DROPDOWN = "dropdown"
    SCROLL_REGION = "scroll_region"


class ElementInteractionSafety(StrEnum):
    SAFE = "safe"
    VERIFY_FIRST = "verify_first"
    ASK_USER = "ask_user"
    BLOCKED = "blocked"


class UIDetectionStatus(StrEnum):
    DETECTED = "detected"
    PARTIAL = "partial"
    LOW_CONFIDENCE = "low_confidence"
    PRIVACY_BLOCKED = "privacy_blocked"
    FAILED = "failed"


class UIDetectionReason(StrEnum):
    SESSION_CREATED = "session_created"
    ACCESSIBILITY_ELEMENTS_DETECTED = "accessibility_elements_detected"
    APP_ELEMENTS_DETECTED = "app_elements_detected"
    VISUAL_ELEMENTS_DETECTED = "visual_elements_detected"
    INTERACTIVE_MAP_BUILT = "interactive_map_built"
    LOW_CONFIDENCE_REJECTED = "low_confidence_rejected"
    PRIVACY_BLOCKED = "privacy_blocked"
    SESSION_NOT_FOUND = "session_not_found"
    RUNTIME_RESET = "runtime_reset"


class UIDetectionRuntimeEventKind(StrEnum):
    SESSION_CREATED = "session_created"
    DETECTION_COMPLETED = "detection_completed"
    DETECTION_BLOCKED = "detection_blocked"
    MAP_BUILT = "map_built"
    RUNTIME_RESET = "runtime_reset"


class ElementConfidenceScore(OrchestrationModel):
    score_id: str = Field(default_factory=lambda: f"element_score_{uuid4().hex}")
    confidence: float = Field(ge=0.0, le=1.0)
    stability: float = Field(ge=0.0, le=1.0)
    ambiguity: float = Field(ge=0.0, le=1.0)
    source: UIDetectionSource
    accepted: bool
    reason: str
    created_at: object = Field(default_factory=utc_now)

    @field_validator("score_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class ElementClassifierConfig(OrchestrationModel):
    accessibility_min_confidence: float = Field(default=0.70, ge=0.0, le=1.0)
    app_parser_min_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    visual_fallback_min_confidence: float = Field(default=0.82, ge=0.0, le=1.0)
    interactive_min_confidence: float = Field(default=0.80, ge=0.0, le=1.0)
    max_interactive_ambiguity: float = Field(default=0.25, ge=0.0, le=1.0)


class RawUIElementCandidate(OrchestrationModel):
    """
    Adapter/parser output before trust calibration.

    This avoids forcing role/label into the Step 0 UIElement contract.
    """

    candidate_id: str = Field(default_factory=lambda: f"ui_candidate_{uuid4().hex}")
    element: UIElement
    label: str | None = None
    role: DetectedUIElementRole = DetectedUIElementRole.UNKNOWN
    source: UIDetectionSource
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("candidate_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class DetectedUIElement(OrchestrationModel):
    detected_id: str = Field(default_factory=lambda: f"detected_ui_{uuid4().hex}")
    element: UIElement
    label: str | None = None
    role: DetectedUIElementRole = DetectedUIElementRole.UNKNOWN
    kind: DetectedUIElementKind
    detection_source: UIDetectionSource
    confidence: ElementConfidenceScore
    privacy: PrivacyClassification
    trust: TrustCalibration
    policy_classification: TrustPolicyClassification
    interaction_safety: ElementInteractionSafety
    interactive: bool = False
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("detected_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class AccessibilityBridgeResult(OrchestrationModel):
    elements: tuple[DetectedUIElement, ...]
    source_available: bool = True
    message: str = "accessibility bridge completed"


class AppParserResult(OrchestrationModel):
    elements: tuple[DetectedUIElement, ...]
    parser_name: str = "generic_app_parser"
    message: str = "app parser completed"


class VisualFallbackResult(OrchestrationModel):
    elements: tuple[DetectedUIElement, ...]
    fallback_used: bool = True
    message: str = "visual fallback completed"


class UIDetectionRequest(OrchestrationModel):
    request_id: str = Field(default_factory=lambda: f"ui_detect_req_{uuid4().hex}")
    session_id: str
    region: ScreenRegion
    text_regions: tuple[OCRTextRegion, ...] = ()
    privacy: PrivacyClassification = PrivacyClassification.WORKSPACE
    prefer_accessibility: bool = True
    allow_visual_fallback: bool = True
    app_name: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("request_id", "session_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class UIDetectionResult(OrchestrationModel):
    result_id: str = Field(default_factory=lambda: f"ui_detect_result_{uuid4().hex}")
    status: UIDetectionStatus
    reason: UIDetectionReason
    request_id: str
    elements: tuple[DetectedUIElement, ...] = ()
    rejected_elements: tuple[DetectedUIElement, ...] = ()
    source_order: tuple[UIDetectionSource, ...] = ()
    message: str
    created_at: object = Field(default_factory=utc_now)

    @field_validator("result_id", "request_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class InteractiveRegionMap(OrchestrationModel):
    map_id: str = Field(default_factory=lambda: f"interactive_map_{uuid4().hex}")
    session_id: str
    elements: tuple[DetectedUIElement, ...]
    safe_count: int = Field(ge=0)
    verify_first_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    created_at: object = Field(default_factory=utc_now)

    @field_validator("map_id", "session_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _counts_must_match(self) -> InteractiveRegionMap:
        total = self.safe_count + self.verify_first_count + self.blocked_count

        if total != len(self.elements):
            raise ValueError("interactive map counts must match element count.")

        return self


class UIDetectionSession(OrchestrationModel):
    session_id: str = Field(default_factory=lambda: f"ui_session_{uuid4().hex}")
    workspace_id: str
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class UIDetectionRuntimeEvent(OrchestrationModel):
    event_id: str = Field(default_factory=lambda: f"ui_event_{uuid4().hex}")
    kind: UIDetectionRuntimeEventKind
    reason: UIDetectionReason
    session_id: str | None = None
    request_id: str | None = None
    element_count: int = Field(default=0, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class UIDetectionRuntimeSnapshot(OrchestrationModel):
    name: str
    session_count: int = Field(ge=0)
    detection_count: int = Field(ge=0)
    element_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    interactive_map_count: int = Field(ge=0)
    safe_interactive_count: int = Field(ge=0)
    verify_first_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    runtime_event_count: int = Field(ge=0)
    last_reason: UIDetectionReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        return _clean_required(value)


class AccessibilityBridge(Protocol):
    def detect(self, request: UIDetectionRequest) -> tuple[RawUIElementCandidate, ...]:
        ...


class VisualFallbackDetector(Protocol):
    def detect(self, request: UIDetectionRequest) -> tuple[RawUIElementCandidate, ...]:
        ...


class FakeAccessibilityBridge:
    def detect(self, request: UIDetectionRequest) -> tuple[RawUIElementCandidate, ...]:
        element = _ui_element(
            role=DetectedUIElementRole.BUTTON,
            bounds=ScreenRegion(
                x=request.region.x + 10,
                y=request.region.y + 10,
                width=80,
                height=30,
            ),
            interactive=True,
        )

        return (
            RawUIElementCandidate(
                element=element,
                label="Run",
                role=DetectedUIElementRole.BUTTON,
                source=UIDetectionSource.ACCESSIBILITY,
            ),
        )


class FakeAppSpecificParser:
    def detect(self, request: UIDetectionRequest) -> tuple[RawUIElementCandidate, ...]:
        candidates: list[RawUIElementCandidate] = []

        for text_region in request.text_regions:
            if text_region.kind == OCRTextKind.CODE:
                candidates.append(
                    RawUIElementCandidate(
                        element=_ui_element(
                            role=DetectedUIElementRole.CODE_EDITOR,
                            bounds=text_region.bounds,
                            interactive=False,
                        ),
                        label="Code Editor",
                        role=DetectedUIElementRole.CODE_EDITOR,
                        source=UIDetectionSource.APP_PARSER,
                    )
                )

            if text_region.kind in {OCRTextKind.TERMINAL, OCRTextKind.ERROR}:
                candidates.append(
                    RawUIElementCandidate(
                        element=_ui_element(
                            role=DetectedUIElementRole.TERMINAL,
                            bounds=text_region.bounds,
                            interactive=False,
                        ),
                        label="Terminal",
                        role=DetectedUIElementRole.TERMINAL,
                        source=UIDetectionSource.APP_PARSER,
                    )
                )

        return tuple(candidates)


class FakeVisualFallbackDetector:
    def detect(self, request: UIDetectionRequest) -> tuple[RawUIElementCandidate, ...]:
        element = _ui_element(
            role=DetectedUIElementRole.BUTTON,
            bounds=ScreenRegion(
                x=request.region.x + 20,
                y=request.region.y + 20,
                width=100,
                height=40,
            ),
            interactive=True,
        )

        return (
            RawUIElementCandidate(
                element=element,
                label="Visual Candidate",
                role=DetectedUIElementRole.BUTTON,
                source=UIDetectionSource.VISUAL_FALLBACK,
            ),
        )


class ElementClassifier:
    def __init__(
        self,
        *,
        trust_runtime: TrustCalibrationRuntime,
        config: ElementClassifierConfig | None = None,
    ) -> None:
        self._trust_runtime = trust_runtime
        self._config = config or ElementClassifierConfig()

    def classify(
        self,
        *,
        candidate: RawUIElementCandidate,
        privacy: PrivacyClassification,
    ) -> DetectedUIElement:
        score = self._score(candidate=candidate, privacy=privacy)
        kind = _kind_for_role(candidate.role)
        observation = self._trust_runtime.calibrate_observation(
            subject_id=candidate.element.element_id,
            subject_kind=TrustSubjectKind.UI_ELEMENT,
            source=_environment_source_for(candidate.source),
            confidence=score.confidence,
            stability=score.stability,
            ambiguity=score.ambiguity,
            privacy=privacy,
            reason="UI element detection calibrated",
            metadata={
                "detection_source": candidate.source.value,
                "role": candidate.role.value,
                "label": candidate.label,
            },
        )
        safety = self._safety_for(
            element=candidate.element,
            score=score,
            policy=observation.policy_classification,
        )

        return DetectedUIElement(
            element=candidate.element,
            label=candidate.label,
            role=candidate.role,
            kind=kind,
            detection_source=candidate.source,
            confidence=score,
            privacy=privacy,
            trust=observation.calibration,
            policy_classification=observation.policy_classification,
            interaction_safety=safety,
            interactive=(
                candidate.element.interactive
                and safety != ElementInteractionSafety.BLOCKED
            ),
            metadata=candidate.metadata,
        )

    def _score(
        self,
        *,
        candidate: RawUIElementCandidate,
        privacy: PrivacyClassification,
    ) -> ElementConfidenceScore:
        if candidate.source == UIDetectionSource.ACCESSIBILITY:
            confidence = 0.94
            minimum = self._config.accessibility_min_confidence
        elif candidate.source == UIDetectionSource.APP_PARSER:
            confidence = 0.86
            minimum = self._config.app_parser_min_confidence
        else:
            confidence = 0.78
            minimum = self._config.visual_fallback_min_confidence

        if candidate.label:
            confidence += 0.02

        if candidate.element.interactive:
            confidence += 0.01

        if privacy in {
            PrivacyClassification.SECRET,
            PrivacyClassification.BLOCKED,
        }:
            confidence = 0.0

        confidence = max(0.0, min(1.0, confidence))
        ambiguity = (
            0.08
            if candidate.source != UIDetectionSource.VISUAL_FALLBACK
            else 0.28
        )

        return ElementConfidenceScore(
            confidence=confidence,
            stability=max(0.0, min(1.0, confidence + 0.03)),
            ambiguity=ambiguity,
            source=candidate.source,
            accepted=confidence >= minimum,
            reason="UI element confidence scored",
        )

    def _safety_for(
        self,
        *,
        element: UIElement,
        score: ElementConfidenceScore,
        policy: TrustPolicyClassification,
    ) -> ElementInteractionSafety:
        if policy == TrustPolicyClassification.BLOCKED:
            return ElementInteractionSafety.BLOCKED

        if not score.accepted:
            return ElementInteractionSafety.BLOCKED

        if not element.interactive:
            return ElementInteractionSafety.SAFE

        if score.confidence < self._config.interactive_min_confidence:
            return ElementInteractionSafety.BLOCKED

        if score.ambiguity > self._config.max_interactive_ambiguity:
            return ElementInteractionSafety.VERIFY_FIRST

        if policy == TrustPolicyClassification.ASK_USER:
            return ElementInteractionSafety.ASK_USER

        if policy in {
            TrustPolicyClassification.REVIEW,
            TrustPolicyClassification.VERIFY_FIRST,
        }:
            return ElementInteractionSafety.VERIFY_FIRST

        return ElementInteractionSafety.SAFE


class UIElementDetector:
    def __init__(
        self,
        *,
        accessibility: AccessibilityBridge | None = None,
        app_parser: FakeAppSpecificParser | None = None,
        visual_fallback: VisualFallbackDetector | None = None,
        classifier: ElementClassifier,
    ) -> None:
        self._accessibility = accessibility or FakeAccessibilityBridge()
        self._app_parser = app_parser or FakeAppSpecificParser()
        self._visual_fallback = visual_fallback or FakeVisualFallbackDetector()
        self._classifier = classifier

    def detect(self, request: UIDetectionRequest) -> UIDetectionResult:
        if request.privacy in {
            PrivacyClassification.SECRET,
            PrivacyClassification.BLOCKED,
        }:
            return UIDetectionResult(
                status=UIDetectionStatus.PRIVACY_BLOCKED,
                reason=UIDetectionReason.PRIVACY_BLOCKED,
                request_id=request.request_id,
                message="privacy policy blocks UI detection",
            )

        source_order: list[UIDetectionSource] = []
        accepted: list[DetectedUIElement] = []
        rejected: list[DetectedUIElement] = []

        if request.prefer_accessibility:
            source_order.append(UIDetectionSource.ACCESSIBILITY)
            self._extend_from_candidates(
                accepted=accepted,
                rejected=rejected,
                candidates=self._accessibility.detect(request),
                privacy=request.privacy,
            )

        source_order.append(UIDetectionSource.APP_PARSER)
        self._extend_from_candidates(
            accepted=accepted,
            rejected=rejected,
            candidates=self._app_parser.detect(request),
            privacy=request.privacy,
        )

        if not accepted and request.allow_visual_fallback:
            source_order.append(UIDetectionSource.VISUAL_FALLBACK)
            self._extend_from_candidates(
                accepted=accepted,
                rejected=rejected,
                candidates=self._visual_fallback.detect(request),
                privacy=request.privacy,
            )

        if not accepted:
            return UIDetectionResult(
                status=UIDetectionStatus.LOW_CONFIDENCE,
                reason=UIDetectionReason.LOW_CONFIDENCE_REJECTED,
                request_id=request.request_id,
                elements=(),
                rejected_elements=tuple(rejected),
                source_order=tuple(source_order),
                message="no accepted UI elements",
            )

        return UIDetectionResult(
            status=UIDetectionStatus.DETECTED,
            reason=_reason_for_source(source_order[0]),
            request_id=request.request_id,
            elements=tuple(accepted),
            rejected_elements=tuple(rejected),
            source_order=tuple(source_order),
            message="UI elements detected",
        )

    def _extend_from_candidates(
        self,
        *,
        accepted: list[DetectedUIElement],
        rejected: list[DetectedUIElement],
        candidates: tuple[RawUIElementCandidate, ...],
        privacy: PrivacyClassification,
    ) -> None:
        for candidate in candidates:
            detected = self._classifier.classify(
                candidate=candidate,
                privacy=privacy,
            )

            if detected.confidence.accepted:
                accepted.append(detected)
            else:
                rejected.append(detected)


class UIDetectionRuntime:
    def __init__(
        self,
        *,
        name: str = "ui_detection_runtime",
        trust_runtime: TrustCalibrationRuntime | None = None,
        classifier_config: ElementClassifierConfig | None = None,
        detector: UIElementDetector | None = None,
    ) -> None:
        cleaned = name.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._trust_runtime = trust_runtime or TrustCalibrationRuntime()
        classifier = ElementClassifier(
            trust_runtime=self._trust_runtime,
            config=classifier_config,
        )
        self._detector = detector or UIElementDetector(classifier=classifier)
        self._sessions: dict[str, UIDetectionSession] = {}
        self._results: list[UIDetectionResult] = []
        self._maps: list[InteractiveRegionMap] = []
        self._events: list[UIDetectionRuntimeEvent] = []
        self._lock = RLock()
        self._last_reason: UIDetectionReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def create_session(
        self,
        *,
        workspace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> UIDetectionSession:
        session = UIDetectionSession(workspace_id=workspace_id, metadata=metadata or {})
        event = self._event(
            kind=UIDetectionRuntimeEventKind.SESSION_CREATED,
            reason=UIDetectionReason.SESSION_CREATED,
            session_id=session.session_id,
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._events.append(event)
            self._last_reason = event.reason

        return session

    def detect_elements(self, request: UIDetectionRequest) -> UIDetectionResult:
        session = self.session_for(request.session_id)

        if session is None:
            result = UIDetectionResult(
                status=UIDetectionStatus.FAILED,
                reason=UIDetectionReason.SESSION_NOT_FOUND,
                request_id=request.request_id,
                message="UI detection session not found",
            )
            self._record_result(result, session_id=request.session_id)

            return result

        result = self._detector.detect(request)
        self._record_result(result, session_id=request.session_id)
        self._touch_session(request.session_id)

        return result

    def build_interactive_region_map(
        self,
        *,
        session_id: str,
        elements: tuple[DetectedUIElement, ...],
    ) -> InteractiveRegionMap:
        if self.session_for(session_id) is None:
            raise ValueError(f"UI detection session not found: {session_id}")

        interactive = tuple(
            element for element in elements if element.element.interactive
        )
        safe_count = sum(
            1
            for element in interactive
            if element.interaction_safety == ElementInteractionSafety.SAFE
        )
        verify_first_count = sum(
            1
            for element in interactive
            if element.interaction_safety == ElementInteractionSafety.VERIFY_FIRST
        )
        blocked_count = sum(
            1
            for element in interactive
            if element.interaction_safety
            in {
                ElementInteractionSafety.ASK_USER,
                ElementInteractionSafety.BLOCKED,
            }
        )
        region_map = InteractiveRegionMap(
            session_id=session_id,
            elements=interactive,
            safe_count=safe_count,
            verify_first_count=verify_first_count,
            blocked_count=blocked_count,
        )
        event = self._event(
            kind=UIDetectionRuntimeEventKind.MAP_BUILT,
            reason=UIDetectionReason.INTERACTIVE_MAP_BUILT,
            session_id=session_id,
            element_count=len(interactive),
        )

        with self._lock:
            self._maps.append(region_map)
            self._events.append(event)
            self._last_reason = event.reason

        return region_map

    def session_for(self, session_id: str) -> UIDetectionSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def results(self) -> tuple[UIDetectionResult, ...]:
        with self._lock:
            return tuple(self._results)

    def maps(self) -> tuple[InteractiveRegionMap, ...]:
        with self._lock:
            return tuple(self._maps)

    def events(self) -> tuple[UIDetectionRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def snapshot(self) -> UIDetectionRuntimeSnapshot:
        with self._lock:
            elements = [
                element
                for result in self._results
                for element in result.elements
            ]
            rejected = [
                element
                for result in self._results
                for element in result.rejected_elements
            ]

            return UIDetectionRuntimeSnapshot(
                name=self.name,
                session_count=len(self._sessions),
                detection_count=len(self._results),
                element_count=len(elements),
                rejected_count=len(rejected),
                interactive_map_count=len(self._maps),
                safe_interactive_count=sum(item.safe_count for item in self._maps),
                verify_first_count=sum(item.verify_first_count for item in self._maps),
                blocked_count=sum(item.blocked_count for item in self._maps),
                runtime_event_count=len(self._events),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        event = self._event(
            kind=UIDetectionRuntimeEventKind.RUNTIME_RESET,
            reason=UIDetectionReason.RUNTIME_RESET,
        )

        with self._lock:
            self._sessions.clear()
            self._results.clear()
            self._maps.clear()
            self._events.clear()
            self._events.append(event)
            self._last_reason = event.reason

    def _record_result(
        self,
        result: UIDetectionResult,
        *,
        session_id: str,
    ) -> None:
        event_kind = (
            UIDetectionRuntimeEventKind.DETECTION_COMPLETED
            if result.status in {UIDetectionStatus.DETECTED, UIDetectionStatus.PARTIAL}
            else UIDetectionRuntimeEventKind.DETECTION_BLOCKED
        )
        event = self._event(
            kind=event_kind,
            reason=result.reason,
            session_id=session_id,
            request_id=result.request_id,
            element_count=len(result.elements),
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
        kind: UIDetectionRuntimeEventKind,
        reason: UIDetectionReason,
        session_id: str | None = None,
        request_id: str | None = None,
        element_count: int = 0,
    ) -> UIDetectionRuntimeEvent:
        return UIDetectionRuntimeEvent(
            kind=kind,
            reason=reason,
            session_id=session_id,
            request_id=request_id,
            element_count=element_count,
        )


def _ui_element(
    *,
    role: DetectedUIElementRole,
    bounds: ScreenRegion,
    interactive: bool,
) -> UIElement:
    data: dict[str, Any] = {
        "bounds": bounds,
        "interactive": interactive,
        "trust": TrustCalibration(
            confidence=0.95,
            stability=0.95,
            ambiguity=0.0,
            source=EnvironmentSource.ACCESSIBILITY,
            reason="fake UI element",
        ),
    }

    if "kind" in UIElement.model_fields:
        data["kind"] = _base_kind_for_role(role)

    if "metadata" in UIElement.model_fields:
        data["metadata"] = {"role": role.value}

    return UIElement(**data)


def _base_kind_for_role(role: DetectedUIElementRole) -> BaseUIElementKind:
    candidates = (
        role.name,
        role.value.upper(),
        "TEXT",
        "UNKNOWN",
    )

    for candidate in candidates:
        value = getattr(BaseUIElementKind, candidate, None)

        if isinstance(value, BaseUIElementKind):
            return value

    return next(iter(BaseUIElementKind))


def _kind_for_role(role: DetectedUIElementRole) -> DetectedUIElementKind:
    mapping = {
        DetectedUIElementRole.BUTTON: DetectedUIElementKind.BUTTON,
        DetectedUIElementRole.INPUT: DetectedUIElementKind.INPUT,
        DetectedUIElementRole.MENU: DetectedUIElementKind.MENU,
        DetectedUIElementRole.MENU_ITEM: DetectedUIElementKind.MENU_ITEM,
        DetectedUIElementRole.TAB: DetectedUIElementKind.TAB,
        DetectedUIElementRole.PANEL: DetectedUIElementKind.PANEL,
        DetectedUIElementRole.TERMINAL: DetectedUIElementKind.TERMINAL,
        DetectedUIElementRole.CODE_EDITOR: DetectedUIElementKind.CODE_EDITOR,
        DetectedUIElementRole.BROWSER_CONTENT: DetectedUIElementKind.BROWSER_CONTENT,
        DetectedUIElementRole.DOCUMENT: DetectedUIElementKind.DOCUMENT,
        DetectedUIElementRole.CHECKBOX: DetectedUIElementKind.CHECKBOX,
        DetectedUIElementRole.DROPDOWN: DetectedUIElementKind.DROPDOWN,
        DetectedUIElementRole.SCROLL_REGION: DetectedUIElementKind.SCROLL_REGION,
    }

    return mapping.get(role, DetectedUIElementKind.TEXT)


def _environment_source_for(source: UIDetectionSource) -> EnvironmentSource:
    if source == UIDetectionSource.ACCESSIBILITY:
        return EnvironmentSource.ACCESSIBILITY

    if source == UIDetectionSource.APP_PARSER:
        return EnvironmentSource.APP_PROFILE

    return EnvironmentSource.VISUAL_DETECTION


def _reason_for_source(source: UIDetectionSource) -> UIDetectionReason:
    if source == UIDetectionSource.ACCESSIBILITY:
        return UIDetectionReason.ACCESSIBILITY_ELEMENTS_DETECTED

    if source == UIDetectionSource.APP_PARSER:
        return UIDetectionReason.APP_ELEMENTS_DETECTED

    return UIDetectionReason.VISUAL_ELEMENTS_DETECTED


def _clean_required(value: str) -> str:
    cleaned = value.strip()

    if not cleaned:
        raise ValueError("field cannot be empty.")

    return cleaned