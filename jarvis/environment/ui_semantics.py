from __future__ import annotations

from enum import StrEnum
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.environment.app_identity import DetectedAppKind
from jarvis.environment.models import (
    EnvironmentSource,
    PrivacyClassification,
    ScreenRegion,
    TrustCalibration,
)
from jarvis.environment.ocr import OCRTextKind, OCRTextRegion
from jarvis.environment.trust_runtime import TrustPolicyClassification
from jarvis.environment.ui_detection import (
    DetectedUIElement,
    DetectedUIElementKind,
)
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class InterfaceKind(StrEnum):
    """
    Semantic interface kind.

    This describes what type of UI JARVIS is looking at.
    """

    CODE_EDITOR = "code_editor"
    TERMINAL = "terminal"
    BROWSER = "browser"
    DOCUMENT = "document"
    FORM = "form"
    DIALOG = "dialog"
    MEDIA_PLAYER = "media_player"
    FILE_PICKER = "file_picker"
    APP_LOADING = "app_loading"
    UNKNOWN = "unknown"


class SemanticSceneKind(StrEnum):
    """
    High-level semantic scene.

    This is what cognition consumes instead of raw pixels.
    """

    CODE_SESSION = "code_session"
    TERMINAL_RUNNING = "terminal_running"
    ERROR_DIALOG = "error_dialog"
    BROWSER_RESEARCH = "browser_research"
    FORM_SENSITIVE = "form_sensitive"
    MEDIA_PLAYER = "media_player"
    FILE_PICKER = "file_picker"
    APP_LOADING = "app_loading"
    CONFIRMATION_DIALOG = "confirmation_dialog"
    UNKNOWN = "unknown"


class ContentClass(StrEnum):
    """
    Classified content type.
    """

    CODE = "code"
    TERMINAL_OUTPUT = "terminal_output"
    ERROR = "error"
    RESEARCH_PAGE = "research_page"
    FORM = "form"
    SENSITIVE_FORM = "sensitive_form"
    MEDIA = "media"
    FILE_SELECTION = "file_selection"
    LOADING = "loading"
    CONFIRMATION = "confirmation"
    UNKNOWN = "unknown"


class SensitivityLevel(StrEnum):
    """
    UI sensitivity classification.
    """

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    SECRET = "secret"


class UISemanticStatus(StrEnum):
    """
    Semantic runtime status.
    """

    UNDERSTOOD = "understood"
    PARTIAL = "partial"
    SENSITIVE_BLOCKED = "sensitive_blocked"
    LOW_TRUST = "low_trust"
    UNKNOWN = "unknown"
    FAILED = "failed"


class UISemanticReason(StrEnum):
    """
    Machine-readable semantic reason.
    """

    SESSION_CREATED = "session_created"
    SCENE_UNDERSTOOD = "scene_understood"
    CONTENT_CLASSIFIED = "content_classified"
    SENSITIVE_UI_DETECTED = "sensitive_ui_detected"
    LOW_TRUST_CONTEXT = "low_trust_context"
    UNKNOWN_SCENE = "unknown_scene"
    SESSION_NOT_FOUND = "session_not_found"
    RUNTIME_RESET = "runtime_reset"


class UISemanticEventKind(StrEnum):
    """
    Runtime event kind.
    """

    SESSION_CREATED = "session_created"
    SEMANTIC_CONTEXT_BUILT = "semantic_context_built"
    SEMANTIC_CONTEXT_BLOCKED = "semantic_context_blocked"
    RUNTIME_RESET = "runtime_reset"


class ContentClassification(OrchestrationModel):
    """
    Content classification result.
    """

    classification_id: str = Field(
        default_factory=lambda: f"content_class_{uuid4().hex}"
    )
    content_class: ContentClass
    interface_kind: InterfaceKind
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    evidence: tuple[str, ...] = ()
    created_at: object = Field(default_factory=utc_now)

    @field_validator("classification_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class SensitiveUIDetection(OrchestrationModel):
    """
    Sensitive UI detection result.
    """

    detection_id: str = Field(default_factory=lambda: f"sensitive_ui_{uuid4().hex}")
    sensitive: bool
    level: SensitivityLevel
    reason: str
    matched_terms: tuple[str, ...] = ()
    privacy: PrivacyClassification = PrivacyClassification.WORKSPACE
    action_blocked: bool = False
    created_at: object = Field(default_factory=utc_now)

    @field_validator("detection_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class SemanticScene(OrchestrationModel):
    """
    Semantic UI scene.

    This is the meaning-level object cognition should consume.
    """

    scene_id: str = Field(default_factory=lambda: f"semantic_scene_{uuid4().hex}")
    kind: SemanticSceneKind
    interface_kind: InterfaceKind
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    region: ScreenRegion | None = None
    trust: TrustCalibration
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("scene_id", "summary")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class UIContextRequest(OrchestrationModel):
    """
    Request to build semantic UI context.

    Inputs come from previous Phase 8 layers:
    OCRRuntime, UIDetectionRuntime, AppIdentityRuntime, WorkspaceGraphRuntime.
    """

    request_id: str = Field(default_factory=lambda: f"ui_context_req_{uuid4().hex}")
    session_id: str
    text_regions: tuple[OCRTextRegion, ...] = ()
    elements: tuple[DetectedUIElement, ...] = ()
    app_kind: DetectedAppKind | None = None
    active_region: ScreenRegion | None = None
    loading: bool = False
    modal_present: bool = False
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("request_id", "session_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class UIContext(OrchestrationModel):
    """
    Final semantic UI context.

    This tells cognition what the UI means and whether it is safe to continue.
    """

    context_id: str = Field(default_factory=lambda: f"ui_context_{uuid4().hex}")
    status: UISemanticStatus
    reason: UISemanticReason
    request_id: str
    scene: SemanticScene
    content: ContentClassification
    sensitive: SensitiveUIDetection
    policy_classification: TrustPolicyClassification
    safe_for_reasoning: bool
    safe_for_action: bool
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("context_id", "request_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _sensitive_blocks_action(self) -> UIContext:
        if self.sensitive.action_blocked and self.safe_for_action:
            raise ValueError("sensitive UI context cannot be safe_for_action.")

        return self


class UISemanticSession(OrchestrationModel):
    """
    UI semantic runtime session.
    """

    session_id: str = Field(default_factory=lambda: f"ui_semantic_{uuid4().hex}")
    workspace_id: str
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class UISemanticRuntimeEvent(OrchestrationModel):
    """
    UI semantic runtime event.
    """

    event_id: str = Field(default_factory=lambda: f"ui_semantic_event_{uuid4().hex}")
    kind: UISemanticEventKind
    reason: UISemanticReason
    session_id: str | None = None
    context_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class UISemanticRuntimeSnapshot(OrchestrationModel):
    """
    Diagnostics for Phase 8 Step 15.
    """

    name: str
    session_count: int = Field(ge=0)
    context_count: int = Field(ge=0)
    sensitive_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    unknown_count: int = Field(ge=0)
    safe_action_count: int = Field(ge=0)
    runtime_event_count: int = Field(ge=0)
    last_reason: UISemanticReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        return _clean_required(value)


class ContentClassifier:
    """
    Classifies UI content from OCR, elements, app identity, and state hints.
    """

    def classify(self, request: UIContextRequest) -> ContentClassification:
        text = _combined_text(request.text_regions)
        lowered = text.lower()
        element_kinds = {element.kind for element in request.elements}

        if request.loading or _contains_any(lowered, ("loading", "please wait")):
            return _classification(
                content_class=ContentClass.LOADING,
                interface_kind=InterfaceKind.APP_LOADING,
                confidence=0.90,
                reason="loading state detected",
                evidence=("loading",),
            )

        if request.modal_present and _contains_any(
            lowered,
            ("delete", "remove", "confirm", "are you sure", "save changes"),
        ):
            return _classification(
                content_class=ContentClass.CONFIRMATION,
                interface_kind=InterfaceKind.DIALOG,
                confidence=0.92,
                reason="confirmation dialog detected",
                evidence=("modal", "confirmation text"),
            )

        if _contains_any(lowered, ("error", "exception", "traceback", "failed")):
            return _classification(
                content_class=ContentClass.ERROR,
                interface_kind=InterfaceKind.DIALOG
                if request.modal_present
                else InterfaceKind.TERMINAL,
                confidence=0.90,
                reason="error content detected",
                evidence=("error keywords",),
            )

        if any(region.kind == OCRTextKind.CODE for region in request.text_regions):
            return _classification(
                content_class=ContentClass.CODE,
                interface_kind=InterfaceKind.CODE_EDITOR,
                confidence=0.90,
                reason="code text region detected",
                evidence=("code OCR",),
            )

        if any(
            region.kind in {OCRTextKind.TERMINAL, OCRTextKind.ERROR}
            for region in request.text_regions
        ):
            return _classification(
                content_class=ContentClass.TERMINAL_OUTPUT,
                interface_kind=InterfaceKind.TERMINAL,
                confidence=0.88,
                reason="terminal text region detected",
                evidence=("terminal OCR",),
            )

        if _looks_sensitive(lowered, element_kinds):
            return _classification(
                content_class=ContentClass.SENSITIVE_FORM,
                interface_kind=InterfaceKind.FORM,
                confidence=0.94,
                reason="sensitive form detected",
                evidence=("sensitive fields",),
            )

        if request.app_kind == DetectedAppKind.BROWSER:
            return _classification(
                content_class=ContentClass.RESEARCH_PAGE,
                interface_kind=InterfaceKind.BROWSER,
                confidence=0.82,
                reason="browser content detected",
                evidence=("browser app",),
            )

        if request.app_kind == DetectedAppKind.MEDIA_APP:
            return _classification(
                content_class=ContentClass.MEDIA,
                interface_kind=InterfaceKind.MEDIA_PLAYER,
                confidence=0.86,
                reason="media app detected",
                evidence=("media app",),
            )

        if DetectedUIElementKind.DOCUMENT in element_kinds:
            return _classification(
                content_class=ContentClass.FILE_SELECTION,
                interface_kind=InterfaceKind.FILE_PICKER,
                confidence=0.80,
                reason="file/document UI element detected",
                evidence=("document element",),
            )

        return _classification(
            content_class=ContentClass.UNKNOWN,
            interface_kind=InterfaceKind.UNKNOWN,
            confidence=0.35,
            reason="semantic scene unknown",
            evidence=(),
        )


class SensitiveUIDetector:
    """
    Detects sensitive UI.

    Sensitive UI is safe for reasoning only in limited form, and never safe for
    blind action.
    """

    _terms = (
        "password",
        "passcode",
        "otp",
        "cvv",
        "card number",
        "credit card",
        "debit card",
        "ssn",
        "aadhaar",
        "token",
        "secret",
        "api key",
        "private key",
        "login",
        "sign in",
        "bank",
        "payment",
    )

    def detect(
        self,
        *,
        request: UIContextRequest,
        classification: ContentClassification,
    ) -> SensitiveUIDetection:
        text = _combined_text(request.text_regions).lower()
        labels = " ".join(
            (element.label or "").lower() for element in request.elements
        )
        combined = f"{text} {labels}"
        matches = tuple(term for term in self._terms if term in combined)

        if classification.content_class == ContentClass.SENSITIVE_FORM:
            matches = tuple(dict.fromkeys((*matches, "sensitive form")))

        if not matches:
            return SensitiveUIDetection(
                sensitive=False,
                level=SensitivityLevel.NONE,
                reason="no sensitive UI detected",
                matched_terms=(),
                privacy=PrivacyClassification.WORKSPACE,
                action_blocked=False,
            )

        level = (
            SensitivityLevel.SECRET
            if any(
                term in matches
                for term in ("password", "otp", "cvv", "api key", "private key")
            )
            else SensitivityLevel.HIGH
        )

        return SensitiveUIDetection(
            sensitive=True,
            level=level,
            reason="sensitive UI detected",
            matched_terms=matches,
            privacy=PrivacyClassification.SECRET,
            action_blocked=True,
        )


class UISemanticRuntime:
    """
    Phase 8 Step 15 UI Semantic Understanding Runtime.

    Responsibilities:
    - classify UI content
    - detect semantic scenes
    - detect sensitive UI
    - build UIContext for cognition
    - prevent blind action in sensitive/unknown/low-trust scenes

    Non-responsibilities:
    - no screen capture
    - no OCR
    - no UI detection
    - no grounding execution
    - no action execution
    """

    def __init__(
        self,
        *,
        name: str = "ui_semantic_runtime",
        classifier: ContentClassifier | None = None,
        sensitive_detector: SensitiveUIDetector | None = None,
    ) -> None:
        cleaned = name.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._classifier = classifier or ContentClassifier()
        self._sensitive_detector = sensitive_detector or SensitiveUIDetector()
        self._sessions: dict[str, UISemanticSession] = {}
        self._contexts: list[UIContext] = []
        self._events: list[UISemanticRuntimeEvent] = []
        self._lock = RLock()
        self._last_reason: UISemanticReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def create_session(
        self,
        *,
        workspace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> UISemanticSession:
        session = UISemanticSession(
            workspace_id=workspace_id,
            metadata=metadata or {},
        )
        event = self._event(
            kind=UISemanticEventKind.SESSION_CREATED,
            reason=UISemanticReason.SESSION_CREATED,
            session_id=session.session_id,
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._events.append(event)
            self._last_reason = event.reason

        return session

    def understand(self, request: UIContextRequest) -> UIContext:
        session = self.session_for(request.session_id)

        if session is None:
            context = _failed_context(request=request)
            self._record_context(context, session_id=request.session_id)
            return context

        content = self._classifier.classify(request)
        sensitive = self._sensitive_detector.detect(
            request=request,
            classification=content,
        )
        scene = _scene_from(
            request=request,
            content=content,
            sensitive=sensitive,
        )
        status = _status_for(content=content, sensitive=sensitive)
        reason = _reason_for(status=status, sensitive=sensitive)
        policy = _policy_for(status=status, sensitive=sensitive, scene=scene)
        safe_for_reasoning = status != UISemanticStatus.FAILED
        safe_for_action = _safe_for_action(
            status=status,
            sensitive=sensitive,
            scene=scene,
            policy=policy,
        )
        context = UIContext(
            status=status,
            reason=reason,
            request_id=request.request_id,
            scene=scene,
            content=content,
            sensitive=sensitive,
            policy_classification=policy,
            safe_for_reasoning=safe_for_reasoning,
            safe_for_action=safe_for_action,
            message=_message_for(scene=scene, sensitive=sensitive),
        )

        self._record_context(context, session_id=request.session_id)
        self._touch_session(request.session_id)

        return context

    def session_for(self, session_id: str) -> UISemanticSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def contexts(self) -> tuple[UIContext, ...]:
        with self._lock:
            return tuple(self._contexts)

    def events(self) -> tuple[UISemanticRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def snapshot(self) -> UISemanticRuntimeSnapshot:
        with self._lock:
            return UISemanticRuntimeSnapshot(
                name=self.name,
                session_count=len(self._sessions),
                context_count=len(self._contexts),
                sensitive_count=sum(
                    1 for context in self._contexts if context.sensitive.sensitive
                ),
                blocked_count=sum(
                    1 for context in self._contexts if not context.safe_for_action
                ),
                unknown_count=sum(
                    1
                    for context in self._contexts
                    if context.scene.kind == SemanticSceneKind.UNKNOWN
                ),
                safe_action_count=sum(
                    1 for context in self._contexts if context.safe_for_action
                ),
                runtime_event_count=len(self._events),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        event = self._event(
            kind=UISemanticEventKind.RUNTIME_RESET,
            reason=UISemanticReason.RUNTIME_RESET,
        )

        with self._lock:
            self._sessions.clear()
            self._contexts.clear()
            self._events.clear()
            self._events.append(event)
            self._last_reason = event.reason

    def _record_context(
        self,
        context: UIContext,
        *,
        session_id: str,
    ) -> None:
        event = self._event(
            kind=(
                UISemanticEventKind.SEMANTIC_CONTEXT_BLOCKED
                if not context.safe_for_action
                else UISemanticEventKind.SEMANTIC_CONTEXT_BUILT
            ),
            reason=context.reason,
            session_id=session_id,
            context_id=context.context_id,
            metadata={
                "scene": context.scene.kind.value,
                "safe_for_action": context.safe_for_action,
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
        kind: UISemanticEventKind,
        reason: UISemanticReason,
        session_id: str | None = None,
        context_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UISemanticRuntimeEvent:
        return UISemanticRuntimeEvent(
            kind=kind,
            reason=reason,
            session_id=session_id,
            context_id=context_id,
            metadata=metadata or {},
        )


def _classification(
    *,
    content_class: ContentClass,
    interface_kind: InterfaceKind,
    confidence: float,
    reason: str,
    evidence: tuple[str, ...],
) -> ContentClassification:
    return ContentClassification(
        content_class=content_class,
        interface_kind=interface_kind,
        confidence=confidence,
        reason=reason,
        evidence=evidence,
    )


def _scene_from(
    *,
    request: UIContextRequest,
    content: ContentClassification,
    sensitive: SensitiveUIDetection,
) -> SemanticScene:
    if sensitive.sensitive:
        kind = SemanticSceneKind.FORM_SENSITIVE
    elif content.content_class == ContentClass.CODE:
        kind = SemanticSceneKind.CODE_SESSION
    elif content.content_class == ContentClass.TERMINAL_OUTPUT:
        kind = SemanticSceneKind.TERMINAL_RUNNING
    elif content.content_class == ContentClass.ERROR:
        kind = (
            SemanticSceneKind.ERROR_DIALOG
            if content.interface_kind == InterfaceKind.DIALOG
            else SemanticSceneKind.TERMINAL_RUNNING
        )
    elif content.content_class == ContentClass.RESEARCH_PAGE:
        kind = SemanticSceneKind.BROWSER_RESEARCH
    elif content.content_class == ContentClass.MEDIA:
        kind = SemanticSceneKind.MEDIA_PLAYER
    elif content.content_class == ContentClass.FILE_SELECTION:
        kind = SemanticSceneKind.FILE_PICKER
    elif content.content_class == ContentClass.LOADING:
        kind = SemanticSceneKind.APP_LOADING
    elif content.content_class == ContentClass.CONFIRMATION:
        kind = SemanticSceneKind.CONFIRMATION_DIALOG
    else:
        kind = SemanticSceneKind.UNKNOWN

    return SemanticScene(
        kind=kind,
        interface_kind=content.interface_kind,
        confidence=content.confidence,
        summary=_summary_for(kind),
        region=request.active_region,
        trust=TrustCalibration(
            confidence=content.confidence,
            stability=max(0.0, min(1.0, content.confidence + 0.04)),
            ambiguity=1.0 - content.confidence,
            source=EnvironmentSource.OS_OBSERVER,
            reason=content.reason,
        ),
    )


def _status_for(
    *,
    content: ContentClassification,
    sensitive: SensitiveUIDetection,
) -> UISemanticStatus:
    if sensitive.sensitive:
        return UISemanticStatus.SENSITIVE_BLOCKED

    if content.confidence < 0.45:
        return UISemanticStatus.UNKNOWN

    if content.confidence < 0.70:
        return UISemanticStatus.LOW_TRUST

    return UISemanticStatus.UNDERSTOOD


def _reason_for(
    *,
    status: UISemanticStatus,
    sensitive: SensitiveUIDetection,
) -> UISemanticReason:
    if sensitive.sensitive:
        return UISemanticReason.SENSITIVE_UI_DETECTED

    if status == UISemanticStatus.UNKNOWN:
        return UISemanticReason.UNKNOWN_SCENE

    if status == UISemanticStatus.LOW_TRUST:
        return UISemanticReason.LOW_TRUST_CONTEXT

    return UISemanticReason.SCENE_UNDERSTOOD


def _policy_for(
    *,
    status: UISemanticStatus,
    sensitive: SensitiveUIDetection,
    scene: SemanticScene,
) -> TrustPolicyClassification:
    if sensitive.sensitive:
        return TrustPolicyClassification.BLOCKED

    if status in {UISemanticStatus.UNKNOWN, UISemanticStatus.LOW_TRUST}:
        return TrustPolicyClassification.REVIEW

    if scene.kind in {
        SemanticSceneKind.CONFIRMATION_DIALOG,
        SemanticSceneKind.ERROR_DIALOG,
        SemanticSceneKind.FILE_PICKER,
    }:
        return TrustPolicyClassification.VERIFY_FIRST

    return TrustPolicyClassification.SAFE


def _safe_for_action(
    *,
    status: UISemanticStatus,
    sensitive: SensitiveUIDetection,
    scene: SemanticScene,
    policy: TrustPolicyClassification,
) -> bool:
    if sensitive.action_blocked:
        return False

    if status != UISemanticStatus.UNDERSTOOD:
        return False

    if policy != TrustPolicyClassification.SAFE:
        return False

    return scene.kind not in {
        SemanticSceneKind.CONFIRMATION_DIALOG,
        SemanticSceneKind.FORM_SENSITIVE,
        SemanticSceneKind.UNKNOWN,
        SemanticSceneKind.APP_LOADING,
    }


def _failed_context(request: UIContextRequest) -> UIContext:
    content = _classification(
        content_class=ContentClass.UNKNOWN,
        interface_kind=InterfaceKind.UNKNOWN,
        confidence=0.0,
        reason="semantic session not found",
        evidence=(),
    )
    sensitive = SensitiveUIDetection(
        sensitive=False,
        level=SensitivityLevel.NONE,
        reason="session missing",
    )
    scene = SemanticScene(
        kind=SemanticSceneKind.UNKNOWN,
        interface_kind=InterfaceKind.UNKNOWN,
        confidence=0.0,
        summary="unknown UI context",
        trust=TrustCalibration(
            confidence=0.0,
            stability=0.0,
            ambiguity=1.0,
            source=EnvironmentSource.OS_OBSERVER,
            reason="semantic session not found",
        ),
    )

    return UIContext(
        status=UISemanticStatus.FAILED,
        reason=UISemanticReason.SESSION_NOT_FOUND,
        request_id=request.request_id,
        scene=scene,
        content=content,
        sensitive=sensitive,
        policy_classification=TrustPolicyClassification.BLOCKED,
        safe_for_reasoning=False,
        safe_for_action=False,
        message="UI semantic session not found",
    )


def _message_for(
    *,
    scene: SemanticScene,
    sensitive: SensitiveUIDetection,
) -> str:
    if sensitive.sensitive:
        return "sensitive UI detected; action blocked"

    return scene.summary


def _summary_for(kind: SemanticSceneKind) -> str:
    summaries = {
        SemanticSceneKind.CODE_SESSION: "code editing session",
        SemanticSceneKind.TERMINAL_RUNNING: "terminal or command output",
        SemanticSceneKind.ERROR_DIALOG: "error dialog requiring attention",
        SemanticSceneKind.BROWSER_RESEARCH: "browser research context",
        SemanticSceneKind.FORM_SENSITIVE: "sensitive form context",
        SemanticSceneKind.MEDIA_PLAYER: "media player context",
        SemanticSceneKind.FILE_PICKER: "file picker context",
        SemanticSceneKind.APP_LOADING: "application loading context",
        SemanticSceneKind.CONFIRMATION_DIALOG: "confirmation dialog context",
        SemanticSceneKind.UNKNOWN: "unknown UI context",
    }

    return summaries[kind]


def _combined_text(text_regions: tuple[OCRTextRegion, ...]) -> str:
    return "\n".join(region.text for region in text_regions)


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _looks_sensitive(
    text: str,
    element_kinds: set[DetectedUIElementKind],
) -> bool:
    if _contains_any(
        text,
        (
            "password",
            "otp",
            "cvv",
            "card number",
            "payment",
            "login",
            "sign in",
            "api key",
            "private key",
        ),
    ):
        return True

    return DetectedUIElementKind.INPUT in element_kinds and _contains_any(
        text,
        ("password", "payment", "card", "otp", "login"),
    )


def _clean_required(value: str) -> str:
    cleaned = value.strip()

    if not cleaned:
        raise ValueError("field cannot be empty.")

    return cleaned