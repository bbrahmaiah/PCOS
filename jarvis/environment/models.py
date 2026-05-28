from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field, field_validator, model_validator

from jarvis.environment.ids import (
    new_app_id,
    new_delta_id,
    new_display_id,
    new_element_id,
    new_environment_id,
    new_environment_memory_id,
    new_event_id,
    new_intent_id,
    new_interaction_id,
    new_process_id,
    new_recovery_id,
    new_region_id,
    new_simulation_id,
    new_snapshot_id,
    new_trust_id,
    new_verification_id,
    new_window_id,
    new_workflow_id,
)
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class EnvironmentSource(StrEnum):
    """
    Source of environment information.

    JARVIS must know where every observation came from.
    """

    ACCESSIBILITY = "accessibility"
    SCREEN_CAPTURE = "screen_capture"
    OCR = "ocr"
    VISUAL_DETECTION = "visual_detection"
    OS_OBSERVER = "os_observer"
    APP_PROFILE = "app_profile"
    USER_INPUT = "user_input"
    MEMORY = "memory"
    SIMULATION = "simulation"
    VERIFICATION = "verification"


class EnvironmentTrustLevel(StrEnum):
    """
    Trust level derived from confidence, stability, ambiguity, and source.
    """

    UNKNOWN = "unknown"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERIFIED = "verified"


class PrivacyClassification(StrEnum):
    """
    Environment privacy classification.

    This exists before capture/OCR so later systems can enforce privacy at the
    boundary, not after sensitive data has already leaked.
    """

    PUBLIC = "public"
    WORKSPACE = "workspace"
    PERSONAL = "personal"
    SENSITIVE = "sensitive"
    SECRET = "secret"
    BLOCKED = "blocked"


class EnvironmentEventKind(StrEnum):
    """
    Typed environment event kinds.
    """

    WINDOW_FOCUSED = "window_focused"
    WINDOW_OPENED = "window_opened"
    WINDOW_CLOSED = "window_closed"
    WINDOW_MOVED = "window_moved"
    APP_STARTED = "app_started"
    APP_EXITED = "app_exited"
    APP_CRASHED = "app_crashed"
    DISPLAY_CHANGED = "display_changed"
    CURSOR_MOVED = "cursor_moved"
    CLIPBOARD_CHANGED = "clipboard_changed"
    FILE_CHANGED = "file_changed"
    MODAL_OPENED = "modal_opened"
    MODAL_CLOSED = "modal_closed"
    UI_CHANGED = "ui_changed"
    WORKFLOW_CHANGED = "workflow_changed"
    VERIFICATION_COMPLETED = "verification_completed"
    RECOVERY_REQUESTED = "recovery_requested"


class DisplayKind(StrEnum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    VIRTUAL = "virtual"
    UNKNOWN = "unknown"


class AppKind(StrEnum):
    IDE = "ide"
    BROWSER = "browser"
    TERMINAL = "terminal"
    DOCUMENT_EDITOR = "document_editor"
    FILE_EXPLORER = "file_explorer"
    MEDIA = "media"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class WindowMode(StrEnum):
    NORMAL = "normal"
    MINIMIZED = "minimized"
    MAXIMIZED = "maximized"
    FULLSCREEN = "fullscreen"
    HIDDEN = "hidden"
    UNKNOWN = "unknown"


class UIElementKind(StrEnum):
    BUTTON = "button"
    TEXT_FIELD = "text_field"
    MENU = "menu"
    MENU_ITEM = "menu_item"
    TAB = "tab"
    LINK = "link"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    SCROLLBAR = "scrollbar"
    CODE_EDITOR = "code_editor"
    TERMINAL = "terminal"
    BROWSER_VIEW = "browser_view"
    DIALOG = "dialog"
    LABEL = "label"
    ICON = "icon"
    LIST_ITEM = "list_item"
    UNKNOWN = "unknown"


class TextRegionKind(StrEnum):
    CODE = "code"
    PROSE = "prose"
    LABEL = "label"
    NUMBER = "number"
    URL = "url"
    TERMINAL = "terminal"
    ERROR = "error"
    UNKNOWN = "unknown"


class InteractionKind(StrEnum):
    OBSERVE = "observe"
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    TYPE_TEXT = "type_text"
    KEY_PRESS = "key_press"
    SCROLL = "scroll"
    DRAG = "drag"
    OPEN_APP = "open_app"
    CLOSE_APP = "close_app"
    FOCUS_WINDOW = "focus_window"
    PASTE = "paste"
    COPY = "copy"


class InteractionRisk(StrEnum):
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    BLOCKED = "blocked"


class VerificationStatus(StrEnum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class RecoveryStrategy(StrEnum):
    NONE = "none"
    RETRY_SAME = "retry_same"
    RETRY_ADJUSTED = "retry_adjusted"
    FIND_ALTERNATIVE = "find_alternative"
    PARTIAL_SUCCESS = "partial_success"
    ROLLBACK = "rollback"
    ESCALATE_TO_USER = "escalate_to_user"


class IntentStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    BLOCKED = "blocked"
    PARTIAL = "partial"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class SimulationStatus(StrEnum):
    NOT_RUN = "not_run"
    PREDICTED = "predicted"
    LOW_CONFIDENCE = "low_confidence"
    UNSAFE = "unsafe"
    FAILED = "failed"


class EnvironmentModel(OrchestrationModel):
    """
    Base model for Phase 8 environment contracts.
    """


class TrustCalibration(EnvironmentModel):
    """
    Standard trust object emitted by all Phase 8 perception, grounding,
    simulation, action, and verification systems.
    """

    trust_id: str = Field(default_factory=new_trust_id)
    confidence: float = Field(ge=0.0, le=1.0)
    stability: float = Field(default=1.0, ge=0.0, le=1.0)
    ambiguity: float = Field(default=0.0, ge=0.0, le=1.0)
    source: EnvironmentSource
    level: EnvironmentTrustLevel = EnvironmentTrustLevel.UNKNOWN
    reason: str
    observed_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("trust_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="before")
    @classmethod
    def _derive_level(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        existing_level = data.get("level")

        if existing_level not in (None, EnvironmentTrustLevel.UNKNOWN, "unknown"):
            return data

        confidence = float(data.get("confidence", 0.0))
        stability = float(data.get("stability", 1.0))
        ambiguity = float(data.get("ambiguity", 0.0))
        score = max(0.0, min(1.0, confidence * stability * (1.0 - ambiguity)))

        if score >= 0.92:
            data["level"] = EnvironmentTrustLevel.VERIFIED
        elif score >= 0.80:
            data["level"] = EnvironmentTrustLevel.HIGH
        elif score >= 0.60:
            data["level"] = EnvironmentTrustLevel.MEDIUM
        else:
            data["level"] = EnvironmentTrustLevel.LOW

        return data

    def effective_score(self) -> float:
        score = self.confidence * self.stability * (1.0 - self.ambiguity)

        return max(0.0, min(1.0, score))


class TrustScore(EnvironmentModel):
    """
    Explicit trust score contract.

    TrustScore is the normalized score used by grounding, perception,
    simulation, verification, and action policy.
    """

    score: float = Field(ge=0.0, le=1.0)
    level: EnvironmentTrustLevel
    confidence: float = Field(ge=0.0, le=1.0)
    stability: float = Field(ge=0.0, le=1.0)
    ambiguity: float = Field(ge=0.0, le=1.0)
    source: EnvironmentSource
    reason: str

    @field_validator("reason")
    @classmethod
    def _required_reason(cls, value: str) -> str:
        return _clean_required(value)


class VisualConfidence(EnvironmentModel):
    """
    Visual confidence contract.

    This separates visual perception confidence from general trust.
    OCR, capture, UI detection, and visual grounding can emit this.
    """

    confidence: float = Field(ge=0.0, le=1.0)
    stability: float = Field(default=1.0, ge=0.0, le=1.0)
    ambiguity: float = Field(default=0.0, ge=0.0, le=1.0)
    source: EnvironmentSource
    region: ScreenRegion | None = None
    explanation: str

    @field_validator("explanation")
    @classmethod
    def _required_explanation(cls, value: str) -> str:
        return _clean_required(value)

    def trust_score(self) -> float:
        score = self.confidence * self.stability * (1.0 - self.ambiguity)

        return max(0.0, min(1.0, score))


class ScreenPoint(EnvironmentModel):
    """
    Typed screen point.

    Raw tuples must never cross Phase 8 boundaries.
    """

    x: int
    y: int


class ScreenRegion(EnvironmentModel):
    """
    Typed screen region.

    All visual coordinates must be wrapped in ScreenRegion.
    """

    region_id: str = Field(default_factory=new_region_id)
    x: int
    y: int
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    scale_factor: float = Field(default=1.0, gt=0)
    display_id: str | None = None

    @field_validator("region_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return self.width * self.height

    def contains_point(self, point: ScreenPoint) -> bool:
        return self.x <= point.x <= self.right and self.y <= point.y <= self.bottom


class DisplayState(EnvironmentModel):
    display_id: str = Field(default_factory=new_display_id)
    kind: DisplayKind = DisplayKind.UNKNOWN
    bounds: ScreenRegion
    scale_factor: float = Field(default=1.0, gt=0)
    primary: bool = False
    active: bool = True

    @field_validator("display_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class AppState(EnvironmentModel):
    app_id: str = Field(default_factory=new_app_id)
    process_id: str = Field(default_factory=new_process_id)
    name: str
    kind: AppKind = AppKind.UNKNOWN
    executable_path: str | None = None
    responsive: bool = True
    trusted_identity: bool = False
    privacy_classification: PrivacyClassification = PrivacyClassification.WORKSPACE

    @field_validator("app_id", "process_id", "name")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class WindowState(EnvironmentModel):
    window_id: str = Field(default_factory=new_window_id)
    app_id: str
    title: str
    bounds: ScreenRegion
    mode: WindowMode = WindowMode.NORMAL
    focused: bool = False
    visible: bool = True
    modal: bool = False
    responsive: bool = True
    trust: TrustCalibration

    @field_validator("window_id", "app_id", "title")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class TextRegion(EnvironmentModel):
    text: str
    bounds: ScreenRegion
    kind: TextRegionKind = TextRegionKind.UNKNOWN
    trust: TrustCalibration
    privacy_classification: PrivacyClassification = PrivacyClassification.WORKSPACE

    @field_validator("text")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class UIElement(EnvironmentModel):
    """
    Semantic UI target.

    JARVIS must interact with UIElement, not raw x/y coordinates.
    """

    element_id: str = Field(default_factory=new_element_id)
    kind: UIElementKind
    bounds: ScreenRegion
    text: str | None = None
    app_id: str | None = None
    window_id: str | None = None
    interactive: bool = False
    enabled: bool = True
    visible: bool = True
    trust: TrustCalibration
    privacy_classification: PrivacyClassification = PrivacyClassification.WORKSPACE
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("element_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _interactive_elements_need_high_trust(self) -> UIElement:
        if not self.interactive:
            return self

        if self.trust.effective_score() < 0.60:
            raise ValueError("interactive UIElement requires trust >= 0.60.")

        return self


class EnvironmentSnapshot(EnvironmentModel):
    """
    Point-in-time environment state evidence.

    It is not the whole truth. It is one observed snapshot.
    """

    snapshot_id: str = Field(default_factory=new_snapshot_id)
    environment_id: str = Field(default_factory=new_environment_id)
    captured_at: object = Field(default_factory=utc_now)
    displays: tuple[DisplayState, ...] = ()
    apps: tuple[AppState, ...] = ()
    windows: tuple[WindowState, ...] = ()
    elements: tuple[UIElement, ...] = ()
    text_regions: tuple[TextRegion, ...] = ()
    focused_window_id: str | None = None
    focused_app_id: str | None = None
    cursor_position: ScreenPoint | None = None
    trust: TrustCalibration

    @field_validator("snapshot_id", "environment_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class RecentStateHistory(EnvironmentModel):
    """
    Recent temporal environment history.

    This is the short rolling history used for recovery, interruption,
    debugging, and workflow continuity.
    """

    history_id: str = Field(default_factory=new_workflow_id)
    snapshots: tuple[EnvironmentSnapshot, ...] = ()
    deltas: tuple[EnvironmentDelta, ...] = ()
    max_items: int = Field(default=20, ge=1)
    created_at: object = Field(default_factory=utc_now)

    @field_validator("history_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _history_must_fit_limit(self) -> RecentStateHistory:
        if len(self.snapshots) > self.max_items:
            raise ValueError("snapshots exceed max_items.")

        if len(self.deltas) > self.max_items:
            raise ValueError("deltas exceed max_items.")

        return self


class EnvironmentState(EnvironmentModel):
    """
    Current live environment state.

    EnvironmentSnapshot is point-in-time evidence.
    EnvironmentState is the current interpreted world model.
    """

    environment_id: str = Field(default_factory=new_environment_id)
    current_snapshot: EnvironmentSnapshot
    recent_history: RecentStateHistory | None = None
    focused_app_id: str | None = None
    focused_window_id: str | None = None
    active_workflow_id: str | None = None
    trust: TrustCalibration
    updated_at: object = Field(default_factory=utc_now)

    @field_validator("environment_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class EnvironmentDelta(EnvironmentModel):
    """
    Temporal change between snapshots.
    """

    delta_id: str = Field(default_factory=new_delta_id)
    previous_snapshot_id: str | None = None
    current_snapshot_id: str
    changed_windows: tuple[str, ...] = ()
    changed_elements: tuple[str, ...] = ()
    appeared_elements: tuple[str, ...] = ()
    disappeared_elements: tuple[str, ...] = ()
    changed_text_regions: tuple[str, ...] = ()
    cause_hint: str | None = None
    observed_at: object = Field(default_factory=utc_now)
    trust: TrustCalibration

    @field_validator("delta_id", "current_snapshot_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class TemporalWorkspaceState(EnvironmentModel):
    """
    Environment state across time.

    This is the foundation for recovery, debugging, and workflow continuity.
    """

    workflow_id: str = Field(default_factory=new_workflow_id)
    current_snapshot: EnvironmentSnapshot
    recent_deltas: tuple[EnvironmentDelta, ...] = ()
    active_since: object = Field(default_factory=utc_now)
    last_user_action: str | None = None
    last_jarvis_action: str | None = None

    @field_validator("workflow_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class EnvironmentEvent(EnvironmentModel):
    event_id: str = Field(default_factory=new_event_id)
    kind: EnvironmentEventKind
    source: EnvironmentSource
    observed_at: object = Field(default_factory=utc_now)
    app_id: str | None = None
    window_id: str | None = None
    element_id: str | None = None
    region: ScreenRegion | None = None
    trust: TrustCalibration
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class GroundingResult(EnvironmentModel):
    """
    Natural language to environment target resolution.
    """

    query: str
    candidates: tuple[UIElement, ...] = ()
    selected_element: UIElement | None = None
    trust: TrustCalibration
    ambiguous: bool = False
    explanation: str

    @field_validator("query", "explanation")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _selected_element_must_be_candidate(self) -> GroundingResult:
        if self.selected_element is None:
            return self

        candidate_ids = {candidate.element_id for candidate in self.candidates}

        if self.selected_element.element_id not in candidate_ids:
            raise ValueError("selected_element must be present in candidates.")

        return self


class IntentState(EnvironmentModel):
    """
    Persistent user intent.

    This survives interruption, pause, blocking, partial success, and resume.
    """

    intent_id: str = Field(default_factory=new_intent_id)
    user_goal: str
    status: IntentStatus = IntentStatus.ACTIVE
    active_subgoal: str | None = None
    blocked_reason: str | None = None
    partial_completion: str | None = None
    resume_token: str | None = None
    workflow_id: str | None = None
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("intent_id", "user_goal")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class SimulationResult(EnvironmentModel):
    """
    Predicted outcome before touching the real desktop.
    """

    simulation_id: str = Field(default_factory=new_simulation_id)
    status: SimulationStatus
    predicted_state_summary: str
    expected_delta: EnvironmentDelta | None = None
    rollback_risk: InteractionRisk = InteractionRisk.LOW
    trust: TrustCalibration
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("simulation_id", "predicted_state_summary")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class InteractionRequest(EnvironmentModel):
    """
    Physical/environment interaction request.

    This is a contract, not execution.
    """

    interaction_id: str = Field(default_factory=new_interaction_id)
    kind: InteractionKind
    risk: InteractionRisk
    target_element: UIElement | None = None
    target_region: ScreenRegion | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    simulation: SimulationResult | None = None
    policy_required: bool = True
    reversible: bool = False
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("interaction_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _target_required_for_physical_interaction(self) -> InteractionRequest:
        if self.kind == InteractionKind.OBSERVE:
            return self

        if self.target_element is None and self.target_region is None:
            raise ValueError(
                "physical interaction requires target_element or target_region."
            )

        if self.target_element is not None and self.risk != InteractionRisk.SAFE:
            if self.target_element.trust.effective_score() < 0.75:
                raise ValueError("non-safe interaction requires target trust >= 0.75.")

        return self


class VerificationResult(EnvironmentModel):
    verification_id: str = Field(default_factory=new_verification_id)
    status: VerificationStatus
    expected_summary: str
    observed_summary: str
    matched: bool
    confidence: float = Field(ge=0.0, le=1.0)
    trust: TrustCalibration
    delta: EnvironmentDelta | None = None
    recovery_needed: bool = False
    verified_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("verification_id", "expected_summary", "observed_summary")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _matched_requires_confirmed_status(self) -> VerificationResult:
        if self.matched and self.status != VerificationStatus.CONFIRMED:
            raise ValueError("matched=True requires CONFIRMED verification status.")

        return self


class RecoveryPlan(EnvironmentModel):
    recovery_id: str = Field(default_factory=new_recovery_id)
    strategy: RecoveryStrategy
    reason: str
    reversible: bool
    requires_user: bool = False
    retry_count: int = Field(default=0, ge=0)
    max_retries: int = Field(default=3, ge=0)
    rollback_interaction_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("recovery_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _retry_count_cannot_exceed_max(self) -> RecoveryPlan:
        if self.retry_count > self.max_retries:
            raise ValueError("retry_count cannot exceed max_retries.")

        return self


class WorkspaceMemoryEntry(EnvironmentModel):
    """
    Semantic workflow memory.

    This is not a screenshot. This is what JARVIS remembers about work.
    """

    memory_id: str = Field(default_factory=new_environment_memory_id)
    workflow_id: str
    app_id: str | None = None
    project_path: str | None = None
    active_files: tuple[str, ...] = ()
    cursor_positions: dict[str, int] = Field(default_factory=dict)
    terminal_directory: str | None = None
    recent_commands: tuple[str, ...] = ()
    visible_errors: tuple[str, ...] = ()
    pending_todos: tuple[str, ...] = ()
    workflow_stage: str | None = None
    privacy_classification: PrivacyClassification = PrivacyClassification.WORKSPACE
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("memory_id", "workflow_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _blocked_memory_cannot_be_stored(self) -> WorkspaceMemoryEntry:
        if self.privacy_classification == PrivacyClassification.BLOCKED:
            raise ValueError("blocked environment memory cannot be stored.")

        return self


class PrivacyZone(EnvironmentModel):
    """
    Capture/OCR exclusion zone.

    Privacy zones are enforced before perception.
    """

    zone_id: str = Field(default_factory=new_region_id)
    name: str
    region: ScreenRegion | None = None
    app_name: str | None = None
    url_pattern: str | None = None
    capture_allowed: bool = False
    ocr_allowed: bool = False
    reason: str

    @field_validator("zone_id", "name", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _must_define_scope(self) -> PrivacyZone:
        if self.region is None and self.app_name is None and self.url_pattern is None:
            raise ValueError("privacy zone requires region, app_name, or url_pattern.")

        return self


def _clean_required(value: str) -> str:
    cleaned = value.strip()

    if not cleaned:
        raise ValueError("field cannot be empty.")

    return cleaned