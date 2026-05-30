from __future__ import annotations

from enum import StrEnum
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.environment.app_identity import DetectedAppKind
from jarvis.environment.models import EnvironmentSource, ScreenRegion, TrustCalibration
from jarvis.environment.ocr import OCRTextRegion
from jarvis.environment.trust_runtime import TrustPolicyClassification
from jarvis.environment.ui_detection import DetectedUIElement, DetectedUIElementKind
from jarvis.environment.ui_semantics import (
    SemanticSceneKind,
    UIContext,
)
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class UIPatternKind(StrEnum):
    SAVE_DIALOG = "save_dialog"
    ERROR_DIALOG = "error_dialog"
    CONFIRMATION_DIALOG = "confirmation_dialog"
    LOGIN_FORM = "login_form"
    FILE_PICKER = "file_picker"
    TERMINAL_PROMPT = "terminal_prompt"
    PROGRESS_BAR = "progress_bar"
    LOADING_SPINNER = "loading_spinner"
    VSCODE_COMMAND_PALETTE = "vscode_command_palette"
    CHROME_WARNING_PAGE = "chrome_warning_page"
    UNKNOWN = "unknown"


class UIPatternSource(StrEnum):
    COMMON_LIBRARY = "common_library"
    APP_SPECIFIC_LIBRARY = "app_specific_library"
    DIALOG_RECOGNIZER = "dialog_recognizer"
    PROGRESS_RECOGNIZER = "progress_recognizer"
    TERMINAL_PROMPT_RECOGNIZER = "terminal_prompt_recognizer"
    FALLBACK_MATCHER = "fallback_matcher"


class UIPatternRisk(StrEnum):
    SAFE = "safe"
    VERIFY_FIRST = "verify_first"
    SENSITIVE = "sensitive"
    BLOCKED = "blocked"


class UIPatternStatus(StrEnum):
    MATCHED = "matched"
    PARTIAL = "partial"
    UNKNOWN = "unknown"
    BLOCKED = "blocked"
    FAILED = "failed"


class UIPatternReason(StrEnum):
    SESSION_CREATED = "session_created"
    PATTERN_REGISTERED = "pattern_registered"
    PATTERN_MATCHED = "pattern_matched"
    COMMON_PATTERN_MATCHED = "common_pattern_matched"
    APP_PATTERN_MATCHED = "app_pattern_matched"
    DIALOG_PATTERN_MATCHED = "dialog_pattern_matched"
    PROGRESS_PATTERN_MATCHED = "progress_pattern_matched"
    TERMINAL_PROMPT_MATCHED = "terminal_prompt_matched"
    PATTERN_UNKNOWN = "pattern_unknown"
    PATTERN_BLOCKED = "pattern_blocked"
    SESSION_NOT_FOUND = "session_not_found"
    RUNTIME_RESET = "runtime_reset"


class UIPatternEventKind(StrEnum):
    SESSION_CREATED = "session_created"
    PATTERN_REGISTERED = "pattern_registered"
    MATCH_COMPLETED = "match_completed"
    MATCH_BLOCKED = "match_blocked"
    RUNTIME_RESET = "runtime_reset"


class PatternSignal(OrchestrationModel):
    """
    A signal used to match a UI pattern.

    Signals can come from text, semantic scene, UI elements, app kind,
    region shape, or runtime metadata.
    """

    signal_id: str = Field(default_factory=lambda: f"pattern_signal_{uuid4().hex}")
    name: str
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    required: bool = False
    matched: bool = False
    evidence: str | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("signal_id", "name")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class UIPatternDefinition(OrchestrationModel):
    """
    A reusable known UI pattern definition.

    Common UI patterns should be recognized from this library instead of
    being fully re-analyzed every time.
    """

    pattern_id: str = Field(default_factory=lambda: f"ui_pattern_{uuid4().hex}")
    kind: UIPatternKind
    name: str
    source: UIPatternSource
    app_kind: DetectedAppKind | None = None
    text_terms: tuple[str, ...] = ()
    element_kinds: tuple[DetectedUIElementKind, ...] = ()
    scene_kinds: tuple[SemanticSceneKind, ...] = ()
    minimum_score: float = Field(default=0.65, ge=0.0, le=1.0)
    risk: UIPatternRisk = UIPatternRisk.SAFE
    policy: TrustPolicyClassification = TrustPolicyClassification.SAFE
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("pattern_id", "name")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _must_have_matcher_signal(self) -> UIPatternDefinition:
        if not (self.text_terms or self.element_kinds or self.scene_kinds):
            raise ValueError("pattern requires text, element, or scene signals.")

        return self


class UIPatternMatchRequest(OrchestrationModel):
    """
    Request to recognize patterns in the current UI context.
    """

    request_id: str = Field(default_factory=lambda: f"pattern_req_{uuid4().hex}")
    session_id: str
    ui_context: UIContext | None = None
    text_regions: tuple[OCRTextRegion, ...] = ()
    elements: tuple[DetectedUIElement, ...] = ()
    app_kind: DetectedAppKind | None = None
    active_region: ScreenRegion | None = None
    loading: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: object = Field(default_factory=utc_now)

    @field_validator("request_id", "session_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class UIPatternMatch(OrchestrationModel):
    """
    One matched UI pattern.
    """

    match_id: str = Field(default_factory=lambda: f"pattern_match_{uuid4().hex}")
    pattern: UIPatternDefinition
    score: float = Field(ge=0.0, le=1.0)
    signals: tuple[PatternSignal, ...]
    source: UIPatternSource
    trust: TrustCalibration
    policy: TrustPolicyClassification
    risk: UIPatternRisk
    region: ScreenRegion | None = None
    message: str
    created_at: object = Field(default_factory=utc_now)

    @field_validator("match_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class UIPatternRecognitionResult(OrchestrationModel):
    """
    Pattern recognition result.
    """

    result_id: str = Field(default_factory=lambda: f"pattern_result_{uuid4().hex}")
    status: UIPatternStatus
    reason: UIPatternReason
    request_id: str
    matches: tuple[UIPatternMatch, ...] = ()
    best_match: UIPatternMatch | None = None
    safe_for_action: bool
    message: str
    created_at: object = Field(default_factory=utc_now)

    @field_validator("result_id", "request_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _best_match_must_be_in_matches(self) -> UIPatternRecognitionResult:
        if self.best_match is None:
            return self

        if self.best_match.match_id not in {match.match_id for match in self.matches}:
            raise ValueError("best_match must be present in matches.")

        return self


class PatternLibrarySnapshot(OrchestrationModel):
    """
    Pattern library diagnostics.
    """

    common_pattern_count: int = Field(ge=0)
    app_specific_pattern_count: int = Field(ge=0)
    total_pattern_count: int = Field(ge=0)


class PatternLibrary:
    """
    Pattern library.

    Holds common and app-specific UI patterns.
    """

    def __init__(self) -> None:
        self._common: dict[str, UIPatternDefinition] = {}
        self._app_specific: dict[str, UIPatternDefinition] = {}

    def register_common(self, pattern: UIPatternDefinition) -> None:
        self._common[pattern.pattern_id] = pattern

    def register_app_specific(self, pattern: UIPatternDefinition) -> None:
        self._app_specific[pattern.pattern_id] = pattern

    def patterns_for(
        self,
        *,
        app_kind: DetectedAppKind | None,
    ) -> tuple[UIPatternDefinition, ...]:
        common = tuple(self._common.values())
        app_patterns = tuple(
            pattern
            for pattern in self._app_specific.values()
            if pattern.app_kind is None or pattern.app_kind == app_kind
        )

        return (*common, *app_patterns)

    def snapshot(self) -> PatternLibrarySnapshot:
        return PatternLibrarySnapshot(
            common_pattern_count=len(self._common),
            app_specific_pattern_count=len(self._app_specific),
            total_pattern_count=len(self._common) + len(self._app_specific),
        )


class CommonUIPatterns:
    """
    Factory for common UI pattern definitions.
    """

    @staticmethod
    def build() -> tuple[UIPatternDefinition, ...]:
        return (
            UIPatternDefinition(
                kind=UIPatternKind.SAVE_DIALOG,
                name="Save Dialog",
                source=UIPatternSource.COMMON_LIBRARY,
                text_terms=("save", "don't save", "cancel"),
                scene_kinds=(SemanticSceneKind.CONFIRMATION_DIALOG,),
                minimum_score=0.60,
                risk=UIPatternRisk.VERIFY_FIRST,
                policy=TrustPolicyClassification.VERIFY_FIRST,
            ),
            UIPatternDefinition(
                kind=UIPatternKind.ERROR_DIALOG,
                name="Error Dialog",
                source=UIPatternSource.COMMON_LIBRARY,
                text_terms=("error", "failed", "exception"),
                scene_kinds=(SemanticSceneKind.ERROR_DIALOG,),
                minimum_score=0.60,
                risk=UIPatternRisk.VERIFY_FIRST,
                policy=TrustPolicyClassification.VERIFY_FIRST,
            ),
            UIPatternDefinition(
                kind=UIPatternKind.CONFIRMATION_DIALOG,
                name="Confirmation Dialog",
                source=UIPatternSource.COMMON_LIBRARY,
                text_terms=("are you sure", "confirm", "delete", "remove"),
                scene_kinds=(SemanticSceneKind.CONFIRMATION_DIALOG,),
                minimum_score=0.60,
                risk=UIPatternRisk.VERIFY_FIRST,
                policy=TrustPolicyClassification.VERIFY_FIRST,
            ),
            UIPatternDefinition(
                kind=UIPatternKind.LOGIN_FORM,
                name="Login Form",
                source=UIPatternSource.COMMON_LIBRARY,
                text_terms=("password", "login", "sign in", "otp"),
                element_kinds=(DetectedUIElementKind.INPUT,),
                scene_kinds=(SemanticSceneKind.FORM_SENSITIVE,),
                minimum_score=0.50,
                risk=UIPatternRisk.SENSITIVE,
                policy=TrustPolicyClassification.BLOCKED,
            ),
            UIPatternDefinition(
                kind=UIPatternKind.FILE_PICKER,
                name="File Picker",
                source=UIPatternSource.COMMON_LIBRARY,
                text_terms=("open", "file name", "cancel"),
                scene_kinds=(SemanticSceneKind.FILE_PICKER,),
                minimum_score=0.60,
                risk=UIPatternRisk.VERIFY_FIRST,
                policy=TrustPolicyClassification.VERIFY_FIRST,
            ),
            UIPatternDefinition(
                kind=UIPatternKind.PROGRESS_BAR,
                name="Progress Bar",
                source=UIPatternSource.COMMON_LIBRARY,
                text_terms=("progress", "%", "completed"),
                minimum_score=0.55,
                risk=UIPatternRisk.SAFE,
                policy=TrustPolicyClassification.SAFE,
            ),
            UIPatternDefinition(
                kind=UIPatternKind.LOADING_SPINNER,
                name="Loading Spinner",
                source=UIPatternSource.COMMON_LIBRARY,
                text_terms=("loading", "please wait"),
                scene_kinds=(SemanticSceneKind.APP_LOADING,),
                minimum_score=0.55,
                risk=UIPatternRisk.SAFE,
                policy=TrustPolicyClassification.REVIEW,
            ),
        )


class AppSpecificPatterns:
    """
    Factory for app-specific pattern definitions.
    """

    @staticmethod
    def build() -> tuple[UIPatternDefinition, ...]:
        return (
            UIPatternDefinition(
                kind=UIPatternKind.TERMINAL_PROMPT,
                name="Terminal Prompt",
                source=UIPatternSource.APP_SPECIFIC_LIBRARY,
                app_kind=DetectedAppKind.TERMINAL,
                text_terms=("ps ", ">", "$", "pytest", "npm"),
                scene_kinds=(SemanticSceneKind.TERMINAL_RUNNING,),
                minimum_score=0.55,
                risk=UIPatternRisk.SAFE,
                policy=TrustPolicyClassification.SAFE,
            ),
            UIPatternDefinition(
                kind=UIPatternKind.VSCODE_COMMAND_PALETTE,
                name="VS Code Command Palette",
                source=UIPatternSource.APP_SPECIFIC_LIBRARY,
                app_kind=DetectedAppKind.IDE,
                text_terms=(">", "command palette", "show all commands"),
                element_kinds=(DetectedUIElementKind.INPUT,),
                minimum_score=0.55,
                risk=UIPatternRisk.SAFE,
                policy=TrustPolicyClassification.SAFE,
            ),
            UIPatternDefinition(
                kind=UIPatternKind.CHROME_WARNING_PAGE,
                name="Chrome Warning Page",
                source=UIPatternSource.APP_SPECIFIC_LIBRARY,
                app_kind=DetectedAppKind.BROWSER,
                text_terms=(
                    "your connection is not private",
                    "advanced",
                    "back to safety",
                ),
                minimum_score=0.55,
                risk=UIPatternRisk.BLOCKED,
                policy=TrustPolicyClassification.BLOCKED,
            ),
        )


class PatternMatcher:
    """
    Scores pattern definitions against request evidence.
    """

    def match(
        self,
        *,
        request: UIPatternMatchRequest,
        patterns: tuple[UIPatternDefinition, ...],
    ) -> tuple[UIPatternMatch, ...]:
        matches: list[UIPatternMatch] = []

        for pattern in patterns:
            signals = _signals_for(pattern=pattern, request=request)
            score = _score_signals(signals)

            if score < pattern.minimum_score:
                continue

            matches.append(
                UIPatternMatch(
                    pattern=pattern,
                    score=score,
                    signals=signals,
                    source=pattern.source,
                    trust=TrustCalibration(
                        confidence=score,
                        stability=max(0.0, min(1.0, score + 0.05)),
                        ambiguity=1.0 - score,
                        source=EnvironmentSource.OS_OBSERVER,
                        reason=f"matched {pattern.name}",
                    ),
                    policy=pattern.policy,
                    risk=pattern.risk,
                    region=request.active_region,
                    message=f"matched UI pattern: {pattern.name}",
                )
            )

        return tuple(sorted(matches, key=lambda match: match.score, reverse=True))


class DialogRecognizer:
    """
    Specialized recognizer for dialog patterns.
    """

    def recognize(self, request: UIPatternMatchRequest) -> tuple[UIPatternMatch, ...]:
        text = _request_text(request)
        lowered = text.lower()
        scene = _scene_kind(request)

        if scene not in {
            SemanticSceneKind.CONFIRMATION_DIALOG,
            SemanticSceneKind.ERROR_DIALOG,
        }:
            return ()

        if "error" in lowered or "failed" in lowered:
            pattern = UIPatternDefinition(
                kind=UIPatternKind.ERROR_DIALOG,
                name="Recognized Error Dialog",
                source=UIPatternSource.DIALOG_RECOGNIZER,
                text_terms=("error", "failed"),
                scene_kinds=(SemanticSceneKind.ERROR_DIALOG,),
                minimum_score=0.5,
                risk=UIPatternRisk.VERIFY_FIRST,
                policy=TrustPolicyClassification.VERIFY_FIRST,
            )
            return PatternMatcher().match(request=request, patterns=(pattern,))

        pattern = UIPatternDefinition(
            kind=UIPatternKind.CONFIRMATION_DIALOG,
            name="Recognized Confirmation Dialog",
            source=UIPatternSource.DIALOG_RECOGNIZER,
            text_terms=("confirm", "are you sure", "delete", "save"),
            scene_kinds=(SemanticSceneKind.CONFIRMATION_DIALOG,),
            minimum_score=0.45,
            risk=UIPatternRisk.VERIFY_FIRST,
            policy=TrustPolicyClassification.VERIFY_FIRST,
        )
        return PatternMatcher().match(request=request, patterns=(pattern,))


class ProgressRecognizer:
    """
    Specialized recognizer for progress/loading patterns.
    """

    def recognize(self, request: UIPatternMatchRequest) -> tuple[UIPatternMatch, ...]:
        text = _request_text(request).lower()
        scene = _scene_kind(request)

        if request.loading or scene == SemanticSceneKind.APP_LOADING:
            pattern = UIPatternDefinition(
                kind=UIPatternKind.LOADING_SPINNER,
                name="Recognized Loading Spinner",
                source=UIPatternSource.PROGRESS_RECOGNIZER,
                text_terms=("loading", "please wait"),
                scene_kinds=(SemanticSceneKind.APP_LOADING,),
                minimum_score=0.45,
                risk=UIPatternRisk.SAFE,
                policy=TrustPolicyClassification.REVIEW,
            )
            return PatternMatcher().match(request=request, patterns=(pattern,))

        if "%" in text or "completed" in text or "progress" in text:
            pattern = UIPatternDefinition(
                kind=UIPatternKind.PROGRESS_BAR,
                name="Recognized Progress Bar",
                source=UIPatternSource.PROGRESS_RECOGNIZER,
                text_terms=("%", "completed", "progress"),
                minimum_score=0.45,
                risk=UIPatternRisk.SAFE,
                policy=TrustPolicyClassification.SAFE,
            )
            return PatternMatcher().match(request=request, patterns=(pattern,))

        return ()


class TerminalPromptRecognizer:
    """
    Specialized recognizer for terminal prompts.
    """

    def recognize(self, request: UIPatternMatchRequest) -> tuple[UIPatternMatch, ...]:
        if request.app_kind != DetectedAppKind.TERMINAL:
            return ()

        text = _request_text(request).lower()

        if not any(token in text for token in ("ps ", ">", "$", "pytest", "npm")):
            return ()

        pattern = UIPatternDefinition(
            kind=UIPatternKind.TERMINAL_PROMPT,
            name="Recognized Terminal Prompt",
            source=UIPatternSource.TERMINAL_PROMPT_RECOGNIZER,
            text_terms=("ps ", ">", "$", "pytest", "npm"),
            scene_kinds=(SemanticSceneKind.TERMINAL_RUNNING,),
            minimum_score=0.45,
            risk=UIPatternRisk.SAFE,
            policy=TrustPolicyClassification.SAFE,
        )

        return PatternMatcher().match(request=request, patterns=(pattern,))


class UIPatternSession(OrchestrationModel):
    """
    UI pattern runtime session.
    """

    session_id: str = Field(default_factory=lambda: f"ui_pattern_{uuid4().hex}")
    workspace_id: str
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class UIPatternRuntimeEvent(OrchestrationModel):
    """
    UI pattern recognition event.
    """

    event_id: str = Field(default_factory=lambda: f"ui_pattern_event_{uuid4().hex}")
    kind: UIPatternEventKind
    reason: UIPatternReason
    session_id: str | None = None
    result_id: str | None = None
    pattern_kind: UIPatternKind | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class UIPatternRuntimeSnapshot(OrchestrationModel):
    """
    Diagnostics for Phase 8 Step 16.
    """

    name: str
    session_count: int = Field(ge=0)
    pattern_count: int = Field(ge=0)
    result_count: int = Field(ge=0)
    matched_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    unknown_count: int = Field(ge=0)
    safe_action_count: int = Field(ge=0)
    runtime_event_count: int = Field(ge=0)
    last_reason: UIPatternReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        return _clean_required(value)


class InterfacePatternRecognitionRuntime:
    """
    Phase 8 Step 16 Interface Pattern Recognition Runtime.

    Responsibilities:
    - recognize common UI patterns instantly
    - maintain common/app-specific pattern libraries
    - run specialized recognizers for dialogs/progress/terminal prompts
    - emit policy/risk decisions
    - block sensitive or dangerous patterns

    Non-responsibilities:
    - no screen capture
    - no OCR
    - no action execution
    - no coordinate automation
    """

    def __init__(
        self,
        *,
        name: str = "interface_pattern_recognition_runtime",
        library: PatternLibrary | None = None,
        matcher: PatternMatcher | None = None,
        dialog_recognizer: DialogRecognizer | None = None,
        progress_recognizer: ProgressRecognizer | None = None,
        terminal_prompt_recognizer: TerminalPromptRecognizer | None = None,
        load_defaults: bool = True,
    ) -> None:
        cleaned = name.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._library = library or PatternLibrary()
        self._matcher = matcher or PatternMatcher()
        self._dialog_recognizer = dialog_recognizer or DialogRecognizer()
        self._progress_recognizer = progress_recognizer or ProgressRecognizer()
        self._terminal_prompt_recognizer = (
            terminal_prompt_recognizer or TerminalPromptRecognizer()
        )
        self._sessions: dict[str, UIPatternSession] = {}
        self._results: list[UIPatternRecognitionResult] = []
        self._events: list[UIPatternRuntimeEvent] = []
        self._lock = RLock()
        self._last_reason: UIPatternReason | None = None

        if load_defaults:
            self.load_default_patterns()

    @property
    def name(self) -> str:
        return self._name

    def load_default_patterns(self) -> None:
        for pattern in CommonUIPatterns.build():
            self._library.register_common(pattern)

        for pattern in AppSpecificPatterns.build():
            self._library.register_app_specific(pattern)

    def create_session(
        self,
        *,
        workspace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> UIPatternSession:
        session = UIPatternSession(
            workspace_id=workspace_id,
            metadata=metadata or {},
        )
        event = self._event(
            kind=UIPatternEventKind.SESSION_CREATED,
            reason=UIPatternReason.SESSION_CREATED,
            session_id=session.session_id,
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._events.append(event)
            self._last_reason = event.reason

        return session

    def register_pattern(self, pattern: UIPatternDefinition) -> None:
        if pattern.source == UIPatternSource.APP_SPECIFIC_LIBRARY:
            self._library.register_app_specific(pattern)
        else:
            self._library.register_common(pattern)

        event = self._event(
            kind=UIPatternEventKind.PATTERN_REGISTERED,
            reason=UIPatternReason.PATTERN_REGISTERED,
            pattern_kind=pattern.kind,
        )

        with self._lock:
            self._events.append(event)
            self._last_reason = event.reason

    def recognize(self, request: UIPatternMatchRequest) -> UIPatternRecognitionResult:
        if self.session_for(request.session_id) is None:
            result = UIPatternRecognitionResult(
                status=UIPatternStatus.FAILED,
                reason=UIPatternReason.SESSION_NOT_FOUND,
                request_id=request.request_id,
                safe_for_action=False,
                message="UI pattern session not found",
            )
            self._record_result(result, session_id=request.session_id)
            return result

        patterns = self._library.patterns_for(app_kind=request.app_kind)
        matches = [
            *self._matcher.match(request=request, patterns=patterns),
            *self._dialog_recognizer.recognize(request),
            *self._progress_recognizer.recognize(request),
            *self._terminal_prompt_recognizer.recognize(request),
        ]
        unique = _unique_matches(tuple(matches))
        best = unique[0] if unique else None

        if best is None:
            result = UIPatternRecognitionResult(
                status=UIPatternStatus.UNKNOWN,
                reason=UIPatternReason.PATTERN_UNKNOWN,
                request_id=request.request_id,
                matches=(),
                best_match=None,
                safe_for_action=False,
                message="no known UI pattern recognized",
            )
        else:
            blocked = best.policy == TrustPolicyClassification.BLOCKED
            safe_for_action = _safe_for_action(best)
            result = UIPatternRecognitionResult(
                status=UIPatternStatus.BLOCKED if blocked else UIPatternStatus.MATCHED,
                reason=_reason_for_best(best),
                request_id=request.request_id,
                matches=unique,
                best_match=best,
                safe_for_action=safe_for_action,
                message=best.message,
            )

        self._record_result(result, session_id=request.session_id)
        self._touch_session(request.session_id)

        return result

    def session_for(self, session_id: str) -> UIPatternSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def library_snapshot(self) -> PatternLibrarySnapshot:
        return self._library.snapshot()

    def results(self) -> tuple[UIPatternRecognitionResult, ...]:
        with self._lock:
            return tuple(self._results)

    def events(self) -> tuple[UIPatternRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def snapshot(self) -> UIPatternRuntimeSnapshot:
        library = self._library.snapshot()

        with self._lock:
            return UIPatternRuntimeSnapshot(
                name=self.name,
                session_count=len(self._sessions),
                pattern_count=library.total_pattern_count,
                result_count=len(self._results),
                matched_count=sum(
                    1
                    for result in self._results
                    if result.status == UIPatternStatus.MATCHED
                ),
                blocked_count=sum(
                    1
                    for result in self._results
                    if result.status == UIPatternStatus.BLOCKED
                ),
                unknown_count=sum(
                    1
                    for result in self._results
                    if result.status == UIPatternStatus.UNKNOWN
                ),
                safe_action_count=sum(
                    1 for result in self._results if result.safe_for_action
                ),
                runtime_event_count=len(self._events),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        event = self._event(
            kind=UIPatternEventKind.RUNTIME_RESET,
            reason=UIPatternReason.RUNTIME_RESET,
        )

        with self._lock:
            self._sessions.clear()
            self._results.clear()
            self._events.clear()
            self._events.append(event)
            self._last_reason = event.reason

    def _record_result(
        self,
        result: UIPatternRecognitionResult,
        *,
        session_id: str,
    ) -> None:
        event = self._event(
            kind=(
                UIPatternEventKind.MATCH_BLOCKED
                if result.status in {UIPatternStatus.BLOCKED, UIPatternStatus.FAILED}
                else UIPatternEventKind.MATCH_COMPLETED
            ),
            reason=result.reason,
            session_id=session_id,
            result_id=result.result_id,
            pattern_kind=result.best_match.pattern.kind
            if result.best_match is not None
            else None,
            metadata={"status": result.status.value},
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
        kind: UIPatternEventKind,
        reason: UIPatternReason,
        session_id: str | None = None,
        result_id: str | None = None,
        pattern_kind: UIPatternKind | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UIPatternRuntimeEvent:
        return UIPatternRuntimeEvent(
            kind=kind,
            reason=reason,
            session_id=session_id,
            result_id=result_id,
            pattern_kind=pattern_kind,
            metadata=metadata or {},
        )


def _signals_for(
    *,
    pattern: UIPatternDefinition,
    request: UIPatternMatchRequest,
) -> tuple[PatternSignal, ...]:
    signals: list[PatternSignal] = []
    text = _request_text(request).lower()
    element_kinds = {element.kind for element in request.elements}
    scene = _scene_kind(request)

    for term in pattern.text_terms:
        signals.append(
            PatternSignal(
                name=f"text:{term}",
                weight=0.36,
                required=False,
                matched=term.lower() in text,
                evidence=term if term.lower() in text else None,
            )
        )

    for element_kind in pattern.element_kinds:
        signals.append(
            PatternSignal(
                name=f"element:{element_kind.value}",
                weight=0.34,
                required=False,
                matched=element_kind in element_kinds,
                evidence=element_kind.value if element_kind in element_kinds else None,
            )
        )

    for scene_kind in pattern.scene_kinds:
        signals.append(
            PatternSignal(
                name=f"scene:{scene_kind.value}",
                weight=0.40,
                required=False,
                matched=scene == scene_kind,
                evidence=scene_kind.value if scene == scene_kind else None,
            )
        )

    if pattern.app_kind is not None:
        signals.append(
            PatternSignal(
                name=f"app:{pattern.app_kind.value}",
                weight=0.25,
                required=True,
                matched=request.app_kind == pattern.app_kind,
                evidence=pattern.app_kind.value
                if request.app_kind == pattern.app_kind
                else None,
            )
        )

    return tuple(signals)


def _score_signals(signals: tuple[PatternSignal, ...]) -> float:
    if not signals:
        return 0.0

    required = tuple(signal for signal in signals if signal.required)

    if required and not all(signal.matched for signal in required):
        return 0.0

    total_weight = sum(signal.weight for signal in signals)
    matched_weight = sum(signal.weight for signal in signals if signal.matched)

    if total_weight <= 0.0:
        return 0.0

    return max(0.0, min(1.0, matched_weight / total_weight))


def _unique_matches(
    matches: tuple[UIPatternMatch, ...],
) -> tuple[UIPatternMatch, ...]:
    best_by_kind: dict[UIPatternKind, UIPatternMatch] = {}

    for match in matches:
        existing = best_by_kind.get(match.pattern.kind)

        if existing is None or match.score > existing.score:
            best_by_kind[match.pattern.kind] = match

    return tuple(
        sorted(best_by_kind.values(), key=lambda match: match.score, reverse=True)
    )


def _request_text(request: UIPatternMatchRequest) -> str:
    chunks = [region.text for region in request.text_regions]

    if request.ui_context is not None:
        chunks.append(request.ui_context.scene.summary)
        chunks.extend(request.ui_context.content.evidence)

    return "\n".join(chunks)


def _scene_kind(request: UIPatternMatchRequest) -> SemanticSceneKind | None:
    return request.ui_context.scene.kind if request.ui_context is not None else None


def _reason_for_best(best: UIPatternMatch) -> UIPatternReason:
    if best.policy == TrustPolicyClassification.BLOCKED:
        return UIPatternReason.PATTERN_BLOCKED

    if best.source == UIPatternSource.APP_SPECIFIC_LIBRARY:
        return UIPatternReason.APP_PATTERN_MATCHED

    if best.source == UIPatternSource.DIALOG_RECOGNIZER:
        return UIPatternReason.DIALOG_PATTERN_MATCHED

    if best.source == UIPatternSource.PROGRESS_RECOGNIZER:
        return UIPatternReason.PROGRESS_PATTERN_MATCHED

    if best.source == UIPatternSource.TERMINAL_PROMPT_RECOGNIZER:
        return UIPatternReason.TERMINAL_PROMPT_MATCHED

    return UIPatternReason.COMMON_PATTERN_MATCHED


def _safe_for_action(match: UIPatternMatch) -> bool:
    if match.policy != TrustPolicyClassification.SAFE:
        return False

    if match.risk in {
        UIPatternRisk.BLOCKED,
        UIPatternRisk.SENSITIVE,
        UIPatternRisk.VERIFY_FIRST,
    }:
        return False

    return match.score >= 0.65


def _clean_required(value: str) -> str:
    cleaned = value.strip()

    if not cleaned:
        raise ValueError("field cannot be empty.")

    return cleaned