from __future__ import annotations

from enum import StrEnum
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.environment.models import (
    AppKind,
    AppState,
    EnvironmentSource,
    PrivacyClassification,
    TrustCalibration,
)
from jarvis.environment.state_runtime import (
    AppResponsiveness,
    EnvironmentStateRuntime,
    ModalKind,
    ModalState,
    ProcessState,
    ProcessStatus,
)
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class DetectedAppKind(StrEnum):
    """
    Phase 8 application identity kind.

    This is detection-layer app classification. It does not replace the
    base Step 0 AppKind contract.
    """

    IDE = "ide"
    BROWSER = "browser"
    TERMINAL = "terminal"
    DOCUMENT_EDITOR = "document_editor"
    MEDIA_APP = "media_app"
    SYSTEM_APP = "system_app"
    FILE_MANAGER = "file_manager"
    CHAT_APP = "chat_app"
    UNKNOWN = "unknown"


class AppIdentityConfidenceLevel(StrEnum):
    """
    App identity confidence level.
    """

    VERIFIED = "verified"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class AppRuntimeState(StrEnum):
    """
    Current app runtime state.
    """

    ACTIVE = "active"
    BACKGROUND = "background"
    LOADING = "loading"
    MODAL_BLOCKED = "modal_blocked"
    UNRESPONSIVE = "unresponsive"
    CRASHED = "crashed"
    EXITED = "exited"
    UNKNOWN = "unknown"


class AppInteractionReadiness(StrEnum):
    """
    Whether JARVIS may consider interacting with this app later.
    """

    READY = "ready"
    VERIFY_FIRST = "verify_first"
    ASK_USER = "ask_user"
    BLOCKED = "blocked"


class SpoofRiskLevel(StrEnum):
    """
    App spoofing risk.
    """

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AppIdentityStatus(StrEnum):
    """
    Identity runtime operation status.
    """

    IDENTIFIED = "identified"
    PARTIAL = "partial"
    UNKNOWN = "unknown"
    BLOCKED = "blocked"
    FAILED = "failed"


class AppIdentityReason(StrEnum):
    """
    Machine-readable app identity reason.
    """

    SESSION_CREATED = "session_created"
    APP_PROFILE_REGISTERED = "app_profile_registered"
    APP_IDENTIFIED = "app_identified"
    APP_KIND_CLASSIFIED = "app_kind_classified"
    MODAL_DETECTED = "modal_detected"
    RESPONSIVENESS_CHECKED = "responsiveness_checked"
    SPOOF_HINT_DETECTED = "spoof_hint_detected"
    STATE_MODEL_BUILT = "state_model_built"
    UNKNOWN_APP_BLOCKED = "unknown_app_blocked"
    SESSION_NOT_FOUND = "session_not_found"
    RUNTIME_RESET = "runtime_reset"


class AppIdentityEventKind(StrEnum):
    """
    App identity runtime event kind.
    """

    SESSION_CREATED = "session_created"
    PROFILE_REGISTERED = "profile_registered"
    IDENTITY_COMPLETED = "identity_completed"
    STATE_COMPLETED = "state_completed"
    IDENTITY_BLOCKED = "identity_blocked"
    RUNTIME_RESET = "runtime_reset"


class AppProfile(OrchestrationModel):
    """
    Known app profile.

    This lets JARVIS reason differently about VS Code, browsers, terminals,
    document editors, media apps, and system apps.
    """

    profile_id: str = Field(default_factory=lambda: f"app_profile_{uuid4().hex}")
    canonical_name: str
    kind: DetectedAppKind
    trusted_publishers: tuple[str, ...] = ()
    executable_names: tuple[str, ...] = ()
    window_title_hints: tuple[str, ...] = ()
    process_name_hints: tuple[str, ...] = ()
    safe_for_reading: bool = True
    safe_for_interaction: bool = False
    requires_verification_before_action: bool = True
    privacy: PrivacyClassification = PrivacyClassification.WORKSPACE
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("profile_id", "canonical_name")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _profile_must_have_identity_hints(self) -> AppProfile:
        if not (
            self.executable_names
            or self.window_title_hints
            or self.process_name_hints
        ):
            raise ValueError("app profile requires at least one identity hint.")

        return self


class SpoofDetectionHint(OrchestrationModel):
    """
    Spoof detection evidence.

    Spoof hints never prove maliciousness alone. They reduce trust and force
    verification/user approval.
    """

    hint_id: str = Field(default_factory=lambda: f"spoof_hint_{uuid4().hex}")
    risk: SpoofRiskLevel
    reason: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    created_at: object = Field(default_factory=utc_now)

    @field_validator("hint_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class AppKindClassification(OrchestrationModel):
    """
    Result of app kind classification.
    """

    classification_id: str = Field(
        default_factory=lambda: f"app_classification_{uuid4().hex}"
    )
    kind: DetectedAppKind
    confidence: float = Field(ge=0.0, le=1.0)
    source: EnvironmentSource
    reason: str
    created_at: object = Field(default_factory=utc_now)

    @field_validator("classification_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class AppIdentityObservation(OrchestrationModel):
    """
    Raw app identity observation.

    This can come from OS process APIs, observer events, accessibility,
    browser adapters, terminal adapters, or fake tests.
    """

    observation_id: str = Field(
        default_factory=lambda: f"app_observation_{uuid4().hex}"
    )
    app_id: str
    process_id: str | None = None
    process_name: str | None = None
    executable_path: str | None = None
    window_title: str | None = None
    publisher: str | None = None
    pid: int | None = Field(default=None, ge=0)
    responsive: bool | None = None
    loading: bool = False
    modal_title: str | None = None
    source: EnvironmentSource = EnvironmentSource.OS_OBSERVER
    observed_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("observation_id", "app_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class AppIdentityResult(OrchestrationModel):
    """
    App identity result.
    """

    result_id: str = Field(default_factory=lambda: f"app_identity_{uuid4().hex}")
    status: AppIdentityStatus
    reason: AppIdentityReason
    observation: AppIdentityObservation
    profile: AppProfile | None = None
    classification: AppKindClassification
    confidence_level: AppIdentityConfidenceLevel
    trust: TrustCalibration
    spoof_hints: tuple[SpoofDetectionHint, ...] = ()
    message: str
    created_at: object = Field(default_factory=utc_now)

    @field_validator("result_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @property
    def verified(self) -> bool:
        return self.confidence_level in {
            AppIdentityConfidenceLevel.VERIFIED,
            AppIdentityConfidenceLevel.HIGH,
        } and not any(
            hint.risk in {SpoofRiskLevel.HIGH, SpoofRiskLevel.CRITICAL}
            for hint in self.spoof_hints
        )


class AppStateModel(OrchestrationModel):
    """
    Live app state model used by graph, grounding, simulation, and tools.

    Unknown or spoofed apps are never ready for interaction.
    """

    state_id: str = Field(default_factory=lambda: f"app_state_model_{uuid4().hex}")
    app_state: AppState
    identity: AppIdentityResult
    runtime_state: AppRuntimeState
    responsiveness: AppResponsiveness
    modal: ModalState | None = None
    loading: bool = False
    interaction_readiness: AppInteractionReadiness
    safe_for_reading: bool
    safe_for_interaction: bool
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("state_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class AppIdentitySession(OrchestrationModel):
    """
    App identity runtime session.
    """

    session_id: str = Field(
        default_factory=lambda: f"app_identity_session_{uuid4().hex}"
    )
    workspace_id: str
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class AppIdentityRuntimeEvent(OrchestrationModel):
    """
    App identity runtime event.
    """

    event_id: str = Field(default_factory=lambda: f"app_identity_event_{uuid4().hex}")
    kind: AppIdentityEventKind
    reason: AppIdentityReason
    session_id: str | None = None
    app_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class AppIdentityRuntimeSnapshot(OrchestrationModel):
    """
    Diagnostics for Phase 8 Step 12.
    """

    name: str
    session_count: int = Field(ge=0)
    profile_count: int = Field(ge=0)
    identity_result_count: int = Field(ge=0)
    state_model_count: int = Field(ge=0)
    verified_count: int = Field(ge=0)
    unknown_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    spoof_hint_count: int = Field(ge=0)
    runtime_event_count: int = Field(ge=0)
    last_reason: AppIdentityReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        return _clean_required(value)


class AppKindClassifier:
    """
    Classifies apps from profiles and identity observations.
    """

    def classify(
        self,
        *,
        observation: AppIdentityObservation,
        profiles: tuple[AppProfile, ...],
    ) -> AppKindClassification:
        profile = _match_profile(observation=observation, profiles=profiles)

        if profile is not None:
            return AppKindClassification(
                kind=profile.kind,
                confidence=0.96,
                source=observation.source,
                reason=f"profile matched: {profile.canonical_name}",
            )

        text = " ".join(
            item.lower()
            for item in (
                observation.process_name or "",
                observation.executable_path or "",
                observation.window_title or "",
            )
        )

        if any(token in text for token in ("code", "pycharm", "idea")):
            kind = DetectedAppKind.IDE
            confidence = 0.82
        elif any(token in text for token in ("chrome", "edge", "firefox", "browser")):
            kind = DetectedAppKind.BROWSER
            confidence = 0.82
        elif any(token in text for token in ("terminal", "powershell", "cmd")):
            kind = DetectedAppKind.TERMINAL
            confidence = 0.82
        elif any(token in text for token in ("word", "doc", "pdf", "notepad")):
            kind = DetectedAppKind.DOCUMENT_EDITOR
            confidence = 0.76
        elif any(token in text for token in ("spotify", "vlc", "media")):
            kind = DetectedAppKind.MEDIA_APP
            confidence = 0.74
        elif any(token in text for token in ("explorer", "settings", "taskmgr")):
            kind = DetectedAppKind.SYSTEM_APP
            confidence = 0.74
        else:
            kind = DetectedAppKind.UNKNOWN
            confidence = 0.25

        return AppKindClassification(
            kind=kind,
            confidence=confidence,
            source=observation.source,
            reason="classified from process/window hints",
        )


class ModalDetector:
    """
    Detects modal state from observation.
    """

    def detect(self, observation: AppIdentityObservation) -> ModalState | None:
        if not observation.modal_title:
            return None

        return ModalState(
            app_id=observation.app_id,
            kind=ModalKind.UNKNOWN,
            title=observation.modal_title,
            open=True,
        )


class ResponsiveStateChecker:
    """
    Determines responsiveness/runtime state.
    """

    def check(
        self,
        *,
        observation: AppIdentityObservation,
        process: ProcessState | None = None,
    ) -> tuple[AppResponsiveness, AppRuntimeState]:
        if process is not None:
            if process.status == ProcessStatus.CRASHED:
                return AppResponsiveness.CRASHED, AppRuntimeState.CRASHED

            if process.status == ProcessStatus.UNRESPONSIVE:
                return (
                    AppResponsiveness.UNRESPONSIVE,
                    AppRuntimeState.UNRESPONSIVE,
                )

            if process.status == ProcessStatus.EXITED:
                return AppResponsiveness.UNKNOWN, AppRuntimeState.EXITED

        if observation.responsive is False:
            return AppResponsiveness.UNRESPONSIVE, AppRuntimeState.UNRESPONSIVE

        if observation.loading:
            return AppResponsiveness.SLOW, AppRuntimeState.LOADING

        return AppResponsiveness.RESPONSIVE, AppRuntimeState.ACTIVE


class SpoofDetector:
    """
    Emits spoof risk hints.
    """

    def detect(
        self,
        *,
        observation: AppIdentityObservation,
        profile: AppProfile | None,
        classification: AppKindClassification,
    ) -> tuple[SpoofDetectionHint, ...]:
        hints: list[SpoofDetectionHint] = []

        if classification.kind == DetectedAppKind.UNKNOWN:
            hints.append(
                SpoofDetectionHint(
                    risk=SpoofRiskLevel.MEDIUM,
                    reason="unknown app identity",
                    evidence={"app_id": observation.app_id},
                )
            )

        if profile is not None:
            if observation.publisher and profile.trusted_publishers:
                if observation.publisher not in profile.trusted_publishers:
                    hints.append(
                        SpoofDetectionHint(
                            risk=SpoofRiskLevel.HIGH,
                            reason="publisher does not match trusted profile",
                            evidence={
                                "publisher": observation.publisher,
                                "expected": profile.trusted_publishers,
                            },
                        )
                    )

            if observation.process_name and profile.executable_names:
                expected = {item.lower() for item in profile.executable_names}
                actual = observation.process_name.lower()

                if actual not in expected:
                    hints.append(
                        SpoofDetectionHint(
                            risk=SpoofRiskLevel.HIGH,
                            reason="process name mismatch for matched profile",
                            evidence={
                                "process_name": observation.process_name,
                                "expected": profile.executable_names,
                            },
                        )
                    )

        if observation.executable_path:
            lowered = observation.executable_path.lower()

            if "\\temp\\" in lowered or "/tmp/" in lowered:
                hints.append(
                    SpoofDetectionHint(
                        risk=SpoofRiskLevel.HIGH,
                        reason="app executable launched from temporary path",
                        evidence={"path": observation.executable_path},
                    )
                )

        return tuple(hints)


class AppIdentityRuntime:
    """
    Phase 8 Step 12 Application Identity & State Detection.

    Responsibilities:
    - classify apps as IDE/browser/terminal/document/media/system/unknown
    - register known app profiles
    - detect modal/loading/crashed/unresponsive state
    - emit spoof detection hints
    - build AppStateModel for graph/grounding/simulation
    - block unknown or spoofed apps from interaction readiness

    Non-responsibilities:
    - no screen capture
    - no OCR
    - no clicking/typing
    - no direct tool execution
    """

    def __init__(
        self,
        *,
        name: str = "app_identity_runtime",
        state_runtime: EnvironmentStateRuntime | None = None,
        classifier: AppKindClassifier | None = None,
        modal_detector: ModalDetector | None = None,
        responsiveness_checker: ResponsiveStateChecker | None = None,
        spoof_detector: SpoofDetector | None = None,
    ) -> None:
        cleaned = name.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._state_runtime = state_runtime
        self._classifier = classifier or AppKindClassifier()
        self._modal_detector = modal_detector or ModalDetector()
        self._responsiveness_checker = (
            responsiveness_checker or ResponsiveStateChecker()
        )
        self._spoof_detector = spoof_detector or SpoofDetector()
        self._sessions: dict[str, AppIdentitySession] = {}
        self._profiles: dict[str, AppProfile] = {}
        self._identity_results: list[AppIdentityResult] = []
        self._state_models: list[AppStateModel] = []
        self._events: list[AppIdentityRuntimeEvent] = []
        self._lock = RLock()
        self._last_reason: AppIdentityReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def create_session(
        self,
        *,
        workspace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> AppIdentitySession:
        session = AppIdentitySession(
            workspace_id=workspace_id,
            metadata=metadata or {},
        )
        event = self._event(
            kind=AppIdentityEventKind.SESSION_CREATED,
            reason=AppIdentityReason.SESSION_CREATED,
            session_id=session.session_id,
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._events.append(event)
            self._last_reason = event.reason

        return session

    def register_profile(self, profile: AppProfile) -> None:
        event = self._event(
            kind=AppIdentityEventKind.PROFILE_REGISTERED,
            reason=AppIdentityReason.APP_PROFILE_REGISTERED,
            app_id=profile.canonical_name,
        )

        with self._lock:
            self._profiles[profile.profile_id] = profile
            self._events.append(event)
            self._last_reason = event.reason

    def register_default_profiles(self) -> None:
        for profile in default_app_profiles():
            self.register_profile(profile)

    def identify_app(
        self,
        *,
        session_id: str,
        observation: AppIdentityObservation,
    ) -> AppIdentityResult:
        if self.session_for(session_id) is None:
            result = self._identity_result(
                status=AppIdentityStatus.FAILED,
                reason=AppIdentityReason.SESSION_NOT_FOUND,
                observation=observation,
                profile=None,
                classification=AppKindClassification(
                    kind=DetectedAppKind.UNKNOWN,
                    confidence=0.0,
                    source=observation.source,
                    reason="session not found",
                ),
                spoof_hints=(),
                message="app identity session not found",
            )
            self._record_identity(result, session_id=session_id)

            return result

        profiles = self.profiles()
        profile = _match_profile(observation=observation, profiles=profiles)
        classification = self._classifier.classify(
            observation=observation,
            profiles=profiles,
        )
        spoof_hints = self._spoof_detector.detect(
            observation=observation,
            profile=profile,
            classification=classification,
        )
        status = (
            AppIdentityStatus.UNKNOWN
            if classification.kind == DetectedAppKind.UNKNOWN
            else AppIdentityStatus.IDENTIFIED
        )
        reason = (
            AppIdentityReason.UNKNOWN_APP_BLOCKED
            if status == AppIdentityStatus.UNKNOWN
            else AppIdentityReason.APP_IDENTIFIED
        )
        result = self._identity_result(
            status=status,
            reason=reason,
            observation=observation,
            profile=profile,
            classification=classification,
            spoof_hints=spoof_hints,
            message=(
                "app identity detected"
                if status == AppIdentityStatus.IDENTIFIED
                else "unknown app identity blocked"
            ),
        )
        self._record_identity(result, session_id=session_id)

        return result

    def build_state_model(
        self,
        *,
        session_id: str,
        observation: AppIdentityObservation,
        process: ProcessState | None = None,
    ) -> AppStateModel:
        identity = self.identify_app(
            session_id=session_id,
            observation=observation,
        )

        modal = self._modal_detector.detect(observation)
        responsiveness, runtime_state = self._responsiveness_checker.check(
            observation=observation,
            process=process,
        )

        if modal is not None:
            runtime_state = AppRuntimeState.MODAL_BLOCKED

        app_state = AppState(
            app_id=observation.app_id,
            process_id=observation.process_id or f"process_{observation.app_id}",
            name=_app_name_from_observation(observation, identity),
            kind=_base_app_kind_for(identity.classification.kind),
            responsive=responsiveness == AppResponsiveness.RESPONSIVE,
            trusted_identity=identity.verified,
        )
        readiness = _interaction_readiness(
            identity=identity,
            responsiveness=responsiveness,
            runtime_state=runtime_state,
            profile=identity.profile,
        )
        state_model = AppStateModel(
            app_state=app_state,
            identity=identity,
            runtime_state=runtime_state,
            responsiveness=responsiveness,
            modal=modal,
            loading=observation.loading,
            interaction_readiness=readiness,
            safe_for_reading=_safe_for_reading(identity),
            safe_for_interaction=readiness == AppInteractionReadiness.READY,
        )
        event = self._event(
            kind=AppIdentityEventKind.STATE_COMPLETED,
            reason=AppIdentityReason.STATE_MODEL_BUILT,
            session_id=session_id,
            app_id=observation.app_id,
        )

        with self._lock:
            self._state_models.append(state_model)
            self._events.append(event)
            self._touch_session(session_id)
            self._last_reason = event.reason

        return state_model

    def session_for(self, session_id: str) -> AppIdentitySession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def profiles(self) -> tuple[AppProfile, ...]:
        with self._lock:
            return tuple(self._profiles.values())

    def identity_results(self) -> tuple[AppIdentityResult, ...]:
        with self._lock:
            return tuple(self._identity_results)

    def state_models(self) -> tuple[AppStateModel, ...]:
        with self._lock:
            return tuple(self._state_models)

    def events(self) -> tuple[AppIdentityRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def snapshot(self) -> AppIdentityRuntimeSnapshot:
        with self._lock:
            return AppIdentityRuntimeSnapshot(
                name=self.name,
                session_count=len(self._sessions),
                profile_count=len(self._profiles),
                identity_result_count=len(self._identity_results),
                state_model_count=len(self._state_models),
                verified_count=sum(
                    1 for result in self._identity_results if result.verified
                ),
                unknown_count=sum(
                    1
                    for result in self._identity_results
                    if result.status == AppIdentityStatus.UNKNOWN
                ),
                blocked_count=sum(
                    1
                    for model in self._state_models
                    if model.interaction_readiness == AppInteractionReadiness.BLOCKED
                ),
                spoof_hint_count=sum(
                    len(result.spoof_hints) for result in self._identity_results
                ),
                runtime_event_count=len(self._events),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        event = self._event(
            kind=AppIdentityEventKind.RUNTIME_RESET,
            reason=AppIdentityReason.RUNTIME_RESET,
        )

        with self._lock:
            self._sessions.clear()
            self._profiles.clear()
            self._identity_results.clear()
            self._state_models.clear()
            self._events.clear()
            self._events.append(event)
            self._last_reason = event.reason

    def _identity_result(
        self,
        *,
        status: AppIdentityStatus,
        reason: AppIdentityReason,
        observation: AppIdentityObservation,
        profile: AppProfile | None,
        classification: AppKindClassification,
        spoof_hints: tuple[SpoofDetectionHint, ...],
        message: str,
    ) -> AppIdentityResult:
        confidence = _confidence_level(
            classification=classification,
            spoof_hints=spoof_hints,
        )

        return AppIdentityResult(
            status=status,
            reason=reason,
            observation=observation,
            profile=profile,
            classification=classification,
            confidence_level=confidence,
            trust=_trust_for_identity(
                classification=classification,
                spoof_hints=spoof_hints,
            ),
            spoof_hints=spoof_hints,
            message=message,
        )

    def _record_identity(
        self,
        result: AppIdentityResult,
        *,
        session_id: str,
    ) -> None:
        event = self._event(
            kind=(
                AppIdentityEventKind.IDENTITY_BLOCKED
                if result.status
                in {
                    AppIdentityStatus.UNKNOWN,
                    AppIdentityStatus.BLOCKED,
                }
                else AppIdentityEventKind.IDENTITY_COMPLETED
            ),
            reason=result.reason,
            session_id=session_id,
            app_id=result.observation.app_id,
        )

        with self._lock:
            self._identity_results.append(result)
            self._events.append(event)
            self._touch_session(session_id)
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
        kind: AppIdentityEventKind,
        reason: AppIdentityReason,
        session_id: str | None = None,
        app_id: str | None = None,
    ) -> AppIdentityRuntimeEvent:
        return AppIdentityRuntimeEvent(
            kind=kind,
            reason=reason,
            session_id=session_id,
            app_id=app_id,
        )


def default_app_profiles() -> tuple[AppProfile, ...]:
    return (
        AppProfile(
            canonical_name="Visual Studio Code",
            kind=DetectedAppKind.IDE,
            trusted_publishers=("Microsoft Corporation",),
            executable_names=("code.exe",),
            window_title_hints=("Visual Studio Code", "VS Code", "JARVIS_OS"),
            process_name_hints=("code.exe", "code"),
            safe_for_reading=True,
            safe_for_interaction=True,
            requires_verification_before_action=True,
        ),
        AppProfile(
            canonical_name="Google Chrome",
            kind=DetectedAppKind.BROWSER,
            trusted_publishers=("Google LLC",),
            executable_names=("chrome.exe",),
            window_title_hints=("Chrome",),
            process_name_hints=("chrome.exe", "chrome"),
            safe_for_reading=True,
            safe_for_interaction=False,
            requires_verification_before_action=True,
        ),
        AppProfile(
            canonical_name="Windows Terminal",
            kind=DetectedAppKind.TERMINAL,
            trusted_publishers=("Microsoft Corporation",),
            executable_names=("WindowsTerminal.exe", "powershell.exe", "cmd.exe"),
            window_title_hints=("PowerShell", "Command Prompt", "Terminal"),
            process_name_hints=("WindowsTerminal.exe", "powershell.exe", "cmd.exe"),
            safe_for_reading=True,
            safe_for_interaction=False,
            requires_verification_before_action=True,
        ),
    )


def _match_profile(
    *,
    observation: AppIdentityObservation,
    profiles: tuple[AppProfile, ...],
) -> AppProfile | None:
    process_name = (observation.process_name or "").lower()
    window_title = (observation.window_title or "").lower()

    for profile in profiles:
        process_hints = {item.lower() for item in profile.process_name_hints}
        executable_hints = {item.lower() for item in profile.executable_names}
        title_hints = tuple(item.lower() for item in profile.window_title_hints)

        if process_name and (
            process_name in process_hints or process_name in executable_hints
        ):
            return profile

        if window_title and any(hint in window_title for hint in title_hints):
            return profile

    return None


def _confidence_level(
    *,
    classification: AppKindClassification,
    spoof_hints: tuple[SpoofDetectionHint, ...],
) -> AppIdentityConfidenceLevel:
    if any(hint.risk == SpoofRiskLevel.CRITICAL for hint in spoof_hints):
        return AppIdentityConfidenceLevel.LOW

    if any(hint.risk == SpoofRiskLevel.HIGH for hint in spoof_hints):
        return AppIdentityConfidenceLevel.MEDIUM

    if classification.confidence >= 0.95:
        return AppIdentityConfidenceLevel.VERIFIED

    if classification.confidence >= 0.80:
        return AppIdentityConfidenceLevel.HIGH

    if classification.confidence >= 0.60:
        return AppIdentityConfidenceLevel.MEDIUM

    if classification.confidence > 0.0:
        return AppIdentityConfidenceLevel.LOW

    return AppIdentityConfidenceLevel.UNKNOWN


def _trust_for_identity(
    *,
    classification: AppKindClassification,
    spoof_hints: tuple[SpoofDetectionHint, ...],
) -> TrustCalibration:
    penalty = 0.0

    for hint in spoof_hints:
        if hint.risk == SpoofRiskLevel.MEDIUM:
            penalty += 0.15
        elif hint.risk == SpoofRiskLevel.HIGH:
            penalty += 0.35
        elif hint.risk == SpoofRiskLevel.CRITICAL:
            penalty += 0.65

    confidence = max(0.0, classification.confidence - penalty)
    ambiguity = 1.0 - confidence

    return TrustCalibration(
        confidence=confidence,
        stability=confidence,
        ambiguity=ambiguity,
        source=classification.source,
        reason=classification.reason,
    )


def _interaction_readiness(
    *,
    identity: AppIdentityResult,
    responsiveness: AppResponsiveness,
    runtime_state: AppRuntimeState,
    profile: AppProfile | None,
) -> AppInteractionReadiness:
    if identity.status == AppIdentityStatus.UNKNOWN:
        return AppInteractionReadiness.BLOCKED

    if not identity.verified:
        return AppInteractionReadiness.BLOCKED

    if responsiveness in {
        AppResponsiveness.UNRESPONSIVE,
        AppResponsiveness.CRASHED,
    }:
        return AppInteractionReadiness.BLOCKED

    if runtime_state in {
        AppRuntimeState.CRASHED,
        AppRuntimeState.EXITED,
        AppRuntimeState.UNRESPONSIVE,
    }:
        return AppInteractionReadiness.BLOCKED

    if runtime_state == AppRuntimeState.MODAL_BLOCKED:
        return AppInteractionReadiness.VERIFY_FIRST

    if profile is None:
        return AppInteractionReadiness.VERIFY_FIRST

    if not profile.safe_for_interaction:
        return AppInteractionReadiness.VERIFY_FIRST

    if profile.requires_verification_before_action:
        return AppInteractionReadiness.VERIFY_FIRST

    return AppInteractionReadiness.READY


def _safe_for_reading(identity: AppIdentityResult) -> bool:
    if identity.profile is None:
        return False

    return(
        identity.profile.safe_for_reading
        and identity.status != AppIdentityStatus.UNKNOWN
    )


def _app_name_from_observation(
    observation: AppIdentityObservation,
    identity: AppIdentityResult,
) -> str:
    if identity.profile is not None:
        return identity.profile.canonical_name

    return (
        observation.process_name
        or observation.window_title
        or observation.app_id
    )


def _base_app_kind_for(kind: DetectedAppKind) -> AppKind:
    candidates = (
        kind.name,
        kind.value.upper(),
        "UNKNOWN",
    )

    for candidate in candidates:
        value = getattr(AppKind, candidate, None)

        if isinstance(value, AppKind):
            return value

    return next(iter(AppKind))


def _clean_required(value: str) -> str:
    cleaned = value.strip()

    if not cleaned:
        raise ValueError("field cannot be empty.")

    return cleaned