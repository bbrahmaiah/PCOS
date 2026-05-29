from __future__ import annotations

from enum import StrEnum
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.environment.models import (
    EnvironmentSource,
    PrivacyZone,
    ScreenRegion,
    TrustCalibration,
)
from jarvis.environment.state_runtime import EnvironmentStateRuntime
from jarvis.environment.visual_priority import (
    EnvironmentBackpressureController,
    VisualLoadLevel,
)
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class InspectionDepth(StrEnum):
    """
    Depth of environment inspection.

    This controls how much visual cognition is allowed.
    """

    NONE = "none"
    PERIPHERAL = "peripheral"
    AMBIENT = "ambient"
    FOCUSED = "focused"
    DEEP = "deep"


class EnvironmentAttentionMode(StrEnum):
    """
    Active attention mode.

    PERIPHERAL: cheap change detection only.
    AMBIENT: focused app/workflow awareness.
    FOCUSED: active region parsing.
    DEEP: full semantic inspection.
    """

    PERIPHERAL = "peripheral"
    AMBIENT = "ambient"
    FOCUSED = "focused"
    DEEP = "deep"


class FocusRegionKind(StrEnum):
    """
    Why a focus region exists.
    """

    ACTIVE_WINDOW = "active_window"
    USER_SELECTED = "user_selected"
    ACTION_TARGET = "action_target"
    MODAL = "modal"
    ERROR_REGION = "error_region"
    WORKSPACE = "workspace"
    PRIVACY_EXCLUSION = "privacy_exclusion"


class CapturePermission(StrEnum):
    """
    Capture permission decision.
    """

    ALLOW = "allow"
    ALLOW_LIMITED = "allow_limited"
    DEFER = "defer"
    BLOCK = "block"


class PrivacyZoneDecision(StrEnum):
    """
    Privacy decision for a region.
    """

    ALLOWED = "allowed"
    REDACT = "redact"
    OCR_BLOCKED = "ocr_blocked"
    CAPTURE_BLOCKED = "capture_blocked"


class AttentionGovernanceReason(StrEnum):
    """
    Machine-readable reason.
    """

    SESSION_CREATED = "session_created"
    POLICY_UPDATED = "policy_updated"
    FOCUS_REGION_ADDED = "focus_region_added"
    FOCUS_REGION_REMOVED = "focus_region_removed"
    PERIPHERAL_AWARENESS_UPDATED = "peripheral_awareness_updated"
    ATTENTION_MODE_CHANGED = "attention_mode_changed"
    CAPTURE_ALLOWED = "capture_allowed"
    CAPTURE_LIMITED = "capture_limited"
    CAPTURE_DEFERRED = "capture_deferred"
    CAPTURE_BLOCKED_BY_PRIVACY = "capture_blocked_by_privacy"
    CAPTURE_BLOCKED_BY_LOAD = "capture_blocked_by_load"
    CAPTURE_BLOCKED_BY_DEPTH = "capture_blocked_by_depth"
    SESSION_NOT_FOUND = "session_not_found"
    RUNTIME_RESET = "runtime_reset"


class AttentionGovernanceEventKind(StrEnum):
    """
    Runtime event kind.
    """

    SESSION_CREATED = "session_created"
    POLICY_UPDATED = "policy_updated"
    FOCUS_REGION_CHANGED = "focus_region_changed"
    AWARENESS_UPDATED = "awareness_updated"
    ATTENTION_CHANGED = "attention_changed"
    CAPTURE_DECIDED = "capture_decided"
    RUNTIME_RESET = "runtime_reset"


class FocusRegion(OrchestrationModel):
    """
    Governed focus region.

    Capture runtime later consumes FocusRegion instead of arbitrary raw bounds.
    """

    focus_id: str = Field(default_factory=lambda: f"focus_{uuid4().hex}")
    kind: FocusRegionKind
    region: ScreenRegion
    priority: int = Field(default=50, ge=0, le=100)
    inspection_depth: InspectionDepth
    source: EnvironmentSource = EnvironmentSource.OS_OBSERVER
    trust: TrustCalibration
    active: bool = True
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("focus_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class PeripheralAwareness(OrchestrationModel):
    """
    Cheap awareness state.

    This lets JARVIS notice important changes without deep inspection.
    """

    active_window_changed: bool = False
    display_changed: bool = False
    cursor_moved_significantly: bool = False
    clipboard_changed: bool = False
    modal_present: bool = False
    app_crashed: bool = False
    last_change_summary: str | None = None
    updated_at: object = Field(default_factory=utc_now)

    def has_important_change(self) -> bool:
        return any(
            (
                self.active_window_changed,
                self.display_changed,
                self.cursor_moved_significantly,
                self.clipboard_changed,
                self.modal_present,
                self.app_crashed,
            )
        )


class VisualAttentionPolicy(OrchestrationModel):
    """
    Main visual attention policy.

    This defines the maximum allowed depth and whether visual inspection can
    escalate under active user/cognitive context.
    """

    default_mode: EnvironmentAttentionMode = EnvironmentAttentionMode.PERIPHERAL
    max_depth: InspectionDepth = InspectionDepth.FOCUSED
    allow_deep_inspection: bool = False
    require_user_intent_for_deep: bool = True
    allow_background_capture: bool = False
    protect_conversation_latency: bool = True

    @model_validator(mode="after")
    def _deep_requires_explicit_permission(self) -> VisualAttentionPolicy:
        if self.max_depth == InspectionDepth.DEEP and not self.allow_deep_inspection:
            raise ValueError("DEEP max_depth requires allow_deep_inspection=True.")

        return self


class CaptureFrequencyPolicy(OrchestrationModel):
    """
    Capture frequency limits by attention level.
    """

    peripheral_hz: float = Field(default=0.5, gt=0)
    ambient_hz: float = Field(default=1.0, gt=0)
    focused_hz: float = Field(default=5.0, gt=0)
    deep_hz: float = Field(default=10.0, gt=0)
    max_burst_hz: float = Field(default=15.0, gt=0)

    @model_validator(mode="after")
    def _frequencies_must_increase_with_depth(self) -> CaptureFrequencyPolicy:
        if not (
            self.peripheral_hz
            <= self.ambient_hz
            <= self.focused_hz
            <= self.deep_hz
            <= self.max_burst_hz
        ):
            raise ValueError("capture frequencies must increase with depth.")

        return self

    def frequency_for(self, depth: InspectionDepth) -> float:
        if depth == InspectionDepth.PERIPHERAL:
            return self.peripheral_hz

        if depth == InspectionDepth.AMBIENT:
            return self.ambient_hz

        if depth == InspectionDepth.FOCUSED:
            return self.focused_hz

        if depth == InspectionDepth.DEEP:
            return self.deep_hz

        return 0.0


class PrivacyZonePolicy(OrchestrationModel):
    """
    Privacy enforcement policy.

    Privacy applies before capture/OCR.
    """

    zones: tuple[PrivacyZone, ...] = ()
    block_secret_by_default: bool = True
    block_unknown_sensitive_apps: bool = True
    allow_redaction: bool = True

    def decision_for(
        self,
        *,
        region: ScreenRegion,
        app_name: str | None = None,
        url: str | None = None,
    ) -> PrivacyZoneDecision:
        for zone in self.zones:
            if zone.app_name and app_name and zone.app_name.lower() in app_name.lower():
                return _decision_from_zone(zone)

            if zone.url_pattern and url and zone.url_pattern in url:
                return _decision_from_zone(zone)

            if zone.region and _regions_overlap(zone.region, region):
                return _decision_from_zone(zone)

        return PrivacyZoneDecision.ALLOWED


class CaptureGovernanceRequest(OrchestrationModel):
    """
    Request to inspect/capture a region.

    This is a governance decision, not capture execution.
    """

    request_id: str = Field(default_factory=lambda: f"capture_request_{uuid4().hex}")
    session_id: str
    region: ScreenRegion
    requested_depth: InspectionDepth
    reason: str
    app_name: str | None = None
    url: str | None = None
    user_initiated: bool = False
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("request_id", "session_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class CaptureGovernanceDecision(OrchestrationModel):
    """
    Decision returned by attention governance.
    """

    decision_id: str = Field(default_factory=lambda: f"capture_decision_{uuid4().hex}")
    request_id: str
    permission: CapturePermission
    reason: AttentionGovernanceReason
    allowed_depth: InspectionDepth
    frequency_hz: float = Field(ge=0)
    privacy_decision: PrivacyZoneDecision
    region: ScreenRegion | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("decision_id", "request_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class EnvironmentAttentionSession(OrchestrationModel):
    """
    One attention governance session.
    """

    session_id: str = Field(default_factory=lambda: f"attention_{uuid4().hex}")
    workspace_id: str
    mode: EnvironmentAttentionMode = EnvironmentAttentionMode.PERIPHERAL
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class AttentionRuntimeEvent(OrchestrationModel):
    """
    Environment attention runtime event.
    """

    event_id: str = Field(default_factory=lambda: f"attention_event_{uuid4().hex}")
    kind: AttentionGovernanceEventKind
    reason: AttentionGovernanceReason
    session_id: str | None = None
    focus_id: str | None = None
    request_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class AttentionOperationResult(OrchestrationModel):
    """
    Result returned by attention runtime operations.
    """

    success: bool
    reason: AttentionGovernanceReason
    event: AttentionRuntimeEvent
    session: EnvironmentAttentionSession | None = None
    decision: CaptureGovernanceDecision | None = None
    message: str

    @field_validator("message")
    @classmethod
    def _required_message(cls, value: str) -> str:
        return _clean_required(value)


class EnvironmentAttentionSnapshot(OrchestrationModel):
    """
    Diagnostics for Phase 8 Step 7.
    """

    name: str
    session_count: int = Field(ge=0)
    focus_region_count: int = Field(ge=0)
    active_focus_region_count: int = Field(ge=0)
    privacy_zone_count: int = Field(ge=0)
    decision_count: int = Field(ge=0)
    allowed_count: int = Field(ge=0)
    limited_count: int = Field(ge=0)
    deferred_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    runtime_event_count: int = Field(ge=0)
    last_reason: AttentionGovernanceReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        return _clean_required(value)


class EnvironmentAttentionRuntime:
    """
    Phase 8 Step 7 Environment Attention & Capture Governance.

    Responsibilities:
    - decide what JARVIS can visually inspect
    - enforce inspection depth
    - manage focus regions
    - control capture frequency
    - enforce privacy zones before capture/OCR
    - protect conversation latency under load/backpressure

    Non-responsibilities:
    - no screen capture
    - no OCR
    - no UI detection
    - no semantic parsing
    """

    def __init__(
        self,
        *,
        name: str = "environment_attention_runtime",
        state_runtime: EnvironmentStateRuntime | None = None,
        visual_policy: VisualAttentionPolicy | None = None,
        frequency_policy: CaptureFrequencyPolicy | None = None,
        privacy_policy: PrivacyZonePolicy | None = None,
    ) -> None:
        cleaned = name.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._state_runtime = state_runtime
        self._visual_policy = visual_policy or VisualAttentionPolicy()
        self._frequency_policy = frequency_policy or CaptureFrequencyPolicy()
        self._privacy_policy = privacy_policy or PrivacyZonePolicy()
        self._backpressure = EnvironmentBackpressureController()
        self._sessions: dict[str, EnvironmentAttentionSession] = {}
        self._focus_regions: dict[str, dict[str, FocusRegion]] = {}
        self._awareness: dict[str, PeripheralAwareness] = {}
        self._decisions: list[CaptureGovernanceDecision] = []
        self._runtime_events: list[AttentionRuntimeEvent] = []
        self._lock = RLock()
        self._last_reason: AttentionGovernanceReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def create_session(
        self,
        *,
        workspace_id: str,
        mode: EnvironmentAttentionMode | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EnvironmentAttentionSession:
        session = EnvironmentAttentionSession(
            workspace_id=workspace_id,
            mode=mode or self._visual_policy.default_mode,
            metadata=metadata or {},
        )
        event = self._event(
            kind=AttentionGovernanceEventKind.SESSION_CREATED,
            reason=AttentionGovernanceReason.SESSION_CREATED,
            session_id=session.session_id,
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._focus_regions[session.session_id] = {}
            self._awareness[session.session_id] = PeripheralAwareness()
            self._runtime_events.append(event)
            self._last_reason = event.reason

        return session

    def update_policy(
        self,
        *,
        visual_policy: VisualAttentionPolicy | None = None,
        frequency_policy: CaptureFrequencyPolicy | None = None,
        privacy_policy: PrivacyZonePolicy | None = None,
    ) -> AttentionRuntimeEvent:
        event = self._event(
            kind=AttentionGovernanceEventKind.POLICY_UPDATED,
            reason=AttentionGovernanceReason.POLICY_UPDATED,
        )

        with self._lock:
            if visual_policy is not None:
                self._visual_policy = visual_policy

            if frequency_policy is not None:
                self._frequency_policy = frequency_policy

            if privacy_policy is not None:
                self._privacy_policy = privacy_policy

            self._runtime_events.append(event)
            self._last_reason = event.reason

        return event

    def update_backpressure(
        self,
        controller: EnvironmentBackpressureController,
    ) -> AttentionRuntimeEvent:
        event = self._event(
            kind=AttentionGovernanceEventKind.POLICY_UPDATED,
            reason=AttentionGovernanceReason.POLICY_UPDATED,
            metadata={
                "load_level": controller.load_level.value,
                "conversation_active": controller.conversation_active,
                "interruption_active": controller.interruption_active,
            },
        )

        with self._lock:
            self._backpressure = controller
            self._runtime_events.append(event)
            self._last_reason = event.reason

        return event

    def add_focus_region(
        self,
        *,
        session_id: str,
        focus_region: FocusRegion,
    ) -> AttentionOperationResult:
        session = self.session_for(session_id)

        if session is None:
            return self._missing_session(session_id)

        event = self._event(
            kind=AttentionGovernanceEventKind.FOCUS_REGION_CHANGED,
            reason=AttentionGovernanceReason.FOCUS_REGION_ADDED,
            session_id=session_id,
            focus_id=focus_region.focus_id,
        )

        with self._lock:
            self._focus_regions[session_id][focus_region.focus_id] = focus_region
            self._touch_session(session_id)
            self._runtime_events.append(event)
            self._last_reason = event.reason

        return AttentionOperationResult(
            success=True,
            reason=AttentionGovernanceReason.FOCUS_REGION_ADDED,
            event=event,
            session=self.session_for(session_id),
            message="focus region added",
        )

    def remove_focus_region(
        self,
        *,
        session_id: str,
        focus_id: str,
    ) -> AttentionOperationResult:
        session = self.session_for(session_id)

        if session is None:
            return self._missing_session(session_id)

        with self._lock:
            region = self._focus_regions[session_id].get(focus_id)

            if region is not None:
                self._focus_regions[session_id][focus_id] = region.model_copy(
                    update={"active": False}
                )

        event = self._event(
            kind=AttentionGovernanceEventKind.FOCUS_REGION_CHANGED,
            reason=AttentionGovernanceReason.FOCUS_REGION_REMOVED,
            session_id=session_id,
            focus_id=focus_id,
        )

        with self._lock:
            self._touch_session(session_id)
            self._runtime_events.append(event)
            self._last_reason = event.reason

        return AttentionOperationResult(
            success=True,
            reason=AttentionGovernanceReason.FOCUS_REGION_REMOVED,
            event=event,
            session=self.session_for(session_id),
            message="focus region removed",
        )

    def update_peripheral_awareness(
        self,
        *,
        session_id: str,
        awareness: PeripheralAwareness,
    ) -> AttentionOperationResult:
        session = self.session_for(session_id)

        if session is None:
            return self._missing_session(session_id)

        event = self._event(
            kind=AttentionGovernanceEventKind.AWARENESS_UPDATED,
            reason=AttentionGovernanceReason.PERIPHERAL_AWARENESS_UPDATED,
            session_id=session_id,
            metadata={"important_change": awareness.has_important_change()},
        )

        with self._lock:
            self._awareness[session_id] = awareness
            self._touch_session(session_id)
            self._runtime_events.append(event)
            self._last_reason = event.reason

        return AttentionOperationResult(
            success=True,
            reason=AttentionGovernanceReason.PERIPHERAL_AWARENESS_UPDATED,
            event=event,
            session=self.session_for(session_id),
            message="peripheral awareness updated",
        )

    def set_attention_mode(
        self,
        *,
        session_id: str,
        mode: EnvironmentAttentionMode,
        user_initiated: bool = False,
    ) -> AttentionOperationResult:
        session = self.session_for(session_id)

        if session is None:
            return self._missing_session(session_id)

        if mode == EnvironmentAttentionMode.DEEP:
            if (
                self._visual_policy.require_user_intent_for_deep
                and not user_initiated
            ):
                event = self._event(
                    kind=AttentionGovernanceEventKind.ATTENTION_CHANGED,
                    reason=AttentionGovernanceReason.CAPTURE_BLOCKED_BY_DEPTH,
                    session_id=session_id,
                )

                with self._lock:
                    self._runtime_events.append(event)
                    self._last_reason = event.reason

                return AttentionOperationResult(
                    success=False,
                    reason=AttentionGovernanceReason.CAPTURE_BLOCKED_BY_DEPTH,
                    event=event,
                    session=session,
                    message="deep inspection requires user intent",
                )

        updated = session.model_copy(
            update={"mode": mode, "updated_at": utc_now()}
        )
        event = self._event(
            kind=AttentionGovernanceEventKind.ATTENTION_CHANGED,
            reason=AttentionGovernanceReason.ATTENTION_MODE_CHANGED,
            session_id=session_id,
        )

        with self._lock:
            self._sessions[session_id] = updated
            self._runtime_events.append(event)
            self._last_reason = event.reason

        return AttentionOperationResult(
            success=True,
            reason=AttentionGovernanceReason.ATTENTION_MODE_CHANGED,
            event=event,
            session=updated,
            message="attention mode changed",
        )

    def decide_capture(
        self,
        request: CaptureGovernanceRequest,
    ) -> CaptureGovernanceDecision:
        session = self.session_for(request.session_id)

        if session is None:
            raise ValueError(f"attention session not found: {request.session_id}")

        privacy = self._privacy_policy.decision_for(
            region=request.region,
            app_name=request.app_name,
            url=request.url,
        )

        permission, reason, allowed_depth = self._permission_for(
            session=session,
            request=request,
            privacy=privacy,
        )
        frequency = (
            self._frequency_policy.frequency_for(allowed_depth)
            if permission
            in {CapturePermission.ALLOW, CapturePermission.ALLOW_LIMITED}
            else 0.0
        )
        decision = CaptureGovernanceDecision(
            request_id=request.request_id,
            permission=permission,
            reason=reason,
            allowed_depth=allowed_depth,
            frequency_hz=frequency,
            privacy_decision=privacy,
            region=request.region if permission != CapturePermission.BLOCK else None,
        )
        event = self._event(
            kind=AttentionGovernanceEventKind.CAPTURE_DECIDED,
            reason=reason,
            session_id=request.session_id,
            request_id=request.request_id,
            metadata={
                "permission": permission.value,
                "depth": allowed_depth.value,
                "privacy": privacy.value,
            },
        )

        with self._lock:
            self._decisions.append(decision)
            self._runtime_events.append(event)
            self._last_reason = event.reason

        return decision

    def active_focus_regions(self, session_id: str) -> tuple[FocusRegion, ...]:
        with self._lock:
            regions = self._focus_regions.get(session_id, {})

            return tuple(region for region in regions.values() if region.active)

    def session_for(self, session_id: str) -> EnvironmentAttentionSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def peripheral_awareness(
        self,
        session_id: str,
    ) -> PeripheralAwareness | None:
        with self._lock:
            return self._awareness.get(session_id)

    def decisions(self) -> tuple[CaptureGovernanceDecision, ...]:
        with self._lock:
            return tuple(self._decisions)

    def runtime_events(self) -> tuple[AttentionRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._runtime_events)

    def snapshot(self) -> EnvironmentAttentionSnapshot:
        with self._lock:
            regions = [
                region
                for values in self._focus_regions.values()
                for region in values.values()
            ]

            return EnvironmentAttentionSnapshot(
                name=self.name,
                session_count=len(self._sessions),
                focus_region_count=len(regions),
                active_focus_region_count=sum(1 for region in regions if region.active),
                privacy_zone_count=len(self._privacy_policy.zones),
                decision_count=len(self._decisions),
                allowed_count=sum(
                    1
                    for decision in self._decisions
                    if decision.permission == CapturePermission.ALLOW
                ),
                limited_count=sum(
                    1
                    for decision in self._decisions
                    if decision.permission == CapturePermission.ALLOW_LIMITED
                ),
                deferred_count=sum(
                    1
                    for decision in self._decisions
                    if decision.permission == CapturePermission.DEFER
                ),
                blocked_count=sum(
                    1
                    for decision in self._decisions
                    if decision.permission == CapturePermission.BLOCK
                ),
                runtime_event_count=len(self._runtime_events),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        event = self._event(
            kind=AttentionGovernanceEventKind.RUNTIME_RESET,
            reason=AttentionGovernanceReason.RUNTIME_RESET,
        )

        with self._lock:
            self._sessions.clear()
            self._focus_regions.clear()
            self._awareness.clear()
            self._decisions.clear()
            self._runtime_events.clear()
            self._runtime_events.append(event)
            self._backpressure = EnvironmentBackpressureController()
            self._last_reason = event.reason

    def _permission_for(
        self,
        *,
        session: EnvironmentAttentionSession,
        request: CaptureGovernanceRequest,
        privacy: PrivacyZoneDecision,
    ) -> tuple[CapturePermission, AttentionGovernanceReason, InspectionDepth]:
        if privacy == PrivacyZoneDecision.CAPTURE_BLOCKED:
            return (
                CapturePermission.BLOCK,
                AttentionGovernanceReason.CAPTURE_BLOCKED_BY_PRIVACY,
                InspectionDepth.NONE,
            )

        if privacy == PrivacyZoneDecision.OCR_BLOCKED:
            return (
                CapturePermission.ALLOW_LIMITED,
                AttentionGovernanceReason.CAPTURE_LIMITED,
                InspectionDepth.PERIPHERAL,
            )

        if self._backpressure.interruption_active:
            return (
                CapturePermission.DEFER,
                AttentionGovernanceReason.CAPTURE_BLOCKED_BY_LOAD,
                InspectionDepth.NONE,
            )

        if (
            self._visual_policy.protect_conversation_latency
            and self._backpressure.conversation_active
            and request.requested_depth 
            in {
                InspectionDepth.FOCUSED,
                InspectionDepth.DEEP
            }
        ):
            return (
                CapturePermission.DEFER,
                AttentionGovernanceReason.CAPTURE_DEFERRED,
                InspectionDepth.NONE,
            )

        if self._backpressure.load_level in {
            VisualLoadLevel.CRITICAL,
            VisualLoadLevel.SHEDDING,
        }:
            return (
                CapturePermission.DEFER,
                AttentionGovernanceReason.CAPTURE_BLOCKED_BY_LOAD,
                InspectionDepth.NONE,
            )

        if not self._depth_allowed(
            requested=request.requested_depth,
            session=session,
            user_initiated=request.user_initiated,
        ):
            return (
                CapturePermission.BLOCK,
                AttentionGovernanceReason.CAPTURE_BLOCKED_BY_DEPTH,
                InspectionDepth.NONE,
            )

        if privacy == PrivacyZoneDecision.REDACT:
            return (
                CapturePermission.ALLOW_LIMITED,
                AttentionGovernanceReason.CAPTURE_LIMITED,
                min_depth(request.requested_depth, InspectionDepth.AMBIENT),
            )

        if request.requested_depth == InspectionDepth.DEEP:
            return (
                CapturePermission.ALLOW_LIMITED,
                AttentionGovernanceReason.CAPTURE_LIMITED,
                InspectionDepth.DEEP,
            )

        return (
            CapturePermission.ALLOW,
            AttentionGovernanceReason.CAPTURE_ALLOWED,
            request.requested_depth,
        )

    def _depth_allowed(
        self,
        *,
        requested: InspectionDepth,
        session: EnvironmentAttentionSession,
        user_initiated: bool,
    ) -> bool:
        if requested == InspectionDepth.NONE:
            return False

        if depth_rank(requested) > depth_rank(self._visual_policy.max_depth):
            return False

        if depth_rank(requested) > depth_rank(_depth_for_mode(session.mode)):
            if not user_initiated:
                return False

        if requested == InspectionDepth.DEEP:
            if not self._visual_policy.allow_deep_inspection:
                return False

            if self._visual_policy.require_user_intent_for_deep:
                return user_initiated

        return True

    def _touch_session(self, session_id: str) -> None:
        session = self._sessions[session_id]
        self._sessions[session_id] = session.model_copy(
            update={"updated_at": utc_now()}
        )

    @staticmethod
    def _event(
        *,
        kind: AttentionGovernanceEventKind,
        reason: AttentionGovernanceReason,
        session_id: str | None = None,
        focus_id: str | None = None,
        request_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AttentionRuntimeEvent:
        return AttentionRuntimeEvent(
            kind=kind,
            reason=reason,
            session_id=session_id,
            focus_id=focus_id,
            request_id=request_id,
            metadata=metadata or {},
        )

    @staticmethod
    def _missing_session(session_id: str) -> AttentionOperationResult:
        event = AttentionRuntimeEvent(
            kind=AttentionGovernanceEventKind.ATTENTION_CHANGED,
            reason=AttentionGovernanceReason.SESSION_NOT_FOUND,
            session_id=session_id,
        )

        return AttentionOperationResult(
            success=False,
            reason=AttentionGovernanceReason.SESSION_NOT_FOUND,
            event=event,
            message="attention session not found",
        )


def depth_rank(depth: InspectionDepth) -> int:
    order = {
        InspectionDepth.NONE: 0,
        InspectionDepth.PERIPHERAL: 1,
        InspectionDepth.AMBIENT: 2,
        InspectionDepth.FOCUSED: 3,
        InspectionDepth.DEEP: 4,
    }

    return order[depth]


def min_depth(left: InspectionDepth, right: InspectionDepth) -> InspectionDepth:
    return left if depth_rank(left) <= depth_rank(right) else right


def _depth_for_mode(mode: EnvironmentAttentionMode) -> InspectionDepth:
    if mode == EnvironmentAttentionMode.PERIPHERAL:
        return InspectionDepth.PERIPHERAL

    if mode == EnvironmentAttentionMode.AMBIENT:
        return InspectionDepth.AMBIENT

    if mode == EnvironmentAttentionMode.FOCUSED:
        return InspectionDepth.FOCUSED

    return InspectionDepth.DEEP


def _decision_from_zone(zone: PrivacyZone) -> PrivacyZoneDecision:
    if not zone.capture_allowed:
        return PrivacyZoneDecision.CAPTURE_BLOCKED

    if not zone.ocr_allowed:
        return PrivacyZoneDecision.OCR_BLOCKED

    return PrivacyZoneDecision.ALLOWED


def _regions_overlap(left: ScreenRegion, right: ScreenRegion) -> bool:
    return not (
        left.right < right.x
        or right.right < left.x
        or left.bottom < right.y
        or right.bottom < left.y
    )

def _clean_required(value: str) -> str:
    cleaned = value.strip()

    if not cleaned:
        raise ValueError("field cannot be empty.")

    return cleaned


def trusted_attention(reason: str = "attention governance") -> TrustCalibration:
    return TrustCalibration(
        confidence=0.95,
        stability=0.95,
        ambiguity=0.0,
        source=EnvironmentSource.OS_OBSERVER,
        reason=reason,
    )


def fake_focus_region(
    *,
    kind: FocusRegionKind = FocusRegionKind.ACTIVE_WINDOW,
    depth: InspectionDepth = InspectionDepth.FOCUSED,
) -> FocusRegion:
    return FocusRegion(
        kind=kind,
        region=ScreenRegion(x=0, y=0, width=800, height=600),
        inspection_depth=depth,
        trust=trusted_attention(),
    )