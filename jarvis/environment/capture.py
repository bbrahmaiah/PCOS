from __future__ import annotations

from enum import StrEnum
from threading import RLock
from typing import Any, Protocol
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.environment.attention import (
    CaptureGovernanceDecision,
    CaptureGovernanceRequest,
    CapturePermission,
    EnvironmentAttentionRuntime,
    InspectionDepth,
    PrivacyZoneDecision,
)
from jarvis.environment.models import (
    DisplayState,
    EnvironmentSource,
    PrivacyZone,
    ScreenRegion,
    TrustCalibration,
)
from jarvis.environment.state_runtime import EnvironmentStateRuntime
from jarvis.environment.visual_priority import (
    VisualLoadLevel,
    VisualPriorityArbitrator,
    VisualTaskDecision,
    VisualTaskKind,
    VisualTaskRequest,
)
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class CaptureMode(StrEnum):
    """
    Capture mode.

    Capture is controlled by attention, not by random screenshot loops.
    """

    REGION = "region"
    FOCUSED_WINDOW = "focused_window"
    DELTA = "delta"
    MULTI_MONITOR = "multi_monitor"


class CaptureStatus(StrEnum):
    """
    Capture lifecycle status.
    """

    REQUESTED = "requested"
    CAPTURED = "captured"
    LIMITED = "limited"
    DEFERRED = "deferred"
    BLOCKED = "blocked"
    FAILED = "failed"


class CaptureReason(StrEnum):
    """
    Machine-readable capture runtime reason.
    """

    SESSION_CREATED = "session_created"
    CAPTURED_REGION = "captured_region"
    CAPTURED_FOCUSED_WINDOW = "captured_focused_window"
    CAPTURED_DELTA = "captured_delta"
    CAPTURED_MULTI_MONITOR = "captured_multi_monitor"
    CAPTURE_LIMITED_BY_ATTENTION = "capture_limited_by_attention"
    CAPTURE_DEFERRED_BY_ATTENTION = "capture_deferred_by_attention"
    CAPTURE_BLOCKED_BY_ATTENTION = "capture_blocked_by_attention"
    CAPTURE_BLOCKED_BY_PRIVACY = "capture_blocked_by_privacy"
    CAPTURE_DEFERRED_BY_PRIORITY = "capture_deferred_by_priority"
    CAPTURE_FAILED = "capture_failed"
    PRIVACY_ZONE_ENFORCED = "privacy_zone_enforced"
    SCHEDULE_CREATED = "schedule_created"
    SESSION_NOT_FOUND = "session_not_found"
    RUNTIME_RESET = "runtime_reset"


class CaptureRuntimeEventKind(StrEnum):
    """
    Runtime event kind.
    """

    SESSION_CREATED = "session_created"
    CAPTURE_REQUESTED = "capture_requested"
    CAPTURE_COMPLETED = "capture_completed"
    CAPTURE_BLOCKED = "capture_blocked"
    CAPTURE_DEFERRED = "capture_deferred"
    PRIVACY_ENFORCED = "privacy_enforced"
    SCHEDULE_CREATED = "schedule_created"
    RUNTIME_RESET = "runtime_reset"


class CapturePixelFormat(StrEnum):
    """
    Capture pixel format metadata.

    Fake-first runtime does not store pixels, but real adapters will declare
    format here.
    """

    RGB = "rgb"
    RGBA = "rgba"
    BGRA = "bgra"
    GRAYSCALE = "grayscale"
    FAKE = "fake"


class CapturePayload(OrchestrationModel):
    """
    Captured image payload descriptor.

    This intentionally does not force raw bytes into the model.
    Real adapters may use buffer_id/path/hash metadata.
    """

    payload_id: str = Field(default_factory=lambda: f"capture_payload_{uuid4().hex}")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    pixel_format: CapturePixelFormat = CapturePixelFormat.FAKE
    byte_count: int = Field(default=0, ge=0)
    content_hash: str | None = None
    buffer_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("payload_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class RegionCapture(OrchestrationModel):
    """
    Region capture result.
    """

    capture_id: str = Field(default_factory=lambda: f"capture_{uuid4().hex}")
    mode: CaptureMode = CaptureMode.REGION
    status: CaptureStatus
    region: ScreenRegion
    payload: CapturePayload | None = None
    governance: CaptureGovernanceDecision
    trust: TrustCalibration
    captured_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("capture_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class DeltaCapture(OrchestrationModel):
    """
    Delta capture result.

    Tracks changed regions between captures without requiring full screenshot
    processing.
    """

    capture_id: str = Field(default_factory=lambda: f"delta_capture_{uuid4().hex}")
    mode: CaptureMode = CaptureMode.DELTA
    status: CaptureStatus
    previous_capture_id: str | None = None
    changed_regions: tuple[ScreenRegion, ...] = ()
    payload: CapturePayload | None = None
    governance: CaptureGovernanceDecision
    trust: TrustCalibration
    captured_at: object = Field(default_factory=utc_now)

    @field_validator("capture_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class MultiMonitorCapture(OrchestrationModel):
    """
    Multi-monitor capture result.
    """

    capture_id: str = Field(default_factory=lambda: f"multi_capture_{uuid4().hex}")
    mode: CaptureMode = CaptureMode.MULTI_MONITOR
    status: CaptureStatus
    displays: tuple[DisplayState, ...]
    captures: tuple[RegionCapture, ...]
    governance: CaptureGovernanceDecision
    trust: TrustCalibration
    captured_at: object = Field(default_factory=utc_now)

    @field_validator("capture_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _captures_required(self) -> MultiMonitorCapture:
        if self.status == CaptureStatus.CAPTURED and not self.captures:
            raise ValueError("captured multi-monitor result requires captures.")

        return self


class CaptureSchedule(OrchestrationModel):
    """
    Capture schedule created from attention governance.

    This is a schedule contract, not a background loop.
    """

    schedule_id: str = Field(default_factory=lambda: f"capture_schedule_{uuid4().hex}")
    session_id: str
    mode: CaptureMode
    depth: InspectionDepth
    frequency_hz: float = Field(ge=0)
    region: ScreenRegion | None = None
    active: bool = True
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("schedule_id", "session_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class PrivacyZoneEnforcerResult(OrchestrationModel):
    """
    Result of capture-level privacy enforcement.
    """

    allowed: bool
    decision: PrivacyZoneDecision
    reason: CaptureReason
    region: ScreenRegion | None = None
    enforced_zone: PrivacyZone | None = None


class ScreenCaptureRequest(OrchestrationModel):
    """
    Runtime request for governed capture.
    """

    request_id: str = Field(default_factory=lambda: f"screen_capture_req_{uuid4().hex}")
    session_id: str
    mode: CaptureMode
    region: ScreenRegion
    depth: InspectionDepth
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


class ScreenCaptureRuntimeEvent(OrchestrationModel):
    """
    Screen capture runtime event.
    """

    event_id: str = Field(default_factory=lambda: f"capture_event_{uuid4().hex}")
    kind: CaptureRuntimeEventKind
    reason: CaptureReason
    session_id: str | None = None
    request_id: str | None = None
    capture_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class ScreenCaptureSession(OrchestrationModel):
    """
    Capture runtime session.
    """

    session_id: str = Field(default_factory=lambda: f"capture_session_{uuid4().hex}")
    attention_session_id: str
    workspace_id: str
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "attention_session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class ScreenCaptureOperationResult(OrchestrationModel):
    """
    Result returned by capture operations.
    """

    success: bool
    reason: CaptureReason
    status: CaptureStatus
    event: ScreenCaptureRuntimeEvent
    region_capture: RegionCapture | None = None
    delta_capture: DeltaCapture | None = None
    multi_monitor_capture: MultiMonitorCapture | None = None
    schedule: CaptureSchedule | None = None
    message: str

    @field_validator("message")
    @classmethod
    def _required_message(cls, value: str) -> str:
        return _clean_required(value)


class ScreenCaptureRuntimeSnapshot(OrchestrationModel):
    """
    Diagnostics for Phase 8 Step 8.
    """

    name: str
    session_count: int = Field(ge=0)
    region_capture_count: int = Field(ge=0)
    delta_capture_count: int = Field(ge=0)
    multi_monitor_capture_count: int = Field(ge=0)
    schedule_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    deferred_count: int = Field(ge=0)
    limited_count: int = Field(ge=0)
    runtime_event_count: int = Field(ge=0)
    last_reason: CaptureReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        return _clean_required(value)


class ScreenCaptureAdapter(Protocol):
    """
    Protocol for real capture adapters.

    Step 8 is fake-first. Real Windows/Desktop capture adapters can implement
    this later without changing runtime governance.
    """

    def capture_region(self, region: ScreenRegion) -> CapturePayload:
        ...


class FakeScreenCaptureAdapter:
    """
    Fake capture adapter for tests and smoke runtime.

    It returns metadata-only fake payloads. No real screen pixels are captured.
    """

    def capture_region(self, region: ScreenRegion) -> CapturePayload:
        return CapturePayload(
            width=region.width,
            height=region.height,
            pixel_format=CapturePixelFormat.FAKE,
            byte_count=region.area,
            content_hash=f"fake_{region.x}_{region.y}_{region.width}_{region.height}",
            buffer_ref=None,
            metadata={"fake": True},
        )


class ScreenCaptureRuntime:
    """
    Phase 8 Step 8 Screen Capture Runtime.

    Responsibilities:
    - perform governed region/focused-window/delta/multi-monitor capture
    - enforce attention decisions before capture
    - enforce privacy zones before OCR/perception
    - consult visual priority before capture
    - use fake-first adapter boundary

    Non-responsibilities:
    - no OCR
    - no text extraction
    - no UI element detection
    - no semantic parsing
    - no physical action execution
    """

    def __init__(
        self,
        *,
        name: str = "screen_capture_runtime",
        attention_runtime: EnvironmentAttentionRuntime,
        priority_arbitrator: VisualPriorityArbitrator | None = None,
        state_runtime: EnvironmentStateRuntime | None = None,
        adapter: ScreenCaptureAdapter | None = None,
    ) -> None:
        cleaned = name.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._attention_runtime = attention_runtime
        self._priority_arbitrator = priority_arbitrator
        self._state_runtime = state_runtime
        self._adapter = adapter or FakeScreenCaptureAdapter()
        self._sessions: dict[str, ScreenCaptureSession] = {}
        self._region_captures: list[RegionCapture] = []
        self._delta_captures: list[DeltaCapture] = []
        self._multi_captures: list[MultiMonitorCapture] = []
        self._schedules: list[CaptureSchedule] = []
        self._runtime_events: list[ScreenCaptureRuntimeEvent] = []
        self._last_region_capture: dict[str, RegionCapture] = {}
        self._lock = RLock()
        self._last_reason: CaptureReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def create_session(
        self,
        *,
        attention_session_id: str,
        workspace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> ScreenCaptureSession:
        session = ScreenCaptureSession(
            attention_session_id=attention_session_id,
            workspace_id=workspace_id,
            metadata=metadata or {},
        )
        event = self._event(
            kind=CaptureRuntimeEventKind.SESSION_CREATED,
            reason=CaptureReason.SESSION_CREATED,
            session_id=session.session_id,
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._runtime_events.append(event)
            self._last_reason = event.reason

        return session

    def capture_region(
        self,
        request: ScreenCaptureRequest,
    ) -> ScreenCaptureOperationResult:
        session = self.session_for(request.session_id)

        if session is None:
            return self._missing_session(request.session_id)

        attention_decision = self._attention_decision_for(
            session=session,
            request=request,
        )
        blocked = self._blocked_by_attention(
            request=request,
            decision=attention_decision,
        )

        if blocked is not None:
            return blocked

        priority_result = self._priority_decision_for(request)

        if priority_result is not None:
            return priority_result

        payload = self._adapter.capture_region(request.region)
        status = (
            CaptureStatus.LIMITED
            if attention_decision.permission == CapturePermission.ALLOW_LIMITED
            else CaptureStatus.CAPTURED
        )
        reason = (
            CaptureReason.CAPTURE_LIMITED_BY_ATTENTION
            if status == CaptureStatus.LIMITED
            else CaptureReason.CAPTURED_REGION
        )
        capture = RegionCapture(
            mode=CaptureMode.REGION,
            status=status,
            region=request.region,
            payload=payload,
            governance=attention_decision,
            trust=_trusted_capture("region captured"),
            metadata={"request_reason": request.reason},
        )
        event = self._event(
            kind=CaptureRuntimeEventKind.CAPTURE_COMPLETED,
            reason=reason,
            session_id=request.session_id,
            request_id=request.request_id,
            capture_id=capture.capture_id,
        )

        with self._lock:
            self._region_captures.append(capture)
            self._last_region_capture[request.session_id] = capture
            self._runtime_events.append(event)
            self._touch_session(request.session_id)
            self._last_reason = event.reason

        return ScreenCaptureOperationResult(
            success=True,
            reason=reason,
            status=status,
            event=event,
            region_capture=capture,
            message="region captured",
        )

    def capture_focused_window(
        self,
        *,
        session_id: str,
        region: ScreenRegion,
        depth: InspectionDepth = InspectionDepth.FOCUSED,
        user_initiated: bool = False,
    ) -> ScreenCaptureOperationResult:
        return self.capture_region(
            ScreenCaptureRequest(
                session_id=session_id,
                mode=CaptureMode.FOCUSED_WINDOW,
                region=region,
                depth=depth,
                reason="focused window capture",
                user_initiated=user_initiated,
            )
        )

    def capture_delta(
        self,
        request: ScreenCaptureRequest,
    ) -> ScreenCaptureOperationResult:
        session = self.session_for(request.session_id)

        if session is None:
            return self._missing_session(request.session_id)

        previous = self._last_region_capture.get(request.session_id)
        region_result = self.capture_region(request)

        if not region_result.success or region_result.region_capture is None:
            return region_result

        current = region_result.region_capture
        changed_regions = self._changed_regions(previous=previous, current=current)

        delta = DeltaCapture(
            status=CaptureStatus.CAPTURED,
            previous_capture_id=previous.capture_id if previous else None,
            changed_regions=changed_regions,
            payload=current.payload,
            governance=current.governance,
            trust=_trusted_capture("delta captured"),
        )
        event = self._event(
            kind=CaptureRuntimeEventKind.CAPTURE_COMPLETED,
            reason=CaptureReason.CAPTURED_DELTA,
            session_id=request.session_id,
            request_id=request.request_id,
            capture_id=delta.capture_id,
        )

        with self._lock:
            self._delta_captures.append(delta)
            self._runtime_events.append(event)
            self._last_reason = event.reason

        return ScreenCaptureOperationResult(
            success=True,
            reason=CaptureReason.CAPTURED_DELTA,
            status=CaptureStatus.CAPTURED,
            event=event,
            delta_capture=delta,
            message="delta capture completed",
        )

    def capture_multi_monitor(
        self,
        *,
        session_id: str,
        displays: tuple[DisplayState, ...],
        depth: InspectionDepth = InspectionDepth.PERIPHERAL,
    ) -> ScreenCaptureOperationResult:
        session = self.session_for(session_id)

        if session is None:
            return self._missing_session(session_id)

        captures: list[RegionCapture] = []

        for display in displays:
            request = ScreenCaptureRequest(
                session_id=session_id,
                mode=CaptureMode.MULTI_MONITOR,
                region=display.bounds,
                depth=depth,
                reason="multi-monitor capture",
            )
            result = self.capture_region(request)

            if result.success and result.region_capture is not None:
                captures.append(result.region_capture)

        if not captures:
            event = self._event(
                kind=CaptureRuntimeEventKind.CAPTURE_BLOCKED,
                reason=CaptureReason.CAPTURE_BLOCKED_BY_ATTENTION,
                session_id=session_id,
            )

            with self._lock:
                self._runtime_events.append(event)
                self._last_reason = event.reason

            return ScreenCaptureOperationResult(
                success=False,
                reason=CaptureReason.CAPTURE_BLOCKED_BY_ATTENTION,
                status=CaptureStatus.BLOCKED,
                event=event,
                message="multi-monitor capture blocked",
            )

        governance = captures[0].governance
        multi = MultiMonitorCapture(
            status=CaptureStatus.CAPTURED,
            displays=displays,
            captures=tuple(captures),
            governance=governance,
            trust=_trusted_capture("multi-monitor captured"),
        )
        event = self._event(
            kind=CaptureRuntimeEventKind.CAPTURE_COMPLETED,
            reason=CaptureReason.CAPTURED_MULTI_MONITOR,
            session_id=session_id,
            capture_id=multi.capture_id,
        )

        with self._lock:
            self._multi_captures.append(multi)
            self._runtime_events.append(event)
            self._last_reason = event.reason

        return ScreenCaptureOperationResult(
            success=True,
            reason=CaptureReason.CAPTURED_MULTI_MONITOR,
            status=CaptureStatus.CAPTURED,
            event=event,
            multi_monitor_capture=multi,
            message="multi-monitor capture completed",
        )

    def create_schedule(
        self,
        *,
        session_id: str,
        mode: CaptureMode,
        depth: InspectionDepth,
        region: ScreenRegion | None = None,
    ) -> ScreenCaptureOperationResult:
        session = self.session_for(session_id)

        if session is None:
            return self._missing_session(session_id)

        frequency = self._attention_runtime.decide_capture(
            CaptureGovernanceRequest(
                session_id=session.attention_session_id,
                region=region or ScreenRegion(x=0, y=0, width=1, height=1),
                requested_depth=depth,
                reason="capture schedule governance",
                user_initiated=False,
            )
        ).frequency_hz
        schedule = CaptureSchedule(
            session_id=session_id,
            mode=mode,
            depth=depth,
            frequency_hz=frequency,
            region=region,
        )
        event = self._event(
            kind=CaptureRuntimeEventKind.SCHEDULE_CREATED,
            reason=CaptureReason.SCHEDULE_CREATED,
            session_id=session_id,
        )

        with self._lock:
            self._schedules.append(schedule)
            self._runtime_events.append(event)
            self._last_reason = event.reason

        return ScreenCaptureOperationResult(
            success=True,
            reason=CaptureReason.SCHEDULE_CREATED,
            status=CaptureStatus.REQUESTED,
            event=event,
            schedule=schedule,
            message="capture schedule created",
        )

    def session_for(self, session_id: str) -> ScreenCaptureSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def region_captures(self) -> tuple[RegionCapture, ...]:
        with self._lock:
            return tuple(self._region_captures)

    def delta_captures(self) -> tuple[DeltaCapture, ...]:
        with self._lock:
            return tuple(self._delta_captures)

    def multi_monitor_captures(self) -> tuple[MultiMonitorCapture, ...]:
        with self._lock:
            return tuple(self._multi_captures)

    def schedules(self) -> tuple[CaptureSchedule, ...]:
        with self._lock:
            return tuple(self._schedules)

    def runtime_events(self) -> tuple[ScreenCaptureRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._runtime_events)

    def snapshot(self) -> ScreenCaptureRuntimeSnapshot:
        with self._lock:
            results = [
                *(capture.status for capture in self._region_captures),
                *(capture.status for capture in self._delta_captures),
                *(capture.status for capture in self._multi_captures),
            ]

            event_reasons = [event.reason for event in self._runtime_events]

            return ScreenCaptureRuntimeSnapshot(
                name=self.name,
                session_count=len(self._sessions),
                region_capture_count=len(self._region_captures),
                delta_capture_count=len(self._delta_captures),
                multi_monitor_capture_count=len(self._multi_captures),
                schedule_count=len(self._schedules),
                blocked_count=sum(
                    1
                    for reason in event_reasons
                    if reason
                    in {
                        CaptureReason.CAPTURE_BLOCKED_BY_ATTENTION,
                        CaptureReason.CAPTURE_BLOCKED_BY_PRIVACY,
                    }
                ),
                deferred_count=sum(
                    1
                    for reason in event_reasons
                    if reason
                    in {
                        CaptureReason.CAPTURE_DEFERRED_BY_ATTENTION,
                        CaptureReason.CAPTURE_DEFERRED_BY_PRIORITY,
                    }
                ),
                limited_count=sum(
                    1 for status in results if status == CaptureStatus.LIMITED
                ),
                runtime_event_count=len(self._runtime_events),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        event = self._event(
            kind=CaptureRuntimeEventKind.RUNTIME_RESET,
            reason=CaptureReason.RUNTIME_RESET,
        )

        with self._lock:
            self._sessions.clear()
            self._region_captures.clear()
            self._delta_captures.clear()
            self._multi_captures.clear()
            self._schedules.clear()
            self._runtime_events.clear()
            self._last_region_capture.clear()
            self._runtime_events.append(event)
            self._last_reason = event.reason

    def _attention_decision_for(
        self,
        *,
        session: ScreenCaptureSession,
        request: ScreenCaptureRequest,
    ) -> CaptureGovernanceDecision:
        return self._attention_runtime.decide_capture(
            CaptureGovernanceRequest(
                session_id=session.attention_session_id,
                region=request.region,
                requested_depth=request.depth,
                reason=request.reason,
                app_name=request.app_name,
                url=request.url,
                user_initiated=request.user_initiated,
                metadata=request.metadata,
            )
        )

    def _blocked_by_attention(
        self,
        *,
        request: ScreenCaptureRequest,
        decision: CaptureGovernanceDecision,
    ) -> ScreenCaptureOperationResult | None:
        if decision.permission == CapturePermission.ALLOW:
            return None

        if decision.permission == CapturePermission.ALLOW_LIMITED:
            return None

        reason = (
            CaptureReason.CAPTURE_BLOCKED_BY_PRIVACY
            if decision.privacy_decision == PrivacyZoneDecision.CAPTURE_BLOCKED
            else CaptureReason.CAPTURE_BLOCKED_BY_ATTENTION
        )
        status = (
            CaptureStatus.DEFERRED
            if decision.permission == CapturePermission.DEFER
            else CaptureStatus.BLOCKED
        )

        if status == CaptureStatus.DEFERRED:
            reason = CaptureReason.CAPTURE_DEFERRED_BY_ATTENTION

        event = self._event(
            kind=(
                CaptureRuntimeEventKind.CAPTURE_DEFERRED
                if status == CaptureStatus.DEFERRED
                else CaptureRuntimeEventKind.CAPTURE_BLOCKED
            ),
            reason=reason,
            session_id=request.session_id,
            request_id=request.request_id,
        )

        with self._lock:
            self._runtime_events.append(event)
            self._last_reason = event.reason

        return ScreenCaptureOperationResult(
            success=False,
            reason=reason,
            status=status,
            event=event,
            message="capture blocked by attention governance",
        )

    def _priority_decision_for(
        self,
        request: ScreenCaptureRequest,
    ) -> ScreenCaptureOperationResult | None:
        if self._priority_arbitrator is None:
            return None

        if self._priority_arbitrator.snapshot().load_level in {
            VisualLoadLevel.CRITICAL,
            VisualLoadLevel.SHEDDING,
        }:
            reason = CaptureReason.CAPTURE_DEFERRED_BY_PRIORITY
            event = self._event(
            kind=CaptureRuntimeEventKind.CAPTURE_DEFERRED,
            reason=reason,
            session_id=request.session_id,
            request_id=request.request_id,
        )

        with self._lock:
            self._runtime_events.append(event)
            self._last_reason = event.reason

        return ScreenCaptureOperationResult(
            success=False,
            reason=reason,
            status=CaptureStatus.DEFERRED,
            event=event,
            message="capture deferred by critical visual load",
        )

        decision = self._priority_arbitrator.arbitrate(
            VisualTaskRequest(
                kind=VisualTaskKind.CAPTURE,
                requested_latency_ms=50,
                requested_cpu_percent=8,
                requested_memory_mb=128,
                can_degrade=True,
                can_shed=False,
                metadata={"capture_request_id": request.request_id},
            )
        )

        if decision.decision in {
            VisualTaskDecision.RUN_NOW,
            VisualTaskDecision.RUN_LIMITED,
        }:
            return None

        reason = CaptureReason.CAPTURE_DEFERRED_BY_PRIORITY
        event = self._event(
            kind=CaptureRuntimeEventKind.CAPTURE_DEFERRED,
            reason=reason,
            session_id=request.session_id,
            request_id=request.request_id,
        )

        with self._lock:
            self._runtime_events.append(event)
            self._last_reason = event.reason

        return ScreenCaptureOperationResult(
            success=False,
            reason=reason,
            status=CaptureStatus.DEFERRED,
            event=event,
            message="capture deferred by visual priority arbitration",
        )

    @staticmethod
    def _changed_regions(
        *,
        previous: RegionCapture | None,
        current: RegionCapture,
    ) -> tuple[ScreenRegion, ...]:
        if previous is None:
            return (current.region,)

        if previous.payload is None or current.payload is None:
            return (current.region,)

        if previous.payload.content_hash != current.payload.content_hash:
            return (current.region,)

        return ()

    def _touch_session(self, session_id: str) -> None:
        session = self._sessions[session_id]
        self._sessions[session_id] = session.model_copy(
            update={"updated_at": utc_now()}
        )

    @staticmethod
    def _event(
        *,
        kind: CaptureRuntimeEventKind,
        reason: CaptureReason,
        session_id: str | None = None,
        request_id: str | None = None,
        capture_id: str | None = None,
    ) -> ScreenCaptureRuntimeEvent:
        return ScreenCaptureRuntimeEvent(
            kind=kind,
            reason=reason,
            session_id=session_id,
            request_id=request_id,
            capture_id=capture_id,
        )

    @staticmethod
    def _missing_session(session_id: str) -> ScreenCaptureOperationResult:
        event = ScreenCaptureRuntimeEvent(
            kind=CaptureRuntimeEventKind.CAPTURE_BLOCKED,
            reason=CaptureReason.SESSION_NOT_FOUND,
            session_id=session_id,
        )

        return ScreenCaptureOperationResult(
            success=False,
            reason=CaptureReason.SESSION_NOT_FOUND,
            status=CaptureStatus.FAILED,
            event=event,
            message="capture session not found",
        )


def _trusted_capture(reason: str) -> TrustCalibration:
    return TrustCalibration(
        confidence=0.95,
        stability=0.95,
        ambiguity=0.0,
        source=EnvironmentSource.SCREEN_CAPTURE,
        reason=reason,
    )


def _clean_required(value: str) -> str:
    cleaned = value.strip()

    if not cleaned:
        raise ValueError("field cannot be empty.")

    return cleaned