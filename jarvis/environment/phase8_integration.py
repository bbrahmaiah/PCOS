from __future__ import annotations

from enum import StrEnum
from threading import RLock
from typing import Any, Protocol
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.environment.models import EnvironmentSource, TrustCalibration
from jarvis.environment.trust_runtime import TrustPolicyClassification
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class Phase8IntegrationBoundary(StrEnum):
    PHASE1_EVENT_BUS = "phase1_event_bus"
    PHASE2_VOICE_ENVIRONMENT_FUSION = "phase2_voice_environment_fusion"
    PHASE3_COGNITION_FUSED_CONTEXT = "phase3_cognition_fused_context"
    PHASE4_MEMORY_GATEWAY = "phase4_memory_gateway"
    PHASE5_ACTION_POLICY = "phase5_action_policy"
    PHASE6_ORCHESTRATION = "phase6_orchestration"
    PHASE7_VISUAL_STREAMING = "phase7_visual_streaming"
    PHASE8_ENVIRONMENT_AWARENESS = "phase8_environment_awareness"


class Phase8IntegrationStatus(StrEnum):
    INTEGRATED = "integrated"
    VERIFIED = "verified"
    BLOCKED = "blocked"
    FAILED = "failed"


class Phase8IntegrationDecision(StrEnum):
    INTEGRATE = "integrate"
    VERIFY = "verify"
    BLOCK = "block"
    FAIL = "fail"


class Phase8IntegrationReason(StrEnum):
    SESSION_CREATED = "session_created"
    EVENT_BUS_BRIDGED = "event_bus_bridged"
    VOICE_ENVIRONMENT_FUSED = "voice_environment_fused"
    COGNITION_CONTEXT_INJECTED = "cognition_context_injected"
    MEMORY_GATEWAY_BRIDGED = "memory_gateway_bridged"
    ACTION_POLICY_BRIDGED = "action_policy_bridged"
    ORCHESTRATION_BRIDGED = "orchestration_bridged"
    VISUAL_STREAMING_BRIDGED = "visual_streaming_bridged"
    ENVIRONMENT_AWARENESS_BRIDGED = "environment_awareness_bridged"
    INTEGRATION_VERIFIED = "integration_verified"
    POLICY_BYPASS_BLOCKED = "policy_bypass_blocked"
    DIRECT_ACTION_BLOCKED = "direct_action_blocked"
    MEMORY_GATEWAY_REQUIRED = "memory_gateway_required"
    SESSION_NOT_FOUND = "session_not_found"
    RUNTIME_RESET = "runtime_reset"


class Phase8IntegrationEventKind(StrEnum):
    SESSION_CREATED = "session_created"
    BRIDGE_INTEGRATED = "bridge_integrated"
    INTEGRATION_VERIFIED = "integration_verified"
    INTEGRATION_BLOCKED = "integration_blocked"
    RUNTIME_RESET = "runtime_reset"


class Phase8IntegrationRisk(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BLOCKED = "blocked"


class Phase8BridgeMode(StrEnum):
    READ_ONLY = "read_only"
    PREPARE_ONLY = "prepare_only"
    POLICY_GATED = "policy_gated"
    STREAMING = "streaming"

class Phase8Bridge(Protocol):
    def build_packet(
        self,
        request: Phase8IntegrationRequest,
    ) -> Phase8IntegrationPacket:
        ...


class Phase8IntegrationPacket(OrchestrationModel):
    """
    Generic cross-phase packet.

    It intentionally carries serialized contracts, not live runtime objects.
    This prevents Phase 8 from directly controlling old runtimes.
    """

    packet_id: str = Field(default_factory=lambda: f"phase8_packet_{uuid4().hex}")
    boundary: Phase8IntegrationBoundary
    source_phase: str
    target_phase: str
    workspace_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    mode: Phase8BridgeMode
    risk: Phase8IntegrationRisk = Phase8IntegrationRisk.LOW
    policy_required: bool = False
    gateway_required: bool = False
    user_visible: bool = True
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("packet_id", "source_phase", "target_phase", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _protect_boundaries(self) -> Phase8IntegrationPacket:
        if _payload_requests_direct_action(self.payload):
            raise ValueError("integration packet cannot request direct action.")

        if self.boundary == Phase8IntegrationBoundary.PHASE4_MEMORY_GATEWAY:
            if not self.gateway_required:
                raise ValueError("Phase 4 integration requires memory gateway.")

        if self.boundary == Phase8IntegrationBoundary.PHASE5_ACTION_POLICY:
            if not self.policy_required:
                raise ValueError("Phase 5 integration requires action policy.")

        return self


class Phase8IntegrationRequest(OrchestrationModel):
    request_id: str = Field(default_factory=lambda: f"phase8_req_{uuid4().hex}")
    session_id: str
    workspace_id: str
    boundary: Phase8IntegrationBoundary
    payload: dict[str, Any] = Field(default_factory=dict)
    source_phase: str = "phase8"
    target_phase: str
    mode: Phase8BridgeMode
    policy_required: bool = False
    gateway_required: bool = False
    user_visible: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("request_id", "session_id", "workspace_id", "target_phase")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class Phase8IntegrationVerification(OrchestrationModel):
    verification_id: str = Field(
        default_factory=lambda: f"phase8_verify_{uuid4().hex}"
    )
    required_boundaries: tuple[Phase8IntegrationBoundary, ...]
    integrated_boundaries: tuple[Phase8IntegrationBoundary, ...]
    missing_boundaries: tuple[Phase8IntegrationBoundary, ...]
    verified: bool
    reason: str
    created_at: object = Field(default_factory=utc_now)

    @field_validator("verification_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class Phase8IntegrationResult(OrchestrationModel):
    result_id: str = Field(default_factory=lambda: f"phase8_result_{uuid4().hex}")
    status: Phase8IntegrationStatus
    decision: Phase8IntegrationDecision
    reason: Phase8IntegrationReason
    packet: Phase8IntegrationPacket | None = None
    verification: Phase8IntegrationVerification | None = None
    trust: TrustCalibration
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("result_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _payload_matches_status(self) -> Phase8IntegrationResult:
        if self.status == Phase8IntegrationStatus.INTEGRATED:
            if self.packet is None:
                raise ValueError("INTEGRATED result requires packet.")

        if self.status == Phase8IntegrationStatus.VERIFIED:
            if self.verification is None:
                raise ValueError("VERIFIED result requires verification.")

        return self


class Phase8IntegrationSession(OrchestrationModel):
    session_id: str = Field(default_factory=lambda: f"phase8_int_{uuid4().hex}")
    workspace_id: str
    integrated_count: int = Field(default=0, ge=0)
    verified_count: int = Field(default=0, ge=0)
    blocked_count: int = Field(default=0, ge=0)
    phase1_event_bus_ready: bool = False
    phase2_voice_fusion_ready: bool = False
    phase3_cognition_context_ready: bool = False
    phase4_memory_gateway_ready: bool = False
    phase5_action_policy_ready: bool = False
    phase6_orchestration_ready: bool = False
    phase7_visual_streaming_ready: bool = False
    phase8_awareness_ready: bool = False
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class Phase8IntegrationRuntimeEvent(OrchestrationModel):
    event_id: str = Field(default_factory=lambda: f"phase8_event_{uuid4().hex}")
    kind: Phase8IntegrationEventKind
    reason: Phase8IntegrationReason
    session_id: str | None = None
    result_id: str | None = None
    packet_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class Phase8IntegrationRuntimeSnapshot(OrchestrationModel):
    name: str
    session_count: int = Field(ge=0)
    result_count: int = Field(ge=0)
    integrated_count: int = Field(ge=0)
    verified_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    runtime_event_count: int = Field(ge=0)
    last_reason: Phase8IntegrationReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        return _clean_required(value)


class EnvironmentEventBusBridge:
    def build_packet(
        self,
        request: Phase8IntegrationRequest,
    ) -> Phase8IntegrationPacket:
        return _packet(
            request=request,
            target_phase="phase1",
            mode=Phase8BridgeMode.READ_ONLY,
            reason_payload={
                "event_type": "environment_awareness",
                "bus_contract": "EnvironmentEvents",
            },
        )


class VoiceEnvironmentFusionBridge:
    def build_packet(
        self,
        request: Phase8IntegrationRequest,
    ) -> Phase8IntegrationPacket:
        return _packet(
            request=request,
            target_phase="phase2",
            mode=Phase8BridgeMode.PREPARE_ONLY,
            reason_payload={
                "fusion_contract": "voice + environment context",
                "voice_safe": True,
            },
        )


class CognitionFusedContextBridge:
    def build_packet(
        self,
        request: Phase8IntegrationRequest,
    ) -> Phase8IntegrationPacket:
        return _packet(
            request=request,
            target_phase="phase3",
            mode=Phase8BridgeMode.READ_ONLY,
            reason_payload={
                "context_contract": "FusedContext",
                "cognition_receives_environment": True,
            },
        )


class WorkflowMemoryGatewayBridge:
    def build_packet(
        self,
        request: Phase8IntegrationRequest,
    ) -> Phase8IntegrationPacket:
        return _packet(
            request=request,
            target_phase="phase4",
            mode=Phase8BridgeMode.POLICY_GATED,
            gateway_required=True,
            reason_payload={
                "memory_contract": "WorkflowMemoryGateway",
                "direct_memory_write": False,
            },
        )


class PhysicalActionPolicyBridge:
    def build_packet(
        self,
        request: Phase8IntegrationRequest,
    ) -> Phase8IntegrationPacket:
        return _packet(
            request=request,
            target_phase="phase5",
            mode=Phase8BridgeMode.POLICY_GATED,
            policy_required=True,
            reason_payload={
                "action_contract": "physical interaction through policy chain",
                "direct_execution": False,
            },
        )


class VisualWorkerOrchestrationBridge:
    def build_packet(
        self,
        request: Phase8IntegrationRequest,
    ) -> Phase8IntegrationPacket:
        return _packet(
            request=request,
            target_phase="phase6",
            mode=Phase8BridgeMode.PREPARE_ONLY,
            reason_payload={
                "worker_contract": "environment visual workers",
                "scheduled_by_orchestration": True,
            },
        )


class VisualContextStreamingBridge:
    def build_packet(
        self,
        request: Phase8IntegrationRequest,
    ) -> Phase8IntegrationPacket:
        return _packet(
            request=request,
            target_phase="phase7",
            mode=Phase8BridgeMode.STREAMING,
            reason_payload={
                "streaming_contract": "visual context precomputed and streamed",
                "blocking_context_build": False,
            },
        )


class EnvironmentAwarenessBridge:
    def build_packet(
        self,
        request: Phase8IntegrationRequest,
    ) -> Phase8IntegrationPacket:
        return _packet(
            request=request,
            target_phase="phase8",
            mode=Phase8BridgeMode.READ_ONLY,
            reason_payload={
                "environment_awareness": True,
                "replaces_previous_phases": False,
            },
        )


class Phase8FullIntegrationRuntime:
    """
    Phase 8 Step 39 Full Phase 1-8 Integration.

    This runtime connects Phase 8 environmental awareness into every previous
    phase while preserving strict boundaries:

    Phase 1: EnvironmentEvents on event bus
    Phase 2: voice + environment fusion
    Phase 3: cognition receives FusedContext
    Phase 4: workflow memory through Memory Gateway
    Phase 5: physical interaction through policy chain
    Phase 6: visual workers scheduled by orchestration
    Phase 7: visual context precomputed and streamed
    Phase 8: environment awareness stays governed
    """

    _required_boundaries: tuple[Phase8IntegrationBoundary, ...] = (
        Phase8IntegrationBoundary.PHASE1_EVENT_BUS,
        Phase8IntegrationBoundary.PHASE2_VOICE_ENVIRONMENT_FUSION,
        Phase8IntegrationBoundary.PHASE3_COGNITION_FUSED_CONTEXT,
        Phase8IntegrationBoundary.PHASE4_MEMORY_GATEWAY,
        Phase8IntegrationBoundary.PHASE5_ACTION_POLICY,
        Phase8IntegrationBoundary.PHASE6_ORCHESTRATION,
        Phase8IntegrationBoundary.PHASE7_VISUAL_STREAMING,
        Phase8IntegrationBoundary.PHASE8_ENVIRONMENT_AWARENESS,
    )

    def __init__(
        self,
        *,
        name: str = "phase8_full_integration_runtime",
    ) -> None:
        cleaned = name.strip()
        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._sessions: dict[str, Phase8IntegrationSession] = {}
        self._results: list[Phase8IntegrationResult] = []
        self._events: list[Phase8IntegrationRuntimeEvent] = []
        self._bridges: dict[Phase8IntegrationBoundary, Phase8Bridge] = {
            Phase8IntegrationBoundary.PHASE1_EVENT_BUS: EnvironmentEventBusBridge(),
            Phase8IntegrationBoundary.PHASE2_VOICE_ENVIRONMENT_FUSION: (
                VoiceEnvironmentFusionBridge()
            ),
            Phase8IntegrationBoundary.PHASE3_COGNITION_FUSED_CONTEXT: (
                CognitionFusedContextBridge()
            ),
            Phase8IntegrationBoundary.PHASE4_MEMORY_GATEWAY: (
                WorkflowMemoryGatewayBridge()
            ),
            Phase8IntegrationBoundary.PHASE5_ACTION_POLICY: (
                PhysicalActionPolicyBridge()
            ),
            Phase8IntegrationBoundary.PHASE6_ORCHESTRATION: (
                VisualWorkerOrchestrationBridge()
            ),
            Phase8IntegrationBoundary.PHASE7_VISUAL_STREAMING: (
                VisualContextStreamingBridge()
            ),
            Phase8IntegrationBoundary.PHASE8_ENVIRONMENT_AWARENESS: (
                EnvironmentAwarenessBridge()
            ),
        }
        self._lock = RLock()
        self._last_reason: Phase8IntegrationReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def create_session(
        self,
        *,
        workspace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> Phase8IntegrationSession:
        session = Phase8IntegrationSession(
            workspace_id=workspace_id,
            metadata=metadata or {},
        )
        event = self._event(
            kind=Phase8IntegrationEventKind.SESSION_CREATED,
            reason=Phase8IntegrationReason.SESSION_CREATED,
            session_id=session.session_id,
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._events.append(event)
            self._last_reason = event.reason

        return session

    def integrate(
        self,
        request: Phase8IntegrationRequest,
    ) -> Phase8IntegrationResult:
        if self.session_for(request.session_id) is None:
            result = _blocked_result(
                status=Phase8IntegrationStatus.FAILED,
                decision=Phase8IntegrationDecision.FAIL,
                reason=Phase8IntegrationReason.SESSION_NOT_FOUND,
                message="phase8 integration session not found",
            )
            self._record_result(result, request.session_id)
            return result

        if _payload_requests_direct_action(request.payload):
            result = _blocked_result(
                status=Phase8IntegrationStatus.BLOCKED,
                decision=Phase8IntegrationDecision.BLOCK,
                reason=Phase8IntegrationReason.DIRECT_ACTION_BLOCKED,
                message="direct action through integration is blocked",
            )
            self._record_result(result, request.session_id)
            return result

        if _payload_bypasses_policy(request):
            result = _blocked_result(
                status=Phase8IntegrationStatus.BLOCKED,
                decision=Phase8IntegrationDecision.BLOCK,
                reason=Phase8IntegrationReason.POLICY_BYPASS_BLOCKED,
                message="policy bypass through integration is blocked",
            )
            self._record_result(result, request.session_id)
            return result

        bridge = self._bridges[request.boundary]
        packet = bridge.build_packet(request)
        result = Phase8IntegrationResult(
            status=Phase8IntegrationStatus.INTEGRATED,
            decision=Phase8IntegrationDecision.INTEGRATE,
            reason=_reason_for_boundary(request.boundary),
            packet=packet,
            trust=_trust(confidence=0.90, reason="phase bridge integrated"),
            message=f"{request.boundary.value} integrated",
        )
        self._record_result(result, request.session_id)
        return result

    def integrate_all(
        self,
        *,
        session_id: str,
        workspace_id: str,
    ) -> tuple[Phase8IntegrationResult, ...]:
        results: list[Phase8IntegrationResult] = []

        for boundary in self._required_boundaries:
            request = Phase8IntegrationRequest(
                session_id=session_id,
                workspace_id=workspace_id,
                boundary=boundary,
                payload={"integration": boundary.value},
                target_phase=_target_phase_for(boundary),
                mode=_mode_for(boundary),
                policy_required=boundary
                == Phase8IntegrationBoundary.PHASE5_ACTION_POLICY,
                gateway_required=boundary
                == Phase8IntegrationBoundary.PHASE4_MEMORY_GATEWAY,
            )
            results.append(self.integrate(request))

        return tuple(results)

    def verify_full_integration(
        self,
        *,
        session_id: str,
    ) -> Phase8IntegrationResult:
        session = self.session_for(session_id)
        if session is None:
            result = _blocked_result(
                status=Phase8IntegrationStatus.FAILED,
                decision=Phase8IntegrationDecision.FAIL,
                reason=Phase8IntegrationReason.SESSION_NOT_FOUND,
                message="phase8 integration session not found",
            )
            self._record_result(result, session_id)
            return result

        integrated = self._integrated_boundaries_for(session_id)
        missing = tuple(
            boundary
            for boundary in self._required_boundaries
            if boundary not in integrated
        )
        verified = not missing

        verification = Phase8IntegrationVerification(
            required_boundaries=self._required_boundaries,
            integrated_boundaries=tuple(integrated),
            missing_boundaries=missing,
            verified=verified,
            reason=(
                "all phase 1-8 integration boundaries are ready"
                if verified
                else "one or more integration boundaries are missing"
            ),
        )

        result = Phase8IntegrationResult(
            status=(
                Phase8IntegrationStatus.VERIFIED
                if verified
                else Phase8IntegrationStatus.BLOCKED
            ),
            decision=(
                Phase8IntegrationDecision.VERIFY
                if verified
                else Phase8IntegrationDecision.BLOCK
            ),
            reason=(
                Phase8IntegrationReason.INTEGRATION_VERIFIED
                if verified
                else Phase8IntegrationReason.POLICY_BYPASS_BLOCKED
            ),
            verification=verification,
            trust=_trust(
                confidence=0.95 if verified else 0.35,
                reason=verification.reason,
            ),
            message=verification.reason,
        )
        self._record_result(result, session_id)
        return result

    def session_for(
        self,
        session_id: str,
    ) -> Phase8IntegrationSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def results(self) -> tuple[Phase8IntegrationResult, ...]:
        with self._lock:
            return tuple(self._results)

    def events(self) -> tuple[Phase8IntegrationRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def snapshot(self) -> Phase8IntegrationRuntimeSnapshot:
        with self._lock:
            return Phase8IntegrationRuntimeSnapshot(
                name=self.name,
                session_count=len(self._sessions),
                result_count=len(self._results),
                integrated_count=sum(
                    1
                    for result in self._results
                    if result.status == Phase8IntegrationStatus.INTEGRATED
                ),
                verified_count=sum(
                    1
                    for result in self._results
                    if result.status == Phase8IntegrationStatus.VERIFIED
                ),
                blocked_count=sum(
                    1
                    for result in self._results
                    if result.status
                    in {
                        Phase8IntegrationStatus.BLOCKED,
                        Phase8IntegrationStatus.FAILED,
                    }
                ),
                runtime_event_count=len(self._events),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        event = self._event(
            kind=Phase8IntegrationEventKind.RUNTIME_RESET,
            reason=Phase8IntegrationReason.RUNTIME_RESET,
        )

        with self._lock:
            self._sessions.clear()
            self._results.clear()
            self._events.clear()
            self._events.append(event)
            self._last_reason = event.reason

    def _integrated_boundaries_for(
        self,
        session_id: str,
    ) -> set[Phase8IntegrationBoundary]:
        with self._lock:
            return {
                result.packet.boundary
                for result in self._results
                if result.packet is not None
                and result.status == Phase8IntegrationStatus.INTEGRATED
                and self._event_belongs_to_session(result.result_id, session_id)
            }

    def _event_belongs_to_session(
        self,
        result_id: str,
        session_id: str,
    ) -> bool:
        return any(
            event.result_id == result_id and event.session_id == session_id
            for event in self._events
        )

    def _record_result(
        self,
        result: Phase8IntegrationResult,
        session_id: str,
    ) -> None:
        event = self._event(
            kind=_event_kind_for(result),
            reason=result.reason,
            session_id=session_id,
            result_id=result.result_id,
            packet_id=result.packet.packet_id if result.packet is not None else None,
            metadata={"status": result.status.value},
        )

        with self._lock:
            self._results.append(result)
            self._events.append(event)
            self._last_reason = result.reason

            session = self._sessions.get(session_id)
            if session is not None:
                updates = _session_updates_for(result, session)
                self._sessions[session_id] = session.model_copy(update=updates)

    @staticmethod
    def _event(
        *,
        kind: Phase8IntegrationEventKind,
        reason: Phase8IntegrationReason,
        session_id: str | None = None,
        result_id: str | None = None,
        packet_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Phase8IntegrationRuntimeEvent:
        return Phase8IntegrationRuntimeEvent(
            kind=kind,
            reason=reason,
            session_id=session_id,
            result_id=result_id,
            packet_id=packet_id,
            metadata=metadata or {},
        )


def _packet(
    *,
    request: Phase8IntegrationRequest,
    target_phase: str,
    mode: Phase8BridgeMode,
    reason_payload: dict[str, Any],
    policy_required: bool | None = None,
    gateway_required: bool | None = None,
) -> Phase8IntegrationPacket:
    payload = dict(request.payload)
    payload.update(reason_payload)

    return Phase8IntegrationPacket(
        boundary=request.boundary,
        source_phase=request.source_phase,
        target_phase=target_phase,
        workspace_id=request.workspace_id,
        payload=payload,
        mode=mode,
        policy_required=(
            request.policy_required
            if policy_required is None
            else policy_required
        ),
        gateway_required=(
            request.gateway_required
            if gateway_required is None
            else gateway_required
        ),
        user_visible=request.user_visible,
    )


def _target_phase_for(boundary: Phase8IntegrationBoundary) -> str:
    mapping = {
        Phase8IntegrationBoundary.PHASE1_EVENT_BUS: "phase1",
        Phase8IntegrationBoundary.PHASE2_VOICE_ENVIRONMENT_FUSION: "phase2",
        Phase8IntegrationBoundary.PHASE3_COGNITION_FUSED_CONTEXT: "phase3",
        Phase8IntegrationBoundary.PHASE4_MEMORY_GATEWAY: "phase4",
        Phase8IntegrationBoundary.PHASE5_ACTION_POLICY: "phase5",
        Phase8IntegrationBoundary.PHASE6_ORCHESTRATION: "phase6",
        Phase8IntegrationBoundary.PHASE7_VISUAL_STREAMING: "phase7",
        Phase8IntegrationBoundary.PHASE8_ENVIRONMENT_AWARENESS: "phase8",
    }
    return mapping[boundary]


def _mode_for(boundary: Phase8IntegrationBoundary) -> Phase8BridgeMode:
    if boundary == Phase8IntegrationBoundary.PHASE7_VISUAL_STREAMING:
        return Phase8BridgeMode.STREAMING

    if boundary in {
        Phase8IntegrationBoundary.PHASE4_MEMORY_GATEWAY,
        Phase8IntegrationBoundary.PHASE5_ACTION_POLICY,
    }:
        return Phase8BridgeMode.POLICY_GATED

    if boundary in {
        Phase8IntegrationBoundary.PHASE2_VOICE_ENVIRONMENT_FUSION,
        Phase8IntegrationBoundary.PHASE6_ORCHESTRATION,
    }:
        return Phase8BridgeMode.PREPARE_ONLY

    return Phase8BridgeMode.READ_ONLY


def _payload_requests_direct_action(payload: dict[str, Any]) -> bool:
    forbidden = {
        "execute_now",
        "direct_click",
        "direct_type",
        "direct_shell",
        "direct_file_write",
        "bypass_policy",
    }
    return any(bool(payload.get(key)) for key in forbidden)


def _payload_bypasses_policy(request: Phase8IntegrationRequest) -> bool:
    if request.boundary == Phase8IntegrationBoundary.PHASE5_ACTION_POLICY:
        return not request.policy_required

    if request.boundary == Phase8IntegrationBoundary.PHASE4_MEMORY_GATEWAY:
        return not request.gateway_required

    return False


def _reason_for_boundary(
    boundary: Phase8IntegrationBoundary,
) -> Phase8IntegrationReason:
    mapping = {
        Phase8IntegrationBoundary.PHASE1_EVENT_BUS: (
            Phase8IntegrationReason.EVENT_BUS_BRIDGED
        ),
        Phase8IntegrationBoundary.PHASE2_VOICE_ENVIRONMENT_FUSION: (
            Phase8IntegrationReason.VOICE_ENVIRONMENT_FUSED
        ),
        Phase8IntegrationBoundary.PHASE3_COGNITION_FUSED_CONTEXT: (
            Phase8IntegrationReason.COGNITION_CONTEXT_INJECTED
        ),
        Phase8IntegrationBoundary.PHASE4_MEMORY_GATEWAY: (
            Phase8IntegrationReason.MEMORY_GATEWAY_BRIDGED
        ),
        Phase8IntegrationBoundary.PHASE5_ACTION_POLICY: (
            Phase8IntegrationReason.ACTION_POLICY_BRIDGED
        ),
        Phase8IntegrationBoundary.PHASE6_ORCHESTRATION: (
            Phase8IntegrationReason.ORCHESTRATION_BRIDGED
        ),
        Phase8IntegrationBoundary.PHASE7_VISUAL_STREAMING: (
            Phase8IntegrationReason.VISUAL_STREAMING_BRIDGED
        ),
        Phase8IntegrationBoundary.PHASE8_ENVIRONMENT_AWARENESS: (
            Phase8IntegrationReason.ENVIRONMENT_AWARENESS_BRIDGED
        ),
    }
    return mapping[boundary]


def _event_kind_for(result: Phase8IntegrationResult) -> Phase8IntegrationEventKind:
    if result.status == Phase8IntegrationStatus.INTEGRATED:
        return Phase8IntegrationEventKind.BRIDGE_INTEGRATED

    if result.status == Phase8IntegrationStatus.VERIFIED:
        return Phase8IntegrationEventKind.INTEGRATION_VERIFIED

    return Phase8IntegrationEventKind.INTEGRATION_BLOCKED


def _session_updates_for(
    result: Phase8IntegrationResult,
    session: Phase8IntegrationSession,
) -> dict[str, Any]:
    updates: dict[str, Any] = {
        "updated_at": utc_now(),
        "integrated_count": session.integrated_count
        + (1 if result.status == Phase8IntegrationStatus.INTEGRATED else 0),
        "verified_count": session.verified_count
        + (1 if result.status == Phase8IntegrationStatus.VERIFIED else 0),
        "blocked_count": session.blocked_count
        + (
            1
            if result.status
            in {
                Phase8IntegrationStatus.BLOCKED,
                Phase8IntegrationStatus.FAILED,
            }
            else 0
        ),
    }

    if result.packet is None:
        return updates

    mapping = {
        Phase8IntegrationBoundary.PHASE1_EVENT_BUS: "phase1_event_bus_ready",
        Phase8IntegrationBoundary.PHASE2_VOICE_ENVIRONMENT_FUSION: (
            "phase2_voice_fusion_ready"
        ),
        Phase8IntegrationBoundary.PHASE3_COGNITION_FUSED_CONTEXT: (
            "phase3_cognition_context_ready"
        ),
        Phase8IntegrationBoundary.PHASE4_MEMORY_GATEWAY: (
            "phase4_memory_gateway_ready"
        ),
        Phase8IntegrationBoundary.PHASE5_ACTION_POLICY: (
            "phase5_action_policy_ready"
        ),
        Phase8IntegrationBoundary.PHASE6_ORCHESTRATION: (
            "phase6_orchestration_ready"
        ),
        Phase8IntegrationBoundary.PHASE7_VISUAL_STREAMING: (
            "phase7_visual_streaming_ready"
        ),
        Phase8IntegrationBoundary.PHASE8_ENVIRONMENT_AWARENESS: (
            "phase8_awareness_ready"
        ),
    }
    updates[mapping[result.packet.boundary]] = True
    return updates


def _blocked_result(
    *,
    status: Phase8IntegrationStatus,
    decision: Phase8IntegrationDecision,
    reason: Phase8IntegrationReason,
    message: str,
) -> Phase8IntegrationResult:
    return Phase8IntegrationResult(
        status=status,
        decision=decision,
        reason=reason,
        trust=_trust(confidence=0.20, reason=message),
        message=message,
    )


def _trust(
    *,
    confidence: float,
    reason: str,
) -> TrustCalibration:
    return TrustCalibration(
        confidence=confidence,
        stability=max(0.0, min(1.0, confidence + 0.05)),
        ambiguity=1.0 - confidence,
        source=EnvironmentSource.OS_OBSERVER,
        reason=reason,
        metadata={"policy": TrustPolicyClassification.SAFE.value},
    )


def _clean_required(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("field cannot be empty.")
    return cleaned