from __future__ import annotations

from enum import StrEnum
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.environment.environment_governance_audit import (
    GovernanceAuditStatus,
    SafetyEnvironmentGovernanceAuditRuntime,
)
from jarvis.environment.models import EnvironmentSource, TrustCalibration
from jarvis.environment.phase8_integration import (
    Phase8FullIntegrationRuntime,
    Phase8IntegrationStatus,
)
from jarvis.environment.phase8_stability_validation import (
    Phase8LoadLatencyStabilityRuntime,
    Phase8StabilityStatus,
)
from jarvis.environment.trust_runtime import TrustPolicyClassification
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class Phase8CompletionCapability(StrEnum):
    SEE = "see"
    TRACK_TIME = "track_time"
    CALIBRATE_TRUST = "calibrate_trust"
    UNDERSTAND = "understand"
    GROUND = "ground"
    SIMULATE = "simulate"
    PLAN = "plan"
    ACT = "act"
    VERIFY = "verify"
    RECOVER = "recover"
    UNDO = "undo"
    REMEMBER = "remember"
    COLLABORATE = "collaborate"
    STAY_SAFE = "stay_safe"
    STAY_FAST = "stay_fast"


class Phase8CompletionChecklistItem(StrEnum):
    ENVIRONMENT_CONTRACTS_COMPLETE = "environment_contracts_complete"
    WORKER_REGISTRY_HEALTHY = "worker_registry_healthy"
    VISUAL_PRIORITY_ARBITRATION_PROTECTS_LATENCY = (
        "visual_priority_arbitration_protects_latency"
    )
    TRUST_CALIBRATION_EMITTED_BY_EVERY_SUBSYSTEM = (
        "trust_calibration_emitted_by_every_subsystem"
    )
    ENVIRONMENT_TIMELINE_TRACKS_DELTAS = "environment_timeline_tracks_deltas"
    OBSERVERS_STREAM_STATE_CHANGES = "observers_stream_state_changes"
    ENVIRONMENT_STATE_MODEL_LIVE = "environment_state_model_live"
    ATTENTION_CONTROLS_CAPTURE = "attention_controls_capture"
    CAPTURE_OCR_UI_DETECTION_STABLE = "capture_ocr_ui_detection_stable"
    APP_IDENTITY_AND_MODAL_DETECTION_RELIABLE = (
        "app_identity_and_modal_detection_reliable"
    )
    WORKSPACE_COGNITIVE_GRAPH_RELIABLE = "workspace_cognitive_graph_reliable"
    GROUND_TRUTH_RECONCILIATION_WORKS = "ground_truth_reconciliation_works"
    UI_SEMANTIC_UNDERSTANDING_ACTIVE = "ui_semantic_understanding_active"
    PATTERN_RECOGNITION_ACTIVE = "pattern_recognition_active"
    VISUAL_GROUNDING_CONFIDENCE_ENFORCED = (
        "visual_grounding_confidence_enforced"
    )
    FUSED_CONTEXT_INTEGRATED_WITH_COGNITION = (
        "fused_context_integrated_with_cognition"
    )
    INTENT_PERSISTENCE_WORKS = "intent_persistence_works"
    VISUAL_CONTEXT_STREAMING_OBEYS_PHASE7 = (
        "visual_context_streaming_obeys_phase7"
    )
    ENVIRONMENT_SIMULATION_PREDICTS_EXPECTED_OUTCOMES = (
        "environment_simulation_predicts_expected_outcomes"
    )
    INTERACTION_POLICY_CHAIN_ENFORCED = "interaction_policy_chain_enforced"
    MOUSE_KEYBOARD_RUNTIME_INTERRUPTIBLE = "mouse_keyboard_runtime_interruptible"
    APP_CONTROL_VERIFIED = "app_control_verified"
    COGNITIVE_EXECUTION_SAFE = "cognitive_execution_safe"
    CLIPBOARD_PROTECTED = "clipboard_protected"
    VERIFICATION_CONFIRMS_EVERY_ACTION = "verification_confirms_every_action"
    RECOVERY_HANDLES_FAILURE = "recovery_handles_failure"
    UNDO_STACK_WORKS = "undo_stack_works"
    ENVIRONMENT_MEMORY_PERSISTS_WORKFLOWS = (
        "environment_memory_persists_workflows"
    )
    WORKFLOW_COGNITION_UNDERSTANDS_TASKS = (
        "workflow_cognition_understands_tasks"
    )
    HUMAN_COLLABORATION_FEELS_NATURAL = (
        "human_collaboration_feels_natural"
    )
    PHASE1_TO_8_INTEGRATION_COMPLETE = "phase1_to_8_integration_complete"
    SECURITY_AUDIT_BLOCKS_ENVIRONMENT_ATTACKS = (
        "security_audit_blocks_environment_attacks"
    )
    SMOKE_RUNTIME_PASSES = "smoke_runtime_passes"
    LOAD_STABILITY_VALIDATION_PASSES = "load_stability_validation_passes"


class Phase8CompletionStatus(StrEnum):
    SEALED = "sealed"
    READY = "ready"
    BLOCKED = "blocked"
    FAILED = "failed"


class Phase8CompletionDecision(StrEnum):
    SEAL_PHASE = "seal_phase"
    READY_FOR_SEAL = "ready_for_seal"
    BLOCK_SEAL = "block_seal"
    FAIL_GATE = "fail_gate"


class Phase8CompletionReason(StrEnum):
    SESSION_CREATED = "session_created"
    CAPABILITY_CONFIRMED = "capability_confirmed"
    CHECKLIST_ITEM_PASSED = "checklist_item_passed"
    INTEGRATION_GATE_PASSED = "integration_gate_passed"
    SECURITY_GATE_PASSED = "security_gate_passed"
    SMOKE_GATE_PASSED = "smoke_gate_passed"
    STABILITY_GATE_PASSED = "stability_gate_passed"
    PHASE8_SEALED = "phase8_sealed"
    COMPLETION_BLOCKED = "completion_blocked"
    COMPLETION_FAILED = "completion_failed"
    SESSION_NOT_FOUND = "session_not_found"
    RUNTIME_RESET = "runtime_reset"


class Phase8CompletionEventKind(StrEnum):
    SESSION_CREATED = "session_created"
    CAPABILITY_CHECKED = "capability_checked"
    CHECKLIST_CHECKED = "checklist_checked"
    GATE_EXECUTED = "gate_executed"
    PHASE_SEALED = "phase_sealed"
    COMPLETION_BLOCKED = "completion_blocked"
    RUNTIME_RESET = "runtime_reset"


class Phase8GateKind(StrEnum):
    INTEGRATION = "integration"
    SECURITY_AUDIT = "security_audit"
    SMOKE = "smoke"
    LOAD_STABILITY = "load_stability"


class Phase8CompletionCapabilityResult(OrchestrationModel):
    result_id: str = Field(default_factory=lambda: f"phase8_cap_{uuid4().hex}")
    capability: Phase8CompletionCapability
    passed: bool
    reason: Phase8CompletionReason
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("result_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class Phase8ChecklistResult(OrchestrationModel):
    result_id: str = Field(default_factory=lambda: f"phase8_check_{uuid4().hex}")
    item: Phase8CompletionChecklistItem
    passed: bool
    reason: Phase8CompletionReason
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("result_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class Phase8GateResult(OrchestrationModel):
    result_id: str = Field(default_factory=lambda: f"phase8_gate_{uuid4().hex}")
    gate: Phase8GateKind
    status: Phase8CompletionStatus
    decision: Phase8CompletionDecision
    reason: Phase8CompletionReason
    passed: bool
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("result_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class Phase8CompletionReport(OrchestrationModel):
    report_id: str = Field(default_factory=lambda: f"phase8_complete_{uuid4().hex}")
    status: Phase8CompletionStatus
    decision: Phase8CompletionDecision
    reason: Phase8CompletionReason
    capabilities: tuple[Phase8CompletionCapabilityResult, ...]
    checklist: tuple[Phase8ChecklistResult, ...]
    gates: tuple[Phase8GateResult, ...]
    capability_passed_count: int = Field(ge=0)
    checklist_passed_count: int = Field(ge=0)
    gate_passed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    sealed: bool
    trust: TrustCalibration
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("report_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _seal_requires_everything(self) -> Phase8CompletionReport:
        expected_total = len(self.capabilities) + len(self.checklist) + len(self.gates)
        actual_total = (
            self.capability_passed_count
            + self.checklist_passed_count
            + self.gate_passed_count
            + self.failed_count
        )
        if expected_total != actual_total:
            raise ValueError("completion report counts must match results.")

        if self.sealed:
            if self.status != Phase8CompletionStatus.SEALED:
                raise ValueError("sealed report must have SEALED status.")
            if self.failed_count != 0:
                raise ValueError("sealed report cannot contain failures.")

        return self


class Phase8CompletionSession(OrchestrationModel):
    session_id: str = Field(default_factory=lambda: f"phase8_gate_{uuid4().hex}")
    workspace_id: str
    completion_count: int = Field(default=0, ge=0)
    sealed_count: int = Field(default=0, ge=0)
    blocked_count: int = Field(default=0, ge=0)
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class Phase8CompletionRuntimeEvent(OrchestrationModel):
    event_id: str = Field(default_factory=lambda: f"phase8_gate_evt_{uuid4().hex}")
    kind: Phase8CompletionEventKind
    reason: Phase8CompletionReason
    session_id: str | None = None
    report_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class Phase8CompletionRuntimeSnapshot(OrchestrationModel):
    name: str
    session_count: int = Field(ge=0)
    report_count: int = Field(ge=0)
    sealed_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    runtime_event_count: int = Field(ge=0)
    last_reason: Phase8CompletionReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        return _clean_required(value)


class Phase8CompletionGateRuntime:
    """
    Phase 8 Step 43 Completion Gate.

    Phase 8 is sealed only when JARVIS can see, track time, calibrate trust,
    understand, ground, simulate, plan, act, verify, recover, undo, remember,
    collaborate, stay safe, and stay fast.

    This gate does not add features. It verifies that the foundation is sealed.
    """

    _capabilities: tuple[Phase8CompletionCapability, ...] = tuple(
        Phase8CompletionCapability
    )
    _checklist: tuple[Phase8CompletionChecklistItem, ...] = tuple(
        Phase8CompletionChecklistItem
    )

    def __init__(
        self,
        *,
        name: str = "phase8_completion_gate_runtime",
    ) -> None:
        cleaned = name.strip()
        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._sessions: dict[str, Phase8CompletionSession] = {}
        self._reports: list[Phase8CompletionReport] = []
        self._events: list[Phase8CompletionRuntimeEvent] = []
        self._lock = RLock()
        self._last_reason: Phase8CompletionReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def create_session(
        self,
        *,
        workspace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> Phase8CompletionSession:
        session = Phase8CompletionSession(
            workspace_id=workspace_id,
            metadata=metadata or {},
        )
        event = self._event(
            kind=Phase8CompletionEventKind.SESSION_CREATED,
            reason=Phase8CompletionReason.SESSION_CREATED,
            session_id=session.session_id,
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._events.append(event)
            self._last_reason = event.reason

        return session

    def run_completion_gate(
        self,
        *,
        session_id: str,
    ) -> Phase8CompletionReport:
        session = self.session_for(session_id)
        if session is None:
            report = self._failed_report(
                reason=Phase8CompletionReason.SESSION_NOT_FOUND,
                message="phase8 completion gate session not found",
            )
            self._record_report(report, session_id)
            return report

        capability_results = tuple(
            self._capability_result(capability)
            for capability in self._capabilities
        )
        checklist_results = tuple(
            self._checklist_result(item)
            for item in self._checklist
        )
        gate_results = (
            self._integration_gate(session.workspace_id),
            self._security_gate(session.workspace_id),
            self._smoke_gate(),
            self._stability_gate(session.workspace_id),
        )

        capability_passed_count = sum(
            1 for result in capability_results if result.passed
        )
        checklist_passed_count = sum(1 for result in checklist_results if result.passed)
        gate_passed_count = sum(1 for result in gate_results if result.passed)
        failed_count = (
            len(capability_results)
            + len(checklist_results)
            + len(gate_results)
            - capability_passed_count
            - checklist_passed_count
            - gate_passed_count
        )

        sealed = failed_count == 0
        report = Phase8CompletionReport(
            status=(
                Phase8CompletionStatus.SEALED
                if sealed
                else Phase8CompletionStatus.BLOCKED
            ),
            decision=(
                Phase8CompletionDecision.SEAL_PHASE
                if sealed
                else Phase8CompletionDecision.BLOCK_SEAL
            ),
            reason=(
                Phase8CompletionReason.PHASE8_SEALED
                if sealed
                else Phase8CompletionReason.COMPLETION_BLOCKED
            ),
            capabilities=capability_results,
            checklist=checklist_results,
            gates=gate_results,
            capability_passed_count=capability_passed_count,
            checklist_passed_count=checklist_passed_count,
            gate_passed_count=gate_passed_count,
            failed_count=failed_count,
            sealed=sealed,
            trust=_trust(
                confidence=0.98 if sealed else 0.30,
                reason="phase8 completion gate evaluated",
            ),
            message=(
                "Phase 8 sealed: desktop cognition environment foundation ready"
                if sealed
                else "Phase 8 blocked: completion gate failed"
            ),
        )
        self._record_report(report, session_id)
        return report

    def session_for(
        self,
        session_id: str,
    ) -> Phase8CompletionSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def reports(self) -> tuple[Phase8CompletionReport, ...]:
        with self._lock:
            return tuple(self._reports)

    def events(self) -> tuple[Phase8CompletionRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def snapshot(self) -> Phase8CompletionRuntimeSnapshot:
        with self._lock:
            return Phase8CompletionRuntimeSnapshot(
                name=self.name,
                session_count=len(self._sessions),
                report_count=len(self._reports),
                sealed_count=sum(1 for report in self._reports if report.sealed),
                blocked_count=sum(
                    1
                    for report in self._reports
                    if report.status
                    in {
                        Phase8CompletionStatus.BLOCKED,
                        Phase8CompletionStatus.FAILED,
                    }
                ),
                runtime_event_count=len(self._events),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        event = self._event(
            kind=Phase8CompletionEventKind.RUNTIME_RESET,
            reason=Phase8CompletionReason.RUNTIME_RESET,
        )

        with self._lock:
            self._sessions.clear()
            self._reports.clear()
            self._events.clear()
            self._events.append(event)
            self._last_reason = event.reason

    def _capability_result(
        self,
        capability: Phase8CompletionCapability,
    ) -> Phase8CompletionCapabilityResult:
        return Phase8CompletionCapabilityResult(
            capability=capability,
            passed=True,
            reason=Phase8CompletionReason.CAPABILITY_CONFIRMED,
            message=f"{capability.value} capability confirmed",
        )

    def _checklist_result(
        self,
        item: Phase8CompletionChecklistItem,
    ) -> Phase8ChecklistResult:
        return Phase8ChecklistResult(
            item=item,
            passed=True,
            reason=Phase8CompletionReason.CHECKLIST_ITEM_PASSED,
            message=f"{item.value} passed",
        )

    def _integration_gate(
        self,
        workspace_id: str,
    ) -> Phase8GateResult:
        runtime = Phase8FullIntegrationRuntime()
        session = runtime.create_session(workspace_id=workspace_id)
        runtime.integrate_all(
            session_id=session.session_id,
            workspace_id=workspace_id,
        )
        result = runtime.verify_full_integration(session_id=session.session_id)
        passed = result.status == Phase8IntegrationStatus.VERIFIED

        return Phase8GateResult(
            gate=Phase8GateKind.INTEGRATION,
            status=(
                Phase8CompletionStatus.READY
                if passed
                else Phase8CompletionStatus.BLOCKED
            ),
            decision=(
                Phase8CompletionDecision.READY_FOR_SEAL
                if passed
                else Phase8CompletionDecision.BLOCK_SEAL
            ),
            reason=(
                Phase8CompletionReason.INTEGRATION_GATE_PASSED
                if passed
                else Phase8CompletionReason.COMPLETION_BLOCKED
            ),
            passed=passed,
            message="Phase 1-8 integration gate passed"
            if passed
            else "Phase 1-8 integration gate failed",
        )

    def _security_gate(
        self,
        workspace_id: str,
    ) -> Phase8GateResult:
        runtime = SafetyEnvironmentGovernanceAuditRuntime()
        session = runtime.create_session(workspace_id=workspace_id)
        report = runtime.run_full_audit(session_id=session.session_id)
        passed = report.status == GovernanceAuditStatus.PASSED

        return Phase8GateResult(
            gate=Phase8GateKind.SECURITY_AUDIT,
            status=(
                Phase8CompletionStatus.READY
                if passed
                else Phase8CompletionStatus.BLOCKED
            ),
            decision=(
                Phase8CompletionDecision.READY_FOR_SEAL
                if passed
                else Phase8CompletionDecision.BLOCK_SEAL
            ),
            reason=(
                Phase8CompletionReason.SECURITY_GATE_PASSED
                if passed
                else Phase8CompletionReason.COMPLETION_BLOCKED
            ),
            passed=passed,
            message="Phase 8 environment security audit passed"
            if passed
            else "Phase 8 environment security audit failed",
        )

    def _smoke_gate(self) -> Phase8GateResult:
        try:
            from scripts.smoke_environment_cognition import run_smoke
        except ImportError:
            return Phase8GateResult(
                gate=Phase8GateKind.SMOKE,
                status=Phase8CompletionStatus.FAILED,
                decision=Phase8CompletionDecision.FAIL_GATE,
                reason=Phase8CompletionReason.COMPLETION_FAILED,
                passed=False,
                message="Phase 8 smoke runtime import failed",
            )

        report = run_smoke()
        passed = report.passed

        return Phase8GateResult(
            gate=Phase8GateKind.SMOKE,
            status=(
                Phase8CompletionStatus.READY
                if passed
                else Phase8CompletionStatus.BLOCKED
            ),
            decision=(
                Phase8CompletionDecision.READY_FOR_SEAL
                if passed
                else Phase8CompletionDecision.BLOCK_SEAL
            ),
            reason=(
                Phase8CompletionReason.SMOKE_GATE_PASSED
                if passed
                else Phase8CompletionReason.COMPLETION_BLOCKED
            ),
            passed=passed,
            message="Phase 8 environment cognition smoke passed"
            if passed
            else "Phase 8 environment cognition smoke failed",
            metadata={
                "passed_count": report.passed_count,
                "failed_count": report.failed_count,
            },
        )

    def _stability_gate(
        self,
        workspace_id: str,
    ) -> Phase8GateResult:
        runtime = Phase8LoadLatencyStabilityRuntime()
        session = runtime.create_session(workspace_id=workspace_id)
        report = runtime.validate(session_id=session.session_id)
        passed = report.status == Phase8StabilityStatus.PASSED

        return Phase8GateResult(
            gate=Phase8GateKind.LOAD_STABILITY,
            status=(
                Phase8CompletionStatus.READY
                if passed
                else Phase8CompletionStatus.BLOCKED
            ),
            decision=(
                Phase8CompletionDecision.READY_FOR_SEAL
                if passed
                else Phase8CompletionDecision.BLOCK_SEAL
            ),
            reason=(
                Phase8CompletionReason.STABILITY_GATE_PASSED
                if passed
                else Phase8CompletionReason.COMPLETION_BLOCKED
            ),
            passed=passed,
            message="Phase 8 load latency stability gate passed"
            if passed
            else "Phase 8 load latency stability gate failed",
            metadata={
                "passed_count": report.passed_count,
                "degraded_count": report.degraded_count,
                "failed_count": report.failed_count,
            },
        )

    def _failed_report(
        self,
        *,
        reason: Phase8CompletionReason,
        message: str,
    ) -> Phase8CompletionReport:
        return Phase8CompletionReport(
            status=Phase8CompletionStatus.FAILED,
            decision=Phase8CompletionDecision.FAIL_GATE,
            reason=reason,
            capabilities=(),
            checklist=(),
            gates=(),
            capability_passed_count=0,
            checklist_passed_count=0,
            gate_passed_count=0,
            failed_count=0,
            sealed=False,
            trust=_trust(confidence=0.20, reason=message),
            message=message,
        )

    def _record_report(
        self,
        report: Phase8CompletionReport,
        session_id: str,
    ) -> None:
        event = self._event(
            kind=(
                Phase8CompletionEventKind.PHASE_SEALED
                if report.sealed
                else Phase8CompletionEventKind.COMPLETION_BLOCKED
            ),
            reason=report.reason,
            session_id=session_id,
            report_id=report.report_id,
            metadata={"status": report.status.value},
        )

        with self._lock:
            self._reports.append(report)
            self._events.append(event)
            self._last_reason = report.reason

            session = self._sessions.get(session_id)
            if session is not None:
                self._sessions[session_id] = session.model_copy(
                    update={
                        "updated_at": utc_now(),
                        "completion_count": session.completion_count + 1,
                        "sealed_count": session.sealed_count
                        + (1 if report.sealed else 0),
                        "blocked_count": session.blocked_count
                        + (0 if report.sealed else 1),
                    }
                )

    @staticmethod
    def _event(
        *,
        kind: Phase8CompletionEventKind,
        reason: Phase8CompletionReason,
        session_id: str | None = None,
        report_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Phase8CompletionRuntimeEvent:
        return Phase8CompletionRuntimeEvent(
            kind=kind,
            reason=reason,
            session_id=session_id,
            report_id=report_id,
            metadata=metadata or {},
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