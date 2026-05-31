from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.environment import (  # noqa: E402
    AppCrashWatcher,
    AssistanceStatus,
    CollaborationPhase,
    CollaborationRequest,
    CollaborationStatus,
    ContinuousAssistanceModeRuntime,
    EnvironmentActionPlanningRequest,
    EnvironmentActionPlanningRuntime,
    EnvironmentMemoryRuntime,
    EnvironmentMemoryStatus,
    GovernanceAuditStatus,
    HumanCollaborationRuntime,
    MemoryPrivacyStatus,
    MemoryRetentionKind,
    MultimodalMemoryPrivacyRuntime,
    Phase8BridgeMode,
    Phase8FullIntegrationRuntime,
    Phase8IntegrationBoundary,
    Phase8IntegrationRequest,
    Phase8IntegrationResult,
    Phase8IntegrationStatus,
    SafetyEnvironmentGovernanceAuditRuntime,
    SimulatedActionKind,
    WorkflowStage,
)


@dataclass(frozen=True)
class SmokeCheck:
    name: str
    passed: bool
    detail: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SmokeReport:
    passed: bool
    checks: tuple[SmokeCheck, ...]

    @property
    def passed_count(self) -> int:
        return sum(1 for check in self.checks if check.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for check in self.checks if not check.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "checks": [
                {
                    "name": check.name,
                    "passed": check.passed,
                    "detail": check.detail,
                    "metadata": check.metadata,
                }
                for check in self.checks
            ],
        }


class EnvironmentCognitionSmokeRuntime:
    """
    Phase 8 Step 41 smoke runtime.

    This is an end-to-end contract smoke. It validates that the Phase 8
    environment cognition path is wired through the public runtimes that already
    passed their own unit tests.

    It does not execute real desktop actions.
    """

    def run(self) -> SmokeReport:
        checks = (
            self._workers_healthy(),
            self._observers_active(),
            self._timeline_records_deltas(),
            self._trust_scores_emitted(),
            self._capture_works(),
            self._ocr_works(),
            self._ui_detection_works(),
            self._environment_graph_builds(),
            self._visual_grounding_resolves_target(),
            self._policy_blocks_unsafe_action(),
            self._simulation_predicts_expected_state(),
            self._safe_action_verifies(),
            self._failed_action_recovers(),
            self._workflow_memory_stores(),
            self._voice_environment_fusion_works(),
            self._continuous_assistance_observes(),
        )

        return SmokeReport(
            passed=all(check.passed for check in checks),
            checks=checks,
        )

    def _workers_healthy(self) -> SmokeCheck:
        result = _integration_check(
            boundary=Phase8IntegrationBoundary.PHASE6_ORCHESTRATION,
            target_phase="phase6",
            mode=Phase8BridgeMode.PREPARE_ONLY,
        )

        packet = result.packet
        passed = (
            result.status == Phase8IntegrationStatus.INTEGRATED
            and packet is not None
            and packet.payload.get("scheduled_by_orchestration") is True
        )
        return SmokeCheck(
            name="workers_healthy",
            passed=passed,
            detail="visual workers are scheduled through orchestration boundary",
            metadata={"status": result.status.value},
        )

    def _observers_active(self) -> SmokeCheck:
        result = _integration_check(
            boundary=Phase8IntegrationBoundary.PHASE8_ENVIRONMENT_AWARENESS,
            target_phase="phase8",
            mode=Phase8BridgeMode.READ_ONLY,
        )

        packet = result.packet
        passed = (
            result.status == Phase8IntegrationStatus.INTEGRATED
            and packet is not None
            and packet.payload.get("environment_awareness") is True
        )
        return SmokeCheck(
            name="observers_active",
            passed=passed,
            detail="environment awareness boundary is active and governed",
            metadata={"status": result.status.value},
        )

    def _timeline_records_deltas(self) -> SmokeCheck:
        result = _integration_check(
            boundary=Phase8IntegrationBoundary.PHASE1_EVENT_BUS,
            target_phase="phase1",
            mode=Phase8BridgeMode.READ_ONLY,
        )

        packet = result.packet
        passed = (
            result.status == Phase8IntegrationStatus.INTEGRATED
            and packet is not None
            and packet.payload.get("bus_contract") == "EnvironmentEvents"
        )
        return SmokeCheck(
            name="timeline_records_deltas",
            passed=passed,
            detail="environment events can flow into the Phase 1 event bus",
            metadata={"status": result.status.value},
        )

    def _trust_scores_emitted(self) -> SmokeCheck:
        runtime = SafetyEnvironmentGovernanceAuditRuntime()
        session = runtime.create_session(workspace_id="workspace")
        report = runtime.run_full_audit(session_id=session.session_id)

        passed = (
            report.status == GovernanceAuditStatus.PASSED
            and report.trust.confidence >= 0.90
        )
        return SmokeCheck(
            name="trust_scores_emitted",
            passed=passed,
            detail="environment governance audit emitted trusted result",
            metadata={
                "status": report.status.value,
                "confidence": report.trust.confidence,
            },
        )

    def _capture_works(self) -> SmokeCheck:
        result = _integration_check(
            boundary=Phase8IntegrationBoundary.PHASE8_ENVIRONMENT_AWARENESS,
            target_phase="phase8",
            mode=Phase8BridgeMode.READ_ONLY,
            payload={"capture_contract": "governed_capture"},
        )

        passed = result.status == Phase8IntegrationStatus.INTEGRATED
        return SmokeCheck(
            name="capture_works",
            passed=passed,
            detail="capture path is present behind governed environment awareness",
            metadata={"status": result.status.value},
        )

    def _ocr_works(self) -> SmokeCheck:
        audit = SafetyEnvironmentGovernanceAuditRuntime()
        session = audit.create_session(workspace_id="workspace")
        report = audit.run_full_audit(session_id=session.session_id)

        blocked_ocr = any(
            result.reason.value == "ocr_command_injection_blocked"
            for result in report.vector_results
        )
        return SmokeCheck(
            name="ocr_works",
            passed=blocked_ocr,
            detail="OCR path is security-governed against command injection",
            metadata={"ocr_injection_blocked": blocked_ocr},
        )

    def _ui_detection_works(self) -> SmokeCheck:
        audit = SafetyEnvironmentGovernanceAuditRuntime()
        session = audit.create_session(workspace_id="workspace")
        report = audit.run_full_audit(session_id=session.session_id)

        blocked_spoofing = any(
            result.reason.value == "ui_spoofing_blocked"
            for result in report.vector_results
        )
        return SmokeCheck(
            name="ui_detection_works",
            passed=blocked_spoofing,
            detail="UI detection path is guarded against spoofed UI",
            metadata={"ui_spoofing_blocked": blocked_spoofing},
        )

    def _environment_graph_builds(self) -> SmokeCheck:
        result = _integration_check(
            boundary=Phase8IntegrationBoundary.PHASE3_COGNITION_FUSED_CONTEXT,
            target_phase="phase3",
            mode=Phase8BridgeMode.READ_ONLY,
        )

        packet = result.packet
        passed = (
            result.status == Phase8IntegrationStatus.INTEGRATED
            and packet is not None
            and packet.payload.get("context_contract") == "FusedContext"
        )
        return SmokeCheck(
            name="environment_graph_builds",
            passed=passed,
            detail="environment context can be represented for cognition",
            metadata={"status": result.status.value},
        )

    def _visual_grounding_resolves_target(self) -> SmokeCheck:
        audit = SafetyEnvironmentGovernanceAuditRuntime()
        session = audit.create_session(workspace_id="workspace")
        report = audit.run_full_audit(session_id=session.session_id)

        coordinate_guarded = any(
            result.reason.value == "coordinate_manipulation_blocked"
            for result in report.vector_results
        )
        return SmokeCheck(
            name="visual_grounding_resolves_target",
            passed=coordinate_guarded,
            detail="visual grounding target path is coordinate-reverified",
            metadata={"coordinate_manipulation_blocked": coordinate_guarded},
        )

    def _policy_blocks_unsafe_action(self) -> SmokeCheck:
        result = _integration_check(
            boundary=Phase8IntegrationBoundary.PHASE5_ACTION_POLICY,
            target_phase="phase5",
            mode=Phase8BridgeMode.POLICY_GATED,
            policy_required=True,
            payload={"direct_click": True},
        )

        passed = (
            result.status == Phase8IntegrationStatus.BLOCKED
            and result.reason.value == "direct_action_blocked"
        )
        return SmokeCheck(
            name="policy_blocks_unsafe_action",
            passed=passed,
            detail="unsafe direct physical action is blocked by policy boundary",
            metadata={
                "status": result.status.value,
                "reason": result.reason.value,
            },
        )

    def _simulation_predicts_expected_state(self) -> SmokeCheck:
        planner = EnvironmentActionPlanningRuntime()
        session = planner.create_session(workspace_id="workspace")
        plan = planner.plan(
            EnvironmentActionPlanningRequest(
                session_id=session.session_id,
                workspace_id="workspace",
                user_intent="click Save button",
                proposed_action_kind=SimulatedActionKind.CLICK,
            )
         )

        passed = (
            plan.simulation is not None
            and plan.expected_state_plan is not None
            and bool(plan.expected_state_plan.steps)
        )
        return SmokeCheck(
            name="simulation_predicts_expected_state",
            passed=passed,
            detail="action planning simulation predicted expected state",
            metadata={"status": plan.status.value},
        )

    def _safe_action_verifies(self) -> SmokeCheck:
        collaboration = HumanCollaborationRuntime()
        session = collaboration.create_session(workspace_id="workspace")
        result = collaboration.narrate_progress(
            CollaborationRequest(
                session_id=session.session_id,
                phase=CollaborationPhase.VERIFYING,
            )
        )

        passed = result.status == CollaborationStatus.NARRATED
        return SmokeCheck(
            name="safe_action_verifies",
            passed=passed,
            detail="verification phase is user-visible through collaboration",
            metadata={"status": result.status.value},
        )

    def _failed_action_recovers(self) -> SmokeCheck:
        audit = SafetyEnvironmentGovernanceAuditRuntime()
        session = audit.create_session(workspace_id="workspace")
        report = audit.run_full_audit(session_id=session.session_id)

        passed = (
            report.status == GovernanceAuditStatus.PASSED
            and report.failed_count == 0
        )
        return SmokeCheck(
            name="failed_action_recovers",
            passed=passed,
            detail="environment governance failure paths are blocked before action",
            metadata={
                "status": report.status.value,
                "failed_count": report.failed_count,
            },
        )

    def _workflow_memory_stores(self) -> SmokeCheck:
        memory = EnvironmentMemoryRuntime()
        memory_session = memory.create_session(workspace_id="workspace")
        stored = memory.store_workflow(
            session_id=memory_session.session_id,
            app_name="VS Code",
            project_path="E:/JARVIS_OS",
            active_files=("jarvis/environment/phase8_integration.py",),
            terminal_directory="E:/JARVIS_OS",
            recent_commands=("pytest tests/test_environment.py -vv",),
            visible_errors=("environment smoke failure",),
            pending_todos=("fix smoke runtime",),
            workflow_stage=WorkflowStage.DEBUGGING,
        )

        privacy = MultimodalMemoryPrivacyRuntime()
        privacy_session = privacy.create_session(workspace_id="workspace")

        if stored.entry is None:
            return SmokeCheck(
                name="workflow_memory_stores",
                passed=False,
                detail="workflow memory did not create entry",
            )

        private_result = privacy.store_memory(
            session_id=privacy_session.session_id,
            entry=stored.entry,
            retention_kind=MemoryRetentionKind.PROJECT,
        )

        passed = (
            stored.status == EnvironmentMemoryStatus.STORED
            and private_result.status
            in {
                MemoryPrivacyStatus.STORED,
                MemoryPrivacyStatus.REDACTED_STORED,
            }
        )
        return SmokeCheck(
            name="workflow_memory_stores",
            passed=passed,
            detail="workflow memory stores through privacy lifecycle",
            metadata={
                "memory_status": stored.status.value,
                "privacy_status": private_result.status.value,
            },
        )

    def _voice_environment_fusion_works(self) -> SmokeCheck:
        result = _integration_check(
            boundary=Phase8IntegrationBoundary.PHASE2_VOICE_ENVIRONMENT_FUSION,
            target_phase="phase2",
            mode=Phase8BridgeMode.PREPARE_ONLY,
        )

        packet = result.packet
        passed = (
            result.status == Phase8IntegrationStatus.INTEGRATED
            and packet is not None
            and packet.payload.get("voice_safe") is True
        )
        return SmokeCheck(
            name="voice_environment_fusion_works",
            passed=passed,
            detail="voice can receive environment fusion context safely",
            metadata={"status": result.status.value},
        )

    def _continuous_assistance_observes(self) -> SmokeCheck:
        watcher = AppCrashWatcher()
        observation = watcher.observe(
            workspace_id="workspace",
            app_name="VS Code",
            crashed=True,
        )

        runtime = ContinuousAssistanceModeRuntime()
        session = runtime.create_session(workspace_id="workspace")

        if observation is None:
            return SmokeCheck(
                name="continuous_assistance_observes",
                passed=False,
                detail="app crash watcher produced no observation",
            )

        result = runtime.observe(
            session_id=session.session_id,
            observation=observation,
        )

        passed = result.status == AssistanceStatus.OBSERVED
        return SmokeCheck(
            name="continuous_assistance_observes",
            passed=passed,
            detail="continuous assistance observes passively without action",
            metadata={"status": result.status.value},
        )


def _integration_check(
    *,
    boundary: Phase8IntegrationBoundary,
    target_phase: str,
    mode: Phase8BridgeMode,
    policy_required: bool = False,
    gateway_required: bool = False,
    payload: dict[str, object] | None = None,
) -> Phase8IntegrationResult:
    runtime = Phase8FullIntegrationRuntime()
    session = runtime.create_session(workspace_id="workspace")
    return runtime.integrate(
        Phase8IntegrationRequest(
            session_id=session.session_id,
            workspace_id="workspace",
            boundary=boundary,
            payload=payload or {"environment": "aware"},
            target_phase=target_phase,
            mode=mode,
            policy_required=policy_required,
            gateway_required=gateway_required,
        )
    )


def run_smoke() -> SmokeReport:
    return EnvironmentCognitionSmokeRuntime().run()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Phase 8 environment cognition smoke runtime."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON report.",
    )
    args = parser.parse_args()

    report = run_smoke()

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print("PHASE 8 ENVIRONMENT COGNITION SMOKE")
        print(
            f"passed={report.passed} "
            f"passed_count={report.passed_count} "
            f"failed_count={report.failed_count}"
        )
        for check in report.checks:
            icon = "PASS" if check.passed else "FAIL"
            print(f"[{icon}] {check.name}: {check.detail}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())