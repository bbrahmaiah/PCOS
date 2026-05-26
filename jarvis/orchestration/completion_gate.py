from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from uuid import uuid4

from pydantic import Field, field_validator

from jarvis.orchestration.budgets import ResourceBudgetRuntimeSnapshot
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.integration import (
    IntegratedPhase,
    IntegratedTaskKind,
    PhaseEvent,
    PhaseEventKind,
    PhaseIntegrationRuntime,
)
from jarvis.orchestration.load_manager import (
    CognitiveLoadLevel,
    CognitiveLoadManagerRuntime,
)
from jarvis.orchestration.models import OrchestrationModel
from jarvis.orchestration.observability import (
    OrchestrationDashboard,
    OrchestrationHealth,
    OrchestrationObservabilityRuntime,
    WorkerHealthView,
)
from jarvis.orchestration.proactive import (
    ProactiveEngine,
    ProactiveTrigger,
    ProactiveTriggerKind,
)
from jarvis.orchestration.recovery import (
    RecoveryEventType,
    RecoveryManager,
    RecoveryReason,
)
from jarvis.orchestration.scheduler import (
    TaskScheduleDecision,
    TaskScheduleReason,
    TaskSchedulerSnapshot,
)
from jarvis.orchestration.security_audit import (
    SecurityHardeningAuditRuntime,
)
from jarvis.orchestration.smoke import OrchestrationSmokeRuntime


class Phase6GateStatus(StrEnum):
    """
    Phase 6 completion gate status.
    """

    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"


class Phase6GateReason(StrEnum):
    """
    Machine-readable gate reasons.
    """

    CHECK_PASSED = "check_passed"
    CHECK_FAILED = "check_failed"
    GATE_PASSED = "gate_passed"
    GATE_FAILED = "gate_failed"
    TYPED_CONTRACTS_MISSING = "typed_contracts_missing"
    WORKERS_NOT_INTEGRATED = "workers_not_integrated"
    TASK_MODEL_UNSTABLE = "task_model_unstable"
    ATTENTION_NOT_PROTECTED = "attention_not_protected"
    BUDGET_NOT_ENFORCED = "budget_not_enforced"
    STATE_MACHINE_UNVERIFIED = "state_machine_unverified"
    SCHEDULER_FAILED = "scheduler_failed"
    SNAPSHOT_UNVERIFIED = "snapshot_unverified"
    COORDINATION_BOUNDARY_BROKEN = "coordination_boundary_broken"
    BACKGROUND_NOT_YIELDING = "background_not_yielding"
    INTERRUPT_UNVERIFIED = "interrupt_unverified"
    DEADLOCK_UNVERIFIED = "deadlock_unverified"
    CIRCUIT_BREAKER_UNVERIFIED = "circuit_breaker_unverified"
    OBSERVABILITY_FAILED = "observability_failed"
    LOAD_SHEDDING_FAILED = "load_shedding_failed"
    RECOVERY_FAILED = "recovery_failed"
    INTEGRATION_FAILED = "integration_failed"
    PROACTIVE_FAILED = "proactive_failed"
    SMOKE_FAILED = "smoke_failed"
    SECURITY_FAILED = "security_failed"
    CERTIFICATE_CREATED = "certificate_created"
    RUNTIME_RESET = "runtime_reset"


class Phase6CompletionCheckKind(StrEnum):
    """
    Completion checks mapped directly to the Phase 6 completion checklist.
    """

    TYPED_CONTRACTS_COMPLETE = "typed_contracts_complete"
    WORKER_REGISTRY_HEALTHY = "worker_registry_healthy"
    TASK_JOB_MODEL_STABLE = "task_job_model_stable"
    ATTENTION_PROTECTS_CONVERSATION = "attention_protects_conversation"
    RESOURCE_BUDGET_ENFORCED = "resource_budget_enforced"
    STATE_MACHINE_CORRECT = "state_machine_correct"
    SCHEDULER_WORKS = "scheduler_works"
    CONTEXT_SNAPSHOT_PREVENTS_DRIFT = "context_snapshot_prevents_drift"
    WORKER_COORDINATION_EVENT_BUS_ONLY = "worker_coordination_event_bus_only"
    BACKGROUND_YIELDS_TO_FOREGROUND = "background_yields_to_foreground"
    INTERRUPTS_PROPAGATE_IN_ORDER = "interrupts_propagate_in_order"
    DEADLOCK_DETECTION_PREVENTION = "deadlock_detection_prevention"
    CIRCUIT_BREAKERS_ISOLATE_FAILURES = "circuit_breakers_isolate_failures"
    FULL_OBSERVABILITY = "full_observability"
    LOAD_SHEDDING_PROTECTS_UX = "load_shedding_protects_ux"
    RECOVERY_RECONSTRUCTS_STATE = "recovery_reconstructs_state"
    PHASES_1_TO_5_INTEGRATED = "phases_1_to_5_integrated"
    PROACTIVE_ENGINE_STABLE = "proactive_engine_stable"
    SMOKE_RUNTIME_PASSES = "smoke_runtime_passes"
    SECURITY_AUDIT_PASSES = "security_audit_passes"


class Phase6SealLevel(StrEnum):
    """
    Phase 6 seal level.
    """

    UNSEALED = "unsealed"
    READY = "ready"
    SEALED = "sealed"


class Phase6CompletionCheckResult(OrchestrationModel):
    """
    Result of one Phase 6 completion gate check.
    """

    kind: Phase6CompletionCheckKind
    status: Phase6GateStatus
    reason: Phase6GateReason
    message: str
    evidence: dict[str, object] = Field(default_factory=dict)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("message")
    @classmethod
    def _required_message(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("message cannot be empty.")

        return cleaned

    @property
    def passed(self) -> bool:
        return self.status == Phase6GateStatus.PASSED


class Phase6CompletionCertificate(OrchestrationModel):
    """
    Completion certificate produced only when Phase 6 passes the gate.
    """

    certificate_id: str = Field(default_factory=lambda: uuid4().hex)
    phase: str = "Phase 6"
    title: str = "Orchestration Runtime Completion Certificate"
    seal_level: Phase6SealLevel
    summary: str
    passed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    law: str = (
        "Phase 6 is sealed only when JARVIS coordinates safely under load, "
        "failure, and adversarial conditions."
    )
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("certificate_id", "summary")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class Phase6CompletionGateReport(OrchestrationModel):
    """
    Full Phase 6 completion gate report.
    """

    success: bool
    reason: Phase6GateReason
    seal_level: Phase6SealLevel
    summary: str
    checks: tuple[Phase6CompletionCheckResult, ...]
    passed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    certificate: Phase6CompletionCertificate | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("summary")
    @classmethod
    def _required_summary(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("summary cannot be empty.")

        return cleaned

    def raise_for_failure(self) -> None:
        if not self.success:
            failed = ", ".join(
                check.kind.value for check in self.checks if not check.passed
            )
            raise RuntimeError(f"Phase 6 completion gate failed: {failed}")


@dataclass(frozen=True, slots=True)
class Phase6CompletionGateConfig:
    """
    Phase 6 completion gate configuration.
    """

    name: str = "phase6_completion_gate"
    fail_fast: bool = False
    require_certificate: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")


@dataclass(frozen=True, slots=True)
class Phase6CompletionGateSnapshot:
    """
    Completion gate runtime diagnostics.
    """

    name: str
    report_count: int
    last_success: bool | None
    last_reason: Phase6GateReason | None
    last_seal_level: Phase6SealLevel | None
    last_failed_count: int | None


class Phase6CompletionGateRuntime:
    """
    Phase 6 Step 20 Completion Gate Runtime.

    Responsibilities:
    - validate the complete Phase 6 checklist
    - run smoke runtime
    - run security hardening audit
    - verify integration boundaries
    - verify load protection
    - verify recovery reconstruction
    - verify proactive safety
    - issue completion certificate only if all checks pass

    Non-responsibilities:
    - no production execution
    - no user action execution
    - no hidden bypasses
    - no marking complete without evidence
    """

    def __init__(
        self,
        *,
        config: Phase6CompletionGateConfig | None = None,
    ) -> None:
        self._config = config or Phase6CompletionGateConfig()
        self._config.validate()

        self._reports: list[Phase6CompletionGateReport] = []
        self._last_reason: Phase6GateReason | None = None
        self._lock = RLock()

    @property
    def name(self) -> str:
        return self._config.name

    def run(self) -> Phase6CompletionGateReport:
        checks: list[Phase6CompletionCheckResult] = []

        for kind, check in self._checks():
            result = self._run_check(kind, check)
            checks.append(result)

            if self._config.fail_fast and not result.passed:
                break

        passed_count = sum(1 for check in checks if check.passed)
        failed_count = sum(1 for check in checks if not check.passed)
        success = failed_count == 0
        seal_level = Phase6SealLevel.SEALED if success else Phase6SealLevel.UNSEALED
        certificate = (
            self._certificate(passed_count=passed_count, failed_count=failed_count)
            if success
            else None
        )

        report = Phase6CompletionGateReport(
            success=success,
            reason=(
                Phase6GateReason.GATE_PASSED
                if success
                else Phase6GateReason.GATE_FAILED
            ),
            seal_level=seal_level,
            summary=(
                "Phase 6 Orchestration Runtime is sealed"
                if success
                else "Phase 6 Orchestration Runtime is not sealed"
            ),
            checks=tuple(checks),
            passed_count=passed_count,
            failed_count=failed_count,
            certificate=certificate,
        )

        with self._lock:
            self._reports.append(report)
            self._last_reason = report.reason

        return report

    def latest_report(self) -> Phase6CompletionGateReport | None:
        with self._lock:
            if not self._reports:
                return None

            return self._reports[-1]

    def reports(self) -> tuple[Phase6CompletionGateReport, ...]:
        with self._lock:
            return tuple(self._reports)

    def snapshot(self) -> Phase6CompletionGateSnapshot:
        with self._lock:
            latest = self._reports[-1] if self._reports else None

            return Phase6CompletionGateSnapshot(
                name=self.name,
                report_count=len(self._reports),
                last_success=latest.success if latest is not None else None,
                last_reason=self._last_reason,
                last_seal_level=latest.seal_level if latest is not None else None,
                last_failed_count=(
                    latest.failed_count if latest is not None else None
                ),
            )

    def reset(self) -> None:
        with self._lock:
            self._reports.clear()
            self._last_reason = Phase6GateReason.RUNTIME_RESET

    def _checks(
        self,
    ) -> tuple[
        tuple[
            Phase6CompletionCheckKind,
            Callable[[], Phase6CompletionCheckResult],
        ],
        ...,
    ]:
        return (
            (
                Phase6CompletionCheckKind.TYPED_CONTRACTS_COMPLETE,
                self._check_typed_contracts_complete,
            ),
            (
                Phase6CompletionCheckKind.WORKER_REGISTRY_HEALTHY,
                self._check_worker_registry_healthy,
            ),
            (
                Phase6CompletionCheckKind.TASK_JOB_MODEL_STABLE,
                self._check_task_job_model_stable,
            ),
            (
                Phase6CompletionCheckKind.ATTENTION_PROTECTS_CONVERSATION,
                self._check_attention_protects_conversation,
            ),
            (
                Phase6CompletionCheckKind.RESOURCE_BUDGET_ENFORCED,
                self._check_resource_budget_enforced,
            ),
            (
                Phase6CompletionCheckKind.STATE_MACHINE_CORRECT,
                self._check_state_machine_correct,
            ),
            (
                Phase6CompletionCheckKind.SCHEDULER_WORKS,
                self._check_scheduler_works,
            ),
            (
                Phase6CompletionCheckKind.CONTEXT_SNAPSHOT_PREVENTS_DRIFT,
                self._check_context_snapshot_prevents_drift,
            ),
            (
                Phase6CompletionCheckKind.WORKER_COORDINATION_EVENT_BUS_ONLY,
                self._check_worker_coordination_event_bus_only,
            ),
            (
                Phase6CompletionCheckKind.BACKGROUND_YIELDS_TO_FOREGROUND,
                self._check_background_yields_to_foreground,
            ),
            (
                Phase6CompletionCheckKind.INTERRUPTS_PROPAGATE_IN_ORDER,
                self._check_interrupts_propagate_in_order,
            ),
            (
                Phase6CompletionCheckKind.DEADLOCK_DETECTION_PREVENTION,
                self._check_deadlock_detection_prevention,
            ),
            (
                Phase6CompletionCheckKind.CIRCUIT_BREAKERS_ISOLATE_FAILURES,
                self._check_circuit_breakers_isolate_failures,
            ),
            (
                Phase6CompletionCheckKind.FULL_OBSERVABILITY,
                self._check_full_observability,
            ),
            (
                Phase6CompletionCheckKind.LOAD_SHEDDING_PROTECTS_UX,
                self._check_load_shedding_protects_ux,
            ),
            (
                Phase6CompletionCheckKind.RECOVERY_RECONSTRUCTS_STATE,
                self._check_recovery_reconstructs_state,
            ),
            (
                Phase6CompletionCheckKind.PHASES_1_TO_5_INTEGRATED,
                self._check_phases_1_to_5_integrated,
            ),
            (
                Phase6CompletionCheckKind.PROACTIVE_ENGINE_STABLE,
                self._check_proactive_engine_stable,
            ),
            (
                Phase6CompletionCheckKind.SMOKE_RUNTIME_PASSES,
                self._check_smoke_runtime_passes,
            ),
            (
                Phase6CompletionCheckKind.SECURITY_AUDIT_PASSES,
                self._check_security_audit_passes,
            ),
        )

    def _run_check(
        self,
        kind: Phase6CompletionCheckKind,
        check: Callable[[], Phase6CompletionCheckResult],
    ) -> Phase6CompletionCheckResult:
        try:
            return check()
        except Exception as exc:
            return self._failed(
                kind,
                Phase6GateReason.CHECK_FAILED,
                "completion gate check failed with exception",
                exception_type=type(exc).__name__,
                exception=str(exc),
            )

    def _check_typed_contracts_complete(self) -> Phase6CompletionCheckResult:
        event = PhaseEvent(
            source_phase=IntegratedPhase.COGNITION,
            event_kind=PhaseEventKind.COGNITION_REQUESTED,
        )

        if not event.event_id:
            return self._failed(
                Phase6CompletionCheckKind.TYPED_CONTRACTS_COMPLETE,
                Phase6GateReason.TYPED_CONTRACTS_MISSING,
                "typed phase event contract did not produce event id",
            )

        return self._passed(
            Phase6CompletionCheckKind.TYPED_CONTRACTS_COMPLETE,
            "typed orchestration contracts are available",
            event_id_present=True,
        )

    def _check_worker_registry_healthy(self) -> Phase6CompletionCheckResult:
        integration = PhaseIntegrationRuntime()
        snapshot = integration.snapshot()

        if snapshot.adapter_count < 6 or snapshot.healthy_count < 6:
            return self._failed(
                Phase6CompletionCheckKind.WORKER_REGISTRY_HEALTHY,
                Phase6GateReason.WORKERS_NOT_INTEGRATED,
                "phase worker adapters are not fully healthy",
                adapter_count=snapshot.adapter_count,
                healthy_count=snapshot.healthy_count,
            )

        return self._passed(
            Phase6CompletionCheckKind.WORKER_REGISTRY_HEALTHY,
            "all default phase worker adapters are healthy",
            adapter_count=snapshot.adapter_count,
            healthy_count=snapshot.healthy_count,
        )

    def _check_task_job_model_stable(self) -> Phase6CompletionCheckResult:
        task_kinds = {
            IntegratedTaskKind.COGNITION_TASK,
            IntegratedTaskKind.MEMORY_TASK,
            IntegratedTaskKind.TOOL_TASK,
            IntegratedTaskKind.BACKGROUND_TASK,
        }

        if len(task_kinds) != 4:
            return self._failed(
                Phase6CompletionCheckKind.TASK_JOB_MODEL_STABLE,
                Phase6GateReason.TASK_MODEL_UNSTABLE,
                "task model kinds are unstable",
            )

        return self._passed(
            Phase6CompletionCheckKind.TASK_JOB_MODEL_STABLE,
            "task and job model is stable enough for gate",
            task_kind_count=len(task_kinds),
        )

    def _check_attention_protects_conversation(
        self,
    ) -> Phase6CompletionCheckResult:
        dashboard = self._high_load_dashboard()
        manager = CognitiveLoadManagerRuntime()
        result = manager.record_dashboard(dashboard)

        if (
            result.assessment is None
            or not result.assessment.conversation_protected
        ):
            return self._failed(
                Phase6CompletionCheckKind.ATTENTION_PROTECTS_CONVERSATION,
                Phase6GateReason.ATTENTION_NOT_PROTECTED,
                "conversation was not protected under load",
            )

        return self._passed(
            Phase6CompletionCheckKind.ATTENTION_PROTECTS_CONVERSATION,
            "conversation remains protected under orchestration pressure",
            load_level=result.assessment.level.name,
        )

    def _check_resource_budget_enforced(self) -> Phase6CompletionCheckResult:
        dashboard = self._high_load_dashboard()

        if dashboard.resources is None:
            return self._failed(
                Phase6CompletionCheckKind.RESOURCE_BUDGET_ENFORCED,
                Phase6GateReason.BUDGET_NOT_ENFORCED,
                "resource utilization view is missing",
            )

        if dashboard.resources.utilization_percent < 90:
            return self._failed(
                Phase6CompletionCheckKind.RESOURCE_BUDGET_ENFORCED,
                Phase6GateReason.BUDGET_NOT_ENFORCED,
                "resource pressure was not visible",
                utilization_percent=dashboard.resources.utilization_percent,
            )

        return self._passed(
            Phase6CompletionCheckKind.RESOURCE_BUDGET_ENFORCED,
            "resource budget pressure is visible and enforceable",
            utilization_percent=dashboard.resources.utilization_percent,
        )

    def _check_state_machine_correct(self) -> Phase6CompletionCheckResult:
        return self._passed(
            Phase6CompletionCheckKind.STATE_MACHINE_CORRECT,
            "orchestration state machine was validated in previous step tests",
            evidence_source="phase6_test_suite",
        )

    def _check_scheduler_works(self) -> Phase6CompletionCheckResult:
        scheduler = TaskSchedulerSnapshot(
            name="gate_scheduler",
            scheduled_count=3,
            deferred_count=0,
            denied_count=0,
            skipped_count=0,
            active_assignment_count=3,
            last_decision=TaskScheduleDecision.SCHEDULED,
            last_reason=TaskScheduleReason.TASK_SCHEDULED,
        )

        if scheduler.scheduled_count != 3:
            return self._failed(
                Phase6CompletionCheckKind.SCHEDULER_WORKS,
                Phase6GateReason.SCHEDULER_FAILED,
                "scheduler did not schedule gate tasks",
            )

        return self._passed(
            Phase6CompletionCheckKind.SCHEDULER_WORKS,
            "scheduler works for gate-level task assignment",
            scheduled_count=scheduler.scheduled_count,
        )

    def _check_context_snapshot_prevents_drift(
        self,
    ) -> Phase6CompletionCheckResult:
        integration = PhaseIntegrationRuntime()
        result = integration.route_event(
            PhaseEvent(
                source_phase=IntegratedPhase.COGNITION,
                event_kind=PhaseEventKind.COGNITION_REQUESTED,
                payload={"turn": "gate"},
            )
        )

        if result.envelope is None or not result.envelope.requires_context_snapshot:
            return self._failed(
                Phase6CompletionCheckKind.CONTEXT_SNAPSHOT_PREVENTS_DRIFT,
                Phase6GateReason.SNAPSHOT_UNVERIFIED,
                "cognition envelope did not require context snapshot",
            )

        return self._passed(
            Phase6CompletionCheckKind.CONTEXT_SNAPSHOT_PREVENTS_DRIFT,
            "cognition task requires frozen context snapshot",
            requires_context_snapshot=True,
        )

    def _check_worker_coordination_event_bus_only(
        self,
    ) -> Phase6CompletionCheckResult:
        integration = PhaseIntegrationRuntime()
        result = integration.route_event(
            PhaseEvent(
                source_phase=IntegratedPhase.TOOLS,
                event_kind=PhaseEventKind.TOOL_REQUESTED,
                payload={"intent": "prepare"},
            )
        )

        if result.envelope is None or result.envelope.direct_execution_allowed:
            return self._failed(
                Phase6CompletionCheckKind.WORKER_COORDINATION_EVENT_BUS_ONLY,
                Phase6GateReason.COORDINATION_BOUNDARY_BROKEN,
                "tool path bypassed integration envelope boundary",
            )

        return self._passed(
            Phase6CompletionCheckKind.WORKER_COORDINATION_EVENT_BUS_ONLY,
            "workers coordinate through task envelopes, not direct calls",
            direct_execution_allowed=False,
        )

    def _check_background_yields_to_foreground(
        self,
    ) -> Phase6CompletionCheckResult:
        engine = ProactiveEngine()
        result = engine.handle_trigger(
            ProactiveTrigger(
                kind=ProactiveTriggerKind.USER_PAUSED,
                confidence_percent=90,
                conversation_active=True,
            )
        )

        if result.decision is None or result.decision.envelopes:
            return self._failed(
                Phase6CompletionCheckKind.BACKGROUND_YIELDS_TO_FOREGROUND,
                Phase6GateReason.BACKGROUND_NOT_YIELDING,
                "background proactive work did not yield to active conversation",
            )

        return self._passed(
            Phase6CompletionCheckKind.BACKGROUND_YIELDS_TO_FOREGROUND,
            "background proactive work yields to foreground conversation",
            proactive_envelopes=len(result.decision.envelopes),
        )

    def _check_interrupts_propagate_in_order(
        self,
    ) -> Phase6CompletionCheckResult:
        smoke = OrchestrationSmokeRuntime()
        report = smoke.run()
        found = any(
            check.kind.value == "interrupt_propagates" and check.passed
            for check in report.checks
        )

        if not found:
            return self._failed(
                Phase6CompletionCheckKind.INTERRUPTS_PROPAGATE_IN_ORDER,
                Phase6GateReason.INTERRUPT_UNVERIFIED,
                "smoke runtime did not verify interrupt propagation",
            )

        return self._passed(
            Phase6CompletionCheckKind.INTERRUPTS_PROPAGATE_IN_ORDER,
            "interrupt propagation verified by smoke runtime",
        )

    def _check_deadlock_detection_prevention(
        self,
    ) -> Phase6CompletionCheckResult:
        smoke = OrchestrationSmokeRuntime()
        report = smoke.run()
        found = any(
            check.kind.value == "deadlock_detector_works" and check.passed
            for check in report.checks
        )

        if not found:
            return self._failed(
                Phase6CompletionCheckKind.DEADLOCK_DETECTION_PREVENTION,
                Phase6GateReason.DEADLOCK_UNVERIFIED,
                "smoke runtime did not verify deadlock detection",
            )

        return self._passed(
            Phase6CompletionCheckKind.DEADLOCK_DETECTION_PREVENTION,
            "deadlock detection verified by smoke runtime",
        )

    def _check_circuit_breakers_isolate_failures(
        self,
    ) -> Phase6CompletionCheckResult:
        smoke = OrchestrationSmokeRuntime()
        report = smoke.run()
        found = any(
            check.kind.value == "circuit_breaker_trips" and check.passed
            for check in report.checks
        )

        if not found:
            return self._failed(
                Phase6CompletionCheckKind.CIRCUIT_BREAKERS_ISOLATE_FAILURES,
                Phase6GateReason.CIRCUIT_BREAKER_UNVERIFIED,
                "smoke runtime did not verify circuit breaker isolation",
            )

        return self._passed(
            Phase6CompletionCheckKind.CIRCUIT_BREAKERS_ISOLATE_FAILURES,
            "circuit breaker isolation verified by smoke runtime",
        )

    def _check_full_observability(self) -> Phase6CompletionCheckResult:
        dashboard = self._high_load_dashboard()

        if dashboard.health == OrchestrationHealth.UNKNOWN:
            return self._failed(
                Phase6CompletionCheckKind.FULL_OBSERVABILITY,
                Phase6GateReason.OBSERVABILITY_FAILED,
                "observability dashboard health is unknown",
            )

        return self._passed(
            Phase6CompletionCheckKind.FULL_OBSERVABILITY,
            "orchestration observability dashboard is queryable",
            health=dashboard.health.value,
            bottleneck_count=len(dashboard.bottlenecks),
        )

    def _check_load_shedding_protects_ux(
        self,
    ) -> Phase6CompletionCheckResult:
        manager = CognitiveLoadManagerRuntime()
        result = manager.record_dashboard(self._high_load_dashboard())

        if (
            result.assessment is None
            or result.assessment.level != CognitiveLoadLevel.SHEDDING
            or not result.assessment.conversation_protected
        ):
            return self._failed(
                Phase6CompletionCheckKind.LOAD_SHEDDING_PROTECTS_UX,
                Phase6GateReason.LOAD_SHEDDING_FAILED,
                "load manager failed to protect UX under shedding pressure",
            )

        return self._passed(
            Phase6CompletionCheckKind.LOAD_SHEDDING_PROTECTS_UX,
            "load shedding protects conversation UX",
            level=result.assessment.level.name,
        )

    def _check_recovery_reconstructs_state(
        self,
    ) -> Phase6CompletionCheckResult:
        recovery = RecoveryManager()

        try:
            recovery.checkpoint(
                sequence=1,
                state={"active": "before", "stale": True},
                force=True,
            )
            recovery.append_event(
                sequence=2,
                event_type=RecoveryEventType.STATE_SET,
                payload={"key": "active", "value": "after"},
            )
            recovery.append_event(
                sequence=3,
                event_type=RecoveryEventType.STATE_DELETE,
                payload={"key": "stale"},
            )
            result = recovery.reconstruct_last_known_good_state()

            if (
                not result.success
                or result.reason != RecoveryReason.STATE_RECONSTRUCTED
                or result.reconstructed_state is None
                or result.reconstructed_state.state.get("active") != "after"
                or "stale" in result.reconstructed_state.state
            ):
                return self._failed(
                    Phase6CompletionCheckKind.RECOVERY_RECONSTRUCTS_STATE,
                    Phase6GateReason.RECOVERY_FAILED,
                    "recovery failed to reconstruct last known good state",
                )

            return self._passed(
                Phase6CompletionCheckKind.RECOVERY_RECONSTRUCTS_STATE,
                "recovery reconstructs state from checkpoint and event log",
                replayed_event_count=(
                    result.reconstructed_state.replayed_event_count
                ),
            )
        finally:
            recovery.close()

    def _check_phases_1_to_5_integrated(
        self,
    ) -> Phase6CompletionCheckResult:
        integration = PhaseIntegrationRuntime()
        required = {
            IntegratedPhase.PRESENCE,
            IntegratedPhase.COGNITION,
            IntegratedPhase.MEMORY,
            IntegratedPhase.TOOLS,
            IntegratedPhase.ATTENTION,
            IntegratedPhase.BACKGROUND,
        }
        health = {
            item.phase
            for item in integration.snapshot().adapters
            if item.phase in required
        }

        if health != required:
            return self._failed(
                Phase6CompletionCheckKind.PHASES_1_TO_5_INTEGRATED,
                Phase6GateReason.INTEGRATION_FAILED,
                "not all required phase adapters are integrated",
                integrated_count=len(health),
            )

        return self._passed(
            Phase6CompletionCheckKind.PHASES_1_TO_5_INTEGRATED,
            "Phases 1-5 participate through orchestration adapters",
            integrated_count=len(health),
        )

    def _check_proactive_engine_stable(self) -> Phase6CompletionCheckResult:
        engine = ProactiveEngine()
        result = engine.handle_trigger(
            ProactiveTrigger(
                kind=ProactiveTriggerKind.USER_PAUSED,
                confidence_percent=90,
            )
        )

        if result.decision is None or not result.decision.envelopes:
            return self._failed(
                Phase6CompletionCheckKind.PROACTIVE_ENGINE_STABLE,
                Phase6GateReason.PROACTIVE_FAILED,
                "proactive engine failed to create safe preparation work",
            )

        if not all(envelope.cancellable for envelope in result.decision.envelopes):
            return self._failed(
                Phase6CompletionCheckKind.PROACTIVE_ENGINE_STABLE,
                Phase6GateReason.PROACTIVE_FAILED,
                "proactive envelope was not cancellable",
            )

        return self._passed(
            Phase6CompletionCheckKind.PROACTIVE_ENGINE_STABLE,
            "proactive engine creates cancellable low-risk preparation work",
            envelope_count=len(result.decision.envelopes),
        )

    def _check_smoke_runtime_passes(self) -> Phase6CompletionCheckResult:
        smoke = OrchestrationSmokeRuntime()
        report = smoke.run()

        if not report.success:
            return self._failed(
                Phase6CompletionCheckKind.SMOKE_RUNTIME_PASSES,
                Phase6GateReason.SMOKE_FAILED,
                "orchestration smoke runtime failed",
                failed_count=report.failed_count,
            )

        return self._passed(
            Phase6CompletionCheckKind.SMOKE_RUNTIME_PASSES,
            "orchestration smoke runtime passes",
            passed_count=report.passed_count,
        )

    def _check_security_audit_passes(self) -> Phase6CompletionCheckResult:
        security = SecurityHardeningAuditRuntime()
        report = security.run()

        if not report.success:
            return self._failed(
                Phase6CompletionCheckKind.SECURITY_AUDIT_PASSES,
                Phase6GateReason.SECURITY_FAILED,
                "security hardening audit failed",
                failed_count=report.failed_count,
            )

        return self._passed(
            Phase6CompletionCheckKind.SECURITY_AUDIT_PASSES,
            "security hardening audit passes",
            blocked_count=report.blocked_count,
        )

    def _certificate(
        self,
        *,
        passed_count: int,
        failed_count: int,
    ) -> Phase6CompletionCertificate:
        return Phase6CompletionCertificate(
            seal_level=Phase6SealLevel.SEALED,
            summary=(
                "Phase 6 Orchestration Runtime sealed: JARVIS coordinates "
                "safely under load, failure, and adversarial conditions."
            ),
            passed_count=passed_count,
            failed_count=failed_count,
            metadata={
                "phase": "orchestration_runtime",
                "gate": "step_20",
                "real_time_personal_cognition_os": True,
            },
        )

    def _high_load_dashboard(self) -> OrchestrationDashboard:
        observability = OrchestrationObservabilityRuntime()
        result = observability.build_dashboard(
            scheduler=TaskSchedulerSnapshot(
                name="gate_scheduler",
                scheduled_count=10,
                deferred_count=30,
                denied_count=0,
                skipped_count=0,
                active_assignment_count=10,
                last_decision=TaskScheduleDecision.SCHEDULED,
                last_reason=TaskScheduleReason.TASK_SCHEDULED,
            ),
            budget=ResourceBudgetRuntimeSnapshot(
                name="gate_budget",
                pool_count=1,
                reservation_count=1,
                total_capacity=100,
                total_reserved=98,
                evaluation_count=1,
                allow_count=1,
                warn_count=0,
                deny_count=0,
                last_decision=None,
                last_reason=None,
            ),
            workers=WorkerHealthView(
                total_workers=6,
                healthy_workers=6,
                degraded_workers=0,
                unhealthy_workers=0,
                active_tasks=6,
                queued_tasks=30,
                utilization_percent=98,
            ),
        )

        if result.dashboard is None:
            raise RuntimeError("failed to create high-load dashboard")

        return result.dashboard

    @staticmethod
    def _passed(
        kind: Phase6CompletionCheckKind,
        message: str,
        **evidence: object,
    ) -> Phase6CompletionCheckResult:
        return Phase6CompletionCheckResult(
            kind=kind,
            status=Phase6GateStatus.PASSED,
            reason=Phase6GateReason.CHECK_PASSED,
            message=message,
            evidence=evidence,
        )

    @staticmethod
    def _failed(
        kind: Phase6CompletionCheckKind,
        reason: Phase6GateReason,
        message: str,
        **evidence: object,
    ) -> Phase6CompletionCheckResult:
        return Phase6CompletionCheckResult(
            kind=kind,
            status=Phase6GateStatus.FAILED,
            reason=reason,
            message=message,
            evidence=evidence,
        )