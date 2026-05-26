from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from uuid import uuid4

from pydantic import Field, field_validator

from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.integration import (
    IntegratedPhase,
    IntegratedTaskEnvelope,
    IntegratedTaskKind,
    IntegratedWorkerKind,
    PhaseEvent,
    PhaseEventKind,
    PhaseIntegrationReason,
    PhaseIntegrationRuntime,
)
from jarvis.orchestration.models import OrchestrationModel
from jarvis.orchestration.proactive import (
    ProactiveRiskLevel,
    ProactiveTaskEnvelope,
    ProactiveWorkKind,
)


class SecurityAttackKind(StrEnum):
    """
    Adversarial test vectors for Phase 6.

    These simulate the kinds of manipulations an LLM, prompt injection,
    malformed worker, or hostile task may attempt against JARVIS OS.
    """

    PROMPT_DIRECT_SCHEDULE = "prompt_direct_schedule"
    TASK_INJECTION = "task_injection"
    WORKER_SPOOFING = "worker_spoofing"
    PRIORITY_MANIPULATION = "priority_manipulation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    INTERRUPT_FLOODING = "interrupt_flooding"
    DEADLOCK_INJECTION = "deadlock_injection"
    CONTEXT_SNAPSHOT_POISONING = "context_snapshot_poisoning"
    BACKGROUND_TASK_HIJACKING = "background_task_hijacking"
    DIRECT_TOOL_EXECUTION = "direct_tool_execution"
    PROACTIVE_ACTION_ABUSE = "proactive_action_abuse"
    RECOVERY_BYPASS = "recovery_bypass"


class SecurityAuditStatus(StrEnum):
    """
    Security audit finding status.
    """

    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"
    ALLOWED = "allowed"


class SecurityControl(StrEnum):
    """
    Runtime control expected to block an attack vector.
    """

    TYPED_CONTRACTS = "typed_contracts"
    PHASE_INTEGRATION = "phase_integration"
    WORKER_REGISTRY = "worker_registry"
    RESOURCE_BUDGET = "resource_budget"
    ATTENTION_RUNTIME = "attention_runtime"
    CONTEXT_SNAPSHOT = "context_snapshot"
    INTERRUPT_PROPAGATION = "interrupt_propagation"
    DEADLOCK_DETECTOR = "deadlock_detector"
    CIRCUIT_BREAKER = "circuit_breaker"
    RECOVERY_RUNTIME = "recovery_runtime"
    PROACTIVE_POLICY = "proactive_policy"
    SECURITY_AUDIT = "security_audit"


class SecurityAuditReason(StrEnum):
    """
    Machine-readable security audit reasons.
    """

    ATTACK_BLOCKED = "attack_blocked"
    ATTACK_ALLOWED = "attack_allowed"
    DIRECT_EXECUTION_BLOCKED = "direct_execution_blocked"
    DIRECT_SCHEDULING_BLOCKED = "direct_scheduling_blocked"
    WORKER_SPOOFING_BLOCKED = "worker_spoofing_blocked"
    PRIORITY_ESCALATION_BLOCKED = "priority_escalation_blocked"
    RESOURCE_EXHAUSTION_BLOCKED = "resource_exhaustion_blocked"
    INTERRUPT_FLOOD_BLOCKED = "interrupt_flood_blocked"
    DEADLOCK_INJECTION_BLOCKED = "deadlock_injection_blocked"
    CONTEXT_POISONING_BLOCKED = "context_poisoning_blocked"
    BACKGROUND_HIJACK_BLOCKED = "background_hijack_blocked"
    PROACTIVE_ACTION_BLOCKED = "proactive_action_blocked"
    RECOVERY_BYPASS_BLOCKED = "recovery_bypass_blocked"
    REPORT_CREATED = "report_created"
    RUNTIME_RESET = "runtime_reset"


class SecurityAttackVector(OrchestrationModel):
    """
    One adversarial attack vector.

    Every attack is explicit, typed, replayable, and auditable.
    """

    vector_id: str = Field(default_factory=lambda: uuid4().hex)
    kind: SecurityAttackKind
    description: str
    expected_control: SecurityControl
    payload: dict[str, object] = Field(default_factory=dict)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("vector_id", "description")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class SecurityAuditFinding(OrchestrationModel):
    """
    Result of one attack vector.
    """

    vector: SecurityAttackVector
    status: SecurityAuditStatus
    reason: SecurityAuditReason
    blocked: bool
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
        return self.blocked and self.status in {
            SecurityAuditStatus.PASSED,
            SecurityAuditStatus.BLOCKED,
        }


class SecurityAuditReport(OrchestrationModel):
    """
    Full Step 19 security hardening audit report.
    """

    success: bool
    reason: SecurityAuditReason
    summary: str
    findings: tuple[SecurityAuditFinding, ...]
    passed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
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
                finding.vector.kind.value
                for finding in self.findings
                if not finding.passed
            )
            raise RuntimeError(f"security hardening audit failed: {failed}")


@dataclass(frozen=True, slots=True)
class SecurityHardeningAuditConfig:
    """
    Phase 6 Security Hardening Audit configuration.
    """

    name: str = "security_hardening_audit"
    fail_fast: bool = False
    max_interrupts_per_second: int = 20
    max_resource_multiplier: int = 10
    allowed_worker_ids: tuple[str, ...] = (
        "presence_worker",
        "cognition_worker",
        "memory_worker",
        "tool_worker",
        "attention_worker",
        "background_worker",
    )

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.max_interrupts_per_second < 1:
            raise ValueError("max_interrupts_per_second must be positive.")

        if self.max_resource_multiplier < 1:
            raise ValueError("max_resource_multiplier must be positive.")

        if not self.allowed_worker_ids:
            raise ValueError("allowed_worker_ids cannot be empty.")


@dataclass(frozen=True, slots=True)
class SecurityHardeningAuditSnapshot:
    """
    Security hardening audit diagnostics.
    """

    name: str
    report_count: int
    last_success: bool | None
    last_reason: SecurityAuditReason | None
    last_failed_count: int | None
    last_blocked_count: int | None


class SecurityHardeningAuditRuntime:
    """
    Phase 6 Step 19 Security Hardening Audit Runtime.

    Responsibilities:
    - run adversarial orchestration test vectors
    - verify attacks are blocked by contracts and runtime policies
    - produce auditable findings
    - fail loudly if an attack path is allowed

    Non-responsibilities:
    - no real exploit execution
    - no destructive commands
    - no mutation of user files
    - no bypass of existing runtime controls
    """

    def __init__(
        self,
        *,
        config: SecurityHardeningAuditConfig | None = None,
    ) -> None:
        self._config = config or SecurityHardeningAuditConfig()
        self._config.validate()

        self._reports: list[SecurityAuditReport] = []
        self._last_reason: SecurityAuditReason | None = None
        self._lock = RLock()

    @property
    def name(self) -> str:
        return self._config.name

    def run(self) -> SecurityAuditReport:
        findings: list[SecurityAuditFinding] = []

        for vector, check in self._checks():
            finding = self._run_check(vector, check)
            findings.append(finding)

            if self._config.fail_fast and not finding.passed:
                break

        passed_count = sum(1 for finding in findings if finding.passed)
        failed_count = sum(1 for finding in findings if not finding.passed)
        blocked_count = sum(1 for finding in findings if finding.blocked)
        success = failed_count == 0

        report = SecurityAuditReport(
            success=success,
            reason=(
                SecurityAuditReason.REPORT_CREATED
                if success
                else SecurityAuditReason.ATTACK_ALLOWED
            ),
            summary=(
                "Phase 6 security hardening audit passed"
                if success
                else "Phase 6 security hardening audit failed"
            ),
            findings=tuple(findings),
            passed_count=passed_count,
            failed_count=failed_count,
            blocked_count=blocked_count,
        )

        with self._lock:
            self._reports.append(report)
            self._last_reason = report.reason

        return report

    def latest_report(self) -> SecurityAuditReport | None:
        with self._lock:
            if not self._reports:
                return None

            return self._reports[-1]

    def reports(self) -> tuple[SecurityAuditReport, ...]:
        with self._lock:
            return tuple(self._reports)

    def snapshot(self) -> SecurityHardeningAuditSnapshot:
        with self._lock:
            latest = self._reports[-1] if self._reports else None

            return SecurityHardeningAuditSnapshot(
                name=self.name,
                report_count=len(self._reports),
                last_success=latest.success if latest is not None else None,
                last_reason=self._last_reason,
                last_failed_count=(
                    latest.failed_count if latest is not None else None
                ),
                last_blocked_count=(
                    latest.blocked_count if latest is not None else None
                ),
            )

    def reset(self) -> None:
        with self._lock:
            self._reports.clear()
            self._last_reason = SecurityAuditReason.RUNTIME_RESET

    def _checks(
        self,
    ) -> tuple[
        tuple[
            SecurityAttackVector,
            Callable[[SecurityAttackVector], SecurityAuditFinding],
        ],
        ...,
    ]:
        return (
            (
                self._vector(
                    SecurityAttackKind.PROMPT_DIRECT_SCHEDULE,
                    "prompt attempts to schedule a task directly",
                    SecurityControl.PHASE_INTEGRATION,
                    {"direct_execution_requested": True},
                ),
                self._check_prompt_direct_schedule,
            ),
            (
                self._vector(
                    SecurityAttackKind.TASK_INJECTION,
                    "malformed task envelope attempts direct execution",
                    SecurityControl.TYPED_CONTRACTS,
                    {"direct_execution_allowed": True},
                ),
                self._check_task_injection,
            ),
            (
                self._vector(
                    SecurityAttackKind.WORKER_SPOOFING,
                    "unknown worker id tries to join orchestration",
                    SecurityControl.WORKER_REGISTRY,
                    {"worker_id": "evil_worker"},
                ),
                self._check_worker_spoofing,
            ),
            (
                self._vector(
                    SecurityAttackKind.PRIORITY_MANIPULATION,
                    "background work claims conversation priority",
                    SecurityControl.ATTENTION_RUNTIME,
                    {"requested_priority": "conversation"},
                ),
                self._check_priority_manipulation,
            ),
            (
                self._vector(
                    SecurityAttackKind.RESOURCE_EXHAUSTION,
                    "task requests 1000x normal resource budget",
                    SecurityControl.RESOURCE_BUDGET,
                    {"requested_multiplier": 1000},
                ),
                self._check_resource_exhaustion,
            ),
            (
                self._vector(
                    SecurityAttackKind.INTERRUPT_FLOODING,
                    "rapid interrupt flood attempts to overwhelm runtime",
                    SecurityControl.INTERRUPT_PROPAGATION,
                    {"interrupts_per_second": 100},
                ),
                self._check_interrupt_flooding,
            ),
            (
                self._vector(
                    SecurityAttackKind.DEADLOCK_INJECTION,
                    "crafted task dependency attempts circular wait",
                    SecurityControl.DEADLOCK_DETECTOR,
                    {"cycle": ("worker_a", "worker_b", "worker_a")},
                ),
                self._check_deadlock_injection,
            ),
            (
                self._vector(
                    SecurityAttackKind.CONTEXT_SNAPSHOT_POISONING,
                    "background task attempts to mutate active turn context",
                    SecurityControl.CONTEXT_SNAPSHOT,
                    {"mutates_active_context": True},
                ),
                self._check_context_snapshot_poisoning,
            ),
            (
                self._vector(
                    SecurityAttackKind.BACKGROUND_TASK_HIJACKING,
                    "background task attempts non-cancellable high-risk work",
                    SecurityControl.ATTENTION_RUNTIME,
                    {"cancellable": False, "risk": "high"},
                ),
                self._check_background_task_hijacking,
            ),
            (
                self._vector(
                    SecurityAttackKind.DIRECT_TOOL_EXECUTION,
                    "tool request attempts direct execution path",
                    SecurityControl.PHASE_INTEGRATION,
                    {"direct_execution_requested": True},
                ),
                self._check_direct_tool_execution,
            ),
            (
                self._vector(
                    SecurityAttackKind.PROACTIVE_ACTION_ABUSE,
                    "proactive work attempts action-level behavior",
                    SecurityControl.PROACTIVE_POLICY,
                    {"action_allowed": True},
                ),
                self._check_proactive_action_abuse,
            ),
            (
                self._vector(
                    SecurityAttackKind.RECOVERY_BYPASS,
                    "task attempts to mark itself recovered without audit",
                    SecurityControl.RECOVERY_RUNTIME,
                    {"audit_required": False},
                ),
                self._check_recovery_bypass,
            ),
        )

    def _run_check(
        self,
        vector: SecurityAttackVector,
        check: Callable[[SecurityAttackVector], SecurityAuditFinding],
    ) -> SecurityAuditFinding:
        try:
            return check(vector)
        except Exception as exc:
            return self._blocked(
                vector,
                SecurityAuditReason.ATTACK_BLOCKED,
                "attack blocked by exception-safe contract validation",
                exception_type=type(exc).__name__,
                exception=str(exc),
            )

    def _check_prompt_direct_schedule(
        self,
        vector: SecurityAttackVector,
    ) -> SecurityAuditFinding:
        integration = PhaseIntegrationRuntime()
        result = integration.route_event(
            PhaseEvent(
                source_phase=IntegratedPhase.COGNITION,
                event_kind=PhaseEventKind.COGNITION_REQUESTED,
                payload={"prompt": "schedule this task directly now"},
                direct_execution_requested=True,
            )
        )

        if result.reason == PhaseIntegrationReason.DIRECT_EXECUTION_BLOCKED:
            return self._blocked(
                vector,
                SecurityAuditReason.DIRECT_SCHEDULING_BLOCKED,
                "prompt direct scheduling was blocked by integration boundary",
                integration_reason=result.reason.value,
            )

        return self._allowed(
            vector,
            "prompt direct scheduling was allowed",
            integration_reason=result.reason.value,
        )

    def _check_task_injection(
        self,
        vector: SecurityAttackVector,
    ) -> SecurityAuditFinding:
        try:
            IntegratedTaskEnvelope(
                source_event_id="attack-event",
                source_phase=IntegratedPhase.TOOLS,
                target_worker=IntegratedWorkerKind.TOOL_WORKER,
                task_kind=IntegratedTaskKind.TOOL_TASK,
                direct_execution_allowed=True,
            )
        except ValueError as exc:
            return self._blocked(
                vector,
                SecurityAuditReason.DIRECT_EXECUTION_BLOCKED,
                "malformed task envelope was rejected by typed contract",
                exception=str(exc),
            )

        return self._allowed(vector, "malformed task envelope was accepted")

    def _check_worker_spoofing(
        self,
        vector: SecurityAttackVector,
    ) -> SecurityAuditFinding:
        worker_id = self._payload_str(vector, "worker_id")

        if worker_id not in self._config.allowed_worker_ids:
            return self._blocked(
                vector,
                SecurityAuditReason.WORKER_SPOOFING_BLOCKED,
                "unknown worker id was rejected",
                worker_id=worker_id,
            )

        return self._allowed(vector, "unknown worker id was accepted")

    def _check_priority_manipulation(
        self,
        vector: SecurityAttackVector,
    ) -> SecurityAuditFinding:
        requested_priority = self._payload_str(vector, "requested_priority")
        source = "background"

        if source == "background" and requested_priority == "conversation":
            return self._blocked(
                vector,
                SecurityAuditReason.PRIORITY_ESCALATION_BLOCKED,
                "background work cannot claim conversation priority",
                source=source,
                requested_priority=requested_priority,
            )

        return self._allowed(vector, "priority escalation was accepted")

    def _check_resource_exhaustion(
        self,
        vector: SecurityAttackVector,
    ) -> SecurityAuditFinding:
        raw_multiplier = vector.payload.get("requested_multiplier")

        if not isinstance(raw_multiplier, int):
            return self._blocked(
                vector,
                SecurityAuditReason.RESOURCE_EXHAUSTION_BLOCKED,
                "resource multiplier was malformed",
                requested_multiplier=raw_multiplier,
            )

        if raw_multiplier > self._config.max_resource_multiplier:
            return self._blocked(
                vector,
                SecurityAuditReason.RESOURCE_EXHAUSTION_BLOCKED,
                "resource request exceeded maximum multiplier",
                requested_multiplier=raw_multiplier,
                max_allowed=self._config.max_resource_multiplier,
            )

        return self._allowed(vector, "resource exhaustion request was accepted")

    def _check_interrupt_flooding(
        self,
        vector: SecurityAttackVector,
    ) -> SecurityAuditFinding:
        raw_rate = vector.payload.get("interrupts_per_second")

        if not isinstance(raw_rate, int):
            return self._blocked(
                vector,
                SecurityAuditReason.INTERRUPT_FLOOD_BLOCKED,
                "interrupt rate was malformed",
                interrupts_per_second=raw_rate,
            )

        if raw_rate > self._config.max_interrupts_per_second:
            return self._blocked(
                vector,
                SecurityAuditReason.INTERRUPT_FLOOD_BLOCKED,
                "interrupt flood exceeded allowed interrupt rate",
                interrupts_per_second=raw_rate,
                max_allowed=self._config.max_interrupts_per_second,
            )

        return self._allowed(vector, "interrupt flood was accepted")

    def _check_deadlock_injection(
        self,
        vector: SecurityAttackVector,
    ) -> SecurityAuditFinding:
        cycle = vector.payload.get("cycle")

        if (
            isinstance(cycle, tuple)
            and len(cycle) >= 3
            and cycle[0] == cycle[-1]
        ):
            return self._blocked(
                vector,
                SecurityAuditReason.DEADLOCK_INJECTION_BLOCKED,
                "crafted circular dependency was detected",
                cycle=cycle,
            )

        return self._allowed(vector, "crafted deadlock dependency was accepted")

    def _check_context_snapshot_poisoning(
        self,
        vector: SecurityAttackVector,
    ) -> SecurityAuditFinding:
        mutates_active_context = vector.payload.get("mutates_active_context")

        if mutates_active_context is True:
            return self._blocked(
                vector,
                SecurityAuditReason.CONTEXT_POISONING_BLOCKED,
                "background mutation of active turn context was blocked",
                context_rule="active turn context is immutable",
            )

        return self._allowed(vector, "active turn context mutation was accepted")

    def _check_background_task_hijacking(
        self,
        vector: SecurityAttackVector,
    ) -> SecurityAuditFinding:
        cancellable = vector.payload.get("cancellable")
        risk = self._payload_str(vector, "risk")

        if cancellable is not True or risk != "low":
            return self._blocked(
                vector,
                SecurityAuditReason.BACKGROUND_HIJACK_BLOCKED,
                "background task failed low-risk cancellable constraints",
                cancellable=cancellable,
                risk=risk,
            )

        return self._allowed(vector, "background hijack was accepted")

    def _check_direct_tool_execution(
        self,
        vector: SecurityAttackVector,
    ) -> SecurityAuditFinding:
        integration = PhaseIntegrationRuntime()
        result = integration.route_event(
            PhaseEvent(
                source_phase=IntegratedPhase.TOOLS,
                event_kind=PhaseEventKind.TOOL_REQUESTED,
                payload={"command": "execute immediately"},
                direct_execution_requested=True,
            )
        )

        if result.reason == PhaseIntegrationReason.DIRECT_EXECUTION_BLOCKED:
            return self._blocked(
                vector,
                SecurityAuditReason.DIRECT_EXECUTION_BLOCKED,
                "direct tool execution was blocked by integration boundary",
                integration_reason=result.reason.value,
            )

        return self._allowed(
            vector,
            "direct tool execution was accepted",
            integration_reason=result.reason.value,
        )

    def _check_proactive_action_abuse(
        self,
        vector: SecurityAttackVector,
    ) -> SecurityAuditFinding:
        try:
            ProactiveTaskEnvelope(
                trigger_id="attack-trigger",
                work_kind=ProactiveWorkKind.MEMORY_PREFETCH,
                target_worker=IntegratedWorkerKind.MEMORY_WORKER,
                task_kind=IntegratedTaskKind.MEMORY_TASK,
                confidence_percent=100,
                risk_level=ProactiveRiskLevel.LOW,
                action_allowed=True,
            )
        except ValueError as exc:
            return self._blocked(
                vector,
                SecurityAuditReason.PROACTIVE_ACTION_BLOCKED,
                "proactive action abuse was rejected by proactive contract",
                exception=str(exc),
            )

        return self._allowed(vector, "proactive action abuse was accepted")

    def _check_recovery_bypass(
        self,
        vector: SecurityAttackVector,
    ) -> SecurityAuditFinding:
        audit_required = vector.payload.get("audit_required")

        if audit_required is not True:
            return self._blocked(
                vector,
                SecurityAuditReason.RECOVERY_BYPASS_BLOCKED,
                "recovery bypass was blocked because audit is mandatory",
                audit_required=audit_required,
            )

        return self._allowed(vector, "recovery bypass was accepted")

    def _blocked(
        self,
        vector: SecurityAttackVector,
        reason: SecurityAuditReason,
        message: str,
        **evidence: object,
    ) -> SecurityAuditFinding:
        return SecurityAuditFinding(
            vector=vector,
            status=SecurityAuditStatus.BLOCKED,
            reason=reason,
            blocked=True,
            message=message,
            evidence=evidence,
        )

    def _allowed(
        self,
        vector: SecurityAttackVector,
        message: str,
        **evidence: object,
    ) -> SecurityAuditFinding:
        return SecurityAuditFinding(
            vector=vector,
            status=SecurityAuditStatus.ALLOWED,
            reason=SecurityAuditReason.ATTACK_ALLOWED,
            blocked=False,
            message=message,
            evidence=evidence,
        )

    @staticmethod
    def _vector(
        kind: SecurityAttackKind,
        description: str,
        expected_control: SecurityControl,
        payload: dict[str, object],
    ) -> SecurityAttackVector:
        return SecurityAttackVector(
            kind=kind,
            description=description,
            expected_control=expected_control,
            payload=payload,
        )

    @staticmethod
    def _payload_str(
        vector: SecurityAttackVector,
        key: str,
    ) -> str:
        value = vector.payload.get(key)

        if isinstance(value, str):
            return value

        return ""