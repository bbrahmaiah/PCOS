from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock

from pydantic import Field, field_validator

from jarvis.memory.cognition_integration import MemoryCognitionBridge
from jarvis.memory.context import (
    MemoryContext,
    MemoryContextBuilder,
    MemoryContextBuildRequest,
    MemoryContextItem,
    MemoryContextItemKind,
)
from jarvis.memory.diagnostics import (
    MemoryDiagnosticsCollector,
    MemoryDiagnosticStatus,
)
from jarvis.memory.gateway import (
    GovernedMemoryGateway,
    MemoryGatewayConfig,
    MemoryGatewayRetrievalResult,
)
from jarvis.memory.lifecycle import (
    MemoryLifecycleDecisionKind,
    MemoryLifecyclePolicy,
)
from jarvis.memory.models import (
    MemoryImportance,
    MemoryKind,
    MemoryModel,
    MemoryPolicyClassification,
    MemoryQuery,
    MemoryRecord,
    MemoryRetention,
    MemoryScope,
    MemorySensitivity,
    MemorySource,
    MemoryWriteRequest,
)
from jarvis.memory.privacy_policy import (
    MemoryPrivacyDecisionKind,
    MemoryPrivacyPolicy,
)
from jarvis.memory.store import InMemoryMemoryStore
from jarvis.runtime.observability.structured_logger import get_logger


class MemorySafetyAuditStatus(StrEnum):
    """
    Status for Phase 4 memory safety audit.
    """

    PASSED = "passed"
    FAILED = "failed"


class MemorySafetyRiskLevel(StrEnum):
    """
    Risk level for a memory safety audit check.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MemorySafetyAuditCheck(MemoryModel):
    """
    One memory safety audit check.
    """

    name: str
    passed: bool
    risk_level: MemorySafetyRiskLevel
    detail: str
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("name", "detail")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class MemorySafetyAuditResult(MemoryModel):
    """
    Full memory safety audit result.
    """

    status: MemorySafetyAuditStatus
    checks: tuple[MemorySafetyAuditCheck, ...]
    metadata: dict[str, object] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == MemorySafetyAuditStatus.PASSED

    @property
    def check_count(self) -> int:
        return len(self.checks)

    @property
    def passed_count(self) -> int:
        return sum(1 for check in self.checks if check.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for check in self.checks if not check.passed)


@dataclass(frozen=True, slots=True)
class MemorySafetyAuditorConfig:
    """
    Configuration for MemorySafetyAuditor.
    """

    name: str = "memory_safety_auditor"
    include_restricted_opt_in_check: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")


@dataclass(frozen=True, slots=True)
class MemorySafetyAuditorSnapshot:
    """
    Observable diagnostics for the safety auditor.
    """

    name: str
    audit_count: int
    last_status: MemorySafetyAuditStatus | None
    last_passed_count: int
    last_failed_count: int
    last_error: str | None


class MemorySafetyAuditor:
    """
    Formal safety auditor for Phase 4 memory runtime.

    Responsibilities:
    - prove sensitive writes are blocked by default
    - prove sensitive retrieval is filtered by default
    - prove restricted context is not injected by default
    - prove restricted context requires explicit opt-in
    - prove delete / clear policies are enforced
    - prove privacy policy catches high-risk memory content
    - prove lifecycle protects pinned and critical memory
    - prove diagnostics catches unsafe context
    - prove cognition-memory path uses gateway bridge

    Non-responsibilities:
    - no LLM calls
    - no production vector DB calls
    - no mutation of global runtime state
    - no direct cognition access to memory store
    """

    def __init__(
        self,
        *,
        config: MemorySafetyAuditorConfig | None = None,
    ) -> None:
        self._config = config or MemorySafetyAuditorConfig()
        self._config.validate()

        self._lock = RLock()
        self._logger = get_logger("memory.safety_auditor")

        self._audit_count = 0
        self._last_status: MemorySafetyAuditStatus | None = None
        self._last_passed_count = 0
        self._last_failed_count = 0
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def audit(self) -> MemorySafetyAuditResult:
        """
        Run complete memory safety audit.
        """

        checks = [
            self._safe_check(
                name="sensitive_write_blocked",
                fn=self._audit_sensitive_write_blocked,
            ),
            self._safe_check(
                name="sensitive_retrieval_filtered",
                fn=self._audit_sensitive_retrieval_filtered,
            ),
            self._safe_check(
                name="restricted_context_filtered",
                fn=self._audit_restricted_context_filtered,
            ),
            self._safe_check(
                name="delete_and_clear_policy",
                fn=self._audit_delete_and_clear_policy,
            ),
            self._safe_check(
                name="privacy_policy_high_risk_detection",
                fn=self._audit_privacy_policy_detection,
            ),
            self._safe_check(
                name="lifecycle_protects_pinned_and_critical",
                fn=self._audit_lifecycle_protection,
            ),
            self._safe_check(
                name="diagnostics_detects_unsafe_context",
                fn=self._audit_diagnostics_detect_unsafe_context,
            ),
            self._safe_check(
                name="cognition_uses_gateway_bridge",
                fn=self._audit_cognition_gateway_bridge,
            ),
        ]

        if self._config.include_restricted_opt_in_check:
            checks.append(
                self._safe_check(
                    name="restricted_context_requires_explicit_opt_in",
                    fn=self._audit_restricted_context_opt_in,
                )
            )

        status = (
            MemorySafetyAuditStatus.PASSED
            if all(check.passed for check in checks)
            else MemorySafetyAuditStatus.FAILED
        )
        result = MemorySafetyAuditResult(
            status=status,
            checks=tuple(checks),
            metadata={
                "auditor": self.name,
                "include_restricted_opt_in_check": (
                    self._config.include_restricted_opt_in_check
                ),
            },
        )
        self._record_result(result)

        self._logger.info(
            "memory_safety_audit_completed",
            auditor=self.name,
            status=result.status.value,
            passed_count=result.passed_count,
            failed_count=result.failed_count,
        )

        return result

    def snapshot(self) -> MemorySafetyAuditorSnapshot:
        """
        Return safety auditor diagnostics.
        """

        with self._lock:
            return MemorySafetyAuditorSnapshot(
                name=self.name,
                audit_count=self._audit_count,
                last_status=self._last_status,
                last_passed_count=self._last_passed_count,
                last_failed_count=self._last_failed_count,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Reset auditor diagnostics.
        """

        with self._lock:
            self._audit_count = 0
            self._last_status = None
            self._last_passed_count = 0
            self._last_failed_count = 0
            self._last_error = None

        self._logger.info("memory_safety_auditor_reset", auditor=self.name)

    def _safe_check(
        self,
        *,
        name: str,
        fn: Callable[[], MemorySafetyAuditCheck],
    ) -> MemorySafetyAuditCheck:
        try:
            return fn()

        except Exception as exc:
            return MemorySafetyAuditCheck(
                name=name,
                passed=False,
                risk_level=MemorySafetyRiskLevel.CRITICAL,
                detail=f"{type(exc).__name__}: {exc}",
                metadata={
                    "auditor": self.name,
                },
            )

    def _audit_sensitive_write_blocked(self) -> MemorySafetyAuditCheck:
        gateway = GovernedMemoryGateway(store=InMemoryMemoryStore())
        result = gateway.remember(
            MemoryWriteRequest(
                kind=MemoryKind.USER_PROFILE,
                text="Highly sensitive memory.",
                sensitivity=MemorySensitivity.SENSITIVE,
            )
        )
        snapshot = gateway.snapshot()

        passed = (
            result.blocked
            and not result.allowed
            and result.record is None
            and snapshot.store_snapshot.record_count == 0
        )

        return MemorySafetyAuditCheck(
            name="sensitive_write_blocked",
            passed=passed,
            risk_level=MemorySafetyRiskLevel.CRITICAL,
            detail="sensitive memory writes are blocked by default",
            metadata={
                "blocked": result.blocked,
                "record_count": snapshot.store_snapshot.record_count,
                "reason": result.reason,
            },
        )

    def _audit_sensitive_retrieval_filtered(self) -> MemorySafetyAuditCheck:
        store = InMemoryMemoryStore()
        store.put(
            MemoryRecord(
                kind=MemoryKind.USER_PROFILE,
                text="Sensitive profile memory.",
                sensitivity=MemorySensitivity.SENSITIVE,
            )
        )
        store.put(
            MemoryRecord(
                kind=MemoryKind.USER_PROFILE,
                text="Private profile memory.",
                sensitivity=MemorySensitivity.PRIVATE,
            )
        )
        gateway = GovernedMemoryGateway(store=store)
        result = gateway.retrieve(MemoryQuery(include_sensitive=True))

        sensitive_returned = any(
            record.sensitivity == MemorySensitivity.SENSITIVE
            for record in result.records
        )
        passed = (
            result.allowed
            and result.query.include_sensitive is False
            and not sensitive_returned
            and result.result_count == 1
        )

        return MemorySafetyAuditCheck(
            name="sensitive_retrieval_filtered",
            passed=passed,
            risk_level=MemorySafetyRiskLevel.CRITICAL,
            detail="sensitive retrieval is filtered by gateway policy",
            metadata={
                "result_count": result.result_count,
                "sensitive_returned": sensitive_returned,
                "governed_include_sensitive": result.query.include_sensitive,
            },
        )

    def _audit_restricted_context_filtered(self) -> MemorySafetyAuditCheck:
        retrieval = self._restricted_retrieval()
        context = MemoryContextBuilder().build(
            MemoryContextBuildRequest(retrievals=(retrieval,))
        )

        restricted_returned = any(
            item.policy_classification == MemoryPolicyClassification.RESTRICTED
            for item in context.items
        )
        passed = context.empty and not restricted_returned

        return MemorySafetyAuditCheck(
            name="restricted_context_filtered",
            passed=passed,
            risk_level=MemorySafetyRiskLevel.CRITICAL,
            detail="restricted memory is not injected into context by default",
            metadata={
                "context_items": context.item_count,
                "restricted_returned": restricted_returned,
            },
        )

    def _audit_restricted_context_opt_in(self) -> MemorySafetyAuditCheck:
        retrieval = self._restricted_retrieval()
        context = MemoryContextBuilder().build(
            MemoryContextBuildRequest(
                retrievals=(retrieval,),
                include_restricted=True,
            )
        )

        restricted_count = sum(
            1
            for item in context.items
            if item.policy_classification == MemoryPolicyClassification.RESTRICTED
        )
        passed = context.item_count == 1 and restricted_count == 1

        return MemorySafetyAuditCheck(
            name="restricted_context_requires_explicit_opt_in",
            passed=passed,
            risk_level=MemorySafetyRiskLevel.HIGH,
            detail="restricted context appears only with explicit opt-in",
            metadata={
                "context_items": context.item_count,
                "restricted_count": restricted_count,
            },
        )

    def _audit_delete_and_clear_policy(self) -> MemorySafetyAuditCheck:
        gateway = GovernedMemoryGateway(
            store=InMemoryMemoryStore(),
            config=MemoryGatewayConfig(
                allow_delete=False,
                allow_clear=False,
            ),
        )
        write = gateway.remember(
            MemoryWriteRequest(
                kind=MemoryKind.PROJECT,
                text="Delete policy test memory.",
            )
        )

        if write.record is None:
            raise RuntimeError("failed to seed delete policy memory.")

        deleted = gateway.delete(write.record.memory_id)
        gateway.clear()
        snapshot = gateway.snapshot()

        passed = (
            deleted is False
            and snapshot.delete_blocked_count == 1
            and snapshot.clear_blocked_count == 1
            and snapshot.store_snapshot.record_count == 1
        )

        return MemorySafetyAuditCheck(
            name="delete_and_clear_policy",
            passed=passed,
            risk_level=MemorySafetyRiskLevel.HIGH,
            detail="delete and clear policies are enforced by gateway",
            metadata={
                "deleted": deleted,
                "delete_blocked_count": snapshot.delete_blocked_count,
                "clear_blocked_count": snapshot.clear_blocked_count,
                "record_count": snapshot.store_snapshot.record_count,
            },
        )

    def _audit_privacy_policy_detection(self) -> MemorySafetyAuditCheck:
        policy = MemoryPrivacyPolicy()

        secret = policy.evaluate_write_request(
            MemoryWriteRequest(
                kind=MemoryKind.USER_PROFILE,
                text="My password is temporary123.",
            )
        )
        critical = policy.evaluate_write_request(
            MemoryWriteRequest(
                kind=MemoryKind.USER_PROFILE,
                text="My seed phrase is alpha beta gamma.",
            )
        )

        passed = (
            secret.decision == MemoryPrivacyDecisionKind.RESTRICT
            and secret.policy_classification
            == MemoryPolicyClassification.RESTRICTED
            and critical.decision == MemoryPrivacyDecisionKind.BLOCK
            and critical.policy_classification
            == MemoryPolicyClassification.BLOCKED
        )

        return MemorySafetyAuditCheck(
            name="privacy_policy_high_risk_detection",
            passed=passed,
            risk_level=MemorySafetyRiskLevel.CRITICAL,
            detail="privacy policy detects secret-like and critical memory",
            metadata={
                "secret_decision": secret.decision.value,
                "critical_decision": critical.decision.value,
                "secret_terms": secret.matched_terms,
                "critical_terms": critical.matched_terms,
            },
        )

    def _audit_lifecycle_protection(self) -> MemorySafetyAuditCheck:
        policy = MemoryLifecyclePolicy()
        pinned = policy.evaluate(
            MemoryRecord(
                kind=MemoryKind.PROJECT,
                text="Pinned memory must be protected.",
                retention=MemoryRetention.PINNED,
            )
        )
        critical = policy.evaluate(
            MemoryRecord(
                kind=MemoryKind.PROJECT,
                text="Critical memory must be retained.",
                importance=MemoryImportance.CRITICAL,
            )
        )

        passed = (
            pinned.decision == MemoryLifecycleDecisionKind.PIN
            and not pinned.delete_recommended
            and critical.decision == MemoryLifecycleDecisionKind.KEEP
            and not critical.delete_recommended
        )

        return MemorySafetyAuditCheck(
            name="lifecycle_protects_pinned_and_critical",
            passed=passed,
            risk_level=MemorySafetyRiskLevel.HIGH,
            detail="lifecycle policy protects pinned and critical memory",
            metadata={
                "pinned_decision": pinned.decision.value,
                "critical_decision": critical.decision.value,
            },
        )

    def _audit_diagnostics_detect_unsafe_context(self) -> MemorySafetyAuditCheck:
        item = MemoryContextItem(
            item_kind=MemoryContextItemKind.RETRIEVED_MEMORY,
            text="Unsafe blocked memory context item.",
            source=MemorySource.CONVERSATION,
            reason="manual unsafe audit item",
            confidence=1.0,
            policy_classification=MemoryPolicyClassification.BLOCKED,
        )
        context = MemoryContext(items=(item,), total_chars=item.char_count)
        diagnostics = MemoryDiagnosticsCollector(
            components={"gateway": GovernedMemoryGateway(store=InMemoryMemoryStore())}
        )
        check = diagnostics.audit_context(context)

        passed = check.status == MemoryDiagnosticStatus.FAILED

        return MemorySafetyAuditCheck(
            name="diagnostics_detects_unsafe_context",
            passed=passed,
            risk_level=MemorySafetyRiskLevel.CRITICAL,
            detail="diagnostics fails context containing blocked memory",
            metadata={
                "diagnostic_status": check.status.value,
                "diagnostic_detail": check.detail,
            },
        )

    def _audit_cognition_gateway_bridge(self) -> MemorySafetyAuditCheck:
        gateway = GovernedMemoryGateway(store=InMemoryMemoryStore())
        gateway.remember(
            MemoryWriteRequest(
                kind=MemoryKind.PROJECT,
                text="Cognition must use memory gateway bridge.",
                importance=MemoryImportance.HIGH,
                tags=("gateway", "cognition"),
            )
        )
        bridge = MemoryCognitionBridge(gateway=gateway)
        result = bridge.build_context_from_text("memory gateway bridge")

        passed = (
            result.allowed
            and not result.blocked
            and result.context.item_count > 0
            and result.as_cognition_metadata()["memory_context_id"]
            == result.context.context_id
        )

        return MemorySafetyAuditCheck(
            name="cognition_uses_gateway_bridge",
            passed=passed,
            risk_level=MemorySafetyRiskLevel.CRITICAL,
            detail="cognition receives memory only through gateway bridge",
            metadata={
                "context_id": result.context.context_id,
                "item_count": result.context.item_count,
                "gateway_reason": result.reason,
            },
        )

    @staticmethod
    def _restricted_retrieval() -> MemoryGatewayRetrievalResult:
        store = InMemoryMemoryStore()
        store.put(
            MemoryRecord(
                kind=MemoryKind.PROJECT,
                scope=MemoryScope.USER,
                text="Sensitive project memory.",
                sensitivity=MemorySensitivity.SENSITIVE,
            )
        )
        gateway = GovernedMemoryGateway(
            store=store,
            config=MemoryGatewayConfig(allow_sensitive_retrieval=True),
        )

        return gateway.retrieve(
            MemoryQuery(
                text="sensitive project memory",
                include_sensitive=True,
            )
        )

    def _record_result(self, result: MemorySafetyAuditResult) -> None:
        with self._lock:
            self._audit_count += 1
            self._last_status = result.status
            self._last_passed_count = result.passed_count
            self._last_failed_count = result.failed_count
            self._last_error = (
                None
                if result.passed
                else f"memory safety audit failed: {result.failed_count} checks"
            )


def audit_phase4_memory_safety() -> MemorySafetyAuditResult:
    """
    Convenience function for scripts and tests.
    """

    return MemorySafetyAuditor().audit()