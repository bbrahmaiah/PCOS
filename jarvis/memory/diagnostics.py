from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from enum import StrEnum
from threading import RLock
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field, field_validator

from jarvis.memory.context import MemoryContext
from jarvis.memory.gateway import MemoryGatewayRetrievalResult
from jarvis.memory.models import (
    MemoryModel,
    MemoryPolicyClassification,
    utc_now,
)
from jarvis.runtime.observability.structured_logger import get_logger


class MemoryDiagnosticStatus(StrEnum):
    """
    Status for one diagnostic check or whole diagnostics report.
    """

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"


class MemoryDiagnosticCategory(StrEnum):
    """
    Category of memory diagnostic check.
    """

    COMPONENT = "component"
    RETRIEVAL_AUDIT = "retrieval_audit"
    CONTEXT_AUDIT = "context_audit"
    GATEWAY_BOUNDARY = "gateway_boundary"


class MemoryDiagnosticCheck(MemoryModel):
    """
    One memory diagnostic check.

    The check is intentionally explainable so memory health can be inspected
    without reading internal implementation details.
    """

    name: str
    category: MemoryDiagnosticCategory
    status: MemoryDiagnosticStatus
    detail: str
    checked_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("name", "detail")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class MemoryRuntimeDiagnostics(MemoryModel):
    """
    Full memory diagnostics report.
    """

    name: str
    status: MemoryDiagnosticStatus
    checks: tuple[MemoryDiagnosticCheck, ...]
    collected_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _name_required(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        return cleaned

    @property
    def check_count(self) -> int:
        return len(self.checks)

    @property
    def healthy_count(self) -> int:
        return sum(
            1
            for check in self.checks
            if check.status == MemoryDiagnosticStatus.HEALTHY
        )

    @property
    def degraded_count(self) -> int:
        return sum(
            1
            for check in self.checks
            if check.status == MemoryDiagnosticStatus.DEGRADED
        )

    @property
    def failed_count(self) -> int:
        return sum(
            1
            for check in self.checks
            if check.status == MemoryDiagnosticStatus.FAILED
        )

    @property
    def passed(self) -> bool:
        return self.status == MemoryDiagnosticStatus.HEALTHY


@dataclass(frozen=True, slots=True)
class MemoryDiagnosticsCollectorConfig:
    """
    Configuration for MemoryDiagnosticsCollector.
    """

    name: str = "memory_diagnostics_collector"
    required_components: tuple[str, ...] = ("gateway",)

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        for component in self.required_components:
            if not component.strip():
                raise ValueError("required_components cannot contain empty names.")


@dataclass(frozen=True, slots=True)
class MemoryDiagnosticsCollectorSnapshot:
    """
    Observable diagnostics for the diagnostics collector itself.
    """

    name: str
    collect_count: int
    audit_retrieval_count: int
    audit_context_count: int
    last_status: MemoryDiagnosticStatus | None
    last_error: str | None


@runtime_checkable
class MemorySnapshotProvider(Protocol):
    """
    Protocol for memory components that expose snapshot diagnostics.
    """

    def snapshot(self) -> object:
        """Return a component snapshot."""


class MemoryDiagnosticsCollector:
    """
    Collects diagnostics from memory runtime components.

    Responsibilities:
    - collect snapshot health from registered memory components
    - detect missing required components
    - audit retrieval explainability
    - audit cognition-ready memory context
    - expose collector diagnostics

    Non-responsibilities:
    - no memory writes
    - no direct MemoryStore access for cognition
    - no LLM calls
    - no policy mutation
    """

    def __init__(
        self,
        *,
        components: Mapping[str, MemorySnapshotProvider] | None = None,
        config: MemoryDiagnosticsCollectorConfig | None = None,
    ) -> None:
        self._config = config or MemoryDiagnosticsCollectorConfig()
        self._config.validate()

        self._components = dict(components or {})
        self._lock = RLock()
        self._logger = get_logger("memory.diagnostics")

        self._collect_count = 0
        self._audit_retrieval_count = 0
        self._audit_context_count = 0
        self._last_status: MemoryDiagnosticStatus | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def register(
        self,
        name: str,
        component: MemorySnapshotProvider,
    ) -> None:
        """
        Register a component for future diagnostics collection.
        """

        cleaned = name.strip()

        if not cleaned:
            raise ValueError("component name cannot be empty.")

        with self._lock:
            self._components[cleaned] = component

        self._logger.info(
            "memory_diagnostics_component_registered",
            collector=self.name,
            component=cleaned,
        )

    def collect(self) -> MemoryRuntimeDiagnostics:
        """
        Collect diagnostics from all registered components.
        """

        with self._lock:
            self._collect_count += 1
            self._last_error = None
            components = dict(self._components)

        checks: list[MemoryDiagnosticCheck] = []

        for required in self._config.required_components:
            if required not in components:
                checks.append(
                    MemoryDiagnosticCheck(
                        name=f"required_component:{required}",
                        category=MemoryDiagnosticCategory.COMPONENT,
                        status=MemoryDiagnosticStatus.FAILED,
                        detail=f"required memory component is missing: {required}",
                        metadata={
                            "component": required,
                        },
                    )
                )

        for name, component in components.items():
            checks.append(self._component_check(name=name, component=component))

        report = MemoryRuntimeDiagnostics(
            name=self.name,
            status=self._overall_status(tuple(checks)),
            checks=tuple(checks),
            metadata={
                "component_count": len(components),
                "required_components": self._config.required_components,
            },
        )
        self._record_report(report)

        self._logger.info(
            "memory_diagnostics_collected",
            collector=self.name,
            status=report.status.value,
            check_count=report.check_count,
            failed_count=report.failed_count,
            degraded_count=report.degraded_count,
        )

        return report

    def audit_retrieval(
        self,
        retrieval: MemoryGatewayRetrievalResult,
    ) -> MemoryDiagnosticCheck:
        """
        Audit one gateway retrieval for explainability.

        Every retrieved result must preserve:
        - source
        - reason
        - confidence
        - timestamp
        - policy classification
        """

        with self._lock:
            self._audit_retrieval_count += 1

        if retrieval.blocked or not retrieval.allowed:
            return MemoryDiagnosticCheck(
                name="retrieval_explainability",
                category=MemoryDiagnosticCategory.RETRIEVAL_AUDIT,
                status=MemoryDiagnosticStatus.DEGRADED,
                detail="retrieval was blocked or not allowed",
                metadata={
                    "reason": retrieval.reason,
                    "result_count": retrieval.result_count,
                },
            )

        missing_count = 0

        for result in retrieval.results:
            explanation = result.explanation

            if not explanation.reason.strip():
                missing_count += 1

            if explanation.retrieved_at.tzinfo is None:
                missing_count += 1

            if explanation.confidence < 0.0 or explanation.confidence > 1.0:
                missing_count += 1

        status = (
            MemoryDiagnosticStatus.HEALTHY
            if missing_count == 0
            else MemoryDiagnosticStatus.FAILED
        )
        detail = (
            "retrieval explanations are complete"
            if missing_count == 0
            else "retrieval explanations are incomplete"
        )

        return MemoryDiagnosticCheck(
            name="retrieval_explainability",
            category=MemoryDiagnosticCategory.RETRIEVAL_AUDIT,
            status=status,
            detail=detail,
            metadata={
                "query_id": retrieval.query.query_id,
                "result_count": retrieval.result_count,
                "missing_field_count": missing_count,
            },
        )

    def audit_context(self, context: MemoryContext) -> MemoryDiagnosticCheck:
        """
        Audit cognition-ready memory context for policy-safe items.
        """

        with self._lock:
            self._audit_context_count += 1

        if context.empty:
            return MemoryDiagnosticCheck(
                name="memory_context_audit",
                category=MemoryDiagnosticCategory.CONTEXT_AUDIT,
                status=MemoryDiagnosticStatus.DEGRADED,
                detail="memory context is empty",
                metadata={
                    "context_id": context.context_id,
                    "item_count": context.item_count,
                },
            )

        blocked_count = sum(
            1
            for item in context.items
            if item.policy_classification
            in {
                MemoryPolicyClassification.BLOCKED,
                MemoryPolicyClassification.REDACTED,
            }
        )
        restricted_count = sum(
            1
            for item in context.items
            if item.policy_classification == MemoryPolicyClassification.RESTRICTED
        )

        if blocked_count > 0:
            status = MemoryDiagnosticStatus.FAILED
            detail = "memory context contains blocked or redacted items"

        elif restricted_count > 0:
            status = MemoryDiagnosticStatus.DEGRADED
            detail = "memory context contains restricted items"

        else:
            status = MemoryDiagnosticStatus.HEALTHY
            detail = "memory context is policy-safe and auditable"

        return MemoryDiagnosticCheck(
            name="memory_context_audit",
            category=MemoryDiagnosticCategory.CONTEXT_AUDIT,
            status=status,
            detail=detail,
            metadata={
                "context_id": context.context_id,
                "item_count": context.item_count,
                "total_chars": context.total_chars,
                "restricted_count": restricted_count,
                "blocked_count": blocked_count,
            },
        )

    def snapshot(self) -> MemoryDiagnosticsCollectorSnapshot:
        """
        Return collector diagnostics.
        """

        with self._lock:
            return MemoryDiagnosticsCollectorSnapshot(
                name=self.name,
                collect_count=self._collect_count,
                audit_retrieval_count=self._audit_retrieval_count,
                audit_context_count=self._audit_context_count,
                last_status=self._last_status,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Reset collector diagnostics.
        """

        with self._lock:
            self._collect_count = 0
            self._audit_retrieval_count = 0
            self._audit_context_count = 0
            self._last_status = None
            self._last_error = None

        self._logger.info("memory_diagnostics_collector_reset", collector=self.name)

    def _component_check(
        self,
        *,
        name: str,
        component: MemorySnapshotProvider,
    ) -> MemoryDiagnosticCheck:
        try:
            snapshot = component.snapshot()
            metadata = self._snapshot_metadata(snapshot)

            status = (
                MemoryDiagnosticStatus.DEGRADED
                if metadata.get("last_error")
                else MemoryDiagnosticStatus.HEALTHY
            )
            detail = (
                f"component {name} reported last_error"
                if status == MemoryDiagnosticStatus.DEGRADED
                else f"component {name} snapshot collected"
            )

            return MemoryDiagnosticCheck(
                name=f"component:{name}",
                category=MemoryDiagnosticCategory.COMPONENT,
                status=status,
                detail=detail,
                metadata={
                    "component": name,
                    "snapshot": metadata,
                },
            )

        except Exception as exc:
            return MemoryDiagnosticCheck(
                name=f"component:{name}",
                category=MemoryDiagnosticCategory.COMPONENT,
                status=MemoryDiagnosticStatus.FAILED,
                detail=f"component snapshot failed: {type(exc).__name__}: {exc}",
                metadata={
                    "component": name,
                },
            )

    @staticmethod
    def _snapshot_metadata(snapshot: object) -> dict[str, object]:
        if isinstance(snapshot, BaseModel):
            return dict(snapshot.model_dump(mode="python"))

        if is_dataclass(snapshot) and not isinstance(snapshot, type):
            return {
                field.name: getattr(snapshot, field.name)
                for field in fields(snapshot)
            }

        if isinstance(snapshot, Mapping):
            return dict(snapshot)

        return {
            "value": str(snapshot),
        }

    @staticmethod
    def _overall_status(
        checks: tuple[MemoryDiagnosticCheck, ...],
    ) -> MemoryDiagnosticStatus:
        if any(check.status == MemoryDiagnosticStatus.FAILED for check in checks):
            return MemoryDiagnosticStatus.FAILED

        if any(check.status == MemoryDiagnosticStatus.DEGRADED for check in checks):
            return MemoryDiagnosticStatus.DEGRADED

        return MemoryDiagnosticStatus.HEALTHY

    def _record_report(self, report: MemoryRuntimeDiagnostics) -> None:
        with self._lock:
            self._last_status = report.status
            self._last_error = (
                None
                if report.status == MemoryDiagnosticStatus.HEALTHY
                else f"memory diagnostics status={report.status.value}"
            )