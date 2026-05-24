from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.memory import (
    GovernedMemoryGateway,
    InMemoryMemoryStore,
    MemoryCognitionBridge,
    MemoryContextBuilder,
    MemoryContextBuildRequest,
    MemoryContextBuildStatus,
    MemoryDiagnosticCategory,
    MemoryDiagnosticCheck,
    MemoryDiagnosticsCollector,
    MemoryDiagnosticsCollectorConfig,
    MemoryDiagnosticStatus,
    MemoryImportance,
    MemoryKind,
    MemoryPolicyClassification,
    MemoryQuery,
    MemoryRuntimeDiagnostics,
    MemoryWriteRequest,
)


class BrokenSnapshotComponent:
    def snapshot(self) -> object:
        raise RuntimeError("snapshot failed")


def make_gateway() -> GovernedMemoryGateway:
    return GovernedMemoryGateway(store=InMemoryMemoryStore())


def seed_gateway(gateway: GovernedMemoryGateway) -> None:
    gateway.remember(
        MemoryWriteRequest(
            kind=MemoryKind.PROJECT,
            text="JARVIS memory diagnostics verify explainability.",
            importance=MemoryImportance.HIGH,
            tags=("jarvis", "diagnostics"),
        )
    )


def test_memory_diagnostics_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        MemoryDiagnosticsCollectorConfig(name=" ").validate()

    with pytest.raises(ValueError):
        MemoryDiagnosticsCollectorConfig(required_components=(" ",)).validate()


def test_memory_diagnostic_check_requires_text_fields() -> None:
    with pytest.raises(ValidationError):
        MemoryDiagnosticCheck(
            name=" ",
            category=MemoryDiagnosticCategory.COMPONENT,
            status=MemoryDiagnosticStatus.HEALTHY,
            detail="valid",
        )

    with pytest.raises(ValidationError):
        MemoryDiagnosticCheck(
            name="component",
            category=MemoryDiagnosticCategory.COMPONENT,
            status=MemoryDiagnosticStatus.HEALTHY,
            detail=" ",
        )


def test_memory_runtime_diagnostics_counts() -> None:
    healthy = MemoryDiagnosticCheck(
        name="healthy",
        category=MemoryDiagnosticCategory.COMPONENT,
        status=MemoryDiagnosticStatus.HEALTHY,
        detail="ok",
    )
    failed = MemoryDiagnosticCheck(
        name="failed",
        category=MemoryDiagnosticCategory.COMPONENT,
        status=MemoryDiagnosticStatus.FAILED,
        detail="bad",
    )
    report = MemoryRuntimeDiagnostics(
        name="report",
        status=MemoryDiagnosticStatus.FAILED,
        checks=(healthy, failed),
    )

    assert report.check_count == 2
    assert report.healthy_count == 1
    assert report.failed_count == 1
    assert report.passed is False


def test_diagnostics_collector_reports_missing_required_component() -> None:
    collector = MemoryDiagnosticsCollector(components={})

    report = collector.collect()

    assert report.status == MemoryDiagnosticStatus.FAILED
    assert report.failed_count == 1
    assert "required_component:gateway" in {check.name for check in report.checks}


def test_diagnostics_collector_collects_component_snapshots() -> None:
    gateway = make_gateway()
    seed_gateway(gateway)
    builder = MemoryContextBuilder()
    bridge = MemoryCognitionBridge(gateway=gateway, context_builder=builder)

    collector = MemoryDiagnosticsCollector(
        components={
            "gateway": gateway,
            "context_builder": builder,
            "cognition_bridge": bridge,
        }
    )

    report = collector.collect()
    snapshot = collector.snapshot()

    assert report.status == MemoryDiagnosticStatus.HEALTHY
    assert report.failed_count == 0
    assert report.check_count == 3
    assert snapshot.collect_count == 1
    assert snapshot.last_status == MemoryDiagnosticStatus.HEALTHY


def test_diagnostics_collector_marks_component_with_last_error_degraded() -> None:
    gateway = make_gateway()

    gateway.clear()

    collector = MemoryDiagnosticsCollector(components={"gateway": gateway})
    report = collector.collect()

    assert report.status == MemoryDiagnosticStatus.DEGRADED
    assert report.degraded_count == 1


def test_diagnostics_collector_marks_broken_component_failed() -> None:
    gateway = make_gateway()
    collector = MemoryDiagnosticsCollector(
        components={
            "gateway": gateway,
            "broken": BrokenSnapshotComponent(),
        }
    )

    report = collector.collect()

    assert report.status == MemoryDiagnosticStatus.FAILED
    assert report.failed_count == 1
    assert any("snapshot failed" in check.detail for check in report.checks)


def test_diagnostics_audits_retrieval_explainability() -> None:
    gateway = make_gateway()
    seed_gateway(gateway)
    retrieval = gateway.retrieve(MemoryQuery(text="memory diagnostics"))
    collector = MemoryDiagnosticsCollector(components={"gateway": gateway})

    check = collector.audit_retrieval(retrieval)
    snapshot = collector.snapshot()

    assert check.status == MemoryDiagnosticStatus.HEALTHY
    assert check.category == MemoryDiagnosticCategory.RETRIEVAL_AUDIT
    assert check.metadata["result_count"] == 1
    assert snapshot.audit_retrieval_count == 1


def test_diagnostics_audits_empty_context_as_degraded() -> None:
    context = MemoryContextBuilder().build(MemoryContextBuildRequest())
    collector = MemoryDiagnosticsCollector(components={"gateway": make_gateway()})

    check = collector.audit_context(context)

    assert context.status == MemoryContextBuildStatus.EMPTY
    assert check.status == MemoryDiagnosticStatus.DEGRADED
    assert check.metadata["item_count"] == 0


def test_diagnostics_audits_policy_safe_context() -> None:
    gateway = make_gateway()
    seed_gateway(gateway)
    retrieval = gateway.retrieve(MemoryQuery(text="memory diagnostics"))
    context = MemoryContextBuilder().build(
        MemoryContextBuildRequest(retrievals=(retrieval,))
    )
    collector = MemoryDiagnosticsCollector(components={"gateway": gateway})

    check = collector.audit_context(context)

    assert check.status == MemoryDiagnosticStatus.HEALTHY
    assert check.detail == "memory context is policy-safe and auditable"
    assert check.metadata["item_count"] == 1


def test_diagnostics_audits_restricted_context_as_degraded() -> None:
    store = InMemoryMemoryStore()
    store.put(
        MemoryWriteRequest(
            kind=MemoryKind.PROJECT,
            text="Sensitive diagnostic memory.",
            sensitivity=__import__("jarvis.memory").memory.MemorySensitivity.SENSITIVE,
        ).to_record()
    )
    gateway = __import__("jarvis.memory").memory.GovernedMemoryGateway(
        store=store,
        config=__import__("jarvis.memory").memory.MemoryGatewayConfig(
            allow_sensitive_retrieval=True
        ),
    )
    retrieval = gateway.retrieve(MemoryQuery(text="diagnostic", include_sensitive=True))
    context = MemoryContextBuilder().build(
        MemoryContextBuildRequest(
            retrievals=(retrieval,),
            include_restricted=True,
        )
    )
    collector = MemoryDiagnosticsCollector(components={"gateway": gateway})

    check = collector.audit_context(context)

    assert check.status == MemoryDiagnosticStatus.DEGRADED
    assert check.metadata["restricted_count"] == 1


def test_diagnostics_collector_register_and_reset() -> None:
    collector = MemoryDiagnosticsCollector(components={})

    collector.register("gateway", make_gateway())
    report = collector.collect()

    assert report.status == MemoryDiagnosticStatus.HEALTHY

    collector.reset()
    snapshot = collector.snapshot()

    assert snapshot.collect_count == 0
    assert snapshot.audit_context_count == 0
    assert snapshot.last_status is None


def test_memory_diagnostics_enum_values_are_stable() -> None:
    assert MemoryDiagnosticStatus.HEALTHY.value == "healthy"
    assert MemoryDiagnosticStatus.DEGRADED.value == "degraded"
    assert MemoryDiagnosticStatus.FAILED.value == "failed"
    assert MemoryDiagnosticCategory.COMPONENT.value == "component"
    assert MemoryDiagnosticCategory.RETRIEVAL_AUDIT.value == "retrieval_audit"
    assert MemoryDiagnosticCategory.CONTEXT_AUDIT.value == "context_audit"
    assert MemoryPolicyClassification.ALLOWED.value == "allowed"