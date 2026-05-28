from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.environment import (
    EnvironmentAttentionRequirement,
    EnvironmentCircuitBreakerPolicy,
    EnvironmentFailurePolicy,
    EnvironmentRestartPolicy,
    EnvironmentSubsystem,
    EnvironmentWorkerBudget,
    EnvironmentWorkerCapability,
    EnvironmentWorkerDescriptor,
    EnvironmentWorkerHealth,
    EnvironmentWorkerKind,
    EnvironmentWorkerPriority,
    EnvironmentWorkerRegistrationReason,
    EnvironmentWorkerRegistrationStatus,
    EnvironmentWorkerRegistry,
    default_environment_workers,
)


def test_budget_rejects_invalid_values() -> None:
    with pytest.raises(ValidationError):
        EnvironmentWorkerBudget(
            latency_budget_ms=0,
            cpu_budget_percent=10,
            memory_budget_mb=128,
        )

    with pytest.raises(ValidationError):
        EnvironmentWorkerBudget(
            latency_budget_ms=10,
            cpu_budget_percent=101,
            memory_budget_mb=128,
        )


def test_descriptor_requires_text_fields() -> None:
    with pytest.raises(ValidationError):
        _descriptor(name=" ")


def test_background_worker_cannot_be_high_priority() -> None:
    with pytest.raises(ValidationError):
        _descriptor(
            priority=EnvironmentWorkerPriority.HIGH,
            background=True,
        )


def test_critical_worker_requires_circuit_breaker() -> None:
    with pytest.raises(ValidationError):
        _descriptor(
            priority=EnvironmentWorkerPriority.CRITICAL,
            circuit=EnvironmentCircuitBreakerPolicy.NONE,
        )


def test_default_workers_include_all_phase8_workers() -> None:
    workers = default_environment_workers()
    kinds = {worker.kind for worker in workers}

    assert len(workers) == 19
    assert len(kinds) == 19
    assert EnvironmentWorkerKind.CAPTURE in kinds
    assert EnvironmentWorkerKind.OCR in kinds
    assert EnvironmentWorkerKind.UI_DETECTION in kinds
    assert EnvironmentWorkerKind.ENVIRONMENT_OBSERVER in kinds
    assert EnvironmentWorkerKind.ENVIRONMENT_STATE in kinds
    assert EnvironmentWorkerKind.ENVIRONMENT_TIMELINE in kinds
    assert EnvironmentWorkerKind.TRUST_CALIBRATION in kinds
    assert EnvironmentWorkerKind.VISUAL_PRIORITY in kinds
    assert EnvironmentWorkerKind.WORKSPACE_GRAPH in kinds
    assert EnvironmentWorkerKind.UI_SEMANTIC in kinds
    assert EnvironmentWorkerKind.VISUAL_GROUNDING in kinds
    assert EnvironmentWorkerKind.INTENT_PERSISTENCE in kinds
    assert EnvironmentWorkerKind.SIMULATION in kinds
    assert EnvironmentWorkerKind.INTERACTION in kinds
    assert EnvironmentWorkerKind.VERIFICATION in kinds
    assert EnvironmentWorkerKind.RECOVERY in kinds
    assert EnvironmentWorkerKind.ENVIRONMENT_MEMORY in kinds
    assert EnvironmentWorkerKind.HUMAN_COLLABORATION in kinds
    assert EnvironmentWorkerKind.SECURITY_AUDIT in kinds


def test_every_default_worker_declares_governance_fields() -> None:
    for worker in default_environment_workers():
        assert worker.capability
        assert worker.subsystem
        assert worker.budget.latency_budget_ms > 0
        assert worker.budget.cpu_budget_percent > 0
        assert worker.budget.memory_budget_mb > 0
        assert worker.attention_requirement
        assert worker.priority
        assert worker.restart_policy
        assert worker.circuit_breaker_policy
        assert worker.failure_policy


def test_registry_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        EnvironmentWorkerRegistry(name=" ")


def test_registry_registers_worker() -> None:
    registry = EnvironmentWorkerRegistry()
    worker = _descriptor()

    result = registry.register(worker)

    assert result.success is True
    assert result.status == EnvironmentWorkerRegistrationStatus.REGISTERED
    assert registry.get(worker.kind) == worker


def test_registry_rejects_duplicate_without_replace() -> None:
    registry = EnvironmentWorkerRegistry()
    worker = _descriptor()

    registry.register(worker)
    result = registry.register(worker)

    assert result.success is False
    assert result.reason == (
        EnvironmentWorkerRegistrationReason.WORKER_DUPLICATE_REJECTED
    )


def test_registry_replaces_worker_when_requested() -> None:
    registry = EnvironmentWorkerRegistry()
    worker = _descriptor()
    replacement = worker.model_copy(update={"name": "ReplacementWorker"})

    registry.register(worker)
    result = registry.register(replacement, replace=True)

    assert result.success is True
    assert result.status == EnvironmentWorkerRegistrationStatus.UPDATED
    assert registry.get(worker.kind) == replacement


def test_registry_registers_defaults_and_becomes_ready() -> None:
    registry = EnvironmentWorkerRegistry()

    results = registry.register_defaults()
    event = registry.readiness_event()

    assert len(results) == 19
    assert all(result.success for result in results)
    assert registry.is_ready() is True
    assert event.reason == (
        EnvironmentWorkerRegistrationReason.ALL_REQUIRED_WORKERS_REGISTERED
    )


def test_registry_reports_missing_required_workers() -> None:
    registry = EnvironmentWorkerRegistry()

    assert EnvironmentWorkerKind.CAPTURE in registry.missing_required_workers()
    assert registry.is_ready() is False

    event = registry.readiness_event()

    assert event.reason == EnvironmentWorkerRegistrationReason.REQUIRED_WORKER_MISSING


def test_registry_updates_health() -> None:
    registry = EnvironmentWorkerRegistry()
    worker = _descriptor()

    registry.register(worker)
    result = registry.update_health(worker.kind, EnvironmentWorkerHealth.DEGRADED)
    updated = registry.get(worker.kind)

    assert result.success is True
    assert updated is not None
    assert updated.health == EnvironmentWorkerHealth.DEGRADED


def test_update_health_rejects_missing_worker() -> None:
    registry = EnvironmentWorkerRegistry()

    result = registry.update_health(
        EnvironmentWorkerKind.CAPTURE,
        EnvironmentWorkerHealth.FAILED,
    )

    assert result.success is False
    assert result.reason == EnvironmentWorkerRegistrationReason.WORKER_NOT_FOUND


def test_workers_can_be_queried_by_subsystem() -> None:
    registry = EnvironmentWorkerRegistry()

    registry.register_defaults()
    perception_workers = registry.workers_for_subsystem(
        EnvironmentSubsystem.VISUAL_PERCEPTION
    )

    assert perception_workers
    assert all(
        worker.subsystem == EnvironmentSubsystem.VISUAL_PERCEPTION
        for worker in perception_workers
    )


def test_snapshot_tracks_registry_state() -> None:
    registry = EnvironmentWorkerRegistry()

    registry.register_defaults()
    registry.update_health(
        EnvironmentWorkerKind.OCR,
        EnvironmentWorkerHealth.DEGRADED,
    )
    snapshot = registry.snapshot()

    assert snapshot.worker_count == 19
    assert snapshot.required_count == 19
    assert snapshot.degraded_count == 1
    assert snapshot.background_count > 0
    assert snapshot.event_count >= 20


def test_registry_reset_clears_workers() -> None:
    registry = EnvironmentWorkerRegistry()

    registry.register_defaults()
    registry.reset()
    snapshot = registry.snapshot()

    assert snapshot.worker_count == 0
    assert snapshot.last_reason == EnvironmentWorkerRegistrationReason.RUNTIME_RESET


def test_critical_defaults_have_circuit_breakers() -> None:
    critical_workers = [
        worker
        for worker in default_environment_workers()
        if worker.priority == EnvironmentWorkerPriority.CRITICAL
    ]

    assert critical_workers
    assert all(
        worker.circuit_breaker_policy != EnvironmentCircuitBreakerPolicy.NONE
        for worker in critical_workers
    )


def test_interaction_and_verification_are_critical() -> None:
    workers = {worker.kind: worker for worker in default_environment_workers()}

    assert workers[EnvironmentWorkerKind.INTERACTION].priority == (
        EnvironmentWorkerPriority.CRITICAL
    )
    assert workers[EnvironmentWorkerKind.VERIFICATION].priority == (
        EnvironmentWorkerPriority.CRITICAL
    )


def test_enum_values_are_stable() -> None:
    assert EnvironmentWorkerKind.CAPTURE.value == "capture_worker"
    assert EnvironmentWorkerCapability.CAPTURE_SCREEN.value == "capture_screen"
    assert EnvironmentWorkerPriority.CRITICAL.value == "critical"
    assert EnvironmentRestartPolicy.ON_FAILURE.value == "on_failure"
    assert EnvironmentFailurePolicy.FAIL_FAST.value == "fail_fast"


def _descriptor(
    *,
    name: str = "CaptureWorker",
    priority: EnvironmentWorkerPriority = EnvironmentWorkerPriority.NORMAL,
    circuit: EnvironmentCircuitBreakerPolicy = (
        EnvironmentCircuitBreakerPolicy.OPEN_ON_FAILURE_RATE
    ),
    background: bool = False,
) -> EnvironmentWorkerDescriptor:
    return EnvironmentWorkerDescriptor(
        kind=EnvironmentWorkerKind.CAPTURE,
        name=name,
        subsystem=EnvironmentSubsystem.VISUAL_PERCEPTION,
        capability=EnvironmentWorkerCapability.CAPTURE_SCREEN,
        budget=EnvironmentWorkerBudget(
            latency_budget_ms=50,
            cpu_budget_percent=10,
            memory_budget_mb=128,
            background_allowed=background,
        ),
        attention_requirement=EnvironmentAttentionRequirement.FOCUSED,
        priority=priority,
        restart_policy=EnvironmentRestartPolicy.ON_FAILURE,
        circuit_breaker_policy=circuit,
        failure_policy=EnvironmentFailurePolicy.DEGRADE_GRACEFULLY,
        health=EnvironmentWorkerHealth.HEALTHY,
    )