from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from jarvis.latency.models import (
    LatencyBudget,
    LatencyBudgetResult,
    LatencyMeasurement,
    LatencyOperation,
    LatencyPercentile,
    LatencySeverity,
    LatencySubsystem,
    LatencyTarget,
    LatencyViolation,
)


@dataclass(frozen=True, slots=True)
class LatencyBudgetRegistryConfig:
    """
    Configuration for Step 0 latency budget registry.
    """

    name: str = "latency_budget_registry"
    register_defaults: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")


class LatencyBudgetRegistry:
    """
    Registry of SLA-style latency contracts.

    This is the source of truth for Phase 7 latency expectations.
    """

    def __init__(
        self,
        *,
        config: LatencyBudgetRegistryConfig | None = None,
        budgets: tuple[LatencyBudget, ...] = (),
    ) -> None:
        self._config = config or LatencyBudgetRegistryConfig()
        self._config.validate()

        self._budgets: dict[LatencyOperation, LatencyBudget] = {}
        self._violations: list[LatencyViolation] = []
        self._lock = RLock()

        if self._config.register_defaults:
            for budget in default_latency_budgets():
                self.register(budget)

        for budget in budgets:
            self.register(budget)

    @property
    def name(self) -> str:
        return self._config.name

    def register(self, budget: LatencyBudget) -> None:
        with self._lock:
            self._budgets[budget.operation] = budget

    def get(self, operation: LatencyOperation) -> LatencyBudget | None:
        with self._lock:
            return self._budgets.get(operation)

    def budgets(self) -> tuple[LatencyBudget, ...]:
        with self._lock:
            return tuple(self._budgets.values())

    def violations(self) -> tuple[LatencyViolation, ...]:
        with self._lock:
            return tuple(self._violations)

    def evaluate(
        self,
        measurement: LatencyMeasurement,
        *,
        percentile: LatencyPercentile = LatencyPercentile.P95,
    ) -> LatencyBudgetResult:
        budget = self.get(measurement.operation)

        if budget is None:
            raise ValueError(
                f"no latency budget registered for {measurement.operation}"
    )

        target = budget.target_for(percentile)

        if target is None:
            raise ValueError(
                "no target registered for "
                 f"{measurement.operation} at p{percentile.value}"
)

        severity = self._severity_for(
            duration_ms=measurement.duration_ms,
            target=target,
        )
        result = LatencyBudgetResult(
            operation=measurement.operation,
            subsystem=measurement.subsystem,
            duration_ms=measurement.duration_ms,
            percentile=percentile,
            severity=severity,
            target_ms=target.target_ms,
            warning_ms=target.warning_ms,
            max_ms=target.max_ms,
            within_target=measurement.duration_ms <= target.target_ms,
            within_max=measurement.duration_ms <= target.max_ms,
            message=self._message_for(severity),
            measurement_id=measurement.measurement_id,
        )

        if severity in {
            LatencySeverity.WARNING,
            LatencySeverity.VIOLATION,
            LatencySeverity.CRITICAL,
        }:
            self._record_violation(measurement=measurement, result=result)

        return result

    def reset_violations(self) -> None:
        with self._lock:
            self._violations.clear()

    def _record_violation(
        self,
        *,
        measurement: LatencyMeasurement,
        result: LatencyBudgetResult,
    ) -> None:
        violation = LatencyViolation(
            operation=measurement.operation,
            subsystem=measurement.subsystem,
            severity=result.severity,
            duration_ms=measurement.duration_ms,
            max_ms=result.max_ms,
            message=result.message,
            measurement_id=measurement.measurement_id,
        )

        with self._lock:
            self._violations.append(violation)

    @staticmethod
    def _severity_for(
        *,
        duration_ms: float,
        target: LatencyTarget,
    ) -> LatencySeverity:
        if duration_ms <= target.target_ms:
            return LatencySeverity.OK

        if duration_ms <= target.warning_ms:
            return LatencySeverity.WARNING

        if duration_ms <= target.max_ms:
            return LatencySeverity.VIOLATION

        return LatencySeverity.CRITICAL

    @staticmethod
    def _message_for(severity: LatencySeverity) -> str:
        if severity == LatencySeverity.OK:
            return "latency is within target"

        if severity == LatencySeverity.WARNING:
            return "latency exceeded target but remains below warning ceiling"

        if severity == LatencySeverity.VIOLATION:
            return "latency exceeded warning ceiling but remains under max budget"

        return "latency exceeded max budget"


def latency_target(
    *,
    operation: LatencyOperation,
    subsystem: LatencySubsystem,
    percentile: LatencyPercentile,
    target_ms: int,
    warning_ms: int,
    max_ms: int,
    description: str,
) -> LatencyTarget:
    return LatencyTarget(
        operation=operation,
        subsystem=subsystem,
        percentile=percentile,
        target_ms=target_ms,
        warning_ms=warning_ms,
        max_ms=max_ms,
        description=description,
    )


def latency_budget(
    *,
    operation: LatencyOperation,
    subsystem: LatencySubsystem,
    owner: str,
    p95_target_ms: int,
    p95_warning_ms: int,
    p95_max_ms: int,
    p99_target_ms: int,
    p99_warning_ms: int,
    p99_max_ms: int,
    description: str,
    hard_realtime: bool = False,
) -> LatencyBudget:
    return LatencyBudget(
        operation=operation,
        subsystem=subsystem,
        owner=owner,
        hard_realtime=hard_realtime,
        targets=(
            latency_target(
                operation=operation,
                subsystem=subsystem,
                percentile=LatencyPercentile.P95,
                target_ms=p95_target_ms,
                warning_ms=p95_warning_ms,
                max_ms=p95_max_ms,
                description=description,
            ),
            latency_target(
                operation=operation,
                subsystem=subsystem,
                percentile=LatencyPercentile.P99,
                target_ms=p99_target_ms,
                warning_ms=p99_warning_ms,
                max_ms=p99_max_ms,
                description=f"{description} p99 tail latency",
            ),
        ),
    )


def default_latency_budgets() -> tuple[LatencyBudget, ...]:
    """
    Default Step 0 latency contracts.

    These are initial Phase 7 contracts. Later steps will refine them with
    measured baselines.
    """

    return (
        latency_budget(
            operation=LatencyOperation.STT_FIRST_TOKEN,
            subsystem=LatencySubsystem.PRESENCE,
            owner="presence_runtime",
            p95_target_ms=150,
            p95_warning_ms=220,
            p95_max_ms=300,
            p99_target_ms=220,
            p99_warning_ms=320,
            p99_max_ms=450,
            description="time from speech start to first STT partial token",
            hard_realtime=True,
        ),
        latency_budget(
            operation=LatencyOperation.STT_FINALIZATION,
            subsystem=LatencySubsystem.PRESENCE,
            owner="presence_runtime",
            p95_target_ms=350,
            p95_warning_ms=500,
            p95_max_ms=700,
            p99_target_ms=500,
            p99_warning_ms=750,
            p99_max_ms=1000,
            description="time from turn end to finalized transcript",
            hard_realtime=True,
        ),
        latency_budget(
            operation=LatencyOperation.MEMORY_RETRIEVAL,
            subsystem=LatencySubsystem.MEMORY,
            owner="memory_runtime",
            p95_target_ms=150,
            p95_warning_ms=250,
            p95_max_ms=400,
            p99_target_ms=250,
            p99_warning_ms=400,
            p99_max_ms=650,
            description="time to retrieve first useful memory context",
        ),
        latency_budget(
            operation=LatencyOperation.CONTEXT_BUILD,
            subsystem=LatencySubsystem.COGNITION,
            owner="cognition_runtime",
            p95_target_ms=80,
            p95_warning_ms=140,
            p95_max_ms=220,
            p99_target_ms=140,
            p99_warning_ms=220,
            p99_max_ms=350,
            description="time to assemble usable model context",
        ),
        latency_budget(
            operation=LatencyOperation.LLM_FIRST_TOKEN,
            subsystem=LatencySubsystem.COGNITION,
            owner="cognition_runtime",
            p95_target_ms=300,
            p95_warning_ms=450,
            p95_max_ms=700,
            p99_target_ms=450,
            p99_warning_ms=700,
            p99_max_ms=1000,
            description="time from LLM request to first streamed token",
            hard_realtime=True,
        ),
        latency_budget(
            operation=LatencyOperation.LLM_FULL_RESPONSE,
            subsystem=LatencySubsystem.COGNITION,
            owner="cognition_runtime",
            p95_target_ms=2500,
            p95_warning_ms=4000,
            p95_max_ms=7000,
            p99_target_ms=4000,
            p99_warning_ms=7000,
            p99_max_ms=12000,
            description="time from LLM request to completed response",
        ),
        latency_budget(
            operation=LatencyOperation.TTS_FIRST_AUDIO,
            subsystem=LatencySubsystem.PRESENCE,
            owner="presence_runtime",
            p95_target_ms=120,
            p95_warning_ms=200,
            p95_max_ms=350,
            p99_target_ms=200,
            p99_warning_ms=350,
            p99_max_ms=550,
            description="time from first TTS chunk to first audio bytes",
            hard_realtime=True,
        ),
        latency_budget(
            operation=LatencyOperation.PLAYBACK_STARTUP,
            subsystem=LatencySubsystem.PLAYBACK,
            owner="presence_runtime",
            p95_target_ms=60,
            p95_warning_ms=100,
            p95_max_ms=180,
            p99_target_ms=100,
            p99_warning_ms=180,
            p99_max_ms=300,
            description="time from audio bytes ready to audible playback",
            hard_realtime=True,
        ),
        latency_budget(
            operation=LatencyOperation.INTERRUPT_RESPONSE,
            subsystem=LatencySubsystem.ORCHESTRATION,
            owner="orchestration_runtime",
            p95_target_ms=250,
            p95_warning_ms=350,
            p95_max_ms=600,
            p99_target_ms=350,
            p99_warning_ms=600,
            p99_max_ms=900,
            description="time from user interruption to stopped active response",
            hard_realtime=True,
        ),
        latency_budget(
            operation=LatencyOperation.RECOVERY_RECONSTRUCT,
            subsystem=LatencySubsystem.RECOVERY,
            owner="recovery_runtime",
            p95_target_ms=300,
            p95_warning_ms=500,
            p95_max_ms=900,
            p99_target_ms=500,
            p99_warning_ms=900,
            p99_max_ms=1500,
            description="time to reconstruct last known good runtime state",
        ),
        latency_budget(
            operation=LatencyOperation.TOOL_FIRST_FEEDBACK,
            subsystem=LatencySubsystem.TOOLS,
            owner="tool_runtime",
            p95_target_ms=200,
            p95_warning_ms=350,
            p95_max_ms=600,
            p99_target_ms=350,
            p99_warning_ms=600,
            p99_max_ms=1000,
            description="time from tool start to first user-visible progress feedback",
        ),
        latency_budget(
            operation=LatencyOperation.ACTION_PROGRESS_FEEDBACK,
            subsystem=LatencySubsystem.TOOLS,
            owner="tool_runtime",
            p95_target_ms=500,
            p95_warning_ms=900,
            p95_max_ms=1500,
            p99_target_ms=900,
            p99_warning_ms=1500,
            p99_max_ms=2500,
            description="time between action progress feedback updates",
        ),
    )