from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from threading import RLock
from types import TracebackType
from typing import Self

from jarvis.latency.budgets import LatencyBudgetRegistry
from jarvis.latency.models import (
    LatencyBudgetResult,
    LatencyEvent,
    LatencyEventKind,
    LatencyMeasurement,
    LatencyOperation,
    LatencyPercentile,
    LatencyRuntimeSnapshot,
    LatencySeverity,
    LatencySpan,
    LatencySubsystem,
    LatencyViolation,
    PercentileSnapshot,
)


@dataclass(frozen=True, slots=True)
class PercentileTrackerConfig:
    """
    Percentile tracker configuration.
    """

    max_samples_per_operation: int = 1000

    def validate(self) -> None:
        if self.max_samples_per_operation < 1:
            raise ValueError("max_samples_per_operation must be positive.")


class PercentileTracker:
    """
    In-memory percentile tracker.

    This intentionally tracks p50/p90/p95/p99/worst instead of average.
    """

    def __init__(
        self,
        *,
        config: PercentileTrackerConfig | None = None,
    ) -> None:
        self._config = config or PercentileTrackerConfig()
        self._config.validate()

        self._samples: dict[
            tuple[LatencyOperation, LatencySubsystem],
            deque[float],
        ] = defaultdict(
            lambda: deque(maxlen=self._config.max_samples_per_operation)
        )
        self._lock = RLock()

    def record(self, measurement: LatencyMeasurement) -> None:
        key = (measurement.operation, measurement.subsystem)

        with self._lock:
            self._samples[key].append(measurement.duration_ms)

    def record_many(self, measurements: Iterable[LatencyMeasurement]) -> None:
        for measurement in measurements:
            self.record(measurement)

    def snapshot_for(
        self,
        *,
        operation: LatencyOperation,
        subsystem: LatencySubsystem,
    ) -> PercentileSnapshot:
        key = (operation, subsystem)

        with self._lock:
            values = tuple(self._samples.get(key, ()))

        return self._snapshot_from_values(
            operation=operation,
            subsystem=subsystem,
            values=values,
        )

    def snapshots(self) -> tuple[PercentileSnapshot, ...]:
        with self._lock:
            items = tuple(
                (key, tuple(values)) for key, values in self._samples.items()
            )

        return tuple(
            self._snapshot_from_values(
                operation=operation,
                subsystem=subsystem,
                values=values,
            )
            for (operation, subsystem), values in items
        )

    def sample_count(
        self,
        *,
        operation: LatencyOperation,
        subsystem: LatencySubsystem,
    ) -> int:
        key = (operation, subsystem)

        with self._lock:
            return len(self._samples.get(key, ()))

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()

    @staticmethod
    def percentile(values: Iterable[float], percentile: LatencyPercentile) -> float:
        ordered = sorted(values)

        if not ordered:
            return 0.0

        if len(ordered) == 1:
            return ordered[0]

        rank = (percentile.value / 100.0) * (len(ordered) - 1)
        lower_index = int(rank)
        upper_index = min(lower_index + 1, len(ordered) - 1)
        fraction = rank - lower_index

        lower = ordered[lower_index]
        upper = ordered[upper_index]

        return lower + (upper - lower) * fraction

    @classmethod
    def _snapshot_from_values(
        cls,
        *,
        operation: LatencyOperation,
        subsystem: LatencySubsystem,
        values: tuple[float, ...],
    ) -> PercentileSnapshot:
        return PercentileSnapshot(
            operation=operation,
            subsystem=subsystem,
            sample_count=len(values),
            p50_ms=cls.percentile(values, LatencyPercentile.P50),
            p90_ms=cls.percentile(values, LatencyPercentile.P90),
            p95_ms=cls.percentile(values, LatencyPercentile.P95),
            p99_ms=cls.percentile(values, LatencyPercentile.P99),
            worst_ms=max(values) if values else 0.0,
        )


@dataclass(frozen=True, slots=True)
class LatencyMeasurementRuntimeConfig:
    """
    Step 0 latency measurement runtime configuration.
    """

    name: str = "latency_measurement_runtime"
    track_percentiles: bool = True
    evaluate_budgets: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")


class LatencyMeasurementRuntime:
    """
    Phase 7 Step 0 latency measurement runtime.

    Responsibilities:
    - record latency measurements
    - evaluate budgets
    - track percentile snapshots
    - emit typed latency events
    - expose runtime diagnostics

    Non-responsibilities:
    - no optimization
    - no profiling graph
    - no pipeline flamegraph
    - no streaming pipeline mutation
    """

    def __init__(
        self,
        *,
        config: LatencyMeasurementRuntimeConfig | None = None,
        budget_registry: LatencyBudgetRegistry | None = None,
        percentile_tracker: PercentileTracker | None = None,
    ) -> None:
        self._config = config or LatencyMeasurementRuntimeConfig()
        self._config.validate()

        self._budget_registry = budget_registry or LatencyBudgetRegistry()
        self._tracker = percentile_tracker or PercentileTracker()

        self._measurements: list[LatencyMeasurement] = []
        self._results: list[LatencyBudgetResult] = []
        self._events: list[LatencyEvent] = []
        self._lock = RLock()
        self._last_severity: LatencySeverity | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def record(
        self,
        measurement: LatencyMeasurement,
    ) -> LatencyBudgetResult | None:
        if self._config.track_percentiles:
            self._tracker.record(measurement)

        result: LatencyBudgetResult | None = None

        if self._config.evaluate_budgets:
            result = self._budget_registry.evaluate(measurement)
            self._last_severity = result.severity

        with self._lock:
            self._measurements.append(measurement)

            if result is not None:
                self._results.append(result)

            self._events.append(
                LatencyEvent(
                    kind=LatencyEventKind.MEASUREMENT_RECORDED,
                    operation=measurement.operation,
                    subsystem=measurement.subsystem,
                    trace_id=measurement.trace_id,
                    span_id=measurement.span_id,
                    duration_ms=measurement.duration_ms,
                    severity=result.severity if result is not None else None,
                )
            )

            if result is not None and result.severity != LatencySeverity.OK:
                self._events.append(
                    LatencyEvent(
                        kind=self._event_kind_for(result.severity),
                        operation=measurement.operation,
                        subsystem=measurement.subsystem,
                        trace_id=measurement.trace_id,
                        span_id=measurement.span_id,
                        duration_ms=measurement.duration_ms,
                        severity=result.severity,
                    )
                )

        return result

    def record_span(self, span: LatencySpan) -> LatencyBudgetResult | None:
        return self.record(span.to_measurement())

    def measurements(self) -> tuple[LatencyMeasurement, ...]:
        with self._lock:
            return tuple(self._measurements)

    def results(self) -> tuple[LatencyBudgetResult, ...]:
        with self._lock:
            return tuple(self._results)

    def events(self) -> tuple[LatencyEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def percentile_snapshots(self) -> tuple[PercentileSnapshot, ...]:
        return self._tracker.snapshots()

    def violations(self) -> tuple[LatencyViolation, ...]:
        return self._budget_registry.violations()

    def snapshot(self) -> LatencyRuntimeSnapshot:
        with self._lock:
            results = tuple(self._results)
            measurement_count = len(self._measurements)
            event_count = len(self._events)
            last_severity = self._last_severity

        warnings = sum(
            1 for result in results if result.severity == LatencySeverity.WARNING
        )
        violations = sum(
            1 for result in results if result.severity == LatencySeverity.VIOLATION
        )
        critical = sum(
            1 for result in results if result.severity == LatencySeverity.CRITICAL
        )

        return LatencyRuntimeSnapshot(
            name=self.name,
            budget_count=len(self._budget_registry.budgets()),
            measurement_count=measurement_count,
            violation_count=violations,
            warning_count=warnings,
            critical_count=critical,
            percentile_count=len(self._tracker.snapshots()),
            last_severity=last_severity,
            metadata={"event_count": event_count},
        )

    def reset(self) -> None:
        with self._lock:
            self._measurements.clear()
            self._results.clear()
            self._events.clear()
            self._last_severity = None

        self._tracker.reset()
        self._budget_registry.reset_violations()

    @staticmethod
    def _event_kind_for(severity: LatencySeverity) -> LatencyEventKind:
        if severity == LatencySeverity.WARNING:
            return LatencyEventKind.BUDGET_WARNING

        if severity == LatencySeverity.VIOLATION:
            return LatencyEventKind.BUDGET_VIOLATION

        return LatencyEventKind.BUDGET_CRITICAL


class LatencyTimer:
    """
    Synchronous context manager for measuring one operation.

    Step 0 only defines the primitive. Step 1 will build pipeline profiler spans.
    """

    def __init__(
        self,
        *,
        runtime: LatencyMeasurementRuntime,
        operation: LatencyOperation,
        subsystem: LatencySubsystem,
        trace_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self._runtime = runtime
        self._operation = operation
        self._subsystem = subsystem
        self._trace_id = trace_id
        self._metadata = metadata or {}
        self._start_ns: int | None = None
        self._span: LatencySpan | None = None

    @property
    def span(self) -> LatencySpan | None:
        return self._span

    def __enter__(self) -> Self:
        self._start_ns = time.perf_counter_ns()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        end_ns = time.perf_counter_ns()

        if self._start_ns is None:
            raise RuntimeError("latency timer exited before entering")

        self._span = LatencySpan(
            operation=self._operation,
            subsystem=self._subsystem,
            start_ns=self._start_ns,
            end_ns=end_ns,
            trace_id=self._trace_id,
            metadata=dict(self._metadata),
        )
        self._runtime.record_span(self._span)