from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator

from jarvis.runtime.observability.metrics import get_metrics
from jarvis.runtime.observability.structured_logger import get_logger


class PerformanceMonitor:
    """
    Measures and records runtime operation latency.

    Used by EventBus, workers, state engine, security checks, and kernel loops.
    """

    def __init__(self) -> None:
        self.metrics = get_metrics()
        self.logger = get_logger("observability.performance")

    @contextmanager
    def measure(
        self,
        operation: str,
        *,
        correlation_id: str | None = None,
    ) -> Iterator[None]:
        clean_operation = operation.strip()

        if not clean_operation:
            raise ValueError("operation cannot be empty.")

        start = perf_counter()
        success = False

        try:
            yield
            success = True
        finally:
            duration_ms = (perf_counter() - start) * 1000
            metric_name = f"latency.{clean_operation}"

            self.metrics.record_timing(metric_name, duration_ms)

            self.logger.info(
                "operation_measured",
                operation=clean_operation,
                duration_ms=round(duration_ms, 3),
                success=success,
                correlation_id=correlation_id,
            )


_default_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    return _default_monitor