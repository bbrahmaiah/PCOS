from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any


@dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    counters: dict[str, int]
    gauges: dict[str, float]
    timings: dict[str, list[float]]


class MetricsRegistry:
    """
    Thread-safe local metrics registry.

    This tracks counters, gauges, and timing samples for the runtime.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._counters: dict[str, int] = {}
        self._gauges: dict[str, float] = {}
        self._timings: dict[str, list[float]] = {}

    def increment(self, name: str, value: int = 1) -> None:
        if value < 0:
            raise ValueError("counter increment value cannot be negative.")

        clean_name = self._validate_name(name)

        with self._lock:
            self._counters[clean_name] = self._counters.get(clean_name, 0) + value

    def set_gauge(self, name: str, value: float) -> None:
        clean_name = self._validate_name(name)

        with self._lock:
            self._gauges[clean_name] = float(value)

    def record_timing(self, name: str, duration_ms: float) -> None:
        if duration_ms < 0:
            raise ValueError("duration_ms cannot be negative.")

        clean_name = self._validate_name(name)

        with self._lock:
            self._timings.setdefault(clean_name, []).append(float(duration_ms))

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            return MetricsSnapshot(
                counters=dict(self._counters),
                gauges=dict(self._gauges),
                timings={key: list(value) for key, value in self._timings.items()},
            )

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._timings.clear()

    def summary(self) -> dict[str, Any]:
        snapshot = self.snapshot()

        timing_summary: dict[str, dict[str, float | int]] = {}

        for name, values in snapshot.timings.items():
            if not values:
                continue

            timing_summary[name] = {
                "count": len(values),
                "min_ms": min(values),
                "max_ms": max(values),
                "avg_ms": sum(values) / len(values),
            }

        return {
            "counters": snapshot.counters,
            "gauges": snapshot.gauges,
            "timings": timing_summary,
        }

    @staticmethod
    def _validate_name(name: str) -> str:
        cleaned = name.strip()

        if not cleaned:
            raise ValueError("metric name cannot be empty.")

        return cleaned


_default_metrics = MetricsRegistry()


def get_metrics() -> MetricsRegistry:
    return _default_metrics