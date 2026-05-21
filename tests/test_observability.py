from __future__ import annotations

import pytest

from jarvis.runtime.observability.metrics import MetricsRegistry
from jarvis.runtime.observability.performance_monitor import PerformanceMonitor
from jarvis.runtime.observability.tracing import Tracer


def test_metrics_counter_increment() -> None:
    metrics = MetricsRegistry()

    metrics.increment("events.published")
    metrics.increment("events.published", 2)

    snapshot = metrics.snapshot()

    assert snapshot.counters["events.published"] == 3


def test_metrics_rejects_negative_increment() -> None:
    metrics = MetricsRegistry()

    with pytest.raises(ValueError):
        metrics.increment("bad.counter", -1)


def test_metrics_rejects_empty_name() -> None:
    metrics = MetricsRegistry()

    with pytest.raises(ValueError):
        metrics.increment("")


def test_metrics_gauge_and_timing_summary() -> None:
    metrics = MetricsRegistry()

    metrics.set_gauge("workers.active", 3)
    metrics.record_timing("event.dispatch", 10.0)
    metrics.record_timing("event.dispatch", 20.0)

    summary = metrics.summary()

    assert summary["gauges"]["workers.active"] == 3.0
    assert summary["timings"]["event.dispatch"]["count"] == 2
    assert summary["timings"]["event.dispatch"]["avg_ms"] == 15.0


def test_performance_monitor_records_timing() -> None:
    monitor = PerformanceMonitor()
    monitor.metrics.reset()

    with monitor.measure("test.operation", correlation_id="abc"):
        pass

    summary = monitor.metrics.summary()

    assert "latency.test.operation" in summary["timings"]
    assert summary["timings"]["latency.test.operation"]["count"] == 1


def test_performance_monitor_rejects_empty_operation() -> None:
    monitor = PerformanceMonitor()

    with pytest.raises(ValueError):
        with monitor.measure(""):
            pass


def test_tracer_span_success() -> None:
    tracer = Tracer(service_name="test_service")

    with tracer.span("test.span", correlation_id="abc") as span:
        assert span.name == "test.span"
        assert span.correlation_id == "abc"


def test_tracer_span_reraises_exception() -> None:
    tracer = Tracer(service_name="test_service")

    with pytest.raises(RuntimeError):
        with tracer.span("failing.span"):
            raise RuntimeError("boom")