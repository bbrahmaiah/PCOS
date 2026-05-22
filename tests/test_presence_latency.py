from __future__ import annotations

import pytest

from jarvis.presence.latency import (
    PresenceLatencyBudget,
    PresenceLatencyMeasurement,
    PresenceLatencyProfiler,
    PresenceLatencyReport,
)


def test_presence_latency_budget_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        PresenceLatencyBudget(engine_start_ms=0).validate()

    with pytest.raises(ValueError):
        PresenceLatencyBudget(voice_pipeline_ms=0).validate()

    with pytest.raises(ValueError):
        PresenceLatencyBudget(response_to_playback_ms=0).validate()

    with pytest.raises(ValueError):
        PresenceLatencyBudget(interruption_ms=0).validate()

    with pytest.raises(ValueError):
        PresenceLatencyBudget(engine_stop_ms=0).validate()


def test_presence_latency_measurement_model() -> None:
    measurement = PresenceLatencyMeasurement(
        name="example",
        duration_ms=10.0,
        budget_ms=20.0,
        within_budget=True,
        details={"ok": True},
    )

    assert measurement.name == "example"
    assert measurement.duration_ms == 10.0
    assert measurement.budget_ms == 20.0
    assert measurement.within_budget is True
    assert measurement.details == {"ok": True}


def test_presence_latency_profiler_rejects_bad_timeout() -> None:
    with pytest.raises(ValueError):
        PresenceLatencyProfiler(timeout_seconds=0)


def test_presence_latency_profiler_rejects_bad_poll_interval() -> None:
    with pytest.raises(ValueError):
        PresenceLatencyProfiler(poll_interval_seconds=0)


def test_presence_latency_profiler_runs_successfully() -> None:
    report = PresenceLatencyProfiler().run()

    assert isinstance(report, PresenceLatencyReport)
    assert report.passed is True
    assert report.errors == ()
    assert report.measurement_count == 5
    assert report.duration_ms >= 0

    names = {item.name for item in report.measurements}

    assert "engine_start" in names
    assert "voice_pipeline" in names
    assert "response_to_playback" in names
    assert "interruption" in names
    assert "engine_stop" in names

    for item in report.measurements:
        assert item.duration_ms >= 0
        assert item.budget_ms > 0