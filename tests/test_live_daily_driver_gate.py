from __future__ import annotations

import pytest

from jarvis.live import (
    LiveDailyDriverCheckKind,
    LiveDailyDriverGateConfig,
    LiveDailyDriverGateStatus,
    LiveDailyDriverProfile,
    LiveDailyDriverRuntimeGate,
)


def test_daily_driver_gate_config_validation() -> None:
    with pytest.raises(ValueError):
        LiveDailyDriverGateConfig(user_label=" ")

    with pytest.raises(ValueError):
        LiveDailyDriverGateConfig(assistant_name=" ")


def test_daily_driver_gate_passes() -> None:
    report = LiveDailyDriverRuntimeGate(
        config=LiveDailyDriverGateConfig(
            profile=LiveDailyDriverProfile.BRONZE,
            user_label="Balu",
            assistant_name="JARVIS",
            use_real_voice_contracts=True,
            auto_health_check=False,
            auto_recover=False,
        )
    ).run()

    assert report.status == LiveDailyDriverGateStatus.PASSED
    assert report.passed is True
    assert report.failed_count == 0


def test_daily_driver_gate_has_all_checks() -> None:
    report = LiveDailyDriverRuntimeGate().run()

    assert len(report.checks) == 9
    assert {check.kind for check in report.checks} == {
        LiveDailyDriverCheckKind.STARTUP,
        LiveDailyDriverCheckKind.BACKGROUND_SPEECH_IGNORED,
        LiveDailyDriverCheckKind.WAKE_DIALOGUE,
        LiveDailyDriverCheckKind.RESPONSE_BOUNDARY,
        LiveDailyDriverCheckKind.INTERRUPTION,
        LiveDailyDriverCheckKind.HEALTH_MONITOR,
        LiveDailyDriverCheckKind.RECOVERY,
        LiveDailyDriverCheckKind.SHUTDOWN,
        LiveDailyDriverCheckKind.NO_SCRIPTED_SPEECH,
    }


def test_daily_driver_gate_metadata_is_preserved() -> None:
    report = LiveDailyDriverRuntimeGate(
        config=LiveDailyDriverGateConfig(
            metadata={"run": "daily-driver"},
        )
    ).run()

    assert report.metadata["run"] == "daily-driver"
    assert report.metadata["step"] == "50K"


def test_daily_driver_gate_blocks_scripted_speech_check() -> None:
    report = LiveDailyDriverRuntimeGate().run()
    check = next(
        item for item in report.checks
        if item.kind == LiveDailyDriverCheckKind.RESPONSE_BOUNDARY
    )

    assert check.passed is True
    assert check.metadata["boundary_status"] == "blocked"


def test_daily_driver_gate_enum_values_are_stable() -> None:
    assert LiveDailyDriverGateStatus.PASSED.value == "passed"
    assert LiveDailyDriverProfile.BRONZE.value == "bronze"
    assert LiveDailyDriverCheckKind.SHUTDOWN.value == "shutdown"