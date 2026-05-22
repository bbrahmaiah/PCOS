from __future__ import annotations

import pytest

from jarvis.presence.validation import (
    PresenceIntegrationValidator,
    PresenceValidationCheck,
    PresenceValidationReport,
)


def test_presence_validation_check_model() -> None:
    check = PresenceValidationCheck(
        name="example",
        passed=True,
        details={"value": 1},
    )

    assert check.name == "example"
    assert check.passed is True
    assert check.details == {"value": 1}


def test_presence_integration_validator_rejects_bad_timeout() -> None:
    with pytest.raises(ValueError):
        PresenceIntegrationValidator(timeout_seconds=0)


def test_presence_integration_validator_rejects_bad_poll_interval() -> None:
    with pytest.raises(ValueError):
        PresenceIntegrationValidator(poll_interval_seconds=0)


def test_presence_integration_validator_passes() -> None:
    report = PresenceIntegrationValidator().run()

    assert isinstance(report, PresenceValidationReport)
    assert report.passed is True
    assert report.failed_count == 0
    assert report.passed_count == len(report.checks)
    assert report.duration_ms >= 0

    check_names = {check.name for check in report.checks}

    assert "engine_started" in check_names
    assert "workers_subscribed" in check_names
    assert "voice_to_transcript_pipeline" in check_names
    assert "dialogue_bridge" in check_names
    assert "response_to_playback_pipeline" in check_names
    assert "interruption_pipeline" in check_names
    assert "presence_state_after_interruption" in check_names
    assert "engine_stopped" in check_names