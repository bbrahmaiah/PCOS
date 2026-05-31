from __future__ import annotations

from scripts.smoke_environment_cognition import (
    EnvironmentCognitionSmokeRuntime,
    SmokeCheck,
    SmokeReport,
    run_smoke,
)


def test_smoke_check_model() -> None:
    check = SmokeCheck(
        name="example",
        passed=True,
        detail="ok",
        metadata={"count": 1},
    )

    assert check.name == "example"
    assert check.passed is True
    assert check.metadata["count"] == 1


def test_smoke_report_counts() -> None:
    report = SmokeReport(
        passed=False,
        checks=(
            SmokeCheck(name="a", passed=True, detail="ok"),
            SmokeCheck(name="b", passed=False, detail="bad"),
        ),
    )

    assert report.passed_count == 1
    assert report.failed_count == 1
    assert report.to_dict()["passed"] is False


def test_environment_cognition_smoke_runtime_passes() -> None:
    report = EnvironmentCognitionSmokeRuntime().run()

    assert report.passed is True
    assert report.failed_count == 0
    assert report.passed_count == len(report.checks)


def test_run_smoke_helper_passes() -> None:
    report = run_smoke()

    assert report.passed is True
    assert report.failed_count == 0


def test_smoke_contains_required_checks() -> None:
    report = run_smoke()
    names = {check.name for check in report.checks}

    assert names == {
        "workers_healthy",
        "observers_active",
        "timeline_records_deltas",
        "trust_scores_emitted",
        "capture_works",
        "ocr_works",
        "ui_detection_works",
        "environment_graph_builds",
        "visual_grounding_resolves_target",
        "policy_blocks_unsafe_action",
        "simulation_predicts_expected_state",
        "safe_action_verifies",
        "failed_action_recovers",
        "workflow_memory_stores",
        "voice_environment_fusion_works",
        "continuous_assistance_observes",
    }