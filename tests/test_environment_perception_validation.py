from __future__ import annotations

import pytest

from jarvis.environment import (
    CaptureCpuSample,
    VisualPerceptionCheckKind,
    VisualPerceptionCheckStatus,
    VisualPerceptionGateReason,
    VisualPerceptionValidationGate,
)


def test_gate_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        VisualPerceptionValidationGate(name=" ")


def test_capture_cpu_sample_budget() -> None:
    sample = CaptureCpuSample(
        capture_cpu_percent=8.0,
        budget_percent=12.0,
    )

    assert sample.within_budget is True


def test_capture_works_check() -> None:
    gate = VisualPerceptionValidationGate()

    result = gate.check_capture_works()

    assert result.kind == VisualPerceptionCheckKind.CAPTURE_WORKS
    assert result.status == VisualPerceptionCheckStatus.PASSED


def test_delta_reduces_cost_check() -> None:
    gate = VisualPerceptionValidationGate()

    result = gate.check_delta_reduces_cost()

    assert result.kind == VisualPerceptionCheckKind.DELTA_REDUCES_COST
    assert result.status == VisualPerceptionCheckStatus.PASSED


def test_privacy_zone_never_captured_check() -> None:
    gate = VisualPerceptionValidationGate()

    result = gate.check_privacy_zone_never_captured()

    assert result.kind == VisualPerceptionCheckKind.PRIVACY_ZONE_NEVER_CAPTURED
    assert result.status == VisualPerceptionCheckStatus.PASSED


def test_ocr_confidence_check() -> None:
    gate = VisualPerceptionValidationGate()

    result = gate.check_ocr_confidence_works()

    assert result.kind == VisualPerceptionCheckKind.OCR_CONFIDENCE_WORKS
    assert result.status == VisualPerceptionCheckStatus.PASSED


def test_low_confidence_ocr_check() -> None:
    gate = VisualPerceptionValidationGate()

    result = gate.check_low_confidence_ocr_flagged()

    assert result.kind == VisualPerceptionCheckKind.LOW_CONFIDENCE_OCR_FLAGGED
    assert result.status == VisualPerceptionCheckStatus.PASSED


def test_code_ocr_preserves_indentation_check() -> None:
    gate = VisualPerceptionValidationGate()

    result = gate.check_code_ocr_preserves_indentation()

    assert result.kind == VisualPerceptionCheckKind.CODE_OCR_PRESERVES_INDENTATION
    assert result.status == VisualPerceptionCheckStatus.PASSED


def test_ui_elements_detected_check() -> None:
    gate = VisualPerceptionValidationGate()

    result = gate.check_ui_elements_detected()

    assert result.kind == VisualPerceptionCheckKind.UI_ELEMENTS_DETECTED
    assert result.status == VisualPerceptionCheckStatus.PASSED


def test_accessibility_priority_check() -> None:
    gate = VisualPerceptionValidationGate()

    result = gate.check_accessibility_priority_works()

    assert result.kind == VisualPerceptionCheckKind.ACCESSIBILITY_PRIORITY_WORKS
    assert result.status == VisualPerceptionCheckStatus.PASSED


def test_multi_monitor_stable_check() -> None:
    gate = VisualPerceptionValidationGate()

    result = gate.check_multi_monitor_stable()

    assert result.kind == VisualPerceptionCheckKind.MULTI_MONITOR_STABLE
    assert result.status == VisualPerceptionCheckStatus.PASSED


def test_capture_cpu_budget_check() -> None:
    gate = VisualPerceptionValidationGate()

    result = gate.check_capture_cpu_within_budget()

    assert result.kind == VisualPerceptionCheckKind.CAPTURE_CPU_WITHIN_BUDGET
    assert result.status == VisualPerceptionCheckStatus.PASSED


def test_run_seals_visual_perception() -> None:
    gate = VisualPerceptionValidationGate()

    report = gate.run()

    assert report.sealed is True
    assert report.reason == VisualPerceptionGateReason.GATE_PASSED
    assert report.failed_count == 0
    assert report.passed_count == 10


def test_snapshot_tracks_reports_and_checks() -> None:
    gate = VisualPerceptionValidationGate()

    gate.run()
    snapshot = gate.snapshot()

    assert snapshot.report_count == 1
    assert snapshot.check_count == 10
    assert snapshot.passed_count == 10
    assert snapshot.sealed_report_count == 1


def test_reset_clears_gate() -> None:
    gate = VisualPerceptionValidationGate()
    gate.run()

    gate.reset()
    snapshot = gate.snapshot()

    assert snapshot.report_count == 0
    assert snapshot.runtime_event_count == 1
    assert snapshot.last_reason == VisualPerceptionGateReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert VisualPerceptionCheckKind.CAPTURE_WORKS.value == "capture_works"
    assert VisualPerceptionCheckStatus.PASSED.value == "passed"
    assert VisualPerceptionGateReason.GATE_PASSED.value == "gate_passed"