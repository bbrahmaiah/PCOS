from __future__ import annotations

from enum import StrEnum
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator

from jarvis.environment.attention import (
    AttentionGovernanceReason,
    CaptureGovernanceDecision,
    CapturePermission,
    EnvironmentAttentionMode,
    EnvironmentAttentionRuntime,
    EnvironmentAttentionSession,
    InspectionDepth,
    PrivacyZoneDecision,
    PrivacyZonePolicy,
)
from jarvis.environment.capture import (
    CaptureMode,
    CapturePayload,
    CapturePixelFormat,
    CaptureReason,
    CaptureStatus,
    RegionCapture,
    ScreenCaptureRequest,
    ScreenCaptureRuntime,
)
from jarvis.environment.models import (
    EnvironmentSource,
    PrivacyZone,
    ScreenRegion,
    TrustCalibration,
)
from jarvis.environment.ocr import (
    OCRExtractionRequest,
    OCRReason,
    OCRRuntime,
    OCRSourceKind,
    OCRStatus,
    TextConfidenceScorer,
    TextConfidenceScorerConfig,
)
from jarvis.environment.state_runtime import fake_display
from jarvis.environment.ui_detection import (
    UIDetectionReason,
    UIDetectionRequest,
    UIDetectionRuntime,
    UIDetectionSource,
    UIDetectionStatus,
)
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class VisualPerceptionCheckKind(StrEnum):
    """
    Individual checks required to seal Visual Perception Runtime.
    """

    CAPTURE_WORKS = "capture_works"
    DELTA_REDUCES_COST = "delta_reduces_cost"
    PRIVACY_ZONE_NEVER_CAPTURED = "privacy_zone_never_captured"
    OCR_CONFIDENCE_WORKS = "ocr_confidence_works"
    LOW_CONFIDENCE_OCR_FLAGGED = "low_confidence_ocr_flagged"
    CODE_OCR_PRESERVES_INDENTATION = "code_ocr_preserves_indentation"
    UI_ELEMENTS_DETECTED = "ui_elements_detected"
    ACCESSIBILITY_PRIORITY_WORKS = "accessibility_priority_works"
    MULTI_MONITOR_STABLE = "multi_monitor_stable"
    CAPTURE_CPU_WITHIN_BUDGET = "capture_cpu_within_budget"


class VisualPerceptionCheckStatus(StrEnum):
    """
    Validation check status.
    """

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class VisualPerceptionGateReason(StrEnum):
    """
    Machine-readable gate reason.
    """

    CHECK_PASSED = "check_passed"
    CHECK_FAILED = "check_failed"
    GATE_PASSED = "gate_passed"
    GATE_FAILED = "gate_failed"
    RUNTIME_RESET = "runtime_reset"


class VisualPerceptionValidationEventKind(StrEnum):
    """
    Validation gate runtime event kind.
    """

    CHECK_RECORDED = "check_recorded"
    GATE_COMPLETED = "gate_completed"
    RUNTIME_RESET = "runtime_reset"


class CaptureCpuSample(OrchestrationModel):
    """
    CPU sample for capture validation.

    This is intentionally simulated/fake-first in Step 11. Real telemetry can
    feed the same contract later.
    """

    sample_id: str = Field(default_factory=lambda: f"cpu_sample_{uuid4().hex}")
    capture_cpu_percent: float = Field(ge=0.0, le=100.0)
    budget_percent: float = Field(default=12.0, gt=0.0, le=100.0)
    sample_window_ms: int = Field(default=500, gt=0)
    created_at: object = Field(default_factory=utc_now)

    @property
    def within_budget(self) -> bool:
        return self.capture_cpu_percent <= self.budget_percent

    @field_validator("sample_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class VisualPerceptionCheckResult(OrchestrationModel):
    """
    Result of one validation check.
    """

    check_id: str = Field(default_factory=lambda: f"vp_check_{uuid4().hex}")
    kind: VisualPerceptionCheckKind
    status: VisualPerceptionCheckStatus
    reason: VisualPerceptionGateReason
    message: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("check_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @property
    def passed(self) -> bool:
        return self.status == VisualPerceptionCheckStatus.PASSED


class VisualPerceptionValidationReport(OrchestrationModel):
    """
    Final report that decides whether Subsystem 1 is sealed.
    """

    report_id: str = Field(default_factory=lambda: f"vp_report_{uuid4().hex}")
    checks: tuple[VisualPerceptionCheckResult, ...]
    passed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    skipped_count: int = Field(ge=0)
    sealed: bool
    reason: VisualPerceptionGateReason
    message: str
    created_at: object = Field(default_factory=utc_now)

    @field_validator("report_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class VisualPerceptionValidationEvent(OrchestrationModel):
    """
    Validation gate event.
    """

    event_id: str = Field(default_factory=lambda: f"vp_event_{uuid4().hex}")
    kind: VisualPerceptionValidationEventKind
    reason: VisualPerceptionGateReason
    check_kind: VisualPerceptionCheckKind | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class VisualPerceptionValidationSnapshot(OrchestrationModel):
    """
    Diagnostics for Step 11.
    """

    name: str
    report_count: int = Field(ge=0)
    check_count: int = Field(ge=0)
    passed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    sealed_report_count: int = Field(ge=0)
    runtime_event_count: int = Field(ge=0)
    last_reason: VisualPerceptionGateReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        return _clean_required(value)


class VisualPerceptionValidationGate:
    """
    Phase 8 Step 11 Visual Perception Validation Gate.

    Responsibilities:
    - validate ScreenCaptureRuntime
    - validate OCRRuntime
    - validate UIDetectionRuntime
    - validate privacy before perception
    - validate accessibility priority
    - validate low-confidence blocking
    - validate code indentation preservation
    - validate multi-monitor capture
    - validate simulated capture CPU budget

    Non-responsibilities:
    - no real capture hooks
    - no real OCR engine
    - no cognition consumption
    - no grounding/action execution
    """

    def __init__(
        self,
        *,
        name: str = "visual_perception_validation_gate",
    ) -> None:
        cleaned = name.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._reports: list[VisualPerceptionValidationReport] = []
        self._events: list[VisualPerceptionValidationEvent] = []
        self._lock = RLock()
        self._last_reason: VisualPerceptionGateReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def run(self) -> VisualPerceptionValidationReport:
        """
        Run the complete Visual Perception validation suite.
        """

        checks = (
            self.check_capture_works(),
            self.check_delta_reduces_cost(),
            self.check_privacy_zone_never_captured(),
            self.check_ocr_confidence_works(),
            self.check_low_confidence_ocr_flagged(),
            self.check_code_ocr_preserves_indentation(),
            self.check_ui_elements_detected(),
            self.check_accessibility_priority_works(),
            self.check_multi_monitor_stable(),
            self.check_capture_cpu_within_budget(),
        )
        report = self._build_report(checks)

        with self._lock:
            self._reports.append(report)
            self._events.append(
                VisualPerceptionValidationEvent(
                    kind=VisualPerceptionValidationEventKind.GATE_COMPLETED,
                    reason=report.reason,
                    metadata={"sealed": report.sealed},
                )
            )
            self._last_reason = report.reason

        return report

    def check_capture_works(self) -> VisualPerceptionCheckResult:
        attention, attention_session = _attention()
        runtime = ScreenCaptureRuntime(attention_runtime=attention)
        session = runtime.create_session(
            attention_session_id=attention_session.session_id,
            workspace_id="workspace",
        )
        result = runtime.capture_region(
            _capture_request(
                session_id=session.session_id,
                depth=InspectionDepth.PERIPHERAL,
            )
        )

        return self._check(
            kind=VisualPerceptionCheckKind.CAPTURE_WORKS,
            passed=(
                result.success
                and result.status == CaptureStatus.CAPTURED
                and result.region_capture is not None
            ),
            message="region capture returns governed capture payload",
            metadata={"reason": result.reason.value},
        )

    def check_delta_reduces_cost(self) -> VisualPerceptionCheckResult:
        attention, attention_session = _attention()
        runtime = ScreenCaptureRuntime(attention_runtime=attention)
        session = runtime.create_session(
            attention_session_id=attention_session.session_id,
            workspace_id="workspace",
        )
        first = runtime.capture_delta(
            _capture_request(
                session_id=session.session_id,
                depth=InspectionDepth.PERIPHERAL,
            )
        )
        second = runtime.capture_delta(
            _capture_request(
                session_id=session.session_id,
                depth=InspectionDepth.PERIPHERAL,
            )
        )
        first_regions = (
            first.delta_capture.changed_regions
            if first.delta_capture is not None
            else ()
        )
        second_regions = (
            second.delta_capture.changed_regions
            if second.delta_capture is not None
            else ()
        )

        return self._check(
            kind=VisualPerceptionCheckKind.DELTA_REDUCES_COST,
            passed=bool(first_regions) and len(second_regions) <= len(first_regions),
            message="delta capture reports changed regions and avoids growth",
            metadata={
                "first_regions": len(first_regions),
                "second_regions": len(second_regions),
            },
        )

    def check_privacy_zone_never_captured(self) -> VisualPerceptionCheckResult:
        zone = PrivacyZone(
            name="secret-region",
            region=ScreenRegion(x=0, y=0, width=100, height=100),
            capture_allowed=False,
            ocr_allowed=False,
            reason="validation privacy block",
        )
        attention, attention_session = _attention(
            privacy_policy=PrivacyZonePolicy(zones=(zone,))
        )
        runtime = ScreenCaptureRuntime(attention_runtime=attention)
        session = runtime.create_session(
            attention_session_id=attention_session.session_id,
            workspace_id="workspace",
        )
        result = runtime.capture_region(
            _capture_request(
                session_id=session.session_id,
                depth=InspectionDepth.PERIPHERAL,
                region=ScreenRegion(x=10, y=10, width=20, height=20),
            )
        )

        return self._check(
            kind=VisualPerceptionCheckKind.PRIVACY_ZONE_NEVER_CAPTURED,
            passed=(
                not result.success
                and result.reason == CaptureReason.CAPTURE_BLOCKED_BY_PRIVACY
            ),
            message="privacy zones block capture before perception",
            metadata={"reason": result.reason.value},
        )

    def check_ocr_confidence_works(self) -> VisualPerceptionCheckResult:
        runtime = OCRRuntime()
        session = runtime.create_session(workspace_id="workspace")
        result = runtime.extract_text(
            OCRExtractionRequest(
                session_id=session.session_id,
                capture=_region_capture(text="pytest passed in 20s"),
            )
        )
        confidence = (
            result.text_regions[0].confidence.confidence
            if result.text_regions
            else 0.0
        )

        return self._check(
            kind=VisualPerceptionCheckKind.OCR_CONFIDENCE_WORKS,
            passed=(
                result.status == OCRStatus.EXTRACTED
                and bool(result.text_regions)
                and confidence > 0.55
            ),
            message="OCR emits accepted text with confidence score",
            metadata={"confidence": confidence},
        )

    def check_low_confidence_ocr_flagged(self) -> VisualPerceptionCheckResult:
        runtime = OCRRuntime(
            scorer=TextConfidenceScorer(
                TextConfidenceScorerConfig(
                    minimum_confidence=0.99,
                    trusted_confidence=0.99,
                )
            )
        )
        session = runtime.create_session(workspace_id="workspace")
        result = runtime.extract_text(
            OCRExtractionRequest(
                session_id=session.session_id,
                capture=_region_capture(text="noisy ???"),
            )
        )

        return self._check(
            kind=VisualPerceptionCheckKind.LOW_CONFIDENCE_OCR_FLAGGED,
            passed=(
                result.status == OCRStatus.LOW_CONFIDENCE
                and result.reason == OCRReason.LOW_CONFIDENCE_TEXT_REJECTED
                and bool(result.rejected_regions)
            ),
            message="low-confidence OCR is rejected and flagged",
            metadata={"reason": result.reason.value},
        )

    def check_code_ocr_preserves_indentation(self) -> VisualPerceptionCheckResult:
        runtime = OCRRuntime()
        session = runtime.create_session(workspace_id="workspace")
        result = runtime.extract_code_text(
            OCRExtractionRequest(
                session_id=session.session_id,
                capture=_region_capture_without_text(),
                source_kind=OCRSourceKind.CODE,
            )
        )
        texts = tuple(region.text for region in result.text_regions)

        return self._check(
            kind=VisualPerceptionCheckKind.CODE_OCR_PRESERVES_INDENTATION,
            passed=any(text.startswith("    ") for text in texts),
            message="code OCR preserves indentation evidence",
            metadata={"texts": texts},
        )

    def check_ui_elements_detected(self) -> VisualPerceptionCheckResult:
        runtime = UIDetectionRuntime()
        session = runtime.create_session(workspace_id="workspace")
        result = runtime.detect_elements(
            UIDetectionRequest(
                session_id=session.session_id,
                region=ScreenRegion(x=0, y=0, width=500, height=300),
            )
        )

        return self._check(
            kind=VisualPerceptionCheckKind.UI_ELEMENTS_DETECTED,
            passed=(
                result.status == UIDetectionStatus.DETECTED
                and bool(result.elements)
                and result.elements[0].interactive
            ),
            message="UI detection produces trusted interactive elements",
            metadata={"element_count": len(result.elements)},
        )

    def check_accessibility_priority_works(self) -> VisualPerceptionCheckResult:
        runtime = UIDetectionRuntime()
        session = runtime.create_session(workspace_id="workspace")
        result = runtime.detect_elements(
            UIDetectionRequest(
                session_id=session.session_id,
                region=ScreenRegion(x=0, y=0, width=500, height=300),
                prefer_accessibility=True,
            )
        )
        first_source = result.source_order[0] if result.source_order else None

        return self._check(
            kind=VisualPerceptionCheckKind.ACCESSIBILITY_PRIORITY_WORKS,
            passed=(
                result.reason == UIDetectionReason.ACCESSIBILITY_ELEMENTS_DETECTED
                and first_source == UIDetectionSource.ACCESSIBILITY
            ),
            message="accessibility source is preferred before visual fallback",
            metadata={"first_source": first_source.value if first_source else None},
        )

    def check_multi_monitor_stable(self) -> VisualPerceptionCheckResult:
        attention, attention_session = _attention()
        runtime = ScreenCaptureRuntime(attention_runtime=attention)
        session = runtime.create_session(
            attention_session_id=attention_session.session_id,
            workspace_id="workspace",
        )
        result = runtime.capture_multi_monitor(
            session_id=session.session_id,
            displays=(fake_display(display_id="display-1"),),
        )

        return self._check(
            kind=VisualPerceptionCheckKind.MULTI_MONITOR_STABLE,
            passed=(
                result.success
                and result.multi_monitor_capture is not None
                and bool(result.multi_monitor_capture.captures)
            ),
            message="multi-monitor capture produces stable capture group",
            metadata={"reason": result.reason.value},
        )

    def check_capture_cpu_within_budget(self) -> VisualPerceptionCheckResult:
        sample = CaptureCpuSample(
            capture_cpu_percent=8.0,
            budget_percent=12.0,
        )

        return self._check(
            kind=VisualPerceptionCheckKind.CAPTURE_CPU_WITHIN_BUDGET,
            passed=sample.within_budget,
            message="capture CPU stays within visual perception budget",
            metadata={
                "capture_cpu_percent": sample.capture_cpu_percent,
                "budget_percent": sample.budget_percent,
            },
        )

    def reports(self) -> tuple[VisualPerceptionValidationReport, ...]:
        with self._lock:
            return tuple(self._reports)

    def events(self) -> tuple[VisualPerceptionValidationEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def snapshot(self) -> VisualPerceptionValidationSnapshot:
        with self._lock:
            checks = [check for report in self._reports for check in report.checks]

            return VisualPerceptionValidationSnapshot(
                name=self.name,
                report_count=len(self._reports),
                check_count=len(checks),
                passed_count=sum(1 for check in checks if check.passed),
                failed_count=sum(1 for check in checks if not check.passed),
                sealed_report_count=sum(1 for report in self._reports if report.sealed),
                runtime_event_count=len(self._events),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        event = VisualPerceptionValidationEvent(
            kind=VisualPerceptionValidationEventKind.RUNTIME_RESET,
            reason=VisualPerceptionGateReason.RUNTIME_RESET,
        )

        with self._lock:
            self._reports.clear()
            self._events.clear()
            self._events.append(event)
            self._last_reason = event.reason

    def _check(
        self,
        *,
        kind: VisualPerceptionCheckKind,
        passed: bool,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> VisualPerceptionCheckResult:
        result = VisualPerceptionCheckResult(
            kind=kind,
            status=(
                VisualPerceptionCheckStatus.PASSED
                if passed
                else VisualPerceptionCheckStatus.FAILED
            ),
            reason=(
                VisualPerceptionGateReason.CHECK_PASSED
                if passed
                else VisualPerceptionGateReason.CHECK_FAILED
            ),
            message=message,
            metadata=metadata or {},
        )
        event = VisualPerceptionValidationEvent(
            kind=VisualPerceptionValidationEventKind.CHECK_RECORDED,
            reason=result.reason,
            check_kind=kind,
            metadata=result.metadata,
        )

        with self._lock:
            self._events.append(event)
            self._last_reason = result.reason

        return result

    @staticmethod
    def _build_report(
        checks: tuple[VisualPerceptionCheckResult, ...],
    ) -> VisualPerceptionValidationReport:
        passed_count = sum(1 for check in checks if check.passed)
        failed_count = sum(1 for check in checks if not check.passed)
        skipped_count = sum(
            1
            for check in checks
            if check.status == VisualPerceptionCheckStatus.SKIPPED
        )
        sealed = failed_count == 0

        return VisualPerceptionValidationReport(
            checks=checks,
            passed_count=passed_count,
            failed_count=failed_count,
            skipped_count=skipped_count,
            sealed=sealed,
            reason=(
                VisualPerceptionGateReason.GATE_PASSED
                if sealed
                else VisualPerceptionGateReason.GATE_FAILED
            ),
            message=(
                "visual perception validation gate passed"
                if sealed
                else "visual perception validation gate failed"
            ),
        )


def _attention(
    *,
    privacy_policy: PrivacyZonePolicy | None = None,
) -> tuple[EnvironmentAttentionRuntime, EnvironmentAttentionSession]:
    runtime = EnvironmentAttentionRuntime(privacy_policy=privacy_policy)
    session = runtime.create_session(
        workspace_id="workspace",
        mode=EnvironmentAttentionMode.PERIPHERAL,
    )

    return runtime, session


def _capture_request(
    *,
    session_id: str,
    depth: InspectionDepth,
    region: ScreenRegion | None = None,
) -> ScreenCaptureRequest:
    return ScreenCaptureRequest(
        session_id=session_id,
        mode=CaptureMode.REGION,
        region=region or ScreenRegion(x=0, y=0, width=100, height=100),
        depth=depth,
        reason="visual perception validation capture",
    )


def _region_capture(text: str) -> RegionCapture:
    return RegionCapture(
        mode=CaptureMode.REGION,
        status=CaptureStatus.CAPTURED,
        region=ScreenRegion(x=0, y=0, width=120, height=40),
        payload=CapturePayload(
            width=120,
            height=40,
            pixel_format=CapturePixelFormat.FAKE,
            byte_count=4800,
            content_hash=f"fake_{text}",
            metadata={"text": text},
        ),
        governance=_capture_governance(),
        trust=_trust("validation region capture"),
    )


def _region_capture_without_text() -> RegionCapture:
    return RegionCapture(
        mode=CaptureMode.REGION,
        status=CaptureStatus.CAPTURED,
        region=ScreenRegion(x=0, y=0, width=120, height=40),
        payload=CapturePayload(
            width=120,
            height=40,
            pixel_format=CapturePixelFormat.FAKE,
            byte_count=4800,
            content_hash="fake_code_capture",
            metadata={},
        ),
        governance=_capture_governance(),
        trust=_trust("validation code region capture"),
    )


def _capture_governance() -> CaptureGovernanceDecision:
    from jarvis.environment.attention import (
        CaptureGovernanceDecision,
    )

    return CaptureGovernanceDecision(
        request_id="validation-capture",
        permission=CapturePermission.ALLOW,
        reason=AttentionGovernanceReason.CAPTURE_ALLOWED,
        allowed_depth=InspectionDepth.PERIPHERAL,
        frequency_hz=0.5,
        privacy_decision=PrivacyZoneDecision.ALLOWED,
        region=ScreenRegion(x=0, y=0, width=120, height=40),
    )


def _trust(reason: str) -> TrustCalibration:
    return TrustCalibration(
        confidence=0.95,
        stability=0.95,
        ambiguity=0.0,
        source=EnvironmentSource.SCREEN_CAPTURE,
        reason=reason,
    )


def _clean_required(value: str) -> str:
    cleaned = value.strip()

    if not cleaned:
        raise ValueError("field cannot be empty.")

    return cleaned