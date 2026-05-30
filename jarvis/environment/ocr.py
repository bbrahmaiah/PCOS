from __future__ import annotations

from enum import StrEnum
from threading import RLock
from typing import Any, Protocol
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.environment.capture import (
    CapturePayload,
    CaptureStatus,
    RegionCapture,
)
from jarvis.environment.models import (
    EnvironmentSource,
    PrivacyClassification,
    ScreenRegion,
    TextRegion,
    TextRegionKind,
    TrustCalibration,
)
from jarvis.environment.trust_runtime import (
    TrustCalibrationRuntime,
    TrustPolicyClassification,
    TrustSubjectKind,
)
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class OCRSourceKind(StrEnum):
    """
    OCR/text extraction source.

    Different sources need different confidence policies.
    """

    GENERIC_OCR = "generic_ocr"
    CODE = "code"
    TERMINAL = "terminal"
    BROWSER = "browser"
    DOCUMENT = "document"


class OCRStatus(StrEnum):
    """
    OCR operation status.
    """

    REQUESTED = "requested"
    EXTRACTED = "extracted"
    LOW_CONFIDENCE = "low_confidence"
    PRIVACY_BLOCKED = "privacy_blocked"
    CAPTURE_INVALID = "capture_invalid"
    FAILED = "failed"


class OCRReason(StrEnum):
    """
    Machine-readable OCR runtime reason.
    """

    SESSION_CREATED = "session_created"
    OCR_EXTRACTED = "ocr_extracted"
    CODE_TEXT_EXTRACTED = "code_text_extracted"
    TERMINAL_TEXT_EXTRACTED = "terminal_text_extracted"
    BROWSER_TEXT_EXTRACTED = "browser_text_extracted"
    DOCUMENT_TEXT_EXTRACTED = "document_text_extracted"
    LOW_CONFIDENCE_TEXT_REJECTED = "low_confidence_text_rejected"
    PRIVACY_BLOCKED = "privacy_blocked"
    CAPTURE_INVALID = "capture_invalid"
    SESSION_NOT_FOUND = "session_not_found"
    RUNTIME_RESET = "runtime_reset"


class OCRRuntimeEventKind(StrEnum):
    """
    OCR runtime event kind.
    """

    SESSION_CREATED = "session_created"
    OCR_REQUESTED = "ocr_requested"
    OCR_COMPLETED = "ocr_completed"
    OCR_BLOCKED = "ocr_blocked"
    OCR_FAILED = "ocr_failed"
    RUNTIME_RESET = "runtime_reset"


class OCRTextKind(StrEnum):
    """
    Higher-level extracted text kind.
    """

    UNKNOWN = "unknown"
    PLAIN_TEXT = "plain_text"
    CODE = "code"
    TERMINAL = "terminal"
    BROWSER = "browser"
    DOCUMENT = "document"
    ERROR = "error"
    COMMAND = "command"


class TextConfidenceScorerConfig(OrchestrationModel):
    """
    Confidence scoring thresholds.
    """

    minimum_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    trusted_confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    code_bonus: float = Field(default=0.04, ge=0.0, le=0.2)
    terminal_bonus: float = Field(default=0.03, ge=0.0, le=0.2)
    document_bonus: float = Field(default=0.02, ge=0.0, le=0.2)
    ambiguity_penalty: float = Field(default=0.15, ge=0.0, le=0.5)

    @model_validator(mode="after")
    def _trusted_must_exceed_minimum(self) -> TextConfidenceScorerConfig:
        if self.trusted_confidence < self.minimum_confidence:
            raise ValueError("trusted_confidence must be >= minimum_confidence.")

        return self


class TextConfidenceScore(OrchestrationModel):
    """
    Confidence score for extracted text.
    """

    score_id: str = Field(default_factory=lambda: f"text_score_{uuid4().hex}")
    confidence: float = Field(ge=0.0, le=1.0)
    stability: float = Field(ge=0.0, le=1.0)
    ambiguity: float = Field(ge=0.0, le=1.0)
    source_kind: OCRSourceKind
    accepted: bool
    reason: str
    created_at: object = Field(default_factory=utc_now)

    @field_validator("score_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class OCRTextRegion(OrchestrationModel):
    """
    OCR output contract.

    This wraps the Step 0 TextRegion contract with source, confidence,
    privacy, and policy metadata.
    """

    text_region_id: str = Field(default_factory=lambda: f"ocr_text_{uuid4().hex}")
    text: str
    bounds: ScreenRegion
    kind: OCRTextKind
    source_kind: OCRSourceKind
    source: EnvironmentSource
    confidence: TextConfidenceScore
    privacy: PrivacyClassification
    trust: TrustCalibration
    policy_classification: TrustPolicyClassification
    capture_id: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text_region_id", "text", "capture_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    def to_environment_text_region(self) -> TextRegion:
        return TextRegion(
            text=self.text,
            kind=_text_region_kind_for(self.kind),
            bounds=self.bounds,
            trust=self.trust,
        )


class OCRExtractionRequest(OrchestrationModel):
    """
    Request to extract text from a governed capture.
    """

    request_id: str = Field(default_factory=lambda: f"ocr_request_{uuid4().hex}")
    session_id: str
    capture: RegionCapture
    source_kind: OCRSourceKind = OCRSourceKind.GENERIC_OCR
    expected_kind: OCRTextKind = OCRTextKind.UNKNOWN
    privacy: PrivacyClassification = PrivacyClassification.WORKSPACE
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("request_id", "session_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class OCRExtractionResult(OrchestrationModel):
    """
    Result of one OCR operation.
    """

    result_id: str = Field(default_factory=lambda: f"ocr_result_{uuid4().hex}")
    status: OCRStatus
    reason: OCRReason
    request_id: str
    capture_id: str | None = None
    text_regions: tuple[OCRTextRegion, ...] = ()
    rejected_regions: tuple[OCRTextRegion, ...] = ()
    message: str
    created_at: object = Field(default_factory=utc_now)

    @field_validator("result_id", "request_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class OCRRuntimeEvent(OrchestrationModel):
    """
    OCR runtime event.
    """

    event_id: str = Field(default_factory=lambda: f"ocr_event_{uuid4().hex}")
    kind: OCRRuntimeEventKind
    reason: OCRReason
    session_id: str | None = None
    request_id: str | None = None
    capture_id: str | None = None
    text_region_count: int = Field(default=0, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class OCRSession(OrchestrationModel):
    """
    OCR runtime session.
    """

    session_id: str = Field(default_factory=lambda: f"ocr_session_{uuid4().hex}")
    workspace_id: str
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class OCRRuntimeSnapshot(OrchestrationModel):
    """
    Diagnostics for Phase 8 Step 9.
    """

    name: str
    session_count: int = Field(ge=0)
    extraction_count: int = Field(ge=0)
    accepted_text_region_count: int = Field(ge=0)
    rejected_text_region_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    runtime_event_count: int = Field(ge=0)
    last_reason: OCRReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        return _clean_required(value)


class OCRAdapter(Protocol):
    """
    Adapter protocol for OCR engines.

    Real OCR engines can plug in later. Step 9 remains fake-first.
    """

    def extract(
        self,
        *,
        payload: CapturePayload,
        source_kind: OCRSourceKind,
    ) -> tuple[str, ...]:
        ...


class FakeOCRAdapter:
    """
    Fake OCR adapter for deterministic tests.
    """

    def extract(
        self,
        *,
        payload: CapturePayload,
        source_kind: OCRSourceKind,
    ) -> tuple[str, ...]:
        text = payload.metadata.get("text")

        if isinstance(text, str) and text.strip():
            return (text,)

        if source_kind == OCRSourceKind.CODE:
            return ("def main() -> None:", "    print('jarvis')")

        if source_kind == OCRSourceKind.TERMINAL:
            return ("pytest passed in 20s",)

        if source_kind == OCRSourceKind.BROWSER:
            return ("Documentation - JARVIS OS",)

        if source_kind == OCRSourceKind.DOCUMENT:
            return ("Phase 8 visual perception runtime",)

        return ("visible text",)


class TextConfidenceScorer:
    """
    Scores extracted text before it becomes data.
    """

    def __init__(
        self,
        config: TextConfidenceScorerConfig | None = None,
    ) -> None:
        self._config = config or TextConfidenceScorerConfig()

    @property
    def config(self) -> TextConfidenceScorerConfig:
        return self._config

    def score(
        self,
        *,
        text: str,
        source_kind: OCRSourceKind,
        privacy: PrivacyClassification,
    ) -> TextConfidenceScore:
        cleaned = text.strip()
        base = 0.72

        if len(cleaned) >= 12:
            base += 0.08

        if any(char.isdigit() for char in cleaned):
            base += 0.03

        if source_kind == OCRSourceKind.CODE:
            base += self._config.code_bonus

        if source_kind == OCRSourceKind.TERMINAL:
            base += self._config.terminal_bonus

        if source_kind == OCRSourceKind.DOCUMENT:
            base += self._config.document_bonus

        ambiguity = self._ambiguity_for(cleaned)

        if privacy in {
            PrivacyClassification.SECRET,
            PrivacyClassification.BLOCKED,
        }:
            base = 0.0

        confidence = max(0.0, min(1.0, base - (ambiguity * 0.25)))
        accepted = confidence >= self._config.minimum_confidence

        return TextConfidenceScore(
            confidence=confidence,
            stability=max(0.0, min(1.0, confidence + 0.05)),
            ambiguity=ambiguity,
            source_kind=source_kind,
            accepted=accepted,
            reason="text confidence scored",
        )

    @staticmethod
    def _ambiguity_for(text: str) -> float:
        if not text:
            return 1.0

        noisy = sum(1 for char in text if char in {" ", "?", "|", "~"})
        ratio = noisy / max(1, len(text))

        return max(0.0, min(1.0, ratio * 3.0))


class CodeTextExtractor:
    source_kind = OCRSourceKind.CODE

    def extract(
        self,
        runtime: OCRRuntime,
        request: OCRExtractionRequest
        ) -> OCRExtractionResult:
        return runtime.extract_text(
            request.model_copy(update={"source_kind": OCRSourceKind.CODE})
        )


class TerminalTextExtractor:
    source_kind = OCRSourceKind.TERMINAL

    def extract(
        self,
        runtime: OCRRuntime,
        request: OCRExtractionRequest
        ) -> OCRExtractionResult:
        return runtime.extract_text(
            request.model_copy(update={"source_kind": OCRSourceKind.TERMINAL})
        )


class BrowserTextExtractor:
    source_kind = OCRSourceKind.BROWSER

    def extract(
        self,
        runtime: OCRRuntime,
        request: OCRExtractionRequest
        ) -> OCRExtractionResult:
        return runtime.extract_text(
            request.model_copy(update={"source_kind": OCRSourceKind.BROWSER})
        )


class DocumentTextExtractor:
    source_kind = OCRSourceKind.DOCUMENT

    def extract(
        self,
        runtime: OCRRuntime,
        request: OCRExtractionRequest
        ) -> OCRExtractionResult:
        return runtime.extract_text(
            request.model_copy(update={"source_kind": OCRSourceKind.DOCUMENT})
        )


class OCRRuntime:
    """
    Phase 8 Step 9 OCR & Text Perception Runtime.

    Responsibilities:
    - extract text from governed captures
    - preserve text bounds
    - score confidence
    - classify source/kind/privacy
    - calibrate trust
    - reject low-confidence or blocked text

    Non-responsibilities:
    - no screen capture
    - no UI element detection
    - no semantic UI grounding
    - no action execution
    """

    def __init__(
        self,
        *,
        name: str = "ocr_runtime",
        trust_runtime: TrustCalibrationRuntime | None = None,
        adapter: OCRAdapter | None = None,
        scorer: TextConfidenceScorer | None = None,
    ) -> None:
        cleaned = name.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._trust_runtime = trust_runtime or TrustCalibrationRuntime()
        self._adapter = adapter or FakeOCRAdapter()
        self._scorer = scorer or TextConfidenceScorer()
        self._sessions: dict[str, OCRSession] = {}
        self._results: list[OCRExtractionResult] = []
        self._events: list[OCRRuntimeEvent] = []
        self._lock = RLock()
        self._last_reason: OCRReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def create_session(
        self,
        *,
        workspace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> OCRSession:
        session = OCRSession(workspace_id=workspace_id, metadata=metadata or {})
        event = self._event(
            kind=OCRRuntimeEventKind.SESSION_CREATED,
            reason=OCRReason.SESSION_CREATED,
            session_id=session.session_id,
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._events.append(event)
            self._last_reason = event.reason

        return session

    def extract_text(
        self,
        request: OCRExtractionRequest,
    ) -> OCRExtractionResult:
        session = self.session_for(request.session_id)

        if session is None:
            return self._result(
                status=OCRStatus.FAILED,
                reason=OCRReason.SESSION_NOT_FOUND,
                request=request,
                message="ocr session not found",
            )

        if request.capture.status not in {
            CaptureStatus.CAPTURED,
            CaptureStatus.LIMITED,
        }:
            return self._result(
                status=OCRStatus.CAPTURE_INVALID,
                reason=OCRReason.CAPTURE_INVALID,
                request=request,
                message="capture is not valid for OCR",
            )

        if request.capture.payload is None:
            return self._result(
                status=OCRStatus.CAPTURE_INVALID,
                reason=OCRReason.CAPTURE_INVALID,
                request=request,
                message="capture has no payload",
            )

        if request.privacy in {
            PrivacyClassification.SECRET,
            PrivacyClassification.BLOCKED,
        }:
            return self._result(
                status=OCRStatus.PRIVACY_BLOCKED,
                reason=OCRReason.PRIVACY_BLOCKED,
                request=request,
                message="privacy policy blocks OCR text extraction",
            )

        raw_lines = self._adapter.extract(
            payload=request.capture.payload,
            source_kind=request.source_kind,
        )
        accepted: list[OCRTextRegion] = []
        rejected: list[OCRTextRegion] = []

        for index, raw_text in enumerate(raw_lines):
            score = self._scorer.score(
                text=raw_text,
                source_kind=request.source_kind,
                privacy=request.privacy,
            )
            region = self._region_for_line(
                base=request.capture.region,
                index=index,
                total=max(1, len(raw_lines)),
            )
            text_kind = self._kind_for(
                expected=request.expected_kind,
                source_kind=request.source_kind,
                text=raw_text,
            )
            observation = self._trust_runtime.calibrate_observation(
                subject_id=f"{request.capture.capture_id}:{index}",
                subject_kind=TrustSubjectKind.OCR_TEXT,
                source=EnvironmentSource.OCR,
                confidence=score.confidence,
                stability=score.stability,
                ambiguity=score.ambiguity,
                privacy=request.privacy,
                reason="OCR text region calibrated",
                metadata={"source_kind": request.source_kind.value},
            )
            text_region = OCRTextRegion(
                text=raw_text,
                bounds=region,
                kind=text_kind,
                source_kind=request.source_kind,
                source=EnvironmentSource.OCR,
                confidence=score,
                privacy=request.privacy,
                trust=observation.calibration,
                policy_classification=observation.policy_classification,
                capture_id=request.capture.capture_id,
            )

            if score.accepted:
                accepted.append(text_region)
            else:
                rejected.append(text_region)

        if not accepted:
            return self._result(
                status=OCRStatus.LOW_CONFIDENCE,
                reason=OCRReason.LOW_CONFIDENCE_TEXT_REJECTED,
                request=request,
                text_regions=(),
                rejected_regions=tuple(rejected),
                message="all OCR text regions rejected by confidence scorer",
            )

        reason = self._reason_for_source(request.source_kind)
        return self._result(
            status=OCRStatus.EXTRACTED,
            reason=reason,
            request=request,
            text_regions=tuple(accepted),
            rejected_regions=tuple(rejected),
            message="OCR text extracted",
        )

    def extract_code_text(
        self,
        request: OCRExtractionRequest,
    ) -> OCRExtractionResult:
        return self.extract_text(
            request.model_copy(update={"source_kind": OCRSourceKind.CODE})
        )

    def extract_terminal_text(
        self,
        request: OCRExtractionRequest,
    ) -> OCRExtractionResult:
        return self.extract_text(
            request.model_copy(update={"source_kind": OCRSourceKind.TERMINAL})
        )

    def extract_browser_text(
        self,
        request: OCRExtractionRequest,
    ) -> OCRExtractionResult:
        return self.extract_text(
            request.model_copy(update={"source_kind": OCRSourceKind.BROWSER})
        )

    def extract_document_text(
        self,
        request: OCRExtractionRequest,
    ) -> OCRExtractionResult:
        return self.extract_text(
            request.model_copy(update={"source_kind": OCRSourceKind.DOCUMENT})
        )

    def session_for(self, session_id: str) -> OCRSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def results(self) -> tuple[OCRExtractionResult, ...]:
        with self._lock:
            return tuple(self._results)

    def events(self) -> tuple[OCRRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def snapshot(self) -> OCRRuntimeSnapshot:
        with self._lock:
            accepted_count = sum(len(result.text_regions) for result in self._results)
            rejected_count = sum(
                len(result.rejected_regions) for result in self._results
            )

            return OCRRuntimeSnapshot(
                name=self.name,
                session_count=len(self._sessions),
                extraction_count=len(self._results),
                accepted_text_region_count=accepted_count,
                rejected_text_region_count=rejected_count,
                blocked_count=sum(
                    1
                    for result in self._results
                    if result.status == OCRStatus.PRIVACY_BLOCKED
                ),
                failed_count=sum(
                    1
                    for result in self._results
                    if result.status == OCRStatus.FAILED
                ),
                runtime_event_count=len(self._events),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        event = self._event(
            kind=OCRRuntimeEventKind.RUNTIME_RESET,
            reason=OCRReason.RUNTIME_RESET,
        )

        with self._lock:
            self._sessions.clear()
            self._results.clear()
            self._events.clear()
            self._events.append(event)
            self._last_reason = event.reason

    def _result(
        self,
        *,
        status: OCRStatus,
        reason: OCRReason,
        request: OCRExtractionRequest,
        message: str,
        text_regions: tuple[OCRTextRegion, ...] = (),
        rejected_regions: tuple[OCRTextRegion, ...] = (),
    ) -> OCRExtractionResult:
        result = OCRExtractionResult(
            status=status,
            reason=reason,
            request_id=request.request_id,
            capture_id=request.capture.capture_id,
            text_regions=text_regions,
            rejected_regions=rejected_regions,
            message=message,
        )
        event = self._event(
            kind=(
                OCRRuntimeEventKind.OCR_COMPLETED
                if status == OCRStatus.EXTRACTED
                else OCRRuntimeEventKind.OCR_BLOCKED
            ),
            reason=reason,
            session_id=request.session_id,
            request_id=request.request_id,
            capture_id=request.capture.capture_id,
            text_region_count=len(text_regions),
        )

        with self._lock:
            self._results.append(result)
            self._events.append(event)
            self._touch_session(request.session_id)
            self._last_reason = reason

        return result

    def _touch_session(self, session_id: str) -> None:
        session = self._sessions.get(session_id)

        if session is None:
            return

        self._sessions[session_id] = session.model_copy(
            update={"updated_at": utc_now()}
        )

    @staticmethod
    def _region_for_line(
        *,
        base: ScreenRegion,
        index: int,
        total: int,
    ) -> ScreenRegion:
        line_height = max(1, base.height // max(1, total))

        return ScreenRegion(
            x=base.x,
            y=base.y + (line_height * index),
            width=base.width,
            height=line_height,
            display_id=base.display_id,
            scale_factor=base.scale_factor,
        )

    @staticmethod
    def _kind_for(
        *,
        expected: OCRTextKind,
        source_kind: OCRSourceKind,
        text: str,
    ) -> OCRTextKind:
        if expected != OCRTextKind.UNKNOWN:
            return expected

        lowered = text.lower()

        if source_kind == OCRSourceKind.CODE:
            return OCRTextKind.CODE

        if source_kind == OCRSourceKind.TERMINAL:
            if "error" in lowered or "failed" in lowered:
                return OCRTextKind.ERROR

            return OCRTextKind.TERMINAL

        if source_kind == OCRSourceKind.BROWSER:
            return OCRTextKind.BROWSER

        if source_kind == OCRSourceKind.DOCUMENT:
            return OCRTextKind.DOCUMENT

        return OCRTextKind.PLAIN_TEXT

    @staticmethod
    def _reason_for_source(source_kind: OCRSourceKind) -> OCRReason:
        if source_kind == OCRSourceKind.CODE:
            return OCRReason.CODE_TEXT_EXTRACTED

        if source_kind == OCRSourceKind.TERMINAL:
            return OCRReason.TERMINAL_TEXT_EXTRACTED

        if source_kind == OCRSourceKind.BROWSER:
            return OCRReason.BROWSER_TEXT_EXTRACTED

        if source_kind == OCRSourceKind.DOCUMENT:
            return OCRReason.DOCUMENT_TEXT_EXTRACTED

        return OCRReason.OCR_EXTRACTED

    @staticmethod
    def _event(
        *,
        kind: OCRRuntimeEventKind,
        reason: OCRReason,
        session_id: str | None = None,
        request_id: str | None = None,
        capture_id: str | None = None,
        text_region_count: int = 0,
    ) -> OCRRuntimeEvent:
        return OCRRuntimeEvent(
            kind=kind,
            reason=reason,
            session_id=session_id,
            request_id=request_id,
            capture_id=capture_id,
            text_region_count=text_region_count,
        )


def _text_region_kind_for(kind: OCRTextKind) -> TextRegionKind:
    if kind == OCRTextKind.CODE:
        return TextRegionKind.CODE

    if kind == OCRTextKind.TERMINAL:
        return TextRegionKind.TERMINAL

    if kind == OCRTextKind.ERROR:
        return TextRegionKind.ERROR

    return TextRegionKind.UNKNOWN


def _clean_required(value: str) -> str:
    cleaned = value.strip()

    if not cleaned:
        raise ValueError("field cannot be empty.")

    return cleaned