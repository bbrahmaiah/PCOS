from __future__ import annotations

import re
from enum import StrEnum
from hashlib import sha256
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.environment.models import EnvironmentSource, TrustCalibration
from jarvis.environment.trust_runtime import TrustPolicyClassification
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class ClipboardOperationKind(StrEnum):
    READ_HASH = "read_hash"
    PREPARE_PASTE = "prepare_paste"
    VERIFY_BEFORE_PASTE = "verify_before_paste"
    VERIFY_AFTER_PASTE = "verify_after_paste"
    CLEAR_CLIPBOARD = "clear_clipboard"


class ClipboardStatus(StrEnum):
    READY = "ready"
    VERIFIED = "verified"
    NEEDS_APPROVAL = "needs_approval"
    BLOCKED = "blocked"
    HIJACK_DETECTED = "hijack_detected"
    FAILED = "failed"


class ClipboardDecision(StrEnum):
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    BLOCK = "block"
    ABORT = "abort"


class ClipboardReason(StrEnum):
    SESSION_CREATED = "session_created"
    HASH_RECORDED = "hash_recorded"
    PASTE_READY = "paste_ready"
    SENSITIVE_CONTENT_BLOCKED = "sensitive_content_blocked"
    SENSITIVE_CONTENT_REQUIRES_APPROVAL = (
        "sensitive_content_requires_approval"
    )
    UNKNOWN_FIELD_BLOCKED = "unknown_field_blocked"
    FOCUS_UNCERTAIN_BLOCKED = "focus_uncertain_blocked"
    CLIPBOARD_HASH_VERIFIED = "clipboard_hash_verified"
    CLIPBOARD_HIJACK_DETECTED = "clipboard_hijack_detected"
    EMPTY_CLIPBOARD_BLOCKED = "empty_clipboard_blocked"
    SESSION_NOT_FOUND = "session_not_found"
    RUNTIME_RESET = "runtime_reset"


class ClipboardEventKind(StrEnum):
    SESSION_CREATED = "session_created"
    OPERATION_ALLOWED = "operation_allowed"
    OPERATION_BLOCKED = "operation_blocked"
    HIJACK_DETECTED = "hijack_detected"
    RUNTIME_RESET = "runtime_reset"


class ClipboardSensitivityKind(StrEnum):
    NONE = "none"
    PASSWORD = "password"
    API_KEY = "api_key"
    TOKEN = "token"
    PRIVATE_KEY = "private_key"
    SECRET_ASSIGNMENT = "secret_assignment"
    COOKIE = "cookie"
    EMAIL = "email"
    POSSIBLE_CREDENTIAL = "possible_credential"


class ClipboardSensitivityLevel(StrEnum):
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ClipboardHijackStatus(StrEnum):
    NOT_CHECKED = "not_checked"
    CLEAN = "clean"
    HIJACKED = "hijacked"


class ClipboardHashPhase(StrEnum):
    BEFORE_PASTE = "before_paste"
    AFTER_PASTE = "after_paste"


class ClipboardHashRecord(OrchestrationModel):
    """
    Hash-only clipboard reference.

    Raw clipboard content must never be stored here.
    """

    record_id: str = Field(default_factory=lambda: f"clipboard_hash_{uuid4().hex}")
    content_hash: str
    content_length: int = Field(ge=0)
    algorithm: str = "sha256"
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("record_id", "content_hash", "algorithm")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @field_validator("content_hash")
    @classmethod
    def _sha256_length(cls, value: str) -> str:
        cleaned = value.strip()

        if len(cleaned) != 64:
            raise ValueError("content_hash must be a sha256 hex digest.")

        return cleaned


class ClipboardSensitivityFinding(OrchestrationModel):
    """
    Sensitivity finding without raw content.

    evidence_hash is a hash of the matched fragment, never the fragment itself.
    """

    finding_id: str = Field(default_factory=lambda: f"clipboard_find_{uuid4().hex}")
    kind: ClipboardSensitivityKind
    level: ClipboardSensitivityLevel
    evidence_hash: str | None = None
    confidence: float = Field(default=0.80, ge=0.0, le=1.0)
    reason: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("finding_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class ClipboardSensitivityScan(OrchestrationModel):
    scan_id: str = Field(default_factory=lambda: f"clipboard_scan_{uuid4().hex}")
    safe: bool
    highest_level: ClipboardSensitivityLevel
    findings: tuple[ClipboardSensitivityFinding, ...] = ()
    finding_count: int = Field(default=0, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("scan_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _count_matches_findings(self) -> ClipboardSensitivityScan:
        if self.finding_count != len(self.findings):
            raise ValueError("finding_count must match findings length.")

        return self


class ClipboardHijackReport(OrchestrationModel):
    report_id: str = Field(default_factory=lambda: f"clipboard_hijack_{uuid4().hex}")
    status: ClipboardHijackStatus
    phase: ClipboardHashPhase
    expected_hash: str
    observed_hash: str
    clean: bool
    reason: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("report_id", "expected_hash", "observed_hash", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class PasteSafetyDecision(OrchestrationModel):
    decision_id: str = Field(default_factory=lambda: f"paste_policy_{uuid4().hex}")
    decision: ClipboardDecision
    reason: ClipboardReason
    safe_to_paste: bool
    requires_approval: bool = False
    target_field_known: bool
    focus_known: bool
    sensitivity_level: ClipboardSensitivityLevel
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("decision_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _approval_not_safe_to_paste(self) -> PasteSafetyDecision:
        if self.requires_approval and self.safe_to_paste:
            raise ValueError("approval-required paste cannot be safe_to_paste.")

        return self


class ClipboardAuditRecord(OrchestrationModel):
    audit_id: str = Field(default_factory=lambda: f"clipboard_audit_{uuid4().hex}")
    operation: ClipboardOperationKind
    status: ClipboardStatus
    decision: ClipboardDecision
    reason: ClipboardReason
    content_hash: str | None = None
    finding_count: int = Field(default=0, ge=0)
    raw_content_logged: bool = False
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("audit_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _raw_content_never_logged(self) -> ClipboardAuditRecord:
        if self.raw_content_logged:
            raise ValueError("raw clipboard content must never be logged.")

        return self


class ClipboardOperationResult(OrchestrationModel):
    result_id: str = Field(default_factory=lambda: f"clipboard_result_{uuid4().hex}")
    operation: ClipboardOperationKind
    status: ClipboardStatus
    decision: ClipboardDecision
    reason: ClipboardReason
    hash_record: ClipboardHashRecord | None = None
    scan: ClipboardSensitivityScan | None = None
    paste_policy: PasteSafetyDecision | None = None
    hijack_report: ClipboardHijackReport | None = None
    audit: ClipboardAuditRecord
    trust: TrustCalibration
    safe_for_physical_paste: bool = False
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("result_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _safe_paste_requires_policy_and_hash(self) -> ClipboardOperationResult:
        if self.safe_for_physical_paste:
            if self.hash_record is None:
                raise ValueError("safe paste requires hash_record.")

            if self.paste_policy is None:
                raise ValueError("safe paste requires paste policy.")

            if not self.paste_policy.safe_to_paste:
                raise ValueError("safe paste requires safe paste policy.")

        return self


class ClipboardRuntimeSession(OrchestrationModel):
    session_id: str = Field(default_factory=lambda: f"clipboard_session_{uuid4().hex}")
    workspace_id: str
    operation_count: int = Field(default=0, ge=0)
    hijack_count: int = Field(default=0, ge=0)
    last_hash: str | None = None
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class ClipboardRuntimeEvent(OrchestrationModel):
    event_id: str = Field(default_factory=lambda: f"clipboard_event_{uuid4().hex}")
    kind: ClipboardEventKind
    reason: ClipboardReason
    session_id: str | None = None
    result_id: str | None = None
    audit_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class ClipboardRuntimeSnapshot(OrchestrationModel):
    name: str
    session_count: int = Field(ge=0)
    result_count: int = Field(ge=0)
    safe_paste_count: int = Field(ge=0)
    approval_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    hijack_count: int = Field(ge=0)
    audit_count: int = Field(ge=0)
    runtime_event_count: int = Field(ge=0)
    last_reason: ClipboardReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        return _clean_required(value)


class ClipboardHashVerifier:
    def hash_text(self, text: str) -> ClipboardHashRecord:
        digest = sha256(text.encode("utf-8")).hexdigest()

        return ClipboardHashRecord(
            content_hash=digest,
            content_length=len(text),
            metadata={"raw_content_logged": False},
        )

    def verify(
        self,
        *,
        expected_hash: str,
        observed_text: str,
        phase: ClipboardHashPhase,
    ) -> ClipboardHijackReport:
        observed = sha256(observed_text.encode("utf-8")).hexdigest()
        clean = expected_hash == observed

        return ClipboardHijackReport(
            status=(
                ClipboardHijackStatus.CLEAN
                if clean
                else ClipboardHijackStatus.HIJACKED
            ),
            phase=phase,
            expected_hash=expected_hash,
            observed_hash=observed,
            clean=clean,
            reason=(
                "clipboard hash verified"
                if clean
                else "clipboard hash changed unexpectedly"
            ),
        )


class ClipboardSensitivityScanner:
    """
    Scans raw clipboard text transiently.

    Raw content is never returned or stored.
    """

    _patterns: tuple[
        tuple[ClipboardSensitivityKind, ClipboardSensitivityLevel, re.Pattern[str]],
        ...,
    ] = (
        (
            ClipboardSensitivityKind.PRIVATE_KEY,
            ClipboardSensitivityLevel.CRITICAL,
            re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        ),
        (
            ClipboardSensitivityKind.API_KEY,
            ClipboardSensitivityLevel.HIGH,
            re.compile(r"(?i)\b(api[_-]?key|x-api-key)\b\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}"),
        ),
        (
            ClipboardSensitivityKind.TOKEN,
            ClipboardSensitivityLevel.HIGH,
            re.compile(r"(?i)\b(token|bearer)\b\s*[:=]?\s*['\"]?[A-Za-z0-9_\-.]{20,}"),
        ),
        (
            ClipboardSensitivityKind.PASSWORD,
            ClipboardSensitivityLevel.HIGH,
            re.compile(r"(?i)\b(password|passwd|pwd)\b\s*[:=]\s*['\"]?.{4,}"),
        ),
        (
            ClipboardSensitivityKind.SECRET_ASSIGNMENT,
            ClipboardSensitivityLevel.HIGH,
            re.compile(r"(?i)\b(secret|client_secret)\b\s*[:=]\s*['\"]?.{8,}"),
        ),
        (
            ClipboardSensitivityKind.COOKIE,
            ClipboardSensitivityLevel.MEDIUM,
            re.compile(r"(?i)\b(cookie|set-cookie|sessionid)\b\s*[:=]"),
        ),
        (
            ClipboardSensitivityKind.EMAIL,
            ClipboardSensitivityLevel.LOW,
            re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        ),
    )

    def scan(self, text: str) -> ClipboardSensitivityScan:
        findings: list[ClipboardSensitivityFinding] = []

        for kind, level, pattern in self._patterns:
            match = pattern.search(text)

            if match is None:
                continue

            fragment = match.group(0)
            findings.append(
                ClipboardSensitivityFinding(
                    kind=kind,
                    level=level,
                    evidence_hash=sha256(fragment.encode("utf-8")).hexdigest(),
                    confidence=_confidence_for_level(level),
                    reason=f"clipboard pattern detected: {kind.value}",
                )
            )

        highest = _highest_level(tuple(findings))

        return ClipboardSensitivityScan(
            safe=highest in {
                ClipboardSensitivityLevel.SAFE,
                ClipboardSensitivityLevel.LOW,
            },
            highest_level=highest,
            findings=tuple(findings),
            finding_count=len(findings),
            metadata={"raw_content_logged": False},
        )


class PasteSafetyPolicy:
    def evaluate(
        self,
        *,
        scan: ClipboardSensitivityScan,
        target_field_known: bool,
        focus_known: bool,
        allow_sensitive: bool,
    ) -> PasteSafetyDecision:
        if not target_field_known:
            return PasteSafetyDecision(
                decision=ClipboardDecision.BLOCK,
                reason=ClipboardReason.UNKNOWN_FIELD_BLOCKED,
                safe_to_paste=False,
                requires_approval=False,
                target_field_known=False,
                focus_known=focus_known,
                sensitivity_level=scan.highest_level,
            )

        if not focus_known:
            return PasteSafetyDecision(
                decision=ClipboardDecision.BLOCK,
                reason=ClipboardReason.FOCUS_UNCERTAIN_BLOCKED,
                safe_to_paste=False,
                requires_approval=False,
                target_field_known=True,
                focus_known=False,
                sensitivity_level=scan.highest_level,
            )

        if scan.highest_level in {
            ClipboardSensitivityLevel.HIGH,
            ClipboardSensitivityLevel.CRITICAL,
        }:
            if allow_sensitive:
                return PasteSafetyDecision(
                    decision=ClipboardDecision.REQUIRE_APPROVAL,
                    reason=ClipboardReason.SENSITIVE_CONTENT_REQUIRES_APPROVAL,
                    safe_to_paste=False,
                    requires_approval=True,
                    target_field_known=True,
                    focus_known=True,
                    sensitivity_level=scan.highest_level,
                )

            return PasteSafetyDecision(
                decision=ClipboardDecision.BLOCK,
                reason=ClipboardReason.SENSITIVE_CONTENT_BLOCKED,
                safe_to_paste=False,
                requires_approval=False,
                target_field_known=True,
                focus_known=True,
                sensitivity_level=scan.highest_level,
            )

        return PasteSafetyDecision(
            decision=ClipboardDecision.ALLOW,
            reason=ClipboardReason.PASTE_READY,
            safe_to_paste=True,
            requires_approval=False,
            target_field_known=True,
            focus_known=True,
            sensitivity_level=scan.highest_level,
        )


class ClipboardHijackDetector:
    def detect(self, report: ClipboardHijackReport) -> bool:
        return report.status == ClipboardHijackStatus.HIJACKED


class ClipboardRuntime:
    """
    Phase 8 Step 28 Clipboard & Sensitive Data Runtime.

    Responsibilities:
    - hash clipboard content without storing raw content
    - scan for sensitive data without returning raw content
    - block unknown-field paste
    - block uncertain-focus paste
    - detect clipboard hijack by hash mismatch
    - audit every clipboard decision

    Non-responsibilities:
    - no real OS clipboard read/write yet
    - no paste execution
    - no raw content logging
    """

    def __init__(
        self,
        *,
        name: str = "clipboard_runtime",
        hasher: ClipboardHashVerifier | None = None,
        scanner: ClipboardSensitivityScanner | None = None,
        paste_policy: PasteSafetyPolicy | None = None,
        hijack_detector: ClipboardHijackDetector | None = None,
    ) -> None:
        cleaned = name.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._hasher = hasher or ClipboardHashVerifier()
        self._scanner = scanner or ClipboardSensitivityScanner()
        self._paste_policy = paste_policy or PasteSafetyPolicy()
        self._hijack_detector = hijack_detector or ClipboardHijackDetector()
        self._sessions: dict[str, ClipboardRuntimeSession] = {}
        self._results: list[ClipboardOperationResult] = []
        self._audits: list[ClipboardAuditRecord] = []
        self._events: list[ClipboardRuntimeEvent] = []
        self._lock = RLock()
        self._last_reason: ClipboardReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def create_session(
        self,
        *,
        workspace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> ClipboardRuntimeSession:
        session = ClipboardRuntimeSession(
            workspace_id=workspace_id,
            metadata=metadata or {},
        )
        event = self._event(
            kind=ClipboardEventKind.SESSION_CREATED,
            reason=ClipboardReason.SESSION_CREATED,
            session_id=session.session_id,
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._events.append(event)
            self._last_reason = event.reason

        return session

    def record_hash(
        self,
        *,
        session_id: str,
        clipboard_text: str,
    ) -> ClipboardOperationResult:
        if self.session_for(session_id) is None:
            result = self._blocked_missing_session(
                operation=ClipboardOperationKind.READ_HASH
            )
            self._record_result(result, session_id)
            return result

        if clipboard_text == "":
            result = _result(
                operation=ClipboardOperationKind.READ_HASH,
                status=ClipboardStatus.BLOCKED,
                decision=ClipboardDecision.BLOCK,
                reason=ClipboardReason.EMPTY_CLIPBOARD_BLOCKED,
                message="empty clipboard cannot be hashed for paste safety",
            )
            self._record_result(result, session_id)
            return result

        hash_record = self._hasher.hash_text(clipboard_text)
        result = _result(
            operation=ClipboardOperationKind.READ_HASH,
            status=ClipboardStatus.VERIFIED,
            decision=ClipboardDecision.ALLOW,
            reason=ClipboardReason.HASH_RECORDED,
            hash_record=hash_record,
            message="clipboard hash recorded without raw content",
        )
        self._record_result(result, session_id, hash_record.content_hash)

        return result

    def prepare_paste(
        self,
        *,
        session_id: str,
        clipboard_text: str,
        target_field_known: bool,
        focus_known: bool,
        allow_sensitive: bool = False,
    ) -> ClipboardOperationResult:
        if self.session_for(session_id) is None:
            result = self._blocked_missing_session(
                operation=ClipboardOperationKind.PREPARE_PASTE
            )
            self._record_result(result, session_id)
            return result

        if clipboard_text == "":
            result = _result(
                operation=ClipboardOperationKind.PREPARE_PASTE,
                status=ClipboardStatus.BLOCKED,
                decision=ClipboardDecision.BLOCK,
                reason=ClipboardReason.EMPTY_CLIPBOARD_BLOCKED,
                message="empty clipboard cannot be pasted",
            )
            self._record_result(result, session_id)
            return result

        hash_record = self._hasher.hash_text(clipboard_text)
        scan = self._scanner.scan(clipboard_text)
        policy = self._paste_policy.evaluate(
            scan=scan,
            target_field_known=target_field_known,
            focus_known=focus_known,
            allow_sensitive=allow_sensitive,
        )

        status = ClipboardStatus.READY
        if policy.requires_approval:
            status = ClipboardStatus.NEEDS_APPROVAL
        elif not policy.safe_to_paste:
            status = ClipboardStatus.BLOCKED

        result = _result(
            operation=ClipboardOperationKind.PREPARE_PASTE,
            status=status,
            decision=policy.decision,
            reason=policy.reason,
            hash_record=hash_record,
            scan=scan,
            paste_policy=policy,
            safe_for_physical_paste=policy.safe_to_paste,
            message=_paste_message(policy),
        )
        self._record_result(result, session_id, hash_record.content_hash)

        return result

    def verify_clipboard_hash(
        self,
        *,
        session_id: str,
        expected_hash: str,
        observed_clipboard_text: str,
        phase: ClipboardHashPhase,
    ) -> ClipboardOperationResult:
        if self.session_for(session_id) is None:
            result = self._blocked_missing_session(
                operation=ClipboardOperationKind.VERIFY_BEFORE_PASTE
            )
            self._record_result(result, session_id)
            return result

        report = self._hasher.verify(
            expected_hash=expected_hash,
            observed_text=observed_clipboard_text,
            phase=phase,
        )
        hijacked = self._hijack_detector.detect(report)
        operation = (
            ClipboardOperationKind.VERIFY_BEFORE_PASTE
            if phase == ClipboardHashPhase.BEFORE_PASTE
            else ClipboardOperationKind.VERIFY_AFTER_PASTE
        )

        if hijacked:
            result = _result(
                operation=operation,
                status=ClipboardStatus.HIJACK_DETECTED,
                decision=ClipboardDecision.ABORT,
                reason=ClipboardReason.CLIPBOARD_HIJACK_DETECTED,
                hijack_report=report,
                message="clipboard hash mismatch detected; abort paste",
            )
            self._record_result(
                result,
                session_id,
                observed_hash=report.observed_hash,
                hijacked=True,
            )
            return result

        result = _result(
            operation=operation,
            status=ClipboardStatus.VERIFIED,
            decision=ClipboardDecision.ALLOW,
            reason=ClipboardReason.CLIPBOARD_HASH_VERIFIED,
            hijack_report=report,
            message="clipboard hash verified",
        )
        self._record_result(result, session_id, observed_hash=report.observed_hash)

        return result

    def session_for(
        self,
        session_id: str,
    ) -> ClipboardRuntimeSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def results(self) -> tuple[ClipboardOperationResult, ...]:
        with self._lock:
            return tuple(self._results)

    def audits(self) -> tuple[ClipboardAuditRecord, ...]:
        with self._lock:
            return tuple(self._audits)

    def events(self) -> tuple[ClipboardRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def snapshot(self) -> ClipboardRuntimeSnapshot:
        with self._lock:
            return ClipboardRuntimeSnapshot(
                name=self.name,
                session_count=len(self._sessions),
                result_count=len(self._results),
                safe_paste_count=sum(
                    1
                    for result in self._results
                    if result.safe_for_physical_paste
                ),
                approval_count=sum(
                    1
                    for result in self._results
                    if result.status == ClipboardStatus.NEEDS_APPROVAL
                ),
                blocked_count=sum(
                    1
                    for result in self._results
                    if result.status
                    in {
                        ClipboardStatus.BLOCKED,
                        ClipboardStatus.FAILED,
                        ClipboardStatus.HIJACK_DETECTED,
                    }
                ),
                hijack_count=sum(
                    1
                    for result in self._results
                    if result.status == ClipboardStatus.HIJACK_DETECTED
                ),
                audit_count=len(self._audits),
                runtime_event_count=len(self._events),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        event = self._event(
            kind=ClipboardEventKind.RUNTIME_RESET,
            reason=ClipboardReason.RUNTIME_RESET,
        )

        with self._lock:
            self._sessions.clear()
            self._results.clear()
            self._audits.clear()
            self._events.clear()
            self._events.append(event)
            self._last_reason = event.reason

    def _blocked_missing_session(
        self,
        *,
        operation: ClipboardOperationKind,
    ) -> ClipboardOperationResult:
        return _result(
            operation=operation,
            status=ClipboardStatus.FAILED,
            decision=ClipboardDecision.BLOCK,
            reason=ClipboardReason.SESSION_NOT_FOUND,
            message="clipboard runtime session not found",
        )

    def _record_result(
        self,
        result: ClipboardOperationResult,
        session_id: str,
        observed_hash: str | None = None,
        hijacked: bool = False,
    ) -> None:
        event = self._event(
            kind=_event_kind_for(result),
            reason=result.reason,
            session_id=session_id,
            result_id=result.result_id,
            audit_id=result.audit.audit_id,
            metadata={
                "status": result.status.value,
                "decision": result.decision.value,
            },
        )

        with self._lock:
            self._results.append(result)
            self._audits.append(result.audit)
            self._events.append(event)
            self._last_reason = result.reason
            self._touch_session(
                session_id=session_id,
                observed_hash=observed_hash,
                hijacked=hijacked,
            )

    def _touch_session(
        self,
        *,
        session_id: str,
        observed_hash: str | None,
        hijacked: bool,
    ) -> None:
        session = self._sessions.get(session_id)

        if session is None:
            return

        self._sessions[session_id] = session.model_copy(
            update={
                "updated_at": utc_now(),
                "operation_count": session.operation_count + 1,
                "hijack_count": session.hijack_count + (1 if hijacked else 0),
                "last_hash": observed_hash or session.last_hash,
            }
        )

    @staticmethod
    def _event(
        *,
        kind: ClipboardEventKind,
        reason: ClipboardReason,
        session_id: str | None = None,
        result_id: str | None = None,
        audit_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ClipboardRuntimeEvent:
        return ClipboardRuntimeEvent(
            kind=kind,
            reason=reason,
            session_id=session_id,
            result_id=result_id,
            audit_id=audit_id,
            metadata=metadata or {},
        )


def _result(
    *,
    operation: ClipboardOperationKind,
    status: ClipboardStatus,
    decision: ClipboardDecision,
    reason: ClipboardReason,
    message: str,
    hash_record: ClipboardHashRecord | None = None,
    scan: ClipboardSensitivityScan | None = None,
    paste_policy: PasteSafetyDecision | None = None,
    hijack_report: ClipboardHijackReport | None = None,
    safe_for_physical_paste: bool = False,
) -> ClipboardOperationResult:
    content_hash = None
    if hash_record is not None:
        content_hash = hash_record.content_hash

    finding_count = 0
    if scan is not None:
        finding_count = scan.finding_count

    audit = ClipboardAuditRecord(
        operation=operation,
        status=status,
        decision=decision,
        reason=reason,
        content_hash=content_hash,
        finding_count=finding_count,
        raw_content_logged=False,
    )
    confidence = _confidence_for_result(
        status=status,
        scan=scan,
        hijack_report=hijack_report,
    )

    return ClipboardOperationResult(
        operation=operation,
        status=status,
        decision=decision,
        reason=reason,
        hash_record=hash_record,
        scan=scan,
        paste_policy=paste_policy,
        hijack_report=hijack_report,
        audit=audit,
        trust=TrustCalibration(
            confidence=confidence,
            stability=max(0.0, min(1.0, confidence + 0.05)),
            ambiguity=1.0 - confidence,
            source=EnvironmentSource.OS_OBSERVER,
            reason="clipboard security boundary decision",
            metadata={"policy": TrustPolicyClassification.REVIEW.value},
        ),
        safe_for_physical_paste=safe_for_physical_paste,
        message=message,
    )


def _event_kind_for(result: ClipboardOperationResult) -> ClipboardEventKind:
    if result.status == ClipboardStatus.HIJACK_DETECTED:
        return ClipboardEventKind.HIJACK_DETECTED

    if result.status in {
        ClipboardStatus.READY,
        ClipboardStatus.VERIFIED,
    }:
        return ClipboardEventKind.OPERATION_ALLOWED

    return ClipboardEventKind.OPERATION_BLOCKED


def _confidence_for_result(
    *,
    status: ClipboardStatus,
    scan: ClipboardSensitivityScan | None,
    hijack_report: ClipboardHijackReport | None,
) -> float:
    if status == ClipboardStatus.HIJACK_DETECTED:
        return 0.95

    if hijack_report is not None and hijack_report.clean:
        return 0.92

    if scan is None:
        return 0.84 if status == ClipboardStatus.VERIFIED else 0.30

    if scan.highest_level == ClipboardSensitivityLevel.CRITICAL:
        return 0.95

    if scan.highest_level == ClipboardSensitivityLevel.HIGH:
        return 0.90

    if scan.highest_level == ClipboardSensitivityLevel.MEDIUM:
        return 0.82

    return 0.78


def _confidence_for_level(level: ClipboardSensitivityLevel) -> float:
    if level == ClipboardSensitivityLevel.CRITICAL:
        return 0.95

    if level == ClipboardSensitivityLevel.HIGH:
        return 0.90

    if level == ClipboardSensitivityLevel.MEDIUM:
        return 0.82

    if level == ClipboardSensitivityLevel.LOW:
        return 0.70

    return 0.50


def _highest_level(
    findings: tuple[ClipboardSensitivityFinding, ...],
) -> ClipboardSensitivityLevel:
    if not findings:
        return ClipboardSensitivityLevel.SAFE

    order = {
        ClipboardSensitivityLevel.SAFE: 0,
        ClipboardSensitivityLevel.LOW: 1,
        ClipboardSensitivityLevel.MEDIUM: 2,
        ClipboardSensitivityLevel.HIGH: 3,
        ClipboardSensitivityLevel.CRITICAL: 4,
    }

    return max((finding.level for finding in findings), key=lambda item: order[item])


def _paste_message(policy: PasteSafetyDecision) -> str:
    return (
        f"paste decision={policy.decision.value}; "
        f"reason={policy.reason.value}; "
        f"sensitivity={policy.sensitivity_level.value}"
    )


def _clean_required(value: str) -> str:
    cleaned = value.strip()

    if not cleaned:
        raise ValueError("field cannot be empty.")

    return cleaned