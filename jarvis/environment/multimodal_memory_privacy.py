from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from re import IGNORECASE, Pattern, compile
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.environment.environment_memory import (
    EnvironmentMemoryScope,
    WorkflowMemoryGateway,
    WorkspaceMemoryEntry,
)
from jarvis.environment.models import EnvironmentSource, TrustCalibration
from jarvis.environment.trust_runtime import TrustPolicyClassification
from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class SensitiveUIKind(StrEnum):
    NONE = "none"
    PASSWORD = "password"
    PAYMENT_FORM = "payment_form"
    API_TOKEN = "api_token"
    PRIVATE_UI = "private_ui"
    PERSONAL_IDENTIFIER = "personal_identifier"


class SensitiveUISeverity(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MemoryPrivacyDecision(StrEnum):
    STORE = "store"
    REDACT_AND_STORE = "redact_and_store"
    BLOCK = "block"
    EXPIRE = "expire"
    CLEAR = "clear"


class MemoryPrivacyStatus(StrEnum):
    STORED = "stored"
    REDACTED_STORED = "redacted_stored"
    BLOCKED = "blocked"
    EXPIRED = "expired"
    CLEARED = "cleared"
    FAILED = "failed"


class MemoryPrivacyReason(StrEnum):
    SESSION_CREATED = "session_created"
    SAFE_MEMORY_STORED = "safe_memory_stored"
    PRIVATE_UI_REDACTED = "private_ui_redacted"
    TOKEN_REDACTED = "token_redacted"
    PASSWORD_BLOCKED = "password_blocked"
    PAYMENT_FORM_BLOCKED = "payment_form_blocked"
    SESSION_MEMORY_EXPIRED = "session_memory_expired"
    PROJECT_MEMORY_CLEARED = "project_memory_cleared"
    MEMORY_GATEWAY_REQUIRED = "memory_gateway_required"
    SESSION_NOT_FOUND = "session_not_found"
    RUNTIME_RESET = "runtime_reset"


class MemoryPrivacyEventKind(StrEnum):
    SESSION_CREATED = "session_created"
    MEMORY_STORED = "memory_stored"
    MEMORY_REDACTED = "memory_redacted"
    MEMORY_BLOCKED = "memory_blocked"
    MEMORY_EXPIRED = "memory_expired"
    PROJECT_MEMORY_CLEARED = "project_memory_cleared"
    RUNTIME_RESET = "runtime_reset"


class MemoryRetentionKind(StrEnum):
    SESSION = "session"
    PROJECT = "project"


class SensitiveUIFinding(OrchestrationModel):
    finding_id: str = Field(
        default_factory=lambda: f"sensitive_ui_{uuid4().hex}"
    )
    kind: SensitiveUIKind
    severity: SensitiveUISeverity
    field_name: str
    reason: str
    content_hash: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("finding_id", "field_name", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class SensitiveUIClassification(OrchestrationModel):
    classification_id: str = Field(
        default_factory=lambda: f"sensitive_ui_class_{uuid4().hex}"
    )
    findings: tuple[SensitiveUIFinding, ...] = ()
    highest_severity: SensitiveUISeverity = SensitiveUISeverity.NONE
    blocked: bool = False
    redaction_required: bool = False
    reason: str = "no sensitive UI detected"
    created_at: object = Field(default_factory=utc_now)

    @field_validator("classification_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class EnvironmentMemoryPolicy(OrchestrationModel):
    """
    Step 35 policy.

    Environment memory must be useful without becoming invasive.
    """

    block_passwords: bool = True
    block_payment_forms: bool = True
    redact_private_ui: bool = True
    redact_tokens: bool = True
    require_memory_gateway: bool = True
    session_ttl_seconds: int = Field(default=3600, ge=0)
    project_memory_persists: bool = True


class MemoryRedactionPolicy(OrchestrationModel):
    replacement_prefix: str = "<redacted"
    preserve_field_shape: bool = True
    hash_redacted_content: bool = True

    @field_validator("replacement_prefix")
    @classmethod
    def _required_prefix(cls, value: str) -> str:
        return _clean_required(value)


class ProjectMemoryRetention(OrchestrationModel):
    retention_id: str = Field(
        default_factory=lambda: f"project_retention_{uuid4().hex}"
    )
    workspace_id: str
    project_path: str
    persistent: bool = True
    cleared: bool = False
    cleared_at: object | None = None
    updated_at: object = Field(default_factory=utc_now)

    @field_validator("retention_id", "workspace_id", "project_path")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class WorkflowMemoryLifecycleRecord(OrchestrationModel):
    lifecycle_id: str = Field(
        default_factory=lambda: f"workflow_lifecycle_{uuid4().hex}"
    )
    entry: WorkspaceMemoryEntry
    scope: EnvironmentMemoryScope
    retention_kind: MemoryRetentionKind
    expires_at: object | None = None
    expired: bool = False
    cleared: bool = False
    stored_through_gateway: bool = True
    classification: SensitiveUIClassification
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("lifecycle_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _gateway_required(self) -> WorkflowMemoryLifecycleRecord:
        if not self.stored_through_gateway:
            raise ValueError("workflow memory must be stored through gateway.")
        return self


class MemoryPrivacyAuditRecord(OrchestrationModel):
    audit_id: str = Field(
        default_factory=lambda: f"memory_privacy_audit_{uuid4().hex}"
    )
    status: MemoryPrivacyStatus
    decision: MemoryPrivacyDecision
    reason: MemoryPrivacyReason
    session_id: str | None = None
    workspace_id: str | None = None
    entry_id: str | None = None
    raw_sensitive_logged: bool = False
    gateway_used: bool = True
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("audit_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)

    @model_validator(mode="after")
    def _privacy_invariants(self) -> MemoryPrivacyAuditRecord:
        if self.raw_sensitive_logged:
            raise ValueError("memory privacy audit must not log raw sensitive data.")

        if not self.gateway_used:
            raise ValueError("memory write must use Memory Gateway.")

        return self


class MemoryPrivacyResult(OrchestrationModel):
    result_id: str = Field(
        default_factory=lambda: f"memory_privacy_result_{uuid4().hex}"
    )
    status: MemoryPrivacyStatus
    decision: MemoryPrivacyDecision
    reason: MemoryPrivacyReason
    entry: WorkspaceMemoryEntry | None = None
    lifecycle: WorkflowMemoryLifecycleRecord | None = None
    classification: SensitiveUIClassification | None = None
    project_retention: ProjectMemoryRetention | None = None
    expired_count: int = Field(default=0, ge=0)
    cleared_count: int = Field(default=0, ge=0)
    audit: MemoryPrivacyAuditRecord
    trust: TrustCalibration
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("result_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class MemoryPrivacySession(OrchestrationModel):
    session_id: str = Field(default_factory=lambda: f"memory_privacy_{uuid4().hex}")
    workspace_id: str
    store_count: int = Field(default=0, ge=0)
    redaction_count: int = Field(default=0, ge=0)
    blocked_count: int = Field(default=0, ge=0)
    expired_count: int = Field(default=0, ge=0)
    cleared_count: int = Field(default=0, ge=0)
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "workspace_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _clean_required(value)


class MemoryPrivacyRuntimeEvent(OrchestrationModel):
    event_id: str = Field(
        default_factory=lambda: f"memory_privacy_event_{uuid4().hex}"
    )
    kind: MemoryPrivacyEventKind
    reason: MemoryPrivacyReason
    session_id: str | None = None
    result_id: str | None = None
    audit_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        return _clean_required(value)


class MemoryPrivacyRuntimeSnapshot(OrchestrationModel):
    name: str
    session_count: int = Field(ge=0)
    lifecycle_count: int = Field(ge=0)
    result_count: int = Field(ge=0)
    stored_count: int = Field(ge=0)
    redacted_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    expired_count: int = Field(ge=0)
    cleared_count: int = Field(ge=0)
    project_retention_count: int = Field(ge=0)
    audit_count: int = Field(ge=0)
    runtime_event_count: int = Field(ge=0)
    last_reason: MemoryPrivacyReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        return _clean_required(value)


class SensitiveUIClassifier:
    """
    Classifies sensitive UI without storing raw sensitive content.
    """

    _password_patterns: tuple[Pattern[str], ...] = (
        compile(r"\bpassword\b", IGNORECASE),
        compile(r"\bpasswd\b", IGNORECASE),
        compile(r"\bpwd\b", IGNORECASE),
    )
    _payment_patterns: tuple[Pattern[str], ...] = (
        compile(r"\bcard number\b", IGNORECASE),
        compile(r"\bcvv\b", IGNORECASE),
        compile(r"\bexpiry\b", IGNORECASE),
        compile(r"\bpayment\b", IGNORECASE),
        compile(r"\bcredit card\b", IGNORECASE),
    )
    _token_patterns: tuple[Pattern[str], ...] = (
        compile(r"\bapi[_-]?key\b", IGNORECASE),
        compile(r"\bsecret\b", IGNORECASE),
        compile(r"\bbearer\s+[a-z0-9._\-]+", IGNORECASE),
        compile(r"\btoken\b", IGNORECASE),
        compile(r"<redacted-command:[a-f0-9]+>", IGNORECASE),
    )
    _private_patterns: tuple[Pattern[str], ...] = (
        compile(r"\bemail inbox\b", IGNORECASE),
        compile(r"\bprivate message\b", IGNORECASE),
        compile(r"\bchat with\b", IGNORECASE),
        compile(r"\bpersonal\b", IGNORECASE),
    )

    def classify(self, entry: WorkspaceMemoryEntry) -> SensitiveUIClassification:
        findings: list[SensitiveUIFinding] = []

        for field_name, value in _entry_text_fields(entry):
            findings.extend(self._classify_value(field_name=field_name, value=value))

        if not findings:
            return SensitiveUIClassification()

        highest = _highest_severity(findings)
        blocked = any(
            finding.kind
            in {
                SensitiveUIKind.PASSWORD,
                SensitiveUIKind.PAYMENT_FORM,
            }
            for finding in findings
        )
        redaction_required = any(
            finding.kind
            in {
                SensitiveUIKind.API_TOKEN,
                SensitiveUIKind.PRIVATE_UI,
                SensitiveUIKind.PERSONAL_IDENTIFIER,
            }
            for finding in findings
        )

        return SensitiveUIClassification(
            findings=tuple(findings),
            highest_severity=highest,
            blocked=blocked,
            redaction_required=redaction_required,
            reason=f"{len(findings)} sensitive finding(s) detected",
        )

    def _classify_value(
        self,
        *,
        field_name: str,
        value: str,
    ) -> tuple[SensitiveUIFinding, ...]:
        findings: list[SensitiveUIFinding] = []

        if _matches_any(value, self._password_patterns):
            findings.append(
                _finding(
                    kind=SensitiveUIKind.PASSWORD,
                    severity=SensitiveUISeverity.CRITICAL,
                    field_name=field_name,
                    value=value,
                    reason="password-like UI must never be stored",
                )
            )

        if _matches_any(value, self._payment_patterns):
            findings.append(
                _finding(
                    kind=SensitiveUIKind.PAYMENT_FORM,
                    severity=SensitiveUISeverity.CRITICAL,
                    field_name=field_name,
                    value=value,
                    reason="payment form UI must never be stored",
                )
            )

        if _matches_any(value, self._token_patterns):
            findings.append(
                _finding(
                    kind=SensitiveUIKind.API_TOKEN,
                    severity=SensitiveUISeverity.HIGH,
                    field_name=field_name,
                    value=value,
                    reason="token-like content must be redacted",
                )
            )

        if _matches_any(value, self._private_patterns):
            findings.append(
                _finding(
                    kind=SensitiveUIKind.PRIVATE_UI,
                    severity=SensitiveUISeverity.MEDIUM,
                    field_name=field_name,
                    value=value,
                    reason="private UI context must be redacted",
                )
            )

        return tuple(findings)


class WorkflowMemoryLifecycle:
    """
    Tracks session expiry, project retention, and project clearing.
    """

    def __init__(self) -> None:
        self._records: list[WorkflowMemoryLifecycleRecord] = []
        self._project_retention: dict[tuple[str, str], ProjectMemoryRetention] = {}
        self._lock = RLock()

    def add(self, record: WorkflowMemoryLifecycleRecord) -> None:
        with self._lock:
            self._records.append(record)
            project_path = record.entry.project_path
            if project_path is not None:
                key = (record.entry.workspace_id, project_path)
                if key not in self._project_retention:
                    self._project_retention[key] = ProjectMemoryRetention(
                        workspace_id=record.entry.workspace_id,
                        project_path=project_path,
                    )

    def expire_session_memories(
        self,
        *,
        now: datetime | None = None,
    ) -> tuple[WorkflowMemoryLifecycleRecord, ...]:
        current = now or _utc_now_datetime()
        expired: list[WorkflowMemoryLifecycleRecord] = []

        with self._lock:
            updated: list[WorkflowMemoryLifecycleRecord] = []
            for record in self._records:
                should_expire = (
                    record.retention_kind == MemoryRetentionKind.SESSION
                    and record.expires_at is not None
                    and _as_datetime(record.expires_at) <= current
                    and not record.expired
                )
                if should_expire:
                    record = record.model_copy(update={"expired": True})
                    expired.append(record)
                updated.append(record)

            self._records = updated

        return tuple(expired)

    def clear_project(
        self,
        *,
        workspace_id: str,
        project_path: str,
    ) -> tuple[WorkflowMemoryLifecycleRecord, ...]:
        cleared: list[WorkflowMemoryLifecycleRecord] = []
        key = (workspace_id, project_path)

        with self._lock:
            updated_records: list[WorkflowMemoryLifecycleRecord] = []
            for record in self._records:
                if (
                    record.entry.workspace_id == workspace_id
                    and record.entry.project_path == project_path
                    and not record.cleared
                ):
                    record = record.model_copy(update={"cleared": True})
                    cleared.append(record)

                updated_records.append(record)

            self._records = updated_records

            retention = self._project_retention.get(key)
            if retention is None:
                retention = ProjectMemoryRetention(
                    workspace_id=workspace_id,
                    project_path=project_path,
                    cleared=True,
                    cleared_at=utc_now(),
                )
            else:
                retention = retention.model_copy(
                    update={
                        "cleared": True,
                        "cleared_at": utc_now(),
                        "updated_at": utc_now(),
                    }
                )
            self._project_retention[key] = retention

        return tuple(cleared)

    def records(self) -> tuple[WorkflowMemoryLifecycleRecord, ...]:
        with self._lock:
            return tuple(self._records)

    def active_records(self) -> tuple[WorkflowMemoryLifecycleRecord, ...]:
        with self._lock:
            return tuple(
                record
                for record in self._records
                if not record.expired and not record.cleared
            )

    def project_retention(self) -> tuple[ProjectMemoryRetention, ...]:
        with self._lock:
            return tuple(self._project_retention.values())


class MultimodalMemoryPrivacyRuntime:
    """
    Phase 8 Step 35 Multimodal Memory Lifecycle & Privacy.

    This runtime is the privacy gate before environment memories are stored.

    It enforces:
    - passwords never stored
    - payment forms never stored
    - private UI redacted
    - session memories expire
    - project memories persist
    - user can clear project memory
    - memory writes go through Memory Gateway only
    """

    def __init__(
        self,
        *,
        name: str = "multimodal_memory_privacy_runtime",
        gateway: WorkflowMemoryGateway | None = None,
        classifier: SensitiveUIClassifier | None = None,
        lifecycle: WorkflowMemoryLifecycle | None = None,
        policy: EnvironmentMemoryPolicy | None = None,
        redaction_policy: MemoryRedactionPolicy | None = None,
    ) -> None:
        cleaned = name.strip()
        if not cleaned:
            raise ValueError("name cannot be empty.")

        self._name = cleaned
        self._gateway = gateway or WorkflowMemoryGateway()
        self._classifier = classifier or SensitiveUIClassifier()
        self._lifecycle = lifecycle or WorkflowMemoryLifecycle()
        self._policy = policy or EnvironmentMemoryPolicy()
        self._redaction_policy = redaction_policy or MemoryRedactionPolicy()
        self._sessions: dict[str, MemoryPrivacySession] = {}
        self._results: list[MemoryPrivacyResult] = []
        self._audits: list[MemoryPrivacyAuditRecord] = []
        self._events: list[MemoryPrivacyRuntimeEvent] = []
        self._lock = RLock()
        self._last_reason: MemoryPrivacyReason | None = None

    @property
    def name(self) -> str:
        return self._name

    def create_session(
        self,
        *,
        workspace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryPrivacySession:
        session = MemoryPrivacySession(
            workspace_id=workspace_id,
            metadata=metadata or {},
        )
        event = self._event(
            kind=MemoryPrivacyEventKind.SESSION_CREATED,
            reason=MemoryPrivacyReason.SESSION_CREATED,
            session_id=session.session_id,
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._events.append(event)
            self._last_reason = event.reason

        return session

    def store_memory(
        self,
        *,
        session_id: str,
        entry: WorkspaceMemoryEntry,
        retention_kind: MemoryRetentionKind,
    ) -> MemoryPrivacyResult:
        if self.session_for(session_id) is None:
            result = _blocked_result(
                status=MemoryPrivacyStatus.FAILED,
                decision=MemoryPrivacyDecision.BLOCK,
                reason=MemoryPrivacyReason.SESSION_NOT_FOUND,
                message="memory privacy session not found",
            )
            self._record_result(result, session_id)
            return result

        classification = self._classifier.classify(entry)

        if self._should_block(classification):
            reason = _blocked_reason(classification)
            result = _blocked_result(
                status=MemoryPrivacyStatus.BLOCKED,
                decision=MemoryPrivacyDecision.BLOCK,
                reason=reason,
                message="sensitive memory blocked before gateway write",
                classification=classification,
                entry=entry,
            )
            self._record_result(result, session_id)
            return result

        stored_entry = entry
        decision = MemoryPrivacyDecision.STORE
        status = MemoryPrivacyStatus.STORED
        reason = MemoryPrivacyReason.SAFE_MEMORY_STORED

        if classification.redaction_required:
            stored_entry = _redact_entry(
                entry=entry,
                classification=classification,
                policy=self._redaction_policy,
            )
            decision = MemoryPrivacyDecision.REDACT_AND_STORE
            status = MemoryPrivacyStatus.REDACTED_STORED
            reason = _redaction_reason(classification)

        self._gateway.store(
            entry=stored_entry,
            scope=_scope_for_retention(retention_kind),
        )

        lifecycle = WorkflowMemoryLifecycleRecord(
            entry=stored_entry,
            scope=_scope_for_retention(retention_kind),
            retention_kind=retention_kind,
            expires_at=self._expires_at(retention_kind),
            stored_through_gateway=True,
            classification=classification,
        )
        self._lifecycle.add(lifecycle)

        result = _result(
            status=status,
            decision=decision,
            reason=reason,
            message="environment memory accepted by privacy lifecycle",
            entry=stored_entry,
            lifecycle=lifecycle,
            classification=classification,
        )
        self._record_result(result, session_id)
        return result

    def expire_session_memories(
        self,
        *,
        session_id: str,
        now: datetime | None = None,
    ) -> MemoryPrivacyResult:
        if self.session_for(session_id) is None:
            result = _blocked_result(
                status=MemoryPrivacyStatus.FAILED,
                decision=MemoryPrivacyDecision.BLOCK,
                reason=MemoryPrivacyReason.SESSION_NOT_FOUND,
                message="memory privacy session not found",
            )
            self._record_result(result, session_id)
            return result

        expired = self._lifecycle.expire_session_memories(now=now)
        result = _result(
            status=MemoryPrivacyStatus.EXPIRED,
            decision=MemoryPrivacyDecision.EXPIRE,
            reason=MemoryPrivacyReason.SESSION_MEMORY_EXPIRED,
            message="expired session-scoped environment memories",
            expired_count=len(expired),
        )
        self._record_result(result, session_id)
        return result

    def clear_project_memory(
        self,
        *,
        session_id: str,
        project_path: str,
    ) -> MemoryPrivacyResult:
        session = self.session_for(session_id)
        if session is None:
            result = _blocked_result(
                status=MemoryPrivacyStatus.FAILED,
                decision=MemoryPrivacyDecision.BLOCK,
                reason=MemoryPrivacyReason.SESSION_NOT_FOUND,
                message="memory privacy session not found",
            )
            self._record_result(result, session_id)
            return result

        cleared = self._lifecycle.clear_project(
            workspace_id=session.workspace_id,
            project_path=project_path,
        )
        retention = ProjectMemoryRetention(
            workspace_id=session.workspace_id,
            project_path=project_path,
            cleared=True,
            cleared_at=utc_now(),
        )
        result = _result(
            status=MemoryPrivacyStatus.CLEARED,
            decision=MemoryPrivacyDecision.CLEAR,
            reason=MemoryPrivacyReason.PROJECT_MEMORY_CLEARED,
            message="project environment memory cleared",
            project_retention=retention,
            cleared_count=len(cleared),
        )
        self._record_result(result, session_id)
        return result

    def session_for(self, session_id: str) -> MemoryPrivacySession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def lifecycle_records(self) -> tuple[WorkflowMemoryLifecycleRecord, ...]:
        return self._lifecycle.records()

    def active_lifecycle_records(self) -> tuple[WorkflowMemoryLifecycleRecord, ...]:
        return self._lifecycle.active_records()

    def project_retention(self) -> tuple[ProjectMemoryRetention, ...]:
        return self._lifecycle.project_retention()

    def gateway_entries(self) -> tuple[WorkspaceMemoryEntry, ...]:
        return self._gateway.snapshot_entries()

    def results(self) -> tuple[MemoryPrivacyResult, ...]:
        with self._lock:
            return tuple(self._results)

    def audits(self) -> tuple[MemoryPrivacyAuditRecord, ...]:
        with self._lock:
            return tuple(self._audits)

    def events(self) -> tuple[MemoryPrivacyRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def snapshot(self) -> MemoryPrivacyRuntimeSnapshot:
        records = self._lifecycle.records()
        with self._lock:
            return MemoryPrivacyRuntimeSnapshot(
                name=self.name,
                session_count=len(self._sessions),
                lifecycle_count=len(records),
                result_count=len(self._results),
                stored_count=sum(
                    1
                    for result in self._results
                    if result.status == MemoryPrivacyStatus.STORED
                ),
                redacted_count=sum(
                    1
                    for result in self._results
                    if result.status == MemoryPrivacyStatus.REDACTED_STORED
                ),
                blocked_count=sum(
                    1
                    for result in self._results
                    if result.status
                    in {
                        MemoryPrivacyStatus.BLOCKED,
                        MemoryPrivacyStatus.FAILED,
                    }
                ),
                expired_count=sum(1 for record in records if record.expired),
                cleared_count=sum(1 for record in records if record.cleared),
                project_retention_count=len(self._lifecycle.project_retention()),
                audit_count=len(self._audits),
                runtime_event_count=len(self._events),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        event = self._event(
            kind=MemoryPrivacyEventKind.RUNTIME_RESET,
            reason=MemoryPrivacyReason.RUNTIME_RESET,
        )
        with self._lock:
            self._sessions.clear()
            self._results.clear()
            self._audits.clear()
            self._events.clear()
            self._events.append(event)
            self._last_reason = event.reason

    def _should_block(self, classification: SensitiveUIClassification) -> bool:
        if self._policy.block_passwords:
            if any(
                finding.kind == SensitiveUIKind.PASSWORD
                for finding in classification.findings
            ):
                return True

        if self._policy.block_payment_forms:
            if any(
                finding.kind == SensitiveUIKind.PAYMENT_FORM
                for finding in classification.findings
            ):
                return True

        return False

    def _expires_at(self, retention_kind: MemoryRetentionKind) -> object | None:
        if retention_kind == MemoryRetentionKind.PROJECT:
            return None

        return _utc_now_datetime() + timedelta(
            seconds=self._policy.session_ttl_seconds
        )

    def _record_result(
        self,
        result: MemoryPrivacyResult,
        session_id: str,
    ) -> None:
        event = self._event(
            kind=_event_kind_for(result),
            reason=result.reason,
            session_id=session_id,
            result_id=result.result_id,
            audit_id=result.audit.audit_id,
            metadata={"status": result.status.value},
        )
        with self._lock:
            self._results.append(result)
            self._audits.append(result.audit)
            self._events.append(event)
            self._last_reason = result.reason
            session = self._sessions.get(session_id)
            if session is not None:
                self._sessions[session_id] = session.model_copy(
                    update={
                        "updated_at": utc_now(),
                        "store_count": session.store_count
                        + (
                            1
                            if result.status
                            in {
                                MemoryPrivacyStatus.STORED,
                                MemoryPrivacyStatus.REDACTED_STORED,
                            }
                            else 0
                        ),
                        "redaction_count": session.redaction_count
                        + (
                            1
                            if result.status == MemoryPrivacyStatus.REDACTED_STORED
                            else 0
                        ),
                        "blocked_count": session.blocked_count
                        + (
                            1
                            if result.status
                            in {
                                MemoryPrivacyStatus.BLOCKED,
                                MemoryPrivacyStatus.FAILED,
                            }
                            else 0
                        ),
                        "expired_count": session.expired_count
                        + result.expired_count,
                        "cleared_count": session.cleared_count
                        + result.cleared_count,
                    }
                )

    @staticmethod
    def _event(
        *,
        kind: MemoryPrivacyEventKind,
        reason: MemoryPrivacyReason,
        session_id: str | None = None,
        result_id: str | None = None,
        audit_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryPrivacyRuntimeEvent:
        return MemoryPrivacyRuntimeEvent(
            kind=kind,
            reason=reason,
            session_id=session_id,
            result_id=result_id,
            audit_id=audit_id,
            metadata=metadata or {},
        )


def _result(
    *,
    status: MemoryPrivacyStatus,
    decision: MemoryPrivacyDecision,
    reason: MemoryPrivacyReason,
    message: str,
    entry: WorkspaceMemoryEntry | None = None,
    lifecycle: WorkflowMemoryLifecycleRecord | None = None,
    classification: SensitiveUIClassification | None = None,
    project_retention: ProjectMemoryRetention | None = None,
    expired_count: int = 0,
    cleared_count: int = 0,
) -> MemoryPrivacyResult:
    audit = MemoryPrivacyAuditRecord(
        status=status,
        decision=decision,
        reason=reason,
        session_id=entry.session_id if entry is not None else None,
        workspace_id=entry.workspace_id if entry is not None else None,
        entry_id=entry.entry_id if entry is not None else None,
        raw_sensitive_logged=False,
        gateway_used=True,
    )

    return MemoryPrivacyResult(
        status=status,
        decision=decision,
        reason=reason,
        entry=entry,
        lifecycle=lifecycle,
        classification=classification,
        project_retention=project_retention,
        expired_count=expired_count,
        cleared_count=cleared_count,
        audit=audit,
        trust=_trust(
            confidence=0.86
            if status
            in {
                MemoryPrivacyStatus.STORED,
                MemoryPrivacyStatus.REDACTED_STORED,
                MemoryPrivacyStatus.EXPIRED,
                MemoryPrivacyStatus.CLEARED,
            }
            else 0.25,
            reason=message,
        ),
        message=message,
    )


def _blocked_result(
    *,
    status: MemoryPrivacyStatus,
    decision: MemoryPrivacyDecision,
    reason: MemoryPrivacyReason,
    message: str,
    classification: SensitiveUIClassification | None = None,
    entry: WorkspaceMemoryEntry | None = None,
) -> MemoryPrivacyResult:
    return _result(
        status=status,
        decision=decision,
        reason=reason,
        message=message,
        classification=classification,
        entry=entry,
    )


def _entry_text_fields(entry: WorkspaceMemoryEntry) -> tuple[tuple[str, str], ...]:
    fields: list[tuple[str, str]] = [
        ("app_name", entry.app_name),
    ]

    if entry.project_path is not None:
        fields.append(("project_path", entry.project_path))

    if entry.terminal_directory is not None:
        fields.append(("terminal_directory", entry.terminal_directory))

    for value in entry.active_files:
        fields.append(("active_files", value))

    for value in entry.recent_commands:
        fields.append(("recent_commands", value))

    for value in entry.visible_errors:
        fields.append(("visible_errors", value))

    for value in entry.pending_todos:
        fields.append(("pending_todos", value))

    for key, value in entry.metadata.items():
        if isinstance(value, str):
            fields.append((f"metadata.{key}", value))

    return tuple(fields)


def _finding(
    *,
    kind: SensitiveUIKind,
    severity: SensitiveUISeverity,
    field_name: str,
    value: str,
    reason: str,
) -> SensitiveUIFinding:
    return SensitiveUIFinding(
        kind=kind,
        severity=severity,
        field_name=field_name,
        reason=reason,
        content_hash=_hash(value),
    )


def _highest_severity(
    findings: tuple[SensitiveUIFinding, ...] | list[SensitiveUIFinding],
) -> SensitiveUISeverity:
    rank = {
        SensitiveUISeverity.NONE: 0,
        SensitiveUISeverity.LOW: 1,
        SensitiveUISeverity.MEDIUM: 2,
        SensitiveUISeverity.HIGH: 3,
        SensitiveUISeverity.CRITICAL: 4,
    }
    return max((finding.severity for finding in findings), key=lambda item: rank[item])


def _matches_any(value: str, patterns: tuple[Pattern[str], ...]) -> bool:
    return any(pattern.search(value) is not None for pattern in patterns)


def _redact_entry(
    *,
    entry: WorkspaceMemoryEntry,
    classification: SensitiveUIClassification,
    policy: MemoryRedactionPolicy,
) -> WorkspaceMemoryEntry:
    sensitive_fields = {finding.field_name for finding in classification.findings}

    def redact_tuple(field_name: str, values: tuple[str, ...]) -> tuple[str, ...]:
        if field_name not in sensitive_fields:
            return values

        return tuple(
            _normalize_redaction_marker(value, policy=policy)
            if value.startswith("<redacted-command:")
            else _redacted(value, policy=policy)
            for value in values
        )

    metadata = dict(entry.metadata)
    for field_name in sensitive_fields:
        if field_name.startswith("metadata."):
            key = field_name.removeprefix("metadata.")
            value = metadata.get(key)
            if isinstance(value, str):
                metadata[key] = _redacted(value, policy=policy)

    return entry.model_copy(
        update={
            "recent_commands": redact_tuple(
                "recent_commands",
                entry.recent_commands,
            ),
            "visible_errors": redact_tuple(
                "visible_errors",
                entry.visible_errors,
            ),
            "pending_todos": redact_tuple(
                "pending_todos",
                entry.pending_todos,
            ),
            "metadata": metadata,
            "policy": TrustPolicyClassification.REVIEW,
        }
    )


def _redacted(value: str, *, policy: MemoryRedactionPolicy) -> str:
    if not policy.hash_redacted_content:
        return f"{policy.replacement_prefix}>"

    return f"{policy.replacement_prefix}:{_hash(value)[:12]}>"


def _normalize_redaction_marker(
    value: str,
    *,
    policy: MemoryRedactionPolicy,
) -> str:
    return f"{policy.replacement_prefix}:{_hash(value)[:12]}>"


def _blocked_reason(
    classification: SensitiveUIClassification,
) -> MemoryPrivacyReason:
    if any(
        finding.kind == SensitiveUIKind.PASSWORD
        for finding in classification.findings
    ):
        return MemoryPrivacyReason.PASSWORD_BLOCKED

    if any(
        finding.kind == SensitiveUIKind.PAYMENT_FORM
        for finding in classification.findings
    ):
        return MemoryPrivacyReason.PAYMENT_FORM_BLOCKED

    return MemoryPrivacyReason.MEMORY_GATEWAY_REQUIRED


def _redaction_reason(
    classification: SensitiveUIClassification,
) -> MemoryPrivacyReason:
    if any(
        finding.kind == SensitiveUIKind.API_TOKEN
        for finding in classification.findings
    ):
        return MemoryPrivacyReason.TOKEN_REDACTED

    return MemoryPrivacyReason.PRIVATE_UI_REDACTED


def _scope_for_retention(retention_kind: MemoryRetentionKind) -> EnvironmentMemoryScope:
    if retention_kind == MemoryRetentionKind.PROJECT:
        return EnvironmentMemoryScope.PROJECT

    return EnvironmentMemoryScope.SESSION


def _event_kind_for(result: MemoryPrivacyResult) -> MemoryPrivacyEventKind:
    if result.status == MemoryPrivacyStatus.REDACTED_STORED:
        return MemoryPrivacyEventKind.MEMORY_REDACTED

    if result.status == MemoryPrivacyStatus.STORED:
        return MemoryPrivacyEventKind.MEMORY_STORED

    if result.status == MemoryPrivacyStatus.EXPIRED:
        return MemoryPrivacyEventKind.MEMORY_EXPIRED

    if result.status == MemoryPrivacyStatus.CLEARED:
        return MemoryPrivacyEventKind.PROJECT_MEMORY_CLEARED

    return MemoryPrivacyEventKind.MEMORY_BLOCKED


def _trust(
    *,
    confidence: float,
    reason: str,
) -> TrustCalibration:
    return TrustCalibration(
        confidence=confidence,
        stability=max(0.0, min(1.0, confidence + 0.05)),
        ambiguity=1.0 - confidence,
        source=EnvironmentSource.OS_OBSERVER,
        reason=reason,
        metadata={"policy": TrustPolicyClassification.SAFE.value},
    )


def _hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _utc_now_datetime() -> datetime:
    return datetime.now(UTC)


def _as_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value

    raise TypeError("expected datetime value.")


def _clean_required(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("field cannot be empty.")
    return cleaned