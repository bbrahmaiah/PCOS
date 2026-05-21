from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from jarvis.runtime.config import get_settings
from jarvis.runtime.observability.structured_logger import get_logger
from jarvis.runtime.security.policy_engine import PermissionResult
from jarvis.runtime.shared.enums import PermissionDecision, RiskLevel


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_audit_id() -> str:
    return uuid4().hex


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """
    Immutable security audit record.
    """

    action: str
    decision: PermissionDecision
    risk: RiskLevel
    reason: str
    request_id: str
    audit_id: str = field(default_factory=new_audit_id)
    identity_id: str | None = None
    correlation_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "identity_id": self.identity_id,
            "action": self.action,
            "decision": self.decision.value,
            "risk": self.risk.value,
            "reason": self.reason,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


class AuditLogger:
    """
    Security audit logger.

    Keeps an in-memory bounded log and writes JSONL audit records to disk.
    """

    def __init__(
        self,
        *,
        log_file: Path | None = None,
        memory_limit: int = 1_000,
    ) -> None:
        if memory_limit < 1:
            raise ValueError("memory_limit must be greater than zero.")

        settings = get_settings()

        self.log_file = log_file or (settings.paths.logs_dir / "security_audit.jsonl")
        self.memory_limit = memory_limit
        self._lock = RLock()
        self._records: list[AuditRecord] = []
        self._logger = get_logger("security.audit")

        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def record(self, record: AuditRecord) -> None:
        if not isinstance(record, AuditRecord):
            raise TypeError("record must be an AuditRecord.")

        with self._lock:
            self._records.append(record)
            self._records = self._records[-self.memory_limit :]

        self._write_jsonl(record)

        self._logger.info(
            "security_audit_recorded",
            audit_id=record.audit_id,
            request_id=record.request_id,
            correlation_id=record.correlation_id,
            action=record.action,
            decision=record.decision.value,
            risk=record.risk.value,
            identity_id=record.identity_id,
        )

    def record_result(
        self,
        result: PermissionResult,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> AuditRecord:
        record = AuditRecord(
            action=result.request.action,
            decision=result.decision,
            risk=result.risk,
            reason=result.reason,
            request_id=result.request.request_id,
            identity_id=result.identity_id,
            correlation_id=result.request.correlation_id,
            metadata=metadata or {},
        )

        self.record(record)

        return record

    def records(self) -> tuple[AuditRecord, ...]:
        with self._lock:
            return tuple(self._records)

    def clear_memory(self) -> None:
        with self._lock:
            self._records.clear()

    def _write_jsonl(self, record: AuditRecord) -> None:
        line = json.dumps(record.to_dict(), ensure_ascii=False, default=str)

        with self.log_file.open("a", encoding="utf-8") as file:
            file.write(line + "\n")