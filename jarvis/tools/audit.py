from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from threading import RLock
from typing import Any

from pydantic import Field, field_validator, model_validator

from jarvis.tools.ids import new_action_result_id, utc_now
from jarvis.tools.models import ActionRisk, ActionStatus, ToolModel


class ActionAuditEventKind(StrEnum):
    """
    Action audit event kinds.

    These are lifecycle observations. They do not execute anything.
    """

    INTENT_RECEIVED = "intent_received"
    PLAN_PROPOSED = "plan_proposed"
    POLICY_EVALUATED = "policy_evaluated"
    VALIDATION_COMPLETED = "validation_completed"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"
    EXECUTION_STARTED = "execution_started"
    EXECUTION_COMPLETED = "execution_completed"
    EXECUTION_FAILED = "execution_failed"
    ACTION_INTERRUPTED = "action_interrupted"
    ACTION_CANCELLED = "action_cancelled"
    ROLLBACK_PLANNED = "rollback_planned"
    ROLLBACK_COMPLETED = "rollback_completed"
    ROLLBACK_FAILED = "rollback_failed"
    MEMORY_CONTEXT_USED = "memory_context_used"
    DESKTOP_CONTEXT_USED = "desktop_context_used"
    SECURITY_DECISION = "security_decision"


class ActionAuditOutcome(StrEnum):
    """
    Audit outcome band.
    """

    INFO = "info"
    ALLOW = "allow"
    DENY = "deny"
    BLOCKED = "blocked"
    APPROVAL_REQUIRED = "approval_required"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLED_BACK = "rolled_back"


class ActionAuditActor(StrEnum):
    """
    Actor that caused or produced an audit event.
    """

    USER = "user"
    COGNITION = "cognition"
    PLANNER = "planner"
    POLICY = "policy"
    VALIDATOR = "validator"
    RUNTIME = "runtime"
    INTERRUPT_CONTROLLER = "interrupt_controller"
    ROLLBACK_EXECUTOR = "rollback_executor"
    MEMORY = "memory"
    SYSTEM = "system"


class ActionAuditSensitivity(StrEnum):
    """
    Sensitivity classification for audit metadata.
    """

    PUBLIC = "public"
    WORKSPACE = "workspace"
    USER_PRIVATE = "user_private"
    SENSITIVE = "sensitive"


class ActionAuditRecord(ToolModel):
    """
    Immutable audit record for one action lifecycle event.

    This record is intentionally generic so every runtime can log into the same
    audit stream without coupling to specific tool implementations.
    """

    audit_id: str = Field(default_factory=new_action_result_id)
    sequence: int = Field(ge=0)
    action_id: str
    event_kind: ActionAuditEventKind
    actor: ActionAuditActor
    outcome: ActionAuditOutcome
    message: str
    risk: ActionRisk = ActionRisk.LOW
    status: ActionStatus | None = None
    sensitivity: ActionAuditSensitivity = ActionAuditSensitivity.USER_PRIVATE
    correlation_id: str | None = None
    parent_action_id: str | None = None
    source_runtime: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    redacted_fields: tuple[str, ...] = ()
    previous_hash: str | None = None
    record_hash: str | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("audit_id", "action_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("correlation_id", "parent_action_id", "source_runtime")
    @classmethod
    def _clean_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None

    @model_validator(mode="after")
    def _ensure_hash_shape(self) -> ActionAuditRecord:
        if self.record_hash is not None and not self.record_hash.strip():
            raise ValueError("record_hash cannot be empty when provided.")

        return self

    @property
    def terminal(self) -> bool:
        return self.event_kind in {
            ActionAuditEventKind.EXECUTION_COMPLETED,
            ActionAuditEventKind.EXECUTION_FAILED,
            ActionAuditEventKind.ACTION_CANCELLED,
            ActionAuditEventKind.ROLLBACK_COMPLETED,
            ActionAuditEventKind.ROLLBACK_FAILED,
        }


@dataclass(frozen=True, slots=True)
class ActionAuditLogConfig:
    """
    Configuration for ActionAuditLog.
    """

    name: str = "action_audit_log"
    persist_jsonl: bool = False
    log_path: str = ".jarvis_audit/actions.jsonl"
    max_records_in_memory: int = 10_000
    redact_sensitive_values: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if not self.log_path.strip():
            raise ValueError("log_path cannot be empty.")

        if self.max_records_in_memory <= 0:
            raise ValueError("max_records_in_memory must be positive.")


@dataclass(frozen=True, slots=True)
class ActionAuditLogSnapshot:
    """
    Observable diagnostics for ActionAuditLog.
    """

    name: str
    record_count: int
    persisted_count: int
    last_sequence: int
    last_event_kind: ActionAuditEventKind | None
    last_action_id: str | None
    last_hash: str | None
    last_error: str | None


class ActionAuditLog:
    """
    Append-only action audit log.

    Responsibilities:
    - record action lifecycle events
    - redact sensitive metadata
    - maintain sequence numbers
    - maintain a simple hash chain
    - optionally persist JSONL records
    - support querying by action_id

    Non-responsibilities:
    - no action execution
    - no policy decision making
    - no rollback execution
    - no memory retrieval
    """

    _SENSITIVE_KEYS = {
        "password",
        "passcode",
        "secret",
        "token",
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "otp",
        "card",
        "cvv",
        "pin",
    }

    def __init__(
        self,
        *,
        config: ActionAuditLogConfig | None = None,
    ) -> None:
        self._config = config or ActionAuditLogConfig()
        self._config.validate()

        self._lock = RLock()
        self._records: list[ActionAuditRecord] = []
        self._persisted_count = 0
        self._last_sequence = -1
        self._last_hash: str | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def record(
        self,
        *,
        action_id: str,
        event_kind: ActionAuditEventKind,
        actor: ActionAuditActor,
        outcome: ActionAuditOutcome,
        message: str,
        risk: ActionRisk = ActionRisk.LOW,
        status: ActionStatus | None = None,
        sensitivity: ActionAuditSensitivity = ActionAuditSensitivity.USER_PRIVATE,
        correlation_id: str | None = None,
        parent_action_id: str | None = None,
        source_runtime: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> ActionAuditRecord:
        """
        Append one immutable audit record.
        """

        with self._lock:
            sequence = self._last_sequence + 1
            safe_data, redacted = self._redact(data or {})
            previous_hash = self._last_hash

            draft = ActionAuditRecord(
                sequence=sequence,
                action_id=action_id,
                event_kind=event_kind,
                actor=actor,
                outcome=outcome,
                message=message,
                risk=risk,
                status=status,
                sensitivity=sensitivity,
                correlation_id=correlation_id,
                parent_action_id=parent_action_id,
                source_runtime=source_runtime,
                data=safe_data,
                redacted_fields=tuple(redacted),
                previous_hash=previous_hash,
            )
            record_hash = self._hash_record(draft)
            record = draft.model_copy(update={"record_hash": record_hash})

            self._records.append(record)
            self._trim_if_needed()

            self._last_sequence = sequence
            self._last_hash = record_hash
            self._last_error = None

            if self._config.persist_jsonl:
                self._persist(record)

            return record

    def record_intent(
        self,
        *,
        action_id: str,
        user_intent: str,
        requested_by: str = "user",
        correlation_id: str | None = None,
    ) -> ActionAuditRecord:
        """
        Record that a user intent entered the action system.
        """

        return self.record(
            action_id=action_id,
            event_kind=ActionAuditEventKind.INTENT_RECEIVED,
            actor=ActionAuditActor.USER,
            outcome=ActionAuditOutcome.INFO,
            message="action intent received",
            correlation_id=correlation_id,
            data={
                "user_intent": user_intent,
                "requested_by": requested_by,
            },
        )

    def record_plan_proposed(
        self,
        *,
        action_id: str,
        summary: str,
        plan_steps: int,
        risk: ActionRisk,
        requires_approval: bool,
        correlation_id: str | None = None,
    ) -> ActionAuditRecord:
        """
        Record that a planner proposed an action plan.
        """

        return self.record(
            action_id=action_id,
            event_kind=ActionAuditEventKind.PLAN_PROPOSED,
            actor=ActionAuditActor.PLANNER,
            outcome=(
                ActionAuditOutcome.APPROVAL_REQUIRED
                if requires_approval
                else ActionAuditOutcome.INFO
            ),
            message=summary,
            risk=risk,
            correlation_id=correlation_id,
            data={
                "plan_steps": plan_steps,
                "requires_approval": requires_approval,
            },
        )

    def record_execution_completed(
        self,
        *,
        action_id: str,
        runtime: str,
        success: bool,
        output_summary: str,
        status: ActionStatus,
    ) -> ActionAuditRecord:
        """
        Record runtime execution completion.
        """

        return self.record(
            action_id=action_id,
            event_kind=(
                ActionAuditEventKind.EXECUTION_COMPLETED
                if success
                else ActionAuditEventKind.EXECUTION_FAILED
            ),
            actor=ActionAuditActor.RUNTIME,
            outcome=(
                ActionAuditOutcome.SUCCEEDED
                if success
                else ActionAuditOutcome.FAILED
            ),
            message=output_summary,
            status=status,
            source_runtime=runtime,
            data={"success": success},
        )

    def record_interruption(
        self,
        *,
        action_id: str,
        message: str,
        cancelled: bool,
    ) -> ActionAuditRecord:
        """
        Record action interruption or cancellation.
        """

        return self.record(
            action_id=action_id,
            event_kind=(
                ActionAuditEventKind.ACTION_CANCELLED
                if cancelled
                else ActionAuditEventKind.ACTION_INTERRUPTED
            ),
            actor=ActionAuditActor.INTERRUPT_CONTROLLER,
            outcome=(
                ActionAuditOutcome.CANCELLED
                if cancelled
                else ActionAuditOutcome.INFO
            ),
            message=message,
        )

    def record_rollback(
        self,
        *,
        action_id: str,
        success: bool,
        message: str,
    ) -> ActionAuditRecord:
        """
        Record rollback outcome.
        """

        return self.record(
            action_id=action_id,
            event_kind=(
                ActionAuditEventKind.ROLLBACK_COMPLETED
                if success
                else ActionAuditEventKind.ROLLBACK_FAILED
            ),
            actor=ActionAuditActor.ROLLBACK_EXECUTOR,
            outcome=(
                ActionAuditOutcome.ROLLED_BACK
                if success
                else ActionAuditOutcome.FAILED
            ),
            message=message,
        )

    def records_for_action(self, action_id: str) -> tuple[ActionAuditRecord, ...]:
        """
        Return audit records for one action.
        """

        with self._lock:
            return tuple(
                record
                for record in self._records
                if record.action_id == action_id
            )

    def all_records(self) -> tuple[ActionAuditRecord, ...]:
        """
        Return all in-memory audit records.
        """

        with self._lock:
            return tuple(self._records)

    def snapshot(self) -> ActionAuditLogSnapshot:
        """
        Return audit log diagnostics.
        """

        with self._lock:
            last = self._records[-1] if self._records else None

            return ActionAuditLogSnapshot(
                name=self.name,
                record_count=len(self._records),
                persisted_count=self._persisted_count,
                last_sequence=self._last_sequence,
                last_event_kind=last.event_kind if last else None,
                last_action_id=last.action_id if last else None,
                last_hash=self._last_hash,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Reset in-memory audit diagnostics and records.

        This does not delete persisted audit files.
        """

        with self._lock:
            self._records.clear()
            self._persisted_count = 0
            self._last_sequence = -1
            self._last_hash = None
            self._last_error = None

    def _trim_if_needed(self) -> None:
        overflow = len(self._records) - self._config.max_records_in_memory

        if overflow > 0:
            del self._records[:overflow]

    def _persist(self, record: ActionAuditRecord) -> None:
        try:
            path = Path(self._config.log_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with path.open("a", encoding="utf-8") as file:
                payload = record.model_dump(mode="json")
                file.write(json.dumps(payload, sort_keys=True) + "\n")

            self._persisted_count += 1

        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            raise

    def _redact(self, data: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        if not self._config.redact_sensitive_values:
            return data, []

        redacted: list[str] = []
        safe = self._redact_value(data, redacted, prefix="data")

        if not isinstance(safe, dict):
            return {}, redacted

        return safe, redacted

    def _redact_value(
        self,
        value: Any,
        redacted: list[str],
        *,
        prefix: str,
    ) -> Any:
        if isinstance(value, dict):
            output: dict[str, Any] = {}

            for key, item in value.items():
                key_text = str(key)
                child_prefix = f"{prefix}.{key_text}"

                if self._is_sensitive_key(key_text):
                    output[key_text] = "[REDACTED]"
                    redacted.append(child_prefix)
                    continue

                output[key_text] = self._redact_value(
                    item,
                    redacted,
                    prefix=child_prefix,
                )

            return output

        if isinstance(value, list):
            return [
                self._redact_value(item, redacted, prefix=f"{prefix}[]")
                for item in value
            ]

        if isinstance(value, tuple):
            return tuple(
                self._redact_value(item, redacted, prefix=f"{prefix}[]")
                for item in value
            )

        return value

    def _is_sensitive_key(self, key: str) -> bool:
        normalized = key.casefold()

        return any(token in normalized for token in self._SENSITIVE_KEYS)

    @staticmethod
    def _hash_record(record: ActionAuditRecord) -> str:
        payload = record.model_dump(mode="json", exclude={"record_hash"})
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))

        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()