from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class RecoveryStrategy(StrEnum):
    """
    Recovery strategy chosen after orchestration failure.

    These are recommendations and records. Execution remains owned by the
    appropriate governed runtime.
    """

    RETRY_WITH_BACKOFF = "retry_with_backoff"
    RECONSTRUCT_FROM_CHECKPOINT = "reconstruct_from_checkpoint"
    RESTART_WORKER = "restart_worker"
    DEGRADE_GRACEFULLY = "degrade_gracefully"
    ABANDON_AND_LOG = "abandon_and_log"


class RecoveryStatus(StrEnum):
    """
    Recovery lifecycle status.
    """

    PENDING = "pending"
    RECOVERING = "recovering"
    RECOVERED = "recovered"
    DEGRADED = "degraded"
    QUARANTINED = "quarantined"
    ABANDONED = "abandoned"
    FAILED = "failed"


class RecoveryFailureKind(StrEnum):
    """
    Failure classes understood by Recovery Runtime.
    """

    WORKER_CRASH = "worker_crash"
    TASK_FAILURE = "task_failure"
    STALE_ASSIGNMENT = "stale_assignment"
    STUCK_TASK = "stuck_task"
    SYSTEM_RESTART = "system_restart"
    INTERRUPTED_RECOVERY = "interrupted_recovery"
    CHECKPOINT_MISSING = "checkpoint_missing"
    UNKNOWN = "unknown"


class RecoveryReason(StrEnum):
    """
    Machine-readable recovery reason.
    """

    CHECKPOINT_CREATED = "checkpoint_created"
    CHECKPOINT_SKIPPED = "checkpoint_skipped"
    EVENT_RECORDED = "event_recorded"
    STATE_RECONSTRUCTED = "state_reconstructed"
    NO_CHECKPOINT_AVAILABLE = "no_checkpoint_available"
    RETRY_ALLOWED = "retry_allowed"
    RETRY_DENIED = "retry_denied"
    TASK_RECOVERABLE = "task_recoverable"
    TASK_UNRECOVERABLE = "task_unrecoverable"
    TASK_QUARANTINED = "task_quarantined"
    WORKER_RESTART_REQUIRED = "worker_restart_required"
    GRACEFUL_DEGRADATION_REQUIRED = "graceful_degradation_required"
    STALE_ASSIGNMENT_DETECTED = "stale_assignment_detected"
    STUCK_TASK_DETECTED = "stuck_task_detected"
    RECOVERY_RECORDED = "recovery_recorded"
    RUNTIME_RESET = "runtime_reset"


class RecoveryEventType(StrEnum):
    """
    Event-log operations used for state reconstruction.
    """

    STATE_SET = "state_set"
    STATE_DELETE = "state_delete"
    TASK_FAILED = "task_failed"
    TASK_RECOVERED = "task_recovered"
    WORKER_FAILED = "worker_failed"
    WORKER_RESTARTED = "worker_restarted"
    TASK_ABANDONED = "task_abandoned"
    TASK_QUARANTINED = "task_quarantined"


class RecoveryEvent(OrchestrationModel):
    """
    Durable event used to replay checkpoint forward.
    """

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    sequence: int = Field(ge=0)
    event_type: RecoveryEventType
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _required_event_id(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("event_id cannot be empty.")

        return cleaned


class RecoveryCheckpoint(OrchestrationModel):
    """
    Durable orchestration checkpoint.

    The state payload is intentionally generic. Higher-level runtimes decide
    what to store. Recovery Runtime stores and reconstructs it safely.
    """

    checkpoint_id: str = Field(default_factory=lambda: uuid4().hex)
    sequence: int = Field(ge=0)
    state: dict[str, Any] = Field(default_factory=dict)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("checkpoint_id")
    @classmethod
    def _required_checkpoint_id(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("checkpoint_id cannot be empty.")

        return cleaned


class RecoverableTaskRecord(OrchestrationModel):
    """
    Task failure record used by Recovery Runtime.

    This model is intentionally independent from scheduler internals.
    """

    task_id: str
    worker_id: str | None = None
    job_id: str | None = None
    failure_kind: RecoveryFailureKind = RecoveryFailureKind.UNKNOWN
    failure_count: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=3, ge=1)
    recoverable: bool = True
    last_error: str | None = None
    age_seconds: int = Field(default=0, ge=0)
    stale_after_seconds: int = Field(default=120, ge=1)
    interrupted: bool = False
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("task_id")
    @classmethod
    def _required_task_id(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("task_id cannot be empty.")

        return cleaned

    @field_validator("worker_id", "job_id")
    @classmethod
    def _clean_optional_ids(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()
        return cleaned or None

    @model_validator(mode="after")
    def _validate_failure_count(self) -> RecoverableTaskRecord:
        if self.failure_count > self.max_attempts:
            raise ValueError("failure_count cannot exceed max_attempts.")

        return self

    @property
    def is_stale(self) -> bool:
        return self.age_seconds >= self.stale_after_seconds

    @property
    def attempts_remaining(self) -> int:
        return max(0, self.max_attempts - self.failure_count)


class RetryDecision(OrchestrationModel):
    """
    Retry policy decision.
    """

    task_id: str
    allowed: bool
    delay_seconds: int = Field(default=0, ge=0)
    reason: RecoveryReason
    attempt_number: int = Field(default=0, ge=0)
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("task_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class RecoveryDecision(OrchestrationModel):
    """
    Full recovery decision for a failed task, worker, or state.
    """

    strategy: RecoveryStrategy
    status: RecoveryStatus
    reason: RecoveryReason
    message: str
    task_id: str | None = None
    worker_id: str | None = None
    retry_delay_seconds: int = Field(default=0, ge=0)
    user_visible: bool = False
    user_message: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("message")
    @classmethod
    def _required_message(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("message cannot be empty.")

        return cleaned

    @model_validator(mode="after")
    def _validate_user_message(self) -> RecoveryDecision:
        if self.user_visible and not self.user_message:
            raise ValueError("user visible recovery requires user_message.")

        return self


class RecoveryAuditEvent(OrchestrationModel):
    """
    Auditable recovery event.

    This answers: what failed, why recovery happened, and what JARVIS decided.
    """

    audit_id: str = Field(default_factory=lambda: uuid4().hex)
    reason: RecoveryReason
    strategy: RecoveryStrategy | None = None
    status: RecoveryStatus
    message: str
    task_id: str | None = None
    worker_id: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("audit_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class ReconstructedState(OrchestrationModel):
    """
    Output of state reconstruction.
    """

    state: dict[str, Any]
    checkpoint_id: str | None = None
    checkpoint_sequence: int | None = None
    replayed_event_count: int = Field(default=0, ge=0)
    reason: RecoveryReason
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecoveryResult(OrchestrationModel):
    """
    Result of a recovery runtime operation.
    """

    reason: RecoveryReason
    success: bool
    message: str
    decision: RecoveryDecision | None = None
    checkpoint: RecoveryCheckpoint | None = None
    reconstructed_state: ReconstructedState | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("message")
    @classmethod
    def _required_message(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("message cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class RecoveryRuntimeConfig:
    """
    Recovery Runtime configuration.

    Checkpoint cadence is explicit because external orchestration controls
    when this runtime is called. This runtime decides whether a checkpoint
    should be written.
    """

    name: str = "recovery_runtime"
    sqlite_path: str = ":memory:"
    checkpoint_interval_seconds: int = 30
    max_retry_attempts: int = 3
    backoff_base_seconds: int = 2
    backoff_max_seconds: int = 60
    stale_assignment_seconds: int = 120
    quarantine_after_failures: int = 3

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.checkpoint_interval_seconds < 1:
            raise ValueError("checkpoint_interval_seconds must be positive.")

        if self.max_retry_attempts < 1:
            raise ValueError("max_retry_attempts must be positive.")

        if self.backoff_base_seconds < 1:
            raise ValueError("backoff_base_seconds must be positive.")

        if self.backoff_max_seconds < self.backoff_base_seconds:
            raise ValueError(
                "backoff_max_seconds cannot be lower than backoff_base_seconds."
            )

        if self.stale_assignment_seconds < 1:
            raise ValueError("stale_assignment_seconds must be positive.")

        if self.quarantine_after_failures < 1:
            raise ValueError("quarantine_after_failures must be positive.")


@dataclass(frozen=True, slots=True)
class RecoveryRuntimeSnapshot:
    """
    Recovery Runtime diagnostics.
    """

    name: str
    checkpoint_count: int
    event_count: int
    audit_count: int
    quarantined_count: int
    last_reason: RecoveryReason | None
    last_status: RecoveryStatus | None


class RetryPolicy:
    """
    Retry policy with exponential backoff.

    Recoverable tasks are retried until max attempts are exhausted.
    Unrecoverable tasks are abandoned and logged.
    """

    def __init__(
        self,
        *,
        config: RecoveryRuntimeConfig | None = None,
    ) -> None:
        self._config = config or RecoveryRuntimeConfig()
        self._config.validate()

    def decide(self, record: RecoverableTaskRecord) -> RetryDecision:
        max_attempts = min(record.max_attempts, self._config.max_retry_attempts)

        if not record.recoverable:
            return RetryDecision(
                task_id=record.task_id,
                allowed=False,
                reason=RecoveryReason.TASK_UNRECOVERABLE,
                attempt_number=record.failure_count,
                message="task is marked unrecoverable",
            )

        if record.failure_count >= max_attempts:
            return RetryDecision(
                task_id=record.task_id,
                allowed=False,
                reason=RecoveryReason.RETRY_DENIED,
                attempt_number=record.failure_count,
                message="retry attempts exhausted",
            )

        attempt_number = record.failure_count + 1
        delay = min(
            self._config.backoff_max_seconds,
            self._config.backoff_base_seconds * (2 ** record.failure_count),
        )

        return RetryDecision(
            task_id=record.task_id,
            allowed=True,
            delay_seconds=delay,
            reason=RecoveryReason.RETRY_ALLOWED,
            attempt_number=attempt_number,
            message="task retry allowed with exponential backoff",
        )


class RecoveryStrategySelector:
    """
    Chooses a recovery strategy from failure state.

    This is decision logic only. It does not execute recovery.
    """

    def __init__(
        self,
        *,
        config: RecoveryRuntimeConfig | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self._config = config or RecoveryRuntimeConfig()
        self._config.validate()
        self._retry_policy = retry_policy or RetryPolicy(config=self._config)

    def decide_for_task(
        self,
        record: RecoverableTaskRecord,
    ) -> RecoveryDecision:
        if record.interrupted:
            return RecoveryDecision(
                strategy=RecoveryStrategy.DEGRADE_GRACEFULLY,
                status=RecoveryStatus.DEGRADED,
                reason=RecoveryReason.GRACEFUL_DEGRADATION_REQUIRED,
                message="task recovery was interrupted; degrading gracefully",
                task_id=record.task_id,
                worker_id=record.worker_id,
                metadata={"failure_kind": record.failure_kind.value},
            )

        if record.failure_kind == RecoveryFailureKind.WORKER_CRASH:
            return RecoveryDecision(
                strategy=RecoveryStrategy.RESTART_WORKER,
                status=RecoveryStatus.RECOVERING,
                reason=RecoveryReason.WORKER_RESTART_REQUIRED,
                message="worker crash requires worker restart",
                task_id=record.task_id,
                worker_id=record.worker_id,
                metadata={"failure_kind": record.failure_kind.value},
            )

        if record.is_stale:
            return RecoveryDecision(
                strategy=RecoveryStrategy.RECONSTRUCT_FROM_CHECKPOINT,
                status=RecoveryStatus.RECOVERING,
                reason=RecoveryReason.STALE_ASSIGNMENT_DETECTED,
                message="stale assignment requires checkpoint reconstruction",
                task_id=record.task_id,
                worker_id=record.worker_id,
                metadata={
                    "age_seconds": record.age_seconds,
                    "stale_after_seconds": record.stale_after_seconds,
                },
            )

        if record.failure_count >= self._config.quarantine_after_failures:
            return RecoveryDecision(
                strategy=RecoveryStrategy.ABANDON_AND_LOG,
                status=RecoveryStatus.QUARANTINED,
                reason=RecoveryReason.TASK_QUARANTINED,
                message="task exceeded quarantine threshold",
                task_id=record.task_id,
                worker_id=record.worker_id,
                user_visible=True,
                user_message="I lost track of that task.",
                metadata={"failure_count": record.failure_count},
            )

        retry = self._retry_policy.decide(record)

        if retry.allowed:
            return RecoveryDecision(
                strategy=RecoveryStrategy.RETRY_WITH_BACKOFF,
                status=RecoveryStatus.RECOVERING,
                reason=RecoveryReason.TASK_RECOVERABLE,
                message="task is recoverable and will be retried",
                task_id=record.task_id,
                worker_id=record.worker_id,
                retry_delay_seconds=retry.delay_seconds,
                metadata={
                    "attempt_number": retry.attempt_number,
                    "retry_reason": retry.reason.value,
                },
            )

        return RecoveryDecision(
            strategy=RecoveryStrategy.ABANDON_AND_LOG,
            status=RecoveryStatus.ABANDONED,
            reason=RecoveryReason.TASK_UNRECOVERABLE,
            message="task is unrecoverable and will be abandoned",
            task_id=record.task_id,
            worker_id=record.worker_id,
            user_visible=True,
            user_message="I lost track of that task.",
            metadata={"retry_reason": retry.reason.value},
        )


class StateReconstructor:
    """
    Reconstructs last known good orchestration state.

    Algorithm:
    1. load latest checkpoint
    2. replay later events in sequence order
    3. return reconstructed state

    Supported event replay:
    - STATE_SET: payload must contain key and value
    - STATE_DELETE: payload must contain key
    """

    def reconstruct(
        self,
        *,
        checkpoint: RecoveryCheckpoint | None,
        events: tuple[RecoveryEvent, ...],
    ) -> ReconstructedState:
        if checkpoint is None:
            return ReconstructedState(
                state={},
                reason=RecoveryReason.NO_CHECKPOINT_AVAILABLE,
            )

        state = dict(checkpoint.state)
        replayed = 0

        for event in sorted(events, key=lambda item: item.sequence):
            if event.sequence <= checkpoint.sequence:
                continue

            if event.event_type == RecoveryEventType.STATE_SET:
                key = event.payload.get("key")

                if isinstance(key, str) and key:
                    state[key] = event.payload.get("value")
                    replayed += 1

            elif event.event_type == RecoveryEventType.STATE_DELETE:
                key = event.payload.get("key")

                if isinstance(key, str) and key:
                    state.pop(key, None)
                    replayed += 1

        return ReconstructedState(
            state=state,
            checkpoint_id=checkpoint.checkpoint_id,
            checkpoint_sequence=checkpoint.sequence,
            replayed_event_count=replayed,
            reason=RecoveryReason.STATE_RECONSTRUCTED,
        )


class RecoveryStore:
    """
    SQLite-backed recovery store.

    Durable enough for Phase 6 while still simple, inspectable, and testable.
    """

    def __init__(self, sqlite_path: str) -> None:
        self._sqlite_path = sqlite_path
        self._connection = self._connect(sqlite_path)
        self._lock = RLock()
        self._init_schema()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def save_checkpoint(self, checkpoint: RecoveryCheckpoint) -> None:
        payload = self._to_json(checkpoint)

        with self._lock:
            self._connection.execute(
                """
                INSERT INTO recovery_checkpoints (
                    checkpoint_id,
                    sequence,
                    payload_json
                )
                VALUES (?, ?, ?)
                """,
                (checkpoint.checkpoint_id, checkpoint.sequence, payload),
            )
            self._connection.commit()

    def save_event(self, event: RecoveryEvent) -> None:
        payload = self._to_json(event)

        with self._lock:
            self._connection.execute(
                """
                INSERT INTO recovery_events (
                    event_id,
                    sequence,
                    event_type,
                    payload_json
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.sequence,
                    event.event_type.value,
                    payload,
                ),
            )
            self._connection.commit()

    def save_audit(self, event: RecoveryAuditEvent) -> None:
        payload = self._to_json(event)

        with self._lock:
            self._connection.execute(
                """
                INSERT INTO recovery_audit_events (
                    audit_id,
                    reason,
                    status,
                    payload_json
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    event.audit_id,
                    event.reason.value,
                    event.status.value,
                    payload,
                ),
            )
            self._connection.commit()

    def latest_checkpoint(self) -> RecoveryCheckpoint | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT payload_json
                FROM recovery_checkpoints
                ORDER BY sequence DESC
                LIMIT 1
                """
            ).fetchone()

        if row is None:
            return None

        return RecoveryCheckpoint.model_validate_json(row[0])

    def events_after(self, sequence: int) -> tuple[RecoveryEvent, ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT payload_json
                FROM recovery_events
                WHERE sequence > ?
                ORDER BY sequence ASC
                """,
                (sequence,),
            ).fetchall()

        return tuple(RecoveryEvent.model_validate_json(row[0]) for row in rows)

    def all_events(self) -> tuple[RecoveryEvent, ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT payload_json
                FROM recovery_events
                ORDER BY sequence ASC
                """
            ).fetchall()

        return tuple(RecoveryEvent.model_validate_json(row[0]) for row in rows)

    def checkpoint_count(self) -> int:
        return self._count("recovery_checkpoints")

    def event_count(self) -> int:
        return self._count("recovery_events")

    def audit_count(self) -> int:
        return self._count("recovery_audit_events")

    def reset(self) -> None:
        with self._lock:
            self._connection.execute("DELETE FROM recovery_checkpoints")
            self._connection.execute("DELETE FROM recovery_events")
            self._connection.execute("DELETE FROM recovery_audit_events")
            self._connection.commit()

    def _count(self, table: str) -> int:
        with self._lock:
            row = self._connection.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()

        return int(row[0])

    def _init_schema(self) -> None:
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS recovery_checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    sequence INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_recovery_checkpoints_sequence
                ON recovery_checkpoints(sequence);

                CREATE TABLE IF NOT EXISTS recovery_events (
                    event_id TEXT PRIMARY KEY,
                    sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_recovery_events_sequence
                ON recovery_events(sequence);

                CREATE TABLE IF NOT EXISTS recovery_audit_events (
                    audit_id TEXT PRIMARY KEY,
                    reason TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            self._connection.commit()

    @staticmethod
    def _connect(sqlite_path: str) -> sqlite3.Connection:
        if sqlite_path != ":memory:":
            path = Path(sqlite_path)
            path.parent.mkdir(parents=True, exist_ok=True)

        connection = sqlite3.connect(sqlite_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _to_json(model: OrchestrationModel) -> str:
        return json.dumps(model.model_dump(mode="json"), default=str)


class RecoveryManager:
    """
    Phase 6 Step 15 Recovery Manager.

    Responsibilities:
    - checkpoint orchestration state
    - append recovery event log
    - reconstruct state after restart
    - classify task/worker recovery decisions
    - record recovery audit events
    - quarantine failing work

    Non-responsibilities:
    - no direct task execution
    - no direct worker mutation
    - no direct scheduler mutation
    """

    def __init__(
        self,
        *,
        config: RecoveryRuntimeConfig | None = None,
        store: RecoveryStore | None = None,
        selector: RecoveryStrategySelector | None = None,
        reconstructor: StateReconstructor | None = None,
    ) -> None:
        self._config = config or RecoveryRuntimeConfig()
        self._config.validate()

        self._store = store or RecoveryStore(self._config.sqlite_path)
        self._selector = selector or RecoveryStrategySelector(
            config=self._config
        )
        self._reconstructor = reconstructor or StateReconstructor()
        self._lock = RLock()

        self._last_checkpoint_sequence: int | None = None
        self._last_checkpoint_age_seconds: int = (
            self._config.checkpoint_interval_seconds
        )
        self._last_reason: RecoveryReason | None = None
        self._last_status: RecoveryStatus | None = None
        self._quarantined_task_ids: set[str] = set()

    @property
    def name(self) -> str:
        return self._config.name

    def checkpoint(
        self,
        *,
        sequence: int,
        state: dict[str, Any],
        force: bool = False,
        elapsed_since_last_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RecoveryResult:
        elapsed = (
            elapsed_since_last_seconds
            if elapsed_since_last_seconds is not None
            else self._last_checkpoint_age_seconds
        )

        if not force and elapsed < self._config.checkpoint_interval_seconds:
            self._record_status(
                reason=RecoveryReason.CHECKPOINT_SKIPPED,
                status=RecoveryStatus.PENDING,
            )

            return RecoveryResult(
                reason=RecoveryReason.CHECKPOINT_SKIPPED,
                success=True,
                message="checkpoint skipped until interval elapses",
                metadata={"elapsed_since_last_seconds": elapsed},
            )

        checkpoint = RecoveryCheckpoint(
            sequence=sequence,
            state=state,
            metadata=metadata or {},
        )
        self._store.save_checkpoint(checkpoint)

        with self._lock:
            self._last_checkpoint_sequence = sequence
            self._last_checkpoint_age_seconds = 0

        self._record_status(
            reason=RecoveryReason.CHECKPOINT_CREATED,
            status=RecoveryStatus.RECOVERED,
        )
        self._audit(
            reason=RecoveryReason.CHECKPOINT_CREATED,
            status=RecoveryStatus.RECOVERED,
            message="orchestration checkpoint created",
            strategy=RecoveryStrategy.RECONSTRUCT_FROM_CHECKPOINT,
            metadata={"sequence": sequence},
        )

        return RecoveryResult(
            reason=RecoveryReason.CHECKPOINT_CREATED,
            success=True,
            message="orchestration checkpoint created",
            checkpoint=checkpoint,
        )

    def append_event(
        self,
        *,
        sequence: int,
        event_type: RecoveryEventType,
        payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RecoveryResult:
        event = RecoveryEvent(
            sequence=sequence,
            event_type=event_type,
            payload=payload or {},
            metadata=metadata or {},
        )
        self._store.save_event(event)

        self._record_status(
            reason=RecoveryReason.EVENT_RECORDED,
            status=RecoveryStatus.PENDING,
        )

        return RecoveryResult(
            reason=RecoveryReason.EVENT_RECORDED,
            success=True,
            message="recovery event recorded",
            metadata={"event_id": event.event_id},
        )

    def reconstruct_last_known_good_state(self) -> RecoveryResult:
        checkpoint = self._store.latest_checkpoint()

        if checkpoint is None:
            reconstructed = self._reconstructor.reconstruct(
                checkpoint=None,
                events=(),
            )
            self._record_status(
                reason=RecoveryReason.NO_CHECKPOINT_AVAILABLE,
                status=RecoveryStatus.DEGRADED,
            )

            return RecoveryResult(
                reason=RecoveryReason.NO_CHECKPOINT_AVAILABLE,
                success=False,
                message="no checkpoint available for reconstruction",
                reconstructed_state=reconstructed,
            )

        events = self._store.events_after(checkpoint.sequence)
        reconstructed = self._reconstructor.reconstruct(
            checkpoint=checkpoint,
            events=events,
        )

        self._record_status(
            reason=RecoveryReason.STATE_RECONSTRUCTED,
            status=RecoveryStatus.RECOVERED,
        )
        self._audit(
            reason=RecoveryReason.STATE_RECONSTRUCTED,
            status=RecoveryStatus.RECOVERED,
            strategy=RecoveryStrategy.RECONSTRUCT_FROM_CHECKPOINT,
            message="state reconstructed from checkpoint and event log",
            metadata={
                "checkpoint_id": checkpoint.checkpoint_id,
                "replayed_event_count": reconstructed.replayed_event_count,
            },
        )

        return RecoveryResult(
            reason=RecoveryReason.STATE_RECONSTRUCTED,
            success=True,
            message="state reconstructed from checkpoint and event log",
            reconstructed_state=reconstructed,
        )

    def recover_task(
        self,
        record: RecoverableTaskRecord,
    ) -> RecoveryResult:
        decision = self._selector.decide_for_task(record)

        if decision.status == RecoveryStatus.QUARANTINED:
            with self._lock:
                self._quarantined_task_ids.add(record.task_id)

        self._record_status(reason=decision.reason, status=decision.status)
        self._audit(
            reason=decision.reason,
            status=decision.status,
            strategy=decision.strategy,
            message=decision.message,
            task_id=record.task_id,
            worker_id=record.worker_id,
            metadata=decision.metadata,
        )

        return RecoveryResult(
            reason=RecoveryReason.RECOVERY_RECORDED,
            success=True,
            message="recovery decision recorded",
            decision=decision,
        )

    def detect_stale_assignments(
        self,
        records: tuple[RecoverableTaskRecord, ...],
    ) -> tuple[RecoverableTaskRecord, ...]:
        stale = tuple(record for record in records if record.is_stale)

        if stale:
            self._record_status(
                reason=RecoveryReason.STALE_ASSIGNMENT_DETECTED,
                status=RecoveryStatus.RECOVERING,
            )

        return stale

    def quarantine_task(
        self,
        record: RecoverableTaskRecord,
        *,
        message: str = "task quarantined by recovery runtime",
    ) -> RecoveryResult:
        with self._lock:
            self._quarantined_task_ids.add(record.task_id)

        decision = RecoveryDecision(
            strategy=RecoveryStrategy.ABANDON_AND_LOG,
            status=RecoveryStatus.QUARANTINED,
            reason=RecoveryReason.TASK_QUARANTINED,
            message=message,
            task_id=record.task_id,
            worker_id=record.worker_id,
            user_visible=True,
            user_message="I lost track of that task.",
        )

        self._record_status(
            reason=RecoveryReason.TASK_QUARANTINED,
            status=RecoveryStatus.QUARANTINED,
        )
        self._audit(
            reason=RecoveryReason.TASK_QUARANTINED,
            status=RecoveryStatus.QUARANTINED,
            strategy=RecoveryStrategy.ABANDON_AND_LOG,
            message=message,
            task_id=record.task_id,
            worker_id=record.worker_id,
        )

        return RecoveryResult(
            reason=RecoveryReason.RECOVERY_RECORDED,
            success=True,
            message="task quarantine recorded",
            decision=decision,
        )

    def is_quarantined(self, task_id: str) -> bool:
        with self._lock:
            return task_id in self._quarantined_task_ids

    def snapshot(self) -> RecoveryRuntimeSnapshot:
        with self._lock:
            return RecoveryRuntimeSnapshot(
                name=self.name,
                checkpoint_count=self._store.checkpoint_count(),
                event_count=self._store.event_count(),
                audit_count=self._store.audit_count(),
                quarantined_count=len(self._quarantined_task_ids),
                last_reason=self._last_reason,
                last_status=self._last_status,
            )

    def reset(self) -> None:
        self._store.reset()

        with self._lock:
            self._last_checkpoint_sequence = None
            self._last_checkpoint_age_seconds = (
                self._config.checkpoint_interval_seconds
            )
            self._last_reason = RecoveryReason.RUNTIME_RESET
            self._last_status = None
            self._quarantined_task_ids.clear()

    def close(self) -> None:
        self._store.close()

    def _record_status(
        self,
        *,
        reason: RecoveryReason,
        status: RecoveryStatus,
    ) -> None:
        with self._lock:
            self._last_reason = reason
            self._last_status = status

    def _audit(
        self,
        *,
        reason: RecoveryReason,
        status: RecoveryStatus,
        message: str,
        strategy: RecoveryStrategy | None = None,
        task_id: str | None = None,
        worker_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        event = RecoveryAuditEvent(
            reason=reason,
            strategy=strategy,
            status=status,
            message=message,
            task_id=task_id,
            worker_id=worker_id,
            metadata=metadata or {},
        )
        self._store.save_audit(event)