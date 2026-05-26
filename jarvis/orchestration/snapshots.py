from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.orchestration.ids import TaskId, utc_now, validate_task_id
from jarvis.orchestration.models import OrchestrationModel


def new_snapshot_id() -> str:
    return f"snapshot_{uuid4().hex}"


def new_turn_id() -> str:
    return f"turn_{uuid4().hex}"


def new_context_write_id() -> str:
    return f"ctxwrite_{uuid4().hex}"


class ContextSnapshotStatus(StrEnum):
    """
    Lifecycle status for a context snapshot.
    """

    ACTIVE = "active"
    SEALED = "sealed"
    EXPIRED = "expired"


class ContextWriteKind(StrEnum):
    """
    Type of context update requested during or after a turn.
    """

    MEMORY_RESULT = "memory_result"
    MEMORY_WRITE = "memory_write"
    TOOL_RESULT = "tool_result"
    SESSION_UPDATE = "session_update"
    BACKGROUND_NOTE = "background_note"
    OBSERVABILITY = "observability"


class ContextWriteDisposition(StrEnum):
    """
    Context write handling decision.
    """

    QUEUED = "queued"
    APPLIED = "applied"
    REJECTED = "rejected"


class ContextSnapshotReason(StrEnum):
    """
    Machine-readable context snapshot reason.
    """

    SNAPSHOT_CREATED = "snapshot_created"
    SNAPSHOT_ALREADY_ACTIVE = "snapshot_already_active"
    SNAPSHOT_SEALED = "snapshot_sealed"
    SNAPSHOT_EXPIRED = "snapshot_expired"
    NO_ACTIVE_SNAPSHOT = "no_active_snapshot"
    WRITE_QUEUED_FOR_NEXT_TURN = "write_queued_for_next_turn"
    WRITE_APPLIED_TO_NEXT_CONTEXT = "write_applied_to_next_context"
    WRITE_REJECTED_ACTIVE_SNAPSHOT_IMMUTABLE = (
        "write_rejected_active_snapshot_immutable"
    )
    WRITE_REJECTED_QUEUE_FULL = "write_rejected_queue_full"
    WRITE_REJECTED_TURN_MISMATCH = "write_rejected_turn_mismatch"


class ConversationTurnContext(OrchestrationModel):
    """
    Immutable context for one conversation turn.

    This is the stable context cognition sees while generating.
    """

    turn_id: str = Field(default_factory=new_turn_id)
    session_id: str | None = None
    user_text: str
    topic: str | None = None
    memory_refs: tuple[str, ...] = ()
    active_task_ids: tuple[TaskId, ...] = ()
    metadata: dict[str, object] = Field(default_factory=dict)
    created_at: object = Field(default_factory=utc_now)

    @field_validator("turn_id", "user_text")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("session_id", "topic")
    @classmethod
    def _clean_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None

    @field_validator("active_task_ids")
    @classmethod
    def _validate_task_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(validate_task_id(value) for value in values)


class SnapshotPolicy(OrchestrationModel):
    """
    Policy for protecting active conversation context.
    """

    freeze_context_during_turn: bool = True
    queue_background_writes: bool = True
    reject_direct_active_mutation: bool = True
    max_queued_writes: int = Field(default=128, gt=0)


class ContextSnapshot(OrchestrationModel):
    """
    Frozen context snapshot for an active turn.

    Background tasks may read this snapshot, but they must not mutate it.
    """

    snapshot_id: str = Field(default_factory=new_snapshot_id)
    turn_context: ConversationTurnContext
    status: ContextSnapshotStatus = ContextSnapshotStatus.ACTIVE
    version: int = Field(default=0, ge=0)
    created_at: object = Field(default_factory=utc_now)
    sealed_at: object | None = None
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("snapshot_id")
    @classmethod
    def _validate_snapshot_id(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("snapshot_id cannot be empty.")

        if not cleaned.startswith("snapshot_"):
            raise ValueError("snapshot_id must start with 'snapshot_'.")

        return cleaned

    @model_validator(mode="after")
    def _validate_status_shape(self) -> ContextSnapshot:
        if self.status == ContextSnapshotStatus.ACTIVE and self.sealed_at is not None:
            raise ValueError("active snapshots cannot have sealed_at.")

        if self.status in {
            ContextSnapshotStatus.SEALED,
            ContextSnapshotStatus.EXPIRED,
        }:
            if self.sealed_at is None:
                raise ValueError("sealed or expired snapshots require sealed_at.")

        return self

    @property
    def active(self) -> bool:
        return self.status == ContextSnapshotStatus.ACTIVE

    @property
    def turn_id(self) -> str:
        return self.turn_context.turn_id

    def seal(self) -> ContextSnapshot:
        """
        Return a sealed copy of this snapshot.
        """

        return self.model_copy(
            update={
                "status": ContextSnapshotStatus.SEALED,
                "version": self.version + 1,
                "sealed_at": utc_now(),
            }
        )

    def expire(self) -> ContextSnapshot:
        """
        Return an expired copy of this snapshot.
        """

        return self.model_copy(
            update={
                "status": ContextSnapshotStatus.EXPIRED,
                "version": self.version + 1,
                "sealed_at": utc_now(),
            }
        )


class PendingContextWrite(OrchestrationModel):
    """
    Background context write queued for the next turn.

    This keeps background work from mutating the active snapshot.
    """

    write_id: str = Field(default_factory=new_context_write_id)
    turn_id: str
    kind: ContextWriteKind
    key: str
    value: object
    source: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("write_id", "turn_id", "key", "source")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("write_id")
    @classmethod
    def _validate_write_id(cls, value: str) -> str:
        if not value.startswith("ctxwrite_"):
            raise ValueError("write_id must start with 'ctxwrite_'.")

        return value


class ContextSnapshotResult(OrchestrationModel):
    """
    Result of a snapshot runtime operation.
    """

    disposition: ContextWriteDisposition
    reason: ContextSnapshotReason
    success: bool
    message: str
    snapshot: ContextSnapshot | None = None
    pending_write: PendingContextWrite | None = None
    queued_writes: tuple[PendingContextWrite, ...] = ()
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("message")
    @classmethod
    def _required_message(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("message cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class ContextSnapshotRuntimeConfig:
    """
    Context snapshot runtime configuration.
    """

    name: str = "context_snapshot_runtime"

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")


@dataclass(frozen=True, slots=True)
class ContextSnapshotRuntimeSnapshot:
    """
    Context snapshot runtime diagnostics.
    """

    name: str
    has_active_snapshot: bool
    active_turn_id: str | None
    queued_write_count: int
    snapshot_count: int
    queued_count: int
    applied_count: int
    rejected_count: int
    last_reason: ContextSnapshotReason | None


class ContextSnapshotRuntime:
    """
    Phase 6 Context Snapshot Runtime.

    Responsibilities:
    - freeze context at the start of a conversation turn
    - prevent active turn context mutation
    - queue background writes for the next turn
    - seal snapshots at turn completion
    - apply queued writes only to future turn context

    Non-responsibilities:
    - no memory writes
    - no task execution
    - no scheduling
    - no tool execution
    """

    def __init__(
        self,
        *,
        config: ContextSnapshotRuntimeConfig | None = None,
        policy: SnapshotPolicy | None = None,
    ) -> None:
        self._config = config or ContextSnapshotRuntimeConfig()
        self._config.validate()

        self._policy = policy or SnapshotPolicy()
        self._active_snapshot: ContextSnapshot | None = None
        self._queued_writes: tuple[PendingContextWrite, ...] = ()
        self._lock = RLock()

        self._snapshot_count = 0
        self._queued_count = 0
        self._applied_count = 0
        self._rejected_count = 0
        self._last_reason: ContextSnapshotReason | None = None

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def policy(self) -> SnapshotPolicy:
        return self._policy

    def active_snapshot(self) -> ContextSnapshot | None:
        """
        Return the active snapshot, if one exists.
        """

        with self._lock:
            return self._active_snapshot

    def begin_turn(
        self,
        context: ConversationTurnContext,
    ) -> ContextSnapshotResult:
        """
        Freeze a new active context snapshot.

        A new turn cannot start while another active turn is still open.
        """

        with self._lock:
            if self._active_snapshot is not None and self._active_snapshot.active:
                result = ContextSnapshotResult(
                    disposition=ContextWriteDisposition.REJECTED,
                    reason=ContextSnapshotReason.SNAPSHOT_ALREADY_ACTIVE,
                    success=False,
                    message="active context snapshot already exists",
                    snapshot=self._active_snapshot,
                    queued_writes=self._queued_writes,
                )
                self._record(result)

                return result

            snapshot = ContextSnapshot(turn_context=context)
            self._active_snapshot = snapshot

            result = ContextSnapshotResult(
                disposition=ContextWriteDisposition.APPLIED,
                reason=ContextSnapshotReason.SNAPSHOT_CREATED,
                success=True,
                message="context snapshot created",
                snapshot=snapshot,
                queued_writes=self._queued_writes,
            )
            self._record(result)

            return result

    def queue_background_write(
        self,
        write: PendingContextWrite,
    ) -> ContextSnapshotResult:
        """
        Queue a background write for a future turn.

        Writes targeting the active turn are queued, not applied.
        """

        with self._lock:
            active = self._active_snapshot

            if active is None:
                result = ContextSnapshotResult(
                    disposition=ContextWriteDisposition.REJECTED,
                    reason=ContextSnapshotReason.NO_ACTIVE_SNAPSHOT,
                    success=False,
                    message="no active snapshot exists",
                    pending_write=write,
                    queued_writes=self._queued_writes,
                )
                self._record(result)

                return result

            if write.turn_id != active.turn_id:
                result = ContextSnapshotResult(
                    disposition=ContextWriteDisposition.REJECTED,
                    reason=ContextSnapshotReason.WRITE_REJECTED_TURN_MISMATCH,
                    success=False,
                    message="write turn_id does not match active snapshot",
                    snapshot=active,
                    pending_write=write,
                    queued_writes=self._queued_writes,
                )
                self._record(result)

                return result

            if len(self._queued_writes) >= self._policy.max_queued_writes:
                result = ContextSnapshotResult(
                    disposition=ContextWriteDisposition.REJECTED,
                    reason=ContextSnapshotReason.WRITE_REJECTED_QUEUE_FULL,
                    success=False,
                    message="context write queue is full",
                    snapshot=active,
                    pending_write=write,
                    queued_writes=self._queued_writes,
                )
                self._record(result)

                return result

            if not self._policy.queue_background_writes:
                result = ContextSnapshotResult(
                    disposition=ContextWriteDisposition.REJECTED,
                    reason=(
                        ContextSnapshotReason
                        .WRITE_REJECTED_ACTIVE_SNAPSHOT_IMMUTABLE
                    ),
                    success=False,
                    message="background writes are disabled",
                    snapshot=active,
                    pending_write=write,
                    queued_writes=self._queued_writes,
                )
                self._record(result)

                return result

            self._queued_writes = self._queued_writes + (write,)

            result = ContextSnapshotResult(
                disposition=ContextWriteDisposition.QUEUED,
                reason=ContextSnapshotReason.WRITE_QUEUED_FOR_NEXT_TURN,
                success=True,
                message="context write queued for next turn",
                snapshot=active,
                pending_write=write,
                queued_writes=self._queued_writes,
            )
            self._record(result)

            return result

    def reject_active_mutation(
        self,
        write: PendingContextWrite,
    ) -> ContextSnapshotResult:
        """
        Explicitly reject direct mutation of active context.

        This method is intentionally strict. Runtime layers may use it when a
        caller attempts to mutate an active snapshot instead of queueing.
        """

        with self._lock:
            active = self._active_snapshot

            result = ContextSnapshotResult(
                disposition=ContextWriteDisposition.REJECTED,
                reason=(
                    ContextSnapshotReason
                    .WRITE_REJECTED_ACTIVE_SNAPSHOT_IMMUTABLE
                ),
                success=False,
                message="active context snapshot is immutable",
                snapshot=active,
                pending_write=write,
                queued_writes=self._queued_writes,
            )
            self._record(result)

            return result

    def seal_turn(self) -> ContextSnapshotResult:
        """
        Seal the active turn snapshot.
        """

        with self._lock:
            active = self._active_snapshot

            if active is None:
                result = ContextSnapshotResult(
                    disposition=ContextWriteDisposition.REJECTED,
                    reason=ContextSnapshotReason.NO_ACTIVE_SNAPSHOT,
                    success=False,
                    message="no active snapshot exists",
                    queued_writes=self._queued_writes,
                )
                self._record(result)

                return result

            sealed = active.seal()
            self._active_snapshot = None

            result = ContextSnapshotResult(
                disposition=ContextWriteDisposition.APPLIED,
                reason=ContextSnapshotReason.SNAPSHOT_SEALED,
                success=True,
                message="context snapshot sealed",
                snapshot=sealed,
                queued_writes=self._queued_writes,
            )
            self._record(result)

            return result

    def expire_turn(self) -> ContextSnapshotResult:
        """
        Expire the active turn snapshot.
        """

        with self._lock:
            active = self._active_snapshot

            if active is None:
                result = ContextSnapshotResult(
                    disposition=ContextWriteDisposition.REJECTED,
                    reason=ContextSnapshotReason.NO_ACTIVE_SNAPSHOT,
                    success=False,
                    message="no active snapshot exists",
                    queued_writes=self._queued_writes,
                )
                self._record(result)

                return result

            expired = active.expire()
            self._active_snapshot = None

            result = ContextSnapshotResult(
                disposition=ContextWriteDisposition.APPLIED,
                reason=ContextSnapshotReason.SNAPSHOT_EXPIRED,
                success=True,
                message="context snapshot expired",
                snapshot=expired,
                queued_writes=self._queued_writes,
            )
            self._record(result)

            return result

    def apply_queued_writes(
        self,
        next_context: ConversationTurnContext,
    ) -> tuple[ConversationTurnContext, tuple[PendingContextWrite, ...]]:
        """
        Apply queued writes to a future turn context.

        This never mutates the prior active snapshot.
        """

        with self._lock:
            writes = self._queued_writes
            metadata = dict(next_context.metadata)

            for write in writes:
                metadata[write.key] = write.value

            updated = next_context.model_copy(update={"metadata": metadata})
            self._queued_writes = ()
            self._applied_count += len(writes)

            return updated, writes

    def queued_writes(self) -> tuple[PendingContextWrite, ...]:
        """
        Return queued writes.
        """

        with self._lock:
            return self._queued_writes

    def snapshot(self) -> ContextSnapshotRuntimeSnapshot:
        """
        Return runtime diagnostics.
        """

        with self._lock:
            active = self._active_snapshot

            return ContextSnapshotRuntimeSnapshot(
                name=self.name,
                has_active_snapshot=active is not None,
                active_turn_id=active.turn_id if active is not None else None,
                queued_write_count=len(self._queued_writes),
                snapshot_count=self._snapshot_count,
                queued_count=self._queued_count,
                applied_count=self._applied_count,
                rejected_count=self._rejected_count,
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        """
        Reset runtime state and metrics.
        """

        with self._lock:
            self._active_snapshot = None
            self._queued_writes = ()
            self._snapshot_count = 0
            self._queued_count = 0
            self._applied_count = 0
            self._rejected_count = 0
            self._last_reason = None

    def _record(self, result: ContextSnapshotResult) -> None:
        self._last_reason = result.reason

        if result.reason == ContextSnapshotReason.SNAPSHOT_CREATED:
            self._snapshot_count += 1

        if result.disposition == ContextWriteDisposition.QUEUED:
            self._queued_count += 1

        elif result.disposition == ContextWriteDisposition.APPLIED:
            self._applied_count += 1

        elif result.disposition == ContextWriteDisposition.REJECTED:
            self._rejected_count += 1