from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from threading import RLock

from pydantic import Field, field_validator

from jarvis.memory.models import (
    MemoryImportance,
    MemoryModel,
    MemoryRecord,
    MemoryRetention,
    MemorySensitivity,
    utc_now,
)
from jarvis.memory.store import MemoryStore
from jarvis.runtime.observability.structured_logger import get_logger


class MemoryLifecycleDecisionKind(StrEnum):
    """
    Lifecycle decision kind for one memory record.
    """

    KEEP = "keep"
    EXPIRE = "expire"
    DELETE = "delete"
    PIN = "pin"
    IGNORE = "ignore"


class MemoryLifecycleReason(StrEnum):
    """
    Stable reason categories for lifecycle decisions.
    """

    PINNED_MEMORY = "pinned_memory"
    CRITICAL_MEMORY = "critical_memory"
    ALREADY_EXPIRED = "already_expired"
    TEMPORARY_EXPIRED = "temporary_expired"
    SESSION_EXPIRED = "session_expired"
    PERSISTENT_ACTIVE = "persistent_active"
    LOW_CONFIDENCE_STALE = "low_confidence_stale"
    SENSITIVE_STALE = "sensitive_stale"
    DEFAULT_KEEP = "default_keep"


class MemoryLifecycleDecision(MemoryModel):
    """
    Lifecycle decision for one memory record.
    """

    memory_id: str
    decision: MemoryLifecycleDecisionKind
    reason: MemoryLifecycleReason
    allowed: bool = True
    delete_recommended: bool = False
    expire_recommended: bool = False
    evaluated_at: datetime = Field(default_factory=utc_now)
    detail: str
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("memory_id", "detail")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class MemoryLifecycleSweepResult(MemoryModel):
    """
    Result of one lifecycle sweep.
    """

    evaluated_count: int
    kept_count: int
    expired_count: int
    deleted_count: int
    ignored_count: int
    decisions: tuple[MemoryLifecycleDecision, ...] = ()
    swept_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @property
    def changed_count(self) -> int:
        return self.expired_count + self.deleted_count


@dataclass(frozen=True, slots=True)
class MemoryLifecyclePolicyConfig:
    """
    Configuration for MemoryLifecyclePolicy.

    Defaults are conservative:
    - pinned and critical memory is kept
    - temporary/session memory can expire
    - deletion requires sweep execution with delete_expired=True
    """

    name: str = "memory_lifecycle_policy"
    temporary_ttl_seconds: int = 60 * 60
    session_ttl_seconds: int = 60 * 60 * 12
    low_confidence_stale_seconds: int = 60 * 60 * 24 * 7
    sensitive_stale_seconds: int = 60 * 60 * 24
    delete_expired: bool = False
    ignore_pinned: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.temporary_ttl_seconds <= 0:
            raise ValueError("temporary_ttl_seconds must be greater than zero.")

        if self.session_ttl_seconds <= 0:
            raise ValueError("session_ttl_seconds must be greater than zero.")

        if self.low_confidence_stale_seconds <= 0:
            raise ValueError("low_confidence_stale_seconds must be greater than zero.")

        if self.sensitive_stale_seconds <= 0:
            raise ValueError("sensitive_stale_seconds must be greater than zero.")


@dataclass(frozen=True, slots=True)
class MemoryLifecyclePolicySnapshot:
    """
    Observable diagnostics for MemoryLifecyclePolicy.
    """

    name: str
    evaluated_count: int
    sweep_count: int
    kept_count: int
    expired_count: int
    deleted_count: int
    ignored_count: int
    last_memory_id: str | None
    last_decision: MemoryLifecycleDecisionKind | None
    last_error: str | None


class MemoryLifecyclePolicy:
    """
    Governs lifecycle decisions for memory records.

    Responsibilities:
    - decide whether records should be kept, expired, pinned, ignored, or deleted
    - protect pinned and critical records
    - identify stale temporary/session/sensitive records
    - produce explainable lifecycle decisions
    - optionally sweep a MemoryStore through the gateway-safe store protocol
    - keep diagnostics

    Non-responsibilities:
    - no summarization
    - no vector compaction
    - no embeddings
    - no LLM calls
    - no cognition logic
    """

    def __init__(
        self,
        *,
        config: MemoryLifecyclePolicyConfig | None = None,
    ) -> None:
        self._config = config or MemoryLifecyclePolicyConfig()
        self._config.validate()

        self._lock = RLock()
        self._logger = get_logger("memory.lifecycle_policy")

        self._evaluated_count = 0
        self._sweep_count = 0
        self._kept_count = 0
        self._expired_count = 0
        self._deleted_count = 0
        self._ignored_count = 0
        self._last_memory_id: str | None = None
        self._last_decision: MemoryLifecycleDecisionKind | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def evaluate(
        self,
        record: MemoryRecord,
        *,
        now: datetime | None = None,
    ) -> MemoryLifecycleDecision:
        """
        Evaluate one memory record.
        """

        current = now or utc_now()
        decision = self._decision_for(record=record, now=current)
        self._record_decision(decision)

        return decision

    def sweep(
        self,
        store: MemoryStore,
        *,
        now: datetime | None = None,
    ) -> MemoryLifecycleSweepResult:
        """
        Evaluate all records in a store and optionally delete expired records.

        This method requires a store implementation that exposes records().
        The in-memory store supports this now; persistent stores can support it
        later through the same method.
        """

        current = now or utc_now()
        records_method = getattr(store, "records", None)

        if records_method is None:
            raise TypeError("memory store does not expose records() for sweeping.")

        records = tuple(records_method())
        decisions: list[MemoryLifecycleDecision] = []
        kept_count = 0
        expired_count = 0
        deleted_count = 0
        ignored_count = 0

        for record in records:
            decision = self.evaluate(record, now=current)
            decisions.append(decision)

            if decision.decision == MemoryLifecycleDecisionKind.IGNORE:
                ignored_count += 1
                continue

            if decision.decision in {
                MemoryLifecycleDecisionKind.KEEP,
                MemoryLifecycleDecisionKind.PIN,
            }:
                kept_count += 1
                continue

            if decision.expire_recommended:
                expired_count += 1

            if decision.delete_recommended and self._config.delete_expired:
                if store.delete(record.memory_id):
                    deleted_count += 1

        with self._lock:
            self._sweep_count += 1

        result = MemoryLifecycleSweepResult(
            evaluated_count=len(records),
            kept_count=kept_count,
            expired_count=expired_count,
            deleted_count=deleted_count,
            ignored_count=ignored_count,
            decisions=tuple(decisions),
            swept_at=current,
            metadata={
                "policy": self.name,
                "delete_expired": self._config.delete_expired,
            },
        )

        self._logger.info(
            "memory_lifecycle_sweep_completed",
            policy=self.name,
            evaluated_count=result.evaluated_count,
            kept_count=result.kept_count,
            expired_count=result.expired_count,
            deleted_count=result.deleted_count,
            ignored_count=result.ignored_count,
        )

        return result

    def snapshot(self) -> MemoryLifecyclePolicySnapshot:
        """
        Return lifecycle diagnostics.
        """

        with self._lock:
            return MemoryLifecyclePolicySnapshot(
                name=self.name,
                evaluated_count=self._evaluated_count,
                sweep_count=self._sweep_count,
                kept_count=self._kept_count,
                expired_count=self._expired_count,
                deleted_count=self._deleted_count,
                ignored_count=self._ignored_count,
                last_memory_id=self._last_memory_id,
                last_decision=self._last_decision,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Reset lifecycle diagnostics.
        """

        with self._lock:
            self._evaluated_count = 0
            self._sweep_count = 0
            self._kept_count = 0
            self._expired_count = 0
            self._deleted_count = 0
            self._ignored_count = 0
            self._last_memory_id = None
            self._last_decision = None
            self._last_error = None

        self._logger.info("memory_lifecycle_policy_reset", policy=self.name)

    def _decision_for(
        self,
        *,
        record: MemoryRecord,
        now: datetime,
    ) -> MemoryLifecycleDecision:
        if self._config.ignore_pinned and record.retention == MemoryRetention.PINNED:
            return MemoryLifecycleDecision(
                memory_id=record.memory_id,
                decision=MemoryLifecycleDecisionKind.PIN,
                reason=MemoryLifecycleReason.PINNED_MEMORY,
                delete_recommended=False,
                expire_recommended=False,
                evaluated_at=now,
                detail="pinned memory is protected from lifecycle expiration",
                metadata={
                    "retention": record.retention.value,
                    "importance": record.importance.value,
                },
            )

        if record.importance == MemoryImportance.CRITICAL:
            return MemoryLifecycleDecision(
                memory_id=record.memory_id,
                decision=MemoryLifecycleDecisionKind.KEEP,
                reason=MemoryLifecycleReason.CRITICAL_MEMORY,
                delete_recommended=False,
                expire_recommended=False,
                evaluated_at=now,
                detail="critical memory is retained",
                metadata={
                    "importance": record.importance.value,
                },
            )

        if record.expired(now=now):
            return MemoryLifecycleDecision(
                memory_id=record.memory_id,
                decision=MemoryLifecycleDecisionKind.DELETE
                if self._config.delete_expired
                else MemoryLifecycleDecisionKind.EXPIRE,
                reason=MemoryLifecycleReason.ALREADY_EXPIRED,
                delete_recommended=self._config.delete_expired,
                expire_recommended=True,
                evaluated_at=now,
                detail="memory record already passed explicit expiration timestamp",
                metadata={
                    "expires_at": record.expires_at.isoformat()
                    if record.expires_at
                    else None,
                },
            )

        age = now - record.updated_at

        if (
            record.retention == MemoryRetention.TEMPORARY
            and age >= timedelta(seconds=self._config.temporary_ttl_seconds)
        ):
            return MemoryLifecycleDecision(
                memory_id=record.memory_id,
                decision=MemoryLifecycleDecisionKind.EXPIRE,
                reason=MemoryLifecycleReason.TEMPORARY_EXPIRED,
                delete_recommended=self._config.delete_expired,
                expire_recommended=True,
                evaluated_at=now,
                detail="temporary memory exceeded lifecycle TTL",
                metadata={
                    "age_seconds": age.total_seconds(),
                    "ttl_seconds": self._config.temporary_ttl_seconds,
                },
            )

        if (
            record.retention == MemoryRetention.SESSION
            and age >= timedelta(seconds=self._config.session_ttl_seconds)
        ):
            return MemoryLifecycleDecision(
                memory_id=record.memory_id,
                decision=MemoryLifecycleDecisionKind.EXPIRE,
                reason=MemoryLifecycleReason.SESSION_EXPIRED,
                delete_recommended=self._config.delete_expired,
                expire_recommended=True,
                evaluated_at=now,
                detail="session memory exceeded lifecycle TTL",
                metadata={
                    "age_seconds": age.total_seconds(),
                    "ttl_seconds": self._config.session_ttl_seconds,
                },
            )

        if (
            record.confidence < 0.3
            and age >= timedelta(seconds=self._config.low_confidence_stale_seconds)
        ):
            return MemoryLifecycleDecision(
                memory_id=record.memory_id,
                decision=MemoryLifecycleDecisionKind.EXPIRE,
                reason=MemoryLifecycleReason.LOW_CONFIDENCE_STALE,
                delete_recommended=self._config.delete_expired,
                expire_recommended=True,
                evaluated_at=now,
                detail="low-confidence memory became stale",
                metadata={
                    "confidence": record.confidence,
                    "age_seconds": age.total_seconds(),
                },
            )

        if (
            record.sensitivity == MemorySensitivity.SENSITIVE
            and age >= timedelta(seconds=self._config.sensitive_stale_seconds)
        ):
            return MemoryLifecycleDecision(
                memory_id=record.memory_id,
                decision=MemoryLifecycleDecisionKind.EXPIRE,
                reason=MemoryLifecycleReason.SENSITIVE_STALE,
                delete_recommended=self._config.delete_expired,
                expire_recommended=True,
                evaluated_at=now,
                detail="sensitive memory exceeded conservative lifecycle window",
                metadata={
                    "sensitivity": record.sensitivity.value,
                    "age_seconds": age.total_seconds(),
                },
            )

        if record.retention == MemoryRetention.PERSISTENT:
            return MemoryLifecycleDecision(
                memory_id=record.memory_id,
                decision=MemoryLifecycleDecisionKind.KEEP,
                reason=MemoryLifecycleReason.PERSISTENT_ACTIVE,
                delete_recommended=False,
                expire_recommended=False,
                evaluated_at=now,
                detail="persistent memory remains active",
                metadata={
                    "retention": record.retention.value,
                },
            )

        return MemoryLifecycleDecision(
            memory_id=record.memory_id,
            decision=MemoryLifecycleDecisionKind.KEEP,
            reason=MemoryLifecycleReason.DEFAULT_KEEP,
            delete_recommended=False,
            expire_recommended=False,
            evaluated_at=now,
            detail="memory remains within lifecycle policy",
            metadata={
                "retention": record.retention.value,
            },
        )

    def _record_decision(self, decision: MemoryLifecycleDecision) -> None:
        with self._lock:
            self._evaluated_count += 1
            self._last_memory_id = decision.memory_id
            self._last_decision = decision.decision
            self._last_error = None

            if decision.decision == MemoryLifecycleDecisionKind.IGNORE:
                self._ignored_count += 1

            elif decision.decision == MemoryLifecycleDecisionKind.DELETE:
                self._deleted_count += 1

            elif decision.decision == MemoryLifecycleDecisionKind.EXPIRE:
                self._expired_count += 1

            elif decision.decision in {
                MemoryLifecycleDecisionKind.KEEP,
                MemoryLifecycleDecisionKind.PIN,
            }:
                self._kept_count += 1

        self._logger.info(
            "memory_lifecycle_decision_evaluated",
            policy=self.name,
            memory_id=decision.memory_id,
            decision=decision.decision.value,
            reason=decision.reason.value,
            delete_recommended=decision.delete_recommended,
            expire_recommended=decision.expire_recommended,
        )