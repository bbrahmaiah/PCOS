from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    """
    Return timezone-aware UTC timestamp.
    """

    return datetime.now(UTC)


def new_id() -> str:
    """
    Return compact stable id for memory records and queries.
    """

    return uuid4().hex


class MemoryModel(BaseModel):
    """
    Base model for Phase 4 memory runtime.

    Memory contracts are frozen, strict, and explicit. Memory is not hidden
    prompt stuffing; it is governed runtime infrastructure.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )


class MemoryKind(StrEnum):
    """
    High-level memory type.
    """

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    USER_PROFILE = "user_profile"
    PROCEDURAL = "procedural"
    PROJECT = "project"
    PREFERENCE = "preference"
    SYSTEM = "system"


class MemoryScope(StrEnum):
    """
    Scope of a memory record.
    """

    SESSION = "session"
    USER = "user"
    PROJECT = "project"
    SYSTEM = "system"


class MemorySensitivity(StrEnum):
    """
    Privacy/sensitivity classification.
    """

    PUBLIC = "public"
    INTERNAL = "internal"
    PRIVATE = "private"
    SENSITIVE = "sensitive"


class MemoryImportance(StrEnum):
    """
    Importance for retrieval and retention.
    """

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class MemorySource(StrEnum):
    """
    Source that created the memory.
    """

    USER_EXPLICIT = "user_explicit"
    CONVERSATION = "conversation"
    COGNITION = "cognition"
    TOOL_RESULT = "tool_result"
    SYSTEM = "system"


class MemoryRetention(StrEnum):
    """
    Retention behavior for a memory record.
    """

    TEMPORARY = "temporary"
    SESSION = "session"
    PERSISTENT = "persistent"
    PINNED = "pinned"


class MemoryPolicyClassification(StrEnum):
    """
    Policy classification for memory access and retrieval.

    This is intentionally separate from sensitivity. Sensitivity describes the
    data. Policy classification describes what the runtime decided.
    """

    ALLOWED = "allowed"
    RESTRICTED = "restricted"
    REDACTED = "redacted"
    BLOCKED = "blocked"


class MemoryRecord(MemoryModel):
    """
    One typed memory record.

    This is the central storage-independent memory contract. It is not coupled
    to SQLite, vectors, files, or any specific database.
    """

    memory_id: str = Field(default_factory=new_id)
    kind: MemoryKind
    scope: MemoryScope = MemoryScope.USER
    text: str
    source: MemorySource = MemorySource.CONVERSATION
    sensitivity: MemorySensitivity = MemorySensitivity.PRIVATE
    importance: MemoryImportance = MemoryImportance.NORMAL
    retention: MemoryRetention = MemoryRetention.PERSISTENT
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("memory_id", "text")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("tags")
    @classmethod
    def _clean_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned_tags = tuple(
            tag.strip().casefold()
            for tag in value
            if tag.strip()
        )

        return tuple(dict.fromkeys(cleaned_tags))

    def expired(self, *, now: datetime | None = None) -> bool:
        """
        Return True if this memory is expired.
        """

        if self.expires_at is None:
            return False

        return self.expires_at <= (now or utc_now())


class MemoryQuery(MemoryModel):
    """
    Typed memory retrieval query.
    """

    query_id: str = Field(default_factory=new_id)
    text: str | None = None
    kinds: tuple[MemoryKind, ...] = ()
    scopes: tuple[MemoryScope, ...] = ()
    tags: tuple[str, ...] = ()
    max_results: int = Field(default=8, ge=1, le=100)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    include_expired: bool = False
    include_sensitive: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def _clean_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None

    @field_validator("tags")
    @classmethod
    def _clean_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned_tags = tuple(
            tag.strip().casefold()
            for tag in value
            if tag.strip()
        )

        return tuple(dict.fromkeys(cleaned_tags))


class MemoryRetrievalExplanation(MemoryModel):
    """
    Explainability contract for one retrieved memory.

    Global Phase 4 rule:
    every retrieval must include source, reason, confidence, timestamp, and
    policy classification.
    """

    source: MemorySource
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    retrieved_at: datetime = Field(default_factory=utc_now)
    policy_classification: MemoryPolicyClassification = (
        MemoryPolicyClassification.ALLOWED
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("reason")
    @classmethod
    def _reason_required(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("reason cannot be empty.")

        return cleaned


class MemorySearchResult(MemoryModel):
    """
    One ranked memory search result.

    The explanation field is mandatory so memory retrieval never becomes magic.
    """

    record: MemoryRecord
    score: float = Field(ge=0.0, le=1.0)
    explanation: MemoryRetrievalExplanation
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def reason(self) -> str:
        return self.explanation.reason

    @property
    def policy_classification(self) -> MemoryPolicyClassification:
        return self.explanation.policy_classification


class MemoryRetrievalResult(MemoryModel):
    """
    Result of one memory retrieval operation.
    """

    query: MemoryQuery
    results: tuple[MemorySearchResult, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def result_count(self) -> int:
        return len(self.results)

    @property
    def records(self) -> tuple[MemoryRecord, ...]:
        return tuple(result.record for result in self.results)


class MemoryWriteRequest(MemoryModel):
    """
    Request to write a memory.

    Later policy layers decide whether this write is allowed, blocked,
    downgraded to temporary, or requires confirmation.
    """

    request_id: str = Field(default_factory=new_id)
    kind: MemoryKind
    text: str
    scope: MemoryScope = MemoryScope.USER
    source: MemorySource = MemorySource.CONVERSATION
    sensitivity: MemorySensitivity = MemorySensitivity.PRIVATE
    importance: MemoryImportance = MemoryImportance.NORMAL
    retention: MemoryRetention = MemoryRetention.PERSISTENT
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("request_id", "text")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("tags")
    @classmethod
    def _clean_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned_tags = tuple(
            tag.strip().casefold()
            for tag in value
            if tag.strip()
        )

        return tuple(dict.fromkeys(cleaned_tags))

    def to_record(self) -> MemoryRecord:
        """
        Convert write request into MemoryRecord.

        Policy approval should happen before this conversion in later steps.
        """

        return MemoryRecord(
            kind=self.kind,
            scope=self.scope,
            text=self.text,
            source=self.source,
            sensitivity=self.sensitivity,
            importance=self.importance,
            retention=self.retention,
            confidence=self.confidence,
            tags=self.tags,
            metadata={
                **self.metadata,
                "write_request_id": self.request_id,
            },
        )