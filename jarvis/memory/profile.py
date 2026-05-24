from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from typing import Any

from pydantic import Field, field_validator

from jarvis.memory.gateway import (
    MemoryGateway,
    MemoryGatewayRetrievalResult,
    MemoryGatewayWriteResult,
)
from jarvis.memory.models import (
    MemoryImportance,
    MemoryKind,
    MemoryModel,
    MemoryQuery,
    MemoryRetention,
    MemoryScope,
    MemorySensitivity,
    MemorySource,
    MemoryWriteRequest,
)
from jarvis.runtime.observability.structured_logger import get_logger


class UserProfileMemoryCategory(StrEnum):
    """
    Category of stable user profile memory.

    Profile memory should be stable, useful, and safe. It should not become a
    dumping ground for every conversation detail.
    """

    PREFERENCE = "preference"
    GOAL = "goal"
    CONSTRAINT = "constraint"
    PROJECT = "project"
    LEARNING = "learning"
    WORKFLOW = "workflow"
    COMMUNICATION_STYLE = "communication_style"
    SYSTEM_PREFERENCE = "system_preference"


class UserProfileMemoryConfidence(StrEnum):
    """
    Human-readable confidence band for profile memory.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERIFIED = "verified"


class UserProfileMemoryFact(MemoryModel):
    """
    Stable user profile fact.

    This is the profile runtime input contract. It is converted into a governed
    MemoryWriteRequest before storage.
    """

    profile_id: str
    text: str
    category: UserProfileMemoryCategory
    confidence_label: UserProfileMemoryConfidence = UserProfileMemoryConfidence.HIGH
    importance: MemoryImportance = MemoryImportance.HIGH
    sensitivity: MemorySensitivity = MemorySensitivity.PRIVATE
    retention: MemoryRetention = MemoryRetention.PERSISTENT
    source: MemorySource = MemorySource.USER_EXPLICIT
    tags: tuple[str, ...] = ()
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("profile_id", "text")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("tags")
    @classmethod
    def _clean_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(tag.strip().casefold() for tag in value if tag.strip())

        return tuple(dict.fromkeys(cleaned))

    def to_write_request(self) -> MemoryWriteRequest:
        """
        Convert profile fact into a governed memory write request.
        """

        return MemoryWriteRequest(
            kind=MemoryKind.USER_PROFILE,
            scope=MemoryScope.USER,
            text=self.text,
            source=self.source,
            sensitivity=self.sensitivity,
            importance=self.importance,
            retention=self.retention,
            confidence=self.confidence,
            tags=(
                "profile",
                self.category.value,
                self.confidence_label.value,
                *self.tags,
            ),
            metadata={
                **self.metadata,
                "profile_id": self.profile_id,
                "profile_category": self.category.value,
                "confidence_label": self.confidence_label.value,
            },
        )


class UserProfileMemoryQuery(MemoryModel):
    """
    Query contract for user profile memory.
    """

    text: str | None = None
    categories: tuple[UserProfileMemoryCategory, ...] = ()
    confidence_labels: tuple[UserProfileMemoryConfidence, ...] = ()
    tags: tuple[str, ...] = ()
    max_results: int = Field(default=8, ge=1, le=100)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    include_sensitive: bool = False
    include_expired: bool = False
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
        cleaned = tuple(tag.strip().casefold() for tag in value if tag.strip())

        return tuple(dict.fromkeys(cleaned))

    def to_memory_query(self) -> MemoryQuery:
        """
        Convert profile query into a normal MemoryQuery.
        """

        tags = [
            "profile",
            *(category.value for category in self.categories),
            *(label.value for label in self.confidence_labels),
            *self.tags,
        ]

        return MemoryQuery(
            text=self.text,
            kinds=(MemoryKind.USER_PROFILE,),
            scopes=(MemoryScope.USER,),
            tags=tuple(dict.fromkeys(tags)),
            max_results=self.max_results,
            min_confidence=self.min_confidence,
            include_sensitive=self.include_sensitive,
            include_expired=self.include_expired,
            metadata={
                **self.metadata,
                "profile_query": True,
            },
        )


@dataclass(frozen=True, slots=True)
class UserProfileMemoryRuntimeConfig:
    """
    Configuration for UserProfileMemoryRuntime.
    """

    name: str = "user_profile_memory_runtime"
    default_importance: MemoryImportance = MemoryImportance.HIGH
    default_sensitivity: MemorySensitivity = MemorySensitivity.PRIVATE
    default_retention: MemoryRetention = MemoryRetention.PERSISTENT
    default_source: MemorySource = MemorySource.USER_EXPLICIT
    default_confidence_label: UserProfileMemoryConfidence = (
        UserProfileMemoryConfidence.HIGH
    )
    default_confidence: float = 0.9

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.default_confidence < 0.0 or self.default_confidence > 1.0:
            raise ValueError("default_confidence must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class UserProfileMemoryRuntimeSnapshot:
    """
    Observable diagnostics for user profile memory runtime.
    """

    name: str
    saved_count: int
    saved_allowed_count: int
    saved_blocked_count: int
    retrieved_count: int
    last_profile_id: str | None
    last_error: str | None


class UserProfileMemoryRuntime:
    """
    Runtime for stable user profile memory.

    Responsibilities:
    - save stable user preferences, goals, constraints, workflows, and projects
    - convert profile facts into governed memory writes
    - retrieve profile memories through MemoryGateway only
    - preserve retrieval explainability and policy boundaries
    - keep diagnostics

    Non-responsibilities:
    - no direct store access
    - no episodic event history
    - no semantic concept extraction
    - no embeddings
    - no LLM calls
    - no sensitive identity inference
    """

    def __init__(
        self,
        *,
        gateway: MemoryGateway,
        config: UserProfileMemoryRuntimeConfig | None = None,
    ) -> None:
        self._config = config or UserProfileMemoryRuntimeConfig()
        self._config.validate()

        self._gateway = gateway
        self._lock = RLock()
        self._logger = get_logger("memory.user_profile_runtime")

        self._saved_count = 0
        self._saved_allowed_count = 0
        self._saved_blocked_count = 0
        self._retrieved_count = 0
        self._last_profile_id: str | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def save(self, fact: UserProfileMemoryFact) -> MemoryGatewayWriteResult:
        """
        Save one profile fact through the MemoryGateway.
        """

        with self._lock:
            self._saved_count += 1
            self._last_profile_id = fact.profile_id
            self._last_error = None

        result = self._gateway.remember(fact.to_write_request())

        with self._lock:
            if result.allowed:
                self._saved_allowed_count += 1

            else:
                self._saved_blocked_count += 1
                self._last_error = result.reason

        self._logger.info(
            "user_profile_memory_saved",
            runtime=self.name,
            profile_id=fact.profile_id,
            category=fact.category.value,
            allowed=result.allowed,
            blocked=result.blocked,
        )

        return result

    def save_text(
        self,
        text: str,
        *,
        profile_id: str,
        category: UserProfileMemoryCategory,
        confidence_label: UserProfileMemoryConfidence | None = None,
        importance: MemoryImportance | None = None,
        sensitivity: MemorySensitivity | None = None,
        retention: MemoryRetention | None = None,
        source: MemorySource | None = None,
        confidence: float | None = None,
        tags: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> MemoryGatewayWriteResult:
        """
        Convenience method for saving profile memory from text.
        """

        fact = UserProfileMemoryFact(
            profile_id=profile_id,
            text=text,
            category=category,
            confidence_label=confidence_label or self._config.default_confidence_label,
            importance=importance or self._config.default_importance,
            sensitivity=sensitivity or self._config.default_sensitivity,
            retention=retention or self._config.default_retention,
            source=source or self._config.default_source,
            confidence=(
                self._config.default_confidence
                if confidence is None
                else confidence
            ),
            tags=tags,
            metadata=metadata or {},
        )

        return self.save(fact)

    def retrieve(
        self,
        query: UserProfileMemoryQuery,
    ) -> MemoryGatewayRetrievalResult:
        """
        Retrieve user profile memories through the MemoryGateway.
        """

        with self._lock:
            self._retrieved_count += 1
            self._last_error = None

        result = self._gateway.retrieve(query.to_memory_query())

        self._logger.info(
            "user_profile_memory_retrieved",
            runtime=self.name,
            result_count=result.result_count,
            allowed=result.allowed,
            reason=result.reason,
        )

        return result

    def snapshot(self) -> UserProfileMemoryRuntimeSnapshot:
        """
        Return profile runtime diagnostics.
        """

        with self._lock:
            return UserProfileMemoryRuntimeSnapshot(
                name=self.name,
                saved_count=self._saved_count,
                saved_allowed_count=self._saved_allowed_count,
                saved_blocked_count=self._saved_blocked_count,
                retrieved_count=self._retrieved_count,
                last_profile_id=self._last_profile_id,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Reset runtime diagnostics.
        """

        with self._lock:
            self._saved_count = 0
            self._saved_allowed_count = 0
            self._saved_blocked_count = 0
            self._retrieved_count = 0
            self._last_profile_id = None
            self._last_error = None

        self._logger.info("user_profile_memory_runtime_reset", runtime=self.name)