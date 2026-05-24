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


class SemanticMemoryDomain(StrEnum):
    """
    Domain/category for semantic memory.

    Semantic memory stores stable knowledge, concepts, project facts, and
    reusable understanding. It should not store event timelines.
    """

    GENERAL = "general"
    USER = "user"
    PROJECT = "project"
    SYSTEM = "system"
    ENGINEERING = "engineering"
    EDUCATION = "education"
    DEBUGGING = "debugging"
    RESEARCH = "research"


class SemanticMemoryFactType(StrEnum):
    """
    Type of semantic fact.
    """

    FACT = "fact"
    CONCEPT = "concept"
    RULE = "rule"
    SUMMARY = "summary"
    DEFINITION = "definition"
    PREFERENCE = "preference"
    CAPABILITY = "capability"
    CONSTRAINT = "constraint"


class SemanticMemoryFact(MemoryModel):
    """
    Stable knowledge item intended for semantic memory.

    This is the semantic runtime input contract. It is converted into a governed
    MemoryWriteRequest before storage.
    """

    fact_id: str
    text: str
    domain: SemanticMemoryDomain = SemanticMemoryDomain.GENERAL
    fact_type: SemanticMemoryFactType = SemanticMemoryFactType.FACT
    importance: MemoryImportance = MemoryImportance.NORMAL
    sensitivity: MemorySensitivity = MemorySensitivity.PRIVATE
    retention: MemoryRetention = MemoryRetention.PERSISTENT
    source: MemorySource = MemorySource.COGNITION
    tags: tuple[str, ...] = ()
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("fact_id", "text")
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
        Convert semantic fact into a governed memory write request.
        """

        return MemoryWriteRequest(
            kind=MemoryKind.SEMANTIC,
            scope=self._scope_for_domain(),
            text=self.text,
            source=self.source,
            sensitivity=self.sensitivity,
            importance=self.importance,
            retention=self.retention,
            confidence=self.confidence,
            tags=(
                "semantic",
                self.domain.value,
                self.fact_type.value,
                *self.tags,
            ),
            metadata={
                **self.metadata,
                "fact_id": self.fact_id,
                "semantic_domain": self.domain.value,
                "fact_type": self.fact_type.value,
            },
        )

    def _scope_for_domain(self) -> MemoryScope:
        if self.domain == SemanticMemoryDomain.PROJECT:
            return MemoryScope.PROJECT

        if self.domain == SemanticMemoryDomain.SYSTEM:
            return MemoryScope.SYSTEM

        return MemoryScope.USER


class SemanticMemoryQuery(MemoryModel):
    """
    Query contract for semantic memory.
    """

    text: str | None = None
    domains: tuple[SemanticMemoryDomain, ...] = ()
    fact_types: tuple[SemanticMemoryFactType, ...] = ()
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
        Convert semantic query into a normal MemoryQuery.
        """

        tags = [
            "semantic",
            *(domain.value for domain in self.domains),
            *(fact_type.value for fact_type in self.fact_types),
            *self.tags,
        ]

        return MemoryQuery(
            text=self.text,
            kinds=(MemoryKind.SEMANTIC,),
            tags=tuple(dict.fromkeys(tags)),
            max_results=self.max_results,
            min_confidence=self.min_confidence,
            include_sensitive=self.include_sensitive,
            include_expired=self.include_expired,
            metadata={
                **self.metadata,
                "semantic_query": True,
            },
        )


@dataclass(frozen=True, slots=True)
class SemanticMemoryRuntimeConfig:
    """
    Configuration for SemanticMemoryRuntime.
    """

    name: str = "semantic_memory_runtime"
    default_domain: SemanticMemoryDomain = SemanticMemoryDomain.GENERAL
    default_fact_type: SemanticMemoryFactType = SemanticMemoryFactType.FACT
    default_importance: MemoryImportance = MemoryImportance.NORMAL
    default_sensitivity: MemorySensitivity = MemorySensitivity.PRIVATE
    default_retention: MemoryRetention = MemoryRetention.PERSISTENT
    default_source: MemorySource = MemorySource.COGNITION

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")


@dataclass(frozen=True, slots=True)
class SemanticMemoryRuntimeSnapshot:
    """
    Observable diagnostics for semantic memory runtime.
    """

    name: str
    learned_count: int
    learned_allowed_count: int
    learned_blocked_count: int
    retrieved_count: int
    last_fact_id: str | None
    last_error: str | None


class SemanticMemoryRuntime:
    """
    Runtime for semantic memory.

    Responsibilities:
    - learn stable facts/concepts/rules/summaries
    - convert semantic facts into governed memory writes
    - retrieve semantic memory through MemoryGateway only
    - preserve retrieval explainability and policy boundaries
    - keep diagnostics

    Non-responsibilities:
    - no direct store access
    - no embeddings
    - no summarization generation
    - no episodic timeline storage
    - no user profile authority
    - no LLM calls
    """

    def __init__(
        self,
        *,
        gateway: MemoryGateway,
        config: SemanticMemoryRuntimeConfig | None = None,
    ) -> None:
        self._config = config or SemanticMemoryRuntimeConfig()
        self._config.validate()

        self._gateway = gateway
        self._lock = RLock()
        self._logger = get_logger("memory.semantic_runtime")

        self._learned_count = 0
        self._learned_allowed_count = 0
        self._learned_blocked_count = 0
        self._retrieved_count = 0
        self._last_fact_id: str | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def learn(self, fact: SemanticMemoryFact) -> MemoryGatewayWriteResult:
        """
        Learn one semantic fact through the MemoryGateway.
        """

        with self._lock:
            self._learned_count += 1
            self._last_fact_id = fact.fact_id
            self._last_error = None

        result = self._gateway.remember(fact.to_write_request())

        with self._lock:
            if result.allowed:
                self._learned_allowed_count += 1

            else:
                self._learned_blocked_count += 1
                self._last_error = result.reason

        self._logger.info(
            "semantic_memory_fact_learned",
            runtime=self.name,
            fact_id=fact.fact_id,
            domain=fact.domain.value,
            fact_type=fact.fact_type.value,
            allowed=result.allowed,
            blocked=result.blocked,
        )

        return result

    def learn_text(
        self,
        text: str,
        *,
        fact_id: str,
        domain: SemanticMemoryDomain | None = None,
        fact_type: SemanticMemoryFactType | None = None,
        importance: MemoryImportance | None = None,
        sensitivity: MemorySensitivity | None = None,
        retention: MemoryRetention | None = None,
        source: MemorySource | None = None,
        confidence: float = 1.0,
        tags: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> MemoryGatewayWriteResult:
        """
        Convenience method for learning a semantic fact from text.
        """

        fact = SemanticMemoryFact(
            fact_id=fact_id,
            text=text,
            domain=domain or self._config.default_domain,
            fact_type=fact_type or self._config.default_fact_type,
            importance=importance or self._config.default_importance,
            sensitivity=sensitivity or self._config.default_sensitivity,
            retention=retention or self._config.default_retention,
            source=source or self._config.default_source,
            confidence=confidence,
            tags=tags,
            metadata=metadata or {},
        )

        return self.learn(fact)

    def retrieve(
        self,
        query: SemanticMemoryQuery,
    ) -> MemoryGatewayRetrievalResult:
        """
        Retrieve semantic memories through the MemoryGateway.
        """

        with self._lock:
            self._retrieved_count += 1
            self._last_error = None

        result = self._gateway.retrieve(query.to_memory_query())

        self._logger.info(
            "semantic_memory_retrieved",
            runtime=self.name,
            result_count=result.result_count,
            allowed=result.allowed,
            reason=result.reason,
        )

        return result

    def snapshot(self) -> SemanticMemoryRuntimeSnapshot:
        """
        Return semantic runtime diagnostics.
        """

        with self._lock:
            return SemanticMemoryRuntimeSnapshot(
                name=self.name,
                learned_count=self._learned_count,
                learned_allowed_count=self._learned_allowed_count,
                learned_blocked_count=self._learned_blocked_count,
                retrieved_count=self._retrieved_count,
                last_fact_id=self._last_fact_id,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Reset runtime diagnostics.
        """

        with self._lock:
            self._learned_count = 0
            self._learned_allowed_count = 0
            self._learned_blocked_count = 0
            self._retrieved_count = 0
            self._last_fact_id = None
            self._last_error = None

        self._logger.info("semantic_memory_runtime_reset", runtime=self.name)