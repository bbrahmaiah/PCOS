from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from threading import RLock
from typing import Protocol, runtime_checkable

from pydantic import Field, field_validator

from jarvis.memory.models import (
    MemoryModel,
    MemoryPolicyClassification,
    MemoryRecord,
    MemorySensitivity,
    new_id,
    utc_now,
)
from jarvis.runtime.observability.structured_logger import get_logger


class MemoryEmbedding(MemoryModel):
    """
    Embedding vector for one text input.

    This is a boundary contract. It does not care whether embeddings come from
    a local model, remote model, fake model, or cached backend.
    """

    embedding_id: str = Field(default_factory=new_id)
    text: str
    vector: tuple[float, ...]
    model_name: str = "deterministic_fake_embedding"
    dimensions: int
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("embedding_id", "text", "model_name")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("vector")
    @classmethod
    def _vector_required(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value:
            raise ValueError("vector cannot be empty.")

        return value


class MemoryVectorDocument(MemoryModel):
    """
    A memory record prepared for vector indexing.
    """

    document_id: str = Field(default_factory=new_id)
    memory_id: str
    text: str
    embedding: MemoryEmbedding
    policy_classification: MemoryPolicyClassification
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("document_id", "memory_id", "text")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @classmethod
    def from_record(
        cls,
        *,
        record: MemoryRecord,
        embedding: MemoryEmbedding,
    ) -> MemoryVectorDocument:
        """
        Build vector document from a memory record.
        """

        classification = (
            MemoryPolicyClassification.RESTRICTED
            if record.sensitivity == MemorySensitivity.SENSITIVE
            else MemoryPolicyClassification.ALLOWED
        )

        return cls(
            memory_id=record.memory_id,
            text=record.text,
            embedding=embedding,
            policy_classification=classification,
            metadata={
                "kind": record.kind.value,
                "scope": record.scope.value,
                "source": record.source.value,
                "sensitivity": record.sensitivity.value,
                "importance": record.importance.value,
                "tags": record.tags,
            },
        )


class MemoryVectorSearchQuery(MemoryModel):
    """
    Vector search query.
    """

    query_id: str = Field(default_factory=new_id)
    text: str
    embedding: MemoryEmbedding
    max_results: int = Field(default=8, ge=1, le=100)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    include_restricted: bool = False
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("query_id", "text")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class MemoryVectorSearchResult(MemoryModel):
    """
    One vector search result.
    """

    document: MemoryVectorDocument
    score: float = Field(ge=0.0, le=1.0)
    reason: str
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("reason")
    @classmethod
    def _reason_required(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("reason cannot be empty.")

        return cleaned


class MemoryVectorSearchResponse(MemoryModel):
    """
    Result of one vector search operation.
    """

    query: MemoryVectorSearchQuery
    results: tuple[MemoryVectorSearchResult, ...] = ()
    searched_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @property
    def result_count(self) -> int:
        return len(self.results)


@dataclass(frozen=True, slots=True)
class MemoryEmbeddingProviderSnapshot:
    """
    Diagnostics for embedding provider.
    """

    name: str
    embed_count: int
    last_text_length: int
    last_dimensions: int
    last_error: str | None


@dataclass(frozen=True, slots=True)
class MemoryVectorIndexSnapshot:
    """
    Diagnostics for vector index.
    """

    name: str
    document_count: int
    upsert_count: int
    search_count: int
    delete_count: int
    clear_count: int
    last_document_id: str | None
    last_query_id: str | None
    last_error: str | None


@runtime_checkable
class MemoryEmbeddingProvider(Protocol):
    """
    Embedding provider boundary.

    Implementations can be fake, local model, remote model, or cached.
    """

    @property
    def name(self) -> str:
        """Stable provider name."""

    def embed_text(self, text: str) -> MemoryEmbedding:
        """Embed one text string."""

    def snapshot(self) -> MemoryEmbeddingProviderSnapshot:
        """Return provider diagnostics."""


@runtime_checkable
class MemoryVectorIndex(Protocol):
    """
    Vector index boundary.

    The memory runtime depends on this protocol, not a concrete vector DB.
    """

    @property
    def name(self) -> str:
        """Stable index name."""

    def upsert(self, document: MemoryVectorDocument) -> MemoryVectorDocument:
        """Insert or replace a vector document."""

    def search(self, query: MemoryVectorSearchQuery) -> MemoryVectorSearchResponse:
        """Search vector documents."""

    def delete(self, document_id: str) -> bool:
        """Delete one vector document."""

    def clear(self) -> None:
        """Clear vector documents."""

    def snapshot(self) -> MemoryVectorIndexSnapshot:
        """Return index diagnostics."""


@dataclass(frozen=True, slots=True)
class DeterministicEmbeddingProviderConfig:
    """
    Deterministic fake embedding provider config.

    This is intentionally fake-first. It lets us test vector boundaries without
    depending on models, GPUs, network calls, or vector databases.
    """

    name: str = "deterministic_embedding_provider"
    dimensions: int = 32

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.dimensions <= 0:
            raise ValueError("dimensions must be greater than zero.")


class DeterministicEmbeddingProvider:
    """
    Deterministic embedding provider for tests and local smoke.

    This is not semantic-quality embedding. It is a stable boundary
    implementation so the system architecture can be verified.
    """

    def __init__(
        self,
        *,
        config: DeterministicEmbeddingProviderConfig | None = None,
    ) -> None:
        self._config = config or DeterministicEmbeddingProviderConfig()
        self._config.validate()

        self._lock = RLock()
        self._logger = get_logger("memory.embedding_provider")

        self._embed_count = 0
        self._last_text_length = 0
        self._last_dimensions = self._config.dimensions
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def embed_text(self, text: str) -> MemoryEmbedding:
        cleaned = text.strip()

        if not cleaned:
            with self._lock:
                self._last_error = "text cannot be empty"

            raise ValueError("text cannot be empty.")

        vector = self._vectorize(cleaned)

        with self._lock:
            self._embed_count += 1
            self._last_text_length = len(cleaned)
            self._last_dimensions = len(vector)
            self._last_error = None

        self._logger.info(
            "memory_text_embedded",
            provider=self.name,
            text_length=len(cleaned),
            dimensions=len(vector),
        )

        return MemoryEmbedding(
            text=cleaned,
            vector=vector,
            model_name=self.name,
            dimensions=len(vector),
            metadata={
                "provider": self.name,
            },
        )

    def snapshot(self) -> MemoryEmbeddingProviderSnapshot:
        with self._lock:
            return MemoryEmbeddingProviderSnapshot(
                name=self.name,
                embed_count=self._embed_count,
                last_text_length=self._last_text_length,
                last_dimensions=self._last_dimensions,
                last_error=self._last_error,
            )

    def _vectorize(self, text: str) -> tuple[float, ...]:
        buckets = [0.0 for _ in range(self._config.dimensions)]

        for index, char in enumerate(text.casefold()):
            bucket = index % self._config.dimensions
            buckets[bucket] += (ord(char) % 97) / 97.0

        norm = sqrt(sum(value * value for value in buckets))

        if norm == 0.0:
            return tuple(0.0 for _ in buckets)

        return tuple(value / norm for value in buckets)


@dataclass(frozen=True, slots=True)
class InMemoryVectorIndexConfig:
    """
    Configuration for InMemoryVectorIndex.
    """

    name: str = "in_memory_vector_index"

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")


class InMemoryVectorIndex:
    """
    In-memory vector index implementation.

    Responsibilities:
    - store vector documents
    - search using cosine similarity
    - respect restricted filtering
    - expose diagnostics

    Non-responsibilities:
    - no persistence
    - no embeddings generation
    - no gateway policy
    - no LLM calls
    """

    def __init__(
        self,
        *,
        config: InMemoryVectorIndexConfig | None = None,
    ) -> None:
        self._config = config or InMemoryVectorIndexConfig()
        self._config.validate()

        self._lock = RLock()
        self._logger = get_logger("memory.vector_index")
        self._documents: dict[str, MemoryVectorDocument] = {}

        self._upsert_count = 0
        self._search_count = 0
        self._delete_count = 0
        self._clear_count = 0
        self._last_document_id: str | None = None
        self._last_query_id: str | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def upsert(self, document: MemoryVectorDocument) -> MemoryVectorDocument:
        with self._lock:
            self._documents[document.document_id] = document
            self._upsert_count += 1
            self._last_document_id = document.document_id
            self._last_error = None

        self._logger.info(
            "memory_vector_document_upserted",
            index=self.name,
            document_id=document.document_id,
            memory_id=document.memory_id,
        )

        return document

    def search(self, query: MemoryVectorSearchQuery) -> MemoryVectorSearchResponse:
        with self._lock:
            self._search_count += 1
            self._last_query_id = query.query_id
            self._last_error = None
            documents = tuple(self._documents.values())

        results = []

        for document in documents:
            if (
                document.policy_classification
                == MemoryPolicyClassification.RESTRICTED
                and not query.include_restricted
            ):
                continue

            score = self._cosine_similarity(
                query.embedding.vector,
                document.embedding.vector,
            )

            if score < query.min_score:
                continue

            results.append(
                MemoryVectorSearchResult(
                    document=document,
                    score=score,
                    reason="cosine similarity vector match",
                    metadata={
                        "index": self.name,
                        "query_id": query.query_id,
                    },
                )
            )

        ranked = tuple(
            sorted(results, key=lambda result: result.score, reverse=True)[
                : query.max_results
            ]
        )

        self._logger.info(
            "memory_vector_search_completed",
            index=self.name,
            query_id=query.query_id,
            result_count=len(ranked),
        )

        return MemoryVectorSearchResponse(
            query=query,
            results=ranked,
            metadata={
                "index": self.name,
            },
        )

    def delete(self, document_id: str) -> bool:
        cleaned = document_id.strip()

        if not cleaned:
            return False

        with self._lock:
            removed = self._documents.pop(cleaned, None)

            if removed is None:
                return False

            self._delete_count += 1
            self._last_document_id = cleaned
            self._last_error = None

        self._logger.info(
            "memory_vector_document_deleted",
            index=self.name,
            document_id=cleaned,
        )

        return True

    def clear(self) -> None:
        with self._lock:
            self._documents.clear()
            self._clear_count += 1
            self._last_document_id = None
            self._last_error = None

        self._logger.info("memory_vector_index_cleared", index=self.name)

    def documents(self) -> tuple[MemoryVectorDocument, ...]:
        with self._lock:
            return tuple(self._documents.values())

    def snapshot(self) -> MemoryVectorIndexSnapshot:
        with self._lock:
            return MemoryVectorIndexSnapshot(
                name=self.name,
                document_count=len(self._documents),
                upsert_count=self._upsert_count,
                search_count=self._search_count,
                delete_count=self._delete_count,
                clear_count=self._clear_count,
                last_document_id=self._last_document_id,
                last_query_id=self._last_query_id,
                last_error=self._last_error,
            )

    @staticmethod
    def _cosine_similarity(
        left: tuple[float, ...],
        right: tuple[float, ...],
    ) -> float:
        if len(left) != len(right):
            return 0.0

        left_norm = sqrt(sum(value * value for value in left))
        right_norm = sqrt(sum(value * value for value in right))

        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0

        dot = sum(
            left_value * right_value
            for left_value, right_value in zip(left, right, strict=True)
        )

        similarity = dot / (left_norm * right_norm)

        return max(0.0, min(1.0, similarity))