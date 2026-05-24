from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Protocol, runtime_checkable

from jarvis.memory.models import (
    MemoryImportance,
    MemoryPolicyClassification,
    MemoryQuery,
    MemoryRecord,
    MemoryRetrievalExplanation,
    MemoryRetrievalResult,
    MemorySearchResult,
    MemorySensitivity,
    MemoryWriteRequest,
    utc_now,
)
from jarvis.runtime.observability.structured_logger import get_logger


@dataclass(frozen=True, slots=True)
class MemoryStoreSnapshot:
    """
    Observable diagnostics for a memory store.
    """

    name: str
    record_count: int
    active_record_count: int
    expired_record_count: int
    write_count: int
    retrieve_count: int
    delete_count: int
    clear_count: int
    last_memory_id: str | None
    last_query_id: str | None
    last_error: str | None


@runtime_checkable
class MemoryStore(Protocol):
    """
    Storage-independent memory store contract.

    All stores must return explainable retrieval results. Cognition should not
    depend on concrete stores directly; a later MemoryGateway will own access.
    """

    @property
    def name(self) -> str:
        """Stable store name."""

    def write(self, request: MemoryWriteRequest) -> MemoryRecord:
        """Write one memory record."""

    def put(self, record: MemoryRecord) -> MemoryRecord:
        """Store an already-created memory record."""

    def get(self, memory_id: str) -> MemoryRecord | None:
        """Get one memory record by id."""

    def retrieve(self, query: MemoryQuery) -> MemoryRetrievalResult:
        """Retrieve ranked, explainable memory results."""

    def delete(self, memory_id: str) -> bool:
        """Delete one memory record by id."""

    def clear(self) -> None:
        """Clear all memory records."""

    def snapshot(self) -> MemoryStoreSnapshot:
        """Return store diagnostics."""


@dataclass(frozen=True, slots=True)
class InMemoryMemoryStoreConfig:
    """
    Configuration for InMemoryMemoryStore.
    """

    name: str = "in_memory_memory_store"
    max_records: int = 1_000

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.max_records <= 0:
            raise ValueError("max_records must be greater than zero.")


class InMemoryMemoryStore:
    """
    Thread-safe in-memory implementation of MemoryStore.

    Responsibilities:
    - store typed MemoryRecord objects
    - retrieve records by query constraints
    - rank results deterministically
    - return retrieval explanations for every result
    - enforce bounded capacity
    - expose diagnostics

    Non-responsibilities:
    - no persistence
    - no embeddings
    - no vector search
    - no cognition access control
    - no direct LLM access
    """

    def __init__(
        self,
        *,
        config: InMemoryMemoryStoreConfig | None = None,
    ) -> None:
        self._config = config or InMemoryMemoryStoreConfig()
        self._config.validate()

        self._lock = RLock()
        self._logger = get_logger("memory.in_memory_store")
        self._records: dict[str, MemoryRecord] = {}

        self._write_count = 0
        self._retrieve_count = 0
        self._delete_count = 0
        self._clear_count = 0
        self._last_memory_id: str | None = None
        self._last_query_id: str | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def write(self, request: MemoryWriteRequest) -> MemoryRecord:
        """
        Convert a write request into a record and store it.

        Later policy layers should approve or reject writes before this method.
        """

        record = request.to_record()

        return self.put(record)

    def put(self, record: MemoryRecord) -> MemoryRecord:
        """
        Store an already-created memory record.
        """

        with self._lock:
            self._records[record.memory_id] = record
            self._write_count += 1
            self._last_memory_id = record.memory_id
            self._last_error = None
            self._evict_if_needed_locked()

        self._logger.info(
            "memory_record_stored",
            store=self.name,
            memory_id=record.memory_id,
            kind=record.kind.value,
            scope=record.scope.value,
            sensitivity=record.sensitivity.value,
            importance=record.importance.value,
        )

        return record

    def get(self, memory_id: str) -> MemoryRecord | None:
        """
        Get one memory record by id.
        """

        cleaned = memory_id.strip()

        if not cleaned:
            return None

        with self._lock:
            return self._records.get(cleaned)

    def retrieve(self, query: MemoryQuery) -> MemoryRetrievalResult:
        """
        Retrieve ranked records matching the query.

        Every result includes:
        - source
        - reason
        - confidence
        - timestamp
        - policy classification
        """

        with self._lock:
            self._retrieve_count += 1
            self._last_query_id = query.query_id
            self._last_error = None

            candidates = tuple(
                record
                for record in self._records.values()
                if self._matches_query(record, query)
            )

            results = tuple(
                sorted(
                    (
                        self._to_search_result(record=record, query=query)
                        for record in candidates
                    ),
                    key=lambda result: result.score,
                    reverse=True,
                )[: query.max_results]
            )

        self._logger.info(
            "memory_records_retrieved",
            store=self.name,
            query_id=query.query_id,
            query_text=query.text,
            result_count=len(results),
        )

        return MemoryRetrievalResult(
            query=query,
            results=results,
            metadata={
                "store": self.name,
            },
        )

    def delete(self, memory_id: str) -> bool:
        """
        Delete one memory record by id.
        """

        cleaned = memory_id.strip()

        if not cleaned:
            return False

        with self._lock:
            removed = self._records.pop(cleaned, None)

            if removed is None:
                return False

            self._delete_count += 1
            self._last_memory_id = cleaned
            self._last_error = None

        self._logger.info(
            "memory_record_deleted",
            store=self.name,
            memory_id=cleaned,
        )

        return True

    def clear(self) -> None:
        """
        Clear all memory records.
        """

        with self._lock:
            self._records.clear()
            self._clear_count += 1
            self._last_memory_id = None
            self._last_error = None

        self._logger.info("memory_store_cleared", store=self.name)

    def records(self) -> tuple[MemoryRecord, ...]:
        """
        Return immutable copy of all stored records.
        """

        with self._lock:
            return tuple(self._records.values())

    def snapshot(self) -> MemoryStoreSnapshot:
        """
        Return store diagnostics.
        """

        with self._lock:
            active_count = sum(
                1 for record in self._records.values() if not record.expired()
            )
            expired_count = len(self._records) - active_count

            return MemoryStoreSnapshot(
                name=self.name,
                record_count=len(self._records),
                active_record_count=active_count,
                expired_record_count=expired_count,
                write_count=self._write_count,
                retrieve_count=self._retrieve_count,
                delete_count=self._delete_count,
                clear_count=self._clear_count,
                last_memory_id=self._last_memory_id,
                last_query_id=self._last_query_id,
                last_error=self._last_error,
            )

    def _matches_query(
        self,
        record: MemoryRecord,
        query: MemoryQuery,
    ) -> bool:
        if not query.include_expired and record.expired():
            return False

        if not query.include_sensitive and (
            record.sensitivity == MemorySensitivity.SENSITIVE
        ):
            return False

        if record.confidence < query.min_confidence:
            return False

        if query.kinds and record.kind not in query.kinds:
            return False

        if query.scopes and record.scope not in query.scopes:
            return False

        if query.tags and not set(query.tags).issubset(set(record.tags)):
            return False

        if query.text is None:
            return True

        return self._text_relevance_score(
            query_text=query.text,
            record_text=record.text,
        ) > 0.0

    def _to_search_result(
        self,
        *,
        record: MemoryRecord,
        query: MemoryQuery,
    ) -> MemorySearchResult:
        score = self._score(record=record, query=query)
        explanation = MemoryRetrievalExplanation(
            source=record.source,
            reason=self._reason(record=record, query=query, score=score),
            confidence=min(record.confidence, score),
            retrieved_at=utc_now(),
            policy_classification=self._policy_classification(record),
            metadata={
                "store": self.name,
                "memory_id": record.memory_id,
                "query_id": query.query_id,
                "importance": record.importance.value,
                "sensitivity": record.sensitivity.value,
            },
        )

        return MemorySearchResult(
            record=record,
            score=score,
            explanation=explanation,
            metadata={
                "store": self.name,
            },
        )

    def _score(
        self,
        *,
        record: MemoryRecord,
        query: MemoryQuery,
    ) -> float:
        text_score = (
            1.0
            if query.text is None
            else self._text_relevance_score(
                query_text=query.text,
                record_text=record.text,
            )
        )
        importance_score = self._importance_score(record.importance)
        confidence_score = record.confidence
        tag_score = self._tag_score(record=record, query=query)

        weighted = (
            text_score * 0.45
            + importance_score * 0.25
            + confidence_score * 0.20
            + tag_score * 0.10
        )

        return max(0.0, min(1.0, weighted))

    def _reason(
        self,
        *,
        record: MemoryRecord,
        query: MemoryQuery,
        score: float,
    ) -> str:
        reasons: list[str] = []

        if query.text is not None:
            overlap = self._term_overlap(query.text, record.text)

            if overlap:
                reasons.append(
                    "matched query terms: " + ", ".join(sorted(overlap))
                )

        if query.kinds:
            reasons.append(f"matched memory kind: {record.kind.value}")

        if query.scopes:
            reasons.append(f"matched memory scope: {record.scope.value}")

        if query.tags:
            reasons.append("matched tags: " + ", ".join(query.tags))

        reasons.append(f"importance={record.importance.value}")
        reasons.append(f"score={score:.2f}")

        return "; ".join(reasons)

    @staticmethod
    def _policy_classification(
        record: MemoryRecord,
    ) -> MemoryPolicyClassification:
        if record.sensitivity == MemorySensitivity.SENSITIVE:
            return MemoryPolicyClassification.RESTRICTED

        if record.sensitivity == MemorySensitivity.PRIVATE:
            return MemoryPolicyClassification.ALLOWED

        if record.sensitivity == MemorySensitivity.INTERNAL:
            return MemoryPolicyClassification.ALLOWED

        return MemoryPolicyClassification.ALLOWED

    def _evict_if_needed_locked(self) -> None:
        if len(self._records) <= self._config.max_records:
            return

        overflow = len(self._records) - self._config.max_records
        evictable = sorted(
            self._records.values(),
            key=lambda record: (
                self._importance_score(record.importance),
                record.created_at,
            ),
        )

        for record in evictable[:overflow]:
            self._records.pop(record.memory_id, None)

    @staticmethod
    def _text_relevance_score(
        *,
        query_text: str,
        record_text: str,
    ) -> float:
        query_terms = InMemoryMemoryStore._terms(query_text)
        record_terms = InMemoryMemoryStore._terms(record_text)

        if not query_terms:
            return 1.0

        overlap = query_terms & record_terms

        if not overlap:
            return 0.0

        return len(overlap) / len(query_terms)

    @staticmethod
    def _term_overlap(
        query_text: str,
        record_text: str,
    ) -> set[str]:
        return InMemoryMemoryStore._terms(query_text) & InMemoryMemoryStore._terms(
            record_text
        )

    @staticmethod
    def _tag_score(
        *,
        record: MemoryRecord,
        query: MemoryQuery,
    ) -> float:
        if not query.tags:
            return 0.0

        query_tags = set(query.tags)
        record_tags = set(record.tags)

        if not query_tags:
            return 0.0

        return len(query_tags & record_tags) / len(query_tags)

    @staticmethod
    def _importance_score(importance: MemoryImportance) -> float:
        scores = {
            MemoryImportance.LOW: 0.25,
            MemoryImportance.NORMAL: 0.5,
            MemoryImportance.HIGH: 0.8,
            MemoryImportance.CRITICAL: 1.0,
        }

        return scores[importance]

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {
            term.strip(".,!?;:()[]{}\"'").casefold()
            for term in text.split()
            if term.strip(".,!?;:()[]{}\"'")
        }