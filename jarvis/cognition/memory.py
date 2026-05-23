from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from threading import RLock
from typing import Any, Protocol, runtime_checkable

from pydantic import Field, field_validator

from jarvis.cognition.models import (
    CognitionContext,
    CognitionContextItem,
    CognitionModel,
    CognitionRequest,
    new_id,
)
from jarvis.runtime.observability.structured_logger import get_logger


class ShortTermMemoryKind(StrEnum):
    """
    Type of short-term memory item.

    Short-term memory is session/runtime memory, not permanent long-term memory.
    """

    SESSION_FACT = "session_fact"
    USER_PREFERENCE = "user_preference"
    ACTIVE_GOAL = "active_goal"
    RECENT_DECISION = "recent_decision"
    PROJECT_CONTEXT = "project_context"
    ERROR_CONTEXT = "error_context"
    SYSTEM_NOTE = "system_note"


class ShortTermMemoryPriority(StrEnum):
    """
    Retrieval priority for short-term memory.
    """

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


def utc_now() -> datetime:
    return datetime.now(UTC)


class ShortTermMemoryItem(CognitionModel):
    """
    One short-term memory item.

    This is intentionally small, typed, and bounded. It is safe to inject into
    CognitionContext later without giving cognition access to storage internals.
    """

    memory_id: str = Field(default_factory=new_id)
    kind: ShortTermMemoryKind = ShortTermMemoryKind.SESSION_FACT
    text: str
    source: str = "cognition"
    priority: ShortTermMemoryPriority = ShortTermMemoryPriority.NORMAL
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("memory_id", "text", "source")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    def expired(
        self,
        *,
        now: datetime | None = None,
    ) -> bool:
        if self.expires_at is None:
            return False

        current = now or utc_now()

        return self.expires_at <= current


class ShortTermMemoryQuery(CognitionModel):
    """
    Query used to retrieve useful short-term memory.
    """

    query_text: str | None = None
    kinds: tuple[ShortTermMemoryKind, ...] = ()
    max_items: int = Field(default=8, ge=1, le=50)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    include_expired: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("query_text")
    @classmethod
    def _query_text_optional_clean(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None


class ShortTermMemoryResult(CognitionModel):
    """
    Result returned from short-term memory retrieval.
    """

    query: ShortTermMemoryQuery
    items: tuple[ShortTermMemoryItem, ...] = ()
    source: str = "short_term_memory"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def item_count(self) -> int:
        return len(self.items)


@dataclass(frozen=True, slots=True)
class ShortTermMemorySnapshot:
    """
    Observable short-term memory diagnostics.
    """

    name: str
    item_count: int
    active_item_count: int
    expired_item_count: int
    remembered_count: int
    retrieved_count: int
    forgotten_count: int
    cleared_count: int
    last_memory_id: str | None
    last_query_text: str | None
    last_error: str | None


@runtime_checkable
class ShortTermMemoryStore(Protocol):
    """
    Short-term memory store contract.

    Future stores can be in-memory, SQLite, Redis, or hybrid. Cognition should
    depend on this protocol, not concrete storage.
    """

    @property
    def name(self) -> str:
        """Stable store name."""

    def remember(self, item: ShortTermMemoryItem) -> ShortTermMemoryItem:
        """Store one memory item."""

    def retrieve(self, query: ShortTermMemoryQuery) -> ShortTermMemoryResult:
        """Retrieve useful memory items."""

    def forget(self, memory_id: str) -> bool:
        """Forget one memory item by id."""

    def clear(self) -> None:
        """Clear all short-term memory."""

    def snapshot(self) -> ShortTermMemorySnapshot:
        """Return store diagnostics."""


@dataclass(frozen=True, slots=True)
class InMemoryShortTermMemoryConfig:
    """
    Configuration for InMemoryShortTermMemoryStore.
    """

    name: str = "in_memory_short_term_memory"
    max_items: int = 128
    max_context_items: int = 8
    max_context_item_chars: int = 700

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.max_items <= 0:
            raise ValueError("max_items must be greater than zero.")

        if self.max_context_items <= 0:
            raise ValueError("max_context_items must be greater than zero.")

        if self.max_context_item_chars <= 0:
            raise ValueError("max_context_item_chars must be greater than zero.")


class InMemoryShortTermMemoryStore:
    """
    Thread-safe in-memory short-term memory store.

    Responsibilities:
    - store bounded temporary memory
    - retrieve relevant items by kind, confidence, and lightweight text match
    - expose CognitionContext for prompt enrichment
    - backfill context with recent/high-priority memory when useful
    - remove expired items safely
    - keep diagnostics

    Non-responsibilities:
    - no vector search
    - no permanent persistence
    - no LLM calls
    - no tool execution
    """

    def __init__(
        self,
        *,
        config: InMemoryShortTermMemoryConfig | None = None,
    ) -> None:
        self._config = config or InMemoryShortTermMemoryConfig()
        self._config.validate()

        self._lock = RLock()
        self._logger = get_logger("cognition.short_term_memory")
        self._items: dict[str, ShortTermMemoryItem] = {}

        self._remembered_count = 0
        self._retrieved_count = 0
        self._forgotten_count = 0
        self._cleared_count = 0
        self._last_memory_id: str | None = None
        self._last_query_text: str | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def remember(self, item: ShortTermMemoryItem) -> ShortTermMemoryItem:
        """
        Store one memory item and enforce bounded capacity.
        """

        with self._lock:
            self._items[item.memory_id] = item
            self._remembered_count += 1
            self._last_memory_id = item.memory_id
            self._last_error = None
            self._evict_if_needed_locked()

        self._logger.info(
            "short_term_memory_remembered",
            store=self.name,
            memory_id=item.memory_id,
            kind=item.kind.value,
            priority=item.priority.value,
        )

        return item

    def remember_text(
        self,
        text: str,
        *,
        kind: ShortTermMemoryKind = ShortTermMemoryKind.SESSION_FACT,
        source: str = "cognition",
        priority: ShortTermMemoryPriority = ShortTermMemoryPriority.NORMAL,
        confidence: float = 1.0,
        expires_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ShortTermMemoryItem:
        """
        Convenience method for storing plain text memory.
        """

        item = ShortTermMemoryItem(
            kind=kind,
            text=text,
            source=source,
            priority=priority,
            confidence=confidence,
            expires_at=expires_at,
            metadata=metadata or {},
        )

        return self.remember(item)

    def retrieve(self, query: ShortTermMemoryQuery) -> ShortTermMemoryResult:
        """
        Retrieve memory items matching query constraints.

        This method is intentionally strict. It returns direct matches only.
        build_context() can add bounded backfill for conversational continuity.
        """

        with self._lock:
            self._last_query_text = query.query_text
            self._last_error = None

            candidates = tuple(
                item
                for item in self._items.values()
                if self._matches_query(item, query)
            )
            ranked = tuple(
                sorted(
                    candidates,
                    key=self._rank_key,
                    reverse=True,
                )[: query.max_items]
            )
            self._retrieved_count += 1

        self._logger.info(
            "short_term_memory_retrieved",
            store=self.name,
            query_text=query.query_text,
            item_count=len(ranked),
        )

        return ShortTermMemoryResult(
            query=query,
            items=ranked,
            metadata={
                "store": self.name,
            },
        )

    def retrieve_for_request(
        self,
        request: CognitionRequest,
        *,
        max_items: int | None = None,
    ) -> ShortTermMemoryResult:
        """
        Retrieve direct short-term memory matches for one cognition request.
        """

        query = ShortTermMemoryQuery(
            query_text=request.text,
            max_items=max_items or self._config.max_context_items,
        )

        return self.retrieve(query)

    def build_context(
        self,
        *,
        request: CognitionRequest | None = None,
        query: ShortTermMemoryQuery | None = None,
    ) -> CognitionContext:
        """
        Build CognitionContext from short-term memory.

        Retrieval is intentionally two-stage:
        1. retrieve direct matches for the query
        2. backfill with recent/high-priority active memory if context has room

        This gives JARVIS continuity without requiring every useful memory item
        to share exact words with the current user request.
        """

        if query is None:
            query = ShortTermMemoryQuery(
                query_text=request.text if request is not None else None,
                max_items=self._config.max_context_items,
            )

        result = self.retrieve(query)
        selected_items = list(result.items[: self._config.max_context_items])

        if len(selected_items) < self._config.max_context_items:
            selected_ids = {item.memory_id for item in selected_items}

            with self._lock:
                backfill_candidates = tuple(
                    sorted(
                        (
                            item
                            for item in self._items.values()
                            if item.memory_id not in selected_ids
                            and self._backfill_allowed(item, query)
                        ),
                        key=self._rank_key,
                        reverse=True,
                    )
                )

            for item in backfill_candidates:
                selected_items.append(item)

                if len(selected_items) >= self._config.max_context_items:
                    break

        items = tuple(
            self._context_item_for_memory_item(item)
            for item in selected_items[: self._config.max_context_items]
        )

        return CognitionContext(
            session_id=request.context.session_id if request is not None else None,
            turn_id=request.turn_id if request is not None else None,
            items=items,
            metadata={
                "source": self.name,
                "memory_item_count": len(items),
                "query_text": query.query_text,
                "direct_match_count": result.item_count,
            },
        )

    def enrich_request(
        self,
        request: CognitionRequest,
        *,
        query: ShortTermMemoryQuery | None = None,
    ) -> CognitionRequest:
        """
        Return a request copy enriched with short-term memory context.

        Existing request context items are preserved and memory items are appended.
        """

        memory_context = self.build_context(request=request, query=query)
        combined_items = (
            *request.context.items,
            *memory_context.items,
        )

        enriched_context = request.context.model_copy(
            update={
                "items": combined_items,
                "metadata": {
                    **request.context.metadata,
                    "short_term_memory_item_count": len(memory_context.items),
                    "short_term_memory_source": self.name,
                },
            }
        )

        return request.model_copy(
            update={
                "context": enriched_context,
                "metadata": {
                    **request.metadata,
                    "short_term_memory_enriched": True,
                },
            }
        )

    def forget(self, memory_id: str) -> bool:
        """
        Forget one item by id.
        """

        cleaned = memory_id.strip()

        if not cleaned:
            return False

        with self._lock:
            removed = self._items.pop(cleaned, None)

            if removed is not None:
                self._forgotten_count += 1
                self._last_memory_id = cleaned
                self._last_error = None

        if removed is not None:
            self._logger.info(
                "short_term_memory_forgotten",
                store=self.name,
                memory_id=cleaned,
            )

            return True

        return False

    def forget_expired(
        self,
        *,
        now: datetime | None = None,
    ) -> int:
        """
        Remove expired items.
        """

        current = now or utc_now()

        with self._lock:
            expired_ids = [
                item.memory_id
                for item in self._items.values()
                if item.expired(now=current)
            ]

            for memory_id in expired_ids:
                self._items.pop(memory_id, None)

            self._forgotten_count += len(expired_ids)

        if expired_ids:
            self._logger.info(
                "short_term_memory_expired_forgotten",
                store=self.name,
                count=len(expired_ids),
            )

        return len(expired_ids)

    def clear(self) -> None:
        """
        Clear all short-term memory.
        """

        with self._lock:
            self._items.clear()
            self._cleared_count += 1
            self._last_memory_id = None
            self._last_error = None

        self._logger.info("short_term_memory_cleared", store=self.name)

    def items(self) -> tuple[ShortTermMemoryItem, ...]:
        """
        Return immutable copy of current items.
        """

        with self._lock:
            return tuple(self._items.values())

    def snapshot(self) -> ShortTermMemorySnapshot:
        """
        Return memory diagnostics.
        """

        with self._lock:
            active_count = sum(
                1 for item in self._items.values() if not item.expired()
            )
            expired_count = len(self._items) - active_count

            return ShortTermMemorySnapshot(
                name=self.name,
                item_count=len(self._items),
                active_item_count=active_count,
                expired_item_count=expired_count,
                remembered_count=self._remembered_count,
                retrieved_count=self._retrieved_count,
                forgotten_count=self._forgotten_count,
                cleared_count=self._cleared_count,
                last_memory_id=self._last_memory_id,
                last_query_text=self._last_query_text,
                last_error=self._last_error,
            )

    def _matches_query(
        self,
        item: ShortTermMemoryItem,
        query: ShortTermMemoryQuery,
    ) -> bool:
        if not query.include_expired and item.expired():
            return False

        if item.confidence < query.min_confidence:
            return False

        if query.kinds and item.kind not in query.kinds:
            return False

        if query.query_text is None:
            return True

        return self._text_relevant(
            query_text=query.query_text,
            item_text=item.text,
        )

    def _backfill_allowed(
        self,
        item: ShortTermMemoryItem,
        query: ShortTermMemoryQuery,
    ) -> bool:
        if not query.include_expired and item.expired():
            return False

        if item.confidence < query.min_confidence:
            return False

        if query.kinds and item.kind not in query.kinds:
            return False

        return True

    def _text_relevant(
        self,
        *,
        query_text: str,
        item_text: str,
    ) -> bool:
        query_terms = self._terms(query_text)
        item_terms = self._terms(item_text)

        if not query_terms:
            return True

        return bool(query_terms & item_terms)

    def _rank_key(
        self,
        item: ShortTermMemoryItem,
    ) -> tuple[int, float, datetime]:
        return (
            self._priority_weight(item.priority),
            item.confidence,
            item.created_at,
        )

    def _context_item_for_memory_item(
        self,
        item: ShortTermMemoryItem,
    ) -> CognitionContextItem:
        return CognitionContextItem(
            kind=f"short_term_memory_{item.kind.value}",
            text=self._bounded_text(item.text, self._config.max_context_item_chars),
            source=self.name,
            metadata={
                "memory_id": item.memory_id,
                "priority": item.priority.value,
                "confidence": item.confidence,
                "source": item.source,
            },
        )

    def _evict_if_needed_locked(self) -> None:
        if len(self._items) <= self._config.max_items:
            return

        overflow = len(self._items) - self._config.max_items
        ranked_oldest_first = sorted(
            self._items.values(),
            key=lambda item: (
                self._priority_weight(item.priority),
                item.created_at,
            ),
        )

        for item in ranked_oldest_first[:overflow]:
            self._items.pop(item.memory_id, None)

    @staticmethod
    def _priority_weight(priority: ShortTermMemoryPriority) -> int:
        weights = {
            ShortTermMemoryPriority.LOW: 0,
            ShortTermMemoryPriority.NORMAL: 1,
            ShortTermMemoryPriority.HIGH: 2,
            ShortTermMemoryPriority.CRITICAL: 3,
        }

        return weights[priority]

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {
            term.strip(".,!?;:()[]{}\"'").casefold()
            for term in text.split()
            if term.strip(".,!?;:()[]{}\"'")
        }

    @staticmethod
    def _bounded_text(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text

        if max_chars <= 3:
            return text[:max_chars]

        return f"{text[: max_chars - 3].rstrip()}..."