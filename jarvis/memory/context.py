from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from threading import RLock
from typing import Protocol, runtime_checkable

from pydantic import Field, field_validator

from jarvis.memory.gateway import MemoryGatewayRetrievalResult
from jarvis.memory.models import (
    MemoryKind,
    MemoryModel,
    MemoryPolicyClassification,
    MemorySource,
    new_id,
    utc_now,
)
from jarvis.memory.summarization import (
    MemorySummaryResult,
    MemorySummaryStatus,
)
from jarvis.runtime.observability.structured_logger import get_logger


class MemoryContextItemKind(StrEnum):
    """
    Kind of item included in a cognition memory context.
    """

    RETRIEVED_MEMORY = "retrieved_memory"
    SUMMARY = "summary"


class MemoryContextBuildStatus(StrEnum):
    """
    Status of one memory context build.
    """

    BUILT = "built"
    EMPTY = "empty"


class MemoryContextItem(MemoryModel):
    """
    One memory item prepared for cognition context.

    This is not raw memory stuffing. It is an auditable, bounded, policy-aware
    context item.
    """

    item_id: str = Field(default_factory=new_id)
    item_kind: MemoryContextItemKind
    text: str
    source_memory_id: str | None = None
    memory_kind: MemoryKind | None = None
    source: MemorySource
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=utc_now)
    policy_classification: MemoryPolicyClassification
    tags: tuple[str, ...] = ()
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("item_id", "text", "reason")
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

    @property
    def char_count(self) -> int:
        return len(self.text)


class MemoryContext(MemoryModel):
    """
    Final memory context object prepared for cognition.

    Future cognition integration should consume this object, not raw store
    results.
    """

    context_id: str = Field(default_factory=new_id)
    items: tuple[MemoryContextItem, ...] = ()
    status: MemoryContextBuildStatus = MemoryContextBuildStatus.BUILT
    built_at: datetime = Field(default_factory=utc_now)
    total_chars: int = Field(default=0, ge=0)
    metadata: dict[str, object] = Field(default_factory=dict)

    @property
    def item_count(self) -> int:
        return len(self.items)

    @property
    def empty(self) -> bool:
        return self.item_count == 0

    def as_text_block(self) -> str:
        """
        Render context as a compact auditable text block.

        This method is for future prompt/context assembly. It preserves reasons
        and policy classification.
        """

        lines: list[str] = []

        for index, item in enumerate(self.items, start=1):
            lines.append(
                f"{index}. {item.text} "
                f"[source={item.source.value}; "
                f"reason={item.reason}; "
                f"confidence={item.confidence:.2f}; "
                f"policy={item.policy_classification.value}]"
            )

        return "\n".join(lines)


class MemoryContextBuildRequest(MemoryModel):
    """
    Request to build cognition-ready memory context.

    Inputs are gateway retrieval results and summary results, not direct store
    access.
    """

    retrievals: tuple[MemoryGatewayRetrievalResult, ...] = ()
    summaries: tuple[MemorySummaryResult, ...] = ()
    max_items: int = Field(default=8, ge=1, le=100)
    max_chars: int = Field(default=2_000, ge=80, le=50_000)
    include_restricted: bool = False
    metadata: dict[str, object] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MemoryContextBuilderConfig:
    """
    Configuration for MemoryContextBuilder.
    """

    name: str = "memory_context_builder"
    summary_score: float = 0.95
    prefer_summaries: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.summary_score < 0.0 or self.summary_score > 1.0:
            raise ValueError("summary_score must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class MemoryContextBuilderSnapshot:
    """
    Observable diagnostics for MemoryContextBuilder.
    """

    name: str
    build_count: int
    built_count: int
    empty_count: int
    last_context_id: str | None
    last_item_count: int
    last_total_chars: int
    last_error: str | None


@runtime_checkable
class MemoryContextBuilderProtocol(Protocol):
    """
    Contract for memory context builders.
    """

    @property
    def name(self) -> str:
        """Stable builder name."""

    def build(self, request: MemoryContextBuildRequest) -> MemoryContext:
        """Build cognition-ready memory context."""

    def snapshot(self) -> MemoryContextBuilderSnapshot:
        """Return builder diagnostics."""


class MemoryContextBuilder:
    """
    Builds bounded, auditable memory context for cognition.

    Responsibilities:
    - consume gateway retrieval results and summary results
    - convert them into MemoryContextItem objects
    - preserve source, reason, confidence, timestamp, and policy classification
    - deduplicate memory ids
    - enforce item and character budgets
    - expose diagnostics

    Non-responsibilities:
    - no direct MemoryStore access
    - no MemoryGateway calls
    - no LLM calls
    - no embeddings
    - no memory writes
    """

    def __init__(
        self,
        *,
        config: MemoryContextBuilderConfig | None = None,
    ) -> None:
        self._config = config or MemoryContextBuilderConfig()
        self._config.validate()

        self._lock = RLock()
        self._logger = get_logger("memory.context_builder")

        self._build_count = 0
        self._built_count = 0
        self._empty_count = 0
        self._last_context_id: str | None = None
        self._last_item_count = 0
        self._last_total_chars = 0
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def build(self, request: MemoryContextBuildRequest) -> MemoryContext:
        """
        Build bounded memory context.
        """

        with self._lock:
            self._build_count += 1
            self._last_error = None

        candidates = self._collect_candidates(request)
        bounded_items = self._apply_budget(
            items=candidates,
            max_items=request.max_items,
            max_chars=request.max_chars,
        )

        status = (
            MemoryContextBuildStatus.EMPTY
            if not bounded_items
            else MemoryContextBuildStatus.BUILT
        )
        context = MemoryContext(
            items=bounded_items,
            status=status,
            total_chars=sum(item.char_count for item in bounded_items),
            metadata={
                **request.metadata,
                "builder": self.name,
                "retrieval_count": len(request.retrievals),
                "summary_count": len(request.summaries),
            },
        )

        self._record_context(context)

        self._logger.info(
            "memory_context_built",
            builder=self.name,
            context_id=context.context_id,
            status=context.status.value,
            item_count=context.item_count,
            total_chars=context.total_chars,
        )

        return context

    def snapshot(self) -> MemoryContextBuilderSnapshot:
        """
        Return builder diagnostics.
        """

        with self._lock:
            return MemoryContextBuilderSnapshot(
                name=self.name,
                build_count=self._build_count,
                built_count=self._built_count,
                empty_count=self._empty_count,
                last_context_id=self._last_context_id,
                last_item_count=self._last_item_count,
                last_total_chars=self._last_total_chars,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Reset builder diagnostics.
        """

        with self._lock:
            self._build_count = 0
            self._built_count = 0
            self._empty_count = 0
            self._last_context_id = None
            self._last_item_count = 0
            self._last_total_chars = 0
            self._last_error = None

        self._logger.info("memory_context_builder_reset", builder=self.name)

    def _collect_candidates(
        self,
        request: MemoryContextBuildRequest,
    ) -> tuple[MemoryContextItem, ...]:
        items: list[MemoryContextItem] = []

        if self._config.prefer_summaries:
            items.extend(self._summary_items(request))
            items.extend(self._retrieval_items(request))
        else:
            items.extend(self._retrieval_items(request))
            items.extend(self._summary_items(request))

        filtered = tuple(
            item
            for item in items
            if self._policy_allowed(
                item=item,
                include_restricted=request.include_restricted,
            )
        )

        return self._deduplicate(filtered)

    def _retrieval_items(
        self,
        request: MemoryContextBuildRequest,
    ) -> list[MemoryContextItem]:
        items: list[MemoryContextItem] = []

        for retrieval in request.retrievals:
            if retrieval.blocked or not retrieval.allowed:
                continue

            for result in retrieval.results:
                explanation = result.explanation
                record = result.record

                items.append(
                    MemoryContextItem(
                        item_kind=MemoryContextItemKind.RETRIEVED_MEMORY,
                        text=record.text,
                        source_memory_id=record.memory_id,
                        memory_kind=record.kind,
                        source=explanation.source,
                        reason=explanation.reason,
                        confidence=explanation.confidence,
                        score=result.score,
                        timestamp=explanation.retrieved_at,
                        policy_classification=(
                            explanation.policy_classification
                        ),
                        tags=record.tags,
                        metadata={
                            "memory_id": record.memory_id,
                            "query_id": retrieval.query.query_id,
                            "gateway_reason": retrieval.reason,
                            "context_source": "retrieval",
                        },
                    )
                )

        return items

    def _summary_items(
        self,
        request: MemoryContextBuildRequest,
    ) -> list[MemoryContextItem]:
        items: list[MemoryContextItem] = []

        for result in request.summaries:
            if result.status != MemorySummaryStatus.SUCCEEDED:
                continue

            if result.summary is None:
                continue

            summary = result.summary
            source_ids = summary.memory_ids
            source = (
                summary.sources[0].source
                if summary.sources
                else MemorySource.SYSTEM
            )

            items.append(
                MemoryContextItem(
                    item_kind=MemoryContextItemKind.SUMMARY,
                    text=summary.text,
                    source_memory_id="summary:" + ",".join(source_ids),
                    memory_kind=None,
                    source=source,
                    reason=(
                        "memory summary built from "
                        f"{summary.source_count} source memories"
                    ),
                    confidence=summary.confidence,
                    score=self._config.summary_score,
                    timestamp=summary.created_at,
                    policy_classification=summary.policy_classification,
                    tags=("summary", summary.summary_kind.value),
                    metadata={
                        "summary_kind": summary.summary_kind.value,
                        "source_memory_ids": source_ids,
                        "context_source": "summary",
                    },
                )
            )

        return items

    @staticmethod
    def _policy_allowed(
        *,
        item: MemoryContextItem,
        include_restricted: bool,
    ) -> bool:
        if item.policy_classification == MemoryPolicyClassification.BLOCKED:
            return False

        if item.policy_classification == MemoryPolicyClassification.REDACTED:
            return False

        if (
            item.policy_classification == MemoryPolicyClassification.RESTRICTED
            and not include_restricted
        ):
            return False

        return True

    @staticmethod
    def _deduplicate(
        items: tuple[MemoryContextItem, ...],
    ) -> tuple[MemoryContextItem, ...]:
        deduped: list[MemoryContextItem] = []
        seen: set[str] = set()

        for item in sorted(
            items,
            key=lambda value: (value.score, value.confidence),
            reverse=True,
        ):
            key = item.source_memory_id or item.text

            if key in seen:
                continue

            seen.add(key)
            deduped.append(item)

        return tuple(deduped)

    def _apply_budget(
        self,
        *,
        items: tuple[MemoryContextItem, ...],
        max_items: int,
        max_chars: int,
    ) -> tuple[MemoryContextItem, ...]:
        selected: list[MemoryContextItem] = []
        used_chars = 0

        for item in items:
            if len(selected) >= max_items:
                break

            remaining = max_chars - used_chars

            if remaining <= 0:
                break

            if item.char_count <= remaining:
                selected.append(item)
                used_chars += item.char_count
                continue

            if remaining > 12:
                truncated = item.model_copy(
                    update={
                        "text": item.text[: remaining - 3].rstrip() + "...",
                        "metadata": {
                            **item.metadata,
                            "truncated": True,
                        },
                    }
                )
                selected.append(truncated)

            break

        return tuple(selected)

    def _record_context(self, context: MemoryContext) -> None:
        with self._lock:
            self._last_context_id = context.context_id
            self._last_item_count = context.item_count
            self._last_total_chars = context.total_chars

            if context.empty:
                self._empty_count += 1
            else:
                self._built_count += 1