from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from pydantic import Field, field_validator

from jarvis.memory.context import (
    MemoryContext,
    MemoryContextBuilder,
    MemoryContextBuildRequest,
)
from jarvis.memory.gateway import MemoryGateway, MemoryGatewayRetrievalResult
from jarvis.memory.models import (
    MemoryKind,
    MemoryModel,
    MemoryPolicyClassification,
    MemoryQuery,
)
from jarvis.memory.summarization import (
    MemorySummarizer,
    MemorySummaryKind,
    MemorySummaryRequest,
    MemorySummaryResult,
)
from jarvis.runtime.observability.structured_logger import get_logger


class MemoryCognitionQuery(MemoryModel):
    """
    Cognition-facing memory query.

    This is intentionally not a raw store query. Cognition uses this contract,
    and the bridge converts it into governed MemoryGateway calls.
    """

    text: str
    kinds: tuple[MemoryKind, ...] = ()
    max_results: int = Field(default=8, ge=1, le=100)
    max_context_items: int = Field(default=8, ge=1, le=100)
    max_context_chars: int = Field(default=2_000, ge=80, le=50_000)
    include_sensitive: bool = False
    include_expired: bool = False
    include_restricted_context: bool = False
    include_summary: bool = False
    summary_kind: MemorySummaryKind = MemorySummaryKind.EXTRACTIVE
    summary_max_chars: int = Field(default=600, ge=80, le=10_000)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def _text_required(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("text cannot be empty.")

        return cleaned

    def to_memory_query(self) -> MemoryQuery:
        """
        Convert to governed MemoryQuery.
        """

        return MemoryQuery(
            text=self.text,
            kinds=self.kinds,
            max_results=self.max_results,
            include_sensitive=self.include_sensitive,
            include_expired=self.include_expired,
            metadata={
                **self.metadata,
                "cognition_memory_query": True,
            },
        )


class MemoryCognitionContextResult(MemoryModel):
    """
    Result returned to cognition after memory context construction.
    """

    query: MemoryCognitionQuery
    retrieval: MemoryGatewayRetrievalResult
    context: MemoryContext
    summary_result: MemorySummaryResult | None = None
    allowed: bool
    blocked: bool = False
    reason: str
    policy_classification: MemoryPolicyClassification
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("reason")
    @classmethod
    def _reason_required(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("reason cannot be empty.")

        return cleaned

    @property
    def context_item_count(self) -> int:
        return self.context.item_count

    @property
    def empty(self) -> bool:
        return self.context.empty

    def as_cognition_metadata(self) -> dict[str, object]:
        """
        Return compact metadata suitable for future CognitionRequest metadata.

        This does not mutate cognition models. Step 12 only establishes the
        safe integration boundary.
        """

        return {
            "memory_context_id": self.context.context_id,
            "memory_context_item_count": self.context.item_count,
            "memory_context_total_chars": self.context.total_chars,
            "memory_retrieval_allowed": self.allowed,
            "memory_retrieval_blocked": self.blocked,
            "memory_retrieval_reason": self.reason,
            "memory_policy_classification": self.policy_classification.value,
            "memory_query_text": self.query.text,
        }


@dataclass(frozen=True, slots=True)
class MemoryCognitionBridgeConfig:
    """
    Configuration for MemoryCognitionBridge.
    """

    name: str = "memory_cognition_bridge"
    enable_summarization: bool = True
    default_max_results: int = 8
    default_max_context_items: int = 8
    default_max_context_chars: int = 2_000

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.default_max_results <= 0:
            raise ValueError("default_max_results must be greater than zero.")

        if self.default_max_context_items <= 0:
            raise ValueError("default_max_context_items must be greater than zero.")

        if self.default_max_context_chars < 80:
            raise ValueError("default_max_context_chars must be at least 80.")


@dataclass(frozen=True, slots=True)
class MemoryCognitionBridgeSnapshot:
    """
    Observable diagnostics for MemoryCognitionBridge.
    """

    name: str
    build_count: int
    built_count: int
    empty_count: int
    blocked_count: int
    summary_count: int
    last_context_id: str | None
    last_query_text: str | None
    last_error: str | None


class MemoryCognitionBridge:
    """
    Safe bridge between cognition and governed memory.

    Responsibilities:
    - accept cognition-facing memory query
    - call MemoryGateway, never MemoryStore
    - optionally summarize retrieved memory
    - build bounded MemoryContext
    - preserve source/reason/confidence/timestamp/policy classification
    - expose diagnostics

    Non-responsibilities:
    - no direct store access
    - no direct vector access
    - no LLM calls
    - no memory writes
    - no cognition runtime mutation
    """

    def __init__(
        self,
        *,
        gateway: MemoryGateway,
        context_builder: MemoryContextBuilder | None = None,
        summarizer: MemorySummarizer | None = None,
        config: MemoryCognitionBridgeConfig | None = None,
    ) -> None:
        self._config = config or MemoryCognitionBridgeConfig()
        self._config.validate()

        self._gateway = gateway
        self._context_builder = context_builder or MemoryContextBuilder()
        self._summarizer = summarizer
        self._lock = RLock()
        self._logger = get_logger("memory.cognition_bridge")

        self._build_count = 0
        self._built_count = 0
        self._empty_count = 0
        self._blocked_count = 0
        self._summary_count = 0
        self._last_context_id: str | None = None
        self._last_query_text: str | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def build_context(
        self,
        query: MemoryCognitionQuery,
    ) -> MemoryCognitionContextResult:
        """
        Build cognition-ready memory context from a governed memory query.
        """

        with self._lock:
            self._build_count += 1
            self._last_query_text = query.text
            self._last_error = None

        retrieval = self._gateway.retrieve(query.to_memory_query())

        if retrieval.blocked or not retrieval.allowed:
            context = self._context_builder.build(MemoryContextBuildRequest())
            result = MemoryCognitionContextResult(
                query=query,
                retrieval=retrieval,
                context=context,
                summary_result=None,
                allowed=False,
                blocked=True,
                reason=retrieval.reason,
                policy_classification=retrieval.policy_classification,
                metadata={
                    "bridge": self.name,
                },
            )
            self._record_result(result)

            return result

        summary_result = self._maybe_summarize(query=query, retrieval=retrieval)
        context = self._context_builder.build(
            MemoryContextBuildRequest(
                retrievals=(retrieval,),
                summaries=()
                if summary_result is None
                else (summary_result,),
                max_items=query.max_context_items,
                max_chars=query.max_context_chars,
                include_restricted=query.include_restricted_context,
                metadata={
                    "bridge": self.name,
                    "query_text": query.text,
                },
            )
        )

        result = MemoryCognitionContextResult(
            query=query,
            retrieval=retrieval,
            context=context,
            summary_result=summary_result,
            allowed=True,
            blocked=False,
            reason="memory context built through governed gateway",
            policy_classification=MemoryPolicyClassification.ALLOWED,
            metadata={
                "bridge": self.name,
                "summary_used": summary_result is not None,
            },
        )
        self._record_result(result)

        self._logger.info(
            "memory_cognition_context_built",
            bridge=self.name,
            context_id=context.context_id,
            query_text=query.text,
            item_count=context.item_count,
            total_chars=context.total_chars,
            summary_used=summary_result is not None,
        )

        return result

    def build_context_from_text(
        self,
        text: str,
        *,
        kinds: tuple[MemoryKind, ...] = (),
        max_results: int | None = None,
        max_context_items: int | None = None,
        max_context_chars: int | None = None,
        include_summary: bool = False,
        include_sensitive: bool = False,
        include_expired: bool = False,
        include_restricted_context: bool = False,
    ) -> MemoryCognitionContextResult:
        """
        Convenience method for cognition to request memory context from text.
        """

        query = MemoryCognitionQuery(
            text=text,
            kinds=kinds,
            max_results=max_results or self._config.default_max_results,
            max_context_items=(
                max_context_items or self._config.default_max_context_items
            ),
            max_context_chars=(
                max_context_chars or self._config.default_max_context_chars
            ),
            include_summary=include_summary,
            include_sensitive=include_sensitive,
            include_expired=include_expired,
            include_restricted_context=include_restricted_context,
        )

        return self.build_context(query)

    def snapshot(self) -> MemoryCognitionBridgeSnapshot:
        """
        Return bridge diagnostics.
        """

        with self._lock:
            return MemoryCognitionBridgeSnapshot(
                name=self.name,
                build_count=self._build_count,
                built_count=self._built_count,
                empty_count=self._empty_count,
                blocked_count=self._blocked_count,
                summary_count=self._summary_count,
                last_context_id=self._last_context_id,
                last_query_text=self._last_query_text,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Reset bridge diagnostics.
        """

        with self._lock:
            self._build_count = 0
            self._built_count = 0
            self._empty_count = 0
            self._blocked_count = 0
            self._summary_count = 0
            self._last_context_id = None
            self._last_query_text = None
            self._last_error = None

        self._logger.info("memory_cognition_bridge_reset", bridge=self.name)

    def _maybe_summarize(
        self,
        *,
        query: MemoryCognitionQuery,
        retrieval: MemoryGatewayRetrievalResult,
    ) -> MemorySummaryResult | None:
        if not self._config.enable_summarization:
            return None

        if not query.include_summary:
            return None

        if self._summarizer is None:
            return None

        if not retrieval.records:
            return None

        result = self._summarizer.summarize(
            MemorySummaryRequest(
                records=retrieval.records,
                summary_kind=query.summary_kind,
                max_chars=query.summary_max_chars,
                include_sensitive=query.include_restricted_context,
                include_expired=query.include_expired,
                metadata={
                    "bridge": self.name,
                    "query_text": query.text,
                },
            )
        )

        with self._lock:
            self._summary_count += 1

        return result

    def _record_result(self, result: MemoryCognitionContextResult) -> None:
        with self._lock:
            self._last_context_id = result.context.context_id

            if result.blocked:
                self._blocked_count += 1
                self._last_error = result.reason
                return

            if result.empty:
                self._empty_count += 1
            else:
                self._built_count += 1