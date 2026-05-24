from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from threading import RLock
from typing import Protocol, runtime_checkable

from pydantic import Field, field_validator

from jarvis.memory.models import (
    MemoryImportance,
    MemoryKind,
    MemoryModel,
    MemoryPolicyClassification,
    MemoryRecord,
    MemorySensitivity,
    MemorySource,
    utc_now,
)
from jarvis.runtime.observability.structured_logger import get_logger


class MemorySummaryKind(StrEnum):
    """
    Type of memory summary being produced.
    """

    EXTRACTIVE = "extractive"
    EPISODIC_TIMELINE = "episodic_timeline"
    SEMANTIC_SYNTHESIS = "semantic_synthesis"
    PROFILE_SUMMARY = "profile_summary"
    PROJECT_SUMMARY = "project_summary"


class MemorySummaryStatus(StrEnum):
    """
    Summary generation status.
    """

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    EMPTY = "empty"


class MemorySummarySource(MemoryModel):
    """
    One source memory used in a summary.

    This keeps summaries auditable. JARVIS must be able to explain what memory
    records were used to create a summary.
    """

    memory_id: str
    kind: MemoryKind
    source: MemorySource
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    policy_classification: MemoryPolicyClassification
    score: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("memory_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class MemorySummaryRequest(MemoryModel):
    """
    Request to summarize memory records.

    This request is storage-independent and LLM-independent. Later summarizers
    can be extractive, local-LLM based, or hybrid.
    """

    records: tuple[MemoryRecord, ...]
    summary_kind: MemorySummaryKind = MemorySummaryKind.EXTRACTIVE
    max_chars: int = Field(default=600, ge=80, le=10_000)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    include_sensitive: bool = False
    include_expired: bool = False
    instruction: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("instruction")
    @classmethod
    def _clean_optional_instruction(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None


class MemorySummary(MemoryModel):
    """
    One auditable memory summary.
    """

    text: str
    summary_kind: MemorySummaryKind
    sources: tuple[MemorySummarySource, ...]
    confidence: float = Field(ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=utc_now)
    policy_classification: MemoryPolicyClassification = (
        MemoryPolicyClassification.ALLOWED
    )
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def _text_required(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("summary text cannot be empty.")

        return cleaned

    @property
    def source_count(self) -> int:
        return len(self.sources)

    @property
    def memory_ids(self) -> tuple[str, ...]:
        return tuple(source.memory_id for source in self.sources)


class MemorySummaryResult(MemoryModel):
    """
    Result of one summarization attempt.
    """

    request: MemorySummaryRequest
    status: MemorySummaryStatus
    summary: MemorySummary | None = None
    failure_reason: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == MemorySummaryStatus.SUCCEEDED

    @property
    def failed(self) -> bool:
        return self.status == MemorySummaryStatus.FAILED

    @property
    def empty(self) -> bool:
        return self.status == MemorySummaryStatus.EMPTY


@dataclass(frozen=True, slots=True)
class MemorySummarizerSnapshot:
    """
    Observable diagnostics for a memory summarizer.
    """

    name: str
    summarized_count: int
    succeeded_count: int
    failed_count: int
    empty_count: int
    last_status: MemorySummaryStatus | None
    last_error: str | None


@runtime_checkable
class MemorySummarizer(Protocol):
    """
    Storage-independent summarizer contract.
    """

    @property
    def name(self) -> str:
        """Stable summarizer name."""

    def summarize(self, request: MemorySummaryRequest) -> MemorySummaryResult:
        """Summarize memory records."""

    def snapshot(self) -> MemorySummarizerSnapshot:
        """Return summarizer diagnostics."""


@dataclass(frozen=True, slots=True)
class ExtractiveMemorySummarizerConfig:
    """
    Configuration for deterministic extractive summarizer.

    This is fake-first and testable. It does not call an LLM.
    """

    name: str = "extractive_memory_summarizer"
    separator: str = " "
    max_source_records: int = 8

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if not self.separator:
            raise ValueError("separator cannot be empty.")

        if self.max_source_records <= 0:
            raise ValueError("max_source_records must be greater than zero.")


class ExtractiveMemorySummarizer:
    """
    Deterministic summarizer for memory records.

    Responsibilities:
    - filter records according to request policy
    - select high-value source memories
    - build a bounded text summary
    - preserve source ids, reasons, confidence, and policy classification
    - keep diagnostics

    Non-responsibilities:
    - no LLM calls
    - no embeddings
    - no persistence
    - no gateway access
    - no memory writes
    """

    def __init__(
        self,
        *,
        config: ExtractiveMemorySummarizerConfig | None = None,
    ) -> None:
        self._config = config or ExtractiveMemorySummarizerConfig()
        self._config.validate()

        self._lock = RLock()
        self._logger = get_logger("memory.extractive_summarizer")

        self._summarized_count = 0
        self._succeeded_count = 0
        self._failed_count = 0
        self._empty_count = 0
        self._last_status: MemorySummaryStatus | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def summarize(self, request: MemorySummaryRequest) -> MemorySummaryResult:
        """
        Summarize memory records deterministically.
        """

        with self._lock:
            self._summarized_count += 1
            self._last_error = None

        try:
            candidates = self._filter_records(request)

            if not candidates:
                result = MemorySummaryResult(
                    request=request,
                    status=MemorySummaryStatus.EMPTY,
                    failure_reason="no memory records eligible for summary",
                    metadata={
                        "summarizer": self.name,
                    },
                )
                self._record(result)

                return result

            selected = self._select_records(candidates)
            summary_text = self._build_summary_text(
                records=selected,
                max_chars=request.max_chars,
            )

            if not summary_text:
                result = MemorySummaryResult(
                    request=request,
                    status=MemorySummaryStatus.EMPTY,
                    failure_reason="summary text was empty after filtering",
                    metadata={
                        "summarizer": self.name,
                    },
                )
                self._record(result)

                return result

            sources = tuple(self._source_for(record) for record in selected)
            summary = MemorySummary(
                text=summary_text,
                summary_kind=request.summary_kind,
                sources=sources,
                confidence=self._summary_confidence(selected),
                policy_classification=self._summary_policy_classification(
                    selected
                ),
                metadata={
                    "summarizer": self.name,
                    "record_count": len(request.records),
                    "selected_count": len(selected),
                },
            )
            result = MemorySummaryResult(
                request=request,
                status=MemorySummaryStatus.SUCCEEDED,
                summary=summary,
                metadata={
                    "summarizer": self.name,
                },
            )
            self._record(result)

            self._logger.info(
                "memory_summary_created",
                summarizer=self.name,
                summary_kind=request.summary_kind.value,
                source_count=summary.source_count,
                text_length=len(summary.text),
            )

            return result

        except Exception as exc:
            result = MemorySummaryResult(
                request=request,
                status=MemorySummaryStatus.FAILED,
                failure_reason=f"{type(exc).__name__}: {exc}",
                metadata={
                    "summarizer": self.name,
                },
            )
            self._record(result)

            return result

    def snapshot(self) -> MemorySummarizerSnapshot:
        """
        Return summarizer diagnostics.
        """

        with self._lock:
            return MemorySummarizerSnapshot(
                name=self.name,
                summarized_count=self._summarized_count,
                succeeded_count=self._succeeded_count,
                failed_count=self._failed_count,
                empty_count=self._empty_count,
                last_status=self._last_status,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Reset summarizer diagnostics.
        """

        with self._lock:
            self._summarized_count = 0
            self._succeeded_count = 0
            self._failed_count = 0
            self._empty_count = 0
            self._last_status = None
            self._last_error = None

        self._logger.info("memory_summarizer_reset", summarizer=self.name)

    def _filter_records(
        self,
        request: MemorySummaryRequest,
    ) -> tuple[MemoryRecord, ...]:
        return tuple(
            record
            for record in request.records
            if self._record_allowed(record=record, request=request)
        )

    @staticmethod
    def _record_allowed(
        *,
        record: MemoryRecord,
        request: MemorySummaryRequest,
    ) -> bool:
        if record.confidence < request.min_confidence:
            return False

        if not request.include_expired and record.expired():
            return False

        if (
            not request.include_sensitive
            and record.sensitivity == MemorySensitivity.SENSITIVE
        ):
            return False

        return True

    def _select_records(
        self,
        records: tuple[MemoryRecord, ...],
    ) -> tuple[MemoryRecord, ...]:
        ranked = sorted(
            records,
            key=lambda record: (
                self._importance_score(record.importance),
                record.confidence,
                record.updated_at,
            ),
            reverse=True,
        )

        return tuple(ranked[: self._config.max_source_records])

    def _build_summary_text(
        self,
        *,
        records: tuple[MemoryRecord, ...],
        max_chars: int,
    ) -> str:
        parts: list[str] = []

        for record in records:
            candidate = record.text.strip()

            if not candidate:
                continue

            next_text = self._config.separator.join((*parts, candidate)).strip()

            if len(next_text) > max_chars:
                remaining = max_chars - len(
                    self._config.separator.join(parts)
                )

                if remaining > 12:
                    parts.append(candidate[: remaining - 3].rstrip() + "...")

                break

            parts.append(candidate)

        return self._config.separator.join(parts).strip()

    @staticmethod
    def _source_for(record: MemoryRecord) -> MemorySummarySource:
        classification = (
            MemoryPolicyClassification.RESTRICTED
            if record.sensitivity == MemorySensitivity.SENSITIVE
            else MemoryPolicyClassification.ALLOWED
        )

        return MemorySummarySource(
            memory_id=record.memory_id,
            kind=record.kind,
            source=record.source,
            reason="selected for extractive memory summary",
            confidence=record.confidence,
            policy_classification=classification,
            score=ExtractiveMemorySummarizer._importance_score(
                record.importance
            ),
        )

    @staticmethod
    def _summary_confidence(records: tuple[MemoryRecord, ...]) -> float:
        if not records:
            return 0.0

        confidence = sum(record.confidence for record in records) / len(records)

        return max(0.0, min(1.0, confidence))

    @staticmethod
    def _summary_policy_classification(
        records: tuple[MemoryRecord, ...],
    ) -> MemoryPolicyClassification:
        if any(
            record.sensitivity == MemorySensitivity.SENSITIVE
            for record in records
        ):
            return MemoryPolicyClassification.RESTRICTED

        return MemoryPolicyClassification.ALLOWED

    @staticmethod
    def _importance_score(importance: MemoryImportance) -> float:
        scores = {
            MemoryImportance.LOW: 0.25,
            MemoryImportance.NORMAL: 0.5,
            MemoryImportance.HIGH: 0.8,
            MemoryImportance.CRITICAL: 1.0,
        }

        return scores[importance]

    def _record(self, result: MemorySummaryResult) -> None:
        with self._lock:
            self._last_status = result.status

            if result.succeeded:
                self._succeeded_count += 1
                self._last_error = None

            elif result.empty:
                self._empty_count += 1
                self._last_error = result.failure_reason

            else:
                self._failed_count += 1
                self._last_error = result.failure_reason

        self._logger.info(
            "memory_summarizer_result_recorded",
            summarizer=self.name,
            status=result.status.value,
            failure_reason=result.failure_reason,
        )