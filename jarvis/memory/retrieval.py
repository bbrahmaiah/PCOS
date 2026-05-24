from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import Field, field_validator

from jarvis.memory.models import (
    MemoryImportance,
    MemoryModel,
    MemoryPolicyClassification,
    MemoryQuery,
    MemoryRecord,
    MemoryRetrievalExplanation,
    MemorySensitivity,
    utc_now,
)


class MemoryRetrievalMatchKind(StrEnum):
    """
    Why a memory matched a query.
    """

    EMPTY_QUERY = "empty_query"
    TEXT_OVERLAP = "text_overlap"
    KIND_MATCH = "kind_match"
    SCOPE_MATCH = "scope_match"
    TAG_MATCH = "tag_match"
    IMPORTANCE_BOOST = "importance_boost"
    CONFIDENCE_BOOST = "confidence_boost"
    RECENCY_BOOST = "recency_boost"


class MemoryRetrievalScoreBreakdown(MemoryModel):
    """
    Explainable score components for one memory retrieval candidate.
    """

    text_score: float = Field(ge=0.0, le=1.0)
    kind_score: float = Field(ge=0.0, le=1.0)
    scope_score: float = Field(ge=0.0, le=1.0)
    tag_score: float = Field(ge=0.0, le=1.0)
    importance_score: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)
    recency_score: float = Field(ge=0.0, le=1.0)
    final_score: float = Field(ge=0.0, le=1.0)
    matched_terms: tuple[str, ...] = ()
    matched_tags: tuple[str, ...] = ()
    match_kinds: tuple[MemoryRetrievalMatchKind, ...] = ()

    @field_validator("matched_terms", "matched_tags")
    @classmethod
    def _clean_tuple_text(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(item.strip().casefold() for item in value if item.strip())

        return tuple(dict.fromkeys(cleaned))


@dataclass(frozen=True, slots=True)
class MemoryRetrievalScorerConfig:
    """
    Weighted scoring config.

    Weights intentionally sum to 1.0 by default.
    """

    text_weight: float = 0.38
    kind_weight: float = 0.12
    scope_weight: float = 0.08
    tag_weight: float = 0.12
    importance_weight: float = 0.12
    confidence_weight: float = 0.12
    recency_weight: float = 0.06
    recency_window_seconds: float = 86_400.0

    def validate(self) -> None:
        weights = (
            self.text_weight,
            self.kind_weight,
            self.scope_weight,
            self.tag_weight,
            self.importance_weight,
            self.confidence_weight,
            self.recency_weight,
        )

        if any(weight < 0.0 for weight in weights):
            raise ValueError("retrieval weights cannot be negative.")

        total = sum(weights)

        if total <= 0.0:
            raise ValueError("retrieval weights must sum above zero.")

        if self.recency_window_seconds <= 0.0:
            raise ValueError("recency_window_seconds must be greater than zero.")


class MemoryRetrievalScorer:
    """
    Dedicated explainable memory retrieval scorer.

    Responsibilities:
    - score one MemoryRecord against one MemoryQuery
    - produce score breakdown
    - produce human-readable retrieval reasons
    - produce policy classification for the candidate

    Non-responsibilities:
    - no storage
    - no gateway access control
    - no embeddings
    - no persistence
    """

    def __init__(
        self,
        *,
        config: MemoryRetrievalScorerConfig | None = None,
    ) -> None:
        self._config = config or MemoryRetrievalScorerConfig()
        self._config.validate()

    @property
    def config(self) -> MemoryRetrievalScorerConfig:
        return self._config

    def score(
        self,
        *,
        record: MemoryRecord,
        query: MemoryQuery,
    ) -> MemoryRetrievalScoreBreakdown:
        """
        Score a memory record for a query with an explainable breakdown.
        """

        matched_terms = self._matched_terms(query.text, record.text)
        matched_tags = self._matched_tags(query.tags, record.tags)

        text_score = self._text_score(
            query_text=query.text,
            record_text=record.text,
            matched_terms=matched_terms,
        )
        kind_score = self._kind_score(record=record, query=query)
        scope_score = self._scope_score(record=record, query=query)
        tag_score = self._tag_score(query_tags=query.tags, matched_tags=matched_tags)
        importance_score = self._importance_score(record.importance)
        confidence_score = record.confidence
        recency_score = self._recency_score(record=record)

        weighted = (
            text_score * self._config.text_weight
            + kind_score * self._config.kind_weight
            + scope_score * self._config.scope_weight
            + tag_score * self._config.tag_weight
            + importance_score * self._config.importance_weight
            + confidence_score * self._config.confidence_weight
            + recency_score * self._config.recency_weight
        )
        total_weight = (
            self._config.text_weight
            + self._config.kind_weight
            + self._config.scope_weight
            + self._config.tag_weight
            + self._config.importance_weight
            + self._config.confidence_weight
            + self._config.recency_weight
        )
        final_score = max(0.0, min(1.0, weighted / total_weight))

        return MemoryRetrievalScoreBreakdown(
            text_score=text_score,
            kind_score=kind_score,
            scope_score=scope_score,
            tag_score=tag_score,
            importance_score=importance_score,
            confidence_score=confidence_score,
            recency_score=recency_score,
            final_score=final_score,
            matched_terms=tuple(sorted(matched_terms)),
            matched_tags=tuple(sorted(matched_tags)),
            match_kinds=self._match_kinds(
                query=query,
                matched_terms=matched_terms,
                matched_tags=matched_tags,
                breakdown_scores={
                    "text": text_score,
                    "kind": kind_score,
                    "scope": scope_score,
                    "tag": tag_score,
                    "importance": importance_score,
                    "confidence": confidence_score,
                    "recency": recency_score,
                },
            ),
        )

    def explain(
        self,
        *,
        record: MemoryRecord,
        query: MemoryQuery,
        breakdown: MemoryRetrievalScoreBreakdown,
    ) -> MemoryRetrievalExplanation:
        """
        Build mandatory retrieval explanation.
        """

        return MemoryRetrievalExplanation(
            source=record.source,
            reason=self.reason(record=record, query=query, breakdown=breakdown),
            confidence=min(record.confidence, breakdown.final_score),
            retrieved_at=utc_now(),
            policy_classification=self.policy_classification(record=record),
            metadata={
                "memory_id": record.memory_id,
                "query_id": query.query_id,
                "kind": record.kind.value,
                "scope": record.scope.value,
                "sensitivity": record.sensitivity.value,
                "importance": record.importance.value,
                "score_breakdown": {
                    "text": breakdown.text_score,
                    "kind": breakdown.kind_score,
                    "scope": breakdown.scope_score,
                    "tag": breakdown.tag_score,
                    "importance": breakdown.importance_score,
                    "confidence": breakdown.confidence_score,
                    "recency": breakdown.recency_score,
                    "final": breakdown.final_score,
                },
                "matched_terms": breakdown.matched_terms,
                "matched_tags": breakdown.matched_tags,
                "match_kinds": tuple(kind.value for kind in breakdown.match_kinds),
            },
        )

    def reason(
        self,
        *,
        record: MemoryRecord,
        query: MemoryQuery,
        breakdown: MemoryRetrievalScoreBreakdown,
    ) -> str:
        """
        Build concise human-readable retrieval reason.
        """

        parts: list[str] = []

        if query.text is None:
            parts.append("included because query requested recent available memory")

        if breakdown.matched_terms:
            parts.append(
                "matched query terms: " + ", ".join(breakdown.matched_terms)
            )

        if query.kinds:
            parts.append(f"matched requested kind: {record.kind.value}")

        if query.scopes:
            parts.append(f"matched requested scope: {record.scope.value}")

        if breakdown.matched_tags:
            parts.append("matched tags: " + ", ".join(breakdown.matched_tags))

        parts.append(f"importance={record.importance.value}")
        parts.append(f"confidence={record.confidence:.2f}")
        parts.append(f"score={breakdown.final_score:.2f}")
        parts.append(
            "policy=" + self.policy_classification(record=record).value
        )

        return "; ".join(parts)

    @staticmethod
    def policy_classification(
        *,
        record: MemoryRecord,
    ) -> MemoryPolicyClassification:
        """
        Classify retrieval policy for a record.

        Step 2 provides a baseline classification. Later policy steps can
        strengthen this.
        """

        if record.sensitivity == MemorySensitivity.SENSITIVE:
            return MemoryPolicyClassification.RESTRICTED

        return MemoryPolicyClassification.ALLOWED

    @staticmethod
    def query_terms(text: str | None) -> set[str]:
        if text is None:
            return set()

        return MemoryRetrievalScorer._terms(text)

    @staticmethod
    def record_terms(text: str) -> set[str]:
        return MemoryRetrievalScorer._terms(text)

    @staticmethod
    def _matched_terms(
        query_text: str | None,
        record_text: str,
    ) -> set[str]:
        if query_text is None:
            return set()

        return MemoryRetrievalScorer._terms(query_text) & MemoryRetrievalScorer._terms(
            record_text
        )

    @staticmethod
    def _matched_tags(
        query_tags: tuple[str, ...],
        record_tags: tuple[str, ...],
    ) -> set[str]:
        return set(query_tags) & set(record_tags)

    @staticmethod
    def _text_score(
        *,
        query_text: str | None,
        record_text: str,
        matched_terms: set[str],
    ) -> float:
        if query_text is None:
            return 1.0

        query_terms = MemoryRetrievalScorer._terms(query_text)

        if not query_terms:
            return 1.0

        if not matched_terms:
            return 0.0

        return len(matched_terms) / len(query_terms)

    @staticmethod
    def _kind_score(
        *,
        record: MemoryRecord,
        query: MemoryQuery,
    ) -> float:
        if not query.kinds:
            return 0.5

        return 1.0 if record.kind in query.kinds else 0.0

    @staticmethod
    def _scope_score(
        *,
        record: MemoryRecord,
        query: MemoryQuery,
    ) -> float:
        if not query.scopes:
            return 0.5

        return 1.0 if record.scope in query.scopes else 0.0

    @staticmethod
    def _tag_score(
        *,
        query_tags: tuple[str, ...],
        matched_tags: set[str],
    ) -> float:
        if not query_tags:
            return 0.0

        return len(matched_tags) / len(query_tags)

    @staticmethod
    def _importance_score(importance: MemoryImportance) -> float:
        scores = {
            MemoryImportance.LOW: 0.25,
            MemoryImportance.NORMAL: 0.5,
            MemoryImportance.HIGH: 0.8,
            MemoryImportance.CRITICAL: 1.0,
        }

        return scores[importance]

    def _recency_score(
        self,
        *,
        record: MemoryRecord,
    ) -> float:
        age_seconds = max(0.0, (utc_now() - record.updated_at).total_seconds())
        normalized = 1.0 - min(1.0, age_seconds / self._config.recency_window_seconds)

        return max(0.0, min(1.0, normalized))

    @staticmethod
    def _match_kinds(
        *,
        query: MemoryQuery,
        matched_terms: set[str],
        matched_tags: set[str],
        breakdown_scores: dict[str, float],
    ) -> tuple[MemoryRetrievalMatchKind, ...]:
        kinds: list[MemoryRetrievalMatchKind] = []

        if query.text is None:
            kinds.append(MemoryRetrievalMatchKind.EMPTY_QUERY)

        if matched_terms:
            kinds.append(MemoryRetrievalMatchKind.TEXT_OVERLAP)

        if breakdown_scores["kind"] >= 1.0:
            kinds.append(MemoryRetrievalMatchKind.KIND_MATCH)

        if breakdown_scores["scope"] >= 1.0:
            kinds.append(MemoryRetrievalMatchKind.SCOPE_MATCH)

        if matched_tags:
            kinds.append(MemoryRetrievalMatchKind.TAG_MATCH)

        if breakdown_scores["importance"] >= 0.8:
            kinds.append(MemoryRetrievalMatchKind.IMPORTANCE_BOOST)

        if breakdown_scores["confidence"] >= 0.8:
            kinds.append(MemoryRetrievalMatchKind.CONFIDENCE_BOOST)

        if breakdown_scores["recency"] >= 0.8:
            kinds.append(MemoryRetrievalMatchKind.RECENCY_BOOST)

        return tuple(dict.fromkeys(kinds))

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {
            term.strip(".,!?;:()[]{}\"'").casefold()
            for term in text.split()
            if term.strip(".,!?;:()[]{}\"'")
        }