from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from jarvis.memory import (
    InMemoryMemoryStore,
    MemoryImportance,
    MemoryKind,
    MemoryPolicyClassification,
    MemoryQuery,
    MemoryRecord,
    MemoryRetrievalMatchKind,
    MemoryRetrievalScorer,
    MemoryRetrievalScorerConfig,
    MemoryScope,
    MemorySensitivity,
    MemoryWriteRequest,
)


def make_record(
    *,
    text: str = "User is building a personal JARVIS memory runtime.",
    kind: MemoryKind = MemoryKind.PROJECT,
    scope: MemoryScope = MemoryScope.USER,
    importance: MemoryImportance = MemoryImportance.NORMAL,
    confidence: float = 1.0,
    sensitivity: MemorySensitivity = MemorySensitivity.PRIVATE,
    tags: tuple[str, ...] = ("jarvis", "memory"),
    updated_at: datetime | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        kind=kind,
        scope=scope,
        text=text,
        importance=importance,
        confidence=confidence,
        sensitivity=sensitivity,
        tags=tags,
        updated_at=updated_at or datetime.now(UTC),
    )


def test_retrieval_scorer_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        MemoryRetrievalScorerConfig(text_weight=-0.1).validate()

    with pytest.raises(ValueError):
        MemoryRetrievalScorerConfig(
            text_weight=0,
            kind_weight=0,
            scope_weight=0,
            tag_weight=0,
            importance_weight=0,
            confidence_weight=0,
            recency_weight=0,
        ).validate()

    with pytest.raises(ValueError):
        MemoryRetrievalScorerConfig(recency_window_seconds=0).validate()


def test_score_breakdown_rejects_invalid_scores() -> None:
    with pytest.raises(ValidationError):
        from jarvis.memory import MemoryRetrievalScoreBreakdown

        MemoryRetrievalScoreBreakdown(
            text_score=1.1,
            kind_score=0,
            scope_score=0,
            tag_score=0,
            importance_score=0,
            confidence_score=0,
            recency_score=0,
            final_score=0,
        )


def test_retrieval_scorer_scores_text_kind_scope_and_tags() -> None:
    scorer = MemoryRetrievalScorer()
    record = make_record(
        text="JARVIS memory runtime stores personal context.",
        kind=MemoryKind.PROJECT,
        scope=MemoryScope.USER,
        tags=("jarvis", "memory"),
    )
    query = MemoryQuery(
        text="memory runtime",
        kinds=(MemoryKind.PROJECT,),
        scopes=(MemoryScope.USER,),
        tags=("jarvis",),
    )

    breakdown = scorer.score(record=record, query=query)

    assert breakdown.text_score == 1.0
    assert breakdown.kind_score == 1.0
    assert breakdown.scope_score == 1.0
    assert breakdown.tag_score == 1.0
    assert breakdown.final_score > 0.7
    assert breakdown.matched_terms == ("memory", "runtime")
    assert breakdown.matched_tags == ("jarvis",)
    assert MemoryRetrievalMatchKind.TEXT_OVERLAP in breakdown.match_kinds
    assert MemoryRetrievalMatchKind.KIND_MATCH in breakdown.match_kinds
    assert MemoryRetrievalMatchKind.SCOPE_MATCH in breakdown.match_kinds
    assert MemoryRetrievalMatchKind.TAG_MATCH in breakdown.match_kinds


def test_retrieval_scorer_scores_empty_query_as_available_memory() -> None:
    scorer = MemoryRetrievalScorer()
    record = make_record()
    query = MemoryQuery()

    breakdown = scorer.score(record=record, query=query)

    assert breakdown.text_score == 1.0
    assert MemoryRetrievalMatchKind.EMPTY_QUERY in breakdown.match_kinds


def test_retrieval_scorer_explains_required_audit_fields() -> None:
    scorer = MemoryRetrievalScorer()
    record = make_record(
        text="JARVIS memory retrieval must explain itself.",
        importance=MemoryImportance.HIGH,
    )
    query = MemoryQuery(text="memory retrieval")
    breakdown = scorer.score(record=record, query=query)
    explanation = scorer.explain(
        record=record,
        query=query,
        breakdown=breakdown,
    )

    assert explanation.source == record.source
    assert "matched query terms" in explanation.reason
    assert explanation.confidence > 0
    assert explanation.retrieved_at.tzinfo is not None
    assert explanation.policy_classification == MemoryPolicyClassification.ALLOWED
    assert explanation.metadata["score_breakdown"]["final"] == breakdown.final_score
    assert "TEXT_OVERLAP".lower() not in explanation.reason.lower()


def test_retrieval_scorer_marks_sensitive_as_restricted() -> None:
    scorer = MemoryRetrievalScorer()
    record = make_record(sensitivity=MemorySensitivity.SENSITIVE)

    assert scorer.policy_classification(
        record=record
    ) == MemoryPolicyClassification.RESTRICTED


def test_retrieval_scorer_recency_score_decays() -> None:
    scorer = MemoryRetrievalScorer(
        config=MemoryRetrievalScorerConfig(recency_window_seconds=60)
    )
    fresh = make_record(updated_at=datetime.now(UTC))
    old = make_record(updated_at=datetime.now(UTC) - timedelta(seconds=120))

    fresh_score = scorer.score(record=fresh, query=MemoryQuery()).recency_score
    old_score = scorer.score(record=old, query=MemoryQuery()).recency_score

    assert fresh_score > old_score
    assert old_score == 0.0


def test_in_memory_store_uses_retrieval_scorer_breakdown() -> None:
    store = InMemoryMemoryStore()
    store.write(
        MemoryWriteRequest(
            kind=MemoryKind.PROJECT,
            text="JARVIS memory retrieval scoring is explainable.",
            importance=MemoryImportance.HIGH,
            tags=("jarvis", "memory"),
        )
    )

    result = store.retrieve(
        MemoryQuery(
            text="memory retrieval",
            kinds=(MemoryKind.PROJECT,),
            tags=("jarvis",),
        )
    )

    assert result.result_count == 1
    search_result = result.results[0]

    assert search_result.score > 0.0
    assert "score_breakdown" in search_result.metadata
    assert search_result.explanation.metadata["matched_terms"] == (
        "memory",
        "retrieval",
    )
    assert search_result.explanation.policy_classification == (
        MemoryPolicyClassification.ALLOWED
    )


def test_retrieval_match_kind_values_are_stable() -> None:
    assert MemoryRetrievalMatchKind.EMPTY_QUERY.value == "empty_query"
    assert MemoryRetrievalMatchKind.TEXT_OVERLAP.value == "text_overlap"
    assert MemoryRetrievalMatchKind.KIND_MATCH.value == "kind_match"
    assert MemoryRetrievalMatchKind.RECENCY_BOOST.value == "recency_boost"