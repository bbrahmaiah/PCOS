from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from jarvis.memory import (
    MemoryImportance,
    MemoryKind,
    MemoryPolicyClassification,
    MemoryQuery,
    MemoryRecord,
    MemoryRetention,
    MemoryRetrievalExplanation,
    MemoryRetrievalResult,
    MemoryScope,
    MemorySearchResult,
    MemorySensitivity,
    MemorySource,
    MemoryWriteRequest,
)


def make_record() -> MemoryRecord:
    return MemoryRecord(
        kind=MemoryKind.PROJECT,
        text="User is building a local JARVIS cognition OS.",
    )


def make_explanation() -> MemoryRetrievalExplanation:
    return MemoryRetrievalExplanation(
        source=MemorySource.CONVERSATION,
        reason="matched project memory query",
        confidence=0.9,
        policy_classification=MemoryPolicyClassification.ALLOWED,
    )


def test_memory_record_defaults_are_safe() -> None:
    record = make_record()

    assert record.memory_id
    assert record.kind == MemoryKind.PROJECT
    assert record.scope == MemoryScope.USER
    assert record.source == MemorySource.CONVERSATION
    assert record.sensitivity == MemorySensitivity.PRIVATE
    assert record.importance == MemoryImportance.NORMAL
    assert record.retention == MemoryRetention.PERSISTENT
    assert record.confidence == 1.0
    assert record.expired() is False


def test_memory_record_rejects_empty_required_fields() -> None:
    with pytest.raises(ValidationError):
        MemoryRecord(kind=MemoryKind.SEMANTIC, text=" ")

    with pytest.raises(ValidationError):
        MemoryRecord(
            memory_id=" ",
            kind=MemoryKind.SEMANTIC,
            text="valid memory",
        )


def test_memory_record_cleans_tags() -> None:
    record = MemoryRecord(
        kind=MemoryKind.PREFERENCE,
        text="User prefers concise spoken responses.",
        tags=(" Jarvis ", "jarvis", " Voice ", " "),
    )

    assert record.tags == ("jarvis", "voice")


def test_memory_record_expiration() -> None:
    expired = MemoryRecord(
        kind=MemoryKind.EPISODIC,
        text="Old temporary note.",
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    active = MemoryRecord(
        kind=MemoryKind.EPISODIC,
        text="Fresh temporary note.",
        expires_at=datetime.now(UTC) + timedelta(seconds=60),
    )

    assert expired.expired() is True
    assert active.expired() is False


def test_memory_query_defaults_are_safe() -> None:
    query = MemoryQuery(text="jarvis memory")

    assert query.query_id
    assert query.text == "jarvis memory"
    assert query.max_results == 8
    assert query.min_confidence == 0.0
    assert query.include_expired is False
    assert query.include_sensitive is False


def test_memory_query_cleans_empty_text_and_tags() -> None:
    query = MemoryQuery(text=" ", tags=(" JARVIS ", "jarvis", " "))

    assert query.text is None
    assert query.tags == ("jarvis",)


def test_memory_query_rejects_invalid_bounds() -> None:
    with pytest.raises(ValidationError):
        MemoryQuery(max_results=0)

    with pytest.raises(ValidationError):
        MemoryQuery(max_results=101)

    with pytest.raises(ValidationError):
        MemoryQuery(min_confidence=-0.1)


def test_memory_retrieval_explanation_requires_reason() -> None:
    with pytest.raises(ValidationError):
        MemoryRetrievalExplanation(
            source=MemorySource.CONVERSATION,
            reason=" ",
            confidence=0.8,
        )


def test_memory_retrieval_explanation_has_required_audit_fields() -> None:
    explanation = make_explanation()

    assert explanation.source == MemorySource.CONVERSATION
    assert explanation.reason == "matched project memory query"
    assert explanation.confidence == 0.9
    assert explanation.retrieved_at.tzinfo is not None
    assert explanation.policy_classification == MemoryPolicyClassification.ALLOWED


def test_memory_search_result_model() -> None:
    record = make_record()
    explanation = make_explanation()
    result = MemorySearchResult(
        record=record,
        score=0.9,
        explanation=explanation,
    )

    assert result.record == record
    assert result.score == 0.9
    assert result.reason == "matched project memory query"
    assert result.policy_classification == MemoryPolicyClassification.ALLOWED


def test_memory_search_result_rejects_invalid_score() -> None:
    with pytest.raises(ValidationError):
        MemorySearchResult(
            record=make_record(),
            score=1.1,
            explanation=make_explanation(),
        )


def test_memory_retrieval_result_exposes_records() -> None:
    record = make_record()
    query = MemoryQuery(text="memory")
    search_result = MemorySearchResult(
        record=record,
        score=0.8,
        explanation=make_explanation(),
    )
    retrieval = MemoryRetrievalResult(
        query=query,
        results=(search_result,),
    )

    assert retrieval.result_count == 1
    assert retrieval.records == (record,)


def test_memory_write_request_to_record() -> None:
    write = MemoryWriteRequest(
        kind=MemoryKind.USER_PROFILE,
        text="User is building JARVIS for education and debugging.",
        scope=MemoryScope.USER,
        source=MemorySource.USER_EXPLICIT,
        importance=MemoryImportance.HIGH,
        tags=("profile", "jarvis"),
    )

    record = write.to_record()

    assert record.kind == MemoryKind.USER_PROFILE
    assert record.text == "User is building JARVIS for education and debugging."
    assert record.source == MemorySource.USER_EXPLICIT
    assert record.importance == MemoryImportance.HIGH
    assert record.tags == ("profile", "jarvis")
    assert record.metadata["write_request_id"] == write.request_id


def test_memory_write_request_cleans_tags() -> None:
    write = MemoryWriteRequest(
        kind=MemoryKind.PREFERENCE,
        text="User prefers direct engineering instructions.",
        tags=(" Direct ", "direct", " Engineering "),
    )

    assert write.tags == ("direct", "engineering")


def test_memory_models_are_frozen() -> None:
    record = make_record()

    with pytest.raises(ValidationError):
        record.text = "mutated"


def test_memory_enum_values_are_stable() -> None:
    assert MemoryKind.EPISODIC.value == "episodic"
    assert MemoryKind.USER_PROFILE.value == "user_profile"
    assert MemoryScope.PROJECT.value == "project"
    assert MemorySensitivity.SENSITIVE.value == "sensitive"
    assert MemoryImportance.CRITICAL.value == "critical"
    assert MemorySource.USER_EXPLICIT.value == "user_explicit"
    assert MemoryRetention.PINNED.value == "pinned"
    assert MemoryPolicyClassification.BLOCKED.value == "blocked"