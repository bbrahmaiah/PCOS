from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from jarvis.memory import (
    ExtractiveMemorySummarizer,
    ExtractiveMemorySummarizerConfig,
    MemoryImportance,
    MemoryKind,
    MemoryPolicyClassification,
    MemoryRecord,
    MemoryRetention,
    MemorySensitivity,
    MemorySource,
    MemorySummarizer,
    MemorySummary,
    MemorySummaryKind,
    MemorySummaryRequest,
    MemorySummarySource,
    MemorySummaryStatus,
)


def make_record(
    *,
    text: str = "JARVIS memory runtime stores governed context.",
    kind: MemoryKind = MemoryKind.SEMANTIC,
    importance: MemoryImportance = MemoryImportance.NORMAL,
    confidence: float = 1.0,
    sensitivity: MemorySensitivity = MemorySensitivity.PRIVATE,
    updated_at: datetime | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        kind=kind,
        text=text,
        importance=importance,
        confidence=confidence,
        sensitivity=sensitivity,
        updated_at=updated_at or datetime.now(UTC),
    )


def test_extractive_summarizer_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        ExtractiveMemorySummarizerConfig(name=" ").validate()

    with pytest.raises(ValueError):
        ExtractiveMemorySummarizerConfig(separator="").validate()

    with pytest.raises(ValueError):
        ExtractiveMemorySummarizerConfig(max_source_records=0).validate()


def test_memory_summary_source_requires_text_fields() -> None:
    with pytest.raises(ValidationError):
        MemorySummarySource(
            memory_id=" ",
            kind=MemoryKind.SEMANTIC,
            source=MemorySource.CONVERSATION,
            reason="valid",
            confidence=1.0,
            policy_classification=MemoryPolicyClassification.ALLOWED,
        )

    with pytest.raises(ValidationError):
        MemorySummarySource(
            memory_id="memory-1",
            kind=MemoryKind.SEMANTIC,
            source=MemorySource.CONVERSATION,
            reason=" ",
            confidence=1.0,
            policy_classification=MemoryPolicyClassification.ALLOWED,
        )


def test_memory_summary_requires_text() -> None:
    source = MemorySummarySource(
        memory_id="memory-1",
        kind=MemoryKind.SEMANTIC,
        source=MemorySource.CONVERSATION,
        reason="selected",
        confidence=1.0,
        policy_classification=MemoryPolicyClassification.ALLOWED,
    )

    with pytest.raises(ValidationError):
        MemorySummary(
            text=" ",
            summary_kind=MemorySummaryKind.EXTRACTIVE,
            sources=(source,),
            confidence=1.0,
        )


def test_extractive_summarizer_satisfies_protocol() -> None:
    summarizer: MemorySummarizer = ExtractiveMemorySummarizer()

    result = summarizer.summarize(
        MemorySummaryRequest(records=(make_record(),))
    )

    assert result.succeeded is True


def test_extractive_summarizer_creates_summary() -> None:
    summarizer = ExtractiveMemorySummarizer()
    first = make_record(
        text="High priority memory should be selected first.",
        importance=MemoryImportance.HIGH,
    )
    second = make_record(
        text="Normal memory should still be included.",
        importance=MemoryImportance.NORMAL,
    )

    result = summarizer.summarize(
        MemorySummaryRequest(
            records=(second, first),
            summary_kind=MemorySummaryKind.SEMANTIC_SYNTHESIS,
        )
    )
    snapshot = summarizer.snapshot()

    assert result.succeeded is True
    assert result.summary is not None
    assert result.summary.summary_kind == MemorySummaryKind.SEMANTIC_SYNTHESIS
    assert result.summary.source_count == 2
    assert result.summary.memory_ids[0] == first.memory_id
    assert "High priority memory" in result.summary.text
    assert snapshot.succeeded_count == 1


def test_extractive_summarizer_filters_low_confidence() -> None:
    summarizer = ExtractiveMemorySummarizer()

    result = summarizer.summarize(
        MemorySummaryRequest(
            records=(
                make_record(text="Low confidence.", confidence=0.2),
                make_record(text="High confidence.", confidence=0.9),
            ),
            min_confidence=0.8,
        )
    )

    assert result.succeeded is True
    assert result.summary is not None
    assert "High confidence." in result.summary.text
    assert "Low confidence." not in result.summary.text


def test_extractive_summarizer_filters_sensitive_by_default() -> None:
    summarizer = ExtractiveMemorySummarizer()

    result = summarizer.summarize(
        MemorySummaryRequest(
            records=(
                make_record(
                    text="Sensitive memory.",
                    sensitivity=MemorySensitivity.SENSITIVE,
                ),
                make_record(text="Private memory."),
            )
        )
    )

    assert result.succeeded is True
    assert result.summary is not None
    assert result.summary.text == "Private memory."
    assert result.summary.policy_classification == (
        MemoryPolicyClassification.ALLOWED
    )


def test_extractive_summarizer_can_include_sensitive() -> None:
    summarizer = ExtractiveMemorySummarizer()

    result = summarizer.summarize(
        MemorySummaryRequest(
            records=(
                make_record(
                    text="Sensitive memory.",
                    sensitivity=MemorySensitivity.SENSITIVE,
                ),
            ),
            include_sensitive=True,
        )
    )

    assert result.succeeded is True
    assert result.summary is not None
    assert result.summary.policy_classification == (
        MemoryPolicyClassification.RESTRICTED
    )
    assert result.summary.sources[0].policy_classification == (
        MemoryPolicyClassification.RESTRICTED
    )


def test_extractive_summarizer_filters_expired_by_default() -> None:
    summarizer = ExtractiveMemorySummarizer()
    expired = MemoryRecord(
        kind=MemoryKind.EPISODIC,
        text="Expired memory.",
        retention=MemoryRetention.TEMPORARY,
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    active = make_record(text="Active memory.")

    result = summarizer.summarize(
        MemorySummaryRequest(records=(expired, active))
    )

    assert result.succeeded is True
    assert result.summary is not None
    assert result.summary.text == "Active memory."


def test_extractive_summarizer_returns_empty_when_no_candidates() -> None:
    summarizer = ExtractiveMemorySummarizer()

    result = summarizer.summarize(MemorySummaryRequest(records=()))

    assert result.empty is True
    assert result.summary is None
    assert result.failure_reason == "no memory records eligible for summary"


def test_extractive_summarizer_respects_max_chars() -> None:
    summarizer = ExtractiveMemorySummarizer()
    long_text = "A" * 300

    result = summarizer.summarize(
        MemorySummaryRequest(
            records=(make_record(text=long_text),),
            max_chars=100,
        )
    )

    assert result.succeeded is True
    assert result.summary is not None
    assert len(result.summary.text) <= 100
    assert result.summary.text.endswith("...")


def test_extractive_summarizer_snapshot_and_reset() -> None:
    summarizer = ExtractiveMemorySummarizer()

    summarizer.summarize(MemorySummaryRequest(records=(make_record(),)))
    summarizer.summarize(MemorySummaryRequest(records=()))

    snapshot = summarizer.snapshot()

    assert snapshot.summarized_count == 2
    assert snapshot.succeeded_count == 1
    assert snapshot.empty_count == 1
    assert snapshot.last_status == MemorySummaryStatus.EMPTY

    summarizer.reset()
    reset_snapshot = summarizer.snapshot()

    assert reset_snapshot.summarized_count == 0
    assert reset_snapshot.succeeded_count == 0
    assert reset_snapshot.last_status is None


def test_memory_summary_enum_values_are_stable() -> None:
    assert MemorySummaryKind.EXTRACTIVE.value == "extractive"
    assert MemorySummaryKind.EPISODIC_TIMELINE.value == "episodic_timeline"
    assert MemorySummaryKind.PROFILE_SUMMARY.value == "profile_summary"
    assert MemorySummaryStatus.SUCCEEDED.value == "succeeded"
    assert MemorySummaryStatus.EMPTY.value == "empty"