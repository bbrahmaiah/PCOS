from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.latency import (
    EmbeddingCache,
    EmbeddingCacheConfig,
    MemoryRetrievalQuery,
    MemoryStreamKind,
    MemoryStreamStatus,
    SpeculativeWorkHint,
    SpeculativeWorkKind,
    StreamingMemoryReason,
    StreamingMemoryRuntime,
    StreamingMemoryRuntimeConfig,
    memory_query_from_hint,
)


def test_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        StreamingMemoryRuntimeConfig(name=" ").validate()


def test_config_rejects_invalid_targets() -> None:
    with pytest.raises(ValueError):
        StreamingMemoryRuntimeConfig(first_result_target_ms=0).validate()

    with pytest.raises(ValueError):
        StreamingMemoryRuntimeConfig(full_context_target_ms=0).validate()


def test_config_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError):
        StreamingMemoryRuntimeConfig(early_stop_confidence=2).validate()


def test_embedding_cache_rejects_invalid_size() -> None:
    with pytest.raises(ValueError):
        EmbeddingCacheConfig(max_entries=0).validate()


def test_embedding_cache_lru_limit() -> None:
    cache = EmbeddingCache(config=EmbeddingCacheConfig(max_entries=2))

    cache.get_or_create("one")
    cache.get_or_create("two")
    cache.get_or_create("three")

    assert cache.size() == 2
    assert cache.get("one") is None
    assert cache.get("two") is not None
    assert cache.get("three") is not None


def test_query_requires_text() -> None:
    with pytest.raises(ValidationError):
        MemoryRetrievalQuery(text=" ")


def test_runtime_creates_session() -> None:
    runtime = StreamingMemoryRuntime()
    state = runtime.create_session(query_text="jarvis memory")

    assert state.status == MemoryStreamStatus.CREATED
    assert runtime.snapshot().session_count == 1


def test_runtime_starts_session() -> None:
    runtime = StreamingMemoryRuntime()
    state = runtime.create_session(query_text="jarvis memory")

    result = runtime.start_session(state.session_id)

    assert result.success is True
    assert result.reason == StreamingMemoryReason.SESSION_STARTED
    assert result.status == MemoryStreamStatus.ACTIVE


def test_runtime_rejects_missing_session_start() -> None:
    runtime = StreamingMemoryRuntime()

    result = runtime.start_session("missing")

    assert result.success is False
    assert result.reason == StreamingMemoryReason.SESSION_NOT_FOUND


def test_runtime_streams_memory_results() -> None:
    runtime = StreamingMemoryRuntime()
    state = runtime.create_session(query_text="jarvis memory")

    runtime.start_session(state.session_id)
    results = runtime.run_available_streams(state.session_id)

    assert results
    assert any(result.memory_result is not None for result in results)
    assert runtime.results_for(state.session_id)


def test_results_are_priority_ordered() -> None:
    runtime = StreamingMemoryRuntime()
    state = runtime.create_session(query_text="jarvis memory")

    runtime.start_session(state.session_id)
    runtime.run_available_streams(state.session_id)

    results = runtime.results_for(state.session_id)

    assert results[0].priority >= results[-1].priority


def test_first_result_latency_is_recorded() -> None:
    runtime = StreamingMemoryRuntime()
    state = runtime.create_session(query_text="jarvis memory")

    runtime.start_session(state.session_id)
    runtime.run_available_streams(state.session_id)
    updated = runtime.state_for(state.session_id)

    assert updated is not None
    assert updated.first_result_latency_ms() is not None


def test_context_confidence_updates_incrementally() -> None:
    runtime = StreamingMemoryRuntime()
    state = runtime.create_session(query_text="jarvis memory")

    runtime.start_session(state.session_id)
    runtime.run_available_streams(state.session_id)
    updated = runtime.state_for(state.session_id)

    assert updated is not None
    assert updated.context_confidence > 0


def test_early_stop_cancels_semantic_stream() -> None:
    runtime = StreamingMemoryRuntime(
        config=StreamingMemoryRuntimeConfig(early_stop_confidence=0.80)
    )
    state = runtime.create_session(query_text="jarvis memory")

    runtime.start_session(state.session_id)
    runtime.run_available_streams(state.session_id)
    updated = runtime.state_for(state.session_id)

    assert updated is not None
    assert updated.early_stopped is True
    assert any(
        event.reason == StreamingMemoryReason.SEMANTIC_STREAM_CANCELLED
        for event in runtime.events_for(state.session_id)
    )


def test_complete_session_builds_report() -> None:
    runtime = StreamingMemoryRuntime()
    state = runtime.create_session(query_text="jarvis memory")

    runtime.start_session(state.session_id)
    runtime.run_available_streams(state.session_id)
    report = runtime.complete_session(state.session_id)

    assert report.status == MemoryStreamStatus.COMPLETED
    assert report.result_count > 0
    assert report.profile_count > 0
    assert report.episodic_count > 0
    assert report.first_result_latency_ms is not None
    assert report.profiler_report is not None


def test_complete_rejects_missing_session() -> None:
    runtime = StreamingMemoryRuntime()

    with pytest.raises(ValueError):
        runtime.complete_session("missing")


def test_cancel_session() -> None:
    runtime = StreamingMemoryRuntime()
    state = runtime.create_session(query_text="jarvis memory")

    runtime.start_session(state.session_id)
    result = runtime.cancel_session(state.session_id)

    assert result.success is True
    assert result.status == MemoryStreamStatus.CANCELLED


def test_fail_session() -> None:
    runtime = StreamingMemoryRuntime()
    state = runtime.create_session(query_text="jarvis memory")

    runtime.start_session(state.session_id)
    result = runtime.fail_session(state.session_id, error="memory failed")

    assert result.success is True
    assert result.status == MemoryStreamStatus.FAILED


def test_snapshot_tracks_counts() -> None:
    runtime = StreamingMemoryRuntime()
    state = runtime.create_session(query_text="jarvis memory")

    runtime.start_session(state.session_id)
    runtime.run_available_streams(state.session_id)
    runtime.complete_session(state.session_id)
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.completed_count == 1
    assert snapshot.result_count > 0
    assert snapshot.report_count == 1


def test_reports_are_queryable() -> None:
    runtime = StreamingMemoryRuntime()
    state = runtime.create_session(query_text="jarvis memory")

    runtime.start_session(state.session_id)
    runtime.run_available_streams(state.session_id)
    report = runtime.complete_session(state.session_id)

    assert runtime.latest_report() == report
    assert runtime.reports() == (report,)


def test_reset_clears_state() -> None:
    runtime = StreamingMemoryRuntime()
    state = runtime.create_session(query_text="jarvis memory")

    runtime.start_session(state.session_id)
    runtime.run_available_streams(state.session_id)
    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.result_count == 0
    assert snapshot.last_reason == StreamingMemoryReason.RUNTIME_RESET


def test_memory_query_from_hint_requires_memory_prefetch() -> None:
    hint = SpeculativeWorkHint(
        session_id="session",
        intent_id="intent",
        kind=SpeculativeWorkKind.CONTEXT_PREWARM,
        reason="not memory",
        confidence=0.9,
    )

    with pytest.raises(ValueError):
        memory_query_from_hint(text="hello", hint=hint)


def test_memory_query_from_hint() -> None:
    hint = SpeculativeWorkHint(
        session_id="session",
        intent_id="intent",
        kind=SpeculativeWorkKind.MEMORY_PREFETCH,
        reason="memory",
        confidence=0.9,
    )

    query = memory_query_from_hint(text="hello", hint=hint)

    assert query.text == "hello"
    assert query.source_hint_id == hint.hint_id
    assert query.speculative is True


def test_enum_values_are_stable() -> None:
    assert MemoryStreamKind.EPISODIC.value == "episodic"
    assert MemoryStreamStatus.ACTIVE.value == "active"
    assert StreamingMemoryReason.RESULT_ACCEPTED.value == "result_accepted"