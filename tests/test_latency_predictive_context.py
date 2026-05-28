from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.latency import (
    ContextBuildRequest,
    ContextCompressionPolicy,
    ContextFragment,
    ContextFragmentCache,
    ContextFragmentCacheConfig,
    ContextFragmentKind,
    ContextSnapshot,
    MemoryResultSource,
    MemoryStreamKind,
    MemoryStreamResult,
    PredictiveContextBuilderRuntime,
    PredictiveContextReason,
    PredictiveContextRuntimeConfig,
    PredictiveContextStatus,
)


def test_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        PredictiveContextRuntimeConfig(name=" ").validate()


def test_config_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError):
        PredictiveContextRuntimeConfig(ready_confidence_threshold=2).validate()


def test_config_requires_two_uncompressed_turns() -> None:
    with pytest.raises(ValueError):
        PredictiveContextRuntimeConfig(max_previous_turns_uncompressed=1).validate()


def test_cache_config_rejects_invalid_size() -> None:
    with pytest.raises(ValueError):
        ContextFragmentCacheConfig(max_entries=0).validate()


def test_fragment_requires_text() -> None:
    with pytest.raises(ValidationError):
        ContextFragment(
            kind=ContextFragmentKind.USER_PROFILE,
            text=" ",
            priority=10,
            token_estimate=1,
        )


def test_cache_lru_limit() -> None:
    cache = ContextFragmentCache(config=ContextFragmentCacheConfig(max_entries=2))

    cache.set("one", _fragment("one"))
    cache.set("two", _fragment("two"))
    cache.set("three", _fragment("three"))

    assert cache.size() == 2
    assert cache.get("one") is None
    assert cache.get("two") is not None
    assert cache.get("three") is not None


def test_runtime_creates_session() -> None:
    runtime = PredictiveContextBuilderRuntime()
    state = runtime.create_session(request=_request())

    assert state.status == PredictiveContextStatus.CREATED
    assert runtime.snapshot().session_count == 1


def test_runtime_starts_and_seeds_context() -> None:
    runtime = PredictiveContextBuilderRuntime()
    state = runtime.create_session(request=_request())

    result = runtime.start_session(state.session_id)

    assert result.success is True
    assert result.status == PredictiveContextStatus.BUILDING
    assert len(runtime.fragments_for(state.session_id)) >= 3


def test_runtime_extends_previous_snapshot() -> None:
    previous = ContextSnapshot(
        turn_id="previous",
        fragments=(_fragment("old context"),),
        token_estimate=2,
    )
    runtime = PredictiveContextBuilderRuntime()
    state = runtime.create_session(
        request=_request(previous_snapshot=previous)
    )

    runtime.start_session(state.session_id)

    assert any(
        fragment.text == "old context"
        for fragment in runtime.fragments_for(state.session_id)
    )


def test_cache_hit_on_second_session() -> None:
    cache = ContextFragmentCache()
    runtime = PredictiveContextBuilderRuntime(cache=cache)

    first = runtime.create_session(request=_request(turn_id="one"))
    runtime.start_session(first.session_id)

    second = runtime.create_session(request=_request(turn_id="two"))
    runtime.start_session(second.session_id)
    updated = runtime.state_for(second.session_id)

    assert updated is not None
    assert updated.cache_hit_count >= 2


def test_consuming_profile_and_episodic_marks_ready() -> None:
    runtime = PredictiveContextBuilderRuntime()
    state = runtime.create_session(request=_request())

    runtime.start_session(state.session_id)
    runtime.consume_memory_result(
        session_id=state.session_id,
        result=_memory_result(
            session_id=state.session_id,
            kind=MemoryStreamKind.PROFILE,
            source=MemoryResultSource.USER_PROFILE,
            confidence=0.92,
        ),
    )
    result = runtime.consume_memory_result(
        session_id=state.session_id,
        result=_memory_result(
            session_id=state.session_id,
            kind=MemoryStreamKind.EPISODIC,
            source=MemoryResultSource.EPISODIC_MEMORY,
            confidence=0.90,
        ),
    )

    assert result.success is True

    updated = runtime.state_for(state.session_id)

    assert updated is not None
    assert updated.ready_for_llm is True
    assert updated.status == PredictiveContextStatus.READY


def test_semantic_can_append_after_ready() -> None:
    runtime = PredictiveContextBuilderRuntime()
    state = runtime.create_session(request=_request())

    runtime.start_session(state.session_id)
    _make_ready(runtime, state.session_id)
    runtime.consume_memory_result(
        session_id=state.session_id,
        result=_memory_result(
            session_id=state.session_id,
            kind=MemoryStreamKind.SEMANTIC,
            source=MemoryResultSource.SEMANTIC_MEMORY,
            confidence=0.80,
        ),
    )

    assert any(
        event.reason == PredictiveContextReason.SEMANTIC_APPENDED
        for event in runtime.events_for(state.session_id)
    )


def test_complete_session_builds_snapshot_and_report() -> None:
    runtime = PredictiveContextBuilderRuntime()
    state = runtime.create_session(request=_request())

    runtime.start_session(state.session_id)
    _make_ready(runtime, state.session_id)
    report = runtime.complete_session(state.session_id)

    assert report.status == PredictiveContextStatus.COMPLETED
    assert report.snapshot.fragments
    assert report.ready_for_llm is True
    assert report.profiler_report is not None


def test_complete_rejects_missing_session() -> None:
    runtime = PredictiveContextBuilderRuntime()

    with pytest.raises(ValueError):
        runtime.complete_session("missing")


def test_compression_does_not_compress_last_two_fragments() -> None:
    runtime = PredictiveContextBuilderRuntime()
    snapshot = ContextSnapshot(
        turn_id="turn",
        fragments=(
            _fragment("old one"),
            _fragment("old two"),
            _fragment("previous turn"),
            _fragment("current turn"),
        ),
        token_estimate=8,
    )

    compressed = runtime.compress_older_fragments(snapshot=snapshot)

    assert len(compressed.fragments) == 3
    assert compressed.fragments[0].kind == ContextFragmentKind.SUMMARY
    assert compressed.fragments[-2].text == "previous turn"
    assert compressed.fragments[-1].text == "current turn"


def test_cache_invalidated_on_memory_write() -> None:
    cache = ContextFragmentCache()
    runtime = PredictiveContextBuilderRuntime(cache=cache)
    state = runtime.create_session(request=_request())

    runtime.start_session(state.session_id)
    assert cache.size() > 0

    runtime.invalidate_cache_on_memory_write()

    assert cache.size() == 0
    assert runtime.snapshot().last_reason == (
        PredictiveContextReason.MEMORY_WRITE_INVALIDATED_CACHE
    )


def test_cancel_session() -> None:
    runtime = PredictiveContextBuilderRuntime()
    state = runtime.create_session(request=_request())

    runtime.start_session(state.session_id)
    result = runtime.cancel_session(state.session_id)

    assert result.success is True
    assert result.status == PredictiveContextStatus.CANCELLED


def test_fail_session() -> None:
    runtime = PredictiveContextBuilderRuntime()
    state = runtime.create_session(request=_request())

    runtime.start_session(state.session_id)
    result = runtime.fail_session(state.session_id, error="context failed")

    assert result.success is True
    assert result.status == PredictiveContextStatus.FAILED


def test_snapshot_tracks_counts() -> None:
    runtime = PredictiveContextBuilderRuntime()
    state = runtime.create_session(request=_request())

    runtime.start_session(state.session_id)
    _make_ready(runtime, state.session_id)
    runtime.complete_session(state.session_id)

    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.completed_count == 1
    assert snapshot.fragment_count > 0
    assert snapshot.report_count == 1


def test_reports_are_queryable() -> None:
    runtime = PredictiveContextBuilderRuntime()
    state = runtime.create_session(request=_request())

    runtime.start_session(state.session_id)
    _make_ready(runtime, state.session_id)
    report = runtime.complete_session(state.session_id)

    assert runtime.latest_report() == report
    assert runtime.reports() == (report,)


def test_reset_clears_state() -> None:
    runtime = PredictiveContextBuilderRuntime()
    state = runtime.create_session(request=_request())

    runtime.start_session(state.session_id)
    runtime.reset()

    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.fragment_count == 0
    assert snapshot.last_reason == PredictiveContextReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert PredictiveContextStatus.BUILDING.value == "building"
    assert ContextFragmentKind.USER_PROFILE.value == "user_profile"
    assert ContextCompressionPolicy.COMPRESSED.value == "compressed"


def _request(
    *,
    turn_id: str = "turn",
    previous_snapshot: ContextSnapshot | None = None,
) -> ContextBuildRequest:
    return ContextBuildRequest(
        turn_id=turn_id,
        user_text="Build predictive context",
        previous_snapshot=previous_snapshot,
    )


def _fragment(text: str) -> ContextFragment:
    return ContextFragment(
        kind=ContextFragmentKind.RECENT_TURN,
        text=text,
        priority=50,
        token_estimate=max(1, len(text.split())),
    )


def _memory_result(
    *,
    session_id: str,
    kind: MemoryStreamKind,
    source: MemoryResultSource,
    confidence: float,
) -> MemoryStreamResult:
    return MemoryStreamResult(
        session_id=session_id,
        query_id="query",
        stream_kind=kind,
        source=source,
        text=f"{kind.value} context",
        priority=90,
        relevance=0.9,
        confidence=confidence,
    )


def _make_ready(
    runtime: PredictiveContextBuilderRuntime,
    session_id: str,
) -> None:
    runtime.consume_memory_result(
        session_id=session_id,
        result=_memory_result(
            session_id=session_id,
            kind=MemoryStreamKind.PROFILE,
            source=MemoryResultSource.USER_PROFILE,
            confidence=0.92,
        ),
    )
    runtime.consume_memory_result(
        session_id=session_id,
        result=_memory_result(
            session_id=session_id,
            kind=MemoryStreamKind.EPISODIC,
            source=MemoryResultSource.EPISODIC_MEMORY,
            confidence=0.90,
        ),
    )