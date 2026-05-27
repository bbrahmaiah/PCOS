from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.latency import (
    TokenChunk,
    TokenStability,
    TokenStreamChunkKind,
    TokenStreamEventKind,
    TokenStreamReason,
    TokenStreamRuntime,
    TokenStreamRuntimeConfig,
    TokenStreamStatus,
)


def test_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        TokenStreamRuntimeConfig(name=" ").validate()


def test_config_rejects_invalid_buffer_limit() -> None:
    with pytest.raises(ValueError):
        TokenStreamRuntimeConfig(max_buffered_chunks_per_stream=0).validate()


def test_chunk_rejects_empty_stream_id() -> None:
    with pytest.raises(ValidationError):
        TokenChunk(stream_id=" ", sequence=0, text="hello")


def test_runtime_creates_stream() -> None:
    runtime = TokenStreamRuntime()
    state = runtime.create_stream(name="answer")

    assert state.status == TokenStreamStatus.CREATED
    assert runtime.snapshot().stream_count == 1
    assert runtime.events_for(state.stream_id)[0].kind == (
        TokenStreamEventKind.STREAM_CREATED
    )


def test_runtime_starts_stream() -> None:
    runtime = TokenStreamRuntime()
    state = runtime.create_stream(name="answer")

    result = runtime.start_stream(state.stream_id)

    assert result.success is True
    assert result.status == TokenStreamStatus.ACTIVE
    assert result.reason == TokenStreamReason.STREAM_STARTED


def test_runtime_rejects_missing_stream_start() -> None:
    runtime = TokenStreamRuntime()

    result = runtime.start_stream("missing")

    assert result.success is False
    assert result.reason == TokenStreamReason.STREAM_NOT_FOUND


def test_runtime_emits_first_token() -> None:
    runtime = TokenStreamRuntime()
    state = runtime.create_stream(name="answer")

    runtime.start_stream(state.stream_id)
    result = runtime.emit_text(stream_id=state.stream_id, text="Hello")

    assert result.success is True
    assert result.chunk is not None
    assert result.chunk.sequence == 0
    assert result.reason == TokenStreamReason.FIRST_TOKEN_RECORDED

    updated = runtime.state_for(state.stream_id)

    assert updated is not None
    assert updated.first_token_latency_ms() is not None


def test_runtime_emits_multiple_tokens_and_assembles_text() -> None:
    runtime = TokenStreamRuntime()
    state = runtime.create_stream(name="answer")

    runtime.start_stream(state.stream_id)
    runtime.emit_text(stream_id=state.stream_id, text="Hello")
    runtime.emit_text(stream_id=state.stream_id, text=", ")
    runtime.emit_text(stream_id=state.stream_id, text="world")

    updated = runtime.state_for(state.stream_id)

    assert updated is not None
    assert updated.final_text == "Hello, world"
    assert updated.chunk_count == 3


def test_runtime_rejects_emit_before_start() -> None:
    runtime = TokenStreamRuntime()
    state = runtime.create_stream(name="answer")

    result = runtime.emit_text(stream_id=state.stream_id, text="Hello")

    assert result.success is False
    assert result.reason == TokenStreamReason.STREAM_NOT_ACTIVE


def test_runtime_detects_sentence_stability() -> None:
    runtime = TokenStreamRuntime()
    state = runtime.create_stream(name="answer")

    runtime.start_stream(state.stream_id)
    runtime.emit_many(
        stream_id=state.stream_id,
        chunks=("Hello", " ", "world."),
    )

    updated = runtime.state_for(state.stream_id)

    assert updated is not None
    assert updated.stable_sentence_count == 1
    assert any(
        event.kind == TokenStreamEventKind.SENTENCE_STABILIZED
        for event in runtime.events_for(state.stream_id)
    )


def test_runtime_can_disable_sentence_stabilization() -> None:
    runtime = TokenStreamRuntime()
    state = runtime.create_stream(name="answer")

    runtime.start_stream(state.stream_id)
    runtime.emit_many(
        stream_id=state.stream_id,
        chunks=("Hello", " ", "world."),
        stabilize_sentences=False,
    )

    updated = runtime.state_for(state.stream_id)

    assert updated is not None
    assert updated.stable_sentence_count == 0


def test_runtime_applies_backpressure_limit() -> None:
    runtime = TokenStreamRuntime(
        config=TokenStreamRuntimeConfig(max_buffered_chunks_per_stream=1)
    )
    state = runtime.create_stream(name="answer")

    runtime.start_stream(state.stream_id)
    first = runtime.emit_text(stream_id=state.stream_id, text="one")
    second = runtime.emit_text(stream_id=state.stream_id, text="two")

    assert first.success is True
    assert second.success is False
    assert second.reason == TokenStreamReason.BACKPRESSURE_LIMIT_REACHED


def test_runtime_completes_stream() -> None:
    runtime = TokenStreamRuntime()
    state = runtime.create_stream(name="answer")

    runtime.start_stream(state.stream_id)
    runtime.emit_text(stream_id=state.stream_id, text="Hello")
    report = runtime.complete_stream(state.stream_id)

    assert report.status == TokenStreamStatus.COMPLETED
    assert report.final_text == "Hello"
    assert report.chunk_count == 1
    assert report.first_token_latency_ms is not None
    assert report.total_latency_ms is not None
    assert report.profiler_report is not None


def test_runtime_rejects_completing_missing_stream() -> None:
    runtime = TokenStreamRuntime()

    with pytest.raises(ValueError):
        runtime.complete_stream("missing")


def test_runtime_cancels_stream() -> None:
    runtime = TokenStreamRuntime()
    state = runtime.create_stream(name="answer")

    runtime.start_stream(state.stream_id)
    result = runtime.cancel_stream(state.stream_id)

    assert result.success is True
    assert result.status == TokenStreamStatus.CANCELLED
    assert result.reason == TokenStreamReason.STREAM_CANCELLED


def test_runtime_rejects_emit_after_cancel() -> None:
    runtime = TokenStreamRuntime()
    state = runtime.create_stream(name="answer")

    runtime.start_stream(state.stream_id)
    runtime.cancel_stream(state.stream_id)
    result = runtime.emit_text(stream_id=state.stream_id, text="late")

    assert result.success is False
    assert result.reason == TokenStreamReason.STREAM_NOT_ACTIVE


def test_runtime_fails_stream() -> None:
    runtime = TokenStreamRuntime()
    state = runtime.create_stream(name="answer")

    runtime.start_stream(state.stream_id)
    result = runtime.fail_stream(state.stream_id, error="model failed")

    assert result.success is True
    assert result.status == TokenStreamStatus.FAILED
    assert result.reason == TokenStreamReason.STREAM_FAILED


def test_snapshot_tracks_status_counts() -> None:
    runtime = TokenStreamRuntime()
    one = runtime.create_stream(name="one")
    two = runtime.create_stream(name="two")

    runtime.start_stream(one.stream_id)
    runtime.emit_text(stream_id=one.stream_id, text="ok")
    runtime.complete_stream(one.stream_id)

    runtime.start_stream(two.stream_id)
    runtime.cancel_stream(two.stream_id)

    snapshot = runtime.snapshot()

    assert snapshot.stream_count == 2
    assert snapshot.completed_count == 1
    assert snapshot.cancelled_count == 1
    assert snapshot.chunk_count == 1


def test_reports_are_queryable() -> None:
    runtime = TokenStreamRuntime()
    state = runtime.create_stream(name="answer")

    runtime.start_stream(state.stream_id)
    runtime.emit_text(stream_id=state.stream_id, text="Hello")
    report = runtime.complete_stream(state.stream_id)

    assert runtime.latest_report() == report
    assert runtime.reports() == (report,)


def test_reset_clears_runtime_state() -> None:
    runtime = TokenStreamRuntime()
    state = runtime.create_stream(name="answer")

    runtime.start_stream(state.stream_id)
    runtime.emit_text(stream_id=state.stream_id, text="Hello")
    runtime.reset()

    snapshot = runtime.snapshot()

    assert snapshot.stream_count == 0
    assert snapshot.chunk_count == 0
    assert snapshot.last_reason == TokenStreamReason.RUNTIME_RESET


def test_manual_sentence_stable_chunk() -> None:
    runtime = TokenStreamRuntime()
    state = runtime.create_stream(name="answer")

    runtime.start_stream(state.stream_id)
    result = runtime.emit_text(
        stream_id=state.stream_id,
        text="Done.",
        kind=TokenStreamChunkKind.SENTENCE,
        stability=TokenStability.SENTENCE_STABLE,
    )

    assert result.success is True

    updated = runtime.state_for(state.stream_id)

    assert updated is not None
    assert updated.stable_sentence_count == 1


def test_enum_values_are_stable() -> None:
    assert TokenStreamStatus.ACTIVE.value == "active"
    assert TokenStreamChunkKind.TOKEN.value == "token"
    assert TokenStreamEventKind.FIRST_TOKEN_EMITTED.value == (
        "first_token_emitted"
    )
    assert TokenStability.SENTENCE_STABLE.value == "sentence_stable"
    assert TokenStreamReason.STREAM_COMPLETED.value == "stream_completed"