from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.latency import (
    SpeechBoundaryKind,
    SpeechChunk,
    StreamingTTSReason,
    StreamingTTSRuntime,
    StreamingTTSRuntimeConfig,
    StreamingTTSStatus,
    TokenStability,
    token_chunk,
)


def test_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        StreamingTTSRuntimeConfig(name=" ").validate()


def test_config_rejects_invalid_minimum_words() -> None:
    with pytest.raises(ValueError):
        StreamingTTSRuntimeConfig(minimum_words_per_chunk=0).validate()


def test_config_rejects_invalid_max_wait() -> None:
    with pytest.raises(ValueError):
        StreamingTTSRuntimeConfig(maximum_wait_ms=0).validate()


def test_config_rejects_invalid_prebuffer_count() -> None:
    with pytest.raises(ValueError):
        StreamingTTSRuntimeConfig(prebuffer_sentence_count=0).validate()


def test_speech_chunk_requires_text() -> None:
    with pytest.raises(ValidationError):
        SpeechChunk(
            session_id="session",
            sequence=0,
            text=" ",
            boundary_kind=SpeechBoundaryKind.SENTENCE_BOUNDARY,
            word_count=0,
            source_token_count=0,
        )


def test_runtime_creates_session() -> None:
    runtime = StreamingTTSRuntime()
    state = runtime.create_session(name="answer")

    assert state.status == StreamingTTSStatus.CREATED
    assert runtime.snapshot().session_count == 1


def test_runtime_starts_session() -> None:
    runtime = StreamingTTSRuntime()
    state = runtime.create_session(name="answer")

    result = runtime.start_session(state.session_id)

    assert result.success is True
    assert result.status == StreamingTTSStatus.ACTIVE
    assert result.reason == StreamingTTSReason.SESSION_STARTED


def test_runtime_rejects_missing_session_start() -> None:
    runtime = StreamingTTSRuntime()

    result = runtime.start_session("missing")

    assert result.success is False
    assert result.reason == StreamingTTSReason.SESSION_NOT_FOUND


def test_runtime_buffers_tokens_without_boundary() -> None:
    runtime = StreamingTTSRuntime()
    state = runtime.create_session(name="answer")

    runtime.start_session(state.session_id)
    result = runtime.accept_token_chunk(
        session_id=state.session_id,
        chunk=token_chunk(text="The "),
    )

    assert result.success is True
    assert result.reason == StreamingTTSReason.TOKEN_BUFFERED
    assert len(runtime.audio_chunks_for(state.session_id)) == 0


def test_runtime_flushes_on_sentence_boundary() -> None:
    runtime = StreamingTTSRuntime()
    state = runtime.create_session(name="answer")

    runtime.start_session(state.session_id)

    for index, text in enumerate(("The ", "weather ", "today ", "is sunny.")):
        result = runtime.accept_token_chunk(
            session_id=state.session_id,
            chunk=token_chunk(sequence=index, text=text),
        )

    assert result.success is True
    assert result.reason == StreamingTTSReason.AUDIO_SYNTHESIZED
    assert result.speech_chunk is not None
    assert result.speech_chunk.boundary_kind == SpeechBoundaryKind.SENTENCE_BOUNDARY
    assert len(runtime.audio_chunks_for(state.session_id)) == 1


def test_runtime_respects_minimum_words_before_boundary() -> None:
    runtime = StreamingTTSRuntime(
        config=StreamingTTSRuntimeConfig(minimum_words_per_chunk=4)
    )
    state = runtime.create_session(name="answer")

    runtime.start_session(state.session_id)
    result = runtime.accept_token_chunk(
        session_id=state.session_id,
        chunk=token_chunk(text="Yes."),
    )

    assert result.success is True
    assert result.reason == StreamingTTSReason.TOKEN_BUFFERED
    assert len(runtime.audio_chunks_for(state.session_id)) == 0


def test_runtime_flushes_sentence_stable_token() -> None:
    runtime = StreamingTTSRuntime()
    state = runtime.create_session(name="answer")

    runtime.start_session(state.session_id)
    result = runtime.accept_token_chunk(
        session_id=state.session_id,
        chunk=token_chunk(
            text="This is already stable",
            stability=TokenStability.SENTENCE_STABLE,
        ),
    )

    assert result.success is True
    assert result.reason == StreamingTTSReason.AUDIO_SYNTHESIZED


def test_runtime_flushes_on_pause_marker() -> None:
    runtime = StreamingTTSRuntime()
    state = runtime.create_session(name="answer")

    runtime.start_session(state.session_id)
    result = runtime.accept_token_chunk(
        session_id=state.session_id,
        chunk=token_chunk(text="Let me think —"),
    )

    assert result.success is True
    assert result.speech_chunk is not None
    assert result.speech_chunk.boundary_kind == SpeechBoundaryKind.PAUSE_MARKER


def test_runtime_manual_flush() -> None:
    runtime = StreamingTTSRuntime()
    state = runtime.create_session(name="answer")

    runtime.start_session(state.session_id)
    runtime.accept_token_chunk(
        session_id=state.session_id,
        chunk=token_chunk(text="Manual flush works"),
    )
    result = runtime.flush(session_id=state.session_id)

    assert result.success is True
    assert result.audio_chunk is not None
    assert result.speech_chunk is not None
    assert result.speech_chunk.boundary_kind == SpeechBoundaryKind.MANUAL_FLUSH


def test_runtime_empty_flush_is_skipped() -> None:
    runtime = StreamingTTSRuntime()
    state = runtime.create_session(name="answer")

    runtime.start_session(state.session_id)
    result = runtime.flush(session_id=state.session_id)

    assert result.success is False
    assert result.reason == StreamingTTSReason.EMPTY_FLUSH_SKIPPED


def test_runtime_tracks_first_audio_latency() -> None:
    runtime = StreamingTTSRuntime()
    state = runtime.create_session(name="answer")

    runtime.start_session(state.session_id)
    runtime.accept_token_chunk(
        session_id=state.session_id,
        chunk=token_chunk(text="This sentence should synthesize now."),
    )

    updated = runtime.state_for(state.session_id)

    assert updated is not None
    assert updated.first_audio_latency_ms() is not None
    assert updated.audio_chunk_count == 1


def test_runtime_prebuffer_event_after_two_chunks() -> None:
    runtime = StreamingTTSRuntime(
        config=StreamingTTSRuntimeConfig(prebuffer_sentence_count=2)
    )
    state = runtime.create_session(name="answer")

    runtime.start_session(state.session_id)

    runtime.accept_token_chunk(
        session_id=state.session_id,
        chunk=token_chunk(text="The first sentence is ready."),
    )
    runtime.accept_token_chunk(
        session_id=state.session_id,
        chunk=token_chunk(sequence=1, text="The second sentence is ready."),
    )

    events = runtime.events_for(state.session_id)

    assert any(event.reason == StreamingTTSReason.PREBUFFER_READY for event in events)


def test_runtime_completes_with_final_flush() -> None:
    runtime = StreamingTTSRuntime()
    state = runtime.create_session(name="answer")

    runtime.start_session(state.session_id)
    runtime.accept_token_chunk(
        session_id=state.session_id,
        chunk=token_chunk(text="This has no final boundary"),
    )

    report = runtime.complete_session(state.session_id)

    assert report.status == StreamingTTSStatus.COMPLETED
    assert report.speech_chunk_count == 1
    assert report.audio_chunk_count == 1
    assert report.profiler_report is not None


def test_runtime_cancels_session() -> None:
    runtime = StreamingTTSRuntime()
    state = runtime.create_session(name="answer")

    runtime.start_session(state.session_id)
    result = runtime.cancel_session(state.session_id)

    assert result.success is True
    assert result.status == StreamingTTSStatus.CANCELLED


def test_runtime_rejects_token_after_cancel() -> None:
    runtime = StreamingTTSRuntime()
    state = runtime.create_session(name="answer")

    runtime.start_session(state.session_id)
    runtime.cancel_session(state.session_id)
    result = runtime.accept_token_chunk(
        session_id=state.session_id,
        chunk=token_chunk(text="late"),
    )

    assert result.success is False
    assert result.reason == StreamingTTSReason.SESSION_NOT_ACTIVE


def test_runtime_fails_session() -> None:
    runtime = StreamingTTSRuntime()
    state = runtime.create_session(name="answer")

    runtime.start_session(state.session_id)
    result = runtime.fail_session(state.session_id, error="tts failed")

    assert result.success is True
    assert result.status == StreamingTTSStatus.FAILED


def test_snapshot_tracks_counts() -> None:
    runtime = StreamingTTSRuntime()
    one = runtime.create_session(name="one")
    two = runtime.create_session(name="two")

    runtime.start_session(one.session_id)
    runtime.accept_token_chunk(
        session_id=one.session_id,
        chunk=token_chunk(text="This sentence should synthesize now."),
    )
    runtime.complete_session(one.session_id)

    runtime.start_session(two.session_id)
    runtime.cancel_session(two.session_id)

    snapshot = runtime.snapshot()

    assert snapshot.session_count == 2
    assert snapshot.completed_count == 1
    assert snapshot.cancelled_count == 1
    assert snapshot.audio_chunk_count == 1


def test_reports_are_queryable() -> None:
    runtime = StreamingTTSRuntime()
    state = runtime.create_session(name="answer")

    runtime.start_session(state.session_id)
    runtime.accept_token_chunk(
        session_id=state.session_id,
        chunk=token_chunk(text="This sentence should synthesize now."),
    )
    report = runtime.complete_session(state.session_id)

    assert runtime.latest_report() == report
    assert runtime.reports() == (report,)


def test_reset_clears_state() -> None:
    runtime = StreamingTTSRuntime()
    state = runtime.create_session(name="answer")

    runtime.start_session(state.session_id)
    runtime.accept_token_chunk(
        session_id=state.session_id,
        chunk=token_chunk(text="This sentence should synthesize now."),
    )
    runtime.reset()

    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.audio_chunk_count == 0
    assert snapshot.last_reason == StreamingTTSReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert StreamingTTSStatus.ACTIVE.value == "active"
    assert SpeechBoundaryKind.SENTENCE_BOUNDARY.value == "sentence_boundary"
    assert StreamingTTSReason.AUDIO_SYNTHESIZED.value == "audio_synthesized"