from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.latency import (
    AudioChunkKind,
    PartialIntentKind,
    PartialTranscriptStability,
    SpeculativeWorkHint,
    SpeculativeWorkKind,
    StreamingSTTReason,
    StreamingSTTRuntime,
    StreamingSTTRuntimeConfig,
    StreamingSTTStatus,
    audio_chunk_metadata,
)


def test_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        StreamingSTTRuntimeConfig(name=" ").validate()


def test_config_rejects_invalid_audio_window() -> None:
    with pytest.raises(ValueError):
        StreamingSTTRuntimeConfig(audio_window_min_ms=0).validate()

    with pytest.raises(ValueError):
        StreamingSTTRuntimeConfig(
            audio_window_min_ms=200,
            audio_window_max_ms=100,
        ).validate()


def test_config_rejects_invalid_confidence_thresholds() -> None:
    with pytest.raises(ValueError):
        StreamingSTTRuntimeConfig(speculative_confidence_threshold=2).validate()

    with pytest.raises(ValueError):
        StreamingSTTRuntimeConfig(
            speculative_discard_similarity_threshold=-1
        ).validate()


def test_speculative_hint_must_be_safe() -> None:
    with pytest.raises(ValidationError):
        SpeculativeWorkHint(
            session_id="session",
            intent_id="intent",
            kind=SpeculativeWorkKind.MEMORY_PREFETCH,
            reason="unsafe",
            confidence=0.9,
            discardable=False,
        )

    with pytest.raises(ValidationError):
        SpeculativeWorkHint(
            session_id="session",
            intent_id="intent",
            kind=SpeculativeWorkKind.MEMORY_PREFETCH,
            reason="unsafe",
            confidence=0.9,
            cancellable=False,
        )


def test_runtime_creates_session() -> None:
    runtime = StreamingSTTRuntime()
    state = runtime.create_session(name="voice")

    assert state.status == StreamingSTTStatus.CREATED
    assert runtime.snapshot().session_count == 1


def test_runtime_starts_session() -> None:
    runtime = StreamingSTTRuntime()
    state = runtime.create_session(name="voice")

    result = runtime.start_session(state.session_id)

    assert result.success is True
    assert result.status == StreamingSTTStatus.ACTIVE
    assert result.reason == StreamingSTTReason.SESSION_STARTED


def test_runtime_rejects_missing_session_start() -> None:
    runtime = StreamingSTTRuntime()

    result = runtime.start_session("missing")

    assert result.success is False
    assert result.reason == StreamingSTTReason.SESSION_NOT_FOUND


def test_runtime_accepts_audio_and_emits_partial() -> None:
    runtime = StreamingSTTRuntime()
    state = runtime.create_session(name="voice")

    runtime.start_session(state.session_id)
    result = runtime.accept_audio_chunk(
        session_id=state.session_id,
        duration_ms=120,
        metadata=audio_chunk_metadata(transcript="what is AI"),
    )

    assert result.success is True
    assert result.partial is not None
    assert result.partial.text == "what is AI"
    assert result.intent is not None
    assert result.intent.kind == PartialIntentKind.QUESTION


def test_runtime_records_first_partial_latency() -> None:
    runtime = StreamingSTTRuntime()
    state = runtime.create_session(name="voice")

    runtime.start_session(state.session_id)
    runtime.accept_audio_chunk(
        session_id=state.session_id,
        duration_ms=120,
        metadata=audio_chunk_metadata(transcript="what is AI"),
    )

    updated = runtime.state_for(state.session_id)

    assert updated is not None
    assert updated.first_partial_latency_ms() is not None


def test_high_confidence_partial_starts_speculative_work() -> None:
    runtime = StreamingSTTRuntime()
    state = runtime.create_session(name="voice")

    runtime.start_session(state.session_id)
    result = runtime.accept_audio_chunk(
        session_id=state.session_id,
        duration_ms=120,
        metadata=audio_chunk_metadata(transcript="debug this error"),
    )

    assert result.intent is not None
    assert result.intent.kind == PartialIntentKind.DEBUGGING
    assert result.hints
    assert all(hint.discardable and hint.cancellable for hint in result.hints)
    assert any(
        hint.kind == SpeculativeWorkKind.MEMORY_PREFETCH
        for hint in result.hints
    )
    assert any(
        hint.kind == SpeculativeWorkKind.CONTEXT_PREWARM
        for hint in result.hints
    )
    assert any(
        hint.kind == SpeculativeWorkKind.TOOL_PLANNER_HINT
        for hint in result.hints
    )


def test_low_confidence_partial_does_not_start_speculation() -> None:
    runtime = StreamingSTTRuntime()
    state = runtime.create_session(name="voice")

    runtime.start_session(state.session_id)
    result = runtime.accept_audio_chunk(
        session_id=state.session_id,
        duration_ms=120,
        metadata=audio_chunk_metadata(
            transcript="maybe something",
            confidence=0.3,
        ),
    )

    assert result.hints == ()


def test_multiple_audio_chunks_update_latest_partial() -> None:
    runtime = StreamingSTTRuntime()
    state = runtime.create_session(name="voice")

    runtime.start_session(state.session_id)
    runtime.accept_audio_chunk(
        session_id=state.session_id,
        duration_ms=120,
        metadata=audio_chunk_metadata(transcript="open"),
    )
    runtime.accept_audio_chunk(
        session_id=state.session_id,
        duration_ms=120,
        metadata=audio_chunk_metadata(transcript="open vscode"),
    )

    updated = runtime.state_for(state.session_id)

    assert updated is not None
    assert updated.audio_chunk_count == 2
    assert updated.partial_count == 2
    assert updated.latest_partial_text == "open vscode"


def test_finalize_confirms_speculative_work_when_transcript_matches() -> None:
    runtime = StreamingSTTRuntime()
    state = runtime.create_session(name="voice")

    runtime.start_session(state.session_id)
    runtime.accept_audio_chunk(
        session_id=state.session_id,
        duration_ms=120,
        metadata=audio_chunk_metadata(transcript="debug this error"),
    )
    report = runtime.finalize_transcript(
        session_id=state.session_id,
        final_text="debug this error",
    )

    assert report.status == StreamingSTTStatus.FINALIZED
    assert report.confirmed_hint_count > 0
    assert report.discarded_hint_count == 0
    assert report.metadata["diverged"] is False
    assert report.profiler_report is not None


def test_finalize_discards_speculative_work_when_transcript_diverges() -> None:
    runtime = StreamingSTTRuntime()
    state = runtime.create_session(name="voice")

    runtime.start_session(state.session_id)
    runtime.accept_audio_chunk(
        session_id=state.session_id,
        duration_ms=120,
        metadata=audio_chunk_metadata(transcript="debug this error"),
    )
    report = runtime.finalize_transcript(
        session_id=state.session_id,
        final_text="tell me a joke",
    )

    assert report.status == StreamingSTTStatus.FINALIZED
    assert report.discarded_hint_count > 0
    assert report.metadata["diverged"] is True


def test_finalize_uses_latest_partial_when_final_text_missing() -> None:
    runtime = StreamingSTTRuntime()
    state = runtime.create_session(name="voice")

    runtime.start_session(state.session_id)
    runtime.accept_audio_chunk(
        session_id=state.session_id,
        duration_ms=120,
        metadata=audio_chunk_metadata(transcript="what is AI"),
    )
    report = runtime.finalize_transcript(session_id=state.session_id)

    assert report.final_transcript == "what is AI"


def test_runtime_rejects_finalize_missing_session() -> None:
    runtime = StreamingSTTRuntime()

    with pytest.raises(ValueError):
        runtime.finalize_transcript(session_id="missing")


def test_runtime_rejects_audio_after_finalize() -> None:
    runtime = StreamingSTTRuntime()
    state = runtime.create_session(name="voice")

    runtime.start_session(state.session_id)
    runtime.accept_audio_chunk(
        session_id=state.session_id,
        duration_ms=120,
        metadata=audio_chunk_metadata(transcript="what is AI"),
    )
    runtime.finalize_transcript(session_id=state.session_id)
    result = runtime.accept_audio_chunk(
        session_id=state.session_id,
        duration_ms=120,
        metadata=audio_chunk_metadata(transcript="late"),
    )

    assert result.success is False
    assert result.reason == StreamingSTTReason.SESSION_NOT_ACTIVE


def test_cancel_session_cancels_speculative_work() -> None:
    runtime = StreamingSTTRuntime()
    state = runtime.create_session(name="voice")

    runtime.start_session(state.session_id)
    runtime.accept_audio_chunk(
        session_id=state.session_id,
        duration_ms=120,
        metadata=audio_chunk_metadata(transcript="research latest AI"),
    )
    result = runtime.cancel_session(state.session_id)

    assert result.success is True
    assert result.status == StreamingSTTStatus.CANCELLED
    assert all(
        hint.status.value == "cancelled"
        for hint in runtime.hints_for(state.session_id)
    )


def test_fail_session() -> None:
    runtime = StreamingSTTRuntime()
    state = runtime.create_session(name="voice")

    runtime.start_session(state.session_id)
    result = runtime.fail_session(state.session_id, error="stt failed")

    assert result.success is True
    assert result.status == StreamingSTTStatus.FAILED


def test_snapshot_tracks_counts() -> None:
    runtime = StreamingSTTRuntime()
    state = runtime.create_session(name="voice")

    runtime.start_session(state.session_id)
    runtime.accept_audio_chunk(
        session_id=state.session_id,
        duration_ms=120,
        metadata=audio_chunk_metadata(transcript="research latest AI"),
    )
    runtime.finalize_transcript(session_id=state.session_id)

    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.finalized_count == 1
    assert snapshot.audio_chunk_count == 1
    assert snapshot.partial_count == 1
    assert snapshot.intent_count == 1
    assert snapshot.report_count == 1


def test_reports_are_queryable() -> None:
    runtime = StreamingSTTRuntime()
    state = runtime.create_session(name="voice")

    runtime.start_session(state.session_id)
    runtime.accept_audio_chunk(
        session_id=state.session_id,
        duration_ms=120,
        metadata=audio_chunk_metadata(transcript="what is AI"),
    )
    report = runtime.finalize_transcript(session_id=state.session_id)

    assert runtime.latest_report() == report
    assert runtime.reports() == (report,)


def test_reset_clears_runtime_state() -> None:
    runtime = StreamingSTTRuntime()
    state = runtime.create_session(name="voice")

    runtime.start_session(state.session_id)
    runtime.accept_audio_chunk(
        session_id=state.session_id,
        duration_ms=120,
        metadata=audio_chunk_metadata(transcript="what is AI"),
    )
    runtime.reset()

    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.partial_count == 0
    assert snapshot.last_reason == StreamingSTTReason.RUNTIME_RESET


def test_end_of_turn_audio_forces_final_stability() -> None:
    runtime = StreamingSTTRuntime()
    state = runtime.create_session(name="voice")

    runtime.start_session(state.session_id)
    result = runtime.accept_audio_chunk(
        session_id=state.session_id,
        duration_ms=120,
        kind=AudioChunkKind.END_OF_TURN,
        metadata=audio_chunk_metadata(
            transcript="final words",
            stability=PartialTranscriptStability.UNSTABLE,
        ),
    )

    assert result.partial is not None
    assert result.partial.stability == PartialTranscriptStability.FINAL


def test_enum_values_are_stable() -> None:
    assert StreamingSTTStatus.ACTIVE.value == "active"
    assert AudioChunkKind.END_OF_TURN.value == "end_of_turn"
    assert PartialIntentKind.DEBUGGING.value == "debugging"
    assert SpeculativeWorkKind.MEMORY_PREFETCH.value == "memory_prefetch"
    assert StreamingSTTReason.FINAL_TRANSCRIPT_READY.value == (
        "final_transcript_ready"
    )