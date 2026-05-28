from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.latency import (
    NaturalnessReason,
    NaturalnessRequest,
    NaturalnessStatus,
    NaturalSpeechChunk,
    ProsodyHintKind,
    ResponseNaturalnessConfig,
    ResponseNaturalnessOptimizerRuntime,
    SpeechChunkKind,
)


def test_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        ResponseNaturalnessConfig(name=" ").validate()


def test_config_rejects_invalid_sentence_limit() -> None:
    with pytest.raises(ValueError):
        ResponseNaturalnessConfig(max_words_per_sentence=3).validate()


def test_config_rejects_invalid_pause_range() -> None:
    with pytest.raises(ValueError):
        ResponseNaturalnessConfig(min_pause_ms=300, max_pause_ms=100).validate()


def test_config_rejects_invalid_filler_range() -> None:
    with pytest.raises(ValueError):
        ResponseNaturalnessConfig(filler_min_ms=1200, filler_max_ms=800).validate()


def test_request_requires_text() -> None:
    with pytest.raises(ValidationError):
        NaturalnessRequest(raw_text=" ")


def test_chunk_word_count_must_match() -> None:
    with pytest.raises(ValidationError):
        NaturalSpeechChunk(
            kind=SpeechChunkKind.SENTENCE,
            text="hello world",
            ssml_text="hello world",
            word_count=1,
        )


def test_runtime_creates_session() -> None:
    runtime = ResponseNaturalnessOptimizerRuntime()
    state = runtime.create_session(
        request=NaturalnessRequest(raw_text="Hello there.")
    )

    assert state.status == NaturalnessStatus.CREATED
    assert runtime.snapshot().session_count == 1


def test_optimizer_splits_long_sentence_for_voice() -> None:
    runtime = ResponseNaturalnessOptimizerRuntime()
    text = (
        "This is a very long sentence that should be split into smaller "
        "spoken chunks because voice output must stay natural and clear."
    )
    state = runtime.create_session(request=NaturalnessRequest(raw_text=text))

    report = runtime.optimize(state.session_id)

    assert report.status == NaturalnessStatus.OPTIMIZED
    assert report.chunk_count > 1
    assert all(chunk.word_count <= 15 for chunk in report.chunks)


def test_optimizer_inserts_pause_at_semantic_boundary() -> None:
    runtime = ResponseNaturalnessOptimizerRuntime()
    state = runtime.create_session(
        request=NaturalnessRequest(
            raw_text="I found three issues — the first one is important."
        )
    )

    report = runtime.optimize(state.session_id)

    assert any(chunk.pause_after_ms > 0 for chunk in report.chunks)
    assert "<break" in report.optimized_ssml


def test_question_gets_rising_prosody() -> None:
    runtime = ResponseNaturalnessOptimizerRuntime()
    state = runtime.create_session(
        request=NaturalnessRequest(raw_text="Do you want me to continue?")
    )

    report = runtime.optimize(state.session_id)

    assert report.chunks[0].prosody_hint == ProsodyHintKind.QUESTION_RISE
    assert 'pitch="+8%"' in report.optimized_ssml


def test_list_item_gets_list_pacing() -> None:
    runtime = ResponseNaturalnessOptimizerRuntime()
    state = runtime.create_session(
        request=NaturalnessRequest(raw_text="1. First check the logs.")
    )

    report = runtime.optimize(state.session_id)

    assert report.chunks[0].kind == SpeechChunkKind.LIST_ITEM
    assert report.chunks[0].prosody_hint == ProsodyHintKind.LIST_PACING
    assert report.chunks[0].pause_after_ms > 0


def test_filler_selected_when_user_waiting() -> None:
    runtime = ResponseNaturalnessOptimizerRuntime()
    state = runtime.create_session(
        request=NaturalnessRequest(
            raw_text="I found the issue.",
            allow_filler=True,
            user_waiting=True,
        )
    )

    report = runtime.optimize(state.session_id)

    assert report.filler_count == 1
    assert report.fillers
    assert 800 <= report.fillers[0].duration_ms <= 1200


def test_filler_does_not_repeat_twice_in_a_row() -> None:
    runtime = ResponseNaturalnessOptimizerRuntime()

    first = runtime.create_session(
        request=NaturalnessRequest(raw_text="First.", user_waiting=True)
    )
    second = runtime.create_session(
        request=NaturalnessRequest(raw_text="Second.", user_waiting=True)
    )

    first_report = runtime.optimize(first.session_id)
    second_report = runtime.optimize(second.session_id)

    assert first_report.fillers
    assert second_report.fillers
    assert first_report.fillers[0].kind != second_report.fillers[0].kind


def test_ssml_escapes_reserved_characters() -> None:
    runtime = ResponseNaturalnessOptimizerRuntime()
    state = runtime.create_session(
        request=NaturalnessRequest(raw_text="Use A < B and B > C.")
    )

    report = runtime.optimize(state.session_id)

    assert "&lt;" in report.optimized_ssml
    assert "&gt;" in report.optimized_ssml


def test_report_is_queryable() -> None:
    runtime = ResponseNaturalnessOptimizerRuntime()
    state = runtime.create_session(
        request=NaturalnessRequest(raw_text="This sounds natural.")
    )

    report = runtime.optimize(state.session_id)

    assert runtime.latest_report() == report
    assert runtime.reports() == (report,)


def test_cancel_session() -> None:
    runtime = ResponseNaturalnessOptimizerRuntime()
    state = runtime.create_session(
        request=NaturalnessRequest(raw_text="Cancel this.")
    )

    result = runtime.cancel_session(state.session_id)

    assert result.success is True
    assert result.status == NaturalnessStatus.CANCELLED


def test_fail_session() -> None:
    runtime = ResponseNaturalnessOptimizerRuntime()
    state = runtime.create_session(
        request=NaturalnessRequest(raw_text="Fail this.")
    )

    result = runtime.fail_session(state.session_id, error="failed")

    assert result.success is True
    assert result.status == NaturalnessStatus.FAILED


def test_snapshot_tracks_counts() -> None:
    runtime = ResponseNaturalnessOptimizerRuntime()
    state = runtime.create_session(
        request=NaturalnessRequest(raw_text="This sounds natural.")
    )

    runtime.optimize(state.session_id)
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.optimized_count == 1
    assert snapshot.report_count == 1


def test_reset_clears_runtime_state() -> None:
    runtime = ResponseNaturalnessOptimizerRuntime()
    state = runtime.create_session(
        request=NaturalnessRequest(raw_text="Reset this.")
    )

    runtime.optimize(state.session_id)
    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.report_count == 0
    assert snapshot.last_reason == NaturalnessReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert NaturalnessStatus.OPTIMIZED.value == "optimized"
    assert SpeechChunkKind.SENTENCE.value == "sentence"
    assert ProsodyHintKind.QUESTION_RISE.value == "question_rise"