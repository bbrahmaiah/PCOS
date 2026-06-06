from __future__ import annotations

from collections.abc import Iterator

import pytest
from pydantic import ValidationError

from jarvis.cognition import (
    CognitionAdapterCapability,
    CognitionAdapterResult,
    CognitionAdapterSnapshot,
    CognitionRequest,
    CognitionRunState,
    CognitionRuntimePolicy,
    CognitionStreamingState,
    CognitionToken,
    FakeCognitionAdapter,
    FakeCognitionConfig,
    SpeechChunkAssembler,
    SpeechChunkKind,
    StreamedSpeechChunk,
    StreamingTokenPipeline,
    StreamingTokenPipelineConfig,
)


class RaisingStreamingAdapter:
    @property
    def name(self) -> str:
        return "raising_streaming_adapter"

    @property
    def capabilities(self) -> tuple[CognitionAdapterCapability, ...]:
        return (CognitionAdapterCapability.STREAMING,)

    def generate(self, request: CognitionRequest) -> CognitionAdapterResult:
        raise NotImplementedError

    def stream(self, request: CognitionRequest) -> Iterator[CognitionToken]:
        yield CognitionToken(
            request_id=request.request_id,
            index=0,
            text="Hello ",
        )
        raise RuntimeError("stream exploded")

    def snapshot(self) -> CognitionAdapterSnapshot:
        return CognitionAdapterSnapshot(
            name=self.name,
            capabilities=self.capabilities,
        )


def make_request(
    *,
    request_id: str = "request-1",
    text: str = "hello jarvis",
    streaming_enabled: bool = True,
) -> CognitionRequest:
    return CognitionRequest(
        request_id=request_id,
        text=text,
        policy=CognitionRuntimePolicy(streaming_enabled=streaming_enabled),
    )


def make_token(
    *,
    request_id: str = "request-1",
    index: int = 0,
    text: str = "Hello.",
    final: bool = False,
) -> CognitionToken:
    return CognitionToken(
        request_id=request_id,
        index=index,
        text=text,
        final=final,
    )


def test_streaming_token_pipeline_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        StreamingTokenPipelineConfig(name=" ").validate()

    with pytest.raises(ValueError):
        StreamingTokenPipelineConfig(sentence_flush_chars=0).validate()

    with pytest.raises(ValueError):
        StreamingTokenPipelineConfig(partial_flush_chars=0).validate()

    with pytest.raises(ValueError):
        StreamingTokenPipelineConfig(fallback_empty_response_text=" ").validate()


def test_streamed_speech_chunk_rejects_invalid_values() -> None:
    with pytest.raises(ValidationError):
        StreamedSpeechChunk(
            request_id=" ",
            index=0,
            text="hello",
            start_token_index=0,
            end_token_index=0,
        )

    with pytest.raises(ValidationError):
        StreamedSpeechChunk(
            request_id="request-1",
            index=0,
            text=" ",
            start_token_index=0,
            end_token_index=0,
        )


def test_speech_chunk_assembler_emits_sentence_chunk() -> None:
    assembler = SpeechChunkAssembler(
        request_id="request-1",
        sentence_flush_chars=220,
        emit_partial_chunks=False,
        partial_flush_chars=120,
    )

    chunks = assembler.accept_token(make_token(text="Hello sir."))

    assert len(chunks) == 1
    assert chunks[0].text == "Hello sir."
    assert chunks[0].kind == SpeechChunkKind.SENTENCE
    assert chunks[0].final is False


def test_speech_chunk_assembler_emits_partial_chunk() -> None:
    assembler = SpeechChunkAssembler(
        request_id="request-1",
        sentence_flush_chars=220,
        emit_partial_chunks=True,
        partial_flush_chars=5,
    )

    chunks = assembler.accept_token(make_token(text="Hello there"))

    assert len(chunks) == 1
    assert chunks[0].kind == SpeechChunkKind.PARTIAL
    assert chunks[0].text == "Hello there"


def test_speech_chunk_assembler_flushes_final_chunk() -> None:
    assembler = SpeechChunkAssembler(
        request_id="request-1",
        sentence_flush_chars=220,
        emit_partial_chunks=False,
        partial_flush_chars=120,
    )

    assert assembler.accept_token(make_token(text="Hello ")) == ()

    chunks = assembler.flush(final=True, end_token_index=0)

    assert len(chunks) == 1
    assert chunks[0].text == "Hello"
    assert chunks[0].kind == SpeechChunkKind.FINAL
    assert chunks[0].final is True


def test_speech_chunk_assembler_rejects_wrong_request_id() -> None:
    assembler = SpeechChunkAssembler(
        request_id="request-1",
        sentence_flush_chars=220,
        emit_partial_chunks=False,
        partial_flush_chars=120,
    )

    with pytest.raises(ValueError):
        assembler.accept_token(make_token(request_id="wrong"))


def test_streaming_token_pipeline_completes_request() -> None:
    adapter = FakeCognitionAdapter(
        config=FakeCognitionConfig(
            default_response="Hello sir. I am listening.",
            streaming_chunk_size=10,
        )
    )
    pipeline = StreamingTokenPipeline(adapter=adapter)

    result = pipeline.stream_request(make_request())
    snapshot = pipeline.snapshot()
    state_snapshot = pipeline.state_store.snapshot()

    assert result.completed is True
    assert result.response is not None
    assert result.response.text == "derived_fake_test_response::hello jarvis"
    assert result.response.token_count == len(result.tokens)
    assert len(result.tokens) > 0
    assert len(result.speech_chunks) > 0
    assert result.speech_chunks[-1].final is True
    assert snapshot.streamed_request_count == 1
    assert snapshot.completed_count == 1
    assert state_snapshot.state == CognitionRunState.COMPLETED


def test_streaming_token_pipeline_creates_sentence_chunks() -> None:
    adapter = FakeCognitionAdapter(
        config=FakeCognitionConfig(
            default_response="One. Two. Three.",
            streaming_chunk_size=5,
        )
    )
    pipeline = StreamingTokenPipeline(adapter=adapter)

    result = pipeline.stream_request(make_request(text="unknown command"))

    assert result.completed is True
    assert len(result.speech_chunks) >= 2
    assert result.speech_chunks[0].text.endswith(".")


def test_streaming_token_pipeline_rejects_overlapping_request() -> None:
    adapter = FakeCognitionAdapter()
    pipeline = StreamingTokenPipeline(adapter=adapter)
    request = make_request()

    assert pipeline.state_store.start_request(request).accepted is True

    result = pipeline.stream_request(make_request(request_id="request-2"))

    assert result.rejected is True
    assert result.state == CognitionStreamingState.REJECTED
    assert result.reason == "another cognition request is already active"


def test_streaming_token_pipeline_handles_adapter_exception() -> None:
    pipeline = StreamingTokenPipeline(adapter=RaisingStreamingAdapter())

    result = pipeline.stream_request(make_request())
    snapshot = pipeline.snapshot()
    state_snapshot = pipeline.state_store.snapshot()

    assert result.failed is True
    assert result.failure is not None
    assert result.failure.message == "RuntimeError: stream exploded"
    assert len(result.tokens) == 1
    assert snapshot.failed_count == 1
    assert state_snapshot.state == CognitionRunState.FAILED


def test_streaming_token_pipeline_can_cancel_before_first_token() -> None:
    adapter = FakeCognitionAdapter()
    pipeline = StreamingTokenPipeline(adapter=adapter)
    request = make_request()

    assert pipeline.state_store.start_request(request).accepted is True
    assert pipeline.state_store.request_cancel(
        request_id=request.request_id,
        reason="user interrupted",
    ).accepted is True

    result = pipeline.stream_request(request)

    assert result.rejected is True


def test_streaming_token_pipeline_reset_clears_counters() -> None:
    pipeline = StreamingTokenPipeline(adapter=FakeCognitionAdapter())

    assert pipeline.stream_request(make_request()).completed is True

    pipeline.reset()
    snapshot = pipeline.snapshot()

    assert snapshot.streamed_request_count == 0
    assert snapshot.completed_count == 0
    assert snapshot.failed_count == 0
    assert snapshot.cancelled_count == 0
    assert snapshot.rejected_count == 0
    assert snapshot.token_count == 0
    assert snapshot.speech_chunk_count == 0
    assert snapshot.last_request_id is None


def test_streaming_state_enum_values_are_stable() -> None:
    assert CognitionStreamingState.COMPLETED.value == "completed"
    assert CognitionStreamingState.FAILED.value == "failed"
    assert CognitionStreamingState.CANCELLED.value == "cancelled"
    assert CognitionStreamingState.REJECTED.value == "rejected"
    assert SpeechChunkKind.PARTIAL.value == "partial"
    assert SpeechChunkKind.SENTENCE.value == "sentence"
    assert SpeechChunkKind.FINAL.value == "final"