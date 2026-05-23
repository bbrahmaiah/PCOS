from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from typing import Any

from pydantic import Field, field_validator

from jarvis.cognition.adapters import StreamingCognitionAdapter
from jarvis.cognition.models import (
    CognitionFailure,
    CognitionFailureKind,
    CognitionModel,
    CognitionRequest,
    CognitionResponse,
    CognitionResponseKind,
    CognitionToken,
    new_id,
)
from jarvis.cognition.state_store import CognitionRunState, CognitionStateStore
from jarvis.runtime.observability.structured_logger import get_logger


class CognitionStreamingState(StrEnum):
    """
    High-level streaming pipeline result state.
    """

    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class SpeechChunkKind(StrEnum):
    """
    Speakable chunk kind emitted from token streaming.
    """

    PARTIAL = "partial"
    SENTENCE = "sentence"
    FINAL = "final"


class StreamedSpeechChunk(CognitionModel):
    """
    Speakable text chunk assembled from streamed cognition tokens.

    This is not audio. It is text prepared for future TTS chunking.
    """

    chunk_id: str = Field(default_factory=new_id)
    request_id: str
    index: int = Field(ge=0)
    text: str
    kind: SpeechChunkKind = SpeechChunkKind.PARTIAL
    final: bool = False
    start_token_index: int = Field(ge=0)
    end_token_index: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("request_id", "text")
    @classmethod
    def _required_text_fields(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field cannot be empty.")

        return value


@dataclass(frozen=True, slots=True)
class StreamingTokenPipelineConfig:
    """
    Configuration for streaming token assembly.

    sentence_flush_chars prevents very long unfinished sentences from delaying
    speech forever.
    """

    name: str = "streaming_token_pipeline"
    sentence_flush_chars: int = 220
    emit_partial_chunks: bool = False
    partial_flush_chars: int = 120
    fallback_empty_response_text: str = "I understand, sir."

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.sentence_flush_chars <= 0:
            raise ValueError("sentence_flush_chars must be greater than zero.")

        if self.partial_flush_chars <= 0:
            raise ValueError("partial_flush_chars must be greater than zero.")

        if not self.fallback_empty_response_text.strip():
            raise ValueError("fallback_empty_response_text cannot be empty.")


@dataclass(frozen=True, slots=True)
class StreamingTokenPipelineResult:
    """
    Result of one streaming cognition execution.
    """

    request_id: str
    state: CognitionStreamingState
    tokens: tuple[CognitionToken, ...] = ()
    speech_chunks: tuple[StreamedSpeechChunk, ...] = ()
    response: CognitionResponse | None = None
    failure: CognitionFailure | None = None
    reason: str | None = None

    @property
    def completed(self) -> bool:
        return self.state == CognitionStreamingState.COMPLETED

    @property
    def failed(self) -> bool:
        return self.state == CognitionStreamingState.FAILED

    @property
    def cancelled(self) -> bool:
        return self.state == CognitionStreamingState.CANCELLED

    @property
    def rejected(self) -> bool:
        return self.state == CognitionStreamingState.REJECTED


@dataclass(frozen=True, slots=True)
class StreamingTokenPipelineSnapshot:
    """
    Observable streaming pipeline diagnostics.
    """

    name: str
    streamed_request_count: int
    completed_count: int
    failed_count: int
    cancelled_count: int
    rejected_count: int
    token_count: int
    speech_chunk_count: int
    last_request_id: str | None
    last_error: str | None


class SpeechChunkAssembler:
    """
    Converts token text into speakable chunks.

    It is intentionally conservative:
    - sentence chunks are emitted at punctuation boundaries
    - optional partial chunks can be emitted by character threshold
    - final flush always emits remaining text
    """

    _sentence_endings = {".", "!", "?"}

    def __init__(
        self,
        *,
        request_id: str,
        sentence_flush_chars: int,
        emit_partial_chunks: bool,
        partial_flush_chars: int,
    ) -> None:
        if not request_id.strip():
            raise ValueError("request_id cannot be empty.")

        if sentence_flush_chars <= 0:
            raise ValueError("sentence_flush_chars must be greater than zero.")

        if partial_flush_chars <= 0:
            raise ValueError("partial_flush_chars must be greater than zero.")

        self._request_id = request_id
        self._sentence_flush_chars = sentence_flush_chars
        self._emit_partial_chunks = emit_partial_chunks
        self._partial_flush_chars = partial_flush_chars
        self._buffer = ""
        self._chunk_index = 0
        self._buffer_start_token_index: int | None = None

    def accept_token(
        self,
        token: CognitionToken,
    ) -> tuple[StreamedSpeechChunk, ...]:
        """
        Accept one token and emit zero or more speakable chunks.
        """

        if token.request_id != self._request_id:
            raise ValueError("token request_id does not match assembler request_id.")

        if self._buffer_start_token_index is None:
            self._buffer_start_token_index = token.index

        self._buffer += token.text

        if token.final:
            return self.flush(final=True, end_token_index=token.index)

        if self._should_flush_sentence():
            return (
                self._make_chunk(
                    text=self._consume_buffer(),
                    kind=SpeechChunkKind.SENTENCE,
                    final=False,
                    end_token_index=token.index,
                ),
            )

        if self._should_flush_partial():
            return (
                self._make_chunk(
                    text=self._consume_buffer(),
                    kind=SpeechChunkKind.PARTIAL,
                    final=False,
                    end_token_index=token.index,
                ),
            )

        return ()

    def flush(
        self,
        *,
        final: bool,
        end_token_index: int,
    ) -> tuple[StreamedSpeechChunk, ...]:
        """
        Flush any remaining buffered text.
        """

        if not self._buffer.strip():
            return ()

        return (
            self._make_chunk(
                text=self._consume_buffer(),
                kind=SpeechChunkKind.FINAL if final else SpeechChunkKind.SENTENCE,
                final=final,
                end_token_index=end_token_index,
            ),
        )

    def _should_flush_sentence(self) -> bool:
        text = self._buffer.strip()

        if not text:
            return False

        return text[-1] in self._sentence_endings or (
            len(text) >= self._sentence_flush_chars
        )

    def _should_flush_partial(self) -> bool:
        if not self._emit_partial_chunks:
            return False

        return len(self._buffer.strip()) >= self._partial_flush_chars

    def _consume_buffer(self) -> str:
        text = " ".join(self._buffer.split())
        self._buffer = ""

        return text

    def _make_chunk(
        self,
        *,
        text: str,
        kind: SpeechChunkKind,
        final: bool,
        end_token_index: int,
    ) -> StreamedSpeechChunk:
        start_token_index = (
            self._buffer_start_token_index
            if self._buffer_start_token_index is not None
            else end_token_index
        )

        chunk = StreamedSpeechChunk(
            request_id=self._request_id,
            index=self._chunk_index,
            text=text,
            kind=kind,
            final=final,
            start_token_index=start_token_index,
            end_token_index=end_token_index,
        )

        self._chunk_index += 1
        self._buffer_start_token_index = None

        return chunk


class StreamingTokenPipeline:
    """
    Streaming cognition pipeline.

    Responsibilities:
    - start a streaming cognition request
    - consume adapter tokens
    - record tokens in CognitionStateStore
    - assemble speakable text chunks
    - produce final CognitionResponse
    - respect cancellation state

    Non-responsibilities:
    - no microphone/STT logic
    - no TTS/playback execution
    - no direct LLM implementation
    - no tool execution
    """

    def __init__(
        self,
        *,
        adapter: StreamingCognitionAdapter,
        state_store: CognitionStateStore | None = None,
        config: StreamingTokenPipelineConfig | None = None,
    ) -> None:
        self._config = config or StreamingTokenPipelineConfig()
        self._config.validate()

        self._adapter = adapter
        self._state_store = state_store or CognitionStateStore()
        self._lock = RLock()
        self._logger = get_logger("cognition.streaming")

        self._streamed_request_count = 0
        self._completed_count = 0
        self._failed_count = 0
        self._cancelled_count = 0
        self._rejected_count = 0
        self._token_count = 0
        self._speech_chunk_count = 0
        self._last_request_id: str | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def state_store(self) -> CognitionStateStore:
        return self._state_store

    def stream_request(
        self,
        request: CognitionRequest,
    ) -> StreamingTokenPipelineResult:
        """
        Stream one cognition request through the adapter.
        """

        with self._lock:
            self._streamed_request_count += 1
            self._last_request_id = request.request_id
            self._last_error = None

        start_transition = self._state_store.start_request(request)

        if not start_transition.accepted:
            return self._reject(
                request=request,
                reason=start_transition.reason or "state store rejected request",
            )

        stream_transition = self._state_store.mark_streaming_started()

        if not stream_transition.accepted:
            failure = self._make_failure(
                request=request,
                message=(
                    stream_transition.reason
                    or "state store rejected streaming start"
                ),
            )
            return self._fail(
                request=request,
                failure=failure,
                tokens=(),
                speech_chunks=(),
            )

        assembler = SpeechChunkAssembler(
            request_id=request.request_id,
            sentence_flush_chars=self._config.sentence_flush_chars,
            emit_partial_chunks=self._config.emit_partial_chunks,
            partial_flush_chars=self._config.partial_flush_chars,
        )

        tokens: list[CognitionToken] = []
        speech_chunks: list[StreamedSpeechChunk] = []

        try:
            for token in self._adapter.stream(request):
                if self._is_cancelling():
                    return self._cancel(
                        request=request,
                        tokens=tuple(tokens),
                        speech_chunks=tuple(speech_chunks),
                        reason="streaming cancelled",
                    )

                record_transition = self._state_store.record_token(token)

                if not record_transition.accepted:
                    failure = self._make_failure(
                        request=request,
                        message=(
                            record_transition.reason
                            or "state store rejected streamed token"
                        ),
                    )
                    return self._fail(
                        request=request,
                        failure=failure,
                        tokens=tuple(tokens),
                        speech_chunks=tuple(speech_chunks),
                    )

                tokens.append(token)

                emitted_chunks = assembler.accept_token(token)
                speech_chunks.extend(emitted_chunks)

                with self._lock:
                    self._token_count += 1
                    self._speech_chunk_count += len(emitted_chunks)

            if tokens:
                final_chunks = assembler.flush(
                    final=True,
                    end_token_index=tokens[-1].index,
                )
                speech_chunks.extend(final_chunks)

                with self._lock:
                    self._speech_chunk_count += len(final_chunks)

            final_text = self._final_text(tokens)
            response = CognitionResponse(
                request_id=request.request_id,
                text=final_text,
                kind=CognitionResponseKind.SPOKEN_REPLY,
                confidence=1.0 if tokens else 0.0,
                token_count=len(tokens),
                metadata={
                    "pipeline": self.name,
                    "streaming": True,
                    "speech_chunk_count": len(speech_chunks),
                },
            )

            complete_transition = self._state_store.complete_request(response)

            if not complete_transition.accepted:
                failure = self._make_failure(
                    request=request,
                    message=(
                        complete_transition.reason
                        or "state store rejected streaming completion"
                    ),
                )
                return self._fail(
                    request=request,
                    failure=failure,
                    tokens=tuple(tokens),
                    speech_chunks=tuple(speech_chunks),
                )

            return self._complete(
                request=request,
                response=response,
                tokens=tuple(tokens),
                speech_chunks=tuple(speech_chunks),
            )

        except Exception as exc:
            failure = self._make_failure(
                request=request,
                message=f"{type(exc).__name__}: {exc}",
                metadata={
                    "exception_type": type(exc).__name__,
                },
            )

            return self._fail(
                request=request,
                failure=failure,
                tokens=tuple(tokens),
                speech_chunks=tuple(speech_chunks),
            )

    def snapshot(self) -> StreamingTokenPipelineSnapshot:
        """
        Return streaming diagnostics.
        """

        with self._lock:
            return StreamingTokenPipelineSnapshot(
                name=self.name,
                streamed_request_count=self._streamed_request_count,
                completed_count=self._completed_count,
                failed_count=self._failed_count,
                cancelled_count=self._cancelled_count,
                rejected_count=self._rejected_count,
                token_count=self._token_count,
                speech_chunk_count=self._speech_chunk_count,
                last_request_id=self._last_request_id,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Reset pipeline counters and state store.
        """

        with self._lock:
            self._streamed_request_count = 0
            self._completed_count = 0
            self._failed_count = 0
            self._cancelled_count = 0
            self._rejected_count = 0
            self._token_count = 0
            self._speech_chunk_count = 0
            self._last_request_id = None
            self._last_error = None

        self._state_store.reset()

        self._logger.info("streaming_token_pipeline_reset", pipeline=self.name)

    def _complete(
        self,
        *,
        request: CognitionRequest,
        response: CognitionResponse,
        tokens: tuple[CognitionToken, ...],
        speech_chunks: tuple[StreamedSpeechChunk, ...],
    ) -> StreamingTokenPipelineResult:
        with self._lock:
            self._completed_count += 1
            self._last_error = None

        self._logger.info(
            "streaming_token_pipeline_completed",
            pipeline=self.name,
            request_id=request.request_id,
            token_count=len(tokens),
            speech_chunk_count=len(speech_chunks),
        )

        return StreamingTokenPipelineResult(
            request_id=request.request_id,
            state=CognitionStreamingState.COMPLETED,
            tokens=tokens,
            speech_chunks=speech_chunks,
            response=response,
        )

    def _fail(
        self,
        *,
        request: CognitionRequest,
        failure: CognitionFailure,
        tokens: tuple[CognitionToken, ...],
        speech_chunks: tuple[StreamedSpeechChunk, ...],
    ) -> StreamingTokenPipelineResult:
        self._state_store.fail_request(failure)

        with self._lock:
            self._failed_count += 1
            self._last_error = failure.message

        self._logger.error(
            "streaming_token_pipeline_failed",
            pipeline=self.name,
            request_id=request.request_id,
            failure_id=failure.failure_id,
            failure_message=failure.message,
        )

        return StreamingTokenPipelineResult(
            request_id=request.request_id,
            state=CognitionStreamingState.FAILED,
            tokens=tokens,
            speech_chunks=speech_chunks,
            failure=failure,
        )

    def _cancel(
        self,
        *,
        request: CognitionRequest,
        tokens: tuple[CognitionToken, ...],
        speech_chunks: tuple[StreamedSpeechChunk, ...],
        reason: str,
    ) -> StreamingTokenPipelineResult:
        self._state_store.cancel_request(
            request_id=request.request_id,
            reason=reason,
        )

        with self._lock:
            self._cancelled_count += 1
            self._last_error = reason

        self._logger.info(
            "streaming_token_pipeline_cancelled",
            pipeline=self.name,
            request_id=request.request_id,
            reason=reason,
        )

        return StreamingTokenPipelineResult(
            request_id=request.request_id,
            state=CognitionStreamingState.CANCELLED,
            tokens=tokens,
            speech_chunks=speech_chunks,
            reason=reason,
        )

    def _reject(
        self,
        *,
        request: CognitionRequest,
        reason: str,
    ) -> StreamingTokenPipelineResult:
        with self._lock:
            self._rejected_count += 1
            self._last_error = reason

        self._logger.info(
            "streaming_token_pipeline_rejected",
            pipeline=self.name,
            request_id=request.request_id,
            reason=reason,
        )

        return StreamingTokenPipelineResult(
            request_id=request.request_id,
            state=CognitionStreamingState.REJECTED,
            reason=reason,
        )

    def _is_cancelling(self) -> bool:
        snapshot = self._state_store.snapshot()

        return snapshot.state == CognitionRunState.CANCELLING

    def _final_text(
        self,
        tokens: list[CognitionToken],
    ) -> str:
        raw_text = "".join(token.text for token in tokens).strip()
        normalized_text = self._normalize_streamed_text(raw_text)

        return normalized_text or self._config.fallback_empty_response_text

    @staticmethod
    def _normalize_streamed_text(text: str) -> str:
        """
        Normalize streamed token text for final response reconstruction.

        Streaming backends may split text at arbitrary boundaries. This keeps
        final response text voice-safe without changing token ownership.
        """

        replacements = {
            ".I": ". I",
            "!I": "! I",
            "?I": "? I",
            ".You": ". You",
            "!You": "! You",
            "?You": "? You",
        }

        normalized = " ".join(text.split())

        for source, target in replacements.items():
            normalized = normalized.replace(source, target)

        return normalized

    def _make_failure(
        self,
        *,
        request: CognitionRequest,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> CognitionFailure:
        return CognitionFailure(
            request_id=request.request_id,
            kind=CognitionFailureKind.ADAPTER_ERROR,
            message=message,
            recoverable=True,
            metadata={
                "pipeline": self.name,
                "adapter": self._adapter.name,
                **(metadata or {}),
            },
        )