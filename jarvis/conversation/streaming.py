from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from threading import RLock

from pydantic import Field, field_validator

from jarvis.conversation.endpointing import EndpointAction, EndpointingDecision
from jarvis.conversation.models import (
    ConversationModel,
    TurnDecisionKind,
    new_conversation_id,
    utc_now,
)
from jarvis.runtime.observability.structured_logger import get_logger


class StreamingLifecycle(StrEnum):
    """
    Lifecycle state for one streaming conversation exchange.
    """

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    RESPONSE_STREAMING = "response_streaming"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"
    COMPLETED = "completed"


class StreamingEventKind(StrEnum):
    """
    Input event accepted by StreamingConversationCoordinator.
    """

    STT_PARTIAL = "stt_partial"
    USER_TURN_FINALIZED = "user_turn_finalized"
    COGNITION_STARTED = "cognition_started"
    COGNITION_TOKEN = "cognition_token"
    COGNITION_COMPLETED = "cognition_completed"
    SPEECH_CHUNK_STARTED = "speech_chunk_started"
    SPEECH_CHUNK_COMPLETED = "speech_chunk_completed"
    INTERRUPT_REQUESTED = "interrupt_requested"
    CANCEL_REQUESTED = "cancel_requested"
    RESET = "reset"


class StreamingCoordinatorAction(StrEnum):
    """
    Action recommended by the streaming coordinator.
    """

    KEEP_LISTENING = "keep_listening"
    START_COGNITION = "start_cognition"
    BUFFER_TOKEN = "buffer_token"
    EMIT_SPEECH_CHUNK = "emit_speech_chunk"
    START_TTS = "start_tts"
    COMPLETE_RESPONSE = "complete_response"
    CANCEL_STREAMS = "cancel_streams"
    RESET_SESSION = "reset_session"


class SpeechChunkKind(StrEnum):
    """
    Type of speech chunk emitted for TTS.
    """

    PARTIAL = "partial"
    FINAL = "final"


class StreamingConversationEvent(ConversationModel):
    """
    One event flowing through the streaming conversation coordinator.
    """

    event_id: str = Field(default_factory=new_conversation_id)
    turn_id: str
    kind: StreamingEventKind
    text: str = ""
    sequence: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("event_id", "turn_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("text")
    @classmethod
    def _clean_text(cls, value: str) -> str:
        return value.strip()


class StreamingSpeechChunk(ConversationModel):
    """
    One voice-ready chunk emitted while cognition is still generating.
    """

    chunk_id: str = Field(default_factory=new_conversation_id)
    turn_id: str
    text: str
    sequence: int = Field(ge=0)
    kind: SpeechChunkKind
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("chunk_id", "turn_id", "text")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @property
    def char_count(self) -> int:
        return len(self.text)


class StreamingCoordinatorOutput(ConversationModel):
    """
    Output of one coordinator step.
    """

    event: StreamingConversationEvent | None = None
    lifecycle: StreamingLifecycle
    actions: tuple[StreamingCoordinatorAction, ...]
    speech_chunk: StreamingSpeechChunk | None = None
    reason: str
    should_start_cognition: bool = False
    should_start_tts: bool = False
    should_cancel_streams: bool = False
    should_keep_listening: bool = True
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("reason")
    @classmethod
    def _reason_required(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("reason cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class StreamingConversationCoordinatorConfig:
    """
    Configuration for streaming conversation coordination.

    This controls early speech chunk emission. Small chunks reduce latency,
    but too-small chunks sound choppy. Sentence boundaries are preferred.
    """

    name: str = "streaming_conversation_coordinator"
    min_speech_chunk_chars: int = 48
    max_speech_chunk_chars: int = 220
    emit_on_sentence_boundary: bool = True
    sentence_endings: tuple[str, ...] = (".", "?", "!")
    comma_soft_boundary: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.min_speech_chunk_chars <= 0:
            raise ValueError("min_speech_chunk_chars must be greater than zero.")

        if self.max_speech_chunk_chars < self.min_speech_chunk_chars:
            raise ValueError(
                "max_speech_chunk_chars must be >= min_speech_chunk_chars."
            )

        if not self.sentence_endings:
            raise ValueError("sentence_endings cannot be empty.")

        if any(not ending for ending in self.sentence_endings):
            raise ValueError("sentence_endings cannot contain empty strings.")


@dataclass(frozen=True, slots=True)
class StreamingConversationCoordinatorSnapshot:
    """
    Observable diagnostics for StreamingConversationCoordinator.
    """

    name: str
    lifecycle: StreamingLifecycle
    event_count: int
    cognition_start_count: int
    token_count: int
    speech_chunk_count: int
    interrupt_count: int
    cancel_count: int
    completed_count: int
    buffered_chars: int
    current_turn_id: str | None
    last_event_kind: StreamingEventKind | None
    last_action: StreamingCoordinatorAction | None
    last_error: str | None


class StreamingConversationCoordinator:
    """
    Coordinates streaming STT, cognition tokens, speech chunks, and cancellation.

    Responsibilities:
    - accept endpointing decisions and stream events
    - start cognition when endpointing finalizes a turn
    - buffer cognition tokens
    - emit TTS-ready speech chunks early
    - keep listening while response is being generated/spoken
    - cancel streams immediately on interruption
    - expose diagnostics

    Non-responsibilities:
    - no microphone access
    - no STT implementation
    - no LLM implementation
    - no TTS implementation
    - no tool execution
    """

    def __init__(
        self,
        *,
        config: StreamingConversationCoordinatorConfig | None = None,
    ) -> None:
        self._config = config or StreamingConversationCoordinatorConfig()
        self._config.validate()

        self._lock = RLock()
        self._logger = get_logger("conversation.streaming_coordinator")

        self._lifecycle = StreamingLifecycle.IDLE
        self._current_turn_id: str | None = None
        self._buffer = ""
        self._speech_sequence = 0

        self._event_count = 0
        self._cognition_start_count = 0
        self._token_count = 0
        self._speech_chunk_count = 0
        self._interrupt_count = 0
        self._cancel_count = 0
        self._completed_count = 0
        self._last_event_kind: StreamingEventKind | None = None
        self._last_action: StreamingCoordinatorAction | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def lifecycle(self) -> StreamingLifecycle:
        with self._lock:
            return self._lifecycle

    def accept_endpointing_decision(
        self,
        decision: EndpointingDecision,
    ) -> StreamingCoordinatorOutput:
        """
        Consume endpointing output and produce streaming actions.

        This is the bridge from adaptive endpointing into streaming cognition.
        """

        turn_id = decision.turn_decision.turn_id

        with self._lock:
            self._current_turn_id = turn_id
            self._last_error = None

        if decision.action == EndpointAction.START_COGNITION:
            event = StreamingConversationEvent(
                turn_id=turn_id,
                kind=StreamingEventKind.USER_TURN_FINALIZED,
                text=decision.turn_decision.transcript,
            )
            return self.accept_event(event)

        if decision.action == EndpointAction.PREPARE_RESPONSE:
            return self._output(
                event=None,
                lifecycle=StreamingLifecycle.THINKING,
                actions=(StreamingCoordinatorAction.START_COGNITION,),
                reason="endpointing prepared response from stable partial turn",
                should_start_cognition=True,
            )

        if decision.action == EndpointAction.CANCEL_RESPONSE:
            event = StreamingConversationEvent(
                turn_id=turn_id,
                kind=StreamingEventKind.CANCEL_REQUESTED,
                text=decision.turn_decision.transcript,
            )
            return self.accept_event(event)

        if decision.action == EndpointAction.INTERRUPT_RESPONSE:
            event = StreamingConversationEvent(
                turn_id=turn_id,
                kind=StreamingEventKind.INTERRUPT_REQUESTED,
                text=decision.turn_decision.transcript,
            )
            return self.accept_event(event)

        lifecycle = (
            StreamingLifecycle.LISTENING
            if decision.turn_decision.decision == TurnDecisionKind.WAIT
            else self._lifecycle
        )

        return self._output(
            event=None,
            lifecycle=lifecycle,
            actions=(StreamingCoordinatorAction.KEEP_LISTENING,),
            reason="endpointing requested continued listening",
            should_keep_listening=True,
            metadata={
                "endpoint_action": decision.action.value,
            },
        )

    def accept_event(
        self,
        event: StreamingConversationEvent,
    ) -> StreamingCoordinatorOutput:
        """
        Accept one streaming event and return recommended actions.
        """

        with self._lock:
            self._event_count += 1
            self._last_event_kind = event.kind
            self._current_turn_id = event.turn_id
            self._last_error = None

        try:
            if event.kind == StreamingEventKind.STT_PARTIAL:
                output = self._handle_stt_partial(event)

            elif event.kind == StreamingEventKind.USER_TURN_FINALIZED:
                output = self._handle_user_turn_finalized(event)

            elif event.kind == StreamingEventKind.COGNITION_STARTED:
                output = self._handle_cognition_started(event)

            elif event.kind == StreamingEventKind.COGNITION_TOKEN:
                output = self._handle_cognition_token(event)

            elif event.kind == StreamingEventKind.COGNITION_COMPLETED:
                output = self._handle_cognition_completed(event)

            elif event.kind == StreamingEventKind.SPEECH_CHUNK_STARTED:
                output = self._handle_speech_chunk_started(event)

            elif event.kind == StreamingEventKind.SPEECH_CHUNK_COMPLETED:
                output = self._handle_speech_chunk_completed(event)

            elif event.kind == StreamingEventKind.INTERRUPT_REQUESTED:
                output = self._handle_interrupt(event)

            elif event.kind == StreamingEventKind.CANCEL_REQUESTED:
                output = self._handle_cancel(event)

            else:
                output = self._handle_reset(event)

            self._record_output(output)

            self._logger.info(
                "streaming_conversation_event_accepted",
                coordinator=self.name,
                turn_id=event.turn_id,
                event_kind=event.kind.value,
                lifecycle=output.lifecycle.value,
                actions=tuple(action.value for action in output.actions),
                speech_chunk_emitted=output.speech_chunk is not None,
            )

            return output

        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"

            raise

    def add_cognition_token(
        self,
        *,
        turn_id: str,
        token: str,
        sequence: int = 0,
    ) -> StreamingCoordinatorOutput:
        """
        Convenience method for cognition token streaming.
        """

        return self.accept_event(
            StreamingConversationEvent(
                turn_id=turn_id,
                kind=StreamingEventKind.COGNITION_TOKEN,
                text=token,
                sequence=sequence,
            )
        )

    def complete_cognition(
        self,
        *,
        turn_id: str,
    ) -> StreamingCoordinatorOutput:
        """
        Convenience method for cognition stream completion.
        """

        return self.accept_event(
            StreamingConversationEvent(
                turn_id=turn_id,
                kind=StreamingEventKind.COGNITION_COMPLETED,
            )
        )

    def snapshot(self) -> StreamingConversationCoordinatorSnapshot:
        """
        Return coordinator diagnostics.
        """

        with self._lock:
            return StreamingConversationCoordinatorSnapshot(
                name=self.name,
                lifecycle=self._lifecycle,
                event_count=self._event_count,
                cognition_start_count=self._cognition_start_count,
                token_count=self._token_count,
                speech_chunk_count=self._speech_chunk_count,
                interrupt_count=self._interrupt_count,
                cancel_count=self._cancel_count,
                completed_count=self._completed_count,
                buffered_chars=len(self._buffer),
                current_turn_id=self._current_turn_id,
                last_event_kind=self._last_event_kind,
                last_action=self._last_action,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Reset coordinator runtime state and diagnostics.
        """

        with self._lock:
            self._lifecycle = StreamingLifecycle.IDLE
            self._current_turn_id = None
            self._buffer = ""
            self._speech_sequence = 0
            self._event_count = 0
            self._cognition_start_count = 0
            self._token_count = 0
            self._speech_chunk_count = 0
            self._interrupt_count = 0
            self._cancel_count = 0
            self._completed_count = 0
            self._last_event_kind = None
            self._last_action = None
            self._last_error = None

        self._logger.info("streaming_conversation_coordinator_reset")

    def _handle_stt_partial(
        self,
        event: StreamingConversationEvent,
    ) -> StreamingCoordinatorOutput:
        with self._lock:
            self._lifecycle = StreamingLifecycle.LISTENING

        return self._output(
            event=event,
            lifecycle=StreamingLifecycle.LISTENING,
            actions=(StreamingCoordinatorAction.KEEP_LISTENING,),
            reason="received partial transcript; keep listening",
            should_keep_listening=True,
        )

    def _handle_user_turn_finalized(
        self,
        event: StreamingConversationEvent,
    ) -> StreamingCoordinatorOutput:
        with self._lock:
            self._lifecycle = StreamingLifecycle.THINKING
            self._buffer = ""
            self._speech_sequence = 0
            self._cognition_start_count += 1

        return self._output(
            event=event,
            lifecycle=StreamingLifecycle.THINKING,
            actions=(StreamingCoordinatorAction.START_COGNITION,),
            reason="user turn finalized; start streaming cognition",
            should_start_cognition=True,
            should_keep_listening=True,
        )

    def _handle_cognition_started(
        self,
        event: StreamingConversationEvent,
    ) -> StreamingCoordinatorOutput:
        with self._lock:
            self._lifecycle = StreamingLifecycle.THINKING

        return self._output(
            event=event,
            lifecycle=StreamingLifecycle.THINKING,
            actions=(StreamingCoordinatorAction.KEEP_LISTENING,),
            reason="cognition stream started",
            should_keep_listening=True,
        )

    def _handle_cognition_token(
        self,
        event: StreamingConversationEvent,
    ) -> StreamingCoordinatorOutput:
        token = event.text

        if not token:
            return self._output(
                event=event,
                lifecycle=self.lifecycle,
                actions=(StreamingCoordinatorAction.BUFFER_TOKEN,),
                reason="empty cognition token ignored",
                should_keep_listening=True,
            )

        with self._lock:
            self._token_count += 1
            self._buffer = self._join_token(self._buffer, token)
            self._lifecycle = StreamingLifecycle.RESPONSE_STREAMING
            buffer = self._buffer

        if self._should_emit_chunk(buffer):
            chunk = self._emit_chunk(
                turn_id=event.turn_id,
                text=buffer,
                kind=SpeechChunkKind.PARTIAL,
            )

            return self._output(
                event=event,
                lifecycle=StreamingLifecycle.RESPONSE_STREAMING,
                actions=(
                    StreamingCoordinatorAction.EMIT_SPEECH_CHUNK,
                    StreamingCoordinatorAction.START_TTS,
                ),
                speech_chunk=chunk,
                reason="speech chunk emitted while cognition continues",
                should_start_tts=True,
                should_keep_listening=True,
            )

        return self._output(
            event=event,
            lifecycle=StreamingLifecycle.RESPONSE_STREAMING,
            actions=(StreamingCoordinatorAction.BUFFER_TOKEN,),
            reason="cognition token buffered",
            should_keep_listening=True,
            metadata={
                "buffered_chars": len(buffer),
            },
        )

    def _handle_cognition_completed(
        self,
        event: StreamingConversationEvent,
    ) -> StreamingCoordinatorOutput:
        with self._lock:
            buffer = self._buffer

        actions: tuple[StreamingCoordinatorAction, ...]
        speech_chunk: StreamingSpeechChunk | None

        if buffer.strip():
            chunk = self._emit_chunk(
                turn_id=event.turn_id,
                text=buffer,
                kind=SpeechChunkKind.FINAL,
            )
            actions = (
                StreamingCoordinatorAction.EMIT_SPEECH_CHUNK,
                StreamingCoordinatorAction.START_TTS,
                StreamingCoordinatorAction.COMPLETE_RESPONSE,
            )
            speech_chunk = chunk
            should_start_tts = True
        else:
            actions = (StreamingCoordinatorAction.COMPLETE_RESPONSE,)
            speech_chunk = None
            should_start_tts = False

        with self._lock:
            self._lifecycle = StreamingLifecycle.COMPLETED
            self._completed_count += 1

        return self._output(
            event=event,
            lifecycle=StreamingLifecycle.COMPLETED,
            actions=actions,
            speech_chunk=speech_chunk,
            reason="cognition stream completed",
            should_start_tts=should_start_tts,
            should_keep_listening=True,
        )

    def _handle_speech_chunk_started(
        self,
        event: StreamingConversationEvent,
    ) -> StreamingCoordinatorOutput:
        with self._lock:
            self._lifecycle = StreamingLifecycle.SPEAKING

        return self._output(
            event=event,
            lifecycle=StreamingLifecycle.SPEAKING,
            actions=(StreamingCoordinatorAction.KEEP_LISTENING,),
            reason="speech chunk playback started",
            should_keep_listening=True,
        )

    def _handle_speech_chunk_completed(
        self,
        event: StreamingConversationEvent,
    ) -> StreamingCoordinatorOutput:
        with self._lock:
            next_lifecycle = (
                StreamingLifecycle.COMPLETED
                if not self._buffer
                else StreamingLifecycle.RESPONSE_STREAMING
            )
            self._lifecycle = next_lifecycle

        return self._output(
            event=event,
            lifecycle=next_lifecycle,
            actions=(StreamingCoordinatorAction.KEEP_LISTENING,),
            reason="speech chunk playback completed",
            should_keep_listening=True,
        )

    def _handle_interrupt(
        self,
        event: StreamingConversationEvent,
    ) -> StreamingCoordinatorOutput:
        with self._lock:
            self._lifecycle = StreamingLifecycle.INTERRUPTED
            self._buffer = ""
            self._interrupt_count += 1

        return self._output(
            event=event,
            lifecycle=StreamingLifecycle.INTERRUPTED,
            actions=(StreamingCoordinatorAction.CANCEL_STREAMS,),
            reason="interruption requested; cancel active streams",
            should_cancel_streams=True,
            should_keep_listening=True,
        )

    def _handle_cancel(
        self,
        event: StreamingConversationEvent,
    ) -> StreamingCoordinatorOutput:
        with self._lock:
            self._lifecycle = StreamingLifecycle.CANCELLED
            self._buffer = ""
            self._cancel_count += 1

        return self._output(
            event=event,
            lifecycle=StreamingLifecycle.CANCELLED,
            actions=(StreamingCoordinatorAction.CANCEL_STREAMS,),
            reason="cancel requested; cancel active streams",
            should_cancel_streams=True,
            should_keep_listening=True,
        )

    def _handle_reset(
        self,
        event: StreamingConversationEvent,
    ) -> StreamingCoordinatorOutput:
        with self._lock:
            self._lifecycle = StreamingLifecycle.IDLE
            self._buffer = ""
            self._speech_sequence = 0

        return self._output(
            event=event,
            lifecycle=StreamingLifecycle.IDLE,
            actions=(StreamingCoordinatorAction.RESET_SESSION,),
            reason="streaming coordinator reset",
            should_keep_listening=True,
        )

    def _emit_chunk(
        self,
        *,
        turn_id: str,
        text: str,
        kind: SpeechChunkKind,
    ) -> StreamingSpeechChunk:
        cleaned = text.strip()

        with self._lock:
            sequence = self._speech_sequence
            self._speech_sequence += 1
            self._speech_chunk_count += 1
            self._buffer = ""

        return StreamingSpeechChunk(
            turn_id=turn_id,
            text=cleaned,
            sequence=sequence,
            kind=kind,
            metadata={
                "coordinator": self.name,
            },
        )

    def _output(
        self,
        *,
        event: StreamingConversationEvent | None,
        lifecycle: StreamingLifecycle,
        actions: tuple[StreamingCoordinatorAction, ...],
        reason: str,
        speech_chunk: StreamingSpeechChunk | None = None,
        should_start_cognition: bool = False,
        should_start_tts: bool = False,
        should_cancel_streams: bool = False,
        should_keep_listening: bool = True,
        metadata: dict[str, object] | None = None,
    ) -> StreamingCoordinatorOutput:
        return StreamingCoordinatorOutput(
            event=event,
            lifecycle=lifecycle,
            actions=actions,
            speech_chunk=speech_chunk,
            reason=reason,
            should_start_cognition=should_start_cognition,
            should_start_tts=should_start_tts,
            should_cancel_streams=should_cancel_streams,
            should_keep_listening=should_keep_listening,
            metadata=metadata or {},
        )

    def _record_output(self, output: StreamingCoordinatorOutput) -> None:
        with self._lock:
            if output.actions:
                self._last_action = output.actions[0]

    def _should_emit_chunk(self, text: str) -> bool:
        cleaned = text.strip()

        if len(cleaned) >= self._config.max_speech_chunk_chars:
            return True

        if len(cleaned) < self._config.min_speech_chunk_chars:
            return False

        if self._config.emit_on_sentence_boundary and cleaned.endswith(
            self._config.sentence_endings
        ):
            return True

        return self._config.comma_soft_boundary and cleaned.endswith(",")

    @staticmethod
    def _join_token(buffer: str, token: str) -> str:
        if not buffer:
            return token.strip()

        if not token:
            return buffer

        if token.startswith((" ", ".", ",", "?", "!", ";", ":")):
            return f"{buffer}{token}"

        return f"{buffer} {token}"