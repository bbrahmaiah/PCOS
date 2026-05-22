from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any

from jarvis.presence.adapters import TextToSpeechAdapter
from jarvis.presence.models import SpeechChunk, SpeechPriority, SpeechRequest
from jarvis.runtime.events import EventBus, RuntimeEvent
from jarvis.runtime.observability.structured_logger import get_logger
from jarvis.runtime.shared.enums import EventCategory, EventType
from jarvis.runtime.workers import BaseWorker


@dataclass(frozen=True, slots=True)
class TTSWorkerSnapshot:
    """
    Immutable diagnostic snapshot for TTSWorker.
    """

    name: str
    subscribed: bool
    processed_responses: int
    synthesis_started: int
    synthesis_completed: int
    synthesis_failures: int
    chunks_published: int
    ignored_events: int
    last_request_id: str | None
    last_response_id: str | None
    last_text: str | None
    last_chunk_id: str | None
    last_error: str | None


class TTSWorker(BaseWorker):
    """
    Event-driven text-to-speech worker.

    Design:
    - consumes dialogue.response_ready events
    - creates a SpeechRequest
    - depends only on TextToSpeechAdapter interface
    - emits audio.speech_chunk_ready events
    - emits tts synthesis lifecycle events
    - does not play audio
    - does not mutate PresenceStateStore
    - does not call cognition or dialogue directly

    Future flow:
        Dialogue/Cognition layer -> dialogue.response_ready
        TTSWorker -> audio.speech_chunk_ready
        AudioPlaybackWorker -> plays speech chunks
        InterruptionWorker -> can stop playback
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        tts_adapter: TextToSpeechAdapter,
        name: str = "tts_worker",
        tick_interval_seconds: float = 0.05,
        daemon: bool = True,
        auto_subscribe: bool = True,
        default_voice_id: str = "jarvis-default",
    ) -> None:
        clean_name = name.strip()
        clean_voice_id = default_voice_id.strip()

        if not clean_name:
            raise ValueError("TTSWorker name cannot be empty.")

        if tick_interval_seconds <= 0:
            raise ValueError("tick_interval_seconds must be greater than zero.")

        if not clean_voice_id:
            raise ValueError("default_voice_id cannot be empty.")

        super().__init__(
            name=clean_name,
            event_bus=event_bus,
            tick_interval_seconds=tick_interval_seconds,
            daemon=daemon,
        )

        self._tts_adapter = tts_adapter
        self._auto_subscribe = auto_subscribe
        self._default_voice_id = clean_voice_id

        self._lock = RLock()
        self._subscribed = False
        self._processed_responses = 0
        self._synthesis_started = 0
        self._synthesis_completed = 0
        self._synthesis_failures = 0
        self._chunks_published = 0
        self._ignored_events = 0
        self._last_request_id: str | None = None
        self._last_response_id: str | None = None
        self._last_text: str | None = None
        self._last_chunk_id: str | None = None
        self._last_error: str | None = None

        self._logger = get_logger("presence.tts_worker")

    @property
    def tts_adapter(self) -> TextToSpeechAdapter:
        return self._tts_adapter

    @property
    def default_voice_id(self) -> str:
        return self._default_voice_id

    def on_start(self) -> None:
        """
        Subscribe to dialogue.response_ready events when the worker starts.
        """

        if not self._auto_subscribe:
            return

        with self._lock:
            if self._subscribed:
                return

            self.event_bus.subscribe(
                event_type=EventType.ASSISTANT_RESPONSE_READY,
                subscriber_name=self.name,
                callback=self.handle_response_ready_event,
            )
            self._subscribed = True

        self._logger.info(
            "tts_worker_subscribed",
            worker=self.name,
            event_type=EventType.ASSISTANT_RESPONSE_READY.value,
        )

    def on_stop(self) -> None:
        """
        Reset TTS adapter state during shutdown.
        """

        try:
            self._tts_adapter.reset()

        except Exception as exc:
            self._record_error(exc)
            raise

        self._logger.info("tts_worker_stopped", worker=self.name)

    def run_once(self) -> None:
        """
        Event-driven worker loop placeholder.

        TTS work happens through handle_response_ready_event().
        """

    def handle_response_ready_event(self, event: RuntimeEvent) -> None:
        """
        Consume one dialogue.response_ready event.
        """

        if event.event_type != EventType.ASSISTANT_RESPONSE_READY:
            self._record_ignored_event()
            return

        text = self._extract_text(event)

        if text is None:
            self._record_ignored_event()
            return

        self.process_response(
            text=text,
            response_id=self._extract_optional_string(event, "response_id"),
            dialogue_request_id=self._extract_optional_string(
                event,
                "request_id",
            ),
            voice_id=self._extract_optional_string(event, "voice_id"),
            interruptible=self._extract_bool(
                event,
                "interruptible",
                default=True,
            ),
            source_event=event,
        )

    def process_response(
        self,
        *,
        text: str,
        response_id: str | None = None,
        dialogue_request_id: str | None = None,
        voice_id: str | None = None,
        interruptible: bool = True,
        source_event: RuntimeEvent | None = None,
    ) -> tuple[SpeechChunk, ...]:
        """
        Synthesize a ready assistant response into speech chunks.
        """

        clean_text = text.strip()

        if not clean_text:
            raise ValueError("text cannot be empty.")

        request = SpeechRequest(
            text=clean_text,
            voice_id=(voice_id or self._default_voice_id),
            priority=SpeechPriority.HIGH,
            interruptible=interruptible,
            correlation_id=source_event.correlation_id if source_event else None,
            metadata={
                "response_id": response_id,
                "dialogue_request_id": dialogue_request_id,
                "source": self.name,
            },
        )

        self._record_synthesis_started(
            request=request,
            response_id=response_id,
            text=clean_text,
        )
        self._publish_synthesis_started(
            request=request,
            response_id=response_id,
            dialogue_request_id=dialogue_request_id,
            source_event=source_event,
        )

        try:
            chunks = self._tts_adapter.synthesize(request)

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._record_synthesis_failure(error=error)
            self._publish_synthesis_failed(
                request=request,
                response_id=response_id,
                error=error,
                source_event=source_event,
            )
            return ()

        self._publish_chunks(
            request=request,
            response_id=response_id,
            dialogue_request_id=dialogue_request_id,
            chunks=chunks,
            source_event=source_event,
        )
        self._record_synthesis_completed(chunks=chunks)
        self._publish_synthesis_completed(
            request=request,
            response_id=response_id,
            dialogue_request_id=dialogue_request_id,
            chunks=chunks,
            source_event=source_event,
        )

        self._logger.info(
            "tts_synthesis_completed",
            worker=self.name,
            request_id=request.request_id,
            response_id=response_id,
            chunk_count=len(chunks),
        )

        return chunks

    def tts_snapshot(self) -> TTSWorkerSnapshot:
        """
        Return TTSWorker-specific diagnostics.
        """

        with self._lock:
            return TTSWorkerSnapshot(
                name=self.name,
                subscribed=self._subscribed,
                processed_responses=self._processed_responses,
                synthesis_started=self._synthesis_started,
                synthesis_completed=self._synthesis_completed,
                synthesis_failures=self._synthesis_failures,
                chunks_published=self._chunks_published,
                ignored_events=self._ignored_events,
                last_request_id=self._last_request_id,
                last_response_id=self._last_response_id,
                last_text=self._last_text,
                last_chunk_id=self._last_chunk_id,
                last_error=self._last_error,
            )

    def _record_synthesis_started(
        self,
        *,
        request: SpeechRequest,
        response_id: str | None,
        text: str,
    ) -> None:
        with self._lock:
            self._processed_responses += 1
            self._synthesis_started += 1
            self._last_request_id = request.request_id
            self._last_response_id = response_id
            self._last_text = text
            self._last_error = None

    def _record_synthesis_completed(
        self,
        *,
        chunks: tuple[SpeechChunk, ...],
    ) -> None:
        with self._lock:
            self._synthesis_completed += 1
            self._chunks_published += len(chunks)

            if chunks:
                self._last_chunk_id = chunks[-1].chunk_id

    def _record_synthesis_failure(self, *, error: str) -> None:
        with self._lock:
            self._synthesis_failures += 1
            self._last_error = error

    def _publish_chunks(
        self,
        *,
        request: SpeechRequest,
        response_id: str | None,
        dialogue_request_id: str | None,
        chunks: tuple[SpeechChunk, ...],
        source_event: RuntimeEvent | None,
    ) -> None:
        for chunk in chunks:
            payload = {
                "chunk": chunk,
                "chunk_id": chunk.chunk_id,
                "request_id": request.request_id,
                "response_id": response_id,
                "dialogue_request_id": dialogue_request_id,
                "chunk_index": chunk.chunk_index,
                "final": chunk.final,
                "byte_count": chunk.byte_count,
                "sample_rate": chunk.sample_rate,
                "created_at": chunk.created_at.isoformat(),
                "metadata": chunk.metadata,
            }
            self._publish_event(
                event_type=EventType.AUDIO_SPEECH_CHUNK_READY,
                payload=payload,
                source_event=source_event,
            )

    def _publish_synthesis_started(
        self,
        *,
        request: SpeechRequest,
        response_id: str | None,
        dialogue_request_id: str | None,
        source_event: RuntimeEvent | None,
    ) -> None:
        self._publish_event(
            event_type=EventType.TTS_SYNTHESIS_STARTED,
            payload={
                "request": request,
                "request_id": request.request_id,
                "response_id": response_id,
                "dialogue_request_id": dialogue_request_id,
                "text": request.text,
                "voice_id": request.voice_id,
                "interruptible": request.interruptible,
            },
            source_event=source_event,
        )

    def _publish_synthesis_completed(
        self,
        *,
        request: SpeechRequest,
        response_id: str | None,
        dialogue_request_id: str | None,
        chunks: tuple[SpeechChunk, ...],
        source_event: RuntimeEvent | None,
    ) -> None:
        self._publish_event(
            event_type=EventType.TTS_SYNTHESIS_COMPLETED,
            payload={
                "request_id": request.request_id,
                "response_id": response_id,
                "dialogue_request_id": dialogue_request_id,
                "chunk_ids": tuple(chunk.chunk_id for chunk in chunks),
                "chunk_count": len(chunks),
                "total_audio_bytes": sum(chunk.byte_count for chunk in chunks),
            },
            source_event=source_event,
        )

    def _publish_synthesis_failed(
        self,
        *,
        request: SpeechRequest,
        response_id: str | None,
        error: str,
        source_event: RuntimeEvent | None,
    ) -> None:
        self._publish_event(
            event_type=EventType.TTS_SYNTHESIS_FAILED,
            payload={
                "request_id": request.request_id,
                "response_id": response_id,
                "text": request.text,
                "error": error,
            },
            source_event=source_event,
        )

        self._logger.error(
            "tts_synthesis_failed",
            worker=self.name,
            request_id=request.request_id,
            response_id=response_id,
            error=error,
        )

    def _publish_event(
        self,
        *,
        event_type: EventType,
        payload: dict[str, Any],
        source_event: RuntimeEvent | None,
    ) -> None:
        if source_event is None:
            event = RuntimeEvent(
                event_type=event_type,
                category=EventCategory.PRESENCE,
                source=self.name,
                payload=payload,
            )
        else:
            event = RuntimeEvent(
                event_type=event_type,
                category=EventCategory.PRESENCE,
                source=self.name,
                correlation_id=source_event.correlation_id,
                payload=payload,
            )

        self.event_bus.publish(event)

    def _record_ignored_event(self) -> None:
        with self._lock:
            self._ignored_events += 1

    def _record_error(self, exc: Exception) -> None:
        error = f"{type(exc).__name__}: {exc}"

        with self._lock:
            self._last_error = error

        self._logger.error(
            "tts_worker_error",
            worker=self.name,
            error=error,
        )

    @staticmethod
    def _extract_text(event: RuntimeEvent) -> str | None:
        value = event.payload.get("text")

        if isinstance(value, str) and value.strip():
            return value

        return None

    @staticmethod
    def _extract_optional_string(
        event: RuntimeEvent,
        key: str,
    ) -> str | None:
        value = event.payload.get(key)

        if isinstance(value, str) and value.strip():
            return value

        return None

    @staticmethod
    def _extract_bool(
        event: RuntimeEvent,
        key: str,
        *,
        default: bool,
    ) -> bool:
        value = event.payload.get(key)

        if isinstance(value, bool):
            return value

        return default