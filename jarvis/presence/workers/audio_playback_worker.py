from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any

from jarvis.presence.adapters import (
    AudioPlaybackAdapter,
    PlaybackResult,
    PlaybackStatus,
)
from jarvis.presence.models import SpeechChunk
from jarvis.presence.state import PresenceStateStore
from jarvis.runtime.events import EventBus, RuntimeEvent
from jarvis.runtime.observability.structured_logger import get_logger
from jarvis.runtime.shared.enums import EventCategory, EventType
from jarvis.runtime.workers import BaseWorker


@dataclass(frozen=True, slots=True)
class AudioPlaybackWorkerSnapshot:
    """
    Immutable diagnostic snapshot for AudioPlaybackWorker.
    """

    name: str
    subscribed: bool
    adapter_playing: bool
    processed_chunks: int
    playback_started: int
    playback_completed: int
    playback_stopped: int
    playback_failed: int
    stop_requests: int
    ignored_events: int
    active_request_id: str | None
    active_chunk_id: str | None
    last_result_id: str | None
    last_status: str | None
    last_error: str | None


class AudioPlaybackWorker(BaseWorker):
    """
    Event-driven audio playback worker.

    Design:
    - consumes audio.speech_chunk_ready events
    - depends only on AudioPlaybackAdapter interface
    - publishes playback lifecycle events
    - optionally updates PresenceStateStore
    - does not synthesize speech
    - does not capture microphone audio
    - does not perform interruption detection
    - does not call cognition/dialogue

    Future flow:
        TTSWorker -> audio.speech_chunk_ready
        AudioPlaybackWorker -> audio.playback_started
        InterruptionWorker -> stop_playback()
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        playback_adapter: AudioPlaybackAdapter,
        presence_store: PresenceStateStore | None = None,
        name: str = "audio_playback_worker",
        tick_interval_seconds: float = 0.05,
        daemon: bool = True,
        auto_subscribe: bool = True,
    ) -> None:
        clean_name = name.strip()

        if not clean_name:
            raise ValueError("AudioPlaybackWorker name cannot be empty.")

        if tick_interval_seconds <= 0:
            raise ValueError("tick_interval_seconds must be greater than zero.")

        super().__init__(
            name=clean_name,
            event_bus=event_bus,
            tick_interval_seconds=tick_interval_seconds,
            daemon=daemon,
        )

        self._playback_adapter = playback_adapter
        self._presence_store = presence_store
        self._auto_subscribe = auto_subscribe

        self._lock = RLock()
        self._subscribed = False
        self._processed_chunks = 0
        self._playback_started = 0
        self._playback_completed = 0
        self._playback_stopped = 0
        self._playback_failed = 0
        self._stop_requests = 0
        self._ignored_events = 0
        self._active_request_id: str | None = None
        self._active_chunk_id: str | None = None
        self._last_result_id: str | None = None
        self._last_status: str | None = None
        self._last_error: str | None = None

        self._logger = get_logger("presence.audio_playback_worker")

    @property
    def playback_adapter(self) -> AudioPlaybackAdapter:
        return self._playback_adapter

    @property
    def presence_store(self) -> PresenceStateStore | None:
        return self._presence_store

    def on_start(self) -> None:
        """
        Subscribe to speech chunk events when the worker starts.
        """

        if not self._auto_subscribe:
            return

        with self._lock:
            if self._subscribed:
                return

            self.event_bus.subscribe(
                event_type=EventType.AUDIO_SPEECH_CHUNK_READY,
                subscriber_name=self.name,
                callback=self.handle_speech_chunk_event,
            )
            self._subscribed = True

        self._logger.info(
            "audio_playback_worker_subscribed",
            worker=self.name,
            event_type=EventType.AUDIO_SPEECH_CHUNK_READY.value,
        )

    def on_stop(self) -> None:
        """
        Stop active playback during shutdown.
        """

        try:
            self.stop_playback(reason="worker_stop")

        except Exception as exc:
            self._record_error(exc)
            raise

        self._logger.info("audio_playback_worker_stopped", worker=self.name)

    def run_once(self) -> None:
        """
        Event-driven worker loop placeholder.

        Playback work happens through handle_speech_chunk_event().
        """

    def handle_speech_chunk_event(self, event: RuntimeEvent) -> None:
        """
        Consume one audio.speech_chunk_ready event.
        """

        if event.event_type != EventType.AUDIO_SPEECH_CHUNK_READY:
            self._record_ignored_event()
            return

        chunk = self._extract_chunk(event)

        if chunk is None:
            self._record_ignored_event()
            return

        self.process_chunk(chunk=chunk, source_event=event)

    def process_chunk(
        self,
        *,
        chunk: SpeechChunk,
        source_event: RuntimeEvent | None = None,
    ) -> PlaybackResult:
        """
        Play one speech chunk.
        """

        self._mark_assistant_speaking_started(
            chunk=chunk,
            source_event=source_event,
        )

        try:
            result = self._playback_adapter.play(chunk)

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            result = self._build_failure_result(chunk=chunk, error=error)

        self._record_playback_result(chunk=chunk, result=result)
        self._publish_playback_result(
            chunk=chunk,
            result=result,
            source_event=source_event,
        )

        if result.status in {
            PlaybackStatus.COMPLETED,
            PlaybackStatus.STOPPED,
            PlaybackStatus.FAILED,
        }:
            self._mark_assistant_speaking_finished(
                request_id=chunk.request_id,
                source_event=source_event,
                reason=result.status.value,
            )

        self._logger.info(
            "audio_playback_result",
            worker=self.name,
            request_id=chunk.request_id,
            chunk_id=chunk.chunk_id,
            status=result.status.value,
            result_id=result.result_id,
        )

        return result

    def stop_playback(
        self,
        *,
        request_id: str | None = None,
        reason: str = "stop_requested",
        source_event: RuntimeEvent | None = None,
    ) -> PlaybackResult | None:
        """
        Stop active playback.

        InterruptionWorker will use this method later.
        """

        with self._lock:
            self._stop_requests += 1

        result = self._playback_adapter.stop(request_id=request_id)

        if result is None:
            return None

        chunk_id = result.chunk_id
        active_request_id = result.request_id

        with self._lock:
            self._playback_stopped += 1
            self._active_request_id = None
            self._active_chunk_id = None
            self._last_result_id = result.result_id
            self._last_status = result.status.value
            self._last_error = None

        self._publish_event(
            event_type=EventType.AUDIO_PLAYBACK_STOPPED,
            payload={
                "result": result,
                "result_id": result.result_id,
                "request_id": active_request_id,
                "chunk_id": chunk_id,
                "status": result.status.value,
                "reason": reason,
                "played_at": result.played_at.isoformat(),
                "metadata": result.metadata,
            },
            source_event=source_event,
        )

        self._mark_assistant_speaking_finished(
            request_id=active_request_id,
            source_event=source_event,
            reason=reason,
        )

        self._logger.info(
            "audio_playback_stopped",
            worker=self.name,
            request_id=active_request_id,
            chunk_id=chunk_id,
            reason=reason,
        )

        return result

    def playback_snapshot(self) -> AudioPlaybackWorkerSnapshot:
        """
        Return AudioPlaybackWorker-specific diagnostics.
        """

        with self._lock:
            return AudioPlaybackWorkerSnapshot(
                name=self.name,
                subscribed=self._subscribed,
                adapter_playing=self._playback_adapter.is_playing,
                processed_chunks=self._processed_chunks,
                playback_started=self._playback_started,
                playback_completed=self._playback_completed,
                playback_stopped=self._playback_stopped,
                playback_failed=self._playback_failed,
                stop_requests=self._stop_requests,
                ignored_events=self._ignored_events,
                active_request_id=self._active_request_id,
                active_chunk_id=self._active_chunk_id,
                last_result_id=self._last_result_id,
                last_status=self._last_status,
                last_error=self._last_error,
            )

    def _mark_assistant_speaking_started(
        self,
        *,
        chunk: SpeechChunk,
        source_event: RuntimeEvent | None,
    ) -> None:
        start_needed = False

        with self._lock:
            if self._active_request_id != chunk.request_id:
                self._active_request_id = chunk.request_id
                self._active_chunk_id = chunk.chunk_id
                start_needed = True
            else:
                self._active_chunk_id = chunk.chunk_id

        if not start_needed:
            return

        if self._presence_store is not None:
            self._presence_store.assistant_response_started(
                speech_request_id=chunk.request_id,
                metadata={
                    "source": self.name,
                    "request_id": chunk.request_id,
                    "chunk_id": chunk.chunk_id,
                },
            )

        self._publish_event(
            event_type=EventType.ASSISTANT_SPEAKING_STARTED,
            payload={
                "request_id": chunk.request_id,
                "chunk_id": chunk.chunk_id,
                "source": self.name,
            },
            source_event=source_event,
        )

    def _mark_assistant_speaking_finished(
        self,
        *,
        request_id: str,
        source_event: RuntimeEvent | None,
        reason: str,
    ) -> None:
        if self._presence_store is not None:
            self._presence_store.assistant_response_finished(
                metadata={
                    "source": self.name,
                    "request_id": request_id,
                    "reason": reason,
                },
            )

        self._publish_event(
            event_type=EventType.ASSISTANT_SPEAKING_STOPPED,
            payload={
                "request_id": request_id,
                "reason": reason,
                "source": self.name,
            },
            source_event=source_event,
        )

    def _record_playback_result(
        self,
        *,
        chunk: SpeechChunk,
        result: PlaybackResult,
    ) -> None:
        with self._lock:
            self._processed_chunks += 1
            self._last_result_id = result.result_id
            self._last_status = result.status.value
            self._last_error = result.error

            if result.status == PlaybackStatus.STARTED:
                self._playback_started += 1
                self._active_request_id = chunk.request_id
                self._active_chunk_id = chunk.chunk_id

            elif result.status == PlaybackStatus.COMPLETED:
                self._playback_completed += 1
                self._active_request_id = None
                self._active_chunk_id = None

            elif result.status == PlaybackStatus.STOPPED:
                self._playback_stopped += 1
                self._active_request_id = None
                self._active_chunk_id = None

            elif result.status == PlaybackStatus.FAILED:
                self._playback_failed += 1
                self._active_request_id = None
                self._active_chunk_id = None

    def _publish_playback_result(
        self,
        *,
        chunk: SpeechChunk,
        result: PlaybackResult,
        source_event: RuntimeEvent | None,
    ) -> None:
        event_type = self._event_type_for_status(result.status)

        self._publish_event(
            event_type=event_type,
            payload={
                "result": result,
                "result_id": result.result_id,
                "request_id": chunk.request_id,
                "chunk_id": chunk.chunk_id,
                "status": result.status.value,
                "played_at": result.played_at.isoformat(),
                "error": result.error,
                "metadata": result.metadata,
            },
            source_event=source_event,
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
            "audio_playback_worker_error",
            worker=self.name,
            error=error,
        )

    @staticmethod
    def _build_failure_result(
        *,
        chunk: SpeechChunk,
        error: str,
    ) -> PlaybackResult:
        return PlaybackResult(
            chunk_id=chunk.chunk_id,
            request_id=chunk.request_id,
            status=PlaybackStatus.FAILED,
            error=error,
        )

    @staticmethod
    def _event_type_for_status(status: PlaybackStatus) -> EventType:
        if status == PlaybackStatus.STARTED:
            return EventType.AUDIO_PLAYBACK_STARTED

        if status == PlaybackStatus.COMPLETED:
            return EventType.AUDIO_PLAYBACK_COMPLETED

        if status == PlaybackStatus.STOPPED:
            return EventType.AUDIO_PLAYBACK_STOPPED

        return EventType.AUDIO_PLAYBACK_FAILED

    @staticmethod
    def _extract_chunk(event: RuntimeEvent) -> SpeechChunk | None:
        chunk = event.payload.get("chunk")

        if isinstance(chunk, SpeechChunk):
            return chunk

        return None