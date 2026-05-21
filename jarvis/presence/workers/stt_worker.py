from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any

from jarvis.presence.adapters import SpeechToTextAdapter
from jarvis.presence.models import AudioFrame, Transcript, TranscriptKind
from jarvis.presence.state import PresenceStateStore
from jarvis.runtime.events import EventBus, RuntimeEvent
from jarvis.runtime.observability.structured_logger import get_logger
from jarvis.runtime.shared.enums import EventCategory, EventType
from jarvis.runtime.workers import BaseWorker


@dataclass(frozen=True, slots=True)
class STTWorkerSnapshot:
    """
    Immutable diagnostic snapshot for STTWorker.
    """

    name: str
    subscribed: bool
    processed_segments: int
    final_transcripts: int
    partial_transcripts: int
    rejected_transcripts: int
    ignored_events: int
    transcription_failures: int
    last_segment_id: str | None
    last_transcript_id: str | None
    last_transcript_kind: str | None
    last_error: str | None


class STTWorker(BaseWorker):
    """
    Event-driven speech-to-text worker.

    Design:
    - consumes audio.speech_segment_completed events
    - depends only on SpeechToTextAdapter interface
    - emits transcript events
    - optionally advances PresenceStateStore to waiting_for_response
    - does not capture microphone audio
    - does not detect wake words
    - does not perform VAD
    - does not perform cognition
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        stt_adapter: SpeechToTextAdapter,
        presence_store: PresenceStateStore | None = None,
        name: str = "stt_worker",
        tick_interval_seconds: float = 0.05,
        daemon: bool = True,
        auto_subscribe: bool = True,
    ) -> None:
        clean_name = name.strip()

        if not clean_name:
            raise ValueError("STTWorker name cannot be empty.")

        if tick_interval_seconds <= 0:
            raise ValueError("tick_interval_seconds must be greater than zero.")

        super().__init__(
            name=clean_name,
            event_bus=event_bus,
            tick_interval_seconds=tick_interval_seconds,
            daemon=daemon,
        )

        self._stt_adapter = stt_adapter
        self._presence_store = presence_store
        self._auto_subscribe = auto_subscribe

        self._lock = RLock()
        self._subscribed = False
        self._processed_segments = 0
        self._final_transcripts = 0
        self._partial_transcripts = 0
        self._rejected_transcripts = 0
        self._ignored_events = 0
        self._transcription_failures = 0
        self._last_segment_id: str | None = None
        self._last_transcript_id: str | None = None
        self._last_transcript_kind: str | None = None
        self._last_error: str | None = None

        self._logger = get_logger("presence.stt_worker")

    @property
    def stt_adapter(self) -> SpeechToTextAdapter:
        return self._stt_adapter

    @property
    def presence_store(self) -> PresenceStateStore | None:
        return self._presence_store

    def on_start(self) -> None:
        """
        Subscribe to completed speech segment events when the worker starts.
        """

        if not self._auto_subscribe:
            return

        with self._lock:
            if self._subscribed:
                return

            self.event_bus.subscribe(
                event_type=EventType.AUDIO_SPEECH_SEGMENT_COMPLETED,
                subscriber_name=self.name,
                callback=self.handle_speech_segment_event,
            )
            self._subscribed = True

        self._logger.info(
            "stt_worker_subscribed",
            worker=self.name,
            event_type=EventType.AUDIO_SPEECH_SEGMENT_COMPLETED.value,
        )

    def on_stop(self) -> None:
        """
        Reset STT adapter state during shutdown.
        """

        try:
            self._stt_adapter.reset()

        except Exception as exc:
            self._record_error(exc)
            raise

        self._logger.info("stt_worker_stopped", worker=self.name)

    def run_once(self) -> None:
        """
        Event-driven worker loop placeholder.

        STT work happens through handle_speech_segment_event().
        """

    def handle_speech_segment_event(self, event: RuntimeEvent) -> None:
        """
        Consume one audio.speech_segment_completed event.
        """

        if event.event_type != EventType.AUDIO_SPEECH_SEGMENT_COMPLETED:
            self._record_ignored_event()
            return

        segment_id = self._extract_segment_id(event)
        frames = self._extract_frames(event)

        if segment_id is None or not frames:
            self._record_ignored_event()
            return

        self.process_segment(
            segment_id=segment_id,
            frames=frames,
            source_event=event,
        )

    def process_segment(
        self,
        *,
        segment_id: str,
        frames: tuple[AudioFrame, ...],
        source_event: RuntimeEvent | None = None,
    ) -> Transcript | None:
        """
        Transcribe a completed speech segment.
        """

        clean_segment_id = segment_id.strip()

        if not clean_segment_id:
            raise ValueError("segment_id cannot be empty.")

        if not frames:
            raise ValueError("frames cannot be empty.")

        try:
            transcript = self._stt_adapter.transcribe(frames)

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

            self._record_transcription_failure(
                segment_id=clean_segment_id,
                error=error,
            )
            self._publish_transcript_rejected(
                segment_id=clean_segment_id,
                frames=frames,
                error=error,
                source_event=source_event,
            )
            return None

        self._record_transcript(
            segment_id=clean_segment_id,
            transcript=transcript,
        )

        self._publish_transcript(
            segment_id=clean_segment_id,
            frames=frames,
            transcript=transcript,
            source_event=source_event,
        )

        if (
            transcript.kind == TranscriptKind.FINAL
            and self._presence_store is not None
        ):
            self._presence_store.transcript_ready(
                metadata={
                    "source": self.name,
                    "segment_id": clean_segment_id,
                    "transcript_id": transcript.transcript_id,
                    "confidence": transcript.confidence,
                    "language": transcript.language,
                },
            )

        self._logger.info(
            "speech_segment_transcribed",
            worker=self.name,
            segment_id=clean_segment_id,
            transcript_id=transcript.transcript_id,
            kind=transcript.kind.value,
            confidence=transcript.confidence,
            language=transcript.language,
        )

        return transcript

    def stt_snapshot(self) -> STTWorkerSnapshot:
        """
        Return STTWorker-specific diagnostics.
        """

        with self._lock:
            return STTWorkerSnapshot(
                name=self.name,
                subscribed=self._subscribed,
                processed_segments=self._processed_segments,
                final_transcripts=self._final_transcripts,
                partial_transcripts=self._partial_transcripts,
                rejected_transcripts=self._rejected_transcripts,
                ignored_events=self._ignored_events,
                transcription_failures=self._transcription_failures,
                last_segment_id=self._last_segment_id,
                last_transcript_id=self._last_transcript_id,
                last_transcript_kind=self._last_transcript_kind,
                last_error=self._last_error,
            )

    def _record_transcript(
        self,
        *,
        segment_id: str,
        transcript: Transcript,
    ) -> None:
        with self._lock:
            self._processed_segments += 1
            self._last_segment_id = segment_id
            self._last_transcript_id = transcript.transcript_id
            self._last_transcript_kind = transcript.kind.value
            self._last_error = None

            if transcript.kind == TranscriptKind.FINAL:
                self._final_transcripts += 1
            elif transcript.kind == TranscriptKind.PARTIAL:
                self._partial_transcripts += 1
            else:
                self._rejected_transcripts += 1

    def _publish_transcript(
        self,
        *,
        segment_id: str,
        frames: tuple[AudioFrame, ...],
        transcript: Transcript,
        source_event: RuntimeEvent | None,
    ) -> None:
        event_type = self._event_type_for_transcript(transcript.kind)

        payload: dict[str, Any] = {
            "transcript": transcript,
            "transcript_id": transcript.transcript_id,
            "segment_id": segment_id,
            "text": transcript.text,
            "kind": transcript.kind.value,
            "confidence": transcript.confidence,
            "language": transcript.language,
            "alternatives": transcript.alternatives,
            "created_at": transcript.created_at.isoformat(),
            "frame_ids": tuple(frame.frame_id for frame in frames),
            "frame_count": len(frames),
            "duration_ms": sum(frame.duration_ms for frame in frames),
            "metadata": transcript.metadata,
        }

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

    def _publish_transcript_rejected(
        self,
        *,
        segment_id: str,
        frames: tuple[AudioFrame, ...],
        error: str,
        source_event: RuntimeEvent | None,
    ) -> None:
        with self._lock:
            self._processed_segments += 1
            self._rejected_transcripts += 1
            self._last_segment_id = segment_id
            self._last_transcript_id = None
            self._last_transcript_kind = TranscriptKind.REJECTED.value
            self._last_error = error

        payload: dict[str, Any] = {
            "segment_id": segment_id,
            "kind": TranscriptKind.REJECTED.value,
            "error": error,
            "frame_ids": tuple(frame.frame_id for frame in frames),
            "frame_count": len(frames),
            "duration_ms": sum(frame.duration_ms for frame in frames),
        }

        if source_event is None:
            event = RuntimeEvent(
                event_type=EventType.PRESENCE_TRANSCRIPT_REJECTED,
                category=EventCategory.PRESENCE,
                source=self.name,
                payload=payload,
            )
        else:
            event = RuntimeEvent(
                event_type=EventType.PRESENCE_TRANSCRIPT_REJECTED,
                category=EventCategory.PRESENCE,
                source=self.name,
                correlation_id=source_event.correlation_id,
                payload=payload,
            )

        self.event_bus.publish(event)

        self._logger.error(
            "speech_segment_rejected",
            worker=self.name,
            segment_id=segment_id,
            error=error,
        )

    def _record_transcription_failure(
        self,
        *,
        segment_id: str,
        error: str,
    ) -> None:
        with self._lock:
            self._transcription_failures += 1
            self._last_segment_id = segment_id
            self._last_error = error

        self._logger.error(
            "stt_worker_failure",
            worker=self.name,
            segment_id=segment_id,
            error=error,
        )

    def _record_ignored_event(self) -> None:
        with self._lock:
            self._ignored_events += 1

    def _record_error(self, exc: Exception) -> None:
        error = f"{type(exc).__name__}: {exc}"

        with self._lock:
            self._last_error = error

        self._logger.error(
            "stt_worker_error",
            worker=self.name,
            error=error,
        )

    @staticmethod
    def _event_type_for_transcript(kind: TranscriptKind) -> EventType:
        if kind == TranscriptKind.FINAL:
            return EventType.PRESENCE_TRANSCRIPT_FINAL

        if kind == TranscriptKind.PARTIAL:
            return EventType.PRESENCE_TRANSCRIPT_PARTIAL

        return EventType.PRESENCE_TRANSCRIPT_REJECTED

    @staticmethod
    def _extract_segment_id(event: RuntimeEvent) -> str | None:
        value = event.payload.get("segment_id")

        if isinstance(value, str) and value.strip():
            return value

        return None

    @staticmethod
    def _extract_frames(event: RuntimeEvent) -> tuple[AudioFrame, ...]:
        value = event.payload.get("frames")

        if not isinstance(value, tuple):
            return ()

        if not all(isinstance(frame, AudioFrame) for frame in value):
            return ()

        return value