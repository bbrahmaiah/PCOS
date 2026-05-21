from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any
from uuid import uuid4

from jarvis.presence.adapters import VoiceActivityAdapter
from jarvis.presence.models import AudioFrame, VoiceActivity, VoiceActivityState
from jarvis.presence.state import PresenceStateStore
from jarvis.runtime.events import EventBus, RuntimeEvent
from jarvis.runtime.observability.structured_logger import get_logger
from jarvis.runtime.shared.enums import EventCategory, EventType
from jarvis.runtime.workers import BaseWorker


def new_speech_segment_id() -> str:
    return uuid4().hex


@dataclass(frozen=True, slots=True)
class VADWorkerSnapshot:
    """
    Immutable diagnostic snapshot for VADWorker.
    """

    name: str
    subscribed: bool
    processed_frames: int
    speech_started_count: int
    speech_ended_count: int
    speech_segments_completed: int
    silence_frames: int
    ignored_events: int
    vad_failures: int
    active_segment: bool
    active_segment_id: str | None
    active_segment_frame_count: int
    last_frame_id: str | None
    last_activity_id: str | None
    last_activity_state: str | None
    last_error: str | None


class VADWorker(BaseWorker):
    """
    Event-driven voice activity detection worker.

    Design:
    - consumes audio.frame_captured events
    - depends only on VoiceActivityAdapter interface
    - emits user speech boundary events
    - emits completed speech segments for future STTWorker
    - optionally updates PresenceStateStore
    - does not capture microphone audio
    - does not detect wake words
    - does not transcribe speech
    - does not perform cognition
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        vad_adapter: VoiceActivityAdapter,
        presence_store: PresenceStateStore | None = None,
        name: str = "vad_worker",
        tick_interval_seconds: float = 0.05,
        daemon: bool = True,
        auto_subscribe: bool = True,
    ) -> None:
        clean_name = name.strip()

        if not clean_name:
            raise ValueError("VADWorker name cannot be empty.")

        if tick_interval_seconds <= 0:
            raise ValueError("tick_interval_seconds must be greater than zero.")

        super().__init__(
            name=clean_name,
            event_bus=event_bus,
            tick_interval_seconds=tick_interval_seconds,
            daemon=daemon,
        )

        self._vad_adapter = vad_adapter
        self._presence_store = presence_store
        self._auto_subscribe = auto_subscribe

        self._lock = RLock()
        self._subscribed = False

        self._processed_frames = 0
        self._speech_started_count = 0
        self._speech_ended_count = 0
        self._speech_segments_completed = 0
        self._silence_frames = 0
        self._ignored_events = 0
        self._vad_failures = 0

        self._active_segment_id: str | None = None
        self._active_segment_frames: list[AudioFrame] = []

        self._last_frame_id: str | None = None
        self._last_activity_id: str | None = None
        self._last_activity_state: str | None = None
        self._last_error: str | None = None

        self._logger = get_logger("presence.vad_worker")

    @property
    def vad_adapter(self) -> VoiceActivityAdapter:
        return self._vad_adapter

    @property
    def presence_store(self) -> PresenceStateStore | None:
        return self._presence_store

    def on_start(self) -> None:
        """
        Subscribe to captured audio frame events when the worker starts.
        """

        if not self._auto_subscribe:
            return

        with self._lock:
            if self._subscribed:
                return

            self.event_bus.subscribe(
                event_type=EventType.AUDIO_FRAME_CAPTURED,
                subscriber_name=self.name,
                callback=self.handle_audio_frame_event,
            )
            self._subscribed = True

        self._logger.info(
            "vad_worker_subscribed",
            worker=self.name,
            event_type=EventType.AUDIO_FRAME_CAPTURED.value,
        )

    def on_stop(self) -> None:
        """
        Reset adapter and clear active segment during shutdown.
        """

        try:
            self._vad_adapter.reset()

            with self._lock:
                self._active_segment_id = None
                self._active_segment_frames.clear()

        except Exception as exc:
            self._record_error(exc)
            raise

        self._logger.info("vad_worker_stopped", worker=self.name)

    def run_once(self) -> None:
        """
        Event-driven worker loop placeholder.

        VAD work happens through handle_audio_frame_event().
        """

    def handle_audio_frame_event(self, event: RuntimeEvent) -> None:
        """
        Consume one audio.frame_captured event.
        """

        if event.event_type != EventType.AUDIO_FRAME_CAPTURED:
            self._record_ignored_event()
            return

        frame = self._extract_frame(event)

        if frame is None:
            self._record_ignored_event()
            return

        self.process_frame(frame)

    def process_frame(self, frame: AudioFrame) -> VoiceActivity:
        """
        Analyze one audio frame and emit speech-boundary events.
        """

        try:
            activity = self._vad_adapter.analyze(frame)

        except Exception as exc:
            self._record_vad_failure(frame, exc)
            raise

        with self._lock:
            self._processed_frames += 1
            self._last_frame_id = frame.frame_id
            self._last_activity_id = activity.activity_id
            self._last_activity_state = activity.state.value
            self._last_error = None

        self._handle_activity(frame, activity)

        return activity

    def vad_snapshot(self) -> VADWorkerSnapshot:
        """
        Return VADWorker-specific diagnostics.
        """

        with self._lock:
            return VADWorkerSnapshot(
                name=self.name,
                subscribed=self._subscribed,
                processed_frames=self._processed_frames,
                speech_started_count=self._speech_started_count,
                speech_ended_count=self._speech_ended_count,
                speech_segments_completed=self._speech_segments_completed,
                silence_frames=self._silence_frames,
                ignored_events=self._ignored_events,
                vad_failures=self._vad_failures,
                active_segment=self._active_segment_id is not None,
                active_segment_id=self._active_segment_id,
                active_segment_frame_count=len(self._active_segment_frames),
                last_frame_id=self._last_frame_id,
                last_activity_id=self._last_activity_id,
                last_activity_state=self._last_activity_state,
                last_error=self._last_error,
            )

    def _handle_activity(
        self,
        frame: AudioFrame,
        activity: VoiceActivity,
    ) -> None:
        if activity.state == VoiceActivityState.SILENCE:
            self._handle_silence()
            return

        if activity.state == VoiceActivityState.SPEECH_STARTED:
            self._handle_speech_started(frame, activity)
            return

        if activity.state == VoiceActivityState.SPEECH_CONTINUING:
            self._handle_speech_continuing(frame)
            return

        if activity.state == VoiceActivityState.SPEECH_ENDED:
            self._handle_speech_ended(frame, activity)
            return

    def _handle_silence(self) -> None:
        with self._lock:
            self._silence_frames += 1

    def _handle_speech_started(
        self,
        frame: AudioFrame,
        activity: VoiceActivity,
    ) -> None:
        with self._lock:
            self._active_segment_id = new_speech_segment_id()
            self._active_segment_frames = [frame]
            segment_id = self._active_segment_id
            self._speech_started_count += 1

        if self._presence_store is not None:
            self._presence_store.user_speech_started(
                metadata={
                    "source": self.name,
                    "frame_id": frame.frame_id,
                    "activity_id": activity.activity_id,
                    "confidence": activity.confidence,
                },
            )

        self._publish_user_started_speaking(frame, activity, segment_id)
        self._publish_speech_segment_started(frame, activity, segment_id)

        self._logger.info(
            "vad_speech_started",
            worker=self.name,
            frame_id=frame.frame_id,
            activity_id=activity.activity_id,
            segment_id=segment_id,
            confidence=activity.confidence,
        )

    def _handle_speech_continuing(self, frame: AudioFrame) -> None:
        with self._lock:
            if self._active_segment_id is None:
                self._active_segment_id = new_speech_segment_id()

            self._active_segment_frames.append(frame)

    def _handle_speech_ended(
        self,
        frame: AudioFrame,
        activity: VoiceActivity,
    ) -> None:
        with self._lock:
            if self._active_segment_id is None:
                self._active_segment_id = new_speech_segment_id()

            self._active_segment_frames.append(frame)
            segment_id = self._active_segment_id
            segment_frames = tuple(self._active_segment_frames)

            self._active_segment_id = None
            self._active_segment_frames = []
            self._speech_ended_count += 1
            self._speech_segments_completed += 1

        if self._presence_store is not None:
            self._presence_store.user_speech_ended(
                metadata={
                    "source": self.name,
                    "frame_id": frame.frame_id,
                    "activity_id": activity.activity_id,
                    "confidence": activity.confidence,
                    "segment_id": segment_id,
                    "frame_count": len(segment_frames),
                },
            )

        self._publish_user_stopped_speaking(frame, activity, segment_id)
        self._publish_speech_segment_completed(
            frame=frame,
            activity=activity,
            segment_id=segment_id,
            segment_frames=segment_frames,
        )

        self._logger.info(
            "vad_speech_ended",
            worker=self.name,
            frame_id=frame.frame_id,
            activity_id=activity.activity_id,
            segment_id=segment_id,
            frame_count=len(segment_frames),
            confidence=activity.confidence,
        )

    def _publish_user_started_speaking(
        self,
        frame: AudioFrame,
        activity: VoiceActivity,
        segment_id: str,
    ) -> None:
        event = RuntimeEvent(
            event_type=EventType.PRESENCE_USER_STARTED_SPEAKING,
            category=EventCategory.PRESENCE,
            source=self.name,
            payload=self._build_activity_payload(
                frame=frame,
                activity=activity,
                segment_id=segment_id,
            ),
        )
        self.event_bus.publish(event)

    def _publish_user_stopped_speaking(
        self,
        frame: AudioFrame,
        activity: VoiceActivity,
        segment_id: str,
    ) -> None:
        event = RuntimeEvent(
            event_type=EventType.PRESENCE_USER_STOPPED_SPEAKING,
            category=EventCategory.PRESENCE,
            source=self.name,
            payload=self._build_activity_payload(
                frame=frame,
                activity=activity,
                segment_id=segment_id,
            ),
        )
        self.event_bus.publish(event)

    def _publish_speech_segment_started(
        self,
        frame: AudioFrame,
        activity: VoiceActivity,
        segment_id: str,
    ) -> None:
        event = RuntimeEvent(
            event_type=EventType.AUDIO_SPEECH_SEGMENT_STARTED,
            category=EventCategory.PRESENCE,
            source=self.name,
            payload=self._build_activity_payload(
                frame=frame,
                activity=activity,
                segment_id=segment_id,
            ),
        )
        self.event_bus.publish(event)

    def _publish_speech_segment_completed(
        self,
        *,
        frame: AudioFrame,
        activity: VoiceActivity,
        segment_id: str,
        segment_frames: tuple[AudioFrame, ...],
    ) -> None:
        event = RuntimeEvent(
            event_type=EventType.AUDIO_SPEECH_SEGMENT_COMPLETED,
            category=EventCategory.PRESENCE,
            source=self.name,
            payload={
                **self._build_activity_payload(
                    frame=frame,
                    activity=activity,
                    segment_id=segment_id,
                ),
                "frames": segment_frames,
                "frame_ids": tuple(item.frame_id for item in segment_frames),
                "frame_count": len(segment_frames),
                "duration_ms": sum(item.duration_ms for item in segment_frames),
            },
        )
        self.event_bus.publish(event)

    @staticmethod
    def _build_activity_payload(
        *,
        frame: AudioFrame,
        activity: VoiceActivity,
        segment_id: str,
    ) -> dict[str, Any]:
        return {
            "segment_id": segment_id,
            "activity_id": activity.activity_id,
            "frame_id": frame.frame_id,
            "state": activity.state.value,
            "is_speech": activity.is_speech,
            "confidence": activity.confidence,
            "energy": activity.energy,
            "detected_at": activity.detected_at.isoformat(),
            "metadata": activity.metadata,
        }

    @staticmethod
    def _extract_frame(event: RuntimeEvent) -> AudioFrame | None:
        frame = event.payload.get("frame")

        if isinstance(frame, AudioFrame):
            return frame

        return None

    def _record_ignored_event(self) -> None:
        with self._lock:
            self._ignored_events += 1

    def _record_vad_failure(
        self,
        frame: AudioFrame,
        exc: Exception,
    ) -> None:
        error = f"{type(exc).__name__}: {exc}"

        with self._lock:
            self._processed_frames += 1
            self._vad_failures += 1
            self._last_frame_id = frame.frame_id
            self._last_error = error

        self._logger.error(
            "vad_worker_failure",
            worker=self.name,
            frame_id=frame.frame_id,
            error=error,
        )

    def _record_error(self, exc: Exception) -> None:
        error = f"{type(exc).__name__}: {exc}"

        with self._lock:
            self._last_error = error

        self._logger.error(
            "vad_worker_error",
            worker=self.name,
            error=error,
        )