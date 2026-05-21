from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from jarvis.presence.adapters import WakeWordAdapter, WakeWordDetection
from jarvis.presence.models import AudioFrame
from jarvis.presence.state import PresenceStateStore
from jarvis.runtime.events import EventBus, RuntimeEvent
from jarvis.runtime.observability.structured_logger import get_logger
from jarvis.runtime.shared.enums import EventCategory, EventType
from jarvis.runtime.workers import BaseWorker


@dataclass(frozen=True, slots=True)
class WakeDetectorWorkerSnapshot:
    """
    Immutable diagnostic snapshot for WakeDetectorWorker.
    """

    name: str
    subscribed: bool
    processed_frames: int
    wake_detections: int
    missed_detections: int
    ignored_events: int
    detection_failures: int
    last_frame_id: str | None
    last_detection_id: str | None
    last_wake_word: str | None
    last_error: str | None


class WakeDetectorWorker(BaseWorker):
    """
    Event-driven worker that detects wake words from captured audio frames.

    Design:
    - consumes audio.frame_captured events
    - depends only on WakeWordAdapter interface
    - emits presence.wake_detected events
    - optionally updates PresenceStateStore
    - does not capture microphone audio
    - does not perform VAD
    - does not perform STT
    - does not perform cognition
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        wake_word_adapter: WakeWordAdapter,
        presence_store: PresenceStateStore | None = None,
        name: str = "wake_detector_worker",
        tick_interval_seconds: float = 0.05,
        daemon: bool = True,
        auto_subscribe: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            event_bus=event_bus,
            tick_interval_seconds=tick_interval_seconds,
            daemon=daemon,
        )

        clean_name = name.strip()

        if not clean_name:
            raise ValueError("WakeDetectorWorker name cannot be empty.")

        if tick_interval_seconds <= 0:
            raise ValueError("tick_interval_seconds must be greater than zero.")

        self._wake_word_adapter = wake_word_adapter
        self._presence_store = presence_store
        self._auto_subscribe = auto_subscribe

        self._lock = RLock()
        self._subscribed = False
        self._processed_frames = 0
        self._wake_detections = 0
        self._missed_detections = 0
        self._ignored_events = 0
        self._detection_failures = 0
        self._last_frame_id: str | None = None
        self._last_detection_id: str | None = None
        self._last_wake_word: str | None = None
        self._last_error: str | None = None

        self._logger = get_logger("presence.wake_detector_worker")

    @property
    def wake_word_adapter(self) -> WakeWordAdapter:
        return self._wake_word_adapter

    @property
    def presence_store(self) -> PresenceStateStore | None:
        return self._presence_store

    def on_start(self) -> None:
        """
        Subscribe to audio frame events when the worker starts.
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
            "wake_detector_subscribed",
            worker=self.name,
            event_type=EventType.AUDIO_FRAME_CAPTURED.value,
        )

    def on_stop(self) -> None:
        """
        Reset adapter state during shutdown.

        The current EventBus API does not require explicit unsubscribe for
        this stage. Future unsubscribe support can be added without changing
        this worker's public contract.
        """

        try:
            self._wake_word_adapter.reset()

        except Exception as exc:
            self._record_error(exc)
            raise

        self._logger.info("wake_detector_stopped", worker=self.name)

    def run_once(self) -> None:
        """
        Event-driven worker loop placeholder.

        Wake detection happens through handle_audio_frame_event().
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

    def process_frame(self, frame: AudioFrame) -> WakeWordDetection | None:
        """
        Process one frame and publish wake event if detected.
        """

        try:
            detection = self._wake_word_adapter.detect(frame)

        except Exception as exc:
            self._record_detection_failure(frame, exc)
            raise

        with self._lock:
            self._processed_frames += 1
            self._last_frame_id = frame.frame_id
            self._last_error = None

            if detection is None:
                self._missed_detections += 1
                return None

            self._wake_detections += 1
            self._last_detection_id = detection.detection_id
            self._last_wake_word = detection.wake_word

        self._handle_detection(detection)

        return detection

    def wake_snapshot(self) -> WakeDetectorWorkerSnapshot:
        """
        Return WakeDetectorWorker-specific diagnostics.
        """

        with self._lock:
            return WakeDetectorWorkerSnapshot(
                name=self.name,
                subscribed=self._subscribed,
                processed_frames=self._processed_frames,
                wake_detections=self._wake_detections,
                missed_detections=self._missed_detections,
                ignored_events=self._ignored_events,
                detection_failures=self._detection_failures,
                last_frame_id=self._last_frame_id,
                last_detection_id=self._last_detection_id,
                last_wake_word=self._last_wake_word,
                last_error=self._last_error,
            )

    def _handle_detection(self, detection: WakeWordDetection) -> None:
        metadata = {
            "detection_id": detection.detection_id,
            "frame_id": detection.frame_id,
            "wake_word": detection.wake_word,
            "confidence": detection.confidence,
            "detected_at": detection.detected_at.isoformat(),
            "metadata": detection.metadata,
        }

        if self._presence_store is not None:
            self._presence_store.wake_detected(
                metadata={
                    "source": self.name,
                    "wake_word": detection.wake_word,
                    "confidence": detection.confidence,
                    "detection_id": detection.detection_id,
                },
            )

        event = RuntimeEvent(
            event_type=EventType.PRESENCE_WAKE_DETECTED,
            category=EventCategory.PRESENCE,
            source=self.name,
            payload=metadata,
        )

        self.event_bus.publish(event)

        self._logger.info(
            "wake_word_detected",
            worker=self.name,
            detection_id=detection.detection_id,
            frame_id=detection.frame_id,
            wake_word=detection.wake_word,
            confidence=detection.confidence,
        )

    @staticmethod
    def _extract_frame(event: RuntimeEvent) -> AudioFrame | None:
        frame = event.payload.get("frame")

        if isinstance(frame, AudioFrame):
            return frame

        return None

    def _record_ignored_event(self) -> None:
        with self._lock:
            self._ignored_events += 1

    def _record_detection_failure(
        self,
        frame: AudioFrame,
        exc: Exception,
    ) -> None:
        error = f"{type(exc).__name__}: {exc}"

        with self._lock:
            self._processed_frames += 1
            self._detection_failures += 1
            self._last_frame_id = frame.frame_id
            self._last_error = error

        self._logger.error(
            "wake_detector_failure",
            worker=self.name,
            frame_id=frame.frame_id,
            error=error,
        )

    def _record_error(self, exc: Exception) -> None:
        error = f"{type(exc).__name__}: {exc}"

        with self._lock:
            self._last_error = error

        self._logger.error(
            "wake_detector_error",
            worker=self.name,
            error=error,
        )