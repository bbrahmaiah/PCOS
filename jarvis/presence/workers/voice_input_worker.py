from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any

from jarvis.presence.adapters import MicrophoneAdapter
from jarvis.presence.models import AudioFrame
from jarvis.runtime.events import EventBus, RuntimeEvent
from jarvis.runtime.observability.structured_logger import get_logger
from jarvis.runtime.shared.enums import EventCategory, EventType
from jarvis.runtime.workers import BaseWorker


@dataclass(frozen=True, slots=True)
class VoiceInputWorkerSnapshot:
    """
    Immutable diagnostic snapshot for VoiceInputWorker.
    """

    name: str
    microphone_running: bool
    auto_start_microphone: bool
    captured_frames: int
    empty_reads: int
    publish_failures: int
    last_frame_id: str | None
    last_error: str | None


class VoiceInputWorker(BaseWorker):
    """
    Presence worker responsible for microphone frame capture.

    Design:
    - depends only on MicrophoneAdapter interface
    - publishes audio.frame_captured events
    - does not perform wake detection
    - does not perform VAD
    - does not perform STT
    - does not mutate Presence state
    - safe for future real-time parallel runtime

    Future flow:
        VoiceInputWorker -> AUDIO_FRAME_CAPTURED event
        WakeDetectorWorker consumes AUDIO_FRAME_CAPTURED
        VADWorker consumes AUDIO_FRAME_CAPTURED
        InterruptionWorker consumes AUDIO_FRAME_CAPTURED
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        microphone: MicrophoneAdapter,
        name: str = "voice_input_worker",
        poll_interval_seconds: float = 0.01,
        daemon: bool = True,
        auto_start_microphone: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            event_bus=event_bus,
            tick_interval_seconds=poll_interval_seconds,
            daemon=daemon,
        )

        clean_name = name.strip()

        if not clean_name:
            raise ValueError("VoiceInputWorker name cannot be empty.")

        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be greater than zero.")

        self._microphone = microphone
        self._auto_start_microphone = auto_start_microphone

        self._lock = RLock()
        self._captured_frames = 0
        self._empty_reads = 0
        self._publish_failures = 0
        self._last_frame_id: str | None = None
        self._last_error: str | None = None

        self._logger = get_logger("presence.voice_input_worker")

    @property
    def microphone(self) -> MicrophoneAdapter:
        return self._microphone

    @property
    def auto_start_microphone(self) -> bool:
        return self._auto_start_microphone

    def on_start(self) -> None:
        """
        Start microphone capture when the worker starts.
        """

        if not self._auto_start_microphone:
            return

        try:
            if not self._microphone.is_running:
                self._microphone.start()

            self._logger.info(
                "voice_input_microphone_started",
                worker=self.name,
            )

        except Exception as exc:
            self._record_error(exc)
            raise

    def on_stop(self) -> None:
        """
        Stop microphone capture when the worker stops.
        """

        try:
            if self._microphone.is_running:
                self._microphone.stop()

            self._logger.info(
                "voice_input_microphone_stopped",
                worker=self.name,
            )

        except Exception as exc:
            self._record_error(exc)
            raise

    def run_once(self) -> None:
        """
        Capture and publish one audio frame if available.

        This method is intentionally small and deterministic so it can be
        called directly in tests and repeatedly by BaseWorker's run loop.
        """

        if self._auto_start_microphone and not self._microphone.is_running:
            self._microphone.start()

        frame = self._microphone.read_frame()

        if frame is None:
            with self._lock:
                self._empty_reads += 1
            return

        self._publish_audio_frame(frame)

        with self._lock:
            self._captured_frames += 1
            self._last_frame_id = frame.frame_id
            self._last_error = None

    def voice_snapshot(self) -> VoiceInputWorkerSnapshot:
        """
        Return VoiceInputWorker-specific diagnostics.

        Kept separate from BaseWorker.snapshot() to avoid overriding the base
        WorkerSnapshot contract.
        """

        with self._lock:
            return VoiceInputWorkerSnapshot(
                name=self.name,
                microphone_running=self._microphone.is_running,
                auto_start_microphone=self._auto_start_microphone,
                captured_frames=self._captured_frames,
                empty_reads=self._empty_reads,
                publish_failures=self._publish_failures,
                last_frame_id=self._last_frame_id,
                last_error=self._last_error,
            )

    def _publish_audio_frame(self, frame: AudioFrame) -> None:
        payload = self._build_audio_frame_payload(frame)

        event = RuntimeEvent(
            event_type=EventType.AUDIO_FRAME_CAPTURED,
            category=EventCategory.PRESENCE,
            source=self.name,
            payload=payload,
        )

        try:
            self.event_bus.publish(event)

            self._logger.info(
                "audio_frame_captured",
                worker=self.name,
                frame_id=frame.frame_id,
                frame_index=frame.frame_index,
                sample_rate=frame.sample_rate,
                channels=frame.channels,
                duration_ms=round(frame.duration_ms, 3),
            )

        except Exception as exc:
            with self._lock:
                self._publish_failures += 1

            self._record_error(exc)
            raise

    @staticmethod
    def _build_audio_frame_payload(frame: AudioFrame) -> dict[str, Any]:
        return {
            "frame_id": frame.frame_id,
            "source": frame.source,
            "sample_rate": frame.sample_rate,
            "channels": frame.channels,
            "sample_width_bytes": frame.sample_width_bytes,
            "frame_index": frame.frame_index,
            "captured_at": frame.captured_at.isoformat(),
            "byte_count": frame.byte_count,
            "sample_count": frame.sample_count,
            "duration_ms": frame.duration_ms,
            "metadata": frame.metadata,
        }

    def _record_error(self, exc: Exception) -> None:
        error = f"{type(exc).__name__}: {exc}"

        with self._lock:
            self._last_error = error

        self._logger.error(
            "voice_input_worker_error",
            worker=self.name,
            error=error,
        )