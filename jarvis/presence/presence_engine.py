from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from jarvis.presence.adapters import (
    AudioPlaybackAdapter,
    FakeAudioPlaybackAdapter,
    FakeMicrophoneAdapter,
    FakeSpeechToTextAdapter,
    FakeTextToSpeechAdapter,
    FakeVoiceActivityAdapter,
    FakeWakeWordAdapter,
    MicrophoneAdapter,
    SpeechToTextAdapter,
    TextToSpeechAdapter,
    VoiceActivityAdapter,
    WakeWordAdapter,
)
from jarvis.presence.state import PresenceStateStore
from jarvis.presence.workers import (
    AudioPlaybackWorker,
    DialogueBridgePolicy,
    DialogueBridgeWorker,
    InterruptionWorker,
    STTWorker,
    TTSWorker,
    VADWorker,
    VoiceInputWorker,
    WakeDetectorWorker,
)
from jarvis.runtime.events import EventBus, RuntimeEvent
from jarvis.runtime.observability.structured_logger import get_logger
from jarvis.runtime.shared.enums import EventCategory, EventType


@dataclass(frozen=True, slots=True)
class PresenceEngineAdapters:
    """
    Adapter bundle for PresenceEngine.

    The engine depends only on adapter interfaces. Fake adapters are the
    deterministic default for tests, simulations, and local development.
    Real adapters can be injected later without changing worker code.
    """

    microphone: MicrophoneAdapter
    wake_word: WakeWordAdapter
    vad: VoiceActivityAdapter
    stt: SpeechToTextAdapter
    tts: TextToSpeechAdapter
    playback: AudioPlaybackAdapter

    @classmethod
    def fake(cls) -> PresenceEngineAdapters:
        return cls(
            microphone=FakeMicrophoneAdapter(),
            wake_word=FakeWakeWordAdapter(default_detect=True),
            vad=FakeVoiceActivityAdapter(),
            stt=FakeSpeechToTextAdapter(),
            tts=FakeTextToSpeechAdapter(),
            playback=FakeAudioPlaybackAdapter(),
        )


@dataclass(frozen=True, slots=True)
class PresenceEngineWorkers:
    """
    Worker bundle owned by PresenceEngine.
    """

    voice_input: VoiceInputWorker
    wake_detector: WakeDetectorWorker
    vad: VADWorker
    stt: STTWorker
    dialogue_bridge: DialogueBridgeWorker
    tts: TTSWorker
    audio_playback: AudioPlaybackWorker
    interruption: InterruptionWorker

    def as_tuple(self) -> tuple[
        VoiceInputWorker,
        WakeDetectorWorker,
        VADWorker,
        STTWorker,
        DialogueBridgeWorker,
        TTSWorker,
        AudioPlaybackWorker,
        InterruptionWorker,
    ]:
        return (
            self.voice_input,
            self.wake_detector,
            self.vad,
            self.stt,
            self.dialogue_bridge,
            self.tts,
            self.audio_playback,
            self.interruption,
        )


@dataclass(frozen=True, slots=True)
class PresenceEngineSnapshot:
    """
    Immutable diagnostic snapshot for PresenceEngine.
    """

    name: str
    running: bool
    started: bool
    stopped: bool
    worker_count: int
    event_bus_name: str
    history_size: int
    current_mode: str
    current_turn_phase: str
    assistant_speaking: bool
    user_speaking: bool
    last_error: str | None


class PresenceEngine:
    """
    High-level Presence runtime orchestration.

    Design:
    - owns or receives one shared EventBus
    - owns or receives one shared PresenceStateStore
    - wires all Presence workers to the same bus/store
    - starts/stops worker subscriptions in deterministic order
    - starts the internal EventBus so async worker delivery actually runs
    - supports fake adapters now and real adapters later
    - does not duplicate worker responsibilities

    Runtime topology:
        VoiceInputWorker      -> audio.frame_captured
        WakeDetectorWorker    -> presence.wake_detected
        VADWorker             -> audio.speech_segment_completed
        STTWorker             -> presence.transcript_final
        DialogueBridgeWorker  -> dialogue.response_requested
        TTSWorker             -> audio.speech_chunk_ready
        AudioPlaybackWorker   -> audio.playback_started/stopped
        InterruptionWorker    -> presence.interrupt_requested
    """

    def __init__(
        self,
        *,
        name: str = "presence_engine",
        event_bus: EventBus | None = None,
        presence_store: PresenceStateStore | None = None,
        adapters: PresenceEngineAdapters | None = None,
        dialogue_policy: DialogueBridgePolicy | None = None,
        auto_start_microphone: bool = True,
    ) -> None:
        clean_name = name.strip()

        if not clean_name:
            raise ValueError("PresenceEngine name cannot be empty.")

        self._name = clean_name
        self._event_bus = event_bus or EventBus(name="presence_event_bus")
        self._presence_store = presence_store or PresenceStateStore(
            event_bus=self._event_bus,
        )
        self._adapters = adapters or PresenceEngineAdapters.fake()

        self._workers = self._build_workers(
            dialogue_policy=dialogue_policy,
            auto_start_microphone=auto_start_microphone,
        )

        self._lock = RLock()
        self._running = False
        self._started = False
        self._stopped = False
        self._last_error: str | None = None

        self._logger = get_logger("presence.engine")

    @property
    def name(self) -> str:
        return self._name

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus

    @property
    def presence_store(self) -> PresenceStateStore:
        return self._presence_store

    @property
    def adapters(self) -> PresenceEngineAdapters:
        return self._adapters

    @property
    def workers(self) -> PresenceEngineWorkers:
        return self._workers

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    def start(self) -> None:
        """
        Start the Presence runtime wiring.

        PresenceEngine owns an EventBus by default, so it must ensure the bus
        is running before async worker events are published. Without this,
        events enter history but async subscribers do not process them.
        """

        with self._lock:
            if self._running:
                return

        try:
            self._event_bus.start()

            for worker in self._workers.as_tuple():
                worker.on_start()

            with self._lock:
                self._running = True
                self._started = True
                self._stopped = False
                self._last_error = None

            self._publish_engine_event(EventType.PRESENCE_ENGINE_STARTED)

            self._logger.info(
                "presence_engine_started",
                engine=self._name,
                worker_count=len(self._workers.as_tuple()),
            )

        except Exception as exc:
            self._record_error(exc)
            raise

    def stop(self) -> None:
        """
        Stop the Presence runtime wiring in reverse order.
        """

        with self._lock:
            if not self._running and self._stopped:
                return

        try:
            for worker in reversed(self._workers.as_tuple()):
                worker.on_stop()

            with self._lock:
                self._running = False
                self._stopped = True

            self._publish_engine_event(EventType.PRESENCE_ENGINE_STOPPED)
            self._event_bus.stop()

            self._logger.info(
                "presence_engine_stopped",
                engine=self._name,
                worker_count=len(self._workers.as_tuple()),
            )

        except Exception as exc:
            self._record_error(exc)
            raise

    def snapshot(self) -> PresenceEngineSnapshot:
        state = self._presence_store.current_state()

        with self._lock:
            return PresenceEngineSnapshot(
                name=self._name,
                running=self._running,
                started=self._started,
                stopped=self._stopped,
                worker_count=len(self._workers.as_tuple()),
                event_bus_name=self._event_bus.name,
                history_size=len(self._event_bus.history()),
                current_mode=state.mode.value,
                current_turn_phase=state.turn_phase.value,
                assistant_speaking=state.assistant_speaking,
                user_speaking=state.user_speaking,
                last_error=self._last_error,
            )

    def publish_response_ready(
        self,
        *,
        text: str,
        response_id: str = "response-1",
        request_id: str = "dialogue-request-1",
        voice_id: str = "jarvis-default",
        interruptible: bool = True,
    ) -> None:
        """
        Helper for deterministic fake runtime tests.

        Future cognition/dialogue systems will publish dialogue.response_ready
        directly. This helper exists so PresenceEngine tests can validate the
        response -> TTS -> playback pipeline without a cognition layer.
        """

        clean_text = text.strip()

        if not clean_text:
            raise ValueError("response text cannot be empty.")

        event = RuntimeEvent(
            event_type=EventType.ASSISTANT_RESPONSE_READY,
            category=EventCategory.DIALOGUE,
            source=self._name,
            payload={
                "response_id": response_id,
                "request_id": request_id,
                "text": clean_text,
                "voice_id": voice_id,
                "interruptible": interruptible,
            },
        )

        self._event_bus.publish_sync(event)

    def _build_workers(
        self,
        *,
        dialogue_policy: DialogueBridgePolicy | None,
        auto_start_microphone: bool,
    ) -> PresenceEngineWorkers:
        return PresenceEngineWorkers(
            voice_input=VoiceInputWorker(
                event_bus=self._event_bus,
                microphone=self._adapters.microphone,
                auto_start_microphone=auto_start_microphone,
            ),
            wake_detector=WakeDetectorWorker(
                event_bus=self._event_bus,
                wake_word_adapter=self._adapters.wake_word,
                presence_store=self._presence_store,
            ),
            vad=VADWorker(
                event_bus=self._event_bus,
                vad_adapter=self._adapters.vad,
                presence_store=self._presence_store,
            ),
            stt=STTWorker(
                event_bus=self._event_bus,
                stt_adapter=self._adapters.stt,
                presence_store=self._presence_store,
            ),
            dialogue_bridge=DialogueBridgeWorker(
                event_bus=self._event_bus,
                policy=dialogue_policy,
            ),
            tts=TTSWorker(
                event_bus=self._event_bus,
                tts_adapter=self._adapters.tts,
            ),
            audio_playback=AudioPlaybackWorker(
                event_bus=self._event_bus,
                playback_adapter=self._adapters.playback,
                presence_store=self._presence_store,
            ),
            interruption=InterruptionWorker(
                event_bus=self._event_bus,
                presence_store=self._presence_store,
            ),
        )

    def _publish_engine_event(self, event_type: EventType) -> None:
        event = RuntimeEvent(
            event_type=event_type,
            category=EventCategory.PRESENCE,
            source=self._name,
            payload={
                "engine": self._name,
                "worker_count": len(self._workers.as_tuple()),
            },
        )

        self._event_bus.publish(event)

    def _record_error(self, exc: Exception) -> None:
        error = f"{type(exc).__name__}: {exc}"

        with self._lock:
            self._last_error = error
            self._running = False

        self._logger.error(
            "presence_engine_error",
            engine=self._name,
            error=error,
        )