from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from jarvis.presence import PresenceEngine, PresenceEngineAdapters
from jarvis.presence.adapters import (
    FakeAudioPlaybackAdapter,
    FakeMicrophoneAdapter,
    FakeSpeechToTextAdapter,
    FakeTextToSpeechAdapter,
    FakeVoiceActivityAdapter,
    FakeWakeWordAdapter,
    make_fake_audio_frame,
)
from jarvis.presence.models import VoiceActivityState
from jarvis.runtime.observability.structured_logger import get_logger
from jarvis.runtime.shared.enums import EventType


@dataclass(frozen=True, slots=True)
class PresenceValidationCheck:
    """
    One validation check result.
    """

    name: str
    passed: bool
    details: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PresenceValidationReport:
    """
    Complete Presence runtime validation result.
    """

    passed: bool
    checks: tuple[PresenceValidationCheck, ...]
    started_at: datetime
    finished_at: datetime
    duration_ms: float

    @property
    def passed_count(self) -> int:
        return sum(1 for check in self.checks if check.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for check in self.checks if not check.passed)


class PresenceIntegrationValidator:
    """
    End-to-end validator for the Phase 2 Presence runtime.

    This validates the real runtime topology with deterministic fake adapters:

        VoiceInputWorker
        -> WakeDetectorWorker
        -> VADWorker
        -> STTWorker
        -> DialogueBridgeWorker
        -> TTSWorker
        -> AudioPlaybackWorker
        -> InterruptionWorker

    Validation rules:
    - keep normal events async
    - keep interruption sync
    - use fake adapters only
    - do not require hardware
    - do not call cognition/LLM
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = 2.0,
        poll_interval_seconds: float = 0.01,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")

        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be greater than zero.")

        self._timeout_seconds = timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._logger = get_logger("presence.validation")

    def run(self) -> PresenceValidationReport:
        started_at = datetime.now(UTC)
        started_perf = time.perf_counter()
        checks: list[PresenceValidationCheck] = []
        engine = self._build_engine()

        try:
            engine.start()

            checks.append(self._check_engine_started(engine))
            checks.append(self._check_workers_subscribed(engine))

            self._drive_voice_pipeline(engine)

            checks.append(self._check_voice_pipeline(engine))
            checks.append(self._check_dialogue_bridge(engine))

            self._prepare_presence_for_response(engine)
            engine.publish_response_ready(text="Yes sir. I am online.")

            self._wait_for(
                lambda: self._history_contains(
                    engine,
                    EventType.AUDIO_PLAYBACK_STARTED,
                )
            )

            checks.append(self._check_response_to_playback(engine))

            self._wait_for(
                lambda: engine.presence_store.current_state().assistant_speaking
                is True
            )

            engine.workers.interruption.process_user_started_speaking(
                segment_id="validation-interrupt-segment",
                frame_id="validation-interrupt-frame",
            )

            self._wait_for(
                lambda: self._history_contains(
                    engine,
                    EventType.AUDIO_PLAYBACK_STOPPED,
                )
            )

            checks.append(self._check_interruption(engine))
            checks.append(self._check_presence_state(engine))

        except Exception as exc:
            checks.append(
                PresenceValidationCheck(
                    name="presence_validation_exception",
                    passed=False,
                    details={
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
            )
            self._logger.error(
                "presence_validation_failed",
                error=f"{type(exc).__name__}: {exc}",
            )

        finally:
            try:
                engine.stop()
                checks.append(self._check_engine_stopped(engine))
            except Exception as exc:
                checks.append(
                    PresenceValidationCheck(
                        name="engine_stopped",
                        passed=False,
                        details={
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                )

        finished_at = datetime.now(UTC)
        duration_ms = (time.perf_counter() - started_perf) * 1000.0
        passed = all(check.passed for check in checks)

        report = PresenceValidationReport(
            passed=passed,
            checks=tuple(checks),
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
        )

        self._logger.info(
            "presence_integration_validation_completed",
            passed=report.passed,
            passed_count=report.passed_count,
            failed_count=report.failed_count,
            duration_ms=round(report.duration_ms, 3),
        )

        return report

    def _build_engine(self) -> PresenceEngine:
        frames = (
            make_fake_audio_frame(frame_index=0),
            make_fake_audio_frame(frame_index=1),
            make_fake_audio_frame(frame_index=2),
        )

        adapters = PresenceEngineAdapters(
            microphone=FakeMicrophoneAdapter(frames=frames),
            wake_word=FakeWakeWordAdapter(detection_pattern=(True, False, False)),
            vad=FakeVoiceActivityAdapter(
                states=(
                    VoiceActivityState.SPEECH_STARTED,
                    VoiceActivityState.SPEECH_CONTINUING,
                    VoiceActivityState.SPEECH_ENDED,
                ),
            ),
            stt=FakeSpeechToTextAdapter(
                text="hello jarvis",
                confidence=0.98,
            ),
            tts=FakeTextToSpeechAdapter(),
            playback=FakeAudioPlaybackAdapter(),
        )

        return PresenceEngine(
            name="presence_validation_engine",
            adapters=adapters,
        )

    def _drive_voice_pipeline(self, engine: PresenceEngine) -> None:
        engine.workers.voice_input.run_once()
        engine.workers.voice_input.run_once()
        engine.workers.voice_input.run_once()

        self._wait_for(
            lambda: self._history_contains(
                engine,
                EventType.PRESENCE_TRANSCRIPT_FINAL,
            )
        )

    def _prepare_presence_for_response(self, engine: PresenceEngine) -> None:
        state = engine.presence_store.current_state()

        if state.assistant_speaking:
            return

        if state.turn_phase.value == "waiting_for_response":
            return

        engine.presence_store.wake_detected(turn_id="validation-turn")
        engine.presence_store.user_speech_started()
        engine.presence_store.user_speech_ended()
        engine.presence_store.transcript_ready()

    def _check_engine_started(
        self,
        engine: PresenceEngine,
    ) -> PresenceValidationCheck:
        snapshot = engine.snapshot()

        return PresenceValidationCheck(
            name="engine_started",
            passed=snapshot.running and snapshot.started,
            details={
                "running": snapshot.running,
                "started": snapshot.started,
                "worker_count": snapshot.worker_count,
                "event_bus_name": snapshot.event_bus_name,
            },
        )

    def _check_workers_subscribed(
        self,
        engine: PresenceEngine,
    ) -> PresenceValidationCheck:
        subscribed = {
            "wake_detector": engine.workers.wake_detector.wake_snapshot().subscribed,
            "vad": engine.workers.vad.vad_snapshot().subscribed,
            "stt": engine.workers.stt.stt_snapshot().subscribed,
            "dialogue_bridge": (
                engine.workers.dialogue_bridge.dialogue_snapshot().subscribed
            ),
            "tts": engine.workers.tts.tts_snapshot().subscribed,
            "audio_playback": (
                engine.workers.audio_playback.playback_snapshot().subscribed
            ),
            "interruption": (
                engine.workers.interruption.interruption_snapshot().subscribed
            ),
        }

        return PresenceValidationCheck(
            name="workers_subscribed",
            passed=all(subscribed.values()),
            details=subscribed,
        )

    def _check_voice_pipeline(
        self,
        engine: PresenceEngine,
    ) -> PresenceValidationCheck:
        required = (
            EventType.AUDIO_FRAME_CAPTURED,
            EventType.PRESENCE_WAKE_DETECTED,
            EventType.PRESENCE_USER_STARTED_SPEAKING,
            EventType.AUDIO_SPEECH_SEGMENT_COMPLETED,
            EventType.PRESENCE_TRANSCRIPT_FINAL,
        )
        missing = self._missing_events(engine, required)

        return PresenceValidationCheck(
            name="voice_to_transcript_pipeline",
            passed=not missing,
            details={
                "required_events": tuple(event.value for event in required),
                "missing_events": tuple(event.value for event in missing),
                "history_size": len(engine.event_bus.history()),
            },
        )

    def _check_dialogue_bridge(
        self,
        engine: PresenceEngine,
    ) -> PresenceValidationCheck:
        found = self._history_contains(
            engine,
            EventType.ASSISTANT_RESPONSE_REQUESTED,
        )

        snapshot = engine.workers.dialogue_bridge.dialogue_snapshot()

        return PresenceValidationCheck(
            name="dialogue_bridge",
            passed=found and snapshot.response_requests >= 1,
            details={
                "event_found": found,
                "response_requests": snapshot.response_requests,
                "last_text": snapshot.last_text,
            },
        )

    def _check_response_to_playback(
        self,
        engine: PresenceEngine,
    ) -> PresenceValidationCheck:
        required = (
            EventType.ASSISTANT_RESPONSE_READY,
            EventType.TTS_SYNTHESIS_STARTED,
            EventType.AUDIO_SPEECH_CHUNK_READY,
            EventType.TTS_SYNTHESIS_COMPLETED,
            EventType.AUDIO_PLAYBACK_STARTED,
            EventType.ASSISTANT_SPEAKING_STARTED,
        )
        missing = self._missing_events(engine, required)
        playback_snapshot = engine.workers.audio_playback.playback_snapshot()

        return PresenceValidationCheck(
            name="response_to_playback_pipeline",
            passed=not missing and playback_snapshot.playback_started >= 1,
            details={
                "required_events": tuple(event.value for event in required),
                "missing_events": tuple(event.value for event in missing),
                "playback_started": playback_snapshot.playback_started,
                "adapter_playing": playback_snapshot.adapter_playing,
            },
        )

    def _check_interruption(
        self,
        engine: PresenceEngine,
    ) -> PresenceValidationCheck:
        required = (
            EventType.INTERRUPT_REQUESTED,
            EventType.AUDIO_PLAYBACK_STOPPED,
            EventType.ASSISTANT_SPEAKING_STOPPED,
        )
        missing = self._missing_events(engine, required)

        interruption_snapshot = engine.workers.interruption.interruption_snapshot()
        playback_snapshot = engine.workers.audio_playback.playback_snapshot()

        return PresenceValidationCheck(
            name="interruption_pipeline",
            passed=(
                not missing
                and interruption_snapshot.interruptions_requested >= 1
                and playback_snapshot.playback_stopped >= 1
                and engine.adapters.playback.is_playing is False
            ),
            details={
                "required_events": tuple(event.value for event in required),
                "missing_events": tuple(event.value for event in missing),
                "interruptions_requested": (
                    interruption_snapshot.interruptions_requested
                ),
                "playback_stopped": playback_snapshot.playback_stopped,
                "adapter_playing": engine.adapters.playback.is_playing,
            },
        )

    def _check_presence_state(
        self,
        engine: PresenceEngine,
    ) -> PresenceValidationCheck:
        state = engine.presence_store.current_state()

        passed = (
            state.assistant_speaking is False
            and state.mode.value in {"interrupted", "listening", "idle"}
        )

        return PresenceValidationCheck(
            name="presence_state_after_interruption",
            passed=passed,
            details={
                "mode": state.mode.value,
                "turn_phase": state.turn_phase.value,
                "assistant_speaking": state.assistant_speaking,
                "user_speaking": state.user_speaking,
                "active_speech_request_id": state.active_speech_request_id,
            },
        )

    def _check_engine_stopped(
        self,
        engine: PresenceEngine,
    ) -> PresenceValidationCheck:
        snapshot = engine.snapshot()

        return PresenceValidationCheck(
            name="engine_stopped",
            passed=snapshot.running is False and snapshot.stopped is True,
            details={
                "running": snapshot.running,
                "stopped": snapshot.stopped,
                "history_size": snapshot.history_size,
            },
        )

    def _missing_events(
        self,
        engine: PresenceEngine,
        required: tuple[EventType, ...],
    ) -> tuple[EventType, ...]:
        history = engine.event_bus.history()
        seen = {event.event_type for event in history}

        return tuple(event_type for event_type in required if event_type not in seen)

    @staticmethod
    def _history_contains(
        engine: PresenceEngine,
        event_type: EventType,
    ) -> bool:
        return any(
            event.event_type == event_type
            for event in engine.event_bus.history()
        )

    def _wait_for(self, predicate: Callable[[], bool]) -> None:
        deadline = time.monotonic() + self._timeout_seconds

        while time.monotonic() < deadline:
            if predicate():
                return

            time.sleep(self._poll_interval_seconds)

        if not predicate():
            raise TimeoutError("Timed out waiting for Presence validation event.")