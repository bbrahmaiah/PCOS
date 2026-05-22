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
class PresenceLatencyBudget:
    """
    Advisory latency budgets for the Presence runtime.

    These are not hard unit-test limits because CI machines vary. The profiler
    reports whether each stage is within budget so we can tune real-time feel.
    """

    engine_start_ms: float = 250.0
    voice_pipeline_ms: float = 750.0
    response_to_playback_ms: float = 500.0
    interruption_ms: float = 150.0
    engine_stop_ms: float = 250.0

    def validate(self) -> None:
        for name, value in (
            ("engine_start_ms", self.engine_start_ms),
            ("voice_pipeline_ms", self.voice_pipeline_ms),
            ("response_to_playback_ms", self.response_to_playback_ms),
            ("interruption_ms", self.interruption_ms),
            ("engine_stop_ms", self.engine_stop_ms),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero.")


@dataclass(frozen=True, slots=True)
class PresenceLatencyMeasurement:
    """
    One measured Presence runtime stage.
    """

    name: str
    duration_ms: float
    budget_ms: float
    within_budget: bool
    details: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PresenceLatencyReport:
    """
    Complete latency profile for a deterministic PresenceEngine run.
    """

    passed: bool
    started_at: datetime
    finished_at: datetime
    duration_ms: float
    measurements: tuple[PresenceLatencyMeasurement, ...]
    errors: tuple[str, ...]

    @property
    def measurement_count(self) -> int:
        return len(self.measurements)

    @property
    def within_budget_count(self) -> int:
        return sum(1 for item in self.measurements if item.within_budget)

    @property
    def over_budget_count(self) -> int:
        return sum(1 for item in self.measurements if not item.within_budget)


class PresenceLatencyProfiler:
    """
    Deterministic latency profiler for the Phase 2 Presence runtime.

    It uses fake adapters so the profile is stable and does not require real
    microphone/speaker hardware. Real-device smoke profiling comes after this.
    """

    def __init__(
        self,
        *,
        budget: PresenceLatencyBudget | None = None,
        timeout_seconds: float = 2.0,
        poll_interval_seconds: float = 0.005,
    ) -> None:
        self._budget = budget or PresenceLatencyBudget()
        self._budget.validate()

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")

        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be greater than zero.")

        self._timeout_seconds = timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._logger = get_logger("presence.latency")

    def run(self) -> PresenceLatencyReport:
        started_at = datetime.now(UTC)
        profile_started = time.perf_counter()
        measurements: list[PresenceLatencyMeasurement] = []
        errors: list[str] = []
        engine = self._build_engine()

        try:
            measurements.append(
                self._measure(
                    name="engine_start",
                    budget_ms=self._budget.engine_start_ms,
                    operation=engine.start,
                    details_factory=lambda: {
                        "running": engine.snapshot().running,
                        "worker_count": engine.snapshot().worker_count,
                    },
                )
            )

            measurements.append(
                self._measure(
                    name="voice_pipeline",
                    budget_ms=self._budget.voice_pipeline_ms,
                    operation=lambda: self._run_voice_pipeline(engine),
                    details_factory=lambda: {
                        "history_size": len(engine.event_bus.history()),
                        "transcript_seen": self._history_contains(
                            engine,
                            EventType.PRESENCE_TRANSCRIPT_FINAL,
                        ),
                        "response_request_seen": self._history_contains(
                            engine,
                            EventType.ASSISTANT_RESPONSE_REQUESTED,
                        ),
                    },
                )
            )

            measurements.append(
                self._measure(
                    name="response_to_playback",
                    budget_ms=self._budget.response_to_playback_ms,
                    operation=lambda: self._run_response_to_playback(engine),
                    details_factory=lambda: {
                        "playback_started": (
                            engine.workers.audio_playback.playback_snapshot()
                            .playback_started
                        ),
                        "assistant_speaking": (
                            engine.presence_store.current_state()
                            .assistant_speaking
                        ),
                    },
                )
            )

            measurements.append(
                self._measure(
                    name="interruption",
                    budget_ms=self._budget.interruption_ms,
                    operation=lambda: self._run_interruption(engine),
                    details_factory=lambda: {
                        "playback_stopped": (
                            engine.workers.audio_playback.playback_snapshot()
                            .playback_stopped
                        ),
                        "interruptions_requested": (
                            engine.workers.interruption.interruption_snapshot()
                            .interruptions_requested
                        ),
                    },
                )
            )

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            errors.append(error)
            self._logger.error("presence_latency_profile_failed", error=error)

        finally:
            try:
                measurements.append(
                    self._measure(
                        name="engine_stop",
                        budget_ms=self._budget.engine_stop_ms,
                        operation=engine.stop,
                        details_factory=lambda: {
                            "running": engine.snapshot().running,
                            "stopped": engine.snapshot().stopped,
                        },
                    )
                )
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")

        finished_at = datetime.now(UTC)
        duration_ms = (time.perf_counter() - profile_started) * 1000.0

        report = PresenceLatencyReport(
            passed=not errors,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            measurements=tuple(measurements),
            errors=tuple(errors),
        )

        self._logger.info(
            "presence_latency_profile_completed",
            passed=report.passed,
            measurement_count=report.measurement_count,
            within_budget_count=report.within_budget_count,
            over_budget_count=report.over_budget_count,
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
            wake_word=FakeWakeWordAdapter(
                detection_pattern=(True, False, False),
            ),
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
            name="presence_latency_engine",
            adapters=adapters,
        )

    def _run_voice_pipeline(self, engine: PresenceEngine) -> None:
        engine.workers.voice_input.run_once()
        engine.workers.voice_input.run_once()
        engine.workers.voice_input.run_once()

        self._wait_for_event(
            engine,
            EventType.AUDIO_SPEECH_SEGMENT_COMPLETED,
        )
        self._wait_for_event(
            engine,
            EventType.PRESENCE_TRANSCRIPT_FINAL,
        )
        self._wait_for_event(
            engine,
            EventType.ASSISTANT_RESPONSE_REQUESTED,
        )

    def _run_response_to_playback(self, engine: PresenceEngine) -> None:
        self._ensure_waiting_for_response(engine)

        engine.publish_response_ready(text="Yes sir. I am online.")

        self._wait_for_event(
            engine,
            EventType.AUDIO_PLAYBACK_STARTED,
        )
        self._wait_for(
            lambda: engine.presence_store.current_state().assistant_speaking
            is True
        )

    def _run_interruption(self, engine: PresenceEngine) -> None:
        self._wait_for(
            lambda: engine.presence_store.current_state().assistant_speaking
            is True
        )

        engine.workers.interruption.process_user_started_speaking(
            segment_id="latency-interrupt-segment",
            frame_id="latency-interrupt-frame",
        )

        self._wait_for_event(
            engine,
            EventType.AUDIO_PLAYBACK_STOPPED,
        )
        self._wait_for(
            lambda: engine.adapters.playback.is_playing is False
        )

    def _ensure_waiting_for_response(self, engine: PresenceEngine) -> None:
        state = engine.presence_store.current_state()

        if state.turn_phase.value == "waiting_for_response":
            return

        if state.assistant_speaking:
            return

        engine.presence_store.wake_detected(turn_id="latency-turn")
        engine.presence_store.user_speech_started()
        engine.presence_store.user_speech_ended()
        engine.presence_store.transcript_ready()

    def _measure(
        self,
        *,
        name: str,
        budget_ms: float,
        operation: Callable[[], None],
        details_factory: Callable[[], dict[str, Any]],
    ) -> PresenceLatencyMeasurement:
        started = time.perf_counter()
        operation()
        duration_ms = (time.perf_counter() - started) * 1000.0

        return PresenceLatencyMeasurement(
            name=name,
            duration_ms=duration_ms,
            budget_ms=budget_ms,
            within_budget=duration_ms <= budget_ms,
            details=details_factory(),
        )

    def _wait_for_event(
        self,
        engine: PresenceEngine,
        event_type: EventType,
    ) -> None:
        self._wait_for(lambda: self._history_contains(engine, event_type))

    def _wait_for(self, predicate: Callable[[], bool]) -> None:
        deadline = time.monotonic() + self._timeout_seconds

        while time.monotonic() < deadline:
            if predicate():
                return

            time.sleep(self._poll_interval_seconds)

        if not predicate():
            raise TimeoutError("Timed out waiting for Presence latency event.")

    @staticmethod
    def _history_contains(
        engine: PresenceEngine,
        event_type: EventType,
    ) -> bool:
        return any(
            event.event_type == event_type
            for event in engine.event_bus.history()
        )