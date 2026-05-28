from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class PerceptualLatencyInteractionSet(StrEnum):
    """
    Interaction recording set.

    Baseline = before optimization.
    Optimized = after optimization.
    """

    BASELINE = "baseline"
    OPTIMIZED = "optimized"


class PerceptualLatencyStatus(StrEnum):
    """
    Perceptual smoke lifecycle status.
    """

    CREATED = "created"
    RECORDING = "recording"
    EVALUATING = "evaluating"
    PASSED = "passed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PerceptualFailureMode(StrEnum):
    """
    Perceptual failures that numeric latency can miss.
    """

    NONE = "none"
    CHOPPY_TTS_SENTENCES = "choppy_tts_sentences"
    TTS_STARTED_TOO_EAGERLY = "tts_started_too_eagerly"
    INTERRUPT_WORD_HANG = "interrupt_word_hang"
    MEMORY_HESITATION_VISIBLE = "memory_hesitation_visible"
    FROZEN_MOMENT = "frozen_moment"
    RESPONSE_FEELS_LATE = "response_feels_late"
    STREAMING_UNNATURAL = "streaming_unnatural"


class PerceptualQuestion(StrEnum):
    """
    Human evaluation questions.
    """

    RESPONDS_BEFORE_EXPECTED = "responds_before_expected"
    INTERRUPTIONS_FEEL_SMOOTH = "interruptions_feel_smooth"
    STREAMING_SPEECH_NATURAL = "streaming_speech_natural"
    NEVER_FEELS_FROZEN = "never_feels_frozen"


class PerceptualLatencyReason(StrEnum):
    """
    Machine-readable perceptual smoke reasons.
    """

    SESSION_CREATED = "session_created"
    RECORDING_STARTED = "recording_started"
    INTERACTION_RECORDED = "interaction_recorded"
    HUMAN_SCORE_RECORDED = "human_score_recorded"
    PERCEPTION_PASSED = "perception_passed"
    PERCEPTION_FAILED = "perception_failed"
    METRIC_PASSED = "metric_passed"
    METRIC_FAILED = "metric_failed"
    REPORT_BUILT = "report_built"
    SESSION_CANCELLED = "session_cancelled"
    SESSION_NOT_FOUND = "session_not_found"
    SESSION_NOT_RECORDING = "session_not_recording"
    INSUFFICIENT_RECORDINGS = "insufficient_recordings"
    RUNTIME_RESET = "runtime_reset"


class PerceptualLatencyEventKind(StrEnum):
    """
    Perceptual smoke event kind.
    """

    SESSION_CREATED = "session_created"
    RECORDING_STARTED = "recording_started"
    INTERACTION_RECORDED = "interaction_recorded"
    HUMAN_SCORE_RECORDED = "human_score_recorded"
    REPORT_BUILT = "report_built"
    SESSION_CANCELLED = "session_cancelled"


class PerceptualInteractionRecording(OrchestrationModel):
    """
    One recorded interaction.

    This stores both measured numbers and perception markers that would be
    captured by listening/reviewing the interaction.
    """

    recording_id: str = Field(default_factory=lambda: uuid4().hex)
    interaction_set: PerceptualLatencyInteractionSet
    prompt: str
    first_audio_ms: float = Field(ge=0)
    first_token_ms: float = Field(ge=0)
    interruption_recovery_ms: float = Field(ge=0)
    tts_gap_count: int = Field(default=0, ge=0)
    frozen_moment_count: int = Field(default=0, ge=0)
    word_hang_count: int = Field(default=0, ge=0)
    memory_hesitation_count: int = Field(default=0, ge=0)
    tts_started_too_eagerly: bool = False
    speech_naturalness_score: float = Field(default=1.0, ge=0, le=1)
    created_at_ns: int = Field(default_factory=time.perf_counter_ns, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("recording_id", "prompt")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class PerceptualHumanScore(OrchestrationModel):
    """
    Human evaluation score for one interaction.

    The four boolean questions map directly to the smoke-test protocol.
    """

    score_id: str = Field(default_factory=lambda: uuid4().hex)
    recording_id: str
    responds_before_expected: bool
    interruptions_feel_smooth: bool
    streaming_speech_natural: bool
    never_feels_frozen: bool
    notes: str = ""
    created_at_ns: int = Field(default_factory=time.perf_counter_ns, ge=0)
    created_at: object = Field(default_factory=utc_now)

    @field_validator("score_id", "recording_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @property
    def passed(self) -> bool:
        return (
            self.responds_before_expected
            and self.interruptions_feel_smooth
            and self.streaming_speech_natural
            and self.never_feels_frozen
        )


class PerceptualLatencyEvaluation(OrchestrationModel):
    """
    Evaluation for one recording.
    """

    evaluation_id: str = Field(default_factory=lambda: uuid4().hex)
    recording_id: str
    interaction_set: PerceptualLatencyInteractionSet
    status: PerceptualLatencyStatus
    metric_reason: PerceptualLatencyReason
    perception_reason: PerceptualLatencyReason
    failure_modes: tuple[PerceptualFailureMode, ...] = ()
    first_audio_ms: float = Field(ge=0)
    speech_naturalness_score: float = Field(ge=0, le=1)
    human_score_passed: bool
    created_at: object = Field(default_factory=utc_now)

    @field_validator("evaluation_id", "recording_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class PerceptualLatencyEvent(OrchestrationModel):
    """
    Typed event for perceptual smoke observability.
    """

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    kind: PerceptualLatencyEventKind
    reason: PerceptualLatencyReason
    recording_id: str | None = None
    interaction_set: PerceptualLatencyInteractionSet | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("event_id", "session_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class PerceptualLatencySessionState(OrchestrationModel):
    """
    State for one perceptual latency smoke run.
    """

    session_id: str = Field(default_factory=lambda: uuid4().hex)
    trace_id: str = Field(default_factory=lambda: uuid4().hex)
    status: PerceptualLatencyStatus = PerceptualLatencyStatus.CREATED
    started_at_ns: int | None = None
    completed_at_ns: int | None = None
    cancelled_at_ns: int | None = None
    baseline_count: int = Field(default=0, ge=0)
    optimized_count: int = Field(default=0, ge=0)
    score_count: int = Field(default=0, ge=0)
    failed_count: int = Field(default=0, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("session_id", "trace_id")
    @classmethod
    def _required_ids(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class PerceptualLatencyResult(OrchestrationModel):
    """
    Result from perceptual smoke operation.
    """

    success: bool
    reason: PerceptualLatencyReason
    session_id: str
    status: PerceptualLatencyStatus
    event: PerceptualLatencyEvent | None = None
    state: PerceptualLatencySessionState | None = None
    message: str
    created_at: object = Field(default_factory=utc_now)

    @field_validator("session_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class PerceptualLatencyReport(OrchestrationModel):
    """
    Final perceptual latency smoke report.
    """

    session_id: str
    trace_id: str
    status: PerceptualLatencyStatus
    baseline_count: int = Field(ge=0)
    optimized_count: int = Field(ge=0)
    score_count: int = Field(ge=0)
    passed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    baseline_median_first_audio_ms: float = Field(ge=0)
    optimized_median_first_audio_ms: float = Field(ge=0)
    perceived_improvement_ms: float
    evaluations: tuple[PerceptualLatencyEvaluation, ...]
    events: tuple[PerceptualLatencyEvent, ...]
    created_at: object = Field(default_factory=utc_now)

    @field_validator("session_id", "trace_id")
    @classmethod
    def _required_ids(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @model_validator(mode="after")
    def _counts_match(self) -> PerceptualLatencyReport:
        if self.passed_count + self.failed_count != len(self.evaluations):
            raise ValueError("passed_count + failed_count must match evaluations.")

        return self


class PerceptualLatencyRuntimeSnapshot(OrchestrationModel):
    """
    Runtime diagnostics for Phase 7 Step 18.
    """

    name: str
    session_count: int = Field(ge=0)
    recording_count: int = Field(ge=0)
    score_count: int = Field(ge=0)
    passed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    cancelled_count: int = Field(ge=0)
    report_count: int = Field(ge=0)
    last_reason: PerceptualLatencyReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class PerceptualLatencyRuntimeConfig:
    """
    Phase 7 Step 18 perceptual smoke configuration.
    """

    name: str = "perceptual_latency_smoke_test"
    required_recordings_per_set: int = 10
    first_audio_target_ms: float = 800.0
    interruption_target_ms: float = 300.0
    minimum_naturalness_score: float = 0.80
    require_perceived_improvement: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.required_recordings_per_set < 1:
            raise ValueError("required_recordings_per_set must be positive.")

        if self.first_audio_target_ms <= 0:
            raise ValueError("first_audio_target_ms must be positive.")

        if self.interruption_target_ms <= 0:
            raise ValueError("interruption_target_ms must be positive.")

        if not 0 <= self.minimum_naturalness_score <= 1:
            raise ValueError("minimum_naturalness_score must be within 0..1.")


class PerceptualLatencySmokeRuntime:
    """
    Phase 7 Step 18 Perceptual Latency Smoke Test.

    Responsibilities:
    - record baseline and optimized interactions
    - record human perception scores
    - catch choppy TTS, frozen moments, word hangs, and visible hesitation
    - compare baseline vs optimized perceived speed
    - require both metrics and perception to pass

    Non-responsibilities:
    - no real audio recording
    - no playback engine
    - no TTS synthesis
    - no model execution
    """

    def __init__(
        self,
        *,
        config: PerceptualLatencyRuntimeConfig | None = None,
    ) -> None:
        self._config = config or PerceptualLatencyRuntimeConfig()
        self._config.validate()

        self._states: dict[str, PerceptualLatencySessionState] = {}
        self._recordings: dict[str, list[PerceptualInteractionRecording]] = {}
        self._scores: dict[str, list[PerceptualHumanScore]] = {}
        self._events: dict[str, list[PerceptualLatencyEvent]] = {}
        self._reports: list[PerceptualLatencyReport] = []
        self._lock = RLock()
        self._last_reason: PerceptualLatencyReason | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def create_session(
        self,
        *,
        trace_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> PerceptualLatencySessionState:
        state = PerceptualLatencySessionState(
            trace_id=trace_id or uuid4().hex,
            metadata=metadata or {},
        )
        event = self._event(
            session_id=state.session_id,
            kind=PerceptualLatencyEventKind.SESSION_CREATED,
            reason=PerceptualLatencyReason.SESSION_CREATED,
        )

        with self._lock:
            self._states[state.session_id] = state
            self._recordings[state.session_id] = []
            self._scores[state.session_id] = []
            self._events[state.session_id] = [event]
            self._last_reason = PerceptualLatencyReason.SESSION_CREATED

        return state

    def start_recording(self, session_id: str) -> PerceptualLatencyResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            if state.status != PerceptualLatencyStatus.CREATED:
                return self._failure(
                    session_id=session_id,
                    reason=PerceptualLatencyReason.SESSION_NOT_RECORDING,
                    status=state.status,
                    message="perceptual session cannot start from current state",
                    state=state,
                )

            started = state.model_copy(
                update={
                    "status": PerceptualLatencyStatus.RECORDING,
                    "started_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = started
            event = self._event(
                session_id=session_id,
                kind=PerceptualLatencyEventKind.RECORDING_STARTED,
                reason=PerceptualLatencyReason.RECORDING_STARTED,
            )
            self._events[session_id].append(event)
            self._last_reason = PerceptualLatencyReason.RECORDING_STARTED

        return PerceptualLatencyResult(
            success=True,
            reason=PerceptualLatencyReason.RECORDING_STARTED,
            session_id=session_id,
            status=PerceptualLatencyStatus.RECORDING,
            event=event,
            state=started,
            message="perceptual smoke recording started",
        )

    def record_interaction(
        self,
        *,
        session_id: str,
        recording: PerceptualInteractionRecording,
    ) -> PerceptualLatencyResult:
        state = self.state_for(session_id)

        if state is None:
            return self._missing_session(session_id)

        if state.status != PerceptualLatencyStatus.RECORDING:
            return self._failure(
                session_id=session_id,
                reason=PerceptualLatencyReason.SESSION_NOT_RECORDING,
                status=state.status,
                message="perceptual session is not recording",
                state=state,
            )

        with self._lock:
            self._recordings[session_id].append(recording)
            current = self._states[session_id]
            baseline_count = current.baseline_count
            optimized_count = current.optimized_count

            if recording.interaction_set == PerceptualLatencyInteractionSet.BASELINE:
                baseline_count += 1
            else:
                optimized_count += 1

            updated = current.model_copy(
                update={
                    "baseline_count": baseline_count,
                    "optimized_count": optimized_count,
                }
            )
            self._states[session_id] = updated
            event = self._event(
                session_id=session_id,
                kind=PerceptualLatencyEventKind.INTERACTION_RECORDED,
                reason=PerceptualLatencyReason.INTERACTION_RECORDED,
                recording_id=recording.recording_id,
                interaction_set=recording.interaction_set,
            )
            self._events[session_id].append(event)
            self._last_reason = PerceptualLatencyReason.INTERACTION_RECORDED

        return PerceptualLatencyResult(
            success=True,
            reason=PerceptualLatencyReason.INTERACTION_RECORDED,
            session_id=session_id,
            status=PerceptualLatencyStatus.RECORDING,
            event=event,
            state=updated,
            message="perceptual interaction recorded",
        )

    def record_human_score(
        self,
        *,
        session_id: str,
        score: PerceptualHumanScore,
    ) -> PerceptualLatencyResult:
        state = self.state_for(session_id)

        if state is None:
            return self._missing_session(session_id)

        if state.status != PerceptualLatencyStatus.RECORDING:
            return self._failure(
                session_id=session_id,
                reason=PerceptualLatencyReason.SESSION_NOT_RECORDING,
                status=state.status,
                message="perceptual session is not recording",
                state=state,
            )

        with self._lock:
            self._scores[session_id].append(score)
            current = self._states[session_id]
            updated = current.model_copy(
                update={"score_count": current.score_count + 1}
            )
            self._states[session_id] = updated
            event = self._event(
                session_id=session_id,
                kind=PerceptualLatencyEventKind.HUMAN_SCORE_RECORDED,
                reason=PerceptualLatencyReason.HUMAN_SCORE_RECORDED,
                recording_id=score.recording_id,
            )
            self._events[session_id].append(event)
            self._last_reason = PerceptualLatencyReason.HUMAN_SCORE_RECORDED

        return PerceptualLatencyResult(
            success=True,
            reason=PerceptualLatencyReason.HUMAN_SCORE_RECORDED,
            session_id=session_id,
            status=PerceptualLatencyStatus.RECORDING,
            event=event,
            state=updated,
            message="human perceptual score recorded",
        )

    def run_simulated_protocol(
        self,
        *,
        session_id: str,
        failing: bool = False,
    ) -> PerceptualLatencyReport:
        state = self.state_for(session_id)

        if state is None:
            raise ValueError(f"perceptual session not found: {session_id}")

        if state.status != PerceptualLatencyStatus.RECORDING:
            raise ValueError("perceptual session is not recording")

        for index in range(self._config.required_recordings_per_set):
            baseline = self._simulated_recording(
                interaction_set=PerceptualLatencyInteractionSet.BASELINE,
                index=index,
                failing=False,
            )
            optimized = self._simulated_recording(
                interaction_set=PerceptualLatencyInteractionSet.OPTIMIZED,
                index=index,
                failing=failing,
            )
            self.record_interaction(session_id=session_id, recording=baseline)
            self.record_interaction(session_id=session_id, recording=optimized)

            self.record_human_score(
                session_id=session_id,
                score=self._simulated_score(baseline, passing=True),
            )
            self.record_human_score(
                session_id=session_id,
                score=self._simulated_score(optimized, passing=not failing),
            )

        return self.build_report(session_id)

    def build_report(self, session_id: str) -> PerceptualLatencyReport:
        state = self.state_for(session_id)

        if state is None:
            raise ValueError(f"perceptual session not found: {session_id}")

        recordings = self.recordings_for(session_id)
        baseline = tuple(
            item
            for item in recordings
            if item.interaction_set == PerceptualLatencyInteractionSet.BASELINE
        )
        optimized = tuple(
            item
            for item in recordings
            if item.interaction_set == PerceptualLatencyInteractionSet.OPTIMIZED
        )

        if (
            len(baseline) < self._config.required_recordings_per_set
            or len(optimized) < self._config.required_recordings_per_set
        ):
            raise ValueError("insufficient baseline or optimized recordings")

        evaluations = tuple(
            self._evaluate_recording(
                recording=recording,
                score=self._score_for_recording(
                    session_id=session_id,
                    recording_id=recording.recording_id,
                ),
            )
            for recording in optimized
        )
        failed_count = sum(
            1
            for evaluation in evaluations
            if evaluation.status == PerceptualLatencyStatus.FAILED
        )
        passed_count = len(evaluations) - failed_count
        baseline_median = statistics.median(
            item.first_audio_ms for item in baseline
        )
        optimized_median = statistics.median(
            item.first_audio_ms for item in optimized
        )
        perceived_improvement = baseline_median - optimized_median

        if (
            self._config.require_perceived_improvement
            and perceived_improvement <= 0
        ):
            failed_count = len(evaluations)
            passed_count = 0

        final_status = (
            PerceptualLatencyStatus.FAILED
            if failed_count > 0
            else PerceptualLatencyStatus.PASSED
        )

        with self._lock:
            current = self._states[session_id]
            completed = current.model_copy(
                update={
                    "status": final_status,
                    "completed_at_ns": time.perf_counter_ns(),
                    "failed_count": failed_count,
                }
            )
            self._states[session_id] = completed
            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=PerceptualLatencyEventKind.REPORT_BUILT,
                    reason=PerceptualLatencyReason.REPORT_BUILT,
                )
            )
            self._last_reason = PerceptualLatencyReason.REPORT_BUILT

        report = PerceptualLatencyReport(
            session_id=session_id,
            trace_id=state.trace_id,
            status=final_status,
            baseline_count=len(baseline),
            optimized_count=len(optimized),
            score_count=len(self.scores_for(session_id)),
            passed_count=passed_count,
            failed_count=failed_count,
            baseline_median_first_audio_ms=baseline_median,
            optimized_median_first_audio_ms=optimized_median,
            perceived_improvement_ms=perceived_improvement,
            evaluations=evaluations,
            events=self.events_for(session_id),
        )

        with self._lock:
            self._reports.append(report)

        return report

    def cancel_session(self, session_id: str) -> PerceptualLatencyResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            cancelled = state.model_copy(
                update={
                    "status": PerceptualLatencyStatus.CANCELLED,
                    "cancelled_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = cancelled
            event = self._event(
                session_id=session_id,
                kind=PerceptualLatencyEventKind.SESSION_CANCELLED,
                reason=PerceptualLatencyReason.SESSION_CANCELLED,
            )
            self._events[session_id].append(event)
            self._last_reason = PerceptualLatencyReason.SESSION_CANCELLED

        return PerceptualLatencyResult(
            success=True,
            reason=PerceptualLatencyReason.SESSION_CANCELLED,
            session_id=session_id,
            status=PerceptualLatencyStatus.CANCELLED,
            event=event,
            state=cancelled,
            message="perceptual smoke session cancelled",
        )

    def state_for(self, session_id: str) -> PerceptualLatencySessionState | None:
        with self._lock:
            return self._states.get(session_id)

    def recordings_for(
        self,
        session_id: str,
    ) -> tuple[PerceptualInteractionRecording, ...]:
        with self._lock:
            return tuple(self._recordings.get(session_id, ()))

    def scores_for(
        self,
        session_id: str,
    ) -> tuple[PerceptualHumanScore, ...]:
        with self._lock:
            return tuple(self._scores.get(session_id, ()))

    def events_for(
        self,
        session_id: str,
    ) -> tuple[PerceptualLatencyEvent, ...]:
        with self._lock:
            return tuple(self._events.get(session_id, ()))

    def reports(self) -> tuple[PerceptualLatencyReport, ...]:
        with self._lock:
            return tuple(self._reports)

    def latest_report(self) -> PerceptualLatencyReport | None:
        with self._lock:
            if not self._reports:
                return None

            return self._reports[-1]

    def snapshot(self) -> PerceptualLatencyRuntimeSnapshot:
        with self._lock:
            states = tuple(self._states.values())

            return PerceptualLatencyRuntimeSnapshot(
                name=self.name,
                session_count=len(states),
                recording_count=sum(
                    len(items) for items in self._recordings.values()
                ),
                score_count=sum(len(items) for items in self._scores.values()),
                passed_count=sum(
                    1
                    for state in states
                    if state.status == PerceptualLatencyStatus.PASSED
                ),
                failed_count=sum(
                    1
                    for state in states
                    if state.status == PerceptualLatencyStatus.FAILED
                ),
                cancelled_count=sum(
                    1
                    for state in states
                    if state.status == PerceptualLatencyStatus.CANCELLED
                ),
                report_count=len(self._reports),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        with self._lock:
            self._states.clear()
            self._recordings.clear()
            self._scores.clear()
            self._events.clear()
            self._reports.clear()
            self._last_reason = PerceptualLatencyReason.RUNTIME_RESET

    def _evaluate_recording(
        self,
        *,
        recording: PerceptualInteractionRecording,
        score: PerceptualHumanScore | None,
    ) -> PerceptualLatencyEvaluation:
        failure_modes = self._failure_modes(recording)
        metric_passed = (
            recording.first_audio_ms <= self._config.first_audio_target_ms
            and recording.interruption_recovery_ms
            <= self._config.interruption_target_ms
            and recording.speech_naturalness_score
            >= self._config.minimum_naturalness_score
            and not failure_modes
        )
        perception_passed = score.passed if score is not None else False
        status = (
            PerceptualLatencyStatus.PASSED
            if metric_passed and perception_passed
            else PerceptualLatencyStatus.FAILED
        )

        return PerceptualLatencyEvaluation(
            recording_id=recording.recording_id,
            interaction_set=recording.interaction_set,
            status=status,
            metric_reason=(
                PerceptualLatencyReason.METRIC_PASSED
                if metric_passed
                else PerceptualLatencyReason.METRIC_FAILED
            ),
            perception_reason=(
                PerceptualLatencyReason.PERCEPTION_PASSED
                if perception_passed
                else PerceptualLatencyReason.PERCEPTION_FAILED
            ),
            failure_modes=failure_modes,
            first_audio_ms=recording.first_audio_ms,
            speech_naturalness_score=recording.speech_naturalness_score,
            human_score_passed=perception_passed,
        )

    @staticmethod
    def _failure_modes(
        recording: PerceptualInteractionRecording,
    ) -> tuple[PerceptualFailureMode, ...]:
        modes: list[PerceptualFailureMode] = []

        if recording.tts_gap_count > 0:
            modes.append(PerceptualFailureMode.CHOPPY_TTS_SENTENCES)

        if recording.tts_started_too_eagerly:
            modes.append(PerceptualFailureMode.TTS_STARTED_TOO_EAGERLY)

        if recording.word_hang_count > 0:
            modes.append(PerceptualFailureMode.INTERRUPT_WORD_HANG)

        if recording.memory_hesitation_count > 0:
            modes.append(PerceptualFailureMode.MEMORY_HESITATION_VISIBLE)

        if recording.frozen_moment_count > 0:
            modes.append(PerceptualFailureMode.FROZEN_MOMENT)

        if recording.speech_naturalness_score < 0.80:
            modes.append(PerceptualFailureMode.STREAMING_UNNATURAL)

        return tuple(modes)

    def _score_for_recording(
        self,
        *,
        session_id: str,
        recording_id: str,
    ) -> PerceptualHumanScore | None:
        for score in self.scores_for(session_id):
            if score.recording_id == recording_id:
                return score

        return None

    @staticmethod
    def _simulated_score(
        recording: PerceptualInteractionRecording,
        *,
        passing: bool,
    ) -> PerceptualHumanScore:
        return PerceptualHumanScore(
            recording_id=recording.recording_id,
            responds_before_expected=passing,
            interruptions_feel_smooth=passing,
            streaming_speech_natural=passing,
            never_feels_frozen=passing,
        )

    @staticmethod
    def _simulated_recording(
        *,
        interaction_set: PerceptualLatencyInteractionSet,
        index: int,
        failing: bool,
    ) -> PerceptualInteractionRecording:
        if interaction_set == PerceptualLatencyInteractionSet.BASELINE:
            first_audio_ms = 900.0 + index
            first_token_ms = 500.0 + index
            interruption_ms = 360.0 + index
            naturalness = 0.72
        else:
            first_audio_ms = 520.0 + index
            first_token_ms = 280.0 + index
            interruption_ms = 180.0 + index
            naturalness = 0.92

        return PerceptualInteractionRecording(
            interaction_set=interaction_set,
            prompt=f"interaction {index}",
            first_audio_ms=first_audio_ms if not failing else 900.0,
            first_token_ms=first_token_ms,
            interruption_recovery_ms=(
                interruption_ms if not failing else 420.0
            ),
            tts_gap_count=1 if failing else 0,
            frozen_moment_count=1 if failing else 0,
            word_hang_count=1 if failing else 0,
            memory_hesitation_count=1 if failing else 0,
            tts_started_too_eagerly=failing,
            speech_naturalness_score=naturalness if not failing else 0.50,
        )

    @staticmethod
    def _event(
        *,
        session_id: str,
        kind: PerceptualLatencyEventKind,
        reason: PerceptualLatencyReason,
        recording_id: str | None = None,
        interaction_set: PerceptualLatencyInteractionSet | None = None,
        metadata: dict[str, object] | None = None,
    ) -> PerceptualLatencyEvent:
        return PerceptualLatencyEvent(
            session_id=session_id,
            kind=kind,
            reason=reason,
            recording_id=recording_id,
            interaction_set=interaction_set,
            metadata=metadata or {},
        )

    @staticmethod
    def _missing_session(session_id: str) -> PerceptualLatencyResult:
        return PerceptualLatencyResult(
            success=False,
            reason=PerceptualLatencyReason.SESSION_NOT_FOUND,
            session_id=session_id,
            status=PerceptualLatencyStatus.FAILED,
            message="perceptual latency session not found",
        )

    @staticmethod
    def _failure(
        *,
        session_id: str,
        reason: PerceptualLatencyReason,
        status: PerceptualLatencyStatus,
        message: str,
        state: PerceptualLatencySessionState | None = None,
    ) -> PerceptualLatencyResult:
        return PerceptualLatencyResult(
            success=False,
            reason=reason,
            session_id=session_id,
            status=status,
            state=state,
            message=message,
        )