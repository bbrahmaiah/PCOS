from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.latency import (
    PerceptualFailureMode,
    PerceptualHumanScore,
    PerceptualInteractionRecording,
    PerceptualLatencyInteractionSet,
    PerceptualLatencyReason,
    PerceptualLatencyRuntimeConfig,
    PerceptualLatencySmokeRuntime,
    PerceptualLatencyStatus,
)


def test_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        PerceptualLatencyRuntimeConfig(name=" ").validate()


def test_config_rejects_invalid_recording_count() -> None:
    with pytest.raises(ValueError):
        PerceptualLatencyRuntimeConfig(required_recordings_per_set=0).validate()


def test_config_rejects_invalid_targets() -> None:
    with pytest.raises(ValueError):
        PerceptualLatencyRuntimeConfig(first_audio_target_ms=0).validate()

    with pytest.raises(ValueError):
        PerceptualLatencyRuntimeConfig(interruption_target_ms=0).validate()


def test_config_rejects_invalid_naturalness_score() -> None:
    with pytest.raises(ValueError):
        PerceptualLatencyRuntimeConfig(minimum_naturalness_score=2).validate()


def test_recording_requires_prompt() -> None:
    with pytest.raises(ValidationError):
        PerceptualInteractionRecording(
            interaction_set=PerceptualLatencyInteractionSet.BASELINE,
            prompt=" ",
            first_audio_ms=1,
            first_token_ms=1,
            interruption_recovery_ms=1,
        )


def test_human_score_passed_property() -> None:
    score = PerceptualHumanScore(
        recording_id="recording",
        responds_before_expected=True,
        interruptions_feel_smooth=True,
        streaming_speech_natural=True,
        never_feels_frozen=True,
    )

    assert score.passed is True


def test_runtime_creates_session() -> None:
    runtime = PerceptualLatencySmokeRuntime()
    state = runtime.create_session()

    assert state.status == PerceptualLatencyStatus.CREATED
    assert runtime.snapshot().session_count == 1


def test_runtime_starts_recording() -> None:
    runtime = PerceptualLatencySmokeRuntime()
    state = runtime.create_session()

    result = runtime.start_recording(state.session_id)

    assert result.success is True
    assert result.status == PerceptualLatencyStatus.RECORDING


def test_record_interaction() -> None:
    runtime = PerceptualLatencySmokeRuntime()
    state = runtime.create_session()

    runtime.start_recording(state.session_id)
    recording = _recording(PerceptualLatencyInteractionSet.BASELINE)
    result = runtime.record_interaction(
        session_id=state.session_id,
        recording=recording,
    )

    assert result.success is True
    assert len(runtime.recordings_for(state.session_id)) == 1


def test_record_human_score() -> None:
    runtime = PerceptualLatencySmokeRuntime()
    state = runtime.create_session()
    recording = _recording(PerceptualLatencyInteractionSet.OPTIMIZED)

    runtime.start_recording(state.session_id)
    runtime.record_interaction(session_id=state.session_id, recording=recording)
    result = runtime.record_human_score(
        session_id=state.session_id,
        score=_score(recording.recording_id, passing=True),
    )

    assert result.success is True
    assert len(runtime.scores_for(state.session_id)) == 1


def test_simulated_protocol_passes() -> None:
    runtime = PerceptualLatencySmokeRuntime()
    state = runtime.create_session()

    runtime.start_recording(state.session_id)
    report = runtime.run_simulated_protocol(session_id=state.session_id)

    assert report.status == PerceptualLatencyStatus.PASSED
    assert report.baseline_count == 10
    assert report.optimized_count == 10
    assert report.failed_count == 0
    assert report.perceived_improvement_ms > 0


def test_simulated_protocol_fails_on_perceptual_failures() -> None:
    runtime = PerceptualLatencySmokeRuntime()
    state = runtime.create_session()

    runtime.start_recording(state.session_id)
    report = runtime.run_simulated_protocol(
        session_id=state.session_id,
        failing=True,
    )

    assert report.status == PerceptualLatencyStatus.FAILED
    assert report.failed_count == 10
    assert any(
        PerceptualFailureMode.CHOPPY_TTS_SENTENCES in evaluation.failure_modes
        for evaluation in report.evaluations
    )


def test_insufficient_recordings_rejected() -> None:
    runtime = PerceptualLatencySmokeRuntime()
    state = runtime.create_session()

    runtime.start_recording(state.session_id)

    with pytest.raises(ValueError):
        runtime.build_report(state.session_id)


def test_choppy_tts_failure_detected() -> None:
    runtime = PerceptualLatencySmokeRuntime(
        config=PerceptualLatencyRuntimeConfig(required_recordings_per_set=1)
    )
    state = runtime.create_session()

    runtime.start_recording(state.session_id)
    baseline = _recording(PerceptualLatencyInteractionSet.BASELINE)
    optimized = _recording(
        PerceptualLatencyInteractionSet.OPTIMIZED,
        tts_gap_count=1,
    )
    runtime.record_interaction(session_id=state.session_id, recording=baseline)
    runtime.record_interaction(session_id=state.session_id, recording=optimized)
    runtime.record_human_score(
        session_id=state.session_id,
        score=_score(baseline.recording_id, passing=True),
    )
    runtime.record_human_score(
        session_id=state.session_id,
        score=_score(optimized.recording_id, passing=True),
    )

    report = runtime.build_report(state.session_id)

    assert report.status == PerceptualLatencyStatus.FAILED
    assert PerceptualFailureMode.CHOPPY_TTS_SENTENCES in (
        report.evaluations[0].failure_modes
    )


def test_human_perception_failure_detected() -> None:
    runtime = PerceptualLatencySmokeRuntime(
        config=PerceptualLatencyRuntimeConfig(required_recordings_per_set=1)
    )
    state = runtime.create_session()

    runtime.start_recording(state.session_id)
    baseline = _recording(PerceptualLatencyInteractionSet.BASELINE)
    optimized = _recording(PerceptualLatencyInteractionSet.OPTIMIZED)
    runtime.record_interaction(session_id=state.session_id, recording=baseline)
    runtime.record_interaction(session_id=state.session_id, recording=optimized)
    runtime.record_human_score(
        session_id=state.session_id,
        score=_score(baseline.recording_id, passing=True),
    )
    runtime.record_human_score(
        session_id=state.session_id,
        score=_score(optimized.recording_id, passing=False),
    )

    report = runtime.build_report(state.session_id)

    assert report.status == PerceptualLatencyStatus.FAILED
    assert report.evaluations[0].perception_reason == (
        PerceptualLatencyReason.PERCEPTION_FAILED
    )


def test_cancel_session() -> None:
    runtime = PerceptualLatencySmokeRuntime()
    state = runtime.create_session()

    result = runtime.cancel_session(state.session_id)

    assert result.success is True
    assert result.status == PerceptualLatencyStatus.CANCELLED


def test_report_is_queryable() -> None:
    runtime = PerceptualLatencySmokeRuntime()
    state = runtime.create_session()

    runtime.start_recording(state.session_id)
    report = runtime.run_simulated_protocol(session_id=state.session_id)

    assert runtime.latest_report() == report
    assert runtime.reports() == (report,)


def test_snapshot_tracks_counts() -> None:
    runtime = PerceptualLatencySmokeRuntime()
    state = runtime.create_session()

    runtime.start_recording(state.session_id)
    runtime.run_simulated_protocol(session_id=state.session_id)
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.recording_count == 20
    assert snapshot.score_count == 20
    assert snapshot.passed_count == 1
    assert snapshot.report_count == 1


def test_reset_clears_runtime_state() -> None:
    runtime = PerceptualLatencySmokeRuntime()
    state = runtime.create_session()

    runtime.start_recording(state.session_id)
    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.report_count == 0
    assert snapshot.last_reason == PerceptualLatencyReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert PerceptualLatencyInteractionSet.BASELINE.value == "baseline"
    assert PerceptualLatencyStatus.PASSED.value == "passed"
    assert PerceptualFailureMode.FROZEN_MOMENT.value == "frozen_moment"


def _recording(
    interaction_set: PerceptualLatencyInteractionSet,
    *,
    tts_gap_count: int = 0,
) -> PerceptualInteractionRecording:
    first_audio = (
        900.0
        if interaction_set == PerceptualLatencyInteractionSet.BASELINE
        else 520.0
    )

    return PerceptualInteractionRecording(
        interaction_set=interaction_set,
        prompt="debug this error",
        first_audio_ms=first_audio,
        first_token_ms=260.0,
        interruption_recovery_ms=180.0,
        tts_gap_count=tts_gap_count,
        speech_naturalness_score=0.92,
    )


def _score(recording_id: str, *, passing: bool) -> PerceptualHumanScore:
    return PerceptualHumanScore(
        recording_id=recording_id,
        responds_before_expected=passing,
        interruptions_feel_smooth=passing,
        streaming_speech_natural=passing,
        never_feels_frozen=passing,
    )