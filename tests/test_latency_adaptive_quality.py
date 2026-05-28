from __future__ import annotations

import pytest

from jarvis.latency import (
    AdaptiveQualityLatencyConfig,
    AdaptiveQualityLatencyRuntime,
    ContextDepthMode,
    MemoryRetrievalMode,
    QualityLatencyPressureSample,
    QualityLatencyProfile,
    QualityLatencyReason,
    QualityLatencyStatus,
    SpeculationMode,
    TTSQualityMode,
)


def test_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        AdaptiveQualityLatencyConfig(name=" ").validate()


def test_config_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError):
        AdaptiveQualityLatencyConfig(full_quality_max_ratio=2).validate()

    with pytest.raises(ValueError):
        AdaptiveQualityLatencyConfig(
            full_quality_max_ratio=0.8,
            balanced_max_ratio=0.7,
        ).validate()


def test_config_rejects_invalid_switch_budget() -> None:
    with pytest.raises(ValueError):
        AdaptiveQualityLatencyConfig(switch_budget_ms=0).validate()


def test_config_rejects_invalid_context_tokens() -> None:
    with pytest.raises(ValueError):
        AdaptiveQualityLatencyConfig(fast_context_tokens=0).validate()


def test_runtime_creates_session() -> None:
    runtime = AdaptiveQualityLatencyRuntime()
    state = runtime.create_session()

    assert state.status == QualityLatencyStatus.ACTIVE
    assert state.current_profile == QualityLatencyProfile.FULL_QUALITY
    assert runtime.snapshot().session_count == 1


def test_full_quality_selected_under_40_percent() -> None:
    runtime = AdaptiveQualityLatencyRuntime()
    state = runtime.create_session()

    result = runtime.evaluate_pressure(
        session_id=state.session_id,
        sample=_sample(0.20),
    )

    assert result.success is True
    assert result.decision is not None
    assert result.decision.profile == QualityLatencyProfile.FULL_QUALITY
    assert result.decision.memory_mode == MemoryRetrievalMode.ALL_STREAMS
    assert result.decision.speculation_mode == SpeculationMode.FULL
    assert result.decision.context_mode == ContextDepthMode.MAXIMUM
    assert result.decision.tts_mode == TTSQualityMode.HIGH_QUALITY
    assert result.decision.use_semantic_memory is True
    assert result.decision.use_episodic_memory is True


def test_balanced_selected_between_40_and_70_percent() -> None:
    runtime = AdaptiveQualityLatencyRuntime()
    state = runtime.create_session()

    result = runtime.evaluate_pressure(
        session_id=state.session_id,
        sample=_sample(0.55),
    )

    assert result.success is True
    assert result.decision is not None
    assert result.decision.profile == QualityLatencyProfile.BALANCED
    assert result.decision.memory_mode == (
        MemoryRetrievalMode.EPISODIC_PROFILE_ONLY
    )
    assert result.decision.speculation_mode == SpeculationMode.TOP_ONE
    assert result.decision.context_mode == (
        ContextDepthMode.COMPRESSED_PREVIOUS_TURNS
    )
    assert result.decision.tts_mode == TTSQualityMode.STANDARD
    assert result.decision.use_semantic_memory is False
    assert result.decision.use_episodic_memory is True


def test_fast_mode_selected_above_70_percent() -> None:
    runtime = AdaptiveQualityLatencyRuntime()
    state = runtime.create_session()

    result = runtime.evaluate_pressure(
        session_id=state.session_id,
        sample=_sample(0.90),
    )

    assert result.success is True
    assert result.decision is not None
    assert result.decision.profile == QualityLatencyProfile.FAST_MODE
    assert result.decision.memory_mode == MemoryRetrievalMode.PROFILE_ONLY
    assert result.decision.speculation_mode == SpeculationMode.DISABLED
    assert result.decision.context_mode == (
        ContextDepthMode.MINIMAL_LAST_TWO_PLUS_PROFILE
    )
    assert result.decision.tts_mode == TTSQualityMode.FASTEST
    assert result.decision.use_semantic_memory is False
    assert result.decision.use_episodic_memory is False
    assert result.decision.use_profile_memory is True


def test_profile_switch_is_recorded() -> None:
    runtime = AdaptiveQualityLatencyRuntime()
    state = runtime.create_session()

    runtime.evaluate_pressure(session_id=state.session_id, sample=_sample(0.20))
    result = runtime.evaluate_pressure(
        session_id=state.session_id,
        sample=_sample(0.90),
    )
    updated = runtime.state_for(state.session_id)

    assert result.success is True
    assert updated is not None
    assert updated.current_profile == QualityLatencyProfile.FAST_MODE
    assert updated.switch_count == 1


def test_switch_latency_under_budget() -> None:
    runtime = AdaptiveQualityLatencyRuntime()
    state = runtime.create_session()

    result = runtime.evaluate_pressure(
        session_id=state.session_id,
        sample=_sample(0.90),
    )

    assert result.reason == QualityLatencyReason.SWITCH_WITHIN_BUDGET
    assert result.decision is not None
    assert result.decision.switch_latency_ms < 50.0


def test_current_decision_returns_latest() -> None:
    runtime = AdaptiveQualityLatencyRuntime()
    state = runtime.create_session()

    runtime.evaluate_pressure(session_id=state.session_id, sample=_sample(0.55))
    decision = runtime.current_decision(state.session_id)

    assert decision is not None
    assert decision.profile == QualityLatencyProfile.BALANCED


def test_missing_session_returns_failure() -> None:
    runtime = AdaptiveQualityLatencyRuntime()

    result = runtime.evaluate_pressure(session_id="missing", sample=_sample(0.5))

    assert result.success is False
    assert result.reason == QualityLatencyReason.SESSION_NOT_FOUND


def test_report_is_built() -> None:
    runtime = AdaptiveQualityLatencyRuntime()
    state = runtime.create_session()

    runtime.evaluate_pressure(session_id=state.session_id, sample=_sample(0.20))
    runtime.evaluate_pressure(session_id=state.session_id, sample=_sample(0.55))
    report = runtime.build_report(state.session_id)

    assert report.session_id == state.session_id
    assert report.sample_count == 2
    assert len(report.decisions) == 2
    assert runtime.latest_report() == report


def test_report_rejects_missing_session() -> None:
    runtime = AdaptiveQualityLatencyRuntime()

    with pytest.raises(ValueError):
        runtime.build_report("missing")


def test_cancel_session() -> None:
    runtime = AdaptiveQualityLatencyRuntime()
    state = runtime.create_session()

    result = runtime.cancel_session(state.session_id)

    assert result.success is True
    assert result.status == QualityLatencyStatus.CANCELLED


def test_fail_session() -> None:
    runtime = AdaptiveQualityLatencyRuntime()
    state = runtime.create_session()

    result = runtime.fail_session(state.session_id, error="adaptive failed")

    assert result.success is True
    assert result.status == QualityLatencyStatus.FAILED


def test_snapshot_tracks_counts() -> None:
    runtime = AdaptiveQualityLatencyRuntime()
    state = runtime.create_session()

    runtime.evaluate_pressure(session_id=state.session_id, sample=_sample(0.90))
    runtime.build_report(state.session_id)
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.active_count == 1
    assert snapshot.decision_count == 1
    assert snapshot.report_count == 1


def test_reset_clears_runtime_state() -> None:
    runtime = AdaptiveQualityLatencyRuntime()
    state = runtime.create_session()

    runtime.evaluate_pressure(session_id=state.session_id, sample=_sample(0.90))
    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.decision_count == 0
    assert snapshot.last_reason == QualityLatencyReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert QualityLatencyProfile.FAST_MODE.value == "fast_mode"
    assert MemoryRetrievalMode.PROFILE_ONLY.value == "profile_only"
    assert SpeculationMode.DISABLED.value == "disabled"
    assert TTSQualityMode.FASTEST.value == "fastest"


def _sample(ratio: float) -> QualityLatencyPressureSample:
    return QualityLatencyPressureSample(
        budget_used_ratio=ratio,
        observed_latency_ms=800.0 * ratio,
        budget_ms=800.0,
        queue_depth=2,
        active_workers=4,
    )