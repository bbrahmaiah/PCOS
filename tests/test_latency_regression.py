from __future__ import annotations

import pytest

from jarvis.latency import (
    LatencyRegressionMachineProfile,
    LatencyRegressionMetric,
    LatencyRegressionReason,
    LatencyRegressionRuntime,
    LatencyRegressionRuntimeConfig,
    LatencyRegressionSample,
    LatencyRegressionStatus,
)


def test_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        LatencyRegressionRuntimeConfig(name=" ").validate()


def test_config_rejects_invalid_targets() -> None:
    with pytest.raises(ValueError):
        LatencyRegressionRuntimeConfig(voice_first_word_p95_ms=0).validate()

    with pytest.raises(ValueError):
        LatencyRegressionRuntimeConfig(memory_retrieval_p95_ms=0).validate()


def test_config_rejects_invalid_guardrails() -> None:
    with pytest.raises(ValueError):
        LatencyRegressionRuntimeConfig(memory_overhead_limit_ratio=-1).validate()

    with pytest.raises(ValueError):
        LatencyRegressionRuntimeConfig(min_samples=0).validate()


def test_runtime_creates_session_with_default_contracts() -> None:
    runtime = LatencyRegressionRuntime()
    state = runtime.create_session()

    assert state.status == LatencyRegressionStatus.CREATED
    assert state.contract_count == 12
    assert runtime.snapshot().session_count == 1


def test_runtime_starts_session() -> None:
    runtime = LatencyRegressionRuntime()
    state = runtime.create_session()

    result = runtime.start_session(state.session_id)

    assert result.success is True
    assert result.status == LatencyRegressionStatus.RUNNING


def test_record_sample() -> None:
    runtime = LatencyRegressionRuntime()
    state = runtime.create_session()

    runtime.start_session(state.session_id)
    result = runtime.record_sample(
        session_id=state.session_id,
        sample=_sample(
            metric=LatencyRegressionMetric.VOICE_FIRST_WORD,
            profile=LatencyRegressionMachineProfile.FAST_MACHINE,
            latency_ms=600.0,
        ),
    )

    assert result.success is True
    assert result.reason == LatencyRegressionReason.SAMPLE_RECORDED
    assert len(runtime.samples_for(state.session_id)) == 1


def test_record_sample_rejects_non_running_session() -> None:
    runtime = LatencyRegressionRuntime()
    state = runtime.create_session()

    result = runtime.record_sample(
        session_id=state.session_id,
        sample=_sample(
            metric=LatencyRegressionMetric.VOICE_FIRST_WORD,
            profile=LatencyRegressionMachineProfile.FAST_MACHINE,
            latency_ms=600.0,
        ),
    )

    assert result.success is False
    assert result.reason == LatencyRegressionReason.SESSION_NOT_RUNNING


def test_simulated_suite_passes() -> None:
    runtime = LatencyRegressionRuntime()
    state = runtime.create_session()

    runtime.start_session(state.session_id)
    report = runtime.run_simulated_suite(session_id=state.session_id)

    assert report.status == LatencyRegressionStatus.PASSED
    assert report.contract_count == 12
    assert report.failed_count == 0
    assert report.passed_count == 12


def test_simulated_suite_fails_when_regressed() -> None:
    runtime = LatencyRegressionRuntime()
    state = runtime.create_session()

    runtime.start_session(state.session_id)
    report = runtime.run_simulated_suite(
        session_id=state.session_id,
        failing=True,
    )

    assert report.status == LatencyRegressionStatus.FAILED
    assert report.failed_count == 12
    assert all(
        evaluation.status == LatencyRegressionStatus.FAILED
        for evaluation in report.evaluations
    )


def test_p95_latency_failure_is_detected() -> None:
    runtime = LatencyRegressionRuntime()
    state = runtime.create_session()

    runtime.start_session(state.session_id)

    for latency in (700.0, 760.0, 810.0, 900.0, 950.0):
        runtime.record_sample(
            session_id=state.session_id,
            sample=_sample(
                metric=LatencyRegressionMetric.VOICE_FIRST_WORD,
                profile=LatencyRegressionMachineProfile.FAST_MACHINE,
                latency_ms=latency,
            ),
        )

    report = runtime.build_report(state.session_id)
    failed = [
        evaluation
        for evaluation in report.evaluations
        if evaluation.contract.metric == LatencyRegressionMetric.VOICE_FIRST_WORD
        and evaluation.contract.machine_profile == (
            LatencyRegressionMachineProfile.FAST_MACHINE
        )
    ][0]

    assert failed.status == LatencyRegressionStatus.FAILED
    assert failed.reason == LatencyRegressionReason.CONTRACT_FAILED_P95


def test_correctness_failure_is_detected() -> None:
    runtime = LatencyRegressionRuntime()
    state = runtime.create_session()

    runtime.start_session(state.session_id)

    for index in range(5):
        runtime.record_sample(
            session_id=state.session_id,
            sample=_sample(
                metric=LatencyRegressionMetric.MEMORY_RETRIEVAL,
                profile=LatencyRegressionMachineProfile.FAST_MACHINE,
                latency_ms=80.0,
                correctness_ok=index != 3,
            ),
        )

    report = runtime.build_report(state.session_id)
    failed = [
        evaluation
        for evaluation in report.evaluations
        if evaluation.contract.metric == LatencyRegressionMetric.MEMORY_RETRIEVAL
        and evaluation.contract.machine_profile == (
            LatencyRegressionMachineProfile.FAST_MACHINE
        )
    ][0]

    assert failed.reason == LatencyRegressionReason.CONTRACT_FAILED_CORRECTNESS


def test_memory_guardrail_failure_is_detected() -> None:
    runtime = LatencyRegressionRuntime()
    state = runtime.create_session()

    runtime.start_session(state.session_id)

    for _ in range(5):
        runtime.record_sample(
            session_id=state.session_id,
            sample=_sample(
                metric=LatencyRegressionMetric.STREAMING_FIRST_TOKEN,
                profile=LatencyRegressionMachineProfile.FAST_MACHINE,
                latency_ms=100.0,
                memory_overhead_ratio=0.25,
            ),
        )

    report = runtime.build_report(state.session_id)
    failed = [
        evaluation
        for evaluation in report.evaluations
        if evaluation.contract.metric == (
            LatencyRegressionMetric.STREAMING_FIRST_TOKEN
        )
        and evaluation.contract.machine_profile == (
            LatencyRegressionMachineProfile.FAST_MACHINE
        )
    ][0]

    assert failed.reason == LatencyRegressionReason.CONTRACT_FAILED_MEMORY


def test_cpu_guardrail_failure_is_detected() -> None:
    runtime = LatencyRegressionRuntime()
    state = runtime.create_session()

    runtime.start_session(state.session_id)

    for _ in range(5):
        runtime.record_sample(
            session_id=state.session_id,
            sample=_sample(
                metric=LatencyRegressionMetric.INTERRUPTION_RECOVERY,
                profile=LatencyRegressionMachineProfile.FAST_MACHINE,
                latency_ms=100.0,
                cpu_spike_ratio=0.40,
            ),
        )

    report = runtime.build_report(state.session_id)
    failed = [
        evaluation
        for evaluation in report.evaluations
        if evaluation.contract.metric == (
            LatencyRegressionMetric.INTERRUPTION_RECOVERY
        )
        and evaluation.contract.machine_profile == (
            LatencyRegressionMachineProfile.FAST_MACHINE
        )
    ][0]

    assert failed.reason == LatencyRegressionReason.CONTRACT_FAILED_CPU


def test_percentile_calculation() -> None:
    assert LatencyRegressionRuntime.percentile((1.0,), 95.0) == 1.0
    assert LatencyRegressionRuntime.percentile((), 95.0) == 0.0
    assert LatencyRegressionRuntime.percentile((100.0, 200.0), 50.0) == 150.0


def test_cancel_session() -> None:
    runtime = LatencyRegressionRuntime()
    state = runtime.create_session()

    result = runtime.cancel_session(state.session_id)

    assert result.success is True
    assert result.status == LatencyRegressionStatus.CANCELLED


def test_report_is_queryable() -> None:
    runtime = LatencyRegressionRuntime()
    state = runtime.create_session()

    runtime.start_session(state.session_id)
    report = runtime.run_simulated_suite(session_id=state.session_id)

    assert runtime.latest_report() == report
    assert runtime.reports() == (report,)


def test_snapshot_tracks_counts() -> None:
    runtime = LatencyRegressionRuntime()
    state = runtime.create_session()

    runtime.start_session(state.session_id)
    runtime.run_simulated_suite(session_id=state.session_id)
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.passed_count == 1
    assert snapshot.report_count == 1


def test_reset_clears_runtime_state() -> None:
    runtime = LatencyRegressionRuntime()
    state = runtime.create_session()

    runtime.start_session(state.session_id)
    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.report_count == 0
    assert snapshot.last_reason == LatencyRegressionReason.RUNTIME_RESET


def test_enum_values_are_stable() -> None:
    assert LatencyRegressionMetric.VOICE_FIRST_WORD.value == "voice_first_word"
    assert LatencyRegressionMachineProfile.HIGH_LOAD.value == "high_load"
    assert LatencyRegressionStatus.PASSED.value == "passed"
    assert LatencyRegressionReason.CONTRACT_FAILED_P95.value == (
        "contract_failed_p95"
    )


def _sample(
    *,
    metric: LatencyRegressionMetric,
    profile: LatencyRegressionMachineProfile,
    latency_ms: float,
    correctness_ok: bool = True,
    memory_overhead_ratio: float = 0.0,
    cpu_spike_ratio: float = 0.0,
) -> LatencyRegressionSample:
    return LatencyRegressionSample(
        metric=metric,
        machine_profile=profile,
        latency_ms=latency_ms,
        correctness_ok=correctness_ok,
        memory_overhead_ratio=memory_overhead_ratio,
        cpu_spike_ratio=cpu_spike_ratio,
    )