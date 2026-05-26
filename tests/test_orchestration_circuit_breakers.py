from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from jarvis.orchestration import (
    BreakerState,
    CircuitBreakerDecision,
    CircuitBreakerReason,
    CircuitBreakerRuntime,
    CircuitBreakerRuntimeConfig,
    FailureThreshold,
    FallbackMode,
    WorkerCapability,
    WorkerFailureKind,
    new_worker_id,
)
from jarvis.orchestration.ids import utc_now


def threshold(max_failures: int = 2) -> FailureThreshold:
    return FailureThreshold(
        max_failures=max_failures,
        initial_probe_delay_ms=5_000,
        max_probe_delay_ms=60_000,
    )


def registered_runtime(
    *,
    capability: WorkerCapability = WorkerCapability.COGNITION,
    max_failures: int = 2,
) -> tuple[CircuitBreakerRuntime, str]:
    runtime = CircuitBreakerRuntime()
    worker_id = new_worker_id()
    runtime.register_worker(
        worker_id=worker_id,
        capability=capability,
        threshold=threshold(max_failures=max_failures),
    )

    return runtime, worker_id


def fail(runtime: CircuitBreakerRuntime, worker_id: str) -> None:
    runtime.record_failure(
        worker_id=worker_id,
        failure_kind=WorkerFailureKind.EXCEPTION,
        message="worker failed",
    )


def probe_allowed_at(
    runtime: CircuitBreakerRuntime,
    worker_id: str,
) -> datetime:
    breaker = runtime.breaker_for(worker_id)

    assert breaker is not None
    assert breaker.next_probe is not None
    assert isinstance(breaker.next_probe.allowed_at, datetime)

    return breaker.next_probe.allowed_at


def test_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        CircuitBreakerRuntimeConfig(name=" ").validate()


def test_threshold_rejects_invalid_backoff() -> None:
    with pytest.raises(ValidationError):
        FailureThreshold(
            max_failures=1,
            initial_probe_delay_ms=10_000,
            max_probe_delay_ms=5_000,
        )


def test_probe_delay_uses_exponential_backoff() -> None:
    item = FailureThreshold(
        max_failures=1,
        initial_probe_delay_ms=5_000,
        max_probe_delay_ms=60_000,
    )

    assert item.probe_delay_for_attempt(0) == 5_000
    assert item.probe_delay_for_attempt(1) == 10_000
    assert item.probe_delay_for_attempt(2) == 20_000


def test_register_worker_creates_closed_breaker() -> None:
    runtime, worker_id = registered_runtime()

    breaker = runtime.breaker_for(worker_id)

    assert breaker is not None
    assert breaker.state == BreakerState.CLOSED
    assert breaker.fallback_mode == FallbackMode.SIMPLIFIED_RESPONSE


def test_allow_request_closed_worker() -> None:
    runtime, worker_id = registered_runtime()

    result = runtime.allow_request(worker_id)

    assert result.success is True
    assert result.decision == CircuitBreakerDecision.ALLOWED
    assert result.reason == CircuitBreakerReason.WORKER_ALLOWED


def test_failure_below_threshold_records_only() -> None:
    runtime, worker_id = registered_runtime(max_failures=3)

    result = runtime.record_failure(
        worker_id=worker_id,
        failure_kind=WorkerFailureKind.TIMEOUT,
        message="timeout",
    )

    assert result.success is True
    assert result.decision == CircuitBreakerDecision.RECORDED
    assert result.reason == CircuitBreakerReason.FAILURE_RECORDED
    assert result.breaker is not None
    assert result.breaker.state == BreakerState.CLOSED


def test_failure_threshold_opens_breaker() -> None:
    runtime, worker_id = registered_runtime(max_failures=2)

    fail(runtime, worker_id)
    result = runtime.record_failure(
        worker_id=worker_id,
        failure_kind=WorkerFailureKind.EXCEPTION,
        message="worker failed again",
    )

    assert result.success is False
    assert result.decision == CircuitBreakerDecision.OPENED
    assert result.reason == CircuitBreakerReason.FAILURE_THRESHOLD_REACHED
    assert result.breaker is not None
    assert result.breaker.state == BreakerState.OPEN
    assert result.fallback_mode == FallbackMode.SIMPLIFIED_RESPONSE


def test_open_breaker_blocks_request_and_returns_fallback() -> None:
    runtime, worker_id = registered_runtime(max_failures=1)

    fail(runtime, worker_id)
    result = runtime.allow_request(worker_id)

    assert result.success is False
    assert result.decision == CircuitBreakerDecision.BLOCKED
    assert result.reason == CircuitBreakerReason.WORKER_ISOLATED_OPEN
    assert result.fallback_mode == FallbackMode.SIMPLIFIED_RESPONSE


def test_memory_worker_uses_stateless_turn_fallback() -> None:
    runtime, worker_id = registered_runtime(
        capability=WorkerCapability.MEMORY,
        max_failures=1,
    )

    fail(runtime, worker_id)
    result = runtime.fallback_for(worker_id)

    assert result.success is True
    assert result.fallback_mode == FallbackMode.STATELESS_TURN


def test_tool_worker_uses_safe_mode_fallback() -> None:
    runtime, worker_id = registered_runtime(
        capability=WorkerCapability.TOOL_ACTION,
        max_failures=1,
    )

    fail(runtime, worker_id)
    result = runtime.fallback_for(worker_id)

    assert result.success is True
    assert result.fallback_mode == FallbackMode.SAFE_MODE_ONLY


def test_background_worker_uses_disabled_fallback() -> None:
    runtime, worker_id = registered_runtime(
        capability=WorkerCapability.BACKGROUND,
        max_failures=1,
    )

    fail(runtime, worker_id)
    result = runtime.fallback_for(worker_id)

    assert result.success is True
    assert result.fallback_mode == FallbackMode.DISABLED_WORKER


def test_probe_before_interval_is_blocked() -> None:
    runtime, worker_id = registered_runtime(max_failures=1)

    fail(runtime, worker_id)
    result = runtime.allow_request(worker_id, now=utc_now())

    assert result.success is False
    assert result.decision == CircuitBreakerDecision.BLOCKED


def test_probe_after_interval_half_opens_breaker() -> None:
    runtime, worker_id = registered_runtime(max_failures=1)

    fail(runtime, worker_id)

    probe_time = probe_allowed_at(
        runtime,
        worker_id,
    ) + timedelta(milliseconds=1)
    result = runtime.allow_request(worker_id, now=probe_time)

    assert result.success is True
    assert result.decision == CircuitBreakerDecision.HALF_OPENED
    assert result.breaker is not None
    assert result.breaker.state == BreakerState.HALF_OPEN


def test_half_open_success_recovers_worker() -> None:
    runtime, worker_id = registered_runtime(max_failures=1)

    fail(runtime, worker_id)

    probe_time = probe_allowed_at(
        runtime,
        worker_id,
    ) + timedelta(milliseconds=1)
    runtime.allow_request(worker_id, now=probe_time)

    result = runtime.record_success(worker_id)

    assert result.success is True
    assert result.decision == CircuitBreakerDecision.RECOVERED
    assert result.reason == CircuitBreakerReason.HALF_OPEN_SUCCESS_RECOVERED
    assert result.breaker is not None
    assert result.breaker.state == BreakerState.CLOSED


def test_half_open_failure_reopens_with_backoff() -> None:
    runtime, worker_id = registered_runtime(max_failures=1)

    fail(runtime, worker_id)
    first_open = runtime.breaker_for(worker_id)

    assert first_open is not None

    first_probe_attempt = first_open.probe_attempt
    probe_time = probe_allowed_at(
        runtime,
        worker_id,
    ) + timedelta(milliseconds=1)

    runtime.allow_request(worker_id, now=probe_time)
    result = runtime.record_failure(
        worker_id=worker_id,
        failure_kind=WorkerFailureKind.EXCEPTION,
        message="probe failed",
    )

    assert result.success is False
    assert result.decision == CircuitBreakerDecision.OPENED
    assert result.reason == CircuitBreakerReason.HALF_OPEN_FAILURE_REOPENED
    assert result.breaker is not None
    assert result.breaker.state == BreakerState.OPEN
    assert result.breaker.probe_attempt > first_probe_attempt


def test_unknown_worker_rejected() -> None:
    runtime = CircuitBreakerRuntime()

    result = runtime.allow_request(new_worker_id())

    assert result.success is False
    assert result.decision == CircuitBreakerDecision.REJECTED
    assert result.reason == CircuitBreakerReason.WORKER_NOT_FOUND


def test_one_worker_failure_does_not_block_other_worker() -> None:
    runtime = CircuitBreakerRuntime()
    bad_worker = new_worker_id()
    good_worker = new_worker_id()

    runtime.register_worker(
        worker_id=bad_worker,
        capability=WorkerCapability.COGNITION,
        threshold=threshold(max_failures=1),
    )
    runtime.register_worker(
        worker_id=good_worker,
        capability=WorkerCapability.COGNITION,
        threshold=threshold(max_failures=1),
    )

    fail(runtime, bad_worker)
    blocked = runtime.allow_request(bad_worker)
    allowed = runtime.allow_request(good_worker)

    assert blocked.success is False
    assert allowed.success is True


def test_snapshot_counts_breaker_state() -> None:
    runtime, worker_id = registered_runtime(max_failures=1)

    fail(runtime, worker_id)
    runtime.fallback_for(worker_id)
    snapshot = runtime.snapshot()

    assert snapshot.breaker_count == 1
    assert snapshot.open_count == 1
    assert snapshot.failure_count == 1
    assert snapshot.fallback_count == 1


def test_reset_clears_runtime_state() -> None:
    runtime, worker_id = registered_runtime(max_failures=1)

    fail(runtime, worker_id)
    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.breaker_count == 0
    assert snapshot.failure_count == 0


def test_enum_values_are_stable() -> None:
    assert BreakerState.CLOSED.value == "closed"
    assert CircuitBreakerDecision.OPENED.value == "opened"
    assert CircuitBreakerReason.FAILURE_THRESHOLD_REACHED.value
    assert WorkerFailureKind.TIMEOUT.value == "timeout"
    assert FallbackMode.SAFE_MODE_ONLY.value == "safe_mode_only"