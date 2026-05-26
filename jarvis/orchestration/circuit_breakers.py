from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from threading import RLock

from pydantic import Field, field_validator, model_validator

from jarvis.orchestration.ids import WorkerId, utc_now, validate_worker_id
from jarvis.orchestration.models import OrchestrationModel, WorkerCapability


class BreakerState(StrEnum):
    """
    Circuit breaker lifecycle.

    CLOSED: normal traffic allowed.
    OPEN: worker isolated, fallback used.
    HALF_OPEN: limited recovery probe allowed.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerDecision(StrEnum):
    """
    Decision returned by the circuit breaker runtime.
    """

    ALLOWED = "allowed"
    BLOCKED = "blocked"
    OPENED = "opened"
    HALF_OPENED = "half_opened"
    RECOVERED = "recovered"
    RECORDED = "recorded"
    FALLBACK = "fallback"
    REJECTED = "rejected"


class CircuitBreakerReason(StrEnum):
    """
    Machine-readable circuit breaker reason.
    """

    WORKER_REGISTERED = "worker_registered"
    WORKER_NOT_FOUND = "worker_not_found"
    WORKER_ALLOWED = "worker_allowed"
    WORKER_ISOLATED_OPEN = "worker_isolated_open"
    WORKER_PROBE_ALLOWED = "worker_probe_allowed"
    WORKER_PROBE_NOT_READY = "worker_probe_not_ready"
    FAILURE_RECORDED = "failure_recorded"
    FAILURE_THRESHOLD_REACHED = "failure_threshold_reached"
    SUCCESS_RECORDED = "success_recorded"
    HALF_OPEN_SUCCESS_RECOVERED = "half_open_success_recovered"
    HALF_OPEN_FAILURE_REOPENED = "half_open_failure_reopened"
    FALLBACK_SELECTED = "fallback_selected"
    RUNTIME_RESET = "runtime_reset"


class WorkerFailureKind(StrEnum):
    """
    Failure classes that can trip a worker circuit breaker.
    """

    TIMEOUT = "timeout"
    EXCEPTION = "exception"
    UNHEALTHY = "unhealthy"
    CANCELLED = "cancelled"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    UNKNOWN = "unknown"


class FallbackMode(StrEnum):
    """
    Safe degradation behavior per worker class.
    """

    NONE = "none"
    SIMPLIFIED_RESPONSE = "simplified_response"
    STATELESS_TURN = "stateless_turn"
    SAFE_MODE_ONLY = "safe_mode_only"
    DISABLED_WORKER = "disabled_worker"


_DEFAULT_FALLBACKS: dict[WorkerCapability, FallbackMode] = {
    WorkerCapability.COGNITION: FallbackMode.SIMPLIFIED_RESPONSE,
    WorkerCapability.CONVERSATION: FallbackMode.SIMPLIFIED_RESPONSE,
    WorkerCapability.MEMORY: FallbackMode.STATELESS_TURN,
    WorkerCapability.TOOL_ACTION: FallbackMode.SAFE_MODE_ONLY,
    WorkerCapability.BACKGROUND: FallbackMode.DISABLED_WORKER,
    WorkerCapability.PRESENCE: FallbackMode.SIMPLIFIED_RESPONSE,
    WorkerCapability.OBSERVABILITY: FallbackMode.NONE,
    WorkerCapability.RECOVERY: FallbackMode.NONE,
    WorkerCapability.ATTENTION: FallbackMode.NONE,
    WorkerCapability.SCHEDULING: FallbackMode.NONE,
}


class FailureThreshold(OrchestrationModel):
    """
    Failure threshold for a worker breaker.

    A breaker opens when failure_count reaches max_failures.
    Recovery probes use exponential backoff beginning at initial_probe_delay_ms.
    """

    max_failures: int = Field(default=3, gt=0)
    initial_probe_delay_ms: int = Field(default=5_000, gt=0)
    max_probe_delay_ms: int = Field(default=60_000, gt=0)

    @model_validator(mode="after")
    def _validate_threshold(self) -> FailureThreshold:
        if self.initial_probe_delay_ms > self.max_probe_delay_ms:
            raise ValueError(
                "initial probe delay cannot exceed max probe delay."
            )

        return self

    def probe_delay_for_attempt(self, attempt: int) -> int:
        """
        Return exponential probe delay for attempt.
        """

        initial_delay: int = self.initial_probe_delay_ms
        max_delay: int = self.max_probe_delay_ms

        if attempt <= 0:
            return initial_delay

        delay: int = initial_delay * (2**attempt)

        if delay > max_delay:
            return max_delay

        return delay


class RecoveryProbe(OrchestrationModel):
    """
    Recovery probe window for an OPEN worker.
    """

    worker_id: WorkerId
    attempt: int = Field(default=0, ge=0)
    allowed_at: object
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("worker_id")
    @classmethod
    def _validate_worker_id(cls, value: str) -> str:
        return validate_worker_id(value)

    def ready(self, now: object | None = None) -> bool:
        """
        Return whether the probe may run.
        """

        final_now = now or utc_now()

        if not isinstance(final_now, datetime):
            return False

        if not isinstance(self.allowed_at, datetime):
            return False

        return final_now >= self.allowed_at


class WorkerFailureRecord(OrchestrationModel):
    """
    Observable worker failure event.
    """

    worker_id: WorkerId
    failure_kind: WorkerFailureKind
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("worker_id")
    @classmethod
    def _validate_worker_id(cls, value: str) -> str:
        return validate_worker_id(value)

    @field_validator("message")
    @classmethod
    def _required_message(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("message cannot be empty.")

        return cleaned


class CircuitBreaker(OrchestrationModel):
    """
    Per-worker circuit breaker state.
    """

    worker_id: WorkerId
    capability: WorkerCapability
    state: BreakerState = BreakerState.CLOSED
    threshold: FailureThreshold = Field(default_factory=FailureThreshold)
    fallback_mode: FallbackMode
    failure_count: int = Field(default=0, ge=0)
    success_count: int = Field(default=0, ge=0)
    probe_attempt: int = Field(default=0, ge=0)
    opened_at: object | None = None
    half_opened_at: object | None = None
    next_probe: RecoveryProbe | None = None
    failures: tuple[WorkerFailureRecord, ...] = ()
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("worker_id")
    @classmethod
    def _validate_worker_id(cls, value: str) -> str:
        return validate_worker_id(value)

    @model_validator(mode="after")
    def _validate_shape(self) -> CircuitBreaker:
        if self.state == BreakerState.OPEN and self.opened_at is None:
            raise ValueError("open breaker requires opened_at.")

        if self.state == BreakerState.OPEN and self.next_probe is None:
            raise ValueError("open breaker requires next_probe.")

        if self.state == BreakerState.HALF_OPEN:
            if self.half_opened_at is None:
                raise ValueError("half-open breaker requires half_opened_at.")

        return self

    @property
    def isolated(self) -> bool:
        return self.state == BreakerState.OPEN

    @property
    def allows_normal_traffic(self) -> bool:
        return self.state == BreakerState.CLOSED

    def open_with_failure(
        self,
        failure: WorkerFailureRecord,
    ) -> CircuitBreaker:
        """
        Return OPEN breaker with next recovery probe configured.
        """

        now = utc_now()
        delay_ms = self.threshold.probe_delay_for_attempt(self.probe_attempt)
        allowed_at = (
            now + timedelta(milliseconds=delay_ms)
            if isinstance(now, datetime)
            else now
        )

        return self.model_copy(
            update={
                "state": BreakerState.OPEN,
                "failure_count": self.failure_count + 1,
                "probe_attempt": self.probe_attempt + 1,
                "opened_at": now,
                "half_opened_at": None,
                "next_probe": RecoveryProbe(
                    worker_id=self.worker_id,
                    attempt=self.probe_attempt,
                    allowed_at=allowed_at,
                ),
                "failures": self.failures + (failure,),
                "updated_at": now,
            }
        )

    def record_failure(
        self,
        failure: WorkerFailureRecord,
    ) -> CircuitBreaker:
        """
        Return breaker after one failure.
        """

        if self.state == BreakerState.HALF_OPEN:
            return self.open_with_failure(failure)

        next_failure_count = self.failure_count + 1

        if next_failure_count >= self.threshold.max_failures:
            return self.open_with_failure(failure)

        return self.model_copy(
            update={
                "failure_count": next_failure_count,
                "failures": self.failures + (failure,),
                "updated_at": utc_now(),
            }
        )

    def half_open(self) -> CircuitBreaker:
        """
        Return HALF_OPEN breaker for one recovery probe.
        """

        return self.model_copy(
            update={
                "state": BreakerState.HALF_OPEN,
                "half_opened_at": utc_now(),
                "updated_at": utc_now(),
            }
        )

    def close_after_success(self) -> CircuitBreaker:
        """
        Return CLOSED breaker after recovery.
        """

        return self.model_copy(
            update={
                "state": BreakerState.CLOSED,
                "failure_count": 0,
                "success_count": self.success_count + 1,
                "probe_attempt": 0,
                "opened_at": None,
                "half_opened_at": None,
                "next_probe": None,
                "updated_at": utc_now(),
            }
        )

    def record_success(self) -> CircuitBreaker:
        """
        Return breaker after successful operation.
        """

        if self.state == BreakerState.HALF_OPEN:
            return self.close_after_success()

        return self.model_copy(
            update={
                "success_count": self.success_count + 1,
                "updated_at": utc_now(),
            }
        )


class CircuitBreakerResult(OrchestrationModel):
    """
    Result of circuit breaker operation.
    """

    decision: CircuitBreakerDecision
    reason: CircuitBreakerReason
    success: bool
    message: str
    worker_id: WorkerId | None = None
    breaker: CircuitBreaker | None = None
    fallback_mode: FallbackMode | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("worker_id")
    @classmethod
    def _validate_worker_id(cls, value: str | None) -> str | None:
        if value is None:
            return None

        return validate_worker_id(value)

    @field_validator("message")
    @classmethod
    def _required_message(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("message cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class CircuitBreakerRuntimeConfig:
    """
    Circuit breaker runtime configuration.
    """

    name: str = "circuit_breaker_runtime"

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")


@dataclass(frozen=True, slots=True)
class CircuitBreakerRuntimeSnapshot:
    """
    Circuit breaker runtime diagnostics.
    """

    name: str
    breaker_count: int
    open_count: int
    half_open_count: int
    closed_count: int
    failure_count: int
    fallback_count: int
    rejected_count: int
    last_reason: CircuitBreakerReason | None


class CircuitBreakerRuntime:
    """
    Phase 6 Circuit Breaker Runtime.

    Responsibilities:
    - isolate failing workers
    - prevent cascade failures
    - select safe fallback behavior
    - allow recovery probes with exponential backoff
    - close breakers after successful half-open probe

    Non-responsibilities:
    - no task execution
    - no direct worker restart
    - no direct scheduler mutation
    - no hidden failure swallowing
    """

    def __init__(
        self,
        *,
        config: CircuitBreakerRuntimeConfig | None = None,
    ) -> None:
        self._config = config or CircuitBreakerRuntimeConfig()
        self._config.validate()

        self._breakers: dict[WorkerId, CircuitBreaker] = {}
        self._lock = RLock()

        self._failure_count = 0
        self._fallback_count = 0
        self._rejected_count = 0
        self._last_reason: CircuitBreakerReason | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def register_worker(
        self,
        *,
        worker_id: WorkerId,
        capability: WorkerCapability,
        threshold: FailureThreshold | None = None,
        fallback_mode: FallbackMode | None = None,
    ) -> CircuitBreakerResult:
        """
        Register a per-worker circuit breaker.
        """

        validated_worker_id = validate_worker_id(worker_id)
        final_fallback = fallback_mode or _DEFAULT_FALLBACKS[capability]
        breaker = CircuitBreaker(
            worker_id=validated_worker_id,
            capability=capability,
            threshold=threshold or FailureThreshold(),
            fallback_mode=final_fallback,
        )

        with self._lock:
            self._breakers[validated_worker_id] = breaker

        result = CircuitBreakerResult(
            decision=CircuitBreakerDecision.RECORDED,
            reason=CircuitBreakerReason.WORKER_REGISTERED,
            success=True,
            message="worker circuit breaker registered",
            worker_id=validated_worker_id,
            breaker=breaker,
        )
        self._record(result)

        return result

    def allow_request(
        self,
        worker_id: WorkerId,
        *,
        now: object | None = None,
    ) -> CircuitBreakerResult:
        """
        Return whether traffic may go to worker.
        """

        breaker = self.breaker_for(worker_id)

        if breaker is None:
            result = CircuitBreakerResult(
                decision=CircuitBreakerDecision.REJECTED,
                reason=CircuitBreakerReason.WORKER_NOT_FOUND,
                success=False,
                message="worker circuit breaker not found",
                worker_id=worker_id,
            )
            self._record(result)

            return result

        if breaker.state == BreakerState.CLOSED:
            result = CircuitBreakerResult(
                decision=CircuitBreakerDecision.ALLOWED,
                reason=CircuitBreakerReason.WORKER_ALLOWED,
                success=True,
                message="worker traffic allowed",
                worker_id=breaker.worker_id,
                breaker=breaker,
            )
            self._record(result)

            return result

        if breaker.state == BreakerState.HALF_OPEN:
            result = CircuitBreakerResult(
                decision=CircuitBreakerDecision.ALLOWED,
                reason=CircuitBreakerReason.WORKER_PROBE_ALLOWED,
                success=True,
                message="half-open worker probe allowed",
                worker_id=breaker.worker_id,
                breaker=breaker,
            )
            self._record(result)

            return result

        if breaker.next_probe is not None and breaker.next_probe.ready(now):
            half_open = breaker.half_open()
            self._store(half_open)

            result = CircuitBreakerResult(
                decision=CircuitBreakerDecision.HALF_OPENED,
                reason=CircuitBreakerReason.WORKER_PROBE_ALLOWED,
                success=True,
                message="worker recovery probe allowed",
                worker_id=half_open.worker_id,
                breaker=half_open,
            )
            self._record(result)

            return result

        result = CircuitBreakerResult(
            decision=CircuitBreakerDecision.BLOCKED,
            reason=CircuitBreakerReason.WORKER_ISOLATED_OPEN,
            success=False,
            message="worker isolated by open circuit breaker",
            worker_id=breaker.worker_id,
            breaker=breaker,
            fallback_mode=breaker.fallback_mode,
        )
        self._record(result)

        return result

    def record_failure(
        self,
        *,
        worker_id: WorkerId,
        failure_kind: WorkerFailureKind,
        message: str,
        metadata: dict[str, object] | None = None,
    ) -> CircuitBreakerResult:
        """
        Record worker failure and open breaker if threshold is reached.
        """

        breaker = self.breaker_for(worker_id)

        if breaker is None:
            result = CircuitBreakerResult(
                decision=CircuitBreakerDecision.REJECTED,
                reason=CircuitBreakerReason.WORKER_NOT_FOUND,
                success=False,
                message="worker circuit breaker not found",
                worker_id=worker_id,
            )
            self._record(result)

            return result

        failure = WorkerFailureRecord(
            worker_id=breaker.worker_id,
            failure_kind=failure_kind,
            message=message,
            metadata=metadata or {},
        )
        updated = breaker.record_failure(failure)
        self._store(updated)
        self._failure_count += 1

        if updated.state == BreakerState.OPEN:
            reason = (
                CircuitBreakerReason.HALF_OPEN_FAILURE_REOPENED
                if breaker.state == BreakerState.HALF_OPEN
                else CircuitBreakerReason.FAILURE_THRESHOLD_REACHED
            )
            result = CircuitBreakerResult(
                decision=CircuitBreakerDecision.OPENED,
                reason=reason,
                success=False,
                message="worker circuit breaker opened",
                worker_id=updated.worker_id,
                breaker=updated,
                fallback_mode=updated.fallback_mode,
            )
            self._record(result)

            return result

        result = CircuitBreakerResult(
            decision=CircuitBreakerDecision.RECORDED,
            reason=CircuitBreakerReason.FAILURE_RECORDED,
            success=True,
            message="worker failure recorded",
            worker_id=updated.worker_id,
            breaker=updated,
        )
        self._record(result)

        return result

    def record_success(self, worker_id: WorkerId) -> CircuitBreakerResult:
        """
        Record worker success.
        """

        breaker = self.breaker_for(worker_id)

        if breaker is None:
            result = CircuitBreakerResult(
                decision=CircuitBreakerDecision.REJECTED,
                reason=CircuitBreakerReason.WORKER_NOT_FOUND,
                success=False,
                message="worker circuit breaker not found",
                worker_id=worker_id,
            )
            self._record(result)

            return result

        updated = breaker.record_success()
        self._store(updated)

        if breaker.state == BreakerState.HALF_OPEN:
            result = CircuitBreakerResult(
                decision=CircuitBreakerDecision.RECOVERED,
                reason=CircuitBreakerReason.HALF_OPEN_SUCCESS_RECOVERED,
                success=True,
                message="worker circuit breaker recovered",
                worker_id=updated.worker_id,
                breaker=updated,
            )
            self._record(result)

            return result

        result = CircuitBreakerResult(
            decision=CircuitBreakerDecision.RECORDED,
            reason=CircuitBreakerReason.SUCCESS_RECORDED,
            success=True,
            message="worker success recorded",
            worker_id=updated.worker_id,
            breaker=updated,
        )
        self._record(result)

        return result

    def fallback_for(self, worker_id: WorkerId) -> CircuitBreakerResult:
        """
        Return fallback mode for isolated worker.
        """

        breaker = self.breaker_for(worker_id)

        if breaker is None:
            result = CircuitBreakerResult(
                decision=CircuitBreakerDecision.REJECTED,
                reason=CircuitBreakerReason.WORKER_NOT_FOUND,
                success=False,
                message="worker circuit breaker not found",
                worker_id=worker_id,
            )
            self._record(result)

            return result

        result = CircuitBreakerResult(
            decision=CircuitBreakerDecision.FALLBACK,
            reason=CircuitBreakerReason.FALLBACK_SELECTED,
            success=True,
            message="fallback selected for worker",
            worker_id=breaker.worker_id,
            breaker=breaker,
            fallback_mode=breaker.fallback_mode,
        )
        self._fallback_count += 1
        self._record(result)

        return result

    def breaker_for(self, worker_id: WorkerId) -> CircuitBreaker | None:
        validated_worker_id = validate_worker_id(worker_id)

        with self._lock:
            return self._breakers.get(validated_worker_id)

    def all_breakers(self) -> tuple[CircuitBreaker, ...]:
        with self._lock:
            return tuple(self._breakers.values())

    def snapshot(self) -> CircuitBreakerRuntimeSnapshot:
        with self._lock:
            breakers = tuple(self._breakers.values())

            return CircuitBreakerRuntimeSnapshot(
                name=self.name,
                breaker_count=len(breakers),
                open_count=sum(
                    1
                    for breaker in breakers
                    if breaker.state == BreakerState.OPEN
                ),
                half_open_count=sum(
                    1
                    for breaker in breakers
                    if breaker.state == BreakerState.HALF_OPEN
                ),
                closed_count=sum(
                    1
                    for breaker in breakers
                    if breaker.state == BreakerState.CLOSED
                ),
                failure_count=self._failure_count,
                fallback_count=self._fallback_count,
                rejected_count=self._rejected_count,
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        with self._lock:
            self._breakers.clear()
            self._failure_count = 0
            self._fallback_count = 0
            self._rejected_count = 0
            self._last_reason = None

    def _store(self, breaker: CircuitBreaker) -> None:
        with self._lock:
            self._breakers[breaker.worker_id] = breaker

    def _record(self, result: CircuitBreakerResult) -> None:
        self._last_reason = result.reason

        if result.decision == CircuitBreakerDecision.REJECTED:
            self._rejected_count += 1