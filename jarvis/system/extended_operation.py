from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from time import perf_counter, sleep

from jarvis.conversation.models import ConversationMode, TurnInputSource
from jarvis.conversation.runtime import RealConversationInput
from jarvis.system.bootstrap import JarvisBootstrapStatus
from jarvis.system.contracts import (
    JarvisPipelineStatus,
    JarvisSystemStatus,
    utc_now,
)
from jarvis.system.live_wiring import LiveDependencyWiring


class ExtendedOperationStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


class ExtendedOperationRunMode(StrEnum):
    SMOKE = "smoke"
    THIRTY_MINUTES = "thirty_minutes"
    TWO_HOURS = "two_hours"
    SIX_HOURS = "six_hours"
    TWENTY_FOUR_HOURS = "twenty_four_hours"
    CUSTOM = "custom"


class ExtendedOperationEventKind(StrEnum):
    BOOT = "boot"
    CYCLE_COMPLETED = "cycle_completed"
    INTERRUPTION_COMPLETED = "interruption_completed"
    MEMORY_WRITE_COMPLETED = "memory_write_completed"
    FAILURE_RECORDED = "failure_recorded"
    SHUTDOWN = "shutdown"


@dataclass(frozen=True, slots=True)
class ExtendedOperationConfig:
    run_mode: ExtendedOperationRunMode = ExtendedOperationRunMode.SMOKE
    session_id: str = "extended_operation"
    cycle_count: int = 5
    cycle_delay_seconds: float = 0.0
    max_failure_count: int = 0
    max_average_latency_ms: float = 2_000.0
    require_clean_shutdown: bool = True
    prompt: str = "What am I building?"
    memory_prompt: str = "Remember that extended operation validation passed."
    interruption_prompt: str = "stop"
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("extended operation session_id cannot be empty.")
        if self.cycle_count < 1:
            raise ValueError("cycle_count must be at least 1.")
        if self.cycle_delay_seconds < 0:
            raise ValueError("cycle_delay_seconds cannot be negative.")
        if self.max_failure_count < 0:
            raise ValueError("max_failure_count cannot be negative.")
        if self.max_average_latency_ms <= 0:
            raise ValueError("max_average_latency_ms must be positive.")
        if not self.prompt.strip():
            raise ValueError("extended operation prompt cannot be empty.")
        if not self.memory_prompt.strip():
            raise ValueError("extended operation memory_prompt cannot be empty.")
        if not self.interruption_prompt.strip():
            raise ValueError(
                "extended operation interruption_prompt cannot be empty."
            )


@dataclass(frozen=True, slots=True)
class ExtendedOperationEvent:
    kind: ExtendedOperationEventKind
    message: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExtendedOperationMetrics:
    boot_latency_ms: float
    shutdown_latency_ms: float
    cycle_count: int
    completed_cycles: int
    failed_cycles: int
    interruption_count: int
    memory_write_count: int
    average_cycle_latency_ms: float
    max_cycle_latency_ms: float
    min_cycle_latency_ms: float
    total_runtime_ms: float
    worker_count: int
    fatal_error_count: int

    @property
    def all_cycles_completed(self) -> bool:
        return self.completed_cycles == self.cycle_count

    @property
    def no_fatal_errors(self) -> bool:
        return self.fatal_error_count == 0


@dataclass(frozen=True, slots=True)
class ExtendedOperationReport:
    status: ExtendedOperationStatus
    config: ExtendedOperationConfig
    metrics: ExtendedOperationMetrics | None
    events: tuple[ExtendedOperationEvent, ...]
    started_at: datetime
    finished_at: datetime
    error: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == ExtendedOperationStatus.PASSED


class ExtendedOperationValidator:
    """
    Step 46E extended operation validation.

    This validates repeated operation of the assembled JARVIS runtime:
    - boot
    - repeated pipeline cycles
    - interruption
    - governed memory write
    - latency measurements
    - worker visibility
    - failure count
    - clean shutdown

    Unit tests use short cycle counts. Real machine validation can run this
    with 30-minute, 2-hour, 6-hour, and 24-hour profiles.
    """

    def __init__(
        self,
        *,
        config: ExtendedOperationConfig,
        wiring: LiveDependencyWiring,
    ) -> None:
        self._config = config
        self._wiring = wiring

    def run(self) -> ExtendedOperationReport:
        started_at = utc_now()
        total_started = perf_counter()
        events: list[ExtendedOperationEvent] = []
        bootstrap = None
        boot_latency_ms = 0.0
        shutdown_latency_ms = 0.0
        worker_count = 0
        completed_cycles = 0
        failed_cycles = 0
        interruption_count = 0
        memory_write_count = 0
        fatal_error_count = 0
        cycle_latencies: list[float] = []

        try:
            bootstrap = self._wiring.build_bootstrap()

            boot_started = perf_counter()
            boot_result = bootstrap.start()
            boot_latency_ms = _elapsed_ms(boot_started)

            _record(
                events,
                kind=ExtendedOperationEventKind.BOOT,
                message="runtime boot attempted",
                metadata={
                    "bootstrap_status": boot_result.status.value,
                    "error": boot_result.error,
                    "boot_latency_ms": boot_latency_ms,
                },
            )

            if (
                boot_result.status != JarvisBootstrapStatus.STARTED
                or boot_result.error is not None
                or bootstrap.system is None
            ):
                raise RuntimeError(boot_result.error or "runtime boot failed")

            system = bootstrap.system
            snapshot = system.snapshot()
            worker_count = len(snapshot.subsystem_health)

            if snapshot.status != JarvisSystemStatus.RUNNING:
                raise RuntimeError(
                    f"system did not enter running state: "
                    f"{snapshot.status.value}"
                )

            for index in range(self._config.cycle_count):
                cycle_started = perf_counter()

                try:
                    result = system.process_user_utterance(
                        text=self._config.prompt,
                        session_id=self._config.session_id,
                    )
                    latency_ms = _elapsed_ms(cycle_started)
                    cycle_latencies.append(latency_ms)

                    if result.status == JarvisPipelineStatus.COMPLETED:
                        completed_cycles += 1
                        _record(
                            events,
                            kind=ExtendedOperationEventKind.CYCLE_COMPLETED,
                            message="operation cycle completed",
                            metadata={
                                "cycle_index": index,
                                "latency_ms": latency_ms,
                                "event_count": len(result.events),
                            },
                        )
                    else:
                        failed_cycles += 1
                        _record(
                            events,
                            kind=ExtendedOperationEventKind.FAILURE_RECORDED,
                            message="operation cycle failed",
                            metadata={
                                "cycle_index": index,
                                "pipeline_status": result.status.value,
                                "reason": result.reason,
                            },
                        )

                except Exception as exc:
                    failed_cycles += 1
                    fatal_error_count += 1
                    latency_ms = _elapsed_ms(cycle_started)
                    cycle_latencies.append(latency_ms)
                    _record(
                        events,
                        kind=ExtendedOperationEventKind.FAILURE_RECORDED,
                        message="operation cycle raised exception",
                        metadata={
                            "cycle_index": index,
                            "error": f"{type(exc).__name__}: {exc}",
                            "latency_ms": latency_ms,
                        },
                    )

                if self._config.cycle_delay_seconds > 0:
                    sleep(self._config.cycle_delay_seconds)

            interruption = system.process_user_utterance(
                text=self._config.interruption_prompt,
                session_id=self._config.session_id,
                signal=_interruption_signal(self._config.interruption_prompt),
            )
            if (
                interruption.status == JarvisPipelineStatus.CANCELLED
                and interruption.cancelled
            ):
                interruption_count += 1

            _record(
                events,
                kind=ExtendedOperationEventKind.INTERRUPTION_COMPLETED,
                message="interruption validation completed",
                metadata={
                    "pipeline_status": interruption.status.value,
                    "cancelled": interruption.cancelled,
                },
            )

            memory_result = system.process_user_utterance(
                text=self._config.memory_prompt,
                session_id=self._config.session_id,
            )
            if (
                memory_result.status == JarvisPipelineStatus.COMPLETED
                and memory_result.response is not None
                and memory_result.response.wrote_memory
            ):
                memory_write_count += 1

            _record(
                events,
                kind=ExtendedOperationEventKind.MEMORY_WRITE_COMPLETED,
                message="governed memory write validation completed",
                metadata={
                    "pipeline_status": memory_result.status.value,
                    "wrote_memory": (
                        memory_result.response.wrote_memory
                        if memory_result.response is not None
                        else False
                    ),
                },
            )

            shutdown_started = perf_counter()
            stop_result = bootstrap.stop()
            shutdown_latency_ms = _elapsed_ms(shutdown_started)

            clean_shutdown = (
                stop_result.status == JarvisBootstrapStatus.STOPPED
                and stop_result.error is None
                and stop_result.system_snapshot is not None
                and stop_result.system_snapshot.status
                == JarvisSystemStatus.STOPPED
            )

            _record(
                events,
                kind=ExtendedOperationEventKind.SHUTDOWN,
                message="runtime shutdown completed",
                metadata={
                    "bootstrap_status": stop_result.status.value,
                    "error": stop_result.error,
                    "shutdown_latency_ms": shutdown_latency_ms,
                    "clean_shutdown": clean_shutdown,
                },
            )

            metrics = _metrics(
                config=self._config,
                boot_latency_ms=boot_latency_ms,
                shutdown_latency_ms=shutdown_latency_ms,
                completed_cycles=completed_cycles,
                failed_cycles=failed_cycles,
                interruption_count=interruption_count,
                memory_write_count=memory_write_count,
                cycle_latencies=cycle_latencies,
                total_runtime_ms=_elapsed_ms(total_started),
                worker_count=worker_count,
                fatal_error_count=fatal_error_count,
            )

            passed = _passes(
                config=self._config,
                metrics=metrics,
                clean_shutdown=clean_shutdown,
            )

            return ExtendedOperationReport(
                status=(
                    ExtendedOperationStatus.PASSED
                    if passed
                    else ExtendedOperationStatus.FAILED
                ),
                config=self._config,
                metrics=metrics,
                events=tuple(events),
                started_at=started_at,
                finished_at=utc_now(),
                error=None if passed else "extended operation thresholds failed",
                metadata=self._config.metadata,
            )

        except Exception as exc:
            if bootstrap is not None:
                try:
                    shutdown_started = perf_counter()
                    stop_result = bootstrap.stop()
                    shutdown_latency_ms = _elapsed_ms(shutdown_started)
                    _record(
                        events,
                        kind=ExtendedOperationEventKind.SHUTDOWN,
                        message="shutdown attempted after validation failure",
                        metadata={
                            "bootstrap_status": stop_result.status.value,
                            "error": stop_result.error,
                            "shutdown_latency_ms": shutdown_latency_ms,
                        },
                    )
                except Exception as stop_exc:
                    _record(
                        events,
                        kind=ExtendedOperationEventKind.SHUTDOWN,
                        message="shutdown failed after validation failure",
                        metadata={
                            "error": f"{type(stop_exc).__name__}: {stop_exc}",
                        },
                    )

            return ExtendedOperationReport(
                status=ExtendedOperationStatus.FAILED,
                config=self._config,
                metrics=None,
                events=tuple(events),
                started_at=started_at,
                finished_at=utc_now(),
                error=f"{type(exc).__name__}: {exc}",
                metadata=self._config.metadata,
            )


def profile_config(
    run_mode: ExtendedOperationRunMode,
    *,
    session_id: str = "extended_operation",
) -> ExtendedOperationConfig:
    if run_mode == ExtendedOperationRunMode.SMOKE:
        return ExtendedOperationConfig(
            run_mode=run_mode,
            session_id=session_id,
            cycle_count=5,
            cycle_delay_seconds=0.0,
        )

    if run_mode == ExtendedOperationRunMode.THIRTY_MINUTES:
        return ExtendedOperationConfig(
            run_mode=run_mode,
            session_id=session_id,
            cycle_count=180,
            cycle_delay_seconds=10.0,
        )

    if run_mode == ExtendedOperationRunMode.TWO_HOURS:
        return ExtendedOperationConfig(
            run_mode=run_mode,
            session_id=session_id,
            cycle_count=720,
            cycle_delay_seconds=10.0,
        )

    if run_mode == ExtendedOperationRunMode.SIX_HOURS:
        return ExtendedOperationConfig(
            run_mode=run_mode,
            session_id=session_id,
            cycle_count=2160,
            cycle_delay_seconds=10.0,
        )

    if run_mode == ExtendedOperationRunMode.TWENTY_FOUR_HOURS:
        return ExtendedOperationConfig(
            run_mode=run_mode,
            session_id=session_id,
            cycle_count=8640,
            cycle_delay_seconds=10.0,
        )

    return ExtendedOperationConfig(
        run_mode=run_mode,
        session_id=session_id,
    )


def _record(
    events: list[ExtendedOperationEvent],
    *,
    kind: ExtendedOperationEventKind,
    message: str,
    metadata: dict[str, object] | None = None,
) -> None:
    events.append(
        ExtendedOperationEvent(
            kind=kind,
            message=message,
            created_at=utc_now(),
            metadata=metadata or {},
        )
    )


def _metrics(
    *,
    config: ExtendedOperationConfig,
    boot_latency_ms: float,
    shutdown_latency_ms: float,
    completed_cycles: int,
    failed_cycles: int,
    interruption_count: int,
    memory_write_count: int,
    cycle_latencies: list[float],
    total_runtime_ms: float,
    worker_count: int,
    fatal_error_count: int,
) -> ExtendedOperationMetrics:
    if cycle_latencies:
        average = sum(cycle_latencies) / len(cycle_latencies)
        maximum = max(cycle_latencies)
        minimum = min(cycle_latencies)
    else:
        average = 0.0
        maximum = 0.0
        minimum = 0.0

    return ExtendedOperationMetrics(
        boot_latency_ms=boot_latency_ms,
        shutdown_latency_ms=shutdown_latency_ms,
        cycle_count=config.cycle_count,
        completed_cycles=completed_cycles,
        failed_cycles=failed_cycles,
        interruption_count=interruption_count,
        memory_write_count=memory_write_count,
        average_cycle_latency_ms=average,
        max_cycle_latency_ms=maximum,
        min_cycle_latency_ms=minimum,
        total_runtime_ms=total_runtime_ms,
        worker_count=worker_count,
        fatal_error_count=fatal_error_count,
    )


def _passes(
    *,
    config: ExtendedOperationConfig,
    metrics: ExtendedOperationMetrics,
    clean_shutdown: bool,
) -> bool:
    if not metrics.all_cycles_completed:
        return False
    if metrics.failed_cycles > config.max_failure_count:
        return False
    if metrics.fatal_error_count > config.max_failure_count:
        return False
    if metrics.interruption_count < 1:
        return False
    if metrics.memory_write_count < 1:
        return False
    if metrics.average_cycle_latency_ms > config.max_average_latency_ms:
        return False
    if config.require_clean_shutdown and not clean_shutdown:
        return False

    return True


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000.0, 3)


def _interruption_signal(text: str) -> RealConversationInput:
    return RealConversationInput(
        transcript=text,
        source=TurnInputSource.INTERRUPTION_WORKER,
        is_speech_active=True,
        is_assistant_speaking=True,
        silence_ms=0,
        speech_ms=250,
        vad_confidence=0.99,
        transcript_stability=1.0,
        conversation_mode=ConversationMode.COMMAND,
    )