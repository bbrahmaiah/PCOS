from __future__ import annotations

import signal
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from jarvis.voice.contracts import utc_now
from jarvis.voice.daily_driver_gate import (
    VoiceDailyDriverGate,
    VoiceDailyDriverGateReport,
    VoiceDailyDriverGateStatus,
)
from jarvis.voice.session_loop import (
    VoiceSessionLoopResult,
    VoiceSessionLoopRuntime,
    VoiceSessionLoopSnapshot,
    VoiceSessionLoopStatus,
)


class VoiceRuntimeLauncherStatus(StrEnum):
    CREATED = "created"
    BOOTING = "booting"
    READY = "ready"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    DEGRADED = "degraded"
    FAILED = "failed"


class VoiceRuntimeLauncherOperation(StrEnum):
    BOOT = "boot"
    RUN = "run"
    STOP = "stop"
    SNAPSHOT = "snapshot"


class VoiceRuntimeLauncherEvent(StrEnum):
    BOOT_STARTED = "boot_started"
    DAILY_DRIVER_GATE_PASSED = "daily_driver_gate_passed"
    DAILY_DRIVER_GATE_DEGRADED = "daily_driver_gate_degraded"
    DAILY_DRIVER_GATE_FAILED = "daily_driver_gate_failed"
    SESSION_STARTED = "session_started"
    SESSION_RUNNING = "session_running"
    SESSION_STOPPED = "session_stopped"
    SHUTDOWN_REQUESTED = "shutdown_requested"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class VoiceRuntimeLauncherConfig:
    run_forever: bool = True
    bounded_cycles: int | None = None
    bounded_seconds: float | None = None
    run_daily_driver_gate: bool = True
    allow_degraded_gate: bool = False
    idle_sleep_seconds: float = 0.02
    stop_on_session_failure: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.bounded_cycles is not None and self.bounded_cycles < 1:
            raise ValueError("bounded_cycles must be positive when provided.")
        if self.bounded_seconds is not None and self.bounded_seconds <= 0:
            raise ValueError("bounded_seconds must be positive when provided.")
        if self.idle_sleep_seconds < 0:
            raise ValueError("idle_sleep_seconds cannot be negative.")


@dataclass(frozen=True, slots=True)
class VoiceRuntimeLauncherResult:
    status: VoiceRuntimeLauncherStatus
    operation: VoiceRuntimeLauncherOperation
    event: VoiceRuntimeLauncherEvent | None
    session_result: VoiceSessionLoopResult | None
    gate_report: VoiceDailyDriverGateReport | None
    reason: str
    latency_ms: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status in {
            VoiceRuntimeLauncherStatus.READY,
            VoiceRuntimeLauncherStatus.RUNNING,
            VoiceRuntimeLauncherStatus.STOPPED,
            VoiceRuntimeLauncherStatus.DEGRADED,
        }


@dataclass(frozen=True, slots=True)
class VoiceRuntimeLauncherSnapshot:
    status: VoiceRuntimeLauncherStatus
    booted: bool
    running: bool
    stop_requested: bool
    boot_count: int
    run_cycles: int
    stop_count: int
    last_event: VoiceRuntimeLauncherEvent | None
    last_error: str | None
    last_latency_ms: float | None
    session_snapshot: VoiceSessionLoopSnapshot | None
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


class VoiceLauncherSessionLoop(Protocol):
    def start(self) -> VoiceSessionLoopResult:
        raise NotImplementedError

    def run(
        self,
        *,
        max_cycles: int | None = None,
        max_seconds: float | None = None,
    ) -> VoiceSessionLoopResult:
        raise NotImplementedError

    def stop(self) -> VoiceSessionLoopResult:
        raise NotImplementedError

    def snapshot(self) -> VoiceSessionLoopSnapshot:
        raise NotImplementedError


class VoiceLauncherDailyDriverGate(Protocol):
    def run(self) -> VoiceDailyDriverGateReport:
        raise NotImplementedError


class VoiceRuntimeLauncher:
    """
    Step 51L real voice launcher.

    This launcher starts the actual voice runtime path. It is not text mode.
    It does not generate final user-facing speech. It only boots, validates,
    supervises, and stops the real voice session loop.
    """

    def __init__(
        self,
        *,
        session_loop: VoiceLauncherSessionLoop | None = None,
        daily_driver_gate: VoiceLauncherDailyDriverGate | None = None,
        config: VoiceRuntimeLauncherConfig | None = None,
    ) -> None:
        self._config = config or VoiceRuntimeLauncherConfig()
        self._session_loop = session_loop or VoiceSessionLoopRuntime()
        self._daily_driver_gate = daily_driver_gate or VoiceDailyDriverGate(
            session_loop=self._session_loop
        )
        self._status = VoiceRuntimeLauncherStatus.CREATED
        self._booted = False
        self._running = False
        self._stop_requested = False
        self._boot_count = 0
        self._run_cycles = 0
        self._stop_count = 0
        self._last_event: VoiceRuntimeLauncherEvent | None = None
        self._last_error: str | None = None
        self._last_latency_ms: float | None = None

    def boot(self) -> VoiceRuntimeLauncherResult:
        started = time.perf_counter()
        self._status = VoiceRuntimeLauncherStatus.BOOTING
        self._last_event = VoiceRuntimeLauncherEvent.BOOT_STARTED

        gate_report: VoiceDailyDriverGateReport | None = None

        if self._config.run_daily_driver_gate:
            try:
                gate_report = self._daily_driver_gate.run()
            except Exception as exc:
                self._status = VoiceRuntimeLauncherStatus.FAILED
                self._last_error = str(exc)
                return self._result(
                    operation=VoiceRuntimeLauncherOperation.BOOT,
                    event=VoiceRuntimeLauncherEvent.ERROR,
                    session_result=None,
                    gate_report=None,
                    reason="daily driver gate raised during boot",
                    started=started,
                    metadata={"error": str(exc)},
                )

            if gate_report.status == VoiceDailyDriverGateStatus.FAILED:
                self._status = VoiceRuntimeLauncherStatus.FAILED
                return self._result(
                    operation=VoiceRuntimeLauncherOperation.BOOT,
                    event=VoiceRuntimeLauncherEvent.DAILY_DRIVER_GATE_FAILED,
                    session_result=None,
                    gate_report=gate_report,
                    reason="daily driver gate failed",
                    started=started,
                )

            if gate_report.status == VoiceDailyDriverGateStatus.DEGRADED:
                if not self._config.allow_degraded_gate:
                    self._status = VoiceRuntimeLauncherStatus.FAILED
                    return self._result(
                        operation=VoiceRuntimeLauncherOperation.BOOT,
                        event=VoiceRuntimeLauncherEvent.DAILY_DRIVER_GATE_DEGRADED,
                        session_result=None,
                        gate_report=gate_report,
                        reason="daily driver gate degraded and policy blocked boot",
                        started=started,
                    )

                self._status = VoiceRuntimeLauncherStatus.DEGRADED
                self._last_event = VoiceRuntimeLauncherEvent.DAILY_DRIVER_GATE_DEGRADED
            else:
                self._last_event = VoiceRuntimeLauncherEvent.DAILY_DRIVER_GATE_PASSED

        self._booted = True
        self._boot_count += 1
        self._status = (
            VoiceRuntimeLauncherStatus.DEGRADED
            if self._status == VoiceRuntimeLauncherStatus.DEGRADED
            else VoiceRuntimeLauncherStatus.READY
        )

        return self._result(
            operation=VoiceRuntimeLauncherOperation.BOOT,
            event=self._last_event,
            session_result=None,
            gate_report=gate_report,
            reason="voice launcher boot completed",
            started=started,
        )

    def run(self) -> VoiceRuntimeLauncherResult:
        started = time.perf_counter()

        if not self._booted:
            boot_result = self.boot()
            if boot_result.status == VoiceRuntimeLauncherStatus.FAILED:
                return boot_result

        try:
            start_result = self._session_loop.start()
        except Exception as exc:
            self._status = VoiceRuntimeLauncherStatus.FAILED
            self._last_error = str(exc)
            return self._result(
                operation=VoiceRuntimeLauncherOperation.RUN,
                event=VoiceRuntimeLauncherEvent.ERROR,
                session_result=None,
                gate_report=None,
                reason="voice session start raised",
                started=started,
                metadata={"error": str(exc)},
            )

        if start_result.status == VoiceSessionLoopStatus.FAILED:
            self._status = VoiceRuntimeLauncherStatus.FAILED
            return self._result(
                operation=VoiceRuntimeLauncherOperation.RUN,
                event=VoiceRuntimeLauncherEvent.ERROR,
                session_result=start_result,
                gate_report=None,
                reason="voice session failed to start",
                started=started,
            )

        self._running = True
        self._stop_requested = False
        self._status = VoiceRuntimeLauncherStatus.RUNNING
        self._last_event = VoiceRuntimeLauncherEvent.SESSION_STARTED

        final_session_result = start_result

        try:
            if self._config.run_forever:
                final_session_result = self._run_forever()
            else:
                final_session_result = self._session_loop.run(
                    max_cycles=self._config.bounded_cycles,
                    max_seconds=self._config.bounded_seconds,
                )
        except KeyboardInterrupt:
            self.request_stop()
            final_session_result = self._session_loop.stop()
        except Exception as exc:
            self._status = VoiceRuntimeLauncherStatus.FAILED
            self._last_error = str(exc)
            return self._result(
                operation=VoiceRuntimeLauncherOperation.RUN,
                event=VoiceRuntimeLauncherEvent.ERROR,
                session_result=final_session_result,
                gate_report=None,
                reason="voice launcher run raised",
                started=started,
                metadata={"error": str(exc)},
            )

        if (
            final_session_result.status == VoiceSessionLoopStatus.FAILED
            and self._config.stop_on_session_failure
        ):
            self._running = False
            self._status = VoiceRuntimeLauncherStatus.FAILED
            return self._result(
                operation=VoiceRuntimeLauncherOperation.RUN,
                event=VoiceRuntimeLauncherEvent.ERROR,
                session_result=final_session_result,
                gate_report=None,
                reason="voice session failed during run",
                started=started,
            )

        self._status = (
            VoiceRuntimeLauncherStatus.STOPPED
            if final_session_result.status == VoiceSessionLoopStatus.STOPPED
            else VoiceRuntimeLauncherStatus.RUNNING
        )

        return self._result(
            operation=VoiceRuntimeLauncherOperation.RUN,
            event=VoiceRuntimeLauncherEvent.SESSION_RUNNING,
            session_result=final_session_result,
            gate_report=None,
            reason="voice launcher run completed",
            started=started,
        )

    def request_stop(self) -> None:
        self._stop_requested = True
        self._last_event = VoiceRuntimeLauncherEvent.SHUTDOWN_REQUESTED

    def stop(self) -> VoiceRuntimeLauncherResult:
        started = time.perf_counter()
        self._status = VoiceRuntimeLauncherStatus.STOPPING
        self._stop_requested = True

        try:
            session_result = self._session_loop.stop()
        except Exception as exc:
            self._status = VoiceRuntimeLauncherStatus.FAILED
            self._last_error = str(exc)
            return self._result(
                operation=VoiceRuntimeLauncherOperation.STOP,
                event=VoiceRuntimeLauncherEvent.ERROR,
                session_result=None,
                gate_report=None,
                reason="voice session stop raised",
                started=started,
                metadata={"error": str(exc)},
            )

        self._running = False
        self._stop_count += 1
        self._status = VoiceRuntimeLauncherStatus.STOPPED
        return self._result(
            operation=VoiceRuntimeLauncherOperation.STOP,
            event=VoiceRuntimeLauncherEvent.SESSION_STOPPED,
            session_result=session_result,
            gate_report=None,
            reason="voice launcher stopped",
            started=started,
        )

    def install_signal_handlers(self) -> None:
        def _handler(signum: int, frame: object) -> None:
            del signum, frame
            self.request_stop()

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def snapshot(self) -> VoiceRuntimeLauncherSnapshot:
        session_snapshot: VoiceSessionLoopSnapshot | None
        try:
            session_snapshot = self._session_loop.snapshot()
        except Exception:
            session_snapshot = None

        return VoiceRuntimeLauncherSnapshot(
            status=self._status,
            booted=self._booted,
            running=self._running,
            stop_requested=self._stop_requested,
            boot_count=self._boot_count,
            run_cycles=self._run_cycles,
            stop_count=self._stop_count,
            last_event=self._last_event,
            last_error=self._last_error,
            last_latency_ms=self._last_latency_ms,
            session_snapshot=session_snapshot,
            created_at=utc_now(),
            metadata=self._config.metadata,
        )

    def _run_forever(self) -> VoiceSessionLoopResult:
        result = self._session_loop.run(max_cycles=1)

        while not self._stop_requested:
            result = self._session_loop.run(max_cycles=1)
            self._run_cycles += 1

            if (
                result.status == VoiceSessionLoopStatus.FAILED
                and self._config.stop_on_session_failure
            ):
                break

            if self._config.idle_sleep_seconds > 0:
                time.sleep(self._config.idle_sleep_seconds)

        self._running = False
        return self._session_loop.stop()

    def _result(
        self,
        *,
        operation: VoiceRuntimeLauncherOperation,
        event: VoiceRuntimeLauncherEvent | None,
        session_result: VoiceSessionLoopResult | None,
        gate_report: VoiceDailyDriverGateReport | None,
        reason: str,
        started: float,
        metadata: dict[str, object] | None = None,
    ) -> VoiceRuntimeLauncherResult:
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._last_latency_ms = latency_ms

        if event is not None:
            self._last_event = event

        return VoiceRuntimeLauncherResult(
            status=self._status,
            operation=operation,
            event=event,
            session_result=session_result,
            gate_report=gate_report,
            reason=reason,
            latency_ms=latency_ms,
            created_at=utc_now(),
            metadata=metadata or {},
        )