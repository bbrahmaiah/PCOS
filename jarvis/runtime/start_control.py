from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol


def utc_now() -> datetime:
    return datetime.now(UTC)


class JarvisOrganKind(StrEnum):
    PHASE1_KERNEL = "phase1_kernel"
    PHASE1_EVENTS = "phase1_events"
    PHASE1_OBSERVABILITY = "phase1_observability"
    PHASE2_PRESENCE = "phase2_presence"
    PHASE2_VOICE = "phase2_voice"
    PHASE3_COGNITION = "phase3_cognition"
    PHASE4_MEMORY = "phase4_memory"
    PHASE5_TOOLS = "phase5_tools"
    PHASE6_ORCHESTRATION = "phase6_orchestration"
    PHASE7_STREAMING_LATENCY = "phase7_streaming_latency"
    PHASE8_ENVIRONMENT = "phase8_environment"
    PHASE9_COGNITIVE_SESSION = "phase9_cognitive_session"
    STEP51_VOICE_LAUNCHER = "step51_voice_launcher"


class JarvisOrganStatus(StrEnum):
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    FAILED = "failed"
    STOPPING = "stopping"
    STOPPED = "stopped"


class JarvisOrganCriticality(StrEnum):
    REQUIRED = "required"
    OPTIONAL = "optional"


class JarvisStartControlStatus(StrEnum):
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    FAILED = "failed"
    STOPPING = "stopping"
    STOPPED = "stopped"


class JarvisStartControlOperation(StrEnum):
    START_ALL = "start_all"
    STOP_ALL = "stop_all"
    RECOVER = "recover"
    HEALTH = "health"
    SNAPSHOT = "snapshot"


class JarvisStartControlEvent(StrEnum):
    START_SEQUENCE_BUILT = "start_sequence_built"
    ORGAN_STARTED = "organ_started"
    ORGAN_DEGRADED = "organ_degraded"
    ORGAN_FAILED = "organ_failed"
    START_COMPLETED = "start_completed"
    STOP_COMPLETED = "stop_completed"
    RECOVERY_COMPLETED = "recovery_completed"
    HEALTH_CHECKED = "health_checked"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class JarvisOrganHealth:
    kind: JarvisOrganKind
    name: str
    status: JarvisOrganStatus
    criticality: JarvisOrganCriticality
    message: str
    latency_ms: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def healthy(self) -> bool:
        return self.status == JarvisOrganStatus.RUNNING

    @property
    def failed_required(self) -> bool:
        return (
            self.criticality == JarvisOrganCriticality.REQUIRED
            and self.status == JarvisOrganStatus.FAILED
        )


@dataclass(frozen=True, slots=True)
class JarvisOrganReport:
    kind: JarvisOrganKind
    name: str
    status: JarvisOrganStatus
    criticality: JarvisOrganCriticality
    operation: JarvisStartControlOperation
    message: str
    latency_ms: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status in {
            JarvisOrganStatus.RUNNING,
            JarvisOrganStatus.DEGRADED,
            JarvisOrganStatus.STOPPED,
        }


class JarvisOrganController(Protocol):
    @property
    def kind(self) -> JarvisOrganKind:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def criticality(self) -> JarvisOrganCriticality:
        raise NotImplementedError

    @property
    def dependencies(self) -> tuple[JarvisOrganKind, ...]:
        raise NotImplementedError

    def start(self) -> JarvisOrganReport:
        raise NotImplementedError

    def stop(self) -> JarvisOrganReport:
        raise NotImplementedError

    def recover(self) -> JarvisOrganReport:
        raise NotImplementedError

    def health(self) -> JarvisOrganHealth:
        raise NotImplementedError


@dataclass(slots=True)
class ManagedJarvisOrganController:
    """
    Generic adapter for real phase runtimes.

    It wraps any runtime object that may expose start/stop/recover/health methods.
    It does not generate speech and does not decide responses.
    """

    kind: JarvisOrganKind
    name: str
    runtime: object
    criticality: JarvisOrganCriticality = JarvisOrganCriticality.REQUIRED
    dependencies: tuple[JarvisOrganKind, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    _status: JarvisOrganStatus = JarvisOrganStatus.CREATED
    _last_error: str | None = None

    def start(self) -> JarvisOrganReport:
        started = time.perf_counter()
        self._status = JarvisOrganStatus.STARTING

        try:
            _call_if_present(self.runtime, "start")
            self._status = JarvisOrganStatus.RUNNING
            message = "organ started"
        except Exception as exc:
            self._status = JarvisOrganStatus.FAILED
            self._last_error = str(exc)
            message = "organ start failed"

        return self._report(
            operation=JarvisStartControlOperation.START_ALL,
            message=message,
            started=started,
        )

    def stop(self) -> JarvisOrganReport:
        started = time.perf_counter()
        self._status = JarvisOrganStatus.STOPPING

        try:
            _call_if_present(self.runtime, "stop")
            self._status = JarvisOrganStatus.STOPPED
            message = "organ stopped"
        except Exception as exc:
            self._status = JarvisOrganStatus.FAILED
            self._last_error = str(exc)
            message = "organ stop failed"

        return self._report(
            operation=JarvisStartControlOperation.STOP_ALL,
            message=message,
            started=started,
        )

    def recover(self) -> JarvisOrganReport:
        started = time.perf_counter()

        try:
            _call_if_present(self.runtime, "recover")
            self._status = JarvisOrganStatus.RUNNING
            message = "organ recovered"
        except Exception as exc:
            self._status = JarvisOrganStatus.FAILED
            self._last_error = str(exc)
            message = "organ recovery failed"

        return self._report(
            operation=JarvisStartControlOperation.RECOVER,
            message=message,
            started=started,
        )

    def health(self) -> JarvisOrganHealth:
        started = time.perf_counter()
        status = self._status

        try:
            health_obj = _call_if_present(self.runtime, "health")
            extracted = _extract_status(health_obj)
            if extracted is not None:
                status = extracted
        except Exception as exc:
            status = JarvisOrganStatus.FAILED
            self._last_error = str(exc)

        return JarvisOrganHealth(
            kind=self.kind,
            name=self.name,
            status=status,
            criticality=self.criticality,
            message="organ health checked",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            created_at=utc_now(),
            metadata={
                "last_error": self._last_error,
                **self.metadata,
            },
        )

    def _report(
        self,
        *,
        operation: JarvisStartControlOperation,
        message: str,
        started: float,
    ) -> JarvisOrganReport:
        return JarvisOrganReport(
            kind=self.kind,
            name=self.name,
            status=self._status,
            criticality=self.criticality,
            operation=operation,
            message=message,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            created_at=utc_now(),
            metadata={
                "last_error": self._last_error,
                **self.metadata,
            },
        )


@dataclass(slots=True)
class VoiceLauncherOrganController:
    """
    Runs Step 51 voice launcher as the final organ.

    It starts the real voice path in a background thread so Start Control
    can keep supervising all organs instead of blocking forever.
    """

    launcher: object
    kind: JarvisOrganKind = JarvisOrganKind.STEP51_VOICE_LAUNCHER
    name: str = "step51_voice_daily_driver_launcher"
    criticality: JarvisOrganCriticality = JarvisOrganCriticality.REQUIRED
    dependencies: tuple[JarvisOrganKind, ...] = (
        JarvisOrganKind.PHASE1_KERNEL,
        JarvisOrganKind.PHASE2_VOICE,
        JarvisOrganKind.PHASE3_COGNITION,
        JarvisOrganKind.PHASE4_MEMORY,
        JarvisOrganKind.PHASE6_ORCHESTRATION,
        JarvisOrganKind.PHASE8_ENVIRONMENT,
        JarvisOrganKind.PHASE9_COGNITIVE_SESSION,
    )

    _status: JarvisOrganStatus = JarvisOrganStatus.CREATED
    _thread: threading.Thread | None = None
    _last_error: str | None = None
    _last_launcher_reason: str | None = None
    _last_launcher_status: str | None = None
    _stop_requested: bool = False

    def start(self) -> JarvisOrganReport:
        started = time.perf_counter()

        if self._thread is not None and self._thread.is_alive():
            self._status = JarvisOrganStatus.RUNNING
            return self._report(
                operation=JarvisStartControlOperation.START_ALL,
                message="voice launcher already running",
                started=started,
            )

        self._status = JarvisOrganStatus.STARTING
        self._stop_requested = False

        def _run() -> None:
            try:
                result = _call_if_present(self.launcher, "run")
                result_status = _extract_status(result)
                result_reason = _extract_reason(result)
                self._last_launcher_status = (
                    result_status.value if result_status is not None else None
                )
                self._last_launcher_reason = result_reason
                if self._stop_requested:
                    self._status = JarvisOrganStatus.STOPPED
                elif result_status == JarvisOrganStatus.RUNNING:
                    self._status = JarvisOrganStatus.RUNNING
                elif result_status == JarvisOrganStatus.DEGRADED:
                    self._status = JarvisOrganStatus.DEGRADED
                    self._last_error = result_reason or "voice launcher degraded"
                elif result_status == JarvisOrganStatus.FAILED:
                    self._status = JarvisOrganStatus.FAILED
                    self._last_error = result_reason or "voice launcher failed"
                else:
                    self._status = JarvisOrganStatus.FAILED
                    self._last_error = _launcher_exit_message(
                        status=result_status,
                        reason=result_reason,
                    )
            except Exception as exc:
                self._status = JarvisOrganStatus.FAILED
                self._last_error = str(exc)

        self._thread = threading.Thread(
            target=_run,
            name="jarvis_voice_launcher",
            daemon=True,
        )
        self._thread.start()
        if self._status == JarvisOrganStatus.STARTING:
            self._status = JarvisOrganStatus.RUNNING

        return self._report(
            operation=JarvisStartControlOperation.START_ALL,
            message="voice launcher started",
            started=started,
        )

    def stop(self) -> JarvisOrganReport:
        started = time.perf_counter()
        self._status = JarvisOrganStatus.STOPPING
        self._stop_requested = True

        try:
            _call_if_present(self.launcher, "request_stop")
            _call_if_present(self.launcher, "stop")
            if self._thread is not None:
                self._thread.join(timeout=2.0)
            self._status = JarvisOrganStatus.STOPPED
            message = "voice launcher stopped"
        except Exception as exc:
            self._status = JarvisOrganStatus.FAILED
            self._last_error = str(exc)
            message = "voice launcher stop failed"

        return self._report(
            operation=JarvisStartControlOperation.STOP_ALL,
            message=message,
            started=started,
        )

    def recover(self) -> JarvisOrganReport:
        stop_report = self.stop()
        if stop_report.status == JarvisOrganStatus.FAILED:
            return stop_report
        return self.start()

    def health(self) -> JarvisOrganHealth:
        started = time.perf_counter()

        if self._thread is not None and self._thread.is_alive():
            status = JarvisOrganStatus.RUNNING
        else:
            status = self._status
            if status == JarvisOrganStatus.STOPPED and not self._stop_requested:
                status = JarvisOrganStatus.FAILED
                self._last_error = _launcher_exit_message(
                    status=JarvisOrganStatus.STOPPED,
                    reason=self._last_launcher_reason,
                )

        return JarvisOrganHealth(
            kind=self.kind,
            name=self.name,
            status=status,
            criticality=self.criticality,
            message="voice launcher health checked",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            created_at=utc_now(),
            metadata={
                "last_error": self._last_error,
                "launcher_status": self._last_launcher_status,
                "launcher_reason": self._last_launcher_reason,
            },
        )

    def _report(
        self,
        *,
        operation: JarvisStartControlOperation,
        message: str,
        started: float,
    ) -> JarvisOrganReport:
        return JarvisOrganReport(
            kind=self.kind,
            name=self.name,
            status=self._status,
            criticality=self.criticality,
            operation=operation,
            message=message,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            created_at=utc_now(),
            metadata={
                "last_error": self._last_error,
                "launcher_status": self._last_launcher_status,
                "launcher_reason": self._last_launcher_reason,
            },
        )


@dataclass(frozen=True, slots=True)
class JarvisStartControlConfig:
    allow_degraded_optional_organs: bool = True
    stop_started_organs_on_failure: bool = True
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class JarvisStartControlResult:
    status: JarvisStartControlStatus
    operation: JarvisStartControlOperation
    event: JarvisStartControlEvent | None
    organ_reports: tuple[JarvisOrganReport, ...]
    health: tuple[JarvisOrganHealth, ...]
    reason: str
    latency_ms: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status in {
            JarvisStartControlStatus.RUNNING,
            JarvisStartControlStatus.DEGRADED,
            JarvisStartControlStatus.STOPPED,
        }


@dataclass(frozen=True, slots=True)
class JarvisStartControlSnapshot:
    status: JarvisStartControlStatus
    started: bool
    stop_requested: bool
    organ_count: int
    running_count: int
    degraded_count: int
    failed_count: int
    last_event: JarvisStartControlEvent | None
    last_error: str | None
    last_latency_ms: float | None
    organ_health: tuple[JarvisOrganHealth, ...]
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


class JarvisStartControlRuntime:
    """
    Connected JARVIS Start Control.

    Starts all organs together as a personal cognitive OS. It does not
    generate final speech. Voice responses still come only through:
    STT -> awareness -> cognition/Ollama -> response boundary -> TTS.
    """

    def __init__(
        self,
        *,
        organs: Sequence[JarvisOrganController],
        config: JarvisStartControlConfig | None = None,
    ) -> None:
        self._organs = tuple(organs)
        self._config = config or JarvisStartControlConfig()
        self._status = JarvisStartControlStatus.CREATED
        self._started = False
        self._stop_requested = False
        self._last_event: JarvisStartControlEvent | None = None
        self._last_error: str | None = None
        self._last_latency_ms: float | None = None
        self._start_order: tuple[JarvisOrganController, ...] = ()

        _validate_unique_organs(self._organs)

    def start_all(self) -> JarvisStartControlResult:
        started = time.perf_counter()
        self._status = JarvisStartControlStatus.STARTING
        self._stop_requested = False
        reports: list[JarvisOrganReport] = []
        started_organs: list[JarvisOrganController] = []

        try:
            self._start_order = _topological_start_order(self._organs)
            self._last_event = JarvisStartControlEvent.START_SEQUENCE_BUILT
        except Exception as exc:
            self._status = JarvisStartControlStatus.FAILED
            self._last_error = str(exc)
            return self._result(
                operation=JarvisStartControlOperation.START_ALL,
                event=JarvisStartControlEvent.ERROR,
                reports=(),
                reason="start sequence failed",
                started=started,
                metadata={"error": str(exc)},
            )

        for organ in self._start_order:
            report = organ.start()
            reports.append(report)

            if report.status == JarvisOrganStatus.RUNNING:
                started_organs.append(organ)
                self._last_event = JarvisStartControlEvent.ORGAN_STARTED
                continue

            if report.status == JarvisOrganStatus.DEGRADED:
                started_organs.append(organ)
                self._last_event = JarvisStartControlEvent.ORGAN_DEGRADED
                continue

            self._last_event = JarvisStartControlEvent.ORGAN_FAILED

            if report.criticality == JarvisOrganCriticality.REQUIRED:
                self._status = JarvisStartControlStatus.FAILED
                self._last_error = report.message

                if self._config.stop_started_organs_on_failure:
                    reports.extend(self._stop_organs(tuple(reversed(started_organs))))

                return self._result(
                    operation=JarvisStartControlOperation.START_ALL,
                    event=JarvisStartControlEvent.ORGAN_FAILED,
                    reports=tuple(reports),
                    reason="required organ failed during start",
                    started=started,
                )

        health = tuple(organ.health() for organ in self._organs)
        failed_required = any(item.failed_required for item in health)
        degraded = any(
            item.status == JarvisOrganStatus.DEGRADED for item in health
        )

        if failed_required:
            self._status = JarvisStartControlStatus.FAILED
            reason = "required organ failed health check"
        elif degraded:
            self._status = JarvisStartControlStatus.DEGRADED
            reason = "connected JARVIS started with degraded organs"
        else:
            self._status = JarvisStartControlStatus.RUNNING
            reason = "connected JARVIS started"

        self._started = self._status in {
            JarvisStartControlStatus.RUNNING,
            JarvisStartControlStatus.DEGRADED,
        }
        self._last_event = JarvisStartControlEvent.START_COMPLETED

        return self._result(
            operation=JarvisStartControlOperation.START_ALL,
            event=JarvisStartControlEvent.START_COMPLETED,
            reports=tuple(reports),
            reason=reason,
            started=started,
        )

    def stop_all(self) -> JarvisStartControlResult:
        started = time.perf_counter()
        self._status = JarvisStartControlStatus.STOPPING
        self._stop_requested = True

        stop_order = (
            tuple(reversed(self._start_order))
            if self._start_order
            else tuple(reversed(self._organs))
        )
        reports = self._stop_organs(stop_order)

        failed = any(report.status == JarvisOrganStatus.FAILED for report in reports)

        self._started = False
        self._status = (
            JarvisStartControlStatus.FAILED
            if failed
            else JarvisStartControlStatus.STOPPED
        )
        self._last_event = JarvisStartControlEvent.STOP_COMPLETED

        return self._result(
            operation=JarvisStartControlOperation.STOP_ALL,
            event=JarvisStartControlEvent.STOP_COMPLETED,
            reports=tuple(reports),
            reason=(
                "connected JARVIS stop failed"
                if failed
                else "connected JARVIS stopped"
            ),
            started=started,
        )

    def recover(self) -> JarvisStartControlResult:
        started = time.perf_counter()
        reports: list[JarvisOrganReport] = []

        for organ in self._organs:
            health = organ.health()
            if health.status in {
                JarvisOrganStatus.DEGRADED,
                JarvisOrganStatus.FAILED,
            }:
                reports.append(organ.recover())

        health_after = tuple(organ.health() for organ in self._organs)
        failed_required = any(item.failed_required for item in health_after)
        degraded = any(
            item.status == JarvisOrganStatus.DEGRADED for item in health_after
        )

        if failed_required:
            self._status = JarvisStartControlStatus.FAILED
            reason = "recovery failed for required organ"
        elif degraded:
            self._status = JarvisStartControlStatus.DEGRADED
            reason = "recovery completed with degraded organs"
        else:
            self._status = JarvisStartControlStatus.RUNNING
            reason = "recovery completed"

        self._last_event = JarvisStartControlEvent.RECOVERY_COMPLETED

        return self._result(
            operation=JarvisStartControlOperation.RECOVER,
            event=JarvisStartControlEvent.RECOVERY_COMPLETED,
            reports=tuple(reports),
            reason=reason,
            started=started,
        )

    def health(self) -> JarvisStartControlResult:
        started = time.perf_counter()
        health = tuple(organ.health() for organ in self._organs)
        unexpected_required_stop = (
            self._started
            and not self._stop_requested
            and any(
                item.criticality == JarvisOrganCriticality.REQUIRED
                and item.status == JarvisOrganStatus.STOPPED
                for item in health
            )
        )
        failed_required = any(item.failed_required for item in health) or (
            unexpected_required_stop
        )
        degraded = any(
            item.status == JarvisOrganStatus.DEGRADED for item in health
        )

        if failed_required:
            self._status = JarvisStartControlStatus.FAILED
            reason = "connected JARVIS health failed"
        elif degraded:
            self._status = JarvisStartControlStatus.DEGRADED
            reason = "connected JARVIS health degraded"
        elif self._started:
            self._status = JarvisStartControlStatus.RUNNING
            reason = "connected JARVIS health ready"
        else:
            reason = "connected JARVIS health checked"

        self._last_event = JarvisStartControlEvent.HEALTH_CHECKED

        return JarvisStartControlResult(
            status=self._status,
            operation=JarvisStartControlOperation.HEALTH,
            event=JarvisStartControlEvent.HEALTH_CHECKED,
            organ_reports=(),
            health=health,
            reason=reason,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            created_at=utc_now(),
            metadata=self._config.metadata,
        )

    def snapshot(self) -> JarvisStartControlSnapshot:
        health = tuple(organ.health() for organ in self._organs)
        running_count = sum(
            1 for item in health if item.status == JarvisOrganStatus.RUNNING
        )
        degraded_count = sum(
            1 for item in health if item.status == JarvisOrganStatus.DEGRADED
        )
        failed_count = sum(
            1 for item in health if item.status == JarvisOrganStatus.FAILED
        )

        return JarvisStartControlSnapshot(
            status=self._status,
            started=self._started,
            stop_requested=self._stop_requested,
            organ_count=len(self._organs),
            running_count=running_count,
            degraded_count=degraded_count,
            failed_count=failed_count,
            last_event=self._last_event,
            last_error=self._last_error,
            last_latency_ms=self._last_latency_ms,
            organ_health=health,
            created_at=utc_now(),
            metadata=self._config.metadata,
        )

    def _stop_organs(
        self,
        organs: tuple[JarvisOrganController, ...],
    ) -> list[JarvisOrganReport]:
        reports: list[JarvisOrganReport] = []
        for organ in organs:
            reports.append(organ.stop())
        return reports

    def _result(
        self,
        *,
        operation: JarvisStartControlOperation,
        event: JarvisStartControlEvent | None,
        reports: tuple[JarvisOrganReport, ...],
        reason: str,
        started: float,
        metadata: dict[str, object] | None = None,
    ) -> JarvisStartControlResult:
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._last_latency_ms = latency_ms

        if event is not None:
            self._last_event = event

        return JarvisStartControlResult(
            status=self._status,
            operation=operation,
            event=event,
            organ_reports=reports,
            health=tuple(organ.health() for organ in self._organs),
            reason=reason,
            latency_ms=latency_ms,
            created_at=utc_now(),
            metadata={
                **self._config.metadata,
                **(metadata or {}),
            },
        )


def _validate_unique_organs(organs: tuple[JarvisOrganController, ...]) -> None:
    seen: set[JarvisOrganKind] = set()
    for organ in organs:
        if organ.kind in seen:
            raise ValueError(f"duplicate organ kind: {organ.kind.value}")
        seen.add(organ.kind)


def _topological_start_order(
    organs: tuple[JarvisOrganController, ...],
) -> tuple[JarvisOrganController, ...]:
    by_kind = {organ.kind: organ for organ in organs}
    indegree: dict[JarvisOrganKind, int] = {organ.kind: 0 for organ in organs}
    graph: dict[JarvisOrganKind, list[JarvisOrganKind]] = defaultdict(list)

    for organ in organs:
        for dependency in organ.dependencies:
            if dependency not in by_kind:
                raise ValueError(
                    f"{organ.kind.value} depends on missing organ "
                    f"{dependency.value}"
                )
            graph[dependency].append(organ.kind)
            indegree[organ.kind] += 1

    queue: deque[JarvisOrganKind] = deque(
        kind for kind, count in indegree.items() if count == 0
    )
    ordered: list[JarvisOrganController] = []

    while queue:
        kind = queue.popleft()
        ordered.append(by_kind[kind])

        for dependent in graph[kind]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                queue.append(dependent)

    if len(ordered) != len(organs):
        raise ValueError("organ dependency cycle detected")

    return tuple(ordered)


def _call_if_present(runtime: object, method_name: str) -> object | None:
    method = getattr(runtime, method_name, None)

    if not callable(method):
        return None

    result: object | None = method()
    return result


def _extract_status(value: object | None) -> JarvisOrganStatus | None:
    if value is None:
        return None

    status = getattr(value, "status", None)
    if isinstance(status, JarvisOrganStatus):
        return status

    if isinstance(status, str):
        try:
            return JarvisOrganStatus(status)
        except ValueError:
            return None

    return None


def _extract_reason(value: object | None) -> str | None:
    if value is None:
        return None

    reason = getattr(value, "reason", None)
    if isinstance(reason, str) and reason.strip():
        return reason.strip()

    message = getattr(value, "message", None)
    if isinstance(message, str) and message.strip():
        return message.strip()

    return None


def _launcher_exit_message(
    *,
    status: JarvisOrganStatus | None,
    reason: str | None,
) -> str:
    status_text = status.value if status is not None else "unknown"
    if reason is None:
        return f"voice launcher exited unexpectedly status={status_text}"
    return f"voice launcher exited unexpectedly status={status_text}: {reason}"
