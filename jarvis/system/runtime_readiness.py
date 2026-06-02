from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from jarvis.conversation.models import ConversationMode, TurnInputSource
from jarvis.conversation.runtime import RealConversationInput
from jarvis.system.bootstrap import JarvisBootstrapStatus
from jarvis.system.contracts import (
    JarvisAskStatus,
    JarvisPipelineStatus,
    JarvisSystemStatus,
    utc_now,
)
from jarvis.system.live_wiring import (
    LiveDependencyWiring,
)


class RuntimeReadinessStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


class RuntimeReadinessCheckKind(StrEnum):
    DEPENDENCY_GRAPH = "dependency_graph"
    BOOT = "boot"
    SYSTEM_RUNNING = "system_running"
    WORKERS_REGISTERED = "workers_registered"
    EVENTBUS_STARTED = "eventbus_started"
    COGNITION_PIPELINE = "cognition_pipeline"
    MEMORY_RETRIEVE = "memory_retrieve"
    MEMORY_WRITE = "memory_write"
    PRESENCE_OUTPUT = "presence_output"
    INTERRUPTION = "interruption"
    RECOVERY_BASELINE = "recovery_baseline"
    SHUTDOWN = "shutdown"


@dataclass(frozen=True, slots=True)
class RuntimeReadinessCheck:
    kind: RuntimeReadinessCheckKind
    passed: bool
    message: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RuntimeReadinessReport:
    status: RuntimeReadinessStatus
    checks: tuple[RuntimeReadinessCheck, ...]
    started_at: datetime
    finished_at: datetime
    error: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == RuntimeReadinessStatus.PASSED

    @property
    def passed_count(self) -> int:
        return sum(1 for check in self.checks if check.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for check in self.checks if not check.passed)


@dataclass(frozen=True, slots=True)
class RuntimeReadinessConfig:
    session_id: str = "runtime_readiness"
    normal_prompt: str = "What am I building?"
    memory_prompt: str = "Remember that my preferred editor is VS Code."
    interruption_prompt: str = "stop"
    recovery_prompt: str = "Are you still ready?"
    require_presence: bool = False
    require_orchestration: bool = False
    minimum_subsystems: int = 3
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("runtime readiness session_id cannot be empty.")
        if not self.normal_prompt.strip():
            raise ValueError("runtime readiness normal_prompt cannot be empty.")
        if not self.memory_prompt.strip():
            raise ValueError("runtime readiness memory_prompt cannot be empty.")
        if not self.interruption_prompt.strip():
            raise ValueError(
                "runtime readiness interruption_prompt cannot be empty."
            )
        if not self.recovery_prompt.strip():
            raise ValueError("runtime readiness recovery_prompt cannot be empty.")
        if self.minimum_subsystems < 1:
            raise ValueError("minimum_subsystems must be at least 1.")


class RuntimeReadinessReview:
    """
    Step 46D hard runtime readiness gate.

    This review proves the assembled JARVIS runtime can:
    - validate dependencies
    - boot
    - register workers
    - start kernel/EventBus path
    - process cognition
    - retrieve memory
    - write governed memory
    - publish presence output when configured
    - cancel through interruption path
    - continue after interruption
    - shut down cleanly

    It does not perform deliberate failure injection.
    That belongs to Step 46D.5.
    """

    def __init__(
        self,
        *,
        config: RuntimeReadinessConfig,
        wiring: LiveDependencyWiring,
    ) -> None:
        self._config = config
        self._wiring = wiring

    def run(self) -> RuntimeReadinessReport:
        started_at = utc_now()
        checks: list[RuntimeReadinessCheck] = []
        bootstrap = None

        try:
            wiring_report = self._wiring.validate_bootstrap_ready()
            _record(
                checks,
                kind=RuntimeReadinessCheckKind.DEPENDENCY_GRAPH,
                passed=wiring_report.succeeded
                and wiring_report.dependency_graph is not None
                and wiring_report.dependency_graph.boot_allowed,
                message="dependency graph validated before runtime boot",
                metadata={
                    "wiring_status": wiring_report.status.value,
                    "error": wiring_report.error,
                    "graph_status": (
                        wiring_report.dependency_graph.status.value
                        if wiring_report.dependency_graph is not None
                        else None
                    ),
                },
            )

            if not wiring_report.succeeded:
                return _report(
                    checks=checks,
                    started_at=started_at,
                    error=wiring_report.error or "dependency wiring failed",
                    metadata=self._config.metadata,
                )

            bootstrap = self._wiring.build_bootstrap()
            boot_result = bootstrap.start()
            boot_passed = (
                boot_result.status == JarvisBootstrapStatus.STARTED
                and boot_result.error is None
                and bootstrap.system is not None
            )
            _record(
                checks,
                kind=RuntimeReadinessCheckKind.BOOT,
                passed=boot_passed,
                message="JarvisSystem booted through bootstrap",
                metadata={
                    "bootstrap_status": boot_result.status.value,
                    "error": boot_result.error,
                },
            )

            if not boot_passed:
                return _report(
                    checks=checks,
                    started_at=started_at,
                    error=boot_result.error or "runtime boot failed",
                    metadata=self._config.metadata,
                )

            system = bootstrap.system
            if system is None:
                raise RuntimeError("bootstrap did not expose JarvisSystem.")

            snapshot = system.snapshot()
            _record(
                checks,
                kind=RuntimeReadinessCheckKind.SYSTEM_RUNNING,
                passed=snapshot.status == JarvisSystemStatus.RUNNING,
                message="JarvisSystem is running",
                metadata={
                    "system_status": snapshot.status.value,
                },
            )

            subsystem_count = len(snapshot.subsystem_health)
            _record(
                checks,
                kind=RuntimeReadinessCheckKind.WORKERS_REGISTERED,
                passed=subsystem_count >= self._config.minimum_subsystems,
                message="runtime workers registered and visible in snapshot",
                metadata={
                    "subsystem_count": subsystem_count,
                    "minimum_subsystems": self._config.minimum_subsystems,
                },
            )

            _record(
                checks,
                kind=RuntimeReadinessCheckKind.EVENTBUS_STARTED,
                passed=snapshot.kernel_snapshot is not None,
                message="kernel snapshot is available, implying runtime started",
                metadata={
                    "kernel_snapshot_present": snapshot.kernel_snapshot
                    is not None,
                },
            )

            normal_result = system.process_user_utterance(
                text=self._config.normal_prompt,
                session_id=self._config.session_id,
            )
            normal_response_ok = (
                normal_result.status == JarvisPipelineStatus.COMPLETED
                and normal_result.response is not None
                and normal_result.response.status == JarvisAskStatus.ANSWERED
            )
            _record(
                checks,
                kind=RuntimeReadinessCheckKind.COGNITION_PIPELINE,
                passed=normal_response_ok,
                message="normal utterance completed cognition pipeline",
                metadata={
                    "pipeline_status": normal_result.status.value,
                    "event_count": len(normal_result.events),
                },
            )
            _record(
                checks,
                kind=RuntimeReadinessCheckKind.MEMORY_RETRIEVE,
                passed=(
                    normal_result.response is not None
                    and normal_result.response.used_memory
                ),
                message="memory retrieval path was used",
                metadata={
                    "used_memory": (
                        normal_result.response.used_memory
                        if normal_result.response is not None
                        else False
                    ),
                },
            )

            memory_result = system.process_user_utterance(
                text=self._config.memory_prompt,
                session_id=self._config.session_id,
            )
            _record(
                checks,
                kind=RuntimeReadinessCheckKind.MEMORY_WRITE,
                passed=(
                    memory_result.status == JarvisPipelineStatus.COMPLETED
                    and memory_result.response is not None
                    and memory_result.response.wrote_memory
                ),
                message="explicit memory request wrote through gateway",
                metadata={
                    "pipeline_status": memory_result.status.value,
                    "wrote_memory": (
                        memory_result.response.wrote_memory
                        if memory_result.response is not None
                        else False
                    ),
                },
            )

            presence_worker = system.presence_worker
            presence_ok = presence_worker is not None
            if presence_worker is not None:
                system.publish_presence_response_ready(
                    text="Runtime readiness presence check."
                )

            _record(
                checks,
                kind=RuntimeReadinessCheckKind.PRESENCE_OUTPUT,
                passed=presence_ok or not self._config.require_presence,
                message="presence output path checked",
                metadata={
                    "presence_configured": presence_ok,
                    "required": self._config.require_presence,
                },
            )

            interrupt_result = system.process_user_utterance(
                text=self._config.interruption_prompt,
                session_id=self._config.session_id,
                signal=_interruption_signal(self._config.interruption_prompt),
            )
            _record(
                checks,
                kind=RuntimeReadinessCheckKind.INTERRUPTION,
                passed=(
                    interrupt_result.status == JarvisPipelineStatus.CANCELLED
                    and interrupt_result.cancelled
                    and interrupt_result.response is None
                ),
                message="interruption cancelled active cognition path",
                metadata={
                    "pipeline_status": interrupt_result.status.value,
                    "cancelled": interrupt_result.cancelled,
                },
            )

            recovery_result = system.process_user_utterance(
                text=self._config.recovery_prompt,
                session_id=self._config.session_id,
            )
            _record(
                checks,
                kind=RuntimeReadinessCheckKind.RECOVERY_BASELINE,
                passed=(
                    recovery_result.status == JarvisPipelineStatus.COMPLETED
                    and recovery_result.response is not None
                    and recovery_result.response.status
                    == JarvisAskStatus.ANSWERED
                ),
                message="runtime continued after interruption",
                metadata={
                    "pipeline_status": recovery_result.status.value,
                },
            )

            stop_result = bootstrap.stop()
            _record(
                checks,
                kind=RuntimeReadinessCheckKind.SHUTDOWN,
                passed=(
                    stop_result.status == JarvisBootstrapStatus.STOPPED
                    and stop_result.error is None
                    and stop_result.system_snapshot is not None
                    and stop_result.system_snapshot.status
                    == JarvisSystemStatus.STOPPED
                ),
                message="JarvisSystem shut down cleanly",
                metadata={
                    "bootstrap_status": stop_result.status.value,
                    "error": stop_result.error,
                },
            )

            return _report(
                checks=checks,
                started_at=started_at,
                error=None,
                metadata=self._config.metadata,
            )

        except Exception as exc:
            if bootstrap is not None:
                try:
                    stop_result = bootstrap.stop()
                    _record(
                        checks,
                        kind=RuntimeReadinessCheckKind.SHUTDOWN,
                        passed=(
                            stop_result.status == JarvisBootstrapStatus.STOPPED
                            and stop_result.error is None
                        ),
                        message="shutdown attempted after readiness failure",
                        metadata={
                            "bootstrap_status": stop_result.status.value,
                            "error": stop_result.error,
                        },
                    )
                except Exception as stop_exc:
                    _record(
                        checks,
                        kind=RuntimeReadinessCheckKind.SHUTDOWN,
                        passed=False,
                        message="shutdown failed after readiness failure",
                        metadata={
                            "error": (
                                f"{type(stop_exc).__name__}: {stop_exc}"
                            ),
                        },
                    )

            return _report(
                checks=checks,
                started_at=started_at,
                error=f"{type(exc).__name__}: {exc}",
                metadata=self._config.metadata,
            )


def _record(
    checks: list[RuntimeReadinessCheck],
    *,
    kind: RuntimeReadinessCheckKind,
    passed: bool,
    message: str,
    metadata: dict[str, object] | None = None,
) -> None:
    checks.append(
        RuntimeReadinessCheck(
            kind=kind,
            passed=passed,
            message=message,
            created_at=utc_now(),
            metadata=metadata or {},
        )
    )


def _report(
    *,
    checks: list[RuntimeReadinessCheck],
    started_at: datetime,
    error: str | None,
    metadata: dict[str, object],
) -> RuntimeReadinessReport:
    status = (
        RuntimeReadinessStatus.PASSED
        if checks and all(check.passed for check in checks) and error is None
        else RuntimeReadinessStatus.FAILED
    )
    return RuntimeReadinessReport(
        status=status,
        checks=tuple(checks),
        started_at=started_at,
        finished_at=utc_now(),
        error=error,
        metadata=metadata,
    )


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