from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from jarvis.conversation.models import ConversationMode, TurnInputSource
from jarvis.conversation.runtime import RealConversationInput
from jarvis.system.bootstrap import (
    JarvisBootstrapConfig,
    JarvisBootstrapStatus,
    JarvisSystemBootstrap,
    JarvisSystemFactoryBundle,
)
from jarvis.system.contracts import (
    JarvisAskStatus,
    JarvisPipelineStatus,
    JarvisSystemStatus,
    utc_now,
)


class JarvisAliveGateStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


class JarvisAliveGateCheckKind(StrEnum):
    BOOT = "boot"
    SNAPSHOT_RUNNING = "snapshot_running"
    NORMAL_PIPELINE = "normal_pipeline"
    GOVERNED_MEMORY_WRITE = "governed_memory_write"
    INTERRUPTION = "interruption"
    SHUTDOWN = "shutdown"


@dataclass(frozen=True, slots=True)
class JarvisAliveGateCheck:
    kind: JarvisAliveGateCheckKind
    passed: bool
    message: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class JarvisAliveGateReport:
    status: JarvisAliveGateStatus
    checks: tuple[JarvisAliveGateCheck, ...]
    started_at: datetime
    finished_at: datetime
    error: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == JarvisAliveGateStatus.PASSED

    @property
    def passed_count(self) -> int:
        return sum(1 for check in self.checks if check.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for check in self.checks if not check.passed)


@dataclass(frozen=True, slots=True)
class JarvisAliveGateConfig:
    name: str = "jarvis_alive_gate_system"
    session_id: str = "alive_gate"
    normal_prompt: str = "What am I building?"
    memory_prompt: str = "Remember that my favorite editor is VS Code."
    interrupt_prompt: str = "stop"
    attach_conversation: bool = True
    attach_presence: bool = True
    attach_orchestration: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("alive gate name cannot be empty.")
        if not self.session_id.strip():
            raise ValueError("alive gate session_id cannot be empty.")
        if not self.normal_prompt.strip():
            raise ValueError("alive gate normal_prompt cannot be empty.")
        if not self.memory_prompt.strip():
            raise ValueError("alive gate memory_prompt cannot be empty.")
        if not self.interrupt_prompt.strip():
            raise ValueError("alive gate interrupt_prompt cannot be empty.")


class JarvisAliveGate:
    """
    Step 46 life-sign gate for the living JARVIS system.

    This gate proves:
    - JarvisSystemBootstrap can start the organism.
    - JarvisSystem reaches RUNNING.
    - User utterance flows through the full pipeline.
    - Governed memory write still works.
    - Interruption cancels the cognition path.
    - Shutdown is clean.

    It does not execute tools, desktop actions, or autonomous tasks.
    """

    def __init__(
        self,
        *,
        config: JarvisAliveGateConfig,
        factories: JarvisSystemFactoryBundle,
    ) -> None:
        self._config = config
        self._factories = factories

    def run(self) -> JarvisAliveGateReport:
        started_at = utc_now()
        checks: list[JarvisAliveGateCheck] = []
        bootstrap = JarvisSystemBootstrap(
            config=JarvisBootstrapConfig(
                name=self._config.name,
                dry_run=False,
                attach_conversation=self._config.attach_conversation,
                attach_presence=self._config.attach_presence,
                attach_orchestration=self._config.attach_orchestration,
                metadata={
                    **self._config.metadata,
                    "gate": "alive",
                },
            ),
            factories=self._factories,
        )

        try:
            boot_result = bootstrap.start()
            boot_passed = (
                boot_result.status == JarvisBootstrapStatus.STARTED
                and boot_result.error is None
            )

            _record_check(
                checks,
                kind=JarvisAliveGateCheckKind.BOOT,
                passed=boot_passed,
                message="JarvisSystemBootstrap started JarvisSystem.",
                metadata={
                    "bootstrap_status": boot_result.status.value,
                    "error": boot_result.error,
                },
            )

            if not boot_passed:
                return _report(
                    checks=checks,
                    started_at=started_at,
                    error=boot_result.error or "bootstrap failed",
                    metadata=self._config.metadata,
                )

            system = bootstrap.system
            if system is None:
                raise RuntimeError("bootstrap did not expose a JarvisSystem.")

            snapshot = system.snapshot()
            _record_check(
                checks,
                kind=JarvisAliveGateCheckKind.SNAPSHOT_RUNNING,
                passed=snapshot.status == JarvisSystemStatus.RUNNING,
                message="JarvisSystem snapshot is running.",
                metadata={
                    "system_status": snapshot.status.value,
                    "subsystem_count": len(snapshot.subsystem_health),
                },
            )

            normal_result = system.process_user_utterance(
                text=self._config.normal_prompt,
                session_id=self._config.session_id,
            )
            _record_check(
                checks,
                kind=JarvisAliveGateCheckKind.NORMAL_PIPELINE,
                passed=(
                    normal_result.status == JarvisPipelineStatus.COMPLETED
                    and normal_result.response is not None
                    and normal_result.response.status
                    == JarvisAskStatus.ANSWERED
                ),
                message="Normal user utterance completed through pipeline.",
                metadata={
                    "pipeline_status": normal_result.status.value,
                    "event_count": len(normal_result.events),
                },
            )

            memory_result = system.process_user_utterance(
                text=self._config.memory_prompt,
                session_id=self._config.session_id,
            )
            _record_check(
                checks,
                kind=JarvisAliveGateCheckKind.GOVERNED_MEMORY_WRITE,
                passed=(
                    memory_result.status == JarvisPipelineStatus.COMPLETED
                    and memory_result.response is not None
                    and memory_result.response.wrote_memory
                ),
                message="Explicit memory request wrote through governed memory.",
                metadata={
                    "pipeline_status": memory_result.status.value,
                    "wrote_memory": (
                        memory_result.response.wrote_memory
                        if memory_result.response is not None
                        else False
                    ),
                },
            )

            interrupt_result = system.process_user_utterance(
                text=self._config.interrupt_prompt,
                session_id=self._config.session_id,
                signal=_interruption_signal(self._config.interrupt_prompt),
            )
            _record_check(
                checks,
                kind=JarvisAliveGateCheckKind.INTERRUPTION,
                passed=(
                    interrupt_result.status == JarvisPipelineStatus.CANCELLED
                    and interrupt_result.cancelled
                    and interrupt_result.response is None
                ),
                message="Interruption cancelled active cognition path.",
                metadata={
                    "pipeline_status": interrupt_result.status.value,
                    "cancelled": interrupt_result.cancelled,
                },
            )

            stop_result = bootstrap.stop()
            _record_check(
                checks,
                kind=JarvisAliveGateCheckKind.SHUTDOWN,
                passed=(
                    stop_result.status == JarvisBootstrapStatus.STOPPED
                    and stop_result.error is None
                    and stop_result.system_snapshot is not None
                    and stop_result.system_snapshot.status
                    == JarvisSystemStatus.STOPPED
                ),
                message="JarvisSystem shut down cleanly.",
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
            try:
                stop_result = bootstrap.stop()
                _record_check(
                    checks,
                    kind=JarvisAliveGateCheckKind.SHUTDOWN,
                    passed=(
                        stop_result.status == JarvisBootstrapStatus.STOPPED
                        and stop_result.error is None
                    ),
                    message="JarvisSystem shutdown attempted after gate failure.",
                    metadata={
                        "bootstrap_status": stop_result.status.value,
                        "error": stop_result.error,
                    },
                )
            except Exception as stop_exc:
                _record_check(
                    checks,
                    kind=JarvisAliveGateCheckKind.SHUTDOWN,
                    passed=False,
                    message="JarvisSystem shutdown failed after gate failure.",
                    metadata={
                        "error": f"{type(stop_exc).__name__}: {stop_exc}",
                    },
                )

            return _report(
                checks=checks,
                started_at=started_at,
                error=f"{type(exc).__name__}: {exc}",
                metadata=self._config.metadata,
            )


def _record_check(
    checks: list[JarvisAliveGateCheck],
    *,
    kind: JarvisAliveGateCheckKind,
    passed: bool,
    message: str,
    metadata: dict[str, object] | None = None,
) -> None:
    checks.append(
        JarvisAliveGateCheck(
            kind=kind,
            passed=passed,
            message=message,
            created_at=utc_now(),
            metadata=metadata or {},
        )
    )


def _report(
    *,
    checks: list[JarvisAliveGateCheck],
    started_at: datetime,
    error: str | None,
    metadata: dict[str, object],
) -> JarvisAliveGateReport:
    status = (
        JarvisAliveGateStatus.PASSED
        if checks and all(check.passed for check in checks) and error is None
        else JarvisAliveGateStatus.FAILED
    )
    return JarvisAliveGateReport(
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