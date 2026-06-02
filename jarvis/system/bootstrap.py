from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from uuid import uuid4

from jarvis.cognition.worker import CognitionWorker
from jarvis.conversation.runtime import RealConversationRuntime
from jarvis.memory.gateway import MemoryGateway
from jarvis.presence import PresenceEngine
from jarvis.runtime.kernel.runtime_kernel import RuntimeKernel
from jarvis.system.assembly import JarvisSystem
from jarvis.system.contracts import JarvisSystemSnapshot, JarvisSystemStatus, utc_now


def new_boot_id() -> str:
    return f"jarvis_boot_{uuid4().hex}"


class JarvisBootstrapStatus(StrEnum):
    CREATED = "created"
    STARTED = "started"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class JarvisBootstrapConfig:
    name: str = "jarvis_system"
    dry_run: bool = False
    attach_conversation: bool = True
    attach_presence: bool = True
    attach_orchestration: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("bootstrap config name cannot be empty.")


@dataclass(frozen=True, slots=True)
class JarvisSystemFactoryBundle:
    memory_gateway: Callable[[], MemoryGateway]
    cognition_worker: Callable[[], CognitionWorker]
    conversation_runtime: Callable[[], RealConversationRuntime] | None = None
    presence_engine: Callable[[], PresenceEngine] | None = None
    orchestration_runtime: Callable[[], object] | None = None
    kernel: Callable[[], RuntimeKernel] = RuntimeKernel


@dataclass(frozen=True, slots=True)
class JarvisBootstrapResult:
    boot_id: str
    status: JarvisBootstrapStatus
    system_snapshot: JarvisSystemSnapshot | None
    started_at: datetime | None
    stopped_at: datetime | None
    error: str | None
    metadata: dict[str, object]

    @property
    def succeeded(self) -> bool:
        return self.status in {
            JarvisBootstrapStatus.STARTED,
            JarvisBootstrapStatus.STOPPED,
        } and self.error is None


class JarvisSystemBootstrap:
    """
    Production-safe bootstrap wrapper for the living JARVIS system.

    This is the bridge from old RuntimeKernel-only boot to full JarvisSystem
    boot. It does not construct real adapters blindly. Dependencies are
    provided through explicit factories.
    """

    def __init__(
        self,
        *,
        config: JarvisBootstrapConfig,
        factories: JarvisSystemFactoryBundle,
    ) -> None:
        self._config = config
        self._factories = factories
        self._system: JarvisSystem | None = None

    @property
    def system(self) -> JarvisSystem | None:
        return self._system

    def build_system(self) -> JarvisSystem:
        conversation_runtime = (
            self._factories.conversation_runtime()
            if (
                self._config.attach_conversation
                and self._factories.conversation_runtime is not None
            )
            else None
        )
        presence_engine = (
            self._factories.presence_engine()
            if (
                self._config.attach_presence
                and self._factories.presence_engine is not None
            )
            else None
        )
        orchestration_runtime = (
            self._factories.orchestration_runtime()
            if (
                self._config.attach_orchestration
                and self._factories.orchestration_runtime is not None
            )
            else None
        )

        return JarvisSystem(
            name=self._config.name,
            memory_gateway=self._factories.memory_gateway(),
            cognition_worker=self._factories.cognition_worker(),
            conversation_runtime=conversation_runtime,
            presence_engine=presence_engine,
            orchestration_runtime=orchestration_runtime,
            kernel=self._factories.kernel(),
        )

    def start(self) -> JarvisBootstrapResult:
        boot_id = new_boot_id()
        started_at: datetime | None = None

        try:
            self._system = self.build_system()
            self._system.start()
            started_at = utc_now()
            snapshot = self._system.snapshot()

            if snapshot.status != JarvisSystemStatus.RUNNING:
                raise RuntimeError(
                    f"JarvisSystem did not enter RUNNING state: "
                    f"{snapshot.status.value}"
                )

            if self._config.dry_run:
                self._system.stop()
                stopped_at = utc_now()
                return JarvisBootstrapResult(
                    boot_id=boot_id,
                    status=JarvisBootstrapStatus.STOPPED,
                    system_snapshot=self._system.snapshot(),
                    started_at=started_at,
                    stopped_at=stopped_at,
                    error=None,
                    metadata={
                        **self._config.metadata,
                        "dry_run": True,
                    },
                )

            return JarvisBootstrapResult(
                boot_id=boot_id,
                status=JarvisBootstrapStatus.STARTED,
                system_snapshot=snapshot,
                started_at=started_at,
                stopped_at=None,
                error=None,
                metadata={
                    **self._config.metadata,
                    "dry_run": False,
                },
            )

        except Exception as exc:
            if self._system is not None:
                try:
                    self._system.stop()
                except Exception:
                    pass

            return JarvisBootstrapResult(
                boot_id=boot_id,
                status=JarvisBootstrapStatus.FAILED,
                system_snapshot=(
                    self._system.snapshot()
                    if self._system is not None
                    else None
                ),
                started_at=started_at,
                stopped_at=utc_now(),
                error=f"{type(exc).__name__}: {exc}",
                metadata={
                    **self._config.metadata,
                    "dry_run": self._config.dry_run,
                },
            )

    def stop(self) -> JarvisBootstrapResult:
        boot_id = new_boot_id()

        if self._system is None:
            return JarvisBootstrapResult(
                boot_id=boot_id,
                status=JarvisBootstrapStatus.STOPPED,
                system_snapshot=None,
                started_at=None,
                stopped_at=utc_now(),
                error=None,
                metadata={"already_stopped": True},
            )

        try:
            self._system.stop()
            return JarvisBootstrapResult(
                boot_id=boot_id,
                status=JarvisBootstrapStatus.STOPPED,
                system_snapshot=self._system.snapshot(),
                started_at=None,
                stopped_at=utc_now(),
                error=None,
                metadata={},
            )
        except Exception as exc:
            return JarvisBootstrapResult(
                boot_id=boot_id,
                status=JarvisBootstrapStatus.FAILED,
                system_snapshot=self._system.snapshot(),
                started_at=None,
                stopped_at=utc_now(),
                error=f"{type(exc).__name__}: {exc}",
                metadata={},
            )


def create_unconfigured_bootstrap(
    *,
    dry_run: bool = True,
) -> JarvisSystemBootstrap:
    """
    Safe placeholder for production bootstrap.

    Step 45 wires bootstrap to JarvisSystem, but it should not guess how to
    construct real memory/cognition/presence dependencies. Step 46 will provide
    the real live factories.
    """
    config = JarvisBootstrapConfig(
        dry_run=dry_run,
        metadata={"mode": "unconfigured"},
    )

    return JarvisSystemBootstrap(
        config=config,
        factories=JarvisSystemFactoryBundle(
            memory_gateway=_missing_memory_gateway,
            cognition_worker=_missing_cognition_worker,
        ),
    )


def _missing_memory_gateway() -> MemoryGateway:
    raise RuntimeError(
        "MemoryGateway factory is not configured. "
        "Step 46 must provide the live memory gateway factory."
    )


def _missing_cognition_worker() -> CognitionWorker:
    raise RuntimeError(
        "CognitionWorker factory is not configured. "
        "Step 46 must provide the live cognition worker factory."
    )


def parse_dry_run_arg(argv: list[str]) -> bool:
    return "--dry-run" in argv or "--check" in argv


def main(argv: list[str] | None = None) -> int:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    dry_run = parse_dry_run_arg(args)

    bootstrap = create_unconfigured_bootstrap(dry_run=dry_run)
    result = bootstrap.start()

    if result.succeeded:
        print(
            "JARVIS bootstrap completed: "
            f"status={result.status.value} dry_run={dry_run}"
        )
        return 0

    print(f"JARVIS bootstrap failed: {result.error}")
    return 1