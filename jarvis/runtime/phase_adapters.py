from __future__ import annotations

import importlib
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from jarvis.runtime.start_control import (
    JarvisOrganController,
    JarvisOrganCriticality,
    JarvisOrganHealth,
    JarvisOrganKind,
    JarvisOrganReport,
    JarvisOrganStatus,
    JarvisStartControlConfig,
    JarvisStartControlOperation,
    JarvisStartControlRuntime,
    VoiceLauncherOrganController,
    utc_now,
)


class JarvisPhaseAdapterStatus(StrEnum):
    READY = "ready"
    FAILED = "failed"


class JarvisPhaseBindingSource(StrEnum):
    OBJECT = "object"
    IMPORT_PATH = "import_path"


@dataclass(frozen=True, slots=True)
class JarvisPhaseLifecyclePolicy:
    start_methods: tuple[str, ...] = ("start", "boot", "initialize")
    stop_methods: tuple[str, ...] = ("stop", "shutdown", "close")
    recover_methods: tuple[str, ...] = ("recover", "restart")
    health_methods: tuple[str, ...] = ("health", "check_health", "snapshot")
    require_start_method: bool = True
    require_stop_method: bool = True
    require_health_method: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.start_methods:
            raise ValueError("start_methods cannot be empty.")
        if not self.stop_methods:
            raise ValueError("stop_methods cannot be empty.")
        if not self.recover_methods:
            raise ValueError("recover_methods cannot be empty.")
        if not self.health_methods:
            raise ValueError("health_methods cannot be empty.")


@dataclass(frozen=True, slots=True)
class JarvisPhaseRuntimeBinding:
    kind: JarvisOrganKind
    name: str
    runtime: object | None = None
    import_path: str | None = None
    criticality: JarvisOrganCriticality = JarvisOrganCriticality.REQUIRED
    dependencies: tuple[JarvisOrganKind, ...] = ()
    lifecycle_policy: JarvisPhaseLifecyclePolicy = field(
        default_factory=JarvisPhaseLifecyclePolicy
    )
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("phase runtime binding name cannot be empty.")
        if self.runtime is None and not self.import_path:
            raise ValueError("phase runtime binding requires runtime or import_path.")
        if self.runtime is not None and self.import_path is not None:
            raise ValueError("provide runtime or import_path, not both.")


@dataclass(frozen=True, slots=True)
class JarvisPhaseAdapterSnapshot:
    kind: JarvisOrganKind
    name: str
    status: JarvisOrganStatus
    started: bool
    last_error: str | None
    last_latency_ms: float | None
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


class RuntimeFactory(Protocol):
    def __call__(self) -> object:
        raise NotImplementedError


class JarvisPhaseRuntimeResolver:
    def resolve(self, binding: JarvisPhaseRuntimeBinding) -> object:
        if binding.runtime is not None:
            return binding.runtime

        if binding.import_path is None:
            raise ValueError(f"{binding.kind.value} has no runtime binding.")

        return self._resolve_import_path(binding.import_path)

    def _resolve_import_path(self, import_path: str) -> object:
        if ":" not in import_path:
            raise ValueError(
                "runtime import_path must use 'module.path:factory_name'."
            )

        module_name, factory_name = import_path.split(":", 1)
        module = importlib.import_module(module_name)
        factory = getattr(module, factory_name)

        if not callable(factory):
            raise TypeError(f"{import_path} is not callable.")

        result: object = factory()
        return result


@dataclass(slots=True)
class JarvisPhaseOrganController:
    """
    Real Phase 1-9 organ adapter.

    It wraps actual phase runtime objects and exposes the Start Control
    contract: start, stop, recover, health.

    It does not generate speech and does not contain conversational text.
    """

    binding: JarvisPhaseRuntimeBinding
    resolver: JarvisPhaseRuntimeResolver = field(
        default_factory=JarvisPhaseRuntimeResolver
    )

    _runtime: object | None = None
    _status: JarvisOrganStatus = JarvisOrganStatus.CREATED
    _started: bool = False
    _last_error: str | None = None
    _last_latency_ms: float | None = None

    @property
    def kind(self) -> JarvisOrganKind:
        return self.binding.kind

    @property
    def name(self) -> str:
        return self.binding.name

    @property
    def criticality(self) -> JarvisOrganCriticality:
        return self.binding.criticality

    @property
    def dependencies(self) -> tuple[JarvisOrganKind, ...]:
        return self.binding.dependencies

    def start(self) -> JarvisOrganReport:
        started = time.perf_counter()
        self._status = JarvisOrganStatus.STARTING

        try:
            runtime = self._get_runtime()
            called = _call_first_present(
                runtime,
                self.binding.lifecycle_policy.start_methods,
            )

            if not called and self.binding.lifecycle_policy.require_start_method:
                raise RuntimeError("no start-compatible method found")

            self._started = True
            self._status = JarvisOrganStatus.RUNNING
            message = "phase organ started"
        except Exception as exc:
            self._started = False
            self._status = JarvisOrganStatus.FAILED
            self._last_error = str(exc)
            message = "phase organ start failed"

        return self._report(
            operation=JarvisStartControlOperation.START_ALL,
            message=message,
            started=started,
        )

    def stop(self) -> JarvisOrganReport:
        started = time.perf_counter()
        self._status = JarvisOrganStatus.STOPPING

        try:
            runtime = self._get_runtime()
            called = _call_first_present(
                runtime,
                self.binding.lifecycle_policy.stop_methods,
            )

            if not called and self.binding.lifecycle_policy.require_stop_method:
                raise RuntimeError("no stop-compatible method found")

            self._started = False
            self._status = JarvisOrganStatus.STOPPED
            message = "phase organ stopped"
        except Exception as exc:
            self._status = JarvisOrganStatus.FAILED
            self._last_error = str(exc)
            message = "phase organ stop failed"

        return self._report(
            operation=JarvisStartControlOperation.STOP_ALL,
            message=message,
            started=started,
        )

    def recover(self) -> JarvisOrganReport:
        started = time.perf_counter()

        try:
            runtime = self._get_runtime()
            called = _call_first_present(
                runtime,
                self.binding.lifecycle_policy.recover_methods,
            )

            if not called:
                self.stop()
                self.start()

            self._started = True
            self._status = JarvisOrganStatus.RUNNING
            message = "phase organ recovered"
        except Exception as exc:
            self._status = JarvisOrganStatus.FAILED
            self._last_error = str(exc)
            message = "phase organ recovery failed"

        return self._report(
            operation=JarvisStartControlOperation.RECOVER,
            message=message,
            started=started,
        )

    def health(self) -> JarvisOrganHealth:
        started = time.perf_counter()
        status = self._status

        try:
            runtime = self._get_runtime()
            health_value = _call_first_present_with_result(
                runtime,
                self.binding.lifecycle_policy.health_methods,
            )

            extracted = _extract_organ_status(health_value)
            if extracted is not None:
                status = extracted
            elif (
                self.binding.lifecycle_policy.require_health_method
                and health_value is None
            ):
                status = JarvisOrganStatus.FAILED
        except Exception as exc:
            status = JarvisOrganStatus.FAILED
            self._last_error = str(exc)

        return JarvisOrganHealth(
            kind=self.kind,
            name=self.name,
            status=status,
            criticality=self.criticality,
            message="phase organ health checked",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            created_at=utc_now(),
            metadata={
                "started": self._started,
                "last_error": self._last_error,
                **self.binding.metadata,
            },
        )

    def snapshot(self) -> JarvisPhaseAdapterSnapshot:
        return JarvisPhaseAdapterSnapshot(
            kind=self.kind,
            name=self.name,
            status=self._status,
            started=self._started,
            last_error=self._last_error,
            last_latency_ms=self._last_latency_ms,
            created_at=utc_now(),
            metadata=self.binding.metadata,
        )

    def _get_runtime(self) -> object:
        if self._runtime is None:
            self._runtime = self.resolver.resolve(self.binding)
        return self._runtime

    def _report(
        self,
        *,
        operation: JarvisStartControlOperation,
        message: str,
        started: float,
    ) -> JarvisOrganReport:
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._last_latency_ms = latency_ms

        return JarvisOrganReport(
            kind=self.kind,
            name=self.name,
            status=self._status,
            criticality=self.criticality,
            operation=operation,
            message=message,
            latency_ms=latency_ms,
            created_at=utc_now(),
            metadata={
                "started": self._started,
                "last_error": self._last_error,
                **self.binding.metadata,
            },
        )


@dataclass(frozen=True, slots=True)
class JarvisConnectedRuntimePlan:
    bindings: tuple[JarvisPhaseRuntimeBinding, ...]
    voice_launcher: object
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


def build_phase_organ_controller(
    binding: JarvisPhaseRuntimeBinding,
) -> JarvisPhaseOrganController:
    return JarvisPhaseOrganController(binding=binding)


def build_connected_start_control_from_plan(
    plan: JarvisConnectedRuntimePlan,
    *,
    config: JarvisStartControlConfig | None = None,
) -> JarvisStartControlRuntime:
    controllers: list[JarvisOrganController] = [
        build_phase_organ_controller(binding) for binding in plan.bindings
    ]
    controllers.append(VoiceLauncherOrganController(launcher=plan.voice_launcher))
    return JarvisStartControlRuntime(organs=tuple(controllers), config=config)


def build_connected_runtime_plan(
    *,
    phase_runtimes: Mapping[JarvisOrganKind, object],
    voice_launcher: object,
) -> JarvisConnectedRuntimePlan:
    required_specs = default_phase_runtime_specs()
    missing = tuple(
        kind.value for kind in required_specs if kind not in phase_runtimes
    )

    if missing:
        raise ValueError(
            "missing real phase runtime bindings: " + ", ".join(missing)
        )

    bindings = tuple(
        JarvisPhaseRuntimeBinding(
            kind=kind,
            name=_phase_name(kind),
            runtime=phase_runtimes[kind],
            dependencies=_phase_dependencies(kind),
            criticality=JarvisOrganCriticality.REQUIRED,
            metadata={"phase_binding": "real_runtime_object"},
        )
        for kind in required_specs
    )

    return JarvisConnectedRuntimePlan(
        bindings=bindings,
        voice_launcher=voice_launcher,
        created_at=utc_now(),
        metadata={"binding_count": len(bindings)},
    )


def default_phase_runtime_specs() -> tuple[JarvisOrganKind, ...]:
    return (
        JarvisOrganKind.PHASE1_KERNEL,
        JarvisOrganKind.PHASE1_EVENTS,
        JarvisOrganKind.PHASE1_OBSERVABILITY,
        JarvisOrganKind.PHASE2_PRESENCE,
        JarvisOrganKind.PHASE2_VOICE,
        JarvisOrganKind.PHASE3_COGNITION,
        JarvisOrganKind.PHASE4_MEMORY,
        JarvisOrganKind.PHASE5_TOOLS,
        JarvisOrganKind.PHASE6_ORCHESTRATION,
        JarvisOrganKind.PHASE7_STREAMING_LATENCY,
        JarvisOrganKind.PHASE8_ENVIRONMENT,
        JarvisOrganKind.PHASE9_COGNITIVE_SESSION,
    )


def _phase_name(kind: JarvisOrganKind) -> str:
    names = {
        JarvisOrganKind.PHASE1_KERNEL: "phase1_kernel",
        JarvisOrganKind.PHASE1_EVENTS: "phase1_events",
        JarvisOrganKind.PHASE1_OBSERVABILITY: "phase1_observability",
        JarvisOrganKind.PHASE2_PRESENCE: "phase2_presence",
        JarvisOrganKind.PHASE2_VOICE: "phase2_voice",
        JarvisOrganKind.PHASE3_COGNITION: "phase3_cognition",
        JarvisOrganKind.PHASE4_MEMORY: "phase4_memory_gateway",
        JarvisOrganKind.PHASE5_TOOLS: "phase5_tool_action_runtime",
        JarvisOrganKind.PHASE6_ORCHESTRATION: "phase6_orchestration",
        JarvisOrganKind.PHASE7_STREAMING_LATENCY: "phase7_streaming_latency",
        JarvisOrganKind.PHASE8_ENVIRONMENT: "phase8_environment_awareness",
        JarvisOrganKind.PHASE9_COGNITIVE_SESSION: (
            "phase9_cognitive_session_goals_personality"
        ),
    }
    return names[kind]


def _phase_dependencies(kind: JarvisOrganKind) -> tuple[JarvisOrganKind, ...]:
    dependencies = {
        JarvisOrganKind.PHASE1_KERNEL: (),
        JarvisOrganKind.PHASE1_EVENTS: (JarvisOrganKind.PHASE1_KERNEL,),
        JarvisOrganKind.PHASE1_OBSERVABILITY: (
            JarvisOrganKind.PHASE1_EVENTS,
        ),
        JarvisOrganKind.PHASE2_PRESENCE: (
            JarvisOrganKind.PHASE1_OBSERVABILITY,
        ),
        JarvisOrganKind.PHASE2_VOICE: (
            JarvisOrganKind.PHASE2_PRESENCE,
            JarvisOrganKind.PHASE1_OBSERVABILITY,
        ),
        JarvisOrganKind.PHASE3_COGNITION: (
            JarvisOrganKind.PHASE1_OBSERVABILITY,
        ),
        JarvisOrganKind.PHASE4_MEMORY: (
            JarvisOrganKind.PHASE1_OBSERVABILITY,
        ),
        JarvisOrganKind.PHASE5_TOOLS: (
            JarvisOrganKind.PHASE1_OBSERVABILITY,
        ),
        JarvisOrganKind.PHASE6_ORCHESTRATION: (
            JarvisOrganKind.PHASE1_OBSERVABILITY,
            JarvisOrganKind.PHASE3_COGNITION,
            JarvisOrganKind.PHASE4_MEMORY,
            JarvisOrganKind.PHASE5_TOOLS,
        ),
        JarvisOrganKind.PHASE7_STREAMING_LATENCY: (
            JarvisOrganKind.PHASE6_ORCHESTRATION,
            JarvisOrganKind.PHASE2_VOICE,
        ),
        JarvisOrganKind.PHASE8_ENVIRONMENT: (
            JarvisOrganKind.PHASE6_ORCHESTRATION,
        ),
        JarvisOrganKind.PHASE9_COGNITIVE_SESSION: (
            JarvisOrganKind.PHASE3_COGNITION,
            JarvisOrganKind.PHASE4_MEMORY,
            JarvisOrganKind.PHASE6_ORCHESTRATION,
        ),
    }
    return dependencies[kind]


def _call_first_present(runtime: object, method_names: Sequence[str]) -> bool:
    for method_name in method_names:
        method = getattr(runtime, method_name, None)
        if callable(method):
            result: object | None = method()
            del result
            return True
    return False


def _call_first_present_with_result(
    runtime: object,
    method_names: Sequence[str],
) -> object | None:
    for method_name in method_names:
        method = getattr(runtime, method_name, None)
        if callable(method):
            result: object | None = method()
            return result
    return None


def _extract_organ_status(value: object | None) -> JarvisOrganStatus | None:
    if value is None:
        return None

    if isinstance(value, JarvisOrganStatus):
        return value

    status = getattr(value, "status", None)

    if isinstance(status, JarvisOrganStatus):
        return status

    if isinstance(status, StrEnum):
        return _status_from_string(status.value)

    if isinstance(status, str):
        return _status_from_string(status)

    return None


def _status_from_string(value: str) -> JarvisOrganStatus | None:
    normalized = value.strip().casefold()

    direct = {
        status.value: status for status in JarvisOrganStatus
    }
    if normalized in direct:
        return direct[normalized]

    aliases = {
        "ready": JarvisOrganStatus.RUNNING,
        "healthy": JarvisOrganStatus.RUNNING,
        "ok": JarvisOrganStatus.RUNNING,
        "active": JarvisOrganStatus.RUNNING,
        "degraded": JarvisOrganStatus.DEGRADED,
        "failed": JarvisOrganStatus.FAILED,
        "error": JarvisOrganStatus.FAILED,
        "stopped": JarvisOrganStatus.STOPPED,
    }
    return aliases.get(normalized)


def read_runtime_binding_imports(path: Path) -> dict[JarvisOrganKind, str]:
    """
    Reads a simple binding file.

    Format:
        phase1_kernel=some.module:create_kernel
        phase1_events=some.module:create_events

    This keeps real runtime construction outside Start Control.
    """
    bindings: dict[JarvisOrganKind, str] = {}

    if not path.exists():
        raise FileNotFoundError(path)

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if "=" not in line:
            raise ValueError(f"invalid binding line: {raw_line}")

        key, import_path = line.split("=", 1)
        kind = JarvisOrganKind(key.strip())
        bindings[kind] = import_path.strip()

    return bindings


def build_plan_from_import_bindings(
    *,
    import_bindings: Mapping[JarvisOrganKind, str],
    voice_launcher: object,
) -> JarvisConnectedRuntimePlan:
    required_specs = default_phase_runtime_specs()
    missing = tuple(
        kind.value for kind in required_specs if kind not in import_bindings
    )

    if missing:
        raise ValueError(
            "missing import bindings for real phase runtimes: "
            + ", ".join(missing)
        )

    bindings = tuple(
        JarvisPhaseRuntimeBinding(
            kind=kind,
            name=_phase_name(kind),
            import_path=import_bindings[kind],
            dependencies=_phase_dependencies(kind),
            criticality=JarvisOrganCriticality.REQUIRED,
            metadata={"phase_binding": "import_path"},
        )
        for kind in required_specs
    )

    return JarvisConnectedRuntimePlan(
        bindings=bindings,
        voice_launcher=voice_launcher,
        created_at=utc_now(),
        metadata={"binding_count": len(bindings)},
    )