from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from jarvis.runtime.start_control import JarvisOrganStatus, utc_now


@dataclass(frozen=True, slots=True)
class PhaseRuntimeHealth:
    status: JarvisOrganStatus
    name: str
    checked_modules: tuple[str, ...]
    missing_modules: tuple[str, ...]
    checked_paths: tuple[str, ...]
    missing_paths: tuple[str, ...]
    latency_ms: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ModuleBackedPhaseRuntime:
    """
    Real phase runtime binding adapter.

    This object represents an already-built phase package as a supervised
    runtime organ. It verifies that the required real modules/files exist.
    It does not generate speech and does not contain conversational output.
    """

    name: str
    required_modules: tuple[str, ...]
    required_paths: tuple[Path, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    _started: bool = False
    _last_health: PhaseRuntimeHealth | None = None

    def start(self) -> None:
        health = self.health()
        if health.status == JarvisOrganStatus.FAILED:
            missing = ", ".join(
                (*health.missing_modules, *health.missing_paths)
            )
            raise RuntimeError(f"{self.name} missing required bindings: {missing}")
        self._started = True

    def stop(self) -> None:
        self._started = False

    def recover(self) -> None:
        self.start()

    def health(self) -> PhaseRuntimeHealth:
        started = time.perf_counter()
        missing_modules: list[str] = []
        missing_paths: list[str] = []

        for module_name in self.required_modules:
            try:
                importlib.import_module(module_name)
            except Exception:
                missing_modules.append(module_name)

        for path in self.required_paths:
            if not path.exists():
                missing_paths.append(str(path))

        status = (
            JarvisOrganStatus.FAILED
            if missing_modules or missing_paths
            else JarvisOrganStatus.RUNNING
        )

        health = PhaseRuntimeHealth(
            status=status,
            name=self.name,
            checked_modules=self.required_modules,
            missing_modules=tuple(missing_modules),
            checked_paths=tuple(str(path) for path in self.required_paths),
            missing_paths=tuple(missing_paths),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            created_at=utc_now(),
            metadata={
                "started": self._started,
                **self.metadata,
            },
        )
        self._last_health = health
        return health


def create_phase1_kernel_runtime() -> ModuleBackedPhaseRuntime:
    return ModuleBackedPhaseRuntime(
        name="phase1_kernel",
        required_modules=(
            "jarvis.runtime.kernel.runtime_kernel",
            "jarvis.runtime.kernel.lifecycle_manager",
            "jarvis.runtime.kernel.scheduler",
            "jarvis.runtime.kernel.cancellation_manager",
            "jarvis.runtime.state.state_engine",
        ),
        metadata={"phase": "phase1", "organ": "kernel"},
    )


def create_phase1_events_runtime() -> ModuleBackedPhaseRuntime:
    return ModuleBackedPhaseRuntime(
        name="phase1_events",
        required_modules=(
            "jarvis.runtime.events.event_bus",
            "jarvis.runtime.events.event_models",
            "jarvis.runtime.events.priorities",
            "jarvis.runtime.events.subscriptions",
        ),
        metadata={"phase": "phase1", "organ": "events"},
    )


def create_phase1_observability_runtime() -> ModuleBackedPhaseRuntime:
    return ModuleBackedPhaseRuntime(
        name="phase1_observability",
        required_modules=(
            "jarvis.runtime.observability.metrics",
            "jarvis.runtime.observability.performance_monitor",
            "jarvis.runtime.observability.structured_logger",
            "jarvis.runtime.observability.tracing",
        ),
        metadata={"phase": "phase1", "organ": "observability"},
    )


def create_phase2_presence_runtime() -> ModuleBackedPhaseRuntime:
    return ModuleBackedPhaseRuntime(
        name="phase2_presence",
        required_modules=(
            "jarvis.presence.presence_engine",
            "jarvis.presence.state.turn_state_machine",
            "jarvis.presence.workers.voice_input_worker",
            "jarvis.presence.workers.vad_worker",
            "jarvis.presence.workers.stt_worker",
            "jarvis.presence.workers.tts_worker",
            "jarvis.presence.workers.audio_playback_worker",
            "jarvis.presence.workers.interruption_worker",
        ),
        metadata={"phase": "phase2", "organ": "presence"},
    )


def create_phase2_voice_runtime() -> ModuleBackedPhaseRuntime:
    return ModuleBackedPhaseRuntime(
        name="phase2_voice",
        required_modules=(
            "jarvis.voice.microphone_capture",
            "jarvis.voice.voice_activity",
            "jarvis.voice.stt_runtime",
            "jarvis.voice.cognition_response",
            "jarvis.voice.tts_runtime",
            "jarvis.voice.playback_runtime",
            "jarvis.voice.barge_in_runtime",
            "jarvis.voice.health_recovery",
            "jarvis.voice.session_loop",
            "jarvis.voice.runtime_launcher",
        ),
        metadata={"phase": "phase2_step51", "organ": "voice"},
    )


def create_phase3_cognition_runtime() -> ModuleBackedPhaseRuntime:
    return ModuleBackedPhaseRuntime(
        name="phase3_cognition",
        required_modules=(
            "jarvis.cognition.engine",
            "jarvis.cognition.ollama_backend",
            "jarvis.cognition.local_llm_adapter",
            "jarvis.cognition.planning",
            "jarvis.cognition.response_bridge",
            "jarvis.cognition.streaming",
            "jarvis.cognition.worker",
        ),
        metadata={"phase": "phase3", "organ": "cognition"},
    )


def create_phase4_memory_runtime() -> ModuleBackedPhaseRuntime:
    return ModuleBackedPhaseRuntime(
        name="phase4_memory",
        required_modules=(
            "jarvis.memory.gateway",
            "jarvis.memory.retrieval",
            "jarvis.memory.semantic",
            "jarvis.memory.episodic",
            "jarvis.memory.profile",
            "jarvis.memory.sqlite_store",
            "jarvis.memory.vector",
            "jarvis.memory.write_policy",
            "jarvis.memory.privacy_policy",
        ),
        metadata={"phase": "phase4", "organ": "memory"},
    )


def create_phase5_tools_runtime() -> ModuleBackedPhaseRuntime:
    return ModuleBackedPhaseRuntime(
        name="phase5_tools",
        required_modules=(
            "jarvis.tools.registry",
            "jarvis.tools.policy",
            "jarvis.tools.validation",
            "jarvis.tools.execution",
            "jarvis.tools.planner",
            "jarvis.tools.shell",
            "jarvis.tools.filesystem",
            "jarvis.tools.browser",
            "jarvis.tools.ide",
            "jarvis.tools.interruption",
            "jarvis.tools.audit",
            "jarvis.tools.approval",
        ),
        metadata={"phase": "phase5", "organ": "tools"},
    )


def create_phase6_orchestration_runtime() -> ModuleBackedPhaseRuntime:
    return ModuleBackedPhaseRuntime(
        name="phase6_orchestration",
        required_modules=(
            "jarvis.orchestration.registry",
            "jarvis.orchestration.scheduler",
            "jarvis.orchestration.interrupts",
            "jarvis.orchestration.recovery",
            "jarvis.orchestration.load_manager",
            "jarvis.orchestration.circuit_breakers",
            "jarvis.orchestration.deadlocks",
            "jarvis.orchestration.coordination",
            "jarvis.orchestration.observability",
            "jarvis.orchestration.proactive",
        ),
        metadata={"phase": "phase6", "organ": "orchestration"},
    )


def create_phase7_streaming_latency_runtime() -> ModuleBackedPhaseRuntime:
    return ModuleBackedPhaseRuntime(
        name="phase7_streaming_latency",
        required_modules=(
            "jarvis.latency.budgets",
            "jarvis.latency.profiler",
            "jarvis.latency.token_streaming",
            "jarvis.latency.streaming_stt",
            "jarvis.latency.streaming_tts",
            "jarvis.latency.streaming_memory",
            "jarvis.latency.predictive_context",
            "jarvis.latency.parallel_pipeline",
            "jarvis.latency.interruption_recovery",
            "jarvis.latency.response_naturalness",
        ),
        metadata={"phase": "phase7", "organ": "streaming_latency"},
    )


def create_phase8_environment_runtime() -> ModuleBackedPhaseRuntime:
    return ModuleBackedPhaseRuntime(
        name="phase8_environment",
        required_modules=(
            "jarvis.environment.capture",
            "jarvis.environment.ocr",
            "jarvis.environment.state_runtime",
            "jarvis.environment.timeline",
            "jarvis.environment.trust_runtime",
            "jarvis.environment.environment_fusion",
            "jarvis.environment.workspace_graph",
            "jarvis.environment.visual_grounding",
            "jarvis.environment.intent_persistence",
            "jarvis.environment.recovery_runtime",
        ),
        metadata={"phase": "phase8", "organ": "environment"},
    )


def create_phase9_cognitive_session_runtime() -> ModuleBackedPhaseRuntime:
    return ModuleBackedPhaseRuntime(
        name="phase9_cognitive_session",
        required_modules=(
            "jarvis.cognitive.attention",
            "jarvis.cognitive.working_memory",
            "jarvis.cognitive.goals",
            "jarvis.cognitive.planning",
            "jarvis.cognitive.personality",
            "jarvis.cognitive.session",
            "jarvis.cognitive.integration",
        ),
        metadata={"phase": "phase9", "organ": "cognitive_session"},
    )