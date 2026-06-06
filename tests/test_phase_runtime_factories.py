from __future__ import annotations

from jarvis.runtime import (
    JarvisOrganStatus,
    ModuleBackedPhaseRuntime,
    create_phase1_events_runtime,
    create_phase1_kernel_runtime,
    create_phase1_observability_runtime,
    create_phase2_presence_runtime,
    create_phase2_voice_runtime,
    create_phase3_cognition_runtime,
    create_phase4_memory_runtime,
    create_phase5_tools_runtime,
    create_phase6_orchestration_runtime,
    create_phase7_streaming_latency_runtime,
    create_phase8_environment_runtime,
    create_phase9_cognitive_session_runtime,
)


def test_module_backed_phase_runtime_health_passes_for_existing_modules() -> None:
    runtime = ModuleBackedPhaseRuntime(
        name="test_runtime",
        required_modules=("jarvis.runtime.start_control",),
    )

    health = runtime.health()

    assert health.status == JarvisOrganStatus.RUNNING
    assert health.missing_modules == ()


def test_module_backed_phase_runtime_health_fails_for_missing_modules() -> None:
    runtime = ModuleBackedPhaseRuntime(
        name="test_runtime",
        required_modules=("jarvis.missing.module",),
    )

    health = runtime.health()

    assert health.status == JarvisOrganStatus.FAILED
    assert health.missing_modules == ("jarvis.missing.module",)


def test_module_backed_phase_runtime_start_requires_health() -> None:
    runtime = ModuleBackedPhaseRuntime(
        name="test_runtime",
        required_modules=("jarvis.runtime.start_control",),
    )

    runtime.start()
    health = runtime.health()

    assert health.status == JarvisOrganStatus.RUNNING
    assert health.metadata["started"] is True


def test_all_phase_factories_create_healthy_module_backed_runtimes() -> None:
    factories = (
        create_phase1_kernel_runtime,
        create_phase1_events_runtime,
        create_phase1_observability_runtime,
        create_phase2_presence_runtime,
        create_phase2_voice_runtime,
        create_phase3_cognition_runtime,
        create_phase4_memory_runtime,
        create_phase5_tools_runtime,
        create_phase6_orchestration_runtime,
        create_phase7_streaming_latency_runtime,
        create_phase8_environment_runtime,
        create_phase9_cognitive_session_runtime,
    )

    for factory in factories:
        runtime = factory()
        assert isinstance(runtime, ModuleBackedPhaseRuntime)
        assert runtime.health().status == JarvisOrganStatus.RUNNING