from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jarvis.runtime import (
    JarvisOrganKind,
    JarvisOrganStatus,
    JarvisPhaseLifecyclePolicy,
    JarvisPhaseOrganController,
    JarvisPhaseRuntimeBinding,
    build_connected_runtime_plan,
    build_connected_start_control_from_plan,
    default_phase_runtime_specs,
    read_runtime_binding_imports,
)


@dataclass
class FakeHealth:
    status: str


@dataclass
class FakeRuntime:
    started: bool = False
    stopped: bool = False
    recovered: bool = False
    fail_start: bool = False

    def start(self) -> None:
        if self.fail_start:
            raise RuntimeError("start failed")
        self.started = True

    def stop(self) -> None:
        self.stopped = True
        self.started = False

    def recover(self) -> None:
        self.recovered = True
        self.started = True

    def health(self) -> FakeHealth:
        return FakeHealth(status="running" if self.started else "stopped")


@dataclass
class FakeVoiceLauncher:
    started: bool = False
    stopped: bool = False

    def run(self) -> None:
        self.started = True

    def request_stop(self) -> None:
        self.stopped = True

    def stop(self) -> None:
        self.stopped = True


def _phase_runtimes() -> dict[JarvisOrganKind, object]:
    return {kind: FakeRuntime() for kind in default_phase_runtime_specs()}


def test_phase_organ_controller_starts_real_runtime() -> None:
    runtime = FakeRuntime()
    controller = JarvisPhaseOrganController(
        binding=JarvisPhaseRuntimeBinding(
            kind=JarvisOrganKind.PHASE1_KERNEL,
            name="kernel",
            runtime=runtime,
        )
    )

    result = controller.start()

    assert result.status == JarvisOrganStatus.RUNNING
    assert runtime.started is True
    assert controller.health().status == JarvisOrganStatus.RUNNING


def test_phase_organ_controller_stops_real_runtime() -> None:
    runtime = FakeRuntime()
    controller = JarvisPhaseOrganController(
        binding=JarvisPhaseRuntimeBinding(
            kind=JarvisOrganKind.PHASE1_KERNEL,
            name="kernel",
            runtime=runtime,
        )
    )

    controller.start()
    result = controller.stop()

    assert result.status == JarvisOrganStatus.STOPPED
    assert runtime.stopped is True


def test_phase_organ_controller_fails_when_start_missing() -> None:
    controller = JarvisPhaseOrganController(
        binding=JarvisPhaseRuntimeBinding(
            kind=JarvisOrganKind.PHASE1_KERNEL,
            name="kernel",
            runtime=object(),
        )
    )

    result = controller.start()

    assert result.status == JarvisOrganStatus.FAILED


def test_phase_organ_controller_can_allow_missing_start_for_optional_runtime() -> None:
    controller = JarvisPhaseOrganController(
        binding=JarvisPhaseRuntimeBinding(
            kind=JarvisOrganKind.PHASE1_KERNEL,
            name="kernel",
            runtime=object(),
            lifecycle_policy=JarvisPhaseLifecyclePolicy(
                require_start_method=False,
                require_stop_method=False,
            ),
        )
    )

    result = controller.start()

    assert result.status == JarvisOrganStatus.RUNNING


def test_build_connected_runtime_plan_requires_all_phase_runtimes() -> None:
    runtimes = _phase_runtimes()
    runtimes.pop(JarvisOrganKind.PHASE4_MEMORY)

    try:
        build_connected_runtime_plan(
            phase_runtimes=runtimes,
            voice_launcher=FakeVoiceLauncher(),
        )
    except ValueError as exc:
        assert "phase4_memory" in str(exc)
    else:
        raise AssertionError("expected missing phase runtime to fail")


def test_build_connected_start_control_from_real_plan() -> None:
    runtimes = _phase_runtimes()
    voice_launcher = FakeVoiceLauncher()
    plan = build_connected_runtime_plan(
        phase_runtimes=runtimes,
        voice_launcher=voice_launcher,
    )
    start_control = build_connected_start_control_from_plan(plan)

    result = start_control.start_all()

    assert result.succeeded is True
    assert voice_launcher.started is True
    assert all(
        isinstance(runtime, FakeRuntime) and runtime.started
        for runtime in runtimes.values()
    )


def test_read_runtime_binding_imports(tmp_path: Path) -> None:
    path = tmp_path / "runtime_bindings.env"
    path.write_text(
        "\n".join(
            (
                "phase1_kernel=some.module:create_kernel",
                "phase1_events=some.module:create_events",
            )
        ),
        encoding="utf-8",
    )

    bindings = read_runtime_binding_imports(path)

    assert bindings[JarvisOrganKind.PHASE1_KERNEL] == (
        "some.module:create_kernel"
    )
    assert bindings[JarvisOrganKind.PHASE1_EVENTS] == (
        "some.module:create_events"
    )


def test_default_phase_runtime_specs_include_all_core_organs() -> None:
    specs = default_phase_runtime_specs()

    assert JarvisOrganKind.PHASE1_KERNEL in specs
    assert JarvisOrganKind.PHASE2_VOICE in specs
    assert JarvisOrganKind.PHASE3_COGNITION in specs
    assert JarvisOrganKind.PHASE4_MEMORY in specs
    assert JarvisOrganKind.PHASE5_TOOLS in specs
    assert JarvisOrganKind.PHASE6_ORCHESTRATION in specs
    assert JarvisOrganKind.PHASE7_STREAMING_LATENCY in specs
    assert JarvisOrganKind.PHASE8_ENVIRONMENT in specs
    assert JarvisOrganKind.PHASE9_COGNITIVE_SESSION in specs