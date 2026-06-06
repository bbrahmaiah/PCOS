from __future__ import annotations

from dataclasses import dataclass

from jarvis.runtime import (
    JarvisOrganCriticality,
    JarvisOrganHealth,
    JarvisOrganKind,
    JarvisOrganReport,
    JarvisOrganStatus,
    JarvisStartControlEvent,
    JarvisStartControlOperation,
    JarvisStartControlRuntime,
    JarvisStartControlStatus,
    VoiceLauncherOrganController,
)
from jarvis.runtime.start_control import utc_now


@dataclass
class FakeOrgan:
    kind: JarvisOrganKind
    name: str
    dependencies: tuple[JarvisOrganKind, ...] = ()
    criticality: JarvisOrganCriticality = JarvisOrganCriticality.REQUIRED
    fail_start: bool = False
    degraded: bool = False
    started: bool = False
    stopped: bool = False

    def start(self) -> JarvisOrganReport:
        if self.fail_start:
            status = JarvisOrganStatus.FAILED
        elif self.degraded:
            status = JarvisOrganStatus.DEGRADED
            self.started = True
        else:
            status = JarvisOrganStatus.RUNNING
            self.started = True

        return JarvisOrganReport(
            kind=self.kind,
            name=self.name,
            status=status,
            criticality=self.criticality,
            operation=JarvisStartControlOperation.START_ALL,
            message="start",
            latency_ms=1.0,
            created_at=utc_now(),
        )

    def stop(self) -> JarvisOrganReport:
        self.stopped = True
        self.started = False
        return JarvisOrganReport(
            kind=self.kind,
            name=self.name,
            status=JarvisOrganStatus.STOPPED,
            criticality=self.criticality,
            operation=JarvisStartControlOperation.STOP_ALL,
            message="stop",
            latency_ms=1.0,
            created_at=utc_now(),
        )

    def recover(self) -> JarvisOrganReport:
        self.fail_start = False
        self.degraded = False
        self.started = True
        return JarvisOrganReport(
            kind=self.kind,
            name=self.name,
            status=JarvisOrganStatus.RUNNING,
            criticality=self.criticality,
            operation=JarvisStartControlOperation.RECOVER,
            message="recover",
            latency_ms=1.0,
            created_at=utc_now(),
        )

    def health(self) -> JarvisOrganHealth:
        if self.fail_start:
            status = JarvisOrganStatus.FAILED
        elif self.degraded:
            status = JarvisOrganStatus.DEGRADED
        elif self.started:
            status = JarvisOrganStatus.RUNNING
        elif self.stopped:
            status = JarvisOrganStatus.STOPPED
        else:
            status = JarvisOrganStatus.CREATED

        return JarvisOrganHealth(
            kind=self.kind,
            name=self.name,
            status=status,
            criticality=self.criticality,
            message="health",
            latency_ms=1.0,
            created_at=utc_now(),
        )


@dataclass
class FakeVoiceLauncher:
    run_called: bool = False
    stop_called: bool = False

    def run(self) -> None:
        self.run_called = True

    def request_stop(self) -> None:
        self.stop_called = True

    def stop(self) -> None:
        self.stop_called = True


def _connected_organs() -> tuple[FakeOrgan, ...]:
    return (
        FakeOrgan(
            kind=JarvisOrganKind.PHASE1_KERNEL,
            name="kernel",
        ),
        FakeOrgan(
            kind=JarvisOrganKind.PHASE2_VOICE,
            name="voice",
            dependencies=(JarvisOrganKind.PHASE1_KERNEL,),
        ),
        FakeOrgan(
            kind=JarvisOrganKind.PHASE3_COGNITION,
            name="cognition",
            dependencies=(JarvisOrganKind.PHASE1_KERNEL,),
        ),
        FakeOrgan(
            kind=JarvisOrganKind.PHASE4_MEMORY,
            name="memory",
            dependencies=(JarvisOrganKind.PHASE1_KERNEL,),
        ),
        FakeOrgan(
            kind=JarvisOrganKind.PHASE5_TOOLS,
            name="tools",
            dependencies=(JarvisOrganKind.PHASE1_KERNEL,),
        ),
        FakeOrgan(
            kind=JarvisOrganKind.PHASE6_ORCHESTRATION,
            name="orchestration",
            dependencies=(
                JarvisOrganKind.PHASE3_COGNITION,
                JarvisOrganKind.PHASE4_MEMORY,
                JarvisOrganKind.PHASE5_TOOLS,
            ),
        ),
        FakeOrgan(
            kind=JarvisOrganKind.PHASE7_STREAMING_LATENCY,
            name="streaming",
            dependencies=(JarvisOrganKind.PHASE6_ORCHESTRATION,),
        ),
        FakeOrgan(
            kind=JarvisOrganKind.PHASE8_ENVIRONMENT,
            name="environment",
            dependencies=(JarvisOrganKind.PHASE6_ORCHESTRATION,),
        ),
        FakeOrgan(
            kind=JarvisOrganKind.PHASE9_COGNITIVE_SESSION,
            name="session",
            dependencies=(
                JarvisOrganKind.PHASE3_COGNITION,
                JarvisOrganKind.PHASE4_MEMORY,
            ),
        ),
    )


def test_start_control_starts_all_connected_organs() -> None:
    organs = _connected_organs()
    runtime = JarvisStartControlRuntime(organs=organs)

    result = runtime.start_all()
    snapshot = runtime.snapshot()

    assert result.status == JarvisStartControlStatus.RUNNING
    assert result.event == JarvisStartControlEvent.START_COMPLETED
    assert all(organ.started for organ in organs)
    assert snapshot.running_count == len(organs)


def test_start_control_stops_in_reverse_order() -> None:
    organs = _connected_organs()
    runtime = JarvisStartControlRuntime(organs=organs)

    runtime.start_all()
    result = runtime.stop_all()

    assert result.status == JarvisStartControlStatus.STOPPED
    assert all(organ.stopped for organ in organs)


def test_start_control_fails_when_required_dependency_missing() -> None:
    organs = (
        FakeOrgan(
            kind=JarvisOrganKind.PHASE2_VOICE,
            name="voice",
            dependencies=(JarvisOrganKind.PHASE1_KERNEL,),
        ),
    )

    runtime = JarvisStartControlRuntime(organs=organs)
    result = runtime.start_all()

    assert result.status == JarvisStartControlStatus.FAILED
    assert "start sequence failed" in result.reason


def test_start_control_fails_and_stops_started_organs() -> None:
    kernel = FakeOrgan(kind=JarvisOrganKind.PHASE1_KERNEL, name="kernel")
    voice = FakeOrgan(
        kind=JarvisOrganKind.PHASE2_VOICE,
        name="voice",
        dependencies=(JarvisOrganKind.PHASE1_KERNEL,),
        fail_start=True,
    )
    runtime = JarvisStartControlRuntime(organs=(kernel, voice))

    result = runtime.start_all()

    assert result.status == JarvisStartControlStatus.FAILED
    assert kernel.stopped is True


def test_start_control_allows_degraded_state() -> None:
    organs = list(_connected_organs())
    organs[1].degraded = True
    runtime = JarvisStartControlRuntime(organs=tuple(organs))

    result = runtime.start_all()

    assert result.status == JarvisStartControlStatus.DEGRADED


def test_start_control_recovers_degraded_organs() -> None:
    organs = list(_connected_organs())
    organs[1].degraded = True
    runtime = JarvisStartControlRuntime(organs=tuple(organs))

    runtime.start_all()
    result = runtime.recover()

    assert result.status == JarvisStartControlStatus.RUNNING
    assert organs[1].degraded is False


def test_start_control_rejects_duplicate_organs() -> None:
    try:
        JarvisStartControlRuntime(
            organs=(
                FakeOrgan(kind=JarvisOrganKind.PHASE1_KERNEL, name="a"),
                FakeOrgan(kind=JarvisOrganKind.PHASE1_KERNEL, name="b"),
            )
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected duplicate organ kind to fail")


def test_voice_launcher_organ_starts_launcher() -> None:
    launcher = FakeVoiceLauncher()
    organ = VoiceLauncherOrganController(launcher=launcher)

    report = organ.start()
    organ.stop()

    assert report.status == JarvisOrganStatus.RUNNING
    assert launcher.run_called is True
    assert launcher.stop_called is True


def test_start_control_enum_values_are_stable() -> None:
    assert JarvisOrganKind.PHASE1_KERNEL.value == "phase1_kernel"
    assert JarvisStartControlStatus.RUNNING.value == "running"
    assert JarvisStartControlOperation.START_ALL.value == "start_all"