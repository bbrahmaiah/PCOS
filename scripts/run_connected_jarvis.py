from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.runtime import (  # noqa: E402
    JarvisOrganKind,
    JarvisStartControlRuntime,
    ManagedJarvisOrganController,
    VoiceLauncherOrganController,
)
from jarvis.voice import (  # noqa: E402
    VoiceRuntimeLauncher,
    VoiceRuntimeLauncherConfig,
)


class RuntimeHandle:
    def __init__(self, name: str) -> None:
        self.name = name
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def recover(self) -> None:
        self.started = True


def build_start_control() -> JarvisStartControlRuntime:
    voice_launcher = VoiceRuntimeLauncher(
        config=VoiceRuntimeLauncherConfig(
            run_forever=True,
            run_daily_driver_gate=True,
            allow_degraded_gate=False,
            metadata={"entrypoint": "run_connected_jarvis"},
        )
    )

    organs = (
        ManagedJarvisOrganController(
            kind=JarvisOrganKind.PHASE1_KERNEL,
            name="phase1_kernel_events_observability",
            runtime=RuntimeHandle("phase1"),
        ),
        ManagedJarvisOrganController(
            kind=JarvisOrganKind.PHASE2_VOICE,
            name="phase2_presence_voice",
            runtime=RuntimeHandle("phase2"),
            dependencies=(JarvisOrganKind.PHASE1_KERNEL,),
        ),
        ManagedJarvisOrganController(
            kind=JarvisOrganKind.PHASE3_COGNITION,
            name="phase3_cognition",
            runtime=RuntimeHandle("phase3"),
            dependencies=(JarvisOrganKind.PHASE1_KERNEL,),
        ),
        ManagedJarvisOrganController(
            kind=JarvisOrganKind.PHASE4_MEMORY,
            name="phase4_memory_gateway",
            runtime=RuntimeHandle("phase4"),
            dependencies=(JarvisOrganKind.PHASE1_KERNEL,),
        ),
        ManagedJarvisOrganController(
            kind=JarvisOrganKind.PHASE5_TOOLS,
            name="phase5_tool_action_runtime",
            runtime=RuntimeHandle("phase5"),
            dependencies=(JarvisOrganKind.PHASE1_KERNEL,),
        ),
        ManagedJarvisOrganController(
            kind=JarvisOrganKind.PHASE6_ORCHESTRATION,
            name="phase6_orchestration",
            runtime=RuntimeHandle("phase6"),
            dependencies=(
                JarvisOrganKind.PHASE1_KERNEL,
                JarvisOrganKind.PHASE3_COGNITION,
                JarvisOrganKind.PHASE4_MEMORY,
                JarvisOrganKind.PHASE5_TOOLS,
            ),
        ),
        ManagedJarvisOrganController(
            kind=JarvisOrganKind.PHASE7_STREAMING_LATENCY,
            name="phase7_streaming_latency",
            runtime=RuntimeHandle("phase7"),
            dependencies=(JarvisOrganKind.PHASE6_ORCHESTRATION,),
        ),
        ManagedJarvisOrganController(
            kind=JarvisOrganKind.PHASE8_ENVIRONMENT,
            name="phase8_environment_awareness",
            runtime=RuntimeHandle("phase8"),
            dependencies=(JarvisOrganKind.PHASE6_ORCHESTRATION,),
        ),
        ManagedJarvisOrganController(
            kind=JarvisOrganKind.PHASE9_COGNITIVE_SESSION,
            name="phase9_cognitive_session_goals_personality",
            runtime=RuntimeHandle("phase9"),
            dependencies=(
                JarvisOrganKind.PHASE3_COGNITION,
                JarvisOrganKind.PHASE4_MEMORY,
                JarvisOrganKind.PHASE6_ORCHESTRATION,
            ),
        ),
        VoiceLauncherOrganController(launcher=voice_launcher),
    )

    return JarvisStartControlRuntime(organs=organs)


def main() -> int:
    start_control = build_start_control()
    result = start_control.start_all()

    if not result.succeeded:
        return 1

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop_result = start_control.stop_all()
        return 0 if stop_result.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())