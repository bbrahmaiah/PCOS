from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.voice import (  # noqa: E402
    VoiceRuntimeLauncher,
    VoiceRuntimeLauncherConfig,
    VoiceRuntimeLauncherStatus,
)


def main() -> int:
    launcher = VoiceRuntimeLauncher(
        config=VoiceRuntimeLauncherConfig(
            run_forever=True,
            run_daily_driver_gate=True,
            allow_degraded_gate=False,
            metadata={"entrypoint": "run_voice_jarvis"},
        )
    )
    launcher.install_signal_handlers()

    result = launcher.run()

    if result.status == VoiceRuntimeLauncherStatus.FAILED:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())