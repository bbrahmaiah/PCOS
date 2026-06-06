from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.runtime import (  # noqa: E402
    build_connected_start_control_from_plan,
    build_plan_from_import_bindings,
    read_runtime_binding_imports,
)
from jarvis.voice import (  # noqa: E402
    VoiceRuntimeLauncher,
    VoiceRuntimeLauncherConfig,
)

BINDINGS_PATH = PROJECT_ROOT / "config" / "runtime_bindings.env"


def main() -> int:
    voice_launcher = VoiceRuntimeLauncher(
        config=VoiceRuntimeLauncherConfig(
            run_forever=True,
            run_daily_driver_gate=True,
            allow_degraded_gate=False,
            metadata={"entrypoint": "run_connected_jarvis"},
        )
    )

    try:
        import_bindings = read_runtime_binding_imports(BINDINGS_PATH)
        plan = build_plan_from_import_bindings(
            import_bindings=import_bindings,
            voice_launcher=voice_launcher,
        )
        start_control = build_connected_start_control_from_plan(plan)
        result = start_control.start_all()
    except Exception as exc:
        print(f"[JARVIS-RUNTIME] startup failed: {exc}")
        return 1

    if not result.succeeded:
        print(f"[JARVIS-RUNTIME] startup blocked: {result.reason}")
        return 1

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop_result = start_control.stop_all()
        return 0 if stop_result.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())