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


def _print_result(prefix: str, result: object) -> None:
    status = getattr(result, "status", None)
    reason = getattr(result, "reason", None)
    message = getattr(result, "message", None)
    event = getattr(result, "event", None)
    metadata = getattr(result, "metadata", None)
    session_result = getattr(result, "session_result", None)

    print(f"[voice] {prefix}")
    print(f"[voice] status={status}")
    print(f"[voice] event={event}")
    print(f"[voice] reason={reason}")
    print(f"[voice] message={message}")
    print(f"[voice] metadata={metadata}")

    if session_result is not None:
        print(f"[voice] session_status={getattr(session_result, 'status', None)}")
        print(f"[voice] session_event={getattr(session_result, 'event', None)}")
        print(f"[voice] session_message={getattr(session_result, 'message', None)}")
        print(f"[voice] session_metadata={getattr(session_result, 'metadata', None)}")


def main() -> int:
    launcher = VoiceRuntimeLauncher(
        config=VoiceRuntimeLauncherConfig(
            run_forever=True,
            run_daily_driver_gate=False,
            allow_degraded_gate=False,
            idle_sleep_seconds=0.05,
            stop_on_session_failure=True,
            metadata={"entrypoint": "run_voice_jarvis_live_smoke"},
        )
    )
    launcher.install_signal_handlers()

    print("[voice] starting live voice runtime")
    result = launcher.run()
    _print_result("launcher returned", result)

    snapshot = launcher.snapshot()
    print(f"[voice] snapshot_status={snapshot.status}")
    print(f"[voice] snapshot_running={snapshot.running}")
    print(f"[voice] snapshot_last_error={snapshot.last_error}")

    if result.status == VoiceRuntimeLauncherStatus.FAILED:
        return 1

    if result.status == VoiceRuntimeLauncherStatus.STOPPED:
        print("[voice] stopped immediately; live microphone loop did not stay active")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())