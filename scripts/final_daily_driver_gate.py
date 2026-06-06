from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.runtime import (  # noqa: E402
    JarvisFinalDailyDriverGate,
    JarvisFinalDailyDriverGateConfig,
    JarvisFinalDailyDriverGateStatus,
    summarize_final_daily_driver_report,
)

BINDINGS_PATH = PROJECT_ROOT / "config" / "runtime_bindings.env"


def main() -> int:
    gate = JarvisFinalDailyDriverGate(
        config=JarvisFinalDailyDriverGateConfig(
            bindings_path=BINDINGS_PATH,
            scan_root=PROJECT_ROOT / "jarvis",
            metadata={"entrypoint": "final_daily_driver_gate"},
        )
    )
    report = gate.run()
    print(summarize_final_daily_driver_report(report))
    return 0 if report.status == JarvisFinalDailyDriverGateStatus.PASSED else 1


if __name__ == "__main__":
    raise SystemExit(main())