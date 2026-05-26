from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.orchestration import OrchestrationSmokeRuntime


def main() -> int:
    runtime = OrchestrationSmokeRuntime()
    report = runtime.run()

    print("=" * 72)
    print("PHASE 6 ORCHESTRATION SMOKE RUNTIME")
    print("=" * 72)
    print(report.summary)
    print(
        f"passed={report.passed_count} "
        f"failed={report.failed_count} "
        f"skipped={report.skipped_count}"
    )
    print("-" * 72)

    for check in report.checks:
        icon = "PASS" if check.passed else "FAIL"
        print(f"[{icon}] {check.kind.value}: {check.message}")

    print("-" * 72)

    if report.success:
        print("NO PHASE 6 SMOKE FAILURES DETECTED")
        return 0

    print("PHASE 6 SMOKE FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())