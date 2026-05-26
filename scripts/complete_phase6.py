from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.orchestration import Phase6CompletionGateRuntime  # noqa: E402


def main() -> int:
    runtime = Phase6CompletionGateRuntime()
    report = runtime.run()

    print("=" * 72)
    print("PHASE 6 COMPLETION GATE")
    print("=" * 72)
    print(report.summary)
    print(
        f"seal={report.seal_level.value} "
        f"passed={report.passed_count} "
        f"failed={report.failed_count}"
    )
    print("-" * 72)

    for check in report.checks:
        icon = "PASS" if check.passed else "FAIL"
        print(f"[{icon}] {check.kind.value}: {check.message}")

    print("-" * 72)

    if report.certificate is not None:
        print("CERTIFICATE")
        print(f"id={report.certificate.certificate_id}")
        print(report.certificate.summary)
        print("-" * 72)

    if report.success:
        print("PHASE 6 SEALED")
        return 0

    print("PHASE 6 NOT SEALED")
    return 1


if __name__ == "__main__":
    sys.exit(main())