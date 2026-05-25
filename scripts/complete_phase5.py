from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.tools import Phase5CompletionGate  # noqa: E402


def main() -> int:
    gate = Phase5CompletionGate()
    report = gate.run()

    print(report.summary)
    print(f"status={report.status.value}")
    print(f"checks={len(report.checks)}")

    for check in report.checks:
        marker = "PASS" if check.passed else "FAIL"
        print(f"[{marker}] {check.kind.value}: {check.detail}")

    if report.passed:
        return 0

    print("failed_checks:")

    for check in report.failed_checks:
        print(f"- {check.kind.value}: {check.detail}")

    return 1


if __name__ == "__main__":
    sys.exit(main())