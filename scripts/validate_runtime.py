from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.runtime.validation import RuntimeIntegrationValidator  # noqa: E402


def main() -> int:
    report = RuntimeIntegrationValidator().run()

    print()
    print("JARVIS Runtime Integration Validation")
    print("------------------------------------")
    print(f"Passed: {report.passed}")
    print(f"Checks: {report.passed_count} passed, {report.failed_count} failed")
    print(f"Duration: {report.duration_ms:.2f} ms")

    for check in report.checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}")

        if check.error:
            print(f"       error={check.error}")

        if check.details:
            print(f"       details={check.details}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())