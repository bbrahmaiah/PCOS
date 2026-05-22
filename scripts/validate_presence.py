from __future__ import annotations

# ruff: noqa: E402
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.presence.validation import PresenceIntegrationValidator


def main() -> int:
    report = PresenceIntegrationValidator().run()

    print()
    print("JARVIS Presence Integration Validation")
    print("-------------------------------------")
    print(f"Passed: {report.passed}")
    print(f"Checks: {report.passed_count} passed, {report.failed_count} failed")
    print(f"Duration: {report.duration_ms:.2f} ms")

    for check in report.checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}")
        print(f"       details={check.details}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())