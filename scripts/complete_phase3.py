from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.cognition import complete_phase3_cognition  # noqa: E402


def main() -> int:
    report = complete_phase3_cognition(project_root=PROJECT_ROOT)

    print()
    print("JARVIS Phase 3 Completion Gate")
    print("------------------------------")
    print(f"Passed: {report.passed}")
    print(f"Checks: {report.passed_count}/{report.total_count}")

    for check in report.checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}: {check.detail}")

    if report.passed:
        print()
        print("PHASE 3 STATUS: COMPLETE")
        print("Cognition runtime is ready for Phase 4 memory integration.")

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())