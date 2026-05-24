from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.memory import audit_phase4_memory_safety  # noqa: E402


def main() -> int:
    result = audit_phase4_memory_safety()

    print()
    print("JARVIS Phase 4 Memory Safety Audit")
    print("----------------------------------")
    print(f"Passed: {result.passed}")
    print(f"Status: {result.status.value}")
    print(f"Checks: {result.passed_count}/{result.check_count}")
    print()

    for check in result.checks:
        marker = "PASS" if check.passed else "FAIL"
        print(
            f"[{marker}] {check.name} "
            f"({check.risk_level.value}): {check.detail}"
        )

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())