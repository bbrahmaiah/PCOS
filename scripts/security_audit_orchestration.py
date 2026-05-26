from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.orchestration import SecurityHardeningAuditRuntime  # noqa: E402


def main() -> int:
    runtime = SecurityHardeningAuditRuntime()
    report = runtime.run()

    print("=" * 72)
    print("PHASE 6 SECURITY HARDENING AUDIT")
    print("=" * 72)
    print(report.summary)
    print(
        f"passed={report.passed_count} "
        f"failed={report.failed_count} "
        f"blocked={report.blocked_count}"
    )
    print("-" * 72)

    for finding in report.findings:
        icon = "BLOCKED" if finding.blocked else "ALLOWED"
        print(
            f"[{icon}] "
            f"{finding.vector.kind.value}: "
            f"{finding.message}"
        )

    print("-" * 72)

    if report.success:
        print("SECURITY HARDENING AUDIT PASSED")
        return 0

    print("SECURITY HARDENING AUDIT FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())