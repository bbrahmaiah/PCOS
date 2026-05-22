from __future__ import annotations

# ruff: noqa: E402
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.presence.latency import PresenceLatencyProfiler


def main() -> int:
    report = PresenceLatencyProfiler().run()

    print()
    print("JARVIS Presence Latency Profile")
    print("-------------------------------")
    print(f"Passed: {report.passed}")
    print(f"Duration: {report.duration_ms:.2f} ms")
    print(
        "Budgets: "
        f"{report.within_budget_count} within, "
        f"{report.over_budget_count} over"
    )

    for item in report.measurements:
        status = "OK" if item.within_budget else "SLOW"
        print(
            f"[{status}] {item.name}: "
            f"{item.duration_ms:.2f} ms "
            f"(budget {item.budget_ms:.2f} ms)"
        )
        print(f"       details={item.details}")

    if report.errors:
        print()
        print("Errors:")
        for error in report.errors:
            print(f" - {error}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())