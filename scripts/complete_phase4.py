from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.memory import complete_phase4_memory  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Phase 4 Memory Runtime completion gate."
    )
    parser.add_argument(
        "--sqlite-path",
        type=Path,
        default=None,
        help="SQLite path used during completion validation.",
    )
    parser.add_argument(
        "--no-sqlite",
        action="store_true",
        help="Disable SQLite validation.",
    )
    parser.add_argument(
        "--no-vector",
        action="store_true",
        help="Disable vector-boundary validation.",
    )
    parser.add_argument(
        "--no-safety-audit",
        action="store_true",
        help="Disable memory safety audit.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    result = complete_phase4_memory(
        sqlite_path=args.sqlite_path,
        include_sqlite=not args.no_sqlite,
        include_vector=not args.no_vector,
        include_safety_audit=not args.no_safety_audit,
    )

    print()
    print("JARVIS Phase 4 Memory Runtime Completion Gate")
    print("--------------------------------------------")
    print(f"Passed: {result.passed}")
    print(f"Status: {result.status.value}")
    print(f"Checks: {result.passed_count}/{result.check_count}")
    print()

    for check in result.checks:
        marker = "PASS" if check.passed else "FAIL"
        print(f"[{marker}] {check.name}: {check.detail}")

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())