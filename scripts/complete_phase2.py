from __future__ import annotations

# ruff: noqa: E402
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.presence.phase2_completion import (
    Phase2CompletionConfig,
    Phase2CompletionGate,
    format_phase2_completion_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the final JARVIS Phase 2 Presence completion gate."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Timeout in seconds for each command.",
    )
    parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="Skip pytest. Use only for fast debugging.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip Presence validation.",
    )
    parser.add_argument(
        "--skip-latency",
        action="store_true",
        help="Skip Presence latency profile.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = Phase2CompletionConfig(
        project_root=PROJECT_ROOT,
        timeout_seconds=args.timeout,
        include_pytest=not args.skip_pytest,
        include_validation=not args.skip_validation,
        include_latency_profile=not args.skip_latency,
    )

    report = Phase2CompletionGate(config).run()
    print(format_phase2_completion_report(report))

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())