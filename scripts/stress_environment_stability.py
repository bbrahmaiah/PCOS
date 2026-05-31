from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.environment import (  # noqa: E402
    Phase8LatencyBudget,
    Phase8LoadLatencyStabilityRuntime,
    Phase8StabilityReport,
    Phase8StabilityStatus,
    Phase8StressProfile,
)


def run_stability_validation() -> Phase8StabilityReport:
    runtime = Phase8LoadLatencyStabilityRuntime()
    session = runtime.create_session(workspace_id="workspace")
    return runtime.validate(
        session_id=session.session_id,
        profile=Phase8StressProfile(),
        budget=Phase8LatencyBudget(),
    )


def report_to_dict(report: Phase8StabilityReport) -> dict[str, Any]:
    return {
        "status": report.status.value,
        "decision": report.decision.value,
        "reason": report.reason.value,
        "passed_count": report.passed_count,
        "degraded_count": report.degraded_count,
        "failed_count": report.failed_count,
        "visual_workers_shed_count": report.visual_workers_shed_count,
        "conversation_latency_protected": report.conversation_latency_protected,
        "memory_leak_detected": report.memory_leak_detected,
        "scenarios": [
            {
                "scenario": result.scenario.value,
                "status": result.status.value,
                "decision": result.decision.value,
                "reason": result.reason.value,
                "visual_workers_shed": result.visual_workers_shed,
                "conversation_protected": result.conversation_protected,
                "memory_leak_detected": result.memory_leak_detected,
                "metrics": [
                    {
                        "kind": metric.kind.value,
                        "value": metric.value,
                        "budget": metric.budget,
                        "passed": metric.passed,
                        "unit": metric.unit,
                    }
                    for metric in result.metrics
                ],
            }
            for result in report.scenario_results
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Phase 8 load, latency, and stability validation."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON report.",
    )
    args = parser.parse_args()

    report = run_stability_validation()

    if args.json:
        print(json.dumps(report_to_dict(report), indent=2, sort_keys=True))
    else:
        print("PHASE 8 LOAD LATENCY STABILITY VALIDATION")
        print(
            f"status={report.status.value} "
            f"passed={report.passed_count} "
            f"degraded={report.degraded_count} "
            f"failed={report.failed_count} "
            f"shed={report.visual_workers_shed_count}"
        )
        for result in report.scenario_results:
            print(
                f"[{result.status.value.upper()}] "
                f"{result.scenario.value}: {result.message}"
            )

    return 0 if report.status == Phase8StabilityStatus.PASSED else 1


if __name__ == "__main__":
    raise SystemExit(main())