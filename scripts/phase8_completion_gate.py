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
    Phase8CompletionGateRuntime,
    Phase8CompletionReport,
    Phase8CompletionStatus,
)


def run_completion_gate() -> Phase8CompletionReport:
    runtime = Phase8CompletionGateRuntime()
    session = runtime.create_session(workspace_id="workspace")
    return runtime.run_completion_gate(session_id=session.session_id)


def report_to_dict(report: Phase8CompletionReport) -> dict[str, Any]:
    return {
        "status": report.status.value,
        "decision": report.decision.value,
        "reason": report.reason.value,
        "sealed": report.sealed,
        "capability_passed_count": report.capability_passed_count,
        "checklist_passed_count": report.checklist_passed_count,
        "gate_passed_count": report.gate_passed_count,
        "failed_count": report.failed_count,
        "gates": [
            {
                "gate": gate.gate.value,
                "passed": gate.passed,
                "status": gate.status.value,
                "reason": gate.reason.value,
                "message": gate.message,
            }
            for gate in report.gates
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Phase 8 completion gate."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON report.",
    )
    args = parser.parse_args()

    report = run_completion_gate()

    if args.json:
        print(json.dumps(report_to_dict(report), indent=2, sort_keys=True))
    else:
        print("PHASE 8 COMPLETION GATE")
        print(
            f"status={report.status.value} "
            f"sealed={report.sealed} "
            f"capabilities={report.capability_passed_count} "
            f"checklist={report.checklist_passed_count} "
            f"gates={report.gate_passed_count} "
            f"failed={report.failed_count}"
        )
        for gate in report.gates:
            icon = "PASS" if gate.passed else "FAIL"
            print(f"[{icon}] {gate.gate.value}: {gate.message}")

    return 0 if report.status == Phase8CompletionStatus.SEALED else 1


if __name__ == "__main__":
    raise SystemExit(main())