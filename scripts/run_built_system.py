from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    passed: bool
    details: str


def run_command(name: str, command: list[str]) -> CheckResult:
    print(f"\n=== {name} ===")
    print(" ".join(command))

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    if completed.stdout:
        print(completed.stdout)

    if completed.stderr:
        print(completed.stderr)

    return CheckResult(
        name=name,
        passed=completed.returncode == 0,
        details=f"exit_code={completed.returncode}",
    )


def run_python_check(name: str, code: str) -> CheckResult:
    return run_command(
        name=name,
        command=[sys.executable, "-c", code],
    )


def main() -> int:
    results: list[CheckResult] = []

    results.append(run_command("Ruff", [sys.executable, "-m", "ruff", "check", "."]))
    results.append(run_command("Mypy", [sys.executable, "-m", "mypy", "."]))

    results.append(
        run_command(
            "Full test suite",
            [sys.executable, "-m", "pytest"],
        )
    )

    results.append(
        run_python_check(
            "Developer Feature Pack Completion Gate",
            """
from pathlib import Path

from jarvis.developer import (
    DeveloperFeaturePackCompletionGate,
    DeveloperFeaturePackGateConfig,
    DeveloperFeaturePackGateStatus,
)

gate = DeveloperFeaturePackCompletionGate(
    config=DeveloperFeaturePackGateConfig(project_root=Path('.'))
)
report = gate.run()
print('status=', report.status)
print('passed=', report.passed)
print('passed_count=', report.passed_count)
print('failed_count=', report.failed_count)
assert report.status == DeveloperFeaturePackGateStatus.PASSED
""",
        )
    )

    results.append(
        run_python_check(
            "Phase 9 Completion Gate",
            """
from jarvis.cognitive import (
    Phase9CompletionGate,
    Phase9CompletionGateConfig,
    Phase9CompletionGateStatus,
)

report = Phase9CompletionGate(
    config=Phase9CompletionGateConfig(user_label='Balu')
).run()

print('status=', report.status)
print('passed=', report.passed)
print('passed_count=', report.passed_count)
print('failed_count=', report.failed_count)

for check in report.checks:
    print(check.kind.value, check.passed)

assert report.status == Phase9CompletionGateStatus.PASSED
""",
        )
    )

    results.append(
        run_python_check(
            "Cognitive live simulation: memory, attention, goal, interruption",
            """
from jarvis.cognitive import (
    AttentionSignalUrgency,
    CognitiveIntegrationEventKind,
    CognitiveIntegrationRequest,
    CognitiveIntegrationRuntime,
    CognitiveIntegrationSource,
    GoalPriority,
    PlanIntentKind,
    WorkingMemoryKind,
    make_cognitive_integration_event,
)

runtime = CognitiveIntegrationRuntime()

memory_event = make_cognitive_integration_event(
    source=CognitiveIntegrationSource.MEMORY,
    kind=CognitiveIntegrationEventKind.MEMORY_RECALL,
    title='Project memory',
    summary='Balu is building JARVIS Phase 9 and testing the whole built system.',
    urgency=AttentionSignalUrgency.IMPORTANT,
    working_memory_kind=WorkingMemoryKind.PROJECT,
)

goal_event = make_cognitive_integration_event(
    source=CognitiveIntegrationSource.CONVERSATION,
    kind=CognitiveIntegrationEventKind.GOAL_REQUEST,
    title='Test whole system',
    summary='Run the whole built JARVIS system up to Phase 9.',
    urgency=AttentionSignalUrgency.IMPORTANT,
    goal_title='Test whole built JARVIS system',
    goal_description=(
        'Validate memory, attention, goals, planning, '
        'personality, and integration.'
    ),
    goal_priority=GoalPriority.HIGH,
    plan_intent_kind=PlanIntentKind.DEVELOPER,
)

interrupt_event = make_cognitive_integration_event(
    source=CognitiveIntegrationSource.PRESENCE,
    kind=CognitiveIntegrationEventKind.INTERRUPTION,
    title='User interrupted',
    summary='Stop speaking and listen.',
    urgency=AttentionSignalUrgency.EMERGENCY,
    assistant_is_speaking=True,
)

result = runtime.ingest(
    CognitiveIntegrationRequest(
        events=(memory_event, goal_event, interrupt_event),
        start_session=True,
        user_label='Balu',
    )
)

session = result.session_result.session

print('status=', result.status)
print('should_interrupt=', result.should_interrupt)
print('attention_items=', len(session.attention.items))
print('working_memory_items=', len(session.working_memory.items))
print('has_active_goal=', session.goals.has_active_goal)
print('has_active_plan=', session.planning.active_plan is not None)
print('personality=', session.personality.name)

assert result.should_interrupt is True
assert session.working_memory.items
assert session.goals.has_active_goal
assert session.planning.active_plan is not None
assert session.personality.name == 'JARVIS'
""",
        )
    )

    print("\n\n==============================")
    print("WHOLE BUILT SYSTEM SUMMARY")
    print("==============================")

    failed = [result for result in results if not result.passed]

    for result in results:
        mark = "PASS" if result.passed else "FAIL"
        print(f"{mark}: {result.name} ({result.details})")

    if failed:
        print("\nWhole built system check FAILED.")
        return 1

    print("\nWhole built system check PASSED.")
    print("Built system validated up to Phase 9.")
    print("Note: real always-on microphone/STT/TTS requires a Live Session Runner.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())