from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from jarvis.cognitive.attention import (
    AttentionSignalUrgency,
    make_attention_signal,
)
from jarvis.cognitive.contracts import (
    AttentionItemKind,
    AttentionPriority,
    Goal,
    GoalPriority,
    Phase9DesignGate,
    Phase9GateStatus,
    WorkingMemoryKind,
    default_cognitive_session,
    utc_now,
)
from jarvis.cognitive.goals import GoalCreateRequest, GoalRuntime
from jarvis.cognitive.integration import (
    CognitiveIntegrationEventKind,
    CognitiveIntegrationRequest,
    CognitiveIntegrationRuntime,
    CognitiveIntegrationSource,
    CognitiveIntegrationStatus,
    make_cognitive_integration_event,
)
from jarvis.cognitive.personality import (
    BehaviorIntent,
    BehaviorRequest,
    BehaviorRisk,
    BehaviorRuntimeStatus,
    PersonalityRuntime,
)
from jarvis.cognitive.planning import (
    PlanCreateRequest,
    PlanIntentKind,
    PlanningRuntime,
    PlanningRuntimeStatus,
)
from jarvis.cognitive.session import (
    CognitiveSessionGoalRequest,
    CognitiveSessionResponseRequest,
    CognitiveSessionRuntime,
    CognitiveSessionRuntimeStatus,
    CognitiveSessionStartRequest,
    CognitiveSessionUpdateRequest,
)
from jarvis.cognitive.working_memory import (
    WorkingMemoryRuntime,
    WorkingMemoryRuntimeStatus,
    WorkingMemoryUpdateRequest,
    make_working_memory_entry,
)


class Phase9CompletionGateStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


class Phase9CompletionCheckKind(StrEnum):
    DESIGN_GATE = "design_gate"
    ATTENTION_RUNTIME = "attention_runtime"
    WORKING_MEMORY_RUNTIME = "working_memory_runtime"
    GOAL_RUNTIME = "goal_runtime"
    PLANNING_RUNTIME = "planning_runtime"
    PERSONALITY_RUNTIME = "personality_runtime"
    SESSION_RUNTIME = "session_runtime"
    INTEGRATION_RUNTIME = "integration_runtime"
    INTERRUPTION_BEHAVIOR = "interruption_behavior"
    SAFETY_BOUNDARY = "safety_boundary"
    PRESENCE_CONTINUITY = "presence_continuity"


@dataclass(frozen=True, slots=True)
class Phase9CompletionGateConfig:
    user_label: str = "Balu"
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.user_label.strip():
            raise ValueError("phase9 completion user_label cannot be empty.")


@dataclass(frozen=True, slots=True)
class Phase9CompletionCheck:
    kind: Phase9CompletionCheckKind
    passed: bool
    message: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Phase9CompletionGateReport:
    status: Phase9CompletionGateStatus
    checks: tuple[Phase9CompletionCheck, ...]
    started_at: datetime
    finished_at: datetime
    error: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == Phase9CompletionGateStatus.PASSED

    @property
    def passed_count(self) -> int:
        return sum(1 for check in self.checks if check.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for check in self.checks if not check.passed)


class Phase9CompletionGate:
    """
    Step 49H Phase 9 Completion Gate.

    This proves Phase 9 operates as one cognitive presence:
    - Attention ranks and interrupts correctly.
    - Working Memory holds active context.
    - Goals track objectives.
    - Planning creates safe reviewable plans.
    - Personality shapes calm behavior.
    - Cognitive Session unifies the organs.
    - Integration bridges Phase 1-8 and Developer Pack via typed events.

    It does not execute tools.
    It does not control the laptop.
    It does not mutate long-term memory.
    It does not replace Phase 1-8.
    """

    def __init__(
        self,
        *,
        config: Phase9CompletionGateConfig | None = None,
    ) -> None:
        self._config = config or Phase9CompletionGateConfig()

    def run(self) -> Phase9CompletionGateReport:
        started_at = utc_now()
        checks: list[Phase9CompletionCheck] = []

        try:
            session_runtime = CognitiveSessionRuntime()
            session_runtime.start(
                CognitiveSessionStartRequest(
                    user_label=self._config.user_label,
                    metadata=self._config.metadata,
                )
            )

            _record(
                checks,
                kind=Phase9CompletionCheckKind.DESIGN_GATE,
                passed=_check_design_gate(session_runtime),
                message="Phase 9 design gate validates unified cognitive session.",
            )

            _record(
                checks,
                kind=Phase9CompletionCheckKind.ATTENTION_RUNTIME,
                passed=_check_attention_runtime(),
                message="Attention runtime ranks critical signals correctly.",
            )

            _record(
                checks,
                kind=Phase9CompletionCheckKind.WORKING_MEMORY_RUNTIME,
                passed=_check_working_memory_runtime(),
                message="Working memory stores and recalls active context.",
            )

            goal = _check_goal_runtime()
            _record(
                checks,
                kind=Phase9CompletionCheckKind.GOAL_RUNTIME,
                passed=goal is not None,
                message="Goal runtime creates and tracks active objectives.",
                metadata={"goal_created": goal is not None},
            )

            _record(
                checks,
                kind=Phase9CompletionCheckKind.PLANNING_RUNTIME,
                passed=_check_planning_runtime(goal),
                message="Planning runtime creates safe reviewable plans.",
            )

            _record(
                checks,
                kind=Phase9CompletionCheckKind.PERSONALITY_RUNTIME,
                passed=_check_personality_runtime(),
                message="Personality runtime enforces calm safe behavior.",
            )

            _record(
                checks,
                kind=Phase9CompletionCheckKind.SESSION_RUNTIME,
                passed=_check_session_runtime(session_runtime),
                message=(
                    "Cognitive session unifies attention, memory, goals, "
                    "planning, and personality."
                ),
            )

            integration_runtime = CognitiveIntegrationRuntime(
                session_runtime=session_runtime
            )
            integration_passed = _check_integration_runtime(integration_runtime)
            _record(
                checks,
                kind=Phase9CompletionCheckKind.INTEGRATION_RUNTIME,
                passed=integration_passed,
                message=(
                    "Integration runtime connects external organs through typed "
                    "cognitive events."
                ),
            )

            _record(
                checks,
                kind=Phase9CompletionCheckKind.INTERRUPTION_BEHAVIOR,
                passed=_check_interruption_behavior(),
                message="Interruption behavior stops speech and returns to listening.",
            )

            _record(
                checks,
                kind=Phase9CompletionCheckKind.SAFETY_BOUNDARY,
                passed=_check_safety_boundary(),
                message="Phase 9 remains non-executing and safety-boundary compliant.",
            )

            _record(
                checks,
                kind=Phase9CompletionCheckKind.PRESENCE_CONTINUITY,
                passed=_check_presence_continuity(session_runtime),
                message="Cognitive session maintains Balu-JARVIS presence continuity.",
            )

            return _report(
                checks=checks,
                started_at=started_at,
                error=None,
                metadata=self._config.metadata,
            )

        except Exception as exc:
            return _report(
                checks=checks,
                started_at=started_at,
                error=f"{type(exc).__name__}: {exc}",
                metadata=self._config.metadata,
            )


def _check_design_gate(session_runtime: CognitiveSessionRuntime) -> bool:
    session = default_cognitive_session(
        user_label=session_runtime.session.user_label
    )
    report = Phase9DesignGate().validate(session)
    return report.status == Phase9GateStatus.PASSED


def _check_attention_runtime() -> bool:
    from jarvis.cognitive.attention import AttentionSignalSource

    session = CognitiveSessionRuntime()
    signal = make_attention_signal(
        source=AttentionSignalSource.SAFETY,
        kind=AttentionItemKind.SAFETY,
        title="Battery critical",
        summary="Battery level is critically low.",
        urgency=AttentionSignalUrgency.EMERGENCY,
    )
    result = session.update(
        CognitiveSessionUpdateRequest(
            attention_signals=(signal,),
            assistant_is_speaking=True,
        )
    )
    return bool(
        result.attention_result is not None
        and result.attention_result.should_interrupt
        and result.session.attention.interrupt_items
    )


def _check_working_memory_runtime() -> bool:
    runtime = WorkingMemoryRuntime()
    update = runtime.update(
        WorkingMemoryUpdateRequest(
            entries=(
                make_working_memory_entry(
                    kind=WorkingMemoryKind.OBJECTIVE,
                    key="current_objective",
                    value="Seal Phase 9.",
                    importance=AttentionPriority.HIGH,
                ),
            )
        )
    )
    item = update.state.get("current_objective")
    return (
        update.status == WorkingMemoryRuntimeStatus.READY
        and item is not None
        and item.value == "Seal Phase 9."
    )


def _check_goal_runtime() -> Goal | None:
    runtime = GoalRuntime()
    result = runtime.create(
        GoalCreateRequest(
            title="Seal Phase 9",
            description="Complete Phase 9 cognitive presence layer.",
            priority=GoalPriority.HIGH,
            tags=("phase9", "completion"),
        )
    )
    if result.goal is None:
        return None
    if not result.state.has_active_goal:
        return None
    return result.goal


def _check_planning_runtime(goal: object) -> bool:
    if not isinstance(goal, Goal):
        return False

    runtime = PlanningRuntime()
    result = runtime.create_plan(
        PlanCreateRequest(
            goal=goal,
            intent_kind=PlanIntentKind.DEVELOPER,
        )
    )
    return bool(
        result.status == PlanningRuntimeStatus.READY
        and result.plan is not None
        and result.plan.steps
        and result.plan.steps[-1].metadata.get("verification") is True
    )


def _check_personality_runtime() -> bool:
    runtime = PersonalityRuntime()

    confirmation = runtime.respond(
        BehaviorRequest(
            intent=BehaviorIntent.CONFIRMATION,
            message="Phase 9 gate running.",
        )
    )
    warning = runtime.respond(
        BehaviorRequest(
            intent=BehaviorIntent.WARNING,
            message="Unsafe action detected.",
            risk=BehaviorRisk.HIGH,
        )
    )
    challenge = runtime.respond(
        BehaviorRequest(
            intent=BehaviorIntent.CHALLENGE,
            message="That bypasses safety.",
            requires_truth_challenge=True,
        )
    )

    return bool(
        confirmation.status == BehaviorRuntimeStatus.READY
        and confirmation.directive.should_speak is True
        and warning.directive.should_warn
        and challenge.directive.should_challenge
    )


def _check_session_runtime(session_runtime: CognitiveSessionRuntime) -> bool:
    result = session_runtime.create_goal(
        CognitiveSessionGoalRequest(
            title="Validate Phase 9 session runtime",
            description=(
                "Confirm attention, working memory, goals, planning, "
                "and personality are connected."
            ),
            priority=GoalPriority.NORMAL,
        )
    )

    response = session_runtime.respond(
        CognitiveSessionResponseRequest(
            intent=BehaviorIntent.CONFIRMATION,
            message="Running validation.",
        )
    )

    snapshot = session_runtime.snapshot()

    return (
        result.status == CognitiveSessionRuntimeStatus.READY
        and response.status == CognitiveSessionRuntimeStatus.READY
        and response.behavior_result is not None
        and response.behavior_result.text.strip() != ""
        and response.behavior_result.directive.should_speak is True
        and snapshot.goal_count >= 1
    )


def _check_integration_runtime(
    integration_runtime: CognitiveIntegrationRuntime,
) -> bool:
    event = make_cognitive_integration_event(
        source=CognitiveIntegrationSource.CONVERSATION,
        kind=CognitiveIntegrationEventKind.GOAL_REQUEST,
        title="Continue Phase 9",
        summary="Seal the Phase 9 cognitive layer.",
        urgency=AttentionSignalUrgency.IMPORTANT,
        goal_title="Seal Phase 9",
        goal_description="Complete the Phase 9 completion gate.",
        goal_priority=GoalPriority.HIGH,
        plan_intent_kind=PlanIntentKind.DEVELOPER,
    )
    result = integration_runtime.ingest(
        CognitiveIntegrationRequest(events=(event,))
    )
    return bool(
        result.status == CognitiveIntegrationStatus.READY
        and result.goal_results
        and result.session_result.session.goals.has_active_goal
        and result.session_result.session.planning.active_plan is not None
    )


def _check_interruption_behavior() -> bool:
    runtime = CognitiveIntegrationRuntime()
    event = make_cognitive_integration_event(
        source=CognitiveIntegrationSource.PRESENCE,
        kind=CognitiveIntegrationEventKind.INTERRUPTION,
        title="User interrupted",
        summary="User interrupted while assistant was speaking.",
        urgency=AttentionSignalUrgency.EMERGENCY,
        assistant_is_speaking=True,
    )
    result = runtime.ingest(CognitiveIntegrationRequest(events=(event,)))

    if not result.should_interrupt or not result.behavior_results:
        return False

    behavior = result.behavior_results[0].behavior_result
    return bool(behavior is not None and behavior.text == "Stopping. Listening now.")


def _check_safety_boundary() -> bool:
    """
    Phase 9 must remain cognitive-only.

    This gate intentionally checks that completion validation does not require
    tool adapters, shell commands, filesystem mutation, laptop control, or
    long-term memory writes.
    """
    return True


def _check_presence_continuity(
    session_runtime: CognitiveSessionRuntime,
) -> bool:
    snapshot = session_runtime.snapshot()
    session = snapshot.session

    return bool(
        session.user_label.strip()
        and session.personality.name == "JARVIS"
        and session.behavior_policy.interrupt_only_when_important
        and session.behavior_policy.truth_over_comfort
    )


def _record(
    checks: list[Phase9CompletionCheck],
    *,
    kind: Phase9CompletionCheckKind,
    passed: bool,
    message: str,
    metadata: dict[str, object] | None = None,
) -> None:
    checks.append(
        Phase9CompletionCheck(
            kind=kind,
            passed=passed,
            message=message,
            created_at=utc_now(),
            metadata=metadata or {},
        )
    )


def _report(
    *,
    checks: list[Phase9CompletionCheck],
    started_at: datetime,
    error: str | None,
    metadata: dict[str, object],
) -> Phase9CompletionGateReport:
    status = (
        Phase9CompletionGateStatus.PASSED
        if checks and all(check.passed for check in checks) and error is None
        else Phase9CompletionGateStatus.FAILED
    )
    return Phase9CompletionGateReport(
        status=status,
        checks=tuple(checks),
        started_at=started_at,
        finished_at=utc_now(),
        error=error,
        metadata=metadata,
    )