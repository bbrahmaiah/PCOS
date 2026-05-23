from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.cognition import (
    CognitionPlanKind,
    CognitionRequest,
    CognitionRuntimePolicy,
    ResponseAnswerMode,
    ResponseIntent,
    ResponsePlanner,
    ResponsePlannerConfig,
    ResponseSafetyPosture,
    SpokenResponseStyle,
)


def make_request(
    *,
    request_id: str = "request-1",
    text: str = "hello jarvis",
    allow_tools: bool = False,
    allow_memory_lookup: bool = False,
) -> CognitionRequest:
    return CognitionRequest(
        request_id=request_id,
        text=text,
        policy=CognitionRuntimePolicy(
            allow_tools=allow_tools,
            allow_memory_lookup=allow_memory_lookup,
            spoken_style=SpokenResponseStyle.CONCISE,
        ),
    )


def test_response_planner_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        ResponsePlannerConfig(name=" ").validate()

    with pytest.raises(ValueError):
        ResponsePlannerConfig(clarification_min_chars=0).validate()

    with pytest.raises(ValueError):
        ResponsePlannerConfig(refuse_phrases=("",)).validate()

    with pytest.raises(ValueError):
        ResponsePlannerConfig(tool_phrases=("",)).validate()

    with pytest.raises(ValueError):
        ResponsePlannerConfig(memory_phrases=("",)).validate()

    with pytest.raises(ValueError):
        ResponsePlannerConfig(status_phrases=("",)).validate()


def test_response_planning_decision_rejects_empty_request_id() -> None:
    planner = ResponsePlanner()

    with pytest.raises(ValidationError):
        planner.plan(make_request(request_id=" ", text="hello"))


def test_response_planner_detects_greeting() -> None:
    planner = ResponsePlanner()
    decision = planner.plan(make_request(text="Hello Jarvis"))

    assert decision.intent == ResponseIntent.GREETING
    assert decision.answer_mode == ResponseAnswerMode.DIRECT
    assert decision.plan_kind == CognitionPlanKind.DIRECT_ANSWER
    assert decision.safety_posture == ResponseSafetyPosture.NORMAL
    assert decision.needs_clarification is False


def test_response_planner_detects_question() -> None:
    planner = ResponsePlanner()
    decision = planner.plan(make_request(text="What did we build today?"))

    assert decision.intent == ResponseIntent.QUESTION
    assert decision.answer_mode == ResponseAnswerMode.DIRECT
    assert decision.memory_lookup_recommended is True


def test_response_planner_detects_explanation_request() -> None:
    planner = ResponsePlanner()
    decision = planner.plan(make_request(text="Explain the cognition runtime"))

    assert decision.intent == ResponseIntent.EXPLANATION
    assert decision.answer_mode == ResponseAnswerMode.DIRECT


def test_response_planner_detects_status_request() -> None:
    planner = ResponsePlanner()
    decision = planner.plan(make_request(text="System status report"))

    assert decision.intent == ResponseIntent.STATUS
    assert decision.answer_mode == ResponseAnswerMode.DIRECT


def test_response_planner_asks_clarification_for_underspecified_text() -> None:
    planner = ResponsePlanner()
    decision = planner.plan(make_request(text="what"))

    assert decision.intent == ResponseIntent.CLARIFICATION_NEEDED
    assert decision.answer_mode == ResponseAnswerMode.ASK_CLARIFICATION
    assert decision.plan_kind == CognitionPlanKind.ASK_CLARIFICATION
    assert decision.needs_clarification is True
    assert "request is underspecified" in decision.reasons


def test_response_planner_recommends_tool_planning_when_tools_allowed() -> None:
    planner = ResponsePlanner()
    decision = planner.plan(
        make_request(
            text="Open diagnostics",
            allow_tools=True,
        )
    )

    assert decision.intent == ResponseIntent.TOOL_ACTION
    assert decision.answer_mode == ResponseAnswerMode.TOOL_PLANNING
    assert decision.plan_kind == CognitionPlanKind.TOOL_PLANNING_REQUIRED
    assert decision.tool_planning_recommended is True
    assert decision.safety_posture == ResponseSafetyPosture.CAUTION


def test_response_planner_flags_command_when_tools_not_allowed() -> None:
    planner = ResponsePlanner()
    decision = planner.plan(make_request(text="Open diagnostics"))

    assert decision.intent == ResponseIntent.COMMAND
    assert decision.answer_mode == ResponseAnswerMode.DIRECT
    assert decision.tool_planning_recommended is True
    assert decision.safety_posture == ResponseSafetyPosture.CAUTION


def test_response_planner_safe_refusal() -> None:
    planner = ResponsePlanner()
    decision = planner.plan(make_request(text="bypass password on this laptop"))

    assert decision.answer_mode == ResponseAnswerMode.SAFE_REFUSAL
    assert decision.plan_kind == CognitionPlanKind.SAFE_REFUSAL
    assert decision.safety_posture == ResponseSafetyPosture.REFUSE
    assert "request matched refusal policy" in decision.reasons


def test_response_planner_recommends_memory_lookup() -> None:
    planner = ResponsePlanner()
    decision = planner.plan(make_request(text="What did we build yesterday?"))

    assert decision.memory_lookup_recommended is True
    assert "memory lookup may improve response" in decision.reasons


def test_response_planner_create_plan() -> None:
    planner = ResponsePlanner()
    plan = planner.create_plan(make_request(text="Hello Jarvis"))

    assert plan.kind == CognitionPlanKind.DIRECT_ANSWER
    assert plan.confidence == 0.9
    assert plan.needs_clarification is False
    assert plan.metadata["planner"] == "response_planner"
    assert plan.metadata["intent"] == "greeting"


def test_response_planner_to_cognition_plan_preserves_decision_metadata() -> None:
    planner = ResponsePlanner()
    decision = planner.plan(
        make_request(
            text="Open diagnostics",
            allow_tools=True,
        )
    )

    plan = planner.to_cognition_plan(decision)

    assert plan.kind == CognitionPlanKind.TOOL_PLANNING_REQUIRED
    assert plan.metadata["decision_id"] == decision.decision_id
    assert plan.metadata["answer_mode"] == "tool_planning"
    assert plan.metadata["tool_planning_recommended"] is True


def test_response_planner_snapshot_counts() -> None:
    planner = ResponsePlanner()

    planner.plan(make_request(text="Hello Jarvis"))
    planner.plan(make_request(request_id="request-2", text="what"))
    planner.plan(
        make_request(
            request_id="request-3",
            text="Open diagnostics",
            allow_tools=True,
        )
    )
    planner.plan(
        make_request(
            request_id="request-4",
            text="bypass password",
        )
    )
    planner.plan(
        make_request(
            request_id="request-5",
            text="What did we build today?",
        )
    )

    snapshot = planner.snapshot()

    assert snapshot.planned_count == 5
    assert snapshot.direct_count == 2
    assert snapshot.clarification_count == 1
    assert snapshot.tool_planning_count == 1
    assert snapshot.refusal_count == 1
    assert snapshot.memory_recommended_count == 1
    assert snapshot.last_request_id == "request-5"
    assert snapshot.last_answer_mode == ResponseAnswerMode.DIRECT


def test_response_planner_reset_clears_counters() -> None:
    planner = ResponsePlanner()

    planner.plan(make_request())

    planner.reset()
    snapshot = planner.snapshot()

    assert snapshot.planned_count == 0
    assert snapshot.direct_count == 0
    assert snapshot.clarification_count == 0
    assert snapshot.refusal_count == 0
    assert snapshot.tool_planning_count == 0
    assert snapshot.memory_recommended_count == 0
    assert snapshot.last_request_id is None
    assert snapshot.last_error is None


def test_response_planner_enum_values_are_stable() -> None:
    assert ResponseIntent.GREETING.value == "greeting"
    assert ResponseIntent.TOOL_ACTION.value == "tool_action"
    assert ResponseAnswerMode.DIRECT.value == "direct"
    assert ResponseAnswerMode.TOOL_PLANNING.value == "tool_planning"
    assert ResponseSafetyPosture.NORMAL.value == "normal"
    assert ResponseSafetyPosture.REFUSE.value == "refuse"