from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.cognition import (
    CognitionContext,
    CognitionContextItem,
    CognitionFailure,
    CognitionFailureKind,
    CognitionPlan,
    CognitionPlanKind,
    CognitionRequest,
    CognitionRequestKind,
    CognitionResponse,
    CognitionResponseKind,
    CognitionRuntimePolicy,
    CognitionSnapshot,
    CognitionToken,
    CognitionTokenKind,
    SpokenResponseStyle,
)


def test_cognition_runtime_policy_defaults_are_safe() -> None:
    policy = CognitionRuntimePolicy()

    assert policy.cancellable is True
    assert policy.streaming_enabled is False
    assert policy.allow_tools is False
    assert policy.allow_memory_lookup is False
    assert policy.max_response_chars == 1_200
    assert policy.timeout_ms == 30_000
    assert policy.spoken_style == SpokenResponseStyle.CONCISE


def test_cognition_runtime_policy_rejects_invalid_bounds() -> None:
    with pytest.raises(ValidationError):
        CognitionRuntimePolicy(max_response_chars=0)

    with pytest.raises(ValidationError):
        CognitionRuntimePolicy(timeout_ms=99)


def test_cognition_context_item_requires_non_empty_fields() -> None:
    with pytest.raises(ValidationError):
        CognitionContextItem(kind="", text="hello")

    with pytest.raises(ValidationError):
        CognitionContextItem(kind="session", text=" ")

    with pytest.raises(ValidationError):
        CognitionContextItem(kind="session", text="hello", source="")


def test_cognition_context_tracks_item_count() -> None:
    item = CognitionContextItem(
        kind="session",
        text="User prefers short answers.",
    )
    context = CognitionContext(
        session_id="session-1",
        turn_id="turn-1",
        items=(item,),
    )

    assert context.item_count == 1
    assert context.items[0].text == "User prefers short answers."


def test_cognition_request_defaults() -> None:
    request = CognitionRequest(
        text="Explain what we built today.",
        transcript_id="transcript-1",
        turn_id="turn-1",
    )

    assert request.kind == CognitionRequestKind.USER_UTTERANCE
    assert request.source == "dialogue"
    assert request.transcript_id == "transcript-1"
    assert request.turn_id == "turn-1"
    assert request.context.item_count == 0
    assert request.policy.cancellable is True
    assert request.request_id


def test_cognition_request_rejects_empty_text_and_source() -> None:
    with pytest.raises(ValidationError):
        CognitionRequest(text=" ")

    with pytest.raises(ValidationError):
        CognitionRequest(text="hello", source=" ")


def test_cognition_plan_defaults() -> None:
    plan = CognitionPlan(request_id="request-1")

    assert plan.kind == CognitionPlanKind.DIRECT_ANSWER
    assert plan.confidence == 1.0
    assert plan.needs_clarification is False
    assert plan.allowed_tool_names == ()
    assert plan.plan_id


def test_cognition_plan_rejects_invalid_confidence() -> None:
    with pytest.raises(ValidationError):
        CognitionPlan(request_id="request-1", confidence=1.1)

    with pytest.raises(ValidationError):
        CognitionPlan(request_id="", confidence=0.5)


def test_cognition_token_model() -> None:
    token = CognitionToken(
        request_id="request-1",
        index=0,
        text="Hello",
    )

    assert token.kind == CognitionTokenKind.TEXT
    assert token.final is False
    assert token.index == 0
    assert token.text == "Hello"


def test_cognition_token_rejects_invalid_values() -> None:
    with pytest.raises(ValidationError):
        CognitionToken(request_id="request-1", index=-1, text="Hello")

    with pytest.raises(ValidationError):
        CognitionToken(request_id="request-1", index=0, text=" ")


def test_cognition_response_model() -> None:
    plan = CognitionPlan(request_id="request-1")
    response = CognitionResponse(
        request_id="request-1",
        text="We built the Presence runtime.",
        plan=plan,
        token_count=6,
    )

    assert response.kind == CognitionResponseKind.SPOKEN_REPLY
    assert response.confidence == 1.0
    assert response.plan == plan
    assert response.token_count == 6
    assert response.response_id


def test_cognition_response_rejects_invalid_values() -> None:
    with pytest.raises(ValidationError):
        CognitionResponse(request_id="request-1", text="")

    with pytest.raises(ValidationError):
        CognitionResponse(request_id="request-1", text="hello", confidence=-0.1)


def test_cognition_failure_model() -> None:
    failure = CognitionFailure(
        request_id="request-1",
        kind=CognitionFailureKind.ADAPTER_ERROR,
        message="adapter failed",
    )

    assert failure.kind == CognitionFailureKind.ADAPTER_ERROR
    assert failure.message == "adapter failed"
    assert failure.recoverable is True
    assert failure.failure_id


def test_cognition_failure_rejects_empty_message() -> None:
    with pytest.raises(ValidationError):
        CognitionFailure(request_id="request-1", message=" ")


def test_cognition_snapshot_defaults() -> None:
    snapshot = CognitionSnapshot()

    assert snapshot.active_request_id is None
    assert snapshot.active_turn_id is None
    assert snapshot.running is False
    assert snapshot.streaming is False
    assert snapshot.cancelling is False
    assert snapshot.completed_count == 0
    assert snapshot.failed_count == 0
    assert snapshot.cancelled_count == 0


def test_cognition_models_are_frozen() -> None:
    request = CognitionRequest(text="hello")

    with pytest.raises(ValidationError):
        request.text = "mutated"


def test_cognition_public_exports_are_available() -> None:
    assert CognitionRequestKind.FOLLOW_UP.value == "follow_up"
    assert CognitionResponseKind.CLARIFICATION.value == "clarification"
    assert CognitionPlanKind.SAFE_REFUSAL.value == "safe_refusal"
    assert CognitionFailureKind.TIMEOUT.value == "timeout"
    assert CognitionTokenKind.FINAL.value == "final"