from __future__ import annotations

import pytest

from jarvis.cognition import (
    CognitionFailure,
    CognitionFailureKind,
    CognitionRequest,
    CognitionResponse,
    ConversationSessionConfig,
    ConversationSessionState,
    ConversationSessionStore,
    ConversationTurn,
    ConversationTurnRole,
)


def make_request(
    *,
    request_id: str = "request-1",
    text: str = "What did we build today?",
    turn_id: str | None = "turn-1",
) -> CognitionRequest:
    return CognitionRequest(
        request_id=request_id,
        text=text,
        turn_id=turn_id,
        transcript_id="transcript-1",
        correlation_id="correlation-1",
    )


def make_response(
    *,
    request_id: str = "request-1",
    text: str = "We built the cognition session context.",
) -> CognitionResponse:
    return CognitionResponse(
        request_id=request_id,
        text=text,
    )


def make_failure(
    *,
    request_id: str = "request-1",
) -> CognitionFailure:
    return CognitionFailure(
        request_id=request_id,
        kind=CognitionFailureKind.ADAPTER_ERROR,
        message="adapter failed",
    )


def test_conversation_session_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        ConversationSessionConfig(session_id=" ").validate()

    with pytest.raises(ValueError):
        ConversationSessionConfig(max_turns=0).validate()

    with pytest.raises(ValueError):
        ConversationSessionConfig(max_context_items=0).validate()

    with pytest.raises(ValueError):
        ConversationSessionConfig(max_item_chars=0).validate()

    with pytest.raises(ValueError):
        ConversationSessionConfig(active_topic_max_chars=0).validate()


def test_conversation_turn_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        ConversationTurn(
            turn_id=" ",
            role=ConversationTurnRole.USER,
            text="hello",
        ).validate()

    with pytest.raises(ValueError):
        ConversationTurn(
            turn_id="turn-1",
            role=ConversationTurnRole.USER,
            text=" ",
        ).validate()


def test_conversation_session_adds_user_request() -> None:
    store = ConversationSessionStore(
        config=ConversationSessionConfig(session_id="session-1")
    )
    request = make_request()

    turn = store.add_user_request(request)
    snapshot = store.snapshot()

    assert turn.role == ConversationTurnRole.USER
    assert turn.text == "What did we build today?"
    assert turn.request_id == "request-1"
    assert snapshot.session_id == "session-1"
    assert snapshot.turn_count == 1
    assert snapshot.user_turn_count == 1
    assert snapshot.last_user_text == "What did we build today?"
    assert snapshot.last_request_id == "request-1"
    assert snapshot.active_topic == "What did we build today?"


def test_conversation_session_adds_assistant_response() -> None:
    store = ConversationSessionStore()
    response = make_response()

    turn = store.add_assistant_response(response)
    snapshot = store.snapshot()

    assert turn.role == ConversationTurnRole.ASSISTANT
    assert turn.text == "We built the cognition session context."
    assert turn.response_id == response.response_id
    assert snapshot.turn_count == 1
    assert snapshot.assistant_turn_count == 1
    assert snapshot.last_assistant_text == response.text
    assert snapshot.last_response_id == response.response_id


def test_conversation_session_adds_failure_turn() -> None:
    store = ConversationSessionStore()
    failure = make_failure()

    turn = store.add_failure(failure)
    snapshot = store.snapshot()

    assert turn.role == ConversationTurnRole.SYSTEM
    assert turn.failure_id == failure.failure_id
    assert "adapter failed" in turn.text
    assert snapshot.failure_count == 1
    assert snapshot.last_failure_id == failure.failure_id


def test_conversation_session_builds_context_from_recent_turns() -> None:
    store = ConversationSessionStore(
        config=ConversationSessionConfig(
            session_id="session-1",
            max_context_items=3,
        )
    )

    store.add_user_request(make_request())
    store.add_assistant_response(make_response())
    context = store.build_context(request=make_request(request_id="request-2"))

    assert context.session_id == "session-1"
    assert context.turn_id == "turn-1"
    assert context.item_count == 2
    assert context.items[0].kind == "conversation_user"
    assert context.items[1].kind == "conversation_assistant"
    assert context.metadata["turn_count"] == 2


def test_conversation_session_enriches_request_with_context() -> None:
    store = ConversationSessionStore(
        config=ConversationSessionConfig(session_id="session-1")
    )
    store.add_user_request(make_request())
    store.add_assistant_response(make_response())

    request = make_request(request_id="request-2", text="What next?")
    enriched = store.enrich_request(request)

    assert enriched.request_id == "request-2"
    assert enriched.context.session_id == "session-1"
    assert enriched.context.item_count == 2
    assert enriched.metadata["session_id"] == "session-1"
    assert enriched.metadata["active_topic"] == "What did we build today?"


def test_conversation_session_bounds_turn_history() -> None:
    store = ConversationSessionStore(
        config=ConversationSessionConfig(max_turns=2)
    )

    store.add_user_request(make_request(request_id="request-1", text="One"))
    store.add_user_request(make_request(request_id="request-2", text="Two"))
    store.add_user_request(make_request(request_id="request-3", text="Three"))

    turns = store.turns()
    snapshot = store.snapshot()

    assert len(turns) == 2
    assert turns[0].text == "Two"
    assert turns[1].text == "Three"
    assert snapshot.turn_count == 2


def test_conversation_session_bounds_context_item_text() -> None:
    store = ConversationSessionStore(
        config=ConversationSessionConfig(max_item_chars=10)
    )

    store.add_user_request(
        make_request(
            text="This is a very long user message.",
        )
    )
    context = store.build_context()

    assert context.item_count == 1
    assert len(context.items[0].text) <= 10
    assert context.items[0].text.endswith("...")


def test_conversation_session_state_transitions() -> None:
    store = ConversationSessionStore()

    initial_state: ConversationSessionState = store.state
    assert initial_state == ConversationSessionState.ACTIVE

    store.pause()
    paused_state: ConversationSessionState = store.state
    assert paused_state == ConversationSessionState.PAUSED

    store.resume()
    resumed_state: ConversationSessionState = store.state
    assert resumed_state == ConversationSessionState.ACTIVE

    store.close()
    closed_state: ConversationSessionState = store.state
    assert closed_state == ConversationSessionState.CLOSED


def test_conversation_session_reset_clears_state() -> None:
    store = ConversationSessionStore()

    store.add_user_request(make_request())
    store.add_assistant_response(make_response())
    store.add_failure(make_failure())

    store.reset()
    snapshot = store.snapshot()

    assert snapshot.state == ConversationSessionState.ACTIVE
    assert snapshot.turn_count == 0
    assert snapshot.failure_count == 0
    assert snapshot.active_topic is None
    assert snapshot.last_request_id is None
    assert snapshot.last_response_id is None
    assert snapshot.last_failure_id is None


def test_conversation_session_role_values_are_stable() -> None:
    assert ConversationTurnRole.USER.value == "user"
    assert ConversationTurnRole.ASSISTANT.value == "assistant"
    assert ConversationTurnRole.SYSTEM.value == "system"


def test_conversation_session_state_values_are_stable() -> None:
    assert ConversationSessionState.ACTIVE.value == "active"
    assert ConversationSessionState.PAUSED.value == "paused"
    assert ConversationSessionState.CLOSED.value == "closed"