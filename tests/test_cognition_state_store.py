from __future__ import annotations

import pytest

from jarvis.cognition import (
    CognitionFailure,
    CognitionFailureKind,
    CognitionRequest,
    CognitionResponse,
    CognitionRunState,
    CognitionRuntimePolicy,
    CognitionStateStore,
    CognitionToken,
)


def make_request(
    *,
    request_id: str = "request-1",
    text: str = "hello jarvis",
    streaming: bool = False,
) -> CognitionRequest:
    return CognitionRequest(
        request_id=request_id,
        text=text,
        turn_id="turn-1",
        correlation_id="correlation-1",
        policy=CognitionRuntimePolicy(streaming_enabled=streaming),
    )


def make_response(
    *,
    request_id: str = "request-1",
    text: str = "Yes sir.",
) -> CognitionResponse:
    return CognitionResponse(
        request_id=request_id,
        text=text,
    )


def make_failure(
    *,
    request_id: str = "request-1",
    message: str = "adapter failed",
) -> CognitionFailure:
    return CognitionFailure(
        request_id=request_id,
        kind=CognitionFailureKind.ADAPTER_ERROR,
        message=message,
    )


def make_token(
    *,
    request_id: str = "request-1",
    index: int = 0,
    text: str = "Hello",
) -> CognitionToken:
    return CognitionToken(
        request_id=request_id,
        index=index,
        text=text,
    )


def test_cognition_state_store_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        CognitionStateStore(name=" ")


def test_cognition_state_store_initial_snapshot() -> None:
    store = CognitionStateStore()
    snapshot = store.snapshot()

    assert snapshot.state == CognitionRunState.IDLE
    assert snapshot.active_request_id is None
    assert snapshot.streaming is False
    assert snapshot.cancelling is False
    assert snapshot.started_count == 0
    assert snapshot.completed_count == 0
    assert snapshot.failed_count == 0
    assert snapshot.cancelled_count == 0


def test_cognition_state_store_starts_non_streaming_request() -> None:
    store = CognitionStateStore()
    request = make_request()

    result = store.start_request(request)
    snapshot = store.snapshot()

    assert result.accepted is True
    assert result.previous_state == CognitionRunState.IDLE
    assert result.current_state == CognitionRunState.THINKING
    assert snapshot.state == CognitionRunState.THINKING
    assert snapshot.active_request_id == "request-1"
    assert snapshot.active_turn_id == "turn-1"
    assert snapshot.active_correlation_id == "correlation-1"
    assert snapshot.started_count == 1
    assert snapshot.last_started_at is not None


def test_cognition_state_store_starts_streaming_request() -> None:
    store = CognitionStateStore()
    request = make_request(streaming=True)

    result = store.start_request(request)
    snapshot = store.snapshot()

    assert result.accepted is True
    assert result.current_state == CognitionRunState.STREAMING
    assert snapshot.streaming is True


def test_cognition_state_store_rejects_overlapping_request() -> None:
    store = CognitionStateStore()

    assert store.start_request(make_request()).accepted is True

    result = store.start_request(make_request(request_id="request-2"))

    assert result.accepted is False
    assert result.reason == "another cognition request is already active"
    assert store.snapshot().active_request_id == "request-1"


def test_cognition_state_store_records_token() -> None:
    store = CognitionStateStore()

    assert store.start_request(make_request()).accepted is True

    result = store.record_token(make_token())
    snapshot = store.snapshot()

    assert result.accepted is True
    assert result.current_state == CognitionRunState.STREAMING
    assert snapshot.token_count == 1
    assert snapshot.last_token is not None
    assert snapshot.last_token.text == "Hello"


def test_cognition_state_store_rejects_token_for_wrong_request() -> None:
    store = CognitionStateStore()

    assert store.start_request(make_request()).accepted is True

    result = store.record_token(make_token(request_id="wrong"))

    assert result.accepted is False
    assert result.reason == "token request_id does not match active request"
    assert store.snapshot().token_count == 0


def test_cognition_state_store_completes_request() -> None:
    store = CognitionStateStore()

    assert store.start_request(make_request()).accepted is True

    result = store.complete_request(make_response())
    snapshot = store.snapshot()

    assert result.accepted is True
    assert result.current_state == CognitionRunState.COMPLETED
    assert snapshot.state == CognitionRunState.COMPLETED
    assert snapshot.active_request_id is None
    assert snapshot.completed_count == 1
    assert snapshot.last_response is not None
    assert snapshot.last_response.text == "Yes sir."
    assert snapshot.last_finished_at is not None


def test_cognition_state_store_rejects_response_for_wrong_request() -> None:
    store = CognitionStateStore()

    assert store.start_request(make_request()).accepted is True

    result = store.complete_request(make_response(request_id="wrong"))

    assert result.accepted is False
    assert result.reason == "response request_id does not match active request"
    assert store.snapshot().completed_count == 0


def test_cognition_state_store_fails_request() -> None:
    store = CognitionStateStore()

    assert store.start_request(make_request()).accepted is True

    result = store.fail_request(make_failure())
    snapshot = store.snapshot()

    assert result.accepted is True
    assert result.current_state == CognitionRunState.FAILED
    assert snapshot.failed_count == 1
    assert snapshot.last_failure is not None
    assert snapshot.last_error == "adapter failed"
    assert snapshot.active_request_id is None


def test_cognition_state_store_rejects_failure_for_wrong_request() -> None:
    store = CognitionStateStore()

    assert store.start_request(make_request()).accepted is True

    result = store.fail_request(make_failure(request_id="wrong"))

    assert result.accepted is False
    assert result.reason == "failure request_id does not match active request"
    assert store.snapshot().failed_count == 0


def test_cognition_state_store_requests_cancel() -> None:
    store = CognitionStateStore()

    request = make_request()
    assert store.start_request(request).accepted is True

    result = store.request_cancel(
        request_id=request.request_id,
        reason="user interrupted",
    )
    snapshot = store.snapshot()

    assert result.accepted is True
    assert result.current_state == CognitionRunState.CANCELLING
    assert snapshot.cancelling is True
    assert snapshot.last_error == "user interrupted"


def test_cognition_state_store_rejects_cancel_for_wrong_request() -> None:
    store = CognitionStateStore()

    assert store.start_request(make_request()).accepted is True

    result = store.request_cancel(request_id="wrong")

    assert result.accepted is False
    assert result.reason == "cancel request_id does not match active request"


def test_cognition_state_store_confirms_cancel() -> None:
    store = CognitionStateStore()

    request = make_request()
    assert store.start_request(request).accepted is True
    assert store.request_cancel(request_id=request.request_id).accepted is True

    result = store.cancel_request(
        request_id=request.request_id,
        reason="cancelled by interruption",
    )
    snapshot = store.snapshot()

    assert result.accepted is True
    assert result.current_state == CognitionRunState.CANCELLED
    assert snapshot.cancelled_count == 1
    assert snapshot.active_request_id is None
    assert snapshot.last_error == "cancelled by interruption"


def test_cognition_state_store_rejects_complete_while_cancelling() -> None:
    store = CognitionStateStore()

    request = make_request()
    assert store.start_request(request).accepted is True
    assert store.request_cancel(request_id=request.request_id).accepted is True

    result = store.complete_request(make_response())

    assert result.accepted is False
    assert result.reason == "cannot complete while cancelling"
    assert store.snapshot().state == CognitionRunState.CANCELLING


def test_cognition_state_store_public_snapshot_model() -> None:
    store = CognitionStateStore()

    assert store.start_request(make_request()).accepted is True
    assert store.record_token(make_token()).accepted is True

    snapshot = store.cognition_snapshot()

    assert snapshot.active_request_id == "request-1"
    assert snapshot.active_turn_id == "turn-1"
    assert snapshot.running is True
    assert snapshot.streaming is True
    assert snapshot.cancelling is False
    assert snapshot.metadata["state"] == "streaming"
    assert snapshot.metadata["token_count"] == 1


def test_cognition_state_store_public_snapshot_after_completion() -> None:
    store = CognitionStateStore()

    assert store.start_request(make_request()).accepted is True
    assert store.complete_request(make_response()).accepted is True

    snapshot = store.cognition_snapshot()

    assert snapshot.running is False
    assert snapshot.completed_count == 1
    assert snapshot.last_response_id is not None
    assert snapshot.metadata["state"] == "completed"


def test_cognition_state_store_reset_clears_state_and_counters() -> None:
    store = CognitionStateStore()

    assert store.start_request(make_request()).accepted is True
    assert store.record_token(make_token()).accepted is True

    store.reset()
    snapshot = store.snapshot()

    assert snapshot.state == CognitionRunState.IDLE
    assert snapshot.active_request_id is None
    assert snapshot.token_count == 0
    assert snapshot.started_count == 0
    assert snapshot.completed_count == 0
    assert snapshot.failed_count == 0
    assert snapshot.cancelled_count == 0


def test_cognition_state_store_rejects_operations_without_active_request() -> None:
    store = CognitionStateStore()

    assert store.record_token(make_token()).accepted is False
    assert store.complete_request(make_response()).accepted is False
    assert store.fail_request(make_failure()).accepted is False
    assert store.request_cancel().accepted is False
    assert store.cancel_request().accepted is False