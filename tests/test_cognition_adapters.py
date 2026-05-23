from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from jarvis.cognition import (
    CancellableCognitionAdapter,
    CognitionAdapter,
    CognitionAdapterCapability,
    CognitionAdapterResult,
    CognitionAdapterSnapshot,
    CognitionAdapterStatus,
    CognitionFailure,
    CognitionFailureKind,
    CognitionRequest,
    CognitionResponse,
    CognitionToken,
    StreamingCognitionAdapter,
    adapter_failure_result,
    adapter_success_result,
    adapter_supports,
    duration_ms_between,
)


class StubCognitionAdapter:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def name(self) -> str:
        return "stub_cognition_adapter"

    @property
    def capabilities(self) -> tuple[CognitionAdapterCapability, ...]:
        return (CognitionAdapterCapability.NON_STREAMING,)

    def generate(self, request: CognitionRequest) -> CognitionAdapterResult:
        self.calls += 1
        started_at = datetime.now(UTC)
        finished_at = started_at + timedelta(milliseconds=10)
        response = CognitionResponse(
            request_id=request.request_id,
            text="Yes sir.",
        )

        return adapter_success_result(
            request=request,
            response=response,
            started_at=started_at,
            finished_at=finished_at,
        )

    def snapshot(self) -> CognitionAdapterSnapshot:
        return CognitionAdapterSnapshot(
            name=self.name,
            capabilities=self.capabilities,
            request_count=self.calls,
            success_count=self.calls,
        )


class StubStreamingCognitionAdapter(StubCognitionAdapter):
    @property
    def capabilities(self) -> tuple[CognitionAdapterCapability, ...]:
        return (
            CognitionAdapterCapability.NON_STREAMING,
            CognitionAdapterCapability.STREAMING,
        )

    def stream(self, request: CognitionRequest) -> Iterator[CognitionToken]:
        yield CognitionToken(
            request_id=request.request_id,
            index=0,
            text="Hello",
        )
        yield CognitionToken(
            request_id=request.request_id,
            index=1,
            text=" sir.",
            final=True,
        )


class StubCancellableAdapter(StubCognitionAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.cancelled_request_id: str | None = None
        self.cancel_reason: str | None = None

    @property
    def capabilities(self) -> tuple[CognitionAdapterCapability, ...]:
        return (
            CognitionAdapterCapability.NON_STREAMING,
            CognitionAdapterCapability.CANCELLATION,
        )

    def cancel(self, *, request_id: str, reason: str | None = None) -> bool:
        self.cancelled_request_id = request_id
        self.cancel_reason = reason
        return True


def make_request() -> CognitionRequest:
    return CognitionRequest(
        request_id="request-1",
        text="hello jarvis",
    )


def make_response(*, request_id: str = "request-1") -> CognitionResponse:
    return CognitionResponse(
        request_id=request_id,
        text="Yes sir.",
    )


def make_failure(*, request_id: str = "request-1") -> CognitionFailure:
    return CognitionFailure(
        request_id=request_id,
        kind=CognitionFailureKind.ADAPTER_ERROR,
        message="adapter failed",
    )


def test_duration_ms_between_returns_positive_duration() -> None:
    started_at = datetime.now(UTC)
    finished_at = started_at + timedelta(milliseconds=25)

    assert duration_ms_between(started_at, finished_at) == 25.0


def test_duration_ms_between_clamps_negative_duration() -> None:
    started_at = datetime.now(UTC)
    finished_at = started_at - timedelta(milliseconds=25)

    assert duration_ms_between(started_at, finished_at) == 0.0


def test_cognition_adapter_result_accepts_success() -> None:
    request = make_request()
    started_at = datetime.now(UTC)
    finished_at = started_at + timedelta(milliseconds=5)

    result = CognitionAdapterResult(
        request_id=request.request_id,
        response=make_response(),
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=5.0,
    )

    assert result.succeeded is True
    assert result.failed is False
    assert result.response is not None
    assert result.failure is None


def test_cognition_adapter_result_accepts_failure() -> None:
    request = make_request()
    started_at = datetime.now(UTC)
    finished_at = started_at + timedelta(milliseconds=5)

    result = CognitionAdapterResult(
        request_id=request.request_id,
        failure=make_failure(),
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=5.0,
    )

    assert result.succeeded is False
    assert result.failed is True
    assert result.response is None
    assert result.failure is not None


def test_cognition_adapter_result_rejects_missing_payload() -> None:
    started_at = datetime.now(UTC)
    finished_at = started_at + timedelta(milliseconds=5)

    with pytest.raises(ValidationError):
        CognitionAdapterResult(
            request_id="request-1",
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=5.0,
        )


def test_cognition_adapter_result_rejects_ambiguous_payload() -> None:
    started_at = datetime.now(UTC)
    finished_at = started_at + timedelta(milliseconds=5)

    with pytest.raises(ValidationError):
        CognitionAdapterResult(
            request_id="request-1",
            response=make_response(),
            failure=make_failure(),
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=5.0,
        )


def test_cognition_adapter_result_rejects_wrong_response_request_id() -> None:
    started_at = datetime.now(UTC)
    finished_at = started_at + timedelta(milliseconds=5)

    with pytest.raises(ValidationError):
        CognitionAdapterResult(
            request_id="request-1",
            response=make_response(request_id="wrong"),
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=5.0,
        )


def test_cognition_adapter_result_rejects_wrong_failure_request_id() -> None:
    started_at = datetime.now(UTC)
    finished_at = started_at + timedelta(milliseconds=5)

    with pytest.raises(ValidationError):
        CognitionAdapterResult(
            request_id="request-1",
            failure=make_failure(request_id="wrong"),
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=5.0,
        )


def test_cognition_adapter_result_rejects_invalid_timestamps() -> None:
    started_at = datetime.now(UTC)
    finished_at = started_at - timedelta(milliseconds=5)

    with pytest.raises(ValidationError):
        CognitionAdapterResult(
            request_id="request-1",
            response=make_response(),
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=0.0,
        )


def test_cognition_adapter_snapshot_defaults() -> None:
    snapshot = CognitionAdapterSnapshot(name="adapter")

    assert snapshot.name == "adapter"
    assert snapshot.status == CognitionAdapterStatus.READY
    assert snapshot.capabilities == ()
    assert snapshot.request_count == 0
    assert snapshot.success_count == 0
    assert snapshot.failure_count == 0
    assert snapshot.cancelled_count == 0
    assert snapshot.streaming_count == 0


def test_cognition_adapter_protocol_accepts_stub() -> None:
    adapter: CognitionAdapter = StubCognitionAdapter()
    request = make_request()

    result = adapter.generate(request)

    assert adapter.name == "stub_cognition_adapter"
    assert result.succeeded is True
    assert result.response is not None
    assert result.response.text == "Yes sir."
    assert adapter.snapshot().request_count == 1


def test_streaming_cognition_adapter_protocol_accepts_stub() -> None:
    adapter: StreamingCognitionAdapter = StubStreamingCognitionAdapter()
    request = make_request()

    tokens = tuple(adapter.stream(request))

    assert len(tokens) == 2
    assert tokens[0].text == "Hello"
    assert tokens[1].final is True
    assert adapter_supports(adapter, CognitionAdapterCapability.STREAMING) is True


def test_cancellable_cognition_adapter_protocol_accepts_stub() -> None:
    adapter = StubCancellableAdapter()
    protocol_adapter: CancellableCognitionAdapter = adapter

    cancelled = protocol_adapter.cancel(
        request_id="request-1",
        reason="user interrupted",
    )

    assert cancelled is True
    assert adapter.cancelled_request_id == "request-1"
    assert adapter.cancel_reason == "user interrupted"


def test_adapter_supports_capability() -> None:
    adapter = StubCognitionAdapter()

    assert adapter_supports(
        adapter,
        CognitionAdapterCapability.NON_STREAMING,
    )
    assert not adapter_supports(
        adapter,
        CognitionAdapterCapability.STREAMING,
    )


def test_adapter_success_result_helper() -> None:
    request = make_request()
    started_at = datetime.now(UTC)
    finished_at = started_at + timedelta(milliseconds=20)

    result = adapter_success_result(
        request=request,
        response=make_response(),
        started_at=started_at,
        finished_at=finished_at,
        metadata={"source": "test"},
    )

    assert result.succeeded is True
    assert result.duration_ms == 20.0
    assert result.metadata["source"] == "test"


def test_adapter_failure_result_helper() -> None:
    request = make_request()
    started_at = datetime.now(UTC)
    finished_at = started_at + timedelta(milliseconds=20)

    result = adapter_failure_result(
        request=request,
        failure=make_failure(),
        started_at=started_at,
        finished_at=finished_at,
        metadata={"source": "test"},
    )

    assert result.failed is True
    assert result.duration_ms == 20.0
    assert result.metadata["source"] == "test"


def test_adapter_capability_values_are_stable() -> None:
    assert CognitionAdapterCapability.NON_STREAMING.value == "non_streaming"
    assert CognitionAdapterCapability.STREAMING.value == "streaming"
    assert CognitionAdapterCapability.CANCELLATION.value == "cancellation"
    assert CognitionAdapterCapability.MEMORY_CONTEXT.value == "memory_context"
    assert CognitionAdapterCapability.TOOL_PLANNING.value == "tool_planning"