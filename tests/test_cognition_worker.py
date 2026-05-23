from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from jarvis.cognition import (
    CognitionAdapterCapability,
    CognitionAdapterResult,
    CognitionAdapterSnapshot,
    CognitionFailure,
    CognitionFailureKind,
    CognitionRequest,
    CognitionResponse,
    CognitionRunState,
    CognitionWorker,
    CognitionWorkerConfig,
    FakeCognitionAdapter,
    FakeCognitionConfig,
    adapter_failure_result,
)


class RaisingCognitionAdapter:
    @property
    def name(self) -> str:
        return "raising_cognition_adapter"

    @property
    def capabilities(self) -> tuple[CognitionAdapterCapability, ...]:
        return (CognitionAdapterCapability.NON_STREAMING,)

    def generate(self, request: CognitionRequest) -> CognitionAdapterResult:
        raise RuntimeError("adapter exploded")

    def snapshot(self) -> CognitionAdapterSnapshot:
        return CognitionAdapterSnapshot(
            name=self.name,
            capabilities=self.capabilities,
            last_error="adapter exploded",
        )


class WrongRequestIdAdapter:
    @property
    def name(self) -> str:
        return "wrong_request_id_adapter"

    @property
    def capabilities(self) -> tuple[CognitionAdapterCapability, ...]:
        return (CognitionAdapterCapability.NON_STREAMING,)

    def generate(self, request: CognitionRequest) -> CognitionAdapterResult:
        started_at = datetime.now(UTC)
        finished_at = started_at + timedelta(milliseconds=1)

        return CognitionAdapterResult(
            request_id=request.request_id,
            response=CognitionResponse(
                request_id=request.request_id,
                text="This response is attached to a wrong result id.",
            ),
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=1.0,
        ).model_copy(update={"request_id": "wrong-request-id"})

    def snapshot(self) -> CognitionAdapterSnapshot:
        return CognitionAdapterSnapshot(
            name=self.name,
            capabilities=self.capabilities,
        )


class FailureAdapter:
    @property
    def name(self) -> str:
        return "failure_adapter"

    @property
    def capabilities(self) -> tuple[CognitionAdapterCapability, ...]:
        return (CognitionAdapterCapability.NON_STREAMING,)

    def generate(self, request: CognitionRequest) -> CognitionAdapterResult:
        started_at = datetime.now(UTC)
        finished_at = started_at + timedelta(milliseconds=1)
        failure = CognitionFailure(
            request_id=request.request_id,
            kind=CognitionFailureKind.ADAPTER_ERROR,
            message="planned failure",
        )

        return adapter_failure_result(
            request=request,
            failure=failure,
            started_at=started_at,
            finished_at=finished_at,
        )

    def snapshot(self) -> CognitionAdapterSnapshot:
        return CognitionAdapterSnapshot(
            name=self.name,
            capabilities=self.capabilities,
        )


def make_request(
    *,
    request_id: str = "request-1",
    text: str = "hello jarvis",
) -> CognitionRequest:
    return CognitionRequest(
        request_id=request_id,
        text=text,
        turn_id="turn-1",
        correlation_id="correlation-1",
    )


def test_cognition_worker_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        CognitionWorkerConfig(name=" ").validate()


def test_cognition_worker_rejects_when_not_started() -> None:
    worker = CognitionWorker(adapter=FakeCognitionAdapter())
    request = make_request()

    result = worker.process_request(request)
    snapshot = worker.snapshot()

    assert result.rejected is True
    assert result.reason == "worker is not started"
    assert snapshot.rejected_count == 1
    assert snapshot.processed_count == 0
    assert snapshot.last_error == "worker is not started"


def test_cognition_worker_start_stop_are_idempotent() -> None:
    worker = CognitionWorker(adapter=FakeCognitionAdapter())

    worker.on_start()
    worker.on_start()

    assert worker.started is True

    worker.on_stop()
    worker.on_stop()

    assert worker.started is False


def test_cognition_worker_processes_successful_request() -> None:
    worker = CognitionWorker(adapter=FakeCognitionAdapter())
    request = make_request(text="can you hear me")

    worker.on_start()
    result = worker.process_request(request)
    snapshot = worker.snapshot()

    assert result.succeeded is True
    assert result.response is not None
    assert result.response.text == "Yes sir. I can hear you clearly."
    assert snapshot.processed_count == 1
    assert snapshot.success_count == 1
    assert snapshot.failure_count == 0
    assert snapshot.state.state == CognitionRunState.COMPLETED
    assert snapshot.state.completed_count == 1


def test_cognition_worker_processes_adapter_failure_result() -> None:
    worker = CognitionWorker(adapter=FailureAdapter())

    worker.on_start()
    result = worker.process_request(make_request())
    snapshot = worker.snapshot()

    assert result.failed is True
    assert result.failure is not None
    assert result.failure.message == "planned failure"
    assert snapshot.failure_count == 1
    assert snapshot.state.state == CognitionRunState.FAILED
    assert snapshot.state.failed_count == 1


def test_cognition_worker_converts_adapter_exception_to_failure() -> None:
    worker = CognitionWorker(adapter=RaisingCognitionAdapter())

    worker.on_start()
    result = worker.process_request(make_request())
    snapshot = worker.snapshot()

    assert result.failed is True
    assert result.failure is not None
    assert result.failure.message == "RuntimeError: adapter exploded"
    assert snapshot.failure_count == 1
    assert snapshot.last_error == "RuntimeError: adapter exploded"


def test_cognition_worker_can_fail_fast_on_adapter_exception() -> None:
    worker = CognitionWorker(
        adapter=RaisingCognitionAdapter(),
        config=CognitionWorkerConfig(fail_fast_on_adapter_error=True),
    )

    worker.on_start()

    with pytest.raises(RuntimeError):
        worker.process_request(make_request())


def test_cognition_worker_rejects_overlapping_request() -> None:
    adapter = FakeCognitionAdapter(
        config=FakeCognitionConfig(default_response="Holding.")
    )
    worker = CognitionWorker(adapter=adapter)

    worker.on_start()

    request = make_request()
    start_transition = worker.state_store.start_request(request)

    assert start_transition.accepted is True

    result = worker.process_request(make_request(request_id="request-2"))
    snapshot = worker.snapshot()

    assert result.rejected is True
    assert result.reason == "another cognition request is already active"
    assert snapshot.rejected_count == 1


def test_cognition_worker_requests_cancel() -> None:
    adapter = FakeCognitionAdapter()
    worker = CognitionWorker(adapter=adapter)
    request = make_request()

    worker.on_start()
    assert worker.state_store.start_request(request).accepted is True

    cancelled = worker.request_cancel(
        request_id=request.request_id,
        reason="user interrupted",
    )
    snapshot = worker.snapshot()

    assert cancelled is True
    assert snapshot.cancel_requested_count == 1
    assert snapshot.state.state == CognitionRunState.CANCELLING
    assert snapshot.state.cancelling is True
    assert snapshot.adapter.cancelled_count == 1


def test_cognition_worker_rejects_cancel_without_active_request() -> None:
    worker = CognitionWorker(adapter=FakeCognitionAdapter())

    worker.on_start()

    assert worker.request_cancel(reason="nothing active") is False
    assert worker.snapshot().last_error == "no active cognition request"


def test_cognition_worker_confirms_cancelled() -> None:
    adapter = FakeCognitionAdapter()
    worker = CognitionWorker(adapter=adapter)
    request = make_request()

    worker.on_start()
    assert worker.state_store.start_request(request).accepted is True
    assert worker.request_cancel(request_id=request.request_id) is True

    confirmed = worker.confirm_cancelled(
        request_id=request.request_id,
        reason="cancelled",
    )
    snapshot = worker.snapshot()

    assert confirmed is True
    assert snapshot.state.state == CognitionRunState.CANCELLED
    assert snapshot.state.cancelled_count == 1


def test_cognition_worker_reset_clears_counters_and_state() -> None:
    worker = CognitionWorker(adapter=FakeCognitionAdapter())

    worker.on_start()
    assert worker.process_request(make_request()).succeeded is True

    worker.reset()
    snapshot = worker.snapshot()

    assert snapshot.processed_count == 0
    assert snapshot.success_count == 0
    assert snapshot.failure_count == 0
    assert snapshot.rejected_count == 0
    assert snapshot.state.state == CognitionRunState.IDLE
    assert snapshot.state.completed_count == 0


def test_cognition_worker_snapshot_includes_adapter_diagnostics() -> None:
    worker = CognitionWorker(adapter=FakeCognitionAdapter())

    worker.on_start()
    assert worker.process_request(make_request()).succeeded is True

    snapshot = worker.snapshot()

    assert snapshot.adapter.name == "fake_cognition_adapter"
    assert snapshot.adapter.request_count == 1
    assert snapshot.adapter.success_count == 1