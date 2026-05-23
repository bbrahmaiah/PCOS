from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from jarvis.cognition import (
    CognitionAdapterCapability,
    CognitionAdapterResult,
    CognitionAdapterSnapshot,
    CognitionEngine,
    CognitionEngineConfig,
    CognitionFailure,
    CognitionFailureKind,
    CognitionPlanKind,
    CognitionRequest,
    CognitionResponse,
    CognitionResponseKind,
    CognitionRuntimePolicy,
    FakeCognitionAdapter,
    adapter_failure_result,
)


class RaisingEngineAdapter:
    @property
    def name(self) -> str:
        return "raising_engine_adapter"

    @property
    def capabilities(self) -> tuple[CognitionAdapterCapability, ...]:
        return (CognitionAdapterCapability.NON_STREAMING,)

    def generate(self, request: CognitionRequest) -> CognitionAdapterResult:
        raise RuntimeError("engine adapter exploded")

    def snapshot(self) -> CognitionAdapterSnapshot:
        return CognitionAdapterSnapshot(
            name=self.name,
            capabilities=self.capabilities,
            last_error="engine adapter exploded",
        )


class FailureEngineAdapter:
    @property
    def name(self) -> str:
        return "failure_engine_adapter"

    @property
    def capabilities(self) -> tuple[CognitionAdapterCapability, ...]:
        return (CognitionAdapterCapability.NON_STREAMING,)

    def generate(self, request: CognitionRequest) -> CognitionAdapterResult:
        started_at = datetime.now(UTC)
        finished_at = started_at + timedelta(milliseconds=1)
        failure = CognitionFailure(
            request_id=request.request_id,
            kind=CognitionFailureKind.ADAPTER_ERROR,
            message="planned engine adapter failure",
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


class WrongResultIdEngineAdapter:
    @property
    def name(self) -> str:
        return "wrong_result_id_engine_adapter"

    @property
    def capabilities(self) -> tuple[CognitionAdapterCapability, ...]:
        return (CognitionAdapterCapability.NON_STREAMING,)

    def generate(self, request: CognitionRequest) -> CognitionAdapterResult:
        started_at = datetime.now(UTC)
        finished_at = started_at + timedelta(milliseconds=1)
        response = CognitionResponse(
            request_id=request.request_id,
            text="wrong result id",
        )

        return CognitionAdapterResult(
            request_id=request.request_id,
            response=response,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=1.0,
        ).model_copy(update={"request_id": "wrong"})

    def snapshot(self) -> CognitionAdapterSnapshot:
        return CognitionAdapterSnapshot(
            name=self.name,
            capabilities=self.capabilities,
        )


def make_request(
    *,
    request_id: str = "request-1",
    text: str = "hello jarvis",
    max_response_chars: int = 1_200,
) -> CognitionRequest:
    return CognitionRequest(
        request_id=request_id,
        text=text,
        policy=CognitionRuntimePolicy(
            max_response_chars=max_response_chars,
        ),
    )


def test_cognition_engine_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        CognitionEngineConfig(name=" ").validate()

    with pytest.raises(ValueError):
        CognitionEngineConfig(fallback_response_text=" ").validate()

    with pytest.raises(ValueError):
        CognitionEngineConfig(max_input_chars=0).validate()


def test_cognition_engine_creates_direct_answer_plan() -> None:
    engine = CognitionEngine(adapter=FakeCognitionAdapter())
    request = make_request(text="hello jarvis")

    plan = engine.create_plan(request)

    assert plan.kind == CognitionPlanKind.DIRECT_ANSWER
    assert plan.confidence == 1.0
    assert plan.needs_clarification is False
    assert plan.metadata["engine"] == "cognition_engine"


def test_cognition_engine_creates_clarification_plan() -> None:
    engine = CognitionEngine(adapter=FakeCognitionAdapter())
    request = make_request(text="what")

    plan = engine.create_plan(request)

    assert plan.kind == CognitionPlanKind.ASK_CLARIFICATION
    assert plan.needs_clarification is True
    assert "empty request text" not in plan.notes


def test_cognition_engine_runs_successfully() -> None:
    engine = CognitionEngine(adapter=FakeCognitionAdapter())
    request = make_request(text="can you hear me")

    result = engine.run(request)
    snapshot = engine.snapshot()

    assert result.succeeded is True
    assert result.response is not None
    assert result.failure is None
    assert result.response.text == "Yes sir. I can hear you clearly."
    assert result.response.plan == result.plan
    assert result.response.metadata["engine"] == "cognition_engine"
    assert snapshot.request_count == 1
    assert snapshot.success_count == 1
    assert snapshot.failure_count == 0
    assert snapshot.last_request_id == "request-1"
    assert snapshot.last_response_id == result.response.response_id


def test_cognition_engine_enforces_response_length_limit() -> None:
    engine = CognitionEngine(adapter=FakeCognitionAdapter())
    request = make_request(
        text="what did we build",
        max_response_chars=20,
    )

    result = engine.run(request)

    assert result.succeeded is True
    assert result.response is not None
    assert len(result.response.text) <= 20


def test_cognition_engine_can_disable_response_length_limit() -> None:
    engine = CognitionEngine(
        adapter=FakeCognitionAdapter(),
        config=CognitionEngineConfig(enforce_spoken_limit=False),
    )
    request = make_request(
        text="what did we build",
        max_response_chars=5,
    )

    result = engine.run(request)

    assert result.succeeded is True
    assert result.response is not None
    assert len(result.response.text) > 5


def test_cognition_engine_rejects_oversized_input() -> None:
    engine = CognitionEngine(
        adapter=FakeCognitionAdapter(),
        config=CognitionEngineConfig(max_input_chars=5),
    )
    request = make_request(text="this is too long")

    result = engine.run(request)
    snapshot = engine.snapshot()

    assert result.failed is True
    assert result.failure is not None
    assert result.failure.kind == CognitionFailureKind.VALIDATION_ERROR
    assert result.failure.message == "request text exceeds engine input limit"
    assert snapshot.failure_count == 1
    assert snapshot.last_error == "request text exceeds engine input limit"


def test_cognition_engine_handles_adapter_failure_result() -> None:
    engine = CognitionEngine(adapter=FailureEngineAdapter())
    request = make_request()

    result = engine.run(request)
    snapshot = engine.snapshot()

    assert result.failed is True
    assert result.failure is not None
    assert result.failure.message == "planned engine adapter failure"
    assert snapshot.failure_count == 1
    assert snapshot.last_failure_id == result.failure.failure_id


def test_cognition_engine_handles_adapter_exception() -> None:
    engine = CognitionEngine(adapter=RaisingEngineAdapter())
    request = make_request()

    result = engine.run(request)
    snapshot = engine.snapshot()

    assert result.failed is True
    assert result.failure is not None
    assert result.failure.message == "RuntimeError: engine adapter exploded"
    assert snapshot.failure_count == 1
    assert snapshot.last_error == "RuntimeError: engine adapter exploded"


def test_cognition_engine_detects_wrong_adapter_result_id() -> None:
    engine = CognitionEngine(adapter=WrongResultIdEngineAdapter())
    request = make_request()

    result = engine.run(request)

    assert result.failed is True
    assert result.failure is not None
    assert result.failure.message == "adapter result request_id does not match request"


def test_cognition_engine_builds_fallback_response() -> None:
    engine = CognitionEngine(adapter=FailureEngineAdapter())
    request = make_request()
    result = engine.run(request)

    assert result.failure is not None

    fallback = engine.fallback_response_for(
        request=request,
        failure=result.failure,
    )
    snapshot = engine.snapshot()

    assert fallback.kind == CognitionResponseKind.ERROR_FALLBACK
    assert fallback.text == "I had trouble thinking that through, sir."
    assert fallback.confidence == 0.0
    assert snapshot.fallback_count == 1


def test_cognition_engine_reset_clears_counters() -> None:
    engine = CognitionEngine(adapter=FakeCognitionAdapter())

    assert engine.run(make_request()).succeeded is True

    engine.reset()
    snapshot = engine.snapshot()

    assert snapshot.request_count == 0
    assert snapshot.success_count == 0
    assert snapshot.failure_count == 0
    assert snapshot.fallback_count == 0
    assert snapshot.last_request_id is None
    assert snapshot.last_response_id is None
    assert snapshot.last_error is None


def test_cognition_engine_snapshot_includes_adapter_snapshot() -> None:
    engine = CognitionEngine(adapter=FakeCognitionAdapter())

    assert engine.run(make_request()).succeeded is True

    snapshot = engine.snapshot()

    assert snapshot.adapter.name == "fake_cognition_adapter"
    assert snapshot.adapter.request_count == 1
    assert snapshot.adapter.success_count == 1