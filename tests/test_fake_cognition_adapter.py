from __future__ import annotations

import pytest

from jarvis.cognition import (
    CancellableCognitionAdapter,
    CognitionAdapter,
    CognitionAdapterCapability,
    CognitionAdapterStatus,
    CognitionFailureKind,
    CognitionRequest,
    FakeCognitionAdapter,
    FakeCognitionConfig,
    FakeCognitionMode,
    StreamingCognitionAdapter,
    adapter_supports,
)


def make_request(
    *,
    request_id: str = "request-1",
    text: str = "hello jarvis",
) -> CognitionRequest:
    return CognitionRequest(
        request_id=request_id,
        text=text,
    )


def test_fake_cognition_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        FakeCognitionConfig(adapter_name=" ").validate()

    with pytest.raises(ValueError):
        FakeCognitionConfig(default_response=" ").validate()

    with pytest.raises(ValueError):
        FakeCognitionConfig(echo_prefix=" ").validate()

    with pytest.raises(ValueError):
        FakeCognitionConfig(streaming_chunk_size=0).validate()

    with pytest.raises(ValueError):
        FakeCognitionConfig(scripted_responses={" ": "hello"}).validate()

    with pytest.raises(ValueError):
        FakeCognitionConfig(scripted_responses={"hello": " "}).validate()


def test_fake_cognition_adapter_protocols() -> None:
    adapter = FakeCognitionAdapter()

    cognition_adapter: CognitionAdapter = adapter
    streaming_adapter: StreamingCognitionAdapter = adapter
    cancellable_adapter: CancellableCognitionAdapter = adapter

    assert cognition_adapter.name == "fake_cognition_adapter"
    assert streaming_adapter.name == "fake_cognition_adapter"
    assert cancellable_adapter.cancel(request_id="request-1") is True


def test_fake_cognition_adapter_capabilities() -> None:
    adapter = FakeCognitionAdapter()

    assert adapter_supports(
        adapter,
        CognitionAdapterCapability.NON_STREAMING,
    )
    assert adapter_supports(
        adapter,
        CognitionAdapterCapability.STREAMING,
    )
    assert adapter_supports(
        adapter,
        CognitionAdapterCapability.CANCELLATION,
    )


def test_fake_cognition_adapter_generates_jarvis_response() -> None:
    adapter = FakeCognitionAdapter()
    request = make_request(text="Hello Jarvis")

    result = adapter.generate(request)

    assert result.succeeded is True
    assert result.response is not None
    assert result.response.text == "derived_fake_test_response::hello jarvis"
    assert result.response.metadata["adapter"] == "fake_cognition_adapter"
    assert result.response.metadata["mode"] == "jarvis"


def test_fake_cognition_adapter_generates_hearing_response() -> None:
    adapter = FakeCognitionAdapter()
    request = make_request(text="Jarvis can you hear me?")

    result = adapter.generate(request)

    assert result.succeeded is True
    assert result.response is not None
    assert result.response.text == (
        "derived_fake_test_response::jarvis can you hear me?"
    )


def test_fake_cognition_adapter_generates_phase3_response() -> None:
    adapter = FakeCognitionAdapter()
    request = make_request(text="What is phase 3?")

    result = adapter.generate(request)

    assert result.succeeded is True
    assert result.response is not None
    assert "Cognition Runtime" in result.response.text


def test_fake_cognition_adapter_echo_mode() -> None:
    adapter = FakeCognitionAdapter(
        config=FakeCognitionConfig(mode=FakeCognitionMode.ECHO)
    )
    request = make_request(text="open diagnostics")

    result = adapter.generate(request)

    assert result.succeeded is True
    assert result.response is not None
    assert result.response.text == "I heard you say: open diagnostics"


def test_fake_cognition_adapter_scripted_response() -> None:
    adapter = FakeCognitionAdapter(
        config=FakeCognitionConfig(
            mode=FakeCognitionMode.SCRIPTED,
            scripted_responses={
                "status report": "All systems are stable.",
            },
        )
    )
    request = make_request(text="Status   Report")

    result = adapter.generate(request)

    assert result.succeeded is True
    assert result.response is not None
    assert result.response.text == "All systems are stable."


def test_fake_cognition_adapter_scripted_mode_fallback() -> None:
    adapter = FakeCognitionAdapter(
        config=FakeCognitionConfig(
            mode=FakeCognitionMode.SCRIPTED,
            default_response="Standing by.",
        )
    )
    request = make_request(text="unknown command")

    result = adapter.generate(request)

    assert result.succeeded is True
    assert result.response is not None
    assert result.response.text == "Standing by."


def test_fake_cognition_adapter_failure_path() -> None:
    adapter = FakeCognitionAdapter()
    request = make_request(text="please simulate cognition failure now")

    result = adapter.generate(request)

    assert result.failed is True
    assert result.failure is not None
    assert result.failure.kind == CognitionFailureKind.ADAPTER_ERROR
    assert result.failure.message == "fake cognition failure requested"


def test_fake_cognition_adapter_cancel_before_generate() -> None:
    adapter = FakeCognitionAdapter()
    request = make_request()

    assert adapter.cancel(request_id=request.request_id, reason="user interrupted")

    result = adapter.generate(request)

    assert result.failed is True
    assert result.failure is not None
    assert result.failure.kind == CognitionFailureKind.CANCELLED


def test_fake_cognition_adapter_streams_tokens() -> None:
    adapter = FakeCognitionAdapter(
        config=FakeCognitionConfig(
            default_response="abcdefghijklmnopqrstuvwxyz",
            streaming_chunk_size=10,
        )
    )
    request = make_request(text="unknown")

    tokens = tuple(adapter.stream(request))

    assert len(tokens) == 3
    assert tokens[0].text == "abcdefghij"
    assert tokens[1].text == "klmnopqrst"
    assert tokens[2].text == "uvwxyz"
    assert tokens[2].final is True


def test_fake_cognition_adapter_snapshot_counts() -> None:
    adapter = FakeCognitionAdapter()

    assert adapter.generate(make_request()).succeeded is True
    assert adapter.generate(
        make_request(
            request_id="request-2",
            text="simulate cognition failure",
        )
    ).failed is True
    assert tuple(adapter.stream(make_request(request_id="request-3")))

    snapshot = adapter.snapshot()

    assert snapshot.name == "fake_cognition_adapter"
    assert snapshot.status == CognitionAdapterStatus.READY
    assert snapshot.request_count == 2
    assert snapshot.success_count == 1
    assert snapshot.failure_count == 1
    assert snapshot.streaming_count == 1
    assert snapshot.last_request_id == "request-2"
    assert snapshot.last_error == "fake cognition failure requested"


def test_fake_cognition_adapter_reset() -> None:
    adapter = FakeCognitionAdapter()

    assert adapter.generate(make_request()).succeeded is True
    assert adapter.cancel(request_id="request-1")

    adapter.reset()
    snapshot = adapter.snapshot()

    assert snapshot.request_count == 0
    assert snapshot.success_count == 0
    assert snapshot.failure_count == 0
    assert snapshot.cancelled_count == 0
    assert snapshot.streaming_count == 0
    assert snapshot.last_request_id is None
    assert snapshot.last_error is None


def test_fake_cognition_mode_values_are_stable() -> None:
    assert FakeCognitionMode.JARVIS.value == "jarvis"
    assert FakeCognitionMode.ECHO.value == "echo"
    assert FakeCognitionMode.SCRIPTED.value == "scripted"