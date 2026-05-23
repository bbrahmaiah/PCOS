from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from jarvis.cognition import (
    CognitionRequest,
    LocalLLMAdapter,
    LocalLLMBackendStatus,
    OllamaBackendConfig,
    OllamaLocalLLMBackend,
    StreamingTokenPipeline,
    is_ollama_connection_error,
)


class FakeOllamaHttpClient:
    def __init__(
        self,
        *,
        response_text: str = "JARVIS cognition is online.",
        fail: bool = False,
    ) -> None:
        self.response_text = response_text
        self.fail = fail
        self.post_count = 0
        self.stream_count = 0
        self.last_path: str | None = None
        self.last_payload: Mapping[str, Any] | None = None
        self.last_timeout_seconds: float | None = None

    def post_json(
        self,
        *,
        path: str,
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        self.post_count += 1
        self.last_path = path
        self.last_payload = payload
        self.last_timeout_seconds = timeout_seconds

        if self.fail:
            raise ConnectionError("ollama unavailable")

        return {
            "response": self.response_text,
            "done": True,
            "eval_count": 12,
            "prompt_eval_count": 8,
        }

    def stream_json(
        self,
        *,
        path: str,
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Iterator[dict[str, Any]]:
        self.stream_count += 1
        self.last_path = path
        self.last_payload = payload
        self.last_timeout_seconds = timeout_seconds

        if self.fail:
            raise ConnectionError("ollama unavailable")

        yield {"response": "JARVIS", "done": False}
        yield {"response": " cognition", "done": False}
        yield {"response": " is online.", "done": True}


def make_request(*, request_id: str = "request-1") -> CognitionRequest:
    return CognitionRequest(
        request_id=request_id,
        text="Confirm cognition is online.",
    )


def test_ollama_backend_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        OllamaBackendConfig(name=" ").validate()

    with pytest.raises(ValueError):
        OllamaBackendConfig(base_url=" ").validate()

    with pytest.raises(ValueError):
        OllamaBackendConfig(model=" ").validate()

    with pytest.raises(ValueError):
        OllamaBackendConfig(timeout_seconds=0).validate()

    with pytest.raises(ValueError):
        OllamaBackendConfig(temperature=-0.1).validate()

    with pytest.raises(ValueError):
        OllamaBackendConfig(top_p=0).validate()

    with pytest.raises(ValueError):
        OllamaBackendConfig(num_predict=0).validate()


def test_ollama_backend_generates_response() -> None:
    client = FakeOllamaHttpClient()
    backend = OllamaLocalLLMBackend(client=client)
    result = backend.generate(
        prompt="Hello",
        system_prompt="System",
        request=make_request(),
    )
    snapshot = backend.snapshot()

    assert result.text == "JARVIS cognition is online."
    assert result.metadata["model"] == "llama3.2:3b"
    assert client.post_count == 1
    assert client.last_path == "/api/generate"
    assert client.last_payload is not None
    assert client.last_payload["stream"] is False
    assert snapshot.status == LocalLLMBackendStatus.READY
    assert snapshot.request_count == 1


def test_ollama_backend_raises_on_empty_response() -> None:
    backend = OllamaLocalLLMBackend(
        client=FakeOllamaHttpClient(response_text=" ")
    )

    with pytest.raises(ValueError):
        backend.generate(
            prompt="Hello",
            system_prompt="System",
            request=make_request(),
        )

    assert backend.snapshot().status == LocalLLMBackendStatus.UNAVAILABLE


def test_ollama_backend_streams_tokens_preserving_whitespace() -> None:
    client = FakeOllamaHttpClient()
    backend = OllamaLocalLLMBackend(client=client)

    tokens = tuple(
        backend.stream(
            prompt="Hello",
            system_prompt="System",
            request=make_request(),
        )
    )
    snapshot = backend.snapshot()

    assert len(tokens) == 3
    assert tokens[0].text == "JARVIS"
    assert tokens[1].text == " cognition"
    assert tokens[2].text == " is online."
    assert tokens[2].final is True
    assert client.stream_count == 1
    assert client.last_payload is not None
    assert client.last_payload["stream"] is True
    assert snapshot.streaming_count == 1
    assert snapshot.status == LocalLLMBackendStatus.READY


def test_ollama_backend_cancel_marks_request_cancelled() -> None:
    backend = OllamaLocalLLMBackend(client=FakeOllamaHttpClient())

    assert backend.cancel(request_id="request-1", reason="user interrupted")

    snapshot = backend.snapshot()

    assert snapshot.cancelled_count == 1
    assert snapshot.last_error == "user interrupted"


def test_ollama_backend_rejects_empty_cancel_request() -> None:
    backend = OllamaLocalLLMBackend(client=FakeOllamaHttpClient())

    assert backend.cancel(request_id=" ") is False
    assert backend.snapshot().cancelled_count == 0


def test_ollama_backend_integrates_with_local_llm_adapter() -> None:
    backend = OllamaLocalLLMBackend(client=FakeOllamaHttpClient())
    adapter = LocalLLMAdapter(backend=backend)

    result = adapter.generate(make_request())

    assert result.succeeded is True
    assert result.response is not None
    assert result.response.text == "JARVIS cognition is online."


def test_ollama_backend_integrates_with_streaming_pipeline() -> None:
    backend = OllamaLocalLLMBackend(client=FakeOllamaHttpClient())
    adapter = LocalLLMAdapter(backend=backend)
    pipeline = StreamingTokenPipeline(adapter=adapter)

    result = pipeline.stream_request(make_request())

    assert result.completed is True
    assert result.response is not None
    assert result.response.text == "JARVIS cognition is online."
    assert len(result.tokens) == 3
    assert len(result.speech_chunks) >= 1


def test_ollama_connection_error_helper() -> None:
    assert is_ollama_connection_error(ConnectionError("offline")) is True
    assert is_ollama_connection_error(RuntimeError("other")) is False