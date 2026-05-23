from __future__ import annotations

from collections.abc import Iterator

import pytest

from jarvis.cognition import (
    CognitionAdapter,
    CognitionAdapterCapability,
    CognitionContext,
    CognitionContextItem,
    CognitionRequest,
    LocalLLMAdapter,
    LocalLLMAdapterConfig,
    LocalLLMBackendResult,
    LocalLLMBackendSnapshot,
    LocalLLMBackendStatus,
    LocalLLMBackendToken,
    StreamingCognitionAdapter,
    adapter_supports,
)


class StubLocalLLMBackend:
    def __init__(
        self,
        *,
        text: str = "Yes sir. I can help with that.",
        confidence: float = 0.88,
        fail_generate: bool = False,
    ) -> None:
        self.text = text
        self.confidence = confidence
        self.fail_generate = fail_generate
        self.request_count = 0
        self.streaming_count = 0
        self.cancelled_count = 0
        self.last_prompt: str | None = None
        self.last_system_prompt: str | None = None
        self.last_request_id: str | None = None
        self.last_cancel_reason: str | None = None

    @property
    def name(self) -> str:
        return "stub_local_llm_backend"

    def generate(
        self,
        *,
        prompt: str,
        system_prompt: str,
        request: CognitionRequest,
    ) -> LocalLLMBackendResult:
        self.request_count += 1
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        self.last_request_id = request.request_id

        if self.fail_generate:
            raise RuntimeError("local model unavailable")

        return LocalLLMBackendResult(
            text=self.text,
            confidence=self.confidence,
            metadata={"backend_mode": "stub"},
        )

    def stream(
        self,
        *,
        prompt: str,
        system_prompt: str,
        request: CognitionRequest,
    ) -> Iterator[LocalLLMBackendToken]:
        self.streaming_count += 1
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        self.last_request_id = request.request_id

        yield LocalLLMBackendToken(text="Yes sir.")
        yield LocalLLMBackendToken(text=" I can help", final=False)
        yield LocalLLMBackendToken(text=" with that.", final=True)

    def cancel(self, *, request_id: str, reason: str | None = None) -> bool:
        self.cancelled_count += 1
        self.last_request_id = request_id
        self.last_cancel_reason = reason
        return True

    def snapshot(self) -> LocalLLMBackendSnapshot:
        return LocalLLMBackendSnapshot(
            name=self.name,
            status=LocalLLMBackendStatus.READY,
            request_count=self.request_count,
            streaming_count=self.streaming_count,
            cancelled_count=self.cancelled_count,
        )


def make_request(
    *,
    text: str = "hello jarvis",
    context: CognitionContext | None = None,
) -> CognitionRequest:
    return CognitionRequest(
        request_id="request-1",
        text=text,
        context=context or CognitionContext(),
    )


def test_local_llm_adapter_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        LocalLLMAdapterConfig(adapter_name=" ").validate()

    with pytest.raises(ValueError):
        LocalLLMAdapterConfig(system_prompt=" ").validate()

    with pytest.raises(ValueError):
        LocalLLMAdapterConfig(max_prompt_chars=0).validate()

    with pytest.raises(ValueError):
        LocalLLMAdapterConfig(max_context_items=-1).validate()

    with pytest.raises(ValueError):
        LocalLLMAdapterConfig(default_confidence=1.1).validate()


def test_local_llm_backend_result_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        LocalLLMBackendResult(text=" ").validate()

    with pytest.raises(ValueError):
        LocalLLMBackendResult(text="hello", confidence=-0.1).validate()


def test_local_llm_backend_token_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        LocalLLMBackendToken(text=" ").validate()


def test_local_llm_adapter_protocols() -> None:
    adapter = LocalLLMAdapter(backend=StubLocalLLMBackend())

    cognition_adapter: CognitionAdapter = adapter
    streaming_adapter: StreamingCognitionAdapter = adapter

    assert cognition_adapter.name == "local_llm_adapter"
    assert streaming_adapter.name == "local_llm_adapter"


def test_local_llm_adapter_capabilities() -> None:
    adapter = LocalLLMAdapter(backend=StubLocalLLMBackend())

    assert adapter_supports(adapter, CognitionAdapterCapability.NON_STREAMING)
    assert adapter_supports(adapter, CognitionAdapterCapability.STREAMING)
    assert adapter_supports(adapter, CognitionAdapterCapability.CANCELLATION)
    assert adapter_supports(adapter, CognitionAdapterCapability.MEMORY_CONTEXT)


def test_local_llm_adapter_generates_response() -> None:
    backend = StubLocalLLMBackend()
    adapter = LocalLLMAdapter(backend=backend)
    request = make_request(text="Can you help me?")

    result = adapter.generate(request)
    snapshot = adapter.snapshot()

    assert result.succeeded is True
    assert result.response is not None
    assert result.response.text == "Yes sir. I can help with that."
    assert result.response.confidence == 0.88
    assert result.response.metadata["adapter"] == "local_llm_adapter"
    assert result.response.metadata["backend"] == "stub_local_llm_backend"
    assert backend.last_prompt is not None
    assert "Can you help me?" in backend.last_prompt
    assert backend.last_system_prompt is not None
    assert snapshot.request_count == 1
    assert snapshot.success_count == 1


def test_local_llm_adapter_handles_backend_failure() -> None:
    backend = StubLocalLLMBackend(fail_generate=True)
    adapter = LocalLLMAdapter(backend=backend)

    result = adapter.generate(make_request())
    snapshot = adapter.snapshot()

    assert result.failed is True
    assert result.failure is not None
    assert result.failure.message == "RuntimeError: local model unavailable"
    assert snapshot.failure_count == 1
    assert snapshot.last_error == "RuntimeError: local model unavailable"


def test_local_llm_adapter_streams_tokens_preserving_whitespace() -> None:
    backend = StubLocalLLMBackend()
    adapter = LocalLLMAdapter(backend=backend)

    tokens = tuple(adapter.stream(make_request()))

    assert len(tokens) == 3
    assert tokens[0].text == "Yes sir."
    assert tokens[1].text == " I can help"
    assert tokens[2].text == " with that."
    assert tokens[2].final is True
    assert tokens[2].kind.value == "final"


def test_local_llm_adapter_cancel_forwards_to_backend() -> None:
    backend = StubLocalLLMBackend()
    adapter = LocalLLMAdapter(backend=backend)

    cancelled = adapter.cancel(
        request_id="request-1",
        reason="user interrupted",
    )
    snapshot = adapter.snapshot()

    assert cancelled is True
    assert backend.cancelled_count == 1
    assert backend.last_cancel_reason == "user interrupted"
    assert snapshot.cancelled_count == 1


def test_local_llm_adapter_rejects_empty_cancel_request_id() -> None:
    backend = StubLocalLLMBackend()
    adapter = LocalLLMAdapter(backend=backend)

    assert adapter.cancel(request_id=" ") is False
    assert backend.cancelled_count == 0


def test_local_llm_adapter_builds_prompt_with_context() -> None:
    context = CognitionContext(
        items=(
            CognitionContextItem(
                kind="session",
                text="User prefers concise engineering answers.",
            ),
        )
    )
    backend = StubLocalLLMBackend()
    adapter = LocalLLMAdapter(backend=backend)

    prompt = adapter.build_prompt(
        make_request(
            text="What should we build next?",
            context=context,
        )
    )

    assert "Relevant context:" in prompt
    assert "User prefers concise engineering answers." in prompt
    assert "What should we build next?" in prompt


def test_local_llm_adapter_can_omit_context() -> None:
    context = CognitionContext(
        items=(
            CognitionContextItem(
                kind="session",
                text="Hidden context.",
            ),
        )
    )
    adapter = LocalLLMAdapter(
        backend=StubLocalLLMBackend(),
        config=LocalLLMAdapterConfig(include_context=False),
    )

    prompt = adapter.build_prompt(make_request(context=context))

    assert "Hidden context." not in prompt
    assert "User said:" in prompt


def test_local_llm_adapter_bounds_prompt_length() -> None:
    adapter = LocalLLMAdapter(
        backend=StubLocalLLMBackend(),
        config=LocalLLMAdapterConfig(max_prompt_chars=40),
    )

    prompt = adapter.build_prompt(make_request(text="x" * 500))

    assert len(prompt) <= 40


def test_local_llm_adapter_reset_clears_adapter_counters() -> None:
    adapter = LocalLLMAdapter(backend=StubLocalLLMBackend())

    assert adapter.generate(make_request()).succeeded is True

    adapter.reset()
    snapshot = adapter.snapshot()

    assert snapshot.request_count == 0
    assert snapshot.success_count == 0
    assert snapshot.failure_count == 0
    assert snapshot.cancelled_count == 0
    assert snapshot.streaming_count == 0
    assert snapshot.last_request_id is None
    assert snapshot.last_error is None


def test_local_llm_backend_status_values_are_stable() -> None:
    assert LocalLLMBackendStatus.READY.value == "ready"
    assert LocalLLMBackendStatus.BUSY.value == "busy"
    assert LocalLLMBackendStatus.UNAVAILABLE.value == "unavailable"