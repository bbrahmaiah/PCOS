from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any
from urllib import request as http_request

import pytest

from jarvis.live import (
    LiveResponseContext,
    LiveResponseGenerationRequest,
    LiveResponseIntent,
    LiveResponseSurface,
    default_live_session_state,
    make_live_turn_id,
)
from jarvis.voice import VoiceOllamaGeneratorConfig, VoiceOllamaResponseGenerator


class FakeStreamingHTTPResponse:
    def __init__(self, lines: tuple[bytes, ...]) -> None:
        self._lines = lines

    def __enter__(self) -> FakeStreamingHTTPResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb

    def __iter__(self) -> Iterator[bytes]:
        return iter(self._lines)


def _request() -> LiveResponseGenerationRequest:
    return LiveResponseGenerationRequest(
        turn_id=make_live_turn_id(),
        intent=LiveResponseIntent.ANSWER,
        surface=LiveResponseSurface.VOICE,
        context=LiveResponseContext(
            live_state=default_live_session_state(),
            user_text="Jarvis status.",
            situation_summary="voice test",
        ),
    )


def test_voice_ollama_generator_streams_response_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_payload: dict[str, Any] = {}

    def fake_urlopen(request: Any, *, timeout: int) -> FakeStreamingHTTPResponse:
        del timeout
        raw_data = request.data
        payload = json.loads(raw_data.decode("utf-8"))
        captured_payload.update(payload)
        return FakeStreamingHTTPResponse(
            (
                b'{"response":"Online","done":false}\n',
                b'{"response":".","done":true}\n',
            )
        )

    monkeypatch.setattr(http_request, "urlopen", fake_urlopen)

    generator = VoiceOllamaResponseGenerator(
        config=VoiceOllamaGeneratorConfig(
            stream_response=True,
            max_sentences=1,
            prewarm_on_prepare=False,
        )
    )

    draft = generator.generate(_request())

    assert captured_payload["stream"] is True
    assert draft.text == "Online."
    assert draft.metadata["streamed"] is True
    assert draft.metadata["stream_chunk_count"] == 2
    assert draft.metadata["first_token_latency_ms"] is not None
