from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from urllib import request
from urllib.error import URLError

from jarvis.live.contracts import LiveResponseGenerationSource
from jarvis.live.response_boundary import (
    LiveResponseDraft,
    LiveResponseGenerationRequest,
    LiveResponseGenerator,
)


class OllamaGeneratorStatus(StrEnum):
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class OllamaGeneratorConfig:
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2:3b"
    timeout_seconds: int = 60
    max_sentences: int = 3
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("ollama base_url cannot be empty.")
        if not self.model.strip():
            raise ValueError("ollama model cannot be empty.")
        if self.timeout_seconds < 1:
            raise ValueError("ollama timeout_seconds must be positive.")
        if self.max_sentences < 1:
            raise ValueError("ollama max_sentences must be positive.")


class OllamaLiveResponseGenerator(LiveResponseGenerator):
    """
    Real Ollama-backed response generator for the live JARVIS runner.

    This class does not:
    - call TTS
    - execute tools
    - access memory directly
    - bypass LiveResponseBoundaryRuntime
    - return scripted conversational responses

    It only generates a response draft from the live response-generation request.
    The 50A.5 boundary still decides whether it may become speech.
    """

    def __init__(self, config: OllamaGeneratorConfig | None = None) -> None:
        self._config = config or OllamaGeneratorConfig()

    def generate(
        self,
        request_data: LiveResponseGenerationRequest,
    ) -> LiveResponseDraft:
        prompt = _build_prompt(request_data, self._config.max_sentences)
        text = self._call_ollama(prompt)

        return LiveResponseDraft(
            text=text,
            generation_source=LiveResponseGenerationSource.RESPONSE_GENERATOR,
            token_count=len(text.split()),
            metadata={
                "provider": "ollama",
                "model": self._config.model,
                "live_generator": "ollama",
            },
        )

    def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model": self._config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.4,
                "num_predict": 220,
            },
        }
        body = json.dumps(payload).encode("utf-8")

        http_request = request.Request(
            url=f"{self._config.base_url.rstrip('/')}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(
                http_request,
                timeout=self._config.timeout_seconds,
            ) as response:
                raw = response.read().decode("utf-8")
        except URLError as exc:
            raise RuntimeError(
                "Ollama request failed. Confirm Ollama is running."
            ) from exc

        data = json.loads(raw)
        text = str(data.get("response", "")).strip()

        if not text:
            raise RuntimeError("Ollama returned an empty response.")

        return _limit_sentences(text, self._config.max_sentences)


def _build_prompt(
    request_data: LiveResponseGenerationRequest,
    max_sentences: int,
) -> str:
    context = request_data.context
    live_state = context.live_state

    memory = "\n".join(context.memory_context) or "No memory context supplied."
    working_memory = (
        "\n".join(context.working_memory_context)
        or "No working memory context supplied."
    )
    attention = (
        "\n".join(context.attention_context)
        or "No attention context supplied."
    )
    goals = "\n".join(context.goal_context) or "No goal context supplied."
    planning = (
        "\n".join(context.planning_context)
        or "No planning context supplied."
    )
    environment = (
        "\n".join(context.environment_context)
        or "No environment context supplied."
    )
    developer = (
        "\n".join(context.developer_context)
        or "No developer context supplied."
    )

    return f"""
You are {live_state.assistant_name}, Balu's local personal cognitive assistant.

You must behave like a calm, concise, respectful executive assistant.
You are not a chatbot.
You must not produce long paragraphs.
You must answer in at most {max_sentences} short sentences.
If the user asks for detail, give detail clearly.
You may be lightly witty, but never fake emotion.
You must be truthful. If something is not known, say so.
You must not claim an action was performed unless the runtime actually performed it.
You must not invent memory.
You must not bypass safety.
You must not mention internal implementation unless asked.

Current user: {live_state.user_label}
Assistant name: {live_state.assistant_name}
Intent: {request_data.intent.value}
Surface: {request_data.surface.value}
Situation: {context.situation_summary}

User said:
{context.user_text}

Memory context:
{memory}

Working memory:
{working_memory}

Attention context:
{attention}

Goals:
{goals}

Planning:
{planning}

Environment:
{environment}

Developer context:
{developer}

Now generate the next spoken response.
""".strip()


def _limit_sentences(text: str, max_sentences: int) -> str:
    normalized = " ".join(text.strip().split())
    if not normalized:
        return normalized

    pieces: list[str] = []
    current = ""

    for char in normalized:
        current += char
        if char in {".", "?", "!"}:
            pieces.append(current.strip())
            current = ""
            if len(pieces) >= max_sentences:
                break

    if not pieces:
        return normalized

    return " ".join(pieces).strip()