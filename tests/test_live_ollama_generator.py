from __future__ import annotations

import pytest

from jarvis.live import (
    LiveResponseGenerationRequest,
    LiveResponseIntent,
    LiveResponseKind,
    LiveResponseSafety,
    LiveResponseSurface,
    LiveSessionConfig,
    LiveSessionStateRuntime,
    OllamaGeneratorConfig,
    make_live_turn_id,
)
from jarvis.live.ollama_generator import _build_prompt


def test_ollama_generator_config_validation() -> None:
    with pytest.raises(ValueError):
        OllamaGeneratorConfig(base_url=" ")

    with pytest.raises(ValueError):
        OllamaGeneratorConfig(model=" ")

    with pytest.raises(ValueError):
        OllamaGeneratorConfig(timeout_seconds=0)


def test_ollama_prompt_contains_live_context() -> None:
    state = LiveSessionStateRuntime(config=LiveSessionConfig()).state
    request = LiveResponseGenerationRequest(
        turn_id=make_live_turn_id(),
        intent=LiveResponseIntent.ANSWER,
        surface=LiveResponseSurface.VOICE,
        context=__import__(
            "jarvis.live.response_boundary",
            fromlist=["LiveResponseContext"],
        ).LiveResponseContext(
            live_state=state,
            cognitive_state=None,
            user_text="Jarvis explain control systems.",
            situation_summary="Testing live Ollama generator.",
            memory_context=("Step 50 is sealed.",),
            working_memory_context=("Current task is Step 51.",),
            attention_context=("User is focused on launcher.",),
            goal_context=("Build real live JARVIS launcher.",),
            planning_context=("Connect runner to Ollama.",),
            environment_context=("PowerShell active.",),
            developer_context=("Running tests.",),
        ),
        response_kind=LiveResponseKind.CONVERSATIONAL,
        safety=LiveResponseSafety.SAFE_TO_SPEAK,
        max_sentences=3,
    )

    prompt = _build_prompt(request, max_sentences=3)

    assert "Jarvis explain control systems." in prompt
    assert "Step 50 is sealed." in prompt
    assert "Build real live JARVIS launcher." in prompt