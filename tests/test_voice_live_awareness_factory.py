from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from jarvis.voice import (
    VoiceAwarenessPriority,
    VoiceAwarenessRequest,
    VoiceAwarenessSource,
    VoiceTranscript,
    VoiceTranscriptKind,
    make_voice_segment_id,
    make_voice_session_id,
    make_voice_transcript_id,
    utc_now,
)
from jarvis.voice.live_awareness_factory import (
    LiveToolAwarenessProvider,
    build_live_voice_awareness_runtime,
)


def _transcript(text: str = "jarvis open my project") -> VoiceTranscript:
    return VoiceTranscript(
        transcript_id=make_voice_transcript_id(),
        session_id=make_voice_session_id(),
        segment_id=make_voice_segment_id(),
        kind=VoiceTranscriptKind.FINAL,
        text=text,
        confidence=0.96,
        created_at=utc_now(),
    )


def _request(text: str = "jarvis open my project") -> VoiceAwarenessRequest:
    return VoiceAwarenessRequest(
        transcript=_transcript(text),
        session_id="voice_session_test",
        user_label="Balu",
        assistant_name="JARVIS",
    )


@dataclass(slots=True)
class FakeToolRegistry:
    def snapshot(self) -> object:
        return SimpleNamespace(
            tool_count=3,
            available_count=2,
            healthy_count=2,
        )


@dataclass(slots=True)
class FakeToolPlanner:
    def snapshot(self) -> object:
        return SimpleNamespace(
            name="tool_action_planner",
            planned_count=4,
        )


def test_live_tool_awareness_reports_safe_tool_pipeline() -> None:
    provider = LiveToolAwarenessProvider(
        registry=FakeToolRegistry(),
        planner=FakeToolPlanner(),
    )

    facts = provider.collect(_request())

    assert len(facts) == 3
    assert {fact.source for fact in facts} == {VoiceAwarenessSource.TOOLS}
    assert facts[0].key == "tool_control_contract"
    assert facts[0].priority == VoiceAwarenessPriority.HIGH
    assert "llm_direct_tool_execution_allowed=false" in facts[0].value
    assert "registered_tools=3" in facts[1].value
    assert "available_tools=2" in facts[1].value
    assert "planned_count=4" in facts[2].value


def test_live_voice_awareness_runtime_includes_tools_provider() -> None:
    runtime = build_live_voice_awareness_runtime()

    sources = tuple(provider.source for provider in runtime._providers)

    assert VoiceAwarenessSource.MEMORY in sources
    assert VoiceAwarenessSource.GOALS in sources
    assert VoiceAwarenessSource.TOOLS in sources
