from __future__ import annotations

from jarvis.voice.contracts import (
    VoiceTranscript,
    VoiceTranscriptKind,
    make_voice_segment_id,
    make_voice_session_id,
    make_voice_transcript_id,
    utc_now,
)
from jarvis.voice.transcript_attention_gate import (
    TranscriptAttentionGate,
    TranscriptAttentionGatePolicy,
)


def _transcript(text: str) -> VoiceTranscript:
    return VoiceTranscript(
        transcript_id=make_voice_transcript_id(),
        session_id=make_voice_session_id(),
        segment_id=make_voice_segment_id(),
        kind=VoiceTranscriptKind.FINAL,
        text=text,
        confidence=0.95,
        created_at=utc_now(),
    )


def test_attention_gate_accepts_short_followup_after_companion_turn() -> None:
    gate = TranscriptAttentionGate(
        TranscriptAttentionGatePolicy(
            min_words_without_wake=2,
            min_words_when_attention_active=1,
            require_wake_or_attention=False,
            require_attention_for_promoted_partials=False,
        )
    )

    first = gate.evaluate(_transcript("can you hear me"), now=1.0)
    followup = gate.evaluate(_transcript("explain"), now=2.0)

    assert first.accepted is True
    assert followup.accepted is True
    assert followup.attention_active is True
    assert followup.reason == "accepted"


def test_attention_gate_blocks_common_silence_hallucination() -> None:
    gate = TranscriptAttentionGate(
        TranscriptAttentionGatePolicy(
            min_words_without_wake=2,
            require_wake_or_attention=False,
            require_attention_for_promoted_partials=False,
        )
    )

    decision = gate.evaluate(_transcript("I'll be right back. Bye."), now=1.0)

    assert decision.accepted is False
    assert decision.reason == "known_silence_hallucination"


def test_attention_gate_blocks_live_log_hallucinations() -> None:
    gate = TranscriptAttentionGate(
        TranscriptAttentionGatePolicy(
            min_words_without_wake=2,
            require_wake_or_attention=False,
            require_attention_for_promoted_partials=False,
        )
    )

    for text in (
        "I miss up.",
        "Buh-bye.",
        "Thank you.",
        "Bless you.",
        "service.",
        "God.",
    ):
        decision = gate.evaluate(_transcript(text), now=1.0)

        assert decision.accepted is False
        assert decision.reason == "known_silence_hallucination"


def test_attention_gate_default_requires_wake_before_attention() -> None:
    gate = TranscriptAttentionGate()

    ignored = gate.evaluate(_transcript("explain artificial intelligence"), now=1.0)
    activated = gate.evaluate(_transcript("Jarvis explain AGI"), now=2.0)
    followup = gate.evaluate(_transcript("AI"), now=3.0)

    assert ignored.accepted is False
    assert ignored.reason == "requires_wake_or_active_attention"
    assert activated.accepted is True
    assert followup.accepted is True
    assert followup.attention_active is True
