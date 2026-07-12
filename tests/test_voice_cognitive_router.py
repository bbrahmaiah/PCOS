from __future__ import annotations

from jarvis.voice import (
    VoiceCognitiveRouteAction,
    VoiceCognitiveRouter,
    VoiceCognitiveRouteRequest,
    VoiceCognitiveRouterPolicy,
    VoiceCognitiveState,
    VoiceTranscript,
    VoiceTranscriptKind,
    make_voice_segment_id,
    make_voice_session_id,
    make_voice_transcript_id,
    utc_now,
)
from jarvis.voice.transcript_attention_gate import TranscriptGateDecision


def _transcript(
    text: str,
    *,
    confidence: float = 0.95,
    stability: float | None = None,
) -> VoiceTranscript:
    metadata: dict[str, object] = {}
    if stability is not None:
        metadata["stability"] = stability
    return VoiceTranscript(
        transcript_id=make_voice_transcript_id(),
        session_id=make_voice_session_id(),
        segment_id=make_voice_segment_id(),
        kind=VoiceTranscriptKind.FINAL,
        text=text,
        confidence=confidence,
        created_at=utc_now(),
        metadata=metadata,
    )


def _gate(
    *,
    accepted: bool = True,
    reason: str = "accepted",
    text: str = "jarvis explain agi",
    wake_detected: bool = True,
    attention_active: bool = False,
    confidence: float = 0.95,
    word_count: int = 3,
) -> TranscriptGateDecision:
    return TranscriptGateDecision(
        accepted=accepted,
        reason=reason,
        normalized_text=text,
        wake_detected=wake_detected,
        attention_active=attention_active,
        confidence=confidence,
        word_count=word_count,
    )


def test_cognitive_router_ignores_rejected_attention_gate() -> None:
    router = VoiceCognitiveRouter()

    decision = router.route(
        VoiceCognitiveRouteRequest(
            transcript=_transcript("background noise"),
            gate_decision=_gate(
                accepted=False,
                reason="requires_wake_or_active_attention",
                text="background noise",
                wake_detected=False,
                word_count=2,
            ),
        )
    )

    assert decision.action == VoiceCognitiveRouteAction.IGNORE
    assert decision.state == VoiceCognitiveState.LISTENING
    assert decision.should_enter_cognition is False


def test_cognitive_router_waits_for_unstable_non_wake_followup() -> None:
    router = VoiceCognitiveRouter(
        policy=VoiceCognitiveRouterPolicy(min_stable_confidence=0.80)
    )

    decision = router.route(
        VoiceCognitiveRouteRequest(
            transcript=_transcript("explain", confidence=0.65, stability=0.60),
            gate_decision=_gate(
                text="explain",
                wake_detected=False,
                attention_active=True,
                confidence=0.65,
                word_count=1,
            ),
        )
    )

    assert decision.action == VoiceCognitiveRouteAction.WAIT_FOR_STABILITY
    assert decision.state == VoiceCognitiveState.STABILIZING_TRANSCRIPT
    assert decision.should_enter_cognition is False


def test_cognitive_router_routes_stable_wake_turn_to_cognition() -> None:
    router = VoiceCognitiveRouter()

    decision = router.route(
        VoiceCognitiveRouteRequest(
            transcript=_transcript("jarvis explain agi", stability=0.95),
            gate_decision=_gate(),
        )
    )

    assert decision.action == VoiceCognitiveRouteAction.RESPOND
    assert decision.state == VoiceCognitiveState.THINKING
    assert decision.should_enter_cognition is True


def test_cognitive_router_marks_tool_intent_for_planning() -> None:
    router = VoiceCognitiveRouter()

    decision = router.route(
        VoiceCognitiveRouteRequest(
            transcript=_transcript("jarvis open calculator", stability=0.96),
            gate_decision=_gate(text="jarvis open calculator"),
        )
    )

    assert decision.action == VoiceCognitiveRouteAction.RESPOND
    assert decision.state == VoiceCognitiveState.PLANNING
    assert decision.tool_planning_recommended is True
