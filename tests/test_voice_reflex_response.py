from __future__ import annotations

from jarvis.voice import (
    VoiceReflexResponseKind,
    VoiceReflexResponsePolicy,
    VoiceReflexResponseRuntime,
    VoiceTranscript,
    VoiceTranscriptKind,
    make_voice_segment_id,
    make_voice_session_id,
    make_voice_transcript_id,
    utc_now,
)


def _transcript(
    text: str,
    *,
    confidence: float = 0.95,
    kind: VoiceTranscriptKind = VoiceTranscriptKind.FINAL,
) -> VoiceTranscript:
    return VoiceTranscript(
        transcript_id=make_voice_transcript_id(),
        session_id=make_voice_session_id(),
        segment_id=make_voice_segment_id(),
        kind=kind,
        text=text,
        confidence=confidence,
        created_at=utc_now(),
    )


def test_reflex_routes_availability_ping_to_cognition() -> None:
    runtime = VoiceReflexResponseRuntime()

    decision = runtime.evaluate(
        _transcript("Jarvis can you hear me"),
        assistant_speaking=False,
    )

    assert decision.accepted is False
    assert decision.kind == VoiceReflexResponseKind.NONE
    assert decision.response_text is None
    assert decision.should_speak is False
    assert decision.should_continue_to_cognition is True
    assert decision.reason == "availability_requires_cognition"
    assert decision.metadata["fixed_spoken_response_blocked"] is True


def test_reflex_routes_explicit_wake_up_word_to_cognition() -> None:
    runtime = VoiceReflexResponseRuntime()

    decision = runtime.evaluate(
        _transcript("wake up Jarvis"),
        assistant_speaking=False,
    )

    assert decision.accepted is False
    assert decision.kind == VoiceReflexResponseKind.NONE
    assert decision.response_text is None
    assert decision.should_continue_to_cognition is True
    assert decision.reason == "availability_requires_cognition"


def test_reflex_stops_playback_without_generating_answer_text() -> None:
    runtime = VoiceReflexResponseRuntime()

    decision = runtime.evaluate(
        _transcript("stop"),
        assistant_speaking=True,
    )

    assert decision.accepted is True
    assert decision.kind == VoiceReflexResponseKind.STOP_PLAYBACK
    assert decision.response_text is None
    assert decision.should_stop_playback is True
    assert decision.should_speak is False


def test_reflex_shutdown_word_stops_session_without_cognition() -> None:
    runtime = VoiceReflexResponseRuntime()

    decision = runtime.evaluate(
        _transcript("Jarvis shut down"),
        assistant_speaking=False,
    )

    assert decision.accepted is True
    assert decision.kind == VoiceReflexResponseKind.SHUTDOWN_SESSION
    assert decision.response_text is None
    assert decision.should_stop_playback is True
    assert decision.should_shutdown_session is True
    assert decision.should_continue_to_cognition is False
    assert decision.reason == "shutdown_session_reflex"


def test_reflex_leaves_real_questions_for_cognition() -> None:
    runtime = VoiceReflexResponseRuntime()

    decision = runtime.evaluate(
        _transcript("explain AGI in two lines"),
        assistant_speaking=False,
    )

    assert decision.accepted is False
    assert decision.should_continue_to_cognition is True
    assert decision.reason == "not_operational_reflex"


def test_reflex_rejects_low_confidence_audio() -> None:
    runtime = VoiceReflexResponseRuntime(
        policy=VoiceReflexResponsePolicy(min_confidence=0.80)
    )

    decision = runtime.evaluate(
        _transcript("jarvis", confidence=0.40),
        assistant_speaking=False,
    )

    assert decision.accepted is False
    assert decision.reason == "low_confidence"
