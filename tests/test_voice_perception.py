from __future__ import annotations

from jarvis.voice import (
    VoicePerceptionIntentState,
    VoicePerceptionRuntime,
    VoiceTranscript,
    VoiceTranscriptKind,
    enrich_transcript_with_perception,
    make_voice_segment_id,
    make_voice_session_id,
    make_voice_transcript_id,
    utc_now,
)


def _transcript(
    text: str,
    *,
    kind: VoiceTranscriptKind = VoiceTranscriptKind.FINAL,
    confidence: float = 0.95,
    metadata: dict[str, object] | None = None,
) -> VoiceTranscript:
    return VoiceTranscript(
        transcript_id=make_voice_transcript_id(),
        session_id=make_voice_session_id(),
        segment_id=make_voice_segment_id(),
        kind=kind,
        text=text,
        confidence=confidence,
        created_at=utc_now(),
        metadata=metadata or {},
    )


def test_voice_perception_marks_wake_partial_as_candidate() -> None:
    runtime = VoicePerceptionRuntime()

    packet = runtime.observe_partial(
        _transcript("Jarvis", kind=VoiceTranscriptKind.PARTIAL)
    )

    assert packet.intent_state == VoicePerceptionIntentState.WAKE_CANDIDATE
    assert packet.wake_detected is True
    assert packet.final is False


def test_voice_perception_repeated_partials_move_toward_stabilizing() -> None:
    runtime = VoicePerceptionRuntime()

    first = runtime.observe_partial(
        _transcript("open calculator", kind=VoiceTranscriptKind.PARTIAL)
    )
    second = runtime.observe_partial(
        _transcript("open calculator", kind=VoiceTranscriptKind.PARTIAL)
    )

    assert first.intent_state == VoicePerceptionIntentState.CAPTURING
    assert second.intent_state == VoicePerceptionIntentState.STABILIZING
    assert second.stability > first.stability


def test_voice_perception_final_stable_text_is_ready_for_routing() -> None:
    runtime = VoicePerceptionRuntime()

    packet = runtime.observe_final(_transcript("jarvis open calculator"))

    assert packet.intent_state == VoicePerceptionIntentState.READY_FOR_ROUTING
    assert packet.ready_for_routing is True
    assert packet.wake_detected is True


def test_voice_perception_known_stt_noise_is_not_routed() -> None:
    runtime = VoicePerceptionRuntime()

    packet = runtime.observe_final(_transcript("happy holidays"))

    assert packet.intent_state == VoicePerceptionIntentState.NOISE
    assert packet.reason == "known_background_or_stt_noise"


def test_voice_perception_marks_user_input_during_speech_as_interruption() -> None:
    runtime = VoicePerceptionRuntime()

    packet = runtime.observe_final(
        _transcript("wait stop there"),
        assistant_speaking=True,
    )

    assert packet.intent_state == VoicePerceptionIntentState.INTERRUPTION
    assert packet.reason == "assistant_speaking_user_input_detected"


def test_voice_perception_keeps_known_noise_as_noise_during_speech() -> None:
    runtime = VoicePerceptionRuntime()

    packet = runtime.observe_final(
        _transcript("happy holidays"),
        assistant_speaking=True,
    )

    assert packet.intent_state == VoicePerceptionIntentState.NOISE
    assert packet.reason == "known_background_or_stt_noise"


def test_enrich_transcript_with_perception_preserves_explicit_stability() -> None:
    runtime = VoicePerceptionRuntime()
    transcript = _transcript(
        "explain",
        confidence=0.66,
        metadata={"stability": 0.40},
    )

    packet = runtime.observe_final(transcript)
    enriched = enrich_transcript_with_perception(transcript, packet)

    assert enriched.metadata["stability"] == 0.40
    assert enriched.metadata["perception_stability"] == 0.40
    assert enriched.metadata["perception_intent_state"] == "stabilizing"
