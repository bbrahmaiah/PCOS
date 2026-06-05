from __future__ import annotations

from jarvis.live import (
    LiveResponseDraft,
    LiveResponseGenerationRequest,
    LiveResponseGenerationSource,
    LiveResponseGenerator,
)
from jarvis.voice import (
    StaticVoiceCognitionContextProvider,
    VoiceCognitionContextItem,
    VoiceCognitionContextKind,
    VoiceCognitionPolicy,
    VoiceCognitionRequest,
    VoiceCognitionResponseRuntime,
    VoiceCognitionStatus,
    VoiceCognitionTranscriptSafety,
    VoiceTranscript,
    VoiceTranscriptKind,
    make_voice_segment_id,
    make_voice_session_id,
    make_voice_transcript_id,
    utc_now,
)


class DerivedFakeResponseGenerator(LiveResponseGenerator):
    def generate(
        self,
        request: LiveResponseGenerationRequest,
    ) -> LiveResponseDraft:
        user_text = request.context.user_text.strip()
        memory = " ".join(request.context.memory_context)
        goal = " ".join(request.context.goal_context)
        text = f"Understood. {user_text} Memory: {memory} Goal: {goal}"
        return LiveResponseDraft(
            text=text,
            generation_source=LiveResponseGenerationSource.RESPONSE_GENERATOR,
            token_count=len(text.split()),
            metadata={"test_generator": True},
        )


def _transcript(
    text: str = "Jarvis explain PID control.",
    *,
    kind: VoiceTranscriptKind = VoiceTranscriptKind.FINAL,
    confidence: float = 0.95,
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


def test_voice_cognition_rejects_partial_by_default() -> None:
    runtime = VoiceCognitionResponseRuntime(
        response_generator=DerivedFakeResponseGenerator()
    )

    result = runtime.think_from_transcript(
        VoiceCognitionRequest(
            transcript=_transcript(
                kind=VoiceTranscriptKind.PARTIAL,
                confidence=0.99,
            )
        )
    )

    assert result.status == VoiceCognitionStatus.IGNORED
    assert result.response is None
    assert result.safety == VoiceCognitionTranscriptSafety.PREDICTION_ONLY


def test_voice_cognition_prefetch_accepts_partial_prediction() -> None:
    runtime = VoiceCognitionResponseRuntime(
        response_generator=DerivedFakeResponseGenerator(),
        context_provider=StaticVoiceCognitionContextProvider(
            (
                VoiceCognitionContextItem(
                    kind=VoiceCognitionContextKind.MEMORY,
                    text="Balu is building real voice JARVIS.",
                ),
            )
        ),
    )

    result = runtime.prefetch_from_partial(
        VoiceCognitionRequest(
            transcript=_transcript(
                "Jarvis explain",
                kind=VoiceTranscriptKind.PARTIAL,
                confidence=0.90,
            )
        )
    )

    assert result.accepted is True
    assert result.safety == VoiceCognitionTranscriptSafety.PREDICTION_ONLY
    assert result.context_pack.memory == (
        "Balu is building real voice JARVIS.",
    )


def test_voice_cognition_rejects_low_confidence_final() -> None:
    runtime = VoiceCognitionResponseRuntime(
        response_generator=DerivedFakeResponseGenerator(),
        policy=VoiceCognitionPolicy(min_dialogue_confidence=0.60),
    )

    result = runtime.think_from_transcript(
        VoiceCognitionRequest(transcript=_transcript(confidence=0.40))
    )

    assert result.status == VoiceCognitionStatus.IGNORED
    assert result.response is None
    assert result.safety == VoiceCognitionTranscriptSafety.NEEDS_CLARIFICATION


def test_voice_cognition_blocks_low_confidence_action_candidate() -> None:
    runtime = VoiceCognitionResponseRuntime(
        response_generator=DerivedFakeResponseGenerator(),
        policy=VoiceCognitionPolicy(min_action_confidence=0.80),
    )

    result = runtime.think_from_transcript(
        VoiceCognitionRequest(
            transcript=_transcript(confidence=0.70),
            allow_action_candidate=True,
        )
    )

    assert result.status == VoiceCognitionStatus.IGNORED
    assert result.response is None
    assert result.safety == VoiceCognitionTranscriptSafety.BLOCKED_FOR_ACTION


def test_voice_cognition_routes_final_transcript_to_live_response() -> None:
    runtime = VoiceCognitionResponseRuntime(
        response_generator=DerivedFakeResponseGenerator(),
        context_provider=StaticVoiceCognitionContextProvider(
            (
                VoiceCognitionContextItem(
                    kind=VoiceCognitionContextKind.MEMORY,
                    text="Balu is building real voice JARVIS.",
                ),
                VoiceCognitionContextItem(
                    kind=VoiceCognitionContextKind.GOAL,
                    text="Complete Step 51E.",
                ),
            )
        ),
    )

    result = runtime.think_from_transcript(
        VoiceCognitionRequest(
            transcript=_transcript("Jarvis explain control systems."),
        )
    )

    assert result.status == VoiceCognitionStatus.THINKING
    assert result.response is not None
    assert "Jarvis explain control systems." in result.response.text
    assert "Balu is building real voice JARVIS." in result.response.text
    assert result.safety == VoiceCognitionTranscriptSafety.SAFE_FOR_DIALOGUE


def test_voice_cognition_context_is_compacted() -> None:
    long_text = "memory " * 200
    runtime = VoiceCognitionResponseRuntime(
        response_generator=DerivedFakeResponseGenerator(),
        context_provider=StaticVoiceCognitionContextProvider(
            (
                VoiceCognitionContextItem(
                    kind=VoiceCognitionContextKind.MEMORY,
                    text=long_text,
                ),
            )
        ),
        policy=VoiceCognitionPolicy(max_context_item_chars=40),
    )

    result = runtime.prefetch_from_partial(
        VoiceCognitionRequest(
            transcript=_transcript(
                kind=VoiceTranscriptKind.PARTIAL,
                confidence=0.90,
            )
        )
    )

    assert result.context_pack.memory
    assert len(result.context_pack.memory[0]) <= 40


def test_voice_cognition_snapshot_tracks_response_and_latency() -> None:
    runtime = VoiceCognitionResponseRuntime(
        response_generator=DerivedFakeResponseGenerator()
    )

    runtime.think_from_transcript(
        VoiceCognitionRequest(transcript=_transcript())
    )
    snapshot = runtime.snapshot()

    assert snapshot.prepared is True
    assert snapshot.responses == 1
    assert snapshot.last_text == "Jarvis explain PID control."
    assert snapshot.last_response_text is not None
    assert snapshot.last_latency_ms is not None


def test_voice_cognition_enum_values_are_stable() -> None:
    assert VoiceCognitionStatus.THINKING.value == "thinking"
    assert VoiceCognitionTranscriptSafety.SAFE_FOR_DIALOGUE.value == (
        "safe_for_dialogue"
    )
    assert VoiceCognitionContextKind.PERSONALITY.value == "personality"