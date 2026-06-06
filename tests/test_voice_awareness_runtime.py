from __future__ import annotations

from dataclasses import dataclass

from jarvis.voice import (
    VoiceAwarenessFact,
    VoiceAwarenessPriority,
    VoiceAwarenessRequest,
    VoiceAwarenessRuntime,
    VoiceAwarenessRuntimeConfig,
    VoiceAwarenessSource,
    VoiceAwarenessStatus,
    VoiceEnvironmentAwarenessProvider,
    VoiceGoalAwarenessProvider,
    VoiceHealthAwarenessProvider,
    VoiceMemoryAwarenessProvider,
    VoicePersonalityAwarenessProvider,
    VoiceResponseBoundaryAwarenessProvider,
    VoiceToolAwarenessProvider,
    VoiceTranscript,
    VoiceTranscriptKind,
    make_voice_segment_id,
    make_voice_session_id,
    make_voice_transcript_id,
    utc_now,
)


def _transcript(text: str) -> VoiceTranscript:
    return VoiceTranscript(
        transcript_id=make_voice_transcript_id(),
        session_id=make_voice_session_id(),
        segment_id=make_voice_segment_id(),
        kind=VoiceTranscriptKind.FINAL,
        text=text,
        confidence=0.96,
        created_at=utc_now(),
    )


def _request(text: str = "continue from here") -> VoiceAwarenessRequest:
    return VoiceAwarenessRequest(
        transcript=_transcript(text),
        session_id="voice_session_test",
        user_label="Balu",
        assistant_name="JARVIS",
    )


def _fact(
    *,
    source: VoiceAwarenessSource,
    key: str,
    value: str,
    confidence: float = 0.9,
    priority: VoiceAwarenessPriority = VoiceAwarenessPriority.NORMAL,
) -> VoiceAwarenessFact:
    return VoiceAwarenessFact(
        source=source,
        key=key,
        value=value,
        confidence=confidence,
        priority=priority,
        created_at=utc_now(),
    )


def _full_awareness_runtime() -> VoiceAwarenessRuntime:
    return VoiceAwarenessRuntime(
        providers=(
            VoiceMemoryAwarenessProvider(
                (
                    _fact(
                        source=VoiceAwarenessSource.MEMORY,
                        key="current_project",
                        value="JARVIS voice daily driver",
                    ),
                )
            ),
            VoiceEnvironmentAwarenessProvider(
                (
                    _fact(
                        source=VoiceAwarenessSource.ENVIRONMENT,
                        key="active_workspace",
                        value="E:\\JARVIS_OS",
                    ),
                )
            ),
            VoiceGoalAwarenessProvider(
                (
                    _fact(
                        source=VoiceAwarenessSource.GOALS,
                        key="active_goal",
                        value="wire awareness into voice cognition",
                    ),
                )
            ),
            VoicePersonalityAwarenessProvider(
                (
                    _fact(
                        source=VoiceAwarenessSource.PERSONALITY,
                        key="behavior",
                        value="calm concise protective",
                    ),
                )
            ),
            VoiceToolAwarenessProvider(
                (
                    _fact(
                        source=VoiceAwarenessSource.TOOLS,
                        key="test_state",
                        value="3744 passed before awareness wiring",
                    ),
                )
            ),
            VoiceHealthAwarenessProvider(
                (
                    _fact(
                        source=VoiceAwarenessSource.HEALTH,
                        key="voice_runtime",
                        value="healthy",
                    ),
                )
            ),
            VoiceResponseBoundaryAwarenessProvider(),
        )
    )


@dataclass
class FailingProvider:
    source: VoiceAwarenessSource = VoiceAwarenessSource.ENVIRONMENT

    def collect(
        self,
        request: VoiceAwarenessRequest,
    ) -> tuple[VoiceAwarenessFact, ...]:
        raise RuntimeError("environment unavailable")


def test_awareness_runtime_rejects_invalid_config() -> None:
    try:
        VoiceAwarenessRuntimeConfig(max_context_chars=100)
    except ValueError:
        pass
    else:
        raise AssertionError("expected invalid max_context_chars to fail")


def test_awareness_runtime_builds_full_awareness_packet() -> None:
    runtime = _full_awareness_runtime()

    packet = runtime.build(_request("what should I do now"))

    assert packet.status == VoiceAwarenessStatus.READY
    assert packet.ready is True
    assert packet.fact_count >= 8
    assert packet.highest_priority == VoiceAwarenessPriority.HIGH
    assert packet.missing_sources == ()
    assert "current_user_transcript=what should I do now" in packet.cognition_context
    assert "current_project=JARVIS voice daily driver" in packet.cognition_context
    assert "active_workspace=E:\\JARVIS_OS" in packet.cognition_context
    assert "active_goal=wire awareness into voice cognition" in packet.cognition_context
    assert "final_speech_origin=cognition_response_boundary" in packet.cognition_context


def test_awareness_runtime_fails_when_required_awareness_is_missing() -> None:
    runtime = VoiceAwarenessRuntime(
        providers=(VoiceResponseBoundaryAwarenessProvider(),),
    )

    packet = runtime.build(_request())

    assert packet.status == VoiceAwarenessStatus.FAILED
    assert VoiceAwarenessSource.MEMORY in packet.missing_sources
    assert VoiceAwarenessSource.ENVIRONMENT in packet.missing_sources
    assert VoiceAwarenessSource.GOALS in packet.missing_sources
    assert VoiceAwarenessSource.PERSONALITY in packet.missing_sources


def test_awareness_runtime_degrades_when_non_required_provider_fails() -> None:
    runtime = VoiceAwarenessRuntime(
        providers=(
            VoiceMemoryAwarenessProvider(
                (
                    _fact(
                        source=VoiceAwarenessSource.MEMORY,
                        key="project",
                        value="JARVIS",
                    ),
                )
            ),
            VoiceGoalAwarenessProvider(
                (
                    _fact(
                        source=VoiceAwarenessSource.GOALS,
                        key="goal",
                        value="validate",
                    ),
                )
            ),
            VoicePersonalityAwarenessProvider(
                (
                    _fact(
                        source=VoiceAwarenessSource.PERSONALITY,
                        key="style",
                        value="calm",
                    ),
                )
            ),
            VoiceResponseBoundaryAwarenessProvider(),
            FailingProvider(),
        ),
        config=VoiceAwarenessRuntimeConfig(require_environment=False),
    )

    packet = runtime.build(_request())

    assert packet.status == VoiceAwarenessStatus.DEGRADED
    assert packet.provider_errors


def test_awareness_runtime_changes_signature_when_environment_changes() -> None:
    first = _full_awareness_runtime().build(_request("continue"))

    second = VoiceAwarenessRuntime(
        providers=(
            VoiceMemoryAwarenessProvider(
                (
                    _fact(
                        source=VoiceAwarenessSource.MEMORY,
                        key="current_project",
                        value="JARVIS voice daily driver",
                    ),
                )
            ),
            VoiceEnvironmentAwarenessProvider(
                (
                    _fact(
                        source=VoiceAwarenessSource.ENVIRONMENT,
                        key="active_workspace",
                        value="browser documentation",
                    ),
                )
            ),
            VoiceGoalAwarenessProvider(
                (
                    _fact(
                        source=VoiceAwarenessSource.GOALS,
                        key="active_goal",
                        value="wire awareness into voice cognition",
                    ),
                )
            ),
            VoicePersonalityAwarenessProvider(
                (
                    _fact(
                        source=VoiceAwarenessSource.PERSONALITY,
                        key="behavior",
                        value="calm concise protective",
                    ),
                )
            ),
            VoiceResponseBoundaryAwarenessProvider(),
        )
    ).build(_request("continue"))

    assert first.signature != second.signature
    assert "active_workspace=browser documentation" in second.cognition_context


def test_awareness_runtime_prioritizes_critical_health_context() -> None:
    runtime = VoiceAwarenessRuntime(
        providers=(
            VoiceMemoryAwarenessProvider(
                (
                    _fact(
                        source=VoiceAwarenessSource.MEMORY,
                        key="project",
                        value="JARVIS",
                    ),
                )
            ),
            VoiceEnvironmentAwarenessProvider(
                (
                    _fact(
                        source=VoiceAwarenessSource.ENVIRONMENT,
                        key="battery",
                        value="critical",
                        priority=VoiceAwarenessPriority.CRITICAL,
                    ),
                )
            ),
            VoiceGoalAwarenessProvider(
                (
                    _fact(
                        source=VoiceAwarenessSource.GOALS,
                        key="goal",
                        value="continue safely",
                    ),
                )
            ),
            VoicePersonalityAwarenessProvider(
                (
                    _fact(
                        source=VoiceAwarenessSource.PERSONALITY,
                        key="style",
                        value="protective",
                    ),
                )
            ),
            VoiceResponseBoundaryAwarenessProvider(),
        )
    )

    packet = runtime.build(_request("continue"))

    assert packet.highest_priority == VoiceAwarenessPriority.CRITICAL
    assert "battery=critical" in packet.cognition_context


def test_awareness_runtime_filters_low_confidence_facts() -> None:
    runtime = VoiceAwarenessRuntime(
        providers=(
            VoiceMemoryAwarenessProvider(
                (
                    _fact(
                        source=VoiceAwarenessSource.MEMORY,
                        key="weak_memory",
                        value="possibly unrelated",
                        confidence=0.1,
                    ),
                )
            ),
            VoiceEnvironmentAwarenessProvider(
                (
                    _fact(
                        source=VoiceAwarenessSource.ENVIRONMENT,
                        key="window",
                        value="terminal",
                    ),
                )
            ),
            VoiceGoalAwarenessProvider(
                (
                    _fact(
                        source=VoiceAwarenessSource.GOALS,
                        key="goal",
                        value="continue",
                    ),
                )
            ),
            VoicePersonalityAwarenessProvider(
                (
                    _fact(
                        source=VoiceAwarenessSource.PERSONALITY,
                        key="style",
                        value="calm",
                    ),
                )
            ),
            VoiceResponseBoundaryAwarenessProvider(),
        ),
    )

    packet = runtime.build(_request())

    assert "weak_memory" not in packet.cognition_context


def test_awareness_runtime_does_not_generate_final_speech() -> None:
    packet = _full_awareness_runtime().build(_request("say something"))

    assert "JARVIS_AWARENESS_CONTEXT" in packet.cognition_context
    assert not hasattr(packet, "response_text")
    assert not hasattr(packet, "spoken_text")
    assert "Final spoken words must be generated by cognition." in (
        packet.cognition_context
    )


def test_awareness_enum_values_are_stable() -> None:
    assert VoiceAwarenessStatus.READY.value == "ready"
    assert VoiceAwarenessSource.ENVIRONMENT.value == "environment"
    assert VoiceAwarenessPriority.CRITICAL.value == "critical"