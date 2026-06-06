from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from jarvis.live import LiveResponse
from jarvis.voice import (
    VoiceAwarenessFact,
    VoiceAwarenessPriority,
    VoiceAwarenessRecord,
    VoiceAwarenessRequest,
    VoiceAwarenessRuntime,
    VoiceAwarenessSource,
    VoiceAwarenessStatus,
    VoiceCognitionRequest,
    VoiceCognitionResult,
    VoiceRealAwarenessIntegration,
    VoiceRealAwarenessIntegrationConfig,
    VoiceRealAwarenessProviderKind,
    VoiceRealAwarenessResponseBoundary,
    VoiceRealAwarenessStatus,
    VoiceTranscript,
    VoiceTranscriptKind,
    make_real_awareness_provider,
    make_real_awareness_runtime,
    make_voice_segment_id,
    make_voice_session_id,
    make_voice_transcript_id,
    utc_now,
)


def _transcript(text: str = "continue from here") -> VoiceTranscript:
    return VoiceTranscript(
        transcript_id=make_voice_transcript_id(),
        session_id=make_voice_session_id(),
        segment_id=make_voice_segment_id(),
        kind=VoiceTranscriptKind.FINAL,
        text=text,
        confidence=0.96,
        created_at=utc_now(),
    )


def _request(text: str = "continue from here") -> VoiceCognitionRequest:
    return VoiceCognitionRequest(
        transcript=_transcript(text),
        user_label="Balu",
        assistant_name="JARVIS",
        metadata={"turn": "voice"},
    )


def _record(
    key: str,
    value: str,
    *,
    priority: VoiceAwarenessPriority = VoiceAwarenessPriority.NORMAL,
) -> VoiceAwarenessRecord:
    return VoiceAwarenessRecord(
        key=key,
        value=value,
        confidence=0.9,
        priority=priority,
    )


def _runtime_full() -> VoiceAwarenessRuntime:
    providers = (
        make_real_awareness_provider(
            source=VoiceAwarenessSource.MEMORY,
            provider_kind=VoiceRealAwarenessProviderKind.MEMORY,
            records=(_record("project", "JARVIS voice runtime"),),
        ),
        make_real_awareness_provider(
            source=VoiceAwarenessSource.ENVIRONMENT,
            provider_kind=VoiceRealAwarenessProviderKind.ENVIRONMENT,
            records=(_record("active_workspace", "E:\\JARVIS_OS"),),
        ),
        make_real_awareness_provider(
            source=VoiceAwarenessSource.GOALS,
            provider_kind=VoiceRealAwarenessProviderKind.GOALS,
            records=(_record("active_goal", "real voice daily driver"),),
        ),
        make_real_awareness_provider(
            source=VoiceAwarenessSource.PERSONALITY,
            provider_kind=VoiceRealAwarenessProviderKind.PERSONALITY,
            records=(_record("behavior", "calm concise protective"),),
        ),
        make_real_awareness_provider(
            source=VoiceAwarenessSource.TOOLS,
            provider_kind=VoiceRealAwarenessProviderKind.TOOLS,
            records=(_record("tool_state", "ready"),),
        ),
        make_real_awareness_provider(
            source=VoiceAwarenessSource.DEVELOPER,
            provider_kind=VoiceRealAwarenessProviderKind.DEVELOPER,
            records=(_record("test_state", "3761 passed before integration"),),
        ),
        make_real_awareness_provider(
            source=VoiceAwarenessSource.HEALTH,
            provider_kind=VoiceRealAwarenessProviderKind.HEALTH,
            records=(
                _record(
                    "voice_health",
                    "healthy",
                    priority=VoiceAwarenessPriority.HIGH,
                ),
            ),
        ),
        make_real_awareness_provider(
            source=VoiceAwarenessSource.RESPONSE_BOUNDARY,
            provider_kind=VoiceRealAwarenessProviderKind.SAFETY,
            records=(
                _record(
                    "final_speech_origin",
                    "cognition_response_boundary",
                    priority=VoiceAwarenessPriority.HIGH,
                ),
            ),
        ),
    )
    return make_real_awareness_runtime(providers=providers)


def _result(text: str = "generated from cognition") -> VoiceCognitionResult:
    response = cast(
        LiveResponse,
        SimpleNamespace(
            response_id="response_id",
            text=text,
            created_at=utc_now(),
            metadata={},
        ),
    )
    return cast(VoiceCognitionResult, SimpleNamespace(response=response))


def test_real_awareness_integration_prepares_cognition_request() -> None:
    integration = VoiceRealAwarenessIntegration(
        awareness_runtime=_runtime_full()
    )

    result = integration.prepare_request(_request("continue"))

    assert result.status == VoiceRealAwarenessStatus.READY
    assert result.awareness_packet.status == VoiceAwarenessStatus.READY
    assert result.cognition_request.metadata["awareness_enabled"] is True
    assert "awareness_context" in result.cognition_request.metadata
    assert "project=JARVIS voice runtime" in str(
        result.cognition_request.metadata["awareness_context"]
    )
    assert (
        result.cognition_request.metadata["response_origin_required"]
        == "cognition_response_boundary"
    )
    assert (
        result.cognition_request.metadata["scripted_conversation_allowed"]
        is False
    )


def test_real_awareness_integration_validates_response_boundary() -> None:
    integration = VoiceRealAwarenessIntegration(
        awareness_runtime=_runtime_full()
    )
    prepared = integration.prepare_request(_request())

    validated = integration.validate_response(
        request=prepared.cognition_request,
        result=_result(),
        awareness_packet=prepared.awareness_packet,
        injection_result=prepared.injection_result,
    )

    assert validated.status == VoiceRealAwarenessStatus.READY
    assert validated.boundary_result is not None
    assert validated.boundary_result.passed is True


def test_real_awareness_boundary_fails_without_awareness_metadata() -> None:
    boundary = VoiceRealAwarenessResponseBoundary()

    result = boundary.validate(request=_request(), result=_result())

    assert result.passed is False
    assert result.status == VoiceRealAwarenessStatus.FAILED
    assert "awareness metadata missing" in result.violations


def test_real_awareness_integration_fails_when_awareness_missing() -> None:
    integration = VoiceRealAwarenessIntegration(
        awareness_runtime=make_real_awareness_runtime(providers=())
    )

    result = integration.prepare_request(_request())

    assert result.status == VoiceRealAwarenessStatus.FAILED
    assert result.awareness_packet.missing_sources


def test_real_awareness_integration_blocks_degraded_when_policy_disallows() -> None:
    class BrokenProvider:
        source = VoiceAwarenessSource.ENVIRONMENT

        def collect(
            self,
            request: VoiceAwarenessRequest,
        ) -> tuple[VoiceAwarenessFact, ...]:
            raise RuntimeError("environment unavailable")

    providers = (
        make_real_awareness_provider(
            source=VoiceAwarenessSource.MEMORY,
            provider_kind=VoiceRealAwarenessProviderKind.MEMORY,
            records=(_record("project", "JARVIS"),),
        ),
        make_real_awareness_provider(
            source=VoiceAwarenessSource.GOALS,
            provider_kind=VoiceRealAwarenessProviderKind.GOALS,
            records=(_record("goal", "validate"),),
        ),
        make_real_awareness_provider(
            source=VoiceAwarenessSource.PERSONALITY,
            provider_kind=VoiceRealAwarenessProviderKind.PERSONALITY,
            records=(_record("style", "calm"),),
        ),
        make_real_awareness_provider(
            source=VoiceAwarenessSource.RESPONSE_BOUNDARY,
            provider_kind=VoiceRealAwarenessProviderKind.SAFETY,
            records=(_record("origin", "cognition_response_boundary"),),
        ),
        BrokenProvider(),
    )

    runtime = make_real_awareness_runtime(providers=providers)
    integration = VoiceRealAwarenessIntegration(
        awareness_runtime=runtime,
        config=VoiceRealAwarenessIntegrationConfig(
            allow_degraded_awareness=False
        ),
    )

    result = integration.prepare_request(_request())

    assert result.status == VoiceRealAwarenessStatus.FAILED


def test_real_awareness_integration_does_not_generate_speech() -> None:
    integration = VoiceRealAwarenessIntegration(
        awareness_runtime=_runtime_full()
    )

    result = integration.prepare_request(_request())

    assert not hasattr(result.awareness_packet, "response_text")
    assert not hasattr(result.awareness_packet, "spoken_text")


def test_real_awareness_enum_values_are_stable() -> None:
    assert VoiceRealAwarenessStatus.READY.value == "ready"
    assert VoiceRealAwarenessProviderKind.MEMORY.value == "memory"