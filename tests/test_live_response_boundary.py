from __future__ import annotations

import pytest

from jarvis.live import (
    LiveDeterministicSystemMessage,
    LiveResponseBoundaryPolicy,
    LiveResponseBoundaryRuntime,
    LiveResponseBoundaryStatus,
    LiveResponseBoundaryViolation,
    LiveResponseContext,
    LiveResponseDraft,
    LiveResponseGenerationRequest,
    LiveResponseGenerationSource,
    LiveResponseIntent,
    LiveResponseKind,
    LiveResponseSafety,
    LiveResponseSurface,
    LiveSessionConfig,
    LiveSessionMode,
    default_live_session_state,
    make_live_response,
    make_live_turn_id,
)


class ContextAwareFakeGenerator:
    def generate(
        self,
        request: LiveResponseGenerationRequest,
    ) -> LiveResponseDraft:
        context = request.context
        parts = [
            context.live_state.user_label,
            request.intent.value,
            context.situation_summary,
            " ".join(context.memory_context),
            " ".join(context.goal_context),
        ]
        text = " | ".join(part for part in parts if part.strip())
        return LiveResponseDraft(
            text=text,
            generation_source=LiveResponseGenerationSource.RESPONSE_GENERATOR,
            token_count=len(text.split()),
            metadata={"fake": True},
        )


class InvalidSourceGenerator:
    def generate(
        self,
        request: LiveResponseGenerationRequest,
    ) -> LiveResponseDraft:
        return LiveResponseDraft(
            text="Invalid generated text.",
            generation_source=LiveResponseGenerationSource.DIAGNOSTIC_SYSTEM,
        )


def _request() -> LiveResponseGenerationRequest:
    state = default_live_session_state(
        config=LiveSessionConfig(
            mode=LiveSessionMode.SAFE_SIMULATION,
            user_label="Balu",
        )
    )
    return LiveResponseGenerationRequest(
        turn_id=make_live_turn_id(),
        intent=LiveResponseIntent.GREETING,
        surface=LiveResponseSurface.VOICE,
        context=LiveResponseContext(
            live_state=state,
            situation_summary="Step 50 live boundary is active.",
            memory_context=("Previous focus was daily-driver runtime.",),
            goal_context=("Current goal is 50A.5.",),
        ),
    )


def test_live_response_boundary_blocks_conversation_without_generator() -> None:
    runtime = LiveResponseBoundaryRuntime()

    result = runtime.generate(_request())

    assert result.status == LiveResponseBoundaryStatus.BLOCKED
    assert result.succeeded is False
    assert result.response is None
    assert result.violation == LiveResponseBoundaryViolation.MISSING_GENERATOR


def test_live_response_boundary_generates_contextual_conversation() -> None:
    runtime = LiveResponseBoundaryRuntime(generator=ContextAwareFakeGenerator())

    result = runtime.generate(_request())

    assert result.status == LiveResponseBoundaryStatus.READY
    assert result.response is not None
    assert result.response.kind == LiveResponseKind.CONVERSATIONAL
    assert result.response.generated_by_cognition is True
    assert "Balu" in result.response.text
    assert "Previous focus was daily-driver runtime." in result.response.text
    assert "Current goal is 50A.5." in result.response.text


def test_live_response_boundary_blocks_invalid_conversational_source() -> None:
    runtime = LiveResponseBoundaryRuntime(generator=InvalidSourceGenerator())

    result = runtime.generate(_request())

    assert result.status == LiveResponseBoundaryStatus.BLOCKED
    assert result.violation == (
        LiveResponseBoundaryViolation.INVALID_GENERATION_SOURCE
    )


def test_live_response_boundary_blocks_safety_blocked_generation() -> None:
    request = _request().__class__(
        turn_id=make_live_turn_id(),
        intent=LiveResponseIntent.WARNING,
        surface=LiveResponseSurface.VOICE,
        context=_request().context,
        safety=LiveResponseSafety.BLOCKED,
    )
    runtime = LiveResponseBoundaryRuntime(generator=ContextAwareFakeGenerator())

    result = runtime.generate(request)

    assert result.status == LiveResponseBoundaryStatus.BLOCKED
    assert result.violation == LiveResponseBoundaryViolation.BLOCKED_BY_SAFETY


def test_live_response_boundary_does_not_generate_diagnostic_text() -> None:
    request = _request().__class__(
        turn_id=make_live_turn_id(),
        intent=LiveResponseIntent.STATUS,
        surface=LiveResponseSurface.VOICE,
        context=_request().context,
        response_kind=LiveResponseKind.DIAGNOSTIC,
    )
    runtime = LiveResponseBoundaryRuntime(generator=ContextAwareFakeGenerator())

    result = runtime.generate(request)

    assert result.status == LiveResponseBoundaryStatus.BLOCKED
    assert result.violation == (
        LiveResponseBoundaryViolation.DIAGNOSTIC_KIND_REQUIRED
    )


def test_deterministic_system_message_allows_diagnostic_only() -> None:
    runtime = LiveResponseBoundaryRuntime()
    result = runtime.system_message(
        LiveDeterministicSystemMessage(
            turn_id=make_live_turn_id(),
            kind=LiveResponseKind.DIAGNOSTIC,
            text="Microphone unavailable.",
            source=LiveResponseGenerationSource.DIAGNOSTIC_SYSTEM,
        )
    )

    assert result.status == LiveResponseBoundaryStatus.READY
    assert result.response is not None
    assert result.response.is_conversational is False
    assert result.response.deterministic_system_response is True


def test_deterministic_system_message_rejects_conversation() -> None:
    with pytest.raises(ValueError):
        LiveDeterministicSystemMessage(
            turn_id=make_live_turn_id(),
            kind=LiveResponseKind.CONVERSATIONAL,
            text="This must not become live scripted conversation.",
            source=LiveResponseGenerationSource.DIAGNOSTIC_SYSTEM,
        )


def test_validate_for_tts_accepts_generated_conversation() -> None:
    runtime = LiveResponseBoundaryRuntime(generator=ContextAwareFakeGenerator())
    generated = runtime.generate(_request())

    assert generated.response is not None

    result = runtime.validate_for_tts(generated.response)

    assert result.status == LiveResponseBoundaryStatus.READY
    assert result.response == generated.response


def test_validate_for_tts_blocks_diagnostic_conversation() -> None:
    runtime = LiveResponseBoundaryRuntime()
    response = make_live_response(
        turn_id=make_live_turn_id(),
        kind=LiveResponseKind.CONVERSATIONAL,
        text="This should not be spoken as live conversation.",
        generation_source=LiveResponseGenerationSource.DIAGNOSTIC_SYSTEM,
        safety=LiveResponseSafety.SAFE_TO_SPEAK,
    )

    result = runtime.validate_for_tts(response)

    assert result.status == LiveResponseBoundaryStatus.BLOCKED
    assert result.violation == (
        LiveResponseBoundaryViolation.INVALID_GENERATION_SOURCE
    )


def test_response_boundary_policy_can_disable_deterministic_system_text() -> None:
    runtime = LiveResponseBoundaryRuntime(
        policy=LiveResponseBoundaryPolicy(
            allow_deterministic_diagnostics=False,
        )
    )
    result = runtime.system_message(
        LiveDeterministicSystemMessage(
            turn_id=make_live_turn_id(),
            kind=LiveResponseKind.DIAGNOSTIC,
            text="Microphone unavailable.",
            source=LiveResponseGenerationSource.DIAGNOSTIC_SYSTEM,
        )
    )

    assert result.status == LiveResponseBoundaryStatus.BLOCKED


def test_response_boundary_snapshot_tracks_counts() -> None:
    runtime = LiveResponseBoundaryRuntime(generator=ContextAwareFakeGenerator())

    runtime.generate(_request())
    runtime.system_message(
        LiveDeterministicSystemMessage(
            turn_id=make_live_turn_id(),
            kind=LiveResponseKind.DIAGNOSTIC,
            text="Microphone unavailable.",
            source=LiveResponseGenerationSource.DIAGNOSTIC_SYSTEM,
        )
    )
    runtime.validate_for_tts(
        make_live_response(
            turn_id=make_live_turn_id(),
            kind=LiveResponseKind.CONVERSATIONAL,
            text="Invalid source.",
            generation_source=LiveResponseGenerationSource.DIAGNOSTIC_SYSTEM,
            safety=LiveResponseSafety.SAFE_TO_SPEAK,
        )
    )

    snapshot = runtime.snapshot()

    assert snapshot.generated_count == 1
    assert snapshot.deterministic_system_count == 1
    assert snapshot.blocked_count == 1


def test_response_boundary_enum_values_are_stable() -> None:
    assert LiveResponseBoundaryStatus.READY.value == "ready"
    assert LiveResponseIntent.GREETING.value == "greeting"
    assert LiveResponseSurface.VOICE.value == "voice"
    assert LiveResponseBoundaryViolation.MISSING_GENERATOR.value == (
        "missing_generator"
    )