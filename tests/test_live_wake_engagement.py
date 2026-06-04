from __future__ import annotations

import pytest

from jarvis.live import (
    LiveEngagementDecision,
    LiveEngagementReason,
    LiveEventBridgeRuntime,
    LiveSessionConfig,
    LiveSessionMode,
    LiveSessionStateRuntime,
    LiveWakeEngagementPolicy,
    LiveWakeEngagementRequest,
    LiveWakeEngagementRuntime,
    LiveWakeEngagementStatus,
)


def _runtime() -> LiveWakeEngagementRuntime:
    state = LiveSessionStateRuntime(
        config=LiveSessionConfig(
            mode=LiveSessionMode.REAL_VOICE,
            real_microphone_enabled=True,
            real_stt_enabled=True,
            real_tts_enabled=True,
        )
    )
    state.start()
    state.mark_ready()
    bridge = LiveEventBridgeRuntime(live_state=state)
    return LiveWakeEngagementRuntime(live_state=state, bridge=bridge)


def test_wake_engagement_policy_validation() -> None:
    with pytest.raises(ValueError):
        LiveWakeEngagementPolicy(wake_word=" ")

    with pytest.raises(ValueError):
        LiveWakeEngagementPolicy(min_speech_probability=1.5)


def test_wake_engagement_ignores_empty_speech() -> None:
    runtime = _runtime()

    result = runtime.evaluate(
        LiveWakeEngagementRequest(text=" ", speech_probability=0.9)
    )

    assert result.status == LiveWakeEngagementStatus.READY
    assert result.decision == LiveEngagementDecision.IGNORE
    assert result.reason == LiveEngagementReason.NO_SPEECH


def test_wake_engagement_ignores_low_confidence() -> None:
    runtime = _runtime()

    result = runtime.evaluate(
        LiveWakeEngagementRequest(
            text="jarvis",
            speech_probability=0.1,
        )
    )

    assert result.decision == LiveEngagementDecision.IGNORE
    assert result.reason == LiveEngagementReason.LOW_CONFIDENCE


def test_wake_engagement_detects_wake_word() -> None:
    runtime = _runtime()

    result = runtime.evaluate(
        LiveWakeEngagementRequest(
            text="Jarvis",
            speech_probability=0.95,
        )
    )

    assert result.status == LiveWakeEngagementStatus.READY
    assert result.decision == LiveEngagementDecision.ENGAGE
    assert result.reason == LiveEngagementReason.WAKE_WORD_DETECTED
    assert result.bridge_result is not None


def test_wake_engagement_ignores_background_speech_when_sleeping() -> None:
    state = LiveSessionStateRuntime(
        config=LiveSessionConfig(
            mode=LiveSessionMode.REAL_VOICE,
            real_microphone_enabled=True,
            real_stt_enabled=True,
            real_tts_enabled=True,
        )
    )
    state.start()
    bridge = LiveEventBridgeRuntime(live_state=state)
    runtime = LiveWakeEngagementRuntime(live_state=state, bridge=bridge)

    result = runtime.evaluate(
        LiveWakeEngagementRequest(
            text="this is just background speech",
            speech_probability=0.95,
        )
    )

    assert result.decision == LiveEngagementDecision.IGNORE
    assert result.reason == LiveEngagementReason.BACKGROUND_SPEECH


def test_wake_engagement_continues_active_session_without_wake_word() -> None:
    runtime = _runtime()
    runtime.evaluate(
        LiveWakeEngagementRequest(
            text="Jarvis",
            speech_probability=0.95,
        )
    )

    # mark state as active after engagement
    runtime.live_state.start_user_turn()

    result = runtime.evaluate(
        LiveWakeEngagementRequest(
            text="continue the explanation",
            speech_probability=0.95,
        )
    )

    assert result.decision in {
        LiveEngagementDecision.CONTINUE_ACTIVE_SESSION,
        LiveEngagementDecision.ENGAGE,
    }


def test_wake_engagement_interrupts_on_stop_word() -> None:
    runtime = _runtime()

    result = runtime.evaluate(
        LiveWakeEngagementRequest(
            text="stop",
            speech_probability=0.95,
            assistant_is_speaking=True,
        )
    )

    assert result.status == LiveWakeEngagementStatus.READY
    assert result.decision == LiveEngagementDecision.INTERRUPT
    assert result.reason == LiveEngagementReason.USER_INTERRUPTED
    assert result.bridge_result is not None
    assert result.bridge_result.should_interrupt is True


def test_wake_engagement_disengages_on_sleep_word() -> None:
    runtime = _runtime()

    result = runtime.evaluate(
        LiveWakeEngagementRequest(
            text="sleep",
            speech_probability=0.95,
        )
    )

    assert result.status == LiveWakeEngagementStatus.READY
    assert result.decision == LiveEngagementDecision.DISENGAGE
    assert result.reason == LiveEngagementReason.USER_DISMISSED


def test_wake_engagement_direct_engage_and_disengage_validate_reason() -> None:
    runtime = _runtime()

    with pytest.raises(ValueError):
        runtime.engage(reason=" ")

    with pytest.raises(ValueError):
        runtime.disengage(reason=" ")


def test_wake_engagement_snapshot_tracks_counts() -> None:
    runtime = _runtime()

    runtime.evaluate(
        LiveWakeEngagementRequest(text="Jarvis", speech_probability=0.95)
    )
    runtime.evaluate(
        LiveWakeEngagementRequest(text="", speech_probability=0.95)
    )
    runtime.evaluate(
        LiveWakeEngagementRequest(text="stop", speech_probability=0.95)
    )

    snapshot = runtime.snapshot()

    assert snapshot.status == LiveWakeEngagementStatus.READY
    assert snapshot.evaluated_count == 3
    assert snapshot.engaged_count >= 1
    assert snapshot.ignored_count >= 1
    assert snapshot.interrupted_count >= 1


def test_wake_engagement_enum_values_are_stable() -> None:
    assert LiveWakeEngagementStatus.READY.value == "ready"
    assert LiveEngagementDecision.ENGAGE.value == "engage"
    assert LiveEngagementReason.WAKE_WORD_DETECTED.value == (
        "wake_word_detected"
    )