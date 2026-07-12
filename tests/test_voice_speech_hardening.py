from __future__ import annotations

import pytest

from jarvis.voice import (
    VoiceHardeningArchitecture,
    VoiceHardeningLayer,
    VoiceHardeningLayerKind,
    VoiceHardeningLayerStatus,
    default_voice_hardening_architecture,
)


def test_voice_hardening_architecture_contains_layers_a_to_h_in_order() -> None:
    architecture = default_voice_hardening_architecture()

    assert architecture.primary_flow == (
        VoiceHardeningLayerKind.HARDWARE,
        VoiceHardeningLayerKind.ECHO_CANCELLATION,
        VoiceHardeningLayerKind.STRONG_VAD,
        VoiceHardeningLayerKind.WHISPER_HALLUCINATION_FILTER,
        VoiceHardeningLayerKind.ATTENTION_GATE,
        VoiceHardeningLayerKind.END_OF_SPEECH,
        VoiceHardeningLayerKind.CONFIDENCE,
        VoiceHardeningLayerKind.SELF_PROTECTION,
    )
    assert (
        tuple(layer.kind for layer in architecture.layers)
        == architecture.primary_flow
    )


def test_voice_hardening_honestly_marks_hardware_and_aec_boundaries() -> None:
    architecture = default_voice_hardening_architecture()

    hardware = architecture.layer(VoiceHardeningLayerKind.HARDWARE)
    echo = architecture.layer(VoiceHardeningLayerKind.ECHO_CANCELLATION)

    assert hardware.status == VoiceHardeningLayerStatus.EXTERNAL_REQUIRED
    assert "headset_microphone_or_directional_microphone" in hardware.controls
    assert echo.status == VoiceHardeningLayerStatus.PARTIAL
    assert "audio_preprocessing_boundary" in echo.controls
    assert "webrtc_vad_pre_gate" in echo.controls
    assert "aec_ns_agc_capability_truth" in echo.controls


def test_voice_hardening_defends_against_whisper_silence_hallucinations() -> None:
    architecture = default_voice_hardening_architecture()

    hallucination_filter = architecture.layer(
        VoiceHardeningLayerKind.WHISPER_HALLUCINATION_FILTER
    )

    assert hallucination_filter.status == VoiceHardeningLayerStatus.CONFIGURED
    assert "max_no_speech_prob" in hallucination_filter.controls
    assert "known_silence_hallucinations" in hallucination_filter.controls
    assert any(
        "thanks for watching" in acceptance
        for acceptance in hallucination_filter.acceptance
    )


def test_voice_hardening_keeps_router_and_self_protection() -> None:
    architecture = default_voice_hardening_architecture()

    attention = architecture.layer(VoiceHardeningLayerKind.ATTENTION_GATE)
    endpoint = architecture.layer(VoiceHardeningLayerKind.END_OF_SPEECH)
    confidence = architecture.layer(VoiceHardeningLayerKind.CONFIDENCE)
    self_protection = architecture.layer(VoiceHardeningLayerKind.SELF_PROTECTION)

    assert "wake_word_required_when_inactive" in attention.controls
    assert "end_silence_ms" in endpoint.controls
    assert "min_action_confidence" in confidence.controls
    assert "assistant_speaking" in self_protection.controls
    assert "active_playback" in self_protection.controls


def test_voice_hardening_architecture_rejects_missing_controls() -> None:
    with pytest.raises(ValueError):
        VoiceHardeningLayer(
            kind=VoiceHardeningLayerKind.STRONG_VAD,
            label="Layer C - Strong VAD",
            purpose="Reject weak noise.",
            status=VoiceHardeningLayerStatus.CONFIGURED,
            runtime_owner="VoiceActivityRuntime",
            controls=(),
            acceptance=("noise is rejected",),
        )


def test_voice_hardening_architecture_rejects_flow_mismatch() -> None:
    architecture = default_voice_hardening_architecture()

    with pytest.raises(ValueError):
        VoiceHardeningArchitecture(
            layers=architecture.layers,
            primary_flow=tuple(reversed(architecture.primary_flow)),
        )
