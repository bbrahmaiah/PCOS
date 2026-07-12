from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class VoiceHardeningLayerKind(StrEnum):
    HARDWARE = "hardware"
    ECHO_CANCELLATION = "echo_cancellation"
    STRONG_VAD = "strong_vad"
    WHISPER_HALLUCINATION_FILTER = "whisper_hallucination_filter"
    ATTENTION_GATE = "attention_gate"
    END_OF_SPEECH = "end_of_speech"
    CONFIDENCE = "confidence"
    SELF_PROTECTION = "self_protection"


class VoiceHardeningLayerStatus(StrEnum):
    CONFIGURED = "configured"
    PARTIAL = "partial"
    EXTERNAL_REQUIRED = "external_required"


@dataclass(frozen=True, slots=True)
class VoiceHardeningLayer:
    kind: VoiceHardeningLayerKind
    label: str
    purpose: str
    status: VoiceHardeningLayerStatus
    runtime_owner: str
    controls: tuple[str, ...]
    acceptance: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("Voice hardening layer label cannot be empty.")
        if not self.purpose.strip():
            raise ValueError("Voice hardening layer purpose cannot be empty.")
        if not self.runtime_owner.strip():
            raise ValueError("Voice hardening layer runtime_owner cannot be empty.")
        if not self.controls:
            raise ValueError("Voice hardening layer controls cannot be empty.")
        if not self.acceptance:
            raise ValueError("Voice hardening layer acceptance cannot be empty.")


@dataclass(frozen=True, slots=True)
class VoiceHardeningArchitecture:
    layers: tuple[VoiceHardeningLayer, ...]
    primary_flow: tuple[VoiceHardeningLayerKind, ...]

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("Voice hardening architecture requires layers.")
        layer_kinds = tuple(layer.kind for layer in self.layers)
        if len(set(layer_kinds)) != len(layer_kinds):
            raise ValueError("Voice hardening layers must be unique.")
        if layer_kinds != self.primary_flow:
            raise ValueError("Voice hardening primary_flow must match layer order.")

    def layer(self, kind: VoiceHardeningLayerKind) -> VoiceHardeningLayer:
        for layer in self.layers:
            if layer.kind == kind:
                return layer
        raise KeyError(kind)


def default_voice_hardening_architecture() -> VoiceHardeningArchitecture:
    layers = (
        VoiceHardeningLayer(
            kind=VoiceHardeningLayerKind.HARDWARE,
            label="Layer A - Hardware",
            purpose="Keep JARVIS from hearing room echo before software sees audio.",
            status=VoiceHardeningLayerStatus.EXTERNAL_REQUIRED,
            runtime_owner="operator_audio_setup",
            controls=(
                "headset_microphone_or_directional_microphone",
                "avoid_laptop_speaker_plus_laptop_microphone_loop",
            ),
            acceptance=(
                "microphone captures the user more strongly than speakers",
                "room playback does not continuously trigger VAD",
            ),
        ),
        VoiceHardeningLayer(
            kind=VoiceHardeningLayerKind.ECHO_CANCELLATION,
            label="Layer B - Echo Cancellation",
            purpose="Suppress playback echo before it becomes speech intent.",
            status=VoiceHardeningLayerStatus.PARTIAL,
            runtime_owner="VoiceAudioPreprocessingRuntime + barge_in_runtime",
            controls=(
                "audio_preprocessing_boundary",
                "webrtc_vad_pre_gate",
                "assistant_speaking_state",
                "active_playback_route_guard",
                "barge_in_echo_guard",
                "aec_ns_agc_capability_truth",
            ),
            acceptance=(
                "assistant playback cannot trigger cognition by itself",
                "AEC/NS/AGC are reported unavailable unless a real engine exists",
            ),
        ),
        VoiceHardeningLayer(
            kind=VoiceHardeningLayerKind.STRONG_VAD,
            label="Layer C - Strong VAD",
            purpose="Promote only sustained, energetic speech into STT.",
            status=VoiceHardeningLayerStatus.CONFIGURED,
            runtime_owner="VoiceActivityRuntime",
            controls=(
                "min_energy",
                "speech_start_ratio",
                "start_trigger_frames",
                "min_speech_ms",
            ),
            acceptance=(
                "short noise bursts stay below the speech boundary",
                "speech requires sustained energy before capture starts",
            ),
        ),
        VoiceHardeningLayer(
            kind=VoiceHardeningLayerKind.WHISPER_HALLUCINATION_FILTER,
            label="Layer D - Whisper Hallucination Filter",
            purpose=(
                "Reject classic Whisper silence hallucinations at the STT "
                "boundary."
            ),
            status=VoiceHardeningLayerStatus.CONFIGURED,
            runtime_owner="VoiceSTTRuntime",
            controls=(
                "max_no_speech_prob",
                "max_compression_ratio",
                "min_avg_logprob",
                "known_silence_hallucinations",
                "min_transcript_chars",
            ),
            acceptance=(
                "no_speech_prob above policy is rejected",
                "common silence phrases such as thanks for watching are rejected",
            ),
        ),
        VoiceHardeningLayer(
            kind=VoiceHardeningLayerKind.ATTENTION_GATE,
            label="Layer E - Attention Gate",
            purpose="Respond only when the user is addressing JARVIS.",
            status=VoiceHardeningLayerStatus.CONFIGURED,
            runtime_owner="TranscriptAttentionGate",
            controls=(
                "wake_word_required_when_inactive",
                "active_attention_window",
                "known_silence_hallucinations",
                "min_words_without_wake",
            ),
            acceptance=(
                "random room speech does not trigger cognition",
                "follow-up speech is accepted only inside active attention",
            ),
        ),
        VoiceHardeningLayer(
            kind=VoiceHardeningLayerKind.END_OF_SPEECH,
            label="Layer F - End Of Speech",
            purpose=(
                "Wait for natural pause boundaries instead of cutting the "
                "user off."
            ),
            status=VoiceHardeningLayerStatus.CONFIGURED,
            runtime_owner="VoiceActivityRuntime + VoiceSessionLoopRuntime",
            controls=(
                "end_silence_ms",
                "max_silence_ms",
                "max_segment_ms",
                "partial_transcript_every_frames",
            ),
            acceptance=(
                "multi-part commands survive short pauses",
                "final transcripts are emitted after a stable endpoint",
            ),
        ),
        VoiceHardeningLayer(
            kind=VoiceHardeningLayerKind.CONFIDENCE,
            label="Layer G - Confidence",
            purpose="Separate prediction, dialogue, and action safety.",
            status=VoiceHardeningLayerStatus.CONFIGURED,
            runtime_owner="VoiceSTTRuntime + VoiceCognitiveRouter",
            controls=(
                "min_partial_confidence",
                "min_final_confidence",
                "min_action_confidence",
                "allow_action_candidate",
            ),
            acceptance=(
                "low-confidence transcripts degrade instead of acting",
                "action candidates require the highest confidence lane",
            ),
        ),
        VoiceHardeningLayer(
            kind=VoiceHardeningLayerKind.SELF_PROTECTION,
            label="Layer H - JARVIS Self Protection",
            purpose="Prevent JARVIS from treating its own speech as user intent.",
            status=VoiceHardeningLayerStatus.CONFIGURED,
            runtime_owner="VoiceSessionLoopRuntime + VoiceCognitiveRouter",
            controls=(
                "assistant_speaking",
                "active_playback",
                "interruptible_playback",
                "barge_in_disposition",
            ),
            acceptance=(
                "TTS playback does not recursively trigger STT responses",
                "true user interruption can stop playback and route a new turn",
            ),
        ),
    )
    return VoiceHardeningArchitecture(
        layers=layers,
        primary_flow=tuple(layer.kind for layer in layers),
    )
