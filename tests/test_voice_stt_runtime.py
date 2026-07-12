from __future__ import annotations

import sys
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from jarvis.voice import (
    VoiceDeviceHealth,
    VoiceInputFrame,
    VoiceInputFrameKind,
    VoiceRuntimeConfig,
    VoiceSTTMode,
    VoiceSTTModelInfo,
    VoiceSTTPolicy,
    VoiceSTTRequest,
    VoiceSTTRuntime,
    VoiceSTTRuntimeStatus,
    VoiceSTTTranscriptSafety,
    VoiceTranscriptKind,
    make_voice_frame_id,
    make_voice_session_id,
    utc_now,
)


@dataclass
class DualLaneFakeSTTAdapter:
    partial_text: str = "Jarvis explain"
    final_text: str = "Jarvis explain PID control."
    partial_confidence: float = 0.65
    final_confidence: float = 0.92
    metadata: dict[str, object] = field(default_factory=dict)
    fail_prepare: bool = False
    fail_transcribe: bool = False
    prepared: bool = False
    closed: bool = False
    prepare_prewarm_values: list[bool] = field(default_factory=list)

    def prepare(
        self,
        config: VoiceRuntimeConfig,
        policy: VoiceSTTPolicy,
    ) -> tuple[VoiceSTTModelInfo, VoiceSTTModelInfo]:
        if self.fail_prepare:
            raise RuntimeError("prepare failed")
        self.prepared = True
        self.prepare_prewarm_values.append(policy.prewarm_on_prepare)
        partial = VoiceSTTModelInfo(
            provider="fake",
            model_name=policy.partial_model_name,
            device=policy.device,
            compute_type=policy.compute_type,
            language=config.stt_language,
            mode=VoiceSTTMode.FAST_PARTIAL,
            health=VoiceDeviceHealth.READY,
        )
        final = VoiceSTTModelInfo(
            provider="fake",
            model_name=policy.final_model_name,
            device=policy.device,
            compute_type=policy.compute_type,
            language=config.stt_language,
            mode=VoiceSTTMode.ACCURATE_FINAL,
            health=VoiceDeviceHealth.READY,
        )
        return partial, final

    def transcribe(
        self,
        request: VoiceSTTRequest,
        config: VoiceRuntimeConfig,
        policy: VoiceSTTPolicy,
    ) -> tuple[str, float, str] | tuple[str, float, str, dict[str, object]]:
        if self.fail_transcribe:
            raise RuntimeError("transcribe failed")

        if request.mode == VoiceSTTMode.FAST_PARTIAL:
            return (
                self.partial_text,
                self.partial_confidence,
                policy.partial_model_name,
                self.metadata,
            )

        return (
            self.final_text,
            self.final_confidence,
            policy.final_model_name,
            self.metadata,
        )

    def close(self) -> None:
        self.closed = True


class CountingWhisperModel:
    created_models: list[str] = []

    def __init__(
        self,
        model_name: str,
        *,
        device: str,
        compute_type: str,
    ) -> None:
        self.created_models.append(model_name)
        self.device = device
        self.compute_type = compute_type


def _frame() -> VoiceInputFrame:
    return VoiceInputFrame(
        frame_id=make_voice_frame_id(),
        session_id=make_voice_session_id(),
        kind=VoiceInputFrameKind.PCM16_MONO,
        sample_rate_hz=16_000,
        channels=1,
        data=b"\x00\x01" * 320,
        captured_at=utc_now(),
        duration_ms=20,
    )


def test_stt_policy_validation() -> None:
    with pytest.raises(ValueError):
        VoiceSTTPolicy(partial_model_name=" ")

    with pytest.raises(ValueError):
        VoiceSTTPolicy(final_model_name=" ")

    with pytest.raises(ValueError):
        VoiceSTTPolicy(partial_beam_size=0)

    with pytest.raises(ValueError):
        VoiceSTTPolicy(min_action_confidence=2.0)


def test_stt_policy_defaults_to_small_for_partial_and_final() -> None:
    policy = VoiceSTTPolicy()

    assert policy.partial_model_name == "small"
    assert policy.final_model_name == "small"


def test_faster_whisper_adapter_reuses_identical_model_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jarvis.voice.stt_runtime import FasterWhisperSTTAdapter

    CountingWhisperModel.created_models = []
    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        SimpleNamespace(WhisperModel=CountingWhisperModel),
    )
    adapter = FasterWhisperSTTAdapter()

    partial, final = adapter.prepare(
        VoiceRuntimeConfig(),
        VoiceSTTPolicy(partial_model_name="small", final_model_name="small"),
    )

    assert CountingWhisperModel.created_models == ["small"]
    assert partial.model_name == "small"
    assert final.model_name == "small"


def test_dual_lane_stt_prepares_both_models() -> None:
    adapter = DualLaneFakeSTTAdapter()
    runtime = VoiceSTTRuntime(adapter=adapter)

    result = runtime.prepare()
    snapshot = runtime.snapshot()

    assert result.status == VoiceSTTRuntimeStatus.READY
    assert adapter.prepared is True
    assert snapshot.partial_model is not None
    assert snapshot.final_model is not None
    assert snapshot.partial_model.mode == VoiceSTTMode.FAST_PARTIAL
    assert snapshot.final_model.mode == VoiceSTTMode.ACCURATE_FINAL


def test_stt_warm_models_forces_prewarm_policy() -> None:
    adapter = DualLaneFakeSTTAdapter()
    runtime = VoiceSTTRuntime(
        adapter=adapter,
        policy=VoiceSTTPolicy(prewarm_on_prepare=False),
    )

    prepared = runtime.prepare()
    warmed = runtime.warm_models()

    assert prepared.status == VoiceSTTRuntimeStatus.READY
    assert warmed.status == VoiceSTTRuntimeStatus.READY
    assert adapter.prepare_prewarm_values == [False, True]
    assert warmed.metadata["prewarmed"] is True


def test_stt_prepare_failure_is_safe() -> None:
    runtime = VoiceSTTRuntime(
        adapter=DualLaneFakeSTTAdapter(fail_prepare=True)
    )

    result = runtime.prepare()

    assert result.status == VoiceSTTRuntimeStatus.FAILED
    assert result.transcript is None


def test_fast_partial_transcript_is_prediction_only() -> None:
    runtime = VoiceSTTRuntime(adapter=DualLaneFakeSTTAdapter())

    result = runtime.transcribe_partial((_frame(), _frame()))

    assert result.status == VoiceSTTRuntimeStatus.TRANSCRIBING
    assert result.transcript is not None
    assert result.candidate is not None
    assert result.transcript.kind == VoiceTranscriptKind.PARTIAL
    assert result.candidate.mode == VoiceSTTMode.FAST_PARTIAL
    assert result.candidate.safety == VoiceSTTTranscriptSafety.PREDICTION_ONLY
    assert result.safe_for_action is False


def test_accurate_final_transcript_is_safe_for_dialogue() -> None:
    runtime = VoiceSTTRuntime(adapter=DualLaneFakeSTTAdapter())

    result = runtime.transcribe_final((_frame(), _frame()))

    assert result.status == VoiceSTTRuntimeStatus.TRANSCRIBING
    assert result.transcript is not None
    assert result.candidate is not None
    assert result.transcript.kind == VoiceTranscriptKind.FINAL
    assert result.candidate.mode == VoiceSTTMode.ACCURATE_FINAL
    assert result.candidate.safety == VoiceSTTTranscriptSafety.SAFE_FOR_DIALOGUE
    assert result.safe_for_action is False


def test_accurate_final_can_be_action_safe_with_policy() -> None:
    runtime = VoiceSTTRuntime(
        adapter=DualLaneFakeSTTAdapter(final_confidence=0.91),
        policy=VoiceSTTPolicy(min_action_confidence=0.70),
    )

    result = runtime.transcribe_final(
        (_frame(), _frame()),
        allow_action_candidate=True,
    )

    assert result.candidate is not None
    assert result.candidate.safety == VoiceSTTTranscriptSafety.SAFE_FOR_ACTION
    assert result.safe_for_action is True


def test_partial_cannot_be_action_safe_even_with_high_confidence() -> None:
    runtime = VoiceSTTRuntime(
        adapter=DualLaneFakeSTTAdapter(partial_confidence=0.99),
        policy=VoiceSTTPolicy(min_action_confidence=0.70),
    )

    result = runtime.transcribe_partial((_frame(),))

    assert result.candidate is not None
    assert result.candidate.safety == VoiceSTTTranscriptSafety.PREDICTION_ONLY
    assert result.safe_for_action is False


def test_low_partial_confidence_degrades() -> None:
    runtime = VoiceSTTRuntime(
        adapter=DualLaneFakeSTTAdapter(partial_confidence=0.10),
        policy=VoiceSTTPolicy(min_partial_confidence=0.30),
    )

    result = runtime.transcribe_partial((_frame(),))

    assert result.status == VoiceSTTRuntimeStatus.DEGRADED
    assert result.transcript is None


def test_low_final_confidence_degrades() -> None:
    runtime = VoiceSTTRuntime(
        adapter=DualLaneFakeSTTAdapter(final_confidence=0.20),
        policy=VoiceSTTPolicy(min_final_confidence=0.45),
    )

    result = runtime.transcribe_final((_frame(),))

    assert result.status == VoiceSTTRuntimeStatus.DEGRADED
    assert result.transcript is None


def test_empty_results_degrade_after_threshold() -> None:
    runtime = VoiceSTTRuntime(
        adapter=DualLaneFakeSTTAdapter(
            partial_text="",
            final_text="",
        ),
        policy=VoiceSTTPolicy(max_empty_results_before_degraded=2),
    )

    first = runtime.transcribe_final((_frame(),))
    second = runtime.transcribe_final((_frame(),))

    assert first.status == VoiceSTTRuntimeStatus.READY
    assert second.status == VoiceSTTRuntimeStatus.DEGRADED


def test_stt_rejects_known_silence_hallucination() -> None:
    runtime = VoiceSTTRuntime(
        adapter=DualLaneFakeSTTAdapter(final_text="Thanks for watching.")
    )

    result = runtime.transcribe_final((_frame(),))

    assert result.status == VoiceSTTRuntimeStatus.READY
    assert result.transcript is None
    assert result.message == "STT transcript rejected by hallucination policy"
    assert result.metadata["rejection_reason"] == "known_silence_hallucination"


@pytest.mark.parametrize("text", ["Bless you.", "service.", "God."])
def test_stt_rejects_latest_live_log_hallucinations(text: str) -> None:
    runtime = VoiceSTTRuntime(adapter=DualLaneFakeSTTAdapter(final_text=text))

    result = runtime.transcribe_final((_frame(),))

    assert result.status == VoiceSTTRuntimeStatus.READY
    assert result.transcript is None
    assert result.metadata["rejection_reason"] == "known_silence_hallucination"


def test_stt_rejects_high_no_speech_probability() -> None:
    runtime = VoiceSTTRuntime(
        adapter=DualLaneFakeSTTAdapter(
            final_text="That was good.",
            metadata={"max_no_speech_prob": 0.91},
        )
    )

    result = runtime.transcribe_final((_frame(),))

    assert result.status == VoiceSTTRuntimeStatus.READY
    assert result.transcript is None
    assert result.metadata["rejection_reason"] == "known_silence_hallucination"


def test_stt_rejects_high_no_speech_probability_for_non_blacklisted_text() -> None:
    runtime = VoiceSTTRuntime(
        adapter=DualLaneFakeSTTAdapter(
            final_text="random background sentence",
            metadata={"max_no_speech_prob": 0.91},
        )
    )

    result = runtime.transcribe_final((_frame(),))

    assert result.status == VoiceSTTRuntimeStatus.READY
    assert result.transcript is None
    assert result.metadata["rejection_reason"] == "high_no_speech_probability"


def test_stt_rejects_too_short_text() -> None:
    runtime = VoiceSTTRuntime(adapter=DualLaneFakeSTTAdapter(final_text="x"))

    result = runtime.transcribe_final((_frame(),))

    assert result.status == VoiceSTTRuntimeStatus.READY
    assert result.transcript is None
    assert result.metadata["rejection_reason"] == "too_short"


def test_stt_keeps_whisper_quality_metadata_on_valid_transcript() -> None:
    runtime = VoiceSTTRuntime(
        adapter=DualLaneFakeSTTAdapter(
            metadata={
                "max_no_speech_prob": 0.02,
                "min_avg_logprob": -0.15,
                "max_compression_ratio": 1.1,
            }
        )
    )

    result = runtime.transcribe_final((_frame(),))

    assert result.transcript is not None
    assert result.transcript.metadata["max_no_speech_prob"] == 0.02
    assert result.metadata["max_compression_ratio"] == 1.1


def test_transcription_failure_is_safe() -> None:
    runtime = VoiceSTTRuntime(
        adapter=DualLaneFakeSTTAdapter(fail_transcribe=True)
    )

    result = runtime.transcribe_final((_frame(),))

    assert result.status == VoiceSTTRuntimeStatus.FAILED
    assert result.transcript is None


def test_stt_snapshot_tracks_latency_and_safety() -> None:
    runtime = VoiceSTTRuntime(adapter=DualLaneFakeSTTAdapter())

    runtime.transcribe_partial((_frame(),))
    runtime.transcribe_final((_frame(),))
    snapshot = runtime.snapshot()

    assert snapshot.partial_transcripts == 1
    assert snapshot.final_transcripts == 1
    assert snapshot.last_latency_ms is not None
    assert snapshot.last_text == "Jarvis explain PID control."
    assert snapshot.last_safety == VoiceSTTTranscriptSafety.SAFE_FOR_DIALOGUE


def test_stt_reset_closes_adapter() -> None:
    adapter = DualLaneFakeSTTAdapter()
    runtime = VoiceSTTRuntime(adapter=adapter)

    runtime.prepare()
    result = runtime.reset()

    assert result.status == VoiceSTTRuntimeStatus.CREATED
    assert adapter.closed is True


def test_stt_request_rejects_empty_frames() -> None:
    with pytest.raises(ValueError):
        VoiceSTTRequest(
            frames=(),
            mode=VoiceSTTMode.FAST_PARTIAL,
        )


def test_stt_enum_values_are_stable() -> None:
    assert VoiceSTTMode.FAST_PARTIAL.value == "fast_partial"
    assert VoiceSTTMode.ACCURATE_FINAL.value == "accurate_final"
    assert VoiceSTTTranscriptSafety.SAFE_FOR_ACTION.value == "safe_for_action"
