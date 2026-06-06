from __future__ import annotations

from dataclasses import dataclass

import pytest

from jarvis.voice import (
    VoiceDeviceHealth,
    VoiceRuntimeConfig,
    VoiceTTSAudioData,
    VoiceTTSAudioFormat,
    VoiceTTSChunkPlan,
    VoiceTTSPolicy,
    VoiceTTSRequest,
    VoiceTTSRuntime,
    VoiceTTSRuntimeStatus,
    VoiceTTSVoiceInfo,
    make_voice_session_id,
)


@dataclass
class FakeTTSAdapter:
    fail_prepare: bool = False
    fail_synthesize: bool = False
    synthesize_calls: int = 0
    closed: bool = False

    def prepare(
        self,
        config: VoiceRuntimeConfig,
        policy: VoiceTTSPolicy,
    ) -> VoiceTTSVoiceInfo:
        if self.fail_prepare:
            raise RuntimeError("prepare failed")
        return VoiceTTSVoiceInfo(
            provider="fake",
            voice_name=policy.voice_name,
            sample_rate_hz=policy.sample_rate_hz,
            audio_format=VoiceTTSAudioFormat.WAV,
            health=VoiceDeviceHealth.READY,
        )

    def synthesize(
        self,
        request: VoiceTTSRequest,
        plan: VoiceTTSChunkPlan,
        config: VoiceRuntimeConfig,
        policy: VoiceTTSPolicy,
    ) -> VoiceTTSAudioData:
        self.synthesize_calls += 1
        if self.fail_synthesize:
            raise RuntimeError("synthesis failed")
        text = plan.text
        return VoiceTTSAudioData(
            audio=b"RIFFfakewav",
            sample_rate_hz=policy.sample_rate_hz,
            duration_ms=max(100, len(text) * 5),
            audio_format=VoiceTTSAudioFormat.WAV,
            latency_ms=25.0,
            metadata={"fake": True},
        )

    def close(self) -> None:
        self.closed = True


def test_tts_policy_validation() -> None:
    with pytest.raises(ValueError):
        VoiceTTSPolicy(voice_name=" ")

    with pytest.raises(ValueError):
        VoiceTTSPolicy(max_chars_per_chunk=10)

    with pytest.raises(ValueError):
        VoiceTTSPolicy(timeout_seconds=0)


def test_tts_runtime_prepares_voice() -> None:
    runtime = VoiceTTSRuntime(adapter=FakeTTSAdapter())

    result = runtime.prepare()

    assert result.status == VoiceTTSRuntimeStatus.READY
    assert result.voice is not None
    assert result.voice.provider == "fake"


def test_tts_prepare_failure_is_safe() -> None:
    runtime = VoiceTTSRuntime(adapter=FakeTTSAdapter(fail_prepare=True))

    result = runtime.prepare()

    assert result.status == VoiceTTSRuntimeStatus.FAILED
    assert result.chunks == ()


def test_tts_synthesizes_text_into_audio_chunk() -> None:
    runtime = VoiceTTSRuntime(adapter=FakeTTSAdapter())

    result = runtime.synthesize_text(
        text="Certainly. I am checking that now.",
        session_id=make_voice_session_id(),
    )

    assert result.status == VoiceTTSRuntimeStatus.SYNTHESIZING
    assert result.succeeded is True
    assert len(result.chunks) == 1
    assert result.chunks[0].audio
    assert result.first_chunk_latency_ms is not None


def test_tts_splits_long_text_for_streaming_readiness() -> None:
    adapter = FakeTTSAdapter()
    runtime = VoiceTTSRuntime(
        adapter=adapter,
        policy=VoiceTTSPolicy(max_chars_per_chunk=60),
    )
    text = (
        "First sentence is short. "
        "Second sentence is also short. "
        "Third sentence should become another chunk."
    )

    result = runtime.synthesize_text(
        text=text,
        session_id=make_voice_session_id(),
    )

    assert len(result.chunks) >= 2
    assert adapter.synthesize_calls == len(result.chunks)
    assert result.metadata["streaming_ready"] is True


def test_tts_empty_text_degrades() -> None:
    runtime = VoiceTTSRuntime(adapter=FakeTTSAdapter())

    result = runtime.synthesize_text(
        text=" ",
        session_id=make_voice_session_id(),
    )

    assert result.status == VoiceTTSRuntimeStatus.DEGRADED
    assert result.chunks == ()


def test_tts_long_text_is_truncated_safely() -> None:
    runtime = VoiceTTSRuntime(
        adapter=FakeTTSAdapter(),
        policy=VoiceTTSPolicy(
            max_chars_per_chunk=60,
            max_total_chars=120,
        ),
    )

    result = runtime.synthesize_text(
        text="hello " * 100,
        session_id=make_voice_session_id(),
    )

    assert result.succeeded is True
    assert result.metadata["degraded_truncation"] is True


def test_tts_synthesis_failure_is_safe() -> None:
    runtime = VoiceTTSRuntime(
        adapter=FakeTTSAdapter(fail_synthesize=True),
        policy=VoiceTTSPolicy(max_retries=0),
    )

    result = runtime.synthesize_text(
        text="This should fail safely.",
        session_id=make_voice_session_id(),
    )

    assert result.status == VoiceTTSRuntimeStatus.FAILED
    assert result.chunks == ()


def test_tts_retries_after_degraded_failure() -> None:
    @dataclass
    class RecoveringAdapter:
        calls: int = 0

        def prepare(
            self,
            config: VoiceRuntimeConfig,
            policy: VoiceTTSPolicy,
        ) -> VoiceTTSVoiceInfo:
            return VoiceTTSVoiceInfo(
                provider="fake",
                voice_name=policy.voice_name,
                sample_rate_hz=policy.sample_rate_hz,
                audio_format=VoiceTTSAudioFormat.WAV,
                health=VoiceDeviceHealth.READY,
            )

        def synthesize(
            self,
            request: VoiceTTSRequest,
            plan: VoiceTTSChunkPlan,
            config: VoiceRuntimeConfig,
            policy: VoiceTTSPolicy,
        ) -> VoiceTTSAudioData:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary failure")
            return VoiceTTSAudioData(
                audio=b"RIFFfakewav",
                sample_rate_hz=policy.sample_rate_hz,
                duration_ms=150,
                audio_format=VoiceTTSAudioFormat.WAV,
                latency_ms=20.0,
            )

        def close(self) -> None:
            return None

    adapter = RecoveringAdapter()
    runtime = VoiceTTSRuntime(
        adapter=adapter,
        policy=VoiceTTSPolicy(max_retries=1),
    )

    result = runtime.synthesize_text(
        text="Recover and synthesize.",
        session_id=make_voice_session_id(),
    )

    assert result.succeeded is True
    assert adapter.calls == 2


def test_tts_snapshot_tracks_counts() -> None:
    runtime = VoiceTTSRuntime(adapter=FakeTTSAdapter())

    runtime.synthesize_text(
        text="Certainly.",
        session_id=make_voice_session_id(),
    )
    snapshot = runtime.snapshot()

    assert snapshot.synthesized_requests == 1
    assert snapshot.synthesized_chunks == 1
    assert snapshot.last_text == "Certainly."
    assert snapshot.last_first_chunk_latency_ms is not None


def test_tts_reset_closes_adapter() -> None:
    adapter = FakeTTSAdapter()
    runtime = VoiceTTSRuntime(adapter=adapter)

    runtime.prepare()
    result = runtime.reset()

    assert result.status == VoiceTTSRuntimeStatus.CREATED
    assert adapter.closed is True


def test_tts_enum_values_are_stable() -> None:
    assert VoiceTTSRuntimeStatus.SYNTHESIZING.value == "synthesizing"
    assert VoiceTTSAudioFormat.WAV.value == "wav"