from __future__ import annotations

from dataclasses import dataclass

import pytest

from jarvis.voice import (
    VoiceDeviceHealth,
    VoicePlaybackAdapterReport,
    VoicePlaybackPolicy,
    VoicePlaybackRuntime,
    VoicePlaybackRuntimeStatus,
    VoiceRuntimeConfig,
    VoiceSpeakerDeviceInfo,
    VoiceTTSChunk,
    VoiceTTSChunkStatus,
    make_voice_session_id,
    make_voice_tts_chunk_id,
    utc_now,
)


@dataclass
class FakeSpeakerAdapter:
    fail_prepare: bool = False
    fail_play: bool = False
    fail_stop: bool = False
    prepared: bool = False
    played: int = 0
    stopped: bool = False
    closed: bool = False

    def prepare(
        self,
        config: VoiceRuntimeConfig,
        policy: VoicePlaybackPolicy,
    ) -> VoiceSpeakerDeviceInfo:
        if self.fail_prepare:
            raise RuntimeError("prepare failed")
        self.prepared = True
        return VoiceSpeakerDeviceInfo(
            provider="fake",
            name="Fake Speaker",
            sample_rate_hz=22_050,
            channels=1,
            health=VoiceDeviceHealth.READY,
        )

    def play(
        self,
        chunk: VoiceTTSChunk,
        policy: VoicePlaybackPolicy,
    ) -> VoicePlaybackAdapterReport:
        if self.fail_play:
            raise RuntimeError("play failed")
        self.played += 1
        return VoicePlaybackAdapterReport(
            played=True,
            sample_rate_hz=chunk.sample_rate_hz,
            duration_ms=chunk.duration_ms,
            latency_ms=15.0,
            bytes_played=len(chunk.audio),
            metadata={"fake": True},
        )

    def stop(self) -> None:
        if self.fail_stop:
            raise RuntimeError("stop failed")
        self.stopped = True

    def close(self) -> None:
        self.closed = True


def _chunk(
    *,
    status: VoiceTTSChunkStatus = VoiceTTSChunkStatus.SYNTHESIZED,
) -> VoiceTTSChunk:
    return VoiceTTSChunk(
        chunk_id=make_voice_tts_chunk_id(),
        session_id=make_voice_session_id(),
        status=status,
        audio=b"RIFFfakewav",
        sample_rate_hz=22_050,
        duration_ms=150,
        created_at=utc_now(),
    )


def test_playback_policy_validation() -> None:
    with pytest.raises(ValueError):
        VoicePlaybackPolicy(max_queue_chunks=0)

    with pytest.raises(ValueError):
        VoicePlaybackPolicy(target_start_latency_ms=0)


def test_playback_prepares_speaker() -> None:
    adapter = FakeSpeakerAdapter()
    runtime = VoicePlaybackRuntime(adapter=adapter)

    result = runtime.prepare()

    assert result.status == VoicePlaybackRuntimeStatus.READY
    assert result.speaker is not None
    assert result.speaker.name == "Fake Speaker"
    assert adapter.prepared is True


def test_playback_prepare_failure_is_safe() -> None:
    runtime = VoicePlaybackRuntime(
        adapter=FakeSpeakerAdapter(fail_prepare=True)
    )

    result = runtime.prepare()

    assert result.status == VoicePlaybackRuntimeStatus.FAILED
    assert result.succeeded is False


def test_playback_enqueue_and_play_next() -> None:
    adapter = FakeSpeakerAdapter()
    runtime = VoicePlaybackRuntime(adapter=adapter)

    enqueue = runtime.enqueue_chunk(_chunk())
    played = runtime.play_next()

    assert enqueue.status == VoicePlaybackRuntimeStatus.QUEUED
    assert played.status == VoicePlaybackRuntimeStatus.PLAYING
    assert played.played_chunks
    assert adapter.played == 1
    assert played.first_audio_latency_ms is not None


def test_playback_play_all_drains_queue() -> None:
    adapter = FakeSpeakerAdapter()
    runtime = VoicePlaybackRuntime(adapter=adapter)

    runtime.enqueue_chunks((_chunk(), _chunk(), _chunk()))
    result = runtime.play_all()
    snapshot = runtime.snapshot()

    assert result.status == VoicePlaybackRuntimeStatus.READY
    assert len(result.played_chunks) == 3
    assert adapter.played == 3
    assert snapshot.queued_chunks == 0
    assert snapshot.played_chunks == 3


def test_playback_rejects_empty_enqueue() -> None:
    runtime = VoicePlaybackRuntime(adapter=FakeSpeakerAdapter())

    result = runtime.enqueue_chunks(())

    assert result.status == VoicePlaybackRuntimeStatus.DEGRADED


def test_playback_rejects_queue_over_capacity() -> None:
    runtime = VoicePlaybackRuntime(
        adapter=FakeSpeakerAdapter(),
        policy=VoicePlaybackPolicy(max_queue_chunks=1),
    )

    result = runtime.enqueue_chunks((_chunk(), _chunk()))

    assert result.status == VoicePlaybackRuntimeStatus.DEGRADED
    assert result.queued_chunks == 0


def test_playback_invalid_chunk_raises() -> None:
    runtime = VoicePlaybackRuntime(adapter=FakeSpeakerAdapter())

    with pytest.raises(ValueError):
        runtime.enqueue_chunk(_chunk(status=VoiceTTSChunkStatus.FAILED))


def test_playback_failure_is_safe() -> None:
    runtime = VoicePlaybackRuntime(adapter=FakeSpeakerAdapter(fail_play=True))

    runtime.enqueue_chunk(_chunk())
    result = runtime.play_next()
    snapshot = runtime.snapshot()

    assert result.status == VoicePlaybackRuntimeStatus.FAILED
    assert snapshot.failed_chunks == 1
    assert snapshot.last_error == "play failed"


def test_playback_stop_clears_queue() -> None:
    adapter = FakeSpeakerAdapter()
    runtime = VoicePlaybackRuntime(adapter=adapter)

    runtime.enqueue_chunks((_chunk(), _chunk()))
    result = runtime.stop()
    snapshot = runtime.snapshot()

    assert result.status == VoicePlaybackRuntimeStatus.STOPPED
    assert result.queued_chunks == 0
    assert adapter.stopped is True
    assert snapshot.stopped_count == 1


def test_playback_stop_failure_degrades() -> None:
    runtime = VoicePlaybackRuntime(adapter=FakeSpeakerAdapter(fail_stop=True))

    result = runtime.stop()

    assert result.status == VoicePlaybackRuntimeStatus.DEGRADED
    assert result.metadata["error"] == "stop failed"


def test_playback_clear_resets_stop_flag_and_queue() -> None:
    runtime = VoicePlaybackRuntime(adapter=FakeSpeakerAdapter())

    runtime.enqueue_chunks((_chunk(), _chunk()))
    runtime.stop()
    result = runtime.clear()

    assert result.status == VoicePlaybackRuntimeStatus.READY
    assert result.queued_chunks == 0


def test_playback_reset_closes_adapter() -> None:
    adapter = FakeSpeakerAdapter()
    runtime = VoicePlaybackRuntime(adapter=adapter)

    runtime.prepare()
    result = runtime.reset()

    assert result.status == VoicePlaybackRuntimeStatus.CREATED
    assert adapter.closed is True


def test_playback_snapshot_tracks_state() -> None:
    runtime = VoicePlaybackRuntime(adapter=FakeSpeakerAdapter())

    runtime.enqueue_chunk(_chunk())
    runtime.play_next()
    snapshot = runtime.snapshot()

    assert snapshot.played_chunks == 1
    assert snapshot.last_first_audio_latency_ms is not None
    assert snapshot.current_playback is not None


def test_playback_enum_values_are_stable() -> None:
    assert VoicePlaybackRuntimeStatus.PLAYING.value == "playing"