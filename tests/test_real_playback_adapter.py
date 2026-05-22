from __future__ import annotations

import pytest

from jarvis.presence.adapters import (
    AudioPlaybackBackend,
    PlaybackStatus,
    RealAudioPlaybackAdapter,
    RealAudioPlaybackConfig,
)
from jarvis.presence.models import SpeechChunk


class StubAudioPlaybackBackend:
    def __init__(self) -> None:
        self._active = False
        self.calls = 0
        self.stop_calls = 0
        self.last_audio_data: bytes | None = None
        self.last_sample_rate: int | None = None
        self.last_channels: int | None = None

    @property
    def active(self) -> bool:
        return self._active

    def play_pcm(
        self,
        *,
        audio_data: bytes,
        sample_rate: int,
        channels: int,
    ) -> None:
        self.calls += 1
        self._active = True
        self.last_audio_data = audio_data
        self.last_sample_rate = sample_rate
        self.last_channels = channels

    def stop(self) -> None:
        self.stop_calls += 1
        self._active = False


class FailingAudioPlaybackBackend(StubAudioPlaybackBackend):
    def play_pcm(
        self,
        *,
        audio_data: bytes,
        sample_rate: int,
        channels: int,
    ) -> None:
        raise RuntimeError("speaker unavailable")


class FailingStopAudioPlaybackBackend(StubAudioPlaybackBackend):
    def stop(self) -> None:
        raise RuntimeError("stop failed")


def make_chunk(
    *,
    audio_data: bytes = b"\x00\x01\x02\x03",
) -> SpeechChunk:
    return SpeechChunk(
        request_id="request-1",
        chunk_id="chunk-1",
        audio_data=audio_data,
        sample_rate=16_000,
        channels=1,
        final=True,
        metadata={"source": "test"},
    )


def test_real_playback_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        RealAudioPlaybackConfig(dtype=" ").validate()


def test_real_playback_adapter_plays_chunk() -> None:
    backend = StubAudioPlaybackBackend()
    adapter = RealAudioPlaybackAdapter(backend=backend)
    chunk = make_chunk()

    result = adapter.play(chunk)

    assert result.status == PlaybackStatus.STARTED
    assert result.chunk_id == chunk.chunk_id
    assert result.request_id == chunk.request_id
    assert result.metadata["adapter"] == "real_playback"
    assert backend.calls == 1
    assert backend.last_audio_data == chunk.audio_data
    assert backend.last_sample_rate == 16_000
    assert backend.last_channels == 1
    assert adapter.is_playing is True
    assert adapter.play_count == 1
    assert adapter.last_chunk_id == "chunk-1"
    assert adapter.last_error is None


def test_speech_chunk_rejects_empty_audio_before_playback() -> None:
    with pytest.raises(ValueError, match="speech chunk audio_data cannot be empty"):
        make_chunk(audio_data=b"")


def test_empty_audio_never_reaches_playback_backend() -> None:
    backend = StubAudioPlaybackBackend()

    with pytest.raises(ValueError, match="speech chunk audio_data cannot be empty"):
        make_chunk(audio_data=b"")

    assert backend.calls == 0
    assert backend.active is False


def test_real_playback_adapter_records_play_failure() -> None:
    backend = FailingAudioPlaybackBackend()
    adapter = RealAudioPlaybackAdapter(backend=backend)
    chunk = make_chunk()

    result = adapter.play(chunk)

    assert result.status == PlaybackStatus.FAILED
    assert result.error == "RuntimeError: speaker unavailable"
    assert result.metadata["adapter"] == "real_playback"
    assert adapter.last_error == "RuntimeError: speaker unavailable"
    assert adapter.play_count == 0


def test_real_playback_adapter_stops_backend() -> None:
    backend = StubAudioPlaybackBackend()
    adapter = RealAudioPlaybackAdapter(backend=backend)

    result = adapter.play(make_chunk())
    assert result.status == PlaybackStatus.STARTED
    assert adapter.is_playing is True

    adapter.stop()

    assert backend.stop_calls == 1
    assert adapter.is_playing is False
    assert adapter.stop_count == 1


def test_real_playback_adapter_records_stop_failure() -> None:
    backend = FailingStopAudioPlaybackBackend()
    adapter = RealAudioPlaybackAdapter(backend=backend)

    with pytest.raises(RuntimeError):
        adapter.stop()

    assert adapter.last_error == "RuntimeError: stop failed"


def test_real_playback_adapter_reset_clears_counters() -> None:
    backend = StubAudioPlaybackBackend()
    adapter = RealAudioPlaybackAdapter(backend=backend)

    adapter.play(make_chunk())

    adapter.reset()

    assert adapter.play_count == 0
    assert adapter.stop_count == 0
    assert adapter.last_chunk_id is None
    assert adapter.last_error is None


def test_audio_playback_backend_protocol_accepts_stub() -> None:
    backend: AudioPlaybackBackend = StubAudioPlaybackBackend()

    backend.play_pcm(
        audio_data=b"\x00\x01",
        sample_rate=16_000,
        channels=1,
    )

    assert backend.active is True