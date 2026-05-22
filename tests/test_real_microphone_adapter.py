from __future__ import annotations

import pytest

from jarvis.presence.adapters import RealMicrophoneAdapter, RealMicrophoneConfig
from jarvis.presence.adapters.real_microphone_adapter import AudioCallback


class StubMicrophoneBackend:
    def __init__(self) -> None:
        self.active = False
        self.closed = False
        self.callback: AudioCallback | None = None
        self.start_count = 0
        self.stop_count = 0
        self.close_count = 0

    def start(self, callback: AudioCallback) -> None:
        self.callback = callback
        self.active = True
        self.start_count += 1

    def stop(self) -> None:
        self.active = False
        self.stop_count += 1

    def close(self) -> None:
        self.closed = True
        self.close_count += 1

    def push(self, audio_data: bytes) -> None:
        assert self.callback is not None
        self.callback(audio_data)


class FailingStartBackend(StubMicrophoneBackend):
    def start(self, callback: AudioCallback) -> None:
        raise RuntimeError("microphone unavailable")


def test_real_microphone_config_defaults() -> None:
    config = RealMicrophoneConfig()

    assert config.sample_rate == 16_000
    assert config.channels == 1
    assert config.frame_duration_ms == 30
    assert config.samples_per_frame == 480
    assert config.expected_frame_bytes == 960


def test_real_microphone_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        RealMicrophoneConfig(sample_rate=0).validate()

    with pytest.raises(ValueError):
        RealMicrophoneConfig(channels=0).validate()

    with pytest.raises(ValueError):
        RealMicrophoneConfig(frame_duration_ms=0).validate()

    with pytest.raises(ValueError):
        RealMicrophoneConfig(queue_max_frames=0).validate()

    with pytest.raises(ValueError):
        RealMicrophoneConfig(dtype="bad").validate()

    with pytest.raises(ValueError):
        RealMicrophoneConfig(overflow_policy="bad").validate()


def test_real_microphone_adapter_start_stop_close() -> None:
    backend = StubMicrophoneBackend()
    adapter = RealMicrophoneAdapter(backend=backend)

    adapter.start()

    assert adapter.started is True
    assert adapter.is_active is True
    assert backend.start_count == 1

    adapter.stop()

    assert adapter.started is False
    assert adapter.is_active is False
    assert backend.stop_count == 1

    adapter.close()

    assert backend.closed is True
    assert backend.close_count == 1


def test_real_microphone_adapter_reads_audio_frames() -> None:
    backend = StubMicrophoneBackend()
    adapter = RealMicrophoneAdapter(backend=backend)

    adapter.start()
    backend.push(b"\x00\x01")
    backend.push(b"\x02\x03")

    frame_0 = adapter.read_frame()
    frame_1 = adapter.read_frame()

    assert frame_0 is not None
    assert frame_1 is not None
    assert frame_0.audio_data == b"\x00\x01"
    assert frame_1.audio_data == b"\x02\x03"
    assert frame_0.frame_index == 0
    assert frame_1.frame_index == 1
    assert frame_0.sample_rate == 16_000
    assert frame_0.channels == 1
    assert frame_0.source == "real_microphone_adapter"
    assert adapter.read_frame() is None
    assert adapter.captured_frames == 2
    assert adapter.read_frames == 2


def test_real_microphone_adapter_drops_oldest_on_overflow() -> None:
    backend = StubMicrophoneBackend()
    config = RealMicrophoneConfig(
        queue_max_frames=2,
        overflow_policy="drop_oldest",
    )
    adapter = RealMicrophoneAdapter(config=config, backend=backend)

    adapter.start()
    backend.push(b"oldest")
    backend.push(b"middle")
    backend.push(b"newest")

    first = adapter.read_frame()
    second = adapter.read_frame()

    assert first is not None
    assert second is not None
    assert first.audio_data == b"middle"
    assert second.audio_data == b"newest"
    assert adapter.dropped_frames == 1


def test_real_microphone_adapter_drops_newest_on_overflow() -> None:
    backend = StubMicrophoneBackend()
    config = RealMicrophoneConfig(
        queue_max_frames=2,
        overflow_policy="drop_newest",
    )
    adapter = RealMicrophoneAdapter(config=config, backend=backend)

    adapter.start()
    backend.push(b"oldest")
    backend.push(b"middle")
    backend.push(b"newest")

    first = adapter.read_frame()
    second = adapter.read_frame()

    assert first is not None
    assert second is not None
    assert first.audio_data == b"oldest"
    assert second.audio_data == b"middle"
    assert adapter.dropped_frames == 1


def test_real_microphone_adapter_raises_on_overflow_policy_raise() -> None:
    backend = StubMicrophoneBackend()
    config = RealMicrophoneConfig(
        queue_max_frames=1,
        overflow_policy="raise",
    )
    adapter = RealMicrophoneAdapter(config=config, backend=backend)

    adapter.start()
    backend.push(b"first")

    with pytest.raises(RuntimeError):
        backend.push(b"second")


def test_real_microphone_adapter_reset_clears_queue_and_counters() -> None:
    backend = StubMicrophoneBackend()
    adapter = RealMicrophoneAdapter(backend=backend)

    adapter.start()
    backend.push(b"\x00\x01")
    assert adapter.read_frame() is not None

    adapter.reset()

    assert adapter.pending_frames == 0
    assert adapter.captured_frames == 0
    assert adapter.read_frames == 0
    assert adapter.dropped_frames == 0
    assert adapter.read_frame() is None


def test_real_microphone_adapter_records_start_failure() -> None:
    backend = FailingStartBackend()
    adapter = RealMicrophoneAdapter(backend=backend)

    with pytest.raises(RuntimeError):
        adapter.start()

    assert adapter.last_error == "RuntimeError: microphone unavailable"


def test_real_microphone_adapter_cannot_restart_after_close() -> None:
    backend = StubMicrophoneBackend()
    adapter = RealMicrophoneAdapter(backend=backend)

    adapter.close()

    with pytest.raises(RuntimeError):
        adapter.start()