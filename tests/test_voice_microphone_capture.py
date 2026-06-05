from __future__ import annotations

from dataclasses import dataclass

import pytest

from jarvis.voice import (
    VoiceDeviceHealth,
    VoiceMicrophoneCapturePolicy,
    VoiceMicrophoneCaptureRuntime,
    VoiceMicrophoneCaptureStatus,
    VoiceMicrophoneDeviceInfo,
    VoiceRuntimeConfig,
)


@dataclass
class FakeMicrophoneAdapter:
    prepared: bool = False
    started: bool = False
    stopped: bool = False
    fail_prepare: bool = False
    fail_start: bool = False
    fail_read: bool = False
    read_count: int = 0

    def prepare(self, config: VoiceRuntimeConfig) -> VoiceMicrophoneDeviceInfo:
        if self.fail_prepare:
            raise RuntimeError("prepare failed")
        self.prepared = True
        return VoiceMicrophoneDeviceInfo(
            name="Fake Microphone",
            index=0,
            sample_rate_hz=config.sample_rate_hz,
            channels=config.channels,
            health=VoiceDeviceHealth.READY,
        )

    def start(self) -> None:
        if self.fail_start:
            raise RuntimeError("start failed")
        self.started = True

    def read_frame(self, frame_bytes: int) -> bytes:
        if self.fail_read:
            raise RuntimeError("read failed")
        self.read_count += 1
        return b"\x00\x01" * frame_bytes

    def stop(self) -> None:
        self.stopped = True


def test_microphone_capture_policy_validation() -> None:
    with pytest.raises(ValueError):
        VoiceMicrophoneCapturePolicy(max_consecutive_failures=0)


def test_microphone_capture_prepares_device() -> None:
    adapter = FakeMicrophoneAdapter()
    runtime = VoiceMicrophoneCaptureRuntime(adapter=adapter)

    result = runtime.prepare()

    assert result.status == VoiceMicrophoneCaptureStatus.READY
    assert result.device is not None
    assert result.device.name == "Fake Microphone"
    assert adapter.prepared is True


def test_microphone_capture_start_prepares_if_needed() -> None:
    adapter = FakeMicrophoneAdapter()
    runtime = VoiceMicrophoneCaptureRuntime(adapter=adapter)

    result = runtime.start()

    assert result.status == VoiceMicrophoneCaptureStatus.CAPTURING
    assert adapter.prepared is True
    assert adapter.started is True


def test_microphone_capture_once_returns_voice_frame() -> None:
    adapter = FakeMicrophoneAdapter()
    runtime = VoiceMicrophoneCaptureRuntime(adapter=adapter)

    runtime.start()
    result = runtime.capture_once()

    assert result.status == VoiceMicrophoneCaptureStatus.CAPTURING
    assert result.frame is not None
    assert result.frame.sample_rate_hz == 16_000
    assert result.frame.channels == 1
    assert result.frame.duration_ms == 20
    assert result.frame.data


def test_microphone_capture_blocks_when_not_started() -> None:
    runtime = VoiceMicrophoneCaptureRuntime(adapter=FakeMicrophoneAdapter())

    result = runtime.capture_once()

    assert result.status == VoiceMicrophoneCaptureStatus.DEGRADED
    assert result.frame is None


def test_microphone_capture_tracks_snapshot_counts() -> None:
    adapter = FakeMicrophoneAdapter()
    runtime = VoiceMicrophoneCaptureRuntime(adapter=adapter)

    runtime.start()
    runtime.capture_once()
    runtime.capture_once()
    snapshot = runtime.snapshot()

    assert snapshot.status == VoiceMicrophoneCaptureStatus.CAPTURING
    assert snapshot.captured_frames == 2
    assert snapshot.captured_bytes > 0


def test_microphone_capture_failure_becomes_failed() -> None:
    adapter = FakeMicrophoneAdapter(fail_read=True)
    runtime = VoiceMicrophoneCaptureRuntime(
        adapter=adapter,
        policy=VoiceMicrophoneCapturePolicy(max_consecutive_failures=2),
    )

    runtime.start()
    first = runtime.capture_once()
    second = runtime.capture_once()

    assert first.status == VoiceMicrophoneCaptureStatus.DEGRADED
    assert second.status == VoiceMicrophoneCaptureStatus.FAILED


def test_microphone_capture_stop() -> None:
    adapter = FakeMicrophoneAdapter()
    runtime = VoiceMicrophoneCaptureRuntime(adapter=adapter)

    runtime.start()
    result = runtime.stop()

    assert result.status == VoiceMicrophoneCaptureStatus.STOPPED
    assert adapter.stopped is True


def test_microphone_device_info_validation() -> None:
    with pytest.raises(ValueError):
        VoiceMicrophoneDeviceInfo(
            name=" ",
            index=0,
            sample_rate_hz=16_000,
            channels=1,
            health=VoiceDeviceHealth.READY,
        )


def test_microphone_capture_enum_values_are_stable() -> None:
    assert VoiceMicrophoneCaptureStatus.CAPTURING.value == "capturing"