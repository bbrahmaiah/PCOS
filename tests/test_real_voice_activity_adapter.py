from __future__ import annotations

import pytest

from jarvis.presence.adapters import (
    EnergyVoiceActivityAdapter,
    EnergyVoiceActivityConfig,
    EnergyWakeWordAdapter,
    EnergyWakeWordConfig,
    extract_int16_audio_features,
)
from jarvis.presence.models import AudioFrame, VoiceActivityState


def pcm16(samples: tuple[int, ...]) -> bytes:
    return b"".join(
        sample.to_bytes(2, byteorder="little", signed=True)
        for sample in samples
    )


def make_frame(samples: tuple[int, ...], frame_index: int = 0) -> AudioFrame:
    return AudioFrame(
        audio_data=pcm16(samples),
        sample_rate=16_000,
        channels=1,
        frame_index=frame_index,
        source="test",
    )


def speech_samples(amplitude: int = 2000, count: int = 160) -> tuple[int, ...]:
    return tuple(
        amplitude if index % 2 == 0 else -amplitude
        for index in range(count)
    )


def silence_samples(count: int = 160) -> tuple[int, ...]:
    return tuple(0 for _ in range(count))


def test_extract_int16_audio_features_for_silence() -> None:
    features = extract_int16_audio_features(pcm16(silence_samples()))

    assert features.rms == 0
    assert features.peak == 0
    assert features.sample_count == 160


def test_extract_int16_audio_features_for_speech_like_audio() -> None:
    features = extract_int16_audio_features(pcm16(speech_samples()))

    assert features.rms > 1000
    assert features.peak == 2000
    assert features.zero_crossing_rate > 0.5


def test_energy_vad_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        EnergyVoiceActivityConfig(sample_width_bytes=1).validate()

    with pytest.raises(ValueError):
        EnergyVoiceActivityConfig(speech_rms_threshold=0).validate()

    with pytest.raises(ValueError):
        EnergyVoiceActivityConfig(speech_start_frames=0).validate()

    with pytest.raises(ValueError):
        EnergyVoiceActivityConfig(speech_end_frames=0).validate()

    with pytest.raises(ValueError):
        EnergyVoiceActivityConfig(min_zero_crossing_rate=0.8).validate()


def test_energy_vad_detects_speech_start_continue_end() -> None:
    adapter = EnergyVoiceActivityAdapter(
        config=EnergyVoiceActivityConfig(
            speech_rms_threshold=300,
            silence_rms_threshold=100,
            speech_start_frames=2,
            speech_end_frames=2,
            min_zero_crossing_rate=0.0,
            max_zero_crossing_rate=1.0,
            adaptive_noise_floor=False,
        )
    )

    first = adapter.detect(make_frame(speech_samples(), frame_index=0))
    second = adapter.detect(make_frame(speech_samples(), frame_index=1))
    third = adapter.detect(make_frame(speech_samples(), frame_index=2))
    fourth = adapter.detect(make_frame(silence_samples(), frame_index=3))
    fifth = adapter.detect(make_frame(silence_samples(), frame_index=4))

    assert first.state == VoiceActivityState.SILENCE
    assert first.is_speech is False
    assert second.state == VoiceActivityState.SPEECH_STARTED
    assert second.is_speech is True
    assert third.state == VoiceActivityState.SPEECH_CONTINUING
    assert third.is_speech is True
    assert fourth.state == VoiceActivityState.SPEECH_CONTINUING
    assert fourth.is_speech is True
    assert fifth.state == VoiceActivityState.SPEECH_ENDED
    assert fifth.is_speech is False


def test_energy_vad_reset_clears_state() -> None:
    adapter = EnergyVoiceActivityAdapter(
        config=EnergyVoiceActivityConfig(
            speech_rms_threshold=300,
            speech_start_frames=1,
            min_zero_crossing_rate=0.0,
            max_zero_crossing_rate=1.0,
            adaptive_noise_floor=False,
        )
    )

    activity = adapter.detect(make_frame(speech_samples()))
    assert activity.state == VoiceActivityState.SPEECH_STARTED

    adapter.reset()

    next_activity = adapter.detect(make_frame(speech_samples()))
    assert next_activity.state == VoiceActivityState.SPEECH_STARTED


def test_energy_wake_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        EnergyWakeWordConfig(rms_threshold=0).validate()

    with pytest.raises(ValueError):
        EnergyWakeWordConfig(required_consecutive_frames=0).validate()

    with pytest.raises(ValueError):
        EnergyWakeWordConfig(cooldown_frames=-1).validate()

    with pytest.raises(ValueError):
        EnergyWakeWordConfig(wake_word=" ").validate()


def test_energy_wake_detects_after_required_frames() -> None:
    adapter = EnergyWakeWordAdapter(
        config=EnergyWakeWordConfig(
            rms_threshold=300,
            required_consecutive_frames=2,
            cooldown_frames=2,
            min_zero_crossing_rate=0.0,
            max_zero_crossing_rate=1.0,
            wake_word="jarvis",
        )
    )

    first = adapter.detect(make_frame(speech_samples(), frame_index=0))
    second = adapter.detect(make_frame(speech_samples(), frame_index=1))

    assert first is None
    assert second is not None
    assert second.wake_word == "jarvis"
    assert second.confidence > 0


def test_energy_wake_respects_cooldown() -> None:
    adapter = EnergyWakeWordAdapter(
        config=EnergyWakeWordConfig(
            rms_threshold=300,
            required_consecutive_frames=1,
            cooldown_frames=2,
            min_zero_crossing_rate=0.0,
            max_zero_crossing_rate=1.0,
        )
    )

    first = adapter.detect(make_frame(speech_samples(), frame_index=0))
    second = adapter.detect(make_frame(speech_samples(), frame_index=1))
    third = adapter.detect(make_frame(speech_samples(), frame_index=2))
    fourth = adapter.detect(make_frame(speech_samples(), frame_index=3))

    assert first is not None
    assert second is None
    assert third is None
    assert fourth is not None


def test_energy_wake_reset_clears_counters() -> None:
    adapter = EnergyWakeWordAdapter(
        config=EnergyWakeWordConfig(
            rms_threshold=300,
            required_consecutive_frames=1,
            cooldown_frames=10,
            min_zero_crossing_rate=0.0,
            max_zero_crossing_rate=1.0,
        )
    )

    first = adapter.detect(make_frame(speech_samples()))
    assert first is not None

    adapter.reset()

    second = adapter.detect(make_frame(speech_samples()))
    assert second is not None