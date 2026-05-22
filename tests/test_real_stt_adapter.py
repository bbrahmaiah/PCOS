from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from jarvis.presence.adapters import (
    RealSpeechToTextAdapter,
    RealSpeechToTextConfig,
    SpeechToTextBackend,
    log_probability_to_confidence,
    pcm_to_wav_bytes,
)
from jarvis.presence.models.transcript import TranscriptKind


@dataclass(slots=True)
class StubSpeechAudioSegment:
    segment_id: str = "segment-1"
    audio_data: bytes = b"\x00\x01\x02\x03"
    sample_rate: int = 16_000
    channels: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


class StubSpeechToTextBackend:
    def __init__(
        self,
        *,
        text: str = "hello jarvis",
        confidence: float = 0.95,
        language: str | None = "en",
    ) -> None:
        self.text = text
        self.confidence = confidence
        self.language = language
        self.calls = 0
        self.last_audio_data: bytes | None = None
        self.last_sample_rate: int | None = None
        self.last_channels: int | None = None
        self.last_language: str | None = None
        self.last_prompt: str | None = None

    def transcribe_pcm(
        self,
        *,
        audio_data: bytes,
        sample_rate: int,
        channels: int,
        language: str | None,
        prompt: str | None,
    ) -> tuple[str, float, str | None, dict[str, Any]]:
        self.calls += 1
        self.last_audio_data = audio_data
        self.last_sample_rate = sample_rate
        self.last_channels = channels
        self.last_language = language
        self.last_prompt = prompt

        return (
            self.text,
            self.confidence,
            self.language,
            {"backend": "stub"},
        )


class FailingSpeechToTextBackend:
    def transcribe_pcm(
        self,
        *,
        audio_data: bytes,
        sample_rate: int,
        channels: int,
        language: str | None,
        prompt: str | None,
    ) -> tuple[str, float, str | None, dict[str, Any]]:
        raise RuntimeError("stt unavailable")


def make_segment(
    *,
    audio_data: bytes = b"\x00\x01\x02\x03",
    metadata: dict[str, Any] | None = None,
) -> StubSpeechAudioSegment:
    return StubSpeechAudioSegment(
        audio_data=audio_data,
        metadata=metadata or {},
    )


def test_real_stt_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        RealSpeechToTextConfig(model_size=" ").validate()

    with pytest.raises(ValueError):
        RealSpeechToTextConfig(device=" ").validate()

    with pytest.raises(ValueError):
        RealSpeechToTextConfig(compute_type=" ").validate()

    with pytest.raises(ValueError):
        RealSpeechToTextConfig(beam_size=0).validate()

    with pytest.raises(ValueError):
        RealSpeechToTextConfig(temperature=-1).validate()

    with pytest.raises(ValueError):
        RealSpeechToTextConfig(min_text_length=-1).validate()


def test_real_stt_adapter_transcribes_segment() -> None:
    backend = StubSpeechToTextBackend(text="hello jarvis", confidence=0.91)
    adapter = RealSpeechToTextAdapter(backend=backend)
    segment = make_segment()

    transcript = adapter.transcribe(segment)

    assert transcript is not None
    assert transcript.segment_id == segment.segment_id
    assert transcript.text == "hello jarvis"
    assert transcript.kind == TranscriptKind.FINAL
    assert transcript.confidence == 0.91
    assert transcript.language == "en"
    assert transcript.metadata["adapter"] == "real_stt"
    assert transcript.metadata["backend"] == "stub"
    assert backend.calls == 1
    assert backend.last_sample_rate == 16_000
    assert backend.last_channels == 1
    assert adapter.transcription_count == 1
    assert adapter.last_text == "hello jarvis"


def test_real_stt_adapter_passes_prompt_from_segment_metadata() -> None:
    backend = StubSpeechToTextBackend()
    adapter = RealSpeechToTextAdapter(backend=backend)
    segment = make_segment(metadata={"prompt": "Jarvis command context"})

    transcript = adapter.transcribe(segment)

    assert transcript is not None
    assert backend.last_prompt == "Jarvis command context"


def test_real_stt_adapter_returns_none_for_empty_audio() -> None:
    backend = StubSpeechToTextBackend()
    adapter = RealSpeechToTextAdapter(backend=backend)
    segment = make_segment(audio_data=b"")

    transcript = adapter.transcribe(segment)

    assert transcript is None
    assert backend.calls == 0
    assert adapter.empty_transcriptions == 1


def test_real_stt_adapter_returns_none_for_empty_text() -> None:
    backend = StubSpeechToTextBackend(text="   ")
    adapter = RealSpeechToTextAdapter(backend=backend)
    segment = make_segment()

    transcript = adapter.transcribe(segment)

    assert transcript is None
    assert backend.calls == 1
    assert adapter.empty_transcriptions == 1


def test_real_stt_adapter_records_backend_failure() -> None:
    adapter = RealSpeechToTextAdapter(backend=FailingSpeechToTextBackend())
    segment = make_segment()

    with pytest.raises(RuntimeError):
        adapter.transcribe(segment)

    assert adapter.last_error == "RuntimeError: stt unavailable"


def test_real_stt_adapter_reset_clears_counters() -> None:
    backend = StubSpeechToTextBackend()
    adapter = RealSpeechToTextAdapter(backend=backend)

    assert adapter.transcribe(make_segment()) is not None

    adapter.reset()

    assert adapter.transcription_count == 0
    assert adapter.empty_transcriptions == 0
    assert adapter.last_text is None
    assert adapter.last_confidence is None
    assert adapter.last_error is None


def test_pcm_to_wav_bytes_creates_wav_container() -> None:
    wav_data = pcm_to_wav_bytes(
        audio_data=b"\x00\x01\x02\x03",
        sample_rate=16_000,
        channels=1,
    )

    assert wav_data.startswith(b"RIFF")
    assert b"WAVE" in wav_data


def test_pcm_to_wav_bytes_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        pcm_to_wav_bytes(
            audio_data=b"\x00\x01",
            sample_rate=0,
            channels=1,
        )

    with pytest.raises(ValueError):
        pcm_to_wav_bytes(
            audio_data=b"\x00\x01",
            sample_rate=16_000,
            channels=0,
        )

    with pytest.raises(ValueError):
        pcm_to_wav_bytes(
            audio_data=b"\x00\x01",
            sample_rate=16_000,
            channels=1,
            sample_width_bytes=0,
        )


def test_log_probability_to_confidence_bounds_output() -> None:
    assert log_probability_to_confidence(0.0) == 1.0
    assert 0.0 < log_probability_to_confidence(-1.0) < 1.0
    assert log_probability_to_confidence(-9999.0) == 0.0


def test_backend_protocol_accepts_stub() -> None:
    backend: SpeechToTextBackend = StubSpeechToTextBackend()

    result = backend.transcribe_pcm(
        audio_data=b"\x00\x01",
        sample_rate=16_000,
        channels=1,
        language="en",
        prompt=None,
    )

    assert result[0] == "hello jarvis"