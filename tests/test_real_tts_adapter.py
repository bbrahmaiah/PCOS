from __future__ import annotations

import pytest

from jarvis.presence.adapters import (
    RealTextToSpeechAdapter,
    RealTextToSpeechConfig,
    TextToSpeechBackend,
    tts_pcm_to_wav_bytes,
    wav_bytes_to_pcm,
)


class StubTextToSpeechBackend:
    def __init__(
        self,
        *,
        audio_data: bytes = b"\x00\x01\x02\x03",
        sample_rate: int = 16_000,
        channels: int = 1,
    ) -> None:
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.channels = channels
        self.calls = 0
        self.last_text: str | None = None
        self.last_voice_id: str | None = None
        self.last_rate: int | None = None
        self.last_volume: float | None = None

    def synthesize_to_wav(
        self,
        *,
        text: str,
        voice_id: str,
        rate: int,
        volume: float,
    ) -> bytes:
        self.calls += 1
        self.last_text = text
        self.last_voice_id = voice_id
        self.last_rate = rate
        self.last_volume = volume

        return tts_pcm_to_wav_bytes(
            audio_data=self.audio_data,
            sample_rate=self.sample_rate,
            channels=self.channels,
        )


class FailingTextToSpeechBackend:
    def synthesize_to_wav(
        self,
        *,
        text: str,
        voice_id: str,
        rate: int,
        volume: float,
    ) -> bytes:
        raise RuntimeError("tts unavailable")


def test_real_tts_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        RealTextToSpeechConfig(voice_id=" ").validate()

    with pytest.raises(ValueError):
        RealTextToSpeechConfig(rate=0).validate()

    with pytest.raises(ValueError):
        RealTextToSpeechConfig(volume=-0.1).validate()

    with pytest.raises(ValueError):
        RealTextToSpeechConfig(volume=1.1).validate()

    with pytest.raises(ValueError):
        RealTextToSpeechConfig(sample_rate_fallback=0).validate()

    with pytest.raises(ValueError):
        RealTextToSpeechConfig(channels_fallback=0).validate()

    with pytest.raises(ValueError):
        RealTextToSpeechConfig(max_text_chars=0).validate()


def test_real_tts_adapter_synthesizes_speech_chunk() -> None:
    backend = StubTextToSpeechBackend()
    adapter = RealTextToSpeechAdapter(backend=backend)

    chunks = adapter.synthesize(
        text="Hello sir.",
        request_id="request-1",
        voice_id="jarvis-voice",
        metadata={"source": "test"},
    )

    assert len(chunks) == 1

    chunk = chunks[0]

    assert chunk.request_id == "request-1"
    assert chunk.audio_data == b"\x00\x01\x02\x03"
    assert chunk.sample_rate == 16_000
    assert chunk.channels == 1
    assert chunk.final is True
    assert chunk.metadata["adapter"] == "real_tts"
    assert chunk.metadata["source"] == "test"
    assert chunk.metadata["text"] == "Hello sir."
    assert chunk.metadata["voice_id"] == "jarvis-voice"
    assert backend.calls == 1
    assert backend.last_text == "Hello sir."
    assert backend.last_voice_id == "jarvis-voice"
    assert adapter.synthesis_count == 1
    assert adapter.last_text == "Hello sir."


def test_real_tts_adapter_uses_default_voice() -> None:
    backend = StubTextToSpeechBackend()
    config = RealTextToSpeechConfig(voice_id="default-voice")
    adapter = RealTextToSpeechAdapter(config=config, backend=backend)

    chunks = adapter.synthesize(
        text="System online.",
        request_id="request-1",
    )

    assert len(chunks) == 1
    assert backend.last_voice_id == "default-voice"
    assert chunks[0].metadata["voice_id"] == "default-voice"


def test_real_tts_adapter_returns_empty_tuple_for_empty_text() -> None:
    backend = StubTextToSpeechBackend()
    adapter = RealTextToSpeechAdapter(backend=backend)

    chunks = adapter.synthesize(
        text="   ",
        request_id="request-1",
    )

    assert chunks == ()
    assert backend.calls == 0
    assert adapter.empty_requests == 1


def test_real_tts_adapter_rejects_too_long_text() -> None:
    backend = StubTextToSpeechBackend()
    config = RealTextToSpeechConfig(max_text_chars=5)
    adapter = RealTextToSpeechAdapter(config=config, backend=backend)

    with pytest.raises(ValueError):
        adapter.synthesize(
            text="too long",
            request_id="request-1",
        )


def test_real_tts_adapter_records_backend_failure() -> None:
    adapter = RealTextToSpeechAdapter(backend=FailingTextToSpeechBackend())

    with pytest.raises(RuntimeError):
        adapter.synthesize(
            text="Hello.",
            request_id="request-1",
        )

    assert adapter.last_error == "RuntimeError: tts unavailable"


def test_real_tts_adapter_reset_clears_counters() -> None:
    backend = StubTextToSpeechBackend()
    adapter = RealTextToSpeechAdapter(backend=backend)

    assert adapter.synthesize(text="Hello.", request_id="request-1")

    adapter.reset()

    assert adapter.synthesis_count == 0
    assert adapter.empty_requests == 0
    assert adapter.last_text is None
    assert adapter.last_error is None


def test_tts_pcm_to_wav_and_wav_to_pcm_round_trip() -> None:
    wav_bytes = tts_pcm_to_wav_bytes(
        audio_data=b"\x00\x01\x02\x03",
        sample_rate=16_000,
        channels=1,
    )

    pcm, sample_rate, channels = wav_bytes_to_pcm(wav_bytes)

    assert pcm == b"\x00\x01\x02\x03"
    assert sample_rate == 16_000
    assert channels == 1


def test_wav_bytes_to_pcm_rejects_empty_bytes() -> None:
    with pytest.raises(ValueError):
        wav_bytes_to_pcm(b"")


def test_text_to_speech_backend_protocol_accepts_stub() -> None:
    backend: TextToSpeechBackend = StubTextToSpeechBackend()

    wav_bytes = backend.synthesize_to_wav(
        text="Hello.",
        voice_id="default",
        rate=180,
        volume=1.0,
    )

    assert wav_bytes.startswith(b"RIFF")