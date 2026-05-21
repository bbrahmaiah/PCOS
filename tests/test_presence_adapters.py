from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.presence.adapters import (
    AudioPlaybackAdapter,
    MicrophoneAdapter,
    MicrophoneDevice,
    PlaybackResult,
    PlaybackStatus,
    SpeechToTextAdapter,
    TextToSpeechAdapter,
    VoiceActivityAdapter,
    WakeWordAdapter,
    WakeWordDetection,
)
from jarvis.presence.models import (
    AudioFrame,
    SpeechChunk,
    SpeechRequest,
    Transcript,
    TranscriptKind,
    VoiceActivity,
    VoiceActivityState,
)


class FakeMicrophoneAdapter(MicrophoneAdapter):
    def __init__(self) -> None:
        self._running = False
        self._frames = [
            AudioFrame(
                source="fake_microphone",
                audio_data=b"\x00\x01",
                frame_index=0,
            )
        ]

    @property
    def is_running(self) -> bool:
        return self._running

    def list_devices(self) -> tuple[MicrophoneDevice, ...]:
        return (
            MicrophoneDevice(
                device_id="fake-device",
                name="Fake Microphone",
                is_default=True,
            ),
        )

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def read_frame(self) -> AudioFrame | None:
        if not self._running or not self._frames:
            return None

        return self._frames.pop(0)


class FakeWakeWordAdapter(WakeWordAdapter):
    def __init__(self, *, should_detect: bool = True) -> None:
        self.should_detect = should_detect
        self.reset_called = False

    def detect(self, frame: AudioFrame) -> WakeWordDetection | None:
        if not self.should_detect:
            return None

        return WakeWordDetection(
            frame_id=frame.frame_id,
            wake_word="jarvis",
            confidence=0.99,
        )

    def reset(self) -> None:
        self.reset_called = True


class FakeVoiceActivityAdapter(VoiceActivityAdapter):
    def __init__(self) -> None:
        self.reset_called = False

    def analyze(self, frame: AudioFrame) -> VoiceActivity:
        return VoiceActivity(
            frame_id=frame.frame_id,
            state=VoiceActivityState.SPEECH_STARTED,
            is_speech=True,
            confidence=0.95,
            energy=0.8,
        )

    def reset(self) -> None:
        self.reset_called = True


class FakeSpeechToTextAdapter(SpeechToTextAdapter):
    def __init__(self) -> None:
        self.reset_called = False

    def transcribe(self, frames: tuple[AudioFrame, ...]) -> Transcript:
        if not frames:
            raise ValueError("frames cannot be empty.")

        return Transcript(
            segment_id="segment-1",
            text="hello jarvis",
            kind=TranscriptKind.FINAL,
            confidence=0.98,
        )

    def reset(self) -> None:
        self.reset_called = True


class FakeTextToSpeechAdapter(TextToSpeechAdapter):
    def __init__(self) -> None:
        self.reset_called = False

    def synthesize(self, request: SpeechRequest) -> tuple[SpeechChunk, ...]:
        return (
            SpeechChunk(
                request_id=request.request_id,
                audio_data=b"\x00\x01",
                final=True,
            ),
        )

    def reset(self) -> None:
        self.reset_called = True


class FakeAudioPlaybackAdapter(AudioPlaybackAdapter):
    def __init__(self) -> None:
        self._playing = False
        self._last_chunk: SpeechChunk | None = None

    @property
    def is_playing(self) -> bool:
        return self._playing

    def play(self, chunk: SpeechChunk) -> PlaybackResult:
        self._playing = True
        self._last_chunk = chunk

        return PlaybackResult(
            chunk_id=chunk.chunk_id,
            request_id=chunk.request_id,
            status=PlaybackStatus.STARTED,
        )

    def stop(self, *, request_id: str | None = None) -> PlaybackResult | None:
        if not self._playing or self._last_chunk is None:
            return None

        chunk = self._last_chunk
        self._playing = False
        self._last_chunk = None

        return PlaybackResult(
            chunk_id=chunk.chunk_id,
            request_id=request_id or chunk.request_id,
            status=PlaybackStatus.STOPPED,
        )


def test_microphone_adapter_contract() -> None:
    adapter = FakeMicrophoneAdapter()

    assert adapter.is_running is False
    assert adapter.list_devices()[0].is_default is True

    adapter.start()

    frame = adapter.read_frame()

    assert adapter.is_running is True
    assert frame is not None
    assert frame.source == "fake_microphone"

    adapter.stop()

    assert adapter.is_running is False
    assert adapter.read_frame() is None


def test_microphone_device_rejects_empty_name() -> None:
    with pytest.raises(ValidationError):
        MicrophoneDevice(device_id="device-1", name="   ")


def test_wake_word_adapter_contract() -> None:
    frame = AudioFrame(source="fake_microphone", audio_data=b"\x00\x01")
    adapter = FakeWakeWordAdapter()

    detection = adapter.detect(frame)

    assert detection is not None
    assert detection.frame_id == frame.frame_id
    assert detection.wake_word == "jarvis"
    assert detection.confidence == 0.99

    adapter.reset()

    assert adapter.reset_called is True


def test_wake_word_adapter_can_return_none() -> None:
    frame = AudioFrame(source="fake_microphone", audio_data=b"\x00\x01")
    adapter = FakeWakeWordAdapter(should_detect=False)

    assert adapter.detect(frame) is None


def test_wake_word_detection_rejects_invalid_confidence() -> None:
    with pytest.raises(ValidationError):
        WakeWordDetection(
            frame_id="frame-1",
            wake_word="jarvis",
            confidence=2.0,
        )


def test_voice_activity_adapter_contract() -> None:
    frame = AudioFrame(source="fake_microphone", audio_data=b"\x00\x01")
    adapter = FakeVoiceActivityAdapter()

    activity = adapter.analyze(frame)

    assert activity.frame_id == frame.frame_id
    assert activity.is_speech is True
    assert activity.state == VoiceActivityState.SPEECH_STARTED

    adapter.reset()

    assert adapter.reset_called is True


def test_speech_to_text_adapter_contract() -> None:
    frame = AudioFrame(source="fake_microphone", audio_data=b"\x00\x01")
    adapter = FakeSpeechToTextAdapter()

    transcript = adapter.transcribe((frame,))

    assert transcript.text == "hello jarvis"
    assert transcript.is_final is True

    adapter.reset()

    assert adapter.reset_called is True


def test_speech_to_text_adapter_rejects_empty_frames() -> None:
    adapter = FakeSpeechToTextAdapter()

    with pytest.raises(ValueError):
        adapter.transcribe(())


def test_text_to_speech_adapter_contract() -> None:
    adapter = FakeTextToSpeechAdapter()
    request = SpeechRequest(text="Yes sir.")

    chunks = adapter.synthesize(request)

    assert len(chunks) == 1
    assert chunks[0].request_id == request.request_id
    assert chunks[0].final is True

    adapter.reset()

    assert adapter.reset_called is True


def test_audio_playback_adapter_contract() -> None:
    adapter = FakeAudioPlaybackAdapter()
    chunk = SpeechChunk(
        request_id="request-1",
        audio_data=b"\x00\x01",
    )

    started = adapter.play(chunk)

    assert adapter.is_playing is True
    assert started.status == PlaybackStatus.STARTED

    stopped = adapter.stop()

    assert stopped is not None
    assert stopped.status == PlaybackStatus.STOPPED
    assert adapter.is_playing is False


def test_audio_playback_stop_returns_none_when_idle() -> None:
    adapter = FakeAudioPlaybackAdapter()

    assert adapter.stop() is None


def test_playback_result_rejects_empty_chunk_id() -> None:
    with pytest.raises(ValidationError):
        PlaybackResult(
            chunk_id="   ",
            request_id="request-1",
            status=PlaybackStatus.STARTED,
        )