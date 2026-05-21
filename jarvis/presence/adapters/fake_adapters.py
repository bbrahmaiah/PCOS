from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from threading import RLock
from typing import Final

from jarvis.presence.adapters.microphone_adapter import (
    MicrophoneAdapter,
    MicrophoneDevice,
)
from jarvis.presence.adapters.playback_adapter import (
    AudioPlaybackAdapter,
    PlaybackResult,
    PlaybackStatus,
)
from jarvis.presence.adapters.stt_adapter import SpeechToTextAdapter
from jarvis.presence.adapters.tts_adapter import TextToSpeechAdapter
from jarvis.presence.adapters.vad_adapter import VoiceActivityAdapter
from jarvis.presence.adapters.wake_word_adapter import (
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

_DEFAULT_SAMPLE_RATE: Final[int] = 16_000
_DEFAULT_CHANNELS: Final[int] = 1
_DEFAULT_SAMPLE_WIDTH_BYTES: Final[int] = 2


def make_fake_audio_frame(
    *,
    source: str = "fake_microphone",
    frame_index: int = 0,
    audio_data: bytes = b"\x00\x01",
    sample_rate: int = _DEFAULT_SAMPLE_RATE,
    channels: int = _DEFAULT_CHANNELS,
    sample_width_bytes: int = _DEFAULT_SAMPLE_WIDTH_BYTES,
) -> AudioFrame:
    """
    Create a deterministic fake audio frame for tests and simulations.
    """

    return AudioFrame(
        source=source,
        audio_data=audio_data,
        sample_rate=sample_rate,
        channels=channels,
        sample_width_bytes=sample_width_bytes,
        frame_index=frame_index,
    )


class FakeMicrophoneAdapter(MicrophoneAdapter):
    """
    Deterministic in-memory microphone adapter.

    Design:
    - no real hardware
    - thread-safe frame queue
    - explicit start/stop behavior
    - usable by future VoiceInputWorker tests
    """

    def __init__(
        self,
        *,
        frames: Iterable[AudioFrame] | None = None,
        device: MicrophoneDevice | None = None,
    ) -> None:
        self._lock = RLock()
        self._running = False
        self._frames: deque[AudioFrame] = deque(frames or ())
        self._device = device or MicrophoneDevice(
            device_id="fake-microphone",
            name="Fake Microphone",
            sample_rate=_DEFAULT_SAMPLE_RATE,
            channels=_DEFAULT_CHANNELS,
            is_default=True,
        )

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def remaining_frames(self) -> int:
        with self._lock:
            return len(self._frames)

    def list_devices(self) -> tuple[MicrophoneDevice, ...]:
        return (self._device,)

    def start(self) -> None:
        with self._lock:
            self._running = True

    def stop(self) -> None:
        with self._lock:
            self._running = False

    def read_frame(self) -> AudioFrame | None:
        with self._lock:
            if not self._running or not self._frames:
                return None

            return self._frames.popleft()

    def push_frame(self, frame: AudioFrame) -> None:
        with self._lock:
            self._frames.append(frame)

    def push_frames(self, frames: Iterable[AudioFrame]) -> None:
        with self._lock:
            self._frames.extend(frames)

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()


class FakeWakeWordAdapter(WakeWordAdapter):
    """
    Deterministic wake-word adapter.

    detection_pattern controls each detect() call:
    - True  => return WakeWordDetection
    - False => return None

    If the pattern is exhausted, default_detect is used.
    """

    def __init__(
        self,
        *,
        wake_word: str = "jarvis",
        confidence: float = 0.99,
        detection_pattern: Iterable[bool] | None = None,
        default_detect: bool = False,
    ) -> None:
        clean_wake_word = wake_word.strip()

        if not clean_wake_word:
            raise ValueError("wake_word cannot be empty.")

        self._lock = RLock()
        self._wake_word = clean_wake_word
        self._confidence = confidence
        self._pattern: deque[bool] = deque(detection_pattern or ())
        self._default_detect = default_detect
        self._detect_calls = 0
        self._reset_count = 0

    @property
    def detect_calls(self) -> int:
        with self._lock:
            return self._detect_calls

    @property
    def reset_count(self) -> int:
        with self._lock:
            return self._reset_count

    def detect(self, frame: AudioFrame) -> WakeWordDetection | None:
        with self._lock:
            self._detect_calls += 1
            should_detect = (
                self._pattern.popleft()
                if self._pattern
                else self._default_detect
            )

        if not should_detect:
            return None

        return WakeWordDetection(
            frame_id=frame.frame_id,
            wake_word=self._wake_word,
            confidence=self._confidence,
        )

    def reset(self) -> None:
        with self._lock:
            self._reset_count += 1


class FakeVoiceActivityAdapter(VoiceActivityAdapter):
    """
    Deterministic VAD adapter.

    It emits configured VoiceActivityState values in order. When exhausted,
    default_state is used.
    """

    def __init__(
        self,
        *,
        states: Iterable[VoiceActivityState] | None = None,
        default_state: VoiceActivityState = VoiceActivityState.SILENCE,
        confidence: float = 0.95,
        energy: float = 0.5,
    ) -> None:
        self._lock = RLock()
        self._states: deque[VoiceActivityState] = deque(states or ())
        self._default_state = default_state
        self._confidence = confidence
        self._energy = energy
        self._analyze_calls = 0
        self._reset_count = 0

    @property
    def analyze_calls(self) -> int:
        with self._lock:
            return self._analyze_calls

    @property
    def reset_count(self) -> int:
        with self._lock:
            return self._reset_count

    def analyze(self, frame: AudioFrame) -> VoiceActivity:
        with self._lock:
            self._analyze_calls += 1
            state = self._states.popleft() if self._states else self._default_state

        return VoiceActivity(
            frame_id=frame.frame_id,
            state=state,
            is_speech=state != VoiceActivityState.SILENCE,
            confidence=self._confidence,
            energy=self._energy if state != VoiceActivityState.SILENCE else 0.0,
        )

    def reset(self) -> None:
        with self._lock:
            self._reset_count += 1


class FakeSpeechToTextAdapter(SpeechToTextAdapter):
    """
    Deterministic STT adapter.

    Converts any non-empty audio segment into a configured Transcript.
    """

    def __init__(
        self,
        *,
        text: str = "hello jarvis",
        kind: TranscriptKind = TranscriptKind.FINAL,
        confidence: float = 0.98,
        language: str = "en",
        segment_id: str = "fake-segment",
    ) -> None:
        self._lock = RLock()
        self._text = text
        self._kind = kind
        self._confidence = confidence
        self._language = language
        self._segment_id = segment_id
        self._transcribe_calls = 0
        self._reset_count = 0
        self._last_frame_count = 0

    @property
    def transcribe_calls(self) -> int:
        with self._lock:
            return self._transcribe_calls

    @property
    def reset_count(self) -> int:
        with self._lock:
            return self._reset_count

    @property
    def last_frame_count(self) -> int:
        with self._lock:
            return self._last_frame_count

    def transcribe(self, frames: tuple[AudioFrame, ...]) -> Transcript:
        if not frames:
            raise ValueError("frames cannot be empty.")

        with self._lock:
            self._transcribe_calls += 1
            self._last_frame_count = len(frames)

        return Transcript(
            segment_id=self._segment_id,
            text=self._text,
            kind=self._kind,
            confidence=self._confidence,
            language=self._language,
        )

    def reset(self) -> None:
        with self._lock:
            self._reset_count += 1


class FakeTextToSpeechAdapter(TextToSpeechAdapter):
    """
    Deterministic TTS adapter.

    Converts a SpeechRequest into configured fake SpeechChunk audio.
    """

    def __init__(
        self,
        *,
        chunk_audio: Iterable[bytes] | None = None,
        sample_rate: int = 24_000,
    ) -> None:
        self._lock = RLock()
        self._chunk_audio = tuple(chunk_audio or (b"\x00\x01",))
        self._sample_rate = sample_rate
        self._synthesize_calls = 0
        self._reset_count = 0
        self._last_request_id: str | None = None

    @property
    def synthesize_calls(self) -> int:
        with self._lock:
            return self._synthesize_calls

    @property
    def reset_count(self) -> int:
        with self._lock:
            return self._reset_count

    @property
    def last_request_id(self) -> str | None:
        with self._lock:
            return self._last_request_id

    def synthesize(self, request: SpeechRequest) -> tuple[SpeechChunk, ...]:
        with self._lock:
            self._synthesize_calls += 1
            self._last_request_id = request.request_id

        return tuple(
            SpeechChunk(
                request_id=request.request_id,
                audio_data=audio,
                sample_rate=self._sample_rate,
                chunk_index=index,
                final=index == len(self._chunk_audio) - 1,
            )
            for index, audio in enumerate(self._chunk_audio)
        )

    def reset(self) -> None:
        with self._lock:
            self._reset_count += 1
            self._last_request_id = None


class FakeAudioPlaybackAdapter(AudioPlaybackAdapter):
    """
    Deterministic playback adapter.

    It records played chunks and supports interruption-style stop().
    """

    def __init__(self, *, fail_playback: bool = False) -> None:
        self._lock = RLock()
        self._is_playing = False
        self._fail_playback = fail_playback
        self._current_chunk: SpeechChunk | None = None
        self._played_chunks: list[SpeechChunk] = []
        self._play_results: list[PlaybackResult] = []
        self._stop_results: list[PlaybackResult] = []

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._is_playing

    @property
    def played_chunks(self) -> tuple[SpeechChunk, ...]:
        with self._lock:
            return tuple(self._played_chunks)

    @property
    def play_results(self) -> tuple[PlaybackResult, ...]:
        with self._lock:
            return tuple(self._play_results)

    @property
    def stop_results(self) -> tuple[PlaybackResult, ...]:
        with self._lock:
            return tuple(self._stop_results)

    def play(self, chunk: SpeechChunk) -> PlaybackResult:
        with self._lock:
            if self._fail_playback:
                result = PlaybackResult(
                    chunk_id=chunk.chunk_id,
                    request_id=chunk.request_id,
                    status=PlaybackStatus.FAILED,
                    error="Fake playback failure.",
                )
                self._play_results.append(result)
                return result

            self._is_playing = True
            self._current_chunk = chunk
            self._played_chunks.append(chunk)

            result = PlaybackResult(
                chunk_id=chunk.chunk_id,
                request_id=chunk.request_id,
                status=PlaybackStatus.STARTED,
            )
            self._play_results.append(result)
            return result

    def stop(self, *, request_id: str | None = None) -> PlaybackResult | None:
        with self._lock:
            if not self._is_playing or self._current_chunk is None:
                return None

            chunk = self._current_chunk
            self._is_playing = False
            self._current_chunk = None

            result = PlaybackResult(
                chunk_id=chunk.chunk_id,
                request_id=request_id or chunk.request_id,
                status=PlaybackStatus.STOPPED,
            )
            self._stop_results.append(result)
            return result

    def complete_current(self) -> PlaybackResult | None:
        """
        Mark the current chunk as completed.

        This is useful for deterministic playback tests without sleeping.
        """

        with self._lock:
            if not self._is_playing or self._current_chunk is None:
                return None

            chunk = self._current_chunk
            self._is_playing = False
            self._current_chunk = None

            result = PlaybackResult(
                chunk_id=chunk.chunk_id,
                request_id=chunk.request_id,
                status=PlaybackStatus.COMPLETED,
            )
            self._play_results.append(result)
            return result