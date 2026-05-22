from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from jarvis.presence.adapters import (
    PlaybackResult,
    PlaybackStatus,
    WakeWordDetection,
    make_fake_audio_frame,
)
from jarvis.presence.full_voice_smoke import (
    FullVoiceSmokeConfig,
    FullVoiceSmokeHarness,
    FullVoiceSmokeReport,
    FullVoiceSmokeTurn,
    make_canned_transcript,
)
from jarvis.presence.models import AudioFrame, SpeechChunk, Transcript, VoiceActivity
from jarvis.presence.models.voice_activity import VoiceActivityState


class StubFullVoiceMicrophone:
    def __init__(
        self,
        *,
        frames: tuple[AudioFrame, ...],
        fail_start: bool = False,
    ) -> None:
        self._frames = list(frames)
        self._started = False
        self._fail_start = fail_start
        self.start_count = 0
        self.stop_count = 0

    def start(self) -> None:
        self.start_count += 1

        if self._fail_start:
            raise RuntimeError("microphone failed")

        self._started = True

    def stop(self) -> None:
        self.stop_count += 1
        self._started = False

    def read_frame(self) -> AudioFrame | None:
        if not self._started or not self._frames:
            return None

        return self._frames.pop(0)


class StubFullVoiceWakeWord:
    def __init__(self, *, detect_on_call: int | None = 1) -> None:
        self._detect_on_call = detect_on_call
        self.calls = 0

    def detect(self, frame: AudioFrame) -> WakeWordDetection | None:
        self.calls += 1

        if self._detect_on_call != self.calls:
            return None

        return WakeWordDetection(
            frame_id=frame.frame_id,
            wake_word="jarvis",
            confidence=0.99,
            metadata={"source": "test"},
        )


class StubFullVoiceVad:
    def __init__(self, *, states: tuple[VoiceActivityState, ...]) -> None:
        self._states = list(states)
        self.calls = 0

    def detect(self, frame: AudioFrame) -> VoiceActivity:
        self.calls += 1
        state = (
            self._states.pop(0)
            if self._states
            else VoiceActivityState.SILENCE
        )

        return VoiceActivity(
            frame_id=frame.frame_id,
            state=state,
            is_speech=state
            in {
                VoiceActivityState.SPEECH_STARTED,
                VoiceActivityState.SPEECH_CONTINUING,
            },
            confidence=0.95,
            energy=1_000.0,
            metadata={"source": "test"},
        )


class StubFullVoiceStt:
    def __init__(self, *, text: str = "hello jarvis") -> None:
        self.text = text
        self.calls = 0

    def transcribe(self, segment: Any) -> Transcript | None:
        self.calls += 1

        return make_canned_transcript(
            segment_id=segment.segment_id,
            text=self.text,
        )


class StubFullVoiceTts:
    def __init__(self) -> None:
        self.calls = 0
        self.last_text: str | None = None
        self.last_metadata: dict[str, Any] | None = None

    def synthesize(
        self,
        *,
        text: str,
        request_id: str,
        voice_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[SpeechChunk, ...]:
        del voice_id

        self.calls += 1
        self.last_text = text
        self.last_metadata = metadata

        return (
            SpeechChunk(
                request_id=request_id,
                chunk_id=uuid4().hex,
                audio_data=b"\x00\x01\x02\x03",
                sample_rate=16_000,
                channels=1,
                final=True,
                metadata={"source": "test_tts"},
            ),
        )


class StubFullVoicePlayback:
    def __init__(self, *, fail_play: bool = False) -> None:
        self._is_playing = False
        self._fail_play = fail_play
        self.calls = 0
        self.stop_calls = 0

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    def play(self, chunk: SpeechChunk) -> PlaybackResult:
        self.calls += 1

        if self._fail_play:
            return PlaybackResult(
                result_id=uuid4().hex,
                chunk_id=chunk.chunk_id,
                request_id=chunk.request_id,
                status=PlaybackStatus.FAILED,
                error="playback failed",
                metadata={"source": "test"},
            )

        self._is_playing = True

        return PlaybackResult(
            result_id=uuid4().hex,
            chunk_id=chunk.chunk_id,
            request_id=chunk.request_id,
            status=PlaybackStatus.STARTED,
            metadata={"source": "test"},
        )

    def stop(self) -> None:
        self.stop_calls += 1
        self._is_playing = False


def make_frames() -> tuple[AudioFrame, ...]:
    return (
        make_fake_audio_frame(frame_index=0),
        make_fake_audio_frame(frame_index=1),
        make_fake_audio_frame(frame_index=2),
    )


def make_harness(
    *,
    require_wake: bool = True,
    wake_detect_on_call: int | None = 1,
    fail_start: bool = False,
    fail_play: bool = False,
) -> tuple[
    FullVoiceSmokeHarness,
    StubFullVoiceMicrophone,
    StubFullVoiceWakeWord,
    StubFullVoiceVad,
    StubFullVoiceStt,
    StubFullVoiceTts,
    StubFullVoicePlayback,
]:
    microphone = StubFullVoiceMicrophone(
        frames=make_frames(),
        fail_start=fail_start,
    )
    wake_word = StubFullVoiceWakeWord(detect_on_call=wake_detect_on_call)
    vad = StubFullVoiceVad(
        states=(
            VoiceActivityState.SPEECH_STARTED,
            VoiceActivityState.SPEECH_CONTINUING,
            VoiceActivityState.SPEECH_ENDED,
        )
    )
    stt = StubFullVoiceStt()
    tts = StubFullVoiceTts()
    playback = StubFullVoicePlayback(fail_play=fail_play)

    config = FullVoiceSmokeConfig(
        duration_seconds=0.5,
        max_frames=10,
        frame_sleep_seconds=0.0,
        require_wake=require_wake,
        stop_after_first_turn=True,
        response_text="Yes sir. I heard you.",
    )

    harness = FullVoiceSmokeHarness(
        config=config,
        microphone=microphone,
        wake_word=wake_word,
        vad=vad,
        stt=stt,
        tts=tts,
        playback=playback,
    )

    return harness, microphone, wake_word, vad, stt, tts, playback


def test_full_voice_smoke_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        FullVoiceSmokeConfig(duration_seconds=0).validate()

    with pytest.raises(ValueError):
        FullVoiceSmokeConfig(max_frames=0).validate()

    with pytest.raises(ValueError):
        FullVoiceSmokeConfig(frame_sleep_seconds=-1).validate()

    with pytest.raises(ValueError):
        FullVoiceSmokeConfig(min_segment_frames=0).validate()

    with pytest.raises(ValueError):
        FullVoiceSmokeConfig(response_text=" ").validate()


def test_make_canned_transcript() -> None:
    transcript = make_canned_transcript(
        segment_id="segment-1",
        text="hello",
    )

    assert transcript.segment_id == "segment-1"
    assert transcript.text == "hello"


def test_full_voice_smoke_harness_completes_one_turn() -> None:
    harness, microphone, wake_word, vad, stt, tts, playback = make_harness()

    report = harness.run()

    assert isinstance(report, FullVoiceSmokeReport)
    assert report.passed is True
    assert report.frames_read == 3
    assert report.wake_detected is True
    assert report.speech_completed is True
    assert report.turn_count == 1
    assert report.transcript_count == 1
    assert report.playback_count == 1
    assert report.turns[0].transcript.text == "hello jarvis"
    assert report.turns[0].response_text == "Yes sir. I heard you."
    assert isinstance(report.turns[0], FullVoiceSmokeTurn)
    assert microphone.start_count == 1
    assert microphone.stop_count == 1
    assert wake_word.calls == 1
    assert vad.calls == 3
    assert stt.calls == 1
    assert tts.calls == 1
    assert playback.calls == 1
    assert playback.stop_calls == 1


def test_full_voice_smoke_harness_can_skip_wake_requirement() -> None:
    harness, _microphone, wake_word, vad, stt, tts, playback = make_harness(
        require_wake=False,
        wake_detect_on_call=None,
    )

    report = harness.run()

    assert report.passed is True
    assert report.wake_detected is True
    assert wake_word.calls == 0
    assert vad.calls == 3
    assert stt.calls == 1
    assert tts.calls == 1
    assert playback.calls == 1


def test_full_voice_smoke_harness_fails_without_wake() -> None:
    harness, _microphone, wake_word, vad, stt, tts, playback = make_harness(
        require_wake=True,
        wake_detect_on_call=None,
    )

    report = harness.run()

    assert report.passed is False
    assert report.turn_count == 0
    assert report.wake_detected is False
    assert wake_word.calls == 3
    assert vad.calls == 0
    assert stt.calls == 0
    assert tts.calls == 0
    assert playback.calls == 0


def test_full_voice_smoke_harness_reports_microphone_failure() -> None:
    harness, microphone, _wake_word, _vad, _stt, _tts, _playback = make_harness(
        fail_start=True,
    )

    report = harness.run()

    assert report.passed is False
    assert report.turn_count == 0
    assert report.errors == ("RuntimeError: microphone failed",)
    assert microphone.start_count == 1
    assert microphone.stop_count == 1


def test_full_voice_smoke_harness_records_failed_playback_result() -> None:
    harness, _microphone, _wake_word, _vad, _stt, _tts, playback = make_harness(
        fail_play=True,
    )

    report = harness.run()

    assert report.passed is True
    assert report.turn_count == 1
    assert report.playback_count == 1
    assert report.turns[0].playback_results[0].status == PlaybackStatus.FAILED
    assert playback.calls == 1