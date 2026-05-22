from __future__ import annotations

from dataclasses import dataclass

import pytest

from jarvis.presence.adapters import WakeWordDetection, make_fake_audio_frame
from jarvis.presence.models import AudioFrame, Transcript, VoiceActivity
from jarvis.presence.models.transcript import TranscriptKind
from jarvis.presence.models.voice_activity import VoiceActivityState
from jarvis.presence.real_smoke import (
    CompletedSpeechSegment,
    RealPresenceListeningSmokeConfig,
    RealPresenceListeningSmokeHarness,
    RealPresenceListeningSmokeReport,
)


class StubSmokeMicrophone:
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


class StubSmokeWakeWord:
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


class StubSmokeVad:
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


class StubSmokeStt:
    def __init__(self, *, text: str = "hello jarvis") -> None:
        self.text = text
        self.calls = 0
        self.last_segment: CompletedSpeechSegment | None = None

    def transcribe(self, segment: CompletedSpeechSegment) -> Transcript | None:
        self.calls += 1
        self.last_segment = segment

        return Transcript(
            segment_id=segment.segment_id,
            text=self.text,
            kind=TranscriptKind.FINAL,
            confidence=0.98,
            language="en",
            metadata={"source": "test"},
        )


@dataclass(frozen=True, slots=True)
class ReportExample:
    report: RealPresenceListeningSmokeReport


def make_frames() -> tuple[AudioFrame, ...]:
    return (
        make_fake_audio_frame(frame_index=0),
        make_fake_audio_frame(frame_index=1),
        make_fake_audio_frame(frame_index=2),
    )


def make_harness(
    *,
    wake_detect_on_call: int | None = 1,
    require_wake: bool = True,
    fail_start: bool = False,
) -> tuple[
    RealPresenceListeningSmokeHarness,
    StubSmokeMicrophone,
    StubSmokeWakeWord,
    StubSmokeVad,
    StubSmokeStt,
]:
    microphone = StubSmokeMicrophone(
        frames=make_frames(),
        fail_start=fail_start,
    )
    wake_word = StubSmokeWakeWord(detect_on_call=wake_detect_on_call)
    vad = StubSmokeVad(
        states=(
            VoiceActivityState.SPEECH_STARTED,
            VoiceActivityState.SPEECH_CONTINUING,
            VoiceActivityState.SPEECH_ENDED,
        )
    )
    stt = StubSmokeStt()

    config = RealPresenceListeningSmokeConfig(
        duration_seconds=0.5,
        max_frames=10,
        frame_sleep_seconds=0.0,
        require_wake=require_wake,
        stop_after_first_transcript=True,
    )

    harness = RealPresenceListeningSmokeHarness(
        config=config,
        microphone=microphone,
        wake_word=wake_word,
        vad=vad,
        stt=stt,
    )

    return harness, microphone, wake_word, vad, stt


def test_real_presence_smoke_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        RealPresenceListeningSmokeConfig(duration_seconds=0).validate()

    with pytest.raises(ValueError):
        RealPresenceListeningSmokeConfig(max_frames=0).validate()

    with pytest.raises(ValueError):
        RealPresenceListeningSmokeConfig(frame_sleep_seconds=-1).validate()

    with pytest.raises(ValueError):
        RealPresenceListeningSmokeConfig(min_segment_frames=0).validate()


def test_completed_speech_segment_model() -> None:
    segment = CompletedSpeechSegment(
        segment_id="segment-1",
        audio_data=b"\x00\x01",
        sample_rate=16_000,
        channels=1,
        metadata={"source": "test"},
    )

    assert segment.segment_id == "segment-1"
    assert segment.audio_data == b"\x00\x01"
    assert segment.sample_rate == 16_000
    assert segment.channels == 1
    assert segment.metadata == {"source": "test"}


def test_real_presence_smoke_harness_transcribes_after_wake_and_vad() -> None:
    harness, microphone, wake_word, vad, stt = make_harness()

    report = harness.run()

    assert report.passed is True
    assert report.frames_read == 3
    assert report.wake_detected is True
    assert report.speech_completed is True
    assert report.transcript_count == 1
    assert report.transcripts[0].text == "hello jarvis"
    assert microphone.start_count == 1
    assert microphone.stop_count == 1
    assert wake_word.calls == 1
    assert vad.calls == 3
    assert stt.calls == 1
    assert stt.last_segment is not None
    assert stt.last_segment.metadata["frame_count"] == 3


def test_real_presence_smoke_harness_can_skip_wake_requirement() -> None:
    harness, _microphone, wake_word, vad, stt = make_harness(
        wake_detect_on_call=None,
        require_wake=False,
    )

    report = harness.run()

    assert report.passed is True
    assert report.wake_detected is True
    assert wake_word.calls == 0
    assert vad.calls == 3
    assert stt.calls == 1


def test_real_presence_smoke_harness_fails_without_wake() -> None:
    harness, _microphone, wake_word, vad, stt = make_harness(
        wake_detect_on_call=None,
        require_wake=True,
    )

    report = harness.run()

    assert report.passed is False
    assert report.transcript_count == 0
    assert report.wake_detected is False
    assert wake_word.calls == 3
    assert vad.calls == 0
    assert stt.calls == 0


def test_real_presence_smoke_harness_reports_microphone_failure() -> None:
    harness, microphone, _wake_word, _vad, _stt = make_harness(
        fail_start=True,
    )

    report = harness.run()

    assert report.passed is False
    assert report.transcript_count == 0
    assert report.errors == ("RuntimeError: microphone failed",)
    assert microphone.start_count == 1
    assert microphone.stop_count == 1