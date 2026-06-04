from __future__ import annotations

import pytest

from jarvis.live import (
    LiveAudioAdapterReport,
    LiveAudioBuffer,
    LiveAudioDeviceKind,
    LiveAudioFrame,
    LiveAudioRuntime,
    LiveAudioRuntimeConfig,
    LiveAudioRuntimeStatus,
    LiveMicrophoneAdapter,
    LivePlaybackAdapter,
    LiveResponse,
    LiveResponseGenerationSource,
    LiveResponseKind,
    LiveResponseSafety,
    LiveSessionConfig,
    LiveSessionMode,
    LiveSessionStateRuntime,
    LiveSTTAdapter,
    LiveTranscript,
    LiveTranscriptKind,
    LiveTTSAdapter,
    LiveTurnId,
    fake_final_transcript,
    fake_input_frame,
    make_live_audio_adapter_report,
    make_live_audio_buffer,
    make_live_response,
)


class FakeMicrophoneAdapter:
    def prepare(
        self,
        config: LiveAudioRuntimeConfig,
    ) -> LiveAudioAdapterReport:
        return make_live_audio_adapter_report(
            kind=LiveAudioDeviceKind.MICROPHONE,
            ready=True,
            message="fake microphone ready",
        )

    def capture_frame(
        self,
        config: LiveAudioRuntimeConfig,
    ) -> LiveAudioFrame:
        return fake_input_frame(config=config)


class FakeSTTAdapter:
    def prepare(
        self,
        config: LiveAudioRuntimeConfig,
    ) -> LiveAudioAdapterReport:
        return make_live_audio_adapter_report(
            kind=LiveAudioDeviceKind.STT,
            ready=True,
            message="fake STT ready",
        )

    def transcribe(
        self,
        frame: LiveAudioFrame,
        turn_id: LiveTurnId,
    ) -> LiveTranscript:
        return fake_final_transcript(
            turn_id=turn_id,
            text="Jarvis continue step 50.",
        )


class FakeTTSAdapter:
    def prepare(
        self,
        config: LiveAudioRuntimeConfig,
    ) -> LiveAudioAdapterReport:
        return make_live_audio_adapter_report(
            kind=LiveAudioDeviceKind.TTS,
            ready=True,
            message="fake TTS ready",
        )

    def synthesize(
        self,
        response: LiveResponse,
    ) -> LiveAudioBuffer:
        return make_live_audio_buffer(
            sample_rate_hz=16000,
            channels=1,
            duration_ms=200,
            pcm=b"fake-pcm",
            response_id=str(response.response_id),
        )


class FakePlaybackAdapter:
    def __init__(self) -> None:
        self.played = False
        self.stopped = False

    def prepare(
        self,
        config: LiveAudioRuntimeConfig,
    ) -> LiveAudioAdapterReport:
        return make_live_audio_adapter_report(
            kind=LiveAudioDeviceKind.PLAYBACK,
            ready=True,
            message="fake playback ready",
        )

    def play(
        self,
        buffer: LiveAudioBuffer,
    ) -> LiveAudioAdapterReport:
        self.played = True
        return make_live_audio_adapter_report(
            kind=LiveAudioDeviceKind.PLAYBACK,
            ready=True,
            message="fake playback accepted buffer",
        )

    def stop(self) -> LiveAudioAdapterReport:
        self.stopped = True
        return make_live_audio_adapter_report(
            kind=LiveAudioDeviceKind.PLAYBACK,
            ready=True,
            message="fake playback stopped",
        )


def _state() -> LiveSessionStateRuntime:
    runtime = LiveSessionStateRuntime(
        config=LiveSessionConfig(
            mode=LiveSessionMode.REAL_VOICE,
            real_microphone_enabled=True,
            real_stt_enabled=True,
            real_tts_enabled=True,
        )
    )
    runtime.start()
    runtime.mark_ready()
    return runtime


def _audio(playback: FakePlaybackAdapter | None = None) -> LiveAudioRuntime:
    return LiveAudioRuntime(
        live_state=_state(),
        microphone=FakeMicrophoneAdapter(),
        stt=FakeSTTAdapter(),
        tts=FakeTTSAdapter(),
        playback=playback or FakePlaybackAdapter(),
    )


def _generated_response(turn_id: LiveTurnId) -> LiveResponse:
    return make_live_response(
        turn_id=turn_id,
        kind=LiveResponseKind.CONVERSATIONAL,
        text="Generated response from response boundary.",
        generation_source=LiveResponseGenerationSource.RESPONSE_GENERATOR,
        safety=LiveResponseSafety.SAFE_TO_SPEAK,
    )


def test_live_audio_runtime_requires_prepare_before_capture() -> None:
    audio = _audio()

    result = audio.capture_frame()

    assert result.status == LiveAudioRuntimeStatus.BLOCKED


def test_live_audio_runtime_prepares_all_adapters() -> None:
    audio = _audio()

    result = audio.prepare()

    assert result.status == LiveAudioRuntimeStatus.READY
    assert audio.snapshot().prepared is True


def test_live_audio_runtime_captures_frame_after_prepare() -> None:
    audio = _audio()
    audio.prepare()

    result = audio.capture_frame()

    assert result.status == LiveAudioRuntimeStatus.READY
    assert result.frame is not None
    assert audio.snapshot().captured_frames == 1


def test_live_audio_runtime_transcribes_current_turn() -> None:
    state = _state()
    turn = state.start_user_turn()
    assert turn.state.current_turn_id is not None

    audio = LiveAudioRuntime(
        live_state=state,
        microphone=FakeMicrophoneAdapter(),
        stt=FakeSTTAdapter(),
        tts=FakeTTSAdapter(),
        playback=FakePlaybackAdapter(),
    )
    audio.prepare()
    frame = fake_input_frame(config=LiveAudioRuntimeConfig())

    result = audio.transcribe_frame(
        frame=frame,
        turn_id=turn.state.current_turn_id,
    )

    assert result.status == LiveAudioRuntimeStatus.READY
    assert result.transcript is not None
    assert result.transcript.kind == LiveTranscriptKind.FINAL
    assert result.live_state_result is not None
    assert result.live_state_result.state.last_transcript == result.transcript


def test_live_audio_runtime_blocks_transcript_for_wrong_turn() -> None:
    state = _state()
    state.start_user_turn()

    audio = LiveAudioRuntime(
        live_state=state,
        microphone=FakeMicrophoneAdapter(),
        stt=FakeSTTAdapter(),
        tts=FakeTTSAdapter(),
        playback=FakePlaybackAdapter(),
    )
    audio.prepare()
    frame = fake_input_frame(config=LiveAudioRuntimeConfig())

    result = audio.transcribe_frame(
        frame=frame,
        turn_id=LiveTurnId("wrong_turn"),
    )

    assert result.status == LiveAudioRuntimeStatus.BLOCKED


def test_live_audio_runtime_synthesizes_only_generated_response() -> None:
    state = _state()
    turn = state.start_user_turn()
    assert turn.state.current_turn_id is not None

    audio = LiveAudioRuntime(
        live_state=state,
        microphone=FakeMicrophoneAdapter(),
        stt=FakeSTTAdapter(),
        tts=FakeTTSAdapter(),
        playback=FakePlaybackAdapter(),
    )
    audio.prepare()
    response = _generated_response(turn.state.current_turn_id)

    result = audio.synthesize_response(response)

    assert result.status == LiveAudioRuntimeStatus.READY
    assert result.buffer is not None
    assert result.response == response


def test_live_audio_runtime_blocks_scripted_conversational_response() -> None:
    state = _state()
    turn = state.start_user_turn()
    assert turn.state.current_turn_id is not None

    audio = LiveAudioRuntime(
        live_state=state,
        microphone=FakeMicrophoneAdapter(),
        stt=FakeSTTAdapter(),
        tts=FakeTTSAdapter(),
        playback=FakePlaybackAdapter(),
    )
    audio.prepare()
    response = make_live_response(
        turn_id=turn.state.current_turn_id,
        kind=LiveResponseKind.CONVERSATIONAL,
        text="Scripted diagnostic conversation.",
        generation_source=LiveResponseGenerationSource.DIAGNOSTIC_SYSTEM,
        safety=LiveResponseSafety.SAFE_TO_SPEAK,
    )

    result = audio.synthesize_response(response)

    assert result.status == LiveAudioRuntimeStatus.BLOCKED


def test_live_audio_runtime_play_response_updates_speaking_state() -> None:
    playback = FakePlaybackAdapter()
    state = _state()
    turn = state.start_user_turn()
    assert turn.state.current_turn_id is not None

    audio = LiveAudioRuntime(
        live_state=state,
        microphone=FakeMicrophoneAdapter(),
        stt=FakeSTTAdapter(),
        tts=FakeTTSAdapter(),
        playback=playback,
    )
    audio.prepare()
    response = _generated_response(turn.state.current_turn_id)

    result = audio.play_response(response)

    assert result.status == LiveAudioRuntimeStatus.READY
    assert result.live_state_result is not None
    assert result.live_state_result.state.assistant_speaking is True
    assert playback.played is True


def test_live_audio_runtime_stop_output_interrupts_state() -> None:
    playback = FakePlaybackAdapter()
    audio = _audio(playback=playback)
    audio.prepare()

    result = audio.stop_output(reason="user interruption")

    assert result.status == LiveAudioRuntimeStatus.READY
    assert result.live_state_result is not None
    assert playback.stopped is True


def test_live_audio_runtime_snapshot_tracks_counts() -> None:
    playback = FakePlaybackAdapter()
    state = _state()
    turn = state.start_user_turn()
    assert turn.state.current_turn_id is not None

    audio = LiveAudioRuntime(
        live_state=state,
        microphone=FakeMicrophoneAdapter(),
        stt=FakeSTTAdapter(),
        tts=FakeTTSAdapter(),
        playback=playback,
    )
    audio.prepare()
    audio.capture_frame()
    response = _generated_response(turn.state.current_turn_id)
    audio.play_response(response)

    snapshot = audio.snapshot()

    assert snapshot.prepared is True
    assert snapshot.captured_frames == 1
    assert snapshot.synthesized_responses == 1
    assert snapshot.played_responses == 1


def test_live_audio_runtime_adapter_protocols_are_satisfied() -> None:
    mic: LiveMicrophoneAdapter = FakeMicrophoneAdapter()
    stt: LiveSTTAdapter = FakeSTTAdapter()
    tts: LiveTTSAdapter = FakeTTSAdapter()
    playback: LivePlaybackAdapter = FakePlaybackAdapter()

    assert mic is not None
    assert stt is not None
    assert tts is not None
    assert playback is not None


def test_live_audio_runtime_config_validation() -> None:
    with pytest.raises(ValueError):
        LiveAudioRuntimeConfig(sample_rate_hz=0)