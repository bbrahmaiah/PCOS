from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast

from jarvis.live import LiveResponse
from jarvis.voice import (
    VoiceActivityDecision,
    VoiceActivityResult,
    VoiceActivityRuntimeStatus,
    VoiceBargeInDisposition,
    VoiceBargeInRequest,
    VoiceBargeInResult,
    VoiceBargeInRuntimeStatus,
    VoiceCognitionRequest,
    VoiceCognitionResult,
    VoiceHealthResult,
    VoiceHealthStatus,
    VoiceInputFrame,
    VoiceInputFrameKind,
    VoiceMicrophoneCaptureOperation,
    VoiceMicrophoneCaptureResult,
    VoiceMicrophoneCaptureSnapshot,
    VoiceMicrophoneCaptureStatus,
    VoicePlaybackOperation,
    VoicePlaybackResult,
    VoicePlaybackRuntimeStatus,
    VoicePlaybackSnapshot,
    VoicePlaybackState,
    VoicePlaybackStatus,
    VoiceSessionLoopEvent,
    VoiceSessionLoopOperation,
    VoiceSessionLoopPolicy,
    VoiceSessionLoopRuntime,
    VoiceSessionLoopStatus,
    VoiceSTTMode,
    VoiceSTTOperation,
    VoiceSTTResult,
    VoiceSTTRuntimeStatus,
    VoiceSTTTranscriptCandidate,
    VoiceSTTTranscriptSafety,
    VoiceTranscript,
    VoiceTranscriptKind,
    VoiceTTSChunk,
    VoiceTTSChunkStatus,
    VoiceTTSOperation,
    VoiceTTSResult,
    VoiceTTSRuntimeStatus,
    make_voice_frame_id,
    make_voice_playback_id,
    make_voice_segment_id,
    make_voice_session_id,
    make_voice_transcript_id,
    make_voice_tts_chunk_id,
    utc_now,
)


def _frame() -> VoiceInputFrame:
    return VoiceInputFrame(
        frame_id=make_voice_frame_id(),
        session_id=make_voice_session_id(),
        kind=VoiceInputFrameKind.PCM16_MONO,
        sample_rate_hz=16_000,
        channels=1,
        data=b"\x00\x01" * 320,
        captured_at=utc_now(),
        duration_ms=20,
    )


def _transcript(
    text: str,
    *,
    kind: VoiceTranscriptKind,
) -> VoiceTranscript:
    return VoiceTranscript(
        transcript_id=make_voice_transcript_id(),
        session_id=make_voice_session_id(),
        segment_id=make_voice_segment_id(),
        kind=kind,
        text=text,
        confidence=0.95,
        created_at=utc_now(),
    )


def _generated_response_from(transcript_text: str) -> LiveResponse:
    return cast(
        LiveResponse,
        SimpleNamespace(
            response_id="response_test",
            text=f"generated_from_transcript::{transcript_text}",
            created_at=utc_now(),
            metadata={"source": "neutral_derived_test_response"},
        ),
    )


def _tts_chunk() -> VoiceTTSChunk:
    return VoiceTTSChunk(
        chunk_id=make_voice_tts_chunk_id(),
        session_id=make_voice_session_id(),
        status=VoiceTTSChunkStatus.SYNTHESIZED,
        audio=b"RIFFfakewav",
        sample_rate_hz=22_050,
        duration_ms=120,
        created_at=utc_now(),
    )


@dataclass
class FakeMicrophone:
    frames: list[VoiceInputFrame] = field(default_factory=lambda: [_frame()])
    started: bool = False
    stopped: bool = False
    fail_capture: bool = False

    def start(self) -> object:
        self.started = True
        return object()

    def capture_once(self) -> VoiceMicrophoneCaptureResult:
        if self.fail_capture:
            return VoiceMicrophoneCaptureResult(
                status=VoiceMicrophoneCaptureStatus.FAILED,
                operation=VoiceMicrophoneCaptureOperation.CAPTURE_ONCE,
                frame=None,
                device=None,
                message="capture_failed",
                created_at=utc_now(),
            )

        frame = self.frames.pop(0) if self.frames else _frame()
        return VoiceMicrophoneCaptureResult(
            status=VoiceMicrophoneCaptureStatus.CAPTURING,
            operation=VoiceMicrophoneCaptureOperation.CAPTURE_ONCE,
            frame=frame,
            device=None,
            message="captured",
            created_at=utc_now(),
        )

    def stop(self) -> object:
        self.stopped = True
        return object()

    def snapshot(self) -> VoiceMicrophoneCaptureSnapshot:
        return VoiceMicrophoneCaptureSnapshot(
            status=VoiceMicrophoneCaptureStatus.CAPTURING,
            device=None,
            captured_frames=1,
            captured_bytes=1,
            consecutive_failures=0,
            last_error=None,
            created_at=utc_now(),
        )


@dataclass
class FakeVad:
    decisions: list[VoiceActivityDecision]

    def analyze_frame(self, frame: VoiceInputFrame) -> VoiceActivityResult:
        decision = (
            self.decisions.pop(0)
            if self.decisions
            else VoiceActivityDecision.SILENCE
        )
        return VoiceActivityResult(
            status=VoiceActivityRuntimeStatus.SPEECH_ACTIVE,
            operation=cast(Any, "analyze_frame"),
            decision=decision,
            segment=None,
            energy=1000.0,
            threshold=500.0,
            message=decision.value,
            created_at=utc_now(),
        )

    def reset(self) -> object:
        return object()

    def snapshot(self) -> object:
        return object()


@dataclass
class FakeSTT:
    partials: int = 0
    finals: int = 0

    def prepare(self) -> object:
        return object()

    def transcribe_partial(
        self,
        frames: tuple[VoiceInputFrame, ...],
    ) -> VoiceSTTResult:
        self.partials += 1
        transcript = _transcript(
            "partial user speech",
            kind=VoiceTranscriptKind.PARTIAL,
        )
        return VoiceSTTResult(
            status=VoiceSTTRuntimeStatus.TRANSCRIBING,
            operation=VoiceSTTOperation.TRANSCRIBE_PARTIAL,
            transcript=transcript,
            candidate=VoiceSTTTranscriptCandidate(
                text=transcript.text,
                confidence=0.9,
                mode=VoiceSTTMode.FAST_PARTIAL,
                safety=VoiceSTTTranscriptSafety.PREDICTION_ONLY,
                safe_for_action=False,
                latency_ms=1.0,
                model_name="fake",
            ),
            model=None,
            message="partial",
            created_at=utc_now(),
        )

    def transcribe_final(
        self,
        frames: tuple[VoiceInputFrame, ...],
        *,
        allow_action_candidate: bool = False,
    ) -> VoiceSTTResult:
        self.finals += 1
        transcript = _transcript(
            "final user request",
            kind=VoiceTranscriptKind.FINAL,
        )
        return VoiceSTTResult(
            status=VoiceSTTRuntimeStatus.TRANSCRIBING,
            operation=VoiceSTTOperation.TRANSCRIBE_FINAL,
            transcript=transcript,
            candidate=VoiceSTTTranscriptCandidate(
                text=transcript.text,
                confidence=0.95,
                mode=VoiceSTTMode.ACCURATE_FINAL,
                safety=VoiceSTTTranscriptSafety.SAFE_FOR_DIALOGUE,
                safe_for_action=False,
                latency_ms=1.0,
                model_name="fake",
            ),
            model=None,
            message="final",
            created_at=utc_now(),
        )

    def reset(self) -> object:
        return object()

    def snapshot(self) -> object:
        return object()


@dataclass
class FakeCognition:
    prepared: bool = False
    prefetches: int = 0
    responses: int = 0

    def prepare(self, *, user_label: str, assistant_name: str) -> object:
        self.prepared = True
        return object()

    def prefetch_from_partial(
        self,
        request: VoiceCognitionRequest,
    ) -> object:
        self.prefetches += 1
        return object()

    def think_from_transcript(
        self,
        request: VoiceCognitionRequest,
    ) -> VoiceCognitionResult:
        self.responses += 1
        response = _generated_response_from(request.transcript.text)

        return cast(
            VoiceCognitionResult,
            SimpleNamespace(response=response),
        )

    def snapshot(self) -> object:
        return object()


@dataclass
class FakeTTS:
    prepared: bool = False
    calls: int = 0
    last_response_text: str | None = None

    def prepare(self) -> object:
        self.prepared = True
        return object()

    def synthesize_response(
        self,
        *,
        response: object,
        session_id: object,
        voice: str | None = None,
    ) -> VoiceTTSResult:
        self.calls += 1
        response_text = cast(Any, response).text
        self.last_response_text = str(response_text)

        return VoiceTTSResult(
            status=VoiceTTSRuntimeStatus.SYNTHESIZING,
            operation=VoiceTTSOperation.SYNTHESIZE_RESPONSE,
            request=None,
            chunks=(_tts_chunk(),),
            plans=(),
            voice=None,
            message="tts",
            latency_ms=1.0,
            first_chunk_latency_ms=1.0,
            created_at=utc_now(),
        )

    def reset(self) -> object:
        return object()

    def snapshot(self) -> object:
        return object()


@dataclass
class FakePlayback:
    prepared: bool = False
    enqueued: int = 0
    played: int = 0
    stopped: bool = False

    def prepare(self) -> object:
        self.prepared = True
        return object()

    def enqueue_chunks(self, chunks: object) -> VoicePlaybackResult:
        self.enqueued += len(tuple(cast(Any, chunks)))
        return VoicePlaybackResult(
            status=VoicePlaybackRuntimeStatus.QUEUED,
            operation=VoicePlaybackOperation.ENQUEUE,
            playback_state=None,
            played_chunks=(),
            queued_chunks=self.enqueued,
            speaker=None,
            message="queued",
            latency_ms=1.0,
            first_audio_latency_ms=None,
            created_at=utc_now(),
        )

    def play_all(self) -> VoicePlaybackResult:
        self.played += 1
        return VoicePlaybackResult(
            status=VoicePlaybackRuntimeStatus.READY,
            operation=VoicePlaybackOperation.PLAY_ALL,
            playback_state=VoicePlaybackState(
                playback_id=make_voice_playback_id(),
                session_id=make_voice_session_id(),
                status=VoicePlaybackStatus.STOPPED,
                chunk_id=None,
                started_at=utc_now(),
                stopped_at=utc_now(),
            ),
            played_chunks=(_tts_chunk(),),
            queued_chunks=0,
            speaker=None,
            message="played",
            latency_ms=1.0,
            first_audio_latency_ms=1.0,
            created_at=utc_now(),
        )

    def stop(self) -> object:
        self.stopped = True
        return object()

    def reset(self) -> object:
        return object()

    def snapshot(self) -> VoicePlaybackSnapshot:
        return VoicePlaybackSnapshot(
            status=VoicePlaybackRuntimeStatus.READY,
            speaker=None,
            queued_chunks=0,
            played_chunks=self.played,
            failed_chunks=0,
            stopped_count=1 if self.stopped else 0,
            current_playback=None,
            last_latency_ms=None,
            last_first_audio_latency_ms=None,
            last_error=None,
            created_at=utc_now(),
        )


@dataclass
class FakeBargeIn:
    prepared: bool = False
    interrupts: int = 0

    def prepare(self, playback: object | None = None) -> object:
        self.prepared = True
        return object()

    def evaluate_transcript(
        self,
        request: VoiceBargeInRequest,
    ) -> VoiceBargeInResult:
        self.interrupts += 1
        return VoiceBargeInResult(
            status=VoiceBargeInRuntimeStatus.INTERRUPTED,
            operation=cast(Any, "evaluate_transcript"),
            disposition=VoiceBargeInDisposition.STOP_PLAYBACK,
            signal=None,
            interrupted_context=None,
            playback_result=None,
            message="interrupted",
            latency_ms=1.0,
            created_at=utc_now(),
        )

    def reset(self) -> object:
        return object()

    def snapshot(self) -> object:
        return object()


@dataclass
class FakeHealth:
    checks: int = 0
    recoveries: int = 0
    status: VoiceHealthStatus = VoiceHealthStatus.HEALTHY

    def check(self) -> VoiceHealthResult:
        self.checks += 1
        return VoiceHealthResult(
            status=self.status,
            operation=cast(Any, "check"),
            subsystem_health=(),
            recovery_attempts=(),
            message="health",
            latency_ms=1.0,
            created_at=utc_now(),
        )

    def recover(self) -> VoiceHealthResult:
        self.recoveries += 1
        self.status = VoiceHealthStatus.HEALTHY
        return self.check()


def test_voice_session_loop_start_prepares_components() -> None:
    cognition = FakeCognition()
    tts = FakeTTS()
    playback = FakePlayback()
    barge_in = FakeBargeIn()

    runtime = VoiceSessionLoopRuntime(
        microphone=FakeMicrophone(),
        vad=FakeVad([VoiceActivityDecision.SILENCE]),
        stt=FakeSTT(),
        cognition=cognition,
        tts=tts,
        playback=playback,
        barge_in=barge_in,
        health=FakeHealth(),
    )

    result = runtime.start()

    assert result.status == VoiceSessionLoopStatus.LISTENING
    assert cognition.prepared is True
    assert tts.prepared is True
    assert playback.prepared is True
    assert barge_in.prepared is True


def test_voice_session_loop_processes_full_voice_turn() -> None:
    stt = FakeSTT()
    cognition = FakeCognition()
    tts = FakeTTS()
    playback = FakePlayback()

    runtime = VoiceSessionLoopRuntime(
        microphone=FakeMicrophone(frames=[_frame(), _frame(), _frame()]),
        vad=FakeVad(
            [
                VoiceActivityDecision.SPEECH_STARTED,
                VoiceActivityDecision.SPEECH_CONTINUED,
                VoiceActivityDecision.SPEECH_ENDED,
            ]
        ),
        stt=stt,
        cognition=cognition,
        tts=tts,
        playback=playback,
        barge_in=FakeBargeIn(),
        health=FakeHealth(),
        policy=VoiceSessionLoopPolicy(partial_transcript_every_frames=2),
    )

    runtime.start()
    runtime.process_once()
    runtime.process_once()
    result = runtime.process_once()
    snapshot = runtime.snapshot()

    assert result.event == VoiceSessionLoopEvent.PLAYBACK_FINISHED
    assert stt.finals == 1
    assert cognition.responses == 1
    assert tts.calls == 1
    assert playback.played == 1
    assert snapshot.responses == 1
    assert tts.last_response_text == "generated_from_transcript::final user request"


def test_voice_session_loop_prefetches_from_partial() -> None:
    stt = FakeSTT()
    cognition = FakeCognition()

    runtime = VoiceSessionLoopRuntime(
        microphone=FakeMicrophone(frames=[_frame(), _frame()]),
        vad=FakeVad(
            [
                VoiceActivityDecision.SPEECH_STARTED,
                VoiceActivityDecision.SPEECH_CONTINUED,
            ]
        ),
        stt=stt,
        cognition=cognition,
        tts=FakeTTS(),
        playback=FakePlayback(),
        barge_in=FakeBargeIn(),
        health=FakeHealth(),
        policy=VoiceSessionLoopPolicy(partial_transcript_every_frames=2),
    )

    runtime.start()
    runtime.process_once()
    result = runtime.process_once()

    assert result.event == VoiceSessionLoopEvent.PARTIAL_TRANSCRIPT
    assert stt.partials == 1
    assert cognition.prefetches == 1


def test_voice_session_loop_handles_barge_in_transcript() -> None:
    barge_in = FakeBargeIn()

    runtime = VoiceSessionLoopRuntime(
        microphone=FakeMicrophone(),
        vad=FakeVad([VoiceActivityDecision.SILENCE]),
        stt=FakeSTT(),
        cognition=FakeCognition(),
        tts=FakeTTS(),
        playback=FakePlayback(),
        barge_in=barge_in,
        health=FakeHealth(),
    )

    runtime.start()
    runtime._assistant_speaking = True  # noqa: SLF001
    result = runtime.handle_transcript(
        _transcript("stop current speech", kind=VoiceTranscriptKind.FINAL)
    )

    assert result.event == VoiceSessionLoopEvent.BARGE_IN_INTERRUPTED
    assert barge_in.interrupts == 1
    assert runtime.snapshot().interruptions == 1


def test_voice_session_loop_health_recovery_runs() -> None:
    health = FakeHealth(status=VoiceHealthStatus.DEGRADED)

    runtime = VoiceSessionLoopRuntime(
        microphone=FakeMicrophone(),
        vad=FakeVad([VoiceActivityDecision.SILENCE]),
        stt=FakeSTT(),
        cognition=FakeCognition(),
        tts=FakeTTS(),
        playback=FakePlayback(),
        barge_in=FakeBargeIn(),
        health=health,
        policy=VoiceSessionLoopPolicy(health_check_every_cycles=1),
    )

    runtime.start()
    runtime.process_once()

    assert health.checks >= 1
    assert health.recoveries >= 1
    assert runtime.snapshot().recoveries >= 1


def test_voice_session_loop_stop_stops_microphone_and_playback() -> None:
    mic = FakeMicrophone()
    playback = FakePlayback()

    runtime = VoiceSessionLoopRuntime(
        microphone=mic,
        vad=FakeVad([VoiceActivityDecision.SILENCE]),
        stt=FakeSTT(),
        cognition=FakeCognition(),
        tts=FakeTTS(),
        playback=playback,
        barge_in=FakeBargeIn(),
        health=FakeHealth(),
    )

    runtime.start()
    result = runtime.stop()

    assert result.status == VoiceSessionLoopStatus.STOPPED
    assert mic.stopped is True
    assert playback.stopped is True


def test_voice_session_loop_run_bounded_cycles() -> None:
    runtime = VoiceSessionLoopRuntime(
        microphone=FakeMicrophone(),
        vad=FakeVad([VoiceActivityDecision.SILENCE]),
        stt=FakeSTT(),
        cognition=FakeCognition(),
        tts=FakeTTS(),
        playback=FakePlayback(),
        barge_in=FakeBargeIn(),
        health=FakeHealth(),
        policy=VoiceSessionLoopPolicy(idle_sleep_seconds=0.0),
    )

    result = runtime.run(max_cycles=3)
    snapshot = runtime.snapshot()

    assert result.operation == VoiceSessionLoopOperation.RUN
    assert snapshot.cycles == 3


def test_voice_session_loop_capture_failures_are_bounded() -> None:
    runtime = VoiceSessionLoopRuntime(
        microphone=FakeMicrophone(fail_capture=True),
        vad=FakeVad([VoiceActivityDecision.SILENCE]),
        stt=FakeSTT(),
        cognition=FakeCognition(),
        tts=FakeTTS(),
        playback=FakePlayback(),
        barge_in=FakeBargeIn(),
        health=FakeHealth(),
        policy=VoiceSessionLoopPolicy(max_consecutive_failures=2),
    )

    runtime.start()
    first = runtime.process_once()
    second = runtime.process_once()

    assert first.status == VoiceSessionLoopStatus.DEGRADED
    assert second.status == VoiceSessionLoopStatus.FAILED


def test_voice_session_loop_enum_values_are_stable() -> None:
    assert VoiceSessionLoopStatus.LISTENING.value == "listening"
    assert VoiceSessionLoopEvent.PLAYBACK_FINISHED.value == "playback_finished"