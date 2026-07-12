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
    VoiceBargeInRuntime,
    VoiceBargeInRuntimeStatus,
    VoiceCognitionRequest,
    VoiceCognitionResult,
    VoiceCognitiveRouter,
    VoiceCognitiveRouterPolicy,
    VoiceHealthResult,
    VoiceHealthStatus,
    VoiceInputFrame,
    VoiceInputFrameKind,
    VoiceLiveSpineMonitor,
    VoiceLiveSpineStatus,
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
    VoiceReflexResponseRuntime,
    VoiceRuntimeConfig,
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
from jarvis.voice.transcript_attention_gate import TranscriptAttentionGatePolicy


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
    confidence: float = 0.95,
    metadata: dict[str, object] | None = None,
) -> VoiceTranscript:
    return VoiceTranscript(
        transcript_id=make_voice_transcript_id(),
        session_id=make_voice_session_id(),
        segment_id=make_voice_segment_id(),
        kind=kind,
        text=text,
        confidence=confidence,
        created_at=utc_now(),
        metadata=metadata or {},
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
            "jarvis final user request",
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


class FakeSTTEmptyFinal(FakeSTT):
    def transcribe_final(
        self,
        frames: tuple[VoiceInputFrame, ...],
        *,
        allow_action_candidate: bool = False,
    ) -> VoiceSTTResult:
        self.finals += 1
        return VoiceSTTResult(
            status=VoiceSTTRuntimeStatus.TRANSCRIBING,
            operation=VoiceSTTOperation.TRANSCRIBE_FINAL,
            transcript=None,
            candidate=None,
            model=None,
            message="empty final",
            created_at=utc_now(),
        )


class FakeSTTBackgroundFinal(FakeSTT):
    def transcribe_final(
        self,
        frames: tuple[VoiceInputFrame, ...],
        *,
        allow_action_candidate: bool = False,
    ) -> VoiceSTTResult:
        self.finals += 1
        transcript = _transcript(
            "background speech without wake word",
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


class FakeSTTNaturalFinalWithoutWake(FakeSTT):
    def transcribe_final(
        self,
        frames: tuple[VoiceInputFrame, ...],
        *,
        allow_action_candidate: bool = False,
    ) -> VoiceSTTResult:
        self.finals += 1
        transcript = _transcript(
            "explain artificial intelligence briefly",
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


class FakeSTTEmptyFinalWithWake(FakeSTTEmptyFinal):
    def transcribe_partial(
        self,
        frames: tuple[VoiceInputFrame, ...],
    ) -> VoiceSTTResult:
        self.partials += 1
        transcript = _transcript(
            "jarvis are you online",
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


class FakeSTTWakePartialLostWakeFinal(FakeSTT):
    def transcribe_partial(
        self,
        frames: tuple[VoiceInputFrame, ...],
    ) -> VoiceSTTResult:
        self.partials += 1
        transcript = _transcript("jarvis", kind=VoiceTranscriptKind.PARTIAL)
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
            "are you online",
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


@dataclass
class FakeCognition:
    prepared: bool = False
    prefetches: int = 0
    responses: int = 0
    requests: list[VoiceCognitionRequest] = field(default_factory=list)

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
        self.requests.append(request)
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
class FakeAsyncPlayback(FakePlayback):
    active: bool = False
    current_state: VoicePlaybackState | None = None

    def play_all(self) -> VoicePlaybackResult:
        self.played += 1
        self.active = True
        self.current_state = VoicePlaybackState(
            playback_id=make_voice_playback_id(),
            session_id=make_voice_session_id(),
            status=VoicePlaybackStatus.PLAYING,
            chunk_id=None,
            started_at=utc_now(),
            stopped_at=None,
            metadata={"async_playback": True},
        )
        return VoicePlaybackResult(
            status=VoicePlaybackRuntimeStatus.PLAYING,
            operation=VoicePlaybackOperation.PLAY_ALL,
            playback_state=self.current_state,
            played_chunks=(),
            queued_chunks=0,
            speaker=None,
            message="async playback started",
            latency_ms=1.0,
            first_audio_latency_ms=0.5,
            created_at=utc_now(),
            metadata={"async_playback": True},
        )

    def complete(self) -> None:
        self.active = False
        self.current_state = None

    def stop(self) -> VoicePlaybackResult:
        self.stopped = True
        self.active = False
        self.current_state = None
        return VoicePlaybackResult(
            status=VoicePlaybackRuntimeStatus.STOPPED,
            operation=VoicePlaybackOperation.STOP,
            playback_state=None,
            played_chunks=(),
            queued_chunks=0,
            speaker=None,
            message="async playback stopped",
            latency_ms=1.0,
            first_audio_latency_ms=None,
            created_at=utc_now(),
            metadata={"async_playback": True},
        )

    def snapshot(self) -> VoicePlaybackSnapshot:
        return VoicePlaybackSnapshot(
            status=(
                VoicePlaybackRuntimeStatus.PLAYING
                if self.active
                else VoicePlaybackRuntimeStatus.READY
            ),
            speaker=None,
            queued_chunks=0,
            played_chunks=self.played,
            failed_chunks=0,
            stopped_count=1 if self.stopped else 0,
            current_playback=self.current_state,
            last_latency_ms=1.0,
            last_first_audio_latency_ms=0.5,
            last_error=None,
            created_at=utc_now(),
        )


@dataclass
class FakeBargeIn:
    prepared: bool = False
    interrupts: int = 0
    disposition: VoiceBargeInDisposition = VoiceBargeInDisposition.STOP_PLAYBACK
    playback: object | None = None

    def prepare(self, playback: object | None = None) -> object:
        self.prepared = True
        self.playback = playback
        return object()

    def evaluate_transcript(
        self,
        request: VoiceBargeInRequest,
    ) -> VoiceBargeInResult:
        self.interrupts += 1
        playback_result = None
        stop = getattr(self.playback, "stop", None)
        if callable(stop):
            playback_result = stop()

        return VoiceBargeInResult(
            status=VoiceBargeInRuntimeStatus.INTERRUPTED,
            operation=cast(Any, "evaluate_transcript"),
            disposition=self.disposition,
            signal=None,
            interrupted_context=None,
            playback_result=playback_result,
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
    assert (
        tts.last_response_text
        == "generated_from_transcript::jarvis final user request"
    )


def test_voice_session_loop_finalizes_holding_completion_without_partial_loop() -> None:
    stt = FakeSTT()
    cognition = FakeCognition()
    tts = FakeTTS()
    playback = FakePlayback()

    runtime = VoiceSessionLoopRuntime(
        config=VoiceRuntimeConfig(max_silence_ms=40),
        microphone=FakeMicrophone(frames=[_frame(), _frame(), _frame()]),
        vad=FakeVad(
            [
                VoiceActivityDecision.SPEECH_STARTED,
                VoiceActivityDecision.HOLDING_FOR_COMPLETION,
                VoiceActivityDecision.HOLDING_FOR_COMPLETION,
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
    holding = runtime.process_once()
    result = runtime.process_once()

    assert holding.message == "holding for speech completion"
    assert holding.event is None
    assert result.event == VoiceSessionLoopEvent.PLAYBACK_FINISHED
    assert stt.partials == 0
    assert stt.finals == 1
    assert cognition.responses == 1
    assert playback.played == 1


def test_voice_session_loop_prefetches_from_wake_partial() -> None:
    stt = FakeSTTEmptyFinalWithWake()
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


def test_voice_session_loop_does_not_prefetch_unattended_partial() -> None:
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
    assert cognition.prefetches == 0


def test_voice_session_loop_rejects_background_final_before_cognition() -> None:
    stt = FakeSTTBackgroundFinal()
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

    assert result.event is None
    assert result.message == "transcript rejected by attention gate"
    gate = cast(dict[str, object], result.metadata["transcript_gate"])
    assert gate["reason"] == "requires_wake_or_active_attention"
    assert cognition.responses == 0
    assert tts.calls == 0
    assert playback.played == 0
    assert snapshot.responses == 0


def test_voice_session_loop_companion_policy_accepts_natural_no_wake_speech() -> None:
    stt = FakeSTTNaturalFinalWithoutWake()
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
        transcript_gate_policy=TranscriptAttentionGatePolicy(
            min_final_confidence=0.55,
            min_words_without_wake=2,
            require_attention_for_promoted_partials=False,
            require_wake_or_attention=False,
        ),
        policy=VoiceSessionLoopPolicy(partial_transcript_every_frames=2),
    )

    runtime.start()
    runtime.process_once()
    runtime.process_once()
    result = runtime.process_once()
    snapshot = runtime.snapshot()

    assert result.event == VoiceSessionLoopEvent.PLAYBACK_FINISHED
    assert cognition.responses == 1
    assert tts.last_response_text == (
        "generated_from_transcript::explain artificial intelligence briefly"
    )
    assert playback.played == 1
    assert snapshot.responses == 1


def test_voice_session_loop_accepts_one_word_followup_in_active_conversation() -> None:
    cognition = FakeCognition()
    tts = FakeTTS()
    playback = FakePlayback()

    runtime = VoiceSessionLoopRuntime(
        microphone=FakeMicrophone(),
        vad=FakeVad([VoiceActivityDecision.SILENCE]),
        stt=FakeSTT(),
        cognition=cognition,
        tts=tts,
        playback=playback,
        barge_in=FakeBargeIn(),
        health=FakeHealth(),
        transcript_gate_policy=TranscriptAttentionGatePolicy(
            min_words_without_wake=2,
            min_words_when_attention_active=1,
            require_attention_for_promoted_partials=False,
            require_wake_or_attention=False,
        ),
    )

    runtime.start()
    first = runtime.handle_transcript(
        _transcript("can you hear me", kind=VoiceTranscriptKind.FINAL)
    )
    followup = runtime.handle_transcript(
        _transcript("explain", kind=VoiceTranscriptKind.FINAL)
    )

    assert first.event == VoiceSessionLoopEvent.PLAYBACK_FINISHED
    assert followup.event == VoiceSessionLoopEvent.PLAYBACK_FINISHED
    assert cognition.responses == 2
    assert tts.last_response_text == "generated_from_transcript::explain"


def test_voice_session_loop_router_holds_unstable_transcript_before_cognition() -> None:
    cognition = FakeCognition()
    tts = FakeTTS()
    playback = FakePlayback()

    runtime = VoiceSessionLoopRuntime(
        microphone=FakeMicrophone(),
        vad=FakeVad([VoiceActivityDecision.SILENCE]),
        stt=FakeSTT(),
        cognition=cognition,
        tts=tts,
        playback=playback,
        barge_in=FakeBargeIn(),
        health=FakeHealth(),
        transcript_gate_policy=TranscriptAttentionGatePolicy(
            min_words_without_wake=1,
            require_attention_for_promoted_partials=False,
            require_wake_or_attention=False,
        ),
        cognitive_router=VoiceCognitiveRouter(
            policy=VoiceCognitiveRouterPolicy(min_stable_confidence=0.80)
        ),
    )

    runtime.start()
    result = runtime.handle_transcript(
        _transcript(
            "explain",
            kind=VoiceTranscriptKind.FINAL,
            confidence=0.66,
            metadata={"stability": 0.40},
        )
    )
    route = cast(dict[str, object], result.metadata["cognitive_route"])
    perception = cast(dict[str, object], result.metadata["perception"])

    assert result.message == "transcript held by cognitive router"
    assert route["action"] == "wait_for_stability"
    assert route["stability"] == 0.40
    assert perception["intent_state"] == "stabilizing"
    assert result.transcript is not None
    assert result.transcript.metadata["perception_stability"] == 0.40
    assert cognition.responses == 0
    assert tts.calls == 0
    assert playback.played == 0


def test_voice_session_loop_tracks_async_playback_until_completion() -> None:
    playback = FakeAsyncPlayback()

    runtime = VoiceSessionLoopRuntime(
        microphone=FakeMicrophone(),
        vad=FakeVad([VoiceActivityDecision.SILENCE]),
        stt=FakeSTT(),
        cognition=FakeCognition(),
        tts=FakeTTS(),
        playback=playback,
        barge_in=FakeBargeIn(),
        health=FakeHealth(),
    )

    runtime.start()
    started = runtime.handle_transcript(
        _transcript("jarvis are you online", kind=VoiceTranscriptKind.FINAL)
    )
    speaking = runtime.snapshot()
    playback.complete()
    finished = runtime.process_once()
    done = runtime.snapshot()

    assert started.event == VoiceSessionLoopEvent.RESPONSE_READY
    assert speaking.assistant_speaking is True
    assert finished.event == VoiceSessionLoopEvent.PLAYBACK_FINISHED
    assert done.assistant_speaking is False
    assert done.played_outputs == 1


def test_voice_session_loop_keeps_playback_on_speech_start() -> None:
    playback = FakeAsyncPlayback()

    runtime = VoiceSessionLoopRuntime(
        microphone=FakeMicrophone(frames=[_frame()]),
        vad=FakeVad([VoiceActivityDecision.SPEECH_STARTED]),
        stt=FakeSTT(),
        cognition=FakeCognition(),
        tts=FakeTTS(),
        playback=playback,
        barge_in=FakeBargeIn(),
        health=FakeHealth(),
    )

    runtime.start()
    started = runtime.handle_transcript(
        _transcript("jarvis explain AGI", kind=VoiceTranscriptKind.FINAL)
    )
    interrupted = runtime.process_once()
    snapshot = runtime.snapshot()

    assert started.event == VoiceSessionLoopEvent.RESPONSE_READY
    assert interrupted.event == VoiceSessionLoopEvent.SPEECH_STARTED
    assert playback.stopped is False
    assert snapshot.interruptions == 0
    assert snapshot.assistant_speaking is True
    assert snapshot.status == VoiceSessionLoopStatus.USER_SPEAKING


def test_voice_session_loop_can_opt_into_speech_start_interruption() -> None:
    playback = FakeAsyncPlayback()

    runtime = VoiceSessionLoopRuntime(
        microphone=FakeMicrophone(frames=[_frame()]),
        vad=FakeVad([VoiceActivityDecision.SPEECH_STARTED]),
        stt=FakeSTT(),
        cognition=FakeCognition(),
        tts=FakeTTS(),
        playback=playback,
        barge_in=FakeBargeIn(),
        health=FakeHealth(),
        policy=VoiceSessionLoopPolicy(interrupt_playback_on_speech_start=True),
    )

    runtime.start()
    started = runtime.handle_transcript(
        _transcript("jarvis explain AGI", kind=VoiceTranscriptKind.FINAL)
    )
    interrupted = runtime.process_once()
    snapshot = runtime.snapshot()

    assert started.event == VoiceSessionLoopEvent.RESPONSE_READY
    assert interrupted.event == VoiceSessionLoopEvent.BARGE_IN_INTERRUPTED
    assert playback.stopped is True
    assert snapshot.interruptions == 1
    assert snapshot.assistant_speaking is False
    assert snapshot.status == VoiceSessionLoopStatus.USER_SPEAKING


def test_voice_session_loop_routes_availability_ping_to_cognition() -> None:
    cognition = FakeCognition()
    tts = FakeTTS()
    playback = FakePlayback()

    runtime = VoiceSessionLoopRuntime(
        microphone=FakeMicrophone(),
        vad=FakeVad([VoiceActivityDecision.SILENCE]),
        stt=FakeSTT(),
        cognition=cognition,
        reflex=VoiceReflexResponseRuntime(),
        tts=tts,
        playback=playback,
        barge_in=FakeBargeIn(),
        health=FakeHealth(),
    )

    runtime.start()
    result = runtime.handle_transcript(
        _transcript("jarvis can you hear me", kind=VoiceTranscriptKind.FINAL)
    )
    snapshot = runtime.snapshot()

    assert result.event == VoiceSessionLoopEvent.PLAYBACK_FINISHED
    assert cognition.responses == 1
    assert tts.last_response_text == (
        "generated_from_transcript::jarvis can you hear me"
    )
    assert playback.played == 1
    assert snapshot.responses == 1
    assert snapshot.metadata["reflex_responses"] == 0
    assert snapshot.metadata["last_reflex_kind"] is None
    assert result.metadata["response_origin"] == "cognition_response_boundary"


def test_voice_session_loop_reflex_stops_async_playback_fast() -> None:
    playback = FakeAsyncPlayback()
    cognition = FakeCognition()

    runtime = VoiceSessionLoopRuntime(
        microphone=FakeMicrophone(),
        vad=FakeVad([VoiceActivityDecision.SILENCE]),
        stt=FakeSTT(),
        cognition=cognition,
        reflex=VoiceReflexResponseRuntime(),
        tts=FakeTTS(),
        playback=playback,
        barge_in=FakeBargeIn(),
        health=FakeHealth(),
        policy=VoiceSessionLoopPolicy(allow_barge_in=False),
    )

    runtime.start()
    first = runtime.handle_transcript(
        _transcript("jarvis explain AGI", kind=VoiceTranscriptKind.FINAL)
    )
    stopped = runtime.handle_transcript(
        _transcript("stop", kind=VoiceTranscriptKind.FINAL)
    )
    snapshot = runtime.snapshot()

    assert first.event == VoiceSessionLoopEvent.RESPONSE_READY
    assert stopped.event == VoiceSessionLoopEvent.BARGE_IN_INTERRUPTED
    assert playback.stopped is True
    assert cognition.responses == 1
    assert snapshot.assistant_speaking is False
    assert snapshot.interruptions == 1
    assert stopped.metadata["response_origin"] == "voice_reflex_operational"


def test_voice_session_loop_shutdown_word_stops_runtime() -> None:
    microphone = FakeMicrophone()
    playback = FakePlayback()
    cognition = FakeCognition()

    runtime = VoiceSessionLoopRuntime(
        microphone=microphone,
        vad=FakeVad([VoiceActivityDecision.SILENCE]),
        stt=FakeSTT(),
        cognition=cognition,
        reflex=VoiceReflexResponseRuntime(),
        tts=FakeTTS(),
        playback=playback,
        barge_in=FakeBargeIn(),
        health=FakeHealth(),
    )

    runtime.start()
    result = runtime.handle_transcript(
        _transcript("jarvis shut down", kind=VoiceTranscriptKind.FINAL)
    )
    snapshot = runtime.snapshot()

    assert result.status == VoiceSessionLoopStatus.STOPPED
    assert result.event == VoiceSessionLoopEvent.STOPPED
    assert result.message == "voice session shutdown by voice command"
    assert microphone.stopped is True
    assert playback.stopped is True
    assert cognition.responses == 0
    assert snapshot.running is False
    assert snapshot.status == VoiceSessionLoopStatus.STOPPED
    assert snapshot.metadata["last_reflex_kind"] == "shutdown_session"
    assert result.metadata["shutdown_word"] is True


def test_voice_session_loop_stops_async_playback_and_answers_barge_in() -> None:
    playback = FakeAsyncPlayback()
    barge_in = FakeBargeIn(disposition=VoiceBargeInDisposition.NEW_QUESTION)
    cognition = FakeCognition()
    tts = FakeTTS()

    runtime = VoiceSessionLoopRuntime(
        microphone=FakeMicrophone(),
        vad=FakeVad([VoiceActivityDecision.SILENCE]),
        stt=FakeSTT(),
        cognition=cognition,
        tts=tts,
        playback=playback,
        barge_in=barge_in,
        health=FakeHealth(),
        transcript_gate_policy=TranscriptAttentionGatePolicy(
            min_words_without_wake=2,
            require_attention_for_promoted_partials=False,
            require_wake_or_attention=False,
        ),
    )

    runtime.start()
    first = runtime.handle_transcript(
        _transcript("jarvis explain AGI", kind=VoiceTranscriptKind.FINAL)
    )
    interrupt = runtime.handle_transcript(
        _transcript("what is AI", kind=VoiceTranscriptKind.FINAL)
    )

    assert first.event == VoiceSessionLoopEvent.RESPONSE_READY
    assert interrupt.event == VoiceSessionLoopEvent.RESPONSE_READY
    assert playback.stopped is True
    assert barge_in.interrupts == 1
    assert cognition.responses == 2
    assert tts.last_response_text == "generated_from_transcript::what is AI"


def test_voice_session_loop_ignores_assistant_echo_during_async_playback() -> None:
    playback = FakeAsyncPlayback()

    runtime = VoiceSessionLoopRuntime(
        microphone=FakeMicrophone(),
        vad=FakeVad([VoiceActivityDecision.SILENCE]),
        stt=FakeSTT(),
        cognition=FakeCognition(),
        tts=FakeTTS(),
        playback=playback,
        barge_in=cast(Any, VoiceBargeInRuntime()),
        health=FakeHealth(),
    )

    runtime.start()
    started = runtime.handle_transcript(
        _transcript("jarvis explain AGI", kind=VoiceTranscriptKind.FINAL)
    )
    runtime._last_response_text = (  # noqa: SLF001
        "What I can do is explain artificial intelligence briefly."
    )
    echo = runtime.handle_transcript(
        _transcript(
            "what i can do is explain artificial intelligence",
            kind=VoiceTranscriptKind.FINAL,
        )
    )
    snapshot = runtime.snapshot()

    assert started.event == VoiceSessionLoopEvent.RESPONSE_READY
    assert echo.event is None
    assert echo.barge_in_result is not None
    assert echo.barge_in_result.status == VoiceBargeInRuntimeStatus.IGNORED
    assert playback.stopped is False
    assert snapshot.assistant_speaking is True
    assert snapshot.interruptions == 0


def test_voice_session_loop_snapshot_feeds_live_spine_monitor() -> None:
    runtime = VoiceSessionLoopRuntime(
        microphone=FakeMicrophone(frames=[_frame(), _frame(), _frame()]),
        vad=FakeVad(
            [
                VoiceActivityDecision.SPEECH_STARTED,
                VoiceActivityDecision.SPEECH_CONTINUED,
                VoiceActivityDecision.SPEECH_ENDED,
            ]
        ),
        stt=FakeSTT(),
        cognition=FakeCognition(),
        tts=FakeTTS(),
        playback=FakePlayback(),
        barge_in=FakeBargeIn(),
        health=FakeHealth(),
        policy=VoiceSessionLoopPolicy(partial_transcript_every_frames=2),
    )
    monitor = VoiceLiveSpineMonitor()

    runtime.start()
    runtime.process_once()
    runtime.process_once()
    runtime.process_once()
    report = monitor.inspect(runtime.snapshot())

    assert report.status == VoiceLiveSpineStatus.HEALTHY
    assert report.checks["fsm_clean"] is True
    assert report.checks["perception_available"] is True


def test_voice_session_loop_restores_wake_partial_when_final_loses_wake() -> None:
    stt = FakeSTTWakePartialLostWakeFinal()
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
    assert cognition.responses == 1
    assert tts.calls == 1
    assert playback.played == 1
    assert snapshot.responses == 1
    assert result.transcript is not None
    assert result.transcript.text == "jarvis are you online"
    assert result.transcript.metadata["resolved_from_wake_partial"] is True
    assert tts.last_response_text == (
        "generated_from_transcript::jarvis are you online"
    )


def test_voice_session_loop_promotes_wake_partial_when_final_is_empty() -> None:
    stt = FakeSTTEmptyFinalWithWake()
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
    assert stt.partials == 1
    assert stt.finals == 1
    assert cognition.responses == 1
    assert tts.calls == 1
    assert playback.played == 1
    assert snapshot.responses == 1
    assert result.transcript is not None
    assert result.transcript.kind == VoiceTranscriptKind.FINAL
    assert result.transcript.metadata["promoted_from_partial"] is True
    assert tts.last_response_text == "generated_from_transcript::jarvis are you online"


def test_voice_session_loop_rejects_unattended_partial_when_final_is_empty() -> None:
    stt = FakeSTTEmptyFinal()
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

    assert result.event is None
    assert result.message == "transcript rejected by attention gate"
    assert stt.partials == 1
    assert stt.finals == 1
    assert cognition.responses == 0
    assert tts.calls == 0
    assert playback.played == 0
    assert snapshot.responses == 0
    assert result.transcript is not None
    assert result.transcript.kind == VoiceTranscriptKind.FINAL
    assert result.transcript.metadata["promoted_from_partial"] is True
    gate = cast(dict[str, object], result.metadata["transcript_gate"])
    assert gate["reason"] == "promoted_partial_without_attention"


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


def test_voice_session_loop_answers_question_after_barge_in() -> None:
    barge_in = FakeBargeIn(disposition=VoiceBargeInDisposition.NEW_QUESTION)
    cognition = FakeCognition()
    tts = FakeTTS()
    playback = FakePlayback()

    runtime = VoiceSessionLoopRuntime(
        microphone=FakeMicrophone(),
        vad=FakeVad([VoiceActivityDecision.SILENCE]),
        stt=FakeSTT(),
        cognition=cognition,
        tts=tts,
        playback=playback,
        barge_in=barge_in,
        health=FakeHealth(),
        transcript_gate_policy=TranscriptAttentionGatePolicy(
            min_words_without_wake=2,
            require_attention_for_promoted_partials=False,
            require_wake_or_attention=False,
        ),
    )

    runtime.start()
    runtime._assistant_speaking = True  # noqa: SLF001
    runtime._last_response_text = "AGI is artificial general intelligence."  # noqa: SLF001
    result = runtime.handle_transcript(
        _transcript("what is AI", kind=VoiceTranscriptKind.FINAL)
    )

    assert result.event == VoiceSessionLoopEvent.PLAYBACK_FINISHED
    assert result.barge_in_result is not None
    assert result.metadata["continued_after_barge_in"] is True
    threaded = cast(dict[str, object], result.metadata["threaded_interruption"])
    assert threaded["interruption_text"] == "what is AI"
    assert threaded["paused_response_text"] == "AGI is artificial general intelligence."
    assert barge_in.interrupts == 1
    assert cognition.responses == 1
    assert cognition.requests[0].working_memory_context
    resume_context = cognition.requests[0].working_memory_context[0]
    assert "Answer the user's current side question first" in resume_context
    assert "AGI is artificial general intelligence." in resume_context
    assert "what is AI" in resume_context
    assert tts.last_response_text == "generated_from_transcript::what is AI"
    snapshot = runtime.snapshot()
    assert snapshot.interruptions == 1
    threaded_snapshot = cast(
        dict[str, object],
        snapshot.metadata["threaded_interruption"],
    )
    assert threaded_snapshot["active_depth"] == 1


def test_voice_session_loop_resumes_thread_after_speech_start_interruption() -> None:
    playback = FakeAsyncPlayback()
    cognition = FakeCognition()

    runtime = VoiceSessionLoopRuntime(
        microphone=FakeMicrophone(frames=[_frame()]),
        vad=FakeVad([VoiceActivityDecision.SPEECH_STARTED]),
        stt=FakeSTT(),
        cognition=cognition,
        tts=FakeTTS(),
        playback=playback,
        barge_in=FakeBargeIn(disposition=VoiceBargeInDisposition.NEW_QUESTION),
        health=FakeHealth(),
        transcript_gate_policy=TranscriptAttentionGatePolicy(
            min_words_without_wake=2,
            require_attention_for_promoted_partials=False,
            require_wake_or_attention=False,
        ),
        policy=VoiceSessionLoopPolicy(interrupt_playback_on_speech_start=True),
    )

    runtime.start()
    started = runtime.handle_transcript(
        _transcript("jarvis explain AGI", kind=VoiceTranscriptKind.FINAL)
    )
    interrupted = runtime.process_once()
    side_answer = runtime.handle_transcript(
        _transcript("what is AI", kind=VoiceTranscriptKind.FINAL)
    )

    assert started.event == VoiceSessionLoopEvent.RESPONSE_READY
    assert interrupted.event == VoiceSessionLoopEvent.BARGE_IN_INTERRUPTED
    assert side_answer.event == VoiceSessionLoopEvent.RESPONSE_READY
    assert playback.stopped is True
    assert cognition.responses == 2
    assert cognition.requests[1].working_memory_context
    resume_context = cognition.requests[1].working_memory_context[0]
    assert "generated_from_transcript::jarvis explain AGI" in resume_context
    assert "what is AI" in resume_context
    threaded = cast(
        dict[str, object],
        side_answer.metadata["threaded_interruption"],
    )
    assert threaded["interruption_kind"] == "speech_start"


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
