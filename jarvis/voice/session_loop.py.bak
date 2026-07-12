from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol, cast

from jarvis.voice.awareness_cognition_bridge import (
    VoiceAwarenessCognitionBridge,
)
from jarvis.voice.barge_in_runtime import (
    VoiceBargeInPlaybackController,
    VoiceBargeInRequest,
    VoiceBargeInResult,
    VoiceBargeInRuntime,
)
from jarvis.voice.cognition_response import (
    VoiceCognitionRequest,
    VoiceCognitionResult,
)
from jarvis.voice.contracts import (
    VoiceInputFrame,
    VoiceRuntimeConfig,
    VoiceTranscript,
    VoiceTranscriptKind,
    default_voice_runtime_config,
    utc_now,
)
from jarvis.voice.health_recovery import (
    VoiceActivityHealthComponent,
    VoiceBargeInHealthComponent,
    VoiceCognitionHealthComponent,
    VoiceHealthComponents,
    VoiceHealthRecoveryRuntime,
    VoiceHealthResult,
    VoiceHealthStatus,
    VoiceMicrophoneHealthComponent,
    VoicePlaybackHealthComponent,
    VoiceSTTHealthComponent,
    VoiceTTSHealthComponent,
)
from jarvis.voice.microphone_capture import (
    VoiceMicrophoneCaptureResult,
    VoiceMicrophoneCaptureRuntime,
    VoiceMicrophoneCaptureStatus,
)
from jarvis.voice.playback_runtime import (
    VoicePlaybackResult,
    VoicePlaybackRuntime,
)
from jarvis.voice.stt_runtime import (
    VoiceSTTResult,
    VoiceSTTRuntime,
)
from jarvis.voice.tts_runtime import (
    VoiceTTSResult,
    VoiceTTSRuntime,
)
from jarvis.voice.voice_activity import (
    VoiceActivityDecision,
    VoiceActivityResult,
    VoiceActivityRuntime,
)


class VoiceSessionLoopStatus(StrEnum):
    CREATED = "created"
    STARTING = "starting"
    LISTENING = "listening"
    USER_SPEAKING = "user_speaking"
    THINKING = "thinking"
    SYNTHESIZING = "synthesizing"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    RECOVERING = "recovering"
    STOPPING = "stopping"
    STOPPED = "stopped"
    DEGRADED = "degraded"
    FAILED = "failed"


class VoiceSessionLoopOperation(StrEnum):
    START = "start"
    PROCESS_ONCE = "process_once"
    PROCESS_FRAME = "process_frame"
    HANDLE_TRANSCRIPT = "handle_transcript"
    HANDLE_BARGE_IN = "handle_barge_in"
    RUN = "run"
    STOP = "stop"
    SNAPSHOT = "snapshot"


class VoiceSessionLoopEvent(StrEnum):
    STARTED = "started"
    FRAME_CAPTURED = "frame_captured"
    SPEECH_STARTED = "speech_started"
    PARTIAL_TRANSCRIPT = "partial_transcript"
    FINAL_TRANSCRIPT = "final_transcript"
    RESPONSE_READY = "response_ready"
    TTS_READY = "tts_ready"
    PLAYBACK_FINISHED = "playback_finished"
    BARGE_IN_INTERRUPTED = "barge_in_interrupted"
    HEALTH_RECOVERED = "health_recovered"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class VoiceSessionLoopPolicy:
    partial_transcript_every_frames: int = 12
    health_check_every_cycles: int = 50
    idle_sleep_seconds: float = 0.01
    max_consecutive_failures: int = 3
    auto_recover: bool = True
    allow_barge_in: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.partial_transcript_every_frames < 1:
            raise ValueError("partial_transcript_every_frames must be positive.")
        if self.health_check_every_cycles < 1:
            raise ValueError("health_check_every_cycles must be positive.")
        if self.idle_sleep_seconds < 0:
            raise ValueError("idle_sleep_seconds cannot be negative.")
        if self.max_consecutive_failures < 1:
            raise ValueError("max_consecutive_failures must be positive.")


@dataclass(frozen=True, slots=True)
class VoiceSessionLoopResult:
    status: VoiceSessionLoopStatus
    operation: VoiceSessionLoopOperation
    event: VoiceSessionLoopEvent | None
    transcript: VoiceTranscript | None
    cognition_result: VoiceCognitionResult | None
    tts_result: VoiceTTSResult | None
    playback_result: VoicePlaybackResult | None
    barge_in_result: VoiceBargeInResult | None
    health_result: VoiceHealthResult | None
    message: str
    latency_ms: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status not in {
            VoiceSessionLoopStatus.DEGRADED,
            VoiceSessionLoopStatus.FAILED,
        }


@dataclass(frozen=True, slots=True)
class VoiceSessionLoopSnapshot:
    status: VoiceSessionLoopStatus
    running: bool
    assistant_speaking: bool
    cycles: int
    captured_frames: int
    speech_segments: int
    partial_transcripts: int
    final_transcripts: int
    responses: int
    tts_outputs: int
    played_outputs: int
    interruptions: int
    recoveries: int
    consecutive_failures: int
    buffered_segment_frames: int
    last_event: VoiceSessionLoopEvent | None
    last_transcript_text: str | None
    last_response_text: str | None
    last_latency_ms: float | None
    last_error: str | None
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


class VoiceSessionMicrophone(Protocol):
    def start(self) -> object:
        raise NotImplementedError

    def capture_once(self) -> VoiceMicrophoneCaptureResult:
        raise NotImplementedError

    def stop(self) -> object:
        raise NotImplementedError

    def snapshot(self) -> object:
        raise NotImplementedError


class VoiceSessionVad(Protocol):
    def analyze_frame(self, frame: VoiceInputFrame) -> VoiceActivityResult:
        raise NotImplementedError

    def reset(self) -> object:
        raise NotImplementedError

    def snapshot(self) -> object:
        raise NotImplementedError


class VoiceSessionSTT(Protocol):
    def prepare(self) -> object:
        raise NotImplementedError

    def transcribe_partial(
        self,
        frames: tuple[VoiceInputFrame, ...],
    ) -> VoiceSTTResult:
        raise NotImplementedError

    def transcribe_final(
        self,
        frames: tuple[VoiceInputFrame, ...],
        *,
        allow_action_candidate: bool = False,
    ) -> VoiceSTTResult:
        raise NotImplementedError

    def reset(self) -> object:
        raise NotImplementedError

    def snapshot(self) -> object:
        raise NotImplementedError


class VoiceSessionCognition(Protocol):
    def prepare(self, *, user_label: str, assistant_name: str) -> object:
        raise NotImplementedError

    def prefetch_from_partial(
        self,
        request: VoiceCognitionRequest,
    ) -> object:
        raise NotImplementedError

    def think_from_transcript(
        self,
        request: VoiceCognitionRequest,
    ) -> VoiceCognitionResult:
        raise NotImplementedError

    def snapshot(self) -> object:
        raise NotImplementedError


class VoiceSessionTTS(Protocol):
    def prepare(self) -> object:
        raise NotImplementedError

    def synthesize_response(
        self,
        *,
        response: object,
        session_id: object,
        voice: str | None = None,
    ) -> VoiceTTSResult:
        raise NotImplementedError

    def reset(self) -> object:
        raise NotImplementedError

    def snapshot(self) -> object:
        raise NotImplementedError


class VoiceSessionPlayback(Protocol):
    def prepare(self) -> object:
        raise NotImplementedError

    def enqueue_chunks(self, chunks: object) -> VoicePlaybackResult:
        raise NotImplementedError

    def play_all(self) -> VoicePlaybackResult:
        raise NotImplementedError

    def stop(self) -> object:
        raise NotImplementedError

    def reset(self) -> object:
        raise NotImplementedError

    def snapshot(self) -> object:
        raise NotImplementedError


class VoiceSessionBargeIn(Protocol):
    def prepare(self, playback: object | None = None) -> object:
        raise NotImplementedError

    def evaluate_transcript(
        self,
        request: VoiceBargeInRequest,
    ) -> VoiceBargeInResult:
        raise NotImplementedError

    def reset(self) -> object:
        raise NotImplementedError

    def snapshot(self) -> object:
        raise NotImplementedError


class VoiceSessionHealth(Protocol):
    def check(self) -> VoiceHealthResult:
        raise NotImplementedError

    def recover(self) -> VoiceHealthResult:
        raise NotImplementedError


class VoiceSessionLoopRuntime:
    """
    Step 51J long-running voice session loop.

    This class connects the voice organs:
    microphone -> VAD -> STT -> cognition -> TTS -> playback.

    It never creates conversational words.
    All user-facing speech must already come from 51E cognition / response
    boundary before this runtime sends it to TTS and playback.
    """

    def __init__(
        self,
        *,
        config: VoiceRuntimeConfig | None = None,
        microphone: VoiceSessionMicrophone | None = None,
        vad: VoiceSessionVad | None = None,
        stt: VoiceSessionSTT | None = None,
        cognition: VoiceSessionCognition | None = None,
        tts: VoiceSessionTTS | None = None,
        playback: VoiceSessionPlayback | None = None,
        barge_in: VoiceSessionBargeIn | None = None,
        health: VoiceSessionHealth | None = None,
        policy: VoiceSessionLoopPolicy | None = None,
    ) -> None:
        self._config = config or default_voice_runtime_config()
        self._microphone = microphone or VoiceMicrophoneCaptureRuntime(
            config=self._config
        )
        self._vad = vad or VoiceActivityRuntime()
        self._stt = stt or VoiceSTTRuntime(config=self._config)
        self._cognition = cognition or VoiceAwarenessCognitionBridge()
        self._tts = tts or VoiceTTSRuntime(config=self._config)
        self._playback = playback or VoicePlaybackRuntime(config=self._config)
        self._barge_in = barge_in or VoiceBargeInRuntime()
        self._policy = policy or VoiceSessionLoopPolicy()

        self._health = health or VoiceHealthRecoveryRuntime(
            components=VoiceHealthComponents(
                microphone=cast(VoiceMicrophoneHealthComponent, self._microphone),
                vad=cast(VoiceActivityHealthComponent, self._vad),
                stt=cast(VoiceSTTHealthComponent, self._stt),
                cognition=cast(VoiceCognitionHealthComponent, self._cognition),
                tts=cast(VoiceTTSHealthComponent, self._tts),
                playback=cast(VoicePlaybackHealthComponent, self._playback),
                barge_in=cast(VoiceBargeInHealthComponent, self._barge_in),
            )
        )

        self._status = VoiceSessionLoopStatus.CREATED
        self._running = False
        self._assistant_speaking = False
        self._cycles = 0
        self._captured_frames = 0
        self._speech_segments = 0
        self._partial_transcripts = 0
        self._final_transcripts = 0
        self._responses = 0
        self._tts_outputs = 0
        self._played_outputs = 0
        self._interruptions = 0
        self._recoveries = 0
        self._consecutive_failures = 0
        self._segment_frames: list[VoiceInputFrame] = []
        self._last_event: VoiceSessionLoopEvent | None = None
        self._last_transcript_text: str | None = None
        self._last_response_text: str | None = None
        self._last_latency_ms: float | None = None
        self._last_error: str | None = None

    def start(self) -> VoiceSessionLoopResult:
        started = time.perf_counter()
        self._status = VoiceSessionLoopStatus.STARTING

        try:
            self._microphone.start()
            self._stt.prepare()
            self._cognition.prepare(
                user_label=self._config.user_label,
                assistant_name=self._config.assistant_name,
            )
            self._tts.prepare()
            self._playback.prepare()
            self._barge_in.prepare(
                cast(VoiceBargeInPlaybackController, self._playback)
            )
        except Exception as exc:
            self._status = VoiceSessionLoopStatus.FAILED
            self._last_error = str(exc)
            return self._result(
                operation=VoiceSessionLoopOperation.START,
                event=VoiceSessionLoopEvent.ERROR,
                message="voice session start failed",
                started=started,
                metadata={"error": str(exc)},
            )

        self._running = True
        self._status = VoiceSessionLoopStatus.LISTENING
        self._last_error = None
        return self._result(
            operation=VoiceSessionLoopOperation.START,
            event=VoiceSessionLoopEvent.STARTED,
            message="voice session started",
            started=started,
        )

    def process_once(self) -> VoiceSessionLoopResult:
        started = time.perf_counter()
        self._cycles += 1

        if not self._running:
            return self._result(
                operation=VoiceSessionLoopOperation.PROCESS_ONCE,
                event=None,
                message="voice session is not running",
                started=started,
            )

        if self._should_health_check():
            health_result = self._check_and_recover_health()
            if health_result.status in {
                VoiceHealthStatus.FAILED,
                VoiceHealthStatus.CRITICAL,
            }:
                self._status = VoiceSessionLoopStatus.RECOVERING
                return self._result(
                    operation=VoiceSessionLoopOperation.PROCESS_ONCE,
                    event=VoiceSessionLoopEvent.HEALTH_RECOVERED,
                    message="voice health recovery required",
                    started=started,
                    health_result=health_result,
                )

        try:
            capture = self._microphone.capture_once()
        except Exception as exc:
            return self._failure(
                operation=VoiceSessionLoopOperation.PROCESS_ONCE,
                started=started,
                message="microphone capture raised",
                error=str(exc),
            )

        if (
            capture.status == VoiceMicrophoneCaptureStatus.FAILED
            or capture.frame is None
        ):
            return self._handle_capture_failure(
                started=started,
                capture=capture,
            )

        self._captured_frames += 1
        return self.process_frame(capture.frame)

    def process_frame(
        self,
        frame: VoiceInputFrame,
    ) -> VoiceSessionLoopResult:
        started = time.perf_counter()

        try:
            activity = self._vad.analyze_frame(frame)
        except Exception as exc:
            return self._failure(
                operation=VoiceSessionLoopOperation.PROCESS_FRAME,
                started=started,
                message="voice activity analysis raised",
                error=str(exc),
            )

        if activity.decision == VoiceActivityDecision.SILENCE:
            self._status = (
                VoiceSessionLoopStatus.SPEAKING
                if self._assistant_speaking
                else VoiceSessionLoopStatus.LISTENING
            )
            return self._result(
                operation=VoiceSessionLoopOperation.PROCESS_FRAME,
                event=None,
                message="silence frame processed",
                started=started,
            )

        if activity.decision == VoiceActivityDecision.SPEECH_STARTED:
            self._segment_frames = [frame]
            self._speech_segments += 1
            self._status = VoiceSessionLoopStatus.USER_SPEAKING
            return self._result(
                operation=VoiceSessionLoopOperation.PROCESS_FRAME,
                event=VoiceSessionLoopEvent.SPEECH_STARTED,
                message="speech segment started",
                started=started,
            )

        if activity.decision in {
            VoiceActivityDecision.SPEECH_CONTINUED,
            VoiceActivityDecision.HOLDING_FOR_COMPLETION,
        }:
            self._segment_frames.append(frame)
            if self._should_partial_transcribe():
                partial = self._transcribe_partial()
                return self._result(
                    operation=VoiceSessionLoopOperation.PROCESS_FRAME,
                    event=VoiceSessionLoopEvent.PARTIAL_TRANSCRIPT,
                    transcript=partial.transcript,
                    message="partial transcript processed",
                    started=started,
                    metadata={
                        "partial_status": partial.status.value,
                        "buffered_frames": len(self._segment_frames),
                    },
                )

            return self._result(
                operation=VoiceSessionLoopOperation.PROCESS_FRAME,
                event=None,
                message="speech frame buffered",
                started=started,
                metadata={"buffered_frames": len(self._segment_frames)},
            )

        if activity.decision == VoiceActivityDecision.SPEECH_ENDED:
            self._segment_frames.append(frame)
            return self._handle_final_segment(started=started)

        return self._result(
            operation=VoiceSessionLoopOperation.PROCESS_FRAME,
            event=None,
            message="voice activity decision ignored",
            started=started,
            metadata={"decision": activity.decision.value},
        )

    def handle_transcript(
        self,
        transcript: VoiceTranscript,
    ) -> VoiceSessionLoopResult:
        started = time.perf_counter()
        return self._handle_transcript(
            transcript=transcript,
            started=started,
        )

    def run(
        self,
        *,
        max_cycles: int | None = None,
        max_seconds: float | None = None,
    ) -> VoiceSessionLoopResult:
        started = time.perf_counter()

        if not self._running:
            start_result = self.start()
            if start_result.status == VoiceSessionLoopStatus.FAILED:
                return start_result

        cycles = 0
        last_result: VoiceSessionLoopResult | None = None

        while self._running:
            if max_cycles is not None and cycles >= max_cycles:
                break
            if max_seconds is not None:
                elapsed = time.perf_counter() - started
                if elapsed >= max_seconds:
                    break

            last_result = self.process_once()
            cycles += 1

            if (
                last_result.status == VoiceSessionLoopStatus.FAILED
                and self._consecutive_failures
                >= self._policy.max_consecutive_failures
            ):
                self._running = False
                break

            if self._policy.idle_sleep_seconds > 0:
                time.sleep(self._policy.idle_sleep_seconds)

        latency_ms = (time.perf_counter() - started) * 1000.0
        self._last_latency_ms = latency_ms

        return VoiceSessionLoopResult(
            status=self._status,
            operation=VoiceSessionLoopOperation.RUN,
            event=last_result.event if last_result is not None else None,
            transcript=last_result.transcript if last_result is not None else None,
            cognition_result=(
                last_result.cognition_result if last_result is not None else None
            ),
            tts_result=last_result.tts_result if last_result is not None else None,
            playback_result=(
                last_result.playback_result if last_result is not None else None
            ),
            barge_in_result=(
                last_result.barge_in_result if last_result is not None else None
            ),
            health_result=(
                last_result.health_result if last_result is not None else None
            ),
            message="voice session run completed",
            latency_ms=latency_ms,
            created_at=utc_now(),
            metadata={"cycles": cycles},
        )

    def stop(self) -> VoiceSessionLoopResult:
        started = time.perf_counter()
        self._status = VoiceSessionLoopStatus.STOPPING
        self._running = False

        try:
            self._playback.stop()
            self._microphone.stop()
        except Exception as exc:
            self._status = VoiceSessionLoopStatus.DEGRADED
            self._last_error = str(exc)
            return self._result(
                operation=VoiceSessionLoopOperation.STOP,
                event=VoiceSessionLoopEvent.ERROR,
                message="voice session stop degraded",
                started=started,
                metadata={"error": str(exc)},
            )

        self._assistant_speaking = False
        self._status = VoiceSessionLoopStatus.STOPPED
        return self._result(
            operation=VoiceSessionLoopOperation.STOP,
            event=VoiceSessionLoopEvent.STOPPED,
            message="voice session stopped",
            started=started,
        )

    def snapshot(self) -> VoiceSessionLoopSnapshot:
        return VoiceSessionLoopSnapshot(
            status=self._status,
            running=self._running,
            assistant_speaking=self._assistant_speaking,
            cycles=self._cycles,
            captured_frames=self._captured_frames,
            speech_segments=self._speech_segments,
            partial_transcripts=self._partial_transcripts,
            final_transcripts=self._final_transcripts,
            responses=self._responses,
            tts_outputs=self._tts_outputs,
            played_outputs=self._played_outputs,
            interruptions=self._interruptions,
            recoveries=self._recoveries,
            consecutive_failures=self._consecutive_failures,
            buffered_segment_frames=len(self._segment_frames),
            last_event=self._last_event,
            last_transcript_text=self._last_transcript_text,
            last_response_text=self._last_response_text,
            last_latency_ms=self._last_latency_ms,
            last_error=self._last_error,
            created_at=utc_now(),
        )

    def _handle_final_segment(
        self,
        *,
        started: float,
    ) -> VoiceSessionLoopResult:
        frames = tuple(self._segment_frames)
        self._segment_frames = []

        if not frames:
            return self._result(
                operation=VoiceSessionLoopOperation.PROCESS_FRAME,
                event=None,
                message="speech ended without frames",
                started=started,
            )

        try:
            stt_result = self._stt.transcribe_final(
                frames,
                allow_action_candidate=False,
            )
        except Exception as exc:
            return self._failure(
                operation=VoiceSessionLoopOperation.PROCESS_FRAME,
                started=started,
                message="final transcription raised",
                error=str(exc),
            )

        if stt_result.transcript is None:
            return self._result(
                operation=VoiceSessionLoopOperation.PROCESS_FRAME,
                event=None,
                message="final transcription returned no transcript",
                started=started,
                metadata={"stt_status": stt_result.status.value},
            )

        self._final_transcripts += 1
        self._last_transcript_text = stt_result.transcript.text
        return self._handle_transcript(
            transcript=stt_result.transcript,
            started=started,
        )

    def _handle_transcript(
        self,
        *,
        transcript: VoiceTranscript,
        started: float,
    ) -> VoiceSessionLoopResult:
        self._last_transcript_text = transcript.text

        if self._assistant_speaking and self._policy.allow_barge_in:
            return self._handle_barge_in(
                transcript=transcript,
                started=started,
            )

        if transcript.kind != VoiceTranscriptKind.FINAL:
            return self._result(
                operation=VoiceSessionLoopOperation.HANDLE_TRANSCRIPT,
                event=None,
                transcript=transcript,
                message="non-final transcript ignored for response",
                started=started,
            )

        self._status = VoiceSessionLoopStatus.THINKING
        cognition_result = self._cognition.think_from_transcript(
            VoiceCognitionRequest(
                transcript=transcript,
                user_label=self._config.user_label,
                assistant_name=self._config.assistant_name,
            )
        )

        if cognition_result.response is None:
            return self._result(
                operation=VoiceSessionLoopOperation.HANDLE_TRANSCRIPT,
                event=None,
                transcript=transcript,
                cognition_result=cognition_result,
                message="cognition produced no response",
                started=started,
            )

        self._responses += 1
        self._last_response_text = cognition_result.response.text
        self._status = VoiceSessionLoopStatus.SYNTHESIZING

        tts_result = self._tts.synthesize_response(
            response=cognition_result.response,
            session_id=transcript.session_id,
        )

        if not tts_result.chunks:
            return self._result(
                operation=VoiceSessionLoopOperation.HANDLE_TRANSCRIPT,
                event=None,
                transcript=transcript,
                cognition_result=cognition_result,
                tts_result=tts_result,
                message="TTS produced no chunks",
                started=started,
            )

        self._tts_outputs += 1
        self._status = VoiceSessionLoopStatus.SPEAKING
        self._assistant_speaking = True

        enqueue_result = self._playback.enqueue_chunks(tts_result.chunks)
        if not enqueue_result.succeeded:
            self._assistant_speaking = False
            self._status = VoiceSessionLoopStatus.DEGRADED
            return self._result(
                operation=VoiceSessionLoopOperation.HANDLE_TRANSCRIPT,
                event=None,
                transcript=transcript,
                cognition_result=cognition_result,
                tts_result=tts_result,
                playback_result=enqueue_result,
                message="playback enqueue failed",
                started=started,
            )

        playback_result = self._playback.play_all()
        self._assistant_speaking = False

        if playback_result.succeeded:
            self._played_outputs += 1
            self._status = VoiceSessionLoopStatus.LISTENING
            event = VoiceSessionLoopEvent.PLAYBACK_FINISHED
        else:
            self._status = VoiceSessionLoopStatus.DEGRADED
            event = VoiceSessionLoopEvent.ERROR

        return self._result(
            operation=VoiceSessionLoopOperation.HANDLE_TRANSCRIPT,
            event=event,
            transcript=transcript,
            cognition_result=cognition_result,
            tts_result=tts_result,
            playback_result=playback_result,
            message="voice response pipeline completed",
            started=started,
        )

    def _handle_barge_in(
        self,
        *,
        transcript: VoiceTranscript,
        started: float,
    ) -> VoiceSessionLoopResult:
        barge_in_result = self._barge_in.evaluate_transcript(
            VoiceBargeInRequest(
                transcript=transcript,
                assistant_speaking=self._assistant_speaking,
                active_response_text=self._last_response_text,
            )
        )

        if barge_in_result.interrupted:
            self._interruptions += 1
            self._assistant_speaking = False
            self._status = VoiceSessionLoopStatus.INTERRUPTED
            event = VoiceSessionLoopEvent.BARGE_IN_INTERRUPTED
        else:
            event = None

        return self._result(
            operation=VoiceSessionLoopOperation.HANDLE_BARGE_IN,
            event=event,
            transcript=transcript,
            barge_in_result=barge_in_result,
            message="barge-in processed",
            started=started,
        )

    def _transcribe_partial(self) -> VoiceSTTResult:
        partial = self._stt.transcribe_partial(tuple(self._segment_frames))

        if partial.transcript is not None:
            self._partial_transcripts += 1
            self._last_transcript_text = partial.transcript.text
            self._cognition.prefetch_from_partial(
                VoiceCognitionRequest(
                    transcript=partial.transcript,
                    user_label=self._config.user_label,
                    assistant_name=self._config.assistant_name,
                )
            )

        return partial

    def _should_partial_transcribe(self) -> bool:
        return (
            len(self._segment_frames) > 0
            and len(self._segment_frames)
            % self._policy.partial_transcript_every_frames
            == 0
        )

    def _should_health_check(self) -> bool:
        return self._cycles % self._policy.health_check_every_cycles == 0

    def _check_and_recover_health(self) -> VoiceHealthResult:
        health = self._health.check()
        if (
            self._policy.auto_recover
            and health.status
            in {
                VoiceHealthStatus.DEGRADED,
                VoiceHealthStatus.CRITICAL,
                VoiceHealthStatus.FAILED,
            }
        ):
            recovered = self._health.recover()
            self._recoveries += 1
            return recovered
        return health

    def _handle_capture_failure(
        self,
        *,
        started: float,
        capture: VoiceMicrophoneCaptureResult,
    ) -> VoiceSessionLoopResult:
        self._consecutive_failures += 1
        self._last_error = capture.message
        self._status = (
            VoiceSessionLoopStatus.FAILED
            if self._consecutive_failures
            >= self._policy.max_consecutive_failures
            else VoiceSessionLoopStatus.DEGRADED
        )
        return self._result(
            operation=VoiceSessionLoopOperation.PROCESS_ONCE,
            event=VoiceSessionLoopEvent.ERROR,
            message="microphone capture failed",
            started=started,
            metadata={
                "capture_status": capture.status.value,
                "capture_message": capture.message,
            },
        )

    def _failure(
        self,
        *,
        operation: VoiceSessionLoopOperation,
        started: float,
        message: str,
        error: str,
    ) -> VoiceSessionLoopResult:
        self._consecutive_failures += 1
        self._last_error = error
        self._status = (
            VoiceSessionLoopStatus.FAILED
            if self._consecutive_failures
            >= self._policy.max_consecutive_failures
            else VoiceSessionLoopStatus.DEGRADED
        )
        return self._result(
            operation=operation,
            event=VoiceSessionLoopEvent.ERROR,
            message=message,
            started=started,
            metadata={"error": error},
        )

    def _result(
        self,
        *,
        operation: VoiceSessionLoopOperation,
        event: VoiceSessionLoopEvent | None,
        message: str,
        started: float,
        transcript: VoiceTranscript | None = None,
        cognition_result: VoiceCognitionResult | None = None,
        tts_result: VoiceTTSResult | None = None,
        playback_result: VoicePlaybackResult | None = None,
        barge_in_result: VoiceBargeInResult | None = None,
        health_result: VoiceHealthResult | None = None,
        metadata: dict[str, object] | None = None,
    ) -> VoiceSessionLoopResult:
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._last_latency_ms = latency_ms

        if event is not None:
            self._last_event = event

        if self._status not in {
            VoiceSessionLoopStatus.DEGRADED,
            VoiceSessionLoopStatus.FAILED,
        }:
            self._consecutive_failures = 0

        return VoiceSessionLoopResult(
            status=self._status,
            operation=operation,
            event=event,
            transcript=transcript,
            cognition_result=cognition_result,
            tts_result=tts_result,
            playback_result=playback_result,
            barge_in_result=barge_in_result,
            health_result=health_result,
            message=message,
            latency_ms=latency_ms,
            created_at=utc_now(),
            metadata=metadata or {},
        )