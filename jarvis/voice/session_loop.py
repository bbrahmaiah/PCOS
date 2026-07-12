from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from typing import Protocol, cast

from jarvis.live import (
    LiveDeterministicSystemMessage,
    LiveResponse,
    LiveResponseBoundaryRuntime,
    LiveResponseGenerationSource,
    LiveResponseKind,
    LiveResponseSafety,
    make_live_turn_id,
)
from jarvis.voice.audio_preprocessing import (
    VoiceAudioPreprocessingResult,
    VoiceAudioPreprocessingRuntime,
    VoiceAudioPreprocessingStatus,
)
from jarvis.voice.awareness_cognition_bridge import (
    VoiceAwarenessCognitionBridge,
)
from jarvis.voice.barge_in_runtime import (
    VoiceBargeInDisposition,
    VoiceBargeInPlaybackController,
    VoiceBargeInRequest,
    VoiceBargeInResult,
    VoiceBargeInRuntime,
)
from jarvis.voice.cognition_response import (
    VoiceCognitionRequest,
    VoiceCognitionResult,
)
from jarvis.voice.cognitive_router import (
    VoiceCognitiveRouteAction,
    VoiceCognitiveRouteDecision,
    VoiceCognitiveRouter,
    VoiceCognitiveRouteRequest,
)
from jarvis.voice.contracts import (
    VoiceInputFrame,
    VoiceRuntimeConfig,
    VoiceTranscript,
    VoiceTranscriptKind,
    default_voice_runtime_config,
    make_voice_interrupt_id,
    make_voice_transcript_id,
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
from jarvis.voice.perception import (
    VoicePerceptionPacket,
    VoicePerceptionPolicy,
    VoicePerceptionRuntime,
    enrich_transcript_with_perception,
)
from jarvis.voice.playback_runtime import (
    VoicePlaybackResult,
    VoicePlaybackRuntime,
    VoicePlaybackRuntimeStatus,
    VoicePlaybackSnapshot,
)
from jarvis.voice.reflex_response import (
    VoiceReflexResponseDecision,
    VoiceReflexResponseKind,
)
from jarvis.voice.stt_runtime import (
    VoiceSTTResult,
    VoiceSTTRuntime,
)
from jarvis.voice.transcript_attention_gate import (
    TranscriptAttentionGate,
    TranscriptAttentionGatePolicy,
    normalize_transcript_text,
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
class VoiceSessionFsmTransition:
    from_status: VoiceSessionLoopStatus
    to_status: VoiceSessionLoopStatus
    reason: str
    allowed: bool
    created_at: datetime

    def to_metadata(self) -> dict[str, object]:
        return {
            "from_status": self.from_status.value,
            "to_status": self.to_status.value,
            "reason": self.reason,
            "allowed": self.allowed,
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class VoiceSessionLoopPolicy:
    partial_transcript_every_frames: int = 12
    health_check_every_cycles: int = 50
    idle_sleep_seconds: float = 0.01
    max_consecutive_failures: int = 3
    auto_recover: bool = True
    allow_barge_in: bool = True
    interrupt_playback_on_speech_start: bool = False
    threaded_interruption_resume_enabled: bool = True
    max_interruption_stack_depth: int = 3
    barge_in_completion_silence_ms: int = 260
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
        if self.max_interruption_stack_depth < 1:
            raise ValueError("max_interruption_stack_depth must be positive.")
        if self.barge_in_completion_silence_ms <= 0:
            raise ValueError("barge_in_completion_silence_ms must be positive.")


@dataclass(frozen=True, slots=True)
class VoiceConversationThreadFrame:
    thread_id: str
    paused_response_text: str
    interruption_text: str
    interruption_kind: str
    resume_context_sent: bool
    created_at: datetime

    def to_metadata(self) -> dict[str, object]:
        return {
            "thread_id": self.thread_id,
            "paused_response_text": _compact_thread_text(
                self.paused_response_text,
                limit=160,
            ),
            "interruption_text": _compact_thread_text(
                self.interruption_text,
                limit=120,
            ),
            "interruption_kind": self.interruption_kind,
            "resume_context_sent": self.resume_context_sent,
            "created_at": self.created_at.isoformat(),
        }


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


class VoiceSessionAudioPreprocessor(Protocol):
    def process_frame(self, frame: VoiceInputFrame) -> VoiceAudioPreprocessingResult:
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


class VoiceSessionReflex(Protocol):
    def evaluate(
        self,
        transcript: VoiceTranscript,
        *,
        assistant_speaking: bool,
    ) -> VoiceReflexResponseDecision:
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


class VoiceSessionCognitiveRouter(Protocol):
    def route(
        self,
        request: VoiceCognitiveRouteRequest,
    ) -> VoiceCognitiveRouteDecision:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def snapshot(self) -> dict[str, object]:
        raise NotImplementedError


class VoiceSessionPerception(Protocol):
    def observe_partial(
        self,
        transcript: VoiceTranscript,
        *,
        assistant_speaking: bool = False,
    ) -> VoicePerceptionPacket:
        raise NotImplementedError

    def observe_final(
        self,
        transcript: VoiceTranscript,
        *,
        assistant_speaking: bool = False,
    ) -> VoicePerceptionPacket:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def snapshot(self) -> dict[str, object]:
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
        audio_preprocessor: VoiceSessionAudioPreprocessor | None = None,
        vad: VoiceSessionVad | None = None,
        stt: VoiceSessionSTT | None = None,
        cognition: VoiceSessionCognition | None = None,
        reflex: VoiceSessionReflex | None = None,
        tts: VoiceSessionTTS | None = None,
        playback: VoiceSessionPlayback | None = None,
        barge_in: VoiceSessionBargeIn | None = None,
        health: VoiceSessionHealth | None = None,
        transcript_gate: TranscriptAttentionGate | None = None,
        transcript_gate_policy: TranscriptAttentionGatePolicy | None = None,
        perception: VoiceSessionPerception | None = None,
        cognitive_router: VoiceSessionCognitiveRouter | None = None,
        policy: VoiceSessionLoopPolicy | None = None,
    ) -> None:
        self._config = config or default_voice_runtime_config()
        self._microphone = microphone or VoiceMicrophoneCaptureRuntime(
            config=self._config
        )
        self._audio_preprocessor = (
            audio_preprocessor or VoiceAudioPreprocessingRuntime()
        )
        self._vad = vad or VoiceActivityRuntime()
        self._stt = stt or VoiceSTTRuntime(config=self._config)
        self._cognition = cognition or VoiceAwarenessCognitionBridge()
        self._reflex = reflex
        self._response_boundary = LiveResponseBoundaryRuntime()
        self._tts = tts or VoiceTTSRuntime(config=self._config)
        self._playback = playback or VoicePlaybackRuntime(config=self._config)
        self._barge_in = barge_in or VoiceBargeInRuntime()
        self._perception = perception or VoicePerceptionRuntime(
            policy=VoicePerceptionPolicy(
                wake_words=frozenset(
                    {
                        self._config.wake_word,
                        "jarvis",
                        "jervis",
                        "jarves",
                    }
                )
            )
        )
        self._cognitive_router = cognitive_router or VoiceCognitiveRouter()
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
        self._fsm_transitions = 0
        self._fsm_violations = 0
        self._last_fsm_transition: VoiceSessionFsmTransition | None = None
        self._last_fsm_violation: VoiceSessionFsmTransition | None = None
        self._running = False
        self._assistant_speaking = False
        self._cycles = 0
        self._captured_frames = 0
        self._preprocessed_frames = 0
        self._dropped_audio_frames = 0
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
        self._last_result_message: str | None = None
        self._async_playback_active = False
        self._last_playback_status: str | None = None
        self._last_audio_preprocessing_status: str | None = None
        self._last_audio_preprocessing_message: str | None = None
        self._last_gate_reason: str | None = None
        self._last_gate_text: str | None = None
        self._last_cognitive_route_action: str | None = None
        self._last_cognitive_route_state: str | None = None
        self._last_cognitive_route_reason: str | None = None
        self._last_perception_intent_state: str | None = None
        self._last_perception_reason: str | None = None
        self._last_perception_confidence: float | None = None
        self._last_perception_stability: float | None = None
        self._last_cognition_status: str | None = None
        self._last_cognition_message: str | None = None
        self._last_cognition_safety: str | None = None
        self._last_runner_status: str | None = None
        self._last_runner_reason: str | None = None
        self._last_wake_decision: str | None = None
        self._last_wake_reason: str | None = None
        self._reflex_responses = 0
        self._last_reflex_kind: str | None = None
        self._last_reflex_reason: str | None = None
        self._transcript_gate = transcript_gate or TranscriptAttentionGate(
            policy=transcript_gate_policy
        )
        self._speech_silence_frames = 0
        self._max_speech_silence_frames = max(
            1,
            self._config.max_silence_ms // self._config.frame_duration_ms,
        )
        self._max_barge_in_silence_frames = max(
            1,
            self._policy.barge_in_completion_silence_ms
            // self._config.frame_duration_ms,
        )
        self._last_accepted_partial: VoiceTranscript | None = None
        self._last_wake_partial: VoiceTranscript | None = None
        self._conversation_thread_stack: list[VoiceConversationThreadFrame] = []
        self._last_thread_event: str | None = None
        self._last_thread_context: dict[str, object] | None = None


    def start(self) -> VoiceSessionLoopResult:
        started = time.perf_counter()
        self._set_status(VoiceSessionLoopStatus.STARTING, reason="start_requested")

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
            self._perception.reset()
            self._cognitive_router.reset()
        except Exception as exc:
            self._set_status(VoiceSessionLoopStatus.FAILED, reason="start_failed")
            self._last_error = str(exc)
            return self._result(
                operation=VoiceSessionLoopOperation.START,
                event=VoiceSessionLoopEvent.ERROR,
                message="voice session start failed",
                started=started,
                metadata={"error": str(exc)},
            )

        self._running = True
        self._set_status(VoiceSessionLoopStatus.LISTENING, reason="start_ready")
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

        playback_completion = self._finish_async_playback_if_done(started=started)
        if playback_completion is not None:
            return playback_completion

        if self._should_health_check():
            health_result = self._check_and_recover_health()
            if health_result.status in {
                VoiceHealthStatus.FAILED,
                VoiceHealthStatus.CRITICAL,
            }:
                self._set_status(
                    VoiceSessionLoopStatus.RECOVERING,
                    reason="health_recovery_required",
                )
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
        preprocessed = self._audio_preprocessor.process_frame(capture.frame)
        self._last_audio_preprocessing_status = preprocessed.status.value
        self._last_audio_preprocessing_message = preprocessed.message
        if preprocessed.status == VoiceAudioPreprocessingStatus.FAILED:
            return self._failure(
                operation=VoiceSessionLoopOperation.PROCESS_ONCE,
                started=started,
                message="audio preprocessing failed",
                error=preprocessed.message,
            )
        if preprocessed.status == VoiceAudioPreprocessingStatus.DROPPED:
            self._dropped_audio_frames += 1
            return self._result(
                operation=VoiceSessionLoopOperation.PROCESS_ONCE,
                event=None,
                message="audio preprocessing dropped frame",
                started=started,
                metadata={
                    "audio_preprocessing": preprocessed.metadata,
                },
            )
        if preprocessed.frame is None:
            return self._failure(
                operation=VoiceSessionLoopOperation.PROCESS_ONCE,
                started=started,
                message="audio preprocessing returned no frame",
                error=preprocessed.message,
            )

        self._preprocessed_frames += 1
        return self.process_frame(preprocessed.frame)

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
            if self._segment_frames:
                self._speech_silence_frames += 1
                silence_limit = self._active_completion_silence_frames()
                if self._speech_silence_frames >= silence_limit:
                    return self._handle_final_segment(started=started)

            self._set_status(
                (
                    VoiceSessionLoopStatus.SPEAKING
                    if self._assistant_speaking
                    else VoiceSessionLoopStatus.LISTENING
                ),
                reason="silence_frame_processed",
            )
            return self._result(
                operation=VoiceSessionLoopOperation.PROCESS_FRAME,
                event=None,
                message="silence frame processed",
                started=started,
            )

        if activity.decision == VoiceActivityDecision.SPEECH_STARTED:
            self._speech_silence_frames = 0
            self._segment_frames = [frame]
            self._last_accepted_partial = None
            self._last_wake_partial = None
            self._speech_segments += 1
            playback_result = self._interrupt_playback_on_user_speech_start()
            self._set_status(
                VoiceSessionLoopStatus.USER_SPEAKING,
                reason="speech_started",
            )
            return self._result(
                operation=VoiceSessionLoopOperation.PROCESS_FRAME,
                event=(
                    VoiceSessionLoopEvent.BARGE_IN_INTERRUPTED
                    if playback_result is not None
                    else VoiceSessionLoopEvent.SPEECH_STARTED
                ),
                playback_result=playback_result,
                message=(
                    "speech segment started and playback interrupted"
                    if playback_result is not None
                    else "speech segment started"
                ),
                started=started,
            )

        if activity.decision == VoiceActivityDecision.SPEECH_CONTINUED:
            self._speech_silence_frames = 0
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

        if activity.decision == VoiceActivityDecision.HOLDING_FOR_COMPLETION:
            self._segment_frames.append(frame)
            self._speech_silence_frames += 1
            silence_limit = self._active_completion_silence_frames()
            if self._speech_silence_frames >= silence_limit:
                return self._handle_final_segment(started=started)

            self._set_status(
                VoiceSessionLoopStatus.USER_SPEAKING,
                reason="holding_for_completion",
            )
            return self._result(
                operation=VoiceSessionLoopOperation.PROCESS_FRAME,
                event=None,
                message="holding for speech completion",
                started=started,
                metadata={
                    "buffered_frames": len(self._segment_frames),
                    "speech_silence_frames": self._speech_silence_frames,
                },
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
        self._set_status(VoiceSessionLoopStatus.STOPPING, reason="stop_requested")
        self._running = False

        try:
            self._playback.stop()
            self._microphone.stop()
        except Exception as exc:
            self._set_status(VoiceSessionLoopStatus.DEGRADED, reason="stop_degraded")
            self._last_error = str(exc)
            return self._result(
                operation=VoiceSessionLoopOperation.STOP,
                event=VoiceSessionLoopEvent.ERROR,
                message="voice session stop degraded",
                started=started,
                metadata={"error": str(exc)},
            )

        self._assistant_speaking = False
        self._set_status(VoiceSessionLoopStatus.STOPPED, reason="stop_completed")
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
            metadata={
                **self._component_snapshot_metadata(),
                "last_result_message": self._last_result_message,
                "last_gate_reason": self._last_gate_reason,
                "last_gate_text": self._last_gate_text,
                "last_cognition_status": self._last_cognition_status,
                "last_cognition_message": self._last_cognition_message,
                "last_cognition_safety": self._last_cognition_safety,
                "last_runner_status": self._last_runner_status,
                "last_runner_reason": self._last_runner_reason,
                "last_wake_decision": self._last_wake_decision,
                "last_wake_reason": self._last_wake_reason,
                "reflex_responses": self._reflex_responses,
                "last_reflex_kind": self._last_reflex_kind,
                "last_reflex_reason": self._last_reflex_reason,
                "last_perception_intent_state": self._last_perception_intent_state,
                "last_perception_reason": self._last_perception_reason,
                "last_perception_confidence": self._last_perception_confidence,
                "last_perception_stability": self._last_perception_stability,
                "last_playback_status": self._last_playback_status,
                "threaded_interruption": self._threaded_interruption_metadata(),
                "fsm_transitions": self._fsm_transitions,
                "fsm_violations": self._fsm_violations,
                "last_fsm_transition": (
                    None
                    if self._last_fsm_transition is None
                    else self._last_fsm_transition.to_metadata()
                ),
                "last_fsm_violation": (
                    None
                    if self._last_fsm_violation is None
                    else self._last_fsm_violation.to_metadata()
                ),
            },
        )

    def live_snapshot(self) -> VoiceSessionLoopSnapshot:
        """
        Cheap hot-path snapshot for live supervision.

        Full snapshots ask every organ for diagnostic state. That is useful for
        operator output, but too expensive for the always-on spine monitor.
        """
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
            metadata=self._live_snapshot_metadata(),
        )

    def _handle_final_segment(
        self,
        *,
        started: float,
    ) -> VoiceSessionLoopResult:
        frames = tuple(self._segment_frames)
        self._segment_frames = []
        self._speech_silence_frames = 0

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
            if self._last_accepted_partial is not None:
                transcript = self._promote_partial_to_final(
                    self._last_accepted_partial
                )
                self._last_accepted_partial = None
                self._last_wake_partial = None
                self._final_transcripts += 1
                self._last_transcript_text = transcript.text
                return self._handle_transcript(
                    transcript=transcript,
                    started=started,
                )

            return self._result(
                operation=VoiceSessionLoopOperation.PROCESS_FRAME,
                event=None,
                message="final transcription returned no transcript",
                started=started,
                metadata={"stt_status": stt_result.status.value},
            )

        resolved_transcript = self._resolve_final_transcript(stt_result.transcript)
        self._last_accepted_partial = None
        self._last_wake_partial = None
        self._final_transcripts += 1
        self._last_transcript_text = resolved_transcript.text
        return self._handle_transcript(
            transcript=resolved_transcript,
            started=started,
        )

    def _resolve_final_transcript(
        self,
        transcript: VoiceTranscript,
    ) -> VoiceTranscript:
        if self._transcript_has_wake(transcript):
            return transcript
        if self._last_wake_partial is None:
            return transcript

        merged_text = _merge_transcript_texts(
            self._last_wake_partial.text,
            transcript.text,
        )
        return VoiceTranscript(
            transcript_id=make_voice_transcript_id(),
            session_id=transcript.session_id,
            segment_id=transcript.segment_id,
            kind=VoiceTranscriptKind.FINAL,
            text=merged_text,
            confidence=max(transcript.confidence, self._last_wake_partial.confidence),
            created_at=utc_now(),
            metadata={
                **transcript.metadata,
                "resolved_from_wake_partial": True,
                "wake_partial_text": self._last_wake_partial.text,
                "wake_partial_transcript_id": str(
                    self._last_wake_partial.transcript_id
                ),
                "source_final_text": transcript.text,
                "source_final_transcript_id": str(transcript.transcript_id),
            },
        )

    def _promote_partial_to_final(
        self,
        transcript: VoiceTranscript,
    ) -> VoiceTranscript:
        return VoiceTranscript(
            transcript_id=make_voice_transcript_id(),
            session_id=transcript.session_id,
            segment_id=transcript.segment_id,
            kind=VoiceTranscriptKind.FINAL,
            text=transcript.text,
            confidence=transcript.confidence,
            created_at=utc_now(),
            metadata={
                **transcript.metadata,
                "promoted_from_partial": True,
                "source_transcript_id": str(transcript.transcript_id),
            },
        )

    def _handle_transcript(
        self,
        *,
        transcript: VoiceTranscript,
        started: float,
    ) -> VoiceSessionLoopResult:
        if transcript.kind == VoiceTranscriptKind.FINAL:
            perception_packet = self._perception.observe_final(
                transcript,
                assistant_speaking=(
                    self._assistant_speaking or self._async_playback_active
                ),
            )
            transcript = enrich_transcript_with_perception(
                transcript,
                perception_packet,
            )
            self._remember_perception(perception_packet)

        self._last_transcript_text = transcript.text
        barge_in_result: VoiceBargeInResult | None = None

        if self._assistant_speaking and self._policy.allow_barge_in:
            barge_loop_result = self._handle_barge_in(
                transcript=transcript,
                started=started,
            )
            barge_in_result = barge_loop_result.barge_in_result
            if (
                barge_in_result is None
                or not barge_in_result.interrupted
                or barge_in_result.disposition
                not in {
                    VoiceBargeInDisposition.NEW_QUESTION,
                    VoiceBargeInDisposition.USER_CORRECTION,
                }
            ):
                return barge_loop_result
            self._capture_threaded_interruption(
                transcript=transcript,
                barge_in_result=barge_in_result,
            )

        if transcript.kind != VoiceTranscriptKind.FINAL:
            return self._result(
                operation=VoiceSessionLoopOperation.HANDLE_TRANSCRIPT,
                event=None,
                transcript=transcript,
                message="non-final transcript ignored for response",
                started=started,
                barge_in_result=barge_in_result,
            )

        reflex_decision = self._evaluate_reflex(transcript)
        if reflex_decision.accepted:
            return self._handle_reflex_decision(
                transcript=transcript,
                decision=reflex_decision,
                started=started,
                barge_in_result=barge_in_result,
            )

        gate_decision = self._transcript_gate.evaluate(transcript)
        if not gate_decision.accepted:
            self._set_status(
                VoiceSessionLoopStatus.LISTENING,
                reason="attention_gate_rejected",
            )
            return self._result(
                operation=VoiceSessionLoopOperation.HANDLE_TRANSCRIPT,
                event=None,
                transcript=transcript,
                message="transcript rejected by attention gate",
                started=started,
                barge_in_result=barge_in_result,
                metadata={
                    "perception": transcript.metadata.get("perception"),
                    "transcript_gate": gate_decision.to_metadata(),
                },
            )

        route_decision = self._cognitive_router.route(
            VoiceCognitiveRouteRequest(
                transcript=transcript,
                gate_decision=gate_decision,
                assistant_speaking=self._assistant_speaking,
                active_playback=self._async_playback_active,
            )
        )
        self._remember_cognitive_route(route_decision)
        if not route_decision.should_enter_cognition:
            self._set_status(
                (
                    VoiceSessionLoopStatus.LISTENING
                    if route_decision.action == VoiceCognitiveRouteAction.IGNORE
                    else VoiceSessionLoopStatus.USER_SPEAKING
                ),
                reason=f"cognitive_router_{route_decision.action.value}",
            )
            return self._result(
                operation=VoiceSessionLoopOperation.HANDLE_TRANSCRIPT,
                event=None,
                transcript=transcript,
                message="transcript held by cognitive router",
                started=started,
                barge_in_result=barge_in_result,
                metadata={
                    "perception": transcript.metadata.get("perception"),
                    "transcript_gate": gate_decision.to_metadata(),
                    "cognitive_route": route_decision.to_metadata(),
                },
            )

        self._set_status(
            VoiceSessionLoopStatus.THINKING,
            reason="cognition_started",
        )
        self._update_pending_thread_interruption_text(transcript)
        thread_context = self._threaded_interruption_context()
        thread_metadata = self._last_thread_context if thread_context else None
        cognition_result = self._cognition.think_from_transcript(
            VoiceCognitionRequest(
                transcript=transcript,
                user_label=self._config.user_label,
                assistant_name=self._config.assistant_name,
                working_memory_context=thread_context,
                planning_context=thread_context,
                metadata={
                    "threaded_interruption": thread_metadata,
                }
                if thread_context
                else {},
            )
        )

        if cognition_result.response is None:
            return self._result(
                operation=VoiceSessionLoopOperation.HANDLE_TRANSCRIPT,
                event=None,
                transcript=transcript,
                cognition_result=cognition_result,
                barge_in_result=barge_in_result,
                message="cognition produced no response",
                started=started,
                metadata={
                    "perception": transcript.metadata.get("perception"),
                    "cognition_status": cognition_result.status.value,
                    "cognition_message": cognition_result.message,
                    "cognition_safety": cognition_result.safety.value,
                    "cognition_metadata": cognition_result.metadata,
                },
            )

        return self._synthesize_and_play_response(
            response=cognition_result.response,
            transcript=transcript,
            cognition_result=cognition_result,
            barge_in_result=barge_in_result,
            started=started,
            metadata={
                "continued_after_barge_in": barge_in_result is not None,
                "barge_in_disposition": (
                    barge_in_result.disposition.value
                    if barge_in_result is not None
                    else None
                ),
                "threaded_interruption": thread_metadata,
                "perception": transcript.metadata.get("perception"),
                "transcript_gate": gate_decision.to_metadata(),
                "cognitive_route": route_decision.to_metadata(),
                "response_origin": "cognition_response_boundary",
            },
        )

    def _evaluate_reflex(
        self,
        transcript: VoiceTranscript,
    ) -> VoiceReflexResponseDecision:
        if self._reflex is None:
            return VoiceReflexResponseDecision(
                accepted=False,
                kind=VoiceReflexResponseKind.NONE,
                normalized_text=normalize_transcript_text(transcript.text),
                response_text=None,
                should_speak=False,
                should_stop_playback=False,
                should_shutdown_session=False,
                should_continue_to_cognition=True,
                confidence=transcript.confidence,
                reason="reflex_not_configured",
            )
        return self._reflex.evaluate(
            transcript,
            assistant_speaking=self._assistant_speaking,
        )

    def _handle_reflex_decision(
        self,
        *,
        transcript: VoiceTranscript,
        decision: VoiceReflexResponseDecision,
        started: float,
        barge_in_result: VoiceBargeInResult | None,
    ) -> VoiceSessionLoopResult:
        self._last_reflex_kind = decision.kind.value
        self._last_reflex_reason = decision.reason

        if decision.should_shutdown_session:
            return self._handle_shutdown_reflex(
                transcript=transcript,
                decision=decision,
                started=started,
                barge_in_result=barge_in_result,
            )

        if decision.should_stop_playback:
            playback_result = None
            interrupted_playback = (
                self._assistant_speaking or self._async_playback_active
            )
            if interrupted_playback:
                playback_result = self._playback.stop()
                self._interruptions += 1

            self._assistant_speaking = False
            self._async_playback_active = False
            self._set_status(
                (
                    VoiceSessionLoopStatus.INTERRUPTED
                    if interrupted_playback
                    else VoiceSessionLoopStatus.LISTENING
                ),
                reason="reflex_stop_playback",
            )
            return self._result(
                operation=VoiceSessionLoopOperation.HANDLE_TRANSCRIPT,
                event=(
                    VoiceSessionLoopEvent.BARGE_IN_INTERRUPTED
                    if interrupted_playback
                    else None
                ),
                transcript=transcript,
                playback_result=(
                    playback_result
                    if isinstance(playback_result, VoicePlaybackResult)
                    else None
                ),
                barge_in_result=barge_in_result,
                message="voice reflex stopped playback",
                started=started,
                metadata={
                    "voice_reflex": decision.to_metadata(),
                    "response_origin": "voice_reflex_operational",
                },
            )

        if decision.should_speak and decision.response_text is not None:
            response = self._make_reflex_response(
                text=decision.response_text,
                decision=decision,
            )
            if response is None:
                self._set_status(
                    VoiceSessionLoopStatus.DEGRADED,
                    reason="reflex_response_blocked",
                )
                return self._result(
                    operation=VoiceSessionLoopOperation.HANDLE_TRANSCRIPT,
                    event=VoiceSessionLoopEvent.ERROR,
                    transcript=transcript,
                    barge_in_result=barge_in_result,
                    message="voice reflex response blocked",
                    started=started,
                    metadata={
                        "voice_reflex": decision.to_metadata(),
                        "response_origin": "voice_reflex_operational",
                    },
                )

            self._reflex_responses += 1
            return self._synthesize_and_play_response(
                response=response,
                transcript=transcript,
                cognition_result=None,
                barge_in_result=barge_in_result,
                started=started,
                metadata={
                    "voice_reflex": decision.to_metadata(),
                    "response_origin": "voice_reflex_operational",
                    "deterministic_system_response": True,
                },
            )

        self._set_status(
            VoiceSessionLoopStatus.LISTENING,
            reason="reflex_accepted_without_speech",
        )
        return self._result(
            operation=VoiceSessionLoopOperation.HANDLE_TRANSCRIPT,
            event=None,
            transcript=transcript,
            barge_in_result=barge_in_result,
            message="voice reflex accepted without speech",
            started=started,
            metadata={
                "voice_reflex": decision.to_metadata(),
                "response_origin": "voice_reflex_operational",
            },
        )

    def _handle_shutdown_reflex(
        self,
        *,
        transcript: VoiceTranscript,
        decision: VoiceReflexResponseDecision,
        started: float,
        barge_in_result: VoiceBargeInResult | None,
    ) -> VoiceSessionLoopResult:
        playback_result = None
        interrupted_playback = self._assistant_speaking or self._async_playback_active

        try:
            if decision.should_stop_playback:
                playback_result = self._playback.stop()
                if interrupted_playback:
                    self._interruptions += 1
            self._microphone.stop()
        except Exception as exc:
            self._running = False
            self._assistant_speaking = False
            self._async_playback_active = False
            self._status = VoiceSessionLoopStatus.DEGRADED
            self._last_error = str(exc)
            return self._result(
                operation=VoiceSessionLoopOperation.HANDLE_TRANSCRIPT,
                event=VoiceSessionLoopEvent.ERROR,
                transcript=transcript,
                playback_result=(
                    playback_result
                    if isinstance(playback_result, VoicePlaybackResult)
                    else None
                ),
                barge_in_result=barge_in_result,
                message="voice shutdown command degraded",
                started=started,
                metadata={
                    "voice_reflex": decision.to_metadata(),
                    "response_origin": "voice_reflex_operational",
                    "error": str(exc),
                },
            )

        self._running = False
        self._assistant_speaking = False
        self._async_playback_active = False
        self._status = VoiceSessionLoopStatus.STOPPED
        return self._result(
            operation=VoiceSessionLoopOperation.HANDLE_TRANSCRIPT,
            event=VoiceSessionLoopEvent.STOPPED,
            transcript=transcript,
            playback_result=(
                playback_result
                if isinstance(playback_result, VoicePlaybackResult)
                else None
            ),
            barge_in_result=barge_in_result,
            message="voice session shutdown by voice command",
            started=started,
            metadata={
                "voice_reflex": decision.to_metadata(),
                "response_origin": "voice_reflex_operational",
                "shutdown_word": True,
            },
        )

    def _make_reflex_response(
        self,
        *,
        text: str,
        decision: VoiceReflexResponseDecision,
    ) -> LiveResponse | None:
        result = self._response_boundary.system_message(
            LiveDeterministicSystemMessage(
                turn_id=make_live_turn_id(),
                kind=LiveResponseKind.DIAGNOSTIC,
                text=text,
                source=LiveResponseGenerationSource.DIAGNOSTIC_SYSTEM,
                safety=LiveResponseSafety.SAFE_TO_SPEAK,
                metadata={
                    "response_origin": "voice_reflex_operational",
                    "voice_reflex_kind": decision.kind.value,
                    "voice_reflex_reason": decision.reason,
                    "fixed_conversational_response": False,
                },
            )
        )
        return result.response if result.succeeded else None

    def _synthesize_and_play_response(
        self,
        *,
        response: LiveResponse,
        transcript: VoiceTranscript,
        cognition_result: VoiceCognitionResult | None,
        barge_in_result: VoiceBargeInResult | None,
        started: float,
        metadata: dict[str, object],
    ) -> VoiceSessionLoopResult:
        self._responses += 1
        self._last_response_text = response.text
        self._set_status(
            VoiceSessionLoopStatus.SYNTHESIZING,
            reason="tts_synthesis_started",
        )

        tts_result = self._tts.synthesize_response(
            response=response,
            session_id=transcript.session_id,
        )

        if not tts_result.chunks:
            return self._result(
                operation=VoiceSessionLoopOperation.HANDLE_TRANSCRIPT,
                event=None,
                transcript=transcript,
                cognition_result=cognition_result,
                tts_result=tts_result,
                barge_in_result=barge_in_result,
                message="TTS produced no chunks",
                started=started,
                metadata=metadata,
            )

        self._tts_outputs += 1
        self._set_status(
            VoiceSessionLoopStatus.SPEAKING,
            reason="playback_started",
        )
        self._assistant_speaking = True

        enqueue_result = self._playback.enqueue_chunks(tts_result.chunks)
        if not enqueue_result.succeeded:
            self._assistant_speaking = False
            self._set_status(
                VoiceSessionLoopStatus.DEGRADED,
                reason="playback_enqueue_failed",
            )
            return self._result(
                operation=VoiceSessionLoopOperation.HANDLE_TRANSCRIPT,
                event=None,
                transcript=transcript,
                cognition_result=cognition_result,
                tts_result=tts_result,
                playback_result=enqueue_result,
                barge_in_result=barge_in_result,
                message="playback enqueue failed",
                started=started,
                metadata=metadata,
            )

        playback_result = self._playback.play_all()

        if _is_async_playback_result(playback_result):
            self._async_playback_active = True
            self._assistant_speaking = True
            self._set_status(
                VoiceSessionLoopStatus.SPEAKING,
                reason="async_playback_started",
            )
            self._last_playback_status = playback_result.status.value
            return self._result(
                operation=VoiceSessionLoopOperation.HANDLE_TRANSCRIPT,
                event=VoiceSessionLoopEvent.RESPONSE_READY,
                transcript=transcript,
                cognition_result=cognition_result,
                tts_result=tts_result,
                playback_result=playback_result,
                barge_in_result=barge_in_result,
                message="voice response playback started",
                started=started,
                metadata={
                    **metadata,
                    "async_playback": True,
                },
            )

        self._assistant_speaking = False
        if playback_result.succeeded:
            self._played_outputs += 1
            self._set_status(
                VoiceSessionLoopStatus.LISTENING,
                reason="playback_finished",
            )
            event = VoiceSessionLoopEvent.PLAYBACK_FINISHED
        else:
            self._set_status(
                VoiceSessionLoopStatus.DEGRADED,
                reason="playback_failed",
            )
            event = VoiceSessionLoopEvent.ERROR

        return self._result(
            operation=VoiceSessionLoopOperation.HANDLE_TRANSCRIPT,
            event=event,
            transcript=transcript,
            cognition_result=cognition_result,
            tts_result=tts_result,
            playback_result=playback_result,
            barge_in_result=barge_in_result,
            message="voice response pipeline completed",
            started=started,
            metadata=metadata,
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
            self._async_playback_active = False
            self._set_status(
                VoiceSessionLoopStatus.INTERRUPTED,
                reason="barge_in_interrupted",
            )
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

    def _interrupt_playback_on_user_speech_start(self) -> VoicePlaybackResult | None:
        if not self._policy.allow_barge_in:
            return None
        if not self._policy.interrupt_playback_on_speech_start:
            return None
        if not (self._assistant_speaking or self._async_playback_active):
            return None

        self._capture_threaded_interruption_from_speech_start()
        playback_result = self._playback.stop()
        self._interruptions += 1
        self._assistant_speaking = False
        self._async_playback_active = False
        self._set_status(
            VoiceSessionLoopStatus.INTERRUPTED,
            reason="speech_start_interrupted_playback",
        )
        return (
            playback_result
            if isinstance(playback_result, VoicePlaybackResult)
            else None
        )

    def _capture_threaded_interruption(
        self,
        *,
        transcript: VoiceTranscript,
        barge_in_result: VoiceBargeInResult,
    ) -> None:
        if barge_in_result.disposition not in {
            VoiceBargeInDisposition.NEW_QUESTION,
            VoiceBargeInDisposition.USER_CORRECTION,
        }:
            return
        self._push_or_update_thread_frame(
            interruption_text=transcript.text,
            interruption_kind=barge_in_result.disposition.value,
        )

    def _capture_threaded_interruption_from_speech_start(self) -> None:
        self._push_or_update_thread_frame(
            interruption_text="user started speaking before final transcription",
            interruption_kind="speech_start",
        )

    def _push_or_update_thread_frame(
        self,
        *,
        interruption_text: str,
        interruption_kind: str,
    ) -> None:
        if not self._policy.threaded_interruption_resume_enabled:
            return
        paused_response_text = (self._last_response_text or "").strip()
        if not paused_response_text:
            return

        interruption = " ".join(interruption_text.strip().split())
        if not interruption:
            interruption = "user interrupted"

        if self._conversation_thread_stack:
            current = self._conversation_thread_stack[-1]
            if (
                current.paused_response_text == paused_response_text
                and not current.resume_context_sent
            ):
                self._conversation_thread_stack[-1] = replace(
                    current,
                    interruption_text=interruption,
                    interruption_kind=interruption_kind,
                )
                self._last_thread_event = "thread_interruption_updated"
                return

        frame = VoiceConversationThreadFrame(
            thread_id=str(make_voice_interrupt_id()),
            paused_response_text=paused_response_text,
            interruption_text=interruption,
            interruption_kind=interruption_kind,
            resume_context_sent=False,
            created_at=utc_now(),
        )
        self._conversation_thread_stack.append(frame)
        max_depth = self._policy.max_interruption_stack_depth
        if len(self._conversation_thread_stack) > max_depth:
            self._conversation_thread_stack = self._conversation_thread_stack[
                -max_depth:
            ]
        self._last_thread_event = "thread_interruption_captured"

    def _threaded_interruption_context(self) -> tuple[str, ...]:
        if not self._policy.threaded_interruption_resume_enabled:
            return ()
        if not self._conversation_thread_stack:
            return ()

        frame = self._conversation_thread_stack[-1]
        if frame.resume_context_sent:
            return ()

        sent_frame = replace(frame, resume_context_sent=True)
        self._conversation_thread_stack[-1] = sent_frame
        context = (
            "Threaded interruption active. "
            "The user interrupted while JARVIS was speaking. "
            "Answer the user's current side question first, then briefly resume "
            "the paused thought if it is still useful. "
            "Do not announce internal resume mechanics. "
            f"Paused JARVIS response: {sent_frame.paused_response_text} "
            f"User interruption: {sent_frame.interruption_text}"
        )
        self._last_thread_event = "thread_resume_context_sent"
        self._last_thread_context = sent_frame.to_metadata()
        return (context,)

    def _update_pending_thread_interruption_text(
        self,
        transcript: VoiceTranscript,
    ) -> None:
        if not self._conversation_thread_stack:
            return
        frame = self._conversation_thread_stack[-1]
        if frame.resume_context_sent:
            return
        if frame.interruption_kind != "speech_start":
            return
        self._conversation_thread_stack[-1] = replace(
            frame,
            interruption_text=transcript.text,
        )
        self._last_thread_event = "thread_interruption_transcript_resolved"

    def _transcribe_partial(self) -> VoiceSTTResult:
        partial = self._stt.transcribe_partial(tuple(self._segment_frames))

        if partial.transcript is not None:
            perception_packet = self._perception.observe_partial(
                partial.transcript,
                assistant_speaking=(
                    self._assistant_speaking or self._async_playback_active
                ),
            )
            transcript = enrich_transcript_with_perception(
                partial.transcript,
                perception_packet,
            )
            partial = VoiceSTTResult(
                status=partial.status,
                operation=partial.operation,
                transcript=transcript,
                candidate=partial.candidate,
                model=partial.model,
                message=partial.message,
                created_at=partial.created_at,
                metadata={
                    **partial.metadata,
                    "perception": perception_packet.to_metadata(),
                },
            )
            self._remember_perception(perception_packet)
            self._partial_transcripts += 1
            self._last_transcript_text = transcript.text

            self._last_accepted_partial = transcript
            wake_detected = self._transcript_has_wake(transcript)
            if wake_detected:
                self._last_wake_partial = transcript

            if wake_detected or self._last_wake_partial is not None:
                self._cognition.prefetch_from_partial(
                    VoiceCognitionRequest(
                        transcript=transcript,
                        user_label=self._config.user_label,
                        assistant_name=self._config.assistant_name,
                    )
                )

        return partial

    def _finish_async_playback_if_done(
        self,
        *,
        started: float,
    ) -> VoiceSessionLoopResult | None:
        if not self._async_playback_active:
            return None

        snapshot = _safe_snapshot(self._playback)
        if not isinstance(snapshot, VoicePlaybackSnapshot):
            return None

        self._last_playback_status = snapshot.status.value
        playback_active = (
            snapshot.status == VoicePlaybackRuntimeStatus.PLAYING
            or snapshot.current_playback is not None
        )
        if playback_active:
            self._assistant_speaking = True
            self._set_status(
                VoiceSessionLoopStatus.SPEAKING,
                reason="async_playback_still_active",
            )
            return None

        self._async_playback_active = False
        self._assistant_speaking = False
        self._played_outputs += 1
        if not self._segment_frames:
            self._set_status(
                VoiceSessionLoopStatus.LISTENING,
                reason="async_playback_finished",
            )

        return self._result(
            operation=VoiceSessionLoopOperation.PROCESS_ONCE,
            event=VoiceSessionLoopEvent.PLAYBACK_FINISHED,
            message="async playback finished",
            started=started,
            metadata={
                "async_playback": True,
                "playback_status": snapshot.status.value,
                "playback_stopped_count": snapshot.stopped_count,
                "playback_last_latency_ms": snapshot.last_latency_ms,
                "playback_first_audio_latency_ms": (
                    snapshot.last_first_audio_latency_ms
                ),
            },
        )

    def _should_partial_transcribe(self) -> bool:
        return (
            len(self._segment_frames) > 0
            and len(self._segment_frames)
            % self._policy.partial_transcript_every_frames
            == 0
        )

    def _active_completion_silence_frames(self) -> int:
        if self._assistant_speaking and self._policy.allow_barge_in:
            return self._max_barge_in_silence_frames

        return self._max_speech_silence_frames

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

    def _set_status(
        self,
        status: VoiceSessionLoopStatus,
        *,
        reason: str,
    ) -> bool:
        if status == self._status:
            return True

        allowed = _voice_session_status_transition_allowed(self._status, status)
        transition = VoiceSessionFsmTransition(
            from_status=self._status,
            to_status=status,
            reason=reason,
            allowed=allowed,
            created_at=utc_now(),
        )
        self._fsm_transitions += 1
        self._last_fsm_transition = transition

        if not allowed:
            self._fsm_violations += 1
            self._last_fsm_violation = transition
            if status not in {
                VoiceSessionLoopStatus.DEGRADED,
                VoiceSessionLoopStatus.FAILED,
                VoiceSessionLoopStatus.RECOVERING,
                VoiceSessionLoopStatus.STOPPING,
                VoiceSessionLoopStatus.STOPPED,
            }:
                return False

        self._status = status
        return allowed

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
        result_metadata = metadata or {}
        self._last_result_message = message
        self._remember_gate_diagnostic(result_metadata)
        self._remember_cognitive_route_diagnostic(result_metadata)
        self._remember_cognition_diagnostic(result_metadata)

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
            metadata=result_metadata,
        )

    def _remember_gate_diagnostic(self, metadata: dict[str, object]) -> None:
        transcript_gate = metadata.get("transcript_gate")
        if not isinstance(transcript_gate, dict):
            return
        reason = transcript_gate.get("reason")
        text = transcript_gate.get("normalized_text")
        self._last_gate_reason = str(reason) if reason is not None else None
        self._last_gate_text = str(text) if text is not None else None

    def _remember_cognitive_route(
        self,
        decision: VoiceCognitiveRouteDecision,
    ) -> None:
        self._last_cognitive_route_action = decision.action.value
        self._last_cognitive_route_state = decision.state.value
        self._last_cognitive_route_reason = decision.reason

    def _remember_perception(
        self,
        packet: VoicePerceptionPacket,
    ) -> None:
        self._last_perception_intent_state = packet.intent_state.value
        self._last_perception_reason = packet.reason
        self._last_perception_confidence = packet.confidence
        self._last_perception_stability = packet.stability

    def _remember_cognitive_route_diagnostic(
        self,
        metadata: dict[str, object],
    ) -> None:
        cognitive_route = metadata.get("cognitive_route")
        if not isinstance(cognitive_route, dict):
            return
        action = cognitive_route.get("action")
        state = cognitive_route.get("state")
        reason = cognitive_route.get("reason")
        if action is not None:
            self._last_cognitive_route_action = str(action)
        if state is not None:
            self._last_cognitive_route_state = str(state)
        if reason is not None:
            self._last_cognitive_route_reason = str(reason)

    def _remember_cognition_diagnostic(
        self,
        metadata: dict[str, object],
    ) -> None:
        status = metadata.get("cognition_status")
        message = metadata.get("cognition_message")
        safety = metadata.get("cognition_safety")
        cognition_metadata = metadata.get("cognition_metadata")
        if isinstance(cognition_metadata, dict):
            runner_status = cognition_metadata.get("runner_status")
            runner_reason = cognition_metadata.get("runner_reason")
            wake_decision = cognition_metadata.get("wake_decision")
            wake_reason = cognition_metadata.get("wake_reason")
        else:
            runner_status = metadata.get("runner_status")
            runner_reason = metadata.get("runner_reason")
            wake_decision = metadata.get("wake_decision")
            wake_reason = metadata.get("wake_reason")

        if status is not None:
            self._last_cognition_status = str(status)
        if message is not None:
            self._last_cognition_message = str(message)
        if safety is not None:
            self._last_cognition_safety = str(safety)
        if runner_status is not None:
            self._last_runner_status = str(runner_status)
        if runner_reason is not None:
            self._last_runner_reason = str(runner_reason)
        if wake_decision is not None:
            self._last_wake_decision = str(wake_decision)
        if wake_reason is not None:
            self._last_wake_reason = str(wake_reason)

    def _transcript_has_wake(self, transcript: VoiceTranscript) -> bool:
        normalized = normalize_transcript_text(transcript.text)
        padded = f" {normalized} "
        wake = normalize_transcript_text(self._config.wake_word)
        return bool(wake) and f" {wake} " in padded

    def _component_snapshot_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {}

        microphone = _safe_snapshot(self._microphone)
        device = getattr(microphone, "device", None)
        if device is not None:
            metadata["microphone_device_name"] = getattr(device, "name", None)
            metadata["microphone_device_index"] = getattr(device, "index", None)

        vad = _safe_snapshot(self._vad)
        metadata["vad_last_energy"] = getattr(vad, "last_energy", None)
        metadata["vad_noise_floor"] = getattr(vad, "noise_floor", None)

        audio_preprocessing = _safe_snapshot(self._audio_preprocessor)
        if isinstance(audio_preprocessing, dict):
            metadata["audio_preprocessing"] = audio_preprocessing
            metadata["audio_preprocessing_provider"] = audio_preprocessing.get(
                "provider"
            )

        stt = _safe_snapshot(self._stt)
        metadata["stt_last_text"] = getattr(stt, "last_text", None)
        metadata["stt_empty_results"] = getattr(stt, "empty_results", None)
        metadata["stt_low_confidence_results"] = getattr(
            stt,
            "low_confidence_results",
            None,
        )

        playback = _safe_snapshot(self._playback)
        if playback is not None:
            playback_status = getattr(playback, "status", None)
            current_playback = getattr(playback, "current_playback", None)
            current_status = getattr(current_playback, "status", None)
            metadata["playback_status"] = getattr(
                playback_status,
                "value",
                playback_status,
            )
            metadata["playback_current_status"] = getattr(
                current_status,
                "value",
                current_status,
            )
            metadata["playback_queued"] = getattr(playback, "queued_chunks", None)
            metadata["playback_stopped_count"] = getattr(
                playback,
                "stopped_count",
                None,
            )
            metadata["playback_last_latency_ms"] = getattr(
                playback,
                "last_latency_ms",
                None,
            )
            metadata["playback_first_audio_latency_ms"] = getattr(
                playback,
                "last_first_audio_latency_ms",
                None,
            )
            metadata["playback_last_error"] = getattr(
                playback,
                "last_error",
                None,
            )

        metadata["cognitive_route_action"] = self._last_cognitive_route_action
        metadata["cognitive_route_state"] = self._last_cognitive_route_state
        metadata["cognitive_route_reason"] = self._last_cognitive_route_reason
        metadata["cognitive_router"] = self._cognitive_router.snapshot()
        metadata["perception_intent_state"] = self._last_perception_intent_state
        metadata["perception_reason"] = self._last_perception_reason
        metadata["perception_confidence"] = self._last_perception_confidence
        metadata["perception_stability"] = self._last_perception_stability
        metadata["perception"] = self._perception.snapshot()
        metadata["threaded_interruption"] = self._threaded_interruption_metadata()
        metadata["fsm_transitions"] = self._fsm_transitions
        metadata["fsm_violations"] = self._fsm_violations
        if self._last_fsm_transition is not None:
            metadata["last_fsm_transition"] = (
                self._last_fsm_transition.to_metadata()
            )
        if self._last_fsm_violation is not None:
            metadata["last_fsm_violation"] = self._last_fsm_violation.to_metadata()

        return {key: value for key, value in metadata.items() if value is not None}

    def _live_snapshot_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "last_result_message": self._last_result_message,
            "last_gate_reason": self._last_gate_reason,
            "last_gate_text": self._last_gate_text,
            "last_cognition_status": self._last_cognition_status,
            "last_cognition_message": self._last_cognition_message,
            "last_cognition_safety": self._last_cognition_safety,
            "last_runner_status": self._last_runner_status,
            "last_runner_reason": self._last_runner_reason,
            "last_wake_decision": self._last_wake_decision,
            "last_wake_reason": self._last_wake_reason,
            "reflex_responses": self._reflex_responses,
            "last_reflex_kind": self._last_reflex_kind,
            "last_reflex_reason": self._last_reflex_reason,
            "audio_preprocessed_frames": self._preprocessed_frames,
            "audio_dropped_frames": self._dropped_audio_frames,
            "last_audio_preprocessing_status": (
                self._last_audio_preprocessing_status
            ),
            "last_audio_preprocessing_message": (
                self._last_audio_preprocessing_message
            ),
            "last_perception_intent_state": self._last_perception_intent_state,
            "last_perception_reason": self._last_perception_reason,
            "last_perception_confidence": self._last_perception_confidence,
            "last_perception_stability": self._last_perception_stability,
            "last_playback_status": self._last_playback_status,
            "playback_status": self._last_playback_status,
            "cognitive_route_action": self._last_cognitive_route_action,
            "cognitive_route_state": self._last_cognitive_route_state,
            "cognitive_route_reason": self._last_cognitive_route_reason,
            "perception_intent_state": self._last_perception_intent_state,
            "perception_reason": self._last_perception_reason,
            "perception_confidence": self._last_perception_confidence,
            "perception_stability": self._last_perception_stability,
            "perception": {
                "intent_state": self._last_perception_intent_state,
                "reason": self._last_perception_reason,
                "confidence": self._last_perception_confidence,
                "stability": self._last_perception_stability,
            },
            "threaded_interruption": self._threaded_interruption_metadata(),
            "fsm_transitions": self._fsm_transitions,
            "fsm_violations": self._fsm_violations,
        }
        if self._last_fsm_transition is not None:
            metadata["last_fsm_transition"] = (
                self._last_fsm_transition.to_metadata()
            )
        if self._last_fsm_violation is not None:
            metadata["last_fsm_violation"] = self._last_fsm_violation.to_metadata()
        return {key: value for key, value in metadata.items() if value is not None}

    def _threaded_interruption_metadata(self) -> dict[str, object]:
        top = (
            self._conversation_thread_stack[-1].to_metadata()
            if self._conversation_thread_stack
            else None
        )
        return {
            "enabled": self._policy.threaded_interruption_resume_enabled,
            "active_depth": len(self._conversation_thread_stack),
            "last_event": self._last_thread_event,
            "last_context": self._last_thread_context,
            "top": top,
        }


def _compact_thread_text(text: str, *, limit: int) -> str:
    normalized = " ".join(text.strip().split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)].rstrip() + "..."


def _merge_transcript_texts(prefix_text: str, final_text: str) -> str:
    prefix_words = normalize_transcript_text(prefix_text).split()
    final_words = normalize_transcript_text(final_text).split()

    if not prefix_words:
        return " ".join(final_words)
    if not final_words:
        return " ".join(prefix_words)

    prefix = " ".join(prefix_words)
    final = " ".join(final_words)
    if final in prefix:
        return prefix
    if prefix in final:
        return final

    overlap = 0
    max_overlap = min(len(prefix_words), len(final_words))
    for size in range(max_overlap, 0, -1):
        if prefix_words[-size:] == final_words[:size]:
            overlap = size
            break

    return " ".join((*prefix_words, *final_words[overlap:]))


def _safe_snapshot(component: object) -> object | None:
    snapshot = getattr(component, "snapshot", None)
    if not callable(snapshot):
        return None
    try:
        result: object = snapshot()
        return result
    except Exception:
        return None


def _voice_session_status_transition_allowed(
    current: VoiceSessionLoopStatus,
    target: VoiceSessionLoopStatus,
) -> bool:
    if current == target:
        return True
    if target in {
        VoiceSessionLoopStatus.DEGRADED,
        VoiceSessionLoopStatus.FAILED,
        VoiceSessionLoopStatus.RECOVERING,
        VoiceSessionLoopStatus.STOPPING,
    }:
        return True

    allowed: dict[VoiceSessionLoopStatus, set[VoiceSessionLoopStatus]] = {
        VoiceSessionLoopStatus.CREATED: {
            VoiceSessionLoopStatus.STARTING,
            VoiceSessionLoopStatus.STOPPING,
        },
        VoiceSessionLoopStatus.STARTING: {
            VoiceSessionLoopStatus.LISTENING,
            VoiceSessionLoopStatus.STOPPING,
        },
        VoiceSessionLoopStatus.LISTENING: {
            VoiceSessionLoopStatus.USER_SPEAKING,
            VoiceSessionLoopStatus.THINKING,
            VoiceSessionLoopStatus.SYNTHESIZING,
            VoiceSessionLoopStatus.STOPPING,
        },
        VoiceSessionLoopStatus.USER_SPEAKING: {
            VoiceSessionLoopStatus.LISTENING,
            VoiceSessionLoopStatus.THINKING,
            VoiceSessionLoopStatus.INTERRUPTED,
            VoiceSessionLoopStatus.STOPPING,
        },
        VoiceSessionLoopStatus.THINKING: {
            VoiceSessionLoopStatus.LISTENING,
            VoiceSessionLoopStatus.SYNTHESIZING,
            VoiceSessionLoopStatus.STOPPING,
        },
        VoiceSessionLoopStatus.SYNTHESIZING: {
            VoiceSessionLoopStatus.SPEAKING,
            VoiceSessionLoopStatus.LISTENING,
            VoiceSessionLoopStatus.STOPPING,
        },
        VoiceSessionLoopStatus.SPEAKING: {
            VoiceSessionLoopStatus.LISTENING,
            VoiceSessionLoopStatus.USER_SPEAKING,
            VoiceSessionLoopStatus.INTERRUPTED,
            VoiceSessionLoopStatus.STOPPING,
        },
        VoiceSessionLoopStatus.INTERRUPTED: {
            VoiceSessionLoopStatus.USER_SPEAKING,
            VoiceSessionLoopStatus.THINKING,
            VoiceSessionLoopStatus.LISTENING,
            VoiceSessionLoopStatus.STOPPING,
        },
        VoiceSessionLoopStatus.RECOVERING: {
            VoiceSessionLoopStatus.LISTENING,
            VoiceSessionLoopStatus.DEGRADED,
            VoiceSessionLoopStatus.FAILED,
            VoiceSessionLoopStatus.STOPPING,
        },
        VoiceSessionLoopStatus.DEGRADED: {
            VoiceSessionLoopStatus.LISTENING,
            VoiceSessionLoopStatus.RECOVERING,
            VoiceSessionLoopStatus.FAILED,
            VoiceSessionLoopStatus.STOPPING,
        },
        VoiceSessionLoopStatus.FAILED: {
            VoiceSessionLoopStatus.RECOVERING,
            VoiceSessionLoopStatus.STOPPING,
            VoiceSessionLoopStatus.STOPPED,
        },
        VoiceSessionLoopStatus.STOPPING: {
            VoiceSessionLoopStatus.STOPPED,
        },
        VoiceSessionLoopStatus.STOPPED: {
            VoiceSessionLoopStatus.STARTING,
        },
    }
    return target in allowed.get(current, set())


def _is_async_playback_result(result: VoicePlaybackResult) -> bool:
    async_playback = result.metadata.get("async_playback")
    return (
        async_playback is True
        and result.status == VoicePlaybackRuntimeStatus.PLAYING
    )

