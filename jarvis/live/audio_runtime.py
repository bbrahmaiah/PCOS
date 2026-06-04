from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol, cast
from uuid import uuid4

from jarvis.live.contracts import (
    LiveAudioFrame,
    LiveAudioFrameKind,
    LiveResponse,
    LiveSessionConfig,
    LiveSessionMode,
    LiveSubsystem,
    LiveSubsystemState,
    LiveSubsystemStatus,
    LiveTranscript,
    LiveTranscriptKind,
    LiveTurnId,
    make_live_audio_frame,
    make_live_transcript,
    utc_now,
)
from jarvis.live.response_boundary import (
    LiveResponseBoundaryResult,
    LiveResponseBoundaryRuntime,
)
from jarvis.live.session_state import (
    LiveSessionStateRuntime,
    LiveSessionStateRuntimeResult,
    LiveSessionStateRuntimeStatus,
)


class LiveAudioRuntimeStatus(StrEnum):
    READY = "ready"
    DEGRADED = "degraded"
    BLOCKED = "blocked"


class LiveAudioRuntimeOperation(StrEnum):
    PREPARE = "prepare"
    CAPTURE_FRAME = "capture_frame"
    TRANSCRIBE_FRAME = "transcribe_frame"
    SYNTHESIZE_RESPONSE = "synthesize_response"
    PLAY_RESPONSE = "play_response"
    STOP_OUTPUT = "stop_output"
    SNAPSHOT = "snapshot"


class LiveAudioDeviceKind(StrEnum):
    MICROPHONE = "microphone"
    STT = "stt"
    TTS = "tts"
    PLAYBACK = "playback"


@dataclass(frozen=True, slots=True)
class LiveAudioRuntimeConfig:
    sample_rate_hz: int = 16000
    channels: int = 1
    frame_duration_ms: int = 20
    allow_loopback_frames: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sample_rate_hz < 1:
            raise ValueError("live audio sample_rate_hz must be positive.")
        if self.channels < 1:
            raise ValueError("live audio channels must be positive.")
        if self.frame_duration_ms < 1:
            raise ValueError("live audio frame_duration_ms must be positive.")


@dataclass(frozen=True, slots=True)
class LiveAudioBuffer:
    buffer_id: str
    sample_rate_hz: int
    channels: int
    duration_ms: int
    pcm: bytes
    created_at: datetime
    response_id: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.buffer_id.strip():
            raise ValueError("live audio buffer_id cannot be empty.")
        if self.sample_rate_hz < 1:
            raise ValueError("live audio buffer sample_rate_hz must be positive.")
        if self.channels < 1:
            raise ValueError("live audio buffer channels must be positive.")
        if self.duration_ms < 1:
            raise ValueError("live audio buffer duration_ms must be positive.")
        if not self.pcm:
            raise ValueError("live audio buffer pcm cannot be empty.")


@dataclass(frozen=True, slots=True)
class LiveAudioAdapterReport:
    kind: LiveAudioDeviceKind
    ready: bool
    message: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise ValueError("live audio adapter report message cannot be empty.")


@dataclass(frozen=True, slots=True)
class LiveAudioRuntimeResult:
    status: LiveAudioRuntimeStatus
    operation: LiveAudioRuntimeOperation
    frame: LiveAudioFrame | None
    transcript: LiveTranscript | None
    buffer: LiveAudioBuffer | None
    response: LiveResponse | None
    boundary_result: LiveResponseBoundaryResult | None
    live_state_result: LiveSessionStateRuntimeResult | None
    reason: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == LiveAudioRuntimeStatus.READY


@dataclass(frozen=True, slots=True)
class LiveAudioRuntimeSnapshot:
    status: LiveAudioRuntimeStatus
    prepared: bool
    captured_frames: int
    transcripts: int
    synthesized_responses: int
    played_responses: int
    blocked_count: int
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


class LiveMicrophoneAdapter(Protocol):
    def prepare(self, config: LiveAudioRuntimeConfig) -> LiveAudioAdapterReport:
        """Prepare microphone capture."""

    def capture_frame(self, config: LiveAudioRuntimeConfig) -> LiveAudioFrame:
        """Capture one audio frame."""


class LiveSTTAdapter(Protocol):
    def prepare(self, config: LiveAudioRuntimeConfig) -> LiveAudioAdapterReport:
        """Prepare STT."""

    def transcribe(
        self,
        frame: LiveAudioFrame,
        turn_id: LiveTurnId,
    ) -> LiveTranscript:
        """Convert audio frame into transcript."""


class LiveTTSAdapter(Protocol):
    def prepare(self, config: LiveAudioRuntimeConfig) -> LiveAudioAdapterReport:
        """Prepare TTS."""

    def synthesize(self, response: LiveResponse) -> LiveAudioBuffer:
        """Convert generated LiveResponse into audio."""


class LivePlaybackAdapter(Protocol):
    def prepare(self, config: LiveAudioRuntimeConfig) -> LiveAudioAdapterReport:
        """Prepare playback."""

    def play(self, buffer: LiveAudioBuffer) -> LiveAudioAdapterReport:
        """Play synthesized audio."""

    def stop(self) -> LiveAudioAdapterReport:
        """Stop playback immediately."""


class LiveAudioRuntime:
    """
    Step 50D Live Audio Runtime.

    This is the governed audio boundary.

    It consumes existing/future microphone, STT, TTS, and playback adapters.
    It does not duplicate Presence Runtime.
    It does not create conversational responses.
    It does not call cognition.
    It does not access memory.
    It does not execute tools.

    TTS may only receive a LiveResponse that passed the 50A.5 response boundary.
    """

    def __init__(
        self,
        *,
        live_state: LiveSessionStateRuntime | None = None,
        response_boundary: LiveResponseBoundaryRuntime | None = None,
        microphone: LiveMicrophoneAdapter | None = None,
        stt: LiveSTTAdapter | None = None,
        tts: LiveTTSAdapter | None = None,
        playback: LivePlaybackAdapter | None = None,
        config: LiveAudioRuntimeConfig | None = None,
    ) -> None:
        self._state = live_state or LiveSessionStateRuntime(
            config=LiveSessionConfig(mode=LiveSessionMode.SAFE_SIMULATION)
        )
        self._boundary = response_boundary or LiveResponseBoundaryRuntime()
        self._microphone = microphone
        self._stt = stt
        self._tts = tts
        self._playback = playback
        self._config = config or LiveAudioRuntimeConfig()
        self._prepared = False
        self._captured_frames = 0
        self._transcripts = 0
        self._synthesized_responses = 0
        self._played_responses = 0
        self._blocked_count = 0

    @property
    def live_state(self) -> LiveSessionStateRuntime:
        return self._state

    def prepare(self) -> LiveAudioRuntimeResult:
        reports = (
            self._prepare_microphone(),
            self._prepare_stt(),
            self._prepare_tts(),
            self._prepare_playback(),
        )
        ready = all(report.ready for report in reports)

        self._prepared = ready

        if not ready:
            self._blocked_count += 1
            return self._result(
                status=LiveAudioRuntimeStatus.DEGRADED,
                operation=LiveAudioRuntimeOperation.PREPARE,
                reason="one or more live audio adapters are not ready",
                metadata={
                    "reports": tuple(report.message for report in reports),
                },
            )

        return self._result(
            status=LiveAudioRuntimeStatus.READY,
            operation=LiveAudioRuntimeOperation.PREPARE,
            reason="live audio runtime prepared",
            metadata={
                "reports": tuple(report.message for report in reports),
            },
        )

    def capture_frame(self) -> LiveAudioRuntimeResult:
        if not self._prepared:
            return self._block(
                operation=LiveAudioRuntimeOperation.CAPTURE_FRAME,
                reason="live audio runtime must be prepared before capture",
            )

        if self._microphone is None:
            return self._block(
                operation=LiveAudioRuntimeOperation.CAPTURE_FRAME,
                reason="microphone adapter is not configured",
            )

        frame = self._microphone.capture_frame(self._config)
        self._captured_frames += 1

        return self._result(
            status=LiveAudioRuntimeStatus.READY,
            operation=LiveAudioRuntimeOperation.CAPTURE_FRAME,
            frame=frame,
            reason="audio frame captured",
        )

    def transcribe_frame(
        self,
        *,
        frame: LiveAudioFrame,
        turn_id: LiveTurnId,
    ) -> LiveAudioRuntimeResult:
        if not self._prepared:
            return self._block(
                operation=LiveAudioRuntimeOperation.TRANSCRIBE_FRAME,
                reason="live audio runtime must be prepared before STT",
            )

        if self._stt is None:
            return self._block(
                operation=LiveAudioRuntimeOperation.TRANSCRIBE_FRAME,
                reason="STT adapter is not configured",
            )

        transcript = self._stt.transcribe(frame, turn_id)
        live_state_result = self._state.transcript_ready(transcript)

        if live_state_result.status == LiveSessionStateRuntimeStatus.BLOCKED:
            return self._block(
                operation=LiveAudioRuntimeOperation.TRANSCRIBE_FRAME,
                reason="live session rejected transcript",
                live_state_result=live_state_result,
            )

        self._transcripts += 1

        return self._result(
            status=LiveAudioRuntimeStatus.READY,
            operation=LiveAudioRuntimeOperation.TRANSCRIBE_FRAME,
            frame=frame,
            transcript=transcript,
            live_state_result=live_state_result,
            reason="audio frame transcribed",
        )

    def synthesize_response(
        self,
        response: LiveResponse,
    ) -> LiveAudioRuntimeResult:
        if not self._prepared:
            return self._block(
                operation=LiveAudioRuntimeOperation.SYNTHESIZE_RESPONSE,
                reason="live audio runtime must be prepared before TTS",
            )

        boundary_result = self._boundary.validate_for_tts(response)
        if not boundary_result.succeeded:
            return self._block(
                operation=LiveAudioRuntimeOperation.SYNTHESIZE_RESPONSE,
                reason="response rejected by live response boundary",
                boundary_result=boundary_result,
            )

        if self._tts is None:
            return self._block(
                operation=LiveAudioRuntimeOperation.SYNTHESIZE_RESPONSE,
                reason="TTS adapter is not configured",
                boundary_result=boundary_result,
            )

        buffer = self._tts.synthesize(response)
        self._synthesized_responses += 1

        return self._result(
            status=LiveAudioRuntimeStatus.READY,
            operation=LiveAudioRuntimeOperation.SYNTHESIZE_RESPONSE,
            buffer=buffer,
            response=response,
            boundary_result=boundary_result,
            reason="generated response synthesized",
        )

    def play_response(
        self,
        response: LiveResponse,
    ) -> LiveAudioRuntimeResult:
        synth = self.synthesize_response(response)
        if not synth.succeeded:
            return synth

        if synth.buffer is None:
            return self._block(
                operation=LiveAudioRuntimeOperation.PLAY_RESPONSE,
                reason="synthesis did not produce an audio buffer",
                boundary_result=synth.boundary_result,
            )

        if self._playback is None:
            return self._block(
                operation=LiveAudioRuntimeOperation.PLAY_RESPONSE,
                reason="playback adapter is not configured",
                boundary_result=synth.boundary_result,
                buffer=synth.buffer,
                response=response,
            )

        live_state_result = self._state.start_speaking(response)
        if live_state_result.status == LiveSessionStateRuntimeStatus.BLOCKED:
            return self._block(
                operation=LiveAudioRuntimeOperation.PLAY_RESPONSE,
                reason="live session rejected speaking state",
                boundary_result=synth.boundary_result,
                buffer=synth.buffer,
                response=response,
                live_state_result=live_state_result,
            )

        playback_report = self._playback.play(synth.buffer)
        if not playback_report.ready:
            return self._block(
                operation=LiveAudioRuntimeOperation.PLAY_RESPONSE,
                reason="playback adapter rejected audio buffer",
                boundary_result=synth.boundary_result,
                buffer=synth.buffer,
                response=response,
                live_state_result=live_state_result,
                metadata={"playback_message": playback_report.message},
            )

        self._played_responses += 1

        return self._result(
            status=LiveAudioRuntimeStatus.READY,
            operation=LiveAudioRuntimeOperation.PLAY_RESPONSE,
            buffer=synth.buffer,
            response=response,
            boundary_result=synth.boundary_result,
            live_state_result=live_state_result,
            reason="generated response playback started",
            metadata={"playback_message": playback_report.message},
        )

    def stop_output(
        self,
        *,
        reason: str,
    ) -> LiveAudioRuntimeResult:
        if not reason.strip():
            raise ValueError("live audio stop reason cannot be empty.")

        playback_report: LiveAudioAdapterReport | None = None
        if self._playback is not None:
            playback_report = self._playback.stop()

        state_result = self._state.interrupt(reason=reason.strip())

        if state_result.status == LiveSessionStateRuntimeStatus.BLOCKED:
            return self._block(
                operation=LiveAudioRuntimeOperation.STOP_OUTPUT,
                reason="live session rejected audio interruption",
                live_state_result=state_result,
            )

        return self._result(
            status=LiveAudioRuntimeStatus.READY,
            operation=LiveAudioRuntimeOperation.STOP_OUTPUT,
            live_state_result=state_result,
            reason="live audio output stopped",
            metadata={
                "playback_message": (
                    playback_report.message
                    if playback_report is not None
                    else "no playback adapter"
                 )
            },
        )

    def snapshot(self) -> LiveAudioRuntimeSnapshot:
        return LiveAudioRuntimeSnapshot(
            status=LiveAudioRuntimeStatus.READY,
            prepared=self._prepared,
            captured_frames=self._captured_frames,
            transcripts=self._transcripts,
            synthesized_responses=self._synthesized_responses,
            played_responses=self._played_responses,
            blocked_count=self._blocked_count,
            created_at=utc_now(),
        )

    def _prepare_microphone(self) -> LiveAudioAdapterReport:
        return self._prepare_adapter(
            adapter=self._microphone,
            kind=LiveAudioDeviceKind.MICROPHONE,
            subsystem=LiveSubsystem.MICROPHONE,
        )

    def _prepare_stt(self) -> LiveAudioAdapterReport:
        return self._prepare_adapter(
            adapter=self._stt,
            kind=LiveAudioDeviceKind.STT,
            subsystem=LiveSubsystem.STT,
        )

    def _prepare_tts(self) -> LiveAudioAdapterReport:
        return self._prepare_adapter(
            adapter=self._tts,
            kind=LiveAudioDeviceKind.TTS,
            subsystem=LiveSubsystem.TTS,
        )

    def _prepare_playback(self) -> LiveAudioAdapterReport:
        return self._prepare_adapter(
            adapter=self._playback,
            kind=LiveAudioDeviceKind.PLAYBACK,
            subsystem=LiveSubsystem.PLAYBACK,
        )

    def _prepare_adapter(
        self,
        *,
        adapter: object,
        kind: LiveAudioDeviceKind,
        subsystem: LiveSubsystem,
    ) -> LiveAudioAdapterReport:
        if adapter is None:
            report = LiveAudioAdapterReport(
                kind=kind,
                ready=False,
                message=f"{kind.value} adapter not configured",
                created_at=utc_now(),
            )
            self._state.update_subsystem(
                LiveSubsystemState(
                    subsystem=subsystem,
                    status=LiveSubsystemStatus.DISABLED,
                    message=report.message,
                    updated_at=utc_now(),
                )
            )
            return report

        prepared = cast(LiveAudioAdapterReport, adapter.prepare(self._config))  # type: ignore[attr-defined]
        self._state.update_subsystem(
            LiveSubsystemState(
                subsystem=subsystem,
                status=(
                    LiveSubsystemStatus.READY
                    if prepared.ready
                    else LiveSubsystemStatus.FAILED
                ),
                message=prepared.message,
                updated_at=utc_now(),
                metadata=prepared.metadata,
            )
        )
        return prepared

    def _block(
        self,
        *,
        operation: LiveAudioRuntimeOperation,
        reason: str,
        frame: LiveAudioFrame | None = None,
        transcript: LiveTranscript | None = None,
        buffer: LiveAudioBuffer | None = None,
        response: LiveResponse | None = None,
        boundary_result: LiveResponseBoundaryResult | None = None,
        live_state_result: LiveSessionStateRuntimeResult | None = None,
        metadata: dict[str, object] | None = None,
    ) -> LiveAudioRuntimeResult:
        self._blocked_count += 1
        return self._result(
            status=LiveAudioRuntimeStatus.BLOCKED,
            operation=operation,
            frame=frame,
            transcript=transcript,
            buffer=buffer,
            response=response,
            boundary_result=boundary_result,
            live_state_result=live_state_result,
            reason=reason,
            metadata=metadata,
        )

    def _result(
        self,
        *,
        status: LiveAudioRuntimeStatus,
        operation: LiveAudioRuntimeOperation,
        frame: LiveAudioFrame | None = None,
        transcript: LiveTranscript | None = None,
        buffer: LiveAudioBuffer | None = None,
        response: LiveResponse | None = None,
        boundary_result: LiveResponseBoundaryResult | None = None,
        live_state_result: LiveSessionStateRuntimeResult | None = None,
        reason: str,
        metadata: dict[str, object] | None = None,
    ) -> LiveAudioRuntimeResult:
        return LiveAudioRuntimeResult(
            status=status,
            operation=operation,
            frame=frame,
            transcript=transcript,
            buffer=buffer,
            response=response,
            boundary_result=boundary_result,
            live_state_result=live_state_result,
            reason=reason,
            created_at=utc_now(),
            metadata=metadata or {},
        )


def make_live_audio_buffer(
    *,
    sample_rate_hz: int,
    channels: int,
    duration_ms: int,
    pcm: bytes,
    response_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> LiveAudioBuffer:
    return LiveAudioBuffer(
        buffer_id=f"audio_buffer_{uuid4().hex}",
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        duration_ms=duration_ms,
        pcm=pcm,
        response_id=response_id,
        created_at=utc_now(),
        metadata=metadata or {},
    )


def make_live_audio_adapter_report(
    *,
    kind: LiveAudioDeviceKind,
    ready: bool,
    message: str,
    metadata: dict[str, object] | None = None,
) -> LiveAudioAdapterReport:
    return LiveAudioAdapterReport(
        kind=kind,
        ready=ready,
        message=message,
        created_at=utc_now(),
        metadata=metadata or {},
    )


def fake_input_frame(
    *,
    config: LiveAudioRuntimeConfig,
    speech_probability: float = 0.8,
) -> LiveAudioFrame:
    return make_live_audio_frame(
        kind=LiveAudioFrameKind.INPUT,
        sample_rate_hz=config.sample_rate_hz,
        channels=config.channels,
        duration_ms=config.frame_duration_ms,
        rms=0.1,
        speech_probability=speech_probability,
    )


def fake_final_transcript(
    *,
    turn_id: LiveTurnId,
    text: str,
) -> LiveTranscript:
    return make_live_transcript(
        turn_id=turn_id,
        kind=LiveTranscriptKind.FINAL,
        text=text,
        confidence=0.95,
    )