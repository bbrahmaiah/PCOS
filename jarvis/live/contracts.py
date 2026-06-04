from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import NewType
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(UTC)


LiveSessionId = NewType("LiveSessionId", str)
LiveEventId = NewType("LiveEventId", str)
LiveTurnId = NewType("LiveTurnId", str)
LiveAudioFrameId = NewType("LiveAudioFrameId", str)
LiveTranscriptId = NewType("LiveTranscriptId", str)
LiveResponseId = NewType("LiveResponseId", str)


class LiveSessionStatus(StrEnum):
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class LiveSessionPhase(StrEnum):
    BOOTING = "booting"
    READY = "ready"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    RECOVERING = "recovering"
    SHUTTING_DOWN = "shutting_down"


class LiveSessionMode(StrEnum):
    SAFE_SIMULATION = "safe_simulation"
    REAL_VOICE = "real_voice"
    REAL_ENVIRONMENT = "real_environment"
    DAILY_DRIVER = "daily_driver"


class LiveInteractionState(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    WAKE_DETECTED = "wake_detected"
    USER_SPEAKING = "user_speaking"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    WAITING_FOR_USER = "waiting_for_user"
    SHUTTING_DOWN = "shutting_down"


class LiveWakeState(StrEnum):
    DISABLED = "disabled"
    SLEEPING = "sleeping"
    LISTENING_FOR_WAKE = "listening_for_wake"
    ENGAGED = "engaged"


class LiveAudioState(StrEnum):
    INACTIVE = "inactive"
    MICROPHONE_READY = "microphone_ready"
    STT_READY = "stt_ready"
    TTS_READY = "tts_ready"
    PLAYBACK_READY = "playback_ready"
    STREAMING_INPUT = "streaming_input"
    STREAMING_OUTPUT = "streaming_output"
    FAILED = "failed"


class LiveSubsystem(StrEnum):
    RUNTIME_KERNEL = "runtime_kernel"
    EVENT_BUS = "event_bus"
    PRESENCE = "presence"
    CONVERSATION = "conversation"
    COGNITION = "cognition"
    MEMORY = "memory"
    TOOLS = "tools"
    ORCHESTRATION = "orchestration"
    LATENCY = "latency"
    ENVIRONMENT = "environment"
    DEVELOPER_PACK = "developer_pack"
    COGNITIVE_SESSION = "cognitive_session"
    MICROPHONE = "microphone"
    STT = "stt"
    TTS = "tts"
    PLAYBACK = "playback"
    WAKE = "wake"
    INTERRUPTION = "interruption"
    HEALTH_MONITOR = "health_monitor"
    RECOVERY = "recovery"
    RESPONSE_GENERATOR = "response_generator"


class LiveSubsystemStatus(StrEnum):
    UNKNOWN = "unknown"
    READY = "ready"
    DEGRADED = "degraded"
    DISABLED = "disabled"
    FAILED = "failed"


class LiveHealthStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"


class LiveEventKind(StrEnum):
    SESSION_START_REQUESTED = "session_start_requested"
    SESSION_STARTED = "session_started"
    SESSION_STOP_REQUESTED = "session_stop_requested"
    SESSION_STOPPED = "session_stopped"
    WAKE_DETECTED = "wake_detected"
    USER_SPEECH_STARTED = "user_speech_started"
    USER_SPEECH_ENDED = "user_speech_ended"
    TRANSCRIPT_READY = "transcript_ready"
    ASSISTANT_RESPONSE_STARTED = "assistant_response_started"
    ASSISTANT_RESPONSE_FINISHED = "assistant_response_finished"
    INTERRUPTION_REQUESTED = "interruption_requested"
    INTERRUPTION_HANDLED = "interruption_handled"
    MEMORY_CONTEXT_UPDATED = "memory_context_updated"
    GOAL_UPDATED = "goal_updated"
    PLAN_UPDATED = "plan_updated"
    ENVIRONMENT_CONTEXT_UPDATED = "environment_context_updated"
    DEVELOPER_SIGNAL_RECEIVED = "developer_signal_received"
    HEALTH_CHANGED = "health_changed"
    RECOVERY_STARTED = "recovery_started"
    RECOVERY_FINISHED = "recovery_finished"
    ERROR = "error"


class LiveEventPriority(StrEnum):
    BACKGROUND = "background"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class LiveShutdownReason(StrEnum):
    USER_REQUEST = "user_request"
    KEYBOARD_INTERRUPT = "keyboard_interrupt"
    HEALTH_FAILURE = "health_failure"
    RECOVERY_FAILED = "recovery_failed"
    DEPENDENCY_MISSING = "dependency_missing"
    UNKNOWN = "unknown"


class LiveAudioFrameKind(StrEnum):
    INPUT = "input"
    OUTPUT = "output"
    LOOPBACK = "loopback"


class LiveTranscriptKind(StrEnum):
    PARTIAL = "partial"
    FINAL = "final"
    INTERRUPTION = "interruption"


class LiveResponseKind(StrEnum):
    CONVERSATIONAL = "conversational"
    DIAGNOSTIC = "diagnostic"
    SAFETY = "safety"
    RECOVERY = "recovery"
    SHUTDOWN = "shutdown"


class LiveResponseGenerationSource(StrEnum):
    RESPONSE_GENERATOR = "response_generator"
    COGNITION_RUNTIME = "cognition_runtime"
    DIAGNOSTIC_SYSTEM = "diagnostic_system"
    EMERGENCY_FALLBACK = "emergency_fallback"


class LiveResponseSafety(StrEnum):
    SAFE_TO_SPEAK = "safe_to_speak"
    REQUIRES_REVIEW = "requires_review"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class LiveAudioFrame:
    frame_id: LiveAudioFrameId
    kind: LiveAudioFrameKind
    sample_rate_hz: int
    channels: int
    duration_ms: int
    created_at: datetime
    rms: float | None = None
    speech_probability: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.frame_id).strip():
            raise ValueError("live audio frame_id cannot be empty.")
        if self.sample_rate_hz < 1:
            raise ValueError("sample_rate_hz must be positive.")
        if self.channels < 1:
            raise ValueError("channels must be positive.")
        if self.duration_ms < 1:
            raise ValueError("duration_ms must be positive.")
        if self.rms is not None and self.rms < 0:
            raise ValueError("rms cannot be negative.")
        if self.speech_probability is not None:
            if not 0.0 <= self.speech_probability <= 1.0:
                raise ValueError(
                    "speech_probability must be between 0 and 1."
                )


@dataclass(frozen=True, slots=True)
class LiveTranscript:
    transcript_id: LiveTranscriptId
    turn_id: LiveTurnId
    kind: LiveTranscriptKind
    text: str
    confidence: float
    started_at: datetime
    finished_at: datetime | None = None
    language: str = "en"
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.transcript_id).strip():
            raise ValueError("live transcript_id cannot be empty.")
        if not str(self.turn_id).strip():
            raise ValueError("live transcript turn_id cannot be empty.")
        if not self.text.strip():
            raise ValueError("live transcript text cannot be empty.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "live transcript confidence must be between 0 and 1."
            )
        if not self.language.strip():
            raise ValueError("live transcript language cannot be empty.")


@dataclass(frozen=True, slots=True)
class LiveResponse:
    response_id: LiveResponseId
    turn_id: LiveTurnId
    kind: LiveResponseKind
    text: str
    generation_source: LiveResponseGenerationSource
    safety: LiveResponseSafety
    created_at: datetime
    token_count: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.response_id).strip():
            raise ValueError("live response_id cannot be empty.")
        if not str(self.turn_id).strip():
            raise ValueError("live response turn_id cannot be empty.")
        if not self.text.strip():
            raise ValueError("live response text cannot be empty.")
        if self.token_count is not None and self.token_count < 0:
            raise ValueError("live response token_count cannot be negative.")

    @property
    def is_conversational(self) -> bool:
        return self.kind == LiveResponseKind.CONVERSATIONAL

    @property
    def generated_by_cognition(self) -> bool:
        return self.generation_source in {
            LiveResponseGenerationSource.RESPONSE_GENERATOR,
            LiveResponseGenerationSource.COGNITION_RUNTIME,
        }

    @property
    def deterministic_system_response(self) -> bool:
        return self.generation_source in {
            LiveResponseGenerationSource.DIAGNOSTIC_SYSTEM,
            LiveResponseGenerationSource.EMERGENCY_FALLBACK,
        }


@dataclass(frozen=True, slots=True)
class LiveSessionConfig:
    user_label: str = "Balu"
    assistant_name: str = "JARVIS"
    mode: LiveSessionMode = LiveSessionMode.SAFE_SIMULATION
    wake_word: str = "jarvis"
    real_microphone_enabled: bool = False
    real_stt_enabled: bool = False
    real_tts_enabled: bool = False
    real_environment_enabled: bool = False
    tools_enabled: bool = False
    developer_pack_enabled: bool = True
    memory_enabled: bool = True
    goal_tracking_enabled: bool = True
    interruption_enabled: bool = True
    health_monitor_enabled: bool = True
    recovery_enabled: bool = True
    max_session_seconds: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.user_label.strip():
            raise ValueError("live session user_label cannot be empty.")
        if not self.assistant_name.strip():
            raise ValueError("live session assistant_name cannot be empty.")
        if not self.wake_word.strip():
            raise ValueError("live session wake_word cannot be empty.")
        if self.max_session_seconds is not None:
            if self.max_session_seconds < 1:
                raise ValueError(
                    "max_session_seconds must be positive when provided."
                )


@dataclass(frozen=True, slots=True)
class LiveSubsystemState:
    subsystem: LiveSubsystem
    status: LiveSubsystemStatus
    message: str
    updated_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise ValueError("live subsystem message cannot be empty.")


@dataclass(frozen=True, slots=True)
class LiveSessionState:
    session_id: LiveSessionId
    status: LiveSessionStatus
    phase: LiveSessionPhase
    mode: LiveSessionMode
    interaction_state: LiveInteractionState
    wake_state: LiveWakeState
    audio_state: LiveAudioState
    health_status: LiveHealthStatus
    user_label: str
    assistant_name: str
    started_at: datetime | None
    updated_at: datetime
    user_present: bool
    microphone_active: bool
    stt_active: bool
    tts_active: bool
    playback_active: bool
    assistant_speaking: bool
    conversation_active: bool
    interruption_enabled: bool
    wake_enabled: bool
    environment_enabled: bool
    memory_enabled: bool
    goal_tracking_enabled: bool
    developer_pack_enabled: bool
    tools_enabled: bool
    subsystem_states: tuple[LiveSubsystemState, ...] = ()
    active_topic: str | None = None
    current_goal_id: str | None = None
    current_plan_id: str | None = None
    current_turn_id: LiveTurnId | None = None
    last_transcript: LiveTranscript | None = None
    last_response: LiveResponse | None = None
    shutdown_reason: LiveShutdownReason | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.session_id).strip():
            raise ValueError("live session_id cannot be empty.")
        if not self.user_label.strip():
            raise ValueError("live session user_label cannot be empty.")
        if not self.assistant_name.strip():
            raise ValueError("live session assistant_name cannot be empty.")

    @property
    def is_running(self) -> bool:
        return self.status == LiveSessionStatus.RUNNING

    @property
    def can_listen(self) -> bool:
        return (
            self.is_running
            and self.microphone_active
            and self.stt_active
            and self.interaction_state
            in {
                LiveInteractionState.IDLE,
                LiveInteractionState.LISTENING,
                LiveInteractionState.WAITING_FOR_USER,
            }
        )

    @property
    def can_speak(self) -> bool:
        return self.is_running and self.tts_active and self.playback_active

    @property
    def can_interrupt(self) -> bool:
        return self.is_running and self.interruption_enabled

    @property
    def ready_subsystems(self) -> tuple[LiveSubsystemState, ...]:
        return tuple(
            state
            for state in self.subsystem_states
            if state.status == LiveSubsystemStatus.READY
        )

    @property
    def failed_subsystems(self) -> tuple[LiveSubsystemState, ...]:
        return tuple(
            state
            for state in self.subsystem_states
            if state.status == LiveSubsystemStatus.FAILED
        )


@dataclass(frozen=True, slots=True)
class LiveSessionEvent:
    event_id: LiveEventId
    kind: LiveEventKind
    priority: LiveEventPriority
    source: LiveSubsystem
    title: str
    summary: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.event_id).strip():
            raise ValueError("live event_id cannot be empty.")
        if not self.title.strip():
            raise ValueError("live event title cannot be empty.")
        if not self.summary.strip():
            raise ValueError("live event summary cannot be empty.")


@dataclass(frozen=True, slots=True)
class LiveSessionSnapshot:
    state: LiveSessionState
    event_count: int
    uptime_seconds: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def running(self) -> bool:
        return self.state.is_running


@dataclass(frozen=True, slots=True)
class LiveSessionDesignCheck:
    subsystem: LiveSubsystem
    passed: bool
    message: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LiveSessionDesignGateReport:
    passed: bool
    checks: tuple[LiveSessionDesignCheck, ...]
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def passed_count(self) -> int:
        return sum(1 for check in self.checks if check.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for check in self.checks if not check.passed)


class LiveSessionDesignGate:
    """
    Step 50A Live Session Design Gate.

    This validates that the live runtime has contracts for every completed
    JARVIS organ without starting real microphone, STT, TTS, tools, or
    environment control.

    It is a contract gate only.
    """

    def validate(self, state: LiveSessionState) -> LiveSessionDesignGateReport:
        required = (
            LiveSubsystem.RUNTIME_KERNEL,
            LiveSubsystem.EVENT_BUS,
            LiveSubsystem.PRESENCE,
            LiveSubsystem.CONVERSATION,
            LiveSubsystem.COGNITION,
            LiveSubsystem.MEMORY,
            LiveSubsystem.TOOLS,
            LiveSubsystem.ORCHESTRATION,
            LiveSubsystem.LATENCY,
            LiveSubsystem.ENVIRONMENT,
            LiveSubsystem.DEVELOPER_PACK,
            LiveSubsystem.COGNITIVE_SESSION,
            LiveSubsystem.MICROPHONE,
            LiveSubsystem.STT,
            LiveSubsystem.TTS,
            LiveSubsystem.PLAYBACK,
            LiveSubsystem.WAKE,
            LiveSubsystem.INTERRUPTION,
            LiveSubsystem.HEALTH_MONITOR,
            LiveSubsystem.RECOVERY,
            LiveSubsystem.RESPONSE_GENERATOR,
        )
        states = {item.subsystem: item for item in state.subsystem_states}
        checks = tuple(
            LiveSessionDesignCheck(
                subsystem=subsystem,
                passed=subsystem in states,
                message=(
                    f"{subsystem.value} contract is present"
                    if subsystem in states
                    else f"{subsystem.value} contract is missing"
                ),
                created_at=utc_now(),
                metadata={
                    "status": (
                        states[subsystem].status.value
                        if subsystem in states
                        else LiveSubsystemStatus.UNKNOWN.value
                    )
                },
            )
            for subsystem in required
        )

        return LiveSessionDesignGateReport(
            passed=all(check.passed for check in checks),
            checks=checks,
            created_at=utc_now(),
            metadata={
                "session_id": str(state.session_id),
                "mode": state.mode.value,
                "status": state.status.value,
                "phase": state.phase.value,
            },
        )


def default_live_session_config() -> LiveSessionConfig:
    return LiveSessionConfig(
        user_label="Balu",
        assistant_name="JARVIS",
        mode=LiveSessionMode.SAFE_SIMULATION,
        wake_word="jarvis",
        real_microphone_enabled=False,
        real_stt_enabled=False,
        real_tts_enabled=False,
        real_environment_enabled=False,
        tools_enabled=False,
        developer_pack_enabled=True,
        memory_enabled=True,
        goal_tracking_enabled=True,
        interruption_enabled=True,
        health_monitor_enabled=True,
        recovery_enabled=True,
        metadata={"step": "50A", "purpose": "live_session_contracts"},
    )


def default_live_session_state(
    *,
    config: LiveSessionConfig | None = None,
) -> LiveSessionState:
    cfg = config or default_live_session_config()
    now = utc_now()
    return LiveSessionState(
        session_id=LiveSessionId(f"live_{uuid4().hex}"),
        status=LiveSessionStatus.CREATED,
        phase=LiveSessionPhase.BOOTING,
        mode=cfg.mode,
        interaction_state=LiveInteractionState.IDLE,
        wake_state=(
            LiveWakeState.LISTENING_FOR_WAKE
            if cfg.wake_word.strip()
            else LiveWakeState.DISABLED
        ),
        audio_state=LiveAudioState.INACTIVE,
        health_status=LiveHealthStatus.HEALTHY,
        user_label=cfg.user_label,
        assistant_name=cfg.assistant_name,
        started_at=None,
        updated_at=now,
        user_present=False,
        microphone_active=cfg.real_microphone_enabled,
        stt_active=cfg.real_stt_enabled,
        tts_active=cfg.real_tts_enabled,
        playback_active=cfg.real_tts_enabled,
        assistant_speaking=False,
        conversation_active=False,
        interruption_enabled=cfg.interruption_enabled,
        wake_enabled=bool(cfg.wake_word.strip()),
        environment_enabled=cfg.real_environment_enabled,
        memory_enabled=cfg.memory_enabled,
        goal_tracking_enabled=cfg.goal_tracking_enabled,
        developer_pack_enabled=cfg.developer_pack_enabled,
        tools_enabled=cfg.tools_enabled,
        subsystem_states=_default_subsystem_states(cfg),
        metadata=cfg.metadata,
    )


def make_live_event(
    *,
    kind: LiveEventKind,
    priority: LiveEventPriority,
    source: LiveSubsystem,
    title: str,
    summary: str,
    metadata: dict[str, object] | None = None,
) -> LiveSessionEvent:
    return LiveSessionEvent(
        event_id=LiveEventId(f"live_evt_{uuid4().hex}"),
        kind=kind,
        priority=priority,
        source=source,
        title=title,
        summary=summary,
        created_at=utc_now(),
        metadata=metadata or {},
    )


def make_live_audio_frame(
    *,
    kind: LiveAudioFrameKind,
    sample_rate_hz: int,
    channels: int,
    duration_ms: int,
    rms: float | None = None,
    speech_probability: float | None = None,
    metadata: dict[str, object] | None = None,
) -> LiveAudioFrame:
    return LiveAudioFrame(
        frame_id=LiveAudioFrameId(f"audio_{uuid4().hex}"),
        kind=kind,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        duration_ms=duration_ms,
        rms=rms,
        speech_probability=speech_probability,
        created_at=utc_now(),
        metadata=metadata or {},
    )


def make_live_turn_id() -> LiveTurnId:
    return LiveTurnId(f"turn_{uuid4().hex}")


def make_live_transcript(
    *,
    turn_id: LiveTurnId,
    kind: LiveTranscriptKind,
    text: str,
    confidence: float,
    finished_at: datetime | None = None,
    language: str = "en",
    metadata: dict[str, object] | None = None,
) -> LiveTranscript:
    return LiveTranscript(
        transcript_id=LiveTranscriptId(f"transcript_{uuid4().hex}"),
        turn_id=turn_id,
        kind=kind,
        text=text,
        confidence=confidence,
        started_at=utc_now(),
        finished_at=finished_at,
        language=language,
        metadata=metadata or {},
    )


def make_live_response(
    *,
    turn_id: LiveTurnId,
    kind: LiveResponseKind,
    text: str,
    generation_source: LiveResponseGenerationSource,
    safety: LiveResponseSafety,
    token_count: int | None = None,
    metadata: dict[str, object] | None = None,
) -> LiveResponse:
    return LiveResponse(
        response_id=LiveResponseId(f"response_{uuid4().hex}"),
        turn_id=turn_id,
        kind=kind,
        text=text,
        generation_source=generation_source,
        safety=safety,
        token_count=token_count,
        created_at=utc_now(),
        metadata=metadata or {},
    )


def _default_subsystem_states(
    config: LiveSessionConfig,
) -> tuple[LiveSubsystemState, ...]:
    now = utc_now()

    base_ready = {
        LiveSubsystem.RUNTIME_KERNEL,
        LiveSubsystem.EVENT_BUS,
        LiveSubsystem.PRESENCE,
        LiveSubsystem.CONVERSATION,
        LiveSubsystem.COGNITION,
        LiveSubsystem.MEMORY,
        LiveSubsystem.TOOLS,
        LiveSubsystem.ORCHESTRATION,
        LiveSubsystem.LATENCY,
        LiveSubsystem.ENVIRONMENT,
        LiveSubsystem.DEVELOPER_PACK,
        LiveSubsystem.COGNITIVE_SESSION,
        LiveSubsystem.WAKE,
        LiveSubsystem.INTERRUPTION,
        LiveSubsystem.HEALTH_MONITOR,
        LiveSubsystem.RECOVERY,
        LiveSubsystem.RESPONSE_GENERATOR,
    }

    conditional = {
        LiveSubsystem.MICROPHONE: config.real_microphone_enabled,
        LiveSubsystem.STT: config.real_stt_enabled,
        LiveSubsystem.TTS: config.real_tts_enabled,
        LiveSubsystem.PLAYBACK: config.real_tts_enabled,
    }

    states: list[LiveSubsystemState] = []

    for subsystem in LiveSubsystem:
        if subsystem in conditional:
            enabled = conditional[subsystem]
            states.append(
                LiveSubsystemState(
                    subsystem=subsystem,
                    status=(
                        LiveSubsystemStatus.READY
                        if enabled
                        else LiveSubsystemStatus.DISABLED
                    ),
                    message=(
                        f"{subsystem.value} enabled"
                        if enabled
                        else f"{subsystem.value} disabled in safe simulation"
                    ),
                    updated_at=now,
                )
            )
            continue

        states.append(
            LiveSubsystemState(
                subsystem=subsystem,
                status=(
                    LiveSubsystemStatus.READY
                    if subsystem in base_ready
                    else LiveSubsystemStatus.UNKNOWN
                ),
                message=f"{subsystem.value} contract initialized",
                updated_at=now,
            )
        )

    return tuple(states)