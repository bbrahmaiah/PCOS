from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from typing import Any

from jarvis.live.contracts import (
    LiveAudioState,
    LiveEventKind,
    LiveEventPriority,
    LiveHealthStatus,
    LiveInteractionState,
    LiveResponse,
    LiveSessionConfig,
    LiveSessionEvent,
    LiveSessionPhase,
    LiveSessionState,
    LiveSessionStatus,
    LiveShutdownReason,
    LiveSubsystem,
    LiveSubsystemState,
    LiveSubsystemStatus,
    LiveTranscript,
    default_live_session_config,
    default_live_session_state,
    make_live_event,
    make_live_turn_id,
    utc_now,
)


class LiveSessionStateRuntimeStatus(StrEnum):
    READY = "ready"
    BLOCKED = "blocked"


class LiveSessionStateOperation(StrEnum):
    START = "start"
    MARK_READY = "mark_ready"
    ENTER_LISTENING = "enter_listening"
    START_USER_TURN = "start_user_turn"
    TRANSCRIPT_READY = "transcript_ready"
    START_THINKING = "start_thinking"
    START_SPEAKING = "start_speaking"
    FINISH_SPEAKING = "finish_speaking"
    INTERRUPT = "interrupt"
    ENTER_RECOVERY = "enter_recovery"
    FINISH_RECOVERY = "finish_recovery"
    UPDATE_SUBSYSTEM = "update_subsystem"
    UPDATE_HEALTH = "update_health"
    STOP = "stop"
    SNAPSHOT = "snapshot"
    CLEAR_EVENTS = "clear_events"


@dataclass(frozen=True, slots=True)
class LiveSessionStateRuntimeResult:
    status: LiveSessionStateRuntimeStatus
    operation: LiveSessionStateOperation
    state: LiveSessionState
    event: LiveSessionEvent | None
    reason: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == LiveSessionStateRuntimeStatus.READY


@dataclass(frozen=True, slots=True)
class LiveSessionStateRuntimeSnapshot:
    status: LiveSessionStateRuntimeStatus
    state: LiveSessionState
    event_count: int
    transition_count: int
    uptime_seconds: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


class LiveSessionStateRuntime:
    """
    Step 50B Live Session State Runtime.

    Owns the mutable live session state for the daily-driver runtime.

    This runtime:
    - tracks session lifecycle
    - tracks turn state
    - tracks speaking/listening/thinking/interruption/recovery
    - tracks subsystem status
    - emits typed live events
    - creates snapshots

    It does not:
    - start microphone
    - run STT/TTS
    - execute tools
    - create conversational text
    - bypass cognition/memory/planning/personality
    """

    def __init__(
        self,
        *,
        config: LiveSessionConfig | None = None,
    ) -> None:
        self._config = config or default_live_session_config()
        self._state = default_live_session_state(config=self._config)
        self._events: list[LiveSessionEvent] = []
        self._transition_count = 0

    @property
    def state(self) -> LiveSessionState:
        return self._state

    @property
    def events(self) -> tuple[LiveSessionEvent, ...]:
        return tuple(self._events)

    def start(self) -> LiveSessionStateRuntimeResult:
        if self._state.status not in {
            LiveSessionStatus.CREATED,
            LiveSessionStatus.STOPPED,
        }:
            return self._blocked(
                operation=LiveSessionStateOperation.START,
                reason="live session can only start from created or stopped state",
            )

        self._transition_count += 1
        self._state = _replace_state(
            self._state,
            status=LiveSessionStatus.STARTING,
            phase=LiveSessionPhase.BOOTING,
            interaction_state=LiveInteractionState.IDLE,
            health_status=LiveHealthStatus.HEALTHY,
            started_at=utc_now(),
            updated_at=utc_now(),
            user_present=True,
            conversation_active=False,
            assistant_speaking=False,
            shutdown_reason=None,
        )
        event = self._record_event(
            kind=LiveEventKind.SESSION_START_REQUESTED,
            priority=LiveEventPriority.NORMAL,
            source=LiveSubsystem.RUNTIME_KERNEL,
            title="Live session start requested",
            summary="Live session state entered starting phase.",
        )
        return self._ready(
            operation=LiveSessionStateOperation.START,
            event=event,
            reason="live session starting",
        )

    def mark_ready(self) -> LiveSessionStateRuntimeResult:
        if self._state.status != LiveSessionStatus.STARTING:
            return self._blocked(
                operation=LiveSessionStateOperation.MARK_READY,
                reason="live session can only become ready from starting state",
            )

        self._transition_count += 1
        self._state = _replace_state(
            self._state,
            status=LiveSessionStatus.RUNNING,
            phase=LiveSessionPhase.READY,
            interaction_state=LiveInteractionState.WAITING_FOR_USER,
            audio_state=_audio_state_for_ready(self._state),
            updated_at=utc_now(),
            conversation_active=True,
        )
        event = self._record_event(
            kind=LiveEventKind.SESSION_STARTED,
            priority=LiveEventPriority.NORMAL,
            source=LiveSubsystem.RUNTIME_KERNEL,
            title="Live session started",
            summary="Live session is running and ready.",
        )
        return self._ready(
            operation=LiveSessionStateOperation.MARK_READY,
            event=event,
            reason="live session ready",
        )

    def enter_listening(self) -> LiveSessionStateRuntimeResult:
        if not self._state.is_running:
            return self._blocked(
                operation=LiveSessionStateOperation.ENTER_LISTENING,
                reason="live session must be running before listening",
            )

        if not self._state.microphone_active or not self._state.stt_active:
            return self._blocked(
                operation=LiveSessionStateOperation.ENTER_LISTENING,
                reason="microphone and STT must be active before listening",
            )

        self._transition_count += 1
        self._state = _replace_state(
            self._state,
            phase=LiveSessionPhase.LISTENING,
            interaction_state=LiveInteractionState.LISTENING,
            audio_state=LiveAudioState.STREAMING_INPUT,
            updated_at=utc_now(),
        )
        event = self._record_event(
            kind=LiveEventKind.WAKE_DETECTED,
            priority=LiveEventPriority.NORMAL,
            source=LiveSubsystem.WAKE,
            title="Listening active",
            summary="Live session entered listening state.",
        )
        return self._ready(
            operation=LiveSessionStateOperation.ENTER_LISTENING,
            event=event,
            reason="live session listening",
        )

    def start_user_turn(self) -> LiveSessionStateRuntimeResult:
        if not self._state.is_running:
            return self._blocked(
                operation=LiveSessionStateOperation.START_USER_TURN,
                reason="live session must be running before user turn",
            )

        turn_id = make_live_turn_id()
        self._transition_count += 1
        self._state = _replace_state(
            self._state,
            phase=LiveSessionPhase.LISTENING,
            interaction_state=LiveInteractionState.USER_SPEAKING,
            current_turn_id=turn_id,
            updated_at=utc_now(),
            user_present=True,
        )
        event = self._record_event(
            kind=LiveEventKind.USER_SPEECH_STARTED,
            priority=LiveEventPriority.NORMAL,
            source=LiveSubsystem.PRESENCE,
            title="User speech started",
            summary="Live session started a user turn.",
            metadata={"turn_id": str(turn_id)},
        )
        return self._ready(
            operation=LiveSessionStateOperation.START_USER_TURN,
            event=event,
            reason="user turn started",
        )

    def transcript_ready(
        self,
        transcript: LiveTranscript,
    ) -> LiveSessionStateRuntimeResult:
        if self._state.current_turn_id != transcript.turn_id:
            return self._blocked(
                operation=LiveSessionStateOperation.TRANSCRIPT_READY,
                reason="transcript turn does not match current live turn",
                metadata={
                    "state_turn_id": str(self._state.current_turn_id or ""),
                    "transcript_turn_id": str(transcript.turn_id),
                },
            )

        self._transition_count += 1
        self._state = _replace_state(
            self._state,
            phase=LiveSessionPhase.LISTENING,
            interaction_state=LiveInteractionState.TRANSCRIBING,
            last_transcript=transcript,
            updated_at=utc_now(),
        )
        event = self._record_event(
            kind=LiveEventKind.TRANSCRIPT_READY,
            priority=LiveEventPriority.NORMAL,
            source=LiveSubsystem.STT,
            title="Transcript ready",
            summary="Live transcript is ready for conversation runtime.",
            metadata={
                "turn_id": str(transcript.turn_id),
                "transcript_id": str(transcript.transcript_id),
                "kind": transcript.kind.value,
                "confidence": transcript.confidence,
            },
        )
        return self._ready(
            operation=LiveSessionStateOperation.TRANSCRIPT_READY,
            event=event,
            reason="transcript accepted",
        )

    def start_thinking(self) -> LiveSessionStateRuntimeResult:
        if self._state.last_transcript is None:
            return self._blocked(
                operation=LiveSessionStateOperation.START_THINKING,
                reason="cannot think before transcript is ready",
            )

        self._transition_count += 1
        self._state = _replace_state(
            self._state,
            phase=LiveSessionPhase.THINKING,
            interaction_state=LiveInteractionState.THINKING,
            audio_state=_audio_state_for_ready(self._state),
            updated_at=utc_now(),
        )
        event = self._record_event(
            kind=LiveEventKind.TRANSCRIPT_READY,
            priority=LiveEventPriority.NORMAL,
            source=LiveSubsystem.CONVERSATION,
            title="Thinking started",
            summary="Live session moved transcript into cognition pipeline.",
            metadata={"turn_id": str(self._state.current_turn_id or "")},
        )
        return self._ready(
            operation=LiveSessionStateOperation.START_THINKING,
            event=event,
            reason="live session thinking",
        )

    def start_speaking(
        self,
        response: LiveResponse,
    ) -> LiveSessionStateRuntimeResult:
        if response.safety.value == "blocked":
            return self._blocked(
                operation=LiveSessionStateOperation.START_SPEAKING,
                reason="blocked live response cannot be spoken",
                metadata={"response_id": str(response.response_id)},
            )

        if not self._state.can_speak:
            return self._blocked(
                operation=LiveSessionStateOperation.START_SPEAKING,
                reason="TTS and playback must be active before speaking",
            )

        self._transition_count += 1
        self._state = _replace_state(
            self._state,
            phase=LiveSessionPhase.SPEAKING,
            interaction_state=LiveInteractionState.SPEAKING,
            audio_state=LiveAudioState.STREAMING_OUTPUT,
            assistant_speaking=True,
            last_response=response,
            updated_at=utc_now(),
        )
        event = self._record_event(
            kind=LiveEventKind.ASSISTANT_RESPONSE_STARTED,
            priority=LiveEventPriority.NORMAL,
            source=LiveSubsystem.TTS,
            title="Assistant response started",
            summary="Live response entered speech output.",
            metadata={
                "turn_id": str(response.turn_id),
                "response_id": str(response.response_id),
                "kind": response.kind.value,
                "generation_source": response.generation_source.value,
            },
        )
        return self._ready(
            operation=LiveSessionStateOperation.START_SPEAKING,
            event=event,
            reason="assistant speaking",
        )

    def finish_speaking(self) -> LiveSessionStateRuntimeResult:
        if not self._state.assistant_speaking:
            return self._blocked(
                operation=LiveSessionStateOperation.FINISH_SPEAKING,
                reason="assistant is not currently speaking",
            )

        self._transition_count += 1
        self._state = _replace_state(
            self._state,
            phase=LiveSessionPhase.READY,
            interaction_state=LiveInteractionState.WAITING_FOR_USER,
            audio_state=_audio_state_for_ready(self._state),
            assistant_speaking=False,
            updated_at=utc_now(),
        )
        event = self._record_event(
            kind=LiveEventKind.ASSISTANT_RESPONSE_FINISHED,
            priority=LiveEventPriority.NORMAL,
            source=LiveSubsystem.PLAYBACK,
            title="Assistant response finished",
            summary="Live response playback finished.",
        )
        return self._ready(
            operation=LiveSessionStateOperation.FINISH_SPEAKING,
            event=event,
            reason="assistant finished speaking",
        )

    def interrupt(
        self,
        *,
        reason: str,
    ) -> LiveSessionStateRuntimeResult:
        if not reason.strip():
            raise ValueError("live interruption reason cannot be empty.")

        if not self._state.can_interrupt:
            return self._blocked(
                operation=LiveSessionStateOperation.INTERRUPT,
                reason="interruption is not available in current state",
            )

        self._transition_count += 1
        self._state = _replace_state(
            self._state,
            phase=LiveSessionPhase.INTERRUPTED,
            interaction_state=LiveInteractionState.INTERRUPTED,
            assistant_speaking=False,
            audio_state=_audio_state_for_ready(self._state),
            updated_at=utc_now(),
            metadata={
                **self._state.metadata,
                "last_interruption_reason": reason.strip(),
            },
        )
        event = self._record_event(
            kind=LiveEventKind.INTERRUPTION_REQUESTED,
            priority=LiveEventPriority.CRITICAL,
            source=LiveSubsystem.INTERRUPTION,
            title="Interruption requested",
            summary="Live session accepted an interruption request.",
            metadata={"reason": reason.strip()},
        )
        return self._ready(
            operation=LiveSessionStateOperation.INTERRUPT,
            event=event,
            reason="live session interrupted",
        )

    def enter_recovery(
        self,
        *,
        subsystem: LiveSubsystem,
        reason: str,
    ) -> LiveSessionStateRuntimeResult:
        if not reason.strip():
            raise ValueError("live recovery reason cannot be empty.")

        self._transition_count += 1
        subsystem_states = _replace_subsystem(
            self._state.subsystem_states,
            LiveSubsystemState(
                subsystem=subsystem,
                status=LiveSubsystemStatus.DEGRADED,
                message=reason.strip(),
                updated_at=utc_now(),
            ),
        )
        self._state = _replace_state(
            self._state,
            status=LiveSessionStatus.DEGRADED,
            phase=LiveSessionPhase.RECOVERING,
            interaction_state=LiveInteractionState.INTERRUPTED,
            health_status=LiveHealthStatus.DEGRADED,
            assistant_speaking=False,
            subsystem_states=subsystem_states,
            updated_at=utc_now(),
        )
        event = self._record_event(
            kind=LiveEventKind.RECOVERY_STARTED,
            priority=LiveEventPriority.HIGH,
            source=LiveSubsystem.RECOVERY,
            title="Recovery started",
            summary="Live session entered recovery.",
            metadata={
                "subsystem": subsystem.value,
                "reason": reason.strip(),
            },
        )
        return self._ready(
            operation=LiveSessionStateOperation.ENTER_RECOVERY,
            event=event,
            reason="live session recovering",
        )

    def finish_recovery(self) -> LiveSessionStateRuntimeResult:
        if self._state.phase != LiveSessionPhase.RECOVERING:
            return self._blocked(
                operation=LiveSessionStateOperation.FINISH_RECOVERY,
                reason="live session is not recovering",
            )

        self._transition_count += 1
        self._state = _replace_state(
            self._state,
            status=LiveSessionStatus.RUNNING,
            phase=LiveSessionPhase.READY,
            interaction_state=LiveInteractionState.WAITING_FOR_USER,
            health_status=LiveHealthStatus.HEALTHY,
            audio_state=_audio_state_for_ready(self._state),
            updated_at=utc_now(),
        )
        event = self._record_event(
            kind=LiveEventKind.RECOVERY_FINISHED,
            priority=LiveEventPriority.NORMAL,
            source=LiveSubsystem.RECOVERY,
            title="Recovery finished",
            summary="Live session recovered and returned to ready state.",
        )
        return self._ready(
            operation=LiveSessionStateOperation.FINISH_RECOVERY,
            event=event,
            reason="live session recovered",
        )

    def update_subsystem(
        self,
        subsystem_state: LiveSubsystemState,
    ) -> LiveSessionStateRuntimeResult:
        self._transition_count += 1
        states = _replace_subsystem(
            self._state.subsystem_states,
            subsystem_state,
        )
        status = _status_from_subsystems(self._state.status, states)
        health = _health_from_subsystems(states)
        self._state = _replace_state(
            self._state,
            status=status,
            health_status=health,
            subsystem_states=states,
            updated_at=utc_now(),
        )
        event = self._record_event(
            kind=LiveEventKind.HEALTH_CHANGED,
            priority=_priority_for_health(health),
            source=subsystem_state.subsystem,
            title="Subsystem state updated",
            summary="Live subsystem state changed.",
            metadata={
                "subsystem": subsystem_state.subsystem.value,
                "status": subsystem_state.status.value,
                "message": subsystem_state.message,
            },
        )
        return self._ready(
            operation=LiveSessionStateOperation.UPDATE_SUBSYSTEM,
            event=event,
            reason="subsystem state updated",
        )

    def update_health(
        self,
        health_status: LiveHealthStatus,
    ) -> LiveSessionStateRuntimeResult:
        self._transition_count += 1
        status = (
            LiveSessionStatus.FAILED
            if health_status == LiveHealthStatus.FAILED
            else self._state.status
        )
        self._state = _replace_state(
            self._state,
            status=status,
            health_status=health_status,
            updated_at=utc_now(),
        )
        event = self._record_event(
            kind=LiveEventKind.HEALTH_CHANGED,
            priority=_priority_for_health(health_status),
            source=LiveSubsystem.HEALTH_MONITOR,
            title="Live health changed",
            summary="Live session health status changed.",
            metadata={"health": health_status.value},
        )
        return self._ready(
            operation=LiveSessionStateOperation.UPDATE_HEALTH,
            event=event,
            reason="health updated",
        )

    def stop(
        self,
        *,
        reason: LiveShutdownReason,
    ) -> LiveSessionStateRuntimeResult:
        if self._state.status == LiveSessionStatus.STOPPED:
            return self._blocked(
                operation=LiveSessionStateOperation.STOP,
                reason="live session is already stopped",
            )

        self._transition_count += 1
        self._state = _replace_state(
            self._state,
            status=LiveSessionStatus.STOPPED,
            phase=LiveSessionPhase.SHUTTING_DOWN,
            interaction_state=LiveInteractionState.SHUTTING_DOWN,
            audio_state=LiveAudioState.INACTIVE,
            assistant_speaking=False,
            conversation_active=False,
            microphone_active=False,
            stt_active=False,
            tts_active=False,
            playback_active=False,
            updated_at=utc_now(),
            shutdown_reason=reason,
        )
        event = self._record_event(
            kind=LiveEventKind.SESSION_STOPPED,
            priority=LiveEventPriority.NORMAL,
            source=LiveSubsystem.RUNTIME_KERNEL,
            title="Live session stopped",
            summary="Live session state stopped safely.",
            metadata={"shutdown_reason": reason.value},
        )
        return self._ready(
            operation=LiveSessionStateOperation.STOP,
            event=event,
            reason="live session stopped",
        )

    def snapshot(self) -> LiveSessionStateRuntimeSnapshot:
        uptime = _uptime_seconds(self._state)
        return LiveSessionStateRuntimeSnapshot(
            status=LiveSessionStateRuntimeStatus.READY,
            state=self._state,
            event_count=len(self._events),
            transition_count=self._transition_count,
            uptime_seconds=uptime,
            created_at=utc_now(),
        )

    def clear_events(self) -> LiveSessionStateRuntimeResult:
        self._events.clear()
        return self._ready(
            operation=LiveSessionStateOperation.CLEAR_EVENTS,
            event=None,
            reason="live session events cleared",
        )

    def _record_event(
        self,
        *,
        kind: LiveEventKind,
        priority: LiveEventPriority,
        source: LiveSubsystem,
        title: str,
        summary: str,
        metadata: dict[str, object] | None = None,
    ) -> LiveSessionEvent:
        event = make_live_event(
            kind=kind,
            priority=priority,
            source=source,
            title=title,
            summary=summary,
            metadata=metadata,
        )
        self._events.append(event)
        return event

    def _ready(
        self,
        *,
        operation: LiveSessionStateOperation,
        event: LiveSessionEvent | None,
        reason: str,
        metadata: dict[str, object] | None = None,
    ) -> LiveSessionStateRuntimeResult:
        return LiveSessionStateRuntimeResult(
            status=LiveSessionStateRuntimeStatus.READY,
            operation=operation,
            state=self._state,
            event=event,
            reason=reason,
            created_at=utc_now(),
            metadata=metadata or {},
        )

    def _blocked(
        self,
        *,
        operation: LiveSessionStateOperation,
        reason: str,
        metadata: dict[str, object] | None = None,
    ) -> LiveSessionStateRuntimeResult:
        return LiveSessionStateRuntimeResult(
            status=LiveSessionStateRuntimeStatus.BLOCKED,
            operation=operation,
            state=self._state,
            event=None,
            reason=reason,
            created_at=utc_now(),
            metadata=metadata or {},
        )


def _replace_state(
    state: LiveSessionState,
    **changes: Any,
) -> LiveSessionState:
    return replace(state, **changes)


def _replace_subsystem(
    states: tuple[LiveSubsystemState, ...],
    updated: LiveSubsystemState,
) -> tuple[LiveSubsystemState, ...]:
    found = False
    result: list[LiveSubsystemState] = []

    for state in states:
        if state.subsystem == updated.subsystem:
            result.append(updated)
            found = True
        else:
            result.append(state)

    if not found:
        result.append(updated)

    return tuple(result)


def _audio_state_for_ready(state: LiveSessionState) -> LiveAudioState:
    if state.playback_active:
        return LiveAudioState.PLAYBACK_READY
    if state.tts_active:
        return LiveAudioState.TTS_READY
    if state.stt_active:
        return LiveAudioState.STT_READY
    if state.microphone_active:
        return LiveAudioState.MICROPHONE_READY
    return LiveAudioState.INACTIVE


def _status_from_subsystems(
    current: LiveSessionStatus,
    states: tuple[LiveSubsystemState, ...],
) -> LiveSessionStatus:
    if any(state.status == LiveSubsystemStatus.FAILED for state in states):
        return LiveSessionStatus.FAILED

    if any(state.status == LiveSubsystemStatus.DEGRADED for state in states):
        if current == LiveSessionStatus.STOPPED:
            return LiveSessionStatus.STOPPED
        return LiveSessionStatus.DEGRADED

    if current in {
        LiveSessionStatus.CREATED,
        LiveSessionStatus.STARTING,
        LiveSessionStatus.RUNNING,
        LiveSessionStatus.PAUSED,
        LiveSessionStatus.STOPPING,
        LiveSessionStatus.STOPPED,
    }:
        return current

    return LiveSessionStatus.RUNNING


def _health_from_subsystems(
    states: tuple[LiveSubsystemState, ...],
) -> LiveHealthStatus:
    if any(state.status == LiveSubsystemStatus.FAILED for state in states):
        return LiveHealthStatus.FAILED

    if any(state.status == LiveSubsystemStatus.DEGRADED for state in states):
        return LiveHealthStatus.DEGRADED

    return LiveHealthStatus.HEALTHY


def _priority_for_health(health: LiveHealthStatus) -> LiveEventPriority:
    if health in {LiveHealthStatus.FAILED, LiveHealthStatus.CRITICAL}:
        return LiveEventPriority.CRITICAL
    if health == LiveHealthStatus.DEGRADED:
        return LiveEventPriority.HIGH
    return LiveEventPriority.NORMAL


def _uptime_seconds(state: LiveSessionState) -> float:
    if state.started_at is None:
        return 0.0

    return max(0.0, (utc_now() - state.started_at).total_seconds())