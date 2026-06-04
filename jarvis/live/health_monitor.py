from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from jarvis.live.audio_runtime import (
    LiveAudioRuntime,
    LiveAudioRuntimeSnapshot,
    LiveAudioRuntimeStatus,
)
from jarvis.live.contracts import (
    LiveEventKind,
    LiveEventPriority,
    LiveHealthStatus,
    LiveSessionConfig,
    LiveSubsystem,
    LiveSubsystemState,
    LiveSubsystemStatus,
    default_live_session_config,
    make_live_event,
    utc_now,
)
from jarvis.live.dialogue_runtime import (
    LiveDialogueRuntime,
    LiveDialogueSnapshot,
)
from jarvis.live.event_bridge import (
    LiveEventBridgeRequest,
    LiveEventBridgeResult,
    LiveEventBridgeRuntime,
    LiveEventBridgeSnapshot,
)
from jarvis.live.interruption_runtime import (
    LiveInterruptionRuntime,
    LiveInterruptionSnapshot,
)
from jarvis.live.response_boundary import (
    LiveResponseBoundaryRuntime,
    LiveResponseBoundarySnapshot,
)
from jarvis.live.session_state import (
    LiveSessionStateRuntime,
    LiveSessionStateRuntimeResult,
)


class LiveHealthMonitorStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"


class LiveHealthMonitorOperation(StrEnum):
    CHECK = "check"
    RECORD_SIGNAL = "record_signal"
    SNAPSHOT = "snapshot"


class LiveHealthSignalKind(StrEnum):
    SESSION_STATE = "session_state"
    AUDIO = "audio"
    EVENT_BRIDGE = "event_bridge"
    RESPONSE_BOUNDARY = "response_boundary"
    DIALOGUE = "dialogue"
    INTERRUPTION = "interruption"
    SUBSYSTEM = "subsystem"
    BLOCKED_PRESSURE = "blocked_pressure"


@dataclass(frozen=True, slots=True)
class LiveHealthMonitorPolicy:
    max_blocked_before_degraded: int = 3
    max_blocked_before_critical: int = 8
    fail_on_failed_subsystem: bool = True
    critical_on_audio_unprepared_in_voice_mode: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_blocked_before_degraded < 1:
            raise ValueError("max_blocked_before_degraded must be positive.")
        if self.max_blocked_before_critical < self.max_blocked_before_degraded:
            raise ValueError(
                "max_blocked_before_critical must be >= degraded threshold."
            )


@dataclass(frozen=True, slots=True)
class LiveHealthSignal:
    kind: LiveHealthSignalKind
    subsystem: LiveSubsystem
    status: LiveHealthMonitorStatus
    message: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise ValueError("live health signal message cannot be empty.")


@dataclass(frozen=True, slots=True)
class LiveHealthMonitorResult:
    status: LiveHealthMonitorStatus
    operation: LiveHealthMonitorOperation
    signals: tuple[LiveHealthSignal, ...]
    bridge_result: LiveEventBridgeResult | None
    state_result: LiveSessionStateRuntimeResult | None
    reason: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def healthy(self) -> bool:
        return self.status == LiveHealthMonitorStatus.HEALTHY

    @property
    def needs_recovery(self) -> bool:
        return self.status in {
            LiveHealthMonitorStatus.DEGRADED,
            LiveHealthMonitorStatus.CRITICAL,
            LiveHealthMonitorStatus.FAILED,
        }


@dataclass(frozen=True, slots=True)
class LiveHealthMonitorSnapshot:
    status: LiveHealthMonitorStatus
    check_count: int
    signal_count: int
    degraded_count: int
    critical_count: int
    failed_count: int
    last_signals: tuple[LiveHealthSignal, ...]
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


class LiveHealthMonitorRuntime:
    """
    Step 50H Live Health Monitor.

    Watches the daily-driver live runtime for degradation.

    It does not:
    - generate conversational responses
    - call TTS
    - execute tools
    - access memory directly
    - recover by itself

    It only emits typed health signals and updates live session health.
    Step 50I Recovery Runtime consumes these signals later.
    """

    def __init__(
        self,
        *,
        live_state: LiveSessionStateRuntime | None = None,
        bridge: LiveEventBridgeRuntime | None = None,
        audio: LiveAudioRuntime | None = None,
        response_boundary: LiveResponseBoundaryRuntime | None = None,
        dialogue: LiveDialogueRuntime | None = None,
        interruption: LiveInterruptionRuntime | None = None,
        config: LiveSessionConfig | None = None,
        policy: LiveHealthMonitorPolicy | None = None,
    ) -> None:
        self._config = config or default_live_session_config()
        self._state = live_state or LiveSessionStateRuntime(config=self._config)
        self._bridge = bridge or LiveEventBridgeRuntime(live_state=self._state)
        self._audio = audio
        self._response_boundary = response_boundary
        self._dialogue = dialogue
        self._interruption = interruption
        self._policy = policy or LiveHealthMonitorPolicy()
        self._signals: list[LiveHealthSignal] = []
        self._check_count = 0

    def check(self) -> LiveHealthMonitorResult:
        self._check_count += 1
        signals: list[LiveHealthSignal] = []

        signals.extend(self._check_session_state())
        signals.extend(self._check_audio())
        signals.extend(self._check_bridge())
        signals.extend(self._check_response_boundary())
        signals.extend(self._check_dialogue())
        signals.extend(self._check_interruption())
        signals.extend(self._check_blocked_pressure())

        self._signals.extend(signals)
        status = _worst_status(signals)
        state_result = self._state.update_health(_to_live_health(status))

        bridge_result: LiveEventBridgeResult | None = None
        if status != LiveHealthMonitorStatus.HEALTHY:
            bridge_result = self._emit_health_event(status=status, signals=signals)

        return LiveHealthMonitorResult(
            status=status,
            operation=LiveHealthMonitorOperation.CHECK,
            signals=tuple(signals),
            bridge_result=bridge_result,
            state_result=state_result,
            reason=_reason_for_status(status),
            created_at=utc_now(),
            metadata={
                "check_count": self._check_count,
                "signal_count": len(signals),
            },
        )

    def record_signal(
        self,
        signal: LiveHealthSignal,
    ) -> LiveHealthMonitorResult:
        self._signals.append(signal)
        status = signal.status
        state_result = self._state.update_subsystem(
            LiveSubsystemState(
                subsystem=signal.subsystem,
                status=_subsystem_status_from_health(signal.status),
                message=signal.message,
                updated_at=utc_now(),
                metadata=signal.metadata,
            )
        )
        bridge_result: LiveEventBridgeResult | None = None
        if status != LiveHealthMonitorStatus.HEALTHY:
            bridge_result = self._emit_health_event(
                status=status,
                signals=(signal,),
            )

        return LiveHealthMonitorResult(
            status=status,
            operation=LiveHealthMonitorOperation.RECORD_SIGNAL,
            signals=(signal,),
            bridge_result=bridge_result,
            state_result=state_result,
            reason=signal.message,
            created_at=utc_now(),
            metadata=signal.metadata,
        )

    def snapshot(self) -> LiveHealthMonitorSnapshot:
        degraded = sum(
            1 for signal in self._signals
            if signal.status == LiveHealthMonitorStatus.DEGRADED
        )
        critical = sum(
            1 for signal in self._signals
            if signal.status == LiveHealthMonitorStatus.CRITICAL
        )
        failed = sum(
            1 for signal in self._signals
            if signal.status == LiveHealthMonitorStatus.FAILED
        )
        return LiveHealthMonitorSnapshot(
            status=_worst_status(self._signals),
            check_count=self._check_count,
            signal_count=len(self._signals),
            degraded_count=degraded,
            critical_count=critical,
            failed_count=failed,
            last_signals=tuple(self._signals[-10:]),
            created_at=utc_now(),
        )

    def _check_session_state(self) -> tuple[LiveHealthSignal, ...]:
        state = self._state.state
        signals: list[LiveHealthSignal] = []

        if state.failed_subsystems and self._policy.fail_on_failed_subsystem:
            for subsystem in state.failed_subsystems:
                signals.append(
                    _signal(
                        kind=LiveHealthSignalKind.SUBSYSTEM,
                        subsystem=subsystem.subsystem,
                        status=LiveHealthMonitorStatus.FAILED,
                        message=subsystem.message,
                        metadata=subsystem.metadata,
                    )
                )

        if state.health_status == LiveHealthStatus.FAILED:
            signals.append(
                _signal(
                    kind=LiveHealthSignalKind.SESSION_STATE,
                    subsystem=LiveSubsystem.HEALTH_MONITOR,
                    status=LiveHealthMonitorStatus.FAILED,
                    message="live session health is failed",
                )
            )

        if state.health_status == LiveHealthStatus.DEGRADED:
            signals.append(
                _signal(
                    kind=LiveHealthSignalKind.SESSION_STATE,
                    subsystem=LiveSubsystem.HEALTH_MONITOR,
                    status=LiveHealthMonitorStatus.DEGRADED,
                    message="live session health is degraded",
                )
            )

        if not signals:
            signals.append(
                _signal(
                    kind=LiveHealthSignalKind.SESSION_STATE,
                    subsystem=LiveSubsystem.HEALTH_MONITOR,
                    status=LiveHealthMonitorStatus.HEALTHY,
                    message="live session state is healthy",
                )
            )

        return tuple(signals)

    def _check_audio(self) -> tuple[LiveHealthSignal, ...]:
        if self._audio is None:
            return (
                _signal(
                    kind=LiveHealthSignalKind.AUDIO,
                    subsystem=LiveSubsystem.MICROPHONE,
                    status=LiveHealthMonitorStatus.DEGRADED,
                    message="live audio runtime is not connected",
                ),
            )

        snapshot = self._audio.snapshot()
        status = _status_from_audio_snapshot(
            snapshot=snapshot,
            real_voice_required=self._state.state.microphone_active,
            policy=self._policy,
        )
        return (
            _signal(
                kind=LiveHealthSignalKind.AUDIO,
                subsystem=LiveSubsystem.MICROPHONE,
                status=status,
                message=_audio_message(snapshot, status),
                metadata={
                    "prepared": snapshot.prepared,
                    "captured_frames": snapshot.captured_frames,
                    "transcripts": snapshot.transcripts,
                    "played_responses": snapshot.played_responses,
                    "blocked_count": snapshot.blocked_count,
                },
            ),
        )

    def _check_bridge(self) -> tuple[LiveHealthSignal, ...]:
        snapshot = self._bridge.snapshot()
        return (
            _signal_from_bridge_snapshot(snapshot),
        )

    def _check_response_boundary(self) -> tuple[LiveHealthSignal, ...]:
        if self._response_boundary is None:
            return (
                _signal(
                    kind=LiveHealthSignalKind.RESPONSE_BOUNDARY,
                    subsystem=LiveSubsystem.RESPONSE_GENERATOR,
                    status=LiveHealthMonitorStatus.DEGRADED,
                    message="live response boundary is not connected",
                ),
            )

        snapshot = self._response_boundary.snapshot()
        return (
            _signal_from_response_boundary_snapshot(snapshot),
        )

    def _check_dialogue(self) -> tuple[LiveHealthSignal, ...]:
        if self._dialogue is None:
            return (
                _signal(
                    kind=LiveHealthSignalKind.DIALOGUE,
                    subsystem=LiveSubsystem.CONVERSATION,
                    status=LiveHealthMonitorStatus.DEGRADED,
                    message="live dialogue runtime is not connected",
                ),
            )

        snapshot = self._dialogue.snapshot()
        return (_signal_from_dialogue_snapshot(snapshot),)

    def _check_interruption(self) -> tuple[LiveHealthSignal, ...]:
        if self._interruption is None:
            return (
                _signal(
                    kind=LiveHealthSignalKind.INTERRUPTION,
                    subsystem=LiveSubsystem.INTERRUPTION,
                    status=LiveHealthMonitorStatus.DEGRADED,
                    message="live interruption runtime is not connected",
                ),
            )

        snapshot = self._interruption.snapshot()
        return (_signal_from_interruption_snapshot(snapshot),)

    def _check_blocked_pressure(self) -> tuple[LiveHealthSignal, ...]:
        blocked = 0

        blocked += self._bridge.snapshot().blocked_count

        if self._audio is not None:
            blocked += self._audio.snapshot().blocked_count

        if self._response_boundary is not None:
            blocked += self._response_boundary.snapshot().blocked_count

        if self._dialogue is not None:
            blocked += self._dialogue.snapshot().blocked_turns

        if self._interruption is not None:
            blocked += self._interruption.snapshot().blocked_count

        if blocked >= self._policy.max_blocked_before_critical:
            status = LiveHealthMonitorStatus.CRITICAL
        elif blocked >= self._policy.max_blocked_before_degraded:
            status = LiveHealthMonitorStatus.DEGRADED
        else:
            status = LiveHealthMonitorStatus.HEALTHY

        return (
            _signal(
                kind=LiveHealthSignalKind.BLOCKED_PRESSURE,
                subsystem=LiveSubsystem.HEALTH_MONITOR,
                status=status,
                message=f"live blocked pressure is {blocked}",
                metadata={"blocked_count": blocked},
            ),
        )

    def _emit_health_event(
        self,
        *,
        status: LiveHealthMonitorStatus,
        signals: tuple[LiveHealthSignal, ...] | list[LiveHealthSignal],
    ) -> LiveEventBridgeResult:
        event = make_live_event(
            kind=LiveEventKind.HEALTH_CHANGED,
            priority=_priority_from_health(status),
            source=LiveSubsystem.HEALTH_MONITOR,
            title="Live health changed",
            summary=_summary_from_signals(signals),
            metadata={
                "health": status.value,
                "signals": tuple(signal.message for signal in signals),
            },
        )
        return self._bridge.bridge_event(
            LiveEventBridgeRequest(
                event=event,
                update_live_state=True,
                update_cognitive_state=True,
                allow_interruptions=True,
            )
        )


def _signal(
    *,
    kind: LiveHealthSignalKind,
    subsystem: LiveSubsystem,
    status: LiveHealthMonitorStatus,
    message: str,
    metadata: dict[str, object] | None = None,
) -> LiveHealthSignal:
    return LiveHealthSignal(
        kind=kind,
        subsystem=subsystem,
        status=status,
        message=message,
        created_at=utc_now(),
        metadata=metadata or {},
    )


def _status_from_audio_snapshot(
    *,
    snapshot: LiveAudioRuntimeSnapshot,
    real_voice_required: bool,
    policy: LiveHealthMonitorPolicy,
) -> LiveHealthMonitorStatus:
    if snapshot.status == LiveAudioRuntimeStatus.BLOCKED:
        return LiveHealthMonitorStatus.CRITICAL

    if snapshot.status == LiveAudioRuntimeStatus.DEGRADED:
        return LiveHealthMonitorStatus.DEGRADED

    if (
        policy.critical_on_audio_unprepared_in_voice_mode
        and real_voice_required
        and not snapshot.prepared
    ):
        return LiveHealthMonitorStatus.CRITICAL

    if snapshot.blocked_count >= policy.max_blocked_before_critical:
        return LiveHealthMonitorStatus.CRITICAL

    if snapshot.blocked_count >= policy.max_blocked_before_degraded:
        return LiveHealthMonitorStatus.DEGRADED

    return LiveHealthMonitorStatus.HEALTHY


def _audio_message(
    snapshot: LiveAudioRuntimeSnapshot,
    status: LiveHealthMonitorStatus,
) -> str:
    if status == LiveHealthMonitorStatus.HEALTHY:
        return "live audio runtime is healthy"
    return (
        "live audio runtime requires attention: "
        f"prepared={snapshot.prepared}, blocked={snapshot.blocked_count}"
    )


def _signal_from_bridge_snapshot(
    snapshot: LiveEventBridgeSnapshot,
) -> LiveHealthSignal:
    status = _status_from_blocked_count(snapshot.blocked_count)
    return _signal(
        kind=LiveHealthSignalKind.EVENT_BRIDGE,
        subsystem=LiveSubsystem.EVENT_BUS,
        status=status,
        message=f"live event bridge blocked count is {snapshot.blocked_count}",
        metadata={
            "bridged_event_count": snapshot.bridged_event_count,
            "bridged_transcript_count": snapshot.bridged_transcript_count,
            "bridged_response_count": snapshot.bridged_response_count,
            "blocked_count": snapshot.blocked_count,
        },
    )


def _signal_from_response_boundary_snapshot(
    snapshot: LiveResponseBoundarySnapshot,
) -> LiveHealthSignal:
    status = _status_from_blocked_count(snapshot.blocked_count)
    return _signal(
        kind=LiveHealthSignalKind.RESPONSE_BOUNDARY,
        subsystem=LiveSubsystem.RESPONSE_GENERATOR,
        status=status,
        message=(
            "live response boundary blocked count is "
            f"{snapshot.blocked_count}"
        ),
        metadata={
            "generated_count": snapshot.generated_count,
            "deterministic_system_count": snapshot.deterministic_system_count,
            "blocked_count": snapshot.blocked_count,
        },
    )


def _signal_from_dialogue_snapshot(
    snapshot: LiveDialogueSnapshot,
) -> LiveHealthSignal:
    status = _status_from_blocked_count(snapshot.blocked_turns)
    return _signal(
        kind=LiveHealthSignalKind.DIALOGUE,
        subsystem=LiveSubsystem.CONVERSATION,
        status=status,
        message=f"live dialogue blocked turns are {snapshot.blocked_turns}",
        metadata={
            "completed_turns": snapshot.completed_turns,
            "blocked_turns": snapshot.blocked_turns,
            "generated_responses": snapshot.generated_responses,
        },
    )


def _signal_from_interruption_snapshot(
    snapshot: LiveInterruptionSnapshot,
) -> LiveHealthSignal:
    status = _status_from_blocked_count(snapshot.blocked_count)
    return _signal(
        kind=LiveHealthSignalKind.INTERRUPTION,
        subsystem=LiveSubsystem.INTERRUPTION,
        status=status,
        message=(
            "live interruption blocked count is "
            f"{snapshot.blocked_count}"
        ),
        metadata={
            "interruption_count": snapshot.interruption_count,
            "resume_count": snapshot.resume_count,
            "cancelled_count": snapshot.cancelled_count,
            "blocked_count": snapshot.blocked_count,
        },
    )


def _status_from_blocked_count(blocked_count: int) -> LiveHealthMonitorStatus:
    if blocked_count >= 8:
        return LiveHealthMonitorStatus.CRITICAL
    if blocked_count >= 3:
        return LiveHealthMonitorStatus.DEGRADED
    return LiveHealthMonitorStatus.HEALTHY


def _worst_status(
    signals: tuple[LiveHealthSignal, ...] | list[LiveHealthSignal],
) -> LiveHealthMonitorStatus:
    if any(signal.status == LiveHealthMonitorStatus.FAILED for signal in signals):
        return LiveHealthMonitorStatus.FAILED
    if any(signal.status == LiveHealthMonitorStatus.CRITICAL for signal in signals):
        return LiveHealthMonitorStatus.CRITICAL
    if any(signal.status == LiveHealthMonitorStatus.DEGRADED for signal in signals):
        return LiveHealthMonitorStatus.DEGRADED
    return LiveHealthMonitorStatus.HEALTHY


def _to_live_health(status: LiveHealthMonitorStatus) -> LiveHealthStatus:
    if status == LiveHealthMonitorStatus.FAILED:
        return LiveHealthStatus.FAILED
    if status == LiveHealthMonitorStatus.CRITICAL:
        return LiveHealthStatus.CRITICAL
    if status == LiveHealthMonitorStatus.DEGRADED:
        return LiveHealthStatus.DEGRADED
    return LiveHealthStatus.HEALTHY


def _subsystem_status_from_health(
    status: LiveHealthMonitorStatus,
) -> LiveSubsystemStatus:
    if status == LiveHealthMonitorStatus.FAILED:
        return LiveSubsystemStatus.FAILED
    if status in {
        LiveHealthMonitorStatus.CRITICAL,
        LiveHealthMonitorStatus.DEGRADED,
    }:
        return LiveSubsystemStatus.DEGRADED
    return LiveSubsystemStatus.READY


def _priority_from_health(status: LiveHealthMonitorStatus) -> LiveEventPriority:
    if status in {
        LiveHealthMonitorStatus.FAILED,
        LiveHealthMonitorStatus.CRITICAL,
    }:
        return LiveEventPriority.CRITICAL
    if status == LiveHealthMonitorStatus.DEGRADED:
        return LiveEventPriority.HIGH
    return LiveEventPriority.NORMAL


def _reason_for_status(status: LiveHealthMonitorStatus) -> str:
    if status == LiveHealthMonitorStatus.HEALTHY:
        return "live health is healthy"
    if status == LiveHealthMonitorStatus.DEGRADED:
        return "live health is degraded"
    if status == LiveHealthMonitorStatus.CRITICAL:
        return "live health is critical"
    return "live health failed"


def _summary_from_signals(
    signals: tuple[LiveHealthSignal, ...] | list[LiveHealthSignal],
) -> str:
    if not signals:
        return "live health changed"

    return "; ".join(signal.message for signal in signals[:5])