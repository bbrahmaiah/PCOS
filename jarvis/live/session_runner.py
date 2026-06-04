from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from jarvis.live.audio_runtime import LiveAudioRuntime, LiveAudioRuntimeResult
from jarvis.live.contracts import (
    LiveResponseSafety,
    LiveSessionConfig,
    LiveShutdownReason,
    LiveTranscriptKind,
    default_live_session_config,
    make_live_transcript,
    utc_now,
)
from jarvis.live.dialogue_runtime import (
    LiveDialogueRequest,
    LiveDialogueResult,
    LiveDialogueRuntime,
    LiveDialogueRuntimeStatus,
)
from jarvis.live.event_bridge import LiveEventBridgeRuntime
from jarvis.live.health_monitor import (
    LiveHealthMonitorResult,
    LiveHealthMonitorRuntime,
    LiveHealthMonitorStatus,
)
from jarvis.live.interruption_runtime import (
    LiveInterruptionRequest,
    LiveInterruptionResult,
    LiveInterruptionRuntime,
)
from jarvis.live.recovery_runtime import (
    LiveRecoveryResult,
    LiveRecoveryRuntime,
    LiveRecoveryRuntimeStatus,
)
from jarvis.live.response_boundary import (
    LiveResponseBoundaryRuntime,
    LiveResponseGenerator,
)
from jarvis.live.session_state import (
    LiveSessionStateRuntime,
    LiveSessionStateRuntimeResult,
    LiveSessionStateRuntimeStatus,
)
from jarvis.live.wake_engagement import (
    LiveWakeEngagementRequest,
    LiveWakeEngagementResult,
    LiveWakeEngagementRuntime,
)


class LiveSessionRunnerStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPED = "stopped"
    FAILED = "failed"


class LiveSessionRunnerOperation(StrEnum):
    START = "start"
    INGEST_TEXT = "ingest_text"
    HANDLE_INTERRUPT = "handle_interrupt"
    CHECK_HEALTH = "check_health"
    RECOVER = "recover"
    SHUTDOWN = "shutdown"
    SNAPSHOT = "snapshot"


@dataclass(frozen=True, slots=True)
class LiveSessionRunnerConfig:
    session_config: LiveSessionConfig = field(
        default_factory=default_live_session_config
    )
    auto_prepare_audio: bool = True
    auto_health_check: bool = True
    auto_recover: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LiveSessionRunnerResult:
    status: LiveSessionRunnerStatus
    operation: LiveSessionRunnerOperation
    state_result: LiveSessionStateRuntimeResult | None
    wake_result: LiveWakeEngagementResult | None
    dialogue_result: LiveDialogueResult | None
    interruption_result: LiveInterruptionResult | None
    audio_result: LiveAudioRuntimeResult | None
    health_result: LiveHealthMonitorResult | None
    recovery_result: LiveRecoveryResult | None
    reason: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status in {
            LiveSessionRunnerStatus.RUNNING,
            LiveSessionRunnerStatus.DEGRADED,
            LiveSessionRunnerStatus.STOPPED,
        }


@dataclass(frozen=True, slots=True)
class LiveSessionRunnerSnapshot:
    status: LiveSessionRunnerStatus
    state_status: str
    state_phase: str
    event_count: int
    health_status: str
    recovery_attempts: int
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


class LiveSessionRunner:
    """
    Step 50J Live Session Runner.

    This is the unified live runtime coordinator.

    It connects:
    - LiveSessionStateRuntime
    - LiveResponseBoundaryRuntime
    - LiveEventBridgeRuntime
    - LiveAudioRuntime
    - LiveWakeEngagementRuntime
    - LiveDialogueRuntime
    - LiveInterruptionRuntime
    - LiveHealthMonitorRuntime
    - LiveRecoveryRuntime

    It does not:
    - hardcode conversational responses
    - call TTS directly
    - execute tools directly
    - access memory directly
    - bypass completed subsystems

    All final dialogue must pass through LiveResponseBoundaryRuntime.
    """

    def __init__(
        self,
        *,
        config: LiveSessionRunnerConfig | None = None,
        response_generator: LiveResponseGenerator | None = None,
        live_state: LiveSessionStateRuntime | None = None,
        response_boundary: LiveResponseBoundaryRuntime | None = None,
        bridge: LiveEventBridgeRuntime | None = None,
        audio: LiveAudioRuntime | None = None,
        wake: LiveWakeEngagementRuntime | None = None,
        dialogue: LiveDialogueRuntime | None = None,
        interruption: LiveInterruptionRuntime | None = None,
        health: LiveHealthMonitorRuntime | None = None,
        recovery: LiveRecoveryRuntime | None = None,
    ) -> None:
        self._config = config or LiveSessionRunnerConfig()
        session_config = self._config.session_config

        self._state = live_state or LiveSessionStateRuntime(config=session_config)
        self._response_boundary = response_boundary or LiveResponseBoundaryRuntime(
            generator=response_generator
        )
        self._bridge = bridge or LiveEventBridgeRuntime(
            live_state=self._state,
            response_boundary=self._response_boundary,
        )
        self._audio = audio
        self._wake = wake or LiveWakeEngagementRuntime(
            live_state=self._state,
            bridge=self._bridge,
            config=session_config,
        )
        self._dialogue = dialogue or LiveDialogueRuntime(
            live_state=self._state,
            bridge=self._bridge,
            response_boundary=self._response_boundary,
            response_generator=response_generator,
            config=session_config,
        )
        self._interruption = interruption or LiveInterruptionRuntime(
            live_state=self._state,
            bridge=self._bridge,
            audio=self._audio,
            dialogue=self._dialogue,
            config=session_config,
        )
        self._health = health or LiveHealthMonitorRuntime(
            live_state=self._state,
            bridge=self._bridge,
            audio=self._audio,
            response_boundary=self._response_boundary,
            dialogue=self._dialogue,
            interruption=self._interruption,
            config=session_config,
        )
        self._recovery = recovery or LiveRecoveryRuntime(
            live_state=self._state,
            bridge=self._bridge,
            health_monitor=self._health,
            config=session_config,
        )
        self._status = LiveSessionRunnerStatus.CREATED

    @property
    def live_state(self) -> LiveSessionStateRuntime:
        return self._state

    @property
    def dialogue(self) -> LiveDialogueRuntime:
        return self._dialogue

    @property
    def health(self) -> LiveHealthMonitorRuntime:
        return self._health

    @property
    def recovery(self) -> LiveRecoveryRuntime:
        return self._recovery

    def start(self) -> LiveSessionRunnerResult:
        start_result = self._state.start()
        if start_result.status == LiveSessionStateRuntimeStatus.BLOCKED:
            self._status = LiveSessionRunnerStatus.FAILED
            return self._result(
                status=self._status,
                operation=LiveSessionRunnerOperation.START,
                state_result=start_result,
                reason=start_result.reason,
            )

        audio_result: LiveAudioRuntimeResult | None = None
        if self._audio is not None and self._config.auto_prepare_audio:
            audio_result = self._audio.prepare()

        ready_result = self._state.mark_ready()
        if ready_result.status == LiveSessionStateRuntimeStatus.BLOCKED:
            self._status = LiveSessionRunnerStatus.FAILED
            return self._result(
                status=self._status,
                operation=LiveSessionRunnerOperation.START,
                state_result=ready_result,
                audio_result=audio_result,
                reason=ready_result.reason,
            )

        health_result: LiveHealthMonitorResult | None = None
        if self._config.auto_health_check:
            health_result = self._health.check()

        self._status = _runner_status_from_health(health_result)

        return self._result(
            status=self._status,
            operation=LiveSessionRunnerOperation.START,
            state_result=ready_result,
            audio_result=audio_result,
            health_result=health_result,
            reason="live session runner started",
        )

    def ingest_text(
        self,
        *,
        text: str,
        speech_probability: float = 1.0,
        confidence: float = 1.0,
        metadata: dict[str, object] | None = None,
    ) -> LiveSessionRunnerResult:
        if not text.strip():
            return self._result(
                status=self._status,
                operation=LiveSessionRunnerOperation.INGEST_TEXT,
                reason="empty live text ignored",
                metadata=metadata or {},
            )

        wake_result = self._wake.evaluate(
            LiveWakeEngagementRequest(
                text=text,
                speech_probability=speech_probability,
                assistant_is_speaking=self._state.state.assistant_speaking,
                metadata=metadata or {},
            )
        )

        if not wake_result.engaged:
            return self._result(
                status=self._status,
                operation=LiveSessionRunnerOperation.INGEST_TEXT,
                wake_result=wake_result,
                reason="text did not engage live session",
                metadata=metadata or {},
            )

        if self._state.state.current_turn_id is None:
            turn_result = self._dialogue.start_turn()
            if turn_result.status == LiveDialogueRuntimeStatus.BLOCKED:
                self._status = LiveSessionRunnerStatus.DEGRADED
                return self._result(
                    status=self._status,
                    operation=LiveSessionRunnerOperation.INGEST_TEXT,
                    wake_result=wake_result,
                    dialogue_result=turn_result,
                    reason=turn_result.reason,
                    metadata=metadata or {},
                )

        turn_id = self._state.state.current_turn_id
        if turn_id is None:
            self._status = LiveSessionRunnerStatus.FAILED
            return self._result(
                status=self._status,
                operation=LiveSessionRunnerOperation.INGEST_TEXT,
                wake_result=wake_result,
                reason="live session has no current turn id",
                metadata=metadata or {},
            )

        transcript = make_live_transcript(
            turn_id=turn_id,
            kind=LiveTranscriptKind.FINAL,
            text=text,
            confidence=confidence,
            metadata=metadata or {},
        )
        dialogue_result = self._dialogue.process_transcript(
            LiveDialogueRequest(
                transcript=transcript,
                user_is_speaking=False,
                assistant_is_speaking=self._state.state.assistant_speaking,
                safety=LiveResponseSafety.SAFE_TO_SPEAK,
                metadata=metadata or {},
            )
        )

        health_result: LiveHealthMonitorResult | None = None
        recovery_result: LiveRecoveryResult | None = None
        if self._config.auto_health_check:
            health_result = self._health.check()
            if self._config.auto_recover and health_result.needs_recovery:
                recovery_result = self._recovery.recover(health_result)

        self._status = _runner_status_from_results(
            dialogue_result=dialogue_result,
            health_result=health_result,
            recovery_result=recovery_result,
        )

        return self._result(
            status=self._status,
            operation=LiveSessionRunnerOperation.INGEST_TEXT,
            wake_result=wake_result,
            dialogue_result=dialogue_result,
            health_result=health_result,
            recovery_result=recovery_result,
            reason="live text ingested through dialogue runtime",
            metadata=metadata or {},
        )

    def handle_interrupt(
        self,
        *,
        text: str,
        confidence: float = 1.0,
        metadata: dict[str, object] | None = None,
    ) -> LiveSessionRunnerResult:
        interruption_result = self._interruption.request_interrupt(
            LiveInterruptionRequest(
                text=text,
                confidence=confidence,
                assistant_is_speaking=self._state.state.assistant_speaking,
                metadata=metadata or {},
            )
        )

        self._status = (
            LiveSessionRunnerStatus.DEGRADED
            if not interruption_result.succeeded
            else self._status
        )

        return self._result(
            status=self._status,
            operation=LiveSessionRunnerOperation.HANDLE_INTERRUPT,
            interruption_result=interruption_result,
            reason=interruption_result.reason,
            metadata=metadata or {},
        )

    def check_health(self) -> LiveSessionRunnerResult:
        health_result = self._health.check()
        self._status = _runner_status_from_health(health_result)

        return self._result(
            status=self._status,
            operation=LiveSessionRunnerOperation.CHECK_HEALTH,
            health_result=health_result,
            reason=health_result.reason,
        )

    def recover(self) -> LiveSessionRunnerResult:
        health_result = self._health.check()
        recovery_result = self._recovery.recover(health_result)

        self._status = (
            LiveSessionRunnerStatus.FAILED
            if recovery_result.status == LiveRecoveryRuntimeStatus.FAILED
            else LiveSessionRunnerStatus.DEGRADED
            if health_result.needs_recovery
            else LiveSessionRunnerStatus.RUNNING
        )

        return self._result(
            status=self._status,
            operation=LiveSessionRunnerOperation.RECOVER,
            health_result=health_result,
            recovery_result=recovery_result,
            reason=recovery_result.reason,
        )

    def shutdown(
        self,
        *,
        reason: LiveShutdownReason = LiveShutdownReason.USER_REQUEST,
    ) -> LiveSessionRunnerResult:
        state_result = self._state.stop(reason=reason)
        if state_result.status == LiveSessionStateRuntimeStatus.BLOCKED:
            self._status = LiveSessionRunnerStatus.DEGRADED
            return self._result(
                status=self._status,
                operation=LiveSessionRunnerOperation.SHUTDOWN,
                state_result=state_result,
                reason=state_result.reason,
            )

        self._status = LiveSessionRunnerStatus.STOPPED
        return self._result(
            status=self._status,
            operation=LiveSessionRunnerOperation.SHUTDOWN,
            state_result=state_result,
            reason="live session runner stopped",
        )

    def snapshot(self) -> LiveSessionRunnerSnapshot:
        state_snapshot = self._state.snapshot()
        health_snapshot = self._health.snapshot()
        recovery_snapshot = self._recovery.snapshot()

        return LiveSessionRunnerSnapshot(
            status=self._status,
            state_status=state_snapshot.state.status.value,
            state_phase=state_snapshot.state.phase.value,
            event_count=state_snapshot.event_count,
            health_status=health_snapshot.status.value,
            recovery_attempts=recovery_snapshot.attempt_count,
            created_at=utc_now(),
        )

    def _result(
        self,
        *,
        status: LiveSessionRunnerStatus,
        operation: LiveSessionRunnerOperation,
        state_result: LiveSessionStateRuntimeResult | None = None,
        wake_result: LiveWakeEngagementResult | None = None,
        dialogue_result: LiveDialogueResult | None = None,
        interruption_result: LiveInterruptionResult | None = None,
        audio_result: LiveAudioRuntimeResult | None = None,
        health_result: LiveHealthMonitorResult | None = None,
        recovery_result: LiveRecoveryResult | None = None,
        reason: str,
        metadata: dict[str, object] | None = None,
    ) -> LiveSessionRunnerResult:
        return LiveSessionRunnerResult(
            status=status,
            operation=operation,
            state_result=state_result,
            wake_result=wake_result,
            dialogue_result=dialogue_result,
            interruption_result=interruption_result,
            audio_result=audio_result,
            health_result=health_result,
            recovery_result=recovery_result,
            reason=reason,
            created_at=utc_now(),
            metadata=metadata or {},
        )


def _runner_status_from_health(
    health_result: LiveHealthMonitorResult | None,
) -> LiveSessionRunnerStatus:
    if health_result is None:
        return LiveSessionRunnerStatus.RUNNING

    if health_result.status == LiveHealthMonitorStatus.HEALTHY:
        return LiveSessionRunnerStatus.RUNNING

    if health_result.status == LiveHealthMonitorStatus.DEGRADED:
        return LiveSessionRunnerStatus.DEGRADED

    if health_result.status == LiveHealthMonitorStatus.CRITICAL:
        return LiveSessionRunnerStatus.DEGRADED

    return LiveSessionRunnerStatus.FAILED


def _runner_status_from_results(
    *,
    dialogue_result: LiveDialogueResult,
    health_result: LiveHealthMonitorResult | None,
    recovery_result: LiveRecoveryResult | None,
) -> LiveSessionRunnerStatus:
    if dialogue_result.status == LiveDialogueRuntimeStatus.BLOCKED:
        return LiveSessionRunnerStatus.DEGRADED

    if recovery_result is not None:
        if recovery_result.status == LiveRecoveryRuntimeStatus.FAILED:
            return LiveSessionRunnerStatus.FAILED

    return _runner_status_from_health(health_result)