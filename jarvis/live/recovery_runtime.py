from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

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
from jarvis.live.event_bridge import (
    LiveEventBridgeRequest,
    LiveEventBridgeResult,
    LiveEventBridgeRuntime,
    LiveEventBridgeStatus,
)
from jarvis.live.health_monitor import (
    LiveHealthMonitorResult,
    LiveHealthMonitorRuntime,
    LiveHealthMonitorStatus,
    LiveHealthSignal,
    LiveHealthSignalKind,
)
from jarvis.live.session_state import (
    LiveSessionStateRuntime,
    LiveSessionStateRuntimeResult,
    LiveSessionStateRuntimeStatus,
)


class LiveRecoveryRuntimeStatus(StrEnum):
    READY = "ready"
    BLOCKED = "blocked"
    FAILED = "failed"


class LiveRecoveryOperation(StrEnum):
    EVALUATE = "evaluate"
    RECOVER = "recover"
    MARK_RECOVERED = "mark_recovered"
    ABANDON = "abandon"
    SNAPSHOT = "snapshot"


class LiveRecoveryAction(StrEnum):
    NONE = "none"
    REFRESH_SUBSYSTEM = "refresh_subsystem"
    RESTART_AUDIO_BOUNDARY = "restart_audio_boundary"
    RECONNECT_BRIDGE = "reconnect_bridge"
    RESET_DIALOGUE_TURN = "reset_dialogue_turn"
    CLEAR_INTERRUPTION_CONTEXT = "clear_interruption_context"
    ENTER_DEGRADED_MODE = "enter_degraded_mode"
    FAIL_SESSION = "fail_session"


@dataclass(frozen=True, slots=True)
class LiveRecoveryPolicy:
    max_recovery_attempts: int = 3
    allow_degraded_mode: bool = True
    fail_on_repeated_critical: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_recovery_attempts < 1:
            raise ValueError("max_recovery_attempts must be positive.")


@dataclass(frozen=True, slots=True)
class LiveRecoveryPlan:
    action: LiveRecoveryAction
    subsystem: LiveSubsystem
    reason: str
    signals: tuple[LiveHealthSignal, ...]
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("live recovery plan reason cannot be empty.")


@dataclass(frozen=True, slots=True)
class LiveRecoveryResult:
    status: LiveRecoveryRuntimeStatus
    operation: LiveRecoveryOperation
    plan: LiveRecoveryPlan | None
    state_result: LiveSessionStateRuntimeResult | None
    bridge_result: LiveEventBridgeResult | None
    reason: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == LiveRecoveryRuntimeStatus.READY


@dataclass(frozen=True, slots=True)
class LiveRecoverySnapshot:
    status: LiveRecoveryRuntimeStatus
    attempt_count: int
    recovered_count: int
    failed_count: int
    last_plan: LiveRecoveryPlan | None
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


class LiveRecoveryRuntime:
    """
    Step 50I Live Recovery Runtime.

    Consumes LiveHealthMonitorResult and attempts safe live recovery.

    It does not:
    - generate user-facing dialogue
    - call TTS
    - execute tools
    - access memory directly
    - bypass event bridge
    - fake recovery with scripted messages

    Recovery is typed state transition + typed event emission only.
    Any user-facing words must be generated later through the response boundary.
    """

    def __init__(
        self,
        *,
        live_state: LiveSessionStateRuntime | None = None,
        bridge: LiveEventBridgeRuntime | None = None,
        health_monitor: LiveHealthMonitorRuntime | None = None,
        config: LiveSessionConfig | None = None,
        policy: LiveRecoveryPolicy | None = None,
    ) -> None:
        self._config = config or default_live_session_config()
        self._state = live_state or LiveSessionStateRuntime(config=self._config)
        self._bridge = bridge or LiveEventBridgeRuntime(live_state=self._state)
        self._health_monitor = health_monitor
        self._policy = policy or LiveRecoveryPolicy()
        self._attempt_count = 0
        self._recovered_count = 0
        self._failed_count = 0
        self._last_plan: LiveRecoveryPlan | None = None

    @property
    def live_state(self) -> LiveSessionStateRuntime:
        return self._state

    def evaluate(
        self,
        health_result: LiveHealthMonitorResult,
    ) -> LiveRecoveryResult:
        plan = _plan_from_health_result(
            health_result=health_result,
            policy=self._policy,
            attempt_count=self._attempt_count,
        )
        self._last_plan = plan

        return LiveRecoveryResult(
            status=LiveRecoveryRuntimeStatus.READY,
            operation=LiveRecoveryOperation.EVALUATE,
            plan=plan,
            state_result=None,
            bridge_result=None,
            reason="live recovery plan evaluated",
            created_at=utc_now(),
            metadata={"action": plan.action.value},
        )

    def recover(
        self,
        health_result: LiveHealthMonitorResult,
    ) -> LiveRecoveryResult:
        plan = _plan_from_health_result(
            health_result=health_result,
            policy=self._policy,
            attempt_count=self._attempt_count,
        )
        self._last_plan = plan

        if plan.action == LiveRecoveryAction.NONE:
            return LiveRecoveryResult(
                status=LiveRecoveryRuntimeStatus.READY,
                operation=LiveRecoveryOperation.RECOVER,
                plan=plan,
                state_result=None,
                bridge_result=None,
                reason="no recovery required",
                created_at=utc_now(),
                metadata={"action": plan.action.value},
            )

        if self._attempt_count >= self._policy.max_recovery_attempts:
            return self._fail_session(
                plan=LiveRecoveryPlan(
                    action=LiveRecoveryAction.FAIL_SESSION,
                    subsystem=plan.subsystem,
                    reason="maximum live recovery attempts exceeded",
                    signals=plan.signals,
                    created_at=utc_now(),
                    metadata=plan.metadata,
                )
            )

        self._attempt_count += 1

        if plan.action == LiveRecoveryAction.FAIL_SESSION:
            return self._fail_session(plan=plan)

        if plan.action == LiveRecoveryAction.ENTER_DEGRADED_MODE:
            return self._enter_degraded_mode(plan=plan)

        return self._recover_subsystem(plan=plan)

    def mark_recovered(
        self,
        *,
        subsystem: LiveSubsystem,
        reason: str,
    ) -> LiveRecoveryResult:
        if not reason.strip():
            raise ValueError("mark recovered reason cannot be empty.")

        state_result = self._state.update_subsystem(
            LiveSubsystemState(
                subsystem=subsystem,
                status=LiveSubsystemStatus.READY,
                message=reason.strip(),
                updated_at=utc_now(),
            )
        )
        finish_result = self._state.finish_recovery()
        state_result = finish_result if finish_result.succeeded else state_result

        event = make_live_event(
            kind=LiveEventKind.RECOVERY_FINISHED,
            priority=LiveEventPriority.NORMAL,
            source=LiveSubsystem.RECOVERY,
            title="Live recovery finished",
            summary="Live recovery completed and subsystem returned ready.",
            metadata={
                "subsystem": subsystem.value,
                "reason": reason.strip(),
            },
        )
        bridge_result = self._bridge.bridge_event(
            LiveEventBridgeRequest(
                event=event,
                update_live_state=False,
                update_cognitive_state=True,
                allow_interruptions=True,
            )
        )

        self._recovered_count += 1

        return LiveRecoveryResult(
            status=LiveRecoveryRuntimeStatus.READY,
            operation=LiveRecoveryOperation.MARK_RECOVERED,
            plan=self._last_plan,
            state_result=state_result,
            bridge_result=bridge_result,
            reason="live recovery marked recovered",
            created_at=utc_now(),
            metadata={"subsystem": subsystem.value},
        )

    def abandon(
        self,
        *,
        subsystem: LiveSubsystem,
        reason: str,
    ) -> LiveRecoveryResult:
        if not reason.strip():
            raise ValueError("abandon recovery reason cannot be empty.")

        plan = LiveRecoveryPlan(
            action=LiveRecoveryAction.FAIL_SESSION,
            subsystem=subsystem,
            reason=reason.strip(),
            signals=(),
            created_at=utc_now(),
        )
        return self._fail_session(plan=plan)

    def snapshot(self) -> LiveRecoverySnapshot:
        status = (
            LiveRecoveryRuntimeStatus.FAILED
            if self._failed_count > 0
            else LiveRecoveryRuntimeStatus.READY
        )
        return LiveRecoverySnapshot(
            status=status,
            attempt_count=self._attempt_count,
            recovered_count=self._recovered_count,
            failed_count=self._failed_count,
            last_plan=self._last_plan,
            created_at=utc_now(),
        )

    def _recover_subsystem(
        self,
        *,
        plan: LiveRecoveryPlan,
    ) -> LiveRecoveryResult:
        state_result = self._state.enter_recovery(
            subsystem=plan.subsystem,
            reason=plan.reason,
        )
        if state_result.status == LiveSessionStateRuntimeStatus.BLOCKED:
            return LiveRecoveryResult(
                status=LiveRecoveryRuntimeStatus.BLOCKED,
                operation=LiveRecoveryOperation.RECOVER,
                plan=plan,
                state_result=state_result,
                bridge_result=None,
                reason=state_result.reason,
                created_at=utc_now(),
                metadata={"action": plan.action.value},
            )

        event = make_live_event(
            kind=LiveEventKind.RECOVERY_STARTED,
            priority=LiveEventPriority.HIGH,
            source=LiveSubsystem.RECOVERY,
            title="Live recovery started",
            summary="Live recovery started for a degraded subsystem.",
            metadata={
                "action": plan.action.value,
                "subsystem": plan.subsystem.value,
                "reason": plan.reason,
            },
        )
        bridge_result = self._bridge.bridge_event(
            LiveEventBridgeRequest(
                event=event,
                update_live_state=False,
                update_cognitive_state=True,
                allow_interruptions=True,
            )
        )

        return LiveRecoveryResult(
            status=(
                LiveRecoveryRuntimeStatus.READY
                if bridge_result.status != LiveEventBridgeStatus.BLOCKED
                else LiveRecoveryRuntimeStatus.BLOCKED
            ),
            operation=LiveRecoveryOperation.RECOVER,
            plan=plan,
            state_result=state_result,
            bridge_result=bridge_result,
            reason="live subsystem recovery started",
            created_at=utc_now(),
            metadata={"action": plan.action.value},
        )

    def _enter_degraded_mode(
        self,
        *,
        plan: LiveRecoveryPlan,
    ) -> LiveRecoveryResult:
        state_result = self._state.enter_recovery(
            subsystem=plan.subsystem,
            reason=plan.reason,
        )

        event = make_live_event(
            kind=LiveEventKind.RECOVERY_STARTED,
            priority=LiveEventPriority.HIGH,
            source=LiveSubsystem.RECOVERY,
            title="Live degraded mode entered",
            summary="Live session entered degraded mode for safe continuation.",
            metadata={
                "action": plan.action.value,
                "subsystem": plan.subsystem.value,
                "reason": plan.reason,
            },
        )
        bridge_result = self._bridge.bridge_event(
            LiveEventBridgeRequest(
                event=event,
                update_live_state=False,
                update_cognitive_state=True,
                allow_interruptions=True,
            )
        )

        return LiveRecoveryResult(
            status=LiveRecoveryRuntimeStatus.READY,
            operation=LiveRecoveryOperation.RECOVER,
            plan=plan,
            state_result=state_result,
            bridge_result=bridge_result,
            reason="live degraded mode entered",
            created_at=utc_now(),
            metadata={"action": plan.action.value},
        )

    def _fail_session(
        self,
        *,
        plan: LiveRecoveryPlan,
    ) -> LiveRecoveryResult:
        state_result = self._state.update_health(LiveHealthStatus.FAILED)

        event = make_live_event(
            kind=LiveEventKind.ERROR,
            priority=LiveEventPriority.CRITICAL,
            source=LiveSubsystem.RECOVERY,
            title="Live recovery failed",
            summary="Live recovery failed and session was marked failed.",
            metadata={
                "action": plan.action.value,
                "subsystem": plan.subsystem.value,
                "reason": plan.reason,
            },
        )
        bridge_result = self._bridge.bridge_event(
            LiveEventBridgeRequest(
                event=event,
                update_live_state=False,
                update_cognitive_state=True,
                allow_interruptions=True,
            )
        )

        self._failed_count += 1

        return LiveRecoveryResult(
            status=LiveRecoveryRuntimeStatus.FAILED,
            operation=LiveRecoveryOperation.RECOVER,
            plan=plan,
            state_result=state_result,
            bridge_result=bridge_result,
            reason="live recovery failed",
            created_at=utc_now(),
            metadata={"action": plan.action.value},
        )


def _plan_from_health_result(
    *,
    health_result: LiveHealthMonitorResult,
    policy: LiveRecoveryPolicy,
    attempt_count: int,
) -> LiveRecoveryPlan:
    worst_signal = _worst_signal(health_result.signals)
    subsystem = worst_signal.subsystem if worst_signal else LiveSubsystem.RECOVERY
    reason = worst_signal.message if worst_signal else health_result.reason

    if health_result.status == LiveHealthMonitorStatus.HEALTHY:
        action = LiveRecoveryAction.NONE
    elif health_result.status == LiveHealthMonitorStatus.DEGRADED:
        action = _degraded_action(worst_signal, policy)
    elif health_result.status == LiveHealthMonitorStatus.CRITICAL:
        action = _critical_action(
            worst_signal=worst_signal,
            policy=policy,
            attempt_count=attempt_count,
        )
    else:
        action = LiveRecoveryAction.FAIL_SESSION

    return LiveRecoveryPlan(
        action=action,
        subsystem=subsystem,
        reason=reason,
        signals=health_result.signals,
        created_at=utc_now(),
        metadata={
            "health_status": health_result.status.value,
            "attempt_count": attempt_count,
        },
    )


def _degraded_action(
    signal: LiveHealthSignal | None,
    policy: LiveRecoveryPolicy,
) -> LiveRecoveryAction:
    if signal is None:
        return LiveRecoveryAction.NONE

    if signal.kind == LiveHealthSignalKind.AUDIO:
        return LiveRecoveryAction.RESTART_AUDIO_BOUNDARY

    if signal.kind == LiveHealthSignalKind.EVENT_BRIDGE:
        return LiveRecoveryAction.RECONNECT_BRIDGE

    if signal.kind == LiveHealthSignalKind.DIALOGUE:
        return LiveRecoveryAction.RESET_DIALOGUE_TURN

    if signal.kind == LiveHealthSignalKind.INTERRUPTION:
        return LiveRecoveryAction.CLEAR_INTERRUPTION_CONTEXT

    if policy.allow_degraded_mode:
        return LiveRecoveryAction.ENTER_DEGRADED_MODE

    return LiveRecoveryAction.REFRESH_SUBSYSTEM


def _critical_action(
    *,
    worst_signal: LiveHealthSignal | None,
    policy: LiveRecoveryPolicy,
    attempt_count: int,
) -> LiveRecoveryAction:
    if policy.fail_on_repeated_critical:
        if attempt_count >= policy.max_recovery_attempts:
            return LiveRecoveryAction.FAIL_SESSION

    if worst_signal is None:
        return LiveRecoveryAction.ENTER_DEGRADED_MODE

    if worst_signal.kind == LiveHealthSignalKind.AUDIO:
        return LiveRecoveryAction.RESTART_AUDIO_BOUNDARY

    if worst_signal.kind == LiveHealthSignalKind.BLOCKED_PRESSURE:
        return LiveRecoveryAction.ENTER_DEGRADED_MODE

    if worst_signal.kind == LiveHealthSignalKind.SUBSYSTEM:
        return LiveRecoveryAction.REFRESH_SUBSYSTEM

    return LiveRecoveryAction.ENTER_DEGRADED_MODE


def _worst_signal(
    signals: tuple[LiveHealthSignal, ...],
) -> LiveHealthSignal | None:
    if not signals:
        return None

    priority = {
        LiveHealthMonitorStatus.FAILED: 4,
        LiveHealthMonitorStatus.CRITICAL: 3,
        LiveHealthMonitorStatus.DEGRADED: 2,
        LiveHealthMonitorStatus.HEALTHY: 1,
    }
    return max(signals, key=lambda signal: priority[signal.status])