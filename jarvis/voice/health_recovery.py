from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from jarvis.voice.barge_in_runtime import (
    VoiceBargeInRuntimeStatus,
    VoiceBargeInSnapshot,
)
from jarvis.voice.cognition_response import (
    VoiceCognitionSnapshot,
    VoiceCognitionStatus,
)
from jarvis.voice.contracts import utc_now
from jarvis.voice.microphone_capture import (
    VoiceMicrophoneCaptureSnapshot,
    VoiceMicrophoneCaptureStatus,
)
from jarvis.voice.playback_runtime import (
    VoicePlaybackRuntimeStatus,
    VoicePlaybackSnapshot,
)
from jarvis.voice.stt_runtime import (
    VoiceSTTRuntimeStatus,
    VoiceSTTSnapshot,
)
from jarvis.voice.tts_runtime import (
    VoiceTTSRuntimeStatus,
    VoiceTTSSnapshot,
)
from jarvis.voice.voice_activity import (
    VoiceActivityRuntimeStatus,
    VoiceActivitySnapshot,
)


class VoiceHealthStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"
    RECOVERING = "recovering"


class VoiceHealthOperation(StrEnum):
    CHECK = "check"
    RECOVER = "recover"
    RESET = "reset"
    SNAPSHOT = "snapshot"


class VoiceHealthSubsystem(StrEnum):
    MICROPHONE = "microphone"
    VAD = "vad"
    STT = "stt"
    COGNITION = "cognition"
    TTS = "tts"
    PLAYBACK = "playback"
    BARGE_IN = "barge_in"


class VoiceRecoveryAction(StrEnum):
    NONE = "none"
    PREPARE = "prepare"
    RESET = "reset"
    CLEAR = "clear"
    STOP = "stop"
    RESTART = "restart"
    FAIL_SAFE = "fail_safe"


@dataclass(frozen=True, slots=True)
class VoiceHealthPolicy:
    max_failed_subsystems: int = 0
    max_degraded_subsystems: int = 2
    max_recovery_attempts_per_check: int = 3
    max_recent_failures_before_critical: int = 3
    require_microphone: bool = True
    require_stt: bool = True
    require_tts: bool = True
    require_playback: bool = True
    require_barge_in: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_failed_subsystems < 0:
            raise ValueError("max_failed_subsystems cannot be negative.")
        if self.max_degraded_subsystems < 0:
            raise ValueError("max_degraded_subsystems cannot be negative.")
        if self.max_recovery_attempts_per_check < 0:
            raise ValueError(
                "max_recovery_attempts_per_check cannot be negative."
            )
        if self.max_recent_failures_before_critical < 1:
            raise ValueError(
                "max_recent_failures_before_critical must be positive."
            )


@dataclass(frozen=True, slots=True)
class VoiceSubsystemHealth:
    subsystem: VoiceHealthSubsystem
    status: VoiceHealthStatus
    message: str
    recoverable: bool
    recommended_action: VoiceRecoveryAction
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise ValueError("voice subsystem health message cannot be empty.")


@dataclass(frozen=True, slots=True)
class VoiceRecoveryAttempt:
    subsystem: VoiceHealthSubsystem
    action: VoiceRecoveryAction
    succeeded: bool
    message: str
    latency_ms: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise ValueError("voice recovery attempt message cannot be empty.")
        if self.latency_ms < 0:
            raise ValueError("voice recovery latency cannot be negative.")


@dataclass(frozen=True, slots=True)
class VoiceHealthResult:
    status: VoiceHealthStatus
    operation: VoiceHealthOperation
    subsystem_health: tuple[VoiceSubsystemHealth, ...]
    recovery_attempts: tuple[VoiceRecoveryAttempt, ...]
    message: str
    latency_ms: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def healthy(self) -> bool:
        return self.status == VoiceHealthStatus.HEALTHY


@dataclass(frozen=True, slots=True)
class VoiceHealthSnapshot:
    status: VoiceHealthStatus
    checks: int
    recovery_runs: int
    recovery_attempts: int
    failed_recoveries: int
    last_message: str | None
    last_latency_ms: float | None
    last_error: str | None
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


class VoiceRecoverableComponent(Protocol):
    def prepare(self) -> object:
        raise NotImplementedError

    def reset(self) -> object:
        raise NotImplementedError


class VoiceClearableComponent(Protocol):
    def clear(self) -> object:
        raise NotImplementedError


class VoiceStoppableComponent(Protocol):
    def stop(self) -> object:
        raise NotImplementedError


class VoiceMicrophoneHealthComponent(Protocol):
    def snapshot(self) -> VoiceMicrophoneCaptureSnapshot:
        raise NotImplementedError

    def prepare(self) -> object:
        raise NotImplementedError

    def stop(self) -> object:
        raise NotImplementedError


class VoiceActivityHealthComponent(Protocol):
    def snapshot(self) -> VoiceActivitySnapshot:
        raise NotImplementedError

    def reset(self) -> object:
        raise NotImplementedError


class VoiceSTTHealthComponent(Protocol):
    def snapshot(self) -> VoiceSTTSnapshot:
        raise NotImplementedError

    def prepare(self) -> object:
        raise NotImplementedError

    def reset(self) -> object:
        raise NotImplementedError


class VoiceCognitionHealthComponent(Protocol):
    def snapshot(self) -> VoiceCognitionSnapshot:
        raise NotImplementedError

    def prepare(self) -> object:
        raise NotImplementedError


class VoiceTTSHealthComponent(Protocol):
    def snapshot(self) -> VoiceTTSSnapshot:
        raise NotImplementedError

    def prepare(self) -> object:
        raise NotImplementedError

    def reset(self) -> object:
        raise NotImplementedError


class VoicePlaybackHealthComponent(Protocol):
    def snapshot(self) -> VoicePlaybackSnapshot:
        raise NotImplementedError

    def prepare(self) -> object:
        raise NotImplementedError

    def clear(self) -> object:
        raise NotImplementedError

    def stop(self) -> object:
        raise NotImplementedError

    def reset(self) -> object:
        raise NotImplementedError


class VoiceBargeInHealthComponent(Protocol):
    def snapshot(self) -> VoiceBargeInSnapshot:
        raise NotImplementedError

    def prepare(self) -> object:
        raise NotImplementedError

    def reset(self) -> object:
        raise NotImplementedError


@dataclass(slots=True)
class VoiceHealthComponents:
    microphone: VoiceMicrophoneHealthComponent | None = None
    vad: VoiceActivityHealthComponent | None = None
    stt: VoiceSTTHealthComponent | None = None
    cognition: VoiceCognitionHealthComponent | None = None
    tts: VoiceTTSHealthComponent | None = None
    playback: VoicePlaybackHealthComponent | None = None
    barge_in: VoiceBargeInHealthComponent | None = None


class VoiceHealthRecoveryRuntime:
    """
    Step 51I voice health + recovery runtime.

    Watches the real voice chain:
    microphone -> VAD -> STT -> cognition -> TTS -> playback -> barge-in.

    It does not generate responses, synthesize speech, play audio, or listen.
    It checks subsystem health and performs bounded recovery actions.
    """

    def __init__(
        self,
        *,
        components: VoiceHealthComponents | None = None,
        policy: VoiceHealthPolicy | None = None,
    ) -> None:
        self._components = components or VoiceHealthComponents()
        self._policy = policy or VoiceHealthPolicy()
        self._status = VoiceHealthStatus.DEGRADED
        self._checks = 0
        self._recovery_runs = 0
        self._recovery_attempts = 0
        self._failed_recoveries = 0
        self._last_message: str | None = None
        self._last_latency_ms: float | None = None
        self._last_error: str | None = None

    def check(self) -> VoiceHealthResult:
        started = time.perf_counter()
        self._checks += 1

        health = (
            self._check_microphone(),
            self._check_vad(),
            self._check_stt(),
            self._check_cognition(),
            self._check_tts(),
            self._check_playback(),
            self._check_barge_in(),
        )

        status = _aggregate_health_status(
            health=health,
            policy=self._policy,
        )
        self._status = status

        message = _message_for_status(status)
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._last_message = message
        self._last_latency_ms = latency_ms

        return VoiceHealthResult(
            status=status,
            operation=VoiceHealthOperation.CHECK,
            subsystem_health=health,
            recovery_attempts=(),
            message=message,
            latency_ms=latency_ms,
            created_at=utc_now(),
            metadata={
                "failed": _count_status(health, VoiceHealthStatus.FAILED),
                "critical": _count_status(health, VoiceHealthStatus.CRITICAL),
                "degraded": _count_status(health, VoiceHealthStatus.DEGRADED),
            },
        )

    def recover(self) -> VoiceHealthResult:
        started = time.perf_counter()
        self._recovery_runs += 1

        before = self.check()
        attempts: list[VoiceRecoveryAttempt] = []

        for health in before.subsystem_health:
            if len(attempts) >= self._policy.max_recovery_attempts_per_check:
                break
            if not health.recoverable:
                continue
            if health.recommended_action == VoiceRecoveryAction.NONE:
                continue

            attempts.append(self._recover_subsystem(health))

        after = self.check()
        status = (
            VoiceHealthStatus.RECOVERING
            if attempts and after.status != VoiceHealthStatus.HEALTHY
            else after.status
        )

        latency_ms = (time.perf_counter() - started) * 1000.0
        message = (
            "voice recovery completed"
            if after.status == VoiceHealthStatus.HEALTHY
            else "voice recovery completed with remaining issues"
        )
        self._status = status
        self._last_message = message
        self._last_latency_ms = latency_ms

        return VoiceHealthResult(
            status=status,
            operation=VoiceHealthOperation.RECOVER,
            subsystem_health=after.subsystem_health,
            recovery_attempts=tuple(attempts),
            message=message,
            latency_ms=latency_ms,
            created_at=utc_now(),
            metadata={
                "attempts": len(attempts),
                "post_recovery_status": after.status.value,
            },
        )

    def reset(self) -> VoiceHealthResult:
        started = time.perf_counter()
        self._status = VoiceHealthStatus.DEGRADED
        self._last_message = None
        self._last_latency_ms = None
        self._last_error = None
        return VoiceHealthResult(
            status=self._status,
            operation=VoiceHealthOperation.RESET,
            subsystem_health=(),
            recovery_attempts=(),
            message="voice health runtime reset",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            created_at=utc_now(),
        )

    def snapshot(self) -> VoiceHealthSnapshot:
        return VoiceHealthSnapshot(
            status=self._status,
            checks=self._checks,
            recovery_runs=self._recovery_runs,
            recovery_attempts=self._recovery_attempts,
            failed_recoveries=self._failed_recoveries,
            last_message=self._last_message,
            last_latency_ms=self._last_latency_ms,
            last_error=self._last_error,
            created_at=utc_now(),
        )

    def _check_microphone(self) -> VoiceSubsystemHealth:
        component = self._components.microphone
        if component is None:
            return _missing(
                VoiceHealthSubsystem.MICROPHONE,
                required=self._policy.require_microphone,
            )

        try:
            snapshot = component.snapshot()
        except Exception as exc:
            return _failed_snapshot(VoiceHealthSubsystem.MICROPHONE, exc)

        if snapshot.status == VoiceMicrophoneCaptureStatus.FAILED:
            return VoiceSubsystemHealth(
                subsystem=VoiceHealthSubsystem.MICROPHONE,
                status=VoiceHealthStatus.FAILED,
                message="microphone capture failed",
                recoverable=True,
                recommended_action=VoiceRecoveryAction.PREPARE,
                metadata={"last_error": snapshot.last_error},
            )

        if snapshot.status == VoiceMicrophoneCaptureStatus.DEGRADED:
            return VoiceSubsystemHealth(
                subsystem=VoiceHealthSubsystem.MICROPHONE,
                status=VoiceHealthStatus.DEGRADED,
                message="microphone capture degraded",
                recoverable=True,
                recommended_action=VoiceRecoveryAction.PREPARE,
                metadata={"failures": snapshot.consecutive_failures},
            )

        return _healthy(VoiceHealthSubsystem.MICROPHONE)

    def _check_vad(self) -> VoiceSubsystemHealth:
        component = self._components.vad
        if component is None:
            return _missing(VoiceHealthSubsystem.VAD, required=False)

        try:
            snapshot = component.snapshot()
        except Exception as exc:
            return _failed_snapshot(VoiceHealthSubsystem.VAD, exc)

        if snapshot.status == VoiceActivityRuntimeStatus.FAILED:
            return VoiceSubsystemHealth(
                subsystem=VoiceHealthSubsystem.VAD,
                status=VoiceHealthStatus.FAILED,
                message="voice activity runtime failed",
                recoverable=True,
                recommended_action=VoiceRecoveryAction.RESET,
            )

        if snapshot.status == VoiceActivityRuntimeStatus.DEGRADED:
            return VoiceSubsystemHealth(
                subsystem=VoiceHealthSubsystem.VAD,
                status=VoiceHealthStatus.DEGRADED,
                message="voice activity runtime degraded",
                recoverable=True,
                recommended_action=VoiceRecoveryAction.RESET,
            )

        return _healthy(VoiceHealthSubsystem.VAD)

    def _check_stt(self) -> VoiceSubsystemHealth:
        component = self._components.stt
        if component is None:
            return _missing(
                VoiceHealthSubsystem.STT,
                required=self._policy.require_stt,
            )

        try:
            snapshot = component.snapshot()
        except Exception as exc:
            return _failed_snapshot(VoiceHealthSubsystem.STT, exc)

        if snapshot.status == VoiceSTTRuntimeStatus.FAILED:
            return VoiceSubsystemHealth(
                subsystem=VoiceHealthSubsystem.STT,
                status=VoiceHealthStatus.FAILED,
                message="STT runtime failed",
                recoverable=True,
                recommended_action=VoiceRecoveryAction.RESET,
                metadata={"last_error": snapshot.last_error},
            )

        if snapshot.status == VoiceSTTRuntimeStatus.DEGRADED:
            return VoiceSubsystemHealth(
                subsystem=VoiceHealthSubsystem.STT,
                status=VoiceHealthStatus.DEGRADED,
                message="STT runtime degraded",
                recoverable=True,
                recommended_action=VoiceRecoveryAction.PREPARE,
                metadata={
                    "empty_results": snapshot.empty_results,
                    "low_confidence_results": snapshot.low_confidence_results,
                },
            )

        return _healthy(VoiceHealthSubsystem.STT)

    def _check_cognition(self) -> VoiceSubsystemHealth:
        component = self._components.cognition
        if component is None:
            return _missing(VoiceHealthSubsystem.COGNITION, required=False)

        try:
            snapshot = component.snapshot()
        except Exception as exc:
            return _failed_snapshot(VoiceHealthSubsystem.COGNITION, exc)

        if snapshot.status == VoiceCognitionStatus.FAILED:
            return VoiceSubsystemHealth(
                subsystem=VoiceHealthSubsystem.COGNITION,
                status=VoiceHealthStatus.FAILED,
                message="voice cognition failed",
                recoverable=True,
                recommended_action=VoiceRecoveryAction.PREPARE,
                metadata={"last_error": snapshot.last_error},
            )

        if snapshot.status == VoiceCognitionStatus.DEGRADED:
            return VoiceSubsystemHealth(
                subsystem=VoiceHealthSubsystem.COGNITION,
                status=VoiceHealthStatus.DEGRADED,
                message="voice cognition degraded",
                recoverable=True,
                recommended_action=VoiceRecoveryAction.PREPARE,
                metadata={"last_error": snapshot.last_error},
            )

        return _healthy(VoiceHealthSubsystem.COGNITION)

    def _check_tts(self) -> VoiceSubsystemHealth:
        component = self._components.tts
        if component is None:
            return _missing(
                VoiceHealthSubsystem.TTS,
                required=self._policy.require_tts,
            )

        try:
            snapshot = component.snapshot()
        except Exception as exc:
            return _failed_snapshot(VoiceHealthSubsystem.TTS, exc)

        if snapshot.status == VoiceTTSRuntimeStatus.FAILED:
            return VoiceSubsystemHealth(
                subsystem=VoiceHealthSubsystem.TTS,
                status=VoiceHealthStatus.FAILED,
                message="TTS runtime failed",
                recoverable=True,
                recommended_action=VoiceRecoveryAction.RESET,
                metadata={"last_error": snapshot.last_error},
            )

        if snapshot.status == VoiceTTSRuntimeStatus.DEGRADED:
            return VoiceSubsystemHealth(
                subsystem=VoiceHealthSubsystem.TTS,
                status=VoiceHealthStatus.DEGRADED,
                message="TTS runtime degraded",
                recoverable=True,
                recommended_action=VoiceRecoveryAction.PREPARE,
                metadata={"last_error": snapshot.last_error},
            )

        return _healthy(VoiceHealthSubsystem.TTS)

    def _check_playback(self) -> VoiceSubsystemHealth:
        component = self._components.playback
        if component is None:
            return _missing(
                VoiceHealthSubsystem.PLAYBACK,
                required=self._policy.require_playback,
            )

        try:
            snapshot = component.snapshot()
        except Exception as exc:
            return _failed_snapshot(VoiceHealthSubsystem.PLAYBACK, exc)

        if snapshot.status == VoicePlaybackRuntimeStatus.FAILED:
            return VoiceSubsystemHealth(
                subsystem=VoiceHealthSubsystem.PLAYBACK,
                status=VoiceHealthStatus.FAILED,
                message="playback runtime failed",
                recoverable=True,
                recommended_action=VoiceRecoveryAction.RESET,
                metadata={"last_error": snapshot.last_error},
            )

        if snapshot.status == VoicePlaybackRuntimeStatus.DEGRADED:
            return VoiceSubsystemHealth(
                subsystem=VoiceHealthSubsystem.PLAYBACK,
                status=VoiceHealthStatus.DEGRADED,
                message="playback runtime degraded",
                recoverable=True,
                recommended_action=VoiceRecoveryAction.PREPARE,
                metadata={"last_error": snapshot.last_error},
            )

        return _healthy(VoiceHealthSubsystem.PLAYBACK)

    def _check_barge_in(self) -> VoiceSubsystemHealth:
        component = self._components.barge_in
        if component is None:
            return _missing(
                VoiceHealthSubsystem.BARGE_IN,
                required=self._policy.require_barge_in,
            )

        try:
            snapshot = component.snapshot()
        except Exception as exc:
            return _failed_snapshot(VoiceHealthSubsystem.BARGE_IN, exc)

        if snapshot.status == VoiceBargeInRuntimeStatus.FAILED:
            return VoiceSubsystemHealth(
                subsystem=VoiceHealthSubsystem.BARGE_IN,
                status=VoiceHealthStatus.FAILED,
                message="barge-in runtime failed",
                recoverable=True,
                recommended_action=VoiceRecoveryAction.RESET,
                metadata={"last_error": snapshot.last_error},
            )

        if snapshot.status == VoiceBargeInRuntimeStatus.DEGRADED:
            return VoiceSubsystemHealth(
                subsystem=VoiceHealthSubsystem.BARGE_IN,
                status=VoiceHealthStatus.DEGRADED,
                message="barge-in runtime degraded",
                recoverable=True,
                recommended_action=VoiceRecoveryAction.PREPARE,
                metadata={"last_error": snapshot.last_error},
            )

        return _healthy(VoiceHealthSubsystem.BARGE_IN)

    def _recover_subsystem(
        self,
        health: VoiceSubsystemHealth,
    ) -> VoiceRecoveryAttempt:
        started = time.perf_counter()
        self._recovery_attempts += 1

        try:
            component = _component_for_subsystem(
                components=self._components,
                subsystem=health.subsystem,
            )
            if component is None:
                return self._attempt(
                    health=health,
                    started=started,
                    succeeded=False,
                    message="component unavailable for recovery",
                )

            _apply_recovery_action(component, health.recommended_action)
            return self._attempt(
                health=health,
                started=started,
                succeeded=True,
                message="recovery action completed",
            )
        except Exception as exc:
            self._failed_recoveries += 1
            self._last_error = str(exc)
            return self._attempt(
                health=health,
                started=started,
                succeeded=False,
                message=f"recovery action failed: {exc}",
            )

    def _attempt(
        self,
        *,
        health: VoiceSubsystemHealth,
        started: float,
        succeeded: bool,
        message: str,
    ) -> VoiceRecoveryAttempt:
        return VoiceRecoveryAttempt(
            subsystem=health.subsystem,
            action=health.recommended_action,
            succeeded=succeeded,
            message=message,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            created_at=utc_now(),
        )


def _healthy(subsystem: VoiceHealthSubsystem) -> VoiceSubsystemHealth:
    return VoiceSubsystemHealth(
        subsystem=subsystem,
        status=VoiceHealthStatus.HEALTHY,
        message=f"{subsystem.value} healthy",
        recoverable=False,
        recommended_action=VoiceRecoveryAction.NONE,
    )


def _missing(
    subsystem: VoiceHealthSubsystem,
    *,
    required: bool,
) -> VoiceSubsystemHealth:
    if required:
        return VoiceSubsystemHealth(
            subsystem=subsystem,
            status=VoiceHealthStatus.FAILED,
            message=f"{subsystem.value} subsystem missing",
            recoverable=False,
            recommended_action=VoiceRecoveryAction.FAIL_SAFE,
        )

    return VoiceSubsystemHealth(
        subsystem=subsystem,
        status=VoiceHealthStatus.DEGRADED,
        message=f"{subsystem.value} subsystem not connected",
        recoverable=False,
        recommended_action=VoiceRecoveryAction.NONE,
    )


def _failed_snapshot(
    subsystem: VoiceHealthSubsystem,
    exc: Exception,
) -> VoiceSubsystemHealth:
    return VoiceSubsystemHealth(
        subsystem=subsystem,
        status=VoiceHealthStatus.FAILED,
        message=f"{subsystem.value} snapshot failed",
        recoverable=True,
        recommended_action=VoiceRecoveryAction.RESET,
        metadata={"error": str(exc)},
    )


def _aggregate_health_status(
    *,
    health: tuple[VoiceSubsystemHealth, ...],
    policy: VoiceHealthPolicy,
) -> VoiceHealthStatus:
    failed = _count_status(health, VoiceHealthStatus.FAILED)
    critical = _count_status(health, VoiceHealthStatus.CRITICAL)
    degraded = _count_status(health, VoiceHealthStatus.DEGRADED)

    if critical > 0:
        return VoiceHealthStatus.CRITICAL
    if failed > policy.max_failed_subsystems:
        return VoiceHealthStatus.FAILED
    if degraded > policy.max_degraded_subsystems:
        return VoiceHealthStatus.CRITICAL
    if degraded > 0 or failed > 0:
        return VoiceHealthStatus.DEGRADED
    return VoiceHealthStatus.HEALTHY


def _count_status(
    health: tuple[VoiceSubsystemHealth, ...],
    status: VoiceHealthStatus,
) -> int:
    return sum(1 for item in health if item.status == status)


def _message_for_status(status: VoiceHealthStatus) -> str:
    if status == VoiceHealthStatus.HEALTHY:
        return "voice runtime healthy"
    if status == VoiceHealthStatus.DEGRADED:
        return "voice runtime degraded"
    if status == VoiceHealthStatus.CRITICAL:
        return "voice runtime critical"
    if status == VoiceHealthStatus.FAILED:
        return "voice runtime failed"
    return "voice runtime recovering"


def _component_for_subsystem(
    *,
    components: VoiceHealthComponents,
    subsystem: VoiceHealthSubsystem,
) -> object | None:
    if subsystem == VoiceHealthSubsystem.MICROPHONE:
        return components.microphone
    if subsystem == VoiceHealthSubsystem.VAD:
        return components.vad
    if subsystem == VoiceHealthSubsystem.STT:
        return components.stt
    if subsystem == VoiceHealthSubsystem.COGNITION:
        return components.cognition
    if subsystem == VoiceHealthSubsystem.TTS:
        return components.tts
    if subsystem == VoiceHealthSubsystem.PLAYBACK:
        return components.playback
    if subsystem == VoiceHealthSubsystem.BARGE_IN:
        return components.barge_in
    return None


def _apply_recovery_action(
    component: object,
    action: VoiceRecoveryAction,
) -> None:
    if action == VoiceRecoveryAction.NONE:
        return
    if action == VoiceRecoveryAction.PREPARE and hasattr(component, "prepare"):
        component.prepare()
        return
    if action == VoiceRecoveryAction.RESET and hasattr(component, "reset"):
        component.reset()
        return
    if action == VoiceRecoveryAction.CLEAR and hasattr(component, "clear"):
        component.clear()
        return
    if action == VoiceRecoveryAction.STOP and hasattr(component, "stop"):
        component.stop()
        return
    raise RuntimeError(f"unsupported recovery action: {action.value}")