from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from jarvis.voice.contracts import utc_now
from jarvis.voice.session_loop import VoiceSessionLoopStatus


class VoiceRuntimeIntegrityStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


class VoiceRuntimeIntegrityCheckKind(StrEnum):
    RESPONSE_ORIGIN = "response_origin"
    NO_FIXED_RESPONSE = "no_fixed_response"
    NO_FAKE_FALLBACK = "no_fake_fallback"
    FSM_CLEAN = "fsm_clean"
    SPEAKING_STATE = "speaking_state"
    RUNNING_STATE = "running_state"
    CRASH_BOUNDARY = "crash_boundary"


class VoiceRuntimeIntegritySnapshot(Protocol):
    @property
    def status(self) -> VoiceSessionLoopStatus:
        raise NotImplementedError

    @property
    def running(self) -> bool:
        raise NotImplementedError

    @property
    def assistant_speaking(self) -> bool:
        raise NotImplementedError

    @property
    def metadata(self) -> dict[str, object]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class VoiceRuntimeIntegrityInput:
    snapshot: VoiceRuntimeIntegritySnapshot
    response_origin: str | None = None
    uses_generated_response: bool = False
    uses_fixed_response: bool = False
    fake_fallback_enabled: bool = False
    deterministic_system_response: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VoiceRuntimeIntegrityCheck:
    kind: VoiceRuntimeIntegrityCheckKind
    passed: bool
    message: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VoiceRuntimeIntegrityReport:
    status: VoiceRuntimeIntegrityStatus
    checks: tuple[VoiceRuntimeIntegrityCheck, ...]
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == VoiceRuntimeIntegrityStatus.PASSED

    @property
    def failed_checks(self) -> tuple[VoiceRuntimeIntegrityCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)


class VoiceRuntimeIntegrityGuard:
    """
    Inspect-only guard for the live voice runtime.

    This is the contract behind "no fixed, no fake, no overlap, no silent
    crash". It does not produce responses or execute tools; it only validates
    that the runtime facts still obey the Pure JARVIS voice architecture.
    """

    def inspect(
        self,
        request: VoiceRuntimeIntegrityInput,
    ) -> VoiceRuntimeIntegrityReport:
        checks = (
            _response_origin_check(request),
            _no_fixed_response_check(request),
            _no_fake_fallback_check(request),
            _fsm_clean_check(request.snapshot),
            _speaking_state_check(request.snapshot),
            _running_state_check(request.snapshot),
            _crash_boundary_check(request.snapshot),
        )
        passed = all(check.passed for check in checks)
        return VoiceRuntimeIntegrityReport(
            status=(
                VoiceRuntimeIntegrityStatus.PASSED
                if passed
                else VoiceRuntimeIntegrityStatus.FAILED
            ),
            checks=checks,
            created_at=utc_now(),
            metadata={
                "response_origin": request.response_origin,
                "uses_generated_response": request.uses_generated_response,
                "uses_fixed_response": request.uses_fixed_response,
                "fake_fallback_enabled": request.fake_fallback_enabled,
                "deterministic_system_response": (
                    request.deterministic_system_response
                ),
                **request.metadata,
            },
        )


def _response_origin_check(
    request: VoiceRuntimeIntegrityInput,
) -> VoiceRuntimeIntegrityCheck:
    # Spoken responses must come through cognition. Silent operational controls
    # may have no response origin.
    no_spoken_response = (
        not request.uses_generated_response
        and not request.uses_fixed_response
        and request.response_origin is None
    )
    passed = no_spoken_response or (
        request.uses_generated_response
        and request.response_origin == "cognition_response_boundary"
    )
    return VoiceRuntimeIntegrityCheck(
        kind=VoiceRuntimeIntegrityCheckKind.RESPONSE_ORIGIN,
        passed=passed,
        message=(
            "response origin is cognition-owned or silent"
            if passed
            else "spoken response bypassed cognition boundary"
        ),
        metadata={
            "response_origin": request.response_origin,
            "uses_generated_response": request.uses_generated_response,
        },
    )


def _no_fixed_response_check(
    request: VoiceRuntimeIntegrityInput,
) -> VoiceRuntimeIntegrityCheck:
    passed = (
        not request.uses_fixed_response
        and not request.deterministic_system_response
    )
    return VoiceRuntimeIntegrityCheck(
        kind=VoiceRuntimeIntegrityCheckKind.NO_FIXED_RESPONSE,
        passed=passed,
        message=(
            "fixed spoken responses are blocked"
            if passed
            else "fixed or deterministic spoken response detected"
        ),
        metadata={
            "uses_fixed_response": request.uses_fixed_response,
            "deterministic_system_response": request.deterministic_system_response,
        },
    )


def _no_fake_fallback_check(
    request: VoiceRuntimeIntegrityInput,
) -> VoiceRuntimeIntegrityCheck:
    passed = not request.fake_fallback_enabled
    return VoiceRuntimeIntegrityCheck(
        kind=VoiceRuntimeIntegrityCheckKind.NO_FAKE_FALLBACK,
        passed=passed,
        message=(
            "fake fallback is disabled"
            if passed
            else "fake fallback is enabled"
        ),
        metadata={"fake_fallback_enabled": request.fake_fallback_enabled},
    )


def _fsm_clean_check(
    snapshot: VoiceRuntimeIntegritySnapshot,
) -> VoiceRuntimeIntegrityCheck:
    fsm_violations = _metadata_int(snapshot.metadata, "fsm_violations")
    passed = fsm_violations == 0
    return VoiceRuntimeIntegrityCheck(
        kind=VoiceRuntimeIntegrityCheckKind.FSM_CLEAN,
        passed=passed,
        message="FSM is clean" if passed else "FSM violations detected",
        metadata={"fsm_violations": fsm_violations},
    )


def _speaking_state_check(
    snapshot: VoiceRuntimeIntegritySnapshot,
) -> VoiceRuntimeIntegrityCheck:
    playback_status = str(snapshot.metadata.get("playback_status") or "")
    passed = _speaking_state_consistent(
        snapshot,
        playback_status=playback_status,
    )
    return VoiceRuntimeIntegrityCheck(
        kind=VoiceRuntimeIntegrityCheckKind.SPEAKING_STATE,
        passed=passed,
        message=(
            "speaking state is consistent"
            if passed
            else "assistant speaking/playback state overlaps illegally"
        ),
        metadata={
            "status": snapshot.status.value,
            "assistant_speaking": snapshot.assistant_speaking,
            "playback_status": playback_status or None,
        },
    )


def _running_state_check(
    snapshot: VoiceRuntimeIntegritySnapshot,
) -> VoiceRuntimeIntegrityCheck:
    passed = _running_state_consistent(snapshot)
    return VoiceRuntimeIntegrityCheck(
        kind=VoiceRuntimeIntegrityCheckKind.RUNNING_STATE,
        passed=passed,
        message=(
            "running state is consistent"
            if passed
            else "running flag conflicts with session status"
        ),
        metadata={
            "status": snapshot.status.value,
            "running": snapshot.running,
        },
    )


def _crash_boundary_check(
    snapshot: VoiceRuntimeIntegritySnapshot,
) -> VoiceRuntimeIntegrityCheck:
    passed = snapshot.status != VoiceSessionLoopStatus.FAILED
    return VoiceRuntimeIntegrityCheck(
        kind=VoiceRuntimeIntegrityCheckKind.CRASH_BOUNDARY,
        passed=passed,
        message=(
            "session is not failed"
            if passed
            else "session entered failed state"
        ),
        metadata={"status": snapshot.status.value},
    )


def _running_state_consistent(snapshot: VoiceRuntimeIntegritySnapshot) -> bool:
    if snapshot.running:
        return snapshot.status not in {
            VoiceSessionLoopStatus.CREATED,
            VoiceSessionLoopStatus.STOPPED,
            VoiceSessionLoopStatus.FAILED,
        }
    return snapshot.status in {
        VoiceSessionLoopStatus.CREATED,
        VoiceSessionLoopStatus.STOPPING,
        VoiceSessionLoopStatus.STOPPED,
        VoiceSessionLoopStatus.FAILED,
    }


def _speaking_state_consistent(
    snapshot: VoiceRuntimeIntegritySnapshot,
    *,
    playback_status: str,
) -> bool:
    if snapshot.assistant_speaking:
        return snapshot.status in {
            VoiceSessionLoopStatus.SPEAKING,
            VoiceSessionLoopStatus.USER_SPEAKING,
            VoiceSessionLoopStatus.INTERRUPTED,
        }
    return playback_status != "playing"


def _metadata_int(metadata: dict[str, object], key: str) -> int:
    value = metadata.get(key)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0
