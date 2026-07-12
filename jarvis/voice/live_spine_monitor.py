from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from jarvis.voice.session_loop import VoiceSessionLoopStatus


class VoiceLiveSpineStatus(StrEnum):
    HEALTHY = "healthy"
    ATTENTION = "attention"
    DEGRADED = "degraded"
    FAILED = "failed"


class VoiceLiveSpineSnapshot(Protocol):
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
    def final_transcripts(self) -> int:
        raise NotImplementedError

    @property
    def responses(self) -> int:
        raise NotImplementedError

    @property
    def interruptions(self) -> int:
        raise NotImplementedError

    @property
    def metadata(self) -> dict[str, object]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class VoiceLiveSpineReport:
    status: VoiceLiveSpineStatus
    message: str
    checks: dict[str, bool]
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def healthy(self) -> bool:
        return self.status == VoiceLiveSpineStatus.HEALTHY

    def to_metadata(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "message": self.message,
            "checks": self.checks,
            "created_at": self.created_at.isoformat(),
            **self.metadata,
        }


class VoiceLiveSpineMonitor:
    """
    Inspect-only monitor for the voice spine.

    It watches the loop invariants that make JARVIS feel alive: sane FSM,
    perception evidence, speaking/listening consistency, and loop progress.
    """

    def inspect(self, snapshot: VoiceLiveSpineSnapshot) -> VoiceLiveSpineReport:
        metadata = snapshot.metadata
        fsm_violations = _metadata_int(metadata, "fsm_violations")
        perception = metadata.get("perception")
        playback_status = str(metadata.get("playback_status") or "")

        checks = {
            "running_state_consistent": _running_state_consistent(snapshot),
            "fsm_clean": fsm_violations == 0,
            "perception_available": isinstance(perception, dict),
            "speaking_state_consistent": _speaking_state_consistent(
                snapshot,
                playback_status=playback_status,
            ),
            "responses_not_ahead_of_transcripts": (
                snapshot.responses <= snapshot.final_transcripts
            ),
        }

        if snapshot.status == VoiceSessionLoopStatus.FAILED:
            status = VoiceLiveSpineStatus.FAILED
            message = "voice spine failed"
        elif snapshot.status == VoiceSessionLoopStatus.DEGRADED:
            status = VoiceLiveSpineStatus.DEGRADED
            message = "voice spine degraded"
        elif not all(checks.values()):
            status = VoiceLiveSpineStatus.ATTENTION
            message = "voice spine needs attention"
        else:
            status = VoiceLiveSpineStatus.HEALTHY
            message = "voice spine healthy"

        return VoiceLiveSpineReport(
            status=status,
            message=message,
            checks=checks,
            created_at=datetime.now().astimezone(),
            metadata={
                "fsm_violations": fsm_violations,
                "perception_intent_state": metadata.get("perception_intent_state"),
                "perception_stability": metadata.get("perception_stability"),
                "playback_status": playback_status or None,
                "interruptions": snapshot.interruptions,
            },
        )


def _running_state_consistent(snapshot: VoiceLiveSpineSnapshot) -> bool:
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
    snapshot: VoiceLiveSpineSnapshot,
    *,
    playback_status: str,
) -> bool:
    if snapshot.assistant_speaking:
        return snapshot.status in {
            VoiceSessionLoopStatus.SPEAKING,
            VoiceSessionLoopStatus.USER_SPEAKING,
            VoiceSessionLoopStatus.INTERRUPTED,
        }
    if playback_status == "playing":
        return False
    return True


def _metadata_int(metadata: dict[str, object], key: str) -> int:
    value = metadata.get(key)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0
