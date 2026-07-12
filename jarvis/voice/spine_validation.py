from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from jarvis.voice.contracts import (
    VoiceTranscript,
    VoiceTranscriptKind,
    make_voice_segment_id,
    make_voice_session_id,
    make_voice_transcript_id,
    utc_now,
)
from jarvis.voice.live_spine_monitor import (
    VoiceLiveSpineMonitor,
    VoiceLiveSpineReport,
)
from jarvis.voice.session_loop import (
    VoiceSessionLoopEvent,
    VoiceSessionLoopResult,
    VoiceSessionLoopSnapshot,
    VoiceSessionLoopStatus,
)

DEFAULT_TEN_TURN_UTTERANCES: tuple[str, ...] = (
    "jarvis can you hear me",
    "jarvis say voice spine turn one is stable",
    "jarvis say voice spine turn two is stable",
    "jarvis say voice spine turn three is stable",
    "jarvis say voice spine turn four is stable",
    "jarvis say voice spine turn five is stable",
    "jarvis say voice spine turn six is stable",
    "jarvis say voice spine turn seven is stable",
    "jarvis say voice spine turn eight is stable",
    "jarvis say voice spine turn nine is stable",
)


class VoiceSpineValidationStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


class VoiceSpineValidationSession(Protocol):
    def start(self) -> VoiceSessionLoopResult:
        raise NotImplementedError

    def handle_transcript(self, transcript: VoiceTranscript) -> VoiceSessionLoopResult:
        raise NotImplementedError

    def snapshot(self) -> VoiceSessionLoopSnapshot:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class VoiceSpineTurnReport:
    turn_index: int
    utterance: str
    passed: bool
    message: str
    session_status: VoiceSessionLoopStatus
    event: VoiceSessionLoopEvent | None
    responses: int
    played_outputs: int
    interruptions: int
    spine_report: VoiceLiveSpineReport
    metadata: dict[str, object] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, object]:
        return {
            "turn_index": self.turn_index,
            "utterance": self.utterance,
            "passed": self.passed,
            "message": self.message,
            "session_status": self.session_status.value,
            "event": None if self.event is None else self.event.value,
            "responses": self.responses,
            "played_outputs": self.played_outputs,
            "interruptions": self.interruptions,
            "spine": self.spine_report.to_metadata(),
            **self.metadata,
        }


@dataclass(frozen=True, slots=True)
class VoiceSpineValidationReport:
    status: VoiceSpineValidationStatus
    turns: tuple[VoiceSpineTurnReport, ...]
    message: str
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == VoiceSpineValidationStatus.PASSED

    @property
    def passed_turns(self) -> int:
        return sum(1 for turn in self.turns if turn.passed)

    def to_metadata(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "passed_turns": self.passed_turns,
            "total_turns": len(self.turns),
            "message": self.message,
            "turns": [turn.to_metadata() for turn in self.turns],
            **self.metadata,
        }


@dataclass(frozen=True, slots=True)
class VoiceSpineValidationPolicy:
    utterances: tuple[str, ...] = DEFAULT_TEN_TURN_UTTERANCES
    require_playback_finished: bool = True
    require_listening_after_turn: bool = True
    require_no_new_interruptions: bool = True
    require_healthy_spine: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.utterances:
            raise ValueError("utterances cannot be empty.")
        if len(self.utterances) < 10:
            raise ValueError("ten-turn validation requires at least 10 utterances.")
        if any(not utterance.strip() for utterance in self.utterances):
            raise ValueError("utterances cannot contain empty text.")


class VoiceSpineValidationRuntime:
    """
    Deterministic Phase 1 voice-spine completion gate.

    This does not pretend to be the real microphone test. It verifies the
    official turn pipeline can complete 10 spoken-turn equivalents without
    false interruption, stuck speaking state, or live spine monitor warnings.
    """

    def __init__(
        self,
        *,
        session: VoiceSpineValidationSession,
        monitor: VoiceLiveSpineMonitor | None = None,
        policy: VoiceSpineValidationPolicy | None = None,
    ) -> None:
        self._session = session
        self._monitor = monitor or VoiceLiveSpineMonitor()
        self._policy = policy or VoiceSpineValidationPolicy()

    def run(self) -> VoiceSpineValidationReport:
        start_result = self._session.start()
        if start_result.status == VoiceSessionLoopStatus.FAILED:
            return VoiceSpineValidationReport(
                status=VoiceSpineValidationStatus.FAILED,
                turns=(),
                message="voice spine validation could not start session",
                metadata={"start_message": start_result.message},
            )

        reports: list[VoiceSpineTurnReport] = []
        previous = self._session.snapshot()
        for turn_index, utterance in enumerate(self._policy.utterances, start=1):
            result = self._session.handle_transcript(_make_final_transcript(utterance))
            snapshot = self._session.snapshot()
            spine_report = self._monitor.inspect(snapshot)
            turn_report = self._evaluate_turn(
                turn_index=turn_index,
                utterance=utterance,
                result=result,
                previous=previous,
                snapshot=snapshot,
                spine_report=spine_report,
            )
            reports.append(turn_report)
            previous = snapshot

        failed = [turn for turn in reports if not turn.passed]
        if failed:
            return VoiceSpineValidationReport(
                status=VoiceSpineValidationStatus.FAILED,
                turns=tuple(reports),
                message=f"voice spine validation failed at turn {failed[0].turn_index}",
                metadata=self._policy.metadata,
            )

        return VoiceSpineValidationReport(
            status=VoiceSpineValidationStatus.PASSED,
            turns=tuple(reports),
            message="voice spine completed ten-turn validation",
            metadata=self._policy.metadata,
        )

    def _evaluate_turn(
        self,
        *,
        turn_index: int,
        utterance: str,
        result: VoiceSessionLoopResult,
        previous: VoiceSessionLoopSnapshot,
        snapshot: VoiceSessionLoopSnapshot,
        spine_report: VoiceLiveSpineReport,
    ) -> VoiceSpineTurnReport:
        failures: list[str] = []
        if self._policy.require_playback_finished and (
            result.event != VoiceSessionLoopEvent.PLAYBACK_FINISHED
        ):
            failures.append("playback_not_finished")
        if self._policy.require_listening_after_turn and (
            snapshot.status != VoiceSessionLoopStatus.LISTENING
            or snapshot.assistant_speaking
        ):
            failures.append("loop_not_listening_after_turn")
        if self._policy.require_no_new_interruptions and (
            snapshot.interruptions != previous.interruptions
        ):
            failures.append("unexpected_interruption")
        if snapshot.responses <= previous.responses:
            failures.append("response_not_recorded")
        if snapshot.played_outputs <= previous.played_outputs:
            failures.append("playback_not_recorded")
        if self._policy.require_healthy_spine and not spine_report.healthy:
            failures.append(f"spine_{spine_report.status.value}")

        passed = not failures
        return VoiceSpineTurnReport(
            turn_index=turn_index,
            utterance=utterance,
            passed=passed,
            message="passed" if passed else ",".join(failures),
            session_status=snapshot.status,
            event=result.event,
            responses=snapshot.responses,
            played_outputs=snapshot.played_outputs,
            interruptions=snapshot.interruptions,
            spine_report=spine_report,
        )


def _make_final_transcript(text: str) -> VoiceTranscript:
    return VoiceTranscript(
        transcript_id=make_voice_transcript_id(),
        session_id=make_voice_session_id(),
        segment_id=make_voice_segment_id(),
        kind=VoiceTranscriptKind.FINAL,
        text=text,
        confidence=0.99,
        created_at=utc_now(),
        metadata={"source": "voice_spine_validation"},
    )
