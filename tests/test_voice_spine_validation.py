from __future__ import annotations

from dataclasses import dataclass

from jarvis.voice import (
    DEFAULT_TEN_TURN_UTTERANCES,
    VoiceSessionLoopEvent,
    VoiceSessionLoopOperation,
    VoiceSessionLoopResult,
    VoiceSessionLoopSnapshot,
    VoiceSessionLoopStatus,
    VoiceSpineValidationPolicy,
    VoiceSpineValidationRuntime,
    VoiceSpineValidationStatus,
    VoiceTranscript,
    utc_now,
)


def _result(
    *,
    status: VoiceSessionLoopStatus = VoiceSessionLoopStatus.LISTENING,
    event: VoiceSessionLoopEvent | None = VoiceSessionLoopEvent.PLAYBACK_FINISHED,
    message: str = "ok",
) -> VoiceSessionLoopResult:
    return VoiceSessionLoopResult(
        status=status,
        operation=VoiceSessionLoopOperation.HANDLE_TRANSCRIPT,
        event=event,
        transcript=None,
        cognition_result=None,
        tts_result=None,
        playback_result=None,
        barge_in_result=None,
        health_result=None,
        message=message,
        latency_ms=1.0,
        created_at=utc_now(),
    )


@dataclass
class FakeValidationSession:
    fail_turn: int | None = None
    started: bool = False
    responses: int = 0
    played_outputs: int = 0
    interruptions: int = 0
    current_status: VoiceSessionLoopStatus = VoiceSessionLoopStatus.CREATED
    assistant_speaking: bool = False

    def start(self) -> VoiceSessionLoopResult:
        self.started = True
        self.current_status = VoiceSessionLoopStatus.LISTENING
        return _result(
            status=VoiceSessionLoopStatus.LISTENING,
            event=VoiceSessionLoopEvent.STARTED,
            message="started",
        )

    def handle_transcript(self, transcript: VoiceTranscript) -> VoiceSessionLoopResult:
        del transcript
        turn = self.responses + 1
        self.responses += 1

        if self.fail_turn == turn:
            self.current_status = VoiceSessionLoopStatus.SPEAKING
            self.assistant_speaking = True
            return _result(
                status=VoiceSessionLoopStatus.SPEAKING,
                event=VoiceSessionLoopEvent.RESPONSE_READY,
                message="stuck speaking",
            )

        self.played_outputs += 1
        self.current_status = VoiceSessionLoopStatus.LISTENING
        self.assistant_speaking = False
        return _result()

    def snapshot(self) -> VoiceSessionLoopSnapshot:
        return VoiceSessionLoopSnapshot(
            status=self.current_status,
            running=self.started,
            assistant_speaking=self.assistant_speaking,
            cycles=self.responses,
            captured_frames=self.responses,
            speech_segments=self.responses,
            partial_transcripts=0,
            final_transcripts=self.responses,
            responses=self.responses,
            tts_outputs=self.responses,
            played_outputs=self.played_outputs,
            interruptions=self.interruptions,
            recoveries=0,
            consecutive_failures=0,
            buffered_segment_frames=0,
            last_event=None,
            last_transcript_text=None,
            last_response_text=None,
            last_latency_ms=1.0,
            last_error=None,
            created_at=utc_now(),
            metadata={
                "perception": {"packets": self.responses},
                "fsm_violations": 0,
                "playback_status": "ready",
            },
        )


def test_voice_spine_validation_passes_ten_turn_gate() -> None:
    runtime = VoiceSpineValidationRuntime(session=FakeValidationSession())

    report = runtime.run()

    assert report.status == VoiceSpineValidationStatus.PASSED
    assert report.passed_turns == 10
    assert len(report.turns) == len(DEFAULT_TEN_TURN_UTTERANCES)
    assert report.turns[0].utterance == "jarvis can you hear me"


def test_voice_spine_validation_fails_when_turn_does_not_finish() -> None:
    runtime = VoiceSpineValidationRuntime(
        session=FakeValidationSession(fail_turn=3),
        policy=VoiceSpineValidationPolicy(
            utterances=DEFAULT_TEN_TURN_UTTERANCES,
        ),
    )

    report = runtime.run()

    assert report.status == VoiceSpineValidationStatus.FAILED
    assert report.passed_turns == 9
    assert report.turns[2].passed is False
    assert "playback_not_finished" in report.turns[2].message
    assert "loop_not_listening_after_turn" in report.turns[2].message
