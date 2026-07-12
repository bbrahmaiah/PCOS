from __future__ import annotations

from jarvis.voice import (
    VoiceRuntimeIntegrityCheckKind,
    VoiceRuntimeIntegrityGuard,
    VoiceRuntimeIntegrityInput,
    VoiceRuntimeIntegrityReport,
    VoiceRuntimeIntegrityStatus,
    VoiceSessionLoopEvent,
    VoiceSessionLoopSnapshot,
    VoiceSessionLoopStatus,
    utc_now,
)


def _snapshot(
    *,
    status: VoiceSessionLoopStatus = VoiceSessionLoopStatus.LISTENING,
    running: bool = True,
    assistant_speaking: bool = False,
    fsm_violations: int = 0,
    playback_status: str | None = None,
) -> VoiceSessionLoopSnapshot:
    metadata: dict[str, object] = {"fsm_violations": fsm_violations}
    if playback_status is not None:
        metadata["playback_status"] = playback_status
    return VoiceSessionLoopSnapshot(
        status=status,
        running=running,
        assistant_speaking=assistant_speaking,
        cycles=1,
        captured_frames=1,
        speech_segments=0,
        partial_transcripts=0,
        final_transcripts=0,
        responses=0,
        tts_outputs=0,
        played_outputs=0,
        interruptions=0,
        recoveries=0,
        consecutive_failures=0,
        buffered_segment_frames=0,
        last_event=VoiceSessionLoopEvent.FRAME_CAPTURED,
        last_transcript_text=None,
        last_response_text=None,
        last_latency_ms=1.0,
        last_error=None,
        created_at=utc_now(),
        metadata=metadata,
    )


def _failed_kinds(
    report: VoiceRuntimeIntegrityReport,
) -> set[VoiceRuntimeIntegrityCheckKind]:
    return {check.kind for check in report.checks if not check.passed}


def test_runtime_integrity_passes_pure_generated_response_path() -> None:
    report = VoiceRuntimeIntegrityGuard().inspect(
        VoiceRuntimeIntegrityInput(
            snapshot=_snapshot(),
            response_origin="cognition_response_boundary",
            uses_generated_response=True,
            uses_fixed_response=False,
            fake_fallback_enabled=False,
        )
    )

    assert report.status == VoiceRuntimeIntegrityStatus.PASSED
    assert report.failed_checks == ()


def test_runtime_integrity_allows_silent_operational_control() -> None:
    report = VoiceRuntimeIntegrityGuard().inspect(
        VoiceRuntimeIntegrityInput(
            snapshot=_snapshot(),
            response_origin=None,
            uses_generated_response=False,
            uses_fixed_response=False,
            fake_fallback_enabled=False,
        )
    )

    assert report.status == VoiceRuntimeIntegrityStatus.PASSED


def test_runtime_integrity_fails_fixed_or_deterministic_response() -> None:
    report = VoiceRuntimeIntegrityGuard().inspect(
        VoiceRuntimeIntegrityInput(
            snapshot=_snapshot(),
            response_origin="voice_reflex_operational",
            uses_generated_response=False,
            uses_fixed_response=True,
            deterministic_system_response=True,
        )
    )

    assert report.status == VoiceRuntimeIntegrityStatus.FAILED
    assert VoiceRuntimeIntegrityCheckKind.NO_FIXED_RESPONSE in _failed_kinds(report)
    assert VoiceRuntimeIntegrityCheckKind.RESPONSE_ORIGIN in _failed_kinds(report)


def test_runtime_integrity_fails_fake_fallback() -> None:
    report = VoiceRuntimeIntegrityGuard().inspect(
        VoiceRuntimeIntegrityInput(
            snapshot=_snapshot(),
            fake_fallback_enabled=True,
        )
    )

    assert report.status == VoiceRuntimeIntegrityStatus.FAILED
    assert VoiceRuntimeIntegrityCheckKind.NO_FAKE_FALLBACK in _failed_kinds(report)


def test_runtime_integrity_fails_fsm_violation() -> None:
    report = VoiceRuntimeIntegrityGuard().inspect(
        VoiceRuntimeIntegrityInput(
            snapshot=_snapshot(fsm_violations=1),
        )
    )

    assert report.status == VoiceRuntimeIntegrityStatus.FAILED
    assert VoiceRuntimeIntegrityCheckKind.FSM_CLEAN in _failed_kinds(report)


def test_runtime_integrity_fails_illegal_speaking_overlap() -> None:
    report = VoiceRuntimeIntegrityGuard().inspect(
        VoiceRuntimeIntegrityInput(
            snapshot=_snapshot(
                status=VoiceSessionLoopStatus.LISTENING,
                assistant_speaking=True,
            ),
        )
    )

    assert report.status == VoiceRuntimeIntegrityStatus.FAILED
    assert VoiceRuntimeIntegrityCheckKind.SPEAKING_STATE in _failed_kinds(report)


def test_runtime_integrity_fails_playback_without_speaking_state() -> None:
    report = VoiceRuntimeIntegrityGuard().inspect(
        VoiceRuntimeIntegrityInput(
            snapshot=_snapshot(
                status=VoiceSessionLoopStatus.LISTENING,
                assistant_speaking=False,
                playback_status="playing",
            ),
        )
    )

    assert report.status == VoiceRuntimeIntegrityStatus.FAILED
    assert VoiceRuntimeIntegrityCheckKind.SPEAKING_STATE in _failed_kinds(report)


def test_runtime_integrity_fails_running_status_mismatch() -> None:
    report = VoiceRuntimeIntegrityGuard().inspect(
        VoiceRuntimeIntegrityInput(
            snapshot=_snapshot(
                status=VoiceSessionLoopStatus.STOPPED,
                running=True,
            ),
        )
    )

    assert report.status == VoiceRuntimeIntegrityStatus.FAILED
    assert VoiceRuntimeIntegrityCheckKind.RUNNING_STATE in _failed_kinds(report)


def test_runtime_integrity_fails_crash_boundary() -> None:
    report = VoiceRuntimeIntegrityGuard().inspect(
        VoiceRuntimeIntegrityInput(
            snapshot=_snapshot(
                status=VoiceSessionLoopStatus.FAILED,
                running=False,
            ),
        )
    )

    assert report.status == VoiceRuntimeIntegrityStatus.FAILED
    assert VoiceRuntimeIntegrityCheckKind.CRASH_BOUNDARY in _failed_kinds(report)
