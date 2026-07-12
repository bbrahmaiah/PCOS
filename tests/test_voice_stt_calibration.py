from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from jarvis.voice import (
    VoiceInputFrame,
    VoiceInputFrameKind,
    VoiceSTTCalibrationPolicy,
    VoiceSTTCalibrationRuntime,
    VoiceSTTCalibrationSample,
    VoiceSTTCalibrationScenarioKind,
    VoiceSTTCalibrationStatus,
    VoiceSTTOperation,
    VoiceSTTResult,
    VoiceSTTRuntimeStatus,
    VoiceTranscript,
    VoiceTranscriptKind,
    build_voice_capture_profile,
    load_voice_capture_profile,
    make_voice_frame_id,
    make_voice_segment_id,
    make_voice_session_id,
    make_voice_transcript_id,
    save_voice_capture_profile,
    utc_now,
)


@dataclass
class FakeCalibrationSTT:
    results: list[VoiceSTTResult]

    def transcribe_final(
        self,
        frames: tuple[VoiceInputFrame, ...],
        *,
        allow_action_candidate: bool = False,
    ) -> VoiceSTTResult:
        del frames, allow_action_candidate
        if not self.results:
            raise RuntimeError("no fake STT result queued")
        return self.results.pop(0)


def _frame() -> VoiceInputFrame:
    return _frame_with_sample(0)


def _frame_with_sample(sample: int) -> VoiceInputFrame:
    data = int(sample).to_bytes(2, byteorder="little", signed=True) * 320
    return VoiceInputFrame(
        frame_id=make_voice_frame_id(),
        session_id=make_voice_session_id(),
        kind=VoiceInputFrameKind.PCM16_MONO,
        sample_rate_hz=16_000,
        channels=1,
        data=data,
        captured_at=utc_now(),
        duration_ms=20,
    )


def _sample(
    kind: VoiceSTTCalibrationScenarioKind,
    label: str,
) -> VoiceSTTCalibrationSample:
    return VoiceSTTCalibrationSample(
        kind=kind,
        label=label,
        frames=(_frame(),),
    )


def _stt_result(
    *,
    text: str | None,
    confidence: float = 0.92,
    rejection_reason: str | None = None,
) -> VoiceSTTResult:
    transcript = None
    if text is not None:
        transcript = VoiceTranscript(
            transcript_id=make_voice_transcript_id(),
            session_id=make_voice_session_id(),
            segment_id=make_voice_segment_id(),
            kind=VoiceTranscriptKind.FINAL,
            text=text,
            confidence=confidence,
            created_at=utc_now(),
        )
    metadata: dict[str, object] = {}
    if rejection_reason is not None:
        metadata["rejection_reason"] = rejection_reason
    return VoiceSTTResult(
        status=(
            VoiceSTTRuntimeStatus.READY
            if transcript is None
            else VoiceSTTRuntimeStatus.TRANSCRIBING
        ),
        operation=VoiceSTTOperation.TRANSCRIBE_FINAL,
        transcript=transcript,
        candidate=None,
        model=None,
        message="fake calibration result",
        created_at=utc_now(),
        metadata=metadata,
    )


def test_stt_calibration_passes_clean_silence_and_wake_word() -> None:
    runtime = VoiceSTTCalibrationRuntime(
        stt=FakeCalibrationSTT(
            [
                _stt_result(text=None, rejection_reason="high_no_speech_probability"),
                _stt_result(text="Jarvis", confidence=0.94),
            ]
        )
    )

    report = runtime.run_samples(
        (
            _sample(VoiceSTTCalibrationScenarioKind.SILENCE, "silence"),
            _sample(VoiceSTTCalibrationScenarioKind.WAKE_WORD, "wake"),
        )
    )

    assert report.status == VoiceSTTCalibrationStatus.PASSED
    assert report.false_transcripts == 0
    assert report.wake_passes == 1
    assert report.wake_acceptance_rate == 1.0
    assert report.sample_results[0].rejection_reason == "high_no_speech_probability"


def test_stt_calibration_fails_false_transcript_during_silence() -> None:
    runtime = VoiceSTTCalibrationRuntime(
        stt=FakeCalibrationSTT([_stt_result(text="thanks for watching")])
    )

    report = runtime.run_samples(
        (_sample(VoiceSTTCalibrationScenarioKind.SILENCE, "silence"),)
    )

    assert report.status == VoiceSTTCalibrationStatus.FAILED
    assert report.false_transcripts == 1
    assert report.sample_results[0].passed is False


def test_stt_calibration_fails_wake_word_below_confidence() -> None:
    runtime = VoiceSTTCalibrationRuntime(
        stt=FakeCalibrationSTT([_stt_result(text="Jarvis", confidence=0.45)]),
        policy=VoiceSTTCalibrationPolicy(min_wake_confidence=0.70),
    )

    report = runtime.run_samples(
        (_sample(VoiceSTTCalibrationScenarioKind.WAKE_WORD, "wake"),)
    )

    assert report.status == VoiceSTTCalibrationStatus.FAILED
    assert report.wake_passes == 0
    assert report.sample_results[0].message == (
        "wake word was not accepted with enough confidence"
    )


def test_stt_calibration_fails_when_wake_text_is_wrong() -> None:
    runtime = VoiceSTTCalibrationRuntime(
        stt=FakeCalibrationSTT([_stt_result(text="service")])
    )

    report = runtime.run_samples(
        (_sample(VoiceSTTCalibrationScenarioKind.WAKE_WORD, "wake"),)
    )

    assert report.status == VoiceSTTCalibrationStatus.FAILED
    assert report.wake_acceptance_rate == 0.0


def test_stt_calibration_rejects_empty_samples() -> None:
    runtime = VoiceSTTCalibrationRuntime(
        stt=FakeCalibrationSTT([]),
    )

    with pytest.raises(ValueError):
        runtime.run_samples(())


def test_stt_calibration_policy_validation() -> None:
    with pytest.raises(ValueError):
        VoiceSTTCalibrationPolicy(wake_words=())

    with pytest.raises(ValueError):
        VoiceSTTCalibrationPolicy(max_false_transcripts=-1)

    with pytest.raises(ValueError):
        VoiceSTTCalibrationPolicy(min_wake_acceptance_rate=1.5)


def test_stt_calibration_builds_and_saves_voice_capture_profile(
    tmp_path: Path,
) -> None:
    runtime = VoiceSTTCalibrationRuntime(
        stt=FakeCalibrationSTT(
            [
                _stt_result(text=None, rejection_reason="high_no_speech_probability"),
                _stt_result(text="Jarvis", confidence=0.94),
            ]
        )
    )
    silence = VoiceSTTCalibrationSample(
        kind=VoiceSTTCalibrationScenarioKind.SILENCE,
        label="silence",
        frames=(_frame_with_sample(120), _frame_with_sample(160)),
    )
    wake = VoiceSTTCalibrationSample(
        kind=VoiceSTTCalibrationScenarioKind.WAKE_WORD,
        label="wake",
        frames=(_frame_with_sample(1900), _frame_with_sample(2200)),
    )

    report = runtime.run_samples((silence, wake))
    profile = build_voice_capture_profile(report)
    profile_path = tmp_path / "voice_profile.json"
    save_voice_capture_profile(profile, profile_path)
    loaded = load_voice_capture_profile(profile_path)

    assert profile.vad_min_energy >= 450.0
    assert profile.stt_min_final_confidence == 0.70
    assert loaded is not None
    assert loaded.vad_min_energy == profile.vad_min_energy
    assert loaded.metadata["source"] == "stt_calibration"
