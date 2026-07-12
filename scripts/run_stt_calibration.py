from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.voice import (  # noqa: E402
    PyAudioMicrophoneAdapter,
    VoiceInputFrame,
    VoiceMicrophoneCaptureRuntime,
    VoiceRuntimeConfig,
    VoiceSTTCalibrationPolicy,
    VoiceSTTCalibrationReport,
    VoiceSTTCalibrationRuntime,
    VoiceSTTCalibrationSample,
    VoiceSTTCalibrationScenarioKind,
    VoiceSTTPolicy,
    VoiceSTTRuntime,
    build_voice_capture_profile,
    save_voice_capture_profile,
)


def main() -> int:
    config = VoiceRuntimeConfig(
        user_label="Tony",
        assistant_name="JARVIS",
        wake_word="jarvis",
    )
    microphone = VoiceMicrophoneCaptureRuntime(
        config=config,
        adapter=PyAudioMicrophoneAdapter(
            device_index=_env_int("JARVIS_MIC_DEVICE_INDEX"),
        ),
    )
    stt = VoiceSTTRuntime(
        config=config,
        policy=VoiceSTTPolicy(
            partial_model_name=_env_text("JARVIS_STT_PARTIAL_MODEL", "base.en"),
            final_model_name=_env_text("JARVIS_STT_FINAL_MODEL", "small"),
            partial_beam_size=_env_int_default("JARVIS_STT_PARTIAL_BEAM_SIZE", 1),
            final_beam_size=_env_int_default("JARVIS_STT_FINAL_BEAM_SIZE", 1),
            min_final_confidence=_env_float_default(
                "JARVIS_STT_MIN_FINAL_CONFIDENCE",
                0.70,
            ),
            prewarm_on_prepare=_env_bool("JARVIS_STT_PREWARM_ON_START", True),
        ),
    )
    runtime = VoiceSTTCalibrationRuntime(
        stt=stt,
        policy=VoiceSTTCalibrationPolicy(
            min_wake_confidence=_env_float_default(
                "JARVIS_CALIBRATION_MIN_WAKE_CONFIDENCE",
                0.70,
            ),
            min_wake_acceptance_rate=_env_float_default(
                "JARVIS_CALIBRATION_MIN_WAKE_RATE",
                0.90,
            ),
            max_false_transcripts=_env_int_default(
                "JARVIS_CALIBRATION_MAX_FALSE_TRANSCRIPTS",
                0,
            ),
        ),
    )

    silence_seconds = _env_float_default("JARVIS_CALIBRATION_SILENCE_SECONDS", 30.0)
    noise_seconds = _env_float_default("JARVIS_CALIBRATION_NOISE_SECONDS", 10.0)
    wake_attempts = _env_int_default("JARVIS_CALIBRATION_WAKE_ATTEMPTS", 5)
    wake_capture_seconds = _env_float_default(
        "JARVIS_CALIBRATION_WAKE_CAPTURE_SECONDS",
        3.0,
    )

    print("[JARVIS_STT_CALIBRATION] status=starting", flush=True)
    samples: list[VoiceSTTCalibrationSample] = []
    microphone.start()
    try:
        print(
            "[JARVIS_STT_CALIBRATION] "
            f"scenario=silence seconds={silence_seconds:.1f} "
            "instruction='stay silent'",
            flush=True,
        )
        samples.append(
            VoiceSTTCalibrationSample(
                kind=VoiceSTTCalibrationScenarioKind.SILENCE,
                label="room_silence",
                frames=_capture_frames(microphone, config, silence_seconds),
            )
        )

        print(
            "[JARVIS_STT_CALIBRATION] "
            f"scenario=noise seconds={noise_seconds:.1f} "
            "instruction='normal room noise, do not address jarvis'",
            flush=True,
        )
        samples.append(
            VoiceSTTCalibrationSample(
                kind=VoiceSTTCalibrationScenarioKind.NOISE,
                label="room_noise",
                frames=_capture_frames(microphone, config, noise_seconds),
            )
        )

        for attempt in range(1, wake_attempts + 1):
            input(
                "[JARVIS_STT_CALIBRATION] "
                f"wake_attempt={attempt}/{wake_attempts} "
                "press Enter, then say 'Jarvis' clearly..."
            )
            samples.append(
                VoiceSTTCalibrationSample(
                    kind=VoiceSTTCalibrationScenarioKind.WAKE_WORD,
                    label=f"wake_word_{attempt}",
                    frames=_capture_frames(
                        microphone,
                        config,
                        wake_capture_seconds,
                    ),
                )
            )
    finally:
        microphone.stop()

    report = runtime.run_samples(tuple(samples))
    _print_report(report)
    profile_path = Path(
        _env_text(
            "JARVIS_VOICE_PROFILE_PATH",
            str(PROJECT_ROOT / "config" / "voice_profile.json"),
        )
    )
    profile = build_voice_capture_profile(report)
    save_voice_capture_profile(profile, profile_path)
    print(
        "[JARVIS_STT_CALIBRATION] "
        f"profile=saved path={profile_path} "
        f"vad_min_energy={profile.vad_min_energy:.1f} "
        f"vad_ratio={profile.vad_speech_start_ratio:.1f} "
        f"final_confidence={profile.stt_min_final_confidence:.2f}",
        flush=True,
    )
    return 0 if report.passed else 1


def _capture_frames(
    microphone: VoiceMicrophoneCaptureRuntime,
    config: VoiceRuntimeConfig,
    seconds: float,
) -> tuple[VoiceInputFrame, ...]:
    frame_count = max(
        1,
        int(seconds * 1000 / config.frame_duration_ms),
    )
    frames = []
    for _ in range(frame_count):
        result = microphone.capture_once()
        if result.frame is None:
            raise RuntimeError(f"microphone capture failed: {result.message}")
        frames.append(result.frame)
        time.sleep(config.frame_duration_ms / 1000)
    return tuple(frames)


def _print_report(report: VoiceSTTCalibrationReport) -> None:
    status = report.status.value
    print(
        "[JARVIS_STT_CALIBRATION] "
        f"status={status} "
        f"false_transcripts={report.false_transcripts} "
        f"wake_passes={report.wake_passes} "
        f"wake_attempts={report.wake_attempts} "
        f"wake_rate={report.wake_acceptance_rate:.2f}",
        flush=True,
    )
    for result in report.sample_results:
        text = result.transcript_text or ""
        reason = result.rejection_reason or ""
        print(
            "[JARVIS_STT_CALIBRATION_SAMPLE] "
            f"label={result.sample.label!r} "
            f"kind={result.sample.kind.value} "
            f"passed={result.passed} "
            f"confidence={result.confidence} "
            f"rejection_reason={reason!r} "
            f"text={text!r} "
            f"message={result.message!r}",
            flush=True,
        )


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    return int(raw.strip())


def _env_int_default(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw.strip())


def _env_float_default(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw.strip())


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be true/false")


def _env_text(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


if __name__ == "__main__":
    raise SystemExit(main())
