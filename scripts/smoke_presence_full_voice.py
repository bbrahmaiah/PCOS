from __future__ import annotations

# ruff: noqa: E402
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.presence.adapters import (
    EnergyVoiceActivityAdapter,
    EnergyVoiceActivityConfig,
    RealSpeechToTextAdapter,
    RealSpeechToTextConfig,
)
from jarvis.presence.full_voice_smoke import (
    FullVoiceSmokeConfig,
    FullVoiceSmokeHarness,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a controlled real microphone → wake/VAD/STT → "
            "fixed response → TTS/playback smoke test."
        )
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=20.0,
        help="Maximum smoke test duration in seconds.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=4_000,
        help="Maximum microphone frames to process.",
    )
    parser.add_argument(
        "--no-wake-required",
        action="store_true",
        help="Skip wake requirement and allow VAD/STT immediately.",
    )
    parser.add_argument(
        "--keep-listening",
        action="store_true",
        help="Do not stop after the first full voice turn.",
    )
    parser.add_argument(
        "--response",
        type=str,
        default="Yes sir. I heard you.",
        help="Fixed safe response to synthesize and play.",
    )
    parser.add_argument(
        "--stt-model",
        type=str,
        default="tiny",
        choices=("tiny", "base", "small", "medium", "large-v3"),
        help="faster-whisper model size. Use tiny/base for fast smoke tests.",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=60.0,
        help="RMS threshold for speech start. Lower if speech is not detected.",
    )
    parser.add_argument(
        "--vad-silence-threshold",
        type=float,
        default=30.0,
        help="Initial silence/noise threshold.",
    )
    parser.add_argument(
        "--speech-start-frames",
        type=int,
        default=1,
        help="Consecutive speech-like frames required to start speech.",
    )
    parser.add_argument(
        "--speech-end-frames",
        type=int,
        default=6,
        help="Consecutive silence frames required to end speech.",
    )
    parser.add_argument(
        "--adaptive-vad",
        action="store_true",
        help="Enable adaptive noise floor. Keep off for first smoke tests.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = FullVoiceSmokeConfig(
        duration_seconds=args.duration,
        max_frames=args.max_frames,
        require_wake=not args.no_wake_required,
        stop_after_first_turn=not args.keep_listening,
        response_text=args.response,
    )

    vad = EnergyVoiceActivityAdapter(
        config=EnergyVoiceActivityConfig(
            speech_rms_threshold=args.vad_threshold,
            silence_rms_threshold=args.vad_silence_threshold,
            speech_start_frames=args.speech_start_frames,
            speech_end_frames=args.speech_end_frames,
            min_zero_crossing_rate=0.0,
            max_zero_crossing_rate=1.0,
            adaptive_noise_floor=args.adaptive_vad,
        )
    )

    stt = RealSpeechToTextAdapter(
        config=RealSpeechToTextConfig(
            model_size=args.stt_model,
            device="cpu",
            compute_type="int8",
            language="en",
        )
    )

    report = FullVoiceSmokeHarness(
        config=config,
        vad=vad,
        stt=stt,
    ).run()

    print()
    print("JARVIS Full Voice Smoke")
    print("-----------------------")
    print(f"Passed: {report.passed}")
    print(f"Duration: {report.duration_ms:.2f} ms")
    print(f"Frames read: {report.frames_read}")
    print(f"Wake detected: {report.wake_detected}")
    print(f"Speech completed: {report.speech_completed}")
    print(f"Turns: {report.turn_count}")
    print(f"Playback results: {report.playback_count}")

    for index, turn in enumerate(report.turns, start=1):
        print()
        print(f"Turn {index}")
        print(f"  heard: {turn.transcript.text}")
        print(f"  response: {turn.response_text}")
        print(f"  chunks: {len(turn.chunks)}")
        for result in turn.playback_results:
            print(f"  playback: {result.status.value}")

    if report.errors:
        print()
        print("Errors:")
        for error in report.errors:
            print(f" - {error}")

    if not report.speech_completed and not report.errors:
        print()
        print("Tuning hint:")
        print(" - Speak clearly after sounddevice_microphone_started appears.")
        print(" - Move closer to the microphone.")
        print(" - Try: --vad-threshold 30 --speech-end-frames 4")
        print(" - Try longer duration: --duration 30")

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())