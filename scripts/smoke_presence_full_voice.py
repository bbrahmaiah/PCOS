from __future__ import annotations

# ruff: noqa: E402
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
        default=15.0,
        help="Maximum smoke test duration in seconds.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=3_000,
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

    report = FullVoiceSmokeHarness(config=config).run()

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

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())