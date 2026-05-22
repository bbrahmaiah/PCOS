from __future__ import annotations

# ruff: noqa: E402
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.presence.full_voice_smoke import FullVoiceSmokeHarness
from jarvis.presence.voice_tuning import (
    VoiceRuntimePreset,
    build_full_voice_smoke_config,
    build_stt_adapter,
    build_vad_adapter,
    format_full_voice_report,
    get_voice_runtime_profile,
    warm_stt_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a tuned real-time JARVIS full voice smoke test."
    )
    parser.add_argument(
        "--preset",
        choices=tuple(preset.value for preset in VoiceRuntimePreset),
        default=VoiceRuntimePreset.FAST.value,
        help="Voice runtime tuning preset.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Override max duration in seconds.",
    )
    parser.add_argument(
        "--wake-required",
        action="store_true",
        help="Require wake detection. Off by default for smoke tests.",
    )
    parser.add_argument(
        "--keep-listening",
        action="store_true",
        help="Continue after the first full voice turn.",
    )
    parser.add_argument(
        "--response",
        type=str,
        default=None,
        help="Override the fixed safe response.",
    )
    parser.add_argument(
        "--no-preload-stt",
        action="store_true",
        help="Do not warm STT before microphone capture.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = get_voice_runtime_profile(args.preset)

    config = build_full_voice_smoke_config(
        profile,
        require_wake=args.wake_required,
        keep_listening=args.keep_listening,
        duration_seconds=args.duration,
        response_text=args.response,
    )
    vad = build_vad_adapter(profile)
    stt = build_stt_adapter(profile)

    print()
    print("JARVIS Voice Smoke Runtime")
    print("--------------------------")
    print(f"Preset: {profile.preset.value}")
    print(f"STT model: {profile.stt_model}")
    print(f"VAD threshold: {profile.vad_threshold}")
    print(f"Speech start frames: {profile.speech_start_frames}")
    print(f"Speech end frames: {profile.speech_end_frames}")

    if not args.no_preload_stt:
        print()
        print("Preloading STT model before microphone capture...")
        warmup = warm_stt_model(stt)

        if not warmup.completed:
            print(f"STT preload failed: {warmup.error}")
            return 1

        print("STT model ready.")

    print()
    print("Start speaking after microphone starts.")

    report = FullVoiceSmokeHarness(
        config=config,
        vad=vad,
        stt=stt,
    ).run()

    print(format_full_voice_report(report))

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())