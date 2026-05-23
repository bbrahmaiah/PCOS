from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.cognition import (  # noqa: E402
    LocalLLMAdapter,
    OllamaBackendConfig,
    OllamaLocalLLMBackend,
    SpokenResponseStyle,
    VoiceCognitionIO,
    VoiceCognitionPlaybackResult,
    VoiceCognitionSmokeConfig,
    VoiceCognitionSmokeRunner,
    VoiceCognitionTranscript,
    create_cognition_runtime,
)


class TextVoiceCognitionIO:
    """
    Text fallback I/O for validating cognition without microphone.
    """

    def __init__(self, *, text: str) -> None:
        self._text = text

    @property
    def name(self) -> str:
        return "text_voice_cognition_io"

    def listen_once(self) -> VoiceCognitionTranscript:
        return VoiceCognitionTranscript(
            text=self._text,
            confidence=1.0,
            source="text",
        )

    def speak(self, text: str) -> VoiceCognitionPlaybackResult:
        print()
        print("JARVIS:")
        print(text)

        return VoiceCognitionPlaybackResult(
            text=text,
            started=True,
            completed=True,
            metadata={
                "mode": "text",
            },
        )


class RealLocalVoiceCognitionIO:
    """
    Minimal real microphone/STT/TTS I/O for Phase 3 smoke.

    This script-level implementation keeps real audio out of cognition modules.
    Cognition still receives only text and returns only text.
    """

    def __init__(
        self,
        *,
        listen_seconds: float,
        sample_rate: int,
        stt_model: str,
    ) -> None:
        self._listen_seconds = listen_seconds
        self._sample_rate = sample_rate
        self._stt_model = stt_model

    @property
    def name(self) -> str:
        return "real_local_voice_cognition_io"

    def listen_once(self) -> VoiceCognitionTranscript:
        sounddevice = importlib.import_module("sounddevice")
        whisper_module = importlib.import_module("faster_whisper")
        whisper_model_class: Any = whisper_module.WhisperModel

        print()
        print(f"Listening for {self._listen_seconds:.1f} seconds...")
        print("Speak now.")

        frame_count = int(self._listen_seconds * self._sample_rate)
        audio = sounddevice.rec(
            frame_count,
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
        )
        sounddevice.wait()

        print("Transcribing...")

        model = whisper_model_class(
            self._stt_model,
            device="cpu",
            compute_type="int8",
        )
        segments, info = model.transcribe(
            audio.reshape(-1),
            language="en",
            beam_size=1,
            vad_filter=True,
        )

        text = " ".join(segment.text.strip() for segment in segments).strip()
        confidence = 0.0 if not text else 1.0

        language = getattr(info, "language", None)

        return VoiceCognitionTranscript(
            text=text,
            confidence=confidence,
            source="real_microphone_stt",
            metadata={
                "language": language,
                "sample_rate": self._sample_rate,
                "listen_seconds": self._listen_seconds,
                "stt_model": self._stt_model,
            },
        )

    def speak(self, text: str) -> VoiceCognitionPlaybackResult:
        pyttsx3 = importlib.import_module("pyttsx3")

        print()
        print("JARVIS:")
        print(text)

        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()

        return VoiceCognitionPlaybackResult(
            text=text,
            started=True,
            completed=True,
            metadata={
                "mode": "pyttsx3",
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test real voice + cognition runtime.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:11434",
        help="Ollama base URL.",
    )
    parser.add_argument(
        "--model",
        default="llama3.2:3b",
        help="Ollama model name.",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="Text fallback instead of microphone.",
    )
    parser.add_argument(
        "--real-voice",
        action="store_true",
        help="Use real microphone, STT, and TTS.",
    )
    parser.add_argument(
        "--listen-seconds",
        type=float,
        default=5.0,
        help="Microphone capture duration.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16_000,
        help="Microphone sample rate.",
    )
    parser.add_argument(
        "--stt-model",
        default="tiny",
        help="faster-whisper model size.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use cognition streaming pipeline.",
    )
    parser.add_argument(
        "--allow-tools",
        action="store_true",
        help="Allow tool planning only. No tool execution happens.",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=160,
        help="Maximum generated tokens.",
    )

    return parser.parse_args()


def build_voice_io(args: argparse.Namespace) -> VoiceCognitionIO:
    if args.real_voice:
        return RealLocalVoiceCognitionIO(
            listen_seconds=args.listen_seconds,
            sample_rate=args.sample_rate,
            stt_model=args.stt_model,
        )

    text = args.text or "Hello Jarvis, confirm cognition is online."

    return TextVoiceCognitionIO(text=text)


def main() -> int:
    args = parse_args()

    backend = OllamaLocalLLMBackend(
        config=OllamaBackendConfig(
            base_url=args.base_url,
            model=args.model,
            num_predict=args.num_predict,
        )
    )
    adapter = LocalLLMAdapter(backend=backend)
    runtime = create_cognition_runtime(adapter=adapter)

    runner = VoiceCognitionSmokeRunner(
        runtime=runtime,
        voice_io=build_voice_io(args),
        config=VoiceCognitionSmokeConfig(
            streaming=args.stream,
            allow_tools=args.allow_tools,
            allow_memory_lookup=True,
            spoken_style=SpokenResponseStyle.CONCISE,
        ),
    )

    print()
    print("JARVIS Real Voice + Cognition Smoke")
    print("-----------------------------------")
    print(f"Model: {args.model}")
    print(f"Streaming: {args.stream}")
    print(f"Real voice: {args.real_voice}")
    print(f"Tool planning only: {args.allow_tools}")
    print("Direct laptop execution: disabled")
    print()

    report = runner.run_once()

    print()
    print("Result")
    print("------")
    print(f"Passed: {report.passed}")
    print(f"Heard: {report.heard_text}")
    print(f"Response: {report.response_text}")

    if report.runtime_result is not None:
        print(f"Action plan created: {report.runtime_result.action_plan is not None}")

    if report.reason is not None:
        print(f"Reason: {report.reason}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())