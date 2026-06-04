from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.live import (  # noqa: E402
    LiveSessionConfig,
    LiveSessionMode,
    LiveSessionRunner,
    LiveSessionRunnerConfig,
    LiveSessionRunnerStatus,
    LiveShutdownReason,
    OllamaGeneratorConfig,
    OllamaLiveResponseGenerator,
)

CONFIG_PATH = PROJECT_ROOT / "config" / "live_jarvis.json"


def main() -> int:
    config_data = _load_config(CONFIG_PATH)
    runner = _build_runner(config_data)

    start = runner.start()
    print(f"[JARVIS] runner={start.status.value} reason={start.reason}")

    if start.status == LiveSessionRunnerStatus.FAILED:
        return 1

    print("[JARVIS] Live runtime started.")
    print("[JARVIS] Type 'Jarvis ...' to engage.")
    print("[JARVIS] Type '/interrupt wait' to simulate interruption.")
    print("[JARVIS] Type '/health' to check health.")
    print("[JARVIS] Type '/shutdown' to stop.")

    while True:
        try:
            user_text = input("Balu> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            _shutdown(runner)
            return 0

        if not user_text:
            continue

        if user_text.casefold() in {"/shutdown", "/quit", "/exit"}:
            _shutdown(runner)
            return 0

        if user_text.casefold() == "/health":
            health = runner.check_health()
            status = (
                health.health_result.status.value
                if health.health_result is not None
                else "unknown"
            )
            print(f"[JARVIS] health={status} runner={health.status.value}")
            continue

        if user_text.casefold().startswith("/interrupt"):
            interrupt_text = user_text.removeprefix("/interrupt").strip()
            if not interrupt_text:
                interrupt_text = "wait"
            result = runner.handle_interrupt(text=interrupt_text)
            interruption_reason = (
                result.interruption_result.reason
                if result.interruption_result is not None
                else result.reason
            )
            print(f"[JARVIS] interruption={interruption_reason}")
            continue

        result = runner.ingest_text(
            text=user_text,
            speech_probability=0.95,
            confidence=0.95,
            metadata={
                "launcher": "run_live_jarvis",
                "mode": "text_daily_driver",
            },
        )

        if result.dialogue_result is None:
            print("[JARVIS] ignored")
            continue

        turn = result.dialogue_result.turn
        response = turn.response if turn is not None else None

        if response is None:
            print(f"[JARVIS] blocked: {result.dialogue_result.reason}")
            continue

        print(f"JARVIS> {response.text}")

        # In text mode, we mark speaking finished immediately.
        runner.dialogue.finish_response()


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing live JARVIS config: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("live JARVIS config must be a JSON object.")

    return cast(dict[str, Any], data)


def _build_runner(config_data: dict[str, Any]) -> LiveSessionRunner:
    user_label = str(config_data.get("user_label", "Balu"))
    assistant_name = str(config_data.get("assistant_name", "JARVIS"))
    wake_word = str(config_data.get("wake_word", "jarvis"))

    generator = OllamaLiveResponseGenerator(
        OllamaGeneratorConfig(
            base_url=str(
                config_data.get(
                    "ollama_base_url",
                    "http://localhost:11434",
                )
            ),
            model=str(config_data.get("ollama_model", "llama3.2:3b")),
            timeout_seconds=int(config_data.get("timeout_seconds", 60)),
            max_sentences=int(config_data.get("max_sentences", 3)),
        )
    )

    session_config = LiveSessionConfig(
        mode=LiveSessionMode.REAL_VOICE,
        user_label=user_label,
        assistant_name=assistant_name,
        wake_word=wake_word,
        real_microphone_enabled=True,
        real_stt_enabled=True,
        real_tts_enabled=True,
    )

    return LiveSessionRunner(
        config=LiveSessionRunnerConfig(
            session_config=session_config,
            auto_prepare_audio=False,
            auto_health_check=False,
            auto_recover=False,
        ),
        response_generator=generator,
    )


def _shutdown(runner: LiveSessionRunner) -> None:
    result = runner.shutdown(reason=LiveShutdownReason.USER_REQUEST)
    print(f"[JARVIS] shutdown={result.status.value} reason={result.reason}")


if __name__ == "__main__":
    raise SystemExit(main())