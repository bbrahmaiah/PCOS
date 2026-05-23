from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.cognition import (  # noqa: E402
    CognitionRequest,
    CognitionRuntimePolicy,
    LocalLLMAdapter,
    OllamaBackendConfig,
    OllamaLocalLLMBackend,
    SpokenResponseStyle,
    StreamingTokenPipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test real local LLM cognition through Ollama.",
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
        "--prompt",
        default="Say one short sentence confirming JARVIS cognition is online.",
        help="Prompt to send to the local model.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming token pipeline.",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=128,
        help="Maximum generated tokens.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    backend = OllamaLocalLLMBackend(
        config=OllamaBackendConfig(
            base_url=args.base_url,
            model=args.model,
            timeout_seconds=args.timeout,
            num_predict=args.num_predict,
        )
    )
    adapter = LocalLLMAdapter(backend=backend)

    request = CognitionRequest(
        request_id="real-local-llm-smoke",
        text=args.prompt,
        policy=CognitionRuntimePolicy(
            streaming_enabled=args.stream,
            allow_tools=False,
            allow_memory_lookup=False,
            spoken_style=SpokenResponseStyle.CONCISE,
            timeout_ms=int(args.timeout * 1000),
        ),
    )

    print()
    print("JARVIS Real Local LLM Smoke")
    print("---------------------------")
    print(f"Base URL: {args.base_url}")
    print(f"Model: {args.model}")
    print(f"Streaming: {args.stream}")
    print()

    if args.stream:
        pipeline = StreamingTokenPipeline(adapter=adapter)
        stream_result = pipeline.stream_request(request)

        print(f"Passed: {stream_result.completed}")
        print(f"State: {stream_result.state.value}")
        print(f"Tokens: {len(stream_result.tokens)}")
        print(f"Speech chunks: {len(stream_result.speech_chunks)}")

        if stream_result.response is not None:
            print()
            print("Response:")
            print(stream_result.response.text)

        if stream_result.failure is not None:
            print()
            print("Failure:")
            print(stream_result.failure.message)

        return 0 if stream_result.completed else 1

    adapter_result = adapter.generate(request)

    print(f"Passed: {adapter_result.succeeded}")

    if adapter_result.response is not None:
        print()
        print("Response:")
        print(adapter_result.response.text)

    if adapter_result.failure is not None:
        print()
        print("Failure:")
        print(adapter_result.failure.message)

    return 0 if adapter_result.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())