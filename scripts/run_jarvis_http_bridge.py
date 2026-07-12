from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.runtime.http_bridge import (  # noqa: E402
    JarvisHttpBridgeConfig,
    create_jarvis_http_server,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the local HTTP bridge for the real JARVIS runtime.",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("JARVIS_HTTP_HOST", "127.0.0.1"),
        help="Bridge host. Defaults to 127.0.0.1.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("JARVIS_HTTP_PORT", "8765")),
        help="Bridge port. Defaults to 8765.",
    )
    parser.add_argument(
        "--ollama-base-url",
        default=os.getenv("JARVIS_OLLAMA_BASE_URL", "http://localhost:11434"),
        help="Ollama base URL.",
    )
    parser.add_argument(
        "--ollama-model",
        default=os.getenv("JARVIS_OLLAMA_MODEL", "llama3.2:3b"),
        help="Ollama model name.",
    )
    args = parser.parse_args(argv)

    config = JarvisHttpBridgeConfig(
        host=args.host,
        port=args.port,
        ollama_base_url=args.ollama_base_url,
        ollama_model=args.ollama_model,
    )
    server = create_jarvis_http_server(config=config)

    print(
        "[JARVIS_HTTP_READY] "
        f"http://{config.host}:{config.port} "
        f"model={config.ollama_model} fakeFallback=false"
    )
    try:
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        print("\n[JARVIS_HTTP_STOPPING]")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
