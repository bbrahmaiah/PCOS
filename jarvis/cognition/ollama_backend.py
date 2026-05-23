from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Protocol, runtime_checkable

from jarvis.cognition.local_llm_adapter import (
    LocalLLMBackendResult,
    LocalLLMBackendSnapshot,
    LocalLLMBackendStatus,
    LocalLLMBackendToken,
)
from jarvis.cognition.models import CognitionRequest
from jarvis.runtime.observability.structured_logger import get_logger


@runtime_checkable
class OllamaHttpClient(Protocol):
    """
    Minimal HTTP client contract for Ollama.

    Tests can provide a fake client. Production uses UrllibOllamaHttpClient.
    """

    def post_json(
        self,
        *,
        path: str,
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """POST JSON and return one JSON object."""

    def stream_json(
        self,
        *,
        path: str,
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Iterator[dict[str, Any]]:
        """POST JSON and stream newline-delimited JSON objects."""


@dataclass(frozen=True, slots=True)
class OllamaBackendConfig:
    """
    Configuration for OllamaLocalLLMBackend.

    This config describes how to talk to a local Ollama server. It does not
    change the cognition architecture.
    """

    name: str = "ollama_local_llm_backend"
    base_url: str = "http://127.0.0.1:11434"
    model: str = "llama3.2:3b"
    timeout_seconds: float = 120.0
    temperature: float = 0.4
    top_p: float = 0.9
    num_predict: int = 512
    keep_alive: str | None = "10m"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if not self.base_url.strip():
            raise ValueError("base_url cannot be empty.")

        if not self.model.strip():
            raise ValueError("model cannot be empty.")

        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")

        if self.temperature < 0:
            raise ValueError("temperature cannot be negative.")

        if self.top_p <= 0 or self.top_p > 1:
            raise ValueError("top_p must be greater than 0 and at most 1.")

        if self.num_predict <= 0:
            raise ValueError("num_predict must be greater than zero.")


class UrllibOllamaHttpClient:
    """
    urllib-based Ollama HTTP client.

    Uses only Python standard library so the project does not gain a dependency
    just to smoke-test local LLM integration.
    """

    def __init__(self, *, base_url: str) -> None:
        cleaned = base_url.rstrip("/")

        if not cleaned:
            raise ValueError("base_url cannot be empty.")

        self._base_url = cleaned

    def post_json(
        self,
        *,
        path: str,
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        request = self._request(path=path, payload=payload)

        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw_body = response.read().decode("utf-8")

        parsed = json.loads(raw_body)

        if not isinstance(parsed, dict):
            raise ValueError("Ollama response was not a JSON object.")

        return parsed

    def stream_json(
        self,
        *,
        path: str,
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Iterator[dict[str, Any]]:
        request = self._request(path=path, payload=payload)

        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()

                if not line:
                    continue

                parsed = json.loads(line)

                if not isinstance(parsed, dict):
                    raise ValueError("Ollama stream item was not a JSON object.")

                yield parsed

    def _request(
        self,
        *,
        path: str,
        payload: Mapping[str, Any],
    ) -> urllib.request.Request:
        data = json.dumps(payload).encode("utf-8")

        return urllib.request.Request(
            url=self._url(path),
            data=data,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

    def _url(self, path: str) -> str:
        cleaned_path = path if path.startswith("/") else f"/{path}"

        return f"{self._base_url}{cleaned_path}"


class OllamaLocalLLMBackend:
    """
    Real local LLM backend for Ollama.

    Responsibilities:
    - translate LocalLLMBackend calls into Ollama /api/generate requests
    - support non-streaming and streaming generation
    - preserve streaming token whitespace
    - support cooperative cancellation checks
    - expose diagnostics

    Non-responsibilities:
    - no cognition orchestration
    - no EventBus
    - no direct TTS/playback
    - no laptop tool execution
    """

    def __init__(
        self,
        *,
        config: OllamaBackendConfig | None = None,
        client: OllamaHttpClient | None = None,
    ) -> None:
        self._config = config or OllamaBackendConfig()
        self._config.validate()

        self._client = client or UrllibOllamaHttpClient(
            base_url=self._config.base_url,
        )
        self._lock = RLock()
        self._logger = get_logger("cognition.ollama_backend")

        self._status = LocalLLMBackendStatus.READY
        self._request_count = 0
        self._streaming_count = 0
        self._cancelled_count = 0
        self._last_error: str | None = None
        self._cancelled_request_ids: set[str] = set()

    @property
    def name(self) -> str:
        return self._config.name

    def generate(
        self,
        *,
        prompt: str,
        system_prompt: str,
        request: CognitionRequest,
    ) -> LocalLLMBackendResult:
        """
        Generate one complete response through Ollama.
        """

        self._mark_busy(request_id=request.request_id)

        try:
            payload = self._payload(
                prompt=prompt,
                system_prompt=system_prompt,
                stream=False,
            )
            response = self._client.post_json(
                path="/api/generate",
                payload=payload,
                timeout_seconds=self._config.timeout_seconds,
            )
            text = self._extract_response_text(response)

            result = LocalLLMBackendResult(
                text=text,
                confidence=0.85,
                metadata={
                    "model": self._config.model,
                    "backend": self.name,
                    "ollama_done": response.get("done"),
                    "eval_count": response.get("eval_count"),
                    "prompt_eval_count": response.get("prompt_eval_count"),
                },
            )
            result.validate()

            self._mark_ready()

            self._logger.info(
                "ollama_backend_generated",
                backend=self.name,
                model=self._config.model,
                request_id=request.request_id,
                text_length=len(result.text),
            )

            return result

        except Exception as exc:
            self._mark_unavailable(f"{type(exc).__name__}: {exc}")

            raise

    def stream(
        self,
        *,
        prompt: str,
        system_prompt: str,
        request: CognitionRequest,
    ) -> Iterator[LocalLLMBackendToken]:
        """
        Stream response tokens through Ollama.

        Ollama returns newline-delimited JSON objects where each object may
        contain a response fragment. Fragment whitespace is preserved.
        """

        with self._lock:
            self._status = LocalLLMBackendStatus.BUSY
            self._streaming_count += 1
            self._last_error = None
            self._cancelled_request_ids.discard(request.request_id)

        try:
            payload = self._payload(
                prompt=prompt,
                system_prompt=system_prompt,
                stream=True,
            )

            for item in self._client.stream_json(
                path="/api/generate",
                payload=payload,
                timeout_seconds=self._config.timeout_seconds,
            ):
                if self._is_cancelled(request.request_id):
                    break

                text = item.get("response")
                done = bool(item.get("done", False))

                if isinstance(text, str) and text.strip():
                    yield LocalLLMBackendToken(
                        text=text,
                        final=done,
                        metadata={
                            "model": self._config.model,
                            "backend": self.name,
                            "ollama_done": done,
                        },
                    )

            self._mark_ready()

        except Exception as exc:
            self._mark_unavailable(f"{type(exc).__name__}: {exc}")

            raise

    def cancel(
        self,
        *,
        request_id: str,
        reason: str | None = None,
    ) -> bool:
        """
        Cooperatively cancel streaming for a request.

        Ollama's plain generate endpoint does not provide a universal hard
        cancel API here. This backend marks the request cancelled so streaming
        iteration can stop safely at the next chunk boundary.
        """

        cleaned = request_id.strip()

        if not cleaned:
            return False

        with self._lock:
            self._cancelled_request_ids.add(cleaned)
            self._cancelled_count += 1
            self._last_error = reason

        self._logger.info(
            "ollama_backend_cancel_requested",
            backend=self.name,
            request_id=cleaned,
            reason=reason,
        )

        return True

    def snapshot(self) -> LocalLLMBackendSnapshot:
        """
        Return backend diagnostics.
        """

        with self._lock:
            return LocalLLMBackendSnapshot(
                name=self.name,
                status=self._status,
                request_count=self._request_count,
                streaming_count=self._streaming_count,
                cancelled_count=self._cancelled_count,
                last_error=self._last_error,
                metadata={
                    "model": self._config.model,
                    "base_url": self._config.base_url,
                    **self._config.metadata,
                },
            )

    def _payload(
        self,
        *,
        prompt: str,
        system_prompt: str,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._config.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": stream,
            "options": {
                "temperature": self._config.temperature,
                "top_p": self._config.top_p,
                "num_predict": self._config.num_predict,
            },
        }

        if self._config.keep_alive is not None:
            payload["keep_alive"] = self._config.keep_alive

        return payload

    @staticmethod
    def _extract_response_text(response: Mapping[str, Any]) -> str:
        value = response.get("response")

        if not isinstance(value, str) or not value.strip():
            raise ValueError("Ollama response did not contain text.")

        return value

    def _mark_busy(self, *, request_id: str) -> None:
        with self._lock:
            self._status = LocalLLMBackendStatus.BUSY
            self._request_count += 1
            self._last_error = None
            self._cancelled_request_ids.discard(request_id)

    def _mark_ready(self) -> None:
        with self._lock:
            self._status = LocalLLMBackendStatus.READY
            self._last_error = None

    def _mark_unavailable(self, error: str) -> None:
        with self._lock:
            self._status = LocalLLMBackendStatus.UNAVAILABLE
            self._last_error = error

        self._logger.error(
            "ollama_backend_failed",
            backend=self.name,
            model=self._config.model,
            error=error,
        )

    def _is_cancelled(self, request_id: str) -> bool:
        with self._lock:
            return request_id in self._cancelled_request_ids


def is_ollama_connection_error(exc: Exception) -> bool:
    """
    Return True when an exception likely means Ollama is not reachable.
    """

    return isinstance(
        exc,
        (
            ConnectionError,
            TimeoutError,
            urllib.error.URLError,
            urllib.error.HTTPError,
        ),
    )