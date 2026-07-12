from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Protocol, cast
from urllib.parse import urlparse

from jarvis.voice import (
    VoiceAwarenessCognitionBridge,
    VoiceAwarenessCognitionBridgeResult,
    VoiceCognitionPolicy,
    VoiceCognitionRequest,
    VoiceCognitionResponseRuntime,
    VoiceCognitionStatus,
    VoiceOllamaGeneratorConfig,
    VoiceOllamaResponseGenerator,
    VoiceTranscript,
    VoiceTranscriptKind,
    make_voice_segment_id,
    make_voice_session_id,
    make_voice_transcript_id,
    utc_now,
)

JsonObject = dict[str, object]


class JarvisHttpCognitionBridge(Protocol):
    def prepare(self, *, user_label: str, assistant_name: str) -> object:
        raise NotImplementedError

    def think_with_awareness(
        self,
        request: VoiceCognitionRequest,
    ) -> VoiceAwarenessCognitionBridgeResult:
        raise NotImplementedError

    def snapshot(self) -> object:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class JarvisHttpBridgeConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    user_label: str = "Balu"
    assistant_name: str = "JARVIS"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout_seconds: int = 25
    ollama_max_sentences: int = 2
    ollama_temperature: float = 0.25
    ollama_num_predict: int = 96
    ollama_keep_alive: str = "20m"
    ollama_prewarm_on_start: bool = False
    ollama_stream_response: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.host.strip():
            raise ValueError("host cannot be empty.")
        if not 0 < self.port < 65_536:
            raise ValueError("port must be 1..65535.")
        if not self.user_label.strip():
            raise ValueError("user_label cannot be empty.")
        if not self.assistant_name.strip():
            raise ValueError("assistant_name cannot be empty.")
        if not self.ollama_base_url.strip():
            raise ValueError("ollama_base_url cannot be empty.")
        if not self.ollama_model.strip():
            raise ValueError("ollama_model cannot be empty.")
        if self.ollama_timeout_seconds <= 0:
            raise ValueError("ollama_timeout_seconds must be positive.")
        if self.ollama_max_sentences <= 0:
            raise ValueError("ollama_max_sentences must be positive.")
        if not 0.0 <= self.ollama_temperature <= 2.0:
            raise ValueError("ollama_temperature must be 0..2.")
        if self.ollama_num_predict <= 0:
            raise ValueError("ollama_num_predict must be positive.")
        if not self.ollama_keep_alive.strip():
            raise ValueError("ollama_keep_alive cannot be empty.")


@dataclass(frozen=True, slots=True)
class JarvisHttpBridgeTurnResult:
    status_code: int
    body: JsonObject


class JarvisHttpBridgeRuntime:
    """
    Local cockpit bridge for the real JARVIS runtime.

    This adapter intentionally does not invent JARVIS behavior. User text is
    routed through the same awareness -> cognition -> Step 50 response boundary
    path used by the connected voice launcher.
    """

    def __init__(
        self,
        *,
        config: JarvisHttpBridgeConfig | None = None,
        cognition: JarvisHttpCognitionBridge | None = None,
    ) -> None:
        self._config = config or JarvisHttpBridgeConfig()
        self._cognition = cognition or _build_default_cognition(self._config)
        self._lock = threading.RLock()
        self._prepared = False
        self._turns = 0
        self._failed_turns = 0
        self._last_latency_ms: float | None = None
        self._last_error: str | None = None
        self._started_at = utc_now()

    @property
    def config(self) -> JarvisHttpBridgeConfig:
        return self._config

    def prepare(self) -> None:
        with self._lock:
            if self._prepared:
                return
            self._cognition.prepare(
                user_label=self._config.user_label,
                assistant_name=self._config.assistant_name,
            )
            self._prepared = True
            self._last_error = None

    def health(self) -> JsonObject:
        status = "ready" if self._prepared else "created"
        return {
            "status": status,
            "runtime": "jarvis_http_bridge",
            "realRuntime": True,
            "fakeFallbackEnabled": False,
            "userLabel": self._config.user_label,
            "assistantName": self._config.assistant_name,
            "ollamaBaseUrl": self._config.ollama_base_url,
            "ollamaModel": self._config.ollama_model,
            "streaming": self._config.ollama_stream_response,
            "turns": self._turns,
            "failedTurns": self._failed_turns,
            "lastLatencyMs": self._last_latency_ms,
            "lastError": self._last_error,
            "startedAt": self._started_at.isoformat(),
        }

    def handle_turn(self, payload: JsonObject) -> JarvisHttpBridgeTurnResult:
        message = _clean_string(payload.get("message"))
        if not message:
            return _turn_result(
                HTTPStatus.BAD_REQUEST,
                text="No real instruction reached the JARVIS bridge.",
                system_check="bad_request",
                analysis="The HTTP bridge received an empty message.",
                source="jarvis_http_bridge",
            )

        started = time.perf_counter()
        active_role = _clean_string(payload.get("activeRole"))
        metadata = _build_metadata(payload=payload, active_role=active_role)
        transcript = VoiceTranscript(
            transcript_id=make_voice_transcript_id(),
            session_id=make_voice_session_id(),
            segment_id=make_voice_segment_id(),
            kind=VoiceTranscriptKind.FINAL,
            text=message,
            confidence=_confidence(payload.get("confidence")),
            created_at=utc_now(),
            metadata={"source": "http_cockpit"},
        )
        request = VoiceCognitionRequest(
            transcript=transcript,
            user_label=self._config.user_label,
            assistant_name=self._config.assistant_name,
            allow_action_candidate=False,
            partial_prediction_only=False,
            metadata=metadata,
        )

        try:
            with self._lock:
                if not self._prepared:
                    self.prepare()
                bridge_result = self._cognition.think_with_awareness(request)
                self._turns += 1
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            self._failed_turns += 1
            self._last_latency_ms = latency_ms
            self._last_error = str(exc)
            return _turn_result(
                HTTPStatus.SERVICE_UNAVAILABLE,
                text=(
                    "Real JARVIS cognition is not reachable. Start Ollama and "
                    "the connected runtime, then retry."
                ),
                system_check="real_runtime_unavailable",
                analysis=str(exc),
                source="jarvis_http_bridge",
                latency_ms=latency_ms,
                alert="REAL JARVIS RUNTIME OFFLINE",
            )

        latency_ms = (time.perf_counter() - started) * 1000.0
        self._last_latency_ms = latency_ms
        self._last_error = None
        return self._response_from_bridge_result(
            bridge_result=bridge_result,
            latency_ms=latency_ms,
        )

    def _response_from_bridge_result(
        self,
        *,
        bridge_result: VoiceAwarenessCognitionBridgeResult,
        latency_ms: float,
    ) -> JarvisHttpBridgeTurnResult:
        cognition_result = bridge_result.cognition_result
        response = (
            cognition_result.response
            if cognition_result is not None
            else None
        )

        if cognition_result is None or response is None:
            self._failed_turns += 1
            status_name = (
                cognition_result.status.value
                if cognition_result is not None
                else bridge_result.status.value
            )
            reason = (
                cognition_result.message
                if cognition_result is not None
                else bridge_result.reason
            )
            self._last_error = reason
            return _turn_result(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                text=(
                    "I heard you, but the real response boundary did not "
                    "release a spoken answer. Please repeat that more clearly."
                ),
                system_check=f"real_runtime_{status_name}",
                analysis=reason,
                source="jarvis_http_bridge",
                latency_ms=latency_ms,
            )

        system_check = "real_runtime_ready"
        alert: str | None = None
        if (
            cognition_result.status == VoiceCognitionStatus.DEGRADED
            or bridge_result.status.value == "degraded"
        ):
            system_check = "real_runtime_degraded"
            alert = "REAL JARVIS RUNTIME DEGRADED"

        metadata = {
            **bridge_result.metadata,
            **cognition_result.metadata,
            "bridgeStatus": bridge_result.status.value,
            "cognitionStatus": cognition_result.status.value,
            "safety": cognition_result.safety.value,
            "awarenessStatus": (
                bridge_result.awareness_packet.status.value
                if bridge_result.awareness_packet is not None
                else None
            ),
            "responseLatencyMs": cognition_result.response_latency_ms,
            "contextLatencyMs": cognition_result.context_latency_ms,
            "firstTokenLatencyMs": response.metadata.get(
                "first_token_latency_ms"
            ),
        }

        return _turn_result(
            HTTPStatus.OK,
            text=response.text,
            system_check=system_check,
            analysis=(
                "Real runtime response produced through awareness, cognition, "
                "and the Step 50 response boundary."
            ),
            source="real_jarvis_runtime",
            latency_ms=latency_ms,
            alert=alert,
            metadata=metadata,
        )


def create_jarvis_http_server(
    *,
    config: JarvisHttpBridgeConfig | None = None,
    runtime: JarvisHttpBridgeRuntime | None = None,
) -> ThreadingHTTPServer:
    bridge_runtime = runtime or JarvisHttpBridgeRuntime(config=config)
    server_config = bridge_runtime.config
    handler_class = _make_handler(bridge_runtime)
    return ThreadingHTTPServer(
        (server_config.host, server_config.port),
        handler_class,
    )


def _build_default_cognition(
    config: JarvisHttpBridgeConfig,
) -> VoiceAwarenessCognitionBridge:
    return VoiceAwarenessCognitionBridge(
        cognition=VoiceCognitionResponseRuntime(
            response_generator=VoiceOllamaResponseGenerator(
                config=VoiceOllamaGeneratorConfig(
                    base_url=config.ollama_base_url,
                    model=config.ollama_model,
                    timeout_seconds=config.ollama_timeout_seconds,
                    max_sentences=config.ollama_max_sentences,
                    temperature=config.ollama_temperature,
                    num_predict=config.ollama_num_predict,
                    keep_alive=config.ollama_keep_alive,
                    prewarm_on_prepare=config.ollama_prewarm_on_start,
                    stream_response=config.ollama_stream_response,
                    metadata={"profile": "http_cockpit_bridge"},
                )
            ),
            policy=VoiceCognitionPolicy(
                min_dialogue_confidence=0.55,
                max_response_sentences=config.ollama_max_sentences,
                require_wake_word_when_sleeping=False,
            ),
        )
    )


def _make_handler(
    runtime: JarvisHttpBridgeRuntime,
) -> type[BaseHTTPRequestHandler]:
    class JarvisHttpBridgeHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in {"/api/health", "/health"}:
                self._send_json(HTTPStatus.OK, runtime.health())
                return
            if parsed.path == "/api/jarvis/status":
                self._send_json(HTTPStatus.OK, runtime.health())
                return
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"error": "Unknown JARVIS bridge endpoint."},
            )

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/jarvis":
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"error": "Unknown JARVIS bridge endpoint."},
                )
                return

            payload_result = self._read_json_body()
            if isinstance(payload_result, str):
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": payload_result},
                )
                return

            result = runtime.handle_turn(payload_result)
            self._send_json(HTTPStatus(result.status_code), result.body)

        def log_message(self, format: str, *args: object) -> None:
            return

        def _read_json_body(self) -> JsonObject | str:
            length_header = self.headers.get("Content-Length", "0")
            try:
                length = int(length_header)
            except ValueError:
                return "Invalid Content-Length."
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                return f"Invalid JSON: {exc.msg}"
            if not isinstance(payload, dict):
                return "JSON body must be an object."
            return cast(JsonObject, payload)

        def _send_json(
            self,
            status: HTTPStatus,
            payload: JsonObject,
        ) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

    return JarvisHttpBridgeHandler


def _turn_result(
    status: HTTPStatus,
    *,
    text: str,
    system_check: str,
    analysis: str,
    source: str,
    latency_ms: float | None = None,
    alert: str | None = None,
    metadata: JsonObject | None = None,
) -> JarvisHttpBridgeTurnResult:
    body: JsonObject = {
        "text": text,
        "alert": alert,
        "command": None,
        "simulationData": [],
        "analysis": analysis,
        "systemCheck": system_check,
        "source": source,
        "latencyMs": latency_ms,
        "fakeFallbackEnabled": False,
        "metadata": metadata or {},
    }
    return JarvisHttpBridgeTurnResult(status_code=status.value, body=body)


def _build_metadata(
    *,
    payload: JsonObject,
    active_role: str,
) -> JsonObject:
    telemetry = payload.get("telemetry")
    history = payload.get("history")
    metadata: JsonObject = {
        "source": "http_cockpit",
        "http_bridge": True,
        "active_role": active_role or "Personal Cognitive Operating System",
    }
    if isinstance(telemetry, dict):
        metadata["telemetry"] = cast(JsonObject, telemetry)
    if isinstance(history, list):
        metadata["recent_history"] = tuple(
            _history_item(item) for item in history[-6:]
        )
    return metadata


def _history_item(item: object) -> str:
    if not isinstance(item, dict):
        return str(item)
    role = _clean_string(item.get("role")) or "unknown"
    content = _clean_string(item.get("content"))
    return f"{role}: {content}" if content else role


def _confidence(value: object) -> float:
    if isinstance(value, int | float):
        return min(max(float(value), 0.0), 1.0)
    return 0.96


def _clean_string(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""
