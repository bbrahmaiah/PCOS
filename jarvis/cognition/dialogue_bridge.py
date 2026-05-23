from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Any, Protocol, runtime_checkable

from jarvis.cognition.models import (
    CognitionRequest,
    CognitionRequestKind,
    CognitionRuntimePolicy,
    SpokenResponseStyle,
)
from jarvis.runtime.observability.structured_logger import get_logger
from jarvis.runtime.shared.enums import EventCategory, EventType


@runtime_checkable
class RuntimeEventLike(Protocol):
    """
    Minimal runtime event shape required by the bridge.

    Properties are used instead of settable attributes so frozen runtime events
    and frozen test events both satisfy the protocol.
    """

    @property
    def event_id(self) -> str:
        """Stable event id."""

    @property
    def correlation_id(self) -> str:
        """Correlation id preserved across child events."""

    @property
    def event_type(self) -> EventType:
        """Runtime event type."""

    @property
    def payload(self) -> dict[str, Any]:
        """Runtime event payload."""

    def child(
        self,
        event_type: EventType,
        category: EventCategory,
        source: str,
        payload: dict[str, Any] | None = None,
        priority: Any | None = None,
    ) -> Any:
        """Create a child event preserving correlation and causation."""


@runtime_checkable
class EventBusLike(Protocol):
    """
    Minimal EventBus shape used by this bridge.
    """

    def subscribe(
        self,
        event_type: EventType,
        callback: Callable[[Any], Any],
        subscriber_name: str,
    ) -> Any:
        """Subscribe to one event type."""

    def publish_sync(self, event: Any) -> None:
        """Publish one event synchronously."""


@dataclass(frozen=True, slots=True)
class DialogueCognitionBridgeConfig:
    """
    Configuration for DialogueCognitionBridgeWorker.
    """

    name: str = "dialogue_cognition_bridge_worker"
    source: str = "dialogue_cognition_bridge_worker"
    default_timeout_ms: int = 30_000
    max_response_chars: int = 1_200
    streaming_enabled: bool = False
    allow_memory_lookup: bool = False
    allow_tools: bool = False
    spoken_style: SpokenResponseStyle = SpokenResponseStyle.CONCISE

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if not self.source.strip():
            raise ValueError("source cannot be empty.")

        if self.default_timeout_ms <= 0:
            raise ValueError("default_timeout_ms must be greater than zero.")

        if self.max_response_chars <= 0:
            raise ValueError("max_response_chars must be greater than zero.")


@dataclass(frozen=True, slots=True)
class DialogueCognitionBridgeResult:
    """
    Result of one dialogue-to-cognition conversion.
    """

    accepted: bool
    cognition_request: CognitionRequest | None = None
    cognition_event: Any | None = None
    reason: str | None = None

    @property
    def rejected(self) -> bool:
        return not self.accepted


@dataclass(frozen=True, slots=True)
class DialogueCognitionBridgeSnapshot:
    """
    Observable bridge diagnostics.
    """

    name: str
    started: bool
    subscribed: bool
    processed_count: int
    published_count: int
    rejected_count: int
    last_dialogue_request_id: str | None
    last_cognition_request_id: str | None
    last_error: str | None


class DialogueCognitionBridgeWorker:
    """
    Converts dialogue.response_requested events into cognition.requested events.

    Responsibilities:
    - subscribe to dialogue.response_requested
    - validate and normalize dialogue payloads
    - build typed CognitionRequest objects
    - publish cognition.requested child events
    - preserve correlation and causation through RuntimeEvent.child()

    Non-responsibilities:
    - no LLM calls
    - no TTS
    - no memory retrieval
    - no tool execution
    - no microphone/audio internals
    """

    def __init__(
        self,
        *,
        event_bus: EventBusLike,
        config: DialogueCognitionBridgeConfig | None = None,
    ) -> None:
        self._config = config or DialogueCognitionBridgeConfig()
        self._config.validate()

        self._event_bus = event_bus
        self._lock = RLock()
        self._logger = get_logger("cognition.dialogue_bridge")

        self._started = False
        self._subscribed = False
        self._subscription: Any | None = None

        self._processed_count = 0
        self._published_count = 0
        self._rejected_count = 0
        self._last_dialogue_request_id: str | None = None
        self._last_cognition_request_id: str | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def started(self) -> bool:
        with self._lock:
            return self._started

    def on_start(self) -> None:
        """
        Subscribe the bridge to dialogue.response_requested.

        Idempotent by design.
        """

        with self._lock:
            if self._started:
                return

            self._subscription = self._subscribe_to_dialogue_requests()
            self._started = True
            self._subscribed = True

        self._logger.info(
            "dialogue_cognition_bridge_subscribed",
            worker=self.name,
            event_type=EventType.DIALOGUE_RESPONSE_REQUESTED.value,
        )

    def on_stop(self) -> None:
        """
        Stop bridge processing.

        The current project EventBus does not require explicit unsubscribe for
        this worker path. Future unsubscribe support can be added around
        self._subscription without changing this public contract.
        """

        with self._lock:
            if not self._started:
                return

            self._started = False
            self._subscribed = False

        self._logger.info(
            "dialogue_cognition_bridge_stopped",
            worker=self.name,
        )

    def process_dialogue_response_requested(
        self,
        source_event: RuntimeEventLike,
    ) -> DialogueCognitionBridgeResult:
        """
        Convert and publish one dialogue.response_requested event.
        """

        with self._lock:
            if not self._started:
                self._rejected_count += 1
                self._last_error = "bridge is not started"

                return DialogueCognitionBridgeResult(
                    accepted=False,
                    reason="bridge is not started",
                )

            self._processed_count += 1

        if source_event.event_type != EventType.DIALOGUE_RESPONSE_REQUESTED:
            return self._reject("unsupported event type")

        try:
            cognition_request = self.build_cognition_request(source_event)

        except ValueError as exc:
            return self._reject(str(exc))

        cognition_event = source_event.child(
            event_type=EventType.COGNITION_REQUESTED,
            category=EventCategory.COGNITION,
            source=self._config.source,
            payload=self._event_payload_for(cognition_request),
        )

        self._event_bus.publish_sync(cognition_event)

        with self._lock:
            self._published_count += 1
            self._last_error = None
            self._last_cognition_request_id = cognition_request.request_id
            self._last_dialogue_request_id = self._dialogue_request_id(
                source_event.payload
            )

        self._logger.info(
            "dialogue_cognition_request_published",
            worker=self.name,
            dialogue_request_id=self._last_dialogue_request_id,
            cognition_request_id=cognition_request.request_id,
            correlation_id=source_event.correlation_id,
        )

        return DialogueCognitionBridgeResult(
            accepted=True,
            cognition_request=cognition_request,
            cognition_event=cognition_event,
        )

    def build_cognition_request(
        self,
        source_event: RuntimeEventLike,
    ) -> CognitionRequest:
        """
        Build a typed CognitionRequest from a dialogue event.
        """

        payload = source_event.payload
        text = self._extract_text(payload)

        if text is None:
            raise ValueError("dialogue payload does not contain response text.")

        dialogue_request_id = self._dialogue_request_id(payload)
        cognition_request_id = self._cognition_request_id(
            dialogue_request_id=dialogue_request_id,
            source_event_id=source_event.event_id,
        )

        policy = CognitionRuntimePolicy(
            cancellable=True,
            streaming_enabled=self._config.streaming_enabled,
            allow_tools=self._config.allow_tools,
            allow_memory_lookup=self._config.allow_memory_lookup,
            max_response_chars=self._config.max_response_chars,
            timeout_ms=self._config.default_timeout_ms,
            spoken_style=self._config.spoken_style,
            metadata={
                "source_event_id": source_event.event_id,
                "dialogue_request_id": dialogue_request_id,
            },
        )

        return CognitionRequest(
            request_id=cognition_request_id,
            kind=CognitionRequestKind.USER_UTTERANCE,
            text=text,
            source="dialogue",
            turn_id=self._optional_str(payload.get("turn_id")),
            transcript_id=self._optional_str(payload.get("transcript_id")),
            correlation_id=source_event.correlation_id,
            policy=policy,
            metadata={
                "source_event_id": source_event.event_id,
                "dialogue_request_id": dialogue_request_id,
                "segment_id": self._optional_str(payload.get("segment_id")),
                "language": self._optional_str(payload.get("language")),
                "confidence": payload.get("confidence"),
            },
        )

    def snapshot(self) -> DialogueCognitionBridgeSnapshot:
        """
        Return bridge diagnostics.
        """

        with self._lock:
            return DialogueCognitionBridgeSnapshot(
                name=self.name,
                started=self._started,
                subscribed=self._subscribed,
                processed_count=self._processed_count,
                published_count=self._published_count,
                rejected_count=self._rejected_count,
                last_dialogue_request_id=self._last_dialogue_request_id,
                last_cognition_request_id=self._last_cognition_request_id,
                last_error=self._last_error,
            )

    def _subscribe_to_dialogue_requests(self) -> Any:
        try:
            return self._event_bus.subscribe(
                event_type=EventType.DIALOGUE_RESPONSE_REQUESTED,
                callback=self.process_dialogue_response_requested,
                subscriber_name=self.name,
            )

        except TypeError:
            return self._event_bus.subscribe(
                EventType.DIALOGUE_RESPONSE_REQUESTED,
                self.process_dialogue_response_requested,
                self.name,
            )

    def _reject(self, reason: str) -> DialogueCognitionBridgeResult:
        with self._lock:
            self._rejected_count += 1
            self._last_error = reason

        self._logger.info(
            "dialogue_cognition_bridge_rejected",
            worker=self.name,
            reason=reason,
        )

        return DialogueCognitionBridgeResult(
            accepted=False,
            reason=reason,
        )

    def _event_payload_for(
        self,
        cognition_request: CognitionRequest,
    ) -> dict[str, Any]:
        return {
            "request_id": cognition_request.request_id,
            "kind": cognition_request.kind.value,
            "text": cognition_request.text,
            "source": cognition_request.source,
            "turn_id": cognition_request.turn_id,
            "transcript_id": cognition_request.transcript_id,
            "correlation_id": cognition_request.correlation_id,
            "context": cognition_request.context.model_dump(mode="json"),
            "policy": cognition_request.policy.model_dump(mode="json"),
            "metadata": cognition_request.metadata,
        }

    def _extract_text(self, payload: dict[str, Any]) -> str | None:
        for key in ("text", "transcript_text", "user_text", "utterance"):
            value = payload.get(key)

            if isinstance(value, str) and value.strip():
                return value.strip()

        return None

    def _dialogue_request_id(self, payload: dict[str, Any]) -> str | None:
        return self._optional_str(payload.get("request_id"))

    def _cognition_request_id(
        self,
        *,
        dialogue_request_id: str | None,
        source_event_id: str,
    ) -> str:
        if dialogue_request_id is not None:
            return f"cognition-{dialogue_request_id}"

        return f"cognition-{source_event_id}"

    @staticmethod
    def _optional_str(value: Any) -> str | None:
        if value is None:
            return None

        cleaned = str(value).strip()

        return cleaned or None