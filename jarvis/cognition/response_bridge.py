from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Any, Protocol, runtime_checkable

from jarvis.runtime.observability.structured_logger import get_logger
from jarvis.runtime.shared.enums import EventCategory, EventType


@runtime_checkable
class RuntimeEventLike(Protocol):
    """
    Minimal runtime event shape required by the response bridge.

    Properties support frozen runtime events and frozen test events.
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
class CognitionDialogueBridgeConfig:
    """
    Configuration for CognitionDialogueBridgeWorker.
    """

    name: str = "cognition_dialogue_bridge_worker"
    source: str = "cognition_dialogue_bridge_worker"
    publish_failure_fallback: bool = True
    failure_fallback_text: str = "I had trouble thinking that through, sir."
    max_response_chars: int = 4_000

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if not self.source.strip():
            raise ValueError("source cannot be empty.")

        if not self.failure_fallback_text.strip():
            raise ValueError("failure_fallback_text cannot be empty.")

        if self.max_response_chars <= 0:
            raise ValueError("max_response_chars must be greater than zero.")


@dataclass(frozen=True, slots=True)
class CognitionDialogueBridgeResult:
    """
    Result of one cognition-to-dialogue conversion.
    """

    accepted: bool
    dialogue_event: Any | None = None
    response_text: str | None = None
    reason: str | None = None

    @property
    def rejected(self) -> bool:
        return not self.accepted


@dataclass(frozen=True, slots=True)
class CognitionDialogueBridgeSnapshot:
    """
    Observable bridge diagnostics.
    """

    name: str
    started: bool
    subscribed: bool
    completed_processed_count: int
    failed_processed_count: int
    published_count: int
    rejected_count: int
    last_cognition_request_id: str | None
    last_cognition_response_id: str | None
    last_dialogue_response_id: str | None
    last_error: str | None


class CognitionDialogueBridgeWorker:
    """
    Converts cognition output events into dialogue.response_ready events.

    Responsibilities:
    - subscribe to cognition.completed
    - subscribe to cognition.failed
    - normalize cognition response payloads
    - publish dialogue.response_ready child events
    - preserve correlation and causation

    Non-responsibilities:
    - no LLM calls
    - no TTS/playback
    - no microphone/STT knowledge
    - no tool execution
    - no memory retrieval
    """

    def __init__(
        self,
        *,
        event_bus: EventBusLike,
        config: CognitionDialogueBridgeConfig | None = None,
    ) -> None:
        self._config = config or CognitionDialogueBridgeConfig()
        self._config.validate()

        self._event_bus = event_bus
        self._lock = RLock()
        self._logger = get_logger("cognition.response_bridge")

        self._started = False
        self._subscribed = False
        self._subscriptions: list[Any] = []

        self._completed_processed_count = 0
        self._failed_processed_count = 0
        self._published_count = 0
        self._rejected_count = 0
        self._last_cognition_request_id: str | None = None
        self._last_cognition_response_id: str | None = None
        self._last_dialogue_response_id: str | None = None
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
        Subscribe to cognition completed/failed events.

        Idempotent by design.
        """

        with self._lock:
            if self._started:
                return

            self._subscriptions = [
                self._subscribe(
                    EventType.COGNITION_COMPLETED,
                    self.process_cognition_completed,
                ),
                self._subscribe(
                    EventType.COGNITION_FAILED,
                    self.process_cognition_failed,
                ),
            ]
            self._started = True
            self._subscribed = True

        self._logger.info(
            "cognition_dialogue_bridge_subscribed",
            worker=self.name,
            completed_event_type=EventType.COGNITION_COMPLETED.value,
            failed_event_type=EventType.COGNITION_FAILED.value,
        )

    def on_stop(self) -> None:
        """
        Stop bridge processing.

        Current EventBus tests do not require explicit unsubscribe. Future
        unsubscribe support can be added without changing this public contract.
        """

        with self._lock:
            if not self._started:
                return

            self._started = False
            self._subscribed = False

        self._logger.info(
            "cognition_dialogue_bridge_stopped",
            worker=self.name,
        )

    def process_cognition_completed(
        self,
        source_event: RuntimeEventLike,
    ) -> CognitionDialogueBridgeResult:
        """
        Convert one cognition.completed event into dialogue.response_ready.
        """

        with self._lock:
            if not self._started:
                self._rejected_count += 1
                self._last_error = "bridge is not started"

                return CognitionDialogueBridgeResult(
                    accepted=False,
                    reason="bridge is not started",
                )

            self._completed_processed_count += 1

        if source_event.event_type != EventType.COGNITION_COMPLETED:
            return self._reject("unsupported event type")

        response_payload = self._extract_response_payload(source_event.payload)
        response_text = self._extract_text(response_payload)

        if response_text is None:
            return self._reject("cognition payload does not contain response text.")

        dialogue_event = self._publish_dialogue_response_ready(
            source_event=source_event,
            response_payload=response_payload,
            response_text=response_text,
            fallback=False,
        )

        return CognitionDialogueBridgeResult(
            accepted=True,
            dialogue_event=dialogue_event,
            response_text=response_text,
        )

    def process_cognition_failed(
        self,
        source_event: RuntimeEventLike,
    ) -> CognitionDialogueBridgeResult:
        """
        Convert cognition.failed into a safe dialogue fallback response.

        This keeps the voice system graceful instead of silently failing.
        """

        with self._lock:
            if not self._started:
                self._rejected_count += 1
                self._last_error = "bridge is not started"

                return CognitionDialogueBridgeResult(
                    accepted=False,
                    reason="bridge is not started",
                )

            self._failed_processed_count += 1

        if source_event.event_type != EventType.COGNITION_FAILED:
            return self._reject("unsupported event type")

        if not self._config.publish_failure_fallback:
            return self._reject("failure fallback publishing is disabled")

        failure_payload = self._extract_failure_payload(source_event.payload)
        response_text = self._config.failure_fallback_text.strip()

        dialogue_event = self._publish_dialogue_response_ready(
            source_event=source_event,
            response_payload=failure_payload,
            response_text=response_text,
            fallback=True,
        )

        return CognitionDialogueBridgeResult(
            accepted=True,
            dialogue_event=dialogue_event,
            response_text=response_text,
        )

    def snapshot(self) -> CognitionDialogueBridgeSnapshot:
        """
        Return bridge diagnostics.
        """

        with self._lock:
            return CognitionDialogueBridgeSnapshot(
                name=self.name,
                started=self._started,
                subscribed=self._subscribed,
                completed_processed_count=self._completed_processed_count,
                failed_processed_count=self._failed_processed_count,
                published_count=self._published_count,
                rejected_count=self._rejected_count,
                last_cognition_request_id=self._last_cognition_request_id,
                last_cognition_response_id=self._last_cognition_response_id,
                last_dialogue_response_id=self._last_dialogue_response_id,
                last_error=self._last_error,
            )

    def _subscribe(
        self,
        event_type: EventType,
        callback: Callable[[RuntimeEventLike], CognitionDialogueBridgeResult],
    ) -> Any:
        try:
            return self._event_bus.subscribe(
                event_type=event_type,
                callback=callback,
                subscriber_name=self.name,
            )

        except TypeError:
            return self._event_bus.subscribe(
                event_type,
                callback,
                self.name,
            )

    def _publish_dialogue_response_ready(
        self,
        *,
        source_event: RuntimeEventLike,
        response_payload: dict[str, Any],
        response_text: str,
        fallback: bool,
    ) -> Any:
        cognition_request_id = self._cognition_request_id(response_payload)
        cognition_response_id = self._cognition_response_id(response_payload)
        dialogue_response_id = self._dialogue_response_id(
            cognition_request_id=cognition_request_id,
            cognition_response_id=cognition_response_id,
            source_event_id=source_event.event_id,
            fallback=fallback,
        )
        clean_text = self._bounded_text(response_text)

        dialogue_event = source_event.child(
            event_type=EventType.DIALOGUE_RESPONSE_READY,
            category=EventCategory.DIALOGUE,
            source=self._config.source,
            payload={
                "response_id": dialogue_response_id,
                "text": clean_text,
                "source": "cognition",
                "cognition_request_id": cognition_request_id,
                "cognition_response_id": cognition_response_id,
                "correlation_id": source_event.correlation_id,
                "fallback": fallback,
                "metadata": {
                    "source_event_id": source_event.event_id,
                    "cognition_request_id": cognition_request_id,
                    "cognition_response_id": cognition_response_id,
                    "kind": self._optional_str(response_payload.get("kind")),
                    "confidence": response_payload.get("confidence"),
                    "fallback": fallback,
                },
            },
        )

        self._event_bus.publish_sync(dialogue_event)

        with self._lock:
            self._published_count += 1
            self._last_error = None
            self._last_cognition_request_id = cognition_request_id
            self._last_cognition_response_id = cognition_response_id
            self._last_dialogue_response_id = dialogue_response_id

        self._logger.info(
            "cognition_dialogue_response_published",
            worker=self.name,
            cognition_request_id=cognition_request_id,
            cognition_response_id=cognition_response_id,
            dialogue_response_id=dialogue_response_id,
            fallback=fallback,
        )

        return dialogue_event

    def _reject(self, reason: str) -> CognitionDialogueBridgeResult:
        with self._lock:
            self._rejected_count += 1
            self._last_error = reason

        self._logger.info(
            "cognition_dialogue_bridge_rejected",
            worker=self.name,
            reason=reason,
        )

        return CognitionDialogueBridgeResult(
            accepted=False,
            reason=reason,
        )

    def _extract_response_payload(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        response = payload.get("response")

        if isinstance(response, dict):
            return response

        return payload

    def _extract_failure_payload(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        failure = payload.get("failure")

        if isinstance(failure, dict):
            return failure

        return payload

    def _extract_text(self, payload: dict[str, Any]) -> str | None:
        for key in ("text", "response_text", "assistant_text", "message"):
            value = payload.get(key)

            if isinstance(value, str) and value.strip():
                return value.strip()

        return None

    def _cognition_request_id(self, payload: dict[str, Any]) -> str | None:
        return self._optional_str(
            payload.get("request_id")
            or payload.get("cognition_request_id")
        )

    def _cognition_response_id(self, payload: dict[str, Any]) -> str | None:
        return self._optional_str(
            payload.get("response_id")
            or payload.get("failure_id")
            or payload.get("cognition_response_id")
        )

    def _dialogue_response_id(
        self,
        *,
        cognition_request_id: str | None,
        cognition_response_id: str | None,
        source_event_id: str,
        fallback: bool,
    ) -> str:
        prefix = "dialogue-fallback" if fallback else "dialogue-response"

        if cognition_response_id is not None:
            return f"{prefix}-{cognition_response_id}"

        if cognition_request_id is not None:
            return f"{prefix}-{cognition_request_id}"

        return f"{prefix}-{source_event_id}"

    def _bounded_text(self, text: str) -> str:
        clean_text = text.strip()

        if len(clean_text) <= self._config.max_response_chars:
            return clean_text

        return clean_text[: self._config.max_response_chars].rstrip()

    @staticmethod
    def _optional_str(value: Any) -> str | None:
        if value is None:
            return None

        cleaned = str(value).strip()

        return cleaned or None