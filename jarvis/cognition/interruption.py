from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Any, Protocol, runtime_checkable

from jarvis.cognition.worker import CognitionWorker
from jarvis.runtime.observability.structured_logger import get_logger
from jarvis.runtime.shared.enums import EventCategory, EventType


@runtime_checkable
class RuntimeEventLike(Protocol):
    """
    Minimal runtime event shape used by interruption/cancellation bridges.

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
    Minimal EventBus shape used by cancellation integration.
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
class PresenceCognitionInterruptBridgeConfig:
    """
    Configuration for presence-to-cognition interruption bridge.
    """

    name: str = "presence_cognition_interrupt_bridge_worker"
    source: str = "presence_cognition_interrupt_bridge_worker"
    default_reason: str = "user interrupted"

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if not self.source.strip():
            raise ValueError("source cannot be empty.")

        if not self.default_reason.strip():
            raise ValueError("default_reason cannot be empty.")


@dataclass(frozen=True, slots=True)
class PresenceCognitionInterruptBridgeResult:
    """
    Result of converting presence.interrupt_requested into cognition.cancel_requested.
    """

    accepted: bool
    cancel_event: Any | None = None
    request_id: str | None = None
    reason: str | None = None

    @property
    def rejected(self) -> bool:
        return not self.accepted


@dataclass(frozen=True, slots=True)
class PresenceCognitionInterruptBridgeSnapshot:
    """
    Observable bridge diagnostics.
    """

    name: str
    started: bool
    subscribed: bool
    processed_count: int
    published_count: int
    rejected_count: int
    last_interrupt_event_id: str | None
    last_cancel_request_id: str | None
    last_reason: str | None
    last_error: str | None


class PresenceCognitionInterruptBridgeWorker:
    """
    Converts presence interruption events into cognition cancellation events.

    Responsibilities:
    - subscribe to presence.interrupt_requested
    - normalize interruption payloads
    - publish cognition.cancel_requested child events
    - preserve correlation and causation

    Non-responsibilities:
    - no model calls
    - no TTS/playback stopping
    - no direct adapter cancellation
    - no audio internals
    """

    def __init__(
        self,
        *,
        event_bus: EventBusLike,
        config: PresenceCognitionInterruptBridgeConfig | None = None,
    ) -> None:
        self._config = config or PresenceCognitionInterruptBridgeConfig()
        self._config.validate()

        self._event_bus = event_bus
        self._lock = RLock()
        self._logger = get_logger("cognition.interrupt_bridge")

        self._started = False
        self._subscribed = False
        self._subscription: Any | None = None

        self._processed_count = 0
        self._published_count = 0
        self._rejected_count = 0
        self._last_interrupt_event_id: str | None = None
        self._last_cancel_request_id: str | None = None
        self._last_reason: str | None = None
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
        Subscribe to presence.interrupt_requested.
        """

        with self._lock:
            if self._started:
                return

            self._subscription = self._subscribe_to_interrupts()
            self._started = True
            self._subscribed = True

        self._logger.info(
            "presence_cognition_interrupt_bridge_subscribed",
            worker=self.name,
            event_type=EventType.PRESENCE_INTERRUPT_REQUESTED.value,
        )

    def on_stop(self) -> None:
        """
        Stop bridge processing.
        """

        with self._lock:
            if not self._started:
                return

            self._started = False
            self._subscribed = False

        self._logger.info(
            "presence_cognition_interrupt_bridge_stopped",
            worker=self.name,
        )

    def process_presence_interrupt_requested(
        self,
        source_event: RuntimeEventLike,
    ) -> PresenceCognitionInterruptBridgeResult:
        """
        Convert one presence interruption into cognition cancellation request.
        """

        with self._lock:
            if not self._started:
                self._rejected_count += 1
                self._last_error = "bridge is not started"

                return PresenceCognitionInterruptBridgeResult(
                    accepted=False,
                    reason="bridge is not started",
                )

            self._processed_count += 1

        if source_event.event_type != EventType.PRESENCE_INTERRUPT_REQUESTED:
            return self._reject("unsupported event type")

        request_id = self._target_request_id(source_event.payload)
        reason = self._reason(source_event.payload)

        cancel_event = source_event.child(
            event_type=EventType.COGNITION_CANCEL_REQUESTED,
            category=EventCategory.COGNITION,
            source=self._config.source,
            payload={
                "request_id": request_id,
                "reason": reason,
                "interrupt_event_id": source_event.event_id,
                "correlation_id": source_event.correlation_id,
                "metadata": {
                    "source_event_id": source_event.event_id,
                    "interrupt_id": self._optional_str(
                        source_event.payload.get("interrupt_id")
                    ),
                    "turn_id": self._optional_str(
                        source_event.payload.get("turn_id")
                    ),
                    "source": self._optional_str(
                        source_event.payload.get("source")
                    ),
                },
            },
        )

        self._event_bus.publish_sync(cancel_event)

        with self._lock:
            self._published_count += 1
            self._last_error = None
            self._last_interrupt_event_id = source_event.event_id
            self._last_cancel_request_id = request_id
            self._last_reason = reason

        self._logger.info(
            "cognition_cancel_requested_from_presence_interrupt",
            worker=self.name,
            request_id=request_id,
            reason=reason,
            correlation_id=source_event.correlation_id,
        )

        return PresenceCognitionInterruptBridgeResult(
            accepted=True,
            cancel_event=cancel_event,
            request_id=request_id,
            reason=reason,
        )

    def snapshot(self) -> PresenceCognitionInterruptBridgeSnapshot:
        """
        Return bridge diagnostics.
        """

        with self._lock:
            return PresenceCognitionInterruptBridgeSnapshot(
                name=self.name,
                started=self._started,
                subscribed=self._subscribed,
                processed_count=self._processed_count,
                published_count=self._published_count,
                rejected_count=self._rejected_count,
                last_interrupt_event_id=self._last_interrupt_event_id,
                last_cancel_request_id=self._last_cancel_request_id,
                last_reason=self._last_reason,
                last_error=self._last_error,
            )

    def _subscribe_to_interrupts(self) -> Any:
        try:
            return self._event_bus.subscribe(
                event_type=EventType.PRESENCE_INTERRUPT_REQUESTED,
                callback=self.process_presence_interrupt_requested,
                subscriber_name=self.name,
            )

        except TypeError:
            return self._event_bus.subscribe(
                EventType.PRESENCE_INTERRUPT_REQUESTED,
                self.process_presence_interrupt_requested,
                self.name,
            )

    def _reject(
        self,
        reason: str,
    ) -> PresenceCognitionInterruptBridgeResult:
        with self._lock:
            self._rejected_count += 1
            self._last_error = reason

        self._logger.info(
            "presence_cognition_interrupt_bridge_rejected",
            worker=self.name,
            reason=reason,
        )

        return PresenceCognitionInterruptBridgeResult(
            accepted=False,
            reason=reason,
        )

    def _target_request_id(
        self,
        payload: dict[str, Any],
    ) -> str | None:
        return self._optional_str(
            payload.get("cognition_request_id")
            or payload.get("request_id")
            or payload.get("active_request_id")
        )

    def _reason(
        self,
        payload: dict[str, Any],
    ) -> str:
        reason = self._optional_str(payload.get("reason"))

        return reason or self._config.default_reason

    @staticmethod
    def _optional_str(value: Any) -> str | None:
        if value is None:
            return None

        cleaned = str(value).strip()

        return cleaned or None


@dataclass(frozen=True, slots=True)
class CognitionCancelWorkerConfig:
    """
    Configuration for CognitionCancelWorker.
    """

    name: str = "cognition_cancel_worker"

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")


@dataclass(frozen=True, slots=True)
class CognitionCancelWorkerResult:
    """
    Result of applying one cognition.cancel_requested event.
    """

    accepted: bool
    request_id: str | None = None
    reason: str | None = None

    @property
    def rejected(self) -> bool:
        return not self.accepted


@dataclass(frozen=True, slots=True)
class CognitionCancelWorkerSnapshot:
    """
    Observable cancellation worker diagnostics.
    """

    name: str
    started: bool
    subscribed: bool
    processed_count: int
    accepted_count: int
    rejected_count: int
    last_request_id: str | None
    last_reason: str | None
    last_error: str | None


class CognitionCancelWorker:
    """
    Applies cognition.cancel_requested events to CognitionWorker.

    Responsibilities:
    - subscribe to cognition.cancel_requested
    - extract target request id and reason
    - call CognitionWorker.request_cancel()
    - expose diagnostics

    Non-responsibilities:
    - no LLM implementation
    - no event bridge from presence
    - no playback stopping
    - no final cancellation confirmation
    """

    def __init__(
        self,
        *,
        event_bus: EventBusLike,
        cognition_worker: CognitionWorker,
        config: CognitionCancelWorkerConfig | None = None,
    ) -> None:
        self._config = config or CognitionCancelWorkerConfig()
        self._config.validate()

        self._event_bus = event_bus
        self._cognition_worker = cognition_worker
        self._lock = RLock()
        self._logger = get_logger("cognition.cancel_worker")

        self._started = False
        self._subscribed = False
        self._subscription: Any | None = None

        self._processed_count = 0
        self._accepted_count = 0
        self._rejected_count = 0
        self._last_request_id: str | None = None
        self._last_reason: str | None = None
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
        Subscribe to cognition.cancel_requested.
        """

        with self._lock:
            if self._started:
                return

            self._subscription = self._subscribe_to_cancel_requests()
            self._started = True
            self._subscribed = True

        self._logger.info(
            "cognition_cancel_worker_subscribed",
            worker=self.name,
            event_type=EventType.COGNITION_CANCEL_REQUESTED.value,
        )

    def on_stop(self) -> None:
        """
        Stop cancellation processing.
        """

        with self._lock:
            if not self._started:
                return

            self._started = False
            self._subscribed = False

        self._logger.info("cognition_cancel_worker_stopped", worker=self.name)

    def process_cognition_cancel_requested(
        self,
        source_event: RuntimeEventLike,
    ) -> CognitionCancelWorkerResult:
        """
        Apply one cognition.cancel_requested event to CognitionWorker.
        """

        with self._lock:
            if not self._started:
                self._rejected_count += 1
                self._last_error = "cancel worker is not started"

                return CognitionCancelWorkerResult(
                    accepted=False,
                    reason="cancel worker is not started",
                )

            self._processed_count += 1

        if source_event.event_type != EventType.COGNITION_CANCEL_REQUESTED:
            return self._reject("unsupported event type")

        request_id = self._optional_str(source_event.payload.get("request_id"))
        reason = self._optional_str(source_event.payload.get("reason"))

        accepted = self._cognition_worker.request_cancel(
            request_id=request_id,
            reason=reason,
        )

        with self._lock:
            self._last_request_id = request_id
            self._last_reason = reason

            if accepted:
                self._accepted_count += 1
                self._last_error = None

            else:
                self._rejected_count += 1
                self._last_error = "cognition worker rejected cancellation"

        if accepted:
            self._logger.info(
                "cognition_cancel_applied",
                worker=self.name,
                request_id=request_id,
                reason=reason,
            )

        else:
            self._logger.info(
                "cognition_cancel_rejected",
                worker=self.name,
                request_id=request_id,
                reason=reason,
            )

        return CognitionCancelWorkerResult(
            accepted=accepted,
            request_id=request_id,
            reason=reason,
        )

    def snapshot(self) -> CognitionCancelWorkerSnapshot:
        """
        Return cancellation worker diagnostics.
        """

        with self._lock:
            return CognitionCancelWorkerSnapshot(
                name=self.name,
                started=self._started,
                subscribed=self._subscribed,
                processed_count=self._processed_count,
                accepted_count=self._accepted_count,
                rejected_count=self._rejected_count,
                last_request_id=self._last_request_id,
                last_reason=self._last_reason,
                last_error=self._last_error,
            )

    def _subscribe_to_cancel_requests(self) -> Any:
        try:
            return self._event_bus.subscribe(
                event_type=EventType.COGNITION_CANCEL_REQUESTED,
                callback=self.process_cognition_cancel_requested,
                subscriber_name=self.name,
            )

        except TypeError:
            return self._event_bus.subscribe(
                EventType.COGNITION_CANCEL_REQUESTED,
                self.process_cognition_cancel_requested,
                self.name,
            )

    def _reject(self, reason: str) -> CognitionCancelWorkerResult:
        with self._lock:
            self._rejected_count += 1
            self._last_error = reason

        self._logger.info(
            "cognition_cancel_worker_rejected",
            worker=self.name,
            reason=reason,
        )

        return CognitionCancelWorkerResult(
            accepted=False,
            reason=reason,
        )

    @staticmethod
    def _optional_str(value: Any) -> str | None:
        if value is None:
            return None

        cleaned = str(value).strip()

        return cleaned or None