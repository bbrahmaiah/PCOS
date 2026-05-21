from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import count
from queue import Empty, PriorityQueue
from threading import Event, RLock, Thread
from typing import Final

from jarvis.runtime.events.event_models import RuntimeEvent
from jarvis.runtime.events.priorities import priority_for_event
from jarvis.runtime.events.subscriptions import (
    EventCallback,
    EventSubscription,
)
from jarvis.runtime.observability.metrics import get_metrics
from jarvis.runtime.observability.performance_monitor import (
    get_performance_monitor,
)
from jarvis.runtime.observability.structured_logger import get_logger
from jarvis.runtime.observability.tracing import get_tracer
from jarvis.runtime.shared.enums import EventPriority, EventType

_PRIORITY_RANK: Final[dict[EventPriority, int]] = {
    EventPriority.CRITICAL: 0,
    EventPriority.HIGH: 1,
    EventPriority.NORMAL: 2,
    EventPriority.LOW: 3,
}


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class DeadLetterEvent:
    """
    Event delivery failure record.

    Dead letters allow the runtime to capture failed or expired events without
    crashing the dispatcher thread.
    """

    event: RuntimeEvent
    reason: str
    recorded_at: datetime = field(default_factory=_utc_now)
    subscriber_name: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class EventBusSnapshot:
    """
    Immutable diagnostic snapshot of EventBus state.
    """

    name: str
    running: bool
    queue_size: int
    subscription_count: int
    history_size: int
    dead_letter_size: int
    published_count: int
    delivered_count: int
    failed_count: int


class EventBus:
    """
    Priority-based, thread-safe runtime event bus.

    This is the communication backbone for JARVIS.

    Guarantees:
    - accepts only RuntimeEvent objects
    - dispatches by priority
    - preserves FIFO order inside the same priority
    - isolates callback failures
    - records failed deliveries into dead letters
    - stores bounded history
    - supports clean start/stop lifecycle
    - integrates logs, metrics, traces, and performance monitoring
    """

    def __init__(
        self,
        *,
        name: str = "runtime_event_bus",
        history_limit: int = 1_000,
        dead_letter_limit: int = 1_000,
        auto_start: bool = False,
    ) -> None:
        clean_name = name.strip()

        if not clean_name:
            raise ValueError("EventBus name cannot be empty.")

        if history_limit < 1:
            raise ValueError("history_limit must be greater than zero.")

        if dead_letter_limit < 1:
            raise ValueError("dead_letter_limit must be greater than zero.")

        self.name = clean_name

        self._lock = RLock()
        self._queue: PriorityQueue[tuple[int, int, RuntimeEvent]] = PriorityQueue()
        self._sequence = count()

        self._subscriptions: dict[
    EventType,
    list[EventSubscription],
] = defaultdict(list)
        self._history: deque[RuntimeEvent] = deque(maxlen=history_limit)
        self._dead_letters: deque[DeadLetterEvent] = deque(maxlen=dead_letter_limit)

        self._stop_requested = Event()
        self._running = False
        self._dispatcher: Thread | None = None

        self._published_count = 0
        self._delivered_count = 0
        self._failed_count = 0

        self._logger = get_logger("events.event_bus")
        self._metrics = get_metrics()
        self._performance = get_performance_monitor()
        self._tracer = get_tracer()

        if auto_start:
            self.start()

    def start(self) -> None:
        """
        Start the background dispatcher thread.

        Safe to call multiple times.
        """

        with self._lock:
            if self._running:
                return

            self._stop_requested.clear()
            self._running = True
            self._dispatcher = Thread(
                target=self._dispatch_loop,
                name=f"{self.name}_dispatcher",
                daemon=True,
            )
            self._dispatcher.start()

        self._metrics.increment("events.bus.started")

        self._logger.info(
            "event_bus_started",
            bus=self.name,
        )

    def stop(self, *, timeout_seconds: float = 5.0) -> None:
        """
        Request clean shutdown.

        The dispatch loop drains queued events before exiting.
        """

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")

        with self._lock:
            if not self._running:
                return

            self._stop_requested.set()
            dispatcher = self._dispatcher

        if dispatcher is not None:
            dispatcher.join(timeout=timeout_seconds)

        with self._lock:
            self._running = False
            self._dispatcher = None

        self._metrics.increment("events.bus.stopped")

        self._logger.info(
            "event_bus_stopped",
            bus=self.name,
        )

    def subscribe(
        self,
        event_type: EventType,
        subscriber_name: str,
        callback: EventCallback,
    ) -> EventSubscription:
        """
        Subscribe a callback to one event type.
        """

        subscription = EventSubscription(
            event_type=event_type,
            subscriber_name=subscriber_name,
            callback=callback,
        )

        with self._lock:
            self._subscriptions[event_type].append(subscription)

        self._metrics.increment("events.subscriptions.created")

        self._logger.info(
            "event_subscription_created",
            bus=self.name,
            event_type=event_type.value,
            subscriber_name=subscription.subscriber_name,
            subscription_id=subscription.subscription_id,
        )

        return subscription

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Remove a subscription by ID.

        Returns True if anything was removed.
        """

        clean_id = subscription_id.strip()

        if not clean_id:
            raise ValueError("subscription_id cannot be empty.")

        removed = False

        with self._lock:
            for event_type, subscriptions in list(self._subscriptions.items()):
                remaining = [
                    subscription
                    for subscription in subscriptions
                    if subscription.subscription_id != clean_id
                ]

                if len(remaining) != len(subscriptions):
                    removed = True

                if remaining:
                    self._subscriptions[event_type] = remaining
                else:
                    self._subscriptions.pop(event_type, None)

        if removed:
            self._metrics.increment("events.subscriptions.removed")

            self._logger.info(
                "event_subscription_removed",
                bus=self.name,
                subscription_id=clean_id,
            )

        return removed

    def publish(self, event: RuntimeEvent) -> str:
        """
        Queue an event for background priority dispatch.

        Returns the event_id.
        """

        self._validate_event(event)

        priority = event.priority or priority_for_event(event.event_type)
        priority_rank = _PRIORITY_RANK[priority]

        with self._lock:
            self._history.append(event)
            self._published_count += 1

        self._queue.put((priority_rank, next(self._sequence), event))
        self._metrics.increment("events.published")

        self._logger.info(
            "event_published",
            bus=self.name,
            event_id=event.event_id,
            correlation_id=event.correlation_id,
            event_type=event.event_type.value,
            category=event.category.value,
            priority=priority.value,
            source=event.source,
            async_dispatch=True,
        )

        return event.event_id

    def publish_sync(self, event: RuntimeEvent) -> int:
        """
        Deliver an event immediately in the current thread.

        This is useful for tests, bootstrap, and deterministic paths.
        Returns successful delivery count.
        """

        self._validate_event(event)

        with self._lock:
            self._history.append(event)
            self._published_count += 1

        self._metrics.increment("events.published")

        self._logger.info(
            "event_published",
            bus=self.name,
            event_id=event.event_id,
            correlation_id=event.correlation_id,
            event_type=event.event_type.value,
            category=event.category.value,
            priority=(event.priority or priority_for_event(event.event_type)).value,
            source=event.source,
            async_dispatch=False,
        )

        return self._deliver(event)

    def drain(self, *, timeout_seconds: float = 5.0) -> bool:
        """
        Wait until the async queue is empty.

        Returns True if drained before timeout.
        """

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")

        completed = Event()

        def wait_for_queue() -> None:
            self._queue.join()
            completed.set()

        waiter = Thread(
            target=wait_for_queue,
            name=f"{self.name}_drain_waiter",
            daemon=True,
        )
        waiter.start()

        return completed.wait(timeout_seconds)

    def snapshot(self) -> EventBusSnapshot:
        with self._lock:
            subscription_count = sum(
                len(subscriptions)
                for subscriptions in self._subscriptions.values()
            )

            return EventBusSnapshot(
                name=self.name,
                running=self._running,
                queue_size=self._queue.qsize(),
                subscription_count=subscription_count,
                history_size=len(self._history),
                dead_letter_size=len(self._dead_letters),
                published_count=self._published_count,
                delivered_count=self._delivered_count,
                failed_count=self._failed_count,
            )

    def history(self) -> tuple[RuntimeEvent, ...]:
        with self._lock:
            return tuple(self._history)

    def dead_letters(self) -> tuple[DeadLetterEvent, ...]:
        with self._lock:
            return tuple(self._dead_letters)

    def clear(self) -> None:
        """
        Clear diagnostic memory.

        Does not remove subscriptions.
        """

        with self._lock:
            self._history.clear()
            self._dead_letters.clear()

    @staticmethod
    def _validate_event(event: RuntimeEvent) -> None:
        if not isinstance(event, RuntimeEvent):
            raise TypeError("EventBus expects RuntimeEvent.")

    def _dispatch_loop(self) -> None:
        while not self._stop_requested.is_set() or not self._queue.empty():
            try:
                _, _, event = self._queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                self._deliver(event)
            finally:
                self._queue.task_done()

    def _deliver(self, event: RuntimeEvent) -> int:
        if event.is_expired:
            self._record_dead_letter(event, reason="event_expired")
            return 0

        with self._lock:
            subscribers = tuple(self._subscriptions.get(event.event_type, ()))

        if not subscribers:
            self._metrics.increment("events.no_subscribers")

            self._logger.debug(
                "event_has_no_subscribers",
                bus=self.name,
                event_id=event.event_id,
                event_type=event.event_type.value,
                correlation_id=event.correlation_id,
            )

            return 0

        delivered = 0

        with self._performance.measure(
            "event_bus.deliver",
            correlation_id=event.correlation_id,
        ):
            with self._tracer.span(
                "event_bus.deliver",
                correlation_id=event.correlation_id,
                event_type=event.event_type.value,
                subscriber_count=len(subscribers),
            ):
                for subscription in subscribers:
                    if self._deliver_to_subscriber(event, subscription):
                        delivered += 1

        return delivered

    def _deliver_to_subscriber(
        self,
        event: RuntimeEvent,
        subscription: EventSubscription,
    ) -> bool:
        try:
            with self._tracer.span(
                "event_bus.subscriber_callback",
                correlation_id=event.correlation_id,
                event_type=event.event_type.value,
                subscriber_name=subscription.subscriber_name,
                subscription_id=subscription.subscription_id,
            ):
                subscription.callback(event)

        except Exception as exc:
            self._record_dead_letter(
                event,
                reason="subscriber_callback_failed",
                subscriber_name=subscription.subscriber_name,
                error=f"{type(exc).__name__}: {exc}",
            )
            return False

        with self._lock:
            self._delivered_count += 1

        self._metrics.increment("events.delivered")

        self._logger.info(
            "event_delivered",
            bus=self.name,
            event_id=event.event_id,
            correlation_id=event.correlation_id,
            event_type=event.event_type.value,
            subscriber_name=subscription.subscriber_name,
            subscription_id=subscription.subscription_id,
        )

        return True

    def _record_dead_letter(
        self,
        event: RuntimeEvent,
        *,
        reason: str,
        subscriber_name: str | None = None,
        error: str | None = None,
    ) -> None:
        dead_letter = DeadLetterEvent(
            event=event,
            reason=reason,
            subscriber_name=subscriber_name,
            error=error,
        )

        with self._lock:
            self._dead_letters.append(dead_letter)
            self._failed_count += 1

        self._metrics.increment("events.dead_lettered")

        self._logger.error(
            "event_dead_lettered",
            bus=self.name,
            event_id=event.event_id,
            correlation_id=event.correlation_id,
            event_type=event.event_type.value,
            reason=reason,
            subscriber_name=subscriber_name,
            error=error,
        )