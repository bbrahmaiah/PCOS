from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
from uuid import uuid4

from jarvis.runtime.events.event_models import RuntimeEvent
from jarvis.runtime.shared.enums import EventType


EventCallback = Callable[[RuntimeEvent], None]


@dataclass(frozen=True, slots=True)
class EventSubscription:
    event_type: EventType
    subscriber_name: str
    callback: EventCallback
    subscription_id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        cleaned_name = self.subscriber_name.strip()

        if not cleaned_name:
            raise ValueError("subscriber_name cannot be empty.")

        if not callable(self.callback):
            raise TypeError("callback must be callable.")

        object.__setattr__(self, "subscriber_name", cleaned_name)