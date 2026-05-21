from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_session_id() -> str:
    return uuid4().hex


@dataclass(frozen=True, slots=True)
class SessionState:
    """
    Immutable state for one active user interaction session.
    """

    session_id: str = field(default_factory=new_session_id)
    user_id: str | None = None
    active_goal: str | None = None
    active_topic: str | None = None
    started_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    ended_at: datetime | None = None

    @property
    def active(self) -> bool:
        return self.ended_at is None

    def with_goal(self, goal: str | None) -> SessionState:
        clean_goal = goal.strip() if isinstance(goal, str) else None

        return replace(
            self,
            active_goal=clean_goal or None,
            updated_at=utc_now(),
        )

    def with_topic(self, topic: str | None) -> SessionState:
        clean_topic = topic.strip() if isinstance(topic, str) else None

        return replace(
            self,
            active_topic=clean_topic or None,
            updated_at=utc_now(),
        )

    def ended(self) -> SessionState:
        now = utc_now()

        return replace(
            self,
            updated_at=now,
            ended_at=now,
        )