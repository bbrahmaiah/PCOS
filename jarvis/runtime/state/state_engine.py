from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any

from jarvis.runtime.events import EventBus, RuntimeEvent
from jarvis.runtime.observability.performance_monitor import get_performance_monitor
from jarvis.runtime.observability.structured_logger import get_logger
from jarvis.runtime.shared.enums import (
    EventCategory,
    EventType,
    RuntimeStatus,
    SystemMode,
)
from jarvis.runtime.state.global_context import ContextSnapshot, GlobalContext
from jarvis.runtime.state.runtime_state import RuntimeState
from jarvis.runtime.state.session_state import SessionState


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    """
    Full immutable snapshot of the runtime state engine.
    """

    runtime: RuntimeState
    session: SessionState | None
    context: ContextSnapshot


class StateEngine:
    """
    Thread-safe state engine for JARVIS.
    """

    def __init__(self, *, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self._lock = RLock()

        self._runtime = RuntimeState()
        self._session: SessionState | None = None
        self._context = GlobalContext()

        self._logger = get_logger("state.engine")
        self._performance = get_performance_monitor()

    def runtime_state(self) -> RuntimeState:
        with self._lock:
            return self._runtime

    def session_state(self) -> SessionState | None:
        with self._lock:
            return self._session

    def context(self) -> GlobalContext:
        return self._context

    def snapshot(self) -> StateSnapshot:
        with self._lock:
            return StateSnapshot(
                runtime=self._runtime,
                session=self._session,
                context=self._context.snapshot(),
            )

    def set_runtime_status(
        self,
        status: RuntimeStatus,
        *,
        last_error: str | None = None,
    ) -> RuntimeState:
        with self._performance.measure("state.set_runtime_status"):
            with self._lock:
                self._runtime = self._runtime.with_status(
                    status,
                    last_error=last_error,
                )
                state = self._runtime

            self._emit_state_event(
                EventType.STATE_UPDATED,
                payload={
                    "field": "runtime.status",
                    "status": state.status.value,
                    "last_error": state.last_error,
                },
            )

            self._logger.info(
                "runtime_status_updated",
                status=state.status.value,
                last_error=state.last_error,
            )

            return state

    def set_system_mode(self, mode: SystemMode) -> RuntimeState:
        with self._performance.measure("state.set_system_mode"):
            with self._lock:
                self._runtime = self._runtime.with_mode(mode)
                state = self._runtime

            self._emit_state_event(
                EventType.STATE_UPDATED,
                payload={
                    "field": "runtime.mode",
                    "mode": state.mode.value,
                },
            )

            self._logger.info(
                "system_mode_updated",
                mode=state.mode.value,
            )

            return state

    def start_session(
        self,
        *,
        user_id: str | None = None,
        active_goal: str | None = None,
        active_topic: str | None = None,
    ) -> SessionState:
        with self._performance.measure("state.start_session"):
            session = SessionState(
                user_id=self._clean_optional(user_id),
                active_goal=self._clean_optional(active_goal),
                active_topic=self._clean_optional(active_topic),
            )

            with self._lock:
                self._session = session

            self._emit_state_event(
                EventType.SESSION_STARTED,
                payload={
                    "session_id": session.session_id,
                    "user_id": session.user_id,
                    "active_goal": session.active_goal,
                    "active_topic": session.active_topic,
                },
            )

            self._logger.info(
                "session_started",
                session_id=session.session_id,
                user_id=session.user_id,
            )

            return session

    def end_session(self) -> SessionState | None:
        with self._performance.measure("state.end_session"):
            with self._lock:
                if self._session is None:
                    return None

                ended_session = self._session.ended()
                self._session = ended_session

            self._emit_state_event(
                EventType.SESSION_ENDED,
                payload={
                    "session_id": ended_session.session_id,
                    "ended_at": ended_session.ended_at.isoformat()
                    if ended_session.ended_at
                    else None,
                },
            )

            self._logger.info(
                "session_ended",
                session_id=ended_session.session_id,
            )

            return ended_session

    def update_session_goal(self, goal: str | None) -> SessionState:
        with self._performance.measure("state.update_session_goal"):
            with self._lock:
                if self._session is None:
                    raise RuntimeError("No active session exists.")

                self._session = self._session.with_goal(goal)
                session = self._session

            self._emit_state_event(
                EventType.STATE_UPDATED,
                payload={
                    "field": "session.active_goal",
                    "session_id": session.session_id,
                    "active_goal": session.active_goal,
                },
            )

            return session

    def update_session_topic(self, topic: str | None) -> SessionState:
        with self._performance.measure("state.update_session_topic"):
            with self._lock:
                if self._session is None:
                    raise RuntimeError("No active session exists.")

                self._session = self._session.with_topic(topic)
                session = self._session

            self._emit_state_event(
                EventType.STATE_UPDATED,
                payload={
                    "field": "session.active_topic",
                    "session_id": session.session_id,
                    "active_topic": session.active_topic,
                },
            )

            return session

    def set_context(self, key: str, value: Any) -> None:
        with self._performance.measure("state.set_context"):
            clean_key = key.strip()
            self._context.set(clean_key, value)

            self._emit_state_event(
                EventType.CONTEXT_UPDATED,
                payload={
                    "operation": "set",
                    "key": clean_key,
                },
            )

    def update_context(self, values: dict[str, Any]) -> None:
        with self._performance.measure("state.update_context"):
            self._context.update_many(values)

            self._emit_state_event(
                EventType.CONTEXT_UPDATED,
                payload={
                    "operation": "update_many",
                    "keys": sorted(values),
                },
            )

    def get_context(self, key: str, default: Any = None) -> Any:
        return self._context.get(key, default)

    def delete_context(self, key: str) -> bool:
        with self._performance.measure("state.delete_context"):
            clean_key = key.strip()
            removed = self._context.delete(clean_key)

            self._emit_state_event(
                EventType.CONTEXT_UPDATED,
                payload={
                    "operation": "delete",
                    "key": clean_key,
                    "removed": removed,
                },
            )

            return removed

    def clear_context(self) -> None:
        with self._performance.measure("state.clear_context"):
            self._context.clear()

            self._emit_state_event(
                EventType.CONTEXT_UPDATED,
                payload={
                    "operation": "clear",
                },
            )

    def _emit_state_event(
        self,
        event_type: EventType,
        *,
        payload: dict[str, object],
    ) -> None:
        event = RuntimeEvent(
            event_type=event_type,
            category=EventCategory.STATE,
            source="state_engine",
            payload=payload,
        )

        self.event_bus.publish(event)

    @staticmethod
    def _clean_optional(value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None