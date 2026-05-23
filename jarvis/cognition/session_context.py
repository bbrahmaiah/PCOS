from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from threading import RLock
from typing import Any

from jarvis.cognition.models import (
    CognitionContext,
    CognitionContextItem,
    CognitionFailure,
    CognitionRequest,
    CognitionResponse,
    new_id,
)
from jarvis.runtime.observability.structured_logger import get_logger


class ConversationTurnRole(StrEnum):
    """
    Role of one conversation turn.
    """

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationSessionState(StrEnum):
    """
    Runtime state of a conversation session.
    """

    ACTIVE = "active"
    PAUSED = "paused"
    CLOSED = "closed"


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """
    One bounded conversation turn.

    This stores text-level conversation state, not audio. It is safe to use as
    cognition context later.
    """

    turn_id: str
    role: ConversationTurnRole
    text: str
    request_id: str | None = None
    response_id: str | None = None
    failure_id: str | None = None
    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.turn_id.strip():
            raise ValueError("turn_id cannot be empty.")

        if not self.text.strip():
            raise ValueError("text cannot be empty.")


@dataclass(frozen=True, slots=True)
class ConversationSessionSnapshot:
    """
    Observable snapshot for one conversation session.
    """

    session_id: str
    state: ConversationSessionState
    turn_count: int
    user_turn_count: int
    assistant_turn_count: int
    failure_count: int
    active_topic: str | None
    last_user_text: str | None
    last_assistant_text: str | None
    last_request_id: str | None
    last_response_id: str | None
    last_failure_id: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ConversationSessionConfig:
    """
    Configuration for ConversationSessionStore.
    """

    session_id: str = field(default_factory=new_id)
    max_turns: int = 24
    max_context_items: int = 10
    max_item_chars: int = 600
    active_topic_max_chars: int = 120

    def validate(self) -> None:
        if not self.session_id.strip():
            raise ValueError("session_id cannot be empty.")

        if self.max_turns <= 0:
            raise ValueError("max_turns must be greater than zero.")

        if self.max_context_items <= 0:
            raise ValueError("max_context_items must be greater than zero.")

        if self.max_item_chars <= 0:
            raise ValueError("max_item_chars must be greater than zero.")

        if self.active_topic_max_chars <= 0:
            raise ValueError("active_topic_max_chars must be greater than zero.")


class ConversationSessionStore:
    """
    Thread-safe short-term conversation session store.

    Responsibilities:
    - track recent user and assistant turns
    - keep bounded short-term session context
    - expose CognitionContext for prompt construction
    - track active topic and recent failures
    - prevent unbounded memory growth

    Non-responsibilities:
    - no long-term memory persistence
    - no vector search
    - no LLM calls
    - no tool execution
    - no audio/STT/TTS internals
    """

    def __init__(
        self,
        *,
        config: ConversationSessionConfig | None = None,
    ) -> None:
        self._config = config or ConversationSessionConfig()
        self._config.validate()

        self._lock = RLock()
        self._logger = get_logger("cognition.session_context")

        self._state = ConversationSessionState.ACTIVE
        self._turns: list[ConversationTurn] = []
        self._failure_count = 0
        self._active_topic: str | None = None
        self._last_request_id: str | None = None
        self._last_response_id: str | None = None
        self._last_failure_id: str | None = None
        self._created_at = utc_now()
        self._updated_at = self._created_at

    @property
    def session_id(self) -> str:
        return self._config.session_id

    @property
    def state(self) -> ConversationSessionState:
        with self._lock:
            return self._state

    def add_user_request(
        self,
        request: CognitionRequest,
    ) -> ConversationTurn:
        """
        Add one user request as a session turn.
        """

        turn = ConversationTurn(
            turn_id=request.turn_id or new_id(),
            role=ConversationTurnRole.USER,
            text=request.text,
            request_id=request.request_id,
            metadata={
                "source": request.source,
                "transcript_id": request.transcript_id,
                "correlation_id": request.correlation_id,
                **request.metadata,
            },
        )

        self._add_turn(turn)
        self._last_request_id = request.request_id
        self._active_topic = self._derive_topic(request.text)
        self._touch()

        self._logger.info(
            "conversation_user_turn_added",
            session_id=self.session_id,
            request_id=request.request_id,
            turn_id=turn.turn_id,
        )

        return turn

    def add_assistant_response(
        self,
        response: CognitionResponse,
    ) -> ConversationTurn:
        """
        Add one assistant response as a session turn.
        """

        turn = ConversationTurn(
            turn_id=new_id(),
            role=ConversationTurnRole.ASSISTANT,
            text=response.text,
            request_id=response.request_id,
            response_id=response.response_id,
            metadata={
                "response_kind": response.kind.value,
                "confidence": response.confidence,
                **response.metadata,
            },
        )

        self._add_turn(turn)
        self._last_response_id = response.response_id
        self._touch()

        self._logger.info(
            "conversation_assistant_turn_added",
            session_id=self.session_id,
            request_id=response.request_id,
            response_id=response.response_id,
            turn_id=turn.turn_id,
        )

        return turn

    def add_failure(
        self,
        failure: CognitionFailure,
    ) -> ConversationTurn:
        """
        Add one cognition failure as a system turn.
        """

        turn = ConversationTurn(
            turn_id=new_id(),
            role=ConversationTurnRole.SYSTEM,
            text=f"Cognition failure: {failure.message}",
            request_id=failure.request_id,
            failure_id=failure.failure_id,
            metadata={
                "failure_kind": failure.kind.value,
                "recoverable": failure.recoverable,
                **failure.metadata,
            },
        )

        self._add_turn(turn)
        self._failure_count += 1
        self._last_failure_id = failure.failure_id
        self._touch()

        self._logger.info(
            "conversation_failure_turn_added",
            session_id=self.session_id,
            request_id=failure.request_id,
            failure_id=failure.failure_id,
            turn_id=turn.turn_id,
        )

        return turn

    def build_context(
        self,
        *,
        request: CognitionRequest | None = None,
    ) -> CognitionContext:
        """
        Build CognitionContext from recent session turns.

        This is what later LocalLLMAdapter can include in prompts.
        """

        with self._lock:
            recent_turns = self._turns[-self._config.max_context_items :]
            items = tuple(
                self._context_item_for_turn(turn)
                for turn in recent_turns
            )

            session_turn_id = (
                request.turn_id
                if request is not None and request.turn_id is not None
                else None
            )

            return CognitionContext(
                session_id=self.session_id,
                turn_id=session_turn_id,
                items=items,
                metadata={
                    "session_state": self._state.value,
                    "active_topic": self._active_topic,
                    "turn_count": len(self._turns),
                    "failure_count": self._failure_count,
                },
            )

    def enrich_request(
        self,
        request: CognitionRequest,
    ) -> CognitionRequest:
        """
        Return a copy of request with session context attached.
        """

        context = self.build_context(request=request)

        return request.model_copy(
            update={
                "context": context,
                "metadata": {
                    **request.metadata,
                    "session_id": self.session_id,
                    "active_topic": self._active_topic,
                },
            }
        )

    def pause(self) -> None:
        with self._lock:
            self._state = ConversationSessionState.PAUSED
            self._touch()

        self._logger.info("conversation_session_paused", session_id=self.session_id)

    def resume(self) -> None:
        with self._lock:
            self._state = ConversationSessionState.ACTIVE
            self._touch()

        self._logger.info("conversation_session_resumed", session_id=self.session_id)

    def close(self) -> None:
        with self._lock:
            self._state = ConversationSessionState.CLOSED
            self._touch()

        self._logger.info("conversation_session_closed", session_id=self.session_id)

    def reset(self) -> None:
        """
        Clear session turns and diagnostics while keeping same session id.
        """

        with self._lock:
            self._state = ConversationSessionState.ACTIVE
            self._turns.clear()
            self._failure_count = 0
            self._active_topic = None
            self._last_request_id = None
            self._last_response_id = None
            self._last_failure_id = None
            self._touch()

        self._logger.info("conversation_session_reset", session_id=self.session_id)

    def turns(self) -> tuple[ConversationTurn, ...]:
        """
        Return immutable copy of recent turns.
        """

        with self._lock:
            return tuple(self._turns)

    def snapshot(self) -> ConversationSessionSnapshot:
        """
        Return session diagnostics.
        """

        with self._lock:
            last_user = self._last_turn_by_role(ConversationTurnRole.USER)
            last_assistant = self._last_turn_by_role(
                ConversationTurnRole.ASSISTANT
            )

            return ConversationSessionSnapshot(
                session_id=self.session_id,
                state=self._state,
                turn_count=len(self._turns),
                user_turn_count=self._count_role(ConversationTurnRole.USER),
                assistant_turn_count=self._count_role(
                    ConversationTurnRole.ASSISTANT
                ),
                failure_count=self._failure_count,
                active_topic=self._active_topic,
                last_user_text=last_user.text if last_user else None,
                last_assistant_text=(
                    last_assistant.text if last_assistant else None
                ),
                last_request_id=self._last_request_id,
                last_response_id=self._last_response_id,
                last_failure_id=self._last_failure_id,
                created_at=self._created_at,
                updated_at=self._updated_at,
            )

    def _add_turn(self, turn: ConversationTurn) -> None:
        turn.validate()

        with self._lock:
            self._turns.append(turn)

            if len(self._turns) > self._config.max_turns:
                overflow = len(self._turns) - self._config.max_turns
                del self._turns[:overflow]

            self._touch()

    def _context_item_for_turn(
        self,
        turn: ConversationTurn,
    ) -> CognitionContextItem:
        text = self._bounded_text(turn.text, self._config.max_item_chars)

        return CognitionContextItem(
            kind=f"conversation_{turn.role.value}",
            text=text,
            source="conversation_session",
            metadata={
                "turn_id": turn.turn_id,
                "request_id": turn.request_id,
                "response_id": turn.response_id,
                "failure_id": turn.failure_id,
            },
        )

    def _derive_topic(self, text: str) -> str:
        clean_text = " ".join(text.split())

        return self._bounded_text(
            clean_text,
            self._config.active_topic_max_chars,
        )

    @staticmethod
    def _bounded_text(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text

        if max_chars <= 3:
            return text[:max_chars]

        return f"{text[: max_chars - 3].rstrip()}..."

    def _last_turn_by_role(
        self,
        role: ConversationTurnRole,
    ) -> ConversationTurn | None:
        for turn in reversed(self._turns):
            if turn.role == role:
                return turn

        return None

    def _count_role(self, role: ConversationTurnRole) -> int:
        return sum(1 for turn in self._turns if turn.role == role)

    def _touch(self) -> None:
        self._updated_at = utc_now()