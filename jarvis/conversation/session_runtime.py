from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from threading import RLock

from pydantic import Field, field_validator

from jarvis.conversation.models import (
    ConversationMode,
    ConversationModel,
    new_conversation_id,
    utc_now,
)
from jarvis.conversation.state_machine import ConversationState
from jarvis.runtime.observability.structured_logger import get_logger


class ConversationTurnRole(StrEnum):
    """
    Speaker role for a conversation turn.
    """

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationContinuityStatus(StrEnum):
    """
    Current continuity status of the live session.
    """

    EMPTY = "empty"
    ACTIVE = "active"
    WAITING_FOR_FOLLOW_UP = "waiting_for_follow_up"
    PAUSED = "paused"
    INTERRUPTED = "interrupted"
    CLOSED = "closed"


class ConversationTopicShift(StrEnum):
    """
    Topic-shift estimate after a new turn is added.
    """

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ConversationFollowUpExpectation(StrEnum):
    """
    Whether the session expects the user to continue.
    """

    NONE = "none"
    LIKELY = "likely"
    REQUIRED = "required"


class ConversationSessionTurn(ConversationModel):
    """
    One short-term conversational turn.

    This is not long-term memory. This is the live conversation scratchpad.
    """

    turn_id: str = Field(default_factory=new_conversation_id)
    role: ConversationTurnRole
    text: str
    topic: str | None = None
    objective: str | None = None
    conversation_mode: ConversationMode = ConversationMode.UNKNOWN
    state: ConversationState | None = None
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("turn_id", "text")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = " ".join(value.strip().split())

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("topic", "objective")
    @classmethod
    def _clean_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = " ".join(value.strip().split())

        return cleaned or None

    @property
    def char_count(self) -> int:
        return len(self.text)


class ConversationSessionSnapshotModel(ConversationModel):
    """
    Immutable session snapshot for cognition/orchestration.
    """

    session_id: str
    status: ConversationContinuityStatus
    active_topic: str | None = None
    current_objective: str | None = None
    follow_up_expectation: ConversationFollowUpExpectation
    turns: tuple[ConversationSessionTurn, ...] = ()
    topic_shift: ConversationTopicShift = ConversationTopicShift.NONE
    summary: str | None = None
    temporary_context: dict[str, object] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("session_id")
    @classmethod
    def _session_id_required(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("session_id cannot be empty.")

        return cleaned

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def last_user_turn(self) -> ConversationSessionTurn | None:
        for turn in reversed(self.turns):
            if turn.role == ConversationTurnRole.USER:
                return turn

        return None

    @property
    def last_assistant_turn(self) -> ConversationSessionTurn | None:
        for turn in reversed(self.turns):
            if turn.role == ConversationTurnRole.ASSISTANT:
                return turn

        return None

    def as_context_block(self) -> str:
        """
        Convert session continuity into a compact cognition-ready context block.
        """

        lines = [
            "Conversation session continuity:",
            f"- status: {self.status.value}",
            f"- active_topic: {self.active_topic or 'unknown'}",
            f"- current_objective: {self.current_objective or 'unknown'}",
            f"- follow_up: {self.follow_up_expectation.value}",
        ]

        if self.summary:
            lines.append(f"- summary: {self.summary}")

        if self.turns:
            lines.append("- recent turns:")

            for turn in self.turns:
                lines.append(f"  - {turn.role.value}: {turn.text}")

        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ConversationSessionRuntimeConfig:
    """
    Configuration for live session continuity.

    Keep this bounded. Long-term memory belongs to Phase 4 Memory Runtime.
    """

    name: str = "conversation_session_runtime"
    max_turns: int = 12
    max_summary_chars: int = 900
    topic_shift_threshold: float = 0.72

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.max_turns <= 0:
            raise ValueError("max_turns must be greater than zero.")

        if self.max_summary_chars <= 0:
            raise ValueError("max_summary_chars must be greater than zero.")

        if self.topic_shift_threshold <= 0.0 or self.topic_shift_threshold > 1.0:
            raise ValueError("topic_shift_threshold must be in (0, 1].")


@dataclass(frozen=True, slots=True)
class ConversationSessionRuntimeSnapshot:
    """
    Observable diagnostics for ConversationSessionRuntime.
    """

    name: str
    session_id: str
    status: ConversationContinuityStatus
    turn_count: int
    user_turn_count: int
    assistant_turn_count: int
    update_count: int
    compaction_count: int
    active_topic: str | None
    current_objective: str | None
    follow_up_expectation: ConversationFollowUpExpectation
    last_topic_shift: ConversationTopicShift
    last_turn_id: str | None
    last_error: str | None


class ConversationSessionRuntime:
    """
    Short-term session continuity runtime for real conversation.

    Responsibilities:
    - track active topic
    - track current user objective
    - keep recent user/assistant turns
    - detect rough topic shifts
    - track follow-up expectation
    - compact bounded short-term context
    - expose cognition-ready context

    Non-responsibilities:
    - no long-term memory persistence
    - no vector retrieval
    - no LLM calls
    - no TTS/STT
    - no tool execution
    """

    _FOLLOW_UP_MARKERS = {
        "tell me more",
        "explain more",
        "continue",
        "go on",
        "what next",
        "next",
        "why",
        "how",
        "then",
        "and",
        "also",
    }

    _QUESTION_STARTS = (
        "what ",
        "why ",
        "how ",
        "when ",
        "where ",
        "can you ",
        "could you ",
        "should i ",
        "do you ",
        "is it ",
        "are we ",
    )

    def __init__(
        self,
        *,
        config: ConversationSessionRuntimeConfig | None = None,
        session_id: str | None = None,
    ) -> None:
        self._config = config or ConversationSessionRuntimeConfig()
        self._config.validate()

        self._session_id = session_id or new_conversation_id()
        self._lock = RLock()
        self._logger = get_logger("conversation.session_runtime")

        self._status = ConversationContinuityStatus.EMPTY
        self._active_topic: str | None = None
        self._current_objective: str | None = None
        self._follow_up_expectation = ConversationFollowUpExpectation.NONE
        self._turns: list[ConversationSessionTurn] = []
        self._summary: str | None = None
        self._temporary_context: dict[str, object] = {}
        self._created_at = utc_now()
        self._updated_at = self._created_at

        self._update_count = 0
        self._compaction_count = 0
        self._last_topic_shift = ConversationTopicShift.NONE
        self._last_turn_id: str | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def session_id(self) -> str:
        return self._session_id

    def add_user_turn(
        self,
        text: str,
        *,
        topic: str | None = None,
        objective: str | None = None,
        conversation_mode: ConversationMode = ConversationMode.UNKNOWN,
        state: ConversationState | None = None,
        metadata: dict[str, object] | None = None,
    ) -> ConversationSessionSnapshotModel:
        """
        Add a user turn and update live continuity.
        """

        turn = ConversationSessionTurn(
            role=ConversationTurnRole.USER,
            text=text,
            topic=topic,
            objective=objective,
            conversation_mode=conversation_mode,
            state=state,
            metadata=metadata or {},
        )

        return self._add_turn(turn)

    def add_assistant_turn(
        self,
        text: str,
        *,
        topic: str | None = None,
        objective: str | None = None,
        conversation_mode: ConversationMode = ConversationMode.UNKNOWN,
        state: ConversationState | None = None,
        expects_follow_up: bool = False,
        metadata: dict[str, object] | None = None,
    ) -> ConversationSessionSnapshotModel:
        """
        Add an assistant turn and update follow-up continuity.
        """

        turn = ConversationSessionTurn(
            role=ConversationTurnRole.ASSISTANT,
            text=text,
            topic=topic,
            objective=objective,
            conversation_mode=conversation_mode,
            state=state,
            metadata={
                **(metadata or {}),
                "expects_follow_up": expects_follow_up,
            },
        )

        return self._add_turn(turn)

    def mark_interrupted(
        self,
        *,
        reason: str = "conversation interrupted",
    ) -> ConversationSessionSnapshotModel:
        """
        Mark the live session as interrupted without deleting continuity.
        """

        with self._lock:
            self._status = ConversationContinuityStatus.INTERRUPTED
            self._updated_at = utc_now()
            self._temporary_context["last_interrupt_reason"] = reason
            self._update_count += 1

        self._logger.info(
            "conversation_session_interrupted",
            runtime=self.name,
            session_id=self.session_id,
            reason=reason,
        )

        return self.snapshot_model()

    def pause(
        self,
        *,
        reason: str = "conversation paused",
    ) -> ConversationSessionSnapshotModel:
        """
        Pause the session while preserving continuity.
        """

        with self._lock:
            self._status = ConversationContinuityStatus.PAUSED
            self._updated_at = utc_now()
            self._temporary_context["pause_reason"] = reason
            self._update_count += 1

        return self.snapshot_model()

    def close(
        self,
        *,
        reason: str = "conversation closed",
    ) -> ConversationSessionSnapshotModel:
        """
        Close the session.
        """

        with self._lock:
            self._status = ConversationContinuityStatus.CLOSED
            self._updated_at = utc_now()
            self._temporary_context["close_reason"] = reason
            self._update_count += 1

        return self.snapshot_model()

    def set_temporary_context(
        self,
        key: str,
        value: object,
    ) -> ConversationSessionSnapshotModel:
        """
        Add or update temporary session context.
        """

        cleaned = key.strip()

        if not cleaned:
            raise ValueError("temporary context key cannot be empty.")

        with self._lock:
            self._temporary_context[cleaned] = value
            self._updated_at = utc_now()
            self._update_count += 1

        return self.snapshot_model()

    def snapshot_model(self) -> ConversationSessionSnapshotModel:
        """
        Return a cognition-ready immutable snapshot.
        """

        with self._lock:
            return ConversationSessionSnapshotModel(
                session_id=self.session_id,
                status=self._status,
                active_topic=self._active_topic,
                current_objective=self._current_objective,
                follow_up_expectation=self._follow_up_expectation,
                turns=tuple(self._turns),
                topic_shift=self._last_topic_shift,
                summary=self._summary,
                temporary_context=dict(self._temporary_context),
                created_at=self._created_at,
                updated_at=self._updated_at,
                metadata={
                    "runtime": self.name,
                    "update_count": self._update_count,
                    "compaction_count": self._compaction_count,
                },
            )

    def snapshot(self) -> ConversationSessionRuntimeSnapshot:
        """
        Return observable diagnostics.
        """

        with self._lock:
            user_turn_count = sum(
                1 for turn in self._turns if turn.role == ConversationTurnRole.USER
            )
            assistant_turn_count = sum(
                1
                for turn in self._turns
                if turn.role == ConversationTurnRole.ASSISTANT
            )

            return ConversationSessionRuntimeSnapshot(
                name=self.name,
                session_id=self.session_id,
                status=self._status,
                turn_count=len(self._turns),
                user_turn_count=user_turn_count,
                assistant_turn_count=assistant_turn_count,
                update_count=self._update_count,
                compaction_count=self._compaction_count,
                active_topic=self._active_topic,
                current_objective=self._current_objective,
                follow_up_expectation=self._follow_up_expectation,
                last_topic_shift=self._last_topic_shift,
                last_turn_id=self._last_turn_id,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Reset this session runtime.
        """

        with self._lock:
            self._status = ConversationContinuityStatus.EMPTY
            self._active_topic = None
            self._current_objective = None
            self._follow_up_expectation = ConversationFollowUpExpectation.NONE
            self._turns.clear()
            self._summary = None
            self._temporary_context.clear()
            self._created_at = utc_now()
            self._updated_at = self._created_at
            self._update_count = 0
            self._compaction_count = 0
            self._last_topic_shift = ConversationTopicShift.NONE
            self._last_turn_id = None
            self._last_error = None

        self._logger.info(
            "conversation_session_runtime_reset",
            runtime=self.name,
            session_id=self.session_id,
        )

    def _add_turn(
        self,
        turn: ConversationSessionTurn,
    ) -> ConversationSessionSnapshotModel:
        with self._lock:
            previous_topic = self._active_topic
            inferred_topic = turn.topic or self._infer_topic(turn.text)
            inferred_objective = turn.objective or self._infer_objective(turn.text)

            self._last_topic_shift = self._topic_shift(
                previous_topic=previous_topic,
                next_topic=inferred_topic,
            )
            self._active_topic = inferred_topic or self._active_topic
            self._current_objective = inferred_objective or self._current_objective
            self._turns.append(turn)
            self._last_turn_id = turn.turn_id
            self._updated_at = utc_now()
            self._update_count += 1

            self._follow_up_expectation = self._follow_up_expectation_for_turn(turn)
            self._status = self._status_after_turn(turn)

            if len(self._turns) > self._config.max_turns:
                self._compact_locked()

        self._logger.info(
            "conversation_session_turn_added",
            runtime=self.name,
            session_id=self.session_id,
            turn_id=turn.turn_id,
            role=turn.role.value,
            status=self.snapshot().status.value,
            active_topic=self.snapshot().active_topic,
        )

        return self.snapshot_model()

    def _status_after_turn(
        self,
        turn: ConversationSessionTurn,
    ) -> ConversationContinuityStatus:
        if turn.role == ConversationTurnRole.ASSISTANT and (
            self._follow_up_expectation != ConversationFollowUpExpectation.NONE
        ):
            return ConversationContinuityStatus.WAITING_FOR_FOLLOW_UP

        return ConversationContinuityStatus.ACTIVE

    def _follow_up_expectation_for_turn(
        self,
        turn: ConversationSessionTurn,
    ) -> ConversationFollowUpExpectation:
        text = turn.text.casefold()

        if turn.role == ConversationTurnRole.ASSISTANT:
            if bool(turn.metadata.get("expects_follow_up")):
                return ConversationFollowUpExpectation.REQUIRED

            if text.endswith("?"):
                return ConversationFollowUpExpectation.LIKELY

            return ConversationFollowUpExpectation.NONE

        if text in self._FOLLOW_UP_MARKERS:
            return ConversationFollowUpExpectation.LIKELY

        if text.startswith(self._QUESTION_STARTS):
            return ConversationFollowUpExpectation.LIKELY

        return ConversationFollowUpExpectation.NONE

    def _compact_locked(self) -> None:
        overflow = len(self._turns) - self._config.max_turns

        if overflow <= 0:
            return

        removed = self._turns[:overflow]
        self._turns = self._turns[overflow:]
        removed_text = " ".join(
            f"{turn.role.value}: {turn.text}" for turn in removed
        )
        self._summary = self._merge_summary(self._summary, removed_text)
        self._compaction_count += 1

    def _merge_summary(
        self,
        existing: str | None,
        new_text: str,
    ) -> str:
        joined = f"{existing or ''} {new_text}".strip()

        if len(joined) <= self._config.max_summary_chars:
            return joined

        return joined[-self._config.max_summary_chars :].strip()

    def _infer_topic(self, text: str) -> str | None:
        normalized = text.casefold()
        words = [
            word.strip(".,?!:;()[]{}\"'")
            for word in normalized.split()
            if len(word.strip(".,?!:;()[]{}\"'")) >= 4
        ]
        stop_words = {
            "jarvis",
            "please",
            "could",
            "would",
            "should",
            "about",
            "explain",
            "what",
            "where",
            "when",
            "this",
            "that",
            "with",
            "from",
            "into",
            "want",
        }
        meaningful = [word for word in words if word not in stop_words]

        if not meaningful:
            return None

        return " ".join(meaningful[:4])

    def _infer_objective(self, text: str) -> str | None:
        normalized = text.casefold()

        if "build" in normalized:
            return "build"

        if "fix" in normalized or "error" in normalized:
            return "debug"

        if "explain" in normalized or "why" in normalized:
            return "understand"

        if "test" in normalized:
            return "validate"

        return None

    def _topic_shift(
        self,
        *,
        previous_topic: str | None,
        next_topic: str | None,
    ) -> ConversationTopicShift:
        if not previous_topic or not next_topic:
            return ConversationTopicShift.NONE

        previous = set(previous_topic.split())
        current = set(next_topic.split())

        if not previous or not current:
            return ConversationTopicShift.NONE

        overlap = len(previous & current) / max(len(previous | current), 1)

        if overlap >= self._config.topic_shift_threshold:
            return ConversationTopicShift.LOW

        if overlap >= 0.35:
            return ConversationTopicShift.MEDIUM

        return ConversationTopicShift.HIGH