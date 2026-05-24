from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from threading import RLock

from pydantic import Field, field_validator

from jarvis.conversation.models import (
    ConversationModel,
    TurnUrgency,
    new_conversation_id,
    utc_now,
)
from jarvis.conversation.session_runtime import (
    ConversationFollowUpExpectation,
    ConversationSessionSnapshotModel,
)
from jarvis.conversation.state_machine import ConversationState
from jarvis.runtime.observability.structured_logger import get_logger


class AttentionSignalKind(StrEnum):
    """
    Input signal type for attention routing.
    """

    USER_TURN = "user_turn"
    ASSISTANT_RESPONSE = "assistant_response"
    INTERRUPTION = "interruption"
    FOLLOW_UP = "follow_up"
    BACKGROUND_TASK = "background_task"
    TOOL_RESULT = "tool_result"
    ENVIRONMENT_CHANGE = "environment_change"
    SYSTEM_NOTICE = "system_notice"


class AttentionTargetKind(StrEnum):
    """
    Type of target that can hold attention.
    """

    CONVERSATION = "conversation"
    USER_OBJECTIVE = "user_objective"
    INTERRUPTION = "interruption"
    TOOL_TASK = "tool_task"
    BACKGROUND_REASONING = "background_reasoning"
    ENVIRONMENT = "environment"
    SYSTEM = "system"


class AttentionPriority(StrEnum):
    """
    Priority level for attention.
    """

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class AttentionDisposition(StrEnum):
    """
    Decision about what to do with a signal.
    """

    IGNORE = "ignore"
    BACKGROUND = "background"
    MONITOR = "monitor"
    FOCUS = "focus"
    INTERRUPT = "interrupt"


class AttentionTargetStatus(StrEnum):
    """
    Runtime status for an attention target.
    """

    ACTIVE = "active"
    BACKGROUNDED = "backgrounded"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class AttentionSignal(ConversationModel):
    """
    One signal entering the attention runtime.

    This is intentionally generic so future workers can submit signals from
    conversation, tools, environment, memory, and orchestration.
    """

    signal_id: str = Field(default_factory=new_conversation_id)
    kind: AttentionSignalKind
    text: str
    priority: AttentionPriority = AttentionPriority.NORMAL
    urgency: TurnUrgency = TurnUrgency.NORMAL
    source: str = "conversation"
    state: ConversationState | None = None
    target_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("signal_id", "text", "source")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = " ".join(value.strip().split())

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("target_id")
    @classmethod
    def _clean_optional_target_id(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None


class AttentionTarget(ConversationModel):
    """
    A focusable runtime target.

    Examples:
    - current conversation
    - active user objective
    - interruption
    - tool task
    - background reasoning job
    """

    target_id: str = Field(default_factory=new_conversation_id)
    kind: AttentionTargetKind
    label: str
    priority: AttentionPriority = AttentionPriority.NORMAL
    status: AttentionTargetStatus = AttentionTargetStatus.ACTIVE
    weight: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("target_id", "label")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = " ".join(value.strip().split())

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class AttentionDecision(ConversationModel):
    """
    Decision produced by AttentionRuntime.
    """

    decision_id: str = Field(default_factory=new_conversation_id)
    signal: AttentionSignal
    disposition: AttentionDisposition
    selected_target: AttentionTarget | None = None
    previous_focus_id: str | None = None
    current_focus_id: str | None = None
    should_interrupt: bool = False
    should_background_existing: bool = False
    should_start_cognition: bool = False
    should_monitor: bool = False
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    decided_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("decision_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = " ".join(value.strip().split())

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class AttentionRuntimeConfig:
    """
    Configuration for the AttentionRuntime.
    """

    name: str = "attention_runtime"
    focus_threshold: float = 0.62
    interrupt_threshold: float = 0.86
    background_threshold: float = 0.35
    max_targets: int = 16

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.max_targets <= 0:
            raise ValueError("max_targets must be greater than zero.")

        thresholds = (
            self.focus_threshold,
            self.interrupt_threshold,
            self.background_threshold,
        )

        if any(value < 0.0 or value > 1.0 for value in thresholds):
            raise ValueError("thresholds must be between 0 and 1.")

        if self.interrupt_threshold < self.focus_threshold:
            raise ValueError("interrupt_threshold must be >= focus_threshold.")


@dataclass(frozen=True, slots=True)
class AttentionRuntimeSnapshot:
    """
    Observable diagnostics for AttentionRuntime.
    """

    name: str
    target_count: int
    signal_count: int
    decision_count: int
    focus_change_count: int
    interrupt_count: int
    background_count: int
    current_focus_id: str | None
    current_focus_label: str | None
    last_disposition: AttentionDisposition | None
    last_signal_kind: AttentionSignalKind | None
    last_error: str | None


class AttentionRuntime:
    """
    Focus-management runtime for continuous conversation.

    Responsibilities:
    - maintain active conversational focus
    - convert signals into focus/background/interrupt decisions
    - preserve interaction persistence
    - weight current user objectives higher than noise
    - support future tools/environment signals
    - expose cognition-ready attention context

    Non-responsibilities:
    - no LLM calls
    - no tool execution
    - no memory persistence
    - no audio/TTS/STT implementation
    """

    def __init__(
        self,
        *,
        config: AttentionRuntimeConfig | None = None,
    ) -> None:
        self._config = config or AttentionRuntimeConfig()
        self._config.validate()

        self._lock = RLock()
        self._logger = get_logger("conversation.attention_runtime")

        self._targets: dict[str, AttentionTarget] = {}
        self._current_focus_id: str | None = None
        self._signal_count = 0
        self._decision_count = 0
        self._focus_change_count = 0
        self._interrupt_count = 0
        self._background_count = 0
        self._last_disposition: AttentionDisposition | None = None
        self._last_signal_kind: AttentionSignalKind | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def submit_signal(self, signal: AttentionSignal) -> AttentionDecision:
        """
        Submit one attention signal and receive a routing decision.
        """

        with self._lock:
            self._signal_count += 1
            self._last_signal_kind = signal.kind
            self._last_error = None

        try:
            decision = self._decide(signal)
            self._record_decision(decision)

            self._logger.info(
                "attention_signal_decided",
                runtime=self.name,
                signal_id=signal.signal_id,
                signal_kind=signal.kind.value,
                disposition=decision.disposition.value,
                current_focus_id=decision.current_focus_id,
                should_interrupt=decision.should_interrupt,
            )

            return decision

        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"

            raise

    def update_from_session(
        self,
        session: ConversationSessionSnapshotModel,
    ) -> AttentionDecision:
        """
        Update attention from live session continuity.
        """

        text = session.active_topic or session.current_objective or "conversation"
        priority = (
            AttentionPriority.HIGH
            if session.follow_up_expectation
            != ConversationFollowUpExpectation.NONE
            else AttentionPriority.NORMAL
        )

        signal = AttentionSignal(
            kind=AttentionSignalKind.USER_TURN,
            text=text,
            priority=priority,
            source="session_runtime",
            target_id=session.session_id,
            metadata={
                "session_id": session.session_id,
                "status": session.status.value,
                "follow_up": session.follow_up_expectation.value,
                "turn_count": session.turn_count,
            },
        )

        return self.submit_signal(signal)

    def focus_target(
        self,
        target: AttentionTarget,
    ) -> AttentionDecision:
        """
        Explicitly focus a target.
        """

        signal = AttentionSignal(
            kind=AttentionSignalKind.SYSTEM_NOTICE,
            text=target.label,
            priority=target.priority,
            source="attention_runtime",
            target_id=target.target_id,
            metadata={
                "explicit_focus": True,
            },
        )

        with self._lock:
            self._targets[target.target_id] = target

        return self.submit_signal(signal)

    def background_target(self, target_id: str) -> AttentionTarget | None:
        """
        Move a target to background.
        """

        with self._lock:
            target = self._targets.get(target_id)

            if target is None:
                return None

            updated = target.model_copy(
                update={
                    "status": AttentionTargetStatus.BACKGROUNDED,
                    "updated_at": utc_now(),
                }
            )
            self._targets[target_id] = updated

            if self._current_focus_id == target_id:
                self._current_focus_id = None

            self._background_count += 1

            return updated

    def complete_target(self, target_id: str) -> AttentionTarget | None:
        """
        Mark a target complete.
        """

        with self._lock:
            target = self._targets.get(target_id)

            if target is None:
                return None

            updated = target.model_copy(
                update={
                    "status": AttentionTargetStatus.COMPLETED,
                    "updated_at": utc_now(),
                }
            )
            self._targets[target_id] = updated

            if self._current_focus_id == target_id:
                self._current_focus_id = None

            return updated

    def cancel_target(self, target_id: str) -> AttentionTarget | None:
        """
        Mark a target cancelled.
        """

        with self._lock:
            target = self._targets.get(target_id)

            if target is None:
                return None

            updated = target.model_copy(
                update={
                    "status": AttentionTargetStatus.CANCELLED,
                    "updated_at": utc_now(),
                }
            )
            self._targets[target_id] = updated

            if self._current_focus_id == target_id:
                self._current_focus_id = None

            return updated

    def current_focus(self) -> AttentionTarget | None:
        """
        Return current focus target.
        """

        with self._lock:
            if self._current_focus_id is None:
                return None

            return self._targets.get(self._current_focus_id)

    def as_context_block(self) -> str:
        """
        Return a compact cognition-ready attention context block.
        """

        with self._lock:
            focus = (
                self._targets.get(self._current_focus_id)
                if self._current_focus_id is not None
                else None
            )
            active_targets = tuple(
                target
                for target in self._targets.values()
                if target.status == AttentionTargetStatus.ACTIVE
            )

        lines = [
            "Attention runtime:",
            f"- current_focus: {focus.label if focus else 'none'}",
            f"- active_targets: {len(active_targets)}",
        ]

        for target in active_targets[:5]:
            lines.append(
                "- target: "
                f"{target.label} "
                f"({target.kind.value}, {target.priority.value}, "
                f"weight={target.weight:.2f})"
            )

        return "\n".join(lines)

    def snapshot(self) -> AttentionRuntimeSnapshot:
        """
        Return observable diagnostics.
        """

        with self._lock:
            focus = (
                self._targets.get(self._current_focus_id)
                if self._current_focus_id is not None
                else None
            )

            return AttentionRuntimeSnapshot(
                name=self.name,
                target_count=len(self._targets),
                signal_count=self._signal_count,
                decision_count=self._decision_count,
                focus_change_count=self._focus_change_count,
                interrupt_count=self._interrupt_count,
                background_count=self._background_count,
                current_focus_id=self._current_focus_id,
                current_focus_label=focus.label if focus else None,
                last_disposition=self._last_disposition,
                last_signal_kind=self._last_signal_kind,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Reset attention runtime.
        """

        with self._lock:
            self._targets.clear()
            self._current_focus_id = None
            self._signal_count = 0
            self._decision_count = 0
            self._focus_change_count = 0
            self._interrupt_count = 0
            self._background_count = 0
            self._last_disposition = None
            self._last_signal_kind = None
            self._last_error = None

        self._logger.info("attention_runtime_reset", runtime=self.name)

    def _decide(self, signal: AttentionSignal) -> AttentionDecision:
        weight = self._weight_signal(signal)
        disposition = self._disposition(signal=signal, weight=weight)
        previous_focus_id = self._current_focus_id
        target = self._target_for_signal(signal=signal, weight=weight)

        should_interrupt = disposition == AttentionDisposition.INTERRUPT
        should_background_existing = (
            previous_focus_id is not None
            and previous_focus_id != target.target_id
            and disposition
            in {
                AttentionDisposition.FOCUS,
                AttentionDisposition.INTERRUPT,
            }
        )
        should_start_cognition = disposition in {
            AttentionDisposition.FOCUS,
            AttentionDisposition.INTERRUPT,
        }
        should_monitor = disposition == AttentionDisposition.MONITOR

        with self._lock:
            if disposition in {
                AttentionDisposition.FOCUS,
                AttentionDisposition.INTERRUPT,
                AttentionDisposition.MONITOR,
            }:
                self._targets[target.target_id] = target

            if disposition in {
                AttentionDisposition.FOCUS,
                AttentionDisposition.INTERRUPT,
            }:
                self._current_focus_id = target.target_id

        return AttentionDecision(
            signal=signal,
            disposition=disposition,
            selected_target=target,
            previous_focus_id=previous_focus_id,
            current_focus_id=(
                target.target_id
                if disposition
                in {
                    AttentionDisposition.FOCUS,
                    AttentionDisposition.INTERRUPT,
                }
                else previous_focus_id
            ),
            should_interrupt=should_interrupt,
            should_background_existing=should_background_existing,
            should_start_cognition=should_start_cognition,
            should_monitor=should_monitor,
            reason=self._reason(disposition),
            confidence=weight,
            metadata={
                "runtime": self.name,
                "signal_weight": weight,
                "signal_priority": signal.priority.value,
                "signal_urgency": signal.urgency.value,
            },
        )

    def _target_for_signal(
        self,
        *,
        signal: AttentionSignal,
        weight: float,
    ) -> AttentionTarget:
        target_id = signal.target_id or signal.signal_id
        existing = self._targets.get(target_id)

        if existing is not None:
            return existing.model_copy(
                update={
                    "priority": signal.priority,
                    "weight": max(existing.weight, weight),
                    "updated_at": utc_now(),
                }
            )

        return AttentionTarget(
            target_id=target_id,
            kind=self._target_kind(signal.kind),
            label=signal.text,
            priority=signal.priority,
            weight=weight,
            metadata={
                "source": signal.source,
                "signal_kind": signal.kind.value,
            },
        )

    def _weight_signal(self, signal: AttentionSignal) -> float:
        priority_weight = {
            AttentionPriority.LOW: 0.2,
            AttentionPriority.NORMAL: 0.45,
            AttentionPriority.HIGH: 0.68,
            AttentionPriority.URGENT: 0.82,
            AttentionPriority.CRITICAL: 0.96,
        }[signal.priority]

        urgency_bonus = {
            TurnUrgency.LOW: 0.0,
            TurnUrgency.NORMAL: 0.05,
            TurnUrgency.HIGH: 0.12,
            TurnUrgency.CRITICAL: 0.2,
        }[signal.urgency]

        kind_bonus = {
            AttentionSignalKind.USER_TURN: 0.12,
            AttentionSignalKind.FOLLOW_UP: 0.16,
            AttentionSignalKind.INTERRUPTION: 0.28,
            AttentionSignalKind.TOOL_RESULT: 0.1,
            AttentionSignalKind.ENVIRONMENT_CHANGE: 0.04,
            AttentionSignalKind.BACKGROUND_TASK: -0.12,
            AttentionSignalKind.ASSISTANT_RESPONSE: -0.05,
            AttentionSignalKind.SYSTEM_NOTICE: 0.0,
        }[signal.kind]

        state_bonus = 0.0

        if signal.state == ConversationState.SPEAKING:
            state_bonus += 0.06

        if signal.state == ConversationState.INTERRUPTED:
            state_bonus += 0.14

        weight = priority_weight + urgency_bonus + kind_bonus + state_bonus

        return max(0.0, min(1.0, weight))

    def _disposition(
        self,
        *,
        signal: AttentionSignal,
        weight: float,
    ) -> AttentionDisposition:
        if signal.kind == AttentionSignalKind.INTERRUPTION:
            return AttentionDisposition.INTERRUPT

        if signal.priority == AttentionPriority.CRITICAL:
            return AttentionDisposition.INTERRUPT

        if (
            weight >= self._config.interrupt_threshold
            and signal.kind
            in {
                AttentionSignalKind.SYSTEM_NOTICE,
                AttentionSignalKind.TOOL_RESULT,
                AttentionSignalKind.ENVIRONMENT_CHANGE,
            }
            and signal.urgency == TurnUrgency.CRITICAL
        ):
            return AttentionDisposition.INTERRUPT

        if weight >= self._config.focus_threshold:
            return AttentionDisposition.FOCUS

        if weight >= self._config.background_threshold:
            return AttentionDisposition.MONITOR

        if signal.kind == AttentionSignalKind.BACKGROUND_TASK:
            return AttentionDisposition.BACKGROUND

        return AttentionDisposition.IGNORE

    @staticmethod
    def _target_kind(signal_kind: AttentionSignalKind) -> AttentionTargetKind:
        if signal_kind == AttentionSignalKind.INTERRUPTION:
            return AttentionTargetKind.INTERRUPTION

        if signal_kind in {
            AttentionSignalKind.USER_TURN,
            AttentionSignalKind.FOLLOW_UP,
            AttentionSignalKind.ASSISTANT_RESPONSE,
        }:
            return AttentionTargetKind.CONVERSATION

        if signal_kind == AttentionSignalKind.TOOL_RESULT:
            return AttentionTargetKind.TOOL_TASK

        if signal_kind == AttentionSignalKind.BACKGROUND_TASK:
            return AttentionTargetKind.BACKGROUND_REASONING

        if signal_kind == AttentionSignalKind.ENVIRONMENT_CHANGE:
            return AttentionTargetKind.ENVIRONMENT

        return AttentionTargetKind.SYSTEM

    @staticmethod
    def _reason(disposition: AttentionDisposition) -> str:
        if disposition == AttentionDisposition.INTERRUPT:
            return "signal requires immediate attention and interruption"

        if disposition == AttentionDisposition.FOCUS:
            return "signal selected as active conversational focus"

        if disposition == AttentionDisposition.MONITOR:
            return "signal should be monitored without stealing focus"

        if disposition == AttentionDisposition.BACKGROUND:
            return "signal should stay in background"

        return "signal ignored because it is below attention threshold"

    def _record_decision(self, decision: AttentionDecision) -> None:
        with self._lock:
            self._decision_count += 1
            self._last_disposition = decision.disposition

            if decision.should_interrupt:
                self._interrupt_count += 1

            if decision.disposition == AttentionDisposition.BACKGROUND:
                self._background_count += 1

            if (
                decision.current_focus_id is not None
                and decision.current_focus_id != decision.previous_focus_id
            ):
                self._focus_change_count += 1

            if len(self._targets) > self._config.max_targets:
                self._trim_targets_locked()

    def _trim_targets_locked(self) -> None:
        active_focus = self._current_focus_id
        sorted_targets = sorted(
            self._targets.values(),
            key=lambda target: (
                target.status != AttentionTargetStatus.ACTIVE,
                -target.weight,
            ),
        )
        keep_ids = {
            target.target_id
            for target in sorted_targets[: self._config.max_targets]
        }

        if active_focus is not None:
            keep_ids.add(active_focus)

        self._targets = {
            target_id: target
            for target_id, target in self._targets.items()
            if target_id in keep_ids
        }