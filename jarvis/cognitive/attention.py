from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from uuid import uuid4

from jarvis.cognitive.contracts import (
    AttentionDecision,
    AttentionItem,
    AttentionItemKind,
    AttentionPriority,
    AttentionState,
    utc_now,
)


class AttentionSignalUrgency(StrEnum):
    BACKGROUND = "background"
    NORMAL = "normal"
    IMPORTANT = "important"
    URGENT = "urgent"
    EMERGENCY = "emergency"


class AttentionSignalSource(StrEnum):
    VOICE = "voice"
    SCREEN = "screen"
    NOTIFICATION = "notification"
    SYSTEM = "system"
    PROJECT = "project"
    MEMORY = "memory"
    TASK = "task"
    SAFETY = "safety"
    RESEARCH = "research"
    USER = "user"


class AttentionRuntimeStatus(StrEnum):
    READY = "ready"
    DEGRADED = "degraded"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class AttentionSignal:
    signal_id: str
    source: AttentionSignalSource
    kind: AttentionItemKind
    title: str
    summary: str
    urgency: AttentionSignalUrgency
    confidence: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.signal_id.strip():
            raise ValueError("attention signal_id cannot be empty.")
        if not self.title.strip():
            raise ValueError("attention signal title cannot be empty.")
        if not self.summary.strip():
            raise ValueError("attention signal summary cannot be empty.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("attention signal confidence must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class AttentionEvaluationRequest:
    signals: tuple[AttentionSignal, ...]
    current_state: AttentionState | None = None
    user_is_speaking: bool = False
    assistant_is_speaking: bool = False
    allow_interruptions: bool = True
    debug: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AttentionEvaluationResult:
    status: AttentionRuntimeStatus
    state: AttentionState
    selected_item: AttentionItem | None
    decision: AttentionDecision
    reason: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def should_interrupt(self) -> bool:
        return self.decision == AttentionDecision.INTERRUPT_NOW

    @property
    def has_focus(self) -> bool:
        return self.selected_item is not None


@dataclass(frozen=True, slots=True)
class AttentionRuntimeSnapshot:
    status: AttentionRuntimeStatus
    state: AttentionState
    last_decision: AttentionDecision | None
    last_reason: str
    evaluated_count: int
    interruption_count: int
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


class AttentionRuntime:
    """
    Phase 9 / Step 49A Attention Runtime.

    The attention runtime ranks incoming signals and decides whether JARVIS
    should ignore, track, focus, or interrupt.

    It is intentionally non-executing:
    - no tool calls
    - no memory writes
    - no speech output
    - no laptop control

    It only produces attention state and decision objects for Conversation,
    Orchestration, Presence, and later Personality/Behavior layers.
    """

    def __init__(self) -> None:
        self._state = AttentionState()
        self._last_decision: AttentionDecision | None = None
        self._last_reason = "attention runtime initialized"
        self._evaluated_count = 0
        self._interruption_count = 0

    @property
    def state(self) -> AttentionState:
        return self._state

    def evaluate(
        self,
        request: AttentionEvaluationRequest,
    ) -> AttentionEvaluationResult:
        self._evaluated_count += 1

        if not request.signals and request.current_state is None:
            self._state = AttentionState()
            self._last_decision = AttentionDecision.IGNORE
            self._last_reason = "no attention signals available"
            return AttentionEvaluationResult(
                status=AttentionRuntimeStatus.DEGRADED,
                state=self._state,
                selected_item=None,
                decision=AttentionDecision.IGNORE,
                reason=self._last_reason,
                created_at=utc_now(),
                metadata=request.metadata,
            )

        existing_items = (
            request.current_state.items
            if request.current_state is not None
            else self._state.items
        )
        new_items = tuple(
            _attention_item_from_signal(signal=signal)
            for signal in request.signals
        )
        merged_items = _merge_attention_items(existing_items, new_items)

        selected = _select_focus_item(merged_items)
        decision = _decision_for_selected_item(
            selected=selected,
            user_is_speaking=request.user_is_speaking,
            assistant_is_speaking=request.assistant_is_speaking,
            allow_interruptions=request.allow_interruptions,
        )
        focused_item_id = (
            selected.item_id
            if selected is not None
            and decision in {
                AttentionDecision.FOCUS,
                AttentionDecision.INTERRUPT_NOW,
            }
            else None
        )

        self._state = AttentionState(
            items=merged_items,
            focused_item_id=focused_item_id,
            created_at=utc_now(),
        )
        self._last_decision = decision
        self._last_reason = _reason_for_decision(
            selected=selected,
            decision=decision,
            user_is_speaking=request.user_is_speaking,
            assistant_is_speaking=request.assistant_is_speaking,
            allow_interruptions=request.allow_interruptions,
        )

        if decision == AttentionDecision.INTERRUPT_NOW:
            self._interruption_count += 1

        return AttentionEvaluationResult(
            status=AttentionRuntimeStatus.READY,
            state=self._state,
            selected_item=selected,
            decision=decision,
            reason=self._last_reason,
            created_at=utc_now(),
            metadata={
                **request.metadata,
                "signal_count": len(request.signals),
                "item_count": len(self._state.items),
                "debug": request.debug,
            },
        )

    def clear(self) -> AttentionEvaluationResult:
        self._state = AttentionState()
        self._last_decision = AttentionDecision.IGNORE
        self._last_reason = "attention state cleared"

        return AttentionEvaluationResult(
            status=AttentionRuntimeStatus.READY,
            state=self._state,
            selected_item=None,
            decision=AttentionDecision.IGNORE,
            reason=self._last_reason,
            created_at=utc_now(),
        )

    def snapshot(self) -> AttentionRuntimeSnapshot:
        return AttentionRuntimeSnapshot(
            status=AttentionRuntimeStatus.READY,
            state=self._state,
            last_decision=self._last_decision,
            last_reason=self._last_reason,
            evaluated_count=self._evaluated_count,
            interruption_count=self._interruption_count,
            created_at=utc_now(),
        )


def make_attention_signal(
    *,
    source: AttentionSignalSource,
    kind: AttentionItemKind,
    title: str,
    summary: str,
    urgency: AttentionSignalUrgency,
    confidence: float = 1.0,
    metadata: dict[str, object] | None = None,
) -> AttentionSignal:
    return AttentionSignal(
        signal_id=f"sig_{uuid4().hex}",
        source=source,
        kind=kind,
        title=title,
        summary=summary,
        urgency=urgency,
        confidence=confidence,
        created_at=utc_now(),
        metadata=metadata or {},
    )


def _attention_item_from_signal(signal: AttentionSignal) -> AttentionItem:
    priority = _priority_for_signal(signal)
    decision = _base_decision_for_priority(priority)

    return AttentionItem(
        item_id=f"att_{signal.signal_id}",
        kind=signal.kind,
        title=signal.title,
        summary=signal.summary,
        priority=priority,
        source=signal.source.value,
        decision=decision,
        created_at=signal.created_at,
        metadata={
            **signal.metadata,
            "signal_id": signal.signal_id,
            "urgency": signal.urgency.value,
            "confidence": signal.confidence,
        },
    )


def _priority_for_signal(signal: AttentionSignal) -> AttentionPriority:
    if signal.confidence < 0.25:
        return AttentionPriority.BACKGROUND

    if signal.urgency == AttentionSignalUrgency.EMERGENCY:
        return AttentionPriority.CRITICAL

    if signal.urgency == AttentionSignalUrgency.URGENT:
        return AttentionPriority.HIGH

    if signal.urgency == AttentionSignalUrgency.IMPORTANT:
        return (
            AttentionPriority.HIGH
            if signal.confidence >= 0.7
            else AttentionPriority.NORMAL
        )

    if signal.urgency == AttentionSignalUrgency.NORMAL:
        return AttentionPriority.NORMAL

    return AttentionPriority.BACKGROUND


def _base_decision_for_priority(priority: AttentionPriority) -> AttentionDecision:
    if priority == AttentionPriority.CRITICAL:
        return AttentionDecision.INTERRUPT_NOW

    if priority == AttentionPriority.HIGH:
        return AttentionDecision.FOCUS

    if priority in {AttentionPriority.NORMAL, AttentionPriority.LOW}:
        return AttentionDecision.TRACK

    return AttentionDecision.IGNORE


def _merge_attention_items(
    existing_items: tuple[AttentionItem, ...],
    new_items: tuple[AttentionItem, ...],
) -> tuple[AttentionItem, ...]:
    by_key: dict[tuple[AttentionItemKind, str, str], AttentionItem] = {}

    for item in (*existing_items, *new_items):
        key = (
            item.kind,
            item.source,
            item.title.strip().lower(),
        )
        current = by_key.get(key)
        if current is None or _rank_priority(item.priority) >= _rank_priority(
            current.priority
        ):
            by_key[key] = item

    return tuple(
        sorted(
            by_key.values(),
            key=lambda item: (
                -_rank_priority(item.priority),
                item.created_at,
                item.title,
            ),
        )
    )


def _select_focus_item(items: tuple[AttentionItem, ...]) -> AttentionItem | None:
    if not items:
        return None

    for item in items:
        if item.priority == AttentionPriority.CRITICAL:
            return item

    for item in items:
        if item.priority == AttentionPriority.HIGH:
            return item

    for item in items:
        if item.priority == AttentionPriority.NORMAL:
            return item

    return None


def _decision_for_selected_item(
    *,
    selected: AttentionItem | None,
    user_is_speaking: bool,
    assistant_is_speaking: bool,
    allow_interruptions: bool,
) -> AttentionDecision:
    if selected is None:
        return AttentionDecision.IGNORE

    if selected.priority == AttentionPriority.CRITICAL:
        return (
            AttentionDecision.INTERRUPT_NOW
            if allow_interruptions
            else AttentionDecision.FOCUS
        )

    if selected.priority == AttentionPriority.HIGH:
        if user_is_speaking:
            return AttentionDecision.TRACK
        if assistant_is_speaking and allow_interruptions:
            return AttentionDecision.FOCUS
        return AttentionDecision.FOCUS

    if selected.priority == AttentionPriority.NORMAL:
        return AttentionDecision.TRACK

    return AttentionDecision.IGNORE


def _reason_for_decision(
    *,
    selected: AttentionItem | None,
    decision: AttentionDecision,
    user_is_speaking: bool,
    assistant_is_speaking: bool,
    allow_interruptions: bool,
) -> str:
    if selected is None:
        return "no focus-worthy attention item found"

    if decision == AttentionDecision.INTERRUPT_NOW:
        return (
            f"critical attention item requires immediate interruption: "
            f"{selected.title}"
        )

    if selected.priority == AttentionPriority.HIGH and user_is_speaking:
        return (
            "high-priority item tracked without interrupting because user "
            "is currently speaking"
        )

    if selected.priority == AttentionPriority.CRITICAL and not allow_interruptions:
        return (
            "critical item focused but interruption was disabled by runtime "
            "policy"
        )

    if assistant_is_speaking and decision == AttentionDecision.FOCUS:
        return (
            "attention item focused while assistant is speaking; no hard "
            "interruption required"
        )

    if decision == AttentionDecision.FOCUS:
        return f"attention focused on: {selected.title}"

    if decision == AttentionDecision.TRACK:
        return f"attention tracking: {selected.title}"

    return "attention item ignored"


def _rank_priority(priority: AttentionPriority) -> int:
    ranks = {
        AttentionPriority.BACKGROUND: 0,
        AttentionPriority.LOW: 1,
        AttentionPriority.NORMAL: 2,
        AttentionPriority.HIGH: 3,
        AttentionPriority.CRITICAL: 4,
    }
    return ranks[priority]