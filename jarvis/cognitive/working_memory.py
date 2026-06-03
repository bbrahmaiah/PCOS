from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from uuid import uuid4

from jarvis.cognitive.contracts import (
    AttentionPriority,
    WorkingMemoryItem,
    WorkingMemoryKind,
    WorkingMemoryState,
    utc_now,
)


class WorkingMemoryRuntimeStatus(StrEnum):
    READY = "ready"
    DEGRADED = "degraded"
    BLOCKED = "blocked"


class WorkingMemoryOperation(StrEnum):
    UPSERT = "upsert"
    REMOVE = "remove"
    CLEAR = "clear"
    RECALL = "recall"
    COMPACT = "compact"


class WorkingMemoryRetention(StrEnum):
    SHORT = "short"
    SESSION = "session"
    EXTENDED = "extended"


@dataclass(frozen=True, slots=True)
class WorkingMemoryEntry:
    kind: WorkingMemoryKind
    key: str
    value: str
    importance: AttentionPriority = AttentionPriority.NORMAL
    source: str = "working_memory_runtime"
    retention: WorkingMemoryRetention = WorkingMemoryRetention.SESSION
    ttl_seconds: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("working memory entry key cannot be empty.")
        if not self.value.strip():
            raise ValueError("working memory entry value cannot be empty.")
        if not self.source.strip():
            raise ValueError("working memory entry source cannot be empty.")
        if self.ttl_seconds is not None and self.ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive when provided.")


@dataclass(frozen=True, slots=True)
class WorkingMemoryUpdateRequest:
    entries: tuple[WorkingMemoryEntry, ...] = ()
    remove_keys: tuple[str, ...] = ()
    clear: bool = False
    max_items: int = 50
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_items < 1:
            raise ValueError("max_items must be at least 1.")


@dataclass(frozen=True, slots=True)
class WorkingMemoryRecallRequest:
    query: str = ""
    kinds: tuple[WorkingMemoryKind, ...] = ()
    minimum_importance: AttentionPriority = AttentionPriority.BACKGROUND
    limit: int = 10
    include_expired: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError("working memory recall limit must be at least 1.")


@dataclass(frozen=True, slots=True)
class WorkingMemoryRuntimeResult:
    status: WorkingMemoryRuntimeStatus
    operation: WorkingMemoryOperation
    state: WorkingMemoryState
    items: tuple[WorkingMemoryItem, ...]
    reason: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == WorkingMemoryRuntimeStatus.READY


@dataclass(frozen=True, slots=True)
class WorkingMemoryRuntimeSnapshot:
    status: WorkingMemoryRuntimeStatus
    state: WorkingMemoryState
    item_count: int
    update_count: int
    recall_count: int
    compact_count: int
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


class WorkingMemoryRuntime:
    """
    Phase 9 / Step 49B Working Memory Runtime.

    Working memory is the active cognitive context of JARVIS:
    - current conversation
    - current project
    - current objective
    - current screen context
    - recent actions
    - active tasks
    - current risks
    - assumptions
    - temporary preferences

    It is intentionally non-executing:
    - no tool calls
    - no long-term memory writes
    - no laptop control
    - no autonomous actions

    It only maintains short-lived cognitive state.
    """

    def __init__(self, *, max_items: int = 50) -> None:
        if max_items < 1:
            raise ValueError("working memory max_items must be at least 1.")

        self._max_items = max_items
        self._state = WorkingMemoryState()
        self._update_count = 0
        self._recall_count = 0
        self._compact_count = 0

    @property
    def state(self) -> WorkingMemoryState:
        return self._state

    def update(
        self,
        request: WorkingMemoryUpdateRequest,
    ) -> WorkingMemoryRuntimeResult:
        self._update_count += 1

        if request.clear:
            self._state = WorkingMemoryState()
            return WorkingMemoryRuntimeResult(
                status=WorkingMemoryRuntimeStatus.READY,
                operation=WorkingMemoryOperation.CLEAR,
                state=self._state,
                items=(),
                reason="working memory cleared",
                created_at=utc_now(),
                metadata=request.metadata,
            )

        items = tuple(
            item
            for item in self._state.items
            if item.key.strip().lower()
            not in {key.strip().lower() for key in request.remove_keys}
        )
        new_items = tuple(
            _item_from_entry(entry=entry)
            for entry in request.entries
        )
        merged = _merge_items(existing_items=items, new_items=new_items)
        compacted = _compact_items(
            items=merged,
            max_items=min(request.max_items, self._max_items),
            now=utc_now(),
        )
        self._state = WorkingMemoryState(
            items=compacted,
            created_at=utc_now(),
        )

        return WorkingMemoryRuntimeResult(
            status=WorkingMemoryRuntimeStatus.READY,
            operation=WorkingMemoryOperation.UPSERT,
            state=self._state,
            items=new_items,
            reason="working memory updated",
            created_at=utc_now(),
            metadata={
                **request.metadata,
                "entry_count": len(request.entries),
                "removed_count": len(request.remove_keys),
                "state_item_count": len(self._state.items),
            },
        )

    def recall(
        self,
        request: WorkingMemoryRecallRequest,
    ) -> WorkingMemoryRuntimeResult:
        self._recall_count += 1
        now = utc_now()

        items = tuple(
            item
            for item in self._state.items
            if request.include_expired or not _is_expired(item=item, now=now)
        )
        matches = _filter_items(
            items=items,
            query=request.query,
            kinds=request.kinds,
            minimum_importance=request.minimum_importance,
            limit=request.limit,
        )

        return WorkingMemoryRuntimeResult(
            status=WorkingMemoryRuntimeStatus.READY,
            operation=WorkingMemoryOperation.RECALL,
            state=self._state,
            items=matches,
            reason="working memory recall completed",
            created_at=utc_now(),
            metadata={
                **request.metadata,
                "query": request.query,
                "match_count": len(matches),
            },
        )

    def compact(self) -> WorkingMemoryRuntimeResult:
        self._compact_count += 1
        compacted = _compact_items(
            items=self._state.items,
            max_items=self._max_items,
            now=utc_now(),
        )
        self._state = WorkingMemoryState(
            items=compacted,
            created_at=utc_now(),
        )

        return WorkingMemoryRuntimeResult(
            status=WorkingMemoryRuntimeStatus.READY,
            operation=WorkingMemoryOperation.COMPACT,
            state=self._state,
            items=compacted,
            reason="working memory compacted",
            created_at=utc_now(),
            metadata={"item_count": len(compacted)},
        )

    def snapshot(self) -> WorkingMemoryRuntimeSnapshot:
        return WorkingMemoryRuntimeSnapshot(
            status=WorkingMemoryRuntimeStatus.READY,
            state=self._state,
            item_count=len(self._state.items),
            update_count=self._update_count,
            recall_count=self._recall_count,
            compact_count=self._compact_count,
            created_at=utc_now(),
            metadata={"max_items": self._max_items},
        )


def make_working_memory_entry(
    *,
    kind: WorkingMemoryKind,
    key: str,
    value: str,
    importance: AttentionPriority = AttentionPriority.NORMAL,
    source: str = "working_memory_runtime",
    retention: WorkingMemoryRetention = WorkingMemoryRetention.SESSION,
    ttl_seconds: int | None = None,
    metadata: dict[str, object] | None = None,
) -> WorkingMemoryEntry:
    return WorkingMemoryEntry(
        kind=kind,
        key=key,
        value=value,
        importance=importance,
        source=source,
        retention=retention,
        ttl_seconds=ttl_seconds,
        metadata=metadata or {},
    )


def _item_from_entry(entry: WorkingMemoryEntry) -> WorkingMemoryItem:
    now = utc_now()
    expires_at = (
        now + timedelta(seconds=entry.ttl_seconds)
        if entry.ttl_seconds is not None
        else None
    )

    return WorkingMemoryItem(
        item_id=f"wm_{uuid4().hex}",
        kind=entry.kind,
        key=entry.key.strip(),
        value=entry.value.strip(),
        importance=entry.importance,
        source=entry.source,
        created_at=now,
        metadata={
            **entry.metadata,
            "retention": entry.retention.value,
            "ttl_seconds": entry.ttl_seconds,
            "expires_at": expires_at.isoformat() if expires_at else "",
        },
    )


def _merge_items(
    *,
    existing_items: tuple[WorkingMemoryItem, ...],
    new_items: tuple[WorkingMemoryItem, ...],
) -> tuple[WorkingMemoryItem, ...]:
    by_key: dict[str, WorkingMemoryItem] = {}

    for item in (*existing_items, *new_items):
        key = item.key.strip().lower()
        current = by_key.get(key)
        if current is None:
            by_key[key] = item
            continue

        if _importance_rank(item.importance) >= _importance_rank(
            current.importance
        ):
            by_key[key] = item

    return tuple(by_key.values())


def _compact_items(
    *,
    items: tuple[WorkingMemoryItem, ...],
    max_items: int,
    now: datetime,
) -> tuple[WorkingMemoryItem, ...]:
    live_items = tuple(item for item in items if not _is_expired(item=item, now=now))
    sorted_items = sorted(
        live_items,
        key=lambda item: (
            -_importance_rank(item.importance),
            item.created_at,
            item.key,
        ),
    )
    return tuple(sorted_items[:max_items])


def _filter_items(
    *,
    items: tuple[WorkingMemoryItem, ...],
    query: str,
    kinds: tuple[WorkingMemoryKind, ...],
    minimum_importance: AttentionPriority,
    limit: int,
) -> tuple[WorkingMemoryItem, ...]:
    query_text = query.strip().lower()
    filtered: list[WorkingMemoryItem] = []

    for item in items:
        if kinds and item.kind not in kinds:
            continue

        if _importance_rank(item.importance) < _importance_rank(
            minimum_importance
        ):
            continue

        if query_text and query_text not in _search_text(item):
            continue

        filtered.append(item)

    filtered.sort(
        key=lambda item: (
            -_importance_rank(item.importance),
            item.created_at,
            item.key,
        )
    )
    return tuple(filtered[:limit])


def _search_text(item: WorkingMemoryItem) -> str:
    return " ".join(
        (
            item.kind.value,
            item.key,
            item.value,
            item.source,
        )
    ).lower()


def _is_expired(
    *,
    item: WorkingMemoryItem,
    now: datetime,
) -> bool:
    expires_at = item.metadata.get("expires_at")
    if not isinstance(expires_at, str) or not expires_at:
        return False

    try:
        parsed = datetime.fromisoformat(expires_at)
    except ValueError:
        return False

    return parsed <= now


def _importance_rank(priority: AttentionPriority) -> int:
    ranks = {
        AttentionPriority.BACKGROUND: 0,
        AttentionPriority.LOW: 1,
        AttentionPriority.NORMAL: 2,
        AttentionPriority.HIGH: 3,
        AttentionPriority.CRITICAL: 4,
    }
    return ranks[priority]