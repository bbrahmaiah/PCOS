from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Event, RLock
from typing import Any
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_cancellation_id() -> str:
    return uuid4().hex


@dataclass(frozen=True, slots=True)
class CancellationSnapshot:
    token_id: str
    reason: str | None
    cancelled: bool
    created_at: datetime
    cancelled_at: datetime | None
    metadata: dict[str, Any]


class CancellationToken:
    """
    Thread-safe cancellation token.

    Future long-running tasks, speech, model calls, browser tasks,
    and action chains should receive one of these.
    """

    def __init__(
        self,
        *,
        token_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        clean_id = (token_id or new_cancellation_id()).strip()

        if not clean_id:
            raise ValueError("cancellation token_id cannot be empty.")

        self.token_id = clean_id
        self.created_at = utc_now()
        self.metadata = dict(metadata or {})

        self._event = Event()
        self._lock = RLock()
        self._reason: str | None = None
        self._cancelled_at: datetime | None = None

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason

    @property
    def cancelled_at(self) -> datetime | None:
        with self._lock:
            return self._cancelled_at

    def cancel(self, reason: str | None = None) -> None:
        clean_reason = reason.strip() if isinstance(reason, str) else reason

        with self._lock:
            if self._event.is_set():
                return

            self._reason = clean_reason or "Cancellation requested."
            self._cancelled_at = utc_now()
            self._event.set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise RuntimeError(self.reason or "Cancellation requested.")

    def snapshot(self) -> CancellationSnapshot:
        with self._lock:
            return CancellationSnapshot(
                token_id=self.token_id,
                reason=self._reason,
                cancelled=self.cancelled,
                created_at=self.created_at,
                cancelled_at=self._cancelled_at,
                metadata=dict(self.metadata),
            )


@dataclass(frozen=True, slots=True)
class CancellationManagerSnapshot:
    token_count: int
    cancelled_count: int
    tokens: tuple[CancellationSnapshot, ...]


class CancellationManager:
    """
    Central cancellation registry.

    This gives JARVIS a clean way to interrupt speech, actions,
    tool execution, model reasoning, and future workflows.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._tokens: dict[str, CancellationToken] = {}

    def create_token(
        self,
        *,
        token_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CancellationToken:
        token = CancellationToken(
            token_id=token_id,
            metadata=metadata,
        )

        with self._lock:
            if token.token_id in self._tokens:
                raise ValueError(f"Cancellation token already exists: {token.token_id}")

            self._tokens[token.token_id] = token

        return token

    def get(self, token_id: str) -> CancellationToken | None:
        clean_id = self._validate_token_id(token_id)

        with self._lock:
            return self._tokens.get(clean_id)

    def require(self, token_id: str) -> CancellationToken:
        token = self.get(token_id)

        if token is None:
            raise KeyError(f"Cancellation token not found: {token_id}")

        return token

    def cancel(self, token_id: str, *, reason: str | None = None) -> bool:
        token = self.get(token_id)

        if token is None:
            return False

        token.cancel(reason)
        return True

    def cancel_all(self, *, reason: str | None = None) -> None:
        with self._lock:
            tokens = tuple(self._tokens.values())

        for token in tokens:
            token.cancel(reason)

    def remove(self, token_id: str) -> bool:
        clean_id = self._validate_token_id(token_id)

        with self._lock:
            existed = clean_id in self._tokens
            self._tokens.pop(clean_id, None)
            return existed

    def clear(self) -> None:
        with self._lock:
            self._tokens.clear()

    def snapshot(self) -> CancellationManagerSnapshot:
        with self._lock:
            snapshots = tuple(token.snapshot() for token in self._tokens.values())

        return CancellationManagerSnapshot(
            token_count=len(snapshots),
            cancelled_count=sum(1 for snapshot in snapshots if snapshot.cancelled),
            tokens=snapshots,
        )

    @staticmethod
    def _validate_token_id(token_id: str) -> str:
        if not isinstance(token_id, str):
            raise TypeError("token_id must be a string.")

        clean_id = token_id.strip()

        if not clean_id:
            raise ValueError("token_id cannot be empty.")

        return clean_id