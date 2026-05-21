from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from threading import RLock
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_identity_id() -> str:
    return uuid4().hex


@dataclass(frozen=True, slots=True)
class SecurityIdentity:
    """
    Authenticated runtime identity.

    This is not face/voice authentication yet.
    It is the runtime identity contract that future biometric auth will feed.
    """

    identity_id: str = field(default_factory=new_identity_id)
    user_id: str = "local_user"
    display_name: str = "Local User"
    authenticated: bool = True
    authenticated_at: datetime = field(default_factory=utc_now)

    def with_display_name(self, display_name: str) -> SecurityIdentity:
        cleaned = display_name.strip()

        if not cleaned:
            raise ValueError("display_name cannot be empty.")

        return replace(self, display_name=cleaned)


class IdentityManager:
    """
    Thread-safe identity manager.

    For now it supports local authenticated identity.
    Later it can be backed by voiceprint, face recognition, PIN, Windows Hello,
    or signed local tokens.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._identity: SecurityIdentity | None = None

    def authenticate_local_user(
        self,
        *,
        user_id: str = "bala",
        display_name: str = "Bala",
    ) -> SecurityIdentity:
        clean_user_id = user_id.strip()
        clean_display_name = display_name.strip()

        if not clean_user_id:
            raise ValueError("user_id cannot be empty.")

        if not clean_display_name:
            raise ValueError("display_name cannot be empty.")

        identity = SecurityIdentity(
            user_id=clean_user_id,
            display_name=clean_display_name,
        )

        with self._lock:
            self._identity = identity

        return identity

    def current_identity(self) -> SecurityIdentity | None:
        with self._lock:
            return self._identity

    def require_identity(self) -> SecurityIdentity:
        identity = self.current_identity()

        if identity is None or not identity.authenticated:
            raise PermissionError("No authenticated identity is active.")

        return identity

    def logout(self) -> None:
        with self._lock:
            self._identity = None