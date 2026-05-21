from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

from jarvis.runtime.shared.enums import RuntimeStatus, SystemMode


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class RuntimeState:
    """
    Immutable runtime state snapshot.
    """

    status: RuntimeStatus = RuntimeStatus.CREATED
    mode: SystemMode = SystemMode.PASSIVE
    started_at: datetime | None = None
    stopped_at: datetime | None = None
    updated_at: datetime = field(default_factory=utc_now)
    last_error: str | None = None

    def with_status(
        self,
        status: RuntimeStatus,
        *,
        last_error: str | None = None,
    ) -> RuntimeState:
        now = utc_now()

        started_at = self.started_at
        stopped_at = self.stopped_at

        if status == RuntimeStatus.RUNNING and started_at is None:
            started_at = now

        if status in {RuntimeStatus.STOPPED, RuntimeStatus.FAILED}:
            stopped_at = now

        return replace(
            self,
            status=status,
            started_at=started_at,
            stopped_at=stopped_at,
            updated_at=now,
            last_error=last_error,
        )

    def with_mode(self, mode: SystemMode) -> RuntimeState:
        return replace(
            self,
            mode=mode,
            updated_at=utc_now(),
        )