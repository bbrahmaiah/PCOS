from __future__ import annotations

from typing import Any
from uuid import UUID

type JSONDict = dict[str, Any]
type Metadata = dict[str, Any]

type WorkerId = str
type EventId = str
type SessionId = str
type CorrelationId = str

type Timestamp = float
type UUIDType = UUID