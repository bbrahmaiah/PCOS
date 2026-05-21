from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from threading import RLock
from typing import Any


@dataclass(frozen=True, slots=True)
class ContextSnapshot:
    """
    Immutable snapshot of global runtime context.
    """

    values: dict[str, Any]
    size: int


class GlobalContext:
    """
    Thread-safe context store for live runtime facts.

    This is working context, not long-term memory.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._values: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        clean_key = self._validate_key(key)

        with self._lock:
            self._values[clean_key] = deepcopy(value)

    def get(self, key: str, default: Any = None) -> Any:
        clean_key = self._validate_key(key)

        with self._lock:
            return deepcopy(self._values.get(clean_key, default))

    def delete(self, key: str) -> bool:
        clean_key = self._validate_key(key)

        with self._lock:
            existed = clean_key in self._values
            self._values.pop(clean_key, None)
            return existed

    def update_many(self, values: dict[str, Any]) -> None:
        if not isinstance(values, dict):
            raise TypeError("values must be a dictionary.")

        cleaned_values: dict[str, Any] = {}

        for key, value in values.items():
            clean_key = self._validate_key(key)
            cleaned_values[clean_key] = deepcopy(value)

        with self._lock:
            self._values.update(cleaned_values)

    def snapshot(self) -> ContextSnapshot:
        with self._lock:
            values = deepcopy(self._values)

        return ContextSnapshot(
            values=values,
            size=len(values),
        )

    def clear(self) -> None:
        with self._lock:
            self._values.clear()

    @staticmethod
    def _validate_key(key: str) -> str:
        if not isinstance(key, str):
            raise TypeError("context key must be a string.")

        clean_key = key.strip()

        if not clean_key:
            raise ValueError("context key cannot be empty.")

        return clean_key