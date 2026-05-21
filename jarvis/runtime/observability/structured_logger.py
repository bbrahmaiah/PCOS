from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from jarvis.runtime.config import get_settings


class JsonFormatter(logging.Formatter):
    """
    JSON log formatter.

    Logs must be machine-readable because future JARVIS diagnostics,
    dashboards, and trace viewers will consume these records.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.threadName,
        }

        extra_fields = getattr(record, "extra_fields", None)
        if isinstance(extra_fields, dict):
            payload.update(extra_fields)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


class StructuredLogger:
    """
    Thin wrapper around Python logging.

    This gives the runtime consistent structured logs with optional extra fields.
    """

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    @property
    def raw(self) -> logging.Logger:
        return self._logger

    def debug(self, message: str, **fields: Any) -> None:
        self._logger.debug(message, extra={"extra_fields": fields})

    def info(self, message: str, **fields: Any) -> None:
        self._logger.info(message, extra={"extra_fields": fields})

    def warning(self, message: str, **fields: Any) -> None:
        self._logger.warning(message, extra={"extra_fields": fields})

    def error(self, message: str, **fields: Any) -> None:
        self._logger.error(message, extra={"extra_fields": fields})

    def exception(self, message: str, **fields: Any) -> None:
        self._logger.exception(message, extra={"extra_fields": fields})


def _level_from_name(level_name: str) -> int:
    normalized = level_name.strip().upper()
    return getattr(logging, normalized, logging.INFO)


def configure_logging() -> None:
    """
    Configure root JARVIS logging once.

    Safe to call multiple times.
    """

    settings = get_settings()
    settings.ensure_directories()

    logger = logging.getLogger("jarvis")
    logger.setLevel(_level_from_name(settings.logging.log_level))
    logger.propagate = False

    if logger.handlers:
        return

    formatter = JsonFormatter()

    if settings.logging.enable_console_logging:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if settings.logging.enable_file_logging:
        log_file: Path = settings.paths.logs_dir / "jarvis_runtime.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=5_000_000,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def get_logger(name: str) -> StructuredLogger:
    configure_logging()

    if name.startswith("jarvis"):
        logger_name = name
    else:
        logger_name = f"jarvis.{name}"

    return StructuredLogger(logger_name)