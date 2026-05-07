from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "event_type"):
            payload["event_type"] = record.event_type
        if hasattr(record, "username"):
            payload["username"] = record.username
        if hasattr(record, "source_type"):
            payload["source_type"] = record.source_type
        if hasattr(record, "details"):
            payload["details"] = record.details
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logger(name: str, log_path: str) -> logging.Logger:
    """Create or reuse a rotating JSON logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    path = Path(log_path)
    if path.parent.as_posix() not in ("", "."):
        path.parent.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        filename=path,
        maxBytes=2_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(JsonFormatter())
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
