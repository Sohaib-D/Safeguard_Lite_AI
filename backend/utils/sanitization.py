from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from backend.core.config import settings
from backend.core.logging_config import configure_logger

CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
DANGEROUS_PREFIXES = ("=", "+", "-", "@")
SUSPICIOUS_TEXT_RE = re.compile(
    r"[<>\"'`;]|--|\b(select|union|drop|insert|delete|update|script)\b", re.IGNORECASE
)
logger = configure_logger("safeguard.backend.sanitization", settings.log_file_path)


def sanitize_filename(filename: str) -> str:
    """Normalize uploaded filenames and remove unsafe characters."""
    base = Path(filename).name
    cleaned = SAFE_NAME_RE.sub("_", base).strip("._")
    return cleaned or "upload.csv"


def sanitize_text(value: Any, max_length: int = 500) -> Any:
    """Strip control characters and neutralize formula-style injection prefixes."""
    if value is None or pd.isna(value):
        return value

    text = str(value).strip()
    text = CONTROL_CHARS_RE.sub("", text)
    if text.startswith(DANGEROUS_PREFIXES):
        text = f"'{text}"
    if len(text) > max_length:
        text = text[:max_length]
    return text


def strip_suspicious_text(value: Any, max_length: int = 500) -> Any:
    """
    Remove characters and keywords commonly used in injection payloads.

    This is a conservative sanitizer for user-controlled identifiers and text fields.
    """
    if value is None or pd.isna(value):
        return value
    text = sanitize_text(value, max_length=max_length)
    if not isinstance(text, str):
        text = str(text)
    sanitized = SUSPICIOUS_TEXT_RE.sub("", text)
    if sanitized != text:
        logger.warning(
            "Suspicious input stripped during sanitization.",
            extra={
                "event_type": "suspicious_input",
                "details": {
                    "original_preview": text[:120],
                    "sanitized_preview": sanitized[:120],
                },
            },
        )
    text = sanitized
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize column names and object cells for safer downstream handling."""
    sanitized = df.copy()
    sanitized.columns = [
        sanitize_text(col, max_length=100) for col in sanitized.columns
    ]

    for col in sanitized.columns:
        if sanitized[col].dtype == object:
            sanitized[col] = sanitized[col].map(
                lambda value: strip_suspicious_text(value)
            )

    return sanitized
