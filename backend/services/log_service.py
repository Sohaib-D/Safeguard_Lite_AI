from __future__ import annotations

from typing import Any

from backend.db.sqlite_store import SQLiteStore


class LogService:
    """Service wrapper around the shared SQLite store."""

    def __init__(self, database_path: str):
        self.store = SQLiteStore(database_path)

    def log_predictions(
        self,
        predictions: list[dict[str, Any]],
        source_type: str,
        username: str | None = None,
    ) -> list[int]:
        inserted_ids = self.store.store_predictions(
            predictions, source_type=source_type, username=username
        )
        self.store.log_activity(
            action="predict",
            username=username,
            source_type=source_type,
            row_count=len(predictions),
            details={
                "prediction_labels": [item["predicted_label"] for item in predictions]
            },
        )
        return inserted_ids

    def log_upload(
        self,
        source_name: str,
        source_type: str,
        records: list[dict[str, Any]],
        username: str | None = None,
    ) -> int:
        return self.store.log_activity(
            action="upload",
            username=username,
            source_type=source_type,
            row_count=len(records),
            details={
                "source_name": source_name,
                "payload_preview": records[:50],
            },
        )

    def log_user_activity(
        self,
        action: str,
        username: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> int:
        return self.store.log_activity(
            action=action, username=username, details=details or {}
        )

    def get_stats(self) -> dict[str, Any]:
        return self.store.get_stats()

    def ping(self) -> bool:
        return self.store.ping()
