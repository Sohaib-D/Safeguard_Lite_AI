from __future__ import annotations
from typing import Any
from sqlalchemy.orm import Session
from backend.db.postgres_store import PostgresStore
from backend.db.session import session_scope

class LogService:
    def __init__(self, db: Session | None = None):
        self._db = db

    def log_activity(
        self, event_type: str, username: str | None, details: dict[str, Any], severity: str = "INFO"
    ) -> None:
        if self._db:
            PostgresStore(self._db).log_activity(event_type, username, details, severity)
        else:
            with session_scope() as db:
                PostgresStore(db).log_activity(event_type, username, details, severity)

    def log_user_activity(self, action: str, username: str, details: dict[str, Any]) -> None:
        """Alias for log_activity to maintain backward compatibility."""
        self.log_activity(action, username, details)

    def get_stats(self) -> dict[str, Any]:
        if self._db:
            return PostgresStore(self._db).get_stats()
        else:
            with session_scope() as db:
                return PostgresStore(db).get_stats()
