from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


class SQLiteStore:
    """Shared SQLite store with transactional CRUD helpers."""

    def __init__(self, database_path: str):
        self.database_path = database_path
        db_path = Path(database_path)
        if db_path.parent.as_posix() not in ("", "."):
            db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.database_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Provide a transactional connection with automatic rollback on error."""
        conn = self._connect()
        try:
            conn.execute("BEGIN")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self.transaction() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    is_admin INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )
                """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scan_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    predicted_label TEXT NOT NULL,
                    predicted_index INTEGER NOT NULL,
                    confidence REAL,
                    probabilities_json TEXT,
                    recommendation_severity TEXT,
                    recommendations_json TEXT,
                    source_type TEXT NOT NULL,
                    created_by TEXT,
                    created_at TEXT NOT NULL
                )
                """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_result_id INTEGER NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (scan_result_id)
                        REFERENCES scan_results(id)
                        ON DELETE CASCADE
                )
                """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS activity_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    action TEXT NOT NULL,
                    source_type TEXT,
                    row_count INTEGER DEFAULT 0,
                    details_json TEXT,
                    created_at TEXT NOT NULL
                )
                """)

    def create_user(
        self, username: str, password_hash: str, is_admin: bool
    ) -> dict[str, Any]:
        now = datetime.utcnow().isoformat()
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO users (username, password_hash, is_admin, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (username, password_hash, int(is_admin), now),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("Failed to create user record.")
            return {
                "id": int(cursor.lastrowid),
                "username": username,
                "is_admin": bool(is_admin),
                "created_at": datetime.fromisoformat(now),
            }

    def get_user_by_username(self, username: str) -> dict[str, Any] | None:
        with self.transaction() as conn:
            row = conn.execute(
                (
                    "SELECT id, username, password_hash, is_admin, created_at "
                    "FROM users WHERE username=?"
                ),
                (username,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": int(row["id"]),
            "username": str(row["username"]),
            "password_hash": str(row["password_hash"]),
            "is_admin": bool(row["is_admin"]),
            "created_at": datetime.fromisoformat(row["created_at"]),
        }

    def admin_exists(self) -> bool:
        with self.transaction() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM users WHERE is_admin=1"
            ).fetchone()[0]
        return bool(count)

    def store_predictions(
        self,
        predictions: list[dict[str, Any]],
        source_type: str,
        username: str | None = None,
    ) -> list[int]:
        now = datetime.utcnow().isoformat()
        inserted_ids: list[int] = []
        with self.transaction() as conn:
            for item in predictions:
                cursor = conn.execute(
                    """
                    INSERT INTO scan_results
                      (predicted_label, predicted_index, confidence, probabilities_json,
                       recommendation_severity, recommendations_json, source_type,
                       created_by, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item["predicted_label"],
                        int(item["predicted_index"]),
                        item.get("confidence"),
                        json.dumps(item.get("class_probabilities")),
                        item.get("recommendation_severity"),
                        json.dumps(item.get("recommendations", [])),
                        source_type,
                        username,
                        now,
                    ),
                )
                if cursor.lastrowid is None:
                    raise RuntimeError("Failed to store scan result.")
                scan_result_id = int(cursor.lastrowid)
                inserted_ids.append(scan_result_id)

                severity = str(item.get("recommendation_severity") or "info").lower()
                if severity not in {"info", "normal"}:
                    recommendations = item.get("recommendations", [])
                    message = (
                        " | ".join(recommendations)
                        if recommendations
                        else "Investigate suspicious activity."
                    )
                    conn.execute(
                        """
                        INSERT INTO alerts
                          (scan_result_id, alert_type, severity, message, created_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            scan_result_id,
                            item["predicted_label"],
                            severity,
                            message,
                            now,
                        ),
                    )
        return inserted_ids

    def log_activity(
        self,
        action: str,
        username: str | None = None,
        source_type: str | None = None,
        row_count: int = 0,
        details: dict[str, Any] | None = None,
    ) -> int:
        now = datetime.utcnow().isoformat()
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO activity_logs
                  (username, action, source_type, row_count, details_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    username,
                    action,
                    source_type,
                    int(row_count),
                    json.dumps(details or {}),
                    now,
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("Failed to write activity log.")
            return int(cursor.lastrowid)

    def get_stats(self) -> dict[str, Any]:
        with self.transaction() as conn:
            total_predictions = conn.execute(
                "SELECT COUNT(*) FROM scan_results"
            ).fetchone()[0]
            total_alerts = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
            total_uploads = conn.execute(
                "SELECT COUNT(*) FROM activity_logs WHERE action='upload'"
            ).fetchone()[0]
            avg_confidence = (
                conn.execute("SELECT AVG(confidence) FROM scan_results").fetchone()[0]
                or 0.0
            )
            latest_prediction = conn.execute(
                "SELECT MAX(created_at) FROM scan_results"
            ).fetchone()[0]
            latest_upload = conn.execute(
                "SELECT MAX(created_at) FROM activity_logs WHERE action='upload'"
            ).fetchone()[0]
            pred_rows = conn.execute(
                (
                    "SELECT predicted_label, COUNT(*) AS count "
                    "FROM scan_results GROUP BY predicted_label"
                )
            ).fetchall()
            upload_rows = conn.execute(
                (
                    "SELECT source_type, COUNT(*) AS count FROM activity_logs "
                    "WHERE action='upload' GROUP BY source_type"
                )
            ).fetchall()
            alert_rows = conn.execute(
                "SELECT severity, COUNT(*) AS count FROM alerts GROUP BY severity"
            ).fetchall()

        return {
            "total_predictions": int(total_predictions),
            "total_alerts": int(total_alerts),
            "total_uploads": int(total_uploads),
            "avg_confidence": round(float(avg_confidence), 4),
            "latest_prediction_at": (
                datetime.fromisoformat(latest_prediction) if latest_prediction else None
            ),
            "latest_upload_at": (
                datetime.fromisoformat(latest_upload) if latest_upload else None
            ),
            "predictions_by_label": {
                row["predicted_label"]: row["count"] for row in pred_rows
            },
            "uploads_by_source": {
                row["source_type"]: row["count"] for row in upload_rows
            },
            "alerts_by_severity": {row["severity"]: row["count"] for row in alert_rows},
        }

    def ping(self) -> bool:
        try:
            with self.transaction() as conn:
                conn.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False
