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
                CREATE TABLE IF NOT EXISTS detection_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_type TEXT NOT NULL,
                    threat_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    score REAL NOT NULL,
                    confidence REAL NOT NULL,
                    description TEXT NOT NULL,
                    explanation_json TEXT,
                    event_context_json TEXT,
                    acknowledged INTEGER NOT NULL DEFAULT 0,
                    acknowledged_by TEXT,
                    acknowledged_at TEXT,
                    ack_comment TEXT,
                    created_at TEXT NOT NULL
                )
                """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS detection_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    source_file TEXT,
                    yaml_text TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL
                )
                """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS response_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id INTEGER NOT NULL,
                    action_type TEXT NOT NULL,
                    target TEXT NOT NULL,
                    parameters_json TEXT,
                    status TEXT NOT NULL,
                    requested_by TEXT NOT NULL,
                    approved_by TEXT,
                    result_json TEXT,
                    rollback_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    executed_at TEXT
                )
                """)
            existing_columns = [row["name"] for row in conn.execute("PRAGMA table_info(detection_alerts)").fetchall()]
            if "acknowledged" not in existing_columns:
                conn.execute(
                    "ALTER TABLE detection_alerts ADD COLUMN acknowledged INTEGER NOT NULL DEFAULT 0"
                )
            if "acknowledged_by" not in existing_columns:
                conn.execute(
                    "ALTER TABLE detection_alerts ADD COLUMN acknowledged_by TEXT"
                )
            if "acknowledged_at" not in existing_columns:
                conn.execute(
                    "ALTER TABLE detection_alerts ADD COLUMN acknowledged_at TEXT"
                )
            if "ack_comment" not in existing_columns:
                conn.execute(
                    "ALTER TABLE detection_alerts ADD COLUMN ack_comment TEXT"
                )
            conn.execute("""
                CREATE TABLE IF NOT EXISTS action_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action_id INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    details_json TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (action_id)
                        REFERENCES response_actions(id)
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
            total_detection_alerts = conn.execute(
                "SELECT COUNT(*) FROM detection_alerts"
            ).fetchone()[0]
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
            "total_detection_alerts": int(total_detection_alerts),
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

    def store_detection_alert(
        self,
        alert_type: str,
        threat_type: str,
        severity: str,
        score: float,
        confidence: float,
        description: str,
        explanation: dict[str, Any] | None = None,
        event_context: dict[str, Any] | None = None,
    ) -> int:
        now = datetime.utcnow().isoformat()
        explanation_json = json.dumps(explanation or {})
        event_context_json = json.dumps(event_context or {})
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO detection_alerts
                  (alert_type, threat_type, severity, score, confidence, description,
                   explanation_json, event_context_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert_type,
                    threat_type,
                    severity,
                    score,
                    confidence,
                    description,
                    explanation_json,
                    event_context_json,
                    now,
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("Failed to store detection alert.")
            return int(cursor.lastrowid)

    def get_detection_alerts(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.transaction() as conn:
            rows = conn.execute(
                "SELECT id, alert_type, threat_type, severity, score, confidence, description, explanation_json, event_context_json, acknowledged, acknowledged_by, acknowledged_at, ack_comment, created_at FROM detection_alerts ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        alerts: list[dict[str, Any]] = []
        for row in rows:
            alerts.append(
                {
                    "id": int(row["id"]),
                    "alert_type": row["alert_type"],
                    "threat_type": row["threat_type"],
                    "severity": row["severity"],
                    "score": float(row["score"]),
                    "confidence": float(row["confidence"]),
                    "description": row["description"],
                    "explanation": json.loads(row["explanation_json"] or "{}"),
                    "event_context": json.loads(row["event_context_json"] or "{}"),
                    "acknowledged": bool(row["acknowledged"]),
                    "acknowledged_by": row["acknowledged_by"],
                    "acknowledged_at": datetime.fromisoformat(row["acknowledged_at"]) if row["acknowledged_at"] else None,
                    "ack_comment": row["ack_comment"],
                    "created_at": datetime.fromisoformat(row["created_at"]),
                }
            )
        return alerts

    def get_detection_alert(self, alert_id: int) -> dict[str, Any] | None:
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT id, alert_type, threat_type, severity, score, confidence, description, explanation_json, event_context_json, acknowledged, acknowledged_by, acknowledged_at, ack_comment, created_at FROM detection_alerts WHERE id = ?",
                (alert_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": int(row["id"]),
            "alert_type": row["alert_type"],
            "threat_type": row["threat_type"],
            "severity": row["severity"],
            "score": float(row["score"]),
            "confidence": float(row["confidence"]),
            "description": row["description"],
            "explanation": json.loads(row["explanation_json"] or "{}"),
            "event_context": json.loads(row["event_context_json"] or "{}"),
            "acknowledged": bool(row["acknowledged"]),
            "acknowledged_by": row["acknowledged_by"],
            "acknowledged_at": datetime.fromisoformat(row["acknowledged_at"]) if row["acknowledged_at"] else None,
            "ack_comment": row["ack_comment"],
            "created_at": datetime.fromisoformat(row["created_at"]),
        }

    def acknowledge_detection_alert(
        self,
        alert_id: int,
        acknowledged_by: str,
        ack_comment: str | None = None,
    ) -> None:
        now = datetime.utcnow().isoformat()
        with self.transaction() as conn:
            conn.execute(
                "UPDATE detection_alerts SET acknowledged = 1, acknowledged_by = ?, acknowledged_at = ?, ack_comment = ? WHERE id = ?",
                (acknowledged_by, now, ack_comment, alert_id),
            )

    def get_response_action(self, action_id: int) -> dict[str, Any] | None:
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM response_actions WHERE id = ?",
                (action_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": int(row["id"]),
            "alert_id": int(row["alert_id"]),
            "action_type": row["action_type"],
            "target": row["target"],
            "parameters_json": row["parameters_json"],
            "status": row["status"],
            "requested_by": row["requested_by"],
            "approved_by": row["approved_by"],
            "result_json": row["result_json"],
            "rollback_json": row["rollback_json"],
            "created_at": datetime.fromisoformat(row["created_at"]),
            "updated_at": datetime.fromisoformat(row["updated_at"]),
            "executed_at": datetime.fromisoformat(row["executed_at"]) if row["executed_at"] else None,
        }

    def get_pending_response_actions(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.transaction() as conn:
            rows = conn.execute(
                "SELECT * FROM response_actions WHERE status = ? ORDER BY created_at ASC LIMIT ?",
                ("pending", limit),
            ).fetchall()
        return [
            {
                "action_id": int(row["id"]),
                "alert_id": int(row["alert_id"]),
                "action_type": row["action_type"],
                "target": row["target"],
                "parameters": json.loads(row["parameters_json"] or "{}"),
                "status": row["status"],
                "requested_by": row["requested_by"],
                "approved_by": row["approved_by"],
                "result": json.loads(row["result_json"] or "{}"),
                "rollback_available": bool(row["rollback_json"]),
                "created_at": datetime.fromisoformat(row["created_at"]),
                "updated_at": datetime.fromisoformat(row["updated_at"]),
            }
            for row in rows
        ]

    def ping(self) -> bool:
        try:
            with self.transaction() as conn:
                conn.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False
