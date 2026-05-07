"""
database.py
───────────
SQLite-backed interaction logger.
Swap DATABASE_URL for PostgreSQL in production.
"""

import sqlite3
import json
import os
from typing import Optional, List, Dict

DATABASE_PATH = os.environ.get("DATABASE_PATH", "chatbot.db")


class Database:
    def __init__(self):
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id    TEXT NOT NULL,
                user_message  TEXT NOT NULL,
                intent        TEXT,
                confidence    REAL,
                response      TEXT,
                escalated     INTEGER DEFAULT 0,
                domain        TEXT DEFAULT 'ecommerce',
                latency_ms    REAL,
                timestamp     TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS escalations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                ticket_id   TEXT NOT NULL,
                created_at  TEXT
            )
        """)
        conn.commit()
        conn.close()

    def log_interaction(self, entry: Dict):
        conn = self._connect()
        conn.execute("""
            INSERT INTO interactions
              (session_id, user_message, intent, confidence, response, escalated, domain, latency_ms, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry["session_id"], entry["user_message"], entry["intent"],
            entry["confidence"], entry["response"], int(entry["escalated"]),
            entry.get("domain", "ecommerce"), entry["latency_ms"], entry["timestamp"]
        ))
        conn.commit()
        conn.close()

    def mark_escalated(self, session_id: str, ticket_id: str):
        from datetime import datetime
        conn = self._connect()
        conn.execute(
            "INSERT INTO escalations (session_id, ticket_id, created_at) VALUES (?, ?, ?)",
            (session_id, ticket_id, datetime.utcnow().isoformat())
        )
        conn.execute(
            "UPDATE interactions SET escalated=1 WHERE session_id=? AND escalated=0",
            (session_id,)
        )
        conn.commit()
        conn.close()

    def get_logs(self, limit: int = 50, domain: Optional[str] = None) -> List[Dict]:
        conn = self._connect()
        if domain:
            rows = conn.execute(
                "SELECT * FROM interactions WHERE domain=? ORDER BY id DESC LIMIT ?",
                (domain, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM interactions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_stats(self) -> Dict:
        conn = self._connect()
        total = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
        escalated = conn.execute("SELECT COUNT(*) FROM interactions WHERE escalated=1").fetchone()[0]
        avg_latency = conn.execute("SELECT AVG(latency_ms) FROM interactions").fetchone()[0] or 0
        avg_confidence = conn.execute("SELECT AVG(confidence) FROM interactions").fetchone()[0] or 0
        domains = conn.execute(
            "SELECT domain, COUNT(*) as cnt FROM interactions GROUP BY domain"
        ).fetchall()
        conn.close()
        return {
            "total_interactions": total,
            "escalated": escalated,
            "avg_latency_ms": round(avg_latency, 2),
            "avg_confidence": round(avg_confidence, 4),
            "by_domain": {row["domain"]: row["cnt"] for row in domains}
        }
