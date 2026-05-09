from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

from backend.db.sqlite_store import SQLiteStore
from backend.schemas.alert import AlertAcknowledgementRequest, DetectionAlert
from backend.services.websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)


class AlertService:
    def __init__(self, database_path: str, websocket_manager: WebSocketManager | None = None):
        self.store = SQLiteStore(database_path)
        self.websocket_manager = websocket_manager

    async def create_alert(
        self,
        alert: DetectionAlert,
    ) -> DetectionAlert:
        saved_id = await asyncio.to_thread(
            self.store.store_detection_alert,
            alert.alert_type,
            alert.threat_type,
            alert.severity,
            alert.score,
            alert.confidence,
            alert.description,
            alert.explanation,
            alert.event_context,
        )
        alert.id = saved_id
        alert.created_at = datetime.utcnow()

        if self.websocket_manager:
            try:
                await self.websocket_manager.broadcast("alert", alert.model_dump())
            except Exception as exc:
                logger.error(f"Failed to broadcast alert: {exc}")

        return alert

    async def list_alerts(self, limit: int = 50) -> list[DetectionAlert]:
        rows = await asyncio.to_thread(self.store.get_detection_alerts, limit)
        return [DetectionAlert(**row) for row in rows]

    async def acknowledge_alert(
        self,
        alert_id: int,
        acknowledgement: AlertAcknowledgementRequest,
    ) -> DetectionAlert:
        await asyncio.to_thread(
            self.store.acknowledge_detection_alert,
            alert_id,
            acknowledgement.acknowledged_by,
            acknowledgement.comment,
        )
        row = await asyncio.to_thread(self.store.get_detection_alert, alert_id)
        alert = DetectionAlert(**row)
        if self.websocket_manager:
            try:
                await self.websocket_manager.broadcast("alert_ack", alert.model_dump())
            except Exception as exc:
                logger.error(f"Failed to broadcast acknowledgement: {exc}")
        return alert
