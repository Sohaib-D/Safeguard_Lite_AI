from __future__ import annotations
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from backend.db.postgres_store import PostgresStore
from backend.db.session import session_scope
from backend.schemas.alert import AlertAcknowledgementRequest, DetectionAlert
from backend.services.websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)

class AlertService:
    def __init__(self, db: Session | None = None, websocket_manager: WebSocketManager | None = None):
        self._db = db
        self.websocket_manager = websocket_manager

    async def create_alert(self, alert: DetectionAlert) -> DetectionAlert:
        if self._db:
            saved_id = PostgresStore(self._db).store_detection_alert(
                alert.alert_type,
                alert.threat_type,
                alert.severity,
                alert.score,
                alert.confidence,
                alert.description,
                alert.explanation,
                alert.event_context,
            )
        else:
            with session_scope() as db:
                saved_id = PostgresStore(db).store_detection_alert(
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
        if self._db:
            rows = PostgresStore(self._db).get_detection_alerts(limit)
        else:
            with session_scope() as db:
                rows = PostgresStore(db).get_detection_alerts(limit)
        return [DetectionAlert(**row) for row in rows]

    async def acknowledge_alert(
        self, alert_id: int, acknowledgement: AlertAcknowledgementRequest
    ) -> DetectionAlert:
        if self._db:
            store = PostgresStore(self._db)
            store.acknowledge_detection_alert(
                alert_id, acknowledgement.acknowledged_by, acknowledgement.comment
            )
            row = store.get_detection_alert(alert_id)
        else:
            with session_scope() as db:
                store = PostgresStore(db)
                store.acknowledge_detection_alert(
                    alert_id, acknowledgement.acknowledged_by, acknowledgement.comment
                )
                row = store.get_detection_alert(alert_id)
                
        if not row:
            raise ValueError(f"Alert {alert_id} not found")
            
        alert = DetectionAlert(**row)
        if self.websocket_manager:
            try:
                await self.websocket_manager.broadcast("alert_ack", alert.model_dump())
            except Exception as exc:
                logger.error(f"Failed to broadcast acknowledgement: {exc}")
        return alert
