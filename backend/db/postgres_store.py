import json
from datetime import datetime
from typing import Any, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func
from backend.db.models import User, ScanResult, Alert, DetectionAlert, DetectionRule, AuditLog

class PostgresStore:
    """SQLAlchemy-based store for PostgreSQL (Supabase)."""

    def __init__(self, session: Session):
        self.session = session

    def create_user(self, username: str, password_hash: str, is_admin: bool = False) -> dict[str, Any]:
        user = User(
            username=username,
            password_hash=password_hash,
            is_admin=is_admin,
            created_at=datetime.utcnow()
        )
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)
        return {
            "id": user.id,
            "username": user.username,
            "is_admin": user.is_admin,
            "created_at": user.created_at
        }

    def get_user_by_username(self, username: str) -> dict[str, Any] | None:
        user = self.session.query(User).filter(User.username == username).first()
        if not user:
            return None
        return {
            "id": user.id,
            "username": user.username,
            "password_hash": user.password_hash,
            "is_admin": user.is_admin,
            "created_at": user.created_at
        }

    def admin_exists(self) -> bool:
        return self.session.query(User).filter(User.is_admin == True).first() is not None

    def store_predictions(
        self,
        predicted_label: str,
        predicted_index: int,
        confidence: float,
        probabilities: dict[str, float],
        recommendation_severity: str,
        recommendations: list[str],
        source_type: str,
        created_by: str | None = None,
    ) -> int:
        scan = ScanResult(
            predicted_label=predicted_label,
            predicted_index=predicted_index,
            confidence=confidence,
            probabilities_json=json.dumps(probabilities),
            recommendation_severity=recommendation_severity,
            recommendations_json=json.dumps(recommendations),
            source_type=source_type,
            created_by=created_by,
            created_at=datetime.utcnow()
        )
        self.session.add(scan)
        self.session.commit()
        return scan.id

    def log_activity(
        self, event_type: str, username: str | None, details: dict[str, Any], severity: str = "INFO"
    ) -> None:
        log = AuditLog(
            event_type=event_type,
            username=username,
            details_json=json.dumps(details),
            severity=severity,
            created_at=datetime.utcnow()
        )
        self.session.add(log)
        self.session.commit()

    def get_stats(self) -> dict[str, Any]:
        total_scans = self.session.query(func.count(ScanResult.id)).scalar()
        total_alerts = self.session.query(func.count(DetectionAlert.id)).scalar()
        recent_threats = self.session.query(DetectionAlert).order_by(DetectionAlert.created_at.desc()).limit(5).all()
        
        return {
            "total_scans": total_scans,
            "total_alerts": total_alerts,
            "recent_threats": [
                {"type": a.threat_type, "severity": a.severity, "time": a.created_at.isoformat()}
                for a in recent_threats
            ]
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
        alert = DetectionAlert(
            alert_type=alert_type,
            threat_type=threat_type,
            severity=severity,
            score=score,
            confidence=confidence,
            description=description,
            explanation_json=json.dumps(explanation) if explanation else None,
            event_context_json=json.dumps(event_context) if event_context else None,
            created_at=datetime.utcnow()
        )
        self.session.add(alert)
        self.session.commit()
        return alert.id

    def get_detection_alerts(self, limit: int = 50) -> list[dict[str, Any]]:
        alerts = self.session.query(DetectionAlert).order_by(DetectionAlert.created_at.desc()).limit(limit).all()
        return [
            {
                "id": a.id,
                "alert_type": a.alert_type,
                "threat_type": a.threat_type,
                "severity": a.severity,
                "score": a.score,
                "confidence": a.confidence,
                "description": a.description,
                "explanation": json.loads(a.explanation_json) if a.explanation_json else {},
                "event_context": json.loads(a.event_context_json) if a.event_context_json else {},
                "acknowledged": a.acknowledged,
                "acknowledged_by": a.acknowledged_by,
                "acknowledged_at": a.acknowledged_at,
                "ack_comment": a.ack_comment,
                "created_at": a.created_at
            }
            for a in alerts
        ]

    def get_detection_alert(self, alert_id: int) -> dict[str, Any] | None:
        a = self.session.query(DetectionAlert).filter(DetectionAlert.id == alert_id).first()
        if not a:
            return None
        return {
            "id": a.id,
            "alert_type": a.alert_type,
            "threat_type": a.threat_type,
            "severity": a.severity,
            "score": a.score,
            "confidence": a.confidence,
            "description": a.description,
            "explanation": json.loads(a.explanation_json) if a.explanation_json else {},
            "event_context": json.loads(a.event_context_json) if a.event_context_json else {},
            "acknowledged": a.acknowledged,
            "acknowledged_by": a.acknowledged_by,
            "acknowledged_at": a.acknowledged_at,
            "ack_comment": a.ack_comment,
            "created_at": a.created_at
        }

    def acknowledge_detection_alert(self, alert_id: int, username: str, comment: str) -> None:
        alert = self.session.query(DetectionAlert).filter(DetectionAlert.id == alert_id).first()
        if alert:
            alert.acknowledged = True
            alert.acknowledged_by = username
            alert.acknowledged_at = datetime.utcnow()
            alert.ack_comment = comment
            self.session.commit()

    def get_response_action(self, action_id: int) -> dict[str, Any] | None:
        row = self.session.query(ResponseAction).filter(ResponseAction.id == action_id).first()
        if not row:
            return None
        return {
            "id": row.id,
            "alert_id": row.alert_id,
            "action_type": row.action_type,
            "target": row.target,
            "parameters_json": row.parameters_json,
            "status": row.status,
            "requested_by": row.requested_by,
            "approved_by": row.approved_by,
            "result_json": row.result_json,
            "rollback_json": row.rollback_json,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
            "executed_at": row.executed_at,
        }

    def get_pending_response_actions(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.session.query(ResponseAction).filter(ResponseAction.status == "pending").order_by(ResponseAction.created_at.asc()).limit(limit).all()
        return [
            {
                "action_id": row.id,
                "alert_id": row.alert_id,
                "action_type": row.action_type,
                "target": row.target,
                "parameters": json.loads(row.parameters_json or "{}"),
                "status": row.status,
                "requested_by": row.requested_by,
                "approved_by": row.approved_by,
                "result": json.loads(row.result_json or "{}"),
                "rollback_available": bool(row.rollback_json),
                "created_at": row.created_at,
                "updated_at": row.updated_at,
            }
            for row in rows
        ]

    def ping(self) -> bool:
        try:
            self.session.execute("SELECT 1")
            return True
        except:
            return False
