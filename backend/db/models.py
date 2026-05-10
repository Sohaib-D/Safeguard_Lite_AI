from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float, ForeignKey, Text, Index
from sqlalchemy.orm import relationship
from backend.db.database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

class ScanResult(Base):
    __tablename__ = "scan_results"
    
    id = Column(Integer, primary_key=True, index=True)
    predicted_label = Column(String, nullable=False)
    predicted_index = Column(Integer, nullable=False)
    confidence = Column(Float)
    probabilities_json = Column(Text)  # JSON stored as text
    recommendation_severity = Column(String)
    recommendations_json = Column(Text)  # JSON stored as text
    source_type = Column(String, nullable=False)
    created_by = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    alerts = relationship("Alert", back_populates="scan_result", cascade="all, delete-orphan")

class Alert(Base):
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    scan_result_id = Column(Integer, ForeignKey("scan_results.id", ondelete="CASCADE"), nullable=False)
    alert_type = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    scan_result = relationship("ScanResult", back_populates="alerts")

class DetectionAlert(Base):
    __tablename__ = "detection_alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    alert_type = Column(String, nullable=False)
    threat_type = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    score = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    description = Column(Text, nullable=False)
    explanation_json = Column(Text)
    event_context_json = Column(Text)
    acknowledged = Column(Boolean, default=False, nullable=False)
    acknowledged_by = Column(String)
    acknowledged_at = Column(DateTime)
    ack_comment = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_detection_alerts_severity", "severity"),
        Index("ix_detection_alerts_acknowledged", "acknowledged"),
    )

class DetectionRule(Base):
    __tablename__ = "detection_rules"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    pattern_type = Column(String, nullable=False)
    pattern_value = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    description = Column(Text)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String, nullable=False)
    username = Column(String)
    details_json = Column(Text)
    severity = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    __table_args__ = (
        Index("ix_audit_logs_event_type", "event_type"),
    )

class ResponseAction(Base):
    __tablename__ = "response_actions"
    
    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(Integer, nullable=False)
    action_type = Column(String, nullable=False)
    target = Column(String, nullable=False)
    parameters_json = Column(Text)
    status = Column(String, nullable=False)
    requested_by = Column(String)
    approved_by = Column(String)
    result_json = Column(Text)
    rollback_json = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    executed_at = Column(DateTime)

class ActionAudit(Base):
    __tablename__ = "action_audit"
    
    id = Column(Integer, primary_key=True, index=True)
    action_id = Column(Integer, ForeignKey("response_actions.id", ondelete="CASCADE"), nullable=False)
    event_type = Column(String, nullable=False)
    actor = Column(String)
    details_json = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
