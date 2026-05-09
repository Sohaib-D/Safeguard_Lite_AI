from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AlertSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DetectionAlert(BaseModel):
    id: int | None = None
    event_id: str | None = None
    alert_type: str
    threat_type: str
    severity: AlertSeverity
    score: float = 0.0
    confidence: float = 0.0
    description: str
    explanation: dict[str, Any] = Field(default_factory=dict)
    event_context: dict[str, Any] = Field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = None
    ack_comment: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AlertAcknowledgementRequest(BaseModel):
    acknowledged_by: str
    comment: str | None = None


class RuleDefinition(BaseModel):
    name: str
    description: str
    enabled: bool = True
    severity: AlertSeverity = AlertSeverity.LOW
    score: float = 0.5
    conditions: list[dict[str, Any]] = Field(default_factory=list)
