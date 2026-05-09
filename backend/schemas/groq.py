from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class GroqAssistantRequest(BaseModel):
    alert_id: int | None = None
    packet_metadata: dict[str, Any] = Field(default_factory=dict)
    detection_result: dict[str, Any] = Field(default_factory=dict)
    threat_intelligence: list[dict[str, Any]] = Field(default_factory=list)
    shap_explanations: dict[str, Any] = Field(default_factory=dict)
    historical_events: list[dict[str, Any]] = Field(default_factory=list)
    system_metrics: dict[str, Any] = Field(default_factory=dict)
    analyst_notes: str | None = None


class GroqAssistantResponse(BaseModel):
    threat_summary: str
    risk_assessment: str
    remediation_recommendations: list[str] = Field(default_factory=list)
    incident_timeline: list[dict[str, Any]] = Field(default_factory=list)
    false_positive_analysis: str
    correlated_events: list[dict[str, Any]] = Field(default_factory=list)
    shap_explanation: str
    incident_report: str
    raw_response: dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
