from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ResponseActionStatus(str, Enum):
    pending = "pending"
    approved = "approved"
    executed = "executed"
    failed = "failed"
    rolled_back = "rolled_back"
    canceled = "canceled"


class ResponseActionType(str, Enum):
    block_ip = "block_ip"
    kill_process = "kill_process"
    disable_outbound = "disable_outbound"
    quarantine_file = "quarantine_file"
    windows_defender_exclusion = "windows_defender_exclusion"
    disable_adapter = "disable_adapter"
    forensic_snapshot = "forensic_snapshot"
    incident_report = "incident_report"


class ResponseActionRequest(BaseModel):
    alert_id: int
    action_type: ResponseActionType
    target: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    requested_by: str
    reason: str


class ResponseActionApproval(BaseModel):
    action_id: int
    approved: bool
    approved_by: str
    justification: str | None = None


class ResponseActionItem(BaseModel):
    action_id: int | None = None
    alert_id: int
    action_type: ResponseActionType
    target: str
    parameters: dict[str, Any]
    status: ResponseActionStatus
    requested_by: str
    approved_by: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)
    rollback_available: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class RollbackRequest(BaseModel):
    action_id: int
    requested_by: str
    reason: str | None = None


class IncidentReportResponse(BaseModel):
    report_id: int
    action_id: int
    report_path: str
    summary: dict[str, Any]
    created_at: datetime


class ForensicSnapshotResponse(BaseModel):
    snapshot_id: int
    action_id: int
    snapshot_path: str
    created_at: datetime
