from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    detail: str
    error_code: str = "request_error"
    errors: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    model_loaded: bool
    database_ok: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class UploadResponse(BaseModel):
    upload_id: int
    rows_logged: int
    source_type: str
    timestamp: datetime


class StatsResponse(BaseModel):
    total_predictions: int
    total_uploads: int
    avg_confidence: float
    latest_prediction_at: datetime | None
    latest_upload_at: datetime | None
    predictions_by_label: dict[str, int]
    uploads_by_source: dict[str, int]


class ModelInfoResponse(BaseModel):
    model_name: str
    label_classes: list[str]
    feature_count: int
    feature_names: list[str]
    raw_input_schema: dict[str, Any]
    training_config: dict[str, Any]
    artifacts: dict[str, bool]
